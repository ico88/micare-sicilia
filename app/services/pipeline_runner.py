"""Background pipeline runner for the guided AMR prediction wizard.

The pipeline runs in a daemon thread; its state is persisted in the DB so
reloading the page shows the current status even after a browser disconnect.
"""
from __future__ import annotations

import json
import logging
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask

from app import db
from app.models import AggregatedObservation, CombinationForecast, PipelineRun, ValidationMetric
from app.services.prophet_training_adapter import db_to_model_format, prophet_output_dir
from micare_sicilia.model import train_forecasts, load_forecasts

logger = logging.getLogger(__name__)

_active_thread: threading.Thread | None = None
_lock = threading.Lock()


class PipelineStopped(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_pipeline(app: Flask, model_folder: str | Path) -> tuple[int, bool]:
    """Start the pipeline in a background thread.

    Returns (run_id, started) where started=False means one is already running.
    """
    global _active_thread
    with app.app_context():
        # If there is already a running run in DB, refuse
        active = PipelineRun.query.filter(PipelineRun.status.in_(["queued", "running"])).first()
        if active is not None:
            return active.id, False

        # Mark any previous interrupted run
        _mark_old_interrupted()

        run = PipelineRun(
            status="queued",
            current_step=0,
            step2_status="pending",
            step3_status="pending",
            step4_status="pending",
            progress=0,
            message="Pipeline in coda.",
        )
        db.session.add(run)
        db.session.commit()
        run_id = run.id

    with _lock:
        if _active_thread and _active_thread.is_alive():
            return run_id, False
        t = threading.Thread(
            target=_run_pipeline,
            args=(app, run_id, str(model_folder)),
            daemon=True,
        )
        _active_thread = t
        t.start()

    return run_id, True


def get_pipeline_status(app: Flask) -> dict[str, Any]:
    """Return current pipeline status from DB."""
    with app.app_context():
        run = PipelineRun.query.order_by(PipelineRun.id.desc()).first()
        if run is None:
            return {"status": "never_run", "current_step": 0}
        return _run_to_dict(run)


def stop_pipeline(run_id: int, app: Flask) -> None:
    """Request stop of a running pipeline by setting DB flag."""
    with app.app_context():
        run = PipelineRun.query.get(run_id)
        if run and run.status in ("queued", "running"):
            run.stop_requested = True
            run.message = "Interruzione richiesta — la pipeline si fermerà al prossimo checkpoint."
            run.updated_at = datetime.utcnow()
            db.session.commit()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mark_old_interrupted() -> None:
    stale = PipelineRun.query.filter(PipelineRun.status.in_(["queued", "running"])).all()
    for r in stale:
        r.status = "interrupted"
        r.message = "Interrotto al riavvio del server."
        r.updated_at = datetime.utcnow()
    if stale:
        db.session.commit()


def _run_to_dict(run: PipelineRun) -> dict[str, Any]:
    summary: dict = {}
    try:
        summary = json.loads(run.summary_json or "{}")
    except Exception:
        pass
    return {
        "id": run.id,
        "status": run.status,
        "current_step": run.current_step,
        "step2_status": run.step2_status,
        "step3_status": run.step3_status,
        "step4_status": run.step4_status,
        "progress": run.progress,
        "message": run.message,
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "summary": summary,
        "stop_requested": run.stop_requested,
    }


def _update_run(app: Flask, run_id: int, **kwargs: Any) -> None:
    """Update PipelineRun in DB."""
    with app.app_context():
        run = PipelineRun.query.get(run_id)
        if run is None:
            return
        for k, v in kwargs.items():
            setattr(run, k, v)
        run.updated_at = datetime.utcnow()
        db.session.commit()


def _check_stop(app: Flask, run_id: int) -> None:
    """Raise PipelineStopped if stop_requested is True in DB."""
    with app.app_context():
        run = PipelineRun.query.get(run_id)
        if run and run.stop_requested:
            raise PipelineStopped("Pipeline interrotta dall'utente.")


# ---------------------------------------------------------------------------
# Aggregated data loader
# ---------------------------------------------------------------------------

def _load_aggregated(app: Flask) -> pd.DataFrame:
    with app.app_context():
        rows = AggregatedObservation.query.all()
        if not rows:
            return pd.DataFrame()
        data = [
            {
                "month": r.month,
                "pathogen": r.pathogen,
                "antibiotic": r.antibiotic,
                "laboratory": r.laboratory,
                "ward": r.ward,
                "sensitive_pct": r.sensitive_pct,
                "intermediate_pct": r.intermediate_pct,
                "resistant_pct": r.resistant_pct,
                "pct_icu": r.pct_icu,
                "pct_inpatient": r.pct_inpatient,
            }
            for r in rows
        ]
        return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Validation metrics helper
# ---------------------------------------------------------------------------

def _save_step_metrics(app: Flask, run_id: int, model_name: str, summary: Any) -> None:
    """Save aggregate RMSE metrics from a TrainingSummary into ValidationMetric."""
    with app.app_context():
        try:
            import joblib
            # cv_metrics_results contains per-target per-combination metrics
            metrics_path = Path(summary.output_dir) / "cv_metrics_results.pkl"
            if not metrics_path.exists():
                return
            cv = joblib.load(metrics_path)
            for target in ("resistenti", "intermedi", "sensibili"):
                rmse_vals = list(cv.get(target, {}).get("rmse", {}).values())
                mape_vals = list(cv.get(target, {}).get("mape", {}).values())
                mase_vals = list(cv.get(target, {}).get("mase", {}).values())
                rmse_arima_vals = list(cv.get(target, {}).get("rmse_arima", {}).values())
                if not rmse_vals:
                    continue
                import numpy as np
                vm = ValidationMetric(
                    model_name=model_name,
                    target=target,
                    rmse=float(np.mean(rmse_vals)),
                    mape=float(np.mean(mape_vals)) if mape_vals else None,
                    mase=float(np.mean(mase_vals)) if mase_vals else None,
                    rmse_arima=float(np.mean(rmse_arima_vals)) if rmse_arima_vals else None,
                    metadata_json=json.dumps({
                        "pipeline_run_id": run_id,
                        "n_combinations": len(rmse_vals),
                    }),
                )
                db.session.add(vm)
            db.session.commit()
        except Exception as exc:
            logger.warning("Impossibile salvare metriche step %s: %s", model_name, exc)
            db.session.rollback()


# ---------------------------------------------------------------------------
# Main pipeline thread
# ---------------------------------------------------------------------------

def _run_pipeline(app: Flask, run_id: int, model_folder: str) -> None:
    logger.info("Pipeline run %d avviata", run_id)
    try:
        _update_run(app, run_id,
                    status="running",
                    current_step=1,
                    progress=2,
                    message="Caricamento dati aggregati dal database.")

        aggregated = _load_aggregated(app)
        if aggregated.empty:
            raise RuntimeError("Nessun dato aggregato nel database. Caricare i file CSV prima di avviare la pipeline.")

        model_df = db_to_model_format(aggregated)
        years = sorted(model_df["data"].dt.year.unique().tolist())
        n_years = len(years)

        _update_run(app, run_id,
                    current_step=1,
                    progress=5,
                    message=f"Dati caricati: {len(aggregated)} righe, {n_years} anni ({years[0]}–{years[-1]}).")

        if n_years < 4:
            raise RuntimeError(
                f"Servono almeno 4 anni di dati per la validazione. Trovati: {n_years} anni."
            )

        _check_stop(app, run_id)

        # ------------------------------------------------------------------
        # STEP 2: train on first 3 years, validate on remaining
        # ------------------------------------------------------------------
        _update_run(app, run_id,
                    current_step=2,
                    step2_status="running",
                    progress=8,
                    message=f"Step 2: addestro su anni {years[:3]} → valido su anni {years[3:]}.")

        step2_dir = Path(model_folder) / "prophet_step2"
        step2_dir.mkdir(parents=True, exist_ok=True)
        training_years_2 = years[:3]
        df2 = model_df[model_df["data"].dt.year.isin(training_years_2)].copy()

        try:
            summary2 = train_forecasts(df2, step2_dir)
            _save_step_metrics(app, run_id, "prophet_step2", summary2)
            _update_run(app, run_id,
                        step2_status="done",
                        progress=35,
                        message=(
                            f"Step 2 completato: {summary2.trained_resistant_models} combinazioni addestrate "
                            f"su {training_years_2}."
                        ))
        except PipelineStopped:
            raise
        except Exception as exc:
            _update_run(app, run_id, step2_status="error",
                        message=f"Step 2 errore: {exc}",
                        error=traceback.format_exc())
            raise

        _check_stop(app, run_id)

        # ------------------------------------------------------------------
        # STEP 3: train on all-but-last year, validate on last
        # ------------------------------------------------------------------
        training_years_3 = years[:-1]
        _update_run(app, run_id,
                    current_step=3,
                    step3_status="running",
                    progress=38,
                    message=f"Step 3: addestro su anni {training_years_3} → valido su {years[-1]}.")

        step3_dir = Path(model_folder) / "prophet_step3"
        step3_dir.mkdir(parents=True, exist_ok=True)
        df3 = model_df[model_df["data"].dt.year.isin(training_years_3)].copy()

        try:
            summary3 = train_forecasts(df3, step3_dir)
            _save_step_metrics(app, run_id, "prophet_step3", summary3)
            _update_run(app, run_id,
                        step3_status="done",
                        progress=65,
                        message=(
                            f"Step 3 completato: {summary3.trained_resistant_models} combinazioni addestrate "
                            f"su {training_years_3}."
                        ))
        except PipelineStopped:
            raise
        except Exception as exc:
            _update_run(app, run_id, step3_status="error",
                        message=f"Step 3 errore: {exc}",
                        error=traceback.format_exc())
            raise

        _check_stop(app, run_id)

        # ------------------------------------------------------------------
        # STEP 4: train on ALL data → pre-compute 24-month forecasts
        # ------------------------------------------------------------------
        _update_run(app, run_id,
                    current_step=4,
                    step4_status="running",
                    progress=68,
                    message="Step 4: addestro finale su tutti gli anni in corso...")

        final_dir = Path(model_folder) / "prophet"
        final_dir.mkdir(parents=True, exist_ok=True)

        try:
            summary4 = train_forecasts(model_df, final_dir)
        except PipelineStopped:
            raise
        except Exception as exc:
            _update_run(app, run_id, step4_status="error",
                        message=f"Step 4 (addestramento) errore: {exc}",
                        error=traceback.format_exc())
            raise

        _update_run(app, run_id,
                    progress=80,
                    message=f"Step 4: addestramento completato ({summary4.trained_resistant_models} combinazioni). Pre-calcolo previsioni in corso...")

        _check_stop(app, run_id)

        # Pre-compute combination forecasts for all trained months
        from micare_sicilia.config import FORECAST_PERIODS_MONTHS
        forecasts = load_forecasts(final_dir)
        combinations = sorted(forecasts["resistenti"].keys())
        today = datetime.utcnow().date().replace(day=1)
        n_months = FORECAST_PERIODS_MONTHS
        future_months = [
            (today.replace(month=((today.month - 1 + i) % 12) + 1,
                           year=today.year + ((today.month - 1 + i) // 12)))
            for i in range(1, n_months + 1)
        ]

        # Build safe reverse-map combo_key → (pathogen, laboratory, antibiotic)
        # Avoids parsing the key string which breaks on names containing underscores
        combo_meta: dict[str, tuple[str, str, str]] = {}
        for _, row in (
            model_df.drop_duplicates(subset=["pathogen", "laboratory", "antibiotic"])
            [["pathogen", "laboratory", "antibiotic", "combinazione_unica"]]
            .iterrows()
        ):
            combo_meta[row["combinazione_unica"]] = (
                str(row["pathogen"]), str(row["laboratory"]), str(row["antibiotic"])
            )

        n_combinations = len(combinations)
        _update_run(app, run_id,
                    progress=82,
                    message=f"Pre-calcolo {n_combinations} combinazioni × {n_months} mesi...")

        # Delete ALL old forecasts — clean slate before inserting new ones
        with app.app_context():
            CombinationForecast.query.delete()
            db.session.commit()

        batch: list[CombinationForecast] = []
        batch_size = 100
        processed = 0

        for combo in combinations:
            _check_stop(app, run_id)

            # Use reverse-map for safe field extraction (handles underscores in names)
            if combo not in combo_meta:
                logger.warning("Combo '%s' non trovata nella mappa, salto.", combo)
                continue
            pathogen, laboratory, antibiotic = combo_meta[combo]

            for month_date in future_months:
                year = month_date.year
                month = month_date.month

                # Get forecast for each target
                try:
                    import pandas as _pd
                    import numpy as _np

                    def _get_yhat(target_name: str) -> tuple[float | None, float | None, float | None]:
                        fc = forecasts[target_name]
                        if combo not in fc:
                            return None, None, None
                        df_fc = fc[combo]
                        # build end-of-month date
                        period = _pd.Period(f"{year}-{month:02d}", freq="M")
                        eom = _pd.to_datetime(period.end_time.date())
                        row = df_fc[df_fc["ds"] == eom]
                        if row.empty:
                            return None, None, None
                        r = row.iloc[0]
                        return (
                            float(_np.clip(r["yhat"], 0, 100)),
                            float(_np.clip(r["yhat_lower"], 0, 100)),
                            float(_np.clip(r["yhat_upper"], 0, 100)),
                        )

                    s_yhat, s_lo, s_hi = _get_yhat("sensibili")
                    i_yhat, i_lo, i_hi = _get_yhat("intermedi")
                    r_yhat, r_lo, r_hi = _get_yhat("resistenti")

                    if s_yhat is None and i_yhat is None and r_yhat is None:
                        continue

                    # Normalize to 100%
                    total = (s_yhat or 0) + (i_yhat or 0) + (r_yhat or 0)
                    if total > 0:
                        scale = 100.0 / total
                    else:
                        scale = 1.0

                    cf = CombinationForecast(
                        pathogen=pathogen,
                        antibiotic=antibiotic,
                        laboratory=laboratory,
                        forecast_month=month_date,
                        sensitive_pct=(s_yhat * scale) if s_yhat is not None else None,
                        intermediate_pct=(i_yhat * scale) if i_yhat is not None else None,
                        resistant_pct=(r_yhat * scale) if r_yhat is not None else None,
                        sensitive_lower=(s_lo * scale) if s_lo is not None else None,
                        sensitive_upper=(s_hi * scale) if s_hi is not None else None,
                        intermediate_lower=(i_lo * scale) if i_lo is not None else None,
                        intermediate_upper=(i_hi * scale) if i_hi is not None else None,
                        resistant_lower=(r_lo * scale) if r_lo is not None else None,
                        resistant_upper=(r_hi * scale) if r_hi is not None else None,
                        pipeline_run_id=run_id,
                    )
                    batch.append(cf)

                    if len(batch) >= batch_size:
                        with app.app_context():
                            db.session.add_all(batch)
                            db.session.commit()
                        batch = []

                except Exception as exc:
                    logger.warning("Errore previsione %s %d-%02d: %s", combo, year, month, exc)
                    continue

            processed += 1
            if processed % 10 == 0:
                pct = 82 + int((processed / n_combinations) * 15)
                _update_run(app, run_id,
                            progress=min(97, pct),
                            message=f"Pre-calcolo previsioni: {processed}/{n_combinations} combinazioni.")

        # flush remaining batch
        if batch:
            with app.app_context():
                db.session.add_all(batch)
                db.session.commit()

        summary_data = {
            "years": years,
            "n_combinations_step2": summary2.trained_resistant_models,
            "n_combinations_step3": summary3.trained_resistant_models,
            "n_combinations_final": summary4.trained_resistant_models,
            "n_forecasts": processed * n_months,
        }

        _update_run(app, run_id,
                    status="success",
                    current_step=4,
                    step4_status="done",
                    progress=100,
                    message=(
                        f"Pipeline completata! {summary4.trained_resistant_models} combinazioni, "
                        f"{processed * n_months} previsioni pre-calcolate per i prossimi {n_months} mesi."
                    ),
                    summary_json=json.dumps(summary_data),
                    finished_at=datetime.utcnow())

    except PipelineStopped as exc:
        _update_run(app, run_id,
                    status="stopped",
                    message=str(exc),
                    finished_at=datetime.utcnow())

    except Exception as exc:
        _update_run(app, run_id,
                    status="error",
                    message=f"Errore pipeline: {exc}",
                    error=traceback.format_exc(),
                    finished_at=datetime.utcnow())
        logger.error("Pipeline run %d fallita: %s", run_id, exc, exc_info=True)

    logger.info("Pipeline run %d terminata", run_id)
