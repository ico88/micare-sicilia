"""Pipeline wizard blueprint."""
from __future__ import annotations

from datetime import datetime

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for

from app import db
from app.models import AggregatedObservation, CombinationForecast, PipelineRun
from app.services.pipeline_runner import get_pipeline_status, start_pipeline, stop_pipeline

bp = Blueprint("pipeline", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_to_dict(run: PipelineRun) -> dict:
    import json
    from datetime import timezone
    summary: dict = {}
    try:
        summary = json.loads(run.summary_json or "{}")
    except Exception:
        pass

    # Compute elapsed and ETA
    elapsed_seconds = None
    eta_seconds = None
    if run.created_at:
        end = run.finished_at or datetime.utcnow()
        elapsed_seconds = (end - run.created_at).total_seconds()
        pct = run.progress or 0
        if run.status in ("running", "queued") and pct > 2:
            total_est = elapsed_seconds * 100 / pct
            eta_seconds = max(0.0, total_est - elapsed_seconds)

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
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
    }


def _data_summary() -> dict:
    """Quick summary of loaded data for Step 1 display."""
    from sqlalchemy import func
    row_count = db.session.query(func.count(AggregatedObservation.id)).scalar() or 0
    year_min = db.session.query(func.min(AggregatedObservation.month)).scalar()
    year_max = db.session.query(func.max(AggregatedObservation.month)).scalar()
    return {
        "row_count": row_count,
        "year_min": year_min.year if year_min else None,
        "year_max": year_max.year if year_max else None,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/pipeline")
def wizard():
    run = PipelineRun.query.order_by(PipelineRun.id.desc()).first()
    run_dict = _run_to_dict(run) if run else None
    data_summary = _data_summary()
    return render_template("pipeline.html", run=run_dict, data=data_summary)


@bp.route("/pipeline/start", methods=["POST"])
def start():
    model_folder = current_app.config["MODEL_FOLDER"]
    run_id, started = start_pipeline(current_app._get_current_object(), model_folder)
    return redirect(url_for("pipeline.wizard"))


@bp.route("/pipeline/stop", methods=["POST"])
def stop():
    run_id = request.form.get("run_id", type=int)
    if run_id:
        stop_pipeline(run_id, current_app._get_current_object())
    return redirect(url_for("pipeline.wizard"))


@bp.route("/pipeline/status.json")
def status_json():
    run = PipelineRun.query.order_by(PipelineRun.id.desc()).first()
    if run is None:
        return jsonify({"status": "never_run", "current_step": 0})
    return jsonify(_run_to_dict(run))


# ---------------------------------------------------------------------------
# Explore
# ---------------------------------------------------------------------------

@bp.route("/explore")
def explore():
    pathogens = db.session.query(CombinationForecast.pathogen).distinct().order_by(CombinationForecast.pathogen).all()
    antibiotics = db.session.query(CombinationForecast.antibiotic).distinct().order_by(CombinationForecast.antibiotic).all()
    laboratories = db.session.query(CombinationForecast.laboratory).distinct().order_by(CombinationForecast.laboratory).all()

    f_pathogen = request.args.get("pathogen", "")
    f_antibiotic = request.args.get("antibiotic", "")
    f_laboratory = request.args.get("laboratory", "")
    f_month = request.args.get("month", "")

    q = CombinationForecast.query
    if f_pathogen:
        q = q.filter(CombinationForecast.pathogen == f_pathogen)
    if f_antibiotic:
        q = q.filter(CombinationForecast.antibiotic == f_antibiotic)
    if f_laboratory:
        q = q.filter(CombinationForecast.laboratory == f_laboratory)
    if f_month:
        try:
            from datetime import date
            y, m = f_month.split("-")
            q = q.filter(CombinationForecast.forecast_month == date(int(y), int(m), 1))
        except Exception:
            pass

    forecasts = q.order_by(
        CombinationForecast.forecast_month,
        CombinationForecast.pathogen,
        CombinationForecast.antibiotic,
    ).limit(500).all()

    # Time series data for selected combo (first matching pathogen+antibiotic+laboratory)
    timeseries = []
    if f_pathogen and f_antibiotic and f_laboratory:
        ts_rows = CombinationForecast.query.filter_by(
            pathogen=f_pathogen,
            antibiotic=f_antibiotic,
            laboratory=f_laboratory,
        ).order_by(CombinationForecast.forecast_month).all()
        timeseries = [
            {
                "month": str(r.forecast_month),
                "s": r.sensitive_pct,
                "i": r.intermediate_pct,
                "r": r.resistant_pct,
                "s_lo": r.sensitive_lower,
                "s_hi": r.sensitive_upper,
                "i_lo": r.intermediate_lower,
                "i_hi": r.intermediate_upper,
                "r_lo": r.resistant_lower,
                "r_hi": r.resistant_upper,
            }
            for r in ts_rows
        ]

    return render_template(
        "explore.html",
        pathogens=[p[0] for p in pathogens],
        antibiotics=[a[0] for a in antibiotics],
        laboratories=[l[0] for l in laboratories],
        forecasts=forecasts,
        timeseries=timeseries,
        f_pathogen=f_pathogen,
        f_antibiotic=f_antibiotic,
        f_laboratory=f_laboratory,
        f_month=f_month,
    )


# ---------------------------------------------------------------------------
# Clinical
# ---------------------------------------------------------------------------

def _forecast_months() -> list[str]:
    rows = db.session.query(CombinationForecast.forecast_month).distinct().order_by(CombinationForecast.forecast_month).all()
    return [r[0].strftime("%Y-%m") for r in rows if r[0]]


def _combo_map() -> dict:
    """Return nested map: {pathogen: {antibiotic: [laboratory, ...]}} for all available forecasts."""
    rows = (
        db.session.query(
            CombinationForecast.pathogen,
            CombinationForecast.antibiotic,
            CombinationForecast.laboratory,
        )
        .distinct()
        .order_by(
            CombinationForecast.pathogen,
            CombinationForecast.antibiotic,
            CombinationForecast.laboratory,
        )
        .all()
    )
    result: dict = {}
    for pathogen, antibiotic, laboratory in rows:
        result.setdefault(pathogen, {}).setdefault(antibiotic, []).append(laboratory)
    return result


@bp.route("/clinical")
def clinical():
    return render_template(
        "clinical.html",
        combo_map=_combo_map(),
        months=_forecast_months(),
        result=None,
        error=None,
        form={},
    )


@bp.route("/clinical/query", methods=["POST"])
def clinical_query():
    pathogens = db.session.query(CombinationForecast.pathogen).distinct().order_by(CombinationForecast.pathogen).all()
    antibiotics = db.session.query(CombinationForecast.antibiotic).distinct().order_by(CombinationForecast.antibiotic).all()
    laboratories = db.session.query(CombinationForecast.laboratory).distinct().order_by(CombinationForecast.laboratory).all()

    form = {
        "pathogen": request.form.get("pathogen", ""),
        "antibiotic": request.form.get("antibiotic", ""),
        "laboratory": request.form.get("laboratory", ""),
        "month": request.form.get("month", ""),
    }

    result = None
    error = None

    try:
        from datetime import date
        y, m = form["month"].split("-")
        month_date = date(int(y), int(m), 1)
        row = CombinationForecast.query.filter_by(
            pathogen=form["pathogen"],
            antibiotic=form["antibiotic"],
            laboratory=form["laboratory"],
            forecast_month=month_date,
        ).first()
        if row is None:
            error = "Nessuna previsione trovata per la combinazione selezionata. Eseguire prima la pipeline."
        else:
            result = {
                "pathogen": row.pathogen,
                "antibiotic": row.antibiotic,
                "laboratory": row.laboratory,
                "month": form["month"],
                "sensitive_pct": row.sensitive_pct,
                "intermediate_pct": row.intermediate_pct,
                "resistant_pct": row.resistant_pct,
                "sensitive_lower": row.sensitive_lower,
                "sensitive_upper": row.sensitive_upper,
                "intermediate_lower": row.intermediate_lower,
                "intermediate_upper": row.intermediate_upper,
                "resistant_lower": row.resistant_lower,
                "resistant_upper": row.resistant_upper,
            }
    except Exception as exc:
        error = f"Errore nella ricerca: {exc}"

    return render_template(
        "clinical.html",
        combo_map=_combo_map(),
        months=_forecast_months(),
        result=result,
        error=error,
        form=form,
    )
