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


def _render_clinical(**kwargs):
    """Render clinical.html with sane defaults for both tabs."""
    defaults = dict(
        combo_map=_combo_map(),
        months=_forecast_months(),
        # Observational tab
        result=None,
        error=None,
        form={},
        active_tab="osservazionale",
        # Predictive tab
        pred_form={},
        pred_results=None,
        pred_error=None,
    )
    defaults.update(kwargs)
    return render_template("clinical.html", **defaults)


@bp.route("/clinical")
def clinical():
    return _render_clinical()


@bp.route("/clinical/query", methods=["POST"])
def clinical_query():
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

    return _render_clinical(result=result, error=error, form=form, active_tab="osservazionale")


@bp.route("/clinical/predictive", methods=["GET", "POST"])
def predictive():
    pred_form: dict = {}
    pred_results = None
    pred_error = None

    if request.method == "POST":
        import numpy as _np
        from collections import defaultdict

        pred_form = {
            "pathogen": request.form.get("pathogen", ""),
            "horizon": request.form.get("horizon", "6"),
        }

        try:
            from datetime import date as _date
            horizon_months = int(pred_form["horizon"])
            today = _date.today()
            m = today.month - 1 + horizon_months
            target_date = _date(today.year + m // 12, m % 12 + 1, 1)
            pred_form["target_month"] = target_date.strftime("%B %Y")
            pred_form["target_date_str"] = target_date.strftime("%Y-%m")

            # Previsioni: tutti i laboratori aggregati a livello regionale
            rows = CombinationForecast.query.filter_by(
                pathogen=pred_form["pathogen"],
                forecast_month=target_date,
            ).all()

            if not rows:
                pred_error = (
                    f"Nessuna previsione disponibile per {pred_form['pathogen']} "
                    f"a {pred_form.get('target_month', '')}. Eseguire prima la pipeline."
                )
            else:
                # Media storica 12 mesi — pesata per numero di isolati (non media di percentuali)
                cutoff = _date(today.year - 1, today.month, 1)
                hist_rows = AggregatedObservation.query.filter(
                    AggregatedObservation.pathogen == pred_form["pathogen"],
                    AggregatedObservation.month >= cutoff,
                ).all()

                hist_acc: dict = defaultdict(lambda: {"s": 0.0, "r": 0.0, "n": 0})
                for h in hist_rows:
                    n = h.samples or 0
                    if n > 0:
                        hist_acc[h.antibiotic]["s"] += h.sensitive_count or 0.0
                        hist_acc[h.antibiotic]["r"] += h.resistant_count or 0.0
                        hist_acc[h.antibiotic]["n"] += n

                hist_summary: dict = {
                    ab: {
                        "s": d["s"] / d["n"] * 100,
                        "r": d["r"] / d["n"] * 100,
                        "n": d["n"],
                    }
                    for ab, d in hist_acc.items() if d["n"] > 0
                }

                # Aggrega previsioni per antibiotico (media su tutti i laboratori)
                ab_data: dict = defaultdict(list)
                for r in rows:
                    ab_data[r.antibiotic].append({
                        "s": r.sensitive_pct or 0.0,
                        "r": r.resistant_pct or 0.0,
                        "r_lo": r.resistant_lower,
                        "r_hi": r.resistant_upper,
                        "s_lo": r.sensitive_lower,
                        "s_hi": r.sensitive_upper,
                    })

                pred_results = []
                for ab, entries in ab_data.items():
                    s_vals = [e["s"] for e in entries]
                    r_vals = [e["r"] for e in entries]
                    r_lo_vals = [e["r_lo"] for e in entries if e["r_lo"] is not None]
                    r_hi_vals = [e["r_hi"] for e in entries if e["r_hi"] is not None]
                    s_lo_vals = [e["s_lo"] for e in entries if e["s_lo"] is not None]
                    s_hi_vals = [e["s_hi"] for e in entries if e["s_hi"] is not None]
                    mean_s = float(_np.mean(s_vals))
                    mean_r = float(_np.mean(r_vals))

                    if mean_s < 1.0 and mean_r < 1.0:
                        continue

                    hist = hist_summary.get(ab)
                    hist_r = hist["r"] if hist else None
                    trend = "stable"
                    if hist_r is not None:
                        delta = mean_r - hist_r
                        if delta > 5:
                            trend = "worse"
                        elif delta < -5:
                            trend = "better"

                    pred_results.append({
                        "antibiotic": ab,
                        "sensitive_pct": mean_s,
                        "resistant_pct": mean_r,
                        "sensitive_lower": float(_np.mean(s_lo_vals)) if s_lo_vals else None,
                        "sensitive_upper": float(_np.mean(s_hi_vals)) if s_hi_vals else None,
                        "resistant_lower": float(_np.mean(r_lo_vals)) if r_lo_vals else None,
                        "resistant_upper": float(_np.mean(r_hi_vals)) if r_hi_vals else None,
                        "n_labs": len(entries),
                        "hist_sensitive_pct": hist["s"] if hist else None,
                        "hist_resistant_pct": hist_r,
                        "hist_n_isolates": hist["n"] if hist else None,
                        "trend": trend,
                    })

                pred_results.sort(key=lambda x: (-x["sensitive_pct"], x["resistant_pct"]))

        except Exception as exc:
            pred_error = f"Errore nella ricerca: {exc}"

    return _render_clinical(
        active_tab="predittiva",
        pred_form=pred_form,
        pred_results=pred_results,
        pred_error=pred_error,
    )
