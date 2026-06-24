from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Blueprint, Response, current_app, flash, jsonify, redirect, render_template, request, url_for

from app import db
from app.models import Prediction, ValidationMetric
from app.services.backtest import BACKTEST_PREDICTIONS, load_rolling_backtest
from app.services.backtest_jobs import get_backtest_job, start_backtest_job
from app.services.database import aggregated_from_db, save_prediction
from app.services.prediction import DISCLAIMER, predict_sir
from app.services.scientific_report import build_scientific_report, report_tables
from app.services.training_jobs import control_training_job, get_training_job, start_training_job

bp = Blueprint("dashboard", __name__)

COMPARABLE_MODELS = [
    "rf_quant_hgb_class",
    "ensemble_rf_hgb",
    "auto_hierarchical",
    "hist_gradient_boosting",
    "random_forest",
    "prophet",
]



@bp.get("/dashboard")
def dashboard():
    data = aggregated_from_db()
    filters = _filter_options(data)
    selected = {
        "pathogen": request.args.get("pathogen") or (filters["pathogens"][0] if filters["pathogens"] else ""),
        "antibiotic": request.args.get("antibiotic") or (filters["antibiotics"][0] if filters["antibiotics"] else ""),
        "laboratory": request.args.get("laboratory") or "",
        "ward": request.args.get("ward") or "",
        "month_from": request.args.get("month_from") or "",
        "month_to": request.args.get("month_to") or "",
    }
    selected_tab = request.args.get("tab") if request.args.get("tab") in {"history", "prediction", "metrics"} else "history"
    history = _filtered_history(data, selected)
    latest_prediction = _latest_prediction_for_selection(selected)
    latest_prediction_comparison = _prediction_actual_comparison(data, latest_prediction)
    recent_predictions = _recent_predictions_for_selection(selected)
    metrics = ValidationMetric.query.order_by(ValidationMetric.model_name, ValidationMetric.target).all()
    month_bounds = _month_bounds(data)
    prediction_year = _selected_prediction_year(request.args.get("prediction_year"), month_bounds[1])
    annual_prediction = _annual_prediction_summary(data, selected, prediction_year)

    return render_template(
        "dashboard.html",
        has_data=not data.empty,
        filters=filters,
        selected=selected,
        selected_tab=selected_tab,
        history_chart=_history_chart_payload(history),
        history_summary=_history_summary(history),
        latest_prediction=latest_prediction,
        prediction_chart=_prediction_chart_payload(latest_prediction),
        latest_prediction_comparison=latest_prediction_comparison,
        recent_predictions=recent_predictions,
        annual_prediction=annual_prediction,
        prediction_month_min=month_bounds[0],
        prediction_month_default=month_bounds[1],
        prediction_year=prediction_year,
        metrics=metrics,
        disclaimer=DISCLAIMER,
    )


@bp.get("/validation")
def validation_report():
    data = aggregated_from_db()
    metrics = ValidationMetric.query.order_by(ValidationMetric.model_name, ValidationMetric.target).all()
    report = build_scientific_report(data, metrics, current_app.config["MODEL_FOLDER"])
    backtest = load_rolling_backtest(current_app.config["MODEL_FOLDER"])
    return render_template("validation_report.html", has_data=not data.empty, report=report, backtest=backtest, disclaimer=DISCLAIMER)


@bp.post("/validation/backtest")
def validation_backtest():
    data = aggregated_from_db()
    if data.empty:
        flash("Carica prima un dataset.", "warning")
        return redirect(url_for("upload.upload"))
    months = int(request.form.get("months") or 6)
    months = max(3, min(months, 12))
    job_id, created = start_backtest_job(current_app._get_current_object(), current_app.config["MODEL_FOLDER"], months=months)
    if created:
        flash("Backtest rolling avviato in background.", "success")
    else:
        flash("Backtest gia in corso: continuo a mostrarti l'avanzamento.", "warning")
    return redirect(url_for("dashboard.backtest_status", job_id=job_id))


@bp.get("/validation/backtest/status")
def backtest_status():
    job = get_backtest_job(request.args.get("job_id"))
    if job is None:
        flash("Nessun backtest in corso.", "warning")
        return redirect(url_for("dashboard.validation_report"))
    return render_template("backtest_status.html", job=job)


@bp.get("/validation/backtest/status.json")
def backtest_status_json():
    job = get_backtest_job(request.args.get("job_id"))
    if job is None:
        return jsonify({"status": "missing", "message": "Job non trovato.", "progress": 0}), 404
    return jsonify(
        {
            "id": job["id"],
            "status": job["status"],
            "progress": job["progress"],
            "stage": job["stage"],
            "message": job["message"],
            "error": job["error"],
            "finished": job["status"] in {"success", "error"},
            "summary": job.get("summary", {}),
        }
    )


@bp.get("/validation/export/backtest_predictions.csv")
def export_backtest_predictions():
    path = Path(current_app.config["MODEL_FOLDER"]) / BACKTEST_PREDICTIONS
    if not path.exists():
        flash("Esegui prima il backtest rolling.", "warning")
        return redirect(url_for("dashboard.validation_report"))
    return Response(
        path.read_text(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=mic_res_sicilia_backtest_predictions.csv"},
    )


@bp.get("/validation/export/<table_name>.csv")
def export_validation_table(table_name: str):
    data = aggregated_from_db()
    metrics = ValidationMetric.query.order_by(ValidationMetric.model_name, ValidationMetric.target).all()
    report = build_scientific_report(data, metrics, current_app.config["MODEL_FOLDER"])
    tables = report_tables(report)
    if table_name not in tables:
        flash("Tabella report non disponibile.", "warning")
        return redirect(url_for("dashboard.validation_report"))
    return _csv_response(tables[table_name], f"mic_res_sicilia_report_{table_name}.csv")


@bp.post("/train")
def train():
    data = aggregated_from_db()
    if data.empty:
        flash("Carica prima un dataset.", "warning")
        return redirect(url_for("upload.upload"))

    job_id, created = start_training_job(current_app._get_current_object(), current_app.config["MODEL_FOLDER"])
    if created:
        flash("Training avviato in background.", "success")
    else:
        flash("Training gia in corso: continuo a mostrarti l'avanzamento.", "warning")
    return redirect(url_for("dashboard.train_status", job_id=job_id))


@bp.get("/train/status")
def train_status():
    job = get_training_job(request.args.get("job_id"))
    if job is None:
        flash("Nessun training in corso.", "warning")
        return redirect(url_for("dashboard.dashboard"))
    return render_template("train_status.html", job=job)


@bp.get("/train/status.json")
def train_status_json():
    job = get_training_job(request.args.get("job_id"))
    if job is None:
        return jsonify({"status": "missing", "message": "Job non trovato.", "progress": 0}), 404
    payload = {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "message": job["message"],
        "error": job["error"],
        "elapsed_seconds": job.get("elapsed_seconds", 0),
        "eta_seconds": job.get("eta_seconds"),
        "eta_text": job.get("eta_text", "Calcolo stima..."),
        "work_completed": job.get("work_completed", 0),
        "work_total": job.get("work_total", 0),
        "work_label": job.get("work_label", ""),
        "finished": job["status"] in {"success", "error", "stopped"},
        "summary": job.get("summary", {}),
    }
    return jsonify(payload)


@bp.post("/train/control/<action>")
def train_control(action: str):
    if action not in {"pause", "resume", "stop"}:
        return jsonify({"status": "error", "message": "Azione non valida."}), 400
    job = control_training_job(request.form.get("job_id"), action)
    if job is None:
        return jsonify({"status": "missing", "message": "Job non trovato."}), 404
    return jsonify(
        {
            "id": job["id"],
            "status": job["status"],
            "progress": job["progress"],
            "stage": job["stage"],
            "message": job["message"],
            "finished": job["status"] in {"success", "error", "stopped"},
        }
    )


@bp.get("/export/aggregated.csv")
def export_aggregated_csv():
    data = aggregated_from_db()
    if data.empty:
        flash("Nessun dato da esportare.", "warning")
        return redirect(url_for("dashboard.dashboard"))

    filters = _filter_options(data)
    selected = {
        "pathogen": request.args.get("pathogen") or (filters["pathogens"][0] if filters["pathogens"] else ""),
        "antibiotic": request.args.get("antibiotic") or (filters["antibiotics"][0] if filters["antibiotics"] else ""),
        "laboratory": request.args.get("laboratory") or "",
        "ward": request.args.get("ward") or "",
        "month_from": request.args.get("month_from") or "",
        "month_to": request.args.get("month_to") or "",
    }
    exported = _filtered_history(data, selected)
    return _csv_response(exported, "mic_res_sicilia_aggregato.csv")


@bp.get("/export/predictions.csv")
def export_predictions_csv():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).all()
    exported = pd.DataFrame(
        [
            {
                "created_at": row.created_at,
                "prediction_month": row.prediction_month,
                "pathogen": row.pathogen,
                "antibiotic": row.antibiotic,
                "laboratory": row.laboratory or "Tutti",
                "ward": row.ward or "Tutti",
                "model_name": row.model_name,
                "quantitative_model": row.quantitative_model,
                "decision_model": row.decision_model,
                "decision_class": row.decision_class,
                "decision_confidence": row.decision_confidence,
                "sensitive_pct": row.sensitive_pct,
                "intermediate_pct": row.intermediate_pct,
                "resistant_pct": row.resistant_pct,
                "reliability": row.reliability,
                "reliability_reason": row.reliability_reason,
            }
            for row in rows
        ]
    )
    return _csv_response(exported, "mic_res_sicilia_previsioni.csv")


@bp.post("/predict")
def predict():
    data = aggregated_from_db()
    if data.empty:
        flash("Carica prima un dataset.", "warning")
        return redirect(url_for("upload.upload"))

    model_names = _requested_prediction_models(request.form.get("model_name"))
    predictions = []
    for model_name in model_names:
        prediction = predict_sir(
            data,
            current_app.config["MODEL_FOLDER"],
            pathogen=request.form["pathogen"],
            antibiotic=request.form["antibiotic"],
            laboratory=request.form["laboratory"],
            ward=request.form.get("ward") or "",
            prediction_month=request.form["prediction_month"],
            model_name=model_name,
        )
        save_prediction(prediction)
        predictions.append(prediction)
    db.session.commit()
    flash("Previsioni generate per tutti i modelli." if len(predictions) > 1 else "Previsione generata.", "success")
    first = predictions[0]
    return redirect(
        url_for(
            "dashboard.dashboard",
            pathogen=first["pathogen"],
            antibiotic=first["antibiotic"],
            laboratory=first["laboratory"],
            ward=first.get("ward") or "",
            tab="prediction",
        )
    )


@bp.post("/predict/year")
def predict_year():
    data = aggregated_from_db()
    if data.empty:
        flash("Carica prima un dataset.", "warning")
        return redirect(url_for("upload.upload"))

    year = _selected_prediction_year(request.form.get("prediction_year"), _month_bounds(data)[1])
    model_names = _requested_prediction_models(request.form.get("model_name"))
    predictions = []
    for model_name in model_names:
        for month in range(1, 13):
            prediction = predict_sir(
                data,
                current_app.config["MODEL_FOLDER"],
                pathogen=request.form["pathogen"],
                antibiotic=request.form["antibiotic"],
                laboratory=request.form["laboratory"],
                ward=request.form.get("ward") or "",
                prediction_month=f"{year}-{month:02d}",
                model_name=model_name,
            )
            save_prediction(prediction)
            predictions.append(prediction)
    db.session.commit()
    if len(model_names) > 1:
        flash(f"Previsione annuale {year} generata per {len(model_names)} modelli ({len(predictions)} righe).", "success")
    else:
        flash(f"Previsione annuale {year} generata su 12 mesi.", "success")
    first = predictions[0]
    return redirect(
        url_for(
            "dashboard.dashboard",
            pathogen=first["pathogen"],
            antibiotic=first["antibiotic"],
            laboratory=first["laboratory"],
            ward=first.get("ward") or "",
            prediction_year=year,
            tab="prediction",
        )
    )


def _requested_prediction_models(model_name: str | None) -> list[str]:
    if model_name == "all_models":
        return COMPARABLE_MODELS
    if model_name in COMPARABLE_MODELS:
        return [model_name]
    return ["rf_quant_hgb_class"]


def _latest_prediction_for_selection(selected: dict[str, str]) -> Prediction | None:
    query = Prediction.query.filter_by(
        pathogen=selected.get("pathogen") or "",
        antibiotic=selected.get("antibiotic") or "",
        laboratory=selected.get("laboratory") or "",
        ward=selected.get("ward") or "",
    )
    return query.order_by(Prediction.created_at.desc()).first()


def _recent_predictions_for_selection(selected: dict[str, str]) -> list[Prediction]:
    query = Prediction.query.filter_by(
        pathogen=selected.get("pathogen") or "",
        antibiotic=selected.get("antibiotic") or "",
        laboratory=selected.get("laboratory") or "",
        ward=selected.get("ward") or "",
    )
    return query.order_by(Prediction.created_at.desc()).limit(8).all()


def _filter_options(data: pd.DataFrame) -> dict[str, list[str]]:
    if data.empty:
        return {"pathogens": [], "antibiotics": [], "laboratories": [], "wards": []}
    return {
        "pathogens": sorted(data["pathogen"].dropna().unique()),
        "antibiotics": sorted(data["antibiotic"].dropna().unique()),
        "laboratories": sorted(data["laboratory"].dropna().unique()),
        "wards": sorted([value for value in data["ward"].dropna().unique() if value]),
    }


def _filtered_history(data: pd.DataFrame, selected: dict[str, str]) -> pd.DataFrame:
    if data.empty:
        return data
    mask = (
        (data["pathogen"] == selected["pathogen"])
        & (data["antibiotic"] == selected["antibiotic"])
    )
    if selected.get("laboratory"):
        mask &= data["laboratory"] == selected["laboratory"]
    if selected.get("ward"):
        mask &= data["ward"].fillna("") == selected["ward"]
    months = pd.to_datetime(data["month"])
    if selected.get("month_from"):
        mask &= months >= pd.to_datetime(selected["month_from"]).to_period("M").to_timestamp("M")
    if selected.get("month_to"):
        mask &= months <= pd.to_datetime(selected["month_to"]).to_period("M").to_timestamp("M")
    return _aggregate_history_for_display(data[mask])


def _aggregate_history_for_display(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    grouped = (
        history.groupby("month", dropna=False)
        .agg(
            samples=("samples", "sum"),
            sensitive_count=("sensitive_count", "sum"),
            intermediate_count=("intermediate_count", "sum"),
            resistant_count=("resistant_count", "sum"),
        )
        .reset_index()
    )
    total = grouped[["sensitive_count", "intermediate_count", "resistant_count"]].sum(axis=1).replace(0, pd.NA)
    grouped["sensitive_pct"] = (grouped["sensitive_count"] / total * 100).fillna(0)
    grouped["intermediate_pct"] = (grouped["intermediate_count"] / total * 100).fillna(0)
    grouped["resistant_pct"] = (grouped["resistant_count"] / total * 100).fillna(0)
    return grouped.sort_values("month")


def _history_summary(history: pd.DataFrame) -> dict:
    if history.empty:
        return {"months": 0, "samples": 0, "latest_month": "", "latest_samples": 0}
    latest = history.sort_values("month").iloc[-1]
    return {
        "months": int(history["month"].nunique()),
        "samples": int(history["samples"].sum()),
        "latest_month": pd.to_datetime(latest["month"]).strftime("%Y-%m"),
        "latest_samples": int(latest["samples"]),
    }


def _prediction_chart_payload(prediction: Prediction | None) -> dict:
    if prediction is None:
        return {"labels": [], "values": []}
    return {
        "labels": ["S", "I", "R"],
        "values": [
            round(prediction.sensitive_pct, 2),
            round(prediction.intermediate_pct, 2),
            round(prediction.resistant_pct, 2),
        ],
    }


def _csv_response(data: pd.DataFrame, filename: str) -> Response:
    csv = data.to_csv(index=False)
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _history_chart_payload(history: pd.DataFrame) -> dict:
    if history.empty:
        return {"labels": [], "s": [], "i": [], "r": [], "samples": []}
    return {
        "labels": [pd.to_datetime(value).strftime("%Y-%m") for value in history["month"]],
        "s": [round(value, 2) for value in history["sensitive_pct"]],
        "i": [round(value, 2) for value in history["intermediate_pct"]],
        "r": [round(value, 2) for value in history["resistant_pct"]],
        "samples": [int(value) for value in history["samples"]],
    }


def _month_bounds(data: pd.DataFrame) -> tuple[str, str]:
    if data.empty:
        return "", ""
    months = pd.to_datetime(data["month"])
    first_month = months.min().to_period("M").strftime("%Y-%m")
    next_month = (months.max().to_period("M") + 1).strftime("%Y-%m")
    return first_month, next_month


def _selected_prediction_year(raw_year: str | None, default_month: str) -> int:
    fallback = int(default_month[:4]) if default_month else pd.Timestamp.today().year
    try:
        year = int(raw_year or fallback)
    except (TypeError, ValueError):
        return fallback
    return max(2000, min(year, 2100))


def _annual_prediction_summary(data: pd.DataFrame, selected: dict[str, str], year: int) -> dict:
    start = pd.Timestamp(year=year, month=1, day=1).date()
    end = pd.Timestamp(year=year, month=12, day=31).date()
    query = (
        Prediction.query.filter_by(
            pathogen=selected.get("pathogen") or "",
            antibiotic=selected.get("antibiotic") or "",
            laboratory=selected.get("laboratory") or "",
            ward=selected.get("ward") or "",
        )
        .filter(Prediction.prediction_month >= start, Prediction.prediction_month <= end)
        .order_by(Prediction.created_at.desc())
    )
    latest_by_model_month: dict[tuple[str, str], Prediction] = {}
    for item in query.all():
        month_key = item.prediction_month.strftime("%Y-%m")
        latest_by_model_month.setdefault((item.model_name, month_key), item)

    rows = []
    for (model_name, month), item in sorted(latest_by_model_month.items(), key=lambda pair: (pair[0][0], pair[0][1])):
        rows.append(
            {
                "month": month,
                "model_name": model_name,
                "decision_class": item.decision_class or _class_from_values(item.sensitive_pct, item.intermediate_pct, item.resistant_pct),
                "sensitive_pct": item.sensitive_pct,
                "intermediate_pct": item.intermediate_pct,
                "resistant_pct": item.resistant_pct,
            }
        )

    summary = {"year": year, "available": bool(rows), "rows": rows, "models": [], "comparison": None}
    if not rows:
        return summary

    annual = pd.DataFrame(rows)
    model_summaries = []
    for model_name, group in annual.groupby("model_name", sort=True):
        means = {
            "sensitive_pct": float(group["sensitive_pct"].mean()),
            "intermediate_pct": float(group["intermediate_pct"].mean()),
            "resistant_pct": float(group["resistant_pct"].mean()),
        }
        worst = group.sort_values("resistant_pct", ascending=False).iloc[0]
        model_summaries.append(
            {
                "model_name": model_name,
                "months": int(len(group)),
                **means,
                "decision_class": _class_from_values(means["sensitive_pct"], means["intermediate_pct"], means["resistant_pct"]),
                "worst_month": worst["month"],
                "worst_resistant_pct": float(worst["resistant_pct"]),
                "comparison": _annual_actual_comparison(data, selected, year, means),
            }
        )

    primary = next((item for item in model_summaries if item["model_name"] == "rf_quant_hgb_class"), model_summaries[0])
    summary.update(
        {
            "models": model_summaries,
            "months": primary["months"],
            "model_name": primary["model_name"],
            "sensitive_pct": primary["sensitive_pct"],
            "intermediate_pct": primary["intermediate_pct"],
            "resistant_pct": primary["resistant_pct"],
            "decision_class": primary["decision_class"],
            "worst_month": primary["worst_month"],
            "worst_resistant_pct": primary["worst_resistant_pct"],
            "comparison": primary["comparison"],
        }
    )
    return summary


def _annual_actual_comparison(data: pd.DataFrame, selected: dict[str, str], year: int, predicted: dict[str, float]) -> dict | None:
    if data.empty:
        return None
    months = pd.to_datetime(data["month"])
    mask = (
        (months.dt.year == year)
        & (data["pathogen"] == selected.get("pathogen"))
        & (data["antibiotic"] == selected.get("antibiotic"))
    )
    if selected.get("laboratory"):
        mask &= data["laboratory"] == selected["laboratory"]
    if selected.get("ward"):
        mask &= data["ward"].fillna("") == selected["ward"]
    actual = _aggregate_history_for_display(data[mask])
    if actual.empty:
        return None
    total_samples = int(actual["samples"].sum())
    actual_means = {
        "sensitive_pct": float(actual["sensitive_pct"].mean()),
        "intermediate_pct": float(actual["intermediate_pct"].mean()),
        "resistant_pct": float(actual["resistant_pct"].mean()),
    }
    rows = []
    for label, attr in [("S", "sensitive_pct"), ("I", "intermediate_pct"), ("R", "resistant_pct")]:
        delta = predicted[attr] - actual_means[attr]
        rows.append({"label": label, "predicted": predicted[attr], "observed": actual_means[attr], "delta": delta, "abs_delta": abs(delta)})
    return {
        "samples": total_samples,
        "months": int(actual["month"].nunique()),
        "rows": rows,
        "mean_abs_delta": sum(row["abs_delta"] for row in rows) / len(rows),
    }


def _class_from_values(sensitive: float, intermediate: float, resistant: float) -> str:
    values = {"S": sensitive, "I": intermediate, "R": resistant}
    return max(values, key=values.get)


def _prediction_actual_comparison(data: pd.DataFrame, prediction: Prediction | None) -> dict | None:
    if prediction is None or data.empty:
        return None

    month = pd.to_datetime(prediction.prediction_month).to_period("M").to_timestamp("M")
    mask = (
        (pd.to_datetime(data["month"]) == month)
        & (data["pathogen"] == prediction.pathogen)
        & (data["antibiotic"] == prediction.antibiotic)
    )
    if prediction.laboratory:
        mask &= data["laboratory"] == prediction.laboratory
    if prediction.ward:
        mask &= data["ward"].fillna("") == prediction.ward
    actual = _aggregate_history_for_display(data[mask])
    if actual.empty:
        return None

    actual_row = actual.iloc[0]
    rows = []
    for label, attr in [
        ("S", "sensitive_pct"),
        ("I", "intermediate_pct"),
        ("R", "resistant_pct"),
    ]:
        predicted = float(getattr(prediction, attr))
        observed = float(actual_row[attr])
        rows.append(
            {
                "label": label,
                "predicted": predicted,
                "observed": observed,
                "delta": predicted - observed,
                "abs_delta": abs(predicted - observed),
            }
        )

    return {
        "month": month.strftime("%Y-%m"),
        "samples": int(actual_row["samples"]),
        "rows": rows,
        "mean_abs_delta": sum(row["abs_delta"] for row in rows) / len(rows),
    }
