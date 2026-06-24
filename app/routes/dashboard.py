from __future__ import annotations

import pandas as pd
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from app import db
from app.models import AggregatedObservation, Prediction, ValidationMetric
from app.services.database import aggregated_from_db, save_prediction, save_training_summary
from app.services.prediction import DISCLAIMER, predict_sir
from app.services.training import train_all_models

bp = Blueprint("dashboard", __name__)


@bp.get("/dashboard")
def dashboard():
    data = aggregated_from_db()
    filters = _filter_options(data)
    selected = {
        "pathogen": request.args.get("pathogen") or (filters["pathogens"][0] if filters["pathogens"] else ""),
        "antibiotic": request.args.get("antibiotic") or (filters["antibiotics"][0] if filters["antibiotics"] else ""),
        "laboratory": request.args.get("laboratory") or (filters["laboratories"][0] if filters["laboratories"] else ""),
        "ward": request.args.get("ward") or "",
    }
    history = _filtered_history(data, selected)
    latest_prediction = Prediction.query.order_by(Prediction.created_at.desc()).first()
    metrics = ValidationMetric.query.order_by(ValidationMetric.model_name, ValidationMetric.target).all()

    return render_template(
        "dashboard.html",
        has_data=not data.empty,
        filters=filters,
        selected=selected,
        history_chart=_history_chart_payload(history),
        latest_prediction=latest_prediction,
        metrics=metrics,
        disclaimer=DISCLAIMER,
    )


@bp.post("/train")
def train():
    data = aggregated_from_db()
    if data.empty:
        flash("Carica prima un dataset.", "warning")
        return redirect(url_for("upload.upload"))

    summary = train_all_models(data, current_app.config["MODEL_FOLDER"])
    save_training_summary(summary)
    db.session.commit()
    flash("Training completato: baseline, HistGradientBoosting e RandomForest aggiornati.", "success")
    return redirect(url_for("dashboard.dashboard"))


@bp.post("/predict")
def predict():
    data = aggregated_from_db()
    if data.empty:
        flash("Carica prima un dataset.", "warning")
        return redirect(url_for("upload.upload"))

    prediction = predict_sir(
        data,
        current_app.config["MODEL_FOLDER"],
        pathogen=request.form["pathogen"],
        antibiotic=request.form["antibiotic"],
        laboratory=request.form["laboratory"],
        ward=request.form.get("ward") or "",
        prediction_month=request.form["prediction_month"],
        model_name=request.form.get("model_name") or "hist_gradient_boosting",
    )
    save_prediction(prediction)
    db.session.commit()
    flash("Previsione generata.", "success")
    return redirect(url_for("dashboard.dashboard", pathogen=prediction["pathogen"], antibiotic=prediction["antibiotic"], laboratory=prediction["laboratory"]))


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
        & (data["laboratory"] == selected["laboratory"])
    )
    if selected.get("ward"):
        mask &= data["ward"].fillna("") == selected["ward"]
    return data[mask].sort_values("month")


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
