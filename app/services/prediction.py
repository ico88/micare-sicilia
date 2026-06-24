from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import next_month_feature
from .training import TARGETS
from .validation import reliability_level


DISCLAIMER = (
    "Sistema di supporto epidemiologico: la previsione non sostituisce "
    "l'antibiogramma del singolo paziente ne il giudizio clinico."
)


def choose_fallback_scope(history: pd.DataFrame, pathogen: str, antibiotic: str, laboratory: str, ward: str | None):
    exact = history[
        (history["pathogen"] == pathogen)
        & (history["antibiotic"] == antibiotic)
        & (history["laboratory"] == laboratory)
        & (history["ward"].fillna("") == (ward or ""))
    ]
    if len(exact) >= 6:
        return "exact", exact

    lab_level = history[(history["pathogen"] == pathogen) & (history["antibiotic"] == antibiotic) & (history["laboratory"] == laboratory)]
    if len(lab_level) >= 6:
        return "laboratory", lab_level

    regional = history[(history["pathogen"] == pathogen) & (history["antibiotic"] == antibiotic)]
    if len(regional) >= 3:
        return "regional", regional

    return "historical_mean", history


def normalize_prediction(values: dict[str, float]) -> dict[str, float]:
    clipped = {target: max(0.0, min(100.0, float(value))) for target, value in values.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {target: 0.0 for target in TARGETS}
    return {target: clipped[target] * 100 / total for target in TARGETS}


def predict_sir(
    history: pd.DataFrame,
    model_dir: str | Path,
    pathogen: str,
    antibiotic: str,
    laboratory: str,
    prediction_month: str,
    ward: str | None = "",
    model_name: str = "hist_gradient_boosting",
    model_rmse: float | None = None,
) -> dict:
    history = history.copy()
    history["month"] = pd.to_datetime(history["month"])
    pathogen = pathogen.strip().upper()
    antibiotic = antibiotic.strip().upper()
    laboratory = laboratory.strip().upper()
    ward = (ward or "").strip().upper()

    scope, scoped_history = choose_fallback_scope(history, pathogen, antibiotic, laboratory, ward)
    samples = int(scoped_history["samples"].tail(6).mean()) if not scoped_history.empty else 0
    historical_std = float(scoped_history[TARGETS].tail(12).std().mean()) if len(scoped_history) > 1 else 999.0

    artifact = Path(model_dir) / f"{model_name}.joblib"
    if artifact.exists() and scope in {"exact", "laboratory", "regional"}:
        bundle = joblib.load(artifact)
        x = next_month_feature(pd.to_datetime(prediction_month), pathogen, antibiotic, laboratory, ward, samples)
        raw_values = {
            target: float(np.clip(bundle["targets"][target].predict(x)[0], 0, 100))
            for target in TARGETS
        }
    else:
        model_name = "baseline_historical"
        if scoped_history.empty:
            means = history[TARGETS].mean().fillna(0)
        else:
            means = scoped_history[TARGETS].tail(12).mean().fillna(0)
        raw_values = means.to_dict()

    normalized = normalize_prediction(raw_values)
    reliability, reason = reliability_level(samples, model_rmse, historical_std, scope)
    return {
        "prediction_month": pd.to_datetime(prediction_month).to_period("M").to_timestamp("M").date(),
        "pathogen": pathogen,
        "antibiotic": antibiotic,
        "laboratory": laboratory,
        "ward": ward,
        "model_name": model_name,
        "sensitive_pct": normalized["sensitive_pct"],
        "intermediate_pct": normalized["intermediate_pct"],
        "resistant_pct": normalized["resistant_pct"],
        "reliability": reliability,
        "reliability_reason": reason,
        "disclaimer": DISCLAIMER,
    }
