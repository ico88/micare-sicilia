from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import next_month_feature
from .hierarchical_training import find_hierarchical_bundle
from .prophet_forecast import forecast_sir_with_prophet
from .training import TARGETS
from .validation import reliability_level

CLASS_LABELS = {"sensitive_pct": "S", "intermediate_pct": "I", "resistant_pct": "R"}
ENSEMBLE_RF_WEIGHT = 0.60
ENSEMBLE_HGB_WEIGHT = 0.40


DISCLAIMER = (
    "Sistema di supporto epidemiologico: la previsione non sostituisce "
    "l'antibiogramma del singolo paziente ne il giudizio clinico."
)


def choose_fallback_scope(history: pd.DataFrame, pathogen: str, antibiotic: str, laboratory: str, ward: str | None):
    if laboratory:
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

    raw_values = None
    decision_values = None
    quantitative_model = model_name
    decision_model = model_name
    x = next_month_feature(pd.to_datetime(prediction_month), pathogen, antibiotic, laboratory, ward, samples)

    if model_name == "prophet":
        raw_values = forecast_sir_with_prophet(scoped_history, prediction_month)
        if raw_values is not None:
            model_name = f"prophet:{scope}"
            quantitative_model = model_name
            decision_model = model_name

    if raw_values is None and model_name == "rf_quant_hgb_class":
        rf_values = _predict_artifact_targets(model_dir, "random_forest", x)
        hgb_values = _predict_artifact_targets(model_dir, "hist_gradient_boosting", x)
        if rf_values is not None:
            raw_values = rf_values
            quantitative_model = "random_forest"
            decision_values = hgb_values or rf_values
            decision_model = "hist_gradient_boosting" if hgb_values is not None else "random_forest"
        else:
            model_name = "random_forest"

    if raw_values is None and model_name == "ensemble_rf_hgb":
        rf_values = _predict_artifact_targets(model_dir, "random_forest", x)
        hgb_values = _predict_artifact_targets(model_dir, "hist_gradient_boosting", x)
        if rf_values is not None and hgb_values is not None:
            raw_values = {
                target: ENSEMBLE_RF_WEIGHT * rf_values[target] + ENSEMBLE_HGB_WEIGHT * hgb_values[target]
                for target in TARGETS
            }
            quantitative_model = f"ensemble_rf{int(ENSEMBLE_RF_WEIGHT * 100)}_hgb{int(ENSEMBLE_HGB_WEIGHT * 100)}"
            decision_values = hgb_values
            decision_model = "hist_gradient_boosting"
        elif rf_values is not None:
            raw_values = rf_values
            quantitative_model = "random_forest"
            decision_values = rf_values
            decision_model = "random_forest"
            model_name = "random_forest"
        elif hgb_values is not None:
            raw_values = hgb_values
            quantitative_model = "hist_gradient_boosting"
            decision_values = hgb_values
            decision_model = "hist_gradient_boosting"
            model_name = "hist_gradient_boosting"
        else:
            model_name = "baseline_historical"

    if raw_values is None and model_name == "auto_hierarchical":
        hierarchical_bundle, hierarchical_metadata = find_hierarchical_bundle(model_dir, pathogen, antibiotic, laboratory, ward)
        if hierarchical_bundle is not None:
            model_name = f"hierarchical_hgb:{hierarchical_metadata['scope']}"
            quantitative_model = model_name
            decision_model = model_name
            scope = hierarchical_metadata["scope"]
            model_rmse = hierarchical_metadata.get("rmse_mean")
            raw_values = {
                target: float(np.clip(hierarchical_bundle["models"][target].predict(x)[0], 0, 100))
                for target in TARGETS
            }

    if raw_values is None:
        artifact_values = _predict_artifact_targets(model_dir, model_name, x) if scope in {"exact", "laboratory", "regional"} else None
        if artifact_values is not None:
            raw_values = artifact_values
            quantitative_model = model_name
            if model_name == "random_forest":
                hgb_values = _predict_artifact_targets(model_dir, "hist_gradient_boosting", x)
                decision_values = hgb_values or artifact_values
                decision_model = "hist_gradient_boosting" if hgb_values is not None else model_name
            else:
                decision_model = model_name
        else:
            model_name = "baseline_historical"
            quantitative_model = model_name
            decision_model = model_name
            if scoped_history.empty:
                means = history[TARGETS].mean().fillna(0)
            else:
                means = scoped_history[TARGETS].tail(12).mean().fillna(0)
            raw_values = means.to_dict()

    normalized = normalize_prediction(raw_values)
    decision = _decision_summary(decision_values or raw_values)
    reliability, reason = reliability_level(samples, model_rmse, historical_std, scope)
    return {
        "prediction_month": pd.to_datetime(prediction_month).to_period("M").to_timestamp("M").date(),
        "pathogen": pathogen,
        "antibiotic": antibiotic,
        "laboratory": laboratory,
        "ward": ward,
        "model_name": model_name,
        "quantitative_model": quantitative_model,
        "decision_model": decision_model,
        "decision_class": decision["class"],
        "decision_confidence": decision["confidence"],
        "sensitive_pct": normalized["sensitive_pct"],
        "intermediate_pct": normalized["intermediate_pct"],
        "resistant_pct": normalized["resistant_pct"],
        "reliability": reliability,
        "reliability_reason": reason,
        "disclaimer": DISCLAIMER,
    }


def _predict_artifact_targets(model_dir: str | Path, model_name: str, x: pd.DataFrame) -> dict[str, float] | None:
    artifact = Path(model_dir) / f"{model_name}.joblib"
    if not artifact.exists():
        return None
    bundle = joblib.load(artifact)
    return {
        target: float(np.clip(bundle["targets"][target].predict(x)[0], 0, 100))
        for target in TARGETS
    }


def _decision_summary(values: dict[str, float]) -> dict[str, float | str]:
    normalized = normalize_prediction(values)
    target = max(TARGETS, key=lambda item: normalized[item])
    return {"class": CLASS_LABELS[target], "confidence": normalized[target]}
