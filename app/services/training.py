from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .feature_engineering import FEATURE_COLUMNS, make_features
from .validation import regression_metrics, sir_classification_metrics

TARGETS = ["sensitive_pct", "intermediate_pct", "resistant_pct"]
CATEGORICAL_FEATURES = ["pathogen", "antibiotic", "laboratory", "ward"]
NUMERIC_FEATURES = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]


def temporal_train_test_split(df: pd.DataFrame, test_months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.sort_values("month").copy()
    unique_months = sorted(data["month"].dropna().unique())
    if len(unique_months) <= test_months:
        split_index = max(1, int(len(unique_months) * 0.8))
        cutoff = unique_months[split_index - 1]
    else:
        cutoff = unique_months[-test_months - 1]
    train = data[data["month"] <= cutoff].copy()
    test = data[data["month"] > cutoff].copy()
    if test.empty:
        return train, train.tail(min(len(train), 3)).copy()
    return train, test


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def build_model(model_name: str):
    if model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, random_state=42)
    elif model_name == "random_forest":
        estimator = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Modello non supportato: {model_name}")

    return Pipeline([("preprocess", _preprocessor()), ("model", estimator)])


def historical_baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    global_mean = train[TARGETS].mean()
    group_means = train.groupby(["pathogen", "antibiotic", "laboratory"], dropna=False)[TARGETS].mean()
    predictions = []

    for _, row in test.iterrows():
        key = (row["pathogen"], row["antibiotic"], row["laboratory"])
        if key in group_means.index:
            values = group_means.loc[key]
        else:
            values = global_mean
        predictions.append(values.to_dict())

    return pd.DataFrame(predictions, index=test.index).fillna(0)


def train_all_models(aggregated: pd.DataFrame, model_dir: str | Path, progress_callback=None) -> dict:
    started_at = time.perf_counter()
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    global_work_total = 10

    def report(progress: int, stage: str, message: str, work_completed: int | None = None) -> None:
        if progress_callback is not None:
            payload = {"work_label": "Training globale", "work_total": global_work_total}
            if work_completed is not None:
                payload["work_completed"] = work_completed
            progress_callback(progress, stage, message, **payload)

    report(10, "Preparazione", "Preparo dataset e split temporale.", work_completed=1)
    data = aggregated.copy()
    data["month"] = pd.to_datetime(data["month"])
    train, test = temporal_train_test_split(data)
    metrics: list[dict] = []
    artifacts: list[dict] = []
    model_test_predictions: dict[str, pd.DataFrame] = {}

    report(20, "Baseline", "Calcolo baseline storica.", work_completed=2)
    baseline_pred = historical_baseline_predictions(train, test)
    baseline_metric = _collect_metrics("baseline_historical", test[TARGETS], baseline_pred)
    metrics.extend(baseline_metric)

    model_progress = {"hist_gradient_boosting": 35, "random_forest": 65}
    target_progress_step = 8
    for model_name in ["hist_gradient_boosting", "random_forest"]:
        display_name = "HistGradientBoosting" if model_name == "hist_gradient_boosting" else "RandomForest"
        report(model_progress[model_name], display_name, f"Preparo feature per {display_name}.")
        target_models = {}
        test_predictions = pd.DataFrame(index=test.index)
        x_train = make_features(train)
        x_test = make_features(test)

        for index, target in enumerate(TARGETS, start=1):
            completed_units = 2 + ((0 if model_name == "hist_gradient_boosting" else len(TARGETS)) + index)
            report(
                model_progress[model_name] + index * target_progress_step,
                display_name,
                f"Addestro target {target}.",
                work_completed=completed_units,
            )
            model = build_model(model_name)
            model.fit(x_train, train[target])
            pred = np.clip(model.predict(x_test), 0, 100)
            test_predictions[target] = pred
            target_models[target] = model
            report(
                model_progress[model_name] + index * target_progress_step,
                display_name,
                f"Completato target {target}.",
                work_completed=completed_units,
            )

        artifact_file = model_path / f"{model_name}.joblib"
        joblib.dump(
            {
                "model_name": model_name,
                "targets": target_models,
                "trained_until": str(train["month"].max().date()),
                "feature_columns": FEATURE_COLUMNS,
            },
            artifact_file,
        )
        artifacts.append(
            {
                "model_name": model_name,
                "artifact_path": str(artifact_file),
                "metadata": {"trained_until": str(train["month"].max().date())},
            }
        )
        metrics.extend(_collect_metrics(model_name, test[TARGETS], test_predictions))
        model_test_predictions[model_name] = test_predictions

    if {"random_forest", "hist_gradient_boosting"}.issubset(model_test_predictions):
        ensemble_predictions = (
            0.60 * model_test_predictions["random_forest"]
            + 0.40 * model_test_predictions["hist_gradient_boosting"]
        )
        metrics.extend(_collect_metrics("ensemble_rf60_hgb40", test[TARGETS], ensemble_predictions))

    if {"random_forest", "hist_gradient_boosting"}.issubset(model_test_predictions):
        rf_hgb = model_test_predictions["random_forest"].copy()
        rf_hgb_class = sir_classification_metrics(
            test[TARGETS].to_numpy(),
            model_test_predictions["hist_gradient_boosting"][TARGETS].to_numpy(),
        )
        metrics.extend(_collect_metrics("rf_quant_hgb_class", test[TARGETS], rf_hgb))
        metrics.append({"model_name": "rf_quant_hgb_class", "target": "sir_class_hgb_decision", **rf_hgb_class})

    report(90, "Metriche", "Calcolo metriche finali e preparo il riepilogo.", work_completed=global_work_total)
    summary_file = model_path / "training_summary.json"
    summary = {
        "metrics": metrics,
        "artifacts": artifacts,
        "training": {
            "aggregated_rows": int(len(data)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "first_month": str(data["month"].min().date()),
            "last_month": str(data["month"].max().date()),
            "trained_until": str(train["month"].max().date()),
            "test_from": str(test["month"].min().date()),
            "test_until": str(test["month"].max().date()),
            "unique_months": int(data["month"].nunique()),
            "unique_pathogens": int(data["pathogen"].nunique()),
            "unique_antibiotics": int(data["antibiotic"].nunique()),
            "unique_laboratories": int(data["laboratory"].nunique()),
            "duration_seconds": round(time.perf_counter() - started_at, 2),
        },
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _collect_metrics(model_name: str, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> list[dict]:
    rows = []
    for target in TARGETS:
        metric = regression_metrics(y_true[target], y_pred[target])
        rows.append({"model_name": model_name, "target": target, **metric})

    class_metric = sir_classification_metrics(y_true[TARGETS].to_numpy(), y_pred[TARGETS].to_numpy())
    rows.append({"model_name": model_name, "target": "sir_class", **class_metric})
    return rows
