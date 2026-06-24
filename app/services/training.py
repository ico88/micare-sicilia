from __future__ import annotations

import json
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


def train_all_models(aggregated: pd.DataFrame, model_dir: str | Path) -> dict:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    data = aggregated.copy()
    data["month"] = pd.to_datetime(data["month"])
    train, test = temporal_train_test_split(data)
    metrics: list[dict] = []
    artifacts: list[dict] = []

    baseline_pred = historical_baseline_predictions(train, test)
    baseline_metric = _collect_metrics("baseline_historical", test[TARGETS], baseline_pred)
    metrics.extend(baseline_metric)

    for model_name in ["hist_gradient_boosting", "random_forest"]:
        target_models = {}
        test_predictions = pd.DataFrame(index=test.index)
        x_train = make_features(train)
        x_test = make_features(test)

        for target in TARGETS:
            model = build_model(model_name)
            model.fit(x_train, train[target])
            pred = np.clip(model.predict(x_test), 0, 100)
            test_predictions[target] = pred
            target_models[target] = model

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

    summary_file = model_path / "training_summary.json"
    summary = {"metrics": metrics, "artifacts": artifacts}
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
