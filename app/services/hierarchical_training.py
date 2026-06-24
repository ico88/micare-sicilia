from __future__ import annotations

import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .feature_engineering import make_features
from .training import TARGETS, build_model, temporal_train_test_split

MIN_MONTHS = 24
MIN_ROWS = 24
MIN_SAMPLES = 100
MAX_MODELS_PER_SCOPE = 250
ARTIFACT_NAME = "hierarchical_hist_gradient_boosting.joblib"


def train_hierarchical_models(aggregated: pd.DataFrame, model_dir: str | Path, progress_callback=None) -> dict:
    started_at = time.perf_counter()
    data = aggregated.copy()
    data["month"] = pd.to_datetime(data["month"])
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    bundles = {}
    registry = []
    scope_specs = [
        ("regional", ["pathogen", "antibiotic"]),
        ("laboratory", ["pathogen", "antibiotic", "laboratory"]),
        ("exact", ["pathogen", "antibiotic", "laboratory", "ward"]),
    ]

    candidates = []
    for scope, columns in scope_specs:
        grouped = (
            data.groupby(columns, dropna=False)
            .agg(rows=("month", "size"), months=("month", "nunique"), samples=("samples", "sum"))
            .reset_index()
        )
        eligible = grouped[
            (grouped["rows"] >= MIN_ROWS)
            & (grouped["months"] >= MIN_MONTHS)
            & (grouped["samples"] >= MIN_SAMPLES)
        ].sort_values(["samples", "months", "rows"], ascending=False).head(MAX_MODELS_PER_SCOPE)
        for _, row in eligible.iterrows():
            criteria = {column: _clean_value(row[column]) for column in columns}
            candidates.append((scope, columns, criteria, int(row["samples"]), int(row["months"]), int(row["rows"])))

    total = max(len(candidates), 1)
    work_total = max(len(candidates) * len(TARGETS), 1)
    work_completed = 0
    for index, (scope, columns, criteria, samples, months, rows) in enumerate(candidates, start=1):
        if progress_callback is not None:
            progress = 72 + min(16, int(work_completed / work_total * 16))
            progress_callback(
                progress,
                "Modelli gerarchici",
                f"Preparo {scope} {index}/{len(candidates)}",
                work_completed=work_completed,
                work_total=work_total,
                work_label="Modelli gerarchici",
            )

        subset = _subset(data, criteria).sort_values("month")
        train, test = temporal_train_test_split(subset, test_months=6)
        if train.empty or test.empty:
            continue

        target_models = {}
        test_predictions = pd.DataFrame(index=test.index)
        x_train = make_features(train)
        x_test = make_features(test)
        rmses = []
        for target in TARGETS:
            if progress_callback is not None:
                progress = 72 + min(16, int(work_completed / work_total * 16))
                progress_callback(
                    progress,
                    "Modelli gerarchici",
                    f"Addestro {scope} {index}/{len(candidates)} · target {target}",
                    work_completed=work_completed,
                    work_total=work_total,
                    work_label="Modelli gerarchici",
                )
            model = build_model("hist_gradient_boosting")
            model.fit(x_train, train[target])
            pred = np.clip(model.predict(x_test), 0, 100)
            test_predictions[target] = pred
            target_models[target] = model
            rmses.append(float(np.sqrt(mean_squared_error(test[target], pred))))
            work_completed += 1
            if progress_callback is not None:
                progress = 72 + min(16, int(work_completed / work_total * 16))
                progress_callback(
                    progress,
                    "Modelli gerarchici",
                    f"Completati {work_completed}/{work_total} target gerarchici",
                    work_completed=work_completed,
                    work_total=work_total,
                    work_label="Modelli gerarchici",
                )

        key = _model_key(scope, criteria)
        metadata = {
            "scope": scope,
            "criteria": criteria,
            "rows": rows,
            "months": months,
            "samples": samples,
            "trained_until": str(train["month"].max().date()),
            "test_from": str(test["month"].min().date()),
            "test_until": str(test["month"].max().date()),
            "rmse_mean": float(np.mean(rmses)),
        }
        bundles[key] = {"models": target_models, "metadata": metadata}
        registry.append(metadata | {"key": key})

    artifact_file = model_path / ARTIFACT_NAME
    joblib.dump({"model_name": "hierarchical_hist_gradient_boosting", "bundles": bundles, "registry": registry}, artifact_file)
    return {
        "artifact_path": str(artifact_file),
        "models_trained": len(bundles),
        "registry": registry,
        "duration_seconds": round(time.perf_counter() - started_at, 2),
        "thresholds": {"min_months": MIN_MONTHS, "min_rows": MIN_ROWS, "min_samples": MIN_SAMPLES},
    }


def find_hierarchical_bundle(model_dir: str | Path, pathogen: str, antibiotic: str, laboratory: str, ward: str | None):
    artifact = Path(model_dir) / ARTIFACT_NAME
    if not artifact.exists():
        return None, None
    bundle = joblib.load(artifact)
    ward = ward or ""
    candidates = []
    if laboratory and ward:
        candidates.append(("exact", {"pathogen": pathogen, "antibiotic": antibiotic, "laboratory": laboratory, "ward": ward}))
    if laboratory:
        candidates.append(("laboratory", {"pathogen": pathogen, "antibiotic": antibiotic, "laboratory": laboratory}))
    candidates.append(("regional", {"pathogen": pathogen, "antibiotic": antibiotic}))

    for scope, criteria in candidates:
        key = _model_key(scope, criteria)
        scoped = bundle.get("bundles", {}).get(key)
        if scoped is not None:
            return scoped, scoped["metadata"]
    return None, None


def _subset(data: pd.DataFrame, criteria: dict[str, str]) -> pd.DataFrame:
    mask = pd.Series(True, index=data.index)
    for column, value in criteria.items():
        mask &= data[column].fillna("").astype(str).str.upper() == value
    return data[mask].copy()


def _model_key(scope: str, criteria: dict[str, str]) -> str:
    parts = [scope] + [f"{key}={_slug(value)}" for key, value in sorted(criteria.items())]
    return "|".join(parts)


def _slug(value: object) -> str:
    text = _clean_value(value)
    return re.sub(r"[^A-Z0-9_.-]+", "_", text) or "ALL"


def _clean_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()
