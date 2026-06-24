from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hierarchical_training import MIN_MONTHS, MIN_ROWS, MIN_SAMPLES
from .training import TARGETS, historical_baseline_predictions, temporal_train_test_split
from .validation import regression_metrics, sir_classification_metrics


def build_scientific_report(aggregated: pd.DataFrame, metrics_rows, model_folder: str | Path) -> dict:
    data = aggregated.copy()
    if data.empty:
        return _empty_report()
    data["month"] = pd.to_datetime(data["month"])

    train, test = temporal_train_test_split(data)
    baseline_pred = historical_baseline_predictions(train, test)

    return {
        "dataset": _dataset_summary(data),
        "coverage": _coverage_summary(data),
        "hierarchical": _hierarchical_summary(data, model_folder),
        "covariate_coverage": _covariate_coverage(data),
        "baseline_backtest": _baseline_backtest_summary(test, baseline_pred),
        "metrics": _metrics_table(metrics_rows),
        "top_pathogens": _top_table(data, "pathogen"),
        "top_antibiotics": _top_table(data, "antibiotic"),
        "lab_distribution": _top_table(data, "laboratory"),
        "monthly_samples": _monthly_samples(data),
    }


def report_tables(report: dict) -> dict[str, pd.DataFrame]:
    return {
        "metrics": pd.DataFrame(report.get("metrics", [])),
        "top_pathogens": pd.DataFrame(report.get("top_pathogens", [])),
        "top_antibiotics": pd.DataFrame(report.get("top_antibiotics", [])),
        "lab_distribution": pd.DataFrame(report.get("lab_distribution", [])),
        "monthly_samples": pd.DataFrame(report.get("monthly_samples", [])),
        "coverage": pd.DataFrame(report.get("coverage", [])),
    }


def _dataset_summary(data: pd.DataFrame) -> dict:
    return {
        "first_month": data["month"].min().strftime("%Y-%m"),
        "last_month": data["month"].max().strftime("%Y-%m"),
        "months": int(data["month"].nunique()),
        "aggregated_rows": int(len(data)),
        "samples": int(data["samples"].sum()),
        "pathogens": int(data["pathogen"].nunique()),
        "antibiotics": int(data["antibiotic"].nunique()),
        "laboratories": int(data["laboratory"].nunique()),
        "wards": int(data["ward"].replace("", pd.NA).dropna().nunique()),
        "pathogen_antibiotic_pairs": int(data[["pathogen", "antibiotic"]].drop_duplicates().shape[0]),
        "lab_combinations": int(data[["pathogen", "antibiotic", "laboratory"]].drop_duplicates().shape[0]),
        "exact_combinations": int(data[["pathogen", "antibiotic", "laboratory", "ward"]].drop_duplicates().shape[0]),
    }


def _coverage_summary(data: pd.DataFrame) -> list[dict]:
    specs = [
        ("regional", ["pathogen", "antibiotic"]),
        ("laboratory", ["pathogen", "antibiotic", "laboratory"]),
        ("exact", ["pathogen", "antibiotic", "laboratory", "ward"]),
    ]
    rows = []
    for scope, columns in specs:
        grouped = (
            data.groupby(columns, dropna=False)
            .agg(rows=("month", "size"), months=("month", "nunique"), samples=("samples", "sum"))
            .reset_index()
        )
        eligible = grouped[
            (grouped["rows"] >= MIN_ROWS)
            & (grouped["months"] >= MIN_MONTHS)
            & (grouped["samples"] >= MIN_SAMPLES)
        ]
        rows.append(
            {
                "scope": scope,
                "combinations": int(len(grouped)),
                "eligible": int(len(eligible)),
                "eligible_pct": float(len(eligible) / len(grouped) * 100) if len(grouped) else 0.0,
                "median_months": float(grouped["months"].median()) if len(grouped) else 0.0,
                "median_samples": float(grouped["samples"].median()) if len(grouped) else 0.0,
            }
        )
    return rows


def _hierarchical_summary(data: pd.DataFrame, model_folder: str | Path) -> dict:
    summary_path = Path(model_folder) / "training_summary.json"
    if not summary_path.exists():
        return {"trained_models": 0, "available": False}
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {"trained_models": 0, "available": False}
    hierarchical = summary.get("hierarchical_training", {})
    return {
        "available": bool(hierarchical),
        "trained_models": int(hierarchical.get("models_trained", 0) or 0),
        "thresholds": hierarchical.get("thresholds", {"min_months": MIN_MONTHS, "min_rows": MIN_ROWS, "min_samples": MIN_SAMPLES}),
    }


def _baseline_backtest_summary(test: pd.DataFrame, baseline_pred: pd.DataFrame) -> dict:
    rows = []
    for target in TARGETS:
        rows.append({"target": target, **regression_metrics(test[target], baseline_pred[target])})
    class_metrics = sir_classification_metrics(test[TARGETS].to_numpy(), baseline_pred[TARGETS].to_numpy())
    return {
        "test_rows": int(len(test)),
        "test_from": test["month"].min().strftime("%Y-%m") if not test.empty else "",
        "test_to": test["month"].max().strftime("%Y-%m") if not test.empty else "",
        "targets": rows,
        "classification": class_metrics,
    }


def _covariate_coverage(data: pd.DataFrame) -> dict | None:
    if "pct_icu" not in data.columns or data["pct_icu"].isna().all():
        return None
    icu_valid = data["pct_icu"].notna()
    inpatient_valid = data.get("pct_inpatient", pd.Series(dtype=float)).notna() if "pct_inpatient" in data.columns else pd.Series(False, index=data.index)
    combos_with = int(
        data[icu_valid | inpatient_valid]
        .groupby(["pathogen", "antibiotic", "laboratory"])
        .ngroups
    )
    return {
        "pct_icu_mean": float(data.loc[icu_valid, "pct_icu"].mean() * 100) if icu_valid.any() else 0.0,
        "pct_inpatient_mean": float(data.loc[data["pct_inpatient"].notna(), "pct_inpatient"].mean() * 100) if "pct_inpatient" in data.columns and data["pct_inpatient"].notna().any() else 0.0,
        "pct_icu_coverage": int(round(icu_valid.mean() * 100)),
        "combinations_with_covariates": combos_with,
    }


def _metrics_table(metrics_rows) -> list[dict]:
    return [
        {
            "model_name": row.model_name,
            "target": row.target,
            "mae": row.mae,
            "rmse": row.rmse,
            "mape": row.mape,
            "mase": row.mase,
            "rmse_arima": row.rmse_arima,
            "accuracy": row.accuracy,
            "f1_macro": row.f1_macro,
        }
        for row in metrics_rows
    ]


def _top_table(data: pd.DataFrame, column: str, limit: int = 20) -> list[dict]:
    grouped = (
        data.groupby(column, dropna=False)
        .agg(samples=("samples", "sum"), rows=("month", "size"), months=("month", "nunique"))
        .sort_values("samples", ascending=False)
        .head(limit)
        .reset_index()
    )
    return grouped.to_dict(orient="records")


def _monthly_samples(data: pd.DataFrame) -> list[dict]:
    grouped = data.groupby(data["month"].dt.to_period("M")).agg(samples=("samples", "sum"), rows=("month", "size")).reset_index()
    grouped["month"] = grouped["month"].astype(str)
    return grouped.to_dict(orient="records")


def _empty_report() -> dict:
    return {
        "dataset": {},
        "coverage": [],
        "hierarchical": {"trained_models": 0, "available": False},
        "covariate_coverage": None,
        "baseline_backtest": {},
        "metrics": [],
        "top_pathogens": [],
        "top_antibiotics": [],
        "lab_distribution": [],
        "monthly_samples": [],
    }
