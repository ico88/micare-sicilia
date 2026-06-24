from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .training import TARGETS, build_model, historical_baseline_predictions
from .validation import regression_metrics

BACKTEST_SUMMARY = "rolling_backtest_summary.json"
BACKTEST_PREDICTIONS = "rolling_backtest_predictions.csv"


def run_rolling_backtest(aggregated: pd.DataFrame, output_dir: str | Path, months: int = 6, progress_callback=None) -> dict:
    started_at = time.perf_counter()
    data = aggregated.copy()
    data["month"] = pd.to_datetime(data["month"])
    unique_months = sorted(data["month"].dropna().unique())
    test_months = unique_months[-months:] if len(unique_months) > months else unique_months[1:]
    rows = []

    if len(test_months) == 0:
        summary = _empty_summary()
        _save_outputs(summary, pd.DataFrame(rows), output_dir)
        return summary

    for index, test_month in enumerate(test_months, start=1):
        if progress_callback is not None:
            progress_callback(10 + int(index / len(test_months) * 80), "Backtest rolling", f"Valido mese {pd.Timestamp(test_month).strftime('%Y-%m')} ({index}/{len(test_months)})")

        train = data[data["month"] < test_month].copy()
        test = data[data["month"] == test_month].copy()
        if train.empty or test.empty:
            continue

        baseline = historical_baseline_predictions(train, test)
        rows.extend(_prediction_rows("baseline_historical", test, baseline, "global"))

        hierarchical = _hierarchical_baseline_predictions(train, test)
        rows.extend(_prediction_rows("hierarchical_baseline", test, hierarchical, "adaptive"))

        hgb = _global_model_predictions(train, test, "hist_gradient_boosting")
        rows.extend(_prediction_rows("hist_gradient_boosting", test, hgb, "global"))

        rf = _global_model_predictions(train, test, "random_forest")
        rows.extend(_prediction_rows("random_forest", test, rf, "global"))

        ensemble = 0.60 * rf + 0.40 * hgb
        rows.extend(_prediction_rows("ensemble_rf60_hgb40", test, ensemble, "global"))
        rows.extend(_prediction_rows("rf_quant_hgb_class", test, rf, "rf_pct_hgb_class", class_pred=hgb))

    predictions = pd.DataFrame(rows)
    summary = _summarize_predictions(predictions)
    summary["months_tested"] = len(test_months)
    summary["test_from"] = pd.Timestamp(test_months[0]).strftime("%Y-%m")
    summary["test_to"] = pd.Timestamp(test_months[-1]).strftime("%Y-%m")
    summary["duration_seconds"] = round(time.perf_counter() - started_at, 2)
    _save_outputs(summary, predictions, output_dir)
    return summary


def load_rolling_backtest(output_dir: str | Path) -> dict:
    path = Path(output_dir) / BACKTEST_SUMMARY
    if not path.exists():
        return _empty_summary() | {"available": False}
    try:
        summary = json.loads(path.read_text())
    except json.JSONDecodeError:
        return _empty_summary() | {"available": False}
    summary["available"] = True
    return summary


def _global_model_predictions(train: pd.DataFrame, test: pd.DataFrame, model_name: str) -> pd.DataFrame:
    from .feature_engineering import make_features

    x_train = make_features(train)
    x_test = make_features(test)
    output = pd.DataFrame(index=test.index)
    for target in TARGETS:
        model = build_model(model_name)
        model.fit(x_train, train[target])
        output[target] = np.clip(model.predict(x_test), 0, 100)
    return output


def _hierarchical_baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    global_mean = train[TARGETS].mean().fillna(0)
    exact = train.groupby(["pathogen", "antibiotic", "laboratory", "ward"], dropna=False)[TARGETS].mean()
    lab = train.groupby(["pathogen", "antibiotic", "laboratory"], dropna=False)[TARGETS].mean()
    regional = train.groupby(["pathogen", "antibiotic"], dropna=False)[TARGETS].mean()
    predictions = []
    for _, row in test.iterrows():
        exact_key = (row["pathogen"], row["antibiotic"], row["laboratory"], row.get("ward") or "")
        lab_key = (row["pathogen"], row["antibiotic"], row["laboratory"])
        regional_key = (row["pathogen"], row["antibiotic"])
        if exact_key in exact.index:
            values = exact.loc[exact_key]
        elif lab_key in lab.index:
            values = lab.loc[lab_key]
        elif regional_key in regional.index:
            values = regional.loc[regional_key]
        else:
            values = global_mean
        predictions.append(values.to_dict())
    return pd.DataFrame(predictions, index=test.index).fillna(0)


def _prediction_rows(model_name: str, test: pd.DataFrame, pred: pd.DataFrame, scope: str, class_pred: pd.DataFrame | None = None) -> list[dict]:
    rows = []
    class_pred = pred if class_pred is None else class_pred
    for idx, actual in test.iterrows():
        predicted = pred.loc[idx]
        class_values = class_pred.loc[idx]
        rows.append(
            {
                "model_name": model_name,
                "scope": scope,
                "month": pd.Timestamp(actual["month"]).strftime("%Y-%m"),
                "pathogen": actual["pathogen"],
                "antibiotic": actual["antibiotic"],
                "laboratory": actual["laboratory"],
                "ward": actual.get("ward") or "",
                "samples": actual["samples"],
                "actual_sensitive_pct": actual["sensitive_pct"],
                "actual_intermediate_pct": actual["intermediate_pct"],
                "actual_resistant_pct": actual["resistant_pct"],
                "pred_sensitive_pct": predicted["sensitive_pct"],
                "pred_intermediate_pct": predicted["intermediate_pct"],
                "pred_resistant_pct": predicted["resistant_pct"],
                "actual_class": _class_name(actual[TARGETS].to_numpy(dtype=float)),
                "pred_class": _class_name(class_values[TARGETS].to_numpy(dtype=float)),
            }
        )
    return rows


def _summarize_predictions(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return _empty_summary()
    rows = []
    for model_name, group in predictions.groupby("model_name"):
        y_true = group[[f"actual_{name}" for name in TARGETS]].to_numpy(dtype=float)
        y_pred = group[[f"pred_{name}" for name in TARGETS]].to_numpy(dtype=float)
        class_true = np.argmax(y_true, axis=1)
        class_pred = np.argmax(y_pred, axis=1)
        row = {
            "model_name": model_name,
            "rows": int(len(group)),
            "accuracy": float(accuracy_score(class_true, class_pred)),
            "f1_macro": float(f1_score(class_true, class_pred, average="macro", zero_division=0)),
        }
        for target, actual_col, pred_col in [
            ("sensitive_pct", "actual_sensitive_pct", "pred_sensitive_pct"),
            ("intermediate_pct", "actual_intermediate_pct", "pred_intermediate_pct"),
            ("resistant_pct", "actual_resistant_pct", "pred_resistant_pct"),
        ]:
            metrics = regression_metrics(group[actual_col], group[pred_col])
            row[f"{target}_mae"] = metrics["mae"]
            row[f"{target}_rmse"] = metrics["rmse"]
        rows.append(row)
    return {"available": True, "models": sorted(rows, key=lambda item: item["model_name"])}


def _save_outputs(summary: dict, predictions: pd.DataFrame, output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / BACKTEST_SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    predictions.to_csv(path / BACKTEST_PREDICTIONS, index=False)


def _empty_summary() -> dict:
    return {"available": False, "models": [], "months_tested": 0, "test_from": "", "test_to": "", "duration_seconds": 0}


def _class_name(values: np.ndarray) -> str:
    return ["S", "I", "R"][int(np.argmax(values))]
