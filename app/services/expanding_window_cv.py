"""
Expanding-window cross-validation by year.

For a dataset spanning years Y0..Yn the folds are:
  Fold 1: train Y0..Yn-2, predict Yn-1..Yn
  Fold 2: train Y0..Yn-1, predict Yn
  (Fold 3 = final model trained on all data, no hold-out)

Each fold produces per-combination metrics (MAE, accuracy) that are
aggregated into a reliability score in [0, 1] shown in the validation
report alongside a tier label (alta / media / bassa / insufficiente).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .feature_engineering import make_features
from .training import TARGETS, build_model
from .validation import regression_metrics

CV_SUMMARY_FILE = "expanding_cv_summary.json"
CV_COMBINATION_FILE = "expanding_cv_combinations.json"

# Thresholds for combination-level reliability tier
_TIER_THRESHOLDS = [
    (0.80, "alta"),
    (0.60, "media"),
    (0.40, "bassa"),
    (0.0, "insufficiente"),
]


def run_expanding_window_cv(
    aggregated: pd.DataFrame,
    output_dir: str | Path,
    progress_callback=None,
) -> dict:
    started_at = time.perf_counter()
    data = aggregated.copy()
    data["month"] = pd.to_datetime(data["month"])
    data["year"] = data["month"].dt.year

    years = sorted(data["year"].unique())
    if len(years) < 3:
        summary = _empty_summary()
        summary["error"] = "Servono almeno 3 anni di dati per la validazione expanding-window."
        _save(summary, [], output_dir)
        return summary

    # Build folds: each fold trains on years[0..cutoff] and predicts years[cutoff+1..]
    # We generate folds such that the test window is at least 1 year
    folds = []
    for cutoff_idx in range(len(years) - 2, len(years) - 1):
        train_years = years[: cutoff_idx + 1]
        test_years = years[cutoff_idx + 1 :]
        folds.append(
            {
                "label": f"Train {train_years[0]}–{train_years[-1]} → Test {test_years[0]}–{test_years[-1]}",
                "train_until": train_years[-1],
                "test_from": test_years[0],
                "test_until": test_years[-1],
            }
        )
    # Second fold: train all but last year, predict last year only
    folds.append(
        {
            "label": f"Train {years[0]}–{years[-2]} → Test {years[-1]}",
            "train_until": years[-2],
            "test_from": years[-1],
            "test_until": years[-1],
        }
    )

    total_steps = len(folds) * 2  # 2 models per fold
    step = 0
    all_rows: list[dict] = []

    for fold_index, fold in enumerate(folds, start=1):
        train = data[data["year"] <= fold["train_until"]].copy()
        test = data[
            (data["year"] >= fold["test_from"]) & (data["year"] <= fold["test_until"])
        ].copy()

        if train.empty or test.empty:
            step += 2
            continue

        for model_name in ["hist_gradient_boosting", "random_forest"]:
            step += 1
            if progress_callback is not None:
                pct = 10 + int(step / total_steps * 80)
                progress_callback(
                    pct,
                    "Expanding-window CV",
                    f"Fold {fold_index}/{len(folds)} · {model_name}",
                )

            x_train = make_features(train)
            x_test = make_features(test)
            pred = pd.DataFrame(index=test.index)
            for target in TARGETS:
                m = build_model(model_name)
                m.fit(x_train, train[target])
                pred[target] = np.clip(m.predict(x_test), 0, 100)

            # Global fold metrics
            y_true_mat = test[TARGETS].to_numpy(dtype=float)
            y_pred_mat = pred[TARGETS].to_numpy(dtype=float)
            class_true = np.argmax(y_true_mat, axis=1)
            class_pred_arr = np.argmax(y_pred_mat, axis=1)

            fold_row: dict = {
                "fold": fold["label"],
                "model_name": model_name,
                "train_years": f"{years[0]}–{fold['train_until']}",
                "test_years": f"{fold['test_from']}–{fold['test_until']}",
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "accuracy": float(accuracy_score(class_true, class_pred_arr)),
                "f1_macro": float(
                    f1_score(class_true, class_pred_arr, average="macro", zero_division=0)
                ),
            }
            for target, col in zip(TARGETS, ["sensitive_pct", "intermediate_pct", "resistant_pct"]):
                m = regression_metrics(test[col], pred[col])
                fold_row[f"{col}_mae"] = m["mae"]
                fold_row[f"{col}_rmse"] = m["rmse"]

            # Per-combination metrics
            combo_cols = ["pathogen", "antibiotic", "laboratory"]
            for combo_key, combo_group in test.groupby(combo_cols, dropna=False):
                if len(combo_group) < 2:
                    continue
                combo_pred = pred.loc[combo_group.index]
                ct = np.argmax(combo_group[TARGETS].to_numpy(dtype=float), axis=1)
                cp = np.argmax(combo_pred[TARGETS].to_numpy(dtype=float), axis=1)
                acc = float(accuracy_score(ct, cp))
                mae_r = float(
                    np.mean(np.abs(combo_group["resistant_pct"].to_numpy(dtype=float) - combo_pred["resistant_pct"].to_numpy(dtype=float)))
                )
                pathogen, antibiotic, laboratory = combo_key
                all_rows.append(
                    {
                        "fold": fold["label"],
                        "model_name": model_name,
                        "train_until": fold["train_until"],
                        "test_years": f"{fold['test_from']}–{fold['test_until']}",
                        "pathogen": pathogen,
                        "antibiotic": antibiotic,
                        "laboratory": laboratory,
                        "n_test": int(len(combo_group)),
                        "accuracy": acc,
                        "resistant_pct_mae": mae_r,
                    }
                )

            fold_row["_type"] = "fold"
            all_rows.append(fold_row)

    if progress_callback is not None:
        progress_callback(92, "Expanding-window CV", "Calcolo reliability score combinazioni…")

    combo_reliability = _compute_combination_reliability(all_rows)
    fold_summary = [r for r in all_rows if r.get("_type") == "fold"]
    for r in fold_summary:
        del r["_type"]

    summary = {
        "available": True,
        "folds": fold_summary,
        "combination_count": len(combo_reliability),
        "high_reliability": sum(1 for c in combo_reliability if c["tier"] == "alta"),
        "medium_reliability": sum(1 for c in combo_reliability if c["tier"] == "media"),
        "low_reliability": sum(1 for c in combo_reliability if c["tier"] == "bassa"),
        "insufficient_reliability": sum(1 for c in combo_reliability if c["tier"] == "insufficiente"),
        "duration_seconds": round(time.perf_counter() - started_at, 2),
        "years_used": [int(y) for y in years],
    }
    _save(summary, combo_reliability, output_dir)

    if progress_callback is not None:
        progress_callback(100, "Expanding-window CV", "Completato.")

    return summary


def load_expanding_cv(output_dir: str | Path) -> dict:
    path = Path(output_dir) / CV_SUMMARY_FILE
    if not path.exists():
        return _empty_summary()
    try:
        return json.loads(path.read_text()) | {"available": True}
    except json.JSONDecodeError:
        return _empty_summary()


def load_combination_reliability(output_dir: str | Path) -> list[dict]:
    path = Path(output_dir) / CV_COMBINATION_FILE
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def _compute_combination_reliability(rows: list[dict]) -> list[dict]:
    combo_rows = [r for r in rows if "pathogen" in r and r.get("_type") != "fold"]
    if not combo_rows:
        return []

    df = pd.DataFrame(combo_rows)
    grouped = (
        df.groupby(["pathogen", "antibiotic", "laboratory"], dropna=False)
        .agg(
            folds_evaluated=("fold", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_resistant_mae=("resistant_pct_mae", "mean"),
            n_test_total=("n_test", "sum"),
        )
        .reset_index()
    )

    # Reliability score: weighted combination of accuracy (70%) and inverse normalised MAE (30%)
    max_mae = grouped["mean_resistant_mae"].quantile(0.95).clip(min=1e-6)
    grouped["mae_score"] = (1 - (grouped["mean_resistant_mae"] / max_mae).clip(0, 1))
    grouped["reliability_score"] = (0.70 * grouped["mean_accuracy"] + 0.30 * grouped["mae_score"]).clip(0, 1)
    grouped["tier"] = grouped["reliability_score"].apply(_tier)

    return grouped.sort_values("reliability_score", ascending=False).to_dict(orient="records")


def _tier(score: float) -> str:
    for threshold, label in _TIER_THRESHOLDS:
        if score >= threshold:
            return label
    return "insufficiente"


def _save(summary: dict, combinations: list[dict], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / CV_SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (path / CV_COMBINATION_FILE).write_text(json.dumps(combinations, indent=2), encoding="utf-8")


def _empty_summary() -> dict:
    return {
        "available": False,
        "folds": [],
        "combination_count": 0,
        "high_reliability": 0,
        "medium_reliability": 0,
        "low_reliability": 0,
        "insufficient_reliability": 0,
        "duration_seconds": 0,
        "years_used": [],
    }
