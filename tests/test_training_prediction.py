import pandas as pd

from app.services.prediction import predict_sir
from app.services.training import train_all_models


def _sample_aggregated():
    rows = []
    for idx, month in enumerate(pd.date_range("2023-01-31", periods=14, freq="ME")):
        rows.append(
            {
                "month": month,
                "pathogen": "KLEPNE",
                "antibiotic": "AMK",
                "laboratory": "IT161",
                "ward": "",
                "samples": 30 + idx,
                "sensitive_count": 15,
                "intermediate_count": 5,
                "resistant_count": 10,
                "sensitive_pct": 50 + idx * 0.2,
                "intermediate_pct": 15,
                "resistant_pct": 35 - idx * 0.2,
            }
        )
    return pd.DataFrame(rows)


def test_train_all_models_creates_metrics(tmp_path):
    summary = train_all_models(_sample_aggregated(), tmp_path)

    assert summary["metrics"]
    assert (tmp_path / "hist_gradient_boosting.joblib").exists()
    assert (tmp_path / "random_forest.joblib").exists()


def test_predict_sir_returns_normalized_percentages(tmp_path):
    data = _sample_aggregated()
    train_all_models(data, tmp_path)

    prediction = predict_sir(data, tmp_path, "KLEPNE", "AMK", "IT161", "2024-06")

    total = prediction["sensitive_pct"] + prediction["intermediate_pct"] + prediction["resistant_pct"]
    assert round(total, 5) == 100
    assert prediction["reliability"] in {"alta", "media", "bassa"}
