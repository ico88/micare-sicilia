import pandas as pd

from app.services.preprocessing import aggregate_observations, normalize_uploaded_dataframe


def test_normalize_wide_mic_dataframe():
    raw = pd.DataFrame(
        {
            "DATA_PRELIEVO": ["01/01/2024", "15/01/2024", "20/02/2024"],
            "MICROORGANISMO": [" klepne ", "KLEPNE", "ESCCOL"],
            "LABORATORIO": ["it161", "IT161", "IT170"],
            "AMK_QUALITATIVO": ["R", "S", "I"],
        }
    )

    normalized = normalize_uploaded_dataframe(raw)

    assert list(normalized["pathogen"].unique()) == ["KLEPNE", "ESCCOL"]
    assert set(normalized["result"]) == {"R", "S", "I"}
    assert normalized["resistant_count"].sum() == 1


def test_aggregate_observations_monthly_percentages():
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "pathogen": ["KLEPNE", "KLEPNE", "KLEPNE"],
            "antibiotic": ["AMK", "AMK", "AMK"],
            "laboratory": ["IT161", "IT161", "IT161"],
            "ward": ["", "", ""],
            "result": ["R", "S", "S"],
            "samples": [1, 1, 1],
            "sensitive_count": [0, 1, 1],
            "intermediate_count": [0, 0, 0],
            "resistant_count": [1, 0, 0],
        }
    )

    aggregated = aggregate_observations(observations)

    assert len(aggregated) == 1
    assert aggregated.loc[0, "samples"] == 3
    assert round(aggregated.loc[0, "sensitive_pct"], 2) == 66.67
    assert round(aggregated.loc[0, "resistant_pct"], 2) == 33.33
