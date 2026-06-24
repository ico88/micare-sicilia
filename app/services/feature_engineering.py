from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = ["year", "month_num", "quarter", "pathogen", "antibiotic", "laboratory", "ward", "samples"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["month"] = pd.to_datetime(features["month"])
    features["year"] = features["month"].dt.year
    features["month_num"] = features["month"].dt.month
    features["quarter"] = features["month"].dt.quarter
    features["ward"] = features["ward"].fillna("")
    return features[FEATURE_COLUMNS]


def next_month_feature(
    month: pd.Timestamp,
    pathogen: str,
    antibiotic: str,
    laboratory: str,
    ward: str | None,
    samples: int,
) -> pd.DataFrame:
    month = pd.to_datetime(month).to_period("M").to_timestamp("M")
    return make_features(
        pd.DataFrame(
            [
                {
                    "month": month,
                    "pathogen": pathogen,
                    "antibiotic": antibiotic,
                    "laboratory": laboratory,
                    "ward": ward or "",
                    "samples": samples,
                }
            ]
        )
    )
