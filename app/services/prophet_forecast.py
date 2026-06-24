from __future__ import annotations

import numpy as np
import pandas as pd

from .training import TARGETS

MIN_PROPHET_MONTHS = 12


def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401
    except Exception:
        return False
    return True


def forecast_sir_with_prophet(history: pd.DataFrame, prediction_month: str) -> dict[str, float] | None:
    if history.empty or len(history) < MIN_PROPHET_MONTHS:
        return None
    try:
        from prophet import Prophet
    except Exception:
        return None

    series = history.copy()
    series["month"] = pd.to_datetime(series["month"])
    monthly = (
        series.groupby("month", dropna=False)[TARGETS]
        .mean()
        .reset_index()
        .sort_values("month")
    )
    if len(monthly) < MIN_PROPHET_MONTHS:
        return None

    target_month = pd.to_datetime(prediction_month).to_period("M").to_timestamp("M")
    raw_values = {}
    for target in TARGETS:
        frame = monthly[["month", target]].rename(columns={"month": "ds", target: "y"})
        model = Prophet(
            yearly_seasonality="auto",
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.8,
        )
        model.fit(frame)
        future = pd.DataFrame({"ds": [target_month]})
        forecast = model.predict(future)
        raw_values[target] = float(np.clip(forecast.loc[0, "yhat"], 0, 100))

    return raw_values
