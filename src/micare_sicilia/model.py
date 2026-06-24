from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

from .config import (
    FORECAST_PERIODS_MONTHS,
    MAX_TRAINING_DATE,
    MIN_MONTHS_PER_COMBINATION,
    TARGETS,
    VARIANCE_THRESHOLD,
)


@dataclass(frozen=True)
class TrainingSummary:
    attempted_combinations: int
    trained_resistant_models: int
    output_dir: Path


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    if len(y_train) < 2:
        return np.nan

    baseline_mae = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    if baseline_mae == 0:
        return np.nan

    model_mae = np.mean(np.abs(y_true - y_pred))
    return float(model_mae / baseline_mae)


def _new_prophet_model() -> Prophet:
    return Prophet(weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)


def process_combination(
    combination: str,
    data: pd.DataFrame,
    variance_threshold: float = VARIANCE_THRESHOLD,
    max_training_date: pd.Timestamp = MAX_TRAINING_DATE,
) -> tuple[str, dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    local_forecasts: dict[str, pd.DataFrame] = {}
    local_metrics: dict[str, dict[str, float]] = {"rmse": {}, "mape": {}, "mase": {}}

    try:
        max_variance = data[list(TARGETS)].var().max()
        if max_variance > variance_threshold:
            return combination, {}, {}
    except Exception:
        return combination, {}, {}

    data = data[data["data"] <= max_training_date].copy()
    cv_dates = pd.to_datetime(["2023-01-31", "2023-06-30", "2023-12-31", "2024-06-30"])
    cv_dates = cv_dates[cv_dates < max_training_date]

    for target in TARGETS:
        prophet_data = data[["data", target]].rename(columns={"data": "ds", target: "y"})
        prophet_data["cap"] = 100.0
        prophet_data["floor"] = 0.0

        y_data = prophet_data["y"].dropna()
        if y_data.empty or len(y_data) < MIN_MONTHS_PER_COMBINATION:
            continue

        final_model = _new_prophet_model()
        final_model.fit(prophet_data)

        future = final_model.make_future_dataframe(periods=FORECAST_PERIODS_MONTHS, freq="ME")
        future["cap"] = 100.0
        future["floor"] = 0.0
        forecast = final_model.predict(future)
        forecast["yhat"] = np.clip(forecast["yhat"], 0, 100)
        local_forecasts[target] = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        rmse_values: list[float] = []
        mape_values: list[float] = []
        mase_values: list[float] = []

        for end_date in cv_dates:
            train = prophet_data[prophet_data["ds"] <= end_date].copy()
            test = prophet_data[
                (prophet_data["ds"] > end_date)
                & (prophet_data["ds"] <= end_date + pd.DateOffset(months=6))
            ].copy()

            if len(train) < 2 or test.empty:
                continue

            cv_model = _new_prophet_model()
            cv_model.fit(train)
            cv_forecast = cv_model.predict(test)

            y_pred = np.clip(cv_forecast["yhat"].values, 0, 100)
            y_true = test["y"].values
            y_train = train["y"].values

            rmse_values.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
            y_true_stable = np.where(y_true == 0, 0.0001, y_true)
            mape_values.append(float(np.mean(np.abs((y_true - y_pred) / y_true_stable)) * 100))
            mase = calculate_mase(y_true, y_pred, y_train)
            if not np.isnan(mase):
                mase_values.append(mase)

        if rmse_values:
            local_metrics["rmse"][target] = float(np.mean(rmse_values))
        if mape_values:
            local_metrics["mape"][target] = float(np.mean(mape_values))
        if mase_values:
            local_metrics["mase"][target] = float(np.mean(mase_values))

    return combination, local_forecasts, local_metrics


def train_forecasts(
    aggregated: pd.DataFrame,
    output_dir: str | Path,
    n_jobs: int = -1,
) -> TrainingSummary:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filtered = aggregated[aggregated["data"].dt.year >= 2019].copy()
    counts = filtered.groupby("combinazione_unica").size()
    valid_combinations = counts[counts >= MIN_MONTHS_PER_COMBINATION].index
    valid_data = filtered[filtered["combinazione_unica"].isin(valid_combinations)].copy()
    combinations = sorted(valid_data["combinazione_unica"].unique())

    forecasts = {target: {} for target in TARGETS}
    metrics = {target: {"rmse": {}, "mape": {}, "mase": {}} for target in TARGETS}

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_combination)(
            combination,
            valid_data[valid_data["combinazione_unica"] == combination].copy(),
        )
        for combination in tqdm(combinations, desc="Addestramento e CV")
    )

    for combination, local_forecasts, local_metrics in results:
        for target in TARGETS:
            if target in local_forecasts:
                forecasts[target][combination] = local_forecasts[target]
            for metric_name, target_metrics in local_metrics.items():
                if target in target_metrics:
                    metrics[target][metric_name][combination] = target_metrics[target]

    joblib.dump(metrics, output_path / "cv_metrics_results.pkl")
    joblib.dump(forecasts["resistenti"], output_path / "previsioni_resistenti.pkl")
    joblib.dump(forecasts["intermedi"], output_path / "previsioni_intermedi.pkl")
    joblib.dump(forecasts["sensibili"], output_path / "previsioni_sensibili.pkl")
    aggregated.to_pickle(output_path / "df_aggregato.pkl")

    return TrainingSummary(
        attempted_combinations=len(combinations),
        trained_resistant_models=len(forecasts["resistenti"]),
        output_dir=output_path,
    )


def load_forecasts(output_dir: str | Path) -> dict[str, dict[str, pd.DataFrame]]:
    output_path = Path(output_dir)
    return {
        "resistenti": joblib.load(output_path / "previsioni_resistenti.pkl"),
        "intermedi": joblib.load(output_path / "previsioni_intermedi.pkl"),
        "sensibili": joblib.load(output_path / "previsioni_sensibili.pkl"),
    }


def predict_percentages(
    output_dir: str | Path,
    pathogen: str,
    laboratory: str,
    antibiotic: str,
    year: int,
    month: int,
) -> dict[str, float]:
    forecasts = load_forecasts(output_dir)
    combination = f"{pathogen}_{laboratory}_{antibiotic}"
    date = pd.to_datetime(pd.Period(f"{year}-{month}", freq="M").end_time.date())

    raw_values = {}
    for target in TARGETS:
        if combination not in forecasts[target]:
            raise KeyError(f"Nessuna previsione trovata per {combination}.")
        match = forecasts[target][combination][forecasts[target][combination]["ds"] == date]
        if match.empty:
            raise KeyError(f"Nessuna previsione trovata per {combination} alla data {date.date()}.")
        raw_values[target] = max(0.0, float(match["yhat"].iloc[0]))

    total = sum(raw_values.values())
    if total <= 0:
        return {target: 0.0 for target in TARGETS}

    return {target: value * 100 / total for target, value in raw_values.items()}
