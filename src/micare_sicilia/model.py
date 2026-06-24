from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm.auto import tqdm

from .config import (
    FORECAST_PERIODS_MONTHS,
    MAX_TRAINING_DATE,
    MIN_MONTHS_PER_COMBINATION,
    TARGETS,
    VARIANCE_THRESHOLD,
)

# Covariati disponibili nel dataset aggregato (aggiunti dalla pipeline dati)
COVARIATE_COLS = ["pct_icu", "pct_inpatient"]


@dataclass(frozen=True)
class TrainingSummary:
    attempted_combinations: int
    trained_resistant_models: int
    output_dir: Path
    baseline_comparison: dict = field(default_factory=dict)


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    if len(y_train) < 2:
        return np.nan

    baseline_mae = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    if baseline_mae == 0:
        return np.nan

    model_mae = np.mean(np.abs(y_true - y_pred))
    return float(model_mae / baseline_mae)


def _new_prophet_model(use_regressors: bool = False) -> Prophet:
    m = Prophet(weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
    if use_regressors:
        for col in COVARIATE_COLS:
            m.add_regressor(col, standardize=True)
    return m


def _arima_cv_rmse(series: pd.Series, cv_dates: list[pd.Timestamp]) -> float:
    """Calcola RMSE medio con walk-forward validation per un modello ARIMA(1,1,1)."""
    rmse_values: list[float] = []
    for end_date in cv_dates:
        train = series[series.index <= end_date].dropna()
        test = series[
            (series.index > end_date) & (series.index <= end_date + pd.DateOffset(months=6))
        ].dropna()
        if len(train) < 6 or test.empty:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                                enforce_stationarity=False, enforce_invertibility=False)
                fit = arima.fit(disp=False)
                pred = fit.forecast(len(test))
                pred = np.clip(pred, 0, 100)
                rmse_values.append(float(np.sqrt(mean_squared_error(test.values, pred))))
        except Exception:
            continue
    return float(np.mean(rmse_values)) if rmse_values else np.nan


def process_combination(
    combination: str,
    data: pd.DataFrame,
    variance_threshold: float = VARIANCE_THRESHOLD,
    max_training_date: pd.Timestamp = MAX_TRAINING_DATE,
) -> tuple[str, dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    local_forecasts: dict[str, pd.DataFrame] = {}
    local_metrics: dict[str, dict[str, float]] = {
        "rmse": {}, "mape": {}, "mase": {},
        "rmse_arima": {},  # baseline ARIMA per confronto
    }

    try:
        max_variance = data[list(TARGETS)].var().max()
        if max_variance > variance_threshold:
            return combination, {}, {}
    except Exception:
        return combination, {}, {}

    data = data[data["data"] <= max_training_date].copy()
    cv_dates = pd.to_datetime(["2023-01-31", "2023-06-30", "2023-12-31", "2024-06-30"])
    cv_dates = cv_dates[cv_dates < max_training_date]

    # Determina se i covariati ICU/inpatient sono disponibili e non tutti NaN
    has_covariates = all(
        col in data.columns and data[col].notna().sum() >= MIN_MONTHS_PER_COMBINATION
        for col in COVARIATE_COLS
    )

    for target in TARGETS:
        cols = ["data", target] + (COVARIATE_COLS if has_covariates else [])
        prophet_data = data[cols].rename(columns={"data": "ds", target: "y"})
        prophet_data["cap"] = 100.0
        prophet_data["floor"] = 0.0

        if has_covariates:
            for col in COVARIATE_COLS:
                prophet_data[col] = prophet_data[col].fillna(prophet_data[col].median())

        y_data = prophet_data["y"].dropna()
        if y_data.empty or len(y_data) < MIN_MONTHS_PER_COMBINATION:
            continue

        # Modello finale Prophet (con regressori se disponibili)
        final_model = _new_prophet_model(use_regressors=has_covariates)
        final_model.fit(prophet_data)

        future = final_model.make_future_dataframe(periods=FORECAST_PERIODS_MONTHS, freq="ME")
        future["cap"] = 100.0
        future["floor"] = 0.0
        if has_covariates:
            for col in COVARIATE_COLS:
                last_val = prophet_data[col].iloc[-1]
                future[col] = future[col] if col in future.columns else last_val
                future[col] = future[col].fillna(last_val)
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

            cv_model = _new_prophet_model(use_regressors=has_covariates)
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

        # Baseline ARIMA — confronto walk-forward sulla stessa serie
        series = prophet_data.set_index("ds")["y"]
        arima_rmse = _arima_cv_rmse(series, list(cv_dates))
        if not np.isnan(arima_rmse):
            local_metrics["rmse_arima"][target] = arima_rmse

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
    metrics = {target: {"rmse": {}, "mape": {}, "mase": {}, "rmse_arima": {}} for target in TARGETS}

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

    # Riepilogo comparativo Prophet vs ARIMA sul target principale (resistenti)
    baseline_comparison = _summarize_baseline_comparison(metrics)
    joblib.dump(baseline_comparison, output_path / "baseline_comparison.pkl")

    return TrainingSummary(
        attempted_combinations=len(combinations),
        trained_resistant_models=len(forecasts["resistenti"]),
        output_dir=output_path,
        baseline_comparison=baseline_comparison,
    )


def _summarize_baseline_comparison(
    metrics: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    """Confronto aggregato RMSE medio Prophet vs ARIMA sul target 'resistenti'."""
    target = "resistenti"
    prophet_rmses = list(metrics[target]["rmse"].values())
    arima_rmses = list(metrics[target]["rmse_arima"].values())
    return {
        "prophet_rmse_mean": float(np.mean(prophet_rmses)) if prophet_rmses else np.nan,
        "arima_rmse_mean": float(np.mean(arima_rmses)) if arima_rmses else np.nan,
        "prophet_rmse_median": float(np.median(prophet_rmses)) if prophet_rmses else np.nan,
        "arima_rmse_median": float(np.median(arima_rmses)) if arima_rmses else np.nan,
        "n_combinations_prophet": len(prophet_rmses),
        "n_combinations_arima": len(arima_rmses),
    }


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
