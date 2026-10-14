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
)

COVARIATE_COLS = ["pct_icu", "pct_inpatient"]

# Clip percentages to this range before logit transform to avoid ±∞
_EPS = 0.5


@dataclass(frozen=True)
class TrainingSummary:
    attempted_combinations: int
    trained_resistant_models: int
    output_dir: Path
    baseline_comparison: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bounded-transform helpers (logit ↔ percentage)
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    """Map percentage [0, 100] → unbounded logit scale."""
    p_c = np.clip(np.asarray(p, dtype=float), _EPS, 100 - _EPS)
    r = p_c / 100.0
    return np.log(r / (1 - r))


def _inv_logit(x: np.ndarray) -> np.ndarray:
    """Map logit scale → percentage [0, 100]."""
    return 100.0 / (1 + np.exp(-np.asarray(x, dtype=float)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    if len(y_train) < 2:
        return np.nan
    baseline_mae = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    if baseline_mae == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / baseline_mae)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _new_prophet_model(use_regressors: bool = False) -> Prophet:
    # Linear growth on logit-transformed data — no cap/floor needed
    m = Prophet(weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
    if use_regressors:
        for col in COVARIATE_COLS:
            m.add_regressor(col, standardize=True)
    return m


# ---------------------------------------------------------------------------
# ARIMA baseline
# ---------------------------------------------------------------------------

def _arima_cv_rmse(series: pd.Series, cv_dates: list[pd.Timestamp]) -> float:
    """Walk-forward ARIMA(1,1,1) RMSE on the original percentage scale."""
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


# ---------------------------------------------------------------------------
# Core per-combination training + CV
# ---------------------------------------------------------------------------

def _dynamic_cv_dates(first_date: pd.Timestamp, last_date: pd.Timestamp) -> pd.DatetimeIndex:
    """Compute walk-forward CV cutoff dates from the actual data window.

    Uses the 60th percentile of the series as the first cutoff and leaves
    at least 6 months of test data after the last cutoff.
    Returns an empty index when the series is too short.
    """
    series_months = max(0, (last_date.year - first_date.year) * 12
                        + last_date.month - first_date.month)
    if series_months < 18:
        return pd.DatetimeIndex([])

    cv_start = first_date + pd.DateOffset(months=max(12, int(series_months * 0.60)))
    cv_end = last_date - pd.DateOffset(months=6)
    if cv_start > cv_end:
        return pd.DatetimeIndex([])

    gap_days = (cv_end - cv_start).days
    n_splits = min(4, max(1, gap_days // 90 + 1))
    return pd.date_range(cv_start, cv_end, periods=n_splits)


def process_combination(
    combination: str,
    data: pd.DataFrame,
    max_training_date: pd.Timestamp = MAX_TRAINING_DATE,
) -> tuple[str, dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    local_forecasts: dict[str, pd.DataFrame] = {}
    local_metrics: dict[str, dict[str, float]] = {
        "rmse": {}, "mape": {}, "mase": {},
        "rmse_arima": {},
    }

    data = data[data["data"] <= max_training_date].copy()
    if data.empty:
        return combination, {}, {}

    first_date = data["data"].min()
    last_date = data["data"].max()
    cv_dates = _dynamic_cv_dates(first_date, last_date)

    has_covariates = all(
        col in data.columns and data[col].notna().sum() >= MIN_MONTHS_PER_COMBINATION
        for col in COVARIATE_COLS
    )

    for target in TARGETS:
        cols = ["data", target] + (COVARIATE_COLS if has_covariates else [])
        df_raw = data[cols].rename(columns={"data": "ds", target: "y"})

        if has_covariates:
            for col in COVARIATE_COLS:
                df_raw[col] = df_raw[col].fillna(df_raw[col].median())

        y_orig_series = df_raw["y"].dropna()
        if y_orig_series.empty or len(y_orig_series) < MIN_MONTHS_PER_COMBINATION:
            continue

        # Build prophet_data with logit-transformed y; keep y_orig for metrics
        prophet_data = df_raw.dropna(subset=["y"]).copy()
        prophet_data["y_orig"] = prophet_data["y"].values
        prophet_data["y"] = _logit(prophet_data["y"].values)

        # Columns fed to Prophet (no y_orig)
        fit_cols = ["ds", "y"] + (COVARIATE_COLS if has_covariates else [])

        # Final model on all data
        final_model = _new_prophet_model(use_regressors=has_covariates)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_model.fit(prophet_data[fit_cols])

        future = final_model.make_future_dataframe(periods=FORECAST_PERIODS_MONTHS, freq="ME")
        if has_covariates:
            for col in COVARIATE_COLS:
                last_val = prophet_data[col].iloc[-1]
                future[col] = future.get(col, last_val)
                future[col] = future[col].fillna(last_val)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = final_model.predict(future)

        # Transform back from logit scale
        forecast["yhat"] = np.clip(_inv_logit(forecast["yhat"].values), 0, 100)
        forecast["yhat_lower"] = np.clip(_inv_logit(forecast["yhat_lower"].values), 0, 100)
        forecast["yhat_upper"] = np.clip(_inv_logit(forecast["yhat_upper"].values), 0, 100)
        local_forecasts[target] = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        # Walk-forward CV (metrics in original percentage scale)
        rmse_values: list[float] = []
        mape_values: list[float] = []
        mase_values: list[float] = []

        for end_date in cv_dates:
            try:
                train = prophet_data[prophet_data["ds"] <= end_date].copy()
                test = prophet_data[
                    (prophet_data["ds"] > end_date)
                    & (prophet_data["ds"] <= end_date + pd.DateOffset(months=6))
                ].copy()

                if len(train) < 2 or test.empty:
                    continue

                test_pred_input = test[["ds"] + (COVARIATE_COLS if has_covariates else [])]

                cv_model = _new_prophet_model(use_regressors=has_covariates)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_model.fit(train[fit_cols])
                    cv_forecast = cv_model.predict(test_pred_input)

                y_pred_raw = np.clip(_inv_logit(cv_forecast["yhat"].values), 0, 100)
                y_true_raw = test["y_orig"].values
                y_train_orig = train["y_orig"].values

                valid = np.isfinite(y_pred_raw) & np.isfinite(y_true_raw)
                if valid.sum() == 0:
                    continue
                y_pred = y_pred_raw[valid]
                y_true = y_true_raw[valid]

                rmse_values.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
                # Denominator floored at 1 pp to avoid inflated MAPE near zero
                y_true_stable = np.where(np.abs(y_true) < 1.0, 1.0, y_true)
                mape_values.append(float(np.mean(np.abs((y_true - y_pred) / y_true_stable)) * 100))
                mase = calculate_mase(y_true, y_pred, y_train_orig)
                if not np.isnan(mase):
                    mase_values.append(mase)
            except Exception:
                continue

        if rmse_values:
            local_metrics["rmse"][target] = float(np.mean(rmse_values))
        if mape_values:
            local_metrics["mape"][target] = float(np.mean(mape_values))
        if mase_values:
            local_metrics["mase"][target] = float(np.mean(mase_values))

        # ARIMA baseline on original (non-logit) series
        series_orig = prophet_data.set_index("ds")["y_orig"]
        arima_rmse = _arima_cv_rmse(series_orig, list(cv_dates))
        if not np.isnan(arima_rmse):
            local_metrics["rmse_arima"][target] = arima_rmse

    return combination, local_forecasts, local_metrics


# ---------------------------------------------------------------------------
# Batch training entry point
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

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
) -> dict[str, dict[str, float]]:
    """Return per-target yhat + CI, normalized to sum to 100%."""
    forecasts = load_forecasts(output_dir)
    combination = f"{pathogen}_{laboratory}_{antibiotic}"
    date = pd.to_datetime(pd.Period(f"{year}-{month}", freq="M").end_time.date())

    raw: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        if combination not in forecasts[target]:
            raise KeyError(f"Nessuna previsione trovata per {combination}.")
        match = forecasts[target][combination][forecasts[target][combination]["ds"] == date]
        if match.empty:
            raise KeyError(f"Nessuna previsione trovata per {combination} alla data {date.date()}.")
        row = match.iloc[0]
        raw[target] = {
            "yhat": max(0.0, float(row["yhat"])),
            "lower": max(0.0, float(row["yhat_lower"])),
            "upper": max(0.0, float(row["yhat_upper"])),
        }

    total = sum(v["yhat"] for v in raw.values())
    if total <= 0:
        return {t: {"yhat": 0.0, "lower": 0.0, "upper": 0.0} for t in TARGETS}

    return {
        target: {
            "yhat": v["yhat"] * 100 / total,
            "lower": v["lower"] * 100 / total,
            "upper": v["upper"] * 100 / total,
        }
        for target, v in raw.items()
    }
