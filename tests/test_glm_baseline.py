import numpy as np
import pandas as pd

from micare_sicilia.model import _binomial_counts, _glm_binomial_cv_rmse


def _synthetic(months: int = 36, trials: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2019-01-31", periods=months, freq="ME")
    p = 1 / (1 + np.exp(-(-0.5 + 0.03 * np.arange(months))))
    r = rng.binomial(trials, p)
    return pd.DataFrame({
        "data": dates,
        "n_resistenti": r.astype(float),
        "n_intermedi": 0.0,
        "n_sensibili": (trials - r).astype(float),
    })


def test_binomial_counts_requires_count_columns():
    assert _binomial_counts(pd.DataFrame({"data": [pd.Timestamp("2020-01-31")]}), "resistenti") is None


def test_glm_cv_rmse_is_finite_and_small_on_logistic_trend():
    counts = _binomial_counts(_synthetic(), "resistenti")
    assert (counts["trials"] == 40).all()
    cv_dates = list(pd.date_range("2020-12-31", "2021-06-30", periods=3))
    rmse = _glm_binomial_cv_rmse(counts, cv_dates)
    assert np.isfinite(rmse)
    # Binomial noise alone with n=40 is ~7-8 pp; the trend must be captured
    assert rmse < 12


def test_glm_cv_rmse_nan_without_enough_history():
    counts = _binomial_counts(_synthetic(months=8), "resistenti")
    assert np.isnan(_glm_binomial_cv_rmse(counts, [pd.Timestamp("2019-03-31")]))
