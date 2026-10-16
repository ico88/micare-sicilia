"""Quarterly %R forecasters evaluated in the test bench (benchmark.py).

Every forecaster sees one combination (pathogen x antibiotic of one population)
up to an origin quarter and returns, per horizon h (quarters ahead):

    (p, var_mean, phi)

p         forecast proportion resistant
var_mean  variance of p itself (estimation uncertainty)
phi       dispersion (>= 1): extra quarter-to-quarter variability beyond binomial
          sampling, so the interval for an observed quarter of n isolates is
          p +/- z * sqrt(var_mean + phi * p (1 - p) / n)
          (0 for Prophet, whose interval already includes the noise it saw)

Inputs are count arrays over quarters: regional n, r (length T) and the same per
laboratory (L x T) for the laboratory-effect model.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-6


@dataclass
class Context:
    n: np.ndarray          # regional tested per quarter
    r: np.ndarray          # regional resistant per quarter
    lab_n: np.ndarray      # laboratories x quarters
    lab_r: np.ndarray
    first_period: str = "2019Q1"   # calendar quarter of index 0 (seasonality needs real dates)


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _dispersion(n: np.ndarray, r: np.ndarray, p_fit: np.ndarray, dof: int) -> float:
    """Pearson dispersion of observed quarters around fitted proportions (>= 1)."""
    mask = n > 0
    if mask.sum() <= dof:
        return 1.0
    resid = (r[mask] - n[mask] * p_fit[mask]) ** 2 / np.maximum(n[mask] * p_fit[mask] * (1 - p_fit[mask]), EPS)
    return float(max(1.0, resid.sum() / (mask.sum() - dof)))


# ---------------------------------------------------------------------------
# Persistence ("as the last 12 months")
# ---------------------------------------------------------------------------

def persistence(ctx: Context, origin: int, horizons, calibrated: bool = False):
    lo = max(0, origin - 3)
    n_used = ctx.n[lo:origin + 1].sum()
    if n_used <= 0:
        return {}
    p = ctx.r[lo:origin + 1].sum() / n_used
    phi = 1.0
    if calibrated:
        # quarter-to-quarter spread of the last two years around their level
        lo2 = max(0, origin - 7)
        n2, r2 = ctx.n[lo2:origin + 1], ctx.r[lo2:origin + 1]
        p2 = r2.sum() / max(n2.sum(), 1)
        phi = _dispersion(n2, r2, np.full(len(n2), p2), dof=1)
    var_mean = phi * max(p * (1 - p), EPS) / n_used
    return {h: (p, var_mean, phi) for h in horizons}


def persistence_calibrated(ctx, origin, horizons):
    return persistence(ctx, origin, horizons, calibrated=True)


# ---------------------------------------------------------------------------
# Binomial trend models (logit scale), recent quarters weighted more,
# trend damped when projected
# ---------------------------------------------------------------------------

HALF_LIFE_QUARTERS = 6
DAMPING = 0.85
MIN_QUARTERS = 6


def _fit_logit(X: np.ndarray, n: np.ndarray, r: np.ndarray, w: np.ndarray):
    """Weighted binomial GLM by IRLS. Returns (beta, cov_unscaled, fitted p) or None."""
    beta = np.zeros(X.shape[1])
    p0 = np.clip((r.sum() + 0.5) / (n.sum() + 1), 0.01, 0.99)
    beta[0] = np.log(p0 / (1 - p0))
    for _ in range(50):
        eta = X @ beta
        p = _expit(eta)
        W = w * n * p * (1 - p)
        z = eta + (r - n * p) / np.maximum(n * p * (1 - p), EPS)
        XtW = X.T * W
        H = XtW @ X + 1e-6 * np.eye(X.shape[1])  # tiny ridge: separable series stay finite
        try:
            new = np.linalg.solve(H, XtW @ z)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(new)):
            return None
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = np.clip(new, -20, 20)
    p = _expit(X @ beta)
    W = w * n * p * (1 - p)
    try:
        cov = np.linalg.inv((X.T * W) @ X + 1e-6 * np.eye(X.shape[1]))
    except np.linalg.LinAlgError:
        return None
    return beta, cov, p


def _damped_steps(h: int) -> float:
    return sum(DAMPING ** k for k in range(1, h + 1))


def trend(ctx: Context, origin: int, horizons):
    t = np.arange(origin + 1)
    n, r = ctx.n[:origin + 1], ctx.r[:origin + 1]
    keep = n > 0
    if keep.sum() < MIN_QUARTERS:
        return {}
    w = 0.5 ** ((origin - t) / HALF_LIFE_QUARTERS)
    X = np.column_stack([np.ones(len(t)), t - origin])
    fit = _fit_logit(X[keep], n[keep], r[keep], w[keep])
    if fit is None:
        return {}
    beta, cov, p_fit = fit
    phi = _dispersion(n[keep], r[keep], p_fit, dof=2)
    out = {}
    for h in horizons:
        x = np.array([1.0, _damped_steps(h)])
        p = float(_expit(x @ beta))
        var_eta = phi * float(x @ cov @ x)
        out[h] = (p, (p * (1 - p)) ** 2 * var_eta, phi)
    return out


def trend_by_laboratory(ctx: Context, origin: int, horizons):
    """Common trend with a level per laboratory; the regional forecast weights each
    laboratory by its share of the last four quarters, so a laboratory that starts
    sending more isolates changes the mix, not the estimated trend."""
    lab_n, lab_r = ctx.lab_n[:, :origin + 1], ctx.lab_r[:, :origin + 1]
    active = lab_n.sum(axis=1) > 0
    lab_n, lab_r = lab_n[active], lab_r[active]
    labs, quarters = lab_n.shape
    if labs == 0 or (ctx.n[:origin + 1] > 0).sum() < MIN_QUARTERS:
        return {}
    if labs == 1:
        return trend(ctx, origin, horizons)
    lab_idx, t_idx = np.nonzero(lab_n > 0)
    n, r = lab_n[lab_idx, t_idx], lab_r[lab_idx, t_idx]
    X = np.zeros((len(n), labs + 1))
    X[np.arange(len(n)), lab_idx] = 1.0
    X[:, -1] = t_idx - origin
    w = 0.5 ** ((origin - t_idx) / HALF_LIFE_QUARTERS)
    fit = _fit_logit(X, n, r, w)
    if fit is None:
        return {}
    beta, cov, p_fit = fit
    phi = _dispersion(n, r, p_fit, dof=labs + 1)
    recent = lab_n[:, max(0, origin - 3):origin + 1].sum(axis=1)
    if recent.sum() <= 0:
        return {}
    share = recent / recent.sum()
    out = {}
    for h in horizons:
        step = _damped_steps(h)
        eta = beta[:labs] + beta[-1] * step
        p_lab = _expit(eta)
        p = float(share @ p_lab)
        grad = np.zeros(labs + 1)
        grad[:labs] = share * p_lab * (1 - p_lab)
        grad[-1] = float(share @ (p_lab * (1 - p_lab))) * step
        out[h] = (p, phi * float(grad @ cov @ grad), phi)
    return out


# ---------------------------------------------------------------------------
# Prophet, as in the original project: %R on the logit scale, linear growth,
# yearly seasonality; its own 80% interval
# ---------------------------------------------------------------------------

PROPHET_MIN_TESTED = 10        # the original pipeline dropped periods with fewer samples
PROPHET_INTERVAL_Z = 1.2816    # Prophet interval_width=0.8, converted back to a standard deviation


def prophet_model(ctx: Context, origin: int, horizons):
    if not horizons:
        return {}
    try:
        from prophet import Prophet
    except ImportError:
        return {}
    for name in ("prophet", "cmdstanpy"):
        logging.getLogger(name).disabled = True
    n, r = ctx.n[:origin + 1], ctx.r[:origin + 1]
    keep = n >= PROPHET_MIN_TESTED
    if keep.sum() < MIN_QUARTERS:
        return {}
    starts = pd.period_range(ctx.first_period, periods=origin + 1 + max(horizons, default=0), freq="Q").start_time
    y = (r[keep] + 0.5) / (n[keep] + 1)
    history = pd.DataFrame({"ds": starts[:origin + 1][keep], "y": np.log(y / (1 - y))})
    # Quarterly data identify at most two yearly harmonics
    model = Prophet(growth="linear", yearly_seasonality=2, weekly_seasonality=False, daily_seasonality=False,
                    interval_width=0.8, uncertainty_samples=300)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(history, seed=0)
            np.random.seed(0)  # interval by simulation: same data, same interval
            future = pd.DataFrame({"ds": [starts[origin + h] for h in horizons]})
            pred = model.predict(future)
    except Exception:
        return {}
    out = {}
    for h, (_, row) in zip(horizons, pred.iterrows()):
        p, lo, hi = (float(_expit(row[c])) for c in ("yhat", "yhat_lower", "yhat_upper"))
        out[h] = (p, ((hi - lo) / (2 * PROPHET_INTERVAL_Z)) ** 2, 0.0)
    return out


FORECASTERS = {
    "persistenza_12m": persistence,
    "persistenza_tarata": persistence_calibrated,
    "trend": trend,
    "trend_laboratori": trend_by_laboratory,
    "prophet": prophet_model,
}
REFERENCE = "persistenza_12m"
