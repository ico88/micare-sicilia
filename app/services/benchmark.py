"""Rolling-origin test bench for quarterly resistance forecasts.

For every quarter of the past taken as "today" (the origin) a forecaster sees
only the data up to that quarter and predicts the %R 1-4 quarters ahead (3, 6,
9, 12 months). The prediction is then compared with what was observed. A model
is worth showing only if it beats the simple references below on these tests.

References:
  persistenza_12m   %R of the last four quarters (pooled counts), with an 80%
                    interval from binomial sampling noise only
  ultimo_trimestre  %R of the last quarter

Per combination and horizon the bench reports the mean absolute error (points
of %R), the bias, the share of observed values inside the 80% interval
(acceptable 75-85%) and the error expected from sampling noise alone, which no
model can go below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .series import MIN_ISOLATES_PER_PERIOD

HORIZONS = (1, 2, 3, 4)          # quarters ahead = 3, 6, 9, 12 months
MIN_HISTORY_QUARTERS = 8         # origins need two years of history (fair to models with seasonality)
INTERVAL_Z = 1.2816              # two-sided 80%
NOISE_MAE_FACTOR = np.sqrt(2 / np.pi)  # E|X| for a normal with unit variance


def persistence_12m(n: np.ndarray, r: np.ndarray, origin: int) -> tuple[float, float]:
    """(p, n_used) from the four quarters ending at origin."""
    lo = max(0, origin - 3)
    n_used = n[lo:origin + 1].sum()
    return (r[lo:origin + 1].sum() / n_used if n_used > 0 else np.nan), n_used


def last_quarter(n: np.ndarray, r: np.ndarray, origin: int) -> tuple[float, float]:
    return (r[origin] / n[origin] if n[origin] > 0 else np.nan), n[origin]


FORECASTERS = {"persistenza_12m": persistence_12m, "ultimo_trimestre": last_quarter}


def run_bench(regional: pd.DataFrame, reading: str = "harmonized") -> pd.DataFrame:
    """One row per pathogen x antibiotic x forecaster x horizon x origin."""
    rows = []
    for (pathogen, antibiotic), group in regional.groupby(["microorganismo", "antibiotic"], sort=False):
        group = group.sort_values("period_start")
        n = group[f"n_{reading}"].to_numpy(dtype=float)
        r = group[f"r_{reading}"].to_numpy(dtype=float)
        periods = group["period"].to_numpy()
        for origin in range(MIN_HISTORY_QUARTERS - 1, len(group) - 1):
            for name, forecaster in FORECASTERS.items():
                p, n_used = forecaster(n, r, origin)
                if np.isnan(p):
                    continue
                for h in HORIZONS:
                    target = origin + h
                    if target >= len(group) or n[target] < MIN_ISOLATES_PER_PERIOD:
                        continue
                    observed = r[target] / n[target]
                    # sampling noise of the target quarter + of the estimate itself
                    sd = np.sqrt(max(p * (1 - p), 1e-6) * (1 / n[target] + 1 / max(n_used, 1)))
                    rows.append({
                        "microorganismo": pathogen, "antibiotic": antibiotic, "modello": name,
                        "orizzonte_mesi": 3 * h, "origine": periods[origin], "trimestre": periods[target],
                        "n": int(n[target]), "previsto": 100 * p, "osservato": 100 * observed,
                        "errore": 100 * (p - observed),
                        "dentro_80": abs(observed - p) <= INTERVAL_Z * sd,
                        "rumore": 100 * NOISE_MAE_FACTOR * np.sqrt(max(p * (1 - p), 1e-6) / n[target]),
                    })
    return pd.DataFrame(rows)


def summarize(bench: pd.DataFrame) -> pd.DataFrame:
    """Per pathogen x antibiotic x forecaster x horizon."""
    if bench.empty:
        return pd.DataFrame()
    keys = ["microorganismo", "antibiotic", "modello", "orizzonte_mesi"]
    out = bench.assign(abs_err=bench["errore"].abs()).groupby(keys).agg(
        prove=("errore", "size"),
        errore_medio=("abs_err", "mean"),
        distorsione=("errore", "mean"),
        copertura_80=("dentro_80", "mean"),
        rumore_atteso=("rumore", "mean"),
        testati_medi=("n", "mean"),
    ).reset_index()
    out["copertura_80"] = 100 * out["copertura_80"]
    out["errore_su_rumore"] = out["errore_medio"] / out["rumore_atteso"]
    return out


def overview(summary: pd.DataFrame) -> pd.DataFrame:
    """Per forecaster x horizon across combinations (each combination weighs the same)."""
    if summary.empty:
        return pd.DataFrame()
    out = summary.groupby(["modello", "orizzonte_mesi"]).agg(
        combinazioni=("errore_medio", "size"),
        errore_medio=("errore_medio", "median"),
        errore_peggiore_10pct=("errore_medio", lambda s: s.quantile(0.9)),
        copertura_80=("copertura_80", "median"),
        rumore_atteso=("rumore_atteso", "median"),
        errore_su_rumore=("errore_su_rumore", "median"),
    ).reset_index()
    return out
