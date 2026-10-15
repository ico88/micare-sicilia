"""Rolling-origin test bench for quarterly resistance forecasts.

For every quarter of the past taken as "today" (the origin) each forecaster
(forecasting.py) sees only the data up to that quarter and predicts the %R
1-4 quarters ahead (3, 6, 9, 12 months); the prediction is compared with what
was observed in that quarter.

Two periods keep the choice of model honest: the model for each combination and
horizon is chosen on origins up to SELECTION_END, and judged on later origins
only (the verification period the choice has not seen).

Per combination, forecaster and horizon: mean absolute error (points of %R),
bias, share of observed values inside the 80% interval (acceptable 75-85%) and
the error expected from sampling noise alone, which no model can go below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecasting import FORECASTERS, REFERENCE, Context
from .series import MIN_ISOLATES_PER_PERIOD

HORIZONS = (1, 2, 3, 4)          # quarters ahead = 3, 6, 9, 12 months
MIN_HISTORY_QUARTERS = 8         # origins need two years of history
INTERVAL_Z = 1.2816              # two-sided 80%
NOISE_MAE_FACTOR = np.sqrt(2 / np.pi)  # E|X| for a normal with unit variance
SELECTION_END = "2022Q4"         # origins up to here choose the model; later ones verify it

KEYS = ["microorganismo", "antibiotic"]


def contexts(lab_series: pd.DataFrame, reading: str = "harmonized"):
    """Yield (pathogen, antibiotic, periods, Context) from laboratory-level quarterly series."""
    periods_all = sorted(lab_series["period"].unique())
    for (pathogen, antibiotic), group in lab_series.groupby(KEYS, sort=False):
        n_tab = group.pivot_table(index="laboratorio", columns="period", values=f"n_{reading}", aggfunc="sum")
        r_tab = group.pivot_table(index="laboratorio", columns="period", values=f"r_{reading}", aggfunc="sum")
        n_tab = n_tab.reindex(columns=periods_all).fillna(0)
        r_tab = r_tab.reindex(columns=periods_all).fillna(0)
        lab_n, lab_r = n_tab.to_numpy(dtype=float), r_tab.to_numpy(dtype=float)
        yield pathogen, antibiotic, np.array(periods_all), Context(lab_n.sum(0), lab_r.sum(0), lab_n, lab_r)


def run_bench(lab_series: pd.DataFrame, reading: str = "harmonized", forecasters=None) -> pd.DataFrame:
    """One row per pathogen x antibiotic x forecaster x horizon x origin."""
    forecasters = forecasters or FORECASTERS
    rows = []
    for pathogen, antibiotic, periods, ctx in contexts(lab_series, reading):
        T = len(periods)
        for origin in range(MIN_HISTORY_QUARTERS - 1, T - 1):
            for name, forecaster in forecasters.items():
                preds = forecaster(ctx, origin, [h for h in HORIZONS if origin + h < T])
                for h, (p, var_mean, phi) in preds.items():
                    target = origin + h
                    n_t = ctx.n[target]
                    if n_t < MIN_ISOLATES_PER_PERIOD or not np.isfinite(p):
                        continue
                    observed = ctx.r[target] / n_t
                    sd = np.sqrt(var_mean + phi * max(p * (1 - p), 1e-6) / n_t)
                    rows.append({
                        "microorganismo": pathogen, "antibiotic": antibiotic, "modello": name,
                        "orizzonte_mesi": 3 * h, "origine": periods[origin], "trimestre": periods[target],
                        "n": int(n_t), "previsto": 100 * p, "osservato": 100 * observed,
                        "basso_80": 100 * max(0.0, p - INTERVAL_Z * sd), "alto_80": 100 * min(1.0, p + INTERVAL_Z * sd),
                        "errore": 100 * (p - observed),
                        "dentro_80": abs(observed - p) <= INTERVAL_Z * sd,
                        "rumore": 100 * NOISE_MAE_FACTOR * np.sqrt(max(p * (1 - p), 1e-6) / n_t),
                    })
    return pd.DataFrame(rows)


def summarize(bench: pd.DataFrame) -> pd.DataFrame:
    """Per pathogen x antibiotic x forecaster x horizon."""
    if bench.empty:
        return pd.DataFrame()
    keys = KEYS + ["modello", "orizzonte_mesi"]
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
    return summary.groupby(["modello", "orizzonte_mesi"]).agg(
        combinazioni=("errore_medio", "size"),
        errore_medio=("errore_medio", "median"),
        errore_peggiore_10pct=("errore_medio", lambda s: s.quantile(0.9)),
        copertura_80=("copertura_80", "median"),
        rumore_atteso=("rumore_atteso", "median"),
        errore_su_rumore=("errore_su_rumore", "median"),
    ).reset_index()


def split_periods(bench: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = bench[bench["origine"] <= SELECTION_END]
    return selection, bench[bench["origine"] > SELECTION_END]


# ---------------------------------------------------------------------------
# Model choice (selection period) and verdict (verification period)
# ---------------------------------------------------------------------------

FALLBACK = "persistenza_tarata"   # same point forecast as the reference, honest interval
MIN_GAIN = 0.10                   # a model must cut the reference error by at least 10%
MAX_ERROR_ON_NOISE = 1.5
MIN_VERIFICATION_ORIGINS = 3
SEMAFORO_ORDER = ("verde", "giallo", "grigio")


def coverage_compatible(inside: int, total: int, target: float = 0.8, alpha: float = 0.05) -> bool:
    """Is the observed coverage compatible with 80%? Exact two-sided binomial test.

    With a few dozen tests per combination a fixed 75-85% band would reject honest
    intervals by chance; the test asks whether the deviation is beyond chance.
    """
    if total == 0:
        return False
    from scipy.stats import binomtest

    return binomtest(inside, total, target).pvalue >= alpha


def _pooled_error(bench: pd.DataFrame) -> pd.Series:
    """Mean absolute error per combination x model across all horizons."""
    return bench.assign(abs_err=bench["errore"].abs()).groupby(KEYS + ["modello"])["abs_err"].mean()


def _pooled_coverage(bench: pd.DataFrame) -> pd.Series:
    return 100 * bench.groupby(KEYS + ["modello"])["dentro_80"].mean()


def choose_and_verify(bench: pd.DataFrame) -> pd.DataFrame:
    """One model per combination (consistent across horizons), verdict per horizon.

    The model is chosen on the selection period from its error averaged over all
    horizons, and kept only if that gain is confirmed on the verification period;
    otherwise the reference with a calibrated interval is shown.

    Semaforo (per horizon, on verification tests of the model shown):
      verde   interval compatible with 80%, error <= 1.5 x noise, and either the model
              confirmed its gain or the reference itself is the forecast (stable resistance)
      giallo  interval compatible with 80% but error larger than that
      grigio  interval not compatible with 80%, or too few verification tests
    A horizon never gets a better light than the shorter ones (semaforo_orizzonte
    keeps the judgement of the horizon alone).
    """
    selection, verification = split_periods(bench)
    sel_err, sel_cov = _pooled_error(selection), _pooled_coverage(selection)
    ver_err = _pooled_error(verification)
    ver_all = verification.assign(abs_err=verification["errore"].abs())
    rows = []
    for (pathogen, antibiotic), combo in ver_all.groupby(KEYS):
        combo_key = (pathogen, antibiotic)
        ref_sel = sel_err.get((*combo_key, REFERENCE), np.nan)
        chosen = FALLBACK
        if np.isfinite(ref_sel):
            candidates = [
                (sel_err[(*combo_key, name)], name)
                for name in combo["modello"].unique()
                if name not in (REFERENCE, FALLBACK) and (*combo_key, name) in sel_err.index
                and sel_err[(*combo_key, name)] <= (1 - MIN_GAIN) * ref_sel
                and 60 <= sel_cov[(*combo_key, name)] <= 95
            ]
            if candidates:
                chosen = min(candidates)[1]
        ref_ver = ver_err.get((*combo_key, REFERENCE), np.nan)
        confirmed = chosen != FALLBACK and ver_err.get((*combo_key, chosen), np.inf) <= (1 - MIN_GAIN) * ref_ver
        shown = chosen if confirmed else FALLBACK

        worst = 0
        for horizon, group in combo.groupby("orizzonte_mesi"):
            g = group[group["modello"] == shown]
            ref_g = group[group["modello"] == REFERENCE]
            err = g["abs_err"].mean() if len(g) else np.nan
            ref_err = ref_g["abs_err"].mean() if len(ref_g) else np.nan
            inside, total = int(g["dentro_80"].sum()), len(g)
            noise = g["rumore"].mean() if total else np.nan
            ratio = err / noise if total else np.nan
            if g["origine"].nunique() < MIN_VERIFICATION_ORIGINS or not coverage_compatible(inside, total):
                semaforo = "grigio"
            elif ratio <= MAX_ERROR_ON_NOISE:
                semaforo = "verde"
            else:
                semaforo = "giallo"
            # Reliability cannot improve further ahead: with few tests per horizon a
            # farther horizon can look better by chance
            worst = max(worst, SEMAFORO_ORDER.index(semaforo))
            rows.append({
                "microorganismo": pathogen, "antibiotic": antibiotic, "orizzonte_mesi": horizon,
                "scelto_su_2020_22": chosen, "mostrato": shown, "semaforo": SEMAFORO_ORDER[worst],
                "semaforo_orizzonte": semaforo,
                "prove_verifica": total, "errore_verifica": err, "errore_riferimento_verifica": ref_err,
                "guadagno_pct": 100 * (1 - err / ref_err) if ref_err else np.nan,
                "copertura_verifica": 100 * inside / total if total else np.nan,
                "rumore_verifica": noise, "errore_su_rumore": ratio,
            })
    return pd.DataFrame(rows)
