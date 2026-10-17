"""Per-laboratory forecast of the next 12 months, chosen by the model search.

The search (model_search.py) compared 30 configurations for this target on tests
before 2025: each laboratory's last 12 months pulled toward the regional figure
(partial pooling, empirical Bayes) beat both the laboratory's own last 12 months and
the regional figure, on the selection period, on the verification period and on the
2025 confirmation looked at once at the end.

Interval and traffic light come from the laboratory's past tests, in two windows so
that the check is not circular:
  calibration   targets 9-16 quarters before the last one: how much wider than binomial
                sampling the interval must be to hold 80% of the time (one for all)
  verification  targets in the last 8 quarters: whether the 80% interval held and how
                large the error was, per combination
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .benchmark import INTERVAL_Z, MAX_ERROR_ON_NOISE, coverage_compatible
from .forecasting import EPS
from .model_search import NOISE_MAE, Combination, combinations, lab_shrunk
from .reference import laboratory_names
from .series import MIN_ISOLATES_PER_PERIOD

WINDOW = 4                # quarters of the laboratory level
PRIOR_SCALE = 3.0         # strength of the pull toward the region: best of 0.5-6 in the model search
VERIFY_QUARTERS = 8
CALIBRATION_QUARTERS = 8
MIN_VERIFICATION_TESTS = 10
# The region is one series: at most one test per target quarter, 8 in the last two years
MIN_REGION_TESTS = 6
MODEL = lab_shrunk(WINDOW, PRIOR_SCALE)


def _tests(c: Combination) -> pd.DataFrame:
    """Every past forecast of the next 12 months with its observed value (>= 30 tested)."""
    rows = []
    T = len(c.periods)
    for origin in range(7, T - 4):
        pred = MODEL(c.n, c.r, origin)[:, 0]
        n12 = c.n[:, origin + 1:origin + 5].sum(axis=1)
        r12 = c.r[:, origin + 1:origin + 5].sum(axis=1)
        for li in np.flatnonzero((n12 >= MIN_ISOLATES_PER_PERIOD) & np.isfinite(pred)):
            rows.append((c.labs[li], origin + 4, pred[li], r12[li] / n12[li], n12[li]))
    return pd.DataFrame(rows, columns=["laboratorio", "target", "previsto", "osservato", "n"])


def dispersion(all_tests: pd.DataFrame) -> float:
    """How much wider than binomial sampling the 80% interval must be (>= 1), chosen so that
    it held 80% of the time in the calibration window. A quantile, not the mean square:
    a few very large errors would widen every interval (89% coverage instead of 80%)."""
    if all_tests.empty:
        return 1.0
    p = all_tests["previsto"].clip(0.005, 0.995)
    z = (all_tests["osservato"] - p).abs() / np.sqrt(p * (1 - p) / all_tests["n"])
    return float(max(1.0, (np.quantile(z, 0.8) / INTERVAL_Z) ** 2))


def forecast_laboratories(lab_series: pd.DataFrame, reading: str, eligible: set) -> dict:
    """{(pathogen, antibiotic): [per-laboratory forecast]} for the eligible combinations."""
    combos = [c for c in combinations(lab_series, reading) if (c.pathogen, c.antibiotic) in eligible]
    if not combos:
        return {}
    T = len(combos[0].periods)
    tests = {(c.pathogen, c.antibiotic): _tests(c) for c in combos}
    verify_from = T - VERIFY_QUARTERS
    calibration = pd.concat([t[(t["target"] < verify_from) & (t["target"] >= verify_from - CALIBRATION_QUARTERS)]
                             for t in tests.values()], ignore_index=True)
    phi = dispersion(calibration)
    names = laboratory_names()
    out = {}
    for c in combos:
        verdict = _verdict(tests[(c.pathogen, c.antibiotic)], verify_from, phi)
        origin = T - 1
        pred = MODEL(c.n, c.r, origin)[:, 0]
        n_last = c.n[:, origin - 3:origin + 1].sum(axis=1)
        rows = []
        for li, code in enumerate(c.labs):
            if n_last[li] <= 0 or not np.isfinite(pred[li]):
                continue
            p = float(pred[li])
            sd = np.sqrt(phi * max(p * (1 - p), EPS) / max(n_last[li], 1))
            rows.append({
                "codice": code, "nome": names.get(code, code), "n_ultimi_12_mesi": int(n_last[li]),
                "pct": 100 * p, "low": 100 * max(0.0, p - INTERVAL_Z * sd), "high": 100 * min(1.0, p + INTERVAL_Z * sd),
                "semaforo": verdict["semaforo"] if n_last[li] >= MIN_ISOLATES_PER_PERIOD else "grigio",
            })
        out[(c.pathogen, c.antibiotic)] = {"laboratori": sorted(rows, key=lambda r: -r["n_ultimi_12_mesi"]),
                                           "verifica": verdict, "dispersione": phi}
    return out


def _verdict(tests: pd.DataFrame, verify_from: int, phi: float, min_tests: int = MIN_VERIFICATION_TESTS) -> dict:
    ver = tests[tests["target"] >= verify_from]
    if len(ver) < min_tests:
        return {"semaforo": "grigio", "prove": int(len(ver)), "errore": None, "copertura": None}
    p = ver["previsto"].clip(0.005, 0.995)
    sd = np.sqrt(phi * p * (1 - p) / ver["n"])
    err = (ver["osservato"] - ver["previsto"]).abs()
    inside = int((err <= INTERVAL_Z * sd).sum())
    noise = float((NOISE_MAE * np.sqrt(p * (1 - p) / ver["n"])).mean())
    ratio = float(err.mean() / noise) if noise > 0 else np.inf
    if not coverage_compatible(inside, len(ver)):
        semaforo = "grigio"
    elif ratio <= MAX_ERROR_ON_NOISE:
        semaforo = "verde"
    else:
        semaforo = "giallo"
    return {"semaforo": semaforo, "prove": int(len(ver)), "errore": 100 * float(err.mean()),
            "copertura": 100 * inside / len(ver), "errore_su_rumore": ratio}


def forecast_region_12_months(lab_series: pd.DataFrame, reading: str, eligible: set) -> dict:
    """{(pathogen, antibiotic): regional %R expected over the next 12 months}.

    Chosen by the model search: over 12 months the error is about a quarter lower than
    per quarter (2.9 against 3.7 points in 2023-2024), and no model beat "as the last 12
    months" by 5%. Interval and traffic light as for the laboratories, on the region."""
    combos = [c for c in combinations(lab_series, reading) if (c.pathogen, c.antibiotic) in eligible]
    if not combos:
        return {}
    regions = [Combination(c.pathogen, c.antibiotic, c.periods, ["regione"], c.n.sum(axis=0)[None, :],
                           c.r.sum(axis=0)[None, :]) for c in combos]
    T = len(regions[0].periods)
    tests = {(c.pathogen, c.antibiotic): _tests(c) for c in regions}
    verify_from = T - VERIFY_QUARTERS
    calibration = pd.concat([t[(t["target"] < verify_from) & (t["target"] >= verify_from - CALIBRATION_QUARTERS)]
                             for t in tests.values()], ignore_index=True)
    phi = dispersion(calibration)
    out = {}
    for c in regions:
        verdict = _verdict(tests[(c.pathogen, c.antibiotic)], verify_from, phi, MIN_REGION_TESTS)
        origin = T - 1
        n_last = float(c.n[0, origin - 3:origin + 1].sum())
        if n_last <= 0:
            continue
        p = float(c.r[0, origin - 3:origin + 1].sum() / n_last)
        sd = np.sqrt(phi * max(p * (1 - p), EPS) / n_last)
        out[(c.pathogen, c.antibiotic)] = {
            "pct": 100 * p, "low": 100 * max(0.0, p - INTERVAL_Z * sd), "high": 100 * min(1.0, p + INTERVAL_Z * sd),
            "n_ultimi_12_mesi": int(n_last),
            "semaforo": verdict["semaforo"] if n_last >= MIN_ISOLATES_PER_PERIOD else "grigio",
            "verifica": verdict, "dispersione": phi,
        }
    return out
