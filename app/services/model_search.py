"""Model search: many forecasters and combinations, judged without fooling ourselves.

Three targets, all from the laboratory-level quarterly series of one population:

  regionale      regional %R per quarter, laboratories pooled as they come (the figure
                 published today)
  pannello       regional %R per quarter at constant panel: each laboratory weighs as in
                 the four quarters before the forecast was made (ECDC repeats its trend
                 analysis on the laboratories that reported in every year, for the same
                 reason: laboratories joining or leaving move the pooled figure)
  laboratorio    %R of one laboratory over the next 12 months (four quarters pooled):
                 the local cumulative antibiogram a clinician uses

Every candidate predicts each laboratory (or the region directly); regional figures are
rebuilt from the laboratory forecasts with the weights of the target.

Three periods, by the quarter a test predicts, never mixed:

  scelta        targets up to 2022: candidates are ranked here
  verifica      targets in 2023-2024: the ranking must hold here to count
  conferma      forecasts made at the end of 2024 for 2025: looked at once, at the end,
                for the configurations already chosen; never used to choose

Every configuration tried is recorded with its errors, so the number of attempts is part
of the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .forecasting import EPS, _expit, _fit_logit, persistence, trend
from .series import MIN_ISOLATES_PER_PERIOD

HORIZONS = (1, 2, 3, 4)
MIN_HISTORY = 8
SELECTION_END = pd.Period("2022Q4", freq="Q")
VERIFICATION_END = pd.Period("2024Q4", freq="Q")
NOISE_MAE = np.sqrt(2 / np.pi)
HALF_LIFE = 6

Forecaster = Callable[[np.ndarray, np.ndarray, int], np.ndarray]  # -> labs x horizons


@dataclass
class Combination:
    pathogen: str
    antibiotic: str
    periods: list[str]
    labs: list[str]
    n: np.ndarray   # labs x quarters, tested
    r: np.ndarray   # labs x quarters, resistant


def combinations(lab_series: pd.DataFrame, reading: str = "harmonized", min_regional_quarters: int = 20):
    """Laboratory x quarter count matrices per pathogen x antibiotic with enough history."""
    periods = sorted(lab_series["period"].unique())
    out = []
    for (pathogen, antibiotic), g in lab_series.groupby(["microorganismo", "antibiotic"], sort=True):
        n = g.pivot_table(index="laboratorio", columns="period", values=f"n_{reading}", aggfunc="sum")
        r = g.pivot_table(index="laboratorio", columns="period", values=f"r_{reading}", aggfunc="sum")
        n, r = n.reindex(columns=periods).fillna(0), r.reindex(columns=periods).fillna(0)
        keep = n.sum(axis=1) > 0
        n, r = n[keep], r[keep]
        if (n.sum(axis=0) >= MIN_ISOLATES_PER_PERIOD).sum() < min_regional_quarters:
            continue
        out.append(Combination(pathogen, antibiotic, periods, list(n.index),
                               n.to_numpy(dtype=float), r.to_numpy(dtype=float)))
    return out


# ---------------------------------------------------------------------------
# Laboratory-level forecasters: (n, r) up to origin -> p per laboratory x horizon
# ---------------------------------------------------------------------------

def _window(a: np.ndarray, origin: int, k: int) -> np.ndarray:
    return a[:, max(0, origin - k + 1):origin + 1].sum(axis=1)


def lab_persistence(k: int) -> Forecaster:
    def f(n, r, origin):
        nn, rr = _window(n, origin, k), _window(r, origin, k)
        p = np.where(nn > 0, rr / np.maximum(nn, EPS), np.nan)
        return np.repeat(p[:, None], len(HORIZONS), axis=1)
    return f


def _prior_strength(nn: np.ndarray, rr: np.ndarray, pooled: float) -> float:
    """Empirical Bayes (method of moments): how many isolates the regional figure is worth
    for one laboratory. Laboratories that truly differ get a weak prior, similar ones a strong one."""
    ok = nn > 0
    if ok.sum() < 2:
        return 1e3
    p = rr[ok] / nn[ok]
    w = nn[ok] / nn[ok].sum()
    spread = float((w * (p - pooled) ** 2).sum())
    noise = float((w * pooled * (1 - pooled) / nn[ok]).sum())
    between = spread - noise
    if between <= 1e-6:
        return 1e3
    return float(np.clip(pooled * (1 - pooled) / between - 1, 1.0, 1e3))


def lab_shrunk(k: int, scale: float = 1.0) -> Forecaster:
    """Partial pooling: each laboratory's last k quarters pulled toward the regional figure,
    more strongly the fewer isolates it has (beta-binomial, prior strength from the data)."""
    def f(n, r, origin):
        nn, rr = _window(n, origin, k), _window(r, origin, k)
        if nn.sum() <= 0:
            return np.full((len(nn), len(HORIZONS)), np.nan)
        pooled = rr.sum() / nn.sum()
        m = scale * _prior_strength(nn, rr, pooled)
        p = (rr + m * pooled) / (nn + m)
        return np.repeat(p[:, None], len(HORIZONS), axis=1)
    return f


def lab_trend(half_life: float = HALF_LIFE, damping: float = 0.85) -> Forecaster:
    """A level per laboratory and one common slope (logit), recent quarters weighted more,
    slope damped when projected; laboratories without data keep the regional level."""
    def f(n, r, origin):
        labs = n.shape[0]
        out = np.full((labs, len(HORIZONS)), np.nan)
        nn, rr = n[:, :origin + 1], r[:, :origin + 1]
        active = nn.sum(axis=1) > 0
        if active.sum() == 0 or (nn.sum(axis=0) > 0).sum() < 6:
            return out
        idx = np.flatnonzero(active)
        lab_i, t_i = np.nonzero(nn[idx] > 0)
        X = np.zeros((len(lab_i), len(idx) + 1))
        X[np.arange(len(lab_i)), lab_i] = 1.0
        X[:, -1] = t_i - origin
        w = 0.5 ** ((origin - t_i) / half_life)
        fit = _fit_logit(X, nn[idx][lab_i, t_i], rr[idx][lab_i, t_i], w)
        if fit is None:
            return out
        beta = fit[0]
        for j, h in enumerate(HORIZONS):
            steps = sum(damping ** s for s in range(1, h + 1))
            out[idx, j] = _expit(beta[:len(idx)] + beta[-1] * steps)
        return out
    return f


def blend(a: Forecaster, b: Forecaster, weight_b: float) -> Forecaster:
    def f(n, r, origin):
        pa, pb = a(n, r, origin), b(n, r, origin)
        out = (1 - weight_b) * pa + weight_b * pb
        return np.where(np.isnan(pb), pa, np.where(np.isnan(pa), pb, out))
    return f


def regional_as_lab(model) -> Forecaster:
    """A regional model (forecasting.py) used for every laboratory: one regional value."""
    def f(n, r, origin):
        from .forecasting import Context
        ctx = Context(n.sum(axis=0), r.sum(axis=0), n, r)
        preds = model(ctx, origin, list(HORIZONS))
        row = np.array([preds[h][0] if h in preds else np.nan for h in HORIZONS])
        return np.repeat(row[None, :], n.shape[0], axis=0)
    return f


def candidates() -> dict[str, Forecaster]:
    """Every configuration the search tries. Adding one here is all it takes."""
    c: dict[str, Forecaster] = {
        "regionale · ultimi 12 mesi": regional_as_lab(persistence),
        "regionale · trend smorzato": regional_as_lab(trend),
    }
    for k in (2, 4, 8):
        c[f"laboratorio · ultimi {3 * k} mesi"] = lab_persistence(k)
        for scale in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
            c[f"laboratorio · ultimi {3 * k} mesi, verso la regione ×{scale:g}"] = lab_shrunk(k, scale)
    for half_life in (4, 6, 8):
        for damping in (0.7, 0.85, 1.0):
            c[f"laboratorio · trend (emivita {half_life}, smorzamento {damping:g})"] = lab_trend(half_life, damping)
    for weight in (0.25, 0.5, 0.75):
        c[f"media: verso la regione 12 mesi + trend, peso trend {weight:g}"] = blend(
            lab_shrunk(4), lab_trend(), weight)
    return c


# ---------------------------------------------------------------------------
# Machine learning: gradient boosting across all combinations and laboratories
# ---------------------------------------------------------------------------

GBM_NAME = "apprendimento automatico · gradient boosting (correzione del verso la regione)"


def _features(c: Combination, origin: int) -> pd.DataFrame:
    """What is known at the origin, per laboratory: own and regional levels and changes."""
    n4, r4 = _window(c.n, origin, 4), _window(c.r, origin, 4)
    n8, r8 = _window(c.n, origin, 8), _window(c.r, origin, 8)
    nprev, rprev = _window(c.n, origin - 4, 4), _window(c.r, origin - 4, 4)
    reg4 = r4.sum() / max(n4.sum(), EPS)
    reg_prev = rprev.sum() / max(nprev.sum(), EPS)
    own4 = np.where(n4 > 0, r4 / np.maximum(n4, EPS), reg4)
    own8 = np.where(n8 > 0, r8 / np.maximum(n8, EPS), reg4)
    own_prev = np.where(nprev > 0, rprev / np.maximum(nprev, EPS), own4)
    base = lab_shrunk(4)(c.n, c.r, origin)[:, 0]
    return pd.DataFrame({
        "base": base, "own4": own4, "own8": own8, "own_change": own4 - own_prev,
        "reg4": reg4, "reg_change": reg4 - reg_prev, "log_n4": np.log1p(n4),
        "share": n4 / max(n4.sum(), EPS), "labs": float((n4 > 0).sum()),
        "pathogen": c.pathogen, "antibiotic": c.antibiotic,
    })


def gbm_predictions(combos: list[Combination], holdout_origin: str = "2024Q4", progress=None):
    """{(combination index, origin index): labs x horizons}. At each origin the model is
    trained only on targets already observed by then (walk-forward), and learns the
    correction to partial pooling, so with little to learn it stays close to it."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    hold = pd.Period(holdout_origin, freq="Q")
    rows = []  # one per combination x laboratory x origin x horizon with an observed quarter
    for ci, c in enumerate(combos):
        periods = [pd.Period(p, freq="Q") for p in c.periods]
        for origin in range(MIN_HISTORY - 1, len(periods) - 1):
            if periods[origin] > hold:
                break
            feats = _features(c, origin)
            for j, h in enumerate(HORIZONS):
                t = origin + h
                obs = np.where(c.n[:, t] > 0, c.r[:, t] / np.maximum(c.n[:, t], EPS), np.nan) if t < len(periods) else None
                frame = feats.assign(ci=ci, origin=origin, origin_q=periods[origin], h=h,
                                     target_q=periods[t] if t < len(periods) else periods[origin] + h,
                                     lab=np.arange(len(feats)),
                                     y=(obs - feats["base"]) if obs is not None else np.nan,
                                     w=c.n[:, t] if t < len(periods) else 0.0)
                rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    for col in ("pathogen", "antibiotic"):
        data[col] = data[col].astype("category")
    cols = ["base", "own4", "own8", "own_change", "reg4", "reg_change", "log_n4", "share", "labs",
            "pathogen", "antibiotic", "h"]
    out: dict[tuple[int, int], np.ndarray] = {}
    origins = sorted(data["origin_q"].unique())
    for number, oq in enumerate(origins, start=1):
        if progress:
            progress(number, len(origins))
        train = data[(data["target_q"] <= oq) & data["y"].notna() & (data["w"] > 0)]
        now = data[data["origin_q"] == oq]
        if len(train) < 500:
            correction = np.zeros(len(now))
        else:
            model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                                  min_samples_leaf=40, l2_regularization=1.0,
                                                  categorical_features="from_dtype", random_state=0)
            model.fit(train[cols], train["y"], sample_weight=train["w"])
            correction = model.predict(now[cols])
        pred = np.clip(now["base"].to_numpy() + correction, 0.0, 1.0)
        for (ci, origin), g in now.assign(pred=pred).groupby(["ci", "origin"]):
            arr = np.full((len(combos[ci].labs), len(HORIZONS)), np.nan)
            arr[g["lab"].to_numpy(), g["h"].to_numpy() - 1] = g["pred"].to_numpy()
            out[(ci, origin)] = arr
    return out


# ---------------------------------------------------------------------------
# Tests: every candidate, every origin, the three targets
# ---------------------------------------------------------------------------

def period_of(test_target: pd.Period, origin: pd.Period, holdout_origin: pd.Period) -> str | None:
    if origin == holdout_origin and test_target > holdout_origin:
        return "conferma"
    if test_target <= SELECTION_END:
        return "scelta"
    if test_target <= VERIFICATION_END and origin < holdout_origin:
        return "verifica"
    return None


def run_tests(combos: list[Combination], models: dict[str, Forecaster] | None = None,
              holdout_origin: str = "2024Q4", progress=None,
              precomputed: dict[str, dict] | None = None) -> pd.DataFrame:
    """One row per candidate x target x combination (x laboratory) x origin x horizon."""
    models = models or candidates()
    hold = pd.Period(holdout_origin, freq="Q")
    rows = []
    for number, c in enumerate(combos, start=1):
        if progress:
            progress(number, len(combos))
        periods = [pd.Period(p, freq="Q") for p in c.periods]
        T = len(periods)
        for origin in range(MIN_HISTORY - 1, T - 1):
            o = periods[origin]
            if o > hold:
                break
            share = _window(c.n, origin, 4)
            if share.sum() <= 0:
                continue
            share = share / share.sum()
            forecasts = {name: m(c.n, c.r, origin) for name, m in models.items()}
            for name, table in (precomputed or {}).items():
                if (number - 1, origin) in table:
                    forecasts[name] = table[(number - 1, origin)]
            # regional and constant-panel targets, per horizon
            for j, h in enumerate(HORIZONS):
                t = origin + h
                if t >= T:
                    break
                which = period_of(periods[t], o, hold)
                if which is None:
                    continue
                n_t, r_t = c.n[:, t], c.r[:, t]
                present = n_t > 0
                # regional as published: pooled counts
                if n_t.sum() >= MIN_ISOLATES_PER_PERIOD:
                    obs = r_t.sum() / n_t.sum()
                    noise = NOISE_MAE * np.sqrt(max(obs * (1 - obs), EPS) / n_t.sum())
                    for name, f in forecasts.items():
                        pred = _weighted(f[:, j], share)
                        if np.isfinite(pred):
                            rows.append((name, "regionale", c.pathogen, c.antibiotic, "", str(o), str(periods[t]),
                                         3 * h, which, 100 * pred, 100 * obs, 100 * noise, n_t.sum()))
                # constant panel: weights of the four quarters before the forecast
                w = share * present
                if n_t[present & (share > 0)].sum() >= MIN_ISOLATES_PER_PERIOD and w.sum() > 0:
                    w = w / w.sum()
                    p_lab = np.where(present, r_t / np.maximum(n_t, EPS), 0)
                    obs = float(w @ p_lab)
                    var = float((w[present] ** 2 * np.maximum(p_lab[present] * (1 - p_lab[present]), EPS)
                                 / n_t[present]).sum())
                    for name, f in forecasts.items():
                        pred = _weighted(f[:, j], w)
                        if np.isfinite(pred):
                            rows.append((name, "pannello", c.pathogen, c.antibiotic, "", str(o), str(periods[t]),
                                         3 * h, which, 100 * pred, 100 * obs, 100 * NOISE_MAE * np.sqrt(var),
                                         n_t[present].sum()))
            # laboratory, next 12 months pooled
            if origin + 4 < T:
                which = period_of(periods[origin + 4], o, hold)
                if which is not None:
                    n12, r12 = c.n[:, origin + 1:origin + 5].sum(axis=1), c.r[:, origin + 1:origin + 5].sum(axis=1)
                    for li in np.flatnonzero(n12 >= MIN_ISOLATES_PER_PERIOD):
                        obs = r12[li] / n12[li]
                        noise = NOISE_MAE * np.sqrt(max(obs * (1 - obs), EPS) / n12[li])
                        for name, f in forecasts.items():
                            pred = np.nanmean(f[li]) if np.isfinite(f[li]).any() else np.nan
                            if np.isfinite(pred):
                                rows.append((name, "laboratorio", c.pathogen, c.antibiotic, c.labs[li], str(o),
                                             str(periods[origin + 4]), 12, which, 100 * pred, 100 * obs,
                                             100 * noise, n12[li]))
    return pd.DataFrame(rows, columns=["modello", "obiettivo", "microorganismo", "antibiotico", "laboratorio",
                                       "origine", "bersaglio", "orizzonte_mesi", "periodo", "previsto",
                                       "osservato", "rumore", "testati"])


def _weighted(p: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(p) & (w > 0)
    if not ok.any():
        return np.nan
    return float((p[ok] * w[ok]).sum() / w[ok].sum())


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

REFERENCE = "regionale · ultimi 12 mesi"


def leaderboard(tests: pd.DataFrame, reference: str = REFERENCE) -> pd.DataFrame:
    """Per target and candidate: error in each period, relative to the reference on the
    same tests (each combination weighs the same), and how often it beats the reference."""
    t = tests.assign(errore=(tests["previsto"] - tests["osservato"]).abs(),
                     scostamento=tests["osservato"] - tests["previsto"])
    keys = ["obiettivo", "microorganismo", "antibiotico", "laboratorio", "origine", "bersaglio", "orizzonte_mesi"]
    ref = t[t["modello"] == reference].set_index(keys)["errore"].rename("errore_rif")
    t = t.join(ref, on=keys).dropna(subset=["errore_rif"])
    out = []
    for (target, model), g in t.groupby(["obiettivo", "modello"]):
        row = {"obiettivo": target, "modello": model}
        for period, gp in g.groupby("periodo"):
            per_combo = gp.groupby(["microorganismo", "antibiotico", "laboratorio"])[["errore", "errore_rif", "rumore"]].mean()
            row[f"errore_{period}"] = float(per_combo["errore"].mean())
            row[f"rif_{period}"] = float(per_combo["errore_rif"].mean())
            row[f"rapporto_{period}"] = float((per_combo["errore"] / per_combo["errore_rif"].clip(lower=0.1)).median())
            row[f"meglio_{period}"] = float((per_combo["errore"] < per_combo["errore_rif"]).mean())
            row[f"rumore_{period}"] = float(per_combo["rumore"].mean())
            row[f"distorsione_{period}"] = float(gp["scostamento"].mean())
            row[f"prove_{period}"] = int(len(gp))
        out.append(row)
    board = pd.DataFrame(out)
    return board.sort_values(["obiettivo", "errore_verifica"]).reset_index(drop=True)


# The baseline each target must beat: for a laboratory, its own last 12 months
TARGET_REFERENCE = {"regionale": REFERENCE, "pannello": REFERENCE, "laboratorio": "laboratorio · ultimi 12 mesi"}


def choose(board: pd.DataFrame, min_gain: float = 0.05) -> pd.DataFrame:
    """Per target, the candidate chosen without looking at 2025: lowest verification error
    among those that beat the target's baseline by min_gain on BOTH selection and verification."""
    rows = []
    for target, g in board.groupby("obiettivo"):
        base = g[g["modello"] == TARGET_REFERENCE[target]].iloc[0]
        ok = g[(g["errore_scelta"] <= (1 - min_gain) * base["errore_scelta"])
               & (g["errore_verifica"] <= (1 - min_gain) * base["errore_verifica"])]
        best = ok.nsmallest(1, "errore_verifica") if len(ok) else g[g["modello"] == base["modello"]]
        rows.append(best.assign(
            riferimento=base["modello"], errore_rif_scelta=base["errore_scelta"],
            errore_rif_verifica=base["errore_verifica"], errore_rif_conferma=base.get("errore_conferma"),
            scelto_perche="batte il riferimento in scelta e in verifica" if len(ok)
            else "nessun candidato batte il riferimento in modo confermato"))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Whole search on an archive, and its workbook
# ---------------------------------------------------------------------------

def run_search(dataset_folder, population: str = "invasivi", reading: str = "harmonized",
               holdout_origin: str = "2024Q4", progress=None) -> dict:
    from datetime import datetime

    from .series import build_series

    say = progress or (lambda message: None)
    say("Costruzione delle serie per laboratorio")
    series = build_series(dataset_folder, "Q")
    series = series[series["popolazione"] == population]
    combos = combinations(series, reading)
    say(f"{len(combos)} combinazioni con storia sufficiente; addestramento del gradient boosting")
    gbm = gbm_predictions(combos, holdout_origin,
                          progress=lambda i, n: say(f"Gradient boosting: data {i} di {n}") if i % 5 == 0 else None)
    models = candidates()
    say(f"Prove di {len(models) + 1} configurazioni")
    tests = run_tests(combos, models, holdout_origin, precomputed={GBM_NAME: gbm},
                      progress=lambda i, n: say(f"Prove: combinazione {i} di {n}") if i % 10 == 0 else None)
    board = leaderboard(tests)
    return {
        "calcolato_il": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "popolazione": population, "lettura": reading, "taglio": holdout_origin,
        "combinazioni": len(combos), "configurazioni": len(models) + 1,
        "prove": tests, "classifica": board, "scelti": choose(board),
    }


def write_workbook(result: dict, path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    readme = pd.DataFrame([
        ("Calcolato il", result["calcolato_il"]),
        ("Popolazione", result["popolazione"]), ("Lettura", result["lettura"]),
        ("Combinazioni", result["combinazioni"]), ("Configurazioni provate", result["configurazioni"]),
        ("Prove", len(result["prove"])),
        ("Periodo di scelta", f"bersagli fino al {SELECTION_END}"),
        ("Periodo di verifica", f"bersagli da {SELECTION_END + 1} a {VERIFICATION_END}"),
        ("Conferma", f"previsioni fatte a fine {result['taglio']} per i 4 trimestri successivi: solo controllo"),
        ("Obiettivo regionale", "% R regionale per trimestre, laboratori sommati come arrivano"),
        ("Obiettivo pannello", "% R regionale per trimestre a pesi fissi: ogni laboratorio pesa come nei 4 trimestri prima della previsione"),
        ("Obiettivo laboratorio", "% R del laboratorio nei 12 mesi successivi, se almeno 30 testati"),
        ("Errore", "media per combinazione dell'errore assoluto, in punti di % R; ogni combinazione pesa uguale"),
        ("Rapporto", "mediana fra combinazioni di errore del modello / errore del riferimento (regionale, ultimi 12 mesi)"),
        ("Rumore", "errore che produrrebbe il solo caso con quel numero di testati"),
        ("Regola di scelta", "errore almeno 5% sotto il riferimento sia in scelta sia in verifica; tra questi il minore in verifica"),
    ], columns=["Voce", "Valore"])
    board = result["classifica"].round(3)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="Leggimi", index=False)
        result["scelti"].round(3).to_excel(writer, sheet_name="Scelti", index=False)
        board.to_excel(writer, sheet_name="Classifica", index=False)
        tests = result["prove"]
        summary = (tests.assign(errore=(tests["previsto"] - tests["osservato"]).abs())
                   .groupby(["obiettivo", "modello", "periodo", "orizzonte_mesi"])["errore"].agg(["mean", "size"])
                   .round(3).reset_index().rename(columns={"mean": "errore", "size": "prove"}))
        summary.to_excel(writer, sheet_name="Per orizzonte", index=False)


SUMMARY_FILE = "ricerca_modelli.json"
WORKBOOK_FILE = "ricerca_modelli.xlsx"


def save_summary(result: dict, folder) -> None:
    """What the page shows: the choice per target and the ranking (no test rows)."""
    import json
    from pathlib import Path

    def records(frame: pd.DataFrame) -> list[dict]:
        return json.loads(frame.to_json(orient="records", force_ascii=False))

    summary = {k: result[k] for k in ("calcolato_il", "popolazione", "lettura", "taglio", "combinazioni", "configurazioni")}
    summary |= {"prove": int(len(result["prove"])), "scelti": records(result["scelti"]),
                "classifica": records(result["classifica"])}
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    (path / SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")


def load_summary(folder) -> dict | None:
    import json
    from pathlib import Path

    path = Path(folder) / SUMMARY_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
