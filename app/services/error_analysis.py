"""Where the forecasts made with data up to the cutoff went wrong, and why.

Built on the out-of-sample comparison (out_of_sample.py). For every combination
the 2025 quarters are pooled into one observed value and set against the last
year before the cutoff (P0, 2024):

  observed change  P1 - P0 = mix + real change         (exact, laboratory by laboratory)
  deviation        P1 - F  = mix + (real change - change forecast)

  mix          laboratories weighing more or less in the regional figure, or
               joining / leaving: sum of (w1 - w0)(p0 - P0), entries w1 (p1 - P0)
  real change  resistance moving inside the laboratories: sum of w1 (p1 - p0)

with w the share of tested isolates and p the % resistant of each laboratory,
F the forecast pooled with the observed isolates of each quarter. A deviation
within 1.96 standard errors of the pooled observed value is called noise.

Per row the deviation of every forecaster is kept, so the model shown can be
compared, case by case, with what the others would have said. "Best with
hindsight" is a diagnosis, never a way to choose the model: choosing on 2025
would spend the only data no model has seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z_NOISE = 1.96
CAUSES = {
    "rumore": "nella variabilità casuale",
    "mix": "cambiato il mix dei laboratori",
    "reale": "resistenza cambiata in modo non previsto",
}


def analyse(rows: list[dict], lab_series: pd.DataFrame, cutoff: pd.Period, models: tuple[str, ...],
            reading: str = "harmonized") -> dict:
    """rows: comparison rows of compute_validation; lab_series: quarterly series of one population."""
    table = pd.DataFrame(rows)
    if table.empty:
        return {}
    table = table[table["semaforo"] != "grigio"]  # only forecasts the operator saw
    if table.empty:
        return {}
    base_periods = [str(p) for p in pd.period_range(cutoff - 3, cutoff, freq="Q")]
    labs = lab_series.set_index(["microorganismo", "antibiotic"]).sort_index()

    combos, lab_rows = [], []
    for (pathogen, antibiotic), group in table.groupby(["microorganismo", "antibiotico"], sort=False):
        if (pathogen, antibiotic) not in labs.index:
            continue
        series = labs.loc[[(pathogen, antibiotic)]]
        combo, contributions = _combination(group, series, base_periods, models, reading)
        if combo is None:
            continue
        combos.append(combo)
        lab_rows.extend(contributions)

    combos.sort(key=lambda c: -abs(c["scostamento"]))
    return {
        "sintesi": _summary(combos, table, models),
        "combinazioni": combos,
        "laboratori": lab_rows,
        "laboratori_sintesi": _laboratory_summary(lab_rows, lab_series, base_periods,
                                                  sorted(table["periodo"].unique()), reading),
        "modelli": _models(table, combos, models),
        "periodo_base": f"{base_periods[0]}–{base_periods[-1]}",
    }


def _combination(group: pd.DataFrame, series: pd.DataFrame, base_periods: list[str], models, reading: str):
    n_col, r_col = f"n_{reading}", f"r_{reading}"
    periods = sorted(group["periodo"].unique())
    base = series[series["period"].isin(base_periods)].groupby("laboratorio")[[n_col, r_col]].sum()
    now = series[series["period"].isin(periods)].groupby("laboratorio")[[n_col, r_col]].sum()
    lab = base.join(now, how="outer", lsuffix="_0", rsuffix="_1").fillna(0)
    n0, r0, n1, r1 = (lab[f"{c}_{s}"] for s in ("0", "1") for c in (n_col, r_col))
    N0, N1 = n0.sum(), n1.sum()
    if N0 <= 0 or N1 <= 0:
        return None, []
    P0, P1 = r0.sum() / N0, r1.sum() / N1
    w0, w1 = n0 / N0, n1 / N1
    p0 = (r0 / n0).where(n0 > 0)
    p1 = (r1 / n1).where(n1 > 0)
    both, new, gone = (n0 > 0) & (n1 > 0), (n0 == 0) & (n1 > 0), (n0 > 0) & (n1 == 0)
    mix = pd.Series(0.0, index=lab.index)
    mix[both] = ((w1 - w0) * (p0 - P0))[both]
    mix[new] = (w1 * (p1 - P0))[new]
    mix[gone] = (-w0 * (p0 - P0))[gone]
    real = pd.Series(0.0, index=lab.index)
    real[both] = (w1 * (p1 - p0))[both]

    weights = group.set_index("periodo")["n"]
    forecast = float((group.set_index("periodo")["previsto"] * weights).sum() / weights.sum()) / 100
    deviation = P1 - forecast
    mix_total, real_total = float(mix.sum()), float(real.sum())
    unforeseen = real_total - (forecast - P0)
    se = np.sqrt(max(P1 * (1 - P1), 1e-6) / N1)
    if abs(deviation) < Z_NOISE * se:
        cause = "rumore"
    elif abs(mix_total) >= abs(unforeseen):
        cause = "mix"
    else:
        cause = "reale"

    signed = group["osservato"] - group["previsto"]
    first = group.iloc[0]
    combo = {
        "microorganismo": first["microorganismo"], "microorganismo_nome": first["microorganismo_nome"],
        "antibiotico": first["antibiotico"], "antibiotico_nome": first["antibiotico_nome"],
        "modello": first["modello"], "trimestri": len(group),
        "n_base": int(N0), "n_osservati": int(N1),
        "base": 100 * P0, "previsto": 100 * forecast, "osservato": 100 * P1,
        "variazione_prevista": 100 * (forecast - P0), "variazione_osservata": 100 * (P1 - P0),
        "scostamento": 100 * deviation, "di_cui_mix": 100 * mix_total, "di_cui_non_previsto": 100 * unforeseen,
        "errore_casuale": 100 * se, "causa": cause,
        "sovrastime": int((signed < 0).sum()), "sottostime": int((signed > 0).sum()),
        "stesso_verso": bool((signed > 0).all() or (signed < 0).all()),
        "laboratori_nuovi": sorted(lab.index[new]), "laboratori_usciti": sorted(lab.index[gone]),
    }
    errors = {name: _mean_abs(group.get(f"errore_{name}")) for name in models}
    errors["mostrato"] = float(group["errore"].mean())
    combo["errori_modelli"] = errors
    complete = {k: v for k, v in errors.items() if v is not None}
    combo["migliore_col_senno_di_poi"] = min(complete, key=complete.get) if complete else None

    contributions = []
    for code in lab.index:
        contributions.append({
            "microorganismo": combo["microorganismo"], "antibiotico": combo["antibiotico"], "laboratorio": code,
            "stato": "nuovo" if new[code] else "uscito" if gone[code] else "presente",
            "n_base": int(n0[code]), "n_osservati": int(n1[code]),
            "peso_base": 100 * float(w0[code]), "peso_osservato": 100 * float(w1[code]),
            "resistenza_base": _pct(p0[code]), "resistenza_osservata": _pct(p1[code]),
            "contributo_mix": 100 * float(mix[code]), "contributo_reale": 100 * float(real[code]),
            "contributo": 100 * float(mix[code] + real[code]),
        })
    return combo, contributions


def _summary(combos: list[dict], table: pd.DataFrame, models) -> dict:
    if not combos:
        return {}
    dev = np.array([c["scostamento"] for c in combos])
    mix = np.array([c["di_cui_mix"] for c in combos])
    unforeseen = np.array([c["di_cui_non_previsto"] for c in combos])
    explained = np.abs(mix).sum() + np.abs(unforeseen).sum()
    signed = table["osservato"] - table["previsto"]
    by_horizon = {int(h): float((g["osservato"] - g["previsto"]).mean()) for h, g in table.groupby("orizzonte_mesi")}
    return {
        "combinazioni": len(combos),
        "cause": {k: sum(c["causa"] == k for c in combos) for k in CAUSES},
        "sovrastima": int((dev < 0).sum()), "sottostima": int((dev > 0).sum()),
        "stesso_verso": sum(c["stesso_verso"] for c in combos),
        "scostamento_medio": float(dev.mean()), "scostamento_assoluto_medio": float(np.abs(dev).mean()),
        "quota_mix": float(100 * np.abs(mix).sum() / explained) if explained else None,
        "quota_non_previsto": float(100 * np.abs(unforeseen).sum() / explained) if explained else None,
        "distorsione_trimestri": float(signed.mean()),
        "distorsione_per_orizzonte": by_horizon,
    }


def _laboratory_summary(lab_rows: list[dict], lab_series: pd.DataFrame, base_periods, periods, reading) -> list[dict]:
    if not lab_rows:
        return []
    rows = pd.DataFrame(lab_rows)
    rows["assoluto"] = rows["contributo"].abs()
    top = rows.loc[rows.groupby(["microorganismo", "antibiotico"])["assoluto"].idxmax()]
    isolates = lab_series.drop_duplicates(["laboratorio", "microorganismo", "period"])
    base = isolates[isolates["period"].isin(base_periods)].groupby("laboratorio")["isolates"].sum()
    now = isolates[isolates["period"].isin(periods)].groupby("laboratorio")["isolates"].sum()
    out = []
    for code, g in rows.groupby("laboratorio"):
        out.append({
            "laboratorio": code,
            "stato": "nuovo" if (g["stato"] == "nuovo").all() else "uscito" if (g["stato"] == "uscito").all()
                     else "presente",
            "isolati_base": int(base.get(code, 0)), "isolati_osservati": int(now.get(code, 0)),
            "combinazioni": int(len(g)),
            "primo_contributo": int((top["laboratorio"] == code).sum()),
            "contributo_medio_assoluto": float(g["assoluto"].mean()),
            "contributo_mix_medio": float(g["contributo_mix"].mean()),
            "contributo_reale_medio": float(g["contributo_reale"].mean()),
        })
    return sorted(out, key=lambda r: (-r["primo_contributo"], -r["contributo_medio_assoluto"]))


def _models(table: pd.DataFrame, combos: list[dict], models) -> dict:
    overall = [{"modello": "mostrato", "confronti": len(table), "errore": float(table["errore"].mean()),
                "distorsione": float((table["osservato"] - table["previsto"]).mean()),
                "migliore_in": sum(c["migliore_col_senno_di_poi"] == "mostrato" for c in combos)}]
    for name in models:
        errors = table.get(f"errore_{name}")
        if errors is None or errors.dropna().empty:
            continue
        predicted = table[f"previsto_{name}"]
        valid = predicted.notna()
        overall.append({"modello": name, "confronti": int(valid.sum()), "errore": float(errors.dropna().mean()),
                        "distorsione": float((table.loc[valid, "osservato"] - predicted[valid]).mean()),
                        "migliore_in": sum(c["migliore_col_senno_di_poi"] == name for c in combos)})
    by_choice = [{"modello_mostrato": name, "confronti": len(g),
                  "combinazioni": int(g[["microorganismo", "antibiotico"]].drop_duplicates().shape[0]),
                  "errore": float(g["errore"].mean()), "dentro": int(g["dentro"].sum())}
                 for name, g in table.groupby("modello")]
    return {"complessivo": overall, "per_modello_mostrato": by_choice}


def _mean_abs(values) -> float | None:
    if values is None:
        return None
    values = pd.to_numeric(values, errors="coerce")
    return None if values.isna().any() or values.empty else float(values.mean())


def _pct(value) -> float | None:
    return None if pd.isna(value) else 100 * float(value)
