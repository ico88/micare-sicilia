"""Forecasts shown to the operator: regional, invasive isolates, next four quarters.

For every pathogen x antibiotic the model and the traffic light come from the
test bench (benchmark.choose_and_verify): chosen on 2020-22, judged on 2023-24.
The forecast starts from the last complete quarter of data, not from today:
the page says which quarters the data cover and which quarters are forecast.

The result is stored as one JSON file, so the page reads it without refitting.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import HORIZONS, INTERVAL_Z, choose_and_verify, contexts, run_bench
from .breakpoints import BREAKPOINTS_VERSION
from .dataset_store import load_isolates
from .forecasting import FORECASTERS
from .reference import laboratory_names
from .series import MIN_ISOLATES_PER_PERIOD, build_series, regional_series, wilson_interval

POPULATION = "invasivi"
READING = "harmonized"
MIN_REPORTABLE_REGIONAL_QUARTERS = 20
RESULT_FILE = "previsioni.json"
QUARTER_NAMES = {1: "1° trimestre", 2: "2° trimestre", 3: "3° trimestre", 4: "4° trimestre"}

ANTIBIOTIC_LABELS = {
    "AMK": "Amikacina", "AMC": "Amoxicillina-clavulanato", "AMP": "Ampicillina", "FEP": "Cefepime",
    "FDC": "Cefiderocol", "CTX": "Cefotaxime", "FOX": "Cefoxitina", "CPT": "Ceftarolina",
    "CAZ": "Ceftazidime", "CZA": "Ceftazidime-avibactam", "CZT": "Ceftolozano-tazobactam",
    "CRO": "Ceftriaxone", "CIP": "Ciprofloxacina", "CLI": "Clindamicina", "COL": "Colistina",
    "SXT": "Cotrimossazolo", "DAP": "Daptomicina", "DOR": "Doripenem", "ERY": "Eritromicina",
    "ETP": "Ertapenem", "FOS": "Fosfomicina", "GEN": "Gentamicina", "GEH": "Gentamicina (alto livello)",
    "IPM": "Imipenem", "IMR": "Imipenem-relebactam", "LVX": "Levofloxacina", "LNZ": "Linezolid",
    "MEM": "Meropenem", "MEV": "Meropenem-vaborbactam", "NIT": "Nitrofurantoina", "ORI": "Oritavancina",
    "OXA": "Oxacillina", "PEN": "Penicillina", "TZP": "Piperacillina-tazobactam", "RIF": "Rifampicina",
    "STH": "Streptomicina (alto livello)", "TEC": "Teicoplanina", "TCY": "Tetraciclina",
    "TGC": "Tigeciclina", "TOB": "Tobramicina", "VAN": "Vancomicina",
}
PATHOGEN_LABELS = {
    "ACISPP": "Acinetobacter spp.", "ENTFAE": "Enterococcus faecalis", "ENTFCM": "Enterococcus faecium",
    "ESCCOL": "Escherichia coli", "KLEPNE": "Klebsiella pneumoniae", "PSEAER": "Pseudomonas aeruginosa",
    "STAAUR": "Staphylococcus aureus", "STRPNE": "Streptococcus pneumoniae", "NEIGON": "Neisseria gonorrhoeae",
}


def quarter_label(period: str | pd.Period) -> str:
    p = pd.Period(period, freq="Q")
    return f"{QUARTER_NAMES[p.quarter]} {p.year}"


def archive_signature(dataset_folder: str | Path) -> str:
    """Stored uploads with their size and time: changes when data are added or removed."""
    files = sorted(Path(dataset_folder).glob("*/upload_*.parquet"))
    return ";".join(f"{f.parent.name}/{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in files)


def compute_forecasts(dataset_folder: str | Path) -> dict:
    isolates = load_isolates(dataset_folder)
    if isolates.empty:
        return {"meta": {"available": False}, "combinations": []}
    quarterly = build_series(dataset_folder, "Q")
    labs = quarterly[quarterly["popolazione"] == POPULATION]
    regional = regional_series(quarterly, POPULATION)

    data_end = pd.Timestamp(isolates["data_prelievo"].max())
    last_period = pd.Period(data_end, freq="Q")
    # A quarter is complete only if the data reach (almost) its last day
    if data_end < last_period.end_time.normalize() - pd.Timedelta(days=7):
        last_period -= 1
    labs = labs[pd.PeriodIndex(labs["period"], freq="Q") <= last_period]
    regional = regional[pd.PeriodIndex(regional["period"], freq="Q") <= last_period]

    reportable = regional.groupby(["microorganismo", "antibiotic"])["reportable"].sum()
    eligible = set(reportable[reportable >= MIN_REPORTABLE_REGIONAL_QUARTERS].index)
    labs_eligible = labs[[k in eligible for k in zip(labs["microorganismo"], labs["antibiotic"])]]
    verdicts = choose_and_verify(run_bench(labs_eligible, READING)) if not labs_eligible.empty else pd.DataFrame()
    verdict_index = verdicts.set_index(["microorganismo", "antibiotic", "orizzonte_mesi"]) if not verdicts.empty else None
    contexts_by_key = {(p, a): (periods, ctx) for p, a, periods, ctx in contexts(labs_eligible, READING)}

    combos = []
    for (pathogen, antibiotic), hist in regional.groupby(["microorganismo", "antibiotic"]):
        hist = hist.sort_values("period_start")
        entry = _history_entry(pathogen, antibiotic, hist)
        entry["laboratori"] = _laboratory_last_year(labs, pathogen, antibiotic, last_period)
        if (pathogen, antibiotic) in eligible and verdict_index is not None:
            periods, ctx = contexts_by_key[(pathogen, antibiotic)]
            entry["previsioni"] = _forecasts(ctx, periods, verdict_index, pathogen, antibiotic)
        else:
            entry["previsioni"] = []
            entry["motivo_assenza"] = (
                f"Dati insufficienti per una previsione: servono almeno {MIN_REPORTABLE_REGIONAL_QUARTERS} "
                f"trimestri con {MIN_ISOLATES_PER_PERIOD} o più isolati testati; qui sono {int(reportable.get((pathogen, antibiotic), 0))}."
            )
        combos.append(entry)

    meta = {
        "available": True,
        "archivio": archive_signature(dataset_folder),
        "calcolato_il": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "popolazione": "isolati invasivi (sangue e liquor), tutti i laboratori della rete",
        "dati_fino_al": data_end.strftime("%d/%m/%Y"),
        "ultimo_trimestre": quarter_label(last_period),
        "trimestri_previsti": [quarter_label(last_period + h) for h in HORIZONS],
        "breakpoint": f"EUCAST v{BREAKPOINTS_VERSION}",
        "combinazioni": len(combos),
        "con_previsione": sum(1 for c in combos if c["previsioni"]),
        "laboratori": {code: laboratory_names().get(code, code) for code in sorted(labs["laboratorio"].unique())},
    }
    return {"meta": meta, "combinations": combos}


def _history_entry(pathogen: str, antibiotic: str, hist: pd.DataFrame) -> dict:
    n = hist[f"n_{READING}"].to_numpy(dtype=float)
    r = hist[f"r_{READING}"].to_numpy(dtype=float)
    last_n, last_r = n[-4:].sum(), r[-4:].sum()
    prev_n, prev_r = n[-8:-4].sum(), r[-8:-4].sum()
    last12 = None
    if last_n > 0:
        low, high = wilson_interval(pd.Series([last_r]), pd.Series([last_n]))
        last12 = {"pct": 100 * last_r / last_n, "n": int(last_n), "low": float(low.iloc[0]), "high": float(high.iloc[0])}
    return {
        "microorganismo": pathogen,
        "microorganismo_nome": PATHOGEN_LABELS.get(pathogen, pathogen),
        "antibiotico": antibiotic,
        "antibiotico_nome": ANTIBIOTIC_LABELS.get(antibiotic, antibiotic),
        "ultimi_12_mesi": last12,
        "tendenza": _trend_words(last_r, last_n, prev_r, prev_n),
        "storico": [
            {"trimestre": row.period, "n": int(row[f"n_{READING}"]), "pct": _num(row[f"pct_r_{READING}"]),
             "low": _num(row[f"low_{READING}"]), "high": _num(row[f"high_{READING}"])}
            for _, row in hist.iterrows()
        ],
    }


def _laboratory_last_year(labs: pd.DataFrame, pathogen: str, antibiotic: str, last_period) -> list[dict]:
    """Observed %R per laboratory over the last four quarters (indicative: few isolates each)."""
    recent = labs[(labs["microorganismo"] == pathogen) & (labs["antibiotic"] == antibiotic)
                  & (pd.PeriodIndex(labs["period"], freq="Q") > last_period - 4)]
    grouped = recent.groupby("laboratorio")[[f"n_{READING}", f"r_{READING}"]].sum()
    grouped = grouped[grouped[f"n_{READING}"] > 0]
    if grouped.empty:
        return []
    low, high = wilson_interval(grouped[f"r_{READING}"], grouped[f"n_{READING}"])
    names = laboratory_names()
    rows = [
        {"codice": lab, "nome": names.get(lab, lab), "n": int(row[f"n_{READING}"]),
         "pct": 100 * row[f"r_{READING}"] / row[f"n_{READING}"], "low": float(low[lab]), "high": float(high[lab]),
         "riportabile": bool(row[f"n_{READING}"] >= MIN_ISOLATES_PER_PERIOD)}
        for lab, row in grouped.iterrows()
    ]
    return sorted(rows, key=lambda r: -r["n"])


def _trend_words(r1, n1, r0, n0) -> str:
    """Last 12 months against the 12 before, with a two-proportion test."""
    if n1 < MIN_ISOLATES_PER_PERIOD or n0 < MIN_ISOLATES_PER_PERIOD:
        return "non valutabile"
    p1, p0 = r1 / n1, r0 / n0
    pooled = (r1 + r0) / (n1 + n0)
    se = np.sqrt(max(pooled * (1 - pooled), 1e-9) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se
    if abs(z) < 1.96:
        return "stabile"
    return f"in aumento ({100 * p0:.0f}% → {100 * p1:.0f}%)" if z > 0 else f"in calo ({100 * p0:.0f}% → {100 * p1:.0f}%)"


def _forecasts(ctx, periods, verdicts: pd.DataFrame, pathogen: str, antibiotic: str) -> list[dict]:
    origin = len(periods) - 1
    expected_n = max(float(ctx.n[max(0, origin - 3):origin + 1].mean()), 1.0)
    out = []
    for h in HORIZONS:
        key = (pathogen, antibiotic, 3 * h)
        if key not in verdicts.index:
            continue
        v = verdicts.loc[key]
        preds = FORECASTERS[v["mostrato"]](ctx, origin, [h])
        if h not in preds:
            continue
        p, var_mean, phi = preds[h]
        sd = np.sqrt(var_mean + phi * max(p * (1 - p), 1e-6) / expected_n)
        target = pd.Period(periods[origin], freq="Q") + h
        out.append({
            "orizzonte_mesi": 3 * h,
            "trimestre": quarter_label(target),
            "periodo": str(target),
            "pct": 100 * p,
            "low": 100 * max(0.0, p - INTERVAL_Z * sd),
            "high": 100 * min(1.0, p + INTERVAL_Z * sd),
            "modello": v["mostrato"],
            "semaforo": v["semaforo"],
            "errore_tipico": _num(v["errore_verifica"]),
            "copertura": _num(v["copertura_verifica"]),
            "prove": int(v["prove_verifica"]),
            "errore_su_rumore": _num(v["errore_su_rumore"]),
            "guadagno_pct": _num(v["guadagno_pct"]),
        })
    return out


def _num(value) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def save_forecasts(result: dict, folder: str | Path) -> Path:
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    target = path / RESULT_FILE
    target.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return target


def load_forecasts(folder: str | Path) -> dict | None:
    path = Path(folder) / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
