"""Out-of-sample validation: the forecasts as they were with data up to CUTOFF,
compared with the quarters observed afterwards.

No model has seen those quarters: the forecasts are recomputed by cutting the
archive at the end of CUTOFF (same series, same test bench, same choice of
model and traffic light), so they are exactly what the page showed then.
Each forecast is scored on the quarters with at least MIN_ISOLATES_PER_PERIOD
isolates; "as the last 12 months" and Prophet are scored on the same quarters.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import REFERENCE, coverage_compatible
from .error_analysis import analyse
from .forecasting import FORECASTERS
from .dataset_store import covered_period, load_isolates
from .forecast_run import (
    METHOD_VERSION, POPULATION, READING, archive_signature, compute_forecasts, last_complete_quarter,
    quarter_label,
)
from .series import MIN_ISOLATES_PER_PERIOD, build_series, regional_series

CUTOFF = "2024Q4"
COMPARED = (REFERENCE, "prophet")
ANALYSED = tuple(FORECASTERS)  # every candidate, for the error analysis
RESULT_FILE = "validazione.json"


def cutoff_label() -> str:
    return pd.Period(CUTOFF, freq="Q").end_time.strftime("%d/%m/%Y")


def validation_possible(dataset_folder: str | Path) -> bool:
    isolates = load_isolates(dataset_folder)
    if isolates.empty:
        return False
    return last_complete_quarter(covered_period(isolates["data_prelievo"])[1]) > pd.Period(CUTOFF, freq="Q")


def compute_validation(dataset_folder: str | Path) -> dict:
    isolates = load_isolates(dataset_folder)
    if isolates.empty:
        return {"available": False}
    last = last_complete_quarter(covered_period(isolates["data_prelievo"])[1])
    cutoff = pd.Period(CUTOFF, freq="Q")
    if last <= cutoff:
        return {"available": False, "ultimo_trimestre": quarter_label(last)}

    quarterly = build_series(dataset_folder, "Q")
    then = compute_forecasts(dataset_folder, until=CUTOFF, quarterly=quarterly, compare=ANALYSED)
    regional = regional_series(quarterly, POPULATION)
    regional = regional[pd.PeriodIndex(regional["period"], freq="Q") > cutoff]
    regional = regional[pd.PeriodIndex(regional["period"], freq="Q") <= last]
    observed = regional.set_index(["microorganismo", "antibiotic", "period"])

    rows = []
    for combo in then["combinations"]:
        for f in combo["previsioni"]:
            key = (combo["microorganismo"], combo["antibiotico"], f["periodo"])
            if key not in observed.index:
                continue
            obs = observed.loc[key]
            n = int(obs[f"n_{READING}"])
            if n < MIN_ISOLATES_PER_PERIOD:
                continue
            pct = 100 * obs[f"r_{READING}"] / n
            row = {
                "microorganismo": combo["microorganismo"], "microorganismo_nome": combo["microorganismo_nome"],
                "antibiotico": combo["antibiotico"], "antibiotico_nome": combo["antibiotico_nome"],
                "trimestre": f["trimestre"], "periodo": f["periodo"], "orizzonte_mesi": f["orizzonte_mesi"],
                "semaforo": f["semaforo"], "modello": f["modello"], "n": n,
                "previsto": f["pct"], "basso": f["low"], "alto": f["high"], "osservato": float(pct),
                "errore": float(abs(f["pct"] - pct)), "scostamento": float(pct - f["pct"]),
                "dentro": bool(f["low"] <= pct <= f["high"]),
            }
            for name in ANALYSED:
                other = combo.get("confronto", {}).get(name, {}).get(f["orizzonte_mesi"])
                row[f"previsto_{name}"] = other
                row[f"errore_{name}"] = None if other is None else float(abs(other - pct))
            rows.append(row)

    table = pd.DataFrame(rows)
    return {
        "available": True,
        "calcolato_il": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "archivio": archive_signature(dataset_folder),
        "metodo": METHOD_VERSION,
        "dati_fino_al": cutoff_label(),
        "trimestri_osservati": [quarter_label(p) for p in pd.period_range(cutoff + 1, last, freq="Q")],
        "sintesi": _summary(table[table["semaforo"] != "grigio"]) if not table.empty else None,
        "per_semaforo": {s: _summary(g) for s, g in table.groupby("semaforo")} if not table.empty else {},
        "per_orizzonte": {int(h): _summary(g) for h, g in table[table["semaforo"] != "grigio"].groupby("orizzonte_mesi")}
                         if not table.empty else {},
        "righe": rows,
        "analisi": _analysis(dataset_folder, rows, cutoff) if rows else {},
    }


def _analysis(dataset_folder, rows: list[dict], cutoff: pd.Period) -> dict:
    """Error analysis on laboratory x patient type strata: a laboratory that starts sending
    another kind of patient (e.g. non-hospitalised) shows up as a new stratum, i.e. as mix."""
    strata = build_series(dataset_folder, "Q", split_by="paziente_ricoverato")
    return analyse(rows, strata[strata["popolazione"] == POPULATION], cutoff, ANALYSED, READING)


def _summary(table: pd.DataFrame) -> dict | None:
    if table.empty:
        return None
    inside, total = int(table["dentro"].sum()), len(table)
    out = {
        "prove": total,
        "combinazioni": int(table[["microorganismo", "antibiotico"]].drop_duplicates().shape[0]),
        "errore": float(table["errore"].mean()),
        "dentro": inside,
        "copertura": 100 * inside / total,
        "copertura_compatibile": bool(coverage_compatible(inside, total)),
    }
    for name in COMPARED:
        errors = table[f"errore_{name}"].dropna()
        out[f"errore_{name}"] = float(errors.mean()) if len(errors) else None
    ref = out[f"errore_{REFERENCE}"]
    out["guadagno_pct"] = 100 * (1 - out["errore"] / ref) if ref else None
    return out


def save_validation(result: dict, folder: str | Path) -> Path:
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    target = path / RESULT_FILE
    target.write_text(json.dumps(result, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return target


def load_validation(folder: str | Path) -> dict | None:
    path = Path(folder) / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value))
