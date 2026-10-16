"""Guided path for the operator: where we are and what to do next.

Four steps, always in this order:
  1 carica    upload the Rete MIC export
  2 controlla read the outcome of the data check
  3 calcola   compute the forecasts
  4 leggi     read the forecasts
The state is derived from what is on disk (validation reports, stored forecasts),
so it needs no bookkeeping of its own.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from . import dataset_store
from .forecast_run import load_forecasts, stale_reason
from .out_of_sample import CUTOFF, load_validation

STEPS = (
    ("carica", "Carica i dati"),
    ("controlla", "Controlla i dati"),
    ("calcola", "Calcola le previsioni"),
    ("leggi", "Leggi le previsioni"),
)
OLD_DATA_DAYS = 180   # data ending earlier than this are flagged as not current


def workflow_state(dataset_folder: str, forecast_folder: str, uploads: list, today: date | None = None) -> dict:
    """uploads: imported UploadedFile rows, oldest first."""
    today = today or date.today()
    reports = []
    for upload in uploads:
        report = dataset_store.load_report(dataset_folder, upload.id)
        if report and report.get("stats", {}).get("isolates_valid"):
            reports.append((upload, report))
    has_data = bool(reports)

    data = None
    if has_data:
        stats = [r["stats"] for _, r in reports]
        data_to = max(s["date_to"] for s in stats)
        last_upload, last_report = reports[-1]
        data = {
            "files": len(reports),
            "date_from": _it_date(min(s["date_from"] for s in stats)),
            "date_to": _it_date(data_to),
            "isolates": sum(s["isolates_valid"] for s in stats),
            "excluded": sum(s.get("rows_excluded", 0) for s in stats),
            "laboratories": max(s.get("laboratories", 0) for s in stats),
            "old": (today - date.fromisoformat(data_to)).days > OLD_DATA_DAYS,
            "last_upload_id": last_upload.id,
            "last_upload_name": last_upload.original_name,
            "warnings": len(last_report.get("warnings", [])),
            "last_excluded": last_report["stats"].get("rows_excluded", 0),
        }

    forecasts = load_forecasts(forecast_folder)
    meta = forecasts["meta"] if forecasts and forecasts["meta"].get("available") else None
    stale_why = stale_reason(meta, dataset_folder)
    stale = bool(stale_why)
    if meta:
        three_months = [f for c in forecasts["combinations"] for f in c["previsioni"] if f["orizzonte_mesi"] == 3]
        meta = meta | {"affidabili_3_mesi": sum(f["semaforo"] == "verde" for f in three_months)}

    # The out-of-sample check needs at least one complete quarter after the cutoff
    first_after = pd.Period(CUTOFF, freq="Q") + 1
    validation_possible = has_data and pd.Timestamp(max(r["stats"]["date_to"] for _, r in reports)) >= (
        first_after.end_time.normalize() - pd.Timedelta(days=7))
    validation = load_validation(forecast_folder)

    if not has_data:
        current = "carica"
    elif not meta or stale:
        current = "calcola"
    else:
        current = "leggi"
    order = [key for key, _ in STEPS]
    status = {}
    for key in order:
        if key == "controlla" and has_data:
            status[key] = "attenzione" if data["warnings"] else "fatto"
        elif order.index(key) < order.index(current):
            status[key] = "fatto"
        elif key == current:
            status[key] = "ora"
        else:
            status[key] = "dopo"
    return {"current": current, "status": status, "data": data, "forecasts": meta, "stale": stale,
            "stale_why": stale_why, "validazione_superata": stale_reason(validation, dataset_folder),
            "steps": STEPS, "validazione_possibile": validation_possible,
            "validazione": (validation or {}).get("sintesi"), "validazione_calcolata": (validation or {}).get("calcolato_il")}


def _it_date(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d/%m/%Y")
