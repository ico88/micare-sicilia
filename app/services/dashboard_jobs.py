"""Background refresh of the archive from the public Rete MIC dashboard.

Extraction (several minutes) and import run in a thread; the guided path polls
the job state (jobs.py). The extraction holds the dashboard's whole history, so it
replaces the previous dashboard extractions (and, on request, the files
uploaded by hand from the same dashboard) instead of adding to them.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask

from app.models import UploadedFile
from app.services import jobs
from app.services.importing import import_file
from app.services.qlik_source import extract_to_csv

FILE_PREFIX = "cruscotto_retemic_"
EXTRACTION_SHARE = 85  # progress percent given to the extraction, the rest to the import


def is_dashboard_upload(upload: UploadedFile) -> bool:
    return upload.original_name.startswith(FILE_PREFIX)


def uploads_to_replace(include_manual: bool) -> list[int]:
    imported = UploadedFile.query.filter_by(status="imported").all()
    return [u.id for u in imported if include_manual or is_dashboard_upload(u)]


def start_dashboard_job(app: Flask, include_manual: bool) -> dict | None:
    """None when started; the job already running otherwise."""
    return jobs.start("cruscotto", app, lambda progress: _refresh(app, include_manual, progress))


def _refresh(app: Flask, include_manual: bool, progress) -> dict:
    path = _new_download_path(Path(app.config["UPLOAD_FOLDER"]))
    stats = extract_to_csv(app.config["FONTE_DATI_URL"], path,
                           progress=lambda message, fraction: progress(message, round(fraction * EXTRACTION_SHARE)))
    progress(f"Scaricate {stats['rows']:,} righe. Controllo e archiviazione in corso.".replace(",", "."),
             EXTRACTION_SHARE)
    try:
        outcome = import_file(path, path.name, app.config["DATASET_FOLDER"],
                              replaces=uploads_to_replace(include_manual))
    except Exception as exc:
        upload_id = getattr(exc, "upload_id", None)
        if upload_id:  # validation failed: its report says why
            exc.job_fields = {"link_endpoint": "upload.validation_report", "link_args": {"upload_id": upload_id}}
        raise
    # Earlier downloads are superseded copies of patient-level data: drop them
    for old in UploadedFile.query.filter(UploadedFile.id.in_(outcome.replaced)).all():
        if is_dashboard_upload(old) and Path(old.filename) != path:
            Path(old.filename).unlink(missing_ok=True)
    return {"message": outcome.message, "link_endpoint": "upload.validation_report",
            "link_args": {"upload_id": outcome.upload_id}}


def _new_download_path(folder: Path) -> Path:
    stem = f"{FILE_PREFIX}{datetime.now():%Y%m%d_%H%M%S}"
    path, n = folder / f"{stem}.csv", 1
    while path.exists():
        n += 1
        path = folder / f"{stem}_{n}.csv"
    return path
