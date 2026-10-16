"""Background refresh of the archive from the public Rete MIC dashboard.

Extraction (several minutes) and import run in a thread; the guided path polls
the job state. The extraction holds the dashboard's whole history, so it
replaces the previous dashboard extractions (and, on request, the files
uploaded by hand from the same dashboard) instead of adding to them.
"""
from __future__ import annotations

import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask

from app.models import UploadedFile
from app.services.importing import import_file
from app.services.qlik_source import extract_to_csv

FILE_PREFIX = "cruscotto_retemic_"
EXTRACTION_SHARE = 85  # progress percent given to the extraction, the rest to the import

_job: dict[str, Any] | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def is_dashboard_upload(upload: UploadedFile) -> bool:
    return upload.original_name.startswith(FILE_PREFIX)


def uploads_to_replace(include_manual: bool) -> list[int]:
    imported = UploadedFile.query.filter_by(status="imported").all()
    return [u.id for u in imported if include_manual or is_dashboard_upload(u)]


def start_dashboard_job(app: Flask, include_manual: bool) -> bool:
    """False when a refresh is already running."""
    global _job, _thread
    with _lock:
        if _job and _job["status"] in {"queued", "running"}:
            return False
        _job = {"status": "queued", "progress": 0, "message": "In attesa di avvio.", "error": "",
                "upload_id": None, "started_at": datetime.now(), "finished_at": None}
        _thread = threading.Thread(target=_run, args=(app, include_manual), daemon=True)
    _thread.start()
    return True


def get_dashboard_job() -> dict[str, Any] | None:
    with _lock:
        return dict(_job) if _job else None


def _update(**updates: Any) -> None:
    with _lock:
        _job.update(updates)


def _run(app: Flask, include_manual: bool) -> None:
    with app.app_context():
        try:
            path = _new_download_path(Path(app.config["UPLOAD_FOLDER"]))
            name = path.name

            def progress(message: str, fraction: float) -> None:
                _update(status="running", progress=round(fraction * EXTRACTION_SHARE), message=message)

            stats = extract_to_csv(app.config["FONTE_DATI_URL"], path, progress=progress)
            _update(progress=EXTRACTION_SHARE,
                    message=f"Scaricate {stats['rows']:,} righe. Controllo e archiviazione in corso.".replace(",", "."))
            outcome = import_file(path, name, app.config["DATASET_FOLDER"],
                                  replaces=uploads_to_replace(include_manual))
            # Earlier downloads are superseded copies of patient-level data: drop them
            for old in UploadedFile.query.filter(UploadedFile.id.in_(outcome.replaced)).all():
                if is_dashboard_upload(old) and Path(old.filename) != path:
                    Path(old.filename).unlink(missing_ok=True)
            _update(status="success", progress=100, message=outcome.message, upload_id=outcome.upload_id,
                    finished_at=datetime.now())
        except Exception as exc:
            _update(status="error", progress=100, message=str(exc), error=traceback.format_exc(),
                    upload_id=getattr(exc, "upload_id", None), finished_at=datetime.now())


def _new_download_path(folder: Path) -> Path:
    stem = f"{FILE_PREFIX}{datetime.now():%Y%m%d_%H%M%S}"
    path, n = folder / f"{stem}.csv", 1
    while path.exists():
        n += 1
        path = folder / f"{stem}_{n}.csv"
    return path
