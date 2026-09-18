from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask

from app.services.database import aggregated_from_db
from app.services.expanding_window_cv import run_expanding_window_cv

_jobs: dict[str, dict[str, Any]] = {}
_active_job_id: str | None = None
_lock = threading.Lock()


def start_expanding_cv_job(app: Flask, output_dir: str | Path) -> tuple[str, bool]:
    global _active_job_id
    with _lock:
        if _active_job_id is not None:
            active = _jobs.get(_active_job_id)
            if active and active["status"] in {"queued", "running"}:
                return _active_job_id, False

        job_id = uuid.uuid4().hex
        _jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "stage": "In coda",
            "message": "Validazione in attesa di avvio.",
            "error": "",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "finished_at": None,
            "summary": {},
        }
        _active_job_id = job_id

    thread = threading.Thread(target=_run, args=(app, job_id, str(output_dir)), daemon=True)
    thread.start()
    return job_id, True


def get_expanding_cv_job(job_id: str | None = None) -> dict[str, Any] | None:
    with _lock:
        selected = job_id or _active_job_id
        if selected is None or selected not in _jobs:
            return None
        return dict(_jobs[selected])


def _update_job(job_id: str, **updates: Any) -> None:
    with _lock:
        _jobs[job_id].update(updates)
        _jobs[job_id]["updated_at"] = datetime.utcnow()


def _run(app: Flask, job_id: str, output_dir: str) -> None:
    global _active_job_id
    with app.app_context():
        try:
            _update_job(job_id, status="running", progress=5, stage="Lettura dati", message="Carico lo storico aggregato.")
            data = aggregated_from_db()
            if data.empty:
                raise RuntimeError("Nessun dato aggregato disponibile.")

            def cb(progress: int, stage: str, message: str) -> None:
                _update_job(job_id, progress=progress, stage=stage, message=message)

            summary = run_expanding_window_cv(data, output_dir, progress_callback=cb)
            _update_job(
                job_id,
                status="success",
                progress=100,
                stage="Completato",
                message=f"Validazione completata. {summary.get('combination_count', 0)} combinazioni valutate.",
                summary=summary,
                finished_at=datetime.utcnow(),
            )
        except Exception as exc:
            _update_job(
                job_id,
                status="error",
                progress=100,
                stage="Errore",
                message=str(exc),
                error=traceback.format_exc(),
                finished_at=datetime.utcnow(),
            )
        finally:
            with _lock:
                if _active_job_id == job_id:
                    _active_job_id = None
