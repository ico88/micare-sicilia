from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask

from app.services.backtest import run_rolling_backtest
from app.services.database import aggregated_from_db

_jobs: dict[str, dict[str, Any]] = {}
_active_job_id: str | None = None
_lock = threading.Lock()


def start_backtest_job(app: Flask, output_dir: str | Path, months: int = 6) -> tuple[str, bool]:
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
            "message": "Backtest in attesa di avvio.",
            "error": "",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "finished_at": None,
            "summary": {},
        }
        _active_job_id = job_id

    thread = threading.Thread(target=_run_backtest_job, args=(app, job_id, str(output_dir), months), daemon=True)
    thread.start()
    return job_id, True


def get_backtest_job(job_id: str | None = None) -> dict[str, Any] | None:
    with _lock:
        selected_job_id = job_id or _active_job_id
        if selected_job_id is None or selected_job_id not in _jobs:
            return None
        return dict(_jobs[selected_job_id])


def _update_job(job_id: str, **updates: Any) -> None:
    with _lock:
        job = _jobs[job_id]
        job.update(updates)
        job["updated_at"] = datetime.utcnow()


def _run_backtest_job(app: Flask, job_id: str, output_dir: str, months: int) -> None:
    global _active_job_id
    with app.app_context():
        try:
            _update_job(job_id, status="running", progress=5, stage="Lettura dati", message="Carico lo storico aggregato.")
            data = aggregated_from_db()
            if data.empty:
                raise RuntimeError("Nessun dato aggregato disponibile per il backtest.")

            def progress_callback(progress: int, stage: str, message: str) -> None:
                _update_job(job_id, progress=progress, stage=stage, message=message)

            summary = run_rolling_backtest(data, output_dir, months=months, progress_callback=progress_callback)
            _update_job(
                job_id,
                status="success",
                progress=100,
                stage="Completato",
                message=f"Backtest completato su {summary.get('months_tested', 0)} mesi.",
                summary=summary,
                finished_at=datetime.utcnow(),
            )
        except Exception as exc:  # pragma: no cover
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
