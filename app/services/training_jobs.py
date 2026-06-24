from __future__ import annotations

import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask

from app import db
from app.services.database import aggregated_from_db, save_training_summary
from app.services.hierarchical_training import train_hierarchical_models
from app.services.training import train_all_models

_jobs: dict[str, dict[str, Any]] = {}
_active_job_id: str | None = None
_lock = threading.Lock()


class TrainingStopped(RuntimeError):
    pass


def start_training_job(app: Flask, model_folder: str | Path) -> tuple[str, bool]:
    global _active_job_id
    with _lock:
        if _active_job_id is not None:
            active = _jobs.get(_active_job_id)
            if active and active["status"] in {"queued", "running", "paused", "stopping"}:
                return _active_job_id, False

        job_id = uuid.uuid4().hex
        _jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "stage": "In coda",
            "message": "Training in attesa di avvio.",
            "error": "",
            "created_at": datetime.utcnow(),
            "started_at": None,
            "paused_at": None,
            "paused_seconds": 0,
            "updated_at": datetime.utcnow(),
            "finished_at": None,
            "elapsed_seconds": 0,
            "eta_seconds": None,
            "eta_text": "Calcolo stima...",
            "work_completed": 0,
            "work_total": 0,
            "work_label": "",
            "pause_requested": False,
            "stop_requested": False,
            "summary": {},
        }
        _active_job_id = job_id

    thread = threading.Thread(target=_run_training_job, args=(app, job_id, str(model_folder)), daemon=True)
    thread.start()
    return job_id, True


def get_training_job(job_id: str | None = None) -> dict[str, Any] | None:
    with _lock:
        selected_job_id = job_id or _active_job_id
        if selected_job_id is None or selected_job_id not in _jobs:
            return None
        return dict(_jobs[selected_job_id])


def _update_job(job_id: str, **updates: Any) -> None:
    with _lock:
        job = _jobs[job_id]
        now = datetime.utcnow()
        job.update(updates)
        if job.get("started_at") is not None:
            paused_seconds = int(job.get("paused_seconds") or 0)
            if job.get("paused_at") is not None:
                paused_seconds += max(0, int((now - job["paused_at"]).total_seconds()))
            elapsed = max(0, int((now - job["started_at"]).total_seconds()) - paused_seconds)
            job["elapsed_seconds"] = elapsed
            if job.get("status") in {"success", "error", "stopped"} or int(job.get("progress") or 0) >= 100:
                job["eta_seconds"] = 0
                job["eta_text"] = "Completato"
            elif job.get("status") == "paused":
                job["eta_seconds"] = None
                job["eta_text"] = "In pausa"
            elif job.get("status") == "stopping":
                job["eta_seconds"] = None
                job["eta_text"] = "Arresto richiesto"
            else:
                eta = _estimate_eta_seconds(job, elapsed)
                job["eta_seconds"] = eta
                job["eta_text"] = _format_duration(eta)
        job["updated_at"] = now


def control_training_job(job_id: str | None, action: str) -> dict[str, Any] | None:
    with _lock:
        if job_id is None or job_id not in _jobs:
            return None
        job = _jobs[job_id]
        now = datetime.utcnow()
        status = job.get("status")
        if action == "pause" and status == "running":
            job["pause_requested"] = True
            job["status"] = "paused"
            job["paused_at"] = now
            job["message"] = "Training in pausa. Puoi riprenderlo o fermarlo definitivamente."
            job["eta_seconds"] = None
            job["eta_text"] = "In pausa"
        elif action == "resume" and status == "paused":
            if job.get("paused_at") is not None:
                job["paused_seconds"] = int(job.get("paused_seconds") or 0) + max(0, int((now - job["paused_at"]).total_seconds()))
            job["paused_at"] = None
            job["pause_requested"] = False
            job["status"] = "running"
            job["message"] = "Training ripreso."
        elif action == "stop" and status in {"queued", "running", "paused"}:
            job["stop_requested"] = True
            job["pause_requested"] = False
            if job.get("paused_at") is not None:
                job["paused_seconds"] = int(job.get("paused_seconds") or 0) + max(0, int((now - job["paused_at"]).total_seconds()))
            job["paused_at"] = None
            job["status"] = "stopping"
            job["message"] = "Arresto richiesto. Il training si fermerà al prossimo checkpoint sicuro."
            job["eta_seconds"] = None
            job["eta_text"] = "Arresto richiesto"
        job["updated_at"] = now
        return dict(job)


def _checkpoint(job_id: str) -> None:
    while True:
        with _lock:
            job = _jobs[job_id]
            if job.get("stop_requested"):
                raise TrainingStopped("Training interrotto dall'utente.")
            if not job.get("pause_requested"):
                if job.get("status") == "paused":
                    job["status"] = "running"
                    job["updated_at"] = datetime.utcnow()
                return
            job["status"] = "paused"
            job["eta_seconds"] = None
            job["eta_text"] = "In pausa"
            job["message"] = "Training in pausa. Puoi riprenderlo o fermarlo definitivamente."
            job["updated_at"] = datetime.utcnow()
        time.sleep(0.5)


def _estimate_eta_seconds(job: dict[str, Any], elapsed: int) -> int | None:
    work_completed = int(job.get("work_completed") or 0)
    work_total = int(job.get("work_total") or 0)
    if work_total > 0 and work_completed > 0:
        seconds_per_unit = elapsed / work_completed
        return max(0, int(seconds_per_unit * (work_total - work_completed)))

    progress = int(job.get("progress") or 0)
    if progress > 0:
        estimated_total = elapsed / (progress / 100)
        return max(0, int(estimated_total - elapsed))
    return None


def _format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "Calcolo stima..."
    if seconds < 60:
        return f"circa {seconds} s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"circa {minutes} min {remaining_seconds:02d} s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"circa {hours} h {remaining_minutes:02d} min"


def _run_training_job(app: Flask, job_id: str, model_folder: str) -> None:
    global _active_job_id
    with app.app_context():
        try:
            _update_job(
                job_id,
                status="running",
                progress=5,
                stage="Lettura dati",
                message="Carico lo storico aggregato.",
                started_at=datetime.utcnow(),
                work_completed=0,
                work_total=0,
                work_label="Preparazione",
            )
            data = aggregated_from_db()
            if data.empty:
                raise RuntimeError("Nessun dato aggregato disponibile per il training.")

            def progress_callback(progress: int, stage: str, message: str, **work: Any) -> None:
                _checkpoint(job_id)
                _update_job(job_id, progress=progress, stage=stage, message=message, **work)
                _checkpoint(job_id)

            summary = train_all_models(data, model_folder, progress_callback=progress_callback)
            hierarchical_summary = train_hierarchical_models(data, model_folder, progress_callback=progress_callback)
            summary["hierarchical_training"] = hierarchical_summary
            _update_job(job_id, progress=95, stage="Salvataggio", message="Salvo modelli e metriche nel database.")
            save_training_summary(summary)
            db.session.commit()
            training_info = summary.get("training", {})
            training_info["hierarchical_models"] = summary.get("hierarchical_training", {}).get("models_trained", 0)
            _update_job(
                job_id,
                status="success",
                progress=100,
                stage="Completato",
                message=(
                    "Training completato: baseline, HistGradientBoosting e RandomForest aggiornati. "
                    f"Righe aggregate: {training_info.get('aggregated_rows', 'n/d')}. "
                    f"Modelli gerarchici: {training_info.get('hierarchical_models', 0)}."
                ),
                summary=training_info,
                finished_at=datetime.utcnow(),
            )
        except TrainingStopped as exc:
            db.session.rollback()
            _update_job(
                job_id,
                status="stopped",
                progress=int(_jobs.get(job_id, {}).get("progress") or 0),
                stage="Interrotto",
                message=str(exc),
                eta_seconds=0,
                eta_text="Interrotto",
                finished_at=datetime.utcnow(),
            )
        except Exception as exc:  # pragma: no cover - defensive background boundary
            db.session.rollback()
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
