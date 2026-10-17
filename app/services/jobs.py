"""Long operations run on the server in a thread, independent of the browser.

One job at a time (they all read or rewrite the archive), kept in memory: the
app runs as a single process (gunicorn, one worker), and a restart drops a job
in progress, which the operator can simply start again. Every page shows the
running job (base.html) and the outcome of the last one for a while after.
"""
from __future__ import annotations

import threading
import traceback
from datetime import datetime, timedelta
from typing import Any, Callable

from flask import Flask

LABELS = {
    "cruscotto": "Scaricamento dal cruscotto regionale",
    "previsioni": "Calcolo delle previsioni",
    "verifica": "Verifica sui dati nuovi",
    "ricerca": "Ricerca dei modelli",
}
SHOW_FINISHED_FOR = timedelta(minutes=30)

# work(progress) -> fields merged into the job: message, link_endpoint, link_args, ...
Progress = Callable[[str, int | None], None]
Work = Callable[[Progress], dict[str, Any]]

_jobs: dict[str, dict[str, Any]] = {}
_threads: dict[str, threading.Thread] = {}
_lock = threading.Lock()


def start(kind: str, app: Flask, work: Work) -> dict[str, Any] | None:
    """Start a job; returns the job already running instead, if any (nothing started)."""
    with _lock:
        busy = _running_unlocked()
        if busy:
            return dict(busy)
        _jobs[kind] = {
            "kind": kind, "label": LABELS[kind], "status": "running", "progress": None,
            "message": "Avvio in corso.", "error": "", "started_at": datetime.now(), "finished_at": None,
            "link_endpoint": None, "link_args": {},
        }
        _threads[kind] = threading.Thread(target=_run, args=(kind, app, work), daemon=True)
    _threads[kind].start()
    return None


def get(kind: str) -> dict[str, Any] | None:
    with _lock:
        return dict(_jobs[kind]) if kind in _jobs else None


def shown() -> dict[str, Any] | None:
    """The running job, else the last one finished in the last SHOW_FINISHED_FOR."""
    with _lock:
        job = _running_unlocked()
        if job:
            return dict(job)
        finished = [j for j in _jobs.values() if j["finished_at"] and
                    datetime.now() - j["finished_at"] <= SHOW_FINISHED_FOR]
        return dict(max(finished, key=lambda j: j["finished_at"])) if finished else None


def status(job: dict[str, Any] | None) -> dict[str, Any]:
    """JSON-safe view for the polling script."""
    if not job:
        return {"status": "none"}
    minutes = int(((job["finished_at"] or datetime.now()) - job["started_at"]).total_seconds() // 60)
    return {"kind": job["kind"], "label": job["label"], "status": job["status"], "progress": job["progress"],
            "message": job["message"], "minutes": minutes}


def wait(kind: str, timeout: float | None = None) -> None:
    thread = _threads.get(kind)
    if thread:
        thread.join(timeout)


def _running_unlocked() -> dict[str, Any] | None:
    return next((j for j in _jobs.values() if j["status"] == "running"), None)


def _update(kind: str, **fields: Any) -> None:
    with _lock:
        _jobs[kind].update(fields)


def _run(kind: str, app: Flask, work: Work) -> None:
    def progress(message: str, percent: int | None = None) -> None:
        _update(kind, message=message, progress=percent)

    with app.app_context():
        try:
            outcome = work(progress)
            _update(kind, status="success", progress=100, finished_at=datetime.now(), **outcome)
        except Exception as exc:
            _update(kind, status="error", message=str(exc) or exc.__class__.__name__,
                    error=traceback.format_exc(), finished_at=datetime.now(),
                    **getattr(exc, "job_fields", {}))
