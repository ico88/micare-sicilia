"""Consultation dashboard: the archive sliced like the regional Qlik dashboard, aggregated."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from flask import Blueprint, current_app, jsonify, render_template, request, send_file

from app.services import explore, jobs

bp = Blueprint("explore", __name__)

FILTER_KEYS = explore.DIMENSIONS + ["antibiotico", "da", "a", "lettura"]


def _folders() -> tuple[str, str]:
    return current_app.config["DATASET_FOLDER"], current_app.config["EXPLORE_FOLDER"]


def _filters() -> dict:
    return {key: request.args.get(key, "").strip() for key in FILTER_KEYS}


def _prepare(app, progress) -> dict:
    progress("Conteggio degli isolati e degli antibiogrammi per laboratorio, paziente e trimestre.")
    results, _ = explore.load_cube(app.config["DATASET_FOLDER"], app.config["EXPLORE_FOLDER"])
    if results.empty:
        raise RuntimeError("Nessun dato in archivio: carica prima l'export della Rete MIC.")
    return {"message": "Dati pronti per la consultazione.", "link_endpoint": "explore.dashboard"}


@bp.get("/consultazione")
def dashboard():
    dataset, cube = _folders()
    has_data = bool(explore.archive_signature(dataset).count("upload_"))
    ready = has_data and explore.cube_ready(dataset, cube)
    busy = None
    if has_data and not ready:
        app = current_app._get_current_object()
        busy = jobs.start("consultazione", app, lambda progress: _prepare(app, progress))
    return render_template("explore.html", has_data=has_data, ready=ready, busy=busy,
                           labels=explore.FILTER_LABELS, readings=explore.READING_LABELS)


@bp.get("/consultazione/dati")
def data():
    dataset, cube = _folders()
    if not explore.cube_ready(dataset, cube):
        return jsonify({"vuoto": True, "in_preparazione": True})
    results, counted = explore.load_cube(dataset, cube)
    return jsonify(explore.query(results, counted, _filters()))


@bp.get("/consultazione/dati.xlsx")
def download():
    dataset, cube = _folders()
    results, counted = explore.load_cube(dataset, cube)
    answer = explore.query(results, counted, _filters())
    if answer.get("vuoto"):
        return jsonify(answer), 404
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in explore.export_frames(answer).items():
            frame.to_excel(writer, sheet_name=name, index=False)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name=f"consultazione_retemic_{datetime.now():%Y%m%d_%H%M}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
