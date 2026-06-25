from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from app import db
from app.models import AggregatedObservation, Observation, Prediction, TrainedModel, UploadedFile, ValidationMetric
from app.services.database import aggregated_to_db, observations_from_db, observations_to_db
from app.services.preprocessing import aggregate_observations, load_tabular_file, normalize_uploaded_dataframe

bp = Blueprint("upload", __name__)


@bp.get("/")
def index():
    return redirect(url_for("upload.upload"))


@bp.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        files = UploadedFile.query.order_by(UploadedFile.uploaded_at.desc()).limit(10).all()
        has_imports = UploadedFile.query.filter_by(status="imported").count() > 0
        return render_template("upload.html", files=files, has_imports=has_imports)

    incoming = request.files.get("dataset")
    if not incoming or incoming.filename == "":
        flash("Seleziona un file CSV/XLSX.", "danger")
        return redirect(url_for("upload.upload"))

    filename = secure_filename(incoming.filename)
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    file_hash = _uploaded_file_hash(incoming)
    duplicate = _find_duplicate_import(file_hash)

    if duplicate is not None:
        uploaded = UploadedFile(
            filename=str(duplicate.filename),
            original_name=incoming.filename,
            status="skipped",
            message=f"File gia importato: {duplicate.original_name}. Importazione saltata.",
        )
        db.session.add(uploaded)
        db.session.commit()
        flash(uploaded.message, "warning")
        return redirect(url_for("upload.upload"))

    saved_path = upload_dir / f"{uuid.uuid4().hex}_{filename}"
    incoming.save(saved_path)

    uploaded = UploadedFile(filename=str(saved_path), original_name=incoming.filename, status="uploaded")
    db.session.add(uploaded)
    db.session.commit()
    uploaded_id = uploaded.id

    try:
        raw = load_tabular_file(saved_path)
        normalized = normalize_uploaded_dataframe(raw)
        uploaded.row_count = len(normalized)
        uploaded.status = "imported"

        observations_to_db(normalized, uploaded)
        db.session.flush()
        all_observations = observations_from_db()
        aggregated = aggregate_observations(all_observations)
        uploaded.message = (
            f"Importate {len(normalized)} nuove osservazioni. "
            f"Storico aggregato: {len(aggregated)} righe mensili."
        )
        aggregated_to_db(aggregated)
        db.session.commit()
        flash(uploaded.message, "success")
        return render_template(
            "preview.html",
            uploaded=uploaded,
            columns=list(raw.columns),
            normalized_preview=normalized.head(20).to_dict(orient="records"),
            aggregated_preview=aggregated.head(20).to_dict(orient="records"),
        )
    except Exception as exc:
        db.session.rollback()
        uploaded = db.session.get(UploadedFile, uploaded_id)
        uploaded.status = "error"
        uploaded.message = str(exc)
        db.session.commit()
        flash(f"Errore import: {exc}", "danger")
        return redirect(url_for("upload.upload"))


@bp.post("/reset-db")
def reset_db():
    Prediction.query.delete()
    ValidationMetric.query.delete()
    TrainedModel.query.delete()
    AggregatedObservation.query.delete()
    Observation.query.delete()
    UploadedFile.query.delete()
    db.session.commit()

    # cancella file modello su disco
    model_folder = Path(current_app.config["MODEL_FOLDER"])
    for item in model_folder.iterdir():
        if item.is_file():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            shutil.rmtree(item, ignore_errors=True)

    flash("Database e modelli azzerati. Puoi ricaricare i file Excel.", "success")
    return redirect(url_for("upload.upload"))


def _uploaded_file_hash(incoming) -> str:
    digest = hashlib.sha256()
    incoming.stream.seek(0)
    for chunk in iter(lambda: incoming.stream.read(1024 * 1024), b""):
        digest.update(chunk)
    incoming.stream.seek(0)
    return digest.hexdigest()


def _file_hash(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_duplicate_import(file_hash: str) -> UploadedFile | None:
    imported_files = UploadedFile.query.filter_by(status="imported").order_by(UploadedFile.uploaded_at.desc()).all()
    for uploaded in imported_files:
        if _file_hash(uploaded.filename) == file_hash:
            return uploaded
    return None
