from __future__ import annotations

import hashlib
import io
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from app import APP_VERSION, db
from app.models import AggregatedObservation, Observation, Prediction, TrainedModel, UploadedFile, ValidationMetric
from app.services import dataset_store
from app.services.database import aggregated_to_db, observation_month_totals_from_db, observations_to_db
from app.services.diagnostics import build_diagnostic
from app.services.forecast_run import RESULT_FILE
from app.services.preprocessing import aggregate_observations, load_tabular_file, normalize_uploaded_dataframe

bp = Blueprint("upload", __name__)


@bp.get("/")
def index():
    return redirect(url_for("pipeline.wizard"))


@bp.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        files = UploadedFile.query.order_by(UploadedFile.uploaded_at.desc()).limit(10).all()
        has_imports = UploadedFile.query.filter_by(status="imported").count() > 0
        dataset_folder = current_app.config["DATASET_FOLDER"]
        validated_ids = {f.id for f in files if dataset_store.load_report(dataset_folder, f.id) is not None}
        return render_template("upload.html", files=files, has_imports=has_imports, validated_ids=validated_ids)

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

    dataset_folder = current_app.config["DATASET_FOLDER"]
    report = None
    try:
        raw = dataset_store.read_raw_file(saved_path)
        if dataset_store.is_mic_export(raw.columns):
            # Isolate-level export: validate and keep the full record for any model
            isolates, ast, report, issues = dataset_store.build_and_validate(raw)
            dataset_store.save_dataset(dataset_folder, uploaded_id, isolates, ast, report, issues)
            normalized = dataset_store.to_observations(isolates, ast)
        else:
            raw = load_tabular_file(saved_path)
            normalized = normalize_uploaded_dataframe(raw)
        uploaded.row_count = len(normalized)
        uploaded.status = "imported"

        observations_to_db(normalized, uploaded)
        db.session.flush()
        aggregated = aggregate_observations(observation_month_totals_from_db())
        validated = (
            f"{report['stats']['isolates_valid']} isolati validi su {report['stats']['rows_read']} righe, "
            if report else ""
        )
        uploaded.message = (
            f"{validated}Importate {len(normalized)} nuove osservazioni. "
            f"Storico aggregato: {len(aggregated)} righe mensili."
        )
        aggregated_to_db(aggregated)
        db.session.commit()
        flash(uploaded.message, "success")
        return render_template(
            "preview.html",
            uploaded=uploaded,
            report=report,
            columns=list(raw.columns),
            normalized_preview=normalized.head(20).to_dict(orient="records"),
            aggregated_preview=aggregated.head(20).to_dict(orient="records"),
        )
    except dataset_store.DatasetValidationError as exc:
        db.session.rollback()
        dataset_store.save_report(dataset_folder, uploaded_id, exc.report)
        uploaded = db.session.get(UploadedFile, uploaded_id)
        uploaded.status = "error"
        uploaded.message = f"Validazione fallita: {exc}"
        db.session.commit()
        flash(uploaded.message, "danger")
        return redirect(url_for("upload.validation_report", upload_id=uploaded_id))
    except Exception as exc:
        db.session.rollback()
        dataset_store.delete_upload_data(dataset_folder, uploaded_id)
        uploaded = db.session.get(UploadedFile, uploaded_id)
        uploaded.status = "error"
        uploaded.message = str(exc)
        db.session.commit()
        flash(f"Errore import: {exc}", "danger")
        return redirect(url_for("upload.upload"))


@bp.get("/diagnostico.xlsx")
def diagnostic():
    uploads = [
        {"id": u.id, "original_name": u.original_name, "status": u.status, "row_count": u.row_count,
         "uploaded_at": u.uploaded_at.strftime("%d/%m/%Y") if u.uploaded_at else ""}
        for u in UploadedFile.query.order_by(UploadedFile.uploaded_at.asc()).all()
    ]
    content = build_diagnostic(current_app.config["DATASET_FOLDER"], uploads, APP_VERSION)
    return send_file(
        io.BytesIO(content),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"diagnostico_mic_res_{datetime.now():%Y%m%d_%H%M}.xlsx",
    )


@bp.get("/upload/<int:upload_id>/validazione")
def validation_report(upload_id: int):
    uploaded = db.session.get(UploadedFile, upload_id) or abort(404)
    report = dataset_store.load_report(current_app.config["DATASET_FOLDER"], upload_id) or abort(404)
    has_issues = dataset_store.issues_path(current_app.config["DATASET_FOLDER"], upload_id).exists()
    return render_template("validation.html", uploaded=uploaded, report=report, has_issues=has_issues)


@bp.get("/upload/<int:upload_id>/anomalie.csv")
def validation_issues(upload_id: int):
    path = dataset_store.issues_path(current_app.config["DATASET_FOLDER"], upload_id)
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name=f"anomalie_upload_{upload_id}.csv")


@bp.post("/reset-db")
def reset_db():
    Prediction.query.delete()
    ValidationMetric.query.delete()
    TrainedModel.query.delete()
    AggregatedObservation.query.delete()
    Observation.query.delete()
    UploadedFile.query.delete()
    db.session.commit()

    shutil.rmtree(current_app.config["DATASET_FOLDER"], ignore_errors=True)
    Path(current_app.config["DATASET_FOLDER"]).mkdir(parents=True, exist_ok=True)
    (Path(current_app.config["FORECAST_FOLDER"]) / RESULT_FILE).unlink(missing_ok=True)

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
