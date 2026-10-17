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
from app.models import UploadedFile
from app.services import dataset_store, model_search
from app.services.dashboard_jobs import start_dashboard_job
from app.services.diagnostics import build_diagnostic
from app.services.forecast_run import RESULT_FILE
from app.services.out_of_sample import RESULT_FILE as VALIDATION_FILE, load_validation
from app.routes.guide import current_workflow
from app.services.importing import import_file

bp = Blueprint("upload", __name__)


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
        flash("Seleziona l'export della Rete MIC (CSV o XLSX).", "danger")
        return redirect(url_for("guide.home"))

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
        return redirect(url_for("guide.home"))

    saved_path = upload_dir / f"{uuid.uuid4().hex}_{filename}"
    incoming.save(saved_path)

    try:
        outcome = import_file(saved_path, incoming.filename, current_app.config["DATASET_FOLDER"])
    except dataset_store.DatasetValidationError as exc:
        flash(f"Validazione fallita: {exc}", "danger")
        return redirect(url_for("upload.validation_report", upload_id=exc.upload_id))
    except Exception as exc:
        flash(f"Errore import: {exc}", "danger")
        return redirect(url_for("guide.home"))

    flash(outcome.message, "success")
    # Guided path: the next thing to do is read the outcome of the check
    return redirect(url_for("upload.validation_report", upload_id=outcome.upload_id))


@bp.post("/upload/cruscotto")
def dashboard_refresh():
    include_manual = request.form.get("sostituisci_manuali") == "1"
    busy = start_dashboard_job(current_app._get_current_object(), include_manual)
    if busy:
        flash(f"È già in corso: {busy['label'].lower()}. Attendi che finisca, poi riprova.", "warning")
    return redirect(url_for("guide.home") + "#passo-carica")


@bp.get("/diagnostico.xlsx")
def diagnostic():
    uploads = [
        {"id": u.id, "original_name": u.original_name, "status": u.status, "row_count": u.row_count,
         "uploaded_at": u.uploaded_at.strftime("%d/%m/%Y") if u.uploaded_at else ""}
        for u in UploadedFile.query.order_by(UploadedFile.uploaded_at.asc()).all()
    ]
    content = build_diagnostic(current_app.config["DATASET_FOLDER"], uploads, APP_VERSION,
                               load_validation(current_app.config["FORECAST_FOLDER"]))
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
    return render_template("validation.html", uploaded=uploaded, report=report, has_issues=has_issues,
                           wf=current_workflow())


@bp.get("/upload/<int:upload_id>/anomalie.csv")
def validation_issues(upload_id: int):
    path = dataset_store.issues_path(current_app.config["DATASET_FOLDER"], upload_id)
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name=f"anomalie_upload_{upload_id}.csv")


@bp.post("/reset-db")
def reset_db():
    UploadedFile.query.delete()
    db.session.commit()
    shutil.rmtree(current_app.config["DATASET_FOLDER"], ignore_errors=True)
    Path(current_app.config["DATASET_FOLDER"]).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(current_app.config["EXPLORE_FOLDER"], ignore_errors=True)
    folder = Path(current_app.config["FORECAST_FOLDER"])
    for name in (RESULT_FILE, VALIDATION_FILE, model_search.SUMMARY_FILE, model_search.WORKBOOK_FILE):
        (folder / name).unlink(missing_ok=True)
    flash("Archivio azzerato. Puoi ricaricare l'export della Rete MIC.", "success")
    return redirect(url_for("guide.home"))


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
