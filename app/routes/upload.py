from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from app import db
from app.models import UploadedFile
from app.services.database import aggregated_to_db, observations_to_db
from app.services.preprocessing import aggregate_observations, load_tabular_file, normalize_uploaded_dataframe

bp = Blueprint("upload", __name__)


@bp.get("/")
def index():
    return redirect(url_for("upload.upload"))


@bp.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        files = UploadedFile.query.order_by(UploadedFile.uploaded_at.desc()).limit(10).all()
        return render_template("upload.html", files=files)

    incoming = request.files.get("dataset")
    if not incoming or incoming.filename == "":
        flash("Seleziona un file CSV/XLSX.", "danger")
        return redirect(url_for("upload.upload"))

    filename = secure_filename(incoming.filename)
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    saved_path = upload_dir / filename
    incoming.save(saved_path)

    uploaded = UploadedFile(filename=str(saved_path), original_name=incoming.filename, status="uploaded")
    db.session.add(uploaded)
    db.session.commit()

    try:
        raw = load_tabular_file(saved_path)
        normalized = normalize_uploaded_dataframe(raw)
        aggregated = aggregate_observations(normalized)
        uploaded.row_count = len(normalized)
        uploaded.status = "imported"
        uploaded.message = f"Importate {len(normalized)} osservazioni, aggregate in {len(aggregated)} righe mensili."

        observations_to_db(normalized, uploaded)
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
        uploaded.status = "error"
        uploaded.message = str(exc)
        db.session.commit()
        flash(f"Errore import: {exc}", "danger")
        return redirect(url_for("upload.upload"))
