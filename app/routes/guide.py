"""Home page: the guided path (carica -> controlla -> calcola -> leggi)."""
from __future__ import annotations

from flask import Blueprint, current_app, render_template

from app.models import UploadedFile
from app.services.workflow import workflow_state

bp = Blueprint("guide", __name__)


def current_workflow() -> dict:
    uploads = UploadedFile.query.filter_by(status="imported").order_by(UploadedFile.uploaded_at.asc()).all()
    return workflow_state(current_app.config["DATASET_FOLDER"], current_app.config["FORECAST_FOLDER"], uploads)


@bp.get("/")
def home():
    return render_template("guide.html", wf=current_workflow())
