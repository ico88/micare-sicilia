from __future__ import annotations

from datetime import datetime

from . import db


class UploadedFile(db.Model):
    __tablename__ = "uploaded_files"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    row_count = db.Column(db.Integer, default=0, nullable=False)
    status = db.Column(db.String(40), default="uploaded", nullable=False)
    message = db.Column(db.Text, default="", nullable=False)
