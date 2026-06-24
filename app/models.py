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


class Observation(db.Model):
    __tablename__ = "observations"

    id = db.Column(db.Integer, primary_key=True)
    uploaded_file_id = db.Column(db.Integer, db.ForeignKey("uploaded_files.id"), nullable=True)
    observed_at = db.Column(db.Date, nullable=False, index=True)
    pathogen = db.Column(db.String(120), nullable=False, index=True)
    antibiotic = db.Column(db.String(80), nullable=False, index=True)
    laboratory = db.Column(db.String(80), nullable=False, index=True)
    ward = db.Column(db.String(120), nullable=True, index=True)
    result = db.Column(db.String(1), nullable=True)
    samples = db.Column(db.Integer, default=1, nullable=False)
    sensitive_count = db.Column(db.Float, default=0, nullable=False)
    intermediate_count = db.Column(db.Float, default=0, nullable=False)
    resistant_count = db.Column(db.Float, default=0, nullable=False)
    sensitive_pct = db.Column(db.Float, nullable=True)
    intermediate_pct = db.Column(db.Float, nullable=True)
    resistant_pct = db.Column(db.Float, nullable=True)


class AggregatedObservation(db.Model):
    __tablename__ = "aggregated_observations"

    id = db.Column(db.Integer, primary_key=True)
    month = db.Column(db.Date, nullable=False, index=True)
    pathogen = db.Column(db.String(120), nullable=False, index=True)
    antibiotic = db.Column(db.String(80), nullable=False, index=True)
    laboratory = db.Column(db.String(80), nullable=False, index=True)
    ward = db.Column(db.String(120), nullable=True, index=True)
    samples = db.Column(db.Integer, default=0, nullable=False)
    sensitive_count = db.Column(db.Float, default=0, nullable=False)
    intermediate_count = db.Column(db.Float, default=0, nullable=False)
    resistant_count = db.Column(db.Float, default=0, nullable=False)
    sensitive_pct = db.Column(db.Float, default=0, nullable=False)
    intermediate_pct = db.Column(db.Float, default=0, nullable=False)
    resistant_pct = db.Column(db.Float, default=0, nullable=False)


class TrainedModel(db.Model):
    __tablename__ = "trained_models"

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(80), nullable=False, index=True)
    target = db.Column(db.String(40), nullable=False, index=True)
    artifact_path = db.Column(db.String(500), nullable=False)
    trained_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    scope = db.Column(db.String(80), default="regional", nullable=False)
    metadata_json = db.Column(db.Text, default="{}", nullable=False)


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    prediction_month = db.Column(db.Date, nullable=False, index=True)
    pathogen = db.Column(db.String(120), nullable=False)
    antibiotic = db.Column(db.String(80), nullable=False)
    laboratory = db.Column(db.String(80), nullable=False)
    ward = db.Column(db.String(120), nullable=True)
    model_name = db.Column(db.String(80), nullable=False)
    quantitative_model = db.Column(db.String(120), default="", nullable=False)
    decision_model = db.Column(db.String(120), default="", nullable=False)
    decision_class = db.Column(db.String(1), default="", nullable=False)
    decision_confidence = db.Column(db.Float, nullable=True)
    sensitive_pct = db.Column(db.Float, nullable=False)
    intermediate_pct = db.Column(db.Float, nullable=False)
    resistant_pct = db.Column(db.Float, nullable=False)
    reliability = db.Column(db.String(20), nullable=False)
    reliability_reason = db.Column(db.Text, default="", nullable=False)


class ValidationMetric(db.Model):
    __tablename__ = "validation_metrics"

    id = db.Column(db.Integer, primary_key=True)
    trained_model_id = db.Column(db.Integer, db.ForeignKey("trained_models.id"), nullable=True)
    model_name = db.Column(db.String(80), nullable=False, index=True)
    target = db.Column(db.String(40), nullable=False)
    mae = db.Column(db.Float, nullable=True)
    rmse = db.Column(db.Float, nullable=True)
    mape = db.Column(db.Float, nullable=True)
    accuracy = db.Column(db.Float, nullable=True)
    f1_macro = db.Column(db.Float, nullable=True)
    metadata_json = db.Column(db.Text, default="{}", nullable=False)


class Breakpoint(db.Model):
    __tablename__ = "breakpoints"

    id = db.Column(db.Integer, primary_key=True)
    pathogen = db.Column(db.String(120), nullable=False, index=True)
    antibiotic = db.Column(db.String(80), nullable=False, index=True)
    mic_value = db.Column(db.String(40), nullable=True)
    interpretation = db.Column(db.String(1), nullable=False)
    source = db.Column(db.String(120), nullable=True)
