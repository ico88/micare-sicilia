from __future__ import annotations

from pathlib import Path

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text as sql_text

db = SQLAlchemy()


def _read_version() -> str:
    """Return version string: VERSION file + branch@sha if available."""
    try:
        v = (base_dir_for_version() / "VERSION").read_text().strip()
    except Exception:
        v = "dev"
    try:
        import subprocess
        cwd = str(base_dir_for_version())
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"{v} ({branch}@{sha})"
    except Exception:
        return v


def base_dir_for_version() -> Path:
    return Path(__file__).resolve().parent.parent


APP_VERSION: str = _read_version()


def create_app(test_config: dict | None = None) -> Flask:
    from .services.qlik_source import DASHBOARD_URL

    app = Flask(__name__, instance_relative_config=True)
    base_dir = Path(__file__).resolve().parent.parent
    instance_dir = base_dir / "instance"
    instance_dir.mkdir(exist_ok=True)

    app.config.from_mapping(
        SECRET_KEY="dev-change-me",
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{instance_dir / 'mic_res_sicilia.sqlite'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=str(base_dir / "data"),
        MODEL_FOLDER=str(base_dir / "models"),
        DATASET_FOLDER=str(instance_dir / "dataset"),
        FORECAST_FOLDER=str(instance_dir / "previsioni"),
        MAX_CONTENT_LENGTH=128 * 1024 * 1024,
        # Public Rete MIC dashboard (Qualità Sicilia SSR) the export is taken from
        FONTE_DATI_URL=DASHBOARD_URL,
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)
    Path(app.config["MODEL_FOLDER"]).mkdir(exist_ok=True)
    Path(app.config["DATASET_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["FORECAST_FOLDER"]).mkdir(parents=True, exist_ok=True)

    db.init_app(app)

    @app.context_processor
    def inject_version():
        return {"app_version": APP_VERSION, "fonte_dati_url": app.config["FONTE_DATI_URL"]}

    from . import models  # noqa: F401
    from .routes.dashboard import bp as dashboard_bp
    from .routes.forecasts import bp as forecasts_bp
    from .routes.guide import bp as guide_bp
    from .routes.pipeline import bp as pipeline_bp
    from .routes.upload import bp as upload_bp

    app.register_blueprint(guide_bp)
    app.register_blueprint(forecasts_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(pipeline_bp)

    with app.app_context():
        db.create_all()
        _ensure_schema_migrations()
        _mark_interrupted_runs()

    return app


def _mark_interrupted_runs() -> None:
    """Mark any pipeline runs left in running/queued state as interrupted on startup."""
    from .models import PipelineRun
    from datetime import datetime
    stale = PipelineRun.query.filter(PipelineRun.status.in_(["queued", "running"])).all()
    for r in stale:
        r.status = "interrupted"
        r.message = "Interrotto al riavvio del server."
        r.updated_at = datetime.utcnow()
    if stale:
        db.session.commit()


def _ensure_schema_migrations() -> None:
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()

    _migrate_table(inspector, tables, "predictions", {
        "quantitative_model": "VARCHAR(120) NOT NULL DEFAULT ''",
        "decision_model": "VARCHAR(120) NOT NULL DEFAULT ''",
        "decision_class": "VARCHAR(1) NOT NULL DEFAULT ''",
        "decision_confidence": "FLOAT",
        "ci_json": "TEXT",
    })
    _migrate_table(inspector, tables, "validation_metrics", {
        "mase": "FLOAT",
        "rmse_arima": "FLOAT",
        "rmse_glm": "FLOAT",
    })
    _migrate_table(inspector, tables, "aggregated_observations", {
        "pct_icu": "FLOAT",
        "pct_inpatient": "FLOAT",
    })
    db.session.commit()


def _migrate_table(inspector, tables: list[str], table: str, additions: dict[str, str]) -> None:
    if table not in tables:
        return
    existing = {col["name"] for col in inspector.get_columns(table)}
    for name, ddl in additions.items():
        if name not in existing:
            db.session.execute(sql_text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
