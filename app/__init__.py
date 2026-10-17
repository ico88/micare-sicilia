from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text as sql_text

db = SQLAlchemy()


def build_info() -> dict[str, str]:
    """Code version: from git, else from BUILD_INFO, which installer.sh and update.sh write
    (the service may run as a user git does not answer to)."""
    root = base_dir_for_version()
    try:
        import subprocess
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=str(root), stderr=subprocess.DEVNULL,
                                           timeout=5).decode().strip()
        return {"commit": git("rev-parse", "--short", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD")}
    except Exception:
        pass
    try:
        lines = (root / "BUILD_INFO").read_text().splitlines()
        return dict(line.split("=", 1) for line in lines if "=" in line)
    except OSError:
        return {}


def _read_version() -> str:
    """Return version string: VERSION file + branch@sha if available."""
    try:
        v = (base_dir_for_version() / "VERSION").read_text().strip()
    except Exception:
        v = "dev"
    info = build_info()
    return f"{v} ({info['branch']}@{info['commit']})" if info.get("commit") else v


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
        DATASET_FOLDER=str(instance_dir / "dataset"),
        FORECAST_FOLDER=str(instance_dir / "previsioni"),
        MAX_CONTENT_LENGTH=128 * 1024 * 1024,
        # Public Rete MIC dashboard (Qualità Sicilia SSR) the export is taken from
        FONTE_DATI_URL=DASHBOARD_URL,
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)
    Path(app.config["DATASET_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["FORECAST_FOLDER"]).mkdir(parents=True, exist_ok=True)

    db.init_app(app)

    @app.context_processor
    def inject_version():
        from .services import jobs
        return {"app_version": APP_VERSION, "fonte_dati_url": app.config["FONTE_DATI_URL"],
                "background_job": jobs.shown(), "now": datetime.now}

    from . import models  # noqa: F401
    from .routes.forecasts import bp as forecasts_bp
    from .routes.guide import bp as guide_bp
    from .routes.upload import bp as upload_bp

    app.register_blueprint(guide_bp)
    app.register_blueprint(forecasts_bp)
    app.register_blueprint(upload_bp)

    with app.app_context():
        db.create_all()
        _drop_legacy_tables()

    return app


# Tables of the earlier Prophet pipeline, removed: derived data only (observations copied
# from the archive, trained models, their predictions); the isolates stay in the archive
LEGACY_TABLES = ("observations", "aggregated_observations", "trained_models", "predictions",
                 "validation_metrics", "pipeline_runs", "combination_forecasts", "breakpoints")


def _drop_legacy_tables() -> None:
    existing = set(inspect(db.engine).get_table_names())
    for table in LEGACY_TABLES:
        if table in existing:
            db.session.execute(sql_text(f"DROP TABLE {table}"))
    db.session.commit()
