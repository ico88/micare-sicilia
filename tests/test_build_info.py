import subprocess

import app as app_module


def test_version_from_build_info_when_git_does_not_answer(tmp_path, monkeypatch):
    (tmp_path / "BUILD_INFO").write_text("commit=abc1234\nbranch=claude/zen-brown-p4pma6\ndate=2026-10-17T06:00:00Z\n")
    (tmp_path / "VERSION").write_text("1.1.0\n")
    monkeypatch.setattr(app_module, "base_dir_for_version", lambda: tmp_path)

    def refuse(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")  # e.g. "dubious ownership" for the service user

    monkeypatch.setattr(subprocess, "check_output", refuse)
    assert app_module.build_info()["commit"] == "abc1234"
    assert app_module._read_version() == "1.1.0 (claude/zen-brown-p4pma6@abc1234)"

    from app.services.diagnostics import _git_commit
    assert _git_commit() == "abc1234"


def test_version_without_git_or_build_info(tmp_path, monkeypatch):
    (tmp_path / "VERSION").write_text("1.1.0\n")
    monkeypatch.setattr(app_module, "base_dir_for_version", lambda: tmp_path)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert app_module._read_version() == "1.1.0"


def test_tables_of_the_earlier_pipeline_are_dropped(tmp_path):
    import sqlite3

    from sqlalchemy import inspect

    from app import create_app, db

    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as con:  # a database left by the earlier version
        con.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, pathogen TEXT)")
        con.execute("INSERT INTO observations (pathogen) VALUES ('ESCCOL')")
        con.execute("CREATE TABLE trained_models (id INTEGER PRIMARY KEY)")
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{path}",
                      "UPLOAD_FOLDER": str(tmp_path / "d"), "DATASET_FOLDER": str(tmp_path / "ds"),
                      "FORECAST_FOLDER": str(tmp_path / "p")})
    with app.app_context():
        tables = set(inspect(db.engine).get_table_names())
        db.session.remove()
        db.engine.dispose()
    assert "observations" not in tables and "trained_models" not in tables and "uploaded_files" in tables
