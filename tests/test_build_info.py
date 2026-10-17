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
