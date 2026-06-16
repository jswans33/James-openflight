"""Tests for the openflight-cloud CLI argument wiring."""

import pytest

from openflight.cloud import cli


class TestArgParsing:
    def test_requires_subcommand(self, capsys):
        rc = cli.main([])
        assert rc != 0

    def test_unknown_subcommand_errors(self):
        with pytest.raises(SystemExit):
            cli.main(["frobnicate"])


class TestDispatch:
    def test_status_dispatches(self, monkeypatch, tmp_path):
        called = {}

        def fake_status(config, log_dir, client=None, out=print):
            called["log_dir"] = log_dir
            out("status-ran")
            return {}

        monkeypatch.setattr(cli.commands, "cmd_status", fake_status)
        rc = cli.main(["status", "--log-dir", str(tmp_path), "--config", str(tmp_path / "c.json")])
        assert rc == 0
        assert called["log_dir"] == tmp_path

    def test_push_passes_dry_run_flag(self, monkeypatch, tmp_path):
        captured = {}

        def fake_push(config, log_dir, client, dry_run=False, retry=False, session=None, out=print):
            captured["dry_run"] = dry_run
            return {"needs_relink": False}

        monkeypatch.setattr(cli.commands, "cmd_push", fake_push)
        cli.main(
            ["push", "--dry-run", "--log-dir", str(tmp_path), "--config", str(tmp_path / "c.json")]
        )
        assert captured["dry_run"] is True

    def test_push_retry_all(self, monkeypatch, tmp_path):
        captured = {}

        def fake_push(config, log_dir, client, dry_run=False, retry=False, session=None, out=print):
            captured.update(retry=retry, session=session)
            return {"needs_relink": False}

        monkeypatch.setattr(cli.commands, "cmd_push", fake_push)
        cli.main(
            ["push", "--retry", "--log-dir", str(tmp_path), "--config", str(tmp_path / "c.json")]
        )
        assert captured == {"retry": True, "session": None}

    def test_push_retry_named_session(self, monkeypatch, tmp_path):
        captured = {}

        def fake_push(config, log_dir, client, dry_run=False, retry=False, session=None, out=print):
            captured.update(retry=retry, session=session)
            return {"needs_relink": False}

        monkeypatch.setattr(cli.commands, "cmd_push", fake_push)
        cli.main(
            [
                "push",
                "--retry",
                "session_20260527",
                "--log-dir",
                str(tmp_path),
                "--config",
                str(tmp_path / "c.json"),
            ]
        )
        assert captured == {"retry": True, "session": "session_20260527"}

    def test_push_no_retry_by_default(self, monkeypatch, tmp_path):
        captured = {}

        def fake_push(config, log_dir, client, dry_run=False, retry=False, session=None, out=print):
            captured.update(retry=retry, session=session)
            return {"needs_relink": False}

        monkeypatch.setattr(cli.commands, "cmd_push", fake_push)
        cli.main(["push", "--log-dir", str(tmp_path), "--config", str(tmp_path / "c.json")])
        assert captured == {"retry": False, "session": None}

    def test_push_returns_nonzero_when_relink_needed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli.commands, "cmd_push", lambda *a, **k: {"needs_relink": True})
        rc = cli.main(["push", "--log-dir", str(tmp_path), "--config", str(tmp_path / "c.json")])
        assert rc != 0

    def test_link_dispatches(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli.commands, "cmd_link", lambda *a, **k: True)
        rc = cli.main(["link", "--config", str(tmp_path / "c.json")])
        assert rc == 0

    def test_link_returns_nonzero_on_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli.commands, "cmd_link", lambda *a, **k: False)
        rc = cli.main(["link", "--config", str(tmp_path / "c.json")])
        assert rc != 0
