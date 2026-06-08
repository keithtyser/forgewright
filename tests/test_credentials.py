"""Unit tests for persisted brain credentials (the setup-wizard store)."""
from __future__ import annotations

import importlib

import forgewright.credentials as creds


def _reload_with_home(monkeypatch, tmp_path):
    monkeypatch.setenv("FORGEWRIGHT_HOME", str(tmp_path))
    importlib.reload(creds)
    return creds


def test_save_load_roundtrip(monkeypatch, tmp_path):
    c = _reload_with_home(monkeypatch, tmp_path)
    assert c.load_credentials() == {}
    c.save_credentials({"brain": "oauth-codex", "openrouter_api_key": "sk-or-x"})
    assert c.credentials_path().exists()
    assert c.load_credentials()["brain"] == "oauth-codex"


def test_apply_populates_env_without_override(monkeypatch, tmp_path):
    c = _reload_with_home(monkeypatch, tmp_path)
    c.save_credentials(
        {"brain": "openrouter:deepseek/deepseek-v4-pro", "openrouter_api_key": "sk-or-saved"}
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    brain = c.apply_credentials()
    assert brain == "openrouter:deepseek/deepseek-v4-pro"
    assert c.os.environ["OPENROUTER_API_KEY"] == "sk-or-saved"

    # an existing env var is not overridden
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-live")
    c.apply_credentials()
    assert c.os.environ["OPENROUTER_API_KEY"] == "sk-or-live"


def test_apply_no_file_is_noop(monkeypatch, tmp_path):
    c = _reload_with_home(monkeypatch, tmp_path)
    assert c.apply_credentials() is None


def test_load_tolerates_corrupt_json(monkeypatch, tmp_path):
    c = _reload_with_home(monkeypatch, tmp_path)
    c.credentials_path().parent.mkdir(parents=True, exist_ok=True)
    c.credentials_path().write_text("{not json", encoding="utf-8")
    assert c.load_credentials() == {}
