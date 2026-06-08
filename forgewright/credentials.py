"""Persisted brain credentials: a small ``~/.forgewright/credentials.json`` written by the
first-run setup wizard (in the TUI) and read by the backend so the user configures their
OpenRouter key / Codex login once.

The file is intentionally tiny and shared by the Node TUI (writer) and the Python backend
(reader/fallback), so direct ``python -m forgewright`` usage benefits too. Secrets are stored
with 0600 permissions where the OS supports it.

Shape::

    {
      "brain": "openrouter:deepseek/deepseek-v4-pro" | "oauth-codex:gpt-5-codex" | ...,
      "openrouter_api_key": "sk-or-...",   # only when brain is openrouter
      "anthropic_api_key": "...",          # optional
      "openai_api_key": "..."              # optional
    }
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# brain-kind -> the env var its API key lives in (Codex/vLLM need no stored key here).
_KEY_FIELDS = {
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
}


def credentials_path() -> Path:
    home = Path(os.environ.get("FORGEWRIGHT_HOME", str(Path.home() / ".forgewright")))
    return home / "credentials.json"


def load_credentials() -> dict:
    p = credentials_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_credentials(data: dict) -> Path:
    p = credentials_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass  # best-effort on platforms without POSIX perms
    return p


def apply_credentials() -> Optional[str]:
    """Populate process env from saved API keys (without overriding ones already set) and
    return the saved ``brain`` shorthand if any. Safe to call when nothing is configured."""
    data = load_credentials()
    for field, env in _KEY_FIELDS.items():
        val = data.get(field)
        if val and not os.environ.get(env):
            os.environ[env] = val
    return data.get("brain")
