"""Thin client for model-forge's introspection (`forge model describe --json`).

model-forge owns the architecture truth (see model_forge.introspect). This helper lets any
Forgewright specialist fetch a normalized ModelSpec dict for a checkpoint so it can derive
arch-correct targets (abliteration suffixes, LoRA modules, family config) instead of assuming
qwen-family naming. Returns None when introspection is unavailable (caller falls back safely).
"""
from __future__ import annotations

import json
from typing import Any, Optional


def model_spec(runner: Any, path: str) -> Optional[dict]:
    """Return the ModelSpec dict for ``path`` via `forge model describe --json`, or None."""
    if runner is None or not path:
        return None
    try:
        res = runner.run(f"model describe {path} --json")
    except Exception:  # noqa: BLE001
        return None
    if not getattr(res, "ok", False):
        return None
    out = (getattr(res, "output", "") or "").strip()
    for line in reversed(out.splitlines()):   # the JSON is the last line; tolerate banners above
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None
