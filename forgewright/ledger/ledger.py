"""Append-only JSONL run ledger: provenance + the source of truth for crash-resume.

Every meaningful event (goal, assistant turn, tool call, stage result, gate decision)
is appended as one JSON line. The agent can re-hydrate state from this after a restart.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _default_dir() -> Path:
    return Path(os.environ.get("FORGEWRIGHT_HOME", str(Path.home() / ".forgewright"))) / "ledger"


class Ledger:
    def __init__(self, run_id: str, ledger_dir: Path | None = None) -> None:
        base = ledger_dir or _default_dir()
        base.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.path = base / f"{run_id}.jsonl"

    def event(self, kind: str, **data: Any) -> None:
        rec = {"ts": time.time(), "run_id": self.run_id, "kind": kind, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
