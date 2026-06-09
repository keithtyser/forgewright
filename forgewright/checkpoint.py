"""Checkpoint/resume of the agent loop.

Artifacts are already durable (the registry) and every event is on the ledger, but the LOOP'S
OWN position is not - a crash mid-run restarts the reasoning from scratch and re-pays for the
context it had already built. This snapshots the live working state (the context window: pinned
goal + running summary + recent turns, plus the step counter and the doom-loop signatures) to a
small JSON file after each step, so a re-launched run with the same run_id picks up where it
stopped instead of re-deriving everything.

It is intentionally lightweight: one file per run, last-write-wins, best-effort (a failed
checkpoint never breaks the loop). The registry/ledger remain the source of truth for provenance;
this is purely a resume optimization for long runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def _default_dir() -> Path:
    base = Path(os.environ.get("FORGEWRIGHT_HOME", str(Path.home() / ".forgewright")))
    return base / "checkpoints"


class Checkpoint:
    def __init__(self, run_id: str, checkpoint_dir: Path | None = None) -> None:
        base = checkpoint_dir or _default_dir()
        base.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.path = base / f"{run_id}.json"

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, state: dict) -> None:
        """Atomically persist the loop state (best-effort; never raises into the loop)."""
        try:
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, default=str), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 - checkpointing must never break the run
            pass

    def load(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt checkpoint is treated as no checkpoint
            return None

    def clear(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
