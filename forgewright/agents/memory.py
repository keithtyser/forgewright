"""Outcome / experience memory - the learning loop across runs.

Every stage the swarm runs already produces a gated Artifact (pass/fail + scores + the chosen
hyperparameters), but nothing reads those outcomes back: each run starts cold. This closes that
loop. The Director records the outcome of every gated stage here; future runs then:
  - ground the PLANNER with a short digest of what has worked / failed for this kind of goal, and
  - let REPAIR policies prefer hyperparameters that historically PASSED the gate for the same
    (stage, family) instead of re-discovering them from scratch.

Storage is an append-only JSONL ledger (one outcome per line) under ~/.forgewright/memory, so it
is durable, inspectable, and itself a dataset of (stage, params -> gate result) pairs for later
training. Reads are tolerant: a missing/garbled line is skipped, never fatal.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional


def _default_path() -> Path:
    base = Path(os.environ.get("FORGEWRIGHT_HOME", str(Path.home() / ".forgewright")))
    return base / "memory" / "outcomes.jsonl"

# the hyperparameters worth remembering per stage (what a repair policy would tune)
_TUNABLES = ("strength", "layer_skip_first", "layer_skip_last", "max_steps", "method", "objective")


class OutcomeMemory:
    """Append-only store + read helpers over past stage outcomes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- write ---------------------------------------------------------------------
    def record(self, *, stage: str, family: str, params: dict, passed: Optional[bool],
               metrics: Optional[dict] = None, verdict: str = "", artifact_id: str = "",
               attempt: int = 1, run_id: str = "") -> None:
        """Persist one stage outcome (best-effort; a write failure never breaks the run)."""
        rec = {
            "ts": time.time(), "stage": stage, "family": family or "",
            "params": {k: params[k] for k in _TUNABLES if k in (params or {})},
            "passed": passed, "metrics": metrics or {}, "verdict": verdict,
            "artifact_id": artifact_id, "attempt": attempt, "run_id": run_id,
        }
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:  # noqa: BLE001 - memory is an optimization, never a blocker
            pass

    # --- read ----------------------------------------------------------------------
    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for ln in self.path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out

    def recall(self, *, stage: Optional[str] = None, family: Optional[str] = None,
               passed: Optional[bool] = None) -> list[dict]:
        """Past outcomes filtered by stage/family/pass, newest first. Insertion order (the file's
        append order) breaks ts ties so ordering is deterministic even at sub-clock resolution."""
        rows = [
            (idx, r) for idx, r in enumerate(self.all())
            if (stage is None or r.get("stage") == stage)
            and (family is None or r.get("family") == family)
            and (passed is None or r.get("passed") is passed)
        ]
        rows.sort(key=lambda ir: (ir[1].get("ts", 0), ir[0]), reverse=True)
        return [r for _, r in rows]

    def best_params(self, *, stage: str, family: str) -> Optional[dict]:
        """The hyperparameters from the most recent PASSING run of (stage, family), if any.
        This is what a repair policy seeds its next attempt with."""
        passing = self.recall(stage=stage, family=family, passed=True)
        for r in passing:
            if r.get("params"):
                return dict(r["params"])
        return None

    def digest(self, *, family: Optional[str] = None, limit: int = 8) -> str:
        """A short, human-readable summary of recent outcomes for grounding the planner.
        Empty string when there is no history (so callers can cheaply skip injecting it)."""
        rows = self.recall(family=family)[:limit]
        if not rows:
            return ""
        lines = []
        for r in rows:
            verdict = "PASS" if r.get("passed") else ("FAIL" if r.get("passed") is False else "?")
            params = ", ".join(f"{k}={v}" for k, v in (r.get("params") or {}).items())
            fam = r.get("family") or "?"
            tail = f" [{params}]" if params else ""
            note = f" - {r['verdict']}" if r.get("verdict") and not r.get("passed") else ""
            lines.append(f"  {r.get('stage','?')}({fam}): {verdict}{tail}{note}")
        return "Past outcomes (most recent first):\n" + "\n".join(lines)
