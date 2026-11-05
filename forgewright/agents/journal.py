"""Durable stage receipts. An interrupted stage must be reconciled before replay."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class RecipeJournal:
    def __init__(self, root: Path, run_id: str, goal: str, recipe, seeds) -> None:
        identity = {"goal": goal, "stages": [(s.specialist_cls.role, s.run_kwargs) for s in recipe],
                    "seeds": [(a.kind, a.uri) for a in seeds]}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:24]
        self.path = root / f"{run_id}-{key}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"receipts": []}

    def save(self, **values) -> None:
        self.state.update(values)
        temp = self.path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(self.state, stream, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(self.path)
