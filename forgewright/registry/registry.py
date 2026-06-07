"""Append-only artifact registry (JSONL) with lineage.

One artifact per line. `register` appends; reads rebuild typed artifacts from the file.
Latest-wins by `created_at` for `latest()`. `lineage` walks `parents` to the roots.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from forgewright.contracts import Artifact, artifact_from_dict


def _default_path() -> Path:
    home = Path(os.environ.get("FORGEWRIGHT_HOME", str(Path.home() / ".forgewright")))
    return home / "registry" / "artifacts.jsonl"


class Registry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- write ---------------------------------------------------------------------

    def register(self, artifact: Artifact) -> Artifact:
        """Append an artifact and return it (id assigned in __post_init__)."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(artifact.to_dict(), default=str) + "\n")
        return artifact

    # --- read ----------------------------------------------------------------------

    def all(self) -> list[Artifact]:
        if not self.path.exists():
            return []
        return [
            artifact_from_dict(json.loads(ln))
            for ln in self.path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    def get(self, artifact_id: str) -> Optional[Artifact]:
        # last write for an id wins (artifacts are immutable, but re-registration is allowed)
        found = None
        for a in self.all():
            if a.id == artifact_id:
                found = a
        return found

    def latest(self, kind: str | None = None, **meta_filters: Any) -> Optional[Artifact]:
        """Most-recent artifact of `kind` whose meta matches all `meta_filters`. Ties on
        `created_at` (coarse clocks) are broken by append order — last registered wins."""
        candidates = [
            (idx, a) for idx, a in enumerate(self.all())
            if (kind is None or a.kind == kind)
            and all(a.meta.get(k) == v for k, v in meta_filters.items())
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda ia: (ia[1].created_at, ia[0]))[1]

    def lineage(self, artifact_id: str) -> list[Artifact]:
        """The artifact and its ancestors (via `parents`), de-duped, leaf-first."""
        by_id = {a.id: a for a in self.all()}
        order: list[Artifact] = []
        seen: set[str] = set()
        stack = [artifact_id]
        while stack:
            aid = stack.pop(0)
            if aid in seen:
                continue
            seen.add(aid)
            art = by_id.get(aid)
            if art is None:
                continue
            order.append(art)
            stack.extend(p for p in art.parents if p not in seen)
        return order
