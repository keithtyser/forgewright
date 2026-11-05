"""Append-only artifact registry (JSONL) with lineage.

One artifact per line. `register` appends; reads rebuild typed artifacts from the file.
Latest-wins by `created_at` for `latest()`. `lineage` walks `parents` to the roots.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional
from filelock import FileLock

from forgewright.contracts import Artifact, artifact_from_dict


def _default_path() -> Path:
    home = Path(os.environ.get("FORGEWRIGHT_HOME", str(Path.home() / ".forgewright")))
    return home / "registry" / "artifacts.jsonl"


class Registry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.path) + ".lock")

    # --- write ---------------------------------------------------------------------

    def register(self, artifact: Artifact) -> Artifact:
        """Append an artifact and return it (id assigned in __post_init__)."""
        with self._lock:
            previous = self.get(artifact.id)
            if previous is not None:
                if previous.to_dict() != artifact.to_dict():
                    raise ValueError(f"Artifact {artifact.id} is immutable; register a new artifact")
                return artifact
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(artifact.to_dict(), default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return artifact

    # --- read ----------------------------------------------------------------------

    def all(self) -> list[Artifact]:
        with self._lock:
            return self._read_all()

    def _read_all(self) -> list[Artifact]:
        if not self.path.exists():
            return []
        artifacts = []
        by_id: dict[str, Artifact] = {}
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if (not isinstance(data, dict) or not isinstance(data.get("id"), str) or not data["id"]
                    or not isinstance(data.get("meta", {}), dict)
                    or not isinstance(data.get("parents", []), list)
                    or any(not isinstance(p, str) or not p for p in data.get("parents", []))):
                    raise ValueError("Invalid artifact record")
                gate = data.get("gate")
                if gate is not None and (not isinstance(gate, dict) or type(gate.get("passed")) is not bool):
                    raise ValueError("Invalid gate record")
                artifact = artifact_from_dict(data)
                if artifact.id in by_id and by_id[artifact.id].to_dict() != artifact.to_dict():
                    raise ValueError("Conflicting records for the same artifact ID")
                by_id[artifact.id] = artifact
                artifacts.append(artifact)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Corrupt artifact registry at {self.path}:{number}; repair before running") from exc
        return artifacts

    def get(self, artifact_id: str) -> Optional[Artifact]:
        # Repeated identical records are permitted for older registries.
        found = None
        for a in self.all():
            if a.id == artifact_id:
                found = a
        return found

    def latest(self, kind: str | None = None, *, include_failed: bool = False, **meta_filters: Any) -> Optional[Artifact]:
        """Most-recent artifact of `kind` whose meta matches all `meta_filters`. Ties on
        `created_at` (coarse clocks) are broken by append order — last registered wins."""
        artifacts = self.all()
        rejected = self._rejected(artifacts)
        candidates = [
            (idx, a) for idx, a in enumerate(artifacts)
            if (kind is None or a.kind == kind)
            and (include_failed or a.id not in rejected)
            and all(a.meta.get(k) == v for k, v in meta_filters.items())
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda ia: (ia[1].created_at, ia[0]))[1]

    @staticmethod
    def _rejected(artifacts: list[Artifact]) -> set[str]:
        rejected = {a.id for a in artifacts if a.gate is not None and not a.gate.passed}
        evaluations = {}
        for artifact in artifacts:
            if artifact.kind == "eval":
                for parent in artifact.parents:
                    evaluations[parent] = artifact.gate is not None and artifact.gate.passed
        rejected.update(aid for aid, passed in evaluations.items() if not passed)
        return rejected

    def select(self, kind: str, *, artifact_id: str | None = None, family: str | None = None,
               run_id: str | None = None, variant: str | None = None) -> Artifact:
        """Select one accepted artifact explicitly or within a named scope."""
        if not any((artifact_id, family, run_id)):
            raise ValueError("Supply an artifact ID, family, or artifact_run_id; global latest is ambiguous")
        artifacts = self.all()
        rejected = self._rejected(artifacts)
        candidates = {a.id: a for a in artifacts if a.kind == kind and a.id not in rejected
                      and (not artifact_id or a.id == artifact_id)
                      and (not family or a.meta.get("family") == family)
                      and (not run_id or a.run_id == run_id)
                      and (not variant or a.meta.get("variant") == variant)}
        if len(candidates) != 1:
            raise ValueError(f"Expected one accepted {kind}, found {len(candidates)}; supply its artifact ID or URI")
        return next(iter(candidates.values()))

    def lineage(self, artifact_id: str, *, strict: bool = True) -> list[Artifact]:
        """The artifact and its ancestors (via `parents`), de-duped, leaf-first."""
        by_id = {a.id: a for a in self.all()}
        if strict:
            active: set[str] = set()
            checked: set[str] = set()
            pending = [(artifact_id, False)]
            while pending:
                aid, leaving = pending.pop()
                if leaving:
                    active.remove(aid)
                    checked.add(aid)
                    continue
                if aid in active:
                    raise ValueError(f"Artifact lineage contains a cycle at {aid}")
                if aid in checked:
                    continue
                if aid not in by_id:
                    raise ValueError(f"Artifact lineage is missing {aid}")
                active.add(aid)
                pending.append((aid, True))
                pending.extend((parent, False) for parent in by_id[aid].parents)
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
