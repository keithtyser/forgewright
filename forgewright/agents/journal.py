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
        # Keep the existing filename: a new key must not bypass an unfinished run.
        from forgewright.tools.forge import ForgeRunner
        settings = {"seeds": [(a.kind, a.uri, a.meta, a.config_hash, a.parents,
                               a.gate.to_dict() if a.gate else None) for a in seeds],
                    "stages": [{"class": f"{s.specialist_cls.__module__}.{s.specialist_cls.__qualname__}",
                                "repo": str(s.init_kwargs.get("runner", ForgeRunner()).repo.resolve()),
                                "init": {k: v for k, v in s.init_kwargs.items() if k not in ("runner", "jobs")},
                                "max_attempts": s.max_attempts,
                                "repair": s.repair, "compensate": s.compensate} for s in recipe]}
        self.settings = json.dumps(settings, sort_keys=True, default=str)

    @staticmethod
    def file_proof(artifact, repo: Path) -> dict | None:
        if artifact.kind not in ("model", "adapter", "dataset") and not (
            artifact.kind == "eval" and artifact.uri
        ):
            return None
        from forgewright.registry.integrity import capture_manifest, IntegrityError
        proof: dict = {"artifact": artifact.id, "uri": artifact.uri, "repo": str(repo.resolve())}
        try:
            proof["manifest"] = capture_manifest(artifact.uri, repo=repo)
        except IntegrityError as error:
            proof["error"] = str(error)
        return proof

    def verify_files(self) -> None:
        from forgewright.registry.integrity import verify_manifest, IntegrityError
        if self.state.get("settings") != self.settings or "file_proofs" not in self.state:
            raise IntegrityError("Saved recipe lacks matching settings and file proofs; reconcile it before reuse")
        for proof in self.state["file_proofs"]:
            if proof.get("error"):
                raise IntegrityError(f"Saved artifact could not be verified: {proof['error']}")
            verify_manifest(proof["uri"], proof.get("manifest"), repo=Path(proof["repo"]))

    def save(self, **values) -> None:
        self.state.update(values)
        temp = self.path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(self.state, stream, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(self.path)
