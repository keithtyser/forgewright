"""Resolve explicit recipe inputs without selecting an unrelated global artifact."""
from __future__ import annotations

from forgewright.contracts import Artifact, DatasetArtifact, ModelArtifact
from forgewright.registry import Registry


def seed_artifact(kind: str, params: dict, registry: Registry) -> Artifact:
    uri = params.get(f"{kind}_uri")
    if kind == "dataset":
        uri = uri or params.get("dataset")
    artifact_id = params.get(f"{kind}_artifact_id")
    if uri and artifact_id:
        raise ValueError("Supply either an artifact ID or a URI, not both")
    if not uri:
        return registry.select(kind, artifact_id=artifact_id, family=params.get("family"),
                               run_id=params.get("artifact_run_id"), variant=params.get("variant"))
    if not isinstance(uri, str):
        raise ValueError("Artifact URI must be a string")
    if kind == "model":
        return ModelArtifact(uri=uri, meta={"family": params.get("family"), "role": "base",
                                            "variant": params.get("variant", "base")})
    return DatasetArtifact(uri=uri, meta={"family": params.get("family"), "source": params.get("source"),
                                         "run_name": params.get("run_name"), "holdout": params.get("holdout")})
