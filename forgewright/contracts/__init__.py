"""Typed artifact contracts — the lingua franca of the post-training swarm.

Specialists never call each other directly: a producer registers an `Artifact` (a typed
handle to a dataset / adapter / model / eval-report on disk or HF, plus provenance), and a
consumer pulls it from the registry. Artifacts are intentionally minimal — a `uri` (where
the bytes live), how it was produced, what it came from (`parents`), and its gate result —
so the contract stays additive and the registry doubles as a provenance graph.
"""
from forgewright.contracts.artifact import (
    ARTIFACT_KINDS,
    AdapterArtifact,
    Artifact,
    DatasetArtifact,
    EvalArtifact,
    Gate,
    ModelArtifact,
    PublishedArtifact,
    ServedEndpoint,
    artifact_from_dict,
)

__all__ = [
    "Artifact",
    "Gate",
    "DatasetArtifact",
    "AdapterArtifact",
    "ModelArtifact",
    "ServedEndpoint",
    "EvalArtifact",
    "PublishedArtifact",
    "ARTIFACT_KINDS",
    "artifact_from_dict",
]
