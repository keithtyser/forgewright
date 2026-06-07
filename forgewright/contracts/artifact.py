"""Artifact dataclasses + (de)serialization.

`Artifact` is the base: a typed handle with provenance. Each `kind` is a thin subclass
that fixes `kind` and documents its conventional `uri` + `meta`. Everything is plain
dataclasses so they round-trip to/from the JSONL registry with no framework.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Gate:
    """A specialist's gate decision on the artifact it produced."""
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _new_id(kind: str) -> str:
    return f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


@dataclass
class Artifact:
    """A typed, provenance-carrying handle to a post-training output.

    uri        — where the bytes live (local path, HF repo id, or http endpoint).
    produced_by— the specialist role that created it (e.g. "SFTTrainer").
    config_hash— hash of the producing config/plan (reproducibility).
    parents    — artifact ids this was derived from (the provenance edges).
    gate       — the producer's gate decision (None = ungated).
    meta       — kind-specific fields (family, base model, scores, etc.).
    """
    kind: str
    uri: str = ""
    produced_by: str = ""
    config_hash: str = ""
    parents: list[str] = field(default_factory=list)
    gate: Optional[Gate] = None
    run_id: str = ""
    hardware: str = ""
    id: str = ""
    created_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _new_id(self.kind)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.gate is not None:
            d["gate"] = self.gate.to_dict()
        return d


# --- kind-fixed subclasses ---------------------------------------------------------
# Each just pins `kind` so producers/consumers declare intent; `uri`/`meta` conventions
# are documented per class.


@dataclass
class DatasetArtifact(Artifact):
    """uri = training JSONL path. meta: {format, rows, holdouts, source}."""
    kind: str = "dataset"


@dataclass
class AdapterArtifact(Artifact):
    """uri = LoRA adapter dir. meta: {base, family, method(sft|grpo), steps}."""
    kind: str = "adapter"


@dataclass
class ModelArtifact(Artifact):
    """uri = model dir / HF id. meta: {role(base|abliterated|quantized|merged), family}."""
    kind: str = "model"


@dataclass
class ServedEndpoint(Artifact):
    """uri = base_url (http). meta: {served_model_name, serving_config, objective, tok_s}."""
    kind: str = "served_endpoint"


@dataclass
class EvalArtifact(Artifact):
    """uri = scores.csv / eval dir. meta: {capability, refusal_rate_harmful, ...}."""
    kind: str = "eval"


@dataclass
class PublishedArtifact(Artifact):
    """uri = HF repo url. meta: {release_class, visibility}."""
    kind: str = "published"


ARTIFACT_KINDS: dict[str, type[Artifact]] = {
    "dataset": DatasetArtifact,
    "adapter": AdapterArtifact,
    "model": ModelArtifact,
    "served_endpoint": ServedEndpoint,
    "eval": EvalArtifact,
    "published": PublishedArtifact,
}


def artifact_from_dict(d: dict[str, Any]) -> Artifact:
    """Rebuild the right Artifact subclass from a registry row."""
    cls = ARTIFACT_KINDS.get(d.get("kind", ""), Artifact)
    data = dict(d)
    gate = data.pop("gate", None)
    art = cls(**data)
    if gate is not None:
        art.gate = Gate(**gate)
    return art
