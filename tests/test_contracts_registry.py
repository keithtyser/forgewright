"""Tests for the swarm lingua franca: artifact contracts, registry, lineage, Specialist."""
from __future__ import annotations

from pathlib import Path

import pytest

from forgewright.agents.base import Specialist, label_reporter
from forgewright.contracts import (
    AdapterArtifact,
    Artifact,
    DatasetArtifact,
    EvalArtifact,
    Gate,
    artifact_from_dict,
)
from forgewright.registry import Registry


def test_artifact_id_and_kind_defaults():
    d = DatasetArtifact(uri="data/train.jsonl", produced_by="DataCurator")
    assert d.kind == "dataset"
    assert d.id.startswith("dataset-")
    a = AdapterArtifact(uri="~/models/x-sft", produced_by="SFTTrainer", parents=[d.id])
    assert a.kind == "adapter" and a.parents == [d.id]


def test_artifact_round_trips_through_dict_with_gate():
    a = AdapterArtifact(
        uri="~/models/x-sft", produced_by="SFTTrainer",
        gate=Gate(passed=True, metrics={"loss": 0.16}, verdict="PASS"),
        meta={"base": "Qwen/Qwen3.5-0.8B", "method": "sft"},
    )
    back = artifact_from_dict(a.to_dict())
    assert isinstance(back, AdapterArtifact)
    assert back.id == a.id and back.uri == a.uri
    assert back.gate is not None and back.gate.passed and back.gate.metrics["loss"] == 0.16
    assert back.meta["method"] == "sft"


def test_registry_register_get_latest(tmp_path: Path):
    reg = Registry(tmp_path / "artifacts.jsonl")
    d1 = reg.register(DatasetArtifact(uri="d1.jsonl", meta={"family": "q08"}))
    d2 = reg.register(DatasetArtifact(uri="d2.jsonl", meta={"family": "q08"}))
    reg.register(DatasetArtifact(uri="other.jsonl", meta={"family": "gemma"}))
    assert reg.get(d1.id).uri == "d1.jsonl"
    assert reg.get("missing") is None
    # latest of a kind filtered by meta
    assert reg.latest("dataset", family="q08").id == d2.id
    assert reg.latest("dataset", family="gemma").uri == "other.jsonl"
    assert reg.latest("adapter") is None


def test_registry_lineage_walks_parents(tmp_path: Path):
    reg = Registry(tmp_path / "artifacts.jsonl")
    d = reg.register(DatasetArtifact(uri="train.jsonl"))
    a = reg.register(AdapterArtifact(uri="adapter", parents=[d.id]))
    e = reg.register(EvalArtifact(uri="scores.csv", parents=[a.id]))
    chain = reg.lineage(e.id)
    ids = [x.id for x in chain]
    assert ids == [e.id, a.id, d.id]  # leaf-first to root


def test_registry_persists_across_instances(tmp_path: Path):
    p = tmp_path / "artifacts.jsonl"
    a = Registry(p).register(DatasetArtifact(uri="x.jsonl"))
    reloaded = Registry(p).get(a.id)  # fresh Registry reading the same file
    assert reloaded is not None and reloaded.uri == "x.jsonl"


def test_label_reporter_tags_role():
    seen = []
    base = lambda kind, data: seen.append((kind, data))  # noqa: E731
    r = label_reporter(base, "SFTTrainer")
    r("tool", {"tool": "forge"})
    assert seen == [("tool", {"tool": "forge", "role": "SFTTrainer"})]
    assert label_reporter(None, "X") is None


class _DummySpecialist(Specialist):
    role = "Dummy"
    accepts = ("dataset",)
    produces = "adapter"

    def system_prompt(self) -> str:
        return "dummy"

    def tools(self):  # not used in this test
        from forgewright.tools.base import ToolRegistry

        return ToolRegistry([])

    def run(self, inputs, goal):
        return AdapterArtifact(uri="out")


def test_specialist_validate_inputs():
    s = _DummySpecialist()
    s.validate_inputs([DatasetArtifact(uri="d.jsonl")])  # ok
    with pytest.raises(ValueError):
        s.validate_inputs([AdapterArtifact(uri="a")])  # wrong kind
