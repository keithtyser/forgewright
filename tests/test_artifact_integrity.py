"""File mutation checks without GPUs, model downloads, or uploads."""
from __future__ import annotations

import json
import os
import shlex

import pytest

from forgewright.agents.base import Specialist
from forgewright.agents.director import Director, Step
from forgewright.agents.evaluator import Evaluator
from forgewright.agents.publisher import Publisher
from forgewright.contracts import EvalArtifact, Gate, ModelArtifact
from forgewright.ledger.ledger import Ledger
from forgewright.permissions import PermissionPolicy
from forgewright.registry import Registry
from forgewright.registry import integrity
from forgewright.tools.base import ToolRegistry, ToolResult
from forgewright.tools.forge import ForgeRunner


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "checkpoint with spaces"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "config.json").write_text('{"model_type":"test"}')
    return path


@pytest.mark.parametrize("change", ["weights", "config", "added", "deleted", "same_timestamp"])
def test_manifest_detects_file_changes(checkpoint, change):
    proof = integrity.capture_manifest(str(checkpoint), checkpoint=True)
    assert integrity.verify_manifest(str(checkpoint), proof, checkpoint=True) == proof
    weights = checkpoint / "model.safetensors"
    if change == "weights":
        weights.write_bytes(b"changed")
    elif change == "config":
        (checkpoint / "config.json").write_text("{}")
    elif change == "added":
        (checkpoint / "tokenizer.json").write_text("{}")
    elif change == "deleted":
        (checkpoint / "config.json").unlink()
    else:
        before = weights.stat()
        weights.write_bytes(b"changed")
        os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(integrity.IntegrityError, match="changed"):
        integrity.verify_manifest(str(checkpoint), proof, checkpoint=True)


def test_manifest_rejects_changed_earlier_file_during_scan(checkpoint, monkeypatch):
    original = integrity._hash_file
    def hash_file(path):
        result = original(path)
        if path.name == "model.safetensors":
            (checkpoint / "config.json").write_text("changed after its hash")
        return result
    monkeypatch.setattr(integrity, "_hash_file", hash_file)
    with pytest.raises(integrity.IntegrityError, match="changed while hashing"):
        integrity.capture_manifest(str(checkpoint))


def test_manifest_does_not_ignore_directory_read_errors(checkpoint, monkeypatch):
    def walk(root, *, followlinks, onerror):
        onerror(PermissionError("unreadable subtree"))
        return iter([])
    monkeypatch.setattr(integrity.os, "walk", walk)
    with pytest.raises(integrity.IntegrityError, match="unreadable subtree"):
        integrity.capture_manifest(str(checkpoint))


@pytest.mark.parametrize("kind", ["missing", "empty", "no_weights", "url"])
def test_checkpoint_requires_local_weight_files(tmp_path, kind):
    path = tmp_path / "model"
    if kind in ("empty", "no_weights"):
        path.mkdir()
    if kind == "no_weights":
        (path / "config.json").write_text("{}")
    uri = "https://example.invalid/model" if kind == "url" else str(path)
    with pytest.raises(integrity.IntegrityError):
        integrity.capture_manifest(uri, checkpoint=True)


def test_manifest_rejects_symlinks(checkpoint, tmp_path):
    link = tmp_path / "linked"
    try:
        link.symlink_to(checkpoint, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires host permission")
    with pytest.raises(integrity.IntegrityError, match="Linked"):
        integrity.capture_manifest(str(link))


@pytest.mark.parametrize("mutate", [False, True])
def test_evaluation_records_candidate_bytes_and_rejects_mutation(checkpoint, tmp_path, monkeypatch, mutate):
    reg = Registry(tmp_path / "r.jsonl")
    model = reg.register(ModelArtifact(uri=str(checkpoint), gate=Gate(True)))
    evaluator = Evaluator(runner=ForgeRunner(tmp_path), registry=reg)
    def measure(model):
        if mutate:
            (checkpoint / "config.json").write_text("{}")
        return {"capability": 0.8}
    monkeypatch.setattr(evaluator, "_measure_model", measure)
    result = evaluator.run([model], baseline={"capability": 0.8})
    assert result.gate.passed is (not mutate)
    assert result.meta["checkpoint"]["sha256"]
    assert bool(result.meta["integrity_error"]) is mutate


def test_missing_checkpoint_never_launches_evaluation(tmp_path, monkeypatch):
    evaluator = Evaluator(runner=ForgeRunner(tmp_path), registry=Registry(tmp_path / "r.jsonl"))
    monkeypatch.setattr(evaluator, "_measure_model", lambda *a: pytest.fail("must not launch"))
    result = evaluator.run([ModelArtifact(uri=str(tmp_path / "missing"))], baseline={"capability": 0.8})
    assert not result.gate.passed and "does not exist" in result.gate.verdict


def publisher_fixture(tmp_path, checkpoint, *, proof=True, permissions=None):
    reg = Registry(tmp_path / "r.jsonl")
    model = reg.register(ModelArtifact(uri=str(checkpoint), gate=Gate(True),
                                      meta={"family": "q", "variant": "trained"}))
    meta = {"checkpoint": integrity.capture_manifest(model.uri, checkpoint=True)} if proof else {}
    reg.register(EvalArtifact(parents=[model.id], gate=Gate(True), meta=meta))
    publisher = Publisher(runner=ForgeRunner(tmp_path), registry=reg,
                          permissions=permissions or PermissionPolicy(auto_approve=True))
    return publisher, model


def upload_receipt():
    return ToolResult(True, json.dumps({"dry_run": False, "family": "q", "variant": "trained",
        "uploaded_at": "now", "uploaded_files": ["model.safetensors"],
        "hub_url": "https://huggingface.co/user/model"}))


@pytest.mark.parametrize("mutation", ["none", "before", "approval", "upload", "legacy"])
def test_publication_checks_files_at_each_boundary(tmp_path, checkpoint, monkeypatch, mutation):
    approvals, calls = [], []
    def mutate():
        (checkpoint / "model.safetensors").write_bytes(b"changed")
    def approve(tool, args):
        approvals.append(args)
        if mutation == "approval":
            mutate()
        return True
    publisher, model = publisher_fixture(tmp_path, checkpoint, proof=mutation != "legacy",
                                        permissions=PermissionPolicy(ask_fn=approve))
    if mutation == "before":
        mutate()
    def upload(**kwargs):
        calls.append(kwargs)
        tokens = shlex.split(kwargs["args"])
        assert tokens[tokens.index("--artifact-path") + 1] == str(checkpoint.resolve())
        if mutation == "upload":
            mutate()
        return upload_receipt()
    monkeypatch.setattr(publisher.publish_tool, "run", upload)
    result = publisher.run([model])
    assert result.gate.passed is (mutation == "none")
    assert len(calls) == (1 if mutation in ("none", "upload") else 0)
    assert len(approvals) == (0 if mutation in ("before", "legacy") else 1)
    if mutation == "upload":
        assert result.meta["receipt"]["uploaded_at"] == "now"
        assert result.gate.verdict.startswith("UNVERIFIED")


@pytest.mark.parametrize("flag", ["--artifact-path elsewhere", "--artifact-path=elsewhere", "--art elsewhere"])
def test_publisher_rejects_upload_source_override(tmp_path, checkpoint, flag):
    publisher, model = publisher_fixture(tmp_path, checkpoint,
        permissions=PermissionPolicy(ask_fn=lambda *a: pytest.fail("must not ask")))
    result = publisher.run([model], publish_args=f"publish-model q trained --execute {flag}")
    assert not result.gate.passed and "artifact-path" in result.gate.verdict


@pytest.mark.parametrize("change", ["none", "modified", "deleted", "legacy", "settings", "seed",
                                    "seed_metadata", "seed_gate", "new_failed_eval"])
def test_completed_recipe_checks_files_and_settings(tmp_path, checkpoint, change):
    calls = []
    seed_file = tmp_path / "seed.bin"
    seed_file.write_bytes(b"seed")
    seed = ModelArtifact(uri=str(seed_file))
    class Stage(Specialist):
        role = "Stage"
        produces = "model"
        def system_prompt(self): return ""
        def tools(self): return ToolRegistry([])
        def run(self, inputs, goal):
            calls.append(goal)
            return ModelArtifact(uri=str(checkpoint), gate=Gate(True))
    director = Director(registry=Registry(tmp_path / "r.jsonl"), ledger=Ledger("run", tmp_path))
    step = Step(Stage)
    first = director.run_recipe("goal", [step], [seed])
    assert first.ok
    if change == "modified":
        (checkpoint / "config.json").write_text("{}")
    elif change == "deleted":
        (checkpoint / "model.safetensors").unlink()
    elif change == "legacy":
        path = next((tmp_path / "recipes").glob("*.json"))
        state = json.loads(path.read_text())
        del state["file_proofs"]
        path.write_text(json.dumps(state))
    elif change == "settings":
        step.max_attempts = 2
    elif change == "seed":
        seed_file.write_bytes(b"changed seed")
    elif change == "seed_metadata":
        seed = ModelArtifact(uri=str(seed_file), meta={"family": "different"})
    elif change == "seed_gate":
        seed = ModelArtifact(uri=str(seed_file), gate=Gate(False))
    elif change == "new_failed_eval":
        director.registry.register(EvalArtifact(parents=[first.final.id], gate=Gate(False)))
    second = director.run_recipe("goal", [step], [seed])
    assert second.ok is (change == "none")
    assert len(calls) == 1
    if change != "none":
        assert second.failed_at == "recovery"
