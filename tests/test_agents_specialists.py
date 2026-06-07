"""Tests for the SFTTrainer + Evaluator specialists (mocked forge/jobs; no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgewright.agents.evaluator import Evaluator
from forgewright.agents.sft import SFTTrainer, _final_loss
from forgewright.contracts import AdapterArtifact, DatasetArtifact
from forgewright.registry import Registry
from forgewright.tools.base import ToolResult


class FakeForge:
    def __init__(self, repo: Path):
        self.repo = repo
        self.calls = []

    def available(self):
        return True

    def run(self, args, timeout=0, **_):
        self.calls.append(args)
        return ToolResult(True, "prepared")


class FakeJobs:
    def __init__(self, log="{'loss': '0.16'}", exit_code=0):
        self._log, self._exit = log, exit_code
        self.launched = []

    def launch(self, command, host=None, cwd=None, name=None):
        self.launched.append(command)
        return {"id": "job-test", "command": command}

    def wait(self, jid, **_):
        return {"id": jid, "status": "finished", "exit_code": self._exit}

    def tail(self, jid, n=80):
        return self._log


def test_final_loss_parses_last():
    assert _final_loss("{'loss': '1.2'}\n{'loss': '0.16'}") == 0.16
    assert _final_loss("no loss here") is None


def test_sft_trainer_produces_lineaged_adapter(tmp_path):
    forge = FakeForge(tmp_path)
    jobs = FakeJobs()
    reg = Registry(tmp_path / "artifacts.jsonl")
    ds = reg.register(DatasetArtifact(
        uri="datasets/finetuning/distill.jsonl", produced_by="DataCurator",
        meta={"family": "qwen35_0_8b", "source": "Qwen/Qwen3.5-0.8B", "run_name": "q08_uplift"},
    ))
    sft = SFTTrainer(runner=forge, jobs=jobs, registry=reg)
    art = sft.run([ds], "uplift it", max_steps=10)

    assert isinstance(art, AdapterArtifact)
    assert art.parents == [ds.id]                       # lineage
    assert art.meta["method"] == "sft" and art.meta["base"] == "Qwen/Qwen3.5-0.8B"
    assert art.gate.passed and art.gate.metrics["final_loss"] == 0.16
    # scaffolded the config + launched the container train
    assert any("prepare --overwrite" in c for c in forge.calls)
    assert any("train_trl_sft.py" in c for c in jobs.launched)
    # config + manifest + registry were written into the (tmp) repo
    assert (tmp_path / "configs" / "finetuning" / "q08_uplift.yaml").exists()
    # the produced artifact is discoverable in the registry with lineage
    assert reg.latest("adapter").id == art.id
    assert [a.id for a in reg.lineage(art.id)] == [art.id, ds.id]


def test_sft_trainer_rejects_wrong_input(tmp_path):
    sft = SFTTrainer(runner=FakeForge(tmp_path), jobs=FakeJobs(), registry=Registry(tmp_path / "r.jsonl"))
    with pytest.raises(ValueError):
        sft.run([AdapterArtifact(uri="x")], "nope")


def test_sft_gate_fails_on_nonzero_exit(tmp_path):
    sft = SFTTrainer(runner=FakeForge(tmp_path), jobs=FakeJobs(exit_code=1),
                     registry=Registry(tmp_path / "r.jsonl"))
    ds = DatasetArtifact(uri="d.jsonl", meta={"family": "q", "source": "Qwen/Qwen3.5-0.8B"})
    art = sft.run([ds], "")
    assert not art.gate.passed and "did not exit cleanly" in art.gate.verdict


def test_evaluator_produces_eval_artifact_from_result(tmp_path):
    forge = FakeForge(tmp_path)
    jobs = FakeJobs()
    reg = Registry(tmp_path / "artifacts.jsonl")
    adapter = reg.register(AdapterArtifact(
        uri="~/models/q08-grpo", produced_by="RLTrainer",
        meta={"base": "Qwen/Qwen3.5-0.8B", "run_name": "q08_grpo"},
    ))
    # simulate the container eval-gate having written its result.json
    rd = tmp_path / "runs" / "eval_gate" / "q08_grpo"
    rd.mkdir(parents=True)
    (rd / "result.json").write_text(json.dumps(
        {"base_score": 0.34, "candidate_score": 1.0, "delta": 0.66, "passed": True, "verdict": "PASS", "n": 50}))

    ev = Evaluator(runner=forge, jobs=jobs, registry=reg)
    art = ev.run([adapter], holdout="datasets/rl/mult_holdout.jsonl")

    assert art.kind == "eval" and art.parents == [adapter.id]
    assert art.gate.passed and art.gate.verdict == "PASS"
    assert art.meta["candidate_score"] == 1.0 and art.meta["delta"] == 0.66
    assert any("eval_gate.py" in c for c in jobs.launched)


def test_evaluator_needs_base_and_holdout(tmp_path):
    ev = Evaluator(runner=FakeForge(tmp_path), jobs=FakeJobs(), registry=Registry(tmp_path / "r.jsonl"))
    with pytest.raises(ValueError):
        ev.run([AdapterArtifact(uri="x", meta={})])  # no base, no holdout
