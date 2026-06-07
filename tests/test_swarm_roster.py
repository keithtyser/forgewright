"""Tests for the rest of the swarm roster (RL/Quantizer/Abliterator/ServingOpt/Publisher),
the recipe library, and Director saga compensation (mocked forge/jobs; no GPU)."""
from __future__ import annotations

import pytest

from forgewright.agents.abliterator import Abliterator
from forgewright.agents.director import Director, Step
from forgewright.agents.publisher import Publisher
from forgewright.agents.quantizer import Quantizer
from forgewright.agents.recipes import RECIPES, build_recipe, uplift
from forgewright.agents.rl_trainer import RLTrainer, _final_reward
from forgewright.contracts import (
    AdapterArtifact,
    DatasetArtifact,
    Gate,
    ModelArtifact,
)
from forgewright.registry import Registry
from forgewright.tools.base import ToolResult


class FakeForge:
    def __init__(self, repo, ok=True):
        self.repo = repo
        self._ok = ok
    def available(self): return True
    def run(self, args, timeout=0, **_): return ToolResult(self._ok, "ok" if self._ok else "boom")


class FakeJobs:
    def __init__(self, log="", exit_code=0):
        self._log, self._exit = log, exit_code
        self.launched = []
    def launch(self, command, host=None, cwd=None, name=None):
        self.launched.append(command); return {"id": "job-x", "command": command}
    def wait(self, jid, **_): return {"id": jid, "status": "finished", "exit_code": self._exit}
    def tail(self, jid, n=80): return self._log


def test_final_reward_parses():
    assert _final_reward("rewards/reward_correct/mean': '0.94'") == 0.94
    assert _final_reward("nothing") is None


def test_rl_trainer_produces_grpo_adapter(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    ds = reg.register(DatasetArtifact(uri="datasets/rl/mult.jsonl",
                                      meta={"family": "q08", "source": "Qwen/Qwen3.5-0.8B",
                                            "run_name": "q08_grpo", "holdout": "datasets/rl/h.jsonl"}))
    art = RLTrainer(runner=FakeForge(tmp_path), jobs=FakeJobs(log="reward_correct/mean': '0.9'"),
                    registry=reg).run([ds], max_steps=20)
    assert art.kind == "adapter" and art.meta["method"] == "grpo" and art.parents == [ds.id]
    assert art.gate.passed and art.meta["holdout"] == "datasets/rl/h.jsonl"


def test_quantizer_produces_quantized_model(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    jobs = FakeJobs()
    base = reg.register(ModelArtifact(uri="~/models/Qwen3.5-9B", meta={"family": "qwen35_9b", "role": "base"}))
    art = Quantizer(runner=FakeForge(tmp_path), jobs=jobs, registry=reg).run([base])
    assert art.kind == "model" and art.meta["role"] == "quantized" and art.parents == [base.id]
    assert art.gate.passed
    assert any("quantize export" in c for c in jobs.launched)            # launched the export job
    assert (tmp_path / "configs" / "quantization" / "qwen35_9b_nvfp4_modelopt.yaml").exists()


def test_abliterator_chains_collect_and_export(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    base = reg.register(ModelArtifact(uri="/home/x/models/Qwen3.5-0.8B", meta={"family": "qwen35_0_8b"}))
    jobs = FakeJobs()
    art = Abliterator(runner=FakeForge(tmp_path), jobs=jobs, registry=reg).run([base], strength=3.0)
    assert art.kind == "model" and art.meta["role"] == "abliterated"
    cmd = jobs.launched[0]
    assert "collect --execute" in cmd and "export --execute" in cmd   # both stages in one job
    assert "run_in_container.sh" in cmd


def test_publisher_is_human_gated(tmp_path):
    from forgewright.permissions import PermissionPolicy

    reg = Registry(tmp_path / "a.jsonl")
    model = reg.register(ModelArtifact(uri="~/models/q-nvfp4",
                                       gate=Gate(passed=True), meta={"family": "q", "variant": "base_nvfp4_modelopt"}))
    # denied -> blocked, nothing published
    denied = Publisher(runner=FakeForge(tmp_path), registry=reg,
                       permissions=PermissionPolicy(ask_fn=lambda t, a: False))
    art = denied.run([model])
    assert not art.gate.passed and art.meta.get("blocked")


def test_publisher_refuses_failed_artifact(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    bad = reg.register(AdapterArtifact(uri="x", gate=Gate(passed=False, verdict="REGRESSION")))
    art = Publisher(runner=FakeForge(tmp_path), registry=reg).run([bad])
    assert not art.gate.passed and "did not pass its gate" in art.gate.verdict


def test_recipe_library_builds_expected_chains():
    steps, seed = uplift(family="q08", source="Qwen/Qwen3.5-0.8B",
                         seed_paths=["s.jsonl"], holdout="h.jsonl")
    roles = [s.specialist_cls.role for s in steps]
    assert roles == ["DataCurator", "SFTTrainer", "Evaluator"] and seed == []
    assert set(RECIPES) >= {"uplift", "task_grpo", "uplift_publish", "quantize_serve", "abliterate"}
    with pytest.raises(ValueError):
        build_recipe("nonsense")


def test_director_runs_saga_compensation_on_failure(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    compensated = []

    from forgewright.agents.base import Specialist
    from forgewright.tools.base import ToolRegistry

    class GoodModel(Specialist):
        role = "GoodModel"; accepts = (); produces = "model"
        def system_prompt(self): return ""
        def tools(self): return ToolRegistry([])
        def run(self, inputs, goal="", **kw):
            a = ModelArtifact(uri="m", gate=Gate(passed=True)); self.registry.register(a); return a

    class FailStage(Specialist):
        role = "FailStage"; accepts = ("model",); produces = "eval"
        def system_prompt(self): return ""
        def tools(self): return ToolRegistry([])
        def run(self, inputs, goal="", **kw):
            from forgewright.contracts import EvalArtifact
            a = EvalArtifact(uri="e", gate=Gate(passed=False, verdict="REGRESSION"))
            self.registry.register(a); return a

    recipe = [
        Step(GoodModel, compensate=lambda art: compensated.append(art.id)),
        Step(FailStage),
    ]
    res = Director(registry=reg).run_recipe("x", recipe, seed_inputs=[])
    assert not res.ok and res.failed_at == "FailStage"
    assert len(compensated) == 1   # the completed GoodModel step was rolled back
