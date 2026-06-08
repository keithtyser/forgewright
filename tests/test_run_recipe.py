"""Tests for the RunRecipeTool bridge (chat -> Director swarm)."""
from __future__ import annotations

from forgewright.agents.run_recipe import RunRecipeTool, _resolve
from forgewright.contracts import ModelArtifact
from forgewright.registry import Registry


def test_resolve_uplift_normalizes_seed_path(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    steps, seed = _resolve("uplift", {"family": "q08", "source": "Qwen/Qwen3.5-0.8B",
                                      "seed_path": "datasets/finetuning/distill_smoke.jsonl",
                                      "holdout": "datasets/rl/h.jsonl"}, reg)
    roles = [s.specialist_cls.role for s in steps]
    assert roles == ["DataCurator", "SFTTrainer", "Evaluator"] and seed == []


def test_resolve_quantize_serve_from_model_uri(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    steps, seed = _resolve("quantize_serve", {"family": "qwen35_9b", "model_uri": "~/models/Qwen3.5-9B",
                                              "objective": "throughput"}, reg)
    assert [s.specialist_cls.role for s in steps] == ["Quantizer", "ServingOptimizer"]
    assert len(seed) == 1 and seed[0].kind == "model"


def test_resolve_quantize_serve_from_registry_latest(tmp_path):
    reg = Registry(tmp_path / "a.jsonl")
    m = reg.register(ModelArtifact(uri="~/models/base", meta={"family": "q", "role": "base"}))
    steps, seed = _resolve("quantize_serve", {}, reg)
    assert seed[0].id == m.id   # pulled the latest model from the registry


def test_resolve_unknown_recipe(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        _resolve("nonsense", {}, Registry(tmp_path / "a.jsonl"))


def test_run_recipe_tool_runs_director(tmp_path, monkeypatch):
    reg = Registry(tmp_path / "a.jsonl")
    tool = RunRecipeTool(registry=reg)
    captured = {}

    # stub the Director so we test the tool wiring (not real GPU specialists)
    import forgewright.agents.run_recipe as rr

    class FakeRes:
        ok = True; failed_at = ""; reason = ""
        def __init__(self):
            self.final = type("A", (), {"id": "eval-1"})()
            self.artifacts = [type("A", (), {"kind": "adapter", "id": "ad-1"})(),
                              type("A", (), {"kind": "eval", "id": "eval-1"})()]

    class FakeDirector:
        def __init__(self, **kw): captured["kw"] = kw
        def run_recipe(self, goal, steps, seed): captured["goal"] = goal; return FakeRes()

    monkeypatch.setattr(rr, "Director", FakeDirector)
    tool.bind(reporter=lambda k, d: None, permissions="POLICY")
    res = tool.run("uplift", {"family": "q08", "source": "Qwen/Qwen3.5-0.8B",
                              "seed_path": "s.jsonl", "holdout": "h.jsonl"}, goal="uplift it")
    assert res.ok and "completed" in res.output and "adapter:ad-1" in res.output
    assert captured["kw"]["permissions"] == "POLICY"   # this turn's approval policy threaded in
    assert captured["goal"] == "uplift it"


def test_run_recipe_tool_handles_json_string_params(tmp_path, monkeypatch):
    import forgewright.agents.run_recipe as rr
    monkeypatch.setattr(rr, "_resolve", lambda *a, **k: ([], []))

    class FakeRes:
        ok = True; failed_at = ""; reason = ""; final = None; artifacts = []

    monkeypatch.setattr(rr, "Director", lambda **kw: type("D", (), {"run_recipe": lambda self, *a: FakeRes()})())
    tool = RunRecipeTool(registry=Registry(tmp_path / "a.jsonl"))
    res = tool.run("uplift", '{"family": "q08"}')   # params as a JSON string
    assert res.ok


def test_plan_recipe_name_from_goal():
    from forgewright.agents.recipes import plan_recipe_name
    assert plan_recipe_name("make qwen an uncensored assistant") == "abliterate"
    assert plan_recipe_name("quantize llama for faster serving") == "quantize_serve"
    assert plan_recipe_name("GRPO on a verifiable math dataset") == "task_grpo"
    assert plan_recipe_name("fine-tune and publish a reasoning model") == "uplift_publish"
    assert plan_recipe_name("uplift the 0.8B with distillation") == "uplift"
    assert plan_recipe_name("do something vague") == "uplift"   # sensible default
