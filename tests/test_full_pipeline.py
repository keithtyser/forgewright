"""Tests for the Merger specialist + the full one-run pipeline (gated after each transform)."""
from __future__ import annotations

from forgewright.agents import merger as merger_mod
from forgewright.agents.merger import Merger
from forgewright.agents.planner import validate_chain
from forgewright.agents.recipes import build_recipe, plan_recipe_name
from forgewright.contracts import AdapterArtifact
from forgewright.registry import Registry
from forgewright.tools.forge import ForgeRunner


class _FakeJobs:
    def __init__(self, exit_code=0):
        self._exit = exit_code

    def launch(self, cmd, host=None, cwd=None, name=None):
        return {"id": "job-test"}

    def wait(self, jid, **kw):
        return {"exit_code": self._exit}


def _merger(tmp_path, exit_code, fresh, monkeypatch):
    monkeypatch.setattr(merger_mod, "wrote_fresh_weights", lambda host, out, since: fresh)
    return Merger(registry=Registry(tmp_path / "r.jsonl"), runner=ForgeRunner(repo=tmp_path),
                  jobs=_FakeJobs(exit_code))


def test_merger_produces_merged_model(tmp_path, monkeypatch):
    m = _merger(tmp_path, exit_code=0, fresh=True, monkeypatch=monkeypatch)
    ad = AdapterArtifact(uri="/home/u/models/Qwen-uplift", meta={"family": "q", "base": "/home/u/models/Qwen"})
    art = m.run([ad])
    assert art.kind == "model" and art.meta["role"] == "merged" and art.gate.passed
    assert "merged" in art.uri and art.parents == [ad.id]


def test_merger_fails_without_fresh_weights(tmp_path, monkeypatch):
    m = _merger(tmp_path, exit_code=0, fresh=False, monkeypatch=monkeypatch)
    art = m.run([AdapterArtifact(uri="/m/a", meta={"family": "q", "base": "/m/Qwen"})])
    assert art.gate.passed is False and "fresh weights" in art.gate.verdict


def test_merger_needs_base(tmp_path, monkeypatch):
    m = _merger(tmp_path, exit_code=0, fresh=True, monkeypatch=monkeypatch)
    import pytest
    with pytest.raises(ValueError):
        m.run([AdapterArtifact(uri="/m/a", meta={"family": "q"})])   # no base


# --- the full recipe -------------------------------------------------------
def test_full_recipe_chains_all_stages_with_evals():
    steps, seed = build_recipe("full", family="q", source="Qwen/Qwen3.5-0.8B",
                               seed_paths=["d.jsonl"], holdout="h.jsonl")
    roles = [s.specialist_cls.role for s in steps]
    assert roles == ["DataCurator", "SFTTrainer", "Evaluator", "Merger", "Abliterator",
                     "Evaluator", "Quantizer", "Evaluator", "ServingOptimizer"]
    assert roles.count("Evaluator") == 3   # gated after uplift, abliterate, quantize


def test_full_recipe_is_type_valid():
    steps, _ = build_recipe("full", family="q", source="s", seed_paths=["d"], holdout="h")
    ok, reason = validate_chain([s.specialist_cls.role for s in steps], None)
    assert ok, reason


def test_plan_recipe_name_picks_full_for_combined_goal():
    assert plan_recipe_name("fine-tune Qwen, abliterate it, and quantize for my GPU") == "full"
    assert plan_recipe_name("uplift then quantize and publish") == "full"
    assert plan_recipe_name("run the full pipeline on qwen") == "full"
    # single-intent goals still map to their focused recipe
    assert plan_recipe_name("just quantize it") == "quantize_serve"
    assert plan_recipe_name("make it uncensored") == "abliterate"
