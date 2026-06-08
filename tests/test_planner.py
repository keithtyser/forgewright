"""Tests for the LLM stage-DAG planner: type validation, parsing, build, and safe fallback."""
from __future__ import annotations

from forgewright.agents.planner import (
    SPECIALISTS,
    build_plan,
    plan_stages,
    validate_chain,
)
from forgewright.brain.provider import AssistantTurn
from forgewright.contracts import ModelArtifact
from forgewright.registry import Registry


class _Brain:
    """Returns a scripted assistant reply (the planner's JSON), like the real Brain.chat."""

    def __init__(self, content):
        self._content = content

    def chat(self, messages, tools=None, tool_choice="auto"):
        return AssistantTurn(self._content)


# --- type validation --------------------------------------------------------
def test_valid_chain_from_goal():
    ok, _ = validate_chain(["DataCurator", "SFTTrainer", "Evaluator"], None)
    assert ok


def test_valid_chain_from_model_seed():
    ok, _ = validate_chain(["Abliterator", "Quantizer", "Evaluator", "Publisher"], "model")
    assert ok


def test_invalid_chain_type_mismatch():
    # SFTTrainer produces an adapter, Abliterator needs a model -> invalid handoff
    ok, reason = validate_chain(["DataCurator", "SFTTrainer", "Abliterator"], None)
    assert not ok and "Abliterator" in reason


def test_invalid_chain_missing_seed():
    ok, reason = validate_chain(["Quantizer"], None)   # needs a model, none upstream
    assert not ok and "Quantizer" in reason


def test_unknown_role_rejected():
    ok, reason = validate_chain(["DataCurator", "Frobnicator"], None)
    assert not ok and "unknown" in reason


# --- LLM planning -----------------------------------------------------------
def test_plan_stages_parses_and_validates():
    brain = _Brain('Here is the plan:\n[{"role": "Abliterator"}, {"role": "Evaluator"}]')
    roles = plan_stages("uncensor my model", brain, {"model_uri": "/m/x"})
    assert roles == ["Abliterator", "Evaluator"]


def test_plan_stages_rejects_invalid_chain():
    # plausible-looking but type-invalid (adapter -> model gap) -> planner returns None
    brain = _Brain('[{"role": "DataCurator"}, {"role": "SFTTrainer"}, {"role": "Abliterator"}]')
    assert plan_stages("train then abliterate", brain, {}) is None


def test_plan_stages_none_without_brain():
    assert plan_stages("anything", None, {}) is None


def test_plan_stages_none_on_garbage():
    assert plan_stages("x", _Brain("no json here"), {}) is None


# --- build_plan -> steps + seed --------------------------------------------
def test_build_plan_makes_steps_and_model_seed(tmp_path):
    brain = _Brain('[{"role": "Abliterator"}, {"role": "Quantizer"}, {"role": "Evaluator"}]')
    reg = Registry(tmp_path / "r.jsonl")
    steps, seed = build_plan("uncensor then quantize", {"model_uri": "/m/Qwen", "family": "q"}, reg, brain)
    assert [s.specialist_cls.role for s in steps] == ["Abliterator", "Quantizer", "Evaluator"]
    assert len(seed) == 1 and isinstance(seed[0], ModelArtifact) and seed[0].uri == "/m/Qwen"
    assert steps[0].run_kwargs == {}   # no abliterate params given


def test_build_plan_returns_none_when_unplannable():
    assert build_plan("x", {}, Registry(), None) is None   # no brain -> caller falls back


def test_roster_in_sync_with_specialists():
    assert SPECIALISTS["SFTTrainer"].produces == "adapter"
    assert "model" in SPECIALISTS["Quantizer"].accepts
