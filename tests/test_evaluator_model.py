"""Tests for the Evaluator model-internal-eval gate (pure decision + run routing)."""
from __future__ import annotations

from forgewright.agents.evaluator import Evaluator, _model_gate
from forgewright.contracts import ModelArtifact
from forgewright.registry import Registry
from forgewright.tools.base import ToolResult


class FakeForge:
    def __init__(self, repo): self.repo = repo
    def available(self): return True
    def run(self, args, timeout=0, **_): return ToolResult(True, "ok")


def test_model_gate_not_served():
    assert not _model_gate({}, None, served=False).passed


def test_model_gate_no_capability():
    g = _model_gate({"refusal_rate_harmful": 0.1}, None)
    assert not g.passed and "no capability" in g.verdict


def test_model_gate_no_baseline_reports():
    g = _model_gate({"capability": 0.81}, None)
    assert g.passed and g.verdict == "EVALUATED"


def test_model_gate_abliterate_pass():
    base = {"capability": 0.81, "refusal_rate_harmful": 0.90}
    cand = {"capability": 0.80, "refusal_rate_harmful": 0.10}
    g = _model_gate(cand, base)
    assert g.passed and g.verdict == "PASS"
    assert round(g.metrics["refusal_drop"], 2) == 0.80


def test_model_gate_capability_regressed():
    base = {"capability": 0.81, "refusal_rate_harmful": 0.90}
    cand = {"capability": 0.60, "refusal_rate_harmful": 0.10}
    g = _model_gate(cand, base)
    assert not g.passed and "capability regressed" in g.verdict


def test_model_gate_refusal_not_dropped():
    base = {"capability": 0.81, "refusal_rate_harmful": 0.90}
    cand = {"capability": 0.80, "refusal_rate_harmful": 0.85}
    g = _model_gate(cand, base)
    assert not g.passed and "refusal did not drop" in g.verdict


def test_model_gate_capability_only_baseline():
    # quant/uplift baseline (no refusal): just capability-hold
    assert _model_gate({"capability": 0.80}, {"capability": 0.81}).passed       # within tol
    assert not _model_gate({"capability": 0.50}, {"capability": 0.81}).passed   # regressed


def test_run_routes_model_to_internal_eval(tmp_path, monkeypatch):
    reg = Registry(tmp_path / "a.jsonl")
    ev = Evaluator(runner=FakeForge(tmp_path), registry=reg)
    called = {}
    monkeypatch.setattr(ev, "_eval_model_internal", lambda model, baseline=None: called.setdefault("hit", model))
    model = ModelArtifact(uri="~/models/q-abl", meta={"family": "q", "variant": "abliterated"})
    ev.run([model])
    assert called["hit"] is model   # model kind -> internal-eval path (not the adapter path)
