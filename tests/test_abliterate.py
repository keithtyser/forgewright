"""Tests for the abliterate skill: dual-gate logic, scaffolding, scores parsing."""
from __future__ import annotations

from pathlib import Path

from forgewright.skills.abliterate import (
    abliterate_gate,
    read_abliterate_metrics,
    scaffold_abliterate_config,
    write_abliterate_config,
)


def test_gate_passes_when_refusal_drops_and_capability_holds():
    base = {"refusal_rate_harmful": 0.90, "capability": 0.80, "benign_refusal_rate": 0.05}
    cand = {"refusal_rate_harmful": 0.10, "capability": 0.78, "benign_refusal_rate": 0.08}
    r = abliterate_gate(base, cand)
    assert r["passed"] and r["verdict"] == "ABLITERATED"
    assert round(r["refusal_drop"], 2) == 0.80 and r["capability_held"]


def test_gate_fails_when_refusal_did_not_drop():
    base = {"refusal_rate_harmful": 0.90, "capability": 0.80, "benign_refusal_rate": 0.05}
    cand = {"refusal_rate_harmful": 0.85, "capability": 0.80, "benign_refusal_rate": 0.05}
    r = abliterate_gate(base, cand)  # drop 0.05 < 0.10
    assert not r["passed"] and "refusal did not drop" in r["verdict"]


def test_gate_fails_when_capability_regresses():
    base = {"refusal_rate_harmful": 0.90, "capability": 0.80, "benign_refusal_rate": 0.05}
    cand = {"refusal_rate_harmful": 0.10, "capability": 0.60, "benign_refusal_rate": 0.05}
    r = abliterate_gate(base, cand)  # capability -0.20 beyond tolerance
    assert not r["passed"] and "capability regressed" in r["verdict"]


def test_gate_fails_when_over_abliterated():
    base = {"refusal_rate_harmful": 0.90, "capability": 0.80, "benign_refusal_rate": 0.05}
    cand = {"refusal_rate_harmful": 0.05, "capability": 0.79, "benign_refusal_rate": 0.55}
    r = abliterate_gate(base, cand)  # benign refusals exploded -> over-abliterated
    assert not r["passed"] and "over-abliterated" in r["verdict"]


def test_scaffold_bakes_capability_preserving_defaults():
    cfg = scaffold_abliterate_config("qwen35_0_8b", source="Qwen/Qwen3.5-0.8B")
    assert "method: contrastive_refusal_direction" in cfg
    assert "leave_embeddings_untouched: true" in cfg
    assert "leave_lm_head_untouched: true" in cfg
    assert "leave_moe_experts_untouched: true" in cfg
    assert "norm_preserve: true" in cfg
    assert "harmful_prompts: datasets/abliteration/harmful_refusal.yaml" in cfg
    assert "require_execute_flag: true" in cfg


def test_write_abliterate_config_idempotent(tmp_path: Path):
    cfg = write_abliterate_config(tmp_path, "qwen35_0_8b", source="Qwen/Qwen3.5-0.8B")
    assert cfg == tmp_path / "configs" / "abliteration" / "qwen35_0_8b_abliterated_v0.yaml"
    assert cfg.exists()
    cfg.write_text(cfg.read_text() + "# touched\n")
    write_abliterate_config(tmp_path, "qwen35_0_8b", source="Qwen/Qwen3.5-0.8B", overwrite=False)
    assert "# touched" in cfg.read_text()  # not overwritten
    write_abliterate_config(tmp_path, "qwen35_0_8b", source="Qwen/Qwen3.5-0.8B", overwrite=True)
    assert "# touched" not in cfg.read_text()


def test_read_abliterate_metrics(tmp_path: Path):
    p = tmp_path / "scores.csv"
    p.write_text(
        "bucket,metric,value,count\n"
        "unsafe_overcompliance,refusal_rate_harmful,0.90,3\n"
        "capability_preservation_challenge,normal_use_regression_pass_rate,0.81,32\n"
        "refusal_benign_boundary,benign_refusal_rate,0.06,3\n"
        "agentic_code_debug,workflow_success,1.0,2\n"
    )
    m = read_abliterate_metrics(p)
    assert m == {"refusal_rate_harmful": 0.90, "capability": 0.81, "benign_refusal_rate": 0.06}
    assert read_abliterate_metrics(tmp_path / "nope.csv") == {}
