"""Tests for the eval-gate: pure gate decision + self-contained script generation."""
from __future__ import annotations

import ast
from pathlib import Path

from forgewright.skills.eval_gate import (
    build_eval_gate_command,
    gate_pass,
    gate_report,
    render_eval_gate,
    write_eval_gate,
)


def test_gate_pass():
    assert gate_pass(0.80, 0.85)               # improvement
    assert gate_pass(0.80, 0.80)               # equal
    assert not gate_pass(0.80, 0.70)           # regression
    assert gate_pass(0.80, 0.78, tolerance=0.03)   # within noise tolerance
    assert not gate_pass(0.80, 0.70, tolerance=0.03)


def test_gate_report():
    r = gate_report(0.5, 0.75)
    assert r["delta"] == 0.25 and r["passed"] and r["verdict"] == "PASS"
    r2 = gate_report(0.9, 0.6)
    assert r2["verdict"] == "REGRESSION" and not r2["passed"]


def test_render_eval_gate_is_self_contained_and_valid():
    src = render_eval_gate()
    ast.parse(src)  # must be valid python
    assert "def numeric_correctness_reward" in src   # reward injected, no forgewright import
    assert "PeftModel.from_pretrained" in src
    assert "expanduser(cfg[" in src                   # ~ handled
    # robust loader: tries both arch classes and FAILS LOUDLY if the adapter is a no-op
    assert "AutoModelForImageTextToText" in src and "AutoModelForCausalLM" in src
    assert "lora_B" in src
    assert "did not apply with any loader" in src


def test_write_eval_gate(tmp_path: Path):
    cfg, script = write_eval_gate(
        tmp_path, "qwen35_0_8b_grpo_v0",
        base="Qwen/Qwen3.5-0.8B", adapter="~/models/x-grpo", holdout="datasets/rl/holdout.jsonl",
        tolerance=0.0,
    )
    assert cfg.exists() and script.exists()
    assert cfg == tmp_path / "runs" / "eval_gate" / "qwen35_0_8b_grpo_v0" / "config.json"
    import json
    c = json.loads(cfg.read_text())
    assert c["base"] == "Qwen/Qwen3.5-0.8B" and c["adapter"] == "~/models/x-grpo"


def test_eval_gate_command():
    cmd = build_eval_gate_command("qwen35_0_8b_grpo_v0")
    assert "model-forge-posttrain-tf5:latest" in cmd
    assert "--gpus all" in cmd
    assert "runs/eval_gate/qwen35_0_8b_grpo_v0/eval_gate.py" in cmd
