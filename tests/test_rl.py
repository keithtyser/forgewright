"""Tests for the GRPO/RLVR trainer: pure reward logic, scar defaults, scaffolding."""
from __future__ import annotations

from pathlib import Path

from forgewright.trainers.rl import (
    build_grpo_plan,
    build_grpo_train_command,
    extract_final_number,
    grpo_scar_defaults,
    has_closed_think,
    numeric_correctness_reward,
    render_grpo_trainer,
    think_format_reward,
    write_grpo_run,
)


def test_extract_final_number():
    assert extract_final_number("the answer is 41") == "41"
    assert extract_final_number("first 13, then 28, so \\boxed{41}") == "41"  # boxed wins
    assert extract_final_number("1,234 apples") == "1234"
    assert extract_final_number("no digits here") is None


def test_numeric_correctness_reward():
    assert numeric_correctness_reward("the result is 41", "41") == 1.0
    assert numeric_correctness_reward("I think 40", "41") == 0.0
    assert numeric_correctness_reward("\\boxed{12}", "12.0") == 1.0
    assert numeric_correctness_reward("no number", "5") == 0.0


def test_think_format_reward():
    assert has_closed_think("<think>reason</think> 41")
    assert not has_closed_think("</think> before <think>")  # wrong order
    assert not has_closed_think("no block")
    assert think_format_reward("<think>x</think>y") == 0.2
    assert think_format_reward("y") == 0.0


def test_scar_defaults_bake_in_the_lessons():
    d = grpo_scar_defaults()
    assert d["beta"] > 0           # KL anchor (the 0.00-collapse fix)
    assert 0 < d["epsilon"] < 1    # PPO clip
    assert d["epsilon_high"] >= d["epsilon"]
    assert d["mask_truncated_completions"] is True
    assert d["learning_rate"] <= 1e-5  # RL LR << SFT LR


def test_build_grpo_plan_applies_defaults_and_overrides():
    plan = build_grpo_plan("qwen35_0_8b_grpo_v0", source="Qwen/Qwen3.5-0.8B",
                           data_path="datasets/rl/math.jsonl", max_steps=30, beta=0.1)
    assert plan["grpo"]["max_steps"] == 30
    assert plan["grpo"]["beta"] == 0.1                 # override applied
    assert plan["grpo"]["mask_truncated_completions"] is True  # scar kept
    assert plan["model"]["source"] == "Qwen/Qwen3.5-0.8B"
    assert plan["data"]["path"] == "datasets/rl/math.jsonl"


def test_render_trainer_is_self_contained():
    src = render_grpo_trainer()
    # reward helpers injected verbatim (no forgewright import needed in the container)
    assert "def numeric_correctness_reward" in src
    assert "def think_format_reward" in src
    assert "GRPOTrainer(" in src and "GRPOConfig(" in src
    assert "beta=h[" in src and "epsilon=h[" in src  # scars wired into the config


def test_write_grpo_run(tmp_path: Path):
    plan, trainer = write_grpo_run(tmp_path, "qwen35_0_8b_grpo_v0",
                                   source="Qwen/Qwen3.5-0.8B", data_path="datasets/rl/math.jsonl")
    assert plan == tmp_path / "runs" / "rl" / "qwen35_0_8b_grpo_v0" / "plan.json"
    assert trainer == tmp_path / "runs" / "rl" / "qwen35_0_8b_grpo_v0" / "train_grpo.py"
    assert plan.exists() and trainer.exists()
    assert "numeric_correctness_reward" in trainer.read_text()


def test_grpo_train_command():
    cmd = build_grpo_train_command("qwen35_0_8b_grpo_v0")
    assert "model-forge-posttrain-tf5:latest" in cmd
    assert "--gpus all" in cmd
    assert "runs/rl/qwen35_0_8b_grpo_v0/train_grpo.py" in cmd
    assert "HF_HOME=$HOME/.forgewright/hf_home" in cmd
