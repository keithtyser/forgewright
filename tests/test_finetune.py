"""Tests for the fine-tune decision logic + scaffolding (pure; no GPU/training)."""
from __future__ import annotations

from pathlib import Path

import pytest

from forgewright.skills.finetune import (
    scaffold_uplift_config,
    scaffold_uplift_manifest,
    select_mode,
    write_finetune_config,
)


@pytest.mark.parametrize(
    "goal, expected",
    [
        ("improve general reasoning and chat style via teacher distillation", "uplift"),
        ("uplift the model's <think> formatting and broad capability", "uplift"),
        ("get GSM8K accuracy from 60% to 80%", "task"),
        ("train with verifiable rewards (RLVR) to solve the puzzles", "task"),
        ("raise pass@1 on the unit-test benchmark", "task"),
        ("make it a better assistant", "uplift"),  # neither signal -> safe default
        # mixed: a concrete verifiable target outranks generic 'reasoning'
        ("improve reasoning so accuracy hits 90%", "task"),
    ],
)
def test_select_mode(goal, expected):
    assert select_mode(goal) == expected


def test_uplift_config_bakes_scar_defaults():
    cfg = scaffold_uplift_config("qwen35_0_8b", source="Qwen/Qwen3.5-0.8B", learning_rate=8e-5)
    # train_on_responses_only + conservative LR + no-Unsloth backend
    assert "assistant_only_loss: true" in cfg
    assert "backend: hf_causal_lm" in cfg
    assert "learning_rate: 8e-05" in cfg
    assert "source: Qwen/Qwen3.5-0.8B" in cfg
    assert "data:\n  manifest: datasets/finetuning/qwen35_0_8b_uplift_v0.yaml" in cfg


def test_uplift_manifest_has_think_and_holdout_hygiene():
    man = scaffold_uplift_manifest("qwen35_0_8b", data_path="datasets/finetuning/d.jsonl")
    assert "reject_unclosed_think: true" in man          # strict <think>
    assert "reject_eval_prompt_overlap: true" in man     # no train/eval contamination
    assert "format: messages" in man
    assert "path: datasets/finetuning/d.jsonl" in man


def test_write_finetune_config_writes_both_and_is_idempotent(tmp_path: Path):
    repo = tmp_path
    cfg, man = write_finetune_config(repo, "qwen35_0_8b", source="Qwen/Qwen3.5-0.8B")
    assert cfg.exists() and man.exists()
    assert cfg == repo / "configs" / "finetuning" / "qwen35_0_8b_uplift_v0.yaml"
    assert man == repo / "datasets" / "finetuning" / "qwen35_0_8b_uplift_v0.yaml"
    # idempotent: a second call without overwrite does not change the file
    before = cfg.read_text()
    cfg.write_text(before + "# touched\n")
    write_finetune_config(repo, "qwen35_0_8b", source="Qwen/Qwen3.5-0.8B", overwrite=False)
    assert "# touched" in cfg.read_text()
    # overwrite=True regenerates
    write_finetune_config(repo, "qwen35_0_8b", source="Qwen/Qwen3.5-0.8B", overwrite=True)
    assert "# touched" not in cfg.read_text()
