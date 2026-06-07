"""Tests for the data-factory config scaffolder (derive logic + file writing)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from forgewright.skills.data_factory import (
    derive_dataset_config,
    derive_source_registry,
    write_dataset_config,
)

_REF = {
    "id": "gemma_ref", "family": "gemma", "variant": "v1", "objective": "capability_sft",
    "output_dir": "datasets/generated/gemma_ref",
    "baseline": {"reference_model": "Jackrong/x"},
    "seed_paths": ["datasets/seeds/gemma.jsonl"],
    "source_registry": "configs/data_sources/gemma_ref.yaml",
    "source_ids": ["gemma_ref_seed"],
    "generation_methods": {"enabled_now": ["self_instruct"]},
    "generation": {"provider": {"type": "openai_compatible"}, "smoke": {"max_generated_candidates": 6}},
    "quality_thresholds": {"min_average_score": 0.7},
    "holdouts": ["evals/prompts/capability_preservation_challenge.yaml"],
}


def test_derive_dataset_config_substitutes_identity_keeps_structure():
    out = derive_dataset_config(_REF, family="qwen35_0_8b", variant="curated_v0",
                                seed_path="datasets/finetuning/distill_smoke.jsonl")
    assert out["id"] == "qwen35_0_8b_curated_v0" and out["family"] == "qwen35_0_8b"
    assert out["output_dir"] == "datasets/generated/qwen35_0_8b_curated_v0"
    assert out["seed_paths"] == ["datasets/finetuning/distill_smoke.jsonl"]
    assert out["source_registry"] == "configs/data_sources/qwen35_0_8b_curated_v0.yaml"
    assert out["source_ids"] == ["qwen35_0_8b_curated_v0_seed"]
    assert "baseline" not in out                                  # reference-specific, dropped
    assert out["generation_methods"] == {"enabled_now": ["self_instruct"]}   # structure kept
    assert out["generation"]["provider"]["type"] == "openai_compatible"


def test_derive_source_registry_points_at_seed():
    reg = derive_source_registry(family="q08", variant="curated_v0", seed_path="datasets/x.jsonl")
    src = reg["sources"]["q08_curated_v0_seed"]
    assert src["type"] == "local_jsonl" and src["path"] == "datasets/x.jsonl"
    assert src["messages_field"] == "messages"


def test_write_dataset_config(tmp_path: Path):
    (tmp_path / "configs" / "datasets").mkdir(parents=True)
    (tmp_path / "configs" / "datasets" / "ref.yaml").write_text(yaml.safe_dump(_REF))
    cfg, reg = write_dataset_config(tmp_path, "qwen35_0_8b", seed_path="datasets/s.jsonl",
                                    reference="ref")
    assert cfg.exists() and reg.exists()
    assert cfg.name == "qwen35_0_8b_curated_v0.yaml" and reg.name == "qwen35_0_8b_curated_v0.yaml"
    cfgd = yaml.safe_load(cfg.read_text())
    assert cfgd["family"] == "qwen35_0_8b" and cfgd["seed_paths"] == ["datasets/s.jsonl"]


def test_write_dataset_config_missing_reference(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        write_dataset_config(tmp_path, "x", seed_path="s.jsonl", reference="nope")
