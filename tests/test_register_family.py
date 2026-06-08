"""Tests for model/experiment auto-registration (derive logic + file writing)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from forgewright.skills.register_family import (
    derive_experiment_config,
    derive_family_config,
    write_family_registration,
)

_REF_FAMILY = {
    "name": "qwen35_9b", "display_name": "Qwen 3.5 9B",
    "architecture": {"family": "qwen"},
    "variants": {"base": {"repo_id": "Qwen/Qwen3.5-9B", "local_dir": "Qwen3.5-9B",
                          "served_model_name": "Qwen/Qwen3.5-9B"}},
    "serve": {"script": "scripts/dgx_spark_serve_qwen.sh"},
    "eval": {"config": "configs/experiments/qwen35_9b_v0.yaml",
             "artifact_config": "configs/experiments/qwen35_9b_artifacts_v0.yaml",
             "output_root": "results/qwen35_9b_v0/base", "run_prefix": "qwen35_9b"},
}
_REF_EXP = {
    "experiment_name": "qwen35_9b_v0_base_eval",
    "model": {"family": "qwen35_9b", "id": "Qwen/Qwen3.5-9B", "variant": "base"},
    "backend": {"engine": "local-openai-compatible", "model_alias": "qwen35_9b_local"},
    "eval": {"output_dir": "results/qwen35_9b_v0/base", "prompt_sets": ["capability_preservation_challenge"]},
}


def test_derive_family_substitutes_identity_keeps_structure():
    out = derive_family_config(_REF_FAMILY, family="qwen35_0_8b", source="Qwen/Qwen3.5-0.8B")
    assert out["name"] == "qwen35_0_8b"
    assert out["variants"]["base"]["repo_id"] == "Qwen/Qwen3.5-0.8B"
    assert out["variants"]["base"]["local_dir"] == "Qwen3.5-0.8B"
    assert out["eval"]["config"] == "configs/experiments/qwen35_0_8b_v0.yaml"
    assert out["eval"]["output_root"] == "results/qwen35_0_8b_v0/base"
    assert out["serve"]["script"] == "scripts/dgx_spark_serve_qwen.sh"   # structure preserved
    assert out["architecture"]["family"] == "qwen"


_LLAMA_SPEC = {
    "model_type": "llama", "attn_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp_modules": ["gate_proj", "up_proj", "down_proj"], "arch_known": True,
}
_PHI3_SPEC = {
    "model_type": "phi3", "attn_modules": ["qkv_proj", "o_proj"],
    "mlp_modules": ["gate_up_proj", "down_proj"], "arch_known": True,
}


def test_derive_family_from_spec_rewrites_architecture_for_non_qwen():
    out = derive_family_config(_REF_FAMILY, family="llama3_8b", source="meta-llama/Llama-3-8B",
                               spec=_LLAMA_SPEC)
    arch = out["architecture"]
    assert arch["family"] == "llama"
    assert "chat_template" not in arch                       # qwen3 reasoning parser dropped
    assert arch["target_discovery"]["common_attention_patterns"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_derive_family_from_spec_keeps_qwen_chat_template():
    out = derive_family_config(_REF_FAMILY, family="qwen35_0_8b", source="Qwen/Qwen3.5-0.8B",
                               spec={"model_type": "qwen3_5", "attn_modules": ["q_proj"], "mlp_modules": ["down_proj"]})
    assert out["architecture"]["chat_template"]["reasoning_parser"] == "qwen3"


def test_derive_family_from_spec_uses_divergent_module_names():
    out = derive_family_config(_REF_FAMILY, family="phi3", source="microsoft/Phi-3", spec=_PHI3_SPEC)
    td = out["architecture"]["target_discovery"]
    assert td["common_attention_patterns"] == ["qkv_proj", "o_proj"]
    assert td["common_mlp_patterns"] == ["gate_up_proj", "down_proj"]


def test_derive_family_with_extra_variant():
    out = derive_family_config(
        _REF_FAMILY, family="q08", source="Qwen/Qwen3.5-0.8B",
        extra_variants={"abliterated": {"local_dir": "Qwen3.5-0.8B-abl", "served_model_name": "local/q08-abl"}})
    assert "abliterated" in out["variants"] and out["variants"]["abliterated"]["local_dir"] == "Qwen3.5-0.8B-abl"


def test_derive_experiment_substitutes_identity():
    out = derive_experiment_config(_REF_EXP, family="qwen35_0_8b", source="Qwen/Qwen3.5-0.8B")
    assert out["experiment_name"] == "qwen35_0_8b_v0_base_eval"
    assert out["model"] == {"family": "qwen35_0_8b", "id": "Qwen/Qwen3.5-0.8B", "variant": "base"}
    assert out["backend"]["model_alias"] == "qwen35_0_8b_local"
    assert out["eval"]["output_dir"] == "results/qwen35_0_8b_v0/base"
    assert out["eval"]["prompt_sets"] == ["capability_preservation_challenge"]  # preserved


def test_write_family_registration(tmp_path: Path):
    # lay down reference configs the writer reads from
    (tmp_path / "configs" / "model_families").mkdir(parents=True)
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    (tmp_path / "configs" / "model_families" / "qwen35_9b.yaml").write_text(yaml.safe_dump(_REF_FAMILY))
    (tmp_path / "configs" / "experiments" / "qwen35_9b_v0.yaml").write_text(yaml.safe_dump(_REF_EXP))

    fam, exp = write_family_registration(tmp_path, "qwen35_0_8b", source="Qwen/Qwen3.5-0.8B")
    assert fam.exists() and exp.exists()
    famd = yaml.safe_load(fam.read_text())
    assert famd["name"] == "qwen35_0_8b" and famd["variants"]["base"]["repo_id"] == "Qwen/Qwen3.5-0.8B"
    expd = yaml.safe_load(exp.read_text())
    assert expd["model"]["family"] == "qwen35_0_8b"


def test_write_family_registration_missing_reference(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        write_family_registration(tmp_path, "x", source="y", reference_family="nope")
