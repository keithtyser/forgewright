"""Tests for the model_spec helper + introspection-driven abliterate/LoRA target derivation."""
from __future__ import annotations

import json

from forgewright.skills.abliterate import scaffold_abliterate_config
from forgewright.skills.finetune import scaffold_uplift_config
from forgewright.skills.modelspec import model_spec
from forgewright.tools.base import ToolResult


class _Forge:
    def __init__(self, payload, ok=True):
        self.payload, self.ok = payload, ok

    def run(self, args, **kw):
        return ToolResult(self.ok, self.payload)


def test_model_spec_parses_trailing_json():
    spec = {"model_type": "llama", "lora_target_modules": ["q_proj", "o_proj"]}
    out = "some banner line\n" + json.dumps(spec)
    assert model_spec(_Forge(out), "/m/x")["model_type"] == "llama"


def test_model_spec_none_on_failure():
    assert model_spec(_Forge("no readable config.json", ok=False), "/m/x") is None
    assert model_spec(_Forge("not json at all"), "/m/x") is None
    assert model_spec(None, "/m/x") is None


def test_abliterate_config_uses_derived_suffixes():
    cfg = scaffold_abliterate_config(
        "phi3fam", source="/m/phi3",
        target_weight_suffixes=["self_attn.o_proj.weight", "mlp.down_proj.weight"])
    assert "self_attn.o_proj.weight: 1.25" in cfg     # attn gets the stronger weight
    assert "mlp.down_proj.weight: 0.75" in cfg
    assert "    - self_attn.o_proj.weight" in cfg and "    - mlp.down_proj.weight" in cfg


def test_abliterate_config_default_suffixes():
    cfg = scaffold_abliterate_config("qwen35_0_8b", source="/m/q")
    assert "self_attn.o_proj.weight" in cfg and "mlp.down_proj.weight" in cfg


def test_finetune_config_uses_derived_lora_targets():
    cfg = scaffold_uplift_config("phi3fam", source="/m/phi3",
                                 lora_target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"])
    assert "target_modules: [qkv_proj, o_proj, gate_up_proj, down_proj]" in cfg


def test_finetune_config_default_lora_targets():
    cfg = scaffold_uplift_config("qwen35_0_8b", source="/m/q")
    assert "target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]" in cfg
