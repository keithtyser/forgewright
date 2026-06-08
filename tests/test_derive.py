"""Tests for the pre-flight derivation tool (fuse GPU capability + model spec into a plan)."""
from __future__ import annotations

import json

from forgewright.tools import derive
from forgewright.tools.base import ToolResult
from forgewright.tools.derive import DerivePlanTool

_SPEC = {
    "architecture": "LlamaForCausalLM", "model_type": "llama", "num_hidden_layers": 32,
    "hidden_size": 4096, "num_attention_heads": 32, "param_count": 8_000_000_000,
    "chat_template_present": True, "is_moe": False, "arch_known": True,
    "abliterate_target_suffixes": ["self_attn.o_proj.weight", "mlp.down_proj.weight"],
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


class _FakeForge:
    def __init__(self, spec=None, ok=True):
        self.spec, self.ok = spec if spec is not None else _SPEC, ok

    def run(self, args, **kw):
        return ToolResult(self.ok, json.dumps(self.spec) if self.ok else "no readable config.json")


def _fake_gpu(monkeypatch, arch, supported, vram="81920"):
    monkeypatch.setattr(
        derive.GPUInspectTool, "run",
        lambda self, host=None: ToolResult(True, "x", {"gpus": [
            {"arch": arch, "sm": "sm_x", "vram_mib": vram, "supported_quant": supported}]}),
    )


def test_blackwell_picks_nvfp4(monkeypatch):
    _fake_gpu(monkeypatch, "blackwell", ["nvfp4", "fp8", "int8", "awq"], vram="131072")
    plan = DerivePlanTool(runner=_FakeForge()).run(model="/m/Llama-3-8B").meta["plan"]
    assert plan["quant_method"] == "nvfp4"
    assert plan["abliterate_target_suffixes"] == ["self_attn.o_proj.weight", "mlp.down_proj.weight"]
    assert plan["lora_target_modules"][0] == "q_proj"
    assert plan["family"] == "llama_3_8b"


def test_ampere_picks_int8(monkeypatch):
    _fake_gpu(monkeypatch, "ampere", ["int8", "awq", "gptq"])
    plan = DerivePlanTool(runner=_FakeForge()).run(model="/m/x").meta["plan"]
    assert plan["quant_method"] == "int8"
    # non-Blackwell -> warn that the default container image needs overriding
    assert any("MODEL_FORGE_POSTTRAIN_IMAGE" in w for w in plan["warnings"])


def test_blackwell_no_container_warning(monkeypatch):
    _fake_gpu(monkeypatch, "blackwell", ["nvfp4"])
    plan = DerivePlanTool(runner=_FakeForge()).run(model="/m/x").meta["plan"]
    assert not any("MODEL_FORGE_POSTTRAIN_IMAGE" in w for w in plan["warnings"])


def test_no_quant_support_serves_bf16(monkeypatch):
    _fake_gpu(monkeypatch, "unknown", [])
    plan = DerivePlanTool(runner=_FakeForge()).run(model="/m/x").meta["plan"]
    assert plan["quant_method"] is None
    assert any("no quantization" in w for w in plan["warnings"])


def test_unknown_arch_warns(monkeypatch):
    _fake_gpu(monkeypatch, "blackwell", ["nvfp4"])
    spec = dict(_SPEC, arch_known=False, model_type="weirdnew")
    plan = DerivePlanTool(runner=_FakeForge(spec)).run(model="/m/x").meta["plan"]
    assert any("not in the known set" in w for w in plan["warnings"])


def test_feasibility_flags_too_big(monkeypatch):
    _fake_gpu(monkeypatch, "ampere", ["int8"], vram="24576")     # 24GB
    spec = dict(_SPEC, param_count=70_000_000_000)               # 70B -> ~140GB bf16
    plan = DerivePlanTool(runner=_FakeForge(spec)).run(model="/m/x").meta["plan"]
    assert "quantize" in plan["feasibility"] or "not fit" in plan["feasibility"]


def test_model_introspection_failure_surfaces(monkeypatch):
    _fake_gpu(monkeypatch, "blackwell", ["nvfp4"])
    res = DerivePlanTool(runner=_FakeForge(ok=False)).run(model="/m/missing")
    assert res.ok is False and "introspect" in res.output
