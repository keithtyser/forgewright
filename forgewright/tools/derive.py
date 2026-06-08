"""Pre-flight derivation: fuse the two ground truths (the GPU + the model) into a run plan.

This is what the agent should call BEFORE launching any GPU job. It inspects the hardware
capability (via gpu_inspect) and introspects the model (via model-forge's `forge model describe`),
then derives the strategy: training precision, the quant method the GPU can run, the abliteration
target modules and LoRA targets for the model's architecture, the family name, and a rough VRAM
feasibility note. Deriving up front (instead of assuming Blackwell + qwen) is what lets the agent
post-train an arbitrary model on arbitrary hardware, and it fails fast instead of deep in a job.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from forgewright.skills.quantize import choose_quant_method
from forgewright.tools.base import Tool, ToolResult
from forgewright.tools.forge import ForgeRunner
from forgewright.tools.gpu import GPUInspectTool


def _first_gpu(host: Optional[str]) -> dict[str, Any]:
    res = GPUInspectTool().run(host=host)
    gpus = (res.meta or {}).get("gpus") or []
    return gpus[0] if gpus else {}


def _feasibility(param_count: Optional[int], vram_mib: Any, quant_method: Optional[str]) -> str:
    """A rough 'can this run here' note from params + VRAM (not a hard gate)."""
    try:
        vram = int(str(vram_mib).strip("[]"))
    except (ValueError, TypeError):
        vram = 0
    if not param_count or vram <= 0:
        return "VRAM/param size unknown; verify it fits before launching."
    gb = vram / 1024.0
    bf16_gb = param_count * 2 / 1e9          # weights in bf16
    if bf16_gb <= gb * 0.6:
        return f"comfortable: ~{bf16_gb:.1f}GB bf16 weights vs {gb:.0f}GB VRAM."
    if bf16_gb <= gb * 0.9:
        return f"tight: ~{bf16_gb:.1f}GB bf16 weights vs {gb:.0f}GB VRAM; LoRA + low batch."
    if quant_method:
        return (f"bf16 (~{bf16_gb:.1f}GB) likely will not fit {gb:.0f}GB VRAM; "
                f"quantize ({quant_method}) and/or shard.")
    return f"bf16 (~{bf16_gb:.1f}GB) exceeds {gb:.0f}GB VRAM and no quant is supported here; use a smaller model."


class DerivePlanTool(Tool):
    name = "derive_plan"
    description = (
        "Pre-flight: given a local model path, inspect this GPU and introspect the model, then "
        "derive the run strategy -- training precision, the quant method the GPU supports "
        "(nvfp4/fp8/int8/awq/none), the abliteration target modules and LoRA targets for the "
        "model's architecture, the family name, and a VRAM feasibility note. Call this BEFORE "
        "launching training/quant so the plan matches the hardware + model and fails fast."
    )
    risk = "read"
    parameters = {
        "type": "object",
        "properties": {
            "model": {"type": "string", "description": "Local path to the model dir (containing config.json)."},
            "host": {"type": "string", "description": "Optional user@host for remote GPU inspection."},
        },
        "required": ["model"],
    }

    def __init__(self, runner: Optional[ForgeRunner] = None) -> None:
        self.forge = runner or ForgeRunner()

    def run(self, model: str, host: Optional[str] = None, **_: Any) -> ToolResult:
        gpu = _first_gpu(host)
        arch = gpu.get("arch", "unknown")
        supported = gpu.get("supported_quant", [])

        described = self.forge.run(f"model describe {model} --json")
        if not described.ok:
            return ToolResult(False, f"could not introspect model at {model}: {described.output}")
        try:
            spec = json.loads(described.output.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return ToolResult(False, f"model describe did not return JSON: {described.output[:200]}")

        quant_method = choose_quant_method(supported)
        family = str(model).rstrip("/").split("/")[-1].lower().replace(".", "_").replace("-", "_")
        plan = {
            "model": model,
            "family": family,
            "precision": "bf16",
            "quant_method": quant_method,            # None => serve/keep bf16
            "abliterate_target_suffixes": spec.get("abliterate_target_suffixes", []),
            "lora_target_modules": spec.get("lora_target_modules", []),
            "is_moe": spec.get("is_moe", False),
            "arch_known": spec.get("arch_known", True),
            "hardware": {"arch": arch, "sm": gpu.get("sm"), "vram_mib": gpu.get("vram_mib"),
                         "supported_quant": supported},
            "model_spec": {k: spec.get(k) for k in
                           ("architecture", "model_type", "num_hidden_layers", "hidden_size",
                            "num_attention_heads", "param_count", "chat_template_present")},
            "feasibility": _feasibility(spec.get("param_count"), gpu.get("vram_mib"), quant_method),
            "warnings": [],
        }
        if not plan["arch_known"]:
            plan["warnings"].append(
                f"architecture '{spec.get('model_type')}' is not in the known set; "
                "verify the abliterate/LoRA target module names before editing weights.")
        if not supported:
            plan["warnings"].append("this GPU supports no quantization; plan serves bf16.")

        return ToolResult(True, _summary(plan), {"plan": plan})


def _summary(p: dict[str, Any]) -> str:
    s, hw = p["model_spec"], p["hardware"]
    params = f"~{s['param_count']:,}" if s.get("param_count") else "?"
    lines = [
        f"plan for {p['model']}  (family {p['family']})",
        f"  model    : {s.get('architecture')} | {s.get('num_hidden_layers')}L x {s.get('hidden_size')}"
        f" | {'MoE' if p['is_moe'] else 'dense'} | {params} params | chat:{'yes' if s.get('chat_template_present') else 'no'}",
        f"  hardware : {hw['arch']} {hw.get('sm')} | {hw.get('vram_mib')} MiB | quant: "
        + (", ".join(hw['supported_quant']) or "none"),
        f"  precision: {p['precision']}",
        f"  quant    : {p['quant_method'] or 'none (serve bf16)'}",
        f"  abliterate targets: {', '.join(p['abliterate_target_suffixes'])}",
        f"  lora targets      : {', '.join(p['lora_target_modules'])}",
        f"  feasibility: {p['feasibility']}",
    ]
    for w in p["warnings"]:
        lines.append(f"  ! {w}")
    return "\n".join(lines)
