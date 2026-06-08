"""Quantize hardening: NVFP4 config scaffolding + the proven runbook.

Captures this session's hard-won lessons so the next quant is hands-off:
- a config template whose gate is **speedup-based** (no static tok/s floor — that
  varies by model + hardware), and
- the documented end-to-end runbook the agent follows, including the gotchas.

Publish-time fixes (HF_HOME redirect + Xet-off) live in ``tools/forge.py``
(``ForgePublishTool``). Long stages run detached via the job manager.
"""
from __future__ import annotations

from pathlib import Path

# Qwen-family NVFP4 template (the proven path). `strategy`/`script` are overridable
# for other architectures (e.g. gemma4_moe_modelopt). Note: `output_subdir` keeps a
# literal `{target_variant}` placeholder that model-forge expands at run time.
QUANT_CONFIG_TEMPLATE = """\
schema_version: model_forge.quantization.v1
name: {family}_{method}_{backend}
description: "Self-quantize {family} to {method} ({backend}); method chosen for the GPU's arch."
family: {family}
source_variant: {source_variant}
target_variant: {target_variant}
method: {method}
backend: {backend}
objective: quantized_quality_retention
hardware_profile: dgx_spark
calibration:
  dataset: cnn_dailymail
  samples: {calib_samples}
  seq_len: {calib_seq}
  batch_size: 4
exclusions:
  apply_recommended_keep_patterns: true
  modules: [lm_head, embed_tokens, router, expert]
export:
  image: model-forge-modelopt-nvfp4:0.43.0
  base_image: vllm-node-tf5:latest
  modelopt_version: 0.43.0
  output_root: ~/models/model-forge-quantized/{family}
  output_subdir: "{{target_variant}}"
  ptq:
    strategy: {strategy}
    script: {script}
    qformat: {qformat}
    dataset: cnn_dailymail
    calib_size: {calib_samples}
    calib_seq: {calib_seq}
    batch_size: 4
    tensor_parallel: 1
    gpu_max_mem_percentage: 0.70
    kv_cache_qformat: {kv_cache_qformat}
    low_memory_mode: true
    trust_remote_code: true
    skip_generate: true
runtime:
  served_model_name: model-forge/{family}-{target_variant}
  container: vllm-node-tf5
  port: {port}
  tensor_parallel: 1
  gpu_memory_utilization: 0.70
  max_model_len: 8192
  vllm_flags:
    trust_remote_code: true
    quantization: {backend}
    kv_cache_dtype: fp8
outputs:
  reports_dir: reports/generated/quantization
  models_dir: ~/models/model-forge-quantized/{family}
# Speedup-based gate — NO static tok/s floor (varies by model + hardware).
gates:
  {gate_key}:
    min_output_speedup: {min_speedup}
    min_decode_heavy_output_speedup: {min_speedup}
"""

# The agent follows this end-to-end; surfaced in the system prompt + as a tool hint.
NVFP4_RUNBOOK = """\
Hardened NVFP4 quantize -> gate -> publish runbook (DGX Spark / GB10):
 1. scaffold_quant_config <family>  (if configs/quantization/<family>_nvfp4_modelopt.yaml is missing).
 2. forge: quantize plan <family> <src> --config <cfg>            (dry sanity check).
 3. launch_job: bash forge quantize export <family> <src> --config <cfg> --execute   (DETACHED; poll to done).
 4. forge: quantize export <family> <src> --config <cfg> --write-plan   (writes quantization_export_plan.json -- the gate needs it).
 5. Bench BOTH for the speedup: serve <quantized> -> bench serve; stop server; serve <src bf16> -> bench serve.
 6. forge: eval <family> <quantized> --internal   AND   eval <family> <src> --internal   (base eval is needed for card/behavior).
 7. forge: quantize card / behavior-report / tokenizer-report   (source-vs-candidate serving summaries + evals).
 8. forge: quantize nvfp4-gate --export-plan .. --serving-summary .. --serving-eval .. --quantization-card .. \
      --behavior-report .. --tokenizer-report .. --run-id .. --write-gate   (must report NVFP4 ready: True).
 9. If ready, forge_publish (handles HF_HOME + Xet): publish-model <family> <quantized> --release-class public_quantized_model \
      --validation-state spark_single_node_validated --source-license-checked + the evidence paths.
      The family-config variant must NOT have promotion.blocked_actions:[hf_upload]; lift it once the gate passes.
Model-snapshot quirks to auto-fix before export:
 - missing generation_config.json  -> synthesize from config.json bos/eos/pad token ids.
 - non-standard shard names model.safetensors-*-of-*.safetensors  -> rename to model-*-of-*.safetensors AND
   update model.safetensors.index.json weight_map (model-forge globs model-*.safetensors).
"""

# Architecture -> (ptq strategy, script). Extend as new families are validated.
_ARCH_STRATEGY = {
    "qwen": ("qwen_text_modelopt", "scripts/quantization/qwen_text_modelopt.py"),
    "gemma": ("gemma4_moe_modelopt", "scripts/quantization/gemma4_moe_nvfp4.py"),
}

# quant method -> (backend, qformat, kv_cache_qformat, gate_key). nvfp4/fp8 export today via
# ModelOpt; int8/awq land with the model-forge Phase-2 export support.
_METHOD_SPEC = {
    "nvfp4": ("modelopt", "nvfp4", "fp8_cast", "nvfp4"),
    "fp8": ("modelopt", "fp8", "fp8_cast", "fp8"),
    "int8": ("modelopt", "int8", "auto", "int8"),
    "awq": ("autoawq", "int4_awq", "auto", "awq"),
}


def choose_quant_method(supported_quant, requested: str | None = None) -> str | None:
    """Pick the quant method for a GPU. ``supported_quant`` is the arch's methods (best-first,
    from gpu_inspect / model-forge). Honor an explicit request if supported; else take the best
    supported. Returns None when the GPU supports no quantization (serve bf16 instead)."""
    supported = [m for m in (supported_quant or []) if m in _METHOD_SPEC]
    if requested:
        return requested if requested in supported else None
    return supported[0] if supported else None


def scaffold_quant_config(
    family: str,
    *,
    method: str = "nvfp4",
    arch: str = "qwen",
    source_variant: str = "base",
    target_variant: str | None = None,
    calib_samples: int = 256,
    calib_seq: int = 1024,
    port: int = 8000,
    min_speedup: float = 1.3,
) -> str:
    """Render a quant config for ``family`` (speedup-gated). ``method`` is chosen for the GPU's
    arch (nvfp4/fp8/int8/awq); the backend/qformat/gate are derived from it."""
    backend, qformat, kv_cache_qformat, gate_key = _METHOD_SPEC.get(method, _METHOD_SPEC["nvfp4"])
    strategy, script = _ARCH_STRATEGY.get(arch, _ARCH_STRATEGY["qwen"])
    return QUANT_CONFIG_TEMPLATE.format(
        family=family,
        method=method,
        backend=backend,
        qformat=qformat,
        kv_cache_qformat=kv_cache_qformat,
        gate_key=gate_key,
        source_variant=source_variant,
        target_variant=target_variant or f"base_{method}_{backend}",
        strategy=strategy,
        script=script,
        calib_samples=calib_samples,
        calib_seq=calib_seq,
        port=port,
        min_speedup=min_speedup,
    )


def quant_config_name(family: str, method: str = "nvfp4") -> str:
    backend = _METHOD_SPEC.get(method, _METHOD_SPEC["nvfp4"])[0]
    return f"{family}_{method}_{backend}"


def write_quant_config(repo: Path, family: str, *, method: str = "nvfp4",
                       overwrite: bool = False, **kwargs) -> Path:
    """Write the scaffolded config into ``<repo>/configs/quantization/`` (idempotent)."""
    cfg = Path(repo) / "configs" / "quantization" / f"{quant_config_name(family, method)}.yaml"
    if cfg.exists() and not overwrite:
        return cfg
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(scaffold_quant_config(family, method=method, **kwargs), encoding="utf-8")
    return cfg
