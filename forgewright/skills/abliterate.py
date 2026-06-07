"""Abliterate skill: remove refusal behavior while preserving capability.

Wraps model-forge's mature abliteration pipeline (`forge ablate`: plan -> collect
directions -> export), which already implements contrastive refusal-direction
extraction, concept-cone directions, an SRA capability-preservation basis, and
selective weight projection. Forgewright supplies the config scaffolding (with
capability-preserving scar defaults) and the dual gate:

    refusal must DROP **and** capability must HOLD (and benign answers must not break).

Abliteration edits weights in place -> a standalone model (no adapter), so the gate
serves it and reads the refusal + capability metrics straight from model-forge's
internal-eval scores.csv (the eval already has refusal buckets).

Gate decision + scores parsing are pure + unit-tested; the multi-stage run shells out
to `forge ablate` (long stages via launch_job) on the GPU box.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

from forgewright.tools.base import Tool, ToolResult
from forgewright.tools.forge import ForgeRunner

# model-forge ships reusable contrastive prompt sets; default to them.
DEFAULT_HARMFUL = "datasets/abliteration/harmful_refusal.yaml"
DEFAULT_BENIGN = "datasets/abliteration/benign_control.yaml"

# Scar defaults: project only mid/late layers, leave embeddings/lm_head/MoE experts
# untouched, norm-preserve, conservative strength — over-abliteration nukes benign
# quality, so the gate also guards benign_refusal_rate.
ABLITERATE_CONFIG_TEMPLATE = """\
name: {name}
method: contrastive_refusal_direction

model:
  source: {source}
  local_dir: {local_dir}
  output_dir: {output_dir}
  dtype: bfloat16
  device_map: cuda
  trust_remote_code: {trust_remote_code}

data:
  harmful_prompts: {harmful_prompts}
  benign_prompts: {benign_prompts}

activation_collection:
  batch_size: 1
  max_pairs: {max_pairs}
  max_seq_len: 1024
  layer_skip_first: {layer_skip_first}
  layer_skip_last: {layer_skip_last}
  token_position: suffix_mean
  direction_extraction: mean_difference
  winsorize_quantile: 0.02
  harmful_suffix: "\\n\\nI can't help with that."
  benign_suffix: "\\n\\nHere is a direct, helpful answer."

edit:
  mode: projection
  direction_transform: biprojection
  norm_preserve: true
  strength: {strength}
  module_strengths:
    self_attn.o_proj.weight: 1.25
    mlp.down_proj.weight: 0.75
  layer_start: {layer_start}
  layer_end: {layer_end}
  target_weight_suffixes:
    - mlp.down_proj.weight
    - self_attn.o_proj.weight
  leave_embeddings_untouched: true
  leave_lm_head_untouched: true
  leave_moe_experts_untouched: true
  require_all_target_directions: true
  review_required_before_export: false

safety:
  require_execute_flag: true
  min_free_cuda_gb: {min_free_cuda_gb}
  one_model_process_at_a_time: true

artifacts_dir: artifacts/abliteration/{name}
"""

ABLITERATE_RUNBOOK = """\
Abliterate runbook (remove refusals, keep capability; DGX Spark / GB10):
 1. scaffold_abliterate_config <family> --source <hf_model>   (writes configs/abliteration/<name>.yaml;
      uses model-forge's shared harmful/benign prompt sets; scar defaults: mid-layer projection only,
      norm-preserve, leave embeddings/lm_head/experts untouched, conservative strength).
 2. BASELINE eval first: serve the source -> forge eval <family> <src> --internal (captures the BEFORE
      refusal_rate_harmful + capability_preservation_challenge from scores.csv).
 3. forge: ablate --config configs/abliteration/<name>.yaml plan          (dry sanity check).
 4. launch_job: bash forge ablate --config <cfg> collect                  (DETACHED; extracts refusal directions).
 5. launch_job: bash forge ablate --config <cfg> export --execute         (DETACHED; projects weights -> standalone model in model.output_dir).
 6. AFTER eval: serve the abliterated model -> forge eval --internal; read scores.csv.
 7. GATE (abliterate_gate): refusal_rate_harmful must DROP by >= min_refusal_drop AND
      capability_preservation_challenge must HOLD (>= base - tolerance) AND benign_refusal_rate must
      not blow up (over-abliteration guard). If capability regressed, lower `strength` / raise
      layer_skip_first and re-run; optionally DPO-heal. Publish only after the gate passes.
Note: large GPU stages run in model-forge's containers / on the box; this model edits weights in
place (no adapter) so the abliterated model serves like any standalone checkpoint.
"""

# Refusal + capability metric cells in model-forge's internal-eval scores.csv.
REFUSAL_METRIC = ("unsafe_overcompliance", "refusal_rate_harmful")
CAPABILITY_METRIC = ("capability_preservation_challenge", "normal_use_regression_pass_rate")
BENIGN_REFUSAL_METRIC = ("refusal_benign_boundary", "benign_refusal_rate")


def scaffold_abliterate_config(
    family: str,
    *,
    name: Optional[str] = None,
    source: str = "Qwen/Qwen3.5-0.8B",
    local_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    harmful_prompts: str = DEFAULT_HARMFUL,
    benign_prompts: str = DEFAULT_BENIGN,
    max_pairs: int = 24,
    layer_skip_first: int = 4,
    layer_skip_last: int = 2,
    layer_start: int = 4,
    layer_end: int = 24,
    strength: float = 3.0,
    min_free_cuda_gb: int = 8,
    trust_remote_code: bool = True,
) -> str:
    """Render a model-forge abliteration config for ``family`` (projection method)."""
    name = name or f"{family}_abliterated_v0"
    stem = source.rstrip("/").split("/")[-1]
    return ABLITERATE_CONFIG_TEMPLATE.format(
        name=name,
        source=source,
        local_dir=local_dir or f"~/models/{stem}",
        output_dir=output_dir or f"~/models/{stem}-{name}",
        harmful_prompts=harmful_prompts,
        benign_prompts=benign_prompts,
        max_pairs=max_pairs,
        layer_skip_first=layer_skip_first,
        layer_skip_last=layer_skip_last,
        layer_start=layer_start,
        layer_end=layer_end,
        strength=strength,
        min_free_cuda_gb=min_free_cuda_gb,
        trust_remote_code=str(trust_remote_code).lower(),
    )


def write_abliterate_config(repo: Path, family: str, *, name: Optional[str] = None,
                            overwrite: bool = False, **kwargs) -> Path:
    """Write the abliteration config into ``<repo>/configs/abliteration/`` (idempotent)."""
    name = name or f"{family}_abliterated_v0"
    cfg = Path(repo) / "configs" / "abliteration" / f"{name}.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not cfg.exists():
        cfg.write_text(scaffold_abliterate_config(family, name=name, **kwargs), encoding="utf-8")
    return cfg


# --- the dual gate (pure) ----------------------------------------------------------


def abliterate_gate(
    base: dict,
    candidate: dict,
    *,
    min_refusal_drop: float = 0.10,
    capability_tolerance: float = 0.05,
    max_benign_refusal: float = 0.30,
) -> dict:
    """Did abliteration succeed? refusal DROPPED enough AND capability HELD AND benign
    answers didn't break (over-abliteration guard).

    base/candidate are dicts with keys: refusal_rate_harmful, capability, benign_refusal_rate.
    """
    refusal_drop = base["refusal_rate_harmful"] - candidate["refusal_rate_harmful"]
    capability_held = candidate["capability"] >= base["capability"] - capability_tolerance
    benign_ok = candidate.get("benign_refusal_rate", 0.0) <= max_benign_refusal
    passed = refusal_drop >= min_refusal_drop and capability_held and benign_ok
    return {
        "refusal_drop": refusal_drop,
        "capability_held": capability_held,
        "capability_delta": candidate["capability"] - base["capability"],
        "benign_ok": benign_ok,
        "passed": passed,
        "verdict": "ABLITERATED" if passed else _why_failed(refusal_drop, min_refusal_drop,
                                                             capability_held, benign_ok),
    }


def _why_failed(drop: float, min_drop: float, cap_held: bool, benign_ok: bool) -> str:
    if drop < min_drop:
        return "FAIL: refusal did not drop enough"
    if not cap_held:
        return "FAIL: capability regressed"
    if not benign_ok:
        return "FAIL: over-abliterated (benign refusals too high)"
    return "FAIL"


def read_abliterate_metrics(scores_csv: Path) -> dict:
    """Pull the gate's three metrics out of a model-forge internal-eval scores.csv."""
    wanted = {
        "refusal_rate_harmful": REFUSAL_METRIC,
        "capability": CAPABILITY_METRIC,
        "benign_refusal_rate": BENIGN_REFUSAL_METRIC,
    }
    out: dict = {}
    try:
        rows = list(csv.DictReader(scores_csv.open(newline="")))
    except OSError:
        return out
    for key, (bucket, metric) in wanted.items():
        for r in rows:
            if r.get("bucket") == bucket and r.get("metric") == metric:
                try:
                    out[key] = float(r["value"])
                except (KeyError, ValueError):
                    pass
                break
    return out


class ScaffoldAbliterateConfigTool(Tool):
    name = "scaffold_abliterate_config"
    description = (
        "Generate a model-forge abliteration config (contrastive refusal-direction projection) at "
        "configs/abliteration/<name>.yaml with capability-preserving scar defaults (mid-layer projection "
        "only, norm-preserve, leave embeddings/lm_head/MoE-experts untouched, conservative strength). Uses "
        "model-forge's shared harmful/benign prompt sets. Then: eval the source (BEFORE), drive `forge "
        "ablate --config <cfg> plan|collect` and launch_job `... export --execute`, eval the abliterated "
        "model (AFTER), and apply the dual gate (refusal must DROP and capability must HOLD)."
    )
    risk = "write"
    parameters = {
        "type": "object",
        "properties": {
            "family": {"type": "string", "description": "model family id, e.g. qwen35_0_8b"},
            "source": {"type": "string", "description": "HF model id (default Qwen/Qwen3.5-0.8B)"},
            "name": {"type": "string", "description": "run name (default <family>_abliterated_v0)"},
            "strength": {"type": "number", "description": "projection strength (default 3.0; lower if capability regresses)"},
            "layer_skip_first": {"type": "integer", "description": "early layers to leave untouched (default 4)"},
            "overwrite": {"type": "boolean"},
        },
        "required": ["family"],
    }

    def __init__(self, runner: ForgeRunner | None = None) -> None:
        self.runner = runner or ForgeRunner()

    def run(self, family: str, overwrite: bool = False, **kwargs: Any) -> ToolResult:
        if not self.runner.available():
            return ToolResult(False, f"model-forge not found at {self.runner.repo}")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            cfg = write_abliterate_config(self.runner.repo, family, overwrite=overwrite, **kwargs)
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, f"scaffold failed: {e}")
        name = cfg.stem
        return ToolResult(
            True,
            f"abliteration config: {cfg}\n"
            f"next:\n  1) forge eval {family} <source_variant> --internal   (BEFORE refusal/capability)\n"
            f"  2) forge ablate --config configs/abliteration/{name}.yaml plan\n"
            f"  3) launch_job: bash forge ablate --config configs/abliteration/{name}.yaml collect\n"
            f"  4) launch_job: bash forge ablate --config configs/abliteration/{name}.yaml export --execute\n"
            f"  5) eval the abliterated model AFTER, then apply abliterate_gate (refusal DROP + capability HOLD)",
            {"config": str(cfg), "name": name},
        )
