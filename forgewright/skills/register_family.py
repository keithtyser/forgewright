"""Auto-register a model with model-forge so `forge serve` / `forge eval` work for it.

model-forge keys serve + eval off two configs: `configs/model_families/<family>.yaml` and
`configs/experiments/<family>_v0.yaml`. Hand-writing them per model is the friction that
blocked the serve-based gates (e.g. for Qwen3.5-0.8B). This derives both from a proven
reference family (default qwen35_9b) by substituting the identity fields, so any qwen-family
model becomes servable + evaluable without hand config. Derivation is pure + unit-tested.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml


def _stem(source: str) -> str:
    return source.rstrip("/").split("/")[-1]


def _architecture_from_spec(spec: dict, ref_arch: Any) -> dict:
    """Rewrite the family config's architecture block from the model's ModelSpec, so serve/eval/
    target-discovery match the actual model (not the qwen reference). Only qwen models keep the
    qwen3 reasoning-parser / enable_thinking chat-template knobs."""
    arch = copy.deepcopy(ref_arch) if isinstance(ref_arch, dict) else {}
    mt = (spec.get("model_type") or "").lower()
    fam = mt.split("_")[0] or "unknown"
    arch["family"] = fam
    arch["tokenizer_family"] = mt or fam
    if "qwen" in mt:
        arch["chat_template"] = {
            "default_chat_template_kwargs": {"enable_thinking": False},
            "reasoning_parser": "qwen3",
        }
    else:
        arch.pop("chat_template", None)   # let vLLM use the model's own tokenizer chat template
    td = arch.setdefault("target_discovery", {})
    td["inspect_before_training_or_ablation"] = True
    if spec.get("attn_modules"):
        td["common_attention_patterns"] = list(spec["attn_modules"])
    if spec.get("mlp_modules"):
        td["common_mlp_patterns"] = list(spec["mlp_modules"])
    td.setdefault("edit_exclusion_patterns", ["embed_tokens", "lm_head", "router", "expert"])
    td["router_or_expert_policy"] = "inspect_model_config"
    return arch


def derive_family_config(
    ref: dict, *, family: str, source: str, local_dir: Optional[str] = None,
    extra_variants: Optional[dict[str, dict]] = None, spec: Optional[dict] = None,
) -> dict:
    """Substitute identity fields of a reference model-family config for a new family. When a
    ModelSpec is given, the architecture block is rewritten from it so non-qwen models get the
    right serve/target structure (else the reference's qwen-family structure is kept)."""
    cfg = copy.deepcopy(ref)
    cfg["name"] = family
    cfg["display_name"] = family.replace("_", " ")
    local = local_dir or _stem(source)
    cfg["variants"] = {
        "base": {"repo_id": source, "local_dir": local, "served_model_name": source},
    }
    if extra_variants:
        cfg["variants"].update(copy.deepcopy(extra_variants))
    if spec:
        cfg["architecture"] = _architecture_from_spec(spec, cfg.get("architecture"))
    # repoint the eval cross-references at this family's experiment configs
    ev = cfg.get("eval")
    if isinstance(ev, dict):
        ev["config"] = f"configs/experiments/{family}_v0.yaml"
        ev["artifact_config"] = f"configs/experiments/{family}_artifacts_v0.yaml"
        ev["output_root"] = f"results/{family}_v0/base"
        ev["run_prefix"] = family
    return cfg


def derive_experiment_config(ref: dict, *, family: str, source: str, variant: str = "base") -> dict:
    """Substitute identity fields of a reference experiment (eval) config."""
    cfg = copy.deepcopy(ref)
    cfg["experiment_name"] = f"{family}_v0_{variant}_eval"
    model = cfg.setdefault("model", {})
    model["family"], model["id"], model["variant"] = family, source, variant
    backend = cfg.get("backend")
    if isinstance(backend, dict):
        backend["model_alias"] = f"{family}_local"
    ev = cfg.get("eval")
    if isinstance(ev, dict):
        ev["output_dir"] = f"results/{family}_v0/{variant}"
    return cfg


def write_family_registration(
    repo: Path, family: str, *, source: str, reference_family: str = "qwen35_9b",
    local_dir: Optional[str] = None, extra_variants: Optional[dict[str, dict]] = None,
    overwrite: bool = False, spec: Optional[dict] = None,
) -> tuple[Path, Path]:
    """Read the reference family + experiment configs and write derived ones for `family`. When a
    ModelSpec is given, the architecture is derived from the actual model (non-qwen support).
    Returns (family_config_path, experiment_config_path)."""
    repo = Path(repo)
    fam_dst = repo / "configs" / "model_families" / f"{family}.yaml"
    exp_dst = repo / "configs" / "experiments" / f"{family}_v0.yaml"
    fam_ref = repo / "configs" / "model_families" / f"{reference_family}.yaml"
    exp_ref = repo / "configs" / "experiments" / f"{reference_family}_v0.yaml"
    if not fam_ref.exists() or not exp_ref.exists():
        raise FileNotFoundError(f"reference configs missing for '{reference_family}'")
    fam_dst.parent.mkdir(parents=True, exist_ok=True)
    exp_dst.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not fam_dst.exists():
        fam = derive_family_config(yaml.safe_load(fam_ref.read_text()), family=family,
                                   source=source, local_dir=local_dir, extra_variants=extra_variants,
                                   spec=spec)
        fam_dst.write_text(yaml.safe_dump(fam, sort_keys=False), encoding="utf-8")
    if overwrite or not exp_dst.exists():
        exp = derive_experiment_config(yaml.safe_load(exp_ref.read_text()), family=family, source=source)
        exp_dst.write_text(yaml.safe_dump(exp, sort_keys=False), encoding="utf-8")
    return fam_dst, exp_dst
