"""Scaffold a model-forge data-factory config so DataCurator can teacher-distill.

model-forge's data factory (`forge data generate -> judge -> verify -> filter -> pack`)
reads `configs/datasets/<family>_<variant>.yaml` + a source registry. Hand-writing the rich
config (generation methods, strategies, quality thresholds, skills) is friction; this derives
both from a proven reference dataset config by substituting identity + seed/source fields.

The factory runs the whole pipeline with `--provider template` (deterministic, no LLM) for
wiring validation, or `--provider openai_compatible` with a teacher endpoint (OpenRouter /
local vLLM via the provider's *_env vars) for real `<think>` CoT distillation.
Derivation is pure + unit-tested.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import yaml

_DEFAULT_REFERENCE = "gemma4_26b_a4b_local_ft_v1_live_teacher_smoke"

_SOURCE_REGISTRY_TEMPLATE = {
    "version": "0.1.0",
    "description": "Forgewright distillation source catalog.",
}


def derive_dataset_config(ref: dict, *, family: str, variant: str, seed_path: str) -> dict:
    """Substitute identity + seed/source fields of a reference data-factory config. Keeps the
    generation methods / strategies / quality thresholds / skills / holdouts structure."""
    cfg = copy.deepcopy(ref)
    ds_id = f"{family}_{variant}"
    cfg["id"] = ds_id
    cfg["family"] = family
    cfg["variant"] = variant
    cfg["objective"] = "capability_sft"
    cfg["output_dir"] = f"datasets/generated/{ds_id}"
    cfg["seed_paths"] = [seed_path]
    cfg["source_registry"] = f"configs/data_sources/{ds_id}.yaml"
    cfg["source_ids"] = [f"{ds_id}_seed"]
    cfg.pop("baseline", None)   # reference-specific target metrics
    # the GB10 disk runs ~85% full; relax the factory's over-strict 15% floor (matches the
    # finetune resource_policy default)
    gen = cfg.get("generation")
    if isinstance(gen, dict):
        rl = gen.setdefault("resource_limits", {})
        rl["min_free_disk_ratio"] = 0.05
        rl["min_free_memory_ratio"] = 0.03
    return cfg


def derive_source_registry(*, family: str, variant: str, seed_path: str) -> dict:
    """The one-source registry the dataset config points at (the seed JSONL)."""
    ds_id = f"{family}_{variant}"
    reg = copy.deepcopy(_SOURCE_REGISTRY_TEMPLATE)
    reg["sources"] = {
        f"{ds_id}_seed": {
            "name": f"{ds_id}_seed", "type": "local_jsonl", "path": seed_path,
            "license": "CC-BY-4.0", "quality_tier": "human_seed",
            "roles": ["capability_sft", "teacher_distillation"], "messages_field": "messages",
        }
    }
    return reg


def write_dataset_config(
    repo: Path, family: str, *, seed_path: str, variant: str = "curated_v0",
    reference: str = _DEFAULT_REFERENCE, overwrite: bool = False,
) -> tuple[Path, Path]:
    """Read the reference data-factory config and write a derived config + source registry.
    Returns (dataset_config_path, source_registry_path)."""
    repo = Path(repo)
    ref_path = repo / "configs" / "datasets" / f"{reference}.yaml"
    if not ref_path.exists():
        raise FileNotFoundError(f"reference dataset config missing: {ref_path}")
    ds_id = f"{family}_{variant}"
    cfg_dst = repo / "configs" / "datasets" / f"{ds_id}.yaml"
    reg_dst = repo / "configs" / "data_sources" / f"{ds_id}.yaml"
    cfg_dst.parent.mkdir(parents=True, exist_ok=True)
    reg_dst.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not cfg_dst.exists():
        cfg = derive_dataset_config(yaml.safe_load(ref_path.read_text()),
                                    family=family, variant=variant, seed_path=seed_path)
        cfg_dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if overwrite or not reg_dst.exists():
        reg = derive_source_registry(family=family, variant=variant, seed_path=seed_path)
        reg_dst.write_text(yaml.safe_dump(reg, sort_keys=False), encoding="utf-8")
    return cfg_dst, reg_dst
