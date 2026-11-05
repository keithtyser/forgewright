"""Register the exact checkpoint consumed by a stage."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from forgewright.contracts import Artifact
from forgewright.skills.register_family import write_family_registration


def register_model(repo: Path, model: Artifact, *, variant: str | None = None) -> str:
    family = model.meta.get("family")
    variant = variant or model.meta.get("variant") or model.id.replace("-", "_")
    for value in (family, variant):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Model family and variant must be safe, nonempty identifiers")
    if not model.uri:
        raise ValueError("Model artifact has no checkpoint URI")
    assert isinstance(family, str)
    path = Path(repo) / "configs" / "model_families" / f"{family}.yaml"
    if not path.exists():
        write_family_registration(repo, family, source=model.uri)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    uri = model.uri
    variants = cfg.setdefault("variants", {})
    existing = dict(variants.get(variant) or {})
    # A registered Hub ID may already point at a downloaded snapshot. Keep that
    # mapping; explicit local paths must not change with MODEL_FORGE_MODELS_DIR.
    local = str(Path(uri).expanduser()) if uri.startswith("~/") else uri
    if not uri.startswith(("/", "~", ".")) and existing.get("repo_id") == uri and existing.get("local_dir"):
        local = existing["local_dir"]
    elif not Path(uri).is_absolute() and not uri.startswith(("/", "~", ".")):
        candidate = Path(repo) / uri
        local = str(candidate.resolve()) if candidate.exists() else uri.rstrip("/").split("/")[-1]
    entry = {
        **existing,
        "repo_id": model.meta.get("base") or uri,
        "local_dir": local,
        "served_model_name": model.meta.get("served_model_name") or f"model-forge/{family}-{variant}",
    }
    # A standalone checkpoint must not resolve through an old adapter path.
    entry.pop("merged_local_dir", None)
    entry.pop("adapter", None)
    if variant != "base":
        entry["base_variant"] = model.meta.get("source_variant") or existing.get("base_variant") or "base"
    if model.meta.get("role") == "quantized":
        entry["quantization"] = str(model.meta["method"]).removesuffix("_modelopt")
        entry["downloadable"] = False
    variants[variant] = entry
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return variant
