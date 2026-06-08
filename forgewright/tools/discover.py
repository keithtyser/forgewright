"""Discovery tool: what models, datasets, families, and recipes are available here.

This is what makes a casual goal workable. Instead of demanding exact paths, the agent
calls `discover` to see the local models, the curated/RL datasets, the registered families,
and the recipe menu, then proposes a concrete plan with sensible defaults. Read-only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from forgewright.tools.base import Tool, ToolResult
from forgewright.tools.forge import ForgeRunner


def _models_dir() -> Path:
    return Path(os.environ.get("MODEL_FORGE_MODELS_DIR", str(Path.home() / "models")))


def discover_assets(models_dir: Path, repo: Path) -> dict[str, Any]:
    """Scan for local models, datasets, and registered families (pure, fs-only)."""
    models_dir, repo = Path(models_dir), Path(repo)
    models = sorted(p.name for p in models_dir.glob("*")
                    if p.is_dir() and not p.name.startswith(".")) if models_dir.exists() else []
    datasets: list[str] = []
    for sub in ("datasets/finetuning", "datasets/rl"):
        d = repo / sub
        if d.exists():
            datasets += sorted(f"{sub}/{p.name}" for p in d.glob("*.jsonl"))
    fam_dir = repo / "configs" / "model_families"
    families = sorted(p.stem for p in fam_dir.glob("*.yaml")) if fam_dir.exists() else []
    return {"models": models, "datasets": datasets, "families": families}


class DiscoverTool(Tool):
    name = "discover"
    description = (
        "List what is available here so you can propose a concrete plan from a vague goal: local "
        "models (~/models), curated + RL datasets (model-forge datasets/finetuning, datasets/rl), "
        "registered model families, and the recipe menu. Call this FIRST when the user's goal does "
        "not name an exact model/dataset, then propose a plan. Read-only."
    )
    risk = "read"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, runner: ForgeRunner | None = None) -> None:
        self.runner = runner or ForgeRunner()

    def run(self, **_: Any) -> ToolResult:
        from forgewright.agents.recipes import RECIPES

        assets = discover_assets(_models_dir(), self.runner.repo)
        lines = [
            "local models (~/models): " + (", ".join(assets["models"]) or "(none)"),
            "datasets: " + (", ".join(assets["datasets"]) or "(none)"),
            "registered families: " + (", ".join(assets["families"]) or "(none)"),
            "recipes: " + ", ".join(sorted(RECIPES)),
        ]
        return ToolResult(True, "\n".join(lines), {**assets, "recipes": sorted(RECIPES)})
