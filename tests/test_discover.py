"""Tests for the discovery tool (what's available, for plan mode)."""
from __future__ import annotations

from pathlib import Path

from forgewright.tools.discover import discover_assets


def test_discover_assets(tmp_path: Path):
    models = tmp_path / "models"
    (models / "Qwen3.5-0.8B").mkdir(parents=True)
    (models / "Qwen3.5-9B").mkdir()
    (models / ".cache").mkdir()                 # hidden -> excluded
    (models / "note.txt").write_text("x")       # file -> excluded
    repo = tmp_path / "model-forge"
    (repo / "datasets" / "finetuning").mkdir(parents=True)
    (repo / "datasets" / "rl").mkdir(parents=True)
    (repo / "configs" / "model_families").mkdir(parents=True)
    (repo / "datasets" / "finetuning" / "distill_smoke.jsonl").write_text("")
    (repo / "datasets" / "rl" / "mult_train.jsonl").write_text("")
    (repo / "configs" / "model_families" / "qwen35_0_8b.yaml").write_text("")

    a = discover_assets(models, repo)
    assert a["models"] == ["Qwen3.5-0.8B", "Qwen3.5-9B"]            # dirs only, no hidden/files
    assert "datasets/finetuning/distill_smoke.jsonl" in a["datasets"]
    assert "datasets/rl/mult_train.jsonl" in a["datasets"]
    assert a["families"] == ["qwen35_0_8b"]


def test_discover_assets_empty(tmp_path: Path):
    a = discover_assets(tmp_path / "nope", tmp_path / "norepo")
    assert a == {"models": [], "datasets": [], "families": []}
