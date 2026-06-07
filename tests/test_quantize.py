"""Tests for the quantize hardening (config scaffolding)."""
from __future__ import annotations

from forgewright.skills.quantize import scaffold_quant_config, write_quant_config


def test_scaffold_qwen():
    cfg = scaffold_quant_config("qwen35_9b", min_speedup=1.3)
    assert "family: qwen35_9b" in cfg
    assert "method: nvfp4" in cfg
    assert "qwen_text_modelopt" in cfg
    assert "min_output_speedup: 1.3" in cfg
    assert "gates:" in cfg
    # speedup-gated => NO static tok/s floor key
    assert "min_output_tokens_per_second" not in cfg


def test_scaffold_gemma_strategy():
    cfg = scaffold_quant_config("gemma4_26b_a4b", arch="gemma")
    assert "gemma4_moe_modelopt" in cfg


def test_write_quant_config_idempotent(tmp_path):
    p = write_quant_config(tmp_path, "qwen35_9b")
    assert p.exists()
    assert p.name == "qwen35_9b_nvfp4_modelopt.yaml"
    first = p.read_text(encoding="utf-8")
    p2 = write_quant_config(tmp_path, "qwen35_9b")  # idempotent, no overwrite
    assert p2 == p
    assert p2.read_text(encoding="utf-8") == first
