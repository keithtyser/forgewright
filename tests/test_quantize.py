"""Tests for the quantize hardening (config scaffolding + capability-gated method selection)."""
from __future__ import annotations

from forgewright.skills.quantize import (
    choose_quant_method,
    quant_config_name,
    scaffold_quant_config,
    write_quant_config,
)


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


def test_choose_quant_method_prefers_best_supported():
    assert choose_quant_method(["nvfp4", "fp8", "int8", "awq"]) == "nvfp4"   # Blackwell
    assert choose_quant_method(["fp8", "int8", "awq"]) == "fp8"              # Hopper/Ada
    assert choose_quant_method(["int8", "awq", "gptq"]) == "int8"           # Ampere
    assert choose_quant_method([]) is None                                   # no quant -> bf16
    # an explicit request is honored only when the arch supports it
    assert choose_quant_method(["fp8", "int8"], requested="fp8") == "fp8"
    assert choose_quant_method(["int8", "awq"], requested="nvfp4") is None   # Ampere can't do NVFP4


def test_scaffold_fp8_method():
    cfg = scaffold_quant_config("qwen35_9b", method="fp8")
    assert "method: fp8" in cfg and "qformat: fp8" in cfg
    assert "name: qwen35_9b_fp8_modelopt" in cfg
    assert quant_config_name("qwen35_9b", "fp8") == "qwen35_9b_fp8_modelopt"


def test_write_quant_config_method_in_filename(tmp_path):
    p = write_quant_config(tmp_path, "qwen35_9b", method="fp8")
    assert p.name == "qwen35_9b_fp8_modelopt.yaml"
