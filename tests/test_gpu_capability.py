"""Tests for gpu_inspect's capability derivation (cc -> arch -> supported quant)."""
from __future__ import annotations

from forgewright.tools.gpu import arch_from_cc, supported_quant_for_cc


def test_arch_from_cc():
    assert arch_from_cc("8.0") == "ampere"
    assert arch_from_cc("8.9") == "ada"
    assert arch_from_cc("9.0") == "hopper"
    assert arch_from_cc("12.1") == "blackwell"
    assert arch_from_cc("7.5") == "volta_turing"
    assert arch_from_cc("") == "unknown"


def test_supported_quant_matches_arch():
    assert supported_quant_for_cc("12.1")[0] == "nvfp4"        # Blackwell prefers NVFP4
    assert "nvfp4" not in supported_quant_for_cc("9.0")        # Hopper has no FP4
    assert supported_quant_for_cc("9.0")[0] == "fp8"
    assert "fp8" not in supported_quant_for_cc("8.0")          # Ampere has no FP8
    assert supported_quant_for_cc("8.0")[0] == "int8"
    assert supported_quant_for_cc("") == ()


def test_gpu_inspect_classifies_capability(monkeypatch):
    import subprocess

    from forgewright.tools.gpu import GPUInspectTool

    class _Proc:
        returncode = 0
        stdout = "NVIDIA A100-SXM4-80GB, 8.0, 81920, 1024, 550.00\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    res = GPUInspectTool().run()
    assert res.ok
    g = res.meta["gpus"][0]
    assert g["arch"] == "ampere" and g["compute_cap"] == "8.0"
    assert g["supported_quant"] == ["int8", "awq", "gptq"] and g["nvfp4_native"] is False
    assert "ampere" in res.output and "quant: int8" in res.output
