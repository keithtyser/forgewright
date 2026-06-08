"""GPU/hardware inspection (local or remote): the hardware ground truth the agent derives a
training/quant strategy from. Reports each GPU's arch class, compute capability, VRAM, and the
quantization methods it can actually run (so the agent picks NVFP4 on Blackwell, FP8 on
Hopper/Ada, INT8/AWQ on Ampere, instead of assuming Blackwell). Mirrors the classification in
model-forge's ``hardware.py`` (kept in sync; model-forge runs in its own container)."""
from __future__ import annotations

import subprocess
from typing import Any

from forgewright.tools.base import Tool, ToolResult

_QUERY = "name,compute_cap,memory.total,memory.used,driver_version"


def _sm(cc: str) -> str:
    cc = cc.strip()
    return f"sm_{cc.replace('.', '')}" if cc[:1].isdigit() else cc


def arch_from_cc(cc: str) -> str:
    """Map a compute capability (e.g. '8.6', '9.0', '12.1') to an arch class."""
    try:
        major, minor = (int(x) for x in cc.strip().split(".")[:2])
    except (ValueError, TypeError):
        return "unknown"
    if major >= 10:
        return "blackwell"
    if (major, minor) == (9, 0):
        return "hopper"
    if (major, minor) == (8, 9):
        return "ada"
    if major == 8:
        return "ampere"
    if major == 7:
        return "volta_turing"
    return "unknown"


def supported_quant_for_cc(cc: str) -> tuple[str, ...]:
    """Quant methods this compute capability can run, best-first (NVFP4=Blackwell, FP8=Hopper/Ada,
    INT8/AWQ/GPTQ broadly)."""
    arch = arch_from_cc(cc)
    if arch == "blackwell":
        return ("nvfp4", "fp8", "int8", "awq")
    if arch in ("hopper", "ada"):
        return ("fp8", "int8", "awq")
    if arch == "ampere":
        return ("int8", "awq", "gptq")
    if arch == "volta_turing":
        return ("int8", "awq")
    return ()


class GPUInspectTool(Tool):
    name = "gpu_inspect"
    description = (
        "Inspect NVIDIA GPUs via nvidia-smi, locally or on a remote host (user@host). Returns "
        "per-GPU name, arch class (ampere/ada/hopper/blackwell), compute capability (SM), VRAM, "
        "driver, and the quantization methods the GPU supports (nvfp4/fp8/int8/awq). Use it to "
        "choose a training/quantization strategy that matches the hardware."
    )
    risk = "read"
    parameters = {
        "type": "object",
        "properties": {"host": {"type": "string", "description": "Optional user@host for remote inspection."}},
    }

    def run(self, host: str | None = None, **_: Any) -> ToolResult:
        cmd = ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"]
        try:
            argv = ["ssh", host, " ".join(cmd)] if host else cmd
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, f"gpu_inspect error: {e}")
        if proc.returncode != 0:
            return ToolResult(False, (proc.stderr or "nvidia-smi failed").strip())

        gpus: list[dict[str, Any]] = []
        for line in proc.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 5:
                continue
            name, cc, mtot, mused, drv = parts[:5]
            quant = supported_quant_for_cc(cc)
            gpus.append(
                {
                    "name": name,
                    "arch": arch_from_cc(cc),
                    "sm": _sm(cc),
                    "compute_cap": cc.strip(),
                    "vram_mib": mtot,
                    "used_mib": mused,
                    "driver": drv,
                    "supported_quant": list(quant),
                    "nvfp4_native": "nvfp4" in quant,
                }
            )
        if not gpus:
            return ToolResult(False, "no GPUs detected")
        summary = "\n".join(
            f"- {g['name']} | {g['arch']} {g['sm']} | {g['vram_mib']} MiB | drv {g['driver']}"
            + (" | quant: " + ", ".join(g["supported_quant"]) if g["supported_quant"] else " | quant: none")
            for g in gpus
        )
        return ToolResult(True, summary, {"gpus": gpus, "count": len(gpus), "host": host or "local"})
