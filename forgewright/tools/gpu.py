"""GPU/hardware inspection (local or remote) with Blackwell / NVFP4 awareness."""
from __future__ import annotations

import subprocess
from typing import Any

from forgewright.tools.base import Tool, ToolResult

_QUERY = "name,compute_cap,memory.total,memory.used,driver_version"
_NVFP4_NATIVE_CC = {"12.0", "12.1"}  # Blackwell SM120 / SM121 → native NVFP4 tensor cores


def _sm(cc: str) -> str:
    cc = cc.strip()
    return f"sm_{cc.replace('.', '')}" if cc[:1].isdigit() else cc


class GPUInspectTool(Tool):
    name = "gpu_inspect"
    description = (
        "Inspect NVIDIA GPUs via nvidia-smi, locally or on a remote host (user@host). "
        "Returns per-GPU name, compute capability (SM), VRAM, and driver, flagging Blackwell "
        "(NVFP4-native) parts. Use it to choose a training/quantization strategy for the hardware."
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
            gpus.append(
                {
                    "name": name,
                    "sm": _sm(cc),
                    "vram_mib": mtot,
                    "used_mib": mused,
                    "driver": drv,
                    "nvfp4_native": cc.strip() in _NVFP4_NATIVE_CC,
                }
            )
        if not gpus:
            return ToolResult(False, "no GPUs detected")
        summary = "\n".join(
            f"- {g['name']} | {g['sm']} | {g['vram_mib']} MiB | drv {g['driver']}"
            + (" | NVFP4-native (Blackwell)" if g["nvfp4_native"] else "")
            for g in gpus
        )
        return ToolResult(True, summary, {"gpus": gpus, "count": len(gpus), "host": host or "local"})
