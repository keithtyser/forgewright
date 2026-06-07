"""Tools that drive model-forge's ``./forge`` CLI — the ML-ops engine on the GPU box.

Forgewright runs ON the box for ML-ops; these tools shell out to ``./forge <args>``
in the model-forge repo, which orchestrates the Docker-based NVFP4 / abliteration /
eval / serve / publish stages. Set ``FORGE_REPO`` to override the repo path.

Quick stages (plan, nvfp4-gate, reports, small evals) run via the ``forge`` tool;
LONG stages (quantize export, finetune/ablate run) should be launched as detached
jobs via ``launch_job`` and polled. Publishing is split into ``forge_publish``
(risk=destructive) so it always hits the approval gate.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from forgewright.tools.base import Tool, ToolResult


def _forge_repo() -> Path:
    return Path(os.environ.get("FORGE_REPO", str(Path.home() / "projects" / "model-forge")))


class ForgeRunner:
    """Runs ``./forge <args>`` inside the model-forge repo with dry-run support."""

    def __init__(self, repo: Path | None = None) -> None:
        self.repo = Path(repo) if repo else _forge_repo()

    def available(self) -> bool:
        return (self.repo / "forge").exists()

    def run(
        self,
        args: str,
        *,
        dry_run: bool = False,
        timeout: int = 3600,
        extra_env: dict | None = None,
    ) -> ToolResult:
        if not self.available():
            return ToolResult(
                False,
                f"model-forge not found at {self.repo} (set FORGE_REPO, or run Forgewright on the GPU box)",
                {"missing": True},
            )
        env = dict(os.environ)
        if dry_run:
            env["MODEL_FORGE_DRY_RUN"] = "1"
        if extra_env:
            env.update({k: str(v) for k, v in extra_env.items()})
        cmd = ["bash", "forge", *shlex.split(args)]
        try:
            proc = subprocess.run(
                cmd, cwd=str(self.repo), env=env, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"`forge {args}` timed out after {timeout}s", {"timeout": True})
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, f"failed to run `forge {args}`: {e}", {"error": True})
        body = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return ToolResult(
            proc.returncode == 0,
            body or f"(exit {proc.returncode})",
            {"exit_code": proc.returncode, "args": args, "dry_run": dry_run},
        ).truncate(16000)


class ForgeTool(Tool):
    name = "forge"
    description = (
        "Run model-forge's ./forge CLI to drive ML-ops stages (quantize, eval, serve, ablate, "
        "finetune, data, variants, hf plan-*, etc.). Pass the subcommand + args as one string, e.g. "
        "'quantize plan --config configs/quantization/gemma4_26b_a4b_nvfp4_modelopt.yaml --family gemma4_26b_a4b'. "
        "Use dry_run=true (MODEL_FORGE_DRY_RUN=1) or the 'plan' subcommands to rehearse before a real "
        "'export'. For LONG stages (quantize export, finetune/ablate run) prefer launch_job instead so "
        "they detach. Do NOT publish with this tool — use forge_publish."
    )
    risk = "exec"
    parameters = {
        "type": "object",
        "properties": {
            "args": {"type": "string", "description": "forge subcommand and arguments"},
            "dry_run": {"type": "boolean", "description": "rehearse without side effects"},
            "timeout": {"type": "integer", "description": "seconds (default 3600)"},
        },
        "required": ["args"],
    }

    def __init__(self, runner: ForgeRunner | None = None) -> None:
        self.runner = runner or ForgeRunner()

    def run(self, args: str, dry_run: bool = False, timeout: int = 3600, **_: Any) -> ToolResult:
        return self.runner.run(args, dry_run=dry_run, timeout=timeout)


class ForgePublishTool(Tool):
    name = "forge_publish"
    description = (
        "Publish a model or dataset to Hugging Face via model-forge "
        "(./forge hf publish-model | publish-dataset ...). IRREVERSIBLE — requires approval. "
        "Always rehearse first with the `forge` tool ('hf plan-model ...' or 'hf publish-* --dry-run')."
    )
    risk = "destructive"
    parameters = {
        "type": "object",
        "properties": {
            "args": {
                "type": "string",
                "description": "hf publish args, e.g. 'publish-model gemma4_26b_a4b base_nvfp4_modelopt --release-class public_quantized_model'",
            },
            "timeout": {"type": "integer", "description": "seconds (default 1800)"},
        },
        "required": ["args"],
    }

    def __init__(self, runner: ForgeRunner | None = None) -> None:
        self.runner = runner or ForgeRunner()

    def run(self, args: str, timeout: int = 1800, **_: Any) -> ToolResult:
        # Bake in the publish workarounds: redirect the HF cache off the root-owned
        # ~/.cache/huggingface and disable Xet (both caused permission failures).
        hf_home = os.path.expanduser("~/.forgewright/hf_home")
        os.makedirs(hf_home, exist_ok=True)
        return self.runner.run(
            f"hf {args}",
            timeout=timeout,
            extra_env={"HF_HOME": hf_home, "HF_HUB_DISABLE_XET": "1"},
        )


class ScaffoldQuantConfigTool(Tool):
    name = "scaffold_quant_config"
    description = (
        "Generate a model-forge NVFP4 ModelOpt quant config for a family (speedup-based gate, no "
        "static tok/s floor) at configs/quantization/<family>_nvfp4_modelopt.yaml. Run this before "
        "quantizing a family that lacks a config. arch: 'qwen' or 'gemma'."
    )
    risk = "write"
    parameters = {
        "type": "object",
        "properties": {
            "family": {"type": "string"},
            "arch": {"type": "string", "description": "qwen | gemma (default qwen)"},
            "source_variant": {"type": "string", "description": "default 'base'"},
            "overwrite": {"type": "boolean"},
        },
        "required": ["family"],
    }

    def __init__(self, runner: ForgeRunner | None = None) -> None:
        self.runner = runner or ForgeRunner()

    def run(
        self, family: str, arch: str = "qwen", source_variant: str = "base", overwrite: bool = False, **_: Any
    ) -> ToolResult:
        from forgewright.skills.quantize import write_quant_config

        if not self.runner.available():
            return ToolResult(False, f"model-forge not found at {self.runner.repo}")
        try:
            cfg = write_quant_config(
                self.runner.repo, family, arch=arch, source_variant=source_variant, overwrite=overwrite
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, f"scaffold failed: {e}")
        return ToolResult(True, f"quant config at {cfg}", {"path": str(cfg)})
