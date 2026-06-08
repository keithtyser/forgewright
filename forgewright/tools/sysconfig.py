"""Environment configuration tool: install/upgrade packages, drivers, and dependencies.

This is how the agent repairs or sets up the environment so work can proceed (e.g. an old
``transformers`` that cannot load a model architecture). It is SYSTEM-MODIFYING, so it carries
the ``system`` risk and is approval-gated by default: the human sees the exact command and
where it runs before anything changes. The agent should prefer the least-invasive scoped change
(a project venv or a user install) over global/system mutations, and retry the blocked step
after a successful fix.
"""
from __future__ import annotations

import subprocess
from typing import Any, Optional

from forgewright.tools.base import Tool, ToolResult
from forgewright.tools.forge import ForgeRunner


class EnvConfigTool(Tool):
    name = "configure_env"
    description = (
        "Configure or repair the environment so a blocked step can proceed: install/upgrade "
        "packages (pip/uv/conda/apt), CUDA or driver components, docker, or other system "
        "dependencies. SYSTEM-MODIFYING and APPROVAL-GATED -- propose the exact command(s) and "
        "get approval first. Prefer the least-invasive scoped change (a project venv or a user "
        "install) over global/system changes. Runs on the LOCAL machine, a remote 'host' (ssh), "
        "or inside the model-forge container (in_container=true; note: container changes do not "
        "persist unless the image is rebuilt -- use the host/venv for durable installs). Returns "
        "stdout/stderr + exit code; after a successful fix, retry the step that was blocked."
    )
    risk = "system"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The setup/maintenance command to run."},
            "host": {"type": "string", "description": "Optional ssh target 'user@host' (else local)."},
            "in_container": {
                "type": "boolean",
                "description": "Run inside the model-forge GPU container (default false).",
            },
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 1800)."},
            "cwd": {"type": "string", "description": "Working directory (optional)."},
        },
        "required": ["command"],
    }

    def __init__(self, runner: Optional[ForgeRunner] = None) -> None:
        self.forge = runner or ForgeRunner()

    def run(
        self,
        command: str,
        host: str | None = None,
        in_container: bool = False,
        timeout: int = 1800,
        cwd: str | None = None,
        **_: Any,
    ) -> ToolResult:
        cmd = command
        if in_container:
            # wrap with the canonical container runner; default cwd to the model-forge repo
            cmd = f"bash scripts/run_in_container.sh {command}"
            cwd = cwd or str(self.forge.repo)
        try:
            if host:
                remote = f"cd {cwd} && {cmd}" if cwd else cmd
                proc = subprocess.run(
                    ["ssh", host, remote], capture_output=True, text=True, timeout=timeout
                )
            else:
                proc = subprocess.run(
                    cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
                )
            body = ((proc.stdout or "") + (proc.stderr or "")).strip()
            where = host or ("container" if in_container else "local")
            return ToolResult(
                ok=proc.returncode == 0,
                output=body or f"(no output; exit {proc.returncode}) [{where}]",
                meta={"exit_code": proc.returncode, "where": where},
            ).truncate()
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"configure_env timed out after {timeout}s "
                              "(for very long installs use launch_job)", {"timeout": True})
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, f"error configuring env: {e}", {"error": True})
