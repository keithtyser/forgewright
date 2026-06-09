"""Local shell execution tool."""
from __future__ import annotations

import os
import subprocess
from typing import Any

from forgewright.tools.base import Tool, ToolResult


class ShellTool(Tool):
    name = "bash"
    description = (
        "Run a shell command on the LOCAL machine. Returns combined stdout/stderr "
        "and the exit code. Use for inspection, running scripts, git, docker, etc."
    )
    risk = "exec"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run."},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 600)."},
            "cwd": {"type": "string", "description": "Working directory (optional)."},
        },
        "required": ["command"],
    }

    def run(self, command: str, timeout: int = 600, cwd: str | None = None, **_: Any) -> ToolResult:
        if not command or not command.strip():
            return ToolResult(False, "empty command (nothing to run)", {"error": True})
        # an empty-string cwd makes subprocess chdir to '' and crash with ENOENT; treat it as unset,
        # and expand a leading ~ (shell=True does not expand ~ in the cwd argument).
        cwd = os.path.expanduser(cwd) if (cwd and cwd.strip()) else None
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            body = ((proc.stdout or "") + (proc.stderr or "")).strip()
            return ToolResult(
                ok=proc.returncode == 0,
                output=body or f"(no output; exit {proc.returncode})",
                meta={"exit_code": proc.returncode},
            ).truncate()
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"command timed out after {timeout}s", {"timeout": True})
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, f"error running command: {e}", {"error": True})
