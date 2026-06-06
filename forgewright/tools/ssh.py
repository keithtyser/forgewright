"""Remote shell execution over SSH (paramiko). The agent's reach onto GPU boxes."""
from __future__ import annotations

from typing import Any

from forgewright.tools.base import Tool, ToolResult


class SSHTool(Tool):
    name = "bash_remote"
    description = (
        "Run a shell command on a REMOTE host over SSH (e.g. a GPU box). "
        "Host is 'user@host' or 'user@host:port'. Returns combined stdout/stderr and exit code. "
        "Use to launch/monitor training, run nvidia-smi, docker, etc. on GPU machines."
    )
    risk = "exec"
    parameters = {
        "type": "object",
        "properties": {
            "host": {"type": "string", "description": "Target as user@host or user@host:port."},
            "command": {"type": "string", "description": "Command to run remotely."},
            "timeout": {"type": "integer", "description": "Timeout seconds (default 600)."},
        },
        "required": ["host", "command"],
    }

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}

    def _client(self, host: str) -> Any:
        import paramiko

        if host in self._clients:
            return self._clients[host]
        user: str | None = None
        port = 22
        h = host
        if "@" in h:
            user, h = h.split("@", 1)
        if ":" in h:
            h, p = h.rsplit(":", 1)
            port = int(p)
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=h,
            port=port,
            username=user,
            allow_agent=True,
            look_for_keys=True,
            timeout=20,
        )
        self._clients[host] = client
        return client

    def run(self, host: str, command: str, timeout: int = 600, **_: Any) -> ToolResult:
        try:
            client = self._client(host)
            _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            out = stdout.read().decode("utf-8", "replace")
            err = stderr.read().decode("utf-8", "replace")
            code = stdout.channel.recv_exit_status()
            body = (out + err).strip()
            return ToolResult(
                ok=code == 0,
                output=body or f"(no output; exit {code})",
                meta={"exit_code": code, "host": host},
            ).truncate()
        except Exception as e:  # noqa: BLE001
            self._clients.pop(host, None)  # drop a possibly-broken connection
            return ToolResult(False, f"ssh error on {host}: {e}", {"error": True, "host": host})
