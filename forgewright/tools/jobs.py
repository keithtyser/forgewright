"""Long-horizon detached job manager (local or remote-over-SSH).

This is the piece that separates a post-training harness from a coding agent:
training runs for hours/days. We launch detached, persist a record, and let the
agent poll across its own context resets. Exit codes are captured via a sentinel
``.code`` file so completion is detectable even after the process is long gone.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from forgewright.tools.base import Tool, ToolResult


def _jobs_dir() -> Path:
    home = Path(os.environ.get("FORGEWRIGHT_HOME", str(Path.home() / ".forgewright")))
    d = home / "runs" / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _new_id() -> str:
    return time.strftime("job-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


def _pid_alive_local(pid: Any) -> bool:
    if not pid:
        return False
    pid = int(pid)
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
        )
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _process_identity(pid: int, host: str | None = None) -> str | None:
    """Return the process creation time, so a reused PID cannot own an old job."""
    try:
        if host or os.name != "nt":
            if host:
                result = subprocess.run(ssh_args(host) + [f"cat /proc/{pid}/stat"],
                                        capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    return None
                stat = result.stdout
            else:
                stat = Path(f"/proc/{pid}/stat").read_text()
            return stat.rsplit(")", 1)[1].split()[19]
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
                return None
            return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    except (OSError, IndexError, subprocess.SubprocessError):
        return None


class JobManager:
    """Shared backend for the job tools. Records persist as JSON under the jobs dir."""

    def __init__(self) -> None:
        self.dir = _jobs_dir()
        self.interrupt = None

    def _rec(self, jid: str) -> Path:
        return self.dir / f"{jid}.json"

    def _log(self, jid: str) -> Path:
        return self.dir / f"{jid}.log"

    def _code(self, jid: str) -> Path:
        return self.dir / f"{jid}.code"

    def _save(self, rec: dict) -> None:
        dst = self._rec(rec["id"])
        tmp = dst.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        tmp.replace(dst)

    def _read(self, jid: str) -> Optional[dict]:
        p = self._rec(jid)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def launch(
        self,
        command: str,
        host: str | None = None,
        cwd: str | None = None,
        name: str | None = None,
    ) -> dict:
        jid = _new_id()
        rec: dict[str, Any] = {
            "id": jid,
            "name": name or "",
            "command": command,
            "host": host or "local",
            "status": "running",
            "pid": None,
            "exit_code": None,
            "started_at": time.time(),
            "finished_at": None,
        }
        rec["status"] = "launching"
        self._save(rec)
        rec["status"] = "running"
        if host:
            workdir = cwd or "~/forgewright-runs"
            rlog, rcode = f"{workdir}/{jid}.log", f"{workdir}/{jid}.code"
            inner = shlex.quote(f"( {command} ); echo $? > {shell_path(rcode)}")
            wrapped = (f"mkdir -p {shell_path(workdir)} && cd {shell_path(workdir)} && "
                       f"{{ nohup setsid bash -c {inner} > {shell_path(rlog)} 2>&1 < /dev/null & echo $!; }}")
            proc = subprocess.run(ssh_args(host) + [wrapped], capture_output=True, text=True, timeout=60)
            pid = proc.stdout.strip().splitlines()[-1] if (proc.returncode == 0 and proc.stdout.strip()) else None
            rec.update(pid=pid, remote_log=rlog, remote_code=rcode, workdir=workdir)
            if proc.returncode != 0 or not str(pid).isdigit():
                rec.update(status="error", launch_error=(proc.stderr or "ssh launch failed").strip())
        else:
            log, code = self._log(jid), self._code(jid)
            wrapped = f"( {command} ) > {shlex.quote(str(log))} 2>&1 ; echo $? > {shlex.quote(str(code))}"
            kwargs: dict[str, Any] = {"creationflags": 0}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            else:
                kwargs["start_new_session"] = True
            try:
                proc2 = subprocess.Popen(["bash", "-lc", wrapped], cwd=cwd, **kwargs)
                rec.update(pid=proc2.pid)
            except FileNotFoundError:
                rec.update(status="error", launch_error="bash not found on PATH for local detached job")
        if rec["status"] == "running" and rec.get("pid"):
            rec["process_identity"] = _process_identity(int(rec["pid"]), host)
        self._save(rec)
        return rec

    def status(self, jid: str) -> Optional[dict]:
        rec = self._read(jid)
        if not rec or rec["status"] in ("finished", "ended", "killed", "error"):
            return rec
        code: Optional[int] = None
        if rec["host"] == "local":
            cp = self._code(jid)
            if cp.exists():
                try:
                    code = int(cp.read_text().strip())
                except ValueError:
                    code = None
            alive = _pid_alive_local(rec.get("pid"))
        else:
            r = subprocess.run(
                ssh_args(rec["host"]) + [f"cat {shell_path(rec['remote_code'])} 2>/dev/null"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0 and r.stdout.strip():
                try:
                    code = int(r.stdout.strip())
                except ValueError:
                    code = None
            if code is None:
                check = subprocess.run(ssh_args(rec["host"]) + [f"kill -0 {int(rec['pid'])} 2>/dev/null"],
                                       capture_output=True, text=True, timeout=30)
                if check.returncode not in (0, 1):
                    rec["status_error"] = "SSH status unavailable"
                    self._save(rec)
                    return rec
                alive = check.returncode == 0
            else:
                alive = False
        if code is not None:
            rec.update(status="finished", exit_code=code, finished_at=time.time())
        elif not alive:
            rec.update(status="ended", finished_at=time.time())
        self._save(rec)
        return rec

    def wait(self, jid: str, *, timeout_s: int = 14400, interval_s: int = 20,
             interrupt=None) -> Optional[dict]:
        """Block until the job reaches a terminal state (or timeout). Returns the final
        record. Used by synchronous specialist stages that must complete before handing
        off the produced artifact."""
        deadline = time.time() + timeout_s
        interrupt = interrupt or self.interrupt
        while time.time() < deadline:
            if interrupt and interrupt():
                self.kill(jid)
                return self.status(jid)
            rec = self.status(jid)
            if not rec or rec["status"] in ("finished", "ended", "killed", "error"):
                return rec
            for _ in range(max(1, interval_s)):
                if interrupt and interrupt():
                    break
                time.sleep(1)
        rec = self._read(jid)
        if rec:
            rec["wait_timed_out"] = True
            self._save(rec)
        self.kill(jid)
        return self.status(jid)

    def tail(self, jid: str, n: int = 80) -> str:
        rec = self._read(jid)
        if not rec:
            return ""
        if rec["host"] == "local":
            lp = self._log(jid)
            if not lp.exists():
                return "(no log yet)"
            return "\n".join(lp.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        r = subprocess.run(
            ssh_args(rec["host"]) + [f"tail -n {int(n)} {shell_path(rec['remote_log'])} 2>/dev/null"],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout or "(no log yet)"

    def kill(self, jid: str) -> bool:
        rec = self._read(jid)
        if not rec or not rec.get("pid") or rec["status"] in ("finished", "ended", "killed", "error"):
            return False
        host = None if rec["host"] == "local" else rec["host"]
        identity = _process_identity(int(rec["pid"]), host)
        if not identity or identity != rec.get("process_identity"):
            rec["cancel_error"] = "Process identity is missing or changed; cancellation refused"
            self._save(rec)
            return False
        try:
            if rec["host"] == "local":
                if os.name == "nt":
                    result = subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(rec["pid"]))], capture_output=True)
                    if result.returncode != 0:
                        return False
                else:
                    getattr(os, "killpg")(int(rec["pid"]), 15)
            else:
                result = subprocess.run(ssh_args(rec["host"]) + [f"kill -- -{int(rec['pid'])}"],
                                        capture_output=True, timeout=30)
                if result.returncode != 0:
                    return False
        except Exception:  # noqa: BLE001
            return False
        rec.update(status="stopping", cancel_requested_at=time.time())
        self._save(rec)
        # Sending a signal is not proof that the process has stopped.
        for _ in range(10):
            current = _process_identity(int(rec["pid"]), host)
            if current is None and host:
                break  # A lost SSH connection is not proof of termination.
            if current != identity:
                rec.update(status="killed", finished_at=time.time())
                self._save(rec)
                return True
            time.sleep(0.1)
        return False

    def list(self) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.dir.glob("*.json"))]


def shell_path(path: str) -> str:
    """Quote a POSIX path while allowing only a leading home-directory expansion."""
    return '"$HOME"/' + shlex.quote(path[2:]) if path.startswith("~/") else shlex.quote(path)


def ssh_args(host: str) -> list[str]:
    if not host or host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("Invalid SSH host")
    if ":" in host:
        name, port = host.rsplit(":", 1)
        return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-p", str(int(port)), name]
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", host]


# --- Tools (share one JobManager instance, wired by the CLI) ---------------------

class LaunchJobTool(Tool):
    name = "launch_job"
    description = (
        "Launch a long-running command as a DETACHED job (local, or on a remote host via SSH). "
        "Returns a job id immediately; the job keeps running after this call. Use for training/"
        "quantization runs, then poll with monitor_job and tail_logs."
    )
    risk = "exec"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "host": {"type": "string", "description": "Optional user@host to run remotely."},
            "cwd": {"type": "string", "description": "Working directory (remote: defaults to ~/forgewright-runs)."},
            "name": {"type": "string", "description": "Optional human label."},
        },
        "required": ["command"],
    }

    def __init__(self, jm: JobManager) -> None:
        self.jm = jm

    def run(self, command: str, host: str | None = None, cwd: str | None = None, name: str | None = None, **_: Any) -> ToolResult:
        rec = self.jm.launch(command, host=host, cwd=cwd, name=name)
        ok = rec["status"] != "error"
        msg = f"launched {rec['id']} (pid={rec.get('pid')}, host={rec['host']})"
        if rec.get("launch_error"):
            msg += f"\nerror: {rec['launch_error']}"
        return ToolResult(ok, msg, rec)


class MonitorJobTool(Tool):
    name = "monitor_job"
    description = "Check a detached job's status (running/finished/ended/killed/error) and exit code."
    risk = "read"
    parameters = {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}

    def __init__(self, jm: JobManager) -> None:
        self.jm = jm

    def run(self, job_id: str, **_: Any) -> ToolResult:
        rec = self.jm.status(job_id)
        if not rec:
            return ToolResult(False, f"no such job: {job_id}")
        return ToolResult(True, f"{rec['id']}: {rec['status']} (exit={rec['exit_code']})", rec)


class TailLogsTool(Tool):
    name = "tail_logs"
    description = "Return the last N lines of a job's log (local or remote)."
    risk = "read"
    parameters = {
        "type": "object",
        "properties": {"job_id": {"type": "string"}, "lines": {"type": "integer", "description": "default 80"}},
        "required": ["job_id"],
    }

    def __init__(self, jm: JobManager) -> None:
        self.jm = jm

    def run(self, job_id: str, lines: int = 80, **_: Any) -> ToolResult:
        out = self.jm.tail(job_id, n=lines)
        return ToolResult(bool(out), out or f"no log for {job_id}", {"job_id": job_id}).truncate()


class KillJobTool(Tool):
    name = "kill_job"
    description = "Terminate a running detached job by id."
    risk = "destructive"
    parameters = {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}

    def __init__(self, jm: JobManager) -> None:
        self.jm = jm

    def run(self, job_id: str, **_: Any) -> ToolResult:
        ok = self.jm.kill(job_id)
        return ToolResult(ok, f"killed {job_id}" if ok else f"could not kill {job_id}", {"job_id": job_id})


class ListJobsTool(Tool):
    name = "list_jobs"
    description = "List all known jobs and their statuses."
    risk = "read"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, jm: JobManager) -> None:
        self.jm = jm

    def run(self, **_: Any) -> ToolResult:
        jobs = self.jm.list()
        if not jobs:
            return ToolResult(True, "(no jobs)", {"count": 0})
        lines = [f"- {j['id']} [{j['status']}] host={j['host']} :: {j['command'][:60]}" for j in jobs]
        return ToolResult(True, "\n".join(lines), {"count": len(jobs)})
