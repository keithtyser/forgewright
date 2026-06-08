"""Small host-side shell helpers shared by specialists that produce on-disk model dirs.

Used to resolve a local checkpoint, and to verify a step actually wrote fresh weights (the
provenance-honesty check). Runs locally (bash) or on a remote host (ssh)."""
from __future__ import annotations

import subprocess
from typing import Optional


def host_run(host: Optional[str], cmd: str, timeout: int = 60) -> str:
    """Run a quick shell command on `host` (ssh) or locally; return stdout (best-effort)."""
    argv = ["ssh", host, cmd] if host else ["bash", "-lc", cmd]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def resolve_local(host: Optional[str], uri: str) -> str:
    """Prefer a local checkpoint over a bare HF id (id -> ~/models/<stem> when the host has it)."""
    if not uri or uri.startswith(("~", "/", ".")):
        return uri
    stem = uri.rstrip("/").split("/")[-1]
    cand = f"~/models/{stem}"
    return cand if host_run(host, f"test -d {cand} && echo yes") else uri


def wrote_fresh_weights(host: Optional[str], out_dir: str, since: float) -> bool:
    """True iff `out_dir` has model weights modified at/after `since` (written by this run)."""
    cmd = (f'find {out_dir} -type f \\( -name "*.safetensors" -o -name "*.bin" \\) '
           f"-newermt @{int(since)} 2>/dev/null | head -1")
    return bool(host_run(host, out_dir and cmd))
