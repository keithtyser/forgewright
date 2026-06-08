"""Serialize backend reporter events + approval requests to the JSON UI stream.

`event_reporter` adapts the `(kind, data)` reporter callback used everywhere (loop, Director,
specialists) into `{"type": kind, ...data}` JSON lines. `StreamApprover` adapts
`PermissionPolicy.ask_fn` into an `approval_request` event + a blocking read of the
`approval_response` — so any specialist's destructive action surfaces at the single UI prompt.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable, Optional

# trainer log keys we surface, normalized to a stable set the UI understands.
_METRIC_KEYS = {
    "loss": "loss", "train_loss": "loss",
    "reward": "reward", "rewards": "reward", "mean_reward": "reward", "reward_mean": "reward",
    "grad_norm": "grad_norm", "grad norm": "grad_norm",
    "kl": "kl", "objective/kl": "kl",
    "learning_rate": "lr", "lr": "lr",
    "epoch": "epoch",
}
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_metrics(text: str) -> dict:
    """Extract structured training metrics from a trainer log line/blob.

    Prefers real structure (HF Trainer prints python-dict literals like
    ``{'loss': 0.31, 'grad_norm': 1.2, 'learning_rate': 9e-05, 'epoch': 0.5}``); falls back to
    ``key: value`` / ``key=value`` pairs and a ``step/total`` (or tqdm ``40/120 [``) count.
    Returns a dict with any of: step, total, loss, reward, grad_norm, kl, lr, epoch."""
    if not text:
        return {}
    out: dict[str, float] = {}
    for m in re.finditer(r"\{[^{}]*\}", text):           # dict literals first (most reliable)
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if isinstance(d, dict):
            for k, v in d.items():
                norm = _METRIC_KEYS.get(str(k).lower())
                if norm and isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.setdefault(norm, float(v))
    for raw, norm in _METRIC_KEYS.items():               # then loose key: value pairs
        if norm in out:
            continue
        mm = re.search(r"(?<![\w/])" + re.escape(raw) + r"[\s:=]+(" + _NUM + ")", text, re.I)
        if mm:
            try:
                out[norm] = float(mm.group(1))
            except ValueError:
                pass
    sm = re.search(r"step\s*[:=]?\s*(\d+)\s*/\s*(\d+)", text, re.I) \
        or re.search(r"(?<!\d)(\d+)\s*/\s*(\d+)\s*[\[\]]", text)   # tqdm "40/120 ["
    if sm:
        out["step"], out["total"] = int(sm.group(1)), int(sm.group(2))
    else:
        sm2 = re.search(r"(?:global_)?step\s*[:=]\s*(\d+)", text, re.I)
        if sm2:
            out["step"] = int(sm2.group(1))
    return out


def metric_tap(report: Callable[[str, dict], None]) -> Callable[[str, dict], None]:
    """Wrap a reporter so tool/progress events that carry trainer output ALSO emit a typed
    ``metric`` event. This is where the swarm's training telemetry becomes structured: the UI
    consumes ``metric`` events directly instead of re-parsing prose."""
    def tapped(kind: str, data: dict) -> None:
        report(kind, data)
        if kind in ("tool", "progress"):
            m = parse_metrics(str(data.get("output") or data.get("text") or ""))
            if m:
                report("metric", {"role": data.get("role", ""), **m})
    return tapped


def json_line_emitter(write: Callable[[str], Any]) -> Callable[[dict], None]:
    """An emitter that writes one compact JSON object per line via `write`
    (e.g. sys.stdout.write, or a socket send)."""
    def emit(obj: dict) -> None:
        write(json.dumps(obj, default=str) + "\n")
    return emit


def event_reporter(emit: Callable[[dict], None]) -> Callable[[str, dict], None]:
    """Reporter callback that ships `(kind, data)` to the UI as `{"type": kind, ...data}`.
    The `role` already injected by `label_reporter` rides along, so the UI can attribute
    each line to the right specialist in the one transcript."""
    def report(kind: str, data: dict) -> None:
        emit({"type": kind, **data})
    return report


class StreamApprover:
    """`PermissionPolicy.ask_fn` over the stream: emit an approval_request, block on the
    matching approval_response. Returns the decision string ('yes'|'all'|'yolo'|'no') so the
    policy can remember 'all' (this tool) or 'yolo' (everything). `read_response` returns the
    next UI->backend message dict (the frontend prompts the human and sends the decision)."""

    def __init__(self, emit: Callable[[dict], None], read_response: Callable[[], Optional[dict]]) -> None:
        self.emit = emit
        self.read_response = read_response

    def __call__(self, tool: Any, args: dict[str, Any]) -> str:
        self.emit({
            "type": "approval_request",
            "tool": getattr(tool, "name", str(tool)),
            "risk": getattr(tool, "risk", ""),
            "args": args,
        })
        while True:
            msg = self.read_response()
            if msg is None:                       # stream closed -> deny (fail safe)
                return "no"
            if msg.get("type") == "approval_response":
                if msg.get("decision"):
                    return str(msg["decision"]).lower()
                return "yes" if msg.get("approved") else "no"
            # ignore unrelated messages until the decision arrives
