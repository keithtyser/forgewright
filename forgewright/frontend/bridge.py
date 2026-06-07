"""Serialize backend reporter events + approval requests to the JSON UI stream.

`event_reporter` adapts the `(kind, data)` reporter callback used everywhere (loop, Director,
specialists) into `{"type": kind, ...data}` JSON lines. `StreamApprover` adapts
`PermissionPolicy.ask_fn` into an `approval_request` event + a blocking read of the
`approval_response` — so any specialist's destructive action surfaces at the single UI prompt.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional


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
    matching approval_response. `read_response` returns the next UI->backend message dict
    (the frontend is responsible for prompting the human and sending the decision)."""

    def __init__(self, emit: Callable[[dict], None], read_response: Callable[[], Optional[dict]]) -> None:
        self.emit = emit
        self.read_response = read_response

    def __call__(self, tool: Any, args: dict[str, Any]) -> bool:
        self.emit({
            "type": "approval_request",
            "tool": getattr(tool, "name", str(tool)),
            "risk": getattr(tool, "risk", ""),
            "args": args,
        })
        while True:
            msg = self.read_response()
            if msg is None:                       # stream closed -> deny (fail safe)
                return False
            if msg.get("type") == "approval_response":
                return bool(msg.get("approved"))
            # ignore unrelated messages until the decision arrives
