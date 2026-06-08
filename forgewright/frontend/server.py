"""Backend server: drive the swarm over the newline-JSON event stream.

This is the backend half of the terminal-kit contract. A frontend (the terminal-kit Node
TUI, or any client) sends `user_msg` / `approval_response` lines on stdin; the server runs
a turn (a Director recipe or an agent), streams `assistant` / `tool` / `progress` /
`approval_request` events back on stdout, and ends each turn with a `done` event.

The turn logic is injected as `handle_turn(text, reporter, permissions)`, so the swarm
wiring stays separate from the I/O + concurrency (and is trivially testable). Mid-turn
approvals work because a reader thread routes `approval_response` into a queue that the
`StreamApprover` (wired as `PermissionPolicy.ask_fn`) blocks on.
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Any, Callable, Optional

from forgewright.frontend.bridge import StreamApprover, event_reporter, json_line_emitter
from forgewright.permissions import PermissionPolicy

HandleTurn = Callable[[str, Callable[[str, dict], None], PermissionPolicy], None]


class BackendServer:
    """Runs one turn at a time, emitting events. Frontend-agnostic."""

    def __init__(self, handle_turn: HandleTurn, emit: Callable[[dict], None]) -> None:
        self.handle_turn = handle_turn
        self.emit = emit
        self.reporter = event_reporter(emit)
        # one policy for the whole session, so "approve all" / "yolo" persist across turns
        self.permissions = PermissionPolicy()

    def run_turn(self, text: str, *, ask_fn: Optional[Callable[[Any, dict], Any]] = None) -> None:
        self.permissions.ask_fn = ask_fn   # rebind this turn's approver onto the session policy
        try:
            self.handle_turn(text, self.reporter, self.permissions)
            self.emit({"type": "done", "ok": True})
        except Exception as e:  # noqa: BLE001 - report the failure, keep the session alive
            self.emit({"type": "done", "ok": False, "error": str(e)})


def route(msg: dict, msg_q: "queue.Queue", approval_q: "queue.Queue") -> None:
    """Send a parsed client message to the turn queue or the approval queue."""
    t = msg.get("type")
    if t == "approval_response":
        approval_q.put(msg)
    elif t in ("user_msg", "shutdown"):
        msg_q.put(None if t == "shutdown" else msg)


def serve_stdio(handle_turn: HandleTurn, *, instream, outstream) -> None:
    """Blocking serve loop over two text streams (stdin/stdout in production). A reader
    thread routes incoming lines; the main loop runs turns with a stream-backed approver."""
    def _write(s: str) -> None:
        outstream.write(s)
        try:
            outstream.flush()
        except Exception:  # noqa: BLE001
            pass

    emit = json_line_emitter(_write)
    server = BackendServer(handle_turn, emit)
    msg_q: "queue.Queue" = queue.Queue()
    approval_q: "queue.Queue" = queue.Queue()

    def reader() -> None:
        for line in instream:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            route(msg, msg_q, approval_q)
        msg_q.put(None)  # stream closed -> end the session

    threading.Thread(target=reader, daemon=True).start()
    emit({"type": "ready"})
    while True:
        msg = msg_q.get()
        if msg is None:
            break
        approver = StreamApprover(emit, approval_q.get)
        server.run_turn(str(msg.get("text", "")), ask_fn=approver)
    emit({"type": "bye"})
