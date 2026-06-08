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

from forgewright.frontend.bridge import (
    StreamApprover,
    event_reporter,
    json_line_emitter,
    metric_tap,
)
from forgewright.permissions import PermissionPolicy

HandleTurn = Callable[[str, Callable[[str, dict], None], PermissionPolicy], None]
# (name, args, reporter, emit) -> None. Handles instant control commands (e.g. /graph, /models)
# that query state and stream events, without running a full agent turn.
HandleCommand = Callable[[str, dict, Callable[[str, dict], None], Callable[[dict], None]], None]


class BackendServer:
    """Runs one turn (or instant command) at a time, emitting events. Frontend-agnostic."""

    def __init__(
        self,
        handle_turn: HandleTurn,
        emit: Callable[[dict], None],
        handle_command: Optional[HandleCommand] = None,
    ) -> None:
        self.handle_turn = handle_turn
        self.handle_command = handle_command
        self.emit = emit
        # metric_tap turns trainer log output into typed `metric` events for the UI
        self.reporter = metric_tap(event_reporter(emit))
        # one policy for the whole session, so "approve all" / "yolo" persist across turns
        self.permissions = PermissionPolicy()

    def run_turn(self, text: str, *, ask_fn: Optional[Callable[[Any, dict], Any]] = None) -> None:
        self.permissions.ask_fn = ask_fn   # rebind this turn's approver onto the session policy
        try:
            self.handle_turn(text, self.reporter, self.permissions)
            self.emit({"type": "done", "ok": True})
        except Exception as e:  # noqa: BLE001 - report the failure, keep the session alive
            self.emit({"type": "done", "ok": False, "error": str(e)})

    def run_command(self, msg: dict) -> None:
        name, args = msg.get("name", ""), msg.get("args", {}) or {}
        try:
            if self.handle_command:
                self.handle_command(name, args, self.reporter, self.emit)
            else:
                self.emit({"type": "assistant", "role": "agent", "content": f"unknown command: {name}"})
            self.emit({"type": "done", "ok": True})
        except Exception as e:  # noqa: BLE001
            self.emit({"type": "done", "ok": False, "error": str(e)})


def route(msg: dict, msg_q: "queue.Queue", approval_q: "queue.Queue") -> None:
    """Send a parsed client message to the turn queue or the approval queue."""
    t = msg.get("type")
    if t == "approval_response":
        approval_q.put(msg)
    elif t in ("user_msg", "command", "shutdown"):
        msg_q.put(None if t == "shutdown" else msg)


def _make_recorder(record_path):
    """Append a complete, replayable session transcript: every event we emit to the UI AND
    every message we receive from it, each tagged with a wall-clock time and a direction. This
    is the full trace (review + future training data); the agent ledger is the loop-only view.
    Best-effort: a recording failure never disturbs the session. Returns (record, close)."""
    if not record_path:
        return (lambda direction, obj: None), (lambda: None)
    import os
    import time

    try:
        os.makedirs(os.path.dirname(str(record_path)), exist_ok=True)
        fh = open(str(record_path), "a", encoding="utf-8")  # noqa: SIM115 - closed by caller
    except OSError:
        return (lambda direction, obj: None), (lambda: None)

    def record(direction: str, obj) -> None:
        try:
            fh.write(json.dumps({"t": time.time(), "dir": direction, "event": obj}, default=str) + "\n")
            fh.flush()
        except Exception:  # noqa: BLE001 - recording is best-effort
            pass

    def close() -> None:
        try:
            fh.close()
        except Exception:  # noqa: BLE001
            pass

    return record, close


def serve_stdio(
    handle_turn: HandleTurn, *, instream, outstream, handle_command: Optional[HandleCommand] = None,
    record_path=None, session_meta: Optional[dict] = None,
) -> None:
    """Blocking serve loop over two text streams (stdin/stdout in production). A reader
    thread routes incoming lines; the main loop runs turns with a stream-backed approver,
    and dispatches instant `command` messages to ``handle_command``. If ``record_path`` is
    given, the entire bidirectional event stream is recorded there as a session transcript."""
    record, close_record = _make_recorder(record_path)
    if session_meta:
        record("meta", session_meta)

    def _write(s: str) -> None:
        outstream.write(s)
        try:
            outstream.flush()
        except Exception:  # noqa: BLE001
            pass

    raw_emit = json_line_emitter(_write)

    def emit(obj: dict) -> None:
        record("out", obj)
        raw_emit(obj)

    server = BackendServer(handle_turn, emit, handle_command=handle_command)
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
            record("in", msg)
            route(msg, msg_q, approval_q)
        msg_q.put(None)  # stream closed -> end the session

    threading.Thread(target=reader, daemon=True).start()
    emit({"type": "ready"})
    while True:
        msg = msg_q.get()
        if msg is None:
            break
        if msg.get("type") == "command":
            server.run_command(msg)
            continue
        approver = StreamApprover(emit, approval_q.get)
        server.run_turn(str(msg.get("text", "")), ask_fn=approver)
    emit({"type": "bye"})
    close_record()
