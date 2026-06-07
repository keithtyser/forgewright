"""Tests for the backend serve-loop (turn emit, approval over the stream, routing, stdio)."""
from __future__ import annotations

import io
import json
import queue

from forgewright.frontend.server import BackendServer, route, serve_stdio


class _DestructiveTool:
    name = "forge_publish"
    risk = "destructive"


def test_run_turn_emits_events_and_done():
    events = []
    def handle(text, reporter, permissions):
        reporter("assistant", {"content": f"working on {text}"})
    BackendServer(handle, events.append).run_turn("a goal")
    assert events[0] == {"type": "assistant", "content": "working on a goal"}
    assert events[-1] == {"type": "done", "ok": True}


def test_run_turn_reports_failure():
    events = []
    def handle(text, reporter, permissions):
        raise RuntimeError("boom")
    BackendServer(handle, events.append).run_turn("x")
    assert events[-1]["type"] == "done" and not events[-1]["ok"] and events[-1]["error"] == "boom"


def test_run_turn_approval_uses_ask_fn():
    events = []
    decisions = []
    def handle(text, reporter, permissions):
        d = permissions.check(_DestructiveTool(), {"args": "publish-model"})
        decisions.append(d.allowed)
    BackendServer(handle, events.append).run_turn("publish", ask_fn=lambda tool, args: True)
    assert decisions == [True]  # destructive -> ask -> ask_fn approved


def test_route_sends_to_right_queue():
    mq, aq = queue.Queue(), queue.Queue()
    route({"type": "user_msg", "text": "hi"}, mq, aq)
    route({"type": "approval_response", "approved": True}, mq, aq)
    route({"type": "shutdown"}, mq, aq)
    assert mq.get()["text"] == "hi"
    assert mq.get() is None            # shutdown -> sentinel
    assert aq.get()["approved"] is True


def test_serve_stdio_streams_turn_and_approval():
    # one turn that needs a destructive approval; the response is already in the stream
    instream = io.StringIO(
        json.dumps({"type": "user_msg", "text": "publish it"}) + "\n"
        + json.dumps({"type": "approval_response", "approved": True}) + "\n"
    )
    outstream = io.StringIO()

    def handle(text, reporter, permissions):
        reporter("assistant", {"content": "about to publish"})
        d = permissions.check(_DestructiveTool(), {"args": "publish-model"})
        reporter("tool", {"tool": "forge_publish", "ok": d.allowed})

    serve_stdio(handle, instream=instream, outstream=outstream)
    types = [json.loads(l)["type"] for l in outstream.getvalue().splitlines() if l.strip()]
    assert types[0] == "ready"
    assert "approval_request" in types       # the gate surfaced to the UI
    assert "done" in types and types[-1] == "bye"
    # the approval was granted -> the publish tool event reports ok
    objs = [json.loads(l) for l in outstream.getvalue().splitlines() if l.strip()]
    pub = [o for o in objs if o.get("type") == "tool" and o.get("tool") == "forge_publish"][0]
    assert pub["ok"] is True
