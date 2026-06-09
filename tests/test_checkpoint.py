"""Tests for checkpoint/resume: Checkpoint store, ContextManager snapshot, Agent resume."""
from __future__ import annotations

from forgewright.brain.provider import AssistantTurn, ToolCall
from forgewright.checkpoint import Checkpoint
from forgewright.context.manager import ContextManager
from forgewright.loop import Agent
from forgewright.tools.base import Tool, ToolRegistry, ToolResult


class _Echo(Tool):
    name = "echo_tool"
    description = "echo"
    risk = "read"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def run(self, text: str = "", **_):
        return ToolResult(True, text)


class _FakeBrain:
    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.seen.append(messages)
        return self.turns.pop(0)


def test_checkpoint_roundtrip(tmp_path):
    cp = Checkpoint("run-x", checkpoint_dir=tmp_path)
    assert cp.exists() is False and cp.load() is None
    cp.save({"step": 3, "ctx": {"msgs": [{"role": "user", "content": "hi"}]}})
    assert cp.exists() is True
    state = cp.load()
    assert state["step"] == 3 and state["ctx"]["msgs"][0]["content"] == "hi"
    cp.clear()
    assert cp.exists() is False


def test_corrupt_checkpoint_is_none(tmp_path):
    cp = Checkpoint("run-bad", checkpoint_dir=tmp_path)
    cp.path.write_text("{ not json", encoding="utf-8")
    assert cp.load() is None


def test_context_snapshot_restore():
    cm = ContextManager("system")
    cm.set_pinned("Goal: do the thing")
    cm.add_user("first message")
    cm.add_assistant(AssistantTurn("ack"))
    snap = cm.snapshot()

    fresh = ContextManager("system")
    fresh.restore(snap)
    msgs = fresh.messages()
    assert any("do the thing" in (m.get("content") or "") for m in msgs)
    assert any(m.get("content") == "first message" for m in msgs)
    assert any(m.get("content") == "ack" for m in msgs)


def test_agent_writes_checkpoint_each_step(tmp_path):
    cp = Checkpoint("run-ck", checkpoint_dir=tmp_path)
    turns = [AssistantTurn("", [ToolCall("1", "echo_tool", {"text": "a"})]),
             AssistantTurn("", [ToolCall("2", "echo_tool", {"text": "b"})]),
             AssistantTurn("done")]
    Agent(_FakeBrain(turns), ToolRegistry([_Echo()]), max_steps=0, checkpoint=cp).run("go")
    # a clean finish clears the checkpoint (nothing to resume)
    assert cp.exists() is False


def test_agent_resumes_from_checkpoint(tmp_path):
    cp = Checkpoint("run-rs", checkpoint_dir=tmp_path)
    # pre-seed a checkpoint as if a prior run got two steps in
    cm = ContextManager("system")
    cm.set_pinned("Goal: original goal")
    cm.add_user("original goal")
    cm.add_assistant(AssistantTurn("did step 1"))
    cp.save({"step": 2, "recent": ["sig"], "ctx": cm.snapshot()})

    brain = _FakeBrain([AssistantTurn("resumed and done")])
    agent = Agent(brain, ToolRegistry([_Echo()]), max_steps=0, checkpoint=cp, resume=True)
    res = agent.run("this goal arg should be ignored on resume")
    assert res.done is True and res.final == "resumed and done"
    # step continued from 2 -> 3 (didn't restart at 1)
    assert res.steps == 3
    # the resumed context carried the prior turns, not the new goal arg
    sent = brain.seen[0]
    assert any("original goal" in (m.get("content") or "") for m in sent)
    assert not any("should be ignored" in (m.get("content") or "") for m in sent)
