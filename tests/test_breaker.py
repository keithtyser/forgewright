"""Tests for the velocity circuit breaker + its wiring into the agent loop."""
from __future__ import annotations

from forgewright.breaker import CircuitBreaker
from forgewright.brain.provider import AssistantTurn, ToolCall
from forgewright.loop import Agent
from forgewright.tools.base import Tool, ToolRegistry, ToolResult


class _Echo(Tool):
    name = "echo_tool"
    description = "echo"
    risk = "read"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def run(self, text: str = "", **_):
        return ToolResult(True, text)


class _Fail(Tool):
    name = "fail_tool"
    description = "always fails"
    risk = "read"
    parameters = {"type": "object", "properties": {"n": {"type": "string"}}}

    def run(self, **_):
        return ToolResult(False, "nope")


class _FakeBrain:
    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools=None, tool_choice="auto"):
        return self.turns.pop(0)


def test_breaker_disabled_when_all_limits_zero():
    b = CircuitBreaker(max_idle_steps=0, max_idle_tokens=0, max_idle_seconds=0)
    assert b.enabled is False
    for _ in range(100):
        b.record_step(1000)
    assert b.tripped() is None   # never trips when disabled


def test_breaker_trips_on_idle_steps():
    b = CircuitBreaker(max_idle_steps=3, max_idle_tokens=0, max_idle_seconds=0)
    b.record_step(); assert b.tripped() is None
    b.record_step(); assert b.tripped() is None
    b.record_step(); assert b.tripped() is not None and "no progress" in b.tripped()


def test_progress_resets_idle_window():
    b = CircuitBreaker(max_idle_steps=3)
    b.record_step(); b.record_step()
    b.record_progress()            # value produced -> window resets
    b.record_step(); b.record_step()
    assert b.tripped() is None     # only 2 idle steps since progress


def test_breaker_trips_on_idle_tokens():
    b = CircuitBreaker(max_idle_steps=0, max_idle_tokens=5000)
    b.record_step(2000); b.record_step(2000)
    assert b.tripped() is None
    b.record_step(2000)
    assert b.tripped() is not None


def test_env_override(monkeypatch):
    monkeypatch.setenv("FORGEWRIGHT_BREAKER_IDLE_STEPS", "2")
    b = CircuitBreaker()
    b.record_step(); b.record_step()
    assert b.tripped() is not None


def test_agent_loop_trips_breaker_on_failing_spin():
    # every step calls a failing tool -> no progress -> breaker stops the unbounded run
    turns = [AssistantTurn("", [ToolCall(str(i), "fail_tool", {"n": str(i)})]) for i in range(20)]
    agent = Agent(_FakeBrain(turns), ToolRegistry([_Fail()]), max_steps=0,
                  breaker=CircuitBreaker(max_idle_steps=4))
    res = agent.run("spin")
    assert res.done is False and "circuit breaker" in res.final and res.steps <= 5


def test_agent_loop_progress_keeps_breaker_open():
    # distinct successful tool calls each step = progress each step -> breaker never trips
    turns = [AssistantTurn("", [ToolCall(str(i), "echo_tool", {"text": str(i)})]) for i in range(6)]
    turns.append(AssistantTurn("done"))
    agent = Agent(_FakeBrain(turns), ToolRegistry([_Echo()]), max_steps=0,
                  breaker=CircuitBreaker(max_idle_steps=3))
    res = agent.run("go")
    assert res.done is True and res.final == "done"
