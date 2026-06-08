"""Slice-0 smoke tests: config parsing, tool schemas, permissions, and the agent loop
(driven by a fake brain so no network/LLM is needed)."""
from __future__ import annotations

from forgewright.brain.provider import AssistantTurn, ToolCall
from forgewright.config import parse_brain_arg, parse_hardware_arg
from forgewright.loop import Agent
from forgewright.permissions import PermissionPolicy
from forgewright.tools.base import Tool, ToolRegistry, ToolResult


def test_parse_brain_arg():
    p = parse_brain_arg("vllm:qwen3.5@http://localhost:8000/v1")
    assert p.kind == "vllm"
    assert p.model == "qwen3.5"
    assert p.api_base == "http://localhost:8000/v1"
    assert p.litellm_model() == "hosted_vllm/qwen3.5"
    assert parse_brain_arg("anthropic:claude-opus-4-8").litellm_model() == "anthropic/claude-opus-4-8"
    r = parse_brain_arg("openrouter:deepseek/deepseek-v4-pro")
    assert r.kind == "openrouter"
    assert r.litellm_model() == "openrouter/deepseek/deepseek-v4-pro"


def test_parse_hardware_arg():
    ts = parse_hardware_arg("local,ssh://keith@gpu1/home/keith/runs")
    assert ts[0].kind == "local"
    assert ts[1].kind == "ssh"
    assert ts[1].host == "keith@gpu1"
    assert ts[1].workdir == "/home/keith/runs"


class EchoTool(Tool):
    name = "echo_tool"
    description = "echo text back"
    risk = "read"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def run(self, text, **_):
        return ToolResult(True, f"echo: {text}")


class DangerTool(Tool):
    name = "danger"
    description = "destructive op"
    risk = "destructive"
    parameters = {"type": "object", "properties": {}}

    def run(self, **_):
        return ToolResult(True, "boom")


class FakeBrain:
    """Returns scripted AssistantTurns; satisfies the duck-typed Brain.chat() interface."""

    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools=None, tool_choice="auto"):
        return self.turns.pop(0)


def test_tool_schema():
    s = ToolRegistry([EchoTool()]).schemas()[0]
    assert s["type"] == "function"
    assert s["function"]["name"] == "echo_tool"
    assert "text" in s["function"]["parameters"]["properties"]


def test_permission_policy():
    deny = PermissionPolicy()  # destructive -> ask, no approver -> deny
    assert deny.check(DangerTool(), {}).allowed is False
    auto = PermissionPolicy(auto_approve=True)
    assert auto.check(DangerTool(), {}).allowed is True
    assert auto.check(EchoTool(), {}).allowed is True  # read -> allow


def test_agent_loop_with_fake_brain():
    turns = [
        AssistantTurn("", [ToolCall("1", "echo_tool", {"text": "hi"})]),
        AssistantTurn("all done"),
    ]
    res = Agent(FakeBrain(turns), ToolRegistry([EchoTool()]), max_steps=5).run("do the thing")
    assert res.done is True
    assert res.steps == 2
    assert res.final == "all done"


def test_unbounded_steps_runs_until_goal_met():
    # max_steps=0 means no step-budget hard stop; the loop runs until a turn has no tool calls
    turns = [AssistantTurn("", [ToolCall(str(i), "echo_tool", {"text": str(i)})]) for i in range(6)]
    turns.append(AssistantTurn("done"))
    res = Agent(FakeBrain(turns), ToolRegistry([EchoTool()]), max_steps=0).run("go")
    assert res.done is True and res.final == "done" and res.steps == 7


def test_doom_loop_guard():
    same = lambda: AssistantTurn("", [ToolCall("x", "echo_tool", {"text": "loop"})])
    turns = [same(), same(), same(), same(), AssistantTurn("stopped")]
    res = Agent(FakeBrain(turns), ToolRegistry([EchoTool()]), max_steps=10).run("loop")
    assert res.done is True
    assert res.final == "stopped"
