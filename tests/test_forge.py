"""Tests for the model-forge CLI driver tools (no real model-forge needed)."""
from __future__ import annotations

from forgewright.tools.forge import ForgePublishTool, ForgeRunner, ForgeTool


def test_forge_runner_missing_repo(tmp_path):
    r = ForgeRunner(repo=tmp_path / "nope")
    assert r.available() is False
    res = r.run("quantize plan")
    assert res.ok is False
    assert "not found" in res.output.lower()


def test_forge_tool_schema():
    t = ForgeTool()
    s = t.schema()
    assert s["function"]["name"] == "forge"
    assert "args" in s["function"]["parameters"]["properties"]
    assert t.risk == "exec"


def test_forge_publish_is_destructive():
    assert ForgePublishTool().risk == "destructive"
