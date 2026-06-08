"""Tests for the environment-configuration tool + its approval-gated `system` risk tier."""
from __future__ import annotations

from forgewright.permissions import PermissionPolicy
from forgewright.tools.forge import ForgeRunner
from forgewright.tools.sysconfig import EnvConfigTool


def test_configure_env_runs_local_command(tmp_path):
    tool = EnvConfigTool(ForgeRunner(repo=tmp_path))
    res = tool.run(command="echo installing transformers")
    assert res.ok and "installing transformers" in res.output
    assert res.meta["where"] == "local" and res.meta["exit_code"] == 0


def test_configure_env_reports_nonzero_exit(tmp_path):
    tool = EnvConfigTool(ForgeRunner(repo=tmp_path))
    res = tool.run(command="exit 7")
    assert not res.ok and res.meta["exit_code"] == 7


def test_configure_env_is_system_risk_and_approval_gated(tmp_path):
    tool = EnvConfigTool(ForgeRunner(repo=tmp_path))
    assert tool.risk == "system"
    # default policy: no approver -> a system change is denied (not silently run)
    assert PermissionPolicy().check(tool, {"command": "pip install -U transformers"}).allowed is False
    # with an approver that says yes -> allowed
    approved = PermissionPolicy(ask_fn=lambda t, a: "yes")
    assert approved.check(tool, {"command": "pip install -U transformers"}).allowed is True
    # "approve all" remembers configure_env for the session
    allp = PermissionPolicy(ask_fn=lambda t, a: "all")
    assert allp.check(tool, {"command": "apt-get install -y x"}).allowed is True
    assert allp.check(tool, {"command": "pip install y"}).allowed is True  # remembered, no re-ask


def test_configure_env_in_registry():
    from forgewright.cli import build_registry

    reg = build_registry()
    assert "configure_env" in reg.names()
    assert reg.get("configure_env").risk == "system"
