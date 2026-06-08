"""Tests for the permission policy: exec asks; yes/all/yolo/no decisions; session memory."""
from __future__ import annotations

from forgewright.permissions import PermissionPolicy


class _Tool:
    def __init__(self, name, risk):
        self.name, self.risk = name, risk


BASH = _Tool("bash", "exec")
BASH2 = _Tool("bash", "exec")
FORGE = _Tool("forge", "exec")
READ = _Tool("read_file", "read")
PUBLISH = _Tool("forge_publish", "destructive")


def test_read_allowed_exec_asks_by_default():
    p = PermissionPolicy(ask_fn=lambda t, a: "no")
    assert p.check(READ, {}).allowed                 # read -> allow
    assert not p.check(BASH, {}).allowed             # exec -> ask -> denied here
    assert not p.check(PUBLISH, {}).allowed          # destructive -> ask


def test_yes_allows_once_only():
    calls = []
    p = PermissionPolicy(ask_fn=lambda t, a: (calls.append(t.name), "yes")[1])
    assert p.check(BASH, {}).allowed
    assert p.check(BASH, {}).allowed
    assert len(calls) == 2                           # asked each time (no memory)


def test_all_remembers_tool_for_session():
    calls = []
    p = PermissionPolicy(ask_fn=lambda t, a: (calls.append(t.name), "all")[1])
    assert p.check(BASH, {}).allowed                 # asks once, remembers "bash"
    assert p.check(BASH2, {}).allowed                # same tool name -> no re-ask
    assert len(calls) == 1
    # a different tool still asks
    p.ask_fn = lambda t, a: "no"
    assert not p.check(FORGE, {}).allowed


def test_yolo_bypasses_everything():
    calls = []
    p = PermissionPolicy(ask_fn=lambda t, a: (calls.append(1), "yolo")[1])
    assert p.check(BASH, {}).allowed                 # asks once, sets yolo
    assert p.check(FORGE, {}).allowed                # everything allowed now
    assert p.check(PUBLISH, {}).allowed
    assert len(calls) == 1


def test_bool_ask_fn_backward_compatible():
    assert PermissionPolicy(ask_fn=lambda t, a: True).check(BASH, {}).allowed
    assert not PermissionPolicy(ask_fn=lambda t, a: False).check(BASH, {}).allowed


def test_auto_approve_and_no_approver():
    assert PermissionPolicy(auto_approve=True).check(PUBLISH, {}).allowed   # yolo/headless
    assert not PermissionPolicy().check(BASH, {}).allowed                   # ask, no approver -> deny
