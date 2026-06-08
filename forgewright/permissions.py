"""Permission gate: autonomy with brakes (not a cage).

Tools declare a ``risk`` level. The policy maps each level to allow / ask / deny.
Default: read/write are allowed (cheap, local), but **exec and destructive ask a human**
so the agent does not just run commands on your machine. The human's answer can be:

  yes  -> allow this one call
  all  -> allow every future call to this tool this session ("approve all similar")
  yolo -> allow everything from now on (bypass all permissions)
  no   -> deny

``ask_fn`` may return any of those strings, or a bool (True/False = yes/no) for simple
callers. Remembered "all" tools and the yolo flag persist for the session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from forgewright.tools.base import Risk, Tool

Mode = str  # "allow" | "ask" | "deny"

DEFAULT_RULES: dict[Risk, Mode] = {
    "read": "allow",
    "write": "allow",
    "exec": "ask",          # bash, forge, launch_job, run_recipe, serving_opt -> ask
    "destructive": "ask",   # publish, deletes -> ask
}


@dataclass
class Decision:
    allowed: bool
    reason: str


class PermissionPolicy:
    def __init__(
        self,
        rules: Optional[dict[Risk, Mode]] = None,
        ask_fn: Optional[Callable[[Tool, dict[str, Any]], Any]] = None,
        auto_approve: bool = False,
    ) -> None:
        self.rules = {**DEFAULT_RULES, **(rules or {})}
        self.ask_fn = ask_fn
        self.auto_approve = auto_approve
        self._allow_tools: set[str] = set()   # "approve all" for these tool names
        self._yolo = False                    # bypass all (set via a 'yolo' decision)

    def check(self, tool: Tool, args: dict[str, Any]) -> Decision:
        if self._yolo or self.auto_approve:
            return Decision(True, "auto-approved (yolo)")
        mode = self.rules.get(tool.risk, "ask")
        if mode == "allow":
            return Decision(True, "allowed by policy")
        if mode == "deny":
            return Decision(False, f"{tool.risk} actions denied by policy")
        # mode == "ask"
        if tool.name in self._allow_tools:
            return Decision(True, f"approved-all for {tool.name}")
        if self.ask_fn is None:
            return Decision(False, "approval required but no approver available")
        decision = self.ask_fn(tool, args)
        if isinstance(decision, bool):
            decision = "yes" if decision else "no"
        decision = str(decision).lower()
        if decision == "yolo":
            self._yolo = True
            return Decision(True, "approved (yolo: bypassing further prompts)")
        if decision == "all":
            self._allow_tools.add(tool.name)
            return Decision(True, f"approved (all future {tool.name} calls)")
        if decision == "yes":
            return Decision(True, "human approved")
        return Decision(False, "human denied")
