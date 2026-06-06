"""Permission gate: autonomy with brakes (not a cage).

Tools declare a ``risk`` level. The policy maps each level to allow / ask / deny.
Default: act freely on read/write/exec (the point of an autonomous harness), but
*ask a human* before destructive/irreversible actions (e.g. publishing weights).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from forgewright.tools.base import Risk, Tool

Mode = str  # "allow" | "ask" | "deny"

DEFAULT_RULES: dict[Risk, Mode] = {
    "read": "allow",
    "write": "allow",
    "exec": "allow",
    "destructive": "ask",
}


@dataclass
class Decision:
    allowed: bool
    reason: str


class PermissionPolicy:
    def __init__(
        self,
        rules: Optional[dict[Risk, Mode]] = None,
        ask_fn: Optional[Callable[[Tool, dict[str, Any]], bool]] = None,
        auto_approve: bool = False,
    ) -> None:
        self.rules = {**DEFAULT_RULES, **(rules or {})}
        self.ask_fn = ask_fn
        self.auto_approve = auto_approve

    def check(self, tool: Tool, args: dict[str, Any]) -> Decision:
        mode = self.rules.get(tool.risk, "ask")
        if mode == "allow":
            return Decision(True, "allowed by policy")
        if mode == "deny":
            return Decision(False, f"{tool.risk} actions denied by policy")
        # mode == "ask"
        if self.auto_approve:
            return Decision(True, "auto-approved")
        if self.ask_fn is not None:
            return Decision(bool(self.ask_fn(tool, args)), "human decision")
        return Decision(False, "approval required but no approver available")
