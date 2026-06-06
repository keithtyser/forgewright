"""Tool layer: the agent's hands.

Each Tool exposes an OpenAI-style JSON schema (so any LiteLLM brain can call it)
and a synchronous ``run()``. A ToolRegistry collects tools, emits their schemas
for the brain, and dispatches calls. ``risk`` feeds the permission gate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Risk = Literal["read", "write", "exec", "destructive"]


@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: dict[str, Any] = field(default_factory=dict)

    def truncate(self, limit: int = 12000) -> "ToolResult":
        """Clip very long output (keep head + tail) so it fits the context window."""
        if len(self.output) <= limit:
            return self
        keep = limit // 2
        dropped = len(self.output) - limit
        clipped = f"{self.output[:keep]}\n...[{dropped} chars truncated]...\n{self.output[-keep:]}"
        return ToolResult(self.ok, clipped, {**self.meta, "truncated": True})


class Tool(ABC):
    """Base class for a callable tool. Subclasses set the class attrs + implement run()."""

    name: str = ""
    description: str = ""
    risk: Risk = "read"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult: ...

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError(f"{type(tool).__name__} has no name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)
