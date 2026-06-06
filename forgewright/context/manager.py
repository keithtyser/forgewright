"""Context manager: holds the brain's message list, with naive compaction.

Keeps a system message + a rolling transcript in OpenAI/LiteLLM message format
(assistant tool_calls paired with tool results). When the transcript grows past a
char budget, older messages are collapsed into a single note while recent turns are
preserved — taking care not to orphan a ``tool`` message from its assistant call.
"""
from __future__ import annotations

import json
from typing import Any

from forgewright.brain.provider import AssistantTurn, ToolCall
from forgewright.tools.base import ToolResult


class ContextManager:
    def __init__(self, system_prompt: str, max_chars: int = 200_000, keep_recent: int = 30) -> None:
        self._system = {"role": "system", "content": system_prompt}
        self._msgs: list[dict[str, Any]] = []
        self.max_chars = max_chars
        self.keep_recent = keep_recent

    def add_user(self, content: str) -> None:
        self._msgs.append({"role": "user", "content": content})

    def add_assistant(self, turn: AssistantTurn) -> None:
        m: dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            m["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in turn.tool_calls
            ]
        self._msgs.append(m)

    def add_tool_result(self, tc: ToolCall, result: ToolResult) -> None:
        self._msgs.append(
            {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result.output}
        )

    def messages(self) -> list[dict[str, Any]]:
        return [self._system, *self._msgs]

    def _size(self) -> int:
        return sum(len(json.dumps(m, default=str)) for m in self._msgs)

    def maybe_compact(self) -> bool:
        if self._size() <= self.max_chars or len(self._msgs) <= self.keep_recent:
            return False
        tail = self._msgs[-self.keep_recent :]
        # Don't let the tail start on an orphaned tool result.
        while tail and tail[0].get("role") == "tool":
            tail = tail[1:]
        elided = len(self._msgs) - len(tail)
        note = {"role": "user", "content": f"[context compacted: {elided} earlier messages elided]"}
        self._msgs = [note, *tail]
        return True
