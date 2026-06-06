"""The brain: a model-agnostic chat + tool-calling wrapper over LiteLLM.

Any backend LiteLLM supports can drive the agent — a local vLLM server, a hosted
API, or (quarantined) a subscription OAuth tap. The rest of Forgewright depends
only on ``Brain.chat(...)`` returning an ``AssistantTurn``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from forgewright.config import ProviderConfig


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class BrainError(RuntimeError):
    pass


class Brain:
    """Thin, retrying wrapper around ``litellm.completion`` for one provider."""

    def __init__(
        self,
        provider: ProviderConfig,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
        timeout: int = 600,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout

    def _kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        tool_choice: str,
    ) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.provider.litellm_model(),
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if (base := self.provider.resolved_api_base()) is not None:
            kw["api_base"] = base
        if (key := self.provider.resolved_api_key()) is not None:
            kw["api_key"] = key
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = tool_choice
        kw.update(self.provider.extra)
        return kw

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> AssistantTurn:
        import litellm  # lazy import keeps `import forgewright` fast

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = litellm.completion(**self._kwargs(messages, tools, tool_choice))
                return self._parse(resp)
            except Exception as e:  # noqa: BLE001 - retried, then surfaced
                last_err = e
                time.sleep(min(2**attempt, 10))
        raise BrainError(
            f"completion failed after {self.max_retries} attempts "
            f"(model={self.provider.litellm_model()!r}): {last_err}"
        )

    @staticmethod
    def _parse(resp: Any) -> AssistantTurn:
        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage: dict[str, int] = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
            }
        return AssistantTurn(content=msg.content or "", tool_calls=calls, usage=usage, raw=resp)
