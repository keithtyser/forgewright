"""The brain: a model-agnostic chat + tool-calling wrapper over LiteLLM.

Any backend LiteLLM supports can drive the agent — a local vLLM server, a hosted
API, or (quarantined) a subscription OAuth tap. The rest of Forgewright depends
only on ``Brain.chat(...)`` returning an ``AssistantTurn``.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Optional

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


class BrainCancelled(BrainError):
    """The caller stopped the request or its retry wait."""


def retry_delay(error: Exception, attempt: int, *, maximum: float = 60) -> float | None:
    """Retry transient transport/server errors; respect a bounded Retry-After."""
    import httpx

    response = getattr(error, "response", None)
    status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    transient = status in {408, 409, 425, 429, 500, 502, 503, 504}
    if status is None:
        transient = isinstance(error, (TimeoutError, ConnectionError, httpx.TransportError)) or (
            type(error).__name__ in {"APIConnectionError", "APITimeoutError", "Timeout"})
    if not transient:
        return None
    headers = getattr(response, "headers", None) or getattr(error, "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is not None:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            try:
                date = parsedate_to_datetime(str(value))
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                delay = (date - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = -1
        if math.isfinite(delay) and delay >= 0:
            return delay if delay <= maximum else None
    return min(maximum, 2 ** min(attempt, 6) * random.uniform(0.8, 1.2))


class Brain:
    """Thin, retrying wrapper around ``litellm.completion`` for one provider."""

    def __init__(
        self,
        provider: ProviderConfig,
        *,
        temperature: float = 0.2,
        # generous: reasoning models (e.g. deepseek-v4-pro) spend hidden reasoning tokens from
        # this budget, so a low cap truncates the visible answer mid-sentence.
        max_tokens: int = 16384,
        max_retries: int = 3,
        timeout: int = 600,
        interrupt: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        if max_retries < 1 or timeout <= 0:
            raise ValueError("max_retries must be at least 1 and timeout must be positive")
        self.max_retries = max_retries  # Total attempts, including the first request.
        self.timeout = timeout
        self.interrupt = interrupt
        # oauth-codex talks to OpenAI's Codex Responses API (ChatGPT login), not LiteLLM, so
        # it delegates to a dedicated client behind the same AssistantTurn contract.
        self._codex = None
        if provider.kind == "oauth-codex":
            from forgewright.brain.codex_oauth import CodexClient

            self._codex = CodexClient(model=provider.model, timeout=timeout)

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
        # One retry owner prevents SDK retries from multiplying our attempt limit.
        kw["num_retries"] = 0
        kw["max_retries"] = 0
        return kw

    def _check_cancelled(self) -> None:
        if self.interrupt and self.interrupt():
            raise BrainCancelled("interrupted")

    def _wait_retry(self, delay: float) -> None:
        while delay > 0:
            self._check_cancelled()
            interval = min(delay, 0.1)
            time.sleep(interval)
            delay -= interval
        self._check_cancelled()

    def _complete(self, request: Callable[[], AssistantTurn]) -> AssistantTurn:
        for attempt in range(self.max_retries):
            self._check_cancelled()
            try:
                result = request()
                self._check_cancelled()
                return result
            except BrainCancelled:
                raise
            except Exception as error:
                self._check_cancelled()
                delay = retry_delay(error, attempt)
                if delay is None or attempt + 1 == self.max_retries:
                    raise BrainError(f"Completion failed after {attempt + 1} attempt(s): {error}") from error
                self._wait_retry(delay)
        raise AssertionError("unreachable")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> AssistantTurn:
        if self._codex is not None:
            client = self._codex
            return self._complete(lambda: client.chat(messages, tools, tool_choice))

        import litellm  # lazy import keeps `import forgewright` fast

        return self._complete(lambda: self._parse(
            litellm.completion(**self._kwargs(messages, tools, tool_choice))))

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
