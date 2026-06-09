"""Context manager: model-aware token budget + layered, summary-based compaction.

Design (grounded in current agent context-engineering practice):
 - The budget is derived from the BRAIN'S OWN context window (via LiteLLM's model registry), so a
   32k model compacts early while a 200k / 1M model rarely compacts at all. Falls back to an env
   override / a conservative default for unknown models.
 - Compaction is layered, lightest-touch first (Anthropic's guidance):
     1) tool-result clearing  -- stub out old tool outputs (they're rarely needed again),
     2) anchored summarization -- fold the oldest non-recent turns into a PERSISTENT summary that
        is merged (not regenerated) each time, preserving goal/decisions/artifact-ids/gate results
        and discarding redundant output. Falls back to a lossy elision note if no summarizer.
 - A PINNED note (the goal + key state) and the persistent summary are never elided, and the most
   recent turns are kept verbatim -- so the agent keeps the thread across very long runs.
 - The registry/ledger/transcripts remain the durable external memory; this only manages what is
   sent to the model each step.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from forgewright.brain.provider import AssistantTurn, ToolCall
from forgewright.tools.base import ToolResult

# Summarizer: (messages_to_fold_in, prior_summary) -> merged summary text.
Summarizer = Callable[[list[dict], Optional[str]], str]

_CHARS_PER_TOKEN = 3.5          # conservative estimate (dense JSON/code) -> compacts a bit early
_DEFAULT_WINDOW = 128_000       # fallback input window for models not in the LiteLLM registry
_MIN_BUDGET_TOKENS = 6_000      # never shrink the working budget below this
_TOOL_STUB = "[older tool output cleared to save context]"


def model_input_window(model: Optional[str]) -> Optional[int]:
    """The model's max INPUT tokens via LiteLLM's registry (handles 200k / 1M / etc.), or None."""
    if not model:
        return None
    try:
        import litellm

        info = litellm.get_model_info(model) or {}
        win = info.get("max_input_tokens") or info.get("max_tokens")
        if win:
            return int(win)
    except Exception:  # noqa: BLE001 - unknown model / offline registry
        pass
    try:
        import litellm

        mt = litellm.get_max_tokens(model)
        if mt:
            return int(mt)
    except Exception:  # noqa: BLE001
        pass
    return None


class ContextManager:
    def __init__(
        self,
        system_prompt: str,
        *,
        model: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
        reserved_output_tokens: int = 16_384,
        keep_recent: int = 20,
        trigger_fraction: float = 0.85,
        summarizer: Optional[Summarizer] = None,
    ) -> None:
        self._system = {"role": "system", "content": system_prompt}
        self._pinned: Optional[dict[str, Any]] = None      # durable note (goal + key state)
        self._summary: Optional[str] = None                # persistent anchored summary
        self._msgs: list[dict[str, Any]] = []
        self.model = model
        self.reserved_output_tokens = reserved_output_tokens
        self.keep_recent = keep_recent
        self.trigger_fraction = trigger_fraction
        self.summarizer = summarizer
        self._budget = self._resolve_budget(model, max_input_tokens)

    # --- configuration (the Agent wires the model + summarizer post-construction) ----
    def configure(self, *, model: Optional[str] = None, summarizer: Optional[Summarizer] = None) -> None:
        if model:
            self.model = model
            self._budget = self._resolve_budget(model, None)
        if summarizer is not None:
            self.summarizer = summarizer

    def set_pinned(self, text: str) -> None:
        """A note kept at the top of context and never elided (e.g. the goal + active artifacts)."""
        self._pinned = {"role": "user", "content": "[pinned context]\n" + text} if text else None

    # --- checkpoint/resume -----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Serialize the working state (pinned + running summary + live turns) for a checkpoint.
        The system prompt is fixed at construction, so it is not snapshotted."""
        return {"pinned": self._pinned, "summary": self._summary, "msgs": self._msgs}

    def restore(self, state: dict[str, Any]) -> None:
        """Rehydrate the working state from a snapshot (used on resume)."""
        if not state:
            return
        self._pinned = state.get("pinned")
        self._summary = state.get("summary")
        self._msgs = list(state.get("msgs") or [])

    def _resolve_budget(self, model: Optional[str], override: Optional[int]) -> int:
        win = override or model_input_window(model)
        if not win:
            env = os.environ.get("FORGEWRIGHT_CONTEXT_TOKENS")
            win = int(env) if (env and env.isdigit()) else _DEFAULT_WINDOW
        # reserve room for the model's own generation + tool schemas/system overhead
        return max(_MIN_BUDGET_TOKENS, int(win) - self.reserved_output_tokens - 4_000)

    # --- transcript building ---------------------------------------------------------
    def add_user(self, content: str) -> None:
        self._msgs.append({"role": "user", "content": content})

    def add_assistant(self, turn: AssistantTurn) -> None:
        m: dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            m["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in turn.tool_calls
            ]
        self._msgs.append(m)

    def add_tool_result(self, tc: ToolCall, result: ToolResult) -> None:
        self._msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result.output})

    def messages(self) -> list[dict[str, Any]]:
        head: list[dict[str, Any]] = [self._system]
        if self._pinned:
            head.append(self._pinned)
        if self._summary:
            head.append({"role": "user", "content": "[summary of earlier conversation]\n" + self._summary})
        return [*head, *self._msgs]

    # --- budgeting -------------------------------------------------------------------
    def _est_tokens(self, msgs: Optional[list[dict]] = None) -> int:
        msgs = self.messages() if msgs is None else msgs
        chars = sum(len(json.dumps(m, default=str)) for m in msgs)
        return int(chars / _CHARS_PER_TOKEN)

    def estimated_tokens(self) -> int:
        return self._est_tokens()

    @property
    def budget_tokens(self) -> int:
        return self._budget

    # --- compaction ------------------------------------------------------------------
    def maybe_compact(self) -> bool:
        threshold = int(self._budget * self.trigger_fraction)
        if self._est_tokens() <= threshold or len(self._msgs) <= self.keep_recent:
            return False
        # 1) lightest touch: clear old tool outputs (keep the recent window's results intact)
        self._clear_old_tool_results()
        if self._est_tokens() <= threshold:
            return True
        # 2) anchored summarization of the oldest non-recent turns
        tail = self._msgs[-self.keep_recent:]
        while tail and tail[0].get("role") == "tool":   # never start the tail on an orphan tool result
            tail = tail[1:]
        old = self._msgs[: len(self._msgs) - len(tail)]
        if old:
            self._summary = self._summarize(old, self._summary)
            self._msgs = tail
        return True

    def _clear_old_tool_results(self) -> None:
        cutoff = len(self._msgs) - self.keep_recent
        for i, m in enumerate(self._msgs):
            if i >= cutoff:
                break
            if m.get("role") == "tool" and m.get("content") and m["content"] != _TOOL_STUB:
                m["content"] = _TOOL_STUB

    def _summarize(self, old: list[dict], prev: Optional[str]) -> str:
        if self.summarizer is not None:
            try:
                merged = self.summarizer(old, prev)
                if merged and merged.strip():
                    return merged.strip()
            except Exception:  # noqa: BLE001 - summary must never break the loop
                pass
        # fallback: lossy elision note, anchored on the prior summary
        note = f"[{len(old)} earlier messages elided]"
        return f"{prev}\n{note}" if prev else note
