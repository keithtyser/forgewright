"""Tests for model-aware budgeting + layered (tool-clear -> summarize) compaction."""
from __future__ import annotations

from forgewright.brain.provider import AssistantTurn, ToolCall
from forgewright.context.manager import ContextManager, model_input_window
from forgewright.tools.base import ToolResult


def _cm(**kw):
    # small window so compaction is easy to trigger deterministically
    kw.setdefault("max_input_tokens", 20_000)
    kw.setdefault("reserved_output_tokens", 2_000)
    kw.setdefault("keep_recent", 6)
    return ContextManager("system", **kw)


def test_budget_is_model_window_minus_reserves():
    cm = _cm()
    assert cm.budget_tokens == 20_000 - 2_000 - 4_000   # 14000


def test_budget_falls_back_to_default_for_unknown_model(monkeypatch):
    monkeypatch.delenv("FORGEWRIGHT_CONTEXT_TOKENS", raising=False)
    cm = ContextManager("system", model="totally-unknown-model-xyz", reserved_output_tokens=16_384)
    assert cm.budget_tokens >= 6_000   # default 128k window -> sizeable budget, never below the floor


def test_env_override_budget(monkeypatch):
    monkeypatch.setenv("FORGEWRIGHT_CONTEXT_TOKENS", "1000000")
    cm = ContextManager("system", model="unknown", reserved_output_tokens=16_384)
    assert cm.budget_tokens > 900_000   # 1M-context handling


def test_model_input_window_known_model_or_none():
    win = model_input_window("gpt-4o")          # in the LiteLLM registry
    assert win is None or win > 10_000          # don't hard-fail if the registry id drifts
    assert model_input_window("") is None


def test_no_compaction_under_budget():
    cm = _cm()
    cm.add_user("hello")
    cm.add_assistant(AssistantTurn("hi"))
    assert cm.maybe_compact() is False
    assert cm._summary is None


def test_summary_compaction_preserves_pinned_and_recent():
    seen = {"calls": 0}

    def summarizer(msgs, prev):
        seen["calls"] += 1
        seen["prev"] = prev
        return (prev or "") + f"|folded {len(msgs)}"

    cm = _cm(summarizer=summarizer)
    cm.set_pinned("Goal: do the thing")
    for i in range(40):
        cm.add_user(f"message number {i} " + "x" * 1200)   # bloat past the budget
    assert cm.maybe_compact() is True
    assert seen["calls"] == 1
    # the summary note + pinned + system are all present; recent kept verbatim
    msgs = cm.messages()
    assert msgs[0]["role"] == "system"
    assert any("pinned context" in m.get("content", "") for m in msgs)
    assert any("summary of earlier conversation" in m.get("content", "") for m in msgs)
    assert len(cm._msgs) == cm.keep_recent


def test_anchored_summary_merges_prior():
    summaries = []

    def summarizer(msgs, prev):
        summaries.append(prev)
        return f"S{len(summaries)}"

    cm = _cm(summarizer=summarizer)
    for _ in range(40):
        cm.add_user("x" * 1500)
    cm.maybe_compact()                         # first summary: prev is None
    for _ in range(40):
        cm.add_user("y" * 1500)
    cm.maybe_compact()                         # second: prev is the first summary (anchored)
    assert summaries[0] is None and summaries[1] == "S1"


def test_tool_result_clearing_first():
    cm = _cm()   # no summarizer
    # many old tool results; clearing them should drop size without needing summarization
    for i in range(40):
        tc = ToolCall(str(i), "tail_logs", {"n": 200})
        cm.add_assistant(AssistantTurn("", [tc]))
        cm.add_tool_result(tc, ToolResult(True, "L" * 2000))
    cm.maybe_compact()
    cleared = [m for m in cm._msgs if m.get("role") == "tool" and "cleared" in (m.get("content") or "")]
    assert cleared, "old tool outputs should be stubbed by the lightest-touch pass"


def test_fallback_without_summarizer_uses_elision_note():
    cm = _cm()   # no summarizer; use text messages so tool-clearing can't save it
    for i in range(60):
        cm.add_user("u" * 1500)
    cm.maybe_compact()
    assert cm._summary is not None and "elided" in cm._summary
