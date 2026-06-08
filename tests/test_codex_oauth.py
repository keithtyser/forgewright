"""Unit tests for the Codex (ChatGPT login) OAuth brain: the pure chat->Responses converter,
the SSE parser, JWT-expiry refresh logic, and --brain parsing. No network is touched."""
from __future__ import annotations

import base64
import json
import time

import pytest

from forgewright.brain.codex_oauth import (
    CodexTokens,
    chat_to_responses,
    needs_refresh,
    parse_responses_sse,
)
from forgewright.config import parse_brain_arg


def _jwt(payload: dict) -> str:
    def seg(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")

    return seg({"alg": "none"}) + "." + seg(payload) + ".sig"


# --- --brain parsing --------------------------------------------------------
def test_parse_brain_oauth_codex_default_model():
    p = parse_brain_arg("oauth-codex")
    assert p.kind == "oauth-codex"
    assert p.model == "gpt-5-codex"


def test_parse_brain_oauth_codex_explicit_model():
    p = parse_brain_arg("oauth-codex:gpt-5.1-codex")
    assert p.kind == "oauth-codex"
    assert p.model == "gpt-5.1-codex"


def test_parse_brain_oauth_claude_still_rejected():
    with pytest.raises(ValueError):
        parse_brain_arg("oauth-claude:whatever")


# --- token refresh logic ----------------------------------------------------
def test_needs_refresh_expired():
    tok = _jwt({"exp": int(time.time()) - 10})
    assert needs_refresh(tok) is True


def test_needs_refresh_fresh():
    tok = _jwt({"exp": int(time.time()) + 3600})
    assert needs_refresh(tok) is False


def test_needs_refresh_within_skew():
    tok = _jwt({"exp": int(time.time()) + 60})  # under the 300s skew
    assert needs_refresh(tok) is True


def test_needs_refresh_opaque_token_no_forced_loop():
    assert needs_refresh("not-a-jwt") is False


def test_tokens_account_id_from_id_token():
    idt = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}})
    tokens = CodexTokens.from_auth_json(
        {"tokens": {"access_token": "a", "refresh_token": "r", "id_token": idt}}
    )
    assert tokens.account_id == "acct-123"


def test_tokens_missing_required_raises():
    from forgewright.brain.provider import BrainError

    with pytest.raises(BrainError):
        CodexTokens.from_auth_json({"tokens": {"access_token": "a"}})


# --- chat -> Responses conversion -------------------------------------------
def test_chat_to_responses_system_becomes_instructions():
    body = chat_to_responses(
        [{"role": "system", "content": "you are forge"}, {"role": "user", "content": "hi"}],
        None,
        "gpt-5-codex",
    )
    assert body["instructions"] == "you are forge"
    assert body["store"] is False
    assert body["input"][0]["role"] == "user"
    assert body["input"][0]["content"][0] == {"type": "input_text", "text": "hi"}


def test_chat_to_responses_tools_flattened():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "run a command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
            },
        }
    ]
    body = chat_to_responses([{"role": "user", "content": "go"}], tools, "gpt-5-codex")
    t = body["tools"][0]
    assert t["type"] == "function"
    assert t["name"] == "shell"
    assert t["parameters"]["properties"]["cmd"]["type"] == "string"
    assert body["tool_choice"] == "auto"


def test_chat_to_responses_assistant_toolcall_and_result():
    messages = [
        {"role": "user", "content": "list"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "shell", "arguments": '{"cmd":"ls"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file.txt"},
    ]
    body = chat_to_responses(messages, None, "gpt-5-codex")
    items = body["input"]
    fc = next(i for i in items if i.get("type") == "function_call")
    assert fc["call_id"] == "call_1"
    assert fc["name"] == "shell"
    assert fc["arguments"] == '{"cmd":"ls"}'
    out = next(i for i in items if i.get("type") == "function_call_output")
    assert out["call_id"] == "call_1"
    assert out["output"] == "file.txt"


# --- SSE parsing ------------------------------------------------------------
def test_parse_responses_sse_text_and_usage():
    raw = (
        'event: response.output_text.delta\ndata: {"delta": "Hel"}\n\n'
        'event: response.output_text.delta\ndata: {"delta": "lo"}\n\n'
        'event: response.completed\ndata: {"response": {"usage": '
        '{"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}}\n\n'
        "data: [DONE]\n\n"
    )
    turn = parse_responses_sse(raw)
    assert turn.content == "Hello"
    assert turn.usage["total_tokens"] == 12
    assert turn.tool_calls == []


def test_parse_responses_sse_function_call_via_deltas():
    raw = (
        'event: response.output_item.added\ndata: {"item": {"type": "function_call", '
        '"id": "fc_1", "call_id": "call_9", "name": "shell"}}\n\n'
        'event: response.function_call_arguments.delta\ndata: {"item_id": "fc_1", "delta": "{\\"cmd\\":"}\n\n'
        'event: response.function_call_arguments.delta\ndata: {"item_id": "fc_1", "delta": "\\"ls\\"}"}\n\n'
        'event: response.completed\ndata: {"response": {"usage": {}}}\n\n'
    )
    turn = parse_responses_sse(raw)
    assert len(turn.tool_calls) == 1
    tc = turn.tool_calls[0]
    assert tc.name == "shell"
    assert tc.id == "call_9"
    assert tc.arguments == {"cmd": "ls"}


def test_parse_responses_sse_function_call_from_completed_fallback():
    raw = (
        'event: response.completed\ndata: {"response": {"usage": {}, "output": ['
        '{"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "read_file", '
        '"arguments": "{\\"path\\":\\"a.txt\\"}"}]}}\n\n'
    )
    turn = parse_responses_sse(raw)
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "read_file"
    assert turn.tool_calls[0].arguments == {"path": "a.txt"}


def test_parse_responses_sse_reasoning_not_in_content():
    raw = (
        'event: response.reasoning_summary_text.delta\ndata: {"delta": "thinking..."}\n\n'
        'event: response.output_text.delta\ndata: {"delta": "answer"}\n\n'
        'event: response.completed\ndata: {"response": {"usage": {}}}\n\n'
    )
    turn = parse_responses_sse(raw)
    assert turn.content == "answer"


def test_parse_responses_sse_failed_raises():
    from forgewright.brain.provider import BrainError

    raw = 'event: response.failed\ndata: {"response": {"error": {"message": "boom"}}}\n\n'
    with pytest.raises(BrainError):
        parse_responses_sse(raw)
