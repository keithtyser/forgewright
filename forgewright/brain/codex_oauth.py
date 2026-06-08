"""Codex (ChatGPT login) OAuth brain backend.

OpenAI sanctions ChatGPT-account login for their open-source Codex CLI, which talks to the
Responses API at ``https://chatgpt.com/backend-api/codex/responses`` with a bearer token
minted by the Codex OAuth client. This module reuses the credentials Codex already wrote to
``~/.codex/auth.json`` (so the user logs in once with the real Codex CLI), refreshes the
access token when it is near expiry, and adapts our chat/tool-calling shape to the Responses
API so it slots in behind the same ``AssistantTurn`` contract as the LiteLLM brain.

The pure functions (`chat_to_responses`, `parse_responses_sse`, `_jwt_expiry`,
`needs_refresh`) are side-effect-free and unit-tested. ``CodexClient`` adds disk IO + HTTP.

Notes / caveats:
- The endpoint is unofficial (it is what the open-source Codex CLI uses). Anthropic's own
  Claude subscription tap is deliberately NOT implemented.
- We never store new credentials except to rewrite the refreshed token triple back into the
  same ``auth.json`` Codex owns.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from forgewright.brain.provider import AssistantTurn, BrainError, ToolCall

# The public Codex OAuth client id (same value the open-source Codex CLI ships).
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
TOKEN_URL = "https://auth.openai.com/oauth/token"
RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_MODEL = "gpt-5-codex"
# Refresh once the access token has under this many seconds of life left.
REFRESH_SKEW_SECONDS = 300


def default_auth_path() -> Path:
    return Path.home() / ".codex" / "auth.json"


# ---------------------------------------------------------------------------
# Credentials (auth.json) + token expiry — pure where possible
# ---------------------------------------------------------------------------
@dataclass
class CodexTokens:
    access_token: str
    refresh_token: str
    id_token: str = ""
    account_id: str = ""

    @classmethod
    def from_auth_json(cls, data: dict[str, Any]) -> "CodexTokens":
        tok = data.get("tokens") or {}
        access = tok.get("access_token") or ""
        refresh = tok.get("refresh_token") or ""
        if not access or not refresh:
            raise BrainError(
                "Codex auth.json is missing tokens.access_token / tokens.refresh_token. "
                "Run `codex login` with the official Codex CLI first."
            )
        return cls(
            access_token=access,
            refresh_token=refresh,
            id_token=tok.get("id_token") or "",
            account_id=tok.get("account_id") or _account_id_from_id_token(tok.get("id_token") or ""),
        )

    def to_auth_json(self, base: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        data = dict(base or {})
        tok = dict(data.get("tokens") or {})
        tok.update(
            {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "id_token": self.id_token,
                "account_id": self.account_id,
            }
        )
        data["tokens"] = tok
        data["last_refresh"] = _utc_now_iso()
        return data


def _b64url_decode(seg: str) -> bytes:
    seg += "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg.encode("ascii"))


def _jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT payload without verifying the signature (we only read exp / account)."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        return json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except Exception:  # noqa: BLE001 - best-effort decode
        return {}


def _jwt_expiry(token: str) -> Optional[int]:
    exp = _jwt_claims(token).get("exp")
    return int(exp) if isinstance(exp, (int, float)) else None


def _account_id_from_id_token(id_token: str) -> str:
    """ChatGPT account id lives under the OpenAI auth claim in the id_token."""
    claims = _jwt_claims(id_token)
    auth = claims.get("https://api.openai.com/auth") or {}
    return auth.get("chatgpt_account_id") or auth.get("account_id") or ""


def needs_refresh(access_token: str, *, now: Optional[float] = None, skew: int = REFRESH_SKEW_SECONDS) -> bool:
    exp = _jwt_expiry(access_token)
    if exp is None:
        return False  # opaque token: assume the server validates it; don't force a refresh loop
    now = time.time() if now is None else now
    return now >= (exp - skew)


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Chat <-> Responses API conversion (pure)
# ---------------------------------------------------------------------------
def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # OpenAI-style content arrays: join any text parts.
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(part.get("text") or part.get("content") or "")
            else:
                out.append(str(part))
        return "".join(out)
    return str(content)


def chat_to_responses(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    model: str,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    """Convert chat-completions messages + tools into a Responses API request body.

    - A leading ``system`` message becomes ``instructions`` (the Responses field for it);
      any later system messages are folded in as developer-role input items.
    - ``user``/``assistant`` text become ``input_text``/``output_text`` content items.
    - assistant ``tool_calls`` become ``function_call`` items; ``tool`` messages become
      ``function_call_output`` items keyed by ``call_id``.
    - tools are flattened to the Responses function shape (name/description/parameters at the
      top level, not nested under ``function``).
    """
    instructions = ""
    input_items: list[dict[str, Any]] = []

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "system":
            text = _content_text(msg.get("content"))
            if i == 0 and not instructions:
                instructions = text
            else:
                input_items.append(
                    {"role": "developer", "content": [{"type": "input_text", "text": text}]}
                )
        elif role == "user":
            input_items.append(
                {"role": "user", "content": [{"type": "input_text", "text": _content_text(msg.get("content"))}]}
            )
        elif role == "assistant":
            text = _content_text(msg.get("content"))
            if text:
                input_items.append(
                    {"role": "assistant", "content": [{"type": "output_text", "text": text}]}
                )
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args or {})
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": args,
                    }
                )
        elif role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or "",
                    "output": _content_text(msg.get("content")),
                }
            )

    body: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "store": False,
        "stream": True,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "include": [],
        "parallel_tool_calls": True,
    }
    if tools:
        body["tools"] = [_flatten_tool(t) for t in tools]
        body["tool_choice"] = tool_choice
    return body


def _flatten_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """chat tool ({type:function, function:{name,description,parameters}}) -> Responses shape."""
    fn = tool.get("function") or tool
    return {
        "type": "function",
        "name": fn.get("name") or "",
        "description": fn.get("description") or "",
        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _iter_sse_events(raw: str):
    """Yield (event_type, data_dict) from an SSE text stream."""
    for block in raw.split("\n\n"):
        event_type = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        yield event_type or (data.get("type") if isinstance(data, dict) else None), data


def parse_responses_sse(raw: str) -> AssistantTurn:
    """Assemble streamed Responses SSE text into an AssistantTurn.

    Handles ``response.output_text.delta`` (visible text), reasoning summary deltas (ignored
    for content but kept out of the answer), function-call items (assembled from added items +
    argument deltas, or read whole from ``response.completed``), and usage from the final
    ``response.completed`` event.
    """
    text_parts: list[str] = []
    # call_id/item_id -> {"name","arguments","call_id"} assembled incrementally
    calls: dict[str, dict[str, str]] = {}
    order: list[str] = []
    usage: dict[str, int] = {}

    def _ensure(key: str) -> dict[str, str]:
        if key not in calls:
            calls[key] = {"name": "", "arguments": "", "call_id": ""}
            order.append(key)
        return calls[key]

    for event_type, data in _iter_sse_events(raw):
        et = event_type or ""
        if et == "response.output_text.delta":
            text_parts.append(str(data.get("delta") or ""))
        elif et in ("response.output_item.added", "response.output_item.done"):
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                key = item.get("id") or item.get("call_id") or str(len(order))
                rec = _ensure(key)
                rec["name"] = item.get("name") or rec["name"]
                rec["call_id"] = item.get("call_id") or rec["call_id"]
                if isinstance(item.get("arguments"), str) and item["arguments"]:
                    rec["arguments"] = item["arguments"]
        elif et == "response.function_call_arguments.delta":
            key = data.get("item_id") or data.get("id") or ""
            if key:
                _ensure(key)["arguments"] += str(data.get("delta") or "")
        elif et == "response.function_call_arguments.done":
            key = data.get("item_id") or data.get("id") or ""
            if key and isinstance(data.get("arguments"), str):
                _ensure(key)["arguments"] = data["arguments"]
        elif et == "response.completed":
            resp = data.get("response") or {}
            usage = _usage_from_response(resp)
            # Fallback: pull any function_calls present in the final output if we missed deltas.
            for item in resp.get("output") or []:
                if item.get("type") == "function_call":
                    key = item.get("id") or item.get("call_id") or str(len(order))
                    rec = _ensure(key)
                    rec["name"] = item.get("name") or rec["name"]
                    rec["call_id"] = item.get("call_id") or rec["call_id"]
                    if isinstance(item.get("arguments"), str) and item["arguments"]:
                        rec["arguments"] = item["arguments"]
        elif et == "response.failed":
            resp = data.get("response") or {}
            err = (resp.get("error") or {}).get("message") or "Codex response failed"
            raise BrainError(f"Codex Responses API error: {err}")

    tool_calls: list[ToolCall] = []
    for key in order:
        rec = calls[key]
        if not rec["name"]:
            continue
        try:
            args = json.loads(rec["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {"_raw": rec["arguments"]}
        tool_calls.append(ToolCall(id=rec["call_id"] or key, name=rec["name"], arguments=args))

    return AssistantTurn(content="".join(text_parts), tool_calls=tool_calls, usage=usage, raw=None)


def _usage_from_response(resp: dict[str, Any]) -> dict[str, int]:
    u = resp.get("usage") or {}
    inp = int(u.get("input_tokens") or 0)
    out = int(u.get("output_tokens") or 0)
    total = int(u.get("total_tokens") or (inp + out))
    return {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": total}


# ---------------------------------------------------------------------------
# Client: disk IO + HTTP (refresh + responses call)
# ---------------------------------------------------------------------------
class CodexClient:
    """Drives the Codex Responses API using ChatGPT-login credentials from ``auth.json``."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        auth_path: Optional[Path] = None,
        timeout: int = 600,
    ) -> None:
        self.model = model
        self.auth_path = auth_path or default_auth_path()
        self.timeout = timeout

    # -- credentials ---------------------------------------------------------
    def _load_auth(self) -> tuple[dict[str, Any], CodexTokens]:
        if not self.auth_path.exists():
            raise BrainError(
                f"No Codex credentials at {self.auth_path}. Install the official Codex CLI and run "
                "`codex login` (ChatGPT account) first, then retry with --brain oauth-codex."
            )
        try:
            data = json.loads(self.auth_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise BrainError(f"Codex auth.json at {self.auth_path} is not valid JSON: {e}") from e
        return data, CodexTokens.from_auth_json(data)

    def _refresh(self, base: dict[str, Any], tokens: CodexTokens) -> CodexTokens:
        import httpx

        try:
            resp = httpx.post(
                TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": tokens.refresh_token,
                    "client_id": CODEX_CLIENT_ID,
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
        except Exception as e:  # noqa: BLE001
            raise BrainError(f"Codex token refresh request failed: {e}") from e
        if resp.status_code != 200:
            raise BrainError(
                f"Codex token refresh failed ({resp.status_code}): {resp.text[:300]}. "
                "Your ChatGPT session may have expired; run `codex login` again."
            )
        payload = resp.json()
        refreshed = CodexTokens(
            access_token=payload.get("access_token") or tokens.access_token,
            refresh_token=payload.get("refresh_token") or tokens.refresh_token,
            id_token=payload.get("id_token") or tokens.id_token,
            account_id=tokens.account_id
            or _account_id_from_id_token(payload.get("id_token") or tokens.id_token),
        )
        try:
            self.auth_path.write_text(
                json.dumps(refreshed.to_auth_json(base), indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # in-memory token still usable for this process even if we can't persist it
        return refreshed

    def _ensure_token(self) -> CodexTokens:
        base, tokens = self._load_auth()
        if needs_refresh(tokens.access_token):
            tokens = self._refresh(base, tokens)
        return tokens

    # -- chat ----------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> AssistantTurn:
        import httpx

        tokens = self._ensure_token()
        body = chat_to_responses(messages, tools, self.model, tool_choice)
        headers = {
            "Authorization": f"Bearer {tokens.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "forgewright-codex/0.1",
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
        }
        if tokens.account_id:
            headers["ChatGPT-Account-Id"] = tokens.account_id

        chunks: list[str] = []
        try:
            with httpx.stream(
                "POST", RESPONSES_URL, headers=headers, json=body, timeout=self.timeout
            ) as resp:
                if resp.status_code != 200:
                    detail = resp.read().decode("utf-8", "replace")[:400]
                    raise BrainError(
                        f"Codex Responses API returned {resp.status_code}: {detail}"
                    )
                for chunk in resp.iter_text():
                    if chunk:
                        chunks.append(chunk)
        except BrainError:
            raise
        except Exception as e:  # noqa: BLE001
            raise BrainError(f"Codex Responses request failed: {e}") from e

        return parse_responses_sse("".join(chunks))
