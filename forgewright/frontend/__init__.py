"""Frontend bridge — the backend half of the conversational UI contract.

The swarm runs on the Python backend; the UI (a terminal-kit Node TUI — terminal-kit by
Cédric Ronvel, MIT, https://www.terminal-kit.com/ — or the built-in Python REPL) is pure
presentation. They speak newline-delimited JSON over a stream:
  backend -> UI : assistant / tool / progress / approval_request / done events
  UI -> backend : user_msg / approval_response
This module turns the existing `reporter` callback + `PermissionPolicy.ask_fn` into that
wire format, so the same backend drives either frontend unchanged.
"""
from forgewright.frontend.bridge import (
    StreamApprover,
    event_reporter,
    json_line_emitter,
)

__all__ = ["event_reporter", "json_line_emitter", "StreamApprover"]
