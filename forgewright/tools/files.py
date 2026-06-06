"""Local filesystem tools: read, write, edit. (Remote files go through bash_remote.)"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from forgewright.tools.base import Tool, ToolResult


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file from the local filesystem (optionally a 1-based line range)."
    risk = "read"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start": {"type": "integer", "description": "1-based start line (optional)."},
            "end": {"type": "integer", "description": "Inclusive end line (optional)."},
        },
        "required": ["path"],
    }

    def run(self, path: str, start: int | None = None, end: int | None = None, **_: Any) -> ToolResult:
        p = Path(path).expanduser()
        if not p.is_file():
            return ToolResult(False, f"not a file: {p}")
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, f"read error: {e}")
        base = start or 1
        chosen = lines[(base - 1) : (end or len(lines))]
        numbered = "\n".join(f"{base + i}\t{ln}" for i, ln in enumerate(chosen))
        return ToolResult(True, numbered or "(empty)", {"path": str(p), "lines": len(chosen)}).truncate(16000)


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or overwrite a UTF-8 text file on the local filesystem. Creates parent dirs."
    risk = "write"
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str, **_: Any) -> ToolResult:
        p = Path(path).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, f"write error: {e}")
        return ToolResult(True, f"wrote {len(content)} chars to {p}", {"path": str(p)})


class EditFileTool(Tool):
    name = "edit_file"
    description = "Replace exact occurrence(s) of old_string with new_string in a local file."
    risk = "write"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)."},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def run(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **_: Any,
    ) -> ToolResult:
        p = Path(path).expanduser()
        if not p.is_file():
            return ToolResult(False, f"not a file: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
        n = text.count(old_string)
        if n == 0:
            return ToolResult(False, "old_string not found")
        if n > 1 and not replace_all:
            return ToolResult(False, f"old_string occurs {n} times; set replace_all or make it unique")
        text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        p.write_text(text, encoding="utf-8")
        return ToolResult(True, f"edited {p} ({n if replace_all else 1} occurrence(s))", {"path": str(p)})
