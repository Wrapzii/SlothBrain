"""Diff viewer tool — compute and display unified diffs.

Supports comparing two file versions (by path) or two arbitrary text blobs.
"""
from __future__ import annotations

import difflib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.config import AppConfig


class DiffTool(Tool):
    """Compute a unified diff between two texts or two files.

    Modes
    -----
    * Pass ``text_a`` and ``text_b`` to diff two strings directly.
    * Pass ``file_a`` and ``file_b`` (relative workspace paths) to diff two files.
    * Pass ``file_a`` and ``text_b`` to diff a file against an in-memory string.
    """

    name = "diff"
    description = (
        "Compute a unified diff between two text strings or two files. "
        "Returns the diff in standard unified diff format."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "text_a": {
                "type": "string",
                "description": "First text blob to compare.",
            },
            "text_b": {
                "type": "string",
                "description": "Second text blob to compare.",
            },
            "file_a": {
                "type": "string",
                "description": "Relative path to the first file (within workspace).",
            },
            "file_b": {
                "type": "string",
                "description": "Relative path to the second file (within workspace).",
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of context lines around each change (default: 3).",
                "default": 3,
            },
            "label_a": {
                "type": "string",
                "description": "Label for the 'from' side (default: 'a').",
                "default": "a",
            },
            "label_b": {
                "type": "string",
                "description": "Label for the 'to' side (default: 'b').",
                "default": "b",
            },
        },
        "required": [],
    }

    def __init__(self, config: "AppConfig") -> None:
        workspace = getattr(config, "tool_workspace_root", "./workspace")
        self._root = Path(workspace).resolve()

    def _safe_path(self, rel: str) -> Path | None:
        try:
            resolved = (self._root / rel).resolve()
            resolved.relative_to(self._root)
            return resolved
        except ValueError:
            return None

    async def execute(
        self,
        text_a: str = "",
        text_b: str = "",
        file_a: str = "",
        file_b: str = "",
        context_lines: int = 3,
        label_a: str = "a",
        label_b: str = "b",
        **kwargs: Any,
    ) -> ToolResult:
        # Resolve inputs
        content_a = text_a
        content_b = text_b

        if file_a:
            path = self._safe_path(file_a)
            if path is None:
                return ToolResult(ok=False, error=f"Path {file_a!r} escapes workspace root")
            if not path.exists():
                return ToolResult(ok=False, error=f"File not found: {file_a!r}")
            content_a = path.read_text(encoding="utf-8", errors="replace")
            label_a = file_a

        if file_b:
            path = self._safe_path(file_b)
            if path is None:
                return ToolResult(ok=False, error=f"Path {file_b!r} escapes workspace root")
            if not path.exists():
                return ToolResult(ok=False, error=f"File not found: {file_b!r}")
            content_b = path.read_text(encoding="utf-8", errors="replace")
            label_b = file_b

        if not content_a and not content_b and not file_a and not file_b:
            return ToolResult(ok=False, error="Provide 'text_a'/'text_b' or 'file_a'/'file_b'")

        lines_a = content_a.splitlines(keepends=True)
        lines_b = content_b.splitlines(keepends=True)

        diff_lines = list(difflib.unified_diff(
            lines_a, lines_b,
            fromfile=label_a,
            tofile=label_b,
            n=context_lines,
        ))

        diff_text = "".join(diff_lines)
        return ToolResult(
            ok=True,
            output={
                "diff": diff_text,
                "changed": bool(diff_text),
                "additions": sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++")),
                "deletions": sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---")),
            },
        )
