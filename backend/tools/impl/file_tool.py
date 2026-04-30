"""File system tool — read, write, append, and list files.

All file operations are restricted to a configurable workspace root
(``AppConfig.tool_workspace_root``) to prevent path traversal.

When the ``list`` action is used and a :class:`WorkspaceIndexTool` is
injected, the listed directory is automatically scheduled for background
indexing so the agent can later perform semantic search over it without any
manual setup.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.config import AppConfig
    from backend.tools.impl.workspace_index_tool import WorkspaceIndexTool

logger = logging.getLogger(__name__)

_MAX_READ_CHARS = 100_000


class FileTool(Tool):
    """Read, write, append, or list files within the workspace root.

    Actions
    -------
    * ``read``   — return the contents of a file.
    * ``write``  — overwrite a file with new content (creates if absent).
    * ``append`` — append content to a file.
    * ``list``   — list files/directories in a directory.
    * ``delete`` — delete a file.
    * ``exists`` — check whether a path exists.
    """

    name = "file"
    description = (
        "Read, write, append, list, delete, or check existence of files "
        "within the configured workspace directory."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "append", "list", "delete", "exists"],
                "description": "File operation to perform.",
            },
            "path": {
                "type": "string",
                "description": "Relative path within the workspace root.",
            },
            "content": {
                "type": "string",
                "description": "Content to write or append (required for 'write'/'append').",
            },
            "encoding": {
                "type": "string",
                "description": "File encoding (default: utf-8).",
                "default": "utf-8",
            },
        },
        "required": ["action", "path"],
    }

    def __init__(
        self,
        config: "AppConfig",
        workspace_index: "WorkspaceIndexTool | None" = None,
    ) -> None:
        workspace = getattr(config, "tool_workspace_root", "./workspace")
        self._root = Path(workspace).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._workspace_index = workspace_index

    def _safe_path(self, rel_path: str) -> Path | None:
        """Resolve *rel_path* relative to the workspace root.

        Returns ``None`` if the resolved path escapes the workspace root
        (path traversal attempt).
        """
        try:
            resolved = (self._root / rel_path).resolve()
            resolved.relative_to(self._root)  # raises if outside root
            return resolved
        except ValueError:
            return None

    async def execute(
        self,
        action: str = "",
        path: str = "",
        content: str = "",
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> ToolResult:
        if not action:
            return ToolResult(ok=False, error="'action' argument is required")
        if not path:
            return ToolResult(ok=False, error="'path' argument is required")

        safe = self._safe_path(path)
        if safe is None:
            return ToolResult(ok=False, error=f"Path {path!r} escapes the workspace root")

        if action == "read":
            return self._read(safe, encoding)
        if action == "write":
            return self._write(safe, content, encoding)
        if action == "append":
            return self._append(safe, content, encoding)
        if action == "list":
            result = self._list(safe)
            if result.ok and self._workspace_index is not None:
                # Auto-trigger background indexing for the listed directory.
                listed_dir = safe if safe.is_dir() else safe.parent
                if self._workspace_index.is_available():
                    self._workspace_index.trigger_auto_index(listed_dir)
            return result
        if action == "delete":
            return self._delete(safe)
        if action == "exists":
            return ToolResult(ok=True, output={"exists": safe.exists(), "path": str(safe)})
        return ToolResult(ok=False, error=f"Unknown action: {action!r}")

    def _read(self, path: Path, encoding: str) -> ToolResult:
        if not path.exists():
            return ToolResult(ok=False, error=f"File not found: {path.name}")
        if not path.is_file():
            return ToolResult(ok=False, error=f"Not a file: {path.name}")
        try:
            content = path.read_text(encoding=encoding, errors="replace")
            return ToolResult(ok=True, output=content[:_MAX_READ_CHARS])
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

    def _write(self, path: Path, content: str, encoding: str) -> ToolResult:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding=encoding)
            return ToolResult(ok=True, output={"written": path.name, "bytes": len(content.encode(encoding))})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

    def _append(self, path: Path, content: str, encoding: str) -> ToolResult:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding=encoding) as fh:
                fh.write(content)
            return ToolResult(ok=True, output={"appended": path.name, "bytes": len(content.encode(encoding))})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

    def _list(self, path: Path) -> ToolResult:
        if not path.exists():
            return ToolResult(ok=False, error=f"Path not found: {path.name}")
        if path.is_file():
            # List the parent directory, highlight the file
            path = path.parent
        entries = []
        try:
            for entry in sorted(path.iterdir()):
                entries.append({
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None,
                })
            return ToolResult(ok=True, output={"path": str(path.relative_to(self._root)), "entries": entries})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

    def _delete(self, path: Path) -> ToolResult:
        if not path.exists():
            return ToolResult(ok=False, error=f"File not found: {path.name}")
        if not path.is_file():
            return ToolResult(ok=False, error="Only files can be deleted; use a shell command for directories")
        try:
            path.unlink()
            return ToolResult(ok=True, output={"deleted": path.name})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
