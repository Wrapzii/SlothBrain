"""Workspace index tool — lazy, usage-driven file-system indexing for the AI agent.

The agent can call this tool to:

* **index** a project directory (recursively, on first access only by default).
* **search** the semantic index to find relevant code or files.
* **status** — list every indexed project root with chunk and file counts.

Indexing is *lazy*: nothing is indexed until the agent or the auto-trigger
mechanism requests it.  When the agent starts working in a new directory it
can call ``{"tool": "workspace_index", "args": {"action": "index", "path": "."}}``
and the folder will be indexed in the background, ready for fast recall.

Auto-triggering
---------------
:class:`~backend.tools.impl.file_tool.FileTool` calls
:meth:`WorkspaceIndexTool.trigger_auto_index` whenever a ``list`` action is
performed, so indexing happens transparently the first time any directory is
explored.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.memory.workspace_indexer import WorkspaceIndexer

logger = logging.getLogger(__name__)


class WorkspaceIndexTool(Tool):
    """Lazy, per-project semantic file index.

    Actions
    -------
    * ``index``  — index a directory (skips already-indexed files by default).
    * ``search`` — semantic search over indexed files.
    * ``status`` — list all indexed project roots.
    """

    name = "workspace_index"
    description = (
        "Index a project directory for fast semantic recall, search indexed "
        "code/files by natural language, or check which projects are indexed. "
        "Indexing is lazy — only touched directories are ever indexed."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["index", "search", "status"],
                "description": (
                    "'index' scans a folder and stores embeddings; "
                    "'search' queries the index; "
                    "'status' lists all indexed projects."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Directory path to index (required for 'index'). "
                    "May be absolute or relative to the current working directory."
                ),
            },
            "query": {
                "type": "string",
                "description": "Natural-language search query (required for 'search').",
            },
            "project_root": {
                "type": "string",
                "description": (
                    "Restrict 'search' results to this project root "
                    "(optional — searches all projects when omitted)."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return for 'search' (default: 8).",
                "default": 8,
            },
            "force": {
                "type": "boolean",
                "description": (
                    "When true, re-index all files even if already indexed "
                    "(default: false)."
                ),
                "default": False,
            },
        },
        "required": ["action"],
    }

    def __init__(self, indexer: "WorkspaceIndexer | None") -> None:
        self._indexer = indexer
        # Tracks directories currently being indexed so we don't double-trigger.
        self._indexing_in_progress: set[str] = set()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def trigger_auto_index(self, directory: str | Path) -> None:
        """Fire-and-forget: schedule background indexing for *directory*.

        Called automatically by :class:`~backend.tools.impl.file_tool.FileTool`
        on every ``list`` action.  No-op if the indexer is unavailable or the
        directory is already being indexed.
        """
        if self._indexer is None:
            return
        resolved = str(Path(directory).resolve())
        if resolved in self._indexing_in_progress:
            return
        self._indexing_in_progress.add(resolved)

        async def _bg() -> None:
            try:
                await self._indexer.index_directory(resolved, force=False)
            except Exception as exc:
                logger.debug("Auto-index failed for %s: %s", resolved, exc)
            finally:
                self._indexing_in_progress.discard(resolved)

        try:
            loop = asyncio.get_event_loop()
            loop.create_task(_bg())
        except RuntimeError:
            # No running loop — skip silently (e.g. during tests)
            self._indexing_in_progress.discard(resolved)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        action: str = "",
        path: str = "",
        query: str = "",
        project_root: str = "",
        limit: int = 8,
        force: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        if not action:
            return ToolResult(ok=False, error="'action' argument is required")

        if self._indexer is None:
            return ToolResult(
                ok=False,
                error=(
                    "Workspace indexer is not available.  "
                    "Ensure lancedb and sentence-transformers are installed."
                ),
            )

        if action == "index":
            return await self._do_index(path, force=force)
        if action == "search":
            return await self._do_search(query, project_root=project_root or None, limit=limit)
        if action == "status":
            return await self._do_status()

        return ToolResult(ok=False, error=f"Unknown action: {action!r}")

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    async def _do_index(self, path: str, *, force: bool) -> ToolResult:
        if not path:
            return ToolResult(ok=False, error="'path' is required for the 'index' action")
        target = Path(path).resolve()
        if not target.exists():
            return ToolResult(ok=False, error=f"Path does not exist: {path!r}")
        if not target.is_dir():
            return ToolResult(ok=False, error=f"Path is not a directory: {path!r}")
        try:
            stats = await self._indexer.index_directory(target, force=force)
            return ToolResult(
                ok=True,
                output={
                    "indexed": str(target),
                    "files_scanned": stats["files_scanned"],
                    "chunks_added": stats["chunks_added"],
                    "files_skipped": stats["files_skipped"],
                },
            )
        except Exception as exc:
            logger.warning("WorkspaceIndexTool index error: %s", exc)
            return ToolResult(ok=False, error=str(exc))

    async def _do_search(
        self,
        query: str,
        *,
        project_root: str | None,
        limit: int,
    ) -> ToolResult:
        if not query:
            return ToolResult(ok=False, error="'query' is required for the 'search' action")
        try:
            results = await self._indexer.search(query, project_root=project_root, limit=limit)
            return ToolResult(ok=True, output={"query": query, "results": results})
        except Exception as exc:
            logger.warning("WorkspaceIndexTool search error: %s", exc)
            return ToolResult(ok=False, error=str(exc))

    async def _do_status(self) -> ToolResult:
        try:
            projects = await self._indexer.get_indexed_projects()
            return ToolResult(ok=True, output={"indexed_projects": projects})
        except Exception as exc:
            logger.warning("WorkspaceIndexTool status error: %s", exc)
            return ToolResult(ok=False, error=str(exc))
