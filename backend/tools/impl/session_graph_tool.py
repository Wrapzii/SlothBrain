"""Session graph / memory graph tool.

Provides semantic search and traversal over the session memory graph.
The session graph is not yet a first-class persistent data structure in
SlothBrain — this tool currently queries the LanceDB memory store with
session-scoped filters as a proxy.

When a dedicated session graph is implemented, this tool can be extended
to traverse edges, cluster related sessions, and rank by recency/relevance.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.memory.lancedb_memory import LanceDBMemory

logger = logging.getLogger(__name__)


class SessionGraphTool(Tool):
    """Search the session memory graph for related past sessions and memories.

    Currently backed by semantic search over the LanceDB memory store,
    optionally filtered by a ``session_id`` metadata tag.
    """

    name = "session_graph"
    description = (
        "Search the session memory graph for related past sessions and memories. "
        "Supports semantic queries and optional session-scoped filtering."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic query describing what to look for.",
            },
            "session_id": {
                "type": "string",
                "description": "Optional session ID to restrict the search.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return (default: 10).",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, memory: "LanceDBMemory | None") -> None:
        self._memory = memory

    async def execute(
        self,
        query: str = "",
        session_id: str = "",
        limit: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        if not query:
            return ToolResult(ok=False, error="'query' argument is required")

        if self._memory is None:
            return ToolResult(ok=False, error="Memory store is not available")

        try:
            results = await self._memory.search(query, limit=limit)

            # Post-filter by session_id if provided
            if session_id:
                results = [
                    r for r in results
                    if r.get("metadata", {}).get("session_id") == session_id
                ]

            return ToolResult(
                ok=True,
                output={
                    "query": query,
                    "session_id_filter": session_id or None,
                    "results": results,
                },
            )
        except Exception as exc:
            logger.warning("SessionGraphTool error: %s", exc)
            return ToolResult(ok=False, error=str(exc))
