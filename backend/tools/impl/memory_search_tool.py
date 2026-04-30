"""Memory search tool — semantic search over the LanceDB memory store.

Wraps :class:`~backend.memory.lancedb_memory.LanceDBMemory` so agents can
query past memories directly as a tool call.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.memory.lancedb_memory import LanceDBMemory

logger = logging.getLogger(__name__)


class MemorySearchTool(Tool):
    """Search the vector memory store for relevant past context.

    Returns a list of the most semantically similar stored memories.
    """

    name = "memory_search"
    description = (
        "Search past memories and conversation context using semantic similarity. "
        "Returns the most relevant stored snippets for a given query."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, memory: "LanceDBMemory | None") -> None:
        self._memory = memory

    async def execute(self, query: str = "", limit: int = 5, **kwargs: Any) -> ToolResult:
        if not query:
            return ToolResult(ok=False, error="'query' argument is required")

        if self._memory is None:
            return ToolResult(ok=False, error="Memory store is not available")

        try:
            results = await self._memory.search(query, limit=limit)
            return ToolResult(ok=True, output={"query": query, "results": results})
        except Exception as exc:
            logger.warning("MemorySearchTool error: %s", exc)
            return ToolResult(ok=False, error=str(exc))
