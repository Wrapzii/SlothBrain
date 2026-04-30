"""Tool registry — stores, looks up, and renders tool descriptions.

The registry is a lightweight in-process dictionary of :class:`~backend.tools.base.Tool`
instances.  It is constructed once during application startup and injected into
agents that need tool-calling capabilities.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from backend.core.semantic_router import SemanticRouter
from backend.tools.base import Tool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# XML fence markers used in the agent prompt / response protocol.
_CALL_OPEN = "<tool_call>"
_CALL_CLOSE = "</tool_call>"


class ToolRegistry:
    """Registry of available :class:`~backend.tools.base.Tool` instances.

    Usage::

        registry = ToolRegistry()
        registry.register(FileTool(...))

        tools = registry.get_tools(context="Read and patch config file")
        prompt_block = registry.render_tool_descriptions(tools)
    """

    def __init__(
        self,
        *,
        semantic_router: SemanticRouter | None = None,
        semantic_top_k: int = 8,
        semantic_min_similarity: float = 0.2,
        critical_bypass_tools: list[str] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._semantic_router = semantic_router
        self._semantic_top_k = max(1, int(semantic_top_k))
        self._semantic_min_similarity = float(semantic_min_similarity)
        self._critical_bypass_tools = set(critical_bypass_tools or [])

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register a tool.  Overwrites any existing tool with the same name."""
        if not tool.name:
            raise ValueError("Tool must have a non-empty name")
        self._tools[tool.name] = tool
        if self._semantic_router is not None:
            self._semantic_router.register_tool(
                tool_name=tool.name,
                description=tool.description,
                parameters_schema=tool.parameters_schema,
            )
        logger.debug("Tool registered: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool by name (no-op if not found)."""
        self._tools.pop(name, None)
        if self._semantic_router is not None:
            self._semantic_router.unregister_tool(name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Return the tool with *name*, or ``None`` if not registered."""
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_tools(self, context: str = "") -> list[Tool]:
        """Return globally available tools, optionally semantically routed by *context*."""
        allowed = self.all_tools()

        if self._semantic_router is None:
            return allowed

        candidate_names = [t.name for t in allowed]
        result = self._semantic_router.get_relevant_tools(
            context=context,
            profile="global",
            candidates=candidate_names,
            top_k=self._semantic_top_k,
            min_similarity=self._semantic_min_similarity,
            critical_tools=list(self._critical_bypass_tools),
        )
        selected = [self._tools[name] for name in result.tool_names if name in self._tools]
        if result.used_fallback:
            logger.debug("Semantic router fallback (%s)", result.reason)
        return selected

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def render_tool_descriptions(self, tools: list[Tool]) -> str:
        """Build the ``<tools>`` block injected into the agent prompt.

        Format::

            <tools>
              <tool name="file">
                <description>Read, write, append or list files ...</description>
                <parameters>{"type": "object", "properties": {...}}</parameters>
              </tool>
              ...
            </tools>
        """
        if not tools:
            return ""

        lines = ["<tools>"]
        for tool in tools:
            schema_str = json.dumps(tool.parameters_schema, ensure_ascii=False)
            lines.append(f'  <tool name="{tool.name}">')
            lines.append(f"    <description>{tool.description}</description>")
            lines.append(f"    <parameters>{schema_str}</parameters>")
            lines.append("  </tool>")
        lines.append("</tools>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_tool_calls(self, response: str) -> list[dict]:
        """Extract all ``<tool_call>`` blocks from a model response.

        Returns a list of dicts, each with keys ``tool`` and ``args``.
        Malformed blocks are skipped with a warning.
        """
        pattern = re.compile(
            r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
        )
        calls: list[dict] = []
        for match in pattern.finditer(response):
            raw = match.group(1).strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Malformed tool_call JSON skipped: %r", raw[:120])
                continue
            if not isinstance(data, dict) or "tool" not in data:
                logger.warning("tool_call missing 'tool' key: %r", raw[:120])
                continue
            calls.append({"tool": str(data["tool"]), "args": data.get("args", {})})
        return calls
