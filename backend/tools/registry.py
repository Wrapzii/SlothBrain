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

        tools = registry.get_tools_for_profile("coding")
        prompt_block = registry.render_tool_descriptions(tools)
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register a tool.  Overwrites any existing tool with the same name."""
        if not tool.name:
            raise ValueError("Tool must have a non-empty name")
        self._tools[tool.name] = tool
        logger.debug("Tool registered: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool by name (no-op if not found)."""
        self._tools.pop(name, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Return the tool with *name*, or ``None`` if not registered."""
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_tools_for_profile(self, profile: str) -> list[Tool]:
        """Return the tools visible to *profile*.

        Uses :data:`~backend.tools.profiles.PROFILES` to resolve the set of
        tool names.  If the profile value is ``"*"`` every registered tool is
        returned.  Unknown profile names fall back to the ``minimal`` profile.
        """
        from backend.tools.profiles import PROFILES, DEFAULT_PROFILE

        spec = PROFILES.get(profile)
        if spec is None:
            logger.warning(
                "Unknown tool profile %r — falling back to %r", profile, DEFAULT_PROFILE
            )
            spec = PROFILES.get(DEFAULT_PROFILE, frozenset())

        if spec == "*":
            return self.all_tools()

        return [t for t in self._tools.values() if t.name in spec]  # type: ignore[operator]

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
