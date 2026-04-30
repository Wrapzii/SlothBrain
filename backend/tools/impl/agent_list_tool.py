"""Agent list tool — list all running sub-agents from the AgentRegistry."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.agents.registry import AgentRegistry


class AgentListTool(Tool):
    """List all currently registered sub-agents and their status."""

    name = "agent_list"
    description = (
        "List all currently active sub-agents with their agent ID, preset, "
        "and task description."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, registry: "AgentRegistry") -> None:
        self._registry = registry

    async def execute(self, **kwargs: Any) -> ToolResult:
        agents = self._registry.list_agents()
        return ToolResult(ok=True, output={"agents": agents, "count": len(agents)})
