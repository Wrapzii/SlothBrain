"""Session management tool.

Provides session lifecycle operations: list, inspect, and terminate sessions
tracked in the running agent registry.  Sessions are identified by the
sub-agent's ``agent_id``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.agents.registry import AgentRegistry


class SessionTool(Tool):
    """Manage agent sessions: list, inspect, or terminate a session.

    Actions
    -------
    * ``list``    — list all active sessions.
    * ``inspect`` — get detailed info about a specific session by agent_id.
    * ``terminate`` — destroy a session by agent_id.
    """

    name = "session"
    description = (
        "Manage agent sessions: list all active sessions, inspect a session by ID, "
        "or terminate a session."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "inspect", "terminate"],
                "description": "Session operation to perform.",
            },
            "agent_id": {
                "type": "string",
                "description": "Agent/session ID (required for 'inspect' and 'terminate').",
            },
        },
        "required": ["action"],
    }

    def __init__(self, registry: "AgentRegistry") -> None:
        self._registry = registry

    async def execute(
        self,
        action: str = "",
        agent_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        if action == "list":
            sessions = self._registry.list_agents()
            return ToolResult(ok=True, output={"sessions": sessions, "count": len(sessions)})

        if action == "inspect":
            if not agent_id:
                return ToolResult(ok=False, error="'agent_id' is required for 'inspect'")
            try:
                agent = self._registry.get(agent_id)
                return ToolResult(ok=True, output=agent.info())
            except KeyError:
                return ToolResult(ok=False, error=f"No session with id {agent_id!r}")

        if action == "terminate":
            if not agent_id:
                return ToolResult(ok=False, error="'agent_id' is required for 'terminate'")
            try:
                self._registry.destroy(agent_id)
                return ToolResult(ok=True, output={"terminated": agent_id})
            except KeyError:
                return ToolResult(ok=False, error=f"No session with id {agent_id!r}")

        return ToolResult(ok=False, error=f"Unknown action: {action!r}")
