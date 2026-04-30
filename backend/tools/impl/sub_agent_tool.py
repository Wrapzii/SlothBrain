"""Sub-agent spawning tool.

Allows the MainAgent (or a sufficiently privileged sub-agent) to delegate
work to a new sub-agent via the :class:`~backend.agents.registry.AgentRegistry`.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 2048


class SubAgentTool(Tool):
    """Spawn a sub-agent with a given preset and task, and return its response.

    The sub-agent runs to completion (single-turn process call) and the
    response is returned as the tool output.
    """

    name = "sub_agent"
    description = (
        "Delegate a sub-task to a specialised sub-agent. "
        "Provide the preset ID and a task description; receives the agent's response."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "preset_id": {
                "type": "string",
                "description": "The preset ID to use for the sub-agent.",
            },
            "task": {
                "type": "string",
                "description": "The task or instruction for the sub-agent.",
            },
            "max_tokens": {
                "type": "integer",
                "description": f"Maximum tokens for the response (default: {_DEFAULT_MAX_TOKENS}).",
                "default": _DEFAULT_MAX_TOKENS,
            },
            "context_size": {
                "type": "integer",
                "description": "Override the preset's context window size.",
            },
        },
        "required": ["preset_id", "task"],
    }

    def __init__(self, registry: "AgentRegistry") -> None:
        self._registry = registry

    async def execute(
        self,
        preset_id: str = "",
        task: str = "",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        context_size: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if not preset_id:
            return ToolResult(ok=False, error="'preset_id' argument is required")
        if not task:
            return ToolResult(ok=False, error="'task' argument is required")

        try:
            agent = self._registry.spawn(
                preset_id=preset_id,
                context_size_override=context_size,
                max_tokens_override=max_tokens,
                task_description=task[:100],
            )
        except KeyError as exc:
            return ToolResult(ok=False, error=f"Preset not found: {exc}")
        except Exception as exc:
            return ToolResult(ok=False, error=f"Failed to spawn sub-agent: {exc}")

        try:
            response = await agent.process(task, max_tokens=max_tokens)
            return ToolResult(
                ok=True,
                output={
                    "agent_id": agent.agent_id,
                    "preset_id": agent.preset_id,
                    "response": response,
                },
            )
        except Exception as exc:
            logger.warning("SubAgentTool process failed: %s", exc)
            return ToolResult(ok=False, error=str(exc))
