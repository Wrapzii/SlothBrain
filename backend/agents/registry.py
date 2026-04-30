"""Registry for dynamically-spawned sub-agent instances."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from backend.agents.preset_manager import PresetManager
from backend.agents.sub_agent import SubAgent

if TYPE_CHECKING:
    from backend.core.llama_client import LlamaClient
    from backend.memory.lancedb_memory import LanceDBMemory


class AgentRegistry:
    def __init__(
        self,
        preset_manager: PresetManager,
        llama_client: "LlamaClient",
        memory: "LanceDBMemory | None" = None,
    ) -> None:
        self._preset_manager = preset_manager
        self._llama_client = llama_client
        self._memory = memory
        self._agents: dict[str, SubAgent] = {}

    def spawn(
        self,
        preset_id: str,
        context_size_override: int | None = None,
        max_tokens_override: int | None = None,
        task_description: str = "",
        tool_profile: str = "minimal",
    ) -> SubAgent:
        """Spawn a sub-agent from a preset.

        The MainAgent (or the API) can pass ``context_size_override`` and
        ``max_tokens_override`` to right-size the agent for the actual task
        instead of always using the preset's static defaults.

        ``tool_profile`` sets which tools the sub-agent can call.  Presets
        may override this via their ``tool_profile`` field; the parameter is
        only used when the preset does not specify a profile.
        """
        preset = self._preset_manager.get_preset(preset_id)
        agent_id = str(uuid.uuid4())
        agent = SubAgent(
            agent_id=agent_id,
            preset=preset,
            llama_client=self._llama_client,
            memory=self._memory,
            context_size_override=context_size_override,
            max_tokens_override=max_tokens_override,
            task_description=task_description,
            tool_profile=tool_profile,
        )
        self._agents[agent_id] = agent
        return agent

    def get(self, agent_id: str) -> SubAgent:
        if agent_id not in self._agents:
            raise KeyError(f"No running agent with id {agent_id!r}")
        return self._agents[agent_id]

    def list_agents(self) -> list[dict]:
        return [a.info() for a in self._agents.values()]

    def destroy(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise KeyError(f"No running agent with id {agent_id!r}")
        del self._agents[agent_id]

    def destroy_all(self) -> None:
        self._agents.clear()
