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

    def spawn(self, preset_id: str) -> SubAgent:
        preset = self._preset_manager.get_preset(preset_id)
        agent_id = str(uuid.uuid4())
        agent = SubAgent(
            agent_id=agent_id,
            preset=preset,
            llama_client=self._llama_client,
            memory=self._memory,
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
