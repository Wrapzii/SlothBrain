"""Sub-agent that runs with a user-defined preset configuration."""
from __future__ import annotations

import logging

from backend.core.llama_client import LlamaClient
from backend.memory.lancedb_memory import LanceDBMemory

logger = logging.getLogger(__name__)


class SubAgent:
    """A dynamically-spawned agent driven by an agent preset."""

    def __init__(
        self,
        agent_id: str,
        preset: dict,
        llama_client: LlamaClient,
        memory: LanceDBMemory | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.preset_id: str = preset["id"]
        self.name: str = preset["name"]
        self.system_prompt: str = preset["system_prompt"]
        self.context_size: int = int(preset.get("context_size", 8192))
        self.temperature: float = float(preset.get("temperature", 0.7))
        self.max_tokens: int = int(preset.get("max_tokens", 1024))
        self._client = llama_client
        self._memory = memory

    def info(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "preset_id": self.preset_id,
            "name": self.name,
            "context_size": self.context_size,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    async def process(self, user_input: str) -> str:
        memory_context = ""
        if self._memory is not None:
            try:
                results = await self._memory.search(user_input, limit=3)
                if results:
                    snippets = "\n".join(f"- {r['text']}" for r in results)
                    memory_context = f"\n\nRelevant context:\n{snippets}"
            except Exception:
                pass

        full_prompt = (
            f"system: {self.system_prompt}"
            f"{memory_context}\n\n"
            f"user: {user_input}\nassistant:"
        )

        # Sub-agents use slot_id=-1 which llama.cpp treats as "any available slot"
        response = await self._client.complete(
            prompt=full_prompt,
            slot_id=-1,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        if self._memory is not None:
            try:
                await self._memory.store(
                    text=f"user: {user_input}\nassistant: {response}",
                    metadata={"agent": self.agent_id, "preset": self.preset_id},
                )
            except Exception:
                pass

        return response
