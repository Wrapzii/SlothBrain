"""Sub-agent that runs with a user-defined preset configuration.

Architecture note
-----------------
A SubAgent is **not** a separate llama.cpp process.  It runs on the *same*
llama-server as the MainAgent, but in an independent conversation slot
(slot_id=-1 → "any available slot").  Each slot has its own KV-cache /
context state so conversations are fully isolated.  This is lightweight:
one model binary, multiple parallel conversation contexts, each capped at
the slot's configured context window.

The MainAgent can override a preset's default context_size and max_tokens at
spawn-time so the allocated context matches the actual task complexity.
"""
from __future__ import annotations

import logging

from backend.core.llama_client import LlamaClient
from backend.memory.lancedb_memory import LanceDBMemory

logger = logging.getLogger(__name__)


class SubAgent:
    """A dynamically-spawned agent driven by an agent preset.

    Parameters
    ----------
    agent_id:
        Unique UUID for this running instance.
    preset:
        Preset dict loaded from PresetManager.
    llama_client:
        Shared LlamaClient (same llama-server as MainAgent).
    memory:
        Optional shared memory store.
    context_size_override:
        When the MainAgent delegates a task it can specify how many context
        tokens this sub-agent should consume.  Overrides the preset default.
    max_tokens_override:
        Maximum tokens in the response.  Overrides the preset default.
    task_description:
        Optional one-line description of why this agent was spawned
        (logged for traceability).
    """

    def __init__(
        self,
        agent_id: str,
        preset: dict,
        llama_client: LlamaClient,
        memory: LanceDBMemory | None = None,
        context_size_override: int | None = None,
        max_tokens_override: int | None = None,
        task_description: str = "",
    ) -> None:
        self.agent_id = agent_id
        self.preset_id: str = preset["id"]
        self.name: str = preset["name"]
        self.system_prompt: str = preset["system_prompt"]
        # Runtime allocation – MainAgent can tune these per task
        self.context_size: int = context_size_override or int(preset.get("context_size", 8192))
        self.temperature: float = float(preset.get("temperature", 0.7))
        self.max_tokens: int = max_tokens_override or int(preset.get("max_tokens", 1024))
        self.task_description: str = task_description
        self._client = llama_client
        self._memory = memory
        if task_description:
            logger.info(
                "SubAgent %s (%s) spawned: ctx=%d max_tok=%d task=%r",
                self.agent_id[:8],
                self.name,
                self.context_size,
                self.max_tokens,
                task_description,
            )

    def info(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "preset_id": self.preset_id,
            "name": self.name,
            "context_size": self.context_size,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "task_description": self.task_description,
        }

    async def process(
        self,
        user_input: str,
        context_size: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Process a message.

        The caller (including MainAgent) may pass per-call overrides for
        context_size and max_tokens to further tune resource usage.
        """
        effective_max_tokens = max_tokens or self.max_tokens

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

        # slot_id=-1 → llama.cpp picks any available slot from its pool
        response = await self._client.complete(
            prompt=full_prompt,
            slot_id=-1,
            max_tokens=effective_max_tokens,
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
