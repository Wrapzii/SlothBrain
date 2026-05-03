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
    assigned_slot_id:
        Concrete llama.cpp slot id allocated for this sub-agent.
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
        assigned_slot_id: int = -1,
        max_tokens_override: int | None = None,
        task_description: str = "",
    ) -> None:
        self.agent_id = agent_id
        self.preset_id: str = preset["id"]
        self.name: str = preset["name"]
        self.system_prompt: str = preset["system_prompt"]
        # Context window is controlled by llama.cpp launch params (-c / -np split).
        # Do not allow runtime per-agent context overrides.
        self.context_size: int = int(preset.get("context_size", 8192))
        self.temperature: float = float(preset.get("temperature", 0.7))
        self.max_tokens: int = max_tokens_override or int(preset.get("max_tokens", 1024))
        self.slot_id: int = int(assigned_slot_id)
        self.task_description: str = task_description
        self._client = llama_client
        self._memory = memory
        if task_description:
            logger.info(
                "SubAgent %s (%s) spawned: slot=%d ctx=%d max_tok=%d task=%r",
                self.agent_id[:8],
                self.name,
                self.slot_id,
                self.context_size,
                self.max_tokens,
                task_description,
            )

    def info(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "preset_id": self.preset_id,
            "name": self.name,
            "slot_id": self.slot_id,
            "context_size": self.context_size,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "task_description": self.task_description,
        }

    async def process(
        self,
        user_input: str,
        max_tokens: int | None = None,
    ) -> str:
        """Process a message.

        The caller (including MainAgent) may pass per-call overrides for
        context_size and max_tokens to further tune resource usage.
        """
        effective_max_tokens = max_tokens or self.max_tokens
        # Rough token budget to keep prompts aligned with context_size.
        # 1 token ~= 4 chars is a common approximation.
        max_prompt_chars = max(self.context_size * 4, 512)

        memory_context = ""
        if self._memory is not None:
            try:
                results = await self._memory.search_advanced(
                    query=user_input,
                    limit=3,
                    metadata_filter={"kind": "turn"},
                    candidate_pool=24,
                )
                if results:
                    snippets = "\n".join(f"- {r['text']}" for r in results)
                    memory_context = f"\n\nRelevant context:\n{snippets}"
            except Exception as exc:
                logger.warning("SubAgent memory search failed: %s", exc.__class__.__name__)

        full_prompt = (
            f"system: {self.system_prompt}"
            f"{memory_context}\n\n"
            f"user: {user_input}\nassistant:"
        )
        if len(full_prompt) > max_prompt_chars:
            # Keep most recent task input; trim older memory context first.
            keep_user = f"user: {user_input}\nassistant:"
            available = max_prompt_chars - len("system: ") - len(self.system_prompt) - len("\n\n") - len(keep_user)
            if available > 0 and memory_context:
                memory_context = memory_context[-available:]
            else:
                memory_context = ""
            full_prompt = (
                f"system: {self.system_prompt}"
                f"{memory_context}\n\n"
                f"{keep_user}"
            )

        # Sub-agents are pinned to an explicitly assigned non-main slot.
        response = await self._client.complete(
            prompt=full_prompt,
            slot_id=self.slot_id,
            max_tokens=effective_max_tokens,
            temperature=self.temperature,
        )

        if self._memory is not None:
            try:
                await self._memory.store(
                    text=f"user: {user_input}\nassistant: {response}",
                    metadata={"agent": self.agent_id, "preset": self.preset_id, "mode": "sub_agent", "kind": "turn"},
                )
            except Exception as exc:
                logger.warning("SubAgent memory store failed: %s", exc.__class__.__name__)

        return response
