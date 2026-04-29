from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from backend.config import AppConfig
from backend.core.slot_manager import SlotManager
from backend.memory.lancedb_memory import LanceDBMemory

if TYPE_CHECKING:
    from backend.agents.registry import AgentRegistry
    from backend.agents.sub_agent import SubAgent

logger = logging.getLogger(__name__)

_PROTECTED_PROMPT_PATH = (
    Path(__file__).parent.parent / "config" / "protected" / "main_system_prompt.txt"
)

_FALLBACK_SYSTEM_PROMPT = (
    "You are a high-performance AI assistant specializing in complex tasks and coding. "
    "Use the provided context and memory to give comprehensive answers."
)


def _load_protected_prompt() -> str:
    """Load the main agent's system prompt from the protected file (read-only)."""
    try:
        return _PROTECTED_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning(
            "Could not read protected system prompt; using fallback."
        )
        return _FALLBACK_SYSTEM_PROMPT


class MainAgent:
    def __init__(
        self,
        slot_manager: SlotManager,
        memory: LanceDBMemory,
        config: AppConfig,
    ) -> None:
        self._slot_manager = slot_manager
        self._memory = memory
        self._config = config
        self.slot_id = config.main_slot
        self.system_prompt = _load_protected_prompt()
        # Injected after construction so we avoid circular imports
        self._registry: AgentRegistry | None = None

    def set_registry(self, registry: "AgentRegistry") -> None:
        """Inject the AgentRegistry so MainAgent can spawn sub-agents."""
        self._registry = registry

    # ------------------------------------------------------------------
    # Sub-agent delegation
    # ------------------------------------------------------------------

    def spawn_sub_agent(
        self,
        preset_id: str,
        task_description: str,
        context_size: int | None = None,
        max_tokens: int | None = None,
    ) -> "SubAgent":
        """Spawn a sub-agent with task-appropriate resource budgets.

        ``context_size`` and ``max_tokens`` override the preset defaults so
        the MainAgent can right-size the allocation for the actual workload.
        If not provided, preset defaults are used.

        Raises RuntimeError if the registry is not set or max_slots exceeded.
        """
        if self._registry is None:
            raise RuntimeError("AgentRegistry not injected into MainAgent")
        return self._registry.spawn(
            preset_id=preset_id,
            context_size_override=context_size,
            max_tokens_override=max_tokens,
            task_description=task_description,
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def process(
        self,
        user_input: str,
        context_from_watcher: str = "",
    ) -> str:
        memory_results: list[dict] = []
        try:
            memory_results = await self._memory.search(user_input, limit=5)
        except Exception as exc:
            logger.warning("MainAgent memory search failed: %s", exc.__class__.__name__)

        memory_context = ""
        if memory_results:
            snippets = "\n".join(f"- {r['text']}" for r in memory_results)
            memory_context = f"\n\nRelevant past context:\n{snippets}"

        watcher_section = ""
        if context_from_watcher:
            watcher_section = f"\n\nWatcher initial assessment:\n{context_from_watcher}"

        full_prompt = (
            f"system: {self.system_prompt}"
            f"{memory_context}"
            f"{watcher_section}\n\n"
            f"user: {user_input}\nassistant:"
        )

        response = await self._slot_manager.send_to_main(
            full_prompt, max_tokens=2048
        )

        try:
            await self._memory.store(
                text=f"user: {user_input}\nassistant: {response}",
                metadata={"agent": "main", "slot": self.slot_id},
            )
        except Exception as exc:
            logger.warning("MainAgent memory store failed: %s", exc.__class__.__name__)

        return response
