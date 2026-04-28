from __future__ import annotations

from backend.config import AppConfig
from backend.core.slot_manager import SlotManager
from backend.memory.lancedb_memory import LanceDBMemory

SYSTEM_PROMPT = (
    "You are a high-performance AI assistant specializing in complex tasks and coding. "
    "Use the provided context and memory to give comprehensive answers."
)


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
        self.system_prompt = SYSTEM_PROMPT

    async def process(
        self,
        user_input: str,
        context_from_watcher: str = "",
    ) -> str:
        memory_results: list[dict] = []
        try:
            memory_results = await self._memory.search(user_input, limit=5)
        except Exception:
            pass

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
        except Exception:
            pass

        return response
