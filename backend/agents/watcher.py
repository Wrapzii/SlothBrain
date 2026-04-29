from __future__ import annotations

import logging

from backend.config import AppConfig
from backend.core.slot_manager import SlotManager
from backend.memory.lancedb_memory import LanceDBMemory
from backend.memory.rolling_context import RollingContext

_HANDOFF_PHRASES = frozenset(
    ["hand off", "handoff", "hand-off", "complex task", "main agent"]
)

SYSTEM_PROMPT = (
    "You are a lightweight always-on assistant. Monitor activity and decide when to "
    "hand off complex tasks to the main agent. Keep responses concise."
)
logger = logging.getLogger(__name__)


class WatcherAgent:
    def __init__(
        self,
        slot_manager: SlotManager,
        rolling_context: RollingContext,
        memory: LanceDBMemory,
        config: AppConfig,
    ) -> None:
        self._slot_manager = slot_manager
        self._rolling_context = rolling_context
        self._memory = memory
        self._config = config
        self.slot_id = config.watcher_slot
        self.system_prompt = SYSTEM_PROMPT

    async def process(self, user_input: str) -> str:
        await self._rolling_context.add_message("user", user_input)
        context = self._rolling_context.get_context_prompt()
        full_prompt = f"system: {self.system_prompt}\n{context}assistant:"
        response = await self._slot_manager.send_to_watcher(
            full_prompt, max_tokens=256
        )
        await self._rolling_context.add_message("assistant", response)
        try:
            await self._memory.store(
                text=f"user: {user_input}\nassistant: {response}",
                metadata={"agent": "watcher", "slot": self.slot_id},
            )
        except Exception as exc:
            logger.warning("WatcherAgent memory store failed: %s", exc.__class__.__name__)
        return response

    async def should_handoff(self, response: str) -> bool:
        lower = response.lower()
        return any(phrase in lower for phrase in _HANDOFF_PHRASES)
