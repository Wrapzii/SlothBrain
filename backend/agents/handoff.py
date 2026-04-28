from __future__ import annotations

from backend.agents.main_agent import MainAgent
from backend.agents.watcher import WatcherAgent


class HandoffManager:
    def __init__(self, watcher: WatcherAgent, main_agent: MainAgent) -> None:
        self._watcher = watcher
        self._main_agent = main_agent

    async def route(self, user_input: str) -> dict:
        watcher_response = await self._watcher.process(user_input)
        should_hand_off = await self._watcher.should_handoff(watcher_response)

        if should_hand_off:
            main_response = await self._main_agent.process(
                user_input,
                context_from_watcher=watcher_response,
            )
            return {
                "agent": "main",
                "response": main_response,
                "handoff": True,
            }

        return {
            "agent": "watcher",
            "response": watcher_response,
            "handoff": False,
        }
