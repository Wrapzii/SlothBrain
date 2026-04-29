from __future__ import annotations

from backend.agents.main_agent import MainAgent
from backend.agents.watcher import WatcherAgent


class HandoffManager:
    """Routes single-turn user messages to the appropriate agent.

    The ``WatcherAgent`` processes every message first.  If it detects a
    complex task (via ``should_handoff``), the ``MainAgent`` is invoked with
    the watcher's initial response as additional context.

    TODO: Replace the phrase-based ``should_handoff`` heuristic with a
          structured response field from the watcher (e.g. a JSON field
          ``handoff: true/false``) to avoid false positives on negations
          such as "I do NOT need to hand off this task".
    """

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
