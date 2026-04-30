from __future__ import annotations

import re

from backend.core.llama_client import LlamaClient


_STOP_SEQUENCES = ["\nuser:", "\nassistant:", "\nsystem:", "\n# Response", "<think>"]
_RESPONSE_PREFIX_RE = re.compile(
    r"(?is)^(?:\s*(?:system|user|assistant):.*?)+\s*(?:#\s*Response\s*)?"
)



def _sanitize_response(response: str) -> str:
    cleaned = (response or "").strip()
    if not cleaned:
        return ""

    cleaned = _RESPONSE_PREFIX_RE.sub("", cleaned).strip()
    if cleaned.lower().startswith("# response"):
        cleaned = cleaned[len("# response") :].strip()

    for marker in ("\nuser:", "\nassistant:", "\nsystem:", "\n# Response"):
        index = cleaned.find(marker)
        if index != -1:
            cleaned = cleaned[:index].rstrip()

    return cleaned


class SlotManager:
    """Manages named inference slot assignments and per-slot conversation history.

    Each slot corresponds to a llama.cpp KV-cache context.  The ``SlotManager``
    wraps ``LlamaClient`` with:
    - Named slot assignment (``watcher_slot``, ``main_slot``).
    - Per-slot message history for UI display purposes.
    - Response sanitisation: strips leaked role prefixes (``system:``,
      ``user:``, ``assistant:``) and truncates at stop sequences.

    TODO: Clear the per-slot history when a slot is reassigned so stale
          messages from a previous session do not appear in the UI.
    """

    def __init__(self, llama_client: LlamaClient) -> None:
        self._client = llama_client
        self._watcher_slot: int | None = None
        self._main_slot: int | None = None
        self._histories: dict[int, list[dict]] = {}

    async def assign_watcher(self, slot_id: int) -> None:
        self._watcher_slot = slot_id
        if slot_id not in self._histories:
            self._histories[slot_id] = []

    async def assign_main(self, slot_id: int) -> None:
        self._main_slot = slot_id
        if slot_id not in self._histories:
            self._histories[slot_id] = []

    async def get_slot_info(self) -> dict:
        slots = await self._client.get_slots()
        return {
            "watcher": self._watcher_slot,
            "main": self._main_slot,
            "slots": slots,
        }

    async def send_to_watcher(self, prompt: str, max_tokens: int = 256) -> str:
        if self._watcher_slot is None:
            raise RuntimeError("Watcher slot not assigned")
        response = await self._client.complete(
            prompt=prompt,
            slot_id=self._watcher_slot,
            max_tokens=max_tokens,
            stop=_STOP_SEQUENCES,
        )
        response = _sanitize_response(response)
        self._histories.setdefault(self._watcher_slot, []).append(
            {"role": "assistant", "content": response}
        )
        return response

    async def send_to_main(self, prompt: str, max_tokens: int = 2048) -> str:
        if self._main_slot is None:
            raise RuntimeError("Main slot not assigned")
        response = await self._client.complete(
            prompt=prompt,
            slot_id=self._main_slot,
            max_tokens=max_tokens,
            stop=_STOP_SEQUENCES,
        )
        response = _sanitize_response(response)
        self._histories.setdefault(self._main_slot, []).append(
            {"role": "assistant", "content": response}
        )
        return response

    def get_history(self, slot_id: int) -> list[dict]:
        return self._histories.get(slot_id, [])
