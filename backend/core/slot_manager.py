from __future__ import annotations

import asyncio
import re
import time

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
    - Named slot assignment (``main_slot``).
    - Per-slot message history for UI display purposes.
    - Response sanitisation: strips leaked role prefixes (``system:``,
      ``user:``, ``assistant:``) and truncates at stop sequences.

    TODO: Clear the per-slot history when a slot is reassigned so stale
          messages from a previous session do not appear in the UI.
    """

    def __init__(self, llama_client: LlamaClient) -> None:
        self._client = llama_client
        self._main_slot: int | None = None
        self._histories: dict[int, list[dict]] = {}
        self._slot_info_cache: dict | None = None
        self._slot_info_cache_ts: float = 0.0
        self._slot_info_cache_ttl_s: float = 2.0
        self._slot_info_lock = asyncio.Lock()

    def set_slot_info_cache_ttl(self, ttl_seconds: float) -> None:
        self._slot_info_cache_ttl_s = max(0.0, float(ttl_seconds))
        # Drop stale cache immediately when policy changes.
        self._slot_info_cache = None
        self._slot_info_cache_ts = 0.0

    async def assign_main(self, slot_id: int) -> None:
        self._main_slot = slot_id
        if slot_id not in self._histories:
            self._histories[slot_id] = []

    async def get_slot_info(self) -> dict:
        now = time.monotonic()
        if (
            self._slot_info_cache_ttl_s > 0
            and self._slot_info_cache is not None
            and (now - self._slot_info_cache_ts) <= self._slot_info_cache_ttl_s
        ):
            return dict(self._slot_info_cache)

        async with self._slot_info_lock:
            now = time.monotonic()
            if (
                self._slot_info_cache_ttl_s > 0
                and self._slot_info_cache is not None
                and (now - self._slot_info_cache_ts) <= self._slot_info_cache_ttl_s
            ):
                return dict(self._slot_info_cache)

            slots = await self._client.get_slots()
            data = {
                "main": self._main_slot,
                "slots": slots,
            }
            if self._slot_info_cache_ttl_s > 0:
                self._slot_info_cache = data
                self._slot_info_cache_ts = now
            return dict(data)

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

        # Some thinking/reasoning model outputs can be reduced to empty text
        # after stop-sequence + sanitization trimming. Retry once without
        # explicit stop sequences to recover a user-visible answer.
        if not response:
            retry = await self._client.complete(
                prompt=prompt,
                slot_id=self._main_slot,
                max_tokens=max_tokens,
            )
            response = _sanitize_response(retry) or (retry or "").strip()

        self._histories.setdefault(self._main_slot, []).append(
            {"role": "assistant", "content": response}
        )
        return response

    def get_history(self, slot_id: int) -> list[dict]:
        return self._histories.get(slot_id, [])
