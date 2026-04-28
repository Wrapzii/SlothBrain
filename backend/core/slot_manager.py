from __future__ import annotations

from backend.core.llama_client import LlamaClient


class SlotManager:
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
        )
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
        )
        self._histories.setdefault(self._main_slot, []).append(
            {"role": "assistant", "content": response}
        )
        return response

    def get_history(self, slot_id: int) -> list[dict]:
        return self._histories.get(slot_id, [])
