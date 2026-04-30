from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.core.slot_manager import SlotManager


@pytest_asyncio.fixture
async def mock_llama_client():
    client = MagicMock()
    client.get_slots = AsyncMock(return_value=[{"id": 0, "state": "idle"}, {"id": 1, "state": "idle"}])
    client.complete = AsyncMock(return_value="Mock response from llama")
    return client


@pytest_asyncio.fixture
async def slot_manager(mock_llama_client):
    sm = SlotManager(llama_client=mock_llama_client)
    await sm.assign_main(1)
    return sm


@pytest.mark.asyncio
async def test_assign_main(mock_llama_client):
    sm = SlotManager(llama_client=mock_llama_client)
    await sm.assign_main(1)
    assert sm._main_slot == 1
    assert 1 in sm._histories


@pytest.mark.asyncio
async def test_get_slot_info(slot_manager, mock_llama_client):
    info = await slot_manager.get_slot_info()
    assert info["main"] == 1
    assert isinstance(info["slots"], list)
    mock_llama_client.get_slots.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_to_main(slot_manager, mock_llama_client):
    response = await slot_manager.send_to_main("Solve this problem", max_tokens=512)
    assert response == "Mock response from llama"
    mock_llama_client.complete.assert_awaited_once_with(
        prompt="Solve this problem",
        slot_id=1,
        max_tokens=512,
        stop=["\nuser:", "\nassistant:", "\nsystem:", "\n# Response", "<think>"],
    )


@pytest.mark.asyncio
async def test_get_slot_info_uses_cache(slot_manager, mock_llama_client):
    slot_manager.set_slot_info_cache_ttl(5.0)

    info1 = await slot_manager.get_slot_info()
    info2 = await slot_manager.get_slot_info()

    assert info1["slots"] == info2["slots"]
    mock_llama_client.get_slots.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_to_main_no_slot(mock_llama_client):
    sm = SlotManager(llama_client=mock_llama_client)
    with pytest.raises(RuntimeError, match="Main slot not assigned"):
        await sm.send_to_main("test")


@pytest.mark.asyncio
async def test_history_recorded(slot_manager):
    await slot_manager.send_to_main("ping")
    history = slot_manager.get_history(1)
    assert len(history) == 1
    assert history[0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_send_to_main_strips_response_heading(slot_manager, mock_llama_client):
    mock_llama_client.complete = AsyncMock(
        return_value="# Response\n\nA clean final answer\nuser: ignored"
    )

    response = await slot_manager.send_to_main("Hello")

    assert response == "A clean final answer"
