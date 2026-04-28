from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.config import AppConfig
from backend.core.resource_manager import ResourceManager


@pytest_asyncio.fixture
def config():
    return AppConfig(
        mode="idle",
        idle_kv_quant="q4",
        active_kv_quant="q8",
        vram_threshold_mb=2048,
    )


@pytest_asyncio.fixture
def mock_llama_client():
    client = MagicMock()
    client.get_metrics = AsyncMock(return_value="# metrics\nvram_used 1024\n")
    return client


@pytest_asyncio.fixture
def resource_manager(config, mock_llama_client):
    return ResourceManager(config=config, llama_client=mock_llama_client)


@pytest.mark.asyncio
async def test_initial_mode(resource_manager):
    assert resource_manager.mode == "idle"


@pytest.mark.asyncio
async def test_set_mode_active(resource_manager):
    await resource_manager.set_mode("active")
    assert resource_manager.mode == "active"


@pytest.mark.asyncio
async def test_set_mode_idle(resource_manager):
    await resource_manager.set_mode("active")
    await resource_manager.set_mode("idle")
    assert resource_manager.mode == "idle"


@pytest.mark.asyncio
async def test_set_mode_invalid(resource_manager):
    with pytest.raises(ValueError, match="Invalid mode"):
        await resource_manager.set_mode("turbo")


@pytest.mark.asyncio
async def test_get_system_stats(resource_manager):
    stats = await resource_manager.get_system_stats()
    assert "cpu_percent" in stats
    assert "ram_used_mb" in stats
    assert "ram_total_mb" in stats
    assert "mode" in stats
    assert stats["mode"] == "idle"
    assert stats["ram_total_mb"] > 0


@pytest.mark.asyncio
async def test_get_kv_quant_idle(resource_manager):
    quant = await resource_manager.get_kv_quant()
    assert quant == "q4"


@pytest.mark.asyncio
async def test_get_kv_quant_active(resource_manager):
    await resource_manager.set_mode("active")
    quant = await resource_manager.get_kv_quant()
    assert quant == "q8"


@pytest.mark.asyncio
async def test_auto_adjust_stays_idle(resource_manager, config):
    # With threshold very high, should remain idle
    config.vram_threshold_mb = 999_999
    await resource_manager.auto_adjust()
    assert resource_manager.mode == "idle"


@pytest.mark.asyncio
async def test_auto_adjust_triggers_idle(resource_manager, config):
    config.vram_threshold_mb = 1  # Set threshold very low
    await resource_manager.set_mode("active")
    await resource_manager.auto_adjust()
    assert resource_manager.mode == "idle"
