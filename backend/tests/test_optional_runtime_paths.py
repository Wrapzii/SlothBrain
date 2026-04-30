from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.main_agent import MainAgent
from backend.config import settings
from backend.core.audit_log import AuditLog
from backend.core.server_manager import ServerManager


@pytest.mark.asyncio
async def test_main_agent_skips_memory_when_disabled() -> None:
    slot_manager = MagicMock()
    slot_manager.send_to_main = AsyncMock(return_value="done")

    agent = MainAgent(
        slot_manager=slot_manager,
        memory=None,
        config=settings,
    )

    response = await agent.process("hi")

    assert response == "done"


@pytest.mark.asyncio
async def test_server_watchdog_returns_cleanly_without_configured_path() -> None:
    snapshot = settings.model_dump()
    try:
        settings.llama_server_path = ""
        manager = ServerManager(config=settings, audit_log=AuditLog())

        await manager._watchdog()

        assert manager.status == "stopped"
    finally:
        for key, value in snapshot.items():
            setattr(settings, key, value)