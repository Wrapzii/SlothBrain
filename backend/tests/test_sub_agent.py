"""Tests for sub-agent slot assignment and runtime overrides."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.agents.sub_agent import SubAgent
from backend.agents.registry import AgentRegistry
from backend.agents.preset_manager import PresetManager

SAMPLE_PRESET = {
    "id": "test-preset-id",
    "name": "Test Agent",
    "system_prompt": "You are a test agent.",
    "context_size": 8192,
    "temperature": 0.7,
    "max_tokens": 1024,
    "description": "",
}


def _make_llama_client():
    client = MagicMock()
    client.complete = AsyncMock(return_value="test response")
    return client


def test_sub_agent_uses_preset_defaults():
    client = _make_llama_client()
    agent = SubAgent("agent-1", SAMPLE_PRESET, client)
    assert agent.context_size == 8192
    assert agent.max_tokens == 1024


def test_sub_agent_assigned_slot_and_max_tokens_override():
    client = _make_llama_client()
    agent = SubAgent("agent-1", SAMPLE_PRESET, client, assigned_slot_id=2, max_tokens_override=2048)
    assert agent.slot_id == 2
    assert agent.context_size == 8192
    assert agent.max_tokens == 2048


def test_sub_agent_task_description_stored():
    client = _make_llama_client()
    agent = SubAgent("agent-1", SAMPLE_PRESET, client, task_description="summarise document")
    assert agent.task_description == "summarise document"


def test_sub_agent_info_includes_task():
    client = _make_llama_client()
    agent = SubAgent("agent-1", SAMPLE_PRESET, client, task_description="code review")
    info = agent.info()
    assert info["task_description"] == "code review"
    assert info["context_size"] == 8192


@pytest.mark.asyncio
async def test_sub_agent_process_uses_override_max_tokens():
    client = _make_llama_client()
    agent = SubAgent("agent-1", SAMPLE_PRESET, client)
    await agent.process("hello", max_tokens=512)
    client.complete.assert_called_once()
    call_kwargs = client.complete.call_args
    assert call_kwargs.kwargs.get("max_tokens") == 512 or call_kwargs.args[2] == 512


@pytest.mark.asyncio
async def test_registry_spawn_with_overrides():
    pm = MagicMock(spec=PresetManager)
    pm.get_preset.return_value = SAMPLE_PRESET
    client = _make_llama_client()

    registry = AgentRegistry(preset_manager=pm, llama_client=client)
    agent = registry.spawn(
        "test-preset-id",
        assigned_slot_id=3,
        max_tokens_override=4096,
        task_description="process long document",
    )
    assert agent.slot_id == 3
    assert agent.context_size == 8192
    assert agent.max_tokens == 4096
    assert agent.task_description == "process long document"


@pytest.mark.asyncio
async def test_registry_spawn_without_overrides_uses_preset():
    pm = MagicMock(spec=PresetManager)
    pm.get_preset.return_value = SAMPLE_PRESET
    client = _make_llama_client()

    registry = AgentRegistry(preset_manager=pm, llama_client=client)
    agent = registry.spawn("test-preset-id")
    assert agent.context_size == 8192
    assert agent.max_tokens == 1024
    assert agent.task_description == ""
