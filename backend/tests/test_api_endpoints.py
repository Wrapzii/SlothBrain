from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import settings
from backend.main import _build_debug_options, _should_use_agentic_mode
from backend.agents.main_agent import MainAgent


def _restore_settings(snapshot: dict) -> None:
    for key, value in snapshot.items():
        setattr(settings, key, value)


def test_chat_returns_503_when_backend_unavailable() -> None:
    snapshot = settings.model_dump()
    try:
        settings.llama_host = "127.0.0.1"
        settings.llama_port = 65534
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/chat", json={"message": "hello", "agent": "auto"})
            assert resp.status_code == 503
    finally:
        _restore_settings(snapshot)


def test_restart_approval_does_not_500_with_missing_server_path() -> None:
    snapshot = settings.model_dump()
    try:
        settings.require_approval_server_restart = True
        settings.llama_server_path = ""

        with TestClient(app, raise_server_exceptions=False) as client:
            queued = client.post("/api/server/restart")
            assert queued.status_code == 200
            approval_id = queued.json()["pending_approval"]["id"]

            approved = client.post(f"/api/approvals/{approval_id}/approve")
            assert approved.status_code == 200
            body = approved.json()
            assert body["approved"] is True
            assert body["action"] == "server_restart"
            assert "error" in body
    finally:
        _restore_settings(snapshot)


def test_emergency_stop_queues_when_approval_required() -> None:
    snapshot = settings.model_dump()
    try:
        settings.require_approval_emergency_stop = True
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/emergency-stop")
            assert resp.status_code == 200
            body = resp.json()
            assert "pending_approval" in body
            assert body["pending_approval"]["action"] == "emergency_stop"
    finally:
        _restore_settings(snapshot)


def test_set_mode_invalid_returns_400() -> None:
    snapshot = settings.model_dump()
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/mode", json={"mode": "turbo"})
            assert resp.status_code == 400
            assert "Invalid mode" in resp.json()["detail"]
    finally:
        _restore_settings(snapshot)


def test_api_key_required_when_configured() -> None:
    snapshot = settings.model_dump()
    try:
        settings.api_key = "secret"
        with TestClient(app, raise_server_exceptions=False) as client:
            denied = client.get("/api/status")
            assert denied.status_code == 401

            allowed = client.get("/api/status", headers={"x-api-key": "secret"})
            assert allowed.status_code == 200
    finally:
        _restore_settings(snapshot)


def test_auto_mode_routes_tool_intent_to_agentic() -> None:
    assert _should_use_agentic_mode(
        "can you try web_fetch on https://bytebrew.cc and tell me what it is?",
        max_steps=1,
        mode="auto",
    ) is True


def test_auto_mode_keeps_simple_prompt_direct() -> None:
    assert _should_use_agentic_mode(
        "hello there",
        max_steps=1,
        mode="auto",
    ) is False


def test_build_debug_options_uses_defaults_when_missing() -> None:
    snapshot = settings.model_dump()
    try:
        settings.debug_loop_enabled = True
        settings.debug_loop_tool_calls_enabled = False
        settings.debug_loop_allowed_tools = ["web_fetch"]

        options = _build_debug_options(None)
        assert options.enabled is True
        assert options.tool_calls_enabled is False
        assert options.allowed_tools == ["web_fetch"]
    finally:
        _restore_settings(snapshot)


def test_build_debug_options_allows_request_overrides() -> None:
    snapshot = settings.model_dump()
    try:
        settings.debug_loop_enabled = False
        settings.debug_loop_tool_calls_enabled = True

        options = _build_debug_options(None)
        assert options.enabled is False

        from backend.main import AgenticDebugRequest

        override = AgenticDebugRequest(
            enabled=True,
            tool_calls_enabled=False,
            allowed_tools=["file", "patch"],
        )
        merged = _build_debug_options(override)
        assert merged.enabled is True
        assert merged.tool_calls_enabled is False
        assert merged.allowed_tools == ["file", "patch"]
    finally:
        _restore_settings(snapshot)


@pytest.mark.asyncio
async def test_direct_mode_sanitizes_pseudo_tool_markup() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        return_value=(
            "<sweep>thinking</sweep>\n"
            "<fetch>url: https://example.com</fetch>\n"
            "<fetch_result>...</fetch_result>"
        )
    )
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)

    response = await agent.process_direct("can you use web_fetch?")

    assert "cannot execute tools in direct mode" in response.lower()


@pytest.mark.asyncio
async def test_direct_mode_sanitizes_think_sloth_markup() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        return_value=(
            "<sloth>\nThinking Process:\n"
            "1. Simulated Content\n"
            "2. Self-Correction/Verification\n"
            "</think>"
        )
    )
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)

    response = await agent.process_direct("can you try web_fetch on https://bytebrew.cc and tell me what it is?")

    assert "cannot execute tools in direct mode" in response.lower()
