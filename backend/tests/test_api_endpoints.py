from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import settings
from backend.main import (
    DiscordDMBridge,
    _best_user_facing_response,
    _build_debug_options,
    _should_use_agentic_mode,
    _strip_agentic_prefix,
    _try_handle_simple_slash_command,
)
from backend.agents.main_agent import MainAgent
from backend.tools.base import Tool, ToolResult
from backend.tools.registry import ToolRegistry


_TEST_DOMAIN = "example-tool-target.test"
_TEST_URL = f"https://{_TEST_DOMAIN}"
_TEST_TITLE = "Example Tool Target"
_TEST_DESCRIPTION = "Neutral fixture page used to validate web tool flow."
_TEST_FINAL_SUMMARY = "The model synthesized a neutral web summary from the fetched tool result."
_TEST_DETAILED_SUMMARY = "A more detailed neutral summary synthesized from the fetched tool result."


class _FakeWebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a URL."
    parameters_schema = {"type": "object", "properties": {"url": {"type": "string"}}}

    def __init__(self) -> None:
        self.urls: list[str] = []

    async def execute(self, **kwargs):
        self.urls.append(str(kwargs.get("url") or ""))
        return ToolResult(
            ok=True,
            output=(
                f"<html><head><title>{_TEST_TITLE}</title>"
                f'<meta name="description" content="{_TEST_DESCRIPTION}" />'
                "</head><body>"
                "<h1>Fixture Overview</h1>"
                "<h2>Fixture Capabilities</h2>"
                "<p>This page contains sentinel text for validating fetched page handling.</p>"
                "<p>The content is intentionally unrelated to any real user-provided site.</p>"
                "</body></html>"
            ),
        )


class _FakeSubAgentTool(Tool):
    name = "sub_agent"
    description = "Delegate work to another agent."
    parameters_schema = {
        "type": "object",
        "properties": {
            "preset_id": {"type": "string"},
            "task": {"type": "string"},
        },
        "required": ["preset_id", "task"],
    }

    def __init__(self) -> None:
        self.tasks: list[str] = []

    async def execute(self, **kwargs):
        task = str(kwargs.get("task") or "")
        self.tasks.append(task)
        return ToolResult(
            ok=True,
            output={
                "agent_id": "agent-1",
                "slot_id": 1,
                "handoff_summary": "delegated response summary",
                "response": "delegated response",
            },
        )


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


def test_auto_mode_keeps_tool_intent_direct_without_slash() -> None:
    assert _should_use_agentic_mode(
        "can you try web_fetch on https://bytebrew.cc and tell me what it is?",
        max_steps=1,
        mode="auto",
    ) is False


def test_auto_mode_keeps_simple_prompt_direct() -> None:
    assert _should_use_agentic_mode(
        "hello there",
        max_steps=1,
        mode="auto",
    ) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["/task do thing", "/agentic do thing", "/research topic", "/ralph improve repo"])
async def test_loop_slash_commands_are_not_deterministic(message: str) -> None:
    assert await _try_handle_simple_slash_command(message) is None


@pytest.mark.parametrize("message", ["/task", "/agentic", "/research", "/ralph"])
def test_bare_loop_slash_commands_strip_to_empty_task(message: str) -> None:
    assert _strip_agentic_prefix(message) == ""


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

    assert "<fetch" not in response.lower()
    assert "clean direct response" in response.lower()


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

    assert "<sloth" not in response.lower()
    assert "clean direct response" in response.lower()


@pytest.mark.asyncio
async def test_direct_mode_fetches_site_summary_without_model_hallucination() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        side_effect=[
            f'<tool_call>{{"tool": "web_fetch", "args": {{"url": "{_TEST_URL}"}}}}</tool_call>',
            _TEST_FINAL_SUMMARY,
        ]
    )
    registry = ToolRegistry()
    web_fetch = _FakeWebFetchTool()
    registry.register(web_fetch)
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)
    agent.set_tool_registry(registry)

    response = await agent.process_direct(f"Summarize this site. {_TEST_DOMAIN}")

    assert web_fetch.urls == [_TEST_URL]
    assert response == _TEST_FINAL_SUMMARY
    assert slot_manager.send_to_main.await_count == 2


@pytest.mark.asyncio
async def test_direct_mode_treats_bare_dot_com_as_web_intent() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        side_effect=[
            f'<tool_call>{{"tool": "web_fetch", "args": {{"url": "{_TEST_URL}"}}}}</tool_call>',
            _TEST_FINAL_SUMMARY,
        ]
    )
    registry = ToolRegistry()
    web_fetch = _FakeWebFetchTool()
    registry.register(web_fetch)
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)
    agent.set_tool_registry(registry)

    response = await agent.process_direct(_TEST_DOMAIN)

    assert web_fetch.urls == [_TEST_URL]
    assert response == _TEST_FINAL_SUMMARY
    first_prompt = slot_manager.send_to_main.await_args_list[0].args[0]
    assert "<tool name=\"web_fetch\">" in first_prompt
    assert "use web_fetch unless recent conversation already contains a successful fetched result" in first_prompt


@pytest.mark.asyncio
async def test_direct_mode_preserves_user_apex_domain_when_model_adds_www() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        side_effect=[
            f'<tool_call>{{"tool": "web_fetch", "args": {{"url": "https://www.{_TEST_DOMAIN}"}}}}</tool_call>',
            _TEST_FINAL_SUMMARY,
        ]
    )
    registry = ToolRegistry()
    web_fetch = _FakeWebFetchTool()
    registry.register(web_fetch)
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)
    agent.set_tool_registry(registry)

    response = await agent.process_direct(f"Summarize this site. {_TEST_DOMAIN}")

    assert response == _TEST_FINAL_SUMMARY
    assert web_fetch.urls == [_TEST_URL]


@pytest.mark.asyncio
async def test_direct_mode_failed_web_fetch_does_not_return_speculation() -> None:
    class FailingWebFetchTool(_FakeWebFetchTool):
        async def execute(self, **kwargs):
            self.urls.append(str(kwargs.get("url") or ""))
            return ToolResult(ok=False, error="DNS lookup failed")

    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        side_effect=[
            f'<tool_call>{{"tool": "web_fetch", "args": {{"url": "{_TEST_URL}"}}}}</tool_call>',
            "Based on the name, this company likely manufactures imaginary parts.",
        ]
    )
    registry = ToolRegistry()
    web_fetch = FailingWebFetchTool()
    registry.register(web_fetch)
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)
    agent.set_tool_registry(registry)

    response = await agent.process_direct(f"Summarize this site. {_TEST_DOMAIN}")

    assert web_fetch.urls == [_TEST_URL]
    assert "DNS lookup failed" in response
    assert "imaginary parts" not in response
    assert "do not have page content" in response


@pytest.mark.asyncio
async def test_direct_mode_followup_refetches_previous_site_from_context() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        side_effect=[
            f'<tool_call>{{"tool": "web_fetch", "args": {{"url": "{_TEST_URL}"}}}}</tool_call>',
            _TEST_DETAILED_SUMMARY,
        ]
    )
    registry = ToolRegistry()
    web_fetch = _FakeWebFetchTool()
    registry.register(web_fetch)
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)
    agent.set_tool_registry(registry)

    response = await agent.process_direct(
        "better summary though? more indepth?",
        conversation_context=[
            f"User: Summarize this site. {_TEST_DOMAIN}",
            f"SlothBrain: Page title: {_TEST_TITLE}.",
        ],
    )

    assert web_fetch.urls == [_TEST_URL]
    assert response == _TEST_DETAILED_SUMMARY
    assert slot_manager.send_to_main.await_count == 2


@pytest.mark.asyncio
async def test_direct_mode_how_did_you_get_that_explains_source_not_refetch() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        return_value="I used the previous tool result from this chat."
    )
    registry = ToolRegistry()
    web_fetch = _FakeWebFetchTool()
    registry.register(web_fetch)
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)
    agent.set_tool_registry(registry)

    response = await agent.process_direct(
        "how did you get that information?",
        conversation_context=[
            f"User: Summarize this site. {_TEST_DOMAIN}",
            f"SlothBrain: Page title: {_TEST_TITLE}.",
        ],
    )

    assert web_fetch.urls == []
    assert response == "I used the previous tool result from this chat."
    prompt = slot_manager.send_to_main.call_args.args[0]
    assert "Recent conversation:" in prompt
    assert "<tool name=\"web_fetch\">" not in prompt


@pytest.mark.asyncio
async def test_direct_mode_records_tool_provenance_for_followup_context() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        side_effect=[
            f'<tool_call>{{"tool": "web_fetch", "args": {{"url": "{_TEST_URL}"}}}}</tool_call>',
            _TEST_FINAL_SUMMARY,
            "I used the prior web_fetch tool result.",
        ]
    )
    registry = ToolRegistry()
    web_fetch = _FakeWebFetchTool()
    registry.register(web_fetch)
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)
    agent.set_tool_registry(registry)

    await agent.process_direct(f"Summarize this site. {_TEST_DOMAIN}")
    response = await agent.process_direct("how did you get that information?")

    assert response == "I used the prior web_fetch tool result."
    assert web_fetch.urls == [_TEST_URL]
    followup_prompt = slot_manager.send_to_main.await_args_list[-1].args[0]
    assert "Tool usage this turn:" in followup_prompt
    assert "tool=web_fetch" in followup_prompt
    assert _TEST_URL in followup_prompt
    assert "<tool name=\"web_fetch\">" not in followup_prompt


@pytest.mark.asyncio
async def test_direct_mode_includes_recent_conversation_context_in_prompt() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(return_value="ok")
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)

    await agent.process_direct(
        "what were we talking about?",
        conversation_context=[
            f"User: Summarize this site. {_TEST_DOMAIN}",
            f"SlothBrain: Page title: {_TEST_TITLE}.",
        ],
    )

    prompt = slot_manager.send_to_main.call_args.args[0]
    assert "Recent conversation:" in prompt
    assert _TEST_DOMAIN in prompt
    assert _TEST_TITLE in prompt


@pytest.mark.asyncio
async def test_direct_mode_reuses_rolling_context_between_turns() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(side_effect=["stored first turn", "ok"])
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)

    await agent.process_direct("remember that the context token is alpha")
    await agent.process_direct("what did I just ask you to remember?")

    second_prompt = slot_manager.send_to_main.call_args.args[0]
    assert "Rolling conversation summary:" in second_prompt
    assert "context token is alpha" in second_prompt
    assert "stored first turn" in second_prompt


@pytest.mark.asyncio
async def test_direct_web_fetch_tool_flow_stores_rag_memory() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        side_effect=[
            f'<tool_call>{{"tool": "web_fetch", "args": {{"url": "{_TEST_URL}"}}}}</tool_call>',
            _TEST_FINAL_SUMMARY,
        ]
    )
    registry = ToolRegistry()
    registry.register(_FakeWebFetchTool())
    memory = SimpleNamespace(store=AsyncMock())
    agent = MainAgent(slot_manager=slot_manager, memory=memory, config=settings)
    agent.set_tool_registry(registry)

    response = await agent.process_direct(f"Summarize this site. {_TEST_DOMAIN}")
    await asyncio.sleep(0)

    memory.store.assert_awaited_once()
    stored = memory.store.call_args.kwargs
    assert _TEST_DOMAIN in stored["text"]
    assert _TEST_FINAL_SUMMARY in stored["text"]
    assert "tool=web_fetch" in stored["text"]
    assert _TEST_URL in stored["text"]
    assert stored["metadata"]["mode"] == "direct"
    assert response in stored["text"]


@pytest.mark.asyncio
async def test_discord_context_records_user_turn_before_background_task() -> None:
    bridge = DiscordDMBridge(
        main_agent=SimpleNamespace(),
        config=SimpleNamespace(),
        registry=SimpleNamespace(get=lambda name: None),
    )

    async def fake_run_bg_task(**kwargs):
        user_key = kwargs["user_key"]
        assert list(bridge._dm_context[user_key])[-1] == "User: hello context"
        kwargs["coro"].close()

    bridge._run_bg_task = fake_run_bg_task

    await bridge._handle_message(
        {"id": "1", "author": "Wrapzii", "author_id": "42", "content": "hello context"}
    )

    assert list(bridge._dm_context["42"]) == ["User: hello context"]


@pytest.mark.asyncio
async def test_direct_mode_does_not_offer_sub_agent_without_delegation_intent() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(return_value="ok")
    registry = ToolRegistry()
    registry.register(_FakeSubAgentTool())
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)
    agent.set_tool_registry(registry)

    await agent.process_direct("use a tool to answer what we were discussing")

    prompt = slot_manager.send_to_main.call_args.args[0]
    assert "<tool name=\"sub_agent\">" not in prompt


@pytest.mark.asyncio
async def test_direct_mode_explicit_sub_agent_handoff_includes_context() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        side_effect=[
            '<tool_call>{"tool": "sub_agent", "args": {"preset_id": "research", "task": "summarize it"}}</tool_call>',
            "delegated final answer",
        ]
    )
    sub_agent = _FakeSubAgentTool()
    registry = ToolRegistry()
    registry.register(sub_agent)
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)
    agent.set_tool_registry(registry)

    response = await agent.process_direct(
        "delegate that to a sub-agent",
        conversation_context=[
            f"User: Summarize this site. {_TEST_DOMAIN}",
            f"SlothBrain: Page title: {_TEST_TITLE}.",
        ],
    )

    assert response == "delegated final answer"
    assert sub_agent.tasks
    assert "Context handed off from the main agent:" in sub_agent.tasks[0]
    assert _TEST_DOMAIN in sub_agent.tasks[0]
    assert _TEST_TITLE in sub_agent.tasks[0]


@pytest.mark.asyncio
async def test_direct_mode_sub_agent_return_has_deterministic_handoff_summary() -> None:
    slot_manager = AsyncMock()
    slot_manager.send_to_main = AsyncMock(
        side_effect=[
            '<tool_call>{"tool": "sub_agent", "args": {"preset_id": "research", "task": "summarize it"}}</tool_call>',
            "<tool_result>raw protocol leaked</tool_result>",
        ]
    )
    registry = ToolRegistry()
    registry.register(_FakeSubAgentTool())
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=settings)
    agent.set_tool_registry(registry)

    response = await agent.process_direct("delegate that to a sub-agent")

    assert "Sub-agent handoff summary" in response
    assert "delegated response summary" in response
    assert "slot=1" in response


def test_best_user_facing_response_salvages_html() -> None:
    text = (
        f"<html><head><title>{_TEST_TITLE}</title>"
        f'<meta name="description" content="{_TEST_DESCRIPTION}" />'
        f"</head><body><h1>{_TEST_TITLE}</h1></body></html>"
    )

    response = _best_user_facing_response(text)

    assert _TEST_TITLE.lower() in response.lower()
    assert _TEST_DESCRIPTION.lower() in response.lower()


def test_extract_agentic_response_skips_synthesis_prompt_parroting() -> None:
    result = {
        "steps": [
            {"result": "Intel shares fell after the company cut guidance and announced new restructuring measures."},
            {"result": "Based on the tool result(s) above, provide the actual findings, data, or answer. Do NOT make up information. Extract and cite what the tool returned."},
        ],
        "summary": "Task execution complete.",
    }

    response = DiscordDMBridge._extract_agentic_response(result)

    assert "intel shares fell" in response.lower()
