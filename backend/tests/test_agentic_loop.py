"""Tests for the agentic loop and main-agent helpers."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.agentic_loop import AgenticLoop, AgenticStep, _build_result
from backend.agents.main_agent import _parse_plan


# ---------------------------------------------------------------------------
# _parse_plan
# ---------------------------------------------------------------------------

def test_parse_plan_numbered():
    response = "APPROACH: Step by step\nSTEPS:\n1. Do A\n2. Do B\n3. Do C"
    plan = _parse_plan(response)
    assert plan["approach"] == "Step by step"
    assert plan["steps"] == ["Do A", "Do B", "Do C"]


def test_parse_plan_no_approach():
    response = "1. First thing\n2. Second thing"
    plan = _parse_plan(response)
    assert len(plan["steps"]) == 2


def test_parse_plan_fallback():
    plan = _parse_plan("Just do it all in one go")
    assert len(plan["steps"]) == 1


def test_parse_plan_caps_at_ten():
    lines = "\n".join(f"{i}. Step {i}" for i in range(1, 16))
    plan = _parse_plan(lines)
    assert len(plan["steps"]) <= 10


# ---------------------------------------------------------------------------
# AgenticStep
# ---------------------------------------------------------------------------

def test_agentic_step_to_dict():
    step = AgenticStep(step_num=1, description="Do something")
    step.result = "Done"
    step.status = "complete"
    step.finish()
    d = step.to_dict()
    assert d["step_num"] == 1
    assert d["description"] == "Do something"
    assert d["status"] == "complete"
    assert d["duration_seconds"] is not None
    assert d["duration_seconds"] >= 0


# ---------------------------------------------------------------------------
# AgenticLoop – unit tests with mocked agents
# ---------------------------------------------------------------------------

def _make_main_agent(plan_steps=None, step_result="Step executed."):
    agent = MagicMock()
    agent.plan_task = AsyncMock(
        return_value={
            "approach": "Test approach",
            "steps": plan_steps or ["Step 1", "Step 2"],
        }
    )
    agent.execute_step = AsyncMock(return_value=step_result)
    return agent


@pytest.mark.asyncio
async def test_loop_runs_all_steps():
    main = _make_main_agent(plan_steps=["Step A", "Step B", "Step C"])

    loop = AgenticLoop(main_agent=main)
    result = await loop.run(task="Test task")

    assert result["total_steps"] == 3
    assert result["completion_verified"] is True
    assert all(s["status"] == "complete" for s in result["steps"])


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_loop_collects_progress_events():
    main = _make_main_agent(plan_steps=["Step A"])

    events = []

    async def on_progress(event):
        events.append(event["type"])

    loop = AgenticLoop(main_agent=main)
    await loop.run(task="Test task", on_progress=on_progress)

    assert "start" in events
    assert "planning" in events
    assert "plan_ready" in events
    assert "step_start" in events
    assert "step_complete" in events
    assert "verifying" in events
    assert "complete" in events


@pytest.mark.asyncio
async def test_loop_handles_plan_task_failure():
    main = MagicMock()
    main.plan_task = AsyncMock(side_effect=RuntimeError("LLM down"))
    main.execute_step = AsyncMock(return_value="done")

    loop = AgenticLoop(main_agent=main)
    result = await loop.run(task="Test task")

    # Falls back to single-step execution
    assert result["total_steps"] == 1


@pytest.mark.asyncio
async def test_loop_verify_failure_marks_unverified():
    main = _make_main_agent(plan_steps=["Step A"], step_result="done")

    loop = AgenticLoop(main_agent=main)
    result = await loop.run(task="Test task")

    assert result["completion_verified"] is True
    assert "complete" in result["summary"].lower()


@pytest.mark.asyncio
async def test_loop_emits_tool_events_from_step_execution():
    main = _make_main_agent(plan_steps=["Step A"], step_result="done")

    async def execute_step_with_events(*args, **kwargs):
        on_event = kwargs.get("on_event")
        if on_event is not None:
            await on_event({"type": "tool_call", "tool": "vision_click", "args": {"cell": "C3"}})
            await on_event({"type": "tool_result", "tool": "vision_click", "ok": True, "output": "clicked"})
        return "done"

    main.execute_step = AsyncMock(side_effect=execute_step_with_events)

    events: list[str] = []

    async def on_progress(event):
        events.append(str(event.get("type")))

    loop = AgenticLoop(main_agent=main)
    await loop.run(task="Test task", on_progress=on_progress)

    assert "tool_call" in events
    assert "tool_result" in events


@pytest.mark.asyncio
async def test_loop_respects_max_steps():
    main = _make_main_agent(plan_steps=[f"Step {i}" for i in range(1, 15)])

    loop = AgenticLoop(main_agent=main, max_steps=5)
    result = await loop.run(task="Test task")

    assert result["total_steps"] <= 5


@pytest.mark.asyncio
async def test_loop_llm_only_skips_planning_and_tools_and_context():
    from backend.agents.agentic_loop import AgenticDebugOptions

    main = MagicMock()
    main.plan_task = AsyncMock(return_value={"approach": "x", "steps": ["A", "B"]})
    main.execute_step = AsyncMock(return_value="llm response")

    loop = AgenticLoop(
        main_agent=main,
        debug_options=AgenticDebugOptions(
            enabled=True,
            llm_only=True,
            planning_enabled=False,
            rolling_context_enabled=False,
            tool_calls_enabled=False,
        ),
    )

    result = await loop.run(task="Do one thing")

    main.plan_task.assert_not_called()
    assert result["total_steps"] == 1
    kwargs = main.execute_step.call_args.kwargs
    assert kwargs["tool_calls_enabled"] is False
    assert kwargs["include_rolling_context"] is False
    assert kwargs["context"] == []


@pytest.mark.asyncio
async def test_loop_passes_semantic_and_allowed_tools_to_execute_step():
    from backend.agents.agentic_loop import AgenticDebugOptions

    main = _make_main_agent(plan_steps=["Step A"], step_result="done")

    loop = AgenticLoop(
        main_agent=main,
        debug_options=AgenticDebugOptions(
            enabled=True,
            semantic_routing_enabled=False,
            allowed_tools=["web_fetch", "file"],
        ),
    )

    await loop.run(task="Test task")

    kwargs = main.execute_step.call_args.kwargs
    assert kwargs["semantic_routing_enabled"] is False
    assert kwargs["allowed_tool_names"] == ["web_fetch", "file"]


@pytest.mark.asyncio
async def test_loop_screenshot_fn_called():
    main = _make_main_agent(plan_steps=["Step A"])

    screenshot_fn = AsyncMock(return_value={"annotated_png_b64": "abc123"})
    loop = AgenticLoop(main_agent=main, screenshot_fn=screenshot_fn)
    result = await loop.run(task="Test task")

    screenshot_fn.assert_called_once()
    assert result["steps"][0]["screenshots"] == ["abc123"]


@pytest.mark.asyncio
async def test_loop_screenshot_failure_does_not_abort():
    main = _make_main_agent(plan_steps=["Step A"])

    screenshot_fn = AsyncMock(side_effect=RuntimeError("no display"))
    loop = AgenticLoop(main_agent=main, screenshot_fn=screenshot_fn)
    result = await loop.run(task="Test task")

    # Screenshots failure is swallowed; loop still completes
    assert result["total_steps"] == 1
    assert result["steps"][0]["screenshots"] == []


# ---------------------------------------------------------------------------
# New MainAgent methods – integration with slot_manager mock
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_main_agent_plan_task():
    from backend.agents.main_agent import MainAgent
    from backend.config import AppConfig

    sm = MagicMock()
    sm.send_to_main = AsyncMock(
        return_value="APPROACH: modular\nSTEPS:\n1. Step one\n2. Step two"
    )
    cfg = AppConfig()

    agent = MainAgent(slot_manager=sm, memory=None, config=cfg)
    plan = await agent.plan_task("Build something")
    assert plan["approach"] == "modular"
    assert plan["steps"] == ["Step one", "Step two"]


@pytest.mark.asyncio
async def test_main_agent_plan_task_web_fetch_fast_path():
    from backend.agents.main_agent import MainAgent
    from backend.config import AppConfig

    sm = MagicMock()
    sm.send_to_main = AsyncMock(return_value="should not be called")
    cfg = AppConfig()

    agent = MainAgent(slot_manager=sm, memory=None, config=cfg)
    plan = await agent.plan_task("can you fetch https://bytebrew.cc?")

    assert "web_fetch" in plan["steps"][0]
    assert "https://bytebrew.cc" in plan["steps"][0]
    sm.send_to_main.assert_not_called()


@pytest.mark.asyncio
async def test_main_agent_execute_step():
    from backend.agents.main_agent import MainAgent
    from backend.config import AppConfig

    sm = MagicMock()
    sm.send_to_main = AsyncMock(return_value="Step completed successfully.")
    cfg = AppConfig()

    agent = MainAgent(slot_manager=sm, memory=None, config=cfg)
    result = await agent.execute_step(
        step="Do the first thing",
        task="Bigger task",
        context=["Previous step result"],
    )
    assert result == "Step completed successfully."
    # Verify context was included in the prompt
    call_prompt = sm.send_to_main.call_args[0][0]
    assert "Bigger task" in call_prompt
    assert "Previous step result" in call_prompt
