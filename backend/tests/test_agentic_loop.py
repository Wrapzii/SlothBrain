"""Tests for the agentic loop and the new watcher/main-agent helpers."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.agentic_loop import AgenticLoop, AgenticStep, _build_result
from backend.agents.watcher import _parse_monitor_response, _parse_verify_response
from backend.agents.main_agent import _parse_plan


# ---------------------------------------------------------------------------
# _parse_monitor_response
# ---------------------------------------------------------------------------

def test_parse_monitor_continue():
    result = _parse_monitor_response("ACTION: continue\nFEEDBACK: looks good")
    assert result["action"] == "continue"
    assert "looks good" in result["feedback"]


def test_parse_monitor_retry():
    result = _parse_monitor_response("ACTION: retry\nFEEDBACK: incomplete result")
    assert result["action"] == "retry"


def test_parse_monitor_done():
    result = _parse_monitor_response("ACTION: done\nFEEDBACK: task fully complete")
    assert result["action"] == "done"


def test_parse_monitor_abort():
    result = _parse_monitor_response("ACTION: abort\nFEEDBACK: impossible request")
    assert result["action"] == "abort"


def test_parse_monitor_fallback_keyword():
    """When no explicit ACTION: label, keyword scan should work."""
    result = _parse_monitor_response("The step failed badly, we should retry it.")
    assert result["action"] == "retry"


def test_parse_monitor_default_continue():
    """Unknown response defaults to continue."""
    result = _parse_monitor_response("Everything seems fine.")
    assert result["action"] == "continue"


# ---------------------------------------------------------------------------
# _parse_verify_response
# ---------------------------------------------------------------------------

def test_parse_verify_yes():
    result = _parse_verify_response("COMPLETE: yes\nFEEDBACK: All steps done.")
    assert result["complete"] is True
    assert "All steps done" in result["feedback"]


def test_parse_verify_no():
    result = _parse_verify_response("COMPLETE: no\nFEEDBACK: Some steps failed.")
    assert result["complete"] is False


def test_parse_verify_not_complete_phrase():
    result = _parse_verify_response("The task is not complete yet.")
    assert result["complete"] is False


def test_parse_verify_default_complete():
    result = _parse_verify_response("Task seems to be done.")
    assert result["complete"] is True


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


def _make_watcher_agent(action="continue", feedback="OK", complete=True):
    agent = MagicMock()
    agent.monitor_step = AsyncMock(
        return_value={"action": action, "feedback": feedback}
    )
    agent.verify_completion = AsyncMock(
        return_value={"complete": complete, "feedback": "Verified."}
    )
    return agent


@pytest.mark.asyncio
async def test_loop_runs_all_steps():
    main = _make_main_agent(plan_steps=["Step A", "Step B", "Step C"])
    watcher = _make_watcher_agent(action="continue")

    loop = AgenticLoop(main_agent=main, watcher_agent=watcher)
    result = await loop.run(task="Test task")

    assert result["total_steps"] == 3
    assert result["completion_verified"] is True
    assert all(s["status"] == "complete" for s in result["steps"])


@pytest.mark.asyncio
async def test_loop_stops_early_on_done():
    main = _make_main_agent(plan_steps=["Step A", "Step B", "Step C"])
    watcher = _make_watcher_agent(action="done")  # watcher says done after first step

    loop = AgenticLoop(main_agent=main, watcher_agent=watcher)
    result = await loop.run(task="Test task")

    # Should stop after the first step
    assert result["total_steps"] == 1


@pytest.mark.asyncio
async def test_loop_aborts_on_abort():
    main = _make_main_agent(plan_steps=["Step A", "Step B"])
    watcher = _make_watcher_agent(action="abort", feedback="Impossible request")

    loop = AgenticLoop(main_agent=main, watcher_agent=watcher)
    result = await loop.run(task="Test task")

    assert result["completion_verified"] is False
    assert "aborted" in result["summary"].lower() or "impossible" in result["summary"].lower()
    assert result["steps"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_loop_retries_on_retry():
    main = _make_main_agent(plan_steps=["Step A"])
    # return retry twice then continue
    call_count = {"n": 0}
    async def mock_monitor(**kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            return {"action": "retry", "feedback": "needs improvement"}
        return {"action": "continue", "feedback": "good"}

    watcher = _make_watcher_agent()
    watcher.monitor_step = mock_monitor

    loop = AgenticLoop(main_agent=main, watcher_agent=watcher)
    result = await loop.run(task="Test task")

    assert result["total_steps"] == 1
    assert result["steps"][0]["retries"] == 2


@pytest.mark.asyncio
async def test_loop_collects_progress_events():
    main = _make_main_agent(plan_steps=["Step A"])
    watcher = _make_watcher_agent(action="continue")

    events = []

    async def on_progress(event):
        events.append(event["type"])

    loop = AgenticLoop(main_agent=main, watcher_agent=watcher)
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
    watcher = _make_watcher_agent()

    loop = AgenticLoop(main_agent=main, watcher_agent=watcher)
    result = await loop.run(task="Test task")

    # Falls back to single-step execution
    assert result["total_steps"] == 1


@pytest.mark.asyncio
async def test_loop_respects_max_steps():
    main = _make_main_agent(plan_steps=[f"Step {i}" for i in range(1, 15)])
    watcher = _make_watcher_agent(action="continue")

    loop = AgenticLoop(main_agent=main, watcher_agent=watcher, max_steps=5)
    result = await loop.run(task="Test task")

    assert result["total_steps"] <= 5


@pytest.mark.asyncio
async def test_loop_screenshot_fn_called():
    main = _make_main_agent(plan_steps=["Step A"])
    watcher = _make_watcher_agent()

    screenshot_fn = AsyncMock(return_value={"annotated_png_b64": "abc123"})
    loop = AgenticLoop(main_agent=main, watcher_agent=watcher, screenshot_fn=screenshot_fn)
    result = await loop.run(task="Test task")

    screenshot_fn.assert_called_once()
    assert result["steps"][0]["screenshots"] == ["abc123"]


@pytest.mark.asyncio
async def test_loop_screenshot_failure_does_not_abort():
    main = _make_main_agent(plan_steps=["Step A"])
    watcher = _make_watcher_agent()

    screenshot_fn = AsyncMock(side_effect=RuntimeError("no display"))
    loop = AgenticLoop(main_agent=main, watcher_agent=watcher, screenshot_fn=screenshot_fn)
    result = await loop.run(task="Test task")

    # Screenshots failure is swallowed; loop still completes
    assert result["total_steps"] == 1
    assert result["steps"][0]["screenshots"] == []


# ---------------------------------------------------------------------------
# New WatcherAgent methods – integration with slot_manager mock
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_watcher_monitor_step():
    from backend.agents.watcher import WatcherAgent
    from backend.config import AppConfig

    sm = MagicMock()
    sm.send_to_watcher = AsyncMock(return_value="ACTION: continue\nFEEDBACK: looks good")
    rc = MagicMock()
    cfg = AppConfig()

    watcher = WatcherAgent(
        slot_manager=sm,
        rolling_context=rc,
        memory=None,
        config=cfg,
    )
    result = await watcher.monitor_step(
        task="Build a website",
        step_description="Research available frameworks",
        step_result="Found React, Vue, Angular",
        step_num=1,
        total_steps=5,
    )
    assert result["action"] == "continue"
    assert "looks good" in result["feedback"]


@pytest.mark.asyncio
async def test_watcher_verify_completion():
    from backend.agents.watcher import WatcherAgent
    from backend.config import AppConfig

    sm = MagicMock()
    sm.send_to_watcher = AsyncMock(return_value="COMPLETE: yes\nFEEDBACK: All done.")
    rc = MagicMock()
    cfg = AppConfig()

    watcher = WatcherAgent(
        slot_manager=sm,
        rolling_context=rc,
        memory=None,
        config=cfg,
    )
    result = await watcher.verify_completion(
        task="Build a website",
        steps_summary=["Step 1: researched", "Step 2: built"],
    )
    assert result["complete"] is True
    assert "All done" in result["feedback"]


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
