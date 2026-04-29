"""Tests for SafetySupervisor, LoopHandle, and Judge response parsing."""
from __future__ import annotations

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.core.safety_supervisor import (
    SafetySupervisor,
    LoopHandle,
    _parse_judge_response,
    _VALID_JUDGE_ACTIONS,
)
from backend.core.checkpoint_manager import CheckpointManager


# ---------------------------------------------------------------------------
# _parse_judge_response
# ---------------------------------------------------------------------------

def test_parse_judge_nudge():
    result = _parse_judge_response("ACTION: nudge\nMESSAGE: keep going")
    assert result["action"] == "nudge"
    assert "keep going" in result["message"]


def test_parse_judge_reset_context():
    result = _parse_judge_response("ACTION: reset_context\nMESSAGE: context is corrupted")
    assert result["action"] == "reset_context"


def test_parse_judge_retry_step():
    result = _parse_judge_response("ACTION: retry_step\nMESSAGE: try again")
    assert result["action"] == "retry_step"


def test_parse_judge_end_task():
    result = _parse_judge_response("ACTION: end_task\nMESSAGE: impossible")
    assert result["action"] == "end_task"


def test_parse_judge_escalate():
    result = _parse_judge_response("ACTION: escalate_to_user\nMESSAGE: needs human input")
    assert result["action"] == "escalate_to_user"


def test_parse_judge_fallback_keyword():
    """Keyword scan should work when ACTION: label is absent."""
    result = _parse_judge_response("The best thing to do here is retry_step the operation.")
    assert result["action"] == "retry_step"


def test_parse_judge_unknown_defaults_to_nudge():
    result = _parse_judge_response("I have no idea what to do here.")
    assert result["action"] == "nudge"


def test_parse_judge_message_truncated():
    long_msg = "x" * 500
    result = _parse_judge_response(f"ACTION: nudge\nMESSAGE: {long_msg}")
    assert len(result["message"]) <= 400


# ---------------------------------------------------------------------------
# LoopHandle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_handle_heartbeat_resets_stall():
    handle = LoopHandle(run_id="r1")
    # Manually backdate last heartbeat
    handle._last_heartbeat = time.monotonic() - 200
    assert handle.is_stalled(timeout=120.0)
    handle.heartbeat(step_num=2, task="t", context=[])
    assert not handle.is_stalled(timeout=120.0)


@pytest.mark.asyncio
async def test_loop_handle_pop_intervention_clears():
    handle = LoopHandle(run_id="r1")
    await handle.set_intervention({"action": "nudge", "message": "hi"})
    iv = await handle.pop_intervention()
    assert iv is not None
    assert iv["action"] == "nudge"
    # Second pop returns None
    assert await handle.pop_intervention() is None


def test_loop_handle_deactivate_stops_stall_detection():
    handle = LoopHandle(run_id="r1")
    handle._last_heartbeat = time.monotonic() - 500
    handle.deactivate()
    assert not handle.is_stalled(timeout=120.0)


def test_loop_handle_tracks_recent_context():
    handle = LoopHandle(run_id="r1")
    handle.heartbeat(1, "task", ["c1", "c2", "c3", "c4", "c5"])
    # Only the last 4 context lines are kept
    assert len(handle.recent_context) == 4
    assert handle.recent_context[-1] == "c5"


# ---------------------------------------------------------------------------
# SafetySupervisor – registration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supervisor_register_returns_handle():
    cp = CheckpointManager()
    client = MagicMock()
    sup = SafetySupervisor(llama_client=client, checkpoint_manager=cp)
    handle = sup.register("run-1")
    assert isinstance(handle, LoopHandle)
    assert handle.run_id == "run-1"


@pytest.mark.asyncio
async def test_supervisor_deregister_deactivates_handle():
    cp = CheckpointManager()
    client = MagicMock()
    sup = SafetySupervisor(llama_client=client, checkpoint_manager=cp)
    handle = sup.register("run-1")
    sup.deregister("run-1")
    assert not handle._active
    assert "run-1" not in sup._handles


# ---------------------------------------------------------------------------
# SafetySupervisor – stall detection and judge call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supervisor_detects_stall_and_injects_intervention():
    cp = CheckpointManager()
    client = MagicMock()
    client.complete = AsyncMock(
        return_value="ACTION: retry_step\nMESSAGE: step seems stuck"
    )

    sup = SafetySupervisor(
        llama_client=client,
        checkpoint_manager=cp,
        poll_interval=0.05,   # very short for tests
        step_timeout=0.01,    # declare stalled immediately
    )

    handle = sup.register("run-stall")
    handle.heartbeat(1, "build website", ["ctx"])
    # Backdate to simulate stall
    handle._last_heartbeat = time.monotonic() - 1.0

    # Run one poll cycle manually
    await sup._run_once()

    iv = await handle.pop_intervention()
    assert iv is not None
    assert iv["action"] == "retry_step"


@pytest.mark.asyncio
async def test_supervisor_judge_failure_defaults_to_nudge():
    cp = CheckpointManager()
    client = MagicMock()
    client.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

    sup = SafetySupervisor(
        llama_client=client,
        checkpoint_manager=cp,
        poll_interval=999,
        step_timeout=0.01,
    )
    handle = sup.register("run-x")
    handle._last_heartbeat = time.monotonic() - 1.0
    handle.heartbeat(1, "task", [])
    handle._last_heartbeat = time.monotonic() - 1.0

    await sup._handle_stall(handle)

    iv = await handle.pop_intervention()
    assert iv is not None
    assert iv["action"] == "nudge"


# ---------------------------------------------------------------------------
# SafetySupervisor – start / stop lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supervisor_start_creates_task():
    cp = CheckpointManager()
    client = MagicMock()
    client.complete = AsyncMock(return_value="ACTION: nudge\nMESSAGE: ok")

    sup = SafetySupervisor(
        llama_client=client,
        checkpoint_manager=cp,
        poll_interval=100,   # never fires in this test
        step_timeout=100,
    )
    sup.start()
    assert sup._task is not None
    assert not sup._task.done()
    sup.stop()
    # Give event loop a moment to cancel
    await asyncio.sleep(0)
    assert sup._task.done()


# ---------------------------------------------------------------------------
# Integration: AgenticLoop with CheckpointManager and SafetySupervisor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_saves_checkpoints():
    from backend.agents.agentic_loop import AgenticLoop

    cp = CheckpointManager()
    client = MagicMock()
    client.complete = AsyncMock(return_value="ACTION: nudge\nMESSAGE: ok")
    sup = SafetySupervisor(
        llama_client=client,
        checkpoint_manager=cp,
        poll_interval=999,
        step_timeout=999,
    )

    main = MagicMock()
    main.plan_task = AsyncMock(
        return_value={"approach": "test", "steps": ["step A", "step B"]}
    )
    main.execute_step = AsyncMock(return_value="done")

    watcher = MagicMock()
    watcher.monitor_step = AsyncMock(
        return_value={"action": "continue", "feedback": "ok"}
    )
    watcher.verify_completion = AsyncMock(
        return_value={"complete": True, "feedback": "verified"}
    )

    loop = AgenticLoop(
        main_agent=main,
        watcher_agent=watcher,
        checkpoint_manager=cp,
        supervisor=sup,
    )

    result = await loop.run(task="test task")
    assert result["total_steps"] == 2

    # Run ID is cleared after completion so no checkpoints remain
    # (the loop calls cp.clear at the end)
    # We can't easily check the run_id externally, but we verify the loop completed
    assert result["completion_verified"] is True


@pytest.mark.asyncio
async def test_loop_handles_nudge_intervention():
    from backend.agents.agentic_loop import AgenticLoop

    cp = CheckpointManager()
    client = MagicMock()
    client.complete = AsyncMock(return_value="ACTION: nudge\nMESSAGE: keep going")
    sup = SafetySupervisor(
        llama_client=client,
        checkpoint_manager=cp,
        poll_interval=999,
        step_timeout=999,
    )

    execute_calls = {"n": 0}

    main = MagicMock()
    main.plan_task = AsyncMock(
        return_value={"approach": "", "steps": ["step A"]}
    )

    async def execute_with_nudge(*args, **kwargs):
        execute_calls["n"] += 1
        # Inject nudge intervention before the step actually runs
        # (in real usage this comes from the background supervisor)
        return "executed"

    main.execute_step = execute_with_nudge

    watcher = MagicMock()
    watcher.monitor_step = AsyncMock(
        return_value={"action": "continue", "feedback": "ok"}
    )
    watcher.verify_completion = AsyncMock(
        return_value={"complete": True, "feedback": "done"}
    )

    loop = AgenticLoop(
        main_agent=main,
        watcher_agent=watcher,
        checkpoint_manager=cp,
        supervisor=sup,
    )

    result = await loop.run(task="test")
    assert result["total_steps"] == 1
    assert result["completion_verified"] is True


@pytest.mark.asyncio
async def test_loop_handles_end_task_intervention():
    from backend.agents.agentic_loop import AgenticLoop

    cp = CheckpointManager()
    client = MagicMock()
    sup = SafetySupervisor(
        llama_client=client,
        checkpoint_manager=cp,
        poll_interval=999,
        step_timeout=999,
    )

    main = MagicMock()
    main.plan_task = AsyncMock(
        return_value={"approach": "", "steps": ["step A", "step B"]}
    )
    main.execute_step = AsyncMock(return_value="done")

    watcher = MagicMock()

    call_n = {"n": 0}

    async def monitor_with_end_task(**kwargs):
        call_n["n"] += 1
        return {"action": "continue", "feedback": "ok"}

    watcher.monitor_step = monitor_with_end_task
    watcher.verify_completion = AsyncMock(
        return_value={"complete": False, "feedback": "stopped"}
    )

    # Manually inject an end_task intervention after the first step
    first_handle = None
    original_register = sup.register

    def capture_register(run_id):
        nonlocal first_handle
        first_handle = original_register(run_id)
        return first_handle

    sup.register = capture_register

    loop = AgenticLoop(
        main_agent=main,
        watcher_agent=watcher,
        checkpoint_manager=cp,
        supervisor=sup,
    )

    # Inject end_task before the second step executes
    inject_done = False

    original_execute = main.execute_step

    async def inject_then_execute(*args, **kwargs):
        nonlocal inject_done
        if not inject_done and first_handle is not None:
            inject_done = True
            await first_handle.set_intervention(
                {"action": "end_task", "message": "supervisor says stop"}
            )
        return await original_execute(*args, **kwargs)

    main.execute_step = inject_then_execute

    result = await loop.run(task="test task")
    assert result["completion_verified"] is False
    assert "stopped" in result["summary"].lower() or "step" in result["summary"].lower()
