"""Tests for the enhanced Python-only SafetySupervisor."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from backend.core.checkpoint_manager import CheckpointManager
from backend.core.safety_supervisor import LoopHandle, SafetySupervisor, _extract_tps_from_metrics


def test_extract_tps_from_metrics_parses_known_metric_name() -> None:
    metrics = """
    # HELP llama_tokens_per_second Current token throughput
    # TYPE llama_tokens_per_second gauge
    llama_tokens_per_second 142.75
    """
    snapshot = _extract_tps_from_metrics(metrics)
    assert snapshot is not None
    assert snapshot.tokens_per_sec == 142.75


@pytest.mark.asyncio
async def test_loop_handle_repeated_tool_call_detection() -> None:
    handle = LoopHandle(run_id="r1")
    handle.configure_detection_thresholds(
        max_repeated_tool_calls=3,
        max_failed_tool_calls=3,
        max_no_progress_steps=3,
        max_empty_or_malformed=2,
        max_give_up_signals=1,
    )

    assert handle.observe_tool_call("file", {"path": "README.md"}) is None
    assert handle.observe_tool_call("file", {"path": "README.md"}) is None
    detected = handle.observe_tool_call("file", {"path": "README.md"})

    assert detected is not None
    assert detected["action"] == "reset_context"
    assert detected["category"] == "looping_tool_calls"


@pytest.mark.asyncio
async def test_loop_handle_failed_tool_detection() -> None:
    handle = LoopHandle(run_id="r2")
    handle.configure_detection_thresholds(
        max_repeated_tool_calls=3,
        max_failed_tool_calls=2,
        max_no_progress_steps=3,
        max_empty_or_malformed=2,
        max_give_up_signals=1,
    )

    assert handle.observe_tool_result(ok=False, output=None, error="boom") is None
    detected = handle.observe_tool_result(ok=False, output=None, error="boom")

    assert detected is not None
    assert detected["category"] == "failed_actions"


@pytest.mark.asyncio
async def test_loop_handle_empty_output_detection() -> None:
    handle = LoopHandle(run_id="r3")
    handle.configure_detection_thresholds(
        max_repeated_tool_calls=3,
        max_failed_tool_calls=3,
        max_no_progress_steps=3,
        max_empty_or_malformed=2,
        max_give_up_signals=1,
    )

    assert handle.observe_model_output("", malformed=False) is None
    detected = handle.observe_model_output("", malformed=False)

    assert detected is not None
    assert detected["category"] == "no_response"


@pytest.mark.asyncio
async def test_loop_handle_give_up_detection() -> None:
    handle = LoopHandle(run_id="r4")
    handle.configure_detection_thresholds(
        max_repeated_tool_calls=3,
        max_failed_tool_calls=3,
        max_no_progress_steps=3,
        max_empty_or_malformed=2,
        max_give_up_signals=1,
    )

    detected = handle.observe_model_output("I am stuck and cannot continue.", malformed=False)

    assert detected is not None
    assert detected["category"] == "model_gave_up"


@pytest.mark.asyncio
async def test_loop_handle_stalled_progress_detection() -> None:
    handle = LoopHandle(run_id="r5")
    handle.configure_detection_thresholds(
        max_repeated_tool_calls=3,
        max_failed_tool_calls=3,
        max_no_progress_steps=3,
        max_empty_or_malformed=2,
        max_give_up_signals=1,
    )

    assert handle.observe_step_result("same output") is None
    assert handle.observe_step_result("same output") is None
    detected = handle.observe_step_result("same output")

    assert detected is not None
    assert detected["category"] == "stalled_progress"


@pytest.mark.asyncio
async def test_supervisor_detects_stall_and_injects_reset_context() -> None:
    cp = CheckpointManager()
    client = MagicMock()

    sup = SafetySupervisor(
        llama_client=client,
        checkpoint_manager=cp,
        poll_interval=0.05,
        step_timeout=0.01,
    )

    handle = sup.register("run-stall")
    cp.save(
        run_id="run-stall",
        task="build website",
        step_num=1,
        step_descriptions=["step 1"],
        context=["ctx"],
        executed_steps=[],
    )
    handle.heartbeat(1, "build website", ["ctx"])
    handle._last_heartbeat = time.monotonic() - 1.0

    await sup._run_once()

    iv = await handle.pop_intervention()
    assert iv is not None
    assert iv["action"] == "reset_context"
    assert iv["category"] == "stalled_step"


@pytest.mark.asyncio
async def test_supervisor_slowdown_triggers_restart_after_consecutive_breaches() -> None:
    cp = CheckpointManager()
    client = MagicMock()
    client.get_metrics = AsyncMock(return_value="llama_tokens_per_second 8")
    server_manager = MagicMock()
    server_manager.restart = AsyncMock(return_value=None)

    sup = SafetySupervisor(
        llama_client=client,
        checkpoint_manager=cp,
        poll_interval=999,
        step_timeout=999,
        server_manager=server_manager,
        slowdown_monitor_enabled=True,
        slowdown_threshold_tps=20.0,
        slowdown_consecutive_polls=2,
        slowdown_restart_enabled=True,
        slowdown_cooldown_seconds=0.0,
    )

    await sup._run_once()
    server_manager.restart.assert_not_awaited()

    await sup._run_once()
    server_manager.restart.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervisor_disables_metrics_monitor_when_endpoint_unsupported() -> None:
    cp = CheckpointManager()
    client = MagicMock()
    req = httpx.Request("GET", "http://127.0.0.1:8080/metrics")
    resp = httpx.Response(status_code=501, request=req)
    client.get_metrics = AsyncMock(side_effect=httpx.HTTPStatusError("unsupported", request=req, response=resp))

    sup = SafetySupervisor(
        llama_client=client,
        checkpoint_manager=cp,
        poll_interval=999,
        step_timeout=999,
        slowdown_monitor_enabled=True,
    )

    await sup._run_once()
    assert sup._metrics_unsupported is True
    assert client.get_metrics.await_count == 1

    await sup._run_once()
    assert client.get_metrics.await_count == 1


@pytest.mark.asyncio
async def test_supervisor_start_creates_task() -> None:
    cp = CheckpointManager()
    client = MagicMock()
    client.get_metrics = AsyncMock(return_value="")

    sup = SafetySupervisor(
        llama_client=client,
        checkpoint_manager=cp,
        poll_interval=100,
        step_timeout=100,
    )
    sup.start()
    assert sup._task is not None
    assert not sup._task.done()
    sup.stop()
    await asyncio.sleep(0)
    assert sup._task.done()
