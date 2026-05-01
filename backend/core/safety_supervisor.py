"""Safety supervisor for the agentic loop.

Pure-Python watchdog for detecting model/runtime failure modes without relying
on an additional LLM judge. The supervisor now classifies failures directly
from execution events (tool calls/results and model outputs), plus heartbeat and
throughput monitoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from collections import deque
from typing import TYPE_CHECKING, Optional

import httpx

if TYPE_CHECKING:
    from backend.core.checkpoint_manager import CheckpointManager, TaskCheckpoint
    from backend.core.llama_client import LlamaClient
    from backend.core.server_manager import ServerManager

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tuneable constants
# ──────────────────────────────────────────────────────────────────────────────

# Seconds a single step may run before being considered stalled.
_DEFAULT_STEP_TIMEOUT: float = 120.0

# How often the supervisor polls active loops.
_DEFAULT_POLL_INTERVAL: float = 15.0

_GIVE_UP_PATTERNS = (
    "i can't",
    "cannot continue",
    "i am stuck",
    "i'm stuck",
    "unable to proceed",
    "give up",
)


@dataclass
class SlowdownSnapshot:
    tokens_per_sec: float
    metric_name: str


def _extract_tps_from_metrics(metrics_text: str) -> SlowdownSnapshot | None:
    """Best-effort parser for llama.cpp Prometheus metrics token throughput.

    We accept several common metric names seen across llama.cpp builds and use
    the first finite value encountered.
    """
    candidates = (
        "llama_tokens_per_second",
        "llamacpp_tokens_per_second",
        "tokens_per_second",
        "generation_tokens_per_second",
    )

    for raw in metrics_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        for metric in candidates:
            if metric in name:
                try:
                    value = float(parts[-1])
                except ValueError:
                    continue
                if value >= 0:
                    return SlowdownSnapshot(tokens_per_sec=value, metric_name=name)
    return None


def _fingerprint_payload(payload: dict) -> str:
    """Stable short hash for tool call argument comparisons."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# LoopHandle – per-run communication channel
# ──────────────────────────────────────────────────────────────────────────────


class LoopHandle:
    """Shared state between a running ``AgenticLoop`` and the supervisor.

    The loop updates this object; the supervisor reads it and injects
    interventions.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.current_step: int = 0
        self.task: str = ""
        self.recent_context: list[str] = []
        self._last_heartbeat: float = time.monotonic()
        self._intervention: dict | None = None
        self._lock = asyncio.Lock()
        self._active: bool = True
        self._tool_call_history: deque[str] = deque(maxlen=16)
        self._progress_fingerprints: deque[str] = deque(maxlen=10)
        self._consecutive_failed_tools: int = 0
        self._consecutive_empty_or_malformed: int = 0
        self._give_up_signal_count: int = 0
        self._last_detected_key: str = ""
        self._last_tool_name: str = ""
        self._failed_tool_trace: deque[str] = deque(maxlen=5)

        self._max_repeated_tool_calls: int = 3
        self._max_failed_tool_calls: int = 3
        self._max_no_progress_steps: int = 3
        self._max_empty_or_malformed: int = 2
        self._max_give_up_signals: int = 1

    # ------------------------------------------------------------------
    # Called by the loop
    # ------------------------------------------------------------------

    def heartbeat(
        self,
        step_num: int,
        task: str,
        context: list[str],
    ) -> None:
        """Signal liveness at the start of each step.

        This is called from the loop's async task and is intentionally
        lock-free (monotonic writes are safe on CPython).
        """
        previous_step = self.current_step
        self._last_heartbeat = time.monotonic()
        self.current_step = step_num
        self.task = task
        # Keep only the 4 most recent context lines to bound memory
        self.recent_context = list(context[-4:]) if context else []
        if previous_step != step_num:
            self._last_detected_key = ""

    def configure_detection_thresholds(
        self,
        *,
        max_repeated_tool_calls: int,
        max_failed_tool_calls: int,
        max_no_progress_steps: int,
        max_empty_or_malformed: int,
        max_give_up_signals: int,
    ) -> None:
        self._max_repeated_tool_calls = max(1, max_repeated_tool_calls)
        self._max_failed_tool_calls = max(1, max_failed_tool_calls)
        self._max_no_progress_steps = max(1, max_no_progress_steps)
        self._max_empty_or_malformed = max(1, max_empty_or_malformed)
        self._max_give_up_signals = max(1, max_give_up_signals)

    def observe_model_output(self, output: str, malformed: bool = False) -> dict | None:
        text = (output or "").strip()
        lowered = text.lower()

        if malformed or not text:
            self._consecutive_empty_or_malformed += 1
        else:
            self._consecutive_empty_or_malformed = 0

        if any(p in lowered for p in _GIVE_UP_PATTERNS):
            self._give_up_signal_count += 1

        if self._consecutive_empty_or_malformed >= self._max_empty_or_malformed:
            return self._build_detection_intervention(
                "no_response",
                "Model returned empty or malformed output repeatedly; restoring checkpoint.",
            )

        if self._give_up_signal_count >= self._max_give_up_signals:
            return self._build_detection_intervention(
                "model_gave_up",
                "Model indicated it is stuck or unable to proceed; restoring checkpoint.",
            )

        return None

    def observe_tool_call(self, tool_name: str, args: dict | None) -> dict | None:
        self._last_tool_name = tool_name
        fp = f"{tool_name}:{_fingerprint_payload(args or {})}"
        self._tool_call_history.append(fp)

        if len(self._tool_call_history) < self._max_repeated_tool_calls:
            return None

        tail = list(self._tool_call_history)[-self._max_repeated_tool_calls :]
        if len(set(tail)) == 1:
            return self._build_detection_intervention(
                "looping_tool_calls",
                f"Detected repeated tool loop for '{tool_name}' with near-identical arguments.",
            )
        return None

    def observe_tool_result(self, ok: bool, output: object, error: str | None) -> dict | None:
        if not ok:
            self._consecutive_failed_tools += 1
            tool_label = self._last_tool_name or "unknown_tool"
            err = (error or "unknown error").strip()
            self._failed_tool_trace.append(f"{tool_label}: {err[:120]}")
        else:
            self._consecutive_failed_tools = 0
            self._failed_tool_trace.clear()
            self._last_detected_key = ""

        if self._consecutive_failed_tools >= self._max_failed_tool_calls:
            recent_failures = "; ".join(self._failed_tool_trace) if self._failed_tool_trace else "no tool details available"
            return self._build_detection_intervention(
                "failed_actions",
                "Multiple consecutive tool failures detected without progress. "
                f"Recent failures: {recent_failures}",
            )

        # Best-effort no-change fingerprinting from successful tool outputs.
        if ok:
            marker = _fingerprint_payload({"output": output, "error": error})
            self._progress_fingerprints.append(marker)
            if len(self._progress_fingerprints) >= self._max_no_progress_steps:
                tail = list(self._progress_fingerprints)[-self._max_no_progress_steps :]
                if len(set(tail)) == 1:
                    return self._build_detection_intervention(
                        "stalled_progress",
                        "Consecutive tool results show no meaningful state change; restoring checkpoint.",
                    )

        return None

    def observe_step_result(self, step_result: str) -> dict | None:
        marker = _fingerprint_payload({"step_result": (step_result or "").strip()[:300]})
        self._progress_fingerprints.append(marker)
        if len(self._progress_fingerprints) >= self._max_no_progress_steps:
            tail = list(self._progress_fingerprints)[-self._max_no_progress_steps :]
            if len(set(tail)) == 1:
                return self._build_detection_intervention(
                    "stalled_progress",
                    "Recent step outputs are effectively unchanged; restoring checkpoint.",
                )
        return None

    def _build_detection_intervention(self, category: str, message: str) -> dict:
        key = f"{category}:{self.current_step}"
        if key == self._last_detected_key:
            return {}
        self._last_detected_key = key
        return {
            "action": "reset_context",
            "category": category,
            "message": message,
        }

    async def pop_intervention(self) -> dict | None:
        """Atomically retrieve and clear any pending supervisor intervention.

        Returns ``None`` when no intervention is pending.
        """
        async with self._lock:
            iv = self._intervention
            self._intervention = None
            return iv

    # ------------------------------------------------------------------
    # Called by the supervisor
    # ------------------------------------------------------------------

    def seconds_since_heartbeat(self) -> float:
        return time.monotonic() - self._last_heartbeat

    def is_stalled(self, timeout: float) -> bool:
        return self._active and self.seconds_since_heartbeat() > timeout

    async def set_intervention(self, intervention: dict) -> None:
        async with self._lock:
            self._intervention = intervention

    def reset_heartbeat(self) -> None:
        """Prevent repeated firings for the same stall event."""
        self._last_heartbeat = time.monotonic()

    def deactivate(self) -> None:
        self._active = False


# ──────────────────────────────────────────────────────────────────────────────
# SafetySupervisor
# ──────────────────────────────────────────────────────────────────────────────


class SafetySupervisor:
    """Python-level watchdog for all running ``AgenticLoop`` instances.

    The supervisor runs an independent asyncio background task that polls
    registered loop handles every ``poll_interval`` seconds.  It is fully
    decoupled from the LLM inference slots and continues working even when
    all slots are busy.

    If the background task crashes (unexpected exception) it is automatically
    restarted so supervision is never silently lost.

    Parameters
    ----------
    llama_client:
        Shared ``LlamaClient`` used to call the Judge.  The call uses
        ``slot_id=-1`` so llama.cpp picks any free slot; if no slot is
        available the call fails and the supervisor defaults to ``nudge``.
    checkpoint_manager:
        Shared ``CheckpointManager`` used to look up the last good checkpoint
        when a stall is detected.
    poll_interval:
        Seconds between supervision polls (default 15).
    step_timeout:
        Seconds before a step is declared stalled (default 120).
    """

    def __init__(
        self,
        llama_client: "LlamaClient",
        checkpoint_manager: "CheckpointManager",
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        step_timeout: float = _DEFAULT_STEP_TIMEOUT,
        server_manager: "ServerManager | None" = None,
        slowdown_monitor_enabled: bool = True,
        slowdown_threshold_tps: float = 20.0,
        slowdown_consecutive_polls: int = 3,
        slowdown_restart_enabled: bool = False,
        slowdown_cooldown_seconds: float = 300.0,
        max_repeated_tool_calls: int = 3,
        max_failed_tool_calls: int = 3,
        max_no_progress_steps: int = 3,
        max_empty_or_malformed: int = 2,
        max_give_up_signals: int = 1,
    ) -> None:
        self._client = llama_client
        self._cp = checkpoint_manager
        self._poll_interval = poll_interval
        self._step_timeout = step_timeout
        self._server_manager = server_manager
        self._slowdown_monitor_enabled = slowdown_monitor_enabled
        self._slowdown_threshold_tps = slowdown_threshold_tps
        self._slowdown_consecutive_polls = slowdown_consecutive_polls
        self._slowdown_restart_enabled = slowdown_restart_enabled
        self._slowdown_cooldown_seconds = slowdown_cooldown_seconds
        self._slowdown_breach_count: int = 0
        self._last_slowdown_action_ts: float = 0.0
        self._metrics_unsupported: bool = False
        self._max_repeated_tool_calls = max_repeated_tool_calls
        self._max_failed_tool_calls = max_failed_tool_calls
        self._max_no_progress_steps = max_no_progress_steps
        self._max_empty_or_malformed = max_empty_or_malformed
        self._max_give_up_signals = max_give_up_signals
        self._handles: dict[str, LoopHandle] = {}
        self._task: Optional["asyncio.Task[None]"] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background supervisor task.

        Safe to call multiple times — a no-op if already running.
        """
        self._running = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_with_restart())
            logger.info(
                "SafetySupervisor started (poll=%.0fs timeout=%.0fs)",
                self._poll_interval,
                self._step_timeout,
            )

    def stop(self) -> None:
        """Stop the supervisor and clean up all handles."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._handles.clear()
        logger.info("SafetySupervisor stopped")

    # ------------------------------------------------------------------
    # Handle management (called by AgenticLoop)
    # ------------------------------------------------------------------

    def register(self, run_id: str) -> LoopHandle:
        """Register a new loop run and return its ``LoopHandle``."""
        handle = LoopHandle(run_id=run_id)
        handle.configure_detection_thresholds(
            max_repeated_tool_calls=self._max_repeated_tool_calls,
            max_failed_tool_calls=self._max_failed_tool_calls,
            max_no_progress_steps=self._max_no_progress_steps,
            max_empty_or_malformed=self._max_empty_or_malformed,
            max_give_up_signals=self._max_give_up_signals,
        )
        self._handles[run_id] = handle
        logger.debug("SafetySupervisor: registered run %s", run_id)
        return handle

    def deregister(self, run_id: str) -> None:
        """Remove a completed loop from monitoring."""
        handle = self._handles.pop(run_id, None)
        if handle:
            handle.deactivate()
        logger.debug("SafetySupervisor: deregistered run %s", run_id)

    # ------------------------------------------------------------------
    # Internal polling loop
    # ------------------------------------------------------------------

    async def _run_with_restart(self) -> None:
        """Wrapper that restarts the supervision loop if it crashes unexpectedly.

        Without this, any unhandled exception in ``_run`` would silently kill
        the supervisor and leave all registered loops unmonitored.
        """
        while self._running:
            try:
                await self._run()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "SafetySupervisor background task crashed (%s: %s); restarting in 5s",
                    exc.__class__.__name__,
                    exc,
                )
                await asyncio.sleep(5)

    async def _run(self) -> None:
        """Main supervision loop – runs until cancelled."""
        while True:
            await asyncio.sleep(self._poll_interval)
            await self._poll()

    async def _poll(self) -> None:
        """Check all active handles and handle any stalls.  Called by _run and
        exposed for testing as ``_run_once``."""
        await self._monitor_throughput_slowdown()

        stalled = [
            h
            for h in list(self._handles.values())
            if h.is_stalled(self._step_timeout)
        ]
        for handle in stalled:
            logger.warning(
                "SafetySupervisor: run %s stalled on step %d "
                "(%.0fs since last heartbeat)",
                handle.run_id,
                handle.current_step,
                handle.seconds_since_heartbeat(),
            )
            # Reset heartbeat first to prevent repeated interventions for
            # the same stall while we're awaiting the Judge.
            handle.reset_heartbeat()
            await self._handle_stall(handle)

    # Alias used in tests to trigger one supervision cycle without the sleep.
    _run_once = _poll

    async def _handle_stall(self, handle: LoopHandle) -> None:
        """Restore the last checkpoint and inject a recovery intervention."""
        cp = self._cp.restore_last(handle.run_id)
        cp_suffix = ""
        if cp is not None:
            cp_suffix = f" Last checkpoint is step {cp.step_num}."
        intervention = {
            "action": "reset_context",
            "category": "stalled_step",
            "message": (
                "Step timed out without heartbeat progress; restoring the last clean checkpoint."
                + cp_suffix
            ),
        }
        await handle.set_intervention(intervention)

    async def _monitor_throughput_slowdown(self) -> None:
        """Detect sustained low llama throughput and report/restart defensively."""
        if not self._slowdown_monitor_enabled or self._metrics_unsupported:
            return

        try:
            metrics_text = await self._client.get_metrics()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {404, 405, 501}:
                self._metrics_unsupported = True
                logger.info(
                    "Slowdown monitor disabled: llama.cpp /metrics endpoint returned %d",
                    status,
                )
            else:
                logger.debug("Slowdown monitor skipped (metrics unavailable): HTTP %d", status)
            self._slowdown_breach_count = 0
            return
        except Exception as exc:
            logger.debug("Slowdown monitor skipped (metrics unavailable): %s", exc.__class__.__name__)
            self._slowdown_breach_count = 0
            return

        snapshot = _extract_tps_from_metrics(metrics_text)
        if snapshot is None:
            self._slowdown_breach_count = 0
            return

        tps = snapshot.tokens_per_sec
        if tps >= self._slowdown_threshold_tps:
            self._slowdown_breach_count = 0
            return

        self._slowdown_breach_count += 1
        logger.warning(
            "SafetySupervisor slowdown sample %d/%d: %.2f tok/s (< %.2f) via %s",
            self._slowdown_breach_count,
            self._slowdown_consecutive_polls,
            tps,
            self._slowdown_threshold_tps,
            snapshot.metric_name,
        )

        if self._slowdown_breach_count < self._slowdown_consecutive_polls:
            return

        now = time.monotonic()
        in_cooldown = (now - self._last_slowdown_action_ts) < self._slowdown_cooldown_seconds
        if in_cooldown:
            return

        self._last_slowdown_action_ts = now
        self._slowdown_breach_count = 0

        logger.error(
            "Detected sustained llama slowdown: %.2f tok/s below threshold %.2f",
            tps,
            self._slowdown_threshold_tps,
        )

        if self._slowdown_restart_enabled and self._server_manager is not None:
            try:
                await self._server_manager.restart(actor="safety_supervisor_slowdown")
                logger.warning("SafetySupervisor triggered llama-server restart after slowdown detection")
            except Exception as exc:
                logger.error("Slowdown restart failed: %s: %s", exc.__class__.__name__, exc)

    # The former LLM Judge path has been intentionally removed.
