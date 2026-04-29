"""Safety supervisor for the agentic loop.

A lightweight *pure-Python* watchdog that monitors running ``AgenticLoop``
instances.  It does **not** depend on the LLM being healthy; it only calls the
Judge LLM opportunistically (best-effort) when a free slot is available.

Architecture
------------
- ``SafetySupervisor`` runs a single asyncio background task (``_run``) that
  polls every ``poll_interval`` seconds.
- Each loop run registers a ``LoopHandle`` with the supervisor at start-up and
  deregisters when it finishes.
- The loop calls ``LoopHandle.heartbeat()`` at the *start* of every step to
  signal liveness.
- If a step runs for longer than ``step_timeout`` seconds the supervisor marks
  it as stalled, restores the last checkpoint, and optionally calls the Judge.
- The Judge returns one of five actions:
    nudge             – send a reminder to the loop to continue
    reset_context     – clear accumulated context back to the checkpoint
    retry_step        – retry just the current step
    end_task          – stop the loop cleanly
    escalate_to_user  – stop and surface the situation to the human operator
- The loop checks for pending interventions via ``LoopHandle.pop_intervention()``
  and applies the recovery action immediately.

TODO: Wrap _call_judge in asyncio.wait_for to prevent a slow Judge from blocking
      the supervisor while other handles need servicing. See BUGS.md BUG-001.
TODO: Add a supervisor metrics endpoint (stall counts, intervention distribution).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from backend.core.checkpoint_manager import CheckpointManager, TaskCheckpoint
    from backend.core.llama_client import LlamaClient

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tuneable constants
# ──────────────────────────────────────────────────────────────────────────────

# Seconds a single step may run before being considered stalled.
_DEFAULT_STEP_TIMEOUT: float = 120.0

# How often the supervisor polls active loops.
_DEFAULT_POLL_INTERVAL: float = 15.0

# Valid Judge actions (sorted by ascending severity so the keyword scan below
# always matches the *least* severe option when multiple appear).
_VALID_JUDGE_ACTIONS = (
    "nudge",
    "reset_context",
    "retry_step",
    "end_task",
    "escalate_to_user",
)

_JUDGE_SYSTEM_PROMPT = (
    "You are an AI task supervisor. A running agent appears to be stuck or has "
    "produced an error. Review the recent history and checkpoint summary, then "
    "decide the best recovery action.\n\n"
    "You MUST respond with a single valid JSON object and nothing else:\n"
    '{"action": "<action>", "message": "<brief explanation>"}\n\n'
    "Valid action values (choose exactly one):\n"
    "  nudge            – send a gentle reminder to continue the current step\n"
    "  reset_context    – clear accumulated context and retry from the checkpoint\n"
    "  retry_step       – retry only the current step without clearing context\n"
    "  end_task         – the task cannot be completed; stop cleanly\n"
    "  escalate_to_user – the situation requires human input\n\n"
    "Severity order (prefer lower severity when uncertain): "
    "nudge < retry_step < reset_context < end_task < escalate_to_user"
)


# ──────────────────────────────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────────────────────────────


def _parse_judge_response(response: str) -> dict:
    """Extract action and message from the Judge LLM response.

    Tries JSON parsing first (preferred — the prompt requests JSON output).
    Falls back to regex key extraction and ultimately to a keyword scan so
    the supervisor never crashes on unexpected model output.

    Returns a dict with keys ``action`` and ``message``.
    """
    # ── Attempt 1: JSON parse ─────────────────────────────────────────────
    # The model may wrap JSON in a markdown code fence; try to extract it.
    # Fall back to the full stripped response if no fence/object is found.
    stripped = response.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        json_candidate = fence_match.group(1)
    else:
        obj_match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
        json_candidate = obj_match.group(0) if obj_match else stripped

    try:
        data = json.loads(json_candidate)
        action = str(data.get("action", "nudge")).strip().lower()
        if action not in _VALID_JUDGE_ACTIONS:
            action = "nudge"
        message = str(data.get("message", "")).strip()[:400]
        return {"action": action, "message": message}
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # ── Attempt 2: Regex key extraction ──────────────────────────────────
    lower = response.lower()
    action = "nudge"
    action_match = re.search(r'"?action"?\s*:\s*"?(\w+)"?', lower)
    if action_match:
        candidate = action_match.group(1).strip()
        if candidate in _VALID_JUDGE_ACTIONS:
            action = candidate
    else:
        # ── Attempt 3: Keyword scan in severity order ─────────────────────
        for keyword in _VALID_JUDGE_ACTIONS:
            if keyword.replace("_", " ") in lower or keyword in lower:
                action = keyword
                break

    message = response.strip()
    msg_match = re.search(
        r'"?message"?\s*:\s*"?(.+)"?', response, re.IGNORECASE | re.DOTALL
    )
    if msg_match:
        message = msg_match.group(1).strip().rstrip('"').strip()[:400]

    return {"action": action, "message": message}


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
        self._last_heartbeat = time.monotonic()
        self.current_step = step_num
        self.task = task
        # Keep only the 4 most recent context lines to bound memory
        self.recent_context = list(context[-4:]) if context else []

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
    ) -> None:
        self._client = llama_client
        self._cp = checkpoint_manager
        self._poll_interval = poll_interval
        self._step_timeout = step_timeout
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
        intervention = await self._call_judge(handle, cp)
        await handle.set_intervention(intervention)

    async def _call_judge(
        self,
        handle: LoopHandle,
        checkpoint: "Optional[TaskCheckpoint]",
    ) -> dict:
        """Ask the Judge LLM for a recovery decision.

        Constructs a structured prompt requesting JSON output, then parses the
        response with ``_parse_judge_response`` (JSON-first, regex fallback).

        Falls back to ``nudge`` on any error so the supervisor never crashes
        the loop.

        TODO: Add asyncio.wait_for with a configurable timeout (e.g. 30 s) to
              prevent a slow Judge from blocking supervision of other handles.
              See BUGS.md BUG-001.
        """
        context_str = "\n".join(handle.recent_context) or "(no context yet)"
        cp_info = ""
        if checkpoint:
            cp_info = (
                f"Last good checkpoint: step {checkpoint.step_num}, "
                f"{len(checkpoint.executed_steps)} step(s) completed."
            )

        prompt = (
            f"system: {_JUDGE_SYSTEM_PROMPT}\n\n"
            f"Task: {handle.task}\n"
            f"Currently on step: {handle.current_step}\n"
            f"Time without progress: {handle.seconds_since_heartbeat():.0f}s\n"
            f"{cp_info}\n"
            f"Recent context:\n{context_str}\n"
            'Respond with only a JSON object: {"action": "...", "message": "..."}\n'
            "assistant:"
        )

        try:
            # slot_id=-1 → llama.cpp picks any free slot (non-blocking)
            response = await self._client.complete(
                prompt=prompt,
                slot_id=-1,
                max_tokens=128,
                temperature=0.3,
            )
            decision = _parse_judge_response(response)
            logger.info(
                "Judge decision for run %s: %s – %s",
                handle.run_id,
                decision["action"],
                decision["message"][:80],
            )
            return decision
        except Exception as exc:
            logger.warning(
                "Judge call failed for run %s (%s); defaulting to nudge",
                handle.run_id,
                exc.__class__.__name__,
            )
            return {
                "action": "nudge",
                "message": "Step appears stalled; nudging to continue.",
            }
