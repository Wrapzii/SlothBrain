"""Agentic loop: orchestrates multi-step task execution.

Architecture
------------
The loop uses three collaborating components:

``MainAgent``
    Plans the task (splits it into ordered steps) and executes each step,
    carrying forward the accumulated context from prior steps.

``CheckpointManager``  *(optional)*
    Saves a clean snapshot of task state immediately before every step.  If
    the ``SafetySupervisor`` or an unrecoverable error forces a context reset,
    the loop restores from the last good checkpoint rather than corrupted state.

``SafetySupervisor``  *(optional)*
    An independent Python-level watchdog that polls the loop every
    ``poll_interval`` seconds.  When it detects that a step has been running
    for longer than ``step_timeout`` seconds it restores the last checkpoint
    and injects a *recovery intervention* via the ``LoopHandle``:

    nudge            – append a reminder and retry the step
    reset_context    – restore context from checkpoint, then retry the step
    retry_step       – retry the current step without touching context
    end_task         – abort the loop cleanly
    escalate_to_user – abort and surface the problem to the operator

An optional ``on_progress`` callback streams structured events to callers for
real-time client updates.

TODO: Add a tool-dispatch layer so ``MainAgent.execute_step`` can call
      registered tools (Python REPL, shell, file I/O) and feed results back
      into context automatically. See TODO.md Phase 3.
TODO: Emit a structured audit event if the loop exits with an unhandled
      exception, so the ``SafetySupervisor`` can detect the failure.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import time
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from backend.agents.main_agent import MainAgent
    from backend.core.checkpoint_manager import CheckpointManager
    from backend.core.diagnostic_recorder import DiagnosticRecorder
    from backend.core.safety_supervisor import LoopHandle, SafetySupervisor

logger = logging.getLogger(__name__)

# Maximum retries for a single step before moving on.
_MAX_STEP_RETRIES = 2

# Keep previews compact so debug events do not overwhelm clients/logs.
_MAX_PREVIEW_CHARS = 300


def _preview(text: str, limit: int = _MAX_PREVIEW_CHARS) -> str:
    value = text or ""
    if len(value) <= limit:
        return value
    return value[:limit] + f" ...[truncated {len(value) - limit} chars]"


@dataclass
class AgenticDebugOptions:
    """Runtime toggles for debug-loop execution.

    Set ``enabled=True`` and flip individual flags to isolate behavior.
    """

    enabled: bool = False
    llm_only: bool = False
    planning_enabled: bool = True
    rolling_context_enabled: bool = True
    tool_calls_enabled: bool = True
    semantic_routing_enabled: bool = True
    checkpointing_enabled: bool = True
    supervisor_enabled: bool = True
    per_event_logging: bool = True
    allowed_tools: list[str] = field(default_factory=list)


class AgenticStep:
    """Represents the state of one step in the agentic execution loop."""

    def __init__(self, step_num: int, description: str) -> None:
        self.step_num = step_num
        self.description = description
        self.result: str = ""
        self.status: str = "pending"  # pending | running | complete | failed
        self.screenshots: list[str] = []  # base64-encoded PNG strings
        self.retries: int = 0
        self._start: float = time.monotonic()
        self._end: float = 0.0

    def finish(self) -> None:
        self._end = time.monotonic()

    def to_dict(self) -> dict:
        duration = round(self._end - self._start, 2) if self._end else None
        return {
            "step_num": self.step_num,
            "description": self.description,
            "result": self.result,
            "status": self.status,
            "screenshots": self.screenshots,
            "retries": self.retries,
            "duration_seconds": duration,
        }


class AgenticLoop:
    """Orchestrates the full agentic task lifecycle.

    Flow
    ----
    1. MainAgent **plans** the task → ordered list of steps.
    2. For each step:
       a. Checkpoint is saved (before execution).
       b. Supervisor heartbeat is updated.
       c. Any pending supervisor intervention is applied first.
       d. MainAgent **executes** the step with accumulated context.
       e. Optional screenshot is captured.
         f. Supervisor observes step output and can intervene.
     3. Loop returns summary based on executed steps.

    Parameters
    ----------
    main_agent:
        The ``MainAgent`` instance responsible for planning and execution.
    max_steps:
        Hard cap on the number of steps executed (default 10).
    screenshot_fn:
        Optional async callable → dict with ``annotated_png_b64`` key.
    checkpoint_manager:
        Optional ``CheckpointManager``; checkpoints are skipped when absent.
    supervisor:
        Optional ``SafetySupervisor``; monitoring is skipped when absent.
    """

    def __init__(
        self,
        main_agent: "MainAgent",
        max_steps: int = 10,
        screenshot_fn: Optional[Callable[[], Awaitable[dict]]] = None,
        checkpoint_manager: Optional["CheckpointManager"] = None,
        supervisor: Optional["SafetySupervisor"] = None,
        debug_options: Optional[AgenticDebugOptions] = None,
        diagnostic: Optional["DiagnosticRecorder"] = None,
    ) -> None:
        self._main = main_agent
        self._max_steps = max_steps
        self._screenshot_fn = screenshot_fn
        self._cp = checkpoint_manager
        self._supervisor = supervisor
        self._debug = debug_options or AgenticDebugOptions()
        self._diagnostic = diagnostic

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        on_progress: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        """Execute a task through the full agentic loop.

        Parameters
        ----------
        task:
            Natural-language description of what should be accomplished.
        on_progress:
            Optional async callback invoked after each event.  Each call
            receives a dict with at least a ``type`` key.

        Returns
        -------
        dict
            Keys: ``task``, ``steps``, ``completion_verified``, ``summary``,
            ``total_steps``, ``duration_seconds``.
        """
        run_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # Inject the diagnostic recorder into the main agent so plan_task and
        # execute_step can record model-level events tied to this run_id.
        if self._diagnostic is not None:
            self._diagnostic.start_run(
                run_id,
                task=task,
                metadata={
                    "slot_id": self._main.slot_id,
                    "max_steps": self._max_steps,
                    "tool_calls_enabled": self._debug.tool_calls_enabled and not self._debug.llm_only,
                    "planning_enabled": self._debug.planning_enabled,
                    "rolling_context_enabled": self._debug.rolling_context_enabled,
                    "semantic_routing_enabled": self._debug.semantic_routing_enabled,
                    "checkpointing_enabled": self._debug.checkpointing_enabled,
                    "supervisor_enabled": self._debug.supervisor_enabled,
                    "allowed_tools": list(self._debug.allowed_tools or []),
                    "llm_only": self._debug.llm_only,
                },
            )
        # Give main_agent a reference to the recorder for model-level events.
        self._main.set_diagnostic_recorder(self._diagnostic)

        async def emit(event_type: str, data: dict | None = None) -> None:
            payload: dict = {"type": event_type, **(data or {})}
            if self._debug.per_event_logging:
                logger.info("agentic_event[%s] %s", run_id, _preview(str(payload)))
            if on_progress is None:
                return
            try:
                await on_progress(payload)
            except Exception as exc:  # pragma: no cover
                logger.debug("Progress callback raised: %s", exc.__class__.__name__)

        # Register with supervisor
        handle: Optional["LoopHandle"] = None
        if self._supervisor is not None and self._debug.supervisor_enabled:
            handle = self._supervisor.register(run_id)

        await emit("start", {"task": task, "run_id": run_id})

        # Initialised to empty dict so finish_run() always receives a valid
        # argument even when _execute raises before producing a result.
        result: dict = {}
        try:
            result = await self._execute(
                run_id=run_id,
                task=task,
                handle=handle,
                emit=emit,
                start_time=start_time,
            )
        except Exception as exc:
            if self._diagnostic is not None:
                self._diagnostic.record(
                    run_id,
                    "run_error",
                    error=exc.__class__.__name__,
                    message=str(exc)[:500],
                )
            raise
        finally:
            # Always deregister and clean up checkpoints
            if self._supervisor is not None and handle is not None:
                self._supervisor.deregister(run_id)
            if self._cp is not None and self._debug.checkpointing_enabled:
                self._cp.clear(run_id)
            # Flush diagnostic bundle to disk and detach from main agent.
            if self._diagnostic is not None:
                self._diagnostic.finish_run(run_id, result)
            self._main.set_diagnostic_recorder(None)

        return result

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _execute(
        self,
        run_id: str,
        task: str,
        handle: Optional["LoopHandle"],
        emit: Callable[[str, dict | None], Awaitable[None]],
        start_time: float,
    ) -> dict:
        # ── 1. Plan ──────────────────────────────────────────────────────────
        plan = {"steps": [task], "approach": "Direct execution"}
        if self._debug.llm_only:
            await emit(
                "planning_skipped",
                {"reason": "debug_llm_only", "steps": [task]},
            )
        elif not self._debug.planning_enabled:
            await emit(
                "planning_skipped",
                {"reason": "debug_planning_disabled", "steps": [task]},
            )
        else:
            await emit("planning", {"task": task})
            try:
                plan = await self._main.plan_task(task, run_id=run_id)
            except Exception as exc:
                logger.warning(
                    "plan_task failed (%s); falling back to single-step execution",
                    exc.__class__.__name__,
                )
                plan = {"steps": [task], "approach": "Direct execution"}

        step_descriptions: list[str] = plan.get("steps") or [task]
        if len(step_descriptions) > self._max_steps:
            step_descriptions = step_descriptions[: self._max_steps]

        await emit(
            "plan_ready",
            {
                "approach": plan.get("approach", ""),
                "steps": step_descriptions,
                "total_steps": len(step_descriptions),
            },
        )

        # ── 2. Execute ───────────────────────────────────────────────────────
        executed: list[AgenticStep] = []
        context: list[str] = []
        final_action = "continue"

        # Use an explicit index so supervisor reset_context can jump back.
        idx = 0
        while idx < len(step_descriptions):
            description = step_descriptions[idx]
            step_num = idx + 1  # 1-based for display/checkpoints

            # ── Checkpoint ───────────────────────────────────────────────
            if self._cp is not None and self._debug.checkpointing_enabled:
                self._cp.save(
                    run_id=run_id,
                    task=task,
                    step_num=step_num,
                    step_descriptions=step_descriptions,
                    context=context,
                    executed_steps=[s.to_dict() for s in executed],
                )
                if self._diagnostic is not None:
                    self._diagnostic.record(
                        run_id,
                        "checkpoint_saved",
                        step_num=step_num,
                    )

            # ── Supervisor heartbeat ──────────────────────────────────────
            if handle is not None:
                handle.heartbeat(
                    step_num=step_num,
                    task=task,
                    context=context,
                )
                await emit(
                    "heartbeat",
                    {
                        "step_num": step_num,
                        "context_items": len(context),
                    },
                )

            step = AgenticStep(step_num=step_num, description=description)
            step.status = "running"

            await emit(
                "step_start",
                {
                    "step_num": step_num,
                    "total_steps": len(step_descriptions),
                    "description": description,
                },
            )
            if self._diagnostic is not None:
                self._diagnostic.record(
                    run_id,
                    "step_start",
                    step_num=step_num,
                    description=description,
                )

            for attempt in range(_MAX_STEP_RETRIES + 1):
                had_execution_error = False
                await emit(
                    "loop_iteration",
                    {
                        "step_num": step_num,
                        "attempt": attempt + 1,
                        "max_attempts": _MAX_STEP_RETRIES + 1,
                    },
                )
                # ── Check for supervisor intervention ─────────────────────
                if handle is not None:
                    intervention = await handle.pop_intervention()
                    if intervention:
                        jump = await self._apply_intervention(
                            intervention=intervention,
                            step=step,
                            idx=idx,
                            run_id=run_id,
                            context=context,
                            executed=executed,
                            emit=emit,
                        )
                        if jump is not None:
                            # jump is the new idx to restart from
                            idx = jump
                            # Rebuild loop state from restored checkpoint
                            cp = (
                                self._cp.restore_last(run_id)
                                if self._cp is not None and self._debug.checkpointing_enabled
                                else None
                            )
                            if cp is not None:
                                context = list(cp.context)
                                # Rebuild executed list from checkpoint
                                executed = [
                                    _dict_to_step(s)
                                    for s in cp.executed_steps
                                ]
                                description = step_descriptions[idx]
                                step = AgenticStep(
                                    step_num=idx + 1,
                                    description=description,
                                )
                                step.status = "running"
                            break  # restart inner attempt loop

                        action = intervention.get("action", "nudge")
                        if action == "end_task":
                            final_action = "abort"
                            step.result = intervention.get(
                                "message", "Task ended by supervisor."
                            )
                            break
                        if action == "escalate_to_user":
                            final_action = "escalate"
                            step.result = intervention.get(
                                "message", "Escalated to user by supervisor."
                            )
                            break
                        if action == "nudge":
                            context.append(
                                f"Supervisor nudge: {intervention.get('message', '')}"
                            )
                            # Fall through to normal execution

                # ── Execute step ─────────────────────────────────────────
                async def _on_step_event(event: dict) -> dict | None:
                    et = event.get("type")

                    # Forward detailed execution events to UI/websocket clients.
                    if et in ("tool_call", "tool_result", "model_error"):
                        await emit(et, event)
                    elif et == "model_output":
                        await emit(
                            "model_output",
                            {"output_preview": str(event.get("output", ""))[:_MAX_PREVIEW_CHARS]},
                        )

                    if handle is None:
                        return None
                    detected: dict | None = None
                    if et == "model_output":
                        detected = handle.observe_model_output(
                            output=str(event.get("output", "")),
                            malformed=False,
                        )
                    elif et == "model_error":
                        detected = handle.observe_model_output(
                            output="",
                            malformed=True,
                        )
                    elif et == "tool_call":
                        detected = handle.observe_tool_call(
                            tool_name=str(event.get("tool", "")),
                            args=event.get("args") if isinstance(event.get("args"), dict) else {},
                        )
                    elif et == "tool_result":
                        detected = handle.observe_tool_result(
                            ok=bool(event.get("ok")),
                            output=event.get("output"),
                            error=event.get("error") if isinstance(event.get("error"), str) else None,
                        )

                    if detected:
                        await handle.set_intervention(detected)
                        return detected
                    return None

                try:
                    effective_context = context if self._debug.rolling_context_enabled else []
                    await emit(
                        "step_input",
                        {
                            "step_num": step_num,
                            "description": description,
                            "context_preview": _preview("\n".join(effective_context[-2:])),
                            "tool_calls_enabled": self._debug.tool_calls_enabled and not self._debug.llm_only,
                            "semantic_routing_enabled": self._debug.semantic_routing_enabled,
                            "rolling_context_enabled": self._debug.rolling_context_enabled,
                        },
                    )
                    result = await self._main.execute_step(
                        step=description,
                        task=task,
                        context=effective_context,
                        on_event=_on_step_event,
                        include_rolling_context=self._debug.rolling_context_enabled,
                        tool_calls_enabled=self._debug.tool_calls_enabled and not self._debug.llm_only,
                        semantic_routing_enabled=self._debug.semantic_routing_enabled,
                        allowed_tool_names=self._debug.allowed_tools or None,
                        run_id=run_id,
                        step_num=step_num,
                    )
                    step.result = result
                    await emit(
                        "step_output",
                        {
                            "step_num": step_num,
                            "output_preview": _preview(result),
                        },
                    )
                    if not (step.result or "").strip():
                        step.result = "Execution error: empty model response"
                except Exception as exc:
                    had_execution_error = True
                    logger.error(
                        "execute_step error (step %d, attempt %d): %s",
                        step_num,
                        attempt + 1,
                        exc,
                    )
                    if (
                        isinstance(exc, ValueError)
                        and "context window exceeded" in str(exc).lower()
                    ):
                        final_action = "abort"
                        step.result = (
                            "Execution aborted: llama.cpp context window exceeded. "
                            "Reduce context growth or reset task state."
                        )
                        break
                    step.result = f"Execution error: {exc.__class__.__name__}"

                if handle is not None:
                    post_detection = handle.observe_step_result(step.result)
                    if post_detection:
                        await handle.set_intervention(post_detection)

                # ── Optional screenshot ───────────────────────────────────
                if self._screenshot_fn is not None:
                    try:
                        shot = await self._screenshot_fn()
                        b64 = (
                            shot.get("annotated_png_b64")
                            or shot.get("image_b64", "")
                        )
                        if b64:
                            step.screenshots.append(b64)
                    except Exception:
                        pass  # screenshots are best-effort

                if final_action == "abort":
                    break

                if had_execution_error and attempt < _MAX_STEP_RETRIES:
                    step.retries += 1
                    await emit(
                        "step_retry",
                        {
                            "step_num": step_num,
                            "attempt": attempt + 2,
                            "feedback": "Retrying after execution error",
                        },
                    )
                    context.append(f"Step {step_num} retry after execution error.")
                    continue

                if final_action == "retry" and attempt < _MAX_STEP_RETRIES:
                    step.retries += 1
                    await emit(
                        "step_retry",
                        {
                            "step_num": step_num,
                            "attempt": attempt + 2,
                            "feedback": "Retry requested",
                        },
                    )
                    context.append(f"Step {step_num} retry requested.")
                    continue  # retry this step

                # continue – move to next step
                break

            if final_action in ("abort", "escalate"):
                step.status = "failed"
            else:
                step.status = "complete"

            step.finish()
            result_snippet = step.result
            if self._debug.rolling_context_enabled:
                context.append(f"Step {step_num} – {description}:\n{result_snippet}")
            executed.append(step)

            await emit("step_complete", step.to_dict())
            if self._diagnostic is not None:
                self._diagnostic.record(
                    run_id,
                    "step_complete",
                    step_num=step_num,
                    status=step.status,
                    retries=step.retries,
                    result_preview=_preview(step.result),
                )

            if final_action in ("abort", "escalate"):
                reason = step.result
                if final_action == "escalate":
                    await emit("escalated", {"step_num": step_num, "reason": reason})
                else:
                    await emit("aborted", {"step_num": step_num, "reason": reason})
                return _build_result(
                    task,
                    executed,
                    False,
                    f"Task stopped at step {step_num}: {reason}",
                    start_time,
                )

            idx += 1  # advance to next step

        # ── 3. Finalize ─────────────────────────────────────────────────────
        await emit("verifying", {"steps_completed": len(executed)})
        verified = len(executed) > 0 and all(s.status == "complete" for s in executed)
        summary = _derive_summary_from_steps(executed, verified)

        result_dict = _build_result(task, executed, verified, summary, start_time)
        if self._diagnostic is not None:
            self._diagnostic.record(
                run_id,
                "run_complete",
                verified=verified,
                total_steps=len(executed),
                duration_seconds=result_dict.get("duration_seconds"),
            )
        await emit(
            "complete",
            {
                "verified": verified,
                "summary": summary,
                "total_steps": len(executed),
                "duration_seconds": result_dict["duration_seconds"],
            },
        )
        return result_dict

    # ------------------------------------------------------------------
    # Supervisor intervention handler
    # ------------------------------------------------------------------

    async def _apply_intervention(
        self,
        intervention: dict,
        step: "AgenticStep",
        idx: int,
        run_id: str,
        context: list[str],
        executed: list["AgenticStep"],
        emit: Callable[[str, dict | None], Awaitable[None]],
    ) -> Optional[int]:
        """Apply a supervisor intervention.

        Returns the new loop index to jump to when a ``reset_context`` is
        requested (so the caller can restart from the checkpoint step), or
        ``None`` when execution should continue in the current step.
        """
        action = intervention.get("action", "nudge")
        message = intervention.get("message", "")

        await emit(
            "supervisor_intervention",
            {
                "step_num": step.step_num,
                "action": action,
                "message": message,
            },
        )
        if self._diagnostic is not None:
            self._diagnostic.record(
                run_id,
                "supervisor_intervention",
                step_num=step.step_num,
                action=action,
                message=message[:300],
            )
        logger.info(
            "Supervisor intervention (run=%s, step=%d): %s – %s",
            run_id,
            step.step_num,
            action,
            message,
        )

        if action == "reset_context":
            # Restore checkpoint and signal the loop to jump back
            cp = self._cp.restore_last(run_id) if self._cp is not None else None
            if cp is not None:
                new_idx = cp.step_num - 1  # 0-based
                if self._diagnostic is not None:
                    self._diagnostic.record(
                        run_id,
                        "checkpoint_restored",
                        restored_to_step=cp.step_num,
                        reason="supervisor_reset_context",
                    )
                await emit(
                    "context_reset",
                    {
                        "restored_to_step": cp.step_num,
                        "message": message,
                    },
                )
                return new_idx
            # No checkpoint – fall back to nudge
            context.append(f"Supervisor reset (no checkpoint): {message}")
            return None

        if action == "retry_step":
            # Emit the retry event so the UI shows it, but don't increment
            # step.retries is only incremented by the outer retry loop.
            await emit(
                "step_retry",
                {
                    "step_num": step.step_num,
                    "attempt": step.retries + 1,
                    "feedback": f"Supervisor: {message}",
                },
            )
            return None  # caller executes the step in the current attempt

        # nudge / end_task / escalate_to_user handled by caller
        return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_result(
    task: str,
    steps: "list[AgenticStep]",
    verified: bool,
    summary: str,
    start_time: float,
) -> dict:
    return {
        "task": task,
        "steps": [s.to_dict() for s in steps],
        "completion_verified": verified,
        "summary": summary,
        "total_steps": len(steps),
        "duration_seconds": round(time.monotonic() - start_time, 2),
    }


def _derive_summary_from_steps(steps: "list[AgenticStep]", verified: bool) -> str:
    """Prefer the last meaningful step output as the task summary."""
    for step in reversed(steps):
        result = (step.result or "").strip()
        if not result:
            continue
        if result in ("Task execution complete.", "Task execution finished with failures."):
            continue
        return result
    return "Task execution complete." if verified else "Task execution finished with failures."


def _dict_to_step(d: dict) -> "AgenticStep":
    """Reconstruct an ``AgenticStep`` from a ``to_dict()`` snapshot."""
    step = AgenticStep(
        step_num=d.get("step_num", 0),
        description=d.get("description", ""),
    )
    step.result = d.get("result", "")
    step.status = d.get("status", "complete")
    step.screenshots = d.get("screenshots", [])
    step.retries = d.get("retries", 0)
    step.finish()  # mark as done so duration is recorded
    return step

