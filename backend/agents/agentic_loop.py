"""Agentic loop: orchestrates multi-step task execution with watcher monitoring.

The MainAgent plans and executes each step while the WatcherAgent observes every
result, provides course-correction feedback, and ultimately verifies that the
overall task has been completed.  An optional ``on_progress`` callback streams
structured events to callers so they can push real-time updates to clients.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from backend.agents.main_agent import MainAgent
    from backend.agents.watcher import WatcherAgent

logger = logging.getLogger(__name__)

# Maximum retries the watcher may request for a single step before moving on.
_MAX_STEP_RETRIES = 2


class AgenticStep:
    """Represents the state of one step in the agentic execution loop."""

    def __init__(self, step_num: int, description: str) -> None:
        self.step_num = step_num
        self.description = description
        self.result: str = ""
        self.watcher_feedback: str = ""
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
            "watcher_feedback": self.watcher_feedback,
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
       a. MainAgent **executes** the step using context from prior steps.
       b. An optional screenshot is captured.
       c. WatcherAgent **monitors** the result and returns one of:
          ``continue`` – proceed to the next step.
          ``retry``    – re-execute the current step (up to _MAX_STEP_RETRIES).
          ``done``     – task is complete; skip remaining steps.
          ``abort``    – something went wrong; stop immediately.
    3. WatcherAgent **verifies** the overall task is complete.

    Parameters
    ----------
    main_agent:
        The MainAgent instance responsible for planning and execution.
    watcher_agent:
        The WatcherAgent instance that monitors progress.
    max_steps:
        Hard cap on the number of steps that will be executed (default 10).
    screenshot_fn:
        Optional async callable that returns a dict with an
        ``annotated_png_b64`` key.  Called after each step execution.
    """

    def __init__(
        self,
        main_agent: "MainAgent",
        watcher_agent: "WatcherAgent",
        max_steps: int = 10,
        screenshot_fn: Optional[Callable[[], Awaitable[dict]]] = None,
    ) -> None:
        self._main = main_agent
        self._watcher = watcher_agent
        self._max_steps = max_steps
        self._screenshot_fn = screenshot_fn

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
        start_time = time.monotonic()

        async def emit(event_type: str, data: dict | None = None) -> None:
            if on_progress is None:
                return
            payload: dict = {"type": event_type, **(data or {})}
            try:
                await on_progress(payload)
            except Exception as exc:  # pragma: no cover
                logger.debug("Progress callback raised: %s", exc.__class__.__name__)

        await emit("start", {"task": task})

        # ── 1. Plan ──────────────────────────────────────────────────────────
        await emit("planning", {"task": task})
        try:
            plan = await self._main.plan_task(task)
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

        for idx, description in enumerate(step_descriptions, start=1):
            step = AgenticStep(step_num=idx, description=description)
            step.status = "running"

            await emit(
                "step_start",
                {
                    "step_num": idx,
                    "total_steps": len(step_descriptions),
                    "description": description,
                },
            )

            for attempt in range(_MAX_STEP_RETRIES + 1):
                # Execute
                try:
                    result = await self._main.execute_step(
                        step=description,
                        task=task,
                        context=context,
                    )
                    step.result = result
                except Exception as exc:
                    logger.error(
                        "execute_step error (step %d, attempt %d): %s",
                        idx,
                        attempt + 1,
                        exc,
                    )
                    step.result = f"Execution error: {exc.__class__.__name__}"

                # Optional screenshot
                if self._screenshot_fn is not None:
                    try:
                        shot = await self._screenshot_fn()
                        b64 = shot.get("annotated_png_b64") or shot.get("image_b64", "")
                        if b64:
                            step.screenshots.append(b64)
                    except Exception:
                        pass  # screenshots are best-effort

                # Watcher assessment
                try:
                    assessment = await self._watcher.monitor_step(
                        task=task,
                        step_description=description,
                        step_result=step.result,
                        step_num=idx,
                        total_steps=len(step_descriptions),
                        context=context,
                    )
                except Exception as exc:
                    logger.warning(
                        "monitor_step failed: %s", exc.__class__.__name__
                    )
                    assessment = {"action": "continue", "feedback": ""}

                step.watcher_feedback = assessment.get("feedback", "")
                final_action = assessment.get("action", "continue")

                await emit(
                    "step_monitored",
                    {
                        "step_num": idx,
                        "action": final_action,
                        "feedback": step.watcher_feedback,
                        "result_preview": step.result[:300],
                    },
                )

                if final_action == "abort":
                    break

                if final_action == "retry" and attempt < _MAX_STEP_RETRIES:
                    step.retries += 1
                    await emit(
                        "step_retry",
                        {
                            "step_num": idx,
                            "attempt": attempt + 2,
                            "feedback": step.watcher_feedback,
                        },
                    )
                    context.append(
                        f"Step {idx} retry feedback: {step.watcher_feedback}"
                    )
                    continue

                # "continue" or "done" – move on
                break

            step.status = "complete" if final_action != "abort" else "failed"
            step.finish()
            context.append(f"Step {idx} – {description}:\n{step.result}")
            executed.append(step)

            await emit("step_complete", step.to_dict())

            if final_action == "abort":
                await emit(
                    "aborted",
                    {"step_num": idx, "reason": step.watcher_feedback},
                )
                return _build_result(
                    task,
                    executed,
                    False,
                    f"Task aborted at step {idx}: {step.watcher_feedback}",
                    start_time,
                )

            if final_action == "done":
                # Watcher determined the task is already complete
                break

        # ── 3. Verify ────────────────────────────────────────────────────────
        await emit("verifying", {"steps_completed": len(executed)})
        try:
            verification = await self._watcher.verify_completion(
                task=task,
                steps_summary=context,
            )
            verified: bool = verification.get("complete", True)
            summary: str = verification.get("feedback", "Task complete.")
        except Exception as exc:
            logger.warning(
                "verify_completion failed: %s", exc.__class__.__name__
            )
            verified = len(executed) > 0
            summary = "Task execution complete."

        result = _build_result(task, executed, verified, summary, start_time)
        await emit(
            "complete",
            {
                "verified": verified,
                "summary": summary,
                "total_steps": len(executed),
                "duration_seconds": result["duration_seconds"],
            },
        )
        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_result(
    task: str,
    steps: list[AgenticStep],
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
