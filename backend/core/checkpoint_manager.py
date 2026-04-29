"""Checkpoint management for the agentic loop.

Before every major step the system saves a clean snapshot of the current task
state.  If the SafetySupervisor detects a stall, or the loop encounters an
unrecoverable error, it can restore the last good checkpoint instead of
continuing with a potentially corrupted context.

Checkpoints are stored in memory only; they are scoped to a single run and are
cleared once the run completes.

TODO: Add optional disk persistence so checkpoints survive backend restarts.
      Serialise each TaskCheckpoint to ``data/checkpoints/{run_id}/step_{n}.json``
      on save and load them on restore. See TODO.md Phase 2 and BUGS.md BUG-004.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Optional

logger = logging.getLogger(__name__)


class TaskCheckpoint:
    """Immutable snapshot of task state captured before executing a step.

    Attributes
    ----------
    task:
        The original task description (never changes).
    step_num:
        The step index (1-based) that was *about to be executed* when this
        checkpoint was taken.
    step_descriptions:
        The full ordered list of step descriptions as planned.
    context:
        Accumulated context lines from all *previously completed* steps.
    executed_steps:
        Serialised ``to_dict()`` output for each step completed so far.
    timestamp:
        ``time.monotonic()`` value at checkpoint creation.
    """

    def __init__(
        self,
        task: str,
        step_num: int,
        step_descriptions: list[str],
        context: list[str],
        executed_steps: list[dict],
        timestamp: float | None = None,
    ) -> None:
        self.task = task
        self.step_num = step_num
        self.step_descriptions = list(step_descriptions)
        self.context = list(context)
        self.executed_steps = list(executed_steps)
        self.timestamp = timestamp if timestamp is not None else time.monotonic()

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "step_num": self.step_num,
            "step_descriptions": self.step_descriptions,
            "context": self.context,
            "executed_steps": self.executed_steps,
            "timestamp": self.timestamp,
        }


class CheckpointManager:
    """Stores and restores agentic-run checkpoints in memory.

    Each run is identified by a ``run_id`` string (typically a UUID).  At most
    ``max_checkpoints_per_run`` checkpoints are kept; the oldest is evicted
    when the cap is exceeded so memory usage stays bounded.

    Parameters
    ----------
    max_checkpoints_per_run:
        Maximum number of checkpoints retained per run (default 20).
    """

    def __init__(self, max_checkpoints_per_run: int = 20) -> None:
        self._max = max_checkpoints_per_run
        # run_id → {step_num → TaskCheckpoint}
        self._store: dict[str, dict[int, TaskCheckpoint]] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        run_id: str,
        task: str,
        step_num: int,
        step_descriptions: list[str],
        context: list[str],
        executed_steps: list[dict],
    ) -> TaskCheckpoint:
        """Save a checkpoint immediately *before* executing ``step_num``.

        Returns the newly created ``TaskCheckpoint``.
        """
        cp = TaskCheckpoint(
            task=task,
            step_num=step_num,
            step_descriptions=deepcopy(step_descriptions),
            context=deepcopy(context),
            executed_steps=deepcopy(executed_steps),
        )
        run_store = self._store.setdefault(run_id, {})
        run_store[step_num] = cp

        # Evict oldest checkpoint when over cap
        if len(run_store) > self._max:
            oldest_key = min(run_store)
            del run_store[oldest_key]

        logger.debug("Checkpoint saved: run=%s step=%d", run_id, step_num)
        return cp

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def restore_last(self, run_id: str) -> Optional[TaskCheckpoint]:
        """Restore the most recent checkpoint for the given run.

        Returns ``None`` if no checkpoints exist for this run.
        """
        run_store = self._store.get(run_id)
        if not run_store:
            return None
        latest_key = max(run_store)
        cp = run_store[latest_key]
        logger.info(
            "Checkpoint restored (last): run=%s step=%d", run_id, cp.step_num
        )
        return cp

    def restore_step(
        self, run_id: str, step_num: int
    ) -> Optional[TaskCheckpoint]:
        """Restore the checkpoint for a specific step.

        Returns ``None`` if no checkpoint exists for that step.
        """
        run_store = self._store.get(run_id)
        if not run_store:
            return None
        cp = run_store.get(step_num)
        if cp:
            logger.info(
                "Checkpoint restored (step %d): run=%s", step_num, run_id
            )
        return cp

    def list_checkpoints(self, run_id: str) -> list[int]:
        """Return a sorted list of step numbers with saved checkpoints."""
        run_store = self._store.get(run_id, {})
        return sorted(run_store)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear(self, run_id: str) -> None:
        """Remove all checkpoints for a completed (or cancelled) run."""
        self._store.pop(run_id, None)
        logger.debug("Checkpoints cleared: run=%s", run_id)
