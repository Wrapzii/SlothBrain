"""Scheduler tool — cron-style job scheduling for agent tasks.

Jobs are stored in a JSON file (``data/scheduler_jobs.json``) so they
survive restarts.  The scheduler runs a background asyncio task that polls
for due jobs every 60 seconds and triggers them via the provided callback.

Actions
-------
* ``add``    — create a new scheduled job.
* ``list``   — list all jobs.
* ``cancel`` — cancel a job by ID.
* ``status`` — get details of a specific job.
"""
from __future__ import annotations

import asyncio
import datetime as dt_module
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

_JOBS_FILE = Path("data/scheduler_jobs.json")
_POLL_INTERVAL = 60.0  # seconds
_DEFAULT_TIMEZONE = "America/New_York"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(s: str) -> datetime:
    # Python 3.11+ handles Z; use replace for 3.10 compat.
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _parse_daily_at(value: str) -> tuple[int, int]:
    parts = (value or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError("daily_at must be HH:MM")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("daily_at must be a valid 24-hour time, HH:MM")
    return hour, minute


def _next_daily_run_utc(daily_at: str, tz_name: str, *, after: datetime | None = None) -> str:
    tz = ZoneInfo(tz_name)
    hour, minute = _parse_daily_at(daily_at)
    base_utc = after or datetime.now(timezone.utc)
    local_now = base_utc.astimezone(tz)
    candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= local_now:
        candidate += dt_module.timedelta(days=1)
    return candidate.astimezone(timezone.utc).isoformat()


class SchedulerTool(Tool):
    """Schedule, list, and cancel cron-style agent tasks.

    Jobs are stored on disk and survive restarts.  Use the ``on_trigger``
    callback to define what happens when a job fires (typically calling
    the agentic loop).
    """

    name = "scheduler"
    description = (
        "Schedule, list, and cancel cron-style tasks that trigger automatically "
        "at a specified time or on a recurring interval."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "cancel", "status"],
                "description": "Scheduler operation to perform.",
            },
            "task": {
                "type": "string",
                "description": "Task description to run when the job fires (required for 'add').",
            },
            "run_at": {
                "type": "string",
                "description": (
                    "ISO 8601 datetime for a one-shot job (e.g. '2025-01-01T12:00:00Z')."
                ),
            },
            "interval_seconds": {
                "type": "number",
                "description": "Repeat interval in seconds for recurring jobs.",
            },
            "daily_at": {
                "type": "string",
                "description": "Repeat daily at a local 24-hour time, HH:MM, e.g. '08:00'.",
            },
            "timezone": {
                "type": "string",
                "description": "IANA timezone for daily_at, e.g. 'America/New_York'.",
            },
            "job_id": {
                "type": "string",
                "description": "Job ID (required for 'cancel' and 'status').",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        on_trigger: Callable[[str], Awaitable[None]] | None = None,
        jobs_file: Path | None = None,
        default_timezone: str = _DEFAULT_TIMEZONE,
    ) -> None:
        self._on_trigger = on_trigger
        self._jobs_file = jobs_file or _JOBS_FILE
        self._default_timezone = default_timezone or _DEFAULT_TIMEZONE
        self._jobs: dict[str, dict] = {}
        self._task: asyncio.Task | None = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._jobs_file.exists():
            try:
                self._jobs = json.loads(self._jobs_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load scheduler jobs: %s", exc)

    def _save(self) -> None:
        try:
            self._jobs_file.parent.mkdir(parents=True, exist_ok=True)
            self._jobs_file.write_text(
                json.dumps(self._jobs, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to save scheduler jobs: %s", exc)

    # ------------------------------------------------------------------
    # Background poll loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._poll_loop())

    def stop(self) -> None:
        """Stop the background polling loop."""
        if self._task and not self._task.done():
            self._task.cancel()

    async def _poll_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(_POLL_INTERVAL)
                await self._fire_due()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Scheduler poll error: %s", exc)

    async def _fire_due(self) -> None:
        now = datetime.now(timezone.utc)
        for job_id, job in list(self._jobs.items()):
            if job.get("cancelled"):
                continue
            if job.get("running"):
                continue
            run_at_str = job.get("next_run_at")
            if not run_at_str:
                continue
            try:
                run_at = _parse_iso(run_at_str)
            except Exception:
                continue
            if now >= run_at:
                logger.info("Scheduler firing job %s: %s", job_id, job.get("task", "")[:60])
                job["running"] = True
                job["last_run_at"] = now.isoformat()
                job["last_error"] = None
                interval = job.get("interval_seconds")
                daily_at = str(job.get("daily_at") or "").strip()
                if daily_at:
                    tz_name = str(job.get("timezone") or self._default_timezone)
                    try:
                        job["next_run_at"] = _next_daily_run_utc(daily_at, tz_name, after=now)
                    except Exception as exc:
                        job["cancelled"] = True
                        job["last_error"] = f"Failed to compute next daily run: {exc}"
                        job["running"] = False
                        self._save()
                        continue
                elif interval:
                    job["next_run_at"] = (now + dt_module.timedelta(seconds=interval)).isoformat()
                else:
                    job["cancelled"] = True
                self._save()

                if self._on_trigger:
                    try:
                        await self._on_trigger(job.get("task", ""))
                        job["consecutive_failures"] = 0
                    except Exception as exc:
                        failures = int(job.get("consecutive_failures") or 0) + 1
                        job["consecutive_failures"] = failures
                        job["last_error"] = str(exc)
                        if failures >= 3:
                            job["cancelled"] = True
                            job["cancel_reason"] = "Cancelled after 3 consecutive trigger failures"
                        logger.warning("Scheduler trigger error (job %s): %s", job_id, exc)
                job["running"] = False
                self._save()

    # ------------------------------------------------------------------
    # Tool execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        action: str = "",
        task: str = "",
        run_at: str = "",
        interval_seconds: float | None = None,
        daily_at: str = "",
        timezone: str = "",
        job_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        if action == "add":
            return self._add(task, run_at, interval_seconds, daily_at, timezone)
        if action == "list":
            return self._list()
        if action == "cancel":
            return self._cancel(job_id)
        if action == "status":
            return self._status(job_id)
        return ToolResult(ok=False, error=f"Unknown action: {action!r}")

    def _add(
        self,
        task: str,
        run_at: str,
        interval_seconds: float | None,
        daily_at: str,
        timezone_name: str,
    ) -> ToolResult:
        if not task:
            return ToolResult(ok=False, error="'task' is required for 'add'")
        if not run_at and not interval_seconds and not daily_at:
            return ToolResult(ok=False, error="Provide 'run_at', 'interval_seconds', or 'daily_at'")
        schedule_modes = sum(1 for value in (run_at, interval_seconds, daily_at) if value)
        if schedule_modes > 1:
            return ToolResult(ok=False, error="Provide only one of 'run_at', 'interval_seconds', or 'daily_at'")

        # Validate run_at if provided
        next_run = run_at
        if run_at:
            try:
                _parse_iso(run_at)
            except Exception:
                return ToolResult(ok=False, error=f"Invalid 'run_at' datetime: {run_at!r}")
        elif daily_at:
            tz_name = timezone_name or self._default_timezone
            try:
                next_run = _next_daily_run_utc(daily_at, tz_name)
            except ZoneInfoNotFoundError:
                return ToolResult(ok=False, error=f"Invalid timezone: {tz_name!r}")
            except Exception as exc:
                return ToolResult(ok=False, error=str(exc))
        elif interval_seconds:
            next_run = (datetime.now(timezone.utc) + dt_module.timedelta(seconds=interval_seconds)).isoformat()

        jid = str(uuid.uuid4())[:8]
        self._jobs[jid] = {
            "job_id": jid,
            "task": task,
            "created_at": _now_iso(),
            "next_run_at": next_run,
            "interval_seconds": interval_seconds,
            "daily_at": daily_at or None,
            "timezone": (timezone_name or self._default_timezone) if daily_at else None,
            "last_run_at": None,
            "cancelled": False,
            "running": False,
            "consecutive_failures": 0,
            "last_error": None,
        }
        self._save()
        return ToolResult(ok=True, output={"job_id": jid, "next_run_at": next_run})

    def _list(self) -> ToolResult:
        return ToolResult(ok=True, output={"jobs": list(self._jobs.values())})

    def _cancel(self, job_id: str) -> ToolResult:
        if not job_id:
            return ToolResult(ok=False, error="'job_id' is required for 'cancel'")
        if job_id not in self._jobs:
            return ToolResult(ok=False, error=f"Job {job_id!r} not found")
        self._jobs[job_id]["cancelled"] = True
        self._save()
        return ToolResult(ok=True, output={"cancelled": job_id})

    def _status(self, job_id: str) -> ToolResult:
        if not job_id:
            return ToolResult(ok=False, error="'job_id' is required for 'status'")
        job = self._jobs.get(job_id)
        if job is None:
            return ToolResult(ok=False, error=f"Job {job_id!r} not found")
        return ToolResult(ok=True, output=job)
