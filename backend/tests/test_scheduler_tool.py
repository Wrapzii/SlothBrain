from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.tools.impl.scheduler_tool import SchedulerTool


@pytest.mark.asyncio
async def test_scheduler_adds_daily_job(tmp_path: Path) -> None:
    tool = SchedulerTool(jobs_file=tmp_path / "jobs.json", default_timezone="America/New_York")

    result = await tool.execute(
        action="add",
        task="research daily AI news",
        daily_at="08:00",
        timezone="America/New_York",
    )

    assert result.ok is True
    job_id = result.output["job_id"]
    payload = json.loads((tmp_path / "jobs.json").read_text(encoding="utf-8"))
    job = payload[job_id]
    assert job["task"] == "research daily AI news"
    assert job["daily_at"] == "08:00"
    assert job["timezone"] == "America/New_York"
    assert job["next_run_at"]


@pytest.mark.asyncio
async def test_scheduler_rejects_invalid_daily_timezone(tmp_path: Path) -> None:
    tool = SchedulerTool(jobs_file=tmp_path / "jobs.json")

    result = await tool.execute(
        action="add",
        task="research daily AI news",
        daily_at="08:00",
        timezone="Nope/Nowhere",
    )

    assert result.ok is False
    assert "Invalid timezone" in (result.error or "")
