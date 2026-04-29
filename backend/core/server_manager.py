"""Manages the llama.cpp server process lifecycle."""
from __future__ import annotations

import asyncio
import json
import logging
import shlex
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from backend.config import AppConfig
    from backend.core.audit_log import AuditLog

logger = logging.getLogger(__name__)

BACKUPS_DIR = Path("data/backups")


class ServerManager:
    """Start, stop, and restart the llama-server process."""

    def __init__(self, config: "AppConfig", audit_log: "AuditLog") -> None:
        self._config = config
        self._audit = audit_log
        self._process: asyncio.subprocess.Process | None = None
        self._restart_times: deque[float] = deque()
        self._watchdog_task: asyncio.Task | None = None
        self._status: str = "stopped"  # "stopped" | "starting" | "running" | "restarting"
        BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> str:
        if self._process is not None and self._process.returncode is None:
            return "running"
        return self._status

    # ------------------------------------------------------------------
    # Rate-limit helpers
    # ------------------------------------------------------------------

    def _prune_restart_times(self) -> None:
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - 3600
        while self._restart_times and self._restart_times[0] < cutoff:
            self._restart_times.popleft()

    def _check_restart_rate(self) -> None:
        self._prune_restart_times()
        limit = self._config.max_restarts_per_hour
        if len(self._restart_times) >= limit:
            raise RuntimeError(
                f"Restart rate limit exceeded: {limit} restarts per hour maximum."
            )

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------

    def _backup_state(self) -> None:
        BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = BACKUPS_DIR / f"settings_{ts}.json"
        try:
            backup_path.write_text(
                json.dumps(self._config.model_dump(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("State backed up to %s", backup_path)
        except OSError as exc:
            logger.warning("Could not write backup: %s", exc)

    # ------------------------------------------------------------------
    # Process control
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            logger.info("llama-server is already running (pid=%d)", self._process.pid)
            return

        server_path = self._config.llama_server_path
        if not server_path:
            raise ValueError("llama_server_path is not configured.")

        args = self._config.llama_server_args or []
        cmd = [server_path, *args]

        logger.info("Starting llama-server: %s", shlex.join(cmd))
        self._status = "starting"
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._status = "running"
        self._audit.record(
            action="server_start",
            actor="system",
            details=f"pid={self._process.pid} cmd={shlex.join(cmd)}",
        )

    async def stop(self) -> None:
        if self._process is None or self._process.returncode is not None:
            self._status = "stopped"
            return
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=10)
        except asyncio.TimeoutError:
            self._process.kill()
            await self._process.wait()
        self._status = "stopped"
        self._audit.record(action="server_stop", actor="system")

    async def restart(self, actor: str = "system") -> None:
        self._check_restart_rate()
        self._backup_state()
        self._status = "restarting"
        self._restart_times.append(datetime.now(timezone.utc).timestamp())
        self._audit.record(action="server_restart", actor=actor)
        await self.stop()
        await asyncio.sleep(1)
        await self.start()

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    async def _watchdog(self) -> None:
        """Poll the llama-server /health endpoint; restart on failure."""
        base_url = f"http://{self._config.llama_host}:{self._config.llama_port}"
        while True:
            await asyncio.sleep(30)
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                    resp = await client.get(f"{base_url}/health")
                    resp.raise_for_status()
            except Exception as exc:
                logger.warning("Watchdog: llama-server health check failed: %s", exc)
                try:
                    await self.restart(actor="watchdog")
                except RuntimeError as rate_err:
                    logger.error("Watchdog restart blocked: %s", rate_err)

    def start_watchdog(self) -> None:
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog())

    def stop_watchdog(self) -> None:
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
