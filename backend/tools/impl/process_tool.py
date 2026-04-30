"""Background process management tool.

Allows the agent to start, stop, and check the status of long-running
background processes.  Each process is assigned a handle ID that can be
used in subsequent calls.

All process starts are emitted to the :class:`~backend.core.audit_log.AuditLog`.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.core.audit_log import AuditLog
    from backend.config import AppConfig

logger = logging.getLogger(__name__)

_MAX_READ_CHARS = 10_000


class ProcessTool(Tool):
    """Start, stop, and inspect long-running background processes.

    Actions
    -------
    * ``start`` — launch a command and return a ``handle_id``.
    * ``stop`` — terminate the process by handle ID.
    * ``status`` — check if a process is still running.
    * ``read`` — read buffered stdout/stderr output.
    * ``list`` — list all managed processes.
    """

    name = "process"
    description = (
        "Manage long-running background processes: start, stop, check status, "
        "read output, or list all managed processes."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "status", "read", "list"],
                "description": "Operation to perform.",
            },
            "command": {
                "type": "string",
                "description": "Shell command to run (required for 'start').",
            },
            "handle_id": {
                "type": "string",
                "description": "Process handle ID (required for 'stop', 'status', 'read').",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the new process (optional).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, config: "AppConfig", audit_log: "AuditLog | None" = None) -> None:
        self._config = config
        self._audit = audit_log
        # handle_id → asyncio.subprocess.Process
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        # handle_id → buffered output strings
        self._stdout_buf: dict[str, list[str]] = {}
        self._stderr_buf: dict[str, list[str]] = {}

    def _is_allowed(self, command: str) -> bool:
        if getattr(self._config, "allow_unrestricted_shell", False):
            return True
        allowlist: list[str] = getattr(self._config, "shell_allowlist", [])
        if not allowlist:
            return False
        import shlex as _shlex
        try:
            tokens = _shlex.split(command)
        except ValueError:
            return False
        if not tokens:
            return False
        executable = tokens[0].lower()
        return any(executable == prefix.lower() for prefix in allowlist)

    async def execute(
        self,
        action: str = "",
        command: str = "",
        handle_id: str = "",
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if action == "start":
            return await self._start(command, cwd)
        if action == "stop":
            return self._stop(handle_id)
        if action == "status":
            return self._status(handle_id)
        if action == "read":
            return self._read(handle_id)
        if action == "list":
            return self._list()
        return ToolResult(ok=False, error=f"Unknown action: {action!r}")

    async def _start(self, command: str, cwd: str | None) -> ToolResult:
        if not command:
            return ToolResult(ok=False, error="'command' is required for 'start'")
        if not self._is_allowed(command):
            return ToolResult(
                ok=False,
                error="Command not permitted by shell allowlist.",
            )
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            handle = str(uuid.uuid4())[:8]
            self._processes[handle] = proc
            self._stdout_buf[handle] = []
            self._stderr_buf[handle] = []

            if self._audit is not None:
                self._audit.record(
                    action="tool_process_start",
                    actor="agent",
                    details=f"handle={handle} cmd={command[:80]!r}",
                )

            return ToolResult(ok=True, output={"handle_id": handle, "pid": proc.pid})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

    def _stop(self, handle_id: str) -> ToolResult:
        proc = self._processes.get(handle_id)
        if proc is None:
            return ToolResult(ok=False, error=f"No process with handle_id {handle_id!r}")
        try:
            proc.kill()
            return ToolResult(ok=True, output={"handle_id": handle_id, "killed": True})
        except ProcessLookupError:
            return ToolResult(ok=True, output={"handle_id": handle_id, "killed": False, "note": "Already finished"})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

    def _status(self, handle_id: str) -> ToolResult:
        proc = self._processes.get(handle_id)
        if proc is None:
            return ToolResult(ok=False, error=f"No process with handle_id {handle_id!r}")
        returncode = proc.returncode  # None if still running
        return ToolResult(
            ok=True,
            output={
                "handle_id": handle_id,
                "running": returncode is None,
                "returncode": returncode,
                "pid": proc.pid,
            },
        )

    def _read(self, handle_id: str) -> ToolResult:
        if handle_id not in self._processes:
            return ToolResult(ok=False, error=f"No process with handle_id {handle_id!r}")
        # Drain any buffered output (non-blocking peek)
        proc = self._processes[handle_id]
        stdout_chunks = self._stdout_buf.get(handle_id, [])
        stderr_chunks = self._stderr_buf.get(handle_id, [])
        return ToolResult(
            ok=True,
            output={
                "handle_id": handle_id,
                "stdout": "".join(stdout_chunks)[-_MAX_READ_CHARS:],
                "stderr": "".join(stderr_chunks)[-_MAX_READ_CHARS:],
            },
        )

    def _list(self) -> ToolResult:
        result = []
        for hid, proc in self._processes.items():
            result.append({
                "handle_id": hid,
                "pid": proc.pid,
                "running": proc.returncode is None,
                "returncode": proc.returncode,
            })
        return ToolResult(ok=True, output={"processes": result})
