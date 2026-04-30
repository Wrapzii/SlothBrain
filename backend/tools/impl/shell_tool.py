"""Shell command execution tool.

Runs shell commands in a subprocess with a configurable timeout.
Commands are gated by an optional allowlist (configured via
``AppConfig.shell_allowlist``).  When the allowlist is empty the tool is
disabled unless ``AppConfig.allow_unrestricted_shell`` is ``True``.

All executions are emitted to the :class:`~backend.core.audit_log.AuditLog`.
"""
from __future__ import annotations

import asyncio
import logging
import shlex
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.core.audit_log import AuditLog
    from backend.config import AppConfig

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_MAX_OUTPUT_CHARS = 20_000


class ShellTool(Tool):
    """Execute an allowlisted shell command and return stdout/stderr.

    Requires ``AppConfig.allow_unrestricted_shell = True`` or the command
    prefix to appear in ``AppConfig.shell_allowlist``.
    """

    name = "shell"
    description = (
        "Execute a shell command and return its stdout and stderr output. "
        "Subject to allowlist restrictions configured by the operator."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "number",
                "description": f"Execution timeout in seconds (default: {_DEFAULT_TIMEOUT}).",
                "default": _DEFAULT_TIMEOUT,
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the command (optional).",
            },
        },
        "required": ["command"],
    }

    def __init__(self, config: "AppConfig", audit_log: "AuditLog | None" = None) -> None:
        self._config = config
        self._audit = audit_log

    def _is_allowed(self, command: str) -> bool:
        if getattr(self._config, "allow_unrestricted_shell", False):
            return True
        allowlist: list[str] = getattr(self._config, "shell_allowlist", [])
        if not allowlist:
            return False
        # Parse with shlex to extract only the executable name, preventing
        # bypass via shell metacharacters (e.g. 'git;rm -rf /').
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        if not tokens:
            return False
        executable = tokens[0].lower()
        return any(executable == prefix.lower() for prefix in allowlist)

    async def execute(
        self,
        command: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if not command:
            return ToolResult(ok=False, error="'command' argument is required")

        if not self._is_allowed(command):
            return ToolResult(
                ok=False,
                error=(
                    "Command not permitted. The shell tool requires an allowlist entry "
                    "or 'allow_unrestricted_shell' to be enabled."
                ),
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or None,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ToolResult(
                    ok=False,
                    error=f"Command timed out after {timeout}s: {command!r}",
                )

            stdout = stdout_b.decode(errors="replace")[:_MAX_OUTPUT_CHARS]
            stderr = stderr_b.decode(errors="replace")[:_MAX_OUTPUT_CHARS]
            returncode = proc.returncode

            if self._audit is not None:
                self._audit.record(
                    action="tool_shell",
                    actor="agent",
                    details=f"cmd={command[:100]!r} rc={returncode}",
                )

            return ToolResult(
                ok=returncode == 0,
                output={"stdout": stdout, "stderr": stderr, "returncode": returncode},
                error=f"Exit code {returncode}" if returncode != 0 else None,
            )
        except Exception as exc:
            logger.warning("ShellTool error: %s", exc)
            return ToolResult(ok=False, error=str(exc))
