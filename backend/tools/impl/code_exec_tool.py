"""Sandboxed Python code execution tool.

Runs Python snippets in an isolated ``exec`` environment with a restricted
set of built-ins.  stdout/stderr are captured via ``io.StringIO``.

Security notes
--------------
* Built-ins are restricted to a safe allowlist.
* ``import`` is blocked inside the sandboxed namespace but ``exec`` is
  not a true sandbox — do not run untrusted third-party code in
  production.  For fully untrusted code, run inside a container or use
  ``RestrictedPython`` / a similar library.
* Execution is run in a thread via ``asyncio.to_thread`` so it does not
  block the event loop.
"""
from __future__ import annotations

import asyncio
import builtins as _builtins_module
import io
import logging
import sys
import traceback
from typing import Any

from backend.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 20_000
_DEFAULT_TIMEOUT = 30.0

# Allowlisted built-in names for the sandboxed namespace.
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bin", "bool", "bytes", "callable", "chr",
    "dict", "dir", "divmod", "enumerate", "filter", "float", "format",
    "frozenset", "getattr", "hasattr", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next",
    "oct", "ord", "pow", "print", "range", "repr", "reversed", "round",
    "set", "setattr", "slice", "sorted", "str", "sum", "tuple", "type",
    "vars", "zip",
)

_SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(_builtins_module, name)
    for name in _SAFE_BUILTIN_NAMES
    if hasattr(_builtins_module, name)
}


def _run_code(code: str) -> dict:
    """Execute *code* in a restricted environment; return stdout/stderr/error."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf
    error: str | None = None
    try:
        exec(compile(code, "<tool_code>", "exec"), namespace)  # noqa: S102
    except SystemExit:
        error = "SystemExit called"
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return {
        "stdout": stdout_buf.getvalue()[:_MAX_OUTPUT_CHARS],
        "stderr": stderr_buf.getvalue()[:_MAX_OUTPUT_CHARS],
        "error": error,
    }


class CodeExecTool(Tool):
    """Execute a Python code snippet in a sandboxed environment.

    Returns ``stdout``, ``stderr``, and any exception traceback.
    The sandbox restricts built-ins to a safe allowlist.
    """

    name = "code_exec"
    description = (
        "Execute a Python code snippet in a sandboxed REPL and return stdout, "
        "stderr, and any exception traceback."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            },
            "timeout": {
                "type": "number",
                "description": f"Execution timeout in seconds (default: {_DEFAULT_TIMEOUT}).",
                "default": _DEFAULT_TIMEOUT,
            },
        },
        "required": ["code"],
    }

    async def execute(self, code: str = "", timeout: float = _DEFAULT_TIMEOUT, **kwargs: Any) -> ToolResult:
        if not code:
            return ToolResult(ok=False, error="'code' argument is required")

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_run_code, code),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult(ok=False, error=f"Code execution timed out after {timeout}s")
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))

        ok = result["error"] is None
        return ToolResult(
            ok=ok,
            output={"stdout": result["stdout"], "stderr": result["stderr"]},
            error=result["error"],
        )
