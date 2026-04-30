"""Abstract base classes for SlothBrain tools.

Every tool the agent can call must subclass :class:`Tool` and implement
:meth:`Tool.execute`.  The agent runtime discovers tools through the
:class:`~backend.tools.registry.ToolRegistry`.

Tool call / result wire format
------------------------------
The model emits tool calls inside XML-ish fences:

    <tool_call>
    {"tool": "file", "args": {"action": "read", "path": "README.md"}}
    </tool_call>

The loop injects results back with:

    <tool_result>
    {"tool": "file", "ok": true, "output": "...file content..."}
    </tool_result>
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Standard return value for every tool call."""

    ok: bool
    output: Any = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "output": self.output, "error": self.error}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class Tool(ABC):
    """Abstract base for all SlothBrain tools.

    Subclasses must set class-level attributes:

    ``name``
        Short snake_case identifier used in tool calls (e.g. ``"file"``).
    ``description``
        One-sentence description shown to the model.
    ``parameters_schema``
        JSON Schema object describing the accepted keyword arguments.

    The ``execute`` method must be implemented and is called with keyword
    arguments matching the schema.  It must return a :class:`ToolResult`.
    """

    #: Tool identifier — must be unique within a registry.
    name: str = ""
    #: Short description visible to the model.
    description: str = ""
    #: JSON Schema for the ``args`` object passed in each tool call.
    parameters_schema: dict = field(default_factory=dict)

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the provided arguments.

        All arguments are passed as keyword arguments.  Implementations
        should validate required parameters and return a :class:`ToolResult`
        with ``ok=False`` and a descriptive ``error`` on failure rather than
        raising exceptions (callers do not handle exceptions from tools).
        """
