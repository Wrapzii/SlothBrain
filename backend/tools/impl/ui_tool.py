"""UI interaction tool — execute desktop action commands.

Wraps :class:`~backend.vision.controller.DesktopController` and exposes the
full action command language (CLICK, TYPE, DRAG, etc.) as a single Tool.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.vision.controller import DesktopController

logger = logging.getLogger(__name__)


class UITool(Tool):
    """Execute a desktop UI action command and optionally capture a new screenshot.

    ``command`` must be one of the supported action strings:

    * ``SCREENSHOT`` — capture current screen state
    * ``CLICK <cell>`` — left-click a grid cell (e.g. ``CLICK A3``)
    * ``RIGHT_CLICK <cell>``
    * ``DOUBLE_CLICK <cell>``
    * ``TYPE "<text>"``
    * ``CLICK_AND_TYPE <cell> "<text>"``
    * ``PRESS <key>`` — key combo (e.g. ``PRESS ctrl+c``)
    * ``SCROLL <cell> <UP|DOWN> [n]``
    * ``DRAG <from_cell> <to_cell>``
    * ``RUN "<command>"`` — launch an app/command via OS shell
    * ``DONE`` — signal task complete
    """

    name = "ui"
    description = (
        "Execute a desktop UI action command (CLICK, TYPE, DRAG, PRESS, SCROLL, "
        "SCREENSHOT, DONE). Optionally capture a new screenshot afterwards."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "The action command string, e.g. 'CLICK A3', 'TYPE \"hello\"', "
                    "'PRESS ctrl+c', 'SCREENSHOT', 'DONE'."
                ),
            },
            "capture_after": {
                "type": "boolean",
                "description": "Whether to take a screenshot after executing the command.",
                "default": False,
            },
            "monitor": {
                "type": "integer",
                "description": (
                    "Monitor index for follow-up screenshots. 0 = all monitors, "
                    "1..N = specific monitor."
                ),
                "default": 0,
            },
            "include_image": {
                "type": "boolean",
                "description": (
                    "Whether follow-up screenshots should include annotated_png_b64. "
                    "Defaults to false to keep payloads small."
                ),
                "default": False,
            },
        },
        "required": ["command"],
    }

    def __init__(self, controller: "DesktopController") -> None:
        self._controller = controller

    async def execute(
        self,
        command: str = "",
        capture_after: bool = False,
        monitor: int = 0,
        include_image: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        if not command:
            return ToolResult(ok=False, error="'command' argument is required")
        try:
            if capture_after:
                result = await asyncio.to_thread(
                    self._controller.execute_command_then_capture,
                    command,
                    monitor,
                    include_image,
                    True,
                )
            else:
                result = await asyncio.to_thread(
                    self._controller.execute_command, command
                )
            ok = result.get("executed", False)
            error = result.get("error")
            return ToolResult(ok=ok, output=result, error=error)
        except Exception as exc:
            logger.warning("UITool failed (command=%r): %s", command, exc)
            return ToolResult(ok=False, error=str(exc))
