"""Screenshot tool — capture the desktop and return annotated state.

Wraps :class:`~backend.vision.controller.DesktopController` and moves
screenshot capability out of the agentic loop body into the tool system.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.vision.controller import DesktopController

logger = logging.getLogger(__name__)


class ScreenshotTool(Tool):
    """Capture the current desktop screen and return OCR-annotated state text.

    The returned ``output`` dict contains:
    ``state_text`` — the grid-annotated text description of the screen.
    ``annotated_png_b64`` — base64-encoded annotated PNG.
    ``width``, ``height`` — screen dimensions in pixels.
    """

    name = "screenshot"
    description = (
        "Capture the current desktop screen. Returns the annotated screen state "
        "as text (suitable for reasoning) and a base64-encoded PNG image."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "monitor": {
                "type": "integer",
                "description": (
                    "Monitor index to capture. 0 = virtual desktop (all monitors), "
                    "1..N = specific monitor."
                ),
                "default": 0,
            },
            "include_image": {
                "type": "boolean",
                "description": "Whether to include annotated_png_b64 in output.",
                "default": False,
            },
        },
        "required": [],
    }

    def __init__(self, controller: "DesktopController") -> None:
        self._controller = controller

    async def execute(self, monitor: int = 0, include_image: bool = False, **kwargs: Any) -> ToolResult:
        try:
            result = await asyncio.to_thread(
                self._controller.capture,
                monitor,
                include_image,
                True,
            )
            return ToolResult(
                ok=True,
                output={
                    "state_text": result.get("state_text", ""),
                    "annotated_png_b64": result.get("annotated_png_b64", ""),
                    "width": result.get("width", 0),
                    "height": result.get("height", 0),
                    "cols": result.get("cols", 0),
                    "rows": result.get("rows", 0),
                },
            )
        except Exception as exc:
            logger.warning("ScreenshotTool failed: %s", exc)
            return ToolResult(ok=False, error=str(exc))
