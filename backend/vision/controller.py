"""High-level desktop controller.

Captures the screen, builds a text description the model can reason about,
and executes model-issued action commands.

Screen-state text format
------------------------
SCREEN_STATE (WxH | C cols × R rows | active_window):
<label> [<ocr_text>]
<label> [<ocr_text>]
...

The model issues single-line action commands and we execute them one by one,
taking a fresh screenshot between each step.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from backend.vision.action_executor import (
    ActionExecutor,
    _ClickAndType,
    parse_action_string,
)
from backend.vision.grid import ScreenGrid
from backend.vision.ocr import _detect_backend, ocr_cell_bytes
from backend.vision.screen_capture import ScreenInfo, capture_screen

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Cells with very short text (e.g. a single blank) are omitted from
# the description to keep the prompt compact.
_MIN_CELL_TEXT_LEN = 2
# Brief pause between a click and the subsequent type action.
# Needed so the OS has time to focus the clicked element before keystrokes arrive.
_CLICK_TO_TYPE_DELAY = 0.08  # seconds


class DesktopController:
    """Captures screen state as text + executes action strings from the model.

    Parameters
    ----------
    cols, rows:
        Grid dimensions. 10×8 is a good default for 1080p.
    screenshot_delay:
        Seconds to wait after an action before taking the next screenshot.
    ocr_enabled:
        Whether to OCR cell content.  Can be disabled for speed / headless
        environments that only need coordinate-based control.
    """

    def __init__(
        self,
        cols: int = 10,
        rows: int = 8,
        screenshot_delay: float = 0.5,
        ocr_enabled: bool = True,
    ) -> None:
        self._cols = cols
        self._rows = rows
        self._delay = screenshot_delay
        self._ocr = ocr_enabled
        self._executor = ActionExecutor()
        self._last_screen: Optional[ScreenInfo] = None
        self._last_grid: Optional[ScreenGrid] = None

    def capabilities(self) -> dict:
        """Return effective runtime capabilities for desktop control."""
        import importlib.util

        screenshot_backend = "none"
        if importlib.util.find_spec("mss") is not None:
            screenshot_backend = "mss"
        elif importlib.util.find_spec("pyautogui") is not None:
            screenshot_backend = "pyautogui"

        input_available = importlib.util.find_spec("pyautogui") is not None
        ocr_backend = _detect_backend()
        ocr_available = ocr_backend != "none"

        return {
            "screenshot_backend": screenshot_backend,
            "input_available": input_available,
            "ocr_backend": ocr_backend,
            "ocr_available": ocr_available,
            # This implementation is text/OCR driven only; no image-to-model path exists yet.
            "multimodal_available": False,
            "vision_run_supported": ocr_available,
        }

    # ------------------------------------------------------------------
    # Screen capture helpers
    # ------------------------------------------------------------------

    def capture(
        self,
        monitor: int = 0,
        include_image: bool = True,
        include_cells: bool = True,
    ) -> dict:
        """Capture the screen and return a structured dict.

        Keys:
          width, height, cols, rows, annotated_png_b64, state_text, cells
        """
        import base64

        info = capture_screen(monitor=monitor)
        self._last_screen = info
        grid = ScreenGrid(info.width, info.height, self._cols, self._rows)
        self._last_grid = grid

        cells: dict[str, str] = {}
        if self._ocr and include_cells:
            for cell in grid.all_cells():
                region_png = grid.extract_cell_image(info.image_bytes, cell.label)
                text = ocr_cell_bytes(region_png)
                if len(text) >= _MIN_CELL_TEXT_LEN:
                    cells[cell.label] = text

        state_text = self._build_state_text(info, grid, cells)

        annotated_b64 = ""
        if include_image:
            try:
                annotated = grid.annotate_image(info.image_bytes)
                annotated_b64 = base64.b64encode(annotated).decode()
            except Exception:
                annotated_b64 = base64.b64encode(info.image_bytes).decode()

        return {
            "width": info.width,
            "height": info.height,
            "cols": self._cols,
            "rows": self._rows,
            "annotated_png_b64": annotated_b64,
            "state_text": state_text,
            "cells": cells,
        }

    def _build_state_text(
        self,
        info: ScreenInfo,
        grid: ScreenGrid,
        cells: dict[str, str],
    ) -> str:
        active = _get_active_window_title()
        header = (
            f"SCREEN_STATE ({info.width}×{info.height} | "
            f"{grid.cols} cols × {grid.rows} rows | active: {active!r})"
        )
        lines = [header]
        for cell in grid.all_cells():
            text = cells.get(cell.label, "")
            if text:
                lines.append(f"{cell.label} [{text}]")
        if len(lines) == 1:
            lines.append("(no text detected in any cell)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def execute_command(self, cmd: str) -> dict:
        """Parse and execute a single action command string.

        Returns a dict with keys: command, executed (bool), error (str|None).
        """
        cmd = cmd.strip()
        upper = cmd.upper()

        if upper == "SCREENSHOT":
            return {"command": cmd, "executed": True, "error": None, "type": "screenshot"}
        if upper == "DONE":
            return {"command": cmd, "executed": True, "error": None, "type": "done"}

        if self._last_grid is None:
            return {"command": cmd, "executed": False, "error": "No screenshot taken yet. Issue SCREENSHOT first.", "type": "error"}

        try:
            action = parse_action_string(cmd, self._last_grid)
        except (ValueError, KeyError) as exc:
            return {"command": cmd, "executed": False, "error": str(exc), "type": "parse_error"}

        try:
            if isinstance(action, _ClickAndType):
                self._executor.execute(action.click)
                time.sleep(_CLICK_TO_TYPE_DELAY)
                self._executor.execute(action.type_)
            elif action is not None:
                self._executor.execute(action)
            return {"command": cmd, "executed": True, "error": None, "type": type(action).__name__}
        except Exception as exc:
            logger.exception("Action execution failed: %s", cmd)
            return {"command": cmd, "executed": False, "error": str(exc), "type": "exec_error"}

    def execute_command_then_capture(
        self,
        cmd: str,
        monitor: int = 0,
        include_image: bool = False,
        include_cells: bool = True,
    ) -> dict:
        """Execute ``cmd``, wait, then take a fresh screenshot.

        Returns ``execute_command`` result merged with new ``capture()`` data.
        """
        result = self.execute_command(cmd)
        if result.get("type") not in ("parse_error", "error", "exec_error"):
            time.sleep(self._delay)
            screen = self.capture(
                monitor=monitor,
                include_image=include_image,
                include_cells=include_cells,
            )
            result["screen"] = screen
        return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_active_window_title() -> str:
    """Return the active window title, best-effort across platforms."""
    try:
        # Windows
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if hwnd:
            length = user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            if title:
                return title
    except Exception:
        pass
    try:
        # Linux (X11)
        import subprocess
        out = subprocess.check_output(
            ["xdotool", "getactivewindow", "getwindowname"],
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        pass
    try:
        # macOS
        import subprocess
        script = (
            'tell application "System Events" to '
            'get name of first application process whose frontmost is true'
        )
        out = subprocess.check_output(
            ["osascript", "-e", script], timeout=2, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        pass
    return "unknown"
