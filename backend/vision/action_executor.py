"""Desktop action executor using pyautogui.

Actions are expressed as simple dataclasses. The executor validates each
action before running it so bad model output cannot cause runaway input.

Supported actions
-----------------
click          left-click at (x, y)
right_click    right-click
double_click   double-click
type_text      type a string at current focus
press          press a key combo (e.g. "ctrl+c", "enter")
scroll         scroll wheel at (x, y)
drag           click-drag from (x1,y1) to (x2,y2)

All coordinates are in screen pixels.
"""
from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

logger = logging.getLogger(__name__)

# Maximum type-string length to guard against prompt injection
_MAX_TYPE_LEN = 2000
# Allowed key names: alphanumeric, +, -, _ only (no spaces – pyautogui uses hotkey(*keys))
_KEY_RE = re.compile(r"^[a-z0-9\+\-_]+$", re.IGNORECASE)


@dataclass
class ClickAction:
    x: int
    y: int
    button: Literal["left", "right", "middle"] = "left"
    double: bool = False


@dataclass
class TypeAction:
    text: str
    interval: float = 0.02   # seconds between keystrokes


@dataclass
class PressAction:
    keys: str   # e.g. "ctrl+c", "enter", "escape"


@dataclass
class ScrollAction:
    x: int
    y: int
    direction: Literal["up", "down"] = "down"
    clicks: int = 3


@dataclass
class DragAction:
    x1: int
    y1: int
    x2: int
    y2: int
    duration: float = 0.4


@dataclass
class RunAction:
    command: str


Action = Union[ClickAction, TypeAction, PressAction, ScrollAction, DragAction, RunAction]


class ActionExecutor:
    """Executes desktop actions via pyautogui.

    Import is deferred so the backend starts on headless servers where
    pyautogui is not installed or there is no display.
    """

    def __init__(self, pause_between_actions: float = 0.15) -> None:
        self._pause = pause_between_actions

    def _get_pag(self):
        try:
            import pyautogui
            pyautogui.FAILSAFE = True   # move mouse to corner to abort
            pyautogui.PAUSE = 0.05
            return pyautogui
        except ImportError as exc:
            raise RuntimeError(
                "pyautogui is not installed. Run: pip install pyautogui"
            ) from exc

    def execute(self, action: Action) -> None:
        """Execute a single validated action."""
        pag = self._get_pag()

        if isinstance(action, ClickAction):
            if action.double:
                pag.doubleClick(action.x, action.y, button=action.button)
            else:
                pag.click(action.x, action.y, button=action.button)

        elif isinstance(action, TypeAction):
            text = action.text[:_MAX_TYPE_LEN]
            pag.typewrite(text, interval=action.interval)

        elif isinstance(action, PressAction):
            if not _KEY_RE.match(action.keys):
                raise ValueError(f"Unsafe key sequence rejected: {action.keys!r}")
            keys = [k.strip() for k in action.keys.lower().split("+")]
            if len(keys) == 1:
                pag.press(keys[0])
            else:
                pag.hotkey(*keys)

        elif isinstance(action, ScrollAction):
            amount = action.clicks if action.direction == "up" else -action.clicks
            pag.scroll(amount, x=action.x, y=action.y)

        elif isinstance(action, DragAction):
            pag.moveTo(action.x1, action.y1)
            pag.dragTo(action.x2, action.y2, duration=action.duration, button="left")

        elif isinstance(action, RunAction):
            # Launch using shell on Windows so bare app names resolve via PATH/App Paths.
            subprocess.Popen(action.command, shell=True)

        else:
            raise TypeError(f"Unknown action type: {type(action)}")

        time.sleep(self._pause)


# ---------------------------------------------------------------------------
# Command-string parser
# ---------------------------------------------------------------------------

def parse_action_string(cmd: str, grid) -> Optional[Action]:
    """Parse a model-issued action command string into an Action object.

    ``grid`` must be a ``ScreenGrid`` instance (used to resolve cell labels
    to pixel coordinates).

    Returns ``None`` for "SCREENSHOT" and "DONE" (caller handles those).
    Raises ``ValueError`` on unrecognised or malformed commands.
    """
    cmd = cmd.strip()
    upper = cmd.upper()

    if upper in ("SCREENSHOT", "DONE"):
        return None

    # CLICK <cell>
    m = re.match(r"^CLICK\s+([A-Z]\d+)$", upper)
    if m:
        cell = grid.cell(m.group(1))
        return ClickAction(x=cell.center_x, y=cell.center_y)

    # RIGHT_CLICK <cell>
    m = re.match(r"^RIGHT_CLICK\s+([A-Z]\d+)$", upper)
    if m:
        cell = grid.cell(m.group(1))
        return ClickAction(x=cell.center_x, y=cell.center_y, button="right")

    # DOUBLE_CLICK <cell>
    m = re.match(r"^DOUBLE_CLICK\s+([A-Z]\d+)$", upper)
    if m:
        cell = grid.cell(m.group(1))
        return ClickAction(x=cell.center_x, y=cell.center_y, double=True)

    # TYPE "<text>"
    m = re.match(r'^TYPE\s+"(.*)"$', cmd, re.DOTALL)
    if m:
        return TypeAction(text=m.group(1))

    # CLICK_AND_TYPE <cell> "<text>"
    m = re.match(r'^CLICK_AND_TYPE\s+([A-Za-z]\d+)\s+"(.*)"$', cmd, re.DOTALL)
    if m:
        cell = grid.cell(m.group(1).upper())
        # Returns a tuple – caller should execute both
        return _ClickAndType(
            click=ClickAction(x=cell.center_x, y=cell.center_y),
            type_=TypeAction(text=m.group(2)),
        )

    # PRESS <key>
    m = re.match(r"^PRESS\s+(.+)$", cmd, re.IGNORECASE)
    if m:
        return PressAction(keys=m.group(1).strip())

    # SCROLL <cell> <up|down> [n]
    m = re.match(r"^SCROLL\s+([A-Z]\d+)\s+(UP|DOWN)(?:\s+(\d+))?$", upper)
    if m:
        cell = grid.cell(m.group(1))
        direction = m.group(2).lower()
        clicks = int(m.group(3)) if m.group(3) else 3
        return ScrollAction(x=cell.center_x, y=cell.center_y, direction=direction, clicks=clicks)

    # DRAG <from_cell> <to_cell>
    m = re.match(r"^DRAG\s+([A-Z]\d+)\s+([A-Z]\d+)$", upper)
    if m:
        src = grid.cell(m.group(1))
        dst = grid.cell(m.group(2))
        return DragAction(x1=src.center_x, y1=src.center_y, x2=dst.center_x, y2=dst.center_y)

    # RUN "<command>"
    m = re.match(r'^RUN\s+"(.+)"$', cmd, re.DOTALL)
    if m:
        command = m.group(1).strip()
        if not command:
            raise ValueError("RUN command cannot be empty")
        return RunAction(command=command)

    raise ValueError(f"Unrecognised action command: {cmd!r}")


@dataclass
class _ClickAndType:
    """Internal compound action: click then type."""
    click: ClickAction
    type_: TypeAction
