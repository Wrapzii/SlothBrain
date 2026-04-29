"""Fast screenshot capture using mss (preferred) with Pillow fallback."""
from __future__ import annotations

import io
import logging
import base64
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class ScreenInfo(NamedTuple):
    width: int
    height: int
    image_bytes: bytes   # PNG bytes


def capture_screen(monitor: int = 1) -> ScreenInfo:
    """Capture the primary monitor and return PNG bytes + dimensions.

    Uses ``mss`` if available, falls back to ``pyautogui.screenshot``.
    Raises ``RuntimeError`` if neither backend is available.
    """
    try:
        import mss
        import mss.tools
        with mss.mss() as sct:
            mon = sct.monitors[monitor]
            shot = sct.grab(mon)
            png = mss.tools.to_png(shot.rgb, shot.size)
            return ScreenInfo(
                width=shot.width,
                height=shot.height,
                image_bytes=png,
            )
    except ImportError:
        pass

    try:
        import pyautogui
        from PIL import Image
        pil_img = pyautogui.screenshot()
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return ScreenInfo(
            width=pil_img.width,
            height=pil_img.height,
            image_bytes=buf.getvalue(),
        )
    except ImportError:
        pass

    raise RuntimeError(
        "No screenshot backend available. Install 'mss' or 'pyautogui'."
    )


def screen_to_base64(info: ScreenInfo) -> str:
    return base64.b64encode(info.image_bytes).decode("utf-8")


def save_screenshot(info: ScreenInfo, path: Path) -> None:
    path.write_bytes(info.image_bytes)
