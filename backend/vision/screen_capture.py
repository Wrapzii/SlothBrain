"""Fast screenshot capture using mss (preferred) with Pillow fallback."""
from __future__ import annotations

import io
import logging
import base64
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Maximum dimension for screenshot to reduce memory and network overhead
_MAX_SCREENSHOT_WIDTH = 1920
_MAX_SCREENSHOT_HEIGHT = 1080
# JPEG quality for compression (0-100; 70 is good for model vision tasks)
_JPEG_QUALITY = 70


class ScreenInfo(NamedTuple):
    width: int
    height: int
    image_bytes: bytes   # JPEG bytes (compressed for efficient transmission)


def capture_screen(monitor: int = 0) -> ScreenInfo:
    """Capture a monitor and return JPEG bytes + dimensions.

    Automatically scales down large screenshots and compresses to JPEG to reduce
    transmission size and GPU memory usage.

    With ``mss``, ``monitor=0`` captures the virtual desktop (all monitors),
    while ``monitor>=1`` captures a specific monitor index.

    Uses ``mss`` if available, falls back to ``pyautogui.screenshot``.
    Raises ``RuntimeError`` if neither backend is available.
    """
    try:
        import mss
        from PIL import Image
        with mss.mss() as sct:
            monitors = sct.monitors
            if not monitors:
                raise RuntimeError("No monitors detected for screenshot capture.")
            if monitor < 0 or monitor >= len(monitors):
                raise ValueError(
                    f"Invalid monitor index {monitor}. Valid range is 0..{len(monitors) - 1}."
                )
            mon = monitors[monitor]
            shot = sct.grab(mon)
            pil_img = Image.frombytes("RGB", shot.size, shot.rgb)
            
            if pil_img.width > _MAX_SCREENSHOT_WIDTH or pil_img.height > _MAX_SCREENSHOT_HEIGHT:
                scale = min(_MAX_SCREENSHOT_WIDTH / pil_img.width, _MAX_SCREENSHOT_HEIGHT / pil_img.height)
                new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
            return ScreenInfo(
                width=pil_img.width,
                height=pil_img.height,
                image_bytes=buf.getvalue(),
            )
    except ImportError:
        pass

    try:
        import pyautogui
        from PIL import Image
        pil_img = pyautogui.screenshot()
        
        if pil_img.width > _MAX_SCREENSHOT_WIDTH or pil_img.height > _MAX_SCREENSHOT_HEIGHT:
            scale = min(_MAX_SCREENSHOT_WIDTH / pil_img.width, _MAX_SCREENSHOT_HEIGHT / pil_img.height)
            new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
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
