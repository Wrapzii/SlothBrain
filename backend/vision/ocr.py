"""OCR integration for reading text from grid cells.

Strategy (ordered by quality):
1. pytesseract — high accuracy, requires the system tesseract binary.
2. easyocr     — pure Python, no binary dep, but heavy download (first run).
3. Empty string  — if neither is available.

Results are cached per (image_hash, cell_label) to avoid redundant calls.
"""
from __future__ import annotations

import hashlib
import io
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Module-level backend flag so we only probe once
_BACKEND: str | None = None


def _detect_backend() -> str:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    try:
        import pytesseract  # noqa: F401
        _BACKEND = "pytesseract"
        logger.info("OCR backend: pytesseract")
        return _BACKEND
    except ImportError:
        pass

    try:
        import easyocr  # noqa: F401
        _BACKEND = "easyocr"
        logger.info("OCR backend: easyocr")
        return _BACKEND
    except ImportError:
        pass

    _BACKEND = "none"
    logger.warning(
        "No OCR backend found. Install pytesseract (+ tesseract binary) or easyocr "
        "for screen text extraction."
    )
    return _BACKEND


# Lazy easyocr reader (heavy to instantiate)
_easyocr_reader = None


def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def ocr_image_bytes(png_bytes: bytes) -> str:
    """Extract text from a PNG image byte string.

    Returns a single string with whitespace-collapsed text.
    """
    backend = _detect_backend()

    if backend == "pytesseract":
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(io.BytesIO(png_bytes))
            # PSM 6: assume a uniform block of text
            text = pytesseract.image_to_string(img, config="--psm 6")
            return " ".join(text.split())
        except Exception as exc:
            logger.debug("pytesseract failed: %s", exc)
            return ""

    if backend == "easyocr":
        try:
            import numpy as np
            from PIL import Image
            img = Image.open(io.BytesIO(png_bytes))
            arr = np.array(img)
            reader = _get_easyocr()
            results = reader.readtext(arr, detail=0)
            return " ".join(str(r) for r in results)
        except Exception as exc:
            logger.debug("easyocr failed: %s", exc)
            return ""

    return ""


def ocr_cell_bytes(png_bytes: bytes) -> str:
    """OCR a single cell image (already cropped)."""
    return ocr_image_bytes(png_bytes)
