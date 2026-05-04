"""Image analysis tool — analyze an image with OCR and/or local llama.cpp vision.

The tool accepts a base64-encoded image (or captures a live screenshot) and
uses the configured backend:

* ``cpu_ocr`` — local OCR only.
* ``llama`` — local llama.cpp multimodal chat-completions only.
* ``auto`` — llama.cpp vision when the loaded model advertises vision,
  otherwise CPU OCR fallback.
"""
from __future__ import annotations

import base64
import io
import logging
import re
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.core.llama_client import LlamaClient

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = (
    "Describe this image in detail. Include any text visible on screen, "
    "the layout, and any notable UI elements or content."
)

_THINKING_HEADING_RE = re.compile(r"(?im)^\s*(?:thinking process|analysis)\s*:\s*")
_REASONING_SECTION_RE = re.compile(
    r"(?im)^\s*\d+\.\s+\*\*(?:analy[sz]e the request|construct(?: the)? response|final answer)[^\n]*\n?"
)


class ImageAnalysisTool(Tool):
    """Analyse an image using a vision-capable model.

    Pass either:
    * ``image_b64`` — a base64-encoded PNG/JPEG string, **or**
    * ``screenshot`` — ``true`` to capture the current screen first.

    Returns a textual description of the image.
    """

    name = "image_analysis"
    description = (
        "Analyze an image or live screenshot with local OCR and/or llama.cpp vision. "
        "Can describe the screen or help locate visual UI elements."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "image_b64": {
                "type": "string",
                "description": "Base64-encoded PNG or JPEG image data.",
            },
            "screenshot": {
                "type": "boolean",
                "description": "Capture a live screenshot and analyse it.",
                "default": False,
            },
            "prompt": {
                "type": "string",
                "description": "Instruction to the vision model (optional).",
            },
            "mode": {
                "type": "string",
                "enum": ["describe", "ground"],
                "description": (
                    "'describe' returns a screen description. 'ground' asks for "
                    "JSON coordinates/bounding box for the requested target."
                ),
                "default": "describe",
            },
        },
        "required": [],
    }

    def __init__(
        self,
        llama_client: "LlamaClient",
        controller: Any = None,
        backend: str = "cpu_ocr",
        llama_slot_id: int = 0,
        cpu_max_text_chars: int = 4000,
    ) -> None:
        self._client = llama_client
        self._controller = controller
        self._backend = (backend or "cpu_ocr").strip().lower()
        self._llama_slot_id = int(llama_slot_id)
        self._cpu_max_text_chars = max(256, int(cpu_max_text_chars))

    async def _llama_vision_available(self) -> tuple[bool, dict]:
        try:
            props = await self._client.get_props()
        except Exception as exc:
            return False, {"error": str(exc)}
        modalities = props.get("modalities")
        has_vision = bool(isinstance(modalities, dict) and modalities.get("vision"))
        return has_vision, props

    @staticmethod
    def _decode_image_b64(image_b64: str):
        payload = (image_b64 or "").strip()
        if "," in payload and payload.lower().startswith("data:image"):
            payload = payload.split(",", 1)[1]
        raw = base64.b64decode(payload, validate=False)

        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        img.load()
        return img

    def _analyze_cpu(self, image_b64: str, prompt: str) -> str:
        img = self._decode_image_b64(image_b64)

        width, height = img.size
        mode = img.mode
        fmt = img.format or "unknown"

        ocr_text = ""
        try:
            import pytesseract

            ocr_text = pytesseract.image_to_string(img) or ""
        except Exception as exc:
            logger.info("CPU OCR unavailable or failed: %s", exc)

        ocr_text = " ".join(ocr_text.split())
        if len(ocr_text) > self._cpu_max_text_chars:
            ocr_text = ocr_text[: self._cpu_max_text_chars] + " ...[truncated]"

        lines = [
            "CPU image analysis result:",
            f"- format: {fmt}",
            f"- size: {width}x{height}",
            f"- color mode: {mode}",
        ]
        if prompt:
            lines.append(f"- prompt considered: {prompt}")
        if ocr_text:
            lines.append("- extracted text:")
            lines.append(ocr_text)
        else:
            lines.append("- extracted text: [none detected]")
        return "\n".join(lines)

    @staticmethod
    def _clean_llama_response(text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = _THINKING_HEADING_RE.sub("", cleaned).strip()
        cleaned = _REASONING_SECTION_RE.sub("", cleaned)
        cleaned = cleaned.replace("**", "")
        cleaned = re.sub(r"(?m)^\s*\*\s+", "- ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    async def _analyze_llama(self, image_b64: str, prompt: str, *, mime_type: str = "image/png") -> str:
        response = await self._client.complete_with_image(
            prompt=prompt,
            image_b64=image_b64,
            max_tokens=1024,
            temperature=0.2,
            mime_type=mime_type,
        )
        return self._clean_llama_response(response)

    async def execute(
        self,
        image_b64: str = "",
        screenshot: bool = False,
        prompt: str = "",
        mode: str = "describe",
        **kwargs: Any,
    ) -> ToolResult:
        import asyncio

        # Obtain image bytes
        b64 = image_b64
        llama_b64 = image_b64
        mime_type = "image/png"
        llama_mime_type = "image/png"
        if screenshot and self._controller is not None:
            try:
                cap = await asyncio.to_thread(self._controller.capture)
                b64 = cap.get("image_b64") or cap.get("annotated_png_b64", "")
                llama_b64 = cap.get("annotated_png_b64") or b64
                mime_type = str(cap.get("image_mime_type") or "image/png")
                llama_mime_type = "image/png" if cap.get("annotated_png_b64") else mime_type
            except Exception as exc:
                return ToolResult(ok=False, error=f"Screenshot failed: {exc}")

        if not b64:
            return ToolResult(
                ok=False,
                error="No image provided. Pass 'image_b64' or set 'screenshot': true.",
            )

        mode = (mode or "describe").strip().lower()
        analysis_prompt = prompt or _DEFAULT_PROMPT
        if mode == "ground":
            target = analysis_prompt
            analysis_prompt = (
                "Find the requested visual target in this screenshot/image. "
                "Return only a compact JSON object with keys: "
                "target, found, x, y, bbox, confidence, reason. "
                "Use pixel coordinates relative to the provided image. "
                f"Requested target: {target}"
            )
        backend = self._backend

        if backend == "auto":
            llama_available, props = await self._llama_vision_available()
            if llama_available:
                try:
                    output = await self._analyze_llama(
                        llama_b64,
                        analysis_prompt,
                        mime_type=llama_mime_type,
                    )
                    if output.strip():
                        model_name = str(props.get("model_name") or props.get("model_alias") or "local")
                        return ToolResult(ok=True, output=f"llama.cpp vision ({model_name}):\n{output}")
                except Exception as exc:
                    logger.info("llama.cpp vision failed; falling back to CPU OCR: %s", exc)
            try:
                return ToolResult(ok=True, output=self._analyze_cpu(b64, analysis_prompt))
            except Exception as exc:
                return ToolResult(ok=False, error=f"Image analysis failed: {exc}")

        if backend == "cpu_ocr":
            try:
                return ToolResult(ok=True, output=self._analyze_cpu(b64, analysis_prompt))
            except Exception as exc:
                return ToolResult(ok=False, error=f"CPU image analysis failed: {exc}")

        if backend == "llama":
            llama_available, _props = await self._llama_vision_available()
            if not llama_available:
                return ToolResult(ok=False, error="llama.cpp server does not report vision=true in /props")
            try:
                response = await self._analyze_llama(
                    llama_b64,
                    analysis_prompt,
                    mime_type=llama_mime_type,
                )
                return ToolResult(ok=bool(response.strip()), output=response, error=None if response.strip() else "Empty vision response")
            except Exception as exc:
                logger.warning("ImageAnalysisTool inference failed: %s", exc)
                return ToolResult(ok=False, error=str(exc))

        return ToolResult(ok=False, error=f"Unknown image_analysis_backend: {backend!r}")
