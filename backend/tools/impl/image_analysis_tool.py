"""Image analysis tool — send an image to a vision-capable model endpoint.

The tool accepts a base64-encoded PNG (or a path to take a live screenshot)
and forwards it to the configured LlamaClient completion endpoint along with
an analysis prompt.  If no multimodal endpoint is available the tool returns
a graceful error rather than raising.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.core.llama_client import LlamaClient

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = (
    "Describe this image in detail. Include any text visible on screen, "
    "the layout, and any notable UI elements or content."
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
        "Analyse an image (base64-encoded or live screenshot) using a vision model. "
        "Returns a detailed textual description."
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

    async def execute(
        self,
        image_b64: str = "",
        screenshot: bool = False,
        prompt: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        import asyncio

        # Obtain image bytes
        b64 = image_b64
        if screenshot and self._controller is not None:
            try:
                cap = await asyncio.to_thread(self._controller.capture)
                b64 = cap.get("annotated_png_b64", "")
            except Exception as exc:
                return ToolResult(ok=False, error=f"Screenshot failed: {exc}")

        if not b64:
            return ToolResult(
                ok=False,
                error="No image provided. Pass 'image_b64' or set 'screenshot': true.",
            )

        analysis_prompt = prompt or _DEFAULT_PROMPT
        backend = self._backend

        if backend in {"cpu_ocr", "auto"}:
            try:
                return ToolResult(ok=True, output=self._analyze_cpu(b64, analysis_prompt))
            except Exception as exc:
                if backend == "cpu_ocr":
                    return ToolResult(ok=False, error=f"CPU image analysis failed: {exc}")
                logger.info("CPU image analysis failed; falling back to llama backend: %s", exc)

        full_prompt = (
            "system: You are a vision analysis assistant.\n\n"
            f"user: [IMAGE: data:image/png;base64,{b64}]\n\n"
            f"{analysis_prompt}\nassistant:"
        )

        try:
            response = await self._client.complete(
                prompt=full_prompt,
                slot_id=self._llama_slot_id,
                max_tokens=1024,
                temperature=0.3,
            )
            return ToolResult(ok=True, output=response)
        except Exception as exc:
            logger.warning("ImageAnalysisTool inference failed: %s", exc)
            return ToolResult(ok=False, error=str(exc))
