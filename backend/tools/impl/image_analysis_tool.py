"""Image analysis tool — send an image to a vision-capable model endpoint.

The tool accepts a base64-encoded PNG (or a path to take a live screenshot)
and forwards it to the configured LlamaClient completion endpoint along with
an analysis prompt.  If no multimodal endpoint is available the tool returns
a graceful error rather than raising.
"""
from __future__ import annotations

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
    ) -> None:
        self._client = llama_client
        self._controller = controller

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
        full_prompt = (
            "system: You are a vision analysis assistant.\n\n"
            f"user: [IMAGE: data:image/png;base64,{b64}]\n\n"
            f"{analysis_prompt}\nassistant:"
        )

        try:
            response = await self._client.complete(
                prompt=full_prompt,
                slot_id=-1,
                max_tokens=1024,
                temperature=0.3,
            )
            return ToolResult(ok=True, output=response)
        except Exception as exc:
            logger.warning("ImageAnalysisTool inference failed: %s", exc)
            return ToolResult(ok=False, error=str(exc))
