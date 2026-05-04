from __future__ import annotations

import pytest

from backend.core.llama_client import LlamaClient
from backend.tools.impl.image_analysis_tool import ImageAnalysisTool


def test_extract_chat_text_falls_back_to_reasoning_content() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "The screenshot shows a Windows desktop.",
                }
            }
        ]
    }

    assert LlamaClient._extract_chat_text(payload) == "The screenshot shows a Windows desktop."


class FakeVisionClient:
    def __init__(self) -> None:
        self.called_with: dict | None = None

    async def get_props(self) -> dict:
        return {
            "model_name": "fake-gemma4-vision",
            "modalities": {"vision": True},
        }

    async def complete_with_image(self, **kwargs) -> str:
        self.called_with = dict(kwargs)
        return "Visible desktop UI with a taskbar."


@pytest.mark.asyncio
async def test_auto_backend_prefers_llamacpp_vision_when_available() -> None:
    client = FakeVisionClient()
    tool = ImageAnalysisTool(
        llama_client=client,  # type: ignore[arg-type]
        backend="auto",
    )

    result = await tool.execute(
        image_b64="iVBORw0KGgo=",
        prompt="Describe the desktop",
    )

    assert result.ok is True
    assert "fake-gemma4-vision" in str(result.output)
    assert "Visible desktop UI" in str(result.output)
    assert client.called_with is not None

