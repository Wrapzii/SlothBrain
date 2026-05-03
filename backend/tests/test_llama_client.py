from __future__ import annotations

import httpx
import pytest

from backend.core.llama_client import LlamaClient


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


class _FakeAsyncClient:
    call_count = 0

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url: str, json: dict):
        _FakeAsyncClient.call_count += 1
        if _FakeAsyncClient.call_count == 1:
            raise httpx.ReadError("transient read failure")
        return _FakeResponse(payload={"content": "ok"})


@pytest.mark.asyncio
async def test_complete_retries_transient_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    client = LlamaClient(host="127.0.0.1", port=8080)
    result = await client.complete(prompt="assistant:", max_tokens=32)

    assert result == "ok"
    assert _FakeAsyncClient.call_count == 2
