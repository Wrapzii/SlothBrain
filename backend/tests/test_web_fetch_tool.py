from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.tools.impl.web_fetch_tool import WebFetchTool


@pytest.mark.asyncio
async def test_web_fetch_retries_www_dns_failure_without_www() -> None:
    requested_urls: list[str] = []

    async def fake_get(self, url, **kwargs):
        requested_urls.append(url)
        if "://www." in url:
            request = httpx.Request("GET", url)
            raise httpx.ConnectError("DNS lookup failed", request=request)
        return httpx.Response(
            200,
            text="<html><head><title>Fallback OK</title></head><body>ok</body></html>",
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", url),
        )

    with patch.object(httpx.AsyncClient, "get", new=fake_get):
        result = await WebFetchTool().execute(url="https://www.example-tool-target.test/path")

    assert result.ok is True
    assert "Fallback OK" in result.output
    assert requested_urls == [
        "https://www.example-tool-target.test/path",
        "https://example-tool-target.test/path",
    ]
