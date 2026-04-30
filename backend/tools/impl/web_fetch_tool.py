"""Web fetch tool — HTTP GET or POST with configurable timeout.

Uses ``httpx`` (already a project dependency) to fetch URLs and return
either the response body text or parsed JSON.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_MAX_RESPONSE_CHARS = 50_000
_USER_AGENT = "SlothBrain/1.0 (+https://github.com/Wrapzii/SlothBrain)"


class WebFetchTool(Tool):
    """Fetch a URL via HTTP GET or POST.

    Returns the response body as text (up to 50 000 characters) or parsed
    JSON.  Suitable for reading web pages, REST APIs, or raw files.
    """

    name = "web_fetch"
    description = (
        "Fetch a URL via HTTP GET or POST. Returns the response body as text or "
        "parsed JSON. Useful for reading web pages, APIs, and remote files."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST"],
                "description": "HTTP method (default: GET).",
                "default": "GET",
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers as a key-value dict.",
            },
            "body": {
                "type": "string",
                "description": "Request body string (used with POST).",
            },
            "json_body": {
                "type": "object",
                "description": "Request body as JSON object (used with POST).",
            },
            "timeout": {
                "type": "number",
                "description": f"Request timeout in seconds (default: {_DEFAULT_TIMEOUT}).",
                "default": _DEFAULT_TIMEOUT,
            },
        },
        "required": ["url"],
    }

    async def execute(
        self,
        url: str = "",
        method: str = "GET",
        headers: dict | None = None,
        body: str = "",
        json_body: dict | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        **kwargs: Any,
    ) -> ToolResult:
        if not url:
            return ToolResult(ok=False, error="'url' argument is required")

        request_headers = {"User-Agent": _USER_AGENT}
        if headers:
            request_headers.update(headers)

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                if method.upper() == "POST":
                    if json_body is not None:
                        resp = await client.post(url, headers=request_headers, json=json_body)
                    else:
                        resp = await client.post(url, headers=request_headers, content=body.encode())
                else:
                    resp = await client.get(url, headers=request_headers)

            status = resp.status_code
            content_type = resp.headers.get("content-type", "")

            # Try JSON first
            output: Any
            if "application/json" in content_type:
                try:
                    output = resp.json()
                except Exception:
                    output = resp.text[:_MAX_RESPONSE_CHARS]
            else:
                output = resp.text[:_MAX_RESPONSE_CHARS]

            if status >= 400:
                return ToolResult(
                    ok=False,
                    output=output,
                    error=f"HTTP {status} {resp.reason_phrase}",
                )
            return ToolResult(ok=True, output=output)

        except httpx.TimeoutException as exc:
            return ToolResult(ok=False, error=f"Request timed out after {timeout}s: {exc}")
        except Exception as exc:
            logger.warning("WebFetchTool error fetching %s: %s", url, exc)
            return ToolResult(ok=False, error=str(exc))
