"""Web search tool — query DuckDuckGo (or SearXNG) and return ranked results.

Uses the DuckDuckGo Lite HTML endpoint (no API key required) by default.
Set ``searxng_url`` at construction time to use a self-hosted SearXNG
instance instead.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from backend.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

_DDG_URL = "https://lite.duckduckgo.com/lite/"
_USER_AGENT = "SlothBrain/1.0 (+https://github.com/Wrapzii/SlothBrain)"
_DEFAULT_MAX_RESULTS = 8
_TIMEOUT = 20.0


class WebSearchTool(Tool):
    """Search the web and return a list of ranked result snippets.

    By default uses DuckDuckGo Lite (no API key required).  Pass a
    ``searxng_url`` to the constructor to use a SearXNG instance instead.
    """

    name = "web_search"
    description = (
        "Search the web for information and return ranked result titles, URLs, and snippets."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": f"Maximum number of results to return (default: {_DEFAULT_MAX_RESULTS}).",
                "default": _DEFAULT_MAX_RESULTS,
            },
        },
        "required": ["query"],
    }

    def __init__(self, searxng_url: str = "") -> None:
        self._searxng_url = searxng_url.rstrip("/") if searxng_url else ""

    async def execute(
        self,
        query: str = "",
        max_results: int = _DEFAULT_MAX_RESULTS,
        **kwargs: Any,
    ) -> ToolResult:
        if not query:
            return ToolResult(ok=False, error="'query' argument is required")

        if self._searxng_url:
            return await self._search_searxng(query, max_results)
        return await self._search_ddg(query, max_results)

    # ------------------------------------------------------------------
    # DuckDuckGo Lite
    # ------------------------------------------------------------------

    async def _search_ddg(self, query: str, max_results: int) -> ToolResult:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=_TIMEOUT) as client:
                resp = await client.post(
                    _DDG_URL,
                    data={"q": query, "kl": "wt-wt"},
                    headers={"User-Agent": _USER_AGENT},
                )
            if resp.status_code >= 400:
                return ToolResult(ok=False, error=f"DuckDuckGo returned HTTP {resp.status_code}")

            results = _parse_ddg_html(resp.text, max_results)
            return ToolResult(ok=True, output={"query": query, "results": results})
        except Exception as exc:
            logger.warning("WebSearchTool DDG error: %s", exc)
            return ToolResult(ok=False, error=str(exc))

    # ------------------------------------------------------------------
    # SearXNG JSON API
    # ------------------------------------------------------------------

    async def _search_searxng(self, query: str, max_results: int) -> ToolResult:
        url = f"{self._searxng_url}/search"
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=_TIMEOUT) as client:
                resp = await client.get(
                    url,
                    params={"q": query, "format": "json"},
                    headers={"User-Agent": _USER_AGENT},
                )
            if resp.status_code >= 400:
                return ToolResult(ok=False, error=f"SearXNG returned HTTP {resp.status_code}")

            data = resp.json()
            raw_results = data.get("results", [])[:max_results]
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                }
                for r in raw_results
            ]
            return ToolResult(ok=True, output={"query": query, "results": results})
        except Exception as exc:
            logger.warning("WebSearchTool SearXNG error: %s", exc)
            return ToolResult(ok=False, error=str(exc))


# ---------------------------------------------------------------------------
# HTML parser for DuckDuckGo Lite
# ---------------------------------------------------------------------------

def _parse_ddg_html(html: str, max_results: int) -> list[dict]:
    """Extract search results from DuckDuckGo Lite HTML response."""
    results: list[dict] = []

    # Each result block: <td class="result-link"> / <td class="result-snippet">
    link_re = re.compile(
        r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', re.DOTALL
    )
    snippet_re = re.compile(
        r'class="result-snippet"[^>]*>(.*?)</td>', re.DOTALL
    )

    links = link_re.findall(html)
    snippets = [re.sub(r"<[^>]+>", "", s) for s in snippet_re.findall(html)]

    for i, (url, title) in enumerate(links[:max_results]):
        clean_title = re.sub(r"<[^>]+>", "", title).strip()
        snippet = snippets[i].strip() if i < len(snippets) else ""
        results.append({"title": clean_title, "url": url, "snippet": snippet})

    return results
