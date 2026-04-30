"""Discord messaging tool.

Sends messages to Discord channels via an incoming webhook URL or a bot
token.  Optionally fetches recent channel history (requires bot token +
``DISCORD_CHANNEL_ID``).

Configuration is passed via constructor arguments; secrets should come from
environment variables / ``.env``, not from model output.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

_DISCORD_API = "https://discord.com/api/v10"
_TIMEOUT = 15.0


class DiscordTool(Tool):
    """Send messages to Discord and optionally read recent channel history.

    Actions
    -------
    * ``send``    — post a message to a channel or webhook.
    * ``history`` — fetch recent messages from a channel (bot token required).
    """

    name = "discord"
    description = (
        "Send messages to a Discord channel via webhook or bot token. "
        "Optionally read recent channel message history."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["send", "history"],
                "description": "Operation to perform.",
            },
            "content": {
                "type": "string",
                "description": "Message content to send (required for 'send').",
            },
            "username": {
                "type": "string",
                "description": "Override webhook display name (webhook mode only).",
            },
            "limit": {
                "type": "integer",
                "description": "Number of recent messages to fetch (default: 20, max: 100).",
                "default": 20,
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        webhook_url: str = "",
        bot_token: str = "",
        channel_id: str = "",
    ) -> None:
        self._webhook_url = webhook_url
        self._bot_token = bot_token
        self._channel_id = channel_id

    async def execute(
        self,
        action: str = "",
        content: str = "",
        username: str = "",
        limit: int = 20,
        **kwargs: Any,
    ) -> ToolResult:
        if action == "send":
            return await self._send(content, username)
        if action == "history":
            return await self._history(min(limit, 100))
        return ToolResult(ok=False, error=f"Unknown action: {action!r}")

    async def _send(self, content: str, username: str) -> ToolResult:
        if not content:
            return ToolResult(ok=False, error="'content' is required for 'send'")

        # Webhook path (no auth required)
        if self._webhook_url:
            payload: dict = {"content": content}
            if username:
                payload["username"] = username
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.post(self._webhook_url, json=payload)
                if resp.status_code in (200, 204):
                    return ToolResult(ok=True, output={"sent": True, "via": "webhook"})
                return ToolResult(
                    ok=False,
                    error=f"Webhook returned HTTP {resp.status_code}: {resp.text[:200]}",
                )
            except Exception as exc:
                return ToolResult(ok=False, error=str(exc))

        # Bot token path
        if self._bot_token and self._channel_id:
            headers = {"Authorization": f"Bot {self._bot_token}", "Content-Type": "application/json"}
            url = f"{_DISCORD_API}/channels/{self._channel_id}/messages"
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.post(url, headers=headers, json={"content": content})
                if resp.status_code == 200:
                    data = resp.json()
                    return ToolResult(ok=True, output={"sent": True, "message_id": data.get("id"), "via": "bot"})
                return ToolResult(
                    ok=False,
                    error=f"Discord API returned HTTP {resp.status_code}: {resp.text[:200]}",
                )
            except Exception as exc:
                return ToolResult(ok=False, error=str(exc))

        return ToolResult(
            ok=False,
            error="No Discord credentials configured. Set webhook_url or (bot_token + channel_id).",
        )

    async def _history(self, limit: int) -> ToolResult:
        if not self._bot_token or not self._channel_id:
            return ToolResult(
                ok=False,
                error="Reading channel history requires a bot token and channel ID.",
            )
        headers = {"Authorization": f"Bot {self._bot_token}"}
        url = f"{_DISCORD_API}/channels/{self._channel_id}/messages"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, headers=headers, params={"limit": limit})
            if resp.status_code != 200:
                return ToolResult(
                    ok=False,
                    error=f"Discord API returned HTTP {resp.status_code}: {resp.text[:200]}",
                )
            messages = [
                {
                    "id": m.get("id"),
                    "author": m.get("author", {}).get("username"),
                    "content": m.get("content", ""),
                    "timestamp": m.get("timestamp"),
                }
                for m in resp.json()
            ]
            return ToolResult(ok=True, output={"messages": messages, "count": len(messages)})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
