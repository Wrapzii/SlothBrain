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
            "prefer_dm": {
                "type": "boolean",
                "description": "When true, prefer bot DM/channel send path before webhook.",
                "default": False,
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
        owner_user_id: str = "",
    ) -> None:
        self._webhook_url = webhook_url
        self._bot_token = bot_token
        self._channel_id = channel_id
        self._owner_user_id = owner_user_id
        self._resolved_dm_channel: str = ""

    async def _resolve_dm_channel(self) -> str:
        """Open (or return cached) DM channel with the owner user."""
        if self._resolved_dm_channel:
            return self._resolved_dm_channel
        if not self._bot_token or not self._owner_user_id:
            return self._channel_id
        headers = {"Authorization": f"Bot {self._bot_token}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    f"{_DISCORD_API}/users/@me/channels",
                    headers=headers,
                    json={"recipient_id": self._owner_user_id},
                )
            if resp.status_code == 200:
                self._resolved_dm_channel = resp.json().get("id", "")
                logger.info("Discord DM channel resolved: %s", self._resolved_dm_channel)
                return self._resolved_dm_channel
            logger.error("Failed to resolve DM channel: HTTP %s %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.error("Failed to resolve DM channel: %s", exc)
        # In DM mode, do not silently fall back to a configured channel.
        # Falling back can cause the DM bridge to ingest and replay channel backlog.
        return ""

    async def execute(
        self,
        action: str = "",
        content: str = "",
        username: str = "",
        prefer_dm: bool = False,
        limit: int = 20,
        **kwargs: Any,
    ) -> ToolResult:
        if action == "send":
            return await self._send(content, username, prefer_dm=prefer_dm)
        if action == "history":
            return await self._history(min(limit, 100))
        return ToolResult(ok=False, error=f"Unknown action: {action!r}")

    async def _send(self, content: str, username: str, prefer_dm: bool = False) -> ToolResult:
        if not content:
            return ToolResult(ok=False, error="'content' is required for 'send'")

        errors: list[str] = []

        async def _try_bot_send() -> ToolResult | None:
            if not self._bot_token:
                return None
            channel = await self._resolve_dm_channel() if self._owner_user_id else self._channel_id
            if not channel:
                errors.append("No channel resolved for send.")
                return None
            headers = {"Authorization": f"Bot {self._bot_token}", "Content-Type": "application/json"}
            url = f"{_DISCORD_API}/channels/{channel}/messages"
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.post(url, headers=headers, json={"content": content})
                if resp.status_code == 200:
                    data = resp.json()
                    return ToolResult(ok=True, output={"sent": True, "message_id": data.get("id"), "via": "bot"})
                errors.append(f"Discord API returned HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as exc:
                errors.append(f"Bot send error: {exc}")
            return None

        if prefer_dm:
            bot_result = await _try_bot_send()
            if bot_result is not None:
                return bot_result

        # Prefer webhook when configured because it targets the configured
        # channel directly and avoids DM-channel resolution edge cases.
        if self._webhook_url:
            payload: dict = {"content": content}
            if username:
                payload["username"] = username
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.post(self._webhook_url, json=payload)
                if resp.status_code in (200, 204):
                    return ToolResult(ok=True, output={"sent": True, "via": "webhook"})
                errors.append(f"Webhook returned HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as exc:
                errors.append(f"Webhook error: {exc}")

        bot_result = await _try_bot_send()
        if bot_result is not None:
            return bot_result

        if errors:
            return ToolResult(ok=False, error=" | ".join(errors))

        return ToolResult(ok=False, error="No Discord credentials configured. Set webhook_url and/or bot_token.")

    async def _history(self, limit: int) -> ToolResult:
        if not self._bot_token:
            return ToolResult(
                ok=False,
                error="Reading channel history requires a bot token.",
            )
        channel = await self._resolve_dm_channel() if self._owner_user_id else self._channel_id
        if not channel:
            return ToolResult(ok=False, error="No channel resolved for history.")
        headers = {"Authorization": f"Bot {self._bot_token}"}
        url = f"{_DISCORD_API}/channels/{channel}/messages"
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
                    "author_id": m.get("author", {}).get("id"),
                    "is_bot": m.get("author", {}).get("bot", False),
                    "content": m.get("content", ""),
                    "timestamp": m.get("timestamp"),
                }
                for m in resp.json()
            ]
            return ToolResult(ok=True, output={"messages": messages, "count": len(messages)})
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
