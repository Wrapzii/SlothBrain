from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from backend.core.approval_queue import PendingApproval

logger = logging.getLogger(__name__)
ApprovalHandler = Callable[[str], Awaitable[dict]]

try:
    import discord
except ImportError:  # pragma: no cover
    discord = None  # type: ignore[assignment]


if discord is None:
    class DiscordBridge:
        def __init__(self, *_: Any, **__: Any) -> None:
            logger.warning("discord.py not installed; DiscordBridge disabled")

        async def start(self) -> None: ...
        async def stop(self) -> None: ...
        async def send_approval(self, approval: PendingApproval) -> None: ...
        async def prompt_owner_for_text(self, prompt: str, timeout_seconds: float = 120.0) -> str:
            raise RuntimeError("Discord bridge unavailable")
else:
    class ApprovalView(discord.ui.View):
        def __init__(self, bridge: "DiscordBridge", approval_id: str):
            super().__init__(timeout=None)
            self.bridge = bridge
            self.approval_id = approval_id

        async def _authorized(self, interaction: discord.Interaction) -> bool:
            if interaction.user.id != self.bridge.owner_user_id:
                await interaction.response.send_message("Only the configured owner can use this control.", ephemeral=True)
                return False
            return True

        @discord.ui.button(label="Approve", style=discord.ButtonStyle.success, custom_id="slothbrain:approve")
        async def approve(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
            if not await self._authorized(interaction):
                return
            result = await self.bridge.approve_handler(self.approval_id)
            await interaction.response.edit_message(content=f"✅ Approved `{self.approval_id}`\n```json\n{result}\n```", view=None)

        @discord.ui.button(label="Reject", style=discord.ButtonStyle.danger, custom_id="slothbrain:reject")
        async def reject(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
            if not await self._authorized(interaction):
                return
            result = await self.bridge.reject_handler(self.approval_id)
            await interaction.response.edit_message(content=f"❌ Rejected `{self.approval_id}`\n```json\n{result}\n```", view=None)


    class DiscordBridge:
        def __init__(self, token: str, owner_user_id: int, approve_handler: ApprovalHandler, reject_handler: ApprovalHandler) -> None:
            self.token = token
            self.owner_user_id = owner_user_id
            self.approve_handler = approve_handler
            self.reject_handler = reject_handler
            self._pending_owner_reply: asyncio.Future[str] | None = None

            intents = discord.Intents.none()
            intents.dm_messages = True
            intents.message_content = True
            self.client = discord.Client(intents=intents)
            self.tree = discord.app_commands.CommandTree(self.client)
            self._task: asyncio.Task | None = None

            @self.client.event
            async def on_ready() -> None:
                logger.info("Discord bridge connected as %s", self.client.user)
                await self.tree.sync()

            @self.client.event
            async def on_message(message: discord.Message) -> None:
                if message.author.bot:
                    return
                if message.author.id != self.owner_user_id:
                    return
                if not isinstance(message.channel, discord.DMChannel):
                    return
                if self._pending_owner_reply and not self._pending_owner_reply.done():
                    self._pending_owner_reply.set_result(message.content.strip())

        async def start(self) -> None:
            if self._task:
                return
            self._task = asyncio.create_task(self.client.start(self.token), name="discord-bridge")

        async def stop(self) -> None:
            if self.client.is_ready() or not self.client.is_closed():
                await self.client.close()
            if self._task:
                try:
                    await self._task
                except Exception:
                    pass
                self._task = None

        async def send_approval(self, approval: PendingApproval) -> None:
            user = self.client.get_user(self.owner_user_id) or await self.client.fetch_user(self.owner_user_id)
            view = ApprovalView(self, approval.id)
            await user.send(f"⚠️ Approval required\n**Action:** `{approval.action}`\n**Description:** {approval.description}\n**ID:** `{approval.id}`", view=view)

        async def prompt_owner_for_text(self, prompt: str, timeout_seconds: float = 120.0) -> str:
            user = self.client.get_user(self.owner_user_id) or await self.client.fetch_user(self.owner_user_id)
            self._pending_owner_reply = asyncio.get_running_loop().create_future()
            await user.send(f"{prompt}\n\nPlease reply in this DM thread.")
            try:
                reply = await asyncio.wait_for(self._pending_owner_reply, timeout=timeout_seconds)
            finally:
                self._pending_owner_reply = None
            await user.send(f"Got it — you said: **{reply}**")
            return reply
