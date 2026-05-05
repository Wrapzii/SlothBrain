from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
import shlex
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.agents.main_agent import MainAgent
from backend.agents.agentic_loop import AgenticDebugOptions
from backend.agents.preset_manager import PresetManager
from backend.agents.registry import AgentRegistry
from backend.benchmarks.benchmark import BenchmarkSuite
from backend.config import settings
from backend.core.approval_queue import ApprovalQueue
from backend.core.audit_log import AuditLog
from backend.core.checkpoint_manager import CheckpointManager
from backend.core.diagnostic_recorder import DiagnosticRecorder
from backend.core.llama_client import LlamaClient
from backend.core.resource_manager import ResourceManager
from backend.core.safety_supervisor import SafetySupervisor
from backend.core.safety_supervisor import _extract_tps_from_metrics
from backend.core.server_manager import ServerManager
from backend.core.slot_manager import SlotManager
from backend.core.semantic_router import SemanticRouter
from backend.memory.lancedb_memory import LanceDBMemory
from backend.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)

_DM_SHORT_ACK_RE = re.compile(r"^(?:yes|yep|yeah|ok|okay|sure|do it|go ahead|continue|proceed)$", re.IGNORECASE)
_LOCAL_FILESYSTEM_ACTION_RE = re.compile(
    r"\b(?:find|locate|list|show|check|open|create|make|build|write|edit|modify|delete|read)\b"
    r".{0,80}\b(?:file|folder|directory|filesystem|workspace|github|documents|downloads|desktop|"
    r"repo(?:sitory)?|project(?:s)?)\b|"
    r"\b(?:github|documents|downloads|desktop)\b.{0,80}\b(?:folder|directory|projects?|files?)\b",
    re.IGNORECASE,
)
_DISCORD_DM_HANDLE_TIMEOUT_SECONDS = 90.0
_DISCORD_DM_MAX_BACKLOG_PER_POLL = 20


def _effective_discord_dm_timeout_seconds() -> float:
    """Return the effective timeout budget for a single DM message.

    Large models can legitimately take longer than 90s for tool-heavy prompts.
    Use the larger of:
    - static DM timeout default
    - llama completion timeout + buffer

    Set SLOTHBRAIN_DISCORD_DM_TIMEOUT_SECONDS=0 to disable timeout entirely.
    """
    configured = float(getattr(settings, "discord_dm_timeout_seconds", _DISCORD_DM_HANDLE_TIMEOUT_SECONDS))
    if configured == 0:
        return 0.0
    base = float(getattr(settings, "llama_completion_timeout_seconds", 600.0))
    adaptive = max(_DISCORD_DM_HANDLE_TIMEOUT_SECONDS, base + 30.0)
    return configured if configured > 0 else adaptive


def _sanitize_user_facing_response(text: str) -> str:
    """Strip protocol / tool-call residue from responses shown to users."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"(?is)<tool_call>.*?(?:</tool_call>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)<tool_result>.*?(?:</tool_result>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)<[a-z_][a-z0-9_:-]*>.*?</[a-z_][a-z0-9_:-]*>", "", cleaned)
    cleaned = re.sub(r"(?is)</?[a-z_][a-z0-9_:-]*>", "", cleaned)
    cleaned = re.sub(r'(?im)^\s*\{\s*"tool"\s*:.*$', "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(task execution complete\.?|task initiated\.?|fetching .* now\.?|would you like me to proceed\??)\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*task\s*result\s*:\s*.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*no tools failed\.?\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*both the initial search and the web fetch executed without issues.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*session terminated\.?\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*sub-agent session terminated successfully\.?\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*detected repeated tool loop for '.*' with near-identical arguments\.?\s*$", "", cleaned)
    cleaned = re.sub(
        r"(?im)^\s*based on the tool result\(s\) above, provide the actual (?:answer|findings, data, or answer)\.?\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*i see\. based on the tool result\(s\) above, provide the actual answer\. do not make up information\.?\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?im)^\s*-\s*user\s*:\s*.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*user\s*:\s*.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*```[a-zA-Z0-9_-]*\s*$", "", cleaned)
    cleaned = re.sub(r"(?is)```\s*```", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*overall\s+task\s*:\s*.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*you\s+are\s+executing\s+a\s+multi-step\s+task.*$", "", cleaned)
    cleaned = re.sub(r"(?is)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _htmlish_to_plain_summary(text: str) -> str:
    raw = (text or "").replace("\\r", "\n").replace("\\n", "\n")
    if not raw.strip():
        return ""

    title = ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = re.sub(r"\s+", " ", title_match.group(1)).strip()

    desc = ""
    desc_match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if not desc_match:
        desc_match = re.search(
            r'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']description["\']',
            raw,
            re.IGNORECASE | re.DOTALL,
        )
    if desc_match:
        desc = re.sub(r"\s+", " ", desc_match.group(1)).strip()

    cleaned = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", raw)
    cleaned = re.sub(r"(?is)<!--.*?-->", " ", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"&nbsp;", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"&amp;", "&", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"&lt;", "<", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"&gt;", ">", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"&quot;", '"', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    parts: list[str] = []
    if title:
        parts.append(f"Page title: {title}.")
    if desc:
        parts.append(desc)
    elif cleaned:
        parts.append(cleaned[:800].strip())
    return " ".join(parts).strip()


def _best_user_facing_response(text: str) -> str:
    cleaned = _sanitize_user_facing_response(text)
    html_summary = _htmlish_to_plain_summary(text)
    if html_summary and (not cleaned or "<" in cleaned or ">" in cleaned):
        return html_summary

    if cleaned:
        return cleaned

    fallback = re.sub(r"(?is)<[^>]+>", " ", text or "")
    fallback = re.sub(r"\s+", " ", fallback).strip()
    return fallback


class DiscordDMBridge:
    """Background task that listens to Discord DM and bridges to SlothBrain chat."""

    def __init__(self, main_agent: "MainAgent", config: Any, registry: "ToolRegistry"):
        self._main_agent = main_agent
        self._config = config
        self._registry = registry
        self._running = False
        self._processed_ids: deque = deque(maxlen=100)  # Track processed message IDs to avoid duplicates
        self._task: asyncio.Task | None = None
        self._bot_user_id: str = ""  # Populated on first message check
        self._history_primed: bool = False
        self._last_poll_at: float = 0.0
        self._last_processed_id: str = ""
        self._last_error: str = ""
        # Keep a short per-user transcript so Discord replies preserve recent context.
        self._dm_context: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=8))
        # Track in-flight background tasks per user so /status can report them.
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._active_task_labels: dict[str, str] = {}
        # Remember the most recent explicit research topic per user so
        # follow-ups like "research what I said" can be resolved deterministically.
        self._last_research_topic: dict[str, str] = {}
        self._state_path = Path("data") / "discord_dm_state.json"
        self._last_seen_message_id: str = ""
        self._load_state()

    @staticmethod
    def _id_gt(a: str, b: str) -> bool:
        try:
            return int(str(a)) > int(str(b))
        except Exception:
            return str(a) > str(b)

    def _load_state(self) -> None:
        try:
            if not self._state_path.exists():
                return
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            last_seen = str(payload.get("last_seen_message_id", "") or "")
            processed = payload.get("processed_ids", [])
            if isinstance(processed, list):
                for msg_id in processed[-100:]:
                    sid = str(msg_id or "").strip()
                    if sid:
                        self._processed_ids.append(sid)
            self._last_seen_message_id = last_seen
        except Exception as exc:
            logger.debug("Discord DM bridge state load failed: %s", exc.__class__.__name__)

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "last_seen_message_id": self._last_seen_message_id,
                "processed_ids": list(self._processed_ids),
                "updated_at": time.time(),
            }
            self._state_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            logger.debug("Discord DM bridge state save failed: %s", exc.__class__.__name__)

    @staticmethod
    def _clean_discord_response(text: str) -> str:
        """Strip protocol/HTML residue from model output before posting to Discord."""
        return _best_user_facing_response(text)

    @staticmethod
    def _render_tools_used_block(tools_used: list[str]) -> str:
        """Render a compact tree-style block of actually-used tools."""
        clean = [t.strip() for t in tools_used if t and t.strip()]
        if not clean:
            return ""
        unique: list[str] = []
        seen: set[str] = set()
        for name in clean:
            if name in seen:
                continue
            seen.add(name)
            unique.append(name)
        lines = ["Tools used:", "📂 execution"]
        for idx, name in enumerate(unique):
            branch = "└──" if idx == len(unique) - 1 else "├──"
            lines.append(f"{branch} 🔧 {name}")
        return "\n" + "\n".join(lines)

    # Matches step outputs that describe intended future work rather than
    # delivering actual results.  These should be skipped when extracting the
    # best agentic reply to show the user.
    _META_COMMENTARY_RE = re.compile(
        r"^(?:"
        r"based on (?:the )?(?:research|information|data|findings|results|previous|prior|above)"
        r"|based on the tool result\(s\) above"
        r"|i will now (?:compile|synthesize|summarize|gather|proceed|write|create|provide|present)"
        r"|i(?:'ll| will) (?:now )?(?:compile|synthesize|summarize|gather|proceed)"
        r"|since (?:no new|the) (?:data|information|research)"
        r"|having (?:gathered|completed|reviewed|researched|analyzed)"
        r"|now (?:that|i(?:'ll| will))\b"
        r"|let me (?:compile|synthesize|summarize|now)"
        r"|extract and cite what the tool returned"
        r")",
        re.IGNORECASE,
    )

    @staticmethod
    def _extract_agentic_response(result: dict) -> str:
        """Prefer the last meaningful step result over the loop's generic summary.

        Skips meta-commentary like 'I will now compile...' and keeps scanning
        backwards through steps for a response that contains real content.
        """
        steps = result.get("steps") if isinstance(result, dict) else None
        fallback: str = ""
        if isinstance(steps, list):
            for step in reversed(steps):
                if not isinstance(step, dict):
                    continue
                step_result = str(step.get("result") or "").strip()
                if not step_result:
                    continue
                if step_result in ("Task execution complete.", "Task execution finished with failures."):
                    continue
                if DiscordDMBridge._META_COMMENTARY_RE.match(step_result):
                    if not fallback:
                        fallback = step_result
                    continue
                return step_result
        if fallback:
            return fallback
        summary = str(result.get("summary") or "").strip() if isinstance(result, dict) else ""
        return summary or "Task execution complete."

    def _should_route_to_agentic(self, user_key: str, content: str) -> bool:
        text = (content or "").strip()
        if not text:
            return False
        lower = text.lower()
        # Only explicit loop commands enter the agentic path. Normal language
        # belongs to direct model chat so conversation is not hard-coded.
        if any(lower.startswith(p) for p in ("/task", "/agentic", "/research", "/ralph")):
            return True
        if _LOCAL_FILESYSTEM_ACTION_RE.search(text):
            return True
        if _DM_SHORT_ACK_RE.match(text):
            history = list(self._dm_context.get(user_key, []))
            recent = "\n".join(history[-4:]).lower()
            if "/task" in recent or "/agentic" in recent:
                return True
        return False

    async def _handle_command(self, user_key: str, content: str) -> str | None:
        text = (content or "").strip()
        lower = text.lower()

        if lower in {"/task", "/agentic", "/research", "/ralph"} or any(
            lower.startswith(p) for p in ("/task ", "/agentic ", "/research ", "/ralph ")
        ):
            return None

        if lower == "/reset":
            self._dm_context[user_key].clear()
            return "Conversation context reset for this Discord chat."

        if lower == "/status":
            stats = await resource_manager.get_system_stats()

            tps_snapshot = "unavailable"
            try:
                metrics = await llama_client.get_metrics()
                slowdown = _extract_tps_from_metrics(metrics)
                if slowdown is not None:
                    tps_snapshot = f"{slowdown.tokens_per_sec:.2f} tok/s"
            except Exception:
                pass

            # Inference state
            inference_busy = _inference_lock.locked()
            active_task_label = self._active_task_labels.get(user_key, "")
            # Also check if any user has an active task (useful if single-user)
            if not active_task_label and self._active_task_labels:
                active_task_label = next(iter(self._active_task_labels.values()))
            if active_task_label:
                task_line = f"Active task: {active_task_label}"
            elif inference_busy:
                task_line = "Active task: inference busy (another user or API request)"
            else:
                task_line = "Active task: none"

            cpp_status = server_manager.status
            # server_manager reports 'stopped' when llama.cpp is managed externally — clarify
            if cpp_status == "stopped" and inference_busy:
                cpp_status = "external / unmanaged (inference active)"
            elif cpp_status == "stopped":
                cpp_status = "stopped (or externally managed)"

            return (
                "**SlothBrain Status**\n"
                f"CPU: {stats.get('cpu_percent', '?')}%  |  "
                f"RAM: {stats.get('ram_used_mb', '?')} / {stats.get('ram_total_mb', '?')} MB\n"
                f"llama.cpp: {cpp_status}\n"
                f"Throughput: {tps_snapshot}\n"
                f"{task_line}"
            )

        if lower in ("/restart", "/restart cpp", "/restart llama", "/restart llamacpp"):
            try:
                await server_manager.restart(actor="discord")
                return "Requested llama.cpp restart successfully."
            except Exception as exc:
                return f"Failed to restart llama.cpp: {exc}"

        if lower in ("/restart slothbrain", "/restart app"):
            return "Restarting the SlothBrain app process from inside the running app is not implemented yet. Use the local launcher for that restart path."

        # Prevent typos like /ststus from being routed into agentic/direct tasks.
        if lower.startswith("/"):
            return "Unknown command. Try /status, /reset, /restart, /task <goal>, /research <topic>, or /ralph <goal>."

        return None

    def _resolve_discord_followup_task(self, user_key: str, content: str) -> str:
        """Resolve shorthand follow-ups to a concrete task string.

        Example: after /research <topic>, a follow-up like
        "research what I said" should map back to the last explicit topic.
        """
        text = (content or "").strip()
        lower = text.lower()
        followup_markers = (
            "research what i said",
            "research what i asked",
            "research that",
            "research it",
            "i'd like you to research what i said",
            "please research what i said",
        )
        if any(marker in lower for marker in followup_markers):
            topic = (self._last_research_topic.get(user_key) or "").strip()
            if topic:
                return f"/research {topic}"
        if lower.startswith("/research "):
            topic = text[10:].strip()
            if topic:
                self._last_research_topic[user_key] = topic
        return text

    def start(self) -> None:
        """Start the DM listener background task."""
        has_channel = bool(self._config.discord_bot_token and (
            getattr(self._config, "discord_owner_user_id", "") or self._config.discord_channel_id
        ))
        if not has_channel:
            logger.info("Discord DM bridge disabled: bot_token + owner_user_id (or channel_id) not configured")
            return
        self._running = True
        self._task = asyncio.create_task(self._listener_loop())
        logger.info("Discord DM listener started")

    def stop(self) -> None:
        """Stop the DM listener background task."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
        logger.info("Discord DM listener stopped")

    async def _listener_loop(self) -> None:
        """Poll Discord channel history and forward new messages to SlothBrain."""
        poll_interval = 5  # seconds
        while self._running:
            try:
                await asyncio.sleep(poll_interval)
                await self._check_new_messages()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Discord DM listener error: %s", exc)
                await asyncio.sleep(poll_interval * 2)  # Back off on error

    async def _check_new_messages(self) -> None:
        """Fetch recent messages and process new ones."""
        self._last_poll_at = time.time()
        discord_tool = self._registry.get("discord")
        if discord_tool is None:
            return

        # Resolve bot's own user ID once so we can filter reliably by ID not username
        if not self._bot_user_id and self._config.discord_bot_token:
            import httpx as _httpx
            try:
                async with _httpx.AsyncClient(timeout=5.0) as client:
                    r = await client.get(
                        "https://discord.com/api/v10/users/@me",
                        headers={"Authorization": f"Bot {self._config.discord_bot_token}"},
                    )
                if r.status_code == 200:
                    self._bot_user_id = r.json().get("id", "")
            except Exception:
                pass

        # Get recent messages from DM channel
        result = await discord_tool.execute(action="history", limit=50)
        if not result.ok or not isinstance(result.output, dict):
            return

        messages = result.output.get("messages", [])
        if not messages:
            return

        max_seen_this_poll = self._last_seen_message_id
        for msg in messages:
            msg_id = str(msg.get("id", "") or "").strip()
            if msg_id and (not max_seen_this_poll or self._id_gt(msg_id, max_seen_this_poll)):
                max_seen_this_poll = msg_id

        # On first successful poll after startup, prime the seen-ID buffer from
        # existing channel history so we only process *new* incoming messages.
        # This prevents replaying old backlog every time the backend restarts.
        if not self._history_primed:
            for msg in messages:
                msg_id = str(msg.get("id", "") or "").strip()
                if msg_id:
                    self._processed_ids.append(msg_id)
            if max_seen_this_poll:
                self._last_seen_message_id = max_seen_this_poll
            self._save_state()
            self._history_primed = True
            logger.info(
                "Discord DM listener primed history with %d message ids",
                len(self._processed_ids),
            )
            return

        # Process messages in chronological order (oldest first)
        messages_to_process = []
        for msg in reversed(messages):
            msg_id = msg.get("id")
            author = msg.get("author", "")
            author_id = msg.get("author_id", "")
            is_bot = msg.get("is_bot", False)
            content = msg.get("content", "").strip()

            # Skip bot's own messages (by ID or bot flag)
            if is_bot or (self._bot_user_id and author_id == self._bot_user_id):
                continue

            # Skip if already processed
            if msg_id in self._processed_ids:
                continue

            # Hard guard against restart replay: skip anything at or below
            # persisted watermark from previous runs.
            if self._last_seen_message_id and not self._id_gt(msg_id, self._last_seen_message_id):
                self._processed_ids.append(msg_id)
                continue

            if content:
                messages_to_process.append(
                    {
                        "id": msg_id,
                        "author": author,
                        "author_id": author_id,
                        "content": content,
                    }
                )

        # Avoid draining huge backlogs in one poll and reduce the chance that
        # stale old messages block responsiveness to current user input.
        if len(messages_to_process) > _DISCORD_DM_MAX_BACKLOG_PER_POLL:
            messages_to_process = messages_to_process[-_DISCORD_DM_MAX_BACKLOG_PER_POLL:]

        # Process new messages
        for msg in messages_to_process:
            try:
                dm_timeout = _effective_discord_dm_timeout_seconds()
                if dm_timeout > 0:
                    await asyncio.wait_for(
                        self._handle_message(msg),
                        timeout=dm_timeout,
                    )
                else:
                    await self._handle_message(msg)
                self._processed_ids.append(msg["id"])
                self._last_processed_id = str(msg.get("id") or "")
                if self._last_processed_id and (
                    not self._last_seen_message_id
                    or self._id_gt(self._last_processed_id, self._last_seen_message_id)
                ):
                    self._last_seen_message_id = self._last_processed_id
                self._save_state()
            except asyncio.TimeoutError:
                logger.error(
                    "Discord DM processing timed out for message %s from %s",
                    msg.get("id"),
                    msg.get("author"),
                )
                self._last_error = (
                    f"timeout processing message {msg.get('id')} from {msg.get('author')}"
                )
                self._processed_ids.append(msg["id"])
                discord_tool = self._registry.get("discord")
                if discord_tool is not None:
                    try:
                        await discord_tool.execute(
                            action="send",
                            prefer_dm=True,
                            content=(
                                "I'm sorry, that request timed out while processing. "
                                "Please try again with a shorter prompt."
                            ),
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Error processing Discord DM from %s: %s", msg.get("author"), exc)
                self._last_error = f"processing error for {msg.get('id')}: {exc.__class__.__name__}"
                # Mark as processed so one bad message cannot poison all future polls.
                self._processed_ids.append(msg["id"])
                self._save_state()

        # Persist latest poll watermark even when no messages were processed.
        if max_seen_this_poll and (
            not self._last_seen_message_id or self._id_gt(max_seen_this_poll, self._last_seen_message_id)
        ):
            self._last_seen_message_id = max_seen_this_poll
            self._save_state()

    async def _handle_message(self, msg: dict) -> None:
        """Forward a Discord DM to SlothBrain and post the response."""
        author = msg.get("author", "User")
        author_id = str(msg.get("author_id", "")).strip()
        user_key = author_id or author
        raw_content = msg.get("content", "")
        content = self._resolve_discord_followup_task(user_key=user_key, content=raw_content)

        logger.info("Discord DM from %s: %s", author, content[:100])

        command_response = await self._handle_command(user_key=user_key, content=content)
        if command_response is not None:
            response = command_response
            response = self._clean_discord_response(response)
            self._dm_context[user_key].append(f"User: {raw_content}")
            self._dm_context[user_key].append(f"SlothBrain: {response}")

            discord_tool = self._registry.get("discord")
            if discord_tool:
                send_result = await discord_tool.execute(action="send", prefer_dm=True, content=response[:1900])
                if send_result.ok:
                    logger.info("Discord DM command response sent")
                else:
                    logger.error("Failed to post Discord DM command response: %s", send_result.error)
            return

        is_task_message = self._should_route_to_agentic(user_key=user_key, content=content)
        response = ""

        # Forward to SlothBrain direct chat or agentic loop depending on user input.
        try:
            if is_task_message:
                from backend.agents.agentic_loop import AgenticLoop

                task_message = _build_discord_task(content.strip())

                if not task_message:
                    response = "Please provide a task after /task, for example: /task summarize https://example.com"
                else:
                    self._dm_context[user_key].append(f"User: {raw_content}")
                    # Run in background so the listener loop stays responsive (e.g. /status).
                    await self._run_bg_task(user_key=user_key, label=task_message, coro=self._run_agentic_task(user_key, task_message, raw_content))
                    discord_tool = self._registry.get("discord")
                    if discord_tool:
                        try:
                            await discord_tool.execute(
                                action="send",
                                prefer_dm=True,
                                content=(
                                    f"Started task: {task_message}\n"
                                    "Use /status to check progress."
                                ),
                            )
                        except Exception:
                            pass
                    return
            else:
                # Direct chat — also run in background so listener isn't blocked.
                self._dm_context[user_key].append(f"User: {raw_content}")
                await self._run_bg_task(user_key=user_key, label=content[:60], coro=self._run_direct_task(user_key, content, raw_content))
                return
        except Exception as exc:
            response = f"[Error processing message: {exc.__class__.__name__}]"
            logger.error("Failed to process Discord DM: %s", exc)

        response = self._clean_discord_response(response)
        if not response:
            response = "I couldn't generate a useful response. Please try again."

        # Persist this turn in the DM context window.
        self._dm_context[user_key].append(f"User: {raw_content}")
        self._dm_context[user_key].append(f"SlothBrain: {response}")

        # Post response back to Discord
        if response:
            reply = response[:1900]
            if len(response) > 1900:
                reply += "\n\n[response truncated...]"  

            discord_tool = self._registry.get("discord")
            if discord_tool:
                send_result = await discord_tool.execute(action="send", content=reply)

                if send_result.ok:
                    logger.info("Discord DM response sent")
                else:
                    logger.error("Failed to post Discord DM response: %s", send_result.error)

    async def _run_bg_task(self, user_key: str, label: str, coro: Any) -> None:
        """Fire *coro* as a background asyncio task, tracking it under user_key."""
        # Cancel any existing task for this user to avoid pile-up
        existing = self._active_tasks.get(user_key)
        if existing and not existing.done():
            logger.info("Discord: user %s sent new task; previous task still running, it will continue in background", user_key)
        task = asyncio.create_task(coro)
        self._active_tasks[user_key] = task
        self._active_task_labels[user_key] = label[:80]

        def _on_done(t: asyncio.Task) -> None:
            self._active_tasks.pop(user_key, None)
            self._active_task_labels.pop(user_key, None)

        task.add_done_callback(_on_done)

    async def _run_agentic_task(self, user_key: str, task_message: str, original_content: str) -> None:
        """Run an agentic loop for *task_message* and post the result to Discord."""
        from backend.agents.agentic_loop import AgenticLoop
        used_tools: list[str] = []

        async def _capture_progress(event: dict | None) -> None:
            if not isinstance(event, dict):
                return
            if event.get("type") != "tool_call":
                return
            name = str(event.get("tool") or "").strip()
            if name:
                used_tools.append(name)

        try:
            debug_options = AgenticDebugOptions()
            lower_original = (original_content or "").strip().lower()
            if lower_original.startswith("/research "):
                debug_options.allowed_tools = ["web_search", "web_fetch"]
                debug_options.semantic_routing_enabled = True
            elif re.search(
                r"\b(file|folder|directory|filesystem|local|workspace|github|documents|downloads|desktop|repo(?:sitory)?|project(?:s)?)\b",
                task_message,
                re.IGNORECASE,
            ):
                debug_options.allowed_tools = ["file", "shell", "patch", "diff", "process"]
                debug_options.semantic_routing_enabled = False

            async with _inference_lock:
                loop = AgenticLoop(
                    main_agent=self._main_agent,
                    max_steps=10,
                    checkpoint_manager=checkpoint_manager,
                    supervisor=safety_supervisor,
                    debug_options=debug_options,
                    diagnostic=diagnostic_recorder,
                )
                result = await loop.run(task=task_message, on_progress=_capture_progress)
            # Defensive: ensure result is a valid dict before extracting
            if isinstance(result, dict):
                response = self._extract_agentic_response(result)
            else:
                response = str(result or "Task execution complete.")
        except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionRefusedError) as exc:
            response = (
                "⚠️ SlothBrain cannot reach the AI inference server. "
                "Make sure llama.cpp is running, then try again."
            )
            logger.error("Discord agentic task — inference server unreachable: %s", exc)
        except Exception as exc:
            response = f"[Agentic task error: {exc.__class__.__name__}: {str(exc)[:60]}]"
            logger.error("Discord agentic task failed: %s", exc, exc_info=True)

        response = self._clean_discord_response(response) or "I couldn't generate a useful response."
        response += self._render_tools_used_block(used_tools)
        self._dm_context[user_key].append(f"SlothBrain: {response}")
        await self._send_to_discord(response)

    async def _run_direct_task(self, user_key: str, content: str, original_content: str) -> None:
        """Run a direct chat turn and post the result to Discord."""
        _diag_run_id = f"discord-direct-{uuid.uuid4().hex[:12]}"
        _diag = diagnostic_recorder
        if _diag is not None and _diag.enabled:
            _diag.start_run(
                _diag_run_id,
                task=content,
                metadata={"mode": "direct", "source": "discord_dm", "user_key": user_key},
            )
        try:
            async with _inference_lock:
                response = await self._main_agent.process_direct(
                    user_input=content,
                    conversation_context=list(self._dm_context.get(user_key, [])),
                )
            if _diag is not None and _diag.enabled:
                _diag.finish_run(
                    _diag_run_id,
                    {"mode": "direct", "completed": True, "completion_verified": True},
                )
        except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionRefusedError) as exc:
            if _diag is not None and _diag.enabled:
                _diag.record(_diag_run_id, "run_error", error=exc.__class__.__name__)
                _diag.finish_run(_diag_run_id, {"mode": "direct", "completed": False})
            response = (
                "⚠️ SlothBrain cannot reach the AI inference server. "
                "Make sure llama.cpp is running, then try again."
            )
            logger.error("Discord direct task — inference server unreachable: %s", exc)
        except Exception as exc:
            if _diag is not None and _diag.enabled:
                _diag.record(_diag_run_id, "run_error", error=exc.__class__.__name__)
                _diag.finish_run(_diag_run_id, {"mode": "direct", "completed": False})
            response = f"[Error: {exc.__class__.__name__}]"
            logger.error("Discord direct task failed: %s", exc)

        response = self._clean_discord_response(response) or "I couldn't generate a useful response."
        self._dm_context[user_key].append(f"SlothBrain: {response}")
        await self._send_to_discord(response)

    async def _send_to_discord(self, response: str) -> None:
        """Helper: send *response* back to the configured Discord channel."""
        reply = response[:1900]
        if len(response) > 1900:
            reply += "\n\n[response truncated...]"
        discord_tool = self._registry.get("discord")
        if discord_tool:
            send_result = await discord_tool.execute(action="send", prefer_dm=True, content=reply)
            if send_result.ok:
                self._last_error = ""
                logger.info("Discord DM response sent")
            else:
                self._last_error = f"send failed: {send_result.error}"
                logger.error("Failed to post Discord DM response: %s", send_result.error)


async def _apply_effective_slot_context_budget() -> None:
    """Clamp main context size to the effective per-slot budget.

    Some llama.cpp configurations split total --ctx-size across -np slots.
    This keeps internal agent context settings aligned with the actual
    available per-slot context to avoid overflow thrashing.
    """
    budget: Optional[int] = None

    manual_cap = int(getattr(settings, "llama_slot_context_cap", 0) or 0)
    if manual_cap > 0:
        budget = manual_cap

    try:
        slot_info = await slot_manager.get_slot_info()
        slots = slot_info.get("slots", []) if isinstance(slot_info, dict) else []
        n_ctx_values = [
            int(s.get("n_ctx", 0))
            for s in slots
            if isinstance(s, dict) and int(s.get("n_ctx", 0)) > 0
        ]
        if n_ctx_values:
            inferred = min(n_ctx_values)
            budget = inferred if budget is None else min(budget, inferred)
    except Exception as exc:
        logger.debug("Could not infer slot context budget from /slots: %s", exc.__class__.__name__)

    if budget is None or budget <= 0:
        return

    old_main = settings.main_context_size
    settings.main_context_size = min(old_main, budget)

    if settings.main_context_size != old_main:
        logger.warning(
            "Context size clamped to effective slot budget=%d (main: %d->%d)",
            budget,
            old_main,
            settings.main_context_size,
        )

# ---------------------------------------------------------------------------
# Singletons – populated during lifespan startup
# ---------------------------------------------------------------------------
llama_client: LlamaClient
slot_manager: SlotManager
resource_manager: ResourceManager
memory: LanceDBMemory
main_agent: MainAgent
benchmark_suite: BenchmarkSuite
preset_manager: PresetManager
agent_registry: AgentRegistry
server_manager: ServerManager
audit_log: AuditLog
approval_queue: ApprovalQueue
checkpoint_manager: CheckpointManager
safety_supervisor: SafetySupervisor
tool_registry: ToolRegistry
discord_dm_bridge: "DiscordDMBridge | None" = None
diagnostic_recorder: "DiagnosticRecorder | None" = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global llama_client, slot_manager, resource_manager
    global memory, main_agent, benchmark_suite
    global preset_manager, agent_registry, server_manager, audit_log, approval_queue
    global checkpoint_manager, safety_supervisor, tool_registry, discord_dm_bridge
    global diagnostic_recorder

    audit_log = AuditLog()
    approval_queue = ApprovalQueue(max_entries=settings.max_pending_approvals)
    server_manager = ServerManager(config=settings, audit_log=audit_log)

    llama_client = LlamaClient(host=settings.llama_host, port=settings.llama_port)
    slot_manager = SlotManager(llama_client=llama_client)
    slot_manager.set_slot_info_cache_ttl(settings.slot_info_cache_ttl_seconds)
    resource_manager = ResourceManager(config=settings, llama_client=llama_client)

    # Run auto-start as a non-blocking background task so startup completes immediately
    # regardless of whether llama.cpp is reachable yet.
    asyncio.create_task(_auto_start_llama_if_missing())

    await slot_manager.assign_main(settings.main_slot)
    await _apply_effective_slot_context_budget()

    try:
        memory = LanceDBMemory(
            db_path=settings.lancedb_path,
            embedding_model=settings.embedding_model,
        )
    except ImportError as exc:
        raise RuntimeError(
            "LanceDB memory is required but failed to initialize. "
            "Install compatible dependencies (for example: pip install -r requirements.txt). "
            f"Original error: {exc}"
        ) from exc

    main_agent = MainAgent(
        slot_manager=slot_manager,
        memory=memory,  # type: ignore[arg-type]
        config=settings,
    )
    benchmark_suite = BenchmarkSuite(llama_client=llama_client, config=settings)

    preset_manager = PresetManager()
    agent_registry = AgentRegistry(
        preset_manager=preset_manager,
        llama_client=llama_client,
        memory=memory,
    )
    # Give the MainAgent a reference to the registry so it can spawn sub-agents
    main_agent.set_registry(agent_registry)

    # ── Tool system ──────────────────────────────────────────────────────────
    semantic_model = (
        settings.semantic_tool_routing_embedding_model.strip()
        or settings.embedding_model
    )
    semantic_router = SemanticRouter(
        embedding_model=semantic_model,
        top_k=settings.semantic_tool_routing_top_k,
        min_similarity=settings.semantic_tool_routing_min_similarity,
        enabled=settings.semantic_tool_routing_enabled,
        critical_tools=settings.semantic_tool_routing_critical_tools,
    )
    tool_registry = ToolRegistry(
        semantic_router=semantic_router,
        semantic_top_k=settings.semantic_tool_routing_top_k,
        semantic_min_similarity=settings.semantic_tool_routing_min_similarity,
        critical_bypass_tools=settings.semantic_tool_routing_critical_tools,
    )
    _register_tools(tool_registry, settings, audit_log, memory, agent_registry, llama_client)
    main_agent.set_tool_registry(tool_registry)

    # Safety infrastructure for the agentic loop
    checkpoint_manager = CheckpointManager()

    # Offline diagnostics recorder (on by default; set SLOTHBRAIN_DIAGNOSTICS_ENABLED=false to disable)
    diagnostic_recorder = DiagnosticRecorder(
        output_dir=settings.diagnostics_output_dir,
        enabled=settings.diagnostics_enabled,
    )
    if diagnostic_recorder.enabled:
        logger.info(
            "Diagnostic recorder enabled; bundles will be written to %s",
            diagnostic_recorder._output_dir,
        )
    else:
        logger.info("Diagnostic recorder disabled.")
    safety_supervisor = SafetySupervisor(
        llama_client=llama_client,
        checkpoint_manager=checkpoint_manager,
        poll_interval=settings.supervisor_poll_interval,
        step_timeout=settings.supervisor_step_timeout,
        server_manager=server_manager,
        slowdown_monitor_enabled=settings.supervisor_slowdown_monitor_enabled,
        slowdown_threshold_tps=settings.supervisor_slowdown_threshold_tps,
        slowdown_consecutive_polls=settings.supervisor_slowdown_consecutive_polls,
        slowdown_restart_enabled=settings.supervisor_slowdown_restart_enabled,
        slowdown_cooldown_seconds=settings.supervisor_slowdown_cooldown_seconds,
        max_repeated_tool_calls=settings.supervisor_max_repeated_tool_calls,
        max_failed_tool_calls=settings.supervisor_max_failed_tool_calls,
        max_no_progress_steps=settings.supervisor_max_no_progress_steps,
        max_empty_or_malformed=settings.supervisor_max_empty_responses,
        max_give_up_signals=settings.supervisor_max_give_up_signals,
    )
    safety_supervisor.start()

    if settings.enable_server_watchdog:
        server_manager.start_watchdog()

    # Discord DM listener
    discord_dm_bridge = DiscordDMBridge(main_agent=main_agent, config=settings, registry=tool_registry)
    discord_dm_bridge.start()

    yield

    # Shutdown – stop all background tasks
    discord_dm_bridge.stop()
    safety_supervisor.stop()
    server_manager.stop_watchdog()
    agent_registry.destroy_all()
    # Stop the scheduler background loop if it was started
    try:
        from backend.tools.impl.scheduler_tool import SchedulerTool
        sched = tool_registry.get("scheduler")
        if isinstance(sched, SchedulerTool):
            sched.stop()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tool registration helper
# ---------------------------------------------------------------------------

def _register_tools(
    registry: "ToolRegistry",
    config: Any,
    audit_log: Any,
    memory: Any,
    agent_registry: Any,
    llama_client: Any,
) -> None:
    """Construct and register all built-in tools into *registry*."""
    from backend.tools.impl.ui_tool import UITool
    from backend.tools.impl.screenshot_tool import ScreenshotTool
    from backend.tools.impl.image_analysis_tool import ImageAnalysisTool
    from backend.tools.impl.web_fetch_tool import WebFetchTool
    from backend.tools.impl.web_search_tool import WebSearchTool
    from backend.tools.impl.shell_tool import ShellTool
    from backend.tools.impl.process_tool import ProcessTool
    from backend.tools.impl.code_exec_tool import CodeExecTool
    from backend.tools.impl.file_tool import FileTool
    from backend.tools.impl.patch_tool import PatchTool
    from backend.tools.impl.diff_tool import DiffTool
    from backend.tools.impl.memory_search_tool import MemorySearchTool
    from backend.tools.impl.session_graph_tool import SessionGraphTool
    from backend.tools.impl.sub_agent_tool import SubAgentTool
    from backend.tools.impl.agent_list_tool import AgentListTool
    from backend.tools.impl.session_tool import SessionTool
    from backend.tools.impl.scheduler_tool import SchedulerTool
    from backend.tools.impl.discord_tool import DiscordTool
    from backend.tools.impl.workspace_index_tool import WorkspaceIndexTool
    from backend.tools.plugin_loader import load_plugins

    # Vision / desktop. Image analysis is available even when desktop input
    # tools are disabled, so callers can pass explicit image_b64 payloads.
    desktop_controller = None
    if getattr(config, "desktop_tools_enabled", True):
        try:
            from backend.vision.controller import DesktopController
            desktop_controller = DesktopController()
            registry.register(UITool(controller=desktop_controller))
            registry.register(ScreenshotTool(controller=desktop_controller))
        except Exception as exc:
            logger.warning("Desktop tools unavailable: %s", exc)
    else:
        logger.info("Desktop tools disabled by configuration")

    registry.register(
        ImageAnalysisTool(
            llama_client=llama_client,
            controller=desktop_controller,
            backend=getattr(config, "image_analysis_backend", "auto"),
            llama_slot_id=int(getattr(config, "image_analysis_llama_slot_id", 0)),
            cpu_max_text_chars=int(getattr(config, "image_analysis_cpu_max_text_chars", 4000)),
        )
    )

    # Web
    registry.register(WebFetchTool())
    registry.register(WebSearchTool(searxng_url=getattr(config, "searxng_url", "")))

    # Shell / process / code
    registry.register(ShellTool(config=config, audit_log=audit_log))
    registry.register(ProcessTool(config=config, audit_log=audit_log))
    # CodeExecTool requires explicit opt-in (exec() is not a true sandbox).
    if getattr(config, "code_exec_enabled", False):
        registry.register(CodeExecTool())
    else:
        logger.info(
            "CodeExecTool not registered (set SLOTHBRAIN_CODE_EXEC_ENABLED=true to enable)"
        )

    # File system + workspace indexing
    workspace_index_tool: Optional[WorkspaceIndexTool] = None
    if getattr(config, "workspace_index_enabled", True):
        try:
            from backend.memory.workspace_indexer import WorkspaceIndexer
            ws_db_path = (
                getattr(config, "workspace_index_db_path", "")
                or getattr(config, "lancedb_path", "./data/lancedb")
            )
            ws_indexer = WorkspaceIndexer(
                db_path=ws_db_path,
                embedding_model=getattr(config, "embedding_model", "all-MiniLM-L6-v2"),
            )
            workspace_index_tool = WorkspaceIndexTool(indexer=ws_indexer)
            registry.register(workspace_index_tool)
        except ImportError as exc:
            raise RuntimeError(
                "Workspace indexing is enabled but its LanceDB dependencies failed to load. "
                "Install compatible dependencies (for example: pip install -r requirements.txt). "
                f"Original error: {exc}"
            ) from exc
    else:
        workspace_index_tool = WorkspaceIndexTool(indexer=None)

    registry.register(FileTool(config=config, workspace_index=workspace_index_tool))
    registry.register(PatchTool(config=config))
    registry.register(DiffTool(config=config))

    # Memory / knowledge
    registry.register(MemorySearchTool(memory=memory))
    registry.register(SessionGraphTool(memory=memory))

    # Agent orchestration
    registry.register(SubAgentTool(registry=agent_registry))
    registry.register(AgentListTool(registry=agent_registry))
    registry.register(SessionTool(registry=agent_registry))

    async def _trigger_scheduled_task(task: str) -> None:
        """Run a persisted scheduler job through the full agentic loop."""
        from backend.agents.agentic_loop import AgenticLoop

        task_message = _strip_agentic_prefix((task or "").strip())
        if not task_message:
            logger.warning("Scheduler skipped empty task")
            return

        used_tools: list[str] = []

        async def _capture_progress(event: dict | None) -> None:
            if not isinstance(event, dict) or event.get("type") != "tool_call":
                return
            name = str(event.get("tool") or "").strip()
            if name:
                used_tools.append(name)

        try:
            async with _inference_lock:
                loop = AgenticLoop(
                    main_agent=main_agent,
                    max_steps=10,
                    checkpoint_manager=checkpoint_manager,
                    supervisor=safety_supervisor,
                    debug_options=AgenticDebugOptions(),
                    diagnostic=diagnostic_recorder,
                )
                result = await loop.run(task=task_message, on_progress=_capture_progress)
            if isinstance(result, dict):
                response = DiscordDMBridge._extract_agentic_response(result)
            else:
                response = str(result or "Scheduled task complete.")
        except Exception as exc:
            response = f"[Scheduled task error: {exc.__class__.__name__}: {str(exc)[:80]}]"
            logger.error("Scheduled task failed: %s", exc, exc_info=True)

        response = _sanitize_user_facing_response(response) or "Scheduled task completed, but no useful response was generated."
        response += DiscordDMBridge._render_tools_used_block(used_tools)

        discord_tool = registry.get("discord")
        if discord_tool is not None:
            try:
                await discord_tool.execute(
                    action="send",
                    prefer_dm=True,
                    content=f"Scheduled task complete:\n{response}"[:1900],
                )
            except Exception as exc:
                logger.warning("Failed to send scheduled task result to Discord: %s", exc.__class__.__name__)

    # Scheduler
    sched = SchedulerTool(
        on_trigger=_trigger_scheduled_task,
        default_timezone=getattr(config, "scheduler_timezone", "America/New_York"),
    )
    registry.register(sched)
    sched.start()

    # Discord (optional — skipped if no credentials configured)
    webhook = getattr(config, "discord_webhook_url", "")
    bot_token = getattr(config, "discord_bot_token", "")
    channel_id = getattr(config, "discord_channel_id", "")
    owner_user_id = getattr(config, "discord_owner_user_id", "")
    if webhook or bot_token:
        registry.register(
            DiscordTool(
                webhook_url=webhook,
                bot_token=bot_token,
                channel_id=channel_id,
                owner_user_id=owner_user_id,
            )
        )

    # Dynamic plugins
    loaded = load_plugins(registry)
    if loaded:
        logger.info("Loaded %d plugin tool(s)", loaded)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
# Increase WebSocket buffer sizes to handle large tool results (screenshots up to ~2MB base64)
# Default is 1MB which was causing "message too big" errors with full-screen JPEG captures
_WS_MAX_SIZE = 16 * 1024 * 1024  # 16 MB
_WS_MAX_QUEUE = 32

app = FastAPI(title="SlothBrain", lifespan=lifespan)

# Configure WebSocket parameters for larger payloads
import uvicorn
# Note: These will be applied when the app is started via uvicorn with:
# uvicorn.run(..., ws_max_size=_WS_MAX_SIZE, ws_max_queue=_WS_MAX_QUEUE)

# Inference concurrency guard. Keep at 1 by default for safety, but allow
# operators to raise it (for example 2) when llama.cpp is configured for
# parallel execution capacity.
_inference_lock: asyncio.Semaphore = asyncio.Semaphore(max(1, int(getattr(settings, "inference_concurrency", 1))))


async def _run_agentic_with_cancel(
    http_request: Request,
    loop_coro,
) -> dict:
    """Run *loop_coro* and cancel it if the HTTP client disconnects.

    Without this guard, timed-out or closed client connections leave an
    orphaned coroutine running on the backend that keeps consuming inference
    slots and thrashes the KV cache alongside any new requests.
    """
    task = asyncio.ensure_future(loop_coro)
    try:
        while not task.done():
            # Poll for client disconnect every 0.5 s.
            done, _ = await asyncio.wait({task}, timeout=0.5)
            if done:
                break
            if await http_request.is_disconnected():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise HTTPException(status_code=499, detail="Client disconnected")
        return task.result()
    except asyncio.CancelledError:
        task.cancel()
        raise


@app.middleware("http")
async def protect_api(request: Request, call_next):
    path = request.url.path
    if path != "/health" and path.startswith("/api"):
        configured_key = settings.api_key.strip()
        header_key = request.headers.get("x-api-key", "").strip()
        bearer_key = _parse_bearer_token(request.headers.get("authorization"))

        if configured_key:
            if header_key != configured_key and bearer_key != configured_key:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Unauthorized"},
                )
        else:
            client_host = request.client.host if request.client else None
            if not _is_loopback_host(client_host):
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Remote API access denied without configured api_key"},
                )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    max_steps: int = 10
    mode: str = "auto"  # "auto" | "direct" | "agentic"
    debug: "Optional[AgenticDebugRequest]" = None


class ModeRequest(BaseModel):
    mode: str


class SettingsUpdate(BaseModel):
    llama_host: Optional[str] = None
    llama_port: Optional[int] = None
    main_slot: Optional[int] = None
    main_context_size: Optional[int] = None
    idle_kv_quant: Optional[str] = None
    active_kv_quant: Optional[str] = None
    vram_threshold_mb: Optional[int] = None
    ram_threshold_mb: Optional[int] = None
    embedding_model: Optional[str] = None
    llama_server_path: Optional[str] = None
    llama_server_args: Optional[list[str]] = None
    llama_launch_profiles: Optional[list[dict[str, Any]]] = None
    default_launch_profile_id: Optional[str] = None
    max_context_size: Optional[int] = None
    max_slots: Optional[int] = None
    max_restarts_per_hour: Optional[int] = None
    require_approval_server_restart: Optional[bool] = None
    allow_auto_recovery_restart_without_approval: Optional[bool] = None
    require_approval_kv_cache_change: Optional[bool] = None
    require_approval_large_context_increase: Optional[bool] = None
    require_approval_emergency_stop: Optional[bool] = None
    semantic_tool_routing_enabled: Optional[bool] = None
    semantic_tool_routing_embedding_model: Optional[str] = None
    semantic_tool_routing_top_k: Optional[int] = None
    semantic_tool_routing_min_similarity: Optional[float] = None
    semantic_tool_routing_critical_tools: Optional[list[str]] = None
    scheduler_timezone: Optional[str] = None
    debug_loop_enabled: Optional[bool] = None
    debug_loop_llm_only: Optional[bool] = None
    debug_loop_planning_enabled: Optional[bool] = None
    debug_loop_rolling_context_enabled: Optional[bool] = None
    debug_loop_tool_calls_enabled: Optional[bool] = None
    debug_loop_semantic_routing_enabled: Optional[bool] = None
    debug_loop_checkpointing_enabled: Optional[bool] = None
    debug_loop_supervisor_enabled: Optional[bool] = None
    debug_loop_per_event_logging: Optional[bool] = None
    debug_loop_allowed_tools: Optional[list[str]] = None
    diagnostics_enabled: Optional[bool] = None
    diagnostics_output_dir: Optional[str] = None


class BenchmarkRequest(BaseModel):
    type: str = "all"  # "speed" | "vram" | "slots" | "all"


class PresetCreate(BaseModel):
    name: str
    description: str = ""
    system_prompt: str
    context_size: int = 8192
    temperature: float = 0.7
    max_tokens: int = 1024


class PresetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    context_size: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class AgentChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = None     # per-call override


class SpawnRequest(BaseModel):
    context_size: Optional[int] = None   # deprecated (ignored; context comes from llama.cpp slot split)
    max_tokens: Optional[int] = None     # override preset default
    task_description: str = ""


class AgenticRequest(BaseModel):
    task: str
    max_steps: int = 10
    debug: "Optional[AgenticDebugRequest]" = None


class AgenticDebugRequest(BaseModel):
    enabled: Optional[bool] = None
    llm_only: Optional[bool] = None
    planning_enabled: Optional[bool] = None
    rolling_context_enabled: Optional[bool] = None
    tool_calls_enabled: Optional[bool] = None
    semantic_routing_enabled: Optional[bool] = None
    checkpointing_enabled: Optional[bool] = None
    supervisor_enabled: Optional[bool] = None
    per_event_logging: Optional[bool] = None
    allowed_tools: Optional[list[str]] = None


class ToolRunRequest(BaseModel):
    name: str
    args: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Bounds for the number of agentic steps accepted via API.
_MIN_AGENTIC_STEPS = 1
_MAX_AGENTIC_STEPS = 20


def _clamp_steps(n: int) -> int:
    """Clamp a requested step count to the valid API range."""
    return min(max(n, _MIN_AGENTIC_STEPS), _MAX_AGENTIC_STEPS)


async def _choose_idle_parallel_slot_id() -> int | None:
    """Return an idle non-main slot ID, or None if no parallel slot is available."""
    try:
        slots = await llama_client.get_slots()
    except Exception:
        return None

    candidates: list[int] = []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        sid = int(slot.get("id", -1))
        if sid <= 0:
            continue
        next_token = slot.get("next_token")
        has_next = bool(next_token.get("has_next_token")) if isinstance(next_token, dict) else False
        if not has_next:
            candidates.append(sid)
    return min(candidates) if candidates else None


def _raise_service_unavailable(exc: Exception, context: str) -> None:
    logger.warning("%s failed: %s (%s)", context, exc.__class__.__name__, str(exc))
    raise HTTPException(
        status_code=503,
        detail=f"{context} unavailable. Ensure llama.cpp server is running and reachable.",
    ) from exc


async def _ensure_llama_available(context: str) -> None:
    """Perform a quick preflight check so API callers get a clean 503 early."""
    try:
        await llama_client.health()
    except (httpx.HTTPError, RuntimeError, ValueError, OSError) as exc:
        _raise_service_unavailable(exc, context)


def _parse_bearer_token(authorization_header: Optional[str]) -> str:
    if not authorization_header:
        return ""
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def _is_loopback_host(host: Optional[str]) -> bool:
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_AGENTIC_COMMAND_PREFIXES = ("/task ", "/agentic ", "/research ", "/ralph ")
_RESEARCH_INTRO = (
    "Research the following topic in depth using web_search and web_fetch tools. "
    "Search multiple angles, fetch key pages, then compile a comprehensive report with cited sources: "
)
_RALPH_INTRO = (
    "You are a code improvement agent (Ralph loop). "
    "Use workspace_index and file tools to find the project or code described. "
    "Read and analyze all relevant source files, identify concrete bugs and improvements, "
    "apply patches iteratively using the patch tool, and provide a full summary of every change made. "
    "Target: "
)


def _strip_agentic_prefix(text: str) -> str:
    """Strip /task, /agentic, /research, or /ralph prefix and return the inner task string.

    For /research and /ralph the inner task is framed with a directive.
    """
    lower = text.lower()
    if lower in {"/task", "/agentic", "/research", "/ralph"}:
        return ""
    for prefix in ("/task ", "/agentic "):
        if lower.startswith(prefix):
            return text[len(prefix):].strip()
    if lower.startswith("/research "):
        inner = text[len("/research "):].strip()
        return _RESEARCH_INTRO + inner
    if lower.startswith("/ralph "):
        inner = text[len("/ralph "):].strip()
        return _RALPH_INTRO + inner
    return text


def _build_discord_task(content: str) -> str:
    """Strip command prefix from Discord message and apply task framing."""
    return _strip_agentic_prefix(content.strip())


def _parse_llama_launch_command(command: str) -> tuple[str, list[str]]:
    """Parse a saved launch command string into executable path and args."""
    parts = shlex.split(command or "", posix=False)
    if not parts:
        raise ValueError("Launch command is empty")
    exe = parts[0].strip().strip('"')
    args = [p for p in parts[1:]]
    return exe, args


async def _auto_start_llama_if_missing() -> None:
    """Start llama.cpp on app boot when not already running.

    Runs as a non-blocking background task — never delays app startup.
    - If /health passes, do nothing.
    - If not running and a default launch card exists, derive executable/args.
    - Attempt server_manager.start(); log warnings on failure.
    """
    try:
        await asyncio.wait_for(llama_client.health(), timeout=3.0)
        return
    except Exception:
        pass

    if not (settings.llama_server_path or "").strip():
        profiles = getattr(settings, "llama_launch_profiles", []) or []
        default_id = (getattr(settings, "default_launch_profile_id", "") or "").strip()
        chosen = None
        if default_id:
            for item in profiles:
                if str(item.get("id", "")).strip() == default_id:
                    chosen = item
                    break
        if chosen is None and profiles:
            chosen = profiles[0]

        if chosen is not None:
            try:
                exe, args = _parse_llama_launch_command(str(chosen.get("command", "")))
                settings.llama_server_path = exe
                settings.llama_server_args = args
                logger.info("Applied default launch profile '%s' for startup", str(chosen.get("name", "default")))
            except Exception as exc:
                logger.warning("Failed to parse default launch profile: %s", exc.__class__.__name__)

    if not (settings.llama_server_path or "").strip():
        logger.info("llama.cpp not reachable and no server path/profile configured — skipping auto-start")
        return

    try:
        await server_manager.start()
        await asyncio.sleep(1.0)
        await asyncio.wait_for(llama_client.health(), timeout=5.0)
        logger.info("Auto-started llama.cpp on app startup")
    except Exception as exc:
        logger.warning("Auto-start on app startup failed: %s", exc.__class__.__name__)


def _should_use_agentic_mode(message: str, max_steps: int, mode: str) -> bool:
    """Decide whether /api/chat should run the full agentic loop.

    Rules:
    - mode=direct forces direct response.
    - mode=agentic forces agentic loop.
    - mode=auto uses simple heuristics so normal chat is direct by default.
    """
    normalized_mode = (mode or "auto").strip().lower()
    if normalized_mode == "direct":
        return False
    if normalized_mode == "agentic":
        return True

    msg = (message or "").strip()
    lower = msg.lower()

    # Explicit trigger prefixes for power users.
    if any(lower.startswith(p) for p in _AGENTIC_COMMAND_PREFIXES):
        return True

    if _LOCAL_FILESYSTEM_ACTION_RE.search(msg):
        return True

    # Otherwise default to direct chat for quick responsiveness. Natural-language
    # task detection for Discord is handled by the DM bridge separately.
    return False


def _build_debug_options(debug: "Optional[AgenticDebugRequest]") -> AgenticDebugOptions:
    defaults = AgenticDebugOptions(
        enabled=settings.debug_loop_enabled,
        llm_only=settings.debug_loop_llm_only,
        planning_enabled=settings.debug_loop_planning_enabled,
        rolling_context_enabled=settings.debug_loop_rolling_context_enabled,
        tool_calls_enabled=settings.debug_loop_tool_calls_enabled,
        semantic_routing_enabled=settings.debug_loop_semantic_routing_enabled,
        checkpointing_enabled=settings.debug_loop_checkpointing_enabled,
        supervisor_enabled=settings.debug_loop_supervisor_enabled,
        per_event_logging=settings.debug_loop_per_event_logging,
        allowed_tools=list(settings.debug_loop_allowed_tools),
    )
    if debug is None:
        return defaults

    payload = debug.model_dump(exclude_none=True)
    for key, value in payload.items():
        setattr(defaults, key, value)
    return defaults



async def _try_handle_simple_slash_command(task_message: str) -> str | None:
    """Handle basic slash commands deterministically without LLM dependence."""
    text = (task_message or "").strip()
    if not text.startswith("/"):
        return None

    command = text.split()[0].lower()
    if command in {"/task", "/agentic", "/research", "/ralph"}:
        # Keep loop commands on the regular agentic path.
        return None

    if command == "/status":
        try:
            stats = await resource_manager.get_system_stats()
        except Exception:
            return "System status is available. Use /api/status for full metrics."

        ram = stats.get("ram_used_mb", "n/a") if isinstance(stats, dict) else "n/a"
        vram = stats.get("vram_used_mb", "n/a") if isinstance(stats, dict) else "n/a"
        cpu = stats.get("cpu_percent", "n/a") if isinstance(stats, dict) else "n/a"
        return f"System status: CPU={cpu}%, RAM={ram}MB, VRAM={vram}MB."

    if command in {"/restart", "/restart_llama", "/restart-llama"}:
        try:
            await server_manager.restart(actor="api")
            return "Requested llama.cpp restart successfully."
        except Exception as exc:
            return f"Failed to restart llama.cpp: {exc}"

    return f"Unknown command '{command}'. Try /status, /restart, /task <goal>, /research <topic>, or /ralph <goal>."


# ---------------------------------------------------------------------------
# Existing endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.get("/api/discord/debug")
async def discord_debug() -> dict:
    """Expose Discord DM bridge state for diagnostics."""
    bridge = discord_dm_bridge
    if bridge is None:
        return {"bridge": "not_initialized"}
    task_state = "none"
    if bridge._task is not None:
        if bridge._task.done():
            exc = bridge._task.exception() if not bridge._task.cancelled() else None
            task_state = f"done (exception={exc})"
        else:
            task_state = "running"
    discord_tool = tool_registry.get("discord") if tool_registry else None
    # Quick history fetch to test connectivity
    history_result = None
    if discord_tool is not None:
        try:
            r = await discord_tool.execute(action="history", limit=3)
            history_result = {"ok": r.ok, "error": r.error, "count": r.output.get("count") if r.ok and r.output else None}
        except Exception as exc:
            history_result = {"error": str(exc)}
    return {
        "bridge_running": bridge._running,
        "task_state": task_state,
        "bot_user_id": bridge._bot_user_id,
        "history_primed": bridge._history_primed,
        "last_poll_at": bridge._last_poll_at,
        "last_processed_id": bridge._last_processed_id,
        "last_error": bridge._last_error,
        "processed_ids_count": len(bridge._processed_ids),
        "discord_tool_registered": discord_tool is not None,
        "history_test": history_result,
    }


@app.get("/api/status")
async def get_status() -> dict:
    stats = await resource_manager.get_system_stats()
    try:
        slot_info = await slot_manager.get_slot_info()
    except Exception as exc:
        logger.warning("Failed to fetch slot info: %s", exc.__class__.__name__)
        slot_info = {"main": settings.main_slot, "slots": []}
    return {**stats, "slots": slot_info}


@app.post("/api/chat")
async def chat(http_request: Request, request: ChatRequest) -> dict:
    """Chat endpoint with auto-routing between direct and agentic modes."""
    slash_response = await _try_handle_simple_slash_command(request.message)
    if slash_response is not None:
        return {
            "agent": "direct",
            "response": _sanitize_user_facing_response(slash_response),
            "handoff": False,
            "result": {
                "mode": "direct",
                "completed": True,
                "deterministic": True,
            },
        }

    use_agentic = _should_use_agentic_mode(request.message, request.max_steps, request.mode)

    if not use_agentic:
        await _ensure_llama_available("Direct chat")
        _diag_run_id = f"direct-{uuid.uuid4().hex[:12]}"
        if diagnostic_recorder is not None and diagnostic_recorder.enabled:
            diagnostic_recorder.start_run(
                _diag_run_id,
                task=request.message,
                metadata={"mode": "direct", "source": "/api/chat"},
            )
        async with _inference_lock:
            try:
                response = await main_agent.process_direct(user_input=request.message)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                if diagnostic_recorder is not None and diagnostic_recorder.enabled:
                    diagnostic_recorder.record(_diag_run_id, "run_error", error=str(exc))
                    diagnostic_recorder.finish_run(_diag_run_id, {"mode": "direct", "completed": False})
                _raise_service_unavailable(exc, "Direct chat")
        if diagnostic_recorder is not None and diagnostic_recorder.enabled:
            diagnostic_recorder.finish_run(
                _diag_run_id,
                {"mode": "direct", "completed": True, "completion_verified": True},
            )
        response = _sanitize_user_facing_response(response)
        if not response:
            response = "I can do that. Please share the response text you'd like me to compact."
        return {
            "agent": "direct",
            "response": response,
            "handoff": False,
            "result": {
                "mode": "direct",
                "completed": True,
            },
        }

    from backend.agents.agentic_loop import AgenticLoop

    await _ensure_llama_available("Agentic loop")

    task_message = _strip_agentic_prefix(request.message.strip())

    async with _inference_lock:
        debug_options = _build_debug_options(request.debug)
        if re.search(
            r"\b(file|folder|directory|filesystem|local|workspace|github|documents|downloads|desktop|repo(?:sitory)?|project(?:s)?)\b",
            task_message,
            re.IGNORECASE,
        ):
            debug_options.allowed_tools = ["file", "shell", "patch", "diff", "process"]
            debug_options.semantic_routing_enabled = False
        loop = AgenticLoop(
            main_agent=main_agent,
            max_steps=_clamp_steps(request.max_steps),
            checkpoint_manager=checkpoint_manager,
            supervisor=safety_supervisor,
            debug_options=debug_options,
            diagnostic=diagnostic_recorder,
        )
        try:
            result = await _run_agentic_with_cancel(
                http_request, loop.run(task=task_message)
            )
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            _raise_service_unavailable(exc, "Agentic loop")

    response_text = _sanitize_user_facing_response(str(result.get("summary", "")))
    return {
        "agent": "agentic",
        "response": response_text,
        "handoff": False,
        "result": result,
    }


@app.post("/api/chat/direct")
async def direct_chat(request: ChatRequest) -> dict:
    """Explicit direct chat endpoint (single-shot, no task planning loop)."""
    slash_response = await _try_handle_simple_slash_command(request.message)
    if slash_response is not None:
        return {
            "agent": "direct",
            "response": _sanitize_user_facing_response(slash_response),
            "handoff": False,
            "result": {
                "mode": "direct",
                "completed": True,
                "deterministic": True,
            },
        }

    await _ensure_llama_available("Direct chat")
    _diag_run_id = f"direct-{uuid.uuid4().hex[:12]}"
    if diagnostic_recorder is not None and diagnostic_recorder.enabled:
        diagnostic_recorder.start_run(
            _diag_run_id,
            task=request.message,
            metadata={"mode": "direct", "source": "/api/chat/direct"},
        )
    async with _inference_lock:
        try:
            response = await main_agent.process_direct(user_input=request.message)
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            if diagnostic_recorder is not None and diagnostic_recorder.enabled:
                diagnostic_recorder.record(_diag_run_id, "run_error", error=str(exc))
                diagnostic_recorder.finish_run(_diag_run_id, {"mode": "direct", "completed": False})
            _raise_service_unavailable(exc, "Direct chat")
    if diagnostic_recorder is not None and diagnostic_recorder.enabled:
        diagnostic_recorder.finish_run(
            _diag_run_id,
            {"mode": "direct", "completed": True, "completion_verified": True},
        )
    response = _sanitize_user_facing_response(response)
    if not response:
        response = "I can do that. Please share the response text you'd like me to compact."
    return {
        "agent": "direct",
        "response": response,
        "handoff": False,
        "result": {
            "mode": "direct",
            "completed": True,
        },
    }


@app.post("/api/chat/agentic")
async def agentic_chat(http_request: Request, request: AgenticRequest) -> dict:
    """Run a multi-step agentic task loop and return the full result.

    The MainAgent plans the task and executes each step in sequence.
    The SafetySupervisor watches for stalls and the CheckpointManager saves
    state before every step so the loop can recover cleanly on failure.
    """
    from backend.agents.agentic_loop import AgenticLoop

    await _ensure_llama_available("Agentic loop")

    async with _inference_lock:
        debug_options = _build_debug_options(request.debug)
        if re.search(
            r"\b(file|folder|directory|filesystem|local|workspace|github|documents|downloads|desktop|repo(?:sitory)?|project(?:s)?)\b",
            request.task,
            re.IGNORECASE,
        ):
            debug_options.allowed_tools = ["file", "shell", "patch", "diff", "process"]
            debug_options.semantic_routing_enabled = False
        loop = AgenticLoop(
            main_agent=main_agent,
            max_steps=_clamp_steps(request.max_steps),
            checkpoint_manager=checkpoint_manager,
            supervisor=safety_supervisor,
            debug_options=debug_options,
            diagnostic=diagnostic_recorder,
        )
        try:
            result = await _run_agentic_with_cancel(
                http_request, loop.run(task=request.task)
            )
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            _raise_service_unavailable(exc, "Agentic loop")
    audit_log.record(
        action="agentic_task",
        actor="api",
        details=request.task[:100],
    )
    return result


# ---------------------------------------------------------------------------
# Diagnostic bundle endpoints
# ---------------------------------------------------------------------------

@app.get("/api/diagnostics/runs")
async def list_diagnostic_runs() -> dict:
    """List all diagnostic run bundles (complete and partial), newest first.

    Each entry contains ``run_id``, ``task``, ``started_at``,
    ``duration_seconds``, ``event_count``, ``completion_verified``,
    ``total_steps``, ``status`` (``"complete"`` | ``"partial"``), and
    ``finding_count`` (if analysis ran).
    """
    if diagnostic_recorder is None or not diagnostic_recorder.enabled:
        return {"enabled": False, "runs": []}
    runs = diagnostic_recorder.list_runs()
    # Enrich with finding_count from analysis.json where available.
    for run in runs:
        analysis = diagnostic_recorder.get_analysis(run["run_id"])
        run["finding_count"] = analysis.get("finding_count") if analysis else None
    return {"enabled": True, "runs": runs}


@app.get("/api/diagnostics/runs/{run_id}")
async def get_diagnostic_run(run_id: str) -> dict:
    """Return the full diagnostic bundle for a specific run_id.

    The bundle is the AI-readable JSON file written to
    ``diagnostics/runs/{run_id}/bundle.json``.  Returns 404 if the run does
    not exist or diagnostics are disabled.
    """
    if diagnostic_recorder is None or not diagnostic_recorder.enabled:
        raise HTTPException(status_code=404, detail="Diagnostics not enabled")
    bundle = diagnostic_recorder.get_bundle(run_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return bundle


@app.get("/api/diagnostics/runs/{run_id}/analysis")
async def get_diagnostic_analysis(run_id: str) -> dict:
    """Return the pre-computed failure-mode analysis for a specific run.

    The analysis is generated automatically when the bundle is written and
    contains machine-readable findings with severity, confidence, and
    suggested code areas.  Returns 404 if the run or analysis does not exist.
    """
    if diagnostic_recorder is None or not diagnostic_recorder.enabled:
        raise HTTPException(status_code=404, detail="Diagnostics not enabled")
    analysis = diagnostic_recorder.get_analysis(run_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"Analysis for run {run_id!r} not found")
    return analysis


@app.get("/api/diagnostics/runs/{run_id}/review-prompt")
async def get_diagnostic_review_prompt(run_id: str) -> dict:
    """Return the model review prompt for a specific run.

    This is a ready-to-paste prompt designed to be sent to a second AI model
    (Qwen / GPT / Claude / etc.) for targeted failure diagnosis.  Contains:
    - condensed event trace
    - automatic findings summary
    - focused diagnostic instructions

    Returns ``{"run_id": ..., "prompt": ...}`` or 404.
    """
    if diagnostic_recorder is None or not diagnostic_recorder.enabled:
        raise HTTPException(status_code=404, detail="Diagnostics not enabled")
    prompt = diagnostic_recorder.get_review_prompt(run_id)
    if prompt is None:
        raise HTTPException(
            status_code=404, detail=f"Review prompt for run {run_id!r} not found"
        )
    return {"run_id": run_id, "prompt": prompt}


@app.get("/api/mode")
async def get_mode() -> dict:
    return {"mode": resource_manager.mode}


@app.post("/api/mode")
async def set_mode(request: ModeRequest) -> dict:
    try:
        await resource_manager.set_mode(request.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"mode": resource_manager.mode}


@app.get("/api/settings")
async def get_settings() -> dict:
    return settings.model_dump()


@app.post("/api/settings")
async def update_settings(update: SettingsUpdate) -> dict:
    data = update.model_dump(exclude_none=True)
    if "vram_threshold_mb" in data and "ram_threshold_mb" not in data:
        data["ram_threshold_mb"] = data["vram_threshold_mb"]

    # Guard KV cache changes
    kv_fields = {"idle_kv_quant", "active_kv_quant"}
    if kv_fields & set(data.keys()) and settings.require_approval_kv_cache_change:
        approval = approval_queue.submit(
            action="kv_cache_change",
            description="KV cache quantisation change requested",
            payload={k: v for k, v in data.items() if k in kv_fields},
        )
        audit_log.record(action="kv_cache_change_queued", actor="api", details=approval.id)
        return {"pending_approval": approval.to_dict()}

    # Guard large context increases
    new_main_ctx = data.get("main_context_size")
    if new_main_ctx and settings.require_approval_large_context_increase:
        if new_main_ctx > settings.main_context_size * 2:
            approval = approval_queue.submit(
                action="large_context_increase",
                description=f"main_context_size increase from {settings.main_context_size} to {new_main_ctx}",
                payload={"main_context_size": new_main_ctx},
            )
            audit_log.record(action="large_context_increase_queued", actor="api", details=approval.id)
            return {"pending_approval": approval.to_dict()}

    before = settings.model_dump()
    for key, value in data.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    audit_log.record(action="settings_update", actor="api", before=before, after=settings.model_dump())
    return settings.model_dump()


@app.post("/api/benchmark")
async def run_benchmark(request: BenchmarkRequest) -> dict:
    bench_type = request.type.lower()
    if bench_type == "speed":
        result = await benchmark_suite.run_inference_speed()
        return {"type": "speed", "results": result}
    if bench_type == "vram":
        result = await benchmark_suite.run_vram_benchmark()
        return {"type": "vram", "results": result}
    if bench_type == "slots":
        result = await benchmark_suite.run_slot_interference()
        return {"type": "slots", "results": result}
    result = await benchmark_suite.run_all()
    return {"type": "all", "results": result}


@app.get("/api/memory/search")
async def memory_search(q: str, limit: int = 5) -> dict:
    if memory is None:
        return {"results": [], "error": "Memory not available"}
    try:
        results = await memory.search(q, limit=limit)
        return {"results": results}
    except Exception as exc:
        logger.error("Memory search failed: %s", exc)
        return {"results": [], "error": "Memory search failed"}


# ---------------------------------------------------------------------------
# Agent Presets
# ---------------------------------------------------------------------------
@app.get("/api/presets")
async def list_presets() -> dict:
    return {"presets": preset_manager.list_presets()}


@app.post("/api/presets", status_code=201)
async def create_preset(body: PresetCreate) -> dict:
    if body.context_size > settings.max_context_size:
        raise HTTPException(
            status_code=400,
            detail=f"context_size {body.context_size} exceeds hard limit {settings.max_context_size}",
        )
    preset = preset_manager.create_preset(body.model_dump())
    audit_log.record(action="preset_created", actor="api", after=preset)
    return preset


@app.get("/api/presets/{preset_id}")
async def get_preset(preset_id: str) -> dict:
    try:
        return preset_manager.get_preset(preset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/presets/{preset_id}")
async def update_preset(preset_id: str, body: PresetUpdate) -> dict:
    data = body.model_dump(exclude_none=True)
    if "context_size" in data and data["context_size"] > settings.max_context_size:
        raise HTTPException(
            status_code=400,
            detail=f"context_size {data['context_size']} exceeds hard limit {settings.max_context_size}",
        )
    try:
        before = preset_manager.get_preset(preset_id)
        updated = preset_manager.update_preset(preset_id, data)
        audit_log.record(action="preset_updated", actor="api", before=before, after=updated)
        return updated
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/presets/{preset_id}", status_code=204)
async def delete_preset(preset_id: str) -> None:
    try:
        before = preset_manager.get_preset(preset_id)
        preset_manager.delete_preset(preset_id)
        audit_log.record(action="preset_deleted", actor="api", before=before)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/presets/{preset_id}/spawn", status_code=201)
async def spawn_agent(preset_id: str, body: SpawnRequest = SpawnRequest()) -> dict:
    if len(agent_registry.list_agents()) >= settings.max_slots:
        raise HTTPException(
            status_code=400,
            detail=f"Max running agents ({settings.max_slots}) reached.",
        )
    assigned_slot_id = await _choose_idle_parallel_slot_id()
    if assigned_slot_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No idle parallel slot is available for sub-agent spawn. "
                "Main agent stays on slot 0; launch with -np >= 2 and ensure a non-main slot is idle."
            ),
        )
    try:
        agent = agent_registry.spawn(
            preset_id,
            assigned_slot_id=assigned_slot_id,
            max_tokens_override=body.max_tokens,
            task_description=body.task_description,
        )
        audit_log.record(action="agent_spawned", actor="api", after=agent.info())
        return agent.info()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------
@app.get("/api/tools")
async def list_tools() -> dict:
    """List tools currently available to semantic routing."""
    tools = tool_registry.get_tools()
    return {
        "profile": "semantic_only",
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "parameters_schema": t.parameters_schema,
            }
            for t in tools
        ],
    }


@app.post("/api/tools/run")
async def run_tool(request: ToolRunRequest) -> dict:
    """Execute a single registered tool by name.

    This endpoint is intended for diagnostics, manual validation, and
    full-pipeline smoke tests. The tool execution path is the same runtime
    path used by agentic loop tool calls.
    """
    tool = tool_registry.get(request.name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {request.name!r}")

    try:
        result = await tool.execute(**request.args)
    except Exception as exc:
        logger.warning("/api/tools/run execution failed for %s: %s", request.name, exc)
        return {
            "tool": request.name,
            "ok": False,
            "output": None,
            "error": str(exc),
        }

    result_dict = result.to_dict()
    audit_log.record(
        action="tool_run_api",
        actor="api",
        details=f"tool={request.name} ok={bool(result_dict.get('ok'))}",
    )
    return {
        "tool": request.name,
        **result_dict,
    }


# ---------------------------------------------------------------------------
# Running Agents
# ---------------------------------------------------------------------------
@app.get("/api/agents")
async def list_agents() -> dict:
    return {"agents": agent_registry.list_agents()}


@app.delete("/api/agents/{agent_id}", status_code=204)
async def destroy_agent(agent_id: str) -> None:
    try:
        audit_log.record(action="agent_destroyed", actor="api", details=agent_id)
        agent_registry.destroy(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/agents/{agent_id}/chat")
async def chat_with_agent(agent_id: str, request: AgentChatRequest) -> dict:
    try:
        agent = agent_registry.get(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        response = await agent.process(
            request.message,
            max_tokens=request.max_tokens,
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        _raise_service_unavailable(exc, "Agent chat")
    return {"agent_id": agent_id, "response": response}


# ---------------------------------------------------------------------------
# Server Management
# ---------------------------------------------------------------------------
@app.get("/api/server/status")
async def get_server_status() -> dict:
    return {"status": server_manager.status}


@app.post("/api/server/restart")
async def restart_server() -> dict:
    if settings.require_approval_server_restart:
        approval = approval_queue.submit(
            action="server_restart",
            description="llama-server restart requested via API",
        )
        audit_log.record(action="server_restart_queued", actor="api", details=approval.id)
        return {"pending_approval": approval.to_dict()}
    try:
        await server_manager.restart(actor="api")
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except (ValueError, OSError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": server_manager.status}


# ---------------------------------------------------------------------------
# Approval Queue
# ---------------------------------------------------------------------------
@app.get("/api/approvals")
async def list_approvals() -> dict:
    return {"approvals": approval_queue.list_pending()}


@app.post("/api/approvals/{approval_id}/approve")
async def approve_action(approval_id: str) -> dict:
    try:
        approval = approval_queue.approve(approval_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    audit_log.record(action="approval_granted", actor="human", details=approval_id)

    # Execute the approved action
    result: dict[str, Any] = {"approved": True, "action": approval.action}
    if approval.action == "server_restart":
        try:
            await server_manager.restart(actor="human")
            result["server_status"] = server_manager.status
        except RuntimeError as exc:
            result["error"] = str(exc)
        except (ValueError, OSError, FileNotFoundError) as exc:
            result["error"] = str(exc)
    elif approval.action in ("kv_cache_change", "large_context_increase"):
        payload = approval.payload or {}
        for key, value in payload.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        result["settings"] = settings.model_dump()
    elif approval.action == "emergency_stop":
        agent_registry.destroy_all()
        await server_manager.stop()
        result["status"] = "stopped"
        result["agents_destroyed"] = True

    return result


@app.post("/api/approvals/{approval_id}/reject")
async def reject_action(approval_id: str) -> dict:
    try:
        approval = approval_queue.reject(approval_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    audit_log.record(action="approval_rejected", actor="human", details=approval_id)
    return {"rejected": True, "action": approval.action}


# ---------------------------------------------------------------------------
# Emergency Stop
# ---------------------------------------------------------------------------
@app.post("/api/emergency-stop")
async def emergency_stop() -> dict:
    if settings.require_approval_emergency_stop:
        approval = approval_queue.submit(
            action="emergency_stop",
            description="Emergency stop requested via API",
        )
        audit_log.record(action="emergency_stop_queued", actor="api", details=approval.id)
        return {"pending_approval": approval.to_dict()}
    audit_log.record(action="emergency_stop", actor="human")
    agent_registry.destroy_all()
    await server_manager.stop()
    return {"status": "stopped", "agents_destroyed": True}


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------
@app.get("/api/audit-log")
async def get_audit_log(n: int = 100) -> dict:
    return {"entries": audit_log.tail(n)}


# ---------------------------------------------------------------------------
# WebSocket – live status stream
# ---------------------------------------------------------------------------
@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket) -> None:
    configured_key = settings.api_key.strip()
    header_key = websocket.headers.get("x-api-key", "").strip()
    bearer_key = _parse_bearer_token(websocket.headers.get("authorization"))
    client_host = websocket.client.host if websocket.client else None

    if configured_key:
        if header_key != configured_key and bearer_key != configured_key:
            await websocket.close(code=1008)
            return
    elif not _is_loopback_host(client_host):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        while True:
            stats = await resource_manager.get_system_stats()
            try:
                slot_info = await slot_manager.get_slot_info()
            except Exception as exc:
                logger.warning("ws/status slot fetch failed: %s", exc.__class__.__name__)
                slot_info = {"main": settings.main_slot, "slots": []}
            payload = {
                **stats,
                "slots": slot_info,
                "server_status": server_manager.status,
                "pending_approvals": len(approval_queue.list_pending()),
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("ws/status disconnected due to error: %s", exc.__class__.__name__)


@app.websocket("/ws/agent-progress")
async def ws_agent_progress(websocket: WebSocket) -> None:
    """Stream real-time agentic-loop progress events to a WebSocket client.

    Protocol
    --------
    1. Client connects and sends a JSON object:  ``{"task": "...", "max_steps": 10}``
    2. Server sends a sequence of progress-event objects (each has a ``type`` field).
    3. After the final ``complete`` event the server sends a ``result`` object and
       closes the connection.
    """
    configured_key = settings.api_key.strip()
    header_key = websocket.headers.get("x-api-key", "").strip()
    bearer_key = _parse_bearer_token(websocket.headers.get("authorization"))
    client_host = websocket.client.host if websocket.client else None

    if configured_key:
        if header_key != configured_key and bearer_key != configured_key:
            await websocket.close(code=1008)
            return
    elif not _is_loopback_host(client_host):
        await websocket.close(code=1008)
        return

    await websocket.accept()

    connection_closed = False

    async def _send(data: dict) -> None:
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    async def _close(code: int = 1000) -> None:
        nonlocal connection_closed
        if connection_closed:
            return
        connection_closed = True
        try:
            await websocket.close(code=code)
        except Exception:
            pass

    try:
        raw = await websocket.receive_text()
        try:
            task_data = json.loads(raw)
        except json.JSONDecodeError:
            await _send({"type": "error", "message": "Invalid JSON"})
            await _close(1000)
            return

        task = (task_data.get("task") or "").strip()
        if not task:
            await _send({"type": "error", "message": "No task provided"})
            await _close(1000)
            return

        max_steps = _clamp_steps(int(task_data.get("max_steps", 10)))
        debug_request = AgenticDebugRequest.model_validate(task_data.get("debug", {}))

        from backend.agents.agentic_loop import AgenticLoop

        if _inference_lock.locked():
            await _send({"type": "error", "message": "Another inference request is already running. Please wait."})
            await _close(1000)
            return

        loop = AgenticLoop(
            main_agent=main_agent,
            max_steps=max_steps,
            checkpoint_manager=checkpoint_manager,
            supervisor=safety_supervisor,
            debug_options=_build_debug_options(debug_request),
        )

        audit_log.record(
            action="agentic_task_ws", actor="api", details=task[:100]
        )

        async with _inference_lock:
            try:
                result = await loop.run(task=task, on_progress=_send)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                await _send(
                    {
                        "type": "error",
                        "message": f"Service unavailable: {exc.__class__.__name__}",
                    }
                )
                await _close(1000)
                return

        await _send({"type": "result", **result})
        await _close(1000)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("ws/agent-progress error: %s", exc)
        await _send({"type": "error", "message": f"Internal error: {exc.__class__.__name__}"})
        await _close(1011)


# ---------------------------------------------------------------------------
# Vision & Desktop Control
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_desktop_controller():
    """Return a module-level DesktopController, lazily imported."""
    from backend.vision.controller import DesktopController
    return DesktopController()


@app.get("/api/vision/status")
async def vision_status() -> dict:
    """Report runtime vision capabilities on this machine."""
    try:
        dc = _get_desktop_controller()
        capabilities = dc.capabilities()
        llama_props: dict[str, Any] = {}
        llama_error = ""
        try:
            llama_props = await llama_client.get_props()
        except Exception as exc:
            llama_error = str(exc)

        modalities = llama_props.get("modalities") if isinstance(llama_props, dict) else {}
        llama_vision = bool(isinstance(modalities, dict) and modalities.get("vision"))
        capabilities["multimodal_available"] = llama_vision
        capabilities["mmproj_configured"] = llama_vision
        capabilities["llama_model_name"] = llama_props.get("model_name", "") if llama_props else ""
        capabilities["llama_model_path"] = llama_props.get("model_path", "") if llama_props else ""
        capabilities["llama_modalities"] = modalities if isinstance(modalities, dict) else {}
        capabilities["image_analysis_backend"] = getattr(settings, "image_analysis_backend", "auto")
        capabilities["vision_run_supported"] = bool(
            capabilities.get("ocr_available") or llama_vision
        )
        notes = [
            "OCR/grid desktop control is available when OCR is installed.",
            "llama.cpp multimodal image analysis is available when /props reports modalities.vision=true.",
        ]
        if not llama_vision:
            notes.append(
                "llama.cpp did not report active vision support"
                + (f": {llama_error}" if llama_error else ".")
            )
        capabilities["notes"] = notes
        return capabilities
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class VisionActionRequest(BaseModel):
    command: str   # e.g. "CLICK B3", "TYPE \"hello\"", "PRESS ctrl+c"


class VisionRunRequest(BaseModel):
    task: str
    max_steps: int = 20


async def _multimodal_screen_summary(screen: dict) -> str:
    """Return a concise llama.cpp vision summary for a captured screen."""
    image_b64 = str(screen.get("annotated_png_b64") or "")
    if not image_b64:
        return ""
    try:
        props = await llama_client.get_props()
    except Exception:
        return ""
    modalities = props.get("modalities") if isinstance(props, dict) else {}
    if not (isinstance(modalities, dict) and modalities.get("vision")):
        return ""
    try:
        return await llama_client.complete_with_image(
            prompt=(
                "This is an annotated desktop screenshot with grid labels. "
                "Describe the visible UI, active app/window, readable text, "
                "and any likely clickable controls. Keep it concise."
            ),
            image_b64=image_b64,
            max_tokens=512,
            temperature=0.2,
        )
    except Exception as exc:
        logger.debug("Multimodal screen summary failed: %s", exc.__class__.__name__)
        return ""


@app.get("/api/vision/screenshot")
async def vision_screenshot() -> dict:
    """Take a screenshot and return screen state as text + annotated PNG (base64)."""
    try:
        dc = _get_desktop_controller()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, dc.capture)
        audit_log.record(action="vision_screenshot", actor="api")
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/vision/action")
async def vision_action(request: VisionActionRequest) -> dict:
    """Execute a single desktop action command (e.g. CLICK A3, PRESS enter)."""
    try:
        dc = _get_desktop_controller()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, dc.execute_command_then_capture, request.command
        )
        audit_log.record(
            action="vision_action",
            actor="api",
            details=request.command,
        )
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/vision/run")
async def vision_run(request: VisionRunRequest) -> dict:
    """Run a multi-step vision-guided task.

    The MainAgent sees the current screen state, issues one action command per
    step, and we execute it. Continues until the model says DONE or max_steps
    is reached.

    Returns a list of step results.
    """
    try:
        dc = _get_desktop_controller()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    capabilities = dc.capabilities()
    if not capabilities.get("vision_run_supported"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Automated vision_run is not supported on this machine yet: "
                "no OCR backend is available, and true multimodal image-to-model support "
                "is not wired into this build. Manual screenshot/action endpoints remain available."
            ),
        )

    loop = asyncio.get_event_loop()
    audit_log.record(action="vision_run_start", actor="api", details=request.task)

    steps: list[dict] = []
    # Initial screenshot
    screen = await loop.run_in_executor(None, dc.capture)

    for step_num in range(request.max_steps):
        multimodal_summary = await _multimodal_screen_summary(screen)
        screen_context = screen["state_text"]
        if multimodal_summary.strip():
            screen_context += "\n\nVISION_MODEL_SUMMARY:\n" + multimodal_summary.strip()

        # Build the prompt for the MainAgent
        prompt = (
            f"You are performing a desktop task. Task: {request.task}\n\n"
            f"{screen_context}\n\n"
            "Issue exactly ONE action command from the allowed syntax, or DONE to finish.\n"
            "Action:"
        )

        try:
            model_response = await main_agent.process(prompt)
        except Exception as exc:
            logger.error("MainAgent failed during vision run: %s", exc.__class__.__name__)
            steps.append({"step": step_num + 1, "error": "Model error; stopping."})
            break

        # Extract the first non-empty line as the command
        command = next(
            (line.strip() for line in model_response.splitlines() if line.strip()),
            "DONE",
        )

        step_result = await loop.run_in_executor(
            None, dc.execute_command_then_capture, command
        )
        step_result["step"] = step_num + 1
        step_result["model_response"] = model_response
        steps.append(step_result)

        audit_log.record(
            action="vision_step",
            actor="main_agent",
            details=f"step={step_num + 1} cmd={command!r}",
        )

        if command.upper() == "DONE":
            break

        # Update screen for next iteration; re-capture if step had no screenshot
        if "screen" in step_result:
            screen = step_result["screen"]
        else:
            try:
                screen = await loop.run_in_executor(None, dc.capture)
            except Exception:
                pass  # keep stale screen rather than crashing

    audit_log.record(
        action="vision_run_end",
        actor="api",
        details=f"task={request.task!r} steps={len(steps)}",
    )
    return {"task": request.task, "steps": steps, "total_steps": len(steps)}

