from __future__ import annotations

import logging
import re
from typing import Optional

from backend.config import AppConfig
from backend.core.slot_manager import SlotManager
from backend.memory.lancedb_memory import LanceDBMemory
from backend.memory.rolling_context import RollingContext

_HANDOFF_PHRASES = frozenset(
    ["hand off", "handoff", "hand-off", "complex task", "main agent"]
)

# Maximum characters kept for watcher feedback strings.
_MAX_MONITOR_FEEDBACK_LEN = 400
_MAX_VERIFY_FEEDBACK_LEN = 600

SYSTEM_PROMPT = (
    "You are a lightweight always-on assistant. Monitor activity and decide when to "
    "hand off complex tasks to the main agent. Keep responses concise."
)

_MONITOR_SYSTEM_PROMPT = (
    "You are an AI task monitor. Your job is to assess whether an agent step was "
    "completed successfully and keep the agent on track. Be concise and decisive. "
    "Reply with exactly two lines:\n"
    "ACTION: <continue|retry|done|abort>\n"
    "FEEDBACK: <brief assessment or guidance>\n\n"
    "Use 'continue' when the step is done and there are more steps to execute. "
    "Use 'done' when the overall task is already complete. "
    "Use 'retry' when the step result is incorrect or incomplete. "
    "Use 'abort' only when the task is impossible or critically broken."
)

_VERIFY_SYSTEM_PROMPT = (
    "You are an AI task verifier. Determine whether the given task was completed "
    "successfully based on the work accomplished. Be concise. "
    "Reply with exactly two lines:\n"
    "COMPLETE: <yes|no>\n"
    "FEEDBACK: <brief verification summary>"
)

logger = logging.getLogger(__name__)


def _parse_monitor_response(response: str) -> dict:
    """Extract ACTION and FEEDBACK from the watcher's monitor response."""
    lower = response.lower()

    # Try explicit ACTION: label first
    action = "continue"
    action_match = re.search(r"action:\s*(\w+)", lower)
    if action_match:
        candidate = action_match.group(1).strip()
        if candidate in ("continue", "retry", "done", "abort"):
            action = candidate
    else:
        # Fall back to keyword scan
        for keyword in ("abort", "done", "retry", "continue"):
            if keyword in lower:
                action = keyword
                break

    feedback = response.strip()
    feedback_match = re.search(r"feedback:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if feedback_match:
        feedback = feedback_match.group(1).strip()

    return {"action": action, "feedback": feedback[:_MAX_MONITOR_FEEDBACK_LEN]}


def _parse_verify_response(response: str) -> dict:
    """Extract COMPLETE and FEEDBACK from the watcher's verify response."""
    lower = response.lower()

    complete = True  # optimistic default
    complete_match = re.search(r"complete:\s*(\w+)", lower)
    if complete_match:
        value = complete_match.group(1).strip()
        complete = value in ("yes", "true", "1")
    elif "not complete" in lower or "incomplete" in lower:
        complete = False

    feedback = response.strip()
    feedback_match = re.search(r"feedback:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if feedback_match:
        feedback = feedback_match.group(1).strip()

    return {"complete": complete, "feedback": feedback[:_MAX_VERIFY_FEEDBACK_LEN]}


class WatcherAgent:
    def __init__(
        self,
        slot_manager: SlotManager,
        rolling_context: RollingContext,
        memory: Optional[LanceDBMemory],
        config: AppConfig,
    ) -> None:
        self._slot_manager = slot_manager
        self._rolling_context = rolling_context
        self._memory = memory
        self._config = config
        self.slot_id = config.watcher_slot
        self.system_prompt = SYSTEM_PROMPT

    async def process(self, user_input: str) -> str:
        await self._rolling_context.add_message("user", user_input)
        context = self._rolling_context.get_context_prompt()
        full_prompt = f"system: {self.system_prompt}\n{context}assistant:"
        response = await self._slot_manager.send_to_watcher(
            full_prompt, max_tokens=256
        )
        await self._rolling_context.add_message("assistant", response)
        if self._memory is not None:
            try:
                await self._memory.store(
                    text=f"user: {user_input}\nassistant: {response}",
                    metadata={"agent": "watcher", "slot": self.slot_id},
                )
            except Exception as exc:
                logger.warning("WatcherAgent memory store failed: %s", exc.__class__.__name__)
        return response

    async def should_handoff(self, response: str) -> bool:
        lower = response.lower()
        return any(phrase in lower for phrase in _HANDOFF_PHRASES)

    # ------------------------------------------------------------------
    # Agentic-loop monitoring
    # ------------------------------------------------------------------

    async def monitor_step(
        self,
        task: str,
        step_description: str,
        step_result: str,
        step_num: int,
        total_steps: int,
        context: list[str] | None = None,
    ) -> dict:
        """Assess a completed step and advise the loop on what to do next.

        Returns a dict with keys:
        - ``action``: one of ``continue``, ``retry``, ``done``, ``abort``.
        - ``feedback``: brief assessment / guidance string.
        """
        context_str = ""
        if context:
            recent = context[-3:]
            context_str = "Recent context:\n" + "\n".join(recent) + "\n"

        prompt = (
            f"system: {_MONITOR_SYSTEM_PROMPT}\n\n"
            f"Overall task: {task}\n"
            f"Step {step_num}/{total_steps}: {step_description}\n"
            f"Step result:\n{step_result[:600]}\n"
            f"{context_str}"
            "assistant:"
        )
        try:
            response = await self._slot_manager.send_to_watcher(prompt, max_tokens=128)
        except Exception as exc:
            logger.warning("monitor_step request failed: %s", exc.__class__.__name__)
            return {"action": "continue", "feedback": ""}

        return _parse_monitor_response(response)

    async def verify_completion(
        self,
        task: str,
        steps_summary: list[str],
    ) -> dict:
        """Verify whether the overall task was completed successfully.

        Returns a dict with keys:
        - ``complete``: bool.
        - ``feedback``: verification summary string.
        """
        summary_str = "\n".join(steps_summary[-6:])
        prompt = (
            f"system: {_VERIFY_SYSTEM_PROMPT}\n\n"
            f"Task: {task}\n"
            f"What was accomplished:\n{summary_str}\n"
            "assistant:"
        )
        try:
            response = await self._slot_manager.send_to_watcher(prompt, max_tokens=150)
        except Exception as exc:
            logger.warning("verify_completion request failed: %s", exc.__class__.__name__)
            return {"complete": True, "feedback": "Verification unavailable."}

        return _parse_verify_response(response)
