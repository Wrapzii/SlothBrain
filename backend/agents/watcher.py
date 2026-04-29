from __future__ import annotations

import json
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
    "completed successfully and keep the agent on track.\n\n"
    "You MUST respond with a single valid JSON object and nothing else:\n"
    '{"action": "<action>", "feedback": "<brief assessment or guidance>"}\n\n'
    "Valid action values (choose exactly one):\n"
    "  continue – the step is done and there are more steps to execute\n"
    "  done     – the overall task is already complete\n"
    "  retry    – the step result is incorrect or incomplete\n"
    "  abort    – the task is impossible or critically broken (use sparingly)"
)

_VERIFY_SYSTEM_PROMPT = (
    "You are an AI task verifier. Determine whether the given task was completed "
    "successfully based on the work accomplished.\n\n"
    "You MUST respond with a single valid JSON object and nothing else:\n"
    '{"complete": true, "feedback": "<brief verification summary>"}\n\n'
    "Set complete to true if the task was fully accomplished, false otherwise."
)

logger = logging.getLogger(__name__)


def _parse_monitor_response(response: str) -> dict:
    """Extract action and feedback from the watcher's monitor response.

    Tries JSON parsing first (preferred — the prompt requests JSON output),
    then falls back to regex key extraction and finally a keyword scan.
    The safest default action is ``continue``.
    """
    stripped = response.strip()

    # ── Attempt 1: JSON parse ─────────────────────────────────────────────
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        json_candidate = fence_match.group(1)
    else:
        obj_match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
        json_candidate = obj_match.group(0) if obj_match else stripped

    try:
        data = json.loads(json_candidate)
        action = str(data.get("action", "continue")).strip().lower()
        if action not in ("continue", "retry", "done", "abort"):
            action = "continue"
        feedback = str(data.get("feedback", "")).strip()[:_MAX_MONITOR_FEEDBACK_LEN]
        return {"action": action, "feedback": feedback}
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # ── Attempt 2: Regex key extraction ──────────────────────────────────
    lower = response.lower()
    action = "continue"
    action_match = re.search(r'"?action"?\s*:\s*"?(\w+)"?', lower)
    if action_match:
        candidate = action_match.group(1).strip()
        if candidate in ("continue", "retry", "done", "abort"):
            action = candidate
    else:
        # ── Attempt 3: Keyword scan ───────────────────────────────────────
        for keyword in ("abort", "done", "retry", "continue"):
            if keyword in lower:
                action = keyword
                break

    feedback = stripped
    feedback_match = re.search(
        r'"?feedback"?\s*:\s*"?(.+)"?', response, re.IGNORECASE | re.DOTALL
    )
    if feedback_match:
        feedback = feedback_match.group(1).strip().rstrip('"').strip()

    return {"action": action, "feedback": feedback[:_MAX_MONITOR_FEEDBACK_LEN]}


def _parse_verify_response(response: str) -> dict:
    """Extract complete flag and feedback from the watcher's verify response.

    Tries JSON parsing first, then falls back to regex and keyword search.
    The optimistic default is ``complete=True`` (assume success when uncertain).
    """
    stripped = response.strip()

    # ── Attempt 1: JSON parse ─────────────────────────────────────────────
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        json_candidate = fence_match.group(1)
    else:
        obj_match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
        json_candidate = obj_match.group(0) if obj_match else stripped

    try:
        data = json.loads(json_candidate)
        raw_complete = data.get("complete", True)
        if isinstance(raw_complete, bool):
            complete = raw_complete
        else:
            complete = str(raw_complete).strip().lower() in ("true", "yes", "1")
        feedback = str(data.get("feedback", "")).strip()[:_MAX_VERIFY_FEEDBACK_LEN]
        return {"complete": complete, "feedback": feedback}
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # ── Attempt 2: Regex key extraction ──────────────────────────────────
    lower = response.lower()
    complete = True
    complete_match = re.search(r'"?complete"?\s*:\s*"?(\w+)"?', lower)
    if complete_match:
        value = complete_match.group(1).strip()
        complete = value in ("yes", "true", "1")
    elif "not complete" in lower or "incomplete" in lower:
        complete = False

    feedback = stripped
    feedback_match = re.search(
        r'"?feedback"?\s*:\s*"?(.+)"?', response, re.IGNORECASE | re.DOTALL
    )
    if feedback_match:
        feedback = feedback_match.group(1).strip().rstrip('"').strip()

    return {"complete": complete, "feedback": feedback[:_MAX_VERIFY_FEEDBACK_LEN]}


class WatcherAgent:
    """Lightweight always-on agent that monitors activity and manages handoffs.

    The WatcherAgent runs on a dedicated low-latency slot and serves two roles:

    1. **Conversational router** — handles simple user messages directly and
       detects complex tasks that should be handed off to the ``MainAgent``.
    2. **Agentic loop monitor** — called by ``AgenticLoop`` to assess each
       completed step and decide ``continue | retry | done | abort``, and to
       perform the final completion verification.

    All structured responses (monitor, verify) are requested in JSON format
    for reliable parsing.  The parsers fall back to regex if JSON is malformed.

    TODO: Replace ``should_handoff`` substring matching with a structured JSON
          field in the watcher's regular response to eliminate false positives
          when the watcher says "I do NOT need to hand off". See BUGS.md BUG-008.
    """

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
