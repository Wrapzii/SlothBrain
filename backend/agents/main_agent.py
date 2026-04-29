from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from backend.config import AppConfig
from backend.core.slot_manager import SlotManager
from backend.memory.lancedb_memory import LanceDBMemory

if TYPE_CHECKING:
    from backend.agents.registry import AgentRegistry
    from backend.agents.sub_agent import SubAgent

logger = logging.getLogger(__name__)

_PROTECTED_PROMPT_PATH = (
    Path(__file__).parent.parent / "config" / "protected" / "main_system_prompt.txt"
)

_FALLBACK_SYSTEM_PROMPT = (
    "You are a high-performance AI assistant specializing in complex tasks and coding. "
    "Use the provided context and memory to give comprehensive answers."
)

# Maximum characters to keep when falling back to a single-step plan.
_MAX_FALLBACK_STEP_LENGTH = 300


def _load_protected_prompt() -> str:
    """Load the main agent's system prompt from the protected file (read-only)."""
    try:
        return _PROTECTED_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning(
            "Could not read protected system prompt; using fallback."
        )
        return _FALLBACK_SYSTEM_PROMPT


def _parse_plan(response: str) -> dict:
    """Parse APPROACH and numbered STEPS from a plan_task response."""
    approach = ""
    approach_match = re.search(r"approach:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if approach_match:
        approach = approach_match.group(1).strip()

    # Extract numbered list items: "1. ...", "2. ...", etc.
    steps = [s.strip() for s in re.findall(r"^\d+\.\s+(.+)", response, re.MULTILINE) if s.strip()]

    if not steps:
        # Fallback: lines after "STEPS:" header
        in_steps = False
        for line in response.splitlines():
            stripped = line.strip()
            if re.match(r"steps?\s*:", stripped, re.IGNORECASE):
                in_steps = True
                continue
            if in_steps and stripped:
                steps.append(stripped.lstrip("-•*").strip())

    if not steps:
        # Last resort: treat the whole response as one step
        steps = [response.strip()[:_MAX_FALLBACK_STEP_LENGTH]]

    return {"approach": approach, "steps": steps[:10]}


class MainAgent:
    def __init__(
        self,
        slot_manager: SlotManager,
        memory: Optional[LanceDBMemory],
        config: AppConfig,
    ) -> None:
        self._slot_manager = slot_manager
        self._memory = memory
        self._config = config
        self.slot_id = config.main_slot
        self.system_prompt = _load_protected_prompt()
        # Injected after construction so we avoid circular imports
        self._registry: AgentRegistry | None = None

    def set_registry(self, registry: "AgentRegistry") -> None:
        """Inject the AgentRegistry so MainAgent can spawn sub-agents."""
        self._registry = registry

    # ------------------------------------------------------------------
    # Sub-agent delegation
    # ------------------------------------------------------------------

    def spawn_sub_agent(
        self,
        preset_id: str,
        task_description: str,
        context_size: int | None = None,
        max_tokens: int | None = None,
    ) -> "SubAgent":
        """Spawn a sub-agent with task-appropriate resource budgets.

        ``context_size`` and ``max_tokens`` override the preset defaults so
        the MainAgent can right-size the allocation for the actual workload.
        If not provided, preset defaults are used.

        Raises RuntimeError if the registry is not set or max_slots exceeded.
        """
        if self._registry is None:
            raise RuntimeError("AgentRegistry not injected into MainAgent")
        return self._registry.spawn(
            preset_id=preset_id,
            context_size_override=context_size,
            max_tokens_override=max_tokens,
            task_description=task_description,
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def process(
        self,
        user_input: str,
        context_from_watcher: str = "",
    ) -> str:
        memory_results: list[dict] = []
        if self._memory is not None:
            try:
                memory_results = await self._memory.search(user_input, limit=5)
            except Exception as exc:
                logger.warning("MainAgent memory search failed: %s", exc.__class__.__name__)

        memory_context = ""
        if memory_results:
            snippets = "\n".join(f"- {r['text']}" for r in memory_results)
            memory_context = f"\n\nRelevant past context:\n{snippets}"

        watcher_section = ""
        if context_from_watcher:
            watcher_section = f"\n\nWatcher initial assessment:\n{context_from_watcher}"

        full_prompt = (
            f"system: {self.system_prompt}"
            f"{memory_context}"
            f"{watcher_section}\n\n"
            f"user: {user_input}\nassistant:"
        )

        response = await self._slot_manager.send_to_main(
            full_prompt, max_tokens=2048
        )

        if self._memory is not None:
            try:
                await self._memory.store(
                    text=f"user: {user_input}\nassistant: {response}",
                    metadata={"agent": "main", "slot": self.slot_id},
                )
            except Exception as exc:
                logger.warning("MainAgent memory store failed: %s", exc.__class__.__name__)

        return response

    # ------------------------------------------------------------------
    # Agentic-loop helpers
    # ------------------------------------------------------------------

    async def plan_task(self, task: str) -> dict:
        """Break a task into an ordered list of actionable steps.

        Returns a dict with keys:
        - ``approach``: brief description of the overall strategy.
        - ``steps``: list of step description strings (max 10).
        """
        plan_prompt = (
            "system: You are a planning AI. Break the following task into clear, "
            "actionable steps that an AI agent can execute one at a time.\n"
            "The plan must be genuinely agentic and end-to-end, not a single-pass answer.\n"
            "Include concrete execution + validation flow: discovery, implementation, testing, "
            "verification, and final reporting.\n"
            "Prefer 5-10 steps when appropriate.\n"
            "Format your response as:\n"
            "APPROACH: <brief description of the overall strategy>\n"
            "STEPS:\n"
            "1. <first step>\n"
            "2. <second step>\n"
            "...\n\n"
            f"Task: {task}\nassistant:"
        )
        try:
            response = await self._slot_manager.send_to_main(
                plan_prompt, max_tokens=1024
            )
        except Exception as exc:
            logger.warning("plan_task failed: %s", exc.__class__.__name__)
            return {"steps": [task], "approach": "Direct execution"}

        return _parse_plan(response)

    async def execute_step(
        self,
        step: str,
        task: str,
        context: list[str] | None = None,
    ) -> str:
        """Execute a single step within a larger task.

        Parameters
        ----------
        step:
            The current step description.
        task:
            The overarching task so the agent maintains focus.
        context:
            Accumulated results from previous steps (most recent last).
        """
        context_section = ""
        if context:
            recent = context[-5:]
            context_section = "\n\nContext from previous steps:\n" + "\n".join(recent)

        step_prompt = (
            f"system: {self.system_prompt}\n\n"
            "You are executing a multi-step task one step at a time.\n"
            f"Overall task: {task}\n"
            f"Current step: {step}"
            f"{context_section}\n\n"
            "Execute this step thoroughly. Perform real follow-through work, then report:\n"
            "1) actions taken, 2) outputs/evidence, 3) what changed, 4) remaining risk.\n"
            "If the step requires tools/checks, run them now rather than deferring.\n"
            "assistant:"
        )

        return await self._slot_manager.send_to_main(step_prompt, max_tokens=2048)
