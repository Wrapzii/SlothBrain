from __future__ import annotations

import json
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
    from backend.tools.registry import ToolRegistry

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

# Maximum tool-call iterations per execute_step call to prevent infinite loops.
_MAX_TOOL_ITERATIONS = 5


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
    """Parse a plan_task response into approach and steps.

    Tries JSON parsing first (preferred — the prompt requests JSON output).
    Falls back to extracting a numbered list, then a ``STEPS:`` header, and
    finally treats the whole response as a single step.
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
        approach = str(data.get("approach", "")).strip()
        raw_steps = data.get("steps", [])
        if isinstance(raw_steps, list) and raw_steps:
            steps = [str(s).strip() for s in raw_steps if str(s).strip()]
            if steps:
                return {"approach": approach, "steps": steps[:10]}
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # ── Attempt 2: Regex for numbered list items ──────────────────────────
    approach = ""
    approach_match = re.search(r"approach:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if approach_match:
        approach = approach_match.group(1).strip()

    steps = [
        s.strip()
        for s in re.findall(r"^\d+\.\s+(.+)", response, re.MULTILINE)
        if s.strip()
    ]

    if not steps:
        # ── Attempt 3: Lines after STEPS: header ─────────────────────────
        in_steps = False
        for line in response.splitlines():
            stripped_line = line.strip()
            if re.match(r"steps?\s*:", stripped_line, re.IGNORECASE):
                in_steps = True
                continue
            if in_steps and stripped_line:
                steps.append(stripped_line.lstrip("-•*").strip())

    if not steps:
        # ── Attempt 4: Last resort ────────────────────────────────────────
        steps = [response.strip()[:_MAX_FALLBACK_STEP_LENGTH]]

    return {"approach": approach, "steps": steps[:10]}


class MainAgent:
    """High-capability agent responsible for planning and executing complex tasks.

    The MainAgent runs on the main inference slot (higher context window) and
    provides three core capabilities:

    1. **Task planning** — ``plan_task`` breaks a natural-language task into an
       ordered list of actionable steps (JSON output for reliable parsing).
    2. **Step execution** — ``execute_step`` executes one step with accumulated
       context from previous steps, maintaining coherent long-running task state.
    3. **Sub-agent delegation** — ``spawn_sub_agent`` creates task-specialised
       ``SubAgent`` instances via the ``AgentRegistry`` for parallel or
       specialised work.

    Memory retrieval is performed before every ``process`` call so relevant
    past context is always available to the model.

    TODO: Add a tool-calling layer so the MainAgent can invoke tools (code
          execution, file I/O, web search) as part of execute_step.
    """

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
        self._tool_registry: "ToolRegistry | None" = None
        self._tool_profile: str = getattr(config, "main_tool_profile", "full")

    def set_registry(self, registry: "AgentRegistry") -> None:
        """Inject the AgentRegistry so MainAgent can spawn sub-agents."""
        self._registry = registry

    def set_tool_registry(self, tool_registry: "ToolRegistry", profile: str | None = None) -> None:
        """Inject the ToolRegistry and optional tool profile override."""
        self._tool_registry = tool_registry
        if profile is not None:
            self._tool_profile = profile

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

        Requests JSON output for reliable parsing.  Falls back to regex
        extraction and then single-step execution on any failure.

        Returns a dict with keys:
        - ``approach``: brief description of the overall strategy.
        - ``steps``: list of step description strings (max 10).
        """
        plan_prompt = (
            "system: You are a planning AI. Break the following task into clear, "
            "actionable steps that an AI agent can execute one at a time.\n\n"
            "You MUST respond with a single valid JSON object and nothing else:\n"
            '{"approach": "<brief strategy description>", '
            '"steps": ["<step 1>", "<step 2>", ...]}\n\n'
            "Rules:\n"
            "- Maximum 10 steps.\n"
            "- Each step should be a single, concrete action.\n"
            "- Steps must be ordered and build on each other.\n\n"
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

        If a ToolRegistry is attached, the model may issue ``<tool_call>``
        blocks in its response.  Each block is parsed, the tool executed, and
        the result injected back into context before the model is called again.
        This loop repeats until no tool calls remain or the iteration limit is
        reached.

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

        # Build tool descriptions block if tools are available
        tools_section = ""
        tools: list = []
        if self._tool_registry is not None:
            tools = self._tool_registry.get_tools_for_profile(self._tool_profile)
            if tools:
                tools_block = self._tool_registry.render_tool_descriptions(tools)
                tools_section = (
                    f"\n\n{tools_block}\n\n"
                    "To use a tool, emit a <tool_call> block with JSON:\n"
                    "<tool_call>\n"
                    '{"tool": "<name>", "args": {<arguments>}}\n'
                    "</tool_call>\n"
                    "The tool result will be provided and you may continue.\n"
                    "When no more tools are needed, provide your final answer."
                )

        step_prompt = (
            f"system: {self.system_prompt}\n\n"
            "You are executing a multi-step task one step at a time.\n"
            f"Overall task: {task}\n"
            f"Current step: {step}"
            f"{context_section}"
            f"{tools_section}\n\n"
            "Execute this step thoroughly and report what you did and what you found.\n"
            "assistant:"
        )

        # Tool-calling loop
        accumulated_tool_context = ""
        for iteration in range(_MAX_TOOL_ITERATIONS):
            prompt = step_prompt + accumulated_tool_context
            response = await self._slot_manager.send_to_main(prompt, max_tokens=2048)

            # No tool registry or no tools → return directly
            if self._tool_registry is None or not tools:
                return response

            tool_calls = self._tool_registry.parse_tool_calls(response)
            if not tool_calls:
                # No tool calls — final answer
                return response

            # Execute each tool call and accumulate results
            tool_result_lines: list[str] = [response]
            for tc in tool_calls:
                tool_name = tc["tool"]
                tool_args = tc.get("args", {})
                tool = self._tool_registry.get(tool_name)
                if tool is None:
                    result_dict = {"ok": False, "error": f"Unknown tool: {tool_name!r}"}
                else:
                    try:
                        tool_result = await tool.execute(**tool_args)
                        result_dict = tool_result.to_dict()
                    except Exception as exc:
                        logger.warning("Tool %s raised: %s", tool_name, exc)
                        result_dict = {"ok": False, "error": str(exc)}

                result_json = json.dumps(
                    {"tool": tool_name, **result_dict},
                    ensure_ascii=False,
                    default=str,
                )
                tool_result_lines.append(
                    f"<tool_result>\n{result_json}\n</tool_result>"
                )

            accumulated_tool_context += "\n" + "\n".join(tool_result_lines) + "\nassistant:"

        # Iteration limit reached — return last response
        return response
