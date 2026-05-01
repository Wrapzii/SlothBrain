from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.agents.agentic_loop import AgenticDebugOptions, AgenticLoop
from backend.agents.main_agent import MainAgent
from backend.config import AppConfig
from backend.tools.base import Tool, ToolResult
from backend.tools.registry import ToolRegistry


class ScriptedSlotManager:
    """Deterministic slot manager for full-pipeline capability contract tests."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def send_to_main(self, prompt: str, max_tokens: int = 512) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            return "No scripted response left."
        return self._responses.pop(0)


class MockTool(Tool):
    name = ""
    description = "Mock capability tool"
    parameters_schema = {
        "type": "object",
        "properties": {"payload": {"type": "string"}},
        "required": ["payload"],
    }

    def __init__(self, tool_name: str, calls: list[dict]) -> None:
        self.name = tool_name
        self.description = f"Mock tool for {tool_name}"
        self._calls = calls

    async def execute(self, **kwargs) -> ToolResult:
        self._calls.append({"tool": self.name, "args": dict(kwargs)})
        return ToolResult(ok=True, output={"tool": self.name, "args": dict(kwargs)})


@dataclass(frozen=True)
class CapabilityScenario:
    name: str
    task: str
    required_tools: tuple[str, ...]


CAPABILITY_SCENARIOS: tuple[CapabilityScenario, ...] = (
    CapabilityScenario(
        name="resume_user_task",
        task="Resume the user's previous coding task after they return.",
        required_tools=("memory_search", "session_graph", "file"),
    ),
    CapabilityScenario(
        name="understand_screen_state",
        task="Understand what is happening on the computer screen.",
        required_tools=("screenshot", "image_analysis"),
    ),
    CapabilityScenario(
        name="open_application",
        task="Open VS Code and focus the editor window.",
        required_tools=("ui",),
    ),
    CapabilityScenario(
        name="search_files",
        task="Search through project files for configuration references.",
        required_tools=("file", "workspace_index"),
    ),
    CapabilityScenario(
        name="start_project_and_code",
        task="Create a new project scaffold and write initial code.",
        required_tools=("file", "patch", "shell"),
    ),
    CapabilityScenario(
        name="edit_and_find_files",
        task="Find project files and apply code edits.",
        required_tools=("file", "diff", "patch"),
    ),
    CapabilityScenario(
        name="browse_and_research",
        task="Browse the web and conduct research for implementation choices.",
        required_tools=("web_fetch", "web_search"),
    ),
    CapabilityScenario(
        name="assistant_configuration",
        task="Configure assistant settings and session preferences.",
        required_tools=("session", "file"),
    ),
    CapabilityScenario(
        name="compile_and_run",
        task="Compile a project, run it, and inspect runtime failures.",
        required_tools=("shell", "process"),
    ),
    CapabilityScenario(
        name="documents_and_folders",
        task="Create documents and organize folder structures.",
        required_tools=("file",),
    ),
    CapabilityScenario(
        name="windows_ai_pipeline",
        task="Use the Windows AI interaction pipeline to operate the computer.",
        required_tools=("screenshot", "ui", "image_analysis"),
    ),
    CapabilityScenario(
        name="ide_ui_project_loop",
        task="Open VS Code, build a project, run it, inspect errors, and click UI controls.",
        required_tools=("ui", "file", "patch", "shell", "process", "screenshot"),
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", CAPABILITY_SCENARIOS, ids=[s.name for s in CAPABILITY_SCENARIOS])
async def test_capability_pipeline_contracts(scenario: CapabilityScenario) -> None:
    tool_calls: list[dict] = []

    blocks: list[str] = []
    for tool in scenario.required_tools:
        blocks.extend(
            [
                "<tool_call>",
                f'{{"tool": "{tool}", "args": {{"payload": "{scenario.name}:{tool}"}}}}',
                "</tool_call>",
            ]
        )
    tool_call_blocks = "\n".join(blocks)

    responses = [
        tool_call_blocks,
        f"Capability contract complete for {scenario.name}.",
    ]
    slot_manager = ScriptedSlotManager(responses=responses)

    config = AppConfig()
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=config)

    registry = ToolRegistry()
    for tool_name in sorted({*scenario.required_tools, "file", "shell", "web_fetch", "ui"}):
        registry.register(MockTool(tool_name, tool_calls))
    agent.set_tool_registry(registry)

    loop = AgenticLoop(
        main_agent=agent,
        max_steps=1,
        debug_options=AgenticDebugOptions(
            enabled=True,
            planning_enabled=False,
            rolling_context_enabled=True,
            tool_calls_enabled=True,
            semantic_routing_enabled=False,
            checkpointing_enabled=False,
            supervisor_enabled=False,
            per_event_logging=True,
            allowed_tools=list(scenario.required_tools),
        ),
    )

    events: list[dict] = []

    async def on_progress(event: dict) -> None:
        events.append(event)

    result = await loop.run(task=scenario.task, on_progress=on_progress)

    assert result["completion_verified"] is True
    assert result["total_steps"] == 1
    assert scenario.name in result["summary"].lower() or "complete" in result["summary"].lower()

    called_tools = [entry["tool"] for entry in tool_calls]
    for required_tool in scenario.required_tools:
        assert required_tool in called_tools

    event_types = [str(e.get("type")) for e in events]
    assert "planning_skipped" in event_types
    assert "loop_iteration" in event_types
    assert "step_input" in event_types
    assert "step_output" in event_types
    assert "step_complete" in event_types


@pytest.mark.asyncio
async def test_llm_only_debug_profile_disables_all_optional_capabilities() -> None:
    slot_manager = ScriptedSlotManager(responses=["Pure LLM response only."])
    agent = MainAgent(slot_manager=slot_manager, memory=None, config=AppConfig())

    loop = AgenticLoop(
        main_agent=agent,
        max_steps=1,
        debug_options=AgenticDebugOptions(
            enabled=True,
            llm_only=True,
            planning_enabled=False,
            rolling_context_enabled=False,
            tool_calls_enabled=False,
            semantic_routing_enabled=False,
            checkpointing_enabled=False,
            supervisor_enabled=False,
            per_event_logging=True,
            allowed_tools=[],
        ),
    )

    result = await loop.run(task="Answer directly with no tools.")

    assert result["completion_verified"] is True
    assert result["total_steps"] == 1
