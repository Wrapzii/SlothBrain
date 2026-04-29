from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agents.handoff import HandoffManager
from backend.agents.main_agent import MainAgent
from backend.agents.preset_manager import PresetManager
from backend.agents.registry import AgentRegistry
from backend.agents.watcher import WatcherAgent
from backend.benchmarks.benchmark import BenchmarkSuite
from backend.config import settings
from backend.core.approval_queue import ApprovalQueue
from backend.core.audit_log import AuditLog
from backend.core.llama_client import LlamaClient
from backend.core.resource_manager import ResourceManager
from backend.core.server_manager import ServerManager
from backend.core.slot_manager import SlotManager
from backend.memory.lancedb_memory import LanceDBMemory
from backend.memory.rolling_context import RollingContext


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons – populated during lifespan startup
# ---------------------------------------------------------------------------
llama_client: LlamaClient
slot_manager: SlotManager
resource_manager: ResourceManager
rolling_context: RollingContext
memory: LanceDBMemory
watcher_agent: WatcherAgent
main_agent: MainAgent
handoff_manager: HandoffManager
benchmark_suite: BenchmarkSuite
preset_manager: PresetManager
agent_registry: AgentRegistry
server_manager: ServerManager
audit_log: AuditLog
approval_queue: ApprovalQueue


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global llama_client, slot_manager, resource_manager, rolling_context
    global memory, watcher_agent, main_agent, handoff_manager, benchmark_suite
    global preset_manager, agent_registry, server_manager, audit_log, approval_queue

    audit_log = AuditLog()
    approval_queue = ApprovalQueue()
    server_manager = ServerManager(config=settings, audit_log=audit_log)

    llama_client = LlamaClient(host=settings.llama_host, port=settings.llama_port)
    slot_manager = SlotManager(llama_client=llama_client)
    resource_manager = ResourceManager(config=settings, llama_client=llama_client)

    await slot_manager.assign_watcher(settings.watcher_slot)
    await slot_manager.assign_main(settings.main_slot)

    rolling_context = RollingContext(
        llama_client=llama_client,
        slot_id=settings.watcher_slot,
        max_tokens=settings.watcher_context_size,
    )

    try:
        memory = LanceDBMemory(
            db_path=settings.lancedb_path,
            embedding_model=settings.embedding_model,
        )
    except ImportError as exc:
        print(f"[WARNING] LanceDB unavailable – memory disabled: {exc}")
        memory = None  # type: ignore[assignment]

    watcher_agent = WatcherAgent(
        slot_manager=slot_manager,
        rolling_context=rolling_context,
        memory=memory,  # type: ignore[arg-type]
        config=settings,
    )
    main_agent = MainAgent(
        slot_manager=slot_manager,
        memory=memory,  # type: ignore[arg-type]
        config=settings,
    )
    handoff_manager = HandoffManager(watcher=watcher_agent, main_agent=main_agent)
    benchmark_suite = BenchmarkSuite(llama_client=llama_client, config=settings)

    preset_manager = PresetManager()
    agent_registry = AgentRegistry(
        preset_manager=preset_manager,
        llama_client=llama_client,
        memory=memory,
    )
    # Give the MainAgent a reference to the registry so it can spawn sub-agents
    main_agent.set_registry(agent_registry)

    yield

    # Shutdown – stop all watchdog tasks
    server_manager.stop_watchdog()
    agent_registry.destroy_all()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="SlothBrain", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    agent: str = "auto"  # "auto" | "watcher" | "main"


class ModeRequest(BaseModel):
    mode: str


class SettingsUpdate(BaseModel):
    llama_host: str | None = None
    llama_port: int | None = None
    watcher_slot: int | None = None
    main_slot: int | None = None
    watcher_context_size: int | None = None
    main_context_size: int | None = None
    idle_kv_quant: str | None = None
    active_kv_quant: str | None = None
    vram_threshold_mb: int | None = None
    embedding_model: str | None = None
    llama_server_path: str | None = None
    llama_server_args: list[str] | None = None
    max_context_size: int | None = None
    max_slots: int | None = None
    max_restarts_per_hour: int | None = None
    require_approval_server_restart: bool | None = None
    require_approval_kv_cache_change: bool | None = None
    require_approval_large_context_increase: bool | None = None


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
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    context_size: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class AgentChatRequest(BaseModel):
    message: str
    context_size: int | None = None   # per-call override
    max_tokens: int | None = None     # per-call override


class SpawnRequest(BaseModel):
    context_size: int | None = None   # override preset default
    max_tokens: int | None = None     # override preset default
    task_description: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Existing endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
async def get_status() -> dict:
    stats = await resource_manager.get_system_stats()
    slot_info = await slot_manager.get_slot_info()
    return {**stats, "slots": slot_info}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    agent_choice = request.agent.lower()
    if agent_choice == "watcher":
        response = await watcher_agent.process(request.message)
        return {"agent": "watcher", "response": response, "handoff": False}
    if agent_choice == "main":
        response = await main_agent.process(request.message)
        return {"agent": "main", "response": response, "handoff": False}
    return await handoff_manager.route(request.message)


@app.get("/api/mode")
async def get_mode() -> dict:
    return {"mode": resource_manager.mode}


@app.post("/api/mode")
async def set_mode(request: ModeRequest) -> dict:
    await resource_manager.set_mode(request.mode)
    return {"mode": resource_manager.mode}


@app.get("/api/settings")
async def get_settings() -> dict:
    return settings.model_dump()


@app.post("/api/settings")
async def update_settings(update: SettingsUpdate) -> dict:
    data = update.model_dump(exclude_none=True)

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
    if body.context_size and body.context_size > settings.max_context_size:
        raise HTTPException(
            status_code=400,
            detail=f"context_size {body.context_size} exceeds hard limit {settings.max_context_size}",
        )
    try:
        agent = agent_registry.spawn(
            preset_id,
            context_size_override=body.context_size,
            max_tokens_override=body.max_tokens,
            task_description=body.task_description,
        )
        audit_log.record(action="agent_spawned", actor="api", after=agent.info())
        return agent.info()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
    response = await agent.process(
        request.message,
        context_size=request.context_size,
        max_tokens=request.max_tokens,
    )
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
    elif approval.action in ("kv_cache_change", "large_context_increase"):
        payload = approval.payload or {}
        for key, value in payload.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        result["settings"] = settings.model_dump()

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
    await websocket.accept()
    try:
        while True:
            stats = await resource_manager.get_system_stats()
            slot_info = await slot_manager.get_slot_info()
            payload = {
                **stats,
                "slots": slot_info,
                "server_status": server_manager.status,
                "pending_approvals": len(approval_queue.list_pending()),
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Vision & Desktop Control
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_desktop_controller():
    """Return a module-level DesktopController, lazily imported."""
    from backend.vision.controller import DesktopController
    return DesktopController()


class VisionActionRequest(BaseModel):
    command: str   # e.g. "CLICK B3", "TYPE \"hello\"", "PRESS ctrl+c"


class VisionRunRequest(BaseModel):
    task: str
    max_steps: int = 20


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

    loop = asyncio.get_event_loop()
    audit_log.record(action="vision_run_start", actor="api", details=request.task)

    steps: list[dict] = []
    # Initial screenshot
    screen = await loop.run_in_executor(None, dc.capture)

    for step_num in range(request.max_steps):
        # Build the prompt for the MainAgent
        prompt = (
            f"You are performing a desktop task. Task: {request.task}\n\n"
            f"{screen['state_text']}\n\n"
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
