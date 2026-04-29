from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from backend.core.checkpoint_manager import CheckpointManager
from backend.core.discord_bridge import DiscordBridge
from backend.core.llama_client import LlamaClient
from backend.core.resource_manager import ResourceManager
from backend.core.safety_supervisor import SafetySupervisor
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
checkpoint_manager: CheckpointManager
safety_supervisor: SafetySupervisor
discord_bridge: DiscordBridge | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global llama_client, slot_manager, resource_manager, rolling_context
    global memory, watcher_agent, main_agent, handoff_manager, benchmark_suite
    global preset_manager, agent_registry, server_manager, audit_log, approval_queue
    global checkpoint_manager, safety_supervisor, discord_bridge

    audit_log = AuditLog()
    approval_queue = ApprovalQueue(max_entries=settings.max_pending_approvals)
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

    # Safety infrastructure for the agentic loop
    checkpoint_manager = CheckpointManager()
    safety_supervisor = SafetySupervisor(
        llama_client=llama_client,
        checkpoint_manager=checkpoint_manager,
        poll_interval=settings.supervisor_poll_interval,
        step_timeout=settings.supervisor_step_timeout,
    )
    safety_supervisor.start()

    if settings.enable_server_watchdog:
        server_manager.start_watchdog()

    if settings.discord_bot_token and settings.discord_owner_user_id:
        discord_bridge = DiscordBridge(
            token=settings.discord_bot_token,
            owner_user_id=settings.discord_owner_user_id,
            approve_handler=_approve_internal,
            reject_handler=_reject_internal,
        )
        await discord_bridge.start()

    yield

    # Shutdown – stop all background tasks
    safety_supervisor.stop()
    server_manager.stop_watchdog()
    if discord_bridge is not None:
        await discord_bridge.stop()
    agent_registry.destroy_all()




async def _approve_internal(approval_id: str) -> dict[str, Any]:
    approval = approval_queue.approve(approval_id)
    audit_log.record(action="approval_granted", actor="human", details=approval_id)

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


async def _reject_internal(approval_id: str) -> dict[str, Any]:
    approval = approval_queue.reject(approval_id)
    audit_log.record(action="approval_rejected", actor="human", details=approval_id)
    return {"rejected": True, "action": approval.action}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="SlothBrain", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    ram_threshold_mb: int | None = None
    embedding_model: str | None = None
    llama_server_path: str | None = None
    llama_server_args: list[str] | None = None
    max_context_size: int | None = None
    max_slots: int | None = None
    max_restarts_per_hour: int | None = None
    require_approval_server_restart: bool | None = None
    require_approval_kv_cache_change: bool | None = None
    require_approval_large_context_increase: bool | None = None
    require_approval_emergency_stop: bool | None = None


class BenchmarkRequest(BaseModel):
    type: str = "all"  # "speed" | "vram" | "slots" | "all"

class DiscordPromptRequest(BaseModel):
    prompt: str
    timeout_seconds: float = 120.0


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
    max_tokens: int | None = None     # per-call override


class SpawnRequest(BaseModel):
    context_size: int | None = None   # override preset default
    max_tokens: int | None = None     # override preset default
    task_description: str = ""


class AgenticRequest(BaseModel):
    task: str
    max_steps: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Bounds for the number of agentic steps accepted via API.
_MIN_AGENTIC_STEPS = 1
_MAX_AGENTIC_STEPS = 20


def _clamp_steps(n: int) -> int:
    """Clamp a requested step count to the valid API range."""
    return min(max(n, _MIN_AGENTIC_STEPS), _MAX_AGENTIC_STEPS)


def _raise_service_unavailable(exc: Exception, context: str) -> None:
    logger.warning("%s failed: %s", context, exc.__class__.__name__)
    raise HTTPException(
        status_code=503,
        detail=f"{context} unavailable. Ensure llama.cpp server is running and reachable.",
    ) from exc


def _parse_bearer_token(authorization_header: str | None) -> str:
    if not authorization_header:
        return ""
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Existing endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
async def get_status() -> dict:
    stats = await resource_manager.get_system_stats()
    try:
        slot_info = await slot_manager.get_slot_info()
    except Exception as exc:
        logger.warning("Failed to fetch slot info: %s", exc.__class__.__name__)
        slot_info = {"watcher": settings.watcher_slot, "main": settings.main_slot, "slots": []}
    return {**stats, "slots": slot_info}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    agent_choice = request.agent.lower()
    try:
        if agent_choice == "watcher":
            response = await watcher_agent.process(request.message)
            return {"agent": "watcher", "response": response, "handoff": False}
        if agent_choice == "main":
            response = await main_agent.process(request.message)
            return {"agent": "main", "response": response, "handoff": False}
        return await handoff_manager.route(request.message)
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        _raise_service_unavailable(exc, "Chat service")




async def _capture_agentic_screenshot() -> dict:
    """Capture a desktop screenshot for agentic-loop progress snapshots."""
    dc = _get_desktop_controller()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, dc.capture)


@app.post("/api/chat/agentic")
async def agentic_chat(request: AgenticRequest) -> dict:
    """Run a multi-step agentic task loop and return the full result.

    The MainAgent plans the task, executes each step in sequence, and the
    WatcherAgent monitors progress and verifies completion.  The
    SafetySupervisor watches for stalls and the CheckpointManager saves state
    before every step so the loop can recover cleanly on failure.
    """
    from backend.agents.agentic_loop import AgenticLoop

    loop = AgenticLoop(
        main_agent=main_agent,
        watcher_agent=watcher_agent,
        max_steps=_clamp_steps(request.max_steps),
        checkpoint_manager=checkpoint_manager,
        supervisor=safety_supervisor,
        screenshot_fn=_capture_agentic_screenshot,
    )
    try:
        result = await loop.run(task=request.task)
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        _raise_service_unavailable(exc, "Agentic loop")
    audit_log.record(
        action="agentic_task",
        actor="api",
        details=request.task[:100],
    )
    return result


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
        if discord_bridge is not None:
            await discord_bridge.send_approval(approval)
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
            if discord_bridge is not None:
                await discord_bridge.send_approval(approval)
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
        if discord_bridge is not None:
            await discord_bridge.send_approval(approval)
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
        return await _approve_internal(approval_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/approvals/{approval_id}/reject")
async def reject_action(approval_id: str) -> dict:
    try:
        return await _reject_internal(approval_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
        if discord_bridge is not None:
            await discord_bridge.send_approval(approval)
        return {"pending_approval": approval.to_dict()}
    audit_log.record(action="emergency_stop", actor="human")
    agent_registry.destroy_all()
    await server_manager.stop()
    return {"status": "stopped", "agents_destroyed": True}



@app.post("/api/discord/prompt")
async def discord_prompt(body: DiscordPromptRequest) -> dict:
    if discord_bridge is None:
        raise HTTPException(status_code=503, detail="Discord bridge is not configured")
    try:
        reply = await discord_bridge.prompt_owner_for_text(body.prompt, timeout_seconds=body.timeout_seconds)
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail="Timed out waiting for Discord owner reply") from exc
    return {"reply": reply}


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
                slot_info = {"watcher": settings.watcher_slot, "main": settings.main_slot, "slots": []}
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

    async def _send(data: dict) -> None:
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    try:
        raw = await websocket.receive_text()
        try:
            task_data = json.loads(raw)
        except json.JSONDecodeError:
            await _send({"type": "error", "message": "Invalid JSON"})
            return

        task = (task_data.get("task") or "").strip()
        if not task:
            await _send({"type": "error", "message": "No task provided"})
            return

        max_steps = _clamp_steps(int(task_data.get("max_steps", 10)))

        from backend.agents.agentic_loop import AgenticLoop

        loop = AgenticLoop(
            main_agent=main_agent,
            watcher_agent=watcher_agent,
            max_steps=max_steps,
            checkpoint_manager=checkpoint_manager,
            supervisor=safety_supervisor,
            screenshot_fn=_capture_agentic_screenshot,
        )

        audit_log.record(
            action="agentic_task_ws", actor="api", details=task[:100]
        )

        try:
            result = await loop.run(task=task, on_progress=_send)
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            await _send(
                {
                    "type": "error",
                    "message": f"Service unavailable: {exc.__class__.__name__}",
                }
            )
            return

        await _send({"type": "result", **result})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("ws/agent-progress error: %s", exc)
        await _send({"type": "error", "message": f"Internal error: {exc.__class__.__name__}"})


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
        capabilities["mmproj_configured"] = False
        capabilities["notes"] = [
            "Automated vision_run currently requires OCR-readable screen text.",
            "True multimodal image-to-model support is not wired into this codebase yet.",
        ]
        return capabilities
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
