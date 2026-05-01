from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
from contextlib import asynccontextmanager
from functools import lru_cache
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
from backend.core.llama_client import LlamaClient
from backend.core.resource_manager import ResourceManager
from backend.core.safety_supervisor import SafetySupervisor
from backend.core.server_manager import ServerManager
from backend.core.slot_manager import SlotManager
from backend.core.semantic_router import SemanticRouter
from backend.memory.lancedb_memory import LanceDBMemory
from backend.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)

_AUTO_AGENTIC_TOOL_INTENT_RE = re.compile(
    r"(\bweb_fetch\b|\buse\b.{0,20}\btool\b|\btry\b.{0,20}\btool\b|\bfetch\b\s+https?://)",
    re.IGNORECASE,
)


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


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global llama_client, slot_manager, resource_manager
    global memory, main_agent, benchmark_suite
    global preset_manager, agent_registry, server_manager, audit_log, approval_queue
    global checkpoint_manager, safety_supervisor, tool_registry

    audit_log = AuditLog()
    approval_queue = ApprovalQueue(max_entries=settings.max_pending_approvals)
    server_manager = ServerManager(config=settings, audit_log=audit_log)

    llama_client = LlamaClient(host=settings.llama_host, port=settings.llama_port)
    slot_manager = SlotManager(llama_client=llama_client)
    slot_manager.set_slot_info_cache_ttl(settings.slot_info_cache_ttl_seconds)
    resource_manager = ResourceManager(config=settings, llama_client=llama_client)

    await slot_manager.assign_main(settings.main_slot)
    await _apply_effective_slot_context_budget()

    try:
        memory = LanceDBMemory(
            db_path=settings.lancedb_path,
            embedding_model=settings.embedding_model,
        )
    except ImportError as exc:
        print(f"[WARNING] LanceDB unavailable – memory disabled: {exc}")
        memory = None  # type: ignore[assignment]

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

    yield

    # Shutdown – stop all background tasks
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

    # Vision / desktop
    try:
        from backend.vision.controller import DesktopController
        controller = DesktopController()
        registry.register(UITool(controller=controller))
        registry.register(
            ImageAnalysisTool(
                llama_client=llama_client,
                controller=controller,
                backend=getattr(config, "image_analysis_backend", "cpu_ocr"),
                llama_slot_id=int(getattr(config, "image_analysis_llama_slot_id", 0)),
                cpu_max_text_chars=int(getattr(config, "image_analysis_cpu_max_text_chars", 4000)),
            )
        )
    except Exception as exc:
        logger.warning("Desktop tools unavailable: %s", exc)

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
            logger.warning("WorkspaceIndexTool not registered (missing deps): %s", exc)
            workspace_index_tool = WorkspaceIndexTool(indexer=None)
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

    # Scheduler
    sched = SchedulerTool()
    registry.register(sched)
    sched.start()

    # Discord (optional — skipped if no credentials configured)
    webhook = getattr(config, "discord_webhook_url", "")
    bot_token = getattr(config, "discord_bot_token", "")
    channel_id = getattr(config, "discord_channel_id", "")
    if webhook or bot_token:
        registry.register(
            DiscordTool(
                webhook_url=webhook,
                bot_token=bot_token,
                channel_id=channel_id,
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

# Only one inference task (direct or agentic) may run at a time.
# This avoids slot thrashing and recurrent-model slowdown when clients send
# overlapping requests (e.g. rapid consecutive messages).
_inference_lock: asyncio.Lock = asyncio.Lock()


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
    max_context_size: Optional[int] = None
    max_slots: Optional[int] = None
    max_restarts_per_hour: Optional[int] = None
    require_approval_server_restart: Optional[bool] = None
    require_approval_kv_cache_change: Optional[bool] = None
    require_approval_large_context_increase: Optional[bool] = None
    require_approval_emergency_stop: Optional[bool] = None
    semantic_tool_routing_enabled: Optional[bool] = None
    semantic_tool_routing_embedding_model: Optional[str] = None
    semantic_tool_routing_top_k: Optional[int] = None
    semantic_tool_routing_min_similarity: Optional[float] = None
    semantic_tool_routing_critical_tools: Optional[list[str]] = None
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
    context_size: Optional[int] = None   # override preset default
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


def _raise_service_unavailable(exc: Exception, context: str) -> None:
    logger.warning("%s failed: %s", context, exc.__class__.__name__)
    raise HTTPException(
        status_code=503,
        detail=f"{context} unavailable. Ensure llama.cpp server is running and reachable.",
    ) from exc


async def _ensure_llama_available(context: str) -> None:
    """Perform a quick preflight check so API callers get a clean 503 early."""
    try:
        await llama_client.health()
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
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
    if lower.startswith("/task ") or lower.startswith("/agentic "):
        return True

    # If caller requests multiple steps, treat as an agentic task.
    if max_steps > 1:
        return True

    # Tool-intent prompts should run through agentic mode so tool execution is real.
    if _AUTO_AGENTIC_TOOL_INTENT_RE.search(msg):
        return True

    # Otherwise default to direct chat for quick responsiveness.
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
        slot_info = {"main": settings.main_slot, "slots": []}
    return {**stats, "slots": slot_info}


@app.post("/api/chat")
async def chat(http_request: Request, request: ChatRequest) -> dict:
    """Chat endpoint with auto-routing between direct and agentic modes."""
    use_agentic = _should_use_agentic_mode(request.message, request.max_steps, request.mode)

    if not use_agentic:
        if _inference_lock.locked():
            raise HTTPException(
                status_code=503,
                detail="Another inference request is already running. Please wait.",
            )

        await _ensure_llama_available("Direct chat")
        async with _inference_lock:
            try:
                response = await main_agent.process_direct(user_input=request.message)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                _raise_service_unavailable(exc, "Direct chat")
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

    if _inference_lock.locked():
        raise HTTPException(
            status_code=503,
            detail="Another inference request is already running. Please wait.",
        )

    task_message = request.message.strip()
    for prefix in ("/task", "/agentic"):
        if task_message.lower().startswith(prefix):
            task_message = task_message[len(prefix):].strip()
            break

    async with _inference_lock:
        loop = AgenticLoop(
            main_agent=main_agent,
            max_steps=_clamp_steps(request.max_steps),
            checkpoint_manager=checkpoint_manager,
            supervisor=safety_supervisor,
            debug_options=_build_debug_options(request.debug),
        )
        try:
            result = await _run_agentic_with_cancel(
                http_request, loop.run(task=task_message)
            )
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            _raise_service_unavailable(exc, "Agentic loop")

    return {
        "agent": "agentic",
        "response": result.get("summary", ""),
        "handoff": False,
        "result": result,
    }


@app.post("/api/chat/direct")
async def direct_chat(request: ChatRequest) -> dict:
    """Explicit direct chat endpoint (single-shot, no task planning loop)."""
    if _inference_lock.locked():
        raise HTTPException(
            status_code=503,
            detail="Another inference request is already running. Please wait.",
        )

    await _ensure_llama_available("Direct chat")
    async with _inference_lock:
        try:
            response = await main_agent.process_direct(user_input=request.message)
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            _raise_service_unavailable(exc, "Direct chat")
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

    if _inference_lock.locked():
        raise HTTPException(
            status_code=503,
            detail="Another inference request is already running. Please wait.",
        )

    async with _inference_lock:
        loop = AgenticLoop(
            main_agent=main_agent,
            max_steps=_clamp_steps(request.max_steps),
            checkpoint_manager=checkpoint_manager,
            supervisor=safety_supervisor,
            debug_options=_build_debug_options(request.debug),
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
