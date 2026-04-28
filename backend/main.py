from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agents.handoff import HandoffManager
from backend.agents.main_agent import MainAgent
from backend.agents.watcher import WatcherAgent
from backend.benchmarks.benchmark import BenchmarkSuite
from backend.config import settings
from backend.core.llama_client import LlamaClient
from backend.core.resource_manager import ResourceManager
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


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global llama_client, slot_manager, resource_manager, rolling_context
    global memory, watcher_agent, main_agent, handoff_manager, benchmark_suite

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

    yield


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


class BenchmarkRequest(BaseModel):
    type: str = "all"  # "speed" | "vram" | "slots" | "all"


# ---------------------------------------------------------------------------
# Endpoints
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
    # auto – route through HandoffManager
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
    for key, value in data.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
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
# WebSocket – live status stream
# ---------------------------------------------------------------------------
@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            stats = await resource_manager.get_system_stats()
            slot_info = await slot_manager.get_slot_info()
            payload = {**stats, "slots": slot_info}
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
