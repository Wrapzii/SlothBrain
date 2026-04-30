"""API client for SlothBrain TUI — mirrors frontend/src/api/client.js."""

import httpx
import asyncio
from typing import Any, AsyncIterator, Optional


BASE = "http://127.0.0.1:8000"


async def get(path: str) -> Any:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE}{path}")
        r.raise_for_status()
        return r.json()


async def post(path: str, data: Optional[dict] = None) -> Any:
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(f"{BASE}{path}", json=data or {})
        r.raise_for_status()
        return r.json()


async def put(path: str, data: dict) -> Any:
    async with httpx.AsyncClient() as c:
        r = await c.put(f"{BASE}{path}", json=data)
        r.raise_for_status()
        return r.json()


async def delete(path: str) -> None:
    async with httpx.AsyncClient() as c:
        r = await c.delete(f"{BASE}{path}")
        r.raise_for_status()


# --- Status / Mode ---

async def get_status() -> dict:
    return await get("/api/status")


async def get_mode() -> dict:
    return await get("/api/mode")


async def set_mode(mode: str) -> dict:
    return await post("/api/mode", {"mode": mode})


# --- Server ---

async def get_server_status() -> dict:
    return await get("/api/server/status")


async def restart_server() -> dict:
    return await post("/api/server/restart")


async def emergency_stop() -> dict:
    return await post("/api/emergency-stop")


# --- Chat ---

async def send_chat(message: str, max_steps: int = 1, mode: str = "auto") -> dict:
    return await post("/api/chat", {"message": message, "max_steps": max_steps, "mode": mode})


async def send_direct_chat(message: str) -> dict:
    return await post("/api/chat/direct", {"message": message})

async def send_agentic_chat(task: str, max_steps: int = 10) -> dict:
    return await post("/api/chat/agentic", {"task": task, "max_steps": max_steps})


async def stream_agentic_chat(task: str, max_steps: int = 10, base: str = BASE) -> AsyncIterator[dict]:
    """Yield live agentic-loop events over the progress websocket."""
    import json
    import websockets  # type: ignore

    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/agent-progress"
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"task": task, "max_steps": max_steps}))
        async for msg in ws:
            yield json.loads(msg)


async def chat_with_agent(agent_id: str, message: str) -> dict:
    return await post(f"/api/agents/{agent_id}/chat", {"message": message})


# --- Settings ---

async def get_settings() -> dict:
    return await get("/api/settings")


async def update_settings(settings: dict) -> dict:
    return await post("/api/settings", settings)


# --- Presets ---

async def list_presets() -> list:
    result = await get("/api/presets")
    return result.get("presets", result) if isinstance(result, dict) else result


async def create_preset(data: dict) -> dict:
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{BASE}/api/presets", json=data)
        r.raise_for_status()
        return r.json()


async def update_preset(preset_id: str, data: dict) -> dict:
    return await put(f"/api/presets/{preset_id}", data)


async def delete_preset(preset_id: str) -> None:
    return await delete(f"/api/presets/{preset_id}")


async def spawn_agent(preset_id: str, overrides: Optional[dict] = None) -> dict:
    return await post(f"/api/presets/{preset_id}/spawn", overrides or {})


# --- Agents ---

async def list_agents() -> list:
    result = await get("/api/agents")
    return result.get("agents", result) if isinstance(result, dict) else result


async def destroy_agent(agent_id: str) -> None:
    return await delete(f"/api/agents/{agent_id}")


# --- Approvals ---

async def list_approvals() -> list:
    result = await get("/api/approvals")
    return result.get("approvals", result) if isinstance(result, dict) else result


async def approve_action(approval_id: str) -> dict:
    return await post(f"/api/approvals/{approval_id}/approve")


async def reject_action(approval_id: str) -> dict:
    return await post(f"/api/approvals/{approval_id}/reject")


# --- Benchmarks ---

async def run_benchmark(kind: str) -> dict:
    return await post("/api/benchmark", {"type": kind})


# --- Audit Log ---

async def get_audit_log(n: int = 50) -> list:
    result = await get(f"/api/audit-log?n={n}")
    return result.get("entries", result) if isinstance(result, dict) else result


# --- Memory ---

async def search_memory(query: str) -> list:
    return await get(f"/api/memory/search?q={query}")


# --- WebSocket status stream ---

async def status_stream(base: str = BASE) -> AsyncIterator[dict]:
    """Yield status dicts from the WebSocket live stream."""
    import websockets  # type: ignore

    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/status"
    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                async for msg in ws:
                    import json
                    yield json.loads(msg)
        except Exception:
            await asyncio.sleep(3)
