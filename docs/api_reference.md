# SlothBrain API Reference

Base URL: `http://localhost:8000`

All `/api/*` endpoints require either:
- A configured `api_key` sent as `X-Api-Key: <key>` header or `Authorization: Bearer <key>`.
- If `api_key` is empty (default), requests must originate from loopback (127.0.0.1 / ::1).

---

## System

### `GET /health`
Liveness check. No auth required.

**Response**
```json
{"status": "ok"}
```

---

### `GET /api/status`
System stats and slot info.

**Response**
```json
{
  "cpu_percent": 12.5,
  "ram_used_mb": 4096.0,
  "ram_total_mb": 16384.0,
  "mode": "idle",
  "slots": {
    "main": 1,
    "slots": [{"id": 0, ...}, {"id": 1, ...}]
  }
}
```

---

### `WS /ws/status`
WebSocket stream of system stats, broadcast every 2 seconds.

---

## Chat

### `POST /api/chat`
Single-turn conversation.

**Request**
```json
{
  "message": "What is the capital of France?",
  "mode": "auto"
}
```
`mode`: `"auto"` | `"direct"` | `"agentic"` (default: `"auto"`)

**Response**
```json
{
  "agent": "direct",
  "response": "The capital of France is Paris.",
  "handoff": false
}
```

---

### `POST /api/chat/agentic`
Run a fully autonomous multi-step task.

**Request**
```json
{
  "task": "Research Python async patterns and write a summary document",
  "max_steps": 5
}
```
`max_steps`: 1–20 (default: 10)

**Response**
```json
{
  "task": "...",
  "steps": [
    {
      "step_num": 1,
      "description": "...",
      "result": "...",
      "status": "complete",
      "screenshots": [],
      "retries": 0,
      "duration_seconds": 4.21
    }
  ],
  "completion_verified": true,
  "summary": "Task completed successfully.",
  "total_steps": 3,
  "duration_seconds": 14.5
}
```

---

## Mode

### `GET /api/mode`
Get current operating mode.

**Response**
```json
{"mode": "idle"}
```

---

### `POST /api/mode`
Set operating mode.

**Request**
```json
{"mode": "active"}
```

---

## Settings

### `GET /api/settings`
Get all configuration values.

### `POST /api/settings`
Update configuration. Critical fields (KV cache, large context increases) may return a `pending_approval` response instead of applying immediately.

**Writable fields:**
`llama_host`, `llama_port`, `main_slot`,
`main_context_size`, `idle_kv_quant`, `active_kv_quant`, `vram_threshold_mb`,
`ram_threshold_mb`, `embedding_model`, `llama_server_path`, `llama_server_args`,
`max_context_size`, `max_slots`, `max_restarts_per_hour`,
`require_approval_server_restart`, `require_approval_kv_cache_change`,
`require_approval_large_context_increase`, `require_approval_emergency_stop`

---

## Memory

### `GET /api/memory/search?q=<query>&limit=5`
Semantic search over long-term memory.

**Response**
```json
{
  "results": [
    {
      "text": "user: ...\nassistant: ...",
      "metadata": {"agent": "main", "slot": 1},
      "timestamp": "2025-01-01T00:00:00+00:00"
    }
  ]
}
```

---

## Agent Presets

### `GET /api/presets`
List all agent presets.

### `POST /api/presets`
Create a preset.

**Request**
```json
{
  "name": "Code Reviewer",
  "description": "Reviews code for bugs and style",
  "system_prompt": "You are an expert code reviewer...",
  "context_size": 8192,
  "temperature": 0.3,
  "max_tokens": 2048
}
```

### `GET /api/presets/{preset_id}`
Get a specific preset.

### `PUT /api/presets/{preset_id}`
Update a preset (partial update).

### `DELETE /api/presets/{preset_id}`
Delete a preset (204 No Content).

### `POST /api/presets/{preset_id}/spawn`
Spawn a sub-agent from a preset.

**Request** (all fields optional — override preset defaults)
```json
{
  "context_size": 16384,
  "max_tokens": 4096,
  "task_description": "Review the authentication module"
}
```

**Response**
```json
{"agent_id": "uuid", "name": "Code Reviewer", ...}
```

---

## Running Sub-Agents

### `GET /api/agents`
List all running sub-agent instances.

### `GET /api/agents/{agent_id}`
Get details of a specific agent.

### `POST /api/agents/{agent_id}/chat`
Send a message to a specific sub-agent.

**Request**
```json
{
  "message": "Review this function: def add(a, b): return a + b",
  "max_tokens": 1024
}
```

### `DELETE /api/agents/{agent_id}`
Destroy a running sub-agent (204 No Content).

---

## Server Management

### `GET /api/server/status`
Get llama-server process status.

### `POST /api/server/start`
Start the llama-server (requires `llama_server_path` to be configured).

### `POST /api/server/stop`
Stop the llama-server.

### `POST /api/server/restart`
Restart the llama-server. If `require_approval_server_restart` is true, returns a `pending_approval` response.

---

## Approvals

### `GET /api/approvals`
List pending approval requests.

**Response**
```json
{
  "approvals": [
    {
      "id": "uuid",
      "action": "server_restart",
      "description": "...",
      "payload": {},
      "created_at": "2025-01-01T00:00:00+00:00",
      "status": "pending"
    }
  ]
}
```

### `POST /api/approvals/{approval_id}/approve`
Approve a pending action.

### `POST /api/approvals/{approval_id}/reject`
Reject a pending action.

---

## Audit Log

### `GET /api/audit?n=100`
Get the last N audit log entries.

**Response**
```json
{
  "entries": [
    {
      "timestamp": "2025-01-01T00:00:00+00:00",
      "actor": "api",
      "action": "settings_update",
      "before": {},
      "after": {},
      "details": ""
    }
  ]
}
```

---

## Benchmarks

### `POST /api/benchmark`
Run benchmarks.

**Request**
```json
{"type": "all"}
```
`type`: `"speed"` | `"vram"` | `"slots"` | `"all"`

**Response**
```json
{"type": "all", "results": {...}}
```
