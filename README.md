# 🧠 SlothBrain

A persistent local AI assistant that connects to a locally-running **llama.cpp** server, manages dual-agent inference slots, dynamically controls GPU resources, and maintains long-term memory via **LanceDB**.

Primary interaction is terminal-based via `backend/cli.py` (no web UI required).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend (Vite)                   │
│  Dashboard │ Chat │ Settings │ Benchmarks                    │
│  WebSocket live stats ─────────────────────────────────────┐│
└─────────────────────────┬───────────────────────────────────┘│
                          │ HTTP / WS                          │
┌─────────────────────────▼───────────────────────────────────▼┐
│                   FastAPI Backend (Python)                     │
│                                                               │
│  ┌────────────────────┐    ┌──────────────────────────────┐  │
│  │   HandoffManager   │    │      ResourceManager          │  │
│  │  ┌─────────────┐   │    │  idle / active mode           │  │
│  │  │WatcherAgent │   │    │  auto VRAM threshold adjust   │  │
│  │  │  Slot 0     │   │    └──────────────────────────────┘  │
│  │  └──────┬──────┘   │                                       │
│  │         │handoff?  │    ┌──────────────────────────────┐  │
│  │  ┌──────▼──────┐   │    │        SlotManager            │  │
│  │  │  MainAgent  │   │    │  per-slot context history     │  │
│  │  │  Slot 1     │   │    └──────────────────────────────┘  │
│  │  └─────────────┘   │                                       │
│  └────────────────────┘    ┌──────────────────────────────┐  │
│                             │       LanceDBMemory           │  │
│  ┌────────────────────┐    │  sentence-transformers embed  │  │
│  │   RollingContext   │    │  ANN search over sessions     │  │
│  │  auto-summarize    │    └──────────────────────────────┘  │
│  └────────────────────┘                                       │
│                                                               │
│  ┌────────────────────┐    ┌──────────────────────────────┐  │
│  │   BenchmarkSuite   │    │       LlamaClient             │  │
│  │  speed/vram/slots  │    │  httpx → llama.cpp REST API   │  │
│  └────────────────────┘    └──────────┬───────────────────┘  │
└──────────────────────────────────────┬┘                      │
                                        │ HTTP                  │
                          ┌─────────────▼────────────┐         │
                          │  llama.cpp server          │         │
                          │  :8080  (2 slots)          │         │
                          └────────────────────────────┘
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- A running [llama.cpp](https://github.com/ggerganov/llama.cpp) server with `--parallel 2`

### Backend

```bash
# From repo root
pip install -r requirements.txt

# Start the API server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal Interface (Recommended)

```bash
# From repo root
python -m backend.cli
# or
python -m backend.cli --agent main
```

### Optional Web Frontend

```bash
cd frontend
npm install
npm run dev        # Dev server on http://localhost:5173
```

---

## Configuration

All settings can be changed via the **Settings** tab in the UI or by setting environment variables prefixed with `SLOTHBRAIN_`:

| Field | Default | Description |
|---|---|---|
| `llama_host` | `127.0.0.1` | llama.cpp server host |
| `llama_port` | `8080` | llama.cpp server port |
| `watcher_slot` | `0` | Slot ID for the Watcher agent |
| `main_slot` | `1` | Slot ID for the Main agent |
| `watcher_context_size` | `4096` | Context window for Watcher |
| `main_context_size` | `32768` | Context window for Main agent |
| `idle_kv_quant` | `q4` | KV cache quantization in idle mode |
| `active_kv_quant` | `q8` | KV cache quantization in active mode |
| `lancedb_path` | `./data/lancedb` | Path for the LanceDB memory store |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-Transformers model for memory |
| `vram_threshold_mb` | `2048` | RAM threshold (MB) that triggers idle mode |
| `mode` | `idle` | Initial operating mode |
| `discord_bot_token` | `` | Optional Discord bot token for HITL notifications and actions |
| `discord_owner_user_id` | `0` | Discord user ID allowed to approve/reject actions (owner-only) |

Example `.env`:
```
SLOTHBRAIN_LLAMA_PORT=8080
SLOTHBRAIN_VRAM_THRESHOLD_MB=4096
SLOTHBRAIN_MODE=active
SLOTHBRAIN_DISCORD_BOT_TOKEN=your_bot_token
SLOTHBRAIN_DISCORD_OWNER_USER_ID=123456789012345678
```

---

## Usage

### Chat (Terminal)

Run:

```bash
python -m backend.cli
```

Built-in commands:

- `/agent auto|watcher|main`
- `/mode idle|active`
- `/status`
- `/help`
- `/quit`

### Chat (Web, optional)

Navigate to the **Chat** tab. Select an agent routing strategy:
- **auto** – Watcher handles the request; if it detects a complex task it hands off to Main.
- **watcher** – Always use the lightweight Watcher agent (slot 0).
- **main** – Always use the high-performance Main agent (slot 1).

### Resource Modes

- **Idle** – Uses `idle_kv_quant` (default q4). Lower VRAM footprint.
- **Active** – Uses `active_kv_quant` (default q8). Higher quality, more VRAM.

Toggle via the Dashboard or `POST /api/mode {"mode": "active"}`.

### REST API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/api/status` | System stats + slot info |
| POST | `/api/chat` | Send a message |
| GET/POST | `/api/mode` | Get/set operating mode |
| GET/POST | `/api/settings` | Get/update configuration |
| POST | `/api/benchmark` | Run benchmarks |
| GET | `/api/memory/search?q=` | Semantic memory search |
| WS | `/ws/status` | Live stats stream (2s interval) |

---

## Benchmarking

From the **Benchmarks** tab or via API:

```bash
# Inference speed across context lengths
curl -X POST http://localhost:8000/api/benchmark -H 'Content-Type: application/json' \
  -d '{"type": "speed"}'

# VRAM / metrics snapshot
curl -X POST http://localhost:8000/api/benchmark -d '{"type": "vram"}'

# Slot interference (parallel requests)
curl -X POST http://localhost:8000/api/benchmark -d '{"type": "slots"}'

# All benchmarks
curl -X POST http://localhost:8000/api/benchmark -d '{"type": "all"}'
```

---

## Memory System

SlothBrain uses **LanceDB** + **sentence-transformers** for persistent long-term memory:

- Every conversation turn (watcher + main) is embedded and stored with metadata.
- On new requests, the Main agent performs an ANN search to retrieve relevant past sessions.
- The `RollingContext` class keeps per-slot message history and automatically summarizes when the token estimate exceeds `summarize_at` (default 3000 tokens).

Memory is stored at `./data/lancedb` (configurable). Delete this directory to reset memory.

---

## Running Tests

```bash
pytest backend/tests/ -v
```
