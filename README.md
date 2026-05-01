# 🧠 SlothBrain

**SlothBrain** is a self-healing, 24/7 autonomous AI system designed to act as a super-smart personal assistant and agentic workflow engine. It runs fully locally against a **llama.cpp** inference server, manages a hierarchy of AI agents and sub-agents, executes multi-step tasks with automatic recovery, and maintains long-term memory via **LanceDB**.

> **Current Phase:** Phase 1 — Core agentic infrastructure ✅  
> See [TODO.md](TODO.md) for the full roadmap.

---

## 🎯 Project Goals

- **Self-healing workflows** — stalled or crashed agent tasks are automatically detected and recovered by the `SafetySupervisor` + `CheckpointManager`.
- **Hierarchical agent management** — a `MainAgent` plans and delegates; dynamically-spawned `SubAgent` instances handle specialised sub-tasks, while the Python `SafetySupervisor` watches for stalls and recovery conditions.
- **Self-completing tasks** — the `AgenticLoop` plans, executes, monitors, and verifies complex multi-step tasks without human intervention.
- **Research, code generation & app development** — agents write and execute code, manage files, and build software end-to-end.
- **Desktop / GUI control** — the `DesktopController` + `ActionExecutor` layer lets agents read screen state (OCR), click, type, and interact with any application.
- **Always-on personal assistant** — persistent long-term memory means the system remembers every past interaction and surfaces relevant context automatically.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                     Python TUI (Textual)                           │
│    Dashboard │ Chat │ Settings │ Benchmarks │ Agents │ Approvals   │
└────────────────────────────┬───────────────────────────────────────┘
                             │ HTTP / WS
┌────────────────────────────▼────────────────────────────────────────┐
│                      FastAPI Backend (Python)                        │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                       AgenticLoop                             │  │
│  │  plan → checkpoint → execute → finalize                      │  │
│  │      SafetySupervisor (heartbeat / Judge / recovery)          │  │
│  │      CheckpointManager (in-memory snapshots per run)          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─────────────────────┐    ┌───────────────────────────────────┐  │
│  │      MainAgent      │    │         ResourceManager            │  │
│  │       Slot 1        │    │  idle / active mode                │  │
│  │     spawns          │    │  auto RAM-threshold mode switch    │  │
│  │    SubAgent(s)      │    └───────────────────────────────────┘  │
│  └─────────────────────┘    ┌───────────────────────────────────┐  │
│                              │           SlotManager              │  │
│                              │  per-slot KV-cache / history       │  │
│                              └───────────────────────────────────┘  │
│                              ┌───────────────────────────────────┐  │
│                              │          AgentRegistry             │  │
│                              │  preset-driven dynamic agents      │  │
│                              └───────────────────────────────────┘  │
│                              ┌───────────────────────────────────┐  │
│                              │         LanceDBMemory              │  │
│  ┌─────────────────────┐    │  sentence-transformers embeds      │  │
│  │   RollingContext    │    │  ANN search over all sessions      │  │
│  │  auto-summarise     │    └───────────────────────────────────┘  │
│  └─────────────────────┘                                            │
│                                                                      │
│  ┌─────────────────────┐    ┌───────────────────────────────────┐  │
│  │  DesktopController  │    │          LlamaClient               │  │
│  │  OCR + ActionExec   │    │  httpx → llama.cpp REST API        │  │
│  └─────────────────────┘    └────────────────┬──────────────────┘  │
│                                               │                      │
│  ┌─────────────────────┐                     │                      │
│  │    ServerManager    │  ──── watchdog ─────┘                      │
│  │  start/stop/restart │                                             │
│  └─────────────────────┘                                            │
└───────────────────────────────────────────────────────────────────┘
                               │ HTTP
                 ┌─────────────▼────────────┐
                 │    llama.cpp server        │
                 │    :8080  (N slots)        │
                 └────────────────────────────┘
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- A running [llama.cpp](https://github.com/ggerganov/llama.cpp) server with `--parallel 2`

### Install Dependencies

```bash
# From repo root
pip install -r requirements.txt
```

For memory and workspace indexing, the runtime also needs the LanceDB data stack that backs `.to_pandas()` query paths. The pinned requirements include `lancedb`, `sentence-transformers`, `numpy`, `pandas`, and `pyarrow`.

### Start SlothBrain

```bash
# From repo root
python run_slothbrain.py
```

This single command starts:
- FastAPI backend on `127.0.0.1:8000`
- Textual TUI in the same terminal session

---

## Configuration

All settings can be changed via the **Settings** tab in the UI or by setting environment variables prefixed with `SLOTHBRAIN_`:

| Field | Default | Description |
|---|---|---|
| `llama_host` | `127.0.0.1` | llama.cpp server host |
| `llama_port` | `8080` | llama.cpp server port |
| `main_slot` | `1` | Slot ID for the Main agent |
| `main_context_size` | `32768` | Context window for Main agent |
| `idle_kv_quant` | `q4` | KV cache quantization in idle mode |
| `active_kv_quant` | `q8` | KV cache quantization in active mode |
| `lancedb_path` | `./data/lancedb` | Path for the LanceDB memory store |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-Transformers model for memory |
| `vram_threshold_mb` | `2048` | RAM threshold (MB) that triggers idle mode |
| `mode` | `idle` | Initial operating mode |

Example `.env`:
```
SLOTHBRAIN_LLAMA_PORT=8080
SLOTHBRAIN_VRAM_THRESHOLD_MB=4096
SLOTHBRAIN_MODE=active
```

---

## Usage

### Chat

Run:

```bash
python run_slothbrain.py
```

In the TUI Chat tab, chat is **Direct** by default for quick replies.

- Send a normal message for a fast single-shot response.
- Prefix with `/task ` to run full **Agentic** mode.
- In Agentic mode, the task is planned and executed step-by-step.
- The final task summary is shown in the chat panel.

### Resource Modes

- **Idle** – Uses `idle_kv_quant` (default q4). Lower VRAM footprint.
- **Active** – Uses `active_kv_quant` (default q8). Higher quality, more VRAM.

Toggle via the Dashboard or `POST /api/mode {"mode": "active"}`.

### Agentic Task Execution

Run a fully autonomous multi-step task:

```bash
curl -X POST http://localhost:8000/api/chat/agentic \
  -H 'Content-Type: application/json' \
  -d '{"task": "Research the latest Python async features and write a summary", "max_steps": 5}'
```

The loop automatically: plans steps → saves checkpoints → executes → finalizes. If a step stalls for > 120 s the `SafetySupervisor` fires the Judge LLM which decides whether to nudge, retry, reset context, end the task, or escalate to you.

### REST API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/api/status` | System stats + slot info |
| POST | `/api/chat` | Agentic chat alias (backward-compatible) |
| POST | `/api/chat/agentic` | Run a full multi-step agentic task |
| GET/POST | `/api/mode` | Get/set operating mode |
| GET/POST | `/api/settings` | Get/update configuration |
| POST | `/api/benchmark` | Run benchmarks |
| GET | `/api/memory/search?q=` | Semantic memory search |
| GET | `/api/presets` | List agent presets |
| POST | `/api/presets` | Create a new agent preset |
| POST | `/api/presets/{id}/spawn` | Spawn a sub-agent from a preset |
| GET | `/api/agents` | List running sub-agents |
| POST | `/api/agents/{id}/chat` | Chat with a specific sub-agent |
| GET | `/api/audit` | View the audit log |
| GET | `/api/approvals` | List pending approvals |
| POST | `/api/approvals/{id}/approve` | Approve a critical action |
| POST | `/api/server/restart` | Restart llama-server (requires approval) |
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

- Every conversation turn is embedded and stored with metadata.
- On new requests, the Main agent performs an ANN search to retrieve relevant past sessions.
- The `RollingContext` class keeps per-slot message history and automatically summarizes when the token estimate exceeds `summarize_at` (default 3000 tokens).

If memory is disabled at startup, verify that the full LanceDB runtime stack is installed: `lancedb`, `sentence-transformers`, `numpy`, `pandas`, and `pyarrow`.

Memory is stored at `./data/lancedb` (configurable). Delete this directory to reset memory.

---

## Running Tests

```bash
pytest backend/tests/ -v
```

Capability contract lane (full-process simulation of core assistant workflows):

```bash
pytest backend/tests/test_capability_pipeline.py -q
```

Manual full-stack LLM suite (real llama.cpp + real runtime wiring):

```bash
SLOTHBRAIN_RUN_MANUAL_LLM_TESTS=1 pytest backend/tests/manual/test_manual_llm_fullstack.py -q -s
```

Optional strict mode (fail if any optional/runtime-dependent tool is unavailable):

```bash
SLOTHBRAIN_RUN_MANUAL_LLM_TESTS=1 SLOTHBRAIN_MANUAL_STRICT=1 pytest backend/tests/manual/test_manual_llm_fullstack.py -q -s
```

This suite validates:
- tool inventory (`GET /api/tools`)
- direct per-tool smoke execution (`POST /api/tools/run`)
- LLM-driven tool call flow through websocket agentic loop (`/ws/agent-progress`)

---

### Comprehensive Validation Suite

Run validation tests to verify critical subsystems with **real LLM I/O visible**:

```bash
export SLOTHBRAIN_RUN_VALIDATION_TESTS=1
pytest backend/tests/manual/test_validation_suite.py -v -s
```

This suite validates with actual LLM execution (not mocks):
- **Rolling Context** - message accumulation, summarization, token counting
- **Checkpoints** - save/restore state preservation, multi-step recovery
- **Memory Cache** - vector embedding, semantic search ranking, retrieval accuracy
- **Slot Manager** - per-slot isolation, history tracking, cache performance
- **Context Tree** - message ordering, parent-child relationships, structure preservation
- **LLM Quality** - response correctness, format compliance, consistency
- **Integration** - full pipeline correctness (context + checkpoints + memory)
- **Performance** - operation latency under load

Every test includes:
- **INPUT**: visible data fed to the system
- **OUTPUT**: what the system produces
- **VALIDATION**: explicit checks of expected behavior

See [VALIDATION.md](VALIDATION.md) for detailed documentation.

CI runs on every push/PR via [.github/workflows/ci-full-pipeline.yml](.github/workflows/ci-full-pipeline.yml):
- `backend-tests`: complete backend test suite
- `capability-contracts`: commit-gated full pipeline capability contracts
- `real-windows-desktop-e2e`: optional self-hosted Windows lane for real desktop/vision execution

Manual self-hosted workflow for true LLM+tool full-stack validation:
- [.github/workflows/manual-llm-fullstack.yml](.github/workflows/manual-llm-fullstack.yml)

---

## Project Structure

```
SlothBrain/
├── backend/
│   ├── agents/          # MainAgent, AgenticLoop, SubAgent
│   ├── benchmarks/      # Speed, VRAM, and slot-interference benchmarks
│   ├── config/          # AppConfig (pydantic-settings, .env support)
│   ├── core/            # LlamaClient, SlotManager, ResourceManager,
│   │                    # CheckpointManager, SafetySupervisor,
│   │                    # ServerManager, ApprovalQueue, AuditLog
│   ├── memory/          # LanceDBMemory, RollingContext
│   ├── tests/           # pytest unit + integration tests
│   ├── vision/          # DesktopController, ActionExecutor, ScreenGrid, OCR
│   ├── cli.py           # Legacy terminal interface (optional)
│   └── main.py          # FastAPI app + lifespan wiring
├── tui/                 # Textual TUI (optional)
├── run_slothbrain.py    # Single-command launcher (backend + TUI)
├── data/                # LanceDB, audit.log, agent presets, backups (gitignored)
├── docs/                # In-depth architecture and API docs
├── IMPLEMENTATION.md    # Architecture deep-dive
├── TODO.md              # Prioritised roadmap
├── BUGS.md              # Known issues
└── requirements.txt
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| **Phase 1** | ✅ Complete | Core agentic infrastructure: AgenticLoop, SafetySupervisor, CheckpointManager, SubAgent system |
| **Phase 2** | 🔄 In Progress | Self-healing reliability: persistent checkpoints, multi-run supervisor, richer Judge decisions |
| **Phase 3** | 📋 Planned | Code execution sandbox: Python REPL, shell, file I/O within controlled environment |
| **Phase 4** | 📋 Planned | Full desktop automation: vision-driven GUI interaction, OCR pipeline improvements |
| **Phase 5** | 📋 Planned | Multi-model routing: hot-swap models per task type; dedicated coding, reasoning, vision models |
| **Phase 6** | 📋 Planned | Self-improvement: the system can update its own presets, prompts, and configuration |

See [TODO.md](TODO.md) for the detailed task list and [IMPLEMENTATION.md](IMPLEMENTATION.md) for the architecture deep-dive.

