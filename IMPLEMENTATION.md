# SlothBrain — Implementation Architecture

This document provides a deep-dive into every major subsystem of SlothBrain.  
For a high-level overview and setup instructions, see [README.md](README.md).

---

## Table of Contents

1. [AgenticLoop](#1-agenticloop)
2. [SafetySupervisor](#2-safetysupervisor)
3. [CheckpointManager](#3-checkpointmanager)
4. [Judge System](#4-judge-system)
5. [MainAgent](#5-mainagent)
6. [WatcherAgent](#6-watcheragent)
7. [SubAgent & AgentRegistry](#7-subagent--agentregistry)
8. [HandoffManager](#8-handoffmanager)
9. [Vision & Action Layer](#9-vision--action-layer)
10. [Resource & Slot Management](#10-resource--slot-management)
11. [Memory System](#11-memory-system)
12. [ServerManager & Watchdog](#12-servermanager--watchdog)
13. [Approval Queue & Audit Log](#13-approval-queue--audit-log)
14. [Configuration](#14-configuration)
15. [Data Flow Diagrams](#15-data-flow-diagrams)

---

## 1. AgenticLoop

**File:** `backend/agents/agentic_loop.py`

The `AgenticLoop` orchestrates multi-step autonomous task execution. It is the central execution engine of SlothBrain.

### Flow

```
run(task)
  ├── MainAgent.plan_task(task)          → ordered step list
  ├── for each step:
  │   ├── CheckpointManager.save(...)    → snapshot before execution
  │   ├── LoopHandle.heartbeat(...)      → signal liveness to supervisor
  │   ├── handle.pop_intervention()     → apply any supervisor action first
  │   ├── MainAgent.execute_step(...)   → execute the step
  │   ├── DesktopController.capture()   → optional screenshot
  │   └── WatcherAgent.monitor_step()  → continue | retry | done | abort
  └── WatcherAgent.verify_completion() → final pass/fail verdict
```

### Key Concepts

| Concept | Description |
|---|---|
| `AgenticStep` | State for one step: description, result, watcher feedback, status, screenshots, retries |
| `on_progress` | Optional async callback invoked after every event (used by the WebSocket endpoint) |
| `_MAX_STEP_RETRIES` | Default 2 — watcher can request retries before the loop moves on |
| `run_id` | UUID generated per `run()` call — used to scope checkpoints and supervisor handles |

### Supervisor Integration

Before executing each step the loop calls `handle.heartbeat()`.  The `SafetySupervisor`
polls this timestamp in the background.  If the step runs for longer than `step_timeout`
(default 120 s) the supervisor injects an *intervention* which the loop reads on the
**next** iteration of its attempt loop via `handle.pop_intervention()`.

### Intervention Actions

| Action | Loop Behaviour |
|---|---|
| `nudge` | Append a reminder to context; continue normal execution |
| `reset_context` | Restore checkpoint; jump back to the checkpoint step |
| `retry_step` | Re-execute the current step without touching context |
| `end_task` | Mark step as failed and return immediately |
| `escalate_to_user` | Same as `end_task` but emits an `escalated` event |

---

## 2. SafetySupervisor

**File:** `backend/core/safety_supervisor.py`

The `SafetySupervisor` is an independent Python-level watchdog that monitors all
running `AgenticLoop` instances.  It operates via a separate `asyncio` task so it
continues working even if the main agent slots are busy.

### Design Principles

- **LLM-independent baseline** — the supervisor detects stalls and defaults to
  `nudge` purely in Python, without needing the LLM to be responsive.
- **Best-effort Judge** — the Judge LLM call uses `slot_id=-1` (any free slot)
  and has a soft timeout.  If it fails, the supervisor falls back to `nudge`.
- **No shared mutable state** — the only shared object is `LoopHandle` which uses
  an `asyncio.Lock` for the intervention slot.

### LoopHandle

Each `AgenticLoop.run()` call receives a `LoopHandle` via `SafetySupervisor.register()`.

```
AgenticLoop                    SafetySupervisor
     │                               │
     │ heartbeat(step, task, ctx)    │
     │──────────────────────────────▶│  _last_heartbeat updated
     │                               │
     │                               │  poll() detects stall
     │                               │  _handle_stall()
     │                               │    restore_last(run_id)
     │                               │    _call_judge()
     │                               │    set_intervention(decision)
     │                               │
     │ pop_intervention()            │
     │◀──────────────────────────────│
     │                               │
```

### Supervisor Lifecycle

```python
supervisor = SafetySupervisor(llama_client, checkpoint_manager)
supervisor.start()   # creates asyncio background task
# ... loop runs ...
supervisor.stop()    # cancels task, clears all handles
```

---

## 3. CheckpointManager

**File:** `backend/core/checkpoint_manager.py`

Provides in-memory snapshots of task state so the `AgenticLoop` can recover from
stalls or errors without starting over.

### Storage Model

```
_store: dict[run_id → dict[step_num → TaskCheckpoint]]
```

Each `TaskCheckpoint` is a deep-copy of:
- `task` — original task string
- `step_num` — 1-based step index (the step *about to be executed*)
- `step_descriptions` — full ordered plan
- `context` — accumulated context lines from completed steps
- `executed_steps` — serialised `to_dict()` output for each finished step

### Eviction Policy

Each run keeps at most `max_checkpoints_per_run` (default 20) snapshots.  When the
cap is exceeded the checkpoint with the lowest step number is evicted.

### Recovery Path

```python
cp = checkpoint_manager.restore_last(run_id)
if cp:
    context = list(cp.context)
    executed = [_dict_to_step(s) for s in cp.executed_steps]
    idx = cp.step_num - 1   # 0-based index to restart from
```

> **TODO:** Add optional disk persistence so checkpoints survive backend restarts.

---

## 4. Judge System

**File:** `backend/core/safety_supervisor.py` — `_call_judge()` + `_JUDGE_SYSTEM_PROMPT`

The Judge is the LLM-based decision-making component of the `SafetySupervisor`.
It is invoked only when a stall is detected and only if a free inference slot is
available.

### Prompt Format

The Judge receives:
- The original task description
- Current step number and time since last heartbeat
- Last checkpoint info (step, count of completed steps)
- Up to 4 recent context lines

It responds with a JSON object:

```json
{
  "action": "nudge|reset_context|retry_step|end_task|escalate_to_user",
  "message": "Brief explanation of the decision"
}
```

### Parser

`_parse_judge_response(response)` uses `json.loads` first.  If that fails it falls
back to regex extraction of `"action":` and `"message":` keys, and ultimately to a
keyword scan in severity order.  The default on any failure is `nudge` — the least
disruptive option.

### Judge Actions by Severity

| Severity | Action | When to use |
|---|---|---|
| 1 (lowest) | `nudge` | Step is slow but probably still making progress |
| 2 | `retry_step` | Step produced bad output; retry without clearing context |
| 3 | `reset_context` | Context appears corrupted; restore from checkpoint |
| 4 | `end_task` | Task is irrecoverable; stop cleanly |
| 5 (highest) | `escalate_to_user` | Human input is required |

---

## 5. MainAgent

**File:** `backend/agents/main_agent.py`

The high-capability agent that plans complex tasks, executes individual steps, and
delegates to sub-agents for specialised work.

### Responsibilities

| Method | Description |
|---|---|
| `plan_task(task)` | Breaks a task into an ordered list of steps (max 10). Returns `{approach, steps}`. |
| `execute_step(step, task, context)` | Executes one step, feeding in accumulated context from previous steps. |
| `process(user_input, context_from_watcher)` | Direct single-turn conversation with memory retrieval/storage. |
| `spawn_sub_agent(preset_id, task, ...)` | Delegates a sub-task to a dynamically-spawned `SubAgent`. |

### System Prompt

Loaded from `backend/config/protected/main_system_prompt.txt`.  If the file is
missing, a built-in fallback prompt is used.  The protected file is read-only at
runtime so the LLM cannot overwrite its own instructions.

### Plan Parsing

`plan_task` requests JSON output:

```json
{
  "approach": "Brief strategy description",
  "steps": ["Step 1 description", "Step 2 description", "..."]
}
```

The parser uses `json.loads` first, then falls back to regex extraction of numbered
list items and a `STEPS:` header.

---

## 6. WatcherAgent

**File:** `backend/agents/watcher.py`

A lightweight always-on monitor that:
- Handles simple conversational turns directly (low latency, small context)
- Detects complex tasks and hands off to `MainAgent`
- Monitors each `AgenticLoop` step and decides `continue | retry | done | abort`
- Performs final verification after the loop completes

### Monitor Response Format

```json
{
  "action": "continue|retry|done|abort",
  "feedback": "Brief assessment"
}
```

### Verify Response Format

```json
{
  "complete": true,
  "feedback": "Verification summary"
}
```

Both parsers use `json.loads` first with a regex/keyword-scan fallback.

---

## 7. SubAgent & AgentRegistry

**Files:** `backend/agents/sub_agent.py`, `backend/agents/registry.py`,
`backend/agents/preset_manager.py`

Sub-agents are lightweight, task-specialised agents spawned on demand.

### How It Works

1. Operator creates a **Preset** (via API or UI): name, system prompt, context
   size, temperature, max tokens.
2. `MainAgent.spawn_sub_agent(preset_id, task_description)` calls
   `AgentRegistry.spawn(preset_id, ...)`.
3. The registry creates a `SubAgent` instance with `slot_id=-1` (llama.cpp picks
   any available slot).
4. The sub-agent runs its conversation in isolation (separate KV-cache context).
5. When done the agent stays alive in the registry until explicitly destroyed.

### Resource Budgeting

`MainAgent` can override preset defaults at spawn time:
```python
agent = self.spawn_sub_agent(
    preset_id="coder",
    task_description="Write a Flask REST API",
    context_size=16384,   # override default for a larger coding task
    max_tokens=4096,
)
```

---

## 8. HandoffManager

**File:** `backend/agents/handoff.py`

Routes single-turn requests to the appropriate agent:

```
user message
    │
    ▼
WatcherAgent.process()
    │
    ├── should_handoff()? ──yes──▶ MainAgent.process(with watcher context)
    │                                    │
    │                                    ▼
    │                             {"agent": "main", "handoff": true, ...}
    │
    └── no ──▶ {"agent": "watcher", "handoff": false, ...}
```

Handoff is triggered when the watcher response contains phrases like:
`"hand off"`, `"handoff"`, `"complex task"`, `"main agent"`.

---

## 9. Vision & Action Layer

**Files:** `backend/vision/`

Enables agents to interact with desktop applications.

### Components

| Component | File | Description |
|---|---|---|
| `DesktopController` | `controller.py` | High-level capture + command execution |
| `ActionExecutor` | `action_executor.py` | Translates `Action` dataclasses to pyautogui calls |
| `ScreenGrid` | `grid.py` | Divides screen into labelled cells (A1, B2, …) for coordinate abstraction |
| `ocr_cell_bytes()` | `ocr.py` | OCR a single grid cell image (pytesseract / easyocr) |
| `capture_screen()` | `screen_capture.py` | Multi-backend screenshot (mss or pyautogui) |

### Action Command Language

The model issues text commands:

```
SCREENSHOT                   # capture current state
CLICK A3                     # left-click cell A3
RIGHT_CLICK B5               # right-click
DOUBLE_CLICK C2              # double-click
CLICK_AND_TYPE D1 "hello"    # click then type
TYPE "some text"             # type at current focus
PRESS ctrl+c                 # key combo
SCROLL E4 DOWN 3             # scroll 3 clicks down
DRAG F2 G7                   # drag from F2 to G7
DONE                         # task complete
```

### Safety Constraints

- `_MAX_TYPE_LEN = 2000` — prevents huge type strings from prompt injection
- `_KEY_RE` — allows only safe key sequences (`[a-z0-9+\-_]+`)
- `pyautogui.FAILSAFE = True` — moving mouse to corner aborts all input

### Capabilities Detection

`DesktopController.capabilities()` returns the available backends at runtime so
the agentic loop can decide whether desktop control is possible.

---

## 10. Resource & Slot Management

### ResourceManager

**File:** `backend/core/resource_manager.py`

| Concept | Description |
|---|---|
| `idle` mode | Uses `idle_kv_quant` (default `q4`) — lower VRAM, faster |
| `active` mode | Uses `active_kv_quant` (default `q8`) — higher quality |
| `auto_adjust()` | Switches to `idle` if RAM usage exceeds `ram_threshold_mb` |
| `get_system_stats()` | Returns CPU %, RAM used/total, current mode |

### SlotManager

**File:** `backend/core/slot_manager.py`

Wraps `LlamaClient` with:
- Named slot assignment (`assign_watcher`, `assign_main`)
- Per-slot conversation history
- Response sanitisation (strips `system:/user:/assistant:` prefixes leaked by some
  models, truncates at stop sequences)

### LlamaClient

**File:** `backend/core/llama_client.py`

Thin async HTTP client over llama.cpp's REST API:

| Method | Endpoint | Description |
|---|---|---|
| `complete()` | `POST /completion` | Run inference on a specific slot (`id_slot`) |
| `get_slots()` | `GET /slots` | Fetch slot status |
| `health()` | `GET /health` | Server liveness |
| `get_metrics()` | `GET /metrics` | Prometheus-format metrics |

`slot_id=-1` tells llama.cpp to pick any free slot — used by sub-agents and the Judge.

> **TODO:** Add configurable retry with exponential back-off for transient HTTP errors.

---

## 11. Memory System

### LanceDBMemory

**File:** `backend/memory/lancedb_memory.py`

Persistent vector store backed by LanceDB + sentence-transformers.

- `store(text, metadata)` — embeds text and appends to the `memories` table.
- `search(query, limit)` — embeds query and performs ANN search.
- Embeddings are computed via `SentenceTransformer` in a thread pool to avoid
  blocking the event loop.
- Initialisation is lazy and protected by an `asyncio.Lock` so the first call
  sets up the table safely.

### RollingContext

**File:** `backend/memory/rolling_context.py`

In-process conversation buffer for the `WatcherAgent`.

- Accumulates `{role, content}` message dicts.
- Estimates token count as `len(content) // 4`.
- When estimate exceeds `summarize_at` (default 3000), calls the LLM to
  summarise the conversation and replaces all messages with the summary.

---

## 12. ServerManager & Watchdog

**File:** `backend/core/server_manager.py`

Manages the llama-server subprocess lifecycle.

| Feature | Description |
|---|---|
| `start()` | Launch `llama_server_path` with `llama_server_args` |
| `stop()` | SIGTERM + 10 s wait, then SIGKILL |
| `restart(actor)` | Rate-limited restart with pre-restart settings backup |
| Watchdog | Background task polls `/health` every 30 s; auto-restarts on failure |
| Rate limiting | Max `max_restarts_per_hour` (default 3) restarts per rolling hour |
| Backups | Settings JSON snapshotted to `data/backups/` before every restart |

---

## 13. Approval Queue & Audit Log

### ApprovalQueue

**File:** `backend/core/approval_queue.py`

Human-in-the-loop gate for critical actions.

Critical actions: `server_restart`, `kv_cache_change`, `large_context_increase`,
`emergency_stop`.

When a critical action is requested the API returns a `pending_approval` dict
instead of executing immediately.  A human must call `POST /api/approvals/{id}/approve`
before the action runs.

### AuditLog

**File:** `backend/core/audit_log.py`

Append-only JSONL log at `data/audit.log`.  Every mutating action records:
`timestamp`, `actor`, `action`, `before`, `after`, `details`.

---

## 14. Configuration

**File:** `backend/config/__init__.py`

`AppConfig` is a `pydantic-settings` model.  All fields can be overridden via:
1. Environment variables with `SLOTHBRAIN_` prefix
2. `.env` file in the project root

See [README.md](README.md) for the full settings table.

---

## 15. Data Flow Diagrams

### Agentic Loop with Safety

```
POST /api/chat/agentic
         │
         ▼
   AgenticLoop.run(task)
         │
         ├─── SafetySupervisor.register(run_id)
         │
         ├─── MainAgent.plan_task(task)
         │         └── JSON → step list
         │
         └─── for step in steps:
                  │
                  ├─ CheckpointManager.save(...)
                  ├─ LoopHandle.heartbeat(...)
                  ├─ LoopHandle.pop_intervention() → apply if any
                  ├─ MainAgent.execute_step(step, ctx)
                  ├─ DesktopController.capture()  [optional]
                  └─ WatcherAgent.monitor_step() → continue/retry/done/abort
                  │
         ├─── WatcherAgent.verify_completion()
         └─── SafetySupervisor.deregister(run_id)
              CheckpointManager.clear(run_id)
```

### Safety Supervisor Poll Cycle

```
SafetySupervisor._run()   [background asyncio task]
         │
         └─── every poll_interval seconds:
                  └─── for each registered LoopHandle:
                           if seconds_since_heartbeat > step_timeout:
                               handle.reset_heartbeat()
                               CheckpointManager.restore_last(run_id)
                               _call_judge(handle, checkpoint)
                                   └── LlamaClient.complete(slot=-1)
                                       JSON parse response
                               handle.set_intervention(decision)
```

### Memory Read/Write

```
MainAgent.process(user_input)
    │
    ├─ LanceDBMemory.search(user_input, limit=5)  ← retrieve relevant past context
    │      └── SentenceTransformer.encode(query) → ANN search
    │
    ├─ LlamaClient.complete(prompt + memory_context)
    │
    └─ LanceDBMemory.store(turn_text, metadata)   ← persist for future retrieval
           └── SentenceTransformer.encode(text) → append to table
```
