# SlothBrain Architecture Overview

This document summarises the key architectural decisions and design patterns used in SlothBrain.

For a full deep-dive into each subsystem, see [IMPLEMENTATION.md](../IMPLEMENTATION.md).

---

## Design Principles

1. **LLM-independent safety baseline** — The `SafetySupervisor` detects stalls and defaults to `nudge` purely in Python without requiring the LLM to be responsive. The Judge is only called opportunistically when a free slot is available.

2. **Graceful degradation** — Every LLM call is wrapped in `try/except`. Failures produce conservative defaults (`nudge`, `continue`, empty context) rather than crashing the system.

3. **Separation of concerns** — Planning (`MainAgent.plan_task`) is separated from execution (`MainAgent.execute_step`), monitoring (`WatcherAgent.monitor_step`), and safety (`SafetySupervisor`). Each component can evolve independently.

4. **Structured outputs** — All LLM responses that must be machine-parsed use JSON format, with multi-level fallback (JSON → regex → keyword scan → safe default). This makes the system robust against models that do not perfectly follow instructions.

5. **Human-in-the-loop gates** — Critical actions (server restart, KV cache changes, large context increases) require explicit human approval via the `ApprovalQueue`. The audit log records every mutation.

6. **Resource isolation** — Sub-agents use `slot_id=-1` (any free slot) so they never block the main watcher/main slots. Context sizes are bounded per preset and per-call overrides let the `MainAgent` right-size allocations.

---

## Component Dependency Graph

```
FastAPI app (main.py)
    │
    ├── AgenticLoop
    │       ├── MainAgent ────── SlotManager ── LlamaClient
    │       │       └── AgentRegistry ── SubAgent
    │       ├── WatcherAgent ─── SlotManager
    │       │       └── RollingContext
    │       ├── CheckpointManager
    │       └── SafetySupervisor ─── LlamaClient (Judge)
    │                            └── CheckpointManager
    │
    ├── HandoffManager ── WatcherAgent + MainAgent
    ├── ResourceManager ── LlamaClient
    ├── ServerManager ── LlamaClient (health check)
    ├── LanceDBMemory (shared by MainAgent + WatcherAgent + SubAgent)
    ├── ApprovalQueue
    └── AuditLog
```

---

## Thread Safety

The backend is entirely `asyncio`-based. All components use async/await.
The only cross-task synchronisation is:
- `LoopHandle._lock` (asyncio.Lock) — protects the intervention slot between the loop task and the supervisor task.
- `LanceDBMemory._init_lock` (asyncio.Lock) — protects table initialisation during the first concurrent store/search calls.
- `RollingContext` — not thread-safe; intended for single-consumer (one WatcherAgent instance per context).

---

## Event Flow (Agentic Task)

```
Client ──POST /api/chat/agentic──▶ FastAPI
                                       │
                                  AgenticLoop.run(task)
                                       │
                              ┌────────▼────────┐
                              │  Plan            │  MainAgent.plan_task()
                              │  → steps[1..N]  │  JSON: {approach, steps}
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  For each step:  │
                              │                  │
                              │  1. Checkpoint   │  CheckpointManager.save()
                              │  2. Heartbeat    │  LoopHandle.heartbeat()
                              │  3. Intervention │  pop_intervention() → apply
                              │  4. Execute      │  MainAgent.execute_step()
                              │  5. Screenshot   │  DesktopController.capture()
                              │  6. Monitor      │  WatcherAgent.monitor_step()
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  Verify          │  WatcherAgent.verify_completion()
                              └────────┬────────┘
                                       │
                              Return result dict
```

---

## Supervisor Event Flow (Background)

```
SafetySupervisor._run_with_restart()  [asyncio background task]
    │
    └── every poll_interval (15s):
            for each LoopHandle:
                if seconds_since_heartbeat > step_timeout (120s):
                    handle.reset_heartbeat()
                    cp = CheckpointManager.restore_last(run_id)
                    decision = await _call_judge(handle, cp)
                        └── LlamaClient.complete(slot=-1, max_tokens=128)
                            JSON parse: {action, message}
                    handle.set_intervention(decision)
                    # AgenticLoop reads this on next attempt iteration
```

---

## Data Persistence

| Data | Storage | Location | Survives restart? |
|---|---|---|---|
| Long-term memory | LanceDB (disk) | `data/lancedb/` | ✅ Yes |
| Agent presets | JSON files | `data/agent_presets/` | ✅ Yes |
| Audit log | JSONL file | `data/audit.log` | ✅ Yes |
| Settings backups | JSON files | `data/backups/` | ✅ Yes |
| Checkpoints | In-memory | Python dict | ❌ No (see BUG-004) |
| Slot history | In-memory | SlotManager | ❌ No |
| RollingContext | In-memory | WatcherAgent | ❌ No |

---

## Configuration Precedence

```
Environment variables (SLOTHBRAIN_*)
    override
.env file
    override
AppConfig defaults
```

All settings are accessible and updatable at runtime via `POST /api/settings`.
Critical settings changes flow through the ApprovalQueue.
