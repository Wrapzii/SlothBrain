# SlothBrain — TODO List

Prioritised task backlog. Items are grouped by phase and ordered highest → lowest priority within each phase.

---

## Phase 1 — Core Infrastructure (Complete ✅)

- [x] AgenticLoop: plan → execute → monitor → verify
- [x] SafetySupervisor: heartbeat detection, Judge LLM, intervention injection
- [x] CheckpointManager: in-memory snapshots, eviction policy
- [x] MainAgent: plan_task, execute_step, spawn_sub_agent
- [x] WatcherAgent: process, monitor_step, verify_completion, should_handoff
- [x] SubAgent + AgentRegistry + PresetManager
- [x] LlamaClient: /completion, /slots, /health, /metrics
- [x] SlotManager: watcher/main slot assignment, response sanitisation
- [x] ResourceManager: idle/active mode, RAM threshold auto-switch
- [x] LanceDBMemory: store + ANN search
- [x] RollingContext: rolling summarisation
- [x] HandoffManager: watcher → main routing
- [x] ServerManager: start/stop/restart, rate limiting, settings backup, watchdog
- [x] ApprovalQueue + AuditLog
- [x] FastAPI backend with auth middleware (loopback-only / api_key)
- [x] DesktopController + ActionExecutor + ScreenGrid + OCR layer

---

## Phase 2 — Self-Healing Reliability (In Progress 🔄)

### High Priority

- [ ] **Persistent checkpoints** — Serialise `TaskCheckpoint` to disk (JSON) so runs survive backend restarts. (`backend/core/checkpoint_manager.py`)
- [ ] **Supervisor restart on task death** — If `AgenticLoop._execute` raises an uncaught exception, have the supervisor detect this and emit a final audit event. Currently the loop may silently die without cleanup.
- [ ] **Configurable Judge timeout** — `_call_judge` has no explicit timeout; a slow Judge response blocks the supervisor. Add `asyncio.wait_for` with a configurable deadline.
- [ ] **LlamaClient retry logic** — Add exponential back-off for transient HTTP errors (connection refused, 503). (`backend/core/llama_client.py`)
- [ ] **Watcher monitor/verify JSON strict mode** — Add a `--strict-json` config flag; when enabled, re-prompt the model if its response is not valid JSON rather than falling back to regex.

### Medium Priority

- [ ] **Multi-run supervisor** — Allow the supervisor to monitor runs across multiple `AgenticLoop` instances running concurrently (e.g. from parallel API requests).
- [ ] **Checkpoint compaction** — When context lines grow very long, summarise them before saving to keep checkpoint size bounded.
- [ ] **Intervention history** — Track all supervisor interventions per run so the Judge can see prior decisions and avoid repeated identical actions.
- [ ] **Graceful shutdown** — On SIGTERM, allow active agentic runs to complete the current step before stopping.

### Low Priority

- [ ] **Supervisor metrics endpoint** — Expose stall counts, intervention counts, and Judge decision distribution via `/api/supervisor/stats`.
- [ ] **Test: supervisor fires while Judge is slow** — Edge case where a second stall is detected before the first Judge call returns.

---

## Phase 3 — Code Execution Sandbox (Planned 📋)

- [ ] **Python REPL tool** — Safe sandboxed Python execution the agent can call as a tool during `execute_step`.
- [ ] **Shell execution tool** — Allowlist-based shell command runner with timeout and output capture.
- [ ] **File I/O tool** — Read/write files within a controlled workspace directory.
- [ ] **Tool registry** — Formal `Tool` interface that `MainAgent` can discover and invoke by name.
- [ ] **Tool result injection** — Feed tool output back into step context automatically.
- [ ] **Code test runner** — Execute `pytest` or `unittest` and return structured results.

---

## Phase 4 — Desktop Automation (Planned 📋)

- [ ] **Multimodal vision** — Pass annotated screenshots directly to a vision-capable model (e.g. LLaVA) instead of relying solely on OCR text.
- [ ] **OCR quality improvements** — Add preprocessing (binarisation, upscaling) before pytesseract to improve accuracy on small or low-contrast text.
- [ ] **Faster screenshot backend** — Profile mss vs. D3DShot on Windows for latency.
- [ ] **Action verification** — After each desktop action, compare pre/post screenshot diffs to verify the action had the expected effect.
- [ ] **Window management** — Tools to list open windows, focus a specific window, resize/move windows.
- [ ] **Clipboard integration** — COPY and PASTE action types for efficient text transfer.

---

## Phase 5 — Multi-Model Routing (Planned 📋)

- [ ] **Model router** — Route tasks to different llama.cpp model instances based on task type (coding, reasoning, vision, summarisation).
- [ ] **Parallel llama.cpp servers** — Support connecting to multiple llama.cpp endpoints.
- [ ] **Dynamic model loading** — API to hot-swap the loaded model without restarting the backend.
- [ ] **Per-agent model binding** — Presets can specify which model endpoint to use.

---

## Phase 6 — Self-Improvement (Planned 📋)

- [ ] **Self-modifying presets** — MainAgent can propose edits to its own system prompt via the approval queue.
- [ ] **Automated benchmark regression** — Run benchmarks after config changes and flag regressions.
- [ ] **Goal tracking** — Persistent goals that survive restarts; the system autonomously works towards them between user sessions.
- [ ] **Self-scheduling** — Cron-style task scheduler the agent can add entries to.

---

## Infrastructure & Ops

- [ ] **Docker Compose** — `docker-compose.yml` for backend + llama.cpp server + optional frontend.
- [ ] **Health dashboard** — Single-page HTML health summary (no build step needed) served at `/health/dashboard`.
- [ ] **Log rotation** — Rotate `data/audit.log` when it exceeds a configurable size.
- [ ] **CI pipeline** — GitHub Actions workflow: lint (ruff), type-check (mypy), test (pytest).
- [ ] **Pre-commit hooks** — ruff + mypy + pytest smoke test.
- [ ] **`.env.example`** — Committed example environment file.
