# SlothBrain — Known Bugs & Issues

Severity scale: **Critical** | **High** | **Medium** | **Low**

---

## Critical

None currently known.

---

## High

### BUG-001 — `_call_judge` has no asyncio timeout
**File:** `backend/core/safety_supervisor.py` — `_call_judge()`  
**Severity:** High  
**Description:** The Judge LLM call (`LlamaClient.complete`) has the default httpx timeout (120 s) but no `asyncio.wait_for` wrapper. If llama.cpp is under heavy load and the Judge call takes > 120 s, the supervisor is effectively blocked on that one run and cannot service other handles during this time.  
**Impact:** Other stalled loops are not detected while the Judge is waiting.  
**Fix:** Wrap the `_client.complete(...)` call in `asyncio.wait_for(coro, timeout=30.0)`.

---

### BUG-002 — `RollingContext` summarisation blocks event loop briefly
**File:** `backend/memory/rolling_context.py` — `_summarize()`  
**Severity:** High  
**Description:** `_summarize()` calls `self._client.complete(...)` directly (async, fine), but it does so while holding a soft logical lock (it is awaited inside `add_message`). If the LLM is slow, every watcher turn stalls until summarisation completes.  
**Impact:** WatcherAgent throughput degrades noticeably when context rolls over.  
**Fix:** Run summarisation as a background task and replace messages only after it completes; in the meantime keep the last N messages as-is.

---

### BUG-003 — `LanceDBMemory._get_table()` creates a seed row then deletes it
**File:** `backend/memory/lancedb_memory.py` — `_get_table()`  
**Severity:** High  
**Description:** LanceDB requires at least one row to infer the schema at table creation time. The code adds a zero-vector dummy row then immediately calls `self._table.delete("text = ''")`. If the backend crashes between these two calls the table is left with a corrupt dummy row that may affect ANN search results.  
**Impact:** Rare data corruption; ANN search results include the zero-vector row.  
**Fix:** Use LanceDB's `schema=` parameter (PyArrow schema) at `create_table` time to avoid needing a seed row.

---

## Medium

### BUG-004 — `CheckpointManager` data is lost on backend restart
**File:** `backend/core/checkpoint_manager.py`  
**Severity:** Medium  
**Description:** All checkpoints are stored in-memory only. If the `uvicorn` process crashes mid-task, all checkpoint data is lost and the `AgenticLoop` cannot recover.  
**Impact:** Any task that was mid-flight is unrecoverable after a crash.  
**Fix:** Serialise checkpoints to `data/checkpoints/{run_id}/step_{n}.json` (see TODO.md Phase 2).

---

### BUG-005 — `PresetManager` uses a relative path `data/agent_presets`
**File:** `backend/agents/preset_manager.py`  
**Severity:** Medium  
**Description:** `PRESETS_DIR = Path("data/agent_presets")` is relative to the current working directory. If the backend is started from a different directory than the project root, preset files cannot be found or are written to the wrong location.  
**Impact:** Agent presets silently created in the wrong directory; `KeyError` on `get_preset`.  
**Fix:** Use `Path(__file__).parent.parent.parent / "data" / "agent_presets"` for a path relative to the source file.

---

### BUG-006 — `ServerManager.start()` swallows subprocess stderr
**File:** `backend/core/server_manager.py`  
**Severity:** Medium  
**Description:** Both `stdout` and `stderr` are set to `asyncio.subprocess.DEVNULL`. If the llama-server fails to start (e.g., model file not found, port conflict), there is no way to diagnose the failure from the SlothBrain logs.  
**Impact:** Difficult to debug llama-server startup failures.  
**Fix:** Redirect stderr to a log file at `data/llama_server.log` and rotate it.

---

### BUG-007 — `apply_intervention` does not guard against missing `_cp`
**File:** `backend/agents/agentic_loop.py` — `_apply_intervention()`  
**Severity:** Medium  
**Description:** When `action == "reset_context"`, the code calls `self._cp.restore_last(run_id)` without checking if `self._cp is not None` first (it does check `if self._cp is not None` in the `else` branch but not inside the `if action == "reset_context":` block itself — wait, on re-reading line 511 it does check. This is partially mitigated but the fallback `context.append(...)` mutates the list that was passed by the outer function making the side effect visible). Low actual risk.  
**Impact:** Low. When supervisor fires `reset_context` without a checkpoint manager the loop appends a fallback note and continues.  
**Fix:** Log a warning; no crash risk.

---

### BUG-008 — `WatcherAgent.should_handoff()` uses simple substring match
**File:** `backend/agents/watcher.py`  
**Severity:** Medium  
**Description:** Handoff detection checks if any of `_HANDOFF_PHRASES` appear anywhere in the response text. A response like "I don't need to hand off this task" would incorrectly trigger a handoff because it contains the phrase "hand off".  
**Impact:** Spurious handoffs to the MainAgent for simple tasks.  
**Fix:** Either use the watcher's structured response (e.g. an explicit `HANDOFF: yes/no` field) or improve the phrase matching to avoid false positives on negations.

---

## Low

### BUG-009 — `AuditLog.tail()` reads entire file into memory
**File:** `backend/core/audit_log.py` — `tail()`  
**Severity:** Low  
**Description:** `tail(n)` reads the entire file and takes the last `n` lines. For long-running deployments the audit log can grow large and this read blocks the event loop momentarily.  
**Impact:** Minor latency spike on `/api/audit` requests after days of use.  
**Fix:** Read the file in reverse using a seek-based approach, or implement log rotation.

---

### BUG-010 — `LlamaClient` creates a new `httpx.AsyncClient` per request
**File:** `backend/core/llama_client.py`  
**Severity:** Low  
**Description:** Each call to `complete()`, `get_slots()`, `health()`, or `get_metrics()` creates and destroys an `httpx.AsyncClient`. This is inefficient; a persistent client with connection pooling would be significantly faster under load.  
**Impact:** Minor per-request overhead; more noticeable when the agentic loop makes many rapid calls.  
**Fix:** Create a single `httpx.AsyncClient` instance in `__init__` and close it explicitly on shutdown.

---

### BUG-011 — `_parse_plan` regex misses multi-word step numbers beyond 9
**File:** `backend/agents/main_agent.py` — `_parse_plan()`  
**Severity:** Low  
**Description:** `re.findall(r"^\d+\.\s+(.+)", ...)` will match `10.` correctly but the plan is capped at 10 steps anyway. Not a real bug, just worth noting.  
**Impact:** None with current 10-step cap.

---

### BUG-012 — `RollingContext.token_estimate` uses a rough heuristic
**File:** `backend/memory/rolling_context.py`  
**Severity:** Low  
**Description:** Token count is estimated as `len(content) // 4`. For non-English text or text with many special tokens this estimate can be significantly off.  
**Impact:** Context may roll over earlier or later than intended for non-English content.  
**Fix:** Use a proper tokeniser (e.g. `tiktoken` or the llama.cpp `/tokenize` endpoint) for accurate counting.
