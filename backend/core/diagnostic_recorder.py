"""Offline agentic failure diagnostics – structured trace bundle recorder.

Every agentic run emits a self-contained JSON bundle to
``diagnostics/runs/{run_id}/bundle.json``.  The bundle is designed to be
read by a human *or* fed to a separate AI model for offline failure analysis,
without requiring a second running llama.cpp instance.

Bundle sections
---------------
* **run_metadata** – slot id, max steps, feature flags.
* **events** – time-ordered list of every significant runtime event:
  - ``run_start``          – task, metadata snapshot.
  - ``planning_request``   – prompt length, prompt preview.
  - ``planning_response``  – raw model reply, empty/error flags.
  - ``plan_parsed``        – approach + steps extracted.
  - ``step_start``         – step number, description.
  - ``checkpoint_saved``   – step num, monotonic timestamp.
  - ``checkpoint_restored``– step num restored to.
  - ``model_request``      – per-iteration prompt len, tools available.
  - ``model_response_raw`` – raw reply (capped), empty/tool-call flags.
  - ``tool_call_parsed``   – tool name + args.
  - ``tool_executed``      – name, ok, output preview, error.
  - ``step_complete``      – result preview, retries.
  - ``supervisor_intervention`` – action, message.
  - ``run_error``          – unhandled exception details.
  - ``run_complete``       – verified, total_steps.
* **final_result** – the dict returned by ``AgenticLoop.run()``.

Usage
-----
Set ``SLOTHBRAIN_DIAGNOSTICS_ENABLED=true`` to activate.  Each run writes
one bundle.  Completed bundles are listed via ``GET /api/diagnostics/runs``
and fetched via ``GET /api/diagnostics/runs/{run_id}``.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1"

# Caps to keep bundles from growing unwieldy on long tool-heavy runs.
_MAX_RESPONSE_CHARS = 8000    # raw model response stored in each event
_MAX_PROMPT_PREVIEW_CHARS = 600
_MAX_TOOL_OUTPUT_CHARS = 3000
_MAX_TOOL_ARG_CHARS = 1000


class DiagnosticRecorder:
    """Captures agentic-run events and writes diagnostic bundles to disk.

    Thread-safety: designed for a single asyncio event loop.  Each run is
    isolated by ``run_id``; concurrent runs from different IDs are safe.

    Parameters
    ----------
    output_dir:
        Root directory for bundles.  Subdirectory ``{output_dir}/{run_id}/``
        is created per run.
    enabled:
        When ``False`` every method is a no-op (zero overhead in production).
    """

    def __init__(
        self,
        output_dir: str | Path = "diagnostics/runs",
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._output_dir = Path(output_dir)
        # run_id → internal run state dict
        self._runs: dict[str, dict] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_id: str, task: str, metadata: dict) -> None:
        """Begin recording for a new run.  Must be called before ``record``."""
        if not self._enabled:
            return
        now = time.monotonic()
        self._runs[run_id] = {
            "task": task,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "start_mono": now,
            "metadata": dict(metadata),
            "events": [],
            "seq": 0,
        }
        self.record(run_id, "run_start", task=task)

    def record(self, run_id: str, event_type: str, **data: Any) -> None:
        """Append a timestamped event to the run's event log."""
        if not self._enabled:
            return
        state = self._runs.get(run_id)
        if state is None:
            return
        state["seq"] += 1
        elapsed = round(time.monotonic() - state["start_mono"], 4)
        event: dict = {"seq": state["seq"], "elapsed_s": elapsed, "type": event_type}
        event.update(data)
        state["events"].append(event)

    def finish_run(self, run_id: str, final_result: dict) -> None:
        """Finalise recording and flush the bundle to disk."""
        if not self._enabled:
            return
        state = self._runs.pop(run_id, None)
        if state is None:
            return
        finished_at = datetime.now(timezone.utc).isoformat()
        duration = round(time.monotonic() - state["start_mono"], 3)
        bundle = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": run_id,
            "task": state["task"],
            "started_at": state["started_at"],
            "finished_at": finished_at,
            "duration_seconds": duration,
            "run_metadata": state["metadata"],
            "events": state["events"],
            "final_result": _safe_serialize(final_result),
        }
        self._write_bundle(run_id, bundle)

    # ------------------------------------------------------------------
    # Typed convenience recorders
    # ------------------------------------------------------------------

    def record_planning_request(
        self,
        run_id: str,
        *,
        prompt: str,
    ) -> None:
        self.record(
            run_id,
            "planning_request",
            prompt_len=len(prompt),
            prompt_preview=prompt[:_MAX_PROMPT_PREVIEW_CHARS],
        )

    def record_planning_response(
        self,
        run_id: str,
        *,
        response: str,
        error: str | None = None,
    ) -> None:
        stripped = (response or "").strip()
        self.record(
            run_id,
            "planning_response",
            response=stripped[:_MAX_RESPONSE_CHARS],
            response_len=len(stripped),
            is_empty=not bool(stripped),
            error=error,
        )

    def record_model_request(
        self,
        run_id: str,
        *,
        step_num: int | None,
        iteration: int,
        prompt: str,
        tools_available: list[str],
    ) -> None:
        self.record(
            run_id,
            "model_request",
            step_num=step_num,
            iteration=iteration,
            prompt_len=len(prompt),
            prompt_preview=prompt[:_MAX_PROMPT_PREVIEW_CHARS],
            tools_available=tools_available,
        )

    def record_model_response(
        self,
        run_id: str,
        *,
        step_num: int | None,
        iteration: int,
        response: str,
        error: str | None = None,
    ) -> None:
        stripped = (response or "").strip()
        self.record(
            run_id,
            "model_response_raw",
            step_num=step_num,
            iteration=iteration,
            response=stripped[:_MAX_RESPONSE_CHARS],
            response_len=len(stripped),
            is_empty=not bool(stripped),
            has_tool_call="<tool_call>" in stripped.lower(),
            error=error,
        )

    def record_tool_call_parsed(
        self,
        run_id: str,
        *,
        step_num: int | None,
        iteration: int,
        tool_name: str,
        args: dict,
    ) -> None:
        self.record(
            run_id,
            "tool_call_parsed",
            step_num=step_num,
            iteration=iteration,
            tool=tool_name,
            args=_truncate_dict(args, _MAX_TOOL_ARG_CHARS),
        )

    def record_tool_executed(
        self,
        run_id: str,
        *,
        step_num: int | None,
        tool_name: str,
        ok: bool,
        output: Any = None,
        error: Any = None,
    ) -> None:
        output_preview = ""
        if output is not None:
            try:
                output_preview = json.dumps(output, ensure_ascii=False, default=str)[
                    :_MAX_TOOL_OUTPUT_CHARS
                ]
            except Exception:
                output_preview = str(output)[:_MAX_TOOL_OUTPUT_CHARS]
        error_str = str(error)[:500] if error is not None else None
        self.record(
            run_id,
            "tool_executed",
            step_num=step_num,
            tool=tool_name,
            ok=ok,
            output_preview=output_preview,
            error=error_str,
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _write_bundle(self, run_id: str, bundle: dict) -> None:
        try:
            run_dir = self._output_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = run_dir / "bundle.json"
            bundle_path.write_text(
                json.dumps(bundle, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Diagnostic bundle written: %s", bundle_path)
        except Exception as exc:
            logger.warning(
                "Failed to write diagnostic bundle for run %s: %s", run_id, exc
            )

    # ------------------------------------------------------------------
    # Read helpers (used by API endpoints)
    # ------------------------------------------------------------------

    def list_runs(self) -> list[dict]:
        """Return summary metadata for all completed bundles on disk, newest first."""
        runs: list[dict] = []
        if not self._output_dir.exists():
            return runs
        entries = sorted(
            (e for e in self._output_dir.iterdir() if e.is_dir()),
            key=lambda e: e.stat().st_mtime,
            reverse=True,
        )
        for entry in entries:
            bundle_path = entry / "bundle.json"
            if not bundle_path.exists():
                continue
            try:
                data = json.loads(bundle_path.read_text(encoding="utf-8"))
                final = data.get("final_result") or {}
                runs.append(
                    {
                        "run_id": data.get("run_id", entry.name),
                        "task": data.get("task", ""),
                        "started_at": data.get("started_at", ""),
                        "duration_seconds": data.get("duration_seconds"),
                        "event_count": len(data.get("events", [])),
                        "completion_verified": final.get("completion_verified"),
                        "total_steps": final.get("total_steps"),
                    }
                )
            except Exception:
                pass
        return runs

    def get_bundle(self, run_id: str) -> dict | None:
        """Load and return a specific bundle by run_id, or None if not found."""
        bundle_path = self._output_dir / run_id / "bundle.json"
        if not bundle_path.exists():
            return None
        try:
            return json.loads(bundle_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Failed to read diagnostic bundle for run %s: %s", run_id, exc
            )
            return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_serialize(value: Any) -> Any:
    """Return a JSON-serializable version of *value*, truncating large strings."""
    if isinstance(value, dict):
        return {k: _safe_serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, str):
        return value[:_MAX_TOOL_OUTPUT_CHARS]
    return value


def _truncate_dict(d: dict, max_chars: int) -> dict:
    """Return a copy of *d* with string values truncated to *max_chars*."""
    out: dict = {}
    for k, v in (d or {}).items():
        if isinstance(v, str):
            out[k] = v[:max_chars]
        elif isinstance(v, (dict, list)):
            try:
                serialized = json.dumps(v, ensure_ascii=False, default=str)
                out[k] = serialized[:max_chars]
            except Exception:
                out[k] = str(v)[:max_chars]
        else:
            out[k] = v
    return out
