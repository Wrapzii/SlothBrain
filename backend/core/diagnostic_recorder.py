"""Offline agentic failure diagnostics – structured trace bundle recorder.

Every run emits a self-contained JSON bundle to
``diagnostics/runs/{run_id}/bundle.json``.  The bundle is designed to be
read by a human *or* fed to a separate AI model for offline failure analysis,
without requiring a second running llama.cpp instance.

Crash safety
------------
Events are flushed **incrementally** so that no data is lost if the process
terminates before :meth:`finish_run` is called:

* :meth:`start_run` creates ``{run_dir}/partial.json`` (run metadata) and
  opens ``{run_dir}/events.jsonl`` for appended writes.
* Every :meth:`record` call appends one JSON line to ``events.jsonl``
  **before** returning.
* :meth:`finish_run` assembles the final ``bundle.json`` from in-memory
  events, then removes ``partial.json``.  If the process crashes after
  ``start_run`` but before ``finish_run``, all events recorded so far are
  preserved in ``events.jsonl``.

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
Diagnostics are **enabled by default**.  Set
``SLOTHBRAIN_DIAGNOSTICS_ENABLED=false`` to opt out.  Each run writes one
bundle.  Completed bundles are listed via ``GET /api/diagnostics/runs`` and
fetched via ``GET /api/diagnostics/runs/{run_id}``.

Path safety
-----------
The output directory must resolve to a path **inside** the project root.
An incorrect ``diagnostics_output_dir`` value that would write outside the
project is rejected at construction time.
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

# Project root is three levels up from this file:
# backend/core/diagnostic_recorder.py → backend/core → backend → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_and_validate_output_dir(output_dir: str | Path) -> Path:
    """Resolve *output_dir* and, for relative paths, verify it stays inside the project root.

    Relative paths are anchored to the project root so that
    ``diagnostics/runs`` always means ``{project_root}/diagnostics/runs``
    regardless of the process working directory.  Paths that would escape
    the project root (e.g. ``../../etc``) are rejected.

    Absolute paths are accepted as-is so that tests and explicit
    deployments can write to arbitrary locations.

    Raises
    ------
    ValueError
        If a relative path resolves outside the project root.
    """
    path = Path(output_dir)
    if path.is_absolute():
        return path.resolve()
    # Relative path: anchor to project root and check for escapes.
    resolved = (_PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT.resolve())
    except ValueError:
        raise ValueError(
            f"diagnostics_output_dir {output_dir!r} resolves to {resolved!r} "
            f"which escapes the project root {_PROJECT_ROOT.resolve()!r}. "
            "Set SLOTHBRAIN_DIAGNOSTICS_OUTPUT_DIR to a path inside the project."
        )
    return resolved


class DiagnosticRecorder:
    """Captures agentic-run events and writes diagnostic bundles to disk.

    Thread-safety: designed for a single asyncio event loop.  Each run is
    isolated by ``run_id``; concurrent runs from different IDs are safe.

    Parameters
    ----------
    output_dir:
        Root directory for bundles.  Subdirectory ``{output_dir}/{run_id}/``
        is created per run.  Must resolve to a path inside the project root.
    enabled:
        When ``False`` every method is a no-op (zero overhead).
    """

    def __init__(
        self,
        output_dir: str | Path = "diagnostics/runs",
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        if enabled:
            try:
                self._output_dir = _resolve_and_validate_output_dir(output_dir)
            except ValueError as exc:
                logger.error(
                    "DiagnosticRecorder: disabling diagnostics due to invalid output_dir — %s", exc
                )
                self._enabled = False
                self._output_dir = Path(output_dir)
        else:
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
        started_at = datetime.now(timezone.utc).isoformat()
        # run_dir is None if the directory could not be created; in that case
        # events are still kept in memory but not incrementally flushed.
        run_dir: Path | None = self._output_dir / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            # Write partial.json immediately so the run is visible even if
            # the process crashes before finish_run() is called.
            partial = {
                "schema_version": _SCHEMA_VERSION,
                "run_id": run_id,
                "task": task,
                "started_at": started_at,
                "run_metadata": dict(metadata),
                "status": "in_progress",
            }
            (run_dir / "partial.json").write_text(  # type: ignore[operator]
                json.dumps(partial, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(
                "DiagnosticRecorder: failed to create run dir for %s: %s", run_id, exc
            )
            run_dir = None

        self._runs[run_id] = {
            "task": task,
            "started_at": started_at,
            "start_mono": now,
            "metadata": dict(metadata),
            "events": [],
            "seq": 0,
            "run_dir": run_dir,
        }
        self.record(run_id, "run_start", task=task)

    def record(self, run_id: str, event_type: str, **data: Any) -> None:
        """Append a timestamped event to the run's event log.

        The event is written to ``events.jsonl`` immediately so no data is
        lost if the process terminates before :meth:`finish_run`.
        """
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

        # Incremental flush — append to events.jsonl right away.
        run_dir: Path | None = state.get("run_dir")
        if run_dir is not None:
            try:
                with (run_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, ensure_ascii=False, default=str))
                    fh.write("\n")
            except Exception as exc:
                logger.warning(
                    "DiagnosticRecorder: failed to write event to events.jsonl for %s: %s",
                    run_id, exc,
                )

    def finish_run(self, run_id: str, final_result: dict) -> None:
        """Finalise recording and flush the complete bundle to disk."""
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
        self._write_bundle(run_id, bundle, run_dir=state.get("run_dir"))

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
        slot_id: int | None = None,
    ) -> None:
        self.record(
            run_id,
            "model_request",
            step_num=step_num,
            iteration=iteration,
            slot_id=slot_id,
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
        slot_id: int | None = None,
    ) -> None:
        stripped = (response or "").strip()
        self.record(
            run_id,
            "model_response_raw",
            step_num=step_num,
            iteration=iteration,
            slot_id=slot_id,
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

    def _write_bundle(self, run_id: str, bundle: dict, run_dir: Path | None = None) -> None:
        if run_dir is None:
            run_dir = self._output_dir / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = run_dir / "bundle.json"
            bundle_path.write_text(
                json.dumps(bundle, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Diagnostic bundle written: %s", bundle_path)
            # Remove partial.json now that bundle.json supersedes it.
            partial_path = run_dir / "partial.json"
            if partial_path.exists():
                try:
                    partial_path.unlink()
                except Exception:
                    pass
            # Auto-run the analyzer immediately after writing the bundle.
            self._run_analyzer(run_id, bundle, run_dir)
        except Exception as exc:
            logger.warning(
                "Failed to write diagnostic bundle for run %s: %s", run_id, exc
            )

    def _run_analyzer(self, run_id: str, bundle: dict, run_dir: Path) -> None:
        """Run the failure-mode analyzer and write analysis artefacts."""
        try:
            # Local import: DiagnosticAnalyzer is a sibling module that itself
            # imports no recorder state, so there is no circular dependency at
            # runtime.  The local import keeps the recorder module lightweight
            # and allows the analyzer to be replaced or disabled independently.
            from backend.core.diagnostic_analyzer import DiagnosticAnalyzer

            analyzer = DiagnosticAnalyzer(run_dir=run_dir)
            findings = analyzer.analyze()
            if findings:
                logger.info(
                    "Diagnostic analyzer found %d finding(s) for run %s: %s",
                    len(findings),
                    run_id,
                    [f["id"] for f in findings],
                )
            else:
                logger.info("Diagnostic analyzer: no failures detected for run %s", run_id)
        except Exception as exc:
            logger.warning(
                "Diagnostic analyzer failed for run %s: %s", run_id, exc
            )

    # ------------------------------------------------------------------
    # Read helpers (used by API endpoints)
    # ------------------------------------------------------------------

    def list_runs(self) -> list[dict]:
        """Return summary metadata for all runs on disk, newest first.

        Includes both completed runs (``bundle.json`` present) and
        partial/crashed runs (``partial.json`` or ``events.jsonl`` present
        without a ``bundle.json``).
        """
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
            if bundle_path.exists():
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
                            "status": "complete",
                        }
                    )
                except Exception:
                    pass
                continue

            # Partial / crashed run — try partial.json first, then events.jsonl.
            partial_path = entry / "partial.json"
            jsonl_path = entry / "events.jsonl"
            try:
                if partial_path.exists():
                    meta = json.loads(partial_path.read_text(encoding="utf-8"))
                    task = meta.get("task", "")
                    started_at = meta.get("started_at", "")
                elif jsonl_path.exists():
                    # Read just the first line (run_start event).
                    with jsonl_path.open(encoding="utf-8") as fh:
                        first = json.loads(fh.readline())
                    task = first.get("task", "")
                    started_at = ""
                else:
                    continue
                event_count = 0
                if jsonl_path.exists():
                    with jsonl_path.open(encoding="utf-8") as fh:
                        event_count = sum(1 for _ in fh)
                runs.append(
                    {
                        "run_id": entry.name,
                        "task": task,
                        "started_at": started_at,
                        "duration_seconds": None,
                        "event_count": event_count,
                        "completion_verified": None,
                        "total_steps": None,
                        "status": "partial",
                    }
                )
            except Exception:
                pass
        return runs

    def get_bundle(self, run_id: str) -> dict | None:
        """Load and return a specific bundle by run_id, or None if not found.

        For partial/crashed runs (no ``bundle.json`` yet) the partial metadata
        and all events from ``events.jsonl`` are assembled and returned with
        ``"status": "partial"``.
        """
        run_dir = self._output_dir / run_id
        bundle_path = run_dir / "bundle.json"
        if bundle_path.exists():
            try:
                return json.loads(bundle_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(
                    "Failed to read diagnostic bundle for run %s: %s", run_id, exc
                )
                return None

        # Partial run: try to assemble from events.jsonl + partial.json.
        jsonl_path = run_dir / "events.jsonl"
        partial_path = run_dir / "partial.json"
        if not jsonl_path.exists() and not partial_path.exists():
            return None
        try:
            meta: dict = {}
            if partial_path.exists():
                meta = json.loads(partial_path.read_text(encoding="utf-8"))
            events: list[dict] = []
            if jsonl_path.exists():
                with jsonl_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                events.append(json.loads(line))
                            except Exception:
                                pass
            return {
                "schema_version": _SCHEMA_VERSION,
                "run_id": run_id,
                "task": meta.get("task", ""),
                "started_at": meta.get("started_at", ""),
                "finished_at": None,
                "duration_seconds": None,
                "run_metadata": meta.get("run_metadata", {}),
                "events": events,
                "final_result": None,
                "status": "partial",
            }
        except Exception as exc:
            logger.warning(
                "Failed to assemble partial bundle for run %s: %s", run_id, exc
            )
            return None

    def get_analysis(self, run_id: str) -> dict | None:
        """Return ``analysis.json`` for *run_id*, or ``None`` if not found."""
        analysis_path = self._output_dir / run_id / "analysis.json"
        if not analysis_path.exists():
            return None
        try:
            return json.loads(analysis_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Failed to read analysis.json for run %s: %s", run_id, exc
            )
            return None

    def get_review_prompt(self, run_id: str) -> str | None:
        """Return ``model_review_prompt.md`` content for *run_id*, or ``None``."""
        path = self._output_dir / run_id / "model_review_prompt.md"
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning(
                "Failed to read model_review_prompt.md for run %s: %s", run_id, exc
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
