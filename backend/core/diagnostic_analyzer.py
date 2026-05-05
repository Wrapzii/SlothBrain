"""Offline agentic failure analyzer.

Reads a completed ``bundle.json`` produced by :class:`DiagnosticRecorder`
and outputs three artefacts in the same run directory:

* ``analysis.json``          – machine-readable list of detected failure modes.
* ``summary.md``             – human-readable narrative describing what went
                               wrong, what worked, and what to inspect next.
* ``model_review_prompt.md`` – ready-to-paste prompt for a second AI model
                               (Qwen / GPT / Claude / etc.) that returns a
                               targeted patch plan.

Failure modes detected
----------------------
Each finding is a dict with these keys:

``id``
    A short ALL_CAPS identifier (e.g. ``LLM_EMPTY_RESPONSE``).
``severity``
    ``"critical"`` | ``"high"`` | ``"medium"`` | ``"low"``.
``confidence``
    Float 0–1 (1 = certain).
``summary``
    One-line human description.
``likely_area``
    Comma-separated list of code areas most likely responsible.
``evidence``
    List of supporting event summaries from the bundle.

Usage
-----
The analyzer is invoked automatically by
:class:`~backend.core.diagnostic_recorder.DiagnosticRecorder` when
``diagnostics_enabled=True`` and a run finishes.  It can also be called
manually:

.. code-block:: python

    from backend.core.diagnostic_analyzer import DiagnosticAnalyzer
    analyzer = DiagnosticAnalyzer(run_dir=Path("diagnostics/runs/my-run"))
    findings = analyzer.analyze()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure-mode identifiers
# ---------------------------------------------------------------------------
_FM_PLANNING_EMPTY_RESPONSE = "PLANNING_EMPTY_RESPONSE"
_FM_PLAN_PARSE_FAILURE = "PLAN_PARSE_FAILURE"
_FM_NO_STEPS_GENERATED = "NO_STEPS_GENERATED"
_FM_LLM_EMPTY_RESPONSE = "LLM_EMPTY_RESPONSE"
_FM_MODEL_DID_NOT_CALL_TOOL = "MODEL_DID_NOT_CALL_REQUIRED_TOOL"
_FM_TOOL_CALL_PARSE_FAILURE = "TOOL_CALL_PARSE_FAILURE"
_FM_TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
_FM_TOOL_RESULT_NOT_USED = "TOOL_RESULT_NOT_USED"
_FM_STEP_EMPTY_RESULT = "STEP_COMPLETED_WITH_EMPTY_RESULT"
_FM_CHECKPOINT_RESTORE_LOOP = "CHECKPOINT_RESTORE_LOOP"
_FM_SUPERVISOR_RETRY_LOOP = "SUPERVISOR_RETRY_LOOP"
_FM_FINALIZER_DROPPED_RESULTS = "FINALIZER_DROPPED_RESULTS"

# Step result sentinel strings set by AgenticLoop when a step produces no
# useful output.  Kept in sync with agentic_loop.py so the analyzer can
# recognise generic "execution failed" placeholders.
_STEP_RESULT_EMPTY_SENTINEL = "Execution error: empty model response"
_STEP_RESULT_ERROR_PREFIX = "Execution error:"


class DiagnosticAnalyzer:
    """Reads a bundle and produces analysis artefacts in the run directory.

    Parameters
    ----------
    run_dir:
        Directory containing ``bundle.json``.  Must already exist when
        :meth:`analyze` is called.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir)
        self._bundle_path = self._run_dir / "bundle.json"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self) -> list[dict]:
        """Load the bundle, detect failure modes, and write output files.

        Returns
        -------
        list[dict]
            The list of findings dicts (empty when no failures are detected).
        """
        bundle = self._load_bundle()
        if bundle is None:
            return []

        findings = detect_failure_modes(bundle)
        self._write_analysis(bundle, findings)
        self._write_summary(bundle, findings)
        self._write_review_prompt(bundle, findings)
        return findings

    def get_analysis(self) -> dict | None:
        """Return the saved ``analysis.json`` content, or ``None``."""
        path = self._run_dir / "analysis.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read analysis.json: %s", exc)
            return None

    def get_review_prompt(self) -> str | None:
        """Return the saved ``model_review_prompt.md`` content, or ``None``."""
        path = self._run_dir / "model_review_prompt.md"
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read model_review_prompt.md: %s", exc)
            return None

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_bundle(self) -> dict | None:
        if not self._bundle_path.exists():
            logger.warning("Bundle not found: %s", self._bundle_path)
            return None
        try:
            return json.loads(self._bundle_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load bundle: %s", exc)
            return None

    def _write_analysis(self, bundle: dict, findings: list[dict]) -> None:
        out = {
            "schema_version": "1",
            "run_id": bundle.get("run_id", ""),
            "task": bundle.get("task", ""),
            "started_at": bundle.get("started_at", ""),
            "duration_seconds": bundle.get("duration_seconds"),
            "final_verified": (bundle.get("final_result") or {}).get("completion_verified"),
            "total_steps": (bundle.get("final_result") or {}).get("total_steps"),
            "finding_count": len(findings),
            "findings": findings,
        }
        try:
            (self._run_dir / "analysis.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to write analysis.json: %s", exc)

    def _write_summary(self, bundle: dict, findings: list[dict]) -> None:
        task = bundle.get("task", "(unknown task)")
        run_id = bundle.get("run_id", "")
        duration = bundle.get("duration_seconds")
        events = bundle.get("events", [])
        final = bundle.get("final_result") or {}
        verified = final.get("completion_verified")
        total_steps = final.get("total_steps", 0)

        lines: list[str] = [
            f"# Diagnostic Summary — {run_id}",
            "",
            f"**Task:** {task}",
            f"**Duration:** {duration}s",
            f"**Steps completed:** {total_steps}",
            f"**Verified:** {'✅ Yes' if verified else '❌ No'}",
            "",
        ]

        # ── Stats ────────────────────────────────────────────────────────
        model_responses = [e for e in events if e.get("type") == "model_response_raw"]
        tool_calls = [e for e in events if e.get("type") == "tool_call_parsed"]
        tool_execs = [e for e in events if e.get("type") == "tool_executed"]
        empty_responses = [e for e in model_responses if e.get("is_empty")]
        failed_tools = [e for e in tool_execs if not e.get("ok")]

        lines += [
            "## Run Statistics",
            "",
            f"- Model responses: {len(model_responses)} ({len(empty_responses)} empty)",
            f"- Tool calls parsed: {len(tool_calls)}",
            f"- Tools executed: {len(tool_execs)} ({len(failed_tools)} failed)",
            "",
        ]

        # ── Findings ─────────────────────────────────────────────────────
        if not findings:
            lines += [
                "## Findings",
                "",
                "No failure modes detected. Run completed normally.",
                "",
            ]
        else:
            lines += [
                f"## Findings ({len(findings)})",
                "",
            ]
            for f in findings:
                severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(
                    f.get("severity", ""), "⚪"
                )
                lines += [
                    f"### {severity_icon} [{f.get('severity', '?').upper()}] {f.get('id')}",
                    "",
                    f"**Summary:** {f.get('summary')}",
                    f"**Likely area:** `{f.get('likely_area')}`",
                    f"**Confidence:** {int(f.get('confidence', 0) * 100)}%",
                ]
                evidence = f.get("evidence", [])
                if evidence:
                    lines.append("")
                    lines.append("**Evidence:**")
                    for item in evidence[:5]:
                        lines.append(f"- {item}")
                lines.append("")

        # ── Next steps ───────────────────────────────────────────────────
        if findings:
            critical = [f for f in findings if f.get("severity") == "critical"]
            lines += [
                "## Suggested Next Steps",
                "",
            ]
            for f in (critical or findings)[:3]:
                lines.append(f"1. Inspect `{f.get('likely_area')}` for `{f.get('id')}`")
            lines.append(
                "\nUse `GET /api/diagnostics/runs/{run_id}/review-prompt` to get a "
                "ready-to-paste prompt for a second AI model."
            )

        try:
            (self._run_dir / "summary.md").write_text(
                "\n".join(lines), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to write summary.md: %s", exc)

    def _write_review_prompt(self, bundle: dict, findings: list[dict]) -> None:
        task = bundle.get("task", "")
        run_id = bundle.get("run_id", "")
        events = bundle.get("events", [])
        final = bundle.get("final_result") or {}
        metadata = bundle.get("run_metadata") or {}

        # Condense events to keep the prompt compact
        event_lines: list[str] = []
        for e in events:
            etype = e.get("type", "")
            elapsed = e.get("elapsed_s", 0)
            step = e.get("step_num")
            step_tag = f" [step {step}]" if step is not None else ""

            if etype == "model_response_raw":
                flag = " ⚠️ EMPTY" if e.get("is_empty") else (
                    " (has_tool_call)" if e.get("has_tool_call") else ""
                )
                slot_tag = f" slot={e['slot_id']}" if e.get("slot_id") is not None else ""
                event_lines.append(
                    f"  [{elapsed:.2f}s]{step_tag} model_response_raw{slot_tag}{flag} "
                    f"len={e.get('response_len', 0)}"
                )
            elif etype == "model_request":
                slot_tag = f" slot={e['slot_id']}" if e.get("slot_id") is not None else ""
                event_lines.append(
                    f"  [{elapsed:.2f}s]{step_tag} model_request{slot_tag} "
                    f"prompt_len={e.get('prompt_len', 0)} "
                    f"tools={e.get('tools_available', [])}"
                )
            elif etype == "tool_call_parsed":
                event_lines.append(
                    f"  [{elapsed:.2f}s]{step_tag} tool_call_parsed tool={e.get('tool')}"
                )
            elif etype == "tool_executed":
                ok_flag = "✅" if e.get("ok") else "❌"
                event_lines.append(
                    f"  [{elapsed:.2f}s]{step_tag} tool_executed {ok_flag} tool={e.get('tool')} "
                    + (f"error={e['error']}" if e.get("error") else "")
                )
            elif etype == "step_complete":
                result_flag = " (empty result)" if not (e.get("result_preview") or "").strip() else ""
                event_lines.append(
                    f"  [{elapsed:.2f}s] step_complete step={e.get('step_num')} "
                    f"status={e.get('status')}{result_flag}"
                )
            elif etype in ("planning_response", "planning_request"):
                flag = " ⚠️ EMPTY" if e.get("is_empty") else ""
                event_lines.append(f"  [{elapsed:.2f}s] {etype}{flag}")
            elif etype == "supervisor_intervention":
                event_lines.append(
                    f"  [{elapsed:.2f}s]{step_tag} supervisor_intervention "
                    f"action={e.get('action')} msg={str(e.get('message', ''))[:80]}"
                )
            elif etype in ("run_start", "run_complete", "run_error"):
                event_lines.append(f"  [{elapsed:.2f}s] {etype}")
            # skip checkpoint_saved / step_start noise to keep prompt compact

        findings_section = ""
        if findings:
            f_lines = ["## Automatic findings\n"]
            for f in findings:
                f_lines.append(
                    f"- [{f.get('severity','?').upper()}] {f.get('id')}: "
                    f"{f.get('summary')} "
                    f"(confidence {int(f.get('confidence', 0)*100)}%)\n"
                    f"  Area: {f.get('likely_area')}"
                )
            findings_section = "\n".join(f_lines) + "\n"
        else:
            findings_section = "## Automatic findings\n\nNo failures automatically detected.\n"

        prompt = f"""You are diagnosing a failed or underperforming SlothBrain agentic run.

## Run info

- run_id: {run_id}
- task: {task}
- slot_id: {metadata.get('slot_id')}
- max_steps: {metadata.get('max_steps')}
- tool_calls_enabled: {metadata.get('tool_calls_enabled')}
- planning_enabled: {metadata.get('planning_enabled')}
- final verified: {final.get('completion_verified')}
- total_steps: {final.get('total_steps')}

## Event trace (condensed)

{chr(10).join(event_lines)}

{findings_section}

## What to focus on

- Model output validity (empty responses, malformed tool calls)
- Whether `<tool_call>` blocks were parsed correctly
- Whether tool results were re-injected into the prompt context
- Slot/run/step consistency (slot_id mismatch between requests)
- Whether the finalizer dropped useful tool results
- System prompt / tool prompt contract violations

Do NOT suggest changing model size unless the trace proves model failure.
Do NOT suggest generic improvements; base every conclusion on specific events.

## Your response format

1. **Most likely failure mode** – one sentence
2. **Supporting evidence** – bullet list of specific event entries from the trace
3. **Exact code areas to inspect** – file:function or file:line_range
4. **Patch plan** – concrete code changes (diffs or pseudocode)
5. **Tests to add** – test names and what they should assert
"""

        try:
            (self._run_dir / "model_review_prompt.md").write_text(
                prompt, encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to write model_review_prompt.md: %s", exc)


# ---------------------------------------------------------------------------
# Heuristic failure-mode detectors
# ---------------------------------------------------------------------------

def detect_failure_modes(bundle: dict) -> list[dict]:
    """Run all heuristic detectors against *bundle* and return findings.

    Each finding is a dict with keys: ``id``, ``severity``, ``confidence``,
    ``summary``, ``likely_area``, ``evidence``.
    """
    events: list[dict] = bundle.get("events", [])
    final: dict = bundle.get("final_result") or {}
    findings: list[dict] = []

    planning_responses = [e for e in events if e.get("type") == "planning_response"]
    plan_parsed = [e for e in events if e.get("type") == "plan_parsed"]
    model_responses = [e for e in events if e.get("type") == "model_response_raw"]
    model_requests = [e for e in events if e.get("type") == "model_request"]
    tool_calls = [e for e in events if e.get("type") == "tool_call_parsed"]
    tool_execs = [e for e in events if e.get("type") == "tool_executed"]
    step_completes = [e for e in events if e.get("type") == "step_complete"]
    supervisor_events = [e for e in events if e.get("type") == "supervisor_intervention"]
    checkpoint_restores = [e for e in events if e.get("type") == "checkpoint_restored"]

    # ── 1. Planning phase empty response ────────────────────────────────
    empty_plans = [e for e in planning_responses if e.get("is_empty")]
    if empty_plans:
        findings.append({
            "id": _FM_PLANNING_EMPTY_RESPONSE,
            "severity": "critical",
            "confidence": 0.95,
            "summary": "Planning model returned empty output; no task steps could be generated.",
            "likely_area": "MainAgent.plan_task, LlamaClient, slot routing",
            "evidence": [
                f"planning_response at +{e.get('elapsed_s', '?')}s: is_empty=True"
                + (f", error={e['error']}" if e.get("error") else "")
                for e in empty_plans[:3]
            ],
        })

    # ── 2. Plan parse produced no steps ──────────────────────────────────
    bad_parses = [e for e in plan_parsed if not e.get("steps")]
    if bad_parses or (planning_responses and not plan_parsed):
        findings.append({
            "id": _FM_NO_STEPS_GENERATED,
            "severity": "critical",
            "confidence": 0.9,
            "summary": "Plan parsing returned zero steps; loop ran with no actionable plan.",
            "likely_area": "MainAgent._parse_plan / plan_task",
            "evidence": (
                [f"plan_parsed at +{e.get('elapsed_s','?')}s: steps={e.get('steps')}" for e in bad_parses[:3]]
                if bad_parses
                else ["planning_response emitted but no plan_parsed event found"]
            ),
        })

    # ── 3. Empty LLM responses during execution ───────────────────────────
    empty_responses = [e for e in model_responses if e.get("is_empty")]
    if empty_responses:
        findings.append({
            "id": _FM_LLM_EMPTY_RESPONSE,
            "severity": "critical",
            "confidence": 0.95,
            "summary": f"{len(empty_responses)} model response(s) were empty during step execution.",
            "likely_area": "LlamaClient.send_to_main, SlotManager, llama.cpp slot health",
            "evidence": [
                f"model_response_raw at +{e.get('elapsed_s','?')}s "
                f"step={e.get('step_num')} iter={e.get('iteration')}: is_empty=True"
                + (f", slot_id={e['slot_id']}" if e.get("slot_id") is not None else "")
                + (f", error={e['error']}" if e.get("error") else "")
                for e in empty_responses[:5]
            ],
        })

    # ── 4. Model did not call a tool when tools were available ───────────
    # Per step+iteration: model_request had tools, model_response had no tool_call
    req_iter_keys = {
        (e.get("step_num"), e.get("iteration"))
        for e in model_requests
        if e.get("tools_available")
    }
    call_iter_keys = {
        (e.get("step_num"), e.get("iteration"))
        for e in tool_calls
    }
    # Iterations where tools were offered but nothing was parsed
    missing_calls = req_iter_keys - call_iter_keys
    # Filter to iterations where the model response was non-empty and had no <tool_call>
    no_call_responses = [
        e for e in model_responses
        if not e.get("is_empty")
        and not e.get("has_tool_call")
        and (e.get("step_num"), e.get("iteration")) in missing_calls
    ]
    if no_call_responses:
        findings.append({
            "id": _FM_MODEL_DID_NOT_CALL_TOOL,
            "severity": "high",
            "confidence": 0.8,
            "summary": (
                f"In {len(no_call_responses)} iteration(s), tools were available but the model "
                "did not emit a <tool_call> block."
            ),
            "likely_area": (
                "system prompt tool-call instructions, MainAgent.execute_step tools_section, "
                "tool prompt contract"
            ),
            "evidence": [
                f"model_response_raw at +{e.get('elapsed_s','?')}s "
                f"step={e.get('step_num')} iter={e.get('iteration')}: "
                f"no tool_call, response_len={e.get('response_len',0)}"
                for e in no_call_responses[:5]
            ],
        })

    # ── 5. Tool call in response but none parsed ──────────────────────────
    # model said <tool_call> but tool_call_parsed never fired
    resp_with_call = [e for e in model_responses if e.get("has_tool_call")]
    resp_keys_with_call = {(e.get("step_num"), e.get("iteration")) for e in resp_with_call}
    parsed_keys = {(e.get("step_num"), e.get("iteration")) for e in tool_calls}
    unmatched = resp_keys_with_call - parsed_keys
    if unmatched:
        findings.append({
            "id": _FM_TOOL_CALL_PARSE_FAILURE,
            "severity": "critical",
            "confidence": 0.9,
            "summary": (
                f"Model emitted <tool_call> in {len(unmatched)} iteration(s) "
                "but no tool call was parsed."
            ),
            "likely_area": "ToolRegistry.parse_tool_calls, tool JSON format",
            "evidence": [
                f"model_response_raw step={s} iter={i} has_tool_call=True → no tool_call_parsed"
                for s, i in sorted(unmatched)[:5]
            ],
        })

    # ── 6. Tool execution failures ────────────────────────────────────────
    failed_execs = [e for e in tool_execs if not e.get("ok")]
    if failed_execs:
        findings.append({
            "id": _FM_TOOL_EXECUTION_FAILED,
            "severity": "high",
            "confidence": 0.95,
            "summary": f"{len(failed_execs)} tool execution(s) failed.",
            "likely_area": "tool implementation, ToolRegistry.execute, environment/filesystem",
            "evidence": [
                f"tool_executed at +{e.get('elapsed_s','?')}s "
                f"tool={e.get('tool')} ok=False error={e.get('error')}"
                for e in failed_execs[:5]
            ],
        })

    # ── 7. Tool result not used (tools ran but step result is empty) ───────
    if tool_execs:
        successful_execs = [e for e in tool_execs if e.get("ok")]
        if successful_execs:
            empty_step_results = [
                e for e in step_completes
                if not (e.get("result_preview") or "").strip()
                or (e.get("result_preview") or "").strip().startswith(_STEP_RESULT_ERROR_PREFIX)
                or (e.get("result_preview") or "").strip() == _STEP_RESULT_EMPTY_SENTINEL
            ]
            if empty_step_results:
                findings.append({
                    "id": _FM_TOOL_RESULT_NOT_USED,
                    "severity": "critical",
                    "confidence": 0.85,
                    "summary": (
                        "Tool(s) executed successfully but step completion had empty/generic result, "
                        "suggesting the model ignored tool output."
                    ),
                    "likely_area": (
                        "MainAgent.execute_step synthesis directive, "
                        "_render_tool_result_answer, _MAX_TOOL_PROMPT_TEXT_CHARS"
                    ),
                    "evidence": [
                        f"step_complete step={e.get('step_num')} result_preview={e.get('result_preview')!r}"
                        for e in empty_step_results[:3]
                    ],
                })

    # ── 8. Steps completed with empty result ─────────────────────────────
    empty_steps = [
        e for e in step_completes
        if not (e.get("result_preview") or "").strip()
    ]
    if empty_steps:
        findings.append({
            "id": _FM_STEP_EMPTY_RESULT,
            "severity": "high",
            "confidence": 0.9,
            "summary": f"{len(empty_steps)} step(s) completed with no result text.",
            "likely_area": "MainAgent.execute_step, model response synthesis",
            "evidence": [
                f"step_complete at +{e.get('elapsed_s','?')}s step={e.get('step_num')} status={e.get('status')}"
                for e in empty_steps[:5]
            ],
        })

    # ── 9. Checkpoint restore loop ────────────────────────────────────────
    if len(checkpoint_restores) >= 3:
        restore_steps = [e.get("restored_to_step") for e in checkpoint_restores]
        findings.append({
            "id": _FM_CHECKPOINT_RESTORE_LOOP,
            "severity": "high",
            "confidence": 0.85,
            "summary": (
                f"Checkpoint was restored {len(checkpoint_restores)} times, "
                "indicating a recovery loop."
            ),
            "likely_area": "SafetySupervisor, AgenticLoop._apply_intervention, CheckpointManager",
            "evidence": [
                f"checkpoint_restored at +{e.get('elapsed_s','?')}s "
                f"restored_to_step={e.get('restored_to_step')} reason={e.get('reason')}"
                for e in checkpoint_restores[:5]
            ],
        })

    # ── 10. Supervisor retry loop ─────────────────────────────────────────
    retry_interventions = [
        e for e in supervisor_events
        if e.get("action") in ("nudge", "retry_step", "reset_context")
    ]
    if len(retry_interventions) >= 3:
        findings.append({
            "id": _FM_SUPERVISOR_RETRY_LOOP,
            "severity": "high",
            "confidence": 0.85,
            "summary": (
                f"Supervisor intervened {len(retry_interventions)} times with retries/nudges, "
                "indicating a stall loop."
            ),
            "likely_area": "SafetySupervisor thresholds, MainAgent.execute_step, LlamaClient",
            "evidence": [
                f"supervisor_intervention at +{e.get('elapsed_s','?')}s "
                f"step={e.get('step_num')} action={e.get('action')}"
                for e in retry_interventions[:5]
            ],
        })

    # ── 11. Finalizer dropped results (tools OK, run unverified) ──────────
    # Two sub-cases:
    #   a) Tools executed successfully but run is unverified (tool result dropped)
    #   b) Model produced non-empty responses but steps came out empty / unverified
    verified = final.get("completion_verified")
    if verified is False:
        successful_tool_execs = [e for e in tool_execs if e.get("ok")]
        non_empty_responses = [e for e in model_responses if not e.get("is_empty")]
        empty_step_results = [
            e for e in step_completes
            if not (e.get("result_preview") or "").strip()
        ]
        if successful_tool_execs and not empty_responses:
            # Tools ran and model wasn't broken — result was clearly dropped.
            findings.append({
                "id": _FM_FINALIZER_DROPPED_RESULTS,
                "severity": "high",
                "confidence": 0.75,
                "summary": (
                    "Run completed unverified despite successful tool executions, "
                    "suggesting finalizer discarded useful results."
                ),
                "likely_area": (
                    "_derive_summary_from_steps, AgenticLoop._build_result, "
                    "MainAgent finalizer input handling"
                ),
                "evidence": [
                    f"{len(successful_tool_execs)} successful tool execution(s) recorded",
                    f"run_complete: completion_verified={verified}",
                    f"total_steps={final.get('total_steps')}",
                ],
            })
        elif non_empty_responses and empty_step_results:
            # Model produced output but it never made it to the step result.
            findings.append({
                "id": _FM_FINALIZER_DROPPED_RESULTS,
                "severity": "high",
                "confidence": 0.7,
                "summary": (
                    f"Model produced {len(non_empty_responses)} non-empty response(s) but "
                    f"{len(empty_step_results)} step(s) completed with empty results and run "
                    "is unverified, suggesting model output was not carried through."
                ),
                "likely_area": (
                    "MainAgent.execute_step result capture, _derive_summary_from_steps, "
                    "AgenticLoop._build_result"
                ),
                "evidence": [
                    f"{len(non_empty_responses)} non-empty model responses",
                    f"{len(empty_step_results)} step(s) with empty result_preview",
                    f"run_complete: completion_verified={verified}",
                ],
            })

    return findings
