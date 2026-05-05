"""Tests for DiagnosticAnalyzer – failure-mode heuristics and output files."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core.diagnostic_analyzer import (
    DiagnosticAnalyzer,
    detect_failure_modes,
    _FM_PLANNING_EMPTY_RESPONSE,
    _FM_NO_STEPS_GENERATED,
    _FM_LLM_EMPTY_RESPONSE,
    _FM_MODEL_DID_NOT_CALL_TOOL,
    _FM_TOOL_CALL_PARSE_FAILURE,
    _FM_TOOL_EXECUTION_FAILED,
    _FM_TOOL_RESULT_NOT_USED,
    _FM_STEP_EMPTY_RESULT,
    _FM_CHECKPOINT_RESTORE_LOOP,
    _FM_SUPERVISOR_RETRY_LOOP,
    _FM_FINALIZER_DROPPED_RESULTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bundle(
    *,
    task: str = "test task",
    run_id: str = "test-run",
    events: list | None = None,
    final_result: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        "schema_version": "1",
        "run_id": run_id,
        "task": task,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:05+00:00",
        "duration_seconds": 5.0,
        "run_metadata": metadata or {"slot_id": 1, "max_steps": 5},
        "events": events or [],
        "final_result": final_result or {"completion_verified": True, "total_steps": 1},
    }


def _write_bundle(run_dir: Path, bundle: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _seq(events: list[dict]) -> list[dict]:
    """Tag events with seq, elapsed_s for a valid bundle."""
    for i, e in enumerate(events):
        e.setdefault("seq", i + 1)
        e.setdefault("elapsed_s", float(i) * 0.1)
    return events


# ---------------------------------------------------------------------------
# detect_failure_modes – unit tests per finding
# ---------------------------------------------------------------------------

def test_no_findings_on_clean_run():
    bundle = _make_bundle(
        events=_seq([
            {"type": "plan_parsed", "steps": ["step A"], "approach": "do it"},
            {"type": "model_request", "step_num": 1, "iteration": 1, "tools_available": [], "prompt_len": 100},
            {"type": "model_response_raw", "step_num": 1, "iteration": 1, "is_empty": False,
             "has_tool_call": False, "response_len": 50, "response": "Done.", "error": None},
            {"type": "step_complete", "step_num": 1, "status": "complete", "result_preview": "All done."},
        ]),
        final_result={"completion_verified": True, "total_steps": 1},
    )
    findings = detect_failure_modes(bundle)
    assert findings == []


def test_planning_empty_response():
    bundle = _make_bundle(events=_seq([
        {"type": "planning_response", "is_empty": True, "response": "", "response_len": 0, "error": None},
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_PLANNING_EMPTY_RESPONSE in ids
    f = next(f for f in findings if f["id"] == _FM_PLANNING_EMPTY_RESPONSE)
    assert f["severity"] == "critical"
    assert f["confidence"] >= 0.9


def test_no_steps_generated_bad_parse():
    bundle = _make_bundle(events=_seq([
        {"type": "planning_response", "is_empty": False, "response": "{}", "response_len": 2, "error": None},
        {"type": "plan_parsed", "steps": [], "approach": "", "source": "model"},
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_NO_STEPS_GENERATED in ids


def test_no_steps_generated_missing_plan_parsed():
    # planning_response emitted but no plan_parsed event
    bundle = _make_bundle(events=_seq([
        {"type": "planning_response", "is_empty": False, "response": "x", "response_len": 1, "error": None},
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_NO_STEPS_GENERATED in ids


def test_llm_empty_response_execution():
    bundle = _make_bundle(events=_seq([
        {"type": "model_request", "step_num": 1, "iteration": 1, "tools_available": [], "prompt_len": 100},
        {"type": "model_response_raw", "step_num": 1, "iteration": 1, "is_empty": True,
         "has_tool_call": False, "response_len": 0, "response": "", "error": None},
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_LLM_EMPTY_RESPONSE in ids


def test_model_did_not_call_tool():
    bundle = _make_bundle(events=_seq([
        {"type": "model_request", "step_num": 1, "iteration": 1,
         "tools_available": ["web_fetch"], "prompt_len": 200},
        {"type": "model_response_raw", "step_num": 1, "iteration": 1,
         "is_empty": False, "has_tool_call": False, "response_len": 80,
         "response": "I will search the web.", "error": None},
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_MODEL_DID_NOT_CALL_TOOL in ids
    f = next(f for f in findings if f["id"] == _FM_MODEL_DID_NOT_CALL_TOOL)
    assert f["severity"] == "high"


def test_tool_call_parse_failure():
    # Model said <tool_call> but no tool_call_parsed fired
    bundle = _make_bundle(events=_seq([
        {"type": "model_request", "step_num": 1, "iteration": 1,
         "tools_available": ["web_fetch"], "prompt_len": 200},
        {"type": "model_response_raw", "step_num": 1, "iteration": 1,
         "is_empty": False, "has_tool_call": True, "response_len": 120,
         "response": "<tool_call>{broken json}</tool_call>", "error": None},
        # No tool_call_parsed event
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_TOOL_CALL_PARSE_FAILURE in ids
    f = next(f for f in findings if f["id"] == _FM_TOOL_CALL_PARSE_FAILURE)
    assert f["severity"] == "critical"


def test_tool_call_parsed_present_no_parse_failure():
    bundle = _make_bundle(events=_seq([
        {"type": "model_request", "step_num": 1, "iteration": 1,
         "tools_available": ["web_fetch"], "prompt_len": 200},
        {"type": "model_response_raw", "step_num": 1, "iteration": 1,
         "is_empty": False, "has_tool_call": True, "response_len": 120,
         "response": '<tool_call>{"tool":"web_fetch"}</tool_call>', "error": None},
        {"type": "tool_call_parsed", "step_num": 1, "iteration": 1,
         "tool": "web_fetch", "args": {"url": "https://x.com"}},
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_TOOL_CALL_PARSE_FAILURE not in ids


def test_tool_execution_failed():
    bundle = _make_bundle(events=_seq([
        {"type": "tool_executed", "step_num": 1, "tool": "web_fetch", "ok": False,
         "error": "Connection refused", "output_preview": ""},
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_TOOL_EXECUTION_FAILED in ids
    f = next(f for f in findings if f["id"] == _FM_TOOL_EXECUTION_FAILED)
    assert "Connection refused" in str(f["evidence"])


def test_tool_result_not_used():
    bundle = _make_bundle(events=_seq([
        {"type": "tool_executed", "step_num": 1, "tool": "web_fetch", "ok": True,
         "output_preview": '{"html": "<html>results</html>"}', "error": None},
        {"type": "step_complete", "step_num": 1, "status": "complete",
         "result_preview": ""},  # empty
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_TOOL_RESULT_NOT_USED in ids


def test_step_empty_result():
    bundle = _make_bundle(events=_seq([
        {"type": "step_complete", "step_num": 1, "status": "complete", "result_preview": ""},
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_STEP_EMPTY_RESULT in ids


def test_checkpoint_restore_loop():
    bundle = _make_bundle(events=_seq([
        {"type": "checkpoint_restored", "restored_to_step": 1, "reason": "supervisor_reset_context"},
        {"type": "checkpoint_restored", "restored_to_step": 1, "reason": "supervisor_reset_context"},
        {"type": "checkpoint_restored", "restored_to_step": 1, "reason": "supervisor_reset_context"},
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_CHECKPOINT_RESTORE_LOOP in ids


def test_supervisor_retry_loop():
    bundle = _make_bundle(events=_seq([
        {"type": "supervisor_intervention", "step_num": 1, "action": "nudge", "message": "slow"},
        {"type": "supervisor_intervention", "step_num": 1, "action": "nudge", "message": "slow"},
        {"type": "supervisor_intervention", "step_num": 1, "action": "reset_context", "message": "stalled"},
    ]))
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_SUPERVISOR_RETRY_LOOP in ids


def test_finalizer_dropped_results():
    bundle = _make_bundle(
        events=_seq([
            {"type": "tool_executed", "step_num": 1, "tool": "web_fetch", "ok": True,
             "output_preview": '{"result": "data"}', "error": None},
        ]),
        final_result={"completion_verified": False, "total_steps": 1},
    )
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]
    assert _FM_FINALIZER_DROPPED_RESULTS in ids


# ---------------------------------------------------------------------------
# DiagnosticAnalyzer – output file generation
# ---------------------------------------------------------------------------

def test_analyzer_writes_all_output_files(tmp_path):
    run_dir = tmp_path / "run-1"
    bundle = _make_bundle(
        run_id="run-1",
        events=_seq([
            {"type": "model_request", "step_num": 1, "iteration": 1,
             "tools_available": ["web_fetch"], "prompt_len": 200},
            {"type": "model_response_raw", "step_num": 1, "iteration": 1,
             "is_empty": True, "has_tool_call": False, "response_len": 0,
             "response": "", "error": None},
        ]),
    )
    _write_bundle(run_dir, bundle)
    analyzer = DiagnosticAnalyzer(run_dir=run_dir)
    findings = analyzer.analyze()

    assert (run_dir / "analysis.json").exists(), "analysis.json not written"
    assert (run_dir / "summary.md").exists(), "summary.md not written"
    assert (run_dir / "model_review_prompt.md").exists(), "model_review_prompt.md not written"
    assert len(findings) > 0


def test_analyzer_analysis_json_structure(tmp_path):
    run_dir = tmp_path / "run-2"
    bundle = _make_bundle(run_id="run-2", events=_seq([
        {"type": "planning_response", "is_empty": True, "response": "",
         "response_len": 0, "error": "TimeoutError"},
    ]))
    _write_bundle(run_dir, bundle)
    DiagnosticAnalyzer(run_dir=run_dir).analyze()

    analysis = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["schema_version"] == "1"
    assert analysis["run_id"] == "run-2"
    assert analysis["finding_count"] >= 1
    assert isinstance(analysis["findings"], list)
    for f in analysis["findings"]:
        assert "id" in f
        assert "severity" in f
        assert "confidence" in f
        assert "summary" in f
        assert "likely_area" in f
        assert "evidence" in f


def test_analyzer_summary_md_contains_findings(tmp_path):
    run_dir = tmp_path / "run-3"
    bundle = _make_bundle(run_id="run-3", events=_seq([
        {"type": "model_response_raw", "step_num": 1, "iteration": 1,
         "is_empty": True, "has_tool_call": False, "response_len": 0,
         "response": "", "error": None},
    ]))
    _write_bundle(run_dir, bundle)
    DiagnosticAnalyzer(run_dir=run_dir).analyze()

    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "# Diagnostic Summary" in summary
    assert _FM_LLM_EMPTY_RESPONSE in summary


def test_analyzer_review_prompt_contains_required_sections(tmp_path):
    run_dir = tmp_path / "run-4"
    bundle = _make_bundle(run_id="run-4", task="search the web", events=_seq([
        {"type": "model_request", "step_num": 1, "iteration": 1,
         "tools_available": ["web_fetch"], "prompt_len": 200, "slot_id": 1},
        {"type": "model_response_raw", "step_num": 1, "iteration": 1,
         "is_empty": False, "has_tool_call": False, "response_len": 80,
         "response": "I will look that up.", "error": None, "slot_id": 1},
    ]))
    _write_bundle(run_dir, bundle)
    DiagnosticAnalyzer(run_dir=run_dir).analyze()

    prompt = (run_dir / "model_review_prompt.md").read_text(encoding="utf-8")
    assert "run_id:" in prompt
    assert "task:" in prompt
    assert "Most likely failure mode" in prompt
    assert "Patch plan" in prompt
    assert "Tests to add" in prompt
    assert "slot_id" in prompt


def test_analyzer_missing_bundle_returns_empty(tmp_path):
    run_dir = tmp_path / "missing-run"
    run_dir.mkdir()
    findings = DiagnosticAnalyzer(run_dir=run_dir).analyze()
    assert findings == []
    assert not (run_dir / "analysis.json").exists()


def test_get_review_prompt_via_analyzer(tmp_path):
    run_dir = tmp_path / "run-5"
    bundle = _make_bundle(run_id="run-5")
    _write_bundle(run_dir, bundle)
    analyzer = DiagnosticAnalyzer(run_dir=run_dir)
    analyzer.analyze()
    prompt = analyzer.get_review_prompt()
    assert prompt is not None
    assert len(prompt) > 100


# ---------------------------------------------------------------------------
# Regression: "known bad run" – model didn't call required tool
# ---------------------------------------------------------------------------

def test_known_bad_run_model_ignored_tools(tmp_path):
    """Regression: task requires tool use, model responds with prose only.

    Scenario
    --------
    - User task: "list everything on Desktop"
    - Planning: valid (3 steps emitted)
    - Model response: non-empty prose, no <tool_call> block
    - Step complete: empty result_preview
    - Run complete: verified=False

    Expected analyzer output
    ------------------------
    - MODEL_DID_NOT_CALL_REQUIRED_TOOL
    - STEP_COMPLETED_WITH_EMPTY_RESULT
    - FINALIZER_DROPPED_RESULTS
    """
    run_dir = tmp_path / "bad-run"
    bundle = _make_bundle(
        run_id="bad-run",
        task="list everything on Desktop",
        events=_seq([
            # Planning went fine
            {"type": "planning_request", "prompt_len": 400, "prompt_preview": ""},
            {"type": "planning_response", "is_empty": False,
             "response": '{"approach":"use file tool","steps":["list files","filter","report"]}',
             "response_len": 78, "error": None},
            {"type": "plan_parsed", "steps": ["list files", "filter results", "report"],
             "approach": "use file tool", "source": "model"},
            # Step 1: model ignores tool, responds with prose
            {"type": "step_start", "step_num": 1, "description": "list files"},
            {"type": "model_request", "step_num": 1, "iteration": 1,
             "tools_available": ["file", "shell"], "prompt_len": 512, "slot_id": 1},
            {"type": "model_response_raw", "step_num": 1, "iteration": 1,
             "is_empty": False, "has_tool_call": False,
             "response": "I will list the files on the Desktop shortly.",
             "response_len": 46, "error": None, "slot_id": 1},
            {"type": "step_complete", "step_num": 1, "status": "complete",
             "result_preview": "", "retries": 0},
            # Step 2: same – model ignores tool again
            {"type": "step_start", "step_num": 2, "description": "filter results"},
            {"type": "model_request", "step_num": 2, "iteration": 1,
             "tools_available": ["file", "shell"], "prompt_len": 620, "slot_id": 1},
            {"type": "model_response_raw", "step_num": 2, "iteration": 1,
             "is_empty": False, "has_tool_call": False,
             "response": "Here are the filtered results (placeholder).",
             "response_len": 44, "error": None, "slot_id": 1},
            {"type": "step_complete", "step_num": 2, "status": "complete",
             "result_preview": "", "retries": 0},
            # Run finished unverified
            {"type": "run_complete", "verified": False, "total_steps": 2},
        ]),
        final_result={"completion_verified": False, "total_steps": 2, "summary": ""},
    )
    _write_bundle(run_dir, bundle)
    findings = detect_failure_modes(bundle)
    ids = [f["id"] for f in findings]

    assert _FM_MODEL_DID_NOT_CALL_TOOL in ids, (
        f"Expected MODEL_DID_NOT_CALL_REQUIRED_TOOL but got: {ids}"
    )
    assert _FM_STEP_EMPTY_RESULT in ids, (
        f"Expected STEP_COMPLETED_WITH_EMPTY_RESULT but got: {ids}"
    )
    assert _FM_FINALIZER_DROPPED_RESULTS in ids, (
        f"Expected FINALIZER_DROPPED_RESULTS but got: {ids}"
    )

    # Also verify the analyzer writes all output files
    DiagnosticAnalyzer(run_dir=run_dir).analyze()
    assert (run_dir / "analysis.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "model_review_prompt.md").exists()

    # Summary must mention the key findings
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert _FM_MODEL_DID_NOT_CALL_TOOL in summary
    assert _FM_STEP_EMPTY_RESULT in summary
