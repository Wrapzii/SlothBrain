"""Tests for DiagnosticRecorder."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from backend.core.diagnostic_recorder import DiagnosticRecorder, _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recorder(tmp_path: Path, enabled: bool = True) -> DiagnosticRecorder:
    return DiagnosticRecorder(output_dir=tmp_path / "diagnostics" / "runs", enabled=enabled)


# ---------------------------------------------------------------------------
# DiagnosticRecorder – basic lifecycle
# ---------------------------------------------------------------------------

def test_recorder_disabled_is_noop(tmp_path):
    rec = _make_recorder(tmp_path, enabled=False)
    rec.start_run("r1", task="test task", metadata={})
    rec.record("r1", "planning_request")
    rec.finish_run("r1", {"completed": True})
    # No files should have been written.
    assert not (tmp_path / "diagnostics" / "runs").exists()


def test_recorder_writes_bundle(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="do something", metadata={"slot_id": 1})
    rec.record("r1", "custom_event", foo="bar")
    rec.finish_run("r1", {"ok": True})

    bundle_path = tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json"
    assert bundle_path.exists(), "bundle.json was not created"

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "1"
    assert bundle["run_id"] == "r1"
    assert bundle["task"] == "do something"
    assert bundle["run_metadata"] == {"slot_id": 1}
    assert bundle["final_result"] == {"ok": True}

    events = bundle["events"]
    # First event is auto-emitted run_start, second is our custom_event.
    types = [e["type"] for e in events]
    assert "run_start" in types
    assert "custom_event" in types


def test_recorder_events_are_sequential(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="seq", metadata={})
    rec.record("r1", "alpha")
    rec.record("r1", "beta")
    rec.record("r1", "gamma")
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    seqs = [e["seq"] for e in bundle["events"]]
    assert seqs == sorted(seqs), "events are not in sequential order"
    assert seqs == list(range(1, len(seqs) + 1)), "seq numbers must start at 1 and be contiguous"


def test_recorder_elapsed_non_negative(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="timing", metadata={})
    time.sleep(0.01)
    rec.record("r1", "midpoint")
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    for event in bundle["events"]:
        assert event["elapsed_s"] >= 0, f"elapsed_s negative for {event}"
    assert bundle["duration_seconds"] > 0


def test_recorder_unknown_run_id_is_noop(tmp_path):
    rec = _make_recorder(tmp_path)
    # Record without starting – should silently do nothing.
    rec.record("unknown-run", "event")
    rec.finish_run("unknown-run", {})
    assert not (tmp_path / "diagnostics" / "runs").exists()


def test_recorder_multiple_concurrent_runs(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("run-a", task="task a", metadata={"slot": 1})
    rec.start_run("run-b", task="task b", metadata={"slot": 2})

    rec.record("run-a", "a_event")
    rec.record("run-b", "b_event")
    rec.record("run-a", "a_event_2")

    rec.finish_run("run-a", {"result": "a"})
    rec.finish_run("run-b", {"result": "b"})

    bundle_a = json.loads((tmp_path / "diagnostics" / "runs" / "run-a" / "bundle.json").read_text())
    bundle_b = json.loads((tmp_path / "diagnostics" / "runs" / "run-b" / "bundle.json").read_text())

    a_types = [e["type"] for e in bundle_a["events"]]
    b_types = [e["type"] for e in bundle_b["events"]]
    assert "a_event" in a_types
    assert "a_event_2" in a_types
    assert "b_event" not in a_types
    assert "b_event" in b_types


# ---------------------------------------------------------------------------
# Incremental writes (crash safety)
# ---------------------------------------------------------------------------

def test_partial_json_written_on_start(tmp_path):
    """partial.json must exist before finish_run is called."""
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="crash-safe", metadata={})
    partial_path = tmp_path / "diagnostics" / "runs" / "r1" / "partial.json"
    assert partial_path.exists(), "partial.json was not written by start_run"
    meta = json.loads(partial_path.read_text())
    assert meta["task"] == "crash-safe"
    assert meta["status"] == "in_progress"


def test_events_jsonl_written_incrementally(tmp_path):
    """Each record() call must flush a line to events.jsonl immediately."""
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="incremental", metadata={})
    rec.record("r1", "event_one", x=1)
    rec.record("r1", "event_two", x=2)

    jsonl_path = tmp_path / "diagnostics" / "runs" / "r1" / "events.jsonl"
    assert jsonl_path.exists(), "events.jsonl was not written"
    lines = [l.strip() for l in jsonl_path.read_text().splitlines() if l.strip()]
    # run_start + event_one + event_two = 3 lines
    assert len(lines) == 3

    types = [json.loads(l)["type"] for l in lines]
    assert types == ["run_start", "event_one", "event_two"]

    # Finish and check bundle exists while partial.json is removed
    rec.finish_run("r1", {"ok": True})
    bundle_path = tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json"
    partial_path = tmp_path / "diagnostics" / "runs" / "r1" / "partial.json"
    assert bundle_path.exists()
    assert not partial_path.exists(), "partial.json should be removed after finish_run"


def test_get_bundle_assembles_partial_run(tmp_path):
    """get_bundle must return event data for runs without bundle.json."""
    rec = _make_recorder(tmp_path)
    rec.start_run("crash-run", task="crash before finish", metadata={})
    rec.record("crash-run", "step_start", step_num=1)
    rec.record("crash-run", "model_request", step_num=1, iteration=1, slot_id=None,
               prompt_len=100, prompt_preview="...", tools_available=[])
    # Simulate crash: do NOT call finish_run

    bundle = rec.get_bundle("crash-run")
    assert bundle is not None
    assert bundle["status"] == "partial"
    assert bundle["task"] == "crash before finish"
    types = [e["type"] for e in bundle["events"]]
    assert "run_start" in types
    assert "step_start" in types
    assert "model_request" in types


def test_list_runs_includes_partial_runs(tmp_path):
    """list_runs must surface partial/crashed runs alongside completed ones."""
    rec = _make_recorder(tmp_path)
    # Complete run
    rec.start_run("done-run", task="done", metadata={})
    rec.record("done-run", "step_start", step_num=1)
    rec.finish_run("done-run", {"completion_verified": True})
    # Partial run (no finish_run)
    rec.start_run("crash-run", task="crashed", metadata={})
    rec.record("crash-run", "step_start", step_num=1)

    runs = rec.list_runs()
    statuses = {r["run_id"]: r["status"] for r in runs}
    assert statuses.get("done-run") == "complete"
    assert statuses.get("crash-run") == "partial"


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def test_relative_output_dir_is_anchored_to_project_root(tmp_path):
    """A relative output_dir must resolve inside the project root."""
    # "diagnostics/runs" is the default: must be valid (inside project root)
    rec = DiagnosticRecorder(output_dir="diagnostics/runs", enabled=True)
    assert rec.enabled
    assert rec._output_dir.is_relative_to(_PROJECT_ROOT)


def test_relative_escape_path_is_rejected():
    """A relative path that escapes the project root must be rejected."""
    rec = DiagnosticRecorder(output_dir="../../tmp/evil", enabled=True)
    # Invalid path should cause diagnostics to be disabled.
    assert not rec.enabled


def test_absolute_output_dir_is_accepted(tmp_path):
    """An absolute path (e.g. in /tmp from tests) must be accepted."""
    rec = DiagnosticRecorder(output_dir=tmp_path, enabled=True)
    assert rec.enabled
    assert rec._output_dir == tmp_path.resolve()


# ---------------------------------------------------------------------------
# Typed convenience recorders
# ---------------------------------------------------------------------------

def test_record_planning_request(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="plan test", metadata={})
    rec.record_planning_request("r1", prompt="system: you are...\n\nTask: plan test\nassistant:")
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    events = {e["type"]: e for e in bundle["events"]}
    assert "planning_request" in events
    assert events["planning_request"]["prompt_len"] > 0
    assert "prompt_preview" in events["planning_request"]


def test_record_planning_response(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="plan test", metadata={})
    rec.record_planning_response("r1", response='{"approach": "do it", "steps": ["step 1"]}')
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    events = {e["type"]: e for e in bundle["events"]}
    assert "planning_response" in events
    assert events["planning_response"]["is_empty"] is False
    assert events["planning_response"]["error"] is None


def test_record_planning_response_empty(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="plan test", metadata={})
    rec.record_planning_response("r1", response="   ", error="TimeoutError")
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    events = {e["type"]: e for e in bundle["events"]}
    assert events["planning_response"]["is_empty"] is True
    assert events["planning_response"]["error"] == "TimeoutError"


def test_record_model_request_and_response(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="execute step", metadata={})
    rec.record_model_request(
        "r1",
        step_num=1,
        iteration=1,
        prompt="system: assistant\nCurrent step: do something\nassistant:",
        tools_available=["web_fetch", "web_search"],
    )
    rec.record_model_response(
        "r1",
        step_num=1,
        iteration=1,
        response='<tool_call>{"tool": "web_fetch", "args": {"url": "https://example.com"}}</tool_call>',
    )
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    model_req = next(e for e in bundle["events"] if e["type"] == "model_request")
    model_resp = next(e for e in bundle["events"] if e["type"] == "model_response_raw")

    assert model_req["step_num"] == 1
    assert model_req["iteration"] == 1
    assert "web_fetch" in model_req["tools_available"]
    assert model_resp["has_tool_call"] is True
    assert model_resp["is_empty"] is False


def test_record_tool_call_parsed(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="tool test", metadata={})
    rec.record_tool_call_parsed(
        "r1",
        step_num=1,
        iteration=1,
        tool_name="web_fetch",
        args={"url": "https://example.com", "timeout": 10},
    )
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    tc = next(e for e in bundle["events"] if e["type"] == "tool_call_parsed")
    assert tc["tool"] == "web_fetch"
    assert "url" in tc["args"]


def test_record_tool_executed_success(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="exec test", metadata={})
    rec.record_tool_executed(
        "r1",
        step_num=1,
        tool_name="web_fetch",
        ok=True,
        output={"html": "<html>hello</html>"},
    )
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    te = next(e for e in bundle["events"] if e["type"] == "tool_executed")
    assert te["ok"] is True
    assert te["error"] is None
    assert len(te["output_preview"]) > 0


def test_record_tool_executed_failure(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="fail test", metadata={})
    rec.record_tool_executed(
        "r1",
        step_num=1,
        tool_name="bad_tool",
        ok=False,
        error="Connection refused",
    )
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    te = next(e for e in bundle["events"] if e["type"] == "tool_executed")
    assert te["ok"] is False
    assert te["error"] == "Connection refused"


# ---------------------------------------------------------------------------
# list_runs / get_bundle
# ---------------------------------------------------------------------------

def test_list_runs_empty(tmp_path):
    rec = _make_recorder(tmp_path)
    assert rec.list_runs() == []


def test_list_runs_returns_completed(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("run-1", task="task one", metadata={})
    rec.finish_run("run-1", {"completion_verified": True, "total_steps": 3})

    runs = rec.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run-1"
    assert runs[0]["task"] == "task one"
    assert runs[0]["completion_verified"] is True
    assert runs[0]["total_steps"] == 3
    assert runs[0]["status"] == "complete"


def test_get_bundle_returns_full_data(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("run-x", task="get me", metadata={"slot_id": 2})
    rec.record("run-x", "check")
    rec.finish_run("run-x", {"result": "done"})

    bundle = rec.get_bundle("run-x")
    assert bundle is not None
    assert bundle["run_id"] == "run-x"
    assert bundle["run_metadata"]["slot_id"] == 2
    assert bundle["final_result"]["result"] == "done"


def test_get_bundle_returns_none_for_unknown(tmp_path):
    rec = _make_recorder(tmp_path)
    assert rec.get_bundle("does-not-exist") is None


# ---------------------------------------------------------------------------
# Large response / arg capping
# ---------------------------------------------------------------------------

def test_response_is_capped(tmp_path):
    from backend.core.diagnostic_recorder import _MAX_RESPONSE_CHARS
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="cap", metadata={})
    big_response = "x" * (_MAX_RESPONSE_CHARS + 1000)
    rec.record_model_response("r1", step_num=1, iteration=1, response=big_response)
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    resp_event = next(e for e in bundle["events"] if e["type"] == "model_response_raw")
    assert len(resp_event["response"]) <= _MAX_RESPONSE_CHARS
    assert resp_event["response_len"] == len(big_response)


# ---------------------------------------------------------------------------
# DiagnosticRecorder – basic lifecycle
# ---------------------------------------------------------------------------

def test_recorder_disabled_is_noop(tmp_path):
    rec = _make_recorder(tmp_path, enabled=False)
    rec.start_run("r1", task="test task", metadata={})
    rec.record("r1", "planning_request")
    rec.finish_run("r1", {"completed": True})
    # No files should have been written.
    assert not (tmp_path / "diagnostics" / "runs").exists()


def test_recorder_writes_bundle(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="do something", metadata={"slot_id": 1})
    rec.record("r1", "custom_event", foo="bar")
    rec.finish_run("r1", {"ok": True})

    bundle_path = tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json"
    assert bundle_path.exists(), "bundle.json was not created"

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "1"
    assert bundle["run_id"] == "r1"
    assert bundle["task"] == "do something"
    assert bundle["run_metadata"] == {"slot_id": 1}
    assert bundle["final_result"] == {"ok": True}

    events = bundle["events"]
    # First event is auto-emitted run_start, second is our custom_event.
    types = [e["type"] for e in events]
    assert "run_start" in types
    assert "custom_event" in types


def test_recorder_events_are_sequential(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="seq", metadata={})
    rec.record("r1", "alpha")
    rec.record("r1", "beta")
    rec.record("r1", "gamma")
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    seqs = [e["seq"] for e in bundle["events"]]
    assert seqs == sorted(seqs), "events are not in sequential order"
    assert seqs == list(range(1, len(seqs) + 1)), "seq numbers must start at 1 and be contiguous"


def test_recorder_elapsed_non_negative(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="timing", metadata={})
    time.sleep(0.01)
    rec.record("r1", "midpoint")
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    for event in bundle["events"]:
        assert event["elapsed_s"] >= 0, f"elapsed_s negative for {event}"
    assert bundle["duration_seconds"] > 0


def test_recorder_unknown_run_id_is_noop(tmp_path):
    rec = _make_recorder(tmp_path)
    # Record without starting – should silently do nothing.
    rec.record("unknown-run", "event")
    rec.finish_run("unknown-run", {})
    assert not (tmp_path / "diagnostics" / "runs").exists()


def test_recorder_multiple_concurrent_runs(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("run-a", task="task a", metadata={"slot": 1})
    rec.start_run("run-b", task="task b", metadata={"slot": 2})

    rec.record("run-a", "a_event")
    rec.record("run-b", "b_event")
    rec.record("run-a", "a_event_2")

    rec.finish_run("run-a", {"result": "a"})
    rec.finish_run("run-b", {"result": "b"})

    bundle_a = json.loads((tmp_path / "diagnostics" / "runs" / "run-a" / "bundle.json").read_text())
    bundle_b = json.loads((tmp_path / "diagnostics" / "runs" / "run-b" / "bundle.json").read_text())

    a_types = [e["type"] for e in bundle_a["events"]]
    b_types = [e["type"] for e in bundle_b["events"]]
    assert "a_event" in a_types
    assert "a_event_2" in a_types
    assert "b_event" not in a_types
    assert "b_event" in b_types


# ---------------------------------------------------------------------------
# Typed convenience recorders
# ---------------------------------------------------------------------------

def test_record_planning_request(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="plan test", metadata={})
    rec.record_planning_request("r1", prompt="system: you are...\n\nTask: plan test\nassistant:")
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    events = {e["type"]: e for e in bundle["events"]}
    assert "planning_request" in events
    assert events["planning_request"]["prompt_len"] > 0
    assert "prompt_preview" in events["planning_request"]


def test_record_planning_response(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="plan test", metadata={})
    rec.record_planning_response("r1", response='{"approach": "do it", "steps": ["step 1"]}')
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    events = {e["type"]: e for e in bundle["events"]}
    assert "planning_response" in events
    assert events["planning_response"]["is_empty"] is False
    assert events["planning_response"]["error"] is None


def test_record_planning_response_empty(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="plan test", metadata={})
    rec.record_planning_response("r1", response="   ", error="TimeoutError")
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    events = {e["type"]: e for e in bundle["events"]}
    assert events["planning_response"]["is_empty"] is True
    assert events["planning_response"]["error"] == "TimeoutError"


def test_record_model_request_and_response(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="execute step", metadata={})
    rec.record_model_request(
        "r1",
        step_num=1,
        iteration=1,
        prompt="system: assistant\nCurrent step: do something\nassistant:",
        tools_available=["web_fetch", "web_search"],
    )
    rec.record_model_response(
        "r1",
        step_num=1,
        iteration=1,
        response='<tool_call>{"tool": "web_fetch", "args": {"url": "https://example.com"}}</tool_call>',
    )
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    model_req = next(e for e in bundle["events"] if e["type"] == "model_request")
    model_resp = next(e for e in bundle["events"] if e["type"] == "model_response_raw")

    assert model_req["step_num"] == 1
    assert model_req["iteration"] == 1
    assert "web_fetch" in model_req["tools_available"]
    assert model_resp["has_tool_call"] is True
    assert model_resp["is_empty"] is False


def test_record_tool_call_parsed(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="tool test", metadata={})
    rec.record_tool_call_parsed(
        "r1",
        step_num=1,
        iteration=1,
        tool_name="web_fetch",
        args={"url": "https://example.com", "timeout": 10},
    )
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    tc = next(e for e in bundle["events"] if e["type"] == "tool_call_parsed")
    assert tc["tool"] == "web_fetch"
    assert "url" in tc["args"]


def test_record_tool_executed_success(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="exec test", metadata={})
    rec.record_tool_executed(
        "r1",
        step_num=1,
        tool_name="web_fetch",
        ok=True,
        output={"html": "<html>hello</html>"},
    )
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    te = next(e for e in bundle["events"] if e["type"] == "tool_executed")
    assert te["ok"] is True
    assert te["error"] is None
    assert len(te["output_preview"]) > 0


def test_record_tool_executed_failure(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="fail test", metadata={})
    rec.record_tool_executed(
        "r1",
        step_num=1,
        tool_name="bad_tool",
        ok=False,
        error="Connection refused",
    )
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    te = next(e for e in bundle["events"] if e["type"] == "tool_executed")
    assert te["ok"] is False
    assert te["error"] == "Connection refused"


# ---------------------------------------------------------------------------
# list_runs / get_bundle
# ---------------------------------------------------------------------------

def test_list_runs_empty(tmp_path):
    rec = _make_recorder(tmp_path)
    assert rec.list_runs() == []


def test_list_runs_returns_completed(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("run-1", task="task one", metadata={})
    rec.finish_run("run-1", {"completion_verified": True, "total_steps": 3})

    runs = rec.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run-1"
    assert runs[0]["task"] == "task one"
    assert runs[0]["completion_verified"] is True
    assert runs[0]["total_steps"] == 3


def test_get_bundle_returns_full_data(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.start_run("run-x", task="get me", metadata={"slot_id": 2})
    rec.record("run-x", "check")
    rec.finish_run("run-x", {"result": "done"})

    bundle = rec.get_bundle("run-x")
    assert bundle is not None
    assert bundle["run_id"] == "run-x"
    assert bundle["run_metadata"]["slot_id"] == 2
    assert bundle["final_result"]["result"] == "done"


def test_get_bundle_returns_none_for_unknown(tmp_path):
    rec = _make_recorder(tmp_path)
    assert rec.get_bundle("does-not-exist") is None


# ---------------------------------------------------------------------------
# Large response / arg capping
# ---------------------------------------------------------------------------

def test_response_is_capped(tmp_path):
    from backend.core.diagnostic_recorder import _MAX_RESPONSE_CHARS
    rec = _make_recorder(tmp_path)
    rec.start_run("r1", task="cap", metadata={})
    big_response = "x" * (_MAX_RESPONSE_CHARS + 1000)
    rec.record_model_response("r1", step_num=1, iteration=1, response=big_response)
    rec.finish_run("r1", {})

    bundle = json.loads((tmp_path / "diagnostics" / "runs" / "r1" / "bundle.json").read_text())
    resp_event = next(e for e in bundle["events"] if e["type"] == "model_response_raw")
    assert len(resp_event["response"]) <= _MAX_RESPONSE_CHARS
    assert resp_event["response_len"] == len(big_response)
