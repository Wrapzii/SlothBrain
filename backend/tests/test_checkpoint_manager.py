"""Tests for CheckpointManager."""
from __future__ import annotations

import pytest
from backend.core.checkpoint_manager import CheckpointManager, TaskCheckpoint


def _make_manager() -> CheckpointManager:
    return CheckpointManager(max_checkpoints_per_run=5)


# ---------------------------------------------------------------------------
# Save / restore
# ---------------------------------------------------------------------------

def test_save_returns_checkpoint():
    mgr = _make_manager()
    cp = mgr.save(
        run_id="r1",
        task="build a website",
        step_num=1,
        step_descriptions=["step A", "step B"],
        context=[],
        executed_steps=[],
    )
    assert isinstance(cp, TaskCheckpoint)
    assert cp.task == "build a website"
    assert cp.step_num == 1
    assert cp.step_descriptions == ["step A", "step B"]


def test_restore_last_returns_most_recent():
    mgr = _make_manager()
    mgr.save("r1", "task", 1, ["A", "B"], [], [])
    mgr.save("r1", "task", 2, ["A", "B"], ["ctx1"], [{"step_num": 1}])
    cp = mgr.restore_last("r1")
    assert cp is not None
    assert cp.step_num == 2
    assert cp.context == ["ctx1"]


def test_restore_last_returns_none_for_unknown_run():
    mgr = _make_manager()
    assert mgr.restore_last("unknown") is None


def test_restore_step_specific():
    mgr = _make_manager()
    mgr.save("r1", "task", 1, ["A"], [], [])
    mgr.save("r1", "task", 3, ["A"], ["c1", "c2"], [])
    cp = mgr.restore_step("r1", 1)
    assert cp is not None
    assert cp.step_num == 1
    assert cp.context == []


def test_restore_step_missing():
    mgr = _make_manager()
    mgr.save("r1", "task", 1, ["A"], [], [])
    assert mgr.restore_step("r1", 99) is None


# ---------------------------------------------------------------------------
# Deep-copy isolation
# ---------------------------------------------------------------------------

def test_checkpoint_context_is_isolated():
    """Mutating the original list must not affect the checkpoint."""
    mgr = _make_manager()
    context = ["step1"]
    mgr.save("r1", "task", 1, ["A"], context, [])
    context.append("step2")  # mutate original
    cp = mgr.restore_last("r1")
    assert cp.context == ["step1"]  # checkpoint unaffected


def test_checkpoint_step_descriptions_isolated():
    mgr = _make_manager()
    steps = ["A", "B"]
    mgr.save("r1", "task", 1, steps, [], [])
    steps.append("C")  # mutate original
    cp = mgr.restore_last("r1")
    assert cp.step_descriptions == ["A", "B"]


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

def test_evicts_oldest_when_over_cap():
    mgr = CheckpointManager(max_checkpoints_per_run=3)
    for i in range(1, 6):  # save steps 1-5
        mgr.save("r1", "task", i, ["A"], [], [])
    keys = mgr.list_checkpoints("r1")
    assert len(keys) == 3
    assert min(keys) >= 3  # oldest checkpoints (1, 2) evicted


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

def test_clear_removes_all_checkpoints():
    mgr = _make_manager()
    mgr.save("r1", "task", 1, [], [], [])
    mgr.save("r1", "task", 2, [], [], [])
    mgr.clear("r1")
    assert mgr.restore_last("r1") is None
    assert mgr.list_checkpoints("r1") == []


def test_clear_does_not_affect_other_runs():
    mgr = _make_manager()
    mgr.save("r1", "task", 1, [], [], [])
    mgr.save("r2", "task", 1, [], [], [])
    mgr.clear("r1")
    assert mgr.restore_last("r2") is not None


# ---------------------------------------------------------------------------
# list_checkpoints
# ---------------------------------------------------------------------------

def test_list_checkpoints_sorted():
    mgr = _make_manager()
    for i in [3, 1, 4, 2]:
        mgr.save("r1", "task", i, [], [], [])
    assert mgr.list_checkpoints("r1") == [1, 2, 3, 4]


def test_list_checkpoints_empty_run():
    mgr = _make_manager()
    assert mgr.list_checkpoints("nonexistent") == []


# ---------------------------------------------------------------------------
# TaskCheckpoint.to_dict
# ---------------------------------------------------------------------------

def test_checkpoint_to_dict():
    cp = TaskCheckpoint(
        task="t",
        step_num=2,
        step_descriptions=["A"],
        context=["ctx"],
        executed_steps=[{"step_num": 1}],
    )
    d = cp.to_dict()
    assert d["task"] == "t"
    assert d["step_num"] == 2
    assert d["context"] == ["ctx"]
    assert isinstance(d["timestamp"], float)
