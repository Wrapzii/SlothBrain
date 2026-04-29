"""Tests for the action executor parser."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from backend.vision.action_executor import (
    ClickAction,
    TypeAction,
    PressAction,
    ScrollAction,
    DragAction,
    _ClickAndType,
    parse_action_string,
)
from backend.vision.grid import ScreenGrid


@pytest.fixture
def grid():
    return ScreenGrid(1000, 800, cols=10, rows=8)


def test_parse_click(grid):
    action = parse_action_string("CLICK A1", grid)
    assert isinstance(action, ClickAction)
    assert action.button == "left"
    assert action.double is False
    assert action.x == grid.cell("A1").center_x


def test_parse_right_click(grid):
    action = parse_action_string("RIGHT_CLICK B2", grid)
    assert isinstance(action, ClickAction)
    assert action.button == "right"


def test_parse_double_click(grid):
    action = parse_action_string("DOUBLE_CLICK C3", grid)
    assert isinstance(action, ClickAction)
    assert action.double is True


def test_parse_type(grid):
    action = parse_action_string('TYPE "hello world"', grid)
    assert isinstance(action, TypeAction)
    assert action.text == "hello world"


def test_parse_click_and_type(grid):
    action = parse_action_string('CLICK_AND_TYPE A1 "some text"', grid)
    assert isinstance(action, _ClickAndType)
    assert action.type_.text == "some text"


def test_parse_press(grid):
    action = parse_action_string("PRESS ctrl+c", grid)
    assert isinstance(action, PressAction)
    assert action.keys == "ctrl+c"


def test_parse_scroll(grid):
    action = parse_action_string("SCROLL D4 DOWN 5", grid)
    assert isinstance(action, ScrollAction)
    assert action.direction == "down"
    assert action.clicks == 5


def test_parse_drag(grid):
    action = parse_action_string("DRAG A1 J8", grid)
    assert isinstance(action, DragAction)
    src = grid.cell("A1")
    dst = grid.cell("J8")
    assert action.x1 == src.center_x
    assert action.x2 == dst.center_x


def test_parse_screenshot_returns_none(grid):
    assert parse_action_string("SCREENSHOT", grid) is None


def test_parse_done_returns_none(grid):
    assert parse_action_string("DONE", grid) is None


def test_parse_unknown_raises(grid):
    with pytest.raises(ValueError):
        parse_action_string("UNKNOWN_CMD A1", grid)


def test_parse_invalid_cell_raises(grid):
    with pytest.raises(KeyError):
        parse_action_string("CLICK Z99", grid)
