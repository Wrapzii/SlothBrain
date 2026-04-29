"""Tests for vision grid module."""
from __future__ import annotations

import io
import pytest
from unittest.mock import patch

from backend.vision.grid import ScreenGrid, GridCell


def test_grid_cell_labels():
    grid = ScreenGrid(1920, 1080, cols=10, rows=8)
    assert grid.cell("A1").label == "A1"
    assert grid.cell("J8").label == "J8"
    assert grid.cell("a1").label == "A1"  # case insensitive


def test_grid_cell_coordinates():
    grid = ScreenGrid(1000, 800, cols=10, rows=8)
    # Each cell is 100×100 px
    a1 = grid.cell("A1")
    assert a1.x == 0
    assert a1.y == 0
    assert a1.w == 100
    assert a1.h == 100
    assert a1.center_x == 50
    assert a1.center_y == 50

    b2 = grid.cell("B2")
    assert b2.x == 100
    assert b2.y == 100


def test_grid_cell_not_found():
    grid = ScreenGrid(1920, 1080, cols=10, rows=8)
    with pytest.raises(KeyError):
        grid.cell("Z99")


def test_pixel_to_cell():
    grid = ScreenGrid(1000, 800, cols=10, rows=8)
    cell = grid.pixel_to_cell(50, 50)
    assert cell.label == "A1"

    cell2 = grid.pixel_to_cell(150, 150)
    assert cell2.label == "B2"


def test_all_cells_count():
    grid = ScreenGrid(1920, 1080, cols=10, rows=8)
    cells = list(grid.all_cells())
    assert len(cells) == 80
