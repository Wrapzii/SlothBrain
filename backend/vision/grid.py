"""Grid-labelling: divide the screen into a named cell matrix.

Each cell is named by column letter (A–Z) and row number (1–N).
A 10×8 grid on 1920×1080 gives 192×135 px cells, which is fine enough
for the model to target UI elements reliably.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterator

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class GridCell:
    col: str          # "A", "B", …
    row: int          # 1, 2, …
    x: int            # pixel x of cell top-left
    y: int            # pixel y of cell top-left
    w: int            # cell width in pixels
    h: int            # cell height in pixels

    @property
    def label(self) -> str:
        return f"{self.col}{self.row}"

    @property
    def center_x(self) -> int:
        return self.x + self.w // 2

    @property
    def center_y(self) -> int:
        return self.y + self.h // 2


class ScreenGrid:
    """Divides a screen of given dimensions into a cols×rows grid."""

    def __init__(self, width: int, height: int, cols: int = 10, rows: int = 8) -> None:
        self.width = width
        self.height = height
        self.cols = cols
        self.rows = rows
        self._cell_w = width // cols
        self._cell_h = height // rows
        self._cells: dict[str, GridCell] = {}
        self._build()

    def _build(self) -> None:
        for ci in range(self.cols):
            for ri in range(self.rows):
                col_letter = chr(ord("A") + ci)
                row_num = ri + 1
                label = f"{col_letter}{row_num}"
                self._cells[label] = GridCell(
                    col=col_letter,
                    row=row_num,
                    x=ci * self._cell_w,
                    y=ri * self._cell_h,
                    w=self._cell_w,
                    h=self._cell_h,
                )

    def cell(self, label: str) -> GridCell:
        label = label.strip().upper()
        if label not in self._cells:
            raise KeyError(f"Grid cell {label!r} not found (grid is {self.cols}×{self.rows})")
        return self._cells[label]

    def all_cells(self) -> Iterator[GridCell]:
        for ci in range(self.cols):
            for ri in range(self.rows):
                label = f"{chr(ord('A') + ci)}{ri + 1}"
                yield self._cells[label]

    def pixel_to_cell(self, px: int, py: int) -> GridCell:
        ci = min(px // self._cell_w, self.cols - 1)
        ri = min(py // self._cell_h, self.rows - 1)
        label = f"{chr(ord('A') + ci)}{ri + 1}"
        return self._cells[label]

    def annotate_image(self, png_bytes: bytes) -> bytes:
        """Return PNG bytes with the grid overlay drawn on top."""
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Probe common font paths across Linux, macOS, Windows; fall back to default
        _FONT_CANDIDATES = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",       # Debian/Ubuntu
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",                # Fedora/RHEL
            "/System/Library/Fonts/Helvetica.ttc",                         # macOS
            "C:/Windows/Fonts/arialbd.ttf",                                # Windows
        ]
        font = ImageFont.load_default()
        for _fp in _FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(_fp, 11)
                break
            except Exception:
                continue

        # Draw grid lines
        for ci in range(self.cols + 1):
            x = ci * self._cell_w
            draw.line([(x, 0), (x, self.height)], fill=(80, 220, 255, 80), width=1)
        for ri in range(self.rows + 1):
            y = ri * self._cell_h
            draw.line([(0, y), (self.width, y)], fill=(80, 220, 255, 80), width=1)

        # Draw cell labels
        for cell in self.all_cells():
            draw.text(
                (cell.x + 3, cell.y + 2),
                cell.label,
                fill=(80, 220, 255, 200),
                font=font,
            )

        composed = Image.alpha_composite(img, overlay).convert("RGB")
        buf = io.BytesIO()
        composed.save(buf, format="PNG")
        return buf.getvalue()

    def extract_cell_image(self, png_bytes: bytes, label: str) -> bytes:
        """Return PNG bytes of a single grid cell."""
        img = Image.open(io.BytesIO(png_bytes))
        cell = self.cell(label)
        region = img.crop((cell.x, cell.y, cell.x + cell.w, cell.y + cell.h))
        buf = io.BytesIO()
        region.save(buf, format="PNG")
        return buf.getvalue()
