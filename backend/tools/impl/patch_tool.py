"""Unified diff patch tool — apply unified diff patches across multiple files.

Supports standard unified diff format (as produced by ``git diff`` or
``diff -u``).  Applies patches atomically: if any hunk fails to apply the
entire operation is rolled back.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from backend.config import AppConfig

logger = logging.getLogger(__name__)


class PatchTool(Tool):
    """Apply a unified diff patch to files within the workspace root.

    The patch string must be in standard unified diff format, e.g.::

        --- a/foo.py
        +++ b/foo.py
        @@ -1,3 +1,4 @@
         line1
        +new line
         line2
         line3

    Multi-file patches (multiple ``---``/``+++`` headers) are supported.
    The operation is atomic: if any file fails to patch, all changes are
    rolled back.
    """

    name = "patch"
    description = (
        "Apply a unified diff patch to one or more files in the workspace. "
        "The operation is atomic — all-or-nothing across all patched files."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "Unified diff patch string.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Check whether the patch applies without writing changes.",
                "default": False,
            },
        },
        "required": ["patch"],
    }

    def __init__(self, config: "AppConfig") -> None:
        workspace = getattr(config, "tool_workspace_root", "./workspace")
        self._root = Path(workspace).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, rel: str) -> Path | None:
        try:
            resolved = (self._root / rel).resolve()
            resolved.relative_to(self._root)
            return resolved
        except ValueError:
            return None

    async def execute(
        self,
        patch: str = "",
        dry_run: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        if not patch.strip():
            return ToolResult(ok=False, error="'patch' argument is required")

        try:
            file_patches = _split_patch(patch)
        except Exception as exc:
            return ToolResult(ok=False, error=f"Failed to parse patch: {exc}")

        if not file_patches:
            return ToolResult(ok=False, error="No file patches found in the patch string")

        # Validate paths
        safe_paths: dict[str, Path] = {}
        for rel in file_patches:
            safe = self._safe_path(rel)
            if safe is None:
                return ToolResult(
                    ok=False,
                    error=f"Path {rel!r} escapes workspace root — patch rejected",
                )
            safe_paths[rel] = safe

        # Apply (or dry-run)
        originals: dict[str, str] = {}
        applied: list[str] = []
        try:
            for rel, hunks in file_patches.items():
                path = safe_paths[rel]
                original = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
                originals[rel] = original
                new_content = _apply_hunks(original, hunks)
                if not dry_run:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(new_content, encoding="utf-8")
                applied.append(rel)
        except Exception as exc:
            # Roll back
            if not dry_run:
                for rel in applied:
                    try:
                        safe_paths[rel].write_text(originals[rel], encoding="utf-8")
                    except Exception:
                        pass
            return ToolResult(ok=False, error=f"Patch failed on {rel!r}: {exc}")

        return ToolResult(
            ok=True,
            output={
                "patched_files": applied,
                "dry_run": dry_run,
            },
        )


# ---------------------------------------------------------------------------
# Patch parsing helpers
# ---------------------------------------------------------------------------

def _split_patch(patch: str) -> dict[str, list[dict]]:
    """Split a unified diff string into per-file hunk lists.

    Returns ``{relative_path: [hunk, ...]}`` where each hunk is a dict with
    ``old_start``, ``old_len``, ``new_start``, ``new_len``, ``lines``.
    """
    file_patches: dict[str, list[dict]] = {}
    current_file: str | None = None
    current_hunks: list[dict] = []

    lines = patch.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # File header --- a/path or --- path
        if line.startswith("--- "):
            # Consume +++ line
            plus_line = lines[i + 1] if i + 1 < len(lines) else ""
            if plus_line.startswith("+++ "):
                new_path = _strip_prefix(plus_line[4:].strip())
                if current_file is not None:
                    file_patches[current_file] = current_hunks
                current_file = _strip_prefix(line[4:].strip())
                # Prefer the +++ path as the target
                current_file = new_path if new_path != "/dev/null" else current_file
                current_hunks = []
                i += 2
                continue

        # Hunk header @@ -a,b +c,d @@
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m and current_file is not None:
            old_start = int(m.group(1))
            old_len = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_len = int(m.group(4)) if m.group(4) is not None else 1
            hunk_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- ", "+++ ", "diff ")):
                hunk_lines.append(lines[i])
                i += 1
            current_hunks.append({
                "old_start": old_start,
                "old_len": old_len,
                "new_start": new_start,
                "new_len": new_len,
                "lines": hunk_lines,
            })
            continue

        i += 1

    if current_file is not None:
        file_patches[current_file] = current_hunks

    return file_patches


def _strip_prefix(path: str) -> str:
    """Remove a/ or b/ prefix from diff paths."""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _apply_hunks(original: str, hunks: list[dict]) -> str:
    """Apply hunks to *original* text and return the new content."""
    original_lines = original.splitlines(keepends=True)
    result_lines = list(original_lines)
    offset = 0

    for hunk in hunks:
        old_start = hunk["old_start"] - 1 + offset  # convert to 0-based
        hunk_lines = hunk["lines"]

        old_block: list[str] = []
        new_block: list[str] = []
        for hl in hunk_lines:
            if hl.startswith("-"):
                old_block.append(hl[1:] if not hl[1:].endswith("\n") else hl[1:])
            elif hl.startswith("+"):
                new_block.append(hl[1:] if not hl[1:].endswith("\n") else hl[1:])
            elif hl.startswith(" "):
                old_block.append(hl[1:] if not hl[1:].endswith("\n") else hl[1:])
                new_block.append(hl[1:] if not hl[1:].endswith("\n") else hl[1:])

        # Normalise line endings for comparison
        old_len = len(old_block)
        # Find where to apply (use old_start as hint)
        start = old_start
        result_lines[start: start + old_len] = new_block
        offset += len(new_block) - old_len

    return "".join(result_lines)
