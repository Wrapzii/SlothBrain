"""Lazy, usage-driven workspace (file system) indexer.

Files and code snippets are indexed on first access — not upfront — keeping
the index small and relevant.  Each project folder gets its own logical
partition in the ``file_index`` LanceDB table, keyed by the resolved
``project_root`` path.

Chunking strategy
-----------------
* Python files are split on ``def `` / ``class `` boundaries so each chunk
  corresponds to roughly one function or class.
* All other text files are split into fixed-size overlapping windows
  (``CHUNK_SIZE`` lines, ``CHUNK_OVERLAP`` lines of overlap).
* Binary files are skipped entirely.

Metadata stored per chunk
--------------------------
* ``project_root`` — absolute path of the indexed project directory.
* ``file_path``    — path of the file relative to *project_root*.
* ``chunk_index``  — zero-based position of this chunk within the file.
* ``chunk_type``   — ``"function"``, ``"class"``, ``"block"``, or ``"raw"``.
* ``last_indexed`` — UTC ISO-8601 timestamp of when the chunk was indexed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import lancedb
    import numpy as np
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    from sentence_transformers import SentenceTransformer
    _DEPS_AVAILABLE = True
    _DEPS_ERROR: Exception | None = None
except ImportError as exc:
    _DEPS_AVAILABLE = False
    _DEPS_ERROR = exc

# ---------------------------------------------------------------------------
# Chunking constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 60          # lines per raw block chunk
CHUNK_OVERLAP = 10       # overlap between consecutive block chunks
MAX_FILE_BYTES = 512_000  # skip files larger than ~500 KB

# File extensions treated as indexable text
_TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".kt", ".scala",
    ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".cs", ".swift",
    ".sh", ".bash", ".zsh",
    ".md", ".txt", ".rst", ".yaml", ".yml",
    ".toml", ".json", ".xml", ".html", ".css",
    ".env", ".cfg", ".ini", ".conf",
}

# Directories to skip when scanning
_SKIP_DIRS = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt",
    ".tox", "eggs", ".eggs",
}


def _is_indexable(path: Path) -> bool:
    if path.suffix.lower() not in _TEXT_EXTENSIONS:
        return False
    try:
        return path.stat().st_size <= MAX_FILE_BYTES
    except OSError:
        return False


def _chunk_python(text: str) -> list[tuple[str, str]]:
    """Split Python source into per-function / per-class chunks.

    Returns a list of ``(chunk_text, chunk_type)`` tuples.
    """
    chunks: list[tuple[str, str]] = []
    lines = text.splitlines(keepends=True)
    pattern = re.compile(r"^(def |class |async def )", re.MULTILINE)
    boundaries: list[int] = [0]
    for m in pattern.finditer(text):
        lineno = text.count("\n", 0, m.start())
        if lineno > 0:
            boundaries.append(lineno)
    boundaries.append(len(lines))

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        chunk_lines = lines[start:end]
        chunk_text = "".join(chunk_lines).strip()
        if not chunk_text:
            continue
        first = chunk_lines[0].lstrip() if chunk_lines else ""
        if first.startswith("class "):
            ctype = "class"
        elif first.startswith("def ") or first.startswith("async def "):
            ctype = "function"
        else:
            ctype = "block"
        chunks.append((chunk_text, ctype))
    return chunks or [(text.strip(), "raw")]


def _chunk_generic(text: str) -> list[tuple[str, str]]:
    """Split text into overlapping line-window chunks."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    chunks: list[tuple[str, str]] = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for start in range(0, len(lines), step):
        chunk_text = "".join(lines[start : start + CHUNK_SIZE]).strip()
        if chunk_text:
            chunks.append((chunk_text, "block"))
    return chunks or [(text.strip(), "raw")]


def _chunk_file(path: Path) -> list[tuple[str, str]]:
    """Return chunks for *path*, choosing the appropriate strategy."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if not text.strip():
        return []
    if path.suffix.lower() == ".py":
        return _chunk_python(text)
    return _chunk_generic(text)


class WorkspaceIndexer:
    """Lazy per-project file-system index backed by LanceDB.

    Parameters
    ----------
    db_path:
        Directory used by :func:`lancedb.connect`.  A sub-table
        ``file_index`` is created/opened inside this database.
    embedding_model:
        ``sentence-transformers`` model name.
    """

    def __init__(self, db_path: str, embedding_model: str) -> None:
        if not _DEPS_AVAILABLE:
            raise ImportError(
                "WorkspaceIndexer requires lancedb, sentence-transformers, numpy, pandas, and pyarrow. "
                "Install them with: pip install lancedb sentence-transformers numpy pandas pyarrow. "
                f"Original import error: {_DEPS_ERROR}"
            )
        self._db_path = db_path
        self._embedding_model_name = embedding_model
        self._db: Any = None
        self._table: Any = None
        self._encoder: Any = None
        self._init_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._embedding_model_name)
        return self._encoder

    def _get_table(self) -> Any:
        if self._db is None:
            self._db = lancedb.connect(self._db_path)
        if self._table is None:
            existing = self._db.table_names()
            if "file_index" in existing:
                self._table = self._db.open_table("file_index")
            else:
                encoder = self._get_encoder()
                dim = encoder.get_sentence_embedding_dimension()
                seed = {
                    "vector": np.zeros(dim, dtype=np.float32),
                    "text": "",
                    "project_root": "",
                    "file_path": "",
                    "chunk_index": 0,
                    "chunk_type": "",
                    "last_indexed": "",
                }
                self._table = self._db.create_table("file_index", data=[seed])
                self._table.delete("text = ''")
        return self._table

    def _encode(self, texts: list[str]) -> "np.ndarray":
        encoder = self._get_encoder()
        return encoder.encode(texts, show_progress_bar=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def index_directory(
        self,
        directory: str | Path,
        *,
        force: bool = False,
    ) -> dict[str, int]:
        """Index all indexable files under *directory*.

        Parameters
        ----------
        directory:
            Absolute (or relative) path to the project root to scan.
        force:
            When ``True``, re-index files even if they have been indexed
            before.  When ``False`` (default), files whose path is already
            present in the index are skipped.

        Returns
        -------
        dict with keys ``files_scanned``, ``chunks_added``, ``files_skipped``.
        """
        root = Path(directory).resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Not a directory: {root}")

        project_root_str = str(root)
        now = datetime.now(timezone.utc).isoformat()

        async with self._init_lock:
            encoder = await asyncio.to_thread(self._get_encoder)
            table = await asyncio.to_thread(self._get_table)

        # Collect already-indexed paths for this project root (skip logic)
        already_indexed: set[str] = set()
        if not force:
            already_indexed = await asyncio.to_thread(
                self._get_indexed_paths, table, project_root_str
            )

        # Collect files to process
        files_to_index: list[Path] = []
        for path in root.rglob("*"):
            if path.is_file() and _is_indexable(path):
                rel = str(path.relative_to(root))
                if not force and rel in already_indexed:
                    continue
                # Skip paths under excluded directories
                if any(part in _SKIP_DIRS for part in path.parts):
                    continue
                files_to_index.append(path)

        files_scanned = len(files_to_index)
        files_skipped = 0
        chunks_added = 0

        for file_path in files_to_index:
            rel = str(file_path.relative_to(root))
            chunks = await asyncio.to_thread(_chunk_file, file_path)
            if not chunks:
                files_skipped += 1
                continue

            # Remove old chunks for this file (in case of re-index)
            if rel in already_indexed:
                await asyncio.to_thread(
                    table.delete,
                    f"project_root = '{project_root_str}' AND file_path = '{rel}'",
                )

            texts = [c[0] for c in chunks]
            ctypes = [c[1] for c in chunks]
            vectors = await asyncio.to_thread(encoder.encode, texts, False)

            rows = [
                {
                    "vector": vectors[i].tolist(),
                    "text": texts[i],
                    "project_root": project_root_str,
                    "file_path": rel,
                    "chunk_index": i,
                    "chunk_type": ctypes[i],
                    "last_indexed": now,
                }
                for i in range(len(texts))
            ]
            await asyncio.to_thread(table.add, rows)
            chunks_added += len(rows)

        logger.info(
            "WorkspaceIndexer: indexed %d files (%d chunks) under %s",
            files_scanned - files_skipped,
            chunks_added,
            root,
        )
        return {
            "files_scanned": files_scanned,
            "chunks_added": chunks_added,
            "files_skipped": files_skipped,
        }

    async def search(
        self,
        query: str,
        *,
        project_root: str | None = None,
        limit: int = 8,
    ) -> list[dict]:
        """Semantic search over the file index.

        Parameters
        ----------
        query:
            Natural-language or code query.
        project_root:
            If given, restrict results to that project root.
        limit:
            Maximum number of chunks to return.
        """
        async with self._init_lock:
            encoder = await asyncio.to_thread(self._get_encoder)
            table = await asyncio.to_thread(self._get_table)

        query_vector = await asyncio.to_thread(encoder.encode, query)

        def _run_search() -> Any:
            q = table.search(query_vector.tolist()).limit(limit)
            if project_root:
                q = q.where(f"project_root = '{project_root}'")
            return q.to_pandas()

        results_df = await asyncio.to_thread(_run_search)

        output: list[dict] = []
        for _, row in results_df.iterrows():
            output.append({
                "file_path": row.get("file_path", ""),
                "project_root": row.get("project_root", ""),
                "chunk_type": row.get("chunk_type", ""),
                "chunk_index": int(row.get("chunk_index", 0)),
                "text": row.get("text", ""),
                "last_indexed": row.get("last_indexed", ""),
            })
        return output

    async def get_indexed_projects(self) -> list[dict]:
        """Return a summary of all indexed project roots."""
        async with self._init_lock:
            table = await asyncio.to_thread(self._get_table)

        def _summarise() -> list[dict]:
            df = table.to_pandas()
            if df.empty:
                return []
            summary: dict[str, dict] = {}
            for _, row in df.iterrows():
                root = row.get("project_root", "")
                if not root:
                    continue
                if root not in summary:
                    summary[root] = {"project_root": root, "chunks": 0, "files": set(), "last_indexed": ""}
                summary[root]["chunks"] += 1
                summary[root]["files"].add(row.get("file_path", ""))
                ts = row.get("last_indexed", "")
                if ts and ts > summary[root]["last_indexed"]:
                    summary[root]["last_indexed"] = ts
            result = []
            for v in summary.values():
                result.append({
                    "project_root": v["project_root"],
                    "chunks": v["chunks"],
                    "file_count": len(v["files"]),
                    "last_indexed": v["last_indexed"],
                })
            return result

        return await asyncio.to_thread(_summarise)

    # ------------------------------------------------------------------
    # Private helpers (run in threads)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_indexed_paths(table: Any, project_root: str) -> set[str]:
        """Return the set of already-indexed file paths for *project_root*."""
        try:
            df = table.to_pandas()
            if df.empty:
                return set()
            mask = df["project_root"] == project_root
            return set(df.loc[mask, "file_path"].unique())
        except Exception:
            return set()
