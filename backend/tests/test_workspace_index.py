"""Tests for WorkspaceIndexer and WorkspaceIndexTool."""
from __future__ import annotations

import asyncio
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# WorkspaceIndexer — unit tests using mocked LanceDB / sentence-transformers
# ---------------------------------------------------------------------------

def _make_indexer(tmp_path: Path):
    """Return a WorkspaceIndexer with all heavy deps mocked out."""
    import numpy as np

    # Build a minimal fake encoder
    fake_encoder = MagicMock()
    fake_encoder.get_sentence_embedding_dimension.return_value = 8

    def _encode(texts_or_text, *args, **kwargs):
        if isinstance(texts_or_text, list):
            return np.zeros((len(texts_or_text), 8), dtype="float32")
        return np.zeros(8, dtype="float32")

    fake_encoder.encode.side_effect = _encode

    # Build a minimal fake LanceDB table
    import pandas as pd
    fake_table = MagicMock()
    fake_table.to_pandas.return_value = pd.DataFrame()
    fake_table.add = MagicMock()
    fake_table.delete = MagicMock()

    def _fake_search(vec):
        result = MagicMock()
        result.limit.return_value = result
        result.where.return_value = result
        result.to_pandas.return_value = pd.DataFrame(
            columns=["file_path", "project_root", "chunk_type", "chunk_index", "text", "last_indexed"]
        )
        return result

    fake_table.search.side_effect = _fake_search

    # Fake DB
    fake_db = MagicMock()
    fake_db.table_names.return_value = ["file_index"]
    fake_db.open_table.return_value = fake_table

    # Patch imports so WorkspaceIndexer believes deps are available
    lancedb_mod = MagicMock()
    lancedb_mod.connect.return_value = fake_db
    st_mod = MagicMock()
    st_mod.SentenceTransformer.return_value = fake_encoder
    np_mod = np

    with patch.dict("sys.modules", {
        "lancedb": lancedb_mod,
        "sentence_transformers": st_mod,
        "numpy": np_mod,
    }):
        import importlib
        import backend.memory.workspace_indexer as wi_mod
        importlib.reload(wi_mod)
        # Force _DEPS_AVAILABLE after reload
        wi_mod._DEPS_AVAILABLE = True
        indexer = wi_mod.WorkspaceIndexer(
            db_path=str(tmp_path / "wsidx"),
            embedding_model="all-MiniLM-L6-v2",
        )
        indexer._db = fake_db
        indexer._table = fake_table
        indexer._encoder = fake_encoder
        return indexer, wi_mod


@pytest.fixture()
def sample_project(tmp_path: Path) -> Path:
    """Create a tiny fake project tree."""
    root = tmp_path / "myproject"
    root.mkdir()
    (root / "main.py").write_text("def hello():\n    print('hi')\n\nclass Foo:\n    pass\n")
    (root / "README.md").write_text("# My Project\nThis is a test.\n")
    sub = root / "sub"
    sub.mkdir()
    (sub / "util.py").write_text("def add(a, b):\n    return a + b\n")
    return root


class TestChunking:
    def test_chunk_python_splits_on_def_class(self):
        from backend.memory.workspace_indexer import _chunk_python
        src = "x = 1\n\ndef foo():\n    pass\n\nclass Bar:\n    pass\n"
        chunks = _chunk_python(src)
        assert len(chunks) >= 2
        types_found = {c[1] for c in chunks}
        assert "function" in types_found or "class" in types_found

    def test_chunk_generic_produces_blocks(self):
        from backend.memory.workspace_indexer import _chunk_generic, CHUNK_SIZE
        lines = [f"line {i}\n" for i in range(CHUNK_SIZE * 3)]
        text = "".join(lines)
        chunks = _chunk_generic(text)
        assert len(chunks) >= 2
        for text, ctype in chunks:
            assert ctype == "block"

    def test_chunk_file_python(self, sample_project: Path):
        from backend.memory.workspace_indexer import _chunk_file
        chunks = _chunk_file(sample_project / "main.py")
        assert len(chunks) >= 1
        assert all(isinstance(c, tuple) and len(c) == 2 for c in chunks)

    def test_chunk_file_markdown(self, sample_project: Path):
        from backend.memory.workspace_indexer import _chunk_file
        chunks = _chunk_file(sample_project / "README.md")
        assert len(chunks) >= 1

    def test_is_indexable_skips_binary(self, tmp_path: Path):
        from backend.memory.workspace_indexer import _is_indexable
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n")
        assert not _is_indexable(f)

    def test_is_indexable_accepts_py(self, sample_project: Path):
        from backend.memory.workspace_indexer import _is_indexable
        assert _is_indexable(sample_project / "main.py")


class TestWorkspaceIndexer:
    @pytest.mark.asyncio
    async def test_index_directory_adds_chunks(self, sample_project: Path, tmp_path: Path):
        indexer, _ = _make_indexer(tmp_path)
        stats = await indexer.index_directory(sample_project)
        assert stats["files_scanned"] >= 3
        assert stats["chunks_added"] >= 3

    @pytest.mark.asyncio
    async def test_index_directory_invalid_path(self, tmp_path: Path):
        indexer, _ = _make_indexer(tmp_path)
        with pytest.raises(ValueError, match="Not a directory"):
            await indexer.index_directory(tmp_path / "nonexistent")

    @pytest.mark.asyncio
    async def test_search_returns_list(self, tmp_path: Path):
        indexer, _ = _make_indexer(tmp_path)
        results = await indexer.search("hello function")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_indexed_projects_empty(self, tmp_path: Path):
        indexer, _ = _make_indexer(tmp_path)
        projects = await indexer.get_indexed_projects()
        assert isinstance(projects, list)


# ---------------------------------------------------------------------------
# WorkspaceIndexTool — unit tests
# ---------------------------------------------------------------------------

def _make_tool(tmp_path: Path):
    """Return a (WorkspaceIndexTool, mock_indexer) pair."""
    mock_indexer = MagicMock()
    mock_indexer.index_directory = AsyncMock(
        return_value={"files_scanned": 5, "chunks_added": 12, "files_skipped": 0}
    )
    mock_indexer.search = AsyncMock(return_value=[
        {"file_path": "main.py", "project_root": str(tmp_path), "chunk_type": "function",
         "chunk_index": 0, "text": "def hello(): ...", "last_indexed": "2026-01-01T00:00:00+00:00"}
    ])
    mock_indexer.get_indexed_projects = AsyncMock(return_value=[
        {"project_root": str(tmp_path), "chunks": 12, "file_count": 5, "last_indexed": "2026-01-01"}
    ])

    from backend.tools.impl.workspace_index_tool import WorkspaceIndexTool
    tool = WorkspaceIndexTool(indexer=mock_indexer)
    return tool, mock_indexer


class TestWorkspaceIndexTool:
    @pytest.mark.asyncio
    async def test_index_action(self, tmp_path: Path):
        tool, mock_indexer = _make_tool(tmp_path)
        result = await tool.execute(action="index", path=str(tmp_path))
        assert result.ok
        assert result.output["files_scanned"] == 5
        mock_indexer.index_directory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_index_action_missing_path(self, tmp_path: Path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(action="index", path="")
        assert not result.ok
        assert "path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_index_action_nonexistent_path(self, tmp_path: Path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(action="index", path=str(tmp_path / "does_not_exist"))
        assert not result.ok

    @pytest.mark.asyncio
    async def test_search_action(self, tmp_path: Path):
        tool, mock_indexer = _make_tool(tmp_path)
        result = await tool.execute(action="search", query="hello function")
        assert result.ok
        assert len(result.output["results"]) == 1
        mock_indexer.search.assert_awaited_once_with("hello function", project_root=None, limit=8)

    @pytest.mark.asyncio
    async def test_search_action_missing_query(self, tmp_path: Path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(action="search", query="")
        assert not result.ok

    @pytest.mark.asyncio
    async def test_status_action(self, tmp_path: Path):
        tool, mock_indexer = _make_tool(tmp_path)
        result = await tool.execute(action="status")
        assert result.ok
        assert len(result.output["indexed_projects"]) == 1

    @pytest.mark.asyncio
    async def test_unknown_action(self, tmp_path: Path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(action="explode")
        assert not result.ok

    @pytest.mark.asyncio
    async def test_no_indexer(self):
        from backend.tools.impl.workspace_index_tool import WorkspaceIndexTool
        tool = WorkspaceIndexTool(indexer=None)
        result = await tool.execute(action="index", path="/tmp")
        assert not result.ok
        assert "not available" in result.error.lower()

    def test_trigger_auto_index_no_op_when_no_indexer(self):
        from backend.tools.impl.workspace_index_tool import WorkspaceIndexTool
        tool = WorkspaceIndexTool(indexer=None)
        # Should not raise
        tool.trigger_auto_index("/tmp")

    def test_trigger_auto_index_deduplicates(self, tmp_path: Path):
        tool, _ = _make_tool(tmp_path)
        resolved = str(tmp_path.resolve())
        tool._indexing_in_progress.add(resolved)
        # Should not schedule a second task (already in progress)
        tool.trigger_auto_index(tmp_path)
        # Still only one entry
        assert resolved in tool._indexing_in_progress


# ---------------------------------------------------------------------------
# FileTool auto-trigger integration
# ---------------------------------------------------------------------------

class TestFileToolAutoTrigger:
    def _make_file_tool(self, tmp_path: Path, workspace_index=None):
        from backend.tools.impl.file_tool import FileTool
        cfg = MagicMock()
        cfg.tool_workspace_root = str(tmp_path)
        return FileTool(config=cfg, workspace_index=workspace_index)

    @pytest.mark.asyncio
    async def test_list_triggers_auto_index(self, tmp_path: Path):
        mock_tool = MagicMock()
        mock_tool.is_available.return_value = True
        mock_tool.trigger_auto_index = MagicMock()

        file_tool = self._make_file_tool(tmp_path, workspace_index=mock_tool)
        result = await file_tool.execute(action="list", path=".")
        assert result.ok
        mock_tool.trigger_auto_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_no_trigger_when_no_workspace_index(self, tmp_path: Path):
        file_tool = self._make_file_tool(tmp_path, workspace_index=None)
        # Should complete without error
        result = await file_tool.execute(action="list", path=".")
        assert result.ok

    @pytest.mark.asyncio
    async def test_read_does_not_trigger(self, tmp_path: Path):
        f = tmp_path / "hello.txt"
        f.write_text("hello")
        mock_tool = MagicMock()
        mock_tool.is_available.return_value = True
        mock_tool.trigger_auto_index = MagicMock()

        file_tool = self._make_file_tool(tmp_path, workspace_index=mock_tool)
        result = await file_tool.execute(action="read", path="hello.txt")
        assert result.ok
        mock_tool.trigger_auto_index.assert_not_called()
