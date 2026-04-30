from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any


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


class LanceDBMemory:
    """Persistent long-term memory backed by LanceDB and sentence-transformers.

    Every conversation turn (user + assistant) is embedded and stored with
    metadata and a UTC timestamp.  Semantic search retrieves the most
    relevant past interactions for any new query.

    Initialisation is lazy and protected by an ``asyncio.Lock`` so the first
    concurrent callers do not race to create the table.

    Parameters
    ----------
    db_path:
        Path to the LanceDB database directory.
    embedding_model:
        ``sentence-transformers`` model name (e.g. ``all-MiniLM-L6-v2``).

    TODO: Replace the seed-row table creation workaround with a PyArrow
          schema passed directly to ``create_table`` to prevent orphaned
          zero-vector rows. See BUGS.md BUG-003.
    TODO: Add a ``delete_by_metadata`` method so old or irrelevant memories
          can be pruned without resetting the entire database.
    """

    def __init__(self, db_path: str, embedding_model: str) -> None:
        if not _DEPS_AVAILABLE:
            raise ImportError(
                "LanceDBMemory requires lancedb, sentence-transformers, numpy, pandas, and pyarrow. "
                "Install them with: pip install lancedb sentence-transformers numpy pandas pyarrow. "
                f"Original import error: {_DEPS_ERROR}"
            )
        self._db_path = db_path
        self._embedding_model_name = embedding_model
        self._db: Any = None
        self._table: Any = None
        self._encoder: Any = None
        self._init_lock = asyncio.Lock()

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._embedding_model_name)
        return self._encoder

    def _get_table(self) -> Any:
        if self._db is None:
            self._db = lancedb.connect(self._db_path)
        if self._table is None:
            existing = self._db.table_names()
            if "memories" in existing:
                self._table = self._db.open_table("memories")
            else:
                encoder = self._get_encoder()
                dim = encoder.get_sentence_embedding_dimension()
                schema = {
                    "vector": np.zeros(dim, dtype=np.float32),
                    "text": "",
                    "metadata": "{}",
                    "timestamp": "",
                }
                self._table = self._db.create_table("memories", data=[schema])
                # Remove seed row immediately
                self._table.delete("text = ''")
        return self._table

    async def store(self, text: str, metadata: dict | None = None) -> None:
        async with self._init_lock:
            encoder = await asyncio.to_thread(self._get_encoder)
            vector = await asyncio.to_thread(encoder.encode, text)
            table = await asyncio.to_thread(self._get_table)
        import json
        row = {
            "vector": vector.tolist(),
            "text": text,
            "metadata": json.dumps(metadata or {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await asyncio.to_thread(table.add, [row])

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        async with self._init_lock:
            encoder = await asyncio.to_thread(self._get_encoder)
            query_vector = await asyncio.to_thread(encoder.encode, query)
            table = await asyncio.to_thread(self._get_table)
        query_vector = query_vector.tolist()

        def _search():
            return (
                table.search(query_vector)
                .limit(limit)
                .to_pandas()
            )

        results = await asyncio.to_thread(_search)
        import json
        output: list[dict] = []
        for _, row in results.iterrows():
            output.append({
                "text": row["text"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "timestamp": row.get("timestamp", ""),
            })
        return output
