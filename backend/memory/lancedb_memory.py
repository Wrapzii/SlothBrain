from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
import math


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
            if hasattr(self._db, "list_tables"):
                existing = self._db.list_tables()
            else:
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
                try:
                    self._table = self._db.create_table("memories", data=[schema])
                    # Remove seed row immediately
                    self._table.delete("text = ''")
                except Exception:
                    self._table = self._db.open_table("memories")
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
        return await self.search_advanced(query=query, limit=limit)

    async def search_advanced(
        self,
        query: str,
        limit: int = 5,
        *,
        metadata_filter: dict[str, Any] | None = None,
        exclude_metadata: dict[str, Any] | None = None,
        candidate_pool: int = 32,
        recency_half_life_days: float = 7.0,
    ) -> list[dict]:
        async with self._init_lock:
            encoder = await asyncio.to_thread(self._get_encoder)
            query_vector = await asyncio.to_thread(encoder.encode, query)
            table = await asyncio.to_thread(self._get_table)
        query_vector = query_vector.tolist()

        def _search():
            return (
                table.search(query_vector)
                .limit(max(limit, candidate_pool))
                .to_pandas()
            )

        results = await asyncio.to_thread(_search)
        import json
        output: list[dict] = []

        def _metadata_match(meta: dict, expected: dict[str, Any]) -> bool:
            for k, v in expected.items():
                if meta.get(k) != v:
                    return False
            return True

        now = datetime.now(timezone.utc)
        half_life_seconds = max(1.0, float(recency_half_life_days) * 86400.0)
        scored: list[tuple[float, dict]] = []

        for _, row in results.iterrows():
            meta = json.loads(row["metadata"]) if row.get("metadata") else {}
            if metadata_filter and not _metadata_match(meta, metadata_filter):
                continue
            if exclude_metadata and _metadata_match(meta, exclude_metadata):
                continue

            text = str(row.get("text", "") or "")
            timestamp = str(row.get("timestamp", "") or "")
            distance = row.get("_distance", None)

            try:
                d = float(distance)
            except Exception:
                d = 1.0

            # Convert vector distance to bounded similarity score.
            similarity = 1.0 / (1.0 + max(0.0, d))

            recency_score = 0.5
            if timestamp:
                try:
                    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    age_seconds = max(0.0, (now - ts).total_seconds())
                    recency_score = math.exp(-math.log(2.0) * (age_seconds / half_life_seconds))
                except Exception:
                    recency_score = 0.5

            combined = (0.85 * similarity) + (0.15 * recency_score)
            scored.append(
                (
                    combined,
                    {
                        "text": text,
                        "metadata": meta,
                        "timestamp": timestamp,
                        "score": round(combined, 6),
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)

        seen_text: set[str] = set()
        for _, item in scored:
            key = (item.get("text", "") or "").strip().lower()
            if not key or key in seen_text:
                continue
            seen_text.add(key)
            output.append(item)
            if len(output) >= limit:
                break

        return output
