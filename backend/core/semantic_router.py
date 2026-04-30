"""Semantic tool router — selects the most relevant tools for a given context.

Uses sentence-transformers embeddings to score each tool in the active profile
against the current task context, then returns the top-k highest-scoring tools.
Falls back to the full profile toolset on any error or when too few tools
pass the configured similarity threshold, so the agent is never left without
tools.

Typical usage::

    router = SemanticRouter(embedding_model="all-MiniLM-L6-v2", k=5)
    router.index_tools(registry.all_tools())           # once at startup

    relevant = router.get_relevant_tools(
        context="write a Python script that reads CSV files",
        profile_tools=registry.get_tools_for_profile("coding"),
    )

The ``context`` string should combine the current task/goal with a short
summary of recent agent messages so the router can pick up on both the
high-level objective and the immediate state of the conversation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.tools.base import Tool

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    _DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DEPS_AVAILABLE = False


class SemanticRouter:
    """Embedding-based semantic router for tool selection.

    At indexing time, tool descriptions are embedded and cached in memory.
    At routing time, the current context is embedded and cosine-similarity is
    computed against every candidate tool in the active profile.  The top-k
    tools above ``threshold`` are returned.

    Parameters
    ----------
    embedding_model:
        ``sentence-transformers`` model name (e.g. ``"all-MiniLM-L6-v2"``).
        Must match the model used by the rest of the application so the shared
        encoder can be reused.
    k:
        Default number of tools to return per routing call.  Callers may
        override this per-call.
    threshold:
        Minimum cosine-similarity score for a tool to be included.  When
        ``0.0`` (the default) the top-*k* tools are always returned regardless
        of their absolute score.  Positive values (e.g. ``0.2``) filter out
        weakly-related tools; when *no* tool exceeds the threshold the router
        falls back to the full profile toolset automatically.
    bypass_tools:
        Tool names that always bypass routing and are unconditionally included
        in every result set (e.g. ``"screenshot"``, ``"ui"``).  Only tools
        that are also present in the active profile will actually be added.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        k: int = 5,
        threshold: float = 0.0,
        bypass_tools: frozenset[str] | None = None,
    ) -> None:
        self._embedding_model_name = embedding_model
        self.k = k
        self.threshold = threshold
        self.bypass_tools: frozenset[str] = bypass_tools or frozenset()
        self._encoder: Any = None
        # tool_name → float32 numpy vector
        self._embeddings: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Encoder lifecycle (lazy)
    # ------------------------------------------------------------------

    def _get_encoder(self) -> Any:
        """Return the shared SentenceTransformer encoder (lazy init)."""
        if not _DEPS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SemanticRouter. "
                "Install it with: pip install sentence-transformers"
            )
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._embedding_model_name)
        return self._encoder

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_tools(self, tools: list["Tool"]) -> None:
        """Compute and cache embeddings for *tools*.

        Safe to call multiple times — new tools are added and existing tools
        are re-indexed.  Embedding failures are logged and silently swallowed
        so tool registration never raises.
        """
        if not tools:
            return
        try:
            encoder = self._get_encoder()
            texts = [t.description or t.name for t in tools]
            vectors = encoder.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
            for tool, vec in zip(tools, vectors):
                self._embeddings[tool.name] = vec
            logger.debug("SemanticRouter indexed %d tools", len(tools))
        except Exception as exc:
            logger.warning("SemanticRouter.index_tools failed: %s", exc)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def get_relevant_tools(
        self,
        context: str,
        profile_tools: list["Tool"],
        k: int | None = None,
        threshold: float | None = None,
    ) -> list["Tool"]:
        """Return the most relevant tools from *profile_tools* for *context*.

        Profile restrictions are always respected — this method only selects
        *within* the supplied ``profile_tools`` list, never adding tools
        outside the profile.

        Parameters
        ----------
        context:
            Current task context string.  Combine the task/goal description
            with recent agent messages for best results.
        profile_tools:
            Full set of tools allowed by the active profile.  The router
            selects a focused subset from this list.
        k:
            Override the instance default number of tools to return.
        threshold:
            Override the instance default minimum cosine-similarity score.

        Returns
        -------
        list[Tool]
            Focused subset of *profile_tools*, sorted by relevance.  Bypass
            tools are always prepended when present in *profile_tools*.
            Falls back to *profile_tools* unchanged on any error.
        """
        effective_k = k if k is not None else self.k
        effective_threshold = threshold if threshold is not None else self.threshold

        if not profile_tools:
            return profile_tools

        # Split tools into bypass (always included) and candidates (ranked).
        bypass = [t for t in profile_tools if t.name in self.bypass_tools]
        candidates = [t for t in profile_tools if t.name not in self.bypass_tools]

        if not candidates:
            # All tools are bypass tools — nothing to rank.
            return profile_tools

        # Index any tools not yet in cache.
        unindexed = [t for t in candidates if t.name not in self._embeddings]
        if unindexed:
            self.index_tools(unindexed)

        try:
            encoder = self._get_encoder()
        except ImportError:
            logger.warning(
                "SemanticRouter: sentence-transformers unavailable — returning full profile toolset"
            )
            return profile_tools

        try:
            import numpy as np  # noqa: PLC0415 — local import to mirror _DEPS_AVAILABLE guard

            query_vec = encoder.encode(
                context, convert_to_numpy=True, show_progress_bar=False
            )
            norm_q = float(np.linalg.norm(query_vec))

            scores: list[tuple[float, Tool]] = []
            for tool in candidates:
                vec = self._embeddings.get(tool.name)
                if vec is None:
                    scores.append((0.0, tool))
                    continue
                norm_v = float(np.linalg.norm(vec))
                if norm_q == 0.0 or norm_v == 0.0:
                    sim = 0.0
                else:
                    sim = float(np.dot(query_vec, vec) / (norm_q * norm_v))
                scores.append((sim, tool))

            scores.sort(key=lambda x: x[0], reverse=True)

            # How many candidate slots remain after reserving bypass slots.
            remaining_k = max(0, effective_k - len(bypass))
            selected = [
                tool
                for sim, tool in scores[:remaining_k]
                if sim >= effective_threshold
            ]

            # If threshold filtering removed everything, fall back gracefully.
            if not selected and effective_threshold > 0.0:
                logger.debug(
                    "SemanticRouter: no tools above threshold %.2f — falling back to full profile toolset",
                    effective_threshold,
                )
                return profile_tools

            result = bypass + selected
            logger.debug(
                "SemanticRouter selected %d/%d tools (k=%d threshold=%.2f)",
                len(result),
                len(profile_tools),
                effective_k,
                effective_threshold,
            )
            return result

        except Exception as exc:
            logger.warning(
                "SemanticRouter.get_relevant_tools failed (%s) — returning full profile toolset",
                exc,
            )
            return profile_tools
