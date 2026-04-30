from __future__ import annotations

import json
import logging
from dataclasses import dataclass

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SemanticRouterResult:
    tool_names: list[str]
    used_fallback: bool
    reason: str
    scores: dict[str, float]


class SemanticRouter:
    """Embedding-based tool router with profile-aware fallback semantics."""

    def __init__(
        self,
        embedding_model: str,
        top_k: int = 8,
        min_similarity: float = 0.2,
        enabled: bool = True,
        critical_tools: list[str] | None = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._top_k = max(1, int(top_k))
        self._min_similarity = float(min_similarity)
        self._enabled = bool(enabled)
        self._critical_tools = set(critical_tools or [])
        self._encoder: SentenceTransformer | None = None
        self._tool_vectors: dict[str, "np.ndarray"] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled and _DEPS_AVAILABLE

    def _get_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._embedding_model)
        return self._encoder

    def _embed_text(self, text: str) -> "np.ndarray":
        encoder = self._get_encoder()
        vec = encoder.encode(text)
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def register_tool(self, tool_name: str, description: str, parameters_schema: dict) -> None:
        if not self.enabled:
            return
        try:
            schema = json.dumps(parameters_schema, ensure_ascii=False, sort_keys=True)
            text = f"{tool_name}\n{description}\n{schema}"
            self._tool_vectors[tool_name] = self._embed_text(text)
        except Exception as exc:  # pragma: no cover
            logger.warning("SemanticRouter failed to register tool %s: %s", tool_name, exc)

    def unregister_tool(self, tool_name: str) -> None:
        self._tool_vectors.pop(tool_name, None)

    def get_relevant_tools(
        self,
        context: str,
        profile: str = "",
        candidates: list[str] | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
        critical_tools: list[str] | None = None,
    ) -> SemanticRouterResult:
        candidates = candidates or []
        candidate_set = set(candidates)
        bypass = set(critical_tools or []).intersection(candidate_set)

        if not candidates:
            return SemanticRouterResult([], True, f"no_candidates:{profile}", {})
        if not self.enabled:
            return SemanticRouterResult(sorted(candidate_set), True, f"router_disabled:{profile}", {})
        if not context.strip():
            chosen = sorted(candidate_set.union(bypass))
            return SemanticRouterResult(chosen, True, "empty_context", {})

        try:
            query_vec = self._embed_text(context)
            rows: list[tuple[str, float]] = []
            for name in candidates:
                tool_vec = self._tool_vectors.get(name)
                if tool_vec is None:
                    continue
                score = float(np.dot(query_vec, tool_vec))
                rows.append((name, score))
        except Exception as exc:  # pragma: no cover
            logger.warning("SemanticRouter retrieval failed: %s", exc)
            chosen = sorted(candidate_set.union(bypass))
            return SemanticRouterResult(chosen, True, "router_error", {})

        if not rows:
            chosen = sorted(candidate_set.union(bypass))
            return SemanticRouterResult(chosen, True, "no_tool_vectors", {})

        threshold = self._min_similarity if min_similarity is None else float(min_similarity)
        k = max(1, int(top_k or self._top_k))
        ranked = sorted(rows, key=lambda x: x[1], reverse=True)
        selected = [name for name, score in ranked if score >= threshold][:k]
        if not selected:
            chosen = sorted(candidate_set.union(bypass))
            return SemanticRouterResult(chosen, True, "low_confidence", {n: s for n, s in ranked})

        selected_set = set(selected).union(bypass).intersection(candidate_set)
        return SemanticRouterResult(sorted(selected_set), False, "ok", {n: s for n, s in ranked})
