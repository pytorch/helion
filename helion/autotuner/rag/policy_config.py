"""Frozen retrieval policy consumed by the live RAG adapter.

Semantic pool size ``M``, final neighbor count ``K``, the shape-aware ranking
rule, and the optional hybrid weight. The adapter uses
:meth:`FrozenRetrievalPolicy.default` (M=8, K=5, ``shape_log_l1``); the head-to-head
campaign holds these fixed across arms so retrieval quality, not policy tuning, is
what the ``contextual_rag_llm`` arm measures.

See ``docs/rag_autotuning_experiment.md``.
"""

from __future__ import annotations

import dataclasses

# The ranking rules implemented by ``helion_rag.rerank``.
RANKING_RULES = frozenset(
    {"semantic_only", "shape_lexicographic", "shape_log_l1", "hybrid"}
)

# Defaults used by every arm of the campaign and by ordinary opt-in RAG runs.
DEFAULT_SEMANTIC_POOL_SIZE = 8
DEFAULT_FINAL_NEIGHBORS = 5
DEFAULT_RANKING_RULE = "shape_log_l1"


@dataclasses.dataclass(frozen=True)
class FrozenRetrievalPolicy:
    """Immutable retrieval hyperparameters for one frozen arm/generation."""

    semantic_pool_size: int = DEFAULT_SEMANTIC_POOL_SIZE
    final_neighbors: int = DEFAULT_FINAL_NEIGHBORS
    ranking_rule: str = DEFAULT_RANKING_RULE
    hybrid_weight: float | None = None
    threshold: float | None = None
    model_id: str | None = None
    tier0_identity: str | None = None

    def __post_init__(self) -> None:
        if self.semantic_pool_size < 1:
            raise ValueError("semantic_pool_size must be positive")
        if self.final_neighbors < 1:
            raise ValueError("final_neighbors must be positive")
        if self.final_neighbors > self.semantic_pool_size:
            raise ValueError("final_neighbors (K) must be <= semantic_pool_size (M)")
        if self.ranking_rule not in RANKING_RULES:
            raise ValueError(f"unsupported ranking rule {self.ranking_rule!r}")
        if self.ranking_rule == "hybrid":
            if self.hybrid_weight is None:
                raise ValueError("hybrid ranking rule requires a hybrid_weight")
            if not 0.0 <= self.hybrid_weight <= 1.0:
                raise ValueError("hybrid_weight must be in [0, 1]")
        elif self.hybrid_weight is not None:
            raise ValueError("hybrid_weight is only valid for the hybrid ranking rule")

    @classmethod
    def default(cls) -> FrozenRetrievalPolicy:
        """Return the legacy unpinned policy (M=8, K=5, shape_log_l1)."""
        return cls()
