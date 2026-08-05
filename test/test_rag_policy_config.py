"""Tests for the frozen retrieval policy (Phase 0, §6.3/§7)."""

from __future__ import annotations

import pytest

from helion.autotuner.rag.policy_config import FrozenRetrievalPolicy


def test_default_matches_legacy_constants() -> None:
    policy = FrozenRetrievalPolicy.default()
    assert policy.semantic_pool_size == 8
    assert policy.final_neighbors == 5
    assert policy.ranking_rule == "shape_log_l1"
    assert policy.hybrid_weight is None
    assert policy.threshold is None
    assert policy.tier0_identity is None


def test_rejects_final_greater_than_pool() -> None:
    with pytest.raises(ValueError, match="final_neighbors"):
        FrozenRetrievalPolicy(semantic_pool_size=4, final_neighbors=8)


def test_rejects_unknown_ranking_rule() -> None:
    with pytest.raises(ValueError, match="unsupported ranking rule"):
        FrozenRetrievalPolicy(ranking_rule="magic")


def test_hybrid_requires_weight() -> None:
    with pytest.raises(ValueError, match="hybrid_weight"):
        FrozenRetrievalPolicy(ranking_rule="hybrid")


def test_hybrid_weight_only_for_hybrid() -> None:
    with pytest.raises(ValueError, match="only valid for the hybrid"):
        FrozenRetrievalPolicy(ranking_rule="shape_log_l1", hybrid_weight=0.5)


def test_hybrid_weight_range() -> None:
    with pytest.raises(ValueError, match="hybrid_weight must be in"):
        FrozenRetrievalPolicy(ranking_rule="hybrid", hybrid_weight=1.5)
