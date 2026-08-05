from __future__ import annotations

import pytest

from helion_rag.rerank import Candidate
from helion_rag.rerank import parse_shape_features
from helion_rag.rerank import rerank

_F32 = "['torch.float32', 'torch.float32']"


def _cand(score, shapes, dtypes=_F32, name="k"):
    return Candidate(
        semantic_score=score, input_shapes=shapes, dtypes=dtypes, payload={"name": name}
    )


# --- parsing ---------------------------------------------------------------
def test_parse_features_ok():
    feat = parse_shape_features("[(16, 32)]", "['torch.float32']")
    assert feat is not None
    assert feat.args[0].rank == 2
    assert feat.args[0].dtype_category == "float"
    assert feat.args[0].dtype_size == 4
    assert feat.args[0].dims == (16, 32)


def test_parse_features_symbolic_returns_none():
    assert parse_shape_features("[(s0, 32)]", "['torch.float32']") is None
    assert parse_shape_features("not a shape", "['torch.float32']") is None


# --- semantic_only ---------------------------------------------------------
def test_semantic_only_preserves_similarity_order():
    cands = [_cand(0.7, "[(8, 8)]"), _cand(0.9, "[(9, 9)]"), _cand(0.8, "[(7, 7)]")]
    res = rerank("[(8, 8)]", "['torch.float32']", cands, rule="semantic_only", k=3)
    assert res.stratum == "semantic"
    assert [n.candidate.semantic_score for n in res.neighbors] == [0.9, 0.8, 0.7]


# --- symbolic fallback -----------------------------------------------------
def test_symbolic_query_falls_back_to_semantic():
    cands = [_cand(0.7, "[(8, 8)]"), _cand(0.9, "[(9, 9)]")]
    res = rerank("[(s0, 8)]", "['torch.float32']", cands, rule="shape_log_l1", k=2)
    assert res.stratum == "symbolic_fallback"
    assert [n.candidate.semantic_score for n in res.neighbors] == [0.9, 0.7]


# --- categorical hard filter -----------------------------------------------
def test_categorical_filter_drops_incompatible_rank_and_dtype():
    q_shapes = "[(16, 16)]"
    q_dtypes = "['torch.float32']"
    good = _cand(0.5, "[(16, 16)]", q_dtypes, name="good")
    wrong_rank = _cand(0.99, "[(16, 16, 16)]", q_dtypes, name="wrong_rank")
    wrong_dtype = _cand(0.99, "[(16, 16)]", "['torch.int32']", name="wrong_dtype")
    res = rerank(
        q_shapes, q_dtypes, [good, wrong_rank, wrong_dtype], rule="shape_log_l1", k=5
    )
    names = [n.candidate.payload["name"] for n in res.neighbors]
    assert names == ["good"]  # incompatible candidates filtered out
    assert res.n_after_filter == 1


# --- shape_lexicographic: prefer closest-lower --------------------------------
def test_shape_lexicographic_prefers_largest_that_fits():
    q = "[(100,)]"
    lower_far = _cand(0.5, "[(10,)]", "['torch.float32']", name="lower_far")
    lower_near = _cand(0.5, "[(90,)]", "['torch.float32']", name="lower_near")
    higher = _cand(0.99, "[(200,)]", "['torch.float32']", name="higher")
    res = rerank(
        q,
        "['torch.float32']",
        [lower_far, higher, lower_near],
        rule="shape_lexicographic",
        k=3,
    )
    names = [n.candidate.payload["name"] for n in res.neighbors]
    # all-lower candidates first (largest that fits), then the higher one
    assert names == ["lower_near", "lower_far", "higher"]


# --- shape_log_l1: nearest in log space ------------------------------------
def test_shape_log_l1_orders_by_log_distance():
    q = "[(64,)]"
    near = _cand(0.1, "[(64,)]", "['torch.float32']", name="near")
    far = _cand(0.99, "[(4096,)]", "['torch.float32']", name="far")
    res = rerank(q, "['torch.float32']", [far, near], rule="shape_log_l1", k=2)
    names = [n.candidate.payload["name"] for n in res.neighbors]
    assert names == ["near", "far"]
    assert res.neighbors[0].shape_score == pytest.approx(0.0)


# --- hybrid: blends semantic and shape -------------------------------------
def test_hybrid_weight_extremes():
    q = "[(64,)]"
    # shape-near but semantically weak vs shape-far but semantically strong.
    shape_near = _cand(0.1, "[(64,)]", "['torch.float32']", name="shape_near")
    sem_strong = _cand(0.99, "[(4096,)]", "['torch.float32']", name="sem_strong")
    cands = [shape_near, sem_strong]

    # weight=1.0 -> pure semantic distance -> semantically strong wins.
    r_sem = rerank(q, "['torch.float32']", cands, rule="hybrid", k=2, hybrid_weight=1.0)
    assert r_sem.neighbors[0].candidate.payload["name"] == "sem_strong"

    # weight=0.0 -> pure shape distance -> shape-near wins.
    r_shape = rerank(
        q, "['torch.float32']", cands, rule="hybrid", k=2, hybrid_weight=0.0
    )
    assert r_shape.neighbors[0].candidate.payload["name"] == "shape_near"


def test_hybrid_records_the_normalized_score_used_for_ordering():
    q = "[(64,)]"
    shape_near = _cand(0.1, "[(64,)]", "['torch.float32']", name="shape_near")
    sem_strong = _cand(0.99, "[(4096,)]", "['torch.float32']", name="sem_strong")

    result = rerank(
        q,
        "['torch.float32']",
        [shape_near, sem_strong],
        rule="hybrid",
        k=2,
        hybrid_weight=0.25,
    )

    assert [neighbor.combined_score for neighbor in result.neighbors] == pytest.approx(
        [0.25, 0.75]
    )


# --- hardware / config hard filters ----------------------------------------
def test_hardware_and_config_predicates_filter():
    cands = [_cand(0.9, "[(8, 8)]", name="a"), _cand(0.8, "[(8, 8)]", name="b")]
    res = rerank(
        "[(8, 8)]",
        _F32,
        cands,
        rule="semantic_only",
        k=5,
        hardware_ok=lambda c: c.payload["name"] == "a",
    )
    assert [n.candidate.payload["name"] for n in res.neighbors] == ["a"]


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        rerank("[(8,)]", "['torch.float32']", [], rule="nonsense", k=1)
