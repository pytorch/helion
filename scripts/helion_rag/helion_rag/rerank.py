"""Shape-aware reranking of the semantic candidate pool (§6.3).

Retrieval is two stages: Qwen returns a semantic pool, then hard filters and a
frozen shape-distance rule rerank the survivors and take the top-K. The rules are:

- ``semantic_only``      — keep Qwen's similarity order.
- ``shape_lexicographic``— categorical exact match, then prefer candidates whose
  numeric dims are all <= the query (closest lower before closest higher). Mirrors
  the conservative direction in ``helion/autotuner/nearest_neighbor_backend.py``.
- ``shape_log_l1``       — minimize mean |log2(1+dim)| difference over aligned
  dims, with a frozen penalty for missing/extra dims.
- ``hybrid``             — a frozen weighted blend of normalized semantic and
  shape_log_l1 distance.

Dynamic / symbolic / missing / unparsable query shapes fall back to
``semantic_only`` and are reported as a separate ``symbolic_fallback`` stratum.
Raw scores and rank-before/after are recorded for instrumentation.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from collections.abc import Callable
from collections.abc import Sequence

RULES = ("semantic_only", "shape_lexicographic", "shape_log_l1", "hybrid")
_MISSING_DIM_PENALTY = 4.0  # frozen per missing/extra dimension (log2 units)
_DEFAULT_HYBRID_WEIGHT = 0.5

# dtype string -> (category, element size in bytes); unknown -> ("other", 0).
_DTYPE_INFO: dict[str, tuple[str, int]] = {
    "torch.float64": ("float", 8),
    "torch.float32": ("float", 4),
    "torch.float16": ("float", 2),
    "torch.bfloat16": ("float", 2),
    "torch.int64": ("int", 8),
    "torch.int32": ("int", 4),
    "torch.int16": ("int", 2),
    "torch.int8": ("int", 1),
    "torch.uint8": ("uint", 1),
    "torch.bool": ("bool", 1),
    "torch.complex64": ("complex", 8),
    "torch.complex128": ("complex", 16),
}


@dataclasses.dataclass(frozen=True)
class ArgFeatures:
    dtype: str
    dtype_category: str
    dtype_size: int
    rank: int
    dims: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ShapeFeatures:
    args: tuple[ArgFeatures, ...]


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One semantic-pool candidate to be reranked."""

    semantic_score: float
    input_shapes: str
    dtypes: str
    payload: dict


@dataclasses.dataclass(frozen=True)
class ScoredNeighbor:
    candidate: Candidate
    semantic_score: float
    shape_score: float | None
    combined_score: float
    rank_before: int
    rank_after: int


@dataclasses.dataclass(frozen=True)
class RerankResult:
    neighbors: tuple[ScoredNeighbor, ...]
    stratum: str  # "shape_aware" | "semantic" | "symbolic_fallback"
    n_candidates: int
    n_after_filter: int


def parse_shape_features(shapes: str, dtypes: str) -> ShapeFeatures | None:
    """Parse the stored shape/dtype reprs; return None if dynamic/symbolic/unparsable."""
    try:
        shape_list = ast.literal_eval(shapes)
        dtype_list = ast.literal_eval(dtypes)
    except (ValueError, SyntaxError, TypeError):
        return None
    if not isinstance(shape_list, (list, tuple)):
        return None
    args: list[ArgFeatures] = []
    for i, sh in enumerate(shape_list):
        dims = tuple(sh) if isinstance(sh, (list, tuple)) else ()
        if not all(isinstance(d, int) for d in dims):
            return None  # a symbolic/non-int dimension
        dtype = str(dtype_list[i]) if i < len(dtype_list) else ""
        category, size = _DTYPE_INFO.get(dtype, ("other", 0))
        args.append(
            ArgFeatures(
                dtype=dtype,
                dtype_category=category,
                dtype_size=size,
                rank=len(dims),
                dims=dims,
            )
        )
    return ShapeFeatures(args=tuple(args))


def _categorical_match(q: ShapeFeatures, c: ShapeFeatures) -> bool:
    """Exact match on arg count and each arg's dtype/category/size/rank."""
    if len(q.args) != len(c.args):
        return False
    return all(
        (qa.dtype, qa.dtype_category, qa.dtype_size, qa.rank)
        == (ca.dtype, ca.dtype_category, ca.dtype_size, ca.rank)
        for qa, ca in zip(q.args, c.args)
    )


def _flat_dims(feat: ShapeFeatures) -> list[int]:
    return [d for arg in feat.args for d in arg.dims]


def _lexicographic_key(q: ShapeFeatures, c: ShapeFeatures) -> tuple[int, float]:
    """Conservative direction: all-dims-<= first (largest that fits), else closest higher."""
    q_dims = _flat_dims(q)
    c_dims = _flat_dims(c)
    total = float(sum(c_dims))
    all_le = all(cd <= qd for qd, cd in zip(q_dims, c_dims))
    return (0, -total) if all_le else (1, total)


def _log_l1_distance(q: ShapeFeatures, c: ShapeFeatures) -> float:
    """Mean |log2(1+dim)| difference over aligned dims + penalty for missing/extra."""
    total = 0.0
    n = 0
    penalty = 0
    for qa, ca in zip(q.args, c.args):
        aligned = min(len(qa.dims), len(ca.dims))
        for i in range(aligned):
            total += abs(math.log2(1 + qa.dims[i]) - math.log2(1 + ca.dims[i]))
            n += 1
        penalty += abs(len(qa.dims) - len(ca.dims))
    mean = total / n if n else 0.0
    return mean + _MISSING_DIM_PENALTY * penalty


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _semantic_only(
    candidates: Sequence[Candidate], k: int, stratum: str
) -> RerankResult:
    order = sorted(range(len(candidates)), key=lambda i: -candidates[i].semantic_score)
    neighbors = tuple(
        ScoredNeighbor(
            candidate=candidates[i],
            semantic_score=candidates[i].semantic_score,
            shape_score=None,
            combined_score=-candidates[i].semantic_score,
            rank_before=rank,
            rank_after=rank,
        )
        for rank, i in enumerate(order[:k])
    )
    return RerankResult(neighbors, stratum, len(candidates), len(candidates))


def rerank(
    query_shapes: str,
    query_dtypes: str,
    candidates: Sequence[Candidate],
    *,
    rule: str,
    k: int,
    hybrid_weight: float = _DEFAULT_HYBRID_WEIGHT,
    hardware_ok: Callable[[Candidate], bool] | None = None,
    config_ok: Callable[[Candidate], bool] | None = None,
) -> RerankResult:
    """Hard-filter then rerank the semantic pool; return the top-``k`` (§6.3)."""
    if rule not in RULES:
        raise ValueError(f"unknown ranking rule {rule!r}; expected one of {RULES}")

    survivors = [
        c
        for c in candidates
        if (hardware_ok is None or hardware_ok(c))
        and (config_ok is None or config_ok(c))
    ]

    q_feat = parse_shape_features(query_shapes, query_dtypes)
    if rule == "semantic_only":
        return _semantic_only(survivors, k, "semantic")
    if q_feat is None:
        return _semantic_only(survivors, k, "symbolic_fallback")

    # Shape rules additionally require categorical (dtype/rank/arity) compatibility.
    filtered: list[tuple[Candidate, ShapeFeatures]] = []
    for c in survivors:
        feat = parse_shape_features(c.input_shapes, c.dtypes)
        if feat is not None and _categorical_match(q_feat, feat):
            filtered.append((c, feat))

    sem_rank = {
        id(c): r
        for r, (c, _) in enumerate(
            sorted(filtered, key=lambda cf: -cf[0].semantic_score)
        )
    }

    if rule == "shape_lexicographic":
        ordered = sorted(
            filtered,
            key=lambda cf: (_lexicographic_key(q_feat, cf[1]), -cf[0].semantic_score),
        )
        shape_scores = [float(sum(_flat_dims(f))) for _, f in ordered]
        combined_scores = shape_scores
    elif rule == "shape_log_l1":
        ordered = sorted(
            filtered,
            key=lambda cf: (_log_l1_distance(q_feat, cf[1]), -cf[0].semantic_score),
        )
        shape_scores = [_log_l1_distance(q_feat, f) for _, f in ordered]
        combined_scores = shape_scores
    else:  # hybrid
        sem_dist = [1.0 - c.semantic_score for c, _ in filtered]
        shape_dist = [_log_l1_distance(q_feat, f) for _, f in filtered]
        sem_norm = _minmax(sem_dist)
        shape_norm = _minmax(shape_dist)
        combined = [
            hybrid_weight * s + (1.0 - hybrid_weight) * h
            for s, h in zip(sem_norm, shape_norm)
        ]
        order = sorted(range(len(filtered)), key=lambda i: combined[i])
        ordered = [filtered[i] for i in order]
        shape_scores = [shape_dist[i] for i in order]
        combined_scores = [combined[i] for i in order]

    neighbors = tuple(
        ScoredNeighbor(
            candidate=c,
            semantic_score=c.semantic_score,
            shape_score=shape_scores[rank_after],
            combined_score=combined_scores[rank_after],
            rank_before=sem_rank[id(c)],
            rank_after=rank_after,
        )
        for rank_after, (c, _) in enumerate(ordered[:k])
    )
    return RerankResult(neighbors, "shape_aware", len(candidates), len(filtered))
