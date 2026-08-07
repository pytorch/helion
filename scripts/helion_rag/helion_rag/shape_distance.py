"""Shape proximity for re-ranking source-similar workloads."""

from __future__ import annotations

import ast
import math


def _parse_shapes(value: str) -> tuple[tuple[int, ...], ...] | None:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, (list, tuple)):
        return None
    shapes = []
    for shape in parsed:
        if not isinstance(shape, (list, tuple)) or not all(
            isinstance(dim, int) and dim > 0 for dim in shape
        ):
            return None
        shapes.append(tuple(shape))
    return tuple(shapes)


def shape_distance(query: str, candidate: str) -> float:
    """Return log2-L2 distance, or infinity for incomparable structures."""
    query_shapes = _parse_shapes(query)
    candidate_shapes = _parse_shapes(candidate)
    if query_shapes is None or candidate_shapes is None:
        return math.inf
    if len(query_shapes) != len(candidate_shapes) or any(
        len(left) != len(right) for left, right in zip(query_shapes, candidate_shapes)
    ):
        return math.inf
    return math.sqrt(
        sum(
            (math.log2(left) - math.log2(right)) ** 2
            for query_shape, candidate_shape in zip(query_shapes, candidate_shapes)
            for left, right in zip(query_shape, candidate_shape)
        )
    )


def shape_relevance(distance: float) -> float:
    """Map distance to a bounded closeness score."""
    return math.exp(-distance) if math.isfinite(distance) else 0.0
