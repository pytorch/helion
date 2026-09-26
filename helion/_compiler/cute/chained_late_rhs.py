"""Byte/phase plan for a late direct RHS in an initialized TCgen05 pair."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
from typing import cast

from . import chained_matmul as chain
from .chained_initialized_accumulator import classify_initialized_accumulator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import GraphInfo


@dataclass(frozen=True)
class LateRhsArenaPlan:
    """A/B remain distinct; final B survives until the residual is evaluated."""

    a_bytes: int
    b_bytes: int
    rhs_bytes: int
    output_bytes: int


def has_late_rhs_candidate(graphs: Sequence[GraphInfo]) -> bool:
    root = chain._root_graph(graphs)
    if root is None:
        return False
    selected = classify_initialized_accumulator(tuple(root.graph.nodes))
    return selected is not None and chain._direct_operand(
        cast("Node", selected.second.args[1])
    )


def _layout_bytes(shape: tuple[int, int], inner: int) -> int | None:
    """Exact cosize of the existing 16-bit, complete K/MN swizzle atoms.

    _layout tiles an eight-row atom with width min(128, lowbit(width*2))
    bytes. Complete atoms have no padding and preserve a dense byte span.
    Unknown/partial atoms are not an arena-reuse proof.
    """
    width = shape[inner]
    height = shape[1 - inner]
    if min(width, height) <= 0:
        return None
    atom_bytes = min(128, (width * 2) & -(width * 2))
    if atom_bytes < 32 or width * 2 % atom_bytes or height % 8:
        return None
    return math.prod(shape) * 2


def resolve_late_rhs(plan: chain.ChainedMatmulPlan) -> LateRhsArenaPlan | None:
    """Resolve sizes before SMEM admission; expression/consumer proof follows.

    Retain the old direct-prefetch envelope. This does not admit bridges,
    exports, FP32 output, partial tiles or a different initialized matcher.
    """
    from .chained_tcgen05 import _prefetch_final_b

    if (
        plan.initialized_accumulator is None
        or len(plan.dots) != 2
        or plan.scan_exports
        or not _prefetch_final_b(plan)
    ):
        return None
    a_bytes = 2 * max(m * k for m, _, k in plan.shapes)
    b_bytes = 2 * max(n * k for _, n, k in plan.shapes)
    m, n, k = plan.shapes[-1]
    rhs_bytes = _layout_bytes((n, k), 0)
    output_bytes = _layout_bytes((m, n), 1)
    if (
        rhs_bytes is None
        or output_bytes is None
        or rhs_bytes > b_bytes
        or output_bytes > a_bytes
    ):
        return None
    return LateRhsArenaPlan(a_bytes, b_bytes, rhs_bytes, output_bytes)
