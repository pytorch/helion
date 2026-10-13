"""Two physical K64 publications for the final initialized K128 contraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from . import chained_matmul as chain
from .chained_initialized_accumulator import classify_initialized_accumulator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import GraphInfo


@dataclass(frozen=True)
class KSchedule:
    mode: str
    stage: int = 1
    shape: tuple[int, int] = (128, 128)
    half_width: int = 64
    # The existing K_SW128 constructor maps each K64 panel to a disjoint
    # 16384-byte span. No allocation is shrunk and no half is overwritten.
    byte_spans: tuple[tuple[int, int], ...] = ((0, 16384), (16384, 32768))


def has_k_schedule_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Structural discovery only; resolved dimensions and leaf proof follow."""
    root = chain._root_graph(graphs)
    if root is None:
        return False
    selected = classify_initialized_accumulator(tuple(root.graph.nodes))
    if selected is None:
        return False
    left, right = (cast("Node", arg) for arg in selected.second.args[:2])
    reduction = left.meta["val"].shape[-1]
    return (
        isinstance(reduction, int)
        and reduction == 128
        and not chain._direct_operand(left)
        and not chain._ancestors(left) & {selected.first, selected.second}
        and chain._direct_operand(right)
    )


def resolve_k_schedule(plan: chain.ChainedMatmulPlan, mode: str) -> KSchedule | None:
    """Preserve all existing seed, arena, alias and full-tile admission gates."""
    if (
        mode not in ("serial64", "overlap64")
        or plan.strategy != "tcgen05_tmem"
        or plan.initialized_accumulator is None
        or plan.late_rhs_reuse is None
        or len(plan.dots) != 2
        or plan.threads != 128
        or plan.scan_exports
        or plan.direct_output
        or plan.dtype not in (torch.bfloat16, torch.float16)
        or any(size % block for _, size, block in plan.axes)
        or any(m != 128 or n % 32 or not 32 <= n <= 256 for m, n, _ in plan.shapes)
        or plan.shapes[-1][2] != 128
    ):
        return None
    left, right = (cast("Node", arg) for arg in plan.dots[-1].args[:2])
    if (
        chain._direct_operand(left)
        or chain._ancestors(left) & set(plan.dots)
        or not chain._direct_operand(right)
        or chain._ancestors(right) & {*plan.dots, *plan.scans}
    ):
        return None
    return KSchedule(mode)
