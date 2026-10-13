"""Share resident loop carries with sequential materialized contraction results.

This is a resource plan, not permission to reorder the body or overlap iterations.
It uses the read/publication events from ``chained_workspace``. Every current
carry starts live before initialization, so simultaneously initialized carries
always have distinct storage, including when the loop executes zero times.

After the final read event, ``advance_carries`` must evaluate *all* next carries
into registers, synchronize, write *all* carry slots, and synchronize again.
The first barrier ends every current-carry/C lifetime before any write-back;
the second publishes the next iteration's initial state. Final exports read
that new state (or the initialized state for zero trips), not the dead body C
views. These cyclic lifetimes cannot be implemented by interleaving each next
carry's evaluation and write-back, even when no output reads an old carry.

Collectives, cached pointwise values, A/B staging and register next-carry images
remain separate allocations. The caller must recollect after graph rewrites.
"""

from __future__ import annotations

from bisect import bisect_right
import math
from typing import TYPE_CHECKING

import torch

from .chained_preparation_cut import _shape_input
from .chained_workspace import _ALIGNMENT
from .chained_workspace import _align
from .chained_workspace import _reuse_requests
from .chained_workspace import plan_contraction_workspace
from .warp_specialized_plan import SharedBufferRequest

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.fx import Node

    from .chained_matmul import ChainedMatmulPlan
    from .warp_specialized_plan import SharedMemoryLayoutPlan


def plan_loop_workspace(
    plan: ChainedMatmulPlan, carry_shapes: tuple[tuple[int, ...], ...]
) -> SharedMemoryLayoutPlan | None:
    """Plan compact FP32 carry/C views for an admitted common TCgen05 loop.

    ``carry_shapes`` must be the resolved logical shapes used by loop emission,
    in region carry order, not padded MMA shapes or host tensor storage spans.
    Concrete metadata dimensions are checked here; resolving symbolic tile
    dimensions remains the admitting caller's responsibility. No current compile
    environment, shape hints, graph rewrites, or emitted code are consulted.

    Original expressions conservatively determine last carry readers, even
    when pointwise residency would avoid some reads. Only already-materialized
    dots/collectives and validated shape metadata stop dependency traversal.
    Return ``None`` for an unsupported contract or a greedy allocation larger
    than separate carry storage plus the existing contraction arena.
    """
    loop, region = plan.loop, plan.region
    if (
        loop is None
        or region is None
        or region is not loop.region
        or region.graph is not loop.body.graph
        or tuple(region.graph.nodes) != region.nodes
        or plan.strategy != "tcgen05_tmem"
        or plan.direct_output
        or plan.initialized_accumulator is not None
        or plan.late_rhs_reuse is not None
        or plan.k_schedule is not None
        or not region.carries
        or len(carry_shapes) != len(region.carries)
        or not plan.dots
        or plan.dots != tuple(spec.node for spec in region.contractions)
        or len(plan.dots) != len(plan.shapes)
        or any(
            len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape)
            for shape in plan.shapes
        )
    ):
        return None
    positions = {node: index for index, node in enumerate(region.nodes)}
    if any(
        node.graph is not region.graph
        or any(
            positions.get(source, len(positions)) >= index
            for source in node.all_input_nodes
        )
        for index, node in enumerate(region.nodes)
    ):
        return None
    carries = {carry.input: index for index, carry in enumerate(region.carries)}
    names = tuple(loop.carry_name(carry.input_index) for carry in region.carries)
    if len(carries) != len(region.carries) or len(set(names)) != len(names):
        return None
    for carry, shape in zip(region.carries, carry_shapes, strict=True):
        if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
            return None
        for node in (carry.input, carry.output):
            value = node.meta.get("val")
            if (
                node not in positions
                or not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.ndim != 2
                or any(
                    type(size) is int and size != resolved
                    for size, resolved in zip(value.shape, shape, strict=True)
                )
            ):
                return None
    units = (
        tuple(group.stages for group in plan.contraction_groups)
        if plan.contraction_groups is not None
        else tuple((stage,) for stage in range(len(plan.dots)))
    )
    if any(not unit for unit in units) or tuple(
        stage for unit in units for stage in unit
    ) != tuple(range(len(plan.dots))):
        return None
    starts = {plan.dots[stage]: 2 * unit[0] + 1 for unit in units for stage in unit}
    collectives = set(region.scans) | set(region.reductions)
    dot_positions = [positions[node] for node in plan.dots]
    # Group admission normally forbids intervening collectives. Do not invent
    # an event order for a group whose physical issue crosses such a boundary.
    if any(
        dot_positions[unit[0]] < positions[node] < dot_positions[unit[-1]]
        for unit in units
        for node in collectives
    ):
        return None
    ends = [0] * len(region.carries)

    def consume(roots: Iterable[Node], event: int) -> bool:
        pending = list(roots)
        visited: set[Node] = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            if node not in positions:
                return False
            if node in carries:
                index = carries[node]
                ends[index] = max(ends[index], event + 1)
            elif node in starts:
                if starts[node] >= event:
                    return False  # Includes illegal dependencies within a group.
            elif node.target is torch.ops.aten.sym_size.int:
                if not _shape_input(node):
                    return False
            elif node not in collectives:
                pending.extend(node.all_input_nodes)
        return True

    for unit in units:
        for stage in unit:
            if not consume(plan.dots[stage].all_input_nodes, 2 * unit[0]):
                return None
    for node in collectives:
        stage = bisect_right(dot_positions, positions[node])
        if not consume(node.all_input_nodes, 2 * stage):
            return None
    final_event = 2 * len(plan.dots)
    if not consume(
        (
            *region.stores,
            plan.store,
            *(export.store for export in plan.scan_exports),
            *region.live_outs,
            *(carry.output for carry in region.carries),
        ),
        final_event,
    ):
        return None
    c_layout = plan_contraction_workspace(plan)
    carry_bytes = tuple(4 * math.prod(shape) for shape in carry_shapes)
    result = _reuse_requests(
        tuple(
            SharedBufferRequest(name, size, _ALIGNMENT, -1, end)
            for name, size, end in zip(names, carry_bytes, ends, strict=True)
        )
        + tuple(
            SharedBufferRequest(
                item.name,
                item.byte_size,
                item.alignment,
                item.live_from,
                item.live_until,
            )
            for item in c_layout.regions
        )
    )
    separate_bytes = c_layout.allocated_bytes + sum(map(_align, carry_bytes))
    return result if result.allocated_bytes <= separate_bytes else None
