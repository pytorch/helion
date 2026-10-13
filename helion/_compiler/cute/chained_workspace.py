"""Reusable FP32 storage for materialized contraction boundaries.

The schedule has separate input-read and output-publish events. A consumer's
result may overwrite a last-used input only after *all* operand and accumulator
reads finish. Emitters using this layout must synchronize between those events
and before advancing to the next stage. Outputs of one grouped issue coexist.

Only contraction results share this arena. Collective results, resident carries,
and A/B staging remain separate allocations, including across loop iterations.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import TYPE_CHECKING

from torch.fx import Node
from torch.utils._pytree import tree_leaves

from .warp_specialized_plan import SharedBufferRegion
from .warp_specialized_plan import SharedBufferRequest
from .warp_specialized_plan import SharedMemoryLayoutPlan

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .chained_matmul import ChainedMatmulPlan


_ALIGNMENT = 128


def _align(size: int) -> int:
    return (size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _reuse_requests(
    requests: tuple[SharedBufferRequest, ...],
) -> SharedMemoryLayoutPlan:
    """Deterministic first-fit placement, largest simultaneous outputs first."""
    placed: list[SharedBufferRegion] = []
    for request in sorted(
        requests, key=lambda item: (item.live_from, -item.byte_size, item.name)
    ):
        conflicts = sorted(
            (
                region
                for region in placed
                if region.live_from < request.live_until
                and request.live_from < region.live_until
            ),
            key=lambda region: region.byte_offset,
        )
        offset = 0
        for region in conflicts:
            if offset + request.byte_size <= region.byte_offset:
                break
            offset = _align(max(offset, region.byte_end))
        placed.append(
            SharedBufferRegion(
                request.name,
                offset,
                request.byte_size,
                request.live_from,
                request.live_until,
                _ALIGNMENT,
            )
        )
    by_name = {region.name: region for region in placed}
    return SharedMemoryLayoutPlan(
        tuple(by_name[request.name] for request in requests),
        _align(max((region.byte_end for region in placed), default=0)),
    )


def plan_contraction_workspace(
    plan: ChainedMatmulPlan,
    *,
    row_padding: int = 0,
    include_final: bool = True,
    skip_operand_reads: frozenset[tuple[int, int]] = frozenset(),
    omit_unused: bool = False,
) -> SharedMemoryLayoutPlan:
    """Allocate logical row-major C views named ``chain_{stage}_c``.

    Traverse expressions only as far as materialized boundaries: reading a dot
    or collective does not reevaluate its inputs. Store values, coordinates and
    predicates, scan exports, and loop carry updates all consume their inputs at
    the end of the body, regardless of their original FX node positions.

    The input plan and any contraction groups must already be admitted by the
    common planner. Physical MMA orientation/padding does not affect C storage.
    ``row_padding`` reserves additional FP32 elements after each logical row,
    for consumers whose shared C layout uses a padded row stride.
    A root emitter that drains its final result directly may omit that buffer
    with ``include_final=False``; all intermediate consumer lifetimes remain.
    A caller with independently proved typed operand bindings can exclude those
    specific A/B reads. Seed, collective and export reads are always retained.
    """
    if type(row_padding) is not int or row_padding < 0:
        raise ValueError("contraction row padding must be a nonnegative integer")
    assert plan.region is not None
    assert len(plan.dots) == len(plan.shapes)
    assert all(
        0 <= stage < len(plan.dots) and operand in (0, 1)
        for stage, operand in skip_operand_reads
    )
    positions = {node: index for index, node in enumerate(plan.region.nodes)}
    dot_positions = [positions[node] for node in plan.dots]
    collectives = set(plan.region.scans) | set(plan.region.reductions)
    stage_by_node = {node: stage for stage, node in enumerate(plan.dots)}
    units = (
        tuple(group.stages for group in plan.contraction_groups)
        if plan.contraction_groups is not None
        else tuple((stage,) for stage in range(len(plan.dots)))
    )
    assert tuple(stage for unit in units for stage in unit) == tuple(
        range(len(plan.dots))
    )
    first_stage = {stage: unit[0] for unit in units for stage in unit}
    starts = [2 * first_stage[stage] + 1 for stage in range(len(plan.dots))]
    ends = [start + 1 for start in starts]
    used: set[int] = set()

    def consume(roots: Iterable[Node], event: int) -> None:
        pending = list(roots)
        visited: set[Node] = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in stage_by_node:
                stage = stage_by_node[node]
                used.add(stage)
                assert starts[stage] < event, "read before contraction publication"
                ends[stage] = max(ends[stage], event + 1)
            elif node not in collectives:
                pending.extend(node.all_input_nodes)

    for unit in units:
        read_event = 2 * unit[0]
        for stage in unit:
            if skip_operand_reads:
                dot = plan.dots[stage]
                consume(
                    (
                        node
                        for node in tree_leaves(
                            (
                                tuple(
                                    arg
                                    for index, arg in enumerate(dot.args)
                                    if (stage, index) not in skip_operand_reads
                                ),
                                dot.kwargs,
                            )
                        )
                        if isinstance(node, Node)
                    ),
                    read_event,
                )
            else:
                consume(plan.dots[stage].all_input_nodes, read_event)
    for node in collectives:
        # This is the same source-order cut used by emit_collectives_before.
        stage = bisect_right(dot_positions, positions[node])
        consume(node.all_input_nodes, 2 * stage)
    if plan.pointwise_cache is not None:
        # Cache publication still evaluates the original recipe when a caller
        # independently replaces later A/B reads with typed shared images.
        # Keep its FP32 dependencies until that publication, including nested
        # cache ancestry conservatively; the cache has separate storage.
        for entry in plan.pointwise_cache.entries:
            consume((entry.node,), 2 * entry.first_stage)
    final_event = 2 * len(plan.dots)
    consume(plan.region.stores, final_event)
    consume((plan.store,), final_event)
    consume((export.store for export in plan.scan_exports), final_event)
    consume(plan.region.live_outs, final_event)
    if plan.loop is not None:
        consume((carry.output for carry in plan.region.carries), final_event)
    return _reuse_requests(
        tuple(
            SharedBufferRequest(
                f"chain_{stage}_c",
                4 * rows * (columns + row_padding),
                _ALIGNMENT,
                starts[stage],
                ends[stage],
            )
            for stage, (rows, columns, _) in enumerate(plan.shapes)
            if (include_final or stage < len(plan.dots) - 1)
            and (not omit_unused or stage in used)
        )
    )
