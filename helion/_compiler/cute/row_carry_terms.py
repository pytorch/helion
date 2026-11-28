"""Build the row-carry tile loop and evaluate its read-only terms."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import cast

from torch._inductor.virtualized import V

from ...language import memory_ops
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    import ast
    from collections.abc import Iterable
    from collections.abc import Iterator

    import torch
    from torch.fx import Node

    from ..device_ir import GraphInfo
    from .computed_fragment import FragmentCompiler
    from .row_reduction_carry import RowLoop
    from .warp_carry_packet import WarpCarryPacket


def empty_loop(header: str) -> ast.For:
    loop = cast("ast.For", statement_from_string(f"{header}:\n    pass"))
    loop.body.clear()
    return loop


def empty_if(condition: str) -> ast.If:
    branch = cast("ast.If", statement_from_string(f"if {condition}:\n    pass"))
    branch.body.clear()
    return branch


def can_unroll_carry(trips: int) -> bool:
    from .fragment_context import UNROLLED_SLOTS

    return 1 < trips <= UNROLLED_SLOTS


def carry_tile_loop(
    c: FragmentCompiler, plan: RowLoop, step: str, tile: int
) -> ast.For:
    """An optionally unrolled K loop with the same lower bound and tile offsets."""
    trips = (plan.end - plan.begin) // tile
    if c.df.config.get("cute_fragment_unrolled_loops") and can_unroll_carry(trips):
        section = c.df.new_var("fragment_carry_section")
        loop = empty_loop(f"for {section} in cutlass.range_constexpr({trips})")
        loop.body.append(
            statement_from_string(f"{step} = {plan.begin} + {section} * {tile}")
        )
        return loop
    return empty_loop(f"for {step} in range({plan.begin}, {plan.end}, {tile})")


@contextmanager
def carry_tile(
    c: FragmentCompiler, plan: RowLoop, step: str, tile: int | torch.SymInt
) -> Iterator[None]:
    from .scalar_fragment_assertion import checked_range_domain

    # This lowering emits its own K loop. As in the generic loop, the
    # interval proves packet alignment only for a step of one full tile.
    offsets, bounds, intervals = c.offsets, c.bounds, c.tile_intervals
    axis = plan.block_id
    domains = c.checked_domains
    # Validate the same range as the ordinary K loop before rebinding its
    # induction variable. A checked load must prove its own tensor bounds.
    domain = (
        checked_range_domain(c, plan.begin, plan.end, tile) if c.checked_root else None
    )
    c.offsets = {**offsets, axis: step}
    c.bounds = {**bounds, axis: str(plan.end)}
    c.tile_intervals = intervals.copy()
    try:
        if domain is not None:
            c.checked_domains = {**domains, step: domain}
        width = c.df.resolved_block_size(axis)
        if type(tile) is int and type(width) is int and tile == width:
            c.record_tile_interval(axis, plan.begin, plan.end)
        yield
    finally:
        c.offsets, c.bounds, c.tile_intervals = offsets, bounds, intervals
        c.checked_domains = domains


def evaluate_terms(
    c: FragmentCompiler,
    mapped: dict[Node, object],
    nodes: Iterable[Node],
    region: list[GraphInfo],
    *,
    carry_packet: WarpCarryPacket | None = None,
) -> dict[Node, object]:
    values = dict(mapped)
    for node in nodes:
        with node.meta["location"], V.set_current_node(node):
            values[node] = (
                c.memory(
                    node,
                    values,
                    False,
                    defer_load=True,
                    deferred_read_region=region,
                    carry_packet=carry_packet,
                )
                if node.target is memory_ops.load
                else c.node(node, values)
            )
    return values
