"""Keep each K-column carry in its owning lane until the terminal warp sum."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ..ast_extension import statement_from_string
from .autotune.config_keys import ROW_DOT_KEY
from .fragment_context import Fragment
from .row_carry_terms import carry_tile
from .row_carry_terms import carry_tile_loop
from .row_carry_terms import empty_loop
from .row_carry_terms import evaluate_terms
from .warp_carry_packet import WarpCarryPacket
from .warp_carry_packet import warp_carry_tile
from .warp_row_dot import row_dot_or_scalar

if TYPE_CHECKING:
    import ast

    from torch.fx import Node

    from ..device_ir import GraphInfo
    from .computed_fragment import FragmentCompiler
    from .row_reduction_carry import RowLoop


def warp_row_carries(
    c: FragmentCompiler,
    region: list[GraphInfo],
    plan: RowLoop,
    mapped: dict[Node, object],
    states: list[Fragment],
    tile: int,
) -> list[Fragment] | None:
    """Fuse independent carries and their sole sums under the existing warp policy.

    One warp owns a row. Lane l retains columns l, l+32, ... in registers for
    the full tiled loop. Each column observes its ordinary FP32 accumulation
    order; the terminal masked sum uses the existing lane fold/shuffle order.
    """
    if (
        c.block_volume is None  # Only shared launches own a physical block here.
        or not warp_carry_tile(tile)
        or any(s.shape != states[0].shape for s in states)
        or any(r.reduction in c.warp_result_nodes for r in plan.results)
        or c.warp_producer_recorder is not None
        or any(
            n in c.cache_nodes
            for carry in plan.results
            for n in (
                cast("Node", carry.reduction.args[0]),
                *cast("Node", carry.reduction.args[0]).all_input_nodes,
            )
        )
    ):
        return None
    c.df.cute_state.lanes.shared_fragment_uses_warps = True
    c.flush_pending_barrier()
    held = [v for v in mapped.values() if isinstance(v, Fragment)]
    c.held.append(held)
    outputs: list[Fragment] = []
    rows = states[0].shape[0]
    try:
        for _carry in plan.results:
            out = c.allocate(Fragment((rows,), torch.float32, lambda _: "0"))
            outputs.append(out)
            held.append(out)
    finally:
        c.held.pop()

    row = c.df.new_var("fragment_carry_row")
    lane = c.df.new_var("fragment_carry_lane")
    step = c.df.new_var("fragment_carry_tile")
    slot = c.df.new_var("fragment_carry_slot")
    count = tile // 32
    walk = empty_loop(
        f"for {row} in range({c.thread} // 32, {rows}, {c.block_volume} // 32)"
    )
    sections = carry_tile_loop(c, plan, step, tile)
    columns = empty_loop(f"for {slot} in cutlass.range_constexpr({count})")
    scalar_column = f"({slot} * 32 + {lane})"
    packet = (
        WarpCarryPacket(
            plan.block_id,
            [],
            row,
            lane,
            c.df.new_var("fragment_carry_column"),
            slot,
            rows,
            count,
        )
        if c.df.config.get("cute_fragment_vector_io")
        and count > 1
        and plan.begin % count == 0
        else None
    )
    coords = (row, packet.column if packet is not None else scalar_column)
    regs = [c.df.new_var("fragment_carry_columns") for _carry in plan.results]
    needed = {n for carry in plan.results for n in carry.nodes}
    inner = plan.results[0].term.graph

    with c.cg.set_statements(cast("list[ast.AST]", walk.body)):
        c.emit(f"{lane} = {c.thread} % 32")
        for reg in regs:
            c.emit(f"{reg} = cute.make_rmem_tensor(({count},), cutlass.Float32)")
            c.emit(f"{reg}.fill(cutlass.Float32(0))")
        with (
            carry_tile(c, plan, step, tile),
            c.cg.set_statements(cast("list[ast.AST]", columns.body)),
        ):
            values = evaluate_terms(
                c,
                mapped,
                (n for n in inner.nodes if n in needed),
                region,
                carry_packet=packet,
            )
            for carry, reg in zip(plan.results, regs, strict=True):
                val = cast("Fragment", values[carry.term]).read(coords)
                c.emit(f"{reg}[{slot}] = {reg}[{slot}] + cutlass.Float32({val})")
        # Only change lane ownership if a read actually matched. Each
        # register still accumulates its original K column across iterations.
        fold_column = scalar_column
        if packet is not None:
            if packet.setup:
                sections.body.extend(packet.setup)
                fold_column = f"({lane} * {count} + {slot})"
            columns.body.insert(
                0, statement_from_string(f"{packet.column} = {fold_column}")
            )
        sections.body.append(columns)
        c.cg.add_statement(sections)

        for carry, reg, state, out in zip(
            plan.results, regs, states, outputs, strict=True
        ):
            # Apply the original mask *after* the last K iteration. In
            # particular, input-load masks alone do not establish ownership of
            # a padded carry coordinate when its recipe maps zero to nonzero.
            resident = Fragment(
                state.shape,
                state.dtype,
                lambda _, r=reg: f"{r}[{slot}]",
                logical_domain=state.logical_domain,
            )
            masked = carry.after_last_tile(c, resident)
            total = c.df.new_var("fragment_carry_sum")
            c.emit(f"{total} = cutlass.Float32(0)")
            fold = empty_loop(f"for {slot} in cutlass.range_constexpr({count})")
            with c.cg.set_statements(cast("list[ast.AST]", fold.body)):
                value = masked.read((row, fold_column))
                c.emit(f"{total} = {total} + cutlass.Float32({value})")
            c.cg.add_statement(fold)
            for distance in (16, 8, 4, 2, 1):
                c.emit(
                    f"{total} = {total} + cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                )
            c.emit(f"if {lane} == 0:\n    {out.read((row,))} = {total}")
            c.folded_row_sums[carry.reduction] = out
    if packet is not None and c.df.config.get(ROW_DOT_KEY):
        c.cg.add_statement(
            row_dot_or_scalar(c, plan, states, outputs, packet, walk, step)
        )
    else:
        c.cg.add_statement(walk)
    c.synchronize()

    def escaped(_: tuple[str, ...]) -> str:
        raise exc.InvalidConfig("a folded row carry must be consumed by its row sum")

    # Keep the reduced allocation alive through getitem, phi and mask. The
    # caller consumes it when it encounters the exact proven reduction node.
    return [
        Fragment(
            s.shape,
            s.dtype,
            escaped,
            dependencies=(out,),
            logical_domain=s.logical_domain,
        )
        for s, out in zip(states, outputs, strict=True)
    ]
