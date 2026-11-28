"""Feed a BF16 row dot's matrix and broadcast vector packets to a warp MMA."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from ._ast_pass_utils import deep_clone
from ._ast_pass_utils import substitute_names
from .fragment_context import Fragment
from .fragment_vector_io import PACKET_BYTES
from .fragment_vector_io import copy_atom
from .row_carry_terms import carry_tile
from .row_carry_terms import carry_tile_loop
from .row_carry_terms import empty_if
from .row_carry_terms import empty_loop
from .warp_row_dot_spec import BF16_PACKET_LANES
from .warp_row_dot_spec import ROW_DOT_K
from .warp_row_dot_spec import ROW_DOT_K_GROUP
from .warp_row_dot_spec import ROW_DOT_ROWS
from .warp_row_dot_spec import ROW_DOT_WARP
from .warp_row_dot_spec import admitted_row_dot_loads

if TYPE_CHECKING:
    from .computed_fragment import FragmentCompiler
    from .row_reduction_carry import RowLoop
    from .warp_carry_packet import CarryPacketSource
    from .warp_carry_packet import WarpCarryPacket


def row_dot_or_scalar(
    c: FragmentCompiler,
    plan: RowLoop,
    states: list[Fragment],
    outputs: list[Fragment],
    packet: WarpCarryPacket,
    scalar: ast.For,
    step: str,
) -> ast.stmt:
    """Keep the scalar walk for unsupported terms, masks or pointer alignment."""
    loads = admitted_row_dot_loads(CompileEnvironment.current(), plan)
    if (
        loads is None
        or packet.lanes != BF16_PACKET_LANES
        or packet.rows % ROW_DOT_ROWS
        or c.block_volume is None
        or any(n not in packet.sources for n in loads)
    ):
        return scalar
    # A terminal _mask_to may discard part of a carry after the last
    # iteration. An MMA reduces K before the store, so it may keep only
    # a row mask. Ask the original mask emitter with a symbolic K coordinate.
    (state,) = states
    (output,) = outputs
    result = plan.results[0]
    row, col, value = (c.df.new_var("row_dot_" + s) for s in ("row", "col", "value"))
    steps: list[ast.AST] = []
    probe = Fragment(
        state.shape, torch.float32, lambda _: value, logical_domain=state.logical_domain
    )
    with c.cg.set_statements(steps):
        masked = result.after_last_tile(c, probe)
        c.emit(f"{output.read((row,))} = {masked.read((row, col))}")
    if any(isinstance(n, ast.Name) and n.id == col for s in steps for n in ast.walk(s)):
        return scalar

    a, b = (packet.sources[n] for n in loads)

    guard = empty_if(
        f"cutlass.const_expr({a.base}.alignment >= {PACKET_BYTES} and {b.base}.alignment >= {PACKET_BYTES})"
    )
    guard.orelse = [scalar]

    def tile_window(source: CarryPacketSource, *coords: str) -> tuple[str, str]:
        # The matcher requires complete tiles starting at zero;
        # row_dot_windows proves a matching, unit-stride last dimension.
        # No other subscript may use K. Thus every inner packet
        # in this K tile has the same validity as its first column. Keep
        # the actual tile offset for captured indices and keep the two
        # matrix row masks independent for partial stripes.
        pointer, valid = source.address((*coords, "0"))
        base = c.df.new_var("row_dot_base")
        c.emit(f"{base} = {pointer}")
        return base, valid

    def read_words(window: tuple[str, str], k: str) -> str:
        base, valid = window
        buf = c.df.new_var("row_dot_packet")
        words = c.df.new_var("row_dot_words")
        c.emit(
            f"{buf} = cute.make_rmem_tensor(({BF16_PACKET_LANES},), cutlass.BFloat16)"
        )
        c.emit(f"{buf}.fill(cutlass.BFloat16(0))")
        c.emit(
            f"if {valid}:\n"
            f"    cute.copy({copy_atom('cutlass.BFloat16')}, "
            f"cute.make_tensor(({base} + {k}).align({PACKET_BYTES}), cute.make_layout(({BF16_PACKET_LANES},))), {buf})"
        )
        c.emit(f"{words} = cute.recast_tensor({buf}, cutlass.Uint32)")
        return words

    stripe, offset, k = (
        c.df.new_var("row_dot_" + s) for s in ("stripe", "offset", "k")
    )
    acc = [c.df.new_var("row_dot_acc") for _ in range(4)]
    walk = empty_loop(
        f"for {stripe} in range({c.thread} // {ROW_DOT_WARP}, {packet.rows // ROW_DOT_ROWS}, {c.block_volume} // {ROW_DOT_WARP})"
    )
    guard.body.append(walk)
    with c.cg.set_statements(cast("list[ast.AST]", walk.body)):
        c.emit(
            f"{row} = {stripe} * {ROW_DOT_ROWS} + ({c.thread} % {ROW_DOT_WARP}) // 4"
        )
        for reg in acc:
            c.emit(f"{reg} = cutlass.Float32(0)")
        tiles = carry_tile_loop(c, plan, step, ROW_DOT_K)
        packets = empty_loop(
            f"for {offset} in cutlass.range_constexpr({ROW_DOT_K // ROW_DOT_K_GROUP})"
        )
        with (
            carry_tile(c, plan, step, ROW_DOT_K),
            c.cg.set_statements(cast("list[ast.AST]", tiles.body)),
        ):
            low, high = (
                tile_window(a, row),
                tile_window(a, f"({row} + {ROW_DOT_ROWS // 2})"),
            )
            vector = tile_window(b)
        tiles.body.append(packets)
        c.cg.add_statement(tiles)
        with c.cg.set_statements(cast("list[ast.AST]", packets.body)):
            c.emit(
                f"{k} = {offset} * {ROW_DOT_K_GROUP} + ({c.thread} % 4) * {BF16_PACKET_LANES}"
            )
            lo, hi, vec = (read_words(window, k) for window in (low, high, vector))
            regs = ", ".join(acc)
            # Each four-lane group supplies eight contiguous K values per
            # lane to both operands. Both receive the same K permutation.
            # The upper eight MMA rows carry another eight matrix rows.
            for i in (0, 2):
                c.emit(
                    f"{regs} = _cute_mma_bf16({lo}[{i}], {hi}[{i}], "
                    f"{lo}[{i + 1}], {hi}[{i + 1}], {vec}[{i}], {vec}[{i + 1}], {regs})"
                )
        owned = empty_if(f"{c.thread} % 4 == 0")
        for delta, reg in ((0, acc[0]), (ROW_DOT_ROWS // 2, acc[2])):
            names = {
                row: ast.parse(f"({row} + {delta})", mode="eval").body,
                value: ast.Name(id=reg, ctx=ast.Load()),
            }
            for s in steps:
                owned.body.append(
                    cast(
                        "ast.stmt",
                        substitute_names(deep_clone(s), names, clone=deep_clone),
                    )
                )
        c.cg.add_statement(owned)
    return guard
