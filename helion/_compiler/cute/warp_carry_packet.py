"""Feed a register-owned row reduction from whole contiguous host packets."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from .autotune.config_keys import ROW_DOT_KEY
from .fragment_vector_io import PACKET_BYTES
from .fragment_vector_io import copy_atom
from .row_carry_terms import empty_if

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable
    from collections.abc import Mapping

    from ...autotuner.config_spec import ConfigSpec
    from .fragment_context import Fragment
    from .fragment_context import FragmentContext


def warp_carry_tile(tile: int) -> bool:
    from .fragment_context import UNROLLED_SLOTS

    return tile % 32 == 0 and 32 <= tile <= 32 * UNROLLED_SLOTS


def uses_warp_carry(spec: ConfigSpec, config: Mapping[str, object]) -> bool:
    return (
        spec.cute_fragment_row_carry_shared
        and bool(config.get("cute_fragment_row_carry"))
        and config.get("cute_fragment_reduction") == "warp"
    )


def has_carry_packet(spec: ConfigSpec, config: Mapping[str, object]) -> bool:
    return uses_warp_carry(spec, config) and bool(
        spec.cute_fragment_row_carry_packet_tiles
    )


@dataclasses.dataclass(frozen=True)
class CarryPacketSource:
    """A matched 16-byte access; the address callback retains the load mask."""

    base: str
    address: Callable[[tuple[str, ...]], tuple[str, str]]


@dataclasses.dataclass
class WarpCarryPacket:
    """One packet per thread and K tile, published before its unrolled columns."""

    axis: int
    setup: list[ast.stmt]
    row: str
    lane: str
    column: str
    slot: str
    rows: int
    lanes: int
    sources: dict[Node, CarryPacketSource] = dataclasses.field(default_factory=dict)

    def accepts_indices(self, indices: list[object]) -> bool:
        # The address recipe runs before the scalar columns. Only capture
        # indices (and their transparent loop-entry aliases) or shape
        # expressions exist there. Do not hoist a computed value from inside
        # the scalar loop, even when its resulting address looks contiguous.
        return all(
            not isinstance(index, Node)
            or index.op == "placeholder"
            or index.target is _tracing_ops._get_symnode
            or index.target is torch.ops.aten.sym_size.int
            or (
                index.target is _tracing_ops._new_var
                and isinstance(index.args[0], Node)
                and index.args[0].op == "placeholder"
            )
            for index in indices
        )

    def wrap(
        self,
        c: FragmentContext,
        fragment: Fragment,
        address: Callable[[tuple[str, ...]], tuple[str, str]],
        *,
        node: Node,
        base: str,
        aligned: bool,
    ) -> Fragment:
        prefix = (self.row,) if len(fragment.shape) == 2 else ()
        expected = (*prefix, self.column)
        packet: str | None = None

        def read(coords: tuple[str, ...]) -> str:
            nonlocal packet
            # Only a read at our exact owner coordinate can use the packet.
            # Broadcasts, reindexed views and index calculations keep their
            # scalar address recipe. No speculative load for an unused view.
            if coords != expected:
                return fragment.read(coords)
            if packet is None:
                if c.df.config.get(ROW_DOT_KEY):
                    self.sources[node] = CarryPacketSource(base, address)
                packet = c.df.new_var("fragment_carry_packet")
                dtype = c.dtype(fragment.dtype)
                with c.cg.set_statements(cast("list[ast.AST]", self.setup)):
                    c.emit(
                        f"{packet} = cute.make_rmem_tensor(({self.lanes},), {dtype})"
                    )
                    c.emit(f"{packet}.fill({c.cast('0', fragment.dtype)})")
                    source, valid = address((*prefix, f"({self.lane} * {self.lanes})"))
                    allowed = empty_if(valid)
                    with c.cg.set_statements(cast("list[ast.AST]", allowed.body)):
                        pointer = c.df.new_var("fragment_carry_pointer")
                        c.emit(f"{pointer} = {source}")
                        vector = (
                            f"cute.copy({copy_atom(dtype)}, cute.make_tensor("
                            f"{pointer}.align({PACKET_BYTES}), cute.make_layout(({self.lanes},))), {packet})"
                        )
                        if aligned:
                            c.emit(vector)
                        else:
                            j = c.df.new_var("fragment_carry_scalar")
                            element = c.cast(
                                f"({pointer} + {j}).load()", fragment.dtype
                            )
                            # The offset is already a whole packet. CuTe's
                            # launcher may specialize this argument's pointer
                            # alignment even when Helion has no base guard.
                            c.emit(
                                f"if cutlass.const_expr({base}.alignment >= {PACKET_BYTES}):\n"
                                f"    {vector}\n"
                                f"else:\n"
                                f"    if (cutlass.Int64({pointer}.toint()) & {PACKET_BYTES - 1}) == 0:\n"
                                f"        {vector}\n"
                                f"    else:\n"
                                f"        for {j} in cutlass.range_constexpr({self.lanes}):\n"
                                f"            {packet}[{j}] = {element}"
                            )
                    c.cg.add_statement(allowed)
            return f"{packet}[{self.slot}]"

        return dataclasses.replace(fragment, element=read)
