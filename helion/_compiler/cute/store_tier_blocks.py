"""Which device loops open ``cute_store_packet_tiers`` to the search.

The knob itself is lowered by ``affine_vector_io`` (the store tiers of its
module docstring); this module holds the structural fact the search surface
gates on, apart from the pass so that the pass does not import the language's
tracing operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language._tracing_ops import _for_loop
from ...language.memory_ops import store
from ..compile_environment import CompileEnvironment
from ..compile_environment import _symint_expr
from ..device_ir import ForLoopGraphInfo
from .affine_vector_io import PACKET_DTYPES

if TYPE_CHECKING:
    from collections.abc import Iterator

    import sympy

    from ...autotuner.config_spec import ConfigSpec
    from ..device_ir import DeviceIR


def _has_dim(value: object, symbol: sympy.Symbol) -> bool:
    return isinstance(value, torch.Tensor) and any(
        isinstance(size, torch.SymInt) and _symint_expr(size) == symbol
        for size in value.shape
    )


def _is_tile(subscript: object, symbol: sympy.Symbol) -> bool:
    """Whether a store's subscript is the tile whose block has the size
    ``symbol`` (a tile subscript is traced as its block size)."""
    return (
        isinstance(subscript, torch.fx.Node)
        and isinstance(size := subscript.meta.get("val"), torch.SymInt)
        and _symint_expr(size) == symbol
    )


def tile_stores(
    device_ir: DeviceIR, config_spec: ConfigSpec
) -> Iterator[tuple[int, ForLoopGraphInfo, sympy.Symbol, torch.Tensor]]:
    """``(block, loop, the block's size symbol, stored tensor)`` of each
    store along the tile of a one-axis ``hl.tile`` loop in a root graph over
    a block that can have vector lanes, into a tensor of a packet dtype with
    at least two dimensions whose last dimension the tile indexes."""
    env = CompileEnvironment.current()
    lane_blocks = set(config_spec.cute_vector_widths.valid_block_ids())
    for root_id in device_ir.root_ids:
        root = device_ir.graphs[root_id].graph
        for node in root.find_nodes(op="call_function", target=_for_loop):
            info = device_ir.graphs[node.args[0]]
            if type(info) is not ForLoopGraphInfo or len(info.block_ids) != 1:
                continue
            block_id = info.block_ids[0]
            if block_id not in lane_blocks:
                continue
            symbol = env.block_sizes[block_id].symbol()
            for write in info.graph.find_nodes(op="call_function", target=store):
                target, subscripts, value = write.args[:3]
                if (
                    isinstance(target, torch.fx.Node)
                    and isinstance(tensor := target.meta.get("val"), torch.Tensor)
                    and tensor.dtype in PACKET_DTYPES
                    and tensor.ndim >= 2
                    and isinstance(subscripts, (list, tuple))
                    and len(subscripts) == tensor.ndim
                    and _is_tile(subscripts[-1], symbol)
                    and isinstance(value, torch.fx.Node)
                    and _has_dim(value.meta.get("val"), symbol)
                ):
                    yield block_id, info, symbol, tensor


def store_tier_blocks(device_ir: DeviceIR, config_spec: ConfigSpec) -> frozenset[int]:
    """The blocks whose device loops open ``cute_store_packet_tiers`` to the
    search: a one-axis ``hl.tile`` loop in a root graph over a block that can
    have vector lanes, which stores along its tile into the last dimension
    of a tensor of a packet dtype with at least two dimensions (a row prefix
    whose alignment the packets depend on).  The tile itself is the
    subscript there: the lanes of a store through a full slice
    (``out[tile_s, tile_r, :]``, a row that a reduction's lanes store) or
    ahead of a component index (``out[tile_m, tile_n, 0]``) are not adjacent
    elements along this loop's tile, and ``affine_vector_io`` stores them in
    no packet.  Structure only: whether a row stride leaves the packets
    unaligned is a fact of the shape, and where every store's alignment is
    proven the knob changes nothing; what the stored tensors' layout rules
    out or proves aligned is the config normalization's to read
    (``store_packet_layout``)."""
    return frozenset(
        block_id
        for block_id, _loop, _symbol, _tensor in tile_stores(device_ir, config_spec)
    )
