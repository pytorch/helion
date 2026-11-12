"""Which block holds each literal dim of a CuTe tile value.

The SIMT lowering holds every block id at one coordinate per thread (and per
lane iteration) and addresses loads, stores and reductions by those
coordinates.  A dim with a block id in its size is held by that block.  A
literal dim (``hl.zeros([tile_m, 64])``, a ``:`` slice whose reduction block
was unified with a literal, a tile whose fixed block size the shape
environment replaced by its value) has no block in its size: a reduction or
a ``:`` store finds the reduction block of its extent by size, never a tile,
whose block carries the loop's extent.  Yet a literal dim can hold a tile's
elements: a load through a tile whose size became a literal, an element-wise
op whose literal result dim meets an operand's tile dim, a view of either,
a loop carry updated by one of those.

``annotate_literal_dims`` follows each literal dim from its producers
(``helion/_compiler/literal_dims.py``): held by a tile block, by size (the
reduction block of its extent), or the same everywhere along it (a factory,
an iota whose lowering matches the active block of its extent itself, a
broadcast).  It refuses (``BackendUnsupported``)
a program that combines a tile-held literal dim with a size-held one or with
another tile, that hands a tile-held literal dim to an op it does not model
(a reduction or a reshape among them: they find a block for the dim by its
size), and a loop carry whose literal dim holds the loop's own tile (a
per-thread scalar cannot carry a lane loop's elements across iterations, and
the tile is gone after the loop).  A store, atomic or mask checks its value
against the blocks its subscript addresses (``check_literal_slot_dims``).
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node

from ... import exc
from ...language._tracing_ops import _phi
from ...language.memory_ops import load
from ...language.view_ops import subscript as hl_subscript
from ..compile_environment import CompileEnvironment
from ..literal_dims import CONTROL_FLOW_TARGETS
from ..literal_dims import FREE
from ..literal_dims import IDENTITY_TARGETS
from ..literal_dims import MATMUL_TARGETS
from ..literal_dims import SIZE
from ..literal_dims import Holder
from ..literal_dims import LiteralDimAnalysis
from ..literal_dims import describe_holder
from ..literal_dims import literal_extent
from ..literal_dims import tile_block
from ..rebound_dims import VIEW_TARGETS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..compile_environment import Config
    from ..device_ir import GraphInfo

CUTE_LITERAL_DIMS_META = "cute_literal_dims"

_SUFFIX = (
    "CuTe finds the block of a literal dim by its size, never a tile, so each "
    "thread would read the elements at two different coordinates"
)


class _CuteLiteralDims(LiteralDimAnalysis):
    def check(self, info: GraphInfo, node: Node) -> None:
        from ..device_ir import ForLoopGraphInfo

        value = self.value(node)
        target = node.target
        if node.op == "placeholder":
            if value is not None and isinstance(info, ForLoopGraphInfo):
                own = {self.env.canonical_block_id(b) for b in info.block_ids}
                for dim, tile in self.tile_held(node):
                    if tile in own:
                        self.refuse(
                            f"loop carry {node.name} (shape {list(value.shape)}) "
                            f"holds block id {tile}'s tile in its literal dim {dim} "
                            "across the iterations of that tile's own loop: each "
                            "thread carries one scalar through the tile's lanes, "
                            "and the tile is gone after the loop"
                        )
            return
        if node.op != "call_function":
            return
        if target is load or target in CONTROL_FLOW_TARGETS:
            return
        if (
            target in IDENTITY_TARGETS
            or target in (_phi, operator.getitem, torch.ops.aten.expand.default)
            or target
            in (torch.ops.aten.permute.default, torch.ops.aten.unsqueeze.default)
            or target is hl_subscript
        ):
            return
        if value is not None and self.pointwise(node):
            _holders, conflict = self.combine(node, value)
            if conflict is not None:
                self.refuse(f"{conflict}: {_SUFFIX}")
            return
        if target in VIEW_TARGETS and self.keeps_dims(node):
            # Keeps the holders (``aligned_dims``).
            return
        if target in MATMUL_TARGETS:
            return
        if _is_store_like(node) or value is None:
            # A store's value meets its subscript's blocks in
            # ``check_literal_slot_dims``; an op without a tile result
            # (``sym_size``) reads no elements.
            return
        for operand in node.all_input_nodes:
            operand_val = self.value(operand)
            for dim, tile in self.tile_held(operand):
                assert operand_val is not None
                self.refuse(
                    f"{target} reads {operand.name} (shape "
                    f"{list(operand_val.shape)}), whose literal dim {dim} "
                    f"holds block id {tile}'s tile: {_SUFFIX}"
                )

    def refuse(self, reason: str) -> None:
        raise exc.BackendUnsupported("cute", reason)

    def run(self) -> None:
        self.propagate()
        for info in self.graphs:
            for node in info.graph.nodes:
                self.check(info, node)
        for node, holders in self.holders.items():
            node.meta[CUTE_LITERAL_DIMS_META] = holders


def _is_store_like(node: Node) -> bool:
    from ...language.atomic_ops import ATOMIC_OPS
    from ...language.memory_ops import store

    return node.target is store or node.target in ATOMIC_OPS


def annotate_literal_dims(graphs: Sequence[GraphInfo], config: Config) -> None:
    """Record ``CUTE_LITERAL_DIMS_META`` and refuse the programs whose literal
    dims a tile and a size hold at once (see the module docstring)."""
    _CuteLiteralDims(graphs, config).run()


def check_literal_slot_dims(
    config: Config,
    value_node: object,
    value: torch.Tensor,
    slot_block_ids: Sequence[int | None],
    *,
    what: str,
) -> None:
    """Refuse a stored, atomic or mask value whose literal dim the subscript
    addresses by another block than the one holding it.

    The value is right-aligned to the subscript's dims (``slot_block_ids``,
    the blocks ``_subscript_slot_dims`` addresses each with): a literal dim
    holding a tile must meet that tile, one held by size must not meet a
    tile.
    """
    env = CompileEnvironment.current()
    recorded = (
        value_node.meta.get(CUTE_LITERAL_DIMS_META)
        if isinstance(value_node, Node)
        else None
    )
    offset = len(slot_block_ids) - value.ndim
    for dim, size in enumerate(value.shape):
        if dim + offset < 0 or value.stride(dim) == 0 or not literal_extent(env, size):
            continue
        holder: Holder = SIZE
        if (
            isinstance(recorded, tuple)
            and dim < len(recorded)
            and recorded[dim] is not None
        ):
            holder = recorded[dim]
        if holder == FREE:
            continue
        slot = slot_block_ids[dim + offset]
        slot_tile = tile_block(env, config, slot)
        if isinstance(holder, int):
            if slot_tile == holder:
                continue
        elif slot_tile is None:
            continue
        name = value_node.name if isinstance(value_node, Node) else "the value"
        slot_holder: Holder = (
            slot_tile if slot_tile is not None else (SIZE if slot is not None else None)
        )
        raise exc.BackendUnsupported(
            "cute",
            f"{what} addresses dim {dim} of {name} (shape {list(value.shape)}), held "
            f"by {describe_holder(holder)}, through a dim addressed by "
            f"{describe_holder(slot_holder)}: {_SUFFIX}",
        )
