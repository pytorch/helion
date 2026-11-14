"""Reject the dims that a Metal thread would read at another element.

A Metal thread holds one element of each value, addressed through the block
ids of the value's dims, and finds the block of a dim that names none (a
literal ``hl.zeros([tile_m, 32])`` dim, a ``:`` slice) by its size: the
reduction block of that extent, never a tile.  Helion's tiles are positional
arrays, though, so a position can pair two dims that Metal holds on different
blocks:

* a loop carry ``hl.zeros([tile_m, 32])`` accumulating ``x[tile_m, tile_k]``
  of ``hl.tile(k, block_size=32)`` holds ``tile_k``'s columns in its literal
  dim, and ``acc.sum(1)`` then reduces over the reduction block of 32;
* ``x[tile_m, tile_n] + y[tile_m, :]`` pairs ``tile_n`` with the slice's
  reduction block; so do a ``:``-wide value stored through a tile and a tile
  stored through ``:`` (``out[tile_m, :] = x[tile_m, tile_n]``);
* ``y[tile_m, tile_k] * x[tile_m, tile_n]`` pairs two unrelated tiles, and
  so does a literal dim holding ``tile_k`` beside ``tile_n``.

Metal paired those by block, so each thread combined elements of different
columns and the kernel returned wrong values.  This refuses them.  So are two
tiles that meet swapped (``x[tm, tn] + y[tn, tm]``, ``out[tm, tn] =
y[tn, tm]``): Helion pairs them by position, as loaded, where Metal's pairing
by block computed a transpose.

Which block holds a literal dim is propagated by the shared analysis
CuTe's check also builds on (``helion/_compiler/literal_dims.py``: through
loops, branches, carries, views and matmuls), since CuTe lowers by the same
model.  The rest is Metal's: a tile is any
non-reduction block whatever its block size (so a refusal does not depend on
the config), ``torch.zeros``-style factories are the same everywhere as
``hl.full`` is, a subscript keeps a dim only for a tile, a ``:`` or ``None``,
and the checks and messages are this module's own.
"""

from __future__ import annotations

from itertools import starmap
import operator
from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node

from ... import exc
from ...language._tracing_ops import _phi
from ...language.memory_ops import load
from ...language.memory_ops import store
from ...language.view_ops import subscript as hl_subscript
from ..compile_environment import _symint_expr
from ..host_function import HostFunction
from ..literal_dims import CONFLICT
from ..literal_dims import CONTROL_FLOW_TARGETS
from ..literal_dims import FREE
from ..literal_dims import IDENTITY_TARGETS
from ..literal_dims import MATMUL_TARGETS
from ..literal_dims import SIZE
from ..literal_dims import Holder
from ..literal_dims import LiteralDimAnalysis
from ..literal_dims import literal_extent
from ..rebound_dims import VIEW_TARGETS
from ..rebound_dims import operand_positions
from ..rebound_dims import rebound_consumer_dims
from ..rebound_dims import rebound_operands
from ..variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..compile_environment import Config
    from ..device_ir import GraphInfo

# Hints, by what the refused program would have to change.
_ONE_BLOCK = "give the literal dim and the tile one block with hl.register_block_size"
_INDEX_SLICE = "index the ':' dim with the tile instead"
_SAME_TILES = "index both operands with the same tiles"
_STORE_THROUGH_TILE = "store it through the tile that holds it"
# A literal dim held by size came from a ':' slice the shape environment
# unified with the literal, or from an op this does not model.
_LITERAL_BY_SIZE = (
    "index the ':' dim it comes from with the tile, or give it the tile's block "
    "with hl.register_block_size"
)

# The factories whose value is the same everywhere along each dim, as
# ``hl.full``'s (``torch.zeros``, ``torch.full``, ``torch.zeros_like``, ...).
_FACTORY_TARGETS = frozenset(
    {
        torch.ops.aten.full.default,
        torch.ops.aten.zeros.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.full_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.ones_like.default,
        torch.ops.aten.new_full.default,
        torch.ops.aten.new_zeros.default,
        torch.ops.aten.new_ones.default,
    }
)

# What holds a dim, for messages and hints: a tile, a ``:`` slice's
# reduction block, or a literal dim held by the reduction block of its extent.
TILE = "tile"
SLICE = "slice"
LITERAL = "literal"
# A control-flow carry's dim that enters and leaves held by different blocks.
CARRIED = "carried"


def _op(target: object) -> str:
    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    assert callable(target)
    return target.__name__


def _describe(kind: str, holder: Holder) -> str:
    if kind == TILE:
        return f"block {holder}'s tile"
    if kind == SLICE:
        return "a ':' slice"
    if kind == CARRIED:
        return "a carry's dim that enters and leaves on different blocks"
    return "a literal dim held by the reduction block of its extent"


def _hint(kinds: Sequence[str]) -> str:
    if SLICE in kinds:
        return _INDEX_SLICE
    if LITERAL in kinds:
        return _LITERAL_BY_SIZE
    return _SAME_TILES


class _MetalLiteralDims(LiteralDimAnalysis):
    def run(self) -> None:
        self.propagate()
        for info in self.graphs:
            for node in info.graph.nodes:
                self.check(info, node)

    def compute(self, info: GraphInfo, node: Node) -> tuple[Holder, ...] | None:
        if node.op == "call_function" and node.target in _FACTORY_TARGETS:
            value = self.value(node)
            return None if value is None else self.default(value, FREE)
        return super().compute(info, node)

    def kind(self, node: Node, dim: int, holder: Holder) -> str:
        if isinstance(holder, int):
            return TILE
        if holder == CONFLICT:
            return CARRIED
        value = self.value(node)
        assert value is not None
        return LITERAL if literal_extent(self.env, value.shape[dim]) else SLICE

    # -- holders: a tile at any block size, slots by BlockSizeOrigin ---------

    def tile(self, block_id: int) -> Holder:
        block_id = self.env.canonical_block_id(block_id)
        return SIZE if self.env.block_sizes[block_id].reduction else block_id

    def subscript_holders(self, node: Node) -> list[Holder] | None:
        """What holds each dim ``tensor[subscript]`` keeps, walked as CuTe's
        ``_subscript_slot_dims``: a tile keeps its block, a ``:`` slice the
        reduction block of its extent, a unit dim and a scalar index (an
        ``hl.grid`` index, ``tile.id``, ``tile.begin``) nothing; None for a
        tensor indexer, which this does not model."""
        tensor = self.value(node.args[0])
        subscript = node.args[1]
        if tensor is None or not isinstance(subscript, (list, tuple)):
            return None
        origins = HostFunction.current().expr_to_origin
        holders: list[Holder] = []
        tensor_dim = 0
        for entry in subscript:
            index = entry.meta.get("val") if isinstance(entry, Node) else entry
            if index is None:
                holders.append(None)
                continue
            unit = tensor_dim < tensor.ndim and self.env.known_equal(
                tensor.shape[tensor_dim], 1
            )
            if isinstance(index, slice):
                holders.append(None if unit else SIZE)
            elif isinstance(index, torch.SymInt):
                symbol = _symint_expr(index)
                origin = origins.get(symbol) if symbol is not None else None
                if origin is not None and isinstance(origin.origin, BlockSizeOrigin):
                    holders.append(None if unit else self.tile(origin.origin.block_id))
            elif not isinstance(index, int):
                return None
            tensor_dim += 1
        return holders

    def load_dims(self, node: Node, value: torch.Tensor) -> tuple[Holder, ...]:
        holders = self.subscript_holders(node)
        if holders is None or len(holders) != value.ndim:
            return self.default(value)
        literal = self.literal_dims(value)
        return tuple(
            h if lit else None for h, lit in zip(holders, literal, strict=True)
        )

    # -- element-wise ops: each position decided on its own ------------------

    def combine(
        self, node: Node, value: torch.Tensor
    ) -> tuple[tuple[Holder, ...], str | None]:
        consumer = rebound_consumer_dims(node)
        if consumer is None or node.target is torch.ops.aten.gather.default:
            return self.default(value), None
        sizes, lower_rank_by_block_id = consumer
        block_ids = [
            self.env.get_block_id(size) if isinstance(size, torch.SymInt) else None
            for size in sizes
        ]
        # Per position, the (kind, holder) of each dim that meets there.
        at: list[list[tuple[str, Holder]]] = [
            []
            if (h := self.size_holder(size)) is None
            else [(TILE, h)]
            if isinstance(h, int)
            else [(SLICE, h)]
            for size in sizes
        ]
        free = [False] * len(sizes)
        operands = [
            (operand, operand_val)
            for operand in rebound_operands(node)
            if (operand_val := self.value(operand)) is not None
        ]
        if any(operand_val.ndim > len(sizes) for _, operand_val in operands):
            # Inductor lowers a sum over a unit dim element-wise: its operand
            # has a dim more than the result, so right-aligning pairs nothing.
            return self.default(value), None
        for operand, operand_val in operands:
            positions = operand_positions(
                self.env,
                operand_val,
                sizes,
                block_ids,
                lower_rank_by_block_id=lower_rank_by_block_id,
            )
            for dim, position in enumerate(positions):
                holder = self.dim_holder(operand, dim)
                if position < 0 or holder is None:
                    continue
                if holder == FREE:
                    free[position] = True
                    continue
                at[position].append((self.kind(operand, dim, holder), holder))
        conflict: str | None = None
        merged: list[Holder] = []
        for position, entries in enumerate(at):
            distinct = list(dict.fromkeys(h for _, h in entries))
            # A dim held by size meeting a tile, or two tiles meeting: each
            # thread would read another element than the position pairs.
            if self.final and conflict is None and len(distinct) > 1:
                kinds = list(dict.fromkeys(entries))
                held = ", ".join(starmap(_describe, kinds))
                conflict = self.message(
                    f"{_op(node.target)} pairs, at its dim {position}, dims held "
                    f"by {held}",
                    _hint([k for k, _ in kinds]),
                )
            tiles = [h for h in distinct if isinstance(h, int)]
            if tiles:
                merged.append(tiles[0])
            elif distinct:
                merged.append(distinct[0])
            else:
                # Same everywhere along the dim (a factory, a broadcast).
                merged.append(FREE if free[position] else None)
        if len(sizes) != value.ndim:
            return self.default(value), conflict
        literal = self.literal_dims(value)
        return tuple(
            h if lit else None for h, lit in zip(merged, literal, strict=True)
        ), conflict

    # -- checks ----------------------------------------------------------------

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
                            f"the loop carry {node.name} (shape {list(value.shape)}) "
                            f"holds block {tile}'s tile in its literal dim {dim} "
                            "across that tile's own loop: a thread carries one "
                            "scalar through the tile's lanes, and the tile is gone "
                            "after the loop",
                            _ONE_BLOCK,
                        )
            return
        if node.op != "call_function":
            return
        if target is _phi:
            for dim, holder in enumerate(self.holders.get(node, ())):
                if holder == CONFLICT:
                    kinds = self.phi_kinds(node, dim)
                    held = " and ".join(starmap(_describe, kinds))
                    self.refuse(
                        f"the control-flow carry {node.name} enters and leaves "
                        f"with its literal dim {dim} held by {held}",
                        _hint([k for k, _ in kinds]),
                    )
            return
        if target is store:
            self.check_slots(node, node.args[2], "a store writes")
            self.check_slots(node, _extra_mask(node, 3), "a store's mask reads")
            return
        if target is load:
            self.check_slots(node, _extra_mask(node, 2), "a load's mask reads")
            return
        if target in CONTROL_FLOW_TARGETS:
            return
        if (
            target in IDENTITY_TARGETS
            or target in (operator.getitem, torch.ops.aten.expand.default)
            or target
            in (torch.ops.aten.permute.default, torch.ops.aten.unsqueeze.default)
            or target is hl_subscript
            or value is None
        ):
            return
        if target in MATMUL_TARGETS:
            for dim, holder in enumerate(self.holders.get(node, ())):
                if holder == CONFLICT:
                    self.refuse(
                        f"{_op(target)} adds its accumulator to the product at "
                        f"its dim {dim}, which a different block holds in each",
                        _SAME_TILES,
                    )
            return
        if self.pointwise(node):
            _holders, conflict = self.combine(node, value)
            if conflict is not None:
                raise exc.BackendUnsupported("metal", conflict)
            return
        if target in VIEW_TARGETS and self.keeps_dims(node):
            return
        for operand in node.all_input_nodes:
            operand_val = self.value(operand)
            for dim, tile in self.tile_held(operand):
                assert operand_val is not None
                self.refuse(
                    f"{_op(target)} reads {operand.name} "
                    f"(shape {list(operand_val.shape)}), "
                    f"whose literal dim {dim} holds block {tile}'s tile, by the "
                    "reduction block of the dim's extent",
                    _ONE_BLOCK,
                )

    def phi_kinds(self, node: Node, dim: int) -> list[tuple[str, Holder]]:
        """The (kind, holder) a carry's dim enters and leaves with."""
        value = self.value(node)
        assert value is not None
        kinds: list[tuple[str, Holder]] = []
        for source in node.args[:2]:
            source_val = self.value(source)
            if source_val is None:
                continue
            assert isinstance(source, Node)
            source_dim = dim - (value.ndim - source_val.ndim)
            if source_dim < 0:
                continue
            holder = self.dim_holder(source, source_dim)
            if holder not in (None, FREE):
                kinds.append((self.kind(source, source_dim, holder), holder))
        return kinds

    def check_slots(self, node: Node, value_node: object, what: str) -> None:
        """A stored value or a load's or store's mask, right-aligned to the
        dims the subscript keeps: a tile must not meet a ``:`` slot nor a dim
        held by size a tile's slot, and a literal dim holding a tile must
        meet that tile; a tile must meet its own tile's slot."""
        value = self.value(value_node)
        slots = self.subscript_holders(node)
        if value is None or slots is None:
            return
        assert isinstance(value_node, Node)
        offset = len(slots) - value.ndim
        for dim in range(value.ndim):
            if dim + offset < 0:
                continue
            holder = self.dim_holder(value_node, dim)
            slot = slots[dim + offset]
            if holder is None or slot is None or holder in (FREE, slot):
                continue
            kind = self.kind(value_node, dim, holder)
            if isinstance(holder, int) and isinstance(slot, int):
                hint = _STORE_THROUGH_TILE
            elif isinstance(holder, int):
                hint = _INDEX_SLICE
            elif isinstance(slot, int):
                hint = _hint([kind])
            else:
                continue
            slot_kind = TILE if isinstance(slot, int) else SLICE
            self.refuse(
                f"{what} dim {dim} of {value_node.name} "
                f"(shape {list(value.shape)}), held by {_describe(kind, holder)}, "
                f"through {_describe(slot_kind, slot)} of its subscript",
                hint,
            )

    def message(self, reason: str, hint: str = _ONE_BLOCK) -> str:
        return (
            f"{reason}.  A Metal thread holds each element by the block ids of "
            "its dims, and finds the block of a dim that names none by its "
            f"size, never a tile, so it would read another element; {hint}"
        )

    def refuse(self, reason: str, hint: str = _ONE_BLOCK) -> None:
        raise exc.BackendUnsupported("metal", self.message(reason, hint))


def _extra_mask(node: Node, index: int) -> object:
    """The ``extra_mask`` argument of an ``hl.load`` (``index`` 2) or
    ``hl.store`` (3), or None."""
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get("extra_mask")


def reject_literal_dims(graphs: Sequence[GraphInfo], config: Config) -> None:
    """Raise ``BackendUnsupported`` where a thread would read another element
    (see the module docstring)."""
    _MetalLiteralDims(graphs, config).run()
