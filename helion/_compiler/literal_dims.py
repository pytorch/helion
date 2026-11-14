"""Which block holds each literal dim of a tile value.

A thread-level (SIMT) lowering holds every block id at one coordinate per
thread (and per lane iteration) and addresses loads, stores and reductions by
those coordinates.  A dim with a block id in its size is held by that block.
A literal dim (``hl.zeros([tile_m, 64])``, a ``:`` slice whose reduction block
was unified with a literal, a tile whose fixed block size the shape
environment replaced by its value) has no block in its size, yet it can hold
a tile's elements: a load through a tile whose size became a literal, an
element-wise op whose literal result dim meets an operand's tile dim, a view
of either, a loop carry updated by one of those.

``LiteralDimAnalysis.propagate`` follows each literal dim from its producers
to a fixpoint: held by a tile block, by size (``SIZE``: the reduction block of
its extent), or the same everywhere along it (``FREE``: a factory, an iota, a
broadcast).  It refuses nothing; a backend subclasses it with the programs
its lowering cannot address (CuTe's ``cute/literal_dims.py``, Metal's
``metal/literal_dims.py``).
"""

from __future__ import annotations

from itertools import starmap
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx.node import Node

from ..language._tracing_ops import _for_loop
from ..language._tracing_ops import _for_loop_step
from ..language._tracing_ops import _if
from ..language._tracing_ops import _mask_to
from ..language._tracing_ops import _new_var
from ..language._tracing_ops import _phi
from ..language._tracing_ops import _while_loop
from ..language.creation_ops import full
from ..language.matmul_ops import dot as hl_dot
from ..language.memory_ops import load
from ..language.reduce_ops import _reduce
from ..language.scan_ops import _associative_scan
from ..language.view_ops import subscript as hl_subscript
from .compile_environment import CompileEnvironment
from .rebound_dims import REBOUND_CHECK_TARGETS
from .rebound_dims import VIEW_TARGETS
from .rebound_dims import block_extent
from .rebound_dims import operand_positions
from .rebound_dims import rebound_consumer_dims
from .rebound_dims import rebound_operands

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .compile_environment import Config
    from .device_ir import GraphInfo

# A literal dim held by the reduction block of its extent.
SIZE = "size"
# A literal dim whose value is the same everywhere along it.
FREE = "free"
# Two holders that disagree.
CONFLICT = "conflict"

# Per dim: a tile block id, SIZE, FREE, CONFLICT, or None for a dim whose
# size names its block (or of extent 1).
Holder = int | str | None

IDENTITY_TARGETS = frozenset(
    {
        _new_var,
        _mask_to,
        torch.ops.aten.clone.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
        torch.ops.aten._to_copy.default,
        torch.ops.prims.convert_element_type.default,
    }
)

_IOTA_TARGETS = frozenset(
    {
        torch.ops.prims.iota.default,
        torch.ops.aten.arange.default,
        torch.ops.aten.arange.start,
        torch.ops.aten.arange.start_step,
    }
)

MATMUL_TARGETS = frozenset(
    {
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.baddbmm.default,
        hl_dot,
    }
)

CONTROL_FLOW_TARGETS = frozenset({_for_loop, _for_loop_step, _if, _while_loop})

# ``REBOUND_CHECK_TARGETS`` that do more than combine their operands per
# element: a tuple reduce or scan runs along a dim.
_NOT_POINTWISE = MATMUL_TARGETS | {_reduce, _associative_scan}


def tile_block(
    env: CompileEnvironment, config: Config, block_id: int | None
) -> int | None:
    """``block_id``'s canonical id if it is a tile block wider than 1."""
    if block_id is None:
        return None
    block_id = env.canonical_block_id(block_id)
    if env.block_sizes[block_id].reduction or block_extent(env, config, block_id) == 1:
        return None
    return block_id


def literal_extent(env: CompileEnvironment, size: object) -> bool:
    """Whether a dim of ``size`` is a literal extent wider than 1: no block id
    and no block size in it (``tile.block_size // 2`` is a sub-coordinate of
    its block, ``_get_dim_local_coord``)."""
    if not isinstance(size, (int, torch.SymInt)):
        return False
    if env.get_block_id(size) is not None or env.known_equal(size, 1):
        return False
    if isinstance(size, torch.SymInt):
        return not any(
            env.get_block_id(symbol) is not None
            for symbol in size.node.expr.free_symbols
        )
    return True


def _merge(left: Holder, right: Holder) -> Holder:
    if left is None or left == FREE:
        return right if right is not None else left
    if right is None or right in (FREE, left):
        return left
    return CONFLICT


def describe_holder(holder: Holder) -> str:
    if isinstance(holder, int):
        return f"block id {holder}'s tile"
    if holder == SIZE:
        return "the reduction block of its extent"
    return "nothing"


class LiteralDimAnalysis:
    def __init__(self, graphs: Sequence[GraphInfo], config: Config) -> None:
        self.env = CompileEnvironment.current()
        self.config = config
        self.graphs = list(graphs)
        self.by_id = {info.graph_id: info for info in self.graphs}
        self.holders: dict[Node, tuple[Holder, ...]] = {}
        self.final = False
        # (loop graph id, init node) -> index of the carried output.
        self.carried: dict[tuple[int, object], int] = {}
        # Body graph id -> the arguments its placeholders bind, read from the
        # call in the graphs under codegen (``node_args`` may belong to the
        # traced graphs these copy).
        self.call_args: dict[int, list[object]] = {}
        for info in self.graphs:
            for node in info.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target in (_for_loop, _for_loop_step):
                    self.bind(node.args[0], node.args[3])
                elif node.target is _if:
                    self.bind(node.args[1], node.args[3])
                    self.bind(node.args[2], node.args[4])
                elif node.target is _while_loop:
                    self.bind(node.args[0], node.args[2])
                    self.bind(node.args[1], node.args[2])
                elif node.target is _phi:
                    self.record_carry(node)

    def bind(self, graph_id: object, args: object) -> None:
        if isinstance(graph_id, int) and isinstance(args, (list, tuple)):
            self.call_args[graph_id] = [*args]

    def record_carry(self, phi: Node) -> None:
        init, update = phi.args[:2]
        if not (
            isinstance(init, Node)
            and isinstance(update, Node)
            and update.target is operator.getitem
        ):
            return
        loop, index = update.args[:2]
        if not isinstance(loop, Node) or not isinstance(index, int):
            return
        if loop.target in (_for_loop, _for_loop_step):
            graph_id = loop.args[0]
        elif loop.target is _while_loop:
            graph_id = loop.args[1]
        else:
            return
        assert isinstance(graph_id, int)
        self.carried[(graph_id, init)] = index

    def outer_arg(self, info: GraphInfo, placeholder: Node) -> object:
        """The value bound to ``placeholder`` of body graph ``info``."""
        from .device_ir import NodeArgsGraphInfo

        index = placeholder.graph.find_nodes(op="placeholder").index(placeholder)
        args = self.call_args.get(info.graph_id)
        if args is None and isinstance(info, NodeArgsGraphInfo):
            # A reduction loop has no call node: match its recorded argument
            # by name in the graphs under codegen.
            recorded = info.node_args[index] if index < len(info.node_args) else None
            if recorded in self.holders or not isinstance(recorded, Node):
                return recorded
            return next(
                (
                    node
                    for other in self.graphs
                    if other is not info
                    for node in other.graph.nodes
                    if node.name == recorded.name
                ),
                None,
            )
        return args[index] if args is not None and index < len(args) else None

    def value(self, node: object) -> torch.Tensor | None:
        if not isinstance(node, Node):
            return None
        value = node.meta.get("val")
        return value if isinstance(value, torch.Tensor) else None

    def dim_holder(self, node: Node, dim: int) -> Holder:
        """What holds dim ``dim`` of ``node``: its tile block, ``SIZE`` for a
        reduction block, or the recorded holder of a literal dim."""
        value = self.value(node)
        assert value is not None
        size = value.shape[dim]
        if self.env.known_equal(size, 1) or value.stride(dim) == 0:
            return FREE
        holder = self.size_holder(size)
        if holder is not None:
            return holder
        if not literal_extent(self.env, size):
            return None
        recorded = self.holders.get(node)
        if recorded is not None and dim < len(recorded) and recorded[dim] is not None:
            return recorded[dim]
        # Not reached yet while the holders grow to their fixpoint.
        return SIZE if self.final else None

    def literal_dims(self, value: torch.Tensor) -> list[bool]:
        return [literal_extent(self.env, size) for size in value.shape]

    def default(self, value: torch.Tensor, holder: Holder = SIZE) -> tuple[Holder, ...]:
        return tuple(holder if lit else None for lit in self.literal_dims(value))

    def subgraph_outputs(self, node: Node) -> list[list[object]]:
        target = node.target
        if target in (_for_loop, _for_loop_step):
            graph_ids = [node.args[0]]
        elif target is _if:
            graph_ids = [node.args[1], node.args[2]]
        else:
            graph_ids = [node.args[1]]
        outputs: list[list[object]] = []
        for graph_id in graph_ids:
            info = self.by_id.get(graph_id) if isinstance(graph_id, int) else None
            if info is None:
                continue
            (result,) = info.graph.find_nodes(op="output")
            values = result.args[0]
            outputs.append([*values] if isinstance(values, (list, tuple)) else [values])
        return outputs

    def compute(self, info: GraphInfo, node: Node) -> tuple[Holder, ...] | None:
        value = self.value(node)
        if value is None:
            return None
        target = node.target
        literal = self.literal_dims(value)
        if node.op == "placeholder":
            outer = self.outer_arg(info, node)
            if outer is None:
                return self.default(value)
            holders = self.copy_dims(outer, value)
            index = self.carried.get((info.graph_id, outer))
            if index is not None:
                (result,) = info.graph.find_nodes(op="output")
                outputs = result.args[0]
                if isinstance(outputs, (list, tuple)) and index < len(outputs):
                    holders = self.merge_dims(
                        holders, self.copy_dims(outputs[index], value)
                    )
            return holders
        if node.op != "call_function":
            return None
        if target is full or target in _IOTA_TARGETS:
            return self.default(value, FREE)
        if target in IDENTITY_TARGETS or target is torch.ops.aten.expand.default:
            return self.copy_dims(node.args[0], value)
        if target is _phi:
            return self.merge_dims(
                self.copy_dims(node.args[0], value), self.copy_dims(node.args[1], value)
            )
        if target is operator.getitem:
            source, index = node.args[:2]
            if isinstance(source, Node) and source.target in CONTROL_FLOW_TARGETS:
                holders = self.default(value, FREE)
                for outputs in self.subgraph_outputs(source):
                    if isinstance(index, int) and index < len(outputs):
                        holders = self.merge_dims(
                            holders, self.copy_dims(outputs[index], value)
                        )
                return holders
            return self.default(value)
        if target is load:
            return self.load_dims(node, value)
        if target is torch.ops.aten.permute.default:
            source = node.args[0]
            dims = node.args[1]
            assert isinstance(source, Node) and isinstance(dims, (list, tuple))
            return tuple(
                self.dim_holder(source, cast("int", dim)) if lit else None
                for dim, lit in zip(dims, literal, strict=True)
            )
        if target in (torch.ops.aten.unsqueeze.default, hl_subscript) or (
            target in VIEW_TARGETS and self.keeps_dims(node)
        ):
            return self.aligned_dims(node, value)
        if target in MATMUL_TARGETS:
            return self.matmul_dims(node, value)
        if self.pointwise(node):
            return self.pointwise_dims(node, value)
        return self.default(value)

    def copy_dims(self, source: object, value: torch.Tensor) -> tuple[Holder, ...]:
        """``source``'s holders right-aligned onto ``value``'s literal dims."""
        source_val = self.value(source)
        literal = self.literal_dims(value)
        if source_val is None:
            return self.default(value)
        assert isinstance(source, Node)
        offset = value.ndim - source_val.ndim
        holders: list[Holder] = []
        for dim, lit in enumerate(literal):
            if not lit:
                holders.append(None)
            elif dim - offset < 0:
                holders.append(FREE)
            else:
                holders.append(self.dim_holder(source, dim - offset))
        return tuple(holders)

    def merge_dims(
        self, left: Sequence[Holder], right: Sequence[Holder]
    ) -> tuple[Holder, ...]:
        return tuple(starmap(_merge, zip(left, right, strict=True)))

    def keeps_dims(self, node: Node) -> bool:
        """Whether a view keeps its source's non-unit dims in order (adding or
        dropping unit dims only)."""
        source = self.value(node.args[0])
        value = self.value(node)
        if source is None or value is None:
            return False
        source_sizes = [s for s in source.shape if not self.env.known_equal(s, 1)]
        sizes = [s for s in value.shape if not self.env.known_equal(s, 1)]
        return len(source_sizes) == len(sizes) and all(
            starmap(self.env.known_equal, zip(source_sizes, sizes, strict=True))
        )

    def aligned_dims(self, node: Node, value: torch.Tensor) -> tuple[Holder, ...]:
        """Holders through a view that keeps the source's non-unit dims in
        order (``unsqueeze``, ``hl.subscript`` with ``None``, ``keeps_dims``)."""
        source = node.args[0]
        source_val = self.value(source)
        if source_val is None:
            return self.default(value)
        assert isinstance(source, Node)
        source_dims = [
            dim
            for dim in range(source_val.ndim)
            if not self.env.known_equal(source_val.shape[dim], 1)
        ]
        holders: list[Holder] = []
        for size in value.shape:
            if self.env.known_equal(size, 1):
                holders.append(None)
                continue
            if not source_dims:
                return self.default(value)
            dim = source_dims.pop(0)
            holders.append(
                self.dim_holder(source, dim) if literal_extent(self.env, size) else None
            )
        return tuple(holders)

    def load_dims(self, node: Node, value: torch.Tensor) -> tuple[Holder, ...]:
        """A tile in the subscript holds the dim it keeps, a literal one when
        the shape environment replaced the tile's block size by its value."""
        subscript = node.args[1]
        if not isinstance(subscript, (list, tuple)):
            return self.default(value)
        holders: list[Holder] = []
        for entry in subscript:
            index = entry.meta.get("val") if isinstance(entry, Node) else entry
            if index is None:
                holders.append(None)
            elif isinstance(index, slice):
                holders.append(SIZE)
            elif (
                isinstance(index, torch.SymInt)
                and (block_id := self.env.get_block_id(index)) is not None
            ):
                holders.append(self.tile(block_id))
            elif isinstance(index, (int, torch.SymInt)):
                continue
            else:
                return self.default(value)
        if len(holders) != value.ndim:
            return self.default(value)
        literal = self.literal_dims(value)
        return tuple(
            holder if lit else None
            for holder, lit in zip(holders, literal, strict=True)
        )

    def matmul_dims(self, node: Node, value: torch.Tensor) -> tuple[Holder, ...]:
        """The rows come from the lhs, the columns from the rhs, batch dims
        from the lhs; an accumulator is added to the product per thread."""
        target = node.target
        if target in (torch.ops.aten.addmm.default, torch.ops.aten.baddbmm.default):
            acc, lhs, rhs = node.args[:3]
        else:
            lhs, rhs = node.args[:2]
            acc = node.args[2] if target is hl_dot and len(node.args) > 2 else None
        lhs_val, rhs_val = self.value(lhs), self.value(rhs)
        if lhs_val is None or rhs_val is None:
            return self.default(value)
        assert isinstance(lhs, Node) and isinstance(rhs, Node)
        literal = self.literal_dims(value)
        holders: list[Holder] = []
        for dim, lit in enumerate(literal):
            if not lit:
                holders.append(None)
            elif dim == value.ndim - 1:
                holders.append(self.dim_holder(rhs, rhs_val.ndim - 1))
            else:
                holders.append(self.dim_holder(lhs, lhs_val.ndim - value.ndim + dim))
        if self.value(acc) is not None:
            holders = list(self.merge_dims(holders, self.copy_dims(acc, value)))
        return tuple(holders)

    def pointwise(self, node: Node) -> bool:
        from .inductor_lowering import PointwiseLowering

        return (
            isinstance(node.meta.get("lowering"), PointwiseLowering)
            or node.target in REBOUND_CHECK_TARGETS
        ) and node.target not in _NOT_POINTWISE

    def pointwise_dims(self, node: Node, value: torch.Tensor) -> tuple[Holder, ...]:
        holders, _conflict = self.combine(node, value)
        return holders

    def combine(
        self, node: Node, value: torch.Tensor
    ) -> tuple[tuple[Holder, ...], str | None]:
        """Merge the operands' holders position by position (placed as
        ``rebound_block_dims`` places them); the result's literal dims take
        the merged holder.  Returns the first disagreement involving a
        literal dim (tiles against tiles are ``rebound_block_dims``'
        business)."""
        consumer = rebound_consumer_dims(node)
        literal = self.literal_dims(value)
        if consumer is None or node.target is torch.ops.aten.gather.default:
            return self.default(value), None
        sizes, lower_rank_by_block_id = consumer
        block_ids = [
            self.env.get_block_id(size) if isinstance(size, torch.SymInt) else None
            for size in sizes
        ]
        merged: list[Holder] = [
            None if literal_extent(self.env, size) else self.size_holder(size)
            for size in sizes
        ]
        has_literal = [literal_extent(self.env, size) for size in sizes]
        sources: list[list[tuple[Node, torch.Tensor, int, Holder]]] = [
            [] for _ in sizes
        ]
        for operand in rebound_operands(node):
            operand_val = self.value(operand)
            if operand_val is None:
                continue
            positions = operand_positions(
                self.env,
                operand_val,
                sizes,
                block_ids,
                lower_rank_by_block_id=lower_rank_by_block_id,
            )
            for dim, position in enumerate(positions):
                if position < 0:
                    continue
                holder = self.dim_holder(operand, dim)
                sources[position].append((operand, operand_val, dim, holder))
                if literal_extent(self.env, operand_val.shape[dim]):
                    has_literal[position] = True
        conflict: str | None = None
        for position, entries in enumerate(sources):
            for operand, operand_val, dim, holder in entries:
                combined = _merge(merged[position], holder)
                if combined == CONFLICT and has_literal[position] and conflict is None:
                    conflict = (
                        f"{node.target} combines dim {dim} of {operand.name} "
                        f"(shape {list(operand_val.shape)}), held by "
                        f"{describe_holder(holder)}, with a dim held by "
                        f"{describe_holder(merged[position])}"
                    )
                merged[position] = combined
        if len(sizes) != value.ndim:
            return self.default(value), conflict
        return (
            tuple(
                (holder if holder != CONFLICT else SIZE) if lit else None
                for holder, lit in zip(merged, literal, strict=True)
            ),
            conflict,
        )

    def tile(self, block_id: int) -> Holder:
        """What holds a dim whose size names ``block_id``: the block if it
        is a tile wider than 1, else ``SIZE``."""
        tile = tile_block(self.env, self.config, block_id)
        return tile if tile is not None else SIZE

    def size_holder(self, size: object) -> Holder:
        if not isinstance(size, torch.SymInt) or self.env.known_equal(size, 1):
            return None
        block_id = self.env.get_block_id(size)
        return None if block_id is None else self.tile(block_id)

    def tile_held(self, node: object) -> list[tuple[int, int]]:
        """The literal dims of ``node`` that hold a tile, with the tile."""
        if not isinstance(node, Node):
            return []
        return [
            (dim, holder)
            for dim, holder in enumerate(self.holders.get(node, ()))
            if isinstance(holder, int)
        ]

    def propagate(self) -> None:
        """Grow ``holders`` to their fixpoint; then ``final`` is set and an
        unreached literal dim reads as held by size."""
        for _ in range(2 * len(self.graphs) + 2):
            changed = False
            for info in self.graphs:
                for node in info.graph.nodes:
                    holders = self.compute(info, node)
                    if holders is not None and self.holders.get(node) != holders:
                        self.holders[node] = holders
                        changed = True
            if not changed:
                break
        self.final = True
