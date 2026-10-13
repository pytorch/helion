"""Logical extents of literal tile dims in a positional root.

A full dim's FakeTensor size may already be its padded power of two: the
carry created by ``hl.zeros([tile, 40, 40])`` has meta shape ``[u0, 64, 64]``
and ``x[tile, :, hl.arange(48)]`` has meta shape ``[u0, u1, 64]``.  Positions
40..63 (or 48..63) are padding, yet the literal ``64`` names no block id, so
the positional evaluator cannot bound them from metadata alone.  The original
lengths survive in FX provenance: ``full`` shape arguments, ``iota`` lengths,
and the host dims a ``load`` slices or indexes.

``LogicalDomains`` records, once per root, one entry per dim of every live
value: an ``int`` logical length (positions at or beyond it are padding),
``TILE`` (an active grid/loop dim, bounded by its origin/end), ``BROADCAST``
(uniform along the dim, e.g. expanded from size 1, so its co-operands decide
the length), or ``UNKNOWN``.
Symbolic dims keep their metadata meaning; the table only decides literal
dims.  Entries never depend on emission state, so materialized snapshots, dot
and carry tiles keep the extents of the node they hold.

Unification: an elementwise consumer broadcasts right-aligned inputs and
skips inputs whose padded size is 1 and ``BROADCAST`` entries; all remaining
entries for a dim must agree, except that an operand spanning the padded
extent exactly defines every lane, so the result does too
(``elementwise_extent``). Any other disagreement yields ``UNKNOWN``.
Loop carries start from the input's entries and are re-solved until the
body result agrees. A broadcast dim can acquire a concrete length; any
later disagreement becomes ``UNKNOWN``. This is monotone, so a fixed point
exists. ``UNKNOWN`` is lazy: it
fails closed only when a literal dim of that value must be bounded (load,
``_mask_to``, sum, scan, store value or mask, or dot K), never by guessing a
length. A store needs no bound along a ``BROADCAST`` dim of its value or mask:
such a dim takes its length from co-operands, and the destination's own index
domain (``_Positional.destination_domain``) is the co-operand that bounds the
write.
"""

from __future__ import annotations

import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast

import sympy
import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_index
from ...language import view_ops
from ...language.matmul_ops import dot
from ..compile_environment import CompileEnvironment
from . import chained_matmul as cm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import ForLoopGraphInfo

Entry = int | str | None
TILE = "tile"
BROADCAST = "broadcast"
UNKNOWN: Entry = None
Extents = tuple[Entry, ...]


class UnprovenDomain(Exception):
    """A literal dim must be bounded but its logical length is unproven."""


CONTRACTIONS = (
    dot,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.baddbmm.default,
)


@dataclasses.dataclass(frozen=True)
class Contraction:
    """``lhs @ rhs (+ acc)`` over the last two dims.

    ``aten.mm``/``aten.bmm``/``aten.baddbmm`` reach the backend without
    ``alpha``/``beta``
    (``apply_dot_requirements`` rejects matmul kwargs), so the accumulator is
    always added with weight one.
    """

    lhs: Node
    rhs: Node
    acc: Node | None


def contraction_of(node: Node) -> Contraction | None:
    """The contraction ``node`` computes, or None for unproven arguments."""
    args = node.args
    if node.target is dot:
        acc = args[2] if len(args) > 2 else None
        if not isinstance(args[0], Node) or not isinstance(args[1], Node):
            return None
        return Contraction(args[0], args[1], acc if isinstance(acc, Node) else None)
    if node.kwargs or not all(isinstance(arg, Node) for arg in args):
        return None
    if node.target in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default) and (
        len(args) == 2
    ):
        return Contraction(cast("Node", args[0]), cast("Node", args[1]), None)
    if node.target is torch.ops.aten.baddbmm.default and len(args) == 3:
        return Contraction(
            cast("Node", args[1]), cast("Node", args[2]), cast("Node", args[0])
        )
    return None


def _unify(entries: Sequence[Entry]) -> Entry:
    defined = [entry for entry in entries if entry != BROADCAST]
    if not defined:
        return BROADCAST
    first = defined[0]
    return first if all(entry == first for entry in defined) else UNKNOWN


def _static(size: object) -> Entry:
    """A logical length from a host size, shape argument or iota length."""
    if isinstance(size, Node):
        size = size.meta.get("val")
    if isinstance(size, bool):
        return UNKNOWN
    if isinstance(size, int):
        return size
    if isinstance(size, torch.SymInt):
        env = CompileEnvironment.current()
        block = env.resolve_block_id(size)
        if block is not None:
            return _block_entry(block)
        expr = env.specialize_expr(cast("sympy.Expr", size._sympy_()))
        if isinstance(expr, sympy.Integer):
            return int(expr)
    return UNKNOWN


def _block_entry(block: int) -> Entry:
    env = CompileEnvironment.current()
    info = env.block_sizes[env.canonical_block_id(block)]
    if not info.reduction:
        return TILE
    return info.size if isinstance(info.size, int) else UNKNOWN


class LogicalDomains:
    def __init__(
        self,
        loop_info: Callable[[Node], ForLoopGraphInfo],
        branch_arms: Callable[[Node], Sequence[tuple[object, object]]],
        graph_of: Callable[[int], torch.fx.Graph],
        scan_leaves: Callable[[Node], tuple[Node, ...]],
    ) -> None:
        self.loop_info = loop_info
        self.branch_arms = branch_arms
        self.graph_of = graph_of
        self.scan_leaves = scan_leaves
        self.extents: dict[Node, Extents] = {}
        self.loop_carries: dict[Node, dict[int, Extents]] = {}
        # Contractions whose K is defined by an operand spanning the padded K
        # extent, and the K dim of each per-consumer ``_mask_to`` operand.
        self.promoted_k: set[Node] = set()
        self.lifted_masks: dict[Node, int] = {}

    # ------------------------------------------------------------------ query

    def k_promoted(self, node: Node) -> bool:
        return node in self.promoted_k

    def lifted_mask_dim(self, node: Node) -> int | None:
        return self.lifted_masks.get(node)

    def broadcast_dim(self, node: Node, dim: int) -> bool:
        """Whether ``dim`` of ``node`` has no length of its own (``BROADCAST``)."""
        return self.extents.get(node, ())[dim : dim + 1] == (BROADCAST,)

    def static(self, node: Node, dim: int) -> int:
        """Logical length of literal dim ``dim``; fail closed when unproven."""
        entry = self.extents.get(node, ())[dim : dim + 1]
        if not entry or not isinstance(entry[0], int) or isinstance(entry[0], bool):
            raise UnprovenDomain(f"unproven logical extent of {node.name} dim {dim}")
        return entry[0]

    # ---------------------------------------------------------------- analysis

    def visit(self, graph: torch.fx.Graph) -> None:
        for node in graph.nodes:
            if node.op == "call_function":
                self.visit_node(node)

    def of(self, node: object, rank: int) -> Extents:
        """Entries of a tensor input, or ``rank`` unknown dims if absent."""
        if isinstance(node, Node) and node in self.extents:
            return self.extents[node]
        return (UNKNOWN,) * rank

    def record(self, node: Node, provenance: Sequence[Entry] | None) -> None:
        """Symbolic dims keep their block meaning; literal dims use provenance."""
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return
        padded = cm._shape(node)
        if provenance is not None and len(provenance) != value.ndim:
            provenance = None
        entries: list[Entry] = []
        for dim, size in enumerate(value.shape):
            if not isinstance(size, int):
                block = CompileEnvironment.current().resolve_block_id(size)
                entries.append(UNKNOWN if block is None else _block_entry(block))
            elif size == 1:
                entries.append(1)
            else:
                entry = UNKNOWN if provenance is None else provenance[dim]
                if isinstance(entry, int) and not 0 < entry <= padded[dim]:
                    entry = UNKNOWN
                entries.append(entry)
        self.extents[node] = tuple(entries)

    def broadcast(self, node: Node, inputs: Sequence[object]) -> list[Entry]:
        rank = node.meta["val"].ndim
        columns: list[list[Entry]] = [[] for _ in range(rank)]
        for value in inputs:
            if not isinstance(value, Node) or not isinstance(
                value.meta.get("val"), torch.Tensor
            ):
                continue
            entries, padded = self.of(value, value.meta["val"].ndim), cm._shape(value)
            offset = rank - len(entries)
            if offset < 0:
                return [UNKNOWN] * rank
            for dim, (entry, size) in enumerate(zip(entries, padded, strict=True)):
                if size != 1:
                    columns[dim + offset].append(entry)
        padded_out = cm._shape(node)
        return [
            self.elementwise_extent(column, padded_out[dim]) if column else 1
            for dim, column in enumerate(columns)
        ]

    @staticmethod
    def elementwise_extent(column: Sequence[Entry], padded: int) -> Entry:
        """Unify one output dim of an elementwise op.

        An operand that spans the whole padded extent (``hl.full([t, 512])``
        with ``next_power_of_2`` sizes) defines every lane; shorter operands
        contribute their padding values there (zero for a masked load), as in
        every padded Helion tile.  Two partial extents that disagree stay
        UNKNOWN.
        """
        unified = _unify(column)
        defined = [entry for entry in column if entry != BROADCAST]
        if (
            unified is UNKNOWN
            and padded in defined
            and all(isinstance(entry, int) for entry in defined)
        ):
            return padded
        return unified

    def visit_node(self, node: Node) -> None:
        target = node.target
        if target is _tracing_ops._host_tensor:
            return
        if target is memory_ops.load:
            self.record(node, self.load(node))
        elif target is torch.ops.prims.iota.default:
            self.record(node, [_static(node.args[0])])
        elif target is creation_ops.full:
            shape = cast("Sequence[object]", node.args[0])
            self.record(node, [_static(size) for size in shape])
        elif target is torch.ops.aten.full.default:
            # Device torch factories are padded to powers of two while tracing
            # (``patch_tensor_factories``), so their shape arguments are not
            # logical lengths. The value is uniform: co-operands decide.
            self.record(node, [BROADCAST] * node.meta["val"].ndim)
        elif target is tile_index:
            self.record(node, [TILE])
        elif target in (_tracing_ops._new_var, _tracing_ops._mask_to):
            source = cast("Node", node.args[0])
            self.record(node, self.of(source, source.meta["val"].ndim))
        elif target in cm._VIEWS:
            self.record(node, self.view(node))
        elif target is torch.ops.aten.where.self:
            self.record(node, self.broadcast(node, node.args))
        elif target is torch.ops.aten.sum.dim_IntList:
            self.record(node, self.reduce(node))
        elif target in CONTRACTIONS:
            self.record(node, self.contraction(node))
        elif target is scan_ops._associative_scan:
            leaves = self.scan_leaves(node)
            rank = leaves[0].meta["val"].ndim
            unified = [
                _unify(column)
                for column in zip(
                    *(self.of(leaf, rank) for leaf in leaves), strict=True
                )
            ]
            self.extents[node] = tuple(unified)  # tuple scans: per-output getitem
            self.record(node, unified)
        elif target is operator.getitem:
            source = cast("Node", node.args[0])
            if source.target is _tracing_ops._for_loop:
                carry = self.loop_carries.get(source, {})
                self.record(node, carry.get(cast("int", node.args[1])))
            elif source.target is scan_ops._associative_scan:
                self.record(node, self.extents.get(source))
            else:
                self.record(node, None)
        elif target is _tracing_ops._phi:
            # The emitter binds the merge to its loop result (zero trips are
            # exact because that tile starts as the loop input).
            after = cast("Node", node.args[1])
            self.record(node, self.of(after, node.meta["val"].ndim))
        elif target is _tracing_ops._for_loop:
            self.loop(node)
        elif target is _tracing_ops._if:
            for graph_id, ports in self.branch_arms(node):
                graph = self.graph_of(cast("int", graph_id))
                self.bind_ports(graph, cast("Sequence[object]", ports))
                self.visit(graph)
        elif (inputs := cm._pointwise_inputs(node)) is not None:
            self.record(node, self.broadcast(node, inputs))
        else:
            self.record(node, None)

    def bind_ports(
        self,
        graph: torch.fx.Graph,
        ports: Sequence[object],
        carried: dict[int, Extents] | None = None,
    ) -> None:
        placeholders = [n for n in graph.nodes if n.op == "placeholder"]
        for index, (placeholder, port) in enumerate(
            zip(placeholders, ports, strict=False)
        ):
            value = placeholder.meta.get("val")
            if not isinstance(value, torch.Tensor):
                continue
            if carried is not None and index in carried:
                self.record(placeholder, carried[index])
            else:
                self.record(placeholder, self.of(port, value.ndim))

    def loop(self, node: Node) -> None:
        info = self.loop_info(node)
        interface = info.loop_interface
        assert interface is not None
        args = cast("Sequence[Node]", node.args[3])
        graph = info.graph
        results = cast(
            "Sequence[Node]", next(n for n in graph.nodes if n.op == "output").args[0]
        )
        carried = {
            carry.input_index: self.of(
                args[carry.input_index], args[carry.input_index].meta["val"].ndim
            )
            for carry in interface.carries
        }
        # Each dim can change twice: BROADCAST -> known -> UNKNOWN. Include
        # one final pass to visit the body with the converged carry entries.
        for _ in range(1 + 2 * sum(len(e) for e in carried.values())):
            self.bind_ports(graph, args, carried)
            self.visit(graph)
            following = {
                carry.input_index: tuple(
                    _unify([before, after])
                    for before, after in zip(
                        carried[carry.input_index],
                        self.of(
                            results[carry.output_index],
                            len(carried[carry.input_index]),
                        ),
                        strict=True,
                    )
                )
                for carry in interface.carries
            }
            if following == carried:
                break
            carried = following
        self.loop_carries[node] = {
            carry.output_index: carried[carry.input_index]
            for carry in interface.carries
        }

    def load(self, node: Node) -> list[Entry] | None:
        source = cast("Node", node.args[0])
        selectors = cast("Sequence[object]", node.args[1])
        if source.target is not _tracing_ops._host_tensor:
            # Internal loads of a tile value with only full/new-axis selectors.
            if not all(index is None or index == slice(None) for index in selectors):
                return None
            entries = iter(self.of(source, source.meta["val"].ndim))
            return [
                1 if index is None else next(entries, UNKNOWN) for index in selectors
            ]
        return self.host_index(source.meta["val"], selectors)

    def host_index(
        self, tensor: torch.Tensor, selectors: Sequence[object]
    ) -> list[Entry] | None:
        """Mirror ``SubscriptIndexing.compute_shape`` with logical lengths."""
        from ..host_function import HostFunction
        from ..variable_origin import BlockSizeOrigin

        env = CompileEnvironment.current()
        sizes = list(tensor.shape)
        values = [s.meta.get("val") if isinstance(s, Node) else s for s in selectors]
        tensors = [
            (selector, value)
            for selector, value in zip(selectors, values, strict=True)
            if isinstance(value, torch.Tensor)
        ]
        broadcast_indexers = env.should_broadcast_tensor_indexers(values)
        result: list[Entry] = []
        for selector, value in zip(selectors, values, strict=True):
            if value is None:
                result.append(1)
                continue
            if not sizes:
                return None
            size = sizes.pop(0)
            if isinstance(value, int):
                continue
            if isinstance(value, torch.SymInt):
                symbol = value._sympy_()
                origin = HostFunction.current().expr_to_origin.get(symbol)
                if origin is not None and isinstance(origin.origin, BlockSizeOrigin):
                    result.append(TILE)
                continue
            if isinstance(value, slice):
                if value == slice(None):
                    result.append(_static(size))
                elif value.step is None and all(
                    isinstance(v, int) for v in (value.start, value.stop)
                ):
                    start, stop = cast("int", value.start), cast("int", value.stop)
                    result.append(max(0, stop - start))
                else:
                    result.append(UNKNOWN)
                continue
            if isinstance(value, torch.Tensor):
                if not broadcast_indexers:
                    result.extend(self.indexer_dims(cast("Node", selector), value))
                elif tensors and selector is tensors[0][0]:
                    result.extend(self.indexer_broadcast(tensors))
                continue
            return None
        return None if sizes else result

    def indexer_dims(self, selector: Node, value: torch.Tensor) -> list[Entry]:
        if value.ndim == 0:
            return []
        env = CompileEnvironment.current()
        entries = self.of(selector, value.ndim)
        for entry, size in zip(entries, value.shape, strict=True):
            if env.size_hint(size) != 1:
                return [entry]
        return [1]

    def indexer_broadcast(
        self, tensors: Sequence[tuple[object, torch.Tensor]]
    ) -> list[Entry]:
        env = CompileEnvironment.current()
        if len(tensors) > 1 and all(value.ndim == 1 for _, value in tensors):
            return [self.of(selector, 1)[0] for selector, _ in tensors]  # Cartesian
        rank = max(value.ndim for _, value in tensors)
        columns: list[list[Entry]] = [[] for _ in range(rank)]
        for selector, value in tensors:
            entries = self.of(selector, value.ndim)
            offset = rank - value.ndim
            for dim, (entry, size) in enumerate(zip(entries, value.shape, strict=True)):
                if env.size_hint(size) != 1:
                    columns[dim + offset].append(entry)
        return [_unify(column) if column else 1 for column in columns]

    def view(self, node: Node) -> list[Entry] | None:
        source = cast("Node", node.args[0])
        rank = source.meta["val"].ndim
        entries = list(self.of(source, rank))
        target = node.target
        if target is torch.ops.aten.permute.default:
            order = [axis % rank for axis in cast("Sequence[int]", node.args[1])]
            return [entries[axis] for axis in order]
        if target in (torch.ops.aten.t.default, torch.ops.aten.transpose.int):
            a, b = (
                (0, 1)
                if target is torch.ops.aten.t.default
                else (
                    cast("int", node.args[1]) % rank,
                    cast("int", node.args[2]) % rank,
                )
            )
            entries[a], entries[b] = entries[b], entries[a]
            return entries
        if target is torch.ops.aten.unsqueeze.default:
            axis = cast("int", node.args[1]) % (rank + 1)
            return [*entries[:axis], 1, *entries[axis:]]
        if target is torch.ops.aten.squeeze.dim:
            axis = cast("int", node.args[1]) % rank
            return (
                entries[:axis] + entries[axis + 1 :]
                if cm._shape(source)[axis] == 1
                else entries
            )
        if target is torch.ops.aten.expand.default:
            sizes = cast("Sequence[object]", node.args[1])
            offset = len(sizes) - rank
            padded = cm._shape(source)
            result: list[Entry] = []
            out_padded = cm._shape(node)
            for dim, size in enumerate(sizes):
                src = dim - offset
                if src >= 0 and (padded[src] != 1 or size == -1):
                    result.append(entries[src])
                    continue
                # A requested size below the padded extent is a logical
                # length; one equal to it (often ``x.shape``, already padded)
                # is only a broadcast whose co-operands decide the length.
                entry = _static(size)
                below = isinstance(entry, int) and entry < out_padded[dim]
                result.append(entry if below or entry == TILE else BROADCAST)
            return result
        if target is view_ops.subscript:
            result = []
            remaining = iter(entries)
            for index in cast("Sequence[object]", node.args[1]):
                if index is None:
                    result.append(1)
                elif isinstance(index, int):
                    next(remaining, UNKNOWN)
                elif index == slice(None):
                    result.append(next(remaining, UNKNOWN))
                elif isinstance(index, Node) and isinstance(
                    value := index.meta.get("val"), torch.Tensor
                ):
                    # A gather is as long as its indexer, as for host loads.
                    next(remaining, UNKNOWN)
                    result.extend(self.indexer_dims(index, value))
                else:
                    next(remaining, UNKNOWN)
                    result.append(UNKNOWN)
            return result
        # reshape/view flatten the padded layout, so a static dim with padding
        # cannot be followed; without static padding every literal is exact.
        if all(
            entry == TILE or (isinstance(entry, int) and entry == size)
            for entry, size in zip(entries, cm._shape(source), strict=True)
        ):
            return list(cm._shape(node))
        return None

    def promote_k(
        self, node: Node, description: Contraction, left: Entry, right: Entry
    ) -> None:
        """Apply ``elementwise_extent`` to K, as every Helion backend does.

        When one operand spans the padded K extent exactly and the other is a
        shorter static extent, K covers the padded extent: the shorter
        operand contributes its padding values (Triton's ``tl.dot`` runs over
        the whole padded K and masks the per-consumer ``_mask_to`` by the
        unified, exact K).  An active loop/grid K (``TILE``, runtime end),
        UNKNOWN, or two partial extents keep their bounds.
        """
        lhs, rhs = description.lhs, description.rhs
        self.promoted_k.discard(node)
        for operand in (lhs, rhs):
            if self.lifted_masks.get(operand) is not None and node in operand.users:
                del self.lifted_masks[operand]
        padded = cm._shape(lhs)[-1]
        if (
            padded != cm._shape(rhs)[-2]
            or _unify([left, right]) is not UNKNOWN
            or self.elementwise_extent([left, right], padded) != padded
        ):
            return
        self.promoted_k.add(node)
        for operand, dim in (
            (lhs, lhs.meta["val"].ndim - 1),
            (rhs, rhs.meta["val"].ndim - 2),
        ):
            if operand.target is _tracing_ops._mask_to and list(operand.users) == [
                node
            ]:
                self.lifted_masks[operand] = dim

    def reduce(self, node: Node) -> list[Entry] | None:
        source = cast("Node", node.args[0])
        rank = source.meta["val"].ndim
        dims = cast("Sequence[int]", node.args[1]) or range(rank)
        axes = {dim % rank for dim in dims}
        keepdim = bool(node.args[2]) if len(node.args) > 2 else False
        entries = self.of(source, rank)
        return [
            1 if axis in axes else entry
            for axis, entry in enumerate(entries)
            if keepdim or axis not in axes
        ]

    def contraction(self, node: Node) -> list[Entry] | None:
        if (description := contraction_of(node)) is None:
            return None
        lhs, rhs = description.lhs, description.rhs
        rank = node.meta["val"].ndim
        left, right = (
            self.of(lhs, lhs.meta["val"].ndim),
            self.of(rhs, rhs.meta["val"].ndim),
        )
        left_padded, right_padded = cm._shape(lhs), cm._shape(rhs)
        batch: list[Entry] = []
        for dim in range(rank - 2):
            column = [
                entries[i]
                for entries, padded in ((left, left_padded), (right, right_padded))
                if 0 <= (i := dim - (rank - len(entries))) < len(entries) - 2
                and padded[i] != 1
            ]
            batch.append(_unify(column) if column else 1)
        result = [*batch, left[-2], right[-1]]
        self.promote_k(node, description, left[-1], right[-2])
        acc = description.acc
        if isinstance(acc, Node) and isinstance(acc.meta.get("val"), torch.Tensor):
            accumulated = self.broadcast(node, [acc])
            result = [
                _unify([own, other]) if other != 1 else own
                for own, other in zip(result, accumulated, strict=True)
            ]
        return result
