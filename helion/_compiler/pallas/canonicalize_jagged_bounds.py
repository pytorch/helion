"""Canonicalizes per-group jagged row bounds on the Pallas device IR.

Kernels like grouped_gemm_jagged iterate over per-group row counts e - s and
index memory at s + tm.index. Left as-is, plan_tiling classifies s + tm.index
as an indirect index (one_hot gather/scatter).

This pass rewrites eligible loops from:

    s, e = offs[g], offs[g + 1]
    for tm, tn in hl.tile([0, 0], [e - s, N]):
        ... a[s + tm.index, tk] ...
        out[s + tm.index, tn] = ...

into:

    s, e = offs[g], offs[g + 1]
    for tm, tn in hl.tile([s, 0], [e, N]):
        ... a[tm.index, tk] ...
        out[tm.index, tn] = ...

The subsequent `remove_unnecessary_tile_index` pass then rewrites tm.index to tm
in load/store subscripts, allowing plan_tiling and ordered_carry to lower the
accesses via aligned DMA windows.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import TypeGuard

import torch

from ...language import _tracing_ops as ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language.atomic_ops import ATOMIC_OPS
from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource
from ..device_ir import DeviceIR
from ..device_ir import ForLoopGraphInfo
from ..device_ir import IfGraphInfo
from ..device_ir import NodeArgsGraphInfo
from ..device_ir import control_flow_parent_entries
from .memory_access import tensors_share_origin_or_storage
from .ordered_carry import capture_map
from .ordered_carry import may_skip_nonempty_group
from .ordered_carry import resolve_node
from .ordered_carry import store_aliases_a_load

if TYPE_CHECKING:
    from ..device_ir import GraphInfo

Node = torch.fx.Node
_M_USERS = (tile_ops.tile_index, creation_ops.full, tile_ops.tile_block_size)
_MEM_OPS = (memory_ops.load, memory_ops.store)


def _is_call(v: object, *targets: object) -> TypeGuard[Node]:
    return (
        isinstance(v, Node)
        and v.op == "call_function"
        and v.target in targets
        and not v.kwargs
    )


def _value(v: object) -> object:
    return v.meta.get("val") if isinstance(v, Node) else v


def _block_id(v: object) -> int | None:
    if _is_call(v, ops._get_symnode, torch.ops.aten.sym_size.int) and isinstance(
        s := _value(v), torch.SymInt
    ):
        return CompileEnvironment.current().get_block_id(s)
    return None


class _Graphs:
    """Cross-graph lookup table for resolving placeholders back to definitions."""

    def __init__(self, graphs: list[GraphInfo], g: int) -> None:
        self.graphs = graphs
        self.g = g
        self.env = CompileEnvironment.current()
        self.parents = control_flow_parent_entries(graphs)
        self.by_graph = {info.graph: info for info in graphs}
        self.outer = capture_map(graphs, self.parents)

    def resolve(self, value: object) -> object:
        return resolve_node(value, self.outer)

    def scope_nodes(self, info: GraphInfo) -> list[Node]:
        """Returns all FX nodes in info and its nested subgraphs."""
        scope = {info.graph}
        for g in reversed(self.graphs):
            if (e := self.parents.get(g.graph_id)) and e[0].graph in scope:
                scope.add(g.graph)
        return [n for g in self.graphs if g.graph in scope for n in g.graph.nodes]


def _load_1d(node: object) -> tuple[torch.Tensor, object] | None:
    if (
        _is_call(node, memory_ops.load)
        and isinstance(t := _value(node.args[0]), torch.Tensor)
        and isinstance(idx := node.args[1], (list, tuple))
        and len(idx) == 1
        and not (node.args[2] or node.args[3])
    ):
        return t, idx[0]
    return None


def _group_bounds(ctx: _Graphs, length: object) -> tuple[Node, Node] | None:
    """Matches length against offs[g + 1] - offs[g] on the 1-D grid ctx.g."""
    if not _is_call(length, torch.ops.aten.sub.Tensor) or len(length.args) != 2:
        return None
    end, start = (ctx.resolve(v) for v in length.args)
    # Check that start and end are 1-D loads from the same offsets tensor at
    # offs[g] and offs[g + 1] (or offs[1 + g]), where g is the grid block id.
    if not (
        isinstance(s := _load_1d(start), tuple)
        and isinstance(e := _load_1d(end), tuple)
        and s[0] is e[0]
        and _block_id(s[1]) == ctx.g
        and _is_call(adj := e[1], operator.add)
    ):
        return None
    g0, one = adj.args if type(adj.args[0]) is not int else adj.args[::-1]
    assert isinstance(start, Node) and isinstance(end, Node)
    return (
        (start, end)
        if type(one) is int and one == 1 and _block_id(g0) == ctx.g
        else None
    )


def _offset_uses(
    ctx: _Graphs, info: ForLoopGraphInfo, m: int, start: Node
) -> dict[Node, Node] | None:
    """Validates all uses of tile m in info and maps s + tm.index -> tm.index."""
    nodes = ctx.scope_nodes(info)
    adds: dict[Node, Node] = {}
    for node in nodes:
        # Reject loops with atomics or coordinate uses of tile m (e.g. tm.begin,
        # tm.end, tm.id); only tile_index, full, and tile_block_size may use m.
        if _is_call(node, *ATOMIC_OPS) or (
            _block_id(node) == m and not all(_is_call(u, *_M_USERS) for u in node.users)
        ):
            return None
        # Every tm.index must be offset by start (s + tm.index or tm.index + s)
        # and consumed directly by memory_ops.load or memory_ops.store.
        if _is_call(node, tile_ops.tile_index) and _block_id(node.args[0]) == m:
            if not node.users:
                return None
            for add in node.users:
                # aten.add.Tensor takes 2 summands (lhs, rhs); a 3rd positional
                # arg would be a non-default alpha multiplier. Since add is a
                # user of node (tm.index), one summand is node and the other
                # must resolve to start.
                if not (
                    _is_call(add, torch.ops.aten.add.Tensor) and len(add.args) == 2
                ):
                    return None
                lhs, rhs = add.args
                other = lhs if rhs is node else rhs
                if (
                    ctx.resolve(other) is not start
                    or not add.users
                    or not all(_is_call(u, *_MEM_OPS) for u in add.users)
                ):
                    return None
                adds[add] = node

    stores = [n for n in nodes if _is_call(n, memory_ops.store)]
    if not adds or len(stores) != 1 or stores[0].graph is not info.graph:
        return None
    # Validate every load/store that consumes s + tm.index (plus the loop's
    # single store).
    for node in {u for add in adds for u in add.users} | set(stores):
        tensor, sub = _value(node.args[0]), node.args[1]
        is_store = node.target is memory_ops.store
        extra_mask = node.args[3 if is_store else 2]
        if not (isinstance(tensor, torch.Tensor) and isinstance(sub, (list, tuple))):
            return None
        is_2d_or_higher_float = tensor.ndim >= 2 and tensor.is_floating_point()
        has_plain_subscripts = (
            is_2d_or_higher_float
            and len(sub) == tensor.ndim
            and extra_mask is None
            and all(_value(i) is not None for i in sub)
        )
        if not has_plain_subscripts:
            return None
        only_leading_dim_uses_offset = sub[0] in adds and not any(
            x in adds for x in (*sub[1:], *node.args[2:])
        )
        col = ctx.resolve(sub[-1])
        col_bid = _block_id(col.args[0] if _is_call(col, tile_ops.tile_index) else col)
        has_valid_col_tile = col_bid not in (None, m, ctx.g)
        if not (only_leading_dim_uses_offset and has_valid_col_tile):
            return None
        if is_store:
            assert col_bid is not None
            col_size = ctx.env.block_sizes[col_bid].size
            covers_full_2d_cols = (
                tensor.ndim == 2
                and col_bid not in DeviceIR.current().noncanonical_task_origin_block_ids
                and isinstance(col_size, (int, torch.SymInt))
                and ctx.env.known_equal(col_size, tensor.shape[1])
            )
            has_other_store_to_tensor = any(
                n is not node
                and _is_call(n, memory_ops.store)
                and isinstance(other := _value(n.args[0]), torch.Tensor)
                and tensors_share_origin_or_storage(other, tensor)
                for g in ctx.graphs
                for n in g.graph.nodes
            )
            if (
                not covers_full_2d_cols
                or store_aliases_a_load(tensor, ctx.graphs)
                or has_other_store_to_tensor
            ):
                return None
    return adds


def _capture_in_subgraph(ctx: _Graphs, value: Node, into: GraphInfo) -> Node:
    """Threads `value` from an ancestor graph into `into`.

    Returns the Node in into.graph that represents `value` (either an existing
    node if already in scope, or a newly added placeholder).
    """
    target = ctx.resolve(value)
    for n in into.graph.nodes:
        if ctx.resolve(n) is target:
            return n
    call, slot = ctx.parents[into.graph_id]
    outer = _capture_in_subgraph(ctx, value, ctx.by_graph[call.graph])
    old = call.args[slot]
    assert isinstance(old, (list, tuple))
    captures = [*old, outer]
    call.update_arg(slot, captures)
    first = next(n for n in into.graph.nodes if n.op != "placeholder")
    with into.graph.inserting_before(first):
        ph = into.graph.placeholder(f"{outer.name}_captured")
    ph.meta = {k: v for k, v in value.meta.items() if k != "lowering"}
    ctx.outer[ph] = outer
    assert isinstance(into, NodeArgsGraphInfo)
    into.node_args.append(outer)
    if isinstance(into, IfGraphInfo) and into.if_arg_names is not None:
        into.if_arg_names.append(ph.name)
    return ph


def _rewrite_loop(ctx: _Graphs, info: ForLoopGraphInfo) -> bool:
    """Tries to canonicalize one hl.tile loop info; returns True if rewritten."""
    entry = ctx.parents.get(info.graph_id)
    if entry is None or not _is_call(entry[0], ops._for_loop):
        return False
    call = entry[0]
    _, begins, ends, _ = call.args
    if (
        not isinstance(begins, (list, tuple))
        or not isinstance(ends, (list, tuple))
        or len(info.block_ids) > 2
        or not all(type(b) is int and b == 0 for b in begins)
        or any(u.users for u in call.users)
    ):
        return False
    for pos, m in enumerate(info.block_ids):
        bounds = _group_bounds(ctx, ctx.resolve(ends[pos]))
        if bounds is None or may_skip_nonempty_group(m, ctx.graphs):
            continue
        start, end = bounds
        if (adds := _offset_uses(ctx, info, m, start)) is None:
            continue
        site = ctx.by_graph[call.graph]
        begins, ends = list(begins), list(ends)
        begins[pos] = _capture_in_subgraph(ctx, start, site)
        ends[pos] = _capture_in_subgraph(ctx, end, site)
        call.update_arg(1, begins)
        call.update_arg(2, ends)
        DeviceIR.current().noncanonical_task_origin_block_ids.add(m)
        for add, idx in adds.items():
            add.replace_all_uses_with(idx)
            add.graph.erase_node(add)
        return True
    return False


def canonicalize_jagged_bounds(graphs: list[GraphInfo]) -> bool:
    """Rewrites every eligible row loop in graphs (runs once per kernel).

    Returns True if at least one loop was rewritten, False otherwise.
    """
    ir = DeviceIR.current()
    # Discard kernels without a single 1-D grid of unit block size (one group
    # per grid iteration) and canonical 0-based task origin.
    if not (
        len(ir.grid_block_ids) == 1
        and len(ir.grid_block_ids[0]) == 1
        and (g := ir.grid_block_ids[0][0]) not in ir.noncanonical_task_origin_block_ids
        and CompileEnvironment.current().block_sizes[g].block_size_source
        == FixedBlockSizeSource(1)
    ):
        return False
    ctx = _Graphs(graphs, g)
    return any(
        [_rewrite_loop(ctx, info) for info in graphs if type(info) is ForLoopGraphInfo]
    )
