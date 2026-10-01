"""Positional whole-root fallback for values whose dims share a canonical axis.

Ordinary CuTe lowering binds one per-thread coordinate per canonical block id.
A value with two non-unit dims on one canonical id (``x @ x.T`` with C == D,
``idx[:, None] > idx[None, :]``) therefore only holds its diagonal, and dots
whose K shares a canonical id with M or N cannot be lowered.  This module
evaluates such a root by tensor *position*:

* pure expressions are evaluated at explicit coordinates by the existing
  ``chained_matmul._Expression`` evaluator;
* dot results and loop carries are materialized in typed shared tiles;
* every effect is one CTA-cooperative strided phase followed by
  ``sync_threads``; loops are CTA-uniform with current/next carry tiles.

The ordinary layout also cannot contract an ``hl.dot`` whose K block is a
persistent reduction split across a synthetic lane loop (K exceeds its thread
budget): that loop wraps the whole root, so the cross-thread reduction sees one
lane of K and, inside a serial loop, each lane carries its own partial state.
Such contractions are treated like collisions.

A repeated-axis value consumed only as an operand of rank-2 ``aten.mm`` nodes
with a static serial K (``y[:, :]`` in ``x[tile, :] @ y[:, :]``) is not a
collision: the per-node direct mm lowering (grouped-N MMA, serial reload, or
its clean rejection) re-reads such operands from host memory by position and
never through the per-thread layout.  Those mm nodes are recorded in
``direct_mm_collision_owners``; ``codegen_mm_cute`` fails closed if one
would leave the direct path.

Likewise the pointer tile and stack load of a native 2D stack store
(``out[:, :, t] = hl.stacktensor_like(x, ptrs[:, :])[t]``) are not
collisions: ``_codegen_cute_store_stack_load`` loops the second pointer axis
and re-reads ``ptrs`` from host memory by position.  Those stores are recorded
in ``stack_store_collision_owners`` and fail closed off that route.

It is planned only after every native/fast route has declined (see
``CuteBackend.pre_codegen``) and only when a live collision exists.  Once a
collision exists, any unsupported construct raises ``BackendUnsupported``:
the ordinary lowering would otherwise silently compute a wrong value.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_index
from ...language import tile_ops
from ...language.matmul_ops import dot
from ..compile_environment import CompileEnvironment
from . import chained_matmul as cm
from . import positional_host_effects as host_effects
from .positional_domains import CONTRACTIONS
from .positional_domains import LogicalDomains
from .positional_domains import UnprovenDomain
from .positional_domains import contraction_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_function import DeviceFunction
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import GraphInfo
    from ..device_ir import HelperFunctionGraphInfo
    from ..generate_ast import GenerateAST
    from ..variable_origin import GridOrigin


class _Unsupported(Exception):
    pass


_EXPRESSION_TARGETS = {
    _tracing_ops._host_tensor,
    _tracing_ops._get_symnode,
    _tracing_ops._mask_to,
    _tracing_ops._new_var,
    tile_ops.tile_begin,
    tile_ops.tile_id,
    tile_index,
    torch.ops.prims.iota.default,
    torch.ops.aten.scalar_tensor.default,
    torch.ops.aten.full.default,
    creation_ops.full,
    memory_ops.load,
    torch.ops.aten.where.self,
    torch.ops.aten.sym_size.int,
    *cm._VIEWS,
    *cm._SCALAR_BINARY,
}
# Scan combine helpers are closed pointwise graphs over [1]-shaped operands.
_SCAN_HELPER_TARGETS = {
    _tracing_ops._new_var,
    torch.ops.aten.where.self,
    torch.ops.aten.scalar_tensor.default,
    creation_ops.full,
    *cm._VIEWS,
}
_EFFECT_TARGETS = {
    _tracing_ops._for_loop,
    _tracing_ops._phi,
    operator.getitem,
    memory_ops.store,
    dot,
}
_MATERIALIZED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Host-effect snapshots only copy and re-read a loaded value, so integer
# tensors (e.g. offsets that index a later store) restore exactly too.
_SNAPSHOT_DTYPES = (
    *_MATERIALIZED_DTYPES,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)
_SCALAR_COMPARISONS = {
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
    operator.eq,
    operator.ne,
}


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def _axis_identity(size: object) -> tuple[str, int] | None:
    env = CompileEnvironment.current()
    block = env.resolve_block_id(size)
    if block is not None:
        return ("block", env.canonical_block_id(block))
    if isinstance(size, int) and size > 1:
        # Loop carry shapes may already contain padded literal dimensions.
        # Ordinary layouts also reuse a coordinate for equal static sizes.
        return ("static", size)
    return None


def _live_nodes(graph: torch.fx.Graph) -> set[Node]:
    """Nodes reaching an effect or a loop result; dead values cannot collide."""
    pending = [
        node
        for node in graph.nodes
        if node.op == "output"
        or node.target in (memory_ops.store, dot, _tracing_ops._for_loop)
    ]
    live: set[Node] = set()
    while pending:
        node = pending.pop()
        if node not in live:
            live.add(node)
            pending.extend(node.all_input_nodes)
    return live


def _direct_mm_owners(node: Node, live: set[Node]) -> list[Node] | None:
    """The mm nodes that read ``node`` by position, if they are its only users."""
    from .matmul_utils import cute_static_serial_matmul_k_extent

    users = [user for user in node.users if user in live]
    if not users or any(
        user.op != "call_function"
        or user.target is not torch.ops.aten.mm.default
        or node not in user.args[:2]
        or cute_static_serial_matmul_k_extent(
            cast("Node", user.args[0]), cast("Node", user.args[1])
        )
        is None
        for user in users
    ):
        return None
    return users


def _native_stack_store(store: Node) -> tuple[Node, Node] | None:
    """(stack load, pointer tile) of a native 2D stack store, else None.

    Mirrors the 2D branch of ``_codegen_cute_store_stack_load``:
    ``out[:, :, ...] = hl.stacktensor_like(x, ptrs[:, :])[...]``.
    """
    if store.op != "call_function" or store.target is not memory_ops.store:
        return None
    out, selectors, value = store.args[:3]
    if not isinstance(out, Node) or out.target is not _tracing_ops._host_tensor:
        return None
    if not isinstance(value, Node) or value.target is not memory_ops.load:
        return None
    # The native 2D route applies store masks but cannot preserve a load mask.
    if len(value.args) > 2 and value.args[2] is not None:
        return None
    stack = value.args[0]
    if not isinstance(stack, tuple) or len(stack) != 2:
        return None
    ptrs = stack[1]
    if not isinstance(ptrs, Node) or ptrs.target is not memory_ops.load:
        return None
    host = ptrs.args[0]
    if not isinstance(host, Node) or host.target is not _tracing_ops._host_tensor:
        return None

    def full(index: object) -> bool:
        return isinstance(index, slice) and index == slice(None)

    ptr_selectors = cast("Sequence[object]", ptrs.args[1])
    selectors = cast("Sequence[object]", selectors)
    if (
        host.meta["val"].ndim != 2
        or len(ptr_selectors) != 2
        or not all(map(full, ptr_selectors))
        or len(selectors) < 3
        or not all(map(full, selectors[:2]))
    ):
        return None
    return value, ptrs


def _stack_store_owners(node: Node, live: set[Node]) -> list[Node] | None:
    """Native 2D stack stores that read ``node`` by position, if its only users.

    ``node`` is either the stack load (its only consumer is the store) or the
    pointer tile (its only consumers are such stack loads).
    """
    stack = node.args[0] if node.target is memory_ops.load else None
    loads = (
        [node]
        if isinstance(stack, tuple)
        else [user for user in node.users if user in live]
    )
    owners: list[Node] = []
    for load in loads:
        stores = [user for user in load.users if user in live]
        if not stores:
            return None
        for store in stores:
            operands = _native_stack_store(store)
            if operands is None or operands[0] is not load:
                return None
            if load is not node and operands[1] is not node:
                return None
            owners.append(store)
    return owners or None


def find_axis_collisions(
    graphs: Sequence[GraphInfo],
    direct_mm_owners: set[Node] | None = None,
    stack_store_owners: set[Node] | None = None,
) -> tuple[str, ...]:
    """Live repeated-axis values and dots whose K repeats M or N.

    With ``direct_mm_owners``, values read only by direct-K ``aten.mm`` nodes
    are not collisions; their mm nodes are added to the set instead.  With
    ``stack_store_owners``, the same holds for native 2D stack stores.
    """
    found: list[str] = []
    for graph in graphs:
        live = _live_nodes(graph.graph)
        for node in graph.graph.nodes:
            if node not in live or node.target is _tracing_ops._host_tensor:
                continue
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor):
                ids = [
                    block
                    for size in value.shape
                    if not (isinstance(size, int) and size == 1)
                    if (block := _axis_identity(size)) is not None
                ]
                if len(ids) != len(set(ids)):
                    if (
                        direct_mm_owners is not None
                        and (owners := _direct_mm_owners(node, live)) is not None
                    ):
                        direct_mm_owners.update(owners)
                    elif (
                        stack_store_owners is not None
                        and (owners := _stack_store_owners(node, live)) is not None
                    ):
                        stack_store_owners.update(owners)
                    else:
                        found.append(f"{graph.graph_id}:{node.name}")
                    continue
            if node.op == "call_function" and node.target is dot:
                lhs, rhs = (cast("Node", a).meta["val"] for a in node.args[:2])
                m, k, n = (
                    _axis_identity(lhs.shape[-2]),
                    _axis_identity(lhs.shape[-1]),
                    _axis_identity(rhs.shape[-1]),
                )
                if k is not None and k in (m, n):
                    found.append(f"{graph.graph_id}:{node.name}:dot")
    return tuple(found)


def find_lane_split_contractions(
    graphs: Sequence[GraphInfo], df: DeviceFunction
) -> tuple[str, ...]:
    """Live ``hl.dot`` nodes whose K block has a synthetic reduction lane loop."""
    from ..reduction_strategy import PersistentReductionStrategy

    env = CompileEnvironment.current()
    split = {
        env.canonical_block_id(strategy.block_index)
        for strategy in df.tile_strategy.strategies
        if isinstance(strategy, PersistentReductionStrategy)
        and strategy._synthetic_cute_lane_var is not None
    }
    if not split:
        return ()
    found: list[str] = []
    for graph in graphs:
        live = _live_nodes(graph.graph)
        for node in graph.graph.nodes:
            if node in live and node.op == "call_function" and node.target is dot:
                k = _axis_identity(cast("Node", node.args[0]).meta["val"].shape[-1])
                if k is not None and k[0] == "block" and k[1] in split:
                    found.append(f"{graph.graph_id}:{node.name}:lane-split K")
    return tuple(found)


def _fast_route_claimed(df: DeviceFunction) -> bool:
    """Routes chosen while building tile strategies keep their existing owner."""
    from ..tile_strategy import PerThreadNDTileStrategy

    state = df.cute_state
    return (
        state.attention_flash_block_ids is not None
        or state.attention_flash_bwd_block_ids is not None
        or any(
            isinstance(strategy, PerThreadNDTileStrategy) and strategy.mma_mode
            for strategy in df.tile_strategy.strategies
        )
    )


# --------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PositionalRootPlan:
    root_graph_id: int
    # (canonical grid block id, static extent, block size), pid order.
    axes: tuple[tuple[int, int, int], ...]
    threads: int
    shared_bytes: int
    shared_capacity: int
    store: Node
    collisions: tuple[str, ...]
    # Logical lengths of literal (possibly padded) dims of every live value.
    domains: LogicalDomains


def _storage(node: Node) -> torch.UntypedStorage | None:
    if node.target is not _tracing_ops._host_tensor:
        return None
    value = node.meta.get("val")
    return value.untyped_storage() if isinstance(value, torch.Tensor) else None


def _tile_bytes(
    node: Node, dtypes: tuple[torch.dtype, ...] = _MATERIALIZED_DTYPES
) -> int:
    value = node.meta["val"]
    if value.dtype not in dtypes:
        raise _Unsupported(f"materialized {value.dtype} value {node.name}")
    size = max(1, math.prod(cm._shape(node))) * value.element_size()
    return (size + 127) // 128 * 128


def _branch_test(node: Node, df: DeviceFunction) -> bool | str:
    """A static branch decision, or a CTA-uniform host-scalar predicate.

    Every phase inside a branch synchronizes the CTA, so a runtime predicate
    is admitted only when all of its symbols are host scalars (kernel
    arguments), never thread, tile, grid or tensor values.
    """
    from ..device_ir import IfGraphInfo
    from ..host_function import HostFunction

    if node.users:
        raise _Unsupported(f"branch with merged outputs at {node.name}")
    info = HostFunction.current().device_ir.graphs[cast("int", node.args[1])]
    test = node.args[0]
    value = test.meta.get("val") if isinstance(test, Node) else test
    selected = df.evaluate_constexpr_condition(value)
    if selected is not None:
        return selected
    if (
        not isinstance(info, IfGraphInfo)
        or info.predicate_is_tensor
        or not isinstance(value, torch.SymBool)
    ):
        raise _Unsupported(f"non-uniform runtime branch at {node.name}")
    env = CompileEnvironment.current()
    expr = env.specialize_expr(cast("sympy.Expr", value.node.expr))
    origins = HostFunction.current().expr_to_origin
    for symbol in expr.free_symbols:
        info_origin = origins.get(symbol)
        if info_origin is None or not info_origin.origin.is_host():
            raise _Unsupported(f"non-host runtime branch symbol {symbol}")
    return df.sympy_expr(expr)


def _branch_arms(node: Node, df: DeviceFunction) -> list[tuple[object, object]]:
    """Live ``(graph id, ports)`` arms: the selected one, or both at runtime."""
    decision = _branch_test(node, df)
    then_arm, else_arm = (node.args[1], node.args[3]), (node.args[2], node.args[4])
    arms = (
        [then_arm, else_arm]
        if not isinstance(decision, bool)
        else ([then_arm] if decision else [else_arm])
    )
    return [arm for arm in arms if arm[0] is not None]


def _live_graphs(
    graphs: Sequence[GraphInfo], df: DeviceFunction
) -> tuple[GraphInfo, ...]:
    """Root, reachable loops and only the config-selected branch bodies."""
    from ..device_ir import ElseGraphInfo
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import HelperFunctionGraphInfo
    from ..device_ir import IfGraphInfo
    from ..device_ir import RootGraphInfo

    by_id = {graph.graph_id: graph for graph in graphs}
    roots = [graph for graph in graphs if isinstance(graph, RootGraphInfo)]
    if len(roots) != 1 or any(
        not isinstance(
            graph,
            (
                RootGraphInfo,
                ForLoopGraphInfo,
                IfGraphInfo,
                ElseGraphInfo,
                # Pure scan combine functions, admitted by their scan node.
                HelperFunctionGraphInfo,
            ),
        )
        for graph in graphs
    ):
        raise _Unsupported("requires one root, counted loops and static branches")
    live: list[GraphInfo] = []
    pending: list[GraphInfo] = [roots[0]]
    while pending:
        graph = pending.pop()
        live.append(graph)
        for node in graph.graph.nodes:
            if node.target is _tracing_ops._for_loop:
                pending.append(by_id[cast("int", node.args[0])])
            elif node.target is _tracing_ops._if:
                pending.extend(
                    by_id[cast("int", graph_id)]
                    for graph_id, _ in _branch_arms(node, df)
                )
    return tuple(live)


def _store_shape(node: Node, df: DeviceFunction) -> tuple[int, ...]:
    from ..indexing_strategy import SubscriptIndexing

    tensor = cast("Node", node.args[0]).meta["val"]
    selectors = [
        index.meta["val"] if isinstance(index, Node) else index
        for index in cast("Sequence[object]", node.args[1])
    ]
    env = CompileEnvironment.current()
    target = env.new_index_result(
        tensor, SubscriptIndexing.compute_shape(tensor, selectors)
    )
    return tuple(cm._get_tile_shape(target, env, df.config))


def _admit_store_value(node: Node, df: DeviceFunction) -> None:
    value = node.args[2]
    if not isinstance(value, Node) or not isinstance(
        value.meta.get("val"), torch.Tensor
    ):
        raise _Unsupported(f"scalar store at {node.name}")
    target_shape, value_shape = _store_shape(node, df), cm._shape(value)
    if len(target_shape) != len(value_shape) or any(
        source != target and source != next_power_of_2(target)
        for source, target in zip(value_shape, target_shape, strict=True)
    ):
        raise _Unsupported(f"broadcast-valued store at {node.name}")


def _store_mask(node: Node) -> object:
    return node.args[3] if len(node.args) > 3 else None


def _admit_store_mask(node: Node, df: DeviceFunction) -> None:
    """A store mask is a literal bool or a Boolean tensor that broadcasts,
    right-aligned, to the destination (a dim may be padded like the value)."""
    mask = _store_mask(node)
    if mask is None or isinstance(mask, bool):
        return
    if (
        not isinstance(mask, Node)
        or not isinstance(value := mask.meta.get("val"), torch.Tensor)
        or value.dtype != torch.bool
    ):
        raise _Unsupported(f"non-Boolean store mask at {node.name}")
    target, shape = _store_shape(node, df), cm._shape(mask)
    if len(shape) > len(target) or any(
        size not in (1, extent, next_power_of_2(extent))
        for size, extent in zip(shape, target[len(target) - len(shape) :], strict=True)
    ):
        raise _Unsupported(f"store mask shape at {node.name}")


def _admit(graphs: Sequence[GraphInfo], df: DeviceFunction) -> tuple[int, int]:
    """Validate every live node; return (root graph id, shared-memory bytes)."""
    from ..device_ir import ForLoopGraphInfo

    graphs = _live_graphs(graphs, df)
    shared = 0
    for graph in graphs:
        if isinstance(graph, ForLoopGraphInfo) and graph.loop_interface is None:
            raise _Unsupported("synthetic loop")
        for node in graph.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node.op != "call_function":
                raise _Unsupported(f"{node.op} {node.name}")
            target = node.target
            if target in (memory_ops.load, memory_ops.store) and not isinstance(
                node.args[0], Node
            ):
                # A (tensor-like, pointer tile) stack tensor has no positional
                # evaluator; only the native 2D stack store reads it by position.
                raise _Unsupported(f"stack tensor access at {node.name}")
            if target is memory_ops.store:
                storage = _storage(cast("Node", node.args[0]))
                if storage is None:
                    raise _Unsupported(f"store without host tensor at {node.name}")
                _admit_store_value(node, df)
                _admit_store_mask(node, df)
            elif target in CONTRACTIONS:
                _admit_contraction(node)
                shared += _tile_bytes(node)
            elif target is scan_ops._associative_scan:
                shared += sum(_tile_bytes(leaf) for leaf in _scan_leaves(node))
            elif target is torch.ops.aten.sum.dim_IntList:
                _sum_axes(node)
                shared += _tile_bytes(node)
            elif target is _tracing_ops._for_loop:
                interface = _loop_info(node).loop_interface
                assert interface is not None
                for carry in interface.carries:
                    outer = cast("Sequence[Node]", node.args[3])[carry.input_index]
                    shared += 2 * _tile_bytes(outer)  # current + next
            elif target is _tracing_ops._phi:
                _phi_source(node)
            elif target is _tracing_ops._if:
                _branch_test(node, df)
            elif target in _SCALAR_COMPARISONS and isinstance(
                node.meta.get("val"), (bool, torch.SymBool)
            ):
                pass  # host-scalar branch tests, rendered by sympy_expr
            elif target in _EXPRESSION_TARGETS or target in _EFFECT_TARGETS:
                pass
            elif cm._pointwise_inputs(node) is None:
                raise _Unsupported(f"node {node.name}: {target}")
    return graphs[0].graph_id, shared


def _admit_contraction(node: Node) -> None:
    contraction = contraction_of(node)
    if contraction is None:
        raise _Unsupported(f"contraction arguments at {node.name}")
    left, right = contraction.lhs.meta["val"], contraction.rhs.meta["val"]
    if left.dtype not in _MATERIALIZED_DTYPES or left.dtype != right.dtype:
        raise _Unsupported(f"dot operand dtypes at {node.name}")
    if left.ndim != right.ndim or left.ndim < 2:
        raise _Unsupported(f"dot rank at {node.name}")
    out = node.meta["val"]
    # aten.mm is exactly rank 2; aten.bmm/baddbmm exactly rank 3.
    rank = 2 if node.target is torch.ops.aten.mm.default else 3
    if node.target is not dot and (left.ndim != rank or out.ndim != rank):
        raise _Unsupported(f"contraction rank at {node.name}")
    acc = contraction.acc
    if acc is None:
        return
    value = acc.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.ndim > out.ndim:
        raise _Unsupported(f"contraction accumulator at {node.name}")
    if node.target is dot and value.dtype != torch.float32:
        raise _Unsupported(f"non-FP32 dot accumulator at {node.name}")
    if node.target is not dot and value.dtype != out.dtype:
        # baddbmm adds its input in the result dtype; no implicit promotion.
        raise _Unsupported(f"baddbmm input dtype at {node.name}")


def _sum_axes(node: Node) -> tuple[int, ...]:
    source = cast("Node", node.args[0])
    rank = len(cm._shape(source))
    dims = node.args[1]
    if not isinstance(dims, (tuple, list)):
        raise _Unsupported(f"dynamic sum dimensions at {node.name}")
    normalized = []
    for dim in dims:
        if not isinstance(dim, int) or not -rank <= dim < rank:
            raise _Unsupported(f"invalid sum dimension at {node.name}")
        normalized.append(dim % rank)
    axes = tuple(range(rank)) if not dims else tuple(normalized)
    if len(set(axes)) != len(axes):
        raise _Unsupported(f"duplicate sum dimensions at {node.name}")
    if source.meta["val"].dtype not in _MATERIALIZED_DTYPES:
        raise _Unsupported(f"sum input dtype at {node.name}")
    return axes


def _scan_helper(node: Node) -> HelperFunctionGraphInfo:
    from ..device_ir import HelperFunctionGraphInfo
    from ..host_function import HostFunction

    helper = HostFunction.current().device_ir.graphs[cast("int", node.args[0])]
    if not isinstance(helper, HelperFunctionGraphInfo):
        raise _Unsupported(f"scan {node.name} lost its combine graph")
    return helper


def _scan_leaves(node: Node) -> tuple[Node, ...]:
    """Validate a scan and its closed, pure combine graph; return its inputs.

    Frontend contract (``scan_ops``): the combine graph is traced over [1]
    operands; placeholders are ``(left..., right...)``, where left is the
    running state in traversal order and right the current element. Reverse
    scans traverse from the last index, as the eager reference does.
    """
    graph_id, inputs, dim, reverse, is_tuple = (*node.args, False)[:5]
    if is_tuple:
        if not isinstance(inputs, (tuple, list)):
            raise _Unsupported(f"scan tuple inputs at {node.name}")
        input_nodes = inputs
    else:
        input_nodes = (inputs,)
    leaves_list: list[Node] = []
    for leaf in input_nodes:
        if not isinstance(leaf, Node):
            raise _Unsupported(f"scan input at {node.name}")
        leaves_list.append(leaf)
    leaves = tuple(leaves_list)
    if (
        node.kwargs
        or not isinstance(dim, int)
        or not isinstance(reverse, bool)
        or not leaves
    ):
        raise _Unsupported(f"scan arguments at {node.name}")
    shapes = {cm._shape(leaf) for leaf in leaves}
    if len(shapes) != 1 or not -len(next(iter(shapes))) <= dim < len(
        next(iter(shapes))
    ):
        raise _Unsupported(f"scan shape or dim at {node.name}")
    helper = _scan_helper(node).graph
    placeholders = helper.find_nodes(op="placeholder")
    outputs = next(iter(helper.find_nodes(op="output"))).args[0]
    outputs = tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)
    if len(placeholders) != 2 * len(leaves) or len(outputs) != len(leaves):
        raise _Unsupported(f"scan combine arity at {node.name}")
    for result, leaf in zip(outputs, leaves, strict=True):
        if (
            not isinstance(result, Node)
            or result.meta["val"].dtype != leaf.meta["val"].dtype
        ):
            raise _Unsupported(f"scan combine dtype at {node.name}")
    for helper_node in helper.nodes:
        if helper_node.op in ("placeholder", "output"):
            continue
        if helper_node.op != "call_function" or not (
            helper_node.target in _SCAN_HELPER_TARGETS
            or cm._pointwise_inputs(helper_node) is not None
        ):
            # No memory, contraction, control flow or collective in a combine.
            raise _Unsupported(f"scan combine node {helper_node.name}")
        if any(source.graph is not helper for source in helper_node.all_input_nodes):
            raise _Unsupported(f"scan combine captures {helper_node.name}")
    for leaf in leaves:
        _tile_bytes(leaf)
    return leaves


def _loop_info(node: Node) -> ForLoopGraphInfo:
    from ..device_ir import ForLoopGraphInfo
    from ..host_function import HostFunction

    info = HostFunction.current().device_ir.graphs[cast("int", node.args[0])]
    if not isinstance(info, ForLoopGraphInfo) or info.loop_interface is None:
        raise _Unsupported("loop without a traced interface")
    if len(node.args) != 4:
        raise _Unsupported("explicit loop step")
    return info


def _phi_source(node: Node) -> Node:
    """``phi(before, after)`` is admitted only as an exact loop carry merge."""
    before, after = (cast("Node", arg) for arg in node.args[:2])
    loop = after.args[0] if after.target is operator.getitem else None
    if not isinstance(loop, Node) or loop.target is not _tracing_ops._for_loop:
        raise _Unsupported(f"phi {node.name} is not a loop carry")
    interface = _loop_info(loop).loop_interface
    assert interface is not None
    slot = cast("int", after.args[1])
    inputs = cast("Sequence[Node]", loop.args[3])
    if not any(
        carry.output_index == slot and inputs[carry.input_index] is before
        for carry in interface.carries
    ):
        raise _Unsupported(f"phi {node.name} does not merge its own loop input")
    return after


def _check_distinct_origins(graph: torch.fx.Graph, active: frozenset[int]) -> None:
    """Every live tile owns its canonical id: coordinates are keyed by it.

    Distinct tiles may share a canonical id (a common registered block size);
    a nested loop would then rebind the enclosing tile's origin.
    """
    from ..host_function import HostFunction

    graphs = HostFunction.current().device_ir.graphs
    env = CompileEnvironment.current()
    for node in graph.nodes:
        if node.target is _tracing_ops._for_loop:
            info = _loop_info(node)
            ids = [env.canonical_block_id(block_id) for block_id in info.block_ids]
            if len(set(ids)) != len(ids) or active.intersection(ids):
                raise _Unsupported("loop tile shares an active canonical block id")
            _check_distinct_origins(info.graph, active.union(ids))
        elif node.target is _tracing_ops._if:
            for graph_id in cast("Sequence[int | None]", node.args[1:3]):
                if graph_id is not None:
                    _check_distinct_origins(graphs[graph_id].graph, active)


def plan_positional_root(
    graphs: Sequence[GraphInfo], df: DeviceFunction
) -> PositionalRootPlan | None:
    """Return a plan only for live collisions after native routes declined."""
    from ..host_function import HostFunction
    from .tcgen05_config import CuteTcgen05Config
    from .thread_budget import check_thread_limit

    direct_mm_owners: set[Node] = set()
    stack_store_owners: set[Node] = set()
    collisions = find_axis_collisions(
        graphs, direct_mm_owners, stack_store_owners
    ) + find_lane_split_contractions(graphs, df)
    df.cute_state.direct_mm_collision_owners = frozenset(direct_mm_owners)
    df.cute_state.stack_store_collision_owners = frozenset(stack_store_owners)
    if not collisions or _fast_route_claimed(df):
        return None
    env = CompileEnvironment.current()
    try:
        if df.config.pid_type != "flat":
            raise _Unsupported("pid type")
        root_graph_id, shared = _admit(graphs, df)
        ir = HostFunction.current().device_ir
        if len(ir.grid_block_ids) != 1 or len(ir.task_families) != 1:
            raise _Unsupported("multiple grids")
        family = ir.task_families[0]
        if family.logical_axis_order != tuple(ir.grid_block_ids[0]):
            raise _Unsupported("grid axis order")
        axes = []
        for block_id in ir.grid_block_ids[0]:
            axis = family.axis(block_id)
            block = df.resolved_block_size(block_id)
            if axis is None or not axis.canonical_origin:
                raise _Unsupported("noncanonical grid origin")
            if not isinstance(block, int) or not isinstance(axis.extent, sympy.Expr):
                raise _Unsupported("dynamic grid extent")
            extent = env.specialize_expr(axis.extent)
            if not isinstance(extent, sympy.Integer) or int(extent) <= 0:
                raise _Unsupported("nonpositive or dynamic grid extent")
            axes.append((env.canonical_block_id(block_id), int(extent), block))
        if len({axis for axis, _, _ in axes}) != len(axes):
            # Coordinates are keyed by canonical id; two grid tiles sharing one
            # (e.g. a common registered block size) would share an origin.
            raise _Unsupported("grid tiles share a canonical block id")
        _check_distinct_origins(
            ir.graphs[root_graph_id].graph, frozenset(axis for axis, _, _ in axes)
        )
        if math.prod(-(-e // b) for _, e, b in axes) > 2**31 - 1:
            raise _Unsupported("grid too large")
        threads = 32 * df.config.num_warps
        check_thread_limit(threads, context="positional root")
        hosts = [
            node
            for graph in graphs
            for node in graph.graph.nodes
            if node.target is _tracing_ops._host_tensor
        ]
        stores = [
            node
            for graph in graphs
            for node in graph.graph.nodes
            if node.target is memory_ops.store
        ]
        if not hosts or not stores:
            raise _Unsupported("root without a host tensor store")
        # Mirror emission: the root and branch arms come from the codegen
        # graph copies, loop bodies from ``_loop_info``.
        by_id = {graph.graph_id: graph.graph for graph in graphs}
        domains = LogicalDomains(
            _loop_info,
            lambda node: _branch_arms(node, df),
            by_id.__getitem__,
            _scan_leaves,
        )
        domains.visit(by_id[root_graph_id])
        device = hosts[0].meta["val"].device
        capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(device)
        # A CPU-only codegen target has no launch capacity; the launch-time
        # target still repeats this admission when compiled for CUDA.
        if capacity and shared > capacity:
            raise _Unsupported(f"{shared} shared bytes exceed {capacity}")
    except (_Unsupported, cm._UnsupportedChain) as error:
        raise exc.BackendUnsupported(
            "cute", f"positional root lowering: {error} (axis collisions: {collisions})"
        ) from error
    return PositionalRootPlan(
        root_graph_id,
        tuple(axes),
        threads,
        shared,
        capacity,
        stores[0],
        collisions,
        domains,
    )


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------


class _Positional(cm._Expression):
    """Positional evaluator with loop-port aliases and Helion tile masks."""

    def __init__(
        self,
        cg: GenerateAST,
        plan: cm.ChainedMatmulPlan,
        boundaries: dict[Node, str],
        aliases: dict[Node, Node],
        origins: dict[int, str],
        extents: dict[int, str],
        domains: LogicalDomains,
    ) -> None:
        super().__init__(cg, plan, boundaries)
        self.aliases = aliases
        self.origins.update(origins)
        self.extents = extents
        self.domains = domains

    def value(self, node: Node, coordinates: tuple[str, ...]) -> str:
        if node in self.aliases:
            return self.value(self.aliases[node], coordinates)
        if node.target is _tracing_ops._mask_to:
            source = cast("Node", node.args[0])
            original = self.value(source, coordinates)
            bounds = self.domain(source, coordinates)
            lifted = self.domains.lifted_mask_dim(node)
            if lifted is not None:
                # A promoted contraction K (``LogicalDomains.promote_k``):
                # keep every other dim's mask, drop only the shorter K bound.
                skip = self.dim_bound(source, lifted, coordinates[lifted])
                bounds = [bound for bound in bounds if bound != skip]
            if not bounds:
                return original
            dtype = CompileEnvironment.current().backend.dtype_str(
                node.meta["val"].dtype
            )
            fill = self.scalar(node.args[1])
            return self.bind(
                f"({original} if {' and '.join(bounds)} else {dtype}({fill}))"
            )
        if node.target is torch.ops.aten.full.default:
            # Uniform: every coordinate, padded or not, holds the fill value.
            if len(coordinates) != node.meta["val"].ndim:
                raise _Unsupported(f"coordinate rank at {node.name}")
            dtype = CompileEnvironment.current().backend.dtype_str(
                node.meta["val"].dtype
            )
            return f"{dtype}({self.scalar(node.args[1])})"
        return super().value(node, coordinates)

    @staticmethod
    def _grid_origin(node: object) -> GridOrigin | None:
        """Tile id/begin and grid-loop symbols are scalars, not tile vectors.

        ``env.get_block_id`` deliberately maps both block-size and grid-index
        symbols to a block id; the base evaluator then treats them as a tile
        size or a vector selector. Positional roots see these symbols inside
        loop bodies (e.g. ``h[tile.id, i_t, :, tile_dv]``).
        """
        from ..host_function import HostFunction
        from ..variable_origin import GridOrigin

        if not isinstance(node, Node) or node.target not in (
            _tracing_ops._get_symnode,
            torch.ops.aten.sym_size.int,
        ):
            return None
        value = node.meta.get("val")
        if not isinstance(value, torch.SymInt):
            return None
        info = HostFunction.current().expr_to_origin.get(value._sympy_())
        origin = None if info is None else info.origin
        return origin if isinstance(origin, GridOrigin) else None

    def scalar_range(self, arg: object) -> tuple[int, int] | None:
        if isinstance(arg, Node) and arg in self.aliases:
            return self.scalar_range(self.aliases[arg])
        if self._grid_origin(arg) is not None:
            # These are runtime indices, not the constant block size used by
            # the base evaluator. Unknown bounds retain signed // and %.
            return None
        return super().scalar_range(arg)

    def scalar(self, arg: object) -> str:
        from ..variable_origin import GridOrigin
        from ..variable_origin import TileBeginOrigin
        from ..variable_origin import TileIdOrigin

        if isinstance(arg, Node) and arg in self.aliases:
            return self.scalar(self.aliases[arg])
        origin = self._grid_origin(arg)
        if origin is not None:
            block = CompileEnvironment.current().canonical_block_id(origin.block_id)
            if block not in self.origins:
                raise _Unsupported(f"inactive grid symbol {cast('Node', arg).name}")
            if isinstance(origin, TileIdOrigin):
                return f"({self.origins[block]} // {self.block_size(block)})"
            if type(origin) is GridOrigin or isinstance(origin, TileBeginOrigin):
                # hl.grid indices have block size one: the origin is the index.
                return self.origins[block]
            raise _Unsupported(f"{type(origin).__name__} scalar")
        return super().scalar(arg)

    def indices(self, node: Node, coordinates: tuple[str, ...]) -> list[str]:
        selectors = cast("Sequence[object]", node.args[1])
        if not any(self._grid_origin(index) is not None for index in selectors):
            return super().indices(node, coordinates)
        result: list[str] = []
        dim = 0
        for index in selectors:
            if self._grid_origin(index) is not None:
                result.append(self.scalar(index))
                continue
            if isinstance(index, Node) and isinstance(
                index.meta.get("val"), torch.Tensor
            ):
                if len(cm._shape(index)) > 1:
                    raise _Unsupported("broadcast index mixed with grid scalars")
                vector = bool(cm._shape(index))
            else:
                vector = index == slice(None) or (
                    isinstance(index, Node)
                    and index.target
                    in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int)
                )
            if vector and dim >= len(coordinates):
                raise _Unsupported(f"index rank at {node.name}")
            result.append(self._index(index, coordinates[dim] if vector else None))
            dim += int(vector)
        if dim != len(coordinates):
            raise _Unsupported(f"index rank at {node.name}")
        return result

    def _load(self, node: Node, coordinates: tuple[str, ...]) -> str:
        # Helion tile tails read zero even when host storage continues (for
        # example a sequence slice of a packed tensor), so host bounds alone
        # are insufficient.
        value = super()._load(node, coordinates)
        bounds = self.domain(node, coordinates)
        if not bounds:
            return value
        dtype = CompileEnvironment.current().backend.dtype_str(node.meta["val"].dtype)
        return self.bind(f"({value} if {' and '.join(bounds)} else {dtype}(0))")

    def domain(self, node: Node, coordinates: tuple[str, ...]) -> list[str]:
        """Logical bounds: active tile tails and padded full-dim positions."""
        if len(coordinates) != node.meta["val"].ndim:
            raise _Unsupported(f"domain rank of {node.name}")
        return [
            bound
            for dim, coordinate in enumerate(coordinates)
            if (bound := self.dim_bound(node, dim, coordinate)) is not None
        ]

    def destination_domain(self, node: Node, coordinates: tuple[str, ...]) -> list[str]:
        """Logical bounds of a store destination: each indexer's own domain.

        Host bounds alone admit padded destination positions: a loaded
        indexer reads 0 there, and a tile continues past its loop end while
        host storage does.  Coordinates reach indexers as in ``indices``:
        Cartesian 1D tensors and tiles take one each, explicit broadcast
        tensors share them right-aligned, a full slice is its own host
        position, and grid and other scalars name one position.
        """
        selectors = cast("Sequence[object]", node.args[1])
        tensors = [
            index
            for index in selectors
            if isinstance(index, Node)
            and isinstance(index.meta.get("val"), torch.Tensor)
            and cm._shape(index)
        ]
        advanced = any(len(cm._shape(index)) > 1 for index in tensors)
        bounds: list[str] = []
        dim = 0
        for index in selectors:
            if any(index is tensor for tensor in tensors):
                index = cast("Node", index)
                shape = cm._shape(index)
                if advanced:
                    offset = len(coordinates) - len(shape)
                    coords = tuple(
                        "0" if size == 1 else coordinates[i + offset]
                        for i, size in enumerate(shape)
                    )
                else:
                    coords, dim = (coordinates[dim],), dim + 1
                bounds += self.domain(index, coords)
            elif index == slice(None):
                dim += 1
            elif (
                isinstance(index, Node)
                and index.target
                in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int)
                and self._grid_origin(index) is None
            ):
                block = self.block_id(index)
                if block not in self.extents:
                    raise _Unsupported(f"inactive tile index at {node.name}")
                bounds.append(
                    f"{self.origins[block]} + ({coordinates[dim]}) "
                    f"< {self.extents[block]}"
                )
                dim += 1
        return bounds

    def store_domain(self, node: Node, coordinates: tuple[str, ...]) -> list[str]:
        """Bounds of a stored value or mask at destination coordinates.

        A ``BROADCAST`` dim has no length of its own: its co-operand decides.
        Here that is the destination, which ``destination_domain`` bounds.
        """
        if len(coordinates) != node.meta["val"].ndim:
            raise _Unsupported(f"domain rank of {node.name}")
        return [
            bound
            for dim, coordinate in enumerate(coordinates)
            if not self.domains.broadcast_dim(node, dim)
            and (bound := self.dim_bound(node, dim, coordinate)) is not None
        ]

    def host_extent(self, size: int | torch.SymInt) -> str:
        """A host tensor extent: static, tile-resolved, or a host scalar."""
        from ..host_function import HostFunction

        if isinstance(size, int):
            return str(size)
        try:
            return str(cm._resolved_extent(size))
        except cm._UnsupportedChain:
            pass
        env = CompileEnvironment.current()
        expr = env.specialize_expr(cast("sympy.Expr", size._sympy_()))
        origins = HostFunction.current().expr_to_origin
        for symbol in expr.free_symbols:
            info = origins.get(symbol)
            if info is None or not info.origin.is_host():
                raise _Unsupported(f"non-host tensor extent symbol {symbol}")
        return self.cg.device_function.sympy_expr(expr)

    def dim_bound(self, node: Node, dim: int, coordinate: str) -> str | None:
        """A full (inactive) dim spans its whole static size, but its tile may
        be padded (C=40 evaluates 64 positions). Pointwise values there are
        not zero, so padded positions are masked like active tile tails."""
        env = CompileEnvironment.current()
        size = node.meta["val"].shape[dim]
        if isinstance(size, int):
            # A literal may already be the padded size (hl.zeros([t, 40, 40])
            # has meta [t, 64, 64]); its logical length comes from provenance.
            if size == 1:
                return None
            try:
                logical = self.domains.static(node, dim)
            except UnprovenDomain as error:
                raise _Unsupported(str(error)) from error
        else:
            block = env.resolve_block_id(size)
            if block is None:
                raise _Unsupported(f"unresolved dim of {node.name}")
            block = env.canonical_block_id(block)
            if block in self.extents:
                return f"{self.origins[block]} + ({coordinate}) < {self.extents[block]}"
            logical = env.block_sizes[block].size
            if not isinstance(logical, int):
                raise _Unsupported(f"inactive tile dim of {node.name}")
        extent = cm._shape(node)[dim]
        if extent < logical:
            raise _Unsupported(f"partial full dim of {node.name}")
        return f"({coordinate}) < {logical}" if extent > logical else None


class PositionalEmitter:
    def __init__(self, cg: GenerateAST, plan: PositionalRootPlan) -> None:
        self.cg = cg
        self.plan = plan
        env = CompileEnvironment.current()
        self.shim = cm.ChainedMatmulPlan(
            plan.root_graph_id,
            (),
            plan.store,
            plan.axes,
            (),
            torch.float32,
            plan.threads,
        )
        self.boundaries: dict[Node, str] = {}
        self.aliases: dict[Node, Node] = {}
        self.origins: dict[int, str] = {}
        self.extents: dict[int, str] = {a: str(e) for a, e, _ in plan.axes}
        self.loop_outputs: dict[Node, dict[int, str]] = {}
        self.scan_outputs: dict[Node, tuple[str, ...]] = {}
        self.allocations: list[str] = []
        self.allocated_bytes = 0
        self.snapshot_bytes = 0
        self.host_aliases = host_effects.HostAliasOracle(env)
        self.counter = 0
        self.index_dtype = env.backend.dtype_str(env.index_dtype)

    def fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"ptp_{prefix}_{self.counter}"

    def expression(self) -> _Positional:
        return _Positional(
            self.cg,
            self.shim,
            self.boundaries,
            self.aliases,
            self.origins,
            self.extents,
            self.plan.domains,
        )

    def allocate(
        self, node: Node, dtypes: tuple[torch.dtype, ...] = _MATERIALIZED_DTYPES
    ) -> str:
        shape = cm._shape(node)
        dtype = CompileEnvironment.current().backend.dtype_str(node.meta["val"].dtype)
        self.allocated_bytes += _tile_bytes(node, dtypes)
        name = self.fresh("tile")
        layout = (
            f"cute.make_layout({tuple(shape)!r})" if shape else "cute.make_layout(1)"
        )
        self.allocations.append(
            f"{name} = cute.make_tensor(cute.arch.alloc_smem({dtype}, "
            f"{max(1, math.prod(shape))}, alignment=128), {layout})"
        )
        return name

    def element_loop(self, shape: Sequence[int]) -> tuple[list[str], tuple[str, ...]]:
        e = self.fresh("e")
        header = [
            (
                f"for {e} in cutlass.range(ptp_thread, {max(1, math.prod(shape))}, "
                f"{self.plan.threads}, unroll=1):"
            )
        ]
        coords = []
        for d, size in enumerate(shape):
            c = self.fresh(f"c{d}")
            header.append(
                f"    {c} = {self.index_dtype}(({e} // {math.prod(shape[d + 1 :])}) % {size})"
            )
            coords.append(c)
        return header, tuple(coords)

    @staticmethod
    def indent(lines: Sequence[str]) -> list[str]:
        return ["    " + line for line in lines]

    def phase(self, header: list[str], body: list[str]) -> list[str]:
        return [*header, *self.indent(body), "cute.arch.sync_threads()"]

    def write_tile(self, node: Node, buffer: str) -> list[str]:
        header, coords = self.element_loop(cm._shape(node))
        expr = self.expression()
        value = expr.value(node, coords)
        dtype = CompileEnvironment.current().backend.dtype_str(node.meta["val"].dtype)
        index = ", ".join(coords) if coords else "0"
        return self.phase(
            header, [*expr.lines, f"{buffer}[{index}] = {dtype}({value})"]
        )

    def copy_tile(self, node: Node, source: str, destination: str) -> list[str]:
        header, coords = self.element_loop(cm._shape(node))
        index = ", ".join(coords) if coords else "0"
        return self.phase(header, [f"{destination}[{index}] = {source}[{index}]"])

    def emit_contraction(self, node: Node) -> list[str]:
        contraction = contraction_of(node)
        assert contraction is not None  # admitted
        lhs, rhs = contraction.lhs, contraction.rhs
        acc = contraction.acc
        out_shape = cm._shape(node)
        lhs_shape, rhs_shape = cm._shape(lhs), cm._shape(rhs)
        k_extent = lhs_shape[-1]
        if rhs_shape[-2] != k_extent:
            raise _Unsupported(f"contraction extent mismatch at {node.name}")
        buffer = self.allocate(node)
        header, coords = self.element_loop(out_shape)
        batch, (row, col) = coords[:-2], coords[-2:]

        def broadcast(shape: tuple[int, ...], last: tuple[str, ...]) -> tuple[str, ...]:
            lead = shape[: len(shape) - len(last)]
            offset = len(batch) - len(lead)
            return (
                *("0" if s == 1 else batch[i + offset] for i, s in enumerate(lead)),
                *last,
            )

        outer = self.expression()
        initial = "cutlass.Float32(0)"
        if acc is not None:
            acc_shape = cm._shape(acc)
            # The accumulator broadcasts like an elementwise operand.
            acc_coords = broadcast(acc_shape, (row, col)[2 - min(2, len(acc_shape)) :])
            acc_coords = tuple(
                "0" if size == 1 else coordinate
                for size, coordinate in zip(acc_shape, acc_coords, strict=True)
            )
            initial = f"cutlass.Float32({outer.value(acc, acc_coords)})"
        total, k = self.fresh("acc"), self.fresh("k")
        inner = self.expression()
        a = inner.value(lhs, broadcast(lhs_shape, (row, k)))
        b = inner.value(rhs, broadcast(rhs_shape, (k, col)))
        # Transformed operands are nonzero at padded/tail K positions, unless
        # an operand spanning the padded K defines every lane (promote_k).
        valid = (
            {}
            if self.plan.domains.k_promoted(node)
            else {
                bound: None
                for bound in (
                    inner.dim_bound(lhs, len(lhs_shape) - 1, k),
                    inner.dim_bound(rhs, len(rhs_shape) - 2, k),
                )
                if bound is not None
            }
        )
        term = f"cutlass.Float32({a}) * cutlass.Float32({b})"
        if valid:
            term = f"({term} if {' and '.join(valid)} else cutlass.Float32(0))"
        dtype = CompileEnvironment.current().backend.dtype_str(node.meta["val"].dtype)
        body = [
            *outer.lines,
            f"{total} = {initial}",
            f"for {k} in cutlass.range({k_extent}, unroll=1):",
            *self.indent([*inner.lines, f"{total} = {total} + {term}"]),
            f"{buffer}[{', '.join(coords)}] = {dtype}({total})",
        ]
        self.boundaries[node] = buffer
        return self.phase(header, body)

    def combine(
        self, node: Node, left: Sequence[str], right: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        """Evaluate the closed combine graph on scalar operands, in order."""
        helper = _scan_helper(node).graph
        placeholders = helper.find_nodes(op="placeholder")
        outputs = next(iter(helper.find_nodes(op="output"))).args[0]
        outputs = tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)
        expr = self.expression()
        for placeholder, value in zip(placeholders, (*left, *right), strict=True):
            expr.fragments[placeholder] = (("0",), value)
        values = [expr.value(cast("Node", result), ("0",)) for result in outputs]
        return expr.lines, values

    def emit_scan(self, node: Node) -> list[str]:
        """One owner per scan line; a sequential inclusive scan in traversal order.

        Positions outside the logical tile domain neither seed nor update the
        state, so a reverse scan over a partial tile never consumes tail data.
        """
        leaves = _scan_leaves(node)
        dim, reverse = cast("int", node.args[2]), cast("bool", node.args[3])
        shape = cm._shape(leaves[0])
        dim %= len(shape)
        buffers = tuple(self.allocate(leaf) for leaf in leaves)
        outer = tuple(size for axis, size in enumerate(shape) if axis != dim)
        header, outer_coords = self.element_loop(outer)
        backend = CompileEnvironment.current().backend
        dtypes = [backend.dtype_str(leaf.meta["val"].dtype) for leaf in leaves]
        step, position = self.fresh("scan_step"), self.fresh("scan_pos")
        seen = self.fresh("scan_seen")
        state = [self.fresh("scan_state") for _ in leaves]
        coords = list(outer_coords)
        coords.insert(dim, position)
        coordinates = tuple(coords)
        expr = self.expression()
        current = [expr.value(leaf, coordinates) for leaf in leaves]
        valid = expr.domain(leaves[0], coordinates)
        names = [self.fresh("scan_value") for _ in leaves]
        combine_lines, combined = self.combine(node, state, names)
        predicate = " and ".join(valid) if valid else "True"
        index = ", ".join(coordinates)
        loop = [
            f"{position} = " + (f"{shape[dim] - 1} - {step}" if reverse else step),
            *expr.lines,
            *(
                f"{name} = {dtype}({value})"
                for name, dtype, value in zip(names, dtypes, current, strict=True)
            ),
            *combine_lines,
            # Selects, not branches: the state type is loop-invariant.
            *(
                f"{name}_next = {dtype}({value})"
                for name, dtype, value in zip(state, dtypes, combined, strict=True)
            ),
            *(
                f"{name} = ({value} if {seen} == 0 else {name}_next) if {predicate} else {name}"
                for name, value in zip(state, names, strict=True)
            ),
            f"{seen} = cutlass.Int32(1) if {predicate} else {seen}",
            *(
                f"{buffer}[{index}] = {name}"
                for buffer, name in zip(buffers, state, strict=True)
            ),
        ]
        body = [
            f"{seen} = cutlass.Int32(0)",
            *(
                f"{name} = {dtype}(0)"
                for name, dtype in zip(state, dtypes, strict=True)
            ),
            f"for {step} in cutlass.range({shape[dim]}, unroll=1):",
            *self.indent(loop),
        ]
        if node.args[4]:
            self.scan_outputs[node] = buffers
        else:
            self.boundaries[node] = buffers[0]
        return self.phase(header, body)

    def emit_sum(self, node: Node) -> list[str]:
        source = cast("Node", node.args[0])
        shape = cm._shape(source)
        axes = _sum_axes(node)
        keepdim = bool(node.args[2]) if len(node.args) > 2 else False
        buffer = self.allocate(node)
        header, coordinates = self.element_loop(cm._shape(node))
        total, reduction = self.fresh("sum"), self.fresh("reduce")
        reduction_shape = tuple(shape[axis] for axis in axes)
        output_position = 0
        input_coordinates = []
        for axis, size in enumerate(shape):
            if axis in axes:
                position = axes.index(axis)
                stride = math.prod(reduction_shape[position + 1 :])
                input_coordinates.append(f"(({reduction} // {stride}) % {size})")
                output_position += int(keepdim)
            else:
                input_coordinates.append(coordinates[output_position])
                output_position += 1
        expr = self.expression()
        value = expr.value(source, tuple(input_coordinates))
        dtype = CompileEnvironment.current().backend.dtype_str(node.meta["val"].dtype)
        index = ", ".join(coordinates) if coordinates else "0"
        body = [
            f"{total} = cutlass.Float32(0)",
            f"for {reduction} in cutlass.range({math.prod(reduction_shape)}, unroll=1):",
            *self.indent(
                [
                    *expr.lines,
                    (
                        f"{total} = {total} + (cutlass.Float32({value}) if "
                        f"{' and '.join(valid)} else cutlass.Float32(0))"
                    )
                    if (valid := expr.domain(source, tuple(input_coordinates)))
                    else f"{total} = {total} + cutlass.Float32({value})",
                ]
            ),
            f"{buffer}[{index}] = {dtype}({total})",
        ]
        self.boundaries[node] = buffer
        return self.phase(header, body)

    def emit_store(self, node: Node) -> list[str]:
        from ...language.memory_ops import _cute_scalar_pointer_expr

        tensor_node, value = cast("Node", node.args[0]), cast("Node", node.args[2])
        # Tensor indexers can have an exact extent (e.g. arange(48)) while
        # the value is padded to 64. Visit the destination domain so padded
        # value lanes cannot write beyond the requested index range.
        header, coords = self.element_loop(_store_shape(node, self.cg.device_function))
        expr = self.expression()
        indices = expr.indices(node, coords)
        tensor = tensor_node.meta["val"]
        name = expr.tensor_name(tensor_node)
        stored = expr.value(value, coords)
        dtype = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
        bounds = [
            f"0 <= ({i}) < {expr.host_extent(s)}"
            for i, s in zip(indices, tensor.shape, strict=True)
        ]
        bounds += expr.destination_domain(node, coords)
        bounds += expr.store_domain(value, coords)
        mask = _store_mask(node)
        if isinstance(mask, Node):
            # Right-aligned broadcast, like a masked load. A position outside
            # the mask's logical domain is not a requested write.
            mask_shape = cm._shape(mask)
            offset = len(coords) - len(mask_shape)
            mask_coords = tuple(
                "0" if size == 1 else coords[i + offset]
                for i, size in enumerate(mask_shape)
            )
            bounds += expr.store_domain(mask, mask_coords)
            bounds.append(expr.value(mask, mask_coords))
        elif mask is False:
            bounds.append("False")
        return self.phase(
            header,
            [
                *expr.lines,
                f"if {' and '.join(dict.fromkeys(bounds))}:",
                f"    {_cute_scalar_pointer_expr(name, indices)}.store({dtype}({stored}))",
            ],
        )

    def emit_loop(self, node: Node) -> list[str]:
        info = _loop_info(node)
        begins, ends, args = node.args[1:4]
        env = CompileEnvironment.current()
        begins = cast("Sequence[object]", begins)
        ends = cast("Sequence[object]", ends)
        if len(begins) != len(info.block_ids) or len(ends) != len(info.block_ids):
            raise _Unsupported("loop bounds do not match loop dims")
        scalars = self.expression()
        dims: list[tuple[int, str, str, int]] = []  # block, begin, bound, step
        lines: list[str] = []
        for block_id, begin_arg, end_arg in zip(
            info.block_ids, begins, ends, strict=True
        ):
            block_id = env.canonical_block_id(block_id)
            block = self.cg.device_function.resolved_block_size(block_id)
            if not isinstance(block, int):
                raise _Unsupported("dynamic loop block")
            begin, end = scalars.scalar(begin_arg), scalars.scalar(end_arg)
            dims.append((block_id, begin, end, block))
        lines += scalars.lines
        bounds = []
        for block_id, begin, end, block in dims:
            bound = self.fresh("end")
            lines.append(f"{bound} = {end}")
            bounds.append((block_id, begin, bound, block))
        args = cast("Sequence[Node]", args)
        interface = info.loop_interface
        assert interface is not None
        placeholders = [n for n in info.graph.nodes if n.op == "placeholder"]
        current: dict[int, str] = {}
        for carry in interface.carries:
            buffer = self.allocate(args[carry.input_index])
            current[carry.output_index] = buffer
            lines += self.write_tile(args[carry.input_index], buffer)
        for index, placeholder in enumerate(placeholders):
            slots = [c for c in interface.carries if c.input_index == index]
            if slots:
                self.boundaries[placeholder] = current[slots[0].output_index]
            else:
                self.aliases[placeholder] = args[index]
        saved = (dict(self.origins), dict(self.extents))
        headers = []
        for block_id, begin, bound, block in bounds:
            offset = self.fresh("loop")
            self.origins[block_id], self.extents[block_id] = offset, bound
            # Loop order follows the traced block order (first is outermost),
            # matching ordinary nested device-loop emission.
            headers.append(
                f"for {offset} in cutlass.range({begin}, {bound}, {block}, unroll=1):"
            )
        body = self.emit_graph(info.graph, nested=True)
        output = next(n for n in info.graph.nodes if n.op == "output")
        results = cast("Sequence[Node]", output.args[0])
        following: dict[int, str] = {}
        for carry in interface.carries:
            buffer = self.allocate(results[carry.output_index])
            following[carry.output_index] = buffer
            body += self.write_tile(results[carry.output_index], buffer)
        # Every next value is complete before any current tile is replaced.
        for carry in interface.carries:
            body += self.copy_tile(
                results[carry.output_index],
                following[carry.output_index],
                current[carry.output_index],
            )
        self.origins, self.extents = saved
        for depth, header in enumerate(headers):
            lines.append("    " * depth + header)
        lines += ["    " * len(headers) + line for line in body]
        self.loop_outputs[node] = current
        return lines

    def nested_graphs(
        self, node: Node
    ) -> list[tuple[torch.fx.Graph, Sequence[object]]]:
        """Regions ``node`` executes: a loop body or the live branch arms."""
        if node.target is _tracing_ops._for_loop:
            return [(_loop_info(node).graph, cast("Sequence[object]", node.args[3]))]
        if node.target is _tracing_ops._if:
            return [
                (
                    self.cg.get_graph(cast("int", graph_id)).graph,
                    cast("Sequence[object]", ports),
                )
                for graph_id, ports in _branch_arms(node, self.cg.device_function)
            ]
        return []

    def emit_snapshots(
        self, nodes: list[Node], index: int, *, nested: bool
    ) -> list[str]:
        """Materialize earlier host loads the effect ``nodes[index]`` may change."""
        try:
            loads = host_effects.loads_to_preserve(
                nodes,
                index,
                materialized=self.boundaries,
                aliases=self.aliases,
                nested_graphs=self.nested_graphs,
                phi_source=_phi_source,
                oracle=self.host_aliases,
            )
        except host_effects.HostEffectProvenanceError as error:
            raise _Unsupported(str(error)) from error
        local = set(nodes)
        lines: list[str] = []
        for load in loads:
            if nested and load not in local:
                # The region's entry effect already covers every enclosing load
                # reachable through its ports.  A snapshot here could leave a
                # tile unwritten (zero-trip loop, untaken arm) for later reads.
                raise _Unsupported(
                    f"enclosing load {load.name} snapshotted inside a nested region"
                )
            dtype = load.meta["val"].dtype
            if dtype not in _SNAPSHOT_DTYPES:
                raise _Unsupported(
                    f"host-effect snapshot of {dtype} load {load.name} "
                    "requires a typed restore"
                )
            self.snapshot_bytes += _tile_bytes(load, _SNAPSHOT_DTYPES)
            buffer = self.allocate(load, _SNAPSHOT_DTYPES)
            lines += self.write_tile(load, buffer)  # evaluated before rebinding
            self.boundaries[load] = buffer
        return lines

    def emit_graph(self, graph: torch.fx.Graph, *, nested: bool = False) -> list[str]:
        lines: list[str] = []
        nodes = list(graph.nodes)
        for index, node in enumerate(nodes):
            if node.op != "call_function":
                continue
            if node.target in (
                memory_ops.store,
                _tracing_ops._for_loop,
                _tracing_ops._if,
            ):
                lines += self.emit_snapshots(nodes, index, nested=nested)
            if node.target in CONTRACTIONS:
                lines += self.emit_contraction(node)
            elif node.target is scan_ops._associative_scan:
                lines += self.emit_scan(node)
            elif node.target is operator.getitem and node.args[0] in self.scan_outputs:
                outputs = self.scan_outputs[cast("Node", node.args[0])]
                self.boundaries[node] = outputs[cast("int", node.args[1])]
            elif node.target is torch.ops.aten.sum.dim_IntList:
                lines += self.emit_sum(node)
            elif node.target is memory_ops.store:
                lines += self.emit_store(node)
            elif node.target is _tracing_ops._for_loop:
                lines += self.emit_loop(node)
            elif node.target is operator.getitem and node.args[0] in self.loop_outputs:
                outputs = self.loop_outputs[cast("Node", node.args[0])]
                self.boundaries[node] = outputs[cast("int", node.args[1])]
            elif node.target is _tracing_ops._phi:
                # The carry tile starts as the loop input: zero trips are exact.
                self.aliases[node] = _phi_source(node)
            elif node.target is _tracing_ops._if:
                lines += self.emit_branch(node)
        return lines

    def emit_branch(self, node: Node) -> list[str]:
        """Inline an output-free branch; runtime tests are CTA-uniform.

        Each arm is emitted lexically: values it materializes (snapshots,
        dots, sums, loop tiles) and its port aliases are discarded afterwards,
        so neither the other arm nor code after the branch reads a tile that
        an untaken arm would have written.  Shared tiles stay statically
        allocated outside the branch and are counted once per emitted arm.
        """
        decision = _branch_test(node, self.cg.device_function)

        def body(graph_id: object, args: object) -> list[str]:
            if graph_id is None:
                return []
            graph = self.cg.get_graph(cast("int", graph_id)).graph
            ports = cast("Sequence[Node]", args)
            placeholders = [n for n in graph.nodes if n.op == "placeholder"]
            if len(placeholders) != len(ports):
                raise _Unsupported(f"branch ports changed at {node.name}")
            saved = (dict(self.boundaries), dict(self.aliases), dict(self.loop_outputs))
            try:
                for placeholder, arg in zip(placeholders, ports, strict=True):
                    self.aliases[placeholder] = arg
                return self.emit_graph(graph, nested=True)
            finally:
                self.boundaries, self.aliases, self.loop_outputs = saved

        if isinstance(decision, bool):
            return (
                body(node.args[1], node.args[3])
                if decision
                else body(node.args[2], node.args[4])
            )
        taken = body(node.args[1], node.args[3]) or ["pass"]
        other = body(node.args[2], node.args[4]) or ["pass"]
        return [f"if {decision}:", *self.indent(taken), "else:", *self.indent(other)]

    def emit(self) -> list[str]:
        from ..program_id import FlatProgramIDs

        df = self.cg.device_function
        pid = df.pid
        if (
            type(pid) is not FlatProgramIDs
            or pid.shared_pid_var is not None
            or [
                CompileEnvironment.current().canonical_block_id(p.block_id)
                for p in pid.pid_info
            ]
            != [axis for axis, _, _ in self.plan.axes]
        ):
            raise _Unsupported("host grid does not match the planned flat axes")
        lines = [
            "ptp_thread = cutlass.Int32(cute.arch.thread_idx()[0])",
            "ptp_pid = cutlass.Int32(cute.arch.block_idx()[0])",
        ]
        stride = 1
        for axis_id, size, block in self.plan.axes:  # FlatProgramIDs: first fastest
            count = -(-size // block)
            lines.append(
                f"chain_origin_{axis_id} = {self.index_dtype}(ptp_pid // {stride} % {count}) * {block}"
            )
            stride *= count
        body = self.emit_graph(self.cg.get_graph(self.plan.root_graph_id).graph)
        if self.allocated_bytes != self.plan.shared_bytes + self.snapshot_bytes:
            raise _Unsupported("emitted shared tiles differ from admission")
        capacity = self.plan.shared_capacity
        if capacity and self.allocated_bytes > capacity:
            raise _Unsupported(
                f"{self.allocated_bytes} shared bytes exceed {capacity} "
                f"({self.snapshot_bytes} for host-effect snapshots)"
            )
        return [*lines, *self.allocations, *body]


def codegen_positional_root(cg: GenerateAST, plan: PositionalRootPlan) -> None:
    """Install the complete body only after every node has been emitted."""
    try:
        emitter = PositionalEmitter(cg, plan)
        lines = emitter.emit()
    except (_Unsupported, cm._UnsupportedChain) as error:
        raise exc.BackendUnsupported(
            "cute", f"positional root lowering: {error}"
        ) from error
    aliases = [
        f"{alias} = {name}" for name, alias in emitter.shim.tensor_aliases.items()
    ]
    df = cg.device_function
    df.preamble = []
    df.body = list(ast.parse("\n".join([*aliases, *lines])).body)
