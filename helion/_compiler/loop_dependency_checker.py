from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .device_ir import GraphInfo


# fx node meta key marking a load that must be preceded by ``tl.debug_barrier()``.
INTRA_LOOP_RAW_BARRIER_META = "_needs_debug_barrier_before"


# A root shared by every kernel argument: the caller may pass one tensor twice
# or overlapping views, so two arguments are never provably distinct storage.
HOST_ARGUMENT_ROOT = "<argument>"
# With ``distinct_arguments`` each argument gets ``<argument NAME>`` instead.
_ARGUMENT_ROOT_PREFIX = "<argument"
# A value whose storage cannot be traced (e.g. computed from globals only).
HOST_UNKNOWN_ROOT = "<unknown>"

# Calls that return a new tensor sharing storage with nothing else.
_FRESH_TORCH_FUNCTIONS = frozenset(
    {
        "arange",
        "clone",
        "empty",
        "empty_like",
        "empty_strided",
        "eye",
        "full",
        "full_like",
        "linspace",
        "ones",
        "ones_like",
        "rand",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "randn_like",
        "randperm",
        "tensor",
        "zeros",
        "zeros_like",
    }
)
_FRESH_TENSOR_METHODS = frozenset(
    {"clone", "new_empty", "new_full", "new_ones", "new_tensor", "new_zeros"}
)


def is_host_argument_root(root: str) -> bool:
    """Whether ``root`` (of ``collect_host_tensor_roots``) is a kernel argument."""
    return root.startswith(_ARGUMENT_ROOT_PREFIX)


def collect_host_tensor_roots(
    body: list[ast.stmt], arg_names: set[str], *, distinct_arguments: bool = False
) -> dict[str, frozenset[str]]:
    """Conservative storage roots of every name the host code binds.

    A root is :data:`HOST_ARGUMENT_ROOT` (``<argument NAME>`` per argument when
    ``distinct_arguments``, for callers that may assume the caller passes
    distinct tensors), an allocation site (a call listed in
    ``_FRESH_TORCH_FUNCTIONS`` / ``_FRESH_TENSOR_METHODS``), or
    :data:`HOST_UNKNOWN_ROOT`.  Any other value is assumed to alias every
    tensor it is computed from: a view, a no-op conversion or an unknown op
    may return its input.  Every binding of a name anywhere in the host code
    (nested ``if`` / ``with`` / ``for`` blocks too) contributes, so the roots
    hold whichever binding runs.  Two host tensors are provably distinct only
    if neither has the unknown root and their roots are disjoint.
    """
    bindings: list[tuple[str, ast.expr]] = []

    def bind(target: ast.expr, value: ast.expr) -> None:
        # ``x[i] = v`` copies into x's storage and does not rebind x, but an
        # attribute store may (``x.data = v``), so it counts as a binding.
        if isinstance(target, ast.Name):
            bindings.append((target.id, value))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                bind(element, value)
        elif isinstance(target, (ast.Starred, ast.Attribute)):
            bind(target.value, value)

    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            if node.value is not None:
                bind(node.target, node.value)
        elif isinstance(node, (ast.For, ast.comprehension)):
            bind(node.target, node.iter)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bind(node.optional_vars, node.context_expr)

    roots: dict[str, frozenset[str]] = {
        name: frozenset(
            {f"<argument {name}>" if distinct_arguments else HOST_ARGUMENT_ROOT}
        )
        for name in arg_names
    }
    for name, _value in bindings:
        roots.setdefault(name, frozenset())

    def name_roots(name: ast.Name) -> frozenset[str]:
        # A shape, a size or another scalar cannot share tensor storage, so
        # ``x.view(rows, cols)`` aliases x and not the tensor ``rows`` came from.
        if _holds_no_storage(name):
            return frozenset()
        return roots.get(name.id, frozenset())

    def expr_roots(expr: ast.expr) -> frozenset[str]:
        if isinstance(expr, ast.Call) and _is_fresh_allocation(expr):
            return frozenset({f"<allocation {expr.lineno}:{expr.col_offset}>"})
        result: frozenset[str] = frozenset()
        for child in ast.iter_child_nodes(expr):
            if isinstance(child, ast.Name):
                result |= name_roots(child)
            elif isinstance(child, ast.expr):
                result |= expr_roots(child)
        return result

    changed = True
    while changed:
        changed = False
        for name, value in bindings:
            value_roots = (
                name_roots(value) if isinstance(value, ast.Name) else expr_roots(value)
            )
            merged = roots[name] | value_roots
            if merged != roots[name]:
                roots[name] = merged
                changed = True
    return {
        name: name_roots or frozenset({HOST_UNKNOWN_ROOT})
        for name, name_roots in roots.items()
    }


def _holds_no_storage(name: ast.Name) -> bool:
    """Whether type propagation proved ``name`` holds no tensor storage."""
    from .ast_extension import ExtendedAST
    from .type_info import CollectionType
    from .type_info import LiteralType
    from .type_info import NumericType
    from .type_info import StringType
    from .type_info import TileIndexType

    def scalar(type_info: object) -> bool:
        if isinstance(type_info, (NumericType, LiteralType, StringType, TileIndexType)):
            return True
        if isinstance(type_info, CollectionType):
            elements = type_info.element_types
            if isinstance(elements, dict):
                return all(scalar(element) for element in elements.values())
            if isinstance(elements, (list, tuple)):
                return all(scalar(element) for element in elements)
        return False

    return isinstance(name, ExtendedAST) and scalar(name._type_info)


def _is_fresh_allocation(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if isinstance(func.value, ast.Name) and func.value.id == "torch":
        return func.attr in _FRESH_TORCH_FUNCTIONS
    return func.attr in _FRESH_TENSOR_METHODS


def mark_intra_loop_raw_barriers(
    graphs: list[GraphInfo],
    root_graph_ids: list[int],
    *,
    mark_in_divergent_control_flow: bool = True,
) -> None:
    """Mark loads that read storage written earlier in a device-loop body.

    Within a single device-loop body, ``qkv[a] = v`` followed by ``qkv[b]`` is a
    read-after-write on the same storage. When the store and the load use
    different-shaped index tensors their Triton thread->element layouts differ, so
    an element written by one thread is read back by another with no
    synchronization in between -- a data race (observed corrupting ~0.8% of
    outputs on B200). Helion already inserts ``tl.debug_barrier()`` for the
    analogous hazard *between* sequential top-level loops
    (``needs_inter_loop_debug_barrier_for_global_raw``); this extends the same
    guarantee to a store->load *within* one loop body.

    We mark the load's FX node; the Triton ``load`` codegen emits a
    ``tl.debug_barrier()`` before it (the CuTe codegen a
    ``cute.arch.sync_threads()``). The barrier flushes every prior write in the
    block, so once emitted the pending-write set is cleared and later loads need a
    new store to re-arm. Storage identity comes from the fake tensor's underlying
    storage, so distinct FX nodes for aliases and views compare equal. The walk
    follows Helion control-flow subgraphs and merges pending writes at joins.

    ``mark_in_divergent_control_flow=False`` skips marking loads inside ``hl.if``
    / while bodies: a CuTe SIMT branch condition can vary per thread, and a
    convergent CTA barrier inside a divergent branch deadlocks.  Loop bodies stay
    markable — Helion device-loop trip structure is uniform across the CTA.
    """
    marker = _IntraLoopRawBarrierMarker(
        graphs, mark_in_divergent_control_flow=mark_in_divergent_control_flow
    )
    for graph_id in root_graph_ids:
        marker.run_graph(graph_id, set())


class _IntraLoopRawBarrierMarker:
    def __init__(
        self,
        graphs: list[GraphInfo],
        *,
        mark_in_divergent_control_flow: bool = True,
    ) -> None:
        self.graphs = graphs
        self.mark_in_divergent_control_flow = mark_in_divergent_control_flow

    def _storage_ids(self, graph_id: int, obj: object) -> set[int]:
        import torch

        from .device_ir import NodeArgsGraphInfo

        if isinstance(obj, (list, tuple)):
            result: set[int] = set()
            for item in obj:
                result.update(self._storage_ids(graph_id, item))
            return result
        if not isinstance(obj, torch.fx.Node):
            return set()
        graph_info = self.graphs[graph_id]
        value = obj.meta.get("val")
        result = (
            {id(value.untyped_storage())} if isinstance(value, torch.Tensor) else set()
        )
        if (
            obj.op == "placeholder"
            and obj.graph is graph_info.graph
            and isinstance(graph_info, NodeArgsGraphInfo)
        ):
            result.update(
                self._storage_ids(graph_id, graph_info.placeholder_to_outer_arg(obj))
            )
        if result:
            return result
        for arg in obj.args:
            result.update(self._storage_ids(graph_id, arg))
        return result

    def run_graph(
        self, graph_id: int, written: set[int], *, divergent: bool = False
    ) -> set[int]:
        from ..language import memory_ops
        from ..language._tracing_ops import _for_loop
        from ..language._tracing_ops import _for_loop_step
        from ..language._tracing_ops import _if
        from ..language._tracing_ops import _while_loop
        from .device_ir import ForLoopGraphInfo
        from .device_ir import IfGraphInfo
        from .device_ir import WhileLoopGraphInfo

        pending = set(written)
        for node in self.graphs[graph_id].graph.nodes:
            if node.op != "call_function":
                continue
            if node.target is memory_ops.store:
                pending.update(self._storage_ids(graph_id, node.args[0]))
                continue
            if node.target is memory_ops.load:
                if node.meta.get(INTRA_LOOP_RAW_BARRIER_META):
                    pending.clear()
                elif pending & self._storage_ids(graph_id, node.args[0]):
                    if divergent and not self.mark_in_divergent_control_flow:
                        # Cannot place a convergent barrier here; leave the
                        # hazard pending so a later uniform load re-arms it.
                        continue
                    node.meta[INTRA_LOOP_RAW_BARRIER_META] = True
                    pending.clear()
                continue
            if node.target is _if:
                if_graph_id = node.args[1]
                assert isinstance(if_graph_id, int)
                if_info = self.graphs[if_graph_id]
                assert isinstance(if_info, IfGraphInfo)
                if_pending = self.run_graph(if_graph_id, pending, divergent=True)
                if if_info.else_branch is None:
                    pending |= if_pending
                else:
                    else_graph_id = (
                        if_info.else_branch
                        if isinstance(if_info.else_branch, int)
                        else if_info.else_branch.graph_id
                    )
                    else_pending = self.run_graph(
                        else_graph_id, pending, divergent=True
                    )
                    pending = if_pending | else_pending
                continue
            if node.target in (_for_loop, _for_loop_step):
                loop_graph_id = node.args[0]
                assert isinstance(loop_graph_id, int)
                loop_info = self.graphs[loop_graph_id]
                assert isinstance(loop_info, ForLoopGraphInfo)
                loop_input = set() if loop_info.needs_barrier_before else pending
                loop_pending = self.run_graph(
                    loop_graph_id, loop_input, divergent=divergent
                )
                # Without a pre-loop barrier, zero iterations preserve the input state.
                pending = (
                    loop_pending
                    if loop_info.needs_barrier_before
                    else pending | loop_pending
                )
                continue
            if node.target is _while_loop:
                body_graph_id = node.args[1]
                assert isinstance(body_graph_id, int)
                body_info = self.graphs[body_graph_id]
                assert isinstance(body_info, WhileLoopGraphInfo)
                condition_pending = self.run_graph(
                    body_info.cond_graph_id, pending, divergent=True
                )
                body_pending = self.run_graph(
                    body_graph_id, condition_pending, divergent=True
                )
                # The condition executes at least once; the body may not execute.
                pending = condition_pending | body_pending
        return pending


def needs_inter_loop_debug_barrier_for_global_raw(
    prev_global_writes: set[str],
    host_loop_reads: frozenset[str],
    *,
    global_barrier_tensor_names: Callable[[frozenset[str]], set[str]],
) -> bool:
    """Whether to emit ``tl.debug_barrier()`` before the next sequential device loop.

    Returns True when the union of host-named global writes accumulated from
    all prior siblings (since the last emitted barrier) intersects the current
    loop's host-named read set.
    """
    cur_global_reads = global_barrier_tensor_names(host_loop_reads)
    return bool(prev_global_writes & cur_global_reads)
