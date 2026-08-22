from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .device_ir import GraphInfo


# fx node meta key marking a load that must be preceded by ``tl.debug_barrier()``.
INTRA_LOOP_RAW_BARRIER_META = "_needs_debug_barrier_before"


def collect_host_tensor_aliases(body: list[ast.stmt]) -> dict[str, str]:
    """Collect conservative base-storage aliases from host-wrapper statements."""
    aliases: dict[str, str] = {}
    for stmt in body:
        if not isinstance(stmt, ast.For):
            _update_host_aliases(stmt, aliases)
    return aliases


def canonical_host_tensor_name(name: str, aliases: dict[str, str]) -> str:
    """Resolve ``name`` through aliases collected by collect_host_tensor_aliases."""
    return _canonical_alias(name, aliases)


_ALIAS_PRESERVING_METHODS = frozenset(
    {
        "as_strided",
        "detach",
        "expand",
        "flatten",
        "movedim",
        "narrow",
        "permute",
        "reshape",
        "select",
        "squeeze",
        "swapaxes",
        "swapdims",
        "transpose",
        "unbind",
        "unflatten",
        "unsqueeze",
        "view",
    }
)


def _canonical_alias(name: str, aliases: dict[str, str]) -> str:
    """Resolve a host name to the base storage name tracked by the analysis."""
    path: list[str] = []
    while (base := aliases.get(name)) is not None and base != name:
        path.append(name)
        name = base
    for alias in path:
        aliases[alias] = name
    return name


def _alias_base_name(expr: ast.expr, aliases: dict[str, str]) -> str | None:
    """Return the conservative base name for a host-side tensor alias expression.

    Basic slicing and the standard view-like tensor methods preserve storage.
    Treating an advanced-indexing subscript as an alias is conservative: it can
    add a dependency but cannot remove one.
    """
    if isinstance(expr, ast.Name):
        return _canonical_alias(expr.id, aliases)
    if isinstance(expr, ast.Subscript):
        return _alias_base_name(expr.value, aliases)
    if isinstance(expr, ast.Attribute) and expr.attr in {"T", "mT", "data"}:
        return _alias_base_name(expr.value, aliases)
    if isinstance(expr, ast.Call):
        if (
            isinstance(expr.func, ast.Attribute)
            and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id == "torch"
            and expr.func.attr in _ALIAS_PRESERVING_METHODS
            and expr.args
        ):
            return _alias_base_name(expr.args[0], aliases)
        if (
            isinstance(expr.func, ast.Attribute)
            and expr.func.attr in _ALIAS_PRESERVING_METHODS
        ):
            return _alias_base_name(expr.func.value, aliases)
    return None


def _target_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.List, ast.Tuple)):
        return tuple(name for element in target.elts for name in _target_names(element))
    return ()


def _update_alias_target(
    target: ast.expr,
    value: ast.expr,
    aliases: dict[str, str],
) -> None:
    """Apply one host assignment to the conservative storage-alias map."""
    if isinstance(target, ast.Name):
        base = _alias_base_name(value, aliases)
        if base is None:
            aliases.pop(target.id, None)
        else:
            aliases[target.id] = base
        return

    if not isinstance(target, (ast.List, ast.Tuple)):
        return

    # Pairwise tuple/list assignment preserves the more precise base for each
    # element.  Calls such as ``q, k, v = qkv.unbind(0)`` return multiple views
    # of one storage, so every unpacked name conservatively aliases the call's
    # receiver even though the RHS is not an AST tuple.
    if isinstance(value, (ast.List, ast.Tuple)) and len(target.elts) == len(value.elts):
        for target_element, value_element in zip(target.elts, value.elts, strict=True):
            _update_alias_target(target_element, value_element, aliases)
        return

    base = _alias_base_name(value, aliases)
    for name in _target_names(target):
        if base is None:
            aliases.pop(name, None)
        else:
            aliases[name] = base


def _update_host_aliases(stmt: ast.stmt, aliases: dict[str, str]) -> None:
    """Update base-storage aliases from a host-wrapper assignment."""
    value: ast.expr | None = None
    targets: tuple[ast.expr, ...] = ()
    if isinstance(stmt, ast.Assign):
        targets = tuple(stmt.targets)
        value = stmt.value
    elif isinstance(stmt, ast.AnnAssign):
        targets = (stmt.target,)
        value = stmt.value
    if value is None:
        return
    for target in targets:
        _update_alias_target(target, value, aliases)


def mark_intra_loop_raw_barriers(
    graphs: list[GraphInfo], root_graph_ids: list[int]
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
    ``tl.debug_barrier()`` before it. The barrier flushes every prior write in the
    block, so once emitted the pending-write set is cleared and later loads need a
    new store to re-arm. Storage identity comes from the fake tensor's underlying
    storage, so distinct FX nodes for aliases and views compare equal. The walk
    follows Helion control-flow subgraphs and merges pending writes at joins.
    """
    marker = _IntraLoopRawBarrierMarker(graphs)
    for graph_id in root_graph_ids:
        marker.run_graph(graph_id, set())


class _IntraLoopRawBarrierMarker:
    def __init__(self, graphs: list[GraphInfo]) -> None:
        self.graphs = graphs

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

    def run_graph(self, graph_id: int, written: set[int]) -> set[int]:
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
                    node.meta[INTRA_LOOP_RAW_BARRIER_META] = True
                    pending.clear()
                continue
            if node.target is _if:
                if_graph_id = node.args[1]
                assert isinstance(if_graph_id, int)
                if_info = self.graphs[if_graph_id]
                assert isinstance(if_info, IfGraphInfo)
                if_pending = self.run_graph(if_graph_id, pending)
                if if_info.else_branch is None:
                    pending |= if_pending
                else:
                    else_graph_id = (
                        if_info.else_branch
                        if isinstance(if_info.else_branch, int)
                        else if_info.else_branch.graph_id
                    )
                    else_pending = self.run_graph(else_graph_id, pending)
                    pending = if_pending | else_pending
                continue
            if node.target in (_for_loop, _for_loop_step):
                loop_graph_id = node.args[0]
                assert isinstance(loop_graph_id, int)
                loop_info = self.graphs[loop_graph_id]
                assert isinstance(loop_info, ForLoopGraphInfo)
                loop_input = set() if loop_info.needs_barrier_before else pending
                loop_pending = self.run_graph(loop_graph_id, loop_input)
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
                condition_pending = self.run_graph(body_info.cond_graph_id, pending)
                body_pending = self.run_graph(body_graph_id, condition_pending)
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
