from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from .. import exc
from .ast_read_writes import ReadWrites

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable

    from .device_ir import GraphInfo


# fx node meta key marking a load that must be preceded by ``tl.debug_barrier()``.
INTRA_LOOP_RAW_BARRIER_META = "_needs_debug_barrier_before"


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


class LoopDependencyChecker:
    """
    A class to check dependencies between top-level for loops in a Helion kernel.

    This class tracks memory accesses (reads and writes) for each top-level for loop
    and raises an error if a later loop reads or writes to anything written in a
    previous loop.
    """

    def __init__(self) -> None:
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self._barrier_after_root: set[int] = set()
        self._root_counter: int = 0
        self.disabled: bool = False

    def insert_barrier_after_root(self, root_id: int) -> None:
        """Record that a barrier separates root_id and root_id+1."""
        self._barrier_after_root.add(root_id)

    def register_loop(self, loop_node: ast.For, root_id: int | None = None) -> None:
        if self.disabled:
            return
        current_root = root_id if root_id is not None else self._root_counter
        if (current_root - 1) in self._barrier_after_root:
            self.reads.clear()
            self.writes.clear()
            self._barrier_after_root.discard(current_root - 1)
        rw = ReadWrites.from_list(loop_node.body)

        self._check_dependencies(rw)

        self.reads |= set(rw.reads)
        self.writes |= set(rw.writes)
        self._root_counter = current_root + 1

    def _check_dependencies(self, rw: ReadWrites) -> None:
        """
        Check for dependencies between the current loop and previous loops.

        Raises:
            exc.LoopDependencyError: If a dependency is detected
        """
        for name in sorted(itertools.chain(rw.reads, rw.writes)):
            if name in self.writes:
                raise exc.LoopDependencyError(name)
