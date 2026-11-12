from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node
    from torch.fx.node import Argument

    from .device_ir import GraphInfo
    from .inductor_lowering import CodegenState


# fx node meta key marking a read (a load, or an ``hl.inline_triton`` or
# ``hl.triton_kernel`` snippet) that must be preceded by a CTA barrier.
INTRA_LOOP_RAW_BARRIER_META = "_needs_debug_barrier_before"


def emit_intra_loop_raw_barrier(state: CodegenState) -> None:
    """The barrier ``mark_intra_loop_raw_barriers`` asks for before the node
    ``state`` lowers: a read that may meet a write earlier in its loop body
    (a load, or an ``hl.inline_triton`` or ``hl.triton_kernel`` snippet)."""
    if state.fx_node is not None and state.fx_node.meta.get(
        INTRA_LOOP_RAW_BARRIER_META
    ):
        for statement in state.device_function.cta_barrier():
            state.add_statement(statement)


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


def _storage_ids(graphs: Sequence[GraphInfo], graph_id: int, obj: object) -> set[int]:
    import torch

    from .device_ir import NodeArgsGraphInfo

    if isinstance(obj, (list, tuple)):
        result: set[int] = set()
        for item in obj:
            result.update(_storage_ids(graphs, graph_id, item))
        return result
    if not isinstance(obj, torch.fx.Node):
        return set()
    graph_info = graphs[graph_id]
    value = obj.meta.get("val")
    result = {id(value.untyped_storage())} if isinstance(value, torch.Tensor) else set()
    if (
        obj.op == "placeholder"
        and obj.graph is graph_info.graph
        and isinstance(graph_info, NodeArgsGraphInfo)
    ):
        result.update(
            _storage_ids(graphs, graph_id, graph_info.placeholder_to_outer_arg(obj))
        )
    if result:
        return result
    for arg in obj.args:
        result.update(_storage_ids(graphs, graph_id, arg))
    return result


def memory_write_target(node: Node) -> Argument | None:
    """The target of an ordinary Helion store or atomic operation."""
    from ..language import _MEMORY_OPS
    from ..language.memory_ops import load

    if node.target in _MEMORY_OPS and node.target is not load:
        return node.args[0]
    return None


def opaque_access_target(node: Node) -> Argument | None:
    """What an op with side effects that is no load, store, atomic or control
    flow may read and write: every input it gets that may name the kernel's
    memory (``hl.inline_triton`` and ``hl.triton_kernel`` may access any
    pointer they are given, as may a distributed copy or barrier), or None
    for any other node.  A value the device code computed names no memory:
    an op given only such values reaches none (``hl.device_print`` of a
    loaded element, a random op, an in-place op on a tile), and is None too.
    ``_phi`` is impure only so that DCE keeps loop carries, and a prefetch
    orders nothing."""
    import torch

    from ..language import memory_ops
    from ..language._tracing_ops import _for_loop
    from ..language._tracing_ops import _for_loop_step
    from ..language._tracing_ops import _if
    from ..language._tracing_ops import _phi
    from ..language._tracing_ops import _while_loop
    from .host_function import HostFunction

    if (
        node.op != "call_function"
        or not node.is_impure()
        or node.target
        in (_phi, memory_ops.prefetch, _for_loop, _for_loop_step, _if, _while_loop)
        or memory_write_target(node) is not None
    ):
        return None
    memory = {
        id(tensor.untyped_storage())
        for tensor in HostFunction.current().tensor_to_origin
    }

    def names_memory(arg: Node) -> bool:
        value = arg.meta.get("val")
        if isinstance(value, torch.Tensor):
            return id(value.untyped_storage()) in memory
        # A scalar is no pointer; anything else (a stack tensor) may be one.
        return not isinstance(
            value, (torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool)
        )

    inputs = [arg for arg in node.all_input_nodes if names_memory(arg)]
    return inputs or None


def graph_memory_access_storage_ids(
    graphs: list[GraphInfo], graph_id: int
) -> tuple[set[int], set[int]]:
    """Return storage identities read and written by a graph and its subgraphs.

    Besides loads, stores, atomics and the graphs of control flow, any other op
    with side effects reads and writes every tensor it is given
    (``opaque_access_target``).
    """
    from ..language import memory_ops
    from ..language._tracing_ops import _for_loop
    from ..language._tracing_ops import _for_loop_step
    from ..language._tracing_ops import _if
    from ..language._tracing_ops import _while_loop
    from ..language.atomic_ops import ATOMIC_OPS
    from .device_ir import IfGraphInfo
    from .device_ir import WhileLoopGraphInfo

    reads: set[int] = set()
    writes: set[int] = set()

    def update_from_subgraph(subgraph_id: int) -> None:
        subgraph_reads, subgraph_writes = graph_memory_access_storage_ids(
            graphs, subgraph_id
        )
        reads.update(subgraph_reads)
        writes.update(subgraph_writes)

    for node in graphs[graph_id].graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is memory_ops.load:
            reads.update(_storage_ids(graphs, graph_id, node.args[0]))
            continue
        if (target := memory_write_target(node)) is not None:
            storage_ids = _storage_ids(graphs, graph_id, target)
            writes.update(storage_ids)
            if node.target in ATOMIC_OPS:
                reads.update(storage_ids)
            continue
        # Handle subgraphs for control flow constructs
        if node.target in (_for_loop, _for_loop_step):
            subgraph_id = node.args[0]
            assert isinstance(subgraph_id, int)
            update_from_subgraph(subgraph_id)
            continue
        if node.target is _if:
            subgraph_id = node.args[1]
            assert isinstance(subgraph_id, int)
            update_from_subgraph(subgraph_id)
            if_info = graphs[subgraph_id]
            assert isinstance(if_info, IfGraphInfo)
            if if_info.else_branch is not None:
                update_from_subgraph(if_info.else_branch)
            continue
        if node.target is _while_loop:
            subgraph_id = node.args[1]
            assert isinstance(subgraph_id, int)
            body_info = graphs[subgraph_id]
            assert isinstance(body_info, WhileLoopGraphInfo)
            update_from_subgraph(body_info.cond_graph_id)
            update_from_subgraph(subgraph_id)
            continue
        if (target := opaque_access_target(node)) is not None:
            storage_ids = _storage_ids(graphs, graph_id, target)
            reads.update(storage_ids)
            writes.update(storage_ids)
    return reads, writes


def mark_intra_loop_raw_barriers(
    graphs: list[GraphInfo],
    root_graph_ids: list[int],
    *,
    mark_in_divergent_control_flow: bool = True,
    atomics: bool = False,
) -> None:
    """Mark loads that read storage written earlier in a device-loop body.

    Within a single device-loop body, ``qkv[a] = v`` followed by ``qkv[b]`` is a
    read-after-write on the same storage. When the store and the load use
    different-shaped index tensors their Triton thread->element layouts differ, so
    an element written by one thread is read back by another with no
    synchronization in between -- a data race (observed corrupting ~0.8% of
    outputs on B200). Helion already inserts ``tl.debug_barrier()`` for the
    analogous hazard *between* sequential top-level loops; this extends the same
    guarantee to a store->load *within* one loop body.

    We mark the load's FX node; the Triton ``load`` codegen emits a
    ``tl.debug_barrier()`` before it (the CuTe codegen a
    ``cute.arch.sync_threads()``). The barrier flushes every prior write in the
    block, so once emitted the pending-write set is cleared and later loads need a
    new write to re-arm. Storage identity comes from the fake tensor's underlying
    storage, so distinct FX nodes for aliases and views compare equal. The walk
    follows Helion control-flow subgraphs and merges pending writes at joins.

    ``mark_in_divergent_control_flow=False`` skips marking loads inside ``hl.if``
    / while bodies: a CuTe SIMT branch condition can vary per thread, and a
    convergent CTA barrier inside a divergent branch deadlocks.  Loop bodies stay
    markable — Helion device-loop trip structure is uniform across the CTA.

    With ``atomics`` (``Backend.intra_loop_barriers_order_atomics``), an
    atomic writes its target, as a store does (``memory_write_target``).  Any
    other op with side effects writes every input it gets that may name the
    kernel's memory (``opaque_access_target``), and an ``hl.inline_triton``
    or ``hl.triton_kernel`` snippet, which may read them too, is marked like a
    load when one holds a pending write: its Triton codegen emits the barrier
    before the snippet.
    """
    marker = _IntraLoopRawBarrierMarker(
        graphs,
        mark_in_divergent_control_flow=mark_in_divergent_control_flow,
        atomics=atomics,
    )
    for graph_id in root_graph_ids:
        marker.run_graph(graph_id, set())


def intra_loop_raw_reread_block_ids(
    graphs: list[GraphInfo],
    root_graph_ids: list[int],
    grid_block_ids: list[list[int]],
    *,
    atomics: bool = False,
) -> set[int]:
    """Block ids of the device loops whose region re-reads storage it stores.

    A loop's region is its body with every loop nested in it.  The walk of
    ``mark_intra_loop_raw_barriers`` is run once per loop (grid loops
    included) from an empty pending-write set, so only a write issued inside
    the region (a store, or with ``atomics`` an atomic) arms a load inside
    it; a write before the loop does not count.

    Such a loop cannot be warp-specialized: Triton's warp specialization
    distributes the body over partitions by data flow -- an epilogue that
    depends only on loads (a descriptor load and the re-read) lands in the
    load partition while the store stays with the MMA's partition -- and the
    CTA barrier Helion emits between the store and the re-read is split per
    partition, so the re-read returned the buffer's previous contents.
    """
    from .device_ir import ForLoopGraphInfo

    regions: list[tuple[int, list[int]]] = [
        (graph_id, block_ids)
        for graph_id, block_ids in zip(root_graph_ids, grid_block_ids, strict=True)
    ]
    regions.extend(
        (info.graph_id, info.block_ids)
        for info in graphs
        if isinstance(info, ForLoopGraphInfo)
    )
    result: set[int] = set()
    for graph_id, block_ids in regions:
        marker = _IntraLoopRawBarrierMarker(graphs, mark=False, atomics=atomics)
        marker.run_graph(graph_id, set())
        if marker.rereads:
            result.update(block_ids)
    return result


class _IntraLoopRawBarrierMarker:
    def __init__(
        self,
        graphs: list[GraphInfo],
        *,
        mark_in_divergent_control_flow: bool = True,
        mark: bool = True,
        atomics: bool = False,
    ) -> None:
        self.graphs = graphs
        self.mark_in_divergent_control_flow = mark_in_divergent_control_flow
        # Whether an atomic arms a later load like a store
        # (``Backend.intra_loop_barriers_order_atomics``).
        self.atomics = atomics
        # ``mark=False`` only counts the loads a marking walk would mark
        # (``rereads``), leaving the graphs and earlier marks untouched.
        self.mark = mark
        self.rereads = 0

    def _read(
        self, node: Node, storages: set[int], pending: set[int], *, divergent: bool
    ) -> None:
        """Mark ``node``, which reads ``storages``, for the barrier before it
        when one of them holds a pending write, and clear the pending writes
        the barrier flushes."""
        if self.mark and node.meta.get(INTRA_LOOP_RAW_BARRIER_META):
            pending.clear()
        elif pending & storages:
            if divergent and not self.mark_in_divergent_control_flow:
                # Cannot place a convergent barrier here; leave the hazard
                # pending so a later uniform read re-arms it.
                return
            if self.mark:
                node.meta[INTRA_LOOP_RAW_BARRIER_META] = True
            self.rereads += 1
            pending.clear()

    def run_graph(
        self, graph_id: int, written: set[int], *, divergent: bool = False
    ) -> set[int]:
        from ..language import memory_ops
        from ..language._tracing_ops import _for_loop
        from ..language._tracing_ops import _for_loop_step
        from ..language._tracing_ops import _if
        from ..language._tracing_ops import _while_loop
        from ..language.inline_triton_ops import inline_triton
        from ..language.inline_triton_ops import triton_kernel
        from .device_ir import ForLoopGraphInfo
        from .device_ir import IfGraphInfo
        from .device_ir import WhileLoopGraphInfo

        pending = set(written)
        for node in self.graphs[graph_id].graph.nodes:
            if node.op != "call_function":
                continue
            if (target := memory_write_target(node)) is not None:
                # A store, or an atomic where this barrier is the one that
                # orders it: what it wrote may not be visible to the other
                # warps yet.
                if self.atomics or node.target is memory_ops.store:
                    pending.update(_storage_ids(self.graphs, graph_id, target))
                continue
            if node.target is memory_ops.load:
                self._read(
                    node,
                    _storage_ids(self.graphs, graph_id, node.args[0]),
                    pending,
                    divergent=divergent,
                )
                continue
            if (target := opaque_access_target(node)) is not None:
                storages = _storage_ids(self.graphs, graph_id, target)
                if node.target in (inline_triton, triton_kernel):
                    self._read(node, storages, pending, divergent=divergent)
                pending.update(storages)
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
                    else_pending = self.run_graph(
                        if_info.else_branch, pending, divergent=True
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
