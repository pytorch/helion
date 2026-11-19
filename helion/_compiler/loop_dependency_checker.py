from __future__ import annotations

import ast
import functools
from itertools import starmap
import operator
from typing import TYPE_CHECKING
from typing import NamedTuple

import torch
from torch.fx import Node
from torch.fx import map_arg
from torch.fx.experimental.symbolic_shapes import statically_known_true

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from torch.fx.node import Argument

    from .device_ir import GraphInfo
    from .inductor_lowering import CodegenState


# fx node meta key marking an access (a load, a store, an atomic, or an
# ``hl.inline_triton`` or ``hl.triton_kernel`` snippet) that must be preceded
# by a CTA barrier.
INTRA_LOOP_RAW_BARRIER_META = "_needs_debug_barrier_before"
# The tensors whose pending writes that barrier flushes (None for a write
# through device pointers, which may reach any memory).
INTRA_LOOP_RAW_BARRIER_FLUSHES_META = "_debug_barrier_before_flushes"
# The tensors whose pending reads that barrier completes, of those a write of
# the kernel may reach (None for a read through device pointers).
INTRA_LOOP_RAW_BARRIER_COMPLETES_META = "_debug_barrier_before_completes"
# fx node meta key on a store an earlier iteration's copy of which it may
# overwrite with no barrier between (``_meets``): through TMA it drains first.
RESTORES_ITSELF_META = "_restores_its_earlier_store"
# The storage of an access through device pointers (a stack tensor): no
# storage id, as ``id`` is never 0.
_THROUGH_POINTERS = 0
# The loop of an access an earlier iteration of a while loop issued: no block
# id, so it tells no iterations apart, and it stays after the loop.
_WHILE = (-1,)


def emit_intra_loop_raw_barrier(
    state: CodegenState, accessed: torch.Tensor | None = None, *, full: bool = False
) -> None:
    """The barrier ``mark_intra_loop_raw_barriers`` asks for before the node
    ``state`` lowers: an access of ``accessed`` (a load, a store or an
    atomic), or an ``hl.inline_triton`` or ``hl.triton_kernel`` snippet, that
    may meet another thread's access earlier in its loop body or in an
    earlier iteration of its loop.  ``full`` asks for the barrier that drains
    TMA stores even after reads alone."""
    if state.fx_node is not None and state.fx_node.meta.get(
        INTRA_LOOP_RAW_BARRIER_META
    ):
        from .compile_environment import CompileEnvironment
        from .cross_loop_codegen import _allocation
        from .cross_loop_codegen import _covering

        flushed = state.fx_node.meta[INTRA_LOOP_RAW_BARRIER_FLUSHES_META]
        # After other threads' reads alone, bar.sync completes them; a write
        # among the pending accesses may be a TMA store, which it drains.
        statements = (
            state.device_function.cta_barrier()
            if flushed or full
            else state.device_function.cta_read_barrier()
        )
        ordered = (
            accessed,
            *flushed,
            *state.fx_node.meta[INTRA_LOOP_RAW_BARRIER_COMPLETES_META],
        )
        tensors = [tensor for tensor in ordered if tensor is not None]
        overlapped = set(CompileEnvironment.current().input_storage_aliases.values())
        # Its fence orders reads of the access's tensor, of what it flushes
        # and of what it completes alone: a ring partition reading none of
        # that drops the fence (and the loop it would replay for it).  Not
        # after an access through device pointers (None), nor where an
        # argument others overlap at run time stands for their storage
        # (``CompileEnvironment.input_storage_aliases``), whose allocations
        # differ.
        if len(tensors) == len(ordered) and not any(
            _storage_id(tensor) in overlapped for tensor in tensors
        ):
            statements = _covering(
                state.device_function,
                statements,
                frozenset(_allocation(tensor) for tensor in tensors),
            )
        for statement in statements:
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


def _storage_id(tensor: torch.Tensor) -> int:
    """The ``id`` of ``tensor``'s untyped storage, where tensor arguments that
    overlap at runtime count as one storage
    (``CompileEnvironment.input_storage_aliases``).  (A test may run a walk
    outside of a compile.)"""
    from .compile_environment import CompileEnvironment

    storage = id(tensor.untyped_storage())
    if CompileEnvironment.has_current():
        return CompileEnvironment.current().input_storage_aliases.get(storage, storage)
    return storage


def _storage_ids(graphs: Sequence[GraphInfo], graph_id: int, obj: object) -> set[int]:
    return set(_storage_tensors(graphs, graph_id, obj))


def _storage_tensors(
    graphs: Sequence[GraphInfo], graph_id: int, obj: object
) -> dict[int, torch.Tensor]:
    """The storages ``obj`` (a node, or nodes) may name, by ``id`` of their
    untyped storage, each with a (fake) tensor viewing it."""
    from .device_ir import NodeArgsGraphInfo

    if isinstance(obj, (list, tuple)):
        result: dict[int, torch.Tensor] = {}
        for item in obj:
            result.update(_storage_tensors(graphs, graph_id, item))
        return result
    if not isinstance(obj, torch.fx.Node):
        return {}
    graph_info = graphs[graph_id]
    value = obj.meta.get("val")
    result = {_storage_id(value): value} if isinstance(value, torch.Tensor) else {}
    if (
        obj.op == "placeholder"
        and obj.graph is graph_info.graph
        and isinstance(graph_info, NodeArgsGraphInfo)
    ):
        result.update(
            _storage_tensors(graphs, graph_id, graph_info.placeholder_to_outer_arg(obj))
        )
    if result:
        return result
    for arg in obj.args:
        result.update(_storage_tensors(graphs, graph_id, arg))
    return result


def memory_write_target(node: Node) -> Argument | None:
    """The target of an ordinary Helion store or atomic operation."""
    from ..language import _MEMORY_OPS
    from ..language.memory_ops import load

    if node.target in _MEMORY_OPS and node.target is not load:
        return node.args[0]
    return None


def reaches_through_pointers(target: object) -> bool:
    """Whether a memory op's ``target`` (its first argument) is a stack
    tensor, ``(tensor_like, dev_ptrs)``: the op reaches whatever memory its
    device pointers name, which no storage of the kernel bounds."""
    return isinstance(target, (list, tuple))


def opaque_access_target(node: Node) -> list[Node] | None:
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


def _child_graph_ids(graphs: Sequence[GraphInfo], node: Node) -> list[int] | None:
    """The graphs a control-flow node runs (a loop's body, an ``if``'s branches,
    a while loop's condition and body), or None for any other node."""
    from ..language._tracing_ops import _for_loop
    from ..language._tracing_ops import _for_loop_step
    from ..language._tracing_ops import _if
    from ..language._tracing_ops import _while_loop
    from .device_ir import IfGraphInfo
    from .device_ir import WhileLoopGraphInfo

    if node.target in (_for_loop, _for_loop_step):
        assert isinstance(node.args[0], int)
        return [node.args[0]]
    if node.target is _if:
        assert isinstance(node.args[1], int)
        if_info = graphs[node.args[1]]
        assert isinstance(if_info, IfGraphInfo)
        return [node.args[1]] + (
            [] if if_info.else_branch is None else [if_info.else_branch]
        )
    if node.target is _while_loop:
        assert isinstance(node.args[1], int)
        body_info = graphs[node.args[1]]
        assert isinstance(body_info, WhileLoopGraphInfo)
        return [body_info.cond_graph_id, node.args[1]]
    return None


def _pointer_storages(target: object) -> set[int]:
    """What an access of a memory op's ``target`` reaches beyond the storages
    it names: ``_THROUGH_POINTERS`` (any storage) through device pointers
    (``reaches_through_pointers``), else nothing."""
    return {_THROUGH_POINTERS} if reaches_through_pointers(target) else set()


def _non_overlapping(target: object) -> bool:
    """Whether the tensor ``target`` names maps distinct elements to distinct
    addresses: each dimension's stride, largest first, spans all the
    dimensions inside it."""
    from .compile_environment import CompileEnvironment

    value = target.meta.get("val") if isinstance(target, Node) else None
    if not isinstance(value, torch.Tensor) or not CompileEnvironment.has_current():
        return False
    env = CompileEnvironment.current()
    if not env.traced_strides_hold(value):
        return False
    dims: list[tuple[int | torch.SymInt, int]] = []
    for dim, size in enumerate(value.size()):
        if statically_known_true(size == 1):
            continue
        # A stride the cache key leaves open, as under dynamic shapes, may be
        # any in a later call, an expanded view's 0 included.
        if (stride := env.known_stride(value, dim)) is None:
            return False
        dims.append((size, stride))
    inner: int | torch.SymInt = 0
    for size, stride in sorted(dims, key=operator.itemgetter(1)):
        if not statically_known_true(stride > inner):
            return False
        inner = inner + (size - 1) * stride
    return True


def _same_index(first: object, second: object) -> bool:
    """Whether two subscripts of FX nodes and constants are the same one."""
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            starmap(_same_index, zip(first, second, strict=True))
        )
    if isinstance(first, Node) or isinstance(second, Node):
        return first is second
    return type(first) is type(second) and first == second


def _keeps_positions(node: Node, shape: torch.Size) -> bool:
    """Whether ``node`` computes each element of its result from the same
    position of its operands, as a pointwise op whose result has the loaded
    ``shape`` does; a transpose or a reduction moves values between
    positions."""
    value = node.meta.get("val")
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.pointwise in node.target.tags
        and isinstance(value, torch.Tensor)
        and len(value.shape) == len(shape)
        and all(
            statically_known_true(size == other)
            for size, other in zip(value.shape, shape, strict=True)
        )
    )


def _ancestors(roots: Argument) -> set[Node]:
    """The FX nodes in ``roots``, and the nodes of their graph they are
    computed from."""
    stack: list[Node] = []
    map_arg(roots, stack.append)
    seen: set[Node] = set()
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(node.all_input_nodes)
    return seen


@functools.cache
def _self_commuting_atomics() -> frozenset[object]:
    """The atomics two of which give the same result in either order."""
    from ..language import atomic_ops

    return frozenset(
        {
            atomic_ops.atomic_add,
            atomic_ops.atomic_and,
            atomic_ops.atomic_max,
            atomic_ops.atomic_min,
            atomic_ops.atomic_or,
            atomic_ops.atomic_xor,
        }
    )


@functools.cache
def flat_offset_loads() -> frozenset[object]:
    """The loads of the tensor they are passed at flat offsets (the packed
    quantized loads): a read of anywhere in the tensor, not of an indexed
    tile, which memory ops of the indexing config do not cover."""
    from ..language import quantized_ops

    return frozenset(
        {
            quantized_ops.load_bfloat16_x16_to_float16,
            quantized_ops.load_float4_e2m1fn_x16_to_float16,
        }
    )


def _ordered_flat_loads() -> frozenset[object]:
    """``flat_offset_loads`` the barrier walk orders: none on a backend whose
    packed-load codegen places no barrier
    (``Backend.intra_loop_barriers_order_flat_loads``), nor outside a compile
    environment (a CuTe heuristic reading ``graph_memory_access_storage_ids``)."""
    from .compile_environment import CompileEnvironment

    if (
        CompileEnvironment.has_current()
        and CompileEnvironment.current().backend.intra_loop_barriers_order_flat_loads
    ):
        return flat_offset_loads()
    return frozenset()


def mark_intra_loop_raw_barriers(
    graphs: list[GraphInfo],
    root_graph_ids: list[int],
    *,
    mark_in_divergent_control_flow: bool = True,
    atomics: bool = False,
    carries: bool = True,
    reads: bool = False,
) -> None:
    """Mark each access that may meet another thread's earlier access of the
    same element with no barrier in between, so its codegen emits a CTA
    barrier before it (``emit_intra_loop_raw_barrier``).

    Two accesses meet when one is a write and they may touch the same
    element (``_meets``): a read after a write, and with ``reads`` a write
    after a read or a write.  Two atomics of one op that commutes with itself
    do not meet.  A snippet (``hl.inline_triton``, ``hl.triton_kernel``)
    reads and writes every tensor it is given, and an access through device
    pointers may reach any storage.  A loop body starts with its own
    accesses pending, as an earlier iteration leaves them (with ``carries``,
    or where the loop reaches through device pointers).
    ``mark_in_divergent_control_flow=False`` leaves a hazard inside ``hl.if``
    / while bodies pending for the next uniform access (a CuTe branch may
    diverge, and a barrier there deadlocks)."""
    marker = _IntraLoopRawBarrierMarker(
        graphs,
        mark_in_divergent_control_flow=mark_in_divergent_control_flow,
        atomics=atomics,
        carries=carries,
        reads=reads,
    )
    marker.moving = _moving_block_ids(graphs)
    for graph_id in root_graph_ids:
        marker.written |= (
            graph_accesses(graphs, graph_id, atomics=atomics).writes().storages()
        )
    for graph_id in root_graph_ids:
        marker.run_graph(graph_id, _Pending())


def intra_loop_raw_reread_block_ids(
    graphs: list[GraphInfo],
    root_graph_ids: list[int],
    grid_block_ids: list[list[int]],
    *,
    atomics: bool = False,
    carries: bool = True,
    reads: bool = False,
) -> set[int]:
    """Block ids of the device loops whose region would take an intra-loop
    barrier (``mark_intra_loop_raw_barriers``, walked from an empty pending
    set over each loop's region, grid loops included).

    Such a loop cannot be warp-specialized: Triton's warp specialization
    distributes the body over partitions by data flow, and the CTA barrier
    Helion emits between the two accesses is split per partition."""
    from .device_ir import ForLoopGraphInfo

    regions: list[tuple[int, list[int], bool]] = [
        (graph_id, block_ids, False)
        for graph_id, block_ids in zip(root_graph_ids, grid_block_ids, strict=True)
    ]
    regions.extend(
        (info.graph_id, info.block_ids, True)
        for info in graphs
        if isinstance(info, ForLoopGraphInfo)
    )
    result: set[int] = set()
    moving = _moving_block_ids(graphs)
    for graph_id, block_ids, loop in regions:
        marker = _IntraLoopRawBarrierMarker(
            graphs, mark=False, atomics=atomics, carries=carries, reads=reads
        )
        marker.moving = moving
        start = _Pending()
        if loop:
            start = marker._iteration_start(
                start, [graph_id], tuple(block_ids), divergent=False
            )
        marker.run_graph(graph_id, start)
        if marker.barriered:
            result.update(block_ids)
    return result


class _Access(NamedTuple):
    """An access the walk tracks: its op (``node``), whether it reads,
    writes or is an atomic, the storage it may reach (``_THROUGH_POINTERS``
    for any), and the block ids of the loop an earlier iteration of which
    issued it (empty for the current iteration, ``_WHILE`` for a while
    loop)."""

    node: Node
    kind: str
    storage: int
    loop: tuple[int, ...] = ()


class _Pending:
    """The accesses issued since the last barrier."""

    __slots__ = ("accesses",)

    def __init__(self, accesses: Iterable[_Access] = ()) -> None:
        self.accesses: set[_Access] = set(accesses)

    def copy(self) -> _Pending:
        return _Pending(self.accesses)

    def update(self, other: _Pending) -> None:
        self.accesses |= other.accesses

    def clear(self) -> None:
        self.accesses.clear()

    def add(self, node: Node, kind: str, storages: Iterable[int]) -> None:
        self.accesses.update(_Access(node, kind, storage) for storage in storages)

    def storages(self) -> set[int]:
        return {access.storage for access in self.accesses}

    def writes(self) -> _Pending:
        """These accesses but the plain reads, for a walk that orders no
        write after a read."""
        return _Pending(a for a in self.accesses if a.kind != "read")

    def carried(self, loop: tuple[int, ...], carry: bool) -> _Pending:
        """These accesses issued by an earlier iteration of ``loop`` (with
        ``carry``), or no longer (without). One an earlier iteration of an
        enclosing loop issued stays so in every iteration of ``loop``."""
        old, new = ((), loop) if carry else (loop, ())
        return _Pending(
            a._replace(loop=new) if a.loop == old else a for a in self.accesses
        )

    def conflicts(
        self, accesses: _Pending, *, reads: bool, moving: frozenset[int] = frozenset()
    ) -> bool:
        """Whether one of ``accesses`` meets one of these pending ones (the
        tiles of ``moving`` may begin elsewhere in another run)."""
        return any(
            _meets(earlier, later, reads=reads, moving=moving)
            for later in accesses.accesses
            for earlier in self.accesses
        )


def _reaches(storage: int, storages: set[int]) -> bool:
    """Whether an access of ``storage`` may meet one of ``storages``."""
    return storage in storages or _THROUGH_POINTERS in (storages | {storage})


def _meets(
    earlier: _Access, later: _Access, *, reads: bool, moving: frozenset[int]
) -> bool:
    """Whether ``later`` may touch an element another thread's ``earlier``
    access touched, one of the two a write: a read (or an atomic, which reads
    its target) after a write, and with ``reads`` a write after either.
    Unless they reach different storages, both are the same atomic op that
    commutes with itself, old values unused (a histogram's or split-K's
    adds), or they touch elements apart (``_apart``)."""
    if earlier.kind == "read" and later.kind == "read":
        return False
    if later.kind == "write" and not reads:
        return False
    if earlier.node is later.node and _keeps_threads(later.node, moving):
        # The same store in an earlier iteration: each element from the same
        # thread, after its earlier copy (``RESTORES_ITSELF_META``).
        return False
    if not _reaches(earlier.storage, {later.storage}):
        return False
    if (
        earlier.kind == later.kind == "atomic"
        and earlier.node.target is later.node.target
        and later.node.target in _self_commuting_atomics()
        and not earlier.node.users
        and not later.node.users
    ):
        return False
    if (
        earlier.kind == "read"
        and not earlier.loop
        and later.kind == "write"
        and _writes_back(earlier.node, later.node)
    ):
        return False
    return not _apart(earlier, later, moving)


def _keeps_threads(store: Node, moving: frozenset[int] = frozenset()) -> bool:
    """Whether every run of ``store`` stores each element from the same
    thread: a store into a tensor (no device pointers) at an index of no
    tensor, which puts each element at one position of its value, and no
    tile on a dimension that may have one element (which would broadcast
    every position of the tile to it) or of a loop whose tiles may begin
    elsewhere in another run (block ids in ``moving``)."""
    from ..language import memory_ops
    from .compile_environment import CompileEnvironment

    target, index = store.args[:2]
    if (
        store.target is not memory_ops.store
        or not isinstance(target, Node)
        or not isinstance(index, (list, tuple))
    ):
        return False
    sizes = iter(target.meta["val"].size())
    for entry in index:
        if entry is None:
            continue  # a new axis
        if entry is Ellipsis:
            return False
        size = next(sizes, None)
        if not isinstance(entry, Node):
            continue  # an integer or a slice
        if not isinstance(entry.meta.get("val"), torch.SymInt):
            return False  # a tensor
        if _index_range(entry) is None and (
            not statically_known_true(size != 1)
            or CompileEnvironment.current().get_block_id(entry.meta["val"]) in moving
        ):
            return False
    return True


def _moving_block_ids(graphs: Sequence[GraphInfo]) -> frozenset[int]:
    """The block ids of the device loops whose begin is not ``_fixed``:
    another run of such a loop may start its tiles elsewhere."""
    from ..language._tracing_ops import _for_loop
    from ..language._tracing_ops import _for_loop_step

    moving: set[int] = set()
    for info in graphs:
        for node in info.graph.nodes:
            if node.op == "call_function" and node.target in (
                _for_loop,
                _for_loop_step,
            ):
                block_ids = graphs[node.args[0]].block_ids
                moving.update(
                    block_id
                    for block_id, begin in zip(block_ids, node.args[1], strict=True)
                    if not _fixed(begin)
                )
    return frozenset(moving)


def _fixed(value: object) -> bool:
    """Whether ``value`` is the same in every iteration of every loop: an
    integer, or an expression of host values and block sizes."""
    from .host_function import HostFunction
    from .variable_origin import BlockSizeOrigin

    if isinstance(value, Node):
        value = value.meta.get("val")
    if type(value) is int:
        return True
    if not isinstance(value, torch.SymInt):
        return False
    origins = HostFunction.current().expr_to_origin
    return all(
        (info := origins.get(symbol)) is not None
        and (info.origin.is_host() or isinstance(info.origin, BlockSizeOrigin))
        for symbol in value._sympy_().free_symbols
    )


def _writes_back(load: Node, store: Node) -> bool:
    """Whether ``store`` writes back what ``load`` read, in the same body
    and iteration (``x[t] = f(x[t])``): the same tensor, mapping distinct
    indices to distinct addresses, at the same index of no tensor
    (``_keeps_threads``: distinct positions name distinct elements), with a
    value computed from the loaded one by pointwise ops
    (``_keeps_positions``) that nothing but the store uses.  Each element is then stored from the value loaded at
    its position, after its load; another thread holding a copy of the
    element may load it late, but its copy reaches nothing but the store:
    directly, where Triton masks its write off, or through a cross-thread
    exchange, which synchronizes every thread before any write."""
    from ..language import memory_ops

    if (
        load.target is not memory_ops.load
        or not _keeps_threads(store)
        or load.graph is not store.graph
        or load.args[0] is not store.args[0]
        or not _non_overlapping(store.args[0])
        or not _same_index(load.args[1], store.args[1])
    ):
        return False
    loaded = load.meta.get("val")
    operands = _ancestors([arg for arg in store.args[2:] if isinstance(arg, Node)])
    if not isinstance(loaded, torch.Tensor) or load not in operands:
        return False
    users = set(load.users)
    stack = list(users)
    while stack:
        node = stack.pop()
        if node is not store:
            if node not in operands or not _keeps_positions(node, loaded.shape):
                return False
            stack.extend(user for user in node.users if user not in users)
            users.update(node.users)
    return True


def _apart(
    earlier: _Access, later: _Access, moving: frozenset[int] = frozenset()
) -> bool:
    """Whether two accesses of one tensor touch no element in common: pieces
    of one store epilogue subtiling split, or, in a tensor that maps distinct
    indices to distinct addresses (``_non_overlapping``), indices that on
    some dimension of more than one element take values that never meet
    (``_index_range``; across iterations, ``_fixed`` ones), or that tell
    apart the iterations of the loop an earlier iteration of which issued
    ``earlier`` (``_loop_block``), on every block of the loop."""
    from ..language import memory_ops
    from ..language.atomic_ops import ATOMIC_OPS

    first, second = earlier.node, later.node
    indexed = (memory_ops.load, memory_ops.store, *ATOMIC_OPS)
    if first.target not in indexed or second.target not in indexed:
        return False
    target = first.args[0]
    if not isinstance(target, Node) or not isinstance(second.args[0], Node):
        return False
    group = first.meta.get("epilogue_subtile_group_id")
    if (
        group is not None
        and group == second.meta.get("epilogue_subtile_group_id")
        and first.graph is second.graph
        and target is second.args[0]
        and (not earlier.loop or _keeps_threads(second, moving))
    ):
        return True
    if target.meta.get("val") is not second.args[0].meta.get("val") or (
        not _non_overlapping(target)
    ):
        return False
    one, other = first.args[1], second.args[1]
    if (
        not isinstance(one, (list, tuple))
        or not isinstance(other, (list, tuple))
        or len(one) != len(other)
        or any(index is None or index is Ellipsis for index in (*one, *other))
    ):
        return False
    fake = target.meta.get("val")
    if not isinstance(fake, torch.Tensor) or fake.dim() != len(one):
        return False
    tiles: set[int] = set()
    for a, b, size in zip(one, other, fake.size(), strict=True):
        # A dimension of one element takes every index to it.
        if not statically_known_true(size != 1):
            continue
        range_a, range_b = _index_range(a), _index_range(b)
        if (
            range_a is not None
            and range_b is not None
            # In another iteration, a loop index held another value.
            and (not earlier.loop or all(map(_fixed, (*range_a, *range_b))))
        ):
            # Exact integers, which 32-bit index arithmetic may wrap.
            for (low, high), (other_low, other_high) in (
                (range_a, range_b),
                (range_b, range_a),
            ):
                if statically_known_true(high < other_low) and statically_known_true(
                    other_high - low < 2**31
                ):
                    return True
        block_id = _loop_block(a, earlier.loop)
        if (
            block_id is not None
            and isinstance(a, Node)
            and isinstance(b, Node)
            and str(b.meta.get("val")) == str(a.meta.get("val"))
        ):
            tiles.add(block_id)
    # Two iterations differ on one of the loop's blocks at least.
    return bool(earlier.loop) and tiles == set(earlier.loop)


def _index_range(
    index: object,
) -> tuple[int | torch.SymInt, int | torch.SymInt] | None:
    """The first and last index an index entry takes: a non-negative
    integer, a slice with such bounds, a scalar (such as a tile's
    ``begin``), a unit-step ``hl.arange`` of 32 or 64 bits, or one of those
    plus or minus an integer, or a broadcast of one; None for anything else
    (a tile, a loaded index)."""
    from ..language.view_ops import subscript
    from .compile_environment import CompileEnvironment

    if type(index) is int:
        return (index, index) if index >= 0 else None
    if isinstance(index, slice):
        start = 0 if index.start is None else index.start
        if index.step in (None, 1) and type(start) is int and type(index.stop) is int:
            if 0 <= start < index.stop:
                return start, index.stop - 1
        return None
    if not isinstance(index, Node):
        return None
    value = index.meta.get("val")
    if isinstance(value, torch.SymInt):
        tiles = {
            info.var._sympy_() for info in CompileEnvironment.current().block_sizes
        }
        return None if value._sympy_().free_symbols & tiles else (value, value)
    if not isinstance(value, torch.Tensor) or value.dtype not in (
        torch.int32,
        torch.int64,
    ):
        # A narrower index wraps.
        return None
    if index.target is subscript:
        # New axes and whole slices only: the same values, broadcast.
        base, entries = index.args[:2]
        if isinstance(entries, (list, tuple)) and all(
            entry is None or entry == slice(None) for entry in entries
        ):
            return _index_range(base)
        return None
    if index.target is torch.ops.prims.iota.default:
        (length,) = index.args
        start = index.kwargs.get("start", 0)
        if (
            index.kwargs.get("step") != 1
            or type(start) is not int
            or not isinstance(length, (int, torch.SymInt))
        ):
            return None
        return start, start + length - 1
    shifts = {
        torch.ops.aten.add.Tensor: 1,
        torch.ops.aten.add.Scalar: 1,
        torch.ops.aten.sub.Tensor: -1,
        torch.ops.aten.sub.Scalar: -1,
    }
    if index.target in shifts and len(index.args) == 2 and not index.kwargs:
        inner, shift = index.args
        if type(shift) is int and (shifted := _index_range(inner)) is not None:
            delta = shifts[index.target] * shift
            return shifted[0] + delta, shifted[1] + delta
    return None


def _loop_block(index: object, block_ids: tuple[int, ...]) -> int | None:
    """The one of ``block_ids`` whose iterations ``index`` tells apart (a
    tile or a grid index of its loop, or a tile's begin, end or id), or
    None."""
    from .host_function import HostFunction
    from .variable_origin import BlockSizeOrigin
    from .variable_origin import GridOrigin
    from .variable_origin import TileCountOrigin
    from .variable_origin import TileExtentOrigin

    value = index.meta.get("val") if isinstance(index, Node) else None
    if not block_ids or not isinstance(value, torch.SymInt):
        return None
    info = HostFunction.current().expr_to_origin.get(value._sympy_())
    origin = None if info is None else info.origin
    if (
        isinstance(origin, (BlockSizeOrigin, GridOrigin))
        and not isinstance(origin, (TileExtentOrigin, TileCountOrigin))
        and origin.block_id in block_ids
    ):
        return origin.block_id
    return None


def graph_accesses(
    graphs: Sequence[GraphInfo],
    graph_id: int,
    *,
    atomics: bool = True,
    tensors: dict[int, torch.Tensor] | None = None,
) -> _Pending:
    """Every access of a graph and its subgraphs, as the marking walk
    classifies them; ``tensors`` collects a tensor of each storage."""
    tensors = {} if tensors is None else tensors
    result = _Pending()
    for node in graphs[graph_id].graph.nodes:
        if (children := _child_graph_ids(graphs, node)) is not None:
            for child in children:
                result.update(
                    graph_accesses(graphs, child, atomics=atomics, tensors=tensors)
                )
        elif (
            accesses := _op_accesses(graphs, graph_id, node, atomics, tensors)
        ) is not None:
            result.update(accesses)
    return result


def _op_accesses(
    graphs: Sequence[GraphInfo],
    graph_id: int,
    node: Node,
    atomics: bool,
    tensors: dict[int, torch.Tensor],
) -> _Pending | None:
    """The accesses of an op that is no control flow (None for one that
    accesses no memory, or an atomic without ``atomics``): a load reads its
    tensor (a packed load anywhere in it), a store writes it, and a snippet
    reads and writes every tensor it is given; one given device pointers
    reaches any storage."""
    from ..language import memory_ops

    def storages(target: object) -> set[int]:
        found = _storage_tensors(graphs, graph_id, target)
        tensors.update(found)
        return set(found) | _pointer_storages(target)

    if node.op != "call_function":
        return None
    accesses = _Pending()
    if node.target is memory_ops.load or node.target in _ordered_flat_loads():
        accesses.add(node, "read", storages(node.args[0]))
    elif (target := memory_write_target(node)) is not None:
        if node.target is not memory_ops.store and not atomics:
            return None
        kind = "write" if node.target is memory_ops.store else "atomic"
        accesses.add(node, kind, storages(target))
    elif (inputs := opaque_access_target(node)) is not None:
        accessed = {
            storage
            for arg in inputs
            for storage in (
                storages(arg)
                if isinstance(arg.meta.get("val"), torch.Tensor)
                else {_THROUGH_POINTERS}
            )
        }
        accesses.add(node, "read", accessed)
        accesses.add(node, "write", accessed)
    else:
        return None
    return accesses


class _IntraLoopRawBarrierMarker:
    def __init__(
        self,
        graphs: list[GraphInfo],
        *,
        mark_in_divergent_control_flow: bool = True,
        mark: bool = True,
        atomics: bool = False,
        carries: bool = True,
        reads: bool = False,
    ) -> None:
        self.graphs = graphs
        self.mark_in_divergent_control_flow = mark_in_divergent_control_flow
        # ``Backend.intra_loop_barriers_order_atomics``.
        self.atomics = atomics
        # ``Backend.intra_loop_barriers_order_reads``.
        self.reads = reads
        # ``Backend.intra_loop_barriers_carry_iterations``.
        self.carries = carries
        # ``mark=False`` leaves the graphs untouched and collects in
        # ``barriered`` the accesses a marking walk would mark.
        self.mark = mark
        self.barriered: set[Node] = set()
        # A tensor of each storage the walk met, by ``_storage_ids`` id.
        self.tensors: dict[int, torch.Tensor] = {}
        # The storages the kernel writes.
        self.written: set[int] = set()
        # The block ids of the loops whose tiles may begin elsewhere in
        # another run (``_moving_block_ids``).
        self.moving: frozenset[int] = frozenset()

    def _tensors(self, storages: Iterable[int]) -> tuple[torch.Tensor | None, ...]:
        return tuple(
            None if storage == _THROUGH_POINTERS else self.tensors[storage]
            for storage in storages
        )

    def _access(
        self,
        node: Node,
        accesses: _Pending,
        pending: _Pending,
        *,
        divergent: bool,
        markable: bool = True,
    ) -> None:
        """Order ``node`` (its ``accesses``) after the pending accesses it
        meets with a barrier before it, then add its accesses."""
        if self.mark and node.meta.get(INTRA_LOOP_RAW_BARRIER_META):
            # Marked by an earlier walk of this graph: its barrier orders all.
            pending.clear()
        elif (
            markable
            and pending.conflicts(accesses, reads=self.reads, moving=self.moving)
            and (self.mark_in_divergent_control_flow or not divergent)
        ):
            self.barriered.add(node)
            if self.mark:
                node.meta[INTRA_LOOP_RAW_BARRIER_META] = True
                writes = pending.writes().storages()
                node.meta[INTRA_LOOP_RAW_BARRIER_FLUSHES_META] = self._tensors(writes)
                node.meta[INTRA_LOOP_RAW_BARRIER_COMPLETES_META] = self._tensors(
                    storage
                    for storage in pending.storages() - writes
                    if _reaches(storage, self.written)
                )
            pending.clear()
        elif (
            self.mark
            and self.reads
            and any(
                earlier.node is node
                and earlier.loop
                and not _apart(earlier, later, self.moving)
                for earlier in pending.accesses
                for later in accesses.accesses
            )
        ):
            node.meta[RESTORES_ITSELF_META] = True
        pending.update(accesses if self.reads else accesses.writes())

    def run_graph(
        self, graph_id: int, start: _Pending, *, divergent: bool = False
    ) -> _Pending:
        from ..language._tracing_ops import _for_loop
        from ..language._tracing_ops import _for_loop_step
        from ..language._tracing_ops import _if
        from ..language._tracing_ops import _while_loop
        from ..language.inline_triton_ops import inline_triton
        from ..language.inline_triton_ops import triton_kernel
        from .device_ir import ForLoopGraphInfo
        from .device_ir import IfGraphInfo
        from .device_ir import WhileLoopGraphInfo

        pending = start.copy()
        for node in self.graphs[graph_id].graph.nodes:
            if node.op != "call_function":
                continue
            accesses = _op_accesses(
                self.graphs, graph_id, node, self.atomics, self.tensors
            )
            if accesses is not None:
                # Only a snippet's codegen can place a barrier before it.
                markable = opaque_access_target(node) is None or node.target in (
                    inline_triton,
                    triton_kernel,
                )
                self._access(
                    node, accesses, pending, divergent=divergent, markable=markable
                )
                continue
            if node.target is _if:
                if_graph_id = node.args[1]
                assert isinstance(if_graph_id, int)
                if_info = self.graphs[if_graph_id]
                assert isinstance(if_info, IfGraphInfo)
                if_pending = self.run_graph(if_graph_id, pending, divergent=True)
                if if_info.else_branch is not None:
                    pending = self.run_graph(
                        if_info.else_branch, pending, divergent=True
                    )
                pending.update(if_pending)
                continue
            if node.target in (_for_loop, _for_loop_step):
                loop_graph_id = node.args[0]
                assert isinstance(loop_graph_id, int)
                loop_info = self.graphs[loop_graph_id]
                assert isinstance(loop_info, ForLoopGraphInfo)
                if loop_info.needs_barrier_before:
                    pending.clear()
                loop = tuple(loop_info.block_ids)
                loop_pending = self.run_graph(
                    loop_graph_id,
                    self._iteration_start(
                        pending, [loop_graph_id], loop, divergent=divergent
                    ),
                    divergent=divergent,
                )
                # Zero iterations preserve the input state.
                pending.update(loop_pending.carried(loop, False))
                continue
            if node.target is _while_loop:
                body_graph_id = node.args[1]
                assert isinstance(body_graph_id, int)
                body_info = self.graphs[body_graph_id]
                assert isinstance(body_info, WhileLoopGraphInfo)
                condition_pending = self.run_graph(
                    body_info.cond_graph_id,
                    self._iteration_start(
                        pending,
                        [body_info.cond_graph_id, body_graph_id],
                        _WHILE,
                        divergent=True,
                    ),
                    divergent=True,
                )
                body_pending = self.run_graph(
                    body_graph_id, condition_pending, divergent=True
                )
                # The condition executes at least once; the body may not execute.
                pending = condition_pending
                pending.update(body_pending)
        return pending

    def _iteration_start(
        self,
        entry: _Pending,
        graph_ids: list[int],
        loop: tuple[int, ...],
        *,
        divergent: bool,
    ) -> _Pending:
        """What may be pending when an iteration of the loop whose iteration
        walks ``graph_ids`` starts: ``entry``, and where the loop carries,
        what an earlier iteration leaves pending, walked until it stops
        growing."""
        start = entry.copy()
        if not self.carries and not any(
            _THROUGH_POINTERS in graph_accesses(self.graphs, graph_id).storages()
            for graph_id in graph_ids
        ):
            return start
        probe = _IntraLoopRawBarrierMarker(
            self.graphs,
            mark_in_divergent_control_flow=self.mark_in_divergent_control_flow,
            mark=False,
            atomics=self.atomics,
            carries=self.carries,
            reads=self.reads,
        )
        probe.tensors = self.tensors
        probe.moving = self.moving
        while True:
            pending = start
            for graph_id in graph_ids:
                pending = probe.run_graph(graph_id, pending, divergent=divergent)
            # What the iteration issued; what was pending before it stays.
            left = _Pending(pending.accesses - start.accesses).carried(loop, True)
            if left.accesses <= start.accesses:
                return start
            start.update(left)
