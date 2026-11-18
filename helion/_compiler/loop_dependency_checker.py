from __future__ import annotations

import ast
import enum
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
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet

    from torch.fx.node import Argument

    from .device_function import DeviceFunction
    from .device_ir import GraphInfo
    from .inductor_lowering import CodegenState


# fx node meta key marking an access (a load, a store, an atomic, or an
# ``hl.inline_triton`` or ``hl.triton_kernel`` snippet) that must be ordered
# after other threads' pending accesses: its ``Barrier``, a CTA barrier
# before it, or for ``Barrier.DRAIN`` only a drain after a TMA store.
INTRA_LOOP_RAW_BARRIER_META = "_needs_debug_barrier_before"
# fx node meta key on a write: the loads it writes back (``_updates_own_read``),
# recorded before epilogue subtiling splits it (``record_own_reads``).
OWN_READS_META = "_writes_back_loads"
# fx node meta key on a write: its index before epilogue subtiling splits it
# into pieces at new indices (``record_own_reads``), whose union it names.
WRITE_INDEX_META = "_index_before_subtiling"
# The tensors whose pending writes that barrier flushes, or pending reads a
# write of the kernel may reach, over every walk (None for an access through
# device pointers, which may reach any memory).
INTRA_LOOP_RAW_BARRIER_FLUSHES_META = "_debug_barrier_before_flushes"
# fx node meta key on a store marked for ``Barrier.READS``: the earlier stores
# of other threads it may overwrite, which the barrier orders before it.
ORDERED_STORES_META = "_orders_stores"
# fx node meta key on a store such a barrier orders, or its own next copy
# (``Barrier.DRAIN``): the barrier drains it where it goes through TMA.  Not
# set by a full barrier, which drains every TMA store.
DRAINED_BY_BARRIER_META = "_drained_by_barrier"
# fx node meta key on a store its codegen lowered: whether through TMA
# (``DeviceFunction.note_tma_store``), which ``bar.sync`` does not complete.
TMA_STORE_META = "_tma_store"
# The storage of an access through device pointers (a stack tensor): no
# storage id, as ``id`` is never 0.
_THROUGH_POINTERS = 0


class Barrier(enum.Enum):
    """The CTA barrier before an access (``INTRA_LOOP_RAW_BARRIER_META``)
    or a loop (``ForLoopGraphInfo.needs_barrier_before``)."""

    # ``DeviceFunction.cta_barrier``: orders every earlier access.
    FULL = enum.auto()
    # ``DeviceFunction.cta_read_barrier``: orders the other threads' earlier
    # plain loads and plain stores, before a write that may overwrite what
    # they loaded or stored.
    READS = enum.auto()
    # ``DeviceFunction.async_store_drain`` before a TMA store alone, nothing
    # before a pointer store (``emit_intra_loop_store_drain``): it orders the
    # same store of the iteration before, issued by the same thread to the
    # same elements (``_IntraLoopRawBarrierMarker._restores_itself``).
    DRAIN = enum.auto()

    def statements(
        self, function: DeviceFunction, stores: Iterable[Node] = ()
    ) -> list[ast.stmt]:
        """The barrier's code, before a write that may overwrite ``stores``
        of other threads: a TMA store among them needs the drain too, as does
        one not lowered yet (an earlier iteration's, later in the body)
        unless it goes through pointers (``drained_store_through_pointers``)."""
        assert self is not Barrier.DRAIN, "emit_intra_loop_store_drain issues it"
        if self is Barrier.FULL:
            return function.cta_barrier()
        if any(
            store.meta.get(
                TMA_STORE_META, not drained_store_through_pointers(store, function)
            )
            for store in stores
        ):
            return [*function.async_store_drain(), *function.cta_read_barrier()]
        return function.cta_read_barrier()


def drained_store_through_pointers(store: Node, function: DeviceFunction) -> bool:
    """Whether ``store``, which a barrier drains where it goes through TMA
    (``DRAINED_BY_BARRIER_META``), goes through pointers instead: in a
    config that warp-specializes a loop with Triton, where a drain beside a
    warp-specialized TMA store is an InvalidConfig
    (``DeviceFunction.async_store_drain``).  The pointer store needs no
    drain, so such configs keep compiling."""
    from .compile_environment import CompileEnvironment

    return bool(store.meta.get(DRAINED_BY_BARRIER_META, False)) and any(
        CompileEnvironment.current().backend.range_warp_specializes(function.config)
    )


class _Atomic(NamedTuple):
    """The pending-write entry of an atomic ``op`` (an ``hl.atomic_*``) to
    ``storage`` (a ``_storage_ids`` id, or ``_THROUGH_POINTERS``), ``used``
    when the program uses the old value it returns."""

    storage: int
    op: object
    used: bool


class _Store(NamedTuple):
    """The pending-write entry of a write of ``storage`` by ``store`` (None
    for a snippet or another op with side effects), ``carried`` from an
    earlier iteration, whose index may name other elements: of each loop
    graph in ``loops`` whose region holds the store."""

    storage: int
    store: Node | None
    carried: bool = False
    loops: frozenset[int] = frozenset()


# A pending-write entry: a store's or an atomic's.
_PendingWrite = _Store | _Atomic


class _PendingRead(NamedTuple):
    """The pending entry of a read of ``storage`` no barrier has ordered yet,
    by ``load`` (None for a snippet or another op that may read it, and for
    a read an earlier iteration carried), by an ``atomic``, or by an
    ``opaque`` op (a snippet's, which may be anything)."""

    storage: int
    load: Node | None
    atomic: bool = False
    opaque: bool = False


# A pending entry: a write, or with ``reads`` a read.
_PendingEntry = _PendingWrite | _PendingRead


def _reaches(storage: int, storages: AbstractSet[int]) -> bool:
    """Whether an access of ``storage`` may meet one of ``storages``."""
    return (
        storage == _THROUGH_POINTERS
        or _THROUGH_POINTERS in storages
        or storage in storages
    )


def _overwrites(entry: _PendingEntry, storages: AbstractSet[int], write: Node) -> bool:
    """Whether ``write``, a write of ``storages``, may overwrite what the
    pending read ``entry`` has yet to read in another thread: unless it
    writes back what that load read (``_updates_own_read``), or elements
    apart from all it read (``_disjoint_accesses``), in the same body and
    iteration.  An atomic's read orders no later atomic: two atomics wait
    for each other as writes (``_waits_for``)."""
    from ..language import memory_ops

    return (
        isinstance(entry, _PendingRead)
        and _reaches(entry.storage, storages)
        and not (entry.atomic and write.target is not memory_ops.store)
        and not _updates_own_read(entry, write)
        and (entry.load is None or not _disjoint_accesses(entry.load, write))
    )


def _restores(entry: _PendingEntry, storages: AbstractSet[int], store: Node) -> bool:
    """Whether ``store``, a store of ``storages``, may overwrite what the
    pending store ``entry`` of another thread stored, which may then land
    after it: unless the two write disjoint elements, in the same body and
    iteration (``_disjoint_accesses``)."""
    return (
        isinstance(entry, _Store)
        and _reaches(entry.storage, storages)
        and (
            entry.store is None
            or entry.carried
            or not _disjoint_accesses(entry.store, store)
        )
    )


def _orders_with_reads(entry: _PendingEntry, store: bool) -> bool:
    """Whether ``Barrier.READS`` orders the pending ``entry`` before a write
    (a ``store``, or an atomic or a snippet): a plain load's read, or before a
    store another store's write.  An atomic's read may be a TMA reduction,
    which ``bar.sync`` does not wait for, and a snippet's may be anything."""
    if isinstance(entry, _PendingRead):
        return not (entry.atomic or entry.opaque)
    return store and isinstance(entry, _Store) and entry.store is not None


def _updates_own_read(read: _PendingRead, write: Node) -> bool:
    """Whether ``write`` writes back what ``read`` loaded, in the same body:
    the same non-overlapping tensor at the same one-to-one index
    (``_non_overlapping``, ``_one_to_one``), with an operand computed from
    the loaded value by pointwise ops alone (``_keeps_positions``), which
    nothing but the write uses (``x[t] = f(x[t])``).

    No two positions of the index then share an address, and each position
    is written from the value loaded at it.  Triton writes each element from
    one thread, after that thread's own load of it.  Another thread holding
    a copy of the element may load it late, but its copy reaches nothing but
    the write: directly, where Triton masks its write off, or through a
    cross-thread exchange, which synchronizes every thread before any write.
    """
    if OWN_READS_META in write.meta:
        return read.load in write.meta[OWN_READS_META]
    load = read.load
    if (
        load is None
        or load.graph is not write.graph
        or load.args[0] is not write.args[0]
        or not _non_overlapping(write.args[0])
        or not _same_index(load.args[1], write.args[1])
        or not _one_to_one_subscript(write.args[0], write.args[1])
    ):
        return False
    loaded = load.meta.get("val")
    operands = _ancestors([arg for arg in write.args[2:] if isinstance(arg, Node)])
    if not isinstance(loaded, torch.Tensor) or load not in operands:
        return False
    users = set(load.users)
    stack = list(users)
    while stack:
        node = stack.pop()
        if node is not write:
            if node not in operands or not _keeps_positions(node, loaded.shape):
                return False
            stack.extend(user for user in node.users if user not in users)
            users.update(node.users)
    return True


def record_own_reads(graph: torch.fx.Graph) -> None:
    """Record on each write of ``graph`` the loads it writes back
    (``OWN_READS_META``) and its index (``WRITE_INDEX_META``).  Epilogue
    subtiling splits a write into pieces at new indices, computed through
    views of its value, which show neither; each piece keeps the record."""
    from ..language import memory_ops

    loads = [node for node in graph.nodes if node.target is memory_ops.load]
    for node in graph.nodes:
        if memory_write_target(node) is not None:
            node.meta[OWN_READS_META] = frozenset(
                load for load in loads if _updates_own_read(_PendingRead(0, load), node)
            )
            node.meta[WRITE_INDEX_META] = node.args[1]


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


# The adds and subtractions an index may shift by (``_one_to_one``,
# ``_shifted_range``).
_ADDS = (operator.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
_SUBS = (operator.sub, torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar)


def _one_to_one_subscript(target: object, subscript: object) -> bool:
    """Whether ``subscript`` maps distinct positions to distinct elements of
    the tensor ``target`` names: each entry ``_one_to_one``, and no tile or
    tensor index on a dimension of size 1, which Triton broadcasts it
    over."""
    value = target.meta.get("val") if isinstance(target, Node) else None
    if not isinstance(value, torch.Tensor) or not isinstance(subscript, (list, tuple)):
        return False
    dim = 0
    for index in subscript:
        if index is None:
            continue
        if index is Ellipsis or not _one_to_one(index):
            return False
        if isinstance(index, Node) and (
            dim >= value.ndim or not statically_known_true(value.size(dim) != 1)
        ):
            return False
        dim += 1
    return True


def _one_to_one(index: object) -> bool:
    """Whether a subscript entry maps distinct positions to distinct indices:
    a slice, a tile or a scalar, a tile's ``index`` or a unit-step
    ``hl.arange`` of 32 or 64 bits, or one plus or minus a scalar.  A loaded
    index or ``hl.arange(n) % k`` may repeat one."""
    from ..language.tile_ops import tile_index

    if not isinstance(index, Node) or not isinstance(
        index.meta.get("val"), torch.Tensor
    ):
        return True
    if index.meta["val"].dtype not in (torch.int32, torch.int64):
        # A narrow index wraps: ``hl.arange(512, dtype=torch.uint8)``.
        return False
    if index.target is tile_index or (
        index.target is torch.ops.prims.iota.default and index.kwargs.get("step") == 1
    ):
        return True
    alpha = index.kwargs.get("alpha", 1)
    if (
        index.target not in (*_ADDS, *_SUBS)
        or len(index.args) != 2
        # ``alpha`` scales the second operand: ``alpha=0`` repeats one index.
        or type(alpha) is not int
        or alpha != 1
    ):
        return False
    tensors = [
        arg
        for arg in index.args
        if isinstance(arg, torch.Tensor)
        or (isinstance(arg, Node) and isinstance(arg.meta.get("val"), torch.Tensor))
    ]
    return (
        len(tensors) == 1 and isinstance(tensors[0], Node) and _one_to_one(tensors[0])
    )


def _same_index(first: object, second: object) -> bool:
    """Whether two subscripts of FX nodes and constants are the same one."""
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            starmap(_same_index, zip(first, second, strict=True))
        )
    if isinstance(first, Node) or isinstance(second, Node):
        return first is second
    return type(first) is type(second) and first == second


def _disjoint_accesses(first: Node, second: Node) -> bool:
    """Whether two accesses (a store and an earlier store or read) touch no
    element in common: pieces of one store epilogue subtiling split, or
    accesses of the same non-overlapping tensor whose indices on some
    dimension of more than one element take values that never meet
    (``_index_range``), such as the two halves of a row.  Each element one
    accesses is then another address than every element the other does,
    with the indices in bounds (an access out of bounds is the program's
    error).  The values are exact integers, which fixed-width
    index arithmetic may wrap: two that differ by less than ``2**31`` stay
    apart.

    Two stores at the same index are not exempt: each element is stored by
    one thread per store, but the two stores may lay their values out over
    the threads differently (a small store takes its value's layout)."""
    group = first.meta.get("epilogue_subtile_group_id")
    target = first.args[0]
    if (
        group is not None
        and group == second.meta.get("epilogue_subtile_group_id")
        # A group id is its store's name, unique only within one graph.
        and first.graph is second.graph
        and target is second.args[0]
    ):
        return True
    if target is not second.args[0] or not _non_overlapping(target):
        return False
    assert isinstance(target, Node)
    one, other = first.args[1], second.args[1]
    if (
        not isinstance(one, (list, tuple))
        or not isinstance(other, (list, tuple))
        or len(one) != len(other)
    ):
        return False
    sizes = iter(target.meta["val"].size())
    for index, other_index in zip(one, other, strict=True):
        if (index is None) != (other_index is None) or Ellipsis in (
            index,
            other_index,
        ):
            # The entries would name different dimensions from here on.
            return False
        if index is None:
            continue
        size = next(sizes, 1)
        ranges = _index_range(index), _index_range(other_index)
        # A dimension of one element takes every index to it.
        if (
            not statically_known_true(size != 1)
            or ranges[0] is None
            or ranges[1] is None
        ):
            continue
        (anchor, low, high), (other_anchor, other_low, other_high) = (
            ranges[0],
            ranges[1],
        )
        if (
            anchor is other_anchor
            and (
                statically_known_true(high < other_low)
                or statically_known_true(other_high < low)
            )
            and statically_known_true(
                torch.sym_max(high, other_high) - torch.sym_min(low, other_low) < 2**31
            )
        ):
            return True
    return False


def _index_range(
    index: object,
) -> tuple[Node | None, int | torch.SymInt, int | torch.SymInt] | None:
    """The values a subscript entry takes, as ``(anchor, low, high)``:
    ``anchor`` plus every integer from ``low`` to ``high``, where the anchor
    is None (zero), one scalar, or a tile's offset; None where unknown.  An
    integer, a slice with bounds, none negative, a scalar (a tile's
    ``begin``, say), a tile or its ``index``, a unit-step ``hl.arange``, a
    broadcast of one of those, and one plus or minus an integer qualify, as
    tensors of 32 or 64 bits.  A tile's positions stop short of its loop's
    extent, past which its mask drops them."""
    from ..language import memory_ops
    from ..language.tile_ops import tile_index
    from ..language.view_ops import subscript
    from .compile_environment import CompileEnvironment

    # A negative integer counts from the end of its dimension.
    if type(index) is int:
        return (None, index, index) if index >= 0 else None
    if isinstance(index, slice):
        start = 0 if index.start is None else index.start
        if (
            index.step not in (None, 1)
            or type(start) is not int
            or type(index.stop) is not int
            or start < 0
            or index.stop < 0
        ):
            return None
        return None, start, index.stop - 1
    if not isinstance(index, Node):
        return None
    value = index.meta.get("val")
    if isinstance(value, torch.SymInt):
        env = CompileEnvironment.current()
        block_id = env.get_block_id(value)
        # Not a tile's ``begin``, ``end`` or ``id``: those name its block too.
        if (
            block_id is not None
            and env.block_sizes[block_id].var._sympy_() == value._sympy_()
        ):
            # A tile: its offset plus every position of the block, or of
            # the loop's extent if shorter.
            size = env.block_sizes[block_id].size
            if isinstance(size, (int, torch.SymInt)):
                return index, 0, torch.sym_min(value, size) - 1
            return index, 0, value - 1
        return _shifted_range(index) or (index, 0, 0)
    if not isinstance(value, torch.Tensor) or value.dtype not in (
        torch.int32,
        torch.int64,
    ):
        # A narrow index wraps: ``hl.arange(64, dtype=torch.uint8) + 224``.
        return None
    if index.target is tile_index:
        tile = index.args[0]
        return _index_range(tile) if isinstance(tile, Node) else None
    if index.target is torch.ops.prims.iota.default:
        (length,) = index.args
        start = index.kwargs.get("start", 0)
        if (
            index.kwargs.get("step") != 1
            or type(start) is not int
            or not isinstance(length, (int, torch.SymInt))
        ):
            return None
        return None, start, start + length - 1
    if index.target is subscript or (
        # A subscript of an index the kernel computes, which reads no memory.
        index.target is memory_ops.load
        and isinstance(base := index.args[0], Node)
        and base.target in (tile_index, torch.ops.prims.iota.default, *_ADDS, *_SUBS)
    ):
        base, entries = index.args[:2]
        # Only new axes and whole slices: the same values, broadcast.
        if isinstance(entries, (list, tuple)) and all(
            entry is None or entry == slice(None) for entry in entries
        ):
            return _index_range(base)
        return None
    if value.numel() == 1:
        # One value: a loaded or computed scalar.
        return index, 0, 0
    return _shifted_range(index)


def _shifted_range(
    index: Node,
) -> tuple[Node | None, int | torch.SymInt, int | torch.SymInt] | None:
    """The ``_index_range`` of ``index`` as one plus or minus an integer, or
    None if it is not one."""
    if (
        index.target not in (*_ADDS, *_SUBS)
        or len(index.args) != 2
        or index.kwargs.get("alpha", 1) != 1
    ):
        return None
    left, right = index.args
    if type(left) is int and index.target in _ADDS:
        left, right = right, left
    if type(right) is not int or (inner := _index_range(left)) is None:
        return None
    anchor, low, high = inner
    shift = right if index.target in _ADDS else -right
    return anchor, low + shift, high + shift


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


def _waits_for(
    entry: _PendingEntry, storages: AbstractSet[int], atomic: object = None
) -> bool:
    """Whether an access of ``storages`` (with ``_THROUGH_POINTERS`` among
    them, of any storage), an ``atomic`` op whose old value goes unused if
    given, waits for the pending write ``entry``: one that may reach them,
    unless both are the same atomic op, that op commutes with itself
    (``_self_commuting_atomics``: a histogram's or split-K's adds) and the
    pending one's old value goes unused too; an exchange or compare-and-swap,
    two different ops, or an old value either uses, see another result in
    the other order.  A pending read is no write to wait for."""
    if isinstance(entry, _PendingRead):
        return False
    if (
        isinstance(entry, _Atomic)
        and atomic is not None
        and entry.op is atomic
        and not entry.used
        and atomic in _self_commuting_atomics()
    ):
        return False
    return _reaches(entry.storage, storages)


def emit_intra_loop_raw_barrier(
    state: CodegenState, accessed: torch.Tensor | None = None, *, full: bool = False
) -> None:
    """The barrier ``mark_intra_loop_raw_barriers`` asks for before the node
    ``state`` lowers: an access of ``accessed`` (a load, a store or an
    atomic), or an ``hl.inline_triton`` or ``hl.triton_kernel`` snippet, that
    may meet another thread's access earlier in its loop body or in an
    earlier iteration of its loop, or through device pointers or a snippet.
    ``full`` asks for the full barrier in any case."""
    if state.fx_node is not None and state.fx_node.meta.get(
        INTRA_LOOP_RAW_BARRIER_META
    ):
        from .compile_environment import CompileEnvironment
        from .cross_loop_codegen import _allocation
        from .cross_loop_codegen import _covering

        barrier = (
            Barrier.FULL if full else state.fx_node.meta[INTRA_LOOP_RAW_BARRIER_META]
        )
        if barrier is Barrier.DRAIN:
            # Only a TMA store needs it, after the store lowers
            # (``emit_intra_loop_store_drain``).
            return
        statements = barrier.statements(
            state.device_function, state.fx_node.meta.get(ORDERED_STORES_META, ())
        )
        ordered = (accessed, *state.fx_node.meta[INTRA_LOOP_RAW_BARRIER_FLUSHES_META])
        tensors = [tensor for tensor in ordered if tensor is not None]
        overlapped = set(CompileEnvironment.current().input_storage_aliases.values())
        # Its fence orders reads of what it flushes and the access alone: a
        # ring partition reading none of that drops the fence (and the loop
        # it would replay for it).  Not after a write through device pointers
        # (None), nor where an argument others overlap at run time stands for
        # their storage (``CompileEnvironment.input_storage_aliases``), whose
        # allocations differ.
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


def emit_intra_loop_store_drain(state: CodegenState) -> None:
    """The drain ``Barrier.DRAIN`` asks for before a store ``state`` lowered
    through TMA, which ``bar.sync`` does not order after the same store of
    the iteration before; a pointer store needs none."""
    node = state.fx_node
    if (
        node is not None
        and node.meta.get(INTRA_LOOP_RAW_BARRIER_META) is Barrier.DRAIN
        and node.meta.get(TMA_STORE_META)
    ):
        for statement in state.device_function.async_store_drain():
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


def graph_memory_access_storage_ids(
    graphs: Sequence[GraphInfo], graph_id: int, *, through_pointers: bool = False
) -> tuple[set[int], set[int]]:
    """Return storage identities read and written by a graph and its subgraphs.

    Besides loads, stores, atomics and the graphs of control flow, any other op
    with side effects reads and writes every tensor it is given
    (``opaque_access_target``).  With ``through_pointers``, an access through
    device pointers (a stack tensor) also adds an entry that may be any
    storage (``reads_meet_writes``).
    """
    from ..language import memory_ops
    from ..language.atomic_ops import ATOMIC_OPS

    reads: set[int] = set()
    writes: set[int] = set()

    def through(target: object) -> set[int]:
        return _pointer_storages(target) if through_pointers else set()

    for node in graphs[graph_id].graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is memory_ops.load or node.target in _ordered_flat_loads():
            reads.update(_storage_ids(graphs, graph_id, node.args[0]))
            reads.update(through(node.args[0]))
            continue
        if (target := memory_write_target(node)) is not None:
            storage_ids = _storage_ids(graphs, graph_id, target) | through(target)
            writes.update(storage_ids)
            if node.target in ATOMIC_OPS:
                reads.update(storage_ids)
            continue
        if (children := _child_graph_ids(graphs, node)) is not None:
            for child in children:
                child_reads, child_writes = graph_memory_access_storage_ids(
                    graphs, child, through_pointers=through_pointers
                )
                reads.update(child_reads)
                writes.update(child_writes)
            continue
        if (target := opaque_access_target(node)) is not None:
            storage_ids = _storage_ids(graphs, graph_id, target)
            reads.update(storage_ids)
            writes.update(storage_ids)
    return reads, writes


def graph_stores(graphs: Sequence[GraphInfo], graph_id: int) -> set[tuple[int, Node]]:
    """The stores (no atomic) of a graph and its subgraphs, each with every
    storage it writes (``_THROUGH_POINTERS`` through device pointers), as
    ``graph_memory_access_storage_ids`` walks them."""
    from ..language import memory_ops

    result: set[tuple[int, Node]] = set()
    for node in graphs[graph_id].graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is memory_ops.store:
            target = node.args[0]
            storages = _storage_ids(graphs, graph_id, target) | _pointer_storages(
                target
            )
            result.update((storage, node) for storage in storages)
        elif (children := _child_graph_ids(graphs, node)) is not None:
            for child in children:
                result.update(graph_stores(graphs, child))
    return result


def _accesses_through_pointers(graphs: Sequence[GraphInfo], graph_id: int) -> bool:
    """Whether the graph ``graph_id``, or one nested in it, accesses memory
    through device pointers (``reaches_through_pointers``)."""
    reads, writes = graph_memory_access_storage_ids(
        graphs, graph_id, through_pointers=True
    )
    return _THROUGH_POINTERS in reads or _THROUGH_POINTERS in writes


def reads_meet_writes(reads: AbstractSet[int], writes: AbstractSet[int]) -> bool:
    """Whether reads of ``reads`` may see writes of ``writes`` (storage ids of
    ``graph_memory_access_storage_ids``, where an access through device
    pointers may reach any storage)."""
    return bool(
        reads & writes
        or (reads and _THROUGH_POINTERS in writes)
        or (writes and _THROUGH_POINTERS in reads)
    )


def _iteration_private_storages(
    graphs: Sequence[GraphInfo], graph_id: int, block_ids: Sequence[int]
) -> set[int]:
    """The storages a loop's iterations access only in disjoint tiles: every
    load, store and atomic of one in the loop's region (its body with its
    nested control flow) goes through one same view, whose strides prove that
    distinct elements are distinct memory, either at one same index, which
    tells apart the iterations along every axis of the loop (``block_ids``;
    ``takes_a_tile``), or at indices that share, for each axis, an item
    telling its iterations apart at one position (``shares_a_tile``); and no
    other op with side effects reaches it.  A write to such a storage in one
    iteration never meets an access of another iteration.  An access through
    device pointers may reach any storage, so a region with one has none."""
    from ..language import memory_ops
    from ..language._tracing_ops import _new_var
    from ..language._tracing_ops import _phi
    from ..language.atomic_ops import ATOMIC_OPS
    from .compile_environment import CompileEnvironment
    from .device_ir import IfGraphInfo
    from .device_ir import NodeArgsGraphInfo
    from .device_ir import ReductionLoopGraphInfo
    from .host_function import HostFunction
    from .variable_origin import BlockSizeOrigin
    from .variable_origin import GridOrigin
    from .variable_origin import TileCountOrigin
    from .variable_origin import TileExtentOrigin

    origins = HostFunction.current().expr_to_origin
    # A rolled reduction loop maps a full slice of its dim to its chunk.
    rolled = isinstance(graphs[graph_id], ReductionLoopGraphInfo)
    keys: dict[int, set[tuple[object, ...]]] = {}
    # The keys an access proves take a tile of the loop.
    tiled: set[tuple[object, ...]] = set()
    # The loop block each key's index items tell apart, by position.
    positions: dict[tuple[object, ...], dict[int, int]] = {}
    shared: set[int] = set()
    parents: dict[int, int] = {}

    def resolve(node: object, graph_id: int) -> object:
        # A nested graph's placeholder stands for its parent's node, up to the
        # loop's own graph, and a ``_new_var`` for the value it copies, so
        # accesses in nested control flow compare equal.  Not a value a
        # nested loop carries, which a ``_phi`` merges with its result: it
        # changes between the nested loop's iterations.
        while isinstance(node, torch.fx.Node):
            if node.target is _new_var and isinstance(node.args[0], torch.fx.Node):
                node = node.args[0]
                continue
            if (
                node.op != "placeholder"
                or graph_id not in parents
                or not isinstance(info := graphs[graph_id], NodeArgsGraphInfo)
                or node.graph is not info.graph
            ):
                break
            outer = info.placeholder_to_outer_arg(node)
            if not isinstance(info, IfGraphInfo) and any(
                user.target is _phi and user.args[0] is outer for user in outer.users
            ):
                break
            node, graph_id = outer, parents[graph_id]
        return node

    def key_of(item: object) -> object:
        if isinstance(item, torch.fx.Node):
            value = item.meta.get("val")
            if isinstance(value, torch.SymInt):
                return ("symbol", str(value))
            return ("node", item)
        return ("literal", repr(item))

    def view_of(target: object) -> object:
        """The view's (storage, dtype, sizes, strides, offset), or None
        unless its strides map distinct elements to distinct memory in every
        call (``_non_overlapping``).  The dtype sets the element width the
        rest count in."""
        value = target.meta.get("val") if isinstance(target, torch.fx.Node) else None
        if not isinstance(value, torch.Tensor) or not _non_overlapping(target):
            return None
        # Overlapping arguments share their storage id
        # (``CompileEnvironment.input_storage_aliases``) but not their fake
        # storage, from which each view's offset counts.
        return tuple(
            str(x)
            for x in (
                id(value.untyped_storage()),
                value.dtype,
                *value.size(),
                *value.stride(),
                value.storage_offset(),
            )
        )

    def iteration_block(value: object) -> int | None:
        """The loop whose iterations ``value`` tells apart: a tile or a grid
        index of it, or a tile's begin, end or id (not its extent or count)."""
        if not isinstance(value, torch.SymInt):
            return None
        info = origins.get(value._sympy_())
        origin = None if info is None else info.origin
        if not isinstance(origin, (BlockSizeOrigin, GridOrigin)) or isinstance(
            origin, (TileExtentOrigin, TileCountOrigin)
        ):
            return None
        return origin.block_id

    def tells_apart(size: int | torch.SymInt) -> bool:
        """Whether a loop index reaches other elements in other iterations in
        a dimension of ``size``: indexing drops a size-one dimension's offset,
        so every iteration reaches the same elements there.  Code generation
        settles a dynamic size by its value at binding, under a guard keying
        the bound kernel (``CompileEnvironment.resolve_broadcast_dim``), or
        proves it covers the loop's extent, at which size one the loop runs
        once there; so does this check."""
        if isinstance(size, int):
            return size != 1
        return CompileEnvironment.current().size_hint(size) != 1

    def tile_positions(
        view: torch.Tensor, index: list[object] | tuple[object, ...]
    ) -> dict[int, int] | None:
        """The loop block each index item that ``tells_apart`` its block on
        its dimension of ``view`` moves with, by the item's position; None
        for an index with an ellipsis."""
        positions: dict[int, int] = {}
        sizes = iter(view.size())
        for position, entry in enumerate(index):
            item = entry.meta.get("val") if isinstance(entry, torch.fx.Node) else entry
            if item is None:
                continue  # a new axis
            if item is Ellipsis:
                return None
            size = next(sizes, None)
            if isinstance(item, torch.Tensor):
                # A tile's ``index``, shifted or not: its positions, offset
                # by the tile's.
                anchor = (_index_range(entry) or (None,))[0]
                item = None if anchor is None else anchor.meta.get("val")
            block = iteration_block(item)
            if size is not None and block is not None and tells_apart(size):
                positions[position] = block
        return positions

    def takes_a_tile(
        node: Node, view: torch.Tensor, index: list[object] | tuple[object, ...]
    ) -> bool:
        """Whether the index tells apart the loop's iterations along each of
        its axes: by an index item of a dimension of ``view`` that
        ``tells_apart`` its block, or, in a rolled reduction loop, for an
        access whose index is a box of full slices, integers and scalars, by
        the loaded or stored extent."""
        positions = tile_positions(view, index)
        if positions is None:
            return False
        blocks: set[int | None] = set(positions.values())
        value = (node.meta if node.target is memory_ops.load else {}).get("val")
        if node.target is not memory_ops.load and isinstance(node.args[2], Node):
            value = node.args[2].meta.get("val")
        if (
            rolled
            and all(
                isinstance(item, (int, torch.SymInt))
                or (isinstance(item, slice) and item == slice(None))
                for item in (
                    entry.meta.get("val") if isinstance(entry, Node) else entry
                    for entry in index
                )
            )
            and isinstance(value, torch.Tensor)
        ):
            blocks.update(iteration_block(size) for size in value.size())
        return blocks.issuperset(block_ids)

    def walk(graph_id: int) -> None:
        for node in graphs[graph_id].graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in (memory_ops.load, memory_ops.store, *ATOMIC_OPS):
                target, index = node.args[0], node.args[1]
                if "epilogue_subtile_group_id" in node.meta:
                    # The pieces of a write take the elements of its index
                    # together.
                    index = node.meta.get(WRITE_INDEX_META, index)
                storages = _storage_ids(graphs, graph_id, target)
                resolved = resolve(target, graph_id)
                view = view_of(resolved)
                if not isinstance(index, (list, tuple)) or view is None:
                    shared.update(storages)
                    continue
                index = [resolve(item, graph_id) for item in index]
                key = (view, *map(key_of, index))
                assert isinstance(resolved, torch.fx.Node)
                if takes_a_tile(node, resolved.meta["val"], index):
                    tiled.add(key)
                for storage in storages:
                    keys.setdefault(storage, set()).add(key)
                positions[key] = tile_positions(resolved.meta["val"], index) or {}
                continue
            if node.target in _ordered_flat_loads():
                # It may read any element, whichever iteration stored it.
                shared.update(_storage_ids(graphs, graph_id, node.args[0]))
                continue
            if (children := _child_graph_ids(graphs, node)) is not None:
                for child in children:
                    parents[child] = graph_id
                    walk(child)
                continue
            if (target := opaque_access_target(node)) is not None:
                shared.update(_storage_ids(graphs, graph_id, target))

    def shares_a_tile(accesses: set[tuple[object, ...]]) -> bool:
        """Whether the accesses (keys) go through one view, with their new
        axes at the same positions, and tell apart the loop's iterations
        along each of its axes by one same index item at one same position:
        a dimension each iteration reaches only in its own tile, whatever
        the other items reach."""
        ordered = list(accesses)
        if len({key[0] for key in ordered}) != 1 or len(set(map(len, ordered))) != 1:
            return False
        covered: set[int] = set()
        for position, items in enumerate(
            zip(*(key[1:] for key in ordered), strict=True)
        ):
            same = len(set(items)) == 1
            if not same and key_of(None) in items:
                return False  # dimensions out of step
            blocks = {positions[key].get(position) for key in ordered}
            if same and len(blocks) == 1:
                covered.update(block for block in blocks if block is not None)
        return covered.issuperset(block_ids)

    if _accesses_through_pointers(graphs, graph_id):
        return set()
    walk(graph_id)
    return {
        storage
        for storage, accesses in keys.items()
        if storage not in shared
        and ((len(accesses) == 1 and accesses <= tiled) or shares_a_tile(accesses))
    }


def mark_intra_loop_raw_barriers(
    graphs: list[GraphInfo],
    root_graph_ids: list[int],
    *,
    mark_in_divergent_control_flow: bool = True,
    atomics: bool = False,
    carries: bool = True,
    reads: bool = False,
) -> None:
    """Mark the accesses in device loops that may meet another thread's
    earlier one: a load or an atomic after a write, and with ``reads`` a
    write after a read or a store after a store.

    Within a single device-loop body, ``qkv[a] = v`` followed by ``qkv[b]`` is a
    read-after-write on the same storage. When the store and the load use
    different-shaped index tensors their Triton thread->element layouts differ, so
    an element written by one thread is read back by another with no
    synchronization in between -- a data race (observed corrupting ~0.8% of
    outputs on B200). Helion already inserts ``tl.debug_barrier()`` for the
    analogous hazard *between* sequential top-level loops; this extends the same
    guarantee to a store->load *within* one loop: in one body, and with
    ``carries`` (``Backend.intra_loop_barriers_carry_iterations``) from a store
    late in one iteration to a load early in the next (``iterate``), except for
    storages the loop's iterations access in disjoint tiles
    (``_iteration_private_storages``).  Without ``carries`` a loop that
    accesses memory through device pointers still carries: such an access may
    reach any storage, which the backend's own barrier pass does not see.

    We mark the access's FX node with its ``Barrier``; its codegen emits the
    barrier before it (``emit_intra_loop_raw_barrier``; on CuTe a
    ``cute.arch.sync_threads()``).  A full barrier completes every pending
    access; ``Barrier.READS`` completes plain loads and the stores it orders
    (``_order``), and the rest stay pending.  Storage identity comes from the
    fake tensor's underlying storage, so distinct FX nodes for aliases and
    views compare equal.  The walk follows Helion control-flow subgraphs and
    merges pending accesses at joins.

    ``mark_in_divergent_control_flow=False`` skips marking accesses (loads,
    stores, atomics, snippets) inside ``hl.if`` / while bodies, leaving the
    hazard pending for the next uniform access: a CuTe SIMT branch condition can vary per thread, and a
    convergent CTA barrier inside a divergent branch deadlocks.  Loop bodies stay
    markable — Helion device-loop trip structure is uniform across the CTA.

    With ``atomics`` (``Backend.intra_loop_barriers_order_atomics``), an
    atomic writes its target, as a store does (``memory_write_target``).  Any
    other op with side effects writes every input it gets that may name the
    kernel's memory (``opaque_access_target``), and an ``hl.inline_triton``
    or ``hl.triton_kernel`` snippet, which may read them too, is marked like a
    load when one holds a pending write: its Triton codegen emits the barrier
    before the snippet.  An atomic is marked too when another thread's pending
    store, or a pending atomic it does not commute with, may reach its target
    (it reads the target first, ``_waits_for``); after the same
    self-commuting atomic (an add after an add) it needs none unless either
    uses its old value.

    With ``reads`` (``Backend.intra_loop_barriers_order_reads``), the walk
    also tracks the reads no barrier has ordered yet, and marks a store (with
    ``atomics`` an atomic, and a snippet) that writes storage one of them
    read: another thread may not have read it yet (a write-after-read).  An
    update of what a load of the same body and iteration read, at its index
    and computed from it (``x[t] = f(x[t])``), needs none (``_overwrites``).
    """
    for info in graphs:
        for node in info.graph.nodes:
            # A pass generating code again lowers its stores again.
            node.meta.pop(TMA_STORE_META, None)
    written: set[int] = set()
    for graph_id in root_graph_ids:
        written |= graph_memory_access_storage_ids(
            graphs, graph_id, through_pointers=True
        )[1]
    marker = _IntraLoopRawBarrierMarker(
        graphs,
        mark_in_divergent_control_flow=mark_in_divergent_control_flow,
        atomics=atomics,
        carries=carries,
        reads=reads,
        written=written,
    )
    for graph_id in root_graph_ids:
        marker.run_graph(graph_id, set())


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
    barrier: one re-reading storage it stores, and with ``reads`` one
    overwriting what it read or stored.

    A loop's region is its body with every loop nested in it.  The walk of
    ``mark_intra_loop_raw_barriers`` (with ``atomics``, ``carries`` and
    ``reads`` as there) runs over each loop's region (grid loops included)
    from an empty pending set, so only an access issued inside the region
    arms another inside it; one before the loop does not count.

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
    private: dict[int, set[int]] = {}
    for graph_id, block_ids in regions:
        marker = _IntraLoopRawBarrierMarker(
            graphs,
            mark=False,
            atomics=atomics,
            carries=carries,
            reads=reads,
            private=private,
        )
        if isinstance(graphs[graph_id], ForLoopGraphInfo):
            marker.iterate(
                [graph_id],
                set(),
                divergent=False,
                private=marker._private(graph_id, block_ids),
            )
        else:
            marker.run_graph(graph_id, set())
        if marker.barriered:
            result.update(block_ids)
    return result


def _carried(
    entry: _PendingEntry, loop: int, region: AbstractSet[torch.fx.Graph]
) -> _PendingEntry:
    """A pending entry as the next iteration of the loop graph ``loop``,
    whose region is the graphs ``region``, sees it: a read's load and a
    store ran in another iteration, whose index may name other elements
    (``_overwrites``, ``_restores``)."""
    if isinstance(entry, _PendingRead):
        return _PendingRead(entry.storage, None, entry.atomic, entry.opaque)
    if isinstance(entry, _Store):
        inside = entry.store is not None and entry.store.graph in region
        return _Store(
            entry.storage,
            entry.store,
            carried=True,
            loops=entry.loops | {loop} if inside else entry.loops,
        )
    return entry


def _order(
    pending: set[_PendingEntry],
    barrier: Barrier,
    stores: AbstractSet[Node] = frozenset(),
) -> None:
    """Drop from ``pending`` what ``barrier`` orders, before a write that may
    overwrite ``stores`` (``Barrier.statements``).  A read barrier completes
    plain loads and those stores: the other writes (maybe TMA stores, which
    only a drain completes), and the reads of atomics (maybe TMA reductions,
    which ``bar.sync`` does not wait for) and of snippets, stay pending.  A
    drain completes the earlier iterations' copies of those stores alone."""
    if barrier is Barrier.DRAIN:
        pending.difference_update(
            [
                entry
                for entry in pending
                if isinstance(entry, _Store) and entry.carried and entry.store in stores
            ]
        )
    elif barrier is Barrier.READS:
        pending.difference_update(
            [
                entry
                for entry in pending
                if _orders_with_reads(entry, store=False)
                or (isinstance(entry, _Store) and entry.store in stores)
            ]
        )
    else:
        pending.clear()


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
        private: dict[int, set[int]] | None = None,
        written: AbstractSet[int] | None = None,
    ) -> None:
        self.graphs = graphs
        # The storages a write of the kernel may reach (None: any), whose
        # reads a barrier's fence covers (``_flush``).
        self.written = written
        self.mark_in_divergent_control_flow = mark_in_divergent_control_flow
        # Whether an atomic arms a later load like a store
        # (``Backend.intra_loop_barriers_order_atomics``).
        self.atomics = atomics
        # Whether a write after another thread's read of its storage takes a
        # barrier (``Backend.intra_loop_barriers_order_reads``).
        self.reads = reads
        # Whether a write of one loop iteration arms a load of the next
        # (``Backend.intra_loop_barriers_carry_iterations``).
        self.carries = carries
        # ``mark=False`` leaves the graphs and earlier marks untouched and
        # collects in ``barriered`` the accesses that would take a barrier:
        # empty exactly when a marking walk would mark none.
        self.mark = mark
        self.barriered: set[Node] = set()
        # ``_iteration_private_storages`` of each loop graph walked (a cache
        # the caller may share between markers).
        self.private: dict[int, set[int]] = {} if private is None else private
        # A tensor of each storage the walk met, by ``_storage_ids`` id.
        self.tensors: dict[int, torch.Tensor] = {}
        # ``_accesses_through_pointers`` of each loop graph ``_carries`` asked.
        self.through_pointers: dict[int, bool] = {}

    def _barrier_before(
        self,
        node: Node,
        waits: Callable[[_PendingEntry], bool],
        pending: set[_PendingEntry],
        *,
        divergent: bool,
        store: bool = False,
    ) -> None:
        """Mark ``node`` for the barrier before it when it waits for a pending
        access (``waits``: a ``_waits_for``, or an ``_overwrites`` or a
        ``_restores`` too for a write), and clear the pending accesses the
        barrier orders (recording what the writes write).  Where
        ``Barrier.READS`` orders each one (``_orders_with_reads``, ``store``
        for a store), it takes that barrier."""
        waited = [entry for entry in pending if waits(entry)]
        marked = node.meta.get(INTRA_LOOP_RAW_BARRIER_META) if self.mark else None
        if not waited:
            if marked:
                self._flush(node, pending)
            return
        if not marked and divergent and not self.mark_in_divergent_control_flow:
            # Cannot place a convergent barrier here; leave the hazard
            # pending so a later uniform read re-arms it.
            return
        stores = frozenset(
            entry.store
            for entry in waited
            if isinstance(entry, _Store) and entry.store is not None
        )
        if marked in (None, Barrier.DRAIN) and all(
            self._restores_itself(entry, node) for entry in waited
        ):
            # No other thread stored what this store overwrites; nothing
            # pins the loop out of warp specialization either.
            barrier = Barrier.DRAIN
        else:
            reads = all(_orders_with_reads(entry, store) for entry in waited)
            barrier = (
                Barrier.READS if reads and marked is not Barrier.FULL else Barrier.FULL
            )
            self.barriered.add(node)
        if self.mark:
            node.meta[INTRA_LOOP_RAW_BARRIER_META] = barrier
            if stores:
                node.meta[ORDERED_STORES_META] = stores | node.meta.get(
                    ORDERED_STORES_META, frozenset()
                )
                if barrier is not Barrier.FULL:
                    # A full barrier drains every TMA store, whichever
                    # these went through.
                    for ordered in stores:
                        ordered.meta[DRAINED_BY_BARRIER_META] = True
            self._flush(node, pending)
        else:
            _order(pending, barrier, stores)

    def _region(self, graph_ids: Sequence[int]) -> set[torch.fx.Graph]:
        """The graphs of a loop whose iteration walks ``graph_ids``, with
        every graph nested in them."""
        region: set[torch.fx.Graph] = set()
        stack = list(graph_ids)
        while stack:
            graph = self.graphs[stack.pop()].graph
            if graph not in region:
                region.add(graph)
                for node in graph.nodes:
                    stack.extend(_child_graph_ids(self.graphs, node) or ())
        return region

    def _restores_itself(self, entry: _PendingEntry, store: Node) -> bool:
        """Whether the pending ``entry`` is ``store`` itself, carried from an
        earlier iteration of a loop whose body holds it, at an index that
        loop does not change: no load, no value the loop carries, and no
        iteration variable of the loop reach the index; and one position of
        the index per element of a tensor whose elements have addresses of
        their own (``_one_to_one_subscript``, ``_non_overlapping``).  Each
        iteration then stores the same elements laid out the same way, each
        element from the same thread (and a TMA store from the same elected
        thread), which orders the two in program order (a TMA store after a
        drain)."""
        if not (
            isinstance(entry, _Store)
            and entry.carried
            and entry.store is store
            and len(entry.loops) == 1
        ):
            return False
        from ..language import memory_ops
        from ..language._tracing_ops import _phi
        from .compile_environment import CompileEnvironment
        from .device_ir import ForLoopGraphInfo

        (loop_id,) = entry.loops
        loop = self.graphs[loop_id]
        index = store.args[1]
        if (
            not isinstance(loop, ForLoopGraphInfo)
            or store.graph is not loop.graph
            or not _non_overlapping(store.args[0])
            or not _one_to_one_subscript(store.args[0], index)
        ):
            return False
        assert isinstance(index, (list, tuple))
        env = CompileEnvironment.current()
        stack: list[Node] = []
        map_arg(index, stack.append)
        seen: set[Node] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            value = node.meta.get("val")
            if isinstance(value, torch.SymInt):
                # A tile's symbol stands for its positions, which its loop
                # moves.
                if env.get_block_id(value) in loop.block_ids:
                    return False
                if node.target is torch.ops.aten.sym_size.int:
                    continue  # a size, the same in every iteration
            if (
                node.target is memory_ops.load
                or node.target in _ordered_flat_loads()
                or memory_write_target(node) is not None
            ):
                return False
            if node.op == "placeholder":
                outer = loop.placeholder_to_outer_arg(node)
                if any(
                    user.target is _phi and user.args[0] is outer
                    for user in outer.users
                ):
                    return False
            stack.extend(node.all_input_nodes)
        return True

    def _private(self, graph_id: int, block_ids: Sequence[int]) -> AbstractSet[int]:
        if not self.carries:
            # Only a loop reaching through device pointers carries, and such
            # a loop has none (``_iteration_private_storages``).
            return frozenset()
        if graph_id not in self.private:
            self.private[graph_id] = _iteration_private_storages(
                self.graphs, graph_id, block_ids
            )
        return self.private[graph_id]

    def _flush(self, node: Node, pending: set[_PendingEntry]) -> None:
        """Drop from ``pending`` what the barrier before ``node`` orders, and
        record a tensor of each storage it drops a write of, or a read of
        that a write of the kernel may reach: a later write that takes no
        barrier of its own may rely on it."""
        flushed = node.meta.setdefault(INTRA_LOOP_RAW_BARRIER_FLUSHES_META, ())
        before = set(pending)
        _order(
            pending,
            node.meta[INTRA_LOOP_RAW_BARRIER_META],
            node.meta.get(ORDERED_STORES_META, frozenset()),
        )
        storages = {
            _THROUGH_POINTERS if tensor is None else _storage_id(tensor)
            for tensor in flushed
        }
        node.meta[INTRA_LOOP_RAW_BARRIER_FLUSHES_META] = (
            *flushed,
            *(
                None if storage == _THROUGH_POINTERS else self.tensors[storage]
                for storage in {
                    entry.storage
                    for entry in before - pending
                    if not isinstance(entry, _PendingRead)
                    or self.written is None
                    or _reaches(entry.storage, self.written)
                }
                - storages
            ),
        )

    def _accessed(self, graph_id: int, target: object) -> set[int]:
        """The storages a load, store or atomic of ``target`` reaches
        (``_storages``, ``_pointer_storages``)."""
        return self._storages(graph_id, target) | _pointer_storages(target)

    def _storages(self, graph_id: int, obj: object) -> set[int]:
        """``_storage_ids``, remembering a tensor of each storage."""
        tensors = _storage_tensors(self.graphs, graph_id, obj)
        self.tensors.update(tensors)
        return set(tensors)

    def _waits(
        self,
        node: Node,
        storages: set[int],
        *,
        reads_first: bool,
        atomic: object = None,
    ) -> Callable[[_PendingEntry], bool]:
        """What a write ``node`` of ``storages`` waits for: where it
        ``reads_first`` (an atomic, a snippet), a pending write it would read
        past (``_waits_for`` with ``atomic``), and with ``reads`` a pending
        read it may overwrite (``_overwrites``), and for a store another
        thread's pending store it may overwrite (``_restores``)."""
        from ..language import memory_ops

        store = node.target is memory_ops.store

        def waits(entry: _PendingEntry) -> bool:
            return (reads_first and _waits_for(entry, storages, atomic)) or (
                self.reads
                and (
                    _overwrites(entry, storages, node)
                    or (store and _restores(entry, storages, node))
                )
            )

        return waits

    def _record_reads(
        self,
        pending: set[_PendingEntry],
        storages: set[int],
        load: Node | None,
        *,
        atomic: bool = False,
        opaque: bool = False,
    ) -> None:
        if self.reads:
            pending.update(
                _PendingRead(storage, load, atomic, opaque) for storage in storages
            )

    def run_graph(
        self, graph_id: int, start: set[_PendingEntry], *, divergent: bool = False
    ) -> set[_PendingEntry]:
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

        pending = set(start)
        for node in self.graphs[graph_id].graph.nodes:
            if node.op != "call_function":
                continue
            if (target := memory_write_target(node)) is not None:
                # A store, or an atomic where this barrier is the one that
                # orders it: what it wrote may not be visible to the other
                # warps yet.
                if self.atomics or node.target is memory_ops.store:
                    storages = self._accessed(graph_id, target)
                    if node.target is memory_ops.store:
                        # It may overwrite what another thread has yet to read,
                        # or has stored.
                        self._barrier_before(
                            node,
                            self._waits(node, storages, reads_first=False),
                            pending,
                            divergent=divergent,
                            store=True,
                        )
                        pending.update(_Store(storage, node) for storage in storages)
                        continue
                    # An atomic first reads its target, where another thread's
                    # pending store, or atomic it does not commute with, may
                    # land after it; where either's old value is used, another
                    # order returns another value.
                    op = node.target
                    self._barrier_before(
                        node,
                        self._waits(
                            node,
                            storages,
                            reads_first=True,
                            atomic=None if node.users else op,
                        ),
                        pending,
                        divergent=divergent,
                    )
                    pending.update(
                        _Atomic(storage, op, bool(node.users)) for storage in storages
                    )
                    # It reads its target first, which a later store of
                    # another thread must not overwrite before it.
                    self._record_reads(pending, storages, None, atomic=True)
                continue
            if node.target is memory_ops.load or node.target in _ordered_flat_loads():
                # Device pointers may name any pending write's storage.
                storages = self._accessed(graph_id, node.args[0])
                self._barrier_before(
                    node,
                    functools.partial(_waits_for, storages=storages),
                    pending,
                    divergent=divergent,
                )
                # A flat-offset load's index is no subscript: no later write
                # is exempt as disjoint from it.
                self._record_reads(
                    pending, storages, node if node.target is memory_ops.load else None
                )
                continue
            if (target := opaque_access_target(node)) is not None:
                storages = self._storages(graph_id, target)
                if node.target in (inline_triton, triton_kernel):
                    # It may read what is pending, and overwrite what another
                    # thread has yet to read.
                    self._barrier_before(
                        node,
                        self._waits(node, storages, reads_first=True),
                        pending,
                        divergent=divergent,
                    )
                pending.update(_Store(storage, None) for storage in storages)
                self._record_reads(pending, storages, None, opaque=True)
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
                loop_input = set(pending)
                if loop_info.needs_barrier_before is not None:
                    _order(
                        loop_input,
                        loop_info.needs_barrier_before,
                        loop_info.barrier_orders_stores,
                    )
                loop_pending = self.iterate(
                    [loop_graph_id],
                    loop_input,
                    divergent=divergent,
                    private=self._private(loop_graph_id, loop_info.block_ids),
                )
                # Zero iterations leave the state after the pre-loop barrier.
                pending = loop_input | loop_pending
                continue
            if node.target is _while_loop:
                body_graph_id = node.args[1]
                assert isinstance(body_graph_id, int)
                body_info = self.graphs[body_graph_id]
                assert isinstance(body_info, WhileLoopGraphInfo)
                cond_graph_id = body_info.cond_graph_id
                if self._carries([cond_graph_id, body_graph_id]):
                    # One iteration runs the condition, then the body; the
                    # loop exits after a condition, run from the entry or
                    # after a body.
                    carried = self.iterate(
                        [cond_graph_id, body_graph_id], pending, divergent=True
                    )
                    pending = self.run_graph(
                        cond_graph_id, pending | carried, divergent=True
                    )
                else:
                    # One pass: the condition executes at least once; the
                    # body may not execute.
                    condition_pending = self.run_graph(
                        cond_graph_id, pending, divergent=True
                    )
                    body_pending = self.run_graph(
                        body_graph_id, condition_pending, divergent=True
                    )
                    pending = condition_pending | body_pending
        return pending

    def iterate(
        self,
        graph_ids: Sequence[int],
        entry: set[_PendingEntry],
        *,
        divergent: bool,
        private: AbstractSet[int] = frozenset(),
    ) -> set[_PendingEntry]:
        """The accesses a loop leaves pending, where one iteration walks
        ``graph_ids`` in order and ``entry`` is pending before the first.

        Where the loop carries (``_carries``), an access late in one
        iteration reaches one early in the next, so the accesses that may be
        pending at an iteration's start (the entry's, or an earlier
        iteration's of a storage not in ``private``,
        ``_iteration_private_storages``) are collected first, walking the
        iteration without marking until that set stops growing; one walk from
        that set then marks every access an earlier iteration's reaches (a
        mark made from a smaller set could later be redundant)."""
        start = set(entry)
        if self._carries(graph_ids):
            mark, self.mark = self.mark, False
            region = self._region(graph_ids)
            try:
                while True:
                    pending = self._walk(graph_ids, start, divergent)
                    carried = {
                        _carried(entry, graph_ids[-1], region)
                        for entry in pending
                        if entry.storage not in private
                    }
                    if not carried - start:
                        break
                    start |= carried
            finally:
                self.mark = mark
            if not mark:
                # The last walk started from that set and marks nothing.
                return pending
        return self._walk(graph_ids, start, divergent)

    def _carries(self, graph_ids: Sequence[int]) -> bool:
        """Whether a write of one iteration of the loop whose iteration walks
        ``graph_ids`` may arm a load of the next: with ``carries``, and else
        where the loop accesses memory through device pointers, which the
        backend's barrier pass does not take to reach other storage."""
        if self.carries:
            return True
        for graph_id in graph_ids:
            if graph_id not in self.through_pointers:
                self.through_pointers[graph_id] = _accesses_through_pointers(
                    self.graphs, graph_id
                )
        return any(self.through_pointers[graph_id] for graph_id in graph_ids)

    def _walk(
        self, graph_ids: Sequence[int], start: set[_PendingEntry], divergent: bool
    ) -> set[_PendingEntry]:
        """The accesses pending after one iteration that ``start`` begins."""
        pending = start
        for graph_id in graph_ids:
            pending = self.run_graph(graph_id, pending, divergent=divergent)
        return pending
