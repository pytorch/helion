"""DSMEM transports (``cross_loop_dsmem``).

A split and combine pair of roots, the first writing the partials of a
reduction indexed by a split tile and the second summing them over that
dimension, passes the partials through global memory and orders them with a
readiness counter (or a root barrier).  A dsmem transport launches the kernel
in clusters of ``DsmemTransport.size`` CTAs instead, runs the producer tasks of
one key (one consumer task) on the CTAs of one cluster and the consumer task on
its rank 0: each producer task sends its slab into rank 0's static shared
memory (``st.async``, whose bytes complete rank 0's mbarrier) and the consumer
task waits for the barrier's phase and reads the slabs from shared memory.  The
dependency's transport is then "cluster": the scheduler orders nothing for it.
The global stores stay, ordered only for the allocation's other loads.  The
split count the source writes (the slab's partials) is the cluster size; the
entry only picks the transport.

A consumer root may instead run a task of each key on every rank, its tasks
the producer's one to one (``DsmemTransport.per_rank``: an axis of the
consumer root the load does not index, of the size's tasks, is its split
axis): each producer task then sends its slab part to every rank of its
cluster, and each consumer task reads the whole slab from its own CTA's
shared memory (an all-gather, e.g. a row's chunks that each chunk's task
needs whole).

A slab of one producer task's tile (``DsmemTransport.local``, size 1) stays on
its CTA: the consumer task that reads it runs on the producer task's CTA (its
root takes the producer root's lanes, or with a per-rank transport of the
same roots its split lanes), and the producer task sends the tile into its own
shared memory.  (A launch without clusters faults on such sends: these slabs
need a cluster transport beside them.)

A load that reads the whole allocation in every task of a root of more tasks
than one cluster runs (``DsmemTransport.replicated``) has its slab in every
CTA: every cluster runs every producer task (rank ``r`` of the launch's
cluster size its tasks ``r``, ``r + size``, ...), each sending its tile to
every rank, and only cluster 0's tasks arrive on the root's counters (the
other clusters' stores write the same values: no task of the producer root
stores an element twice).

Candidates (``TileDependencyGraph.dsmem_candidates``: a load and its
allocation's stores, every write of it in one earlier root) do not depend on
the config.  Every size a rule reads takes this config's value
(``CompileEnvironment.config_value``: block sizes, tunables and host helpers
over them), never a size hint.  ``ConfigSpec.normalize`` turns off the entries
whose slab facts (``dsmem_slab_facts``) break a rule of the config alone (the
cluster size, the slabs' bytes, one cluster size for every entry), and
codegen checks the rest; an enabled entry that fails regenerates the kernel
with it off (``DsmemFallback``, logged), and a reason of the whole config
(the pipeline, packet dispatch, continuations, the cluster occupancy query)
turns off every entry at once:

- C1 (the slab, ``dsmem_transports``): the stores and the load run once a
  task (in their root's own graph, outside any loop or branch) and at most 8
  CTAs make a cluster (a local slab's root reads every axis of more than one
  task); the load and the stores index whole dimensions (full slices),
  tiles of a root axis and single elements (the load's are its key); the ranges
  the load reads are strided runs that do not overlap, and every range a
  store writes lies in one of them.  The slab is the load's region in row-major
  run order, at most ``MAX_CLUSTER_SLAB`` bytes for every entry together.
  32-bit elements, shapes and strides static under the config, no masks.
- C2 (the schedule, ``check_dispatch``): task lanes (``cross_loop_pipeline``
  static or dynamic, no packets); both roots resident and unguarded, without
  continuations, claimed tasks, reordered tasks or L2 grouping; the producer
  root a single wave of ``size`` times the consumer's tasks (per-rank and
  local: as many), with a split axis of ``size`` tasks: the producer tasks of
  key ``k`` (``k``'s coordinates on the other axes, in their task order) store
  exactly what consumer task ``k`` (per-rank: each of key ``k``'s, one a rank
  on the consumer's split axis) reads, element by element (no element twice),
  whatever the split axis's place in either task order.  A local slab's
  consumer task reads what the producer task its CTA runs writes, under the
  per-rank split lanes of the same roots when they have any (whose roots run
  one task a rank), else one to one.  A replicated slab's producer tasks
  write it exactly once, which every consumer task reads whole; its producer
  root (resident, unguarded, as above) stores no element twice in a task
  (each store once a task, outside loops; two stores of one allocation
  apart, as the barrier walk proves them), makes no atomic or cross-rank
  access, reads nothing it stores and no later root writes after its
  accesses (``check_replicated``); its consumer root may run any way.  A
  launch needs a cluster transport beside local and replicated slabs.
- C3 (codegen, ``cluster_layout``): a tile spreads over the threads of the
  partition that moves it with every element on as many threads (a tile of
  fewer elements than threads on several, of which one sends it), so the
  bytes a barrier expects arrive once.

Dispatch (``cross_loop_codegen``): a producer worker takes the key its
cluster's rank 0 would (the lane rank 0 takes, divided by ``size``; a
bijection for every rotation of the lanes) and the split axis coordinate of
its rank (``split_task``); rank 0 of a cluster takes the consumer task of the
key its cluster runs (per-rank: every rank, its split coordinate's).  The
launch runs as many workers as clusters of ``size`` keep resident
(``dsmem_worker_count``), a multiple of ``size``.

Shared memory (``gluon_ops.CLUSTER_SMEM``, declared and initialized at kernel
entry, then a CTA barrier or the ring's ``warp_specialize`` orders the init
before every thread's waits): the ``i``-th enabled entry's barrier is at byte
``8 + 8 i`` (one arrival at entry plus the slab's bytes complete phase 0, once
a launch), its slab at ``DsmemTransport.buffer``.  Once its barriers are
armed, thread 0 raises a flag at bytes 0 to 7 to the launch's value (a hash of
its epoch, launch state and cluster id with the top bit set: never 0, and
matching another kernel's leftover data only by chance;
``gluon_ops.cluster_flag``, assigned to ``CLUSTER_FLAG``); a producer task
polls the flags of the ranks it sends to before sending (``CLUSTER_READY``),
and every CTA clears its flag at the kernel's end (``cluster_exit``).  The
hardware cluster barrier would need every thread, which a warp-specialized
kernel's padding warps never are.
"""

from __future__ import annotations

import dataclasses
from itertools import chain
from itertools import combinations
from itertools import starmap
import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

from .. import exc
from ..autotuner.config_spec import DsmemLayout
from ..autotuner.config_spec import DsmemSlabFacts
from ..autotuner.config_spec import dsmem_replicated
from ..runtime.triton.launcher import cluster_workers
from ..runtime.triton.launcher import get_cluster_capacity
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .compile_environment import FixedBlockSizeSource
from .compile_environment import LoopSpecBlockSizeSource
from .host_function import HostFunction
from .loop_dependency_checker import accesses_apart
from .tile_dependency import TileDependencyKind
from .tile_dependency import _access_interval_expression
from .tile_dependency import coordinate_axis_symbol

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .device_function import DeviceFunction
    from .tile_dependency import CoordinateDomain
    from .tile_dependency import IntegerExpression
    from .tile_dependency import TaskAxis
    from .tile_dependency import TaskFamily
    from .tile_dependency import TileAccess
    from .tile_dependency import TileDependencyGraph

# The token a producer task holds once the CTAs it sends to have armed their
# barriers (their flags, polled at the task's start: ``gluon_ops.cluster_ready``),
# and the statement that raises this CTA's flag (``gluon_ops.cluster_flag``).
CLUSTER_READY = "helion_cluster_ready"
CLUSTER_FLAG = "helion_cluster_flag"
# The C2 check enumerates every slab element of every consumer task.
_MAX_CHECKED_ELEMENTS = 1 << 22


class DsmemFallback(exc.Base):
    """Enabled dsmem transports failed a rule only codegen checks: regenerate
    the kernel with their ``cross_loop_dsmem`` entries off (every enabled one
    for a reason of the whole config)."""

    def __init__(self, indices: Iterable[int], reason: str) -> None:
        super().__init__(reason)
        self.indices = tuple(sorted(indices))

    def report(self) -> str:
        return f"cross_loop_dsmem fallback: {self}"


@dataclasses.dataclass(frozen=True)
class DsmemTransport:
    """A load carried from its allocation's stores (one root's) to its root
    in launch clusters of ``size`` CTAs: one slab a key, from byte ``buffer``
    of the shared memory of its cluster's rank 0, of every rank (per-rank and
    replicated) or of the producer task's own CTA (local), completing the
    barrier at byte ``barrier`` there."""

    index: int  # the config entry
    name: str
    stores: tuple[TileAccess, ...]
    load: TileAccess
    slab: _Slab
    buffer: int
    barrier: int
    # A per-rank consumer's split axis (its root's axis the load does not
    # index): every rank runs a consumer task of the key and receives the whole
    # slab.  None: the key's consumer task runs on rank 0.
    consumer_axis: int | None = None
    # A replicated slab's cluster size (the launch's): every cluster runs every
    # producer task.  None: not replicated.
    replicas: int | None = None

    @property
    def producer_root(self) -> int:
        return self.stores[0].root

    @property
    def consumer_root(self) -> int:
        return self.load.root

    @property
    def per_rank(self) -> bool:
        return self.consumer_axis is not None

    @property
    def replicated(self) -> bool:
        return self.replicas is not None

    @property
    def local(self) -> bool:
        """A slab of one producer task's tile, which stays on its CTA."""
        return self.slab.size == 1 and not self.replicated

    @property
    def size(self) -> int:
        """The cluster size (a replicated slab's producer tasks are
        ``slab.size``)."""
        return self.slab.size if self.replicas is None else self.replicas

    @property
    def numel(self) -> int:
        return self.slab.numel

    @property
    def nbytes(self) -> int:
        return self.slab.nbytes

    @property
    def vec(self) -> int:
        """The most 32-bit elements a thread may move at once: 4 or 2 where
        the slab keeps 16 or 8 bytes aligned, else 1."""
        return self.slab.vec

    def fallback(self, rule: str) -> DsmemFallback:
        return DsmemFallback((self.index,), f"{self.name}: {rule}")


def config_fallback(
    transports: Mapping[int, DsmemTransport], rule: str
) -> DsmemFallback:
    """Every entry of ``transports`` goes off at once (every enabled one for a
    reason of the whole config)."""
    return DsmemFallback((t.index for t in transports.values()), rule)


def dsmem_transports(device_function: DeviceFunction) -> dict[int, DsmemTransport]:
    """The transports this config enables, by load (access id; C1 checked),
    from the dependency graph and the config alone, so the roots' codegen and
    the schedule agree."""
    if device_function.dsmem is None:
        device_function.dsmem = _select(device_function)
    return device_function.dsmem


def dsmem_worker_count(device_function: DeviceFunction, num_sm: int) -> int:
    """The SMs a persistent launch of ``device_function`` takes: ``num_sm``
    (the SMs ``persistent_reserved_sms`` leaves), and with dsmem transports at
    most the SMs that clusters of their size keep resident less the reserved
    ones (a GPC's SMs left over by its clusters stay idle, and reserved SMs
    may take whole clusters' worth from it), rounded down to a multiple of the
    size (``cluster_workers``, as the host's ``get_num_sm``)."""
    transports = dsmem_transports(device_function)
    if not transports:
        return num_sm
    env = CompileEnvironment.current()
    size = cluster_size(transports)
    if env.device.type != "cuda":
        raise config_fallback(transports, "C2: clusters need an NVIDIA CUDA device")
    if size == 1:
        # A launch without clusters faults on its slabs' sends.
        raise config_fallback(
            transports, "C2: local or replicated slabs without a cluster transport"
        )
    try:
        # One CTA an SM: ``num_sm_multiplier`` CTAs share each.
        capacity = get_cluster_capacity(env.device, size)
    except RuntimeError as error:
        # E.g. a cluster size the device (a MIG slice) cannot launch.
        raise config_fallback(transports, f"C2: {error}") from None
    workers = cluster_workers(
        num_sm, capacity, size, env.settings.persistent_reserved_sms
    )
    if workers < size:
        raise config_fallback(
            transports, f"C2: clusters of {size} keep {capacity} SMs resident"
        )
    return workers


def cluster_size(transports: Mapping[int, DsmemTransport]) -> int:
    """The launch's cluster size: every transport's but local ones
    (``_select`` keeps one size, which replicated ones take), 1 without such
    transports."""
    return max((t.size for t in transports.values()), default=1)


def replicated_load(load: TileAccess) -> bool:
    """Whether a load reads the whole allocation (every subscript a full
    slice): its slab is every consumer task's, a replicated one's when the
    consumer root runs on more CTAs than a cluster."""
    return all(load.subscript_is_full_slice)


def cluster_smem_bytes(transports: Mapping[int, DsmemTransport]) -> int:
    """The shared memory the transports declare: a multiple of 1024 bytes, so
    the dynamic shared memory after it keeps its alignment."""
    end = max(t.buffer + t.nbytes for t in transports.values())
    return (end + 1023) // 1024 * 1024


def cluster_exit(transports: Mapping[int, DsmemTransport]) -> ast.stmt:
    """The kernel's last statement, clearing this CTA's flag once its
    replicated slabs have arrived: their senders poll the flag, and a CTA
    exits with no send to its shared memory outstanding
    (``gluon_ops.cluster_exit``)."""
    waits = ",".join(str(t.barrier) for t in transports.values() if t.replicated)
    return statement_from_string(f"_helion_gluon.cluster_exit({waits!r})")


def cluster_entry(transports: Mapping[int, DsmemTransport]) -> ast.stmt:
    """The kernel's entry statement declaring and arming ``CLUSTER_SMEM``."""
    expects = ",".join(
        str(t.nbytes) for t in sorted(transports.values(), key=lambda t: t.barrier)
    )
    return statement_from_string(
        f"_helion_gluon.cluster_entry({cluster_smem_bytes(transports)}, {expects!r})"
    )


def config_int(value: object, config: Config) -> int | None:
    """``value`` as the integer ``config`` gives it (specialized sizes, block
    sizes, tunables and host helpers over them), or None."""
    env = CompileEnvironment.current()
    if isinstance(value, torch.SymInt):
        value = value._sympy_()
    if isinstance(value, sympy.Basic):
        value = env.specialize_expr(cast("sympy.Expr", value))
        if isinstance(value, sympy.Integer):
            return int(value)
    return env.config_value(value, config)


def _block_size(axis: int, config: Config) -> int | None:
    env = CompileEnvironment.current()
    return config_int(env.block_sizes[axis].from_config(config), config)


def _select(device_function: DeviceFunction) -> dict[int, DsmemTransport]:
    graph = HostFunction.current().device_ir.tile_dependency_graph
    config = device_function.config
    flags = cast("list[bool]", config.get("cross_loop_dsmem", None) or [])
    if graph is None or not any(flags):
        return {}
    device_ir = HostFunction.current().device_ir
    enabled = [
        index for index, flag in enumerate(flags) if flag
    ]  # the entries a reason of the whole config turns off
    if config.cross_loop_pipeline == "barrier":
        # Phase loops: no task lanes to put a key's tasks on one cluster.
        raise DsmemFallback(enabled, "C2: cross_loop_pipeline='barrier'")
    transports: dict[int, DsmemTransport] = {}
    layout = DsmemLayout()
    for index, ((stores, load), flag) in enumerate(
        zip(graph.dsmem_candidates, flags, strict=True)
    ):
        if not flag:
            continue
        name = load.tensor_name or f"allocation {load.allocation_id}"
        refusal = _static_refusal(stores, load, device_ir.root_ids)
        if refusal is not None:
            raise DsmemFallback((index,), f"{name}: C1: {refusal}")
        slab = _slab(
            stores,
            load,
            lambda axis: _block_size(axis, config),
            lambda value: config_int(value, config),
        )
        if isinstance(slab, str):
            raise DsmemFallback((index,), f"{name}: C1: {slab}")
        family = graph.task_families[load.root]
        unread = _unread_axes(
            load,
            family,
            lambda axis: _block_size(axis, config),
            lambda value: config_int(value, config),
        )
        replicated = replicated_load(load) and dsmem_replicated(
            slab.size, [_axis_tasks(family, axis, config) for axis in unread]
        )
        if not replicated and len(unread) > (slab.size > 1):
            raise DsmemFallback(
                (index,), f"{name}: C1: consumer root axes the load does not index"
            )
        cluster = 1 if replicated else slab.size
        refusal = layout.refusal(cluster, slab.nbytes)
        if refusal is not None:
            raise DsmemFallback((index,), f"{name}: C1: {refusal}")
        buffer, barrier = layout.take(cluster, slab.nbytes)
        transports[load.access_id] = DsmemTransport(
            index=index,
            name=name,
            stores=stores,
            load=load,
            slab=slab,
            buffer=buffer,
            barrier=barrier,
            consumer_axis=unread[0] if unread and not replicated else None,
            # The launch's cluster size, set below once the others are known.
            replicas=1 if replicated else None,
        )
    size = cluster_size({a: t for a, t in transports.items() if not t.replicated})
    return {
        load_id: (
            dataclasses.replace(transport, replicas=size)
            if transport.replicated
            else transport
        )
        for load_id, transport in transports.items()
    }


def _unread_axes(
    load: TileAccess,
    family: TaskFamily,
    block_size: Callable[[int], int | None],
    value_of: Callable[[object], int | None],
) -> list[int]:
    """The consumer root's axes of more than one task (or an unknown count)
    the load's subscripts do not index: one is a per-rank consumer's split
    axis (every task along it reads the same slab)."""

    def single(axis: TaskAxis) -> bool:
        extent = None if axis.extent is None else value_of(axis.extent)
        block = block_size(axis.block_id)
        return extent is not None and block is not None and extent <= block

    return [axis.block_id for axis in _unread(load, family) if not single(axis)]


def _unread(load: TileAccess, family: TaskFamily) -> list[TaskAxis]:
    """The consumer root's axes the load's subscripts do not index."""
    read = {
        block_id for block_id in load.subscript_affine_block_ids if block_id is not None
    } | {span[0] for span in load.subscript_dense_spans if span is not None}
    return [axis for axis in family.axes if axis.block_id not in read]


def _axis_tasks(family: TaskFamily, block_id: int, config: Config) -> int | None:
    """The tasks of a root's axis under ``config`` (None: unknown)."""
    axis = next(axis for axis in family.axes if axis.block_id == block_id)
    extent = None if axis.extent is None else config_int(axis.extent, config)
    block = _block_size(block_id, config)
    return None if extent is None or block is None else -(-extent // block)


def _static(
    values: Sequence[IntegerExpression], value_of: Callable[[object], int | None]
) -> tuple[int, ...] | None:
    out = []
    for value in values:
        resolved = value_of(value)
        if resolved is None:
            return None
        out.append(resolved)
    return tuple(out)


def _single(access: TileAccess, position: int) -> bool:
    """Whether a subscript position names one element."""
    return access.subscript_is_scalar[position] or (
        position < len(access.subscript_static_extents)
        and access.subscript_static_extents[position] == 1
    )


@dataclasses.dataclass(frozen=True)
class _Slab:
    """C1's slab: per subscript position of each store and of the load, the
    slab stride of its index and whether it counts from its range's first
    element (a single element, and a store's range of a dimension the load
    reads whole, are their own index)."""

    store_strides: tuple[tuple[int, ...], ...]
    load_strides: tuple[int, ...]
    store_ranges: tuple[tuple[bool, ...], ...]
    load_ranges: tuple[bool, ...]
    numel: int
    size: int
    vec: int

    @property
    def nbytes(self) -> int:
        return 4 * self.numel


def _spans(
    access: TileAccess, shape: tuple[int, ...], block_size: Callable[[int], int | None]
) -> list[int | None] | str:
    """Per subscript position, the elements a task's index spans: a whole
    dimension or a tile of a root axis (None: one element)."""
    spans: list[int | None] = []
    for position, dim in enumerate(access.subscript_dims):
        if access.subscript_is_full_slice[position]:
            size = shape[dim]
        elif _single(access, position):
            size = 1
        else:
            axis = access.subscript_affine_block_ids[position]
            size = (
                None
                if axis is None or access.subscript_index_scales[position] != 1
                else block_size(axis)
            )
            if size is None:
                return "an index that is no whole dimension, tile or element"
        spans.append(None if size == 1 else size)
    return spans


def _static_refusal(
    stores: Sequence[TileAccess], load: TileAccess, root_ids: Sequence[int]
) -> str | None:
    """The C1 rule the stores and the load break under every config, or None."""
    accesses = (*stores, load)
    if any(access.graph_id != root_ids[access.root] for access in accesses):
        return "an access in a loop or branch"
    if (
        load.dtype is None
        or load.dtype.itemsize != 4
        or any(store.dtype != load.dtype for store in stores)
    ):
        return "not 32-bit elements"
    if any(access.has_explicit_mask for access in accesses):
        return "a masked access"
    return None


def _slab(
    stores: Sequence[TileAccess],
    load: TileAccess,
    block_size: Callable[[int], int | None],
    value_of: Callable[[object], int | None],
) -> _Slab | str:
    """C1's slab of the stores and the load (which ``_static_refusal``
    accepts), or what fails."""
    shapes = [_static(access.tensor_shape, value_of) for access in (*stores, load)]
    strides = [_static(access.tensor_strides, value_of) for access in (*stores, load)]
    if None in shapes or None in strides:
        return "a dynamic shape or stride"
    spans: list[list[int | None]] = []
    for access, shape, stride in zip((*stores, load), shapes, strides, strict=True):
        assert shape is not None and stride is not None
        if sorted(access.subscript_dims) != list(range(len(shape))):
            return "a subscript that does not index every dimension once"
        access_spans = _spans(access, shape, block_size)
        if isinstance(access_spans, str):
            return access_spans
        if any(
            stride[d] == 0 and span is not None
            for d, span in zip(access.subscript_dims, access_spans, strict=True)
        ):
            return "a range over a broadcast (stride 0) dimension"
        spans.append(access_spans)
    load_spans, load_stride = spans[-1], cast("tuple[int, ...]", strides[-1])
    # The load's region: the ranges it reads (its single elements are its key),
    # merged into runs that nest exactly, outermost first.
    region = sorted(
        (
            (load_stride[dim], span)
            for dim, span in zip(load.subscript_dims, load_spans, strict=True)
            if span is not None and span > 1
        ),
        reverse=True,
    )
    runs: list[tuple[int, int]] = []
    for stride, extent in region:
        if runs and runs[-1][0] == stride * extent:
            runs[-1] = (stride, runs[-1][1] * extent)
        elif runs and runs[-1][0] < stride * extent:
            return "the load's region overlaps itself"
        else:
            runs.append((stride, extent))
    run_strides = [
        math.prod(extent for _, extent in runs[i + 1 :]) for i in range(len(runs))
    ]
    numel = math.prod(extent for _, extent in runs)

    def slab_stride(stride: int, count: int) -> int | None:
        """The slab stride of an index of ``count`` values a ``stride`` apart in
        the tensor (None: outside the region, or across two of its runs)."""
        if count == 1:
            return 0
        for (run, extent), run_stride in zip(runs, run_strides, strict=True):
            if stride % run == 0 and stride // run * (count - 1) < extent:
                return stride // run * run_stride
        return None

    load_strides = tuple(
        0 if span is None else cast("int", slab_stride(load_stride[dim], span))
        for dim, span in zip(load.subscript_dims, load_spans, strict=True)
    )
    # A store's index of a dimension the load reads whole is its place in the
    # slab; of one the load reads a tile of, its place in that tile.
    whole = {
        dim
        for position, dim in enumerate(load.subscript_dims)
        if load.subscript_is_full_slice[position]
    }
    store_strides = []
    # The elements a producer task stores: its stores' tiles together.
    tile = 0
    for store, store_spans, shape, stride in zip(
        stores, spans[:-1], shapes[:-1], strides[:-1], strict=True
    ):
        assert shape is not None and stride is not None
        each, store_tile = [], 1
        for dim, span in zip(store.subscript_dims, store_spans, strict=True):
            if span is None:
                # A single element outside the load's region is the key.
                each.append(slab_stride(stride[dim], shape[dim]) or 0)
                continue
            each_stride = slab_stride(stride[dim], shape[dim] if dim in whole else span)
            if each_stride is None:
                return "a range a store writes is not in one run of the load's"
            each.append(each_stride)
            store_tile *= span
        store_strides.append(tuple(each))
        tile += store_tile
    if numel % tile:
        return f"a slab of {numel} elements from tiles of {tile}"

    def vector(strides: Sequence[int], spans: Sequence[int | None]) -> int:
        """4 or else 2 (16 or 8 bytes) when the access's last range (the
        value's last dimension) is contiguous in the slab, a multiple of it
        long, and every other stride keeps that alignment; else 1."""
        ranges = [
            (stride, span)
            for stride, span in zip(strides, spans, strict=True)
            if span is not None and span > 1
        ]
        if not ranges or ranges[-1][0] != 1:
            return 1
        return next(
            (
                vec
                for vec in (4, 2)
                if ranges[-1][1] % vec == 0
                and all(stride % vec == 0 for stride, _ in ranges[:-1])
            ),
            1,
        )

    return _Slab(
        store_strides=tuple(store_strides),
        load_strides=load_strides,
        store_ranges=tuple(
            tuple(
                span is not None and dim not in whole
                for dim, span in zip(store.subscript_dims, store_spans, strict=True)
            )
            for store, store_spans in zip(stores, spans[:-1], strict=True)
        ),
        load_ranges=tuple(span is not None for span in load_spans),
        numel=numel,
        size=numel // tile,
        vec=min(
            vector(load_strides, load_spans),
            *starmap(vector, zip(store_strides, spans[:-1], strict=True)),
        ),
    )


def dsmem_slab_facts(
    graph: TileDependencyGraph, root_ids: Sequence[int]
) -> tuple[DsmemSlabFacts | None, ...]:
    """Per candidate, the factors of its slab's elements (the load's ranges),
    of its stores' tiles (their ranges) and of a whole-allocation load's
    consumer root axes that ``ConfigSpec.normalize`` evaluates under a config:
    each an integer, a tunable expression or a block's index into
    ``block_sizes``; None when a factor has no such form, which leaves the
    candidate to codegen.  A candidate that breaks a C1 rule under every
    config is never carried."""
    env = CompileEnvironment.current()
    spec_blocks = env.config_spec.block_sizes

    def factor(value: object) -> object | None:
        if isinstance(value, torch.SymInt):
            value = value._sympy_()
        if isinstance(value, sympy.Basic):
            value = env.specialize_expr(cast("sympy.Expr", value))
            if isinstance(value, sympy.Integer):
                value = int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, sympy.Basic):
            return env.config_value_expression(value)
        return None

    def block(axis: int | None) -> object | None:
        if axis is None:
            return None
        info = env.block_sizes[axis]
        source = info.block_size_source
        if isinstance(source, FixedBlockSizeSource):
            return factor(source.value)
        if not isinstance(source, LoopSpecBlockSizeSource) or (
            # The size-1 rule of ``LoopSpecBlockSizeSource.from_config`` may
            # fix it at 1: left to codegen.
            isinstance(info.size, (int, torch.SymInt)) and env.known_equal(info.size, 1)
        ):
            return None
        try:
            return ("block", spec_blocks.block_id_to_index(axis))
        except KeyError:
            return None

    def factors(access: TileAccess) -> tuple[object, ...] | None:
        out: list[object] = []
        for position, dim in enumerate(access.subscript_dims):
            if access.subscript_is_full_slice[position]:
                value = factor(access.tensor_shape[dim])
            elif _single(access, position):
                continue
            elif access.subscript_index_scales[position] != 1:
                return None
            else:
                value = block(access.subscript_affine_block_ids[position])
            if value is None:
                return None
            out.append(value)
        return tuple(out)

    facts: list[DsmemSlabFacts | None] = []
    for stores, load in graph.dsmem_candidates:
        if _static_refusal(stores, load, root_ids) is not None:
            facts.append(DsmemSlabFacts(numel=(), tiles=(), carried=False))
            continue
        numel, tiles = factors(load), tuple(map(factors, stores))
        # A load of the whole allocation is replicated or not by its consumer
        # root's task counts under the config (``_select``).
        axes = (
            tuple(
                (factor(axis.extent), block(axis.block_id))
                for axis in _unread(load, graph.task_families[load.root])
            )
            if replicated_load(load)
            else None
        )
        unknown = None in (numel, *tiles, *chain(*(axes or ())))
        facts.append(
            None
            if unknown
            else DsmemSlabFacts(
                numel=cast("tuple[object, ...]", numel),
                tiles=cast("tuple[tuple[object, ...], ...]", tiles),
                axes=axes,
            )
        )
    return tuple(facts)


def cluster_layout(
    shape: Sequence[int], vec: int, num_warps: int
) -> tuple[list[int], list[int], list[int], int] | None:
    """C3: a blocked layout (elements, lanes and warps a dimension) of ``vec``
    contiguous elements of the last dimension a thread for ``shape``, and the
    threads that hold each element (the lanes and warps the shape leaves
    over, on its first dimension), or None."""
    rank = len(shape)
    per = [1] * (rank - 1) + [vec]
    if rank == 0 or shape[-1] % vec:
        return None
    lanes, warps = [1] * rank, [1] * rank
    left_lanes, left_warps = 32, num_warps
    for d in reversed(range(rank)):
        lanes[d] = math.gcd(shape[d] // per[d], left_lanes)
        left_lanes //= lanes[d]
    for d in range(rank):
        warps[d] = math.gcd(shape[d] // (per[d] * lanes[d]), left_warps)
        left_warps //= warps[d]
    lanes[0] *= left_lanes
    warps[0] *= left_warps
    return per, lanes, warps, left_lanes * left_warps


def send_layout(shape: Sequence[int], vec: int, num_warps: int) -> tuple[int, int]:
    """The elements a thread moves at once (at most ``vec``) and the threads
    that hold each element of a tile of ``shape``: the fewest copies, then the
    most elements a thread."""
    best = None
    for each in (4, 2, 1):
        dims = cluster_layout(shape, each, num_warps) if each <= vec else None
        if dims is not None and (best is None or dims[3] < best[1]):
            best = (each, dims[3])
    assert best is not None  # one element a thread always fits
    return best


def cluster_recv_layout(
    shape: Sequence[int], vec: int, num_warps: int
) -> tuple[list[int], list[int], list[int]] | None:
    """The blocked layout a slab is read in: the leading (split) dimension
    whole in every thread, so the sum over it the combine does takes no data
    from other threads, and ``vec`` contiguous elements of the last; lanes and
    warps over the other dimensions, every element on exactly one thread.
    None when no such layout covers ``shape``."""
    rank = len(shape)
    if rank < 2 or shape[-1] % vec:
        return None
    per = [shape[0]] + [1] * (rank - 2) + [vec]
    lanes, warps = [1] * rank, [1] * rank
    left_lanes, left_warps = 32, num_warps
    for d in reversed(range(1, rank)):
        lanes[d] = math.gcd(shape[d] // per[d], left_lanes)
        left_lanes //= lanes[d]
    for d in range(1, rank):
        warps[d] = math.gcd(shape[d] // (per[d] * lanes[d]), left_warps)
        left_warps //= warps[d]
    if left_lanes != 1 or left_warps != 1:
        return None
    return per, lanes, warps


def split_task(key: str, rank: str, size: int, multiplier: int) -> str:
    """The producer task of ``key``'s ``rank``-th part: ``key`` on the axes
    besides the split axis (task order), ``rank`` on the split axis, whose
    tasks are ``multiplier`` apart (the faster axes' task count)."""
    if multiplier == 1:
        return f"({key}) * {size} + {rank}"
    return (
        f"({key}) % {multiplier} + ({key}) // {multiplier} * {multiplier * size}"
        f" + ({rank}) * {multiplier}"
    )


def _split_task_of(size: int, multiplier: int) -> Callable[[int, int], int]:
    """``split_task`` on integers: C2 checks the formula the kernel runs."""
    return eval(f"lambda key, rank: {split_task('key', 'rank', size, multiplier)}")


def check_dispatch(
    transports: Mapping[int, DsmemTransport],
    *,
    config: Config,
    domains: Sequence[CoordinateDomain],
    geometries: Sequence[tuple[tuple[int, ...], Mapping[int, object], object]],
    workers: int,
) -> dict[int, tuple[int, int | None]]:
    """C2's counts and elements (the caller checks the roots' dispatch): per
    transport (by load) but local and replicated ones, the multiplier of its
    producer root's split axis (``split_task``), one axis for all transports
    of a root, and a per-rank consumer root's (else None)."""
    # Per producer root, the split axes every transport so far accepts, and
    # those transports' loads.
    accepted: dict[int, list[int]] = {}
    loads: dict[int, list[int]] = {}
    consumer_multipliers: dict[int, int | None] = {}
    # Per consumer root, the split axis multiplier its per-rank slabs share.
    consumer_splits: dict[int, int] = {}
    with unset_fake_temporarily():
        roles: dict[int, tuple[bool, int]] = {}
        # Local slabs last: they take the lanes their roots' other transports
        # give.
        for transport in sorted(transports.values(), key=lambda t: t.local):
            producer, consumer = transport.producer_root, transport.consumer_root
            size = transport.size
            # A replicated slab's consumer runs anywhere: no role.
            for root, role in (
                (producer, (False, consumer)),
                *(() if transport.replicated else ((consumer, (True, producer)),)),
            ):
                if roles.setdefault(root, role) != role:
                    raise transport.fallback("C2: a root in two cluster roles")
            # ``dsmem_worker_count`` keeps every worker resident in whole
            # clusters.
            assert workers % size == 0, (workers, size)
            if any(
                t.replicated != transport.replicated
                for t in transports.values()
                if t.producer_root == producer
            ):
                raise transport.fallback(
                    "C2: a producer root of replicated and other slabs"
                )
            if transport.replicated:
                # Every cluster's producer tasks write the slab once, which
                # every consumer task reads whole (the same for each: the
                # load's subscripts are full slices).
                stored = _task_elements(transport, "store", config, domains, geometries)
                (loaded,) = _task_elements(
                    transport, "load", config, domains, geometries
                )
                every = torch.arange(transport.numel, device="cpu")
                if not _key_matches(stored, *loaded, every):
                    raise transport.fallback(
                        "C2: the producer tasks write other than the slab once"
                    )
                continue
            # The task counts before the elements, which cost substitutions a
            # task.
            producers = _task_count(transport, producer, geometries)
            consumers = _task_count(transport, consumer, geometries)
            if producers > workers:
                raise transport.fallback("C2: a producer root of more than one wave")
            if producers != (
                consumers if transport.per_rank or transport.local else size * consumers
            ):
                raise transport.fallback(
                    f"C2: {producers} producer tasks for {consumers} consumer tasks"
                )
            stored = _task_elements(transport, "store", config, domains, geometries)
            loaded = _task_elements(transport, "load", config, domains, geometries)
            every = torch.arange(transport.numel, device="cpu")
            if transport.local:
                if not all(
                    _key_matches([stored[task]], *loaded[consumer_task], every)
                    for task, consumer_task in _local_tasks(
                        transport, transports, accepted, consumer_multipliers, producers
                    )
                ):
                    raise transport.fallback(
                        "C2: a consumer task reads other than what the producer"
                        " task of its CTA writes"
                    )
                continue
            # Per key, what each of its consumer tasks reads: per rank on a
            # per-rank consumer's split axis, else the key's one task.
            reads_by_key = [[entry] for entry in loaded]
            consumer_multiplier = None
            if transport.per_rank:
                consumer_multiplier = _axis_multiplier(
                    geometries[consumer], transport.consumer_axis, size
                )
                if consumer_multiplier is None:
                    raise transport.fallback(
                        f"C2: the consumer root's split axis is not {size} tasks"
                    )
                if (
                    consumer_splits.setdefault(consumer, consumer_multiplier)
                    != consumer_multiplier
                ):
                    raise transport.fallback(
                        "C2: per-rank slabs of one consumer root on different axes"
                    )
                consumer_task = _split_task_of(size, consumer_multiplier)
                reads_by_key = [
                    [loaded[consumer_task(key, rank)] for rank in range(size)]
                    for key in range(consumers // size)
                ]
            axis_order, counts, _ = geometries[producer]
            # Each axis of ``size`` tasks may be the split axis, fastest first
            # (where producer task ``size * k + rank`` is key ``k``'s).
            candidates = []
            multiplier = 1
            for axis in axis_order:
                count = counts[axis]
                assert isinstance(count, int)
                if count == size:
                    candidates.append(multiplier)
                multiplier *= count
            matching = []
            for multiplier in candidates:
                task = _split_task_of(size, multiplier)
                if all(
                    _key_matches(
                        [stored[task(key, rank)] for rank in range(size)],
                        storage,
                        slab,
                        every,
                    )
                    for key, reads in enumerate(reads_by_key)
                    for storage, slab in reads
                ):
                    matching.append(multiplier)
            if not matching:
                raise transport.fallback(
                    f"C2: no axis of {size} producer tasks whose tasks of one key"
                    " write exactly what that key's consumer task reads"
                )
            common = [m for m in accepted.get(producer, matching) if m in matching]
            if not common:
                raise transport.fallback(
                    "C2: allocations of one producer root split on different axes"
                )
            accepted[producer] = common
            loads.setdefault(producer, []).append(transport.load.access_id)
            consumer_multipliers[transport.load.access_id] = consumer_multiplier
    return {
        load_id: (accepted[producer][0], consumer_multipliers[load_id])
        for producer, ids in loads.items()
        for load_id in ids
    }


def check_replicated(transport: DsmemTransport, graph: TileDependencyGraph) -> None:
    """C2 for a replicated slab's producer root, which every cluster runs while
    only cluster 0's tasks arrive: each cluster's tasks must read what cluster
    0's read, and store the same values.  So no atomic or cross-rank access,
    no read of what the root itself stores, and no later root's write ordered
    after an access of the root's (the other clusters may still be reading,
    or storing again what the later root overwrites).  And no task stores an
    element twice, for a later cluster's first value could land after
    cluster 0's second and its arrival: each store runs once a task (outside
    loops, which also keeps out nested-loop arrivals, every cluster's), and
    two stores of one allocation touch no common element (``_stored_apart``).
    Two tasks that store one element race without clusters too: the consumer
    reads one of their values either way.  (The other clusters' identical
    stores cost less than guarding them.)"""
    root = transport.producer_root
    accesses = [access for access in graph.accesses if access.root == root]
    stores = [access for access in accesses if access.kind == "store"]
    stored = {store.allocation_id for store in stores}
    for access in accesses:
        if access.is_atomic or access.owner_rank is not None:
            raise transport.fallback(
                "C2: an atomic or cross-rank access in a replicated slab's"
                " producer root"
            )
        if access.kind == "load" and access.allocation_id in stored:
            raise transport.fallback(
                "C2: a replicated slab's producer root reads what it stores"
            )
    for edge in graph.edges:
        if edge.producer_root != root:
            continue
        for dependency in edge.access_dependencies:
            if dependency.kind is not TileDependencyKind.READ_AFTER_WRITE or (
                graph.transport(dependency) not in ("counter", "cluster")
            ):
                raise transport.fallback(
                    "C2: a later root's write after a replicated slab's producer"
                    " root's access"
                )
    if not all(_once_a_task(graph, store) for store in stores):
        raise transport.fallback(
            "C2: a replicated slab's producer root stores in a loop"
        )
    for first, second in combinations(stores, 2):
        if first.allocation_id == second.allocation_id and not _stored_apart(
            first, second
        ):
            raise transport.fallback(
                "C2: a replicated slab's producer root may store an element twice"
            )


def _stored_apart(first: TileAccess, second: TileAccess) -> bool:
    """Whether two stores of one task write no common element, as the barrier
    walk decides it for two accesses of one iteration (``accesses_apart``)."""
    graphs = HostFunction.current().device_ir.graphs

    def node(access: TileAccess) -> torch.fx.Node:
        return list(graphs[access.graph_id].graph.nodes)[access.graph_node_index]

    return accesses_apart(node(first), node(second))


def _once_a_task(graph: TileDependencyGraph, access: TileAccess) -> bool:
    """Whether ``access`` runs at most once a task of its root: in the root
    or its branches, under no loop."""
    sites = graph.execution_sites

    def unlooped(site_id: int | None) -> bool:
        while site_id is not None:
            if sites[site_id].kind not in ("root", "branch"):
                return False
            site_id = sites[site_id].parent_site_id
        return True

    site_ids = graph.site_ids_by_access[access.access_id]
    return bool(site_ids) and all(map(unlooped, site_ids))


def _local_tasks(
    transport: DsmemTransport,
    transports: Mapping[int, DsmemTransport],
    accepted: Mapping[int, list[int]],
    consumer_multipliers: Mapping[int, int | None],
    tasks: int,
) -> list[tuple[int, int]]:
    """A local slab's (producer task, consumer task) pairs a CTA runs: the
    split lanes of its roots' per-rank transport, else one to one."""
    pair = next(
        (
            t
            for t in transports.values()
            if t.producer_root == transport.producer_root and not t.local
        ),
        None,
    )
    if pair is None:
        return [(task, task) for task in range(tasks)]
    consumer_multiplier = consumer_multipliers[pair.load.access_id]
    # The pair has the local slab's roots (one cluster role a root), whose
    # counts a rank 0 consumer's transport (``size`` producer tasks a consumer
    # task) and a local one (one) cannot both pass: the pair is per-rank.
    assert consumer_multiplier is not None
    producer_task = _split_task_of(pair.size, accepted[pair.producer_root][0])
    consumer_task = _split_task_of(pair.size, consumer_multiplier)
    return [
        (producer_task(key, rank), consumer_task(key, rank))
        for key in range(tasks // pair.size)
        for rank in range(pair.size)
    ]


def _axis_multiplier(
    geometry: tuple[tuple[int, ...], Mapping[int, object], object],
    axis: int | None,
    size: int,
) -> int | None:
    """The task-index multiplier of ``axis`` in a root's task order (the
    faster axes' task count), if it has ``size`` tasks; else None."""
    axis_order, counts, _ = geometry
    multiplier = 1
    for candidate in axis_order:
        count = counts[candidate]
        assert isinstance(count, int)
        if candidate == axis:
            return multiplier if count == size else None
        multiplier *= count
    return None


def _key_matches(
    parts: Sequence[tuple[torch.Tensor, torch.Tensor]],
    storage: torch.Tensor,
    slab: torch.Tensor,
    every: torch.Tensor,
) -> bool:
    """Whether a key's producer tasks write exactly (storage and slab offsets)
    what its consumer task reads, the whole slab once."""
    written = torch.cat([part[0] for part in parts])
    written_slab = torch.cat([part[1] for part in parts])
    order, written_order = storage.argsort(), written.argsort()
    return (
        storage.numel() == written.numel()
        and torch.equal(storage[order], written[written_order])
        and torch.equal(slab[order], written_slab[written_order])
        and torch.equal(slab.sort().values, every)
    )


def _task_count(
    transport: DsmemTransport,
    root: int,
    geometries: Sequence[tuple[tuple[int, ...], Mapping[int, object], object]],
) -> int:
    """The tasks of ``root`` (static counts only)."""
    axis_order, counts, _ = geometries[root]
    if not all(isinstance(counts[axis], int) for axis in axis_order):
        raise transport.fallback("C2: a dynamic task count")
    return math.prod(counts[axis] for axis in axis_order)  # pyrefly: ignore [no-matching-overload]


def _task_elements(
    transport: DsmemTransport,
    kind: str,
    config: Config,
    domains: Sequence[CoordinateDomain],
    geometries: Sequence[tuple[tuple[int, ...], Mapping[int, object], object]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per task of the transport's producer or consumer root (task order),
    the storage offsets of the elements its stores or its load touch and their
    slab offsets (a replicated slab's load: its first task's, every task's)."""
    if kind == "load":
        return _access_elements(
            transport,
            transport.load,
            transport.slab.load_strides,
            transport.slab.load_ranges,
            1
            if transport.replicated
            else _task_count(transport, transport.consumer_root, geometries),
            config,
            domains,
            geometries,
        )
    tasks = _task_count(transport, transport.producer_root, geometries)
    parts = [
        _access_elements(
            transport, store, strides, ranges, tasks, config, domains, geometries
        )
        for store, strides, ranges in zip(
            transport.stores,
            transport.slab.store_strides,
            transport.slab.store_ranges,
            strict=True,
        )
    ]
    return [
        (
            torch.cat([part[task][0] for part in parts]),
            torch.cat([part[task][1] for part in parts]),
        )
        for task in range(tasks)
    ]


def _access_elements(
    transport: DsmemTransport,
    access: TileAccess,
    strides: Sequence[int],
    ranges: Sequence[bool],
    tasks: int,
    config: Config,
    domains: Sequence[CoordinateDomain],
    geometries: Sequence[tuple[tuple[int, ...], Mapping[int, object], object]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``_task_elements`` of one access, for its root's first ``tasks``."""
    kind = access.kind
    axis_order, counts, _ = geometries[access.root]
    # A producer task touches one part of the slab, a consumer task all of it.
    touched = (
        transport.numel // transport.slab.size if kind == "store" else transport.numel
    )
    if tasks * touched > _MAX_CHECKED_ELEMENTS:
        raise transport.fallback("C2: too many elements to check")
    coordinate_symbols = {coordinate_axis_symbol(axis) for axis in axis_order}

    def resolved(expr: object) -> sympy.Expr | None:
        """``expr`` with every symbol but the task coordinates at this
        config's value."""
        expr = sympy.sympify(expr)
        subs = {}
        for symbol in expr.free_symbols - coordinate_symbols:
            value = config_int(symbol, config)
            if value is None:
                return None
            subs[symbol] = sympy.Integer(value)
        return expr.xreplace(subs) if subs else expr

    intervals: list[tuple[sympy.Expr, sympy.Expr] | None] = []
    for position in range(len(access.subscript_dims)):
        interval = _access_interval_expression(
            access, position=position, domain=domains[access.root]
        )
        bounds = None if interval is None else tuple(map(resolved, interval))
        intervals.append(
            None
            if bounds is None or None in bounds
            else cast("tuple[sympy.Expr, sympy.Expr]", bounds)
        )
    tensor_strides = _static(
        access.tensor_strides, lambda value: config_int(value, config)
    )
    # Offsets relative to the load's view (the allocation's offset may be
    # dynamic; the views' difference is not).
    storage_offset = config_int(
        sympy.sympify(access.storage_offset)
        - sympy.sympify(transport.load.storage_offset),
        config,
    )
    if None in intervals:
        raise transport.fallback(
            f"C2: the {kind}'s interval of subscript {intervals.index(None)} is unknown"
        )
    if tensor_strides is None or storage_offset is None:
        raise transport.fallback(
            f"C2: the {kind}'s strides or offset from the load's view are dynamic"
        )
    out = []
    for task in range(tasks):
        # Task 0 takes no count (a replicated load's counts may be dynamic).
        coordinates = {coordinate_axis_symbol(axis): 0 for axis in axis_order}
        multiplier = 1
        for axis in axis_order if task else ():
            count = counts[axis]
            assert isinstance(count, int)
            coordinates[coordinate_axis_symbol(axis)] = task // multiplier % count
            multiplier *= count
        # On the host whatever the default device.
        storage = torch.tensor([storage_offset], device="cpu")
        slab = torch.tensor([0], device="cpu")
        for position, interval in enumerate(intervals):
            assert interval is not None
            begin, end = (int(sympy.sympify(e).subs(coordinates)) for e in interval)
            index = torch.arange(begin, end, device="cpu")
            dim = access.subscript_dims[position]
            relative = index - begin if ranges[position] else index
            storage = (storage[:, None] + index * tensor_strides[dim]).flatten()
            slab = (slab[:, None] + relative * strides[position]).flatten()
        out.append((storage, slab))
    return out
