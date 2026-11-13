"""Cluster edges (``cross_loop_cluster_edges``).

A split and combine pair of roots, the first writing the partials of a
reduction indexed by a split tile and the second summing them over that
dimension, passes the partials through global memory and orders them with a
readiness counter (or a root barrier).  A cluster edge launches the kernel in
clusters of ``ClusterEdge.size`` CTAs instead, runs the producer tasks of one
key (one consumer task) on the CTAs of one cluster and the consumer task on its
rank 0: each producer task sends its slab into rank 0's static shared memory
(``st.async``, whose bytes complete rank 0's mbarrier) and the consumer task
waits for the barrier's phase and reads the slabs from shared memory.  The
dependency's transport is then "cluster": the scheduler orders nothing for it.
The global store stays, ordered by nothing in the launch: the allocation holds
what it held.

Candidates (``TileDependencyGraph.cluster_candidate_ids``) do not depend on the
config.  An enabled one that fails a rule regenerates the kernel with its entry
off (``ClusterEdgeFallback``, logged):

- C1 (the slab, ``cluster_edges``): the store and the load run once a task (in
  their root's own graph, outside any loop or branch) and at most 8 CTAs make a
  cluster; the load reads whole dimensions (full
  slices) and single elements (its key), the store too; the whole dimensions
  the load reads are strided runs that do not overlap, and every dimension the
  store writes lies in one of them.  The slab is the load's region in row-major
  run order.  32-bit elements, static shapes and strides, no masks.
- C2 (the schedule, ``check_dispatch``): task lanes (``cross_loop_pipeline``
  static or dynamic, no packets); both
  roots resident and unguarded, without continuations or L2 grouping; the
  producer root a single wave of ``size`` times the consumer's tasks; producer
  tasks ``size * k`` to ``size * k + size - 1`` write exactly what consumer task
  ``k`` reads, element by element (no element twice).
- C3 (codegen, ``cluster_layout``): every tile spreads over its partition's
  threads without duplicates, so the bytes rank 0 expects arrive once.

Dispatch (``cross_loop_codegen``): a producer worker takes the lane its
cluster's rank 0 would, rounded down to a multiple of ``size``, plus its rank,
a bijection for every rotation of the lanes; rank 0 of a cluster takes the
consumer task of the key whose producer tasks its cluster runs.

Shared memory (``gluon_ops.CLUSTER_SMEM``, declared and initialized at kernel
entry): edge ``i``'s barrier is at byte ``8 + 8 i`` (one arrival at entry plus
the slab's bytes complete phase 0, once a launch), its slab at
``ClusterEdge.buffer``.  Every thread arrives on the hardware cluster barrier
once its CTA's barriers are initialized (at entry, or at a ring partition's
start); a producer task waits for it before sending (``CLUSTER_READY``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

from .. import exc
from ..runtime.triton.launcher import get_cluster_capacity
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .tile_dependency import _access_interval_expression
from .tile_dependency import coordinate_axis_symbol

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .device_function import DeviceFunction
    from .tile_dependency import CoordinateDomain
    from .tile_dependency import IntegerExpression
    from .tile_dependency import TileAccess

CLUSTER_SMEM_HEADER = 128
# The token a producer task holds once every CTA of its cluster has armed its
# barriers (the cluster barrier, waited for at the task's start).
CLUSTER_READY = "helion_cluster_ready"
_MAX_EDGES = (CLUSTER_SMEM_HEADER - 8) // 8
# The portable cluster size limit.
_MAX_SIZE = 8
# The C2 check enumerates every slab element of every consumer task.
_MAX_CHECKED_ELEMENTS = 1 << 22


class ClusterEdgeFallback(exc.Base):
    """An enabled cluster edge failed a rule only codegen checks: regenerate the
    kernel with its ``cross_loop_cluster_edges`` entry off."""

    def __init__(self, index: int, reason: str) -> None:
        super().__init__(reason)
        self.index = index

    def report(self) -> str:
        return f"cluster edge fallback: {self}"


@dataclasses.dataclass(frozen=True)
class ClusterEdge:
    """An allocation carried from its store's root to its load's root in
    launch clusters of ``size`` CTAs: one slab a key, in rank 0's shared
    memory from byte ``buffer``, completing the barrier at byte ``barrier``."""

    index: int  # the config entry
    allocation_id: int
    name: str
    store: TileAccess
    load: TileAccess
    slab: _Slab
    buffer: int
    barrier: int

    @property
    def size(self) -> int:
        return self.slab.size

    @property
    def numel(self) -> int:
        return self.slab.numel

    @property
    def nbytes(self) -> int:
        return 4 * self.slab.numel

    @property
    def vec(self) -> int:
        """32-bit elements a thread moves at once."""
        return self.slab.vec

    def fallback(self, rule: str) -> ClusterEdgeFallback:
        return ClusterEdgeFallback(self.index, f"{self.name}: {rule}")


def cluster_edges(device_function: DeviceFunction) -> dict[int, ClusterEdge]:
    """The edges this config enables, by allocation (C1 checked), from the
    dependency graph and the config alone, so the roots' codegen and the
    schedule agree."""
    if device_function.cluster_edges is None:
        device_function.cluster_edges = _select(device_function)
    return device_function.cluster_edges


def cluster_smem_bytes(edges: Mapping[int, ClusterEdge]) -> int:
    """The shared memory the edges declare: a multiple of 1024 bytes, so the
    dynamic shared memory after it keeps its alignment."""
    end = max(edge.buffer + edge.nbytes for edge in edges.values())
    return (end + 1023) // 1024 * 1024


def cluster_entry(edges: Mapping[int, ClusterEdge]) -> ast.stmt:
    """The kernel's entry statement declaring and arming ``CLUSTER_SMEM``."""
    expects = ",".join(
        str(edge.nbytes) for edge in sorted(edges.values(), key=lambda e: e.barrier)
    )
    return statement_from_string(
        f"_helion_gluon.cluster_entry({cluster_smem_bytes(edges)}, {expects!r})"
    )


def _select(device_function: DeviceFunction) -> dict[int, ClusterEdge]:
    graph = HostFunction.current().device_ir.tile_dependency_graph
    flags = cast(
        "list[bool]", device_function.config.get("cross_loop_cluster_edges", None) or []
    )
    if graph is None or not any(flags):
        return {}
    env = CompileEnvironment.current()
    device_ir = HostFunction.current().device_ir

    def block_size(axis: int) -> int | None:
        size = env.block_sizes[axis].from_config(device_function.config)
        return size if isinstance(size, int) else None

    edges: dict[int, ClusterEdge] = {}
    offset = CLUSTER_SMEM_HEADER
    for index, (allocation_id, enabled) in enumerate(
        zip(graph.cluster_candidate_ids, flags, strict=True)
    ):
        if not enabled:
            continue
        store, load = graph.cluster_accesses(allocation_id)
        name = load.tensor_name or f"allocation {allocation_id}"
        if device_function.config.cross_loop_pipeline == "barrier":
            # Phase loops: no task lanes to put a key's tasks on one cluster.
            raise ClusterEdgeFallback(
                index, f"{name}: C2: cross_loop_pipeline='barrier'"
            )
        if any(
            access.graph_id != device_ir.root_ids[access.root]
            for access in (store, load)
        ):
            raise ClusterEdgeFallback(
                index, f"{name}: C1: an access in a loop or branch"
            )
        slab = _slab(store, load, block_size)
        if isinstance(slab, str):
            raise ClusterEdgeFallback(index, f"{name}: C1: {slab}")
        if slab.size > _MAX_SIZE:
            raise ClusterEdgeFallback(
                index, f"{name}: C1: clusters of {slab.size} (at most {_MAX_SIZE})"
            )
        if len(edges) == _MAX_EDGES:
            raise ClusterEdgeFallback(
                index, f"{name}: C1: more than {_MAX_EDGES} edges"
            )
        if any(edge.size != slab.size for edge in edges.values()):
            raise ClusterEdgeFallback(
                index, f"{name}: C1: clusters of {slab.size}, unlike an earlier edge's"
            )
        buffer = (offset + 15) // 16 * 16
        edges[allocation_id] = ClusterEdge(
            index=index,
            allocation_id=allocation_id,
            name=name,
            store=store,
            load=load,
            slab=slab,
            buffer=buffer,
            barrier=8 + 8 * len(edges),
        )
        offset = buffer + edges[allocation_id].nbytes
    return edges


def _static(values: Sequence[IntegerExpression]) -> tuple[int, ...] | None:
    out = []
    for value in values:
        value = sympy.sympify(value)
        if not value.is_Integer:
            return None
        out.append(int(value))
    return tuple(out)


def _single(access: TileAccess, position: int) -> bool:
    """Whether a subscript position names one element."""
    return access.subscript_is_scalar[position] or (
        position < len(access.subscript_static_extents)
        and access.subscript_static_extents[position] == 1
    )


@dataclasses.dataclass(frozen=True)
class _Slab:
    """C1's slab: per subscript position of the store and the load, the slab
    stride of its index and whether it spans a range (counted from the
    range's first element; a single element is its own index)."""

    store_strides: tuple[int, ...]
    load_strides: tuple[int, ...]
    store_ranges: tuple[bool, ...]
    load_ranges: tuple[bool, ...]
    numel: int
    size: int
    vec: int


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


def _slab(
    store: TileAccess, load: TileAccess, block_size: Callable[[int], int | None]
) -> _Slab | str:
    """C1's slab of the store and the load, or what fails."""
    if load.dtype is None or load.dtype.itemsize != 4 or store.dtype != load.dtype:
        return "not 32-bit elements"
    if store.has_explicit_mask or load.has_explicit_mask:
        return "a masked access"
    store_shape, load_shape = _static(store.tensor_shape), _static(load.tensor_shape)
    store_stride, load_stride = (
        _static(store.tensor_strides),
        _static(load.tensor_strides),
    )
    if None in (store_shape, load_shape, store_stride, load_stride):
        return "a dynamic shape or stride"
    assert store_shape is not None and load_shape is not None
    assert store_stride is not None and load_stride is not None
    for access, shape in ((store, store_shape), (load, load_shape)):
        if sorted(access.subscript_dims) != list(range(len(shape))):
            return "a subscript that does not index every dimension once"
    store_spans = _spans(store, store_shape, block_size)
    load_spans = _spans(load, load_shape, block_size)
    if isinstance(store_spans, str):
        return store_spans
    if isinstance(load_spans, str):
        return load_spans
    if any(
        stride == 0 and span is not None
        for stride, span in (
            *(
                (store_stride[d], s)
                for d, s in zip(store.subscript_dims, store_spans, strict=True)
            ),
            *(
                (load_stride[d], s)
                for d, s in zip(load.subscript_dims, load_spans, strict=True)
            ),
        )
    ):
        return "a range over a broadcast (stride 0) dimension"
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
        0 if span is None else slab_stride(load_stride[dim], span) or 0
        for dim, span in zip(load.subscript_dims, load_spans, strict=True)
    )
    store_strides = []
    tile = 1
    for dim, span in zip(store.subscript_dims, store_spans, strict=True):
        if span is None:
            # A single element outside the load's region is the key.
            stride = slab_stride(store_stride[dim], store_shape[dim]) or 0
        else:
            stride = slab_stride(store_stride[dim], span)
            if stride is None:
                return "a range the store writes is not in one run of the load's"
            tile *= span
        store_strides.append(stride)
    if numel % tile or numel // tile < 2:
        return f"a slab of {numel} elements from tiles of {tile}"

    def vector(strides: Sequence[int], spans: Sequence[int | None]) -> int:
        """4 when the access's last range (the value's last dimension) is
        contiguous in the slab and every other stride keeps 16 bytes aligned."""
        ranges = [
            (stride, span)
            for stride, span in zip(strides, spans, strict=True)
            if span is not None and span > 1
        ]
        if not ranges or ranges[-1][0] != 1 or ranges[-1][1] % 4:
            return 1
        return 4 if all(stride % 4 == 0 for stride, _ in ranges[:-1]) else 1

    return _Slab(
        store_strides=tuple(store_strides),
        load_strides=load_strides,
        store_ranges=tuple(span is not None for span in store_spans),
        load_ranges=tuple(span is not None for span in load_spans),
        numel=numel,
        size=numel // tile,
        vec=min(vector(store_strides, store_spans), vector(load_strides, load_spans)),
    )


def cluster_layout(
    shape: Sequence[int], vec: int, num_warps: int
) -> tuple[list[int], list[int], list[int]] | None:
    """C3: a blocked layout (elements, lanes and warps a dimension) of ``vec``
    contiguous elements of the last dimension a thread that puts every element
    of ``shape`` on exactly one thread, or None."""
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
    if left_lanes != 1 or left_warps != 1:
        return None
    return per, lanes, warps


def check_dispatch(
    edges: Mapping[int, ClusterEdge],
    *,
    domains: Sequence[CoordinateDomain],
    geometries: Sequence[tuple[tuple[int, ...], Mapping[int, object], object]],
    workers: int,
    ctas_per_sm: int,
) -> None:
    """C2's counts and elements (the caller checks the roots' dispatch)."""
    with unset_fake_temporarily():
        _check_elements(
            edges,
            domains=domains,
            geometries=geometries,
            workers=workers,
            ctas_per_sm=ctas_per_sm,
        )


def _check_elements(
    edges: Mapping[int, ClusterEdge],
    *,
    domains: Sequence[CoordinateDomain],
    geometries: Sequence[tuple[tuple[int, ...], Mapping[int, object], object]],
    workers: int,
    ctas_per_sm: int,
) -> None:
    roles: dict[int, tuple[bool, int]] = {}
    env = CompileEnvironment.current()
    for edge in edges.values():
        producer, consumer = edge.store.root, edge.load.root
        for root, role in ((producer, (False, consumer)), (consumer, (True, producer))):
            if roles.setdefault(root, role) != role:
                raise edge.fallback("C2: a root in two cluster roles")
        if workers % edge.size:
            raise edge.fallback(f"C2: {workers} workers in clusters of {edge.size}")
        if env.device.type == "cuda":
            # Every worker resident (``persistent_reserved_sms`` leaves SMs out).
            capacity = get_cluster_capacity(env.device, edge.size) * ctas_per_sm
            if workers > capacity:
                raise edge.fallback(
                    f"C2: {workers} workers, of which clusters of {edge.size} "
                    f"keep {capacity} resident"
                )
        stored = _task_elements(edge, "store", domains, geometries)
        loaded = _task_elements(edge, "load", domains, geometries)
        if len(stored) > workers:
            raise edge.fallback("C2: a producer root of more than one wave")
        if len(stored) != edge.size * len(loaded):
            raise edge.fallback(
                f"C2: {len(stored)} producer tasks for {len(loaded)} consumer tasks"
            )
        every = torch.arange(edge.numel)
        for key, (storage, slab) in enumerate(loaded):
            parts = stored[edge.size * key : edge.size * (key + 1)]
            written = torch.cat([part[0] for part in parts])
            written_slab = torch.cat([part[1] for part in parts])
            order, written_order = storage.argsort(), written.argsort()
            if not (
                torch.equal(storage[order], written[written_order])
                and torch.equal(slab[order], written_slab[written_order])
                and torch.equal(slab.sort().values, every)
            ):
                raise edge.fallback(
                    f"C2: producer tasks {edge.size * key}..{edge.size * (key + 1) - 1}"
                    f" do not write exactly what consumer task {key} reads"
                )


def _task_elements(
    edge: ClusterEdge,
    kind: str,
    domains: Sequence[CoordinateDomain],
    geometries: Sequence[tuple[tuple[int, ...], Mapping[int, object], object]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per task of the edge's store or load root (task order), the storage
    offsets of the elements it touches and their slab offsets."""
    if kind == "store":
        access, strides, ranges = (
            edge.store,
            edge.slab.store_strides,
            edge.slab.store_ranges,
        )
    else:
        access, strides, ranges = (
            edge.load,
            edge.slab.load_strides,
            edge.slab.load_ranges,
        )
    axis_order, counts, _ = geometries[access.root]
    if not all(isinstance(counts[axis], int) for axis in axis_order):
        raise edge.fallback("C2: a dynamic task count")
    tasks = math.prod(counts[axis] for axis in axis_order)  # pyrefly: ignore [no-matching-overload]
    if tasks * edge.numel > _MAX_CHECKED_ELEMENTS * edge.size:
        raise edge.fallback("C2: too many elements to check")
    intervals = [
        _access_interval_expression(
            access, position=position, domain=domains[access.root]
        )
        for position in range(len(access.subscript_dims))
    ]
    tensor_strides = _static(access.tensor_strides)
    # Offsets relative to the load's view (the allocation's offset may be
    # dynamic; the views' difference is not).
    storage_offset = _static(
        [sympy.sympify(access.storage_offset) - sympy.sympify(edge.load.storage_offset)]
    )
    if None in intervals:
        raise edge.fallback(
            f"C2: the {kind}'s interval of subscript {intervals.index(None)} is unknown"
        )
    if tensor_strides is None or storage_offset is None:
        raise edge.fallback(
            f"C2: the {kind}'s strides or offset from the load's view are dynamic"
        )
    out = []
    for task in range(tasks):
        coordinates: dict[sympy.Symbol, int] = {}
        multiplier = 1
        for axis in axis_order:
            count = counts[axis]
            assert isinstance(count, int)
            coordinates[coordinate_axis_symbol(axis)] = task // multiplier % count
            multiplier *= count
        storage = torch.tensor([storage_offset[0]])
        slab = torch.tensor([0])
        for position, interval in enumerate(intervals):
            assert interval is not None
            begin, end = (int(sympy.sympify(e).subs(coordinates)) for e in interval)
            index = torch.arange(begin, end)
            dim = access.subscript_dims[position]
            relative = index - begin if ranges[position] else index
            storage = (storage[:, None] + index * tensor_strides[dim]).flatten()
            slab = (slab[:, None] + relative * strides[position]).flatten()
        out.append((storage, slab))
    return out
