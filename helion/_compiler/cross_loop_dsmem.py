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
The global store stays, ordered by nothing in the launch: the allocation holds
what it held.  The split count the source writes (the slab's partials) is the
cluster size; the entry only picks the transport.

Candidates (``TileDependencyGraph.dsmem_candidate_ids``) do not depend on the
config.  Every size a rule reads takes this config's value
(``CompileEnvironment.config_value``: block sizes, tunables and host helpers
over them), never a size hint.  ``ConfigSpec.normalize`` turns off the entries
whose slab facts (``dsmem_slab_facts``) break a rule of the config alone (the
cluster size, the slabs' bytes, one cluster size for every entry), and
codegen checks the rest; an enabled entry that fails regenerates the kernel
with it off (``DsmemFallback``, logged), and a reason of the whole config
(the pipeline, packet dispatch, continuations, the cluster occupancy query)
turns off every entry at once:

- C1 (the slab, ``dsmem_transports``): the store and the load run once a task
  (in their root's own graph, outside any loop or branch) and at most 8 CTAs
  make a cluster; the load and the store index whole dimensions (full slices),
  tiles of a root axis and single elements (the load's are its key); the ranges
  the load reads are strided runs that do not overlap, and every range the
  store writes lies in one of them.  The slab is the load's region in row-major
  run order, at most ``MAX_CLUSTER_SLAB`` bytes for every entry together.
  32-bit elements, shapes and strides static under the config, no masks.
- C2 (the schedule, ``check_dispatch``): task lanes (``cross_loop_pipeline``
  static or dynamic, no packets); both roots resident and unguarded, without
  continuations, claimed tasks, reordered tasks or L2 grouping; the producer
  root a single wave of ``size`` times the consumer's tasks, with a split axis
  of ``size`` tasks: the producer tasks of key ``k`` (``k``'s coordinates on
  the other axes, in their task order) write exactly what consumer task ``k``
  reads, element by element (no element twice), whatever the split axis's
  place in the task order.
- C3 (codegen, ``cluster_layout``): every tile spreads over the threads of the
  partition that moves it without duplicates, so the bytes rank 0 expects
  arrive once.

Dispatch (``cross_loop_codegen``): a producer worker takes the key its
cluster's rank 0 would (the lane rank 0 takes, divided by ``size``; a
bijection for every rotation of the lanes) and the split axis coordinate of
its rank (``split_task``); rank 0 of a cluster takes the consumer task of the
key its cluster runs.  The launch runs as many workers as clusters of
``size`` keep resident (``dsmem_worker_count``), a multiple of ``size``.

Shared memory (``gluon_ops.CLUSTER_SMEM``, declared and initialized at kernel
entry): entry ``i``'s barrier is at byte ``8 + 8 i`` (one arrival at entry
plus the slab's bytes complete phase 0, once a launch), its slab at
``DsmemTransport.buffer``.  Once its barriers are armed, thread 0 raises a
flag at bytes 0 to 7 to the launch's value (a hash of its epoch, launch state
and cluster id with the top bit set: never 0, and matching another kernel's
leftover data only by chance; ``gluon_ops.cluster_flag``, assigned to
``CLUSTER_FLAG``); a producer task polls rank 0's flag before sending
(``CLUSTER_READY``), and every CTA clears its flag at the kernel's end
(``cluster_exit``).  The hardware cluster barrier would need every thread,
which a warp-specialized kernel's padding warps never are.
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
from ..autotuner.config_spec import DsmemLayout
from ..autotuner.config_spec import DsmemSlabFacts
from ..runtime.triton.launcher import cluster_workers
from ..runtime.triton.launcher import get_cluster_capacity
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .compile_environment import FixedBlockSizeSource
from .compile_environment import LoopSpecBlockSizeSource
from .host_function import HostFunction
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
    from .tile_dependency import TileAccess
    from .tile_dependency import TileDependencyGraph

# The token a producer task holds once its cluster's rank 0 has armed its
# barriers (rank 0's flag, polled at the task's start), and the statement that
# raises the flag (``gluon_ops.cluster_flag``).
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
    """A reason of the whole config: every enabled entry goes off at once."""
    return DsmemFallback((t.index for t in transports.values()), rule)


def dsmem_transports(device_function: DeviceFunction) -> dict[int, DsmemTransport]:
    """The transports this config enables, by allocation (C1 checked), from
    the dependency graph and the config alone, so the roots' codegen and the
    schedule agree."""
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
    """The launch's cluster size: every transport's (``_select`` keeps one
    size), 1 without transports."""
    return next(iter(transports.values())).size if transports else 1


def cluster_smem_bytes(transports: Mapping[int, DsmemTransport]) -> int:
    """The shared memory the transports declare: a multiple of 1024 bytes, so
    the dynamic shared memory after it keeps its alignment."""
    end = max(t.buffer + t.nbytes for t in transports.values())
    return (end + 1023) // 1024 * 1024


def cluster_exit() -> ast.stmt:
    """The kernel's last statement, clearing this CTA's flag
    (``gluon_ops.cluster_exit``)."""
    return statement_from_string("_helion_gluon.cluster_exit()")


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
    for index, (allocation_id, flag) in enumerate(
        zip(graph.dsmem_candidate_ids, flags, strict=True)
    ):
        if not flag:
            continue
        store, load = graph.dsmem_accesses(allocation_id)
        name = load.tensor_name or f"allocation {allocation_id}"
        refusal = _static_refusal(store, load, device_ir.root_ids)
        if refusal is not None:
            raise DsmemFallback((index,), f"{name}: C1: {refusal}")
        slab = _slab(
            store,
            load,
            lambda axis: _block_size(axis, config),
            lambda value: config_int(value, config),
        )
        if isinstance(slab, str):
            raise DsmemFallback((index,), f"{name}: C1: {slab}")
        refusal = layout.refusal(slab.size, slab.nbytes)
        if refusal is not None:
            raise DsmemFallback((index,), f"{name}: C1: {refusal}")
        buffer, barrier = layout.take(slab.size, slab.nbytes)
        transports[allocation_id] = DsmemTransport(
            index=index,
            allocation_id=allocation_id,
            name=name,
            store=store,
            load=load,
            slab=slab,
            buffer=buffer,
            barrier=barrier,
        )
    return transports


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
    store: TileAccess, load: TileAccess, root_ids: Sequence[int]
) -> str | None:
    """The C1 rule the store and the load break under every config, or None."""
    if any(access.graph_id != root_ids[access.root] for access in (store, load)):
        return "an access in a loop or branch"
    if load.dtype is None or load.dtype.itemsize != 4 or store.dtype != load.dtype:
        return "not 32-bit elements"
    if store.has_explicit_mask or load.has_explicit_mask:
        return "a masked access"
    return None


def _slab(
    store: TileAccess,
    load: TileAccess,
    block_size: Callable[[int], int | None],
    value_of: Callable[[object], int | None],
) -> _Slab | str:
    """C1's slab of the store and the load (which ``_static_refusal``
    accepts), or what fails."""
    store_shape = _static(store.tensor_shape, value_of)
    load_shape = _static(load.tensor_shape, value_of)
    store_stride = _static(store.tensor_strides, value_of)
    load_stride = _static(load.tensor_strides, value_of)
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
        store_ranges=tuple(span is not None for span in store_spans),
        load_ranges=tuple(span is not None for span in load_spans),
        numel=numel,
        size=numel // tile,
        vec=min(vector(store_strides, store_spans), vector(load_strides, load_spans)),
    )


def dsmem_slab_facts(
    graph: TileDependencyGraph, root_ids: Sequence[int]
) -> tuple[DsmemSlabFacts | None, ...]:
    """Per candidate, the factors of its slab's elements (the load's ranges)
    and of the tile one producer task stores (the store's ranges) that
    ``ConfigSpec.normalize`` multiplies out under a config: each an integer,
    a tunable expression or a block's index into ``block_sizes``; None when
    a factor has no such form, which leaves the candidate to codegen.  A
    candidate that breaks a C1 rule under every config is never carried."""
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
    for allocation_id in graph.dsmem_candidate_ids:
        store, load = graph.dsmem_accesses(allocation_id)
        if _static_refusal(store, load, root_ids) is not None:
            facts.append(DsmemSlabFacts(numel=(), tile=(), carried=False))
            continue
        numel, tile = factors(load), factors(store)
        facts.append(
            None
            if numel is None or tile is None
            else DsmemSlabFacts(numel=numel, tile=tile)
        )
    return tuple(facts)


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
) -> dict[int, int]:
    """C2's counts and elements (the caller checks the roots' dispatch): per
    allocation, the multiplier of its producer root's split axis
    (``split_task``), one axis for all allocations of a root."""
    # Per producer root, the split axes every allocation so far accepts.
    accepted: dict[int, list[int]] = {}
    allocations: dict[int, list[int]] = {}
    with unset_fake_temporarily():
        roles: dict[int, tuple[bool, int]] = {}
        for transport in transports.values():
            producer, consumer = transport.store.root, transport.load.root
            size = transport.size
            for root, role in (
                (producer, (False, consumer)),
                (consumer, (True, producer)),
            ):
                if roles.setdefault(root, role) != role:
                    raise transport.fallback("C2: a root in two cluster roles")
            # ``dsmem_worker_count`` keeps every worker resident in whole
            # clusters.
            assert workers % size == 0, (workers, size)
            # The task counts before the elements, which cost substitutions a
            # task.
            producers = _task_count(transport, producer, geometries)
            consumers = _task_count(transport, consumer, geometries)
            if producers > workers:
                raise transport.fallback("C2: a producer root of more than one wave")
            if producers != size * consumers:
                raise transport.fallback(
                    f"C2: {producers} producer tasks for {consumers} consumer tasks"
                )
            stored = _task_elements(transport, "store", config, domains, geometries)
            loaded = _task_elements(transport, "load", config, domains, geometries)
            every = torch.arange(transport.numel, device="cpu")
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
                    for key, (storage, slab) in enumerate(loaded)
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
            allocations.setdefault(producer, []).append(transport.allocation_id)
    return {
        allocation: accepted[producer][0]
        for producer, ids in allocations.items()
        for allocation in ids
    }


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
    """Per task of the transport's store or load root (task order), the
    storage offsets of the elements it touches and their slab offsets."""
    if kind == "store":
        access, strides, ranges = (
            transport.store,
            transport.slab.store_strides,
            transport.slab.store_ranges,
        )
    else:
        access, strides, ranges = (
            transport.load,
            transport.slab.load_strides,
            transport.slab.load_ranges,
        )
    axis_order, counts, _ = geometries[access.root]
    tasks = _task_count(transport, access.root, geometries)
    # A producer task touches one part of the slab, a consumer task all of it.
    touched = transport.numel // transport.size if kind == "store" else transport.numel
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
        coordinates: dict[sympy.Symbol, int] = {}
        multiplier = 1
        for axis in axis_order:
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
