"""Device helpers that shorten the tcgen05 kernel prologue.

Two things cost the TMA-load warp of a one-tile-per-CTA tcgen05 kernel a few
hundred nanoseconds between CTA entry and its first TMA issue, which is the
critical path of the CTAs the hardware dispatches last (their prologue runs
after the dependent-launch release):

* ``cutlass.Int32`` ``//`` and ``%`` lower to Python floor semantics
  (``arith.floordivsi`` / ``arith.remsi`` plus sign fix-ups, six to eight
  uniform-datapath instructions each).  Tile indices, tile offsets and virtual
  pids are never negative, so :func:`floordiv_nonneg` / :func:`mod_nonneg`
  emit a shift / mask for power-of-two divisors and the truncating
  ``arith.divsi`` / ``arith.remsi`` otherwise.
* The DSL pipeline constructors initialize every mbarrier from one elected
  thread of warp 0 in a serial loop (18 barriers for a 7-stage AB ring and a
  2-stage accumulator pipeline), and every warp joins the cluster rendezvous
  behind that chain.  :func:`make_tma_umma_pipeline` /
  :func:`make_umma_async_pipeline` build the pipeline objects without the
  init and :func:`init_pipeline_barriers` spreads the inits over the warps
  the caller names (the epilogue warps are idle during the prologue), the
  way cuBLAS's nvjet kernels do.
"""

from __future__ import annotations

import dataclasses
from functools import cache
import inspect
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import cutlass
from cutlass._mlir.dialects import arith
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cutlass_dsl import if_generate
from cutlass.pipeline import CooperativeGroup
from cutlass.pipeline import PipelineOp
from cutlass.pipeline import PipelineTmaUmma
from cutlass.pipeline import PipelineUmmaAsync
from cutlass.pipeline.helpers import MbarrierArray
from cutlass.pipeline.helpers import MbarrierLayout
from cutlass.pipeline.helpers import _get_thread_arrive_count
from cutlass.pipeline.profiling import register_barrier
from packaging.version import Version

from .cutedsl_compat import installed_cute_dsl_version

if TYPE_CHECKING:
    from collections.abc import Sequence

# The DSL releases whose private pipeline surface the constructors below
# mirror: ``PipelineTmaUmma.create`` / ``PipelineUmmaAsync.create`` minus the
# serial warp-0 mbarrier init (``MbarrierArray.__init__``'s fields and
# validation, the ``_compute_*`` mask helpers, ``_get_thread_arrive_count``,
# ``MbarrierLayout.V0``, ``register_barrier``).  4.7.1 (what fbcode runs)
# ships ``cutlass/pipeline`` byte-identical to 4.7.0.
SPLIT_INIT_VALIDATED_DSL_VERSIONS = frozenset({Version("4.7.0"), Version("4.7.1")})
# The exact surface the mirror was lifted from: ``MbarrierArray.__init__``'s
# parameters and the two pipeline dataclasses' field order (positional
# construction below).
_MBARRIER_ARRAY_INIT_PARAMETERS = (
    "self",
    "barrier_storage",
    "num_stages",
    "agent",
    "tx_count",
    "mbarrier_layout",
    "name",
    "loc",
    "ip",
)
_PIPELINE_TMA_UMMA_FIELDS = (
    "sync_object_full",
    "sync_object_empty",
    "num_stages",
    "producer_mask",
    "consumer_mask",
    "is_leader_cta",
    "cta_group",
)
_PIPELINE_UMMA_ASYNC_FIELDS = (
    "sync_object_full",
    "sync_object_empty",
    "num_stages",
    "producer_mask",
    "consumer_mask",
    "cta_group",
)


@cache
def _private_surface_matches(installed: Version | None) -> bool:
    """Whether ``installed`` is a validated DSL release and the loaded DSL
    still exposes exactly the private surface the constructors mirror.  When
    it does not, :func:`split_barrier_init_supported` is False and the codegen
    falls back to the DSL constructors (``create(..., defer_sync=True)``, the
    serial warp-0 inits; ``cute_mma`` ``tcgen05_split_barrier_init``)."""
    if installed not in SPLIT_INIT_VALIDATED_DSL_VERSIONS:
        return False
    # The DSL pins a documentation signature ``(self)`` on the decorated
    # ``__init__``; the wrapped function carries the real one.
    init_parameters = tuple(
        inspect.signature(inspect.unwrap(MbarrierArray.__init__)).parameters
    )
    if init_parameters != _MBARRIER_ARRAY_INIT_PARAMETERS:
        return False
    tma_umma_fields = tuple(f.name for f in dataclasses.fields(PipelineTmaUmma))
    umma_async_fields = tuple(f.name for f in dataclasses.fields(PipelineUmmaAsync))
    if (
        tma_umma_fields != _PIPELINE_TMA_UMMA_FIELDS
        or umma_async_fields != _PIPELINE_UMMA_ASYNC_FIELDS
    ):
        return False
    helpers = (
        PipelineTmaUmma._compute_mcast_arrival_mask,
        PipelineTmaUmma._compute_is_leader_cta,
        PipelineUmmaAsync._compute_tmem_sync_mask,
        PipelineUmmaAsync._compute_peer_cta_rank,
        _get_thread_arrive_count,
        register_barrier,
        MbarrierLayout.V0,
    )
    return all(helper is not None for helper in helpers)


def split_barrier_init_supported() -> bool:
    """Whether the clustered kernels may build their pipelines through the
    no-init constructors below.  Probed on first use and cached per installed
    DSL version (never at import), so a process that patches the version
    probe gets the answer for the version it reports while the patch is in
    effect, and the real one afterwards."""
    return _private_surface_matches(installed_cute_dsl_version())


def _const_like(value: int, like: object, *, loc: object, ip: object) -> object:
    """An ``arith.constant`` of ``like``'s integer type."""
    return arith.constant(
        cast("Any", like).type,
        value,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def floordiv_nonneg(
    x: object, d: object, *, loc: object = None, ip: object = None
) -> object:
    """``x // d`` for ``x >= 0`` and a positive divisor ``d``.

    A power-of-two Python ``d`` is a right shift, another Python ``d`` the
    truncating ``arith.divsi`` (equal to the floor division when ``x >= 0``);
    anything else keeps the DSL's ``//``.
    """
    if isinstance(x, int) or not isinstance(d, int):
        return cast("Any", x) // d
    numeric = cast("Any", x)
    if d == 1:
        return numeric
    if d & (d - 1) == 0:
        return numeric >> (d.bit_length() - 1)
    value = numeric.ir_value(loc=loc, ip=ip)
    return type(numeric)(
        arith.divsi(value, _const_like(d, value, loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def mod_nonneg(
    x: object, d: object, *, loc: object = None, ip: object = None
) -> object:
    """``x % d`` for ``x >= 0`` and a positive divisor ``d`` (see
    :func:`floordiv_nonneg`)."""
    if isinstance(x, int) or not isinstance(d, int):
        return cast("Any", x) % d
    numeric = cast("Any", x)
    if d == 1:
        return type(numeric)(0)
    if d & (d - 1) == 0:
        return numeric & (d - 1)
    value = numeric.ir_value(loc=loc, ip=ip)
    return type(numeric)(
        arith.remsi(value, _const_like(d, value, loc=loc, ip=ip), loc=loc, ip=ip)
    )


def _uninitialized_mbarrier_array(
    barrier_storage: cute.Pointer,
    num_stages: int,
    agent: tuple[PipelineOp, CooperativeGroup],
    tx_count: object = 0,
    *,
    name: str = "",
    loc: object = None,
    ip: object = None,
) -> MbarrierArray:
    """An ``MbarrierArray`` whose barriers the caller initializes later
    (``MbarrierArray.__init__`` initializes them from warp 0 at once); the
    DSL constructor's fields and validation otherwise."""
    assert split_barrier_init_supported(), (
        "the installed CuTe DSL is not the release these constructors mirror"
    )
    if num_stages <= 0:
        raise ValueError("Error: Mbarrier stage count must be greater than 0.")
    op_type, group = agent
    arrive_count = _get_thread_arrive_count(group, loc=loc, ip=ip)
    if isinstance(arrive_count, int) and arrive_count <= 0:
        raise ValueError("Error: Mbarrier arrive count must be greater than 0.")
    if op_type is PipelineOp.TmaLoad and isinstance(tx_count, int) and tx_count < 0:
        raise ValueError(
            "Error: Mbarrier tx count must not be less than 0 for TMA ops."
        )
    array = cast("Any", object.__new__(MbarrierArray))
    array.barrier_storage = barrier_storage
    array.tx_count = tx_count
    array.num_stages = num_stages
    array.op_type, array.cg = op_type, group
    array.arrive_count = arrive_count
    array.mbarrier_layout = MbarrierLayout.V0
    array.name = name
    array.mbarrier_base = barrier_storage
    return array


@dsl_user_op
def make_tma_umma_pipeline(
    *,
    num_stages: int,
    producer_group: CooperativeGroup,
    consumer_group: CooperativeGroup,
    tx_count: object,
    barrier_storage: cute.Pointer,
    cta_layout_vmnk: cute.Layout,
    name: str = "",
    loc: object = None,
    ip: object = None,
) -> PipelineTmaUmma:
    """``PipelineTmaUmma.create(..., defer_sync=True)`` without the mbarrier
    inits: pair with :func:`init_pipeline_barriers` and the caller's
    ``pipeline_init_arrive`` / ``pipeline_init_wait``.  The DSL constructor's
    validation and ``register_barrier`` (profiling) semantics otherwise."""
    storage = barrier_storage.align(min_align=8)
    sync_object_full = _uninitialized_mbarrier_array(
        storage,
        num_stages,
        (PipelineOp.TmaLoad, producer_group),
        tx_count,
        name=f"{name}.full" if name else name,
        loc=loc,
        ip=ip,
    )
    sync_object_empty = _uninitialized_mbarrier_array(
        storage + num_stages,
        num_stages,
        (PipelineOp.TCGen05Mma, consumer_group),
        name=f"{name}.empty" if name else name,
        loc=loc,
        ip=ip,
    )
    if name:
        register_barrier(name, num_stages, "PipelineTmaUmma")
    if cute.size(cta_layout_vmnk, loc=loc, ip=ip) == 1:
        producer_mask = None
        is_leader_cta = True
    else:
        producer_mask = PipelineTmaUmma._compute_mcast_arrival_mask(
            cta_layout_vmnk, (1, 1), loc=loc, ip=ip
        )
        is_leader_cta = PipelineTmaUmma._compute_is_leader_cta(
            cta_layout_vmnk, loc=loc, ip=ip
        )
    cta_group = (
        cute.nvgpu.tcgen05.CtaGroup.ONE
        if cute.size(cta_layout_vmnk, mode=[0], loc=loc, ip=ip) == 1
        else cute.nvgpu.tcgen05.CtaGroup.TWO
    )
    return PipelineTmaUmma(
        sync_object_full,
        sync_object_empty,
        num_stages,
        producer_mask,
        producer_mask,
        is_leader_cta,
        cta_group,
    )


@dsl_user_op
def make_umma_async_pipeline(
    *,
    num_stages: int,
    producer_group: CooperativeGroup,
    consumer_group: CooperativeGroup,
    barrier_storage: cute.Pointer,
    cta_layout_vmnk: cute.Layout,
    name: str = "",
    loc: object = None,
    ip: object = None,
) -> PipelineUmmaAsync:
    """``PipelineUmmaAsync.create(..., defer_sync=True)`` without the mbarrier
    inits (see :func:`make_tma_umma_pipeline`)."""
    storage = barrier_storage.align(min_align=8)
    sync_object_full = _uninitialized_mbarrier_array(
        storage,
        num_stages,
        (PipelineOp.TCGen05Mma, producer_group),
        name=f"{name}.full" if name else name,
        loc=loc,
        ip=ip,
    )
    sync_object_empty = _uninitialized_mbarrier_array(
        storage + num_stages,
        num_stages,
        (PipelineOp.AsyncThread, consumer_group),
        name=f"{name}.empty" if name else name,
        loc=loc,
        ip=ip,
    )
    if name:
        register_barrier(name, num_stages, "PipelineUmmaAsync")
    if cute.size(cta_layout_vmnk, loc=loc, ip=ip) == 1:
        producer_mask = None
    else:
        producer_mask = PipelineUmmaAsync._compute_tmem_sync_mask(
            cta_layout_vmnk, loc=loc, ip=ip
        )
    if cute.size(cta_layout_vmnk, mode=[0], loc=loc, ip=ip) == 1:
        consumer_mask = None
        cta_group = cute.nvgpu.tcgen05.CtaGroup.ONE
    else:
        consumer_mask = PipelineUmmaAsync._compute_peer_cta_rank(loc=loc, ip=ip)
        cta_group = cute.nvgpu.tcgen05.CtaGroup.TWO
    return PipelineUmmaAsync(
        sync_object_full,
        sync_object_empty,
        num_stages,
        producer_mask,
        consumer_mask,
        cta_group,
    )


@dsl_user_op
def init_pipeline_barriers(
    warp_idx: object,
    warps: Sequence[int],
    *pipelines: object,
    loc: object = None,
    ip: object = None,
) -> None:
    """Initialize the mbarriers of ``pipelines`` (built by the constructors
    above), spreading their full / empty arrays round-robin over ``warps``;
    one elected thread per warp issues its arrays' ``mbarrier.init``.  The
    caller fences and synchronizes afterwards (``pipeline_init_arrive`` /
    ``pipeline_init_wait``), exactly as after the DSL's own constructors."""
    arrays: list[Any] = []
    for pipeline in pipelines:
        arrays.extend(
            (
                cast("Any", pipeline).sync_object_full,
                cast("Any", pipeline).sync_object_empty,
            )
        )
    arrays.sort(key=lambda array: -array.num_stages)
    per_warp: dict[int, list[MbarrierArray]] = {warp: [] for warp in warps}
    loads = dict.fromkeys(warps, 0)
    for array in arrays:
        warp = min(warps, key=lambda w: loads[w])
        per_warp[warp].append(array)
        loads[warp] += array.num_stages
    for warp, owned in per_warp.items():
        if not owned:
            continue

        def then_body(owned: list[MbarrierArray] = owned) -> None:
            with cute.arch.elect_one(loc=loc, ip=ip):
                for array in owned:
                    for index in range(array.num_stages):
                        cute.arch.mbarrier_init(
                            array.get_barrier(index, loc=loc, ip=ip),
                            array.arrive_count,
                            loc=loc,
                            ip=ip,
                        )

        if_generate(warp_idx == cutlass.Int32(warp), then_body, loc=loc, ip=ip)


@dsl_user_op
def init_pipeline_barriers_pair(
    warp_idx: object,
    first_warp: int,
    pipeline: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """Initialize one pipeline's full barriers from warp ``first_warp`` and its
    empty barriers from warp ``first_warp + 1`` inside ONE guarded block: the
    two arrays are contiguous (``storage`` and ``storage + num_stages``) and
    the warp selects its base and arrive count arithmetically, so the other
    warps skip a single branch (each skipped block costs the TMA warp a far
    branch) while the init chain of any one warp halves."""
    full = cast("Any", pipeline).sync_object_full
    empty = cast("Any", pipeline).sync_object_empty
    num_stages = full.num_stages
    assert empty.num_stages == num_stages
    which = warp_idx - cutlass.Int32(first_warp)

    def then_body() -> None:
        base = full.mbarrier_base + which * num_stages
        count = cutlass.Int32(full.arrive_count) + which * (
            cutlass.Int32(empty.arrive_count) - cutlass.Int32(full.arrive_count)
        )
        with cute.arch.elect_one(loc=loc, ip=ip):
            for index in range(num_stages):
                cute.arch.mbarrier_init(base + index, count, loc=loc, ip=ip)

    if_generate(
        (warp_idx >= cutlass.Int32(first_warp))
        & (warp_idx <= cutlass.Int32(first_warp + 1)),
        then_body,
        loc=loc,
        ip=ip,
    )
