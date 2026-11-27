"""Two independent attention heads per CTA with shared conversion warpgroups.

Each 128-thread compute group owns a 256-column TMEM slab. Four K and four V
conversion warps alternate the two heads; the controls issue QK, TMA and PV,
and optionally classify value metadata.
Physical warp modulo four selects the TMEM row quarter, so all eight converter
warps keep their full groups. Named barriers 1/2 and 3/4 belong to the separate
compute groups. Register quotas stay inside disjoint role branches: 144 for
compute, 72 for conversion and 48 for all four control warps (61,440 total).
Optional barrier 5 retires the allocation before output stores: both 128-thread
compute groups arrive after their final reads and the 32-thread QK role frees it.
Every role then returns when its own work ends; without that retirement, or
when the caller needs one common end, all 640 threads join before the release.
Barrier 6 is the setup rendezvous. The loader warp initializes the pipeline
barriers and only arrives, so its page stream starts while the other roles
finish their setup; the QK role owns the query barrier and copies.
With a key clip the loader issues each boundary tile's K, V and scale boxes at
the key offset the compute groups derive for that tile's mask. Both sides
compute that offset from kernel arguments and compile-time constants alone, so
the loader needs nothing the setup rendezvous publishes.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as experimental_cuda
import cutlass.experimental.primitives as prims
import cutlass.utils as utils

from ._mma_dag_tcgen05_runtime import mbar_wait
from ._mma_dag_tcgen05_runtime import mbar_wait_sleep
from ._paged_attention_compute import logical_attention_key_shift
from ._paged_attention_compute import normalize_output
from ._paged_attention_compute import output_normalization_factor
from ._paged_attention_io import load_query_element
from ._paged_attention_io import store_output_element
from ._paged_attention_io import store_row_element
from ._paged_attention_partial_compute import load_query_panel
from ._paged_attention_partial_compute import mma_128xnx128
from ._paged_attention_partial_compute import online_softmax_per_key
from ._paged_attention_partial_compute import store_query_panel
from ._paged_attention_partial_pipeline import attention_page_coordinates
from ._paged_attention_reference_compute import online_softmax_reference
from ._paged_attention_runtime import _load_words4
from ._paged_attention_runtime import _packed_scale_layout_supported
from ._paged_attention_runtime import enqueue_paged_query
from ._paged_attention_runtime import enqueue_paged_tma_operand
from ._paged_attention_runtime import stage_positive_zero_value_tile
from ._paged_attention_runtime import stage_scaled_s8_rows_to_tmem
from ._paged_attention_runtime import stage_scaled_s8_transpose_to_tmem
from .device.warp_reduce import warp_row_sum


@cute.jit
def _wait_non_mma(
    barrier: cute.Pointer,
    phase: cutlass.Int32 | cutlass.Int64,
    sleep_waits: cutlass.Constexpr,
) -> None:
    """Keep issuer waits prompt while optionally suspending the other roles."""
    if cutlass.const_expr(sleep_waits):
        mbar_wait_sleep(barrier, phase)
    else:
        mbar_wait(barrier, phase)


@cute.jit
def _compute_head(
    output: cute.Tensor,
    lse: cute.Tensor | None,
    probability: cute.Tensor,
    state: cute.Tensor,
    scratch: cute.Tensor,
    qk_done: cute.Pointer,
    score_free: cute.Pointer,
    probability_ready: cute.Pointer,
    pv_done: cute.Pointer,
    base: cutlass.Int32,
    tile: cutlass.Int32,
    pair_head: cutlass.Int32,
    first_page: cutlass.Int64,
    last_page: cutlass.Int64,
    count: cutlass.Int64,
    length: cutlass.Int32,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    query_rows: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    window: cutlass.Constexpr,
    score_scale: cutlass.Float32,
    fast_math: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    io_layouts: cutlass.Constexpr,
    reference_scores: cutlass.Constexpr,
    base2_scores: cutlass.Constexpr,
    reflected_dtype: cutlass.Constexpr,
    output_reciprocal: cutlass.Constexpr,
    output_scale_log2: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
    capture_pdl_wait_mode: cutlass.Int32,
    maximum_output: cute.Tensor | None,
    denominator_output: cute.Tensor | None,
    statistic_layouts: cutlass.Constexpr,
    coefficients_capped: cutlass.Constexpr,
    probability_schedule: cutlass.Constexpr,
    output_offset: cutlass.Constexpr,
    rowwise_rescale_threshold: cutlass.Constexpr,
    early_tmem_retirement: cutlass.Constexpr,
    headwide_rescale: cutlass.Constexpr,
    key_clip: cutlass.Constexpr,
) -> None:
    cute.arch.setmaxregister_increase(144)
    thread = cute.arch.thread_idx()[0] % 128
    warp = cute.arch.make_warp_uniform(thread // 32)
    lane = thread % 32
    output_address = (warp * 32 << 16) | (base + output_offset)
    denominator_address = None
    if cutlass.const_expr(probability_schedule == "early_tmem"):
        denominator_address = (warp * 32 << 16) | (base + 128 + 2 * query_columns)
    zero = cutlass.Array(cutlass.Float32, query_columns, alignment=16)
    for query in cutlass.range_constexpr(query_columns):
        zero[query] = cutlass.Float32(0)
    store_query_panel(output_address, zero, query_columns)
    if cutlass.const_expr(probability_schedule == "early_tmem"):
        store_query_panel(denominator_address, zero, query_columns)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=2 + pair_head * 2, number_of_threads=128)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    partial_columns = 1 if probability_schedule == "early_tmem" else query_columns
    # The TMEM schedule loads its real partials inside the recurrence. Keep a
    # typed placeholder for CuTe's shared call signature; it emits no storage.
    partials = cute.make_rmem_tensor((partial_columns,), cutlass.Float32)
    for query in cutlass.range_constexpr(partial_columns):
        partials[query] = cutlass.Float32(0)
    reference_invalid = cutlass.Boolean(True)
    for step in cutlass.range(count):
        _logical_page, logical_key_base = attention_page_coordinates(
            first_page, last_page, step, reflected_dtype
        )
        _wait_non_mma(qk_done, step & 1, sleep_waits)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        if cutlass.const_expr(reference_scores):
            maximum_changed, reference_invalid = online_softmax_reference(
                base + 128,
                probability,
                state,
                scratch,
                thread,
                logical_key_base,
                length,
                query_len,
                group,
                window,
                score_scale,
                1 + pair_head * 2,
                partials,
                reference_invalid,
                fast_math,
                score_free,
                coordinate_dtype,
                query_columns,
                base2_scores,
                convergent_branches=True,
                group_barrier=False,
                probability_schedule=probability_schedule,
                denominator_address=denominator_address,
                previous_pv=pv_done,
                previous_phase=(step - 1) & 1,
                have_previous=step > 0,
                previous_sleep=sleep_waits,
                rowwise_rescale_threshold=rowwise_rescale_threshold,
                headwide_rescale=headwide_rescale,
                key_clip=key_clip,
            )
        else:
            # Only reference-score publication opts out here. Keep the per-key
            # helper's default group barrier until that elision is measured.
            maximum_changed = online_softmax_per_key(
                base + 128,
                probability,
                state,
                scratch,
                thread,
                logical_key_base,
                length,
                query_len,
                group,
                window,
                score_scale,
                1 + pair_head * 2,
                partials,
                fast_math,
                score_free,
                coordinate_dtype,
                query_columns,
                base2_scores,
                coefficients_capped,
                key_clip=key_clip,
            )
        if maximum_changed:
            previous = load_query_panel(output_address, query_columns)
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            rescaled = cutlass.Array(cutlass.Float32, query_columns, alignment=16)
            for query in cutlass.range_constexpr(query_columns):
                rescaled[query] = previous[query] * state[2, query]
            store_query_panel(output_address, rescaled, query_columns)
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        # Each warp owns its P columns and O rows. The softmax helper's
        # proxy fence and the TMEM fence/sync below publish both through
        # four arrivals. Serial waits below; early schedules wait for prior
        # PV inside the next softmax call before rewriting either panel.
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.sync_warp()
        if lane == 0:
            prims.mbarrier_arrive(probability_ready)
        if cutlass.const_expr(probability_schedule == "serial"):
            _wait_non_mma(pv_done, step & 1, sleep_waits)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    # The loop leaves the final denominators. In the nonserial schedules its
    # last PV may still be accumulating O: reduce the denominators inside that
    # wait, since the reduction reads neither shared P nor O.
    if cutlass.const_expr(probability_schedule == "early_tmem"):
        partials = load_query_panel(denominator_address, query_columns)
    # Transpose row ownership while retaining the exact ascending-XOR sum tree.
    if cutlass.const_expr(query_columns >= 32):
        row, row_total = warp_row_sum([partials[query] for query in range(32)], lane)
        scratch[warp, row] = row_total
    if cutlass.const_expr(query_columns == 8 or query_columns == 40):
        offset = query_columns - 8
        tail_row, tail_total = warp_row_sum(
            [partials[offset + query] for query in range(8)], lane
        )
        if lane < 8:
            scratch[warp, offset + tail_row] = tail_total
    cute.arch.barrier(barrier_id=1 + pair_head * 2, number_of_threads=128)
    total = cutlass.Float32(0)
    if thread < query_columns:
        total = cutlass.Float32(scratch[0, thread]) + cutlass.Float32(
            scratch[1, thread]
        )
        total = total + cutlass.Float32(scratch[2, thread])
        total = total + cutlass.Float32(scratch[3, thread])
        state[1, thread] = total
        if cutlass.const_expr(maximum_output is None):
            safe_total = total
            if total == 0:
                safe_total = cutlass.Float32(1)
            state[3, thread] = output_normalization_factor(
                safe_total, output_reciprocal, output_scale_log2=output_scale_log2
            )
    cute.arch.barrier(barrier_id=1 + pair_head * 2, number_of_threads=128)
    if cutlass.const_expr(probability_schedule != "serial"):
        if count > 0:
            _wait_non_mma(pv_done, (count - 1) & 1, sleep_waits)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    # O is final once the last PV completed (or was zeroed for an empty range).
    result = load_query_panel(output_address, query_columns)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    if cutlass.const_expr(early_tmem_retirement):
        # Both heads finish their last PV and O reads before either slab is
        # freed. Output math/stores use only the retained register copy.
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_arrive(5, thread_count=288)
    # All global stores of this audited schedule are below this point. Even an
    # empty attention interval reaches it. Other roles only read global memory.
    if capture_pdl_wait_mode != 0:
        cute.arch.griddepcontrol_wait()
    for query in cutlass.range_constexpr(query_columns):
        if cutlass.const_expr(query < query_rows):
            if cutlass.const_expr(maximum_output is not None):
                value = cutlass.Float32(result[query])
            else:
                value = cutlass.BFloat16(
                    normalize_output(result[query], state[3, query], output_reciprocal)
                )
            store_output_element(
                output,
                tile,
                query,
                thread,
                value,
                heads,
                group,
                io_layouts,
            )
    if cutlass.const_expr(maximum_output is not None):
        assert maximum_output is not None and denominator_output is not None
        if thread < query_rows:
            assert cutlass.const_expr(statistic_layouts is not None)
            store_row_element(
                maximum_output,
                tile,
                thread,
                cutlass.Float32(state[0, thread]),
                heads,
                group,
                statistic_layouts[0],
            )
            store_row_element(
                denominator_output,
                tile,
                thread,
                cutlass.Float32(state[1, thread]),
                heads,
                group,
                statistic_layouts[1],
            )
    if cutlass.const_expr(lse is not None):
        assert lse is not None
        if thread < query_rows:
            maximum = cutlass.Float32(state[0, thread])
            if cutlass.const_expr(base2_scores):
                maximum = maximum * cutlass.Float32(math.log(2))
            store_row_element(
                lse,
                tile,
                thread,
                maximum + cute.math.log(state[1, thread], fastmath=fast_math),
                heads,
                group,
                None if io_layouts is None else io_layouts[2],
            )
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def _prepare_value_metadata(
    scales: cute.Pointer,
    free: cute.Pointer,
    flags: cute.Tensor,
    ready: cute.Pointer,
    source: cute.Tensor,
    step: cutlass.Int64,
    pair_head: cutlass.Int32,
    physical: cutlass.Int32,
    first_head: cutlass.Int32,
    sleep_waits: cutlass.Constexpr,
    read_overlap: cutlass.Constexpr = False,
) -> None:
    """Publish a scale tile, owned until every lane of all four V warps releases it.

    Flags encode copied nonzero scales (0), a positive-zero value tile (1),
    or an invalid descriptor coordinate requiring the original two TMAs (2).
    The source is the proved (key, head, subpage, page) Int32-word view.
    """
    lane = cute.arch.lane_idx()
    ring = 2 * (step & 1) + pair_head
    if cutlass.const_expr(not read_overlap):
        _wait_non_mma(free + ring, (step // 2 & 1) ^ 1, sleep_waits)
    # Match the original TMA's Int32 head arithmetic before widening offsets.
    encoded_head = cutlass.Int32(first_head + pair_head)
    valid = (
        (cutlass.Int64(physical) >= 0)
        & (cutlass.Int64(physical) < cute.size(source, mode=[3]))
        & (cutlass.Int64(encoded_head) >= 0)
        & (cutlass.Int64(encoded_head) < cute.size(source, mode=[1]))
    )
    if valid:
        offset = cute.crd2idx(
            (
                cutlass.Int64(lane) * 4,
                cutlass.Int64(encoded_head),
                cutlass.Int64(0),
                cutlass.Int64(physical),
            ),
            source.layout,
        )
        # All outer strides and the starting key are multiples of four words.
        words = _load_words4((source.iterator + cute.assume(offset, divby=4)).align(16))
        zero = cute.arch.vote_all_sync((words[0] | words[1] | words[2] | words[3]) == 0)
        # Global inputs remain immutable throughout this kernel. Only those
        # reads may overlap the previous consumer; every shared effect below
        # still waits for all 128 reader releases on this exact slot.
        if cutlass.const_expr(read_overlap):
            _wait_non_mma(free + ring, (step // 2 & 1) ^ 1, sleep_waits)
        shared = cute.make_tensor(
            cute.recast_ptr(scales + ring * 512, dtype=cutlass.Int32),
            cute.make_layout((128,)),
        )
        if not zero:
            for index in cutlass.range_constexpr(4):
                shared[lane * 4 + index] = words[index]
            # A later invalid coordinate reuses this slot through TMA's async
            # proxy. Each writer orders its generic stores before that handoff;
            # the warp join and release/acquire barriers alone do not do so.
            cute.arch.fence_proxy(kind="async.shared", space="cta")
        if lane == 0:
            flags[ring] = cutlass.Int32(zero)
    else:
        if cutlass.const_expr(read_overlap):
            _wait_non_mma(free + ring, (step // 2 & 1) ^ 1, sleep_waits)
        if lane == 0:
            flags[ring] = cutlass.Int32(2)
    # Publish each writer directly. The lane-zero release preceded by a warp
    # join passes numerical tests, but its SASS has no WARPSYNC and racecheck
    # reports other lanes' accesses. All 32 arrivals make ownership explicit;
    # this is not evidence of a numerical failure or invalid PTX ordering.
    prims.mbarrier_arrive(ready + ring)


@cute.jit
def _enqueue_operand_tile(
    raw: cute.Pointer,
    scales: cute.Pointer,
    ready: cute.Pointer,
    free: cute.Pointer,
    operand_map: experimental_cuda.TensorMap,
    scale_map: experimental_cuda.TensorMap,
    step: cutlass.Int64,
    pair_head: cutlass.Int32,
    physical: cutlass.Int32,
    first_head: cutlass.Int32,
    l2_hints: cutlass.Constexpr = ("", ""),
    sleep_waits: cutlass.Constexpr = False,
    metadata_ready: cute.Pointer | None = None,
    metadata_flags: cute.Tensor | None = None,
    key: cutlass.Int32 = cutlass.Int32(0),  # noqa: B008
) -> None:
    ring = 2 * (step & 1) + pair_head
    if cutlass.const_expr(metadata_ready is not None):
        # Acquiring metadata also acquires the preceding raw-free phase. The
        # loader retains the original V order for every lag/drain schedule.
        assert metadata_flags is not None
        _wait_non_mma(metadata_ready + ring, step // 2 & 1, sleep_waits)
        flag = metadata_flags[ring]
        if flag == 1:
            prims.mbarrier_arrive(ready + ring)
        elif flag == 2:
            enqueue_paged_tma_operand(
                operand_map,
                scale_map,
                raw + cute.assume(ring * 16384, divby=16384),
                scales + ring * 512,
                ready + ring,
                physical,
                cutlass.Int32(0),
                first_head + pair_head,
                128,
                l2_hints=l2_hints,
            )
        else:
            enqueue_paged_tma_operand(
                operand_map,
                scale_map,
                raw + cute.assume(ring * 16384, divby=16384),
                scales + ring * 512,
                ready + ring,
                physical,
                cutlass.Int32(0),
                first_head + pair_head,
                128,
                l2_hints=l2_hints,
                include_scales=False,
            )
    else:
        _wait_non_mma(free + ring, (step // 2 & 1) ^ 1, sleep_waits)
        enqueue_paged_tma_operand(
            operand_map,
            scale_map,
            raw + cute.assume(ring * 16384, divby=16384),
            scales + ring * 512,
            ready + ring,
            physical,
            cutlass.Int32(0),
            first_head + pair_head,
            128,
            l2_hints=l2_hints,
            key=key,
        )


def _pipeline_barriers(value_metadata: bool) -> tuple[tuple[str, int, int], ...]:
    """The pipeline barriers in block order, as ``(name, width, arrivals)``.

    One barrier per raw ring slot (four) or per head (two), each initialized
    to ``arrivals``; the value-metadata transport adds its four.
    """
    barriers = (
        ("raw_k_ready", 4, 1),
        ("raw_k_free", 4, 4),
        ("raw_v_ready", 4, 1),
        ("raw_v_free", 4, 128 if value_metadata else 4),
        ("key_ready", 2, 4),
        ("key_free", 2, 1),
        ("value_ready", 2, 4),
        ("value_free", 2, 1),
        ("score_free", 2, 4),
        ("score_ready", 2, 1),
        ("probability_ready", 2, 4),
        ("pv_done", 2, 1),
    )
    if value_metadata:
        barriers += (("metadata_ready", 4, 32),)
    return barriers


def _pipeline_barrier_plan(
    value_metadata: bool,
) -> tuple[dict[str, int], int, tuple[tuple[tuple[int, int], ...], ...]]:
    """Where each pipeline barrier lies and how the block is initialized.

    Returns the first slot of every entry of ``_pipeline_barriers``, the
    block's slot count and its initialization rounds. The block is padded to
    whole rounds of 32 so that every lane of the initializing warp executes
    each round's instruction and the rounds never diverge. A round holds the
    arrival counts of its 32 slots as ``(count - 1, signed lane mask)`` terms.
    """
    offsets: dict[str, int] = {}
    counts: list[int] = []
    for name, width, arrivals in _pipeline_barriers(value_metadata):
        offsets[name] = len(counts)
        counts += [arrivals] * width
    counts += [1] * (-len(counts) % 32)
    rounds = []
    for start in range(0, len(counts), 32):
        terms = []
        for value in sorted(set(counts[start : start + 32]) - {1}):
            mask = sum(1 << lane for lane in range(32) if counts[start + lane] == value)
            terms.append((value - 1, mask - (1 << 32) if mask >= 1 << 31 else mask))
        rounds.append(tuple(terms))
    return offsets, len(counts), tuple(rounds)


class _SharedArray(NamedTuple):
    """One shared-memory array of the paired body: its name there, element
    type, element count and byte alignment, and the optional part of the
    body it belongs to (``None``: every body allocates it)."""

    name: str
    dtype: type[cutlass.Numeric]
    count: int
    alignment: int
    guard: str | None = None


def _pipeline_shared_memory(
    query_columns: int, value_metadata: bool
) -> tuple[_SharedArray, ...]:
    """The paired body's shared memory, one entry per array in allocation
    order.

    Four raw K and four raw V pages of 16 KiB with their scales, the query
    and probability panels and the per-head state and scratch rows of both
    heads, the role slot and one block for every pipeline barrier
    (``_pipeline_barrier_plan``). A query map adds its barrier and the
    value-metadata transport its flags. The body takes every array from
    here (``_allocate_pipeline_shared_memory``), so a new array is a new
    entry, and replaying the entries gives what the body occupies.
    """
    panel = 2 * (query_columns * 128)
    state = 2 * (4 * query_columns)
    return (
        _SharedArray("raw_k", cutlass.Int8, 4 * 16384, 1024),
        _SharedArray("raw_v", cutlass.Int8, 4 * 16384, 1024),
        _SharedArray("scale_k", cutlass.Uint8, 4 * 512, 128),
        _SharedArray("scale_v", cutlass.Uint8, 4 * 512, 128),
        _SharedArray("shared_q_raw", cutlass.BFloat16, panel, 1024),
        _SharedArray("probability_raw", cutlass.BFloat16, panel, 1024),
        _SharedArray("state_raw", cutlass.Float32, state, 16),
        _SharedArray("scratch_raw", cutlass.Float32, state, 16),
        _SharedArray("slot", cutlass.Int32, 1, 4),
        _SharedArray(
            "barriers", cutlass.Int64, _pipeline_barrier_plan(value_metadata)[1], 8
        ),
        _SharedArray("query_ready", cutlass.Int64, 1, 8, guard="query_map"),
        _SharedArray("metadata_flags", cutlass.Int32, 4, 16, guard="value_metadata"),
    )


def _allocate_pipeline_shared_memory(
    smem: utils.SmemAllocator, arrays: tuple[_SharedArray, ...], parts: frozenset[str]
) -> SimpleNamespace:
    """Allocate ``arrays`` in table order for a body with the optional
    ``parts``: one attribute per array, ``None`` for those of a part the
    body lacks."""
    return SimpleNamespace(
        **{
            array.name: (
                smem.allocate_array(
                    array.dtype, array.count, byte_alignment=array.alignment
                )
                if array.guard is None or array.guard in parts
                else None
            )
            for array in arrays
        }
    )


def _pipeline_parts(query_map: bool, value_metadata: bool) -> frozenset[str]:
    """The optional parts of a body, as ``_SharedArray.guard`` names them."""
    return frozenset(
        name
        for name, present in (
            ("query_map", query_map),
            ("value_metadata", value_metadata),
        )
        if present
    )


@cute.jit
def paged_attention_paired_body(
    q: cute.Tensor,
    pages: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor | None,
    tile: cutlass.Int32,
    first_page: cutlass.Int64,
    last_page: cutlass.Int64,
    length: cutlass.Int32,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    query_rows: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    window: cutlass.Constexpr,
    score_scale: cutlass.Float32,
    fast_math: cutlass.Constexpr,
    kmap: experimental_cuda.TensorMap,
    ksmap: experimental_cuda.TensorMap,
    vmap: experimental_cuda.TensorMap,
    vsmap: experimental_cuda.TensorMap,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr = 40,
    io_layouts: cutlass.Constexpr = None,
    reference_scores: cutlass.Constexpr = False,
    base2_scores: cutlass.Constexpr = False,
    reflected_dtype: cutlass.Constexpr = None,
    v_lag: cutlass.Constexpr = 0,
    output_reciprocal: cutlass.Constexpr = False,
    key_post_scale_log2: cutlass.Constexpr = 0,
    value_post_scale_log2: cutlass.Constexpr = 0,
    output_scale_log2: cutlass.Constexpr = 0,
    tma_load_hints: cutlass.Constexpr = ("", "", "", ""),
    sleep_waits: cutlass.Constexpr = False,
    query_map: experimental_cuda.TensorMap | None = None,
    # A typed immutable scalar keeps unguarded calls constant-foldable.
    capture_pdl_wait_mode: cutlass.Int32 = cutlass.Int32(0),  # noqa: B008
    capture_release_tail: cutlass.Constexpr = False,
    maximum_output: cute.Tensor | None = None,
    denominator_output: cute.Tensor | None = None,
    statistic_layouts: cutlass.Constexpr = None,
    coefficients_capped: cutlass.Constexpr = False,
    probability_schedule: cutlass.Constexpr = "serial",
    value_positive_zero: cutlass.Constexpr = False,
    value_metadata_scales: cute.Tensor | None = None,
    value_metadata_read_overlap: cutlass.Constexpr = False,
    rowwise_rescale_threshold: cutlass.Constexpr = None,
    early_tmem_retirement: cutlass.Constexpr = False,
    headwide_rescale: cutlass.Constexpr = False,
    join_before_exit: cutlass.Constexpr = False,
    key_clip: cutlass.Constexpr = "off",
) -> None:
    assert cutlass.const_expr(rowwise_rescale_threshold is None or reference_scores)
    assert cutlass.const_expr(v_lag in (0, 1, 2, 3))
    assert cutlass.const_expr(key_clip in ("off", "leading", "trailing", "both"))
    # The metadata actor copies unshifted scale words for the shifted raw tile.
    assert cutlass.const_expr(key_clip == "off" or value_metadata_scales is None)
    assert cutlass.const_expr(query_columns in (8, 32, 40))
    assert cutlass.const_expr(
        not value_metadata_read_overlap or value_metadata_scales is not None
    )
    if cutlass.const_expr(value_metadata_scales is not None):
        assert cutlass.const_expr(
            value_positive_zero and -22 <= value_post_scale_log2 <= -14
        )
        assert cutlass.const_expr(cute.size(value_metadata_scales, mode=[0]) == 128)
        assert cutlass.const_expr(cute.size(value_metadata_scales, mode=[2]) == 1)
    assert cutlass.const_expr((maximum_output is None) == (denominator_output is None))
    if cutlass.const_expr(maximum_output is not None):
        assert cutlass.const_expr(
            base2_scores
            and coefficients_capped
            and not reference_scores
            and lse is None
            and not output_reciprocal
            and output_scale_log2 == 0
        )
    else:
        assert cutlass.const_expr(not coefficients_capped)
        assert cutlass.const_expr(not base2_scores or reference_scores)
    assert cutlass.const_expr(
        not headwide_rescale
        or (
            rowwise_rescale_threshold is not None
            and reference_scores
            and query_rows == query_columns
        )
    )
    assert cutlass.const_expr(probability_schedule in ("serial", "early", "early_tmem"))
    assert cutlass.const_expr(probability_schedule == "serial" or reference_scores)
    # The early_tmem layout packs S/O/L after K[0:64], V[64:128], fitting each
    # head's 256-column slab for N=8,32,40. Other modes have just S/O.
    # O is the schedule-dependent offset shared by the PV issuer and the
    # compute warps that read/rescale it. L is local to the compute helper.
    output_offset = 128 + query_columns if probability_schedule == "early_tmem" else 192
    thread = cute.arch.thread_idx()[0]
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = thread % 32
    # The selected transport proves an even first item and even heads/request.
    first_head = tile % heads
    request = tile // heads
    count = cutlass.Int64(0)
    if last_page > first_page:
        count = last_page - first_page
    # Dynamic shared allocations reduce address setup across the warp roles.
    # ``_pipeline_shared_memory`` lists them; the body allocates nothing else.
    shared = _allocate_pipeline_shared_memory(
        utils.SmemAllocator(),
        _pipeline_shared_memory(query_columns, value_metadata_scales is not None),
        _pipeline_parts(query_map is not None, value_metadata_scales is not None),
    )
    raw_k = shared.raw_k
    raw_v = shared.raw_v
    scale_k = shared.scale_k
    scale_v = shared.scale_v
    raw_layout = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3), 0, cute.make_layout((128, 128), stride=(128, 1))
    )
    scale_layout = cute.make_layout((128, 4), stride=(4, 1))
    shared_q_raw = shared.shared_q_raw
    probability_raw = shared.probability_raw
    state_raw = shared.state_raw
    scratch_raw = shared.scratch_raw
    slot = shared.slot
    # One block holds every pipeline barrier so the loader warp can initialize
    # it with one instruction per 32 barriers: lane L owns barrier 32 * r + L.
    barrier_offsets, _barrier_count, barrier_rounds = _pipeline_barrier_plan(
        value_metadata_scales is not None
    )
    barriers = shared.barriers
    raw_k_ready = barriers + barrier_offsets["raw_k_ready"]
    raw_k_free = barriers + barrier_offsets["raw_k_free"]
    raw_v_ready = barriers + barrier_offsets["raw_v_ready"]
    raw_v_free = barriers + barrier_offsets["raw_v_free"]
    key_ready = barriers + barrier_offsets["key_ready"]
    key_free = barriers + barrier_offsets["key_free"]
    value_ready = barriers + barrier_offsets["value_ready"]
    value_free = barriers + barrier_offsets["value_free"]
    score_free = barriers + barrier_offsets["score_free"]
    score_ready = barriers + barrier_offsets["score_ready"]
    probability_ready = barriers + barrier_offsets["probability_ready"]
    pv_done = barriers + barrier_offsets["pv_done"]
    # ``None`` without a query map.
    query_ready = shared.query_ready
    metadata_flags = None
    metadata_ready = None
    if cutlass.const_expr(value_metadata_scales is not None):
        metadata_flags = cute.make_tensor(shared.metadata_flags, cute.make_layout((4,)))
        metadata_ready = barriers + barrier_offsets["metadata_ready"]
    if warp == 17:
        # The loader starts the page stream without waiting for the other
        # roles, so it initializes the barriers that stream needs itself.
        for init_round in cutlass.range_constexpr(len(barrier_rounds)):
            expected = cutlass.Int32(1)
            for term in cutlass.range_constexpr(len(barrier_rounds[init_round])):
                extra, lanes = barrier_rounds[init_round][term]
                expected = expected + extra * ((cutlass.Int32(lanes) >> lane) & 1)
            prims.mbarrier_init(barriers + (32 * init_round + lane), expected)
        prims.fence_mbarrier_init()
        # Lane zero alone issues the page stream, against barriers the other
        # lanes initialized, and program order relates only one thread's own
        # operations: join the warp behind every lane's fence first, as the
        # conversion warps do before a lane-zero action.
        cute.arch.sync_warp()
    if warp == 16:
        # The whole warp allocates before any of its lanes takes another path.
        prims.tcgen05_alloc(slot, 512, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
        if cutlass.const_expr(query_map is not None):
            # Only the QK issuer waits for shared Q, so its warp owns that
            # barrier and the copies; the setup rendezvous below publishes
            # the barrier to the warp's other lanes.
            if lane == 0:
                assert query_ready is not None
                prims.mbarrier_init(query_ready, 1)
                prims.fence_mbarrier_init()
                enqueue_paged_query(
                    query_map,
                    shared_q_raw,
                    query_ready,
                    request,
                    first_head,
                    query_columns,
                )
    if thread < 2 * query_columns:
        h = thread // query_columns
        query = thread % query_columns
        state = cute.make_tensor(
            state_raw + h * (4 * query_columns),
            cute.make_layout((4, query_columns), stride=(query_columns, 1)),
        )
        state[0, query] = cutlass.Float32(float("-inf"))
        state[1, query] = cutlass.Float32(0)
        state[2, query] = cutlass.Float32(0)
        state[3, query] = cutlass.Float32(1)
    if cutlass.const_expr(query_map is None):
        shared_q = cute.make_tensor(
            shared_q_raw, cute.make_layout((2 * query_columns * 128,))
        )
        for start in cutlass.range(0, 2 * query_columns * 128, 640, unroll_full=True):
            element = start + thread
            if element < 2 * query_columns * 128:
                pair_head = element // (query_columns * 128)
                inner = element % (query_columns * 128)
                query_row = inner // 128
                dimension = inner % 128
                offset = (
                    pair_head * query_columns * 128
                    + query_row * 64
                    + ((dimension % 64) ^ ((query_row % 8) * 8))
                    + (dimension // 64) * (query_columns * 64)
                )
                value = cutlass.BFloat16(0)
                if query_row < query_rows:
                    value = load_query_element(
                        q,
                        tile + pair_head,
                        query_row,
                        dimension,
                        heads,
                        group,
                        io_layouts,
                    )
                shared_q[offset] = value
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    # Setup rendezvous. The loader only arrives: each of its lanes initialized
    # and fenced its barriers and the warp then synchronized above, so lane
    # zero's stream follows every lane's initialization; it reads no state
    # another role prepares (not the TMEM base, the row state or shared Q);
    # and every consumer of what it loads waits on one of those barriers. The
    # other 608 threads proceed only after all 640 arrivals, each of which
    # follows its lane's fence, hence after the loader's initialization.
    # The QK role's allocation writes the TMEM slot, so only a thread that
    # waited here reads it: the loader, which has no use for the base, would
    # race with that write.
    base = cutlass.Int32(0)
    if warp == 17:
        prims.barrier_cta_arrive(6, thread_count=640)
    else:
        prims.barrier_cta_sync(6, thread_count=640)
        base = cute.make_tensor(slot, cute.make_layout((1,)))[0]
    if cutlass.const_expr(query_map is not None):
        # Only the QK issuer consumes shared Q; the other roles start on KV
        # without it. Empty KV loops still wait here, so the query copies
        # have landed before this role frees TMEM and returns.
        if warp == 16:
            mbar_wait(query_ready, cutlass.Int32(0))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if warp < 8:
        head = cute.arch.make_warp_uniform(warp // 4)
        probability = cute.make_tensor(
            probability_raw + head * query_columns * 128,
            cute.make_layout((query_columns * 128,)),
        )
        state = cute.make_tensor(
            state_raw + head * 4 * query_columns,
            cute.make_layout((4, query_columns), stride=(query_columns, 1)),
        )
        scratch = cute.make_tensor(
            scratch_raw + head * 4 * query_columns,
            cute.make_layout((4, query_columns), stride=(query_columns, 1)),
        )
        _compute_head(
            output,
            lse,
            probability,
            state,
            scratch,
            score_ready + head,
            score_free + head,
            probability_ready + head,
            pv_done + head,
            base + head * 256,
            tile + head,
            head,
            first_page,
            last_page,
            count,
            length,
            query_len,
            group,
            query_rows,
            heads,
            window,
            score_scale,
            fast_math,
            coordinate_dtype,
            query_columns,
            io_layouts,
            reference_scores,
            base2_scores,
            reflected_dtype,
            output_reciprocal,
            output_scale_log2,
            sleep_waits,
            capture_pdl_wait_mode,
            maximum_output,
            denominator_output,
            statistic_layouts,
            coefficients_capped,
            probability_schedule,
            output_offset,
            rowwise_rescale_threshold,
            early_tmem_retirement,
            headwide_rescale,
            key_clip,
        )
    elif warp < 12:
        cute.arch.setmaxregister_decrease(72)
        row = (warp % 4) * 32 + lane
        for step in cutlass.range(count):
            for pair_head in cutlass.range(2, unroll=1):
                head = cute.arch.make_warp_uniform(pair_head)
                ring = 2 * (step & 1) + head
                _wait_non_mma(raw_k_ready + ring, step // 2 & 1, sleep_waits)
                _wait_non_mma(key_free + head, (step & 1) ^ 1, sleep_waits)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                staged_scales = cute.make_tensor(
                    (scale_k + ring * 512).align(16), scale_layout
                )
                assert _packed_scale_layout_supported(staged_scales)
                stage_scaled_s8_rows_to_tmem(
                    cute.make_tensor(
                        raw_k + cute.assume(ring * 16384, divby=16384), raw_layout
                    ),
                    staged_scales,
                    row,
                    base + head * 256,
                    128,
                    32,
                    post_scale_log2=key_post_scale_log2,
                )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                cute.arch.sync_warp()
                if lane == 0:
                    prims.mbarrier_arrive(key_ready + head)
                    prims.mbarrier_arrive(raw_k_free + ring)
    elif warp < 16:
        cute.arch.setmaxregister_decrease(72)
        dimension = (warp % 4) * 32
        for step in cutlass.range(count):
            for pair_head in cutlass.range(2, unroll=1):
                head = cute.arch.make_warp_uniform(pair_head)
                ring = 2 * (step & 1) + head
                _wait_non_mma(raw_v_ready + ring, step // 2 & 1, sleep_waits)
                _wait_non_mma(value_free + head, (step & 1) ^ 1, sleep_waits)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                staged_scales = cute.make_tensor(
                    (scale_v + ring * 512).align(16), scale_layout
                )
                assert _packed_scale_layout_supported(staged_scales)
                zero_tile = False
                if cutlass.const_expr(value_metadata_scales is not None):
                    assert metadata_flags is not None
                    zero_tile = metadata_flags[ring] == 1
                if zero_tile:
                    stage_positive_zero_value_tile(dimension, base + head * 256 + 64)
                else:
                    stage_scaled_s8_transpose_to_tmem(
                        cute.make_tensor(
                            raw_v + cute.assume(ring * 16384, divby=16384), raw_layout
                        ),
                        staged_scales,
                        dimension,
                        lane,
                        base + head * 256 + 64,
                        128,
                        post_scale_log2=value_post_scale_log2,
                        positive_zero=value_positive_zero,
                    )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                cute.arch.sync_warp()
                if lane == 0:
                    prims.mbarrier_arrive(value_ready + head)
                if cutlass.const_expr(value_metadata_scales is not None):
                    # Release each lane's flag/scale reads explicitly for the
                    # metadata actor, as with the producer's release above.
                    # This opt-in path uses 128 arrivals instead of four;
                    # TMEM publication keeps its four warp arrivals above.
                    prims.mbarrier_arrive(raw_v_free + ring)
                elif lane == 0:
                    prims.mbarrier_arrive(raw_v_free + ring)
    else:
        cute.arch.setmaxregister_decrease(48)
        if warp == 16:
            for step in cutlass.range(count):
                for head in cutlass.range_constexpr(2):
                    mbar_wait(score_free + head, (step & 1) ^ 1)
                    mbar_wait(key_ready + head, step & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    mma_128xnx128(
                        base + head * 256,
                        base + head * 256 + 128,
                        shared_q_raw + head * (query_columns * 128),
                        score_ready + head,
                        False,
                        query_columns,
                    )
                    if prims.elect_sync():
                        prims.tcgen05_commit(
                            key_free + head, group=prims.CTAGroup.CTA_1
                        )
            if cutlass.const_expr(early_tmem_retirement):
                # One allocation-wide phase: two compute groups plus this warp.
                # Zero-page tiles take the same path: each compute group
                # arrives after reading its zeroed panel.
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prims.barrier_cta_sync(5, thread_count=288)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                prims.tcgen05_dealloc(
                    cutlass.inttoptr(base, 6, cutlass.Int32),
                    512,
                    group=prims.CTAGroup.CTA_1,
                )
        elif warp == 17:
            if lane == 0:
                if cutlass.const_expr(v_lag == 0):
                    for step in cutlass.range(count):
                        if cutlass.const_expr(capture_release_tail):
                            # Both values are Int64; nonnegative count minus
                            # four cannot overflow, including empty ranges.
                            if step == count - 4:
                                cute.arch.griddepcontrol_launch_dependents()
                        logical_page, logical_key_base = attention_page_coordinates(
                            first_page, last_page, step, reflected_dtype
                        )
                        physical = cutlass.Int32(pages[request, logical_page])
                        key_shift = logical_attention_key_shift(
                            logical_key_base,
                            128,
                            length,
                            query_columns,
                            query_len,
                            group,
                            window,
                            coordinate_dtype,
                            key_clip,
                        )
                        for head in cutlass.range_constexpr(2):
                            _enqueue_operand_tile(
                                raw_k,
                                scale_k,
                                raw_k_ready,
                                raw_k_free,
                                kmap,
                                ksmap,
                                step,
                                cutlass.Int32(head),
                                physical,
                                first_head,
                                l2_hints=tma_load_hints[:2],
                                sleep_waits=sleep_waits,
                                key=key_shift,
                            )
                        for head in cutlass.range_constexpr(2):
                            _enqueue_operand_tile(
                                raw_v,
                                scale_v,
                                raw_v_ready,
                                raw_v_free,
                                vmap,
                                vsmap,
                                step,
                                cutlass.Int32(head),
                                physical,
                                first_head,
                                metadata_ready=metadata_ready,
                                metadata_flags=metadata_flags,
                                l2_hints=tma_load_hints[2:],
                                sleep_waits=sleep_waits,
                                key=key_shift,
                            )
                else:
                    # K(i) precedes V(i-lag). Iterating steps then two heads
                    # avoids overflow from doubling an INT64 step count.
                    # A bounded drain also avoids adding lag to that count.
                    # Each delayed V tile reuses its K tile's page and shift.
                    previous_page = cutlass.Int32(0)
                    older_page = cutlass.Int32(0)
                    previous_shift = cutlass.Int32(0)
                    older_shift = cutlass.Int32(0)
                    for step in cutlass.range(count):
                        if cutlass.const_expr(capture_release_tail):
                            if step == count - 4:
                                cute.arch.griddepcontrol_launch_dependents()
                        logical_page, logical_key_base = attention_page_coordinates(
                            first_page, last_page, step, reflected_dtype
                        )
                        physical = cutlass.Int32(pages[request, logical_page])
                        key_shift = logical_attention_key_shift(
                            logical_key_base,
                            128,
                            length,
                            query_columns,
                            query_len,
                            group,
                            window,
                            coordinate_dtype,
                            key_clip,
                        )
                        for head in cutlass.range_constexpr(2):
                            _enqueue_operand_tile(
                                raw_k,
                                scale_k,
                                raw_k_ready,
                                raw_k_free,
                                kmap,
                                ksmap,
                                step,
                                cutlass.Int32(head),
                                physical,
                                first_head,
                                l2_hints=tma_load_hints[:2],
                                sleep_waits=sleep_waits,
                                key=key_shift,
                            )
                            distance = (v_lag + 1 - head) // 2
                            if cutlass.const_expr(distance == 0) or step >= distance:
                                delayed_page = physical
                                delayed_shift = key_shift
                                if cutlass.const_expr(distance == 1):
                                    delayed_page = previous_page
                                    delayed_shift = previous_shift
                                elif cutlass.const_expr(distance == 2):
                                    delayed_page = older_page
                                    delayed_shift = older_shift
                                _enqueue_operand_tile(
                                    raw_v,
                                    scale_v,
                                    raw_v_ready,
                                    raw_v_free,
                                    vmap,
                                    vsmap,
                                    step - distance,
                                    cutlass.Int32((head - v_lag) & 1),
                                    delayed_page,
                                    first_head,
                                    metadata_ready=metadata_ready,
                                    metadata_flags=metadata_flags,
                                    l2_hints=tma_load_hints[2:],
                                    sleep_waits=sleep_waits,
                                    key=delayed_shift,
                                )
                        older_page = previous_page
                        previous_page = physical
                        older_shift = previous_shift
                        previous_shift = key_shift
                    # Drain at most three V tiles from already-loaded page
                    # IDs. Guards precede subtraction, including empty ranges.
                    for tail in cutlass.range_constexpr(v_lag):
                        distance = (v_lag - tail + 1) // 2
                        if count >= distance:
                            delayed_page = previous_page
                            delayed_shift = previous_shift
                            if cutlass.const_expr(distance == 2):
                                delayed_page = older_page
                                delayed_shift = older_shift
                            _enqueue_operand_tile(
                                raw_v,
                                scale_v,
                                raw_v_ready,
                                raw_v_free,
                                vmap,
                                vsmap,
                                count - distance,
                                cutlass.Int32((tail - v_lag) & 1),
                                delayed_page,
                                first_head,
                                metadata_ready=metadata_ready,
                                metadata_flags=metadata_flags,
                                l2_hints=tma_load_hints[2:],
                                sleep_waits=sleep_waits,
                                key=delayed_shift,
                            )
                if cutlass.const_expr(capture_release_tail):
                    # Every CTA releases even for short or zero-trip ranges.
                    # Repeating a long range's earlier release is harmless.
                    cute.arch.griddepcontrol_launch_dependents()
        elif warp == 18:
            for step in cutlass.range(count):
                for head in cutlass.range_constexpr(2):
                    mbar_wait(probability_ready + head, step & 1)
                    mbar_wait(value_ready + head, step & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    mma_128xnx128(
                        base + head * 256 + 64,
                        base + head * 256 + output_offset,
                        probability_raw + head * (query_columns * 128),
                        pv_done + head,
                        True,
                        query_columns,
                        b_mn_major=True,
                    )
                    if prims.elect_sync():
                        prims.tcgen05_commit(
                            value_free + head, group=prims.CTAGroup.CTA_1
                        )
        elif warp == 19:
            if cutlass.const_expr(value_metadata_scales is not None):
                for step in cutlass.range(count):
                    logical_page, _logical_key_base = attention_page_coordinates(
                        first_page, last_page, step, reflected_dtype
                    )
                    physical = cutlass.Int32(pages[request, logical_page])
                    for head in cutlass.range_constexpr(2):
                        _prepare_value_metadata(
                            scale_v,
                            raw_v_free,
                            metadata_flags,
                            metadata_ready,
                            value_metadata_scales,
                            step,
                            cutlass.Int32(head),
                            physical,
                            first_head,
                            sleep_waits,
                            read_overlap=value_metadata_read_overlap,
                        )
    # With the allocation retired above, nothing else needs a common end: the
    # conversion warps consumed every raw tile, the issuers' MMAs completed
    # before the last O read, and the QK role waited for the query copies and
    # frees TMEM itself. Joining would only keep finished warps resident until
    # the last store. A caller that releases dependents after the body
    # (``join_before_exit``) still gets one common end.
    if cutlass.const_expr(join_before_exit or not early_tmem_retirement):
        cute.arch.barrier()
    if cutlass.const_expr(not early_tmem_retirement):
        if warp == 16:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(
                cutlass.inttoptr(base, 6, cutlass.Int32),
                512,
                group=prims.CTAGroup.CTA_1,
            )
