# pyrefly: ignore-errors
from __future__ import annotations

import operator

import cutlass
import cutlass.cute as cute

from .device.sync import store_shared_remote_f32
from .device.sync import store_shared_remote_f32x2
from .reduce_call import cooperative_column_share
from .reduce_call import cooperative_column_slots
from .reduce_call import use_interleaved_shared_reduce
from .reduce_call import use_serial_block_reduce

try:
    from cutlass.base_dsl.enums import Arch
except ImportError:
    from cutlass.base_dsl.arch import Arch

try:
    from cutlass.experimental import primitives as prims
except ImportError:
    prims = None

_REDUX_SYNC_AVAILABLE = prims is not None


@cute.jit
def _warp_reduce_sum(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction_sum(value, threads_in_group=threads_in_group)


def _smem_read(tensor: cute.Tensor, index: cutlass.Int32, dtype: type) -> cute.Numeric:
    """``tensor[index]`` of a shared-memory buffer of ``dtype`` elements.

    CuTe DSL 4.7 allocates an unsigned integer buffer (Uint8, Uint32) as the
    signed integer of its width and reads it back as such: the value would
    not join with an unsigned identity (``TYPE_UNSTABLE_JOIN``) and would
    compare with the top bit set as negative.  The conversion keeps the bits.
    """
    value = tensor[index]
    return value if type(value) is dtype else dtype(value)


def _cute_nan_max(a: cute.Numeric, b: cute.Numeric) -> cute.Numeric:
    """``max`` that propagates NaN like torch.amax (see ``nan_extremum_expr``)."""
    if type(a) is cutlass.Float32:
        return cute.arch.fmax(a, b, nan=True)
    return cute.math.max(a, b, propagate_nan=True)


def _cute_nan_min(a: cute.Numeric, b: cute.Numeric) -> cute.Numeric:
    """``min`` that propagates NaN like torch.amin (see ``nan_extremum_expr``)."""
    if type(a) is cutlass.Float32:
        return cute.arch.fmin(a, b, nan=True)
    return cute.math.min(a, b, propagate_nan=True)


@cute.jit
def _warp_reduce_max(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction(
        value, _cute_nan_max, threads_in_group=threads_in_group
    )


@cute.jit
def _warp_reduce_min(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction(
        value, _cute_nan_min, threads_in_group=threads_in_group
    )


@cute.jit
def _warp_reduce_max_native_fp32(
    value: cute.Numeric, *, threads_in_group: int
) -> cute.Numeric:
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    if cutlass.const_expr(
        _REDUX_SYNC_AVAILABLE
        and type(value) is cutlass.Float32
        and threads_in_group == 32
        and arch.is_family_of(Arch.sm_100f)
    ):
        return cutlass.Float32(
            prims.redux_sync(value, prims.ReductionKind.FMAX, 0xFFFFFFFF, nan=True)
        )
    return _warp_reduce_max(value, threads_in_group=threads_in_group)


@cute.jit
def _warp_reduce_min_native_fp32(
    value: cute.Numeric, *, threads_in_group: int
) -> cute.Numeric:
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    if cutlass.const_expr(
        _REDUX_SYNC_AVAILABLE
        and type(value) is cutlass.Float32
        and threads_in_group == 32
        and arch.is_family_of(Arch.sm_100f)
    ):
        return cutlass.Float32(
            prims.redux_sync(value, prims.ReductionKind.FMIN, 0xFFFFFFFF, nan=True)
        )
    return _warp_reduce_min(value, threads_in_group=threads_in_group)


_NATIVE_FP32_WARP_DISPATCH = {
    "max": _warp_reduce_max_native_fp32,
    "min": _warp_reduce_min_native_fp32,
}


@cute.jit
def _warp_reduce_prod(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction(
        value,
        operator.mul,
        threads_in_group=threads_in_group,
    )


@cute.jit
def _warp_reduce_any(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction(
        value,
        operator.or_,
        threads_in_group=threads_in_group,
    )


# Reduction type -> (warp_op, combine): the warp-level shuffle reduction and
# the scalar combine every grouped body below is templated over.  The
# ``@cute.jit`` bodies take the pair as plain Python arguments (resolved at
# trace time), so one body per form serves every reduction type.  ``any``
# folds its Int32 0/1 mask (``CuteBackend.widen_reduction_acc_dtype``) with
# or; a Boolean max would be a signed i1 max, which folds as an and.
_OP_TABLE: dict[str, tuple[object, object]] = {
    "sum": (_warp_reduce_sum, operator.add),
    "max": (_warp_reduce_max, _cute_nan_max),
    "min": (_warp_reduce_min, _cute_nan_min),
    "prod": (_warp_reduce_prod, operator.mul),
    "any": (_warp_reduce_any, operator.or_),
}


def _resolve_ops(
    reduction_type: str,
    *,
    fp32_redux: bool = False,
    allowed: tuple[str, ...] | None = None,
) -> tuple[object, object]:
    """``(warp_op, combine)`` of ``reduction_type`` (restricted to ``allowed``
    when given); ``fp32_redux`` selects the native fp32 ``redux.sync`` warp
    extrema where they exist."""
    entry = _OP_TABLE.get(reduction_type)
    if entry is None or (allowed is not None and reduction_type not in allowed):
        raise ValueError(f"unsupported CuTe reduction type: {reduction_type!r}")
    warp_op, combine = entry
    if fp32_redux and reduction_type in _NATIVE_FP32_WARP_DISPATCH:
        warp_op = _NATIVE_FP32_WARP_DISPATCH[reduction_type]
    return warp_op, combine


def _cute_scalar_combine(reduction_type: str) -> object:
    """The scalar combine of ``reduction_type``."""
    return _resolve_ops(reduction_type)[1]


@cute.jit
def _cute_grouped_reduce_warp_body(
    input_value: cute.Numeric,
    warp_op: object,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
) -> cute.Numeric:
    lane_in_group = lane_expr % group_span
    lane_mod_pre = lane_in_group % pre
    selected = warp_op(
        input_value if lane_mod_pre == 0 else identity,
        threads_in_group=group_span,
    )
    for p in cutlass.range_constexpr(1, pre):
        reduced = warp_op(
            input_value if lane_mod_pre == p else identity,
            threads_in_group=group_span,
        )
        selected = reduced if lane_mod_pre == p else selected
    return selected


def _cute_grouped_reduce_warp(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    pre: int,
    group_span: int,
    fp32_redux: bool = False,
) -> cute.Numeric:
    warp_op, _ = _resolve_ops(reduction_type, fp32_redux=fp32_redux)
    return _cute_grouped_reduce_warp_body(
        input_value, warp_op, identity, lane_expr, pre=pre, group_span=group_span
    )


@cute.jit
def _cute_grouped_reduce_shared_two_stage_body(
    input_value: cute.Numeric,
    warp_op: object,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    dtype = type(identity)
    warps_per_group = group_span // 32
    partials_size = group_count * pre * warps_per_group
    results_size = group_count * pre
    smem_size = partials_size + results_size
    smem_ptr = cute.arch.alloc_smem(dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_id = lane_var // group_span
    lane_in_warp = lane_var % 32
    warp_in_group = lane_in_group_var // 32
    partials_base = group_id * (pre * warps_per_group)
    results_base = partials_size + group_id * pre

    for p in cutlass.range_constexpr(pre):
        masked_input = input_value if lane_mod_pre_var == p else identity
        warp_partial = warp_op(masked_input, threads_in_group=32)
        partial_idx = partials_base + p * warps_per_group + warp_in_group
        if lane_in_warp == 0:
            smem[partial_idx] = warp_partial
        cute.arch.sync_threads()

        if warp_in_group == 0:
            stage2_input = (
                _smem_read(
                    smem, partials_base + p * warps_per_group + lane_in_warp, dtype
                )
                if lane_in_warp < warps_per_group
                else identity
            )
            group_result = warp_op(stage2_input, threads_in_group=32)
            if lane_in_warp == 0:
                smem[results_base + p] = group_result
        cute.arch.sync_threads()

    return _smem_read(smem, results_base + lane_mod_pre_var, dtype)


# Serial cross-warp combine for whole-CTA groups with few warps: warp
# reduce, lane 0 of each warp writes its partial to SMEM, one sync, then
# EVERY thread folds the ``warps`` partials serially.  Compared with the
# two-stage form this drops the second warp shuffle reduce and its guards
# (~5 SHFL + WARPSYNC per site) and the result is a fold over SMEM loads,
# which anchors the scheduler the same way (a shuffle-produced reduction
# result as the subtrahend of a bulk exp sweep makes LLVM hoist the whole
# sweep above the reduce and double the live set).  The trailing sync
# keeps a multi-trip call site from racing the next trip's writes (both
# passes touch the same slots, unlike the two-stage's disjoint regions).


@cute.jit
def _cute_grouped_reduce_shared_serial_body(
    input_value: cute.Numeric,
    warp_op: object,
    combine: object,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    group_span: int,
) -> cute.Numeric:
    dtype = type(identity)
    warps = group_span // 32
    smem_ptr = cute.arch.alloc_smem(dtype, warps)
    smem = cute.make_tensor(smem_ptr, (warps,))
    lane_in_warp = lane_var % 32
    # Only used for group_count == 1 groups spanning the whole CTA, so
    # ``lane_var`` is the CTA thread index and ``lane_var // 32`` its warp.
    warp = lane_var // 32
    warp_partial = warp_op(input_value, threads_in_group=32)
    if lane_in_warp == 0:
        smem[warp] = warp_partial
    cute.arch.sync_threads()
    result = _cute_fold_warp_partials(
        smem, 0, combine, identity, stride=1, count=warps, serial=True
    )
    cute.arch.sync_threads()
    return result


def _cute_grouped_reduce_shared_serial(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    *,
    group_span: int,
    fp32_redux: bool = False,
) -> cute.Numeric:
    warp_op, combine = _resolve_ops(reduction_type, fp32_redux=fp32_redux)
    return _cute_grouped_reduce_shared_serial_body(
        input_value,
        warp_op,
        combine,
        identity,
        lane_var,
        group_span,
    )


def _cute_grouped_reduce_shared_columns(
    values: list,
    reduction_type: str,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    *,
    pre: int,
    group_span: int,
    group_count: int,
    fold: str = "serial",
) -> list:
    """Reduce ``len(values)`` independent per-thread partials across a thread
    group with one shared-memory round trip.

    Each element of ``values`` is a separate reduction (one output column of
    a register tile).  The group is ``group_span`` consecutive linear threads
    and ``lane_in_group_var`` the thread's position in it; ``pre`` sibling
    coordinates occupy the low strides of the group, so the reduction
    coordinate is ``lane_in_group // pre`` and the ``pre`` threads sharing a
    reduction coordinate hold distinct columns.

    * ``pre == 1`` (reduction threads consecutive, ``group_span`` a multiple
      of 32): every warp first folds each column with a shuffle butterfly, so
      all of its lanes hold the warp partial and write it to the warp's column
      slot without a lane predicate; after one barrier every thread sums its
      group's ``group_span // 32`` partials per column.
    * ``pre > 1`` (threads laid out along the vector axis first),
      ``fold="serial"``: every thread writes its partials, and after one
      barrier every thread sums the ``group_span // pre`` partials of its own
      sibling coordinate per column.
    * ``pre > 1``, ``fold="warp"`` (``group_span`` a multiple of 32, ``pre``
      a divisor of 32): every warp first folds each column across the
      ``32 // pre`` reduction coordinates it holds with a strided shuffle
      butterfly (xor offsets ``pre .. 16`` keep ``lane % pre`` fixed), the
      lanes of one sibling class then hold the same warp partial and write it
      to that class's slot of the warp (same value, same address, no lane
      predicate); after one barrier every thread sums its sibling class's
      ``group_span // 32`` warp partials per column.  Against the serial form
      this trades ``group_span // pre`` staged reads per column for
      ``log2(32 // pre)`` shuffles plus ``group_span // 32`` reads; the sum
      order differs (a tree within the warp, then over the warps), so the
      register-tile emitter only requests it through an autotuner knob.
    * ``pre > 1``, ``fold="cooperative"`` (``group_span`` a multiple of 32):
      every thread writes its partials; after one barrier the group's
      ``pre * len(values)`` outputs are spread over its lanes, a few lanes
      per output summing strided subsets of its partials and folding them
      with a shuffle butterfly into a shared output slot, and after a second
      barrier every thread reads its sibling class's outputs
      (``_cute_cooperative_column_sums``).  A thread reads a handful of
      staged partials instead of ``group_span // pre`` per column, for one
      more barrier; the sum order differs (strided partial sums, then a
      tree), so the register-tile emitter also requests it only through an
      autotuner knob.

    The serial and warp forms cost the whole tile two barriers (the
    cooperative form three), against one two-stage reduce (two barriers, and
    ``pre`` masked shuffles per stage) per column.  The
    trailing barrier matters at a multi-trip call site (a persistent
    ``pid_type`` loop has no other barrier per trip): without it a fast warp's
    next-trip writes could overwrite partials a slower warp has not read yet.
    Called during ``@cute.kernel`` tracing as plain Python (no dynamic control
    flow), like the vector store helpers; returns one value per column.
    """
    if reduction_type not in ("sum", "prod"):
        raise ValueError(f"unsupported CuTe column reduction type: {reduction_type!r}")
    if fold not in ("serial", "warp", "cooperative"):
        raise ValueError(f"unsupported CuTe column reduction fold: {fold!r}")
    warp_fold, combine = _resolve_ops(reduction_type)
    width = len(values)
    dtype = type(identity)
    group_id = lane_var // group_span
    results = []
    if pre > 1 and fold == "cooperative":
        if group_span % 32:
            raise ValueError(
                "the cooperative fold of a CuTe column reduction needs a "
                f"warp-aligned group (group_span={group_span})"
            )
        results = _cute_cooperative_column_sums(
            values,
            combine,
            identity,
            group_id,
            lane_in_group_var,
            pre=pre,
            group_span=group_span,
            group_count=group_count,
        )
        cute.arch.sync_threads()
        return results
    if pre > 1 and fold == "warp":
        if group_span % 32 or 32 % pre:
            raise ValueError(
                "the warp fold of a CuTe column reduction needs a warp-aligned "
                f"group and a sibling count dividing a warp (pre={pre}, "
                f"group_span={group_span})"
            )
        results = _cute_warp_fold_column_sums(
            values,
            combine,
            identity,
            lane_var,
            lane_in_group_var,
            pre=pre,
            group_span=group_span,
            group_count=group_count,
        )
        cute.arch.sync_threads()
        return results
    if pre == 1:
        warps_per_group = group_span // 32
        smem_size = group_count * warps_per_group * width
        smem_ptr = cute.arch.alloc_smem(dtype, smem_size)
        smem = cute.make_tensor(smem_ptr, (smem_size,))
        warp_in_group = lane_in_group_var // 32
        group_base = group_id * (warps_per_group * width)
        slot_base = group_base + warp_in_group * width
        for column, value in enumerate(values):
            smem[slot_base + column] = warp_fold(value, threads_in_group=32)
        cute.arch.sync_threads()
        for column in range(width):
            total = identity
            for warp_index in range(warps_per_group):
                total = combine(
                    total,
                    _smem_read(smem, group_base + warp_index * width + column, dtype),
                )
            results.append(total)
        cute.arch.sync_threads()
        return results
    reduce_extent = group_span // pre
    smem_size = group_count * group_span * width
    smem_ptr = cute.arch.alloc_smem(dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_base = group_id * (group_span * width)
    slot_base = group_base + lane_in_group_var * width
    for column, value in enumerate(values):
        smem[slot_base + column] = value
    cute.arch.sync_threads()
    sibling_base = group_base + (lane_in_group_var % pre) * width
    for column in range(width):
        total = identity
        for reduce_index in range(reduce_extent):
            total = combine(
                total,
                _smem_read(
                    smem, sibling_base + reduce_index * (pre * width) + column, dtype
                ),
            )
        results.append(total)
    cute.arch.sync_threads()
    return results


def _cute_cooperative_column_sums(
    values: list,
    combine: object,
    identity: cute.Numeric,
    group_id: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    *,
    pre: int,
    group_span: int,
    group_count: int,
) -> list:
    """``_cute_grouped_reduce_shared_columns(..., fold="cooperative")``.

    The serial fold has every thread sum all ``group_span // pre`` staged
    partials of each of its columns, redundantly across the group: a
    1024-thread group with ``pre = 16`` reads 512 values per thread for eight
    columns.  Here the group's ``pre * width`` outputs are spread over its
    threads: ``share`` consecutive lanes (``cooperative_column_share``) each
    sum every ``share``-th partial of one output and fold their sums with a
    shuffle butterfly, after which each of them holds the output and stores
    it to the block's slot (same value, same address, one instruction of one
    warp, no lane predicate); after a second barrier every thread reads the
    ``width`` outputs of its sibling class.  The lane blocks take the
    outputs in turn (several each when they are fewer than the outputs), and
    a lane reads ``group_span // pre // share`` staged partials per output
    it takes: in the example eight (then three shuffles and eight output
    reads) instead of 512.  The group reads every staged partial once when
    the blocks match the outputs one to one (``share * pre * width ==
    group_span``); when ``share`` is capped by a warp or by the reduction
    extent, the blocks outnumber the outputs and the extra ones recompute an
    output another block also computes, reading its partials again.  Every
    block of a pass stores to a slot of its own (``cooperative_column_slots``),
    so blocks of different warps never store to one slot, not even the same
    value.  The lanes of a block read partials ``pre * width`` words apart,
    a ``share``-way bank conflict when that stride is a multiple of 32
    words (8-way in the example, on eight reads instead of 512).  One more
    barrier than the serial fold; the sum order (strided partial sums,
    then a tree) differs from the serial fold's.  Shared memory: the staged
    partials plus ``cooperative_column_slots`` output slots per group.  Like
    ``_cute_warp_fold_column_sums`` it stops before the trailing barrier,
    which the caller emits.
    """
    width = len(values)
    dtype = type(identity)
    reduce_extent = group_span // pre
    outputs = pre * width
    slots = cooperative_column_slots(pre, width, group_span)
    staged = group_count * group_span * width
    smem_ptr = cute.arch.alloc_smem(dtype, staged + group_count * slots)
    smem = cute.make_tensor(smem_ptr, (staged + group_count * slots,))
    stage_base = group_id * (group_span * width)
    result_base = staged + group_id * slots
    slot_base = stage_base + lane_in_group_var * width
    for column, value in enumerate(values):
        smem[slot_base + column] = value
    cute.arch.sync_threads()
    share = cooperative_column_share(pre, width, group_span)
    owners = group_span // share
    part = lane_in_group_var % share
    block = lane_in_group_var // share
    for first in range(0, outputs, owners):
        # Plain Python during tracing, so no branch on the lane: block ``k``
        # of the pass takes output ``first + k`` and stores it to slot
        # ``first + k``, the slot its readers read; a block past the last
        # output wraps onto an earlier one, recomputes it identically and
        # stores the copy to a slot of its own that nothing reads.
        output = (first + block) % outputs
        sibling = output // width
        column = output % width
        total = identity
        for step in range(reduce_extent // share):
            partial = part + step * share
            total = combine(
                total,
                _smem_read(
                    smem, stage_base + (partial * pre + sibling) * width + column, dtype
                ),
            )
        for shift in range(share.bit_length() - 2, -1, -1):
            total = combine(
                total,
                dtype(
                    cute.arch.shuffle_sync_bfly(
                        total, offset=1 << shift, mask=-1, mask_and_clamp=31
                    )
                ),
            )
        smem[result_base + first + block] = total
    cute.arch.sync_threads()
    sibling_base = result_base + (lane_in_group_var % pre) * width
    return [_smem_read(smem, sibling_base + column, dtype) for column in range(width)]


def _cute_warp_fold_column_sums(
    values: list,
    combine: object,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    *,
    pre: int,
    group_span: int,
    group_count: int,
) -> list:
    """``_cute_grouped_reduce_shared_columns(..., fold="warp")`` for ``pre > 1``,
    up to (not including) the trailing barrier the caller emits before it
    returns (the barrier that guards the slots' reuse by a persistent CTA's
    next trip).

    ``pre`` divides 32 and ``group_span`` is a multiple of 32 (checked by the
    caller), so a warp holds ``32 // pre`` reduction coordinates of every
    sibling class and the groups are warp aligned.  Plain Python during
    tracing: no dynamic control flow, so the lanes of one sibling class all
    write their (identical) warp partial to the class's slot instead of a
    predicated representative.  Shared memory: ``group_span // 32`` warps x
    ``pre`` sibling classes x ``len(values)`` columns per group.
    """
    width = len(values)
    dtype = type(identity)
    warps_per_group = group_span // 32
    slots = pre * width
    smem_size = group_count * warps_per_group * slots
    smem_ptr = cute.arch.alloc_smem(dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_id = lane_var // group_span
    warp_in_group = lane_in_group_var // 32
    lane_mod_pre = lane_in_group_var % pre
    partials_base = group_id * (warps_per_group * slots)
    slot_base = partials_base + warp_in_group * slots + lane_mod_pre * width
    for column, value in enumerate(values):
        for shift in range(pre.bit_length() - 1, 5):
            value = combine(
                value,
                dtype(
                    cute.arch.shuffle_sync_bfly(
                        value, offset=1 << shift, mask=-1, mask_and_clamp=31
                    )
                ),
            )
        smem[slot_base + column] = value
    cute.arch.sync_threads()
    sibling_base = partials_base + lane_mod_pre * width
    results = []
    for column in range(width):
        total = identity
        for warp_index in range(warps_per_group):
            total = combine(
                total,
                _smem_read(smem, sibling_base + warp_index * slots + column, dtype),
            )
        results.append(total)
    return results


# Interleaved groups (``pre > 1``: the ``pre`` sibling coordinates occupy the
# low strides of a group, so the threads of one reduction are ``pre`` apart)
# combine with two barriers and no per-sibling unrolling.  The two-stage form
# above repeats its masked shuffle stages and both barriers ``pre`` times
# (``range_constexpr(pre)``): at ``pre`` = 256 one call site unrolls into 512
# barriers, and a tile with 8 such sites per 16 vector copies made the DSL
# compile consume 40 GB without finishing.
#
# * ``pre < 32`` (a power of two): a butterfly over the offsets ``>= pre``
#   folds exactly the lanes of a warp that share ``lane % pre``, so lanes
#   ``0 .. pre - 1`` hold their sibling's warp partial; after one barrier the
#   first ``pre`` threads of the group fold the ``group_span // 32`` warp
#   partials of their sibling.  Shared memory: the two-stage form's layout.
# * ``pre >= 32`` (a multiple of 32): every thread writes its value, and after
#   one barrier the first ``pre`` threads of the group fold the
#   ``group_span // pre`` values of their sibling (at most 32 of them since
#   ``group_span <= 1024``).  Shared memory: ``group_span`` values per group
#   plus the results, never more than the two-stage form needs at this
#   ``pre``, so the emitter's budget check stays conservative.


@cute.jit
def _cute_warp_tree_fold(
    smem: cute.Tensor,
    base: cutlass.Int32,
    combine: object,
    identity: cute.Numeric,
    *,
    stride: cutlass.Constexpr[int],
    count: cutlass.Constexpr[int],
) -> cute.Numeric:
    """Fold ``count`` shared values ``smem[base + w * stride]`` in the order of
    the two-stage form's second stage: a 32-lane descending butterfly
    (offsets 16, 8, 4, 2, 1; partner second) with lane ``w`` holding value
    ``w`` and the lanes beyond ``count`` holding the identity, as
    ``cute.arch.warp_reduction`` computes it.  Identity partners are exact
    no-ops, so this reproduces that reduction tree bit for bit."""
    dtype = type(identity)
    values = []
    for w in cutlass.range_constexpr(32):
        if cutlass.const_expr(w < count):
            values.append(_smem_read(smem, base + w * stride, dtype))
        else:
            values.append(identity)
    for shift in cutlass.range_constexpr(4, -1, -1):
        for i in cutlass.range_constexpr(1 << shift):
            values[i] = combine(values[i], values[i + (1 << shift)])
    return values[0]


@cute.jit
def _cute_fold_warp_partials(
    smem: cute.Tensor,
    base: cutlass.Int32,
    combine: object,
    identity: cute.Numeric,
    *,
    stride: cutlass.Constexpr[int],
    count: cutlass.Constexpr[int],
    serial: cutlass.Constexpr[bool],
) -> cute.Numeric:
    """Fold the ``count`` warp partials ``smem[base + w * stride]`` as the
    form ``_cute_grouped_reduce_shared_two_stage`` dispatches to does: in warp
    order for the serial form (``use_serial_block_reduce``), else in the
    two-stage form's tree order (``_cute_warp_tree_fold``)."""
    if cutlass.const_expr(serial):
        dtype = type(identity)
        total = _smem_read(smem, base, dtype)
        for w in cutlass.range_constexpr(1, count):
            total = combine(total, _smem_read(smem, base + w * stride, dtype))
    else:
        total = _cute_warp_tree_fold(
            smem, base, combine, identity, stride=stride, count=count
        )
    return total


@cute.jit
def _cute_strided_warp_butterfly(
    value: cute.Numeric,
    combine: object,
    identity: cute.Numeric,
    *,
    pre: cutlass.Constexpr[int],
) -> cute.Numeric:
    """Fold ``value`` over the lanes of its warp that share ``lane % pre``:
    the interleaved forms' first stage.

    The two-stage form reduces each sibling with a full-warp descending
    butterfly (offsets 16 .. 1, partner second) over identity-masked lanes.
    Its stages with offset >= ``pre`` pair two lanes of the sibling; the
    others pair a lane with a masked one, which leaves the value alone but
    for the sign of a zero sum.  This is the former, in the same order."""
    dtype = type(identity)
    for shift in cutlass.range_constexpr(4, pre.bit_length() - 2, -1):
        value = combine(
            value,
            dtype(
                cute.arch.shuffle_sync_bfly(
                    value, offset=1 << shift, mask=-1, mask_and_clamp=31
                )
            ),
        )
    return value


@cute.jit
def _cute_warp_tree_fold_strided_siblings(
    smem: cute.Tensor,
    base: cutlass.Int32,
    sibling: cutlass.Int32,
    combine: object,
    identity: cute.Numeric,
    *,
    pre: cutlass.Constexpr[int],
    count: cutlass.Constexpr[int],
) -> cute.Numeric:
    """Second-stage tree of the two-stage form for ``pre >= 32``: warp ``w``
    of the group holds sibling ``sibling`` at lane ``sibling % 32`` exactly
    when ``(w - sibling // 32) % (pre // 32) == 0`` (its masked warp reduce
    is that one value), every other warp contributes the identity."""
    dtype = type(identity)
    ratio = pre // 32
    first_warp = sibling // 32
    lane = sibling % 32
    values = []
    for w in cutlass.range_constexpr(32):
        if cutlass.const_expr(w < count):
            held = ((w - first_warp) % ratio) == 0
            value = _smem_read(smem, base + w * 32 + lane, dtype)
            values.append(value if held else identity)
        else:
            values.append(identity)
    for shift in cutlass.range_constexpr(4, -1, -1):
        for i in cutlass.range_constexpr(1 << shift):
            values[i] = combine(values[i], values[i + (1 << shift)])
    return values[0]


@cute.jit
def _cute_grouped_reduce_shared_interleaved_warp_body(
    input_value: cute.Numeric,
    combine: object,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    dtype = type(identity)
    warps_per_group = group_span // 32
    partials_size = group_count * pre * warps_per_group
    results_size = group_count * pre
    smem_ptr = cute.arch.alloc_smem(dtype, partials_size + results_size)
    smem = cute.make_tensor(smem_ptr, (partials_size + results_size,))
    group_id = lane_var // group_span
    lane_in_warp = lane_var % 32
    warp_in_group = lane_in_group_var // 32
    partials_base = group_id * (pre * warps_per_group)
    results_base = partials_size + group_id * pre
    value = _cute_strided_warp_butterfly(input_value, combine, identity, pre=pre)
    if lane_in_warp < pre:
        # Warp-major partials: consecutive threads of the second stage read
        # consecutive words.
        smem[partials_base + warp_in_group * pre + lane_in_warp] = value
    cute.arch.sync_threads()
    if lane_in_group_var < pre:
        total = _cute_warp_tree_fold(
            smem,
            partials_base + lane_in_group_var,
            combine,
            identity,
            stride=pre,
            count=warps_per_group,
        )
        smem[results_base + lane_in_group_var] = total
    cute.arch.sync_threads()
    return _smem_read(smem, results_base + lane_mod_pre_var, dtype)


@cute.jit
def _cute_grouped_reduce_shared_interleaved_block_body(
    input_value: cute.Numeric,
    combine: object,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    dtype = type(identity)
    partials_size = group_count * group_span
    results_size = group_count * pre
    smem_ptr = cute.arch.alloc_smem(dtype, partials_size + results_size)
    smem = cute.make_tensor(smem_ptr, (partials_size + results_size,))
    group_id = lane_var // group_span
    group_base = group_id * group_span
    results_base = partials_size + group_id * pre
    smem[group_base + lane_in_group_var] = input_value
    cute.arch.sync_threads()
    if lane_in_group_var < pre:
        # With ``pre`` a multiple of 32 a warp holds at most one lane of a
        # sibling, so the two-stage form's masked warp reduce is that lane's
        # value; its second stage is the warp-indexed butterfly below.
        total = _cute_warp_tree_fold_strided_siblings(
            smem,
            group_base,
            lane_in_group_var,
            combine,
            identity,
            pre=pre,
            count=group_span // 32,
        )
        smem[results_base + lane_in_group_var] = total
    cute.arch.sync_threads()
    return _smem_read(smem, results_base + lane_mod_pre_var, dtype)


def _cute_grouped_reduce_shared_interleaved(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    pre: int,
    group_span: int,
    group_count: int,
) -> cute.Numeric:
    combine = _cute_scalar_combine(reduction_type)
    body = (
        _cute_grouped_reduce_shared_interleaved_warp_body
        if pre < 32
        else _cute_grouped_reduce_shared_interleaved_block_body
    )
    return body(
        input_value,
        combine,
        identity,
        lane_var,
        lane_in_group_var,
        lane_mod_pre_var,
        pre=pre,
        group_span=group_span,
        group_count=group_count,
    )


def _cute_grouped_reduce_shared_two_stage(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    pre: int,
    group_span: int,
    group_count: int,
    fp32_redux: bool = False,
) -> cute.Numeric:
    # Despite the name (kept for call-site stability) this dispatcher owns
    # ALL shared-memory cross-warp combines: small whole-CTA groups route
    # to the cheaper serial form, interleaved groups to the two-barrier
    # interleaved form, everything else to the two-stage form.
    if use_serial_block_reduce(pre, group_span, group_count):
        return _cute_grouped_reduce_shared_serial(
            input_value,
            reduction_type,
            identity,
            lane_var,
            group_span=group_span,
            fp32_redux=fp32_redux,
        )
    if use_interleaved_shared_reduce(pre, group_span):
        return _cute_grouped_reduce_shared_interleaved(
            input_value,
            reduction_type,
            identity,
            lane_var,
            lane_in_group_var,
            lane_mod_pre_var,
            pre=pre,
            group_span=group_span,
            group_count=group_count,
        )
    warp_op, _ = _resolve_ops(reduction_type, fp32_redux=fp32_redux)
    return _cute_grouped_reduce_shared_two_stage_body(
        input_value,
        warp_op,
        identity,
        lane_var,
        lane_in_group_var,
        lane_mod_pre_var,
        pre=pre,
        group_span=group_span,
        group_count=group_count,
    )


@cute.jit
def _cute_grouped_reduce_shared_tree_body(
    input_value: cute.Numeric,
    combine: object,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    dtype = type(identity)
    smem_size = num_threads + group_count * pre
    smem_ptr = cute.arch.alloc_smem(dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_base = lane_var - lane_in_group_var
    group_id = lane_var // group_span
    result_base = num_threads + group_id * pre

    for p in cutlass.range_constexpr(pre):
        smem[lane_var] = input_value if lane_mod_pre_var == p else identity
        cute.arch.sync_threads()
        stride = 1
        while stride < group_span:
            if (
                lane_in_group_var % (stride * 2) == 0
                and lane_in_group_var + stride < group_span
            ):
                lhs = _smem_read(smem, lane_var, dtype)
                rhs = _smem_read(smem, group_base + lane_in_group_var + stride, dtype)
                smem[lane_var] = combine(lhs, rhs)
            cute.arch.sync_threads()
            stride *= 2

        if lane_in_group_var == 0:
            smem[result_base + p] = _smem_read(smem, lane_var, dtype)
        cute.arch.sync_threads()

    return _smem_read(smem, result_base + lane_mod_pre_var, dtype)


def _cute_grouped_reduce_shared_tree(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    pre: int,
    group_span: int,
    num_threads: int,
    group_count: int,
) -> cute.Numeric:
    _, combine = _resolve_ops(reduction_type)
    return _cute_grouped_reduce_shared_tree_body(
        input_value,
        combine,
        identity,
        lane_var,
        lane_in_group_var,
        lane_mod_pre_var,
        pre=pre,
        group_span=group_span,
        num_threads=num_threads,
        group_count=group_count,
    )


@cute.jit
def _cute_argreduce_index_impl(
    smem: cute.Tensor,
    valid_smem: cute.Tensor,
    start_idx: cutlass.Int32,
    stride: cutlass.Int32,
    *,
    extent: cutlass.Constexpr[int],
    dtype: cutlass.Constexpr,
    is_max: cutlass.Constexpr[bool],
) -> cutlass.Int64:
    best_index = cutlass.Int64(0)
    best_value = _smem_read(smem, start_idx, dtype)
    best_valid = _smem_read(valid_smem, start_idx, cutlass.Int32)
    for candidate_index in cutlass.range_constexpr(1, extent):
        candidate_offset = start_idx + stride * candidate_index
        candidate = _smem_read(smem, candidate_offset, dtype)
        candidate_valid = _smem_read(valid_smem, candidate_offset, cutlass.Int32)
        if cutlass.const_expr(is_max):
            improves = candidate > best_value
        else:
            improves = candidate < best_value
        better = candidate_valid != cutlass.Int32(0) and (
            best_valid == cutlass.Int32(0)
            or (
                best_valid != cutlass.Int32(0)
                and (
                    improves
                    # A NaN wins over every number, the first NaN over later ones.
                    or (candidate != candidate and best_value == best_value)
                    or (
                        candidate == best_value
                        and cutlass.Int64(candidate_index) < best_index
                    )
                )
            )
        )
        if better:
            best_value = candidate
            best_valid = candidate_valid
            best_index = cutlass.Int64(candidate_index)
    return best_index


def _cute_argreduce_index(
    smem: cute.Tensor,
    valid_smem: cute.Tensor,
    start_idx: cutlass.Int32,
    stride: cutlass.Int32,
    *,
    extent: int,
    reduction_type: str,
    dtype: type,
) -> cutlass.Int64:
    if reduction_type not in ("argmax", "argmin"):
        raise ValueError(f"unsupported CuTe argreduce type: {reduction_type!r}")
    return _cute_argreduce_index_impl(
        smem,
        valid_smem,
        start_idx,
        stride,
        extent=extent,
        dtype=dtype,
        is_max=reduction_type == "argmax",
    )


@cute.jit
def _cute_grouped_reduce_cluster_body(
    input_value: cute.Numeric,
    warp_op: object,
    combine: object,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    buf_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    group_span: int,
    cluster_n: int,
) -> cute.Numeric:
    """Combine ``input_value`` across ``group_span`` CTA threads AND the
    ``cluster_n`` CTAs of the thread-block cluster that cooperate on one
    reduction group (e.g. one softmax row split across the cluster).

    Every warp's partial is warp-reduced, then pushed into EVERY peer CTA's
    SMEM receive buffer with ``st.async`` + an mbarrier transaction count;
    after the mbarrier wait each CTA folds all ``warps * cluster_n``
    partials locally.  Small exchanges use a serial fold; larger exchanges
    distribute the slots over warp lanes and finish with a warp reduction.
    Each (textual) call site must execute exactly once per kernel — the
    mbarrier is single-phase.

    ``buf_ptr`` (``warps * cluster_n`` Float32 slots) and ``mbar_ptr`` (one
    Int64 mbarrier, arrival count 1) must be allocated and initialized in
    the kernel preamble, followed by ``mbarrier_init_fence`` +
    ``cluster_arrive_relaxed`` + ``cluster_wait`` so peers' barriers are
    live before the first remote store (the device-function codegen emits
    this once for all sites).

    ``identity``'s dtype must be Float32 (the hoist pass rewrites the
    identity to the fp32 accumulator dtype).
    """
    warps = group_span // 32
    slots = warps * cluster_n
    buf = cute.make_tensor(buf_ptr, (slots,))
    mbar = mbar_ptr
    warp_partial = warp_op(input_value, threads_in_group=32)
    rank = cutlass.Int32(cute.arch.block_idx_in_cluster())
    warp = lane_var // 32
    lane = lane_var % 32
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, slots * 4)
    if lane < cluster_n:
        store_shared_remote_f32(
            cutlass.Float32(warp_partial),
            buf.iterator + (rank * warps + warp),
            mbar,
            lane,
        )
    cute.arch.mbarrier_wait(mbar, phase=0)
    result = identity
    if cutlass.const_expr(slots <= 8):
        for i in cutlass.range_constexpr(slots):
            result = combine(result, buf[i])
        return result
    for i in cutlass.range_constexpr((slots + 31) // 32):
        slot = lane + i * 32
        if slot < slots:
            result = combine(result, buf[slot])
    return warp_op(result, threads_in_group=32)


def _cute_grouped_reduce_cluster(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    buf_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    *,
    group_span: int,
    cluster_n: int,
    fp32_redux: bool = False,
) -> cute.Numeric:
    # The exchange buffers Float32 partials: no product form.
    warp_op, combine = _resolve_ops(
        reduction_type, fp32_redux=fp32_redux, allowed=("sum", "max", "min")
    )
    return _cute_grouped_reduce_cluster_body(
        input_value,
        warp_op,
        combine,
        identity,
        lane_var,
        buf_ptr,
        mbar_ptr,
        group_span,
        cluster_n,
    )


def _cute_grouped_reduce_block(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    *,
    group_span: int,
    fp32_redux: bool = False,
) -> cute.Numeric:
    """CTA-local grouped reduce over one group spanning the whole CTA
    (``pre == 1``, ``group_count == 1``): warp reduce + SMEM cross-warp
    combine.  Used by the cluster online-pair rewrite to relocalize the
    first (max) reduction of an online-softmax pair — the cross-CTA
    combine happens later in ``_cute_grouped_reduce_cluster_online_pair``."""
    return _cute_grouped_reduce_shared_two_stage(
        input_value,
        reduction_type,
        identity,
        lane_var,
        lane_var,
        cutlass.Int32(0),
        pre=1,
        group_span=group_span,
        group_count=1,
        fp32_redux=fp32_redux,
    )


def _frame_rescale(
    difference: cute.Numeric, scale: float | None, fastmath: bool
) -> cute.Numeric:
    """The online-softmax rescale of a frame ``difference``: ``exp2(difference
    * scale)``, or the natural ``exp(difference)`` for a ``None`` scale -- the
    spelling of the exponentials the matched sum folds
    (``analysis.online_pair._exponent``)."""
    if scale is None:
        return cute.math.exp(difference, fastmath=fastmath)
    return cute.math.exp2(difference * scale, fastmath=fastmath)


@cute.jit
def _cute_grouped_reduce_cluster_online_pair_body(
    local_sum: cute.Numeric,
    local_max: cute.Numeric,
    lane_var: cutlass.Int32,
    buf_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    group_span: int,
    cluster_n: int,
    scale: float | None,
    fastmath: bool,
    fp32_redux: bool,
) -> tuple[cute.Numeric, cute.Numeric]:
    """Single-exchange cluster combine for an online-softmax (max, sum)
    pair.  ``local_max`` is this CTA's slice max (CTA-uniform; relocalized
    site-A result) and ``local_sum`` is the per-thread partial of
    ``sum(exp2(f(x) - local_max * scale))`` over the CTA's slice
    (``sum(exp(f(x) - local_max))`` for a ``None`` scale).

    The per-thread sum is block-reduced within the CTA, then ONE
    ``st.async`` v2.f32 exchange pushes the ``(local_max, cta_sum)`` pair
    into every peer CTA's ``cluster_n``-slot receive buffer; after the
    mbarrier wait every CTA folds the pairs with the online-softmax
    rescale ``sum += s_r * exp2((m_r - m) * scale)`` (``_frame_rescale``).
    Compared with two ``_cute_grouped_reduce_cluster`` calls this saves a
    full cluster round-trip and shrinks the fold from ``warps * cluster_n``
    to ``cluster_n`` slots.

    ``buf_ptr`` must point at ``cluster_n`` SMEM ``Int64`` slots (8-byte
    aligned pairs) and ``mbar_ptr`` at one initialized mbarrier; like
    ``_cute_grouped_reduce_cluster`` the mbarrier is single-phase, so each
    (textual) call site must execute exactly once per kernel.

    Returns ``(group_max, group_sum)``: the cluster-wide max and the sum
    rescaled into the ``group_max`` frame — ``sum_j exp2(f(x_j) -
    group_max * scale)`` (``sum_j exp(f(x_j) - group_max)`` for a ``None``
    scale) over the whole cluster row.
    """
    cta_sum = _cute_grouped_reduce_shared_two_stage(
        local_sum,
        "sum",
        cutlass.Float32(0.0),
        lane_var,
        lane_var,
        cutlass.Int32(0),
        pre=1,
        group_span=group_span,
        group_count=1,
    )
    pairs = cute.make_tensor(buf_ptr, (cluster_n,))
    vals = cute.make_tensor(
        cute.recast_ptr(buf_ptr, dtype=cutlass.Float32), (2 * cluster_n,)
    )
    rank = cutlass.Int32(cute.arch.block_idx_in_cluster())
    lane32 = cutlass.Int32(lane_var)
    if lane32 // 32 == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, cluster_n * 8)
    if lane32 < cluster_n:
        store_shared_remote_f32x2(
            cutlass.Float32(local_max),
            cutlass.Float32(cta_sum),
            pairs.iterator + rank,
            mbar_ptr,
            lane32,
        )
    cute.arch.mbarrier_wait(mbar_ptr, phase=0)
    if cutlass.const_expr(cluster_n <= 8):
        # Serial fold over the received pairs: no shuffles, and the small
        # LDS burst (<= 8 pairs) stays under the live-set peak.
        # NaN-propagating like the CTA-local max and torch.amax: a NaN in
        # any CTA's slice makes the row max NaN.
        group_max = vals[0]
        for w in cutlass.range_constexpr(1, cluster_n):
            group_max = cute.arch.fmax(group_max, vals[2 * w], nan=True)
        group_sum = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(cluster_n):
            group_sum = group_sum + vals[2 * w + 1] * _frame_rescale(
                vals[2 * w] - group_max, scale, fastmath
            )
    else:
        # cluster_n = 16: warp-parallel fold — each lane owns one received
        # pair, two warp reductions combine them.  At this width a serial
        # constexpr fold batches the 16 pair loads into one register burst
        # (32 registers) right where the cached exp values already peak
        # the live set; the warp shuffles keep the register profile flat.
        lane_in_warp = lane32 % 32
        slot = lane_in_warp % cluster_n
        pair_max = vals[2 * slot]
        pair_sum = vals[2 * slot + 1]
        if lane_in_warp >= cluster_n:
            pair_max = -cutlass.Float32.inf
            pair_sum = cutlass.Float32(0.0)
        if cutlass.const_expr(fp32_redux):
            group_max = _warp_reduce_max_native_fp32(pair_max, threads_in_group=32)
        else:
            group_max = cute.arch.warp_reduction_max(
                pair_max, op=_cute_nan_max, threads_in_group=32
            )
        rescaled = pair_sum * _frame_rescale(pair_max - group_max, scale, fastmath)
        group_sum = cute.arch.warp_reduction_sum(rescaled, threads_in_group=32)
    return group_max, group_sum


def _cute_grouped_reduce_cluster_online_pair(
    local_sum: cute.Numeric,
    local_max: cute.Numeric,
    lane_var: cutlass.Int32,
    buf_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    *,
    group_span: int,
    cluster_n: int,
    scale: float | None,
    fastmath: bool = False,
    fp32_redux: bool = False,
) -> tuple[cute.Numeric, cute.Numeric]:
    return _cute_grouped_reduce_cluster_online_pair_body(
        local_sum,
        local_max,
        lane_var,
        buf_ptr,
        mbar_ptr,
        group_span,
        cluster_n,
        scale,
        fastmath,
        fp32_redux,
    )


@cute.jit
def _cute_grouped_reduce_cluster_warp_online_pair_body(
    local_sum: cute.Numeric,
    local_max: cute.Numeric,
    lane_var: cutlass.Int32,
    buf_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    group_span: int,
    cluster_n: int,
    scale: float | None,
    fastmath: bool,
    fp32_redux: bool,
    owner_only: bool,
) -> tuple[cute.Numeric, cute.Numeric]:
    """Exchange one online-softmax pair per warp through one DSM barrier.

    ``local_max`` has already been reduced within each warp; ``local_sum``
    is the per-thread partial in that warp's maximum frame. Keeping both
    reductions within the warp removes the CTA-wide maximum and sum barriers.

    ``owner_only`` requires the single-row scalar epilogue proof. Every CTA
    still receives a complete local mirror and every thread waits for all
    incoming complete-tx bytes. The final fold uses only that local mirror;
    no remote receive buffer is read after another CTA can exit. Non-owners
    return a defined neutral maximum and sum that the proved epilogue ignores.
    """
    warp_sum = cute.arch.warp_reduction_sum(local_sum, threads_in_group=32)
    slots = group_span // 32 * cluster_n
    pairs = cute.make_tensor(buf_ptr, (slots,))
    vals = cute.make_tensor(
        cute.recast_ptr(buf_ptr, dtype=cutlass.Float32), (2 * slots,)
    )
    lane32 = cutlass.Int32(lane_var)
    lane_in_warp = lane32 % 32
    warp = lane32 // 32
    rank = cutlass.Int32(cute.arch.block_idx_in_cluster())
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, slots * 8)
    if lane_in_warp < cluster_n:
        store_shared_remote_f32x2(
            cutlass.Float32(local_max),
            cutlass.Float32(warp_sum),
            pairs.iterator + warp * cluster_n + rank,
            mbar_ptr,
            lane_in_warp,
        )
    cute.arch.mbarrier_wait(mbar_ptr, phase=0)
    group_max = -cutlass.Float32.inf
    group_sum = cutlass.Float32(0)
    fold_active = True
    if cutlass.const_expr(owner_only):
        fold_active = warp == 0 and rank == 0
    if fold_active:
        group_max = -cutlass.Float32.inf
        for index in cutlass.range_constexpr(-(-slots // 32)):
            slot = lane_in_warp + index * 32
            value = vals[2 * slot] if slot < slots else -cutlass.Float32.inf
            group_max = cute.arch.fmax(group_max, value, nan=True)
        if cutlass.const_expr(fp32_redux):
            group_max = _warp_reduce_max_native_fp32(group_max, threads_in_group=32)
        else:
            group_max = cute.arch.warp_reduction_max(
                group_max, op=_cute_nan_max, threads_in_group=32
            )
        group_sum = cutlass.Float32(0)
        for index in cutlass.range_constexpr(-(-slots // 32)):
            slot = lane_in_warp + index * 32
            if slot < slots:
                group_sum += vals[2 * slot + 1] * _frame_rescale(
                    vals[2 * slot] - group_max, scale, fastmath
                )
        group_sum = cute.arch.warp_reduction_sum(group_sum, threads_in_group=32)
    return (group_max, group_sum)


def _cute_grouped_reduce_cluster_warp_online_pair(
    local_sum: cute.Numeric,
    local_max: cute.Numeric,
    lane_var: cutlass.Int32,
    buf_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    *,
    group_span: int,
    cluster_n: int,
    scale: float | None,
    fastmath: bool = False,
    fp32_redux: bool = False,
    owner_only: bool = False,
) -> tuple[cute.Numeric, cute.Numeric]:
    return _cute_grouped_reduce_cluster_warp_online_pair_body(
        local_sum,
        local_max,
        lane_var,
        buf_ptr,
        mbar_ptr,
        group_span,
        cluster_n,
        scale,
        fastmath,
        fp32_redux,
        owner_only,
    )


# Fragment form of the two-stage grouped reduce: combines ``count`` per-thread
# accumulators (a register fragment) across the group with ONE pair of
# barriers instead of ``count`` two-stage reduces.  Stage 1 folds each value
# across the warp with a strided butterfly (xor offsets 16 down to ``pre``
# keep the interleaved sibling coordinate ``lane % pre`` fixed), one
# representative lane per sibling class stages the warp partials of all
# ``count`` values in shared memory, stage 2 has one thread per (sibling,
# value) slot fold the ``group_span // 32`` warp partials, and every thread
# reads back its sibling class's ``count`` results.  Both stages combine in
# the order of the form ``_cute_grouped_reduce_shared_two_stage`` dispatches
# the same call to, through the stages that form uses
# (``_cute_strided_warp_butterfly``, ``_cute_fold_warp_partials``), so the
# V-loop sink that replaces ``count`` per-lane reduces by this one
# (``sink_vector_loops``) leaves every result bitwise unchanged.
# ``finalize_shared_reduce_groups`` resizes both calls' ``group_count``, which
# picks the serial form.


@cute.jit
def _cute_grouped_reduce_shared_two_stage_fragment_body(
    values: cute.Tensor,
    results: cute.Tensor,
    combine: object,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    count: int,
    pre: int,
    group_span: int,
    group_count: int,
    serial: bool,
) -> None:
    dtype = type(identity)
    warps_per_group = group_span // 32
    slots = pre * count
    partials_size = group_count * slots * warps_per_group
    results_size = group_count * slots
    smem_ptr = cute.arch.alloc_smem(dtype, partials_size + results_size)
    smem = cute.make_tensor(smem_ptr, (partials_size + results_size,))
    group_id = lane_var // group_span
    lane_in_warp = lane_var % 32
    warp_in_group = lane_in_group_var // 32
    partials_base = group_id * (slots * warps_per_group)
    results_base = partials_size + group_id * slots
    for i in cutlass.range_constexpr(count):
        value = _cute_strided_warp_butterfly(values[i], combine, identity, pre=pre)
        if lane_in_warp < pre:
            # Warp-major partials: the ``pre`` writing lanes of a warp hit
            # ``count``-strided words and the second stage's consecutive
            # threads read consecutive words, so neither stage bank-conflicts
            # (slot-major placed the warps of one slot ``warps_per_group``
            # words apart: a 4-way store and a 16-way load conflict).
            smem[partials_base + warp_in_group * slots + lane_in_warp * count + i] = (
                value
            )
    cute.arch.sync_threads()
    if lane_in_group_var < slots:
        total = _cute_fold_warp_partials(
            smem,
            partials_base + lane_in_group_var,
            combine,
            identity,
            stride=slots,
            count=warps_per_group,
            serial=serial,
        )
        smem[results_base + lane_in_group_var] = total
    cute.arch.sync_threads()
    for i in cutlass.range_constexpr(count):
        results[i] = _smem_read(
            smem, results_base + lane_mod_pre_var * count + i, dtype
        )


def _cute_grouped_reduce_shared_two_stage_fragment(
    values: cute.Tensor,
    results: cute.Tensor,
    reduction_type: str,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    count: int,
    pre: int,
    group_span: int,
    group_count: int,
) -> None:
    """Reduce the ``count`` values of ``values`` into ``results`` across the
    ``group_span``-thread groups (``pre`` interleaved siblings kept apart)."""
    _cute_grouped_reduce_shared_two_stage_fragment_body(
        values,
        results,
        _cute_scalar_combine(reduction_type),
        identity,
        lane_var,
        lane_in_group_var,
        lane_mod_pre_var,
        count,
        pre,
        group_span,
        group_count,
        use_serial_block_reduce(pre, group_span, group_count),
    )
