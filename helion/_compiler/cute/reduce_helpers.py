# pyrefly: ignore-errors
from __future__ import annotations

import operator

import cutlass
import cutlass.cute as cute


@cute.jit
def _warp_reduce_sum(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction_sum(value, threads_in_group=threads_in_group)


@cute.jit
def _warp_reduce_max(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction_max(value, threads_in_group=threads_in_group)


@cute.jit
def _warp_reduce_min(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction(
        value,
        lambda a, b: min(b, a),
        threads_in_group=threads_in_group,
    )


@cute.jit
def _warp_reduce_prod(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction(
        value,
        operator.mul,
        threads_in_group=threads_in_group,
    )


@cute.jit
def _cute_grouped_reduce_warp_sum(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
) -> cute.Numeric:
    lane_in_group = lane_expr % group_span
    lane_mod_pre = lane_in_group % pre
    selected = _warp_reduce_sum(
        input_value if lane_mod_pre == 0 else identity,
        threads_in_group=group_span,
    )
    for p in cutlass.range_constexpr(1, pre):
        reduced = _warp_reduce_sum(
            input_value if lane_mod_pre == p else identity,
            threads_in_group=group_span,
        )
        selected = reduced if lane_mod_pre == p else selected
    return selected


@cute.jit
def _cute_grouped_reduce_warp_max(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
) -> cute.Numeric:
    lane_in_group = lane_expr % group_span
    lane_mod_pre = lane_in_group % pre
    selected = _warp_reduce_max(
        input_value if lane_mod_pre == 0 else identity,
        threads_in_group=group_span,
    )
    for p in cutlass.range_constexpr(1, pre):
        reduced = _warp_reduce_max(
            input_value if lane_mod_pre == p else identity,
            threads_in_group=group_span,
        )
        selected = reduced if lane_mod_pre == p else selected
    return selected


@cute.jit
def _cute_grouped_reduce_warp_min(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
) -> cute.Numeric:
    lane_in_group = lane_expr % group_span
    lane_mod_pre = lane_in_group % pre
    selected = _warp_reduce_min(
        input_value if lane_mod_pre == 0 else identity,
        threads_in_group=group_span,
    )
    for p in cutlass.range_constexpr(1, pre):
        reduced = _warp_reduce_min(
            input_value if lane_mod_pre == p else identity,
            threads_in_group=group_span,
        )
        selected = reduced if lane_mod_pre == p else selected
    return selected


@cute.jit
def _cute_grouped_reduce_warp_prod(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
) -> cute.Numeric:
    lane_in_group = lane_expr % group_span
    lane_mod_pre = lane_in_group % pre
    selected = _warp_reduce_prod(
        input_value if lane_mod_pre == 0 else identity,
        threads_in_group=group_span,
    )
    for p in cutlass.range_constexpr(1, pre):
        reduced = _warp_reduce_prod(
            input_value if lane_mod_pre == p else identity,
            threads_in_group=group_span,
        )
        selected = reduced if lane_mod_pre == p else selected
    return selected


_WARP_DISPATCH = {
    "sum": _cute_grouped_reduce_warp_sum,
    "max": _cute_grouped_reduce_warp_max,
    "min": _cute_grouped_reduce_warp_min,
    "prod": _cute_grouped_reduce_warp_prod,
}


def _cute_grouped_reduce_warp(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    pre: int,
    group_span: int,
) -> cute.Numeric:
    impl = _WARP_DISPATCH.get(reduction_type)
    if impl is None:
        raise ValueError(f"unsupported CuTe reduction type: {reduction_type!r}")
    return impl(input_value, identity, lane_expr, pre=pre, group_span=group_span)


@cute.jit
def _cute_grouped_reduce_shared_two_stage_sum(
    input_value: cute.Numeric,
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
        warp_partial = _warp_reduce_sum(masked_input, threads_in_group=32)
        partial_idx = partials_base + p * warps_per_group + warp_in_group
        if lane_in_warp == 0:
            smem[partial_idx] = warp_partial
        cute.arch.sync_threads()

        if warp_in_group == 0:
            stage2_input = (
                smem[partials_base + p * warps_per_group + lane_in_warp]
                if lane_in_warp < warps_per_group
                else identity
            )
            group_result = _warp_reduce_sum(stage2_input, threads_in_group=32)
            if lane_in_warp == 0:
                smem[results_base + p] = group_result
        cute.arch.sync_threads()

    return smem[results_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_two_stage_max(
    input_value: cute.Numeric,
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
        warp_partial = _warp_reduce_max(masked_input, threads_in_group=32)
        partial_idx = partials_base + p * warps_per_group + warp_in_group
        if lane_in_warp == 0:
            smem[partial_idx] = warp_partial
        cute.arch.sync_threads()

        if warp_in_group == 0:
            stage2_input = (
                smem[partials_base + p * warps_per_group + lane_in_warp]
                if lane_in_warp < warps_per_group
                else identity
            )
            group_result = _warp_reduce_max(stage2_input, threads_in_group=32)
            if lane_in_warp == 0:
                smem[results_base + p] = group_result
        cute.arch.sync_threads()

    return smem[results_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_two_stage_min(
    input_value: cute.Numeric,
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
        warp_partial = _warp_reduce_min(masked_input, threads_in_group=32)
        partial_idx = partials_base + p * warps_per_group + warp_in_group
        if lane_in_warp == 0:
            smem[partial_idx] = warp_partial
        cute.arch.sync_threads()

        if warp_in_group == 0:
            stage2_input = (
                smem[partials_base + p * warps_per_group + lane_in_warp]
                if lane_in_warp < warps_per_group
                else identity
            )
            group_result = _warp_reduce_min(stage2_input, threads_in_group=32)
            if lane_in_warp == 0:
                smem[results_base + p] = group_result
        cute.arch.sync_threads()

    return smem[results_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_two_stage_prod(
    input_value: cute.Numeric,
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
        warp_partial = _warp_reduce_prod(masked_input, threads_in_group=32)
        partial_idx = partials_base + p * warps_per_group + warp_in_group
        if lane_in_warp == 0:
            smem[partial_idx] = warp_partial
        cute.arch.sync_threads()

        if warp_in_group == 0:
            stage2_input = (
                smem[partials_base + p * warps_per_group + lane_in_warp]
                if lane_in_warp < warps_per_group
                else identity
            )
            group_result = _warp_reduce_prod(stage2_input, threads_in_group=32)
            if lane_in_warp == 0:
                smem[results_base + p] = group_result
        cute.arch.sync_threads()

    return smem[results_base + lane_mod_pre_var]


_TWO_STAGE_DISPATCH = {
    "sum": _cute_grouped_reduce_shared_two_stage_sum,
    "max": _cute_grouped_reduce_shared_two_stage_max,
    "min": _cute_grouped_reduce_shared_two_stage_min,
    "prod": _cute_grouped_reduce_shared_two_stage_prod,
}


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
    result = smem[0]
    for w in cutlass.range_constexpr(1, warps):
        result = combine(result, smem[w])
    cute.arch.sync_threads()
    return result


def _cute_scalar_combine_max(a: cute.Numeric, b: cute.Numeric) -> cute.Numeric:
    return cute.arch.fmax(a, b)


@cute.jit
def _cute_scalar_combine_min(a: cute.Numeric, b: cute.Numeric) -> cute.Numeric:
    return min(b, a)


def _cute_scalar_combine_f32_min(a: cute.Numeric, b: cute.Numeric) -> cute.Numeric:
    return cute.arch.fmin(a, b)


@cute.jit
def _cute_scalar_combine_generic_max(a: cute.Numeric, b: cute.Numeric) -> cute.Numeric:
    return max(b, a)


_SERIAL_DISPATCH = {
    "sum": (_warp_reduce_sum, operator.add),
    "prod": (_warp_reduce_prod, operator.mul),
}

# Above this many warps the serial fold's per-thread chain outgrows the
# two-stage form's constant shuffle cost.
_SERIAL_MAX_WARPS = 8


def _cute_grouped_reduce_shared_serial(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    *,
    group_span: int,
) -> cute.Numeric:
    if reduction_type == "max":
        warp_op = _warp_reduce_max
        # cute.arch.fmax/fmin are one FMNMX; Python max/min lower to
        # compare+select.
        combine = (
            _cute_scalar_combine_max
            if type(identity) is cutlass.Float32
            else _cute_scalar_combine_generic_max
        )
    elif reduction_type == "min":
        warp_op = _warp_reduce_min
        combine = (
            _cute_scalar_combine_f32_min
            if type(identity) is cutlass.Float32
            else _cute_scalar_combine_min
        )
    else:
        entry = _SERIAL_DISPATCH.get(reduction_type)
        if entry is None:
            raise ValueError(f"unsupported CuTe reduction type: {reduction_type!r}")
        warp_op, combine = entry
    return _cute_grouped_reduce_shared_serial_body(
        input_value,
        warp_op,
        combine,
        identity,
        lane_var,
        group_span,
    )


def _use_serial_block_reduce(pre: int, group_span: int, group_count: int) -> bool:
    # group_count == 1 with pre == 1 means ONE group spanning the whole
    # CTA at every current emitter (they all size group_count as
    # num_threads // group_span); the serial body's warp indexing relies
    # on that invariant.
    return (
        pre == 1
        and group_count == 1
        and group_span % 32 == 0
        and 2 <= group_span // 32 <= _SERIAL_MAX_WARPS
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
) -> cute.Numeric:
    # Despite the name (kept for call-site stability) this dispatcher owns
    # ALL shared-memory cross-warp combines: small whole-CTA groups route
    # to the cheaper serial form, everything else to the two-stage form.
    if _use_serial_block_reduce(pre, group_span, group_count):
        return _cute_grouped_reduce_shared_serial(
            input_value,
            reduction_type,
            identity,
            lane_var,
            group_span=group_span,
        )
    impl = _TWO_STAGE_DISPATCH.get(reduction_type)
    if impl is None:
        raise ValueError(f"unsupported CuTe reduction type: {reduction_type!r}")
    return impl(
        input_value,
        identity,
        lane_var,
        lane_in_group_var,
        lane_mod_pre_var,
        pre=pre,
        group_span=group_span,
        group_count=group_count,
    )


@cute.jit
def _cute_grouped_reduce_shared_tree_sum(
    input_value: cute.Numeric,
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
                smem[lane_var] = (
                    smem[lane_var] + smem[group_base + lane_in_group_var + stride]
                )
            cute.arch.sync_threads()
            stride *= 2

        if lane_in_group_var == 0:
            smem[result_base + p] = smem[lane_var]
        cute.arch.sync_threads()

    return smem[result_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_tree_max(
    input_value: cute.Numeric,
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
                lhs = smem[lane_var]
                rhs = smem[group_base + lane_in_group_var + stride]
                smem[lane_var] = max(rhs, lhs)
            cute.arch.sync_threads()
            stride *= 2

        if lane_in_group_var == 0:
            smem[result_base + p] = smem[lane_var]
        cute.arch.sync_threads()

    return smem[result_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_tree_min(
    input_value: cute.Numeric,
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
                lhs = smem[lane_var]
                rhs = smem[group_base + lane_in_group_var + stride]
                smem[lane_var] = min(rhs, lhs)
            cute.arch.sync_threads()
            stride *= 2

        if lane_in_group_var == 0:
            smem[result_base + p] = smem[lane_var]
        cute.arch.sync_threads()

    return smem[result_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_tree_prod(
    input_value: cute.Numeric,
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
                smem[lane_var] = (
                    smem[lane_var] * smem[group_base + lane_in_group_var + stride]
                )
            cute.arch.sync_threads()
            stride *= 2

        if lane_in_group_var == 0:
            smem[result_base + p] = smem[lane_var]
        cute.arch.sync_threads()

    return smem[result_base + lane_mod_pre_var]


_TREE_DISPATCH = {
    "sum": _cute_grouped_reduce_shared_tree_sum,
    "max": _cute_grouped_reduce_shared_tree_max,
    "min": _cute_grouped_reduce_shared_tree_min,
    "prod": _cute_grouped_reduce_shared_tree_prod,
}


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
    impl = _TREE_DISPATCH.get(reduction_type)
    if impl is None:
        raise ValueError(f"unsupported CuTe reduction type: {reduction_type!r}")
    return impl(
        input_value,
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
def _cute_argmax_index_impl(
    smem: cute.Tensor,
    valid_smem: cute.Tensor,
    start_idx: cutlass.Int32,
    stride: cutlass.Int32,
    *,
    extent: cutlass.Constexpr[int],
) -> cutlass.Int64:
    best_index = cutlass.Int64(0)
    best_value = smem[start_idx]
    best_valid = valid_smem[start_idx]
    for candidate_index in cutlass.range_constexpr(1, extent):
        candidate_offset = start_idx + stride * candidate_index
        candidate = smem[candidate_offset]
        candidate_valid = valid_smem[candidate_offset]
        better = candidate_valid != cutlass.Int32(0) and (
            best_valid == cutlass.Int32(0)
            or (
                best_valid != cutlass.Int32(0)
                and (
                    candidate > best_value
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


@cute.jit
def _cute_argmin_index_impl(
    smem: cute.Tensor,
    valid_smem: cute.Tensor,
    start_idx: cutlass.Int32,
    stride: cutlass.Int32,
    *,
    extent: cutlass.Constexpr[int],
) -> cutlass.Int64:
    best_index = cutlass.Int64(0)
    best_value = smem[start_idx]
    best_valid = valid_smem[start_idx]
    for candidate_index in cutlass.range_constexpr(1, extent):
        candidate_offset = start_idx + stride * candidate_index
        candidate = smem[candidate_offset]
        candidate_valid = valid_smem[candidate_offset]
        better = candidate_valid != cutlass.Int32(0) and (
            best_valid == cutlass.Int32(0)
            or (
                best_valid != cutlass.Int32(0)
                and (
                    candidate < best_value
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


_ARGREDUCE_DISPATCH = {
    "argmax": _cute_argmax_index_impl,
    "argmin": _cute_argmin_index_impl,
}


def _cute_argreduce_index(
    smem: cute.Tensor,
    valid_smem: cute.Tensor,
    start_idx: cutlass.Int32,
    stride: cutlass.Int32,
    *,
    extent: int,
    reduction_type: str,
) -> cutlass.Int64:
    impl = _ARGREDUCE_DISPATCH.get(reduction_type)
    if impl is None:
        raise ValueError(f"unsupported CuTe argreduce type: {reduction_type!r}")
    return impl(smem, valid_smem, start_idx, stride, extent=extent)


# Per-thread V-fold helpers for vectorized loads.  Used by the looped
# reduction strategy to collapse a length-V vector load into a scalar
# before the warp-level reduction.


@cute.jit
def _cute_pre_vec_fold_sum(vec: object, *, V: cutlass.Constexpr[int]) -> object:
    acc = vec[0]
    for i in cutlass.range_constexpr(1, V):
        acc = acc + vec[i]
    return acc


@cute.jit
def _cute_pre_vec_fold_max(vec: object, *, V: cutlass.Constexpr[int]) -> object:
    acc = vec[0]
    for i in cutlass.range_constexpr(1, V):
        candidate = vec[i]
        acc = max(acc, candidate)
    return acc


@cute.jit
def _cute_pre_vec_fold_min(vec: object, *, V: cutlass.Constexpr[int]) -> object:
    acc = vec[0]
    for i in cutlass.range_constexpr(1, V):
        candidate = vec[i]
        acc = min(acc, candidate)
    return acc


@cute.jit
def _cute_pre_vec_fold_prod(vec: object, *, V: cutlass.Constexpr[int]) -> object:
    acc = vec[0]
    for i in cutlass.range_constexpr(1, V):
        acc = acc * vec[i]
    return acc


def _cute_pre_vec_fold(vec: object, reduction_type: str, *, V: int) -> object:
    if reduction_type == "sum":
        return _cute_pre_vec_fold_sum(vec, V=V)
    if reduction_type == "max":
        return _cute_pre_vec_fold_max(vec, V=V)
    if reduction_type == "min":
        return _cute_pre_vec_fold_min(vec, V=V)
    if reduction_type == "prod":
        return _cute_pre_vec_fold_prod(vec, V=V)
    raise ValueError(f"unsupported CuTe pre-vec-fold type: {reduction_type!r}")


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
    partials locally.  Each (textual) call site must execute exactly once
    per kernel — the mbarrier is single-phase.

    ``buf_ptr`` (``warps * cluster_n`` Float32 slots) and ``mbar_ptr`` (one
    Int64 mbarrier, arrival count 1) must be allocated and initialized in
    the kernel preamble, followed by ``mbarrier_init_fence`` +
    ``cluster_arrive_relaxed`` + ``cluster_wait`` so peers' barriers are
    live before the first remote store (the device-function codegen emits
    this once for all sites).

    ``identity``'s dtype must be Float32 (the hoist pass rewrites the
    identity to the fp32 accumulator dtype).
    """
    from helion._compiler.cute.cluster_helpers import store_shared_remote_f32

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
    for i in cutlass.range_constexpr(slots):
        result = combine(result, buf[i])
    return result


_CLUSTER_DISPATCH = {
    "sum": (_warp_reduce_sum, operator.add),
    "max": (_warp_reduce_max, _cute_scalar_combine_max),
    "min": (_warp_reduce_min, _cute_scalar_combine_min),
}


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
) -> cute.Numeric:
    impl = _CLUSTER_DISPATCH.get(reduction_type)
    if impl is None:
        raise ValueError(f"unsupported CuTe reduction type: {reduction_type!r}")
    warp_op, combine = impl
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
    )


@cute.jit
def _cute_grouped_reduce_cluster_online_pair_body(
    local_sum: cute.Numeric,
    local_max: cute.Numeric,
    lane_var: cutlass.Int32,
    buf_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    group_span: int,
    cluster_n: int,
    scale: float,
    fastmath: bool,
) -> tuple[cute.Numeric, cute.Numeric]:
    """Single-exchange cluster combine for an online-softmax (max, sum)
    pair.  ``local_max`` is this CTA's slice max (CTA-uniform; relocalized
    site-A result) and ``local_sum`` is the per-thread partial of
    ``sum(exp2(f(x) - local_max * scale))`` over the CTA's slice.

    The per-thread sum is block-reduced within the CTA, then ONE
    ``st.async`` v2.f32 exchange pushes the ``(local_max, cta_sum)`` pair
    into every peer CTA's ``cluster_n``-slot receive buffer; after the
    mbarrier wait every CTA folds the pairs with the online-softmax
    rescale ``sum += s_r * exp2((m_r - m) * scale)``.  Compared with two
    ``_cute_grouped_reduce_cluster`` calls this saves a full cluster
    round-trip and shrinks the fold from ``warps * cluster_n`` to
    ``cluster_n`` slots.

    ``buf_ptr`` must point at ``cluster_n`` SMEM ``Int64`` slots (8-byte
    aligned pairs) and ``mbar_ptr`` at one initialized mbarrier; like
    ``_cute_grouped_reduce_cluster`` the mbarrier is single-phase, so each
    (textual) call site must execute exactly once per kernel.

    Returns ``(group_max, group_sum)``: the cluster-wide max and the sum
    rescaled into the ``group_max`` frame — ``sum_j exp2(f(x_j) -
    group_max * scale)`` over the whole cluster row.
    """
    from helion._compiler.cute.cluster_helpers import store_shared_remote_f32x2

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
        group_max = vals[0]
        for w in cutlass.range_constexpr(1, cluster_n):
            group_max = cute.arch.fmax(group_max, vals[2 * w])
        group_sum = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(cluster_n):
            group_sum = group_sum + vals[2 * w + 1] * cute.math.exp2(
                (vals[2 * w] - group_max) * scale, fastmath=fastmath
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
        group_max = cute.arch.warp_reduction_max(pair_max, threads_in_group=32)
        rescaled = pair_sum * cute.math.exp2(
            (pair_max - group_max) * scale, fastmath=fastmath
        )
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
    scale: float,
    fastmath: bool = False,
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
    )
