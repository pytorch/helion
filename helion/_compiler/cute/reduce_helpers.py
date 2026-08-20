# pyrefly: ignore-errors
from __future__ import annotations

import operator

import cutlass
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import T


@cute.jit
def _cute_bf16x2_fma(
    lhs: cutlass.Uint32,
    rhs: cutlass.Uint32,
    acc: cutlass.Uint32,
) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [lhs.ir_value(), rhs.ir_value(), acc.ir_value()],
            "fma.rn.bf16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _cute_bf16x2_accumulate3(
    branch_pair: cutlass.Uint32,
    stream_pair: cutlass.Uint32,
    branch_acc: cutlass.Uint32,
    stream_acc: cutlass.Uint32,
    cross_acc: cutlass.Uint32,
) -> tuple[cutlass.Uint32, cutlass.Uint32, cutlass.Uint32]:
    """Accumulate three packed BF16x2 products with one rounding each."""

    return (
        _cute_bf16x2_fma(branch_pair, branch_pair, branch_acc),
        _cute_bf16x2_fma(stream_pair, stream_pair, stream_acc),
        _cute_bf16x2_fma(stream_pair, branch_pair, cross_acc),
    )


@cute.jit
def _cute_bf16x2_sum_to_f32(value: cutlass.Uint32) -> cutlass.Float32:
    """Widen both halves of a BF16x2 word and return their FP32 sum."""
    lo = cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [value.ir_value()],
            "{ .reg .b32 t; shl.b32 t, $1, 16; mov.b32 $0, t; }",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    hi = cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [value.ir_value()],
            "{ .reg .b32 t; and.b32 t, $1, 0xffff0000; mov.b32 $0, t; }",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return lo + hi


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
def _cute_grouped_reduce_shared_two_stage_sum3(
    input_value_0: cute.Numeric,
    input_value_1: cute.Numeric,
    input_value_2: cute.Numeric,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> tuple[cute.Numeric, cute.Numeric, cute.Numeric]:
    """Reduce three independent sums while sharing the CTA barriers."""
    dtype = type(identity)
    warps_per_group = group_span // 32
    partials_size = group_count * pre * warps_per_group
    results_size = group_count * pre
    value_stride = partials_size + results_size
    smem_ptr = cute.arch.alloc_smem(dtype, 3 * value_stride)
    smem = cute.make_tensor(smem_ptr, (3 * value_stride,))
    group_id = lane_var // group_span
    lane_in_warp = lane_var % 32
    warp_in_group = lane_in_group_var // 32
    partials_base = group_id * (pre * warps_per_group)
    results_base = partials_size + group_id * pre

    for p in cutlass.range_constexpr(pre):
        masked_0 = input_value_0 if lane_mod_pre_var == p else identity
        masked_1 = input_value_1 if lane_mod_pre_var == p else identity
        masked_2 = input_value_2 if lane_mod_pre_var == p else identity
        warp_partial_0 = _warp_reduce_sum(masked_0, threads_in_group=32)
        warp_partial_1 = _warp_reduce_sum(masked_1, threads_in_group=32)
        warp_partial_2 = _warp_reduce_sum(masked_2, threads_in_group=32)
        partial_idx = partials_base + p * warps_per_group + warp_in_group
        if lane_in_warp == 0:
            smem[partial_idx] = warp_partial_0
            smem[value_stride + partial_idx] = warp_partial_1
            smem[2 * value_stride + partial_idx] = warp_partial_2
        cute.arch.sync_threads()

        if warp_in_group == 0:
            stage2_idx = partials_base + p * warps_per_group + lane_in_warp
            stage2_0 = smem[stage2_idx] if lane_in_warp < warps_per_group else identity
            stage2_1 = (
                smem[value_stride + stage2_idx]
                if lane_in_warp < warps_per_group
                else identity
            )
            stage2_2 = (
                smem[2 * value_stride + stage2_idx]
                if lane_in_warp < warps_per_group
                else identity
            )
            group_result_0 = _warp_reduce_sum(stage2_0, threads_in_group=32)
            group_result_1 = _warp_reduce_sum(stage2_1, threads_in_group=32)
            group_result_2 = _warp_reduce_sum(stage2_2, threads_in_group=32)
            if lane_in_warp == 0:
                smem[results_base + p] = group_result_0
                smem[value_stride + results_base + p] = group_result_1
                smem[2 * value_stride + results_base + p] = group_result_2
        cute.arch.sync_threads()

    result_idx = results_base + lane_mod_pre_var
    return (
        smem[result_idx],
        smem[value_stride + result_idx],
        smem[2 * value_stride + result_idx],
    )


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
