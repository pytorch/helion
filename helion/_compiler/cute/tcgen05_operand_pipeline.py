"""Dormant raw-FP32 TMA to exact-RNA UMMA pipeline helpers.

These helpers do not enable a lowering. Their caller must own every lane of
each conversion warp, prove each operand's physical span, initialize pipelines
in a converged region, and register every active warp with the scheduler.
"""

from __future__ import annotations

from typing import cast

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline


@cute.jit
def make_converted_operand_pipeline(
    raw: pipeline.PipelineTmaUmma,
    stages: cutlass.Constexpr[int],
    converter_warps: cutlass.Constexpr[int],
    defer_sync: cutlass.Constexpr[bool] = False,
    barrier_storage: cute.Pointer | None = None,
) -> pipeline.PipelineAsyncUmma:
    """Allocate only converted-full barriers; UMMA retains the old empty ring.

    When ``defer_sync`` is true the caller must execute the common barrier-init
    fence and CTA synchronization after creating every pipeline, before roles
    diverge. This helper only supports one-CTA ownership.
    """
    assert stages > 0 and raw.num_stages == stages
    assert type(converter_warps) is int and converter_warps in (1, 2, 4, 8)
    assert raw.cta_group is cute.nvgpu.tcgen05.CtaGroup.ONE
    assert raw.producer_mask is None and raw.consumer_mask is None
    if cutlass.const_expr(barrier_storage is None):
        barriers = cute.arch.alloc_smem(cutlass.Int64, stages, alignment=8)
    else:
        # The complete caller's storage plan owns a disjoint, eight-byte
        # aligned Int64 cell per stage. This changes addresses only; raw-empty
        # remains the original UMMA-owned synchronization object below.
        assert barrier_storage is not None
        assert barrier_storage.dtype is cutlass.Int64
        assert barrier_storage.memspace is cute.AddressSpace.smem
        assert barrier_storage.alignment >= 8
        barriers = barrier_storage
    full = pipeline.PipelineAsync._make_sync_object(
        barriers,
        stages,
        (
            pipeline.PipelineOp.AsyncThread,
            pipeline.CooperativeGroup(pipeline.Agent.Thread, converter_warps * 32),
        ),
        name="helion_converted_operand",
        phase="full",
    )
    if cutlass.const_expr(not defer_sync):
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
    return pipeline.PipelineAsyncUmma(
        full,
        raw.sync_object_empty,
        stages,
        None,
        None,
        cute.nvgpu.tcgen05.CtaGroup.ONE,
    )


@cute.jit
def exact_tf32_rna_word(value: cutlass.Float32) -> cutlass.Float32:
    """Preserve the existing RNA sequence, including low-payload NaN behavior."""
    rounded = cutlass.Uint32(cute.arch.cvt_f32_tf32(value))
    prepared = rounded & cutlass.Uint32(0xFFFFE000)
    return cast("cutlass.Float32", prepared.bitcast(cutlass.Float32))


@cute.jit
def _convert_operand_stage(
    operand: cute.Tensor,
    stage: cutlass.Int32,
    words: cutlass.Constexpr[int],
    stride_words: cutlass.Constexpr[int],
    alignment_bytes: cutlass.Constexpr[int],
    converter_warps: cutlass.Constexpr[int],
    converter_warp_rank: cutlass.Int32,
) -> None:
    assert operand.element_type is cutlass.Float32
    pointer = operand.iterator
    assert isinstance(pointer, cute.Pointer)
    assert pointer.memspace is cute.AddressSpace.smem
    assert pointer.alignment >= alignment_bytes >= 16
    assert type(converter_warps) is int and converter_warps in (1, 2, 4, 8)
    assert words > 0 and words % (128 * converter_warps) == 0
    assert stride_words >= words and stride_words % 4 == 0
    # Recast without a swizzle removes pointer swizzling. Visit the proved
    # physical storage once, independently of the MMA coordinate permutation.
    base = cute.recast_ptr(pointer, dtype=cutlass.Float32)
    lane = converter_warp_rank * 32 + cute.arch.lane_idx()
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
    )
    for packet in cutlass.range(words // (128 * converter_warps), unroll=1):
        position = stage * stride_words + packet * (128 * converter_warps) + lane * 4
        position = cute.assume(position, divby=4)
        source = cute.make_tensor(base + position, cute.make_layout(4))
        registers = cute.make_rmem_tensor(4, cutlass.Float32)
        cute.copy(atom, source, registers)
        for item in cutlass.range_constexpr(4):
            registers[item] = exact_tf32_rna_word(registers[item])
        cute.copy(atom, registers, source)


@cute.jit
def convert_and_publish_operand_stage(
    raw: pipeline.PipelineTmaUmma,
    converted: pipeline.PipelineAsyncUmma,
    state: pipeline.PipelineState,
    a: cute.Tensor,
    b: cute.Tensor,
    a_words: cutlass.Constexpr[int],
    a_stride_words: cutlass.Constexpr[int],
    a_alignment_bytes: cutlass.Constexpr[int],
    b_words: cutlass.Constexpr[int],
    b_stride_words: cutlass.Constexpr[int],
    b_alignment_bytes: cutlass.Constexpr[int],
    converter_warps: cutlass.Constexpr[int],
    converter_warp_rank: cutlass.Int32,
) -> None:
    """Run on all conversion lanes, once for an actual occupied raw stage.

    The caller's role guard must prove ``0 <= converter_warp_rank <
    converter_warps``. Every warp advances its own consumer state exactly once
    after this call. An empty K loop makes no call and no advance. This function
    never recycles the shared empty barriers: only MMA completion may do that.
    """
    assert type(converter_warps) is int and converter_warps in (1, 2, 4, 8)
    raw.consumer_wait(state)
    _convert_operand_stage(
        a,
        state.index,
        a_words,
        a_stride_words,
        a_alignment_bytes,
        converter_warps,
        converter_warp_rank,
    )
    _convert_operand_stage(
        b,
        state.index,
        b_words,
        b_stride_words,
        b_alignment_bytes,
        converter_warps,
        converter_warp_rank,
    )
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    converted.producer_commit(state)
