"""SM100 native E2M1/UE4M3 and E4M3/UE8M0 block-scaled pipelines.

Prepared operands contain only byte permutations, sign normalization and zero
padding. The native consumer keeps them packed through TMA and shared memory.
MXFP8 operands are read directly from their original K-major tensors; only
their scale bytes need preparation into the native 128-row, four-group layout.
Its producer, MMA and output roles communicate through CUTLASS pipelines. Scale
TMEM is disjoint from the accumulator, including while the epilogue drains it.
"""

from __future__ import annotations

from typing import cast

from cuda.bindings.driver import (  # pyrefly: ignore [missing-import]
    CUstream,  # noqa: TC002
)
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.utils import blackwell_helpers
from cutlass.utils import blockscaled_layout
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout
from cutlass.utils.layout import LayoutEnum

from .block_scaled_config import block_scaled_workspace_shape


@dsl_user_op
def _scale_copy(
    shared: cute.Tensor,
    tensor: cute.Tensor,
    stage: cutlass.Int32,
    group: tcgen05.CtaGroup,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> None:
    source = cute.filter_zeros(shared, loc=loc, ip=ip)
    target = cute.filter_zeros(tensor, loc=loc, ip=ip)
    atom = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(group), tensor.element_type, loc=loc, ip=ip
    )
    copy = tcgen05.make_s2t_copy(atom, target, loc=loc, ip=ip)
    thread = copy.get_slice(0)
    descriptor = tcgen05.get_s2t_smem_desc_tensor(
        copy, thread.partition_S(source), loc=loc, ip=ip
    )
    cute.copy(
        copy,
        descriptor[None, None, None, None, stage],
        thread.partition_D(target),
        loc=loc,
        ip=ip,
    )


def _compact_a_layouts(
    full: cute.ComposedLayout, logical_rows: int, bk: int
) -> tuple[cute.ComposedLayout, cute.ComposedLayout]:
    """Keep the MMA row view while copying only its live 128-byte K windows."""
    k_stride = cute.get(full.outer.stride, 2)
    # A single 128-byte window has a scalar stride. Wider tiles retain a
    # second K mode whose window stride follows the compact row count.
    if cute.rank(cute.get(full.outer.shape, 2)) == 2:
        k_stride = (cute.get(k_stride, 0), logical_rows * 128)
    strides = (
        cute.get(full.outer.stride, 0),
        cute.get(full.outer.stride, 1),
        k_stride,
        logical_rows * bk,
    )
    mma = cute.make_composed_layout(
        full.inner, 0, cute.make_layout(full.outer.shape, stride=strides)
    )
    load = cute.make_composed_layout(
        full.inner,
        0,
        cute.make_layout(
            (logical_rows, (128, bk // 128), 1),
            stride=(128, (1, logical_rows * 128), 0),
        ),
    )
    return mma, load


@cute.jit
def _transpose_sf_stage_inplace(
    raw_sa: cute.Tensor,
    raw_sb: cute.Tensor,
    stage: cutlass.Int32,
    k_block: cutlass.Int32,
    m_start: cutlass.Int32,
    n_start: cutlass.Int32,
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    k: cutlass.Constexpr,
    bk: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    logical_m: cutlass.Constexpr,
) -> None:
    """Read a complete TMA scale stage before permuting its storage in place."""
    lane = cutlass.Int32(cute.arch.thread_idx()[0]) % 32
    k_chunks = bk // 128
    base_a = stage * (128 * k_chunks)
    base_b = stage * (128 * n_chunks * k_chunks)
    words_a = cute.make_rmem_tensor((k_chunks, 4), cutlass.Int32)
    words_b = cute.make_rmem_tensor((k_chunks, n_chunks, 4), cutlass.Int32)
    for chunk in cutlass.range_constexpr(k_chunks):
        word_k = k_block * k_chunks + chunk
        for segment in cutlass.range_constexpr(4):
            row = lane + segment * 32
            value = cutlass.Int32(0)
            if m_start + row < m and row < logical_m and word_k < (k + 127) // 128:
                value = raw_sa[base_a + chunk * 128 + row]
            if cutlass.const_expr(k % 128 != 0):
                if word_k == k // 128:
                    value = value & ((1 << ((k % 128) // 32 * 8)) - 1)
            words_a[chunk, segment] = value
        for n_chunk in cutlass.range_constexpr(n_chunks):
            for segment in cutlass.range_constexpr(4):
                row = lane + segment * 32
                value = cutlass.Int32(0)
                if n_start + n_chunk * 128 + row < n and word_k < (k + 127) // 128:
                    value = raw_sb[
                        base_b + chunk * (128 * n_chunks) + n_chunk * 128 + row
                    ]
                if cutlass.const_expr(k % 128 != 0):
                    if word_k == k // 128:
                        value = value & ((1 << ((k % 128) // 32 * 8)) - 1)
                words_b[chunk, n_chunk, segment] = value
    # B also permutes its K-chunk and N-chunk modes. Preload both complete
    # buffers before any stores, even when these outer modes overlap.
    cute.arch.sync_warp()
    for chunk in cutlass.range_constexpr(k_chunks):
        for segment in cutlass.range_constexpr(4):
            raw_sa[base_a + chunk * 128 + lane * 4 + segment] = words_a[chunk, segment]
        for n_chunk in cutlass.range_constexpr(n_chunks):
            for segment in cutlass.range_constexpr(4):
                raw_sb[
                    base_b
                    + n_chunk * (128 * k_chunks)
                    + chunk * 128
                    + lane * 4
                    + segment
                ] = words_b[chunk, n_chunk, segment]


class Sm100BlockScaledGemm:
    """One native tile family with independently tunable K and stage depth."""

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        bn: int,
        bk: int,
        stages: int,
        cluster_m: int,
        persistent: bool,
        num_sm: int,
        round_dtypes: tuple[type[cutlass.Numeric], ...] = (),
        *,
        operand_format: str = "nvfp4",
        fused_scales: bool = False,
        sf_tma: bool = False,
        sf_inplace: bool = False,
        compact_m: bool = False,
        logical_m_per_cta: int = 128,
        lhs_scale_row_stride: int = 1,
        lhs_scale_column_stride: int = 0,
        rhs_scale_row_stride: int = 1,
        rhs_scale_column_stride: int = 0,
    ) -> None:
        assert operand_format in ("nvfp4", "mxfp8")
        self.operand_format = operand_format
        self.fused_scales = fused_scales
        self.sf_tma = sf_tma
        self.sf_inplace = sf_inplace
        assert not sf_tma or fused_scales
        assert not sf_inplace or (sf_tma and fused_scales)
        self.compact_m = compact_m and logical_m_per_cta < 128 and m > 1
        self.logical_m_per_cta = logical_m_per_cta
        assert logical_m_per_cta in (32, 64, 128)
        assert fused_scales or logical_m_per_cta == 128
        self.lhs_scale_row_stride = lhs_scale_row_stride
        self.lhs_scale_column_stride = lhs_scale_column_stride
        self.rhs_scale_row_stride = rhs_scale_row_stride
        self.rhs_scale_column_stride = rhs_scale_column_stride
        self.operand_dtype = (
            cutlass.Float4E2M1FN if operand_format == "nvfp4" else cutlass.Float8E4M3FN
        )
        self.scale_dtype = (
            cutlass.Float8E4M3FN if operand_format == "nvfp4" else cutlass.Float8E8M0FNU
        )
        self.scale_vector_size = 16 if operand_format == "nvfp4" else 32
        self.mma_k = 64 if operand_format == "nvfp4" else 32
        self.scale_chunk_k = 4 * self.scale_vector_size
        assert m > 0 and n > 0 and k > 0 and k % self.scale_vector_size == 0
        assert bn in (
            (8, 16, 32, 64, 128, 224, 256) if operand_format == "mxfp8" else (128, 256)
        )
        assert bn != 8 or cluster_m == 1
        assert bk in (64, 128, 256)
        assert bk % self.scale_chunk_k == 0
        assert stages in (2, 3, 4, 5, 6, 8, 10, 12) and cluster_m in (1, 2)
        self.m, self.n, self.k = m, n, k
        self.mp, self.np, self.kp = block_scaled_workspace_shape(m, n, k, bn, bk)
        self.bm, self.bn, self.bk = 128 * cluster_m, bn, bk
        self.sf_n_chunks = (bn + 127) // 128
        self.stages = stages
        self.epi_n = min(bn, 64)
        self.cluster_m = cluster_m
        self.round_dtypes = round_dtypes
        self.group = tcgen05.CtaGroup.ONE if cluster_m == 1 else tcgen05.CtaGroup.TWO
        self.tile = (self.bm, bn, bk)
        self.qa_bytes = self.mp * self.kp // 2 if operand_format == "nvfp4" else 0
        self.qb_bytes = self.np * self.kp // 2 if operand_format == "nvfp4" else 0
        self.sa_bytes = self.mp * self.kp // self.scale_vector_size
        self.sb_bytes = self.np * self.kp // self.scale_vector_size
        self.qb_offset = self.qa_bytes
        self.sa_offset = self.qb_offset + self.qb_bytes
        self.sb_offset = self.sa_offset + self.sa_bytes
        self.workspace_bytes = self.sb_offset + self.sb_bytes
        self.m_tiles = (m + self.logical_m_per_cta * cluster_m - 1) // (
            self.logical_m_per_cta * cluster_m
        )
        self.n_tiles = (n + bn - 1) // bn
        self.tile_count = self.m_tiles * self.n_tiles
        self.clusters = (
            min(self.tile_count, max(1, num_sm // cluster_m))
            if persistent
            else self.tile_count
        )

    @cute.jit
    def __call__(
        self,
        workspace: cute.Tensor,
        output: cute.Tensor,
        alpha: cutlass.Float32,
        stream: CUstream,
    ) -> None:
        assert self.operand_format == "nvfp4"
        workspace_pointer = cast("cute.Pointer", workspace.iterator)
        qa = cute.make_tensor(
            cute.recast_ptr(workspace_pointer, dtype=cutlass.Float4E2M1FN),
            cute.make_layout((self.mp, self.kp, 1), stride=(self.kp, 1, 0)),
        )
        qb = cute.make_tensor(
            cute.recast_ptr(
                workspace_pointer + self.qb_offset, dtype=cutlass.Float4E2M1FN
            ),
            cute.make_layout((self.np, self.kp, 1), stride=(self.kp, 1, 0)),
        )
        sa = cute.make_tensor(
            cute.recast_ptr(workspace_pointer + self.sa_offset, dtype=cutlass.Int16),
            cute.make_layout((256, self.kp // 64, self.mp // 128)),
        )
        sb = cute.make_tensor(
            cute.recast_ptr(workspace_pointer + self.sb_offset, dtype=cutlass.Int16),
            cute.make_layout((256, self.kp // 64, self.np // 128)),
        )
        self._launch(qa, qb, sa, sb, output, alpha, stream)

    @cute.jit
    def mxfp8(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        workspace: cute.Tensor,
        output: cute.Tensor,
        alpha: cutlass.Float32,
        stream: CUstream,
    ) -> None:
        """Read native FP8 operands directly and consume prepared E8M0 scales."""
        assert self.operand_format == "mxfp8"
        a_strides = cast("tuple[int | cutlass.Int64, int | cutlass.Int64]", a.stride)
        b_strides = cast("tuple[int | cutlass.Int64, int | cutlass.Int64]", b.stride)
        qa = cute.make_tensor(
            cute.recast_ptr(a.iterator, dtype=self.operand_dtype),
            cute.make_layout((self.m, self.k, 1), stride=(*a_strides, 0)),
        )
        qb = cute.make_tensor(
            cute.recast_ptr(b.iterator, dtype=self.operand_dtype),
            cute.make_layout((self.n, self.k, 1), stride=(*b_strides, 0)),
        )
        workspace_pointer = cast("cute.Pointer", workspace.iterator)
        sa = cute.make_tensor(
            cute.recast_ptr(workspace_pointer, dtype=cutlass.Int16),
            cute.make_layout((256, self.kp // self.scale_chunk_k, self.mp // 128)),
        )
        sb = cute.make_tensor(
            cute.recast_ptr(workspace_pointer + self.sb_offset, dtype=cutlass.Int16),
            cute.make_layout((256, self.kp // self.scale_chunk_k, self.np // 128)),
        )
        self._launch(qa, qb, sa, sb, output, alpha, stream)

    @cute.jit
    def mxfp8_packed(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sa: cute.Tensor,
        sb: cute.Tensor,
        output: cute.Tensor,
        alpha: cutlass.Float32,
        stream: CUstream,
    ) -> None:
        assert self.operand_format == "mxfp8" and self.fused_scales
        a_strides = cast("tuple[int | cutlass.Int64, int | cutlass.Int64]", a.stride)
        b_strides = cast("tuple[int | cutlass.Int64, int | cutlass.Int64]", b.stride)
        qa = cute.make_tensor(
            cute.recast_ptr(a.iterator, dtype=self.operand_dtype),
            cute.make_layout((self.m, self.k, 1), stride=(*a_strides, 0)),
        )
        qb = cute.make_tensor(
            cute.recast_ptr(b.iterator, dtype=self.operand_dtype),
            cute.make_layout((self.n, self.k, 1), stride=(*b_strides, 0)),
        )
        self._launch(qa, qb, sa, sb, output, alpha, stream)

    @cute.jit
    def _launch(
        self,
        qa: cute.Tensor,
        qb: cute.Tensor,
        sa: cute.Tensor,
        sb: cute.Tensor,
        output: cute.Tensor,
        alpha: cutlass.Float32,
        stream: CUstream,
    ) -> None:
        mma = blackwell_helpers.make_blockscaled_trivial_tiled_mma(
            self.operand_dtype,
            self.operand_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.scale_dtype,
            self.scale_vector_size,
            self.group,
            (self.bm, self.bn),
        )
        cluster = cute.tiled_divide(
            cute.make_layout((self.cluster_m, 1, 1)), (mma.thr_id.shape,)
        )
        al = blackwell_helpers.make_smem_layout_a(
            mma, self.tile, self.operand_dtype, self.stages
        )
        bl = blackwell_helpers.make_smem_layout_b(
            mma, self.tile, self.operand_dtype, self.stages
        )
        sal = blockscaled_layout.make_smem_layout_sfa(
            mma, self.tile, self.scale_vector_size, self.stages
        )
        sbl = blockscaled_layout.make_smem_layout_sfb(
            mma, self.tile, self.scale_vector_size, self.stages
        )
        sa_window = cute.make_layout((256, self.bk // self.scale_chunk_k, 1))
        sb_window = cute.make_layout(
            (256, self.bk // self.scale_chunk_k, self.sf_n_chunks)
        )
        assert cute.cosize(sal) == cute.cosize(sa_window) * 2 * self.stages
        assert cute.cosize(sbl) == cute.cosize(sb_window) * 2 * self.stages
        if cutlass.const_expr(self.compact_m):
            # Valid MMA rows occupy one logical stage. The remaining rows of
            # the physical MMA view are dead output lanes and overlap the next
            # stage; cosize keeps an explicit tail allocation for its final read.
            al, load_layout = _compact_a_layouts(al, self.logical_m_per_cta, self.bk)
            a_atom, a_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(self.group),
                qa,
                load_layout,
                (self.logical_m_per_cta, self.bk, 1),
            )
        else:
            a_atom, a_tma = cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(self.group),
                qa,
                cute.slice_(al, (None, None, None, 0)),
                self.tile,
                mma,
                cluster.shape,
            )
        b_atom, b_tma = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(self.group),
            qb,
            cute.slice_(bl, (None, None, None, 0)),
            self.tile,
            mma,
            cluster.shape,
        )
        # A statically unit row mode disappears from CuTe's coordinate tensor,
        # although the TMA descriptor and its tile retain both K and row modes.
        # Restore only the coordinate domain; descriptor bounds still describe
        # the original allocation, so TMA zero-fills the extra coordinates.
        if cutlass.const_expr(self.operand_format == "mxfp8" and self.m == 1):
            a_tma = cute.make_tensor(
                (0, 0),
                cute.make_layout(
                    (2, self.k, 1),
                    stride=(
                        cute.ScaledBasis(1, mode=[1]),
                        cute.ScaledBasis(1, mode=[0]),
                        0,
                    ),
                ),
            )
        if cutlass.const_expr(self.operand_format == "mxfp8" and self.n == 1):
            b_tma = cute.make_tensor(
                (0, 0),
                cute.make_layout(
                    (2, self.k, 1),
                    stride=(
                        cute.ScaledBasis(1, mode=[1]),
                        cute.ScaledBasis(1, mode=[0]),
                        0,
                    ),
                ),
            )
        if cutlass.const_expr(self.fused_scales and self.sf_tma):
            sa_raw = cute.make_tensor(
                cute.recast_ptr(sa.iterator, dtype=cutlass.Int32),
                cute.make_layout(
                    (self.m, (self.k + 127) // 128, 1),
                    stride=(self.lhs_scale_row_stride, self.lhs_scale_column_stride, 0),
                ),
            )
            sb_raw = cute.make_tensor(
                cute.recast_ptr(sb.iterator, dtype=cutlass.Int32),
                cute.make_layout(
                    (self.n, (self.k + 127) // 128, 1),
                    stride=(self.rhs_scale_row_stride, self.rhs_scale_column_stride, 0),
                ),
            )
            sa_raw_layout = cute.make_layout(
                (128, self.bk // 128, 1), stride=(1, 128, 0)
            )
            sb_raw_layout = cute.make_layout(
                (128 * self.sf_n_chunks, self.bk // 128, 1),
                stride=(1, 128 * self.sf_n_chunks, 0),
            )
            sa_atom, sa_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                sa_raw,
                sa_raw_layout,
                sa_raw_layout.shape,
            )
            sb_atom, sb_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                sb_raw,
                sb_raw_layout,
                sb_raw_layout.shape,
            )
            if cutlass.const_expr(self.m == 1):
                sa_tma = cute.make_tensor(
                    (0, 0),
                    cute.make_layout(
                        (2, (self.k + 127) // 128, 1),
                        stride=(
                            cute.ScaledBasis(1, mode=[0]),
                            cute.ScaledBasis(1, mode=[1]),
                            0,
                        ),
                    ),
                )
            if cutlass.const_expr(self.n == 1):
                sb_tma = cute.make_tensor(
                    (0, 0),
                    cute.make_layout(
                        (2, (self.k + 127) // 128, 1),
                        stride=(
                            cute.ScaledBasis(1, mode=[0]),
                            cute.ScaledBasis(1, mode=[1]),
                            0,
                        ),
                    ),
                )
        elif cutlass.const_expr(self.fused_scales):
            sa_atom, sa_tma = a_atom, sa
            sb_atom, sb_tma = b_atom, sb
        else:
            sa_atom, sa_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(self.group),
                sa,
                sa_window,
                sa_window.shape,
            )
            sb_atom, sb_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(self.group),
                sb,
                sb_window,
                sb_window.shape,
            )
        self.tx_bytes = self.cluster_m * (
            (
                self.logical_m_per_cta * self.bk
                if self.compact_m
                else cute.size_in_bytes(self.operand_dtype, al) // self.stages
            )
            + cute.size_in_bytes(self.operand_dtype, bl) // self.stages
            + (0 if self.fused_scales else cute.cosize(sal) // self.stages)
            + (0 if self.fused_scales else cute.cosize(sbl) // self.stages)
        )
        self.kernel(
            output,
            alpha,
            mma,
            cluster,
            a_atom,
            a_tma,
            b_atom,
            b_tma,
            sa_atom,
            sa_tma,
            sb_atom,
            sb_tma,
            al,
            bl,
            sal,
            sbl,
        ).launch(
            grid=(self.clusters * self.cluster_m, 1, 1),
            block=(224 if self.fused_scales else 192, 1, 1),
            cluster=(self.cluster_m, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        output: cute.Tensor,
        alpha: cutlass.Float32,
        mma: cute.TiledMma,
        cluster: cute.Layout,
        a_atom: cute.CopyAtom,
        a_tma: cute.Tensor,
        b_atom: cute.CopyAtom,
        b_tma: cute.Tensor,
        sa_atom: cute.CopyAtom,
        sa_tma: cute.Tensor,
        sb_atom: cute.CopyAtom,
        sb_tma: cute.Tensor,
        al: cute.ComposedLayout,
        bl: cute.ComposedLayout,
        sal: cute.Layout,
        sbl: cute.Layout,
    ) -> None:
        output_pointer = cast("cute.Pointer", output.iterator)
        output_strides = cast(
            "tuple[int | cutlass.Int64, int | cutlass.Int64]", output.stride
        )
        tid = cutlass.Int32(cute.arch.thread_idx()[0])
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block = cutlass.Int32(cute.arch.block_idx()[0])
        v = block % self.cluster_m
        leader = v == 0
        first_tile = block // self.cluster_m
        tile_stride = cutlass.Int32(cute.arch.grid_dim()[0]) // self.cluster_m
        ap = cute.arch.alloc_smem(
            cutlass.Uint8,
            cute.size_in_bytes(self.operand_dtype, al),
            alignment=1024,
        )
        bp = cute.arch.alloc_smem(
            cutlass.Uint8,
            cute.size_in_bytes(self.operand_dtype, bl),
            alignment=1024,
        )
        a = cute.make_tensor(
            cute.recast_ptr(ap, al.inner, dtype=self.operand_dtype), al.outer
        )
        if cutlass.const_expr(self.compact_m):
            a_load = cute.make_tensor(
                cute.recast_ptr(ap, al.inner, dtype=self.operand_dtype),
                cute.make_layout(
                    (self.logical_m_per_cta, (128, self.bk // 128), 1, self.stages),
                    stride=(
                        128,
                        (1, self.logical_m_per_cta * 128),
                        0,
                        self.logical_m_per_cta * self.bk,
                    ),
                ),
            )
        else:
            a_load = a
        b = cute.make_tensor(
            cute.recast_ptr(bp, bl.inner, dtype=self.operand_dtype), bl.outer
        )
        sap = cute.arch.alloc_smem(self.scale_dtype, cute.cosize(sal), alignment=128)
        sbp = cute.arch.alloc_smem(self.scale_dtype, cute.cosize(sbl), alignment=128)
        sa = cute.make_tensor(sap, sal)
        sb = cute.make_tensor(sbp, sbl)
        sa_chunks = cute.make_tensor(
            cute.recast_ptr(sap, dtype=cutlass.Int16),
            cute.make_layout((256, self.bk // self.scale_chunk_k, 1, self.stages)),
        )
        sb_chunks = cute.make_tensor(
            cute.recast_ptr(sbp, dtype=cutlass.Int16),
            cute.make_layout(
                (256, self.bk // self.scale_chunk_k, self.sf_n_chunks, self.stages)
            ),
        )
        ab_barriers = cute.arch.alloc_smem(cutlass.Int64, 2 * self.stages, alignment=8)
        c_barriers = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)
        ab_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                (1 + 32 * self.cluster_m) if self.sf_inplace else 1,
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tx_bytes,
            barrier_storage=ab_barriers,
            cta_layout_vmnk=cluster,
            defer_sync=True,
        )
        if cutlass.const_expr(self.fused_scales and self.sf_tma):
            if cutlass.const_expr(self.sf_inplace):
                assert cute.cosize(sal) == 128 * (self.bk // 128) * self.stages * 4
                assert cute.cosize(sbl) == (
                    128 * self.sf_n_chunks * (self.bk // 128) * self.stages * 4
                )
                raw_sa_ptr = cute.recast_ptr(sap, dtype=cutlass.Int32)
                raw_sb_ptr = cute.recast_ptr(sbp, dtype=cutlass.Int32)
            else:
                raw_sa_ptr = cute.arch.alloc_smem(
                    cutlass.Int32, 128 * (self.bk // 128) * self.stages, alignment=128
                )
                raw_sb_ptr = cute.arch.alloc_smem(
                    cutlass.Int32,
                    128 * self.sf_n_chunks * (self.bk // 128) * self.stages,
                    alignment=128,
                )
            raw_sa_shared = cute.make_tensor(
                raw_sa_ptr,
                cute.make_layout(
                    (128, self.bk // 128, 1, self.stages),
                    stride=(1, 128, 0, 128 * (self.bk // 128)),
                ),
            )
            raw_sb_shared = cute.make_tensor(
                raw_sb_ptr,
                cute.make_layout(
                    (128 * self.sf_n_chunks, self.bk // 128, 1, self.stages),
                    stride=(
                        1,
                        128 * self.sf_n_chunks,
                        0,
                        128 * self.sf_n_chunks * (self.bk // 128),
                    ),
                ),
            )
            # These flat word views name the actual complete stage buffers.
            # Indexing a Tensor is the public DSL memory-access interface;
            # a Pointer only describes the buffer's base address.
            raw_sa_words = cute.make_tensor(
                raw_sa_ptr,
                cute.make_layout(128 * (self.bk // 128) * self.stages),
            )
            raw_sb_words = cute.make_tensor(
                raw_sb_ptr,
                cute.make_layout(
                    128 * self.sf_n_chunks * (self.bk // 128) * self.stages
                ),
            )
            if cutlass.const_expr(self.sf_inplace):
                raw_sf_pipe = None
                raw_sf_barriers = cute.arch.alloc_smem(
                    cutlass.Int64, self.stages, alignment=8
                )
                raw_sf_full = pipeline.MbarrierArray(
                    barrier_storage=raw_sf_barriers,
                    num_stages=self.stages,
                    agent=(
                        pipeline.PipelineOp.TmaLoad,
                        pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                    ),
                    tx_count=(128 + 128 * self.sf_n_chunks) * (self.bk // 128) * 4,
                )
            else:
                raw_sf_full = None
                raw_sf_barriers = cute.arch.alloc_smem(
                    cutlass.Int64, 2 * self.stages, alignment=8
                )
                raw_sf_pipe = pipeline.PipelineTmaAsync.create(
                    num_stages=self.stages,
                    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
                    tx_count=(128 + 128 * self.sf_n_chunks) * (self.bk // 128) * 4,
                    barrier_storage=raw_sf_barriers,
                    cta_layout_vmnk=cluster,
                    tidx=tid,
                    enable_multicast_signaling=True,
                    defer_sync=True,
                )
        else:
            # Disabled transfer modes have no buffers or pipeline. Keep this
            # absence explicit, then refine it inside the corresponding mode.
            raw_sa_shared, raw_sb_shared = None, None
            raw_sa_words, raw_sb_words = None, None
            raw_sf_full, raw_sf_pipe = None, None
        if cutlass.const_expr(self.fused_scales):
            if cutlass.const_expr(self.sf_inplace):
                sf_target = None
                if cutlass.const_expr(self.cluster_m == 2):
                    sf_target = pipeline.PipelineAsyncUmma._compute_leading_cta_rank(
                        self.cluster_m
                    )
                sf_pipe = pipeline.PipelineAsyncUmma(
                    ab_pipe.sync_object_full.recast_to_new_op_type(
                        pipeline.PipelineOp.AsyncThread
                    ),
                    ab_pipe.sync_object_empty,
                    self.stages,
                    sf_target,
                    ab_pipe.consumer_mask,
                    self.group,
                )
            else:
                sf_barriers = cute.arch.alloc_smem(
                    cutlass.Int64, 2 * self.stages, alignment=8
                )
                sf_pipe = pipeline.PipelineAsyncUmma.create(
                    num_stages=self.stages,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, 32 * self.cluster_m
                    ),
                    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                    barrier_storage=sf_barriers,
                    cta_layout_vmnk=cluster,
                    defer_sync=True,
                )
        else:
            sf_pipe = None
        c_pipe = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 128 * self.cluster_m
            ),
            barrier_storage=c_barriers,
            cta_layout_vmnk=cluster,
            defer_sync=True,
        )
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster, is_relaxed=True)
        allocator = utils.TmemAllocator(
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=1, num_threads=224 if self.fused_scales else 192
            ),
            is_two_cta=self.cluster_m == 2,
        )
        c_layout = mma.make_fragment_C(mma.partition_shape_C((self.bm, self.bn))).layout
        sa_layout = blockscaled_layout.make_tmem_layout_sfa(
            mma,
            self.tile,
            self.scale_vector_size,
            cute.slice_(sal, (None, None, None, 0)),
        )
        sb_layout = blockscaled_layout.make_tmem_layout_sfb(
            mma,
            self.tile,
            self.scale_vector_size,
            cute.slice_(sbl, (None, None, None, 0)),
        )
        # Layout footprints are TMEM columns when inspected through a fake
        # pointer. Real pointers are retrieved only after allocation completes.
        c_columns = self.bn
        sa_columns = self.bk // self.scale_vector_size
        sb_columns = self.sf_n_chunks * (self.bk // self.scale_vector_size)
        columns = max(32, 1 << (c_columns + sa_columns + sb_columns - 1).bit_length())
        assert columns <= 512
        allocator.allocate(columns)
        allocator.wait_for_alloc()
        tmem = allocator.retrieve_ptr(cutlass.Float32)
        acc = cute.make_tensor(tmem, c_layout)
        tsa = cute.make_tensor(
            cute.recast_ptr(tmem + c_columns, dtype=self.scale_dtype), sa_layout
        )
        tsb = cute.make_tensor(
            cute.recast_ptr(tmem + c_columns + sa_columns, dtype=self.scale_dtype),
            sb_layout,
        )
        assert tcgen05.find_tmem_tensor_col_offset(acc) == c_columns
        assert tcgen05.find_tmem_tensor_col_offset(tsa) == sa_columns, (
            f"SFA columns {tcgen05.find_tmem_tensor_col_offset(tsa)} != {sa_columns}"
        )
        assert tcgen05.find_tmem_tensor_col_offset(tsb) == sb_columns, (
            f"SFB columns {tcgen05.find_tmem_tensor_col_offset(tsb)} != {sb_columns}"
        )
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster)
        if warp == 5:
            cpasync.prefetch_descriptor(a_atom)
            cpasync.prefetch_descriptor(b_atom)
            if cutlass.const_expr(not self.fused_scales or self.sf_tma):
                cpasync.prefetch_descriptor(sa_atom)
                cpasync.prefetch_descriptor(sb_atom)
            state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.stages
            )
            thread_mma = mma.get_slice(v)
            for tile_index in cutlass.range(
                first_tile, self.tile_count, tile_stride, unroll=1
            ):
                mi = tile_index // self.n_tiles
                ni = tile_index % self.n_tiles
                if cutlass.const_expr(self.compact_m):
                    ga = cute.local_tile(
                        a_tma,
                        (self.logical_m_per_cta, self.bk, 1),
                        (mi * self.cluster_m + v, None, 0),
                    )
                else:
                    ga = thread_mma.partition_A(
                        cute.local_tile(
                            cute.domain_offset(
                                (
                                    mi * self.logical_m_per_cta * self.cluster_m
                                    + v * (self.logical_m_per_cta - 128),
                                    0,
                                ),
                                a_tma[None, None, 0],
                            ),
                            (self.bm, self.bk),
                            (0, None),
                        )
                    )
                gb = thread_mma.partition_B(
                    cute.local_tile(
                        b_tma[None, None, 0], (self.bn, self.bk), (ni, None)
                    )
                )
                if cutlass.const_expr(self.sf_tma):
                    m_start = (
                        mi * self.logical_m_per_cta * self.cluster_m
                        + v * self.logical_m_per_cta
                    )
                    n_start = (
                        (ni // 2 * 128)
                        if cutlass.const_expr(self.bn == 64)
                        else ni * self.bn
                    )
                    raw_gsa = cute.local_tile(
                        cute.domain_offset((m_start, 0, 0), sa_tma),
                        (128, self.bk // 128, 1),
                        (0, None, 0),
                    )
                    raw_gsb = cute.local_tile(
                        cute.domain_offset((n_start, 0, 0), sb_tma),
                        (128 * self.sf_n_chunks, self.bk // 128, 1),
                        (0, None, 0),
                    )
                else:
                    raw_gsa, raw_gsb = None, None
                if cutlass.const_expr(not self.fused_scales):
                    gsa = cute.local_tile(
                        sa_tma,
                        (256, self.bk // self.scale_chunk_k, 1),
                        (0, None, mi * self.cluster_m + v),
                    )
                    gsb = cute.local_tile(
                        sb_tma,
                        (256, self.bk // self.scale_chunk_k, self.sf_n_chunks),
                        (0, None, ni // 2 if cutlass.const_expr(self.bn == 64) else ni),
                    )
                else:
                    gsa, gsb = None, None
                da, ga = cpasync.tma_partition(
                    a_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(a_load, 0, 3),
                    cute.group_modes(ga, 0, 3),
                )
                db, gb = cpasync.tma_partition(
                    b_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(b, 0, 3),
                    cute.group_modes(gb, 0, 3),
                )
                if cutlass.const_expr(not self.fused_scales):
                    assert gsa is not None and gsb is not None
                    dsa, gsa = cpasync.tma_partition(
                        sa_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sa_chunks, 0, 3),
                        cute.group_modes(gsa, 0, 3),
                    )
                    dsb, gsb = cpasync.tma_partition(
                        sb_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sb_chunks, 0, 3),
                        cute.group_modes(gsb, 0, 3),
                    )
                else:
                    dsa, dsb = None, None
                if cutlass.const_expr(self.sf_tma):
                    raw_dsa, raw_gsa = cpasync.tma_partition(
                        sa_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_sa_shared, 0, 3),
                        cute.group_modes(raw_gsa, 0, 3),
                    )
                    raw_dsb, raw_gsb = cpasync.tma_partition(
                        sb_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_sb_shared, 0, 3),
                        cute.group_modes(raw_gsb, 0, 3),
                    )
                else:
                    raw_dsa, raw_dsb = None, None
                for ki in cutlass.range(self.kp // self.bk, unroll=1):
                    if cutlass.const_expr(self.sf_inplace):
                        # The same empty signal protects operands and scales.
                        # Raw SF TMA may overwrite a stage only after UTCCP/MMA
                        # has finished consuming its prior contents.
                        ab_pipe.producer_acquire(state)
                    if cutlass.const_expr(self.sf_tma):
                        assert raw_gsa is not None and raw_gsb is not None
                        assert raw_dsa is not None and raw_dsb is not None
                        if cutlass.const_expr(self.sf_inplace):
                            assert raw_sf_full is not None
                            raw_sf_full.arrive_and_expect_tx(
                                state.index, raw_sf_full.tx_count
                            )
                            raw_barrier = raw_sf_full.get_barrier(state.index)
                        else:
                            assert raw_sf_pipe is not None
                            raw_sf_pipe.producer_acquire(state)
                            raw_barrier = raw_sf_pipe.producer_get_barrier(state)
                        cute.copy(
                            sa_atom,
                            raw_gsa[None, ki],
                            raw_dsa[None, state.index],
                            tma_bar_ptr=raw_barrier,
                        )
                        cute.copy(
                            sb_atom,
                            raw_gsb[None, ki],
                            raw_dsb[None, state.index],
                            tma_bar_ptr=raw_barrier,
                        )
                        if cutlass.const_expr(not self.sf_inplace):
                            assert raw_sf_pipe is not None
                            raw_sf_pipe.producer_commit(state)
                    if cutlass.const_expr(not self.sf_inplace):
                        ab_pipe.producer_acquire(state)
                    barrier = ab_pipe.producer_get_barrier(state)
                    cute.copy(
                        a_atom, ga[None, ki], da[None, state.index], tma_bar_ptr=barrier
                    )
                    cute.copy(
                        b_atom, gb[None, ki], db[None, state.index], tma_bar_ptr=barrier
                    )
                    if cutlass.const_expr(not self.fused_scales):
                        assert gsa is not None and gsb is not None
                        assert dsa is not None and dsb is not None
                        cute.copy(
                            sa_atom,
                            gsa[None, ki],
                            dsa[None, state.index],
                            tma_bar_ptr=barrier,
                        )
                        cute.copy(
                            sb_atom,
                            gsb[None, ki],
                            dsb[None, state.index],
                            tma_bar_ptr=barrier,
                        )
                    ab_pipe.producer_commit(state)
                    state.advance()
            ab_pipe.producer_tail(state.clone())
            if cutlass.const_expr(self.sf_tma and not self.sf_inplace):
                assert raw_sf_pipe is not None
                raw_sf_pipe.producer_tail(state.clone())
        if warp == 4:
            if leader:
                ra = mma.make_fragment_A(a)
                rb = mma.make_fragment_B(b)
                state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.stages
                )
                cstate = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )
                for _tile_index in cutlass.range(
                    first_tile, self.tile_count, tile_stride, unroll=1
                ):
                    mma_sfb = tsb
                    if cutlass.const_expr(self.bn == 64):
                        # Two N64 tiles share one prepared 128-row SFB chunk.
                        # The upper 64 rows start two Float32 TMEM columns past
                        # the unshifted scale base, as required by tcgen05.
                        offset = cutlass.Int32((_tile_index % self.n_tiles % 2) * 2)
                        mma_sfb = cute.make_tensor(
                            cute.recast_ptr(
                                tmem + c_columns + sa_columns + offset,
                                dtype=self.scale_dtype,
                            ),
                            sb_layout,
                        )
                    c_pipe.producer_acquire(cstate)
                    mma.set(tcgen05.Field.ACCUMULATE, False)
                    for _ki in cutlass.range(self.kp // self.bk, unroll=1):
                        ab_pipe.consumer_wait(state)
                        if cutlass.const_expr(
                            self.fused_scales and not self.sf_inplace
                        ):
                            assert sf_pipe is not None
                            sf_pipe.consumer_wait(state)
                        _scale_copy(sa, tsa, state.index, self.group)
                        _scale_copy(sb, tsb, state.index, self.group)
                        for kk in cutlass.range_constexpr(self.bk // self.mma_k):
                            mma.set(tcgen05.Field.SFA, tsa[None, None, kk].iterator)
                            mma.set(tcgen05.Field.SFB, mma_sfb[None, None, kk].iterator)
                            cute.gemm(
                                mma,
                                acc,
                                ra[None, None, kk, state.index],
                                rb[None, None, kk, state.index],
                                acc,
                            )
                            mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipe.consumer_release(state)
                        if cutlass.const_expr(
                            self.fused_scales and not self.sf_inplace
                        ):
                            assert sf_pipe is not None
                            sf_pipe.consumer_release(state)
                        state.advance()
                    c_pipe.producer_commit(cstate)
                    cstate.advance()
                c_pipe.producer_tail(cstate)
        if cutlass.const_expr(self.fused_scales):
            assert sf_pipe is not None
            if warp == 6:
                lane = tid % 32
                if cutlass.const_expr(not self.sf_tma):
                    raw_sa = cute.recast_ptr(sa_tma.iterator, dtype=cutlass.Int32)
                    raw_sb = cute.recast_ptr(sb_tma.iterator, dtype=cutlass.Int32)
                else:
                    raw_sa, raw_sb = None, None
                dst_sa = cute.recast_ptr(sap, dtype=cutlass.Int32)
                dst_sb = cute.recast_ptr(sbp, dtype=cutlass.Int32)
                state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.stages
                )
                if cutlass.const_expr(self.sf_tma):
                    raw_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.stages
                    )
                else:
                    raw_state = None
                for tile_index in cutlass.range(
                    first_tile, self.tile_count, tile_stride, unroll=1
                ):
                    mi = tile_index // self.n_tiles
                    ni = tile_index % self.n_tiles
                    m_start = (
                        mi * self.logical_m_per_cta * self.cluster_m
                        + v * self.logical_m_per_cta
                    )
                    n_start = (
                        (ni // 2 * 128)
                        if cutlass.const_expr(self.bn == 64)
                        else ni * self.bn
                    )
                    for ki in cutlass.range(self.kp // self.bk, unroll=1):
                        sf_pipe.producer_acquire(state)
                        if cutlass.const_expr(self.sf_tma):
                            assert raw_state is not None
                            if cutlass.const_expr(self.sf_inplace):
                                assert raw_sf_full is not None
                                raw_sf_full.wait(raw_state.index, raw_state.phase)
                            else:
                                assert raw_sf_pipe is not None
                                raw_sf_pipe.consumer_wait(raw_state)
                        if cutlass.const_expr(self.sf_inplace):
                            assert raw_sa_words is not None and raw_sb_words is not None
                            _transpose_sf_stage_inplace(
                                raw_sa_words,
                                raw_sb_words,
                                state.index,
                                ki,
                                m_start,
                                n_start,
                                self.m,
                                self.n,
                                self.k,
                                self.bk,
                                self.sf_n_chunks,
                                self.logical_m_per_cta,
                            )
                        else:
                            for chunk in cutlass.range_constexpr(self.bk // 128):
                                word_k = ki * (self.bk // 128) + chunk
                                for segment in cutlass.range_constexpr(4):
                                    row = lane + segment * 32
                                    gm = m_start + row
                                    value = cutlass.Int32(0)
                                    if (
                                        gm < self.m
                                        and row < self.logical_m_per_cta
                                        and word_k < (self.k + 127) // 128
                                    ):
                                        if cutlass.const_expr(self.sf_tma):
                                            assert raw_sa_words is not None
                                            value = raw_sa_words[
                                                state.index * (128 * (self.bk // 128))
                                                + chunk * 128
                                                + row
                                            ]
                                        else:
                                            assert raw_sa is not None
                                            value = raw_sa[
                                                cutlass.Int64(gm)
                                                * self.lhs_scale_row_stride
                                                + cutlass.Int64(word_k)
                                                * self.lhs_scale_column_stride
                                            ]
                                    if cutlass.const_expr(self.k % 128 != 0):
                                        if word_k == self.k // 128:
                                            value = value & (
                                                (1 << ((self.k % 128) // 32 * 8)) - 1
                                            )
                                    dst_sa[
                                        state.index * (128 * (self.bk // 128))
                                        + chunk * 128
                                        + lane * 4
                                        + segment
                                    ] = value
                                for n_chunk in cutlass.range_constexpr(
                                    self.sf_n_chunks
                                ):
                                    for segment in cutlass.range_constexpr(4):
                                        row = lane + segment * 32
                                        gn = n_start + n_chunk * 128 + row
                                        value = cutlass.Int32(0)
                                        if (
                                            gn < self.n
                                            and word_k < (self.k + 127) // 128
                                        ):
                                            if cutlass.const_expr(self.sf_tma):
                                                assert raw_sb_words is not None
                                                value = raw_sb_words[
                                                    state.index
                                                    * (
                                                        128
                                                        * self.sf_n_chunks
                                                        * (self.bk // 128)
                                                    )
                                                    + chunk * (128 * self.sf_n_chunks)
                                                    + n_chunk * 128
                                                    + row
                                                ]
                                            else:
                                                assert raw_sb is not None
                                                value = raw_sb[
                                                    cutlass.Int64(gn)
                                                    * self.rhs_scale_row_stride
                                                    + cutlass.Int64(word_k)
                                                    * self.rhs_scale_column_stride
                                                ]
                                        if cutlass.const_expr(self.k % 128 != 0):
                                            if word_k == self.k // 128:
                                                value = value & (
                                                    (1 << ((self.k % 128) // 32 * 8))
                                                    - 1
                                                )
                                        dst_sb[
                                            state.index
                                            * (
                                                128
                                                * (self.bk // 128)
                                                * self.sf_n_chunks
                                            )
                                            + n_chunk * (128 * (self.bk // 128))
                                            + chunk * 128
                                            + lane * 4
                                            + segment
                                        ] = value
                        cute.arch.fence_view_async_shared()
                        if cutlass.const_expr(self.sf_tma and not self.sf_inplace):
                            assert raw_sf_pipe is not None and raw_state is not None
                            raw_sf_pipe.consumer_release(raw_state)
                        sf_pipe.producer_commit(state)
                        state.advance()
                        if cutlass.const_expr(self.sf_tma):
                            assert raw_state is not None
                            raw_state.advance()
                sf_pipe.producer_tail(state.clone())
        if warp < 4:
            epi_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
            cp = cute.arch.alloc_smem(
                output.element_type, 128 * self.epi_n, alignment=128
            )
            cl = cute.make_composed_layout(
                cute.make_swizzle(3, 3, 3),
                0,
                cute.make_layout((128, self.epi_n), stride=(self.epi_n, 1)),
            )
            shared_c = cute.make_tensor(cp, cl)
            c_tiles = cute.local_tile(
                transform_partitioned_tensor_layout(acc), (128, self.epi_n), (0, None)
            )
            atom = blackwell_helpers.get_tmem_load_op(
                (128, self.bn, self.bk),
                LayoutEnum.ROW_MAJOR,
                output.element_type,
                cutlass.Float32,
                (128, self.epi_n),
                self.cluster_m == 2,
            )
            copy = tcgen05.make_tmem_copy(atom, c_tiles[None, None, 0])
            thread = copy.get_slice(tid)
            destination = thread.partition_D(shared_c)
            regs = cute.make_rmem_tensor(destination.shape, cutlass.Float32)
            narrowed = cute.make_rmem_tensor(destination.shape, output.element_type)
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), output.element_type, num_bits_per_copy=128
            )
            cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            for tile_index in cutlass.range(
                first_tile, self.tile_count, tile_stride, unroll=1
            ):
                m_start = (
                    tile_index // self.n_tiles * self.logical_m_per_cta * self.cluster_m
                    + v * self.logical_m_per_cta
                )
                n_start = tile_index % self.n_tiles * self.bn
                c_pipe.consumer_wait(cstate)
                for ni in cutlass.range(
                    (self.bn + self.epi_n - 1) // self.epi_n, unroll=1
                ):
                    cute.copy(copy, thread.partition_S(c_tiles[None, None, ni]), regs)
                    rounded = regs.load() * alpha
                    for round_index in cutlass.range_constexpr(len(self.round_dtypes)):
                        rounded = rounded.to(self.round_dtypes[round_index])
                    narrowed.store(rounded.to(output.element_type))
                    cute.autovec_copy(narrowed, destination)
                    epi_barrier.arrive_and_wait()
                    for vi in cutlass.range_constexpr(self.epi_n // 8):
                        flat = (tid + vi * 128) * 8
                        row = flat // self.epi_n
                        col = flat % self.epi_n
                        gm = m_start + row
                        gn = n_start + ni * self.epi_n + col
                        if (
                            gm < self.m
                            and gn < self.n
                            and row < self.logical_m_per_cta
                            and ni * self.epi_n + col < self.bn
                        ):
                            address = (
                                cutlass.Int64(gm) * output_strides[0]
                                + cutlass.Int64(gn) * output_strides[1]
                            )
                            if (
                                cutlass.const_expr(output_pointer.alignment >= 16)
                                and gn + 7 < self.n
                                and ni * self.epi_n + col + 7 < self.bn
                                and address % 8 == 0
                                and output_strides[1] == 1
                            ):
                                src = cute.make_tensor(
                                    shared_c.iterator
                                    + cute.assume(
                                        cute.crd2idx((row, col), shared_c.layout),
                                        divby=8,
                                    ),
                                    cute.make_layout(8),
                                )
                                dst = cute.make_tensor(
                                    output_pointer + cute.assume(address, divby=8),
                                    cute.make_layout(8),
                                )
                                cute.copy(store_atom, src, dst)
                            else:
                                for tail in cutlass.range_constexpr(8):
                                    if (
                                        gn + tail < self.n
                                        and ni * self.epi_n + col + tail < self.bn
                                    ):
                                        output[
                                            cutlass.Int64(gm), cutlass.Int64(gn) + tail
                                        ] = shared_c[row, col + tail]
                    epi_barrier.arrive_and_wait()
                cute.arch.fence_view_async_tmem_load()
                c_pipe.consumer_release(cstate)
                cstate.advance()
        cute.arch.sync_threads()
        if cutlass.const_expr(self.cluster_m == 2):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        allocator.relinquish_alloc_permit()
        allocator.free(tmem)
