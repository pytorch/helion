"""SM100 packed E2M1/UE4M3 pipeline for proved block-scaled contractions.

Prepared operands contain only byte permutations, sign normalization and zero
padding. The native consumer keeps them packed through TMA and shared memory.
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
    ) -> None:
        assert m > 0 and n > 0 and k > 0 and k % 16 == 0
        assert bn in (128, 256) and bk in (64, 128, 256)
        assert stages in (2, 3, 4, 5) and cluster_m in (1, 2)
        self.m, self.n, self.k = m, n, k
        self.mp, self.np, self.kp = block_scaled_workspace_shape(m, n, k, bn, bk)
        self.bm, self.bn, self.bk = 128 * cluster_m, bn, bk
        self.stages = stages
        self.cluster_m = cluster_m
        self.round_dtypes = round_dtypes
        self.group = tcgen05.CtaGroup.ONE if cluster_m == 1 else tcgen05.CtaGroup.TWO
        self.tile = (self.bm, bn, bk)
        self.qa_bytes = self.mp * self.kp // 2
        self.qb_bytes = self.np * self.kp // 2
        self.sa_bytes = self.mp * self.kp // 16
        self.sb_bytes = self.np * self.kp // 16
        self.qb_offset = self.qa_bytes
        self.sa_offset = self.qb_offset + self.qb_bytes
        self.sb_offset = self.sa_offset + self.sa_bytes
        self.workspace_bytes = self.sb_offset + self.sb_bytes
        self.m_tiles = (m + self.bm - 1) // self.bm
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
        mma = blackwell_helpers.make_blockscaled_trivial_tiled_mma(
            cutlass.Float4E2M1FN,
            cutlass.Float4E2M1FN,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float8E4M3FN,
            16,
            self.group,
            (self.bm, self.bn),
        )
        cluster = cute.tiled_divide(
            cute.make_layout((self.cluster_m, 1, 1)), (mma.thr_id.shape,)
        )
        al = blackwell_helpers.make_smem_layout_a(
            mma, self.tile, cutlass.Float4E2M1FN, self.stages
        )
        bl = blackwell_helpers.make_smem_layout_b(
            mma, self.tile, cutlass.Float4E2M1FN, self.stages
        )
        sal = blockscaled_layout.make_smem_layout_sfa(mma, self.tile, 16, self.stages)
        sbl = blockscaled_layout.make_smem_layout_sfb(mma, self.tile, 16, self.stages)
        sa_window = cute.make_layout((256, self.bk // 64, 1))
        sb_window = cute.make_layout((256, self.bk // 64, self.bn // 128))
        assert cute.cosize(sal) == cute.cosize(sa_window) * 2 * self.stages
        assert cute.cosize(sbl) == cute.cosize(sb_window) * 2 * self.stages
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
            cute.size_in_bytes(cutlass.Float4E2M1FN, al) // self.stages
            + cute.size_in_bytes(cutlass.Float4E2M1FN, bl) // self.stages
            + cute.cosize(sal) // self.stages
            + cute.cosize(sbl) // self.stages
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
            block=(192, 1, 1),
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
        ap = cute.arch.alloc_smem(cutlass.Uint8, cute.cosize(al) // 2, alignment=1024)
        bp = cute.arch.alloc_smem(cutlass.Uint8, cute.cosize(bl) // 2, alignment=1024)
        a = cute.make_tensor(
            cute.recast_ptr(ap, al.inner, dtype=cutlass.Float4E2M1FN), al.outer
        )
        b = cute.make_tensor(
            cute.recast_ptr(bp, bl.inner, dtype=cutlass.Float4E2M1FN), bl.outer
        )
        sap = cute.arch.alloc_smem(
            cutlass.Float8E4M3FN, cute.cosize(sal), alignment=128
        )
        sbp = cute.arch.alloc_smem(
            cutlass.Float8E4M3FN, cute.cosize(sbl), alignment=128
        )
        sa = cute.make_tensor(sap, sal)
        sb = cute.make_tensor(sbp, sbl)
        sa_chunks = cute.make_tensor(
            cute.recast_ptr(sap, dtype=cutlass.Int16),
            cute.make_layout((256, self.bk // 64, 1, self.stages)),
        )
        sb_chunks = cute.make_tensor(
            cute.recast_ptr(sbp, dtype=cutlass.Int16),
            cute.make_layout((256, self.bk // 64, self.bn // 128, self.stages)),
        )
        ab_barriers = cute.arch.alloc_smem(cutlass.Int64, 2 * self.stages, alignment=8)
        c_barriers = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)
        ab_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tx_bytes,
            barrier_storage=ab_barriers,
            cta_layout_vmnk=cluster,
            defer_sync=True,
        )
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
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=192),
            is_two_cta=self.cluster_m == 2,
        )
        c_layout = mma.make_fragment_C(mma.partition_shape_C((self.bm, self.bn))).layout
        sa_layout = blockscaled_layout.make_tmem_layout_sfa(
            mma, self.tile, 16, cute.slice_(sal, (None, None, None, 0))
        )
        sb_layout = blockscaled_layout.make_tmem_layout_sfb(
            mma, self.tile, 16, cute.slice_(sbl, (None, None, None, 0))
        )
        # Layout footprints are TMEM columns when inspected through a fake
        # pointer. Real pointers are retrieved only after allocation completes.
        c_columns = self.bn
        sa_columns = self.bk // 16
        sb_columns = (self.bn // 128) * (self.bk // 16)
        columns = 1 << (c_columns + sa_columns + sb_columns - 1).bit_length()
        assert columns <= 512
        allocator.allocate(columns)
        allocator.wait_for_alloc()
        tmem = allocator.retrieve_ptr(cutlass.Float32)
        acc = cute.make_tensor(tmem, c_layout)
        tsa = cute.make_tensor(
            cute.recast_ptr(tmem + c_columns, dtype=cutlass.Float8E4M3FN), sa_layout
        )
        tsb = cute.make_tensor(
            cute.recast_ptr(tmem + c_columns + sa_columns, dtype=cutlass.Float8E4M3FN),
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
                ga = thread_mma.partition_A(
                    cute.local_tile(
                        a_tma[None, None, 0], (self.bm, self.bk), (mi, None)
                    )
                )
                gb = thread_mma.partition_B(
                    cute.local_tile(
                        b_tma[None, None, 0], (self.bn, self.bk), (ni, None)
                    )
                )
                gsa = cute.local_tile(
                    sa_tma, (256, self.bk // 64, 1), (0, None, mi * self.cluster_m + v)
                )
                gsb = cute.local_tile(
                    sb_tma, (256, self.bk // 64, self.bn // 128), (0, None, ni)
                )
                da, ga = cpasync.tma_partition(
                    a_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(a, 0, 3),
                    cute.group_modes(ga, 0, 3),
                )
                db, gb = cpasync.tma_partition(
                    b_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(b, 0, 3),
                    cute.group_modes(gb, 0, 3),
                )
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
                for ki in cutlass.range(self.kp // self.bk, unroll=1):
                    ab_pipe.producer_acquire(state)
                    barrier = ab_pipe.producer_get_barrier(state)
                    cute.copy(
                        a_atom, ga[None, ki], da[None, state.index], tma_bar_ptr=barrier
                    )
                    cute.copy(
                        b_atom, gb[None, ki], db[None, state.index], tma_bar_ptr=barrier
                    )
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
            ab_pipe.producer_tail(state)
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
                    c_pipe.producer_acquire(cstate)
                    mma.set(tcgen05.Field.ACCUMULATE, False)
                    for _ki in cutlass.range(self.kp // self.bk, unroll=1):
                        ab_pipe.consumer_wait(state)
                        _scale_copy(sa, tsa, state.index, self.group)
                        _scale_copy(sb, tsb, state.index, self.group)
                        for kk in cutlass.range_constexpr(self.bk // 64):
                            mma.set(tcgen05.Field.SFA, tsa[None, None, kk].iterator)
                            mma.set(tcgen05.Field.SFB, tsb[None, None, kk].iterator)
                            cute.gemm(
                                mma,
                                acc,
                                ra[None, None, kk, state.index],
                                rb[None, None, kk, state.index],
                                acc,
                            )
                            mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipe.consumer_release(state)
                        state.advance()
                    c_pipe.producer_commit(cstate)
                    cstate.advance()
                c_pipe.producer_tail(cstate)
        if warp < 4:
            epi_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
            cp = cute.arch.alloc_smem(output.element_type, 128 * 64, alignment=128)
            cl = cute.make_composed_layout(
                cute.make_swizzle(3, 3, 3),
                0,
                cute.make_layout((128, 64), stride=(64, 1)),
            )
            shared_c = cute.make_tensor(cp, cl)
            c_tiles = cute.local_tile(
                transform_partitioned_tensor_layout(acc), (128, 64), (0, None)
            )
            atom = blackwell_helpers.get_tmem_load_op(
                (128, self.bn, self.bk),
                LayoutEnum.ROW_MAJOR,
                output.element_type,
                cutlass.Float32,
                (128, 64),
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
                m_start = tile_index // self.n_tiles * self.bm + v * 128
                n_start = tile_index % self.n_tiles * self.bn
                c_pipe.consumer_wait(cstate)
                for ni in cutlass.range(self.bn // 64, unroll=1):
                    cute.copy(copy, thread.partition_S(c_tiles[None, None, ni]), regs)
                    rounded = regs.load() * alpha
                    for round_index in cutlass.range_constexpr(len(self.round_dtypes)):
                        rounded = rounded.to(self.round_dtypes[round_index])
                    narrowed.store(rounded.to(output.element_type))
                    cute.autovec_copy(narrowed, destination)
                    epi_barrier.arrive_and_wait()
                    for vi in cutlass.range_constexpr(8):
                        flat = (tid + vi * 128) * 8
                        row = flat // 64
                        col = flat % 64
                        gm = m_start + row
                        gn = n_start + ni * 64 + col
                        if gm < self.m and gn < self.n:
                            address = (
                                cutlass.Int64(gm) * output_strides[0]
                                + cutlass.Int64(gn) * output_strides[1]
                            )
                            if (
                                cutlass.const_expr(output_pointer.alignment >= 16)
                                and gn + 7 < self.n
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
                                    if gn + tail < self.n:
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
