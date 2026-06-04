# Fused rowwise-scale FP8 scaled_mm on top of the CUTLASS CuTe DSL persistent GEMM.
#
#   out[m,n] = scale_a[m] * scale_b[n] * sum_k(a_fp8[m,k] * b_fp8[k,n])
#
# scale_a [M] f32, scale_b [N] f32, a/b fp8 e4m3, out bf16.
#
# The scale is fused INTO the epilogue: after the accumulator is loaded from TMEM
# to registers (still f32), we multiply each register element by
# scale_a[m]*scale_b[n] BEFORE casting to bf16. The per-register (m,n) global
# coords are obtained from an identity coordinate tensor partitioned IDENTICALLY
# to the accumulator fragment (the EVT Row/Col-broadcast trick, done by hand).
from typing import Any, Optional, Tuple, Type, Union
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Int32
from cutlass.cute.nvgpu import cpasync, tcgen05

import cute_dense_gemm_persistent as G

# reuse host-side helpers
prepare_tensors = G.prepare_tensors


# ---------------------------------------------------------------------------
# Custom TMA-store epilogue WITH fused rowwise scale.
# Mirrors cutlass.utils.gemm.sm100.epilogue_tma_store, plus scale multiply.
# ---------------------------------------------------------------------------
@cute.jit
def epilogue_tma_store_scaled(
    gemm_kernel: Any,
    epi_tidx: Int32,
    warp_idx: Int32,
    tma_atom_c: cute.CopyAtom,
    tCtAcc_base: cute.Tensor,
    sC: cute.Tensor,
    tCgC_base: cute.Tensor,
    tCgSA_base: cute.Tensor,  # scale_a broadcast (M,N,L) stride(1,0,0), part like tCgC
    tCgSB_base: cute.Tensor,  # scale_b broadcast (M,N,L) stride(0,1,0), part like tCgC
    epi_tile: cute.Tile,
    num_tiles_executed: Int32,
    mma_tile_coord_mnl: Tuple[Int32, Int32, Int32],
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    c_pipeline: pipeline.PipelineTmaStore,
) -> pipeline.PipelineState:
    from cutlass.utils.gemm.sm100 import (
        transform_partitioned_tensor_layout,
        epilogue_tmem_copy_and_partition,
        epilogue_smem_copy_and_partition,
    )

    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    tCgSA = transform_partitioned_tensor_layout(tCgSA_base)
    tCgSB = transform_partitioned_tensor_layout(tCgSB_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )

    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
        gemm_kernel, tiled_copy_t2r, tTR_rC, epi_tidx, sC
    )

    # Partition the scale broadcast tensors the SAME way as the accumulator's
    # global C destination, so a vectorized copy fills register fragments
    # aligned element-for-element with tTR_rAcc (EVT Col/Row broadcast).
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    sA_epi = cute.flat_divide(tCgSA, epi_tile)
    sB_epi = cute.flat_divide(tCgSB, epi_tile)
    tTR_gSA_base = thr_copy_t2r.partition_D(sA_epi)
    tTR_gSB_base = thr_copy_t2r.partition_D(sB_epi)

    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )

    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=32 * len(gemm_kernel.epilogue_warp_id),
    )

    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    tTR_gSA = tTR_gSA_base[(None, None, None, None, None, *mma_tile_coord_mnl)]
    tTR_gSB = tTR_gSB_base[(None, None, None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

    tTR_gSA = cute.group_modes(tTR_gSA, 3, cute.rank(tTR_gSA))
    tTR_gSB = cute.group_modes(tTR_gSB, 3, cute.rank(tTR_gSB))

    # Issue the scale_b gmem load BEFORE waiting on the accumulator, so its
    # memory latency overlaps with the MMA wait. scale_a is uniform over a
    # thread's N-fragment (broadcast over N) -> read as a SINGLE scalar.
    if cutlass.const_expr(gemm_kernel.fuse_scale):
        tTR_rSB_full = cute.make_rmem_tensor(tTR_gSB.shape, gemm_kernel.acc_dtype)
        cute.autovec_copy(tTR_gSB, tTR_rSB_full)

    acc_pipeline.consumer_wait(acc_consumer_state)

    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    num_prev_subtiles = num_tiles_executed * subtile_cnt

    for subtile_idx in range(subtile_cnt):
        # Load accumulator (f32) from TMEM to registers
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

        # Apply rowwise scale on the f32 accumulator BEFORE the bf16 cast.
        if cutlass.const_expr(gemm_kernel.fuse_scale):
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            sa = tTR_gSA[(0, 0, 0, subtile_idx)]
            sb_vec = tiled_copy_r2s.retile(
                tTR_rSB_full[(None, None, None, subtile_idx)]
            ).load()
            acc_vec = acc_vec * sa * sb_vec
        else:
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
        acc_vec = acc_vec.to(gemm_kernel.c_dtype)
        tRS_rC.store(acc_vec)

        c_buffer = (num_prev_subtiles + subtile_idx) % gemm_kernel.num_c_stage
        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
        cute.arch.fence_proxy("async.shared", space="cta")
        epilog_sync_barrier.arrive_and_wait()

        if warp_idx == gemm_kernel.epilogue_warp_id[0]:
            cute.copy(
                tma_atom_c,
                bSG_sC[(None, c_buffer)],
                bSG_gC[(None, subtile_idx)],
            )
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        epilog_sync_barrier.arrive_and_wait()

    epilog_sync_barrier.arrive_and_wait()

    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()
    return acc_consumer_state


class ScaledGemmKernel(G.PersistentDenseGemmKernel):
    fuse_scale = True

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if a.element_type is cutlass.Float32 else None),
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if b.element_type is cutlass.Float32 else None),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )

        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        self.kernel(
            tiled_mma,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            scale_a, scale_b,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive = pipeline.pipeline_init_arrive
        pipeline_init_wait = pipeline.pipeline_init_wait
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # Broadcast scale tensors over the C (M, N, L) shape:
        #  - scale_a: value depends on m only  -> stride (1, 0, 0)
        #  - scale_b: value depends on n only  -> stride (0, 1, 0)
        sa_bc = cute.make_tensor(
            mScaleA.iterator,
            cute.make_layout(mC_mnl.shape, stride=(1, 0, 0)),
        )
        sb_bc = cute.make_tensor(
            mScaleB.iterator,
            cute.make_layout(mC_mnl.shape, stride=(0, 1, 0)),
        )
        gSA_mnl = cute.local_tile(
            sa_bc, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gSB_mnl = cute.local_tile(
            sb_bc, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgSA = thr_mma.partition_C(gSA_mnl)
        tCgSB = thr_mma.partition_C(gSB_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # ---- TMA load warp ----
        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_producer.tail()

        # ---- MMA warp ----
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblk_crd = (None, None, kblk_idx, handle.index)
                            cute.gemm(
                                tiled_mma, tCtAcc,
                                tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        handle.release()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)

        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        # ---- Epilogue warps ----
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * len(self.epilogue_warp_id)
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                num_tiles_executed = tile_sched.num_tiles_executed
                acc_consumer_state = epilogue_tma_store_scaled(
                    self,
                    tidx,
                    warp_idx,
                    tma_atom_c,
                    tCtAcc_base,
                    sC,
                    tCgC,
                    tCgSA,
                    tCgSB,
                    epi_tile,
                    num_tiles_executed,
                    mma_tile_coord_mnl,
                    acc_consumer_state,
                    acc_pipeline,
                    c_pipeline,
                )
            c_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


@cute.jit
def scaled_bmm(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,   # (l, m, k)
    b: cute.Tensor,   # (l, k, n)
    c: cute.Tensor,   # (l, m, n)
    scale_a: cute.Tensor,  # (m,)
    scale_b: cute.Tensor,  # (n,)
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c, scale_a, scale_b, max_active_clusters, stream)


@lru_cache(maxsize=8)
def compile_scaled_bmm(
    mnkl,
    a, b, c, scale_a, scale_b,
    acc_dtype,
    a_major, b_major, c_major,
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    max_active_clusters=None,
    use_2cta_instrs=True,
    use_tma_store=True,
    fuse_scale=True,
):
    from cutlass.cute.runtime import make_fake_stream

    gemm = ScaledGemmKernel(
        acc_dtype, use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, use_tma_store
    )
    gemm.fuse_scale = fuse_scale
    if not gemm.can_implement(
        mnkl, a.element_type, b.element_type, c.element_type, a_major, b_major, c_major
    ):
        raise testing.CantImplementError("config not implementable")
    stream = make_fake_stream()
    return cute.compile(
        scaled_bmm, gemm, a, b, c, scale_a, scale_b, max_active_clusters, stream
    )


def build(M, K, N, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1), fuse_scale=True):
    """Build a fused scaled_mm. Returns (compiled, tensors dict, call_fn)."""
    import torch
    from cutlass.cute.runtime import from_dlpack

    DT_AB, DT_C, DT_ACC = cutlass.Float8E4M3FN, cutlass.BFloat16, cutlass.Float32
    mnkl = (M, N, K, 1)
    a_f32, b_f32, c_f32, a_st, b_st, c_st = prepare_tensors(
        mnkl, DT_AB, DT_AB, DT_C, "k", "k", "n"
    )
    a_ = create_cute_tensor_for_fp8(a_st, DT_AB, 2, source_f32_tensor=a_f32)
    b_ = create_cute_tensor_for_fp8(b_st, DT_AB, 1, source_f32_tensor=b_f32)
    c_ = create_cute_tensor_for_fp8(c_st, DT_C, 2, source_f32_tensor=c_f32)

    sa_t = (torch.rand(M, device="cuda", dtype=torch.float32) + 0.5)
    sb_t = (torch.rand(N, device="cuda", dtype=torch.float32) + 0.5)
    sa_ = from_dlpack(sa_t, assumed_align=4)
    sb_ = from_dlpack(sb_t, assumed_align=4)

    maxcl = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    compiled = compile_scaled_bmm(
        mnkl, a_, b_, c_, sa_, sb_, DT_ACC, "k", "k", "n",
        mma_tiler_mn, cluster_shape_mn, maxcl, True, True, fuse_scale,
    )
    tensors = dict(
        a_f32=a_f32, b_f32=b_f32, c_st=c_st,
        a_=a_, b_=b_, c_=c_, sa_=sa_, sb_=sb_, sa_t=sa_t, sb_t=sb_t,
    )
    return compiled, tensors


# import after class def to avoid circulars
from cutlass.utils import create_cute_tensor_for_fp8  # noqa: E402
