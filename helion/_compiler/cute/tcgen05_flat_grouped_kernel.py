"""Helion-owned complete kernel for a typed flat grouped RNA plan.

The template has no benchmark or external-library dependency. Plan fields
specialize complete raw rings, role arrivals, provider spans and packed storage.
The role-local descriptor updates and epilogue are the ordinary Helion native
schedule; the public caller and its guarded dispatch are composed separately.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .tcgen05_grouped_descriptors import WRAP_EXTENT

if TYPE_CHECKING:
    from .tcgen05_flat_grouped_plan import FlatGroupedRnaSchedule

# The scheduler may overwrite a work record as soon as all warp leaders
# release it. Broadcast each leader's shared loads before acknowledging reuse.
_KERNEL = """
from helion._compiler.cute._flash_runtime import ld_volatile_shared_u32

@cute.jit
def _helion_warp_work_field(work_tile, index):
    value = cutlass.Int32(0)
    if cute.arch.lane_idx() == cutlass.Int32(0):
        value = ld_volatile_shared_u32(work_tile.iterator + index)
    return cute.arch.shuffle_sync(value, 0)


@cute.kernel
def _helion_native_grouped_rna(offsets, a_packed, b_grouped, bias, out, tcgen05_grouped_ab_tensormaps, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tcgen05_tma_store_atom, tcgen05_tma_store_tensor):
    tcgen05_storage_pool = cute.arch.alloc_smem(cutlass.Uint8, __native_rna_plan_0, alignment=1024)
    tcgen05_converted_full_ptr = cute.recast_ptr(tcgen05_storage_pool + __native_rna_plan_1, dtype=cutlass.Int64)
    tcgen05_work_tile_smem_ptr = cute.recast_ptr(tcgen05_storage_pool + __native_rna_offset_work_tile, dtype=cutlass.Int32)
    tcgen05_work_tile_smem_tensor = cute.make_tensor(tcgen05_work_tile_smem_ptr, cute.make_layout((9,), stride=(1,)))
    tcgen05_work_tile_smem = tcgen05_work_tile_smem_tensor
    symnode_0 = __native_rna_plan_2
    tcgen05_warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tcgen05_lane_idx = cute.arch.lane_idx()
    mma_tidx = cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(cute.arch.thread_idx()[1]) * cutlass.Int32(32)
    mma_copy_tidx = mma_tidx
    mma_active = cutlass.Int32(cute.arch.thread_idx()[1]) < cutlass.Int32(2)
    tcgen05_tma_warp = tcgen05_warp_idx == cutlass.Int32(5)
    tcgen05_exec_active = tcgen05_warp_idx == cutlass.Int32(4)
    tcgen05_epi_active = tcgen05_warp_idx < cutlass.Int32(4)
    tcgen05_epi_tidx = tcgen05_lane_idx + tcgen05_warp_idx * cutlass.Int32(32) if tcgen05_epi_active else cutlass.Int32(0)
    mma_slice_tidx = cutlass.Int32(0)
    tiled_mma = cutlass.utils.blackwell_helpers.make_trivial_tiled_mma(cutlass.TFloat32, cutlass.TFloat32, cute.nvgpu.OperandMajorMode.MN, cute.nvgpu.OperandMajorMode.K, cutlass.Float32, cute.nvgpu.tcgen05.CtaGroup.ONE, (128, 128), cute.nvgpu.tcgen05.OperandSource.SMEM)
    thr_mma = tiled_mma.get_slice(mma_slice_tidx)
    tcgen05_cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))
    sA_layout = cutlass.utils.blackwell_helpers.make_smem_layout_a(tiled_mma, (128, 128, __native_rna_plan_3), cutlass.Float32, __native_rna_plan_4, is_k_major=False)
    sB_layout = cutlass.utils.blackwell_helpers.make_smem_layout_b(tiled_mma, (128, 128, __native_rna_plan_3), cutlass.Float32, __native_rna_plan_4, is_k_major=True)
    tcgen05_c_layout = cutlass.utils.layout.LayoutEnum.COL_MAJOR
    tcgen05_epi_tile = (cute.make_layout(128), cute.make_layout(32))
    tcgen05_tmem_load_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x4), cutlass.Float32)
    tcgen05_epilogue_rest_mode = cute.make_layout(1, stride=0)
    acc_frag_base = tiled_mma.make_fragment_C(cute.append(tiled_mma.partition_shape_C((128, 128)), 2))
    tcgen05_acc_tmem_cols = cutlass.utils.get_num_tmem_alloc_cols(acc_frag_base, arch='sm_100')
    tcgen05_tmem_holding_buf = cute.recast_ptr(tcgen05_storage_pool + __native_rna_plan_5, dtype=cutlass.Int32)
    tcgen05_tmem_dealloc_mbar_ptr = cute.recast_ptr(tcgen05_storage_pool + __native_rna_plan_6, dtype=cutlass.Int64)
    tcgen05_tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(barrier_id=1, num_threads=160)
    tcgen05_tmem_allocator = cutlass.utils.TmemAllocator(tcgen05_tmem_holding_buf, barrier_for_retrieve=tcgen05_tmem_alloc_barrier, allocator_warp_id=0, is_two_cta=False, two_cta_tmem_dealloc_mbar_ptr=tcgen05_tmem_dealloc_mbar_ptr)
    tcgen05_acc_pipeline_barriers = cute.recast_ptr(tcgen05_storage_pool + __native_rna_plan_7, dtype=cutlass.Int64)
    tcgen05_acc_pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
    tcgen05_acc_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, cutlass.Int32(4))
    tcgen05_acc_pipeline = cutlass.pipeline.PipelineUmmaAsync.create(num_stages=2, producer_group=tcgen05_acc_pipeline_producer_group, consumer_group=tcgen05_acc_pipeline_consumer_group, barrier_storage=tcgen05_acc_pipeline_barriers, cta_layout_vmnk=tcgen05_cluster_layout_vmnk)
    tcgen05_acc_producer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    tcgen05_acc_consumer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    tcgen05_sched_pipeline_mbars = cute.recast_ptr(tcgen05_storage_pool + __native_rna_plan_8, dtype=cutlass.Int64)
    tcgen05_sched_pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)
    tcgen05_sched_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, cutlass.Int32(__native_rna_plan_9))
    tcgen05_sched_pipeline = cutlass.pipeline.PipelineAsync.create(num_stages=1, producer_group=tcgen05_sched_pipeline_producer_group, consumer_group=tcgen05_sched_pipeline_consumer_group, barrier_storage=tcgen05_sched_pipeline_mbars)
    tcgen05_sched_pipeline_producer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
    tcgen05_sched_pipeline_consumer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
    if tcgen05_tma_warp:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
    if tcgen05_warp_idx == cutlass.Int32(0):
        cute.nvgpu.cpasync.prefetch_descriptor(tcgen05_tma_store_atom)
    smem_a = cute.recast_ptr(tcgen05_storage_pool + __native_rna_offset_a, dtype=cutlass.Float32)
    sA = cute.make_tensor(cute.recast_ptr(smem_a, sA_layout.inner, dtype=cutlass.Float32), sA_layout.outer)
    smem_b = cute.recast_ptr(tcgen05_storage_pool + __native_rna_offset_b, dtype=cutlass.Float32)
    sB = cute.make_tensor(cute.recast_ptr(smem_b, sB_layout.inner, dtype=cutlass.Float32), sB_layout.outer)
    tcgen05_tCrA = tiled_mma.make_fragment_A(cute.make_tensor(cute.recast_ptr(smem_a, sA_layout.inner, dtype=cutlass.TFloat32), sA.layout))
    tcgen05_tCrB = tiled_mma.make_fragment_B(cute.make_tensor(cute.recast_ptr(smem_b, sB_layout.inner, dtype=cutlass.TFloat32), sB.layout))
    sA_tma_layout = cute.slice_(sA_layout, (None, None, None, 0))
    sB_tma_layout = cute.slice_(sB_layout, (None, None, None, 0))
    tma_thr_mma = tiled_mma.get_slice(cutlass.Int32(0))
    if __native_wrapped_descriptors:
        pass
    else:
        tcgen05_grouped_tensormap_manager = cutlass.utils.TensorMapManager(cutlass.utils.TensorMapUpdateMode.SMEM, 128)
        tcgen05_grouped_tensormap_grid_dim = cute.arch.grid_dim()
        tcgen05_grouped_tensormap_workspace_idx = cute.arch.block_idx()[2] * tcgen05_grouped_tensormap_grid_dim[1] * tcgen05_grouped_tensormap_grid_dim[0] + cute.arch.block_idx()[1] * tcgen05_grouped_tensormap_grid_dim[0] + cute.arch.block_idx()[0]
        tcgen05_grouped_tensormap_a_ptr = tcgen05_grouped_tensormap_manager.get_tensormap_ptr(tcgen05_grouped_ab_tensormaps[tcgen05_grouped_tensormap_workspace_idx, 0, None].iterator)
        tcgen05_grouped_tensormap_b_ptr = tcgen05_grouped_tensormap_manager.get_tensormap_ptr(tcgen05_grouped_ab_tensormaps[tcgen05_grouped_tensormap_workspace_idx, 1, None].iterator)
        tcgen05_grouped_tensormap_a_desc_ptr = tcgen05_grouped_tensormap_manager.get_tensormap_ptr(tcgen05_grouped_tensormap_a_ptr, cute.AddressSpace.generic)
        tcgen05_grouped_tensormap_b_desc_ptr = tcgen05_grouped_tensormap_manager.get_tensormap_ptr(tcgen05_grouped_tensormap_b_ptr, cute.AddressSpace.generic)
        tcgen05_grouped_tensormap_smem_ptr = cute.recast_ptr(tcgen05_storage_pool + __native_rna_offset_ab_descriptors, dtype=cutlass.Int64)
        tcgen05_grouped_tensormap_a_smem_ptr = tcgen05_grouped_tensormap_smem_ptr
        tcgen05_grouped_tensormap_b_smem_ptr = tcgen05_grouped_tensormap_a_smem_ptr + 16
        tcgen05_grouped_tensormap_init_done = cutlass.Boolean(False)
        tcgen05_grouped_tensormap_last_group = cutlass.Int32(-1)
    tma_cta_layout = cute.make_layout(1)
    tcgen05_ab_pipeline_mbars = cute.recast_ptr(tcgen05_storage_pool + __native_rna_plan_10, dtype=cutlass.Int64)
    tcgen05_ab_pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)
    tcgen05_ab_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, cutlass.Int32(1))
    tcgen05_ab_pipeline_tx_count = cute.size_in_bytes(cutlass.Float32, sA_tma_layout) + cute.size_in_bytes(cutlass.Float32, sB_tma_layout)
    tcgen05_ab_pipeline = cutlass.pipeline.PipelineTmaUmma.create(num_stages=__native_rna_plan_4, producer_group=tcgen05_ab_pipeline_producer_group, consumer_group=tcgen05_ab_pipeline_consumer_group, tx_count=tcgen05_ab_pipeline_tx_count, barrier_storage=tcgen05_ab_pipeline_mbars, cta_layout_vmnk=tcgen05_cluster_layout_vmnk)
    tcgen05_ab_producer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, __native_rna_plan_4)
    tcgen05_ab_consumer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, __native_rna_plan_4)
    if tcgen05_epi_active:
        tcgen05_tmem_allocator.allocate(tcgen05_acc_tmem_cols)
    tcgen05_exec_acc_tmem_ptr = cute.make_ptr(cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16)
    tcgen05_epi_acc_tmem_ptr = cute.make_ptr(cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16)
    cute.arch.sync_threads()
    if tcgen05_exec_active or tcgen05_epi_active:
        tcgen05_tmem_allocator.wait_for_alloc()
    if tcgen05_exec_active:
        tcgen05_exec_acc_tmem_ptr = tcgen05_tmem_allocator.retrieve_ptr(cutlass.Float32)
    if tcgen05_epi_active:
        tcgen05_epi_acc_tmem_ptr = tcgen05_tmem_allocator.retrieve_ptr(cutlass.Float32)
    tcgen05_exec_acc_frag_base = cute.make_tensor(tcgen05_exec_acc_tmem_ptr, acc_frag_base.layout)
    tcgen05_epi_acc_frag_base = cute.make_tensor(tcgen05_epi_acc_tmem_ptr, acc_frag_base.layout)
    tcgen05_epilog_sync_barrier = cutlass.pipeline.NamedBarrier(barrier_id=cutlass.Int32(2), num_threads=cutlass.Int32(128))
    tcgen05_c_pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, cutlass.Int32(128))
    tcgen05_c_pipeline = cutlass.pipeline.PipelineTmaStore.create(num_stages=2, producer_group=tcgen05_c_pipeline_producer_group)
    if __native_wrapped_descriptors:
        pass
    else:
        tcgen05_grouped_d_tensormap_manager = cutlass.utils.TensorMapManager(cutlass.utils.TensorMapUpdateMode.SMEM, 128)
        tcgen05_grouped_d_tensormap_grid_dim = cute.arch.grid_dim()
        tcgen05_grouped_d_tensormap_workspace_idx = cute.arch.block_idx()[2] * tcgen05_grouped_d_tensormap_grid_dim[1] * tcgen05_grouped_d_tensormap_grid_dim[0] + cute.arch.block_idx()[1] * tcgen05_grouped_d_tensormap_grid_dim[0] + cute.arch.block_idx()[0]
        tcgen05_grouped_d_tensormap_ptr = tcgen05_grouped_d_tensormap_manager.get_tensormap_ptr(tcgen05_grouped_ab_tensormaps[tcgen05_grouped_d_tensormap_workspace_idx, 2, None].iterator)
        tcgen05_grouped_d_tensormap_desc_ptr = tcgen05_grouped_d_tensormap_manager.get_tensormap_ptr(tcgen05_grouped_d_tensormap_ptr, cute.AddressSpace.generic)
        tcgen05_grouped_d_tensormap_smem_ptr = cute.recast_ptr(tcgen05_storage_pool + __native_rna_offset_d_descriptor, dtype=cutlass.Int64)
        tcgen05_grouped_d_tensormap_manager.init_tensormap_from_atom(tcgen05_tma_store_atom, tcgen05_grouped_d_tensormap_smem_ptr, 0)
        tcgen05_grouped_d_tensormap_manager.fence_tensormap_initialization()
        tcgen05_grouped_d_tensormap_last_group = cutlass.Int32(-1)
    tcgen05_kernel_desc = type('Tcgen05KernelDesc', (), {'cta_tile_shape_mnk': (128, 128, __native_rna_plan_3), 'c_layout': cutlass.utils.layout.LayoutEnum.COL_MAJOR, 'c_dtype': cutlass.Float32, 'acc_dtype': cutlass.Float32, 'epilog_sync_bar_id': cutlass.Int32(2), 'epilogue_warp_id': (cutlass.Int32(0), cutlass.Int32(1), cutlass.Int32(2), cutlass.Int32(3)), 'num_c_stage': cutlass.Int32(2), 'use_2cta_instrs': False})()
    tcgen05_store_epi_tile = (cute.make_layout(128), cute.make_layout(32))
    tcgen05_sD_layout = cutlass.utils.blackwell_helpers.make_smem_layout_epi(cutlass.Float32, cutlass.utils.layout.LayoutEnum.COL_MAJOR, tcgen05_store_epi_tile, 2)
    tcgen05_sD_ptr = cute.recast_ptr(tcgen05_storage_pool + __native_rna_offset_d, dtype=cutlass.Float32)
    tcgen05_sD = cute.make_tensor(cute.recast_ptr(tcgen05_sD_ptr, tcgen05_sD_layout.inner, dtype=cutlass.Float32), tcgen05_sD_layout.outer)
    tcgen05_tAcc = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout(tcgen05_epi_acc_frag_base)
    tcgen05_converted_pipeline = make_converted_operand_pipeline(tcgen05_ab_pipeline, __native_rna_plan_4, __native_rna_plan_11, defer_sync=True, barrier_storage=tcgen05_converted_full_ptr)
    tcgen05_converter_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, __native_rna_plan_4)
    tcgen05_prefix_ptr = cute.recast_ptr(tcgen05_storage_pool + __native_rna_offset_prefix, dtype=cutlass.Int32)
    tcgen05_prefix = cute.make_tensor(tcgen05_prefix_ptr, cute.make_layout((__native_rna_plan_12,), stride=(1,)))
    if __native_block_prefix:
        tcgen05_prefix_chunk_summaries_ptr = cute.recast_ptr(tcgen05_storage_pool + __native_prefix_summary_offset, dtype=cutlass.Int32)
        tcgen05_prefix_chunk_summaries = cute.make_tensor(tcgen05_prefix_chunk_summaries_ptr, cute.make_layout((__native_prefix_summary_words,), stride=(1,)))
    assert cute.cosize(sA_layout.outer) == __native_rna_operand_words
    assert cute.cosize(sB_layout.outer) == __native_rna_operand_words
    assert sA.layout.stride[3] == __native_rna_plan_13
    assert sB.layout.stride[3] == __native_rna_plan_13
    if __native_block_prefix:
        build_flat_tile_prefix(offsets, tcgen05_prefix, tcgen05_prefix_chunk_summaries, __native_prefix_warps, __native_rna_plan_12, __native_rna_plan_14, __native_rna_plan_2, __native_rna_plan_15, __native_rna_plan_16, __native_rna_plan_17, __native_rna_plan_18, 128, 128)
    else:
        if cutlass.Int32(cute.arch.thread_idx()[1]) == 0:
            build_flat_tile_prefix(offsets, tcgen05_prefix, __native_rna_plan_12, __native_rna_plan_14, __native_rna_plan_2, __native_rna_plan_15, __native_rna_plan_16, __native_rna_plan_17, __native_rna_plan_18, 128, 128)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5):
        tcgen05_sched_pipeline.consumer_wait(tcgen05_sched_pipeline_consumer_state)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        tcgen05_role_local_0_valid = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(2)) == cutlass.Int32(1)
        while tcgen05_role_local_0_valid:
            tcgen05_grouped_cta_tile_idx_m = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(0))
            tcgen05_grouped_cta_tile_idx_n = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(1))
            tcgen05_grouped_metadata_idx = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(3))
            tcgen05_grouped_group_idx = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(3))
            tcgen05_grouped_a_element_base = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(4))
            tcgen05_grouped_problem_m = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(5))
            tcgen05_grouped_problem_n = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(6))
            tcgen05_grouped_problem_k = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(7))
            tcgen05_grouped_output_element_base = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(8))
            cute.arch.sync_warp()
            if cute.arch.lane_idx() == cutlass.Int32(0):
                tcgen05_sched_pipeline.consumer_release(tcgen05_sched_pipeline_consumer_state)
            tcgen05_sched_pipeline_consumer_state.advance()
            cute.arch.sync_warp()
            if __native_wrapped_descriptors:
                gA_tma = cute.local_tile(tma_tensor_a[None, None, tcgen05_grouped_group_idx], (128, __native_rna_plan_3), (tcgen05_grouped_cta_tile_idx_n, None))
                tcgen05_grouped_a_row_base = cutlass.Int64(tcgen05_grouped_a_element_base) // cutlass.Int64(a_packed.layout.stride[0])
                tcgen05_grouped_b_tma = cute.domain_offset((cutlass.Int32(__native_rna_wrap_extent) - tcgen05_grouped_problem_m, None), tma_tensor_b[None, None, cutlass.Int32(__native_rna_wrap_extent), tcgen05_grouped_a_row_base + cutlass.Int64(tcgen05_grouped_problem_m)])
                gB_tma = cute.local_tile(tcgen05_grouped_b_tma, (128, __native_rna_plan_3), (tcgen05_grouped_cta_tile_idx_m, None))
            else:
                tcgen05_grouped_tensormap_group_changed = tcgen05_grouped_metadata_idx != tcgen05_grouped_tensormap_last_group
                if tcgen05_grouped_tensormap_group_changed:
                    if not tcgen05_grouped_tensormap_init_done:
                        tcgen05_grouped_tensormap_manager.init_tensormap_from_atom(tma_atom_a, tcgen05_grouped_tensormap_a_smem_ptr, 5)
                        tcgen05_grouped_tensormap_manager.init_tensormap_from_atom(tma_atom_b, tcgen05_grouped_tensormap_b_smem_ptr, 5)
                        tcgen05_grouped_tensormap_manager.fence_tensormap_initialization()
                        tcgen05_grouped_tensormap_init_done = cutlass.Boolean(True)
                    tcgen05_grouped_tensormap_a_base = b_grouped.iterator + cutlass.Int32(tcgen05_grouped_group_idx) * cutlass.Int32(b_grouped.layout.stride[0])
                    tcgen05_grouped_tensormap_real_a = cute.make_tensor(tcgen05_grouped_tensormap_a_base, cute.make_layout((tcgen05_grouped_problem_n, tcgen05_grouped_problem_k), stride=(b_grouped.layout.stride[1], b_grouped.layout.stride[2])))
                    tcgen05_grouped_tensormap_b_base = a_packed.iterator + cutlass.Int32(tcgen05_grouped_a_element_base)
                    tcgen05_grouped_tensormap_real_b = cute.make_tensor(tcgen05_grouped_tensormap_b_base, cute.make_layout((tcgen05_grouped_problem_m, tcgen05_grouped_problem_k), stride=(a_packed.layout.stride[0], a_packed.layout.stride[1])))
                    tcgen05_grouped_tensormap_manager.update_tensormap((tcgen05_grouped_tensormap_real_a, tcgen05_grouped_tensormap_real_b), (tma_atom_a, tma_atom_b), (tcgen05_grouped_tensormap_a_ptr, tcgen05_grouped_tensormap_b_ptr), 5, (tcgen05_grouped_tensormap_a_smem_ptr, tcgen05_grouped_tensormap_b_smem_ptr))
                    tcgen05_grouped_tensormap_last_group = tcgen05_grouped_metadata_idx
                gA_tma = cute.local_tile(tma_tensor_a, (128, __native_rna_plan_3), (tcgen05_grouped_cta_tile_idx_n, None))
                gB_tma = cute.local_tile(tma_tensor_b, (128, __native_rna_plan_3), (tcgen05_grouped_cta_tile_idx_m, None))
            gA_tma_part = tma_thr_mma.partition_A(gA_tma)
            gB_tma_part = tma_thr_mma.partition_B(gB_tma)
            tma_sA, tma_gA = cute.nvgpu.cpasync.tma_partition(tma_atom_a, 0, tma_cta_layout, cute.group_modes(sA, 0, cute.rank(sA) - 1), cute.group_modes(gA_tma_part, 0, cute.rank(gA_tma_part) - 1))
            tma_sB, tma_gB = cute.nvgpu.cpasync.tma_partition(tma_atom_b, 0, tma_cta_layout, cute.group_modes(sB, 0, cute.rank(sB) - 1), cute.group_modes(gB_tma_part, 0, cute.rank(gB_tma_part) - 1))
            for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(cutlass.Int32(symnode_0)), cutlass.Int32(__native_rna_plan_3)):
                tcgen05_tma_k_tile = tile_offset_2 // cutlass.Int32(__native_rna_plan_3)
                tcgen05_tma_full_tile = tcgen05_grouped_cta_tile_idx_m * cutlass.Int32(128) < tcgen05_grouped_problem_m and tcgen05_grouped_cta_tile_idx_n * cutlass.Int32(128) < tcgen05_grouped_problem_n and (tile_offset_2 < cutlass.Int32(__native_rna_plan_2))
                tcgen05_ab_producer_try_token = cutlass.Boolean(0)
                if tcgen05_tma_full_tile:
                    tcgen05_ab_producer_try_token = tcgen05_ab_pipeline.producer_try_acquire(tcgen05_ab_producer_state)
                    tcgen05_ab_pipeline.producer_acquire(tcgen05_ab_producer_state, tcgen05_ab_producer_try_token)
                    if __native_wrapped_descriptors:
                        pass
                    else:
                        if tcgen05_grouped_tensormap_group_changed and tcgen05_tma_k_tile == cutlass.Int32(0):
                            tcgen05_grouped_tensormap_manager.fence_tensormap_update(tcgen05_grouped_tensormap_a_ptr)
                            tcgen05_grouped_tensormap_manager.fence_tensormap_update(tcgen05_grouped_tensormap_b_ptr)
                    tcgen05_tma_barrier = tcgen05_ab_pipeline.producer_get_barrier(tcgen05_ab_producer_state)
                    cute.copy(tma_atom_a, tma_gA[None, tcgen05_tma_k_tile], tma_sA[None, tcgen05_ab_producer_state.index], tma_bar_ptr=tcgen05_tma_barrier, tma_desc_ptr=tcgen05_grouped_tensormap_a_desc_ptr)
                    cute.copy(tma_atom_b, tma_gB[None, tcgen05_tma_k_tile], tma_sB[None, tcgen05_ab_producer_state.index], tma_bar_ptr=tcgen05_tma_barrier, tma_desc_ptr=tcgen05_grouped_tensormap_b_desc_ptr)
                    tcgen05_ab_pipeline.producer_commit(tcgen05_ab_producer_state)
                    tcgen05_ab_producer_state.advance()
            tcgen05_sched_pipeline.consumer_wait(tcgen05_sched_pipeline_consumer_state)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            tcgen05_role_local_0_valid = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(2)) == cutlass.Int32(1)
        cute.arch.sync_warp()
        if cute.arch.lane_idx() == cutlass.Int32(0):
            tcgen05_sched_pipeline.consumer_release(tcgen05_sched_pipeline_consumer_state)
        tcgen05_sched_pipeline_consumer_state.advance()
        cute.arch.sync_warp()
    if cutlass.Int32(6) <= cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(__native_rna_plan_9):
        tcgen05_sched_pipeline.consumer_wait(tcgen05_sched_pipeline_consumer_state)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        tcgen05_converter_valid = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(2)) == cutlass.Int32(1)
        while tcgen05_converter_valid:
            tcgen05_grouped_cta_tile_idx_m = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(0))
            tcgen05_grouped_cta_tile_idx_n = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(1))
            tcgen05_grouped_problem_m = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(5))
            tcgen05_grouped_problem_n = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(6))
            cute.arch.sync_warp()
            if cute.arch.lane_idx() == cutlass.Int32(0):
                tcgen05_sched_pipeline.consumer_release(tcgen05_sched_pipeline_consumer_state)
            tcgen05_sched_pipeline_consumer_state.advance()
            cute.arch.sync_warp()
            for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(__native_rna_plan_2), cutlass.Int32(__native_rna_plan_3)):
                convert_and_publish_operand_stage(tcgen05_ab_pipeline, tcgen05_converted_pipeline, tcgen05_converter_state, sA, sB, __native_rna_plan_13, __native_rna_plan_13, 128, __native_rna_plan_13, __native_rna_plan_13, 128, __native_rna_plan_11, tcgen05_warp_idx - cutlass.Int32(6))
                tcgen05_converter_state.advance()
            tcgen05_sched_pipeline.consumer_wait(tcgen05_sched_pipeline_consumer_state)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            tcgen05_converter_valid = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(2)) == cutlass.Int32(1)
        cute.arch.sync_warp()
        if cute.arch.lane_idx() == cutlass.Int32(0):
            tcgen05_sched_pipeline.consumer_release(tcgen05_sched_pipeline_consumer_state)
        tcgen05_sched_pipeline_consumer_state.advance()
        cute.arch.sync_warp()
    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(4):
        tcgen05_sched_pipeline.consumer_wait(tcgen05_sched_pipeline_consumer_state)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        tcgen05_role_local_1_valid = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(2)) == cutlass.Int32(1)
        while tcgen05_role_local_1_valid:
            tcgen05_grouped_cta_tile_idx_m = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(0))
            tcgen05_grouped_cta_tile_idx_n = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(1))
            tcgen05_grouped_problem_m = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(5))
            tcgen05_grouped_problem_n = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(6))
            cute.arch.sync_warp()
            if cute.arch.lane_idx() == cutlass.Int32(0):
                tcgen05_sched_pipeline.consumer_release(tcgen05_sched_pipeline_consumer_state)
            tcgen05_sched_pipeline_consumer_state.advance()
            cute.arch.sync_warp()
            acc_frag = tcgen05_exec_acc_frag_base[None, None, None, tcgen05_acc_producer_state.index]
            tcgen05_acc_pipeline.producer_acquire(tcgen05_acc_producer_state)
            tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, False)
            for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(cutlass.Int32(symnode_0)), cutlass.Int32(__native_rna_plan_3)):
                mma_stage = tcgen05_ab_consumer_state.index
                sA_mma = sA[None, None, None, mma_stage]
                sB_mma = sB[None, None, None, mma_stage]
                tcgen05_tma_full_tile = tcgen05_grouped_cta_tile_idx_m * cutlass.Int32(128) < tcgen05_grouped_problem_m and tcgen05_grouped_cta_tile_idx_n * cutlass.Int32(128) < tcgen05_grouped_problem_n and (tile_offset_2 < cutlass.Int32(__native_rna_plan_2))
                tcgen05_ab_consumer_try_token = cutlass.Boolean(0)
                if tcgen05_tma_full_tile:
                    tcgen05_ab_consumer_try_token = tcgen05_converted_pipeline.consumer_try_wait(tcgen05_ab_consumer_state)
                    tcgen05_converted_pipeline.consumer_wait(tcgen05_ab_consumer_state, tcgen05_ab_consumer_try_token)
                for _tcgen05_kblk_idx in range(cute.size(tcgen05_tCrA, mode=[2])):
                    cute.gemm(tiled_mma, acc_frag, [tcgen05_tCrA[None, None, cutlass.Int32(_tcgen05_kblk_idx), mma_stage]], [tcgen05_tCrB[None, None, cutlass.Int32(_tcgen05_kblk_idx), mma_stage]], acc_frag)
                    tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)
                if tcgen05_tma_full_tile:
                    tcgen05_converted_pipeline.consumer_release(tcgen05_ab_consumer_state)
                    tcgen05_ab_consumer_state.advance()
            tcgen05_acc_pipeline.producer_commit(tcgen05_acc_producer_state)
            tcgen05_acc_producer_state.advance()
            tcgen05_sched_pipeline.consumer_wait(tcgen05_sched_pipeline_consumer_state)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            tcgen05_role_local_1_valid = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(2)) == cutlass.Int32(1)
        cute.arch.sync_warp()
        if cute.arch.lane_idx() == cutlass.Int32(0):
            tcgen05_sched_pipeline.consumer_release(tcgen05_sched_pipeline_consumer_state)
        tcgen05_sched_pipeline_consumer_state.advance()
        cute.arch.sync_warp()
    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4):
        tcgen05_tma_store_role_tile = cutlass.Int32(0)
        tcgen05_sched_pipeline.consumer_wait(tcgen05_sched_pipeline_consumer_state)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        tcgen05_role_local_2_valid = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(2)) == cutlass.Int32(1)
        while tcgen05_role_local_2_valid:
            tcgen05_grouped_cta_tile_idx_m = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(0))
            tcgen05_grouped_cta_tile_idx_n = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(1))
            tcgen05_grouped_metadata_idx = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(3))
            tcgen05_grouped_problem_m = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(5))
            tcgen05_grouped_problem_n = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(6))
            tcgen05_grouped_output_element_base = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(8))
            cute.arch.sync_warp()
            if cute.arch.lane_idx() == cutlass.Int32(0):
                tcgen05_sched_pipeline.consumer_release(tcgen05_sched_pipeline_consumer_state)
            tcgen05_sched_pipeline_consumer_state.advance()
            cute.arch.sync_warp()
            if True:
                if tcgen05_warp_idx == cutlass.Int32(0):
                    tcgen05_c_pipeline.producer_acquire()
                if __native_wrapped_descriptors:
                    tcgen05_grouped_d_row_base = cutlass.Int64(tcgen05_grouped_output_element_base) // cutlass.Int64(out.layout.stride[0])
                    tcgen05_grouped_d_tma_rank2 = cute.domain_offset((None, cutlass.Int32(__native_rna_wrap_extent) - tcgen05_grouped_problem_m), tcgen05_tma_store_tensor[None, None, cutlass.Int32(__native_rna_wrap_extent), tcgen05_grouped_d_row_base + cutlass.Int64(tcgen05_grouped_problem_m)])
                    tcgen05_grouped_d_tma = cute.make_tensor(tcgen05_grouped_d_tma_rank2.iterator, cute.append(tcgen05_grouped_d_tma_rank2.layout, cute.make_layout(1, stride=0)))
                    tcgen05_gC = cute.local_tile(tcgen05_grouped_d_tma, (128, 128), (tcgen05_grouped_cta_tile_idx_n, tcgen05_grouped_cta_tile_idx_m, 0))
                else:
                    tcgen05_grouped_d_tensormap_group_changed = tcgen05_grouped_metadata_idx != tcgen05_grouped_d_tensormap_last_group
                    if tcgen05_grouped_d_tensormap_group_changed:
                        tcgen05_grouped_d_tensormap_base = out.iterator + cutlass.Int32(tcgen05_grouped_output_element_base)
                        tcgen05_grouped_d_tensormap_d_nm = cute.make_tensor(tcgen05_grouped_d_tensormap_base, cute.make_layout((tcgen05_grouped_problem_n, tcgen05_grouped_problem_m, cutlass.Int32(1)), stride=(out.layout.stride[1], out.layout.stride[0], cutlass.Int32(0))))
                        tcgen05_grouped_d_tensormap_manager.update_tensormap((tcgen05_grouped_d_tensormap_d_nm,), (tcgen05_tma_store_atom,), (tcgen05_grouped_d_tensormap_ptr,), 0, (tcgen05_grouped_d_tensormap_smem_ptr,))
                        if tcgen05_warp_idx == cutlass.Int32(0):
                            tcgen05_grouped_d_tensormap_manager.fence_tensormap_update(tcgen05_grouped_d_tensormap_ptr)
                        tcgen05_grouped_d_tensormap_last_group = tcgen05_grouped_metadata_idx
                    tcgen05_gC = cute.local_tile(tcgen05_tma_store_tensor, (128, 128), (tcgen05_grouped_cta_tile_idx_n, tcgen05_grouped_cta_tile_idx_m, 0))
                tcgen05_tCgC_base = thr_mma.partition_C(tcgen05_gC)
                tcgen05_tCgC = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout(tcgen05_tCgC_base)
                tcgen05_tCgC_planned = cute.make_tensor(tcgen05_tCgC.iterator, cute.append(cute.append(cute.append(tcgen05_tCgC.layout, tcgen05_epilogue_rest_mode), tcgen05_epilogue_rest_mode), tcgen05_epilogue_rest_mode))
                tcgen05_tAcc_epi = cute.flat_divide(tcgen05_tAcc, tcgen05_store_epi_tile)
                tcgen05_tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(tcgen05_tmem_load_atom, tcgen05_tAcc_epi[None, None, 0, 0, 0])
                tcgen05_thr_copy_t2r = tcgen05_tiled_copy_t2r.get_slice(tcgen05_epi_tidx)
                tcgen05_tTR_tAcc_base = tcgen05_thr_copy_t2r.partition_S(tcgen05_tAcc_epi)
                tcgen05_tCgC_epi = cute.flat_divide(tcgen05_tCgC_planned, tcgen05_store_epi_tile)
                tcgen05_tTR_gC = tcgen05_thr_copy_t2r.partition_D(tcgen05_tCgC_epi)
                tcgen05_tTR_rAcc = cute.make_rmem_tensor(tcgen05_tTR_gC[None, None, None, 0, 0, 0, 0, 0].shape, cutlass.Float32)
                tcgen05_tTR_rD = cute.make_rmem_tensor(tcgen05_tTR_rAcc.shape, cutlass.Float32)
                tcgen05_selected_nm_stsm_atom = cutlass.utils.blackwell_helpers.get_smem_store_op(cutlass.utils.layout.LayoutEnum.COL_MAJOR, cutlass.Float32, cutlass.Float32, tcgen05_tiled_copy_t2r)
                tcgen05_tiled_copy_r2s = cute.make_tiled_copy_D(tcgen05_selected_nm_stsm_atom, tcgen05_tiled_copy_t2r)
                tcgen05_thr_copy_r2s = tcgen05_tiled_copy_r2s.get_slice(tcgen05_epi_tidx)
                tcgen05_tRS_sD = tcgen05_thr_copy_r2s.partition_D(tcgen05_sD)
                tcgen05_tRS_rD = tcgen05_tiled_copy_r2s.retile(tcgen05_tTR_rD)
                tcgen05_tRS_rAcc = tcgen05_tiled_copy_r2s.retile(tcgen05_tTR_rAcc)
                tcgen05_bSG_sD, tcgen05_bSG_gD_partitioned = cute.nvgpu.cpasync.tma_partition(tcgen05_tma_store_atom, 0, cute.make_layout(1), cute.group_modes(tcgen05_sD, 0, 2), cute.group_modes(tcgen05_tCgC_epi, 0, 2))
                tcgen05_bSG_gD = tcgen05_bSG_gD_partitioned[None, None, None, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)]
                tcgen05_bSG_gD = cute.group_modes(tcgen05_bSG_gD, 1, cute.rank(tcgen05_bSG_gD))
                tcgen05_tTR_tAcc_stage = tcgen05_tTR_tAcc_base[None, None, None, None, None, tcgen05_acc_consumer_state.index]
                tcgen05_tTR_tAcc = cute.group_modes(tcgen05_tTR_tAcc_stage, 3, cute.rank(tcgen05_tTR_tAcc_stage))
                tcgen05_subtile_count = cutlass.const_expr(cute.size(tcgen05_tTR_tAcc.shape, mode=[3]))
                for _tcgen05_subtile in cutlass.range(tcgen05_subtile_count, unroll_full=True):
                    if _tcgen05_subtile != 0 and tcgen05_warp_idx == cutlass.Int32(0):
                        tcgen05_c_pipeline.producer_acquire()
                    if _tcgen05_subtile == 0:
                        tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)
                    tcgen05_tTR_tAcc_nm = tcgen05_tTR_tAcc[None, None, None, cutlass.Int32(_tcgen05_subtile)]
                    cute.copy(tcgen05_tiled_copy_t2r, tcgen05_tTR_tAcc_nm, tcgen05_tTR_rAcc)
                    if True:
                        tcgen05_cC = cute.make_identity_tensor((128, 128))
                        tcgen05_tCcC_base = thr_mma.partition_C(tcgen05_cC)
                        tcgen05_tCcC = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout(tcgen05_tCcC_base)
                        tcgen05_tCcC_epi = cute.flat_divide(tcgen05_tCcC, tcgen05_store_epi_tile)
                        tcgen05_tTR_cC = tcgen05_thr_copy_t2r.partition_D(tcgen05_tCcC_epi)
                        tcgen05_tRS_cC = tcgen05_tiled_copy_r2s.retile(tcgen05_tTR_cC)
                        tcgen05_tRS_cC_grouped = cute.group_modes(tcgen05_tRS_cC, 3, cute.rank(tcgen05_tRS_cC))
                        tcgen05_tRS_cC_subtile = tcgen05_tRS_cC_grouped[None, None, None, cutlass.Int32(_tcgen05_subtile)]
                        for _epi_i in cutlass.range_constexpr(cute.size(tcgen05_tRS_rAcc)):
                            _epi_coord = tcgen05_tRS_cC_subtile[_epi_i]
                            _epi_column = cutlass.Int32(tcgen05_grouped_cta_tile_idx_n) * cutlass.Int32(128) + cutlass.Int32(_epi_coord[0])
                            _epi_bias = cutlass.Float32(0.0)
                            if True and _epi_column < tcgen05_grouped_problem_n:
                                _epi_bias = (bias.iterator + cutlass.Int64(tcgen05_grouped_metadata_idx) * cutlass.Int64(bias.layout.stride[0]) + cutlass.Int64(_epi_column) * cutlass.Int64(bias.layout.stride[1])).load()
                            if True:
                                tcgen05_tRS_rD[_epi_i] = tcgen05_tRS_rAcc[_epi_i] + _epi_bias
                            else:
                                tcgen05_tRS_rD[_epi_i] = tcgen05_tRS_rAcc[_epi_i]
                        if _tcgen05_subtile == tcgen05_subtile_count - 1:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)
                    tcgen05_epilog_sync_barrier.arrive_and_wait()
                    tcgen05_c_buffer = (tcgen05_tma_store_role_tile * cutlass.Int32(tcgen05_subtile_count) + cutlass.Int32(_tcgen05_subtile)) % cutlass.Int32(2)
                    cute.copy(tcgen05_tiled_copy_r2s, tcgen05_tRS_rD, tcgen05_tRS_sD[None, None, None, tcgen05_c_buffer])
                    cute.arch.fence_view_async_shared()
                    tcgen05_epilog_sync_barrier.arrive_and_wait()
                    if tcgen05_warp_idx == cutlass.Int32(0):
                        cute.copy(tcgen05_tma_store_atom, tcgen05_bSG_sD[None, tcgen05_c_buffer], tcgen05_bSG_gD[None, cutlass.Int32(_tcgen05_subtile)], tma_desc_ptr=tcgen05_grouped_d_tensormap_desc_ptr)
                        tcgen05_c_pipeline.producer_commit()
                if tcgen05_epi_active:
                    tcgen05_acc_consumer_state.advance()
            tcgen05_tma_store_role_tile = tcgen05_tma_store_role_tile + cutlass.Int32(1)
            tcgen05_sched_pipeline.consumer_wait(tcgen05_sched_pipeline_consumer_state)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            tcgen05_role_local_2_valid = _helion_warp_work_field(tcgen05_work_tile_smem, cutlass.Int32(2)) == cutlass.Int32(1)
        cute.arch.sync_warp()
        if cute.arch.lane_idx() == cutlass.Int32(0):
            tcgen05_sched_pipeline.consumer_release(tcgen05_sched_pipeline_consumer_state)
        tcgen05_sched_pipeline_consumer_state.advance()
        cute.arch.sync_warp()
    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(__native_rna_plan_9):
        tcgen05_ordinal = cutlass.Int64(cute.arch.block_idx()[2])
        tcgen05_stride = cutlass.Int64(cute.arch.grid_dim()[2])
        tcgen05_continue = cutlass.Boolean(True)
        while tcgen05_continue:
            tcgen05_sched_pipeline.producer_acquire(tcgen05_sched_pipeline_producer_state)
            tcgen05_record = resolve_flat_nm_work(offsets, tcgen05_prefix, tcgen05_ordinal, __native_rna_plan_12, __native_rna_plan_14, __native_rna_plan_2, __native_rna_plan_15, __native_rna_plan_16, __native_rna_plan_17, __native_rna_plan_18, 128, 128)
            if cute.arch.lane_idx() == cutlass.Int32(0):
                for _record_i in cutlass.range_constexpr(9):
                    tcgen05_work_tile_smem[cutlass.Int32(_record_i)] = tcgen05_record[_record_i]
                tcgen05_sched_pipeline.producer_commit(tcgen05_sched_pipeline_producer_state)
            tcgen05_sched_pipeline_producer_state.advance()
            tcgen05_continue = tcgen05_record[2] == cutlass.Int32(1)
            tcgen05_ordinal = tcgen05_ordinal + tcgen05_stride
            cute.arch.sync_warp()
    if tcgen05_warp_idx == cutlass.Int32(0):
        tcgen05_c_pipeline.producer_tail()
    if tcgen05_tma_warp:
        tcgen05_ab_pipeline.producer_tail(tcgen05_ab_producer_state)
    if tcgen05_exec_active:
        tcgen05_tmem_alloc_barrier.arrive()
    if tcgen05_exec_active:
        tcgen05_acc_pipeline.producer_tail(tcgen05_acc_producer_state)
    tcgen05_tmem_allocator = cutlass.utils.TmemAllocator(tcgen05_tmem_holding_buf, barrier_for_retrieve=tcgen05_tmem_alloc_barrier, allocator_warp_id=0, is_two_cta=False, two_cta_tmem_dealloc_mbar_ptr=tcgen05_tmem_dealloc_mbar_ptr, num_allocated_columns=tcgen05_acc_tmem_cols, initialize_mbarrier=False)
    cute.arch.sync_threads()
    if tcgen05_epi_active:
        tcgen05_tmem_allocator.relinquish_alloc_permit()
    if tcgen05_epi_active:
        tcgen05_tmem_alloc_barrier.arrive_and_wait()
    if tcgen05_epi_active:
        tcgen05_tmem_allocator.free(tcgen05_epi_acc_tmem_ptr)
"""


def render_flat_grouped_rna_kernel(schedule: FlatGroupedRnaSchedule) -> str:
    offsets = {p.allocation.name: p.offset for p in schedule.storage.placements}
    values = {
        **(
            {
                "__native_prefix_summary_offset": offsets[
                    "tcgen05_prefix_chunk_summaries_ptr"
                ],
                "__native_prefix_summary_words": schedule.block_prefix.summary_bytes
                // 4,
                "__native_prefix_warps": schedule.block_prefix.warps,
            }
            if schedule.block_prefix is not None
            else {}
        ),
        "__native_rna_wrap_extent": WRAP_EXTENT,
        "__native_rna_plan_0": schedule.storage.bytes,
        **(
            {"__native_rna_plan_1": offsets["tcgen05_converted_full_ptr"]}
            if not schedule.tma_rn
            else {}
        ),
        "__native_rna_plan_2": schedule.reduction,
        "__native_rna_plan_3": schedule.block_k,
        "__native_rna_plan_4": schedule.ab_stages,
        "__native_rna_plan_5": offsets["tcgen05_tmem_holding_buf"],
        "__native_rna_plan_6": offsets["tcgen05_tmem_dealloc_mbar_ptr"],
        "__native_rna_plan_7": offsets["tcgen05_acc_pipeline_barriers"],
        "__native_rna_plan_8": offsets["tcgen05_sched_pipeline_mbars"],
        "__native_rna_plan_9": schedule.roles.scheduler_warp,
        "__native_rna_plan_10": offsets["tcgen05_ab_pipeline_mbars"],
        "__native_rna_plan_11": schedule.roles.converter_warps,
        "__native_rna_plan_12": schedule.provider.groups,
        "__native_rna_plan_13": 128 * schedule.block_k,
        "__native_rna_plan_14": schedule.provider.a.elements,
        "__native_rna_plan_15": schedule.provider.a.row_stride,
        "__native_rna_plan_16": schedule.provider.output.elements,
        "__native_rna_plan_17": schedule.columns,
        "__native_rna_plan_18": schedule.provider.output.row_stride,
        "__native_rna_operand_words": 128 * schedule.block_k * schedule.ab_stages,
        "__native_rna_offset_work_tile": offsets["tcgen05_work_tile_smem_ptr"],
        "__native_rna_offset_a": offsets["smem_a"],
        "__native_rna_offset_b": offsets["smem_b"],
        "__native_rna_offset_ab_descriptors": offsets[
            "tcgen05_grouped_tensormap_smem_ptr"
        ],
        "__native_rna_offset_d_descriptor": offsets[
            "tcgen05_grouped_d_tensormap_smem_ptr"
        ],
        "__native_rna_offset_d": offsets["tcgen05_sD_ptr"],
        "__native_rna_offset_prefix": offsets["tcgen05_prefix_ptr"],
    }

    permit_sites = {"allocate": 0, "relinquish_alloc_permit": 0}

    class Substitute(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
            if (
                schedule.tma_rn
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id
                in {
                    "tcgen05_converted_full_ptr",
                    "tcgen05_converted_pipeline",
                    "tcgen05_converter_state",
                }
            ):
                return None
            return self.generic_visit(node)

        def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
            if (
                isinstance(node.test, ast.Name)
                and node.test.id == "__native_block_prefix"
            ):
                selected = (
                    node.body if schedule.block_prefix is not None else node.orelse
                )
                return [self.visit(statement) for statement in selected]
            if (
                schedule.resident_ctas == 2
                and isinstance(node.test, ast.Name)
                and node.test.id == "tcgen05_epi_active"
                and not node.orelse
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Call)
            ):
                call = node.body[0].value
                if (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "tcgen05_tmem_allocator"
                    and call.func.attr in permit_sites
                ):
                    permit_sites[call.func.attr] += 1
                    if call.func.attr == "relinquish_alloc_permit":
                        return []
                    # The same allocator warp relinquishes after its sole
                    # allocation. Publication, common wait and final free stay.
                    node.body.extend(
                        ast.parse(
                            "tcgen05_tmem_allocator.relinquish_alloc_permit()"
                        ).body
                    )
                    return self.generic_visit(node)
            if schedule.tma_rn and (
                isinstance(node.test, ast.Compare)
                and len(node.test.ops) == 2
                and isinstance(node.test.ops[0], ast.LtE)
                and isinstance(node.test.ops[1], ast.Lt)
                and isinstance(node.test.comparators[-1], ast.Call)
                and any(
                    isinstance(arg, ast.Name) and arg.id == "__native_rna_plan_9"
                    for arg in node.test.comparators[-1].args
                )
            ):
                # The sole converter-role guard in this owned template. RN
                # removes its mailbox consumer and every per-stage SIMT task.
                return []
            if (
                isinstance(node.test, ast.Name)
                and node.test.id == "__native_wrapped_descriptors"
            ):
                selected = (
                    node.body if schedule.descriptors is not None else node.orelse
                )
                return [
                    self.visit(statement)
                    for statement in selected
                    if not isinstance(statement, ast.Pass)
                ]
            return self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> ast.AST:
            if (
                schedule.descriptors is not None
                and ast.unparse(node.func) == "cute.copy"
            ):
                node.keywords = [
                    keyword
                    for keyword in node.keywords
                    if keyword.arg != "tma_desc_ptr"
                ]
            return self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if schedule.tma_rn and node.id == "tcgen05_converted_pipeline":
                # The original TMA full/UMMA empty ring now directly connects
                # TMA and MMA. Releasing it remains ordered after async MMA.
                return ast.copy_location(
                    ast.Name(id="tcgen05_ab_pipeline", ctx=node.ctx), node
                )
            if node.id in values:
                return ast.Constant(value=values[node.id])
            return node

    tree = Substitute().visit(ast.parse(_KERNEL))
    if schedule.resident_ctas == 2:
        assert permit_sites == {"allocate": 1, "relinquish_alloc_permit": 1}
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
