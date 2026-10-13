"""Original DV2 adapter: complete role walks, not whole-role renderer callbacks.

Only scalar statements and existing native leaves are retained as text. Every
role, range, first/previous/final and tail branch is an explicit BodyProgram
node; the common interpreter owns their traversal. No expression is rederived.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .chained_body_program import BodyProgram
from .prepared_continuation import ContinuationOpcode
from .prepared_epoch_body import EpochBranch
from .prepared_epoch_body import EpochLoop
from .prepared_epoch_body import EpochPredicate
from .prepared_epoch_body import EpochRange
from .prepared_epoch_issue import descriptor_issue_schedules
from .prepared_epoch_memory import EpochMemoryAction
from .prepared_epoch_memory import EpochMemoryOperation
from .prepared_epoch_product import DescriptorProductCycle
from .prepared_epoch_protocol import EpochEvent
from .prepared_epoch_protocol import EpochPort
from .prepared_epoch_protocol import EpochPosition
from .prepared_epoch_protocol import EpochResourceOperation
from .prepared_epoch_protocol import EpochSynchronization
from .prepared_epoch_protocol import EpochSynchronizationAction
from .prepared_epoch_setup import EpochSetup
from .prepared_epoch_setup import EpochSetupKind
from .prepared_epoch_state import DescriptorStateCycle

if TYPE_CHECKING:
    from .prepared_epoch import PreparedEpoch
    from .prepared_tcgen_binding import PreparedProjectionHost

HEADER = "@cute.kernel\ndef _helion_epoch_kernel(tma_desc_kd: cutlass.GridConstant[cuda.TensorMap], tma_desc_w: cutlass.GridConstant[cuda.TensorMap], tma_desc_qd: cutlass.GridConstant[cuda.TensorMap], tma_desc_v: cutlass.GridConstant[cuda.TensorMap], tma_desc_diag: cutlass.GridConstant[cuda.TensorMap], tma_desc_qk: cutlass.GridConstant[cuda.TensorMap], tma_desc_o: cutlass.GridConstant[cuda.TensorMap], v: cute.Tensor, cu_seqlens: cute.Tensor, cu_chunks: cute.Tensor, initial_state: cute.Tensor, state: cute.Tensor, out: cute.Tensor, head_base: cutlass.Int32, SCALE: cutlass.Float32, K2_RAW_STAGE_COUNT: cutlass.Constexpr, K2_TMA_MBAR_STAGE_COUNT: cutlass.Constexpr, K2_QSTATE_STAGE_COUNT: cutlass.Constexpr, TMEM_ALLOC_COLS: cutlass.Constexpr, KDA_CG1_REGS: cutlass.Constexpr, KDA_SERVICE_REGS: cutlass.Constexpr, PREPARED_EDGE: cutlass.Constexpr[bool]=False, CONTINUATION_PROGRAM: cutlass.Constexpr=()) -> None:"
DECLARATIONS = (
    '"kernel 2, DV-split: each CTA owns half the hidden dimension.\\n\\n    Grid `(num_sequences, launch_heads * 2, 1)`; bidy = head * 2 + half.\\n    Identical schedule and mbar topology to the base chain with M=64\\n    MMAs (PTX Layout F: 16 rows per warp quadrant, lane alignment 0);\\n    kd/W/QK\'/diag are read identically by both halves, v/o/state are\\n    DV-split (one 64-elem s128 TMA segment at value offset half*64).\\n    "',
    "tidx, _, _ = cute.arch.thread_idx()",
    "bidx, bidy, _ = cute.arch.block_idx()",
    "dv_half = bidy % 2",
    "bidy = head_base + bidy // 2",
    "warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())",
    "lane = tidx % THREADS_PER_WARP",
    "sequence_start = cutlass.Int32(cu_seqlens[bidx])",
    "sequence_end = cutlass.Int32(cu_seqlens[bidx + 1])",
    "seqlen = sequence_end - sequence_start",
    "num_chunks = cute.ceil_div(seqlen, BT)",
    "chunk_base = cutlass.Int32(cu_chunks[bidx])",
    "input_dtype = v.element_type",
    "tma_mbar = cutlass.Array(cutlass.Int64, K2_TMA_MBAR_STAGE_COUNT, space=cutlass.AddressSpace.smem, alignment=8)",
    "raw_ready_mbar = cutlass.Array(cutlass.Int64, K2_RAW_STAGE_COUNT, space=cutlass.AddressSpace.smem, alignment=8)",
    "raw_consumed_mbar = cutlass.Array(cutlass.Int64, K2_RAW_STAGE_COUNT, space=cutlass.AddressSpace.smem, alignment=8)",
    "state_input_ready_l_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)",
    "state_input_ready_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)",
    "u_input_ready_mbar = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)",
    "update_ready_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)",
    "shared_acc_ready_mbar = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)",
    "k_restore_consumed_l_mbar = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)",
    "k_restore_consumed_mbar = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)",
    "qstate_acc_ready_mbar = cutlass.Array(cutlass.Int64, K2_QSTATE_STAGE_COUNT, space=cutlass.AddressSpace.smem, alignment=8)",
    "stateq_done_mbar = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)",
    "output_ready_mbar = cutlass.Array(cutlass.Int64, K2_QSTATE_STAGE_COUNT, space=cutlass.AddressSpace.smem, alignment=8)",
    "final_state_stored_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)",
    "tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)",
    "kd_smem = cutlass.Array(input_dtype, K2_RAW_STAGE_COUNT * TILE_ELEMS, space=cutlass.AddressSpace.smem, alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES)",
    "w_smem = cutlass.Array(input_dtype, K2_RAW_STAGE_COUNT * TILE_ELEMS, space=cutlass.AddressSpace.smem, alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES)",
    "qd_smem = cutlass.Array(input_dtype, K2_RAW_STAGE_COUNT * TILE_ELEMS, space=cutlass.AddressSpace.smem, alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES)",
    "v_raw_smem = cutlass.Array(input_dtype, K2_RAW_STAGE_COUNT * V_TILE_ELEMS, space=cutlass.AddressSpace.smem, alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES)",
    "diag_raw_smem = cutlass.Array(cutlass.Float32, K2_RAW_STAGE_COUNT * DIAG_REC_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024)",
    "qk_smem = cutlass.Array(input_dtype, K2_RAW_STAGE_COUNT * QK_REC_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024)",
    "o_smem = cutlass.Array(out.element_type, K2_O_SMEM_TILE_SIZE, space=cutlass.AddressSpace.smem, alignment=O_TMA_SWIZZLE_ALIGNMENT_BYTES)",
)
GLOBALS = (
    "BT",
    "DIAG_REC_ELEMS",
    "DK",
    "DV_HALF",
    "K2_OUTPUT_SMEM_STAGE_COUNT",
    "K2_O_SMEM_STAGE_SIZE",
    "K2_O_SMEM_TILE_SIZE",
    "K2_TX_BYTES",
    "KDA_TMEM_STATE_COL_OFFSET",
    "O_TMA_SWIZZLE_ALIGNMENT_BYTES",
    "QK_REC_ELEMS",
    "RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES",
    "ROLES",
    "TCGEN05_F16_K_ATOM",
    "THREADS_PER_WARP",
    "TILE_ELEMS",
    "V_TILE_ELEMS",
    "advance_ring_stage",
    "cta_sync",
    "cuda",
    "cute",
    "cutlass",
    "epilogue_chain_stage_store",
    "epilogue_chain_tail_store",
    "final_state_stored_arrive",
    "final_state_stored_wait",
    "is_compute_group0_warp",
    "is_compute_group1_warp",
    "is_service_warpgroup",
    "is_tmem_user_warp",
    "output_drain_sync",
    "output_ready_arrive",
    "output_ready_wait",
    "prims",
    "prepared_residual_point",
    "prepared_output_point",
    "raw_consumed_arrive",
    "raw_consumed_wait",
    "raw_ready_arrive",
    "raw_ready_wait",
    "state_input_ready_wait",
    "state_input_ready_arrive",
    "tcgen05_chain_issue_delta_half_mma",
    "tcgen05_chain_issue_qkv_mma",
    "tcgen05_chain_load_qstate_output_regs",
    "tcgen05_chain_stage_vmx_input_tmem",
    "tcgen05_chain_store_output_smem",
    "tcgen05_commit",
    "tcgen05_issue_state_k_mma",
    "tcgen05_issue_state_projection_mma",
    "tcgen05_qstate_acc_tmem_col_offset",
    "tcgen05_rescale_state_dv2_half_regs",
    "tcgen05_shared_input_tmem_col_offset",
    "tcgen05_stage_state_input_dv2_half_tmem",
    "tcgen05_store_final_state_tmem",
    "tcgen05_store_initial_state_tmem",
    "tcgen05_wait_acc_buffer_ready",
    "tma_chain_stage_load_inputs",
    "tma_transfer_wait",
    "tmem_user_sync",
    "update_ready_arrive",
    "update_ready_wait",
)


def original_epoch_program(
    epoch: PreparedEpoch, host: PreparedProjectionHost
) -> BodyProgram:
    from . import chunk_recurrence_sm100 as original

    issue_schedules = descriptor_issue_schedules(host)
    first_raw = EpochEvent(
        host,
        EpochPort.RAW_READY,
        EpochPosition.FIRST,
        ContinuationOpcode.WAIT_TOGGLE,
        returns_phase=False,
    )
    right_raw = EpochEvent(
        host,
        EpochPort.RAW_READY,
        EpochPosition.CURRENT,
        ContinuationOpcode.WAIT_TOGGLE,
        returns_phase=False,
    )
    left_raw = EpochEvent(
        host,
        EpochPort.RAW_READY,
        EpochPosition.CURRENT,
        ContinuationOpcode.WAIT_TOGGLE,
        returns_phase=False,
    )
    first_left = DescriptorStateCycle(host, "right", True, 0, first_raw)
    # This half follows the first half's completed raw wait in the same role.
    first_right = DescriptorStateCycle(host, "right", True, 1, first_raw)
    later_right = DescriptorStateCycle(host, "right", False, 1, right_raw)
    later_left = DescriptorStateCycle(host, "left", False, 0, left_raw)
    first_residual = DescriptorProductCycle(host, "residual", True)
    later_residual = DescriptorProductCycle(host, "residual")
    output = DescriptorProductCycle(host, "output")
    assert output.release is not None and output.space_ready is not None
    assert output.output_complete is not None
    return BodyProgram(
        (
            EpochBranch(
                EpochPredicate.TMA,
                BodyProgram(
                    (
                        EpochBranch(
                            EpochPredicate.ELECTED,
                            BodyProgram(
                                (
                                    EpochRange(
                                        EpochLoop.INIT_TMA,
                                        BodyProgram(
                                            (
                                                EpochEvent(
                                                    host,
                                                    EpochPort.TMA,
                                                    EpochPosition.INITIALIZE,
                                                    None,
                                                    returns_phase=False,
                                                ),
                                            )
                                        ),
                                    ),
                                    EpochRange(
                                        EpochLoop.INIT_RAW,
                                        BodyProgram(
                                            (
                                                EpochEvent(
                                                    host,
                                                    EpochPort.RAW_READY,
                                                    EpochPosition.INITIALIZE,
                                                    None,
                                                    returns_phase=False,
                                                ),
                                                EpochEvent(
                                                    host,
                                                    EpochPort.RAW_FREE,
                                                    EpochPosition.INITIALIZE,
                                                    None,
                                                    returns_phase=False,
                                                ),
                                            )
                                        ),
                                    ),
                                )
                            ),
                            BodyProgram(()),
                        ),
                    )
                ),
                BodyProgram(
                    (
                        EpochBranch(
                            EpochPredicate.CHAIN,
                            BodyProgram(
                                (
                                    EpochBranch(
                                        EpochPredicate.ELECTED,
                                        BodyProgram(
                                            (
                                                EpochEvent(
                                                    host,
                                                    EpochPort.STATE_LEFT,
                                                    EpochPosition.INITIALIZE,
                                                    None,
                                                    returns_phase=False,
                                                ),
                                                EpochEvent(
                                                    host,
                                                    EpochPort.STATE_RIGHT,
                                                    EpochPosition.INITIALIZE,
                                                    None,
                                                    returns_phase=False,
                                                ),
                                                EpochRange(
                                                    EpochLoop.INIT_PAIR,
                                                    BodyProgram(
                                                        (
                                                            EpochEvent(
                                                                host,
                                                                EpochPort.INPUT,
                                                                EpochPosition.INITIALIZE,
                                                                None,
                                                                returns_phase=False,
                                                            ),
                                                            EpochEvent(
                                                                host,
                                                                EpochPort.PROJECTED,
                                                                EpochPosition.INITIALIZE,
                                                                None,
                                                                returns_phase=False,
                                                            ),
                                                            EpochEvent(
                                                                host,
                                                                EpochPort.LEFT_DONE,
                                                                EpochPosition.INITIALIZE,
                                                                None,
                                                                returns_phase=False,
                                                            ),
                                                            EpochEvent(
                                                                host,
                                                                EpochPort.RIGHT_DONE,
                                                                EpochPosition.INITIALIZE,
                                                                None,
                                                                returns_phase=False,
                                                            ),
                                                            EpochEvent(
                                                                host,
                                                                EpochPort.QUERY_DONE,
                                                                EpochPosition.INITIALIZE,
                                                                None,
                                                                returns_phase=False,
                                                            ),
                                                        )
                                                    ),
                                                ),
                                                EpochEvent(
                                                    host,
                                                    EpochPort.UPDATE,
                                                    EpochPosition.INITIALIZE,
                                                    None,
                                                    returns_phase=False,
                                                ),
                                                EpochRange(
                                                    EpochLoop.INIT_OUTPUT,
                                                    BodyProgram(
                                                        (
                                                            EpochEvent(
                                                                host,
                                                                EpochPort.QUERY_ACC,
                                                                EpochPosition.INITIALIZE,
                                                                None,
                                                                returns_phase=False,
                                                            ),
                                                        )
                                                    ),
                                                ),
                                                EpochEvent(
                                                    host,
                                                    EpochPort.FINAL,
                                                    EpochPosition.INITIALIZE,
                                                    None,
                                                    returns_phase=False,
                                                ),
                                            )
                                        ),
                                        BodyProgram(()),
                                    ),
                                )
                            ),
                            BodyProgram(
                                (
                                    EpochBranch(
                                        EpochPredicate.EPILOGUE,
                                        BodyProgram(
                                            (
                                                EpochBranch(
                                                    EpochPredicate.ELECTED,
                                                    BodyProgram(
                                                        (
                                                            EpochRange(
                                                                EpochLoop.INIT_OUTPUT,
                                                                BodyProgram(
                                                                    (
                                                                        EpochEvent(
                                                                            host,
                                                                            EpochPort.OUTPUT_FREE,
                                                                            EpochPosition.INITIALIZE,
                                                                            None,
                                                                            returns_phase=False,
                                                                        ),
                                                                    )
                                                                ),
                                                            ),
                                                        )
                                                    ),
                                                    BodyProgram(()),
                                                ),
                                            )
                                        ),
                                        BodyProgram(()),
                                    ),
                                )
                            ),
                        ),
                    )
                ),
            ),
            EpochSynchronizationAction(host, EpochSynchronization.INITIALIZED),
            EpochSynchronizationAction(host, EpochSynchronization.CTA),
            EpochBranch(
                EpochPredicate.TMEM_USERS,
                BodyProgram(
                    (
                        EpochBranch(
                            EpochPredicate.CHAIN,
                            BodyProgram(
                                (
                                    EpochSynchronizationAction(
                                        host, EpochResourceOperation.ALLOCATE
                                    ),
                                )
                            ),
                            BodyProgram(()),
                        ),
                        EpochSynchronizationAction(
                            host, EpochSynchronization.TMEM_USERS
                        ),
                        EpochBranch(
                            EpochPredicate.CHAIN,
                            BodyProgram(
                                (
                                    EpochSynchronizationAction(
                                        host, EpochResourceOperation.PERMIT
                                    ),
                                )
                            ),
                            BodyProgram(()),
                        ),
                        EpochSynchronizationAction(
                            host, EpochSynchronization.TMEM_USERS
                        ),
                    )
                ),
                BodyProgram(()),
            ),
            EpochSynchronizationAction(host, EpochSynchronization.CTA),
            EpochBranch(
                EpochPredicate.SERVICE,
                BodyProgram(
                    (
                        EpochSynchronizationAction(
                            host, EpochResourceOperation.SERVICE_REGISTERS
                        ),
                    )
                ),
                BodyProgram(()),
            ),
            EpochBranch(
                EpochPredicate.TMA,
                BodyProgram(
                    (
                        EpochSetup(host, EpochSetupKind.RAW_STAGE),
                        EpochSetup(host, EpochSetupKind.RAW_CONSUMED_PHASE),
                        EpochSetup(host, EpochSetupKind.ISSUE_MBAR_SLOT),
                        EpochSetup(host, EpochSetupKind.WAIT_MBAR_SLOT),
                        EpochSetup(host, EpochSetupKind.READY_STAGE),
                        EpochSetup(host, EpochSetupKind.TMA_PHASE),
                        EpochRange(
                            EpochLoop.ALL,
                            BodyProgram(
                                (
                                    EpochSetup(host, EpochSetupKind.WS_CHUNK),
                                    EpochSetup(host, EpochSetupKind.WS_ROW_START),
                                    EpochSetup(host, EpochSetupKind.V_ROW_START),
                                    EpochEvent(
                                        host,
                                        EpochPort.RAW_FREE,
                                        EpochPosition.RAW_CURSOR,
                                        ContinuationOpcode.WAIT_TOGGLE,
                                        returns_phase=False,
                                    ),
                                    EpochMemoryAction(
                                        host, EpochMemoryOperation.INPUTS
                                    ),
                                    EpochSetup(host, EpochSetupKind.ISSUE_MBAR_SLOT_2),
                                    EpochBranch(
                                        EpochPredicate.PIPELINE_FULL,
                                        BodyProgram(
                                            (
                                                EpochEvent(
                                                    host,
                                                    EpochPort.TMA,
                                                    EpochPosition.TMA_CURSOR,
                                                    ContinuationOpcode.WAIT,
                                                    returns_phase=False,
                                                ),
                                                EpochSetup(
                                                    host,
                                                    EpochSetupKind.WAIT_MBAR_SLOT_2,
                                                ),
                                                EpochSetup(
                                                    host, EpochSetupKind.TMA_PHASE_2
                                                ),
                                                EpochEvent(
                                                    host,
                                                    EpochPort.RAW_READY,
                                                    EpochPosition.READY_CURSOR,
                                                    ContinuationOpcode.RELEASE,
                                                    returns_phase=False,
                                                ),
                                                EpochSetup(
                                                    host, EpochSetupKind.READY_STAGE_2
                                                ),
                                            )
                                        ),
                                        BodyProgram(()),
                                    ),
                                    EpochSetup(host, EpochSetupKind.RAW_STAGE_2),
                                    EpochSetup(
                                        host, EpochSetupKind.RAW_CONSUMED_PHASE_2
                                    ),
                                )
                            ),
                        ),
                        EpochSetup(host, EpochSetupKind.TMA_FULL),
                        EpochSetup(host, EpochSetupKind.TMA_TAIL),
                        EpochRange(
                            EpochLoop.TAIL,
                            BodyProgram(
                                (
                                    EpochEvent(
                                        host,
                                        EpochPort.TMA,
                                        EpochPosition.TMA_CURSOR,
                                        ContinuationOpcode.WAIT,
                                        returns_phase=False,
                                    ),
                                    EpochSetup(host, EpochSetupKind.WAIT_MBAR_SLOT_2),
                                    EpochSetup(host, EpochSetupKind.TMA_PHASE_2),
                                    EpochEvent(
                                        host,
                                        EpochPort.RAW_READY,
                                        EpochPosition.READY_CURSOR,
                                        ContinuationOpcode.RELEASE,
                                        returns_phase=False,
                                    ),
                                    EpochSetup(host, EpochSetupKind.READY_STAGE_2),
                                )
                            ),
                        ),
                    )
                ),
                BodyProgram(
                    (
                        EpochBranch(
                            EpochPredicate.QUERY,
                            BodyProgram(
                                (
                                    EpochSetup(host, EpochSetupKind.TMEM_RAW_ADDR),
                                    EpochSetup(host, EpochSetupKind.SI_PHASE12),
                                    EpochSetup(host, EpochSetupKind.UPD_PHASE12),
                                    EpochRange(
                                        EpochLoop.ALL,
                                        BodyProgram(
                                            (
                                                EpochSetup(
                                                    host, EpochSetupKind.RAW_STAGE_3
                                                ),
                                                EpochSetup(
                                                    host, EpochSetupKind.QSTATE_STAGE
                                                ),
                                                EpochSetup(
                                                    host, EpochSetupKind.KR_STAGE
                                                ),
                                                EpochSetup(
                                                    host, EpochSetupKind.ACC_STAGE
                                                ),
                                                EpochBranch(
                                                    EpochPredicate.PREVIOUS,
                                                    BodyProgram(
                                                        (
                                                            EpochSetup(
                                                                host,
                                                                EpochSetupKind.PREV12,
                                                            ),
                                                            EpochEvent(
                                                                host,
                                                                EpochPort.QUERY_ACC,
                                                                EpochPosition.PREVIOUS_QUERY,
                                                                ContinuationOpcode.WAIT_TOGGLE,
                                                                returns_phase=False,
                                                            ),
                                                            EpochEvent(
                                                                host,
                                                                EpochPort.RAW_FREE,
                                                                EpochPosition.PREVIOUS_QUERY,
                                                                ContinuationOpcode.RELEASE,
                                                                returns_phase=False,
                                                            ),
                                                        )
                                                    ),
                                                    BodyProgram(()),
                                                ),
                                                *issue_schedules[
                                                    original.ROLES.super_mma
                                                ],
                                            )
                                        ),
                                    ),
                                )
                            ),
                            BodyProgram(
                                (
                                    EpochBranch(
                                        EpochPredicate.CHAIN,
                                        BodyProgram(
                                            (
                                                EpochSetup(
                                                    host, EpochSetupKind.TMEM_RAW_ADDR
                                                ),
                                                EpochSetup(
                                                    host, EpochSetupKind.SI_L_PHASE
                                                ),
                                                EpochSetup(
                                                    host, EpochSetupKind.SI_PHASE
                                                ),
                                                EpochSetup(
                                                    host, EpochSetupKind.UPD_PHASE
                                                ),
                                                EpochRange(
                                                    EpochLoop.ALL,
                                                    BodyProgram(
                                                        (
                                                            EpochSetup(
                                                                host,
                                                                EpochSetupKind.RAW_STAGE_3,
                                                            ),
                                                            EpochSetup(
                                                                host,
                                                                EpochSetupKind.RAW_PHASE,
                                                            ),
                                                            EpochSetup(
                                                                host,
                                                                EpochSetupKind.ACC_STAGE,
                                                            ),
                                                            EpochSetup(
                                                                host,
                                                                EpochSetupKind.QSTATE_STAGE,
                                                            ),
                                                            EpochSetup(
                                                                host,
                                                                EpochSetupKind.KR_STAGE,
                                                            ),
                                                            EpochSetup(
                                                                host,
                                                                EpochSetupKind.KD_STAGE_SMEM,
                                                            ),
                                                            EpochSetup(
                                                                host,
                                                                EpochSetupKind.W_STAGE_SMEM,
                                                            ),
                                                            *issue_schedules[
                                                                original.ROLES.tcgen05_mma
                                                            ],
                                                        )
                                                    ),
                                                ),
                                                EpochEvent(
                                                    host,
                                                    EpochPort.FINAL,
                                                    EpochPosition.FIRST,
                                                    ContinuationOpcode.WAIT_TOGGLE,
                                                    returns_phase=False,
                                                ),
                                                EpochSetup(
                                                    host, EpochSetupKind.TMEM_PTR
                                                ),
                                                EpochSynchronizationAction(
                                                    host, EpochResourceOperation.RETIRE
                                                ),
                                            )
                                        ),
                                        BodyProgram(
                                            (
                                                EpochBranch(
                                                    EpochPredicate.RIGHT,
                                                    BodyProgram(
                                                        (
                                                            EpochSynchronizationAction(
                                                                host,
                                                                EpochResourceOperation.COMPUTE_REGISTERS,
                                                            ),
                                                            EpochSetup(
                                                                host,
                                                                EpochSetupKind.TMEM_RAW_ADDR,
                                                            ),
                                                            EpochMemoryAction(
                                                                host,
                                                                EpochMemoryOperation.IMPORT_STATE,
                                                            ),
                                                            EpochBranch(
                                                                EpochPredicate.NONEMPTY,
                                                                BodyProgram(
                                                                    (
                                                                        EpochSetup(
                                                                            host,
                                                                            EpochSetupKind.DIAG_RAW_STAGE,
                                                                        ),
                                                                        *first_left.pack(),
                                                                        first_raw,
                                                                        *first_left.decay(),
                                                                        *first_right.pack(),
                                                                        *first_right.decay(),
                                                                        EpochEvent(
                                                                            host,
                                                                            EpochPort.PROJECTED,
                                                                            EpochPosition.FIRST,
                                                                            ContinuationOpcode.WAIT_TOGGLE,
                                                                            returns_phase=False,
                                                                        ),
                                                                        EpochSetup(
                                                                            host,
                                                                            EpochSetupKind.VMX_LANE,
                                                                        ),
                                                                        *first_residual.actions,
                                                                        EpochEvent(
                                                                            host,
                                                                            EpochPort.UPDATE,
                                                                            EpochPosition.CURRENT,
                                                                            ContinuationOpcode.RELEASE,
                                                                            returns_phase=False,
                                                                        ),
                                                                    )
                                                                ),
                                                                BodyProgram(()),
                                                            ),
                                                            EpochRange(
                                                                EpochLoop.LATER,
                                                                BodyProgram(
                                                                    (
                                                                        EpochSetup(
                                                                            host,
                                                                            EpochSetupKind.PREV,
                                                                        ),
                                                                        EpochSetup(
                                                                            host,
                                                                            EpochSetupKind.RAW_STAGE_3,
                                                                        ),
                                                                        EpochSetup(
                                                                            host,
                                                                            EpochSetupKind.DIAG_RAW_STAGE_2,
                                                                        ),
                                                                        EpochSetup(
                                                                            host,
                                                                            EpochSetupKind.ACC_STAGE,
                                                                        ),
                                                                        EpochSetup(
                                                                            host,
                                                                            EpochSetupKind.PREV_KR_PHASE,
                                                                        ),
                                                                        EpochEvent(
                                                                            host,
                                                                            EpochPort.QUERY_DONE,
                                                                            EpochPosition.PREVIOUS,
                                                                            ContinuationOpcode.WAIT_TOGGLE,
                                                                            returns_phase=False,
                                                                        ),
                                                                        EpochEvent(
                                                                            host,
                                                                            EpochPort.RIGHT_DONE,
                                                                            EpochPosition.PREVIOUS,
                                                                            ContinuationOpcode.WAIT_TOGGLE,
                                                                            returns_phase=False,
                                                                        ),
                                                                        *later_right.pack(),
                                                                        EpochEvent(
                                                                            host,
                                                                            EpochPort.RAW_FREE,
                                                                            EpochPosition.PREVIOUS,
                                                                            ContinuationOpcode.RELEASE,
                                                                            returns_phase=False,
                                                                        ),
                                                                        right_raw,
                                                                        *later_right.decay(),
                                                                        EpochEvent(
                                                                            host,
                                                                            EpochPort.PROJECTED,
                                                                            EpochPosition.CURRENT,
                                                                            ContinuationOpcode.WAIT_TOGGLE,
                                                                            returns_phase=False,
                                                                        ),
                                                                        EpochSetup(
                                                                            host,
                                                                            EpochSetupKind.VMX_LANE,
                                                                        ),
                                                                        *later_residual.actions,
                                                                        EpochEvent(
                                                                            host,
                                                                            EpochPort.UPDATE,
                                                                            EpochPosition.CURRENT,
                                                                            ContinuationOpcode.RELEASE,
                                                                            returns_phase=False,
                                                                        ),
                                                                    )
                                                                ),
                                                            ),
                                                            EpochBranch(
                                                                EpochPredicate.NONEMPTY,
                                                                BodyProgram(
                                                                    (
                                                                        EpochSetup(
                                                                            host,
                                                                            EpochSetupKind.LAST,
                                                                        ),
                                                                        EpochSetup(
                                                                            host,
                                                                            EpochSetupKind.LAST_KR_PHASE,
                                                                        ),
                                                                        EpochEvent(
                                                                            host,
                                                                            EpochPort.LEFT_DONE,
                                                                            EpochPosition.FINAL,
                                                                            ContinuationOpcode.WAIT_TOGGLE,
                                                                            returns_phase=False,
                                                                        ),
                                                                        EpochEvent(
                                                                            host,
                                                                            EpochPort.RIGHT_DONE,
                                                                            EpochPosition.FINAL,
                                                                            ContinuationOpcode.WAIT_TOGGLE,
                                                                            returns_phase=False,
                                                                        ),
                                                                    )
                                                                ),
                                                                BodyProgram(()),
                                                            ),
                                                            EpochSetup(
                                                                host,
                                                                EpochSetupKind.FINAL_LANE,
                                                            ),
                                                            EpochMemoryAction(
                                                                host,
                                                                EpochMemoryOperation.EXPORT_STATE,
                                                            ),
                                                            EpochEvent(
                                                                host,
                                                                EpochPort.FINAL,
                                                                EpochPosition.CURRENT,
                                                                ContinuationOpcode.RELEASE,
                                                                returns_phase=False,
                                                            ),
                                                        )
                                                    ),
                                                    BodyProgram(
                                                        (
                                                            EpochBranch(
                                                                EpochPredicate.GROUP0,
                                                                BodyProgram(
                                                                    (
                                                                        EpochBranch(
                                                                            EpochPredicate.OUTPUT,
                                                                            BodyProgram(
                                                                                (
                                                                                    EpochSynchronizationAction(
                                                                                        host,
                                                                                        EpochResourceOperation.COMPUTE_REGISTERS,
                                                                                    ),
                                                                                    EpochSetup(
                                                                                        host,
                                                                                        EpochSetupKind.TMEM_RAW_ADDR,
                                                                                    ),
                                                                                    EpochRange(
                                                                                        EpochLoop.ALL,
                                                                                        BodyProgram(
                                                                                            (
                                                                                                EpochSetup(
                                                                                                    host,
                                                                                                    EpochSetupKind.QSTATE_STAGE,
                                                                                                ),
                                                                                                EpochSetup(
                                                                                                    host,
                                                                                                    EpochSetupKind.QSTATE_PHASE,
                                                                                                ),
                                                                                                EpochSetup(
                                                                                                    host,
                                                                                                    EpochSetupKind.OUTPUT_STAGE,
                                                                                                ),
                                                                                                EpochEvent(
                                                                                                    host,
                                                                                                    EpochPort.QUERY_ACC,
                                                                                                    EpochPosition.CURRENT,
                                                                                                    ContinuationOpcode.WAIT_TOGGLE,
                                                                                                    returns_phase=False,
                                                                                                ),
                                                                                                *output.schedule.phases[
                                                                                                    0
                                                                                                ],
                                                                                                output.release,
                                                                                                EpochBranch(
                                                                                                    EpochPredicate.OUTPUT_LEADER,
                                                                                                    BodyProgram(
                                                                                                        (
                                                                                                            EpochSynchronizationAction(
                                                                                                                host,
                                                                                                                EpochResourceOperation.OUTPUT_SPACE,
                                                                                                            ),
                                                                                                        )
                                                                                                    ),
                                                                                                    BodyProgram(
                                                                                                        ()
                                                                                                    ),
                                                                                                ),
                                                                                                output.space_ready,
                                                                                                EpochSetup(
                                                                                                    host,
                                                                                                    EpochSetupKind.OUTPUT_STAGE_BASE,
                                                                                                ),
                                                                                                *output.schedule.phases[
                                                                                                    2
                                                                                                ],
                                                                                                output.output_complete,
                                                                                                EpochSetup(
                                                                                                    host,
                                                                                                    EpochSetupKind.OUTPUT_CHUNK_START,
                                                                                                ),
                                                                                                EpochBranch(
                                                                                                    EpochPredicate.FULL_OUTPUT,
                                                                                                    BodyProgram(
                                                                                                        (
                                                                                                            EpochBranch(
                                                                                                                EpochPredicate.OUTPUT_LEADER,
                                                                                                                BodyProgram(
                                                                                                                    (
                                                                                                                        EpochMemoryAction(
                                                                                                                            host,
                                                                                                                            EpochMemoryOperation.OUTPUT_FULL,
                                                                                                                            output.binding,
                                                                                                                        ),
                                                                                                                    )
                                                                                                                ),
                                                                                                                BodyProgram(
                                                                                                                    ()
                                                                                                                ),
                                                                                                            ),
                                                                                                        )
                                                                                                    ),
                                                                                                    BodyProgram(
                                                                                                        (
                                                                                                            EpochBranch(
                                                                                                                EpochPredicate.OUTPUT_LEADER,
                                                                                                                BodyProgram(
                                                                                                                    (
                                                                                                                        EpochMemoryAction(
                                                                                                                            host,
                                                                                                                            EpochMemoryOperation.OUTPUT_TAIL,
                                                                                                                            output.binding,
                                                                                                                        ),
                                                                                                                    )
                                                                                                                ),
                                                                                                                BodyProgram(
                                                                                                                    ()
                                                                                                                ),
                                                                                                            ),
                                                                                                        )
                                                                                                    ),
                                                                                                ),
                                                                                            )
                                                                                        ),
                                                                                    ),
                                                                                    EpochBranch(
                                                                                        EpochPredicate.OUTPUT_LEADER,
                                                                                        BodyProgram(
                                                                                            (
                                                                                                EpochSynchronizationAction(
                                                                                                    host,
                                                                                                    EpochResourceOperation.OUTPUT_COMPLETE,
                                                                                                ),
                                                                                            )
                                                                                        ),
                                                                                        BodyProgram(
                                                                                            ()
                                                                                        ),
                                                                                    ),
                                                                                    EpochSynchronizationAction(
                                                                                        host,
                                                                                        EpochSynchronization.OUTPUT,
                                                                                    ),
                                                                                    EpochEvent(
                                                                                        host,
                                                                                        EpochPort.FINAL,
                                                                                        EpochPosition.CURRENT,
                                                                                        ContinuationOpcode.RELEASE,
                                                                                        returns_phase=False,
                                                                                    ),
                                                                                )
                                                                            ),
                                                                            BodyProgram(
                                                                                (
                                                                                    EpochSynchronizationAction(
                                                                                        host,
                                                                                        EpochResourceOperation.COMPUTE_REGISTERS,
                                                                                    ),
                                                                                    EpochSetup(
                                                                                        host,
                                                                                        EpochSetupKind.TMEM_RAW_ADDR,
                                                                                    ),
                                                                                    EpochBranch(
                                                                                        EpochPredicate.NONEMPTY,
                                                                                        BodyProgram(
                                                                                            (
                                                                                                EpochEvent(
                                                                                                    host,
                                                                                                    EpochPort.STATE_LEFT,
                                                                                                    EpochPosition.FIRST,
                                                                                                    ContinuationOpcode.WAIT_TOGGLE,
                                                                                                    returns_phase=False,
                                                                                                ),
                                                                                                EpochSynchronizationAction(
                                                                                                    host,
                                                                                                    EpochSynchronization.ACQUIRED_TMEM,
                                                                                                ),
                                                                                                EpochEvent(
                                                                                                    host,
                                                                                                    EpochPort.UPDATE,
                                                                                                    EpochPosition.CURRENT,
                                                                                                    ContinuationOpcode.RELEASE,
                                                                                                    returns_phase=False,
                                                                                                ),
                                                                                            )
                                                                                        ),
                                                                                        BodyProgram(
                                                                                            ()
                                                                                        ),
                                                                                    ),
                                                                                    EpochRange(
                                                                                        EpochLoop.LATER,
                                                                                        BodyProgram(
                                                                                            (
                                                                                                EpochSetup(
                                                                                                    host,
                                                                                                    EpochSetupKind.PREV,
                                                                                                ),
                                                                                                EpochSetup(
                                                                                                    host,
                                                                                                    EpochSetupKind.RAW_STAGE_3,
                                                                                                ),
                                                                                                EpochSetup(
                                                                                                    host,
                                                                                                    EpochSetupKind.PREV_KR_PHASE,
                                                                                                ),
                                                                                                EpochEvent(
                                                                                                    host,
                                                                                                    EpochPort.QUERY_DONE,
                                                                                                    EpochPosition.PREVIOUS,
                                                                                                    ContinuationOpcode.WAIT_TOGGLE,
                                                                                                    returns_phase=False,
                                                                                                ),
                                                                                                EpochEvent(
                                                                                                    host,
                                                                                                    EpochPort.LEFT_DONE,
                                                                                                    EpochPosition.PREVIOUS,
                                                                                                    ContinuationOpcode.WAIT_TOGGLE,
                                                                                                    returns_phase=False,
                                                                                                ),
                                                                                                *later_left.pack(),
                                                                                                left_raw,
                                                                                                EpochSetup(
                                                                                                    host,
                                                                                                    EpochSetupKind.DIAG_RAW_STAGE_2,
                                                                                                ),
                                                                                                *later_left.decay(),
                                                                                                EpochEvent(
                                                                                                    host,
                                                                                                    EpochPort.UPDATE,
                                                                                                    EpochPosition.CURRENT,
                                                                                                    ContinuationOpcode.RELEASE,
                                                                                                    returns_phase=False,
                                                                                                ),
                                                                                            )
                                                                                        ),
                                                                                    ),
                                                                                )
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                BodyProgram(()),
                                                            ),
                                                        )
                                                    ),
                                                ),
                                            )
                                        ),
                                    ),
                                )
                            ),
                        ),
                    )
                ),
            ),
        )
    )
