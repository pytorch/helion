"""Closed original scalar setup, never an issuer/wait/publication callback."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from . import chained_matmul as chain

if TYPE_CHECKING:
    from .prepared_tcgen_binding import PreparedProjectionHost


class EpochSetupKind(Enum):
    RAW_STAGE = "raw_stage = cutlass.Int32(0)"
    RAW_CONSUMED_PHASE = "raw_consumed_phase = cutlass.Int32(1)"
    ISSUE_MBAR_SLOT = "issue_mbar_slot = cutlass.Int32(0)"
    WAIT_MBAR_SLOT = "wait_mbar_slot = cutlass.Int32(0)"
    READY_STAGE = "ready_stage = cutlass.Int32(0)"
    TMA_PHASE = "tma_phase = cutlass.Int32(0)"
    TMA_FULL = "tma_full = cutlass.Int32(num_chunks >= cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1))"
    TMA_TAIL = "tma_tail = tma_full * cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1) + (cutlass.Int32(1) - tma_full) * num_chunks"
    WS_CHUNK = "ws_chunk = chunk_base + chunk"
    WS_ROW_START = "ws_row_start = ws_chunk * cutlass.Int32(BT)"
    V_ROW_START = "v_row_start = sequence_start + chunk * cutlass.Int32(BT)"
    ISSUE_MBAR_SLOT_2 = "issue_mbar_slot, _ = advance_ring_stage(issue_mbar_slot, 1, K2_TMA_MBAR_STAGE_COUNT)"
    RAW_STAGE_2 = (
        "raw_stage, raw_wrapped = advance_ring_stage(raw_stage, 1, K2_RAW_STAGE_COUNT)"
    )
    RAW_CONSUMED_PHASE_2 = "raw_consumed_phase = raw_consumed_phase ^ raw_wrapped"
    WAIT_MBAR_SLOT_2 = "wait_mbar_slot, wait_wrapped = advance_ring_stage(wait_mbar_slot, 1, K2_TMA_MBAR_STAGE_COUNT)"
    TMA_PHASE_2 = "tma_phase = tma_phase ^ wait_wrapped"
    READY_STAGE_2 = (
        "ready_stage, _ = advance_ring_stage(ready_stage, 1, K2_RAW_STAGE_COUNT)"
    )
    TMEM_RAW_ADDR = "tmem_raw_addr = tmem_ptr_i32.load()"
    SI_PHASE12 = "si_phase12 = cutlass.Int32(0)"
    UPD_PHASE12 = "upd_phase12 = cutlass.Int32(0)"
    RAW_STAGE_3 = "raw_stage = chunk % K2_RAW_STAGE_COUNT"
    QSTATE_STAGE = "qstate_stage = chunk % K2_QSTATE_STAGE_COUNT"
    KR_STAGE = "kr_stage = chunk % 2"
    ACC_STAGE = "acc_stage = chunk % 2"
    XPACK_COL12 = "xpack_col12 = tcgen05_shared_input_tmem_col_offset(acc_stage)"
    SI_L_PHASE = "si_l_phase = cutlass.Int32(0)"
    SI_PHASE = "si_phase = cutlass.Int32(0)"
    UPD_PHASE = "upd_phase = cutlass.Int32(0)"
    TMEM_PTR = "tmem_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Float32)"
    PREV12 = "prev12 = chunk - cutlass.Int32(1)"
    RAW_PHASE = "raw_phase = chunk // K2_RAW_STAGE_COUNT % 2"
    KD_STAGE_SMEM = "kd_stage_smem = kd_smem.subview(raw_stage * TILE_ELEMS)"
    W_STAGE_SMEM = "w_stage_smem = w_smem.subview(raw_stage * TILE_ELEMS)"
    XPACK_COL = "xpack_col = tcgen05_shared_input_tmem_col_offset(acc_stage)"
    FINAL_LANE = "final_lane = cute.arch.lane_idx()"
    DIAG_RAW_STAGE = "diag_raw_stage = diag_raw_smem.subview(0)"
    VMX_LANE = "vmx_lane = cute.arch.lane_idx()"
    PREV = "prev = chunk - cutlass.Int32(1)"
    DIAG_RAW_STAGE_2 = (
        "diag_raw_stage = diag_raw_smem.subview(raw_stage * DIAG_REC_ELEMS)"
    )
    PREV_KR_PHASE = "prev_kr_phase = prev // 2 % 2"
    LAST = "last = num_chunks - cutlass.Int32(1)"
    LAST_KR_PHASE = "last_kr_phase = last // 2 % 2"
    QSTATE_PHASE = "qstate_phase = chunk // K2_QSTATE_STAGE_COUNT % 2"
    OUTPUT_STAGE = "output_stage = chunk % K2_OUTPUT_SMEM_STAGE_COUNT"
    OUTPUT_STAGE_BASE = "output_stage_base = output_stage * K2_O_SMEM_STAGE_SIZE"
    OUTPUT_CHUNK_START = "output_chunk_start = chunk * BT"


@dataclass(frozen=True)
class EpochSetup:
    owner: PreparedProjectionHost
    operation: EpochSetupKind

    def facts(self) -> object:
        if not isinstance(self.operation, EpochSetupKind):
            raise chain._UnsupportedChain("unknown original scalar setup")
        return id(self), id(self.owner), self.operation, self.operation.value

    def emit(self) -> list[str]:
        self.facts()
        return [self.operation.value]
