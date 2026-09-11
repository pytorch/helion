from __future__ import annotations

import importlib
import inspect

import pytest

from helion._testing import skipUnlessBackends

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
pytestmark = skipUnlessBackends(["cute"])
decomp = importlib.import_module("helion._compiler.cute.chunk_recurrence_dv4_sm100")
primitives = importlib.import_module("helion._compiler.cute.kda_device_primitives")


def test_warp_dv4_geometry_and_arena() -> None:
    assert decomp.RECURRENCE_DV_PARTITIONS == 4
    assert decomp.DV_HALF == 32
    assert decomp.COMPUTE_WARPS == 4
    assert decomp.REC_THREADS == 192
    assert decomp.INPUT_STAGES == 6
    assert decomp.VO_VECTORS_PER_ROW == 4
    assert decomp.OUTPUT_TAIL_TASKS == 64
    assert decomp.OUTPUT_TAIL_ROUNDS == 2
    assert decomp.SMEM_DYNAMIC_BYTES == 97_536


def test_warp_dv4_tensor_map_geometry() -> None:
    specs = decomp.recurrence_tensor_map_specs(
        kd_ptr=0x100000,
        aq_ptr=0x200000,
        gt_ptr=0x300000,
        v_ptr=0x400000,
        out_ptr=0x500000,
        heads=12,
        total_tokens=8192,
        total_chunks=515,
        sequences=6,
        state_ptr=0x600000,
    )
    assert specs["v"].box_dim == (32, 16, 1)
    assert specs["v"].swizzle == "NONE"
    assert specs["out"].box_dim == (32, 16, 1)
    assert specs["state_in"].box_dim == (64, 64, 1)
    assert specs["state_in"] == specs["state_out"]

    # The diagnostic DV32 schedule uses a linear V/O stage to isolate its
    # fragment mapping from the S64 TensorMap swizzle.
    for row in range(16):
        for column in range(32):
            assert decomp.vo_idx(row, column) == row * 32 + column


def test_v2_workspace_factor_key_permutation() -> None:
    for lane in range(32):
        matrix_id = lane // 8
        row = lane % 8 + 8 * (matrix_id % 2)
        for key_block in range(8):
            logical_key = key_block * 16 + 8 * (matrix_id // 2)
            assert decomp.factor_a_fragment_ptr(lane, key_block) == (
                decomp.factor_idx(row, logical_key ^ 8)
            )


def test_warp_dv4_uses_shared_device_primitives() -> None:
    for name in (
        "ldmatrix_x2",
        "ldmatrix_x2_trans",
        "ldmatrix_x4_trans",
        "mma_m16n8k16_bf16",
        "movmatrix_b16",
        "pack_bf16x2",
        "stmatrix_x2",
        "stmatrix_x2_trans",
        "store_vec8_bf16",
        "tma_load_3d",
        "tma_store_3d",
        "tma_store_commit_group",
        "tma_store_wait_read",
        "vec8_bf16",
        "warp_arrive",
    ):
        assert getattr(decomp, name) is getattr(primitives, name)


def test_warp_dv4_exposes_only_the_matched_state_contract() -> None:
    source = inspect.getsource(decomp.emit_bt16_recurrence)
    entry_source = inspect.getsource(decomp._recurrence_entry)
    spec_parameters = inspect.signature(decomp.recurrence_tensor_map_specs).parameters
    for removed_parameter in (
        "FACTOR_KEY_XOR",
        "HAS_STATE_IN",
        "HAS_STATE_OUT",
        "STATE_FP32",
    ):
        assert removed_parameter not in source
        assert removed_parameter not in entry_source
    assert "state_ptr" in spec_parameters
    assert "state_in_ptr" not in spec_parameters
    assert "state_out_ptr" not in spec_parameters
    assert "state_dtype" not in spec_parameters
