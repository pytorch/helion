"""CPU checks of emitted local TMA tiles, partitions, and sequence advances."""

from __future__ import annotations

import ast
import os
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable

from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import dense_score_plan

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.device_function import DeviceFunction

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
ir = pytest.importorskip("cutlass._mlir.ir")
sm100 = pytest.importorskip("cutlass.utils.blackwell_helpers")


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    initialized = torch.cuda.is_initialized()
    with (
        patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=True),
        _mock_cuda_unavailable(),
        _forbid_native_compile(),
        patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CPU-only test")
        ),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            yield
    assert torch.cuda.is_initialized() == initialized


def _local_tile_call(
    family: str,
    dtype: torch.dtype,
    sequence: int,
    stages: int,
    width: int,
    precompute: bool,
    operand: str,
) -> ast.Call:
    flags = cute_flash.FLASH_PIPELINE_FAMILY_FLAGS[family]
    config = cute_flash.resolve_flash_config(
        64,
        sequence // 128,
        {
            "cute_flash_pipeline_family": family,
            "cute_flash_persistent": True,
            "cute_flash_kv_tile_n": width,
            "cute_flash_kv_stage": stages,
            "cute_flash_kv_order": "descending",
            "cute_flash_softmax_disc": False,
            "cute_flash_stat_transport": "single",
            "cute_flash_precompute_qk_desc": precompute,
            "cute_flash_exp2_packet": "1x1",
            "cute_flash_e2e_schedule": "16/4",
            "cute_flash_rowmax": "software",
            "cute_flash_epi_tma": False,
        },
        dtype=dtype,
        num_bh=64,
        standard_dense_output=True,
        target_device_capability=(10, 3),
    )
    assert config.local_tma_partition
    assert config.tensor_4d_tma is flags.tensor_4d_tma
    assert config.use_clc_scheduler is flags.use_clc_scheduler
    assert (config.kv_tile_n, config.kv_stage) == (width, stages)
    assert config.precompute_qk_desc is precompute
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=64,
        num_kv=sequence // 128,
        sequence_extent=sequence,
        num_bh=64,
        total_tiles=64 * sequence // 256,
        cfg=config,
        has_lse=False,
        io_dtype="cutlass.Float16" if dtype is torch.float16 else "cutlass.BFloat16",
        score_plan=dense_score_plan(64),
        tensor_4d_batch=2 if flags.tensor_4d_tma else 0,
        tensor_4d_heads=32 if flags.tensor_4d_tma else 0,
        target_device_capability=(10, 3),
    )
    calls = [
        node.value
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == f"flash_g{operand}"
            for target in node.targets
        )
    ]
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call, ast.Call)
    assert ast.unparse(call.func) == "cute.local_tile"
    return call


_FAMILIES = (
    "fa4_local_tma",
    "fa4_local_tma_4d",
    "fa4_clc_local_tma",
    "fa4_clc_local_tma_4d",
)
_FAMILY_DTYPES = tuple(
    (family, dtype)
    for family in _FAMILIES
    for dtype in (torch.float16, torch.bfloat16)
    # The existing rank-4 family is FP16-only; do not broaden its eligibility.
    if dtype is torch.float16 or not family.endswith("_4d")
)


@pytest.mark.parametrize(("family", "dtype"), _FAMILY_DTYPES)
@pytest.mark.parametrize(("sequence", "stages"), ((25600, 3), (32768, 9)))
@pytest.mark.parametrize("width", (128, 160))
@pytest.mark.parametrize("precompute", (False, True))
@pytest.mark.parametrize("operand", ("K", "V"))
def test_emitted_local_tma_partition_and_sequence_advance(
    family: str,
    dtype: torch.dtype,
    sequence: int,
    stages: int,
    width: int,
    precompute: bool,
    operand: str,
) -> None:
    call = _local_tile_call(family, dtype, sequence, stages, width, precompute, operand)
    element_type = cutlass.Float16 if dtype is torch.float16 else cutlass.BFloat16
    group = cute.nvgpu.tcgen05.CtaGroup.ONE
    major_k = cute.nvgpu.OperandMajorMode.K
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    if operand == "K":
        mma = sm100.make_trivial_tiled_mma(
            element_type,
            element_type,
            major_k,
            major_k,
            cutlass.Float32,
            group,
            (128, width),
        )
        tile = (128, width, 64)
        shape, stride = (sequence, 64), (64, 1)
    else:
        mma = sm100.make_trivial_tiled_mma(
            element_type,
            element_type,
            major_k,
            cute.nvgpu.OperandMajorMode.MN,
            cutlass.Float32,
            group,
            (128, 64),
            cute.nvgpu.tcgen05.OperandSource.TMEM,
        )
        tile = (128, 64, width)
        shape, stride = (64, sequence), (1, 64)
    rank4 = cute_flash.FLASH_PIPELINE_FAMILY_FLAGS[family].tensor_4d_tma
    shape = (*shape, 32, 2) if rank4 else (*shape, 64)
    stride = (
        (*stride, sequence * 64, 32 * sequence * 64)
        if rank4
        else (*stride, sequence * 64)
    )
    layout = sm100.make_smem_layout_b(mma, tile, element_type, stages)
    gmem = cute.make_tensor(
        cute.make_ptr(element_type, 0, cute.AddressSpace.gmem, assumed_align=128),
        cute.make_layout(shape, stride=stride),
    )
    cluster = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma.thr_id.shape,))
    atom, tensor_map = cute.nvgpu.make_tiled_tma_atom_B(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(group),
        gmem,
        cute.select(layout, mode=[0, 1, 2]),
        tile,
        mma,
        cluster.shape,
    )
    current = tensor_map[None, None, 0, 0] if rank4 else tensor_map[None, None, 0]
    # Execute the compiler's emitted tiler operands against real CuTe layouts.
    # K's first-rank size can be padded correctly despite a wrong 128-row advance.
    local = cute.local_tile(
        current, ast.literal_eval(call.args[1]), ast.literal_eval(call.args[2])
    )
    partition = mma.get_slice(0).partition_B(local)
    gmem_group = cute.group_modes(partition, 0, 3)
    smem = cute.make_tensor(
        cute.make_ptr(element_type, 0, cute.AddressSpace.smem, assumed_align=1024),
        layout,
    )
    smem_group = cute.group_modes(smem, 0, 3)
    shared, global_ = cute.nvgpu.cpasync.tma_partition(
        atom, 0, cute.make_layout(1), smem_group, gmem_group
    )
    assert int(cute.size(shared, mode=[0])) == int(cute.size(global_, mode=[0]))
    iterator = tuple(int(value) for value in gmem_group[None, 1].iterator)
    assert iterator[1] == width
    assert int(gmem_group.shape[1]) == (sequence + width - 1) // width
