from __future__ import annotations

import ast
import os
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compat import requires_cuda_version
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_DEEPGEMM_SELECTED_COMPACT_METADATA_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_DEEPGEMM_SELECTED_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_DEEPGEMM_SELECTED_SOURCE_M_TILE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SELECTED_ACCUMULATOR_VIEW_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SELECTED_D_STORE_VIEW_CONFIG_KEY,
)
from helion._testing import DEVICE
from helion._testing import matchesBackends
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime import _append_cute_wrapper_plan

pytestmark = skipUnlessBackends(["cute"])
if matchesBackends(["cute"]):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")


def _aligned_m(actual_m: int) -> int:
    return (
        (actual_m + TCGEN05_DEEPGEMM_SELECTED_SOURCE_M_TILE - 1)
        // TCGEN05_DEEPGEMM_SELECTED_SOURCE_M_TILE
    ) * TCGEN05_DEEPGEMM_SELECTED_SOURCE_M_TILE


def _deepgemm_selected_config(*, compact: bool = True) -> helion.Config:
    config = helion.Config(
        block_sizes=[256, 128, 64],
        l2_groupings=[1],
        loop_orders=[[0, 1, 2]],
        num_stages=7,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=7,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
    )
    config.config[TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY] = True
    config.config[TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY] = True
    config.config[TCGEN05_DEEPGEMM_SELECTED_CONFIG_KEY] = True
    if compact:
        config.config[TCGEN05_DEEPGEMM_SELECTED_COMPACT_METADATA_CONFIG_KEY] = True
    config.config[TCGEN05_SELECTED_ACCUMULATOR_VIEW_CONFIG_KEY] = "nm"
    config.config[TCGEN05_SELECTED_D_STORE_VIEW_CONFIG_KEY] = "nm_transposed"
    return config


@helion.kernel(backend="cute", static_shapes=False)
def _deepgemm_selected_kernel(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    m_total_aligned, k = a_packed.shape
    _g, n, k2 = b_grouped.shape
    assert k == k2
    assert work_tile_metadata.size(1) == 4
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), 256, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        store_m = work_tile_metadata[work_id, 3]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        store_rows = local_m < store_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


@helion.kernel(backend="cute", static_shapes=False)
def _deepgemm_selected_out_kernel(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m_total_aligned, k = a_packed.shape
    _g, n, k2 = b_grouped.shape
    assert k == k2
    assert out.size(0) == m_total_aligned
    assert out.size(1) == n
    assert work_tile_metadata.size(1) == 4
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), 256, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        store_m = work_tile_metadata[work_id, 3]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        store_rows = local_m < store_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


@helion.kernel(backend="cute", static_shapes=False)
def _deepgemm_selected_bad_epilogue_kernel(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    m_total_aligned, k = a_packed.shape
    _g, n, k2 = b_grouped.shape
    assert k == k2
    assert work_tile_metadata.size(1) == 4
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), 256, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        store_m = work_tile_metadata[work_id, 3]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        store_rows = local_m < store_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            (acc + 1.0).to(out.dtype),
            extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


def _make_selected_args(
    m_sizes: tuple[int, ...] = (17, 11),
    *,
    n: int = 128,
    k: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    real_groups: tuple[int, ...] | None = None,
    dirty_padding: bool = False,
    compact: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if real_groups is None:
        real_groups = tuple(range(len(m_sizes)))
    assert len(real_groups) == len(m_sizes)
    starts: list[int] = []
    cursor = 0
    for actual_m in m_sizes:
        starts.append(cursor)
        cursor += _aligned_m(actual_m)
    a_packed = torch.zeros((cursor, k), device=DEVICE, dtype=dtype)
    for start, actual_m in zip(starts, m_sizes, strict=True):
        a_packed[start : start + actual_m].normal_()
        if dirty_padding:
            a_packed[start + actual_m : start + _aligned_m(actual_m)].normal_()
    group_count = max(real_groups) + 1 if real_groups else 0
    b_grouped = torch.randn((group_count, n, k), device=DEVICE, dtype=dtype)
    worklist_rows: list[list[int]] = []
    for real_group, start, actual_m in zip(real_groups, starts, m_sizes, strict=True):
        aligned_m = _aligned_m(actual_m)
        if compact:
            worklist_rows.append([real_group, start, actual_m, aligned_m])
            continue
        for tile_start in range(0, aligned_m, TCGEN05_DEEPGEMM_SELECTED_SOURCE_M_TILE):
            valid_m = min(
                TCGEN05_DEEPGEMM_SELECTED_SOURCE_M_TILE,
                max(actual_m - tile_start, 0),
            )
            store_m = min(
                TCGEN05_DEEPGEMM_SELECTED_SOURCE_M_TILE,
                aligned_m - tile_start,
            )
            worklist_rows.append([real_group, start + tile_start, valid_m, store_m])
    work_tile_metadata = torch.tensor(
        worklist_rows,
        device=DEVICE,
        dtype=torch.int32,
    )
    return a_packed, b_grouped, work_tile_metadata


def _code_for(
    args: tuple[torch.Tensor, ...] | None = None,
    *,
    kernel: helion.Kernel = _deepgemm_selected_kernel,
    config: helion.Config | None = None,
) -> str:
    if args is None:
        args = _make_selected_args()
    if config is None:
        config = _deepgemm_selected_config()
    kernel.reset()
    bound = kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        return bound.to_triton_code(config)


def _wrapper_plans_from_code(code: str) -> list[dict[str, object]]:
    marker = "._helion_cute_wrapper_plans = "
    line = next(line for line in code.splitlines() if marker in line)
    payload = line.split(marker, 1)[1]
    freeze_prefix = "helion.runtime._freeze_cute_wrapper_plans("
    if payload.startswith(freeze_prefix) and payload.endswith(")"):
        payload = payload[len(freeze_prefix) : -1]
    return list(ast.literal_eval(payload))


_COMPACT_ASPECT_RASTER_SYMBOLS = (
    "tcgen05_grouped_selected_source_n_tiles",
    "tcgen05_grouped_selected_source_m_tiles",
    "tcgen05_grouped_selected_source_m_fast_linear",
)


def _assert_selected_output(
    out: torch.Tensor,
    args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    a_packed, b_grouped, work_tile_metadata = args
    rows = work_tile_metadata.detach().cpu().tolist()
    for real_group, start, valid_m, store_m in rows:
        expected = (
            a_packed[start : start + valid_m].float() @ b_grouped[real_group].float().T
        ).to(out.dtype)
        torch.testing.assert_close(
            out[start : start + valid_m],
            expected,
            rtol=3e-2,
            atol=3e-2,
        )
        padding_start = start + valid_m
        padding_end = start + store_m
        torch.testing.assert_close(
            out[padding_start:padding_end],
            torch.zeros_like(out[padding_start:padding_end]),
            rtol=0,
            atol=0,
        )


def _run_selected_bound(
    kernel: helion.Kernel,
    args: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    kernel.reset()
    bound = kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_deepgemm_selected_config())
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        out = bound(*args)
    torch.cuda.synchronize()
    assert isinstance(out, torch.Tensor)
    return out


def _require_codegen_cuda() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("tcgen05 DeepGEMM selected codegen test needs CUDA fake inputs")


def _require_runtime_cuda13_sm100() -> None:
    _require_codegen_cuda()
    if not requires_cuda_version("13"):
        pytest.skip("tcgen05 DeepGEMM selected runtime test requires CUDA >= 13")
    from helion._compiler.cute.mma_support import get_cute_mma_support

    with torch.cuda.device(DEVICE):
        major, _minor = torch.cuda.get_device_capability(DEVICE)
    if major < 10:
        pytest.skip("tcgen05 requires SM100+")
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")


@pytest.mark.parametrize(("n", "k"), [(224, 64), (128, 64), (256, 128)])
def test_deepgemm_selected_codegen_source_and_wrapper_plan(n: int, k: int) -> None:
    _require_codegen_cuda()

    code = _code_for(_make_selected_args((1, 127, 224, 256), n=n, k=k))

    assert code.count("StaticPersistentGroupTileScheduler.create") == 1
    assert code.count(".group_search_result") == 1
    assert "tcgen05_grouped_selected_sched" in code
    assert "tcgen05_grouped_selected_group_search_result" in code
    assert "cute.arch.alloc_smem(cutlass.Int32, 10, alignment=16)" in code
    assert (
        "tcgen05_grouped_selected_metadata_idx = "
        "tcgen05_grouped_selected_group_search_result.group_idx"
    ) in code
    assert (
        "tcgen05_sched_pipeline_consumer_group = "
        "cutlass.pipeline.CooperativeGroup("
        "cutlass.pipeline.Agent.Thread, cutlass.Int32(6))"
    ) in code
    assert "consumer_mask=cutlass.Int32(0)" not in code
    assert "_cute_store_shared_remote_x4" not in code
    assert "clc_response" not in code
    assert (
        "tcgen05_grouped_selected_cta_tile_idx_m = "
        "tcgen05_grouped_selected_group_search_result.cta_tile_idx_n"
    ) in code
    assert (
        "tcgen05_grouped_selected_cta_tile_idx_n = "
        "tcgen05_grouped_selected_group_search_result.cta_tile_idx_m // "
        "cutlass.Int32(2)"
    ) in code
    assert (
        "tcgen05_grouped_selected_source_n_tiles = "
        "(tcgen05_grouped_selected_group_search_result.problem_shape_m + "
        "cutlass.Int32(256) - 1) // cutlass.Int32(256)"
    ) in code
    assert (
        "tcgen05_grouped_selected_source_m_tiles = "
        "tcgen05_grouped_selected_group_search_result.problem_shape_n // "
        "cutlass.Int32(224)"
    ) in code
    assert (
        "if tcgen05_grouped_selected_source_m_tiles <= "
        "tcgen05_grouped_selected_source_n_tiles:"
    ) in code
    assert (
        "tcgen05_grouped_selected_source_m_fast_linear = "
        "tcgen05_grouped_selected_cta_tile_idx_m * "
        "tcgen05_grouped_selected_source_n_tiles + "
        "tcgen05_grouped_selected_cta_tile_idx_n"
    ) in code
    assert (
        "tcgen05_grouped_selected_cta_tile_idx_m = "
        "tcgen05_grouped_selected_source_m_fast_linear % "
        "tcgen05_grouped_selected_source_m_tiles"
    ) in code
    assert (
        "tcgen05_grouped_selected_cta_tile_idx_n = "
        "tcgen05_grouped_selected_source_m_fast_linear // "
        "tcgen05_grouped_selected_source_m_tiles"
    ) in code
    assert (
        "tcgen05_grouped_selected_problem_m = "
        "tcgen05_grouped_selected_group_search_result.problem_shape_n"
    ) in code
    assert (
        "tcgen05_grouped_selected_problem_n = "
        "tcgen05_grouped_selected_group_search_result.problem_shape_m"
    ) in code
    assert (
        "tcgen05_grouped_cta_tile_count_k = tcgen05_work_tile_smem[cutlass.Int32(2)]"
    ) in code
    assert "TensorMapManager" in code
    assert "update_tensormap" in code
    assert (
        "tcgen05_grouped_tensormap_group_changed = "
        "tcgen05_grouped_metadata_idx != tcgen05_grouped_tensormap_last_group"
    ) in code
    assert "tcgen05_grouped_tensormap_last_group = tcgen05_grouped_metadata_idx" in code
    assert code.count("tcgen05_grouped_d_tensormap_last_group = cutlass.Int32(-1)") == 1
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_a_desc_ptr" in code
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_b_desc_ptr" in code
    assert "tcgen05_grouped_d_tensormap_manager" in code
    assert "cute.nvgpu.tcgen05.CtaGroup.TWO" in code
    assert "CtaGroup.ONE" not in code
    assert "SwapABCompactPrefixKernel" not in code
    assert "grouped_deepgemm_swap_ab" not in code
    assert "_cutlass_grouped_gemm_kernel" not in code
    assert "GroupedGemmKernel" not in code
    assert "deepgemm_selected_markers" not in code
    assert "virtual_pid" not in code
    assert "epilogue_tmem_copy_and_partition" not in code
    assert "epilogue_smem_copy_and_partition" not in code
    assert "CopyUniversalOp" not in code
    assert "Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x4)" in code
    assert "StMatrix8x8x16bOp(transpose=True, num_matrices=4)" in code
    assert "tcgen05_epi_tile = (cute.make_layout(128), cute.make_layout(32))" in code
    assert (
        "tcgen05_store_epi_tile = (cute.make_layout(128), cute.make_layout(32))" in code
    )
    assert (
        "tiled_mma = cutlass.utils.blackwell_helpers.make_trivial_tiled_mma("
        "cutlass.BFloat16, cutlass.BFloat16, cute.nvgpu.OperandMajorMode.K, "
        "cute.nvgpu.OperandMajorMode.K, cutlass.Float32, "
        "cute.nvgpu.tcgen05.CtaGroup.TWO, (256, 224)"
    ) in code
    assert (
        "gA_tma = cute.local_tile(tma_tensor_a, (256, 64), "
        "(tcgen05_grouped_cta_tile_idx_n, None))"
    ) in code
    assert (
        "gB_tma = cute.local_tile(tma_tensor_b, (224, 64), "
        "(tcgen05_grouped_cta_tile_idx_m, None))"
    ) in code
    assert (
        "gA_tma = cute.local_tile(tma_tensor_a, (256, 64), "
        "(tcgen05_grouped_cta_tile_idx_n, None, 0))"
    ) not in code
    assert (
        "gB_tma = cute.local_tile(tma_tensor_b, (224, 64), "
        "(tcgen05_grouped_cta_tile_idx_m, None, 0))"
    ) not in code
    assert "tcgen05_grouped_tensormap_real_a = cute.make_tensor" in code
    assert (
        "cute.make_layout((tcgen05_grouped_problem_n, "
        "tcgen05_grouped_problem_k), "
        "stride=(b_grouped.layout.stride[1], b_grouped.layout.stride[2]))"
    ) in code
    assert "tcgen05_grouped_tensormap_real_b = cute.make_tensor" in code
    assert (
        "cute.make_layout((tcgen05_grouped_problem_m, "
        "tcgen05_grouped_problem_k), "
        "stride=(a_packed.layout.stride[0], a_packed.layout.stride[1]))"
    ) in code
    assert (
        "tcgen05_grouped_problem_n, tcgen05_grouped_problem_k, cutlass.Int32(1)"
    ) not in code
    assert (
        "tcgen05_grouped_problem_m, tcgen05_grouped_problem_k, cutlass.Int32(1)"
    ) not in code
    assert "tcgen05_grouped_d_tensormap_d_nm = cute.make_tensor" in code
    assert "stride=(out.layout.stride[1], out.layout.stride[0]" in code
    assert "cute.local_tile(tcgen05_tma_store_tensor, (256, 224)" in code
    assert "tcgen05_tAcc_epi = cute.flat_divide(tcgen05_tAcc" in code
    assert "tcgen05_tTR_tAcc_nm" in code
    assert "tcgen05_selected_nm_source_m_subtile" not in code
    assert "tcgen05_bSG_gD[None, cutlass.Int32(_tcgen05_subtile)]" in code

    grouped_plan = next(
        plan
        for plan in _wrapper_plans_from_code(code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped_plan["deepgemm_selected"] is True
    assert grouped_plan["deepgemm_selected_compact_metadata"] is True
    assert grouped_plan["worklist_metadata"] is True
    assert grouped_plan["dynamic_ab_tensormaps"] is True
    assert grouped_plan["dynamic_ab_tensormap_rank"] == 2
    assert grouped_plan["dynamic_d_tensormap"] is True
    assert grouped_plan["bm"] == 256
    assert grouped_plan["bn"] == 128
    assert grouped_plan["bk"] == 64
    assert grouped_plan["source_m_tile"] == TCGEN05_DEEPGEMM_SELECTED_SOURCE_M_TILE
    assert grouped_plan["group_count"] == 4
    assert grouped_plan["cluster_m"] == 2
    assert grouped_plan["cluster_n"] == 1
    assert grouped_plan["accumulator_view"] == "nm"
    assert grouped_plan["output_view"] == "nm"
    assert grouped_plan["d_store_view"] == "nm_transposed"
    assert grouped_plan["d_store_layout"] == "cutlass.utils.layout.LayoutEnum.COL_MAJOR"
    assert grouped_plan["mma_bm"] == 256
    assert grouped_plan["mma_bn"] == 224
    assert grouped_plan["source_bm"] == 224
    assert grouped_plan["source_bn"] == 256
    assert grouped_plan["grouped_scheduler_view"] == "nm"
    assert grouped_plan["selected_store_wave"] == "nm_explicit_128x32"
    assert grouped_plan["epi_tile_m"] == 128
    assert grouped_plan["epi_tile_n"] == 32
    assert grouped_plan["d_store_box_n"] == 32
    assert grouped_plan["ab_descriptor_view"] == "swapped_nm"
    ab_plan = next(
        plan
        for plan in _wrapper_plans_from_code(code)
        if plan["kind"] == "tcgen05_ab_tma"
    )
    assert ab_plan["bm"] == 256
    assert ab_plan["bn"] == 224
    assert ab_plan["ab_stage_count"] == 7
    assert ab_plan["accumulator_view"] == "nm"
    assert ab_plan["output_view"] == "nm"
    assert ab_plan["source_bm"] == 224
    assert ab_plan["source_bn"] == 256
    assert ab_plan["ab_descriptor_view"] == "swapped_nm"
    assert ab_plan["lhs_rank3_grouped_nt"] is True
    assert ab_plan["dynamic_ab_tensormaps"] is True
    assert ab_plan["dynamic_ab_tensormap_rank"] == 2
    lhs_idx = int(ab_plan["lhs_idx"])
    rhs_idx = int(ab_plan["rhs_idx"])
    wrapper_body: list[str] = []
    wrapper_call_args: list[str] = []
    _append_cute_wrapper_plan(wrapper_body, wrapper_call_args, ab_plan)
    wrapper = "\n".join(wrapper_body)
    assert (
        f"(arg{lhs_idx}_shape1, arg{lhs_idx}_shape2), "
        f"stride=(arg{lhs_idx}_stride1, arg{lhs_idx}_stride2)"
    ) in wrapper
    assert (
        f"(arg{lhs_idx}_shape1, arg{lhs_idx}_shape2, arg{lhs_idx}_shape0)"
        not in wrapper
    )
    assert (
        f"(arg{rhs_idx}_shape0, arg{rhs_idx}_shape1), "
        f"stride=(arg{rhs_idx}_stride0, arg{rhs_idx}_stride1)"
    ) in wrapper
    assert f"(arg{rhs_idx}_shape0, arg{rhs_idx}_shape1, 1)" not in wrapper
    d_plan = next(
        plan
        for plan in _wrapper_plans_from_code(code)
        if plan["kind"] == "tcgen05_d_tma"
    )
    assert d_plan["bm"] == 256
    assert d_plan["bn"] == 224
    assert d_plan["accumulator_view"] == "nm"
    assert d_plan["output_view"] == "nm"
    assert d_plan["d_store_view"] == "nm_transposed"
    assert d_plan["d_store_layout"] == "cutlass.utils.layout.LayoutEnum.COL_MAJOR"
    assert d_plan["selected_store_wave"] == "nm_explicit_128x32"
    assert d_plan["selected_store_accumulator_view"] == "nm"
    assert d_plan["selected_store_d_view"] == "nm_transposed"
    assert d_plan["rank3_mnl_tensor"] is True
    assert d_plan["epi_tile_m"] == 128
    assert d_plan["epi_tile_n"] == 32
    assert d_plan["d_store_box_n"] == 32


def test_deepgemm_selected_incoherent_orientation_request_rejects() -> None:
    _require_codegen_cuda()

    config = _deepgemm_selected_config()
    config.config[TCGEN05_SELECTED_ACCUMULATOR_VIEW_CONFIG_KEY] = "nm"
    config.config[TCGEN05_SELECTED_D_STORE_VIEW_CONFIG_KEY] = "normal"

    with pytest.raises(helion.exc.BackendUnsupported, match="orientation"):
        _code_for(config=config)


@pytest.mark.parametrize(
    ("valid_m", "expected_rows"),
    [
        (1, [[0, 0, 1, 224]]),
        (224, [[0, 0, 224, 224]]),
        (225, [[0, 0, 225, 448]]),
        (256, [[0, 0, 256, 448]]),
    ],
)
def test_deepgemm_selected_worklist_uses_compact_real_group_rows(
    valid_m: int,
    expected_rows: list[list[int]],
) -> None:
    _require_codegen_cuda()

    _a_packed, _b_grouped, work_tile_metadata = _make_selected_args((valid_m,))

    assert work_tile_metadata.detach().cpu().tolist() == expected_rows


def test_deepgemm_selected_noncompact_split_rows_still_work() -> None:
    _require_runtime_cuda13_sm100()

    args = _make_selected_args((225,), n=224, k=64, dirty_padding=True, compact=False)
    assert args[2].detach().cpu().tolist() == [
        [0, 0, 224, 224],
        [0, 224, 1, 224],
    ]
    config = _deepgemm_selected_config(compact=False)
    code = _code_for(args, config=config)
    grouped_plan = next(
        plan
        for plan in _wrapper_plans_from_code(code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped_plan["deepgemm_selected"] is True
    assert "deepgemm_selected_compact_metadata" not in grouped_plan
    assert grouped_plan["group_count"] == 2
    assert "tcgen05_grouped_selected_tile_start" not in code
    for symbol in _COMPACT_ASPECT_RASTER_SYMBOLS:
        assert symbol not in code

    _deepgemm_selected_kernel.reset()
    bound = _deepgemm_selected_kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(config)
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        out = bound(*args)
    torch.cuda.synchronize()

    _assert_selected_output(out, args)


def test_deepgemm_selected_ab7_normalize_gate_is_selected_only() -> None:
    _require_codegen_cuda()

    args = _make_selected_args()
    _deepgemm_selected_kernel.reset()
    with patch_cute_mma_support():
        bound = _deepgemm_selected_kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True

    selected = _deepgemm_selected_config()
    bound.env.config_spec.normalize(selected, _fix_invalid=True)
    assert selected.config["tcgen05_ab_stages"] == 7

    unrelated = _deepgemm_selected_config()
    unrelated.config.pop(TCGEN05_DEEPGEMM_SELECTED_CONFIG_KEY)
    with pytest.raises(helion.exc.InvalidConfig):
        bound.env.config_spec.normalize(unrelated, _fix_invalid=True)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("fp16", "bf16_operands"),
        ("k96", "k_multiple_64"),
        ("n196", f"{TCGEN05_DEEPGEMM_SELECTED_CONFIG_KEY}|n_multiple_32"),
    ],
)
def test_deepgemm_selected_rejects_static_envelope_mismatch(
    case: str,
    match: str,
) -> None:
    _require_codegen_cuda()

    if case == "fp16":
        args = _make_selected_args(dtype=torch.float16)
    elif case == "k96":
        args = _make_selected_args(k=96)
    else:
        assert case == "n196"
        args = _make_selected_args(n=196)

    with pytest.raises(helion.exc.BackendUnsupported, match=match):
        _code_for(args)


def test_deepgemm_selected_rejects_noncontiguous_a() -> None:
    _require_codegen_cuda()

    a_packed, b_grouped, work_tile_metadata = _make_selected_args()
    strided_a = torch.empty(
        (a_packed.size(1), a_packed.size(0)),
        device=DEVICE,
        dtype=a_packed.dtype,
    ).T
    strided_a.copy_(a_packed)
    assert strided_a.shape == a_packed.shape
    assert strided_a.stride(1) != 1

    with pytest.raises(helion.exc.BackendUnsupported, match="contiguous_a_packed"):
        _code_for((strided_a, b_grouped, work_tile_metadata))


def test_deepgemm_selected_rejects_noncontiguous_b() -> None:
    _require_codegen_cuda()

    a_packed, b_grouped, work_tile_metadata = _make_selected_args()
    strided_b = torch.empty(
        (b_grouped.size(0), b_grouped.size(2), b_grouped.size(1)),
        device=DEVICE,
        dtype=b_grouped.dtype,
    ).transpose(1, 2)
    strided_b.copy_(b_grouped)
    assert strided_b.shape == b_grouped.shape
    assert strided_b.stride(2) != 1

    with pytest.raises(
        helion.exc.BackendUnsupported,
        match=f"{TCGEN05_DEEPGEMM_SELECTED_CONFIG_KEY}|contiguous_b_grouped",
    ):
        _code_for((a_packed, strided_b, work_tile_metadata))


def test_deepgemm_selected_rejects_non_identity_epilogue() -> None:
    _require_codegen_cuda()

    with pytest.raises(helion.exc.BackendUnsupported, match="identity BF16 store"):
        _code_for(kernel=_deepgemm_selected_bad_epilogue_kernel)


@pytest.mark.parametrize(("n", "k"), [(224, 64), (256, 64), (224, 128)])
@pytest.mark.parametrize(
    ("m_sizes", "real_groups"),
    [
        ((1, 127, 224, 256), None),
        ((64, 32, 96), None),
    ],
)
def test_deepgemm_selected_runtime_correctness_and_padding_zero(
    m_sizes: tuple[int, ...],
    real_groups: tuple[int, ...] | None,
    n: int,
    k: int,
) -> None:
    _require_runtime_cuda13_sm100()

    args = _make_selected_args(
        m_sizes,
        real_groups=real_groups,
        n=n,
        k=k,
        dirty_padding=True,
    )
    _deepgemm_selected_kernel.reset()
    bound = _deepgemm_selected_kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_deepgemm_selected_config())
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        out = bound(*args)
    torch.cuda.synchronize()

    _assert_selected_output(out, args)


def test_deepgemm_selected_runtime_compact_mixed_aspect_n512_dirty_padding() -> None:
    _require_runtime_cuda13_sm100()

    args = _make_selected_args((224, 449, 256), n=512, k=64, dirty_padding=True)
    assert args[2].detach().cpu().tolist() == [
        [0, 0, 224, 224],
        [1, 224, 449, 672],
        [2, 896, 256, 448],
    ]
    out = _run_selected_bound(_deepgemm_selected_kernel, args)

    _assert_selected_output(out, args)


@pytest.mark.parametrize("valid_m", [1, 224, 225, 256])
def test_deepgemm_selected_runtime_valid_m_boundaries(valid_m: int) -> None:
    _require_runtime_cuda13_sm100()

    args = _make_selected_args((valid_m,), n=224, k=64)
    out = _run_selected_bound(_deepgemm_selected_kernel, args)

    _assert_selected_output(out, args)


def test_deepgemm_selected_one_valid_row_orientation_and_output_sentinel() -> None:
    _require_runtime_cuda13_sm100()

    a_packed, b_grouped, work_tile_metadata = _make_selected_args(
        (1,),
        n=224,
        k=64,
    )
    a_packed.zero_()
    b_grouped.zero_()
    a_packed[0, 0] = 1
    expected = torch.arange(
        1,
        b_grouped.size(1) + 1,
        device=DEVICE,
        dtype=torch.float32,
    ).to(b_grouped.dtype)
    b_grouped[0, :, 0] = expected
    sentinel = torch.tensor(-77.0, device=DEVICE, dtype=a_packed.dtype)
    out = torch.full(
        (a_packed.size(0), b_grouped.size(1)),
        sentinel,
        device=DEVICE,
        dtype=a_packed.dtype,
    )

    result = _run_selected_bound(
        _deepgemm_selected_out_kernel,
        (a_packed, b_grouped, work_tile_metadata, out),
    )

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        result[1:],
        torch.zeros_like(result[1:]),
        rtol=0,
        atol=0,
    )


def test_deepgemm_selected_source_m_224_boundary_uses_next_tile() -> None:
    _require_runtime_cuda13_sm100()

    a_packed, b_grouped, work_tile_metadata = _make_selected_args(
        (256,),
        n=224,
        k=64,
    )
    assert work_tile_metadata.detach().cpu().tolist() == [[0, 0, 256, 448]]
    a_packed.zero_()
    b_grouped.zero_()
    group0 = torch.arange(
        1,
        b_grouped.size(1) + 1,
        device=DEVICE,
        dtype=torch.float32,
    ).to(b_grouped.dtype)
    b_grouped[0, :, 0] = group0
    for row, scale in ((0, 2.0), (128, 3.0), (223, 5.0), (224, 7.0), (225, 11.0)):
        a_packed[row, 0] = scale

    result = _run_selected_bound(
        _deepgemm_selected_kernel,
        (a_packed, b_grouped, work_tile_metadata),
    )

    torch.testing.assert_close(result[0], (group0.float() * 2.0).to(result.dtype))
    torch.testing.assert_close(result[128], (group0.float() * 3.0).to(result.dtype))
    torch.testing.assert_close(result[223], (group0.float() * 5.0).to(result.dtype))
    torch.testing.assert_close(result[224], (group0.float() * 7.0).to(result.dtype))
    torch.testing.assert_close(result[225], (group0.float() * 11.0).to(result.dtype))
    torch.testing.assert_close(result[256:448], torch.zeros_like(result[256:448]))


def test_deepgemm_selected_bound_rejects_rank2_dynamic_ab_outer_stride_reuse() -> None:
    _require_runtime_cuda13_sm100()

    args = _make_selected_args((17,), n=224, k=64)
    a_packed, b_grouped, work_tile_metadata = args
    padded_a_storage = torch.empty(
        (a_packed.size(0), a_packed.size(1) + 16),
        device=DEVICE,
        dtype=a_packed.dtype,
    )
    padded_b_storage = torch.empty(
        (b_grouped.size(0), b_grouped.size(1), b_grouped.size(2) + 16),
        device=DEVICE,
        dtype=b_grouped.dtype,
    )
    strided_a = padded_a_storage[:, : a_packed.size(1)]
    strided_b = padded_b_storage[:, :, : b_grouped.size(2)]
    strided_a.copy_(a_packed)
    strided_b.copy_(b_grouped)
    assert strided_a.shape == a_packed.shape
    assert strided_a.stride(1) == 1
    assert strided_a.stride(0) != strided_a.size(1)
    assert strided_b.shape == b_grouped.shape
    assert strided_b.stride(2) == 1
    assert strided_b.stride(1) != strided_b.size(2)
    assert strided_b.stride(0) != strided_b.size(1) * strided_b.size(2)

    _deepgemm_selected_kernel.reset()
    bound = _deepgemm_selected_kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_deepgemm_selected_config())
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        first = bound(*args)
        torch.cuda.synchronize()
        _assert_selected_output(first, args)
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match="rank-2 dynamic A/B TensorMaps require contiguous",
        ):
            bound(strided_a, strided_b, work_tile_metadata)


def test_deepgemm_selected_metadata_mutation_rebuilds_cache() -> None:
    _require_runtime_cuda13_sm100()

    args = _make_selected_args((1, 1), real_groups=(0, 1), n=224, k=64)
    a_packed, b_grouped, work_tile_metadata = args
    a_packed.zero_()
    b_grouped.zero_()
    first_row = 0
    second_row = _aligned_m(1)
    a_packed[first_row, 0] = 1
    a_packed[second_row, 0] = 1
    group0 = torch.arange(
        1,
        b_grouped.size(1) + 1,
        device=DEVICE,
        dtype=torch.float32,
    ).to(b_grouped.dtype)
    group1 = (group0.float() + 1000).to(b_grouped.dtype)
    b_grouped[0, :, 0] = group0
    b_grouped[1, :, 0] = group1

    _deepgemm_selected_kernel.reset()
    bound = _deepgemm_selected_kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_deepgemm_selected_config())
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        first = bound(*args)
        torch.cuda.synchronize()
        work_tile_metadata[:, 0] = torch.tensor(
            [1, 0],
            device=DEVICE,
            dtype=work_tile_metadata.dtype,
        )
        second = bound(*args)
        torch.cuda.synchronize()

    torch.testing.assert_close(first[first_row], group0, rtol=0, atol=0)
    torch.testing.assert_close(first[second_row], group1, rtol=0, atol=0)
    torch.testing.assert_close(second[first_row], group1, rtol=0, atol=0)
    torch.testing.assert_close(second[second_row], group0, rtol=0, atol=0)


def test_deepgemm_selected_metadata_cache_revalidates_rhs_group_count() -> None:
    _require_runtime_cuda13_sm100()

    args = _make_selected_args((1, 1), real_groups=(0, 1), n=224, k=64)
    _deepgemm_selected_kernel.reset()
    bound = _deepgemm_selected_kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_deepgemm_selected_config())
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        first = bound(*args)
        torch.cuda.synchronize()
        bad_b_grouped = torch.empty(
            (1, args[1].size(1), args[1].size(2)),
            device=DEVICE,
            dtype=args[1].dtype,
        )
        bad_args = (args[0], bad_b_grouped, args[2])
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match="outside B_grouped",
        ):
            bound(*bad_args)

    _assert_selected_output(first, args)


def test_deepgemm_selected_graph_replay_after_warmup() -> None:
    _require_runtime_cuda13_sm100()

    args = _make_selected_args((1, 127, 224, 256), n=224, k=128)
    _deepgemm_selected_kernel.reset()
    bound = _deepgemm_selected_kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_deepgemm_selected_config())
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        warmup = bound(*args)
        torch.cuda.synchronize()
        _assert_selected_output(warmup, args)

        args[0].normal_()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        captured: list[torch.Tensor] = []
        with torch.cuda.graph(graph):
            captured.append(bound(*args))
        torch.cuda.synchronize()

        captured[0].fill_(-7.0)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

    _assert_selected_output(captured[0], args)


def test_deepgemm_selected_runtime_rejects_invalid_compact_aligned_m() -> None:
    _require_runtime_cuda13_sm100()

    args_list = list(_make_selected_args((17, 11)))
    bad_metadata = args_list[2].clone()
    bad_metadata[0, 3] = 1
    args_list[2] = bad_metadata
    args = tuple(args_list)
    _deepgemm_selected_kernel.reset()
    bound = _deepgemm_selected_kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_deepgemm_selected_config())
    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        pytest.raises(helion.exc.BackendUnsupported, match="aligned_m"),
    ):
        bound(*args)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("row_hole", "row holes"),
        ("repeated_groups", "unique real group ids"),
        ("non_dense_groups", "dense real group ids"),
        ("start_misaligned", "starts aligned"),
        ("actual_zero", "actual_m"),
        ("actual_gt_aligned", "actual_m"),
        ("aligned_not_224", "aligned_m"),
    ],
)
def test_deepgemm_selected_runtime_rejects_bad_worklist_contract(
    case: str,
    match: str,
) -> None:
    _require_runtime_cuda13_sm100()

    if case == "row_hole":
        args = list(_make_selected_args((127, 224)))
        a_packed, _b_grouped, work_tile_metadata = args
        padded_a = torch.zeros(
            (a_packed.size(0) + 256, a_packed.size(1)),
            device=DEVICE,
            dtype=a_packed.dtype,
        )
        padded_a[:127].copy_(a_packed[:127])
        padded_a[448 : 448 + 224].copy_(a_packed[224 : 224 + 224])
        bad_metadata = work_tile_metadata.clone()
        bad_metadata[1, 1] = 448
        args[0] = padded_a
        args[2] = bad_metadata
    elif case == "repeated_groups":
        args = list(_make_selected_args((127, 224)))
        bad_metadata = args[2].clone()
        bad_metadata[1, 0] = 0
        args[2] = bad_metadata
    elif case == "non_dense_groups":
        args = list(_make_selected_args((127, 224), real_groups=(0, 2)))
    elif case == "start_misaligned":
        args = list(_make_selected_args((224,)))
        args[2] = torch.tensor(
            [[0, 128, 96, 224]],
            device=DEVICE,
            dtype=torch.int32,
        )
    elif case == "actual_zero":
        args = list(_make_selected_args((224,)))
        args[2] = torch.tensor(
            [[0, 0, 0, 224]],
            device=DEVICE,
            dtype=torch.int32,
        )
    elif case == "actual_gt_aligned":
        args = list(_make_selected_args((224,)))
        args[2] = torch.tensor(
            [[0, 0, 225, 224]],
            device=DEVICE,
            dtype=torch.int32,
        )
    else:
        assert case == "aligned_not_224"
        args = list(_make_selected_args((224,)))
        args[2] = torch.tensor(
            [[0, 0, 128, 256]],
            device=DEVICE,
            dtype=torch.int32,
        )

    selected_args = tuple(args)
    _deepgemm_selected_kernel.reset()
    bound = _deepgemm_selected_kernel.bind(selected_args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_deepgemm_selected_config())
    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        pytest.raises(helion.exc.BackendUnsupported, match=match),
    ):
        bound(*selected_args)
