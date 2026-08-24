from __future__ import annotations

import ast
import os
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compat import requires_cuda_version
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_WORKLIST_NM
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
)
from helion._testing import DEVICE
from helion._testing import matchesBackends
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
if matchesBackends(["cute"]):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")


def _aligned_m(
    actual_m: int,
    tile: int = TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
) -> int:
    return ((actual_m + tile - 1) // tile) * tile


def _selected_config(block_k: int = 128) -> helion.Config:
    config = helion.Config(
        block_sizes=[256, 128, block_k],
        l2_groupings=[1],
        loop_orders=[[0, 1, 2]],
        num_stages=7,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages={64: 7, 128: 3}[block_k],
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
    )
    config.config[TCGEN05_GROUPED_MODE_CONFIG_KEY] = TCGEN05_GROUPED_MODE_WORKLIST_NM
    config.config[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY] = (
        TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
        if block_k == 128
        else TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE
    )
    return config


@helion.kernel(backend="cute", static_shapes=False)
def _selected_kernel(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
    row_alpha: hl.constexpr = 1,  # pyrefly: ignore[bad-function-definition]
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
        if row_alpha == 1:
            row_index = global_m_start + local_m
        else:
            row_index = torch.add(global_m_start, local_m, alpha=2)
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


def _make_args(
    m_sizes: tuple[int, ...] = (17, 11),
    *,
    n: int = 128,
    k: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    dirty_padding: bool = False,
    source_m_tile: int = TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    starts: list[int] = []
    cursor = 0
    for actual_m in m_sizes:
        starts.append(cursor)
        cursor += _aligned_m(actual_m, source_m_tile)

    a_packed = torch.zeros((cursor, k), device=DEVICE, dtype=dtype)
    for start, actual_m in zip(starts, m_sizes, strict=True):
        a_packed[start : start + actual_m].normal_()
        if dirty_padding:
            a_packed[
                start + actual_m : start + _aligned_m(actual_m, source_m_tile)
            ].normal_()
    b_grouped = torch.randn((len(m_sizes), n, k), device=DEVICE, dtype=dtype)
    work_tile_metadata = torch.tensor(
        [
            [group, start, actual_m, _aligned_m(actual_m, source_m_tile)]
            for group, (start, actual_m) in enumerate(zip(starts, m_sizes, strict=True))
        ],
        device=DEVICE,
        dtype=torch.int32,
    )
    return a_packed, b_grouped, work_tile_metadata


def _configured_bound(args: tuple[torch.Tensor, ...], block_k: int = 128):
    _selected_kernel.reset()
    bound = _selected_kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_selected_config(block_k))
    return bound


def _code_for(
    args: tuple[torch.Tensor, ...],
    config: helion.Config | None = None,
) -> str:
    if config is None:
        config = _selected_config()
    _selected_kernel.reset()
    bound = _selected_kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        return bound.to_triton_code(config)


def _wrapper_plans(code: str) -> list[dict[str, object]]:
    marker = "._helion_cute_wrapper_plans = "
    payload = next(line for line in code.splitlines() if marker in line).split(
        marker, 1
    )[1]
    return list(ast.literal_eval(payload))


def _assert_output(
    out: torch.Tensor,
    args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    a_packed, b_grouped, work_tile_metadata = args
    for group, start, valid_m, store_m in work_tile_metadata.cpu().tolist():
        expected = (
            a_packed[start : start + valid_m].float() @ b_grouped[group].float().T
        ).to(out.dtype)
        torch.testing.assert_close(
            out[start : start + valid_m],
            expected,
            rtol=3e-2,
            atol=3e-2,
        )
        torch.testing.assert_close(
            out[start + valid_m : start + store_m],
            torch.zeros_like(out[start + valid_m : start + store_m]),
            rtol=0,
            atol=0,
        )


def _require_codegen_cuda() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("tcgen05 selected-path codegen needs CUDA fake inputs")


def _require_runtime_cuda13_sm100() -> None:
    _require_codegen_cuda()
    if not requires_cuda_version("13"):
        pytest.skip("tcgen05 selected-path runtime needs CUDA >= 13")
    from helion._compiler.cute.mma_support import get_cute_mma_support

    with torch.cuda.device(DEVICE):
        major, _minor = torch.cuda.get_device_capability(DEVICE)
    if major < 10:
        pytest.skip("tcgen05 requires SM100+")
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")


def test_grouped_worklist_nm_codegen_and_wrapper_plan() -> None:
    _require_codegen_cuda()

    code = _code_for(_make_args((1, 127, 224, 256), n=224, k=128))

    assert code.count("StaticPersistentGroupTileScheduler.create") == 1
    assert "TensorMapManager" in code
    assert "update_tensormap" in code
    assert "cute.nvgpu.tcgen05.CtaGroup.TWO" in code
    assert "(256, 256)" in code
    assert "cute.local_tile(tma_tensor_a, (256, 128)" in code
    assert "cute.local_tile(tma_tensor_b, (256, 128)" in code
    assert "cute.local_tile(tcgen05_tma_store_tensor, (256, 256)," in code
    assert "StMatrix8x8x16bOp(transpose=True, num_matrices=4)" in code
    assert "cute.arch.alloc_smem(cutlass.Int32, 9" in code

    plan = next(
        plan
        for plan in _wrapper_plans(code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert {
        "orientation": plan["orientation"],
        "worklist_metadata": plan["worklist_metadata"],
        "dynamic_ab_tensormaps": plan["dynamic_ab_tensormaps"],
        "dynamic_d_tensormap": plan["dynamic_d_tensormap"],
    } == {
        "orientation": "nm",
        "worklist_metadata": True,
        "dynamic_ab_tensormaps": True,
        "dynamic_d_tensormap": True,
    }
    d_plan = next(
        plan for plan in _wrapper_plans(code) if plan["kind"] == "tcgen05_d_tma"
    )
    assert (d_plan["bm"], d_plan["bn"], d_plan["orientation"]) == (256, 256, "nm")


def test_grouped_worklist_nm_legacy_bk64_codegen() -> None:
    _require_codegen_cuda()

    config = _selected_config(64)
    config.config.pop(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY)
    code = _code_for(
        _make_args(
            (1, 127, 224, 256),
            n=224,
            k=64,
            source_m_tile=TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE,
        ),
        config,
    )

    assert "(256, 224, 64)" in code
    assert "cute.local_tile(tma_tensor_a, (256, 64)" in code
    assert "cute.local_tile(tma_tensor_b, (224, 64)" in code
    assert "cute.local_tile(tcgen05_tma_store_tensor, (256, 224)," in code
    plan = next(
        plan
        for plan in _wrapper_plans(code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert plan["bk"] == 64
    assert plan["source_m_tile"] == TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE
    d_plan = next(
        plan for plan in _wrapper_plans(code) if plan["kind"] == "tcgen05_d_tma"
    )
    assert (d_plan["bm"], d_plan["bn"], d_plan["orientation"]) == (256, 224, "nm")


def test_grouped_worklist_nm_rejects_over_budget_host_metadata_profile() -> None:
    _require_codegen_cuda()

    config = _selected_config(64)
    config.config[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY] = (
        TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
    )
    args = _make_args(
        (224, 256),
        n=224,
        k=64,
        source_m_tile=TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
    )
    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="grouped N,M worklist generated allocations require",
    ):
        _code_for(args, config)


def test_grouped_worklist_nm_rejects_alpha_scaled_row() -> None:
    _require_codegen_cuda()

    args = (*_make_args((224, 256), n=224, k=128), 2)
    _selected_kernel.reset()
    bound = _selected_kernel.bind(args)
    assert not bound.env.config_spec.cute_tcgen05_search_enabled
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
        pytest.raises(
            helion.exc.BackendUnsupported,
            match="rank3 grouped semantic proof failed",
        ),
    ):
        bound.to_triton_code(_selected_config())


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("fp16", "bf16_operands"),
        ("k96", "k_multiple_block_k"),
        ("n196", f"{TCGEN05_GROUPED_MODE_CONFIG_KEY}|n_multiple_32"),
        ("strided_b", f"{TCGEN05_GROUPED_MODE_CONFIG_KEY}|contiguous_b_grouped"),
    ),
)
def test_grouped_worklist_nm_rejects_ineligible_inputs(case: str, match: str) -> None:
    _require_codegen_cuda()
    if case == "fp16":
        args = _make_args(dtype=torch.float16)
    elif case == "k96":
        args = _make_args(k=96)
    elif case == "n196":
        args = _make_args(n=196)
    else:
        a_packed, b_grouped, work_tile_metadata = _make_args()
        strided_b = torch.empty(
            (b_grouped.size(0), b_grouped.size(2), b_grouped.size(1)),
            device=DEVICE,
            dtype=b_grouped.dtype,
        ).transpose(1, 2)
        strided_b.copy_(b_grouped)
        args = (a_packed, strided_b, work_tile_metadata)

    with pytest.raises(helion.exc.BackendUnsupported, match=match):
        _code_for(args)


@pytest.mark.parametrize(
    ("block_k", "source_m_tile"),
    (
        (64, TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE),
        (128, TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE),
    ),
)
def test_grouped_worklist_nm_runtime_and_graph_replay(
    block_k: int,
    source_m_tile: int,
) -> None:
    _require_runtime_cuda13_sm100()

    m_sizes = (224, 449, 256)
    args = _make_args(
        m_sizes,
        n=512,
        k=2 * block_k,
        dirty_padding=True,
        source_m_tile=source_m_tile,
    )
    expected_metadata = []
    start = 0
    for group, actual_m in enumerate(m_sizes):
        store_m = _aligned_m(actual_m, source_m_tile)
        expected_metadata.append([group, start, actual_m, store_m])
        start += store_m
    assert args[2].cpu().tolist() == expected_metadata
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        bound = _configured_bound(args, block_k)
        warmup = bound(*args)
        torch.cuda.synchronize()
        _assert_output(warmup, args)

        args[0].normal_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = bound(*args)
        torch.cuda.synchronize()

        captured.fill_(-7.0)
        graph.replay()
        torch.cuda.synchronize()

    _assert_output(captured, args)
