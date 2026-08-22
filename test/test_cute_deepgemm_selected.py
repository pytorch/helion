from __future__ import annotations

import ast
import os
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compat import requires_cuda_version
from helion._compiler.cute.device_state import Tcgen05GroupedSchedulerMode
from helion._compiler.cute.strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_WORKLIST_NM
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_MAILBOX_FIELD_COUNT,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
)
from helion._testing import DEVICE
from helion._testing import matchesBackends
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.cute.launcher import _append_cute_wrapper_plan
from helion.runtime.cute.launcher import _tcgen05_grouped_runtime_nm_tile_records
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_dynamic_ab_tensormaps
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_fixed_tensormaps
from helion.runtime.cute.launcher import (
    _validate_tcgen05_grouped_runtime_direct_clc_grid,
)

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = skipUnlessBackends(["cute"])
if matchesBackends(["cute"]):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")


def _aligned_m(
    actual_m: int,
    tile: int = TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
) -> int:
    return ((actual_m + tile - 1) // tile) * tile


def _selected_config(
    block_k: int = 128,
    source_m_tile: int | None = None,
    *,
    ab_stages: int | None = None,
    cluster_m: int = 2,
    consumer_regs: int | None = None,
    runtime_direct: bool | None = None,
    clc: bool = False,
) -> helion.Config:
    if source_m_tile is None:
        source_m_tile = (
            TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
            if block_k == 128
            else TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
        )
    # BK64/source-256 cannot fit the historical AB7 schedule in CTA SMEM.
    if ab_stages is None:
        ab_stages = {
            (64, TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE): 7,
            (64, TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT): 7,
            (64, TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE): 6,
            (128, TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE): 3,
            (128, TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT): 3,
            (128, TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE): 3,
        }[(block_k, source_m_tile)]
    config = helion.Config(
        block_sizes=[256, 128, block_k],
        l2_groupings=[1],
        loop_orders=[[0, 1, 2]],
        num_stages=7,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=cluster_m,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=ab_stages,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
    )
    config.config[TCGEN05_GROUPED_MODE_CONFIG_KEY] = TCGEN05_GROUPED_MODE_WORKLIST_NM
    config.config[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY] = source_m_tile
    if consumer_regs is not None:
        config.config["tcgen05_consumer_regs"] = consumer_regs
    if runtime_direct is not None:
        config.config[TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY] = runtime_direct
    if clc:
        if runtime_direct is not True:
            raise ValueError("grouped CLC test config requires runtime_direct=True")
        config.config.update(
            {
                "tcgen05_strategy": "role_local_with_scheduler",
                "tcgen05_warp_spec_scheduler_warps": 1,
                "tcgen05_persistence_model": "clc_persistent",
            }
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
    mn_major_b: bool = False,
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
    if mn_major_b:
        b_grouped = b_grouped.transpose(1, 2).contiguous().transpose(1, 2)
    work_tile_metadata = torch.tensor(
        [
            [group, start, actual_m, _aligned_m(actual_m, source_m_tile)]
            for group, (start, actual_m) in enumerate(zip(starts, m_sizes, strict=True))
        ],
        device=DEVICE,
        dtype=torch.int32,
    )
    return a_packed, b_grouped, work_tile_metadata


def _configured_bound(
    args: tuple[torch.Tensor, ...],
    block_k: int = 128,
    *,
    ab_stages: int | None = None,
    source_m_tile: int | None = None,
    cluster_m: int = 2,
    consumer_regs: int | None = None,
    l2_swizzle_size: int | None = None,
    runtime_direct: bool | None = None,
    clc: bool = False,
):
    _selected_kernel.reset()
    bound = _selected_kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    config = _selected_config(
        block_k,
        source_m_tile,
        ab_stages=ab_stages,
        cluster_m=cluster_m,
        consumer_regs=consumer_regs,
        runtime_direct=runtime_direct,
        clc=clc,
    )
    if l2_swizzle_size is not None:
        config.config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = l2_swizzle_size
    bound.set_config(config)
    return bound


def _code_for(
    args: tuple[torch.Tensor, ...],
    config: helion.Config | None = None,
) -> str:
    if config is None:
        config = _selected_config(runtime_direct=True)
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


def _call_count(
    tree: ast.AST,
    receiver: str,
    method: str,
    first_arg: str | None = None,
) -> int:
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == receiver
        and node.func.attr == method
        and (
            first_arg is None
            or (
                bool(node.args)
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == first_arg
            )
        )
        for node in ast.walk(tree)
    )


def _expected_panel_coords(
    *,
    m_tiles: int,
    n_tiles: int,
    configured_panel_size: int,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    panel_size = min(configured_panel_size, n_tiles)
    panel_span = panel_size * m_tiles
    for linear in range(m_tiles * n_tiles):
        panel = linear // panel_span
        within = linear % panel_span
        width = min(panel_size, n_tiles - panel * panel_size)
        tile_m = within // width
        tile_n = panel * panel_size + within % width
        if panel % 2 == 1:
            tile_m = m_tiles - 1 - tile_m
        result.append((tile_m, tile_n))
    return result


def test_grouped_worklist_nm_runtime_tile_table_mapping() -> None:
    rows = [[2, 0, 257, 448], [0, 448, 513, 672], [1, 1120, 65, 224]]
    problem_sizes = [(2560, row[3], 128, 1) for row in rows]
    records = _tcgen05_grouped_runtime_nm_tile_records(
        rows,
        problem_sizes,
        source_tile_m=224,
        source_tile_n=256,
        l2_swizzle_size=8,
    )

    cursor = 0
    for metadata_idx, (real_group, start, actual_m, aligned_m) in enumerate(rows):
        m_tiles = aligned_m // 224
        n_tiles = 10
        count = m_tiles * n_tiles
        group_records = records[cursor : cursor + count]
        assert [(record[0], record[1]) for record in group_records] == (
            _expected_panel_coords(
                m_tiles=m_tiles,
                n_tiles=n_tiles,
                configured_panel_size=8,
            )
        )
        for record in group_records:
            tile_m = record[0]
            assert record[2:8] == (
                metadata_idx,
                real_group,
                aligned_m,
                2560,
                128,
                start,
            )
            assert record[8] == min(224, max(actual_m - tile_m * 224, 0))
            assert record[9] == min(224, aligned_m - tile_m * 224)
        cursor += count
    assert cursor == len(records)


@pytest.mark.parametrize("l2_swizzle_size", (1, 8))
def test_grouped_worklist_nm_tile_table_excludes_zero_tile_groups(
    l2_swizzle_size: int,
) -> None:
    records = _tcgen05_grouped_runtime_nm_tile_records(
        [[0, 0, 0, 0], [1, 0, 33, 224]],
        [(256, 0, 128, 1), (256, 224, 128, 1)],
        source_tile_m=224,
        source_tile_n=256,
        l2_swizzle_size=l2_swizzle_size,
    )

    # The zero-tile row is absent before either raster path performs division
    # or modulo; the surviving row retains its original metadata index.
    assert records.tolist() == [[0, 0, 1, 1, 224, 256, 128, 0, 33, 224]]


@pytest.mark.parametrize(
    "problem_size",
    (
        (128, 64, 64, 2),
        (128, 32, 64, 1),
    ),
)
def test_grouped_worklist_nm_runtime_tile_table_rejects_invalid_problem_shape(
    problem_size: tuple[int, int, int, int],
) -> None:
    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="require batch 1 and problem M matching",
    ):
        _tcgen05_grouped_runtime_nm_tile_records(
            [[0, 0, 33, 64]],
            [problem_size],
            source_tile_m=32,
            source_tile_n=128,
            l2_swizzle_size=1,
        )


def test_grouped_worklist_nm_runtime_direct_clc_checks_grid_z_limit() -> None:
    _validate_tcgen05_grouped_runtime_direct_clc_grid(
        TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS
    )
    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="requires at most 65535 exact tile records",
    ):
        _validate_tcgen05_grouped_runtime_direct_clc_grid(
            TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS + 1
        )


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


def _capture_and_replay(
    bound: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    *,
    poison: float = float("nan"),
) -> torch.Tensor:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(*args)
    torch.cuda.synchronize()

    captured.fill_(poison)
    graph.replay()
    torch.cuda.synchronize()
    return captured


def _run_graph_replay(
    args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    block_k: int = 128,
    *,
    ab_stages: int | None = None,
    source_m_tile: int | None = None,
    cluster_m: int = 2,
    consumer_regs: int | None = None,
    l2_swizzle_size: int | None = None,
    runtime_direct: bool | None = None,
    clc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        bound = _configured_bound(
            args,
            block_k,
            ab_stages=ab_stages,
            source_m_tile=source_m_tile,
            cluster_m=cluster_m,
            consumer_regs=consumer_regs,
            l2_swizzle_size=l2_swizzle_size,
            runtime_direct=runtime_direct,
            clc=clc,
        )
        warmup = bound(*args)
        torch.cuda.synchronize()
        _assert_output(warmup, args)
        warmup = warmup.clone()
        captured = _capture_and_replay(bound, args)

    assert bool(torch.isfinite(captured).all().item())
    _assert_output(captured, args)
    return warmup, captured


def _refresh_worklist_without_recompile(
    bound: Any,
    args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    m_sizes: tuple[int, ...],
    group_ids: tuple[int, ...],
    source_m_tile: int,
) -> None:
    cached_path = bound.get_cached_path()
    compile_cache_size = len(bound._compile_cache)
    metadata = []
    start = 0
    for group, actual_m in zip(group_ids, m_sizes, strict=True):
        store_m = _aligned_m(actual_m, source_m_tile)
        metadata.append([group, start, actual_m, store_m])
        start += store_m
    assert start == args[0].size(0)
    args[0].normal_()
    args[2].copy_(torch.tensor(metadata, device=DEVICE, dtype=torch.int32))
    refreshed = bound(*args)
    torch.cuda.synchronize()
    _assert_output(refreshed, args)
    assert len(bound._compile_cache) == compile_cache_size
    assert bound.get_cached_path() == cached_path


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

    assert "StaticPersistentGroupTileScheduler.create" not in code
    assert "tcgen05_grouped_runtime_tile_records.iterator" in code
    assert "TensorMapManager" in code
    assert "update_tensormap" in code
    assert "cute.nvgpu.tcgen05.CtaGroup.TWO" in code
    assert "cute.local_tile(tma_tensor_a, (256, 128)" in code
    assert "cute.local_tile(tcgen05_tma_tensor_b_tail, (256, 128)" in code

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
        "scheduler_mode": plan["scheduler_mode"],
        "bm": plan["bm"],
    } == {
        "orientation": "nm",
        "worklist_metadata": True,
        "dynamic_ab_tensormaps": True,
        "dynamic_d_tensormap": True,
        "scheduler_mode": Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT.value,
        "bm": 256,
    }


def test_grouped_worklist_nm_runtime_direct_clc_uses_exact_record_ids() -> None:
    _require_codegen_cuda()

    config = _selected_config(
        block_k=64,
        source_m_tile=256,
        ab_stages=6,
        consumer_regs=256,
        runtime_direct=True,
        clc=True,
    )
    config.config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = 8
    code = _code_for(
        _make_args((1, 127, 256, 256), n=512, k=128),
        config,
    )

    assert "_cute_issue_clc_query_nomulticast" in code
    assert "StaticPersistentGroupTileScheduler.create" not in code
    assert "tcgen05_grouped_runtime_tile_records.iterator" in code
    assert (
        "_runtime_direct_linear_idx = tcgen05_work_tile_smem[cutlass.Int32(2)]" in code
    )
    assert (
        "tcgen05_clc_cluster_bidz = cutlass.Int32(cute.arch.block_idx()[2])"
    ) in code
    plans = _wrapper_plans(code)
    grouped_plan = next(
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert (
        grouped_plan["scheduler_mode"] == Tcgen05GroupedSchedulerMode.RUNTIME_CLC.value
    )
    wrapper_body: list[str] = []
    wrapper_call_args: list[str] = []
    _append_cute_wrapper_plan(
        wrapper_body,
        wrapper_call_args,
        grouped_plan,
        num_sm=148,
    )
    assert "    grid_z = tcgen05_grouped_total_clusters" in wrapper_body
    assert not any(
        "StaticPersistentGroupTileScheduler.get_grid_shape" in line
        for line in wrapper_body
    )
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["use_pdl"] is True


def test_grouped_worklist_nm_k_major_b_swizzle_override_is_load_bearing() -> None:
    _require_codegen_cuda()

    config = _selected_config(
        block_k=64,
        source_m_tile=TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE,
        cluster_m=1,
        runtime_direct=True,
    )
    config.config["tcgen05_layout_overrides_smem_swizzle_b"] = 128
    code = _code_for(
        _make_args(
            (1, 17),
            n=256,
            k=128,
            source_m_tile=TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE,
        ),
        config,
    )

    device_b_layouts = [
        line.strip()
        for line in code.splitlines()
        if line.strip().startswith("sB_layout = ")
    ]
    assert device_b_layouts
    assert all("SmemLayoutAtomKind.K_SW128" in line for line in device_b_layouts)
    assert all("order=(1, 2, 3)" in line for line in device_b_layouts)

    ab_plan = next(
        plan for plan in _wrapper_plans(code) if plan["kind"] == "tcgen05_ab_tma"
    )
    assert ab_plan["b_k_major"] is True
    assert ab_plan["smem_swizzle_b"] == 128
    wrapper_body: list[str] = []
    wrapper_call_args: list[str] = []
    _append_cute_wrapper_plan(wrapper_body, wrapper_call_args, ab_plan)
    wrapper_source = "\n".join(wrapper_body)
    assert "SmemLayoutAtomKind.K_SW128" in wrapper_source
    assert "order=(1, 2, 3)" in wrapper_source


def test_grouped_worklist_nm_runtime_direct_clc_rejects_dynamic_tensormaps() -> None:
    _require_codegen_cuda()

    config = _selected_config(
        block_k=64,
        source_m_tile=256,
        ab_stages=6,
        consumer_regs=256,
        runtime_direct=True,
        clc=True,
    )
    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="requires fixed full-allocation TensorMaps",
    ):
        # N=224 is a valid worklist extent, but it is not a whole 256-wide
        # CtaGroup.TWO MMA tile and therefore needs per-CTA dynamic TensorMaps.
        _code_for(_make_args((1, 127, 256), n=224, k=128), config)


def test_grouped_worklist_nm_can_retain_scheduler_mailbox() -> None:
    _require_codegen_cuda()

    code = _code_for(
        _make_args((1, 127, 224, 256), n=224, k=128),
        _selected_config(runtime_direct=False),
    )

    assert "StaticPersistentGroupTileScheduler.create" in code
    assert "tcgen05_grouped_runtime_tile_records.iterator" not in code
    assert "tcgen05_work_tile_smem" in code
    assert (
        "cutlass.Int32(0) < tcgen05_grouped_selected_source_m_tiles <= "
        "tcgen05_grouped_selected_source_n_tiles"
    ) in code
    assert TCGEN05_GROUPED_WORKLIST_MAILBOX_FIELD_COUNT == 9
    assert (
        "cute.arch.alloc_smem(cutlass.Int32, "
        f"{TCGEN05_GROUPED_WORKLIST_MAILBOX_FIELD_COUNT},"
    ) in code
    plan = next(
        plan
        for plan in _wrapper_plans(code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert (
        plan["scheduler_mode"] == Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH.value
    )


@pytest.mark.parametrize(
    ("block_k", "source_m_tile"),
    (
        (64, TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE),
        (64, TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT),
        (64, TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE),
        (128, TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE),
        (128, TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT),
        (128, TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE),
    ),
)
def test_grouped_worklist_nm_runtime_n_tail_descriptors_codegen(
    block_k: int,
    source_m_tile: int,
) -> None:
    _require_codegen_cuda()

    config = _selected_config(block_k, source_m_tile)
    code = _code_for(
        _make_args((1, 257), n=512, k=128, source_m_tile=source_m_tile),
        config,
    )

    assert "cutlass.experimental.primitives.inline_ptx(" in code
    assert (
        "tcgen05_tma_b_peer_delta = mma_slice_tidx * "
        "(tcgen05_tma_runtime_mma_n // cutlass.Int32(2) - "
        f"cutlass.Int32({source_m_tile // 2}))"
    ) in code
    assert (
        "tcgen05_tma_tensor_b_tail = cute.domain_offset("
        "(tcgen05_tma_b_peer_delta, 0), tma_tensor_b)"
    ) in code
    assert (
        f"cute.local_tile(tcgen05_tma_tensor_b_tail, ({source_m_tile}, {block_k})"
        in code
    )
    assert "n_dim=0, m_dim=256" in code
    assert f"if tcgen05_runtime_mma_n == cutlass.Int32({source_m_tile}):" in code
    assert "tcgen05.mma.cta_group::2.kind::f16" in code
    tree = ast.parse(code)
    instr_desc_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith("Tcgen05InstrDesc.build")
    )
    instr_desc_dtypes = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in instr_desc_call.keywords
        if keyword.arg in {"a_dtype", "b_dtype", "c_dtype"}
    }
    assert instr_desc_dtypes == {
        "a_dtype": "cutlass.BFloat16",
        "b_dtype": "cutlass.BFloat16",
        "c_dtype": "cutlass.Float32",
    }
    tail_predicate = f"tcgen05_grouped_valid_m <= cutlass.Int32({source_m_tile - 16})"
    tail_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == tail_predicate
    ]
    assert len(tail_guards) == 2
    tma_guard = next(
        node for node in tail_guards if "tcgen05_tma_b_peer_delta" in ast.unparse(node)
    )
    mma_guard = next(
        node for node in tail_guards if "Tcgen05InstrDesc.build" in ast.unparse(node)
    )
    assert "tcgen05_tma_tensor_b_tail" in ast.unparse(tma_guard)
    assert "Tcgen05InstrDesc.build" not in ast.unparse(tma_guard)
    assert "tcgen05_tma_b_peer_delta" not in ast.unparse(mma_guard)


@pytest.mark.parametrize(
    ("block_k", "source_m_tile"),
    (
        (64, TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT),
        (128, TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE),
    ),
)
def test_grouped_worklist_nm_fixed_full_allocation_tensormaps_codegen(
    block_k: int,
    source_m_tile: int,
) -> None:
    _require_codegen_cuda()

    code = _code_for(
        _make_args((1, 257), n=512, k=128, source_m_tile=source_m_tile),
        _selected_config(block_k, source_m_tile),
    )

    assert "TensorMapManager" not in code
    assert "update_tensormap" not in code

    plans = _wrapper_plans(code)
    grouped_plan = next(
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped_plan["source_m_tile"] == source_m_tile
    assert grouped_plan["fixed_tensormaps"] is True
    assert grouped_plan["dynamic_ab_tensormap_rank"] == 2
    assert "dynamic_ab_tensormaps" not in grouped_plan
    assert "dynamic_d_tensormap" not in grouped_plan

    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["bn"] == source_m_tile
    assert ab_plan["fixed_ab_tensormaps"] is True
    assert "dynamic_ab_tensormaps" not in ab_plan

    d_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_d_tma")
    assert d_plan["fixed_tensormap"] is True
    assert d_plan["rank3_mnl_tensor"] is True


def test_grouped_worklist_nm_partial_m_store_builds_identity_mask() -> None:
    _require_codegen_cuda()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
    code = _code_for(
        _make_args((1, 257), n=512, k=128, source_m_tile=source_m_tile),
        _selected_config(64, source_m_tile),
    )
    tree = ast.parse(code)
    valid_m_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == f"tcgen05_grouped_valid_m == cutlass.Int32({source_m_tile})"
    ]
    assert len(valid_m_guards) == 1
    valid_m_guard = valid_m_guards[0]
    full_body = ast.Module(body=valid_m_guard.body, type_ignores=[])
    tail_body = ast.Module(body=valid_m_guard.orelse, type_ignores=[])
    assert _call_count(full_body, "cute", "make_identity_tensor") == 0
    assert _call_count(full_body, "cute", "where") == 0
    assert _call_count(tail_body, "cute", "make_identity_tensor") == 1
    assert _call_count(tail_body, "cute", "where") == 1


@pytest.mark.parametrize(
    "source_m_tile",
    (
        TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
        TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
    ),
)
def test_grouped_worklist_nm_bk128_ab2_keeps_unsplit_tma_store_codegen(
    source_m_tile: int,
) -> None:
    _require_codegen_cuda()

    config = _selected_config(128, source_m_tile, ab_stages=2)
    code = _code_for(
        _make_args(
            (224, 449, 256),
            n=512,
            k=256,
            source_m_tile=source_m_tile,
        ),
        config,
    )

    assert "while tcgen05_role_local_2_valid:" in code
    assert "tcgen05_role_local_2_full_valid" not in code
    assert "tcgen05_role_local_2_edge_valid" not in code
    assert "tcgen05_edge_src" not in code
    assert "cute.copy(tcgen05_tma_store_atom" in code


def test_grouped_worklist_nm_one_cta_codegen() -> None:
    _require_codegen_cuda()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
    config = _selected_config(64, source_m_tile, cluster_m=1)
    code = _code_for(
        _make_args((24, 23), n=512, k=128, source_m_tile=source_m_tile),
        config,
    )

    assert "cute.nvgpu.tcgen05.CtaGroup.ONE" in code
    assert "cute.nvgpu.tcgen05.CtaGroup.TWO" not in code
    assert "cute.local_tile(tma_tensor_a, (128, 64)" in code
    assert f"cute.local_tile(tma_tensor_b, ({source_m_tile}, 64)" in code

    plans = _wrapper_plans(code)
    grouped_plan = next(
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert "num_sm_multiplier" not in grouped_plan
    assert {
        "bm": grouped_plan["bm"],
        "bn": grouped_plan["bn"],
        "bk": grouped_plan["bk"],
        "cluster_m": grouped_plan["cluster_m"],
        "cluster_n": grouped_plan["cluster_n"],
        "source_m_tile": grouped_plan["source_m_tile"],
        "fixed_tensormaps": grouped_plan["fixed_tensormaps"],
    } == {
        "bm": 128,
        "bn": source_m_tile,
        "bk": 64,
        "cluster_m": 1,
        "cluster_n": 1,
        "source_m_tile": source_m_tile,
        "fixed_tensormaps": True,
    }

    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert (ab_plan["bm"], ab_plan["bn"], ab_plan["bk"]) == (
        128,
        source_m_tile,
        64,
    )
    d_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_d_tma")
    assert (d_plan["bm"], d_plan["bn"]) == (128, source_m_tile)


def test_grouped_worklist_nm_one_cta_runtime_direct_uses_physical_n_tile() -> None:
    _require_codegen_cuda()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
    config = _selected_config(
        64,
        source_m_tile,
        cluster_m=1,
        runtime_direct=True,
    )
    code = _code_for(
        _make_args((24, 23, 19, 17, 20, 15), n=4096, k=128),
        config,
    )

    assert "cute.nvgpu.tcgen05.CtaGroup.ONE" in code
    assert "StaticPersistentGroupTileScheduler.create" not in code
    assert "tcgen05_grouped_runtime_tile_records.iterator" in code
    grouped_plan = next(
        plan
        for plan in _wrapper_plans(code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert (
        grouped_plan["scheduler_mode"]
        == Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT.value
    )
    assert grouped_plan["bm"] == 128


@pytest.mark.parametrize(
    ("block_k", "ab_stages", "consumer_regs"),
    ((64, 7, 240), (128, 5, 256)),
)
def test_grouped_worklist_nm_one_cta_fixed_tensormap_mn_major_b_codegen(
    block_k: int,
    ab_stages: int,
    consumer_regs: int,
) -> None:
    _require_codegen_cuda()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
    config = _selected_config(
        block_k,
        source_m_tile,
        ab_stages=ab_stages,
        cluster_m=1,
        consumer_regs=consumer_regs,
    )
    code = _code_for(
        _make_args(
            (24, 23, 19, 17, 20, 18),
            n=512,
            k=2 * block_k,
            mn_major_b=True,
            source_m_tile=source_m_tile,
        ),
        config,
    )

    assert "cute.nvgpu.tcgen05.CtaGroup.ONE" in code
    assert (
        "cutlass.BFloat16, cutlass.BFloat16, "
        "cute.nvgpu.OperandMajorMode.MN, cute.nvgpu.OperandMajorMode.K"
    ) in code
    assert f"cute.local_tile(tma_tensor_a, (128, {block_k}, 1)" in code
    assert "tcgen05_grouped_cta_tile_idx_n, None, tcgen05_grouped_group_idx" in code

    plans = _wrapper_plans(code)
    grouped_plan = next(
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped_plan["orientation"] == "nm"
    assert grouped_plan["fixed_tensormaps"] is True
    assert grouped_plan["bm"] == 128
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert (ab_plan["bm"], ab_plan["bn"], ab_plan["bk"]) == (
        128,
        source_m_tile,
        block_k,
    )
    assert ab_plan["fixed_ab_tensormaps"] is True
    assert ab_plan["fixed_grouped_b_rank3"] is True
    assert ab_plan["a_k_major"] is False
    assert ab_plan["b_k_major"] is True


def test_grouped_worklist_nm_one_cta_runtime_direct_upper_n_record_count() -> None:
    rows = [
        [group, group * 32, actual_m, 32]
        for group, actual_m in enumerate((24, 23, 19, 17, 20, 15))
    ]
    records = _tcgen05_grouped_runtime_nm_tile_records(
        rows,
        [(4096, 32, 2048, 1)] * len(rows),
        source_tile_m=32,
        source_tile_n=128,
        l2_swizzle_size=1,
    )

    assert len(records) == 6 * 32
    for metadata_idx in range(6):
        group_records = records[metadata_idx * 32 : (metadata_idx + 1) * 32]
        assert [record[1] for record in group_records] == list(range(32))
        assert {record[2] for record in group_records} == {metadata_idx}


def test_grouped_worklist_nm_panel_raster_codegen_and_mapping() -> None:
    _require_codegen_cuda()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
    args = _make_args(
        (257, 513),
        n=2560,
        k=128,
        source_m_tile=source_m_tile,
    )
    runtime_panel_config = _selected_config(
        64,
        source_m_tile,
        runtime_direct=True,
    )
    runtime_panel_config.config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = 8
    runtime_panel_code = _code_for(args, runtime_panel_config)
    grouped_plan = next(
        plan
        for plan in _wrapper_plans(runtime_panel_code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert (
        grouped_plan["scheduler_mode"]
        == Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT.value
    )
    panel_l2_swizzle_size = grouped_plan["l2_swizzle_size"]
    assert panel_l2_swizzle_size == 8
    runtime_records = _tcgen05_grouped_runtime_nm_tile_records(
        [[0, 0, 257, 448], [1, 448, 513, 672]],
        [(2560, 448, 128, 1), (2560, 672, 128, 1)],
        source_tile_m=source_m_tile,
        source_tile_n=256,
        l2_swizzle_size=panel_l2_swizzle_size,
    )
    assert runtime_records
    assert [(record[0], record[1]) for record in runtime_records[:20]] == (
        _expected_panel_coords(
            m_tiles=2,
            n_tiles=10,
            configured_panel_size=8,
        )
    )


def test_grouped_worklist_nm_fixed_tensormap_rejects_misaligned_d() -> None:
    _require_codegen_cuda()

    m_total = 224
    n = 256
    k = 128
    a_packed = torch.empty((m_total, k), dtype=torch.bfloat16, device=DEVICE)
    b_grouped = torch.empty((1, n, k), dtype=torch.bfloat16, device=DEVICE)
    aligned_output = torch.empty((m_total, n), dtype=torch.bfloat16, device=DEVICE)
    d_storage = torch.empty(m_total * n + 1, dtype=torch.bfloat16, device=DEVICE)
    output = d_storage[1:].view(m_total, n)
    assert output.data_ptr() % 16 != 0

    plan: dict[str, object] = {
        "fixed_tensormaps": True,
        "orientation": "nm",
        "worklist_metadata": True,
        "dynamic_ab_tensormap_rank": 2,
        "lhs_idx": 0,
        "rhs_idx": 1,
        "n_size": n,
        "k_total_size": k,
        "bm": 256,
        "bk": 64,
    }
    cute_kernel = type("FixedTensorMapKernel", (), {})()
    cute_kernel._helion_cute_wrapper_plans = [
        {"kind": "tcgen05_d_tma", "fixed_tensormap": True, "d_idx": 2}
    ]

    _validate_tcgen05_grouped_fixed_tensormaps(
        cute_kernel,
        plan,
        (a_packed, b_grouped, aligned_output),
    )
    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="16-byte-aligned D base",
    ):
        _validate_tcgen05_grouped_fixed_tensormaps(
            cute_kernel,
            plan,
            (a_packed, b_grouped, output),
        )


def test_grouped_worklist_nm_validator_accepts_exact_mn_major_b_only() -> None:
    _require_codegen_cuda()

    a_packed = torch.empty((32, 64), dtype=torch.bfloat16, device=DEVICE)
    physical_gkn = torch.empty((2, 64, 32), dtype=torch.bfloat16, device=DEVICE)
    mn_major_b = physical_gkn.transpose(1, 2)
    padded_gkn = torch.empty((2, 64, 33), dtype=torch.bfloat16, device=DEVICE)
    padded_mn_major_b = padded_gkn[:, :, :32].transpose(1, 2)
    plan: dict[str, object] = {
        "fixed_ab_tensormaps": True,
        "dynamic_ab_tensormap_rank": 2,
        "lhs_idx": 0,
        "rhs_idx": 1,
    }

    _validate_tcgen05_grouped_dynamic_ab_tensormaps(
        plan,
        (a_packed, mn_major_b),
    )
    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="contiguous K-major or MN-major grouped B",
    ):
        _validate_tcgen05_grouped_dynamic_ab_tensormaps(
            plan,
            (a_packed, padded_mn_major_b),
        )


def test_grouped_worklist_nm_legacy_bk64_codegen() -> None:
    _require_codegen_cuda()

    config = _selected_config(64)
    config.config.pop(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY)
    code = _code_for(
        _make_args(
            (1, 127, 224, 256),
            n=224,
            k=64,
            source_m_tile=TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
        ),
        config,
    )

    assert "(256, 224, 64)" in code
    assert "cute.local_tile(tma_tensor_a, (256, 64)" in code
    assert "tcgen05_tma_tensor_b_tail" not in code
    assert "cute.local_tile(tma_tensor_b, (224, 64)" in code
    plan = next(
        plan
        for plan in _wrapper_plans(code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert plan["bk"] == 64
    assert plan["source_m_tile"] == TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT


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
        padded_b = torch.empty(
            (b_grouped.size(0), b_grouped.size(1), 2 * b_grouped.size(2)),
            device=DEVICE,
            dtype=b_grouped.dtype,
        )
        strided_b = padded_b[:, :, ::2]
        strided_b.copy_(b_grouped)
        args = (a_packed, strided_b, work_tile_metadata)

    with pytest.raises(helion.exc.BackendUnsupported, match=match):
        _code_for(args)


@pytest.mark.parametrize(
    ("block_k", "source_m_tile"),
    (
        (64, TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT),
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
    panel8_shape_profile = (
        block_k == 64
        and source_m_tile == TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
    )
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        bound = _configured_bound(
            args,
            block_k,
            source_m_tile=source_m_tile,
            consumer_regs=256 if panel8_shape_profile else None,
            l2_swizzle_size=8 if panel8_shape_profile else None,
            runtime_direct=True,
        )
        warmup = bound(*args)
        torch.cuda.synchronize()
        _assert_output(warmup, args)
        # Reuse the same bound kernel with different runtime group boundaries
        # and a non-identity metadata-row-to-group mapping.  Total aligned M is
        # unchanged, so this exercises the scheduler metadata refresh rather
        # than recompilation or a new allocation shape.
        mutated_m_sizes = (
            (225, 224, 449)
            if source_m_tile == TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
            else (257, 224, 256)
        )
        mutated_group_ids = (2, 0, 1)
        _refresh_worklist_without_recompile(
            bound,
            args,
            mutated_m_sizes,
            mutated_group_ids,
            source_m_tile,
        )

        args[0].normal_()
        captured = _capture_and_replay(bound, args, poison=-7.0)

    _assert_output(captured, args)


@pytest.mark.parametrize("runtime_direct", (False, True))
def test_grouped_worklist_nm_zero_token_groups_runtime_and_graph_replay(
    runtime_direct: bool,
) -> None:
    _require_runtime_cuda13_sm100()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
    args = _make_args(
        (0, 224, 0, 449),
        n=512,
        k=128,
        dirty_padding=True,
        source_m_tile=source_m_tile,
    )
    args[2][2, 1] = 0
    assert args[2].cpu().tolist() == [
        [0, 0, 0, 0],
        [1, 0, 224, 224],
        [2, 0, 0, 0],
        [3, 224, 449, 672],
    ]
    _run_graph_replay(
        args,
        64,
        source_m_tile=source_m_tile,
        runtime_direct=runtime_direct,
    )


def test_grouped_worklist_nm_zero_valid_tail_runtime_and_graph_replay() -> None:
    _require_runtime_cuda13_sm100()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
    n = 256
    k = 128
    aligned_m = 2 * source_m_tile
    args = (
        torch.randn((aligned_m, k), device=DEVICE, dtype=torch.bfloat16),
        torch.randn((1, n, k), device=DEVICE, dtype=torch.bfloat16),
        torch.tensor(
            [[0, 0, 1, aligned_m]],
            device=DEVICE,
            dtype=torch.int32,
        ),
    )
    records = _tcgen05_grouped_runtime_nm_tile_records(
        args[2].cpu().tolist(),
        [(n, aligned_m, k, 1)],
        source_tile_m=source_m_tile,
        source_tile_n=256,
        l2_swizzle_size=1,
    )
    assert records[:, [0, 8, 9]].tolist() == [
        [0, 1, source_m_tile],
        [1, 0, source_m_tile],
    ]

    config = _selected_config(
        64,
        source_m_tile,
        ab_stages=3,
        runtime_direct=True,
    )
    code = _code_for(args, config)
    assert "max(tcgen05_grouped_valid_m, cutlass.Int32(1))" in code
    assert "tcgen05_runtime_mma_n = cutlass.Int32(0)" not in code
    assert f"tcgen05_tma_runtime_mma_n = cutlass.Int32({source_m_tile})" in code
    _run_graph_replay(
        args,
        64,
        ab_stages=3,
        source_m_tile=source_m_tile,
        runtime_direct=True,
    )


def test_grouped_worklist_nm_nonzero_overlap_is_rejected_at_launch() -> None:
    _require_runtime_cuda13_sm100()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
    args = _make_args(
        (224, 224),
        n=512,
        k=128,
        source_m_tile=source_m_tile,
    )
    args[2][1, 1] = 0
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        bound = _configured_bound(
            args,
            64,
            ab_stages=3,
            source_m_tile=source_m_tile,
            runtime_direct=True,
        )
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match="overlapping A rows",
        ):
            bound(*args)


def test_grouped_worklist_nm_all_empty_is_rejected_at_launch() -> None:
    _require_runtime_cuda13_sm100()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
    args = _make_args(
        (0, 0),
        n=512,
        k=128,
        source_m_tile=source_m_tile,
    )
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        bound = _configured_bound(
            args,
            64,
            source_m_tile=source_m_tile,
            runtime_direct=True,
        )
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match="all-empty worklist",
        ):
            bound(*args)


def test_grouped_worklist_nm_runtime_direct_clc_runtime_and_graph_replay() -> None:
    _require_runtime_cuda13_sm100()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
    m_sizes = (1024, 1024, 1024)
    n = 4096
    args = _make_args(
        m_sizes,
        n=n,
        k=128,
        dirty_padding=True,
        source_m_tile=source_m_tile,
    )
    logical_clusters = sum(
        _aligned_m(actual_m, source_m_tile) // source_m_tile for actual_m in m_sizes
    ) * (n // 256)
    assert (
        logical_clusters
        > torch.cuda.get_device_properties(DEVICE).multi_processor_count // 2
    )
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        bound = _configured_bound(
            args,
            64,
            ab_stages=6,
            source_m_tile=source_m_tile,
            consumer_regs=256,
            l2_swizzle_size=8,
            runtime_direct=True,
            clc=True,
        )
        warmup = bound(*args)
        torch.cuda.synchronize()
        _assert_output(warmup, args)
        mutated_m_sizes = (767, 1024, 1280)
        mutated_group_ids = (2, 0, 1)
        _refresh_worklist_without_recompile(
            bound,
            args,
            mutated_m_sizes,
            mutated_group_ids,
            source_m_tile,
        )

        args[0].normal_()
        captured = _capture_and_replay(bound, args)

    assert bool(torch.isfinite(captured).all().item())
    _assert_output(captured, args)


def test_grouped_worklist_nm_panel_raster_runtime_and_graph_replay() -> None:
    _require_runtime_cuda13_sm100()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
    args = _make_args(
        (257, 513),
        n=2560,
        k=128,
        dirty_padding=True,
        source_m_tile=source_m_tile,
    )
    warmup, captured = _run_graph_replay(
        args,
        64,
        source_m_tile=source_m_tile,
        l2_swizzle_size=8,
        runtime_direct=True,
    )
    torch.testing.assert_close(captured, warmup, rtol=0, atol=0)


def test_grouped_worklist_nm_bk128_ab2_runtime_and_graph_replay() -> None:
    _require_runtime_cuda13_sm100()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
    args = _make_args(
        (224, 449, 256),
        n=512,
        k=256,
        dirty_padding=True,
        source_m_tile=source_m_tile,
    )
    _run_graph_replay(
        args,
        128,
        ab_stages=2,
        source_m_tile=source_m_tile,
    )


def test_grouped_worklist_nm_small_source_public_profile_graph_replay() -> None:
    _require_runtime_cuda13_sm100()

    args = _make_args(
        (24, 23, 19, 17, 20, 18),
        n=4096,
        k=2048,
        dirty_padding=True,
        source_m_tile=TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE,
    )
    assert args[2].cpu().tolist() == [
        [0, 0, 24, 32],
        [1, 32, 23, 32],
        [2, 64, 19, 32],
        [3, 96, 17, 32],
        [4, 128, 20, 32],
        [5, 160, 18, 32],
    ]
    _, captured = _run_graph_replay(
        args,
        64,
        source_m_tile=TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE,
    )
    a_packed, b_grouped, work_tile_metadata = args
    for group, start, valid_m, _store_m in work_tile_metadata.cpu().tolist():
        oracle = a_packed[start : start + valid_m].float() @ b_grouped[group].float().T
        actual = captured[start : start + valid_m].double()
        oracle = oracle.double()
        denominator = (actual.square() + oracle.square()).sum()
        diff = 1 - 2 * (actual * oracle).sum() / denominator
        assert float(diff.item()) <= 1e-5


def test_grouped_worklist_nm_one_cta_mn_major_legacy_graph_replay() -> None:
    _require_runtime_cuda13_sm100()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
    args = _make_args(
        (24, 23, 19, 17, 20, 18),
        n=512,
        k=128,
        dirty_padding=True,
        mn_major_b=True,
        source_m_tile=source_m_tile,
    )
    n, k = args[1].shape[1:]
    assert tuple(args[1].stride()) == (n * k, 1, n)
    _run_graph_replay(
        args,
        64,
        source_m_tile=source_m_tile,
        cluster_m=1,
        consumer_regs=240,
    )


def test_grouped_worklist_nm_one_cta_direct_graph_replay() -> None:
    _require_runtime_cuda13_sm100()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
    # Include a <=16-row group so CTA-group::1 exercises its runtime UMMA-N=16
    # descriptor path in addition to the full physical N=32 path.
    actual_ms = (24, 23, 19, 17, 20, 15)
    args = _make_args(
        actual_ms,
        n=4096,
        k=2048,
        dirty_padding=True,
        source_m_tile=source_m_tile,
    )
    expected_metadata = []
    start = 0
    for group, actual_m in enumerate(actual_ms):
        expected_metadata.append([group, start, actual_m, source_m_tile])
        start += source_m_tile
    assert args[2].cpu().tolist() == expected_metadata
    _run_graph_replay(
        args,
        64,
        source_m_tile=source_m_tile,
        cluster_m=1,
        runtime_direct=True,
    )


@pytest.mark.parametrize(
    ("block_k", "ab_stages", "consumer_regs", "mn_major_b"),
    (
        pytest.param(128, 5, 256, False, id="k-major-bk128"),
        pytest.param(64, 7, 240, True, id="mn-major-bk64"),
    ),
)
def test_grouped_worklist_nm_one_cta_promoted_profiles_graph_replay(
    block_k: int,
    ab_stages: int,
    consumer_regs: int,
    mn_major_b: bool,
) -> None:
    _require_runtime_cuda13_sm100()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
    args = _make_args(
        (24, 23, 19, 17, 20, 18),
        n=512,
        k=2 * block_k,
        dirty_padding=True,
        mn_major_b=mn_major_b,
        source_m_tile=source_m_tile,
    )
    n, k = args[1].shape[1:]
    expected_b_stride = (n * k, 1, n) if mn_major_b else (n * k, k, 1)
    assert tuple(args[1].stride()) == expected_b_stride
    warmup, captured = _run_graph_replay(
        args,
        block_k,
        ab_stages=ab_stages,
        source_m_tile=source_m_tile,
        cluster_m=1,
        consumer_regs=consumer_regs,
        runtime_direct=True,
    )
    torch.testing.assert_close(captured, warmup, rtol=0, atol=0)


def test_grouped_worklist_nm_one_cta_multitile_dynamic_graph_replay() -> None:
    _require_runtime_cuda13_sm100()

    source_m_tile = TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
    args = _make_args(
        (65, 33),
        n=160,
        k=128,
        dirty_padding=True,
        source_m_tile=source_m_tile,
    )
    assert args[2].cpu().tolist() == [
        [0, 0, 65, 96],
        [1, 96, 33, 64],
    ]
    warmup, captured = _run_graph_replay(
        args,
        64,
        source_m_tile=source_m_tile,
        cluster_m=1,
    )
    torch.testing.assert_close(captured, warmup, rtol=0, atol=0)
