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

pytestmark = skipUnlessBackends(["cute"])
if matchesBackends(["cute"]):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")


def _selected_config(
    block_k: int = 128,
    *,
    grouped_mode: bool = True,
    source_m_tile: int | None = None,
) -> helion.Config:
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
    if grouped_mode:
        config.config[TCGEN05_GROUPED_MODE_CONFIG_KEY] = (
            TCGEN05_GROUPED_MODE_WORKLIST_NM
        )
        config.config[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY] = (
            source_m_tile
            if source_m_tile is not None
            else TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
            if block_k == 128
            else TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT
        )
    return config


@helion.kernel(backend="cute", static_shapes=True)
def _device_split_sizes_kernel(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    split_sizes: torch.Tensor,
) -> torch.Tensor:
    m_total, k = a_packed.shape
    groups, n, k2 = b_grouped.shape
    assert k == k2
    assert groups == 8
    assert split_sizes.size(0) == 8
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty(
        m_total,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for group_tile, tile_m, tile_n in hl.tile(
        [groups, m_total, n],
        block_size=[1, block_m, block_n],
    ):
        group = group_tile.index.sum()
        group_m = split_sizes[group]
        s0 = split_sizes[0]
        s1 = split_sizes[1]
        s2 = split_sizes[2]
        s3 = split_sizes[3]
        s4 = split_sizes[4]
        s5 = split_sizes[5]
        s6 = split_sizes[6]
        # Prefix addition is commutative; use a deliberately noncanonical order
        # to ensure the semantic matcher does not depend on source ordering.
        group_start = (
            torch.where(group > 4, s4, 0)
            + torch.where(group > 0, s0, 0)
            + torch.where(group > 6, s6, 0)
            + torch.where(group > 1, s1, 0)
            + torch.where(group > 5, s5, 0)
            + torch.where(group > 2, s2, 0)
            + torch.where(group > 3, s3, 0)
        )
        local_m = tile_m.index
        row_index = group_start + local_m
        valid_rows = local_m < group_m
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
                b_grouped[group, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _device_split_sizes_two_groups_kernel(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    split_sizes: torch.Tensor,
) -> torch.Tensor:
    m_total, k = a_packed.shape
    groups, n, k2 = b_grouped.shape
    assert k == k2
    assert groups == 2
    assert split_sizes.size(0) == 2
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty(
        m_total,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for group_tile, tile_m, tile_n in hl.tile(
        [groups, m_total, n],
        block_size=[1, block_m, block_n],
    ):
        group = group_tile.index.sum()
        group_m = split_sizes[group]
        s0 = split_sizes[0]
        group_start = torch.where(group > 0, s0, 0)
        local_m = tile_m.index
        row_index = group_start + local_m
        valid_rows = local_m < group_m
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
                b_grouped[group, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _device_split_sizes_near_miss_kernel(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    split_sizes: torch.Tensor,
    variant: hl.constexpr,
) -> torch.Tensor:
    """Packed-shaped kernels whose source semantics must not be rescheduled."""
    m_total, k = a_packed.shape
    groups, n, k2 = b_grouped.shape
    assert k == k2
    assert groups == 8
    assert split_sizes.size(0) == 8
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty(
        m_total,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    grid_groups = groups - 1 if variant == 0 else groups
    grid_m = m_total - 1 if variant == 1 else m_total
    grid_n = n - 32 if variant == 2 else n
    reduction_k = k - 64 if variant == 3 else k
    for group_tile, tile_m, tile_n in hl.tile(
        [grid_groups, grid_m, grid_n],
        block_size=[1, block_m, block_n],
    ):
        group = group_tile.index.sum()
        group_m = split_sizes[group]
        s0 = split_sizes[0]
        s1 = split_sizes[1]
        s2 = split_sizes[2]
        s3 = split_sizes[3]
        if variant == 6:
            s3 = split_sizes[4]
        s4 = split_sizes[4]
        s5 = split_sizes[5]
        s6 = split_sizes[6]
        if variant == 5:
            group_start = (
                torch.add(
                    torch.where(group > 0, s0, 0),
                    torch.where(group > 1, s1, 0),
                    alpha=2,
                )
                + torch.where(group > 2, s2, 0)
                + torch.where(group > 3, s3, 0)
                + torch.where(group > 4, s4, 0)
                + torch.where(group > 5, s5, 0)
                + torch.where(group > 6, s6, 0)
            )
        else:
            group_start = (
                torch.where(group > 0, s0, 0)
                + torch.where(group > 1, s1, 0)
                + torch.where(group > 2, s2, 0)
                + torch.where(group > 3, s3, 0)
                + torch.where(group > 4, s4, 0)
                + torch.where(group > 5, s5, 0)
                + torch.where(group > 6, s6, 0)
            )
        local_m = tile_m.index
        if variant == 4:
            row_index = torch.add(group_start, local_m, alpha=2)
        else:
            row_index = group_start + local_m
        valid_rows = local_m < group_m
        store_rows = valid_rows
        if variant == 7:
            store_rows = local_m < s0
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(reduction_k, block_size=block_k):
            if variant == 9:
                a_blk = hl.load(
                    a_packed,
                    [row_index, tile_k + 1],
                    extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
                )
            else:
                a_blk = hl.load(
                    a_packed,
                    [row_index, tile_k],
                    extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
                )
            if variant == 10:
                b_blk = b_grouped[group, tile_n + 1, tile_k].T
            elif variant == 11:
                b_blk = b_grouped[group, tile_n, tile_k + 1].T
            else:
                b_blk = b_grouped[group, tile_n, tile_k].T
            acc = torch.addmm(
                acc,
                a_blk,
                b_blk,
            )
        if variant == 8:
            acc = acc + group_m.to(torch.float32)
        if variant == 12:
            hl.store(
                out,
                [row_index, tile_n + 1],
                acc.to(out.dtype),
                extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
            )
        else:
            hl.store(
                out,
                [row_index, tile_n],
                acc.to(out.dtype),
                extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
            )
    return out


def _make_device_split_sizes_args(
    m_sizes: tuple[int, ...],
    *,
    n: int = 128,
    k: int = 128,
    split_dtype: torch.dtype = torch.int32,
    split_stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(m_sizes) >= 2
    assert split_stride > 0
    a_packed = torch.randn(
        (sum(m_sizes), k),
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    b_grouped = torch.randn(
        (len(m_sizes), n, k),
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    split_storage = torch.empty(
        len(m_sizes) * split_stride,
        device=DEVICE,
        dtype=split_dtype,
    )
    split_sizes = split_storage[::split_stride]
    split_sizes.copy_(torch.tensor(m_sizes, device=DEVICE, dtype=split_dtype))
    return a_packed, b_grouped, split_sizes


def _configured_device_split_sizes_bound(
    args: tuple[torch.Tensor, ...], block_k: int = 128
):
    _device_split_sizes_kernel.reset()
    bound = _device_split_sizes_kernel.bind(args)
    assert bound.env.config_spec.cute_tcgen05_search_enabled
    bound.set_config(_selected_config(block_k))
    return bound


def _code_for(
    kernel: helion.Kernel[torch.Tensor],
    args: tuple[torch.Tensor, ...],
    config: helion.Config | None = None,
) -> str:
    if config is None:
        config = _selected_config()
    kernel.reset()
    bound = kernel.bind(args)
    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        return bound.to_triton_code(config)


def _wrapper_plans(code: str) -> list[dict[str, object]]:
    marker = "._helion_cute_wrapper_plans = "
    line = next((line for line in code.splitlines() if marker in line), None)
    if line is None:
        return []
    payload = line.split(marker, 1)[1]
    return list(ast.literal_eval(payload))


def _assert_device_split_sizes_output(
    out: torch.Tensor,
    args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    a_packed, b_grouped, split_sizes = args
    start = 0
    for group, group_m in enumerate(split_sizes.cpu().tolist()):
        end = start + group_m
        expected = (a_packed[start:end].float() @ b_grouped[group].float().T).to(
            out.dtype
        )
        torch.testing.assert_close(
            out[start:end],
            expected,
            rtol=3e-2,
            atol=3e-2,
        )
        start = end
    assert start == a_packed.size(0)


def _require_codegen_cuda() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("tcgen05 grouped split-size codegen needs CUDA fake inputs")


def _require_runtime_cuda13_sm100() -> None:
    _require_codegen_cuda()
    if not requires_cuda_version("13"):
        pytest.skip("tcgen05 grouped split-size runtime needs CUDA >= 13")
    from helion._compiler.cute.mma_support import get_cute_mma_support

    with torch.cuda.device(DEVICE):
        major, _minor = torch.cuda.get_device_capability(DEVICE)
    if major < 10:
        pytest.skip("tcgen05 requires SM100+")
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")


def _grouped_plan(code: str) -> dict[str, object]:
    return next(
        plan
        for plan in _wrapper_plans(code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )


def _bk64_device_split_smem_bytes(group_count: int) -> int:
    from helion._compiler.cute.tcgen05_constants import (
        tcgen05_grouped_worklist_smem_bytes,
    )

    return tcgen05_grouped_worklist_smem_bytes(
        group_count=group_count,
        device_split_sizes=True,
        sched_stage_count=1,
        bm=256,
        bn=TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
        bk=64,
        dtype_bytes=2,
        ab_stages=7,
        acc_stages=2,
        c_stages=2,
        cluster_m=2,
    )


def test_grouped_device_split_smem_accounting_includes_fixed_allocations() -> None:
    # Eight groups land exactly on B200/GB300's 227-KiB opt-in limit once
    # mailboxes, TensorMaps, barriers, and alignment are included.
    assert _bk64_device_split_smem_bytes(8) == 227 * 1024
    assert _bk64_device_split_smem_bytes(51) > 227 * 1024


@pytest.mark.parametrize(
    ("block_k", "source_m_tile"),
    (
        (64, TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT),
        (128, TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE),
        (128, TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT),
    ),
)
def test_grouped_device_split_sizes_codegen_and_wrapper_plan(
    block_k: int,
    source_m_tile: int,
) -> None:
    _require_codegen_cuda()

    args = _make_device_split_sizes_args(
        (0, 1, 127, 224, 256, 449, 0, 991),
        n=224,
        k=block_k,
    )
    code = _code_for(
        _device_split_sizes_kernel,
        args,
        _selected_config(block_k, source_m_tile=source_m_tile),
    )

    kernel_def = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Attribute) and decorator.attr == "kernel"
            for decorator in node.decorator_list
        )
    )
    kernel_args = {arg.arg for arg in kernel_def.args.args}
    assert "tcgen05_grouped_problem_sizes" not in kernel_args
    assert "tcgen05_grouped_starts" not in kernel_args
    assert "tcgen05_grouped_real_groups" not in kernel_args

    plan = _grouped_plan(code)
    assert {
        "device_split_sizes": plan["device_split_sizes"],
        "group_count": plan["group_count"],
        "bk": plan["bk"],
        "source_m_tile": plan["source_m_tile"],
        "cluster_m": plan["cluster_m"],
        "cluster_n": plan["cluster_n"],
        "m_size": plan["m_size"],
        "orientation": plan["orientation"],
        "worklist_metadata": plan["worklist_metadata"],
        "dynamic_ab_tensormaps": plan["dynamic_ab_tensormaps"],
        "dynamic_ab_tensormap_rank": plan["dynamic_ab_tensormap_rank"],
        "dynamic_d_tensormap": plan["dynamic_d_tensormap"],
    } == {
        "device_split_sizes": True,
        "group_count": 8,
        "bk": block_k,
        "source_m_tile": source_m_tile,
        "cluster_m": 2,
        "cluster_n": 1,
        "m_size": args[0].size(0),
        "orientation": "nm",
        "worklist_metadata": True,
        "dynamic_ab_tensormaps": True,
        "dynamic_ab_tensormap_rank": 2,
        "dynamic_d_tensormap": True,
    }
    assert "real_groups_arg" not in plan
    assert "iterator + cutlass.Int64(7) * cutlass.Int64(" in code


def test_grouped_device_split_sizes_rejects_over_budget_smem() -> None:
    _require_codegen_cuda()

    from helion._compiler.cute.tcgen05_config import CuteTcgen05Config

    args = _make_device_split_sizes_args(
        (224, 224, 224, 224, 224, 224, 224, 224),
        n=224,
        k=64,
    )
    required = _bk64_device_split_smem_bytes(8)
    with (
        patch.object(
            CuteTcgen05Config,
            "per_cta_smem_capacity_bytes",
            return_value=required - 1,
        ),
        pytest.raises(
            helion.exc.BackendUnsupported,
            match=(
                rf"require {required} bytes of per-CTA SMEM, exceeding the "
                rf"{required - 1}-byte capacity"
            ),
        ),
    ):
        _code_for(_device_split_sizes_kernel, args, _selected_config(64))


def test_grouped_device_split_sizes_two_group_codegen() -> None:
    _require_codegen_cuda()

    args = _make_device_split_sizes_args((511, 1537), n=224)
    code = _code_for(_device_split_sizes_two_groups_kernel, args)

    plan = _grouped_plan(code)
    assert plan["device_split_sizes"] is True
    assert plan["group_count"] == 2
    assert plan["m_size"] == args[0].size(0)
    assert plan["orientation"] == "nm"


def test_grouped_device_split_sizes_without_grouped_mode_falls_back() -> None:
    _require_codegen_cuda()

    args = _make_device_split_sizes_args(
        (32, 64, 96, 128, 160, 192, 224, 256),
        n=512,
    )
    code = _code_for(
        _device_split_sizes_kernel,
        args,
        _selected_config(grouped_mode=False),
    )

    assert not any(
        plan["kind"] == "tcgen05_grouped_static_persistent"
        for plan in _wrapper_plans(code)
    )


@pytest.mark.parametrize(
    ("variant", "reason"),
    (
        (0, "truncated group grid"),
        (1, "truncated M grid"),
        (2, "truncated N grid"),
        (3, "truncated K reduction"),
        (4, "alpha-scaled row add"),
        (5, "alpha-scaled prefix add"),
        (6, "wrong prefix split index"),
        (7, "store mask differs from load mask"),
        (8, "group size has an observable non-scaffold consumer"),
        (9, "shifted A K coordinate"),
        (10, "shifted B N coordinate"),
        (11, "shifted B K coordinate"),
        (12, "shifted output N coordinate"),
    ),
)
def test_grouped_device_split_sizes_rejects_near_misses(
    variant: int,
    reason: str,
) -> None:
    _require_codegen_cuda()

    tensor_args = _make_device_split_sizes_args(
        (32, 64, 96, 128, 160, 192, 224, 256),
        n=512,
    )
    args = (*tensor_args, variant)
    _device_split_sizes_near_miss_kernel.reset()
    bound = _device_split_sizes_near_miss_kernel.bind(args)
    assert not bound.env.config_spec.cute_tcgen05_search_enabled, reason
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
        pytest.raises(
            helion.exc.BackendUnsupported,
            match=(
                "rank3 grouped semantic proof failed|MMA RHS was not grouped rank-3"
            ),
        ),
    ):
        bound.to_triton_code(_selected_config())


@pytest.mark.parametrize(
    ("block_k", "split_dtype", "split_stride"),
    (
        (64, torch.int64, 2),
        (128, torch.int32, 1),
    ),
)
def test_grouped_device_split_sizes_runtime_edges_and_graph_replay(
    block_k: int,
    split_dtype: torch.dtype,
    split_stride: int,
) -> None:
    _require_runtime_cuda13_sm100()

    args = _make_device_split_sizes_args(
        (0, 224, 17, 0, 449, 1, 256, 1101),
        n=224,
        k=2 * block_k,
        split_dtype=split_dtype,
        split_stride=split_stride,
    )
    assert args[2].stride() == (split_stride,)
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        if block_k == 64:
            from helion._compiler.cute.tcgen05_config import CuteTcgen05Config

            # This exact-cap profile is compiled and launched below, tying the
            # accounting-only boundary check to the generated JIT kernel.
            assert _bk64_device_split_smem_bytes(8) == (
                CuteTcgen05Config.per_cta_smem_capacity_bytes(DEVICE)
            )
        bound = _configured_device_split_sizes_bound(args, block_k)
        warmup = bound(*args)
        torch.cuda.synchronize()
        _assert_device_split_sizes_output(warmup, args)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = bound(*args)
        torch.cuda.synchronize()

        new_split_sizes = (113, 0, 224, 200, 0, 1, 300, 1210)
        assert sum(new_split_sizes) == args[0].size(0)
        args[0].normal_()
        args[2].copy_(torch.tensor(new_split_sizes, device=DEVICE, dtype=args[2].dtype))
        captured.fill_(-7.0)
        graph.replay()
        torch.cuda.synchronize()

        _assert_device_split_sizes_output(captured, args)

        # The source grid contains only M rows even if a malformed partition
        # length exceeds M.  The replacement scheduler must preserve that
        # finite extent rather than constructing an out-of-bounds TensorMap.
        oversized_split_sizes = (args[0].size(0) + 1, 0, 0, 0, 0, 0, 0, 0)
        args[0].normal_()
        args[2].copy_(
            torch.tensor(oversized_split_sizes, device=DEVICE, dtype=args[2].dtype)
        )
        captured.fill_(-7.0)
        graph.replay()
        torch.cuda.synchronize()

        expected = (args[0].float() @ args[1][0].float().T).to(captured.dtype)
        torch.testing.assert_close(captured, expected, rtol=3e-2, atol=3e-2)

        # Preserve raw prefix sums while clipping each group's visible interval.
        # Group 2 starts at -50, and its source local-M domain still has only M
        # entries, so it writes the first M-50 output rows and leaves the tail.
        negative_split_sizes = (
            -100,
            50,
            args[0].size(0) + 50,
            0,
            0,
            0,
            0,
            0,
        )
        args[0].normal_()
        args[2].copy_(
            torch.tensor(negative_split_sizes, device=DEVICE, dtype=args[2].dtype)
        )
        captured.fill_(-7.0)
        graph.replay()
        torch.cuda.synchronize()

        visible_rows = args[0].size(0) - 50
        expected = (args[0][:visible_rows].float() @ args[1][2].float().T).to(
            captured.dtype
        )
        torch.testing.assert_close(
            captured[:visible_rows], expected, rtol=3e-2, atol=3e-2
        )
        torch.testing.assert_close(
            captured[visible_rows:],
            torch.full_like(captured[visible_rows:], -7.0),
            rtol=0,
            atol=0,
        )

        if split_dtype is torch.int64:
            # Values outside Int32 must be clipped before the scheduler's
            # Int32 metadata conversion, not narrowed and wrapped.
            huge_first = torch.iinfo(torch.int32).max + 17
            wide_split_sizes = (
                huge_first,
                args[0].size(0) - huge_first,
                0,
                0,
                0,
                0,
                0,
                0,
            )
            args[0].normal_()
            args[2].copy_(
                torch.tensor(wide_split_sizes, device=DEVICE, dtype=args[2].dtype)
            )
            captured.fill_(-7.0)
            graph.replay()
            torch.cuda.synchronize()

            expected = (args[0].float() @ args[1][0].float().T).to(captured.dtype)
            torch.testing.assert_close(captured, expected, rtol=3e-2, atol=3e-2)
