from __future__ import annotations

import ast
import os
import re
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from helion._compiler.cute.strategies import Tcgen05PersistenceModel
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_DIRECT
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_DYNAMIC
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_STATIC
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
)
from helion._testing import DEVICE
from helion._testing import matchesBackends
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl
from helion.runtime import _append_cute_wrapper_plan

pytestmark = skipUnlessBackends(["cute"])
if matchesBackends(["cute"]):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")


def _require_cuda(reason: str) -> None:
    if DEVICE.type != "cuda":
        pytest.skip(reason)


def _rank3_rhs_tma_config() -> helion.Config:
    return helion.Config(
        block_sizes=[128, 128, 128],
        l2_groupings=[1],
        loop_orders=[[0, 1]],
        num_stages=2,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
    )


def _grouped_config(*, block_n: int = 128, block_k: int = 128) -> helion.Config:
    config = _rank3_rhs_tma_config()
    config.config["block_sizes"] = [128, block_n, block_k]
    config.config[TCGEN05_GROUPED_MODE_CONFIG_KEY] = TCGEN05_GROUPED_MODE_STATIC
    return config


def _dynamic_bk64_config(*, direct: bool = False) -> helion.Config:
    config = _grouped_config(block_n=64, block_k=64)
    config.config[TCGEN05_GROUPED_MODE_CONFIG_KEY] = (
        TCGEN05_GROUPED_MODE_DIRECT if direct else TCGEN05_GROUPED_MODE_DYNAMIC
    )
    return config


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt(
    a: torch.Tensor, b_grouped: torch.Tensor, layout: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _g, n, _k = b_grouped.size()
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_with_mn_tails(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = layout[tile_m]
        valid_rows = row_group_ids == safe_group_id
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out.dtype),
            out[tile_m, tile_n],
        )
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = layout[tile_m]
        valid_rows = row_group_ids == safe_group_id
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        group_k = k_sizes[safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index < group_k
            a_tile = a[tile_m, tile_k]
            b_tile = b_grouped[safe_group_id, tile_n, tile_k]
            masked_a = torch.where(valid_k[None, :], a_tile, torch.zeros_like(a_tile))
            masked_b = torch.where(valid_k[None, :], b_tile, torch.zeros_like(b_tile))
            acc = torch.addmm(acc, masked_a, masked_b.T)
        out[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out.dtype),
            out[tile_m, tile_n],
        )
    return out


@helion.kernel(backend="cute")
def _bad_k_missing_b_mask(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        group_k = k_sizes[safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index < group_k
            a_tile = a[tile_m, tile_k]
            masked_a = torch.where(valid_k[None, :], a_tile, torch.zeros_like(a_tile))
            acc = torch.addmm(
                acc,
                masked_a,
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _bad_k_arbitrary_mask(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        group_k = k_sizes[safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index <= group_k
            a_tile = a[tile_m, tile_k]
            b_tile = b_grouped[safe_group_id, tile_n, tile_k]
            masked_a = torch.where(valid_k[None, :], a_tile, torch.zeros_like(a_tile))
            masked_b = torch.where(valid_k[None, :], b_tile, torch.zeros_like(b_tile))
            acc = torch.addmm(acc, masked_a, masked_b.T)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _bad_k_without_source_proof(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        group_k = k_sizes[safe_group_id]
        k_noop = (group_k - group_k).to(torch.float32)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = (acc + k_noop).to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _bad_k_group_provenance(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    k_layout: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        k_group_id = k_layout[tile_m.begin]
        k_safe_group_id = torch.where(k_group_id >= 0, k_group_id, 0)
        group_k = k_sizes[k_safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index < group_k
            a_tile = a[tile_m, tile_k]
            b_tile = b_grouped[safe_group_id, tile_n, tile_k]
            masked_a = torch.where(valid_k[None, :], a_tile, torch.zeros_like(a_tile))
            masked_b = torch.where(valid_k[None, :], b_tile, torch.zeros_like(b_tile))
            acc = torch.addmm(acc, masked_a, masked_b.T)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _bad_mn_tail_zero_store(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        valid_rows = layout[tile_m] == safe_group_id
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out.dtype),
            torch.zeros_like(acc).to(out.dtype),
        )
    return out


def _make_full_args(
    *,
    groups: int = 4,
    m_per_group: int = 128,
    n: int = 256,
    k: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = groups * m_per_group
    row_ids = torch.arange(m, device=DEVICE)
    a = torch.zeros((m, k), device=DEVICE, dtype=dtype)
    a[row_ids, row_ids % k] = 1
    group = torch.arange(groups, device=DEVICE, dtype=torch.float32)[:, None, None]
    col = torch.arange(n, device=DEVICE, dtype=torch.float32)[None, :, None]
    kk = torch.arange(k, device=DEVICE, dtype=torch.float32)[None, None, :]
    b_grouped = (group * 37.0 + (col % 17) * 1.25 + (kk % 13) * 0.125).to(dtype)
    layout = torch.arange(groups, device=DEVICE, dtype=torch.int64).repeat_interleave(
        m_per_group
    )
    return a, b_grouped, layout


def _make_mn_tail_args(
    *, k: int = 128, dtype: torch.dtype = torch.float16
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    padded_m = 256
    max_n = 192
    m_sizes = (128, 16)
    a = torch.randn((padded_m, k), device=DEVICE, dtype=dtype)
    b_grouped = torch.randn((2, max_n, k), device=DEVICE, dtype=dtype)
    layout = torch.full((padded_m,), -1, device=DEVICE, dtype=torch.int32)
    for group, m_size in enumerate(m_sizes):
        start = group * 128
        layout[start : start + m_size] = group
    n_sizes = torch.tensor((192, 160), device=DEVICE, dtype=torch.int32)
    out = torch.full((padded_m, max_n), -77.0, device=DEVICE, dtype=dtype)
    return a, b_grouped, layout, n_sizes, out


def _make_k_proof_args() -> tuple[torch.Tensor, ...]:
    a, b_grouped, layout = _make_full_args(
        groups=2,
        n=128,
        k=64,
        dtype=torch.float16,
    )
    k_sizes = torch.tensor((32, 64), device=DEVICE, dtype=torch.int32)
    out = torch.empty((256, 128), device=DEVICE, dtype=torch.float16)
    return a, b_grouped, layout, k_sizes, out


def _make_mismatched_k_group_args() -> tuple[torch.Tensor, ...]:
    a, b_grouped, layout, k_sizes, out = _make_k_proof_args()
    return a, b_grouped, layout, (1 - layout).contiguous(), k_sizes, out


def _make_documented_mixed_k_args(
    *, dtype: torch.dtype = torch.float16
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    k_values = (32, 1536, 16, 16)
    m_values = (128, 16, 128, 16)
    n_values = (64, 128, 64, 64)
    groups = len(k_values)
    padded_m = groups * 128
    max_n = max(n_values)
    max_k = max(k_values)
    a_values = torch.arange(
        padded_m * max_k, device=DEVICE, dtype=torch.float32
    ).reshape(padded_m, max_k)
    b_values = torch.arange(
        groups * max_n * max_k, device=DEVICE, dtype=torch.float32
    ).reshape(groups, max_n, max_k)
    a = (((a_values % 23) - 11) / 257).to(dtype)
    b_grouped = (((b_values % 29) - 14) / 257).to(dtype)
    layout = torch.full((padded_m,), -1, device=DEVICE, dtype=torch.int32)
    for group, (m_size, group_k) in enumerate(zip(m_values, k_values, strict=True)):
        rows = slice(group * 128, group * 128 + m_size)
        layout[rows] = group
        if group_k < max_k:
            a[rows, group_k:] = 3.0 + group
            b_grouped[group, :, group_k:] = -2.0 - group
    n_sizes = torch.tensor(n_values, device=DEVICE, dtype=torch.int32)
    k_sizes = torch.tensor(k_values, device=DEVICE, dtype=torch.int32)
    out = torch.full((padded_m, max_n), -77.0, device=DEVICE, dtype=dtype)
    return a, b_grouped, layout, n_sizes, k_sizes, out


def _make_mixed_non_row_major_args() -> tuple[torch.Tensor, ...]:
    args = list(_make_documented_mixed_k_args())
    a = args[0]
    strided = torch.empty((a.size(1), a.size(0)), device=DEVICE, dtype=a.dtype).T
    strided.copy_(a)
    assert strided.stride(1) != 1
    args[0] = strided
    return tuple(args)


def _make_non_k_contiguous_args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, _b_grouped, layout = _make_full_args()
    b_grouped = torch.empty(
        (4, 128, 256), device=DEVICE, dtype=torch.bfloat16
    ).transpose(1, 2)
    assert b_grouped.stride(2) != 1
    return a, b_grouped, layout


def _assert_mixed_result(
    out: torch.Tensor,
    args: tuple[torch.Tensor, ...],
    *,
    tolerance: float = 4e-2,
) -> None:
    a, b_grouped, layout, n_sizes, k_sizes, _out = args
    sentinel = torch.full_like(out, -77.0)
    for group in range(b_grouped.size(0)):
        rows = torch.nonzero(layout == group, as_tuple=False).flatten()
        group_n = int(n_sizes[group].item())
        group_k = int(k_sizes[group].item())
        expected = (a[rows, :group_k] @ b_grouped[group, :group_n, :group_k].T).to(
            out.dtype
        )
        torch.testing.assert_close(
            out[rows, :group_n], expected, rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(
            out[rows, group_n:], sentinel[rows, group_n:], rtol=0, atol=0
        )
    invalid_rows = torch.nonzero(layout < 0, as_tuple=False).flatten()
    torch.testing.assert_close(
        out[invalid_rows], sentinel[invalid_rows], rtol=0, atol=0
    )


def _assert_grouped_result(
    out: torch.Tensor,
    args: tuple[torch.Tensor, ...],
    *,
    invalid_value: float | None = None,
) -> None:
    a, b_grouped, layout = args[:3]
    expected = torch.empty_like(out)
    if invalid_value is not None:
        expected.fill_(invalid_value)
    n_sizes = args[3] if len(args) > 3 else None
    for group in range(b_grouped.size(0)):
        rows = torch.nonzero(layout == group, as_tuple=False).flatten()
        group_n = int(n_sizes[group]) if n_sizes is not None else out.size(1)
        expected[rows, :group_n] = (
            a[rows].float() @ b_grouped[group, :group_n].float().T
        ).to(out.dtype)
    torch.testing.assert_close(out, expected, rtol=4e-2, atol=4e-2)


def _code_for(
    kernel: Any,
    args: tuple[Any, ...] | None = None,
    config: helion.Config | None = None,
) -> str:
    if args is None:
        args = _make_full_args()
    if config is None:
        config = _rank3_rhs_tma_config()
    bound = kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        return bound.to_triton_code(config)


def _wrapper_plans(code: str) -> list[dict[str, Any]]:
    marker = "._helion_cute_wrapper_plans = "
    line = next(line for line in code.splitlines() if marker in line)
    payload = line.split(marker, 1)[1]
    return list(ast.literal_eval(payload))


def _wrapper_plan(code: str, kind: str) -> dict[str, Any]:
    return next(plan for plan in _wrapper_plans(code) if plan["kind"] == kind)


def _assert_group_scheduler(code: str) -> dict[str, Any]:
    assert "StaticPersistentGroupTileScheduler.create" in code
    assert "StaticPersistentTileScheduler.create" not in code
    assert ".group_search_result" in code
    assert "virtual_pid" not in code
    assert "GroupedGemmKernel" not in code
    assert "tcgen05_rhs_safe_group" not in code
    return _wrapper_plan(code, "tcgen05_grouped_static_persistent")


def _assert_regular_fallback(code: str) -> None:
    assert "'rhs_rank3_grouped_nt': True" not in code
    assert "tcgen05_rhs_safe_group" not in code
    assert "tcgen05_grouped_static_persistent" not in code
    assert "StaticPersistentGroupTileScheduler.create" not in code


def _seed_configs(
    kernel: Any,
    args: tuple[torch.Tensor, ...],
    mode: str,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        bound = kernel.bind(args)
    spec = bound.config_spec
    raw = [
        config.config
        for config in spec.compiler_seed_configs
        if config.config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) == mode
    ]
    normalized = [
        config.config
        for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
        if config.config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) == mode
    ]
    return spec, raw, normalized


def test_rank3_rhs_grouped_nt_codegen_uses_nkg_tma_view() -> None:
    _require_cuda("rank3 RHS B TMA codegen test needs CUDA fake inputs")
    code = _code_for(_rank3_rhs_grouped_nt)

    assert "'rhs_rank3_grouped_nt': True" in code
    assert "StaticPersistentTileScheduler.create" in code
    assert "block=(32, 6, 1)" in code
    assert "tcgen05_rhs_safe_group" in code
    assert "tcgen05_rhs_group = (" in code
    assert ".layout.stride[0])).load()" in code
    assert "cute.slice_(tma_tensor_b" not in code
    assert "tma_tensor_b[tcgen05_rhs_safe_group" not in code
    assert any(
        re.search(
            r"cute\.local_tile\(tma_tensor_b, \(128, 128\), "
            r"\([^,]+ // cutlass\.Int32\(128\), None, tcgen05_rhs_safe_group\)\)",
            line.strip(),
        )
        for line in code.splitlines()
        if "cute.local_tile(tma_tensor_b" in line
    )

    ab_plan = _wrapper_plan(code, "tcgen05_ab_tma")
    assert ab_plan["rhs_rank3_grouped_nt"] is True
    rhs_idx = int(ab_plan["rhs_idx"])
    body: list[str] = []
    call_args: list[str] = []
    _append_cute_wrapper_plan(body, call_args, ab_plan)
    wrapper = "\n".join(body)
    assert f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape2, arg{rhs_idx}_shape0)" in wrapper
    assert (
        f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride2, arg{rhs_idx}_stride0)"
        in wrapper
    )
    assert ".mark_layout_dynamic(" not in wrapper


def test_rank3_rhs_grouped_static_codegen_uses_group_scheduler() -> None:
    _require_cuda("rank3 RHS B TMA codegen test needs CUDA fake inputs")
    code = _code_for(
        _rank3_rhs_grouped_nt,
        _make_full_args(),
        _grouped_config(),
    )

    grouped_plan = _assert_group_scheduler(code)
    assert "cutlass.utils.create_initial_search_state()" in code
    assert ".group_idx" in code
    assert ".cta_tile_idx_m" in code
    assert ".cta_tile_idx_n" in code
    assert ".problem_shape_m" in code
    assert ".problem_shape_n" in code
    assert ".problem_shape_k" in code
    assert "tcgen05_grouped_global_m_start" in code
    assert "tcgen05_grouped_group_idx" in code
    assert ".layout.iterator" not in code
    assert re.search(
        r"cute\.local_tile\(tma_tensor_b, \(128, 128\), "
        r"\([^,]+ // cutlass\.Int32\(128\), None, tcgen05_grouped_group_idx\)\)",
        code,
    )
    assert grouped_plan["group_count"] == 4
    assert grouped_plan["bm"] == 128
    assert grouped_plan["bn"] == 128
    assert grouped_plan["bk"] == 128
    assert grouped_plan["n_size"] == 256
    assert grouped_plan["k_total_size"] == 128
    assert "layout_idx" in grouped_plan
    ab_plan = _wrapper_plan(code, "tcgen05_ab_tma")
    assert ab_plan["rhs_rank3_grouped_nt"] is True
    assert ab_plan.get("dynamic_ab_tensormaps") is not True


def test_rank3_rhs_grouped_static_dynamic_bk64_mixed_tail_direct_codegen() -> None:
    _require_cuda("rank3 RHS B TMA codegen test needs CUDA fake inputs")
    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
        _make_documented_mixed_k_args(),
        _dynamic_bk64_config(direct=True),
    )

    grouped_plan = _assert_group_scheduler(code)
    assert {
        key: grouped_plan[key]
        for key in (
            "group_count",
            "bn",
            "bk",
            "k_total_size",
            "m_tail_preserve",
            "n_tail_preserve",
            "grouped_static_has_m_tail",
            "grouped_static_has_n_tail",
            "dynamic_ab_tensormaps",
            "dynamic_d_tensormap",
            "direct_pointer_metadata",
        )
    } == {
        "group_count": 4,
        "bn": 64,
        "bk": 64,
        "k_total_size": 1536,
        "m_tail_preserve": True,
        "n_tail_preserve": True,
        "grouped_static_has_m_tail": True,
        "grouped_static_has_n_tail": False,
        "dynamic_ab_tensormaps": True,
        "dynamic_d_tensormap": True,
        "direct_pointer_metadata": True,
    }
    for marker in (
        "cutlass.utils.TensorMapManager",
        "update_tensormap",
        "fence_tensormap_update",
        "tma_desc_ptr=tcgen05_grouped_tensormap_a_desc_ptr",
        "tma_desc_ptr=tcgen05_grouped_tensormap_b_desc_ptr",
        "tma_desc_ptr=tcgen05_grouped_d_tensormap_desc_ptr",
        "tcgen05_grouped_direct_pointers",
        "tcgen05_grouped_direct_strides",
        "tile_offset_2 < tcgen05_grouped_problem_k",
    ):
        assert marker in code
    # A, B, the full-tile D descriptor, and the tail D descriptor.
    assert code.count("cute.nvgpu.cpasync.prefetch_descriptor(") == 4
    assert "tcgen05_tiled_copy_r2g" not in code
    assert "tcgen05_store_mask" not in code
    assert _wrapper_plan(code, "tcgen05_ab_tma")["dynamic_ab_tensormaps"] is True
    assert _wrapper_plan(code, "tcgen05_d_tma")["rank3_mnl_tensor"] is True


@pytest.mark.parametrize(
    ("k", "expected_bk", "expected_ab_stages"),
    ((16, 16, 2), (96, 32, 3), (192, 64, 3)),
)
def test_rank3_rhs_grouped_static_mn_tail_seed_and_codegen(
    k: int, expected_bk: int, expected_ab_stages: int
) -> None:
    _require_cuda("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")
    args = _make_mn_tail_args(k=k)
    spec, raw, normalized = _seed_configs(
        _rank3_rhs_grouped_nt_with_mn_tails,
        args,
        TCGEN05_GROUPED_MODE_STATIC,
    )

    assert "cute_tcgen05_grouped_static_common_k" in spec.autotuner_heuristics
    assert "cute_tcgen05_grouped_dynamic_bk64" not in spec.autotuner_heuristics
    assert len(raw) == len(normalized) == 1
    for seed in (raw[0], normalized[0]):
        assert seed["block_sizes"] == [128, 64, expected_bk]
        assert seed[TCGEN05_GROUPED_MODE_CONFIG_KEY] == TCGEN05_GROUPED_MODE_STATIC
        assert seed[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] == (
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value
        )
        assert seed["tcgen05_ab_stages"] == expected_ab_stages

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails,
        args,
        helion.Config.from_dict(normalized[0]),
    )
    grouped_plan = _assert_group_scheduler(code)
    assert grouped_plan["bk"] == expected_bk
    assert grouped_plan["k_total_size"] == k
    assert grouped_plan["m_tail_preserve"] is True
    assert grouped_plan["n_tail_preserve"] is True
    assert grouped_plan["grouped_static_has_m_tail"] is True
    assert grouped_plan["grouped_static_has_n_tail"] is True
    assert grouped_plan.get("dynamic_ab_tensormaps") is not True
    assert _wrapper_plan(code, "tcgen05_ab_tma")["ab_stage_count"] == (
        expected_ab_stages
    )


def test_rank3_rhs_grouped_static_dynamic_bk64_compiler_seed() -> None:
    _require_cuda("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")
    args = _make_documented_mixed_k_args()
    spec, raw, normalized = _seed_configs(
        _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
        args,
        TCGEN05_GROUPED_MODE_DYNAMIC,
    )

    assert "cute_tcgen05_grouped_static_common_k" not in spec.autotuner_heuristics
    assert "cute_tcgen05_grouped_dynamic_bk64" in spec.autotuner_heuristics
    assert len(raw) == len(normalized) == 1
    for seed in (raw[0], normalized[0]):
        assert seed["block_sizes"] == [128, 64, 64]
        assert seed[TCGEN05_GROUPED_MODE_CONFIG_KEY] == TCGEN05_GROUPED_MODE_DYNAMIC
        assert seed[TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY] == 3
        assert seed["tcgen05_ab_stages"] == 4


@pytest.mark.parametrize(
    ("kernel", "args_fn"),
    [
        (_bad_k_missing_b_mask, _make_k_proof_args),
        (_bad_k_arbitrary_mask, _make_k_proof_args),
        (_bad_k_without_source_proof, _make_k_proof_args),
        (_bad_k_group_provenance, _make_mismatched_k_group_args),
        (
            _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
            _make_mixed_non_row_major_args,
        ),
    ],
    ids=(
        "one-sided-k-mask",
        "arbitrary-k-mask",
        "no-source-k-proof",
        "mismatched-k-group",
        "strided-a",
    ),
)
def test_dynamic_bk64_seed_rejects_unsafe_proof(kernel: Any, args_fn: Any) -> None:
    _require_cuda("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")
    spec, raw, normalized = _seed_configs(
        kernel,
        args_fn(),
        TCGEN05_GROUPED_MODE_DYNAMIC,
    )
    assert "cute_tcgen05_grouped_dynamic_bk64" not in spec.autotuner_heuristics
    assert raw == []
    assert normalized == []


@pytest.mark.parametrize(
    "target",
    (
        torch.ops.aten.baddbmm.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.bmm.dtype,
    ),
)
def test_grouped_proof_rejects_unproven_aten_targets(target: Any) -> None:
    from typing import cast

    from helion._compiler.cute.cute_mma import _GroupedMmaAxes
    from helion._compiler.cute.cute_mma import _prove_rank3_rhs_grouped_mma

    graph = torch.fx.Graph()
    node = graph.call_function(target)
    assert (
        _prove_rank3_rhs_grouped_mma(
            cast("Any", None),
            node,
            config={},
            axes=_GroupedMmaAxes(0, 1, 2),
        )
        is None
    )


@pytest.mark.parametrize(
    ("kernel", "args_fn", "config_fn"),
    [
        (_rank3_rhs_grouped_nt, _make_non_k_contiguous_args, _rank3_rhs_tma_config),
        (
            _bad_mn_tail_zero_store,
            _make_mn_tail_args,
            lambda: _grouped_config(block_n=64, block_k=128),
        ),
    ],
    ids=("strided-rhs", "non-preserving-tail-store"),
)
def test_rank3_rhs_unsafe_patterns_use_generic_fallback(
    kernel: Any, args_fn: Any, config_fn: Any
) -> None:
    _require_cuda("rank3 RHS B TMA fallback test needs CUDA fake inputs")
    code = _code_for(kernel, args_fn(), config_fn())
    _assert_regular_fallback(code)
    assert "virtual_pid" in code


@pytest.mark.parametrize(
    ("args_fn", "config_fn", "match"),
    [
        (
            lambda: _make_mn_tail_args(k=64),
            _dynamic_bk64_config,
            "exact_k_sizes",
        ),
        (
            lambda: _make_mn_tail_args(k=48),
            lambda: _grouped_config(block_n=64, block_k=16),
            "common_k_block_pair_allowlisted",
        ),
    ],
    ids=("dynamic-without-k-sizes", "unsupported-common-k-pair"),
)
def test_grouped_static_rejects_unproven_k_contract(
    args_fn: Any, config_fn: Any, match: str
) -> None:
    _require_cuda("rank3 RHS B TMA codegen test needs CUDA fake inputs")
    with pytest.raises(helion.exc.BackendUnsupported, match=match):
        _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails,
            args_fn(),
            config_fn(),
        )


def _require_tcgen05_runtime_test() -> None:
    _require_cuda("rank3 RHS B TMA runtime test needs CUDA")
    from helion._compiler.cute.mma_support import get_cute_mma_support

    with torch.cuda.device(DEVICE):
        major, _minor = torch.cuda.get_device_capability(DEVICE)
    if major < 10:
        pytest.skip("tcgen05 requires SM100+")
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")


def _run_configured(
    kernel: Any, args: tuple[torch.Tensor, ...], config: helion.Config
) -> torch.Tensor:
    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        bound = kernel.bind(args)
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        bound.set_config(config)
        out = bound(*args)
        torch.cuda.synchronize()
    return out


def test_rank3_rhs_grouped_static_native_runtime() -> None:
    _require_tcgen05_runtime_test()
    args = _make_full_args()
    out = _run_configured(_rank3_rhs_grouped_nt, args, _grouped_config())
    _assert_grouped_result(out, args)


def test_rank3_rhs_unsafe_store_fallback_runtime() -> None:
    _require_tcgen05_runtime_test()
    args = _make_mn_tail_args()
    out = _run_configured(
        _bad_mn_tail_zero_store,
        args,
        _grouped_config(block_n=64, block_k=128),
    )
    assert out is args[-1]
    _assert_grouped_result(out, args, invalid_value=0)


@pytest.mark.parametrize("direct", (False, True))
def test_rank3_rhs_grouped_static_dynamic_bk64_runtime(direct: bool) -> None:
    _require_tcgen05_runtime_test()
    args = _make_documented_mixed_k_args()
    out = _run_configured(
        _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
        args,
        _dynamic_bk64_config(direct=direct),
    )
    assert out is args[-1]
    _assert_mixed_result(out, args)
