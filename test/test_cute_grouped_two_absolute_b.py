from __future__ import annotations

import ast
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_shared_rhs_grouped import _plans
from test.test_cute_shared_rhs_grouped import _target

import helion
from helion._compiler.cute.grouped_full_coverage import (
    full_allocation_b_two_cta_profile_supported,
)
from helion._compiler.cute.grouped_full_coverage import (
    full_allocation_b_two_cta_smem_upper_bound,
)
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Generator

    from helion.runtime.kernel import BoundKernel


KEY = "tcgen05_grouped_full_coverage"
SHAPES = (
    (512, 256, 128, 4),
    (2560, 512, 512, 8),
    (10240, 1024, 1024, 16),
    (768, 512, 384, 5),
    (768, 512, 640, 5),
)


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        _target(),
    ):
        yield
    torch.set_num_threads(previous)


def _bound(
    m: int = 512, n: int = 256, k: int = 128, groups: int = 4
) -> tuple[BoundKernel, tuple[torch.Tensor, ...]]:
    args = (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.arange(groups + 1, dtype=torch.int32) * (m // groups),
    )
    # This profile uses the explicit group/row/column coordinate contract.
    kernel = helion.kernel(
        grouped_gemm_jagged.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="full",
        cute_segmented_matmul_tiling=True,
    )
    return kernel._bind_isolated(args), args


def _config() -> helion.Config:
    return helion.Config(
        block_sizes=[256, 128, 128],
        l2_groupings=[1],
        loop_orders=[[0, 1, 2]],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=3,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_consumer_regs=240,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_source_m_tile=256,
        tcgen05_l2_swizzle_size=1,
    )


def _absolute_b(source: str) -> bool:
    return (
        "cute.local_tile(cute.domain_offset((tcgen05_grouped_global_m_start, "
        "cutlass.Int32(0)), tma_tensor_b), (256, 128)"
    ) in source


@pytest.mark.parametrize("shape", SHAPES)
@skipUnlessBackends(["cute"])
def test_two_ordinary_uses_absolute_b_and_retains_D_and_problem_k(
    shape: tuple[int, ...],
) -> None:
    bound, _args = _bound(*shape)
    source = bound.to_code(_config())
    assert _absolute_b(source)
    assert "init_tensormap_from_atom(tma_atom_b" not in source
    assert "tcgen05_grouped_tensormap_real_b =" not in source
    assert "fence_tensormap_update(tcgen05_grouped_tensormap_b_ptr)" not in source
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_b_desc_ptr" not in source
    assert "tcgen05_grouped_d_tensormap_manager.update_tensormap" in source
    assert "tcgen05_grouped_d_tensormap_manager.fence_tensormap_update" in source
    assert (
        "tcgen05_grouped_problem_k = tcgen05_work_tile_smem[cutlass.Int32(7)]" in source
    )
    assert "tile_offset_3 < tcgen05_grouped_problem_k" in source
    assert "tcgen05_row_union_covered" not in source
    grouped = next(
        p for p in _plans(source) if p["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped["dynamic_ab_tensormaps"] and grouped["dynamic_d_tensormap"]
    assert grouped["cluster_m"] == 2 and grouped["source_m_tile"] == 256
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)]
    assert sum(ast.unparse(n.func) == "cute.arch.alloc_smem" for n in calls) == 13
    assert sum(ast.unparse(n.func) == "cute.gemm" for n in calls) == 1


@pytest.mark.parametrize("shape", SHAPES)
@skipUnlessBackends(["cute"])
def test_correctness_port_does_not_admit_two_dense(shape: tuple[int, ...]) -> None:
    bound, _args = _bound(*shape)
    config = _config()
    config.config[KEY] = "fixed_tma_dense"
    with pytest.raises(InvalidConfig, match="ONE/source32"):
        bound.to_code(config)


@skipUnlessBackends(["cute"])
def test_two_values_fresh_storage_and_offset_stride_do_not_select_descriptor() -> None:
    bound, values = _bound(2560, 512, 512, 8)
    config = _config()
    expected = bound.to_code(config)
    values[2].copy_(torch.tensor([0, 1, 193, 2543, 2560, 2560, 2560, 2560, 2560]))
    backing = torch.empty(18, dtype=torch.int32)
    backing[::2].copy_(values[2])
    arguments = (values[0].clone(), values[1].clone(), backing[::2])
    assert all(
        a.data_ptr() != b.data_ptr() for a, b in zip(arguments, values, strict=True)
    )
    kernel = helion.kernel(
        grouped_gemm_jagged.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="full",
        cute_segmented_matmul_tiling=True,
    )
    other = kernel._bind_isolated(arguments)
    assert other.to_code(config) == expected


@pytest.mark.parametrize(
    "updates",
    (
        {"tcgen05_ab_stages": 2},
        {"tcgen05_ab_stages": 4},
        {"tcgen05_consumer_regs": 256},
        {"tcgen05_acc_stages": 1},
        {"tcgen05_c_stages": 4},
        {"tcgen05_grouped_worklist_source_m_tile": 224},
        {"tcgen05_grouped_runtime_direct": True},
        {"tcgen05_sched_stage_count": 2},
    ),
)
@skipUnlessBackends(["cute"])
def test_other_two_profiles_keep_original_descriptor_or_decline(
    updates: dict[str, object],
) -> None:
    bound, _args = _bound()
    config = _config()
    config.config.update(deepcopy(updates))
    try:
        source = bound.to_code(config)
    except (InvalidConfig, BackendUnsupported):
        return
    assert not _absolute_b(source)
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_b_desc_ptr" in source


@skipUnlessBackends(["cute"])
def test_failed_index_proof_keeps_ordinary_dynamic_b() -> None:
    bound, _args = _bound()
    with patch(
        "helion._compiler.cute.cute_mma._tcgen05_full_allocation_b_index_domain",
        return_value=False,
    ):
        source = bound.to_code(_config())
    assert not _absolute_b(source)
    assert "init_tensormap_from_atom(tma_atom_b" in source


@skipUnlessBackends(["cute"])
def test_failed_ownership_proof_retains_existing_semantic_rejection() -> None:
    bound, _args = _bound()
    with (
        patch(
            "helion._compiler.cute.cute_mma._shared_rhs_region_is_exclusive",
            return_value=False,
        ),
        pytest.raises(BackendUnsupported, match="rank3 grouped semantic proof failed"),
    ):
        bound.to_code(_config())


@pytest.mark.parametrize("capacity_delta", (-1, 0))
@skipUnlessBackends(["cute"])
def test_shared_capacity_uses_conservative_complete_bound(capacity_delta: int) -> None:
    bound, _args = _bound()
    upper = full_allocation_b_two_cta_smem_upper_bound(4)
    with patch.object(
        CuteTcgen05Config,
        "per_cta_smem_capacity_bytes",
        return_value=upper + capacity_delta,
    ):
        source = bound.to_code(_config())
    assert _absolute_b(source) is (capacity_delta == 0)


@pytest.mark.parametrize("offset_dtype", (torch.int32, torch.int64))
@skipUnlessBackends(["cute"])
def test_two_scope_is_typed_without_changing_other_metadata_paths(
    offset_dtype: torch.dtype,
) -> None:
    _old, args = _bound()
    kernel = helion.kernel(
        grouped_gemm_jagged.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="full",
        cute_segmented_matmul_tiling=True,
    )
    bound = kernel._bind_isolated((*args[:2], args[2].to(offset_dtype)))
    assert _absolute_b(bound.to_code(_config())) is (offset_dtype == torch.int32)


@pytest.mark.parametrize("index", range(7))
@pytest.mark.parametrize("kind", ("float", "wrong"))
def test_complete_two_profile_is_typed(index: int, kind: str) -> None:
    values: list[object] = [256, 256, 128, 3, 240, 2, 1]
    assert full_allocation_b_two_cta_profile_supported(*values)
    values[index] = float(values[index]) if kind == "float" else 0
    assert not full_allocation_b_two_cta_profile_supported(*values)


@pytest.mark.parametrize("shape", SHAPES)
@skipUnlessBackends(["cute"])
def test_dense_local_witness_preserves_one_cta_carriers(shape: tuple[int, ...]) -> None:
    bound, _args = _bound(*shape)
    (group,) = (
        candidate
        for candidate in bound.config_spec.compiler_coverage_groups
        if candidate.key == KEY
    )
    assert [witness.value for witness in group.witnesses] == [
        "off",
        "fixed_tma_dense",
        "fixed_tma_dense",
        "fixed_tma_dense_local",
    ]
    for witness in group.witnesses:
        assert witness.carrier["tcgen05_cluster_m"] == 1
        assert witness.carrier["tcgen05_grouped_worklist_source_m_tile"] == 32
