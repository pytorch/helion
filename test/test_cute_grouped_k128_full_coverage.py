from __future__ import annotations

import ast
from copy import deepcopy
import itertools
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_full_coverage import _dump
from test.test_cute_grouped_full_coverage import _inverse
from test.test_cute_shared_rhs_grouped import _plans
from test.test_cute_shared_rhs_grouped import _target

import helion
from helion._compiler.autotuner_heuristics.cute import grouped_full_coverage_configs
from helion._compiler.cute.grouped_full_coverage import full_coverage_pipeline_supported
from helion._compiler.cute.grouped_full_coverage import full_coverage_smem_upper_bound
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Generator

    from helion.runtime.kernel import BoundKernel


KEY = "tcgen05_grouped_full_coverage"


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
    m: int = 640, n: int = 128, k: int = 128, groups: int = 4, **settings: object
) -> tuple[BoundKernel, tuple[torch.Tensor, ...]]:
    args = (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.arange(groups + 1, dtype=torch.int32) * (m // groups),
    )
    kernel = helion.kernel(
        grouped_gemm_jagged.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="full",
        cute_segmented_matmul_tiling=True,
        **settings,
    )
    return kernel._bind_isolated(args), args


def _configs(bound: BoundKernel, block_k: int = 128) -> list[helion.Config]:
    assert bound.host_function is not None
    return grouped_full_coverage_configs(
        bound.env, bound.host_function.device_ir, block_k=block_k
    )


@pytest.mark.parametrize(
    "block_k,ab_stages,registers",
    tuple(itertools.product((32, 64, 128, 256), (2, 3, 4, 5), (240, 256))),
)
def test_only_complete_typed_pipeline_profiles(
    block_k: int, ab_stages: int, registers: int
) -> None:
    assert full_coverage_pipeline_supported(block_k, ab_stages, registers) is (
        (block_k, ab_stages, registers) in ((64, 2, 240), (128, 4, 256))
    )


@pytest.mark.parametrize("index", (0, 1, 2))
def test_pipeline_profile_rejects_equal_valued_float(index: int) -> None:
    values: list[object] = [128, 4, 256]
    values[index] = float(values[index])
    assert not full_coverage_pipeline_supported(*values)


@pytest.mark.parametrize(
    "m,n,k,groups",
    (
        (640, 128, 128, 4),
        (2560, 512, 512, 8),
        (10240, 1024, 1024, 16),
        (672, 256, 384, 4),
        (672, 256, 640, 4),
    ),
)
@skipUnlessBackends(["cute"])
def test_public_k128_complete_inverse_and_unchanged_allocations(
    m: int, n: int, k: int, groups: int
) -> None:
    bound, args = _bound(m, n, k, groups)
    control, dense = _configs(bound)
    ordinary = bound.to_code(control)
    candidate = bound.to_code(dense)
    inverse, counts = _inverse(candidate)
    assert _dump(inverse) == _dump(ast.parse(ordinary))
    assert counts == {
        "metadata": 1,
        "flag": 1,
        "init": 2,
        "changed": 1,
        "copy": 1,
        "dense": 1,
    }
    assert _plans(ordinary) == _plans(candidate)
    assert full_coverage_smem_upper_bound(groups, 128, 4) < 232448
    allocations = []
    for source in (ordinary, candidate):
        allocations.append(
            [
                _dump(node)
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call)
                and ast.unparse(node.func) == "cute.arch.alloc_smem"
            ]
        )
    assert allocations[0] == allocations[1] and len(allocations[0]) == 13
    assert dense.block_sizes == [256, 128, 128]
    assert dense["tcgen05_ab_stages"] == 4
    assert dense["tcgen05_consumer_regs"] == 256
    assert "setmaxregister_increase(256)" in candidate
    assert "tcgen05_work_tile_smem[cutlass.Int32(7)] =" not in candidate


@pytest.mark.parametrize(
    "updates",
    (
        {"tcgen05_ab_stages": 2},
        {"tcgen05_ab_stages": 5},
        {"tcgen05_consumer_regs": 240},
        {"tcgen05_acc_stages": 1},
        {"tcgen05_c_stages": 4},
        {"tcgen05_cluster_m": 2},
        {"tcgen05_grouped_worklist_source_m_tile": 128},
        {"tcgen05_grouped_runtime_direct": True},
        {"tcgen05_sched_stage_count": 2},
        {"tcgen05_l2_swizzle_size": 2},
        {"tcgen05_ab_producer_advance_mode": "skip"},
    ),
)
@pytest.mark.parametrize("mode", ["fixed_tma_dense", "fixed_tma_dense_local"])
@skipUnlessBackends(["cute"])
def test_k128_incomplete_or_other_schedule_is_not_admitted(
    updates: dict[str, object],
    mode: str,
) -> None:
    bound, args = _bound()
    _, config = _configs(bound)
    config.config[KEY] = mode
    config.config.update(deepcopy(updates))
    with pytest.raises((InvalidConfig, BackendUnsupported)):
        bound.to_code(config)


@pytest.mark.parametrize("capacity_delta", (-1, 0))
@skipUnlessBackends(["cute"])
def test_k128_capacity_guard_covers_alignment_and_both_ab_barriers(
    capacity_delta: int,
) -> None:
    bound, args = _bound()
    _, dense = _configs(bound)
    upper = full_coverage_smem_upper_bound(4, 128, 4)
    with patch.object(
        CuteTcgen05Config,
        "per_cta_smem_capacity_bytes",
        return_value=upper + capacity_delta,
    ):
        assert bool(_configs(bound)) is (capacity_delta == 0)
        if capacity_delta == 0:
            assert KEY in bound.to_code(dense)
        else:
            with pytest.raises(BackendUnsupported):
                bound.to_code(dense)


@skipUnlessBackends(["cute"])
def test_new_declaration_is_owned_and_old_pair_is_an_unchanged_prefix() -> None:
    bound, args = _bound()
    old_seeds = deepcopy(bound.config_spec.compiler_seed_configs)
    old_pair = _configs(bound, 64)
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
    assert group.witnesses[0].carrier == group.witnesses[1].carrier == old_pair[0]
    _, dense = _configs(bound)
    assert group.witnesses[2].carrier.config == {
        key: value for key, value in dense.config.items() if key != KEY
    }
    assert group.witnesses[3].carrier == group.witnesses[2].carrier
    assert group.version == 2
    mutated = group.witnesses[2].carrier
    mutated.block_sizes[2] = 16
    assert group.witnesses[2].carrier.block_sizes[2] == 128
    assert bound.config_spec.compiler_seed_configs == old_seeds


@skipUnlessBackends(["cute"])
def test_k64_only_binding_preserves_its_original_two_declarations() -> None:
    bound, args = _bound(k=192)
    assert len(_configs(bound, 64)) == 2 and _configs(bound) == []
    (group,) = (
        candidate
        for candidate in bound.config_spec.compiler_coverage_groups
        if candidate.key == KEY
    )
    assert [witness.value for witness in group.witnesses] == [
        "off",
        "fixed_tma_dense",
        "fixed_tma_dense_local",
    ]
    assert group.witnesses[2].carrier == group.witnesses[0].carrier
