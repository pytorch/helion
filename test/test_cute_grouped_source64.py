from __future__ import annotations

from copy import deepcopy
import itertools
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_grouped_coverage_search import _search

import helion
from helion._compiler.autotuner_heuristics.cute import (
    CuteTcgen05GroupedSource64Heuristic,
)
from helion._compiler.autotuner_heuristics.cute import grouped_full_coverage_configs
from helion._compiler.cute.grouped_full_coverage import full_coverage_pipeline_supported
from helion._compiler.cute.grouped_full_coverage import full_coverage_smem_upper_bound
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_constants import (
    resolve_tcgen05_grouped_worklist_mma_shape,
)
from helion._testing import skipUnlessBackends
from helion.autotuner.metrics import AutotuneMetrics
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Sequence

    from helion.autotuner.base_search import PopulationMember
    from helion.runtime.kernel import BoundKernel


KEY = "tcgen05_grouped_full_coverage"
WIDTH = "tcgen05_grouped_worklist_source_m_tile"
FLAGS = {
    "cute_region_fission": True,
    "cute_full_slice_matmul_tiling": True,
    "cute_segmented_matmul_tiling": True,
    "cute_flatten_nested_reductions": True,
    "cute_materialize_transformed_operands": True,
}


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
    *,
    dtype: torch.dtype = torch.bfloat16,
    index_dtype: torch.dtype = torch.int32,
    k: int = 128,
    m: int = 640,
    **settings: object,
) -> tuple[BoundKernel, tuple[torch.Tensor, ...]]:
    args = (
        torch.empty((m, k), dtype=dtype),
        torch.empty((k, 128), dtype=dtype),
        torch.tensor([0, 64, 256, 448, m], dtype=index_dtype),
    )
    kernel = helion.kernel(
        grouped_gemm_jagged.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="full",
        **FLAGS,
        **settings,
    )
    return kernel._bind_isolated(args), args


@pytest.fixture(scope="module")
def family() -> tuple[BoundKernel, tuple[torch.Tensor, ...]]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        _target(),
    ):
        return _bound()


def _wide(bound: BoundKernel) -> helion.Config:
    assert bound.host_function is not None
    config = CuteTcgen05GroupedSource64Heuristic.get_seed_config(
        bound.env, bound.host_function.device_ir
    )
    assert config is not None
    return config


def test_physical_profile_keeps_old_domains_and_limits_source64() -> None:
    for width, bk, cluster in itertools.product((32, 64, 224, 256), (64, 128), (1, 2)):
        expected = None
        if cluster == 2 and width in (32, 224, 256):
            expected = (256, width)
        elif cluster == 1 and (width == 32 or (width == 64 and bk == 128)):
            expected = (128, width)
        assert (
            resolve_tcgen05_grouped_worklist_mma_shape(
                source_m_tile=width, block_k=bk, cluster_m=cluster
            )
            == expected
        )
    for value in (64.0, True, "64", 16, 128):
        assert (
            resolve_tcgen05_grouped_worklist_mma_shape(
                source_m_tile=value, block_k=128, cluster_m=1
            )
            is None
        )


def test_only_complete_typed_source64_pipeline_and_fixed_c_ring() -> None:
    for width, bk, ab, registers in itertools.product(
        (32, 64), (64, 128), (2, 3, 4, 5), (240, 256)
    ):
        expected = (
            width == 32 and (bk, ab, registers) in ((64, 2, 240), (128, 4, 256))
        ) or (width == 64 and (bk, ab, registers) == (128, 4, 256))
        assert (
            full_coverage_pipeline_supported(bk, ab, registers, source_m_tile=width)
            is expected
        )
    assert not full_coverage_pipeline_supported(128, 4, 256, source_m_tile=64.0)
    for groups in (1, 4, 8, 31):
        small = full_coverage_smem_upper_bound(groups, 128, 4)
        wide = full_coverage_smem_upper_bound(groups, 128, 4, source_m_tile=64)
        assert wide - small == 4 * (64 - 32) * 128 * 2


@pytest.mark.parametrize(
    "updates",
    (
        {KEY: "off"},
        {KEY: "fixed_tma_dense"},
        {WIDTH: 64.0},
        {"block_sizes": [256, 128, 64]},
        {"tcgen05_ab_stages": 2},
        {"tcgen05_ab_stages": 5},
        {"tcgen05_consumer_regs": 240},
        {"tcgen05_acc_stages": 1},
        {"tcgen05_c_stages": 4},
        {"tcgen05_cluster_m": 2},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_num_epi_warps": 8},
        {"tcgen05_sched_stage_count": 2},
        {"tcgen05_l2_swizzle_size": 2},
        {"tcgen05_grouped_runtime_direct": True},
        {"tcgen05_grouped_external_direct_pointers": True},
        {"tcgen05_ab_producer_advance_mode": "skip"},
    ),
)
@skipUnlessBackends(["cute"])
def test_incomplete_or_other_source64_schedule_rejected(
    family: tuple[BoundKernel, tuple[torch.Tensor, ...]], updates: dict[str, object]
) -> None:
    bound, args = family
    config = _wide(bound)
    config.config.update(deepcopy(updates))
    with pytest.raises((InvalidConfig, BackendUnsupported)):
        bound.to_code(config)


@skipUnlessBackends(["cute"])
def test_shared_memory_limit_is_used_at_seed_and_final_codegen(
    family: tuple[BoundKernel, tuple[torch.Tensor, ...]],
) -> None:
    bound, args = family
    config = _wide(bound)
    upper = full_coverage_smem_upper_bound(4, 128, 4, source_m_tile=64)
    assert bound.host_function is not None
    with patch.object(
        CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=upper - 1
    ):
        assert (
            CuteTcgen05GroupedSource64Heuristic.get_seed_config(
                bound.env, bound.host_function.device_ir
            )
            is None
        )
        with pytest.raises(BackendUnsupported):
            bound.to_code(config)
    with patch.object(
        CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=upper
    ):
        assert _wide(bound) == config
        assert "tcgen05_grouped_full_coverage" in bound.to_code(config)


@pytest.mark.parametrize(
    "dtype,index_dtype,k",
    (
        (torch.float16, torch.int32, 128),
        (torch.bfloat16, torch.int64, 128),
        (torch.bfloat16, torch.int32, 192),
    ),
)
@skipUnlessBackends(["cute"])
def test_unsupported_semantics_do_not_create_source64_seed(
    dtype: torch.dtype, index_dtype: torch.dtype, k: int
) -> None:
    bound, args = _bound(dtype=dtype, index_dtype=index_dtype, k=k)
    assert bound.host_function is not None
    assert (
        CuteTcgen05GroupedSource64Heuristic.get_seed_config(
            bound.env, bound.host_function.device_ir
        )
        is None
    )
    assert all(
        config.get(WIDTH) != 64 for config in bound.config_spec.compiler_seed_configs
    )


@pytest.mark.parametrize("strategy", (None, "from_best_available"))
@skipUnlessBackends(["cute"])
def test_source64_reaches_original_full_search_first_benchmark(
    strategy: str | None,
) -> None:
    bound, args = _bound(autotune_initial_population_strategy=strategy)
    source64 = _wide(bound)
    source32_seeds = [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.get(WIDTH) != 64
    ]
    assert len(source32_seeds) + 1 == len(bound.config_spec.compiler_seed_configs)
    assert bound.config_spec.compiler_seed_configs[-1] == source64
    search = _search(bound, args)
    search._autotune_metrics = AutotuneMetrics()
    delivered: list[helion.Config] = []

    class HeldBenchmark(Exception):
        pass

    def hold(members: Sequence[PopulationMember], *, desc: str) -> None:
        assert desc == "Initial population"
        delivered.extend(deepcopy(member.config) for member in members)
        raise HeldBenchmark

    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
        patch.object(search, "benchmark_population", side_effect=hold),
        pytest.raises(HeldBenchmark),
    ):
        random.seed(2026091764)
        search._autotune()
    wide = [config for config in delivered if config.get(WIDTH) == 64]
    assert len(wide) == 1 and wide[0].get(KEY) == "fixed_tma_dense_local"
    assert wide[0].block_sizes == [256, 128, 128]
    assert wide[0]["tcgen05_ab_stages"] == 4 and wide[0]["tcgen05_acc_stages"] == 2
    assert wide[0] in search._pinned_finalist_configs
    assert search.config_gen.strict_config_pair(wide[0])[1] == wide[0]
    assert any(
        config.get(WIDTH) == 32 and config.get(KEY) == "fixed_tma_dense_local"
        for config in delivered
    )
    outcomes = [
        outcome
        for outcome in search.compiler_coverage_outcomes
        if outcome.mechanism == "cute.grouped_full_coverage"
    ]
    assert len(outcomes) == 4
    assert all(
        outcome.outcome in ("added", "already_present")
        and outcome.effective in delivered
        for outcome in outcomes
    )


@pytest.mark.parametrize("disabled", (False, True))
@skipUnlessBackends(["cute"])
def test_off_or_disabled_search_does_not_deliver_source64(disabled: bool) -> None:
    settings = (
        {"disable_autotuner_heuristics": True}
        if disabled
        else {"autotune_config_overrides": {KEY: "off"}}
    )
    bound, args = _bound(**settings)
    search = _search(bound, args)
    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
    ):
        random.seed(2026091764)
        rows = search._generate_initial_population_flat()
    assert all(
        search.config_gen.unflatten(row).get(KEY, "off") == "off" for row in rows
    )
    assert all(search.config_gen.unflatten(row).get(WIDTH) != 64 for row in rows)
    # An explicit proved source64 request remains independently serializable.
    assert bound.host_function is not None
    _, config = grouped_full_coverage_configs(
        bound.env, bound.host_function.device_ir, block_k=128
    )
    config.config.update({KEY: "fixed_tma_dense_local", WIDTH: 64})
    assert "tcgen05_grouped_full_coverage" in bound.to_code(config)


@skipUnlessBackends(["cute"])
def test_source64_keeps_stricter_dense_m_divisibility() -> None:
    bound, args = _bound(m=672, k=384)
    assert bound.host_function is not None
    assert (
        CuteTcgen05GroupedSource64Heuristic.get_seed_config(
            bound.env, bound.host_function.device_ir
        )
        is None
    )
    _, config = grouped_full_coverage_configs(
        bound.env, bound.host_function.device_ir, block_k=128
    )
    config.config[KEY] = "fixed_tma_dense_local"
    assert "tcgen05_grouped_full_coverage" in bound.to_code(config)
    config.config[WIDTH] = 64
    with pytest.raises(BackendUnsupported):
        bound.to_code(config)
