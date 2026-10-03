from __future__ import annotations

from copy import deepcopy
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_shared_rhs_grouped import _target

import helion
from helion._testing import skipUnlessBackends
from helion.autotuner import LFBOTreeSearch
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.metrics import AutotuneMetrics
from helion.exc import InvalidConfig
from helion.runtime.settings import default_autotuner_fn

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Sequence

    from helion.autotuner.base_search import PopulationMember
    from helion.runtime.kernel import BoundKernel


KEY = "tcgen05_grouped_full_coverage"


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    for name in (
        "HELION_AUTOTUNE_EFFORT",
        "HELION_AUTOTUNER",
        "HELION_AUTOTUNER_INITIAL_POPULATION",
        "HELION_AUTOTUNE_CONFIG_OVERRIDES",
        "HELION_CUTE_REGION_FISSION",
        "HELION_CUTE_FULL_SLICE_MATMUL_TILING",
        "HELION_CUTE_SEGMENTED_MATMUL_TILING",
        "HELION_CUTE_FLATTEN_NESTED_REDUCTIONS",
        "HELION_CUTE_MATERIALIZE_TRANSFORMED_OPERANDS",
    ):
        monkeypatch.delenv(name, raising=False)
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
    shape: int = 0, **settings: object
) -> tuple[BoundKernel, tuple[torch.Tensor, ...]]:
    settings = {"cute_segmented_matmul_tiling": True} | settings
    m, n, k, groups = ((640, 128, 128, 4), (2560, 512, 512, 8))[shape]
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
        **settings,
    )
    return kernel.bind(args), args


def _search(
    bound: BoundKernel, args: tuple[torch.Tensor, ...], **options: object
) -> LFBOTreeSearch:
    # CPU tensor inputs have no production CUDA cache key. Hold only that query;
    # the ordinary full-effort factory and search constructor remain real.
    with bound.env, patch.object(LocalAutotuneCache, "_generate_key"):
        cache = default_autotuner_fn(bound, args, **options)
    assert isinstance(cache, LocalAutotuneCache)
    search = cache.autotuner
    assert type(search) is LFBOTreeSearch and search.initial_population == 100
    return search


@pytest.mark.parametrize("shape", (0, 1))
@pytest.mark.parametrize(
    ("strategy", "pad"),
    (
        (None, True),
        ("from_random", True),
        ("from_best_available", False),
        ("from_best_available", True),
    ),
)
def test_ordinary_full_search_discovers_dense_without_extra_seeds(
    shape: int, strategy: str | None, pad: bool
) -> None:
    bound, args = _bound(shape, autotune_initial_population_strategy=strategy)
    spec = bound.config_spec
    assert bound.host_function is not None
    groups = spec.compiler_coverage_groups
    (group,) = (candidate for candidate in groups if candidate.key == KEY)
    assert group.key == KEY and group.domain == (
        "off",
        "fixed_tma_dense",
        "fixed_tma_dense_local",
    )
    assert [witness.value for witness in group.witnesses] == [
        "off",
        "fixed_tma_dense",
        "fixed_tma_dense",
        "fixed_tma_dense_local",
    ]
    assert group.witnesses[0].carrier == group.witnesses[1].carrier
    assert all(KEY not in witness.carrier for witness in group.witnesses)
    old_seeds = deepcopy(spec.compiler_seed_configs)
    assert all(
        KEY not in seed
        for seed in old_seeds
        if seed.get("tcgen05_grouped_worklist_source_m_tile") != 64
    )
    wide = [
        seed
        for seed in old_seeds
        if seed.get("tcgen05_grouped_worklist_source_m_tile") == 64
    ]
    assert len(wide) == 1 and wide[0][KEY] == "fixed_tma_dense_local"
    search = _search(bound, args, best_available_pad_random=pad)
    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
    ):
        random.seed(2026091459)
        rows = search._generate_initial_population_flat()
    accepted = [
        outcome.effective
        for outcome in search.compiler_coverage_outcomes
        if outcome.mechanism == group.mechanism
        and outcome.outcome in ("added", "already_present")
    ]
    assert {config.get(KEY, "off") for config in accepted} == {
        "off",
        "fixed_tma_dense",
        "fixed_tma_dense_local",
    }
    assert len(rows) <= max(100, len(old_seeds) + 1) + sum(
        outcome.outcome == "added" for outcome in search.compiler_coverage_outcomes
    )
    assert len(search.compiler_coverage_outcomes) == sum(
        len(candidate.witnesses) for candidate in groups
    )
    assert spec.compiler_seed_configs == old_seeds
    assert all(config in search._pinned_finalist_configs for config in accepted)
    for config in accepted:
        assert config is not None
        _flat, checked = search.config_gen.strict_config_pair(config)
        assert checked == config
        assert (
            (KEY + " =") in bound.to_code(config)
            if config.get(KEY) in ("fixed_tma_dense", "fixed_tma_dense_local")
            else (KEY + " =") not in bound.to_code(config)
        )


@pytest.mark.parametrize("shape", (0, 1))
def test_dense_reaches_actual_first_benchmark(shape: int) -> None:
    bound, args = _bound(shape)
    search = _search(bound, args)
    search._autotune_metrics = AutotuneMetrics()
    delivered = []

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
        random.seed(2026091461)
        search._autotune()
    dense = [
        entry.effective
        for entry in search.compiler_coverage_outcomes
        if entry.requested.get(KEY) == "fixed_tma_dense"
    ]
    assert len(dense) == 2 and all(config is not None for config in dense)
    assert all(
        config in delivered and config in search._pinned_finalist_configs
        for config in dense
    )
    local = [
        entry.effective
        for entry in search.compiler_coverage_outcomes
        if entry.requested.get(KEY) == "fixed_tma_dense_local"
    ]
    assert len(local) == 1 and local[0] is not None
    assert local[0] in delivered and local[0] in search._pinned_finalist_configs


@pytest.mark.parametrize("disabled", (False, True))
def test_explicit_off_and_disabled_sampling_keep_full_serializable_domain(
    disabled: bool,
) -> None:
    settings = (
        {"disable_autotuner_heuristics": True}
        if disabled
        else {"autotune_config_overrides": {KEY: "off"}}
    )
    bound, args = _bound(cute_segmented_matmul_tiling=True, **settings)
    groups = bound.config_spec.compiler_coverage_groups
    (group,) = (candidate for candidate in groups if candidate.key == KEY)
    assert group.domain == ("off", "fixed_tma_dense", "fixed_tma_dense_local")
    search = _search(bound, args)
    with bound.env:
        rows = search._generate_initial_population_flat()
    if disabled:
        assert "compiler_coverage_outcomes" not in vars(search)
    else:
        assert not any(
            entry.requested.get(KEY) in ("fixed_tma_dense", "fixed_tma_dense_local")
            and entry.outcome in ("added", "already_present")
            for entry in search.compiler_coverage_outcomes
        )
    assert all(
        search.config_gen.unflatten(row).get(KEY, "off") == "off" for row in rows
    )
    for mode in ("fixed_tma_dense", "fixed_tma_dense_local"):
        explicit = group.witnesses[1].carrier
        explicit.config[KEY] = mode
        assert (KEY + " =") in bound.to_code(explicit)


def test_bad_mode_and_repaired_geometry_do_not_count_as_coverage() -> None:
    bound, args = _bound(
        cute_segmented_matmul_tiling=True,
        autotune_config_overrides={"tcgen05_cluster_m": 2},
    )
    search = _search(bound, args)
    (group,) = (
        candidate
        for candidate in bound.config_spec.compiler_coverage_groups
        if candidate.key == KEY
    )
    malformed = group.witnesses[0].carrier
    malformed.config[KEY] = "unknown"
    with pytest.raises((InvalidConfig, ValueError)):
        search.config_gen.strict_config_pair(malformed)
    with bound.env:
        search._generate_initial_population_flat()
    assert not any(
        entry.effective is not None
        and entry.effective.get(KEY) in ("fixed_tma_dense", "fixed_tma_dense_local")
        and entry.outcome in ("added", "already_present")
        for entry in search.compiler_coverage_outcomes
    )
