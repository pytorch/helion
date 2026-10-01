from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_preparation_cut
from helion.runtime.kernel import BoundKernel

_METHOD = "_with_retained_scan_preparation_seeds"
_BUNDLE = {
    "cute_chained_compact_preparation",
    "cute_chained_scan_producer_retention",
    "cute_chained_operand_retention",
    "cute_chained_preparation_cohorts",
    "cute_chained_leaf_pipeline",
    "cute_chained_leaf_count",
}


@pytest.fixture(scope="module")
def retained_scan_seeds():
    kernel, args = _kda_fixture()
    original = CuteChainedMatmulHeuristic._with_retained_scan_preparation_seeds
    observations = []

    def observe(env, ir, seeds):
        values = copy.deepcopy([seed.config for seed in seeds])
        result = original(env, ir, seeds)
        assert all(a is b for a, b in zip(seeds, result[: len(seeds)], strict=True))
        assert values == [seed.config for seed in seeds]
        observations.append(len(result) - len(seeds))
        return result

    with _cpu_codegen():
        with patch.object(
            BoundKernel, "to_code", side_effect=AssertionError("no emission")
        ):
            with patch.object(
                CuteChainedMatmulHeuristic,
                _METHOD,
                side_effect=lambda env, ir, seeds: seeds,
            ):
                old = kernel._bind_isolated(args)
            with patch.object(CuteChainedMatmulHeuristic, _METHOD, side_effect=observe):
                bound = kernel._bind_isolated(args)
        assert observations == [2]
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            yield old, bound


def _propose(bound, seeds):
    with bound.env:
        return CuteChainedMatmulHeuristic._with_retained_scan_preparation_seeds(
            bound.env, bound.host_function.device_ir, seeds
        )


def test_retained_scan_seeds_preserve_prefix_default_and_full100(retained_scan_seeds):
    old, bound = retained_scan_seeds
    spec = bound.config_spec
    parents = old.config_spec.compiler_seed_configs
    seeds = spec.compiler_seed_configs
    assert seeds[: len(parents)] == parents
    assert spec.default_config() == old.config_spec.default_config()
    assert len(seeds) == len(parents) + 2
    overrides = bound.settings.autotune_config_overrides
    original = old.config_spec.create_config_generation(overrides=overrides)
    generation = spec.create_config_generation(overrides=overrides)
    old_pairs = original.seed_flat_config_pairs()
    pairs = generation.seed_flat_config_pairs()
    assert pairs[: len(old_pairs)] == old_pairs
    assert len(pairs) == len(old_pairs) + 2
    population = generation.random_population_flat(100)
    assert len(population) == 100
    for seed, (flat, normalized), consumer in zip(
        seeds[-2:], pairs[-2:], (4, 8), strict=True
    ):
        matches = [
            parent
            for parent in parents
            if set(seed.config) - set(parent.config) == _BUNDLE
            and all(
                seed.config.get(key) == value for key, value in parent.config.items()
            )
        ]
        assert matches
        assert seed.config.get("cute_chained_pipeline_consumer_warps", 4) == consumer
        assert (
            seed.config["cute_chained_preparation_cohorts"]
            == (seed.num_warps - consumer) // 4
        )
        assert seed.config["cute_chained_leaf_pipeline"] == "rectangular_tma"
        assert seed.config["cute_chained_leaf_count"] == 4
        assert generation.unflatten(flat) == normalized
        assert flat in population
    assert _propose(bound, seeds) is seeds


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("cute_chained_compact_preparation", False),
        ("cute_chained_scan_producer_retention", False),
        ("cute_chained_operand_retention", False),
        ("cute_chained_preparation_cohorts", 99),
        ("cute_chained_leaf_pipeline", "legacy"),
        ("cute_chained_leaf_count", 2),
    ),
)
def test_retained_scan_seeds_preserve_conflicts_and_overrides(
    retained_scan_seeds, key, value
):
    old, bound = retained_scan_seeds
    parents = [
        helion.Config.from_dict(seed.config | {key: value})
        for seed in old.config_spec.compiler_seed_configs
    ]
    before = copy.deepcopy([parent.config for parent in parents])
    assert _propose(bound, parents) is parents
    assert [parent.config for parent in parents] == before
    parents = old.config_spec.compiler_seed_configs
    overrides = bound.settings.autotune_config_overrides | {key: value}
    with patch.object(bound.settings, "autotune_config_overrides", overrides):
        assert _propose(bound, parents) is parents


@pytest.mark.parametrize(
    "flag",
    (
        "cute_chained_preparation_pipeline_search_enabled",
        "cute_chained_tcgen05_search_enabled",
        "cute_chained_group_search_enabled",
        "cute_chained_scan_search_enabled",
        "cute_chained_pointwise_residency_search_enabled",
    ),
)
def test_retained_scan_seeds_require_graph_and_hardware_facts(
    retained_scan_seeds, flag
):
    old, bound = retained_scan_seeds
    parents = old.config_spec.compiler_seed_configs
    with patch.object(bound.config_spec, flag, False):
        assert _propose(bound, parents) is parents


def test_retained_scan_seeds_require_preparation_cut(retained_scan_seeds):
    old, bound = retained_scan_seeds
    parents = old.config_spec.compiler_seed_configs
    with patch.object(
        chained_preparation_cut, "plan_preparation_cut", return_value=None
    ):
        assert _propose(bound, parents) is parents


@pytest.mark.parametrize("index", (-2, -1))
def test_retained_scan_seed_public_source(retained_scan_seeds, index):
    old, bound = retained_scan_seeds
    config = bound.config_spec.compiler_seed_configs[index]
    with patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")):
        source = bound.to_code(config)
    assert "chain_scan_producer_" in source
    assert "chain_leaf_1_tensor" in source
    assert "chain_leaf_2_tensor" in source
    assert "chain_cohort = chain_thread // 128" in source


def test_retained_scan_seed_incomplete_leaf_coverage_still_rejects(retained_scan_seeds):
    old, bound = retained_scan_seeds
    config = helion.Config.from_dict(
        bound.config_spec.compiler_seed_configs[-2].config
        | {"cute_chained_leaf_count": 2}
    )
    with (
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
        pytest.raises(
            exc.BackendUnsupported, match="scan producer coordinate/native proof failed"
        ),
    ):
        bound.to_code(config)
