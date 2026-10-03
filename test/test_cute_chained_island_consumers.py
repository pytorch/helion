from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_collective_retention_search import _config as _pipeline_config
from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
from .test_cute_chained_island_publication import _capture
from .test_cute_chained_island_publication import _nonlinear_polynomial_sequence
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_register_emission import _register_config
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_preparation_actions as actions
from helion.autotuner.config_spec import BACKEND_SPECIFIC_KEYS
from helion.autotuner.config_spec import CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_ISLAND_CONSUMERS_KEY as KEY
from helion.autotuner.config_spec import VALID_KEYS
from helion.autotuner.config_spec import ConfigSpec
from helion.runtime.kernel import BoundKernel


def _config(enabled=None, *, generic=False):
    config = _register_config(
        3,
        cute_chained_register_islands=True,
        cute_chained_compact_preparation=True,
        **(
            {
                "block_sizes": [],
                "cute_chained_pointwise_cache_bytes": 0,
                "cute_chained_group_contractions": False,
                "cute_chained_scan_schedule": "serial",
            }
            if generic
            else {}
        ),
    )
    if enabled is not None:
        config.config[KEY] = enabled
    return config


def _fixture(dtype=torch.bfloat16, steps=1):
    return _nonlinear_polynomial_sequence, (
        torch.empty((steps, 32, 128), dtype=dtype),
        torch.empty((steps, 32, 128), dtype=dtype),
        torch.empty((32, 128), dtype=torch.float32),
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("steps", (0, 1, 7))
def test_public_generic_source_is_exact_private_accepted_body(dtype, steps):
    kernel, args = _fixture(dtype, steps)
    captured = []
    original = actions.build_accepted_preparation

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    with patch.object(actions, "build_accepted_preparation", observe):
        public = _source(kernel, args, _config(True, generic=True))
    assert len(captured) == 1
    accepted = captured[0]
    assert accepted.island_publications and accepted.island_reads
    assert all(
        item.consumed and item.matches() for item in accepted.island_publications
    )
    assert all(item.matches(accepted) for item in accepted.island_reads)
    assert public == _capture(True, fixture=(kernel, args))[0]
    assert (
        _source(kernel, args, _config(None, generic=True))
        == _source(kernel, args, _config(False, generic=True))
        == _capture(False, fixture=(kernel, args))[0]
    )


def test_public_kda_source_is_exact_private_accepted_body():
    kernel, args = _kda_fixture()
    assert _source(kernel, args, _config(True)) == _capture(True)[0]
    assert (
        _source(kernel, args, _config(None))
        == _source(kernel, args, _config(False))
        == _capture(False)[0]
    )


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize("value", (None, 0, 1, 0.0, "true", [], {}))
def test_strict_boolean_no_repair(retention_bound, repair, value):
    config = _config(True)
    config.config[KEY] = value
    with pytest.raises(exc.InvalidConfig, match=KEY):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "missing",
    (
        "cute_chained_mma_schedule",
        "cute_chained_preparation_pipeline",
        "cute_chained_preparation_cohorts",
        "cute_chained_compact_preparation",
        "cute_chained_register_islands",
    ),
)
def test_prerequisites_are_not_injected(retention_bound, repair, missing):
    config = _config(True)
    config.config.pop(missing)
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize(
    "field",
    (
        "cute_chained_loop_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
        "cute_chained_tcgen05_search_enabled",
    ),
)
def test_missing_discovery_declines_coordinate_and_authority(retention_bound, field):
    spec = retention_bound.config_spec
    with patch.object(spec, field, False):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(True))


def test_append_only_flat_coordinate_and_canonical_false(retention_bound):
    spec = retention_bound.config_spec
    assert KEY in VALID_KEYS and KEY in BACKEND_SPECIFIC_KEYS
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, False)
    assert ConfigSpec._requests_cute_chained_loop({KEY: True})
    assert not ConfigSpec._requests_cute_chained_loop({KEY: False})
    fields = spec._flat_fields()
    assert tuple(fields)[-5:] == (
        "cute_native_matmul_metadata",
        "cute_chained_drain_tile_columns",
        KEY,
        "cute_chained_async_vector_store",
        CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY,
    )
    assert fields[CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY].default() is False
    assert (
        spec.default_config().config.get(CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY, False)
        is False
    )
    assert fields[KEY].search_values() == [False, True]
    generation = spec.create_config_generation()
    for enabled in (False, True):
        normalized = spec.normalized_config(
            _pipeline_config(
                num_warps=16,
                cute_chained_preparation_cohorts=3,
                cute_chained_compact_preparation=True,
                cute_chained_register_islands=True,
                **{KEY: enabled},
            )
        )
        assert generation.unflatten(generation.flatten(normalized)) == normalized
        assert normalized.config.get(KEY, False) is enabled
    assert spec.normalized_config(
        helion.Config.from_dict({KEY: False})
    ) == spec.normalized_config(helion.Config())
    seeds = copy.deepcopy([seed.config for seed in spec.compiler_seed_configs])
    assert all(
        seed.get(CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY, False) is False for seed in seeds
    )
    current = generation.seed_flat_config_pairs()
    with patch.object(
        spec, "_flat_fields", return_value={k: v for k, v in fields.items() if k != KEY}
    ):
        old = spec.create_config_generation().seed_flat_config_pairs()
    assert len(current) == len(old)
    for (flat, config), (old_flat, old_config) in zip(current, old, strict=True):
        assert flat[:-3] + flat[-2:] == old_flat and flat[-3] is False
        assert config == old_config
        assert config.config.get(CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY, False) is False
    assert seeds == [seed.config for seed in spec.compiler_seed_configs]


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_real_earlier_parents_append_at_most_two_without_source_emission(dtype):
    kernel, args = _fixture(dtype)
    island_stages = []
    island_seeds = CuteChainedMatmulHeuristic._with_island_consumer_seeds

    def record_island_seeds(env, ir, seeds):
        parents = copy.deepcopy([seed.config for seed in seeds])
        result = island_seeds(env, ir, seeds)
        island_stages.append((parents, copy.deepcopy([seed.config for seed in result])))
        return result

    with (
        _cpu_codegen(),
        patch.object(
            BoundKernel,
            "to_code",
            side_effect=AssertionError("seeding must not emit source"),
        ),
    ):
        with patch.object(
            CuteChainedMatmulHeuristic,
            "_with_island_consumer_seeds",
            side_effect=lambda env, ir, seeds: seeds,
        ):
            old = kernel._bind_isolated(args)
            old_default = old.config_spec.default_config()
        with patch.object(
            CuteChainedMatmulHeuristic,
            "_with_island_consumer_seeds",
            side_effect=record_island_seeds,
        ):
            bound = kernel._bind_isolated(args)
        spec = bound.config_spec
        ((parents, seeds),) = island_stages
        assert seeds[: len(parents)] == parents
        # Later seed passes may append siblings derived from the new islands.
        assert [seed.config for seed in spec.compiler_seed_configs][
            : len(seeds)
        ] == seeds
        assert spec.default_config() == old_default
        additions = seeds[len(parents) :]
        assert 0 < len(additions) <= 2
        bundle = {
            KEY,
            "cute_chained_preparation_pipeline",
            "cute_chained_warp_mma_rows",
            "cute_chained_preparation_cohorts",
            "cute_chained_compact_preparation",
            "cute_chained_register_islands",
        }
        for sibling in additions:
            matching = [
                parent
                for parent in parents
                if all(sibling.get(key) == value for key, value in parent.items())
                and set(sibling) - set(parent) <= bundle
            ]
            assert matching and sibling[KEY] is True
            assert sibling["cute_chained_warp_mma_rows"] == 32
            normalized = spec.normalized_config(helion.Config.from_dict(sibling))
            generation = spec.create_config_generation()
            assert generation.unflatten(generation.flatten(normalized)) == normalized


def test_positive_request_cannot_fall_back_without_actual_publication():
    kernel, args = _fixture()
    config = _config(True, generic=True)
    config.config["cute_chained_vector_group"] = True
    with pytest.raises(
        (exc.BackendUnsupported, ValueError), match="island|publication"
    ):
        _source(kernel, args, config)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("cute_chained_preparation_pipeline", False),
        ("cute_chained_warp_mma_rows", 64),
        ("cute_chained_preparation_cohorts", 99),
        ("cute_chained_compact_preparation", False),
        ("cute_chained_register_islands", False),
        ("cute_chained_group_contractions", True),
        ("cute_chained_pointwise_cache_bytes", 1024),
        ("cute_chained_vector_group", True),
        ("cute_chained_broadcast_retention", True),
        (KEY, False),
    ),
)
def test_seed_bundle_never_repairs_an_explicit_parent(key, value):
    kernel, args = _fixture()
    original = CuteChainedMatmulHeuristic._with_island_consumer_seeds
    visited = []

    def check(env, ir, seeds):
        # The original pool, graph and configuration context are real. Only
        # this test's explicit parent choice differs, and it must survive.
        parents = [
            helion.Config.from_dict(seed.config | {key: value}) for seed in seeds
        ]
        before = copy.deepcopy([parent.config for parent in parents])
        assert original(env, ir, parents) is parents
        assert [parent.config for parent in parents] == before
        visited.append(True)
        return seeds

    with (
        _cpu_codegen(),
        patch.object(CuteChainedMatmulHeuristic, "_with_island_consumer_seeds", check),
        patch.object(BoundKernel, "to_code", side_effect=AssertionError("no emission")),
    ):
        kernel._bind_isolated(args)
    assert visited == [True]


@pytest.mark.parametrize(
    "overrides",
    (
        {KEY: False},
        {"cute_chained_warp_mma_rows": 64},
        {"cute_chained_preparation_pipeline": False},
        {"cute_chained_pointwise_cache_bytes": 1024},
        {"cute_chained_group_contractions": True},
    ),
)
def test_seed_bundle_respects_explicit_user_overrides(overrides):
    kernel, args = _fixture()
    original = CuteChainedMatmulHeuristic._with_island_consumer_seeds
    visited = []

    def check(env, ir, seeds):
        before = copy.deepcopy([seed.config for seed in seeds])
        with patch.object(env.settings, "autotune_config_overrides", overrides):
            assert original(env, ir, seeds) is seeds
        assert [seed.config for seed in seeds] == before
        visited.append(True)
        return seeds

    with (
        _cpu_codegen(),
        patch.object(CuteChainedMatmulHeuristic, "_with_island_consumer_seeds", check),
    ):
        kernel._bind_isolated(args)
    assert visited == [True]


@pytest.mark.parametrize("missing", ("geometry", "cut", "islands"))
def test_seed_match_requires_actual_complete_planners(missing):
    from helion._compiler.cute import chained_preparation_cut
    from helion._compiler.cute import chained_register_islands
    from helion._compiler.cute import chained_tcgen_stage

    targets = {
        "geometry": (chained_tcgen_stage, "stage_geometry", None),
        "cut": (chained_preparation_cut, "plan_preparation_cut", None),
        "islands": (chained_register_islands, "plan_register_islands", ()),
    }
    kernel, args = _fixture()
    original = CuteChainedMatmulHeuristic._with_island_consumer_seeds
    visited = []

    def check(env, ir, seeds):
        module, name, value = targets[missing]
        with patch.object(module, name, return_value=value):
            assert original(env, ir, seeds) is seeds
        visited.append(True)
        return seeds

    with (
        _cpu_codegen(),
        patch.object(CuteChainedMatmulHeuristic, "_with_island_consumer_seeds", check),
    ):
        kernel._bind_isolated(args)
    assert visited == [True]


def test_unresolved_configured_dimension_declines_without_changing_parent_pool():
    from helion._compiler.compile_environment import BlockSizeInfo

    wrapped, args = _kda_fixture()
    kernel = wrapped.fn.__globals__["kda_prefill_native_math_bt32"]
    original = CuteChainedMatmulHeuristic._with_island_consumer_seeds
    visited = []

    def check(env, ir, seeds):
        before = copy.deepcopy([seed.config for seed in seeds])
        with (
            patch.object(env, "get_block_id", return_value=None) as unresolved,
            patch.object(
                BlockSizeInfo, "size_hint", side_effect=AssertionError("no seed hints")
            ),
        ):
            result = original(env, ir, seeds)
        visited.append(
            unresolved.call_count > 0
            and result is seeds
            and [seed.config for seed in seeds] == before
        )
        return seeds

    with (
        _cpu_codegen(),
        patch.object(CuteChainedMatmulHeuristic, "_with_island_consumer_seeds", check),
    ):
        kernel._bind_isolated(args)
    assert visited == [True]


def test_original_kda_fixture_explicit_group_override_cannot_seed_singletons():
    kernel, args = _kda_fixture()
    assert (
        kernel.settings.autotune_config_overrides["cute_chained_group_contractions"]
        is True
    )
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
    assert all(
        not seed.config.get(KEY, False)
        for seed in bound.config_spec.compiler_seed_configs
    )


def test_original_kda_unwrapped_pool_keeps_unproved_reductions_unseeded():
    from helion._compiler.compile_environment import BlockSizeInfo

    wrapped, args = _kda_fixture()
    kernel = wrapped.fn.__globals__["kda_prefill_native_math_bt32"]
    assert not kernel.settings.autotune_config_overrides
    original = CuteChainedMatmulHeuristic._with_island_consumer_seeds
    observed = []

    def observe(env, ir, seeds):
        before = copy.deepcopy([seed.config for seed in seeds])
        with patch.object(
            BlockSizeInfo, "size_hint", side_effect=AssertionError("no seed hints")
        ):
            result = original(env, ir, seeds)
        observed.append(result is seeds and [seed.config for seed in seeds] == before)
        return result

    with (
        _cpu_codegen(),
        patch.object(
            CuteChainedMatmulHeuristic, "_with_island_consumer_seeds", observe
        ),
    ):
        bound = kernel._bind_isolated(args)
    assert observed == [True]
    assert all(
        not seed.config.get(KEY, False)
        for seed in bound.config_spec.compiler_seed_configs
    )
