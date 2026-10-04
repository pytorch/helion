from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_preparation_cut import _inputs
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_residency_search import _root_residency
import helion
from helion import exc
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import BACKEND_SPECIFIC_KEYS
from helion.autotuner.config_spec import CUTE_CHAINED_COLLECTIVE_RETENTION_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_STMATRIX_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_LEAF_COUNT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_NATIVE_VECTOR_READS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OPERAND_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import VALID_KEYS
from helion.autotuner.config_spec import ConfigSpec


def _config(enabled=False, **overrides):
    return helion.Config.from_dict(
        {
            "num_warps": 8,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_warp_mma_rows": 32,
            "cute_chained_preparation_pipeline": True,
            KEY: enabled,
            **overrides,
        }
    )


@pytest.fixture(scope="module")
def retention_bound():
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)):
            yield bound


def test_constructor_backend_default_and_loop_request(retention_bound):
    spec = retention_bound.config_spec
    assert KEY in VALID_KEYS and KEY in BACKEND_SPECIFIC_KEYS
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, False)
    explicit = helion.Config(cute_chained_collective_retention=False)
    assert explicit[KEY] is False
    assert spec.normalized_config(explicit) == spec.normalized_config(helion.Config())
    assert KEY not in spec.normalized_config(explicit).config
    assert not ConfigSpec._requests_cute_chained_loop({KEY: False})
    assert ConfigSpec._requests_cute_chained_loop({KEY: True})


@pytest.mark.parametrize("enabled", [False, True])
def test_exact_bool_choices_roundtrip_and_override(retention_bound, enabled):
    spec = retention_bound.config_spec
    field = spec._flat_fields()[KEY]
    assert isinstance(field, EnumFragment)
    assert field.default() is False and field.search_values() == [False, True]
    config = spec.normalized_config(_config(enabled))
    assert config.config.get(KEY, False) is enabled
    assert (KEY in config.config) == enabled
    generation = spec.create_config_generation()
    assert generation.unflatten(generation.flatten(config)) == config
    assert generation.flatten(config)[_flat_scalar_index(spec, KEY)] is enabled
    override = spec.create_config_generation(overrides={KEY: enabled})
    assert (
        override.unflatten(override.flatten(spec.normalized_config(_config())))
        == config
    )


def test_append_only_field_preserves_old_values_order_defaults_and_seed_objects(
    retention_bound,
):
    spec = retention_bound.config_spec
    seeds = tuple(spec.compiler_seed_configs)
    snapshots = [copy.deepcopy(seed.config) for seed in seeds]
    with patch.dict(
        spec.user_defined_tunables, {"retention_test_choice": EnumFragment((3, 5))}
    ):
        fields = spec._flat_fields()
        assert tuple(fields)[-19:] == (
            CUTE_CHAINED_LEAF_COUNT_KEY,
            KEY,
            CUTE_CHAINED_OPERAND_RETENTION_KEY,
            CUTE_CHAINED_FRONTIER_TILE_COLUMNS_KEY,
            CUTE_CHAINED_NATIVE_VECTOR_READS_KEY,
            CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY,
            CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY,
            CUTE_CHAINED_FRONTIER_STMATRIX_KEY,
            CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY,
            "cute_min_blocks_per_mp",
            "cute_chained_compact_preparation",
            "cute_chained_leaf_issue_batching",
            "cute_chained_broadcast_retention",
            "cute_chained_completed_member_store",
            "cute_native_matmul_metadata",
            "cute_chained_drain_tile_columns",
            "cute_chained_island_consumers",
            "cute_chained_async_vector_store",
            "cute_chained_fragment_epilogues",
        )
        assert spec.flat_key_layout()[-18] == (KEY, 1, False)
        index = _flat_scalar_index(spec, KEY)
        default = spec.default_config()
        generation = spec.create_config_generation()
        pairs = generation.seed_flat_config_pairs()
        with patch.object(
            spec,
            "_flat_fields",
            return_value={key: field for key, field in fields.items() if key != KEY},
        ):
            old = spec.create_config_generation()
            flat = generation.flatten(default)
            assert flat[:index] + flat[index + 1 :] == old.flatten(default)
            assert flat[index] is False
            assert spec.default_config() == default
            for (flat, config), (old_flat, old_config) in zip(
                pairs, old.seed_flat_config_pairs(), strict=True
            ):
                assert flat[:index] + flat[index + 1 :] == old_flat
                assert flat[index] is False
                assert config == old_config
        assert KEY not in default.config
    assert all(KEY not in seed.config for seed in seeds)
    assert all(
        left is right
        for left, right in zip(seeds, spec.compiler_seed_configs, strict=True)
    )
    assert [seed.config for seed in seeds] == snapshots


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "value", [None, 0, 1, -1, 2, 0.0, 1.0, "false", "true", [], {}]
)
def test_strict_bool_is_checked_before_autotune_repair(retention_bound, value, repair):
    with pytest.raises(exc.InvalidConfig, match=KEY):
        retention_bound.config_spec.normalize(_config(value), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "prerequisite", ["cute_chained_mma_schedule", "cute_chained_preparation_pipeline"]
)
def test_no_missing_prerequisite_is_injected(retention_bound, prerequisite, repair):
    config = _config(True)
    config.config.pop(prerequisite)
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)
    assert prerequisite not in config.config


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"cute_chained_mma_schedule": "coalesced"},
        {"cute_chained_mma_schedule": None},
        {"cute_chained_preparation_pipeline": False},
        {"cute_chained_warp_mma_rows": 0},
        {"num_warps": 4},
    ],
)
def test_inactive_and_incompatible_preparation_rejects(
    retention_bound, overrides, repair
):
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(
            _config(True, **overrides), _fix_invalid=repair
        )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "flag",
    [
        "cute_chained_loop_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
    ],
)
def test_discovery_is_required_without_inventing_new_flags(
    retention_bound, flag, repair
):
    spec = retention_bound.config_spec
    with patch.object(spec, flag, False):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(True), _fix_invalid=repair)


@pytest.mark.parametrize("backend", ["triton", "pallas"])
@pytest.mark.parametrize("repair", [False, True])
def test_other_backends_only_accept_stripped_default(retention_bound, backend, repair):
    spec = retention_bound.config_spec
    with patch.object(spec, "backend_name", backend):
        default = {KEY: False}
        spec.normalize(default, _fix_invalid=repair)
        assert KEY not in default
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(True), _fix_invalid=repair)


@pytest.mark.parametrize("fast_math", [False, True])
def test_retention_does_not_require_or_change_fast_math(retention_bound, fast_math):
    with patch.object(_typed_sequence.settings, "fast_math", fast_math):
        config = retention_bound.config_spec.normalized_config(_config(True))
        assert config[KEY] is True
        assert _typed_sequence.settings.fast_math is fast_math
        assert "fast_math" not in config.config


@pytest.mark.parametrize("leaf_pipeline", ["legacy", "rectangular_tma"])
def test_no_leaf_or_register_island_prerequisite(retention_bound, leaf_pipeline):
    config = retention_bound.config_spec.normalized_config(
        _config(True, cute_chained_leaf_pipeline=leaf_pipeline)
    )
    assert config[KEY] is True
    assert "cute_chained_leaf_count" not in config.config
    assert "cute_chained_register_islands" not in config.config


def test_root_has_no_coordinate_and_cannot_be_promoted_to_preparation():
    with _cpu_codegen():
        bound = _root_residency._bind_isolated(
            (
                torch.empty((128, 16), dtype=torch.bfloat16),
                torch.empty((16, 32), dtype=torch.bfloat16),
            )
        )
        spec = bound.config_spec
        assert not spec.cute_chained_loop_search_enabled
        assert KEY not in spec._flat_fields()
        assert spec.normalized_config(
            helion.Config(cute_chained_collective_retention=False)
        ) == spec.normalized_config(helion.Config())
        for repair in (False, True):
            with pytest.raises(exc.InvalidConfig):
                spec.normalize(_config(True), _fix_invalid=repair)
