from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_collective_retention_search import _config as _base_config
from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
import helion
from helion import exc
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import BACKEND_SPECIFIC_KEYS
from helion.autotuner.config_spec import CUTE_CHAINED_COLLECTIVE_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_STMATRIX_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_NATIVE_VECTOR_READS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OPERAND_RETENTION_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import VALID_KEYS
from helion.autotuner.config_spec import ConfigSpec


def _config(enabled=False, **overrides):
    return _base_config(**{KEY: enabled, **overrides})


def test_default_registration_and_exact_append_only_coordinate(retention_bound):
    spec = retention_bound.config_spec
    assert KEY in VALID_KEYS and KEY in BACKEND_SPECIFIC_KEYS
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, False)
    explicit = helion.Config(cute_chained_operand_retention=False)
    assert spec.normalized_config(explicit) == spec.normalized_config(helion.Config())
    assert KEY not in spec.normalized_config(explicit).config
    assert not ConfigSpec._requests_cute_chained_loop({KEY: False})
    assert ConfigSpec._requests_cute_chained_loop({KEY: True})
    seeds = tuple(spec.compiler_seed_configs)
    saved = [copy.deepcopy(seed.config) for seed in seeds]
    fields = spec._flat_fields()
    assert tuple(fields)[-18:] == (
        CUTE_CHAINED_COLLECTIVE_RETENTION_KEY,
        KEY,
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
    assert spec.flat_key_layout()[-17] == (KEY, 1, False)
    index = _flat_scalar_index(spec, KEY)
    assert isinstance(fields[KEY], EnumFragment)
    assert fields[KEY].search_values() == [False, True]
    generation = spec.create_config_generation()
    default = spec.default_config()
    pairs = generation.seed_flat_config_pairs()
    with patch.object(
        spec, "_flat_fields", return_value={k: v for k, v in fields.items() if k != KEY}
    ):
        previous = spec.create_config_generation()
        flat = generation.flatten(default)
        assert flat[:index] + flat[index + 1 :] == previous.flatten(default)
        assert flat[index] is False
        for (flat, config), (old_flat, old_config) in zip(
            pairs, previous.seed_flat_config_pairs(), strict=True
        ):
            assert flat[:index] + flat[index + 1 :] == old_flat
            assert flat[index] is False
            assert config == old_config
    assert [seed.config for seed in seeds] == saved
    assert all(a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True))


@pytest.mark.parametrize("enabled", [False, True])
def test_exact_bool_roundtrip_and_independent_override(retention_bound, enabled):
    spec = retention_bound.config_spec
    config = spec.normalized_config(_config(enabled))
    assert config.config.get(KEY, False) is enabled
    generation = spec.create_config_generation()
    assert generation.unflatten(generation.flatten(config)) == config
    assert generation.flatten(config)[_flat_scalar_index(spec, KEY)] is enabled
    override = spec.create_config_generation(overrides={KEY: enabled})
    assert (
        override.unflatten(override.flatten(spec.normalized_config(_config())))
        == config
    )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [None, 0, 1, -1, 0.0, "false", "true", [], {}])
def test_strict_boolean_before_repair(retention_bound, repair, value):
    with pytest.raises(exc.InvalidConfig, match=KEY):
        retention_bound.config_spec.normalize(_config(value), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "key", ["cute_chained_mma_schedule", "cute_chained_preparation_pipeline"]
)
def test_does_not_invent_prerequisites(retention_bound, repair, key):
    config = _config(True)
    config.config.pop(key)
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)
    assert key not in config.config


@pytest.mark.parametrize("backend", ["triton", "pallas"])
def test_other_backend_accepts_only_stripped_default(retention_bound, backend):
    spec = retention_bound.config_spec
    with patch.object(spec, "backend_name", backend):
        default = {KEY: False}
        spec.normalize(default)
        assert KEY not in default
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(True))


@pytest.mark.parametrize(
    "flag",
    [
        "cute_chained_loop_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
    ],
)
def test_requires_structural_discovery(retention_bound, flag):
    spec = retention_bound.config_spec
    with patch.object(spec, flag, False):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(True))


def test_retention_does_not_inject_other_optimizations(retention_bound):
    config = retention_bound.config_spec.normalized_config(_config(True))
    assert config[KEY] is True
    assert all(
        key not in config.config
        for key in (
            "cute_chained_register_islands",
            "cute_chained_leaf_count",
            "cute_chained_collective_retention",
            "fast_math",
        )
    )
