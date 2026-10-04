from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from .test_cute_chained_broadcast_retention import KEY
from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
from .test_cute_chained_leaf_batching import _config as _base_config
import helion
from helion import exc
from helion.autotuner.config_spec import BACKEND_SPECIFIC_KEYS
from helion.autotuner.config_spec import VALID_KEYS
from helion.autotuner.config_spec import ConfigSpec


def _config(enabled):
    config = _base_config()
    config.config[KEY] = enabled
    return config


def test_append_only_false_default_and_seed_prefix(retention_bound):
    spec = retention_bound.config_spec
    assert KEY in VALID_KEYS and KEY in BACKEND_SPECIFIC_KEYS
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, False)
    assert not ConfigSpec._requests_cute_chained_loop({KEY: False})
    assert ConfigSpec._requests_cute_chained_loop({KEY: True})
    fields = spec._flat_fields()
    assert tuple(fields)[-8:] == (
        "cute_chained_leaf_issue_batching",
        KEY,
        "cute_chained_completed_member_store",
        "cute_native_matmul_metadata",
        "cute_chained_drain_tile_columns",
        "cute_chained_island_consumers",
        "cute_chained_async_vector_store",
        "cute_chained_fragment_epilogues",
    )
    assert fields[KEY].search_values() == [False, True]
    seeds = copy.deepcopy([seed.config for seed in spec.compiler_seed_configs])
    generation = spec.create_config_generation()
    current = generation.seed_flat_config_pairs()
    index = _flat_scalar_index(spec, KEY)
    with patch.object(
        spec, "_flat_fields", return_value={k: v for k, v in fields.items() if k != KEY}
    ):
        previous = spec.create_config_generation().seed_flat_config_pairs()
    for (flat, config), (old_flat, old_config) in zip(current, previous, strict=True):
        assert flat[:index] + flat[index + 1 :] == old_flat
        assert flat[index] is False
        assert config == old_config and KEY not in config.config
    assert seeds == [seed.config for seed in spec.compiler_seed_configs]
    for enabled in (False, True):
        normalized = spec.normalized_config(_config(enabled))
        assert generation.unflatten(generation.flatten(normalized)) == normalized
        assert normalized.config.get(KEY, False) is enabled
    assert spec.normalized_config(
        helion.Config.from_dict({KEY: False})
    ) == spec.normalized_config(helion.Config())


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [None, 0, 1, 0.0, "true", [], {}])
def test_strict_boolean_no_repair(retention_bound, repair, value):
    config = _config(True)
    config.config[KEY] = value
    with pytest.raises(exc.InvalidConfig, match=KEY):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "missing",
    [
        "cute_chained_mma_schedule",
        "cute_chained_preparation_pipeline",
        "cute_chained_compact_preparation",
        "cute_chained_preparation_cohorts",
    ],
)
def test_requires_actual_physical_pipeline_without_injection(
    retention_bound, repair, missing
):
    config = _config(True)
    config.config.pop(missing)
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize(
    "field",
    [
        "cute_chained_loop_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
    ],
)
def test_undiscovered_has_no_coordinate_or_authority(retention_bound, field):
    spec = retention_bound.config_spec
    with patch.object(spec, field, False):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(True))


def test_root_cannot_invent_owner_lifetime(retention_bound):
    spec = retention_bound.config_spec
    with (
        patch.object(spec, "cute_chained_loop_search_enabled", False),
        patch.object(spec, "cute_chained_tcgen05_search_enabled", True),
    ):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(True))
