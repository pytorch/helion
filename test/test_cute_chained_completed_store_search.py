from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
import helion
from helion import exc
from helion.autotuner.config_spec import BACKEND_SPECIFIC_KEYS
from helion.autotuner.config_spec import CUTE_CHAINED_COMPLETED_MEMBER_STORE_KEY as KEY
from helion.autotuner.config_spec import VALID_KEYS
from helion.autotuner.config_spec import ConfigSpec


def _config(enabled=True):
    config = helion.Config(
        num_warps=8,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_preparation_pipeline=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_group_contractions=True,
    )
    config.config[KEY] = enabled
    return config


def test_default_appended_coordinate_preserves_entire_seed_pool(retention_bound):
    spec = retention_bound.config_spec
    assert KEY in VALID_KEYS and KEY in BACKEND_SPECIFIC_KEYS
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, False)
    assert not ConfigSpec._requests_cute_chained_loop({KEY: False})
    assert ConfigSpec._requests_cute_chained_loop({KEY: True})
    fields = spec._flat_fields()
    assert tuple(fields)[-7:] == (
        "cute_chained_broadcast_retention",
        KEY,
        "cute_native_matmul_metadata",
        "cute_chained_drain_tile_columns",
        "cute_chained_island_consumers",
        "cute_chained_async_vector_store",
        "cute_chained_fragment_epilogues",
    )
    assert fields[KEY].search_values() == [False, True]
    seeds = tuple(spec.compiler_seed_configs)
    old_values = copy.deepcopy([seed.config for seed in seeds])
    generation = spec.create_config_generation()
    pairs = generation.seed_flat_config_pairs()
    index = _flat_scalar_index(spec, KEY)
    with patch.object(
        spec, "_flat_fields", return_value={k: v for k, v in fields.items() if k != KEY}
    ):
        old_pairs = spec.create_config_generation().seed_flat_config_pairs()
    for (flat, config), (old_flat, old_config) in zip(pairs, old_pairs, strict=True):
        assert flat[:index] + flat[index + 1 :] == old_flat
        assert flat[index] is False and config == old_config
        assert KEY not in config.config
    assert all(a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True))
    assert old_values == [seed.config for seed in spec.compiler_seed_configs]
    assert spec.normalized_config(
        helion.Config.from_dict({KEY: False})
    ) == spec.normalized_config(helion.Config())
    for enabled in (False, True):
        config = spec.normalized_config(_config(enabled))
        assert config.config.get(KEY, False) is enabled
        assert config.config.get("cute_chained_compact_preparation", False) is False
        assert generation.unflatten(generation.flatten(config)) == config
        flat = generation.flatten(config)
        flat[index] = not enabled
        assert generation.unflatten(flat).config.get(KEY, False) is not enabled


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [None, 0, 1, 0.0, "true", [], {}])
def test_strict_boolean_never_repaired(retention_bound, repair, value):
    config = _config(value)
    with pytest.raises(exc.InvalidConfig, match=KEY):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "missing", ["cute_chained_mma_schedule", "cute_chained_preparation_pipeline"]
)
def test_prerequisites_are_not_injected(retention_bound, repair, missing):
    config = _config()
    config.config.pop(missing)
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)
    assert missing not in config.config


@pytest.mark.parametrize(
    "field",
    [
        "cute_chained_loop_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
        "cute_chained_tcgen05_search_enabled",
    ],
)
def test_discovery_required_and_root_has_no_coordinate(retention_bound, field):
    spec = retention_bound.config_spec
    with patch.object(spec, field, False):
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config())
        assert KEY not in spec._flat_fields()


@pytest.mark.parametrize("backend", ["triton", "pallas"])
def test_other_backend_rejects_positive_but_strips_false(retention_bound, backend):
    spec = retention_bound.config_spec
    with patch.object(spec, "backend_name", backend):
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config())
        config = {KEY: False}
        spec._normalize_cute_chained_retention(config)
        assert config == {}


@pytest.mark.parametrize(
    "conflict,value",
    [
        ("cute_chained_mma_schedule", "warp"),
        ("cute_chained_preparation_pipeline", False),
    ],
)
def test_wrong_family_never_invents_completion(retention_bound, conflict, value):
    config = _config()
    config.config[conflict] = value
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(config, _fix_invalid=True)
