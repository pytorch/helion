from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
from .test_cute_chained_frontier_ownership_search import _config as _ownership_config
import helion
from helion import exc
from helion._compiler.pallas.backend import PallasBackend
from helion._compiler.triton.backend import TritonBackend
from helion.autotuner.config_spec import BACKEND_SPECIFIC_KEYS
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_STMATRIX_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import VALID_KEYS
from helion.autotuner.config_spec import ConfigSpec


def _config(value=False, **overrides):
    return _ownership_config(32, **{KEY: value, **overrides})


def test_stmatrix_default_and_flat_coordinate_preserve_every_existing_seed(
    retention_bound,
):
    spec = retention_bound.config_spec
    assert KEY in VALID_KEYS and KEY in BACKEND_SPECIFIC_KEYS
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, False)
    assert not ConfigSpec._requests_cute_chained_loop({KEY: False})
    assert ConfigSpec._requests_cute_chained_loop({KEY: True})
    assert spec.normalized_config(
        helion.Config(cute_chained_frontier_stmatrix=False)
    ) == spec.normalized_config(helion.Config())
    fields = spec._flat_fields()
    assert tuple(fields)[-12:] == (
        KEY,
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
    assert fields[KEY].search_values() == [False, True]
    seeds = tuple(spec.compiler_seed_configs)
    old_seeds = [copy.deepcopy(seed.config) for seed in seeds]
    generation = spec.create_config_generation()
    pairs = generation.seed_flat_config_pairs()
    index = _flat_scalar_index(spec, KEY)
    with patch.object(
        spec, "_flat_fields", return_value={k: v for k, v in fields.items() if k != KEY}
    ):
        old = spec.create_config_generation()
        for (flat, config), (old_flat, old_config) in zip(
            pairs, old.seed_flat_config_pairs(), strict=True
        ):
            assert flat[:index] + flat[index + 1 :] == old_flat
            assert flat[index] is False
            assert config == old_config
    assert [seed.config for seed in seeds] == old_seeds
    assert all(a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True))
    for value in (False, True):
        normalized = spec.normalized_config(_config(value))
        flat = generation.flatten(normalized)
        assert flat[_flat_scalar_index(spec, KEY)] is value
        assert generation.unflatten(flat) == normalized
        assert (KEY in normalized.config) is value
    assert spec.normalized_config(_config(True)).config == spec.normalized_config(
        _config()
    ).config | {KEY: True}


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize("value", (None, 0, 1, 0.0, 1.0, "true", [], {}))
def test_stmatrix_strict_bool_precedes_repair(retention_bound, repair, value):
    with pytest.raises(exc.InvalidConfig, match=KEY):
        retention_bound.config_spec.normalize(_config(value), _fix_invalid=repair)


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "missing",
    (
        "cute_chained_mma_schedule",
        "cute_chained_preparation_pipeline",
        "cute_chained_pointwise_vectorize",
        "cute_chained_vector_group",
    ),
)
def test_stmatrix_does_not_inject_prerequisites(retention_bound, repair, missing):
    config = _config(True)
    config.config.pop(missing)
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize(
    "flag",
    (
        "cute_chained_loop_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
    ),
)
def test_stmatrix_requires_structural_discovery(retention_bound, flag):
    spec = retention_bound.config_spec
    with patch.object(spec, flag, False):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(True))


@pytest.mark.parametrize("backend", (TritonBackend(), PallasBackend()))
def test_stmatrix_other_backends_only_strip_false(backend):
    spec = ConfigSpec(backend=backend, device=torch.device("cpu"), num_sm=148)
    assert not spec.supports_config_key(KEY) and KEY not in spec._flat_fields()
    config: dict[str, object] = {KEY: False}
    spec.normalize(config)
    assert KEY not in config
    with pytest.raises(exc.InvalidConfig):
        spec.normalize(_config(True))
