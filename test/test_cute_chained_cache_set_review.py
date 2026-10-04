"""Independent integration checks; these do not exercise planner ranking."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_cache_set_search import cache_set_bound as cache_set_bound
from .test_cute_chained_legacy_family import _LEGACY_KEYS
from .test_cute_chained_legacy_family import _resident_parent
from .test_cute_chained_legacy_family import overlap_bound as overlap_bound
import helion
from helion import exc
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_MMA_SCHEDULE_KEY as SCHEDULE
from helion.autotuner.config_spec import (
    CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY as BUDGET,
)
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_ENTRIES_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_LAYOUT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_COHORTS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_UNROLL_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_REGISTER_ISLANDS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SEED_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_VECTOR_GROUP_KEY
from helion.autotuner.config_spec import CUTE_GRID_WORK_ORDER_KEY
from helion.autotuner.config_spec import CUTE_NATIVE_MATMUL_METADATA_KEY


@pytest.mark.parametrize("schedule", ("coalesced", "tcgen05_tmem"))
@pytest.mark.parametrize("entries", (2, 4))
@pytest.mark.parametrize("repair", (False, True))
def test_ineffective_multi_cache_request_cannot_fall_back_to_other_lowering(
    cache_set_bound, schedule, entries, repair
):
    config = helion.Config.from_dict(
        {SCHEDULE: schedule, BUDGET: 4096, KEY: entries, "num_warps": 4}
    )
    cache_set_bound.config_spec.normalize(config, _fix_invalid=repair)
    assert config[KEY] == entries
    with pytest.raises(exc.BackendUnsupported, match="effective typed residency"):
        cache_set_bound.to_code(config)


@pytest.mark.parametrize("schedule", ("coalesced", "tcgen05_tmem"))
def test_explicit_one_preserves_default_source_byte_for_byte(cache_set_bound, schedule):
    values = {SCHEDULE: schedule, BUDGET: 4096, "num_warps": 4}
    plain = cache_set_bound.to_code(helion.Config.from_dict(values))
    explicit = cache_set_bound.to_code(helion.Config.from_dict(values | {KEY: 1}))
    assert plain == explicit


def test_legacy_default_and_all_six_seeds_keep_inactive_count(overlap_bound):
    spec = overlap_bound.config_spec
    default = spec.default_config()
    assert KEY not in default.config
    assert spec.normalized_config(default.config | {KEY: 1}) == default
    assert overlap_bound.to_code(default.config | {KEY: 1}) == overlap_bound.to_code(
        default
    )
    legacy = [seed for seed in spec.compiler_seed_configs if _LEGACY_KEYS[0] in seed]
    assert len(legacy) == 6
    generation = ConfigGeneration(spec)
    for seed in legacy:
        normalized = generation.unflatten(generation.flatten(seed))
        assert KEY not in normalized.config
        assert spec.normalized_config(normalized) == normalized
        assert (
            generation.unflatten(
                generation.flatten(helion.Config.from_dict(seed.config | {KEY: 1}))
            )
            == normalized
        )


def test_new_coordinate_appends_to_complete_existing_schema_and_seed_pool(
    overlap_bound,
):
    spec = overlap_bound.config_spec
    current_fields = spec._flat_fields()
    expected_suffix = (
        KEY,
        CUTE_CHAINED_SEED_TILE_COLUMNS_KEY,
        CUTE_CHAINED_POINTWISE_CACHE_LAYOUT_KEY,
        CUTE_CHAINED_VECTOR_GROUP_KEY,
    )
    if spec.cute_chained_preparation_pipeline_search_enabled:
        expected_suffix += (
            CUTE_CHAINED_PREPARATION_COHORTS_KEY,
            CUTE_CHAINED_PREPARATION_UNROLL_KEY,
            CUTE_CHAINED_REGISTER_ISLANDS_KEY,
        )
    else:
        assert CUTE_CHAINED_PREPARATION_COHORTS_KEY not in current_fields
        assert CUTE_CHAINED_PREPARATION_UNROLL_KEY not in current_fields
        assert CUTE_CHAINED_REGISTER_ISLANDS_KEY not in current_fields
    # These independent opt-ins were appended after the original cache schema.
    inactive_keys = (CUTE_NATIVE_MATMUL_METADATA_KEY,)
    if spec.cute_chained_pointwise_residency_search_enabled:
        inactive_keys += (CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY,)
    expected_suffix += inactive_keys
    # Existing grid-work-order was missing from the packed-DV2 expectation.
    if spec.cute_work_order_candidates and spec.cute_chunk_prefill_task_order is None:
        expected_suffix += (CUTE_GRID_WORK_ORDER_KEY,)
        assert current_fields[CUTE_GRID_WORK_ORDER_KEY].default() == tuple(
            "identity" for axis in spec.cute_work_order_axes
        )
    else:
        assert CUTE_GRID_WORK_ORDER_KEY not in current_fields
    expected_suffix += (CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY,)
    inactive_keys += (CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY,)
    assert tuple(current_fields)[-len(expected_suffix) :] == expected_suffix
    default = spec.default_config()
    for key in inactive_keys:
        assert current_fields[key].default() is False
        assert default.config.get(key, False) is False
        assert all(
            seed.config.get(key, False) is False for seed in spec.compiler_seed_configs
        )
    index = _flat_scalar_index(spec, KEY)
    seed_index = _flat_scalar_index(spec, CUTE_CHAINED_SEED_TILE_COLUMNS_KEY)
    old_fields = {key: field for key, field in current_fields.items() if key != KEY}
    with patch.object(spec, "_flat_fields", return_value=old_fields):
        old_pairs = ConfigGeneration(spec).seed_flat_config_pairs()
    new_pairs = ConfigGeneration(spec).seed_flat_config_pairs()
    assert len(new_pairs) == len(old_pairs)
    for (old_flat, old_config), (new_flat, new_config) in zip(
        old_pairs, new_pairs, strict=True
    ):
        assert new_flat[:index] + new_flat[index + 1 :] == old_flat
        assert new_flat[index] == 1 and new_flat[seed_index] == 0
        assert new_config == old_config
        assert all(new_config.config.get(key, False) is False for key in inactive_keys)


def test_legacy_override_removes_only_inherited_cache_count(overlap_bound):
    spec = overlap_bound.config_spec
    parent = helion.Config.from_dict(_resident_parent(overlap_bound).config | {KEY: 4})
    overrides = {_LEGACY_KEYS[0]: 4, _LEGACY_KEYS[1]: 72}
    generation = spec.create_config_generation(overrides=overrides)
    restored = generation.unflatten(generation.flatten(parent))
    assert SCHEDULE not in restored.config
    assert KEY not in restored.config and BUDGET not in restored.config
    assert restored[_LEGACY_KEYS[1]] == 72


@pytest.mark.parametrize("repair", (False, True))
def test_explicit_multi_cache_and_legacy_cap_conflict_cannot_be_repaired(
    overlap_bound, repair
):
    spec = overlap_bound.config_spec
    config = _resident_parent(overlap_bound).config | {KEY: 2, _LEGACY_KEYS[1]: 72}
    with pytest.raises(exc.InvalidConfig, match="register caps conflict"):
        spec.normalize(config, _fix_invalid=repair)
