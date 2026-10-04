from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_collective_retention_search import _config as _pipeline_config
from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_scan_producer import _scan_sequence
from .test_cute_chained_scan_producer import _sequence_args
import helion
from helion import exc
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import BACKEND_SPECIFIC_KEYS
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_STMATRIX_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_NATIVE_VECTOR_READS_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import VALID_KEYS
from helion.autotuner.config_spec import ConfigSpec


def _config(enabled=False, **overrides):
    return _pipeline_config(
        **{
            "cute_chained_operand_retention": True,
            "cute_chained_pointwise_vectorize": True,
            KEY: enabled,
            **overrides,
        }
    )


def test_registered_default_preserves_seed_and_user_coordinate_prefix(retention_bound):
    spec = retention_bound.config_spec
    assert KEY in VALID_KEYS and KEY in BACKEND_SPECIFIC_KEYS
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, False)
    explicit = helion.Config.from_dict({KEY: False})
    assert spec.normalized_config(explicit) == spec.normalized_config(helion.Config())
    assert KEY not in spec.normalized_config(explicit).config
    assert not ConfigSpec._requests_cute_chained_loop({KEY: False})
    assert ConfigSpec._requests_cute_chained_loop({KEY: True})
    seeds = tuple(spec.compiler_seed_configs)
    snapshots = [copy.deepcopy(seed.config) for seed in seeds]
    with patch.dict(
        spec.user_defined_tunables, {"native_read_user": EnumFragment((3, 5))}
    ):
        fields = spec._flat_fields()
        assert tuple(fields)[-15:] == (
            KEY,
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
        assert spec.flat_key_layout()[-15] == (KEY, 1, False)
        index = _flat_scalar_index(spec, KEY)
        field = fields[KEY]
        assert isinstance(field, EnumFragment)
        assert field.search_values() == [False, True]
        generation = spec.create_config_generation()
        default = spec.default_config()
        pairs = generation.seed_flat_config_pairs()
        with patch.object(
            spec,
            "_flat_fields",
            return_value={k: v for k, v in fields.items() if k != KEY},
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
    assert [seed.config for seed in seeds] == snapshots
    assert all(a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True))


@pytest.mark.parametrize("enabled", (False, True))
def test_bool_roundtrip_and_independent_override(retention_bound, enabled):
    spec = retention_bound.config_spec
    normalized = spec.normalized_config(_config(enabled))
    assert normalized.config.get(KEY, False) is enabled
    generation = spec.create_config_generation()
    assert generation.unflatten(generation.flatten(normalized)) == normalized
    assert generation.flatten(normalized)[_flat_scalar_index(spec, KEY)] is enabled
    override = spec.create_config_generation(overrides={KEY: enabled})
    assert (
        override.unflatten(override.flatten(spec.normalized_config(_config())))
        == normalized
    )


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize("value", (None, 0, 1, -1, 0.0, "true", [], {}))
def test_only_boolean_values_even_under_repair(retention_bound, repair, value):
    with pytest.raises(exc.InvalidConfig, match=KEY):
        retention_bound.config_spec.normalize(_config(value), _fix_invalid=repair)


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "key",
    (
        "cute_chained_mma_schedule",
        "cute_chained_preparation_pipeline",
        "cute_chained_operand_retention",
        "cute_chained_pointwise_vectorize",
    ),
)
def test_missing_prerequisites_are_not_invented(retention_bound, repair, key):
    requested = _config(True)
    requested.config.pop(key)
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(requested, _fix_invalid=repair)
    assert key not in requested.config


@pytest.mark.parametrize("backend", ("triton", "pallas"))
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
    (
        "cute_chained_loop_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
    ),
)
def test_search_requires_common_pipeline_discovery(retention_bound, flag):
    spec = retention_bound.config_spec
    with patch.object(spec, flag, False):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(True))


def test_request_does_not_select_ownership_or_change_math(retention_bound):
    spec = retention_bound.config_spec
    before = spec.normalized_config(_config())
    after = spec.normalized_config(_config(True))
    assert after.config == before.config | {KEY: True}
    assert all(
        key not in after.config
        for key in (
            "cute_chained_frontier_tile_columns",
            "cute_chained_vector_group",
            "cute_chained_register_islands",
            "fast_math",
        )
    )


def _leaf_scan_config():
    config = _config(
        True,
        cute_chained_scan_schedule="warp",
        cute_chained_scan_producer_retention=True,
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_leaf_count=2,
    )
    config.config.pop("cute_chained_operand_retention")
    return config


@pytest.fixture
def leaf_scan_bound():
    values = _sequence_args()
    with _cpu_codegen():
        bound = _scan_sequence._bind_isolated(values)
        with bound.env.use_runtime_arg_values(_runtime_values(_scan_sequence, values)):
            yield bound


@pytest.mark.parametrize("repair", (False, True))
def test_native_scan_leaves_do_not_require_unrelated_operand_retention(
    leaf_scan_bound, repair
):
    requested = _leaf_scan_config()
    leaf_scan_bound.config_spec.normalize(requested, _fix_invalid=repair)
    assert requested.config[KEY] is True
    assert "cute_chained_operand_retention" not in requested.config


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "key",
    (
        "cute_chained_scan_schedule",
        "cute_chained_scan_producer_retention",
        "cute_chained_leaf_pipeline",
        "cute_chained_pointwise_vectorize",
    ),
)
def test_native_scan_leaf_prerequisites_are_not_invented(leaf_scan_bound, repair, key):
    requested = _leaf_scan_config()
    requested.config.pop(key)
    with pytest.raises(exc.InvalidConfig):
        leaf_scan_bound.config_spec.normalize(requested, _fix_invalid=repair)
    assert key not in requested.config
