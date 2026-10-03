from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_collective_retention_search import _config as _pipeline_config
from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
import helion
from helion import exc
from helion._compiler.pallas.backend import PallasBackend
from helion._compiler.triton.backend import TritonBackend
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import BACKEND_SPECIFIC_KEYS
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_STMATRIX_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_TILE_COLUMNS_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_NATIVE_VECTOR_READS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OPERAND_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import VALID_CUTE_CHAINED_FRONTIER_TILE_COLUMNS
from helion.autotuner.config_spec import VALID_KEYS
from helion.autotuner.config_spec import ConfigSpec

if TYPE_CHECKING:
    from helion._compiler.backend import Backend
    from helion.runtime.kernel import BoundKernel


def _config(columns: object = 0, **overrides: object) -> helion.Config:
    return _pipeline_config(
        **{
            "cute_chained_pointwise_vectorize": True,
            "cute_chained_vector_group": True,
            KEY: columns,
            **overrides,
        }
    )


def test_registered_default_is_removed_without_family_discovery(
    retention_bound: BoundKernel,
) -> None:
    spec = retention_bound.config_spec
    assert KEY in VALID_KEYS and KEY in BACKEND_SPECIFIC_KEYS
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, 0)
    assert not ConfigSpec._requests_cute_chained_loop({KEY: 0})
    assert ConfigSpec._requests_cute_chained_loop({KEY: 32})
    explicit = helion.Config(cute_chained_frontier_tile_columns=0)
    assert spec.normalized_config(explicit) == spec.normalized_config(helion.Config())
    assert KEY not in spec.normalized_config(explicit).config
    for flag in (
        "cute_chained_loop_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
    ):
        with patch.object(spec, flag, False):
            assert KEY not in spec._flat_fields()
            assert spec.normalized_config(explicit) == spec.normalized_config(
                helion.Config()
            )


def test_last_coordinate_preserves_every_prior_seed_default_and_user_field(
    retention_bound: BoundKernel,
) -> None:
    spec = retention_bound.config_spec
    seeds = tuple(spec.compiler_seed_configs)
    snapshots = [copy.deepcopy(seed.config) for seed in seeds]
    with patch.dict(
        spec.user_defined_tunables, {"ownership_choice": EnumFragment((3, 5))}
    ):
        fields = spec._flat_fields()
        assert tuple(fields)[-17:] == (
            CUTE_CHAINED_OPERAND_RETENTION_KEY,
            KEY,
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
        assert spec.flat_key_layout()[-16] == (KEY, 1, False)
        index = _flat_scalar_index(spec, KEY)
        field = fields[KEY]
        assert isinstance(field, EnumFragment)
        assert field.search_values() == [0, 8, 16, 32, 64, 128, 256]
        assert field.default() == 0
        assert field.search_values() == list(VALID_CUTE_CHAINED_FRONTIER_TILE_COLUMNS)
        generation = spec.create_config_generation()
        default = spec.default_config()
        pairs = generation.seed_flat_config_pairs()
        with patch.object(
            spec,
            "_flat_fields",
            return_value={k: v for k, v in fields.items() if k != KEY},
        ):
            prior = spec.create_config_generation()
            flat = generation.flatten(default)
            assert flat[:index] + flat[index + 1 :] == prior.flatten(default)
            assert flat[index] == 0
            assert spec.default_config() == default
            for (flat, config), (old_flat, old_config) in zip(
                pairs, prior.seed_flat_config_pairs(), strict=True
            ):
                assert flat[:index] + flat[index + 1 :] == old_flat
                assert flat[index] == 0
                assert config == old_config
        assert KEY not in default.config
    assert [seed.config for seed in seeds] == snapshots
    assert all(a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True))


@pytest.mark.parametrize("columns", [0, 8, 16, 32, 64, 128, 256])
def test_choices_roundtrip_partial_flatten_and_explicit_override(
    retention_bound: BoundKernel, columns: int
) -> None:
    spec = retention_bound.config_spec
    requested = _config(columns)
    normalized = spec.normalized_config(requested)
    assert normalized.config.get(KEY, 0) == columns
    assert (KEY in normalized.config) is (columns != 0)
    generation = spec.create_config_generation()
    flat = generation.flatten(normalized)
    assert flat[_flat_scalar_index(spec, KEY)] == columns
    assert generation.flatten(requested) == flat
    assert generation.unflatten(flat) == normalized
    override = spec.create_config_generation(overrides={KEY: columns})
    assert (
        override.unflatten(override.flatten(spec.normalized_config(_config())))
        == normalized
    )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "value", [None, False, True, -1, 1, 7, 24, 512, 0.0, 32.0, "32", [], {}]
)
def test_strict_type_and_value_validation_precedes_repair(
    retention_bound: BoundKernel, repair: bool, value: object
) -> None:
    with pytest.raises(exc.InvalidConfig, match=KEY):
        retention_bound.config_spec.normalize(_config(value), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "key",
    [
        "cute_chained_mma_schedule",
        "cute_chained_preparation_pipeline",
        "cute_chained_pointwise_vectorize",
        "cute_chained_vector_group",
    ],
)
def test_missing_explicit_prerequisites_are_not_repaired(
    retention_bound: BoundKernel, repair: bool, key: str
) -> None:
    requested = _config(32)
    requested.config.pop(key)
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(requested, _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"cute_chained_mma_schedule": "coalesced"},
        {"cute_chained_preparation_pipeline": False},
        {"cute_chained_pointwise_vectorize": False},
        {"cute_chained_vector_group": False},
        {"cute_affine_scan_schedule": "associative"},
        {"cute_chained_direct_output": True},
        {"cute_chunk_recurrence_register_cap": 72},
    ],
)
def test_conflicting_families_and_disabled_prerequisites_fail_closed(
    retention_bound: BoundKernel, repair: bool, overrides: dict[str, object]
) -> None:
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(
            _config(32, **overrides), _fix_invalid=repair
        )


@pytest.mark.parametrize(
    "flag",
    [
        "cute_chained_loop_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
    ],
)
def test_active_request_requires_structural_discovery(
    retention_bound: BoundKernel, flag: str
) -> None:
    spec = retention_bound.config_spec
    with patch.object(spec, flag, False):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(32))


@pytest.mark.parametrize("backend", [TritonBackend(), PallasBackend()])
def test_other_backends_only_accept_stripped_zero(backend: Backend) -> None:
    spec = ConfigSpec(backend=backend, device=torch.device("cpu"), num_sm=148)
    assert not spec.supports_config_key(KEY)
    default: dict[str, object] = {KEY: 0}
    spec.normalize(default)
    assert KEY not in default
    assert KEY not in spec._flat_fields()
    with pytest.raises(exc.InvalidConfig):
        spec.normalize(_config(32))


def test_request_does_not_inject_unrelated_optimizations(
    retention_bound: BoundKernel,
) -> None:
    spec = retention_bound.config_spec
    ordinary = spec.normalized_config(_config())
    retiled = spec.normalized_config(_config(32))
    assert retiled.config == ordinary.config | {KEY: 32}
    assert all(
        key not in retiled.config
        for key in (
            "cute_chained_operand_retention",
            "cute_chained_register_islands",
            "cute_chained_leaf_count",
            "cute_chained_collective_retention",
            "fast_math",
        )
    )
