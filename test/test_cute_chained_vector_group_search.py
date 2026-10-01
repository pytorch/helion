from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_legacy_family import overlap_bound as overlap_bound
from .test_cute_chained_residency_search import residency_bound as residency_bound
import helion
from helion import exc
from helion._compiler.triton.backend import TritonBackend
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import CUTE_CHAINED_COLLECTIVE_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_STMATRIX_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_LEAF_COUNT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_MMA_SCHEDULE_KEY as SCHEDULE
from helion.autotuner.config_spec import CUTE_CHAINED_NATIVE_VECTOR_READS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OPERAND_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_ENTRIES_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_VECTORIZE_KEY as VECTOR
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_COHORTS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_UNROLL_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_REGISTER_ISLANDS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_VECTOR_GROUP_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY
from helion.autotuner.config_spec import ConfigSpec

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


def _config(enabled: object = True, **overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "num_warps": 4,
            SCHEDULE: "tcgen05_tmem",
            VECTOR: True,
            KEY: enabled,
            **overrides,
        }
    )


def test_public_default_and_strict_family_signal(residency_bound: BoundKernel) -> None:
    spec = residency_bound.config_spec
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, False)
    field = spec._flat_fields()[KEY]
    assert isinstance(field, EnumFragment)
    assert field.default() is False
    assert field.search_values() == [False, True]
    explicit = helion.Config(cute_chained_vector_group=False)
    assert spec.normalized_config(explicit) == spec.normalized_config(helion.Config())
    assert explicit[KEY] is False
    assert KEY not in spec.default_config().config
    assert not ConfigSpec._requests_cute_chained_loop({KEY: False})
    assert ConfigSpec._requests_cute_chained_loop({KEY: True})


@pytest.mark.parametrize("enabled", (False, True))
def test_roundtrip_and_explicit_override(
    residency_bound: BoundKernel, enabled: bool
) -> None:
    spec = residency_bound.config_spec
    generation = spec.create_config_generation()
    requested = _config(enabled)
    normalized = spec.normalized_config(requested)
    assert normalized.config.get(KEY, False) is enabled
    assert (KEY in normalized.config) is enabled
    index = _flat_scalar_index(spec, KEY)
    flat = generation.flatten(normalized)
    assert flat[index] is enabled
    assert generation.flatten(requested)[index] is enabled
    assert generation.unflatten(flat) == normalized
    overrides = spec.create_config_generation(overrides={KEY: enabled})
    parent = spec.normalized_config(_config(False))
    assert overrides.unflatten(overrides.flatten(parent)) == normalized


def _assert_complete_old_prefix(spec: ConfigSpec) -> None:
    seeds = tuple(spec.compiler_seed_configs)
    snapshots = [copy.deepcopy(seed.config) for seed in seeds]
    default = spec.default_config()
    fields = spec._flat_fields()
    suffix = (KEY,)
    if not spec.cute_chained_loop_search_enabled:
        suffix += (CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY,)
    if spec.cute_chained_preparation_pipeline_search_enabled:
        suffix += (
            CUTE_CHAINED_PREPARATION_COHORTS_KEY,
            CUTE_CHAINED_PREPARATION_UNROLL_KEY,
            CUTE_CHAINED_REGISTER_ISLANDS_KEY,
            CUTE_CHAINED_LEAF_COUNT_KEY,
            CUTE_CHAINED_COLLECTIVE_RETENTION_KEY,
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
        )
    suffix += ("cute_native_matmul_metadata",)
    if spec.cute_chained_tcgen05_search_enabled and (
        not spec.cute_chained_loop_search_enabled
        or spec.cute_chained_preparation_pipeline_search_enabled
    ):
        suffix += ("cute_chained_drain_tile_columns",)
    if (
        spec.cute_chained_tcgen05_search_enabled
        and spec.cute_chained_preparation_pipeline_search_enabled
    ):
        suffix += ("cute_chained_island_consumers",)
    if spec.cute_chained_pointwise_residency_search_enabled:
        assert CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY in fields
        assert CUTE_CHAINED_POINTWISE_CACHE_ENTRIES_KEY in fields
        suffix += (CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY,)
        nested = fields[CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY]
        assert isinstance(nested, EnumFragment)
        assert nested.default() is False
        assert nested.search_values() == [False, True]
    else:
        assert CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY not in fields
    assert default.config.get(CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY, False) is False
    assert all(
        seed.config.get(CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY, False) is False
        for seed in seeds
    )
    if "cute_chained_async_vector_store" in fields:
        suffix += ("cute_chained_async_vector_store",)
    assert tuple(fields)[-len(suffix) :] == suffix
    index = _flat_scalar_index(spec, KEY)
    generation = spec.create_config_generation()
    pairs = generation.seed_flat_config_pairs()
    assert pairs
    assert all(
        config.config.get(CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY, False) is False
        for _, config in pairs
    )
    with patch.object(
        spec,
        "_flat_fields",
        return_value={key: field for key, field in fields.items() if key != KEY},
    ):
        previous = spec.create_config_generation()
        old_pairs = previous.seed_flat_config_pairs()
        assert spec.default_config() == default
        flat = generation.flatten(default)
        assert index == len(flat) - len(suffix)
        assert flat[:index] + flat[index + 1 :] == previous.flatten(default)
        previous_restored = previous.unflatten(previous.flatten(default))
        assert len(pairs) == len(old_pairs)
        for (flat, config), (old_flat, old_config) in zip(
            pairs, old_pairs, strict=True
        ):
            assert flat[:index] + flat[index + 1 :] == old_flat
            assert flat[index] is False
            assert config == old_config
    assert KEY not in default.config
    assert all(KEY not in seed.config for seed in seeds)
    assert all(a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True))
    assert [seed.config for seed in seeds] == snapshots
    assert spec.flat_key_layout()[-len(suffix)] == (KEY, 1, False)
    assert generation.unflatten(generation.flatten(default)) == previous_restored


def test_old_coordinate_and_complete_seed_prefix(residency_bound: BoundKernel) -> None:
    with patch.dict(
        residency_bound.config_spec.user_defined_tunables,
        {"user_choice": EnumFragment((7, 11))},
    ):
        _assert_complete_old_prefix(residency_bound.config_spec)


def test_inactive_source_is_identical(residency_bound: BoundKernel) -> None:
    explicit = _config(False)
    implicit = _config(False)
    implicit.config.pop(KEY)
    assert residency_bound.to_code(explicit) == residency_bound.to_code(implicit)


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize("enabled", (None, 0, 1, 0.0, 1.0, "", "true", [], {}))
def test_nonbool_rejected_before_repair(
    residency_bound: BoundKernel, enabled: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match=f"{KEY} must be bool"):
        residency_bound.config_spec.normalize(_config(enabled), _fix_invalid=repair)


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "overrides",
    (
        {SCHEDULE: None},
        {SCHEDULE: "coalesced"},
        {SCHEDULE: "cp_async"},
        {SCHEDULE: "tcgen05"},
        {SCHEDULE: "invalid"},
        {VECTOR: False},
        {VECTOR: None},
        {VECTOR: 1},
    ),
)
def test_requires_explicit_tcgen_and_vectorization_before_repair(
    residency_bound: BoundKernel, overrides: dict[str, object], repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
        residency_bound.config_spec.normalize(_config(**overrides), _fix_invalid=repair)


@pytest.mark.parametrize("key", (SCHEDULE, VECTOR))
def test_prerequisite_cannot_be_injected_by_normalization(
    residency_bound: BoundKernel, key: str
) -> None:
    requested = _config()
    requested.config.pop(key)
    with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
        residency_bound.config_spec.normalize(requested, _fix_invalid=True)


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "overrides",
    (
        {"cute_affine_scan_schedule": "warp"},
        {"cute_chained_direct_output": True},
        {"cute_chunk_recurrence_register_cap": 72},
        {"cute_chunk_recurrence_pipeline": "compact"},
    ),
)
def test_special_families_rejected_before_repair(
    residency_bound: BoundKernel, overrides: dict[str, object], repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="requires the common contraction"):
        residency_bound.config_spec.normalize(_config(**overrides), _fix_invalid=repair)


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "attribute,value",
    (
        ("backend_name", "triton"),
        ("cute_chained_matmul_search_enabled", False),
        ("cute_chained_tcgen05_search_enabled", False),
    ),
)
def test_discovery_runtime_and_backend_guards(
    residency_bound: BoundKernel, attribute: str, value: object, repair: bool
) -> None:
    spec = residency_bound.config_spec
    with patch.object(spec, attribute, value):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
            spec.normalize(_config(), _fix_invalid=repair)
        assert spec.normalized_config(
            helion.Config.from_dict({KEY: False})
        ) == spec.normalized_config(helion.Config())


def test_legacy_prefill_does_not_claim_active_group(
    residency_bound: BoundKernel,
) -> None:
    spec = residency_bound.config_spec
    with (
        patch.object(
            spec, "cute_chunk_prefill_task_order", EnumFragment(("identity",))
        ),
        pytest.raises(exc.InvalidConfig, match="requires the common contraction"),
    ):
        spec.normalize(_config(), _fix_invalid=True)


@pytest.mark.parametrize("schedule", ("coalesced", "tcgen05_tmem"))
def test_flat_activation_cannot_silently_repair_parent(
    residency_bound: BoundKernel, schedule: str
) -> None:
    spec = residency_bound.config_spec
    generation = spec.create_config_generation()
    parent = spec.normalized_config(
        _config(False, **{SCHEDULE: schedule, VECTOR: False})
    )
    flat = generation.flatten(parent)
    flat[_flat_scalar_index(spec, KEY)] = True
    with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
        generation.unflatten(flat)


def test_unbound_and_other_backend_do_not_advertise(
    residency_bound: BoundKernel,
) -> None:
    with _cpu_codegen():
        for backend in (residency_bound.config_spec.backend, TritonBackend()):
            spec = ConfigSpec(backend=backend, device=torch.device("cpu"), num_sm=148)
            assert KEY not in spec._flat_fields()
            assert KEY not in spec.default_config().config
            assert spec.normalized_config(
                helion.Config(cute_chained_vector_group=False)
            ) == spec.normalized_config(helion.Config())
            with pytest.raises(
                exc.InvalidConfig, match="requires explicit tcgen05_tmem"
            ):
                spec.normalized_config(_config())
        assert not spec.supports_config_key(KEY)


def test_legacy_default_and_all_six_seeds_preserved(overlap_bound: BoundKernel) -> None:
    spec = overlap_bound.config_spec
    _assert_complete_old_prefix(spec)
    default = spec.default_config()
    assert SCHEDULE not in default.config and KEY not in default.config
    generation = spec.create_config_generation()
    legacy = [
        config
        for _, config in generation.seed_flat_config_pairs()
        if SCHEDULE not in config.config
    ]
    assert len(legacy) == 6
    for config in legacy:
        assert KEY not in config.config
        assert generation.unflatten(generation.flatten(config)) == config


def test_legacy_override_removes_only_inherited_group(
    overlap_bound: BoundKernel,
) -> None:
    spec = overlap_bound.config_spec
    parent = spec.normalized_config(_config(block_sizes=[64]))
    legacy = {CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY: 4}
    generation = spec.create_config_generation(overrides=legacy)
    result = generation.unflatten(generation.flatten(parent))
    assert SCHEDULE not in result.config and KEY not in result.config
    explicit = spec.create_config_generation(overrides={**legacy, KEY: True})
    result = explicit.unflatten(explicit.flatten(parent))
    assert result[KEY] is True and result[SCHEDULE] == "tcgen05_tmem"
