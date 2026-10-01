from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_legacy_family import overlap_bound as overlap_bound
from .test_cute_chained_mma_selection_search import _selection_loop
from .test_cute_chained_residency_search import _loop_residency
from .test_cute_chained_residency_search import _root_residency
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
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_LAYOUT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_COHORTS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_UNROLL_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_REGISTER_ISLANDS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SEED_TILE_COLUMNS_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_VECTOR_GROUP_KEY
from helion.autotuner.config_spec import CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY
from helion.autotuner.config_spec import CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY
from helion.autotuner.config_spec import ConfigSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module", params=(False, True), ids=("plain", "cache"))
def seed_loop_bound(request: pytest.FixtureRequest) -> Iterator[BoundKernel]:
    with _cpu_codegen():
        if request.param:
            yield _loop_residency._bind_isolated(
                (
                    torch.empty((3, 128, 16), dtype=torch.bfloat16),
                    torch.empty((3, 16, 32), dtype=torch.bfloat16),
                    torch.empty((128, 32), dtype=torch.float32),
                )
            )
        else:
            yield _selection_loop._bind_isolated(
                (
                    torch.empty((3, 128, 16), dtype=torch.bfloat16),
                    torch.empty((3, 16, 32), dtype=torch.bfloat16),
                    torch.empty((128, 32), dtype=torch.float32),
                    128,
                    False,
                )
            )


def _config(columns: object = 32, **overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {"num_warps": 4, SCHEDULE: "tcgen05_tmem", KEY: columns, **overrides}
    )


def test_strict_field_default_and_public_constructor(
    seed_loop_bound: BoundKernel,
) -> None:
    spec = seed_loop_bound.config_spec
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, 0)
    field = spec._flat_fields()[KEY]
    assert isinstance(field, EnumFragment)
    assert field.default() == 0
    assert field.search_values() == [0, 32, 64]
    requested = helion.Config(
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_seed_tile_columns=0,
    )
    implicit = helion.Config(num_warps=4, cute_chained_mma_schedule="tcgen05_tmem")
    assert spec.normalized_config(requested) == spec.normalized_config(implicit)
    assert KEY not in spec.normalized_config(requested).config
    assert requested[KEY] == 0
    assert not ConfigSpec._requests_cute_chained_loop({KEY: 0})
    assert ConfigSpec._requests_cute_chained_loop({KEY: 32})


@pytest.mark.parametrize("columns", (0, 32, 64))
def test_roundtrip_and_explicit_override(
    seed_loop_bound: BoundKernel, columns: int
) -> None:
    spec = seed_loop_bound.config_spec
    generation = spec.create_config_generation()
    requested = _config(columns)
    normalized = spec.normalized_config(requested)
    assert normalized.config.get(KEY, 0) == columns
    assert (KEY in normalized.config) is (columns != 0)
    flat = generation.flatten(normalized)
    index = _flat_scalar_index(spec, KEY)
    assert flat[index] == columns
    assert generation.flatten(requested)[index] == columns
    assert generation.unflatten(flat) == normalized
    overrides = spec.create_config_generation(overrides={KEY: columns})
    restored = overrides.unflatten(
        overrides.flatten(spec.normalized_config(_config(0)))
    )
    assert restored == normalized


def _assert_old_prefix_and_seeds(spec: ConfigSpec) -> None:
    seeds = tuple(spec.compiler_seed_configs)
    snapshots = [copy.deepcopy(seed.config) for seed in seeds]
    default = spec.default_config()
    fields = spec._flat_fields()
    suffix = (KEY,)
    if spec.cute_chained_pointwise_residency_search_enabled:
        suffix = (
            CUTE_CHAINED_POINTWISE_CACHE_ENTRIES_KEY,
            KEY,
            CUTE_CHAINED_POINTWISE_CACHE_LAYOUT_KEY,
        )
    suffix += (CUTE_CHAINED_VECTOR_GROUP_KEY,)
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
    if "cute_grid_work_order" in fields:
        suffix += ("cute_grid_work_order",)
    suffix += ("cute_chained_fragment_epilogues",)
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
        assert flat[:index] + flat[index + 1 :] == previous.flatten(default)
        assert len(pairs) == len(old_pairs)
        for (flat, config), (old_flat, old_config) in zip(
            pairs, old_pairs, strict=True
        ):
            assert flat[:index] + flat[index + 1 :] == old_flat and flat[index] == 0
            assert config == old_config
    assert KEY not in default.config
    assert all(KEY not in seed.config for seed in seeds)
    assert all(a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True))
    assert [seed.config for seed in seeds] == snapshots
    assert (KEY, 1, False) in spec.flat_key_layout()
    assert generation.unflatten(generation.flatten(default)) == default


def test_appended_after_every_old_coordinate_and_user_tunable(
    seed_loop_bound: BoundKernel,
) -> None:
    spec = seed_loop_bound.config_spec
    with patch.dict(spec.user_defined_tunables, {"user_choice": EnumFragment((7, 11))}):
        _assert_old_prefix_and_seeds(spec)


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "columns", (True, False, None, -1, 1, 16, 128, 32.0, 0.0, "32", [], {})
)
def test_invalid_values_reject_before_repair(
    seed_loop_bound: BoundKernel, columns: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match=f"{KEY} must be 0, 32 or 64"):
        seed_loop_bound.config_spec.normalize(_config(columns), _fix_invalid=repair)


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "overrides",
    (
        {SCHEDULE: None},
        {SCHEDULE: "coalesced"},
        {SCHEDULE: "cp_async"},
        {SCHEDULE: "tcgen05"},
        {SCHEDULE: "invalid"},
        {"cute_affine_scan_schedule": "warp"},
        {"cute_chained_direct_output": True},
        {CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY: 72},
        {"cute_chunk_recurrence_pipeline": "compact"},
    ),
)
def test_incompatible_families_fail_before_repair(
    seed_loop_bound: BoundKernel, overrides: dict[str, object], repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
        seed_loop_bound.config_spec.normalize(_config(**overrides), _fix_invalid=repair)


@pytest.mark.parametrize("repair", (False, True))
def test_positive_requires_explicit_schedule(
    seed_loop_bound: BoundKernel, repair: bool
) -> None:
    requested = _config()
    requested.config.pop(SCHEDULE)
    with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
        seed_loop_bound.config_spec.normalize(requested, _fix_invalid=repair)


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "attribute,value",
    (
        ("backend_name", "triton"),
        ("cute_chained_matmul_search_enabled", False),
        ("cute_chained_tcgen05_search_enabled", False),
    ),
)
def test_discovery_is_required_for_positive_and_flat_search(
    seed_loop_bound: BoundKernel, attribute: str, value: object, repair: bool
) -> None:
    spec = seed_loop_bound.config_spec
    with patch.object(spec, attribute, value):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
            spec.normalize(_config(), _fix_invalid=repair)
        assert spec.normalized_config(
            helion.Config.from_dict({KEY: 0})
        ) == spec.normalized_config(helion.Config())


@pytest.mark.parametrize("repair", (False, True))
def test_legacy_prefill_cannot_consume_seed_tiles(
    seed_loop_bound: BoundKernel, repair: bool
) -> None:
    spec = seed_loop_bound.config_spec
    with (
        patch.object(
            spec, "cute_chunk_prefill_task_order", EnumFragment(("identity",))
        ),
        pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"),
    ):
        spec.normalize(_config(), _fix_invalid=repair)


def test_inactive_flat_parent_cannot_enable_seed_tiles(
    seed_loop_bound: BoundKernel,
) -> None:
    spec = seed_loop_bound.config_spec
    generation = spec.create_config_generation()
    flat = generation.flatten(
        spec.normalized_config(_config(0, **{SCHEDULE: "coalesced"}))
    )
    flat[_flat_scalar_index(spec, KEY)] = 32
    with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
        generation.unflatten(flat)


def test_root_admits_seed_tiles_but_triton_rejects() -> None:
    with _cpu_codegen():
        root = _root_residency._bind_isolated(
            (
                torch.empty((128, 16), dtype=torch.bfloat16),
                torch.empty((16, 32), dtype=torch.bfloat16),
            )
        )
        triton = ConfigSpec(
            backend=TritonBackend(), device=torch.device("cpu"), num_sm=148
        )
        assert not triton.supports_config_key(KEY)
        for spec in (root.config_spec, triton):
            assert spec.normalized_config(
                helion.Config.from_dict({KEY: 0})
            ) == spec.normalized_config(helion.Config())
        assert KEY in root.config_spec._flat_fields()
        assert not root.config_spec.cute_chained_loop_search_enabled
        assert root.config_spec.normalized_config(_config())[KEY] == 32
        assert KEY not in triton._flat_fields()
        for repair in (False, True):
            with pytest.raises(
                exc.InvalidConfig, match="requires explicit tcgen05_tmem"
            ):
                triton.normalize(_config(), _fix_invalid=repair)


def test_legacy_defaults_and_entire_seed_prefix_are_unchanged(
    overlap_bound: BoundKernel,
) -> None:
    spec = overlap_bound.config_spec
    default = spec.default_config()
    assert SCHEDULE not in default.config
    explicit_zero = helion.Config.from_dict(default.config | {KEY: 0})
    assert spec.normalized_config(explicit_zero) == default
    _assert_old_prefix_and_seeds(spec)
    pairs = spec.create_config_generation().seed_flat_config_pairs()
    legacy = [config for _, config in pairs if SCHEDULE not in config.config]
    assert len(legacy) == 6
    assert all(KEY not in config.config for config in legacy)


def test_legacy_override_discards_only_inherited_seed_tiles(
    overlap_bound: BoundKernel,
) -> None:
    spec = overlap_bound.config_spec
    parent = spec.normalized_config(_config(block_sizes=[64]))
    overrides = {CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY: 4}
    generation = spec.create_config_generation(overrides=overrides)
    restored = generation.unflatten(generation.flatten(parent))
    assert SCHEDULE not in restored.config and KEY not in restored.config
    explicit = spec.create_config_generation(overrides=overrides | {KEY: 32})
    result = explicit.unflatten(explicit.flatten(parent))
    assert result[SCHEDULE] == "tcgen05_tmem" and result[KEY] == 32
