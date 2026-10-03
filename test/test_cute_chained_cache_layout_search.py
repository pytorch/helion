from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_cache_set_search import cache_set_bound as cache_set_bound
from .test_cute_chained_legacy_family import overlap_bound as overlap_bound
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
from helion.autotuner.config_spec import (
    CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY as BUDGET,
)
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_ENTRIES_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_LAYOUT_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_COHORTS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_UNROLL_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_REGISTER_ISLANDS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SEED_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_VECTOR_GROUP_KEY
from helion.autotuner.config_spec import CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY
from helion.autotuner.config_spec import ConfigSpec

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


def _config(layout: object = "xor", **overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {"num_warps": 4, SCHEDULE: "coalesced", BUDGET: 4096, KEY: layout, **overrides}
    )


def test_public_constructor_default_and_family_request(
    cache_set_bound: BoundKernel,
) -> None:
    spec = cache_set_bound.config_spec
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, "auto")
    explicit = helion.Config(cute_chained_pointwise_cache_layout="auto")
    assert spec.normalized_config(explicit) == spec.normalized_config(helion.Config())
    assert KEY not in spec.normalized_config(explicit).config
    assert explicit[KEY] == "auto"
    assert not ConfigSpec._requests_cute_chained_loop({KEY: "auto"})
    assert ConfigSpec._requests_cute_chained_loop({KEY: "xor"})


@pytest.mark.parametrize("layout", ("auto", "xor"))
@pytest.mark.parametrize("budget", (4096, 16384))
def test_roundtrip_and_override(
    cache_set_bound: BoundKernel, layout: str, budget: int
) -> None:
    spec = cache_set_bound.config_spec
    requested = _config(layout, **{BUDGET: budget})
    normalized = spec.normalized_config(requested)
    generation = spec.create_config_generation()
    assert normalized.config.get(KEY, "auto") == layout
    assert (KEY in normalized.config) == (layout != "auto")
    index = _flat_scalar_index(spec, KEY)
    assert generation.flatten(requested)[index] == layout
    assert generation.flatten(normalized)[index] == layout
    assert generation.unflatten(generation.flatten(normalized)) == normalized
    overrides = spec.create_config_generation(overrides={KEY: layout, BUDGET: budget})
    parent = spec.normalized_config(_config("auto"))
    assert overrides.unflatten(overrides.flatten(parent)) == normalized


def _assert_old_prefix(spec: ConfigSpec) -> None:
    seeds = tuple(spec.compiler_seed_configs)
    snapshots = [copy.deepcopy(seed.config) for seed in seeds]
    default = spec.default_config()
    fields = spec._flat_fields()
    keys = tuple(fields)
    field_index = keys.index(KEY)
    suffix = (
        ()
        if spec.cute_chained_loop_search_enabled
        or not spec.cute_chained_tcgen05_search_enabled
        else (CUTE_CHAINED_SEED_TILE_COLUMNS_KEY,)
    )
    if spec.cute_chained_tcgen05_search_enabled:
        suffix += (CUTE_CHAINED_VECTOR_GROUP_KEY,)
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
        suffix += ("cute_chained_pointwise_cache_nested",)
    if "cute_chained_async_vector_store" in fields:
        suffix += ("cute_chained_async_vector_store",)
    if "cute_grid_work_order" in fields:
        suffix += ("cute_grid_work_order",)
    suffix += ("cute_chained_fragment_epilogues",)
    assert keys[field_index + 1 :] == suffix
    previous_key = (
        CUTE_CHAINED_SEED_TILE_COLUMNS_KEY
        if spec.cute_chained_loop_search_enabled
        and spec.cute_chained_tcgen05_search_enabled
        else CUTE_CHAINED_POINTWISE_CACHE_ENTRIES_KEY
    )
    assert keys[field_index - 1] == previous_key
    fragment = fields[KEY]
    assert isinstance(fragment, EnumFragment)
    assert fragment.default() == "auto"
    assert fragment.search_values() == ["auto", "xor"]
    assert spec.flat_key_layout()[field_index] == (KEY, 1, False)
    index = _flat_scalar_index(spec, KEY)
    generation = spec.create_config_generation()
    pairs = generation.seed_flat_config_pairs()
    with patch.object(
        spec, "_flat_fields", return_value={k: v for k, v in fields.items() if k != KEY}
    ):
        previous = spec.create_config_generation()
        old_pairs = previous.seed_flat_config_pairs()
        old_roundtrip = previous.unflatten(previous.flatten(default))
        flat = generation.flatten(default)
        assert flat[:index] + flat[index + 1 :] == previous.flatten(default)
        assert spec.default_config() == default
        for (flat, config), (old_flat, old_config) in zip(
            pairs, old_pairs, strict=True
        ):
            assert flat[:index] + flat[index + 1 :] == old_flat
            assert flat[index] == "auto"
            assert config == old_config
    assert KEY not in default.config
    assert all(KEY not in seed.config for seed in seeds)
    assert all(a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True))
    assert [seed.config for seed in seeds] == snapshots
    assert generation.unflatten(generation.flatten(default)) == old_roundtrip


def test_appended_coordinate_preserves_every_old_field_seed_and_default(
    cache_set_bound: BoundKernel,
) -> None:
    with patch.dict(
        cache_set_bound.config_spec.user_defined_tunables,
        {"user_choice": EnumFragment((7, 11))},
    ):
        _assert_old_prefix(cache_set_bound.config_spec)


def test_every_admitted_schedule_and_implicit_coalesced(
    cache_set_bound: BoundKernel,
) -> None:
    spec = cache_set_bound.config_spec
    generation = spec.create_config_generation()
    for schedule in spec._cute_chained_mma_schedules():
        config = spec.normalized_config(_config(**{SCHEDULE: schedule}))
        assert config[KEY] == "xor" and config[SCHEDULE] == schedule
        assert generation.unflatten(generation.flatten(config)) == config
    implicit = _config()
    implicit.config.pop(SCHEDULE)
    assert spec.normalized_config(implicit) == spec.normalized_config(_config())


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "layout", (None, True, False, 0, 1, 1.0, "", "row_major", "XOR", [], {})
)
def test_invalid_values_fail_before_repair(
    cache_set_bound: BoundKernel, layout: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be 'auto' or 'xor'"):
        cache_set_bound.config_spec.normalize(_config(layout), _fix_invalid=repair)


@pytest.mark.parametrize("repair", (False, True))
def test_positive_requires_budget_before_repair(
    cache_set_bound: BoundKernel, repair: bool
) -> None:
    for budget in ({}, {BUDGET: 0}):
        with pytest.raises(exc.InvalidConfig, match="positive pointwise cache bytes"):
            cache_set_bound.config_spec.normalize(
                {KEY: "xor", SCHEDULE: "coalesced", **budget}, _fix_invalid=repair
            )


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize("schedule", (None, "invalid", 0, False))
def test_bad_schedule_fails_before_repair(
    cache_set_bound: BoundKernel, schedule: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="admitted common MMA schedule"):
        cache_set_bound.config_spec.normalize(
            _config(**{SCHEDULE: schedule}), _fix_invalid=repair
        )


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "attribute,value",
    (
        ("backend_name", "triton"),
        ("cute_chained_matmul_search_enabled", False),
        ("cute_chained_pointwise_residency_search_enabled", False),
    ),
)
def test_missing_admission_rejects_and_removes_search(
    cache_set_bound: BoundKernel, attribute: str, value: object, repair: bool
) -> None:
    spec = cache_set_bound.config_spec
    with patch.object(spec, attribute, value):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig, match="reused common-region"):
            spec.normalize(_config(), _fix_invalid=repair)
        assert spec.normalized_config({KEY: "auto"}) == spec.normalized_config({})


@pytest.mark.parametrize("repair", (False, True))
def test_tcgen_admission_cannot_be_repaired(
    cache_set_bound: BoundKernel, repair: bool
) -> None:
    with (
        patch.object(
            cache_set_bound.config_spec, "cute_chained_tcgen05_search_enabled", False
        ),
        pytest.raises(exc.InvalidConfig, match="admitted common MMA schedule"),
    ):
        cache_set_bound.config_spec.normalize(
            _config(**{SCHEDULE: "tcgen05_tmem"}), _fix_invalid=repair
        )


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
def test_conflicting_families_reject(
    cache_set_bound: BoundKernel, overrides: dict[str, object], repair: bool
) -> None:
    with pytest.raises(
        exc.InvalidConfig,
        match="common contraction lowering|not implemented for contraction loops",
    ):
        cache_set_bound.config_spec.normalize(_config(**overrides), _fix_invalid=repair)


@pytest.mark.parametrize("repair", (False, True))
def test_legacy_prefill_rejects(cache_set_bound: BoundKernel, repair: bool) -> None:
    with (
        patch.object(
            cache_set_bound.config_spec,
            "cute_chunk_prefill_task_order",
            EnumFragment(("identity",)),
        ),
        pytest.raises(exc.InvalidConfig, match="explicit shared prefill family"),
    ):
        cache_set_bound.config_spec.normalize(_config(), _fix_invalid=repair)


def test_unflatten_cannot_activate_without_budget(cache_set_bound: BoundKernel) -> None:
    spec = cache_set_bound.config_spec
    generation = spec.create_config_generation()
    flat = generation.flatten(spec.normalized_config(_config("auto", **{BUDGET: 0})))
    flat[_flat_scalar_index(spec, KEY)] = "xor"
    with pytest.raises(exc.InvalidConfig, match="positive pointwise cache bytes"):
        generation.unflatten(flat)


def test_triton_and_unbound_cute_preserve_default_but_reject_xor(
    cache_set_bound: BoundKernel,
) -> None:
    with _cpu_codegen():
        specs = (
            ConfigSpec(backend=TritonBackend(), device=torch.device("cpu"), num_sm=148),
            ConfigSpec(
                backend=cache_set_bound.config_spec.backend,
                device=torch.device("cpu"),
                target_device_capability=(10, 3),
                num_sm=148,
            ),
        )
        assert not specs[0].supports_config_key(KEY)
        for spec in specs:
            assert KEY not in spec._flat_fields()
            assert spec.normalized_config({KEY: "auto"}) == spec.normalized_config({})
            for repair in (False, True):
                with pytest.raises(exc.InvalidConfig, match="reused common-region"):
                    spec.normalize(_config(), _fix_invalid=repair)


def test_legacy_defaults_and_seed_prefix_preserved(overlap_bound: BoundKernel) -> None:
    spec = overlap_bound.config_spec
    default = spec.default_config()
    assert SCHEDULE not in default.config
    assert spec.normalized_config(default.config | {KEY: "auto"}) == default
    _assert_old_prefix(spec)


def test_legacy_override_drops_only_inherited_layout(
    overlap_bound: BoundKernel,
) -> None:
    spec = overlap_bound.config_spec
    parent = spec.normalized_config(
        _config(block_sizes=[64], **{SCHEDULE: "tcgen05_tmem"})
    )
    overrides = {CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY: 4}
    generation = spec.create_config_generation(overrides=overrides)
    result = generation.unflatten(generation.flatten(parent))
    assert SCHEDULE not in result.config and KEY not in result.config
    explicit = spec.create_config_generation(overrides=overrides | {KEY: "xor"})
    result = explicit.unflatten(explicit.flatten(parent))
    assert result[SCHEDULE] == "tcgen05_tmem" and result[KEY] == "xor"
