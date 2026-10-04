from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_search import _loop_search
from .test_cute_chained_residency_search import _loop_residency
from .test_cute_chained_residency_search import _root_residency
import helion
from helion import exc
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
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_ENTRIES_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_LAYOUT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_COHORTS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_UNROLL_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_REGISTER_ISLANDS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SEED_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_VECTOR_GROUP_KEY
from helion.autotuner.config_spec import ConfigSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module", params=["root", "loop"])
def cache_set_bound(request: pytest.FixtureRequest) -> Iterator[BoundKernel]:
    with _cpu_codegen():
        if request.param == "root":
            yield _root_residency._bind_isolated(
                (
                    torch.empty((128, 16), dtype=torch.bfloat16),
                    torch.empty((16, 32), dtype=torch.bfloat16),
                )
            )
        else:
            yield _loop_residency._bind_isolated(
                (
                    torch.empty((3, 128, 16), dtype=torch.bfloat16),
                    torch.empty((3, 16, 32), dtype=torch.bfloat16),
                    torch.empty((128, 32), dtype=torch.float32),
                )
            )


def _config(entries: object = 2, **overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {"num_warps": 4, SCHEDULE: "coalesced", BUDGET: 4096, KEY: entries, **overrides}
    )


def _flat_scalar_index(spec: ConfigSpec, wanted: str) -> int:
    index = 0
    for key, count, sequence in spec.flat_key_layout():
        if key == wanted:
            assert count == 1 and not sequence
            return index
        index += count
    raise AssertionError(f"Missing scalar coordinate: {wanted}")


def test_constructor_and_missing_field_defaults(cache_set_bound: BoundKernel) -> None:
    spec = cache_set_bound.config_spec
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, 1)
    explicit = helion.Config(
        num_warps=4,
        cute_chained_mma_schedule="coalesced",
        cute_chained_pointwise_cache_entries=1,
    )
    implicit = helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced")
    assert spec.normalized_config(explicit) == spec.normalized_config(implicit)
    assert KEY not in spec.normalized_config(explicit).config
    assert explicit[KEY] == 1  # Normalization does not mutate caller-owned configs.
    default = spec.default_config()
    assert KEY not in default.config
    with patch.object(spec, "cute_chained_pointwise_residency_search_enabled", False):
        assert spec.default_config() == default


def test_unbound_spec_defaults_do_not_expose_cache_set(
    cache_set_bound: BoundKernel,
) -> None:
    spec = ConfigSpec(
        backend=cache_set_bound.config_spec.backend,
        device=torch.device("cpu"),
        target_device_capability=(10, 3),
        num_sm=148,
    )
    assert not spec.cute_chained_pointwise_residency_search_enabled
    assert KEY not in spec._flat_fields()
    assert KEY not in spec.default_config().config
    explicit = helion.Config(cute_chained_pointwise_cache_entries=1)
    assert spec.normalized_config(explicit) == spec.normalized_config(helion.Config())


@pytest.mark.parametrize("entries", [1, 2, 4])
@pytest.mark.parametrize("budget", [4096, 16384])
def test_simple_flatten_and_normalized_roundtrip(
    cache_set_bound: BoundKernel, entries: int, budget: int
) -> None:
    spec = cache_set_bound.config_spec
    generation = spec.create_config_generation()
    requested = _config(entries, **{BUDGET: budget})
    normalized = spec.normalized_config(requested)
    assert normalized.config.get(KEY, 1) == entries
    assert (KEY in normalized.config) is (entries != 1)
    # Partial configs and normalized configs must encode this named coordinate
    # identically, even though default normalization removes the public key.
    index = _flat_scalar_index(spec, KEY)
    assert generation.flatten(requested)[index] == entries
    flat = generation.flatten(normalized)
    assert flat[index] == entries
    assert generation.unflatten(flat) == normalized


def test_appended_coordinate_preserves_old_order(cache_set_bound: BoundKernel) -> None:
    spec = cache_set_bound.config_spec
    with patch.dict(spec.user_defined_tunables, {"user_choice": EnumFragment((7, 11))}):
        fields = spec._flat_fields()
        suffix = (
            (
                KEY,
                CUTE_CHAINED_SEED_TILE_COLUMNS_KEY,
                CUTE_CHAINED_POINTWISE_CACHE_LAYOUT_KEY,
            )
            if spec.cute_chained_loop_search_enabled
            else (
                KEY,
                CUTE_CHAINED_POINTWISE_CACHE_LAYOUT_KEY,
                CUTE_CHAINED_SEED_TILE_COLUMNS_KEY,
            )
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
        suffix += ("cute_chained_fragment_epilogues",)
        assert tuple(fields)[-len(suffix) :] == suffix
        field = fields[KEY]
        assert isinstance(field, EnumFragment)
        assert field.default() == 1
        assert field.search_values() == [1, 2, 4]
        assert spec.flat_key_layout()[-len(suffix) :] == [
            (key, 1, False) for key in suffix
        ]
        if spec.cute_chained_loop_search_enabled:
            seed_tiles = fields[CUTE_CHAINED_SEED_TILE_COLUMNS_KEY]
            assert isinstance(seed_tiles, EnumFragment)
            assert seed_tiles.default() == 0
            assert seed_tiles.search_values() == [0, 32, 64]
        with patch.object(
            spec, "cute_chained_pointwise_residency_search_enabled", False
        ):
            without_cache = spec._flat_fields()
        assert [
            key
            for key in fields
            if key
            not in (
                BUDGET,
                KEY,
                CUTE_CHAINED_POINTWISE_CACHE_LAYOUT_KEY,
                "cute_chained_pointwise_cache_nested",
            )
        ] == list(without_cache)
        # The existing budget retains its original position immediately after
        # the other common scratch/scan options, ahead of root/loop schedules.
        prefix = ["block_sizes", "num_warps", SCHEDULE]
        if spec.cute_chained_scratch_layout_search_enabled:
            prefix.append("cute_chained_scratch_layout")
        if spec.cute_chained_scan_search_enabled:
            prefix.append("cute_chained_scan_schedule")
        assert list(fields)[: len(prefix) + 1] == [*prefix, BUDGET]


def test_override_is_explicit_and_default_seeds_remain_single_entry(
    cache_set_bound: BoundKernel,
) -> None:
    spec = cache_set_bound.config_spec
    old_default = spec.default_config()
    generation = spec.create_config_generation(
        overrides={KEY: 4, BUDGET: 4096, SCHEDULE: "coalesced"}
    )
    restored = generation.unflatten(generation.flatten(old_default))
    assert restored[KEY] == 4
    assert restored[BUDGET] == 4096
    assert spec.default_config() == old_default
    assert all(KEY not in seed.config for seed in spec.compiler_seed_configs)


def test_all_admitted_schedules_roundtrip(cache_set_bound: BoundKernel) -> None:
    spec = cache_set_bound.config_spec
    generation = spec.create_config_generation()
    for schedule in spec._cute_chained_mma_schedules():
        config = spec.normalized_config(_config(**{SCHEDULE: schedule}))
        assert config[KEY] == 2
        assert config[SCHEDULE] == schedule
        assert generation.unflatten(generation.flatten(config)) == config


def test_explicit_tcgen_requires_admission_before_repair(
    cache_set_bound: BoundKernel,
) -> None:
    spec = cache_set_bound.config_spec
    with patch.object(spec, "cute_chained_tcgen05_search_enabled", False):
        for repair in (False, True):
            with pytest.raises(exc.InvalidConfig, match="admitted common MMA schedule"):
                spec.normalize(
                    _config(**{SCHEDULE: "tcgen05_tmem"}), _fix_invalid=repair
                )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("entries", [True, False, None, -1, 0, 3, 8, 1.0, 2.0, "2"])
def test_invalid_entries_reject_before_repair(
    cache_set_bound: BoundKernel, entries: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be 1, 2 or 4"):
        cache_set_bound.config_spec.normalize(_config(entries), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("entries", [2, 4])
def test_inactive_budget_cannot_be_repaired_into_cache_set(
    cache_set_bound: BoundKernel, entries: int, repair: bool
) -> None:
    spec = cache_set_bound.config_spec
    for budget in ({BUDGET: 0}, {}):
        values: dict[str, object] = {KEY: entries, SCHEDULE: "coalesced", **budget}
        with pytest.raises(exc.InvalidConfig, match="positive pointwise cache bytes"):
            spec.normalize(values, _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("schedule", [None, "not-a-schedule", 1, False])
def test_invalid_or_legacy_schedule_cannot_be_repaired(
    cache_set_bound: BoundKernel, schedule: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="admitted common MMA schedule"):
        cache_set_bound.config_spec.normalize(
            _config(**{SCHEDULE: schedule}), _fix_invalid=repair
        )


@pytest.mark.parametrize("repair", [False, True])
def test_legacy_prefill_rejects(cache_set_bound: BoundKernel, repair: bool) -> None:
    spec = cache_set_bound.config_spec
    with (
        patch.object(spec, "cute_chunk_prefill_task_order", object()),
        pytest.raises(exc.InvalidConfig, match="explicit shared prefill family"),
    ):
        spec.normalize(_config(), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [{"cute_affine_scan_schedule": "warp"}, {"cute_chained_direct_output": True}],
)
def test_other_lowering_rejects(
    cache_set_bound: BoundKernel, overrides: dict[str, object], repair: bool
) -> None:
    with pytest.raises(
        exc.InvalidConfig,
        match="common contraction lowering|not implemented for contraction loops",
    ):
        cache_set_bound.config_spec.normalize(_config(**overrides), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "attribute,value",
    [
        ("backend_name", "triton"),
        ("cute_chained_matmul_search_enabled", False),
        ("cute_chained_pointwise_residency_search_enabled", False),
    ],
)
def test_missing_admission_rejects(
    cache_set_bound: BoundKernel, attribute: str, value: object, repair: bool
) -> None:
    spec = cache_set_bound.config_spec
    with (
        patch.object(spec, attribute, value),
        pytest.raises(exc.InvalidConfig, match="reused common-region"),
    ):
        spec.normalize(_config(), _fix_invalid=repair)


def test_unflatten_rejects_inactive_coordinate(cache_set_bound: BoundKernel) -> None:
    spec = cache_set_bound.config_spec
    generation = spec.create_config_generation()
    flat = generation.flatten(spec.normalized_config(_config(1, **{BUDGET: 0})))
    flat[_flat_scalar_index(spec, KEY)] = 2
    with pytest.raises(exc.InvalidConfig, match="positive pointwise cache bytes"):
        generation.unflatten(flat)


def test_non_reused_loop_keeps_its_search_unchanged() -> None:
    with _cpu_codegen():
        bound = _loop_search._bind_isolated(
            (
                torch.empty((3, 128, 32), dtype=torch.bfloat16),
                torch.empty((3, 32, 128), dtype=torch.bfloat16),
                torch.empty((3, 32, 128), dtype=torch.bfloat16),
                torch.empty((128, 128), dtype=torch.float32),
            )
        )
        spec = bound.config_spec
        assert not spec.cute_chained_pointwise_residency_search_enabled
        assert BUDGET not in spec._flat_fields()
        assert KEY not in spec._flat_fields()
        assert KEY not in spec.default_config().config
