from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from .test_cute_chained_loop import _affine_args
from .test_cute_chained_loop import _affine_chain
from .test_cute_chained_tcgen05 import _tcgen_config_bound
from .test_cute_chunk_recurrence import _bt16_fp32_chain
from .test_cute_chunk_recurrence import _fake_inputs
from .test_cute_chunk_recurrence import _real_dispatch_cpu
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChunkRecurrenceHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import CUTE_CHAINED_ASYNC_VECTOR_STORE_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_MMA_SCHEDULE_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_ENTRIES_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_LAYOUT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SEED_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_VECTOR_GROUP_KEY
from helion.autotuner.config_spec import CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY
from helion.autotuner.config_spec import CUTE_CHUNK_RECURRENCE_PIPELINE_KEY
from helion.autotuner.config_spec import CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY
from helion.autotuner.config_spec import CUTE_NATIVE_MATMUL_METADATA_KEY

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])

_LEGACY_KEYS = (
    CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY,
    CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY,
    CUTE_CHUNK_RECURRENCE_PIPELINE_KEY,
)


@pytest.fixture(params=(1, 2), ids=("short-dv4", "packed-dv2"))
def overlap_bound(request: pytest.FixtureRequest) -> Iterator[BoundKernel]:
    with (
        _real_dispatch_cpu(),
        # Exercise SM103's existing promotion policy without initializing CUDA.
        patch.object(CuteChunkRecurrenceHeuristic, "should_promote", return_value=True),
    ):
        yield _bt16_fp32_chain._bind_isolated(
            (
                *_fake_inputs(fp32_state=True, sequences=request.param),
                128**-0.5,
            )
        )


def test_promoted_legacy_default_retains_family_and_register_cap(
    overlap_bound: BoundKernel,
) -> None:
    spec = overlap_bound.config_spec
    promoted = spec.compiler_default_config
    assert promoted is not None
    default = spec.default_config()
    assert CUTE_CHAINED_MMA_SCHEDULE_KEY not in default.config
    for key in _LEGACY_KEYS:
        assert default[key] == promoted[key]
    generation = ConfigGeneration(spec)
    assert generation.unflatten(generation.flatten(default)) == default
    assert spec.normalized_config(default) == default
    source = overlap_bound.to_code(default)
    assert "'state_dtype': 'float32'" in source
    assert "chain_loop_index" not in source
    if promoted[CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY] == 4:
        assert default[CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY] == 72
        assert "--maxrregcount=72" in source
        assert "'kind': 'chunk_recurrence_warp_dv4'" in source
    else:
        assert default[CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY] is None
        assert "'kind': 'chunk_recurrence_sm100'" in source


def test_all_legacy_seeds_survive_flat_transfer_and_generate_legacy_source(
    overlap_bound: BoundKernel,
) -> None:
    spec = overlap_bound.config_spec
    generation = ConfigGeneration(spec)
    legacy_seeds = [
        seed for seed in spec.compiler_seed_configs if _LEGACY_KEYS[0] in seed.config
    ]
    assert len(legacy_seeds) == 6
    pairs = generation.seed_flat_config_pairs()
    transferred = {
        tuple(config[key] for key in _LEGACY_KEYS): (flat, config)
        for flat, config in pairs
        if CUTE_CHAINED_MMA_SCHEDULE_KEY not in config.config
    }
    assert len(transferred) == 6
    for seed in legacy_seeds:
        flat, config = transferred[tuple(seed[key] for key in _LEGACY_KEYS)]
        assert generation.flatten(seed) == flat
        assert generation.unflatten(flat) == config
        assert generation.flatten(config) == flat
        assert spec.normalized_config(config) == config
        assert "'state_dtype': 'float32'" in overlap_bound.to_code(config)


def test_common_seed_prefix_reference_and_search_choices_are_unchanged(
    overlap_bound: BoundKernel,
) -> None:
    spec = overlap_bound.config_spec
    common_seeds = [
        seed
        for seed in spec.compiler_seed_configs
        if _LEGACY_KEYS[0] not in seed.config
    ]
    # Disable only the added coexistence surface to reconstruct the preceding
    # common schema. No graph, schedule, seed-order or emitter is mocked.
    with patch.object(spec, "_cute_chained_legacy_loop_fragments", return_value={}):
        old_fields = spec._flat_fields()
        old_reference = spec.autotune_reference_config()
        with patch.object(spec, "compiler_seed_configs", common_seeds):
            old_pairs = ConfigGeneration(spec).seed_flat_config_pairs()
    generation = ConfigGeneration(spec)
    fields = spec._flat_fields()
    # These opt-in coordinates follow the complete legacy coexistence suffix.
    # Preserve every earlier field in order and name the exact final suffix.
    suffix = (
        CUTE_CHAINED_POINTWISE_CACHE_ENTRIES_KEY,
        CUTE_CHAINED_SEED_TILE_COLUMNS_KEY,
        CUTE_CHAINED_POINTWISE_CACHE_LAYOUT_KEY,
        CUTE_CHAINED_VECTOR_GROUP_KEY,
        CUTE_NATIVE_MATMUL_METADATA_KEY,
        CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY,
    )
    if "cute_grid_work_order" in old_fields:
        suffix += ("cute_grid_work_order",)
    suffix += ("cute_chained_fragment_epilogues",)
    # These promoted legacy graphs do not discover common preparation, so the
    # asynchronous shared sink is not an effective search coordinate here.
    assert not spec.cute_chained_preparation_pipeline_search_enabled
    assert CUTE_CHAINED_ASYNC_VECTOR_STORE_KEY not in old_fields
    assert CUTE_CHAINED_ASYNC_VECTOR_STORE_KEY not in fields
    assert tuple(old_fields)[-len(suffix) :] == suffix
    assert tuple(fields) == (
        *tuple(old_fields)[: -len(suffix)],
        *_LEGACY_KEYS,
        *suffix,
    )
    assert spec.autotune_reference_config() == old_reference
    for key in (
        CUTE_NATIVE_MATMUL_METADATA_KEY,
        CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY,
    ):
        field = fields[key]
        assert isinstance(field, EnumFragment)
        assert field.default() is False
        assert field.search_values() == [False, True]
    for key, old_field in old_fields.items():
        if isinstance(old_field, EnumFragment):
            new_field = fields[key]
            assert isinstance(new_field, EnumFragment)
            assert new_field.default() == old_field.default()
            assert new_field.search_values() == old_field.search_values()
    for key in _LEGACY_KEYS:
        field = fields[key]
        assert isinstance(field, EnumFragment)
        assert field.search_values() == [field.default()]
    schedule = fields[CUTE_CHAINED_MMA_SCHEDULE_KEY]
    assert isinstance(schedule, EnumFragment)
    assert schedule.choices == (*spec._cute_chained_mma_schedules(), None)
    assert schedule.search_values() == list(spec._cute_chained_mma_schedules())
    pairs = generation.seed_flat_config_pairs()
    expected_prefix = [config for _, config in old_pairs]
    assert [config for _, config in pairs[: len(old_pairs)]] == expected_prefix
    assert len(pairs) == len(old_pairs) + 6
    population = generation.random_population_flat(len(pairs))
    assert [generation.unflatten(flat) for flat in population] == [
        config for _, config in pairs
    ]


def test_common_family_canonicalizes_inactive_legacy_partition(
    overlap_bound: BoundKernel,
) -> None:
    spec = overlap_bound.config_spec
    common = spec.autotune_reference_config()
    generation = ConfigGeneration(spec)
    for partitions in (2, 4):
        variant = helion.Config.from_dict(
            common.config | {CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY: partitions}
        )
        assert spec.normalized_config(variant) == common
        assert generation.unflatten(generation.flatten(variant)) == common


def _resident_parent(bound: BoundKernel) -> helion.Config:
    return bound.config_spec.normalized_config(
        helion.Config(
            block_sizes=[64],
            num_warps=4,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_group_contractions=True,
            cute_chained_pointwise_cache_bytes=4096,
            cute_chained_pointwise_unroll=4,
            cute_chained_warp_mma_rows=16,
        )
    )


@pytest.mark.parametrize("resident", (False, True), ids=("reference", "resident"))
@pytest.mark.parametrize(
    "overrides",
    (
        {_LEGACY_KEYS[0]: 4},
        {_LEGACY_KEYS[0]: 4, _LEGACY_KEYS[1]: 72},
        {_LEGACY_KEYS[0]: 2, _LEGACY_KEYS[2]: "compact"},
    ),
)
def test_legacy_only_overrides_replace_inherited_common_family(
    overlap_bound: BoundKernel, overrides: dict[str, object], resident: bool
) -> None:
    spec = overlap_bound.config_spec
    parent = (
        _resident_parent(overlap_bound)
        if resident
        else spec.autotune_reference_config()
    )
    generation = spec.create_config_generation(overrides=overrides)
    result = generation.unflatten(generation.flatten(parent))
    assert CUTE_CHAINED_MMA_SCHEDULE_KEY not in result.config
    assert result.block_sizes == parent.block_sizes
    assert result.num_warps == parent.num_warps
    for key, value in overrides.items():
        assert result[key] == value
    assert not result.config.get("cute_chained_group_contractions")
    assert not result.config.get("cute_chained_pointwise_cache_bytes")
    assert not result.config.get("cute_chained_warp_mma_rows")
    assert result.config.get("cute_chained_pointwise_unroll", 1) == 1
    assert generation.unflatten(generation.flatten(result)) == result
    assert "'state_dtype': 'float32'" in overlap_bound.to_code(result)


def test_override_hook_discards_only_inherited_common_keys(
    overlap_bound: BoundKernel,
) -> None:
    # Test the merge boundary independently of graph admission: an inherited
    # scan child must be removed too, even though this recurrence has no scan.
    parent = _resident_parent(overlap_bound).config | {
        "cute_chained_scan_schedule": "warp",
        "cute_chained_user_value": 7,
        "num_warps": 8,
    }
    overrides = {_LEGACY_KEYS[0]: 4, "cute_chained_pointwise_cache_bytes": 0}
    merged = parent | overrides
    with patch.dict(
        overlap_bound.config_spec.user_defined_tunables,
        {"cute_chained_user_value": EnumFragment((7,))},
    ):
        overlap_bound.config_spec.prepare_override_normalization(merged, overrides)
    assert {
        key: value for key, value in merged.items() if key.startswith("cute_chained_")
    } == {"cute_chained_pointwise_cache_bytes": 0, "cute_chained_user_value": 7}
    assert merged["block_sizes"] == parent["block_sizes"]
    assert merged["num_warps"] == 8
    assert merged[_LEGACY_KEYS[0]] == 4


def test_explicit_common_override_keeps_precedence(
    overlap_bound: BoundKernel,
) -> None:
    spec = overlap_bound.config_spec
    parent = _resident_parent(overlap_bound)
    overrides = {_LEGACY_KEYS[0]: 4, "cute_chained_pointwise_unroll": 8}
    generation = spec.create_config_generation(overrides=overrides)
    result = generation.unflatten(generation.flatten(parent))
    assert result[CUTE_CHAINED_MMA_SCHEDULE_KEY] == "tcgen05_tmem"
    assert result["cute_chained_pointwise_unroll"] == 8
    assert result["cute_chained_group_contractions"] is True
    assert result["cute_chained_warp_mma_rows"] == 16
    assert generation.unflatten(generation.flatten(result)) == result


@pytest.mark.parametrize(
    "overrides",
    (
        {_LEGACY_KEYS[0]: 4, "cute_chained_auxiliary_cache": True},
        {_LEGACY_KEYS[0]: 4, "num_warps": 3},
        {
            _LEGACY_KEYS[0]: 4,
            _LEGACY_KEYS[1]: 72,
            CUTE_CHAINED_MMA_SCHEDULE_KEY: "coalesced",
        },
    ),
)
def test_override_cannot_erase_explicit_conflicts_or_repair_generic_geometry(
    overlap_bound: BoundKernel, overrides: dict[str, object]
) -> None:
    spec = overlap_bound.config_spec
    generation = spec.create_config_generation(overrides=overrides)
    with pytest.raises(exc.InvalidConfig):
        generation.unflatten(generation.flatten(spec.autotune_reference_config()))


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "overrides",
    (
        {CUTE_CHAINED_MMA_SCHEDULE_KEY: "coalesced", _LEGACY_KEYS[1]: 72},
        {CUTE_CHAINED_MMA_SCHEDULE_KEY: "coalesced", _LEGACY_KEYS[2]: "compact"},
        {
            CUTE_CHAINED_MMA_SCHEDULE_KEY: None,
            "cute_chained_pointwise_cache_bytes": 4096,
        },
        {
            CUTE_CHAINED_MMA_SCHEDULE_KEY: None,
            "cute_chained_group_contractions": True,
        },
        {
            CUTE_CHAINED_MMA_SCHEDULE_KEY: None,
            "cute_chained_warp_mma_rows": 16,
        },
    ),
)
def test_conflicting_explicit_family_controls_reject_before_repair(
    overlap_bound: BoundKernel, overrides: dict[str, object], repair: bool
) -> None:
    config = helion.Config.from_dict(
        {"block_sizes": [64], _LEGACY_KEYS[0]: 4, **overrides}
    )
    with pytest.raises(exc.InvalidConfig, match="conflict|requires"):
        overlap_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize("partitions,cap", ((4, 128), (2, 72)))
def test_legacy_sentinel_preserves_register_cap_safety(
    overlap_bound: BoundKernel, partitions: int, cap: int
) -> None:
    config = helion.Config.from_dict(
        {
            "block_sizes": [64],
            CUTE_CHAINED_MMA_SCHEDULE_KEY: None,
            _LEGACY_KEYS[0]: partitions,
            _LEGACY_KEYS[1]: cap,
        }
    )
    with pytest.raises(exc.InvalidConfig, match="must be one of|must be None"):
        overlap_bound.config_spec.normalize(config)


@pytest.mark.parametrize("loop", (False, True), ids=("root", "loop"))
def test_unrelated_common_graph_has_no_legacy_surface_or_sentinel(loop: bool) -> None:
    with _real_dispatch_cpu():
        bound = (
            _affine_chain._bind_isolated(_affine_args())
            if loop
            else _tcgen_config_bound(n=64)
        )
        spec = bound.config_spec
        fields = spec._flat_fields()
        assert not set(fields) & set(_LEGACY_KEYS)
        schedule = fields[CUTE_CHAINED_MMA_SCHEDULE_KEY]
        assert isinstance(schedule, EnumFragment)
        assert schedule.choices == spec._cute_chained_mma_schedules()
        assert schedule.search_choices is None
        with pytest.raises(exc.InvalidConfig, match="overlapping loop families"):
            spec.normalized_config(helion.Config(cute_chained_mma_schedule=None))
