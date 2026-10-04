from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _inputs
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_residency_search import _root_residency
import helion
from helion import exc
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import CUTE_CHAINED_COLLECTIVE_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_STMATRIX_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_LEAF_COUNT_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_NATIVE_VECTOR_READS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OPERAND_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_REGISTER_ISLANDS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import ConfigSpec


def _config(count=1, **overrides):
    return helion.Config.from_dict(
        {
            "num_warps": 8,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_warp_mma_rows": 32,
            "cute_chained_preparation_pipeline": True,
            "cute_chained_leaf_pipeline": "rectangular_tma",
            KEY: count,
            **overrides,
        }
    )


@pytest.fixture(scope="module")
def leaf_set_bound():
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)):
            yield bound


def test_public_constructor_default_and_family_request(leaf_set_bound):
    spec = leaf_set_bound.config_spec
    assert spec.supports_config_key(KEY)
    assert spec.flatten_missing_field_default(KEY, {}) == (True, 1)
    explicit = helion.Config(cute_chained_leaf_count=1)
    assert explicit[KEY] == 1
    assert spec.normalized_config(explicit) == spec.normalized_config(helion.Config())
    assert KEY not in spec.normalized_config(explicit).config
    assert not ConfigSpec._requests_cute_chained_loop({KEY: 1})
    assert ConfigSpec._requests_cute_chained_loop({KEY: 2})
    assert ConfigSpec._requests_cute_chained_loop({KEY: 4})


@pytest.mark.parametrize("count", [1, 2, 4])
def test_exact_choices_roundtrip_and_explicit_override(leaf_set_bound, count):
    spec = leaf_set_bound.config_spec
    field = spec._flat_fields()[KEY]
    assert isinstance(field, EnumFragment)
    assert field.default() == 1 and field.search_values() == [1, 2, 4]
    config = spec.normalized_config(_config(count))
    assert config.config.get(KEY, 1) == count
    assert (KEY in config.config) == (count != 1)
    generation = spec.create_config_generation()
    assert generation.unflatten(generation.flatten(config)) == config
    layout = spec.flat_key_layout()
    field_index = next(i for i, item in enumerate(layout) if item[0] == KEY)
    index = sum(size for _, size, _ in layout[:field_index])
    assert generation.flatten(config)[index] == count
    override = spec.create_config_generation(overrides={KEY: count})
    assert (
        override.unflatten(override.flatten(spec.normalized_config(_config())))
        == config
    )


def test_appended_coordinate_keeps_all_previous_fields_seeds_and_defaults(
    leaf_set_bound,
):
    spec = leaf_set_bound.config_spec
    seeds = tuple(spec.compiler_seed_configs)
    snapshots = [copy.deepcopy(seed.config) for seed in seeds]
    with patch.dict(
        spec.user_defined_tunables, {"leaf_test_choice": EnumFragment((3, 5))}
    ):
        fields = spec._flat_fields()
        assert tuple(fields)[-20:] == (
            CUTE_CHAINED_REGISTER_ISLANDS_KEY,
            KEY,
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
            "cute_native_matmul_metadata",
            "cute_chained_drain_tile_columns",
            "cute_chained_island_consumers",
            "cute_chained_async_vector_store",
            "cute_chained_fragment_epilogues",
        )
        layout = spec.flat_key_layout()
        field_index = next(i for i, item in enumerate(layout) if item[0] == KEY)
        index = sum(size for _, size, _ in layout[:field_index])
        assert layout[field_index] == (KEY, 1, False)
        default = spec.default_config()
        generation = spec.create_config_generation()
        pairs = generation.seed_flat_config_pairs()
        with patch.object(
            spec,
            "_flat_fields",
            return_value={key: value for key, value in fields.items() if key != KEY},
        ):
            old = spec.create_config_generation()
            flat_default = generation.flatten(default)
            assert flat_default[:index] + flat_default[index + 1 :] == old.flatten(
                default
            )
            assert flat_default[index] == 1
            assert spec.default_config() == default
            for (flat, config), (old_flat, old_config) in zip(
                pairs, old.seed_flat_config_pairs(), strict=True
            ):
                assert flat[:index] + flat[index + 1 :] == old_flat and flat[index] == 1
                assert config == old_config
        assert KEY not in default.config
    assert all(KEY not in seed.config for seed in seeds)
    assert all(
        left is right
        for left, right in zip(seeds, spec.compiler_seed_configs, strict=True)
    )
    assert [seed.config for seed in seeds] == snapshots


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("count", [True, False, None, 0, -1, 3, 8, 1.0, "2", [], {}])
def test_strict_count_before_autotune_repair(leaf_set_bound, count, repair):
    with pytest.raises(exc.InvalidConfig, match=KEY):
        leaf_set_bound.config_spec.normalize(_config(count), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "prerequisite",
    [
        "cute_chained_mma_schedule",
        "cute_chained_preparation_pipeline",
        "cute_chained_leaf_pipeline",
    ],
)
def test_missing_prerequisites_are_never_injected(leaf_set_bound, prerequisite, repair):
    config = _config(2)
    config.config.pop(prerequisite)
    with pytest.raises(exc.InvalidConfig):
        leaf_set_bound.config_spec.normalize(config, _fix_invalid=repair)
    assert prerequisite not in config.config


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"cute_chained_mma_schedule": "coalesced"},
        {"cute_chained_preparation_pipeline": False},
        {"cute_chained_leaf_pipeline": "legacy"},
        {"cute_chained_leaf_pipeline": "paired_tma"},
    ],
)
def test_inactive_or_wrong_lowering_rejects_nondefault(
    leaf_set_bound, overrides, repair
):
    with pytest.raises(exc.InvalidConfig):
        leaf_set_bound.config_spec.normalize(
            _config(2, **overrides), _fix_invalid=repair
        )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "flag",
    [
        "cute_chained_loop_search_enabled",
        "cute_chained_preparation_pipeline_search_enabled",
    ],
)
def test_discovery_is_required_and_unavailable_coordinate_is_absent(
    leaf_set_bound, flag, repair
):
    spec = leaf_set_bound.config_spec
    with patch.object(spec, flag, False):
        assert KEY not in spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(2), _fix_invalid=repair)


@pytest.mark.parametrize("backend", ["triton", "pallas"])
@pytest.mark.parametrize("repair", [False, True])
def test_other_backends_allow_only_canonical_default(leaf_set_bound, backend, repair):
    spec = leaf_set_bound.config_spec
    with patch.object(spec, "backend_name", backend):
        default = {KEY: 1}
        spec.normalize(default, _fix_invalid=repair)
        assert KEY not in default
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(2), _fix_invalid=repair)


def test_root_has_no_leaf_count_coordinate_and_nondefault_cannot_select_loop():
    with _cpu_codegen():
        bound = _root_residency._bind_isolated(
            (
                torch.empty((128, 16), dtype=torch.bfloat16),
                torch.empty((16, 32), dtype=torch.bfloat16),
            )
        )
        spec = bound.config_spec
        assert not spec.cute_chained_loop_search_enabled
        assert KEY not in spec._flat_fields()
        assert spec.normalized_config(
            helion.Config(cute_chained_leaf_count=1)
        ) == spec.normalized_config(helion.Config())
        for repair in (False, True):
            with pytest.raises(exc.InvalidConfig):
                spec.normalize(_config(2), _fix_invalid=repair)
