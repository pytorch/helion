"""Independent root seed-coordinate compatibility checks, CPU only."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_residency_search import _root_residency
import helion
from helion import exc
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import CUTE_CHAINED_DRAIN_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_MMA_SCHEDULE_KEY as SCHEDULE
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SEED_TILE_COLUMNS_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_VECTOR_GROUP_KEY
from helion.autotuner.config_spec import CUTE_NATIVE_MATMUL_METADATA_KEY

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.autotuner.config_spec import ConfigSpec


@pytest.fixture(scope="module")
def root_spec() -> Iterator[ConfigSpec]:
    with _cpu_codegen():
        bound = _root_residency._bind_isolated(
            (
                torch.empty((128, 16), dtype=torch.bfloat16),
                torch.empty((16, 32), dtype=torch.bfloat16),
            )
        )
        assert not bound.config_spec.cute_chained_loop_search_enabled
        yield bound.config_spec


def test_root_appends_one_coordinate_preserving_all_prior_values_and_seeds(
    root_spec: ConfigSpec,
) -> None:
    with patch.dict(
        root_spec.user_defined_tunables, {"user_choice": EnumFragment((7, 11))}
    ):
        fields = root_spec._flat_fields()
        assert tuple(fields)[-7:] == (
            KEY,
            CUTE_CHAINED_VECTOR_GROUP_KEY,
            CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY,
            CUTE_NATIVE_MATMUL_METADATA_KEY,
            CUTE_CHAINED_DRAIN_TILE_COLUMNS_KEY,
            CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY,
            CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY,
        )
        index = _flat_scalar_index(root_spec, KEY)
        field = fields[KEY]
        assert isinstance(field, EnumFragment)
        assert field.default() == 0 and field.search_values() == [0, 32, 64]
        default = root_spec.default_config()
        fragment = fields[CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY]
        assert isinstance(fragment, EnumFragment)
        assert fragment.default() is False
        assert default.config.get(CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY, False) is False
        nested = fields[CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY]
        assert isinstance(nested, EnumFragment)
        assert nested.default() is False
        assert (
            default.config.get(CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY, False) is False
        )
        seeds = tuple(root_spec.compiler_seed_configs)
        seed_values = copy.deepcopy([seed.config for seed in seeds])
        assert all(
            seed.config.get(CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY, False) is False
            for seed in seeds
        )
        assert all(
            seed.config.get(CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY, False) is False
            for seed in seeds
        )
        generation = root_spec.create_config_generation()
        pairs = generation.seed_flat_config_pairs()
        with patch.object(
            root_spec,
            "_flat_fields",
            return_value={name: field for name, field in fields.items() if name != KEY},
        ):
            old = root_spec.create_config_generation()
            old_pairs = old.seed_flat_config_pairs()
            old_roundtrip = old.unflatten(old.flatten(default))
            assert root_spec.default_config() == default
            flat = generation.flatten(default)
            assert flat[:index] + flat[index + 1 :] == old.flatten(default)
            for (flat, config), (old_flat, old_config) in zip(
                pairs, old_pairs, strict=True
            ):
                assert flat[:index] + flat[index + 1 :] == old_flat and flat[index] == 0
                assert config == old_config
                assert (
                    config.config.get(CUTE_CHAINED_FRAGMENT_EPILOGUES_KEY, False)
                    is False
                )
                assert (
                    config.config.get(CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY, False)
                    is False
                )
        assert generation.unflatten(generation.flatten(default)) == old_roundtrip
        assert all(
            before is after
            for before, after in zip(
                seeds, root_spec.compiler_seed_configs, strict=True
            )
        )
        assert [seed.config for seed in seeds] == seed_values
        assert all(KEY not in seed.config for seed in seeds)
        assert KEY not in default.config


@pytest.mark.parametrize("columns", (0, 32, 64))
def test_root_zero_and_positive_roundtrip(root_spec: ConfigSpec, columns: int) -> None:
    config = root_spec.normalized_config(
        {SCHEDULE: "tcgen05_tmem", KEY: columns, "num_warps": 4}
    )
    generation = root_spec.create_config_generation()
    assert config.config.get(KEY, 0) == columns
    assert generation.flatten(config)[_flat_scalar_index(root_spec, KEY)] == columns
    assert generation.unflatten(generation.flatten(config)) == config
    if columns == 0:
        assert KEY not in config.config
        assert config == root_spec.normalized_config(
            {SCHEDULE: "tcgen05_tmem", "num_warps": 4}
        )


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize(
    "overrides",
    (
        {SCHEDULE: None},
        {SCHEDULE: "coalesced"},
        {"cute_affine_scan_schedule": "warp"},
        {"cute_chained_direct_output": True},
        {"cute_chunk_recurrence_register_cap": 72},
        {"cute_chunk_recurrence_pipeline": "compact"},
    ),
)
def test_root_cannot_repair_incompatible_family(
    root_spec: ConfigSpec, overrides: dict[str, object], repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
        root_spec.normalize(
            {SCHEDULE: "tcgen05_tmem", KEY: 32, "num_warps": 4, **overrides},
            _fix_invalid=repair,
        )


def test_root_inherited_positive_rejects_non_tcgen_override(
    root_spec: ConfigSpec,
) -> None:
    parent = root_spec.normalized_config(
        {SCHEDULE: "tcgen05_tmem", KEY: 32, "num_warps": 4}
    )
    changed = root_spec.create_config_generation(overrides={SCHEDULE: "coalesced"})
    with pytest.raises(exc.InvalidConfig, match="requires explicit tcgen05_tmem"):
        changed.unflatten(changed.flatten(parent))
    disabled = root_spec.create_config_generation(
        overrides={SCHEDULE: "coalesced", KEY: 0}
    )
    restored = disabled.unflatten(disabled.flatten(parent))
    assert restored[SCHEDULE] == "coalesced" and KEY not in restored.config
    zero = root_spec.normalized_config(helion.Config.from_dict({KEY: 0}))
    assert zero == root_spec.normalized_config(helion.Config())
