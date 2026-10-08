from __future__ import annotations

import dataclasses
import random

import pytest
import torch

from .test_cute_flash_length_invariance import _memoized_flash_fragments
from .test_cute_flash_resident_choice import _spec as _dense_spec
from .test_cute_flash_stateful_choice import _spec as _causal_spec
from helion._compiler.cute import cute_flash as flash
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    ("causal", "key", "value"),
    (
        (False, flash.FLASH_EXP2_PACKET_KEY, "deg2_16x6"),
        (False, flash.FLASH_P_STORE_REP_KEY, 32),
        (True, flash.FLASH_P_STORE_REP_KEY, 32),
    ),
)
def test_fixed_controls_remove_only_incompatible_resident_search_choices(
    causal: bool, key: str, value: object, dtype: torch.dtype
) -> None:
    # The expanded domain requires two independent complete coverage designs.
    # Keep all 100 configs and allow both designs to finish under a bounded timeout.
    head_dim = 64 if dtype is torch.float16 else 128
    spec = (
        _causal_spec(head_dim, 32, dtype)
        if causal
        else _dense_spec(head_dim, 48, dtype)
    )
    overrides = {key: value}
    if not causal and key == flash.FLASH_P_STORE_REP_KEY:
        # Rep32 P stores require Rep32 S loads and cannot use the upstream
        # per-chunk P16 release; pin both when asking for a complete design.
        overrides[flash.FLASH_S_LOAD_REP_KEY] = 32
        overrides[flash.FLASH_P_CHUNK_ARRIVE_KEY] = False
    generation = spec.create_config_generation(overrides=overrides)
    resident = "resident_stateful" if causal else "resident_value_graph"
    saved = random.getstate()
    try:
        random.seed(20261003)
        with _memoized_flash_fragments():
            population = generation.random_population(100)
    finally:
        random.setstate(saved)
    assert len(population) == 100
    assert all(
        config.config[flash.FLASH_SOFTMAX_LOWERING_KEY] != resident
        for config in population
    )
    family = generation._flash_pipeline_family_override
    fields = (
        spec._flat_fields()
        if family is None
        else spec._flat_fields_with_flash_family(family)
    )
    fragment = fields[flash.FLASH_SOFTMAX_LOWERING_KEY]
    assert isinstance(fragment, EnumFragment)
    reference = ConfigGeneration(
        spec,
        overrides=generation._override_values,
        _flash_pipeline_family_override=generation._flash_pipeline_family_override,
        _field_view={
            **fields,
            flash.FLASH_SOFTMAX_LOWERING_KEY: dataclasses.replace(
                fragment,
                search_choices=tuple(
                    choice
                    for choice in fragment._active_choices()
                    if choice != resident
                ),
            ),
        },
    )
    if key == flash.FLASH_EXP2_PACKET_KEY:
        assert all(config.config[key] == value for config in population)
        # Packet-owned parents already leave some ordinary child axes
        # unreachable. Preserve that telemetry against the resident-free domain.
        with _memoized_flash_fragments():
            assert (
                generation.flash_structural_coverage_uncovered_values()
                == reference.flash_structural_coverage_uncovered_values()
            )
            assert (
                generation.flash_structural_coverage_uncovered_interactions()
                == reference.flash_structural_coverage_uncovered_interactions()
            )
    else:
        assert generation.flash_structural_coverage_uncovered_values() == []
        assert generation.flash_structural_coverage_uncovered_interactions() == []
        # Ordinary auto/standard paths already canonicalize some fixed controls.
        # The new mode must leave that entire pre-resident population unchanged.
        saved = random.getstate()
        try:
            random.seed(20261003)
            with _memoized_flash_fragments():
                assert population == reference.random_population(100)
        finally:
            random.setstate(saved)
    fields = generation._flat_fields()
    fragment = fields[flash.FLASH_SOFTMAX_LOWERING_KEY]
    assert isinstance(fragment, EnumFragment)
    assert fragment._active_choices() == ("auto", "standard")
    # Accepted explicit values remain available to the resolver, which rejects
    # a contradictory user request instead of silently changing the request.
    assert resident in fragment.choices
    assert (flash.FLASH_SOFTMAX_LOWERING_KEY, resident) not in (
        generation.flash_structural_coverage_active_values()
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("causal", (False, True))
def test_compatible_fixed_controls_preserve_resident_search_domain(
    causal: bool, dtype: torch.dtype
) -> None:
    head_dim = 64 if dtype is torch.float16 else 128
    spec = (
        _causal_spec(head_dim, 32, dtype)
        if causal
        else _dense_spec(head_dim, 48, dtype)
    )
    ordinary = spec.create_config_generation()
    compatible = spec.create_config_generation(
        overrides={flash.FLASH_P_STORE_REP_KEY: 16}
    )
    assert (
        compatible._flat_fields()[flash.FLASH_SOFTMAX_LOWERING_KEY]
        == ordinary._flat_fields()[flash.FLASH_SOFTMAX_LOWERING_KEY]
    )


def test_direct_generation_preserves_its_subset_view_and_explicit_errors() -> None:
    spec = _dense_spec(128, 48, torch.bfloat16)
    fields = dict(spec._flat_fields())
    fields.pop(flash.FLASH_ROWMAX_KEY)
    generation = ConfigGeneration(
        spec,
        overrides={flash.FLASH_P_STORE_REP_KEY: 32},
        _field_view=fields,
    )
    assert set(generation._flat_fields()) == set(fields)
    assert generation._flat_fields() is not fields
    key = flash.FLASH_SOFTMAX_LOWERING_KEY
    original = fields[key]
    assert isinstance(original, EnumFragment)
    assert "resident_value_graph" in original._active_choices()
    indices, is_sequence = generation._key_to_flat_indices[key]
    assert not is_sequence and len(indices) == 1
    assert generation.flat_spec[indices[0]] == generation._flat_fields()[key]
    explicit = ConfigGeneration(
        spec,
        overrides={key: "resident_value_graph", flash.FLASH_P_STORE_REP_KEY: 32},
    )
    with pytest.raises(InvalidConfig, match="requires cute_flash_p_store_rep=16"):
        explicit.unflatten(explicit.default_flat())


@pytest.mark.timeout(120)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_unpaired_store_override_retains_unrelated_coverage_gap(
    dtype: torch.dtype,
) -> None:
    head_dim = 64 if dtype is torch.float16 else 128
    spec = _dense_spec(head_dim, 48, dtype)
    generation = spec.create_config_generation(
        overrides={flash.FLASH_P_STORE_REP_KEY: 32}
    )
    expected = [(flash.FLASH_S_LOAD_REP_KEY, 16)]
    if dtype is torch.float16:
        # The FP16 dense domain advertises per-chunk P release, whose P16
        # layout cannot survive this P32 override. Keep that gap visible too.
        expected.append((flash.FLASH_P_CHUNK_ARRIVE_KEY, True))
    with _memoized_flash_fragments():
        assert generation.flash_structural_coverage_uncovered_values() == expected
