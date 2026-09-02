from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
import functools
import hashlib
import itertools
import json
import os
import random
from typing import TYPE_CHECKING
from typing import Any
from typing import TypedDict
from typing import TypeVar
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import SOFTCAP_KIND
from helion._compiler.cute.attention_plan import AttentionScoreModifier
from helion._compiler.cute.attention_plan import dense_score_plan
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_generation import _flash_log_maximin_refinements
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.device_function import DeviceFunction

# Length-invariance is an equality chain, so comparing the two extreme lengths
# of a legality class gives the same guarantee as comparing every member;
# test_flash_length_classes_preserve_only_structural_legality_boundaries pins
# the interior lengths to their class.
_ALIGNED_LENGTHS = (4, 4096)
_LENGTH_CLASSES = (
    pytest.param((3, 97), id="odd"),
    pytest.param((2, 386), id="paired"),
    pytest.param(_ALIGNED_LENGTHS, id="div4"),
)
_DENSE_DEG1_PACKETS = frozenset(("deg1_16x8", "deg1_8x2_corr10"))
_SEMANTIC_CASES = (
    pytest.param(torch.float16, 64, False, id="fp16-d64-dense"),
    pytest.param(torch.float16, 64, True, id="fp16-d64-causal"),
    pytest.param(torch.float16, 128, False, id="fp16-d128-dense"),
    pytest.param(torch.float16, 128, True, id="fp16-d128-causal"),
    pytest.param(torch.bfloat16, 64, False, id="bf16-d64-dense"),
    pytest.param(torch.bfloat16, 64, True, id="bf16-d64-causal"),
    pytest.param(torch.bfloat16, 128, False, id="bf16-d128-dense"),
    pytest.param(torch.bfloat16, 128, True, id="bf16-d128-causal"),
)
# Spans both dtypes, both head dims, and dense+causal with the fewest cases.
_POPULATION_CASES = (
    pytest.param(torch.float16, 64, True, id="fp16-d64-causal"),
    pytest.param(torch.bfloat16, 128, False, id="bf16-d128-dense"),
)

_T = TypeVar("_T")


class _ShapeOptions(TypedDict):
    num_bh: int
    dtype: torch.dtype
    is_causal: bool
    standard_dense_output: bool
    standard_causal_output: bool


def _shape_options(dtype: torch.dtype, is_causal: bool) -> _ShapeOptions:
    return {
        "num_bh": 64,
        "dtype": dtype,
        "is_causal": is_causal,
        "standard_dense_output": not is_causal,
        "standard_causal_output": is_causal,
    }


def _active_choices(fragment: EnumFragment) -> frozenset[object]:
    choices = (
        fragment.choices if fragment.search_choices is None else fragment.search_choices
    )
    return frozenset(choices)


def _active_choice_sets(
    head_dim: int,
    num_kv: int,
    *,
    dtype: torch.dtype,
    is_causal: bool,
) -> dict[str, frozenset[object]]:
    fragments = cute_flash.flash_autotune_fragments(
        head_dim,
        num_kv,
        **_shape_options(dtype, is_causal),
    )
    assert all(isinstance(fragment, EnumFragment) for fragment in fragments.values())
    return {
        key: _active_choices(fragment)
        for key, fragment in fragments.items()
        if isinstance(fragment, EnumFragment)
    }


def _ordered_surface(
    head_dim: int,
    num_kv: int,
    *,
    dtype: torch.dtype,
    is_causal: bool,
) -> dict[str, tuple[object, tuple[object, ...], tuple[object, ...]]]:
    fragments = cute_flash.flash_autotune_fragments(
        head_dim,
        num_kv,
        **_shape_options(dtype, is_causal),
    )
    return {
        key: (
            fragment.default(),
            tuple(fragment.choices),
            tuple(
                fragment.choices
                if fragment.search_choices is None
                else fragment.search_choices
            ),
        )
        for key, fragment in fragments.items()
        if isinstance(fragment, EnumFragment)
    }


def _config_fingerprints(configs: list[helion.Config]) -> frozenset[str]:
    return frozenset(
        json.dumps(config.config, sort_keys=True, separators=(",", ":"))
        for config in configs
    )


def _seed_config_fingerprints(
    head_dim: int,
    num_kv: int,
    *,
    dtype: torch.dtype,
    is_causal: bool,
) -> frozenset[str]:
    spec = _flash_config_spec(
        head_dim=head_dim,
        num_kv=num_kv,
        dtype=dtype,
        is_causal=is_causal,
    )
    generation = spec.create_config_generation()
    seeds = cute_flash.flash_attention_seed_configs(
        head_dim,
        num_kv,
        **_shape_options(dtype, is_causal),
    )
    normalized = [
        generation.canonicalize_flat(generation.flatten(seed))[1] for seed in seeds
    ]
    return _config_fingerprints(normalized)


@contextlib.contextmanager
def _memoized_flash_fragments() -> Iterator[None]:
    """Memoize flash_autotune_fragments while a coverage design is built.

    The structural-coverage design calls it tens of thousands of times with
    only a handful of distinct argument tuples, dominating its cost. The first
    repeat of each key is re-verified against the real function so
    nondeterminism would still fail, and every hit returns a fresh dict.
    """
    real = cute_flash.flash_autotune_fragments
    cache: dict[object, dict[str, Any]] = {}
    verified: set[object] = set()

    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        key = (args, tuple(sorted(kwargs.items())))
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = real(*args, **kwargs)
        elif key not in verified:
            verified.add(key)
            assert real(*args, **kwargs) == hit
        return dict(hit)

    with patch.object(cute_flash, "flash_autotune_fragments", wrapper):
        yield


@functools.cache
def _shared_flash_generation(
    head_dim: int,
    num_kv: int,
    dtype: torch.dtype,
    is_causal: bool,
) -> ConfigGeneration:
    """Warmed ConfigGeneration shared across read-only tests.

    The flash structural-coverage validation costs seconds per instance and
    depends only on these four parameters, so tests that merely query the
    surface share one instance. Tests that mutate or patch the spec or the
    generation must build their own.
    """
    generation = _flash_config_spec(
        head_dim=head_dim,
        num_kv=num_kv,
        dtype=dtype,
        is_causal=is_causal,
    ).create_config_generation()
    with _memoized_flash_fragments():
        generation.flash_deterministic_population_configs()
    return generation


def _structural_coverage_configs(
    head_dim: int,
    num_kv: int,
    *,
    dtype: torch.dtype,
    is_causal: bool,
) -> tuple[ConfigGeneration, list[helion.Config]]:
    generation = _shared_flash_generation(head_dim, num_kv, dtype, is_causal)
    return generation, generation.flash_deterministic_population_configs()


def _effective_pipeline_families(
    head_dim: int,
    num_kv: int,
    *,
    dtype: torch.dtype,
    is_causal: bool,
    requires_ws_overlap: bool = False,
) -> frozenset[str]:
    candidates = (
        ("ws_overlap",)
        if requires_ws_overlap
        else cute_flash.FLASH_AUTOTUNE_PIPELINE_FAMILIES
    )
    return frozenset(
        family
        for family in candidates
        if cute_flash.resolve_flash_config(
            head_dim,
            num_kv,
            {cute_flash.FLASH_PIPELINE_FAMILY_KEY: family},
            **_shape_options(dtype, is_causal),
            requires_ws_overlap=requires_ws_overlap,
        ).pipeline_family
        == family
    )


def _assert_structural_coverage(
    head_dim: int,
    num_kv: int,
    *,
    dtype: torch.dtype,
    is_causal: bool,
) -> None:
    generation, configs = _structural_coverage_configs(
        head_dim,
        num_kv,
        dtype=dtype,
        is_causal=is_causal,
    )
    fields = generation.config_spec._flat_fields()
    axes = {
        key: fragment
        for key, fragment in fields.items()
        if key in cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
        and isinstance(fragment, EnumFragment)
    }
    goals = {
        (key, value)
        for key, fragment in axes.items()
        for value in (
            fragment.choices
            if fragment.search_choices is None
            else fragment.search_choices
        )
    }
    covered = {(key, config.config.get(key)) for config in configs for key in axes}

    assert configs
    assert goals <= covered, goals - covered
    assert generation.flash_structural_coverage_uncovered_values() == []
    assert generation.flash_structural_coverage_uncovered_interactions() == []
    assert all(
        generation.canonicalize_flat(generation.flatten(config))[1] == config
        for config in configs
    )
    expected_families = _effective_pipeline_families(
        head_dim,
        num_kv,
        dtype=dtype,
        is_causal=is_causal,
    ) - ({"fa4_deep_1cta"} if is_causal else set())
    assert (
        _active_choices(
            cast("EnumFragment", fields[cute_flash.FLASH_PIPELINE_FAMILY_KEY])
        )
        == expected_families
    )


def _assert_all_equal(values: dict[int, _T]) -> None:
    reference_length = next(iter(values))
    reference = values[reference_length]
    assert all(value == reference for value in values.values()), values


def _flash_config_spec(
    *,
    head_dim: int,
    num_kv: int,
    dtype: torch.dtype,
    is_causal: bool,
    num_bh: int = 64,
    tensor_4d_heads: int | None = None,
    standard_dense_output: bool | None = None,
    standard_causal_output: bool | None = None,
    requires_ws_overlap: bool = False,
    supports_tensor_4d_tma: bool = True,
    output_requires_tma: bool = False,
) -> ConfigSpec:
    spec = ConfigSpec(backend=CuteBackend())
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=head_dim,
        num_kv=num_kv,
        num_bh=num_bh,
        tensor_4d_heads=tensor_4d_heads,
        dtype=dtype,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=is_causal,
        requires_ws_overlap=requires_ws_overlap,
        standard_dense_output=(
            not is_causal if standard_dense_output is None else standard_dense_output
        ),
        standard_causal_output=(
            is_causal if standard_causal_output is None else standard_causal_output
        ),
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        output_requires_tma=output_requires_tma,
    )
    return spec


def test_tensor_4d_tma_capability_requires_bhsd_inputs_and_dense_scores() -> None:
    batch, heads, seq, head_dim = 2, 4, 128, 64
    bases = [
        torch.empty(batch, heads, seq, head_dim, dtype=torch.float16) for _ in range(3)
    ]
    q_view, k_view, v_view = (
        value.reshape(batch * heads, seq, head_dim) for value in bases
    )
    common = {
        "batch": batch * heads,
        "seq": seq,
        "head_dim": head_dim,
        "dtype": torch.float16,
    }

    assert cute_flash._flash_values_support_tensor_4d_tma(
        q_view,
        k_view,
        v_view,
        score_plan=dense_score_plan(head_dim),
        **common,
    )
    assert not cute_flash._flash_values_support_tensor_4d_tma(
        torch.empty_like(q_view),
        k_view,
        v_view,
        score_plan=dense_score_plan(head_dim),
        **common,
    )
    assert not cute_flash._flash_values_support_tensor_4d_tma(
        q_view,
        k_view,
        v_view,
        score_plan=dataclasses.replace(
            dense_score_plan(head_dim),
            modifiers=(AttentionScoreModifier(SOFTCAP_KIND, value_log2=1.0),),
        ),
        **common,
    )


def test_tensor_4d_tma_capability_filters_and_canonicalizes_search_families() -> None:
    capable = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
        supports_tensor_4d_tma=True,
    )
    incapable = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
        supports_tensor_4d_tma=False,
    )

    capable_families = _active_choices(
        cast(
            "EnumFragment",
            capable._flat_fields()[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
        )
    )
    incapable_families = _active_choices(
        cast(
            "EnumFragment",
            incapable._flat_fields()[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
        )
    )
    assert any(
        cute_flash.FLASH_PIPELINE_FAMILY_FLAGS[family].tensor_4d_tma
        for family in capable_families
    )
    assert all(
        not cute_flash.FLASH_PIPELINE_FAMILY_FLAGS[family].tensor_4d_tma
        for family in incapable_families
    )
    assert all(
        not cute_flash.FLASH_PIPELINE_FAMILY_FLAGS[
            seed.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
        ].tensor_4d_tma
        for seed in incapable.autotune_seed_configs()
    )

    stale = helion.Config(
        block_sizes=[1, 128, 128],
        cute_flash_pipeline_family="fa4_tma_4d",
    )
    incapable.normalize(stale)
    assert stale.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4"

    with pytest.raises(InvalidConfig, match=r"is not (?:legal|effective)"):
        generation = incapable.create_config_generation(
            overrides={
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_tma_4d",
            }
        )
        generation.unflatten(generation.default_flat())


@pytest.mark.parametrize(("dtype", "head_dim", "is_causal"), _SEMANTIC_CASES)
@pytest.mark.parametrize("num_kv_values", _LENGTH_CLASSES)
def test_flash_search_surfaces_are_length_invariant(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    num_kv_values: tuple[int, ...],
) -> None:
    surfaces: dict[int, object] = {}
    seed_fingerprints: dict[int, frozenset[str]] = {}
    resolved_defaults: dict[int, object] = {}
    for num_kv in num_kv_values:
        # _ordered_surface records defaults, legal choices, and active search
        # choices per fragment, so it subsumes the active choice sets.
        surfaces[num_kv] = _ordered_surface(
            head_dim,
            num_kv,
            dtype=dtype,
            is_causal=is_causal,
        )
        seed_fingerprints[num_kv] = _seed_config_fingerprints(
            head_dim,
            num_kv,
            dtype=dtype,
            is_causal=is_causal,
        )
        with patch.dict(os.environ, {}, clear=True):
            resolved_defaults[num_kv] = cute_flash.flash_effective_config_values(
                cute_flash.resolve_flash_config(
                    head_dim,
                    num_kv,
                    dtype=dtype,
                    num_bh=64,
                    is_causal=is_causal,
                    standard_dense_output=not is_causal,
                    standard_causal_output=is_causal,
                )
            )
    _assert_all_equal(surfaces)
    _assert_all_equal(seed_fingerprints)
    _assert_all_equal(resolved_defaults)


# Merges the per-length invariance checks that each need the same expensive
# warmed generations: deterministic-coverage fingerprints and leaf catalogs,
# low-confound schedule anchors, starting-path and family-probe limits, the
# terminal coordinate-surface catalog, and coordinate-neighbor projections.
@pytest.mark.parametrize("is_causal", (False, True), ids=("dense", "causal"))
def test_flash_structural_design_is_length_invariant(is_causal: bool) -> None:
    random_state = random.getstate()
    try:
        random.seed(0)
        before = random.getstate()
        fingerprints: dict[int, frozenset[str]] = {}
        leaf_catalogs: dict[int, frozenset[object]] = {}
        anchor_fingerprints: dict[int, tuple[str, ...]] = {}
        starting_path_limits: dict[int, tuple[int, ...]] = {}
        family_probe_limits: dict[int, int] = {}
        terminal_catalogs: dict[int, str] = {}
        neighbor_projections: dict[int, tuple[object, ...]] = {}
        for num_kv in _ALIGNED_LENGTHS:
            generation, configs = _structural_coverage_configs(
                64,
                num_kv,
                dtype=torch.float16,
                is_causal=is_causal,
            )
            fingerprints[num_kv] = _config_fingerprints(configs)
            leaf_catalogs[num_kv] = frozenset(
                cute_flash.flash_structural_leaf_from_config(config.config)
                for config in configs
            )
            anchors = generation.flash_low_confound_schedule_anchor_configs()
            assert anchors
            anchor_fingerprints[num_kv] = tuple(
                json.dumps(config.config, sort_keys=True, separators=(",", ":"))
                for config in anchors
            )
            starting_path_limits[num_kv] = tuple(
                generation.flash_structural_starting_path_limit(
                    minimum=14,
                    retained_families=None,
                    retained_candidates_per_leaf=retained,
                )
                for retained in (1, 2)
            )
            family_probe_limits[num_kv] = (
                generation.flash_structural_family_probe_path_limit(4, 1)
            )
            catalog = generation.flash_terminal_coordinate_surface_catalog(radius=2)
            assert catalog["schema_version"] == 1
            assert catalog["radius"] == 2
            assert catalog["leaves"]
            payload = json.dumps(
                catalog,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            terminal_catalogs[num_kv] = hashlib.sha256(payload.encode()).hexdigest()

            base = generation.unflatten(generation.default_flat())
            leaf = cute_flash.flash_structural_leaf_from_config(base.config)
            assert leaf is not None
            overrides = {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: leaf.pipeline_family,
                cute_flash.FLASH_SOFTMAX_DISC_KEY: leaf.softmax_disc,
            }
            if leaf.compound_exp2_packet is not None:
                overrides[cute_flash.FLASH_EXP2_PACKET_KEY] = leaf.compound_exp2_packet
            leaf_generation = generation.config_spec.create_config_generation(
                overrides=overrides
            )
            projections = generation.canonicalize_coordinate_projections(
                leaf_generation.coordinate_neighbor_projections(
                    leaf_generation.flatten(base),
                    radius=2,
                ),
                base_config=base,
            )
            neighbor_projections[num_kv] = tuple(
                (
                    projection.flat_index,
                    projection.key,
                    projection.sequence_index,
                    projection.from_value,
                    projection.to_value,
                    (
                        "different_leaf"
                        if projection.config is not None
                        and cute_flash.flash_structural_leaf_from_config(
                            projection.config.config
                        )
                        != leaf
                        else projection.outcome
                    ),
                    (
                        None
                        if projection.config is None
                        else json.dumps(
                            projection.config.config,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                )
                for projection in projections
            )
        # The design consumes no randomness, so none of the surfaces above can
        # depend on the seed either.
        assert random.getstate() == before
        _assert_all_equal(fingerprints)
        _assert_all_equal(leaf_catalogs)
        _assert_all_equal(anchor_fingerprints)
        _assert_all_equal(starting_path_limits)
        _assert_all_equal(family_probe_limits)
        _assert_all_equal(terminal_catalogs)
        _assert_all_equal(neighbor_projections)

        if is_causal:
            assert next(iter(family_probe_limits.values())) == 0
        else:
            generation = _shared_flash_generation(
                64, _ALIGNED_LENGTHS[0], torch.float16, False
            )
            leaves = generation.flash_structural_leaf_catalog()
            ordinary_families = {
                leaf.pipeline_family
                for leaf in leaves
                if leaf.compound_exp2_packet is None
            }
            compound_count = sum(
                leaf.compound_exp2_packet is not None for leaf in leaves
            )
            assert (
                next(iter(family_probe_limits.values()))
                == 1 + len(ordinary_families) + compound_count
            )
    finally:
        random.setstate(random_state)


def test_flash_coordinate_neighbors_reach_dense_heldout_refinements() -> None:
    generation = _shared_flash_generation(64, 48, torch.float16, False)
    requested = generation.unflatten(generation.default_flat())
    requested.config.update(
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta",
            cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
            cute_flash.FLASH_EXP2_PACKET_KEY: "1x1",
            cute_flash.FLASH_STAT_TRANSPORT_KEY: "single",
            cute_flash.FLASH_RESCALE_THRESHOLD_KEY: 8.0,
            cute_flash.FLASH_CORR_TILE_SIZE_KEY: 8,
        }
    )
    leaf_generation = generation.config_spec.create_config_generation(
        overrides={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta",
            cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
        }
    )
    _flat, base = leaf_generation.canonicalize_flat(leaf_generation.flatten(requested))
    leaf = cute_flash.flash_structural_leaf_from_config(base.config)
    assert leaf is not None
    projections = generation.canonicalize_coordinate_projections(
        leaf_generation.coordinate_neighbor_projections(
            leaf_generation.flatten(base),
            radius=2,
        ),
        base_config=base,
    )

    requested_moves = {
        (projection.key, projection.to_value): projection
        for projection in projections
        if projection.outcome == "candidate"
        and projection.config is not None
        and cute_flash.flash_structural_leaf_from_config(projection.config.config)
        == leaf
    }
    assert (
        cute_flash.FLASH_STAT_TRANSPORT_KEY,
        "single_final",
    ) in requested_moves
    assert (cute_flash.FLASH_CORR_TILE_SIZE_KEY, 32) in requested_moves


@pytest.mark.parametrize(("dtype", "head_dim", "is_causal"), _SEMANTIC_CASES)
def test_flash_structural_coverage_reaches_and_qualifies_every_active_value(
    dtype: torch.dtype, head_dim: int, is_causal: bool
) -> None:
    _assert_structural_coverage(
        head_dim,
        48,
        dtype=dtype,
        is_causal=is_causal,
    )
    generation, configs = _structural_coverage_configs(
        head_dim,
        48,
        dtype=dtype,
        is_causal=is_causal,
    )
    assert len(configs) == len(set(configs))
    assert generation.flash_structural_coverage_underqualified_values() == []
    assert generation.flash_structural_coverage_underqualified_leaves() == []
    qualification_prefix_count = (
        generation.flash_structural_qualification_prefix_count()
    )
    parent_prefix_count = generation.flash_structural_parent_coverage_prefix_count()
    assert 0 < parent_prefix_count <= qualification_prefix_count
    assert 0 < qualification_prefix_count <= len(configs)
    parent_prefix = configs[:parent_prefix_count]
    qualification_prefix = configs[:qualification_prefix_count]
    leaf_catalog = generation.flash_structural_leaf_catalog()
    assert leaf_catalog == generation.flash_structural_coverage_active_leaves()
    for leaf in leaf_catalog:
        total_count = sum(
            cute_flash.flash_structural_leaf_from_config(config.config) == leaf
            for config in configs
        )
        prefix_count = sum(
            cute_flash.flash_structural_leaf_from_config(config.config) == leaf
            for config in qualification_prefix
        )
        required = 1 if leaf.compound_exp2_packet is not None else 2
        assert prefix_count >= min(required, total_count)
    fragments = generation.config_spec._flat_fields()
    family_values = _active_choices(
        cast("EnumFragment", fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY])
    )
    all_packet_values = _active_choices(
        cast("EnumFragment", fragments[cute_flash.FLASH_EXP2_PACKET_KEY])
    )
    packet_values = {
        packet
        for packet in all_packet_values
        if cute_flash.flash_exp2_packet_is_compound(packet)
    }

    for key, values in (
        (cute_flash.FLASH_PIPELINE_FAMILY_KEY, family_values),
        (cute_flash.FLASH_EXP2_PACKET_KEY, all_packet_values),
    ):
        for value in values:
            assert any(config.config[key] == value for config in parent_prefix)

    for value in family_values:
        assert (
            sum(
                config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == value
                for config in configs
            )
            >= 2
        )
        assert (
            sum(
                config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == value
                for config in qualification_prefix
            )
            >= 2
        )

    for value in packet_values:
        assert any(
            config.config[cute_flash.FLASH_EXP2_PACKET_KEY] == value
            for config in configs
        )
        assert any(
            config.config[cute_flash.FLASH_EXP2_PACKET_KEY] == value
            for config in qualification_prefix
        )


def test_flash_structural_coverage_does_not_consult_compiler_seeds() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    with (
        patch.object(
            spec,
            "autotune_seed_configs",
            side_effect=AssertionError("coverage must come from live fragments"),
        ),
        _memoized_flash_fragments(),
    ):
        coverage = (
            spec.create_config_generation().flash_deterministic_population_configs()
        )

    assert coverage


def test_flash_low_confound_schedule_anchors_follow_live_fragments() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    generation = spec.create_config_generation()
    axis_choices: dict[str, tuple[object, ...]] = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: ("fa4", "fa4_local_tma"),
        cute_flash.FLASH_EXP2_PACKET_KEY: ("1x1", "4x2"),
        cute_flash.FLASH_SOFTMAX_DISC_KEY: (True, False),
    }
    axis_indices: dict[str, int] = {}
    for key, choices in axis_choices.items():
        indices, is_sequence = generation._key_to_flat_indices[key]
        assert not is_sequence
        assert len(indices) == 1
        index = indices[0]
        fragment = generation.flat_spec[index]
        assert isinstance(fragment, EnumFragment)
        assert set(choices) <= set(fragment.choices)
        generation.flat_spec[index] = dataclasses.replace(
            fragment,
            search_choices=choices,
            coverage_choices=None,
        )
        axis_indices[key] = index

    anchors = generation.flash_low_confound_schedule_anchor_configs()
    expected_values = list(itertools.product(*axis_choices.values()))
    actual_values = [
        tuple(config.config[key] for key in axis_choices) for config in anchors
    ]
    assert actual_values == expected_values

    base = generation._fragment_default_flat()
    for anchor, values in zip(anchors, expected_values, strict=True):
        flat = copy.deepcopy(base)
        for key, value in zip(axis_choices, values, strict=True):
            flat[axis_indices[key]] = value
        assert generation.unflatten(flat) == anchor


@pytest.mark.parametrize(
    "overrides",
    (
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"},
        {cute_flash.FLASH_EXP2_PACKET_KEY: "4x2"},
        {cute_flash.FLASH_SOFTMAX_DISC_KEY: False},
    ),
    ids=("family", "packet", "softmax-protocol"),
)
def test_flash_low_confound_schedule_anchors_restrict_to_schedule_overrides(
    overrides: dict[str, object],
) -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    generation = spec.create_config_generation(overrides=overrides)
    anchors = generation.flash_low_confound_schedule_anchor_configs()
    unoverridden = spec.create_config_generation()
    expected = [
        config
        for config in unoverridden.flash_low_confound_schedule_anchor_configs()
        if all(config.config[key] == value for key, value in overrides.items())
    ]

    assert anchors
    assert anchors == expected


@pytest.mark.parametrize(
    ("retained_candidates_per_leaf", "expected"),
    ((1, 6), (2, 8)),
    ids=("one-candidate", "two-candidates"),
)
def test_flash_starting_path_limit_handles_asymmetric_family_widths(
    retained_candidates_per_leaf: int,
    expected: int,
) -> None:
    leaves = [
        cute_flash.FlashStructuralLeaf("fa4", None, True),
        cute_flash.FlashStructuralLeaf("fa4", None, False),
        cute_flash.FlashStructuralLeaf("ws_overlap", None, True),
        cute_flash.FlashStructuralLeaf("fa4_clc", None, True),
        cute_flash.FlashStructuralLeaf("fa4_2cta", "deg2_16x6", False),
        cute_flash.FlashStructuralLeaf("fa4_2cta", "deg1_16x8", False),
    ]
    generation = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    ).create_config_generation()

    with patch.object(
        generation, "flash_structural_leaf_catalog", return_value=leaves
    ) as catalog:
        limit = generation.flash_structural_starting_path_limit(
            minimum=1,
            retained_families=2,
            retained_candidates_per_leaf=retained_candidates_per_leaf,
        )

    assert limit == expected
    catalog.assert_called_once_with()


@pytest.mark.parametrize(
    ("retained_candidates_per_leaf", "expected"),
    ((1, 7), (2, 10)),
    ids=("one-candidate", "two-candidates"),
)
def test_flash_starting_path_limit_covers_every_live_family_when_unlimited(
    retained_candidates_per_leaf: int,
    expected: int,
) -> None:
    leaves = [
        cute_flash.FlashStructuralLeaf("fa4", None, True),
        cute_flash.FlashStructuralLeaf("fa4", None, False),
        cute_flash.FlashStructuralLeaf("ws_overlap", None, True),
        cute_flash.FlashStructuralLeaf("fa4_clc", None, True),
        cute_flash.FlashStructuralLeaf("fa4_2cta", "deg2_16x6", False),
        cute_flash.FlashStructuralLeaf("fa4_2cta", "deg1_16x8", False),
    ]
    generation = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    ).create_config_generation()

    with patch.object(generation, "flash_structural_leaf_catalog", return_value=leaves):
        limit = generation.flash_structural_starting_path_limit(
            minimum=1,
            retained_families=None,
            retained_candidates_per_leaf=retained_candidates_per_leaf,
        )

    assert limit == expected


@pytest.mark.parametrize("retained_candidates_per_leaf", (1, 2))
def test_flash_starting_path_limit_reserves_every_compound_only_leaf(
    retained_candidates_per_leaf: int,
) -> None:
    leaves = [
        cute_flash.FlashStructuralLeaf("fa4_2cta", "deg2_16x6", False),
        cute_flash.FlashStructuralLeaf("fa4_2cta", "deg1_16x8", False),
        cute_flash.FlashStructuralLeaf("fa4", "deg1_8x2_corr10", False),
    ]
    generation = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    ).create_config_generation()

    with patch.object(generation, "flash_structural_leaf_catalog", return_value=leaves):
        limit = generation.flash_structural_starting_path_limit(
            minimum=1,
            retained_families=4,
            retained_candidates_per_leaf=retained_candidates_per_leaf,
        )

    # Compound-only families are not parent-promotion candidates. Each live
    # compound leaf still receives capacity in addition to the global path.
    assert limit == 1 + len(leaves)


def test_flash_path_limits_follow_live_compound_catalog() -> None:
    generation = _shared_flash_generation(64, 48, torch.float16, False)
    leaves = generation.flash_structural_leaf_catalog()
    compound_count = sum(leaf.compound_exp2_packet is not None for leaf in leaves)
    live_family_count = len(
        {leaf.pipeline_family for leaf in leaves if leaf.compound_exp2_packet is None}
    )

    limit = generation.flash_structural_starting_path_limit(
        minimum=14,
        retained_families=None,
        retained_candidates_per_leaf=2,
    )
    assert compound_count > 0
    assert limit > 14

    # The family probe is disabled when the cap is unlimited or already covers
    # every live family, or when no candidates are retained per leaf.
    assert generation.flash_structural_family_probe_path_limit(None, 1) == 0
    assert (
        generation.flash_structural_family_probe_path_limit(live_family_count, 1) == 0
    )
    assert generation.flash_structural_family_probe_path_limit(4, 0) == 0


# The two semantic cases span both dtypes, both head dims, and dense+causal;
# each length class is checked at its two extremes for both cases.
@pytest.mark.parametrize(("dtype", "head_dim", "is_causal"), _POPULATION_CASES)
@pytest.mark.parametrize(
    "compiler_seeded", (False, True), ids=("cold", "compiler-seeded")
)
@pytest.mark.parametrize("num_kv_values", _LENGTH_CLASSES)
def test_flash_full_population_is_length_invariant(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    compiler_seeded: bool,
    num_kv_values: tuple[int, ...],
) -> None:
    random_state = random.getstate()
    try:
        populations = {}
        for num_kv in num_kv_values:
            spec = _flash_config_spec(
                head_dim=head_dim,
                num_kv=num_kv,
                dtype=dtype,
                is_causal=is_causal,
            )
            spec.compiler_seed_configs = (
                spec.autotune_seed_configs() if compiler_seeded else []
            )
            generation = spec.create_config_generation()
            random.seed(20260815)
            with _memoized_flash_fragments():
                configs = [
                    generation.unflatten(flat)
                    for flat in generation.random_population_flat(100)
                ]
            assert len(configs) == 100
            exact = generation.flash_exact_effective_search_space_configs(100)
            if exact is None:
                assert len(set(configs)) == 100
            else:
                assert set(configs) == set(exact)
            populations[num_kv] = tuple(
                json.dumps(config.config, sort_keys=True, separators=(",", ":"))
                for config in configs
            )
        _assert_all_equal(populations)
    finally:
        random.setstate(random_state)


def test_flash_structural_coverage_distinguishes_ordinary_and_compound_2cta() -> None:
    generation, configs = _structural_coverage_configs(
        64,
        48,
        dtype=torch.float16,
        is_causal=False,
    )
    prefix = configs[: generation.flash_structural_qualification_prefix_count()]
    expected = {
        cute_flash.FlashStructuralLeaf("fa4_2cta", None, True),
        cute_flash.FlashStructuralLeaf("fa4_2cta", None, False),
        cute_flash.FlashStructuralLeaf("fa4_2cta", "deg2_16x6", False),
        cute_flash.FlashStructuralLeaf("fa4_2cta", "deg1_16x8", False),
        cute_flash.FlashStructuralLeaf("fa4_2cta", "deg1_8x2_corr10", False),
    }

    catalog = set(generation.flash_structural_leaf_catalog())
    assert expected <= catalog
    for leaf in expected:
        required = 1 if leaf.compound_exp2_packet is not None else 2
        assert (
            sum(
                cute_flash.flash_structural_leaf_from_config(config.config) == leaf
                for config in prefix
            )
            >= required
        )

    ordinary_leaves = {
        leaf
        for leaf in catalog
        if leaf.pipeline_family == "fa4_2cta" and leaf.compound_exp2_packet is None
    }
    assert {leaf.softmax_disc for leaf in ordinary_leaves} == {True, False}
    for ordinary in ordinary_leaves:
        ordinary_packets = {
            config.config[cute_flash.FLASH_EXP2_PACKET_KEY]
            for config in prefix
            if cute_flash.flash_structural_leaf_from_config(config.config) == ordinary
        }
        assert ordinary_packets
        assert not any(
            cute_flash.flash_exp2_packet_is_compound(packet)
            for packet in ordinary_packets
        )


@pytest.mark.parametrize(
    "num_kv",
    (
        pytest.param(1, id="singleton"),
        pytest.param(33, id="odd"),
        pytest.param(34, id="paired-only"),
        pytest.param(48, id="divisible-by-four"),
    ),
)
def test_flash_structural_coverage_reaches_every_legality_class(
    num_kv: int,
) -> None:
    _assert_structural_coverage(
        64,
        num_kv,
        dtype=torch.float16,
        is_causal=False,
    )
    generation, configs = _structural_coverage_configs(
        64,
        num_kv,
        dtype=torch.float16,
        is_causal=False,
    )
    assert generation.flash_structural_coverage_underqualified_values() == []
    assert generation.flash_structural_coverage_underqualified_leaves() == []
    leaf_catalog = set(generation.flash_structural_leaf_catalog())
    assert leaf_catalog == {
        cute_flash.flash_structural_leaf_from_config(config.config)
        for config in configs
    }
    qualification_prefix = configs[
        : generation.flash_structural_qualification_prefix_count()
    ]
    for leaf in leaf_catalog:
        total_count = sum(
            cute_flash.flash_structural_leaf_from_config(config.config) == leaf
            for config in configs
        )
        assert sum(
            cute_flash.flash_structural_leaf_from_config(config.config) == leaf
            for config in qualification_prefix
        ) >= min(2, total_count)
    family_values = _active_choices(
        cast(
            "EnumFragment",
            generation.config_spec._flat_fields()[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
        )
    )
    for family in family_values:
        assert (
            sum(
                config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == family
                for config in configs
            )
            >= 2
        )


def test_flash_clc_single_head_coverage_rejects_auto_aliases() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        num_bh=1,
        dtype=torch.float16,
        is_causal=False,
    )
    generation = spec.create_config_generation()
    with _memoized_flash_fragments():
        coverage = generation.flash_deterministic_population_configs()
    clc_configs = [
        config
        for config in coverage
        if str(config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY]).startswith(
            "fa4_clc"
        )
    ]
    assert clc_configs
    assert all(
        config.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY] == 1
        for config in clc_configs
    )
    assert generation.flash_structural_coverage_underqualified_values() == []


@pytest.mark.parametrize("head_dim", (64, 128), ids=("d64", "d128"))
def test_flash_staged_epilogue_interaction_covers_every_combination(
    head_dim: int,
) -> None:
    generation, coverage = _structural_coverage_configs(
        head_dim,
        48,
        dtype=torch.float16,
        is_causal=False,
    )
    staged = {
        (
            config.config[cute_flash.FLASH_EPI_STG_STORE_KEY],
            config.config[cute_flash.FLASH_EPI_STG_GMEM_KEY],
        )
        for config in coverage
        if config.config[cute_flash.FLASH_EPI_STG_KEY]
    }
    assert staged >= {
        ("slice", "stage"),
        ("slice", "pair"),
        ("whole", "stage"),
        ("whole", "pair"),
    }
    assert generation.flash_structural_coverage_uncovered_interactions() == []


def test_flash_staged_epilogue_interaction_keeps_fixed_dependencies() -> None:
    spec = _flash_config_spec(
        head_dim=128,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    generation = spec.create_config_generation(
        overrides={cute_flash.FLASH_EPI_TMA_KEY: False}
    )
    with _memoized_flash_fragments():
        coverage = generation.flash_deterministic_population_configs()
    staged = {
        (
            config.config[cute_flash.FLASH_EPI_STG_STORE_KEY],
            config.config[cute_flash.FLASH_EPI_STG_GMEM_KEY],
        )
        for config in coverage
        if config.config[cute_flash.FLASH_EPI_STG_KEY]
    }
    assert staged >= {
        ("slice", "stage"),
        ("slice", "pair"),
        ("whole", "stage"),
        ("whole", "pair"),
    }
    active = generation.flash_structural_coverage_active_interactions()
    assert active
    assert all(
        values[0] is False
        for keys, values in active
        if keys == cute_flash.FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS[0]
    )
    assert generation.flash_structural_coverage_uncovered_interactions() == []


@pytest.mark.parametrize("num_bh", (360, 720))
def test_flash_clc_all_divisors_have_bounded_anchors_and_full_qualification(
    num_bh: int,
) -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        num_bh=num_bh,
        dtype=torch.float16,
        is_causal=False,
    )
    fragment = cast(
        "EnumFragment",
        spec._flat_fields()[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY],
    )
    legal = tuple(
        value for value in fragment.choices if type(value) is int and value > 0
    )
    assert fragment.search_choices is not None
    active = tuple(
        value for value in fragment.search_choices if type(value) is int and value > 0
    )
    assert active == legal
    assert fragment.coverage_choices is not None
    anchors = tuple(
        value for value in fragment.coverage_choices if type(value) is int and value > 0
    )
    assert 0 < len(anchors) <= 8
    refinements = _flash_log_maximin_refinements(legal, anchors)
    planned = (*anchors, *refinements)
    assert set(planned) == set(legal)
    assert len(planned) == len(legal)
    assert set(legal) <= {
        fragment.default(),
        *fragment.pattern_neighbors(fragment.default()),
    }
    with patch(
        "helion.autotuner.config_fragment.random.choice", return_value=refinements[0]
    ):
        assert fragment.random() == refinements[0]

    generation = spec.create_config_generation()
    with _memoized_flash_fragments():
        coverage = generation.flash_deterministic_population_configs()
    assert len(coverage) <= generation.flash_structural_population_budget(100)
    assert generation.flash_structural_coverage_uncovered_values() == []
    assert generation.flash_structural_coverage_underqualified_values() == []
    assert generation.flash_structural_coverage_uncovered_interactions() == []

    catalogs = generation.flash_clc_lane_catalog()
    witnesses = generation.flash_clc_lane_witnesses()
    assert catalogs
    for leaf, catalog in catalogs.items():
        assert catalog["legal_values"] == legal
        assert catalog["search_values"] == legal
        assert catalog["anchor_values"] == anchors
        assert catalog["refinement_values"] == refinements
        assert catalog["attempted_values"] == planned
        assert {
            value for witness_leaf, value in witnesses if witness_leaf == leaf
        } == set(legal)

    leaf, catalog = next(iter(catalogs.items()))
    source = copy.deepcopy(witnesses[(leaf, catalog["attempted_values"][0])])
    source.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY] = refinements[0]
    _flat, canonical = generation.canonicalize_flat(generation.flatten(source))
    assert cute_flash.flash_structural_leaf_from_config(canonical.config) == leaf
    assert canonical.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY] == refinements[0]


def test_flash_clc_effective_default_is_reserved_inside_coverage_budget() -> None:
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_CLC_HEADS": "10"}):
        spec = _flash_config_spec(
            head_dim=64,
            num_kv=130,
            num_bh=180,
            tensor_4d_heads=36,
            dtype=torch.float16,
            is_causal=False,
        )
        fragments = cute_flash.flash_autotune_fragments(
            64,
            130,
            num_bh=180,
            tensor_4d_heads=36,
            dtype=torch.float16,
            standard_dense_output=True,
            pipeline_family_override="fa4_clc",
        )
        fragment = cast(
            "EnumFragment", fragments[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY]
        )
        legal = tuple(
            value for value in fragment.choices if type(value) is int and value > 0
        )
        search = tuple(
            value
            for value in fragment.search_choices or ()
            if type(value) is int and value > 0
        )
        anchors = tuple(
            value
            for value in fragment.coverage_choices or ()
            if type(value) is int and value > 0
        )
        assert fragment.default() == 10
        assert len(legal) == 18
        assert search == legal
        assert len(anchors) <= 8
        assert {1, 5, 10, 36, 180} <= set(anchors)

        overrides = {
            key: value.default()
            for key, value in fragments.items()
            if key != cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY
        }
        overrides["block_sizes"] = [1, 128, 128]
        generation = spec.create_config_generation(overrides=overrides)
        catalogs = generation.flash_clc_lane_catalog()
        assert catalogs
        assert all(catalog["search_values"] == legal for catalog in catalogs.values())
        assert all(len(catalog["anchor_values"]) <= 8 for catalog in catalogs.values())
        assert all(
            set(catalog["attempted_values"]) == set(legal)
            for catalog in catalogs.values()
        )
        assert all(18 in catalog["attempted_values"] for catalog in catalogs.values())

        exact = generation.flash_exact_effective_search_space_configs(100)
        assert exact is not None
        assert {
            config.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY] for config in exact
        } == set(legal)

        clc_index = generation._key_to_flat_indices[
            cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY
        ][0][0]
        x = generation.default_flat()
        a, b, c = (copy.deepcopy(x) for _ in range(3))
        a[clc_index] = 10
        b[clc_index] = 18
        c[clc_index] = 1
        with (
            patch("helion.autotuner.config_generation.random.random", return_value=0.0),
            patch("helion.autotuner.config_fragment.random.choice", return_value=18),
        ):
            mutated = generation.differential_mutation(x, a, b, c, 1.0)
        assert mutated[clc_index] == 18
        assert (
            generation.unflatten(mutated).config[
                cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY
            ]
            == 18
        )


def test_flash_clc_legacy_divisor_override_outside_coverage_remains_legal() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        num_bh=4096,
        dtype=torch.float16,
        is_causal=False,
    )
    generation = spec.create_config_generation(
        overrides={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_clc",
            cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY: 128,
        }
    )
    with _memoized_flash_fragments():
        coverage = generation.flash_deterministic_population_configs()
    assert coverage
    assert all(
        config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4_clc"
        and config.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY] == 128
        for config in coverage
    )


def test_flash_length_classes_preserve_only_structural_legality_boundaries() -> None:
    singleton = _active_choice_sets(64, 1, dtype=torch.float16, is_causal=False)
    divisible_by_four = {
        num_kv: _active_choice_sets(64, num_kv, dtype=torch.float16, is_causal=False)
        for num_kv in (4, 32, 48, 96, 384)
    }
    paired_only = {
        num_kv: _active_choice_sets(64, num_kv, dtype=torch.float16, is_causal=False)
        for num_kv in (2, 34, 98, 386)
    }
    odd = {
        num_kv: _active_choice_sets(64, num_kv, dtype=torch.float16, is_causal=False)
        for num_kv in (3, 33, 49, 97)
    }

    _assert_all_equal(divisible_by_four)
    _assert_all_equal(paired_only)
    _assert_all_equal(odd)
    divisible_families = next(iter(divisible_by_four.values()))[
        cute_flash.FLASH_PIPELINE_FAMILY_KEY
    ]
    paired_families = next(iter(paired_only.values()))[
        cute_flash.FLASH_PIPELINE_FAMILY_KEY
    ]
    odd_families = next(iter(odd.values()))[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    singleton_families = singleton[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    assert singleton_families == _effective_pipeline_families(
        64, 1, dtype=torch.float16, is_causal=False
    )
    assert odd_families == _effective_pipeline_families(
        64, 33, dtype=torch.float16, is_causal=False
    )
    assert paired_families == _effective_pipeline_families(
        64, 34, dtype=torch.float16, is_causal=False
    )
    assert divisible_families == _effective_pipeline_families(
        64, 48, dtype=torch.float16, is_causal=False
    )
    assert singleton_families == odd_families == frozenset(("ws_overlap",))
    assert paired_families < divisible_families


@pytest.mark.parametrize("requested_stage", (2, 3, 4, 10))
def test_singleton_kv_stage_is_canonicalized_before_search(
    requested_stage: int,
) -> None:
    resolved = cute_flash.resolve_flash_config(
        64,
        1,
        {cute_flash.FLASH_KV_STAGE_KEY: requested_stage},
        **_shape_options(torch.float16, False),
    )
    assert resolved.kv_stage == 1

    spec = _flash_config_spec(
        head_dim=64,
        num_kv=1,
        dtype=torch.float16,
        is_causal=False,
    )
    fragment = cast("EnumFragment", spec._flat_fields()[cute_flash.FLASH_KV_STAGE_KEY])
    assert fragment.choices == (1,)
    repaired = helion.Config.from_dict(
        {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_KV_STAGE_KEY: requested_stage,
        }
    )
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired.config[cute_flash.FLASH_KV_STAGE_KEY] == 1


@pytest.mark.parametrize(("dtype", "head_dim", "is_causal"), _SEMANTIC_CASES)
@pytest.mark.parametrize(
    ("num_kv", "expected_by_shape"),
    (
        pytest.param(
            1,
            {(64, False): 2, (64, True): 2, (128, False): 2, (128, True): 2},
            id="singleton",
        ),
        pytest.param(
            3,
            {(64, False): 18, (64, True): 12, (128, False): 6, (128, True): 4},
            id="odd",
        ),
    ),
)
def test_small_flash_search_spaces_are_exhaustive(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    num_kv: int,
    expected_by_shape: dict[tuple[int, bool], int],
) -> None:
    spec = _flash_config_spec(
        head_dim=head_dim,
        num_kv=num_kv,
        dtype=dtype,
        is_causal=is_causal,
    )
    generation = spec.create_config_generation()
    exact = generation.flash_exact_effective_search_space_configs(100)
    assert exact is not None
    assert len(exact) == expected_by_shape[(head_dim, is_causal)]

    random_state = random.getstate()
    try:
        random.seed(20260815)
        population = generation.random_population_flat(100)
    finally:
        random.setstate(random_state)
    population_configs = {generation.unflatten(flat) for flat in population}
    assert len(population) == 100
    assert population_configs == set(exact)


def test_large_flash_search_space_is_not_partially_enumerated() -> None:
    generation = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    ).create_config_generation()
    assert generation.flash_exact_effective_search_space_configs(100) is None


@pytest.mark.parametrize(
    ("is_causal", "schedule", "expected_transport", "expected_stage"),
    (
        pytest.param(True, "16/4", "ring2", 6, id="causal-disc"),
        pytest.param(False, "xu", "single", 6, id="dense-xu-single"),
        pytest.param(False, "16/4", "single", 6, id="dense-single"),
    ),
)
def test_whole_row_ring2_limits_kv_pipeline_depth(
    is_causal: bool,
    schedule: str,
    expected_transport: str,
    expected_stage: int,
) -> None:
    resolved = cute_flash.resolve_flash_config(
        64,
        64,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_KV_STAGE_KEY: 6,
            cute_flash.FLASH_E2E_SCHEDULE_KEY: schedule,
            cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
            cute_flash.FLASH_STAT_TRANSPORT_KEY: "ring2",
        },
        **_shape_options(torch.float16, is_causal),
    )
    assert resolved.stat_transport == expected_transport
    assert resolved.kv_stage == expected_stage


def test_causal_unsplit_loop_uses_one_exp2_cadence() -> None:
    common = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/6",
        cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "16/4",
        cute_flash.FLASH_P_STORE_REP_KEY: 32,
    }
    unsplit = cute_flash.resolve_flash_config(
        64,
        768,
        {**common, cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: False},
        **_shape_options(torch.float16, True),
    )
    split = cute_flash.resolve_flash_config(
        64,
        768,
        {**common, cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True},
        **_shape_options(torch.float16, True),
    )

    assert not unsplit.causal_loop_split
    assert unsplit.masked_e2e_schedule == "inherit"
    assert (unsplit.masked_e2e_freq, unsplit.masked_e2e_res) == (16, 6)
    assert unsplit.p_store_repetition == 16
    assert split.causal_loop_split
    assert split.masked_e2e_schedule == "16/4"
    assert (split.masked_e2e_freq, split.masked_e2e_res) == (16, 4)
    assert split.p_store_repetition == 16

    coverage = _shared_flash_generation(
        64, 48, torch.float16, True
    ).flash_deterministic_population_configs()
    assert any(
        config.config[cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY] == "16/4"
        and config.config[cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY] is True
        for config in coverage
    )
    assert all(
        config.config[cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY] == "inherit"
        or config.config[cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY] is True
        for config in coverage
    )


@pytest.mark.parametrize(
    ("head_dim", "requested_stage", "expected_stage"),
    (
        pytest.param(64, 10, 4, id="d64"),
        pytest.param(128, 10, 2, id="d128"),
    ),
)
@pytest.mark.parametrize(
    "requires_ws_overlap",
    (
        pytest.param(False, id="pinned-ws"),
        pytest.param(True, id="required-ws"),
    ),
)
def test_ws_pipeline_depth_is_bounded_by_separate_ring_storage(
    head_dim: int,
    requested_stage: int,
    expected_stage: int,
    requires_ws_overlap: bool,
) -> None:
    resolved = cute_flash.resolve_flash_config(
        head_dim,
        64,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_KV_STAGE_KEY: requested_stage,
        },
        **_shape_options(torch.float16, False),
        requires_ws_overlap=requires_ws_overlap,
    )
    assert resolved.pipeline_family == "ws_overlap"
    assert resolved.kv_stage == expected_stage


@pytest.mark.parametrize(
    ("head_dim", "requested_stage", "expected_stage"),
    (
        pytest.param(64, 10, 4, id="d64"),
        pytest.param(128, 3, 2, id="d128"),
    ),
)
@pytest.mark.parametrize("requires_ws_overlap", (False, True))
def test_ws_legacy_kv_stage_is_canonicalized_before_fixed_config_validation(
    head_dim: int,
    requested_stage: int,
    expected_stage: int,
    requires_ws_overlap: bool,
) -> None:
    with (
        patch(
            "helion.autotuner.config_spec.get_target_device_capability",
            return_value=(10, 0),
        ),
        patch(
            "helion.autotuner.config_spec.supports_tensor_descriptor",
            return_value=True,
        ),
        patch("helion.autotuner.config_spec.get_num_xcd", return_value=1),
        patch("helion.autotuner.config_spec.device_num_sm", return_value=148),
    ):
        spec = _flash_config_spec(
            head_dim=head_dim,
            num_kv=48,
            dtype=torch.float16,
            is_causal=False,
            requires_ws_overlap=requires_ws_overlap,
        )
    fixed = helion.Config.from_dict(
        {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_KV_STAGE_KEY: requested_stage,
        }
    )
    spec.normalize(fixed)
    assert fixed.config[cute_flash.FLASH_KV_STAGE_KEY] == expected_stage


@pytest.mark.parametrize(
    ("head_dim", "expected_stages"),
    (
        pytest.param(64, frozenset((2, 3, 4)), id="d64"),
        pytest.param(128, frozenset((2,)), id="d128"),
    ),
)
@pytest.mark.parametrize(
    "requires_ws_overlap",
    (
        pytest.param(False, id="pinned-ws"),
        pytest.param(True, id="required-ws"),
    ),
)
def test_ws_kv_stage_search_covers_exact_storage_capacity(
    head_dim: int,
    expected_stages: frozenset[int],
    requires_ws_overlap: bool,
) -> None:
    fragments = cute_flash.flash_autotune_fragments(
        head_dim,
        48,
        **_shape_options(torch.float16, False),
        requires_ws_overlap=requires_ws_overlap,
        pipeline_family_override=None if requires_ws_overlap else "ws_overlap",
    )
    kv_stage = cast("EnumFragment", fragments[cute_flash.FLASH_KV_STAGE_KEY])
    assert _active_choices(kv_stage) == expected_stages
    for stage in expected_stages:
        resolved = cute_flash.resolve_flash_config(
            head_dim,
            48,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                cute_flash.FLASH_KV_STAGE_KEY: stage,
            },
            **_shape_options(torch.float16, False),
            requires_ws_overlap=requires_ws_overlap,
        )
        assert resolved.kv_stage == stage


@pytest.mark.parametrize(
    ("head_dim", "kv_stage"),
    (
        pytest.param(64, 12, id="d64-direct-cap"),
        pytest.param(128, 5, id="d128-direct-cap"),
    ),
)
def test_direct_output_kv_depth_fixed_override_remains_legal(
    head_dim: int,
    kv_stage: int,
) -> None:
    spec = _flash_config_spec(
        head_dim=head_dim,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    generation = spec.create_config_generation(
        overrides={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_KV_STAGE_KEY: kv_stage,
            cute_flash.FLASH_EPI_TMA_KEY: False,
            cute_flash.FLASH_EPI_STG_KEY: False,
        }
    )

    assert generation._override_values[cute_flash.FLASH_KV_STAGE_KEY] == kv_stage
    default = generation.unflatten(generation.default_flat())
    assert default.config[cute_flash.FLASH_KV_STAGE_KEY] == kv_stage
    assert default.config[cute_flash.FLASH_EPI_TMA_KEY] is False
    assert default.config[cute_flash.FLASH_EPI_STG_KEY] is False


@pytest.mark.parametrize(
    ("head_dim", "family", "expected_values", "expected_search"),
    (
        pytest.param(
            64,
            "ws_overlap",
            (2, 3, 4, 6, 8, 10),
            (2, 3, 4),
            id="d64-ws",
        ),
        pytest.param(128, "ws_overlap", (2, 3), (2,), id="d128-ws"),
        pytest.param(
            64,
            "fa4_deep_1cta",
            (2, 3, 4),
            (2, 3, 4),
            id="d64-separate",
        ),
        pytest.param(
            128,
            "fa4_deep_1cta",
            (2,),
            (2,),
            id="d128-separate",
        ),
    ),
)
def test_aliased_kv_expansion_does_not_broaden_other_pipeline_domains(
    head_dim: int,
    family: str,
    expected_values: tuple[int, ...],
    expected_search: tuple[int, ...],
) -> None:
    fragments = cute_flash.flash_autotune_fragments(
        head_dim,
        48,
        **_shape_options(torch.float16, False),
        pipeline_family_override=family,
    )
    kv_stage = cast("EnumFragment", fragments[cute_flash.FLASH_KV_STAGE_KEY])

    assert frozenset(kv_stage.choices) == frozenset(expected_values)
    assert kv_stage.search_choices is not None
    assert frozenset(kv_stage.search_choices) == frozenset(expected_search)


@pytest.mark.parametrize("is_causal", (False, True))
def test_d64_aliased_kv_depths_resolve_for_every_search_family(
    is_causal: bool,
) -> None:
    options = _shape_options(torch.float16, is_causal)
    fragments = cute_flash.flash_autotune_fragments(64, 48, **options)
    family_fragment = cast(
        "EnumFragment", fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    )
    aliased_families = {
        cast("str", family)
        for family in _active_choices(family_fragment)
        if (
            flags := cute_flash.FLASH_PIPELINE_FAMILY_FLAGS[cast("str", family)]
        ).topology
        == "fa4"
        and not flags.separate_kv_rings
    }

    assert aliased_families
    for family in aliased_families:
        for stage in (5, 7, 9):
            resolved = cute_flash.resolve_flash_config(
                64,
                48,
                {
                    cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
                    cute_flash.FLASH_KV_STAGE_KEY: stage,
                },
                **options,
            )
            assert resolved.pipeline_family == family
            assert resolved.kv_stage == stage


@pytest.mark.parametrize(
    ("head_dim", "epi_tma", "epi_stg", "expected_cap"),
    (
        pytest.param(64, False, False, 12, id="d64-direct-output"),
        pytest.param(64, False, True, 10, id="d64-staged-output"),
        pytest.param(64, True, False, 10, id="d64-tma-output"),
        pytest.param(128, False, False, 5, id="d128-direct-output"),
        pytest.param(128, False, True, 3, id="d128-staged-output"),
        pytest.param(128, True, False, 3, id="d128-tma-output"),
    ),
)
def test_aliased_kv_depth_normalizes_to_output_storage_capacity(
    head_dim: int,
    epi_tma: bool,
    epi_stg: bool,
    expected_cap: int,
) -> None:
    resolved = cute_flash.resolve_flash_config(
        head_dim,
        48,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_KV_STAGE_KEY: 99,
            cute_flash.FLASH_EPI_TMA_KEY: epi_tma,
            cute_flash.FLASH_EPI_STG_KEY: epi_stg,
        },
        **_shape_options(torch.float16, False),
    )

    assert resolved.kv_stage == expected_cap


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize(
    "search_mode",
    (
        pytest.param("family", id="pinned-family"),
        pytest.param("topology", id="pinned-topology"),
        pytest.param("required", id="required-ws"),
    ),
)
def test_ws_search_has_bounded_effective_active_value_coverage(
    head_dim: int,
    search_mode: str,
) -> None:
    requires_ws_overlap = search_mode == "required"
    fragments = cute_flash.flash_autotune_fragments(
        head_dim,
        48,
        **_shape_options(torch.float16, False),
        requires_ws_overlap=requires_ws_overlap,
        pipeline_family_override="ws_overlap" if search_mode == "family" else None,
        topology_override="ws_overlap" if search_mode == "topology" else None,
    )
    enum_fragments = {
        key: cast("EnumFragment", fragment)
        for key, fragment in fragments.items()
        if isinstance(fragment, EnumFragment)
    }
    base = {key: fragment.default() for key, fragment in enum_fragments.items()}
    active_values = [
        (key, value)
        for key, fragment in enum_fragments.items()
        for value in _active_choices(fragment)
    ]

    assert _active_choices(
        enum_fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    ) == frozenset(("ws_overlap",))
    assert len(active_values) <= 52
    for key, value in active_values:
        requested = {**base, key: value}
        resolved = cute_flash.resolve_flash_config(
            head_dim,
            48,
            requested,
            **_shape_options(torch.float16, False),
            requires_ws_overlap=requires_ws_overlap,
        )
        effective = cute_flash.flash_effective_config_values(resolved)
        assert effective[key] == value, (key, value, effective[key])


@pytest.mark.parametrize(
    ("head_dim", "expected_stages"),
    (
        pytest.param(64, frozenset((2, 3, 4)), id="d64"),
        pytest.param(128, frozenset((2,)), id="d128"),
    ),
)
@pytest.mark.parametrize(
    "requires_ws_overlap",
    (
        pytest.param(False, id="pinned-ws"),
        pytest.param(True, id="required-ws"),
    ),
)
def test_ws_structural_generation_is_bounded_and_complete(
    head_dim: int,
    expected_stages: frozenset[int],
    requires_ws_overlap: bool,
) -> None:
    with (
        patch(
            "helion.autotuner.config_spec.get_target_device_capability",
            return_value=(10, 0),
        ),
        patch(
            "helion.autotuner.config_spec.supports_tensor_descriptor",
            return_value=True,
        ),
        patch("helion.autotuner.config_spec.get_num_xcd", return_value=1),
        patch("helion.autotuner.config_spec.device_num_sm", return_value=148),
    ):
        spec = _flash_config_spec(
            head_dim=head_dim,
            num_kv=48,
            dtype=torch.float16,
            is_causal=False,
            requires_ws_overlap=requires_ws_overlap,
        )
    overrides = (
        {}
        if requires_ws_overlap
        else {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
    )
    generation = spec.create_config_generation(overrides=overrides)
    with _memoized_flash_fragments():
        configs = generation.flash_deterministic_population_configs()

    assert configs
    assert len(configs) <= 3
    assert generation.flash_structural_coverage_uncovered_values() == []
    assert {
        config.config[cute_flash.FLASH_KV_STAGE_KEY] for config in configs
    } == expected_stages


@pytest.mark.parametrize("head_dim", (64, 128), ids=("d64", "d128"))
def test_output_tma_structural_design_covers_every_advertised_value(
    head_dim: int,
) -> None:
    spec = _flash_config_spec(
        head_dim=head_dim,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
        supports_tensor_4d_tma=False,
        output_requires_tma=True,
    )
    generation = spec.create_config_generation()
    with _memoized_flash_fragments():
        assert generation.flash_deterministic_population_configs()
    assert generation.flash_structural_coverage_uncovered_values() == []
    packed_reduce = cast(
        "EnumFragment", spec._flat_fields()[cute_flash.FLASH_PACKED_REDUCE_KEY]
    )
    assert _active_choices(packed_reduce) == frozenset((True,))


def test_output_tma_rejects_an_explicit_non_tma_override() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
        supports_tensor_4d_tma=False,
        output_requires_tma=True,
    )
    with pytest.raises(InvalidConfig, match="not legal for this output shape"):
        spec.create_config_generation(overrides={cute_flash.FLASH_EPI_TMA_KEY: False})

    config = helion.Config(
        block_sizes=[1, 128, 128],
        cute_flash_pipeline_family="fa4",
        cute_flash_epi_tma=False,
    )
    with pytest.raises(InvalidConfig, match="not legal for this output shape"):
        spec.normalize(config)


def test_pinned_two_cta_search_excludes_canonicalized_child_values() -> None:
    fragments = cute_flash.flash_autotune_fragments(
        64,
        512,
        num_bh=2,
        dtype=torch.float16,
        standard_dense_output=True,
        pipeline_family_override="fa4_2cta",
    )

    assert _active_choices(
        cast("EnumFragment", fragments[cute_flash.FLASH_RECOMPUTE_TILE_COORDS_KEY])
    ) == frozenset((False,))
    assert _active_choices(
        cast("EnumFragment", fragments[cute_flash.FLASH_EPI_STG_GMEM_KEY])
    ) == frozenset(("stage",))
    assert _active_choices(
        cast("EnumFragment", fragments[cute_flash.FLASH_PACKED_REDUCE_KEY])
    ) == frozenset((True,))
    assert (
        len(
            _active_choices(
                cast("EnumFragment", fragments[cute_flash.FLASH_PERSISTENT_KEY])
            )
        )
        > 1
    )
    assert (
        len(
            _active_choices(
                cast("EnumFragment", fragments[cute_flash.FLASH_EPI_STG_KEY])
            )
        )
        > 1
    )


@pytest.mark.parametrize("family", cute_flash.FLASH_AUTOTUNE_PIPELINE_FAMILIES)
def test_fixed_family_search_surface_has_complete_deterministic_coverage(
    family: str,
) -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=512,
        num_bh=2,
        tensor_4d_heads=1,
        dtype=torch.float16,
        is_causal=False,
    )
    effective = cute_flash.resolve_flash_config(
        64,
        512,
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: family},
        num_bh=2,
        dtype=torch.float16,
        standard_dense_output=True,
    )
    if effective.pipeline_family != family:
        pytest.skip("family is not legal for this semantic class")

    generation = ConfigGeneration(
        spec,
        _flash_pipeline_family_override=family,
    )
    with _memoized_flash_fragments():
        configs = generation.flash_deterministic_population_configs()

    assert configs
    assert {
        config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] for config in configs
    } == {family}
    assert generation.flash_structural_coverage_uncovered_values() == []
    assert generation.flash_structural_coverage_uncovered_interactions() == []


def test_output_tma_fix_invalid_repairs_non_tma_family() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
        supports_tensor_4d_tma=False,
        output_requires_tma=True,
    )
    config = helion.Config(
        block_sizes=[1, 128, 128],
        cute_flash_pipeline_family="ws_overlap",
        cute_flash_epi_tma=False,
    )

    spec.normalize(config, _fix_invalid=True)

    assert config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] != "ws_overlap"
    assert config.config[cute_flash.FLASH_EPI_TMA_KEY] is True
    assert config.config[cute_flash.FLASH_EPI_STG_KEY] is False


@pytest.mark.parametrize("family", ("fa4_local_tma", "fa4_local_tma_4d"))
def test_local_tma_family_override_pins_persistence(family: str) -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    generation = spec.create_config_generation(
        overrides={cute_flash.FLASH_PIPELINE_FAMILY_KEY: family}
    )

    assert generation._override_values[cute_flash.FLASH_PERSISTENT_KEY] is True
    for config in generation.random_population(4):
        assert config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == family
        assert config.config[cute_flash.FLASH_PERSISTENT_KEY] is True

    with pytest.raises(InvalidConfig, match="requires cute_flash_persistent=True"):
        spec.create_config_generation(
            overrides={
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
                cute_flash.FLASH_PERSISTENT_KEY: False,
            }
        )


def test_compound_packet_override_pins_its_parent_schedule() -> None:
    spec = _flash_config_spec(
        head_dim=128,
        num_kv=48,
        dtype=torch.bfloat16,
        is_causal=False,
    )
    packet = "deg2_16x6"
    generation = spec.create_config_generation(
        overrides={cute_flash.FLASH_EXP2_PACKET_KEY: packet}
    )

    assert (
        generation._override_values[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4_2cta"
    )
    # random_population would validate the packet-pinned structural design,
    # which is far more expensive than the pin propagation under test; sampled
    # configs go through the same override normalization.
    random_state = random.getstate()
    try:
        random.seed(20260815)
        for _ in range(4):
            config = generation.random_config()
            assert config.config[cute_flash.FLASH_EXP2_PACKET_KEY] == packet
            assert config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4_2cta"
    finally:
        random.setstate(random_state)

    with pytest.raises(InvalidConfig, match="requires cute_flash_pipeline_family"):
        spec.create_config_generation(
            overrides={
                cute_flash.FLASH_EXP2_PACKET_KEY: packet,
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            }
        )


def test_exact_effective_enumeration_counts_overrides_as_singletons() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    fields = spec._flat_fields_with_flash_family("fa4")
    overrides = {
        key: fragment.default()
        for key, fragment in fields.items()
        if isinstance(fragment, EnumFragment)
        and key
        not in (
            cute_flash.FLASH_PIPELINE_FAMILY_KEY,
            cute_flash.FLASH_WAIT_HINT_KEY,
        )
    }
    overrides[cute_flash.FLASH_PIPELINE_FAMILY_KEY] = "fa4"
    generation = spec.create_config_generation(overrides=overrides)

    configs = generation.flash_exact_effective_search_space_configs(2)
    assert configs is not None
    assert len(configs) == 2
    assert {config.config[cute_flash.FLASH_WAIT_HINT_KEY] for config in configs} == {
        0,
        10_000_000,
    }


def test_exact_effective_enumeration_does_not_require_legal_default_point() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    fields = spec._flat_fields_with_flash_family("fa4")
    overrides = {
        key: fragment.default()
        for key, fragment in fields.items()
        if isinstance(fragment, EnumFragment) and key != cute_flash.FLASH_OTHER_REGS_KEY
    }
    overrides[cute_flash.FLASH_PIPELINE_FAMILY_KEY] = "fa4"
    overrides[cute_flash.FLASH_CORR_REGS_KEY] = 88
    generation = spec.create_config_generation(overrides=overrides)

    configs = generation.flash_exact_effective_search_space_configs(7)

    assert configs is not None
    assert len(configs) == 1
    assert configs[0].config[cute_flash.FLASH_CORR_REGS_KEY] == 88
    assert configs[0].config[cute_flash.FLASH_OTHER_REGS_KEY] == 24


def test_override_collapsed_flash_search_does_not_require_two_witnesses() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    fragments = spec._flat_fields()
    overrides = {
        key: fragment.default()
        for key in cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
        if key != cute_flash.FLASH_PIPELINE_FAMILY_KEY
        and isinstance((fragment := fragments[key]), EnumFragment)
    }
    generation = spec.create_config_generation(overrides=overrides)

    with _memoized_flash_fragments():
        assert generation.flash_deterministic_population_configs()
    assert generation.flash_structural_coverage_uncovered_values()
    assert generation.flash_structural_coverage_underqualified_values()


def test_flash_leaf_conditional_generator_uses_real_family_surfaces() -> None:
    generation = _shared_flash_generation(64, 48, torch.float16, False)
    representatives = {}
    for config in generation.flash_deterministic_population_configs():
        leaf = cute_flash.flash_structural_leaf_from_config(config.config)
        if leaf is not None:
            representatives.setdefault(leaf, config)
    assert set(representatives) == set(generation.flash_structural_leaf_catalog())

    search = LFBOPatternSearch.__new__(LFBOPatternSearch)
    search.config_spec = generation.config_spec
    search.config_gen = generation
    search.num_neighbors = 60
    search.radius = 2
    search.num_neighbors_cap = -1

    random_state = random.getstate()
    try:
        random.seed(123)
        with _memoized_flash_fragments():
            for leaf, config in representatives.items():
                current = PopulationMember(
                    fn=lambda: None,
                    perfs=[1.0],
                    flat_values=generation.flatten(config),
                    config=config,
                    status="ok",
                )
                neighbors = search._generate_flash_leaf_neighbors(current, leaf)
                assert neighbors
                neighbor_configs = [generation.unflatten(flat) for flat in neighbors]
                assert len(neighbor_configs) == len(set(neighbor_configs))
                assert all(
                    cute_flash.flash_structural_leaf_from_config(item.config) == leaf
                    for item in neighbor_configs
                )
    finally:
        random.setstate(random_state)


def _pipeline_lanes_by_leaf(
    num_kv: int,
    *,
    head_dim: int = 128,
    is_causal: bool = True,
) -> dict[cute_flash.FlashStructuralLeaf, tuple[tuple[str, object], ...]]:
    generation = _shared_flash_generation(head_dim, num_kv, torch.float16, is_causal)
    search = LFBOPatternSearch.__new__(LFBOPatternSearch)
    search.config_spec = generation.config_spec
    search.config_gen = generation
    return {
        leaf: search._flash_pipeline_lanes(leaf)
        for leaf in generation.flash_structural_leaf_catalog()
    }


def test_pipeline_qualification_lanes_are_length_invariant() -> None:
    # Both lengths are paired but not cluster-aligned. The normalized family
    # domains, rather than a sequence-length table, must define the lanes.
    short = _pipeline_lanes_by_leaf(34)
    long = _pipeline_lanes_by_leaf(98)

    assert short == long
    assert short
    assert all(
        lane[0] in (cute_flash.FLASH_KV_STAGE_KEY, cute_flash.FLASH_S_STAGE_KEY)
        for lanes in short.values()
        for lane in lanes
    )
    assert any(
        lane[0] == cute_flash.FLASH_KV_STAGE_KEY
        for lanes in short.values()
        for lane in lanes
    )


@pytest.mark.parametrize(
    ("head_dim", "direct_only_depths"),
    (
        pytest.param(64, (11, 12), id="d64"),
        pytest.param(128, (4, 5), id="d128"),
    ),
)
def test_high_depth_pipeline_witnesses_select_direct_output(
    head_dim: int,
    direct_only_depths: tuple[int, ...],
) -> None:
    generation = _shared_flash_generation(head_dim, 48, torch.float16, False)
    leaf = cute_flash.FlashStructuralLeaf("fa4", None, True)
    witnesses = generation.flash_pipeline_lane_witnesses()

    assert generation.flash_structural_coverage_uncovered_values() == []
    for depth in direct_only_depths:
        config = witnesses[(leaf, cute_flash.FLASH_KV_STAGE_KEY, depth)]
        assert config.config[cute_flash.FLASH_KV_STAGE_KEY] == depth
        assert config.config[cute_flash.FLASH_EPI_TMA_KEY] is False
        assert config.config[cute_flash.FLASH_EPI_STG_KEY] is False


@pytest.mark.parametrize(
    ("dtype", "head_dim", "is_causal"),
    (
        pytest.param(torch.float16, 64, False, id="fp16-d64-dense"),
        pytest.param(torch.bfloat16, 128, True, id="bf16-d128-causal"),
    ),
)
def test_pipeline_lane_catalog_has_deterministic_witnesses(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
) -> None:
    generation = _shared_flash_generation(head_dim, 48, dtype, is_causal)
    catalog = generation.flash_pipeline_lane_catalog()
    witnesses = generation.flash_pipeline_lane_witnesses()
    expected_witness_keys = {
        (leaf, key, value) for leaf, lanes in catalog.items() for key, value in lanes
    }

    assert set(witnesses) == expected_witness_keys
    assert any(leaf.compound_exp2_packet is not None for leaf in catalog)
    for (leaf, key, value), config in witnesses.items():
        flat, normalized = generation.canonicalize_flat(generation.flatten(config))
        assert generation.unflatten(flat) == normalized
        assert cute_flash.flash_structural_leaf_from_config(normalized.config) == leaf
        assert normalized.config[key] == value


def test_singleton_pipeline_space_adds_no_qualification_lanes() -> None:
    lanes_by_leaf = _pipeline_lanes_by_leaf(
        1,
        head_dim=64,
        is_causal=False,
    )

    assert lanes_by_leaf
    assert all(not lanes for lanes in lanes_by_leaf.values())


def test_fixed_pipeline_depths_add_no_qualification_lanes() -> None:
    spec = _flash_config_spec(
        head_dim=128,
        num_kv=34,
        dtype=torch.float16,
        is_causal=True,
    )
    generation = spec.create_config_generation(
        overrides={
            cute_flash.FLASH_KV_STAGE_KEY: 2,
            cute_flash.FLASH_S_STAGE_KEY: 2,
        }
    )
    search = LFBOPatternSearch.__new__(LFBOPatternSearch)
    search.config_spec = spec
    search.config_gen = generation

    with _memoized_flash_fragments():
        leaves = generation.flash_structural_leaf_catalog()
    assert leaves
    for leaf in leaves:
        assert search._flash_pipeline_lanes(leaf) == ()


def test_inactive_small_biased_setting_is_canonicalized() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    fixed = helion.Config.from_dict(
        {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_SMALL_BIASED_KEY: False,
        }
    )
    spec.normalize(fixed)
    assert fixed.config[cute_flash.FLASH_SMALL_BIASED_KEY] is True


@pytest.mark.parametrize(
    ("head_dim", "is_causal"),
    (
        pytest.param(64, True, id="causal-d64"),
        pytest.param(128, False, id="dense-d128"),
    ),
)
def test_unsupported_role_chain_setting_is_canonicalized(
    head_dim: int,
    is_causal: bool,
) -> None:
    spec = _flash_config_spec(
        head_dim=head_dim,
        num_kv=48,
        dtype=torch.float16,
        is_causal=is_causal,
    )
    fixed = helion.Config.from_dict(
        {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_ROLE_CHAIN_KEY: True,
        }
    )
    spec.normalize(fixed)
    assert fixed.config[cute_flash.FLASH_ROLE_CHAIN_KEY] is False


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("standard_causal_output", (False, True))
def test_causal_search_excludes_unacknowledged_whole_row_transport(
    dtype: torch.dtype,
    standard_causal_output: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in tuple(os.environ):
        if key.startswith("HELION_CUTE_FLASH"):
            monkeypatch.delenv(key)
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=768,
        dtype=dtype,
        is_causal=True,
        standard_causal_output=standard_causal_output,
    )
    softmax_fragment = cast(
        "EnumFragment", spec._flat_fields()[cute_flash.FLASH_SOFTMAX_DISC_KEY]
    )
    assert _active_choices(softmax_fragment) == frozenset((True,))
    masked_schedule_fragment = cast(
        "EnumFragment",
        spec._flat_fields()[cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY],
    )
    expected_masked_schedules = (
        frozenset(("inherit", "xu", "16/4", "8/2"))
        if standard_causal_output
        else frozenset(("inherit",))
    )
    assert frozenset(masked_schedule_fragment.choices) >= frozenset(
        ("inherit", "xu", "16/4", "8/2")
    )
    assert _active_choices(masked_schedule_fragment) == expected_masked_schedules
    with _memoized_flash_fragments():
        coverage = spec.create_config_generation(
            overrides={}
        ).flash_deterministic_population_configs()
    assert {
        config.config[cute_flash.FLASH_SOFTMAX_DISC_KEY] for config in coverage
    } == {True}

    fixed = helion.Config.from_dict(
        {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
            cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "16/4",
            cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        }
    )
    spec.normalize(fixed)
    assert fixed.config[cute_flash.FLASH_SOFTMAX_DISC_KEY] is True
    assert fixed.config[cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY] == (
        "16/4" if standard_causal_output else "inherit"
    )


def test_fa4_ineligible_causal_config_accepts_legacy_masked_cadence() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=47,
        dtype=torch.float16,
        is_causal=True,
    )
    fragment = cast(
        "EnumFragment",
        spec._flat_fields()[cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY],
    )
    assert frozenset(fragment.choices) == frozenset(("inherit", "xu", "16/4", "8/2"))
    assert _active_choices(fragment) == frozenset(("inherit",))

    fixed = helion.Config.from_dict(
        {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "16/4",
        }
    )
    spec.normalize(fixed)
    assert fixed.config[cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY] == "inherit"


@pytest.mark.parametrize(
    ("num_kv", "is_causal", "requires_ws_overlap"),
    (
        (48, True, False),
        (34, False, False),
        (49, False, False),
        (48, False, True),
    ),
)
def test_family_pinned_generation_rejects_structurally_ineligible_family(
    num_kv: int,
    is_causal: bool,
    requires_ws_overlap: bool,
) -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=num_kv,
        dtype=torch.float16,
        is_causal=is_causal,
        requires_ws_overlap=requires_ws_overlap,
    )
    eligible = _effective_pipeline_families(
        64,
        num_kv,
        dtype=torch.float16,
        is_causal=is_causal,
        requires_ws_overlap=requires_ws_overlap,
    )
    family_fragment = cast(
        "EnumFragment", spec._flat_fields()[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    )
    expected_active = eligible - ({"fa4_deep_1cta"} if is_causal else set())
    assert _active_choices(family_fragment) == expected_active
    ineligible = set(cute_flash.FLASH_AUTOTUNE_PIPELINE_FAMILIES) - eligible
    for family in ineligible:
        with pytest.raises(InvalidConfig, match=r"is not (?:legal|effective)"):
            generation = spec.create_config_generation(
                overrides={cute_flash.FLASH_PIPELINE_FAMILY_KEY: family}
            )
            generation.unflatten(generation.default_flat())


def test_causal_deep_ring_family_requires_an_explicit_search_pin() -> None:
    default_fragments = cute_flash.flash_autotune_fragments(
        64,
        768,
        **_shape_options(torch.float16, True),
    )
    default_family = cast(
        "EnumFragment", default_fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    )
    assert "fa4_deep_1cta" not in _active_choices(default_family)

    pinned_fragments = cute_flash.flash_autotune_fragments(
        64,
        768,
        **_shape_options(torch.float16, True),
        pipeline_family_override="fa4_deep_1cta",
    )
    pinned_family = cast(
        "EnumFragment", pinned_fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    )
    assert _active_choices(pinned_family) == frozenset(("fa4_deep_1cta",))


@pytest.mark.parametrize(
    (
        "family",
        "kv_stage",
        "disc_pipe_depth",
        "packet",
        "e2e_schedule",
        "masked_e2e_schedule",
    ),
    (
        pytest.param(
            "fa4_deep_1cta",
            3,
            3,
            "4x2",
            "8/2",
            "inherit",
            id="deep-kv3-disc3",
        ),
        pytest.param(
            "fa4",
            10,
            1,
            "4x1",
            "16/8",
            "16/4",
            id="fa4-kv10-disc1",
        ),
    ),
)
def test_causal_split_rep32_timeout_configs_are_canonicalized(
    family: str,
    kv_stage: int,
    disc_pipe_depth: int,
    packet: str,
    e2e_schedule: str,
    masked_e2e_schedule: str,
) -> None:
    common = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
        cute_flash.FLASH_KV_STAGE_KEY: kv_stage,
        cute_flash.FLASH_KV_ORDER_KEY: "descending",
        cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
        cute_flash.FLASH_P_STORE_REP_KEY: 32,
        cute_flash.FLASH_DISC_PIPE_KEY: disc_pipe_depth,
        cute_flash.FLASH_EXP2_PACKET_KEY: packet,
        cute_flash.FLASH_E2E_SCHEDULE_KEY: e2e_schedule,
        cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: masked_e2e_schedule,
    }
    mixed = cute_flash.resolve_flash_config(
        64,
        768,
        {**common, cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True},
        **_shape_options(torch.float16, True),
    )
    unsplit_store = cute_flash.resolve_flash_config(
        64,
        768,
        {**common, cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: False},
        **_shape_options(torch.float16, True),
    )

    assert mixed.kv_order == "ascending"
    assert mixed.pipeline_family == family
    assert mixed.kv_stage == kv_stage
    assert mixed.disc_pipe_depth == disc_pipe_depth
    assert mixed.exp2_packet == packet
    assert mixed.p_store_repetition == 16
    assert unsplit_store.kv_order == "ascending"
    assert unsplit_store.pipeline_family == family
    assert unsplit_store.kv_stage == kv_stage
    assert unsplit_store.disc_pipe_depth == disc_pipe_depth
    assert unsplit_store.exp2_packet == packet
    assert unsplit_store.p_store_repetition == 32


@pytest.mark.parametrize(
    "family",
    (
        "fa4",
        "ws_overlap",
        "fa4_clc",
        "fa4_clc_tma_4d",
        "fa4_clc_local_tma",
        "fa4_clc_local_tma_4d",
    ),
)
def test_family_pinned_generation_rejects_conflicting_manual_packet(
    family: str,
) -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    with pytest.raises(InvalidConfig, match="requires cute_flash_pipeline_family"):
        generation = spec.create_config_generation(
            overrides={
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
                cute_flash.FLASH_EXP2_PACKET_KEY: "deg1_16x8",
            }
        )
        generation.unflatten(generation.default_flat())


@pytest.mark.parametrize(
    ("key", "conflicting_value", "required_value"),
    (
        (cute_flash.FLASH_Q_TILE_COUNT_KEY, 1, 2),
        (cute_flash.FLASH_P_STORE_REP_KEY, 32, 16),
        (cute_flash.FLASH_S_LOAD_REP_KEY, 16, 32),
        (cute_flash.FLASH_SOFTMAX_DISC_KEY, True, False),
        (cute_flash.FLASH_E2E_SCHEDULE_KEY, "8/2", "16/8"),
    ),
)
def test_compound_packet_rejects_conflicting_explicit_children(
    key: str,
    conflicting_value: object,
    required_value: object,
) -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    common = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta",
        cute_flash.FLASH_EXP2_PACKET_KEY: "deg1_16x8",
    }
    with pytest.raises(InvalidConfig, match=rf"requires {key}="):
        generation = spec.create_config_generation(
            overrides={**common, key: conflicting_value}
        )
        generation.unflatten(generation.default_flat())

    generation = spec.create_config_generation(
        overrides={**common, key: required_value}
    )
    config = generation.unflatten(generation.default_flat())
    assert config.config[key] == required_value


def test_compound_packet_preserves_parameterized_required_child() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.bfloat16,
        is_causal=True,
    )
    generation = spec.create_config_generation(
        overrides={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_EXP2_PACKET_KEY: "hybrid_deg1_16x8",
            cute_flash.FLASH_DISC_PIPE_KEY: 3,
        }
    )
    config = generation.unflatten(generation.default_flat())
    assert config.config[cute_flash.FLASH_DISC_PIPE_KEY] == 3


def test_unpinned_compound_packet_rejects_conflicting_explicit_child() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    with pytest.raises(InvalidConfig, match="requires cute_flash_p_store_rep=16"):
        generation = spec.create_config_generation(
            overrides={
                cute_flash.FLASH_EXP2_PACKET_KEY: "deg1_16x8",
                cute_flash.FLASH_P_STORE_REP_KEY: 32,
            }
        )
        generation.unflatten(generation.default_flat())


def test_ws_pinned_generation_removes_dead_packet_aliases() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    fragments = cute_flash.flash_autotune_fragments(
        64,
        48,
        **_shape_options(torch.float16, False),
        pipeline_family_override="ws_overlap",
    )
    packet_fragment = cast("EnumFragment", fragments[cute_flash.FLASH_EXP2_PACKET_KEY])
    assert _active_choices(packet_fragment) == frozenset(("1x1",))

    with pytest.raises(InvalidConfig, match="is not effective with"):
        generation = spec.create_config_generation(
            overrides={
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                cute_flash.FLASH_EXP2_PACKET_KEY: "4x1",
            }
        )
        generation.unflatten(generation.default_flat())

    generation = spec.create_config_generation(
        overrides={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_EXP2_PACKET_KEY: "4x1",
        }
    )
    config = generation.unflatten(generation.default_flat())
    assert config.config[cute_flash.FLASH_EXP2_PACKET_KEY] == "4x1"


@pytest.mark.parametrize(
    "family",
    (
        "fa4_clc",
        "fa4_clc_tma_4d",
        "fa4_clc_local_tma",
        "fa4_clc_local_tma_4d",
    ),
)
def test_pinned_clc_families_reject_disabled_persistence(family: str) -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    with pytest.raises(InvalidConfig, match="requires cute_flash_persistent=True"):
        generation = spec.create_config_generation(
            overrides={
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
                cute_flash.FLASH_PERSISTENT_KEY: False,
            }
        )
        generation.unflatten(generation.default_flat())


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("is_causal", (False, True))
def test_d128_fa4_search_retains_general_schedule_and_arithmetic_choices(
    dtype: torch.dtype,
    is_causal: bool,
) -> None:
    keys = {
        cute_flash.FLASH_E2E_SCHEDULE_KEY,
        cute_flash.FLASH_E2E_OFFSET_KEY,
        cute_flash.FLASH_E2E_OFFSET0_KEY,
        cute_flash.FLASH_DISC_PIPE_KEY,
        cute_flash.FLASH_RESCALE_THRESHOLD_KEY,
        cute_flash.FLASH_RESCALE_CHUNK_COLS_KEY,
        cute_flash.FLASH_SOFTMAX_REGS_KEY,
        cute_flash.FLASH_CORR_REGS_KEY,
        cute_flash.FLASH_OTHER_REGS_KEY,
        cute_flash.FLASH_CORR_TILE_SIZE_KEY,
    }
    if is_causal:
        keys.add(cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY)
    else:
        keys.add(cute_flash.FLASH_PERSISTENT_LOOP_KEY)

    surfaces: dict[int, dict[str, frozenset[object]]] = {}
    for num_kv in _ALIGNED_LENGTHS:
        fragments = cute_flash.flash_autotune_fragments(
            128,
            num_kv,
            **_shape_options(dtype, is_causal),
            pipeline_family_override="fa4",
        )
        surfaces[num_kv] = {
            key: _active_choices(cast("EnumFragment", fragments[key])) for key in keys
        }
        assert all(len(choices) > 1 for choices in surfaces[num_kv].values())
        assert _active_choices(
            cast("EnumFragment", fragments[cute_flash.FLASH_PACKED_REDUCE_KEY])
        ) == frozenset((True,))
    _assert_all_equal(surfaces)


@pytest.mark.parametrize(
    ("is_causal", "num_kv", "values", "preserved"),
    (
        pytest.param(
            False,
            2048,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta",
                cute_flash.FLASH_EXP2_PACKET_KEY: "deg2_16x6",
                cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/6",
                cute_flash.FLASH_E2E_OFFSET_KEY: 3,
                cute_flash.FLASH_E2E_OFFSET0_KEY: 6,
                cute_flash.FLASH_RESCALE_THRESHOLD_KEY: 12.0,
                cute_flash.FLASH_CORR_TILE_SIZE_KEY: 16,
            },
            {
                cute_flash.FLASH_E2E_OFFSET_KEY: 3,
                cute_flash.FLASH_E2E_OFFSET0_KEY: 6,
                cute_flash.FLASH_RESCALE_THRESHOLD_KEY: 12.0,
                cute_flash.FLASH_CORR_TILE_SIZE_KEY: 16,
            },
            id="dense",
        ),
        pytest.param(
            True,
            4096,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
                cute_flash.FLASH_EXP2_PACKET_KEY: (
                    "causal_hd128_resident3_013_prefetch2_deg2_early_acquire"
                ),
                cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/6",
                cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "16/6",
                cute_flash.FLASH_E2E_OFFSET_KEY: 4,
                cute_flash.FLASH_E2E_OFFSET0_KEY: 6,
                cute_flash.FLASH_RESCALE_THRESHOLD_KEY: 12.0,
                cute_flash.FLASH_CORR_TILE_SIZE_KEY: 8,
            },
            {
                cute_flash.FLASH_E2E_OFFSET_KEY: 4,
                cute_flash.FLASH_E2E_OFFSET0_KEY: 6,
                cute_flash.FLASH_RESCALE_THRESHOLD_KEY: 12.0,
                cute_flash.FLASH_CORR_TILE_SIZE_KEY: 8,
            },
            id="causal",
        ),
    ),
)
def test_validated_bf16_d128_values_survive_config_normalization(
    is_causal: bool,
    num_kv: int,
    values: dict[str, object],
    preserved: dict[str, object],
) -> None:
    spec = _flash_config_spec(
        head_dim=128,
        num_kv=num_kv,
        dtype=torch.bfloat16,
        is_causal=is_causal,
    )
    config = helion.Config.from_dict({"block_sizes": [1, 128, 128], **values})
    spec.normalize(config)

    assert all(config.config[key] == value for key, value in preserved.items())


def test_legacy_exp2_schedule_normalization_preserves_valid_cadence() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=768,
        dtype=torch.float16,
        is_causal=False,
    )

    legacy_split = helion.Config(
        block_sizes=[1, 128, 128],
        cute_flash_e2e_schedule="xu",
        cute_flash_exp2_impl="split",
    )
    spec.normalize(legacy_split)
    assert legacy_split.config[cute_flash.FLASH_E2E_SCHEDULE_KEY] == "16/4"
    assert legacy_split.config[cute_flash.FLASH_E2E_OFFSET_KEY] == 2

    legacy_wide = helion.Config(
        block_sizes=[1, 128, 128],
        cute_flash_e2e_schedule="xu",
        cute_flash_exp2_impl="split",
        cute_flash_e2e_freq=32,
        cute_flash_e2e_res=4,
        cute_flash_e2e_offset=31,
        cute_flash_e2e_offset0=31,
    )
    spec.normalize(legacy_wide)
    assert legacy_wide.config[cute_flash.FLASH_E2E_OFFSET_KEY] == 31
    assert legacy_wide.config[cute_flash.FLASH_E2E_OFFSET0_KEY] == 31

    for invalid_offset in (-1, 99):
        invalid = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_e2e_schedule="16/4",
            cute_flash_e2e_offset=invalid_offset,
        )
        with pytest.raises(InvalidConfig):
            spec.normalize(invalid)

    for schedule, invalid_offset, expected in (
        ("16/4", -1, (2, 0)),
        ("8/2", 99, (3, 3)),
    ):
        repaired = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_e2e_schedule=schedule,
            cute_flash_e2e_offset=invalid_offset,
            cute_flash_e2e_offset0=invalid_offset,
        )
        spec.normalize(repaired, _fix_invalid=True)
        assert (
            repaired.config[cute_flash.FLASH_E2E_OFFSET_KEY],
            repaired.config[cute_flash.FLASH_E2E_OFFSET0_KEY],
        ) == expected

    causal_spec = _flash_config_spec(
        head_dim=64,
        num_kv=768,
        dtype=torch.float16,
        is_causal=True,
    )
    masked_cadence = helion.Config(
        block_sizes=[1, 128, 128],
        cute_flash_e2e_schedule="xu",
        cute_flash_masked_e2e_schedule="16/4",
        cute_flash_e2e_offset=15,
        cute_flash_e2e_offset0=14,
        cute_flash_causal_kv_order="descending",
        cute_flash_causal_loop_split=True,
    )
    causal_spec.normalize(masked_cadence)
    assert masked_cadence.config[cute_flash.FLASH_E2E_SCHEDULE_KEY] == "xu"
    assert masked_cadence.config[cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY] == "16/4"
    assert masked_cadence.config[cute_flash.FLASH_E2E_OFFSET_KEY] == 15
    assert masked_cadence.config[cute_flash.FLASH_E2E_OFFSET0_KEY] == 14


def test_causal_lpt_swizzle_is_bounded_to_the_stress_tested_envelope() -> None:
    resolved = cute_flash.resolve_flash_config(
        64,
        48,
        {cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: 0},
        num_bh=3,
        is_causal=True,
        standard_causal_output=True,
    )
    assert resolved.causal_lpt_swizzle == 1

    fragments = cute_flash.flash_autotune_fragments(
        64,
        48,
        num_bh=3,
        is_causal=True,
        standard_causal_output=True,
    )
    swizzle = fragments[cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY]
    assert isinstance(swizzle, EnumFragment)
    assert 0 in swizzle.choices
    assert swizzle.search_choices == (1,)

    wide = cute_flash.resolve_flash_config(
        64,
        48,
        {cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: 64},
        num_bh=64,
        is_causal=True,
        standard_causal_output=True,
    )
    assert wide.causal_lpt_swizzle == 1

    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=True,
    )
    legacy_wide = helion.Config(
        block_sizes=[1, 128, 128],
        cute_flash_causal_lpt_swizzle=64,
    )
    spec.normalize(legacy_wide)
    assert legacy_wide.config[cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] == 1


@pytest.mark.parametrize("num_kv", _ALIGNED_LENGTHS)
@pytest.mark.parametrize("packet", sorted(_DENSE_DEG1_PACKETS))
def test_dense_fp16_degree1_packets_are_not_shape_remapped(
    num_kv: int, packet: str
) -> None:
    packet_choices = _active_choice_sets(
        64, num_kv, dtype=torch.float16, is_causal=False
    )[cute_flash.FLASH_EXP2_PACKET_KEY]
    assert packet_choices >= _DENSE_DEG1_PACKETS
    config = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta",
        cute_flash.FLASH_EXP2_PACKET_KEY: packet,
        cute_flash.FLASH_P_STORE_REP_KEY: 16,
        cute_flash.FLASH_S_LOAD_REP_KEY: 32,
    }
    with patch.dict(os.environ, {}, clear=True):
        resolved = cute_flash.resolve_flash_config(
            64,
            num_kv,
            config,
            dtype=torch.float16,
            is_causal=False,
            standard_dense_output=True,
        )
    assert resolved.exp2_packet == packet


def _emit_dense_source(
    num_kv: int,
    overrides: dict[str, object],
    *,
    head_dim: int = 64,
) -> str:
    values: dict[str, object] = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_PERSISTENT_KEY: True,
        **overrides,
    }
    with patch.dict(os.environ, {}, clear=True):
        config = cute_flash.resolve_flash_config(
            head_dim,
            num_kv,
            values,
            dtype=torch.float16,
            num_bh=64,
            standard_dense_output=True,
        )
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=head_dim,
        num_kv=num_kv,
        sequence_extent=num_kv * 128,
        num_bh=64,
        total_tiles=64 * num_kv // 2,
        cfg=config,
        has_lse=False,
        io_dtype="cutlass.Float16",
        score_plan=dense_score_plan(head_dim),
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_new_aliased_kv_depths_remain_distinct_after_normalization() -> None:
    spec = _flash_config_spec(
        head_dim=64,
        num_kv=48,
        dtype=torch.float16,
        is_causal=False,
    )
    normalized_configs: list[helion.Config] = []
    sources: set[str] = set()
    for kv_stage in (5, 7, 9):
        config = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_kv_stage=kv_stage,
            cute_flash_epi_tma=False,
            cute_flash_epi_stg=False,
        )
        spec.normalize(config)
        assert config.config[cute_flash.FLASH_KV_STAGE_KEY] == kv_stage
        normalized_configs.append(config)
        source = _emit_dense_source(
            48,
            {
                cute_flash.FLASH_KV_STAGE_KEY: kv_stage,
                cute_flash.FLASH_EPI_TMA_KEY: False,
                cute_flash.FLASH_EPI_STG_KEY: False,
            },
        )
        assert f"num_stages={kv_stage}" in source
        sources.add(source)

    assert len(_config_fingerprints(normalized_configs)) == 3
    assert len(sources) == 3


@pytest.mark.parametrize("num_kv", (32, 4096))
@pytest.mark.parametrize(
    ("key", "baseline", "variant", "dependencies"),
    (
        (
            cute_flash.FLASH_PERSISTENT_LOOP_KEY,
            "while",
            "counted",
            {},
        ),
        (
            cute_flash.FLASH_SP_ROW_SUM_KEY,
            "fragment",
            "whole",
            {cute_flash.FLASH_SOFTMAX_DISC_KEY: False},
        ),
        (
            cute_flash.FLASH_SOFTMAX_SETUP_KEY,
            "shared",
            "stage_local",
            {},
        ),
        (
            cute_flash.FLASH_EPI_TMA_SETUP_KEY,
            "shared",
            "role_local",
            {cute_flash.FLASH_EPI_TMA_KEY: True},
        ),
    ),
)
def test_source_schedule_choices_change_emitted_program_at_short_and_long_lengths(
    num_kv: int,
    key: str,
    baseline: object,
    variant: object,
    dependencies: dict[str, object],
) -> None:
    baseline_source = _emit_dense_source(
        num_kv,
        {**dependencies, key: baseline},
    )
    variant_source = _emit_dense_source(
        num_kv,
        {**dependencies, key: variant},
    )
    assert variant_source != baseline_source


def test_d128_dense_persistent_loop_choices_emit_different_programs() -> None:
    while_source = _emit_dense_source(
        384,
        {cute_flash.FLASH_PERSISTENT_LOOP_KEY: "while"},
        head_dim=128,
    )
    counted_source = _emit_dense_source(
        384,
        {cute_flash.FLASH_PERSISTENT_LOOP_KEY: "counted"},
        head_dim=128,
    )

    assert while_source != counted_source
    assert "while flash_tile_id <" in while_source
    assert "for flash_tile_iter in cutlass.range(flash_tile_count, unroll=1)" in (
        counted_source
    )


@pytest.mark.parametrize(
    ("key", "value", "dependencies", "expected"),
    (
        (
            cute_flash.FLASH_PERSISTENT_LOOP_KEY,
            "counted",
            {cute_flash.FLASH_PERSISTENT_KEY: False},
            "while",
        ),
        (
            cute_flash.FLASH_SP_ROW_SUM_KEY,
            "whole",
            {cute_flash.FLASH_SOFTMAX_DISC_KEY: True},
            "fragment",
        ),
        (
            cute_flash.FLASH_EPI_TMA_SETUP_KEY,
            "role_local",
            {cute_flash.FLASH_EPI_TMA_KEY: False},
            "shared",
        ),
    ),
)
def test_inactive_source_schedule_choices_canonicalize(
    key: str,
    value: object,
    dependencies: dict[str, object],
    expected: object,
) -> None:
    resolved = cute_flash.resolve_flash_config(
        64,
        384,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            **dependencies,
            key: value,
        },
        dtype=torch.float16,
        standard_dense_output=True,
    )
    assert cute_flash.flash_effective_config_values(resolved)[key] == expected


@pytest.mark.parametrize("num_kv", (512, 2048))
def test_unsafe_cga2_local_stage_local_softmax_canonicalizes(num_kv: int) -> None:
    values: dict[str, object] = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_cga2_local",
        cute_flash.FLASH_PERSISTENT_KEY: True,
        cute_flash.FLASH_SOFTMAX_DISC_KEY: True,
        cute_flash.FLASH_STAT_TRANSPORT_KEY: "single",
        cute_flash.FLASH_RESCALE_THRESHOLD_KEY: 0.0,
        cute_flash.FLASH_SOFTMAX_SETUP_KEY: "stage_local",
    }
    resolved = cute_flash.resolve_flash_config(
        64,
        num_kv,
        values,
        dtype=torch.float16,
        num_bh=64,
        standard_dense_output=True,
    )

    assert resolved.use_cga2_local_cta
    assert resolved.softmax_setup == "shared"

    with (
        patch(
            "helion.autotuner.config_spec.get_target_device_capability",
            return_value=(10, 0),
        ),
        patch(
            "helion.autotuner.config_spec.supports_tensor_descriptor",
            return_value=True,
        ),
        patch("helion.autotuner.config_spec.get_num_xcd", return_value=1),
        patch("helion.autotuner.config_spec.device_num_sm", return_value=148),
    ):
        spec = _flash_config_spec(
            head_dim=64,
            num_kv=num_kv,
            dtype=torch.float16,
            is_causal=False,
        )
    assert "stage_local" in _active_choices(
        cast(
            "EnumFragment",
            spec._flat_fields()[cute_flash.FLASH_SOFTMAX_SETUP_KEY],
        )
    )
    generation = spec.create_config_generation()
    requested = helion.Config.from_dict({"block_sizes": [1, 128, 128], **values})
    _, normalized = generation.canonicalize_flat(generation.flatten(requested))
    assert normalized.config[cute_flash.FLASH_SOFTMAX_SETUP_KEY] == "shared"

    leaf_generation = spec.create_config_generation(
        overrides={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_cga2_local",
            cute_flash.FLASH_SOFTMAX_DISC_KEY: True,
        }
    )
    projections = generation.canonicalize_coordinate_projections(
        leaf_generation.coordinate_neighbor_projections(
            leaf_generation.flatten(normalized), radius=1
        ),
        base_config=normalized,
    )
    stage_local = next(
        projection
        for projection in projections
        if projection.key == cute_flash.FLASH_SOFTMAX_SETUP_KEY
        and projection.to_value == "stage_local"
    )
    assert stage_local.outcome == "incumbent_alias"
    assert stage_local.config == normalized


@pytest.mark.parametrize(
    "overrides",
    (
        {cute_flash.FLASH_RESCALE_THRESHOLD_KEY: 8.0},
        {cute_flash.FLASH_PERSISTENT_KEY: False},
        {cute_flash.FLASH_SOFTMAX_DISC_KEY: False},
        {cute_flash.FLASH_STAT_TRANSPORT_KEY: "ring2"},
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta",
            cute_flash.FLASH_PERSISTENT_KEY: False,
        },
    ),
)
def test_cga2_local_stage_local_softmax_safety_controls(
    overrides: dict[str, object],
) -> None:
    resolved = cute_flash.resolve_flash_config(
        64,
        2048,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_cga2_local",
            cute_flash.FLASH_PERSISTENT_KEY: True,
            cute_flash.FLASH_SOFTMAX_DISC_KEY: True,
            cute_flash.FLASH_STAT_TRANSPORT_KEY: "single",
            cute_flash.FLASH_RESCALE_THRESHOLD_KEY: 0.0,
            cute_flash.FLASH_SOFTMAX_SETUP_KEY: "stage_local",
            **overrides,
        },
        dtype=torch.float16,
        num_bh=64,
        standard_dense_output=True,
    )

    assert resolved.softmax_setup == "stage_local"


def test_persistent_loop_canonicalizes_for_clc() -> None:
    resolved = cute_flash.resolve_flash_config(
        64,
        384,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_clc",
            cute_flash.FLASH_PERSISTENT_KEY: True,
            cute_flash.FLASH_PERSISTENT_LOOP_KEY: "counted",
        },
        dtype=torch.float16,
        num_bh=64,
        standard_dense_output=True,
    )
    assert resolved.use_clc_scheduler
    assert resolved.persistent_loop == "while"
