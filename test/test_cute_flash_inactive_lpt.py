"""The causal launch group is a free coordinate of every fa4 lowering.

Only the WS leaf and dense kernels keep the inactive ``0``; no lowering owns
the coordinate, so explicit groups are honoured (and clamped to the grid) on
``auto``/``standard`` as on the explicit stateful lowering.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
from .test_cute_flash_stateful_lpt import _spec
from helion._compiler.cute import cute_flash as flash
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with _mock_cuda_unavailable(), _forbid_native_compile():
        yield


@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("topology", ("fa4", "ws_overlap"))
def test_causal_launch_group_is_free_on_every_lowering(
    causal: bool, topology: str
) -> None:
    # The fa4 causal body decodes its grid pair-major inside the requested
    # group (0 names FA4's L2-sized group, sized by the emitter); the WS leaf
    # and dense kernels keep the inactive 0.
    for lowering in ("auto", "standard"):
        for width in (0, 1, 8, 64):
            resolved = flash.resolve_flash_config(
                64,
                32,
                {
                    flash.FLASH_PIPELINE_FAMILY_KEY: topology,
                    flash.FLASH_SOFTMAX_LOWERING_KEY: lowering,
                    flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: width,
                },
                is_causal=causal,
                num_bh=64,
                standard_causal_output=causal,
                standard_dense_output=not causal,
            )
            owned = flash._flash_resident_softmax_overrides(lowering)
            assert flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY not in owned
            expected = width if causal and resolved.topology == "fa4" else 0
            assert resolved.causal_lpt_swizzle == expected


@pytest.mark.parametrize("disc", (False, True))
@pytest.mark.parametrize("width", (None, 0, 1))
def test_causal_ws_leaf_retains_historical_width_aliases(
    disc: bool, width: int | None
) -> None:
    spec = _spec(64)
    overrides = {
        flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
        flash.FLASH_SOFTMAX_DISC_KEY: disc,
    }
    if width is not None:
        overrides[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] = width
    generation = spec.create_config_generation(overrides=overrides)
    fragment = generation._flat_fields()[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY]
    assert isinstance(fragment, EnumFragment)
    assert fragment._active_choices() == (0,)
    flat, config = generation.canonicalize_flat(generation.default_flat())
    assert config.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] == 0
    assert config.config[flash.FLASH_PIPELINE_FAMILY_KEY] == "ws_overlap"
    # WS has always canonicalized both requests to its DISC implementation.
    assert config.config[flash.FLASH_SOFTMAX_DISC_KEY] is True
    assert generation.canonicalize_flat(flat)[1] == config
    for alias in (0, 1):
        requested = {**config.config, flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: alias}
        assert spec.normalized_config(requested) == config
    root = ConfigGeneration(spec)
    owned = root.flash_owned_coordinate_indices(config)
    identities = root._flat_coordinate_identities()
    # No lowering owns the launch group; the WS leaf's fragment is what keeps
    # it at 0.
    assert flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY not in {identities[i][0] for i in owned}
    assert flash.FLASH_SOFTMAX_LOWERING_KEY not in {identities[i][0] for i in owned}


@pytest.mark.parametrize("alias", (0, 1))
def test_narrowed_ws_alias_domain_is_intersected_without_replacement(
    alias: int,
) -> None:
    spec = _spec(64)
    fields = spec._flat_fields_with_flash_family("ws_overlap")
    key = flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY
    fragment = fields[key]
    assert isinstance(fragment, EnumFragment)
    fields[key] = dataclasses.replace(
        fragment, search_choices=(alias,), coverage_choices=(alias,)
    )
    generation = ConfigGeneration(
        spec,
        overrides={flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"},
        _field_view=fields,
        _flash_pipeline_family_override="ws_overlap",
    )
    assert generation._flat_fields()[key]._active_choices() == (alias,)
    assert generation._flat_fields()[key].coverage_choices == (alias,)
    assert generation.canonicalize_flat(generation.default_flat())[1].config[key] == 0
    assert fields[key]._active_choices() == (alias,)


@pytest.mark.parametrize("topology", ("fa4", "ws_overlap"))
def test_wide_nonstateful_fixed_request_is_legal(topology: str) -> None:
    # An explicit group is honoured on every lowering of the fa4 body; the WS
    # leaf folds it to its inactive 0.
    spec = _spec(64)
    generation = spec.create_config_generation(
        overrides={
            flash.FLASH_PIPELINE_FAMILY_KEY: topology,
            flash.FLASH_SOFTMAX_LOWERING_KEY: "auto",
            flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: 8,
        }
    )
    _flat, config = generation.canonicalize_flat(generation.default_flat())
    assert config.config[flash.FLASH_SOFTMAX_LOWERING_KEY] == "auto"
    assert config.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] == (
        8 if topology == "fa4" else 0
    )
    fields = spec._flat_fields()
    key = flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY
    fragment = fields[key]
    assert isinstance(fragment, EnumFragment)
    fields[key] = dataclasses.replace(
        fragment, search_choices=(8,), coverage_choices=(8,)
    )
    narrowed = ConfigGeneration(
        spec,
        overrides={flash.FLASH_SOFTMAX_LOWERING_KEY: "auto"},
        _field_view=fields,
        _flash_pipeline_family_override=topology,
    )
    assert narrowed._flat_fields()[key]._active_choices() == (8,)
    assert fields[key]._active_choices() == (8,)


def test_full_qualification_terminal_catalog_retains_ws_zero() -> None:
    generation = ConfigGeneration(_spec(64))
    surface = generation.flash_terminal_coordinate_surface_catalog()
    leaves = generation.flash_structural_leaf_catalog()
    assert len(surface["leaves"]) == len(leaves)
    ws = [item for item in surface["leaves"] if item["leaf"]["family"] == "ws_overlap"]
    assert {item["leaf"]["softmax_disc"] for item in ws} == {True}
    for item in ws:
        width = next(
            coordinate
            for coordinate in item["coordinates"]
            if coordinate["key"] == flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY
        )
        assert width["active_values"] == [0]
    assert not generation.flash_structural_coverage_underqualified_leaves()


@pytest.mark.parametrize(
    ("selectors", "expected"),
    (
        ({}, "fa4"),
        ({flash.FLASH_TOPOLOGY_KEY: "ws_overlap"}, "ws_overlap"),
        ({flash.FLASH_TOPOLOGY_KEY: "fa4"}, "fa4"),
        ({flash.FLASH_TOPOLOGY_KEY: "unknown"}, "ws_overlap"),
        (
            {
                flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
                flash.FLASH_TOPOLOGY_KEY: "ws_overlap",
            },
            "fa4",
        ),
    ),
)
def test_topology_lookup_uses_structural_selectors_before_children(
    selectors: dict[str, object], expected: str
) -> None:
    spec = _spec(64)
    controls = {
        **selectors,
        flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_value_graph",
        flash.FLASH_ROW_SUM_SCHEDULE_KEY: "pre_acquire",
    }
    # These children are incompatible on a causal workload. Topology lookup
    # must leave their established ordered validation to its callers.
    assert spec._cute_flash_config_topology(controls) == expected
    assert spec._resolve_cute_flash_config(selectors).topology == expected


@pytest.mark.parametrize("reason", ("forced_ws", "odd_kv"))
def test_topology_lookup_respects_workload_forced_ws(reason: str) -> None:
    spec = _spec(64)
    if reason == "forced_ws":
        spec._cute_flash_requires_ws_overlap = True
    else:
        spec._cute_flash_num_kv = 3
    request = {flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
    assert spec._cute_flash_config_topology(request) == "ws_overlap"
    assert spec._resolve_cute_flash_config(request).topology == "ws_overlap"


def test_family_view_alone_retains_ws_conditional_domain() -> None:
    spec = _spec(64)
    generation = ConfigGeneration(spec, _flash_pipeline_family_override="ws_overlap")
    assert generation._override_values == {}
    assert generation._flat_fields()[
        flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY
    ]._active_choices() == (0,)
    config = generation.canonicalize_flat(generation.default_flat())[1]
    assert config.config[flash.FLASH_PIPELINE_FAMILY_KEY] == "ws_overlap"
    assert config.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] == 0


@pytest.mark.parametrize(
    "lowering", ("auto", "standard", "resident_value_graph", "resident_stateful")
)
def test_no_lowering_owns_the_launch_group(lowering: str) -> None:
    owned = flash._flash_resident_softmax_overrides(lowering)
    assert flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY not in owned
    assert owned.get(flash.FLASH_ROW_SUM_SCHEDULE_KEY) == (
        None if lowering == "resident_stateful" else "post_acquire"
    )
