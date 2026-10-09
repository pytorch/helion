from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
from .test_cute_flash_resident_choice import _spec as _dense_spec
from .test_cute_flash_stateful_choice import _spec as _causal_spec
import helion
from helion._compiler.cute.cute_flash import _flash_resident_softmax_overrides
from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.search_space_logger import canonical_config_id
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch


@pytest.fixture(autouse=True)
def _no_native_compile():
    with _forbid_native_compile():
        yield


@pytest.fixture
def protected_cpu():
    # Keep local coordinate legality independent of the host GPU visibility.
    with _mock_cuda_unavailable():
        yield


def _surface(*, causal=True, dtype=torch.float16, mode="resident_stateful"):
    spec = _causal_spec(64, 32, dtype) if causal else _dense_spec(128, 48, dtype)
    generation = spec.create_config_generation(
        overrides={
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_softmax_disc": False,
            "cute_flash_kv_stage": 2,
        }
    )
    base, parent = generation.canonicalize_flat(
        generation.flatten(helion.Config(cute_flash_softmax_lowering=mode))
    )
    return spec, generation, base, parent


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("causal", (False, True))
def test_owned_coordinates_follow_implementation_and_keep_lowering_switch(
    causal, dtype, protected_cpu
):
    mode = "resident_stateful" if causal else "resident_value_graph"
    _specification, generation, base, parent = _surface(
        causal=causal, dtype=dtype, mode=mode
    )
    owned = generation.flash_owned_coordinate_indices(parent)
    identities = generation._flat_coordinate_identities()
    required = _flash_resident_softmax_overrides(mode, is_causal=causal, topology="fa4")
    assert owned == [
        index
        for index, (key, _sequence_index) in enumerate(identities)
        if key in required and index not in generation.overridden_flat_indices
    ]
    full = generation.coordinate_neighbor_projections(base, radius=2)
    active = generation.coordinate_neighbor_projections(
        base, radius=2, frozen_indices=owned
    )
    assert active == [item for item in full if item.flat_index not in owned]
    assert all(item.config == parent for item in full if item.flat_index in owned)
    assert (
        generation.coordinate_neighbor_projections(
            base, radius=2, limit=12, frozen_indices=owned
        )
        == active[:12]
    )
    switches = [item for item in active if item.key == "cute_flash_softmax_lowering"]
    assert {item.config.config["cute_flash_softmax_lowering"] for item in switches} >= {
        "auto",
        "standard",
    }
    for switch in switches:
        if switch.config.config["cute_flash_softmax_lowering"] in ("auto", "standard"):
            assert [
                identities[index][0]
                for index in generation.flash_owned_coordinate_indices(switch.config)
            ] == ["cute_flash_causal_lpt_swizzle", "cute_flash_row_sum_schedule"]


@pytest.mark.parametrize("mode", ("auto", "standard"))
def test_ordinary_modes_retain_original_coordinates_and_skip_inactive_rowsum(
    mode, protected_cpu
):
    _specification, generation, base, parent = _surface(mode=mode)
    owned = generation.flash_owned_coordinate_indices(parent)
    identities = generation._flat_coordinate_identities()
    assert [identities[index][0] for index in owned] == [
        "cute_flash_causal_lpt_swizzle",
        "cute_flash_row_sum_schedule",
    ]
    original = generation.coordinate_neighbor_projections(base, radius=2)
    assert (
        generation.coordinate_neighbor_projections(
            base, radius=2, limit=12, frozen_indices=owned
        )
        == [item for item in original if item.flat_index not in owned][:12]
    )
    assert all(item.config == parent for item in original if item.flat_index in owned)


def _resident_attempt():
    spec, generation, base, parent = _surface()
    search = LFBOPatternSearch.__new__(LFBOPatternSearch)
    search.config_spec = spec
    search.config_gen = spec.create_config_generation()
    search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
    search.radius = 2
    search.num_neighbors_cap = -1
    search._surrogate_select = lambda candidates, count: candidates[:count]
    flat, global_parent = search.config_gen.canonicalize_flat(
        search.config_gen.flatten(parent)
    )
    member = PopulationMember(Mock(), [1.0], flat, global_parent, status="ok")
    leaf = flash_structural_leaf_from_config(parent.config)
    constraints = (("cute_flash_kv_stage", 2),)
    selected, children, ledger = search._flash_conditional_parent_candidates(
        [member], leaf, constraints, {global_parent}, 12
    )
    return generation, base, global_parent, children, ledger


def test_resident_owned_prefix_yields_a_child_without_extra_raw_proposals(
    protected_cpu,
):
    generation, base, parent, children, ledger = _resident_attempt()
    old = generation.coordinate_neighbor_projections(base, radius=2, limit=12)
    assert len(old) == 12 and all(item.outcome == "incumbent_alias" for item in old)
    assert len(children) == 1 and children[0].config != parent
    assert ledger["schema_version"] == 2
    assert ledger["consumed"] == 12
    assert ledger["allocations"] == [
        {"parent_config_id": canonical_config_id(parent), "limit": 12}
    ]
    assert len(ledger["attempts"][0]["proposals"]) == 12
    assert ledger["attempts"][0]["owned_coordinate_indices"]
