from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import helion
from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config
from helion.autotuner import LFBOPatternSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.search_space_logger import canonical_config_id
from helion.exc import InvalidConfig


class _Fragment:
    def __init__(self, values):
        self.values = values

    def pattern_neighbors(self, current, radius=1):
        return [value for value in self.values if value != current]


class _ConditionalGeneration:
    """Small conditional surface with inherited, fixed parent schedule values."""

    coordinate_neighbor_projections = ConfigGeneration.coordinate_neighbor_projections
    flash_owned_coordinate_indices = ConfigGeneration.flash_owned_coordinate_indices
    keys = ("cute_flash_e2e_schedule", "cute_flash_packed_reduce", "cute_flash_s_stage")
    block_size_indices = ()
    num_warps_index = -1
    _advanced_controls_files = ()
    process_group_name = None

    def __init__(self, values=(True, False)):
        self.overridden_flat_indices = {2}
        self._override_values = {}
        self.flat_spec = [_Fragment(()), _Fragment(values), _Fragment((2,))]
        self.config_spec = SimpleNamespace(
            cute_flash_search_enabled=False,
            backend=SimpleNamespace(
                autotune_config_is_viable=lambda spec, config: True
            ),
            create_config_generation=lambda **kwargs: self,
        )

    def _flat_coordinate_identities(self):
        return [(key, None) for key in self.keys]

    def flatten(self, config):
        return [config.config[key] for key in self.keys]

    def canonicalize_flat(self, raw):
        if raw[1] == "invalid":
            raise InvalidConfig("invalid test projection")
        normalized = [*raw]
        if raw[1] == "alias":
            normalized[1] = True
        return normalized, helion.Config(
            cute_flash_pipeline_family="ws_overlap",
            cute_flash_exp2_packet="1x1",
            cute_flash_softmax_disc=True,
            **dict(zip(self.keys, normalized, strict=True)),
        )


def _fixture(values=(True, False)):
    generation = _ConditionalGeneration(values)

    def member(schedule, packed, perf):
        flat, config = generation.canonicalize_flat([schedule, packed, 2])
        return PopulationMember(Mock(), [perf], flat, config, status="ok")

    top = member("16/8", True, 1.0)
    alternate = member("16/6", True, 2.0)
    measured_child = member("16/8", False, 3.0)
    members = [measured_child, alternate, top]  # The producer must rank them.
    search = LFBOPatternSearch.__new__(LFBOPatternSearch)
    search.config_spec = generation.config_spec
    search.config_gen = generation
    search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
    search.radius = 2
    search.num_neighbors_cap = -1
    search._flash_leaf_config_generation = Mock(return_value=generation)
    search._surrogate_select = Mock(
        side_effect=lambda candidates, count: candidates[:count]
    )
    leaf = flash_structural_leaf_from_config(top.config.config)
    assert leaf is not None
    constraints = (*search._flash_leaf_constraints(leaf), ("cute_flash_s_stage", 2))
    return search, members, leaf, constraints


def _run(*, limit=6, values=(True, False), all_known=False):
    search, members, leaf, constraints = _fixture(values)
    known = {member.config for member in members}
    if all_known:
        known.add(search.config_gen.canonicalize_flat(["16/6", False, 2])[1])
    rng = random.getstate()
    try:
        random.seed(732)
        parent, children, ledger = search._flash_conditional_parent_candidates(
            members, leaf, constraints, known, limit
        )
    finally:
        random.setstate(rng)
    return search, parent, children, ledger


def test_saturated_best_parent_falls_back_with_one_shared_budget():
    search, parent, children, ledger = _run()
    assert parent is not None and parent.config["cute_flash_e2e_schedule"] == "16/6"
    assert len(children) == 1
    assert children[0].config["cute_flash_packed_reduce"] is False
    assert [attempt["consumed"] for attempt in ledger["attempts"]] == [2, 2]
    assert ledger["consumed"] == 4
    assert sum(item["limit"] for item in ledger["allocations"]) == 6
    assert ledger["attempts"][0]["novel_config_ids"] == []
    assert ledger["attempts"][1]["novel_config_ids"] == [
        canonical_config_id(child.config) for child in children
    ]
    search._surrogate_select.assert_called_once()


@pytest.mark.parametrize("limit", [0, 1, 2, 6])
def test_bounded_empty_prefix_is_not_claimed_exhausted(limit):
    _search, parent, children, ledger = _run(limit=limit, all_known=True)
    assert parent is None and children == []
    assert ledger["consumed"] == limit
    assert len(ledger["attempts"]) == min(3, limit)
    assert all(item["limit"] > 0 for item in ledger["allocations"])
    assert "space_exhausted" not in ledger


def test_raw_invalid_alias_and_unchanged_draws_count_against_budget():
    _search, parent, children, ledger = _run(
        limit=12, values=(True, False, "invalid", "alias"), all_known=True
    )
    assert parent is None and not children
    assert ledger["consumed"] == 12
    assert [item["outcome"] for item in ledger["attempts"][0]["proposals"][:3]] == [
        "known",
        "invalid",
        "known",
    ]
    assert all(len(item["proposals"]) == 4 for item in ledger["attempts"])


def test_coordinate_limit_counts_invalid_and_normalized_alias_requests():
    generation = _ConditionalGeneration(("invalid", "alias", False))
    base = ["16/8", True, 2]
    full = generation.coordinate_neighbor_projections(base, radius=2)
    assert [item.outcome for item in full] == [
        "invalid",
        "incumbent_alias",
        "candidate",
    ]
    assert (
        generation.coordinate_neighbor_projections(base, radius=2, limit=2) == full[:2]
    )
    assert generation.coordinate_neighbor_projections(base, radius=2, limit=0) == []


@pytest.mark.parametrize("tree", [False, True])
def test_optional_proposal_recording_preserves_default_random_draws(tree):
    from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch

    prototype, members, _leaf, _constraints = _fixture()
    search_type = LFBOTreeSearch if tree else LFBOPatternSearch
    search = search_type.__new__(search_type)
    search.__dict__.update(prototype.__dict__)
    search.num_neighbors = 19
    search.surrogate = None
    search._autotune_metrics = SimpleNamespace(num_generations=2)
    base = members[0].flat_values
    saved = random.getstate()
    try:
        random.seed(872)
        plain = search._generate_neighbors(base)
        plain_state = random.getstate()
        random.seed(872)
        recorded = []
        observed = search._generate_neighbors(base, proposal_callback=recorded.append)
        observed_state = random.getstate()
    finally:
        random.setstate(saved)
    assert plain == observed
    assert plain_state == observed_state
    assert len(recorded) == 19
    assert observed == [flat for flat in recorded if flat != base]
