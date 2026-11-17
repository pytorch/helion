"""Search the prerequisites of a proved capture policy as one transition."""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import replace
import functools
from types import SimpleNamespace

import numpy as np
import pytest

import helion
from helion._compiler.cute.autotune.search_surface import (
    _normalize_cute_pdl_capture_overlap,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion.autotuner import LFBOPatternSearch
from helion.autotuner import LFBOTreeSearch
from helion.autotuner import PatternSearch
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.search_surface import ConfigGenerationExtension

pytestmark = pytest.mark.cpu_only
PDL = "cute_pdl"
CAPTURE = "cute_pdl_capture_overlap"
SPREAD = "cute_pdl_spread"
RELEASE = "cute_pdl_release"


@functools.cache
def _bound():
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(
            paged_online_attention,
            (*paged_attention_args(rows=8), None, True),
        )


@pytest.fixture(autouse=True)
def _cpu_only():
    with cpu_render_target(fake_device_sm_count=148):
        yield


@contextmanager
def _compilation():
    bound = _bound()
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        yield bound


@pytest.fixture
def compiler_scope():
    with _compilation():
        yield


def _surface(*, overrides=None, family="attention_paired", capture_search=True):
    bound = _bound()
    fields = None
    if capture_search:
        # These neighbor tests model a custom search whose caller explicitly
        # accepts immutable captured graphs. Production search leaves capture
        # overlap off and release placement at auto.
        fields = dict(bound.config_spec._flat_fields())
        fields[CAPTURE] = replace(fields[CAPTURE], search_choices=(False, True))
        fields[RELEASE] = replace(
            fields[RELEASE], search_choices=("auto", "loader_tail")
        )
    generation = bound.config_spec.create_config_generation(overrides=overrides)
    if fields is not None:
        generation = ConfigGeneration(
            bound.config_spec,
            overrides=generation._override_values,
            _field_view=fields,
        )
    parent = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps=family,
        cute_min_blocks_per_mp=1,
        cute_mma_dag_v_lag=3 if family == "attention_paired" else 0,
        cute_pdl=False,
        cute_pdl_spread=True,
    )
    base, config = generation.canonicalize_flat(generation.flatten(parent))
    slots = {key: entry[0][0] for key, entry in generation._key_to_flat_indices.items()}
    return generation, base, config, slots


def _assert_capture(config, *, tail=False):
    assert config.get(PDL) is True
    assert config.get(CAPTURE) is True
    assert config.get(SPREAD) is False
    assert config.get(RELEASE, "auto") == ("loader_tail" if tail else "auto")
    assert config.get("cute_mma_dag_pass_warps") == "attention_paired"
    assert config.get("cute_mma_dag_v_lag") == 3


def test_default_search_and_compiler_seeds_do_not_enable_capture(compiler_scope):
    generation, base, _parent, slots = _surface(capture_search=False)
    assert generation.flat_spec[slots[PDL]]._active_choices() == (False, True)
    assert generation.flat_spec[slots[CAPTURE]]._active_choices() == (False,)
    assert generation.flat_spec[slots[RELEASE]]._active_choices() == ("auto",)
    moves = generation.coordinate_neighbor_projections(base)
    assert any(move.key == PDL and move.to_value is True for move in moves)
    assert not any(move.key == CAPTURE and move.to_value is True for move in moves)
    assert not any(
        move.key == RELEASE and move.to_value == "loader_tail" for move in moves
    )
    seeds = _bound().config_spec.compiler_seed_configs
    assert seeds
    for seed in seeds:
        assert not seed.get(CAPTURE, False)
        assert seed.get(RELEASE, "auto") != "loader_tail"


def test_explicit_capture_survives_default_search_domain_and_renders():
    with _compilation():
        generation, _base, parent, _slots = _surface(capture_search=False)
        selected = {PDL: True, CAPTURE: True, SPREAD: False, RELEASE: "loader_tail"}
        explicit = helion.Config.from_dict(dict(parent) | selected)
        ((flat, config),) = generation.user_seed_flat_config_pairs([explicit])
        assert flat in generation.random_population_flat(
            2, user_seed_configs=[explicit]
        )
        _assert_capture(config, tail=True)
        assert generation.strict_unflatten(flat) == config

        overridden, _base, _parent, _slots = _surface(
            overrides=selected, capture_search=False
        )
        override_flat, override_config = overridden.strict_config_pair(parent)
        assert all(override_config.get(key) == value for key, value in selected.items())
        assert overridden.strict_unflatten(override_flat) == override_config
        assert override_config == config
    # Both actual transfer routes select the same config; render it only once.
    source = _bound().to_code(config)
    assert "capture_pdl_wait_mode" in source
    assert "capture_release_tail=True" in source


@pytest.mark.parametrize(
    "requested",
    [
        {PDL: False, CAPTURE: True},
        {PDL: True, CAPTURE: True, SPREAD: True},
        {PDL: True, CAPTURE: True, RELEASE: "exit"},
        {PDL: True, CAPTURE: False, RELEASE: "loader_tail"},
        {
            PDL: True,
            CAPTURE: True,
            "cute_mma_dag_pass_warps": "attention",
        },
    ],
)
def test_explicit_incompatible_pdl_request_is_rejected(compiler_scope, requested):
    _generation, _base, parent, _slots = _surface(
        family=requested.get("cute_mma_dag_pass_warps", "attention_paired")
    )
    config = helion.Config.from_dict(dict(parent) | {SPREAD: False} | requested)
    # Spreading can fail its schedule precondition before capture validation.
    message = (
        "PDL shared-memory padding" if requested.get(SPREAD) else "capture|loader_tail"
    )
    with pytest.raises(helion.exc.InvalidConfig, match=message):
        _bound()._normalized_config_copy(config)
    assert all(config.get(key) == value for key, value in requested.items())


@pytest.mark.parametrize("fix_invalid", [False, True])
def test_capture_normalizer_requires_spread_off(compiler_scope, fix_invalid):
    _generation, _base, parent, _slots = _surface()
    config = dict(parent) | {PDL: True, CAPTURE: True, SPREAD: True}
    if fix_invalid:
        _normalize_cute_pdl_capture_overlap(
            _bound().config_spec, config, fix_invalid=True
        )
        assert not config.get(CAPTURE, False)
        assert config[SPREAD] is True
    else:
        with pytest.raises(helion.exc.InvalidConfig, match="spread"):
            _normalize_cute_pdl_capture_overlap(
                _bound().config_spec, config, fix_invalid=False
            )


def test_real_paired_neighbors_activate_pdl_capture_and_tail_without_rewriting_parent(
    compiler_scope,
):
    generation, base, parent, slots = _surface()
    before = copy.deepcopy(base)
    parent_hash = hash(parent)
    moves = generation.coordinate_neighbor_projections(base)
    capture = next(
        move for move in moves if move.key == CAPTURE and move.to_value is True
    )
    assert capture.outcome == "candidate"
    _assert_capture(capture.config)
    assert capture.requested_flat_values[slots[CAPTURE]] is True
    assert capture.requested_flat_values[slots[PDL]] is True
    assert capture.requested_flat_values[slots[SPREAD]] is False
    pdl = next(move for move in moves if move.key == PDL and move.to_value is True)
    assert pdl.outcome == "candidate"
    assert pdl.config.get(PDL) is True and pdl.config.get(SPREAD) is False
    assert not pdl.config.get(CAPTURE, False)
    tail = next(
        move for move in moves if move.key == RELEASE and move.to_value == "loader_tail"
    )
    _assert_capture(tail.config, tail=True)
    assert base == before
    assert hash(parent) == parent_hash
    assert parent.get(PDL) is False and parent.get(SPREAD) is True
    # Explicit invalid/inert configs retain their old semantics and identities.
    raw = [*base]
    raw[slots[CAPTURE]] = True
    assert not generation.unflatten(raw).get(CAPTURE, False)
    round_trip, same = generation.canonicalize_flat(generation.flatten(parent))
    assert round_trip == before and same == parent


@pytest.mark.parametrize("pin", [PDL, SPREAD, RELEASE])
def test_frozen_prerequisite_refuses_capture_transition(compiler_scope, pin):
    generation, base, _parent, slots = _surface()
    if pin == RELEASE:
        base[slots[RELEASE]] = "exit"
    original = [*base]
    raw = [*base]
    raw[slots[CAPTURE]] = True
    assert (
        generation.conditional_neighbor(base, raw, frozen_indices=[slots[pin]]) == raw
    )
    assert base == original
    moves = generation.coordinate_neighbor_projections(
        base, frozen_indices=[slots[pin]]
    )
    capture = next(
        move for move in moves if move.key == CAPTURE and move.to_value is True
    )
    assert not capture.config.get(CAPTURE, False)


@pytest.mark.parametrize("overrides", [{PDL: False}, {SPREAD: True}, {RELEASE: "exit"}])
def test_explicit_override_is_never_changed_by_capture(compiler_scope, overrides):
    generation, base, _parent, slots = _surface(overrides=overrides)
    raw = [*base]
    raw[slots[CAPTURE]] = True
    assert generation.conditional_neighbor(base, raw) == raw


def test_restricted_search_choice_is_not_reintroduced(compiler_scope):
    generation, base, _parent, slots = _surface()
    generation.flat_spec[slots[PDL]] = EnumFragment(
        (False, True), search_choices=(False,)
    )
    raw = [*base]
    raw[slots[CAPTURE]] = True
    assert generation.conditional_neighbor(base, raw) == raw


@pytest.mark.parametrize("refusal", ["family", "unproved", "explicit"])
def test_other_family_and_missing_proofs_keep_raw_neighbors(
    compiler_scope, refusal, monkeypatch
):
    generation, base, _parent, slots = _surface(
        family="attention" if refusal == "family" else "attention_paired"
    )
    if refusal == "unproved":
        monkeypatch.setattr(
            generation.config_spec, "cute_pdl_capture_overlap_available", False
        )
    elif refusal == "explicit":
        monkeypatch.setattr(generation.config_spec, "has_explicit_pdl", True)
    raw = [*base]
    raw[slots[CAPTURE]] = True
    assert generation.conditional_neighbor(base, raw) == raw


def test_backend_without_conditional_policies_keeps_identity(compiler_scope):
    generation, base, _parent, slots = _surface()
    generation.extension = ConfigGenerationExtension(generation)
    raw = [*base]
    raw[slots[CAPTURE]] = True
    assert generation.conditional_neighbor(base, raw) is raw


def test_pattern_and_random_neighbors_use_the_same_prerequisites(
    compiler_scope, monkeypatch
):
    generation, base, _parent, slots = _surface()
    search = PatternSearch.__new__(PatternSearch)
    search.config_gen = generation
    search.num_neighbors_cap = -1
    neighbors = search._generate_neighbors(base)
    captures = [
        generation.unflatten(row) for row in neighbors if row[slots[CAPTURE]] is True
    ]
    assert captures
    _assert_capture(captures[0])

    random_search = LFBOPatternSearch.__new__(LFBOPatternSearch)
    random_search.config_gen = generation
    random_search.num_neighbors = 1
    random_search.num_neighbors_cap = -1
    random_search.radius = 1
    monkeypatch.setattr(
        "helion.autotuner.surrogate_pattern_search.random.randint", lambda low, high: 1
    )
    monkeypatch.setattr(
        "helion.autotuner.surrogate_pattern_search.random.sample",
        lambda rows, count: [next(row for row in rows if row[0] == slots[CAPTURE])],
    )
    fixed = {index: base[index] for index in generation.block_size_indices}
    if generation.num_warps_index >= 0:
        fixed[generation.num_warps_index] = base[generation.num_warps_index]
    proposals = []
    (neighbor,) = random_search._generate_neighbors(
        base, fixed_flat_values=fixed, proposal_callback=proposals.append
    )
    _assert_capture(generation.unflatten(neighbor))
    assert proposals == [neighbor]
    # Conditional leaf pins must still win over the prerequisite repair.
    (refused,) = random_search._generate_neighbors(
        base, fixed_flat_values={**fixed, slots[PDL]: False}
    )
    assert refused[slots[PDL]] is False


def test_tree_scores_complete_conditional_configs_and_keeps_frozen_prerequisites(
    compiler_scope,
):
    generation, base, _parent, slots = _surface()
    search = LFBOTreeSearch.__new__(LFBOTreeSearch)
    search.config_gen = generation
    search.num_neighbors = 1
    search.num_neighbors_cap = -1
    search.radius = 1
    search._autotune_metrics = SimpleNamespace(num_generations=2)
    search._encoded_to_flat_mapping = None
    mapping = search._get_encoded_to_flat_mapping()
    capture_feature = mapping[slots[CAPTURE]][0] + 1
    pdl_feature = mapping[slots[PDL]][0] + 1
    spread_feature = mapping[slots[SPREAD]][0] + 1
    observations = []

    class Estimator:
        tree_ = SimpleNamespace(feature=np.array([capture_feature], dtype=int))

        def decision_path(self, values):
            return SimpleNamespace(indices=np.array([0], dtype=int))

        def predict_proba(self, values):
            observations.append(values.copy())
            scores = values[:, capture_feature]
            return np.column_stack((1 - scores, scores))

    search.surrogate = SimpleNamespace(estimators_=[Estimator()])
    fixed = {index: base[index] for index in generation.block_size_indices}
    if generation.num_warps_index >= 0:
        fixed[generation.num_warps_index] = base[generation.num_warps_index]
    (neighbor,) = search._generate_neighbors(base, fixed_flat_values=fixed)
    _assert_capture(generation.unflatten(neighbor))
    scored = observations[0][1]
    assert scored[capture_feature] == scored[pdl_feature] == 1
    assert scored[spread_feature] == 0
    # Refused coupling stays an ordinary raw request and normalizes inactive.
    (refused,) = search._generate_neighbors(
        base, fixed_flat_values={**fixed, slots[PDL]: False}
    )
    assert not generation.unflatten(refused).get(CAPTURE, False)


def test_capture_disable_and_pinned_child_keep_independent_pdl(compiler_scope):
    generation, base, _parent, slots = _surface()
    raw = [*base]
    raw[slots[CAPTURE]] = True
    enabled, enabled_config = generation.canonicalize_flat(
        generation.conditional_neighbor(base, raw)
    )
    _assert_capture(enabled_config)
    disabled = [*enabled]
    disabled[slots[CAPTURE]] = False
    assert generation.conditional_neighbor(enabled, disabled) == disabled
    normal = generation.unflatten(disabled)
    assert normal.get(PDL) is True and not normal.get(CAPTURE, False)
    tail = [*base]
    tail[slots[RELEASE]] = "loader_tail"
    assert (
        generation.conditional_neighbor(base, tail, frozen_indices=[slots[CAPTURE]])
        == tail
    )


def test_generated_neighbor_renders_capture_contract():
    with _compilation() as bound:
        generation, base, _parent, slots = _surface()
        raw = [*base]
        raw[slots[CAPTURE]] = True
        capture = generation.unflatten(generation.conditional_neighbor(base, raw))
    source = bound.to_code(capture)
    assert "paired-output-stores-v1" in source
    assert "capture_pdl_wait_mode" in source
