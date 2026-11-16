"""Search the prerequisites of a proved capture policy as one transition.

The policy is searched only where the benchmark times back-to-back captured
launches (rotating input banks); every other search keeps the paired
schedule's default policy in the configs it tries.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import functools
import json
import random
from types import SimpleNamespace
from unittest.mock import Mock

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
from helion.autotuner import LLMGuidedSearch
from helion.autotuner import PatternSearch
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.llm.configs import parse_response_configs
from helion.autotuner.search_surface import ConfigGenerationExtension

pytestmark = pytest.mark.cpu_only
PDL = "cute_pdl"
CAPTURE = "cute_pdl_capture_overlap"
SPREAD = "cute_pdl_spread"
RELEASE = "cute_pdl_release"


def _bound(capture_search=True):
    return _bind(capture_search)


@functools.cache
def _bind(capture_search):
    kernel = paged_online_attention
    if capture_search:
        # Rotating input banks time the kernel's calls back to back inside one
        # captured graph, the benchmark that observes the capture policy.
        kernel = helion.kernel(
            paged_online_attention.fn,
            backend="cute",
            static_shapes=True,
            autotune_benchmark_input_banks=2,
        )
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(kernel, (*paged_attention_args(rows=8), None, True))


@pytest.fixture(autouse=True)
def _cpu_only():
    with cpu_render_target(fake_device_sm_count=148):
        yield


@contextmanager
def _compilation(capture_search=True):
    bound = _bound(capture_search)
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        yield bound


@pytest.fixture
def compiler_scope():
    with _compilation():
        yield


def _surface(*, overrides=None, family="attention_paired", capture_search=True):
    bound = _bound(capture_search)
    assert bound.config_spec.cute_pdl_capture_overlap_searched is capture_search
    generation = bound.config_spec.create_config_generation(overrides=overrides)
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


def _policy(config):
    return (
        config.get(PDL, False),
        config.get(CAPTURE, False),
        config.get(RELEASE, "auto"),
        config.get(SPREAD, False),
    )


def _paired(config):
    return config.get("cute_mma_dag_pass_warps") == "attention_paired"


def _family(config):
    """A config without its dependent-launch policy."""
    return {
        key: value
        for key, value in config.items()
        if key not in (PDL, CAPTURE, RELEASE, SPREAD)
    }


def test_search_without_bank_cycles_keeps_the_default_policy():
    default = (True, True, "loader_tail", False)
    with _compilation(False):
        generation, base, parent, slots = _surface(capture_search=False)
        assert generation.flat_spec[slots[PDL]]._active_choices() == (False, True)
        assert generation.flat_spec[slots[CAPTURE]]._active_choices() == (False,)
        assert generation.flat_spec[slots[RELEASE]]._active_choices() == ("auto",)
        # The transferred config named ``cute_pdl=False`` and spread padding:
        # eager timing cannot rank either against the default, so the
        # searched config carries the default policy.
        assert _policy(parent) == default
        assert tuple(base[slots[key]] for key in (PDL, CAPTURE, RELEASE, SPREAD)) == (
            default
        )
        # No move leaves it: a policy coordinate normalizes back to the
        # incumbent, every other paired neighbor keeps the policy.
        moves = generation.coordinate_neighbor_projections(base)
        for move in moves:
            if move.key in (PDL, CAPTURE, RELEASE, SPREAD):
                assert move.outcome == "incumbent_alias"
            elif (
                move.config is not None
                and move.config.get("cute_mma_dag_pass_warps") == "attention_paired"
            ):
                assert _policy(move.config) == default
        assert any(move.key == "cute_mma_dag_v_lag" for move in moves)
        # Random draws and mutations of the paired schedule keep it too.
        drawn = generation.random_flat()
        drawn[slots["cute_mma_dag"]] = "tcgen05"
        drawn[slots["cute_mma_dag_pass_warps"]] = "attention_paired"
        for key in (PDL, CAPTURE, SPREAD):
            drawn[slots[key]] = not base[slots[key]]
        generation.extension.repair_flat_config(drawn)
        assert tuple(drawn[slots[key]] for key in (PDL, CAPTURE, RELEASE, SPREAD)) == (
            default
        )
        # The other schedules have no capture contract: their dependent
        # launch is an ordinary searched knob.
        _generation, _base, unpaired, _slots = _surface(
            family="attention", capture_search=False
        )
        assert _policy(unpaired) == (False, False, "auto", True)
        # A fixed override that opts out declines the policy.
        _generation, _base, pinned, _slots = _surface(
            overrides={PDL: False}, capture_search=False
        )
        assert _policy(pinned) == (False, False, "auto", True)
        seeds = _bound(False).config_spec.compiler_seed_configs
        assert seeds
        assert any(
            seed.get("cute_mma_dag_pass_warps") == "attention_paired" for seed in seeds
        )
        for seed in seeds:
            paired = seed.get("cute_mma_dag_pass_warps") == "attention_paired"
            assert seed.get(PDL, False) is paired
            assert seed.get(CAPTURE, False) is paired
            assert seed.get(RELEASE, "auto") == ("loader_tail" if paired else "auto")


def test_bank_cycle_search_samples_and_seeds_the_policy():
    eager_seeds = _bound(False).config_spec.compiler_seed_configs
    with _compilation():
        generation, _base, parent, slots = _surface()
        assert generation.flat_spec[slots[PDL]]._active_choices() == (False, True)
        assert generation.flat_spec[slots[CAPTURE]]._active_choices() == (False, True)
        assert generation.flat_spec[slots[RELEASE]]._active_choices() == (
            "auto",
            "loader_tail",
        )
        # A transferred config keeps the policy it names (the pad is inert).
        assert _policy(parent) == (False, False, "auto", True)
        # The leading paired family comes with each policy on the diagonal:
        # overlap released from the loader's tail (the default), overlap with
        # the automatic release, and no dependent launch.
        bound = _bound()
        seeds = bound.config_spec.compiler_seed_configs
        head = seeds[:3]
        assert [_policy(seed) for seed in head] == [
            (True, True, "loader_tail", False),
            (True, True, "auto", False),
            (False, False, "auto", False),
        ]
        policy_keys = {PDL, CAPTURE, RELEASE}
        assert (
            len(
                {
                    tuple(
                        sorted(
                            (key, repr(value))
                            for key, value in seed.config.items()
                            if key not in policy_keys
                        )
                    )
                    for seed in head
                }
            )
            == 1
        )
        pairs = generation.seed_flat_config_pairs()
        assert [_policy(config) for _flat, config in pairs[:3]] == [
            _policy(seed) for seed in head
        ]
        for _flat, config in pairs[:3]:
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, config
            )
        # Every other paired seed carries the default policy, as without banks.
        assert all(
            _policy(seed) == (True, True, "loader_tail", False)
            for seed in seeds[3:]
            if seed.get("cute_mma_dag_pass_warps") == "attention_paired"
        )
        assert [seed for seed in seeds if seed not in head] == [
            seed for seed in eager_seeds if seed != head[0]
        ]


@pytest.mark.parametrize(
    "seed,tail,entry,pdl",
    [(0, True, False, True), (1, False, True, True), (2, False, False, False)],
)
def test_policy_seeds_render_their_release(seed, tail, entry, pdl):
    bound = _bound()
    config = bound.config_spec.compiler_seed_configs[seed]
    source = bound.to_code(config)
    assert ("._helion_cute_use_pdl = True" in source) is pdl
    assert ("capture_pdl_contract" in source) is pdl
    assert ("capture_release_tail=True" in source) is tail
    assert ("cute.arch.griddepcontrol_launch_dependents()" in source) is entry


@pytest.mark.parametrize(
    "overrides,policy",
    [
        # The two documented opt-outs decline the seeds' guarded policy,
        ({PDL: False}, (False, False, "auto", False)),
        ({CAPTURE: False}, (False, False, "auto", False)),
        # as does every other override that policy cannot keep: a release a
        # capture launch cannot use, spread padding, ordinary dependent
        # launch asked for by name.
        ({RELEASE: "exit"}, (False, False, "exit", False)),
        ({SPREAD: True}, (False, False, "auto", True)),
        ({PDL: True, CAPTURE: False}, (True, False, "auto", False)),
        # A release named beside the overlap's opt-out keeps ordinary
        # dependent launch on it as well.
        ({CAPTURE: False, RELEASE: "auto"}, (True, False, "auto", False)),
        ({CAPTURE: False, RELEASE: "entry"}, (True, False, "entry", False)),
        # Overrides it can keep leave it in place.
        ({RELEASE: "entry"}, (True, True, "entry", False)),
        ({RELEASE: "last_wave"}, (True, True, "last_wave", False)),
        ({PDL: True}, (True, True, "loader_tail", False)),
    ],
)
def test_overrides_never_drop_the_paired_seeds(overrides, policy):
    # Each paired seed spells out the guarded default, whose options are
    # legal only together. An override of one of them must not leave the
    # others behind for strict seed transfer to reject: the paired schedule
    # would vanish from the search. The leading seed is the kernel's
    # no-config default, so it has to stay first.
    with _compilation(False):
        spec = _bound(False).config_spec
        seeded = [
            config
            for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
            if _paired(config)
        ]
        assert seeded and all(
            _policy(config) == (True, True, "loader_tail", False) for config in seeded
        )
        failures: list[str] = []
        generation = spec.create_config_generation(overrides=overrides)
        pairs = generation.seed_flat_config_pairs(failures.append)
        kept = [(flat, config) for flat, config in pairs if _paired(config)]
        assert not failures
        assert len(kept) == len(seeded)
        assert pairs[0] == kept[0]
        for (flat, config), seed in zip(kept, seeded, strict=True):
            assert _policy(config) == policy
            # Only the dependent-launch policy differs from the seed.
            assert _family(config) == _family(seed)
            # The flat row a search starts from decodes to that policy as is.
            assert _policy(generation.strict_unflatten(flat)) == policy


@pytest.mark.parametrize(
    "overrides,policies",
    [
        # The opt-outs fold the three policies of the leading family into
        # the declined one; without the overlap the automatic release is
        # ordinary dependent launch.
        ({PDL: False}, [(False, False, "auto", False)]),
        (
            {CAPTURE: False},
            [(False, False, "auto", False), (True, False, "auto", False)],
        ),
        # A release named beside that opt-out keeps ordinary dependent
        # launch in both seeds that carried the overlap.
        (
            {CAPTURE: False, RELEASE: "auto"},
            [(True, False, "auto", False), (False, False, "auto", False)],
        ),
        # An override the policy keeps leaves the diagonal in place.
        (
            {SPREAD: False},
            [
                (True, True, "loader_tail", False),
                (True, True, "auto", False),
                (False, False, "auto", False),
            ],
        ),
    ],
)
def test_overrides_never_drop_the_bank_cycle_policy_seeds(
    compiler_scope, overrides, policies
):
    # With rotating banks the leading family is seeded on each policy. The
    # same overrides must keep it, and every other paired family, in the
    # search, the leading family first.
    spec = _bound().config_spec
    seeded = [
        config
        for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
        if _paired(config)
    ]
    assert [_policy(config) for config in seeded[:3]] == [
        (True, True, "loader_tail", False),
        (True, True, "auto", False),
        (False, False, "auto", False),
    ]
    failures: list[str] = []
    generation = spec.create_config_generation(overrides=overrides)
    pairs = generation.seed_flat_config_pairs(failures.append)
    kept = [config for _flat, config in pairs if _paired(config)]
    assert not failures
    head = [config for config in kept if _family(config) == _family(seeded[0])]
    assert kept[: len(head)] == head and pairs[0][1] == head[0]
    assert [_policy(config) for config in head] == policies
    families = [_family(config) for config in seeded[2:]]
    assert [_family(config) for config in kept[len(head) - 1 :]] == families
    for flat, config in pairs:
        assert _policy(generation.strict_unflatten(flat)) == _policy(config)


def test_overrides_asking_for_capture_are_never_dropped():
    # Declining is for a policy the candidate carried. Capture overlap forced
    # over the unpaired seeds, which have no ``cute_pdl``, still fails, and so
    # does an explicit config that is inconsistent whatever is overridden.
    with _compilation(False):
        spec = _bound(False).config_spec
        failures: list[str] = []
        generation = spec.create_config_generation(overrides={CAPTURE: True})
        pairs = generation.seed_flat_config_pairs(failures.append)
        assert pairs and all(_paired(config) for _flat, config in pairs)
        assert all(_policy(config)[:2] == (True, True) for _flat, config in pairs)
        assert failures and all("cute_pdl_capture_overlap=True" in f for f in failures)
        _generation, _base, parent, _slots = _surface(capture_search=False)
        inconsistent = helion.Config.from_dict(
            dict(parent) | {PDL: True, CAPTURE: True, SPREAD: False, RELEASE: "exit"}
        )
        lagged = spec.create_config_generation(overrides={"cute_mma_dag_v_lag": 3})
        with pytest.raises(helion.exc.InvalidConfig, match="capture"):
            lagged.strict_config_pair(inconsistent)


@pytest.mark.parametrize(
    "overrides,policy",
    [
        # A fixed override states the option it names and nothing else:
        # while what it states is still a guarded policy, the options it
        # leaves out keep their default,
        ({SPREAD: False}, (True, True, "loader_tail", False)),
        ({PDL: True}, (True, True, "loader_tail", False)),
        ({CAPTURE: True}, (True, True, "loader_tail", False)),
        ({RELEASE: "auto"}, (True, True, "auto", False)),
        ({RELEASE: "entry"}, (True, True, "entry", False)),
        # and so does a search pinned to the paired schedule.
        (
            {"cute_mma_dag_pass_warps": "attention_paired"},
            (True, True, "loader_tail", False),
        ),
    ],
)
def test_search_without_bank_cycles_keeps_what_an_override_leaves_out(
    overrides, policy
):
    with _compilation(False):
        generation, base, parent, slots = _surface(
            overrides=overrides, capture_search=False
        )
        # The transferred config named ``cute_pdl=False`` and spread padding.
        assert _policy(parent) == policy
        assert _policy(generation.strict_unflatten(base)) == policy
        # No move steps off it: eager timing cannot rank the overlap and
        # reads the capture test's host time as a loss.
        moves = [
            move
            for move in generation.coordinate_neighbor_projections(base)
            if move.key in (PDL, CAPTURE, RELEASE, SPREAD)
        ]
        assert {move.key for move in moves} == {PDL, CAPTURE, RELEASE, SPREAD} - set(
            overrides
        )
        assert all(move.outcome == "incumbent_alias" for move in moves)
        # Nor does a mutation of the policy coordinates.
        mutated = [*base]
        for key in (PDL, CAPTURE, SPREAD):
            mutated[slots[key]] = not base[slots[key]]
        assert _policy(generation.unflatten(mutated)) == policy


@pytest.mark.parametrize(
    "overrides,drawn",
    [
        (
            {"cute_mma_dag_pass_warps": "attention_paired"},
            [{"cute_mma_dag_pass_warps": "attention"}],
        ),
        (
            {"cute_mma_dag": "tcgen05", "cute_mma_dag_pass_warps": "attention_paired"},
            [
                {"cute_mma_dag_pass_warps": "attention"},
                {"cute_mma_dag": "warp"},
                {"cute_mma_dag": "warp", "cute_mma_dag_pass_warps": "off"},
                # A draw the attention schedule rejects is decoded as
                # another schedule, with its knobs fixed, before the
                # overrides put it back on the paired one.
                {"cute_mma_dag_stages": 8},
            ],
        ),
    ],
)
def test_search_pinned_to_the_paired_schedule_keeps_the_default_in_every_candidate(
    overrides, drawn
):
    # An override that selects the paired schedule makes a paired candidate
    # of a row whose own schedule slots hold another value. A random draw
    # holds any value there, and a mutation leaves an overridden slot as
    # drawn, so every row descended from the draw does too. The overrides
    # are layered over a row after it is repaired and decoded: the default
    # policy has to reach the candidate they make of it.
    default = (True, True, "loader_tail", False)
    off = {PDL: False, CAPTURE: False, RELEASE: "auto", SPREAD: True}
    names = (PDL, CAPTURE, RELEASE, SPREAD)
    with _compilation(False):
        generation, base, parent, slots = _surface(
            overrides=overrides, capture_search=False
        )
        assert _paired(parent) and _policy(parent) == default
        frozen = sorted(generation.overridden_flat_indices)
        assert frozen == sorted(slots[key] for key in overrides)
        # Each row as a draw leaves it: its own values in the schedule
        # slots and no dependent launch, on the knobs of a paired config.
        rows = []
        for own in drawn:
            row = [*base]
            for key, value in {**off, **own}.items():
                row[slots[key]] = value
            rows.append(row)
        for row in rows:
            flat, config = generation.canonicalize_flat(row)
            assert _paired(config) and _policy(config) == default
            # The canonical row agrees with the config it decodes to.
            assert tuple(flat[slots[key]] for key in names) == default
            # Crossed with the paired incumbent, the row keeps the draw's
            # overridden slots and is still a paired candidate.
            child = generation.differential_mutation(row, base, base, base, 1.0)
            assert [child[i] for i in frozen] == [row[i] for i in frozen]
            mutated = generation.unflatten(child)
            assert _paired(mutated) and _policy(mutated) == default
        # No coordinate move from such a row steps off the policy.
        moves = generation.coordinate_neighbor_projections(rows[0])
        assert any(move.outcome == "candidate" for move in moves)
        for move in moves:
            if move.key in names:
                assert move.outcome == "incumbent_alias"
            elif move.config is not None and _paired(move.config):
                assert _policy(move.config) == default
        # Whole draws: every one that decodes to a paired config has it.
        # Seeded, so that a draw that breaks the rule is drawn again; the
        # process's generator then goes back to where it was.
        state = random.getstate()
        random.seed(20261126)
        try:
            for _ in range(200):
                try:
                    config = generation.unflatten(generation.random_flat())
                except helion.exc.InvalidConfig:
                    continue
                assert not _paired(config) or _policy(config) == default
        finally:
            random.setstate(state)
    # With bank cycles the search samples the policy, and such a row keeps
    # the policy its slots name.
    with _compilation():
        generation, base, _parent, slots = _surface(overrides=overrides)
        row = [*base]
        for key, value in {**off, **drawn[0]}.items():
            row[slots[key]] = value
        config = generation.unflatten(row)
        assert _paired(config) and _policy(config)[:2] == (False, False)


@pytest.mark.parametrize(
    "overrides,policy,moves",
    [
        # An override the policy cannot keep declines it, and the search is
        # that of a kernel without the default: ordinary dependent launch
        # and spread padding are its knobs.
        ({PDL: False}, (False, False, "auto", True), {SPREAD}),
        ({CAPTURE: False}, (False, False, "auto", True), {PDL, SPREAD}),
        # So is the search under a release named beside the overlap's
        # opt-out. The decline hook leaves a candidate that shows nothing of
        # the policy alone: reading the overridden release as the tail
        # release would decline ordinary dependent launch in every
        # candidate, and these moves would stop being candidates.
        (
            {CAPTURE: False, RELEASE: "auto"},
            (False, False, "auto", True),
            {PDL, SPREAD},
        ),
        (
            {CAPTURE: False, RELEASE: "entry"},
            (False, False, "entry", True),
            {PDL, SPREAD},
        ),
    ],
)
def test_search_without_bank_cycles_follows_an_override_that_declines(
    overrides, policy, moves
):
    with _compilation(False):
        generation, base, parent, _slots = _surface(
            overrides=overrides, capture_search=False
        )
        assert _policy(parent) == policy
        candidates = {
            move.key
            for move in generation.coordinate_neighbor_projections(base)
            if move.key in (PDL, CAPTURE, RELEASE, SPREAD)
            and move.outcome == "candidate"
        }
        assert candidates == moves


@pytest.mark.parametrize(
    "overrides,policy",
    [
        # A search without bank cycles gives the seed the default policy:
        (None, (True, True, "loader_tail", False)),
        # only the opt-outs, as fixed overrides, keep it out of the search.
        ({PDL: False}, (False, False, "auto", False)),
        ({CAPTURE: False}, (False, False, "auto", False)),
    ],
)
def test_a_seed_cannot_opt_a_search_out_of_the_policy_but_an_override_does(
    overrides, policy
):
    # An application that edits captured graphs must keep the overlap out of
    # the config a search returns. Naming ``cute_pdl=False`` in a seed does
    # not do that; ``autotune_config_overrides`` does.
    seed = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        cute_pdl=False,
    )
    with _compilation(False):
        spec = _bound(False).config_spec
        # Compiled as written, the config has no dependent launch.
        assert _policy(spec.normalized_config(seed)) == (False, False, "auto", False)
        generation = spec.create_config_generation(overrides=overrides)
        slots = {
            key: entry[0][0] for key, entry in generation._key_to_flat_indices.items()
        }
        ((flat, config),) = generation.user_seed_flat_config_pairs([seed])
        assert _paired(config)
        assert _policy(config) == policy
        # A cached config of an earlier search is decoded from its flat row:
        # one stored without dependent launch reads the same way.
        cached = [*flat]
        for key, value in ((PDL, False), (CAPTURE, False), (RELEASE, "auto")):
            cached[slots[key]] = value
        assert _policy(generation.unflatten(cached)) == policy
        # So does every paired config the search starts from.
        seeds = generation.seed_flat_config_pairs()
        population = [
            generation.unflatten(row)
            for row in generation.random_population_flat(
                len(seeds) + 2, user_seed_configs=[seed]
            )
        ]
        assert config in population
        paired = [member for member in population if _paired(member)]
        assert len(paired) > 1
        assert all(_policy(member) == policy for member in paired)
    # With bank cycles the search samples the policy, and the seed keeps
    # what it names.
    if overrides is None:
        with _compilation():
            banked = _bound().config_spec.create_config_generation()
            ((_flat, kept),) = banked.user_seed_flat_config_pairs([seed])
            assert _paired(kept)
            assert _policy(kept) == (False, False, "auto", False)


def _reply(spec, named, *, paired=True):
    """The options an LLM names to put the kernel on a compiler seed's
    schedule, the paired one or another, with ``named`` for its
    dependent-launch policy."""
    seed = next(seed for seed in spec.compiler_seed_configs if _paired(seed) is paired)
    return {**_family(seed), **named}


def _parsed(spec, reply):
    """``reply`` as an LLM-guided search takes it in: merged over the
    reference config and normalized."""
    (config,) = parse_response_configs(
        json.dumps([reply]),
        config_spec=spec,
        default_config_dict=dict(spec.autotune_reference_config()),
        log=Mock(),
    )
    return config


@pytest.mark.parametrize(
    "named,overrides,policy,returned",
    [
        # The reply names no dependent-launch option; the reference config
        # it is merged over names ``cute_pdl=False``.
        ({}, None, (True, True, "loader_tail", False), (PDL, CAPTURE, RELEASE)),
        # It names another policy: ordinary dependent launch on a release,
        # spread padding (inert without ``cute_pdl``).
        (
            {PDL: True, RELEASE: "entry"},
            None,
            (True, True, "loader_tail", False),
            (CAPTURE, RELEASE),
        ),
        (
            {SPREAD: True},
            None,
            (True, True, "loader_tail", False),
            (PDL, CAPTURE, RELEASE, SPREAD),
        ),
        # It spells out the kept policy: nothing changes.
        (
            {PDL: True, CAPTURE: True, RELEASE: "loader_tail"},
            None,
            (True, True, "loader_tail", False),
            (),
        ),
        # An override the policy keeps is part of what the search keeps.
        (
            {},
            {RELEASE: "entry"},
            (True, True, "entry", False),
            (PDL, CAPTURE, RELEASE),
        ),
        # Under an opt-out the search keeps none, and a reply that spells
        # the policy out loses it.
        ({}, {CAPTURE: False}, (False, False, "auto", False), ()),
        (
            {PDL: True, CAPTURE: True, RELEASE: "loader_tail"},
            {CAPTURE: False},
            (False, False, "auto", False),
            (PDL, CAPTURE, RELEASE),
        ),
    ],
)
def test_a_proposal_on_the_paired_schedule_has_the_policy_of_its_search(
    named, overrides, policy, returned
):
    # A proposal from outside the flat fields is never decoded from a flat
    # config and no override is layered over it, so neither the repair that
    # keeps the policy nor the hook that declines it reaches it; the backend
    # holds it to both (``ConfigSpec.search_proposal``).
    with _compilation(False):
        spec = _bound(False).config_spec
        config = _parsed(spec, _reply(spec, named))
        assert _paired(config)
        stated = copy.deepcopy(config.config)
        proposal, changed = spec.search_proposal(config, overrides)
        assert changed == returned
        assert _policy(proposal) == policy
        assert _family(proposal) == _family(config)
        # A proposal that states that policy already is admitted as it is,
        # and the caller's config is never written.
        assert (proposal is config) is (not returned)
        assert config.config == stated
        # The same config has that policy as a seed of the search.
        generation = spec.create_config_generation(overrides=overrides)
        ((_flat, seeded),) = generation.user_seed_flat_config_pairs([config])
        assert _policy(seeded) == policy


def test_a_schedule_override_moves_a_seed_and_leaves_a_proposal_its_schedule():
    # An override to another schedule is layered over every candidate a
    # search decodes, a seed among them: it leaves the paired schedule, and
    # the overlap with it. Nothing is layered over a proposal from outside
    # the flat fields, so it is benchmarked on the schedule its reply names.
    # The search keeps no policy under such an override: the proposal has
    # the policy its reply states, the overlap included.
    overrides = {"cute_mma_dag_pass_warps": "attention"}
    off = (False, False, "auto", False)
    ordinary = (True, False, "auto", False)
    default = (True, True, "loader_tail", False)
    spelled = {PDL: True, CAPTURE: True, RELEASE: "loader_tail"}
    with _compilation(False):
        spec = _bound(False).config_spec
        generation = spec.create_config_generation(overrides=overrides)
        replies = []
        for named, stated, seed_policy in (
            ({}, off, off),
            ({PDL: True}, ordinary, ordinary),
            (spelled, default, off),
        ):
            reply = _reply(spec, named)
            replies.append(reply)
            config = _parsed(spec, reply)
            assert _paired(config) and _policy(config) == stated
            proposal, returned = spec.search_proposal(config, overrides)
            assert proposal is config and returned == ()
            ((_flat, seeded),) = generation.user_seed_flat_config_pairs([config])
            assert seeded.get("cute_mma_dag_pass_warps") == "attention"
            assert _policy(seeded) == seed_policy
        # An LLM-guided search under the override takes the replies in as
        # they are.
        search = LLMGuidedSearch.__new__(LLMGuidedSearch)
        search.config_spec = spec
        search._default_config_dict = dict(spec.autotune_reference_config())
        search.settings = SimpleNamespace(autotune_config_overrides=overrides)
        messages: list[str] = []
        search.log = messages.append
        admitted = search._parse_configs(json.dumps({"configs": replies}))
        assert all(map(_paired, admitted))
        assert [_policy(config) for config in admitted] == [off, ordinary, default]
        assert messages == ["Parsed 3 valid configs from LLM response"]


def test_a_proposal_under_bank_cycles_keeps_the_policy_it_names():
    spelled = {PDL: True, CAPTURE: True, RELEASE: "loader_tail"}
    with _compilation():
        spec = _bound().config_spec
        # A search over bank cycles samples the policy: a proposal keeps
        # what it names, as a seed does.
        for named in ({}, {PDL: True}, spelled):
            config = _parsed(spec, _reply(spec, named))
            assert _paired(config)
            proposal, returned = spec.search_proposal(config)
            assert proposal is config
            assert returned == ()


@pytest.mark.parametrize(
    "capture_search,named,overrides,policy,returned",
    [
        # A search over bank cycles samples the policy and keeps none. A
        # fixed override of one of its options holds in every candidate all
        # the same: one that names a release,
        (
            True,
            {PDL: True, CAPTURE: True, RELEASE: "auto"},
            {RELEASE: "loader_tail"},
            (True, True, "loader_tail", False),
            (RELEASE,),
        ),
        # overrides that turn the policy on over a reply without it,
        (
            True,
            {},
            {PDL: True, CAPTURE: True},
            (True, True, "auto", False),
            (PDL, CAPTURE),
        ),
        # and the opt-outs. The default spelled out loses dependent launch;
        # the overlap on the automatic release stays as ordinary dependent
        # launch.
        (
            True,
            {PDL: True, CAPTURE: True, RELEASE: "loader_tail"},
            {PDL: False},
            (False, False, "auto", False),
            (PDL, CAPTURE, RELEASE),
        ),
        (
            True,
            {PDL: True, CAPTURE: True},
            {CAPTURE: False},
            (True, False, "auto", False),
            (CAPTURE,),
        ),
        # Without bank cycles a search keeps no policy where its overrides
        # decline it, and what they fix beside that holds there too.
        (
            False,
            {},
            {PDL: True, CAPTURE: False},
            (True, False, "auto", False),
            (PDL,),
        ),
        (
            False,
            {PDL: True, RELEASE: "entry"},
            {RELEASE: "exit"},
            (True, False, "exit", False),
            (RELEASE,),
        ),
    ],
)
def test_a_proposal_gets_the_overridden_options_where_its_search_keeps_no_policy(
    capture_search, named, overrides, policy, returned
):
    # The overrides of the policy's own options reach a proposal on the
    # paired schedule the way they reach a candidate: layered over it, with
    # the candidate's decline.
    with _compilation(capture_search):
        spec = _bound(capture_search).config_spec
        config = _parsed(spec, _reply(spec, named))
        assert _paired(config)
        stated = copy.deepcopy(config.config)
        proposal, changed = spec.search_proposal(config, overrides)
        assert changed == returned
        assert _policy(proposal) == policy
        assert _family(proposal) == _family(config)
        assert config.config == stated
        # The same config has that policy as a seed of the search.
        generation = spec.create_config_generation(overrides=overrides)
        ((_flat, seeded),) = generation.user_seed_flat_config_pairs([config])
        assert _policy(seeded) == policy


def test_a_proposal_on_which_an_override_cannot_hold_is_dropped():
    # Capture overlap fixed over a reply without ``cute_pdl`` is no candidate
    # of the search: strict normalization rejects the proposal as it rejects
    # that seed, and the search drops it.
    overrides = {CAPTURE: True}
    reason = "cute_pdl_capture_overlap=True requires"
    with _compilation():
        spec = _bound().config_spec
        config = _parsed(spec, _reply(spec, {}))
        assert _paired(config) and _policy(config)[0] is False
        with pytest.raises(helion.exc.InvalidConfig, match=reason):
            spec.search_proposal(config, overrides)
        failures: list[str] = []
        generation = spec.create_config_generation(overrides=overrides)
        assert not generation.user_seed_flat_config_pairs([config], failures.append)
        assert len(failures) == 1 and reason in failures[0]
        search = LLMGuidedSearch.__new__(LLMGuidedSearch)
        search.config_spec = spec
        search._default_config_dict = dict(spec.autotune_reference_config())
        search.settings = SimpleNamespace(autotune_config_overrides=overrides)
        messages: list[str] = []
        search.log = messages.append
        assert search._parse_configs(json.dumps({"configs": [_reply(spec, {})]})) == []
        parsed, dropped = messages
        assert parsed == "Parsed 1 valid configs from LLM response"
        assert dropped.startswith(
            "Dropped an LLM proposal that is not legal with its options as "
            f"this search has them: {reason}"
        )


def test_a_proposal_off_the_paired_schedule_is_left_alone():
    # There is no capture contract to keep or to decline.
    with _compilation(False):
        spec = _bound(False).config_spec
        config = _parsed(spec, _reply(spec, {PDL: True}, paired=False))
        assert not _paired(config)
        for overrides in (None, {CAPTURE: False}):
            proposal, returned = spec.search_proposal(config, overrides)
            assert proposal is config
            assert returned == ()


def test_an_llm_guided_search_benchmarks_a_paired_reply_with_its_own_policy():
    # The search hands each config of a reply to the backend with its own
    # overrides (``LLMGuidedSearch._admit_proposals``).
    spelled = {PDL: True, CAPTURE: True, RELEASE: "loader_tail"}
    with _compilation(False):
        spec = _bound(False).config_spec
        search = LLMGuidedSearch.__new__(LLMGuidedSearch)
        search.config_spec = spec
        search._default_config_dict = dict(spec.autotune_reference_config())
        messages: list[str] = []
        search.log = messages.append
        response = json.dumps(
            {
                "configs": [
                    _reply(spec, {}),
                    _reply(spec, spelled),
                    _reply(spec, {PDL: True}, paired=False),
                ]
            }
        )
        changed = (
            "An LLM proposal states options as this search does not propose "
            f"them ({PDL}, {CAPTURE}, {RELEASE}); benchmarking it as the "
            "search has them"
        )
        for overrides, policy in (
            # The reply that names no policy gets the kept one,
            ({}, (True, True, "loader_tail", False)),
            # and under an opt-out the reply that spells it out loses it.
            ({CAPTURE: False}, (False, False, "auto", False)),
        ):
            search.settings = SimpleNamespace(autotune_config_overrides=overrides)
            messages.clear()
            unnamed, named, unpaired = search._parse_configs(response)
            assert _paired(unnamed) and _paired(named)
            assert _policy(unnamed) == _policy(named) == policy
            assert not _paired(unpaired)
            assert _policy(unpaired) == (True, False, "auto", False)
            assert messages == ["Parsed 3 valid configs from LLM response", changed]


def test_explicit_capture_survives_default_search_domain_and_renders():
    with _compilation(False):
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
    source = _bound(False).to_code(config)
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
