from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import json
import os
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from test.cute_population_contracts import _pointwise
from test.cute_population_contracts import _target
from test.test_cute_flash_length_invariance import _flash_config_spec

import helion
from helion._compiler.cute.backend import CuteBackend
from helion._dist_utils import sync_seed
from helion._testing import skipUnlessBackends
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.compiler_coverage import CompilerCoverageGroup
from helion.autotuner.compiler_coverage import CoverageWitness
from helion.autotuner.compiler_coverage import append_compiler_coverage
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import CuteLaneLayoutSpec
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.exc import InvalidConfig
from helion.runtime.config import Config
from helion.runtime.settings import Settings
from helion.runtime.settings import default_autotuner_fn

if TYPE_CHECKING:
    from helion.autotuner.config_generation import ConfigGeneration

DOMAINS = {
    "coverage_mode": ("off", "a", "b", "c"),
    "coverage_bool": (False, True),
    "coverage_layout": ("runtime", "static"),
}


def make_spec(*, expanded: bool = True, many_seeds: bool = False) -> ConfigSpec:
    fields = {"old_choice": EnumFragment(tuple(range(256)))}
    if expanded:
        fields.update(
            {
                "coverage_mode": EnumFragment(DOMAINS["coverage_mode"]),
                "coverage_bool": BooleanFragment(),
                "coverage_layout": EnumFragment(DOMAINS["coverage_layout"]),
            }
        )
    spec = ConfigSpec(
        backend=CuteBackend(),
        user_defined_tunables=fields,
        target_device_capability=None,
    )
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=128))
    spec.compiler_default_config = Config(block_sizes=[64], old_choice=250)
    spec.compiler_seed_configs = [
        Config(block_sizes=[32], old_choice=value)
        for value in range(1, 121 if many_seeds else 4)
    ]
    if expanded:
        for key, domain in DOMAINS.items():
            witnesses = tuple(
                CoverageWitness(spec.compiler_seed_configs[0], value)
                for value in domain[1:]
            )
            spec.register_compiler_coverage_group(
                CompilerCoverageGroup(key, 1, key, domain, domain[0], witnesses)
            )
    return spec


def make_search(
    spec: ConfigSpec,
    *,
    cls: type[PatternSearch] = LFBOTreeSearch,
    strategy: InitialPopulationStrategy = InitialPopulationStrategy.FROM_RANDOM,
    count: int = 100,
    pad: bool = True,
    seeds: tuple[Config, ...] = (),
    overrides: dict[str, object] | None = None,
    disabled: bool = False,
) -> PatternSearch:
    # Run the real constructor and production assembly without starting a
    # benchmark provider or binding/launching a device kernel.
    settings = Settings(
        backend="cute",
        autotune_seed_configs=seeds,
        autotune_config_overrides=overrides or {},
        disable_autotuner_heuristics=disabled,
    )
    kernel = SimpleNamespace(
        settings=settings,
        config_spec=spec,
        env=SimpleNamespace(process_group_name=None),
    )
    search = cls(
        kernel,
        (),
        initial_population=count,
        initial_population_strategy=strategy,
        best_available_pad_random=pad,
    )
    search.log = Mock()
    search._find_similar_cached_configs = Mock(return_value=[])
    return search


def old_slots(generation: ConfigGeneration, row: list[object]) -> list[object]:
    return [
        copy.deepcopy(row[index])
        for key, (indices, _) in generation._key_to_flat_indices.items()
        if key not in DOMAINS
        for index in indices
    ]


@pytest.mark.parametrize("cls", (PatternSearch, LFBOPatternSearch, LFBOTreeSearch))
@pytest.mark.parametrize(
    ("strategy", "pad"),
    (
        (InitialPopulationStrategy.FROM_RANDOM, True),
        (InitialPopulationStrategy.FROM_BEST_AVAILABLE, False),
        (InitialPopulationStrategy.FROM_BEST_AVAILABLE, True),
    ),
)
def test_actual_initial_assembly_preserves_base_and_rng(cls, strategy, pad) -> None:
    seeds = (Config(block_sizes=[64], old_choice=240),)
    old = make_search(
        make_spec(expanded=False), cls=cls, strategy=strategy, pad=pad, seeds=seeds
    )
    full = make_search(make_spec(), cls=cls, strategy=strategy, pad=pad, seeds=seeds)
    compiler_seeds = copy.deepcopy(full.config_spec.compiler_seed_configs)
    identity = full.config_spec.cache_fingerprint_hash()
    random.seed(1234)
    expected = old._generate_initial_population_flat()
    rng = random.getstate()
    random.seed(1234)
    actual = full._generate_initial_population_flat()
    assert random.getstate() == rng
    assert [
        old_slots(full.config_gen, row) for row in actual[: len(expected)]
    ] == expected
    assert len(actual) == len(expected) + 5
    assert full.config_spec.compiler_seed_configs == compiler_seeds
    assert full.config_spec.cache_fingerprint_hash() == identity
    assert (
        len(
            [
                entry
                for entry in full.compiler_coverage_outcomes
                if entry.outcome == "added"
            ]
        )
        == 5
    )
    assert full.config_gen._initial_sampling is False


def test_empty_registry_is_identity_view_and_legacy_cache_hash() -> None:
    spec = make_spec(expanded=False)
    generation = spec.create_config_generation()
    assert generation.initial_population_view() is generation
    assert spec.projected_cache_fingerprint_hash() == spec.cache_fingerprint_hash()
    search = make_search(spec)
    with patch.object(
        search, "_append_compiler_coverage", side_effect=AssertionError("append")
    ):
        assert len(search._generate_initial_population_flat()) == 100
    assert "compiler_coverage" not in search._algorithm_cache_policy()


def test_persistent_reduction_raw_row_is_not_normalized_during_lift() -> None:
    def spec(expanded: bool) -> ConfigSpec:
        result = ConfigSpec(
            backend=CuteBackend(),
            target_device_capability=None,
            user_defined_tunables={"coverage_bool": BooleanFragment()}
            if expanded
            else {},
        )
        result.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=128))
        result.cute_lane_layouts.append(CuteLaneLayoutSpec(block_id=0))
        result.reduction_block_ids.add(0)
        if expanded:
            result.register_compiler_coverage_group(
                CompilerCoverageGroup(
                    "reduction", 1, "coverage_bool", (False, True), False, ()
                )
            )
        return result

    old = spec(False).create_config_generation()
    full = spec(True).create_config_generation().initial_population_view()
    random.seed(2026091441)
    expected = [old.random_flat() for _ in range(32)]
    rng = random.getstate()
    random.seed(2026091441)
    rows = [full.random_flat() for _ in range(32)]
    assert random.getstate() == rng
    assert [old_slots(full, row) for row in rows] == expected
    index = full._key_to_flat_indices["cute_lane_layouts"][0][0]
    row = next(row for row in rows if row[index] == "strided")
    assert full.unflatten(copy.deepcopy(row))["cute_lane_layouts"] == ["blocked"]
    assert row[index] == "strided"


@pytest.mark.parametrize("biased", (False, True))
def test_invalid_raw_rows_survive_projected_sampling(biased) -> None:
    generation = make_spec().create_config_generation().initial_population_view()
    sampler = generation._compiler_coverage_sampler
    assert sampler is not None
    raw = sampler.default_flat()
    raw[0] = "invalid-block-size"
    method = "biased_random_flat" if biased else "random_flat"
    with patch.object(sampler, method, return_value=raw):
        actual = getattr(generation, method)()
    assert old_slots(generation, actual) == raw
    with pytest.raises((InvalidConfig, ValueError, TypeError, AssertionError)):
        generation.unflatten(actual)


@pytest.mark.parametrize("count", (-1, 0, 1, 3))
def test_small_and_nonpositive_targets_keep_old_prefix(count) -> None:
    old = make_search(make_spec(expanded=False), count=count)
    full = make_search(make_spec(), count=count)
    random.seed(3456)
    expected = old._generate_initial_population_flat()
    rng = random.getstate()
    random.seed(3456)
    actual = full._generate_initial_population_flat()
    assert random.getstate() == rng
    assert [
        old_slots(full.config_gen, row) for row in actual[: len(expected)]
    ] == expected
    assert len(actual) == len(expected) + (5 if count > 0 else 0)


@pytest.mark.parametrize("pad", (False, True))
def test_oversized_best_available_is_not_sliced(pad) -> None:
    search = make_search(
        make_spec(many_seeds=True),
        strategy=InitialPopulationStrategy.FROM_BEST_AVAILABLE,
        count=100,
        pad=pad,
        seeds=(
            Config(block_sizes=[64], old_choice=240),
            Config(block_sizes=[64], old_choice=241),
        ),
    )
    search.set_best_available_seed_configs([Config(block_sizes=[64], old_choice=242)])
    rows = search._generate_initial_population_flat()
    assert len(rows) == 124 + 5


def test_explicit_seed_modes_are_not_projected_or_globally_vetoed() -> None:
    seeds = (
        Config(block_sizes=[64], old_choice=200, coverage_bool=False),
        Config(block_sizes=[64], old_choice=200, coverage_bool=True),
    )
    search = make_search(make_spec(), seeds=seeds)
    rows = search._generate_initial_population_flat()
    configs = [search.config_gen.unflatten(copy.deepcopy(row)) for row in rows]
    matching = [
        config["coverage_bool"]
        for config in configs
        if config["old_choice"] == 200 and config["block_sizes"] == [64]
    ]
    assert False in matching and True in matching
    assert any(
        entry.outcome == "added" and entry.mechanism == "coverage_bool"
        for entry in search.compiler_coverage_outcomes
    )


@pytest.mark.parametrize("disabled", (False, True))
def test_global_legacy_override_and_disabled_heuristics(disabled) -> None:
    overrides = (
        None if disabled else {key: domain[0] for key, domain in DOMAINS.items()}
    )
    search = make_search(make_spec(), overrides=overrides, disabled=disabled)
    rows = search._generate_initial_population_flat()
    assert len(rows) == 100
    for row in rows:
        config = search.config_gen.unflatten(copy.deepcopy(row))
        assert all(config[key] == domain[0] for key, domain in DOMAINS.items())
    if disabled:
        for _ in range(20):
            config = search.config_gen.random_config()
            assert all(config[key] == domain[0] for key, domain in DOMAINS.items())
        projections = search.config_gen.coordinate_neighbor_projections(rows[0])
        assert all(item.key not in DOMAINS for item in projections)


def test_disabled_heuristics_retain_explicit_full_domain_seed() -> None:
    search = make_search(
        make_spec(),
        disabled=True,
        seeds=(Config(block_sizes=[32], old_choice=1, coverage_bool=True),),
    )
    configs = [
        search.config_gen.unflatten(copy.deepcopy(row))
        for row in search._generate_initial_population_flat()
    ]
    assert any(config["coverage_bool"] is True for config in configs)
    assert len(configs) == 100


def test_full_neighbors_and_later_random_can_reach_new_modes() -> None:
    generation = make_spec().create_config_generation()
    projections = generation.coordinate_neighbor_projections(generation.default_flat())
    assert set(DOMAINS).issubset(
        {item.key for item in projections if item.outcome == "candidate"}
    )
    random.seed(93)
    assert any(generation.random_config()["coverage_bool"] for _ in range(20))


def test_registry_owns_nested_payload_and_rejects_duplicates() -> None:
    carrier = Config(block_sizes=[32], loop_orders=[[0, 1]])
    witness = CoverageWitness(carrier, True)
    carrier.config["loop_orders"][0][0] = 100
    returned = witness.carrier
    returned.config["block_sizes"][0] = 999
    assert witness.carrier.config == {"block_sizes": [32], "loop_orders": [[0, 1]]}
    with pytest.raises(FrozenInstanceError):
        witness.value = False
    spec = make_spec()
    with pytest.raises(ValueError, match="Duplicate"):
        spec.register_compiler_coverage_group(spec.compiler_coverage_groups[0])


@pytest.mark.parametrize(
    ("domain", "legacy", "value"),
    (
        ((False, True), 0, True),
        ((False, True), False, 1),
        (("off", "on"), "off", "bad"),
    ),
)
def test_typed_domains_reject_invalid_modes(domain, legacy, value) -> None:
    with pytest.raises(ValueError):
        CompilerCoverageGroup(
            "test", 1, "test", domain, legacy, (CoverageWitness(Config(), value),)
        )


def test_registration_rejects_non_scalar_and_stale_fields() -> None:
    spec = make_spec(expanded=False)
    group = CompilerCoverageGroup("block", 1, "block_sizes", (32, 64), 32, ())
    with pytest.raises(ValueError, match="non-scalar"):
        spec.register_compiler_coverage_group(group)
    spec.user_defined_tunables["power"] = PowerOfTwoFragment(1, 4, 1)
    with pytest.raises(ValueError, match="Boolean/Enum"):
        spec.register_compiler_coverage_group(
            CompilerCoverageGroup("power", 1, "power", (1, 2, 4), 1, ())
        )
    spec = make_spec()
    spec.user_defined_tunables["old_choice"] = EnumFragment((0, 1))
    with pytest.raises(ValueError, match="changed"):
        spec.create_config_generation().initial_population_view()


def test_policy_identity_is_ordered_but_not_part_of_structural_layout() -> None:
    spec = make_spec()
    other = make_spec()
    group = other.compiler_coverage_groups[0]
    other._compiler_coverage_groups = (
        CompilerCoverageGroup(
            group.mechanism,
            group.version + 1,
            group.key,
            group.domain,
            group.legacy,
            tuple(reversed(group.witnesses)),
        ),
        *other.compiler_coverage_groups[1:],
    )
    assert spec.structural_fingerprint_hash() == other.structural_fingerprint_hash()
    assert spec.cache_fingerprint_hash() != other.cache_fingerprint_hash()
    assert (
        spec.projected_cache_fingerprint_hash()
        == make_spec(expanded=False).cache_fingerprint_hash()
    )
    assert spec.cache_fingerprint_hash() != spec.projected_cache_fingerprint_hash()


def test_strict_witness_rejects_repair_and_geometry_changes() -> None:
    spec = make_spec()
    original = spec.normalize

    def normalize(config, *, _fix_invalid=False):
        if config.get("coverage_bool"):
            if config.get("old_choice") == 1:
                config["block_sizes"] = [64]
            elif _fix_invalid:
                config["coverage_bool"] = False
            else:
                raise InvalidConfig("test consumer has no legal lowering")
        original(config, _fix_invalid=_fix_invalid)

    generation = spec.create_config_generation()
    with patch.object(spec, "normalize", side_effect=normalize):
        rows, outcomes = append_compiler_coverage(
            [], generation, cached_configs=lambda: (), pin=Mock()
        )
    assert not any(
        generation.unflatten(copy.deepcopy(row))["coverage_bool"] for row in rows
    )
    assert any(
        entry.mechanism == "coverage_bool"
        and entry.outcome == "changed_carrier_geometry"
        for entry in outcomes
    )


def test_addition_budget_dedup_and_carrier_accounting() -> None:
    spec = make_spec()
    group = spec.compiler_coverage_groups[0]
    carrier = Config(block_sizes=[64], old_choice=211)
    spec._compiler_coverage_groups = (
        CompilerCoverageGroup(
            group.mechanism,
            1,
            group.key,
            group.domain,
            group.legacy,
            tuple(CoverageWitness(carrier, value) for value in group.domain),
        ),
    )
    generation = spec.create_config_generation()
    cached = Mock(
        return_value=[Config(block_sizes=[32], old_choice=210, coverage_mode="a")]
    )
    rows, outcomes = append_compiler_coverage(
        [], generation, cached_configs=cached, pin=Mock()
    )
    assert len(rows) == 4
    cached.assert_not_called()
    assert all(entry.outcome == "added" for entry in outcomes)
    again, outcomes = append_compiler_coverage(
        rows, generation, cached_configs=lambda: (), pin=Mock()
    )
    assert again == rows
    assert all(entry.outcome == "already_present" for entry in outcomes)


def test_expanded_warm_entries_only_fill_their_own_unused_budget() -> None:
    generation = make_spec().create_config_generation()
    warm = [
        Config(block_sizes=[32], old_choice=value, coverage_mode="a")
        for value in range(200, 205)
    ]
    warm += [
        Config(block_sizes=[32], old_choice=230, coverage_mode="b", coverage_bool=True)
    ]
    warm += [Config(block_sizes=[32], old_choice=231, coverage_bool=True)]
    rows, outcomes = append_compiler_coverage(
        [], generation, cached_configs=lambda: warm, pin=Mock()
    )
    assert len(rows) == 7
    assert (
        len(
            [
                entry
                for entry in outcomes
                if entry.origin == "cache" and entry.outcome == "added"
            ]
        )
        == 2
    )
    assert all(
        generation.unflatten(copy.deepcopy(row))["old_choice"] != 230 for row in rows
    )


@pytest.mark.parametrize("pad", (False, True))
@pytest.mark.parametrize("remote", (False, True))
def test_actual_cache_parser_filter_and_two_layout_transfer(
    tmp_path, pad, remote
) -> None:
    old_spec = make_spec(expanded=False)
    spec = make_spec()
    old_gen = old_spec.create_config_generation()
    full_gen = spec.create_config_generation()
    payloads = []

    def cache(generation, config, *, spec_hash=None, hardware="coverage-cpu"):
        flat = generation.flatten(config)
        return {
            "key": {
                "fields": {
                    "hardware": hardware,
                    "specialization_key": "coverage-specialization",
                    "config_spec_hash": spec_hash
                    or generation.config_spec.cache_fingerprint_hash(),
                }
            },
            "config": config.to_json(),
            "flat_config": flat,
        }

    for value in (201, 202):
        payloads.append(cache(old_gen, Config(block_sizes=[64], old_choice=value)))
    payloads.extend(
        (
            cache(
                full_gen, Config(block_sizes=[32], old_choice=211, coverage_mode="a")
            ),
            cache(
                full_gen, Config(block_sizes=[32], old_choice=212, coverage_bool=True)
            ),
            cache(
                full_gen,
                Config(
                    block_sizes=[32],
                    old_choice=213,
                    coverage_bool=True,
                    coverage_mode="a",
                ),
            ),
            cache(
                full_gen,
                Config(block_sizes=[32], old_choice=214, coverage_layout="static"),
                hardware="other-device",
            ),
            cache(
                full_gen,
                Config(block_sizes=[32], old_choice=215, coverage_layout="static"),
                spec_hash="old-coverage-policy",
            ),
        )
    )
    malformed = cache(
        full_gen, Config(block_sizes=[32], old_choice=216, coverage_layout="static")
    )
    malformed["flat_config"] = old_gen.default_flat()
    payloads.append(malformed)
    payloads = [json.dumps(payload) for payload in reversed(payloads)]
    if not remote:
        for index, payload in enumerate(payloads):
            filename = tmp_path / f"{index}.best_config"
            filename.write_text(payload)
            os.utime(filename, (1000 - index, 1000 - index))
    backend = SimpleNamespace(list=Mock(return_value=payloads)) if remote else None

    def search(current_spec):
        result = make_search(
            current_spec,
            strategy=InitialPopulationStrategy.FROM_BEST_AVAILABLE,
            pad=pad,
        )
        result._get_current_hardware_and_specialization = Mock(
            return_value=("coverage-cpu", "coverage-specialization")
        )
        result._find_similar_cached_configs = Mock(
            wraps=BaseSearch._find_similar_cached_configs.__get__(result)
        )
        return result

    old = search(old_spec)
    full = search(spec)
    with (
        patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ),
        patch("helion.autotuner.base_cache.should_skip_cache", return_value=False),
        patch(
            "helion.autotuner.remote_cache._load_remote_backend_if_configured",
            return_value=backend,
        ),
    ):
        random.seed(217)
        expected = old._generate_initial_population_flat()
        rng = random.getstate()
        random.seed(217)
        actual = full._generate_initial_population_flat()
        assert random.getstate() == rng
    assert [
        old_slots(full.config_gen, row) for row in actual[: len(expected)]
    ] == expected
    assert len(actual) == len(expected) + 7
    requests = full._find_similar_cached_configs.call_args_list
    assert len(requests) == 2
    assert requests[0].kwargs == {"config_spec_hash": old_spec.cache_fingerprint_hash()}
    assert requests[1].kwargs == {}
    appended = [
        full.config_gen.unflatten(copy.deepcopy(row)) for row in actual[len(expected) :]
    ]
    assert {config["old_choice"] for config in appended} == {1, 211, 212}
    if remote:
        assert all(
            call.kwargs == {"max_results": 40} for call in backend.list.call_args_list
        )


def test_advanced_controls_field_and_sync_seed_calls_are_preserved() -> None:
    old = make_spec(expanded=False).create_config_generation(
        advanced_controls_files=["first", "second"]
    )
    full = make_spec().create_config_generation(
        advanced_controls_files=["first", "second"]
    )
    view = full.initial_population_view()
    with patch(
        "helion.autotuner.config_generation.sync_seed", wraps=sync_seed
    ) as calls:
        random.seed(992)
        expected = old.random_population_flat(100)
        state = random.getstate()
        old_calls = list(calls.call_args_list)
        calls.reset_mock()
        random.seed(992)
        actual = view.random_population_flat(100)
        assert random.getstate() == state
        assert calls.call_args_list == old_calls
    assert [old_slots(view, row) for row in actual] == expected
    assert "advanced_controls_file" in view._key_to_flat_indices
    assert full._field_view is None
    assert view._field_view is not None
    assert view._field_view["block_sizes"] is not full.config_spec.block_sizes
    assert full.config_spec.projected_cache_fingerprint_hash(
        advanced_controls_files=["first", "second"]
    ) == old.config_spec.cache_fingerprint_hash(
        advanced_controls_files=["first", "second"]
    )


@pytest.mark.parametrize("outcome", ("raise", "repair_off"))
def test_full_effective_validation_precedes_counting(outcome) -> None:
    spec = make_spec()
    original = spec.normalize

    def normalize(config, *, _fix_invalid=False):
        if config.get("coverage_bool"):
            if outcome == "repair_off" or _fix_invalid:
                config["coverage_bool"] = False
            else:
                raise InvalidConfig("complete consumer geometry is unsupported")
        original(config, _fix_invalid=_fix_invalid)

    generation = spec.create_config_generation()
    with patch.object(spec, "normalize", side_effect=normalize):
        rows, outcomes = append_compiler_coverage(
            [], generation, cached_configs=lambda: (), pin=Mock()
        )
    assert len(rows) == 4
    result = next(entry for entry in outcomes if entry.mechanism == "coverage_bool")
    assert (
        result.outcome == "mode_not_effective"
        if outcome == "repair_off"
        else result.outcome.startswith("invalid:")
    )


@pytest.mark.parametrize("strategy", tuple(InitialPopulationStrategy))
def test_overrides_apply_before_strict_witness_admission(strategy) -> None:
    search = make_search(
        make_spec(),
        strategy=strategy,
        overrides={"block_sizes": [64], "coverage_bool": True},
    )
    configs = [
        search.config_gen.unflatten(copy.deepcopy(row))
        for row in search._generate_initial_population_flat()
    ]
    assert all(
        config["block_sizes"] == [64] and config["coverage_bool"] is True
        for config in configs
    )
    assert (
        len(
            [
                entry
                for entry in search.compiler_coverage_outcomes
                if entry.mechanism == "coverage_bool" and entry.outcome == "added"
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    ("pad", "target", "exact"),
    ((False, 2, False), (True, 8, False), (False, 2, True), (True, 0, True)),
)
def test_coverage_is_after_every_flash_best_available_return(
    pad, target, exact
) -> None:
    # Real full ConfigGeneration and PatternSearch; only the expensive flash
    # structural design is held at a small deterministic set for branch testing.
    spec = _flash_config_spec(
        head_dim=64, num_kv=48, dtype=torch.float16, is_causal=False
    )
    spec.user_defined_tunables.update(
        {
            "flash_old_choice": EnumFragment(tuple(range(256))),
            "coverage_bool": BooleanFragment(),
        }
    )
    carrier = spec.default_config()
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            "flash",
            1,
            "coverage_bool",
            (False, True),
            False,
            (CoverageWitness(carrier, True),),
        )
    )
    search = make_search(
        spec,
        strategy=InitialPopulationStrategy.FROM_BEST_AVAILABLE,
        count=target,
        pad=pad,
    )
    generation = search.config_gen.initial_population_view()
    configs = [
        generation.unflatten(
            generation.flatten(Config(**{**carrier.config, "flash_old_choice": value}))
        )
        for value in (20, 21)
    ]
    generation.flash_deterministic_population_configs = lambda: configs
    generation.flash_structural_population_budget = lambda target: 1
    generation.flash_structural_qualification_prefix_count = lambda: 1
    generation.flash_exact_effective_search_space_configs = lambda target: (
        configs if exact else None
    )
    with (
        patch.object(
            search.config_gen, "initial_population_view", return_value=generation
        ),
        patch.object(
            search,
            "_generate_best_available_population_flat",
            return_value=[generation.flatten(configs[1])],
        ),
    ):
        random.seed(415)
        expected = search._generate_initial_population_base(generation=generation)
        rng = random.getstate()
        random.seed(415)
        actual = search._generate_initial_population_flat()
        assert random.getstate() == rng
    assert actual[: len(expected)] == expected
    assert len(actual) == len(expected) + (1 if target > 0 else 0)


@skipUnlessBackends(["cute"])
def test_public_default_full_lfbo_empty_registry_uses_ordinary_source() -> None:
    with _target():
        args = (torch.empty(37),)
        bound = helion.kernel(_pointwise, backend="cute", autotune_effort="full").bind(
            args
        )
        source = bound.to_code(bound.config_spec.default_config())
        # CPU tensors have no GPU runtime name for the disk-cache key. Hold
        # just that query; real cache parsing/filtering is tested separately.
        with patch("helion.autotuner.local_cache.LocalAutotuneCache._generate_key"):
            cache = default_autotuner_fn(bound, args)
        search = cache.autotuner
        assert type(search) is LFBOTreeSearch
        assert search.initial_population == 100
        assert search.config_spec.compiler_coverage_groups == ()
        assert len(search._generate_initial_population_flat()) == 100
        assert bound.to_code(bound.config_spec.default_config()) == source


@pytest.mark.parametrize("cls", (PatternSearch, LFBOPatternSearch, LFBOTreeSearch))
def test_actual_autotune_loop_delivers_all_added_witnesses_to_first_benchmark(
    cls,
) -> None:
    search = make_search(make_spec(), cls=cls)
    search._autotune_metrics = AutotuneMetrics()

    class HeldBenchmark(Exception):
        pass

    delivered = []

    def hold(members, *, desc):
        assert desc == "Initial population"
        delivered.extend(copy.deepcopy(member.config) for member in members)
        raise HeldBenchmark

    with (
        patch.object(search, "benchmark_population", side_effect=hold),
        pytest.raises(HeldBenchmark),
    ):
        search._autotune()
    added = [
        entry.effective
        for entry in search.compiler_coverage_outcomes
        if entry.outcome == "added"
    ]
    assert len(added) == 5
    assert all(config in delivered for config in added)
    assert all(config in search._pinned_finalist_configs for config in added)
    assert len(delivered) == len(set(delivered))
