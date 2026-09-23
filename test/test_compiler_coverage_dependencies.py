from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from dataclasses import replace
import random
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from test.test_compiler_coverage import make_search
from test.test_compiler_coverage import make_spec

from helion.autotuner.compiler_coverage import CompilerCoverageGroup
from helion.autotuner.compiler_coverage import CoverageDependency
from helion.autotuner.compiler_coverage import CoverageWitness
from helion.autotuner.compiler_coverage import append_compiler_coverage
from helion.autotuner.config_fragment import BooleanFragment
from helion.runtime.config import Config

if TYPE_CHECKING:
    from helion.autotuner.config_spec import ConfigSpec


def dependent_group(
    *,
    key: str = "coupled",
    dependencies: tuple[CoverageDependency, ...] | None = None,
    carrier: Config | None = None,
) -> CompilerCoverageGroup:
    return CompilerCoverageGroup(
        key,
        1,
        key,
        (False, True),
        False,
        tuple(
            CoverageWitness(
                carrier
                if carrier is not None
                else Config(block_sizes=[32], old_choice=1, coverage_mode="a"),
                value,
            )
            for value in (False, True)
        ),
        dependencies=(CoverageDependency("coverage_mode", "coverage_mode", "a"),)
        if dependencies is None
        else dependencies,
    )


def add_group(spec: ConfigSpec, group: CompilerCoverageGroup) -> None:
    spec.user_defined_tunables[group.key] = BooleanFragment()
    spec.register_compiler_coverage_group(group)


def test_actual_lfbo_preserves_complete_old_population_and_rng() -> None:
    old_spec = make_spec()
    spec = make_spec()
    old_groups = [group.policy() for group in spec.compiler_coverage_groups]
    old_default = copy.deepcopy(spec.compiler_default_config)
    old_seeds = copy.deepcopy(spec.compiler_seed_configs)
    add_group(spec, dependent_group())
    assert [
        group.policy() for group in spec.compiler_coverage_groups[:-1]
    ] == old_groups
    assert spec.compiler_default_config == old_default
    assert spec.compiler_seed_configs == old_seeds
    old, new = make_search(old_spec), make_search(spec)
    random.seed(1713)
    expected = old._generate_initial_population_flat()
    state = random.getstate()
    random.seed(1713)
    actual = new._generate_initial_population_flat()
    assert random.getstate() == state
    new_indices = new.config_gen._key_to_flat_indices["coupled"][0]
    assert [
        [value for index, value in enumerate(row) if index not in new_indices]
        for row in actual[: len(expected)]
    ] == expected
    assert len(actual) == len(expected) + 1
    config = new.config_gen.unflatten(copy.deepcopy(actual[-1]))
    assert config["coverage_mode"] == "a" and config["coupled"] is True
    outcomes = [
        entry.outcome
        for entry in new.compiler_coverage_outcomes
        if entry.mechanism == "coupled"
    ]
    assert outcomes == ["already_present", "added"]


@pytest.mark.parametrize(
    "entry",
    (
        CoverageDependency("missing", "coverage_mode", "a"),
        CoverageDependency("coverage_mode", "missing", "a"),
        CoverageDependency("coverage_mode", "coverage_bool", "a"),
        CoverageDependency("coupled", "coupled", True),
        CoverageDependency("later", "later", True),
        CoverageDependency("coverage_mode", "coverage_mode", "unknown"),
        CoverageDependency("coverage_bool", "coverage_bool", 1),
    ),
)
def test_missing_unknown_self_future_and_typed_domain_reject(entry) -> None:
    spec = make_spec()
    before = spec.compiler_coverage_groups
    with pytest.raises(ValueError):
        add_group(spec, dependent_group(dependencies=(entry,)))
    assert spec.compiler_coverage_groups == before


@pytest.mark.parametrize(
    "entries",
    (
        (
            CoverageDependency("coverage_mode", "coverage_mode", "a"),
            CoverageDependency("coverage_mode", "coverage_mode", "b"),
        ),
        (
            CoverageDependency("coverage_mode", "coverage_mode", "a"),
            CoverageDependency("other", "coverage_mode", "a"),
        ),
    ),
)
def test_duplicate_contradictory_declarations_reject(entries) -> None:
    with pytest.raises(ValueError, match="Duplicate or contradictory"):
        dependent_group(dependencies=entries)


@pytest.mark.parametrize(
    "carrier",
    (
        Config(block_sizes=[32], old_choice=1),
        Config(block_sizes=[32], old_choice=1, coverage_mode="b"),
        Config(block_sizes=[32], old_choice=1, coverage_mode="a", coverage_bool=True),
        Config(block_sizes=[32], old_choice=1, coverage_mode="a", coupled=True),
    ),
)
def test_carrier_must_explicitly_match_dependencies_only(carrier) -> None:
    with pytest.raises(ValueError, match="carrier"):
        add_group(make_spec(), dependent_group(carrier=carrier))


@pytest.mark.parametrize("complete", (False, True))
@pytest.mark.parametrize("contradictory", (False, True))
def test_transitive_dependencies_are_explicit_and_consistent(
    complete: bool, contradictory: bool
) -> None:
    spec = make_spec()
    add_group(spec, dependent_group())
    entries = [CoverageDependency("coupled", "coupled", True)]
    if complete:
        entries.append(
            CoverageDependency(
                "coverage_mode", "coverage_mode", "b" if contradictory else "a"
            )
        )
    group = dependent_group(
        key="third",
        dependencies=tuple(entries),
        carrier=Config(block_sizes=[32], old_choice=1, coverage_mode="a", coupled=True),
    )
    if complete and not contradictory:
        add_group(spec, group)
        rows, outcomes = append_compiler_coverage(
            [], spec.create_config_generation(), cached_configs=lambda: (), pin=Mock()
        )
        assert len(rows) == 7
        assert outcomes[-1].outcome == "added"
        assert outcomes[-1].effective is not None
        assert outcomes[-1].effective["third"] is True
    else:
        with pytest.raises(ValueError, match="transitive"):
            add_group(spec, group)


def test_stale_reordered_registry_cannot_create_cycle() -> None:
    spec = make_spec()
    add_group(spec, dependent_group())
    spec._compiler_coverage_groups = (
        spec.compiler_coverage_groups[-1],
        *spec.compiler_coverage_groups[:-1],
    )
    with pytest.raises(ValueError, match="earlier"):
        spec.create_config_generation().initial_population_view()


@pytest.mark.parametrize("change", ("drop", "other", "wrong_type", "geometry"))
def test_strict_normalization_cannot_change_a_dependency_or_carrier(change) -> None:
    spec = make_spec()
    add_group(spec, dependent_group())
    original = spec.normalize

    def normalize(config, *, _fix_invalid=False):
        original(config, _fix_invalid=_fix_invalid)
        if config.get("coupled") is True:
            if change == "drop":
                config.pop("coverage_mode", None)
            elif change == "other":
                config["coverage_mode"] = "b"
            elif change == "wrong_type":
                config["coverage_mode"] = 1
            else:
                config["old_choice"] = 2

    with patch.object(spec, "normalize", side_effect=normalize):
        rows, outcomes = append_compiler_coverage(
            [], spec.create_config_generation(), cached_configs=lambda: (), pin=Mock()
        )
    assert len(rows) == 5
    assert outcomes[-1].outcome.startswith("invalid:") or (
        outcomes[-1].outcome == "changed_carrier_geometry"
    )


@pytest.mark.parametrize("value", ("off", "a", "b"))
def test_global_override_cannot_change_a_declared_dependency(value: str) -> None:
    spec = make_spec()
    add_group(spec, dependent_group())
    search = make_search(spec, overrides={"coverage_mode": value})
    rows = search._generate_initial_population_flat()
    configs = [search.config_gen.unflatten(copy.deepcopy(row)) for row in rows]
    assert all(config["coverage_mode"] == value for config in configs)
    assert any(config["coupled"] is True for config in configs) is (value == "a")


def test_legacy_policy_bytes_and_dependent_cache_identity() -> None:
    spec = make_spec()
    for group in spec.compiler_coverage_groups:
        assert group.policy() == {
            "mechanism": group.mechanism,
            "version": group.version,
            "key": group.key,
            "domain": group.domain,
            "legacy": group.legacy,
            "witnesses": tuple(
                (witness._carrier_json, witness.value) for witness in group.witnesses
            ),
            "max_additions": 4,
        }
    other = make_spec()
    group = dependent_group()
    add_group(spec, group)
    add_group(
        other,
        replace(
            group,
            dependencies=(CoverageDependency("coverage_mode", "coverage_mode", "b"),),
            witnesses=tuple(
                CoverageWitness(
                    Config(block_sizes=[32], old_choice=1, coverage_mode="b"), value
                )
                for value in (False, True)
            ),
        ),
    )
    assert spec.structural_fingerprint_hash() == other.structural_fingerprint_hash()
    assert spec.cache_fingerprint_hash() != other.cache_fingerprint_hash()
    assert (
        spec.projected_cache_fingerprint_hash()
        == other.projected_cache_fingerprint_hash()
    )


def test_declarations_and_carriers_remain_deep_owned() -> None:
    dependency = CoverageDependency("coverage_mode", "coverage_mode", "a")
    entries = [dependency]
    group = dependent_group(dependencies=tuple(entries))
    entries.clear()
    for attribute in ("mechanism", "key", "value"):
        with pytest.raises(FrozenInstanceError):
            setattr(dependency, attribute, "changed")
    carrier = group.witnesses[0].carrier
    carrier.config["coverage_mode"] = "b"
    assert group.witnesses[0].carrier["coverage_mode"] == "a"
    assert group.dependencies == (dependency,)


def test_multi_active_warm_cache_filter_is_unchanged() -> None:
    spec = make_spec()
    add_group(spec, dependent_group())
    warm = [Config(block_sizes=[32], old_choice=210, coverage_mode="a", coupled=True)]
    rows, outcomes = append_compiler_coverage(
        [], spec.create_config_generation(), cached_configs=lambda: warm, pin=Mock()
    )
    assert len(rows) == 6
    assert not any(entry.origin == "cache" for entry in outcomes)


def test_missing_control_counts_toward_the_existing_four_addition_limit() -> None:
    spec = make_spec()
    group = dependent_group(
        carrier=Config(block_sizes=[64], old_choice=240, coverage_mode="a")
    )
    add_group(spec, group)
    rows, outcomes = append_compiler_coverage(
        [], spec.create_config_generation(), cached_configs=lambda: (), pin=Mock()
    )
    assert len(rows) == 7
    assert [entry.outcome for entry in outcomes[-2:]] == ["added", "added"]
