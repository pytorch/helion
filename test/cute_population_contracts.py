"""Audit the legacy population budget and the separate coverage additions."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

from helion._hardware import HardwareInfo
from helion._testing import patch_cute_mma_support
from helion.autotuner.compiler_coverage import same_value
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.pattern_search import PatternSearch
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.base_search import PopulationMember
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.config_generation import FlatConfig
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.config import Config


def make_unbenchmarked_member(
    generation: ConfigGeneration, flat: FlatConfig
) -> PopulationMember | None:
    """Run real canonicalization and backend viability without starting a search."""
    search = PatternSearch.__new__(PatternSearch)
    search.config_gen = generation
    search.config_spec = generation.config_spec
    return search.make_unbenchmarked(flat)


def with_flat_min_blocks_default(spec: ConfigSpec, config: Config) -> Config:
    """Express the optional public zero explicitly in the admitted flat schema."""
    result = deepcopy(config)
    if "cute_min_blocks_per_mp" in spec._flat_fields():
        result.config.setdefault("cute_min_blocks_per_mp", 0)
    return result


def checked_initial_population(search: PatternSearch) -> list[FlatConfig]:
    """Keep N exact base rows; account for every later declared witness.

    The old builder is observed directly before the append. This checks its
    nominal budget, exact prefix, and final RNG state without changing its
    inputs, seeds, cache policy or random draws. Appended rows are additionally
    matched to independently normalized immutable declarations and final pins.
    """
    base_calls = []
    coverage_calls = []
    build_base = search._generate_initial_population_base
    build_coverage = search._generate_initial_population_with_coverage

    def observe_base(**kwargs):
        result = build_base(**kwargs)
        base_calls.append((deepcopy(result), random.getstate()))
        return result

    def observe_coverage():
        result = build_coverage()
        coverage_calls.append(deepcopy(result))
        return result

    with (
        patch.object(search, "_generate_initial_population_base", observe_base),
        patch.object(
            search, "_generate_initial_population_with_coverage", observe_coverage
        ),
    ):
        final_result = search._generate_initial_population_flat()
    assert len(coverage_calls) == 1
    result = coverage_calls[0]
    _check_full_seed_coverage(search, result, final_result)
    assert len(base_calls) == 1
    base, state = base_calls[0]
    assert len(base) == search.initial_population
    assert result[: len(base)] == base
    assert random.getstate() == state
    generation = search.config_gen
    groups = generation.config_spec.compiler_coverage_groups
    if not groups or not generation.compiler_coverage_enabled:
        assert result == base
        return final_result

    declarations = []
    for group in groups:
        for witness in group.witnesses:
            requested = witness.carrier
            requested.config[group.key] = witness.value
            declarations.append((group, witness, requested))
    outcomes = search.compiler_coverage_outcomes
    assert len(outcomes) == len(declarations)
    base_configs = {generation.canonicalize_flat(deepcopy(row))[1] for row in base}
    expected_additions = []
    seen = set(base_configs)
    owned_keys = {group.key for group in groups}
    for (group, witness, requested), outcome in zip(
        declarations, outcomes, strict=True
    ):
        assert outcome.origin == "declared"
        assert outcome.mechanism == group.mechanism
        assert outcome.requested == requested
        try:
            flat, effective = generation.strict_config_pair(requested)
            _, control = generation.strict_config_pair(witness.carrier)
        except InvalidConfig:
            assert outcome.outcome.startswith("invalid:")
            continue
        assert outcome.effective == effective
        if not same_value(effective.config.get(group.key, group.legacy), witness.value):
            assert outcome.outcome == "mode_not_effective"
            continue
        assert {
            key: value for key, value in control.config.items() if key not in owned_keys
        } == {
            key: value
            for key, value in effective.config.items()
            if key not in owned_keys
        }
        for dependency in group.dependencies:
            assert same_value(control.config[dependency.key], dependency.value)
            assert same_value(effective.config[dependency.key], dependency.value)
        if effective in seen:
            assert outcome.outcome == "already_present"
        else:
            assert outcome.outcome == "added"
            expected_additions.append(flat)
            seen.add(effective)
        assert effective in search._pinned_finalist_configs
    assert result[len(base) :] == expected_additions
    assert len(result) == search.initial_population + len(expected_additions)
    return final_result


def _check_full_seed_coverage(
    search: PatternSearch, population: list[FlatConfig], result: list[FlatConfig]
) -> None:
    """Audit the separate full-effort seed append after declared coverage."""
    outcomes = search.seed_coverage_outcomes
    if not search._full_seed_coverage_enabled():
        assert not outcomes and result == population
        return
    generation = search.config_gen
    spec = generation.config_spec
    requests = [
        (f"compiler_seed:{index}", seed, False)
        for index, seed in enumerate(spec.compiler_seed_configs)
    ] + [(w.mechanism, w.config, True) for w in spec.compiler_seed_witnesses]
    assert len(outcomes) == len(requests)
    seen = set()
    for row in population:
        try:
            seen.add(generation.canonicalize_flat(deepcopy(row))[1])
        except InvalidConfig:
            continue
    additions = []
    for (origin, requested, supplemental), outcome in zip(
        requests, outcomes, strict=True
    ):
        assert outcome.origin == origin and outcome.requested == requested
        carrier = deepcopy(requested)
        if (
            "pid_type" not in generation._flat_fields()
            and carrier.config.get("pid_type") == "flat"
        ):
            carrier.config.pop("pid_type")
        try:
            flat, effective = generation.strict_config_pair(carrier)
        except InvalidConfig:
            assert outcome.status.startswith("invalid:") and outcome.effective is None
            continue
        if supplemental and any(
            isinstance(fragment, EnumFragment)
            and not any(
                same_value(value, choice) for choice in fragment._active_choices()
            )
            for fragment, value in zip(generation.flat_spec, flat, strict=True)
        ):
            assert outcome.status.startswith("invalid:") and outcome.effective is None
            continue
        assert outcome.effective == effective
        if supplemental and any(
            key not in effective.config or effective.config[key] != value
            for key, value in carrier.config.items()
        ):
            assert outcome.status == "overridden"
            continue
        if effective in seen:
            assert outcome.status == "already_present"
        else:
            assert outcome.status == "pending"
            additions.append(flat)
            seen.add(effective)
        assert effective in search._pinned_finalist_configs
    assert result == [*population, *additions]


def _pointwise(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


def _cpu_target() -> ExitStack:
    stack = ExitStack()
    for name, value in (
        ("helion.runtime.kernel.target_device_capability", (10, 0)),
        ("helion._compiler.compile_environment.target_device_capability", (10, 0)),
        ("helion.runtime.get_num_sm", 148),
        ("helion._compat._is_hip", False),
    ):
        stack.enter_context(patch(name, return_value=value))
    return stack


def _target() -> ExitStack:
    stack = _cpu_target()
    stack.enter_context(
        patch_cute_mma_support(
            SimpleNamespace(
                universal=True,
                warp_f16bf16=True,
                warpgroup_f16bf16=True,
                tcgen05_f16bf16=True,
            )
        )
    )
    for name, value in (
        (
            "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_capacity_bytes",
            232448,
        ),
        (
            "helion._hardware.get_hardware_info",
            HardwareInfo(
                device_kind="cuda",
                hardware_name="NVIDIA B200",
                runtime_version="13.0",
                compute_capability="sm100",
            ),
        ),
    ):
        stack.enter_context(patch(name, return_value=value))
    return stack
