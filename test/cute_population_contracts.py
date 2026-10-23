"""Audit the legacy population budget and the separate coverage additions."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from functools import partial
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._hardware import HardwareInfo
from helion._testing import patch_cute_mma_support
from helion.autotuner.compiler_coverage import same_value
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    import helion
    from helion.autotuner.config_generation import FlatConfig
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.config import Config
    from helion.runtime.kernel import BoundKernel


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
    build_base = search._generate_initial_population_base

    def observe_base(**kwargs):
        result = build_base(**kwargs)
        base_calls.append((deepcopy(result), random.getstate()))
        return result

    with patch.object(search, "_generate_initial_population_base", observe_base):
        result = search._generate_initial_population_flat()
    assert len(base_calls) == 1
    base, state = base_calls[0]
    assert len(base) == search.initial_population
    assert result[: len(base)] == base
    assert random.getstate() == state
    generation = search.config_gen
    groups = generation.config_spec.compiler_coverage_groups
    if not groups or not generation.compiler_coverage_enabled:
        assert result == base
        return result

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
    return result


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


def _legacy_order(
    spec: ConfigSpec, seeds: list[helion.Config], parameters: tuple[str, ...]
) -> list[helion.Config]:
    return seeds


def _legacy_registered_seeds(
    bound: BoundKernel, *, require_no_parameters: bool = False
) -> list[helion.Config]:
    assert bound.host_function is not None
    with (
        bound.env,
        patch.object(
            CuteCollectiveMatmulHeuristic,
            "_interleave_bounded_k_seeds",
            side_effect=_legacy_order,
        ) as interleaved,
    ):
        legacy = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    if require_no_parameters:
        assert all(not call.args[2] for call in interleaved.call_args_list)
    # Other registered heuristics retain their original positions. Replace
    # only this heuristic's permutation when reproducing the old population.
    members = set(legacy)
    selected = [
        seed for seed in bound.config_spec.compiler_seed_configs if seed in members
    ]
    assert len(selected) == len(legacy)
    iterator = iter(legacy)
    return [
        next(iterator) if seed in members else seed
        for seed in bound.config_spec.compiler_seed_configs
    ]


def _initial_population(
    bound: BoundKernel,
    args: tuple[object, ...],
    *,
    seeds: list[helion.Config] | None = None,
    count: int | None = None,
    user_seeds: list[helion.Config] | None = None,
    random_seed: int | None = None,
    assert_canonical: bool = True,
) -> list[helion.Config]:
    state = random.getstate() if random_seed is not None else None
    try:
        if random_seed is not None:
            random.seed(random_seed)
        profile = get_effort_profile("full").pattern_search
        assert profile is not None
        with (
            bound.env,
            patch.object(
                bound.config_spec,
                "compiler_seed_configs",
                bound.config_spec.compiler_seed_configs if seeds is None else seeds,
            ),
            patch.object(bound.settings, "autotune_seed_configs", user_seeds),
        ):
            search = PatternSearch(
                bound,
                args,
                initial_population=profile.initial_population
                if count is None
                else count,
                initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
            )
            flat = checked_initial_population(search)
            result = [search.config_gen.unflatten(value) for value in flat]
            if assert_canonical:
                assert [search.config_gen.flatten(config) for config in result] == flat
            assert result[0] == search.config_gen.unflatten(
                search.config_gen.default_flat()
            )
            assert len(result) == len(set(result))
            assert all(
                search.config_gen.unflatten(search.config_gen.flatten(config)) == config
                for config in result
            )
            return result
    finally:
        if state is not None:
            random.setstate(state)


_seeded_population = partial(
    _initial_population, random_seed=46291, assert_canonical=False
)
