from __future__ import annotations

from copy import deepcopy
import random
from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _cpu_bind
from ._cute_binding import _forbid_cuda_work
from ._cute_binding import _mock_cuda_unavailable
from .cute_population_contracts import _target
from .test_compiler_coverage import DOMAINS
from .test_compiler_coverage import make_search
from .test_compiler_coverage import make_spec
from .test_cute_grouped_ab_k import _unsupported_k_loop
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.exc import InvalidConfig
from helion.runtime.config import Config


def _effective_registry(generation):
    """Enumerate raw declarations independently of the production pair helper."""
    result = []
    seen = set()
    for declared in generation.config_spec.compiler_seed_configs:
        try:
            flat = generation.flatten(deepcopy(declared))
            effective = generation.unflatten(flat)
            flat = generation.flatten(effective)
        except (InvalidConfig, ValueError, TypeError, KeyError, AssertionError):
            continue
        if effective not in seen:
            seen.add(effective)
            result.append((flat, effective))
    return result


@pytest.mark.parametrize("cls", [PatternSearch, LFBOPatternSearch, LFBOTreeSearch])
@pytest.mark.parametrize(
    "strategy,pad",
    [
        (InitialPopulationStrategy.FROM_RANDOM, True),
        (InitialPopulationStrategy.FROM_BEST_AVAILABLE, False),
        (InitialPopulationStrategy.FROM_BEST_AVAILABLE, True),
    ],
)
@pytest.mark.parametrize("disabled", [False, True])
def test_requested_seed_overflow_preserves_full_old_prefix_rng_and_cache_queries(
    cls, strategy, pad, disabled
) -> None:
    previous_spec = make_spec(many_seeds=True)
    requested_spec = make_spec(many_seeds=True)
    requested_spec.compiler_seed_overflow_requested = True
    user_seed = (Config(block_sizes=[64], old_choice=240),)
    previous = make_search(
        previous_spec,
        cls=cls,
        strategy=strategy,
        pad=pad,
        count=8,
        seeds=user_seed,
        disabled=disabled,
    )
    requested = make_search(
        requested_spec,
        cls=cls,
        strategy=strategy,
        pad=pad,
        count=8,
        seeds=user_seed,
        disabled=disabled,
    )
    registry = deepcopy(requested_spec.compiler_seed_configs)
    saved_rng = random.getstate()
    try:
        random.seed(1234)
        old = previous._generate_initial_population_flat()
        after_old = random.getstate()
        random.seed(1234)
        full = requested._generate_initial_population_flat()
        assert random.getstate() == after_old
    finally:
        random.setstate(saved_rng)
    assert full[: len(old)] == old
    assert requested.initial_population == previous.initial_population == 8
    assert (
        requested._find_similar_cached_configs.call_count
        == previous._find_similar_cached_configs.call_count
    )
    assert requested_spec.compiler_seed_configs == registry
    seen = {requested.config_gen.unflatten(deepcopy(row)) for row in old}
    additions = []
    for flat, effective in _effective_registry(requested.config_gen):
        assert effective in requested._pinned_finalist_configs
        if effective not in seen:
            additions.append(flat)
            seen.add(effective)
    assert full[len(old) :] == additions
    assert len({requested.config_gen.unflatten(row) for row in full}) == len(full)
    assert getattr(requested, "compiler_coverage_outcomes", []) == getattr(
        previous, "compiler_coverage_outcomes", []
    )
    if disabled:
        assert not getattr(requested, "compiler_coverage_outcomes", [])
        for row in full:
            effective = requested.config_gen.unflatten(row)
            assert all(
                effective.get(key, domain[0]) == domain[0]
                for key, domain in DOMAINS.items()
            )


def test_seed_overflow_deduplicates_after_scalar_overrides_and_pins_effective_values() -> (
    None
):
    spec = make_spec(expanded=False, many_seeds=True)
    spec.compiler_seed_overflow_requested = True
    spec.compiler_seed_configs += [deepcopy(spec.compiler_seed_configs[-1])]
    search = make_search(spec, count=3, overrides={"old_choice": 77})
    with patch.object(
        search.config_gen, "random_flat", side_effect=AssertionError("overflow sampled")
    ):
        population = [search.config_gen.default_flat()]
        result = search._append_requested_compiler_seeds(population)
    assert result[:1] == population
    effective = [search.config_gen.unflatten(row) for row in result]
    assert len(effective) == len(set(effective))
    assert all(config["old_choice"] == 77 for config in effective)
    assert all(config["old_choice"] == 77 for config in search._pinned_finalist_configs)
    assert {config for _, config in _effective_registry(search.config_gen)} <= set(
        effective
    )


@pytest.mark.parametrize(
    "empty,requested", [(False, False), (True, False), (True, True)]
)
def test_nonrequesting_and_empty_seed_registries_keep_identity(
    empty, requested
) -> None:
    spec = make_spec(expanded=False)
    spec.compiler_seed_overflow_requested = requested
    if empty:
        spec.compiler_seed_configs = []
    search = make_search(spec, count=8)
    population = [search.config_gen.default_flat()]
    with patch.object(
        search.config_gen,
        "seed_flat_config_pairs",
        side_effect=AssertionError("seed transfer"),
    ):
        assert search._append_requested_compiler_seeds(population) is population


@pytest.mark.parametrize("count", [-1, 0])
def test_nonpositive_nominal_population_does_not_request_overflow(count) -> None:
    spec = make_spec(many_seeds=True)
    spec.compiler_seed_overflow_requested = True
    search = make_search(spec, count=count)
    population = []
    with patch.object(
        search.config_gen,
        "seed_flat_config_pairs",
        side_effect=AssertionError("seed transfer"),
    ):
        assert search._append_requested_compiler_seeds(population) is population


def test_overflow_cache_identity_changes_only_for_an_active_request() -> None:
    spec = make_spec(expanded=False, many_seeds=True)
    old = spec.cache_fingerprint_hash()
    projected = spec.projected_cache_fingerprint_hash()
    spec.compiler_seed_overflow_requested = True
    assert spec.cache_fingerprint_hash() != old
    assert spec.projected_cache_fingerprint_hash() == projected
    spec.compiler_seed_overflow_requested = False
    assert spec.cache_fingerprint_hash() == old
    spec.compiler_seed_configs = []
    empty = spec.cache_fingerprint_hash()
    spec.compiler_seed_overflow_requested = True
    assert spec.cache_fingerprint_hash() == empty


def test_impure_grouped_worklist_does_not_request_overflow() -> None:
    args = (
        torch.empty((512, 256), dtype=torch.bfloat16),
        torch.empty((4, 512, 256), dtype=torch.bfloat16),
        torch.tensor([0, 128, 256, 384, 512], dtype=torch.int32),
        1,
    )
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_unsupported_k_loop, args)
    assert not bound.config_spec.compiler_seed_overflow_requested


def test_malformed_existing_row_is_preserved_without_blocking_valid_overflow() -> None:
    spec = make_spec(expanded=False)
    spec.compiler_seed_overflow_requested = True
    search = make_search(spec, count=1)
    malformed = search.config_gen.default_flat()
    malformed[0] = "invalid-coordinate"
    before = deepcopy(malformed)
    result = search._append_requested_compiler_seeds([malformed])
    assert result[0] == before
    assert malformed == before
    effective = {search.config_gen.unflatten(row) for row in result[1:]}
    assert {config for _, config in _effective_registry(search.config_gen)} <= effective
