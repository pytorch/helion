from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from test.test_cute_philox_stream import cpu_only  # noqa: F401

import helion
from helion._compiler.autotuner_heuristics.cute import CutePointwiseVecHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
import helion.language as hl

pytestmark = [pytest.mark.usefixtures("cpu_only"), skipUnlessBackends(["cute"])]


def _uniform_pointwise(x: torch.Tensor, seed: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * hl.rand([tile], seed=seed)
    return out


def _ordinary_pointwise(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * 2
    return out


def _bind(function, args, **settings):
    return helion.kernel(
        function,
        backend="cute",
        static_shapes=False,
        **settings,
    )._bind_isolated(args)


@pytest.mark.parametrize(
    ("length", "dtype"),
    [(1024, torch.float32), (4096, torch.float32), (8192, torch.float16)],
)
def test_packet_seed_prefix_and_full_population(length, dtype):
    bound = _bind(
        _uniform_pointwise,
        (torch.empty(length, dtype=dtype), -1),
        cute_rng_stream="philox4",
    )
    spec = bound.config_spec
    assert spec.cute_rng_packet_enabled
    with bound.env:
        with patch.object(spec, "cute_rng_packet_enabled", False):
            original = CutePointwiseVecHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        expanded = CutePointwiseVecHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert original and expanded
        assert expanded[: len(original)] == original
        assert [config.config for config in expanded[len(original) :]] == [
            deepcopy(config.config) | {"cute_rng_packet": True} for config in original
        ]
        assert all("cute_rng_packet" not in config.config for config in original)
        generation = ConfigGeneration(spec)
        messages = []
        seeds = generation.seed_flat_config_pairs(messages.append)
        assert not messages
        expected = list(
            dict.fromkeys(
                generation.canonicalize_flat(generation.flatten(config))[1]
                for config in expanded
            )
        )
        assert [config for _, config in seeds] == expected
        default = generation.default_flat()
        assert generation.unflatten(default).config["cute_rng_packet"] is False
        full = get_effort_profile("full").pattern_search
        assert full is not None
        assert 1 + len(seeds) < full.initial_population
        population = generation.random_population_flat(full.initial_population)
        assert len(population) == full.initial_population
        assert population[0] == default
        assert population[1 : len(seeds) + 1] == [flat for flat, _ in seeds]
        assert any(config.config["cute_rng_packet"] for _, config in seeds)
        # The stored source family's prefetch/bounds alternatives survive the
        # seed expansion; no packet candidate silently changes those controls.
        false = [
            config.config for _, config in seeds if not config.config["cute_rng_packet"]
        ]
        true = [
            config.config for _, config in seeds if config.config["cute_rng_packet"]
        ]
        assert true == [config | {"cute_rng_packet": True} for config in false]
        before = deepcopy(spec.compiler_seed_configs)
        assert (
            CutePointwiseVecHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            == expanded
        )
        assert spec.compiler_seed_configs == before


@pytest.mark.parametrize("stream", ["word0", "philox4"])
def test_nonrandom_pointwise_seeds_have_no_rng_choice(stream):
    bound = _bind(_ordinary_pointwise, (torch.empty(4096),), cute_rng_stream=stream)
    assert not bound.config_spec.cute_rng_packet_enabled
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        seeds = generation.seed_flat_config_pairs()
        assert seeds
        assert all("cute_rng_packet" not in config.config for _, config in seeds)
        assert "cute_rng_packet" not in generation._key_to_flat_indices


def test_word0_rng_keeps_original_seed_domain():
    bound = _bind(_uniform_pointwise, (torch.empty(4096), 123), cute_rng_stream="word0")
    assert bound.settings.cute_rng_stream == "word0"
    assert not bound.config_spec.cute_rng_packet_enabled
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        assert "cute_rng_packet" not in generation._key_to_flat_indices
        assert all(
            "cute_rng_packet" not in config.config
            for _, config in generation.seed_flat_config_pairs()
        )


def test_heuristics_disabled_retains_mutable_packet_choice_without_seeds():
    bound = _bind(
        _uniform_pointwise,
        (torch.empty(4096), 123),
        cute_rng_stream="philox4",
        disable_autotuner_heuristics=True,
    )
    assert bound.config_spec.cute_rng_packet_enabled
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        assert generation.seed_flat_config_pairs() == []
        indices, _ = generation._key_to_flat_indices["cute_rng_packet"]
        (index,) = indices
        assert generation.flat_spec[index].search_values() == [False, True]
        assert (
            generation.unflatten(generation.default_flat()).config["cute_rng_packet"]
            is False
        )


def test_vector_ineligible_extent_does_not_invent_packet_geometry():
    bound = _bind(
        _uniform_pointwise,
        (torch.empty(257), 123),
        cute_rng_stream="philox4",
    )
    with bound.env:
        assert (
            CutePointwiseVecHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            is None
        )
        assert not bound.config_spec.compiler_seed_configs
