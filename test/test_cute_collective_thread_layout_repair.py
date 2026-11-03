"""The collective MMA's thread layout is derived from the block sizes in config generation."""

from __future__ import annotations

import random

import pytest

from test.test_cute_collective_tf32 import _config
from test.test_cute_collective_tf32 import _jagged_inputs
from test.test_cute_collective_tf32 import _kernel

import helion
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@pytest.mark.parametrize(
    "bm,bn,threads",
    [(32, 32, [0, 4, 32, 1]), (32, 64, [0, 2, 64, 1]), (64, 64, [0, 2, 64, 1])],
)
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
def test_enabled_collective_gets_the_admitted_thread_layout(
    bm: int, bn: int, threads: list[int], compute: str
) -> None:
    bound = cpu_bind(_kernel(), _jagged_inputs(batch=3, rows=137, k=64, n=64))
    config = _config(bm, bn, 32, compute=compute, copy="async_cached")
    # A thread layout the lowering declines (the lane epilogue's default).
    config.config["num_threads"] = [0, 4, 32, 1] if bn != 32 else [0, 8, 16, 1]
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        repaired = generation.unflatten(generation.flatten(config))
    assert repaired.config["num_threads"] == threads
    assert repaired.config["block_sizes"] == [1, bm, bn, 32]
    if compute == "warp":
        assert "cute.gemm(" in bound.to_code(repaired)


def test_disabled_collective_keeps_its_thread_layout() -> None:
    bound = cpu_bind(_kernel(), _jagged_inputs(batch=3, rows=137, k=64, n=64))
    config = _config(32, 64, 32, compute="warp", copy="async_cached")
    config.config["cute_collective_mma"] = False
    config.config["num_threads"] = [0, 4, 32, 1]
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        repaired = generation.unflatten(generation.flatten(config))
    assert repaired.config["num_threads"] == [0, 4, 32, 1]
    assert repaired.config.get("cute_collective_mma", False) is False


def test_perturbed_collective_layout_is_repaired_and_engages() -> None:
    bound = cpu_bind(_kernel(), _jagged_inputs(batch=3, rows=137, k=64, n=64))
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        keys = generation._key_to_flat_indices
        thread_indices = keys["num_threads"][0]
        width_indices = keys["cute_vector_widths"][0]
        flat = generation.flatten(
            _config(32, 64, 32, compute="warp", copy="async_cached")
        )
        # Threads and vector widths the lowering declines; everything else is
        # the known-good collective config.
        flat[thread_indices[1]], flat[thread_indices[2]], flat[thread_indices[3]] = (
            8,
            16,
            2,
        )
        flat[width_indices[2]] = 2
        repaired = generation.unflatten(flat)
    assert repaired.config["num_threads"][1:] == [2, 64, 1]
    assert repaired.config["cute_vector_widths"][1:] == [1, 1, 1]
    assert "cute.gemm(" in bound.to_code(repaired)


def test_random_collective_configs_never_fall_back() -> None:
    # The sampler draws from ``random``: a fixed seed makes the 24 configs
    # (and whatever they render) the same from run to run, restored so the
    # process-wide RNG is left as found.  Seed 3 renders 7 of the 24 (the
    # most of seeds 0-39; the rest are refusals, which assert nothing).
    state = random.getstate()
    try:
        random.seed(3)
        bound = cpu_bind(_kernel(), _jagged_inputs(batch=3, rows=137, k=64, n=64))
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            keys = generation._key_to_flat_indices
            mma_index = keys["cute_collective_mma"][0][0]
            compute_index = keys["cute_collective_compute"][0][0]
            block_indices = keys["block_sizes"][0]
            configs = []
            for sample in range(24):
                flat = generation.random_flat()
                # Enable the warp collective on an admissible tile; every
                # other slot (threads, vector widths, copies, stages, ...)
                # stays random.
                flat[mma_index] = True
                flat[compute_index] = "warp"
                flat[block_indices[1]] = (32, 64, 128)[sample % 3]
                flat[block_indices[2]] = (32, 64)[sample % 2]
                configs.append(generation.unflatten(flat))
    finally:
        random.setstate(state)
    rendered = 0
    for config in configs:
        _b, bm, bn, _bk = config.config["block_sizes"]
        assert config.config["cute_collective_mma"] is True
        assert config.config["num_threads"][1:] == [128 // bn, bn, 1], config
        assert config.config["cute_vector_widths"][1:] == [1, 1, 1], config
        # Other random knobs may still make the lowering refuse the kernel
        # outright (a failed config for the autotuner); whatever lowers must
        # lower to the collective MMA, never silently to the lane loops.
        try:
            source = bound.to_code(config)
        except helion.exc.BackendUnsupported:
            continue
        assert "cute.gemm(" in source, config
        rendered += 1
    # A config-space change shifts the RNG stream; the test proves nothing
    # when every sample is a refusal.
    assert rendered > 0, "seed 3 rendered none of the 24 configs (7 expected)"
