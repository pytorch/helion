from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_computed_sweep_cache_cuda import _check_cache
from .test_cute_fuser_occurrences import _args
from .test_cute_fuser_occurrences import _state_reduction
import helion._compiler.autotuner_heuristics as heuristics
from helion._compiler.autotuner_heuristics.cute import (
    CutePersistentSubwarpRowsHeuristic,
)
from helion._compiler.autotuner_heuristics.cute_computed_sweep import (
    append_vector_computed_sweep_seed,
)
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vector_seed_preserves_complete_old_pool_and_default(
    dtype: torch.dtype,
) -> None:
    args = _args(dtype, 32, 128)
    with _cpu_codegen():
        with patch.object(
            heuristics,
            "append_vector_computed_sweep_seed",
            side_effect=lambda env, ir, seeds: seeds,
        ):
            old = _state_reduction._bind_isolated(args)
        current = _state_reduction._bind_isolated(args)
        before = old.config_spec.compiler_seed_configs
        after = current.config_spec.compiler_seed_configs
        assert after[:-1] == before
        assert len(after) == len(before) + 1
        assert any(seed["cute_reduction_reloads"] == ["computed"] for seed in before)
        assert (
            old.config_spec.compiler_default_config
            == current.config_spec.compiler_default_config
        )
        assert current.host_function is not None
        spec = current.config_spec
        plan = CutePersistentSubwarpRowsHeuristic._plan(
            current.env, current.host_function.device_ir
        )
        assert plan is not None
        row, reduction, threads = plan
        assert threads == 16
        # This real graph's reduction ID is not its vector-sequence position.
        assert reduction != spec.cute_vector_widths.valid_block_ids().index(reduction)
        seed = after[-1]
        vectors = cast("list[int]", seed["cute_vector_widths"])
        assert spec.cute_vector_widths.config_get(vectors, reduction, 1) == 8
        for block in spec.cute_vector_widths.valid_block_ids():
            if block != reduction:
                assert spec.cute_vector_widths.config_get(vectors, block, 1) == 1
        thread_counts = cast("list[int]", seed["num_threads"])
        assert spec.num_threads.config_get(thread_counts, row, 0) == 16
        assert spec.num_threads.config_get(thread_counts, reduction, 0) == 16
        assert seed["block_sizes"] == [16]
        assert seed["cute_reduction_reloads"] == ["computed"]
        with current.env:
            generation = ConfigGeneration(spec)
            flat = generation.flatten(seed)
            normalized = generation.unflatten(flat)
            assert generation.flatten(normalized) == flat
        source = current.to_code(normalized)
        control = current.to_code(
            spec.normalized_config(
                normalized.config | {"cute_reduction_reloads": ["gmem"]}
            )
        )
        _check_cache(source, control, vectorized=True)


@pytest.mark.parametrize(
    "dtype,extent",
    [
        (torch.bfloat16, 64),
        (torch.bfloat16, 117),
        (torch.bfloat16, 256),
        (torch.float32, 128),
    ],
)
def test_vector_seed_requires_existing_complete_dtype_sized_coverage(
    dtype: torch.dtype, extent: int
) -> None:
    with _cpu_codegen():
        bound = _state_reduction._bind_isolated(_args(dtype, 32, extent))
        assert bound.host_function is not None
        original = list(bound.config_spec.compiler_seed_configs)
        with bound.env:
            result = append_vector_computed_sweep_seed(
                bound.env, bound.host_function.device_ir, original
            )
        assert result is original
        assert (
            "cute_vector_computed_sweep_cache"
            not in bound.config_spec.autotuner_heuristics
        )


@pytest.mark.parametrize(
    "override,accepted",
    [
        ({"cute_reduction_reloads": ["computed"]}, True),
        ({"cute_reduction_reloads": ["auto"]}, False),
        ({"cute_reduction_reloads": ["register"]}, False),
        ({"cute_reduction_reloads": ["gmem"]}, False),
        ({"num_threads": [16, 16]}, True),
        ({"num_threads": [1, 32]}, False),
        ({"cute_vector_widths": [8, 1, 1, 1]}, True),
        ({"cute_vector_widths": [4, 1, 1, 1]}, False),
        ({"pid_type": "flat"}, True),
        ({"pid_type": "xyz"}, False),
    ],
)
def test_vector_seed_preserves_explicit_overrides(
    override: dict[str, object], accepted: bool
) -> None:
    with _cpu_codegen():
        bound = _state_reduction._bind_isolated(_args(torch.bfloat16, 32, 128))
        assert bound.host_function is not None
        with (
            bound.env,
            patch.object(bound.settings, "autotune_config_overrides", override),
        ):
            result = append_vector_computed_sweep_seed(
                bound.env, bound.host_function.device_ir, []
            )
        assert bool(result) is accepted


def test_vector_seed_deduplicates_and_invalid_normalization_keeps_pool() -> None:
    with _cpu_codegen():
        bound = _state_reduction._bind_isolated(_args(torch.bfloat16, 32, 128))
        assert bound.host_function is not None
        ir = bound.host_function.device_ir
        original = list(bound.config_spec.compiler_seed_configs)
        with bound.env:
            assert (
                append_vector_computed_sweep_seed(bound.env, ir, original) is original
            )
            with patch.object(
                bound.config_spec,
                "normalized_config",
                side_effect=InvalidConfig("unsupported candidate"),
            ):
                assert (
                    append_vector_computed_sweep_seed(bound.env, ir, original)
                    is original
                )
