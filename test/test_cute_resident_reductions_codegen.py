from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from examples.layer_norm import layer_norm_bwd
from examples.rms_norm import rms_norm_bwd
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.autotuner_heuristics.cute_resident_reductions import (
    CuteResidentReductionHeuristic,
)
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.resident_reductions import proven_resident_tensor_alignments
from helion._compiler.device_function import DeviceFunction
from helion._compiler.device_function import TensorArg
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration


def _kernel_and_arguments(
    kind: str, dtype: torch.dtype, *, static_shapes: bool = False
) -> tuple[helion.Kernel, tuple[object, ...]]:
    x = torch.empty((64, 4096), dtype=dtype)
    dy = torch.empty_like(x)
    weight = torch.empty(4096, dtype=dtype)
    if kind == "layer_norm":
        function = layer_norm_bwd.fn
        args = (dy, x, torch.empty(64), torch.empty(64), weight, True)
    else:
        function = rms_norm_bwd.fn
        args = (dy, x, weight, torch.empty((64, 1)))
    return helion.kernel(
        function,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ), args


@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
@skipUnlessBackends(["cute"])
def test_original_normalization_seeds_survive_search_and_emit_resident_pipeline(
    kind: str, dtype: torch.dtype, static_shapes: bool
) -> None:
    kernel, args = _kernel_and_arguments(kind, dtype, static_shapes=static_shapes)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(kernel, args)
        host = bound.host_function
        assert host is not None
        spec = bound.config_spec
        assert CuteResidentReductionHeuristic in get_heuristics("cute")
        seeds = CuteResidentReductionHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        assert seeds and all(seed in spec.compiler_seed_configs for seed in seeds)
        assert {
            (
                tuple(seed.block_sizes),
                max(seed.num_threads),
                seed.config["cute_reduction_group_rows"],
                seed.config["cute_reduction_schedule"],
            )
            for seed in seeds
        } == {
            ((32, 1), 256, 2, "pipelined"),
            ((32, 1), 256, 1, "resident"),
            ((32, 1), 128, 1, "pipelined"),
            ((16, 1), 512, 2, "pipelined"),
            ((32, 1), 512, 4, "pipelined"),
        }
        generation = ConfigGeneration(spec)
        surviving = [config for _flat, config in generation.seed_flat_config_pairs()]
        for seed in seeds:
            normalized = spec.normalized_config(seed)
            flat, roundtrip = generation.canonicalize_flat(generation.flatten(seed))
            assert roundtrip in surviving
            assert generation.unflatten(flat) == roundtrip
            for key in (
                "block_sizes",
                "num_threads",
                "cute_vector_widths",
                "cute_reduction_schedule",
                "cute_reduction_group_rows",
                "cute_cluster_n",
                "cute_lane_layouts",
                "cute_reduction_reloads",
            ):
                assert roundtrip.config[key] == normalized.config[key]
            assert roundtrip.num_threads == seed.num_threads
            code = bound.to_code(roundtrip)
            assert "resident_state" in code
            assert "_cute_resident_sums" in code
            assert "_helion_lane_reduce" not in code
            threads = max(seed.num_threads)
            assert f"block=({threads}, 1, 1)" in code
            if seed.config["cute_reduction_schedule"] == "pipelined":
                assert "_cute_resident_copy_async" in code
                assert "cute.arch.cp_async_wait_group(1)" in code
                assert "cute.arch.cp_async_wait_group(0)" in code
            else:
                assert "_cute_resident_load_vector" in code
                assert "_cute_resident_copy_async" not in code


@skipUnlessBackends(["cute"])
def test_resident_knobs_reject_unproved_config_values() -> None:
    kernel, args = _kernel_and_arguments("layer_norm", torch.bfloat16)
    bound = _cpu_bind(kernel, args)
    host = bound.host_function
    assert host is not None
    seeds = CuteResidentReductionHeuristic.get_seed_configs(bound.env, host.device_ir)
    assert seeds
    for values in (
        {"cute_reduction_schedule": "unknown"},
        {"cute_reduction_group_rows": True},
        {"cute_reduction_group_rows": 5},
    ):
        with pytest.raises(helion.exc.InvalidConfig):
            bound.config_spec.normalized_config(
                helion.Config.from_dict(seeds[0].config | values)
            )


@pytest.mark.parametrize("offset,alignment", [(0, 16), (1, 2), (2, 4), (4, 8), (8, 16)])
def test_fresh_allocation_view_alignment_accounts_for_storage_offset(
    offset: int, alignment: int
) -> None:
    tensor = torch.empty(32, dtype=torch.float16)[offset:]
    function = cast(
        "DeviceFunction", SimpleNamespace(arguments=[TensorArg("view", tensor, None)])
    )
    env = SimpleNamespace(
        input_sources={}, compiler_fact_specialization_facts=frozenset()
    )
    with patch.object(CompileEnvironment, "current", return_value=env):
        assert proven_resident_tensor_alignments(function) == {"view": alignment}
