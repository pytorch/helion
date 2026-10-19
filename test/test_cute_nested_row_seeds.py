from __future__ import annotations

from unittest.mock import patch

from examples.jagged_layer_norm import jagged_layer_norm_kernel
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.autotuner_heuristics.cute import CuteNestedRowHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteSiblingRowHeuristic
from helion._compiler.cute.loop_nesting import tile_loop_paths
from helion._compiler.cute.thread_budget import tile_loop_thread_count
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("columns", [32, 69, 128, 512])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_nested_jagged_passes_receive_reachable_coherent_seeds(
    columns: int, static_shapes: bool
) -> None:
    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = _cpu_bind(
        kernel,
        (torch.empty((731, columns)), torch.empty(18, dtype=torch.int64), 1e-6),
    )
    host = bound.host_function
    assert host is not None
    ir = host.device_ir
    spec = bound.config_spec
    paths = tile_loop_paths(ir, ir.graphs)
    assert CuteNestedRowHeuristic in get_heuristics("cute")
    seeds = CuteNestedRowHeuristic.get_seed_configs(bound.env, ir)
    assert seeds
    assert all(seed in spec.compiler_seed_configs for seed in seeds)
    for seed in seeds:
        normalized = seed.config.copy()
        spec.normalize(normalized)
        assert normalized["num_threads"] == seed.config["num_threads"]
        with bound.env:
            assert tile_loop_thread_count(bound.env, ir, ir.graphs, seed) <= 256
        for root, feature, reduction in paths:
            assert spec.num_threads.config_get(seed.num_threads, root[0]) == 1
            assert spec.num_threads.config_get(seed.num_threads, reduction[0]) == 1
            assert spec.num_threads.config_get(seed.num_threads, feature[0]) > 1
        with patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ):
            source = bound.to_code(seed)
        assert "def _helion_jagged_layer_norm_kernel" in source


def test_nested_seeds_respect_config_sequence_registration_order() -> None:
    kernel = helion.kernel(jagged_layer_norm_kernel.fn, backend="cute")
    bound = _cpu_bind(
        kernel,
        (torch.empty((731, 128)), torch.empty(18, dtype=torch.int64), 1e-6),
    )
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    original = CuteNestedRowHeuristic.get_seed_configs(bound.env, host.device_ir)
    spec.cute_vector_widths[:] = list(reversed(spec.cute_vector_widths))
    spec.num_threads[:] = list(reversed(spec.num_threads))
    reordered = CuteNestedRowHeuristic.get_seed_configs(bound.env, host.device_ir)
    assert len(original) == len(reordered)
    for first, second in zip(original, reordered, strict=True):
        assert first.block_sizes == second.block_sizes
        assert list(reversed(first.num_threads)) == second.num_threads


def test_nested_seeds_do_not_claim_a_plain_row_kernel() -> None:
    @helion.kernel(backend="cute", autotune_effort="none")
    def row_sum(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
        for row in hl.tile(x.size(0)):
            out[row] = x[row, :].sum(dim=-1)
        return out

    bound = _cpu_bind(row_sum, (torch.empty((17, 128)),))
    host = bound.host_function
    assert host is not None
    assert not CuteNestedRowHeuristic.get_seed_configs(bound.env, host.device_ir)


@pytest.mark.parametrize("static_shapes", [False, True])
def test_flat_jagged_search_reaches_wide_coherent_vector_sweeps(
    static_shapes: bool,
) -> None:
    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn,
        backend="cute",
        static_shapes=static_shapes,
        cute_flatten_nested_reductions=True,
        autotune_effort="none",
    )
    with _mock_cuda_unavailable():
        bound = _cpu_bind(
            kernel, (torch.empty((731, 128)), torch.empty(18, dtype=torch.int64))
        )
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    seeds = CuteSiblingRowHeuristic.get_seed_configs(bound.env, host.device_ir)
    assert seeds and all(seed in spec.compiler_seed_configs for seed in seeds)
    assert any(
        seed.block_sizes == [1, 32768, 32768, 32768]
        and seed.num_threads == [1, 512, 512, 512]
        and seed.config["cute_vector_widths"] == [1, 4, 4, 4]
        for seed in seeds
    )
    for seed in seeds:
        normalized = seed.config.copy()
        spec.normalize(normalized)
        assert normalized["block_sizes"] == seed.block_sizes
        assert normalized["num_threads"] == seed.num_threads
        for item, size in zip(spec.block_sizes, seed.block_sizes, strict=True):
            fragment = item._fragment(spec)
            assert fragment.low <= size <= fragment.high
        with patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ):
            assert "def _helion_jagged_layer_norm_kernel" in bound.to_code(seed)
