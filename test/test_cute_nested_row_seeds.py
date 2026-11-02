from __future__ import annotations

from unittest.mock import patch

from examples.jagged_layer_norm import jagged_layer_norm_kernel
from examples.jagged_layer_norm import reference_jagged_layer_norm_pytorch
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
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

_SEED_CODEGEN_SHARDS = 4


@pytest.mark.parametrize("columns", [32, 69, 128, 512])
@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("seed_shard", range(_SEED_CODEGEN_SHARDS))
def test_nested_jagged_passes_receive_reachable_coherent_seeds(
    columns: int, static_shapes: bool, seed_shard: int
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
    # The gathers vectorize along the features (four fp32 per 16-byte packet)
    # only when the feature extent is a multiple of the packet (69 is not)
    # and the flattened view's base alignment is provable: a dynamic-shape
    # view carries a symbolic storage offset, which no bound residue covers.
    sunk = [seed for seed in seeds if seed.config.get("cute_vloop_sink")]
    assert bool(sunk) == (columns % 4 == 0 and static_shapes)
    # The largest case generates 111 sources; split that work across test cases
    # so CI's per-test timeout does not terminate a worker mid-codegen.
    for seed in seeds[seed_shard::_SEED_CODEGEN_SHARDS]:
        normalized = seed.config.copy()
        spec.normalize(normalized)
        assert normalized["num_threads"] == seed.config["num_threads"]
        vectorized = seed.config.get("cute_vloop_sink") is True
        with bound.env:
            assert tile_loop_thread_count(bound.env, ir, ir.graphs, seed) <= (
                512 if vectorized else 256
            )
        for root, feature, reduction in paths:
            assert spec.num_threads.config_get(seed.num_threads, root[0]) == 1
            feature_threads = spec.num_threads.config_get(seed.num_threads, feature[0])
            reduction_threads = spec.num_threads.config_get(
                seed.num_threads, reduction[0]
            )
            feature_tile = spec.block_sizes.config_get(seed.block_sizes, feature[0])
            widths = seed.config["cute_vector_widths"]
            if vectorized:
                # V contiguous features per thread, the rest of the block
                # along the jagged rows; registers kept for the rows in flight.
                assert spec.cute_vector_widths.config_get(widths, feature[0]) == 4
                assert spec.cute_vector_widths.config_get(widths, reduction[0]) == 1
                assert feature_threads * 4 == feature_tile
                assert reduction_threads > 1
                assert seed.config["cute_lane_unroll"] in (1, 2, 4, 8)
                assert seed.config["cute_min_blocks_per_mp"] == 1
            else:
                assert reduction_threads == 1
                assert feature_threads > 1
        with patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ):
            source = bound.to_code(seed)
        assert "def _helion_jagged_layer_norm_kernel" in source
        if vectorized:
            # Every pass gathers its rows as predicated 16-byte packets, and
            # the normalized rows are stored as packets.
            assert source.count("_cute_load_v16b_if(x_flat.iterator") >= 3
            assert "_cute_store_u32_vec(out_flat.iterator" in source
            assert "x_flat.layout.stride[0])).load()" not in source


@pytest.mark.parametrize("columns", [32, 69, 128, 512])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_nested_jagged_default_is_the_primary_seed(
    columns: int, static_shapes: bool
) -> None:
    # The generic default threads only the root tile: one thread per jagged
    # row walked every feature and element serially (7.4 ms vs 0.059 ms).
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
    spec = bound.config_spec
    primary = CuteNestedRowHeuristic.get_seed_configs(bound.env, host.device_ir)[0]
    assert spec.compiler_default_config == primary
    default = spec.default_config()
    assert default.block_sizes == primary.block_sizes
    assert default.num_threads == primary.num_threads
    for _, feature, _ in tile_loop_paths(host.device_ir, host.device_ir.graphs):
        assert spec.num_threads.config_get(default.num_threads, feature[0]) > 1
    # On sm_90+ the seeded default launches as a programmatic dependent of
    # the previous kernel: its launch and prologue overlap that kernel's
    # tail (16 rows x 32 features: 2.17 -> 2.05 us per call in a stream of
    # calls), and it waits ahead of its first global access.
    assert default.config["cute_pdl"] is True
    with patch(
        "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
        return_value=232448,
    ):
        source = bound.to_code(default)
    assert "griddepcontrol" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("static_shapes", [False, True])
def test_nested_jagged_default_matches_reference(static_shapes: bool) -> None:
    torch.manual_seed(0)
    lengths = torch.randint(0, 40, (17,))
    lengths[3] = 0
    x_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    x_offsets = x_offsets.to(DEVICE)
    x = torch.randn((int(lengths.sum()), 69), device=DEVICE)
    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    torch.testing.assert_close(
        kernel(x, x_offsets, 1e-6),
        reference_jagged_layer_norm_pytorch(x, x_offsets, 1e-6),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nested_jagged_vectorized_default_matches_reference() -> None:
    # 128 features, static shapes: the primary seed vectorizes the gathers
    # (V-loop sunk, packet loads and stores, lane-folded mean and variance).
    static_shapes = True
    torch.manual_seed(0)
    lengths = torch.randint(0, 300, (23,))
    lengths[3] = 0
    lengths[5] = 1
    x_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    x_offsets = x_offsets.to(DEVICE)
    x = torch.randn((int(lengths.sum()), 128), device=DEVICE)
    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = kernel.bind((x, x_offsets, 1e-6))
    default = bound.config_spec.default_config()
    assert default.config.get("cute_vloop_sink") is True
    torch.testing.assert_close(
        kernel(x, x_offsets, 1e-6),
        reference_jagged_layer_norm_pytorch(x, x_offsets, 1e-6),
        rtol=1e-4,
        atol=1e-4,
    )


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
