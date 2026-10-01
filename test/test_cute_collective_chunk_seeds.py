from __future__ import annotations

import operator

import pytest
import torch

from test._cute_binding import _cpu_bind

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _partitioned_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    knob = hl.register_tunable("partitions", PowerOfTwoFragment(1, 256))
    chunk = helion.next_power_of_2(helion.cdiv(k, knob))
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for row, column, outer in hl.tile([m, n, k], block_size=[None, None, chunk]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(outer.begin, outer.end):
            acc = torch.addmm(acc, a[row, inner], b[inner, column])
        hl.atomic_add(out, [row, column], acc)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _direct_chunk_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    chunk = hl.register_tunable("tile_extent", PowerOfTwoFragment(1, 256))
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for row, column, outer in hl.tile([m, n, k], block_size=[None, None, chunk]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(outer.begin, outer.end):
            acc = torch.addmm(acc, a[row, inner], b[inner, column])
        hl.atomic_add(out, [row, column], acc)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_with_unrelated_knob(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    gain = hl.register_tunable("gain", PowerOfTwoFragment(1, 256))
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile([m, n]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner] * gain, b[inner, column])
        out[row, column] = acc.to(out.dtype)
    return out


@pytest.mark.parametrize("direct_chunk", [False, True])
@pytest.mark.parametrize("k", [70, 16384])
def test_config_owned_chunks_keep_collective_geometry_through_seed_transfer(
    direct_chunk: bool, k: int
) -> None:
    inputs = (
        torch.empty((64, k), dtype=torch.float16),
        torch.empty((k, 128), dtype=torch.float16),
    )
    kernel = _direct_chunk_matmul if direct_chunk else _partitioned_matmul
    parameter = "tile_extent" if direct_chunk else "partitions"
    bound = _cpu_bind(kernel, inputs)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        transferred = [config for _flat, config in generation.seed_flat_config_pairs()]
    choices = [
        seed
        for seed in transferred
        if seed.get("cute_collective_compute") == "tcgen05"
        and seed.get("cute_collective_copy") == "async_cached"
        and seed[parameter] > 1
    ]
    assert choices
    for seed in choices:
        chunk = (
            seed[parameter]
            if direct_chunk
            else helion.next_power_of_2(helion.cdiv(k, seed[parameter]))
        )
        assert chunk >= seed.block_sizes[2]
        columns = seed.block_sizes[1]
        assert seed.num_threads == [128 // columns, columns, 1]
    chosen = max(choices, key=operator.itemgetter(parameter))
    code = bound.to_code(chosen)
    assert code.count("tcgen05.CtaGroup.ONE") == 1
    assert code.count("cute.arch.atomic_add(") == 1
    if k == 16384 and not direct_chunk:
        assert 64 in {seed[parameter] for seed in choices}
        assert any(seed.block_sizes[2] == 128 for seed in choices)


def test_unrelated_user_parameter_does_not_multiply_collective_seeds() -> None:
    inputs = (
        torch.empty((64, 256), dtype=torch.float16),
        torch.empty((256, 128), dtype=torch.float16),
    )
    bound = _cpu_bind(_matmul_with_unrelated_knob, inputs)
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds
    assert all("gain" not in seed.config for seed in seeds)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("direct_chunk", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_transferred_chunk_seed_tails_and_mutating_graphs(
    direct_chunk: bool, dtype: torch.dtype
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(717)
    a = (torch.randn((97, 536), device=CUDA_DEVICE, dtype=dtype) * 0.025)[:, :530]
    b = (torch.randn((530, 80), device=CUDA_DEVICE, dtype=dtype) * 0.025)[:, :71]
    kernel = _direct_chunk_matmul if direct_chunk else _partitioned_matmul
    parameter = "tile_extent" if direct_chunk else "partitions"
    bound = kernel._bind_isolated((a, b))
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        choices = [
            config
            for _flat, config in generation.seed_flat_config_pairs()
            if config.get("cute_collective_compute") == "tcgen05"
            and config.get("cute_collective_copy") == "async_cached"
            and config[parameter] > 1
        ]
    assert choices
    chosen = max(choices, key=operator.itemgetter(parameter))
    assert "tcgen05.CtaGroup.ONE" in bound.to_code(chosen)
    bound.set_config(chosen)

    def run() -> torch.Tensor:
        return bound(a, b)

    def expected() -> torch.Tensor:
        return (a.float() @ b.float()).to(dtype)

    torch.testing.assert_close(run(), expected(), atol=2e-3, rtol=1e-2)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        a.uniform_(-0.05, 0.05)
        b.uniform_(-0.05, 0.05)
        torch.testing.assert_close(run(), expected(), atol=2e-3, rtol=1e-2)
        torch.testing.assert_close(replay(), expected(), atol=2e-3, rtol=1e-2)
