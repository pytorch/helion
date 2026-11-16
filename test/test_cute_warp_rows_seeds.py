"""The warp-per-row, rows-per-CTA seeds of ``CuteReductionTileHeuristic``
(``warp_rows_seed_configs``): short rows packed two and four to a CTA with one
warp each, the form that beats quack on batch_softmax's 1024-wide rows where
the one-row seeds spend a 32-thread CTA per row."""

from __future__ import annotations

import pytest
import torch

from test.cute_population_contracts import tcgen05_mma_support

import helion
from helion._compiler.cute.autotune.heuristics.cute import CuteReductionTileHeuristic
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _cpu_only():
    with cpu_render_target(mma_support=tcgen05_mma_support()):
        yield


def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        values = x[tile_m, :]
        x_max = torch.amax(values, dim=1)
        exp_vals = torch.exp(values - x_max[:, None])
        out[tile_m, :] = exp_vals / torch.sum(exp_vals, dim=1)[:, None]
    return out


def _seeds(fn, args):
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="full"
    )
    bound = cpu_bind(kernel, args)
    assert CuteReductionTileHeuristic.is_eligible(
        bound.env, bound.host_function.device_ir
    )
    spec = bound.config_spec
    return (
        bound,
        spec,
        CuteReductionTileHeuristic.warp_rows_seed_configs(
            bound.env, bound.host_function.device_ir
        ),
    )


def test_batched_softmax_gets_two_and_four_rows_per_cta() -> None:
    batch_softmax = import_example("batch_softmax").batch_softmax

    bound, spec, seeds = _seeds(
        batch_softmax.fn, (torch.empty(8, 256, 1024, dtype=torch.float16),)
    )
    assert [s.config["block_sizes"] for s in seeds] == [[1, 2], [1, 4]]
    for seed, rows in zip(seeds, (2, 4), strict=True):
        config = seed.config
        assert config["num_threads"] == [0, rows, 32]
        assert config["reduction_loops"] == [1024]
        assert spec.cute_vector_widths.config_get(config["cute_vector_widths"], 2) == 8
        assert (
            spec.cute_lane_layouts.config_get(config["cute_lane_layouts"], 2)
            == "strided"
        )
        assert config["cute_reduction_reloads"] == ["register_prefetch"]
        normalized = dict(config)
        spec.normalize(normalized)
        assert normalized["block_sizes"] == config["block_sizes"]
        assert normalized["num_threads"] == config["num_threads"]
        assert seed in spec.compiler_seed_configs


def test_canonical_row_softmax_gets_the_seeds_too() -> None:
    _bound, spec, seeds = _seeds(
        softmax_rows, (torch.empty(2048, 2048, dtype=torch.bfloat16),)
    )
    assert [s.config["block_sizes"] for s in seeds] == [[2], [4]]
    assert seeds[0].config["num_threads"] == [2, 32]
    assert seeds[0].config["reduction_loops"] == [2048]
    assert all(seed in spec.compiler_seed_configs for seed in seeds)


def test_long_rows_get_no_warp_rows_seed() -> None:
    _bound, _spec, seeds = _seeds(
        softmax_rows, (torch.empty(64, 16384, dtype=torch.bfloat16),)
    )
    assert seeds == []
