from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import import_example
from helion._testing.nvfp4_gemv_args import nvfp4_gemv_args
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig
import helion.language as hl

_examples_nvfp4_gemv = import_example("nvfp4_gemv")
_nvfp4_gemv_fp4in_body = _examples_nvfp4_gemv._nvfp4_gemv_fp4in_body


def _config(groups: int, threads: int, pack: int, **extra: object) -> helion.Config:
    return helion.Config(
        block_sizes=[1, groups],
        num_threads=[1, threads],
        cute_vector_widths=[1, 1],
        cute_lane_layouts=["blocked", "blocked"],
        cute_tiles_per_block=pack,
        **extra,
    )


def test_pack_counts_and_knob_rules() -> None:
    """The one-element grid axis of the gemv is packable; the knob needs a
    flat pid type, no cluster split and a divisor of the tile count."""
    args, _ = nvfp4_gemv_args(2048, 512)
    kernel = helion.kernel(_nvfp4_gemv_fp4in_body, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    spec = bound.config_spec
    assert spec.cute_tile_pack_counts == {0: 2048}
    assert spec.default_config().config.get("cute_tiles_per_block") == 1
    generation = ConfigGeneration(spec)
    config = _config(64, 32, 8)
    spec.normalize(config)
    restored = generation.unflatten(generation.flatten(config))
    assert restored.config["cute_tiles_per_block"] == 8
    with pytest.raises(InvalidConfig):
        spec.normalize(_config(64, 32, 8, cute_cluster_n=2))
    with pytest.raises(InvalidConfig):
        spec.normalize(_config(64, 32, 3))
    repaired = _config(64, 32, 8, cute_cluster_n=2)
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired.config["cute_tiles_per_block"] == 1


@pytest.mark.parametrize("pack", [2, 4, 8])
def test_packed_tiles_share_a_cta(pack: int) -> None:
    """``pack`` rows per CTA: the pid carries the thread sub-group (a thread
    axis of extent ``pack`` ahead of the body's own axes), the grid shrinks by
    ``pack``, the K reduction sees ``pack`` interleaved siblings, and the result
    is exact."""
    rows, kbytes = 256, 128
    args, expected = nvfp4_gemv_args(rows, kbytes)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body,
        backend="cute",
        static_shapes=True,
        config=_config(16, 16, pack),
    )
    bound = kernel.bind(args)
    source = bound.to_code(_config(16, 16, pack))
    assert (
        f"pid_0 = cutlass.Int32(cute.arch.block_idx()[0]) * {pack}"
        " + cutlass.Int32(cute.arch.thread_idx()[0])"
    ) in source
    assert re.search(
        rf"_launcher\(.*\({rows // pack},\).*block=\({pack}, 16, 1\)", source
    )
    assert f"pre={pack}," in source
    kernel(*args)
    torch.testing.assert_close(args[4].float(), expected.float(), rtol=0, atol=0)


def test_unpacked_config_is_unchanged() -> None:
    rows, kbytes = 256, 128
    args, expected = nvfp4_gemv_args(rows, kbytes)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body,
        backend="cute",
        static_shapes=True,
        config=_config(16, 16, 1),
    )
    source = kernel.bind(args).to_code(_config(16, 16, 1))
    assert "pid_0 = cutlass.Int32(cute.arch.block_idx()[0])\n" in source
    assert re.search(rf"_launcher\(.*\({rows},\).*block=\(16, 1, 1\)", source)
    kernel(*args)
    torch.testing.assert_close(args[4].float(), expected.float(), rtol=0, atol=0)


@pytest.mark.parametrize("pack", [2, 8])
def test_warp_wide_tiles_pack_one_warp_per_row(pack: int) -> None:
    """With a warp-wide K extent the layout planner orders the packed axis
    above the body's axis (``block=(32, pack, 1)``): every warp owns one row,
    the K reduction is a single warp shuffle, and the result is exact."""
    rows, kbytes = 256, 256
    args, expected = nvfp4_gemv_args(rows, kbytes)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body,
        backend="cute",
        static_shapes=True,
        config=_config(32, 32, pack),
    )
    bound = kernel.bind(args)
    source = bound.to_code(_config(32, 32, pack))
    assert (
        f"pid_0 = cutlass.Int32(cute.arch.block_idx()[0]) * {pack}"
        " + cutlass.Int32(cute.arch.thread_idx()[1])"
    ) in source
    assert re.search(
        rf"_launcher\(.*\({rows // pack},\).*block=\(32, {pack}, 1\)", source
    )
    assert "_cute_grouped_reduce_warp(" in source
    assert "_cute_grouped_reduce_shared_two_stage" not in source
    kernel(*args)
    torch.testing.assert_close(args[4].float(), expected.float(), rtol=0, atol=0)


def test_tile_pack_seed_opens_the_lane_unroll_search() -> None:
    """The compiler seed proposes the packed warp-per-row form (and the
    unpacked 32-thread form for tall inputs) with the lane unroll set, which
    is what lets the search explore ``cute_lane_unroll`` at all."""
    rows, kbytes = 2048, 2048
    args, _expected = nvfp4_gemv_args(rows, kbytes)
    kernel = helion.kernel(_nvfp4_gemv_fp4in_body, backend="cute", static_shapes=True)
    spec = kernel.bind(args).config_spec
    assert "cute_tile_pack_warp_per_row" in spec.autotuner_heuristics
    seeds = [
        seed.config
        for seed in spec.compiler_seed_configs
        if "cute_tiles_per_block" in seed.config
    ]
    assert [seed["cute_tiles_per_block"] for seed in seeds] == [8, 1, 4, 2]
    primary = seeds[0]
    assert primary["block_sizes"] == [1, 256]
    assert primary["num_threads"] == [1, 32]
    assert primary["cute_lane_unroll"] == 8
    assert primary["cute_lane_layouts"][1] == "strided"
    assert primary["cute_vector_widths"][1] == 1
    assert all(
        spec.normalized_config(helion.Config(**seed)).config == seed for seed in seeds
    )


@helion.kernel(backend="cute", static_shapes=True)
def _row_scale(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i, j in hl.tile([x.size(0), x.size(1)], block_size=[1, None]):
        out[i, j] = x[i, j] * s[i]
    return out


def test_tile_pack_is_refused_under_l2_grouping() -> None:
    """A two-axis grid with a one-element M axis is packable and L2-groupable,
    but ``L2GroupingProgramIDs`` decodes the pid without the pack term (the
    launch would shrink while every sub-group computed the same tile): the
    knob is refused with any L2 grouping above one, and a repaired config
    keeps the grouping, drops the pack and stays exact."""
    torch.manual_seed(0)
    x = torch.randn(64, 256, device=DEVICE)
    s = torch.randn(64, device=DEVICE)
    bound = _row_scale.bind((x, s))
    spec = bound.config_spec
    assert spec.cute_tile_pack_counts == {0: 64}
    assert len(spec.l2_groupings) == 1
    packed = helion.Config(block_sizes=[64], l2_groupings=[1], cute_tiles_per_block=2)
    spec.normalize(packed)
    assert packed.config["cute_tiles_per_block"] == 2
    with pytest.raises(InvalidConfig, match="L2 grouping"):
        spec.normalize(
            helion.Config(block_sizes=[64], l2_groupings=[2], cute_tiles_per_block=2)
        )
    grouped = helion.Config(block_sizes=[64], l2_groupings=[2], cute_tiles_per_block=2)
    spec.normalize(grouped, _fix_invalid=True)
    assert grouped.config["cute_tiles_per_block"] == 1
    assert grouped.config["l2_groupings"] == [2]
    expected = x * s[:, None]
    for config in (packed, grouped):
        kernel = helion.kernel(
            _row_scale.fn, backend="cute", static_shapes=True, config=config
        )
        torch.testing.assert_close(kernel(x, s), expected, rtol=0, atol=0)


def _add_one_per_row(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        for a, b, c in hl.tile([x.size(1), x.size(2), x.size(3)]):
            out[row, a, b, c] = x[row, a, b, c] + 1
    return out


def test_repair_packs_only_tiles_that_fit_one_block() -> None:
    """Packing two tiles of 4 x 256 threads per CTA asked for 2048 threads,
    and the launch fell back to the packed axis alone, ``block=(2, 1, 1)``:
    each tile ran on one thread.  The autotuner's repair drops the pack, so
    the tile launches its own threads and the output is exact."""
    x = torch.randn(4, 64, 64, 256, device=DEVICE, dtype=torch.bfloat16)
    drawn = helion.Config(
        block_sizes=[32, 1, 256],
        num_threads=[4, 0, 256],
        loop_orders=[[2, 1, 0]],
        flatten_loops=[True],
        cute_vector_widths=[4, 8, 4, 4],
        cute_lane_layouts=["blocked", "strided", "strided", "blocked"],
        cute_tiles_per_block=2,
    )
    bound = helion.kernel(_add_one_per_row, backend="cute", static_shapes=True).bind(
        (x,)
    )
    generation = ConfigGeneration(bound.config_spec)
    repaired = generation.unflatten(generation.flatten(drawn))
    assert repaired.config["cute_tiles_per_block"] == 1
    assert "block=(256, 4, 1)" in bound.to_code(repaired)
    bound.set_config(repaired)
    torch.testing.assert_close(bound(x), x + 1, rtol=0, atol=0)


def _add_one_below_rows(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n, a, b = x.shape
    for outer in hl.tile(n, block_size=1):
        for ta, tb, tc in hl.tile([outer.begin, 0, 0], [outer.end, a, b]):
            out[ta, tb, tc] = x[ta, tb, tc] + 1
    return out


def test_auto_counts_fit_beside_the_packed_tiles() -> None:
    """Eight packed tiles of an auto-threaded 8 x 32 tile took 256 threads
    each, 2048 together: the launch fell back to the packed axis alone
    (``block=(8, 1, 1)``) and each tile ran on one thread.  The planner fits
    the auto counts into the 128 threads the pack leaves each tile, and the
    output is exact."""
    x = torch.randn(64, 8, 32, device=DEVICE, dtype=torch.bfloat16)
    config = helion.Config(
        block_sizes=[1, 8, 32], num_threads=[1, 0, 0], cute_tiles_per_block=8
    )
    kernel = helion.kernel(
        _add_one_below_rows, backend="cute", static_shapes=True, config=config
    )
    assert "block=(8, 8, 16)" in kernel.bind((x,)).to_code(config)
    torch.testing.assert_close(kernel(x), x + 1, rtol=0, atol=0)
