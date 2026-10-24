from __future__ import annotations

import re

from examples.nvfp4_gemv import _nvfp4_gemv_fp4in_body
from examples.nvfp4_gemv import make_fp8_scales
from examples.nvfp4_gemv import reference_nvfp4_gemv_fp4in
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


def _gemv_args(rows: int, kbytes: int) -> tuple[tuple[object, ...], torch.Tensor]:
    torch.manual_seed(0)
    groups = kbytes // 8
    weight = torch.randint(0, 256, (rows, kbytes), dtype=torch.uint8, device=DEVICE)
    weight = weight.view(torch.float4_e2m1fn_x2)
    x = torch.randint(0, 256, (kbytes,), dtype=torch.uint8, device=DEVICE)
    x = x.view(torch.float4_e2m1fn_x2)
    wscale = make_fp8_scales((rows, groups), DEVICE)
    xscale = make_fp8_scales((groups,), DEVICE)
    expected = reference_nvfp4_gemv_fp4in(weight, x, wscale, xscale)
    out = torch.empty(rows, dtype=torch.bfloat16, device=DEVICE)
    args = (
        weight.view(rows, groups, 8),
        x.view(groups, 8),
        wscale.reshape(-1),
        xscale.reshape(-1),
        out,
        1.0,
    )
    return args, expected


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
    args, _ = _gemv_args(2048, 512)
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
    args, expected = _gemv_args(rows, kbytes)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body,
        backend="cute",
        static_shapes=True,
        config=_config(16, 16, pack),
    )
    bound = kernel.bind(args)
    source = bound.to_triton_code(_config(16, 16, pack))
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
    args, expected = _gemv_args(rows, kbytes)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body,
        backend="cute",
        static_shapes=True,
        config=_config(16, 16, 1),
    )
    source = kernel.bind(args).to_triton_code(_config(16, 16, 1))
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
    args, expected = _gemv_args(rows, kbytes)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body,
        backend="cute",
        static_shapes=True,
        config=_config(32, 32, pack),
    )
    bound = kernel.bind(args)
    source = bound.to_triton_code(_config(32, 32, pack))
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
    args, _expected = _gemv_args(rows, kbytes)
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
