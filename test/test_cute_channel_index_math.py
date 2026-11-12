"""The channel of a flat NCHW index, ``(i // HW) % C``, on the GPU: one
quotient per vector when HW is a multiple of the vector width, unsigned and
unguarded under proven bounds; a row index scaled past Int32 under int64
indexing, whose division stays Int64; and split sums over de-linearized NCHW
rows with their masks compared once per vector."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_index_math import flat_channel_config
from helion._testing.cute_index_math import flat_channel_scale
from helion._testing.cute_index_math import row_scale
from helion._testing.cute_index_math import split_row_config
from helion._testing.cute_index_math import split_row_input
from helion._testing.cute_index_math import split_row_moments
from helion._testing.cute_index_math import split_row_moments_dynamic

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("side", [14, 28])
def test_flat_channel_gather_is_exact(side: int) -> None:
    # 14x14 rows: an 8-wide vector straddles a channel boundary every other
    # row, so lanes of one vector gather different channels.  28x28 rows
    # (784 = 98 * 8): every vector divides its base instead of each lane.
    x = torch.randn(4, 24, side, side, device=DEVICE, dtype=torch.bfloat16)
    scale = torch.randn(24, device=DEVICE, dtype=torch.float32)
    bound = flat_channel_scale.bind((x, scale))
    config = flat_channel_config(bound, proven_bounds=True)
    bound.set_config(config)
    code = bound.to_code(config)
    assert "cutlass.Uint32(" in code, code
    assert "< 24 else" not in code, code
    assert ("cutlass.Uint32(lane_base_0) //" in code) is (side == 28), code
    expected = (x.float() * scale[None, :, None, None]).to(x.dtype)
    torch.testing.assert_close(bound(x, scale), expected, atol=0, rtol=0)


def test_int64_row_index_products_are_exact() -> None:
    # (i // 8) * 2**20 passes Int32 from i = 2**14 on: the proven
    # nonnegative quotient must stay an Int64.
    x = torch.empty(65536, device=DEVICE)
    bound = row_scale[torch.int64].bind((x,))
    config = bound.config_spec.normalized_config(
        helion.Config(block_sizes=[1024], num_threads=[128], cute_proven_bounds=True)
    )
    bound.set_config(config)
    expected = torch.arange(65536, device=DEVICE) // 8 * 1048576
    torch.testing.assert_close(bound(x), expected, atol=0, rtol=0)


def test_uniform_lane_compares_keep_split_sums() -> None:
    # Rows of 192 (24 vectors of 8) in 4 splits of 288: two whole 128-row
    # chunks and one whose last 12 of 16 vectors are masked.  Knob on
    # compares each vector once: counts and sums stay bit for bit, and the
    # sums of squares differ only by FMA contraction, at most an ulp of the
    # running sum for each of a vector's 8 fused lanes, carried through the
    # same combination (16 ulps of these nonnegative sums).  The CPU tests
    # show knob off renders the code without the key.
    x = split_row_input(6, 192, DEVICE)
    bound = split_row_moments.bind((x,))
    off, on = (
        bound.compile_config(split_row_config(bound, uniform_compares=flag))(x)
        for flag in (False, True)
    )
    flat = x.reshape(4, 288, 16).float()
    expected = (torch.full_like(off[0], 288.0), flat.sum(1), (flat * flat).sum(1))
    names = ("count", "sums", "squares")
    for name, b, c, ref in zip(names, off, on, expected, strict=True):
        if name == "squares":
            ulps = (c.view(torch.int32) - b.view(torch.int32)).abs().max().item()
            assert ulps <= 16, ulps
        else:
            torch.testing.assert_close(c, b, atol=0, rtol=0, msg=name)
        torch.testing.assert_close(c, ref, msg=name)


@pytest.mark.parametrize(("batch", "hw"), [(32, 3136), (6, 192)])
def test_delinearized_rows_seed_matches_eager(batch: int, hw: int) -> None:
    # The seed of ``cute_delinearized_row_reduction``: 32-byte bf16 packets
    # over ``r % hw``, the lane unroll and one compare per vector, in 4096-
    # (here) and 512-row chunks (the loop's 288 rows) on SM100+; before
    # SM100 its 16-byte packets give 2048-row chunks here.
    x = split_row_input(batch, hw, DEVICE)
    bound = split_row_moments.bind((x,))
    (seed,) = bound.config_spec.compiler_seed_configs
    outs = bound.compile_config(bound.config_spec.normalized_config(seed))(x)
    flat = x.reshape(4, batch * hw // 4, 16).float()
    expected = (
        torch.full_like(outs[0], batch * hw // 4),
        flat.sum(1),
        (flat * flat).sum(1),
    )
    for out, ref in zip(outs, expected, strict=True):
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-2)


def test_uniform_lane_compares_match_eager_under_dynamic_shapes() -> None:
    # Under dynamic shapes the split length and the row bounds are run-time
    # values whose factors nothing proves, so the knob keeps every per-lane
    # compare (the CPU tests show the code is the knob-off code); one kernel
    # serves rows of 192 and of 196, which are no multiple of the vector.
    small = split_row_input(6, 192, DEVICE)
    bound = split_row_moments_dynamic.bind((small,))
    kernel = bound.compile_config(split_row_config(bound, uniform_compares=True))
    for x in (small, split_row_input(6, 196, DEVICE)):
        total = x.shape[0] * x.shape[1]
        rows = (total + 3) // 4
        flat = torch.zeros(4 * rows, 16, device=DEVICE)
        flat[:total] = x.reshape(total, 16).float()
        valid = torch.zeros(4 * rows, device=DEVICE)
        valid[:total] = 1.0
        flat = flat.view(4, rows, 16)
        count = valid.view(4, rows).sum(1, keepdim=True).expand(4, 16)
        expected = (count, flat.sum(1), (flat * flat).sum(1))
        for out, ref in zip(kernel(x), expected, strict=True):
            torch.testing.assert_close(out, ref)
