"""The channel of a flat NCHW index, ``(i // HW) % C``, on the GPU: one
quotient per vector when HW is a multiple of the vector width, unsigned and
unguarded under proven bounds; and a row index scaled past Int32 under int64
indexing, whose division stays Int64."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_index_math import flat_channel_config
from helion._testing.cute_index_math import flat_channel_scale
from helion._testing.cute_index_math import row_scale

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
