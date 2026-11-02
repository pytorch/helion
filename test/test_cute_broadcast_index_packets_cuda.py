"""Execute gathered packets and their scalar tail path with changing live IDs."""

from __future__ import annotations

import pytest
import torch

from test.test_cute_broadcast_index_packets import gathered_sum
from test.test_cute_broadcast_index_packets import packet_config

from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("limit", [127, 128])
def test_broadcast_packets_respect_inactive_rows_and_tail(
    dtype: torch.dtype, limit: int
) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(91)
    x = torch.randn((12, 256), generator=generator, device=DEVICE, dtype=dtype)
    ids = torch.tensor([-1, 12, 0, 3, 7, 11, 2], device=DEVICE, dtype=torch.int32)
    args = (x, ids, limit, 1)
    bound = gathered_sum._bind_isolated(args)
    run = bound.compile_config(packet_config(4))
    original = x.clone()
    for replacement in (None, [5, 0, 12, -1, 4, 8, 6]):
        if replacement is not None:
            ids.copy_(torch.tensor(replacement, device=DEVICE, dtype=torch.int32))
        expected = torch.zeros_like(ids, dtype=torch.float32)
        valid = (ids >= 0) & (ids < x.shape[0])
        expected[valid] = x[ids[valid].long(), :limit].float().sum(-1)
        actual = run(*args)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
