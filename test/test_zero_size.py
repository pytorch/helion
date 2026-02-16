from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@onlyBackends(["triton"])
class TestZeroSizeTensors(RefEagerTestBase, TestCase):
    def test_pointwise_zero_rows(self) -> None:
        @helion.kernel(autotune_effort="none")
        def pointwise_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile, :] = x[tile, :] + y[tile, :]
            return out

        x = torch.randn([0, 32], device=DEVICE)
        y = torch.randn([0, 32], device=DEVICE)
        _, result = code_and_output(pointwise_add, (x, y))
        torch.testing.assert_close(result, x + y)

    def test_reduce_zero_inner_dim(self) -> None:
        @helion.kernel(autotune_effort="none")
        def row_sums(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for tile in hl.tile(x.size(0)):
                rows = x[tile, :]
                out[tile] = torch.sum(rows, dim=1)
            return out

        x = torch.randn([5, 0], device=DEVICE)
        _, result = code_and_output(row_sums, (x,))
        torch.testing.assert_close(result, torch.sum(x, dim=1))

    def test_local_zero_width_allocation(self) -> None:
        @helion.kernel(autotune_effort="none")
        def zero_width(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                scratch = hl.zeros([tile, 0], dtype=x.dtype)
                out[tile, :] = scratch
            return out

        x = torch.empty([4, 0], device=DEVICE)
        _, result = code_and_output(zero_width, (x,))
        torch.testing.assert_close(result, torch.zeros_like(x))
