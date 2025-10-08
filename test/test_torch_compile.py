from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import skipIfPy314
from helion._testing import skipIfRefEager
import helion.language as hl


class TestTorchCompile(RefEagerTestBase, TestCase):
    @skipIfRefEager("does not work with ref eager")
    @skipIfPy314("torch.compile not yet supported on Python 3.14")
    def test_add_kernel(self):
        @helion.kernel(config=helion.Config(block_sizes=[1, 2]))
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return add(x, y)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        out = add(x, y)
        compiled_add = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_add(x, y)

        torch.testing.assert_close(out, x + y)
        torch.testing.assert_close(compiled_out, x + y)


if __name__ == "__main__":
    unittest.main()
