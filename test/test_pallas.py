from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


def _has_pallas() -> bool:
    try:
        import jax  # noqa: F401
        from jax.experimental import pallas  # noqa: F401

        return True
    except ImportError:
        return False


skipUnlessPallas = unittest.skipUnless(_has_pallas(), "JAX/Pallas not available")


@helion.kernel(backend="pallas", static_shapes=True)
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@skipUnlessPallas
class TestPallas(TestCase):
    def test_add_1d(self) -> None:
        args = (torch.randn(1024), torch.randn(1024))
        code, result = code_and_output(add_kernel, args, block_size=256)
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)

    def test_add_large(self) -> None:
        args = (torch.randn(4096), torch.randn(4096))
        code, result = code_and_output(add_kernel, args, block_size=512)
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
