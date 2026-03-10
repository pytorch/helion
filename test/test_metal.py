from __future__ import annotations

import unittest

import torch

import helion
import helion.language as hl

requires_mps = unittest.skipUnless(
    torch.backends.mps.is_available(),
    "MPS backend not available",
)


@helion.kernel(backend="metal")
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="metal")
def softmax(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :]
        amax = torch.amax(values, dim=1, keepdim=True)
        exp = torch.exp(values - amax)
        sum_exp = torch.sum(exp, dim=1, keepdim=True)
        out[tile_n, :] = exp / sum_exp
    return out


class TestMetalBackend(unittest.TestCase):
    @requires_mps
    def test_vector_add(self) -> None:
        device = torch.device("mps")
        x = torch.randn(1024, device=device, dtype=torch.float32)
        y = torch.randn(1024, device=device, dtype=torch.float32)
        result = vector_add(x, y)
        expected = x + y
        torch.testing.assert_close(result, expected)

    @requires_mps
    def test_vector_add_large(self) -> None:
        device = torch.device("mps")
        x = torch.randn(100000, device=device, dtype=torch.float32)
        y = torch.randn(100000, device=device, dtype=torch.float32)
        result = vector_add(x, y)
        expected = x + y
        torch.testing.assert_close(result, expected)

    @requires_mps
    def test_softmax(self) -> None:
        device = torch.device("mps")
        x = torch.randn(128, 256, device=device, dtype=torch.float32)
        result = softmax(x)
        expected = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    @requires_mps
    def test_softmax_autotune(self) -> None:
        """Softmax with autotuning — searches over block_sizes."""

        @helion.kernel(backend="metal", autotune_effort="quick")
        def softmax_tuned(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1, keepdim=True)
                exp = torch.exp(values - amax)
                sum_exp = torch.sum(exp, dim=1, keepdim=True)
                out[tile_n, :] = exp / sum_exp
            return out

        device = torch.device("mps")
        x = torch.randn(128, 256, device=device, dtype=torch.float32)
        result = softmax_tuned(x)
        expected = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
