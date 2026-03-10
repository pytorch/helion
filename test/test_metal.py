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


@helion.kernel(backend="metal")
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _k2, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
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
    def test_matmul(self) -> None:
        device = torch.device("mps")
        M, N, K = 128, 128, 128
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_non_square(self) -> None:
        device = torch.device("mps")
        M, K, N = 128, 64, 256
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_large_square(self) -> None:
        device = torch.device("mps")
        M, K, N = 512, 512, 512
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_non_power_of_2(self) -> None:
        device = torch.device("mps")
        M, K, N = 100, 200, 150
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_prime_sizes(self) -> None:
        device = torch.device("mps")
        M, K, N = 97, 113, 89
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_tall_skinny(self) -> None:
        device = torch.device("mps")
        M, K, N = 1000, 64, 37
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_wide_flat(self) -> None:
        device = torch.device("mps")
        M, K, N = 33, 500, 1000
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_small(self) -> None:
        device = torch.device("mps")
        M, K, N = 7, 11, 13
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_vector_add_non_power_of_2(self) -> None:
        device = torch.device("mps")
        x = torch.randn(999, device=device, dtype=torch.float32)
        y = torch.randn(999, device=device, dtype=torch.float32)
        result = vector_add(x, y)
        expected = x + y
        torch.testing.assert_close(result, expected)

    @requires_mps
    def test_softmax_non_power_of_2(self) -> None:
        device = torch.device("mps")
        x = torch.randn(100, 200, device=device, dtype=torch.float32)
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
