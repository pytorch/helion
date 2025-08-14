from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRocm
import helion.language as hl


class TestViews(RefEagerTestBase, TestCase):
    def test_softmax_unsqueeze(self):
        @helion.kernel(config={"block_size": 1})
        def softmax(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1).unsqueeze(1)
                exp = torch.exp(values - amax)
                sum_exp = torch.unsqueeze(torch.sum(exp, dim=1), -1)
                out[tile_n, :] = exp / sum_exp
            return out

        x = torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16)
        code, result = code_and_output(softmax, (x,))
        torch.testing.assert_close(
            result, torch.nn.functional.softmax(x, dim=1), rtol=1e-2, atol=1e-1
        )
        self.assertExpectedJournal(code)

    @skipIfRocm("too slow on rocm")
    def test_softmax_view_reshape(self):
        @helion.kernel(config={"block_size": 1})
        def softmax(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1).view(tile_n, 1)
                exp = torch.exp(values - amax)
                sum_exp = torch.reshape(torch.sum(exp, dim=1), [tile_n, 1])
                out[tile_n, :] = exp / sum_exp
            return out

        x = torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16)
        code, result = code_and_output(softmax, (x,))
        torch.testing.assert_close(
            result, torch.nn.functional.softmax(x, dim=1), rtol=1e-2, atol=1e-1
        )
        self.assertExpectedJournal(code)

    def test_squeeze(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                out[tile_n, tile_m] = x[tile_n, tile_m] + y[tile_m, :].squeeze(
                    1
                ).unsqueeze(0)
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024, 1], device=DEVICE),
        )
        code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1][:, 0].unsqueeze(0))
        self.assertExpectedJournal(code)

    def test_transpose(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                out[tile_n, tile_m] = x[tile_n, tile_m] + y[tile_m, :].transpose(0, 1)
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024, 1], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1].transpose(0, 1))

    def test_expand(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                out[tile_n, tile_m] = x[tile_n, tile_m] + y[tile_n, :].expand(
                    tile_n, tile_m
                )
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024, 1], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_expand_as(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                a = x[tile_n, tile_m]
                b = y[tile_m].expand_as(a)
                out[tile_n, tile_m] = a + b
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_expand_slicing(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "pointer"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                a = x[tile_n, tile_m]
                b = y[tile_m]
                out[tile_n, tile_m] = a + b[None, :]
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_expand_implicit(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "pointer"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                a = x[tile_n, tile_m]
                b = y[tile_m]
                out[tile_n, tile_m] = a + b
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_reshape_input_types(self):
        @helion.kernel(static_shapes=True)
        def reshape_reduction_dim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"

            out = torch.zeros(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )

            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])

                # Test different reshape input types
                reshaped_acc = acc.reshape(-1, tile_m.block_size * tile_n.block_size)
                reshaped_acc = reshaped_acc.reshape(
                    tile_m.block_size, tile_n.block_size
                )
                reshaped_acc = reshaped_acc.flatten(0)
                reshaped_acc = reshaped_acc.reshape(tile_m, tile_n)
                reshaped_acc = reshaped_acc.reshape(
                    tile_m.block_size * 2 // 2, tile_n.block_size + 1 - 1
                )
                out[tile_m, tile_n] = reshaped_acc

            return out

        x = torch.randn(8, 16, device=DEVICE)
        y = torch.randn(16, 32, device=DEVICE)
        _code, result = code_and_output(reshape_reduction_dim, (x, y))
        expected = torch.matmul(x, y)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
