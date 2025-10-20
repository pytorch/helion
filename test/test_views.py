from __future__ import annotations

import unittest

import torch

import helion
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfPy314
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

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_squeeze(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "tensor_descriptor"})
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

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_transpose(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "tensor_descriptor"})
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

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_expand(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "tensor_descriptor"})
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

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_expand_as(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "tensor_descriptor"})
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

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
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

    def test_split_join_roundtrip(self):
        @helion.kernel(config={"block_size": 64})
        def fn(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                lo, hi = hl.split(x[tile, :])
                out[tile, :] = hl.join(hi, lo)
            return out

        x = torch.randn([256, 2], device=DEVICE)
        code, result = code_and_output(fn, (x,))
        expected = torch.stack((x[:, 1], x[:, 0]), dim=-1)
        torch.testing.assert_close(result, expected)
        self.assertIn("tl.split", code)
        self.assertIn("tl.join", code)

    def test_join_broadcast_scalar(self):
        @helion.kernel(config={"block_size": 64})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.empty([n, 2], dtype=x.dtype, device=x.device)
            for tile in hl.tile(n):
                scalar = hl.load(y, [0])
                out[tile, :] = hl.join(x[tile], scalar)
            return out

        x = torch.randn([128], device=DEVICE)
        y = torch.randn([1], device=DEVICE)
        code, result = code_and_output(fn, (x, y))
        broadcast_y = torch.broadcast_to(y, x.shape)
        expected = torch.stack((x, broadcast_y), dim=-1)
        torch.testing.assert_close(result, expected)
        self.assertIn("tl.join", code)

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

    def test_reshape_sum(self):
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0], dtype=x.dtype)
                for tile1, tile2 in hl.tile([x.size(1), x.size(2)]):
                    acc += x[tile0, tile1, tile2].reshape(tile0, -1).sum(-1)
                out[tile0] = acc
            return out

        x = torch.randn(3, 4, 5, device=DEVICE)
        code, result = code_and_output(fn, (x,))
        expected = x.sum(dim=(1, 2))
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    def test_stack_power_of_2(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def test_stack_power_of_2_kernel(
            a: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
            M, N = a.shape
            result = torch.zeros(M * 2, N, dtype=a.dtype, device=a.device)

            for tile_m in hl.tile(M):
                for tile_n in hl.tile(N):
                    a_tile = a[tile_m, tile_n]
                    b_tile = b[tile_m, tile_n]

                    # Stack tensors along dim=1 (creates [BLOCK_M, 2, BLOCK_N])
                    stacked = torch.stack([a_tile, b_tile], dim=1)

                    # Reshape to [BLOCK_M * 2, BLOCK_N]
                    reshaped = stacked.reshape(tile_m.block_size * 2, tile_n.block_size)

                    result[
                        (tile_m.begin * 2) : (tile_m.begin * 2 + tile_m.block_size * 2),
                        tile_n,
                    ] = reshaped

            return result

        M, N = 64, 128
        device = DEVICE

        a = torch.randn(M, N, dtype=torch.float32, device=device)
        b = torch.randn(M, N, dtype=torch.float32, device=device)

        result = test_stack_power_of_2_kernel(a, b)
        expected = torch.zeros(M * 2, N, dtype=torch.float32, device=device)
        expected[0::2] = a  # Every 2nd row starting from 0
        expected[1::2] = b  # Every 2nd row starting from 1
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_stack_non_power_of_2(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def test_stack_non_power_of_2_kernel(
            a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
        ) -> torch.Tensor:
            M, N = a.shape
            result = torch.zeros(M, 3, N, dtype=a.dtype, device=a.device)

            for tile_m in hl.tile(M):
                for tile_n in hl.tile(N):
                    a_tile = a[tile_m, tile_n]
                    b_tile = b[tile_m, tile_n]
                    c_tile = c[tile_m, tile_n]

                    # Stack tensors along dim=1 (creates [BLOCK_M, 3, BLOCK_N])
                    stacked = torch.stack([a_tile, b_tile, c_tile], dim=1)

                    result[tile_m, :, tile_n] = stacked

            return result

        M, N = 65, 129
        device = DEVICE

        a = torch.randn(M, N, dtype=torch.float32, device=device)
        b = torch.randn(M, N, dtype=torch.float32, device=device)
        c = torch.randn(M, N, dtype=torch.float32, device=device)

        code, result = code_and_output(test_stack_non_power_of_2_kernel, (a, b, c))
        expected = torch.stack([a, b, c], dim=1)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        self.assertExpectedJournal(code)

    @skipIfPy314("torch.compile not yet supported on Python 3.14")
    def test_stack_dim0(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def test_stack_dim0_kernel(
            a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
        ) -> torch.Tensor:
            M, N = a.shape
            result = torch.zeros(3, M, N, dtype=a.dtype, device=a.device)

            for tile_m in hl.tile(M):
                for tile_n in hl.tile(N):
                    a_tile = a[tile_m, tile_n]
                    b_tile = b[tile_m, tile_n]
                    c_tile = c[tile_m, tile_n]

                    # Stack 3 tensors along dim=0
                    # This creates [3, BLOCK_M, BLOCK_N]
                    stacked = torch.stack([a_tile, b_tile, c_tile], dim=0)

                    result[:, tile_m, tile_n] = stacked

            return result

        M, N = 65, 129
        device = DEVICE

        a = torch.randn(M, N, dtype=torch.float32, device=device)
        b = torch.randn(M, N, dtype=torch.float32, device=device)
        c = torch.randn(M, N, dtype=torch.float32, device=device)

        code, result = code_and_output(test_stack_dim0_kernel, (a, b, c))
        expected = torch.stack([a, b, c], dim=0)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        self.assertExpectedJournal(code)

        # Verify torch.compile still decomposes aten.stack to aten.cat
        from torch._inductor import config as inductor_config

        def capture_graph(graph):
            self._graph = str(graph)
            return graph

        with inductor_config.patch(post_grad_custom_pre_pass=capture_graph):
            torch.compile(
                lambda x, y, z: torch.stack([x, y, z], dim=0), backend="inductor"
            )(
                torch.randn(4, 4, device=device),
                torch.randn(4, 4, device=device),
                torch.randn(4, 4, device=device),
            )
        assert "aten.cat" in self._graph and "aten.stack" not in self._graph


if __name__ == "__main__":
    unittest.main()
