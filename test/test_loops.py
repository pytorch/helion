from __future__ import annotations

import functools
from pathlib import Path
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
import helion.language as hl

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


@helion.kernel
def device_loop_3d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    for tile_a in hl.tile(a):
        for tile_b, tile_c, tile_d in hl.tile([b, c, d]):
            out[tile_a, tile_b, tile_c, tile_d] = torch.sin(
                x[tile_a, tile_b, tile_c, tile_d]
            )
    return out


@helion.kernel()
def nested_loop_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    # Outer loop becomes grid (no tl.range)
    for tile_outer in hl.tile(x.size(0)):
        # Inner loop becomes device loop with tl.range
        for tile_inner in hl.tile(x.size(1)):
            out[tile_outer, tile_inner] = x[tile_outer, tile_inner] + 1
    return out


class TestLoops(TestCase):
    def test_pointwise_device_loop(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.pointwise_device_loop,
            args,
            block_sizes=[32, 32],
        )
        torch.testing.assert_close(result, torch.sigmoid(args[0] + 1))
        self.assertExpectedJournal(code)

    def test_3d_device_loop0(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[1, 8, 8, 8],
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedJournal(code)

    def test_3d_device_loop1(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[2, 8, 4, 1],
            loop_order=[1, 0, 2],
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedJournal(code)

    def test_3d_device_loop2(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[4, 128, 1, 1],
            flatten_loops=[True],
            loop_order=[2, 0, 1],
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedJournal(code)

    def test_3d_device_loop3(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[2, 8, 4, 1],
            loop_order=[2, 0, 1],
            indexing="block_ptr",
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedJournal(code)

    def test_loop_fixed_block(self):
        @helion.kernel(config={"block_sizes": [], "indexing": "block_ptr"})
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            a, b, c = x.shape
            for tile_a, tile_b in hl.tile([a, b], block_size=[4, 8]):
                for tile_c in hl.tile(c, block_size=16):
                    out[tile_a, tile_b, tile_c] = torch.sin(x[tile_a, tile_b, tile_c])
            return out

        args = (torch.randn([128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedJournal(code)

    def test_loop_arg_block(self):
        @helion.kernel(config={"block_sizes": [], "indexing": "block_ptr"})
        def fn(x: torch.Tensor, block_size: int) -> torch.Tensor:
            out = torch.empty_like(x)
            (a,) = x.shape
            for tile_a in hl.tile(a, block_size=block_size):
                out[tile_a] = torch.sin(x[tile_a])
            return out

        args = (torch.randn([1024], device=DEVICE), 32)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedJournal(code)

    def test_three_level_matmul(self):
        @helion.kernel(static_shapes=True)
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for tile_m in hl.tile(m):
                for tile_n in hl.tile(n):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                    out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([256, 512], device=DEVICE),
            torch.randn([512, 128], device=DEVICE),
        )
        code, result = code_and_output(matmul, args, block_sizes=[16, 64, 64])
        torch.testing.assert_close(
            result, functools.reduce(torch.matmul, args), atol=1e-1, rtol=1e-2
        )
        self.assertExpectedJournal(code)

    def test_data_dependent_bounds1(self):
        @helion.kernel()
        def fn(x: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            bs = hl.register_block_size(x.size(1))
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0, bs])
                for tile1 in hl.tile(end[0], block_size=bs):
                    acc += x[tile0, tile1]
                out[tile0] = acc.sum(-1)
            return out

        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.tensor([200], device=DEVICE, dtype=torch.int64),
        )
        code, result = code_and_output(fn, args, block_sizes=[32, 32])
        self.assertExpectedJournal(code)
        torch.testing.assert_close(result, args[0][:, : args[1][0].item()].sum(-1))

    def test_data_dependent_bounds2(self):
        @helion.kernel()
        def fn(x: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0])
                for tile1 in hl.tile(end[0]):
                    acc += x[tile0, tile1].sum(-1)
                out[tile0] = acc
            return out

        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.tensor([200], device=DEVICE, dtype=torch.int32),
        )
        code, result = code_and_output(
            fn, args, block_sizes=[32, 32], indexing="block_ptr"
        )
        self.assertExpectedJournal(code)
        torch.testing.assert_close(result, args[0][:, : args[1][0].item()].sum(-1))

    def test_data_dependent_bounds3(self):
        @helion.kernel()
        def fn(x: torch.Tensor, end0: torch.Tensor, end1: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0], dtype=x.dtype)
                for tile1, tile2 in hl.tile([end0[0], end1[0]]):
                    # TODO(jansel): make this version work
                    # acc += x[tile0, tile1, tile2].reshape(tile0, -1).sum(-1)
                    acc += x[tile0, tile1, tile2].sum(-1).sum(-1)
                out[tile0] = acc
            return out

        args = (
            torch.randn([32, 256, 256], device=DEVICE, dtype=torch.float64),
            torch.tensor([200], device=DEVICE, dtype=torch.int64),
            torch.tensor([150], device=DEVICE, dtype=torch.int64),
        )
        code, result = code_and_output(fn, args, block_sizes=[32, 32, 32])
        self.assertExpectedJournal(code)
        torch.testing.assert_close(
            result, args[0][:, : args[1][0].item(), : args[2][0].item()].sum(-1).sum(-1)
        )

    def test_data_dependent_bounds4(self):
        @helion.kernel()
        def fn(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            bs = hl.register_block_size(8192)
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0, bs])
                for tile1 in hl.tile(begin[0], end[0], block_size=bs):
                    acc += x[tile0, tile1]
                out[tile0] = acc.sum(-1)
            return out

        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.tensor([100], device=DEVICE, dtype=torch.int64),
            torch.tensor([200], device=DEVICE, dtype=torch.int64),
        )
        code, result = code_and_output(fn, args, block_sizes=[32, 32])
        self.assertExpectedJournal(code)
        torch.testing.assert_close(
            result, args[0][:, args[1][0].item() : args[2][0].item()].sum(-1)
        )

    def test_data_dependent_bounds5(self):
        @helion.kernel()
        def fn(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0])
                for (tile1,) in hl.tile([begin[0]], [end[0]]):
                    acc += x[tile0, tile1].sum(-1)
                out[tile0] = acc
            return out

        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.tensor([100], device=DEVICE, dtype=torch.int64),
            torch.tensor([200], device=DEVICE, dtype=torch.int64),
        )
        code, result = code_and_output(fn, args, block_sizes=[32, 32])
        self.assertExpectedJournal(code)
        torch.testing.assert_close(
            result, args[0][:, args[1][0].item() : args[2][0].item()].sum(-1)
        )

    def test_register_block_size_minimum(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            bs = hl.register_block_size(32, 256)
            for tile0 in hl.tile(x.size(0), block_size=bs):
                out[tile0] = x[tile0] + 1
            return out

        args = (torch.randn([1024], device=DEVICE, dtype=torch.float32),)
        code, result = code_and_output(fn, args, block_size=64)
        torch.testing.assert_close(result, args[0] + 1)
        spec = fn.bind(args).config_spec.block_sizes[0]
        self.assertEqual(spec.size_hint, 1024)
        self.assertEqual(spec.min_size, 32)
        self.assertEqual(spec.max_size, 256)

    def test_reorder_with_register_block_size(self):
        @helion.kernel(
            config={
                "block_sizes": [64, 32],
                "indexing": "block_ptr",
                "loop_order": [1, 0],
            }
        )
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            bs0 = hl.register_block_size(1024)
            bs1 = hl.register_block_size(1024)
            for tile0, tile1 in hl.tile(x.size(), block_size=[bs0, bs1]):
                out[tile0, tile1] = x[tile0, tile1] + 1
            return out

        args = (torch.randn([2048, 2048], device=DEVICE),)
        code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + 1)
        self.assertExpectedJournal(code)

    def test_l2_grouping_with_register_block_size(self):
        @helion.kernel(
            config={
                "block_sizes": [32, 16],
                "indexing": "block_ptr",
                "l2_grouping": 8,
            }
        )
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            bs0 = hl.register_block_size(1024)
            bs1 = hl.register_block_size(1024)
            for tile0, tile1 in hl.tile(x.size(), block_size=[bs0, bs1]):
                out[tile0, tile1] = x[tile0, tile1] + 1
            return out

        args = (torch.randn([2048, 2048], device=DEVICE),)
        code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + 1)
        self.assertExpectedJournal(code)

    def test_multiple_for_loop_1d(self):
        @helion.kernel
        def addToBoth(a, b, c):
            x0, c0 = a
            x1, c1 = b
            x2, c2 = c
            for tile in hl.tile(x0.size()):
                x0[tile] += c0
            for tile in hl.tile(x1.size()):
                x1[tile] += c1
            for tile in hl.tile(x2.size()):
                x2[tile] += c2
            return x0, x1, x2

        constants = [2, 4, 8]
        args = [(torch.ones(5, device=DEVICE), constants[i]) for i in range(3)]
        eager_results = [t + c for t, c in args]

        code, compiled_result = code_and_output(addToBoth, args)

        assert isinstance(compiled_result, tuple)
        for e, c in zip(eager_results, compiled_result, strict=False):
            torch.testing.assert_close(e, c)

        self.assertExpectedJournal(code)

    def test_multiple_for_loop_2d(self):
        @helion.kernel
        def addToBoth(a, b, c):
            x0, c0 = a
            x1, c1 = b
            x2, c2 = c

            a_n, a_m = x0.shape
            b_n, b_m = x1.shape
            c_n, c_m = x2.shape

            for tile_n in hl.tile(a_n):
                for tile_m in hl.tile(a_m):
                    x0[tile_n, tile_m] += c0
            for tile_n in hl.tile(b_n):
                for tile_m in hl.tile(b_m):
                    x1[tile_n, tile_m] += c1
            for tile_n in hl.tile(c_n):
                for tile_m in hl.tile(c_m):
                    x2[tile_n, tile_m] += c2
            return x0, x1, x2

        constants = [2, 4, 8]
        args = [(torch.ones(5, 10, device=DEVICE), constants[i]) for i in range(3)]
        eager_results = [t + c for t, c in args]

        code, compiled_result = code_and_output(addToBoth, args)

        assert isinstance(compiled_result, tuple)
        for e, c in zip(eager_results, compiled_result, strict=False):
            torch.testing.assert_close(e, c)

        self.assertExpectedJournal(code)

    def test_multiple_for_loop_2d_multiple_tile(self):
        @helion.kernel
        def addToBoth(a, b, c):
            x0, c0 = a
            x1, c1 = b
            x2, c2 = c

            a_n, a_m = x0.shape
            b_n, b_m = x1.shape
            c_n, c_m = x2.shape

            for tile_n, tile_m in hl.tile([a_n, a_m]):
                x0[tile_n, tile_m] += c0
            for tile_n, tile_m in hl.tile([b_n, b_m]):
                x1[tile_n, tile_m] += c1
            for tile_n, tile_m in hl.tile([c_n, c_m]):
                x2[tile_n, tile_m] += c2
            return x0, x1, x2

        constants = [2, 4, 8]
        args = [(torch.ones(5, 10, device=DEVICE), constants[i]) for i in range(3)]
        eager_results = [t + c for t, c in args]

        code, compiled_result = code_and_output(addToBoth, args)

        assert isinstance(compiled_result, tuple)
        for e, c in zip(eager_results, compiled_result, strict=False):
            torch.testing.assert_close(e, c)

        self.assertExpectedJournal(code)

    def test_chebyshev_polynomials(self):
        """Test nested loops with sequential computation - Chebyshev polynomials."""

        def chebyshev_torch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            # x has shape (B, C)
            # w has shape (N, C), where N corresponds to order of Chebyshev polynomials
            # this function combines building Chebyshev polynomials with x and contracting with w, i.e.
            # 1. (B, C) -> (B, N, C)
            # 2. (B, N, C), (N, C) -> (B, C)
            assert w.size(0) >= 2
            # build weighted Chebyshev polynomials
            T0 = torch.ones_like(x)
            T1 = x
            acc = T0 * w[0] + T1 * w[1]
            for n in range(2, w.size(0)):
                T_new = 2 * x * T1 - T0
                acc = acc + T_new * w[n]
                T0 = T1
                T1 = T_new
            return acc

        @helion.kernel(use_default_config=True)
        def chebyshev_kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            B, C = x.shape
            N, C = w.shape
            hl.specialize(N)
            out = torch.zeros((B, C), device=x.device, dtype=x.dtype)
            assert N >= 2, "assume N>= 2 for simplicity"
            for b_tile, c_tile in hl.tile([B, C]):
                in_x = x[b_tile, c_tile]
                T0 = hl.full((b_tile, c_tile), 1.0, x.dtype)
                T1 = in_x
                acc = w[0, c_tile][None, :] * T0 + w[1, c_tile][None, :] * T1
                two_x = 2.0 * in_x
                for order in hl.tile(2, N, block_size=1):
                    new_T = two_x * T1 - T0
                    acc = acc + w[order, c_tile] * new_T
                    T0 = T1
                    T1 = new_T
                out[b_tile, c_tile] = acc
            return out

        # test tensors
        args = (
            torch.randn(123, 64, device=DEVICE, dtype=torch.float32),
            torch.randn(5, 64, device=DEVICE, dtype=torch.float32),
        )

        code, result = code_and_output(chebyshev_kernel, args)
        expected = chebyshev_torch(args[0], args[1])
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
        self.assertExpectedJournal(code)

    def test_loop_unroll1(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
                for i in [1, 2, 3]:
                    out[tile] += i
            return out

        x = torch.randn(4, device=DEVICE)
        code, output = code_and_output(fn, (x,))
        torch.testing.assert_close(output, x + 6)
        self.assertExpectedJournal(code)

    def test_loop_unroll2(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            a = 1
            b = 2
            c = 3
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
                for i in (a, b, c):
                    out[tile] += i
            return out

        x = torch.randn(4, device=DEVICE)
        code, output = code_and_output(fn, (x,))
        torch.testing.assert_close(output, x + 6)
        self.assertExpectedJournal(code)

    def test_variable_assignment_phi_nodes(self):
        """Test for phi node issue with variable assignments like U1 = two_x.

        This test ensures that simple variable assignments create new variables
        rather than aliases, preventing phi node issues when the source variable
        gets mutated in loops.
        """

        @helion.kernel(use_default_config=True)
        def kernel_with_assignment(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            B, C = x.shape
            N, _ = w.shape
            hl.specialize(N)
            grad_x = torch.zeros_like(x)

            for b_tile, c_tile in hl.tile([B, C]):
                in_x = x[b_tile, c_tile]
                two_x = 2.0 * in_x

                # This assignment should create a new variable, not an alias
                U1 = two_x
                U0 = hl.full((b_tile, c_tile), 1.0, x.dtype)

                acc = w[0, c_tile] * U0 + w[1, c_tile] * U1

                for order in hl.tile(2, N, block_size=1):
                    acc += w[order, c_tile] * U1
                    U_new = two_x * U1 - U0
                    U0 = U1
                    U1 = U_new

                grad_x[b_tile, c_tile] = acc
            return grad_x

        @helion.kernel(use_default_config=True)
        def kernel_without_assignment(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            B, C = x.shape
            N, _ = w.shape
            hl.specialize(N)
            grad_x = torch.zeros_like(x)

            for b_tile, c_tile in hl.tile([B, C]):
                in_x = x[b_tile, c_tile]
                two_x = 2.0 * in_x

                # Direct use without assignment
                U1 = 2.0 * in_x
                U0 = hl.full((b_tile, c_tile), 1.0, x.dtype)

                acc = w[0, c_tile] * U0 + w[1, c_tile] * U1

                for order in hl.tile(2, N, block_size=1):
                    acc += w[order, c_tile] * U1
                    U_new = two_x * U1 - U0
                    U0 = U1
                    U1 = U_new

                grad_x[b_tile, c_tile] = acc
            return grad_x

        # Test with small tensor
        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        w = torch.randn(5, 8, device=DEVICE, dtype=torch.float32)

        code1, result1 = code_and_output(kernel_with_assignment, (x, w))
        code2, result2 = code_and_output(kernel_without_assignment, (x, w))

        # Both should produce identical results
        torch.testing.assert_close(result1, result2, rtol=1e-5, atol=1e-5)

    def test_range_unroll_factors(self):
        # Test configuration validation - that range_unroll_factors works
        args = (torch.randn([64, 32], device=DEVICE),)

        # Test with range_unroll_factors = [0, 0] (no unrolling for device loop)
        code0, result0 = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_unroll_factors=[0, 0]
        )

        # Test with range_unroll_factors = [0, 2] (unroll factor 2 for device loop)
        code2, result2 = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_unroll_factors=[0, 2]
        )

        torch.testing.assert_close(result0, result2)
        torch.testing.assert_close(result0, args[0] + 1)
        self.assertNotEqual(code0, code2)
        self.assertNotIn("loop_unroll_factor", code0)
        self.assertExpectedJournal(code2)

    @unittest.skipIf(
        DEVICE.type != "cuda" or torch.cuda.get_device_capability() < (12, 0),
        "Warp specialization requires CUDA compute capability >= 12.0",
    )
    def test_range_warp_specialize(self):
        # Test configuration validation - that range_warp_specialize works
        args = (torch.randn([64, 32], device=DEVICE),)

        # Test with range_warp_specializes = [None, None] (no warp specialization for device loop)
        code_none, result_none = code_and_output(
            nested_loop_kernel,
            args,
            block_sizes=[32, 16],
            range_warp_specializes=[None, None],
        )

        # Test with range_warp_specializes = [None, True] (warp specialization enabled for device loop)
        code_true, result_true = code_and_output(
            nested_loop_kernel,
            args,
            block_sizes=[32, 16],
            range_warp_specializes=[None, True],
        )

        # Test with range_warp_specializes = [None, False] (warp specialization disabled for device loop)
        code_false, result_false = code_and_output(
            nested_loop_kernel,
            args,
            block_sizes=[32, 16],
            range_warp_specializes=[None, False],
        )

        torch.testing.assert_close(result_none, result_true)
        torch.testing.assert_close(result_none, result_false)
        torch.testing.assert_close(result_none, args[0] + 1)

        # Ensure different code is generated for different settings
        self.assertNotEqual(code_none, code_true)
        self.assertNotEqual(code_none, code_false)
        self.assertNotEqual(code_true, code_false)

        # Check that warp_specialize appears in the generated code
        self.assertNotIn("warp_specialize", code_none)
        self.assertIn("warp_specialize=True", code_true)
        self.assertIn("warp_specialize=False", code_false)

    def test_range_num_stages(self):
        # Test configuration validation - that range_num_stages works
        args = (torch.randn([64, 32], device=DEVICE),)

        # Test with range_num_stages = [0, 0] (no num_stages for device loop)
        code0, result0 = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_num_stages=[0, 0]
        )

        # Test with range_num_stages = [0, 3] (num_stages=3 for device loop)
        code3, result3 = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_num_stages=[0, 3]
        )

        torch.testing.assert_close(result0, result3)
        torch.testing.assert_close(result0, args[0] + 1)
        self.assertNotEqual(code0, code3)
        # Check that range_num_stages parameter appears in tl.range call
        self.assertNotIn(
            "tl.range(0, x_size_1.to(tl.int32), _BLOCK_SIZE_1, num_stages=", code0
        )
        self.assertIn(
            "tl.range(0, x_size_1.to(tl.int32), _BLOCK_SIZE_1, num_stages=3)", code3
        )

    def test_range_multi_buffers(self):
        # Test configuration validation - that range_multi_buffers works
        args = (torch.randn([64, 32], device=DEVICE),)

        # Test with range_multi_buffers = [None, None] (no disallow_acc_multi_buffer for device loop)
        code_none, result_none = code_and_output(
            nested_loop_kernel,
            args,
            block_sizes=[32, 16],
            range_multi_buffers=[None, None],
        )

        # Test with range_multi_buffers = [None, True] (disallow_acc_multi_buffer=False for device loop)
        code_true, result_true = code_and_output(
            nested_loop_kernel,
            args,
            block_sizes=[32, 16],
            range_multi_buffers=[None, True],
        )

        # Test with range_multi_buffers = [None, False] (disallow_acc_multi_buffer=True for device loop)
        code_false, result_false = code_and_output(
            nested_loop_kernel,
            args,
            block_sizes=[32, 16],
            range_multi_buffers=[None, False],
        )

        torch.testing.assert_close(result_none, result_true)
        torch.testing.assert_close(result_none, result_false)
        torch.testing.assert_close(result_none, args[0] + 1)
        self.assertNotEqual(code_none, code_true)
        self.assertNotEqual(code_none, code_false)
        self.assertNotEqual(code_true, code_false)
        # Check that disallow_acc_multi_buffer parameter appears in tl.range call
        self.assertNotIn("disallow_acc_multi_buffer", code_none)
        self.assertIn("disallow_acc_multi_buffer=False", code_true)
        self.assertIn("disallow_acc_multi_buffer=True", code_false)

    def test_range_flatten(self):
        # Test configuration validation - that range_flatten works
        args = (torch.randn([64, 32], device=DEVICE),)

        # Test with range_flattens = [None, None] (default, no flatten parameter)
        code_none, result_none = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_flattens=[None, None]
        )

        # Test with range_flattens = [None, True] (flatten=True for device loop)
        code_true, result_true = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_flattens=[None, True]
        )

        # Test with range_flattens = [None, False] (flatten=False for device loop)
        code_false, result_false = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_flattens=[None, False]
        )

        torch.testing.assert_close(result_none, result_true)
        torch.testing.assert_close(result_none, result_false)
        torch.testing.assert_close(result_none, args[0] + 1)
        self.assertNotEqual(code_none, code_true)
        self.assertNotEqual(code_none, code_false)
        self.assertNotEqual(code_true, code_false)
        # Check that flatten parameter appears in tl.range call
        self.assertNotIn("flatten", code_none)
        self.assertIn("flatten=True", code_true)
        self.assertIn("flatten=False", code_false)

    def test_static_range_2d(self):
        @helion.kernel()
        def nested_loop_kernel_2d(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            # The return value of hl.specialized is a LiteralType and thus a tl.constexpr.
            # TODO(joydddd): support static_range in for tile_m in hl.tile([x.size(1)])
            M = hl.specialize(x.size(1))
            N = hl.specialize(x.size(2))
            # Outer loop becomes grid (no tl.range)
            for tile_outer in hl.tile(x.size(0)):
                # Inner loop becomes device loop with tl.range / tl.static_range
                # Specialize on x.size(1) to allow range_staitic
                for tile_m, tile_n in hl.tile([M, N]):
                    out[tile_outer, tile_m, tile_n] = x[tile_outer, tile_m, tile_n] + 1
            return out

        args = (torch.randn([64, 32, 4], device=DEVICE),)

        # Test with static_ranges = [True] (use tl.static_range for device loop)
        code_true, result_true = code_and_output(
            nested_loop_kernel_2d, args, block_sizes=[16, 16, 1], static_ranges=[True]
        )

        # Test with static_ranges = [False] (use tl.range for device loop)
        code_false, result_false = code_and_output(
            nested_loop_kernel_2d, args, block_sizes=[16, 16, 1], static_ranges=[False]
        )

        # Test default
        code_default, result_default = code_and_output(
            nested_loop_kernel_2d, args, block_sizes=[16, 16, 1]
        )

        # Ignore range kwargs when static_range is set to True.
        code_ignore, result_ignore = code_and_output(
            nested_loop_kernel_2d,
            args,
            block_sizes=[16, 16, 1],
            static_ranges=[True],
            range_unroll_factors=[2],
            range_num_stages=[3],
            range_multi_buffers=[True],
            range_flattens=[True],
        )

        torch.testing.assert_close(result_false, result_true)
        torch.testing.assert_close(result_true, args[0] + 1)
        self.assertEqual(code_default, code_false)
        self.assertEqual(code_ignore, code_true)
        self.assertNotEqual(code_true, code_false)
        # Check that tl.range / tl.static_range is used according to setups.
        self.assertIn("tl.range", code_false)
        self.assertIn("tl.static_range", code_true)

    def test_static_range_scalar(self):
        @helion.kernel()
        def nested_loop_kernel_scalar(x: torch.Tensor) -> torch.Tensor:
            world_size = 4
            # Outer loop becomes grid (no tl.range)
            for tile_outer in hl.tile(x.size(0)):
                # Inner loop becomes device loop with tl.range / tl.static_range
                # Specialize on x.size(1) to allow range_staitic
                for _rank in range(world_size):
                    x[tile_outer] = x[tile_outer] + 1
            return x

        x = torch.randn([64], device=DEVICE)

        # Test with static_ranges = [True] (use tl.static_range for device loop)
        code_true, result_true = code_and_output(
            nested_loop_kernel_scalar,
            (x.clone(),),
            block_sizes=[16],
            static_ranges=[True],
        )

        # Test with static_ranges = [False] (use tl.range for device loop)
        code_false, result_false = code_and_output(
            nested_loop_kernel_scalar,
            (x.clone(),),
            block_sizes=[16],
            static_ranges=[False],
        )

        # Test default
        code_default, result_default = code_and_output(
            nested_loop_kernel_scalar,
            (x.clone(),),
            block_sizes=[
                16,
            ],
        )

        torch.testing.assert_close(result_default, result_true)
        torch.testing.assert_close(result_default, result_false)
        torch.testing.assert_close(result_default, x + 4)
        self.assertNotEqual(code_default, code_true)
        self.assertNotEqual(code_true, code_false)
        self.assertEqual(code_default, code_false)
        # Check that tl.range / tl.static_range is used according to setups.
        self.assertIn("tl.range", code_false)
        self.assertIn("tl.static_range", code_true)

    @unittest.skip("TODO(joydddd): handle constexpr type casting.")
    def test_static_range_casting(self):
        @helion.kernel()
        def nested_loop_kernel_w_casting(x: torch.Tensor) -> torch.Tensor:
            world_size = 4
            # Outer loop becomes grid (no tl.range)
            for tile_outer in hl.tile(x.size(0)):
                # Inner loop becomes device loop with tl.range / tl.static_range
                # Specialize on x.size(1) to allow range_staitic
                for rank in range(world_size):
                    x[tile_outer] = x[tile_outer] + rank
            return x

        x = torch.randn([64], device=DEVICE)

        # Test with static_ranges = [True] (use tl.static_range for device loop)
        code, result = code_and_output(
            nested_loop_kernel_w_casting,
            (x.clone(),),
            block_sizes=[16],
            static_ranges=[True],
        )

        torch.testing.assert_close(result, x + 5)
        self.assertIn("tl.static_range", code)


if __name__ == "__main__":
    unittest.main()
