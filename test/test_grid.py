from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


def grid_2d_pytorch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bi, bj, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        bi,
        bj,
        m,
        n,
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    for i in range(bi):
        for j in range(bj):
            out[i, j] = torch.mm(x[i, j], y)
    return out


class TestGrid(RefEagerTestBase, TestCase):
    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    @patch.object(_compat, "_min_dot_size", lambda *args: (16, 16, 16))
    def test_grid_1d(self):
        @helion.kernel(static_shapes=True)
        def grid_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            b, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for i in hl.grid(b):
                for tile_m, tile_n in hl.tile([m, n]):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = torch.addmm(acc, x[i, tile_m, tile_k], y[tile_k, tile_n])
                    out[i, tile_m, tile_n] = acc
            return out

        def grid_1d_pytorch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            b, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for i in range(b):
                out[i] = torch.mm(x[i], y)
            return out

        args = (
            torch.randn([8, 16, 32], device=DEVICE, dtype=torch.float16),
            torch.randn([32, 4], device=DEVICE, dtype=torch.float16),
        )
        code, result = code_and_output(grid_1d, args)
        torch.testing.assert_close(result, grid_1d_pytorch(args[0], args[1]))
        self.assertExpectedJournal(code)

        # test again with block_ptr indexing
        code, result = code_and_output(
            grid_1d, args, block_sizes=[16, 16, 16], indexing="tensor_descriptor"
        )
        torch.testing.assert_close(result, grid_1d_pytorch(args[0], args[1]))
        self.assertExpectedJournal(code)

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_grid_2d_idx_list(self):
        @helion.kernel(static_shapes=True)
        def grid_2d_idx_list(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            bi, bj, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                bi,
                bj,
                m,
                n,
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for i, j in hl.grid([bi, bj]):
                for tile_m, tile_n in hl.tile([m, n]):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = torch.addmm(
                            acc, x[i, j, tile_m, tile_k], y[tile_k, tile_n]
                        )
                    out[i, j, tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([3, 4, 64, 32], device=DEVICE, dtype=torch.float16),
            torch.randn([32, 16], device=DEVICE, dtype=torch.float16),
        )

        code, result = code_and_output(grid_2d_idx_list, args)
        torch.testing.assert_close(result, grid_2d_pytorch(args[0], args[1]))
        self.assertExpectedJournal(code)

        code, result = code_and_output(
            grid_2d_idx_list,
            args,
            block_sizes=[64, 32, 16],
            indexing="tensor_descriptor",
        )
        torch.testing.assert_close(result, grid_2d_pytorch(args[0], args[1]))
        self.assertExpectedJournal(code)

    def test_grid_2d_idx_nested(self):
        @helion.kernel(static_shapes=True)
        def grid_2d_idx_nested(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            bi, bj, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                bi,
                bj,
                m,
                n,
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for i in hl.grid(bi):
                for j in hl.grid(bj):
                    for tile_m, tile_n in hl.tile([m, n]):
                        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                        for tile_k in hl.tile(k):
                            acc = torch.addmm(
                                acc, x[i, j, tile_m, tile_k], y[tile_k, tile_n]
                            )
                        out[i, j, tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([3, 4, 64, 32], device=DEVICE, dtype=torch.float16),
            torch.randn([32, 16], device=DEVICE, dtype=torch.float16),
        )
        code, result = code_and_output(grid_2d_idx_nested, args)
        torch.testing.assert_close(result, grid_2d_pytorch(args[0], args[1]))
        self.assertExpectedJournal(code)

    def test_grid_begin_end(self):
        @helion.kernel(autotune_effort="none")
        def grid_begin_end(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in hl.grid(2, n - 2):  # grid(begin, end)
                out[i] = x[i] * 2
            return out

        def grid_begin_end_pytorch(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in range(2, n - 2):
                out[i] = x[i] * 2
            return out

        x = torch.randn([16], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(grid_begin_end, (x,))
        torch.testing.assert_close(result, grid_begin_end_pytorch(x))
        self.assertExpectedJournal(code)

    def test_grid_begin_end_step(self):
        @helion.kernel(autotune_effort="none")
        def grid_begin_end_step(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in hl.grid(0, n, 2):  # grid(begin, end, step)
                out[i] = x[i] * 2
            return out

        def grid_begin_end_step_pytorch(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in range(0, n, 2):
                out[i] = x[i] * 2
            return out

        x = torch.randn([16], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(grid_begin_end_step, (x,))
        torch.testing.assert_close(result, grid_begin_end_step_pytorch(x))
        self.assertExpectedJournal(code)

    def test_grid_end_step_kwarg(self):
        @helion.kernel(autotune_effort="none")
        def grid_end_step_kwarg(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in hl.grid(n, step=2):  # grid(end, step=step)
                out[i] = x[i] * 2
            return out

        def grid_end_step_kwarg_pytorch(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in range(0, n, 2):
                out[i] = x[i] * 2
            return out

        x = torch.randn([16], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(grid_end_step_kwarg, (x,))
        torch.testing.assert_close(result, grid_end_step_kwarg_pytorch(x))
        self.assertExpectedJournal(code)

    def test_grid_multidim_begin_end(self):
        @helion.kernel(autotune_effort="none")
        def grid_multidim_begin_end(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros_like(x)
            for i, j in hl.grid(
                [1, 1], [m - 1, n - 1]
            ):  # multidimensional grid(begin, end)
                out[i, j] = x[i, j] * 2
            return out

        def grid_multidim_begin_end_pytorch(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros_like(x)
            for i in range(1, m - 1):
                for j in range(1, n - 1):
                    out[i, j] = x[i, j] * 2
            return out

        x = torch.randn([8, 8], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(grid_multidim_begin_end, (x,))
        torch.testing.assert_close(result, grid_multidim_begin_end_pytorch(x))
        self.assertExpectedJournal(code)

    def test_grid_multidim_begin_end_step(self):
        @helion.kernel(autotune_effort="none")
        def grid_multidim_begin_end_step(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros_like(x)
            for i, j in hl.grid(
                [0, 0], [m, n], [2, 3]
            ):  # multidimensional grid(begin, end, step)
                out[i, j] = x[i, j] * 2
            return out

        def grid_multidim_begin_end_step_pytorch(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros_like(x)
            for i in range(0, m, 2):
                for j in range(0, n, 3):
                    out[i, j] = x[i, j] * 2
            return out

        x = torch.randn([8, 9], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(grid_multidim_begin_end_step, (x,))
        torch.testing.assert_close(result, grid_multidim_begin_end_step_pytorch(x))
        self.assertExpectedJournal(code)

    def test_tile_begin_end(self):
        @helion.kernel(autotune_effort="none")
        def tile_begin_end(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(2, 10):  # tile(begin, end) - simple range [2, 10)
                out[tile] = x[tile] * 2
            return out

        def tile_begin_end_pytorch(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            # Tile should process all indices in range [2, 10) in chunks
            for i in range(2, 10):
                out[i] = x[i] * 2
            return out

        x = torch.randn([15], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(tile_begin_end, (x,), block_size=4)
        torch.testing.assert_close(result, tile_begin_end_pytorch(x))
        self.assertExpectedJournal(code)

    def test_range_as_grid_basic(self):
        """Test that range() works as an alias for hl.grid() in device code."""

        @helion.kernel(autotune_effort="none")
        def range_kernel(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_zeros(batch)
            for tile_batch in hl.tile(batch):
                for i in range(10):  # This should work now as alias for hl.grid(10)
                    out[tile_batch] += x[tile_batch] + i
            return out

        x = torch.randn(35, device=DEVICE)

        # Reference: sum over i of (x + i) = 10*x + sum(0..9) = 10*x + 45
        expected = 10 * x + 45

        code, result = code_and_output(range_kernel, (x,))
        torch.testing.assert_close(result, expected)

    def test_range_with_begin_end(self):
        """Test that range(begin, end) works as alias for hl.grid(begin, end)."""

        @helion.kernel(autotune_effort="none")
        def range_begin_end_kernel(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_zeros(batch)
            for tile_batch in hl.tile(batch):
                for i in range(2, 7):  # range(begin, end)
                    out[tile_batch] += x[tile_batch] * i
            return out

        x = torch.randn(20, device=DEVICE)

        # Reference: x * sum(range(2, 7)) = x * sum(2,3,4,5,6) = x * 20
        expected = x * 20

        code, result = code_and_output(range_begin_end_kernel, (x,))
        torch.testing.assert_close(result, expected)

    def test_range_with_step(self):
        """Test that range(begin, end, step) works as alias for hl.grid(begin, end, step)."""

        @helion.kernel(autotune_effort="none")
        def range_step_kernel(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_zeros(batch)
            for tile_batch in hl.tile(batch):
                for i in range(1, 10, 2):  # range(begin, end, step)
                    out[tile_batch] += x[tile_batch] / i
            return out

        x = torch.randn(6, device=DEVICE)

        # Reference: x * sum(1/i for i in range(1, 10, 2)) = x * sum(1/1, 1/3, 1/5, 1/7, 1/9)
        # = x * (1 + 1/3 + 1/5 + 1/7 + 1/9) = x * sum([1, 1/3, 1/5, 1/7, 1/9])
        reciprocal_sum = sum(1.0 / i for i in range(1, 10, 2))
        expected = x * reciprocal_sum

        code, result = code_and_output(range_step_kernel, (x,))
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    def test_range_with_tensor_size(self):
        """Test that range(tensor.size(dim)) works with dynamic tensor dimensions."""

        @helion.kernel(autotune_effort="none")
        def range_tensor_size_kernel(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_zeros(batch)
            for tile_batch in hl.tile(batch):
                for _ in range(x.size(1)):  # Use tensor dimension in range
                    out[tile_batch] += x[tile_batch, 0]  # Just use first column
            return out

        x = torch.randn(8, 5, device=DEVICE)  # 8 rows, 5 columns

        # Reference: Each row adds x[row, 0] for x.size(1) times = x[:, 0] * x.size(1)
        expected = x[:, 0] * x.size(1)

        code, result = code_and_output(range_tensor_size_kernel, (x,))
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
