from __future__ import annotations

import contextlib
import io
import math
import unittest

import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import assert_ref_eager_mode
import helion.language as hl


class TestRefEagerMisc(TestCase):
    def test_print_intermediate_tensor(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def print_intermediate_tensor_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                x_val = x[tile_m, tile_n]
                y_val = y[tile_m, tile_n]
                sum_val = x_val + y_val
                print("x: ", x_val)
                print("y: ", y_val)
                print("sum: ", sum_val)
                out[tile_m, tile_n] = sum_val
            return out

        x = torch.ones([2, 2], device=DEVICE, dtype=torch.float32) * 10.0
        y = torch.ones([2, 2], device=DEVICE, dtype=torch.float32) * 5.0
        expected = x + y

        # Capture stdout to check print output
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            result = print_intermediate_tensor_kernel(x, y)

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

        # Check that the print statements produced output
        output = captured_output.getvalue()
        self.assertIn("x: ", output)
        self.assertIn("y: ", output)
        self.assertIn("sum: ", output)
        self.assertIn("[[10., 10.]", output)  # x values
        self.assertIn("[[5., 5.]", output)  # y values
        self.assertIn("[[15., 15.]", output)  # sum values

    def test_print_in_invalid_helion_kernel(self):
        """Test that print works even in invalid Helion kernels in ref eager mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def incorrect_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                val = x[tile_m, tile_n]
                print("processing tile: ", val)
                # `pass` below causes this kernel to be invalid.
                # But we show that in ref-eager mode, the `print` statement above still works,
                # which is useful for debugging.
                pass  # noqa: PIE790
            return x

        x = torch.ones([2, 2], device=DEVICE, dtype=torch.float32) * math.pi

        # Capture stdout to check print output
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            _ = incorrect_kernel(x)

        # Check that the print statement produced output
        output = captured_output.getvalue()
        self.assertIn("processing tile: ", output)
        self.assertIn("[[3.14", output)  # The value printed

    def test_ref_eager_kernel_config(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
            return out

        with assert_ref_eager_mode():
            x = torch.randn(128, 128, device=DEVICE)
            result = kernel(x)
            expected = x * 2.0
            torch.testing.assert_close(result, expected)

    def test_block_size_support(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n], block_size=2):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
            return out

        with assert_ref_eager_mode():
            x = torch.randn(128, 128, device=DEVICE)
            result = kernel(x)
            expected = x * 2.0
            torch.testing.assert_close(result, expected)

    def test_tile_begin_with_block_size_1(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.empty_like(x)
            for tile in hl.tile(n, block_size=1):
                out[tile] = x[tile] + tile.begin
            return out

        with assert_ref_eager_mode():
            x = torch.zeros(8, device=DEVICE)
            result = kernel(x)
            expected = torch.arange(8, device=DEVICE, dtype=torch.float32)
            torch.testing.assert_close(result, expected)

    def test_tile_broadcast_shape_mismatch(self):
        """Test that implicit broadcasting between different tile dims raises ShapeMismatch."""

        def invalid_broadcast_kernel_impl(x: torch.Tensor) -> torch.Tensor:
            M, N = x.shape
            out = torch.zeros_like(x)
            for row_tile, col_tile in hl.tile([M, N]):
                row_idx = row_tile.index  # shape [tile0]
                col_idx = col_tile.index  # shape [tile1]
                # Create 2D grid of differences
                diff = row_idx[:, None] - col_idx  # shape [tile0, tile1]
                # BUG: row_idx has shape [tile0], diff has shape [tile0, tile1]
                # PyTorch right-aligns: [tile0] -> [1, tile0]
                # Then broadcasts [1, tile0] with [tile0, tile1], which incorrectly
                # aligns tile0 dimension with tile1 dimension
                result = (row_idx > 0) & (diff >= 0)
                out[row_tile, col_tile] = result.float()
            return out

        # Test ref mode
        ref_kernel = helion.kernel(ref_mode=helion.RefMode.EAGER)(
            invalid_broadcast_kernel_impl
        )
        with assert_ref_eager_mode():
            x = torch.randn(64, 64, device=DEVICE)
            with self.assertRaises(exc.ShapeMismatch) as cm:
                ref_kernel(x)

            # Verify error message format matches codegen style with symbolic names
            error_msg = str(cm.exception)
            self.assertIn("Shape mismatch between", error_msg)
            # Should contain symbolic shape names like [1, u0] and [u0, u1]
            self.assertIn("[1, u0]", error_msg)
            self.assertIn("[u0, u1]", error_msg)
            # Should contain stack trace pointing to the exact line
            self.assertRegex(
                error_msg,
                r"test_ref_eager\.py:\d+: result = \(row_idx > 0\) & \(diff >= 0\)",
            )

        # Verify compile mode also raises the same error (wrapped in InvalidConfig)
        compile_kernel = helion.kernel(invalid_broadcast_kernel_impl)
        x = torch.randn(64, 64, device=DEVICE)
        with self.assertRaises((exc.ShapeMismatch, exc.InvalidConfig)) as cm:
            compile_kernel(x)
        # Check that ShapeMismatch is the root cause with expected message
        error = cm.exception
        if isinstance(error, exc.InvalidConfig):
            self.assertIsInstance(error.__cause__, exc.ShapeMismatch)
            error = error.__cause__
        error_msg = str(error)
        self.assertIn("[1, u0]", error_msg)
        self.assertIn("[u0, u1]", error_msg)

    def test_reduction_broadcast_shape_mismatch(self):
        """Test that reduction followed by broadcast with different tile dims fails."""

        def invalid_reduction_kernel_impl(x: torch.Tensor) -> torch.Tensor:
            M, N = x.shape
            out = torch.zeros_like(x)
            for tile_m, tile_n in hl.tile([M, N]):
                data = x[tile_m, tile_n]  # [tile0, tile1]
                row_sum = data.sum(dim=1)  # [tile0]
                col_sum = data.sum(dim=0)  # [tile1]
                # [tile0] + [tile1] - improper broadcast between different tiles
                combined = row_sum + col_sum
                out[tile_m, tile_n] = combined.unsqueeze(-1).expand(
                    -1, data.shape[1]
                )
            return out

        # Test ref mode
        ref_kernel = helion.kernel(ref_mode=helion.RefMode.EAGER)(
            invalid_reduction_kernel_impl
        )
        with assert_ref_eager_mode():
            x = torch.randn(32, 32, device=DEVICE)
            with self.assertRaises(exc.ShapeMismatch) as cm:
                ref_kernel(x)

            # Verify error message format
            error_msg = str(cm.exception)
            self.assertIn("Shape mismatch between", error_msg)
            self.assertIn("[u0]", error_msg)
            self.assertIn("[u1]", error_msg)
            # Should contain stack trace pointing to the exact line
            self.assertRegex(
                error_msg, r"test_ref_eager\.py:\d+: combined = row_sum \+ col_sum"
            )

        # Verify compile mode also raises the same error (wrapped in InvalidConfig)
        compile_kernel = helion.kernel(invalid_reduction_kernel_impl)
        x = torch.randn(32, 32, device=DEVICE)
        with self.assertRaises((exc.ShapeMismatch, exc.InvalidConfig)) as cm:
            compile_kernel(x)
        # Check that ShapeMismatch is the root cause with expected message
        error = cm.exception
        if isinstance(error, exc.InvalidConfig):
            self.assertIsInstance(error.__cause__, exc.ShapeMismatch)
            error = error.__cause__
        error_msg = str(error)
        self.assertIn("[u0]", error_msg)
        self.assertIn("[u1]", error_msg)


if __name__ == "__main__":
    unittest.main()
