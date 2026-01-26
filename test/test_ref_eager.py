from __future__ import annotations

import contextlib
import io
import math
import unittest

import torch

import helion
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

    def test_store_with_duplicate_indices_raises_error(self):
        """Test that hl.store with duplicate indices raises an error in ref mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel_with_dup_store(
            out: torch.Tensor, idx: torch.Tensor, val: torch.Tensor
        ):
            mask = torch.ones_like(idx, dtype=torch.bool)
            hl.store(out, [idx], val, extra_mask=mask)

        out = torch.zeros(4, device=DEVICE)
        idx = torch.tensor(
            [0, 0, 1], device=DEVICE, dtype=torch.int64
        )  # duplicate index
        val = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)

        with self.assertRaises(helion.exc.DuplicateStoreIndicesError):
            kernel_with_dup_store(out, idx, val)

    def test_store_dtype_conversion(self):
        """Test that hl.store properly converts dtype in ref eager mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            # Output tensor is bfloat16 (same as input)
            out = torch.empty_like(x)
            out_flat = out.view(-1)

            for tile_m, tile_n in hl.tile([m, n]):
                # Accumulator is float32 for numeric precision
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                acc += x[tile_m, tile_n].to(torch.float32)

                # Compute flat indices for store
                flat_indices = tile_m.index[:, None] * n + tile_n.index[None, :]

                # Store float32 acc into bfloat16 out_flat
                # This requires dtype conversion in prepare_args
                hl.store(out_flat, [flat_indices], acc)

            return out

        with assert_ref_eager_mode():
            x = torch.randn(8, 8, device=DEVICE, dtype=torch.bfloat16)
            result = kernel(x)
            torch.testing.assert_close(
                result.to(torch.float32), x.to(torch.float32), atol=1e-2, rtol=1e-2
            )

    def test_load_2d_indexing_without_extra_mask(self):
        """Test that hl.load with two 1D tensor indices produces 2D output in ref eager mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(mask: torch.Tensor) -> torch.Tensor:
            n = mask.size(0)
            out = torch.zeros_like(mask)
            for tile_i, tile_j in hl.tile([n, n]):
                # Load with two 1D tensor indices - should produce [tile_I, tile_J] output
                vals = hl.load(mask, [tile_i.index, tile_j.index])
                out[tile_i, tile_j] = vals
            return out

        with assert_ref_eager_mode():
            mask = torch.tril(torch.ones(4, 4, device=DEVICE, dtype=torch.float32))
            result = kernel(mask)
            torch.testing.assert_close(result, mask)

    def test_load_3d_indexing_without_extra_mask(self):
        """Test that hl.load with three 1D tensor indices produces 3D output in ref eager mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            d0, d1, d2 = x.shape
            out = torch.zeros_like(x)
            for tile_i, tile_j, tile_k in hl.tile([d0, d1, d2]):
                # Load with three 1D tensor indices - should produce [tile_I, tile_J, tile_K] output
                vals = hl.load(x, [tile_i.index, tile_j.index, tile_k.index])
                out[tile_i, tile_j, tile_k] = vals
            return out

        with assert_ref_eager_mode():
            x = torch.arange(24, device=DEVICE, dtype=torch.float32).reshape(2, 3, 4)
            result = kernel(x)
            torch.testing.assert_close(result, x)


if __name__ == "__main__":
    unittest.main()
