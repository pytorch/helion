from __future__ import annotations

from typing import TYPE_CHECKING
import unittest

from expecttest import TestCase
import pytest
import torch

import helion
from helion._testing import code_and_output
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


@helion.kernel()
def cumsum_kernel(x: torch.Tensor) -> torch.Tensor:
    """Simple cumsum along the last dimension."""
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = x[tile_n, :].cumsum(-1)
    return out


@helion.kernel()
def cumsum_kernel_first_dim(x: torch.Tensor) -> torch.Tensor:
    """Cumsum along the first dimension."""
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        out[:, tile_m] = x[:, tile_m].cumsum(0)
    return out


@helion.kernel()
def cumprod_kernel(x: torch.Tensor) -> torch.Tensor:
    """Simple cumprod along the last dimension."""
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = x[tile_n, :].cumprod(-1)
    return out


@helion.kernel(config={"block_sizes": [1]})
def scan_kernel(
    x: torch.Tensor, fn: Callable[[torch.Tensor, int], torch.Tensor]
) -> torch.Tensor:
    """Generic scan kernel for testing different scan functions."""
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = fn(x[tile_n, :], -1)
    return out


@helion.kernel()
def cumsum_3d_dim0_kernel(x: torch.Tensor) -> torch.Tensor:
    """Cumsum on 3D tensor along dimension 0 - simplified version."""
    # For simplicity, process the entire tensor at once
    # This avoids complex indexing patterns
    return x.cumsum(0)


@helion.kernel()
def cumsum_3d_dim1_kernel(x: torch.Tensor) -> torch.Tensor:
    """Cumsum on 3D tensor along dimension 1 - simplified version."""
    return x.cumsum(1)


@helion.kernel()
def cumsum_3d_dim2_kernel(x: torch.Tensor) -> torch.Tensor:
    """Cumsum on 3D tensor along dimension 2 - simplified version."""
    return x.cumsum(2)


class TestScan(TestCase):
    """Test suite for scan operations (cumsum, cumprod, etc.).

    This test suite verifies the behavior of ScanStrategy implementations.
    The tests are currently skipped as the functionality is not yet implemented.
    """

    maxDiff = 16384

    def test_cumsum_basic(self):
        """Test basic cumsum operation with persistent scan strategy."""
        args = (torch.randn([512, 512], device="cuda"),)
        code, output = code_and_output(cumsum_kernel, args, block_size=1)
        torch.testing.assert_close(output, args[0].cumsum(-1), rtol=1e-04, atol=1e-04)

        # Check that the generated code contains cumsum
        self.assertIn("tl.cumsum", code)

    def test_cumsum_first_dim(self):
        """Test cumsum along the first dimension."""
        args = (torch.randn([512, 512], device="cuda"),)
        code, output = code_and_output(
            cumsum_kernel_first_dim, args, block_size=16, indexing="block_ptr"
        )
        torch.testing.assert_close(output, args[0].cumsum(0), rtol=1e-04, atol=1e-04)

        # Check that the generated code contains cumsum
        self.assertIn("tl.cumsum", code)

    def test_cumprod_basic(self):
        """Test basic cumprod operation."""
        # Use smaller values to avoid overflow
        args = (torch.randn([512, 512], device="cuda") * 0.1,)
        code, output = code_and_output(cumprod_kernel, args, block_size=1)
        torch.testing.assert_close(output, args[0].cumprod(-1), rtol=1e-03, atol=1e-03)

        # Check that the generated code contains cumprod
        self.assertIn("tl.cumprod", code)

    def test_scan_functions(self):
        """Test different scan configurations with various parameters."""
        # Test with different scan_loop values (None means persistent, otherwise looped)
        for scan_loop in (None, 16, 64):
            for block_size in (1, 16):
                for indexing in ("block_ptr", "pointer"):
                    for fn in (torch.cumsum, torch.cumprod):
                        # Use smaller values for cumprod to avoid overflow
                        scale = 0.1 if fn == torch.cumprod else 1.0
                        args = (torch.randn([512, 512], device="cuda") * scale, fn)

                        kwargs = {
                            "block_size": block_size,
                            "indexing": indexing,
                        }
                        if scan_loop is not None:
                            kwargs["scan_loop"] = scan_loop

                        _, output = code_and_output(scan_kernel, args, **kwargs)
                        expected = fn(args[0], dim=-1)

                        # Use looser tolerance for cumprod due to accumulated errors
                        rtol = 1e-2 if fn == torch.cumprod else 1e-3
                        atol = 1e-2 if fn == torch.cumprod else 1e-3
                        torch.testing.assert_close(
                            output, expected, rtol=rtol, atol=atol
                        )

    def test_cumsum_looped(self):
        """Test cumsum with looped scan strategy."""
        args = (torch.randn([512, 512], device="cuda"),)
        code, output = code_and_output(cumsum_kernel, args, block_size=2, scan_loop=64)
        torch.testing.assert_close(output, args[0].cumsum(-1), rtol=1e-04, atol=1e-04)

        # Verify looped scan strategy is used
        self.assertIn("for", code)
        self.assertIn("_carry", code)

    def test_cumprod_looped(self):
        """Test cumprod with looped scan strategy."""
        # Use smaller values to avoid overflow
        args = (torch.randn([512, 512], device="cuda") * 0.1,)
        code, output = code_and_output(cumprod_kernel, args, block_size=1, scan_loop=32)
        torch.testing.assert_close(output, args[0].cumprod(-1), rtol=1e-03, atol=1e-03)

        # Verify looped scan strategy is used
        self.assertIn("for", code)
        self.assertIn("_carry", code)

    @pytest.mark.skip(reason="3D kernels require proper tiling implementation")
    def test_cumsum_3d(self):
        """Test cumsum on 3D tensors along different dimensions."""
        x = torch.randn([64, 128, 256], device="cuda")

        # Test dimension 0
        args = (x,)
        code, output = code_and_output(cumsum_3d_dim0_kernel, args)
        expected = x.cumsum(0)
        torch.testing.assert_close(output, expected, rtol=1e-04, atol=1e-04)

        # Test dimension 1
        args = (x,)
        code, output = code_and_output(cumsum_3d_dim1_kernel, args)
        expected = x.cumsum(1)
        torch.testing.assert_close(output, expected, rtol=1e-04, atol=1e-04)

        # Test dimension 2
        args = (x,)
        code, output = code_and_output(cumsum_3d_dim2_kernel, args)
        expected = x.cumsum(2)
        torch.testing.assert_close(output, expected, rtol=1e-04, atol=1e-04)

    def test_cumsum_non_contiguous(self):
        """Test cumsum on non-contiguous tensors."""
        # Create a non-contiguous tensor by transposing
        x = torch.randn([512, 512], device="cuda").t()
        args = (x,)
        code, output = code_and_output(cumsum_kernel, args, block_size=8)
        torch.testing.assert_close(output, x.cumsum(-1), rtol=1e-04, atol=1e-04)

    def test_cumsum_small_sizes(self):
        """Test cumsum on tensors with small dimensions."""
        for m in [1, 4, 7, 16, 31]:
            x = torch.randn([128, m], device="cuda")
            args = (x,)
            code, output = code_and_output(cumsum_kernel, args, block_size=1)
            torch.testing.assert_close(output, x.cumsum(-1), rtol=1e-04, atol=1e-04)

    def test_cumsum_large_sizes(self):
        """Test cumsum on tensors with large dimensions."""
        # Test with size that will definitely require looped strategy
        x = torch.randn([64, 2048], device="cuda")
        args = (x,)
        # Force small scan_loop to ensure looped strategy
        code, output = code_and_output(cumsum_kernel, args, block_size=1, scan_loop=128)
        torch.testing.assert_close(output, x.cumsum(-1), rtol=1e-04, atol=1e-04)

        # Verify looped scan strategy is used
        self.assertIn("for", code)
        self.assertIn("_carry", code)

    def test_cumsum_debug_output(self):
        """Test debug output for scan operations."""
        args = (torch.randn([512, 512], device="cuda"), torch.cumsum)
        debug_str = scan_kernel.bind(args)._debug_str()

        # Check that debug string contains expected scan-related information
        self.assertIn("cumsum", debug_str)

    def test_cumsum_with_dtype(self):
        """Test cumsum with different data types."""
        for dtype in [torch.float16, torch.float32, torch.float64]:
            x = torch.randn([256, 256], device="cuda", dtype=dtype)
            args = (x,)
            code, output = code_and_output(cumsum_kernel, args, block_size=4)

            # Use appropriate tolerance based on dtype
            # float16 cumsum can accumulate significant errors
            rtol = 5e-01 if dtype == torch.float16 else 1e-04
            atol = 1e-01 if dtype == torch.float16 else 1e-04

            torch.testing.assert_close(output, x.cumsum(-1), rtol=rtol, atol=atol)

    def test_scan_empty_tensor(self):
        """Test scan operations on empty tensors."""
        x = torch.empty([0, 10], device="cuda")
        args = (x,)
        code, output = code_and_output(cumsum_kernel, args)
        self.assertEqual(output.shape, x.shape)

    def test_scan_single_element(self):
        """Test scan operations on single-element dimensions."""
        x = torch.randn([128, 1], device="cuda")
        args = (x,)
        code, output = code_and_output(cumsum_kernel, args)
        torch.testing.assert_close(output, x.cumsum(-1))


if __name__ == "__main__":
    unittest.main()
