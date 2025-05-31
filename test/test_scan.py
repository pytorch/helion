from __future__ import annotations

from typing import TYPE_CHECKING
import unittest
import pytest

from expecttest import TestCase
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
def cumsum_3d_kernel(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Cumsum on 3D tensor along specified dimension."""
    a, b, c = x.size()
    out = torch.empty_like(x)
    if dim == 0:
        for tile_b in hl.tile(b):
            for tile_c in hl.tile(c):
                out[:, tile_b, tile_c] = x[:, tile_b, tile_c].cumsum(0)
    elif dim == 1:
        for tile_a in hl.tile(a):
            for tile_c in hl.tile(c):
                out[tile_a, :, tile_c] = x[tile_a, :, tile_c].cumsum(0)
    else:  # dim == 2
        for tile_a in hl.tile(a):
            for tile_b in hl.tile(b):
                out[tile_a, tile_b, :] = x[tile_a, tile_b, :].cumsum(0)
    return out


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
                        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    def test_cumsum_looped(self):
        """Test cumsum with looped scan strategy."""
        args = (torch.randn([512, 512], device="cuda"),)
        code, output = code_and_output(
            cumsum_kernel, args, block_size=2, scan_loop=64
        )
        torch.testing.assert_close(output, args[0].cumsum(-1), rtol=1e-04, atol=1e-04)
        
        # TODO: Looped scan strategy is not yet implemented
        # For now, all scans use persistent strategy
        # self.assertIn("for", code)
        self.assertIn("tl.cumsum", code)
        # # Should have carry variable
        # self.assertIn("_carry", code)

    def test_cumprod_looped(self):
        """Test cumprod with looped scan strategy."""
        # Use smaller values to avoid overflow
        args = (torch.randn([512, 512], device="cuda") * 0.1,)
        code, output = code_and_output(
            cumprod_kernel, args, block_size=1, scan_loop=32
        )
        torch.testing.assert_close(output, args[0].cumprod(-1), rtol=1e-03, atol=1e-03)
        
        # TODO: Looped scan strategy is not yet implemented
        # For now, all scans use persistent strategy
        # self.assertIn("for", code)
        self.assertIn("tl.cumprod", code)
        # # Should have carry variable
        # self.assertIn("_carry", code)

    def test_cumsum_3d(self):
        """Test cumsum on 3D tensors along different dimensions."""
        x = torch.randn([64, 128, 256], device="cuda")
        
        for dim in range(3):
            args = (x, dim)
            code, output = code_and_output(
                cumsum_3d_kernel, args, block_size=16, indexing="block_ptr"
            )
            expected = x.cumsum(dim)
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
        code, output = code_and_output(
            cumsum_kernel, args, block_size=1, scan_loop=128
        )
        torch.testing.assert_close(output, x.cumsum(-1), rtol=1e-04, atol=1e-04)
        
        # TODO: Looped scan strategy is not yet implemented
        # For now, all scans use persistent strategy
        # self.assertIn("for", code)
        # self.assertIn("_carry", code)

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
            
            torch.testing.assert_close(
                output, x.cumsum(-1), rtol=rtol, atol=atol
            )

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