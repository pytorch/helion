from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


class TestRand(RefEagerTestBase, TestCase):
    def test_rand_different_seeds_tiled(self):
        """Test that different torch.manual_seed values produce different outputs."""
        @helion.kernel
        def rand_kernel_tiled_2d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
            return output
        
        x = torch.ones(64, 64, device=DEVICE)
        
        torch.manual_seed(42)
        _code1, output1 = code_and_output(rand_kernel_tiled_2d, (x,))
        
        torch.manual_seed(123)
        _code2, output2 = code_and_output(rand_kernel_tiled_2d, (x,))
        
        # Different seeds should produce different outputs
        self.assertFalse(torch.allclose(output1, output2))

    def test_rand_same_seed_tiled(self):
        """Test that same torch.manual_seed values produce identical outputs."""
        @helion.kernel
        def rand_kernel_tiled_2d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
            return output
        
        x = torch.ones(64, 64, device=DEVICE)
        
        torch.manual_seed(42)
        _code1, output1 = code_and_output(rand_kernel_tiled_2d, (x,))
        
        torch.manual_seed(42)
        _code2, output2 = code_and_output(rand_kernel_tiled_2d, (x,))
        
        # Same seed should produce identical outputs
        torch.testing.assert_close(output1, output2)

    def test_rand_output_range_tiled(self):
        """Test that torch.rand_like produces values in [0, 1) range."""
        @helion.kernel
        def rand_kernel_tiled_2d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
            return output
        
        x = torch.ones(32, 32, device=DEVICE)  # 1024 total elements
        torch.manual_seed(42)
        _code, output = code_and_output(rand_kernel_tiled_2d, (x,))
        
        # All values should be in [0, 1) range
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output < 1.0))
        
        # With 1024 samples, we expect reasonable distribution
        # Check mean is around 0.5 (expected value for uniform distribution)
        self.assertTrue(0.4 < output.mean().item() < 0.6)
        
        # Check we have reasonable spread of values
        self.assertTrue(output.min().item() < 0.2)
        self.assertTrue(output.max().item() > 0.8)

    def test_rand_state_advances_tiled(self):
        """Test that RNG state advances between kernel calls."""
        @helion.kernel
        def rand_kernel_tiled_2d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
            return output
        
        x = torch.ones(64, 64, device=DEVICE)
        
        torch.manual_seed(42)
        _code1, output1 = code_and_output(rand_kernel_tiled_2d, (x,))
        # No manual_seed call here - RNG state should advance
        _code2, output2 = code_and_output(rand_kernel_tiled_2d, (x,))
        
        # Sequential calls should produce different outputs (RNG state advanced)
        self.assertFalse(torch.allclose(output1, output2))

    def test_rand_2d_non_square(self):
        """Test 2D RNG with non-square tensors."""
        @helion.kernel
        def rand_kernel_2d_rect(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
            return output
        
        x = torch.ones(128, 256, device=DEVICE)  # Non-square
        torch.manual_seed(99)
        _code, output = code_and_output(rand_kernel_2d_rect, (x,))
        
        # All values should be in [0, 1) range
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output < 1.0))
        
        # Check uniqueness - 2D should generate different values for each element
        unique_values = output.unique().numel()
        total_values = output.numel()
        
        # With a good RNG, we should have mostly unique values
        # Allow some duplicates due to quantization
        uniqueness_ratio = unique_values / total_values
        print(f"Unique values: {unique_values}, Total: {total_values}, Percentage: {uniqueness_ratio * 100:.2f}%")
        
        # Expect at least 95% unique values for good 2D RNG
        self.assertGreater(uniqueness_ratio, 0.95)
        
        # Also check that values are well-distributed across the tensor
        # Sample some values to ensure variety
        sample_indices = torch.randperm(output.numel())[:10]
        sample_values = output.flatten()[sample_indices]
        print(f"Sample values: {sample_values}")
        
        # Check we have good min/max spread
        print(f"Min: {output.min().item():.6f}, Max: {output.max().item():.6f}")
        self.assertLess(output.min().item(), 0.01)
        self.assertGreater(output.max().item(), 0.99)

    def test_rand_3d_tensor(self):
        """Test 3D RNG with tiled operations."""
        @helion.kernel
        def rand_kernel_3d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            b, m, n = x.shape
            for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
                output[tile_b, tile_m, tile_n] = torch.rand_like(x[tile_b, tile_m, tile_n])
            return output
        
        x = torch.ones(16, 32, 64, device=DEVICE)  # 3D tensor
        torch.manual_seed(77)
        _code, output = code_and_output(rand_kernel_3d, (x,))
        
        # All values should be in [0, 1) range
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output < 1.0))
        
        # Check uniqueness - 3D should generate different values for each element
        unique_values = output.unique().numel()
        total_values = output.numel()
        
        # With a good RNG, we should have mostly unique values
        uniqueness_ratio = unique_values / total_values
        print(f"3D Unique values: {unique_values}, Total: {total_values}, Percentage: {uniqueness_ratio * 100:.2f}%")
        
        # Expect at least 95% unique values for good 3D RNG
        self.assertGreater(uniqueness_ratio, 0.95)
        
        # Check distribution across dimensions
        # Mean should be around 0.5 for each 2D slice
        for b_idx in range(x.shape[0]):
            slice_mean = output[b_idx].mean().item()
            self.assertTrue(0.35 < slice_mean < 0.65, 
                          f"Slice {b_idx} mean {slice_mean} is not well distributed")
        
        # Verify different seeds produce different results
        torch.manual_seed(88)
        _code2, output2 = code_and_output(rand_kernel_3d, (x,))
        self.assertFalse(torch.allclose(output, output2))

    def test_two_independent_rng_ops(self):
        """Test two independent RNG operations in the same kernel."""
        @helion.kernel
        def two_rng_kernel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            output1 = torch.zeros_like(x)
            output2 = torch.zeros_like(x)
            
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                # First RNG operation
                output1[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
                # Second RNG operation - independent
                output2[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
            
            return output1, output2
        
        x = torch.ones(64, 64, device=DEVICE)
        torch.manual_seed(42)
        _code, (result1, result2) = code_and_output(two_rng_kernel, (x,))
        
        # Both should be in [0, 1) range
        self.assertTrue(torch.all(result1 >= 0.0))
        self.assertTrue(torch.all(result1 < 1.0))
        self.assertTrue(torch.all(result2 >= 0.0))
        self.assertTrue(torch.all(result2 < 1.0))
        
        # They should be different from each other
        self.assertFalse(torch.allclose(result1, result2))
        
        # Each should have proper distribution
        self.assertTrue(0.45 < result1.mean().item() < 0.55)
        self.assertTrue(0.45 < result2.mean().item() < 0.55)

    def test_two_rng_reproducibility(self):
        """Test that two RNG ops are reproducible with the same seed."""
        @helion.kernel
        def two_rng_kernel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            output1 = torch.zeros_like(x)
            output2 = torch.zeros_like(x)
            
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output1[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
                output2[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
            
            return output1, output2
        
        x = torch.ones(32, 32, device=DEVICE)
        
        # First run
        torch.manual_seed(42)
        _code1, (res1_a, res2_a) = code_and_output(two_rng_kernel, (x,))
        
        # Second run with same seed
        torch.manual_seed(42)
        _code2, (res1_b, res2_b) = code_and_output(two_rng_kernel, (x,))
        
        # Should be identical
        torch.testing.assert_close(res1_a, res1_b)
        torch.testing.assert_close(res2_a, res2_b)


if __name__ == "__main__":
    unittest.main()