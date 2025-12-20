from __future__ import annotations

from typing import Callable
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfCpu
import helion.language as hl


@skipIfCpu("needs to be debugged")
class TestRNG(RefEagerTestBase, TestCase):
    def test_rand(self):
        """Test RNG seeding behavior, reproducibility, output range, and distribution."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def rand_kernel_tiled_2d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
            return output

        # Test with different tensor sizes for different aspects
        x_small = torch.ones(32, 32, device=DEVICE)  # For distribution tests
        x_large = torch.ones(64, 64, device=DEVICE)  # For seeding tests

        # Test 1: Different seeds produce different outputs
        torch.manual_seed(42)
        _code1, output1 = code_and_output(rand_kernel_tiled_2d, (x_large,))

        torch.manual_seed(123)
        _code2, output2 = code_and_output(rand_kernel_tiled_2d, (x_large,))

        self.assertFalse(
            torch.allclose(output1, output2),
            "Different seeds should produce different outputs",
        )

        # Test 2: Same seed produces identical outputs (reproducibility)
        torch.manual_seed(42)
        _code3, output3 = code_and_output(rand_kernel_tiled_2d, (x_large,))

        torch.testing.assert_close(
            output1, output3, msg="Same seed should produce identical outputs"
        )

        # Test 3: RNG state advances between calls
        torch.manual_seed(42)
        _code4, output4 = code_and_output(rand_kernel_tiled_2d, (x_large,))
        # No manual_seed here - RNG state should advance
        _code5, output5 = code_and_output(rand_kernel_tiled_2d, (x_large,))

        self.assertFalse(
            torch.allclose(output4, output5),
            "Sequential calls should produce different outputs (RNG state advanced)",
        )

        # Test 4: Output range and distribution properties
        torch.manual_seed(42)
        _code6, output6 = code_and_output(rand_kernel_tiled_2d, (x_small,))

        # All values should be in [0, 1) range
        self.assertTrue(torch.all(output6 >= 0.0), "All values should be >= 0")
        self.assertTrue(torch.all(output6 < 1.0), "All values should be < 1")

        # Check distribution properties
        mean_val = output6.mean().item()
        self.assertTrue(
            0.4 < mean_val < 0.6,
            f"Mean {mean_val:.3f} should be around 0.5 for uniform distribution",
        )

        # Check spread of values
        min_val = output6.min().item()
        max_val = output6.max().item()
        self.assertTrue(
            min_val < 0.2, f"Min value {min_val:.3f} should be < 0.2 for good spread"
        )
        self.assertTrue(
            max_val > 0.8, f"Max value {max_val:.3f} should be > 0.8 for good spread"
        )

    def test_rand_3d_tensor(self):
        """Test 3D RNG with tiled operations."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def rand_kernel_3d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            b, m, n = x.shape
            for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
                output[tile_b, tile_m, tile_n] = torch.rand_like(
                    x[tile_b, tile_m, tile_n]
                )
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
        print(
            f"3D Unique values: {unique_values}, Total: {total_values}, Percentage: {uniqueness_ratio * 100:.2f}%"
        )

        # Expect at least 95% unique values for good 3D RNG
        self.assertGreater(uniqueness_ratio, 0.95)

        # Check distribution across dimensions
        # Mean should be around 0.5 for each 2D slice
        for b_idx in range(x.shape[0]):
            slice_mean = output[b_idx].mean().item()
            self.assertTrue(
                0.35 < slice_mean < 0.65,
                f"Slice {b_idx} mean {slice_mean} is not well distributed",
            )

        # Verify different seeds produce different results
        torch.manual_seed(88)
        _code2, output2 = code_and_output(rand_kernel_3d, (x,))
        self.assertFalse(torch.allclose(output, output2))

    def test_multiple_rng_ops(self):
        """Test multiple RNG operations: independence, reproducibility, mixed rand/randn."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def multiple_rng_ops_kernel(
            x: torch.Tensor,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            # Two independent rand operations
            rand1 = torch.zeros_like(x)
            rand2 = torch.zeros_like(x)

            # Mixed rand and randn
            uniform = torch.zeros_like(x)
            normal = torch.zeros_like(x)

            # Multiple randn for sum
            randn_a = torch.zeros_like(x)
            randn_b = torch.zeros_like(x)
            randn_c = torch.zeros_like(x)

            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                # Two independent rand operations
                rand1[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
                rand2[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])

                # Mixed rand and randn
                uniform[tile_m, tile_n] = torch.rand_like(x[tile_m, tile_n])
                normal[tile_m, tile_n] = torch.randn_like(x[tile_m, tile_n])

                # Multiple randn
                randn_a[tile_m, tile_n] = torch.randn_like(x[tile_m, tile_n])
                randn_b[tile_m, tile_n] = torch.randn_like(x[tile_m, tile_n])
                randn_c[tile_m, tile_n] = torch.randn_like(x[tile_m, tile_n])

            # Combine the three randn outside the loop
            randn_sum = randn_a + randn_b + randn_c

            return rand1, rand2, uniform, normal, randn_sum

        x = torch.ones(64, 64, device=DEVICE)

        # Test 1: Independence and distribution properties
        torch.manual_seed(42)
        _code1, (rand1, rand2, uniform, normal, randn_sum) = code_and_output(
            multiple_rng_ops_kernel, (x,)
        )

        # Check two independent rand operations
        self.assertTrue(
            torch.all(rand1 >= 0.0) and torch.all(rand1 < 1.0),
            "First rand output should be in [0, 1)",
        )
        self.assertTrue(
            torch.all(rand2 >= 0.0) and torch.all(rand2 < 1.0),
            "Second rand output should be in [0, 1)",
        )
        self.assertFalse(
            torch.allclose(rand1, rand2),
            "Two independent RNG ops should produce different outputs",
        )
        self.assertTrue(
            0.45 < rand1.mean().item() < 0.55,
            f"First rand mean {rand1.mean().item():.3f} should be ~0.5",
        )
        self.assertTrue(
            0.45 < rand2.mean().item() < 0.55,
            f"Second rand mean {rand2.mean().item():.3f} should be ~0.5",
        )

        # Check mixed rand and randn
        self.assertTrue(
            torch.all(uniform >= 0.0) and torch.all(uniform < 1.0),
            "Uniform (rand) values should be in [0, 1)",
        )
        self.assertTrue(
            0.4 < uniform.mean().item() < 0.6,
            f"Uniform mean {uniform.mean().item():.3f} should be ~0.5",
        )
        self.assertTrue(
            -0.2 < normal.mean().item() < 0.2,
            f"Normal mean {normal.mean().item():.3f} should be ~0",
        )
        self.assertTrue(
            0.9 < normal.std().item() < 1.1,
            f"Normal std {normal.std().item():.3f} should be ~1",
        )
        self.assertTrue(
            torch.any(normal < 0.0), "Normal distribution should have negative values"
        )
        self.assertFalse(
            torch.allclose(uniform, normal),
            "Uniform and normal distributions should be different",
        )

        # Check sum of multiple randn
        expected_std = 3**0.5
        mean = randn_sum.mean().item()
        std = randn_sum.std().item()
        self.assertTrue(-0.2 < mean < 0.2, f"Combined mean {mean:.3f} should be ~0")
        self.assertTrue(
            expected_std * 0.9 < std < expected_std * 1.1,
            f"Combined std {std:.3f} should be ~{expected_std:.3f}",
        )

        # Test 2: Reproducibility with same seed
        torch.manual_seed(42)
        _code2, outputs_a = code_and_output(multiple_rng_ops_kernel, (x,))

        torch.manual_seed(42)
        _code3, outputs_b = code_and_output(multiple_rng_ops_kernel, (x,))

        # All outputs should be identical with same seed
        for i, (a, b) in enumerate(zip(outputs_a, outputs_b, strict=False)):
            torch.testing.assert_close(
                a, b, msg=f"Output {i} should be identical with same seed"
            )

        # Verify generated code with expected journal
        self.assertExpectedJournal(_code1)

    def test_randn_different_seeds_tiled(self):
        """Test that different torch.manual_seed values produce different outputs for randn."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def randn_kernel_tiled_2d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = torch.randn_like(x[tile_m, tile_n])
            return output

        x = torch.ones(64, 64, device=DEVICE)

        torch.manual_seed(42)
        _code1, output1 = code_and_output(randn_kernel_tiled_2d, (x,))

        torch.manual_seed(123)
        _code2, output2 = code_and_output(randn_kernel_tiled_2d, (x,))

        # Different seeds should produce different outputs
        self.assertFalse(torch.allclose(output1, output2))

    def test_randn_normal_distribution(self):
        """Test that torch.randn_like produces normal distribution (mean≈0, std≈1)."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def randn_kernel_tiled_2d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = torch.randn_like(x[tile_m, tile_n])
            return output

        x = torch.ones(128, 128, device=DEVICE)  # 16384 samples for better statistics
        torch.manual_seed(42)
        _code, output = code_and_output(randn_kernel_tiled_2d, (x,))

        # Check mean is close to 0
        mean = output.mean().item()
        self.assertTrue(-0.1 < mean < 0.1, f"Mean {mean} is not close to 0")

        # Check std is close to 1
        std = output.std().item()
        self.assertTrue(0.95 < std < 1.05, f"Std {std} is not close to 1")

        # Check we have values outside [-1, 1] (characteristic of normal distribution)
        self.assertTrue(torch.any(output < -1.0))
        self.assertTrue(torch.any(output > 1.0))

        # Roughly 68% should be within 1 std
        within_1_std = (
            torch.logical_and(output > -1.0, output < 1.0).float().mean().item()
        )
        self.assertTrue(
            0.63 < within_1_std < 0.73, f"Values within 1 std: {within_1_std}"
        )

    def test_randn_3d_tensor(self):
        """Test 3D randn with tiled operations."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def randn_kernel_3d(x: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)
            b, m, n = x.shape
            for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
                output[tile_b, tile_m, tile_n] = torch.randn_like(
                    x[tile_b, tile_m, tile_n]
                )
            return output

        x = torch.ones(8, 32, 64, device=DEVICE)  # 3D tensor
        torch.manual_seed(77)
        _code, output = code_and_output(randn_kernel_3d, (x,))

        # Check overall distribution
        mean = output.mean().item()
        std = output.std().item()
        self.assertTrue(-0.1 < mean < 0.1, f"3D mean {mean} not close to 0")
        self.assertTrue(0.95 < std < 1.05, f"3D std {std} not close to 1")

        # Check distribution across dimensions
        for b_idx in range(x.shape[0]):
            slice_mean = output[b_idx].mean().item()
            slice_std = output[b_idx].std().item()
            self.assertTrue(
                -0.3 < slice_mean < 0.3,
                f"Slice {b_idx} mean {slice_mean} is not well distributed",
            )
            self.assertTrue(
                0.85 < slice_std < 1.15,
                f"Slice {b_idx} std {slice_std} is not well distributed",
            )

    def _test_rng_with_dynamic_tile_sizes(self, rng_func, is_uniform, rng_name):
        """Common test logic for RNG operations with dynamic tile sizes."""

        # Single kernel that takes an RNG callable as a parameter
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def rng_kernel(
            x: torch.Tensor,
            rng_func: Callable[[int, int, torch.dtype], torch.Tensor],
        ) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = rng_func(tile_m, tile_n, x.dtype)
            return output

        x = torch.ones(64, 64, device=DEVICE)
        torch.manual_seed(42)
        _code, output = code_and_output(rng_kernel, (x, rng_func))

        # Check distribution properties based on RNG type
        if is_uniform:
            # For rand: values in [0, 1), mean ~0.5
            self.assertTrue(
                torch.all(output >= 0.0), f"{rng_name}: All values should be >= 0"
            )
            self.assertTrue(
                torch.all(output < 1.0), f"{rng_name}: All values should be < 1"
            )
            mean_val = output.mean().item()
            self.assertTrue(
                0.4 < mean_val < 0.6,
                f"{rng_name}: Mean {mean_val:.3f} should be ~0.5",
            )
        else:
            # For randn: mean ~0, std ~1
            mean_val = output.mean().item()
            std_val = output.std().item()
            self.assertTrue(
                -0.15 < mean_val < 0.15, f"{rng_name}: Mean {mean_val:.3f} should be ~0"
            )
            self.assertTrue(
                0.9 < std_val < 1.1, f"{rng_name}: Std {std_val:.3f} should be ~1"
            )

        # Test reproducibility with same seed
        torch.manual_seed(42)
        _code2, output2 = code_and_output(rng_kernel, (x, rng_func))
        torch.testing.assert_close(
            output,
            output2,
            msg=f"{rng_name}: Same seed should produce identical outputs",
        )

        # Test that different seeds produce different outputs
        torch.manual_seed(99)
        _code3, output3 = code_and_output(rng_kernel, (x, rng_func))
        self.assertFalse(
            torch.allclose(output, output3),
            f"{rng_name}: Different seeds should produce different outputs",
        )

    def test_rand_with_dynamic_tile_sizes(self):
        """Test torch.rand with dynamic tile dimensions."""
        self._test_rng_with_dynamic_tile_sizes(
            rng_func=lambda tile_m, tile_n, dtype: torch.rand(
                (tile_m, tile_n), dtype=dtype, device=DEVICE
            ),
            is_uniform=True,
            rng_name="rand",
        )

    def test_rand_like_with_dynamic_tile_sizes(self):
        """Test torch.rand_like with dynamic tile dimensions."""
        self._test_rng_with_dynamic_tile_sizes(
            rng_func=lambda tile_m, tile_n, dtype: torch.rand_like(
                torch.ones((tile_m, tile_n), dtype=dtype, device=DEVICE)
            ),
            is_uniform=True,
            rng_name="rand_like",
        )

    def test_randn_with_dynamic_tile_sizes(self):
        """Test torch.randn with dynamic tile dimensions."""
        self._test_rng_with_dynamic_tile_sizes(
            rng_func=lambda tile_m, tile_n, dtype: torch.randn(
                (tile_m, tile_n), dtype=dtype, device=DEVICE
            ),
            is_uniform=False,
            rng_name="randn",
        )

    def test_randn_like_with_dynamic_tile_sizes(self):
        """Test torch.randn_like with dynamic tile dimensions."""
        self._test_rng_with_dynamic_tile_sizes(
            rng_func=lambda tile_m, tile_n, dtype: torch.randn_like(
                torch.ones((tile_m, tile_n), dtype=dtype, device=DEVICE)
            ),
            is_uniform=False,
            rng_name="randn_like",
        )

    def test_rand_like_with_specialized_dimension(self):
        """Test torch.rand_like with specialized (constant) dimensions."""

        @helion.kernel(config=helion.Config(block_sizes=[64, 128]))
        def matmul_with_rand(
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            # Specialize n to make it a constant dimension
            n = hl.specialize(n)

            out = torch.empty(
                [m, n],
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    mm = torch.matmul(x[tile_m, tile_k], y[tile_k, :])
                    acc = acc + mm
                # This rand_like has shape [tile_m, n] where:
                # - tile_m is a block dimension
                # - n is a specialized (constant) dimension
                noise = torch.rand_like(acc, dtype=torch.float32)
                acc = acc + noise * 0.01  # Small noise
                out[tile_m, :] = acc.to(out.dtype)
            return out

        m, k, n = 256, 512, 64
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float16)

        torch.manual_seed(42)
        code, result = code_and_output(matmul_with_rand, (x, y))

        # Verify the output shape
        self.assertEqual(result.shape, (m, n))

        # Verify reproducibility
        torch.manual_seed(42)
        _code2, result2 = code_and_output(matmul_with_rand, (x, y))
        torch.testing.assert_close(result, result2)

        # Verify different seeds produce different results
        torch.manual_seed(123)
        _code3, result3 = code_and_output(matmul_with_rand, (x, y))
        self.assertFalse(torch.allclose(result, result3))

        # Verify generated code
        self.assertExpectedJournal(code)

    def test_rand_like_nested_tiles_issue_1208(self):
        """Test torch.rand_like with nested tiles (regression test for issue #1208).

        This test reproduces the bug where torch.rand_like() failed with nested tiles
        because the RNG codegen incorrectly used dimension indices instead of block_ids
        when constructing index variable names.
        """

        @helion.kernel(
            autotune_effort="none",
            static_shapes=True,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        def nested_tiles_rand(q: torch.Tensor) -> torch.Tensor:
            B, T, H = q.shape
            out = torch.empty((B, T, H), device=q.device, dtype=q.dtype)

            for tile_b, tile_q in hl.tile([B, T]):
                qs = q[tile_b, tile_q, :]
                for tile_k in hl.tile(T):
                    ks = q[tile_b, tile_k, :]
                    # logits has shape [tile_b, tile_q, tile_k]
                    # The third dimension uses indices_3 (from the inner loop)
                    # not indices_2 (from H dimension)
                    logits = qs @ ks.transpose(-1, -2)

                    # This used to fail because rand_like incorrectly used
                    # indices_2 (size H=32) instead of indices_3 (size tile_k=16)
                    rand = torch.rand_like(logits)

                    mask = ((logits + rand) > 0).float()
                    out[tile_b, tile_q, :] = torch.matmul(mask, q[tile_b, tile_q, :])

            return out

        q = torch.randn(2, 16, 32, device=DEVICE, dtype=torch.float32)
        torch.manual_seed(42)
        code, result = code_and_output(nested_tiles_rand, (q,))

        # Verify output shape
        self.assertEqual(result.shape, (2, 16, 32))

        # Verify reproducibility
        torch.manual_seed(42)
        _code2, result2 = code_and_output(nested_tiles_rand, (q,))
        torch.testing.assert_close(result, result2)

        # Verify different seeds produce different results
        torch.manual_seed(123)
        _code3, result3 = code_and_output(nested_tiles_rand, (q,))
        self.assertFalse(torch.allclose(result, result3))

        # Verify generated code
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
