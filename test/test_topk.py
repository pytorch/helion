"""
Tests for torch.topk support in Helion device loops.

Limitations:
- Only dim=-1 (last dimension) supported
- k must be compile-time constant (use hl.specialize(k))
- float64/int64 not supported (use float32/int32)
- Max input size: 65536 elements per row

Supported: float32, float16, bfloat16, int32, largest=True/False, sorted=True/False
"""
from __future__ import annotations

import unittest

import torch

import helion
import helion.exc
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfCpu
import helion.language as hl


# Use block_sizes larger than tensor dimensions to test masking logic
# Use hl.constexpr for largest parameter to make it a compile-time constant
@helion.kernel(config=helion.Config(block_sizes=[64]))
def topk_2d_kernel(
    x: torch.Tensor, k: int, largest: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    n, m = x.size()
    values = torch.empty([n, k], dtype=x.dtype, device=x.device)
    indices = torch.empty([n, k], dtype=torch.int64, device=x.device)

    for tile_n in hl.tile(n):
        v, idx = torch.topk(x[tile_n, :], k, dim=-1, largest=largest)
        values[tile_n, :] = v
        indices[tile_n, :] = idx

    return values, indices


@helion.kernel(config=helion.Config(block_sizes=[64]))
def topk_unsorted_kernel(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    n, m = x.size()
    values = torch.empty([n, k], dtype=x.dtype, device=x.device)
    indices = torch.empty([n, k], dtype=torch.int64, device=x.device)

    for tile_n in hl.tile(n):
        v, idx = torch.topk(x[tile_n, :], k, dim=-1, sorted=False)
        values[tile_n, :] = v
        indices[tile_n, :] = idx

    return values, indices


def verify_topk_result(
    x: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    k: int,
    largest: bool,
    dim: int = -1,
) -> bool:
    """Verify that topk result is valid.

    Checks:
    1. Values gathered from x using indices match the returned values
    2. The returned values are the correct top-k values (sorted comparison)
    """
    # Check that gathering x at indices gives us the values
    gathered = x.gather(dim=dim, index=indices)
    if not torch.allclose(gathered, values, rtol=1e-3, atol=1e-3):
        return False

    # Check that the values are the correct top-k (order may differ for ties)
    expected_v, _ = torch.topk(x, k, dim=dim, largest=largest)
    our_sorted = values.sort(dim=dim, descending=largest)[0]
    exp_sorted = expected_v.sort(dim=dim, descending=largest)[0]

    return torch.allclose(our_sorted, exp_sorted, rtol=1e-3, atol=1e-3)


@skipIfCpu("topk requires CUDA")
class TestTopK(RefEagerTestBase, TestCase):
    """Tests for torch.topk support in Helion."""

    def test_topk_float32_largest(self):
        """Test topk with float32 dtype and largest=True."""
        torch.manual_seed(42)
        x = torch.randn([32, 64], device=DEVICE, dtype=torch.float32)

        code, (values, indices) = code_and_output(topk_2d_kernel, (x, 10, True), block_size=32)

        expected_v, expected_i = torch.topk(x, 10, dim=-1, largest=True)
        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)
        self.assertTrue(torch.equal(indices, expected_i))
        self.assertExpectedJournal(code)

    def test_topk_float32_smallest(self):
        """Test topk with float32 dtype and largest=False."""
        torch.manual_seed(42)
        x = torch.randn([32, 64], device=DEVICE, dtype=torch.float32)

        code, (values, indices) = code_and_output(topk_2d_kernel, (x, 10, False), block_size=32)

        expected_v, expected_i = torch.topk(x, 10, dim=-1, largest=False)
        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)
        self.assertTrue(torch.equal(indices, expected_i))
        self.assertExpectedJournal(code)

    def test_topk_float16(self):
        """Test topk with float16 dtype."""
        torch.manual_seed(42)
        x = torch.randn([32, 64], device=DEVICE, dtype=torch.float16)

        code, (values, indices) = code_and_output(topk_2d_kernel, (x, 10, True), block_size=32)

        # For float16, ties may have different order due to precision
        self.assertTrue(verify_topk_result(x, values, indices, 10, True))
        self.assertExpectedJournal(code)

    def test_topk_bfloat16(self):
        """Test topk with bfloat16 dtype."""
        torch.manual_seed(42)
        x = torch.randn([32, 64], device=DEVICE, dtype=torch.bfloat16)

        code, (values, indices) = code_and_output(topk_2d_kernel, (x, 10, True), block_size=32)

        # For bfloat16, ties may have different order due to precision
        self.assertTrue(verify_topk_result(x, values, indices, 10, True))
        self.assertExpectedJournal(code)

    def test_topk_int32_largest(self):
        """Test topk with int32 dtype and largest=True."""
        # Use unique values to avoid tie-breaking issues
        torch.manual_seed(42)
        x = torch.arange(256, device=DEVICE, dtype=torch.int32).view(8, 32)
        for i in range(8):
            x[i] = x[i, torch.randperm(32, device=DEVICE)]

        code, (values, indices) = code_and_output(topk_2d_kernel, (x, 5, True), block_size=32)

        expected_v, expected_i = torch.topk(x, 5, dim=-1, largest=True)
        torch.testing.assert_close(values, expected_v)
        self.assertTrue(torch.equal(indices, expected_i))
        self.assertExpectedJournal(code)

    def test_topk_int32_smallest(self):
        """Test topk with int32 dtype and largest=False."""
        # Use unique values to avoid tie-breaking issues
        torch.manual_seed(42)
        x = torch.arange(256, device=DEVICE, dtype=torch.int32).view(8, 32)
        for i in range(8):
            x[i] = x[i, torch.randperm(32, device=DEVICE)]

        code, (values, indices) = code_and_output(topk_2d_kernel, (x, 5, False), block_size=32)

        expected_v, expected_i = torch.topk(x, 5, dim=-1, largest=False)
        torch.testing.assert_close(values, expected_v)
        self.assertTrue(torch.equal(indices, expected_i))
        self.assertExpectedJournal(code)

    def test_topk_k1(self):
        """Test topk with k=1."""
        torch.manual_seed(42)
        x = torch.randn([16, 64], device=DEVICE)

        code, (values, indices) = code_and_output(topk_2d_kernel, (x, 1, True), block_size=32)

        expected_v, _ = torch.topk(x, 1, dim=-1, largest=True)
        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)
        self.assertExpectedJournal(code)

    def test_topk_k16(self):
        """Test topk with k=16 (power of 2)."""
        torch.manual_seed(42)
        x = torch.randn([16, 64], device=DEVICE)

        code, (values, indices) = code_and_output(topk_2d_kernel, (x, 16, True), block_size=32)

        expected_v, _ = torch.topk(x, 16, dim=-1, largest=True)
        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)
        self.assertExpectedJournal(code)

    def test_topk_k7(self):
        """Test topk with k=7 (non-power of 2)."""
        torch.manual_seed(42)
        x = torch.randn([16, 64], device=DEVICE)

        code, (values, indices) = code_and_output(topk_2d_kernel, (x, 7, True), block_size=32)

        expected_v, _ = torch.topk(x, 7, dim=-1, largest=True)
        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)
        self.assertExpectedJournal(code)

    def test_topk_known_values_float32(self):
        """Test topk with known float32 input values for exact verification."""
        x = torch.full([4, 16], 0.0, device=DEVICE, dtype=torch.float32)
        x[0, 5] = 10.0
        x[0, 10] = 9.0
        x[0, 3] = 8.0
        x[1, 15] = 6.0
        x[1, 0] = 5.0
        x[1, 7] = 4.0

        code, (values, indices) = code_and_output(topk_2d_kernel, (x, 3, True), block_size=32)

        self.assertEqual(values[0].tolist(), [10.0, 9.0, 8.0])
        self.assertEqual(indices[0].tolist(), [5, 10, 3])
        self.assertEqual(values[1].tolist(), [6.0, 5.0, 4.0])
        self.assertEqual(indices[1].tolist(), [15, 0, 7])
        self.assertExpectedJournal(code)

    def test_topk_known_values_int32(self):
        """Test topk with known int32 input values for exact verification."""
        x = torch.full([4, 16], 0, device=DEVICE, dtype=torch.int32)
        x[0, 5] = 100
        x[0, 10] = 90
        x[0, 3] = 80
        x[1, 15] = 60
        x[1, 0] = 50
        x[1, 7] = 40

        code, (values, indices) = code_and_output(topk_2d_kernel, (x, 3, True), block_size=32)

        self.assertEqual(values[0].tolist(), [100, 90, 80])
        self.assertEqual(indices[0].tolist(), [5, 10, 3])
        self.assertEqual(values[1].tolist(), [60, 50, 40])
        self.assertEqual(indices[1].tolist(), [15, 0, 7])
        self.assertExpectedJournal(code)

    def test_topk_sorted_false(self):
        """Test that sorted=False skips the sorting step."""
        torch.manual_seed(42)
        x = torch.randn([32, 64], device=DEVICE)

        code, (values, indices) = code_and_output(
            topk_unsorted_kernel, (x, 10), block_size=32
        )

        # Should use tl.topk but NOT tl.sort
        self.assertIn("tl.topk", code)
        self.assertNotIn("tl.sort", code)

        # Values should be correct (just maybe not sorted)
        expected_v, _ = torch.topk(x, 10, dim=-1, largest=True)
        our_sorted = values.sort(dim=-1, descending=True)[0]
        exp_sorted = expected_v.sort(dim=-1, descending=True)[0]
        torch.testing.assert_close(our_sorted, exp_sorted, rtol=1e-4, atol=1e-4)

        # Verify indices point to correct values
        gathered = x.gather(dim=-1, index=indices)
        torch.testing.assert_close(gathered, values, rtol=1e-4, atol=1e-4)
        self.assertExpectedJournal(code)

    def test_topk_1d_tensor(self):
        """Test topk with 1D tensor input."""

        @helion.kernel(config=helion.Config(block_sizes=[128]))
        def topk_1d_kernel(
            x: torch.Tensor, k: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            k = hl.specialize(k)
            (n,) = x.size()
            values = torch.empty([k], dtype=x.dtype, device=x.device)
            indices = torch.empty([k], dtype=torch.int64, device=x.device)
            for tile_n in hl.tile(n):
                v, idx = torch.topk(x[tile_n], k, dim=-1, largest=True)
                values[:] = v
                indices[:] = idx
            return values, indices

        torch.manual_seed(42)
        x = torch.randn([64], device=DEVICE)

        code, (values, indices) = code_and_output(topk_1d_kernel, (x, 5), block_size=64)

        expected_v, expected_i = torch.topk(x, 5, dim=-1, largest=True)
        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)
        self.assertTrue(torch.equal(indices, expected_i))
        self.assertExpectedJournal(code)

    def test_topk_3d_tensor(self):
        """Test topk with 3D tensor input."""

        @helion.kernel(config=helion.Config(block_sizes=[64, 64]))
        def topk_3d_kernel(
            x: torch.Tensor, k: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            k = hl.specialize(k)
            b, n, m = x.size()
            values = torch.empty([b, n, k], dtype=x.dtype, device=x.device)
            indices = torch.empty([b, n, k], dtype=torch.int64, device=x.device)
            for tile_b, tile_n in hl.tile([b, n]):
                v, idx = torch.topk(x[tile_b, tile_n, :], k, dim=-1, largest=True)
                values[tile_b, tile_n, :] = v
                indices[tile_b, tile_n, :] = idx
            return values, indices

        torch.manual_seed(42)
        x = torch.randn([4, 8, 32], device=DEVICE)

        code, (values, indices) = code_and_output(topk_3d_kernel, (x, 5), block_size=[32, 32])

        expected_v, expected_i = torch.topk(x, 5, dim=-1, largest=True)
        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)
        self.assertTrue(torch.equal(indices, expected_i))
        self.assertExpectedJournal(code)

    def test_topk_4d_tensor(self):
        """Test topk with 4D tensor input."""

        @helion.kernel(config=helion.Config(block_sizes=[4, 4, 8]))
        def topk_4d_kernel(
            x: torch.Tensor, k: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            k = hl.specialize(k)
            a, b, n, m = x.size()
            values = torch.empty([a, b, n, k], dtype=x.dtype, device=x.device)
            indices = torch.empty([a, b, n, k], dtype=torch.int64, device=x.device)
            for tile_a, tile_b, tile_n in hl.tile([a, b, n]):
                v, idx = torch.topk(
                    x[tile_a, tile_b, tile_n, :], k, dim=-1, largest=True
                )
                values[tile_a, tile_b, tile_n, :] = v
                indices[tile_a, tile_b, tile_n, :] = idx
            return values, indices

        torch.manual_seed(42)
        x = torch.randn([2, 2, 4, 32], device=DEVICE)

        code, (values, indices) = code_and_output(topk_4d_kernel, (x, 5), block_size=[4, 4, 8])

        expected_v, expected_i = torch.topk(x, 5, dim=-1, largest=True)
        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)
        self.assertTrue(torch.equal(indices, expected_i))
        self.assertExpectedJournal(code)


@skipIfCpu("topk requires CUDA")
class TestTopKErrors(RefEagerTestBase, TestCase):
    """Test error handling for unsupported topk configurations."""

    def test_error_dim_not_last(self):
        """Test that dim != -1 raises an error."""

        @helion.kernel(config=helion.Config(block_sizes=[64]))
        def topk_dim0(
            x: torch.Tensor, k: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            k = hl.specialize(k)
            n, m = x.size()
            values = torch.empty([k, m], dtype=x.dtype, device=x.device)
            indices = torch.empty([k, m], dtype=torch.int64, device=x.device)
            for tile_n in hl.tile(n):
                v, idx = torch.topk(x[tile_n, :], k, dim=0, largest=True)
                values[:, :] = v
                indices[:, :] = idx
            return values, indices

        x = torch.randn([32, 64], device=DEVICE)
        with self.assertRaises(helion.exc.InductorLoweringError) as ctx:
            topk_dim0(x, 10)
        self.assertIn("dim=-1", str(ctx.exception))

    def test_error_unsupported_dtype_float64(self):
        """Test that float64 dtype raises an error."""
        x = torch.randn([32, 64], device=DEVICE, dtype=torch.float64)
        with self.assertRaises(helion.exc.InductorLoweringError) as ctx:
            topk_2d_kernel(x, 10, True)
        self.assertIn("float64", str(ctx.exception))

    def test_error_unsupported_dtype_int64(self):
        """Test that int64 dtype raises an error."""
        x = torch.randint(0, 100, [32, 64], device=DEVICE, dtype=torch.int64)
        with self.assertRaises(helion.exc.InductorLoweringError) as ctx:
            topk_2d_kernel(x, 10, True)
        self.assertIn("int64", str(ctx.exception))

    def test_error_input_size_exceeds_maximum(self):
        """Test that input size > 65536 raises an error."""
        x = torch.randn([4, 100000], device=DEVICE)
        with self.assertRaises(Exception) as ctx:
            topk_2d_kernel(x, 10, True)
        error_msg = str(ctx.exception).lower()
        self.assertTrue(
            "exceeds" in error_msg or "maximum" in error_msg or "numel" in error_msg
        )

    def test_error_k_not_specialized(self):
        """Test that non-specialized k raises an error."""

        @helion.kernel(config=helion.Config(block_sizes=[64]))
        def topk_no_specialize(
            x: torch.Tensor, k: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            n, m = x.size()
            values = torch.empty([n, k], dtype=x.dtype, device=x.device)
            indices = torch.empty([n, k], dtype=torch.int64, device=x.device)
            for tile_n in hl.tile(n):
                v, idx = torch.topk(x[tile_n, :], k, dim=-1, largest=True)
                values[tile_n, :] = v
                indices[tile_n, :] = idx
            return values, indices

        x = torch.randn([32, 64], device=DEVICE)
        with self.assertRaises(helion.exc.ShapeSpecializingAllocation) as ctx:
            topk_no_specialize(x, 10)
        self.assertIn("specialize", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
