"""Tests for broadcast operation masking with padded reductions.

These tests verify that broadcast operations between padded tensors and reduced
values correctly handle padding to avoid corruption.
"""

import torch

import helion
import helion.language as hl
from helion._testing import DEVICE
from helion._testing import TestCase


class TestBroadcastMasking(TestCase):
    """Test that broadcast operations preserve padding correctly."""

    def test_broadcast_sub_padded_minus_reduced(self):
        """Test x - mean where x is padded and mean is reduced.

        This is the classic layer norm centering pattern.
        Without masking: padding becomes 0 - mean = -mean (corruption!)
        Subsequent reduction (variance) would include corrupted padding.
        With masking: padding stays 0, reductions are correct.
        """

        @helion.kernel(config={"block_sizes": [32]})
        def centered_variance(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                x_part = x[tile_m, :]
                mean = x_part.mean(dim=1, keepdim=True)
                centered_x = x_part - mean  # Broadcast subtraction
                # Subsequent reduction to expose corruption
                variance = (centered_x * centered_x).mean(dim=1)
                out[tile_m] = variance
            return out

        x = torch.randn([100, 100], device=DEVICE)
        result = centered_variance(x)

        # Reference implementation
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        expected = (centered * centered).mean(dim=1)

        torch.testing.assert_close(result, expected)

    def test_broadcast_sub_reduced_minus_padded(self):
        """Test mean - x where mean is reduced and x is padded.

        Less common but still valid pattern. Tests reversed operand order.
        Without masking: padding becomes mean - 0 = mean (corruption!)
        """

        @helion.kernel(config={"block_sizes": [32]})
        def negated_centered_sum(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                x_part = x[tile_m, :]
                mean = x_part.mean(dim=1, keepdim=True)
                negated = mean - x_part  # Reverse broadcast subtraction
                # Subsequent reduction to expose corruption
                out[tile_m] = negated.sum(dim=1)
            return out

        x = torch.randn([100, 100], device=DEVICE)
        result = negated_centered_sum(x)
        mean = x.mean(dim=1, keepdim=True)
        negated = mean - x
        expected = negated.sum(dim=1)
        torch.testing.assert_close(result, expected)

    def test_broadcast_add_padded_plus_reduced(self):
        """Test x + offset where x is padded and offset is reduced.

        Without masking: padding becomes 0 + offset = offset (corruption!)
        When we then reduce again, the corrupted padding affects the result.
        With masking: padding stays 0, subsequent reductions are correct.
        """

        @helion.kernel(config={"block_sizes": [32]})
        def add_then_sum(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                x_part = x[tile_m, :]
                offset = x_part.mean(dim=1, keepdim=True)
                shifted = x_part + offset  # Broadcast addition
                # Subsequent reduction - this is where corruption matters!
                out[tile_m] = shifted.sum(dim=1)
            return out

        x = torch.randn([100, 100], device=DEVICE)
        result = add_then_sum(x)

        # Reference: padding stays 0 after addition, doesn't affect sum
        offset = x.mean(dim=1, keepdim=True)
        shifted = x + offset
        expected = shifted.sum(dim=1)
        torch.testing.assert_close(result, expected)

    def test_broadcast_add_reduced_plus_padded(self):
        """Test offset + x where offset is reduced and x is padded.

        Similar to above but with operands reversed. Tests that masking
        works regardless of operand order.
        """

        @helion.kernel(config={"block_sizes": [32]})
        def add_reverse_then_sum(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                x_part = x[tile_m, :]
                offset = x_part.mean(dim=1, keepdim=True)
                shifted = offset + x_part  # Reverse broadcast addition
                # Subsequent reduction to expose corruption
                out[tile_m] = shifted.sum(dim=1)
            return out

        x = torch.randn([100, 100], device=DEVICE)
        result = add_reverse_then_sum(x)

        offset = x.mean(dim=1, keepdim=True)
        shifted = offset + x
        expected = shifted.sum(dim=1)
        torch.testing.assert_close(result, expected)

    def test_multiple_broadcast_operations(self):
        """Test multiple broadcast operations in sequence.

        Verifies that masking is correctly applied for each operation
        independently, with subsequent reduction to expose any corruption.
        """

        @helion.kernel(config={"block_sizes": [32]})
        def multi_broadcast_sum(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                x_part = x[tile_m, :]
                mean = x_part.mean(dim=1, keepdim=True)
                # Multiple unsafe broadcasts
                centered = x_part - mean
                offset_centered = centered + (mean * 0.1)
                final = offset_centered - (mean * 0.01)
                # Subsequent reduction to expose any corruption
                out[tile_m] = final.sum(dim=1)
            return out

        x = torch.randn([100, 100], device=DEVICE)
        result = multi_broadcast_sum(x)
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        offset_centered = centered + (mean * 0.1)
        final = offset_centered - (mean * 0.01)
        expected = final.sum(dim=1)
        torch.testing.assert_close(result, expected)

    def test_nonpow2_reduction_dimension(self):
        """Test broadcast masking with non-power-of-2 reduction dimension.

        This ensures masking works correctly even when reduction dimension
        requires padding (not a power of 2). With subsequent reduction to
        expose corruption.
        """

        @helion.kernel(config={"block_sizes": [2]})
        def nonpow2_centered_variance(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                x_part = x[tile_m, :]
                mean = x_part.mean(dim=1, keepdim=True)
                centered = x_part - mean
                # Subsequent reduction to expose corruption
                variance = (centered * centered).mean(dim=1)
                out[tile_m] = variance
            return out

        # Use 1536 which is not a power of 2
        x = torch.randn([2, 1536], device=DEVICE)
        result = nonpow2_centered_variance(x)
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        expected = (centered * centered).mean(dim=1)
        torch.testing.assert_close(result, expected)

    def test_tuple_valued_reduction(self):
        """Test broadcasts sourced from tuple-valued reductions (max returns values, indices).

        Ensures masking logic sees reductions even when the reduction node feeds
        tuple outputs, so padding remains clean for subsequent broadcasts.
        """

        @helion.kernel(config={"block_sizes": [32]})
        def var_mean_centered(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                x_part = x[tile_m, :]
                variance, mean = torch.ops.aten.var_mean.correction(
                    x_part,
                    [1],
                    correction=0,
                    keepdim=True,
                )
                centered = x_part - mean  # Broadcast subtraction with tuple-valued reduction output
                out[tile_m] = centered.sum(dim=1)
            return out

        x = torch.randn([64, 64], device=DEVICE)
        result = var_mean_centered(x)

        _, mean = torch.ops.aten.var_mean.correction(
            x,
            [1],
            correction=0,
            keepdim=True,
        )
        centered = x - mean
        expected = centered.sum(dim=1)
        torch.testing.assert_close(result, expected)
