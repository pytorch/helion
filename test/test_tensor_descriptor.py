from __future__ import annotations

import unittest

import torch

import helion
from helion._compat import get_tensor_descriptor_fn_name
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import check_example
from helion._testing import code_and_output
import helion.language as hl

class TestTensorDescriptor(RefEagerTestBase, TestCase):
    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_permutation_when_stride_one_not_last(self):
        """Test that permutation is applied when stride==1 is not the last dimension."""

        @helion.kernel(use_default_config=True)
        def kernel_with_permutation(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] + 1.0
            return result

        # Create tensor where stride==1 is the first dimension (not last)
        # This should trigger permutation logic
        x_base = torch.randn([8, 16], device=DEVICE, dtype=torch.float32)
        x = x_base.t().contiguous().t()  # This creates stride=[1, 8]

        # Verify the stride pattern we want
        self.assertEqual(x.stride(), (1, 8))
        self.assertEqual(x.stride(0), 1)  # First dimension has stride 1
        self.assertEqual(x.stride(1), 8)  # Second dimension has stride 8

        code, result = code_and_output(
            kernel_with_permutation,
            (x,),
            indexing="tensor_descriptor",
            block_sizes=[8, 8],
        )

        # Check that the result is correct
        expected = x + 1.0
        torch.testing.assert_close(result, expected)

        # Check that the generated code contains permutation calls
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        # The tensor descriptor should be created with permuted dimensions
        # (sizes and strides should be reordered so stride==1 dim is last)

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_no_permutation_when_stride_one_already_last(self):
        """Test that no permutation is applied when stride==1 is already last."""

        @helion.kernel(use_default_config=True)
        def kernel_no_permutation(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] * 2.0
            return result

        # Create tensor where stride==1 is already the last dimension
        x = torch.randn([8, 16], device=DEVICE, dtype=torch.float32)

        # Verify the stride pattern (last dimension should have stride 1)
        self.assertEqual(x.stride(), (16, 1))
        self.assertEqual(x.stride(-1), 1)  # Last dimension has stride 1

        code, result = code_and_output(
            kernel_no_permutation,
            (x,),
            indexing="tensor_descriptor",
            block_sizes=[8, 8],
        )

        # Check that the result is correct
        expected = x * 2.0
        torch.testing.assert_close(result, expected)

        # Check that the generated code contains tensor descriptor
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        # Should not contain permute calls since no permutation needed
        self.assertNotIn("tl.permute", code)

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_3d_tensor_permutation(self):
        """Test permutation with 3D tensor where stride==1 is in middle."""

        @helion.kernel(use_default_config=True)
        def kernel_3d_permutation(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] + 10.0
            return result

        # Create 3D tensor where stride==1 is the middle dimension
        # We'll use as_strided to create a tensor with stride pattern [64, 1, 4]
        # This gives byte strides [256, 4, 16] where 256%16==0 and 16%16==0
        storage_size = 4 * 8 * 16  # Enough storage for the tensor
        base_tensor = torch.randn(storage_size, device=DEVICE, dtype=torch.float32)
        x = base_tensor.as_strided([4, 8, 4], [64, 1, 4])

        code, result = code_and_output(
            kernel_3d_permutation,
            (x,),
            indexing="tensor_descriptor",
            block_sizes=[8, 8, 8],
        )

        # Check correctness
        expected = x + 10.0
        torch.testing.assert_close(result, expected)

        # Should contain both tensor descriptor and permute operations
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        self.assertIn("tl.permute", code)

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_matrix_transpose_case(self):
        """Test a common case: transposed matrix operations."""

        @helion.kernel(use_default_config=True)
        def kernel_transpose_case(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] * x[tile]  # Element-wise square
            return result

        # Create a transposed matrix (common in many GPU kernels)
        x_orig = torch.randn([16, 12], device=DEVICE, dtype=torch.float32)
        x = x_orig.t()  # Transpose: shape=[12, 16], stride=[1, 12]

        # Verify this is the problematic case: stride==1 is first, not last
        self.assertEqual(x.shape, (12, 16))
        self.assertEqual(x.stride(), (1, 12))

        code, result = code_and_output(
            kernel_transpose_case,
            (x,),
            indexing="tensor_descriptor",
            block_sizes=[8, 8],
        )

        # Check correctness
        expected = x * x
        torch.testing.assert_close(result, expected)

        # Should handle the permutation properly
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        self.assertIn("tl.permute", code)

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_permutation_with_different_block_sizes(self):
        """Test that permutation works correctly with different block sizes."""

        @helion.kernel(use_default_config=True)
        def kernel_different_blocks(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] + 5.0
            return result

        # Create tensor where stride==1 is not last
        x_base = torch.randn([12, 24], device=DEVICE, dtype=torch.float32)
        x = x_base.t().contiguous().t()  # stride=[1, 12]

        self.assertEqual(x.stride(), (1, 12))

        code, result = code_and_output(
            kernel_different_blocks,
            (x,),
            indexing="tensor_descriptor",
            block_sizes=[8, 8],
        )

        expected = x + 5.0
        torch.testing.assert_close(result, expected)

        # Should contain permutation and tensor descriptor
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        self.assertIn("tl.permute", code)

        # The block sizes should also be permuted in the tensor descriptor
        # This is important for correctness

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_kl_div_fallback_for_skinny_tile(self):
        """Ensure KL div example falls back to pointer indexing for 1xN tiles."""

        torch.manual_seed(0)
        B, T, V = 8, 512, 4096

        @helion.kernel(
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
            static_shapes=True,
            configs=[
                helion.Config(
                    block_sizes=[128],
                    indexing="tensor_descriptor",
                    num_stages=5,
                    num_warps=1,
                    pid_type="persistent_interleaved",
                    range_flattens=[None, True],
                    range_multi_buffers=[None, True],
                    range_num_stages=[2, 3],
                    range_unroll_factors=[1, 2],
                )
            ],
        )
        def kl_div_forward(
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            log_target: bool = False,
        ) -> torch.Tensor:
            BT, V_local = y_pred.shape
            assert y_true.shape == y_pred.shape, (
                f"Shape mismatch: {y_true.shape} != {y_pred.shape}"
            )

            loss = torch.zeros((BT,), dtype=torch.float32, device=y_pred.device)

            kl_loss = torch.zeros_like(y_pred)

            block_size_n = hl.register_block_size(V_local)

            BT_SIZE = helion.cdiv(BT, BT)
            for tile_bt in hl.tile(BT, block_size=BT_SIZE):
                loss_sum = hl.zeros([tile_bt, block_size_n], dtype=torch.float32)

                for tile_v in hl.tile(V_local, block_size=block_size_n):
                    y_pred_val = y_pred[tile_bt, tile_v]
                    y_true_val = y_true[tile_bt, tile_v]

                    if log_target:
                        prob_true = torch.exp(y_true_val)
                        kl_loss[tile_bt, tile_v] = prob_true * (y_true_val - y_pred_val)
                    else:
                        log_true = torch.log(torch.clamp(y_true_val, min=1e-10))
                        kl_loss[tile_bt, tile_v] = y_true_val * (log_true - y_pred_val)

                    loss_sum += kl_loss[tile_bt, tile_v]

                loss[tile_bt] = loss_sum.sum(dim=-1)

            final_loss = torch.sum(loss) / BT
            return final_loss

        y_pred = torch.randn(B * T, V, device=DEVICE, dtype=torch.float32).log_softmax(
            dim=-1
        )
        y_true = torch.randn(B * T, V, device=DEVICE, dtype=torch.float32).softmax(
            dim=-1
        )

        args = (y_pred, y_true, False)
        code, helion_out = code_and_output(
            kl_div_forward,
            args,
            block_sizes=[128],
            indexing="tensor_descriptor",
            num_stages=5,
            num_warps=1,
            pid_type="persistent_interleaved",
            range_flattens=[None, True],
            range_multi_buffers=[None, True],
            range_num_stages=[2, 3],
            range_unroll_factors=[1, 2],
        )
        # Confirm tensor-descriptor indexing is skipped to avoid illegal 1xN TMA tiles.
        self.assertNotIn(get_tensor_descriptor_fn_name(), code)
        self.assertTrue(torch.isfinite(helion_out))
        self.assertEqual(helion_out.shape, torch.Size([]))

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_store_operation_permutation(self):
        """Test that store operations also handle permutation correctly."""

        @helion.kernel(use_default_config=True)
        def kernel_store_permutation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Both tensors might need permutation
            for tile in hl.tile(x.size()):
                y[tile] = x[tile] * 3.0
            return y

        # Create input and output tensors with stride==1 not last
        x_base = torch.randn([8, 12], device=DEVICE, dtype=torch.float32)
        x = x_base.t().contiguous().t()  # stride=[1, 8]

        y_base = torch.zeros([8, 12], device=DEVICE, dtype=torch.float32)
        y = y_base.t().contiguous().t()  # stride=[1, 8]

        self.assertEqual(x.stride(), (1, 8))
        self.assertEqual(y.stride(), (1, 8))

        code, result = code_and_output(
            kernel_store_permutation,
            (x, y),
            indexing="tensor_descriptor",
            block_sizes=[8, 8],
        )

        expected = x * 3.0
        torch.testing.assert_close(result, expected)

        # Should have permutation for both load and store
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        self.assertIn("tl.permute", code)

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_attention_tensor_descriptor(self):
        args = (
            torch.randn(2, 32, 1024, 64, dtype=torch.float16, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=torch.float16, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=torch.float16, device=DEVICE),
        )
        self.assertExpectedJournal(
            check_example(
                "attention",
                args,
                torch.nn.functional.scaled_dot_product_attention(*args),
                block_sizes=[128, 64],
                indexing="tensor_descriptor",
            )
        )

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_attention_td_dynamic(self):
        args = (
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
        )
        self.assertExpectedJournal(
            check_example(
                "attention",
                args,
                torch.nn.functional.scaled_dot_product_attention(*args),
                fn_name="attention_dynamic",
                block_sizes=[16, 16],
                indexing="tensor_descriptor",
            )
        )

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_minimum_16_byte_block_size_fallback(self):
        """Test that tensor descriptor falls back when block size is too small."""

        @helion.kernel(use_default_config=True)
        def kernel_small_block(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] + 1.0
            return result

        # Create a tensor with proper stride alignment
        x = torch.randn([8, 16], device=DEVICE, dtype=torch.float32)

        # Use small block sizes that would result in < 16 bytes in last dimension
        # block_sizes=[4, 2] means last dimension block size = 2
        # 2 * 4 bytes (float32) = 8 bytes < 16 bytes required
        # With the fix, this should fall back to another indexing strategy
        code, result = code_and_output(
            kernel_small_block,
            (x,),
            indexing="tensor_descriptor",  # Request tensor descriptor
            block_sizes=[4, 2],  # Small block size in last dimension
        )

        # Should fall back to block_ptr or pointer indexing instead of tensor descriptor
        # If our fix works, this should NOT contain tensor descriptor
        self.assertNotIn(get_tensor_descriptor_fn_name(), code)

        # But should still work correctly
        expected = x + 1.0
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
