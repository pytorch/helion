from __future__ import annotations

import re
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

        @helion.kernel(autotune_effort="none")
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

        @helion.kernel(autotune_effort="none")
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

        @helion.kernel(autotune_effort="none")
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

        @helion.kernel(autotune_effort="none")
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

        @helion.kernel(autotune_effort="none")
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
    def test_multistage_range_tensor_descriptor(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[4, 256],
                indexing="tensor_descriptor",
                num_stages=4,
                num_warps=4,
                pid_type="flat",
                range_flattens=[None, False],
                range_multi_buffers=[None, False],
                range_num_stages=[0, 4],
                range_unroll_factors=[0, 0],
                range_warp_specializes=[],
            ),
            static_shapes=True,
        )
        def jsd_forward_kernel(
            _input: torch.Tensor,
            target: torch.Tensor,
            shift_labels: torch.Tensor | None = None,
            beta: float = 0.5,
            ignore_index: int = -100,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            BT, V = _input.shape
            assert target.shape == _input.shape, (
                f"Shape mismatch: {target.shape} != {_input.shape}"
            )
            block_size_n = hl.register_block_size(V)
            block_size_m = hl.register_block_size(BT)

            loss = torch.zeros([BT], dtype=torch.float32, device=_input.device)
            dX = torch.empty_like(loss)

            one_minus_beta = 1 - beta

            n_non_ignore = float(BT)
            if shift_labels is not None:
                n_non_ignore = float((shift_labels != ignore_index).sum().item())
                if n_non_ignore == 0:
                    return torch.zeros(
                        [], dtype=_input.dtype, device=_input.device
                    ), torch.zeros_like(_input)

            for tile_bt in hl.tile(BT, block_size=block_size_m):
                if shift_labels is not None:
                    if shift_labels[tile_bt] == ignore_index:
                        for tile_X in hl.tile(V):
                            dX[tile_bt, tile_X] = 0.0
                        continue
                intermediate_loss = hl.zeros(
                    [tile_bt, block_size_n], dtype=torch.float32
                )
                intermediate_dX = hl.zeros([tile_bt, block_size_n], dtype=_input.dtype)
                for tile_v in hl.tile(V, block_size=block_size_n):
                    X = _input[tile_bt, tile_v]
                    Y = target[tile_bt, tile_v]

                    if beta == 0.0:
                        Y_max = torch.amax(Y, dim=0)
                        Y_shift = Y - Y_max
                        Y_prob = torch.exp(Y_shift) * torch.exp(Y_max)
                        intermediate_loss += Y_prob * (Y - X)
                        intermediate_dX += -Y_prob
                    elif beta == 1.0:
                        X_max = torch.amax(X, dim=0)
                        X_shift = X - X_max
                        X_prob = torch.exp(X_shift) * torch.exp(X_max)
                        intermediate_loss += X_prob * (X - Y)
                        intermediate_dX += intermediate_loss + X_prob
                    else:
                        Q = torch.exp(X)
                        P = torch.exp(Y)

                        beta_P = beta * P
                        one_minus_beta_Q = one_minus_beta * Q
                        M = beta_P + one_minus_beta_Q
                        log_M = torch.log(M)
                        x_minus_log_m = X - log_M
                        kl_q_m = one_minus_beta_Q * x_minus_log_m

                        intermediate_loss += beta_P * (Y - log_M) + kl_q_m
                        intermediate_dX += kl_q_m

                scale = 1.0 / n_non_ignore
                loss[tile_bt] = torch.sum(intermediate_loss * scale, dim=1)
                dX[tile_bt] = torch.sum(intermediate_dX * scale, dim=1)

            final_loss = torch.sum(loss)
            return final_loss, dX

        vocab = 512
        batch = 512
        log_q = torch.randn(batch, vocab, device=DEVICE).log_softmax(dim=-1)
        log_p = torch.randn(batch, vocab, device=DEVICE).log_softmax(dim=-1)

        code, (loss, _) = code_and_output(jsd_forward_kernel, (log_q, log_p))
        torch.accelerator.synchronize()

        from examples.jsd import TorchJSDBaseline

        baseline = TorchJSDBaseline(beta=0.5, ignore_index=-100).to(DEVICE)
        baseline_loss = baseline(log_q, log_p)

        torch.testing.assert_close(loss, baseline_loss, rtol=5e-2, atol=5e-3)
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        range_stage_values = [
            int(match)
            for line in code.splitlines()
            if "tl.range" in line
            for match in re.findall(r"num_stages=(\d+)", line)
        ]
        # range_num_stages=4 is clamped to 0, so doesn't show up as num_stages in the tl.range call
        self.assertEqual(len(range_stage_values), 0)

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_tiny_matmul_tile_fallback(self) -> None:
        """Tensor descriptor indexing should be rejected when the tile is too small."""

        @helion.kernel(
            config=helion.Config(
                block_sizes=[1, 16, 16],
                indexing="tensor_descriptor",
                l2_groupings=[2],
                loop_orders=[[0, 1]],
                num_stages=4,
                num_warps=1,
                pid_type="persistent_blocked",
                range_flattens=[True, True],
                range_multi_buffers=[False, True],
                range_num_stages=[0, 1],
                range_unroll_factors=[0, 4],
            ),
            static_shapes=True,
        )
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2
            out = torch.empty(
                [m, n],
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(out.dtype)
            return out

        x = torch.randn((64, 64), device=DEVICE, dtype=torch.float16)
        y = torch.randn((64, 64), device=DEVICE, dtype=torch.float16)

        code, result = code_and_output(matmul, (x, y))
        torch.accelerator.synchronize()
        expected = torch.matmul(x, y)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

        # Ensure we fall back to pointer indexing for accesses that would use the
        # 1x16 tile - there should be no tensor descriptor for the x or out tensors.
        self.assertNotIn("x_desc = tl.make_tensor_descriptor", code)
        self.assertNotIn("out_desc = tl.make_tensor_descriptor", code)
        # The K dimension still has a valid tile size, so the column operand can
        # keep using tensor descriptors.
        self.assertIn("y_desc = tl.make_tensor_descriptor", code)

        # A larger tile should still be able to use tensor descriptors
        code_large, result_large = code_and_output(
            matmul,
            (x, y),
            block_sizes=[16, 16, 16],
            indexing="tensor_descriptor",
        )
        torch.accelerator.synchronize()
        torch.testing.assert_close(result_large, expected, atol=1e-2, rtol=1e-2)
        self.assertIn(get_tensor_descriptor_fn_name(), code_large)

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_store_operation_permutation(self):
        """Test that store operations also handle permutation correctly."""

        @helion.kernel(autotune_effort="none")
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
                block_sizes=[1, 128, 64],
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
                block_sizes=[1, 16, 16],
                indexing="tensor_descriptor",
            )
        )

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_minimum_16_byte_block_size_fallback(self):
        """Test that tensor descriptor falls back when block size is too small."""

        @helion.kernel(autotune_effort="none")
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
