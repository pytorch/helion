from __future__ import annotations

import unittest

from expecttest import TestCase
import torch

import helion
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import check_example
from helion._testing import code_and_output
import helion.language as hl


class TestTensorDescriptor(TestCase):
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
            block_sizes=[4, 8],
        )

        # Check that the result is correct
        expected = x + 1.0
        torch.testing.assert_close(result, expected)

        # Check that the generated code contains permutation calls
        self.assertIn("tl.make_tensor_descriptor", code)
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
            block_sizes=[4, 8],
        )

        # Check that the result is correct
        expected = x * 2.0
        torch.testing.assert_close(result, expected)

        # Check that the generated code contains tensor descriptor
        self.assertIn("tl.make_tensor_descriptor", code)
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

        # Verify stride pattern - middle dimension should have stride 1, others 16-byte aligned
        self.assertEqual(x.stride(), (64, 1, 4))  # Expected stride pattern
        self.assertEqual(x.stride()[1], 1)  # middle dimension has stride 1

        # Check 16-byte alignment for non-stride-1 dimensions
        element_size = x.element_size()
        for dim in range(x.ndim):
            stride = x.stride(dim)
            if stride != 1:
                byte_stride = stride * element_size
                self.assertEqual(
                    byte_stride % 16,
                    0,
                    f"Dim {dim} not 16-byte aligned: stride={stride}, byte_stride={byte_stride}",
                )

        code, result = code_and_output(
            kernel_3d_permutation,
            (x,),
            indexing="tensor_descriptor",
            block_sizes=[2, 4, 2],
        )

        # Check correctness
        expected = x + 10.0
        torch.testing.assert_close(result, expected)

        # Should contain both tensor descriptor and permute operations
        self.assertIn("tl.make_tensor_descriptor", code)
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
            block_sizes=[4, 8],
        )

        # Check correctness
        expected = x * x
        torch.testing.assert_close(result, expected)

        # Should handle the permutation properly
        self.assertIn("tl.make_tensor_descriptor", code)
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
            block_sizes=[4, 8],
        )

        expected = x + 5.0
        torch.testing.assert_close(result, expected)

        # Should contain permutation and tensor descriptor
        self.assertIn("tl.make_tensor_descriptor", code)
        self.assertIn("tl.permute", code)

        # The block sizes should also be permuted in the tensor descriptor
        # This is important for correctness

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
            block_sizes=[4, 8],
        )

        expected = x * 3.0
        torch.testing.assert_close(result, expected)

        # Should have permutation for both load and store
        self.assertIn("tl.make_tensor_descriptor", code)
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
        self.assertExpectedInline(
            check_example(
                "attention",
                args,
                torch.nn.functional.scaled_dot_product_attention(*args),
                block_sizes=[128, 64],
                indexing="tensor_descriptor",
            ),
            """\
from __future__ import annotations

import math
import torch
import helion
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_compat import libdevice

helion.runtime.set_triton_allocator()

@triton.jit
def _attention_kernel(q_view, k_view, v_view, out, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    q_view_desc = tl.make_tensor_descriptor(q_view, [64, 1024, 64], [65536, 64, 1], [1, _BLOCK_SIZE_1, 64])
    k_view_desc = tl.make_tensor_descriptor(k_view, [64, 512, 64], [32768, 64, 1], [1, _BLOCK_SIZE_3, 64])
    v_view_desc = tl.make_tensor_descriptor(v_view, [64, 512, 64], [32768, 64, 1], [1, _BLOCK_SIZE_3, 64])
    out_desc = tl.make_tensor_descriptor(out, [64, 1024, 64], [65536, 64, 1], [1, _BLOCK_SIZE_1, 64])
    num_blocks_0 = 64
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    m_i = tl.full([1, _BLOCK_SIZE_1], float('-inf'), tl.float32)
    l_i = tl.full([1, _BLOCK_SIZE_1], 1.0, tl.float32)
    acc = tl.full([1, _BLOCK_SIZE_1, 64], 0.0, tl.float32)
    q = q_view_desc.load([offset_0, offset_1, 0])
    for offset_2 in tl.range(0, 512, _BLOCK_SIZE_3):
        q_copy = q
        m_i_copy = m_i
        l_i_copy = l_i
        acc_copy = acc
        q_copy_0 = q_copy
        m_i_copy_0 = m_i_copy
        l_i_copy_0 = l_i_copy
        acc_copy_0 = acc_copy
        k = tl.permute(k_view_desc.load([offset_0, offset_2, 0]), [0, 2, 1])
        qk = tl.reshape(tl.dot(tl.reshape(q_copy_0, [_BLOCK_SIZE_1, 64]), tl.reshape(k, [64, _BLOCK_SIZE_3]), input_precision='tf32'), [1, _BLOCK_SIZE_1, _BLOCK_SIZE_3])
        amax = tl.max(qk, 2)
        v_0 = tl.full([], 0.18033688, tl.float16)
        v_1 = amax * v_0
        v_2 = v_1.to(tl.float32)
        v_3 = triton_helpers.maximum(m_i_copy_0, v_2)
        v_4 = tl.full([], 0.18033688, tl.float16)
        v_5 = qk * v_4
        subscript = v_3[:, :, None]
        v_6 = v_5.to(tl.float32)
        v_7 = v_6 - subscript
        v_8 = libdevice.exp2(v_7)
        l_ij = tl.sum(v_8, 2)
        v_9 = m_i_copy_0 - v_3
        v_10 = libdevice.exp2(v_9)
        v_11 = l_i_copy_0 * v_10
        l_i = v_11 + l_ij
        subscript_1 = v_10[:, :, None]
        v_13 = acc_copy_0 * subscript_1
        v = v_view_desc.load([offset_0, offset_2, 0])
        v_14 = v_8.to(tl.float16)
        acc = tl.reshape(tl.dot(tl.reshape(v_14, [_BLOCK_SIZE_1, _BLOCK_SIZE_3]), tl.reshape(v, [_BLOCK_SIZE_3, 64]), acc=tl.reshape(v_13, [_BLOCK_SIZE_1, 64]), input_precision='tf32'), [1, _BLOCK_SIZE_1, 64])
        m_i = v_3
    subscript_2 = l_i[:, :, None]
    v_15 = acc / subscript_2
    v_16 = v_15.to(tl.float16)
    out_desc.store([offset_0, offset_1, 0], v_16)

def attention(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = 64
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 128
    _RDIM_SIZE_2 = 64
    _BLOCK_SIZE_3 = 64
    _attention_kernel[64 * triton.cdiv(1024, _BLOCK_SIZE_1),](q_view, k_view, v_view, out, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)
    return out.view(q_in.size())

def _attention_make_precompiler(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = 64
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 128
    _RDIM_SIZE_2 = 64
    _BLOCK_SIZE_3 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_attention_kernel)(q_view, k_view, v_view, out, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)""",
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
        self.assertExpectedInline(
            check_example(
                "attention",
                args,
                torch.nn.functional.scaled_dot_product_attention(*args),
                fn_name="attention_dynamic",
                block_sizes=[16, 16],
                indexing="tensor_descriptor",
            ),
            """\
from __future__ import annotations

import math
import torch
import helion
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_compat import libdevice

helion.runtime.set_triton_allocator()

@triton.jit
def _attention_kernel(q_view, k_view, v_view, out, k_view_size_0, k_view_size_2, out_size_0, out_size_1, q_in_size_1, q_view_size_0, q_view_size_1, v_view_size_0, v_view_size_1, k_view_stride_0, k_view_stride_1, k_view_stride_2, out_stride_0, out_stride_1, out_stride_2, q_view_stride_0, q_view_stride_1, q_view_stride_2, v_view_stride_0, v_view_stride_1, v_view_stride_2, m_dim, n_dim, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    q_view_desc = tl.make_tensor_descriptor(q_view, [q_view_size_0, q_view_size_1, 64], [q_view_stride_0, q_view_stride_1, q_view_stride_2], [1, _BLOCK_SIZE_1, 64])
    k_view_desc = tl.make_tensor_descriptor(k_view, [k_view_size_0, k_view_size_2, 64], [k_view_stride_0, k_view_stride_2, k_view_stride_1], [1, _BLOCK_SIZE_3, 64])
    v_view_desc = tl.make_tensor_descriptor(v_view, [v_view_size_0, v_view_size_1, 64], [v_view_stride_0, v_view_stride_1, v_view_stride_2], [1, _BLOCK_SIZE_3, 64])
    out_desc = tl.make_tensor_descriptor(out, [out_size_0, out_size_1, 64], [out_stride_0, out_stride_1, out_stride_2], [1, _BLOCK_SIZE_1, 64])
    num_blocks_0 = q_in_size_1
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < m_dim
    m_i = tl.full([1, _BLOCK_SIZE_1], float('-inf'), tl.float32)
    l_i = tl.full([1, _BLOCK_SIZE_1], 1.0, tl.float32)
    acc = tl.full([1, _BLOCK_SIZE_1, 64], 0.0, tl.float32)
    q = q_view_desc.load([offset_0, offset_1, 0])
    for offset_2 in tl.range(0, n_dim.to(tl.int32), _BLOCK_SIZE_3):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
        mask_3 = indices_2 < n_dim
        q_copy = q
        m_i_copy = m_i
        l_i_copy = l_i
        acc_copy = acc
        q_copy_0 = q_copy
        m_i_copy_0 = m_i_copy
        l_i_copy_0 = l_i_copy
        acc_copy_0 = acc_copy
        k = tl.permute(k_view_desc.load([offset_0, offset_2, 0]), [0, 2, 1])
        qk = tl.reshape(tl.dot(tl.reshape(q_copy_0, [_BLOCK_SIZE_1, 64]), tl.reshape(k, [64, _BLOCK_SIZE_3]), input_precision='tf32'), [1, _BLOCK_SIZE_1, _BLOCK_SIZE_3])
        _mask_to_2 = tl.where(tl.broadcast_to(mask_1[None, :, None] & mask_3[None, None, :], [1, _BLOCK_SIZE_1, _BLOCK_SIZE_3]), qk, float('-inf'))
        amax = tl.max(_mask_to_2, 2)
        v_0 = 0.18033688
        v_1 = amax * v_0
        v_2 = triton_helpers.maximum(m_i_copy_0, v_1)
        v_3 = 0.18033688
        v_4 = qk * v_3
        subscript = v_2[:, :, None]
        v_5 = v_4 - subscript
        v_6 = libdevice.exp2(v_5)
        _mask_to_3 = tl.where(tl.broadcast_to(mask_1[None, :, None] & mask_3[None, None, :], [1, _BLOCK_SIZE_1, _BLOCK_SIZE_3]), v_6, 0)
        l_ij = tl.sum(_mask_to_3, 2)
        v_7 = m_i_copy_0 - v_2
        v_8 = libdevice.exp2(v_7)
        v_9 = l_i_copy_0 * v_8
        l_i = v_9 + l_ij
        subscript_1 = v_8[:, :, None]
        v_11 = acc_copy_0 * subscript_1
        v = v_view_desc.load([offset_0, offset_2, 0])
        acc = tl.reshape(tl.dot(tl.reshape(_mask_to_3, [_BLOCK_SIZE_1, _BLOCK_SIZE_3]), tl.reshape(v, [_BLOCK_SIZE_3, 64]), acc=tl.reshape(v_11, [_BLOCK_SIZE_1, 64]), input_precision='tf32'), [1, _BLOCK_SIZE_1, 64])
        m_i = v_2
    subscript_2 = l_i[:, :, None]
    v_12 = acc / subscript_2
    out_desc.store([offset_0, offset_1, 0], v_12)

def attention(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = 64
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 16
    _RDIM_SIZE_2 = 64
    _BLOCK_SIZE_3 = 16
    _attention_kernel[q_in.size(1) * triton.cdiv(m_dim, _BLOCK_SIZE_1),](q_view, k_view, v_view, out, k_view.size(0), k_view.size(2), out.size(0), out.size(1), q_in.size(1), q_view.size(0), q_view.size(1), v_view.size(0), v_view.size(1), k_view.stride(0), k_view.stride(1), k_view.stride(2), out.stride(0), out.stride(1), out.stride(2), q_view.stride(0), q_view.stride(1), q_view.stride(2), v_view.stride(0), v_view.stride(1), v_view.stride(2), m_dim, n_dim, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)
    return out.view(q_in.size())

def _attention_make_precompiler(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = 64
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 16
    _RDIM_SIZE_2 = 64
    _BLOCK_SIZE_3 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_attention_kernel)(q_view, k_view, v_view, out, k_view.size(0), k_view.size(2), out.size(0), out.size(1), q_in.size(1), q_view.size(0), q_view.size(1), v_view.size(0), v_view.size(1), k_view.stride(0), k_view.stride(1), k_view.stride(2), out.stride(0), out.stride(1), out.stride(2), q_view.stride(0), q_view.stride(1), q_view.stride(2), v_view.stride(0), v_view.stride(1), v_view.stride(2), m_dim, n_dim, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)""",
        )


if __name__ == "__main__":
    unittest.main()
