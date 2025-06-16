from __future__ import annotations

import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


@helion.kernel()
def atomic_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Test basic atomic_add functionality."""
    for i in hl.tile(x.size(0)):
        hl.atomic_add(x, [i], y[i])
    return x


@helion.kernel(static_shapes=True)
def atomic_add_overlap_kernel(
    x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Test atomic_add with overlapping indices."""
    for i in hl.tile([y.size(0)]):
        idx = indices[i]
        hl.atomic_add(x, [idx], y[i])
    return x


@helion.kernel()
def atomic_add_2d_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Test atomic_add with 2D indexing."""
    for i, j in hl.tile([y.size(0), y.size(1)]):
        hl.atomic_add(x, [i, j], y[i, j])
    return x


@helion.kernel()
def atomic_add_float_kernel(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Test atomic_add with a float constant value and reading from lookup"""
    for i in hl.tile(indices.size(0)):
        idx = indices[i]
        hl.atomic_add(x, [idx], 2.0)
    return x


@helion.kernel()
def atomic_add_w_tile_attr(x: torch.Tensor) -> torch.Tensor:
    """Test atomic_add where the index is a symbolic int"""
    y = torch.zeros_like(x, device=x.device, dtype=torch.int32)
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(y, [tile.begin], 1)
    return y


class TestAtomicOperations(TestCase):
    maxDiff = 16384

    def test_basic_atomic_add(self):
        x = torch.zeros(10, device=DEVICE)
        y = torch.ones(10, device=DEVICE)
        args = (x, y)

        code, result = code_and_output(
            atomic_add_kernel,
            args,
            block_sizes=[32],
        )

        expected = torch.ones(10, device=DEVICE)
        torch.testing.assert_close(result, expected)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _atomic_add_kernel_kernel(x, y, x_size_0, x_stride_0, y_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    load = tl.load(y + indices_0 * y_stride_0, mask_0, other=0)
    tl.atomic_add(x + indices_0 * x_stride_0, load, mask=mask_0, sem='relaxed')

def atomic_add_kernel(x: torch.Tensor, y: torch.Tensor):
    \"\"\"Test basic atomic_add functionality.\"\"\"
    _BLOCK_SIZE_0 = 32
    _atomic_add_kernel_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, y, x.size(0), x.stride(0), y.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return x

def _atomic_add_kernel_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    \"\"\"Test basic atomic_add functionality.\"\"\"
    _BLOCK_SIZE_0 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_atomic_add_kernel_kernel)(x, y, x.size(0), x.stride(0), y.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_overlapping_atomic_add(self):
        # Test with overlapping indices
        x = torch.zeros(5, device=DEVICE)
        y = torch.ones(10, device=DEVICE)
        indices = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], device=DEVICE)
        args = (x, y, indices)

        code, result = code_and_output(
            atomic_add_overlap_kernel,
            args,
            block_sizes=[32],
        )

        expected = torch.ones(5, device=DEVICE) * 2
        torch.testing.assert_close(result, expected)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _atomic_add_overlap_kernel_kernel(indices, y, x, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < 10
    idx = tl.load(indices + indices_0 * 1, mask_0, other=0)
    load_1 = tl.load(y + indices_0 * 1, mask_0, other=0)
    tl.atomic_add(x + idx * 1, load_1, mask=mask_0, sem='relaxed')

def atomic_add_overlap_kernel(x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor):
    \"\"\"Test atomic_add with overlapping indices.\"\"\"
    _BLOCK_SIZE_0 = 32
    _atomic_add_overlap_kernel_kernel[triton.cdiv(10, _BLOCK_SIZE_0),](indices, y, x, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return x

def _atomic_add_overlap_kernel_make_precompiler(x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor):
    \"\"\"Test atomic_add with overlapping indices.\"\"\"
    _BLOCK_SIZE_0 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_atomic_add_overlap_kernel_kernel)(indices, y, x, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_2d_atomic_add(self):
        """Test atomic_add with 2D tensor indexing."""
        x = torch.zeros(3, 4, device=DEVICE)
        y = torch.ones(3, 4, device=DEVICE)
        args = (x, y)

        code, result = code_and_output(
            atomic_add_2d_kernel,
            args,
            block_sizes=[8, 8],
        )

        expected = torch.ones(3, 4, device=DEVICE)
        torch.testing.assert_close(result, expected)
        self.assertIn("atomic_add", code)

    def test_atomic_add_code_generation(self):
        """Test that the generated code contains atomic_add."""
        x = torch.zeros(10, device=DEVICE)
        y = torch.ones(10, device=DEVICE)
        args = (x, y)

        code, _ = code_and_output(atomic_add_kernel, args)
        self.assertIn("atomic_add", code)

    def test_atomic_add_float(self):
        """Test that atomic_add works with float constants."""
        x = torch.zeros(5, device=DEVICE, dtype=torch.float32)

        indices = torch.tensor([0, 1, 2, 2, 3, 3, 3, 4], device=DEVICE)
        expected = torch.tensor(
            [2.0, 2.0, 4.0, 6.0, 2.0], device=DEVICE, dtype=torch.float32
        )

        args = (x, indices)
        code, result = code_and_output(
            atomic_add_float_kernel,
            args,
            block_sizes=[32],
        )

        torch.testing.assert_close(result, expected)

    def test_atomic_add_invalid_sem(self):
        """Test that atomic_add raises with an invalid sem value."""
        x = torch.zeros(10, device=DEVICE)
        y = torch.ones(10, device=DEVICE)

        @helion.kernel()
        def bad_atomic_add_kernel(x: torch.Tensor, y: torch.Tensor):
            for i in hl.tile(x.size(0)):
                hl.atomic_add(x, [i], y[i], sem="ERROR")
            return x

        with self.assertRaises(helion.exc.InternalError) as ctx:
            code_and_output(
                bad_atomic_add_kernel,
                (x, y),
                block_sizes=[32],
            )
        self.assertIn("Invalid memory semantic 'ERROR'", str(ctx.exception))

    def test_atomic_add_w_tile_attr(self):
        """Test atomic_add where the index is a symbolic int"""
        x = torch.randn(20, device=DEVICE)
        code, result = code_and_output(
            atomic_add_w_tile_attr,
            (x,),
            block_sizes=[2],
        )

        expected = torch.tensor([1, 0], device=DEVICE, dtype=torch.int32).repeat(10)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
