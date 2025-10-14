from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRocm
import helion.language as hl


class TestInlineAsmElementwise(RefEagerTestDisabled, TestCase):
    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    def test_inline_asm_simple(self):
        """Test basic inline_asm_elementwise with simple assembly"""

        @helion.kernel(autotune_effort="none")
        def kernel_simple_asm(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                val = x[tile]
                # Simple mov instruction - copy input to output
                result_val = hl.inline_asm_elementwise(
                    "mov.u32 $0, $1;",
                    "=r,r",
                    [val],
                    dtype=val.dtype,
                    is_pure=True,
                    pack=1,
                )
                result[tile] = result_val
            return result

        x = torch.randint(0, 100, [16], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(kernel_simple_asm, (x,))
        self.assertExpectedJournal(code)
        torch.testing.assert_close(result, x)

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    def test_inline_asm_shift_operation(self):
        """Test inline_asm_elementwise with shift operation (similar to Triton test)"""

        @helion.kernel(autotune_effort="none")
        def kernel_shift_asm(x: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                val_x = x[tile]
                val_y = y[tile]
                shift_val = hl.full(tile, n, dtype=torch.int32)
                # Shift left wrap operation
                result_val = hl.inline_asm_elementwise(
                    "shf.l.wrap.b32 $0, $1, $2, $3;",
                    "=r,r,r,r",
                    [val_x, val_y, shift_val],
                    dtype=torch.int32,
                    is_pure=True,
                    pack=1,
                )
                result[tile] = result_val
            return result

        shape = [128]
        x = torch.randint(0, 2**16, shape, device=DEVICE, dtype=torch.int32)
        y = torch.randint(0, 2**16, shape, device=DEVICE, dtype=torch.int32)
        n = 17

        code, result = code_and_output(kernel_shift_asm, (x, y, n))
        self.assertExpectedJournal(code)

        # Expected: (y << n) | (x >> (32 - n))
        expected = (y << n) | (x >> (32 - n))
        torch.testing.assert_close(result, expected)

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    def test_inline_asm_multiple_outputs(self):
        """Test inline_asm_elementwise with multiple outputs"""

        @helion.kernel(autotune_effort="none")
        def kernel_multiple_outputs(
            a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_c = torch.empty_like(a)
            result_d = torch.empty_like(a)

            for tile in hl.tile(a.shape):
                val_a = a[tile]
                val_b = b[tile]

                # C = A - B, D = B - A
                c_val, d_val = hl.inline_asm_elementwise(
                    """
                    sub.u32 $0, $2, $3;
                    sub.u32 $1, $3, $2;
                    """,
                    "=r,=r,r,r",
                    [val_a, val_b],
                    dtype=(torch.int32, torch.int32),
                    is_pure=True,
                    pack=1,
                )
                result_c[tile] = c_val
                result_d[tile] = d_val

            return result_c, result_d

        shape = [64]
        a = torch.randint(0, 2**16, shape, device=DEVICE, dtype=torch.int32)
        b = torch.randint(0, 2**16, shape, device=DEVICE, dtype=torch.int32)

        code, (result_c, result_d) = code_and_output(kernel_multiple_outputs, (a, b))
        self.assertExpectedJournal(code)

        # Expected results
        expected_c = a - b
        expected_d = b - a

        torch.testing.assert_close(result_c, expected_c)
        torch.testing.assert_close(result_d, expected_d)

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    def test_inline_asm_packed(self):
        """Test inline_asm_elementwise with pack > 1"""

        @helion.kernel(autotune_effort="none")
        def kernel_packed_asm(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                val = x[tile]
                # Shift 4x8bit values together, pack=4
                result_val = hl.inline_asm_elementwise(
                    "and.b32 $0, $1, 0x1F1F1F1F; shl.b32 $0, $0, 3;",
                    "=r,r",
                    [val],
                    dtype=torch.int8,
                    is_pure=True,
                    pack=4,
                )
                result[tile] = result_val
            return result

        shape = [512]
        x = torch.randint(0, 256, shape, device=DEVICE, dtype=torch.uint8)

        code, result = code_and_output(kernel_packed_asm, (x,))
        self.assertExpectedJournal(code)

        # Expected: x shifted left by 3 (x << 3)
        expected = x << 3
        torch.testing.assert_close(result, expected)

    def test_inline_asm_error_cases(self):
        """Test error cases for inline_asm_elementwise"""

        @helion.kernel(autotune_effort="none")
        def kernel_invalid_asm(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                # Should raise error - invalid dtype
                result_val = hl.inline_asm_elementwise(
                    "mov.u32 $0, $1;",
                    "=r,r",
                    [x[tile]],
                    dtype="invalid_dtype",  # Invalid dtype
                    is_pure=True,
                    pack=1,
                )
                result[tile] = result_val
            return result

        x = torch.randint(0, 100, [16], device=DEVICE, dtype=torch.int32)
        with self.assertRaises(helion.exc.InvalidAPIUsage):
            code, result = code_and_output(kernel_invalid_asm, (x,))

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    def test_inline_asm_empty_args(self):
        """Test inline_asm_elementwise with empty args (should work like Triton)"""

        @helion.kernel(autotune_effort="none")
        def kernel_empty_args(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                # Empty args should work - generates output with context shape
                result_val = hl.inline_asm_elementwise(
                    "mov.u32 $0, 42;",  # No input registers, just output constant
                    "=r",  # Only output constraint
                    [],  # Empty args
                    dtype=torch.int32,
                    is_pure=True,
                    pack=1,
                )
                result[tile] = result_val
            return result

        x = torch.randint(0, 100, [16], device=DEVICE, dtype=torch.int32)
        # This should work without error
        code, result = code_and_output(kernel_empty_args, (x,))
        self.assertExpectedJournal(code)

        # Should create a tensor filled with 42
        expected = torch.full([16], 42, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

    @skipIfRocm("only works on cuda")
    def test_inline_asm_basic_compilation(self):
        """Test that inline_asm_elementwise compiles without errors (no CUDA requirement)"""

        @helion.kernel(autotune_effort="none")
        def kernel_basic(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                # Simple compilation test
                result_val = hl.inline_asm_elementwise(
                    "mov.u32 $0, $1;",
                    "=r,r",
                    [x[tile]],
                    dtype=torch.int32,
                    is_pure=True,
                    pack=1,
                )
                result[tile] = result_val
            return result

        x = torch.randint(0, 100, [16], device=DEVICE, dtype=torch.int32)
        # Just test that it compiles
        code, result = code_and_output(kernel_basic, (x,))
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
