from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing import skipIfXPU
import helion.language as hl
from helion.runtime.settings import _get_backend

# The call each backend lowers hl.inline_asm_elementwise to.
_INLINE_ASM_CALL = (
    "_cute_inline_asm_elementwise"
    if _get_backend() == "cute"
    else "tl.inline_asm_elementwise"
)

# Two f32 lanes per asm invocation, the same PTX examples/blackwell_attention.py uses.
_MUL_F32X2_ASM = """
{
    .reg .b64 ra, rb, rc;
    mov.b64 ra, { $2, $3 };
    mov.b64 rb, { $4, $5 };
    mul.f32x2 rc, ra, rb;
    mov.b64 { $0, $1 }, rc;
}
"""


@onlyBackends(["triton", "cute"])
class TestInlineAsmElementwise(RefEagerTestDisabled, TestCase):
    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
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
        torch.testing.assert_close(result, x)
        self.assertIn(_INLINE_ASM_CALL, code)

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
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

        # Expected: (y << n) | (x >> (32 - n))
        expected = (y << n) | (x >> (32 - n))
        torch.testing.assert_close(result, expected)
        self.assertIn(_INLINE_ASM_CALL, code)

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
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

        # Expected results
        expected_c = a - b
        expected_d = b - a

        torch.testing.assert_close(result_c, expected_c)
        torch.testing.assert_close(result_d, expected_d)

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
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
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

        # Should create a tensor filled with 42
        expected = torch.full([16], 42, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

    @skipIfRocm("only works on cuda")
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
    @skipIfXPU("PTX inline assembly (mov.u32) not supported on XPU backend")
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


@onlyBackends(["triton", "cute"])
class TestInlineAsmElementwisePacked(RefEagerTestDisabled, TestCase):
    """``pack > 1`` lowerings shared by the Triton and CuTe backends."""

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
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

        # Expected: x shifted left by 3 (x << 3)
        expected = x << 3
        torch.testing.assert_close(result, expected)

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
    @skipIfCudaCapabilityLessThan((10, 0), reason="mul.f32x2 requires sm_100")
    def test_inline_asm_packed_f32x2_broadcast(self):
        """pack=2 f32x2 PTX with a broadcast second operand (blackwell_attention)."""

        @helion.kernel(autotune_effort="none")
        def kernel_mul_f32x2(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.shape):
                a = x[tile_m, tile_n]
                b = alpha[tile_m]
                result[tile_m, tile_n] = hl.inline_asm_elementwise(
                    _MUL_F32X2_ASM,
                    "=r,=r,r,r,r,r",
                    [a, b[:, None]],
                    dtype=torch.float32,
                    is_pure=True,
                    pack=2,
                )
            return result

        x = torch.randn([64, 128], device=DEVICE, dtype=torch.float32)
        alpha = torch.randn([64], device=DEVICE, dtype=torch.float32)
        _code, result = code_and_output(kernel_mul_f32x2, (x, alpha))
        torch.testing.assert_close(result, x * alpha[:, None])

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
    def test_inline_asm_packed_multiple_outputs(self):
        """pack=2 over 16-bit lanes with two packed outputs."""

        @helion.kernel(autotune_effort="none")
        def kernel_packed_outputs(
            a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_c = torch.empty_like(a)
            result_d = torch.empty_like(a)
            for tile in hl.tile(a.shape):
                # Two f16x2 registers in, swapped out: C = B, D = A
                c_val, d_val = hl.inline_asm_elementwise(
                    "mov.b32 $0, $3; mov.b32 $1, $2;",
                    "=r,=r,r,r",
                    [a[tile], b[tile]],
                    dtype=(torch.float16, torch.float16),
                    is_pure=True,
                    pack=2,
                )
                result_c[tile] = c_val
                result_d[tile] = d_val
            return result_c, result_d

        a = torch.randn([256], device=DEVICE, dtype=torch.float16)
        b = torch.randn([256], device=DEVICE, dtype=torch.float16)
        _code, (result_c, result_d) = code_and_output(kernel_packed_outputs, (a, b))
        torch.testing.assert_close(result_c, b)
        torch.testing.assert_close(result_d, a)


@onlyBackends(["cute"])
class TestCuteInlineAsmElementwise(RefEagerTestDisabled, TestCase):
    def test_pure_inline_asm_computed_scan_and_reuse(self) -> None:
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def kernel(
            x: torch.Tensor, bias: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            output = torch.empty_like(x)
            reused = torch.empty_like(x)
            for row, col in hl.tile(x.shape, block_size=[2, 32]):
                values = hl.inline_asm_elementwise(
                    "add.f32 $0, $1, $2;",
                    "=f,f,f",
                    [x[row, col], bias[row, None]],
                    dtype=torch.float32,
                    is_pure=True,
                    pack=1,
                )
                output[row, col] = hl.cumsum(values, dim=-1)
                reused[row, col] = values * 2
            return output, reused

        x = torch.arange(5 * 65, device=DEVICE, dtype=torch.float32).reshape(5, 65)
        bias = torch.arange(5, device=DEVICE, dtype=torch.float32)
        _, (output, reused) = code_and_output(kernel, (x, bias))
        values = x + bias[:, None]
        expected = torch.cat([part.cumsum(-1) for part in values.split(32, -1)], -1)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(reused, values * 2, rtol=0, atol=0)

    def test_impure_inline_asm_computed_scan_rejected(self) -> None:
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                values = hl.inline_asm_elementwise(
                    "mov.b32 $0, $1;",
                    "=f,f",
                    [x[row, :]],
                    dtype=torch.float32,
                    is_pure=False,
                    pack=1,
                )
                output[row, :] = hl.cumsum(values, dim=-1)
            return output

        x = torch.ones(3, 65, device=DEVICE)
        with self.assertRaisesRegex(helion.exc.InvalidConfig, "fragment"):
            code_and_output(
                kernel, (x,), block_sizes=[1], cute_fragment_scan="cooperative"
            )

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
    def test_inline_asm_simple(self):
        @helion.kernel(autotune_effort="none")
        def kernel_simple_asm(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.size(0), block_size=16):
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
        code, result = code_and_output(kernel_simple_asm, (x,))
        torch.testing.assert_close(result, x)
        self.assertIn("_cute_inline_asm_elementwise", code)

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
    def test_inline_asm_multiple_outputs(self):
        @helion.kernel(autotune_effort="none")
        def kernel_multiple_outputs(
            a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_c = torch.empty_like(a)
            result_d = torch.empty_like(a)
            for tile in hl.tile(a.size(0), block_size=16):
                c_val, d_val = hl.inline_asm_elementwise(
                    """
                    sub.u32 $0, $2, $3;
                    sub.u32 $1, $3, $2;
                    """,
                    "=r,=r,r,r",
                    [a[tile], b[tile]],
                    dtype=(torch.int32, torch.int32),
                    is_pure=True,
                    pack=1,
                )
                result_c[tile] = c_val
                result_d[tile] = d_val
            return result_c, result_d

        shape = [16]
        a = torch.randint(0, 2**16, shape, device=DEVICE, dtype=torch.int32)
        b = torch.randint(0, 2**16, shape, device=DEVICE, dtype=torch.int32)
        _code, (result_c, result_d) = code_and_output(kernel_multiple_outputs, (a, b))
        torch.testing.assert_close(result_c, a - b)
        torch.testing.assert_close(result_d, b - a)

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
    @skipIfCudaCapabilityLessThan((10, 0), reason="mul.f32x2 requires sm_100")
    def test_inline_asm_packed_vector_width(self):
        """pack=2 under a vectorized lane layout: the extra ``pack=`` keyword must
        make the pure-lane-packet matcher decline the call, not miscompile it."""

        @helion.kernel(
            autotune_effort="none",
            static_shapes=False,
            config=helion.Config(
                block_sizes=[1024],
                num_threads=[128],
                cute_vector_widths=[4],
                cute_lane_layouts=["blocked"],
                cute_cluster_n=1,
            ),
        )
        def kernel_mul_f32x2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                result[tile] = hl.inline_asm_elementwise(
                    _MUL_F32X2_ASM,
                    "=r,=r,r,r,r,r",
                    [x[tile], y[tile]],
                    dtype=torch.float32,
                    is_pure=True,
                    pack=2,
                )
            return result

        x = torch.randn([4093], device=DEVICE, dtype=torch.float32)
        y = torch.randn([4093], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(kernel_mul_f32x2, (x, y))
        torch.testing.assert_close(result, x * y)
        self.assertIn("pack=2", code)

    @pytest.mark.skipif(
        DEVICE.type != "cuda", reason="inline_asm_elementwise is only supported on CUDA"
    )
    @skipIfRocm("only works on cuda")
    @skipIfTileIR("TileIR does not support inline_asm_elementwise")
    def test_pack_impure_error(self):
        """``pack > 1`` with ``is_pure=False`` is rejected: the per-lane lowering
        would replay the asm's side effects ``pack`` times per element."""

        @helion.kernel(autotune_effort="none")
        def kernel_impure_packed(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.size(0), block_size=16):
                result[tile] = hl.inline_asm_elementwise(
                    "mov.b32 $0, $1;",
                    "=r,r",
                    [x[tile]],
                    dtype=torch.float16,
                    is_pure=False,
                    pack=2,
                )
            return result

        x = torch.randn([16], device=DEVICE, dtype=torch.float16)
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"hl\.inline_asm_elementwise with pack != 1 and is_pure=False",
        ):
            code_and_output(kernel_impure_packed, (x,))


if __name__ == "__main__":
    unittest.main()
