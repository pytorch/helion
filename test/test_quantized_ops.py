from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfTileIR
import helion.language as hl


def _dequant_e2m1(nibbles: torch.Tensor) -> torch.Tensor:
    sign = ((nibbles >> 3) & 1).to(torch.float32)
    u = (nibbles & 0x7).to(torch.float32)
    abs_val = torch.where(
        u < 4.0,
        u * 0.5,
        torch.where(u < 6.0, u - 2.0, u * 2.0 - 8.0),
    )
    return abs_val * (1.0 - 2.0 * sign)


def _assert_same_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Bitwise equality (NaNs only need to be NaN), so -0.0 and subnormals count."""
    int_dtype = {2: torch.int16, 4: torch.int32}[actual.element_size()]
    nan = torch.isnan(expected)
    torch.testing.assert_close(torch.isnan(actual), nan)
    torch.testing.assert_close(
        actual[~nan].view(int_dtype), expected[~nan].view(int_dtype)
    )


@onlyBackends(["cute"])
class TestCuteQuantizedOps(RefEagerTestDisabled, TestCase):
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="FP4/FP8 conversion instructions require Blackwell"
    )
    def test_fp8_e4m3fn_to_float32(self):
        @helion.kernel(autotune_effort="none")
        def fp8_to_f32(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
            for tile in hl.tile(x.size(0), block_size=16):
                out[tile] = x[tile].to(torch.float32)
            return out

        x = (torch.rand((16,), device=DEVICE, dtype=torch.float32) + 0.5).to(
            torch.float8_e4m3fn
        )
        code, result = code_and_output(fp8_to_f32, (x,))
        torch.testing.assert_close(result, x.to(torch.float32))
        self.assertIn("_cute_fp8e4m3fn_to_float32", code)

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="FP4/FP8 conversion instructions require Blackwell"
    )
    def test_float4_e2m1fn_x2_to_float32(self):
        @helion.kernel(autotune_effort="none")
        def fp4_to_f32_pair(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            lo = torch.empty(x.shape, dtype=torch.float32, device=x.device)
            hi = torch.empty(x.shape, dtype=torch.float32, device=x.device)
            for tile in hl.tile(x.size(0), block_size=16):
                lo_value, hi_value = hl.float4_e2m1fn_x2_to_float32(x[tile])
                lo[tile] = lo_value
                hi[tile] = hi_value
            return lo, hi

        raw = torch.tensor(
            [
                0x10,
                0x21,
                0x32,
                0x43,
                0x54,
                0x65,
                0x76,
                0x87,
                0x98,
                0xA9,
                0xBA,
                0xCB,
                0xDC,
                0xED,
                0xFE,
                0xFF,
            ],
            dtype=torch.uint8,
            device=DEVICE,
        )
        code, (lo, hi) = code_and_output(
            fp4_to_f32_pair,
            (raw.view(torch.float4_e2m1fn_x2),),
        )
        raw_i32 = raw.to(torch.int32)
        torch.testing.assert_close(lo, _dequant_e2m1(raw_i32 & 0xF))
        torch.testing.assert_close(hi, _dequant_e2m1((raw_i32 >> 4) & 0xF))
        self.assertIn("_cute_float4_e2m1fn_x2_to_float32", code)

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="FP4 conversion instructions require Blackwell"
    )
    def test_load_float4_e2m1fn_x16_to_float16(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def fp4_to_f16_lanes(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                (offsets.size(0), 16), dtype=torch.float16, device=x.device
            )
            for tile in hl.tile(offsets.size(0), block_size=4):
                lanes = hl.load_float4_e2m1fn_x16_to_float16(
                    x,
                    offsets[tile],
                    extra_mask=tile.index < offsets.size(0),
                )
                for i in hl.static_range(16):
                    out[tile, i] = lanes[i]
            return out

        raw = torch.arange(16, dtype=torch.uint8, device=DEVICE) * 17
        offsets = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        code, result = code_and_output(fp4_to_f16_lanes, (raw, offsets))

        raw_i32 = raw.view(2, 8).to(torch.int32)
        nibbles = torch.stack((raw_i32 & 0xF, (raw_i32 >> 4) & 0xF), dim=-1)
        expected = _dequant_e2m1(nibbles.reshape(2, 16)).to(torch.float16)
        torch.testing.assert_close(result, expected)
        self.assertIn("cute.arch.load", code)
        self.assertIn("cutlass.Uint64", code)
        self.assertIn("_cute_float4_e2m1fn_x16_to_float16", code)

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="Packed BF16 conversion requires Blackwell"
    )
    def test_load_bfloat16_x16_to_float16(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def bf16_to_f16_lanes(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                (offsets.size(0), 16), dtype=torch.float16, device=x.device
            )
            for tile in hl.tile(offsets.size(0), block_size=4):
                lanes = hl.load_bfloat16_x16_to_float16(
                    x,
                    offsets[tile],
                    extra_mask=tile.index < offsets.size(0),
                )
                for i in hl.static_range(16):
                    out[tile, i] = lanes[i]
            return out

        values = torch.randn(32, dtype=torch.bfloat16, device=DEVICE)
        offsets = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        code, result = code_and_output(bf16_to_f16_lanes, (values, offsets))

        torch.testing.assert_close(result, values.view(2, 16).to(torch.float16))
        self.assertIn("cute.arch.load", code)
        self.assertIn("_cute_bfloat16_x16_to_float16", code)

    @skipIfNotCUDA()
    def test_grouped_row_index_precedence(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def grouped_row_indices(out: torch.Tensor) -> torch.Tensor:
            for program in hl.grid(out.size(0)):
                row = program * 2 + 1
                out[program] = (row // 128) * 100 + row % 32
            return out

        out = torch.empty(130, dtype=torch.int64, device=DEVICE)
        code, result = code_and_output(grouped_row_indices, (out,))

        program = torch.arange(130, dtype=torch.int64, device=DEVICE)
        row = program * 2 + 1
        expected = (row // 128) * 100 + row % 32
        torch.testing.assert_close(result, expected)
        self.assertIn("//", code)
        self.assertIn("%", code)


@onlyBackends(["triton"])
class TestTritonQuantizedOps(RefEagerTestDisabled, TestCase):
    @unittest.skipIf(DEVICE.type != "cuda", "needs a CUDA or ROCm GPU")
    @skipIfTileIR("TileIR's plain decode is untested")
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="FP4/FP8 conversion instructions require Blackwell"
    )
    def test_float4_e2m1fn_x2_to_float32(self):
        # Also covers fp4x2 host args viewed as uint8 for Triton.
        @helion.kernel(autotune_effort="none")
        def fp4_to_f32_pair(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            lo = torch.empty(x.shape, dtype=torch.float32, device=x.device)
            hi = torch.empty(x.shape, dtype=torch.float32, device=x.device)
            for tile in hl.tile(x.size(0), block_size=16):
                lo_value, hi_value = hl.float4_e2m1fn_x2_to_float32(x[tile])
                lo[tile] = lo_value
                hi[tile] = hi_value
            return lo, hi

        # Every byte, so the low and high nibbles differ; bits catch -0.0.
        raw = torch.arange(256, dtype=torch.uint8, device=DEVICE)
        code, (lo, hi) = code_and_output(
            fp4_to_f32_pair,
            (raw.view(torch.float4_e2m1fn_x2),),
        )
        raw_i32 = raw.to(torch.int32)
        _assert_same_bits(lo, _dequant_e2m1(raw_i32 & 0xF))
        _assert_same_bits(hi, _dequant_e2m1((raw_i32 >> 4) & 0xF))
        self.assertIn(" = load.to(tl.int16)", code)
        if torch.version.hip is None:
            self.assertIn("[fp4_packed_", code)
            self.assertIn("=f,=f,h", code)
            self.assertIn("tl.inline_asm_elementwise", code)
        else:
            # PTX does not lower on ROCm; the decode is plain arithmetic.
            self.assertNotIn("tl.inline_asm_elementwise", code)

    @unittest.skipIf(DEVICE.type != "cuda", "needs a CUDA or ROCm GPU")
    @skipIfTileIR("TileIR's plain decode is untested")
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="FP4 conversion instructions require Blackwell"
    )
    def test_load_float4_e2m1fn_x16_to_float16(self):
        @helion.kernel(autotune_effort="none")
        def fp4_to_f16_lanes(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                (offsets.size(0), 16),
                dtype=torch.float16,
                device=x.device,
            )
            for tile in hl.tile(offsets.size(0), block_size=4):
                lanes = hl.load_float4_e2m1fn_x16_to_float16(
                    x,
                    offsets[tile],
                    extra_mask=tile.index < offsets.size(0),
                )
                for i in hl.static_range(16):
                    out[tile, i] = lanes[i]
            return out

        raw = torch.arange(256, dtype=torch.uint8, device=DEVICE)
        offsets = torch.arange(32, dtype=torch.int64, device=DEVICE)
        code, result = code_and_output(fp4_to_f16_lanes, (raw, offsets))

        raw_i32 = raw.view(32, 8).to(torch.int32)
        nibbles = torch.stack((raw_i32 & 0xF, (raw_i32 >> 4) & 0xF), dim=-1)
        expected = _dequant_e2m1(nibbles.reshape(32, 16)).to(torch.float16)
        _assert_same_bits(result, expected)
        self.assertIn("tl.pointer_type(tl.uint64)", code)
        if torch.version.hip is None:
            self.assertIn("[fp4_qword_", code)
            self.assertIn("=h,=h", code)
        else:
            self.assertIn("fp4_qword_", code)
            self.assertNotIn("tl.inline_asm_elementwise", code)

    @unittest.skipIf(DEVICE.type != "cuda", "needs a CUDA or ROCm GPU")
    @skipIfTileIR("TileIR's plain decode is untested")
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="Packed BF16 conversion requires Blackwell"
    )
    def test_load_bfloat16_x16_to_float16(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def bf16_to_f16_lanes(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                (offsets.size(0), 16), dtype=torch.float16, device=x.device
            )
            for tile in hl.tile(offsets.size(0), block_size=4):
                lanes = hl.load_bfloat16_x16_to_float16(
                    x,
                    offsets[tile],
                    extra_mask=tile.index < offsets.size(0),
                )
                for i in hl.static_range(16):
                    out[tile, i] = lanes[i]
            return out

        # Distinct lanes plus the edge cases: signed zero, f16 subnormal and
        # underflow, f16 overflow to inf, inf and NaN.
        edges = [0.0, -0.0, 2.0**-24, 2.0**-30, 70000.0, -1e30, float("inf")]
        edges += [float("-inf"), float("nan"), 65504.0, -3.5, 1e-40]
        values = torch.cat([torch.randn(20), torch.tensor(edges)]).to(
            device=DEVICE, dtype=torch.bfloat16
        )
        offsets = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        code, result = code_and_output(bf16_to_f16_lanes, (values, offsets))

        _assert_same_bits(result, values.view(2, 16).to(torch.float16))
        self.assertIn("tl.pointer_type(tl.uint64)", code)
        if torch.version.hip is None:
            self.assertIn("[bf16_qword_", code)
            self.assertIn("cvt.rn.f16.bf16", code)
        else:
            self.assertIn("bf16_qword_", code)
            self.assertNotIn("tl.inline_asm_elementwise", code)


if __name__ == "__main__":
    unittest.main()
