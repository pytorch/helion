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


if __name__ == "__main__":
    unittest.main()
