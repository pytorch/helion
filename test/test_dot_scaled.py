from __future__ import annotations

import functools
import os
import sys
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfRefEager
import helion.language as hl
from helion.runtime.settings import _get_backend

# tl.dot_scaled requires SM 10.0+ (B200 / compute capability 10.0)
requires_sm100 = skipIfCudaCapabilityLessThan(
    (10, 0), reason="tl.dot_scaled requires CUDA capability >= 10.0 (B200+)"
)


@functools.lru_cache(maxsize=1)
def _supports_fp16_dot_scaled() -> bool:
    if _get_backend() not in ("triton", "tileir"):
        return False
    import triton
    import triton.language as tl

    @triton.jit
    def _probe(
        x_ptr,
        x_scale_ptr,
        y_ptr,
        y_scale_ptr,
        out_ptr,
    ):
        offs_m = tl.arange(0, 16)
        offs_n = tl.arange(0, 16)
        offs_k = tl.arange(0, 64)
        offs_s = tl.arange(0, 2)
        x = tl.load(x_ptr + (offs_m[:, None] * 64 + offs_k[None, :]))
        x_scale = tl.load(x_scale_ptr + (offs_m[:, None] * 2 + offs_s[None, :]))
        y = tl.load(y_ptr + (offs_k[:, None] * 16 + offs_n[None, :]))
        y_scale = tl.load(y_scale_ptr + (offs_n[:, None] * 2 + offs_s[None, :]))
        out = tl.dot_scaled(x, x_scale, "fp16", y, y_scale, "fp16")
        tl.store(out_ptr + (offs_m[:, None] * 16 + offs_n[None, :]), out)

    x = torch.randn(16, 64, device=DEVICE, dtype=torch.float16)
    x_scale = torch.full((16, 2), 127, device=DEVICE, dtype=torch.uint8)
    y = torch.randn(64, 16, device=DEVICE, dtype=torch.float16)
    y_scale = torch.full((16, 2), 127, device=DEVICE, dtype=torch.uint8)
    out = torch.empty((16, 16), device=DEVICE, dtype=torch.float32)

    # Suppress stderr during the probe because Triton may emit noisy LLVM/PTXAS
    # warnings when the hardware does not support fp16 dot_scaled.
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                _probe[(1,)](x, x_scale, y, y_scale, out, num_warps=1, num_stages=1)
                torch.cuda.synchronize()
            except RuntimeError:
                return False
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
    return True


@onlyBackends(["triton"])
class TestDotScaled(TestCase):
    def _skip_if_fp16_dot_scaled_unsupported(self) -> None:
        if not _supports_fp16_dot_scaled():
            self.skipTest(
                "fp16 tl.dot_scaled is unsupported on this Triton/CUDA target"
            )

    @requires_sm100
    def test_invalid_format_string(self):
        """Verify that an invalid format string raises ValueError."""
        with self.assertRaises((ValueError, helion.exc.InternalError)):

            @helion.kernel(config=helion.Config(block_sizes=[32, 32, 32]))
            def bad_format_kernel(
                x: torch.Tensor,
                x_scale: torch.Tensor,
                y: torch.Tensor,
                y_scale: torch.Tensor,
            ) -> torch.Tensor:
                m, k = x.size()
                _, n = y.size()
                out = torch.empty([m, n], dtype=torch.float32, device=x.device)
                for tile_m, tile_n in hl.tile([m, n]):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = hl.dot_scaled(
                            x[tile_m, tile_k],
                            x_scale[tile_m, tile_k],
                            "invalid_format",
                            y[tile_k, tile_n],
                            y_scale[tile_k, tile_n],
                            "e4m3",
                            acc=acc,
                        )
                    out[tile_m, tile_n] = acc
                return out

            x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
            x_scale = torch.ones(64, 4, device=DEVICE, dtype=torch.float32)
            y = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
            y_scale = torch.ones(4, 64, device=DEVICE, dtype=torch.float32)
            bad_format_kernel(x, x_scale, y, y_scale)

    @requires_sm100
    def test_3d_tensor_rejected(self):
        """Verify that 3D tensors are rejected."""
        with self.assertRaises((ValueError, helion.exc.ControlFlowTensorMismatch)):

            @helion.kernel(config=helion.Config(block_sizes=[32, 32, 32]))
            def bad_3d_kernel(
                x: torch.Tensor,
                x_scale: torch.Tensor,
                y: torch.Tensor,
                y_scale: torch.Tensor,
            ) -> torch.Tensor:
                b, m, k = x.size()
                _, _, n = y.size()
                out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
                for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
                    acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = hl.dot_scaled(
                            x[tile_b, tile_m, tile_k],
                            x_scale[tile_b, tile_m, tile_k],
                            "e4m3",
                            y[tile_b, tile_k, tile_n],
                            y_scale[tile_b, tile_k, tile_n],
                            "e4m3",
                            acc=acc,
                        )
                    out[tile_b, tile_m, tile_n] = acc
                return out

            x = torch.randn(2, 64, 64, device=DEVICE, dtype=torch.float32)
            x_scale = torch.ones(2, 64, 4, device=DEVICE, dtype=torch.float32)
            y = torch.randn(2, 64, 64, device=DEVICE, dtype=torch.float32)
            y_scale = torch.ones(2, 4, 64, device=DEVICE, dtype=torch.float32)
            bad_3d_kernel(x, x_scale, y, y_scale)

    @requires_sm100
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_codegen_contains_dot_scaled(self):
        """Verify generated Triton code contains tl.dot_scaled(.

        Uses fp16 format with float16 data.  Scale tensors are uint8
        (representing float8_e8m0) with shape [dim, K // 32].
        K is not tiled -- full K loaded via ':' so scale slicing is correct.
        """
        self._skip_if_fp16_dot_scaled_unsupported()
        M, N, K = 64, 64, 64
        BLOCK = 32
        SCALE_FACTOR = 32  # uint8 scale => 32 elements per scale

        @helion.kernel(config=helion.Config(block_sizes=[BLOCK, BLOCK]))
        def scaled_kernel(
            x: torch.Tensor,
            x_scale: torch.Tensor,
            y: torch.Tensor,
            y_scale: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = hl.dot_scaled(
                    x[tile_m, :],
                    x_scale[tile_m, :],
                    "fp16",
                    y[:, tile_n],
                    y_scale[tile_n, :],
                    "fp16",
                )
            return out

        x = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        x_scale = torch.full(
            (M, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )
        y = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        y_scale = torch.full(
            (N, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )

        code, result = code_and_output(scaled_kernel, (x, x_scale, y, y_scale))
        self.assertIn("tl.dot_scaled(", code)

    @requires_sm100
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_with_accumulator(self):
        """Verify codegen with fused accumulator path using e4m3 format."""
        M, N, K = 64, 64, 64
        BLOCK = 32
        SCALE_FACTOR = 32

        @helion.kernel(config=helion.Config(block_sizes=[BLOCK, BLOCK]))
        def scaled_acc_kernel(
            x: torch.Tensor,
            x_scale: torch.Tensor,
            y: torch.Tensor,
            y_scale: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                acc = hl.dot_scaled(
                    x[tile_m, :],
                    x_scale[tile_m, :],
                    "e4m3",
                    y[:, tile_n],
                    y_scale[tile_n, :],
                    "e4m3",
                    acc=acc,
                )
                out[tile_m, tile_n] = acc
            return out

        x = (torch.randn(M, K, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        x_scale = torch.full(
            (M, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )
        y = (torch.randn(K, N, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        y_scale = torch.full(
            (N, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )

        code, result = code_and_output(scaled_acc_kernel, (x, x_scale, y, y_scale))
        self.assertIn("tl.dot_scaled(", code)
        self.assertIn("acc=", code)

    @requires_sm100
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_out_dtype_float32(self):
        """Verify codegen with explicit out_dtype=float32.

        Note: Triton currently only supports float32 for dot_scaled out_dtype.
        """
        self._skip_if_fp16_dot_scaled_unsupported()
        M, N, K = 64, 64, 64
        BLOCK = 32
        SCALE_FACTOR = 32

        @helion.kernel(config=helion.Config(block_sizes=[BLOCK, BLOCK]))
        def scaled_out_dtype_kernel(
            x: torch.Tensor,
            x_scale: torch.Tensor,
            y: torch.Tensor,
            y_scale: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = hl.dot_scaled(
                    x[tile_m, :],
                    x_scale[tile_m, :],
                    "fp16",
                    y[:, tile_n],
                    y_scale[tile_n, :],
                    "fp16",
                    out_dtype=torch.float32,
                )
            return out

        x = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        x_scale = torch.full(
            (M, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )
        y = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        y_scale = torch.full(
            (N, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )

        code, result = code_and_output(
            scaled_out_dtype_kernel, (x, x_scale, y, y_scale)
        )
        self.assertIn("tl.dot_scaled(", code)
        self.assertIn("out_dtype=tl.float32", code)

    @requires_sm100
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_no_acc_codegen(self):
        """Verify codegen without accumulator (acc=None)."""
        self._skip_if_fp16_dot_scaled_unsupported()
        M, N, K = 64, 64, 64
        BLOCK = 32
        SCALE_FACTOR = 32

        @helion.kernel(config=helion.Config(block_sizes=[BLOCK, BLOCK]))
        def scaled_no_acc_kernel(
            x: torch.Tensor,
            x_scale: torch.Tensor,
            y: torch.Tensor,
            y_scale: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = hl.dot_scaled(
                    x[tile_m, :],
                    x_scale[tile_m, :],
                    "fp16",
                    y[:, tile_n],
                    y_scale[tile_n, :],
                    "fp16",
                )
            return out

        x = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        x_scale = torch.full(
            (M, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )
        y = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        y_scale = torch.full(
            (N, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )

        code, result = code_and_output(scaled_no_acc_kernel, (x, x_scale, y, y_scale))
        self.assertIn("tl.dot_scaled(", code)

    @requires_sm100
    def test_numerical_correctness_fp16(self):
        """Verify dot_scaled with fp16 format produces correct output.

        With scale=127 (e8m0 for 1.0), dot_scaled(x, scale, 'fp16',
        y, scale, 'fp16') should match torch.mm(x.float(), y.float()).
        """
        self._skip_if_fp16_dot_scaled_unsupported()
        M, N, K = 64, 64, 64
        BLOCK = 32
        SCALE_FACTOR = 32

        @helion.kernel(config=helion.Config(block_sizes=[BLOCK, BLOCK]))
        def scaled_kernel(
            x: torch.Tensor,
            x_scale: torch.Tensor,
            y: torch.Tensor,
            y_scale: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = hl.dot_scaled(
                    x[tile_m, :],
                    x_scale[tile_m, :],
                    "fp16",
                    y[:, tile_n],
                    y_scale[tile_n, :],
                    "fp16",
                )
            return out

        x = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        x_scale = torch.full(
            (M, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )
        y = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        y_scale = torch.full(
            (N, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )

        result = scaled_kernel(x, x_scale, y, y_scale)
        expected = torch.mm(x.float(), y.float())
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    @requires_sm100
    def test_numerical_correctness_e4m3(self):
        """Verify dot_scaled with e4m3 format produces correct output.

        With scale=127 (e8m0 for 1.0), dot_scaled should match
        torch.mm(x.float(), y.float()) within FP8 precision.
        """
        M, N, K = 64, 64, 64
        BLOCK = 32
        SCALE_FACTOR = 32

        @helion.kernel(config=helion.Config(block_sizes=[BLOCK, BLOCK]))
        def scaled_kernel(
            x: torch.Tensor,
            x_scale: torch.Tensor,
            y: torch.Tensor,
            y_scale: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                acc = hl.dot_scaled(
                    x[tile_m, :],
                    x_scale[tile_m, :],
                    "e4m3",
                    y[:, tile_n],
                    y_scale[tile_n, :],
                    "e4m3",
                    acc=acc,
                )
                out[tile_m, tile_n] = acc
            return out

        x = (torch.randn(M, K, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        x_scale = torch.full(
            (M, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )
        y = (torch.randn(K, N, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        y_scale = torch.full(
            (N, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )

        result = scaled_kernel(x, x_scale, y, y_scale)
        expected = torch.mm(x.float(), y.float())
        torch.testing.assert_close(result, expected, atol=0.5, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
