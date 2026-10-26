from __future__ import annotations

from pathlib import Path
import re
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfLowVRAM
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
import helion.language as hl
from helion.runtime.settings import _get_backend

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


@helion.kernel
def cast_after_div(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(ref)
    for tile in helion.language.tile(out.size()):
        a = x[tile].to(torch.float32)
        p = a / (1 + torch.exp(-a))
        out[tile] = p.to(ref.dtype)
    return out


@onlyBackends(["triton", "cute"])
class TestGenerateAst(RefEagerTestBase, TestCase):
    def test_add1d(self):
        args = (torch.randn([4096], device=DEVICE), torch.randn([4096], device=DEVICE))
        code, result = code_and_output(basic_kernels.add, args, block_size=1024)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add2d(self):
        args = (
            torch.randn([100, 500], device=DEVICE),
            torch.randn([100, 500], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[1024, 1], flatten_loop=True
        )
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add2d_loop_order(self):
        args = (
            torch.randn([100, 500], device=DEVICE),
            torch.randn([100, 500], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add,
            args,
            block_sizes=[1024, 1],
            flatten_loops=[True],
            loop_order=(1, 0),
        )
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add3d(self):
        args = (
            torch.randn([100, 500, 10], device=DEVICE),
            torch.randn([100, 500, 10], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[1024, 1, 1], flatten_loop=True
        )
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add3d_xy_grid(self):
        args = (
            torch.randn([100, 500, 10], device=DEVICE),
            torch.randn([100, 500, 10], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[16, 16, 16], pid_type="xyz"
        )
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add2d_xyz_l2_grouping(self):
        args = (
            torch.randn([64, 128], device=DEVICE),
            torch.randn([64, 128], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add,
            args,
            block_sizes=[16, 16],
            l2_groupings=[8],
            pid_type="xyz",
        )
        torch.testing.assert_close(result, args[0] + args[1])

    def test_add3d_reorder(self):
        args = (
            torch.randn([100, 500, 10], device=DEVICE),
            torch.randn([100, 500, 10], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add,
            args,
            block_sizes=[1024, 1, 1],
            flatten_loop=True,
            loop_order=(2, 0, 1),
        )
        torch.testing.assert_close(result, args[0] + args[1])

    @skipIfLowVRAM(
        "Test requires sufficient free VRAM for [512, 512, 512] tensors",
        required_bytes=3 * 1024**3,
    )
    def test_add_tilend0(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[8, 16, 32], loop_order=(0, 1, 2)
        )
        torch.testing.assert_close(result, args[0] + args[1])

    @skipIfLowVRAM(
        "Test requires sufficient free VRAM for [512, 512, 512] tensors",
        required_bytes=3 * 1024**3,
    )
    def test_add_tilend1(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[8, 16, 32], loop_order=(2, 1, 0)
        )
        torch.testing.assert_close(result, args[0] + args[1])

    @skipIfLowVRAM(
        "Test requires sufficient free VRAM for [512, 512, 512] tensors",
        required_bytes=3 * 1024**3,
    )
    def test_add_tilend2(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[1, 32, 32], loop_order=(0, 1, 2)
        )
        torch.testing.assert_close(result, args[0] + args[1])

    @skipIfLowVRAM(
        "Test requires sufficient free VRAM for [512, 512, 512] tensors",
        required_bytes=3 * 1024**3,
    )
    def test_add_tilend3(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add,
            args,
            block_sizes=[1, 32, 1],
            loop_order=(0, 2, 1),
            num_warps=8,
            num_stages=1,
        )
        torch.testing.assert_close(result, args[0] + args[1])

    def test_torch_ops_pointwise(self):
        args = (
            torch.randn([1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.torch_ops_pointwise,
            args,
            block_size=128,
        )
        torch.testing.assert_close(
            result, torch.sigmoid(torch.add(torch.sin(args[0]), torch.cos(args[1])))
        )

    def test_hl_zeros_usage(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_zeros_usage,
            args,
            block_sizes=[32, 32],
        )
        torch.testing.assert_close(result, args[0] * 2)

    def test_hl_full_usage(self):
        args = (torch.randn([512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_full_usage,
            args,
            block_size=128,
        )
        torch.testing.assert_close(result, args[0] * 2 + 1)

    def test_hl_zeros_flat(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_zeros_usage,
            args,
            block_sizes=[128, 1],
            flatten_loops=[True],
        )
        torch.testing.assert_close(result, args[0] * 2)

    def test_torch_empty_no_device(self):
        args = (torch.randn([512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.torch_empty_no_device,
            args,
            block_size=128,
        )
        torch.testing.assert_close(result, args[0])

    def test_torch_zeros_no_device(self):
        args = (torch.randn([512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.torch_zeros_no_device,
            args,
            block_size=128,
        )
        torch.testing.assert_close(result, args[0] * 2)

    def test_torch_full_no_device(self):
        args = (torch.randn([512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.torch_full_no_device,
            args,
            block_size=128,
        )
        torch.testing.assert_close(result, args[0] * 2 + 1)

    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_torch_empty_no_device_injects_device(self):
        args = (torch.randn([512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.torch_empty_no_device,
            args,
            block_size=128,
        )
        self.assertIn("x.device", code)
        torch.testing.assert_close(result, args[0])

    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_torch_empty_with_device_no_duplicate(self):
        args = (torch.randn([512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.torch_empty_with_device,
            args,
            block_size=128,
        )
        torch.testing.assert_close(result, args[0])
        # device= already present in user code, should not inject a second one
        non_comment_lines = [
            line for line in code.splitlines() if not line.strip().startswith("#")
        ]
        self.assertEqual("\n".join(non_comment_lines).count("device="), 1)

    def test_inplace_mul(self):
        args = (torch.randn([512, 512], device=DEVICE), 4)
        eager_result = args[0] * args[1]
        code, result = code_and_output(
            basic_kernels.inplace_mul,
            args,
            block_size=[128, 1],
            flatten_loop=True,
        )
        torch.testing.assert_close(result, eager_result)

    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_final_cast_enforced_for_to_dtype(self):
        x = torch.randn([1024], device=DEVICE, dtype=torch.bfloat16)
        ref = torch.empty_like(x)
        code, result = code_and_output(cast_after_div, (x, ref), block_size=256)
        if _get_backend() != "cute":
            self.assertIn("tl.cast", code)
            self.assertIn("tl.bfloat16", code)
        else:
            self.assertIn("cutlass.BFloat16", code)

    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_sigmoid_scalar_autocast(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[32],
                indexing="block_ptr",
            ),
            static_shapes=True,
        )
        def se_block_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)

            for tile_m in hl.tile(m):
                x_tile = x[tile_m, :]
                sigmoid_result = torch.sigmoid(x_tile @ w[:, :])
                acc = 2.0 * x_tile * sigmoid_result
                out[tile_m, :] = acc.to(x.dtype)

            return out

        m, n = 4096, 128
        dtype = torch.bfloat16

        x = torch.randn(m, n, device=DEVICE, dtype=dtype)
        w = torch.randn(n, n, device=DEVICE, dtype=dtype)

        code, result = code_and_output(se_block_fwd, (x, w))
        if _get_backend() == "cute":
            self.assertIn("cute.gemm", code)
            self.assertIn("cute.nvgpu.warp.MmaF16BF16Op", code)
            self.assertNotIn("dot_serial_result", code)
        x_fp32 = x.to(torch.float32)
        w_fp32 = w.to(torch.float32)
        expected = (2.0 * x_fp32 * torch.sigmoid(x_fp32 @ w_fp32)).to(dtype)

        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-1)

    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_fast_sigmoid(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[32],
                indexing="block_ptr",
            ),
            static_shapes=True,
            fast_math=True,
        )
        def se_block_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)

            for tile_m in hl.tile(m):
                x_tile = x[tile_m, :]
                sigmoid_result = torch.sigmoid(x_tile @ w[:, :])
                acc = 2.0 * x_tile * sigmoid_result
                out[tile_m, :] = acc.to(x.dtype)

            return out

        m, n = 4096, 128
        dtype = torch.bfloat16

        x = torch.randn(m, n, device=DEVICE, dtype=dtype)
        w = torch.randn(n, n, device=DEVICE, dtype=dtype)

        code, result = code_and_output(se_block_fwd, (x, w))
        if _get_backend() == "triton":
            self.assertIn("fast_dividef", code)
            self.assertIn("fast_expf", code)
        elif _get_backend() == "cute":
            self.assertIn("cute.gemm", code)
            self.assertIn("cute.nvgpu.warp.MmaF16BF16Op", code)
            self.assertNotIn("dot_serial_result", code)

        x_fp32 = x.to(torch.float32)
        w_fp32 = w.to(torch.float32)
        expected = (2.0 * x_fp32 * torch.sigmoid(x_fp32 @ w_fp32)).to(dtype)

        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-1)

    @skipIfRefEager("codegen pins")
    @skipIfTileIR("the approximate divide is Triton-only; TileIR keeps the exact form")
    def test_fast_math_division(self):
        """``fast_math`` lowers plain fp32 (and upcast half-precision) divisions to
        the approximate ``fast_dividef`` on the Triton backend; without the
        setting, for fp64, and for the rounding-mode divisions the division stays
        the default ``/``."""

        def div_kernel(x: torch.Tensor, y: torch.Tensor, s: float) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                quot = x[tile_m, :] / (y[tile_m, :] + 2.0)
                out[tile_m, :] = ((quot / 4.0) / s).to(x.dtype)
            return out

        def trunc_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = torch.div(
                    x[tile_m, :], y[tile_m, :], rounding_mode="trunc"
                )
            return out

        def int_div_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] / y[tile_m, :]
            return out

        def kernel(fn, *, fast_math):
            return helion.kernel(
                config=helion.Config(block_sizes=[32]),
                static_shapes=True,
                fast_math=fast_math,
            )(fn)

        tensor_division = re.compile(r"= \w+ / \w+\b")
        for dtype, tol in (
            (torch.float32, 1e-5),
            (torch.float16, 1e-2),
            (torch.bfloat16, 1e-2),
        ):
            x = torch.randn(256, 64, device=DEVICE, dtype=dtype)
            y = torch.rand(256, 64, device=DEVICE, dtype=dtype)
            expected = (((x.float() / (y.float() + 2.0)) / 4.0) / 1.5).to(dtype)
            code, result = code_and_output(
                kernel(div_kernel, fast_math=True), (x, y, 1.5)
            )
            if _get_backend() == "triton":
                # The two tensor-valued divisions (tensor / tensor, tensor / float
                # argument) go through fast_dividef on fp32 operands with no cast
                # back; the division by the constant 4.0 is already a multiply.
                self.assertEqual(code.count("fast_dividef("), 2)
                self.assertNotRegex(code, tensor_division)
                self.assertNotIn("tl.cast(fast_dividef", code)
            torch.testing.assert_close(result, expected, atol=tol, rtol=tol)
            code, result = code_and_output(
                kernel(div_kernel, fast_math=False), (x, y, 1.5)
            )
            self.assertNotIn("fast_dividef", code)
            torch.testing.assert_close(result, expected, atol=tol, rtol=tol)

        # fp64 keeps the exact division under the setting
        x = torch.randn(256, 64, device=DEVICE, dtype=torch.float64)
        y = torch.rand(256, 64, device=DEVICE, dtype=torch.float64)
        code, result = code_and_output(kernel(div_kernel, fast_math=True), (x, y, 1.5))
        self.assertNotIn("fast_dividef", code)
        torch.testing.assert_close(result, ((x / (y + 2.0)) / 4.0) / 1.5)

        # integer true division is promoted to fp32 first and takes the fast form
        xi = torch.randint(-50, 50, (256, 64), device=DEVICE, dtype=torch.int32)
        yi = torch.randint(1, 9, (256, 64), device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(kernel(int_div_kernel, fast_math=True), (xi, yi))
        if _get_backend() == "triton":
            self.assertEqual(code.count("fast_dividef("), 1)
        torch.testing.assert_close(result, xi / yi, atol=1e-5, rtol=1e-5)

        # A rounding-mode division keeps the default quotient (trunc must see
        # the same value with or without the setting; Triton's default divide
        # is itself 2 ulp, so the reference comparison allows one step).  The
        # CuTe backend does not lower rounding-mode divisions at all.
        if _get_backend() != "triton":
            return
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(256, 64, device=DEVICE, dtype=dtype) * 8
            y = (torch.rand(256, 64, device=DEVICE, dtype=dtype) + 0.5).to(dtype)
            code, result = code_and_output(kernel(trunc_kernel, fast_math=True), (x, y))
            self.assertNotIn("fast_dividef", code)
            _, default_result = code_and_output(
                kernel(trunc_kernel, fast_math=False), (x, y)
            )
            torch.testing.assert_close(result, default_result, atol=0, rtol=0)
            torch.testing.assert_close(
                result, torch.div(x, y, rounding_mode="trunc"), atol=1.0, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
