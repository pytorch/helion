from __future__ import annotations

import functools
import os
import sys
from typing import TYPE_CHECKING
import unittest

import torch
import torch.nn.functional as F

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl
from helion.runtime.settings import _get_backend

if TYPE_CHECKING:
    from collections.abc import Callable


# tl.dot_scaled requires SM 10.0+ (B200 / compute capability 10.0)
def requires_sm100(fn: Callable) -> Callable:
    return skipIfNotCUDA()(
        skipIfCudaCapabilityLessThan(
            (10, 0), reason="tl.dot_scaled requires CUDA capability >= 10.0 (B200+)"
        )(fn)
    )


@functools.lru_cache(maxsize=1)
def _supports_fp16_dot_scaled() -> bool:
    if _get_backend() == "cute":
        # The CuTe fallback dequantizes fp16 operands itself.
        return True
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

    x = torch.randn(16, 64, device=DEVICE, dtype=HALF_DTYPE)
    x_scale = torch.full((16, 2), 127, device=DEVICE, dtype=torch.uint8)
    y = torch.randn(64, 16, device=DEVICE, dtype=HALF_DTYPE)
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


def _assert_scaled_dot_codegen(test: TestCase, code: str) -> None:
    if _get_backend() == "cute":
        # The CuTe fallback dequantizes each operand element by the e8m0 scale
        # of its K group before the scalar dot.
        test.assertIn("cute.arch.load(x_scale.iterator", code)
        test.assertIn("cute.arch.load(y_scale.iterator", code)
        test.assertIn("cute.math.exp2(", code)
        return
    test.assertIn("tl.dot_scaled(", code)


def _reference_dot_scaled(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``x[m, k] * 2**(x_scale[m, k // g] - 127)`` times the same for ``y[k, n]``
    with ``y_scale[n, k // g]``, summed over ``k``."""
    group = x.shape[1] // x_scale.shape[1]
    x_factor = torch.exp2(x_scale.float() - 127).repeat_interleave(group, dim=1)
    y_factor = torch.exp2(y_scale.float() - 127).repeat_interleave(group, dim=1)
    return (x.float() * x_factor) @ (y.float() * y_factor.T)


def _pack_e8m0_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pack four scale bytes per MN-major int32 word."""
    packed = F.pad(scales, (0, (-scales.shape[1]) % 4)).contiguous().view(torch.int32)
    aligned_rows = (packed.shape[0] + 3) // 4 * 4
    storage = torch.empty(
        (packed.shape[1], aligned_rows), device=scales.device, dtype=torch.int32
    )
    result = storage.T[: packed.shape[0]]
    result.copy_(packed)
    return result


def _packed_mxfp8_inputs(
    m: int, n: int, k: int, *, scales: tuple[int, int] = (120, 136)
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    torch.manual_seed(0)
    x = (torch.randn(m, k, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
    y = (torch.randn(n, k, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
    x_scale = torch.randint(*scales, (m, k // 32), device=DEVICE, dtype=torch.uint8)
    y_scale = torch.randint(*scales, (n, k // 32), device=DEVICE, dtype=torch.uint8)
    expected = _reference_dot_scaled(x, x_scale, y.T, y_scale)
    return (x, _pack_e8m0_scales(x_scale), y, _pack_e8m0_scales(y_scale)), expected


@helion.kernel(autotune_effort="none", static_shapes=True)
def _packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``x @ y.T`` from E4M3 data and MN-major packed E8M0 scale words."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(m * n)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                x,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < k),
            )
            rhs = hl.load(
                y,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                x_scale, [rows, groups.index // 4], extra_mask=rows[:, None] < m
            )
            rhs_words = hl.load(
                y_scale, [cols, groups.index // 4], extra_mask=cols[:, None] < n
            )
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        hl.store(
            flat_out,
            [rows[:, None] * n + cols[None, :]],
            acc,
            extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
        )
    return out


@helion.kernel(
    autotune_effort="none",
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _split_k_packed_mxfp8_gemm(
    a: torch.Tensor, scale_a: torch.Tensor, weights: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    """``_packed_mxfp8_gemm`` with K split over a ``partitions`` grid axis."""
    m, k = a.shape
    n = weights.size(0)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(16, 256)
    block_groups = hl.register_block_size(2, 32)
    partitions = hl.register_tunable("partitions", PowerOfTwoFragment(1, 16, 1))
    span = (k // 32 + partitions - 1) // partitions
    partial = torch.empty((partitions, m, n), dtype=torch.float32, device=a.device)
    flat_out = out.view(m * n)
    flat_partial = partial.view(partitions * m * n)
    for task_m, task_n, part in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n, partitions]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        end = min((part + 1) * span, k // 32)
        for groups in hl.tile(part * span, end, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                a,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < end * 32),
            )
            rhs = hl.load(
                weights,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < end * 32),
            )
            lhs_words = hl.load(
                scale_a,
                [rows, groups.index // 4],
                extra_mask=(rows[:, None] < m) & (groups.index[None, :] < end),
            )
            rhs_words = hl.load(
                scale_b,
                [cols, groups.index // 4],
                extra_mask=(cols[:, None] < n) & (groups.index[None, :] < end),
            )
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        if partitions == 1:
            hl.store(
                flat_out,
                [rows[:, None] * n + cols[None, :]],
                acc.to(out.dtype),
                extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
            )
        else:
            hl.store(
                flat_partial,
                [(part * m + rows[:, None]) * n + cols[None, :]],
                acc,
                extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
            )
    if partitions > 1:
        return partial.sum(dim=0).to(torch.bfloat16)
    return out


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    config=helion.Config(block_sizes=[128, 128, 4], num_warps=4, num_stages=3),
)
def _gathered_packed_mxfp8_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    y: torch.Tensor,
    y_scale: torch.Tensor,
    perm: torch.Tensor,
) -> torch.Tensor:
    """``(x @ y.T)[perm]`` with the LHS rows gathered (MoE style)."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        slots = task_m * block_m + hl.arange(block_m)
        rows = hl.load(perm, [slots], extra_mask=slots < m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                x,
                [rows, columns],
                extra_mask=(slots[:, None] < m) & (columns[None, :] < k),
            )
            rhs = hl.load(
                y,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                x_scale, [rows, groups.index // 4], extra_mask=slots[:, None] < m
            )
            rhs_words = hl.load(
                y_scale, [cols, groups.index // 4], extra_mask=cols[:, None] < n
            )
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        hl.store(
            out,
            [slots, cols],
            acc,
            extra_mask=(slots[:, None] < m) & (cols[None, :] < n),
        )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _packed_mxfp8_gemm_with_bias_rows(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    y: torch.Tensor,
    y_scale: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """An FP32 ``x @ y.T`` and ``bias`` broadcast over its rows, both flat."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    rows_of_bias = torch.empty([m, n], dtype=torch.float32, device=x.device)
    flat_out = out.view(m * n)
    flat_bias = rows_of_bias.view(m * n)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                x,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < k),
            )
            rhs = hl.load(
                y,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                x_scale, [rows, groups.index // 4], extra_mask=rows[:, None] < m
            )
            rhs_words = hl.load(
                y_scale, [cols, groups.index // 4], extra_mask=cols[:, None] < n
            )
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        mask = (rows[:, None] < m) & (cols[None, :] < n)
        flat_index = rows[:, None] * n + cols[None, :]
        hl.store(flat_out, [flat_index], acc, extra_mask=mask)
        hl.store(
            flat_bias, [flat_index], hl.load(bias, [cols])[None, :], extra_mask=mask
        )
    return out, rows_of_bias


@helion.kernel(autotune_effort="none", static_shapes=True)
def _copy_then_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """Copy ``x`` in one phase, then contract the copy after a barrier."""
    m, k = x.size()
    n, _ = y.size()
    copy = torch.empty_like(x)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(128, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for tile_m, tile_k in hl.tile([m, k]):
        copy[tile_m, tile_k] = x[tile_m, tile_k]
    hl.barrier()
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                copy,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < k),
            )
            rhs = hl.load(
                y,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                x_scale, [rows, groups.index // 4], extra_mask=rows[:, None] < m
            )
            rhs_words = hl.load(
                y_scale, [cols, groups.index // 4], extra_mask=cols[:, None] < n
            )
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        hl.store(
            out, [rows, cols], acc, extra_mask=(rows[:, None] < m) & (cols[None, :] < n)
        )
    return out


@onlyBackends(["triton", "cute"])
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

        x = torch.randn(M, K, device=DEVICE, dtype=HALF_DTYPE)
        x_scale = torch.full(
            (M, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )
        y = torch.randn(K, N, device=DEVICE, dtype=HALF_DTYPE)
        y_scale = torch.full(
            (N, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )

        code, result = code_and_output(scaled_kernel, (x, x_scale, y, y_scale))
        _assert_scaled_dot_codegen(self, code)
        if _get_backend() == "triton":
            self.assertNotIn("mxfp8_lhs_words", code)

    @requires_sm100
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_native_lowering(self):
        """Descriptor operands, UTCCP-ordered scales and a descriptor store."""
        args, expected = _packed_mxfp8_inputs(128, 256, 256)
        code, result = code_and_output(_packed_mxfp8_gemm, args)
        self.assertIn("mxfp8_lhs_words", code)
        self.assertIn(".trans(1, 0, 3, 2, 4).trans(0, 3, 2, 1, 4)", code)
        self.assertIn("_BLOCK_SIZE_0 = tl.constexpr(128)", code)
        self.assertIn("_BLOCK_SIZE_1 = tl.constexpr(256)", code)
        self.assertIn("_BLOCK_SIZE_2 = tl.constexpr(4)", code)
        self.assertIn("warp_specialize=True", code)
        self.assertIn("(x_desc.load(", code)
        self.assertIn("(y_desc.load(", code)
        self.assertIn("out_desc.store(", code)
        self.assertNotIn("tl.make_tensor_descriptor", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_sm100
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_compact_swap(self):
        """A tile of at most 32 rows is issued as the transposed MMA."""
        args, expected = _packed_mxfp8_inputs(8, 256, 256)
        code, result = code_and_output(_packed_mxfp8_gemm, args)
        self.assertIn("_BLOCK_SIZE_0 = tl.constexpr(16)", code)
        self.assertIn("_BLOCK_SIZE_1 = tl.constexpr(128)", code)
        self.assertIn("tl.dot_scaled(tl.permute(", code)
        self.assertNotIn("mxfp8_lhs_words", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_sm100
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_unaligned_output_stride(self):
        """An output row stride that is not 16-byte aligned keeps pointer stores."""
        args, expected = _packed_mxfp8_inputs(128, 129, 256)
        code, result = code_and_output(_packed_mxfp8_gemm, args)
        self.assertIn("mxfp8_lhs_words", code)
        self.assertNotIn("out_desc.store(", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_sm100
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_split_k_alignment(self):
        """Native loads are proved for each split, whatever its tunable is named.

        K = 2560 holds 80 groups: 4 partitions of 20 groups split on whole
        scale words and 4-group tiles, 16 partitions of 5 groups do not, and
        8-group tiles cross the end of a 20-group partition.
        """
        args, expected = _packed_mxfp8_inputs(128, 256, 2560, scales=(124, 130))
        seed = _split_k_packed_mxfp8_gemm.bind(args).config_spec.default_config()
        self.assertEqual(seed["partitions"], 4)
        for overrides, words, operand_descriptors in (
            ({}, True, True),
            ({"partitions": 16}, False, False),
            ({"block_sizes": [128, 256, 8]}, True, False),
        ):
            with self.subTest(**overrides):
                code, result = code_and_output(
                    _split_k_packed_mxfp8_gemm, args, **{**seed.config, **overrides}
                )
                self.assertEqual("mxfp8_lhs_words" in code, words)
                # Operand descriptor loads feed the dot directly.
                self.assertEqual("(a_desc.load(" in code, operand_descriptors)
                self.assertEqual("(weights_desc.load(" in code, operand_descriptors)
                torch.testing.assert_close(
                    result.float(), expected, atol=2e-2, rtol=1e-2
                )

    @requires_sm100
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_dynamic_gathered_rows(self):
        """Gathered rows of a dynamic-shape kernel keep generic scale loads."""
        for m in (128, 256):
            with self.subTest(m=m):
                args, expected = _packed_mxfp8_inputs(m, 256, 256, scales=(124, 130))
                perm = torch.randperm(m, device=DEVICE, dtype=torch.int32)
                code, result = code_and_output(
                    _gathered_packed_mxfp8_gemm, (*args, perm)
                )
                self.assertNotIn("mxfp8_lhs_words", code)
                torch.testing.assert_close(
                    result, expected[perm.long()], atol=2e-2, rtol=1e-2
                )

    @requires_sm100
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_epilogue_stores(self):
        """A broadcast row and an FP32 tile too large to stage keep pointer
        stores, so the warp-specialized seed still fits and is correct."""
        args, expected = _packed_mxfp8_inputs(128, 256, 256)
        bias = torch.randn(256, device=DEVICE)
        code, (result, rows_of_bias) = code_and_output(
            _packed_mxfp8_gemm_with_bias_rows, (*args, bias)
        )
        self.assertIn("warp_specialize=True", code)
        self.assertIn("mxfp8_lhs_words", code)
        self.assertNotIn("_desc.store(", code)
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(rows_of_bias, bias.expand(128, 256))

    @requires_sm100
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_fences_tma_after_barrier(self):
        """Descriptor loads under pointer indexing still get the async-proxy
        fence that orders them after a barrier."""
        args, expected = _packed_mxfp8_inputs(128, 256, 256, scales=(126, 128))
        code, result = code_and_output(
            _copy_then_packed_mxfp8_gemm,
            args,
            block_sizes=[128, 256, 4, 64, 64],
            num_warps=4,
            num_stages=3,
            pid_type="persistent_blocked",
            indexing="pointer",
        )
        self.assertIn("(copy_desc.load(", code)
        self.assertIn("fence.proxy.async", code)
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=1e-2)

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
        _assert_scaled_dot_codegen(self, code)
        if _get_backend() != "cute":
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

        x = torch.randn(M, K, device=DEVICE, dtype=HALF_DTYPE)
        x_scale = torch.full(
            (M, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )
        y = torch.randn(K, N, device=DEVICE, dtype=HALF_DTYPE)
        y_scale = torch.full(
            (N, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )

        code, result = code_and_output(
            scaled_out_dtype_kernel, (x, x_scale, y, y_scale)
        )
        _assert_scaled_dot_codegen(self, code)
        if _get_backend() != "cute":
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

        x = torch.randn(M, K, device=DEVICE, dtype=HALF_DTYPE)
        x_scale = torch.full(
            (M, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )
        y = torch.randn(K, N, device=DEVICE, dtype=HALF_DTYPE)
        y_scale = torch.full(
            (N, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )

        code, result = code_and_output(scaled_no_acc_kernel, (x, x_scale, y, y_scale))
        _assert_scaled_dot_codegen(self, code)

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

        x = torch.randn(M, K, device=DEVICE, dtype=HALF_DTYPE)
        x_scale = torch.full(
            (M, K // SCALE_FACTOR), 127, device=DEVICE, dtype=torch.uint8
        )
        y = torch.randn(K, N, device=DEVICE, dtype=HALF_DTYPE)
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

    @requires_sm100
    def test_numerical_correctness_random_scales(self):
        """Distinct per-group scales (including bytes above 127) must be read
        at each element's own K group: x_scale[m, k // 32] and y_scale[n, k // 32]."""
        M, N, K = 64, 64, 128
        SCALE_FACTOR = 32

        @helion.kernel(config=helion.Config(block_sizes=[32, 32]))
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

        torch.manual_seed(0)
        x = (torch.randn(M, K, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        y = (torch.randn(K, N, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        x_scale = torch.randint(
            120, 136, (M, K // SCALE_FACTOR), device=DEVICE, dtype=torch.uint8
        )
        y_scale = torch.randint(
            120, 136, (N, K // SCALE_FACTOR), device=DEVICE, dtype=torch.uint8
        )

        result = scaled_kernel(x, x_scale, y, y_scale)
        expected = _reference_dot_scaled(x, x_scale, y, y_scale)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-3)

    @requires_sm100
    @skipIfRefEager("the reference dot_scaled does not unpack mixed fp4 formats")
    def test_mixed_formats(self):
        """fp8 x packed fp4 (K counted per element), and bf16 with no scale."""
        if _get_backend() == "cute":
            self.skipTest("mixed formats: triton backend")
        M, N, K = 128, 128, 256

        @helion.kernel(config=helion.Config(block_sizes=[64, 64]))
        def fp8_fp4(
            x: torch.Tensor,
            x_scale: torch.Tensor,
            y: torch.Tensor,
            y_scale: torch.Tensor,
        ) -> torch.Tensor:
            out = torch.empty(
                [x.size(0), y.size(1)], dtype=torch.float32, device=x.device
            )
            for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
                out[tile_m, tile_n] = hl.dot_scaled(
                    x[tile_m, :], x_scale[tile_m, :], "e4m3",
                    y[:, tile_n], y_scale[tile_n, :], "e2m1",
                )  # fmt: skip
            return out

        @helion.kernel(config=helion.Config(block_sizes=[64, 64]))
        def bf16_fp4(
            x: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty(
                [x.size(0), y.size(1)], dtype=torch.float32, device=x.device
            )
            for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
                out[tile_m, tile_n] = hl.dot_scaled(
                    x[tile_m, :], None, "bf16", y[:, tile_n], y_scale[tile_n, :], "e2m1"
                )
            return out

        torch.manual_seed(0)
        y = torch.randint(0, 256, (K // 2, N), device=DEVICE, dtype=torch.uint8)
        y_scale = torch.full((N, K // 32), 127, device=DEVICE, dtype=torch.uint8)
        values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=DEVICE)
        codes = torch.stack([y & 15, y >> 4], 1).flatten(0, 1).long()
        w = torch.where(codes >= 8, -values[codes % 8], values[codes % 8])
        x8 = (torch.randn(M, K, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        x_scale = torch.full((M, K // 32), 127, device=DEVICE, dtype=torch.uint8)
        result = fp8_fp4(x8, x_scale, y, y_scale)
        torch.testing.assert_close(result, x8.float() @ w, atol=1e-3, rtol=1e-3)
        x16 = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
        result = bf16_fp4(x16, y, y_scale)
        torch.testing.assert_close(result, x16.float() @ w, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
