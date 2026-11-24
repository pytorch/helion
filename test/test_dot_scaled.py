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
from helion._testing import is_cuda
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfFn
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing.packed_mxfp8 import dynamic_packed_mxfp8_gemm
from helion._testing.packed_mxfp8 import pack_e8m0_scales
from helion._testing.packed_mxfp8 import packed_mxfp8_gemm
from helion._testing.packed_mxfp8 import packed_mxfp8_inputs
from helion._testing.packed_mxfp8 import two_output_packed_mxfp8_gemm
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


# The native packed MXFP8 lowering and its seed take sm100/sm103 only.
# TileIR declines it (``mxfp8.native_packed_mxfp8_target``).
def requires_packed_mxfp8_target(fn: Callable) -> Callable:
    return requires_sm100(
        skipIfTileIR("the native packed MXFP8 lowering is the Triton backend's")(
            skipIfFn(
                lambda: not is_cuda() or torch.cuda.get_device_capability()[0] != 10,
                reason="the native packed MXFP8 lowering targets CUDA capability 10.x",
            )(fn)
        )
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


def _packed_mxfp8_inputs(
    m: int, n: int, k: int, *, scales: tuple[int, int] = (120, 136), pad: int = 0
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    args, (x_scale, y_scale) = packed_mxfp8_inputs(
        m, n, k, device=DEVICE, scales=scales, pad=pad
    )
    return args, _reference_dot_scaled(args[0], x_scale, args[2].T, y_scale)


@helion.kernel(autotune_effort="none", static_shapes=True)
def _wide_k_packed_mxfp8_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    y: torch.Tensor,
    y_scale: torch.Tensor,
    min_groups: hl.constexpr,
) -> torch.Tensor:
    """``packed_mxfp8_gemm`` with a K tile of at least ``min_groups`` groups."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(m * n)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(min_groups, 32)
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
    """``packed_mxfp8_gemm`` with K split over a ``partitions`` grid axis."""
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
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(m * n)
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
            flat_out,
            [rows[:, None] * n + cols[None, :]],
            acc.to(out.dtype),
            extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
        )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _partial_k_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``x[:, :200] @ y[:, :200].T`` of 256-column operands: the K loop ends
    inside the tensor, so its tile mask clips columns the tensor holds."""
    m, _ = x.size()
    n, _ = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(128, 128)
    block_n = hl.register_block_size(256, 256)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for tile_k in hl.tile(200, block_size=256):
            columns = tile_k.begin + hl.arange(tile_k.block_size)
            groups = hl.arange(8)
            lhs = hl.load(x, [rows, columns], extra_mask=rows[:, None] < m)
            rhs = hl.load(y, [cols, columns], extra_mask=cols[:, None] < n)
            lhs_words = hl.load(
                x_scale, [rows, groups // 4], extra_mask=rows[:, None] < m
            )
            rhs_words = hl.load(
                y_scale, [cols, groups // 4], extra_mask=cols[:, None] < n
            )
            shift = ((groups % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        hl.store(
            out, [rows, cols], acc, extra_mask=(rows[:, None] < m) & (cols[None, :] < n)
        )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _sentinel_gathered_packed_mxfp8_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    y: torch.Tensor,
    y_scale: torch.Tensor,
    rows_of: torch.Tensor,
) -> torch.Tensor:
    """``(x @ y.T)[rows_of]``, where a ``-1`` row (an empty slot) is masked."""
    _, k = x.size()
    n, _ = y.size()
    slots_total = rows_of.size(0)
    out = torch.empty([slots_total, n], dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(slots_total + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        slots = task_m * block_m + hl.arange(block_m)
        rows = hl.load(rows_of, [slots], extra_mask=slots < slots_total)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                x,
                [rows, columns],
                extra_mask=(rows[:, None] >= 0) & (columns[None, :] < k),
            )
            rhs = hl.load(
                y,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                x_scale, [rows, groups.index // 4], extra_mask=rows[:, None] >= 0
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
            extra_mask=(slots[:, None] < slots_total) & (cols[None, :] < n),
        )
    return out


@helion.kernel(static_shapes=True)
def _two_packed_mxfp8_gemms(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x @ y.T`` from two dots of one loop."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    twice = torch.empty([m, n], dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        acc_twice = hl.zeros([block_m, block_n], dtype=torch.float32)
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
            acc_twice = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc_twice
            )
        mask = (rows[:, None] < m) & (cols[None, :] < n)
        hl.store(out, [rows, cols], acc, extra_mask=mask)
        hl.store(twice, [rows, cols], acc_twice, extra_mask=mask)
    return out, twice


@helion.kernel(autotune_effort="none", static_shapes=True)
def _k_major_rhs_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y_t: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``x @ y_t`` with the RHS loaded as ``[K, N]`` rather than as the
    transpose of an ``[N, K]`` load."""
    m, k = x.size()
    _, n = y_t.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
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
                y_t,
                [columns, cols],
                extra_mask=(columns[:, None] < k) & (cols[None, :] < n),
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
            acc = hl.dot_scaled(lhs, lhs_scale, "e4m3", rhs, rhs_scale, "e4m3", acc=acc)
        hl.store(
            out, [rows, cols], acc, extra_mask=(rows[:, None] < m) & (cols[None, :] < n)
        )
    return out


@helion.kernel(static_shapes=True)
def _tiled_packed_mxfp8_gemm_into(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    y: torch.Tensor,
    y_scale: torch.Tensor,
    rows_of_out: torch.Tensor,
) -> torch.Tensor:
    """``x @ y.T`` into the first rows of an output of ``rows_of_out.size(0)``
    rows filled with 7, from a tile loop whose tile mask stops at ``x``'s
    rows."""
    m, k = x.size()
    n, _ = y.size()
    rows_out, cols_out = rows_of_out.size(0), n
    out = torch.full([rows_out, cols_out], 7.0, dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(rows_out * cols_out)
    block_groups = hl.register_block_size(4)
    for _ in hl.grid(1):
        for tile_m, tile_n in hl.tile([m, n], block_size=[128, 256]):
            rows = tile_m.begin + hl.arange(tile_m.block_size)
            cols = tile_n.begin + hl.arange(tile_n.block_size)
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
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
                [rows[:, None] * cols_out + cols[None, :]],
                acc.to(out.dtype),
                extra_mask=(rows[:, None] < rows_out) & (cols[None, :] < cols_out),
            )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _unmasked_store_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``packed_mxfp8_gemm`` with an unmasked flattened store."""
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
        hl.store(flat_out, [rows[:, None] * n + cols[None, :]], acc.to(out.dtype))
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _full_k_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``x @ y.T`` from one K tile over all of K (padded when K is not a
    power of two)."""
    m, k = x.size()
    n, _ = y.size()
    k = hl.specialize(k)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(128, 128)
    block_n = hl.register_block_size(128, 128)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        columns = hl.arange(k)
        groups = hl.arange(k // 32)
        lhs = hl.load(x, [rows, columns], extra_mask=rows[:, None] < m)
        rhs = hl.load(y, [cols, columns], extra_mask=cols[:, None] < n)
        lhs_words = hl.load(x_scale, [rows, groups // 4], extra_mask=rows[:, None] < m)
        rhs_words = hl.load(y_scale, [cols, groups // 4], extra_mask=cols[:, None] < n)
        shift = (groups % 4) * 8
        lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
        rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
        acc = hl.dot_scaled(lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3")
        hl.store(
            out, [rows, cols], acc, extra_mask=(rows[:, None] < m) & (cols[None, :] < n)
        )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _shifted_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``packed_mxfp8_gemm`` reading operand column ``c + 1`` for column
    ``c``: the operand tiles start one byte past a 16-byte boundary."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
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
            columns = groups.begin * 32 + hl.arange(block_groups * 32) + 1
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
            out, [rows, cols], acc, extra_mask=(rows[:, None] < m) & (cols[None, :] < n)
        )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _scale_writing_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``x @ y.T`` that writes each ``x_scale`` word back after loading it."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
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
            hl.store(
                x_scale,
                [rows, groups.index // 4],
                lhs_words,
                extra_mask=rows[:, None] < m,
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


@helion.kernel(autotune_effort="none", static_shapes=True)
def _packed_mxfp8_gemm_into(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    y: torch.Tensor,
    y_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """``packed_mxfp8_gemm`` writing the caller's ``out``."""
    m, k = x.size()
    n, _ = y.size()
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
            acc.to(out.dtype),
            extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
        )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _tile_rows_packed_mxfp8_gemm(
    a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    """``a @ b.T`` with tile row indices (``sa[row, ...]``) over all of K."""
    m, k = a.shape
    n = b.size(0)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for row, col in hl.tile([m, n]):
        group = hl.arange(k // 32)
        left_words = hl.load(sa, [row, group // 4])
        left_scale = ((left_words >> ((group % 4) * 8).unsqueeze(0)) & 255).to(
            torch.uint8
        )
        right_words = hl.load(sb, [col, group // 4])
        right_scale = ((right_words >> ((group % 4) * 8).unsqueeze(0)) & 255).to(
            torch.uint8
        )
        out[row, col] = hl.dot_scaled(
            a[row, :], left_scale, "e4m3", b[col, :].T, right_scale, "e4m3"
        )
    return out


@helion.kernel(
    autotune_effort="none",
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _replicated_packed_mxfp8_gemm(
    a: torch.Tensor, scale_a: torch.Tensor, weights: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    """``a @ weights.T`` computed by every one of ``replicas`` grid slices,
    which all write the same output: a power-of-two grid tunable that does
    not split K."""
    m, k = a.shape
    n = weights.size(0)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(16, 256)
    block_groups = hl.register_block_size(2, 32)
    replicas = hl.register_tunable("replicas", PowerOfTwoFragment(1, 16, 1))
    flat_out = out.view(m * n)
    for task_m, task_n, _ in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n, replicas]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                a,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < k),
            )
            rhs = hl.load(
                weights,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                scale_a, [rows, groups.index // 4], extra_mask=rows[:, None] < m
            )
            rhs_words = hl.load(
                scale_b, [cols, groups.index // 4], extra_mask=cols[:, None] < n
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
            acc.to(out.dtype),
            extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
        )
    return out


@helion.kernel(static_shapes=True)
def _descending_packed_mxfp8_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    y: torch.Tensor,
    y_scale: torch.Tensor,
    rows_of_out: torch.Tensor,
) -> torch.Tensor:
    """``x @ y[:128].T`` stored from a descending root grid of one slot
    (``slot == -1``): its columns -128..-1 of rows 128.. lie, in the
    flattened output, in the row before."""
    m, k = x.size()
    n, _ = y.size()
    rows_out, cols_out = rows_of_out.size(0), 128
    out = torch.full([rows_out, cols_out], 7.0, dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(rows_out * cols_out)
    block_groups = hl.register_block_size(4)
    for slot in hl.grid(-1, -2, -1):
        rows = hl.arange(128)
        cols = (slot + 1) * 128 + hl.arange(128)
        out_rows = rows + 128
        out_cols = slot * 128 + hl.arange(128)
        acc = hl.zeros([128, 128], dtype=torch.float32)
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
            [out_rows[:, None] * cols_out + out_cols[None, :]],
            acc.to(out.dtype),
            extra_mask=(out_rows[:, None] < rows_out) & (out_cols[None, :] < cols_out),
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

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_native_lowering(self):
        """Descriptor operands, UTCCP-ordered scales and a descriptor store."""
        args, expected = _packed_mxfp8_inputs(128, 256, 256)
        code, result = code_and_output(packed_mxfp8_gemm, args)
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

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_compact_swap(self):
        """A tile of at most 32 rows is issued as the transposed MMA."""
        args, expected = _packed_mxfp8_inputs(8, 256, 256)
        code, result = code_and_output(packed_mxfp8_gemm, args)
        self.assertIn("_BLOCK_SIZE_0 = tl.constexpr(16)", code)
        self.assertIn("_BLOCK_SIZE_1 = tl.constexpr(128)", code)
        self.assertIn("tl.dot_scaled(tl.permute(", code)
        self.assertNotIn("mxfp8_lhs_words", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_unaligned_output_stride(self):
        """An output row stride that is not 16-byte aligned keeps pointer stores."""
        args, expected = _packed_mxfp8_inputs(128, 129, 256)
        code, result = code_and_output(packed_mxfp8_gemm, args)
        self.assertIn("mxfp8_lhs_words", code)
        self.assertNotIn("out_desc.store(", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_split_k_alignment(self):
        """Native loads are proved for each split, whatever its tunable is named.

        K = 2560 holds 80 groups: 4 partitions of 20 groups split on whole
        scale words and 4-group tiles, 16 partitions of 5 groups do not, and
        8-group tiles cross the end of a 20-group partition.
        """
        args, expected = _packed_mxfp8_inputs(128, 256, 2560, scales=(124, 130))
        spec = _split_k_packed_mxfp8_gemm.bind(args).config_spec
        # The promoted default keeps the tunable's declared default; the
        # split that fills the GPU seeds the search.
        self.assertEqual(spec.default_config()["partitions"], 1)
        seeds = [
            seed
            for seed in spec.compiler_seed_configs
            if seed.config.get("partitions") == 4
        ]
        self.assertEqual(len(seeds), 1)
        seed = seeds[0]
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

    @requires_packed_mxfp8_target
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

    @requires_packed_mxfp8_target
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

    @requires_packed_mxfp8_target
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
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)
        # The barrier drains TMA stores, which a warp-specialized loop's
        # cannot be: a config warp-specializing the GEMM's loop keeps pointer
        # stores for the flattened output, and still generates.
        code = _copy_then_packed_mxfp8_gemm.bind(args).to_triton_code(
            helion.Config(
                block_sizes=[128, 256, 4, 64, 64],
                num_warps=4,
                num_stages=3,
                pid_type="persistent_blocked",
                indexing="pointer",
                range_warp_specializes=[None, True, None],
            )
        )
        self.assertNotIn("_desc.store(", code)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_dynamic_shapes(self):
        """An operand dimension the kernel does not specialize (``y``'s K)
        still lowers through a descriptor, so the warp-specialized seed reads
        every operand natively."""
        args, expected = _packed_mxfp8_inputs(256, 512, 1280)
        code, result = code_and_output(dynamic_packed_mxfp8_gemm, args)
        self.assertIn("warp_specialize=True", code)
        self.assertIn("(x_desc.load(", code)
        self.assertIn("(y_desc.load(", code)
        self.assertIn("out_desc.store(", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_seed_fits_shared_memory(self):
        """The seed sizes its operand ring for the K tile it gets: an 8-group
        tile (99 KiB a stage at 128x256) takes 2 stages, and a 16-group one,
        for which 2 stages do not fit, keeps the generic default."""
        args, expected = _packed_mxfp8_inputs(256, 512, 2048)
        for min_groups, seeded in ((8, True), (16, False)):
            with self.subTest(min_groups=min_groups):
                spec = _wide_k_packed_mxfp8_gemm.bind((*args, min_groups)).config_spec
                self.assertEqual(
                    "triton_b200_packed_mxfp8" in spec.autotuner_heuristics, seeded
                )
                if seeded:
                    config = spec.default_config()
                    self.assertEqual(config["block_sizes"], [128, 256, 8])
                    self.assertEqual(config["num_stages"], 2)
                code, result = code_and_output(
                    _wide_k_packed_mxfp8_gemm, (*args, min_groups)
                )
                self.assertEqual("mxfp8_lhs_words" in code, seeded)
                torch.testing.assert_close(
                    result.float(), expected, atol=2e-2, rtol=1e-2
                )

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_tma_store_fits_shared_memory(self):
        """The output's TMA store stages its tile beside the operand ring, so
        it is taken only while both fit an sm100 block's shared memory: a
        config whose ring leaves no room keeps its pointer store and runs."""
        args, expected = _packed_mxfp8_inputs(256, 512, 1280)
        bound = dynamic_packed_mxfp8_gemm.bind(args)
        seed = bound.config_spec.default_config()
        self.assertEqual(seed["num_stages"], 3)
        self.assertIn("out_desc.store(", bound.to_triton_code(seed))
        # Four stages of a 128x256 tile with 4-group K tiles (49.5 KiB each)
        # and the 64 KiB BF16 staging tile exceed 227 KiB.
        four_stages = helion.Config(
            **{
                **seed.config,
                "num_stages": 4,
                "range_num_stages": [
                    4 if stages else stages
                    for stages in seed.config["range_num_stages"]
                ],
            }
        )
        code, result = code_and_output(
            dynamic_packed_mxfp8_gemm, args, **four_stages.config
        )
        self.assertIn("warp_specialize=True", code)
        self.assertNotIn("out_desc.store(", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_two_outputs_keep_pointer_stores(self):
        """A second flattened output's store needs shared memory beside the
        first's TMA staging tile and the seed's operand ring, which leave no
        room for it: a kernel writing two outputs keeps pointer stores for
        both, and runs."""
        # Scales near one keep the FP16 output finite.
        args, expected = _packed_mxfp8_inputs(256, 512, 1280, scales=(124, 130))
        code, (out, out2) = code_and_output(two_output_packed_mxfp8_gemm, args)
        self.assertIn("warp_specialize=True", code)
        self.assertIn("(x_desc.load(", code)
        self.assertNotIn("_desc.store(", code)
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(out2.float(), 2 * expected, atol=4e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_warp_specialize_needs_whole_dot(self):
        """An operand without a descriptor (a base not 16-byte aligned) gets
        its own bound kernel, which keeps the seed off warp specialization;
        a warp-specialized config keeps the generic lowering rather than mix
        it with descriptor loads."""
        (x, x_scale, y, y_scale), expected = _packed_mxfp8_inputs(128, 256, 256)
        aligned = packed_mxfp8_gemm.bind((x, x_scale, y, y_scale))
        self.assertTrue(
            aligned.config_spec.default_config()["range_warp_specializes"][0]
        )
        storage = torch.empty(y.numel() + 1, device=DEVICE, dtype=y.dtype)
        unaligned_y = storage[1:].view_as(y)
        unaligned_y.copy_(y)
        args = (x, x_scale, unaligned_y, y_scale)
        bound = packed_mxfp8_gemm.bind(args)
        self.assertIsNot(bound, aligned)
        seed = bound.config_spec.default_config()
        self.assertEqual(seed["range_warp_specializes"][0], False)
        code, result = code_and_output(packed_mxfp8_gemm, args)
        self.assertIn("(x_desc.load(", code)
        self.assertNotIn("(y_desc.load(", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)
        code = bound.to_triton_code(
            helion.Config(**{**seed.config, "range_warp_specializes": [True, False]})
        )
        self.assertIn("warp_specialize=True", code)
        self.assertNotIn("_desc.load(", code)
        self.assertNotIn("mxfp8_lhs_words", code)
        self.assertNotIn("_desc.store(", code)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_tile_masks(self):
        """Lanes a tile mask clips read as zero, as in the generic lowering:
        an operand whose K loop ends inside the tensor keeps its generic load,
        and scale bytes of groups past the last K tile (here NaN padding)
        are zeroed."""
        args, _ = _packed_mxfp8_inputs(128, 256, 256, scales=(124, 130))
        x, x_scale, y, y_scale = args
        # Each column's factor from its group's scale byte, first 200 columns.
        x_factor, y_factor = (
            torch.exp2(
                words.contiguous().view(torch.uint8).float() - 127
            ).repeat_interleave(32, dim=1)[:, :200]
            for words in (x_scale, y_scale)
        )
        expected = (x[:, :200].float() * x_factor) @ (y[:, :200].float() * y_factor).T
        code, result = code_and_output(_partial_k_packed_mxfp8_gemm, args)
        self.assertIn("mxfp8_lhs_words", code)
        self.assertNotIn("(x_desc.load(", code)
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=1e-2)

        # 41 groups: the last 4-group tile holds three padding groups.
        args, expected = _packed_mxfp8_inputs(128, 256, 1312, pad=255)
        code, result = code_and_output(packed_mxfp8_gemm, args)
        self.assertIn("mxfp8_lhs_words", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_masked_gathered_rows(self):
        """Gathered rows with ``-1`` slots: the seed does not warp-specialize a
        dot whose gathered operand keeps pointer loads, and the pointer word
        load reads no row outside the scale tensor."""
        args, expected = _packed_mxfp8_inputs(128, 256, 256, scales=(124, 130))
        rows_of = torch.randperm(128, device=DEVICE, dtype=torch.int32)
        rows_of[::5] = -1
        bound = _sentinel_gathered_packed_mxfp8_gemm.bind((*args, rows_of))
        self.assertFalse(
            bound.config_spec.default_config()["range_warp_specializes"][0]
        )
        code, result = code_and_output(
            _sentinel_gathered_packed_mxfp8_gemm, (*args, rows_of)
        )
        (words,) = [
            line
            for line in code.splitlines()
            if "mxfp8_lhs_words" in line and "= tl.load(" in line
        ]
        self.assertIn("[None, :] >= 0)", words)
        valid = (rows_of >= 0)[:, None]
        torch.testing.assert_close(
            result,
            torch.where(valid, expected[rows_of.clamp(min=0).long()], 0.0),
            atol=2e-2,
            rtol=1e-2,
        )

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_needs_one_dot(self):
        """Two packed dots in a kernel keep the generic lowering: the bound
        kernel keys descriptor alignment only for a kernel's single packed
        dot, so per-op ``indexing`` must decide every descriptor."""
        args, expected = _packed_mxfp8_inputs(128, 256, 256, scales=(124, 130))
        code, (result, twice) = code_and_output(
            _two_packed_mxfp8_gemms,
            args,
            block_sizes=[128, 128, 4],
            indexing="tensor_descriptor",
        )
        self.assertNotIn("mxfp8_lhs_words", code)
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(twice, expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_needs_transposed_rhs(self):
        """An RHS loaded as ``[K, N]`` is not the K-major operand the native
        lowering and its seed take."""
        (x, x_scale, y, y_scale), expected = _packed_mxfp8_inputs(
            128, 256, 256, scales=(124, 130)
        )
        args = (x, x_scale, y.T.contiguous(), y_scale)
        bound = _k_major_rhs_packed_mxfp8_gemm.bind(args)
        self.assertNotIn(
            "triton_b200_packed_mxfp8", bound.config_spec.autotuner_heuristics
        )
        code, result = code_and_output(_k_major_rhs_packed_mxfp8_gemm, args)
        self.assertNotIn("mxfp8_lhs_words", code)
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_descriptor_store_lanes(self):
        """A flattened store keeps pointer stores unless the block store
        writes exactly its lanes: a store without a bound per dimension rules
        the block store out.  A tile loop whose tile mask stops inside
        ``out`` (rows 100 to 127 stay untouched) cannot take it either: its
        stores may overlap across iterations, which pins the loop out of
        warp specialization (``intra_loop_raw_reread_block_ids``)."""
        args, expected = _packed_mxfp8_inputs(100, 256, 256, scales=(124, 130))
        rows_of_out = torch.empty(128, device=DEVICE)
        # Ask to warp-specialize the tile loop (the second of the kernel's
        # loops); its pinned slot ignores the request.
        config = {
            "block_sizes": [4],
            "range_warp_specializes": [None, True, None, None],
        }
        bound = _tiled_packed_mxfp8_gemm_into.bind((*args, rows_of_out))
        self.assertTrue(bound.config_spec.range_warp_specialize[1].pinned)
        code, result = code_and_output(
            _tiled_packed_mxfp8_gemm_into, (*args, rows_of_out), **config
        )
        self.assertNotIn("warp_specialize=True", code)
        self.assertIn("(x_desc.load(", code)
        self.assertNotIn("_desc.store(", code)
        torch.testing.assert_close(result[:100].float(), expected, atol=2e-2, rtol=1e-2)
        self.assertTrue(bool((result[100:] == 7).all()))

        args, expected = _packed_mxfp8_inputs(128, 256, 256, scales=(124, 130))
        code, result = code_and_output(_unmasked_store_packed_mxfp8_gemm, args)
        self.assertIn("warp_specialize=True", code)
        self.assertNotIn("_desc.store(", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_descending_store_columns(self):
        """Columns indexed from a descending grid may go below zero, where the
        flattened store writes the row before and a block store would drop
        the lanes: the output keeps pointer stores."""
        args, expected = _packed_mxfp8_inputs(128, 256, 256, scales=(124, 130))
        # Warp-specialize the persistent loop over the grid (the first config
        # entry); a device loop storing the output would be pinned out of
        # warp specialization (``intra_loop_raw_reread_block_ids``).
        code, result = code_and_output(
            _descending_packed_mxfp8_gemm,
            (*args, torch.empty(256, device=DEVICE)),
            block_sizes=[4],
            pid_type="persistent_blocked",
            range_warp_specializes=[True, None],
        )
        self.assertIn("warp_specialize=True", code)
        self.assertIn("(x_desc.load(", code)
        self.assertNotIn("_desc.store(", code)
        # slot -1 stores rows 127..254.
        self.assertTrue(bool((result[:127] == 7).all()))
        torch.testing.assert_close(
            result[127:255].float(), expected[:, :128], atol=2e-2, rtol=1e-2
        )
        self.assertTrue(bool((result[255] == 7).all()))

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_declined_shapes(self):
        """Shapes the native lowering does not take keep the generic one: a
        K tile padded to a power of two, and an empty operand."""
        for k, native in ((256, True), (384, False)):
            with self.subTest(k=k):
                args, expected = _packed_mxfp8_inputs(128, 128, k, scales=(124, 130))
                code, result = code_and_output(_full_k_packed_mxfp8_gemm, args)
                self.assertEqual("mxfp8_lhs_words" in code, native)
                torch.testing.assert_close(result, expected, atol=2e-2, rtol=1e-2)

        args, _ = _packed_mxfp8_inputs(128, 0, 256)
        self.assertEqual(tuple(packed_mxfp8_gemm(*args).shape), (128, 0))

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_unaligned_offsets(self):
        """A descriptor block starting off a 16-byte boundary is not taken."""
        (x, x_scale, y, y_scale), _ = _packed_mxfp8_inputs(
            128, 256, 256, scales=(124, 130)
        )
        code, result = code_and_output(
            _shifted_packed_mxfp8_gemm, (x, x_scale, y, y_scale)
        )
        self.assertIn("mxfp8_lhs_words", code)
        self.assertNotIn("(x_desc.load(", code)
        self.assertNotIn("(y_desc.load(", code)

        def shifted(data: torch.Tensor) -> torch.Tensor:
            return F.pad(data.float()[:, 1:], (0, 1)).to(data.dtype)

        x_bytes, y_bytes = (
            words.contiguous().view(torch.uint8)[:, :8] for words in (x_scale, y_scale)
        )
        expected = _reference_dot_scaled(shifted(x), x_bytes, shifted(y).T, y_bytes)
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_phase_writes(self):
        """The native loads stand at the dot, not where the loads they replace
        do: a dot whose phase writes one of its tensors keeps the generic
        loads."""
        args, expected = _packed_mxfp8_inputs(128, 256, 256, scales=(124, 130))
        code, result = code_and_output(
            _scale_writing_packed_mxfp8_gemm, args, block_sizes=[128, 256, 4]
        )
        self.assertNotIn("mxfp8_lhs_words", code)
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_broadcast_scale_rows(self):
        """A scale tensor of one row, which generic indexing broadcasts to
        every row of the tile, keeps the generic loads."""
        (x, _, y, y_scale), _ = _packed_mxfp8_inputs(128, 256, 256, scales=(124, 130))
        torch.manual_seed(1)
        one_row = torch.randint(124, 130, (1, 8), device=DEVICE, dtype=torch.uint8)
        y_bytes = y_scale.contiguous().view(torch.uint8)[:, :8]
        expected = _reference_dot_scaled(x, one_row.expand(128, 8), y.T, y_bytes)
        code, result = code_and_output(
            packed_mxfp8_gemm, (x, pack_e8m0_scales(one_row), y, y_scale)
        )
        self.assertNotIn("mxfp8_lhs_words", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_caller_output(self):
        """A caller's flattened output keeps pointer stores: the bound kernel
        keys no alignment for it, and a later call may pass a base off 16
        bytes."""
        args, expected = _packed_mxfp8_inputs(128, 256, 256, scales=(124, 130))
        out = torch.empty(128, 256, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(_packed_mxfp8_gemm_into, (*args, out))
        self.assertIn("warp_specialize=True", code)
        self.assertNotIn("_desc.store(", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)
        storage = torch.empty(out.numel() + 1, device=DEVICE, dtype=out.dtype)
        result = _packed_mxfp8_gemm_into(*args, storage[1:].view(128, 256))
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_int64_indexing(self):
        """Descriptor offsets are cast to int32, so a kernel indexing in
        int64 loads no operand or scale word through a descriptor."""
        kernel = helion.kernel(
            packed_mxfp8_gemm.fn,
            autotune_effort="none",
            static_shapes=True,
            index_dtype=torch.int64,
        )
        args, expected = _packed_mxfp8_inputs(128, 256, 256, scales=(124, 130))
        code, result = code_and_output(kernel, args)
        self.assertNotIn("_desc.load(", code)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_seed_needs_split_k(self):
        """The seed sizes a power-of-two grid tunable only when the K loop's
        bounds use its axis (a split-K partition), not one that replicates."""
        args, expected = _packed_mxfp8_inputs(128, 256, 2560, scales=(124, 130))
        spec = _replicated_packed_mxfp8_gemm.bind(args).config_spec
        self.assertIn("triton_b200_packed_mxfp8", spec.autotuner_heuristics)
        self.assertEqual(
            [seed.config.get("replicas", 1) for seed in spec.compiler_seed_configs],
            [1] * len(spec.compiler_seed_configs),
        )
        _, result = code_and_output(_replicated_packed_mxfp8_gemm, args)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_seed_needs_static_sizes(self):
        """The seed is sized for M, N and K, so a kernel leaving them dynamic
        (no ``hl.specialize``) gets no seed."""
        kernel = helion.kernel(
            packed_mxfp8_gemm.fn, autotune_effort="none", static_shapes=False
        )
        args, expected = _packed_mxfp8_inputs(128, 256, 256, scales=(124, 130))
        bound = kernel.bind(args)
        self.assertNotIn(
            "triton_b200_packed_mxfp8", bound.config_spec.autotuner_heuristics
        )
        _, result = code_and_output(kernel, args)
        torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=1e-2)

    @requires_packed_mxfp8_target
    @onlyBackends(["triton"])
    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_packed_mxfp8_tile_rows(self):
        """Scale words indexed by a tile (``sa[row, ...]``) keep the generic
        loads: the pointer form needs a row vector."""
        (a, sa, b, sb), expected = _packed_mxfp8_inputs(
            256, 256, 512, scales=(124, 130)
        )
        code, result = code_and_output(
            _tile_rows_packed_mxfp8_gemm, (a, b, sa, sb), block_sizes=[128, 128]
        )
        self.assertNotIn("mxfp8_lhs_words", code)
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
