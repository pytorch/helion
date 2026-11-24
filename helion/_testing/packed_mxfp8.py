"""Packed MXFP8 GEMM kernels and fixtures shared by the native lowering's
tests (``triton/packed_mxfp8.py``): E4M3 operands with E8M0 scales packed
four per int32 word in DeepGEMM's MN-major layout."""

from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch
import torch.nn.functional as F

from .cute_codegen import cuda_metadata_trace
import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Generator


def pack_e8m0_scales(scales: torch.Tensor, pad: int = 0) -> torch.Tensor:
    """Pack four scale bytes per MN-major int32 word, padding the last word
    with ``pad`` bytes."""
    packed = (
        F.pad(scales, (0, (-scales.shape[1]) % 4), value=pad)
        .contiguous()
        .view(torch.int32)
    )
    aligned_rows = (packed.shape[0] + 3) // 4 * 4
    storage = torch.empty(
        (packed.shape[1], aligned_rows), device=scales.device, dtype=torch.int32
    )
    result = storage.T[: packed.shape[0]]
    result.copy_(packed)
    return result


def packed_mxfp8_inputs(
    m: int,
    n: int,
    k: int,
    *,
    device: torch.device | str,
    scales: tuple[int, int] = (120, 136),
    pad: int = 0,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """Seeded E4M3 operands ``x`` ([m, k]) and ``y`` ([n, k]) with E8M0 scale
    bytes drawn from ``scales``: the packed kernels' arguments (scales packed
    by ``pack_e8m0_scales``) and the unpacked scale bytes of each."""
    torch.manual_seed(0)
    x = (torch.randn(m, k, device=device) * 0.5).to(torch.float8_e4m3fn)
    y = (torch.randn(n, k, device=device) * 0.5).to(torch.float8_e4m3fn)
    x_scale = torch.randint(*scales, (m, k // 32), device=device, dtype=torch.uint8)
    y_scale = torch.randint(*scales, (n, k // 32), device=device, dtype=torch.uint8)
    args = (x, pack_e8m0_scales(x_scale, pad), y, pack_e8m0_scales(y_scale, pad))
    return args, (x_scale, y_scale)


@helion.kernel(autotune_effort="none", static_shapes=True)
def packed_mxfp8_gemm(
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
        rows = hl.arange(block_m) + task_m * block_m
        cols = hl.arange(block_n) + task_n * block_n
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


@helion.kernel(autotune_effort="none", static_shapes=True)
def two_output_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``packed_mxfp8_gemm`` that also stores ``2 * x @ y.T`` in FP16: two
    flattened outputs, whose stores each need shared memory."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(m * n)
    out2 = torch.empty([m, n], dtype=torch.float16, device=x.device)
    flat_out2 = out2.view(m * n)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = hl.arange(block_m) + task_m * block_m
        cols = hl.arange(block_n) + task_n * block_n
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
        hl.store(
            flat_out2,
            [rows[:, None] * n + cols[None, :]],
            (acc * 2).to(torch.float16),
            extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
        )
    return out, out2


@helion.kernel(autotune_effort="none", static_shapes=False)
def dynamic_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``packed_mxfp8_gemm`` with dynamic shapes, specializing only ``x``'s
    sizes and ``n``: ``y``'s K and every stride stay symbolic."""
    m, k = x.size()
    n = y.size(0)
    hl.specialize(m)
    hl.specialize(n)
    hl.specialize(k)
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(m * n)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = hl.arange(block_m) + task_m * block_m
        cols = hl.arange(block_n) + task_n * block_n
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


@contextmanager
def packed_mxfp8_cpu_codegen() -> Generator[None, None, None]:
    """Generate Triton source for the native packed MXFP8 lowering in a
    CPU-only process, for a 148-SM sm100 target: kernels trace on ``cuda:0``
    (``cute_codegen.cuda_metadata_trace``), and the tensor-descriptor,
    host-descriptor and warp-specialization probes report support, whatever
    devices the process sees."""
    with ExitStack() as stack:
        stack.enter_context(cuda_metadata_trace())
        for name in (
            "helion._compat._supports_tensor_descriptor",
            "helion._compat._supports_host_tensor_descriptor",
            "helion.language.loops._supports_warp_specialize",
        ):
            stack.enter_context(patch(name, return_value=True))
        yield
