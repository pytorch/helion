"""Packed MXFP8 dense GEMM for Blackwell (sm100/sm103) -- Triton backend.

``out = dequant(a) @ dequant(b).T`` in BF16 with FP32 accumulation, where the
E4M3 operands ``a [M, K]`` and ``b [N, K]`` are both K-major and every 32
consecutive K values share one E8M0 exponent (MXFP8, OCP MX v1.0).

The exponents are packed four per int32 word in MN-major order: ``scale[row,
word]`` holds groups ``4 * word .. 4 * word + 3`` of ``row`` in its bytes
(little-endian), with strides ``(1, rows rounded up to 4)``. This is the layout
DeepGEMM's sm100 ``fp8_gemm_nt`` consumes and the order tcgen05's UTCCP scale
copy expects; :func:`pack_mxfp8_scales` builds it from plain ``uint8`` scales.

On sm100/sm103 Helion's Triton backend lowers this kernel's ``hl.dot_scaled``
natively: operands and scale words load through host tensor descriptors, the
words are unpacked once per K tile into UTCCP order, 16/32-row tiles issue the
transposed MMA, and the flattened BF16 output is stored through a 2-D
descriptor. The ``split_k`` tunable splits K over a grid axis for small grids;
partial sums are reduced in FP32 on the host side.

Benchmarked against ``torch._scaled_mm`` with ``BlockWise1x32`` scales (cuBLASLt
MXFP8) and, when it is installed, DeepGEMM's ``fp8_gemm_nt`` with the same
packed scales (``recipe=(1, 1, 32)``). Scale repacking for the baselines is
done once per shape, outside the timed region.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import helion
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


@helion.aot_kernel(
    backend="triton",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def mxfp8_gemm(
    a: torch.Tensor,  # [M, K] float8_e4m3fn
    b: torch.Tensor,  # [N, K] float8_e4m3fn
    scale_a: torch.Tensor,  # [M, ceil(K / 128)] int32, MN-major packed E8M0
    scale_b: torch.Tensor,  # [N, ceil(K / 128)] int32, MN-major packed E8M0
) -> torch.Tensor:
    """Return BF16 ``dequant(a) @ dequant(b).T`` ``[M, N]``."""
    m, k = a.shape
    n = b.size(0)
    assert b.size(1) == k and k % 32 == 0
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert scale_a.dtype == scale_b.dtype == torch.int32
    hl.specialize(m)
    hl.specialize(n)
    hl.specialize(k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(16, 256)
    block_groups = hl.register_block_size(2, 32)
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 16, 1))
    groups_per_split = (k // 32 + split_k - 1) // split_k
    partial = torch.empty((split_k, m, n), dtype=torch.float32, device=a.device)
    flat_out = out.view(m * n)
    flat_partial = partial.view(split_k * m * n)
    for task_m, task_n, part in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n, split_k]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        group_end = min((part + 1) * groups_per_split, k // 32)
        for groups in hl.tile(
            part * groups_per_split, group_end, block_size=block_groups
        ):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                a,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < group_end * 32),
            )
            rhs = hl.load(
                b,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < group_end * 32),
            )
            lhs_words = hl.load(
                scale_a,
                [rows, groups.index // 4],
                extra_mask=(rows[:, None] < m) & (groups.index[None, :] < group_end),
            )
            rhs_words = hl.load(
                scale_b,
                [cols, groups.index // 4],
                extra_mask=(cols[:, None] < n) & (groups.index[None, :] < group_end),
            )
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        if split_k == 1:
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
    if split_k > 1:
        return partial.sum(dim=0).to(torch.bfloat16)
    return out


def pack_mxfp8_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pack ``uint8`` E8M0 scales ``[rows, K / 32]`` into MN-major int32 words.

    Returns a ``[rows, ceil(K / 128)]`` int32 view with strides
    ``(1, rows rounded up to 4)`` (16-byte-aligned columns). Padding groups
    hold exponent 127 (scale 1.0) and padding rows hold zeros.
    """
    groups = scales.shape[1]
    padded = F.pad(scales.view(torch.uint8), (0, (-groups) % 4), value=127)
    words = padded.contiguous().view(torch.int32)
    aligned_rows = (words.shape[0] + 3) // 4 * 4
    storage = torch.zeros(
        (words.shape[1], aligned_rows), device=scales.device, dtype=torch.int32
    )
    packed = storage.T[: words.shape[0]]
    packed.copy_(words)
    return packed


def reference_mxfp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a_u8: torch.Tensor,
    scale_b_u8: torch.Tensor,
) -> torch.Tensor:
    """FP32 dequantize-then-matmul reference from unpacked ``uint8`` scales."""

    def dequantize(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        factors = torch.exp2(scales.to(torch.float32) - 127.0)
        return data.to(torch.float32) * factors.repeat_interleave(32, dim=1)

    return (dequantize(a, scale_a_u8) @ dequantize(b, scale_b_u8).T).to(torch.bfloat16)


def make_inputs(
    m: int, n: int, k: int, *, seed: int = 0
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor]]:
    """``(a, b, packed scale_a, packed scale_b)`` and the unpacked scales."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(m, k, device="cuda", generator=generator) / 3).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn(n, k, device="cuda", generator=generator) / 3).to(
        torch.float8_e4m3fn
    )
    scale_a = torch.randint(
        125, 129, (m, k // 32), device="cuda", dtype=torch.uint8, generator=generator
    )
    scale_b = torch.randint(
        125, 129, (n, k // 32), device="cuda", dtype=torch.uint8, generator=generator
    )
    args = (a, b, pack_mxfp8_scales(scale_a), pack_mxfp8_scales(scale_b))
    return args, (scale_a, scale_b)


def _blocked_scales(scales: torch.Tensor) -> torch.Tensor:
    """cuBLASLt's swizzled 32x4x4 ``BlockWise1x32`` scale layout."""
    rows, groups = scales.shape
    row_blocks, group_blocks = -(-rows // 128), -(-groups // 4)
    padded = torch.full(
        (row_blocks * 128, group_blocks * 4),
        127,
        dtype=torch.uint8,
        device=scales.device,
    )
    padded[:rows, :groups] = scales
    blocked = (
        padded.view(row_blocks, 128, group_blocks, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
    )
    return blocked.view(torch.float8_e8m0fnu)


def _torch_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    blocked_a: torch.Tensor,
    blocked_b: torch.Tensor,
) -> torch.Tensor:
    return torch._scaled_mm(a, b.t(), blocked_a, blocked_b, out_dtype=torch.bfloat16)


@functools.cache
def _deep_gemm() -> ModuleType | None:
    """DeepGEMM (the sm100 MXFP8 GEMM consuming the same packed scales), from
    ``HELION_DEEPGEMM_ROOT`` (the nightly's pinned build) or the environment.
    The pretuned test env has only torch + helion, so it is optional."""
    root = os.environ.get("HELION_DEEPGEMM_ROOT")
    if root:
        sys.path.insert(0, str(Path(root).expanduser().resolve()))
    try:
        import deep_gemm  # pyrefly: ignore[missing-import]
    except ImportError:
        return None
    return deep_gemm if hasattr(deep_gemm, "fp8_gemm_nt") else None


def baselines(
    args: tuple[torch.Tensor, ...], unpacked: tuple[torch.Tensor, torch.Tensor]
) -> list[tuple[str, Callable[[], object]]]:
    """Named zero-argument baseline calls for one shape's inputs.

    ``torch._scaled_mm`` (cuBLASLt MXFP8) is always available; DeepGEMM is added
    when it is importable and runs MXFP8 (``recipe=(1, 1, 32)``). The SUMMARY
    speedup is Helion vs the fastest baseline.
    """
    a, b, scale_a, scale_b = args
    blocked_a, blocked_b = (_blocked_scales(scales) for scales in unpacked)
    calls: list[tuple[str, Callable[[], object]]] = [
        ("torch", lambda: _torch_scaled_mm(a, b, blocked_a, blocked_b))
    ]
    deep_gemm = _deep_gemm()
    if deep_gemm is not None:
        out = torch.empty((a.size(0), b.size(0)), device=a.device, dtype=torch.bfloat16)

        def deepgemm_call() -> torch.Tensor:
            deep_gemm.fp8_gemm_nt((a, scale_a), (b, scale_b), out, recipe=(1, 1, 32))
            return out

        try:
            deepgemm_call()
        except (RuntimeError, TypeError) as error:
            print(f"Skipping the DeepGEMM baseline: {error}")
        else:
            torch.testing.assert_close(
                out,
                reference_mxfp8_gemm(a, b, *unpacked),
                atol=2e-2,
                rtol=1.6e-2,
            )
            calls.append(("deepgemm", deepgemm_call))
    return calls


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py).

    True: these decode-to-prefill GEMMs take a few microseconds, so per-call host
    launch overhead would dominate eager timing. The shared _bench loop clears
    the L2 cache before every replay (cold L2).
    """
    return True


# Decode-to-prefill MXFP8 dense GEMM shapes (M, N, K): token counts from 1 to
# 1024 against four weight shapes with K from 1280 to 5120.
SHAPES = [
    (1, 1792, 5120),
    (8, 8192, 1280),
    (16, 2048, 4096),
    (32, 5120, 2048),
    (128, 1792, 5120),
    (1024, 8192, 1280),
]


def correctness_check(
    kernel: object = mxfp8_gemm, shapes: list[tuple[int, int, int]] | None = None
) -> None:
    """Check ``kernel`` against the FP32 dequantization reference."""
    for m, n, k in shapes or SHAPES:
        args, unpacked = make_inputs(m, n, k)
        actual = kernel(*args)  # pyrefly: ignore[not-callable]
        expected = reference_mxfp8_gemm(args[0], args[1], *unpacked)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1.6e-2)


def main(verbose: bool = True, kernel: object = mxfp8_gemm) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    def make_calls(shape: tuple[int, int, int]) -> tuple:
        m, n, k = shape
        args, unpacked = make_inputs(m, n, k)
        expected = reference_mxfp8_gemm(args[0], args[1], *unpacked)
        torch.testing.assert_close(
            kernel(*args),  # pyrefly: ignore[not-callable]
            expected,
            atol=2e-2,
            rtol=1.6e-2,
        )

        def helion_call() -> None:
            kernel(*args)  # pyrefly: ignore[not-callable]

        return helion_call, baselines(args, unpacked), f"{m:>6d}  {n:>6d}  {k:>6d}"

    return run_sweep(
        SHAPES,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'M':>6s}  {'N':>6s}  {'K':>6s}",
    )


if __name__ == "__main__":
    main()
