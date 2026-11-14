"""GEMM bodies for the tcgen05 cp.async A producer's tests.

Undecorated: the CPU codegen tests bind them under their target
(``helion.kernel(fn, ...)``), the GPU tests wrap them once.
"""

from __future__ import annotations

import torch

import helion.language as hl


def residual_gemm(
    a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc + residual[tile_m, tile_n]
    return out


def gemm_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``a @ b`` into a caller's FP32 ``out`` (any layout)."""
    m, k = a.size()
    _, n = b.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def two_output_gemm(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``a @ b`` stored twice: two output stores of one accumulator."""
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    out2 = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        out2[tile_m, tile_n] = acc
    return out, out2


def keyword_addmm_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` through ``torch.addmm`` keywords written B first."""
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(input=acc, mat2=b[tile_k, tile_n], mat1=a[tile_m, tile_k])
        out[tile_m, tile_n] = acc
    return out
