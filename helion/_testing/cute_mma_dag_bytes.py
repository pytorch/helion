"""Small matrix products for byte operand staging tests."""

from __future__ import annotations

import pytest
import torch

from helion._compiler.cute.mma_support import get_cute_mma_support
import helion.language as hl


def byte_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    byte_rhs: hl.constexpr,
    transpose_byte: hl.constexpr,
    wrap_byte: hl.constexpr,
) -> torch.Tensor:
    m = a.size(1) if transpose_byte and not byte_rhs else a.size(0)
    n = hl.specialize(b.size(0) if transpose_byte and byte_rhs else b.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        av = a[:, tm].T if transpose_byte and not byte_rhs else a[tm, :]
        bv = b[:, :].T if transpose_byte and byte_rhs else b[:, :]
        result = hl.zeros([tm, n], dtype=torch.float32)
        for _repeat in hl.grid(2):
            if byte_rhs:
                byte_value = bv + 1 if wrap_byte else bv
                result = hl.dot(av, (byte_value.float() * 0.5).to(av.dtype), acc=result)
            else:
                byte_value = av + 1 if wrap_byte else av
                result = hl.dot((byte_value.float() * 0.5).to(bv.dtype), bv, acc=result)
        out[tm, :] = result
    return out


def byte_loop_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = a.size(1), hl.specialize(b.size(1))
    steps = hl.specialize(a.size(0))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        result = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(steps):
            values = (a[step, tm, :].float() * 0.5).to(b.dtype)
            result = hl.dot(values, b[:, :], acc=result)
        out[tm, :] = result
    return out


def require_byte_mma_mode(mode: str) -> None:
    """Require SM100-family support for native tcgen05 F16/BF16 byte recipes."""
    if mode == "tcgen05" and (
        torch.cuda.get_device_capability()[0] != 10
        or not get_cute_mma_support().tcgen05_f16bf16
    ):
        pytest.skip("requires SM100-family TCgen05 F16/BF16 support")


def byte_rhs_tiled_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile((a.size(0), b.size(1))):
        result = hl.zeros((tm, tn), dtype=torch.float32)
        for _repeat in hl.grid(2):
            result = hl.dot(a[tm, :], (b[:, tn].float() * 0.5).to(a.dtype), acc=result)
        out[tm, tn] = result
    return out
