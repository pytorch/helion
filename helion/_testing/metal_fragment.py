"""Kernels the Metal fragment-matmul tests share: the GPU module
(``test/test_metal_fragment_mma.py``) runs them and the CPU-only ones
(``test/test_metal_fragment_config.py``, ``test/test_metal_fragment_split.py``,
``test/test_metal_fragment_rows.py``) read their config specs, code and MSL."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

F32 = torch.float32


def e4m3(b: torch.Tensor) -> torch.Tensor:
    """E4M3 byte (an integer tensor) -> float32, via half bits times 2^8."""
    h = ((b & 0x7F) << 7) | ((b & 0x80) << 8)
    return h.to(torch.int16).view(torch.float16).to(F32) * 256.0


def fragment_config(
    block_sizes: list[int], num_threads: list[int] | None = None, **knobs: object
) -> helion.Config:
    """A ``metal_matmul="fragment"`` config at ``num_warps=1`` (one SIMD
    group runs each K loop) unless ``knobs`` set it."""
    extra = {} if num_threads is None else {"num_threads": num_threads}
    return helion.Config.from_dict(
        {
            "block_sizes": block_sizes,
            **extra,
            "metal_matmul": "fragment",
            "num_warps": 1,
            **knobs,
        }
    )


@helion.kernel(backend="metal", static_shapes=True)
def k_from_16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x[:, 16:] @ y[16:]: a K loop from a known offset, over operands it
    converts to half, whose loads a fragment run can issue a tile early."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(16, k):
            a = x[tile_m, tile_k].to(torch.float16)
            b = y[tile_k, tile_n].to(torch.float16)
            acc = torch.addmm(acc, a, b)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def nvfp4_rows(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """nvfp4 x @ W.T: an E2M1 code (nibble k % 8 of word k // 8) through
    half bits, times its E4M3 group scale; bf16 result."""
    m, k = x.size()
    n = wq.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            words = wq[tile_n, tile_k.index // 8]
            nib = (words >> ((tile_k.index % 8) * 4)[None, :]) & 0xF
            q = (((nib & 7) << 9) | ((nib & 8) << 12)).to(torch.int16)
            sc = e4m3(s[tile_n, tile_k.index // 16].to(torch.int32)) * 16384.0
            w = q.view(torch.float16).to(F32) * sc
            acc = torch.addmm(acc, x[tile_m, tile_k], w.T.to(x.dtype))
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def fp8_rows(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """mxfp8 x @ W.T: an E4M3 code (byte k % 4 of word k // 4) times its
    E8M0 scale per 32 elements, as hl.dot."""
    m, k = x.size()
    n = wq.size(0)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            words = wq[tile_n, tile_k.index // 4]
            byte = (words >> ((tile_k.index % 4) * 8)[None, :]) & 0xFF
            sc = (s[tile_n, tile_k.index // 32].to(torch.int32) << 23).view(F32)
            w = (e4m3(byte) * sc).to(x.dtype)
            acc = hl.dot(x[tile_m, tile_k], w.T, acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def views_epilogue(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Both operands device views; a stored epilogue reads a bias over N."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = torch.relu(acc + bias[tile_n][None, :])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def seeded_rhs(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Only the rhs computed, so the accumulator is transposed, and seeded
    from c."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = c[tile_m, tile_n].to(F32)
        for tile_k in hl.tile(k):
            b = (y[tile_k, tile_n].to(F32) * 2.0).to(x.dtype)
            acc = torch.addmm(acc, x[tile_m, tile_k], b)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def two_groups(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two matmuls of x in one K loop, each result stored."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    out2 = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        a = hl.zeros([tile_m, tile_n], dtype=F32)
        b = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            xt = x[tile_m, tile_k]
            a = torch.addmm(a, xt, y[tile_k, tile_n])
            b = torch.addmm(b, xt, z[tile_k, tile_n])
        out[tile_m, tile_n] = a
        out2[tile_m, tile_n] = b
    return out, out2


@helion.kernel(backend="metal", static_shapes=True)
def filled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x @ y + 0.5: an accumulator that starts from 0.5."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.full([tile_m, tile_n], 0.5, dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def nvfp4_cpu_args(rows: int = 8) -> tuple[torch.Tensor, ...]:
    """CPU arguments for ``nvfp4_rows`` (``rows`` rows, K 256, N 48), for
    the tests that read only its config spec and code."""
    return (
        torch.randn(rows, 256).to(torch.bfloat16),
        torch.randint(-(2**31), 2**31 - 1, (48, 32), dtype=torch.int32),
        torch.randint(0, 0x60, (48, 16), dtype=torch.uint8),
    )
