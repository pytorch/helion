"""Kernels the Metal fragment-matmul tests share: the GPU module
(``test/test_metal_fragment_mma.py``) runs them, and the CPU-only
``test/test_metal_fragment_*.py`` modules read their config specs, seeds,
code and MSL."""

from __future__ import annotations

import re

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


@helion.kernel(backend="metal", static_shapes=True)
def affine_mm(
    x: torch.Tensor, w: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor
) -> torch.Tensor:
    """x @ W.T for W in MLX's affine format: ``W[n, k] = scales[n, g] * code +
    biases[n, g]``, ``g = k // group``, with the codes packed little-endian in
    int16 words w [N, K * bits / 16]; the bit width and the group follow
    from the shapes.  A code that straddles two words (5 bits: 4 of every
    16; 6 bits: 2 of every 8) reads both.  The product runs on float
    operands, the weight unrounded."""
    m, k = x.size()
    n, kw = w.size()
    bits = 16 * kw // k
    group = k // scales.size(1)
    mask = (1 << bits) - 1
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            bit = tile_k.index * bits
            lo = w[tile_n, bit // 16].to(torch.int32) & 0xFFFF
            if 16 % bits == 0:
                q = (lo >> (bit % 16)[None, :]) & mask
            else:
                nxt = torch.clamp(bit // 16 + 1, max=kw - 1)
                hi = w[tile_n, nxt].to(torch.int32) & 0xFFFF
                q = ((lo | (hi << 16)) >> (bit % 16)[None, :]) & mask
            grp = tile_k.index // group
            deq = q.to(F32) * scales[tile_n, grp].to(F32) + biases[tile_n, grp].to(F32)
            acc = torch.addmm(acc, x[tile_m, tile_k].to(F32), deq.T)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def affine_mm_f16(
    x: torch.Tensor, w: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor
) -> torch.Tensor:
    """:func:`affine_mm` with the dequantized weight rounded to x's dtype, so
    the product runs on x as it is, a 16-bit device view beside the weight:
    MLX's quantized matmul over many rows on an M5.  The codes may be packed
    in 8-bit words too, and a row of w may hold one word past its codes,
    which a code's next word then reads unclamped (:func:`affine_args`)."""
    m, k = x.size()
    n, kw = w.size()
    word = 8 * w.element_size()
    padded = word * kw % k != 0
    bits = word * (kw - padded) // k
    group = k // scales.size(1)
    mask = (1 << bits) - 1
    ones = (1 << word) - 1
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            bit = tile_k.index * bits
            lo = w[tile_n, bit // word].to(torch.int32) & ones
            if word % bits == 0:
                q = (lo >> (bit % word)[None, :]) & mask
            else:
                if padded:
                    nxt = bit // word + 1
                else:
                    nxt = torch.clamp(bit // word + 1, max=kw - 1)
                hi = w[tile_n, nxt].to(torch.int32) & ones
                q = ((lo | (hi << word)) >> (bit % word)[None, :]) & mask
            grp = tile_k.index // group
            deq = q.to(F32) * scales[tile_n, grp].to(F32) + biases[tile_n, grp].to(F32)
            acc = torch.addmm(acc, x[tile_m, tile_k], deq.to(x.dtype).T)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def affine_args(
    bits: int,
    group: int,
    m: int,
    n: int,
    k: int,
    device: str = "cpu",
    words: torch.dtype = torch.int16,
    padded: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Arguments of :func:`affine_mm`: random fp16 rows, codes packed in
    ``words`` (``torch.int16`` or, for :func:`affine_mm_f16`, ``torch.uint8``),
    one word more a row if ``padded``, and scales and biases."""
    gen = torch.Generator().manual_seed(bits * 1000 + m)
    x = torch.randn(m, k, generator=gen).to(torch.float16)
    word = 8 * words.itemsize
    low, high = (0, 256) if words == torch.uint8 else (-(2**15), 2**15)
    w = torch.randint(
        low, high, (n, k * bits // word + padded), generator=gen, dtype=words
    )
    scales = (torch.rand(n, k // group, generator=gen) * 0.02).to(torch.float16)
    biases = (torch.randn(n, k // group, generator=gen) * 0.02).to(torch.float16)
    return tuple(t.to(device) for t in (x, w, scales, biases))


def affine_reference(
    args: tuple[torch.Tensor, ...], weight: torch.dtype = torch.float64
) -> torch.Tensor:
    """:func:`affine_mm` in float64 on the CPU, the dequantized weight rounded
    to ``weight`` first (:func:`affine_mm_f16`'s ``torch.float16``)."""
    x, w, scales, biases = (t.cpu() for t in args)
    n, kw = w.shape
    k = x.size(1)
    word = 8 * w.element_size()
    bits = word * (kw - (word * kw % k != 0)) // k
    group = k // scales.size(1)
    words = w.to(torch.int64) & ((1 << word) - 1)
    bit = torch.arange(k) * bits
    lo = words[:, bit // word]
    hi = words[:, torch.clamp(bit // word + 1, max=kw - 1)]
    codes = ((lo | (hi << word)) >> (bit % word)) & ((1 << bits) - 1)
    grp = torch.arange(k) // group
    deq = codes.double() * scales.double()[:, grp] + biases.double()[:, grp]
    return x.double() @ deq.to(weight).double().T


def fragment_transposed(msl: str) -> bool:
    """Whether a fragment matmul's accumulator is transposed in its MSL: its
    A fragments' rows run over N (``fragment_mma.transposed``)."""
    a_rows = re.search(r"int (_e_[mn]) = 8 \* \(_fi >> 1\) \+ _fm;", msl)
    assert a_rows is not None, "no fragment run"
    return a_rows.group(1) == "_e_n"


@helion.kernel(backend="metal", static_shapes=True)
def batched_k_loop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x @ y per batch: a ``torch.baddbmm`` K loop on each batch tile, the
    lhs computed (scaled), the rhs a view, a stored epilogue."""
    b, m, k = x.size()
    n = y.size(2)
    out = torch.empty([b, m, n], dtype=F32, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            a = (x[tile_b, tile_m, tile_k] * 0.5).to(x.dtype)
            acc = torch.baddbmm(acc, a, y[tile_b, tile_k, tile_n])
        out[tile_b, tile_m, tile_n] = acc + 1.0
    return out


@helion.kernel(backend="metal", static_shapes=True)
def batched_whole_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x @ y.T per batch over a ':' K, as attention's scores: one
    ``torch.bmm`` run per batch tile and N tile of an inner loop.  The lhs,
    loaded before the loop, is computed (re-run at each element); the rhs
    is a transposed view."""
    b, m, k = x.size()
    n = y.size(1)
    out = torch.empty([b, m, n], dtype=F32, device=x.device)
    for tile_b, tile_m in hl.tile([b, m]):
        q = x[tile_b, tile_m, :]
        for tile_n in hl.tile(n):
            s = torch.bmm(q, y[tile_b, tile_n, :].transpose(1, 2), F32)
            out[tile_b, tile_m, tile_n] = s
    return out


@helion.kernel(backend="metal", static_shapes=True)
def matmul_softmax(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """softmax(x @ y, dim=1), N whole: two row reductions of the K loop's
    accumulator (its max and the sum of its exponentials) that the store's
    element loop reads at each element."""
    m, k = x.size()
    n = hl.specialize(y.size(1))
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        mx = torch.amax(acc, 1)
        p = torch.exp(acc - mx[:, None])
        out[tile_m, :] = p / p.sum(1)[:, None]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_lse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """logsumexp(x @ y, dim=1), online over N tiles: a ':' K matmul per N
    tile, whose row max and row sum update a running max and sum the N loop
    carries, then stored per row."""
    m = x.size(0)
    hl.specialize(x.size(1))
    n = y.size(1)
    lse = torch.empty([m], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        m_i = hl.full([tile_m], float("-inf"), dtype=F32)
        l_i = hl.zeros([tile_m], dtype=F32)
        q = x[tile_m, :]
        for tile_n in hl.tile(n):
            s = torch.mm(q, y[:, tile_n])
            m_new = torch.maximum(m_i, torch.amax(s, 1))
            p = torch.exp(s - m_new[:, None])
            l_i = l_i * torch.exp(m_i - m_new) + torch.sum(p, 1)
            m_i = m_new
        lse[tile_m] = m_i + torch.log(l_i)
    return lse


@helion.kernel(backend="metal", static_shapes=True)
def online_rescale(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """An accumulator rescaled before each N tile's matmul by a row value,
    as attention rescales its output: ``acc = acc * exp(m - m_new)[:, None]
    + w[:, tile_n] @ v[tile_n]``, m the running row max of x @ y; the
    epilogue reads the final max."""
    m = x.size(0)
    hl.specialize(x.size(1))
    n = y.size(1)
    d = hl.specialize(v.size(1))
    out = torch.empty([m, d], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        m_i = hl.full([tile_m], float("-inf"), dtype=F32)
        acc = hl.zeros([tile_m, d], dtype=F32)
        for tile_n in hl.tile(n):
            s = torch.mm(x[tile_m, :], y[:, tile_n])
            m_new = torch.maximum(m_i, torch.amax(s, 1))
            acc = acc * torch.exp(m_i - m_new)[:, None]
            acc = torch.addmm(acc, w[tile_m, tile_n], v[tile_n, :])
            m_i = m_new
        out[tile_m, :] = acc * torch.exp(m_i)[:, None]
    return out


def online_rescale_ref(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, v: torch.Tensor, block_n: int
) -> torch.Tensor:
    """``online_rescale``'s result in float64 on the CPU, over N tiles of
    ``block_n``: each tile's product scaled by exp of the running max after
    it."""
    xd, yd, wd, vd = (t.cpu().double() for t in (x, y, w, v))
    s = xd @ yd
    out = torch.zeros(x.size(0), v.size(1), dtype=torch.float64)
    run = torch.full((x.size(0),), float("-inf"), dtype=torch.float64)
    for n0 in range(0, y.size(1), block_n):
        run = torch.maximum(run, s[:, n0 : n0 + block_n].amax(1))
        out += (wd[:, n0 : n0 + block_n] @ vd[n0 : n0 + block_n]) * run.exp()[:, None]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def chained_rhs(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``w @ exp(x @ y)`` over M tiles: each tile's scores, a ':' K matmul,
    feed the rhs of the accumulating matmul, whose K is their M (and whose
    M, w's rows, a ':' slice)."""
    m = x.size(0)
    hl.specialize(x.size(1))
    n = y.size(1)
    d = hl.specialize(w.size(0))
    out = torch.empty([d, n], dtype=F32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([d, tile_n], dtype=F32)
        for tile_m in hl.tile(m):
            s = torch.mm(x[tile_m, :], y[:, tile_n])
            acc = torch.addmm(acc, w[:, tile_m], torch.exp(s))
        out[:, tile_n] = acc
    return out
