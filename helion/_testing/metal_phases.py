"""Kernels the Metal ``hl.barrier()`` tests share: the CPU-only module
(``test/test_metal_phase_codegen.py``) reads the code generated for them and
the GPU one (``test/test_metal_phases.py``) runs them."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="metal", static_shapes=True)
def pingpong(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Four phases, each reading what other threadgroups wrote in the one
    before (reversed), alternating between two buffers so every phase reads
    lines a core may already hold from an earlier phase."""
    n = a.size(0)
    for t in hl.tile(n):
        b[t] = a[n - 1 - t.index] * 0.5 + 1.0
    hl.barrier()
    for t in hl.tile(n):
        a[t] = b[n - 1 - t.index] * 0.5 + 1.0
    hl.barrier()
    for t in hl.tile(n):
        b[t] = a[n - 1 - t.index] * 0.5 + 1.0
    hl.barrier()
    for t in hl.tile(n):
        a[t] = b[n - 1 - t.index] * 0.5 + 1.0
    return a


@helion.kernel(backend="metal", static_shapes=True)
def row_sums_then_scale(x: torch.Tensor) -> torch.Tensor:
    """Two roots in the first phase, a reduction over their outputs in the
    second, and an intermediate written through one view and read through
    another."""
    m, n = x.size()
    s = torch.empty([m], dtype=torch.float32, device=x.device)
    y = torch.empty([m, n], dtype=torch.float32, device=x.device)
    y2 = y.view(m * n)
    xf = x.view(m * n)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        s[tm] = torch.sum(x[tm, :].to(torch.float32), -1)
    for t in hl.tile(m * n):
        y2[t] = xf[t].to(torch.float32) * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        rev = s[m - 1 - tm.index]
        out[tm, tn] = y[tm, tn] + rev.unsqueeze(1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def matvec_then_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A few-row matmul whose K is split over threads (a 2-D thread layout),
    then a row norm over its output written as a tile reduction (a 1-D one):
    the roots lay the threadgroup out differently."""
    m, k = x.size()
    n = w.size(0)
    bk = hl.register_block_size(k)
    bn = hl.register_block_size(n)
    y = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn, bk], dtype=torch.float32)
        for tk in hl.tile(k, block_size=bk):
            acc = acc + x[tm, tk][:, None, :] * w[tn, tk][None, :, :]
        y[tm, tn] = acc.sum(2)
    hl.barrier()
    for tm in hl.tile(m):
        ss = hl.zeros([tm, bn], dtype=torch.float32)
        for th in hl.tile(n, block_size=bn):
            v = y[tm, th]
            ss = ss + v * v
        inv = torch.rsqrt(ss.sum(1) / n)
        for th in hl.tile(n, block_size=bn):
            out[tm, th] = y[tm, th] * inv[:, None]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scale_then_reverse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Phase 0 updates ``x`` in place and writes ``y``; phase 1 reads both
    reversed.  With tiles narrower than a cache line, a threadgroup's read in
    phase 0 can cache a line whose other half another threadgroup writes."""
    n = x.size(0)
    out = torch.empty_like(x)
    for t in hl.tile(n):
        x[t] = x[t] * 2.0 + 1.0
        y[t] = x[t] * 0.5
    hl.barrier()
    for t in hl.tile(n):
        out[t] = x[n - 1 - t.index] + y[n - 1 - t.index]
    return out


@helion.kernel(backend="metal", static_shapes=False)
def three_phases(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Three phases over inputs that may be empty: phase 1 has no tasks when
    ``b`` is empty, phase 0 none when ``a`` is."""
    n = a.size(0)
    m = b.size(0)
    out = torch.empty_like(a)
    for t in hl.tile(n):
        a[t] = a[t] + 1.0
    hl.barrier()
    for t in hl.tile(m):
        b[t] = b[t] * 2.0
    hl.barrier()
    for t in hl.tile(n):
        out[t] = a[n - 1 - t.index] * 3.0
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scale_then_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Phase 0 scales ``x`` into an intermediate that phase 1 multiplies by
    ``w``: an operand handed from one phase to the next, which an MPP
    matmul stages through its coherent alias."""
    m, k = x.size()
    n = w.size(1)
    xs = torch.empty_like(x)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for t0, t1 in hl.tile([m, k]):
        xs[t0, t1] = x[t0, t1] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, xs[tm, tk], w[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scale_by_first(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Phase 0 writes ``2 * x[0]`` into the 0-d ``scale`` through a view, and
    phase 1 multiplies ``x`` by it: a 0-d tensor handed on."""
    out = torch.empty_like(x)
    flat = scale.view(1)
    for tile in hl.tile(1):
        flat[tile] = x[tile] * 2.0
    hl.barrier()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * hl.load(scale, [])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def write_then_matmul(
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """Phase 0 writes ``y`` and phase 1 multiplies ``a`` by ``w``: a call
    that passes ``y`` as ``a`` hands phase 0's writes to the matmul."""
    m, k = a.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for t0, t1 in hl.tile([m, k]):
        y[t0, t1] = x[t0, t1] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], w[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def bounded_by_tensor(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Phase 1 runs to a bound read from the 0-d ``n``: a data-dependent
    loop bound, which rules out the flat launch."""
    out = torch.zeros_like(x)
    for t in hl.tile(x.size(0)):
        out[t] = x[t] + 1
    hl.barrier()
    for t in hl.tile(n, block_size=32):
        out[t] = out[t] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def two_phase_matmuls(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two phases, each a K-loop matmul of ``x`` into its own output, reading
    nothing the kernel writes."""
    m, k = x.size()
    n = y.size(1)
    a = torch.empty([m, n], dtype=torch.float32, device=x.device)
    b = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        a[tm, tn] = acc
    hl.barrier()
    for tm2, tn2 in hl.tile([m, n]):
        acc2 = hl.zeros([tm2, tn2], dtype=torch.float32)
        for tk2 in hl.tile(k):
            acc2 = torch.addmm(acc2, x[tm2, tk2], z[tk2, tn2])
        b[tm2, tn2] = acc2
    return a, b
