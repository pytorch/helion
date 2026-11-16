"""Kernels the Metal multi-root tests share: the CPU-only module
(``test/test_metal_multi_root.py``) reads the code generated for them and the
GPU one (``test/test_metal.py``) runs them."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="metal", static_shapes=True)
def update_then_scale(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two roots and no barrier: the first updates ``x`` in place, the second
    scales ``y``.  Tiles of different sizes lay them out in different
    threads, and both run in the wider root's threadgroup."""
    out = torch.empty_like(y)
    for t in hl.tile(x.size(0)):
        x[t] = x[t] * 2.0 + 1.0
    for t in hl.tile(y.size(0)):
        out[t] = y[t] * 0.5
    return out


@helion.kernel(backend="metal", static_shapes=True)
def matmul_then_scale(
    x: torch.Tensor, w: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A matmul root beside an elementwise one, without a barrier."""
    m, k = x.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    ys = torch.empty_like(y)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[tk, tn])
        out[tm, tn] = acc
    for t in hl.tile(y.size(0)):
        ys[t] = y[t] * 2.0
    return out, ys


@helion.kernel(backend="metal", static_shapes=True)
def update_beside_rowsum(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """An in-place update beside a row sum, without a barrier.  The sum's
    reduction takes the threadgroup's threads, and every root runs its
    setup, so the update's recorded layout is as wide."""
    s = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for t in hl.tile(y.size(0)):
        y[t] = y[t] * 2.0 + 1.0
    for tm in hl.tile(x.size(0)):
        s[tm] = x[tm, :].sum(-1)
    return s


@helion.kernel(backend="metal", static_shapes=True)
def scale_then_update_then_tile(
    x: torch.Tensor, c: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Two phases: ``x`` scaled in place, then each program updates one
    element of ``c`` in place and scales ``y`` in a loop whose tiles take the
    threads.  The update depends on no thread, so the threads of ``y``'s
    tiles would repeat it."""
    out = torch.empty_like(y)
    for t in hl.tile(x.size(0)):
        x[t] = x[t] * 2.0
    hl.barrier()
    for t in hl.tile(c.size(0)):
        c[t] = c[t] * 2.0 + 1.0
        for ty in hl.tile(y.size(0)):
            out[ty] = y[ty] * 0.5
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scale_then_copy_then_tile(
    x: torch.Tensor, c: torch.Tensor, d: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """:func:`scale_then_update_then_tile` with the update reading ``d``:
    ``c[t] = d[t] * 2.0 + 1.0`` is an in-place update only when ``c`` and
    ``d`` share a storage, which shows at the call alone."""
    out = torch.empty_like(y)
    for t in hl.tile(x.size(0)):
        x[t] = x[t] * 2.0
    hl.barrier()
    for t in hl.tile(c.size(0)):
        c[t] = d[t] * 2.0 + 1.0
        for ty in hl.tile(y.size(0)):
            out[ty] = y[ty] * 0.5
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scale_then_toggle_then_tile(
    x: torch.Tensor, c: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """:func:`scale_then_update_then_tile` with the update a sign flip by a
    test of the element: its stores write constants, but the test reads
    what they write."""
    out = torch.empty_like(y)
    for t in hl.tile(x.size(0)):
        x[t] = x[t] * 2.0
    hl.barrier()
    for i in hl.grid(c.size(0)):
        if hl.load(c, [i]) > 0:
            hl.store(c, [i], -1.0)
        else:
            hl.store(c, [i], 1.0)
        for ty in hl.tile(y.size(0)):
            out[ty] = y[ty] * 0.5
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scale_then_round_trip_then_tile(
    x: torch.Tensor, c: torch.Tensor, tmp: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """:func:`scale_then_update_then_tile` with the update made through a
    scratch tensor, written and read back on the way."""
    out = torch.empty_like(y)
    for t in hl.tile(x.size(0)):
        x[t] = x[t] * 2.0
    hl.barrier()
    for t in hl.tile(c.size(0)):
        tmp[t] = c[t] + 1.0
        c[t] = tmp[t] * 2.0
        for ty in hl.tile(y.size(0)):
            out[ty] = y[ty] * 0.5
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scale_then_accumulate_then_tile(
    x: torch.Tensor, c: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """:func:`scale_then_update_then_tile` with the update carried through a
    loop: ``acc`` adds up ``c[t]`` four times, then ``c[t] = acc``."""
    out = torch.empty_like(y)
    for t in hl.tile(x.size(0)):
        x[t] = x[t] * 2.0
    hl.barrier()
    for t in hl.tile(c.size(0)):
        acc = hl.zeros([t], dtype=torch.float32)
        for _ in hl.tile(4):
            acc = acc + c[t]
        c[t] = acc
        for ty in hl.tile(y.size(0)):
            out[ty] = y[ty] * 0.5
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scatter_beside_scale(
    x: torch.Tensor, out: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Two roots without a barrier: each row of the first scatters 1.0 into
    ``out`` at an index summed over a loop, beside an elementwise root.  The
    scatter's address reaches its row's thread only through the sum the
    loop carries."""
    ys = torch.empty_like(y)
    for t in hl.tile(x.size(0)):
        acc = hl.zeros([t], dtype=torch.int64)
        for tk in hl.tile(x.size(1)):
            acc = acc + x[t, tk].sum(-1)
        out[acc] = 1.0
    for t2 in hl.tile(y.size(0)):
        ys[t2] = y[t2] * 2.0
    return ys


@helion.kernel(backend="metal", static_shapes=True)
def scale_then_add_rowsum(y: torch.Tensor, x: torch.Tensor, s: torch.Tensor) -> None:
    """Two phases: ``y`` scaled in place, then each program adds a row sum of
    ``x`` into its element of ``s``.  The update's address depends on no
    thread; its value reads the sum, which every thread of the reduction
    gets, and each of those threads would apply it."""
    for t in hl.tile(y.size(0)):
        y[t] = y[t] * 2.0
    hl.barrier()
    for tm in hl.tile(x.size(0)):
        s[tm] = x[tm, :].sum(-1) + s[tm]


@helion.kernel(backend="metal", static_shapes=True)
def scale_then_add_masked_rowsum(
    y: torch.Tensor, x: torch.Tensor, s: torch.Tensor
) -> None:
    """:func:`scale_then_add_rowsum` with the update masked by the sum, which,
    like the value, every thread of the reduction gets."""
    for t in hl.tile(y.size(0)):
        y[t] = y[t] * 2.0
    hl.barrier()
    for tm in hl.tile(x.size(0)):
        r = x[tm, :].sum(-1)
        hl.store(s, [tm], r + s[tm], extra_mask=r > 0)


@helion.kernel(backend="metal", static_shapes=True)
def grid_beside_row(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two roots without a barrier: a 2-D tile beside a 1-D one, whose
    per-axis maximum can outgrow a threadgroup."""
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    for tm, tn in hl.tile(x.size()):
        out_x[tm, tn] = x[tm, tn] * 2.0
    for t in hl.tile(y.size(0)):
        out_y[t] = y[t] * 0.5
    return out_x, out_y


@helion.kernel(backend="metal", static_shapes=True)
def write_then_read_back(
    x: torch.Tensor, s_src: torch.Tensor, s: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two roots without a barrier: each tile of the first writes its rows of
    ``s``, then every thread of the tile reads its row back to scale ``x``;
    the second scales ``y``.  Writing a row depends on no column thread, but
    each thread reads back what it wrote, so no thread the first root lays
    out may skip the write."""
    out = torch.empty_like(x)
    out_y = torch.empty_like(y)
    for tm, tn in hl.tile(x.size()):
        s[tm] = s_src[tm] * 2.0
        out[tm, tn] = x[tm, tn] * s[tm][:, None]
    for t in hl.tile(y.size(0)):
        out_y[t] = y[t] * 0.5
    return out, out_y
