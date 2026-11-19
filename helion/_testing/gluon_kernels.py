"""Gluon-backend kernels shared by the GPU tests (``test/test_gluon.py``,
``test/test_gluon_ring.py``) and ``test/test_gluon_units.py`` (their code
generated on CPU)."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True, backend="gluon")
def dot_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def nvfp4_mm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """nvfp4 (e4m3 scales per 16) times nvfp4, both packed along K."""
    m = x.size(0)
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for kk in hl.grid(y.size(0) // 128):
            acc = hl.dot_scaled(
                x[tile_m, hl.arange(128) + kk * 128],
                x_scale[tile_m, hl.arange(16) + kk * 16],
                "e2m1",
                y[hl.arange(128) + kk * 128, tile_n],
                y_scale[tile_n, hl.arange(16) + kk * 16],
                "e2m1",
                acc=acc,
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def mm_then_mm(x: torch.Tensor, w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A long dot of inputs, then a dot of its output: both on the MMA
    partition's channel, the second's activations from the gated producer
    (plain loads under barrier mode)."""
    m, k = x.size()
    _, n = w.size()
    _, n2 = v.size()
    y = torch.empty([m, n], dtype=x.dtype, device=x.device)
    out = torch.empty([m, n2], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        y[tile_m, tile_n] = acc.to(x.dtype)
    for tile_p, tile_q in hl.tile([m, n2]):
        acc2 = hl.zeros([tile_p, tile_q], dtype=torch.float32)
        for tile_j in hl.tile(n):
            acc2 = hl.dot(y[tile_p, tile_j], v[tile_j, tile_q], acc=acc2)
        out[tile_p, tile_q] = acc2
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def scale_then_wx(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weight rows as the MMA's M (swap-AB) over the first root's output."""
    m, k = x.size()
    n = w.size(0)
    y = torch.empty_like(x)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
        out[tile_n, tile_m] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def scale_then_two_wx(
    x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scale_then_wx`` for two weights, each product a root (task) of its own."""
    m, k = x.size()
    n = w1.size(0)
    y = torch.empty_like(x)
    out1 = torch.empty([n, m], dtype=torch.float32, device=x.device)
    out3 = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w1[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
        out1[tile_n, tile_m] = acc
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w3[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
        out3[tile_n, tile_m] = acc
    return out1, out3


@helion.kernel(static_shapes=True, backend="gluon")
def scale_then_wx_and_f16_vx(
    x: torch.Tensor, w: torch.Tensor, xh: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scale_then_wx`` and, after its K loop in the same task, an fp16 one
    (tiles of another shape)."""
    m, k = x.size()
    n = w.size(0)
    kv = xh.size(1)
    y = torch.empty_like(x)
    yh = torch.empty_like(xh)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    outh = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, kv]):
        yh[tile_r, tile_c] = xh[tile_r, tile_c] * 2
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
        out[tile_n, tile_m] = acc
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(kv):
            acc = hl.dot(v[tile_n, tile_k], yh[tile_m, tile_k].T, acc=acc)
        outh[tile_n, tile_m] = acc
    return out, outh


@helion.kernel(static_shapes=True, backend="gluon")
def scale_then_wx_and_nvfp4(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scale_then_wx`` and, in the same task, an nvfp4 product (e4m3 scales
    per 16, both packed along K)."""
    m, k = x.size()
    n = w.size(0)
    y = torch.empty_like(x)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    scaled = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
        out[tile_n, tile_m] = acc
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for kk in hl.grid(b.size(0) // 128):
            acc = hl.dot_scaled(
                a[tile_n, hl.arange(128) + kk * 128],
                a_scale[tile_n, hl.arange(16) + kk * 16],
                "e2m1",
                b[hl.arange(128) + kk * 128, tile_m],
                b_scale[tile_m, hl.arange(16) + kk * 16],
                "e2m1",
                acc=acc,
            )
        scaled[tile_n, tile_m] = acc
    return out, scaled


@helion.kernel(static_shapes=True, backend="gluon")
def scale_then_nested_wx(
    x: torch.Tensor, w: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scale_then_wx`` with a second K loop, over ``v``, in the first's body."""
    m, k = x.size()
    n = w.size(0)
    y = torch.empty_like(x)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    inner_out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
            inner = hl.zeros([tile_n, tile_m], dtype=torch.float32)
            for tile_j in hl.tile(k):
                inner = hl.dot(v[tile_n, tile_j], y[tile_m, tile_j].T, acc=inner)
            inner_out[tile_n, tile_m] = inner
        out[tile_n, tile_m] = acc
    return out, inner_out
