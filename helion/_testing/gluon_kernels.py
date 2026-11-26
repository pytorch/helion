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
def split_k_mm(x: torch.Tensor, y: torch.Tensor, offset: int = 0) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    offset = hl.specialize(offset)
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, 256]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for inner_k in hl.tile(outer_k.begin + offset, outer_k.end):
            acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
        hl.atomic_add(out, [tile_m, tile_n], acc)
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
def mm_then_scaled_mm(
    x: torch.Tensor, w: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``mm_then_mm`` with the second dot's activation doubled in the body (a
    filled operand, ``gluon_ring.ring_fill``)."""
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
            a = (y[tile_p, tile_j].to(torch.float32) * 2).to(x.dtype)
            acc2 = hl.dot(a, v[tile_j, tile_q], acc=acc2)
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
def scale_then_chunked_wx(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``scale_then_wx`` over K chunks, a task each, FP32 partials per chunk: the
    weight a chunked view ([n, chunks, k / chunks], [128, 1, 64] boxes)."""
    m, k = x.size()
    n, chunks, chunk_k = w.size()
    y = torch.empty_like(x)
    y_chunks = y.view(m, chunks, chunk_k)
    out = torch.empty([chunks, n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_h, tile_m in hl.tile([n, chunks, m], block_size=[None, 1, None]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(chunk_k):
            acc = hl.dot(
                w[tile_n, tile_h.begin, tile_k],
                y_chunks[tile_m, tile_h.begin, tile_k].T,
                acc=acc,
            )
        out[tile_h.begin, tile_n, tile_m] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def scale_then_normed_wx(
    x: torch.Tensor, g: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """``scale_then_wx`` with the activation the body computes each K step from
    the first root's output and a vector, both loaded ahead of the step's weight
    tile, in the weight's dtype (a norm and a quantization): the body fills its
    slot of the step (``gluon_ring.ring_fill``)."""
    m, k = x.size()
    n = w.size(0)
    y = torch.empty_like(x)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            a = (
                y[tile_m, tile_k].to(torch.float32)
                * g[tile_k].to(torch.float32)[None, :]
            ).to(w.dtype)
            acc = hl.dot(w[tile_n, tile_k], a.T, acc=acc)
        out[tile_n, tile_m] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def wx_then_scaled_vx(
    x: torch.Tensor, w: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Two K loops of a task: ``w`` times the input, then ``v`` times the input
    doubled in the body each K step, which fills its slot of the step: no
    producer streams the gated group the filled operand takes, so the body
    arrives for it at the first loop's steps (the weight producer only waits)."""
    m, k = x.size()
    n = w.size(0)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w[tile_n, tile_k], x[tile_m, tile_k].T, acc=acc)
        acc2 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            a = (x[tile_m, tile_k].to(torch.float32) * 2).to(x.dtype)
            acc2 = hl.dot(v[tile_n, tile_k], a.T, acc=acc2)
        out[tile_n, tile_m] = acc + acc2
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def gated_wy_then_scaled_vx(
    x: torch.Tensor, g: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The first root doubles ``x`` into ``y``; the second multiplies ``w1`` by
    ``y`` (a TMA operand gated on the first root); the third multiplies ``w2``
    by ``x`` scaled by ``g`` in the body each K step, whose gating group the
    second root's ``y`` streams on the same channel: the body could not arrive
    for that group, so the computed operand is not filled and its dot issues
    from the body."""
    m, k = x.size()
    n = w1.size(0)
    y = torch.empty_like(x)
    out1 = torch.empty([n, m], dtype=torch.float32, device=x.device)
    out2 = torch.empty([n, m], dtype=torch.float32, device=x.device)
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
            a = (
                x[tile_m, tile_k].to(torch.float32)
                * g[tile_k].to(torch.float32)[None, :]
            ).to(x.dtype)
            acc = hl.dot(w2[tile_n, tile_k], a.T, acc=acc)
        out2[tile_n, tile_m] = acc
    return out1, out2


@helion.kernel(static_shapes=True, backend="gluon")
def gated_wy_beside_scaled_vx(
    x: torch.Tensor, g: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """``gated_wy_then_scaled_vx``'s two dots in one K loop."""
    m, k = x.size()
    n = w1.size(0)
    y = torch.empty_like(x)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        acc2 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w1[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
            a = (
                x[tile_m, tile_k].to(torch.float32)
                * g[tile_k].to(torch.float32)[None, :]
            ).to(x.dtype)
            acc2 = hl.dot(w2[tile_n, tile_k], a.T, acc=acc2)
        out[tile_n, tile_m] = acc + acc2
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def filled_beside_two_dots(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """One K loop of three dots: ``v`` times an operand the body computes (from
    the tile's indices, in ``x``'s dtype), which fills its slot of the step, then
    ``w`` times ``y`` and ``u`` times ``z`` (TMA operands): the MMA channel's
    other steps run between the body's fills."""
    m, k = x.size()
    n = w.size(0)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        acc2 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        acc3 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            a = (
                tile_m.index[:, None].to(torch.float32) * 0.01
                + tile_k.index[None, :].to(torch.float32) * 0.001
            ).to(x.dtype)
            acc = hl.dot(v[tile_n, tile_k], a.T, acc=acc)
            acc2 = hl.dot(w[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc2)
            acc3 = hl.dot(u[tile_n, tile_k], z[tile_m, tile_k].T, acc=acc3)
        out[tile_n, tile_m] = acc + acc2 + acc3
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def filled_after_a_lagging_body(
    x: torch.Tensor, w: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A body-only root (a chain of dependent updates of ``x``), then ``w``
    times ``x`` (TMA operands, which the MMA partition runs while the body is
    still in the first root), then ``v`` times the first root's output, which
    the body fills each K step."""
    m, k = x.size()
    n = w.size(0)
    y = torch.empty_like(x)
    out1 = torch.empty([n, m], dtype=torch.float32, device=x.device)
    out2 = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        a = x[tile_r, tile_c].to(torch.float32)
        for _ in hl.tile(2048, block_size=1):
            a = a * 0.5 + 1.0
        y[tile_r, tile_c] = a.to(x.dtype)
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w[tile_n, tile_k], x[tile_m, tile_k].T, acc=acc)
        out1[tile_n, tile_m] = acc
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            a = (y[tile_m, tile_k].to(torch.float32) * 2).to(x.dtype)
            acc = hl.dot(v[tile_n, tile_k], a.T, acc=acc)
        out2[tile_n, tile_m] = acc
    return out1, out2


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


@helion.kernel(static_shapes=True, backend="gluon")
def split_k_parts(
    x: torch.Tensor, w4: torch.Tensor, part: torch.Tensor
) -> torch.Tensor:
    """Split K: the tasks of one block of 64 rows write their parts
    (``part``), the next root sums them.  ``w4``: [n / 64, k / 128, 64, 128]."""
    t, k = x.size()
    blocks, steps, rows, block_k = w4.size()
    splits = part.size(0)
    out = torch.empty([t, blocks * rows], dtype=torch.float32, device=x.device)
    out3 = out.view(t, blocks, rows)
    part4 = part.view(splits, t, blocks, rows)
    x3 = x.view(t, steps, block_k)
    k_len = k // splits
    for tile_s, tile_b in hl.tile([splits, blocks], block_size=[1, 1]):
        acc = hl.zeros([rows, t], dtype=torch.float32)
        for tile_k in hl.tile(k_len, block_size=block_k):
            kb = (tile_s.begin * k_len + tile_k.begin) // block_k
            w_k = hl.load(
                w4,
                [
                    tile_b.id + hl.zeros([], dtype=torch.int32),
                    kb + hl.zeros([], dtype=torch.int32),
                    slice(None),
                    slice(None),
                ],
            )
            acc = hl.dot(w_k, x3[:, kb, :].permute(1, 0), acc=acc)
        part4[tile_s.begin, :, tile_b.begin, :] = acc.permute(1, 0)
    for tile_c in hl.tile(blocks, block_size=1):
        out3[:, tile_c.begin, :] = torch.sum(part4[:, :, tile_c.begin, :], dim=0)
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def split_k_halves(
    x: torch.Tensor, w4: torch.Tensor, part: torch.Tensor
) -> torch.Tensor:
    """``split_k_parts`` with each task's K range in two halves, stored apart
    (``part``: [splits, 2, t, n])."""
    t, k = x.size()
    blocks, steps, rows, block_k = w4.size()
    splits = part.size(0)
    out = torch.empty([t, blocks * rows], dtype=torch.float32, device=x.device)
    out3 = out.view(t, blocks, rows)
    part5 = part.view(splits, 2, t, blocks, rows)
    x3 = x.view(t, steps, block_k)
    half = k // splits // 2
    for tile_s, tile_b in hl.tile([splits, blocks], block_size=[1, 1]):
        low = hl.zeros([rows, t], dtype=torch.float32)
        high = hl.zeros([rows, t], dtype=torch.float32)
        for tile_k in hl.tile(half, block_size=block_k):
            kb = (tile_s.begin * 2 * half + tile_k.begin) // block_k
            low = hl.dot(w4[tile_b.begin, kb, :, :], x3[:, kb, :].T, acc=low)
            kh = kb + half // block_k
            high = hl.dot(w4[tile_b.begin, kh, :, :], x3[:, kh, :].T, acc=high)
        part5[tile_s.begin, 0, :, tile_b.begin, :] = low.T
        part5[tile_s.begin, 1, :, tile_b.begin, :] = high.T
    for tile_c in hl.tile(blocks, block_size=1):
        halves = torch.sum(part5[:, :, :, tile_c.begin, :], dim=0)
        out3[:, tile_c.begin, :] = torch.sum(halves, dim=0)
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def crossed_parts(x: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Two parts from one root, transposed: the tasks of one ``c`` write
    ``p[:, c]`` along the first task axis and ``q[:, c]`` along the second."""
    a_n, b_n, n = x.size()
    out = torch.empty([b_n, n], dtype=torch.float32, device=x.device)
    for tile_a, tile_b in hl.tile([a_n, b_n], block_size=[1, 1]):
        v = x[tile_a.begin, tile_b.begin, :].to(torch.float32)
        p[tile_a.begin, tile_b.begin, :] = v
        q[tile_b.begin, tile_a.begin, :] = v * 2
    for tile_c in hl.tile(b_n, block_size=1):
        c = tile_c.begin
        out[c, :] = p[:, c, :].sum(0) + q[:, c, :].sum(0)
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def gather_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, partial: torch.Tensor, gathered: torch.Tensor
) -> torch.Tensor:
    """An RMSNorm over a row's parts: each part's task writes its sum of
    squares (``partial``: [rows, parts]) and its values (``gathered``); each
    part's norm task reads the row's sums and its part's values.
    ``block_sizes``: both roots' part width."""
    rows, hidden = x.size()
    out = torch.empty_like(x)
    for tile_t, tile_n in hl.tile([rows, hidden], block_size=[1, None]):
        values = x[tile_t.begin, tile_n] * 2.0
        partial[tile_t.begin, tile_n.id] = torch.sum(values * values)
        gathered[tile_t.begin, tile_n] = values
    for tile_t, tile_n in hl.tile([rows, hidden], block_size=[1, None]):
        scale = torch.rsqrt(torch.sum(partial[tile_t.begin, :]) / hidden + 1e-6)
        out[tile_t.begin, tile_n] = (
            gathered[tile_t.begin, tile_n] * scale * weight[tile_n]
        )
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def scaled_gather_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    rows_in: torch.Tensor,
    table: torch.Tensor,
    sums: torch.Tensor,
    partial: torch.Tensor,
    gathered: torch.Tensor,
) -> torch.Tensor:
    """``gather_rms_norm`` times a table's total plus one of its row sums: a
    first root writes the table (``table``, a row a task from ``rows_in``)
    and each row's sum (``sums``); every norm task reads the table whole and
    one sum.  ``block_sizes``: both norm roots' part width."""
    rows, hidden = x.size()
    out = torch.empty_like(x)
    for tile_e in hl.tile(table.size(0), block_size=1):
        table[tile_e.begin, :] = rows_in[tile_e.begin, :] * 0.5
        sums[tile_e] = torch.sum(rows_in[tile_e.begin, :])[None]
    for tile_t, tile_n in hl.tile([rows, hidden], block_size=[1, None]):
        values = x[tile_t.begin, tile_n] * 2.0
        partial[tile_t.begin, tile_n.id] = torch.sum(values * values)
        gathered[tile_t.begin, tile_n] = values
    for tile_t, tile_n in hl.tile([rows, hidden], block_size=[1, None]):
        scale = torch.rsqrt(torch.sum(partial[tile_t.begin, :]) / hidden + 1e-6)
        total = (
            torch.sum(torch.sum(table[:, :], dim=1)) + sums[tile_t.begin % sums.size(0)]
        )
        out[tile_t.begin, tile_n] = (
            gathered[tile_t.begin, tile_n] * scale * weight[tile_n]
        ) * total
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def split_k_gemv(x: torch.Tensor, w: torch.Tensor, part: torch.Tensor) -> torch.Tensor:
    """Split K over output tiles of a tuned width, the split the slower task
    axis: each task writes its part (``part``), the next root sums a tile's.
    ``block_sizes``: the parts' tile, their K step and the sum's tile."""
    t, k = x.size()
    n = w.size(0)
    splits = part.size(0)
    out = torch.empty([t, n], dtype=torch.float32, device=x.device)
    k_len = k // splits
    for tile_n, tile_s in hl.tile([n, splits], block_size=[None, 1]):
        acc = hl.zeros([tile_n, t], dtype=torch.float32)
        for tile_k in hl.tile(tile_s.begin * k_len, tile_s.begin * k_len + k_len):
            acc = hl.dot(w[tile_n, tile_k], x[:, tile_k].permute(1, 0), acc=acc)
        part[tile_s.begin, :, tile_n] = acc.permute(1, 0)
    for tile_c in hl.tile(n):
        out[:, tile_c] = part[:, :, tile_c].sum(0)
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def live_mm_scale_mm(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """A matmul and a second matmul over the first ``rows[0]`` rows, between
    them each of the first ``rows[1]`` rows scaled by its sum over every column
    tile (rows past either bound stay zero).  Each root waits on the one before
    key by key (the row tile, every root's slowest axis)."""
    m, k = x.size()
    _, n = y.size()
    _, o = w.size()
    h = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    h_sum = torch.zeros([n // 128, m], dtype=torch.float32, device=x.device)
    scaled = torch.zeros([m, n], dtype=torch.bfloat16, device=x.device)
    out = torch.zeros([m, o], dtype=torch.float32, device=x.device)
    for tile_n, tile_m in hl.tile([n, m], block_size=[128, None]):
        if tile_m.begin < rows[0]:
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
            h[tile_m, tile_n] = acc
            h_sum[tile_n.id, tile_m] = torch.sum(acc, dim=1)
    for tile_c, tile_r in hl.tile([n, m], block_size=[64, None]):
        if tile_r.begin < rows[1]:
            total = torch.sum(h_sum[:, tile_r], dim=0)
            scaled[tile_r, tile_c] = (h[tile_r, tile_c] * total[:, None]).to(
                torch.bfloat16
            )
    for tile_o, tile_q in hl.tile([o, m], block_size=[128, None]):
        if tile_q.begin < rows[0]:
            acc2 = hl.zeros([tile_q, tile_o], dtype=torch.float32)
            for tile_j in hl.tile(n):
                acc2 = hl.dot(scaled[tile_q, tile_j], w[tile_j, tile_o], acc=acc2)
            out[tile_q, tile_o] = acc2
    return out
