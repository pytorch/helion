from __future__ import annotations

import ast
import functools
import itertools
import os
import re
import tempfile
import unittest
from unittest import mock

import torch

from helion.runtime.settings import _get_backend

if _get_backend() != "triton":
    raise unittest.SkipTest("the gluon tests run in the triton lane")

import triton
import triton.language as tl

import helion
from helion._compat import gluon_api_available
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessGluon
from helion._testing.gluon_kernels import filled_after_a_lagging_body
from helion._testing.gluon_kernels import filled_beside_two_dots
from helion._testing.gluon_kernels import gather_rms_norm
from helion._testing.gluon_kernels import live_mm_scale_mm
from helion._testing.gluon_kernels import mm_then_mm
from helion._testing.gluon_kernels import mm_then_scaled_mm
from helion._testing.gluon_kernels import scale_then_chunked_wx
from helion._testing.gluon_kernels import scale_then_nested_wx
from helion._testing.gluon_kernels import scale_then_normed_wx
from helion._testing.gluon_kernels import scale_then_two_wx
from helion._testing.gluon_kernels import scale_then_wx
from helion._testing.gluon_kernels import scale_then_wx_and_f16_vx
from helion._testing.gluon_kernels import scale_then_wx_and_nvfp4
from helion._testing.gluon_kernels import scaled_gather_rms_norm
from helion._testing.gluon_kernels import split_k_gemv
from helion._testing.gluon_kernels import split_k_halves
from helion._testing.gluon_kernels import split_k_parts
from helion._testing.gluon_kernels import wx_then_scaled_vx
from helion._testing.timeline_checks import check_written
from helion._testing.triton_codegen import GLUON_RING_CONFIG as RING
from helion._testing.triton_codegen import gluon_tail_ring_config
import helion.language as hl
from helion.runtime import timeline
from helion.runtime.triton import dist_utils
from helion.runtime.triton.launcher import get_cluster_capacity


@helion.kernel(static_shapes=True, backend="gluon")
def mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
def mm_times_total(
    x: torch.Tensor, y: torch.Tensor, s: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    part = torch.empty_like(s)
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(s.size(0), block_size=16):
        part[tile_s] = s[tile_s] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        # Only the first ``rows`` rows are live: a data-dependent task count.
        if tile_m.begin < rows[0]:
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
            # Every task reads the first root whole: one wait serves all of them.
            out[tile_m, tile_n] = acc * torch.sum(part[:])
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def live_barrier_chain(
    x: torch.Tensor, w: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``x + 1``, the first ``rows`` rows of ``w`` times all of it, then ``x +
    1`` plus the product's total: root barriers chain the roots, the last one's
    read of ``x + 1`` relying on the middle one's barrier, which it publishes
    after waiting on the first one's."""
    m, n = x.size()
    r, _ = w.size()
    act = torch.empty_like(x)
    out = torch.zeros([r, n], dtype=torch.float32, device=x.device)
    res = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        act[tile_m, tile_n] = x[tile_m, tile_n] + 1
    for tile_r in hl.tile(r, block_size=16):
        if tile_r.begin < rows[0]:
            weight = w[tile_r, :]
            out[tile_r, :] = hl.dot(
                weight, act[:, :].to(torch.bfloat16), out_dtype=torch.float32
            )
    for tile_m, tile_n in hl.tile([m, n]):
        res[tile_m, tile_n] = act[tile_m, tile_n] + torch.sum(out[:, :], dim=[0, 1])
    return act, out, res


@helion.kernel(static_shapes=True, backend="gluon")
def scale_both_then_live_mm(
    x: torch.Tensor, w: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """Both dot operands come from earlier roots, and only the first ``rows``
    rows are live: a claimed task loop the weight producer has no step in."""
    m, k = x.size()
    _, n = w.size()
    y = torch.empty_like(x)
    v = torch.empty_like(w)
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_r, tile_c in hl.tile([k, n]):
        v[tile_r, tile_c] = w[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        if tile_m.begin < rows[0]:
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = hl.dot(y[tile_m, tile_k], v[tile_k, tile_n], acc=acc)
            out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def mm_pair(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Two dots of one shape in one K loop (a fused gate/up projection)."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        a = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        b = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            a = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=a)
            b = hl.dot(x[tile_m, tile_k], z[tile_k, tile_n], acc=b)
        out[tile_m, tile_n] = a * 2 - b
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def mm_both(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two roots, each with one dot of the same shape."""
    m, k = x.size()
    _, n = y.size()
    out_y = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out_z = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out_y[tile_m, tile_n] = acc
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], z[tile_k, tile_n], acc=acc)
        out_z[tile_m, tile_n] = acc
    return out_y, out_z


@helion.kernel(static_shapes=True, backend="gluon")
def mm_two_loops(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """One root whose task runs two K loops, each into its own same-shape accumulator."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        a = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            a = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=a)
        b = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            b = hl.dot(x[tile_m, tile_k], z[tile_k, tile_n], acc=b)
        out[tile_m, tile_n] = a * 2 - b
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def mm_plus_row_sums(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A dot and, after it on the same ring channel, a register read."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        sums = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
            sums = sums + torch.sum(w[tile_m, tile_k].float(), -1)
        out[tile_m, tile_n] = acc + sums[:, None]
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def mm_cast(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A dot of a ring slot and a register operand cast from another."""
    m, k = x.size()
    _, n = w.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], w[tile_k, tile_n].to(x.dtype), acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def mm_cast_plus_row_sums(
    x: torch.Tensor, w: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``mm_cast`` with a register read's step after the dot's on its channel."""
    m, k = x.size()
    _, n = w.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        sums = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], w[tile_k, tile_n].to(x.dtype), acc=acc)
            sums = sums + torch.sum(v[tile_m, tile_k].float(), -1)
        out[tile_m, tile_n] = acc + sums[:, None]
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def scaled_mm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[128, 64]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for kk in hl.grid(k // 128):
            acc = hl.dot_scaled(
                x[tile_m, kk * 128 + hl.arange(128)],
                x_scale[tile_m, kk * 4 + hl.arange(4)],
                "e4m3",
                y[kk * 128 + hl.arange(128), tile_n],
                y_scale[tile_n, kk * 4 + hl.arange(4)],
                "e4m3",
                acc=acc,
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def fp4_mm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """mxfp4 operands packed along K, y's tile N-contiguous (MN-major)."""
    m, packed = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[128, 128]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for kk in hl.grid(packed // 64):
            acc = hl.dot_scaled(
                x[tile_m, kk * 64 + hl.arange(64)],
                x_scale[tile_m, kk * 4 + hl.arange(4)],
                "e2m1",
                y[kk * 64 + hl.arange(64), tile_n],
                y_scale[tile_n, kk * 4 + hl.arange(4)],
                "e2m1",
                acc=acc,
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def fp4_rows_by_tokens(
    w: torch.Tensor, w_scale: torch.Tensor, x: torch.Tensor, x_scale: torch.Tensor
) -> torch.Tensor:
    """nvfp4 weight rows times 16 tokens in 512-wide K steps: each 128-row tile's
    step scales as stored for tcgen05.cp (``w_scale[tile, step]``, [16, 256]
    viewed [128, 32]), the tokens' as loaded ([16, 32], through registers)."""
    n, packed = w.size()
    tokens = x.size(0)
    steps = packed // 256
    w_steps = w.view(n, steps, 256)
    x_steps = x.view(tokens, steps, 256)
    x_scale_steps = x_scale.view(tokens, steps, 32)
    out = torch.empty([n, tokens], dtype=torch.float32, device=w.device)
    for tile in hl.tile(n // 128, block_size=1):
        acc = hl.zeros([128, tokens], dtype=torch.float32)
        for step in hl.tile(steps, block_size=1):
            acc = hl.dot_scaled(
                w_steps[tile.begin * 128 + hl.arange(128), step.begin, :],
                w_scale[tile.begin, step.begin, :, :]
                .reshape(8, 32, 4, 4)
                .permute(2, 1, 0, 3)
                .reshape(128, 32),
                "e2m1",
                x_steps[:, step.begin, :].T,
                x_scale_steps[:, step.begin, :],
                "e2m1",
                acc=acc,
            )
        out[tile.begin * 128 + hl.arange(128), :] = acc
    return out


@functools.cache
def _fp4_rows_by_tokens_reference() -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """``fp4_rows_by_tokens`` inputs in 12 K steps, with distinct scales per step
    and tile, and the one-stage ring's output every deeper ring must reproduce
    exactly."""
    e4m3 = torch.float8_e4m3fn
    n, steps = 768, 12
    w = torch.randint(0, 256, [n, steps * 256], dtype=torch.uint8, device=DEVICE)
    x = torch.randint(0, 256, [16, steps * 256], dtype=torch.uint8, device=DEVICE)
    w_scale = torch.rand([n // 128, steps, 16, 256], device=DEVICE) + 0.5
    x_scale = torch.rand([16, steps * 32], device=DEVICE) + 0.5
    args = (w, w_scale.to(e4m3), x, x_scale.to(e4m3))
    _, expected = code_and_output(fp4_rows_by_tokens, args, **{**RING, "num_stages": 1})
    return args, expected


@helion.kernel(static_shapes=True, backend="gluon")
def fp4_mm_nt(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    y: torch.Tensor,
    y_scale: torch.Tensor,
    block_n: hl.constexpr,
) -> torch.Tensor:
    """mxfp4 with both operands K-major; ``block_n`` columns per tile."""
    m, packed = x.size()
    n = y.size(0)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[128, block_n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for kk in hl.grid(packed // 64):
            acc = hl.dot_scaled(
                x[tile_m, kk * 64 + hl.arange(64)],
                x_scale[tile_m, kk * 4 + hl.arange(4)],
                "e2m1",
                y[tile_n, kk * 64 + hl.arange(64)].T,
                y_scale[tile_n, kk * 4 + hl.arange(4)],
                "e2m1",
                acc=acc,
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def fp4_broadcast_mv(
    w: torch.Tensor, w_scale: torch.Tensor, a: torch.Tensor, a_scale: torch.Tensor
) -> torch.Tensor:
    """A weight-times-vector MMA whose B is one fp4 row broadcast to 16 columns."""
    m, packed = w.size()
    out = torch.empty([m, 16], dtype=torch.float32, device=w.device)
    for tile_m in hl.tile(m, block_size=256):
        acc = hl.zeros([tile_m, 16], dtype=torch.float32)
        for tile_g in hl.tile(packed // 8, block_size=32):
            index = tile_g.begin * 8 + hl.arange(256)
            acc = hl.dot_scaled(
                w[tile_m, index],
                w_scale[tile_m, tile_g],
                "e2m1",
                a[index][:, None].expand(256, 16),
                a_scale[tile_g][None, :].expand(16, 32),
                "e2m1",
                acc=acc,
            )
        out[tile_m, :] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def scale_then_mm(
    x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = w.size()
    y = torch.empty_like(x)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    # The next launch starts at once and waits for this one at its entry (its
    # code prefetch ahead of the wait).
    hl.prefetch_code()
    hl.pdl_wait()
    hl.pdl_launch_dependents()
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(y[tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return y, out


@helion.kernel(static_shapes=True, backend="gluon")
def mm_then_scale(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A dot, then a root of its output: the next launch starts once a worker
    reaches the second root."""
    m, k = x.size()
    _, n = w.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    y = torch.empty([m, n], dtype=torch.float32, device=x.device)
    hl.pdl_wait()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    hl.pdl_launch_dependents()
    for tile_r, tile_c in hl.tile([m, n]):
        y[tile_r, tile_c] = out[tile_r, tile_c] * 2 + b[tile_r, tile_c]
    return out, y


@helion.kernel(static_shapes=True, backend="gluon")
def mm_prefetch_mm(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two dot roots, a code prefetch at entry and one between them."""
    m, k = x.size()
    _, n = w1.size()
    _, p = w2.size()
    y = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    out = torch.empty([m, p], dtype=torch.float32, device=x.device)
    hl.prefetch_code()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], w1[tile_k, tile_n], acc=acc)
        y[tile_m, tile_n] = acc.to(torch.bfloat16)
    hl.prefetch_code()
    for tile_i, tile_p in hl.tile([m, p]):
        acc2 = hl.zeros([tile_i, tile_p], dtype=torch.float32)
        for tile_j in hl.tile(n):
            acc2 = hl.dot(y[tile_i, tile_j], w2[tile_j, tile_p], acc=acc2)
        out[tile_i, tile_p] = acc2
    return y, out


@helion.kernel(static_shapes=True, backend="gluon")
def scale_boxprefetch_mm(
    x: torch.Tensor, w: torch.Tensor, wp: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scale_then_mm`` with a root between the two that only warms the L2
    for the dot's weights, as TMA boxes of ``wp`` (``w`` viewed as 64-row
    blocks): one tensor-form prefetch per block."""
    m, k = x.size()
    _, n = w.size()
    y = torch.empty_like(x)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_p in hl.tile(wp.size(0), block_size=1):
        hl.prefetch(wp, [tile_p.begin, slice(None), slice(None)])
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(y[tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return y, out


@helion.kernel(static_shapes=True, backend="gluon")
def viewprefetch_copy(w: torch.Tensor) -> torch.Tensor:
    """Prefetch ``w`` as 64-row boxes and copy it through a kernel-code view:
    the copy's descriptor needs a zero storage offset, the prefetch only an
    aligned base."""
    k, n = w.size()
    wp = w.view(-1, 64, n)
    wv = w.view(k, n)
    out = torch.empty_like(w)
    for tile_p in hl.tile(wp.size(0), block_size=1):
        hl.prefetch(wp, [tile_p.begin, slice(None), slice(None)])
    for tile_k, tile_n in hl.tile([k, n]):
        out[tile_k, tile_n] = wv[tile_k, tile_n]
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def boxprefetch_copy(w: torch.Tensor, wp: torch.Tensor) -> torch.Tensor:
    """Prefetch ``wp`` (a view of ``w``) one leading index at a time, then copy
    ``w``: how a prefetch lowers, without a ring or a dot."""
    out = torch.empty_like(w)
    for tile_p in hl.tile(wp.size(0), block_size=1):
        hl.prefetch(wp, [tile_p.begin, slice(None), slice(None)])
    for tile_k, tile_n in hl.tile(w.size()):
        out[tile_k, tile_n] = w[tile_k, tile_n]
    return out


@helion.kernel(static_shapes=False, backend="gluon")
def dynboxprefetch_copy(w: torch.Tensor, wp: torch.Tensor) -> torch.Tensor:
    """``boxprefetch_copy`` with dynamic shapes, every size specialized but not
    the prefetched tensor's strides."""
    hl.specialize(w.size())
    hl.specialize(wp.size())
    out = torch.empty_like(w)
    for tile_p in hl.tile(wp.size(0), block_size=1):
        hl.prefetch(wp, [tile_p.begin, slice(None), slice(None)])
    for tile_k, tile_n in hl.tile(w.size()):
        out[tile_k, tile_n] = w[tile_k, tile_n]
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def halfboxprefetch_copy(w: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
    """``boxprefetch_copy`` prefetching each 64-row block of ``w`` as two boxes
    of column halves (``wh``: ``w`` viewed as blocks x rows x halves x
    columns), a scalar index between the spanned dims."""
    out = torch.empty_like(w)
    for tile_p, tile_h in hl.tile([wh.size(0), wh.size(2)], block_size=[1, 1]):
        hl.prefetch(wh, [tile_p.begin, slice(None), tile_h.begin, slice(None)])
    for tile_k, tile_n in hl.tile(w.size()):
        out[tile_k, tile_n] = w[tile_k, tile_n]
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def scale_idprefetch_mm(
    x: torch.Tensor, w: torch.Tensor, wp: torch.Tensor, ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scale_boxprefetch_mm`` prefetching the blocks of ``wp`` that ``ids``
    lists (-1: none), as an expert prefetch reads its routing."""
    m, k = x.size()
    _, n = w.size()
    y = torch.empty_like(x)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_p in hl.tile(ids.size(0), block_size=1):
        block = ids[tile_p.begin]
        if block >= 0:
            hl.prefetch(wp, [block, slice(None), slice(None)])
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(y[tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return y, out


# The config every tensor-form prefetch test runs (``scale_then_mm`` shapes).
PREFETCH_CONFIG = {
    "block_sizes": [128, 64, 128, 64, 64],
    "pid_type": "persistent_blocked",
    "num_sm_multiplier": 1,
    "num_warps": 4,
    "indexing": "tensor_descriptor",
    "prefetch_indexing": "tensor_descriptor",
}


@functools.cache
def _prefetch_reference() -> tuple[torch.Tensor, ...]:
    """Inputs and the static-pipeline ``scale_then_mm`` outputs every
    prefetching variant must reproduce exactly (a prefetch changes no value)."""
    x = torch.randn([256, 1024], device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
    _, (y, out) = code_and_output(
        scale_then_mm, (x, w), cross_loop_pipeline="static", **PREFETCH_CONFIG
    )
    return x, w, y, out


@helion.kernel(static_shapes=True, backend="gluon")
def route_then_mm(
    x: torch.Tensor, w: torch.Tensor, perm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = w.size()
    order = torch.empty_like(perm)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    hl.pdl_wait()
    hl.pdl_launch_dependents()
    for tile_b in hl.tile(perm.size(0)):
        order[tile_b] = perm[tile_b]
    for tile_m, tile_n in hl.tile([m, n], block_size=[128, 64]):
        # The rows come from what the first root wrote (a router's choice).
        rows = order[tile_m.begin // 128] * 128
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[rows + hl.arange(128), tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out, order


@triton.jit
def _spin_until_set(flag):
    while tl.atomic_add(flag, 0, sem="acquire") == 0:
        pass


@helion.kernel(static_shapes=True, backend="gluon")
def mm_after_flag(x: torch.Tensor, y: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        # A user wait: x may be written by another kernel until flag is set.
        hl.triton_kernel(_spin_until_set, args=(flag,), output_like=None)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def scale_then_w13x(
    x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two weights of one shape read in one K loop (a gate/up pair)."""
    m, k = x.size()
    n = w1.size(0)
    y = torch.empty_like(x)
    out1 = torch.empty([n, m], dtype=torch.float32, device=x.device)
    out3 = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        acc1 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        acc3 = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc1 = hl.dot(w1[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc1)
            acc3 = hl.dot(w3[tile_n, tile_k], y[tile_m, tile_k].T, acc=acc3)
        out1[tile_n, tile_m] = acc1
        out3[tile_n, tile_m] = acc3
    return out1, out3


@helion.kernel(static_shapes=True, backend="gluon")
def scale_then_ewx(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``scale_then_wx`` with the second weight of a stack ([e, n, k])."""
    m, k = x.size()
    n = w.size(1)
    y = torch.empty_like(x)
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w[1, tile_n, tile_k], y[tile_m, tile_k].T, acc=acc)
        out[tile_n, tile_m] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def scale_by_last_row(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            last = x[x.size(0) - 1, tile_n]
            out[tile_m, tile_n] = x[tile_m, tile_n] * last[None, :]
    return out


def _replay(run: object, inputs: list[tuple[torch.Tensor, ...]]) -> list[object]:
    """``run`` on each input back to back in a CUDA graph, so that each launch
    overlaps the last (PDL)."""
    run(*inputs[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        results = list(itertools.starmap(run, inputs))
    graph.replay()
    return results


def _live_mm_scale_mm_ref(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """``live_mm_scale_mm`` in torch (bounds whole row tiles)."""
    m, n = x.size(0), y.size(1)
    live, scale_live = (min(max(int(b), 0), m) for b in rows.tolist())
    h = torch.zeros([m, n], device=x.device)
    h[:live] = x[:live].float() @ y.float()
    total = h.view(m, n // 128, 128).sum(-1).sum(-1)
    scaled = torch.zeros([m, n], device=x.device, dtype=torch.bfloat16)
    scaled[:scale_live] = (h[:scale_live] * total[:scale_live, None]).bfloat16()
    out = torch.zeros([m, w.size(1)], device=x.device)
    out[:live] = scaled[:live].float() @ w.float()
    return out


def _group_tickets(
    live: list[int], tasks: list[int], lags: list[int], tickets: int
) -> list[tuple[int, int] | None]:
    """``_group_ticket``'s (root, task) of each ticket, by enumerating rounds."""
    keys = [n // a for n, a in zip(live, tasks, strict=True)]
    order: list[tuple[int, int]] = []
    for r in range(max(o + k for o, k in zip(lags, keys, strict=True))):
        for root, (a, o, k) in enumerate(zip(tasks, lags, keys, strict=True)):
            if 0 <= r - o < k:
                order.extend((root, (r - o) * a + u) for u in range(a))
    return [order[t] if t < len(order) else None for t in range(tickets)]


def _cluster_workers(splits: int) -> int:
    """The workers a persistent launch in clusters of ``splits`` takes on
    this device: the count the generated host code asks ``get_num_sm`` for."""
    return helion.runtime.get_num_sm(DEVICE, cluster_size=splits)


def _one_wave(splits: int, most: int, test: TestCase) -> int:
    """At most ``most`` keys of ``splits`` producer tasks that run in one wave
    of clusters on this device (skipping ``test`` where no cluster fits)."""
    if get_cluster_capacity(DEVICE, splits) < splits:
        test.skipTest(f"no whole cluster of {splits} on this device")
    return min(most, _cluster_workers(splits) // splits)


def _functions(code: str) -> dict[str, str]:
    """The generated module's functions, by name."""
    return {chunk.split("(", 1)[0]: chunk for chunk in code.split("\ndef ")[1:]}


if gluon_api_available():
    # (gluon_ops needs the Gluon API, which each test checks for.)
    from helion._compiler.triton import gluon_ops

    _CLUSTER_VALUE_ASM = tl.constexpr(
        "{ .reg .b32 t, u, w; .reg .b64 v, s, c; "
        + gluon_ops._CLUSTER_VALUE
        + "mov.b64 $0, v; }"
    )

    @triton.jit
    def _cluster_values(epochs, states, values, n, BLOCK: tl.constexpr):
        """``gluon_ops``' cluster flag value of each epoch and launch state (in a
        launch without clusters: cluster 0)."""
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n
        epoch = tl.load(epochs + offsets, mask=mask).to(tl.uint32)
        state = tl.load(states + offsets, mask=mask)
        value = tl.inline_asm_elementwise(
            _CLUSTER_VALUE_ASM,
            "=l,r,l",
            [epoch, state],
            dtype=tl.int64,
            is_pure=True,
            pack=1,
        )
        tl.store(values + offsets, value, mask=mask)


class TestGluonRing(TestCase):
    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_trigger_between_roots_in_a_ring(self) -> None:
        """The trigger between the loops runs in the body before the second
        root's tasks, after its own wait. Back-to-back launches (PDL) stay
        exact."""
        x = torch.randn([256, 1024], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        base = {
            "block_sizes": [128, 64, 64, 128, 64],
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 1,
            "num_warps": 4,
            "indexing": "tensor_descriptor",
        }
        bound = mm_then_scale.bind((x, w, b))
        slots = len(bound.config_spec.range_warp_specialize)
        ring = {
            **base,
            "num_stages": 2,
            "cross_loop_pipeline": "static",
            "range_warp_specializes": [None] * (slots - 1) + [True],
        }
        code, results = code_and_output(mm_then_scale, (x, w, b), **ring)
        static = bound.compile_config(
            helion.Config(**base, cross_loop_pipeline="static")
        )
        for result, expected in zip(results, static(x, w, b), strict=True):
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
        functions = _functions(code)
        # Only the body triggers.
        for name, function in functions.items():
            if "_producer" in name:
                self.assertNotIn("gdc_launch_dependents", function)
        body = functions["_helion_mm_then_scale_body"]
        trigger = body.index("gdc_launch_dependents()")
        self.assertLess(body.index("_launch("), trigger)
        self.assertLess(body.index("x_ring_desc, ("), trigger)
        self.assertLess(trigger, body.index("b_ring_desc, ("))
        self.assertEqual(code.count("gdc_launch_dependents()"), 1)
        inputs = [(x + launch, w, b) for launch in range(4)]
        replayed = _replay(bound.compile_config(helion.Config(**ring)), inputs)
        for got, args in zip(replayed, inputs, strict=True):
            for result, expected in zip(got, static(*args), strict=True):
                torch.testing.assert_close(result, expected, atol=0, rtol=0)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_dot(self) -> None:
        x = torch.randn([256, 512], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([512, 128], device=DEVICE, dtype=torch.bfloat16)
        for specialize in (None, True):
            with self.subTest(range_warp_specializes=specialize):
                code, out = code_and_output(
                    mm,
                    (x, y),
                    block_sizes=[128, 64, 64],
                    range_warp_specializes=[None, specialize],
                    **RING,
                )
                expected = x.float() @ y.float()
                torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
                self.assertIn("_helion_ring.ring_put(", code)
                self.assertIn("_helion_gluon.tmem_dot(", code)
                # A warp-specialized K loop issues its dot from the MMA
                # partition, into an accumulator ring the body takes.
                self.assertEqual("def _helion_mm_mma(" in code, bool(specialize))
                self.assertEqual("_helion_gluon.acc_take(" in code, bool(specialize))
                self.assertNotIn("warp_specialize=", code)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_two_dots_in_one_loop(self) -> None:
        """Each MMA-partition dot of a loop gets its own accumulator ring and
        starts it fresh at its own first MMA."""
        x = torch.randn([512, 1024], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        z = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        expected = 2 * (x.float() @ y.float()) - x.float() @ z.float()
        for specialize in (None, True):
            with self.subTest(range_warp_specializes=specialize):
                code, out = code_and_output(
                    mm_pair,
                    (x, y, z),
                    block_sizes=[128, 128, 64],
                    range_warp_specializes=[None, specialize],
                    **RING,
                )
                torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
                self.assertEqual(
                    code.count("= _helion_gluon.acc_buffers("), 2 if specialize else 0
                )

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_same_shape_dots_share_a_ring(self) -> None:
        """Same-shape dots of different loops share one accumulator ring."""
        x = torch.randn([512, 1024], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        z = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        code, (out_y, out_z) = code_and_output(
            mm_both,
            (x, y, z),
            block_sizes=[128, 128, 64, 128, 128, 64],
            range_warp_specializes=[None, True, None, True],
            **RING,
        )
        torch.testing.assert_close(out_y, x.float() @ y.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out_z, x.float() @ z.float(), atol=1e-2, rtol=1e-2)
        self.assertEqual(code.count("= _helion_gluon.acc_buffers("), 1)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_task_accumulators_get_their_own_rings(self) -> None:
        """Same-shape dots of two loops of one task take one accumulator ring
        each, so the MMA runs a whole task ahead of the body (sharing one
        double-buffered ring would hold it to the task the body is in)."""
        x = torch.randn([512, 1024], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        z = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            mm_two_loops,
            (x, y, z),
            block_sizes=[128, 128, 64, 64],
            range_warp_specializes=[True, None, None],
            **RING,
        )
        torch.testing.assert_close(
            out,
            2 * (x.float() @ y.float()) - x.float() @ z.float(),
            atol=2e-2,
            rtol=1e-2,
        )
        self.assertEqual(code.count("= _helion_gluon.acc_buffers("), 2)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_mma_and_body_dots_get_their_own_channels(self) -> None:
        """An MMA-partition dot and a body dot of the same gating group each
        get a channel: each consumer counts and waits for its own steps."""
        x = torch.randn([512, 1024], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        z = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        code, (out_y, out_z) = code_and_output(
            mm_both,
            (x, y, z),
            block_sizes=[128, 128, 64, 128, 128, 64],
            range_warp_specializes=[None, True, None, None],
            **RING,
        )
        torch.testing.assert_close(out_y, x.float() @ y.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out_z, x.float() @ z.float(), atol=1e-2, rtol=1e-2)
        self.assertIn("def _helion_mm_both_mma(", code)
        # Two ungated channels: their full and empty barriers.
        self.assertEqual(code.count("= _helion_ring.ring_barriers(_helion_ring_S"), 4)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_dot_then_register_read(self) -> None:
        """The channel's last step is the register read's, so the body reads the
        dot's accumulator after its own commit of the MMAs."""
        x = torch.randn([512, 1024], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([512, 1024], device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            mm_plus_row_sums, (x, y, w), block_sizes=[128, 128, 64], **RING
        )
        expected = x.float() @ y.float() + w.float().sum(-1, keepdim=True)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        self.assertIn("_helion_ring.ring_read(", code)
        self.assertRegex(code, r"_helion_gluon\.acc_read\(\w+, \w+, (True|False)\)")

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_register_channel_gives_up_slots_to_fit(self) -> None:
        """A channel a dot reads streams at num_stages, whichever partition issues
        the dot; one the body only reads into registers streams at num_stages too
        unless the ring would outgrow the shared memory (less an fp32
        accumulator tile Triton stages beside it), then gives up slots."""
        x = torch.randn([512, 1024], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([512, 1024], device=DEVICE, dtype=torch.bfloat16)
        # The body issues the dot: its register read shares the dot's channel.
        # Warp-specialized, the MMA partition issues it on a channel of its own:
        # 32 KB slots, 64 KB of accumulator staging beside the ring; a BF16
        # [128, 64] read (16 KB slots) fits two, an FP8 one (8 KB) all four.
        for dtype, specialize, want in (
            (torch.bfloat16, None, [4]),
            (torch.bfloat16, True, [2, 4]),
            (torch.float8_e4m3fn, True, [4, 4]),
        ):
            with self.subTest(dtype=dtype, range_warp_specializes=specialize):
                wd = w.to(dtype)
                expected = x.float() @ y.float() + wd.float().sum(-1, keepdim=True)
                code, out = code_and_output(
                    mm_plus_row_sums,
                    (x, y, wd),
                    block_sizes=[128, 128, 64],
                    range_warp_specializes=[None, specialize],
                    **{**RING, "num_stages": 4},
                )
                torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
                depths = {
                    int(c): int(d)
                    for c, d in re.findall(
                        r"_helion_ring_S(\d) = tl.constexpr\((\d+)\)", code
                    )
                }
                self.assertEqual(sorted(depths.values()), want)
                (read,) = set(re.findall(r"ring_read\([^)]*_helion_ring_S(\d),", code))
                self.assertEqual(depths[int(read)], min(want))

    @skipUnlessGluon()
    @skipIfRefEager("checks generated code")
    def test_small_box_slots_stay_aligned(self) -> None:
        """A TMA copy lands 128-byte aligned: a box of 64 bytes alone in its lane
        takes two boxes a slot (each step's copy aligned)."""
        x = torch.randn([64, 256], device=DEVICE)
        code, out = code_and_output(
            scale_by_last_row,
            (x,),
            block_sizes=[16, 16],
            indexing=["tensor_descriptor", "pointer", "pointer"],
            num_stages=3,
        )
        self.assertIn("_helion_ring.ring_read(", code)
        torch.testing.assert_close(out, x * x[-1])

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_unresident_grid_goes_without_the_ring(self) -> None:
        """More CTAs a SM than its thread slots hold cannot share its register
        file: the kernel goes without the ring (the int4 autotune candidates of 64
        CTAs a SM had ``maxnreg=0``)."""
        x = torch.randn([256, 512], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([512, 128], device=DEVICE, dtype=torch.float16)
        code, out = code_and_output(
            mm_cast,
            (x, w),
            block_sizes=[128, 64, 64],
            **{**RING, "num_sm_multiplier": 64},
        )
        self.assertNotIn("_helion_ring.", code)
        expected = x.float() @ w.to(torch.bfloat16).float()
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_register_operand_joins_its_step(self) -> None:
        """A register operand of the open step's dot streams as a site of that
        step (its dot frees it), the MMA restaging it waited for only by the next
        restage, on the dot's own barrier (a register read's step between them
        orders no MMA); two CTAs a SM split the register file."""
        x, v = (
            torch.randn([256, 512], device=DEVICE, dtype=torch.bfloat16) for _ in "xv"
        )
        w = torch.randn([512, 128], device=DEVICE, dtype=torch.float16)
        expected = x.float() @ w.to(torch.bfloat16).float()
        for ctas, sums in ((1, False), (2, False), (1, True)):
            with self.subTest(num_sm_multiplier=ctas, row_sums=sums):
                code, out = code_and_output(
                    mm_cast_plus_row_sums if sums else mm_cast,
                    (x, w, v) if sums else (x, w),
                    block_sizes=[128, 64, 64],
                    **{**RING, "num_sm_multiplier": ctas},
                )
                rows = v.float().sum(-1, keepdim=True) if sums else 0
                torch.testing.assert_close(out, expected + rows, atol=1e-2, rtol=1e-2)
                self.assertRegex(code, r"_helion_ring\.ring_read\([^\n]*, False\)")
                self.assertRegex(code, r"stage\([^()]*, tmem_mma, tmem_count, 1\)")
                self.assertRegex(code, r"tmem_dot\([^\n]*, tmem_mma\)")
                self.assertEqual("native_kernel(maxnreg=" in code, ctas > 1)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_filled_operand(self) -> None:
        """An activation the body computes each K step (the first root's output
        times a vector, in the weight's dtype) fills its slot of the dot's step:
        the dot issues from the MMA partition, bitwise as the body's MMA of the
        staged activation (the loop not warp-specialized), across overlapping
        (PDL) launches."""
        x = torch.randn([64, 1024], device=DEVICE, dtype=torch.bfloat16)
        g = torch.rand([1024], device=DEVICE, dtype=torch.bfloat16) + 0.5
        for dtype in (torch.float8_e4m3fn, torch.bfloat16):
            with self.subTest(dtype=dtype):
                w = torch.randn([256, 1024], device=DEVICE).to(dtype)
                bound = scale_then_normed_wx.bind((x, g, w))
                ring = gluon_tail_ring_config(bound)
                code, out = code_and_output(scale_then_normed_wx, (x, g, w), **ring)
                self.assertIn("_helion_ring.ring_fill(", code)
                self.assertIn("def _helion_scale_then_normed_wx_mma(", code)
                self.assertNotIn("_helion_gluon.stage(", code)
                flags = [None] * len(ring["range_warp_specializes"])
                staged = bound.compile_config(
                    helion.Config(**{**ring, "range_warp_specializes": flags})
                )
                torch.testing.assert_close(out, staged(x, g, w), atol=0, rtol=0)
                a = (2 * x.float() * g.float()).to(dtype).float()
                torch.testing.assert_close(out, w.float() @ a.T, atol=1e-2, rtol=1e-3)
                inputs = [(x + launch, g, w) for launch in range(4)]
                filled = bound.compile_config(helion.Config(**ring))
                for got, args in zip(_replay(filled, inputs), inputs, strict=True):
                    torch.testing.assert_close(got, staged(*args), atol=0, rtol=0)
        # A filled group no producer streams: the weight producer arrives for it.
        w = torch.randn([256, 1024], device=DEVICE, dtype=torch.bfloat16)
        bound = wx_then_scaled_vx.bind((x, w, w))
        slots = len(bound.config_spec.range_warp_specialize)
        config = {**RING, "block_sizes": [128, 32, 64, 64]}
        code, out = code_and_output(
            wx_then_scaled_vx,
            (x, w, w),
            range_warp_specializes=[True] * slots,
            **config,
        )
        self.assertIn("_helion_ring.ring_fill(", code)
        self.assertNotIn("def _helion_wx_then_scaled_vx_producer1(", code)
        staged = bound.compile_config(helion.Config(**config))(x, w, w)
        torch.testing.assert_close(out, staged, atol=0, rtol=0)
        # Barrier mode (its phase loops share the roots' statements): one
        # arrival a step for the filled group.
        x = torch.randn([128, 1024], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([1024, 128], device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        base = {**RING, "block_sizes": [128, 128, 64, 128, 64, 64], "num_warps": 4}
        _, expected = code_and_output(
            mm_then_scaled_mm, (x, w, v), cross_loop_pipeline="static", **base
        )
        slots = len(mm_then_scaled_mm.bind((x, w, v)).config_spec.range_warp_specialize)
        code, out = code_and_output(
            mm_then_scaled_mm,
            (x, w, v),
            range_warp_specializes=[True] * slots,
            cross_loop_pipeline="barrier",
            **base,
        )
        self.assertIn("_helion_ring.ring_fill(", code)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_filled_operand_between_other_steps(self) -> None:
        """Two plain dots step the MMA channel between the body's fills, at a
        depth of two: the body waits for the slot of each step it passes
        (``ring_pass``), so it stays within a phase of the MMA's frees; without
        those waits its fill's parity wait aliases and it overwrites a slot the
        MMA still reads."""
        x, y, z = (
            torch.randn([64, 4096], device=DEVICE, dtype=torch.bfloat16)
            for _ in range(3)
        )
        v, w, u = (
            torch.randn([1024, 4096], device=DEVICE, dtype=torch.bfloat16)
            for _ in range(3)
        )
        args = (x, y, z, v, w, u)
        bound = filled_beside_two_dots.bind(args)
        slots = len(bound.config_spec.range_warp_specialize)
        config = {**RING, "block_sizes": [128, 32, 64], "num_stages": 2}
        code, out = code_and_output(
            filled_beside_two_dots,
            args,
            range_warp_specializes=[True] * slots,
            **config,
        )
        self.assertEqual(code.count("_helion_ring.ring_fill("), 1)
        self.assertEqual(code.count("_helion_ring.ring_pass("), 2)
        staged = bound.compile_config(helion.Config(**config))(*args)
        torch.testing.assert_close(out, staged, atol=0, rtol=0)
        filled = bound.compile_config(
            helion.Config(**config, range_warp_specializes=[True] * slots)
        )
        for _ in range(20):
            torch.testing.assert_close(filled(*args), staged, atol=0, rtol=0)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_filled_operand_after_a_lagging_body(self) -> None:
        """The MMA runs a root of TMA steps while the body is still in a
        body-only root before it, then a root the body fills: the body passes
        the earlier root's steps a lap or more behind the MMA's frees unless
        it arrives for them (``ring_pass``); with only the producer arriving
        there, its parity waits alias and the launch hangs."""
        # 256 tasks a root: each CTA runs tasks of all three.
        x = torch.randn([128, 1024], device=DEVICE, dtype=torch.bfloat16)
        w, v = (
            torch.randn([16384, 1024], device=DEVICE, dtype=torch.bfloat16)
            for _ in range(2)
        )
        args = (x, w, v)
        bound = filled_after_a_lagging_body.bind(args)
        slots = len(bound.config_spec.range_warp_specialize)
        config = {**RING, "block_sizes": [8, 64, 128, 64, 64, 128, 64, 64]}
        code, out = code_and_output(
            filled_after_a_lagging_body,
            args,
            range_warp_specializes=[True] * slots,
            **config,
        )
        self.assertIn("_helion_ring.ring_fill(", code)
        self.assertRegex(code, r"_helion_ring\.ring_pass\([^\n]*, 1\)")
        staged = bound.compile_config(helion.Config(**config))(*args)
        for got, expected in zip(out, staged, strict=True):
            torch.testing.assert_close(got, expected, atol=0, rtol=0)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_dot_scaled(self) -> None:
        x = (torch.randn([256, 512], device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        y = (torch.randn([512, 128], device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        x_scale = torch.randint(124, 130, [256, 16], device=DEVICE, dtype=torch.uint8)
        y_scale = torch.randint(124, 130, [128, 16], device=DEVICE, dtype=torch.uint8)
        code, out = code_and_output(
            scaled_mm,
            (x, x_scale, y, y_scale),
            range_warp_specializes=[None, True],
            **RING,
        )
        factor_x = torch.exp2(x_scale.float() - 127).repeat_interleave(32, dim=1)
        factor_y = torch.exp2(y_scale.float() - 127).repeat_interleave(32, dim=1)
        expected = (x.float() * factor_x) @ (y.float() * factor_y.T)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        # Operands ride the ring; 4-byte scale rows are below a TMA box, so the
        # scales reach the dot from registers and the body issues it.
        self.assertEqual(code.count("_helion_ring.ring_acquire("), 2)
        # Both operands share the step's full barrier: only the first waits.
        self.assertEqual(code.count("WAIT=False"), 1)
        self.assertIn("_helion_gluon.tmem_dot_scaled(", code)
        self.assertNotIn("_mma(", code)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_dot_scaled_register_operands(self) -> None:
        """A register B (a broadcast) is staged K-major for the block-scaled MMA."""
        e4m3 = torch.float8_e4m3fn
        w = torch.randint(0, 256, [512, 1024], dtype=torch.uint8, device=DEVICE)
        w_scale = (torch.rand([512, 128], device=DEVICE) + 0.5).to(e4m3)
        a = torch.randint(0, 256, [1024], dtype=torch.uint8, device=DEVICE)
        a_scale = (torch.rand([128], device=DEVICE) + 0.5).to(e4m3)
        args = (w, w_scale, a, a_scale)
        # The weight scales are a register gather.
        indexing = ["tensor_descriptor", "pointer", *["tensor_descriptor"] * 3]
        config = {**RING, "indexing": indexing}
        _, expected = code_and_output(
            fp4_broadcast_mv, args, **{**config, "num_stages": 1}
        )
        code, out = code_and_output(fp4_broadcast_mv, args, **config)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        self.assertEqual(code.count("_helion_ring.ring_acquire("), 1)
        # B reaches Gluon as its unbroadcast source, staged with K contiguous.
        self.assertRegex(code, r"_helion_gluon\.stage\(\w+, True, \(\w+, \w+\)\)")
        self.assertIn("_helion_gluon.tmem_dot_scaled(", code)

    @skipUnlessGluon()
    @skipIfRefEager("checks generated code")
    def test_dot_scaled_fp4_mn_major(self) -> None:
        """tcgen05 reads fp4 K-major only: an MN-major slot is restaged."""
        x, y = (
            torch.randint(0, 256, shape, dtype=torch.uint8, device=DEVICE)
            for shape in ([256, 256], [256, 128])
        )
        x_scale, y_scale = (
            torch.randint(124, 130, [rows, 16], dtype=torch.uint8, device=DEVICE)
            for rows in (256, 128)
        )
        args = (x, x_scale, y, y_scale)
        _, expected = code_and_output(fp4_mm, args, **{**RING, "num_stages": 1})
        code, out = code_and_output(fp4_mm, args, **RING)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        self.assertEqual(code.count("_helion_ring.ring_acquire("), 2)

    @skipUnlessGluon()
    @skipIfRefEager("checks generated code")
    def test_dot_scaled_scales_per_ring_slot(self) -> None:
        """A ring dot's block scales take a tensor-memory buffer per slot of its
        channel, which the ring allocates for the kernel (``slot_scales``): the
        copied weight scale (tcgen05.cp) and the tokens' scale stored from
        registers alike.  A slot fills again only after the MMA reading it
        completed, so no step waits for an MMA before writing its scales, and an
        in-flight MMA's scales are never rewritten (by a cp after an MMA either,
        which PTX ISA 9.3 does not order): every step's distinct scales count."""
        args, expected = _fp4_rows_by_tokens_reference()
        for stages, specialize in ((2, True), (4, True), (3, None)):
            with self.subTest(num_stages=stages, range_warp_specializes=specialize):
                code, out = code_and_output(
                    fp4_rows_by_tokens,
                    args,
                    range_warp_specializes=[None, specialize],
                    **{**RING, "num_stages": stages},
                )
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                self.assertEqual(
                    "def _helion_fp4_rows_by_tokens_mma(" in code, bool(specialize)
                )
                # One buffer per slot of the channel, for each scale.
                self.assertEqual(code.count("_helion_gluon.slot_scales("), 2)
                self.assertRegex(
                    code,
                    r"_helion_scales0 = _helion_gluon\.slot_scales\(tl\.float8e4nv, "
                    r"128, 32, _helion_ring_S\d+\)",
                )
                # Under the MMA partition the weight scale is copied (tcgen05.cp),
                # the tokens' stored from registers.
                copied = "True, False" if specialize else r"\w+, \w+"
                self.assertRegex(
                    code,
                    rf"_helion_gluon\.tmem_dot_scaled\([^\n]*{copied}, False, "
                    r"_helion_scales0, _helion_scales1\)",
                )

    @skipUnlessGluon()
    @skipIfRefEager("checks generated code")
    def test_dot_scaled_scales_without_tensor_memory_per_slot(self) -> None:
        """Without tensor memory for a block-scale buffer per ring slot beside the
        accumulator rings, the ring's scaled dots allocate their own scales and
        wait for their MMAs.  A 128-column budget leaves the 4-slot ring's scales
        ([128, 32] and [32, 32]: 48 columns a slot) no room beside the [128, 16]
        accumulator ring."""
        args, expected = _fp4_rows_by_tokens_reference()
        with mock.patch("helion._compiler.triton.gluon_ring._TMEM_COLUMNS", 128):
            code, out = code_and_output(
                fp4_rows_by_tokens,
                args,
                range_warp_specializes=[None, True],
                **{**RING, "num_stages": 4},
            )
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        self.assertNotIn("_helion_gluon.slot_scales(", code)
        self.assertIn("_helion_scales0 = None", code)

    @skipUnlessGluon()
    @skipIfRefEager("checks generated code")
    def test_dot_scaled_fp4_n8(self) -> None:
        """tcgen05 takes an 8-column block-scaled tile (Triton's rule asks 16)."""
        x, y = (
            torch.randint(0, 256, shape, dtype=torch.uint8, device=DEVICE)
            for shape in ([128, 256], [16, 256])
        )
        x_scale, y_scale = (
            torch.randint(124, 130, [rows, 16], dtype=torch.uint8, device=DEVICE)
            for rows in (128, 16)
        )
        args = (x, x_scale, y, y_scale)
        _, expected = code_and_output(fp4_mm_nt, (*args, 16), **RING)
        code, out = code_and_output(fp4_mm_nt, (*args, 8), **RING)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        self.assertIn("_helion_gluon.tmem_dot_scaled(", code)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_gated_two_roots(self) -> None:
        """The second root's operand is the first root's output: its producer
        streams it after the body's dependency waits, across overlapping (PDL)
        launches.  Static task counts are never claimed, under dynamic dispatch
        too."""
        x = torch.randn([256, 1024], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        base = {
            "block_sizes": [128, 64, 128, 64, 64],
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 1,
            "num_warps": 4,
            "indexing": "tensor_descriptor",
        }
        slots = len(scale_then_mm.bind((x, w)).config_spec.range_warp_specialize)
        _, (_, expected) = code_and_output(
            scale_then_mm, (x, w), cross_loop_pipeline="static", **base
        )
        for pipeline, specialize in itertools.product(
            ("static", "dynamic"), (None, True)
        ):
            with self.subTest(pipeline=pipeline, range_warp_specializes=specialize):
                flags = [None] * (slots - 1) + [specialize]
                ring = {
                    "num_stages": 2,
                    "range_warp_specializes": flags,
                    "cross_loop_pipeline": pipeline,
                    **base,
                }
                code, (_, out) = code_and_output(scale_then_mm, (x, w), **ring)
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                inputs = [(x + launch, w) for launch in range(4)]
                bound = scale_then_mm.bind((x, w))
                results = _replay(bound.compile_config(helion.Config(**ring)), inputs)
                references = list(
                    itertools.starmap(
                        bound.compile_config(
                            helion.Config(cross_loop_pipeline="static", **base)
                        ),
                        inputs,
                    )
                )
                for (y, out), (y_ref, out_ref) in zip(results, references, strict=True):
                    torch.testing.assert_close(y, y_ref, atol=0, rtol=0)
                    torch.testing.assert_close(out, out_ref, atol=0, rtol=0)
                self.assertIn("def _helion_scale_then_mm_producer1(", code)
                self.assertEqual(
                    "def _helion_scale_then_mm_mma(" in code, bool(specialize)
                )
                functions = _functions(code)
                # The launch state's read waits for the previous grid; its
                # rewrite runs once, after the partitions join.
                self.assertIn("gdc_wait()", functions["_helion_scale_then_mm_launch"])
                self.assertIn(
                    "tl.store(tile_dependency_state",
                    functions["_helion_scale_then_mm_finish"],
                )
                shell = functions["_helion_scale_then_mm"]
                self.assertLess(
                    shell.index("warp_specialize("),
                    shell.index("_helion_scale_then_mm_finish("),
                )
                self.assertNotIn("_launch(", shell)
                self.assertNotIn("gdc_wait", shell)
                # Every partition waits and reads the state itself, once.
                names = ["body", "producer0", "producer1", *["mma"] * bool(specialize)]
                for name in names:
                    fn = functions[f"_helion_scale_then_mm_{name}"]
                    self.assertEqual(fn.count("= _helion_scale_then_mm_launch("), 1)
                    self.assertNotIn("gdc_wait()", fn)
                    self.assertNotIn("tl.store(tile_dependency_state", fn)
                body = functions["_helion_scale_then_mm_body"]
                # The code prefetch runs ahead of the wait (in the idle MMA partition
                # when there is one).
                prefetching = (
                    functions["_helion_scale_then_mm_mma"] if specialize else body
                )
                at = prefetching.index(
                    "call.uni helion_prefetch_code_fn_1,"
                    if specialize
                    else "helion_prefetch_code.prefetch_code("
                )
                self.assertLess(at, prefetching.index("_launch("))
                self.assertIn("return (tile_dependency_epoch,)", body)
                self.assertNotIn("def _helion_scale_then_mm_claimer(", code)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_code_prefetch_between_roots_in_mma_partition(self) -> None:
        """Code prefetches run in the MMA partition and nowhere else: the entry
        one ahead of the launch-state read (no grid wait to take the place of),
        the one between two dot roots after the first root's MMAs and before the
        second's."""
        x = torch.randn([256, 512], device=DEVICE, dtype=torch.bfloat16)
        w1 = torch.randn([512, 256], device=DEVICE, dtype=torch.bfloat16)
        w2 = torch.randn([256, 128], device=DEVICE, dtype=torch.bfloat16)
        base = {
            "block_sizes": [128, 64, 64, 128, 64, 64],
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 1,
            "num_warps": 4,
            "indexing": "tensor_descriptor",
        }
        _, (y_ref, expected) = code_and_output(
            mm_prefetch_mm, (x, w1, w2), cross_loop_pipeline="static", **base
        )
        slots = len(mm_prefetch_mm.bind((x, w1, w2)).config_spec.range_warp_specialize)
        for pipeline in ("static", "dynamic"):
            with self.subTest(pipeline=pipeline):
                code, (y, out) = code_and_output(
                    mm_prefetch_mm,
                    (x, w1, w2),
                    num_stages=2,
                    range_warp_specializes=[True] * slots,
                    cross_loop_pipeline=pipeline,
                    **base,
                )
                torch.testing.assert_close(y, y_ref, atol=0, rtol=0)
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                # The MMA partition's one warp calls the prefetch, on every 16th
                # CTA counted from the last worker (claimed dispatch too).
                self.assertEqual(code.count("call.uni helion_prefetch_code_fn_"), 2)
                self.assertNotIn("helion_prefetch_code.prefetch_code(", code)
                mma = _functions(code)["_helion_mm_prefetch_mm_mma"]
                entry, at = (
                    m.start()
                    for m in re.finditer("call.uni helion_prefetch_code_fn_1,", mma)
                )
                self.assertLess(entry, mma.index("_launch("))
                self.assertLess(mma.index("tmem_dot"), at)
                self.assertLess(at, mma.rindex("tmem_dot"))
                self.assertIn("[(tl.num_programs(0) - 1 - tl.program_id(0)).to(", mma)

    def _prefetch_case(self, kernel, args: tuple[object, ...]) -> str:
        """Run a prefetching ``scale_then_mm`` variant in a ring and check it
        against the shared static ``scale_then_mm`` reference; its code."""
        x, w, y_ref, expected = _prefetch_reference()
        code, (y, out) = code_and_output(
            kernel,
            (x, w, *args),
            num_stages=2,
            cross_loop_pipeline="dynamic",
            **PREFETCH_CONFIG,
        )
        torch.testing.assert_close(y, y_ref, atol=0, rtol=0)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        return code

    def _copy_case(self, kernel, w: torch.Tensor, view: torch.Tensor) -> str:
        """Run a prefetching copy of ``w`` (``view``: the prefetched view of
        it) with tensor-descriptor indexing and box prefetches, and check the
        copy; its code."""
        code, out = code_and_output(
            kernel,
            (w, view),
            block_sizes=[64, 64],
            pid_type="persistent_blocked",
            indexing="tensor_descriptor",
            prefetch_indexing="tensor_descriptor",
        )
        torch.testing.assert_close(out, w, atol=0, rtol=0)
        return code

    def _assert_prefetch_in_body(self, code: str, kernel_name: str) -> None:
        """The tensor prefetches issue from the body partition (inlined, or in
        a task function it calls)."""
        functions = _functions(code)
        body = functions[f"_helion_{kernel_name}_body"]
        holders = [
            name
            for name, fn in functions.items()
            if "cp.async.bulk.prefetch.tensor" in fn
        ]
        self.assertTrue(holders)
        for name in holders:
            self.assertTrue(
                name == f"_helion_{kernel_name}_body" or f"{name}(" in body, name
            )

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_tensor_form_prefetch(self) -> None:
        """A prefetch of whole trailing dims in the tensor_descriptor form is
        one TMA box prefetch from a tensor map in global memory (built on the
        host)."""
        _, w, _, _ = _prefetch_reference()
        wp = w.view(16, 64, 256)
        code = self._prefetch_case(scale_boxprefetch_mm, (wp,))
        self.assertIn("cp.async.bulk.prefetch.tensor.3d.L2.global.tile", code)
        self.assertNotIn("cp.async.bulk.prefetch.L2.global", code)
        self.assertIn("helion_tensor_maps.prefetch_map(wp, [1, 64, 256])", code)
        # One warp of the body's four issues each box.
        self.assertIn("mov.u32 t, %warpid; and.b32 t, t, 3;", code)
        # The map is acquired once at the partition's entry, not per prefetch.
        body = _functions(code)["_helion_scale_boxprefetch_mm_body"]
        self.assertEqual(body.count("fence.proxy.tensormap"), 1)
        self.assertLess(body.index("fence.proxy.tensormap"), body.index("for "))
        self._assert_prefetch_in_body(code, "scale_boxprefetch_mm")

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_tensor_form_prefetch_long_span(self) -> None:
        """A spanned slice longer than a TMA box dim is consecutive boxes, not
        the bulk form; so is a box over the encoder's byte limit (a 128 KB
        [1, 256, 256] box halves into two)."""
        _, w, _, _ = _prefetch_reference()
        code = self._copy_case(boxprefetch_copy, w, w.view(2, 512, 256))
        self.assertEqual(
            code.count("cp.async.bulk.prefetch.tensor.3d.L2.global.tile"), 4
        )
        self.assertNotIn("cp.async.bulk.prefetch.L2.global", code)
        self.assertIn("helion_tensor_maps.prefetch_map(wp, [1, 128, 256])", code)
        # Each box 128 rows past the one before (innermost dim first).
        for rows in ("0", "0 + 128", "0 + 256", "0 + 384"):
            self.assertIn(
                f"[wp_pf_map_acquired, 0, {rows}, tl.cast(offset_0, tl.int32)]", code
            )

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_tensor_form_prefetch_misaligned_base(self) -> None:
        """The bound kernel is keyed on the prefetched input's base alignment:
        a misaligned call after an aligned one keeps the bulk form instead of
        building a tensor map the encoder rejects."""
        storage = torch.randn([1024 * 256 + 8], device=DEVICE, dtype=torch.bfloat16)
        aligned, misaligned = (
            self._copy_case(boxprefetch_copy, w, w.view(16, 64, 256))
            for w in (
                storage[offset : offset + 1024 * 256].view(1024, 256)
                for offset in (0, 1)
            )
        )
        self.assertIn("cp.async.bulk.prefetch.tensor", aligned)
        self.assertIn("cp.async.bulk.prefetch.L2.global", misaligned)
        self.assertNotIn("cp.async.bulk.prefetch.tensor", misaligned)
        self.assertNotIn("prefetch_map(", misaligned)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_tensor_form_prefetch_keyed_on_base_only(self) -> None:
        """A view made in the kernel's host code is keyed on its input's base
        like a passed tensor, and a descriptor reading that input through
        another kernel-code view keys a zero storage offset too: an aligned base
        at a nonzero offset keeps the box form, and a misaligned base after it
        gets its own bound kernel (the bulk form)."""
        storage = torch.randn([1024 * 256 + 8], device=DEVICE, dtype=torch.bfloat16)
        codes = []
        for offset in (8, 1):
            w = storage[offset : offset + 1024 * 256].view(1024, 256)
            code, out = code_and_output(
                viewprefetch_copy,
                (w,),
                block_sizes=[64, 64],
                pid_type="persistent_blocked",
                indexing="tensor_descriptor",
                prefetch_indexing="tensor_descriptor",
            )
            torch.testing.assert_close(out, w, atol=0, rtol=0)
            codes.append(code)
        self.assertIn("cp.async.bulk.prefetch.tensor.3d.L2.global.tile", codes[0])
        self.assertIn("cp.async.bulk.prefetch.L2.global", codes[1])
        self.assertNotIn("prefetch_map(", codes[1])

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_tensor_form_prefetch_falls_back_to_bulk(self) -> None:
        """A dynamic-shape kernel's strides are the first call's only, unless
        specialized: a later call may bring rows the encoder rejects, so the
        prefetch keeps the bulk form (the static fallbacks are
        test_prefetch_possible)."""
        _, w, _, _ = _prefetch_reference()
        code = self._copy_case(dynboxprefetch_copy, w, w.view(16, 64, 256))
        self.assertIn("cp.async.bulk.prefetch.L2.global", code)
        self.assertNotIn("prefetch_map(", code)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_tensor_form_prefetch_scalar_between_spans(self) -> None:
        """A scalar index between full slices is a box extent of 1: still one
        tensor-form prefetch per box."""
        _, w, _, _ = _prefetch_reference()
        code = self._copy_case(halfboxprefetch_copy, w, w.view(16, 64, 2, 128))
        self.assertIn("cp.async.bulk.prefetch.tensor.4d.L2.global.tile", code)
        self.assertNotIn("cp.async.bulk.prefetch.L2.global", code)
        self.assertIn("helion_tensor_maps.prefetch_map(wh, [1, 64, 1, 128])", code)
        # Innermost first: the columns, the half's index, the rows, the block's.
        self.assertIn(
            "[wh_pf_map_acquired, 0, tl.cast(offset_1, tl.int32), 0, "
            "tl.cast(offset_0, tl.int32)]",
            code,
        )

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_loaded_index_tensor_form_prefetch(self) -> None:
        """A prefetch at a loaded block index (an expert prefetch reading its
        routing) keeps the tensor form: the index is a box coordinate."""
        _, w, _, _ = _prefetch_reference()
        ids = torch.tensor([3, -1, 7, 0], device=DEVICE, dtype=torch.int32)
        code = self._prefetch_case(scale_idprefetch_mm, (w.view(16, 64, 256), ids))
        self.assertIn("cp.async.bulk.prefetch.tensor.3d.L2.global.tile", code)
        # The loaded block index is the outermost coordinate.
        self.assertIn(
            "[wp_pf_map_acquired, 0, 0, tl.cast(block_copy_0, tl.int32)]", code
        )
        self._assert_prefetch_in_body(code, "scale_idprefetch_mm")

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_megakernel_timeline(self) -> None:
        """A ring megakernel's timeline: the compute partition stamps every task
        (its producers and MMA partition none), the outputs unchanged."""
        x = torch.randn([256, 1024], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        slots = len(scale_then_mm.bind((x, w)).config_spec.range_warp_specialize)
        ring = {
            "block_sizes": [128, 64, 128, 64, 64],
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 1,
            "num_warps": 4,
            "indexing": "tensor_descriptor",
            "num_stages": 2,
            "range_warp_specializes": [None] * (slots - 1) + [True],
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(timeline._buffers, clear=True),
        ):
            for pipeline in ("static", "dynamic"):
                kernel = helion.kernel(
                    scale_then_mm.fn,
                    static_shapes=True,
                    backend="gluon",
                    megakernel_timeline=os.path.join(directory, pipeline),
                )
                for launch in range(2):
                    args = (x + launch, w)
                    code, out = code_and_output(
                        kernel, args, cross_loop_pipeline=pipeline, **ring
                    )
                    _, expected = code_and_output(
                        scale_then_mm, args, cross_loop_pipeline=pipeline, **ring
                    )
                    for got, want in zip(out, expected, strict=True):
                        torch.testing.assert_close(got, want, atol=0, rtol=0)
                # (Every partition takes the buffer, a launch-state argument.)
                functions = _functions(code)
                # The entry clock reads right after the grid wait, ahead of the
                # launch state, in the function every partition reads it with.
                launch_state = functions["_helion_scale_then_mm_launch"]
                clock = launch_state.index("%globaltimer")
                self.assertLess(launch_state.index("gdc_wait()"), clock)
                self.assertLess(clock, launch_state.index("tile_dependency_epoch ="))
                self.assertIn("%globaltimer", functions["_helion_scale_then_mm_body"])
                for name in ("producer0", "producer1", "mma"):
                    self.assertNotIn(
                        "%globaltimer", functions[f"_helion_scale_then_mm_{name}"]
                    )
            written = check_written(self, launches=2)
            self.assertEqual(
                [os.path.basename(path).split(".")[0] for path, _ in written],
                ["dynamic", "static"],
            )

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_shared_slot_skips(self) -> None:
        """Steps of both gating groups share the channel's slots: the gated
        producer waits for the slot of every step it fills nothing of (here the
        first root's), else it would run rounds ahead of the slots' phases, and
        arrives on the step's full barrier, else a step could complete without
        it and leave it a whole phase behind (an aliased parity wait).  Under
        barrier mode, whose grid barrier only the body runs, the first root's
        output stays a plain load after it (no gated producer)."""
        x = torch.randn([128, 8192], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([8192, 128], device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        base = {
            "block_sizes": [128, 128, 64, 128, 64, 64],
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 1,
            "num_warps": 4,
            "indexing": "tensor_descriptor",
        }
        slots = len(mm_then_mm.bind((x, w, v)).config_spec.range_warp_specialize)
        _, expected = code_and_output(
            mm_then_mm, (x, w, v), cross_loop_pipeline="static", **base
        )
        for pipeline in ("static", "barrier"):
            with self.subTest(pipeline=pipeline):
                code, out = code_and_output(
                    mm_then_mm,
                    (x, w, v),
                    num_stages=2,
                    range_warp_specializes=[True] * slots,
                    cross_loop_pipeline=pipeline,
                    **base,
                )
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                self.assertIn("warp_specialize(", code)
                if pipeline == "barrier":
                    self.assertNotIn("def _helion_mm_then_mm_producer1(", code)
                    continue
                gated = code.split("def _helion_mm_then_mm_producer1(")[1]
                gated = gated.split("\ndef ")[0]
                self.assertIn("ring_skip(", gated)
                self.assertIn("ring_put(", gated)
                # Both groups arrive on every full phase; the skip passes its full.
                self.assertIn("ring_barriers(_helion_ring_S0, 2)", code)
                self.assertIn("_helion_ring.ring_skip(_helion_ring_full0,", code)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_paced_first_fill(self) -> None:
        """Beside a gated producer, the weight producer paces its first fill (one
        step in flight, waiting on its own pace barriers) on the CTAs a root
        loading outside the ring runs ahead on: the fill predicate is that root's
        guard.  With the first root ringed too, nothing loads ahead and the fill
        streams.  The gated producer streams either way."""
        x = torch.randn([256, 1024], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([1024, 256], device=DEVICE, dtype=torch.bfloat16)
        base = {
            "block_sizes": [128, 64, 128, 64, 64],
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 1,
            "num_warps": 4,
            "indexing": "tensor_descriptor",
            "cross_loop_pipeline": "static",
        }

        def puts(config: dict[str, object]) -> tuple[list[str], list[str], str]:
            _, (_, expected) = code_and_output(scale_then_mm, (x, w), **config)
            code, (_, out) = code_and_output(
                scale_then_mm, (x, w), num_stages=3, **config
            )
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            producers = [
                [line.strip() for line in p.split("\ndef ")[0].splitlines()]
                for p in code.split("def _helion_scale_then_mm_producer")[1:]
            ]
            weights, gated = [[ln for ln in p if "ring_put" in ln] for p in producers]
            self.assertTrue(weights and gated)
            self.assertTrue(all("ring_put(" in line for line in gated))
            return weights, gated, code

        weights, _, code = puts(base)
        self.assertTrue(all("ring_put(" in line for line in weights))
        self.assertNotIn("ring_pace_flush(", code)
        # The scale root loads x by pointer, outside the ring: the weight
        # producer paces on that root's CTAs and flushes its owed arrivals.
        indexing = ["pointer", *["tensor_descriptor"] * 4]
        weights, _, code = puts({**base, "indexing": indexing})
        self.assertTrue(all("ring_put_paced(" in line for line in weights))
        self.assertTrue(all("_helion_ring_pace0" in line for line in weights))
        guard = "tl.program_id(0) >= 0 and tl.program_id(0) < 32, 1)"
        self.assertTrue(all(line.endswith(guard) for line in weights), weights)
        self.assertIn("ring_pace_flush(", code)
        # A root loading by pointer behind the ring's first step does not.
        indexing = [*["tensor_descriptor"] * 3, "pointer", "tensor_descriptor"]
        _, _, code = puts({**base, "indexing": indexing})
        self.assertNotIn("ring_put_paced(", code)
        # A first fill that waits on a dependency (the rows come from the first
        # root) is cut back to plain puts: no pace barriers are set up for it.
        x = torch.randn([512, 256], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([256, 128], device=DEVICE, dtype=torch.bfloat16)
        perm = torch.randperm(4, device=DEVICE, dtype=torch.int32)
        base = {**base, "block_sizes": [4, 64]}
        _, (expected, _) = code_and_output(route_then_mm, (x, w, perm), **base)
        code, (out, _) = code_and_output(
            route_then_mm, (x, w, perm), num_stages=3, **base
        )
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        self.assertIn("ring_put(", code)
        self.assertNotIn("ring_put_paced(", code)
        self.assertNotIn("_helion_ring_pace", code)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_coordinates_written_in_kernel(self) -> None:
        """A ring load whose coordinates read what an earlier root writes: its
        producer keeps the dependency waits, as a gated one does.  Barrier mode
        has none it could keep (only the body runs its grid barrier): the ring
        falls back."""
        x = torch.randn([512, 256], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([256, 128], device=DEVICE, dtype=torch.bfloat16)
        perms = [torch.randperm(4, device=DEVICE, dtype=torch.int32) for _ in range(4)]
        base = {
            "block_sizes": [4, 64],
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 1,
            "num_warps": 4,
            "indexing": "tensor_descriptor",
        }
        bound = route_then_mm.bind((x, w, perms[0]))
        flags = len(bound.config_spec.range_warp_specialize)
        for pipeline, specialize in itertools.product(
            ("barrier", "static", "dynamic"), (None, True)
        ):
            with self.subTest(pipeline=pipeline, range_warp_specializes=specialize):
                config = helion.Config(
                    num_stages=2,
                    range_warp_specializes=[None] * (flags - 1) + [specialize],
                    cross_loop_pipeline=pipeline,
                    **base,
                )
                if pipeline == "barrier":
                    self.assertNotIn("ring_put(", bound.to_triton_code(config))
                results = _replay(
                    bound.compile_config(config), [(x, w, p) for p in perms]
                )
                for perm, (out, order) in zip(perms, results, strict=True):
                    rows = perm[:, None] * 128 + torch.arange(128, device=DEVICE)
                    expected = x[rows.flatten()].float() @ w.float()
                    torch.testing.assert_close(order, perm, atol=0, rtol=0)
                    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        code = bound.to_triton_code(config)
        self.assertIn("_helion_ring.ring_put(", code)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_claimed_tasks_wait_once_for_an_invariant_counter(self) -> None:
        """Dynamic workers claim a data-dependent task count past their first
        wave (static ones keep fixed ownership), and wait for a counter that is
        the same for all of the tasks once, before the claim loop."""
        x = torch.randn([128 * 64, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 256], device=DEVICE, dtype=torch.bfloat16)
        # The first root's 256 tasks run on every worker (at kernel scope).
        s = torch.randn([4096], device=DEVICE, dtype=torch.float32)
        # 48 of 64 row tiles live: 192 tasks, more than one wave.
        rows = torch.tensor([48 * 128], device=DEVICE, dtype=torch.int32)
        args = (x, y, s, rows)
        base = {
            **RING,
            # s, part (store), rows, x, y, part (load), out.
            "indexing": [
                *("pointer", "pointer", "pointer", "tensor_descriptor"),
                *("tensor_descriptor", "pointer", "tensor_descriptor"),
            ],
            "block_sizes": [64, 128, 64],
        }
        _, expected = code_and_output(mm_times_total, args, **{**base, "num_stages": 1})
        for pipeline in ("static", "dynamic"):
            with self.subTest(pipeline=pipeline):
                code, out = code_and_output(
                    mm_times_total, args, **base, cross_loop_pipeline=pipeline
                )
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                self.assertIn("_helion_ring.ring_acquire(", code)
                claims = "helion_dist_utils._claim(" in code
                self.assertEqual(claims, pipeline == "dynamic")
        body = code.split("def _helion_mm_times_total_body(")[1].split("\ndef ")[0]
        before, loop = body.split("while tl.cast(tile_dependency_claim", 1)
        self.assertIn("ld.acquire", before)
        self.assertNotIn("ld.acquire", loop)
        # Back to back, each launch takes exactly its live count of claims and
        # leaves the counter reset, with fewer live tasks than workers or none.
        bound = mm_times_total.bind(args)
        inputs = [
            (x, y, s, torch.tensor([t * 128], device=DEVICE, dtype=torch.int32))
            for t in (48, 16, 64, 0, 48)
        ]
        dynamic = {**base, "cross_loop_pipeline": "dynamic"}
        results = _replay(bound.compile_config(helion.Config(**dynamic)), inputs)
        reference = bound.compile_config(helion.Config(**{**dynamic, "num_stages": 1}))
        for launch, out in zip(inputs, results, strict=True):
            torch.testing.assert_close(out, reference(*launch), atol=0, rtol=0)
        # 64-bit indexing: the claimed task index takes the program ids' type.
        wide = helion.kernel(
            mm_times_total.fn, static_shapes=True, index_dtype=torch.int64
        )
        _, out = code_and_output(wide, args, **dynamic)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_claim_group_interleaves_keyed_roots(self) -> None:
        """Claimed roots each waiting on the one before key by key (their row
        tile) claim one ticket stream (``_group_ticket``) under claim groups: the
        scaling root's keys a wave of claims behind the matmul's, not after its
        last claim, and the second matmul's after the first's live keys and a
        wave behind the scaling root's.  Without groups each root claims its own
        tasks."""
        x = torch.randn([128 * 64, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 512], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([512, 512], device=DEVICE, dtype=torch.bfloat16) / 64
        rows = torch.tensor([48 * 128, 48 * 128], device=DEVICE, dtype=torch.int32)
        bound = live_mm_scale_mm.bind((x, y, w, rows))
        ungrouped = helion.Config(
            **RING, block_sizes=[128, 64, 128, 128, 64], cross_loop_pipeline="dynamic"
        )
        code = bound.to_triton_code(ungrouped)
        self.assertNotIn("helion_dist_utils._group_claim(", code)
        config = helion.Config(**ungrouped.config, cross_loop_claim_groups=True)
        with self.assertLogs("helion._compiler.cross_loop_codegen", "INFO") as logs:
            code = bound.to_triton_code(config)
        self.assertIn("helion_dist_utils._group_claim(", code)
        self.assertNotIn("helion_dist_utils._claim(", code)
        # The scaling root reads ``h_sum`` and ``h`` through the claiming
        # producer's channel (its ``ring_skip`` steps), so every ticket here
        # issues a ring step and ``task_pace`` never waits (test_gluon_units'
        # ``TestClaimGroups`` reads them through plain loads, where it does).
        self.assertIn("_helion_ring.task_pace(", code)
        # 64 row keys of 4 matmul tasks: the scaling root lags a wave of the
        # workers' claims, the second matmul a wave of both roots' behind it
        # and, at runtime, after the first matmul's live keys (at most 64).
        match = re.search(r"_group_claim\([^,]+, (\d+),", code)  # the workers
        assert match is not None
        workers = int(match[1])
        lag = min(-(-workers // 4), 64)  # (clamped after the 64 keys)
        second = min(max(64, lag + -(-workers // (4 + 8))), lag + 64)
        self.assertIn(
            "Roots [0, 1, 2] claim one interleaved ticket stream "
            f"(key lags [0, {lag}, {second}] rounds)",
            "\n".join(logs.output),
        )
        self.assertRegex(
            code,
            rf"_group_claim\(.*tl\.minimum\(tl\.maximum\(.* // 4, \d+\), {lag + 64}\)",
        )
        # Back to back, each launch takes exactly its live count of claims and
        # leaves the counter reset: fewer live tasks than workers, none, every
        # task live (a bound past the last row), and bounds varying apart, one
        # of them negative (a root with no live task).
        bounds = [(48, 48), (16, 16), (64, 64), (0, 0), (80, 80), (48, -1), (40, 20)]
        inputs = [
            (x, y, w, torch.tensor(b, device=DEVICE, dtype=torch.int32) * 128)
            for b in bounds
        ]
        results = _replay(bound.compile_config(config), inputs)
        for launch, result in zip(inputs, results, strict=True):
            # The bf16 scaled rows round apart by an ulp under another sum order.
            expected = _live_mm_scale_mm_ref(*launch)
            scale = expected.abs().max().item()
            torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2 * scale)

    @skipUnlessGluon()
    def test_group_ticket(self) -> None:
        """``_group_ticket`` maps tickets to (root, task) round by round: root
        ``i``'s key ``r - rounds[i]`` in round ``r``, roots in order, then the end."""

        @triton.jit
        def decode(out, n0, n1, n2, n3, n4, END: tl.constexpr):
            t = tl.program_id(0)
            coded = dist_utils._group_ticket(
                t, (n0, n1, n2, n3, n4), (36, 9, 36, 4, 9), (0, 5, 69, 74, 80), END
            )
            tl.store(out + t, coded.to(tl.int64))

        end = 5 << 32
        tasks, lags = [36, 9, 36, 4, 9], [0, 5, 69, 74, 80]
        for live in (
            [576, 144, 576, 64, 90],
            [360, 90, 360, 40, 0],
            [36, 9, 36, 4, 9],
            [0, 18, 0, 8, 18],
            [0, 0, 0, 0, 0],
        ):
            with self.subTest(live=live):
                tickets = sum(live) + 7
                out = torch.empty([tickets], device=DEVICE, dtype=torch.int64)
                decode[(tickets,)](out, *live, END=end)
                expected = [
                    end if pair is None else pair[0] << 32 | pair[1]
                    for pair in _group_tickets(live, tasks, lags, tickets)
                ]
                self.assertEqual(out.tolist(), expected)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_leader_claims(self) -> None:
        """The weight producer steps in every claimed task loop, so it issues the
        claims itself (no claimer partition), each when it is done issuing the
        previous task's loads: the task channel holds the ring's tasks too."""
        x = torch.randn([128 * 64, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 256], device=DEVICE, dtype=torch.bfloat16)
        s = torch.randn([4096], device=DEVICE, dtype=torch.float32)
        rows = torch.tensor([48 * 128], device=DEVICE, dtype=torch.int32)
        args = (x, y, s, rows)
        bound = mm_times_total.bind(args)
        slots = len(bound.config_spec.range_warp_specialize)
        base = {
            **RING,
            "indexing": [
                *("pointer", "pointer", "pointer", "tensor_descriptor"),
                *("tensor_descriptor", "pointer", "tensor_descriptor"),
            ],
            "block_sizes": [64, 128, 64],
            "cross_loop_pipeline": "dynamic",
        }
        leader = {**base, "range_warp_specializes": [None] * (slots - 1) + [True]}
        code = bound.to_triton_code(helion.Config(**leader))
        functions = _functions(code)
        self.assertIn("_helion_mm_times_total_mma", functions)
        self.assertNotIn("_helion_mm_times_total_claimer", functions)
        # Without PDL every partition still reads the launch state, unwaited.
        self.assertNotIn("gdc_wait", functions["_helion_mm_times_total_launch"])
        for name in ("body", "producer0", "mma"):
            fn = functions[f"_helion_mm_times_total_{name}"]
            self.assertEqual(fn.count("= _helion_mm_times_total_launch("), 1)
        for call in ("task_put(", "helion_dist_utils._claim("):
            issuers = [name for name, fn in functions.items() if call in fn]
            self.assertEqual(issuers, ["_helion_mm_times_total_producer0"], call)
        self.assertIn("_helion_task_S = tl.constexpr(5)", code)
        takers = sum("task_take(" in fn for fn in functions.values())
        self.assertEqual(takers, 2)
        self.assertIn(f"ring_barriers(_helion_task_S, {takers})", code)
        # Back to back, each launch takes exactly its live count of claims.
        inputs = [
            (x, y, s, torch.tensor([t * 128], device=DEVICE, dtype=torch.int32))
            for t in (48, 16, 64, 0, 48)
        ]
        results = _replay(bound.compile_config(helion.Config(**leader)), inputs)
        reference = bound.compile_config(helion.Config(**{**base, "num_stages": 1}))
        for launch, out in zip(inputs, results, strict=True):
            torch.testing.assert_close(out, reference(*launch), atol=0, rtol=0)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_claimer_fallback(self) -> None:
        """The weight producer has no step in the claimed task loop (only the
        gated producer and the MMA partition do), so a claimer partition issues
        the claims, at most one task ahead of the MMA."""
        x = torch.randn([128 * 64, 128], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([128, 256], device=DEVICE, dtype=torch.bfloat16)
        rows = torch.tensor([48 * 128], device=DEVICE, dtype=torch.int32)
        bound = scale_both_then_live_mm.bind((x, w, rows))
        slots = len(bound.config_spec.range_warp_specialize)
        base = {
            **RING,
            "block_sizes": [128, 64, 64, 128, 64, 128, 64],
            "cross_loop_pipeline": "dynamic",
        }
        ring = {**base, "range_warp_specializes": [None] * (slots - 1) + [True]}
        code = bound.to_triton_code(helion.Config(**ring))
        functions = _functions(code)
        # The weight producer streams the first roots' register reads.
        self.assertRegex(
            functions["_helion_scale_both_then_live_mm_producer0"],
            r"ring_put(_paced)?\(",
        )
        issuers = [name for name, fn in functions.items() if "task_put(" in fn]
        self.assertEqual(issuers, ["_helion_scale_both_then_live_mm_claimer"])
        claimer = functions["_helion_scale_both_then_live_mm_claimer"]
        self.assertEqual(claimer.count("= _helion_scale_both_then_live_mm_launch("), 1)
        self.assertIn("_helion_task_S = tl.constexpr(3)", code)
        inputs = [
            (x, w, torch.tensor([t * 128], device=DEVICE, dtype=torch.int32))
            for t in (48, 0, 64)
        ]
        results = _replay(bound.compile_config(helion.Config(**ring)), inputs)
        reference = bound.compile_config(helion.Config(**{**base, "num_stages": 1}))
        for launch, out in zip(inputs, results, strict=True):
            torch.testing.assert_close(out, reference(*launch), atol=0, rtol=0)

    def _check_tails(
        self,
        cases: list[tuple[helion.Kernel, tuple[torch.Tensor, ...], dict, int, int]],
    ) -> None:
        """Each case (kernel, args, config, its tail_fill count, the first
        weight's tiles or 0) bit for bit as without tails, every candidate
        loop enabled."""
        for kernel, args, config, fills, tiles in cases:
            bound = kernel.bind(args)
            tails = [True] * len(bound.config_spec.cross_loop_tmem_prefetch)
            with self.subTest(kernel=kernel.name, fills=fills, tiles=tiles):
                code, outs = code_and_output(
                    kernel, args, **config, cross_loop_tmem_prefetch=tails
                )
                _, references = code_and_output(kernel, args, **config)
                self.assertEqual(code.count("tail_fill("), fills)
                if tiles:
                    self.assertIn(f"tail_buffers(w_ring_desc, {tiles}, 128, 64)", code)
                for got, expected in zip(
                    outs if isinstance(outs, tuple) else (outs,),
                    references if isinstance(references, tuple) else (references,),
                    strict=True,
                ):
                    torch.testing.assert_close(got, expected, atol=0, rtol=0)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_cross_loop_tmem_prefetch(self) -> None:
        """A K loop behind its task's gate whose weight (operand A) streams
        ungated takes a tail (``cross_loop_tmem_prefetch``), bit for bit as
        without it, with more tasks than workers: a worker refilling the tiles
        its last task read behind ring steps (14 of 16 steps); each loop all
        tail, the ring deeper than it (2 of 2), so a fill overwrites tiles the
        previous task's MMAs read one ring step earlier (after the fill's wait
        for them): with a paced first fill, and in two tasks' loops sharing a
        buffer; and a task's bf16 and fp16 loops, a buffer each (two buffers
        and two accumulator rings in all of tensor memory). The generated
        code: test_gluon_units.TestTmemPrefetch."""
        x = torch.randn([32, 1024], device=DEVICE, dtype=torch.bfloat16)
        sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        many = torch.randn([(sms + 4) * 128, 1024], device=DEVICE, dtype=torch.bfloat16)
        ring = gluon_tail_ring_config(scale_then_wx.bind((x, many)))
        # The first root loading x by pointer: the weight producer paces its
        # first fill.
        paced = {**ring, "indexing": ["pointer", *["tensor_descriptor"] * 4]}
        short = (x[:, :128], many[:, :128], torch.randn_like(many[:, :128]))
        two_ring = gluon_tail_ring_config(
            scale_then_two_wx.bind(short),
            block_sizes=[32, 64, *[128, 32, 64] * 2],
            range_warp_specializes=[None, True, None, True, None],
        )
        mixed = (
            x[:, :256],
            many[:256, :256],
            torch.randn([32, 1024], device=DEVICE, dtype=torch.float16),
            torch.randn([256, 1024], device=DEVICE, dtype=torch.float16),
        )
        mixed_ring = gluon_tail_ring_config(
            scale_then_wx_and_f16_vx.bind(mixed),
            task_loop=2,
            block_sizes=[32, 64, 32, 64, 128, 32, 64, 64],
        )
        self._check_tails(
            [
                (scale_then_wx, (x, many), ring, 1, 14),
                (scale_then_wx, short[:2], paced, 1, 2),
                (scale_then_two_wx, short, two_ring, 2, 0),
                (scale_then_wx_and_f16_vx, mixed, mixed_ring, 2, 4),
            ]
        )

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_cross_loop_tmem_prefetch_loop_shapes(self) -> None:
        """Tails bit for bit as without them: one weight of a stack (a unit
        leading box dim), a chunked view's (a unit inner one), two weights in
        one loop (the first dot takes the tail), a K loop inside another's body (the outer one takes the tail;
        two accumulator rings, or a nested loop's, leave fewer columns), and an
        nvfp4 dot's block scales beside a tail (9 tiles: the scales' buffer per
        ring slot takes 96 columns)."""
        x = torch.randn([32, 1024], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([256, 1024], device=DEVICE, dtype=torch.bfloat16)
        ring = gluon_tail_ring_config(scale_then_wx.bind((x, w)))
        stacked = torch.stack([torch.randn_like(w), w])
        w3, v = torch.randn_like(w), torch.randn_like(w)
        nested_ring = gluon_tail_ring_config(
            scale_then_nested_wx.bind((x, w, v)), task_loop=1
        )
        # nvfp4 operands packed along K (any byte is two e2m1 values), finite
        # e4m3 scales.
        a, b = (
            torch.randint(0, 256, shape, device=DEVICE, dtype=torch.uint8)
            for shape in ([256, 256], [256, 32])
        )
        a_scale, b_scale = (
            torch.rand([rows, 32], device=DEVICE).to(torch.float8_e4m3fn)
            for rows in (256, 32)
        )
        nvfp4 = (x, w, a, a_scale, b, b_scale)
        nvfp4_ring = gluon_tail_ring_config(
            scale_then_wx_and_nvfp4.bind(nvfp4), task_loop=1
        )
        chunked = (x, w.view(256, 2, 512))
        chunked_ring = gluon_tail_ring_config(scale_then_chunked_wx.bind(chunked))
        self._check_tails(
            [
                (scale_then_ewx, (x, stacked), ring, 1, 14),
                (scale_then_chunked_wx, chunked, chunked_ring, 1, 8),
                (scale_then_w13x, (x, w, w3), ring, 1, 0),
                (scale_then_nested_wx, (x, w, v), nested_ring, 1, 0),
                (scale_then_wx_and_nvfp4, nvfp4, nvfp4_ring, 1, 9),
            ]
        )

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_dsmem(self) -> None:
        """A split and combine pair's parts travel in clusters of two CTAs
        (``cross_loop_dsmem``): a block's two K splits run on the CTAs
        of one cluster, which send them into rank 0's shared memory, and its
        sum on rank 0, which reads them there.  Every block count fits one
        wave of clusters on this device."""
        x = torch.randn([16, 1024], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16)
        blocks = _one_wave(2, 16, self)
        w = w[: 64 * blocks]
        w4 = w.view(blocks, 64, 8, 128).permute(0, 2, 1, 3).contiguous()
        part = torch.zeros([2, 16, 64 * blocks], device=DEVICE, dtype=torch.float32)
        args = (x, w4, part)
        bound = split_k_parts.bind(args)
        self.assertEqual(bound.config_spec.cross_loop_dsmem.length, 1)
        ring = {**RING, "num_warps": 4, "cross_loop_pipeline": "static"}
        code, out = code_and_output(
            split_k_parts, args, **ring, cross_loop_dsmem=[True]
        )
        plain, reference = code_and_output(split_k_parts, args, **ring)
        torch.testing.assert_close(out, reference, atol=0, rtol=0)
        self.assertNotIn("cluster_send", plain)
        self.assertIn("_cluster_size=2", code)
        body = _functions(code)["_helion_split_k_parts_body"]
        self.assertEqual(body.count("cluster_send("), 1)
        self.assertEqual(body.count("cluster_recv("), 1)
        # One slab of 2 x 16 x 64 fp32 parts after the 128-byte header.
        self.assertIn("cluster_entry(9216, '8192')", code)
        # Launches back to back on changing inputs match the counter edge's.
        run = bound.compile_config(helion.Config(**ring, cross_loop_dsmem=[True]))
        run_reference = bound.compile_config(helion.Config(**ring))
        for launch in range(3):
            x_launch = x + launch
            torch.testing.assert_close(
                run(x_launch, w4, part),
                run_reference(x_launch, w4, part),
                atol=0,
                rtol=0,
            )
        # Each task's two K halves stored apart: the slab takes both stores'
        # tiles (2 x 2 x 16 x 64 fp32), each store sending its own.
        halves = torch.zeros([2, 2, 16, 64 * blocks], device=DEVICE)
        code_h, out_h = code_and_output(
            split_k_halves, (x, w4, halves), **ring, cross_loop_dsmem=[True]
        )
        _, reference_h = code_and_output(split_k_halves, (x, w4, halves), **ring)
        torch.testing.assert_close(out_h, reference_h, atol=0, rtol=0)
        body_h = _functions(code_h)["_helion_split_k_halves_body"]
        self.assertEqual(body_h.count("cluster_send("), 2)
        self.assertIn("cluster_entry(17408, '16384')", code_h)
        # A slab past 48 KB of static shared memory: 2 x 128 x 64 fp32 parts.
        x128 = torch.randn([128, 1024], device=DEVICE, dtype=torch.bfloat16)
        part128 = torch.zeros([2, 128, 64 * blocks], device=DEVICE, dtype=torch.float32)
        code128, out128 = code_and_output(
            split_k_parts, (x128, w4, part128), **ring, cross_loop_dsmem=[True]
        )
        self.assertIn("cluster_entry(66560, '65536')", code128)
        torch.testing.assert_close(
            out128, (w.float() @ x128.float().T).T, atol=1e-3, rtol=1e-3
        )
        # Four and eight splits a block: clusters of four and eight, the
        # launch taking the SMs whole clusters keep resident (no
        # ``persistent_reserved_sms``), which the host asks for.
        for splits in (4, 8):
            if get_cluster_capacity(DEVICE, splits) < splits:
                continue
            blocks = min(w4.size(0), _cluster_workers(splits) // splits)
            part_s = torch.zeros([splits, 16, 64 * blocks], device=DEVICE)
            code_s, out_s = code_and_output(
                split_k_parts,
                (x, w4[:blocks], part_s),
                **ring,
                cross_loop_dsmem=[True],
            )
            self.assertIn(f"_cluster_size={splits}", code_s)
            self.assertIn(f"cluster_size={splits})", code_s)
            torch.testing.assert_close(
                out_s,
                (w[: 64 * blocks].float() @ x.float().T).T,
                atol=1e-3,
                rtol=1e-3,
            )

    @skipUnlessGluon()
    def test_cluster_flag_value_is_a_hash(self) -> None:
        """The value a CTA raises its flag to is splitmix64's finalizer of the
        launch epoch, its cluster's id (0 here: a launch without clusters) and
        the launch state's address, with the top bit set: never the 0 every
        CTA clears its flag to, nor a structured word another kernel's data
        would hold (an epoch of 1 in cluster 0 of a 1 MiB-aligned state)."""
        epochs = [0, 1, 7, 0xFFFFFFFF]
        states = [1 << 20, 1 << 20, 0x12345670, 0]
        values = torch.empty([4], device=DEVICE, dtype=torch.int64)
        _cluster_values[(1,)](
            torch.tensor(epochs, device=DEVICE),
            torch.tensor(states, device=DEVICE),
            values,
            4,
            BLOCK=4,
        )
        mask = (1 << 64) - 1

        def mix(v: int) -> int:
            v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9 & mask
            v = (v ^ (v >> 27)) * 0x94D049BB133111EB & mask
            return (v ^ (v >> 31)) | 1 << 63

        expected = [mix(e ^ s) for e, s in zip(epochs, states, strict=True)]
        # The values as unsigned words.
        got = [value % (1 << 64) for value in values.tolist()]
        self.assertEqual(got, expected)
        self.assertEqual(len(set(got)), len(got))
        self.assertNotIn(0, got)
        self.assertNotIn((1 << 63) | 1, got)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_dsmem_any_loop_order(self) -> None:
        """The split axis may take any place in the producer root's task
        order: split slower (``hl.tile([n, splits])``'s default) or first
        (``loop_orders``), a key's parts run on one cluster and match the
        counter edge's bit for bit.  Parts of 64 rows by 2 columns, one
        element a thread, send single words; the receive of both parts takes
        two words a thread.  Parts of 16 x 16 (``block_sizes`` 16), which four
        elements a thread would put on 64 of the 128 threads (C3), send two a
        thread.  Every tile count fits one wave of clusters on this device."""
        x = torch.randn([16, 1024], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16)
        keys = _one_wave(2, 32, self)
        w = w[: 64 * min(keys, 16)]
        part = torch.zeros([2, 16, w.size(0)], device=DEVICE, dtype=torch.float32)
        config = {
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 1,
            "num_warps": 4,
            "cross_loop_pipeline": "static",
        }
        tiles = {**config, "block_sizes": [64, 128, 64]}
        _, reference = code_and_output(split_k_gemv, (x, w, part), **tiles)
        torch.testing.assert_close(
            reference, (w.float() @ x.float().T).T, atol=1e-3, rtol=1e-3
        )
        for order in ({}, {"loop_orders": [[1, 0]]}):
            code, out = code_and_output(
                split_k_gemv,
                (x, w, part),
                **tiles,
                **order,
                cross_loop_dsmem=[True],
            )
            self.assertIn("cluster_send", code)
            torch.testing.assert_close(out, reference, atol=0, rtol=0)
        x64 = torch.randn([64, 1024], device=DEVICE, dtype=torch.bfloat16)
        narrow = (x64, w[: 2 * keys], torch.zeros([2, 64, 2 * keys], device=DEVICE))
        columns = {**config, "block_sizes": [2, 128, 2]}
        _, reference = code_and_output(split_k_gemv, narrow, **columns)
        code, out = code_and_output(
            split_k_gemv, narrow, **columns, cross_loop_dsmem=[True]
        )
        self.assertRegex(code, r"(?m)cluster_send\(.*, 8, 1, 0, 1\)$")
        self.assertIn("tl.float32, 8, 2)", code)
        torch.testing.assert_close(out, reference, atol=0, rtol=0)
        small = {**config, "block_sizes": [16, 128, 16]}
        _, reference = code_and_output(split_k_gemv, (x, w, part), **small)
        code, out = code_and_output(
            split_k_gemv, (x, w, part), **small, cross_loop_dsmem=[True]
        )
        self.assertRegex(code, r"(?m)cluster_send\(.*, 2, 0, 1\)$")
        torch.testing.assert_close(out, reference, atol=0, rtol=0)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_dsmem_all_gather(self) -> None:
        """Per-rank consumers and local slabs: a row's four part sums of
        squares (one element a task, which one of the threads holding it
        sends) reach every CTA of its cluster, and each part's values stay on
        the CTA whose norm task reads them, matching the counter edges bit
        for bit."""
        rows = _one_wave(4, 8, self)
        x = torch.randn([rows, 1024], device=DEVICE, dtype=torch.float32)
        weight = torch.randn([1024], device=DEVICE, dtype=torch.float32)
        args = (x, weight, torch.zeros([rows, 4], device=DEVICE), torch.zeros_like(x))
        config = {
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 1,
            "num_warps": 4,
            "cross_loop_pipeline": "static",
            "block_sizes": [256, 256],
        }
        _, reference = code_and_output(gather_rms_norm, args, **config)
        doubled = x * 2
        expected = (
            doubled * torch.rsqrt(doubled.pow(2).mean(1, keepdim=True) + 1e-6) * weight
        )
        torch.testing.assert_close(reference, expected, atol=1e-5, rtol=1e-5)
        code, out = code_and_output(
            gather_rms_norm, args, **config, cross_loop_dsmem=[True, True]
        )
        self.assertIn("_cluster_size=4", code)
        # The sums: one element to every rank, from one of 128 threads.
        for rank in range(4):
            self.assertRegex(code, rf"(?m)cluster_send\(.*, 8, 1, {rank}, 128\)$")
        # The values: to this CTA (-1).
        self.assertRegex(code, r"(?m)cluster_send\(.*, 16, 2, -1, 1\)$")
        torch.testing.assert_close(out, reference, atol=0, rtol=0)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_dsmem_masked_tiles(self) -> None:
        """A row of six parts (a hidden size that is no power of two): the norm
        tasks read the row's six sums as a masked tile of eight, from the
        slab where the mask holds (the padding lanes read its first element
        and take the masked load's zero), in clusters of six; a replicated
        table of six rows read the same way. Both match the counter edges bit
        for bit."""
        rows = _one_wave(6, 8, self)
        x = torch.randn([rows, 1536], device=DEVICE, dtype=torch.float32)
        weight = torch.randn([1536], device=DEVICE, dtype=torch.float32)
        args = (x, weight, torch.zeros([rows, 6], device=DEVICE), torch.zeros_like(x))
        config = {
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 1,
            "num_warps": 4,
            "cross_loop_pipeline": "static",
            "block_sizes": [256, 256],
        }
        _, reference = code_and_output(gather_rms_norm, args, **config)
        code, out = code_and_output(
            gather_rms_norm, args, **config, cross_loop_dsmem=[True, True]
        )
        self.assertIn("_cluster_size=6", code)
        for rank in range(6):
            self.assertRegex(code, rf"(?m)cluster_send\(.*, 8, 1, {rank}, 128\)$")
        # The sums' receive: under the tile's mask, its padding lanes zero.
        self.assertRegex(code, r"tl\.where\(mask_\d+, _helion_gluon\.cluster_recv\(")
        torch.testing.assert_close(out, reference, atol=0, rtol=0)
        table = torch.randn([6, 16], device=DEVICE, dtype=torch.float32)
        args = (
            x,
            weight,
            table,
            torch.zeros_like(table),
            torch.zeros([6], device=DEVICE),
            torch.zeros([rows, 6], device=DEVICE),
            torch.zeros_like(x),
        )
        _, reference = code_and_output(scaled_gather_rms_norm, args, **config)
        code, out = code_and_output(
            scaled_gather_rms_norm,
            args,
            **config,
            cross_loop_dsmem=[True, False, True, True],
        )
        self.assertIn("_cluster_size=6", code)
        for rank in range(6):
            self.assertRegex(code, rf"(?m)cluster_send\(.*, 8, 1, {rank}, 8\)$")
        torch.testing.assert_close(out, reference, atol=0, rtol=0)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_dsmem_replicated(self) -> None:
        """A whole table that every task of a root on more CTAs than a
        cluster reads: every cluster runs the table's tasks, each sending its
        row to every rank, and cluster 0 alone arrives (on the row sums'
        counter), matching the counter edges bit for bit."""
        rows = _one_wave(4, 8, self)
        x = torch.randn([rows, 1024], device=DEVICE, dtype=torch.float32)
        weight = torch.randn([1024], device=DEVICE, dtype=torch.float32)
        table = torch.randn([8, 16], device=DEVICE, dtype=torch.float32)
        args = (
            x,
            weight,
            table,
            torch.zeros_like(table),
            torch.zeros([8], device=DEVICE),
            torch.zeros([rows, 4], device=DEVICE),
            torch.zeros_like(x),
        )
        config = {
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 1,
            "num_warps": 4,
            "cross_loop_pipeline": "static",
            "block_sizes": [256, 256],
        }
        _, reference = code_and_output(scaled_gather_rms_norm, args, **config)
        doubled = x * 2
        expected = (
            doubled
            * torch.rsqrt(doubled.pow(2).mean(1, keepdim=True) + 1e-6)
            * weight
            * ((table * 0.5).sum() + table.sum(1)[torch.arange(rows) % 8, None])
        )
        torch.testing.assert_close(reference, expected, atol=1e-4, rtol=1e-4)
        # Alone, without a cluster transport to launch clusters, it falls back.
        flags = [True, False, False, False]
        code, out = code_and_output(
            scaled_gather_rms_norm, args, **config, cross_loop_dsmem=flags
        )
        self.assertNotIn("cluster_send", code)
        torch.testing.assert_close(out, reference, atol=0, rtol=0)
        flags = [True, False, True, True]
        code, out = code_and_output(
            scaled_gather_rms_norm, args, **config, cross_loop_dsmem=flags
        )
        # Each row (the first barrier's, 16 elements a task: 8 copies over 128
        # threads) to every rank of its cluster.
        for rank in range(4):
            self.assertRegex(code, rf"(?m)cluster_send\(.*, 8, 1, {rank}, 8\)$")
        torch.testing.assert_close(out, reference, atol=0, rtol=0)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_rejected_ring_falls_back(self) -> None:
        """A ring the lowering rejects regenerates the kernel without it: a user's
        synchronization a producer cannot reproduce before its loads, or no TMA
        load to stream (only the store uses a descriptor)."""
        x = torch.randn([256, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        flag = torch.ones([1], device=DEVICE, dtype=torch.int32)
        pointers = {"indexing": ["pointer", "pointer", "tensor_descriptor"]}
        for bound, extra in (
            (mm_after_flag.bind((x, y, flag)), {}),
            (mm.bind((x, y)), pointers),
        ):
            config = helion.Config(block_sizes=[128, 64, 64], **{**RING, **extra})
            self.assertNotIn("warp_specialize", bound.to_triton_code(config))

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_loops_one_after_another_share_an_accumulator_ring(self) -> None:
        """Dots of two K loops of a task, one after the other, are never live
        together: when tensor memory has no room for the second's ring it
        shares the first's, and the ring stays."""
        x = torch.randn([512, 1024], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([1024, 512], device=DEVICE, dtype=torch.bfloat16)
        z = torch.randn([1024, 512], device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            mm_two_loops,
            (x, y, z),
            # A 128x256 accumulator ring takes all 512 columns.
            block_sizes=[128, 256, 64, 64],
            range_warp_specializes=[True, None, None],
            **RING,
        )
        torch.testing.assert_close(
            out,
            2 * (x.float() @ y.float()) - x.float() @ z.float(),
            atol=2e-2,
            rtol=1e-2,
        )
        self.assertIn("warp_specialize", code)
        self.assertEqual(code.count("= _helion_gluon.acc_buffers("), 1)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_one_dot_lowering_without_a_ring(self) -> None:
        """Without a ring the dot keeps its loop-carried accumulator in tensor
        memory the same way, its operands staged from registers, each restage
        waiting for the MMA that read the last; TMA copies read in place, their
        MMA complete (the next copy reuses the memory)."""
        x = torch.randn([256, 512], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([512, 128], device=DEVICE, dtype=torch.bfloat16)
        for indexing in ("pointer", "tensor_descriptor"):
            with self.subTest(indexing=indexing):
                code, out = code_and_output(
                    mm, (x, y), block_sizes=[128, 64, 64], indexing=indexing
                )
                expected = x.float() @ y.float()
                torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
                self.assertNotIn("_helion_ring", code.split("def _helion_mm")[1])
                staged = (
                    "_helion_gluon.stage(" if indexing == "pointer" else "TRANSPOSED="
                )
                self.assertIn(staged, code)
                sync = indexing == "tensor_descriptor"
                self.assertRegex(
                    code, rf"_helion_gluon\.tmem_dot\(.*, tmem_mma, .*, {sync}\)"
                )
                if indexing == "pointer":
                    self.assertRegex(code, r"stage\(\w+, \w+, None, False, tmem_mma, ")
                # (Its barrier, set up per tile, is invalidated after the loop.)
                self.assertIn("acc_free(tmem_mma)", code)
                self.assertNotIn("tl.dot(", code)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_native_kernel(self) -> None:
        x = torch.randn([256, 512], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([512, 128], device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            mm,
            (x, y),
            block_sizes=[128, 64, 64],
            range_warp_specializes=[None, True],
            **RING,
        )
        torch.testing.assert_close(out, x.float() @ y.float(), atol=1e-2, rtol=1e-2)
        self.assertIn("@_helion_gluon.native_kernel", code)
        self.assertIn("gluon_ops import tl_namespace as tl", code)
        self.assertNotIn("@triton.jit", code)

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_publishing_root_scopes_barrier_wait(self) -> None:
        """A guarded root waiting on one root barrier and publishing another
        waits in each worker's first task, and a worker with no live task once
        before it publishes: the barrier publication follows the wait in the
        partition that publishes."""
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.float32)
        w = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        # No live task, some of the workers' tasks, every task.
        for live in (0, 16, 64):
            with self.subTest(live=live):
                rows = torch.tensor([live], device=DEVICE, dtype=torch.int32)
                code, (act, out, res) = code_and_output(
                    live_barrier_chain,
                    (x, w, rows),
                    **RING,
                    block_sizes=[32, 32, 32, 32],
                    cross_loop_pipeline="static",
                )
                expected = w.float() @ (x + 1).to(torch.bfloat16).float()
                expected[live:] = 0
                torch.testing.assert_close(act, x + 1, atol=0, rtol=0)
                torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)
                torch.testing.assert_close(
                    res, x + 1 + expected.sum(), rtol=1e-4, atol=1e-2
                )
        body = next(
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_helion_live_barrier_chain_body"
        )
        # The middle root's block: its task loop, the wait of a worker with no
        # live task, then the publication; no wait at the root's entry.
        block = next(
            node.body
            for node in ast.walk(body)
            if isinstance(node, ast.If)
            and any(
                isinstance(s, ast.For) and "w_desc" in ast.unparse(s) for s in node.body
            )
        )
        steps = {
            s: "tasks"
            if isinstance(s, ast.For)
            else "wait"
            if "root_barrier_wait" in ast.unparse(s)
            else "publish"
            for s in block
            if isinstance(s, ast.For)
            or any(
                text in ast.unparse(s) for text in ("root_barrier_wait", "atomic_add")
            )
        }
        self.assertEqual(list(steps.values()), ["tasks", "wait", "publish"])
        tasks, no_task_wait, _ = steps
        assert isinstance(tasks, ast.For) and isinstance(no_task_wait, ast.If)
        self.assertTrue(ast.unparse(no_task_wait.test).startswith("not "))
        first = [
            s
            for s in tasks.body
            if isinstance(s, ast.If) and "virtual_pid ==" in ast.unparse(s.test)
        ]
        self.assertEqual(len(first), 1)
        self.assertIn("root_barrier_wait", ast.unparse(first[0]))


if __name__ == "__main__":
    unittest.main()
