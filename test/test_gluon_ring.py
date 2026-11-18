from __future__ import annotations

import functools
import itertools
import os
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
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessGluon
from helion._testing.gluon_kernels import mm_then_mm
from helion._testing.gluon_kernels import scale_then_nested_wx
from helion._testing.gluon_kernels import scale_then_two_wx
from helion._testing.gluon_kernels import scale_then_wx
from helion._testing.gluon_kernels import scale_then_wx_and_f16_vx
from helion._testing.gluon_kernels import scale_then_wx_and_nvfp4
from helion._testing.timeline_checks import check_written
from helion._testing.triton_codegen import GLUON_RING_CONFIG as RING
from helion._testing.triton_codegen import gluon_tail_ring_config
import helion.language as hl
from helion.runtime import timeline


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
    """A dot of inputs no earlier launch writes, then a root of its output: the
    producer streams the dot's operands before the grid wait, and the next
    launch starts once a worker reaches the second root."""
    m, k = x.size()
    _, n = w.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    y = torch.empty([m, n], dtype=torch.float32, device=x.device)
    hl.pdl_wait(independent=(x, w))
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


@helion.kernel(static_shapes=True, backend="gluon")
def scale_hintprefetch_mm(
    x: torch.Tensor,
    w: torch.Tensor,
    wp: torch.Tensor,
    ids: torch.Tensor,
    routes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scale_idprefetch_mm`` reading the whole list in each task, which a last
    root rewrites (the next launch's hint list)."""
    m, k = x.size()
    _, n = w.size()
    y = torch.empty_like(x)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([m, k]):
        y[tile_r, tile_c] = x[tile_r, tile_c] * 2
    for tile_p in hl.tile(ids.size(0), block_size=1):
        hot = ids[:]
        block = torch.sum(torch.where(hl.arange(ids.size(0)) == tile_p.begin, hot, 0))
        if block >= 0:
            hl.prefetch(wp, [block, slice(None), slice(None)])
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(y[tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    for tile_s in hl.tile(ids.size(0), block_size=16):
        ids[tile_s] = routes[tile_s]
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


@helion.kernel(static_shapes=True, backend="gluon")
def ring_mm_in_dry_loop(
    x: torch.Tensor, xm: torch.Tensor, wm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A chain of narrow roots (p, q, then e after d) with a ring matmul root
    between e's anchor and e on e's workers: e's dry-pass loop runs the matmul
    (a ring root takes its ring step and accumulators from the kernel scope)."""
    batch, width = x.size()
    rows, k = xm.size()
    n = wm.size(1)
    a = torch.empty_like(x)
    p = torch.empty((batch, 4), dtype=torch.float32, device=x.device)
    q = torch.empty((batch, 4), dtype=torch.float32, device=x.device)
    d = torch.empty_like(x)
    m = torch.empty((rows, n), dtype=torch.float32, device=x.device)
    side = torch.empty((batch, 8), dtype=torch.float32, device=x.device)
    e = torch.empty((batch, 4), dtype=torch.float32, device=x.device)
    for tile_batch, tile_width in hl.tile([batch, width]):
        a[tile_batch, tile_width] = x[tile_batch, tile_width] + 1
    for row, col in hl.tile([batch, 4], block_size=[1, 1]):
        p[row, col] = torch.sum(a[row, :], dim=-1)[:, None]
    for row, col in hl.tile([batch, 4], block_size=[1, 1]):
        q[row, col] = torch.sum(p[row, :], dim=-1)[:, None]
    for tile_batch, tile_width in hl.tile([batch, width]):
        d[tile_batch, tile_width] = (
            a[tile_batch, tile_width] * torch.sum(q[tile_batch, :], dim=-1)[:, None]
        )
    for tile_m, tile_n in hl.tile([rows, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(xm[tile_m, tile_k], wm[tile_k, tile_n], acc=acc)
        m[tile_m, tile_n] = acc
    for row, col in hl.tile([batch, 8], block_size=[1, 1]):
        side[row, col] = torch.sum(a[row, :], dim=-1)[:, None]
    for row, col in hl.tile([batch, 4], block_size=[1, 1]):
        e[row, col] = torch.sum(d[row, :], dim=-1)[:, None]
    return e, m, side


def _replay(run: object, inputs: list[tuple[torch.Tensor, ...]]) -> list[object]:
    """``run`` on each input back to back in a CUDA graph, so that each launch
    overlaps the last (PDL)."""
    run(*inputs[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        results = list(itertools.starmap(run, inputs))
    graph.replay()
    return results


def _functions(code: str) -> dict[str, str]:
    """The generated module's functions, by name."""
    return {chunk.split("(", 1)[0]: chunk for chunk in code.split("\ndef ")[1:]}


class TestGluonRing(TestCase):
    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_independent_loads_precede_the_grid_wait(self) -> None:
        """``hl.pdl_wait(independent=...)``: the producer issues the first root's
        operand fill before its launch-state read (the grid wait); the trigger
        between the loops runs in the body before the second root's tasks, after
        its own wait. Back-to-back launches (PDL) stay exact."""
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
        producer = functions["_helion_mm_then_scale_producer0"]
        self.assertLess(producer.index("w_ring_desc, ("), producer.index("_launch("))
        # Fills of what is not independent (the written ``out``, ``b``) stay
        # after the read, and only the body triggers.
        for name, function in functions.items():
            if "_producer" in name:
                self.assertNotIn("gdc_launch_dependents", function)
                for fill in ("out_ring_desc, (", "b_ring_desc, ("):
                    if fill in function:
                        self.assertLess(
                            function.index("_launch("), function.index(fill)
                        )
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
                    "prefetch code here" if specialize else "prefetch_code()"
                )
                self.assertLess(at, prefetching.index("_launch("))
                self.assertIn("return (tile_dependency_epoch,)", body)
                self.assertNotIn("def _helion_scale_then_mm_claimer(", code)

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
    def test_hint_list_rewrite_waits_for_no_read(self) -> None:
        """A prefetch-only root's read of a list a later root rewrites is a hint:
        the rewrite need not wait for it (the read may see either list)."""
        x, w, _, _ = _prefetch_reference()
        ids = torch.tensor([3, -1, 7, 0] * 4, device=DEVICE, dtype=torch.int32)
        routes = torch.arange(16, device=DEVICE, dtype=torch.int32)
        args = (w.view(16, 64, 256), ids, routes)
        bound = scale_hintprefetch_mm.bind((x, w, *args))
        graph = bound.host_function.device_ir.tile_dependency_graph
        # Roots: the scale, the prefetches, the matmul, the rewrite.
        self.assertFalse(
            [e for e in graph.edges if 1 in (e.producer_root, e.consumer_root)]
        )
        code = self._prefetch_case(scale_hintprefetch_mm, args)
        torch.testing.assert_close(ids, routes, atol=0, rtol=0)
        self._assert_prefetch_in_body(code, "scale_hintprefetch_mm")

    @skipIfRefEager("checks generated code")
    @skipUnlessGluon()
    def test_static_dry_pass_runs_ring_roots_in_its_loop(self) -> None:
        """Under the dynamic pipeline a ring kernel's workers claim tasks root
        by root, so a waiting root can still take a dry pass (static
        ``cross_loop_dry_passes``); a ring matmul on its workers between its
        anchor and it runs in the dry-pass loop."""
        workers = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        x = torch.randn([2, 64 * workers], device=DEVICE, dtype=torch.float32)
        xm = torch.randn([128 * workers, 64], device=DEVICE, dtype=torch.bfloat16)
        wm = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        a = x + 1
        d = a * (16 * a.sum(-1, keepdim=True))
        with self.assertLogs("helion._compiler.cross_loop_codegen", "INFO") as logs:
            code, (e, m, side) = code_and_output(
                ring_mm_in_dry_loop,
                (x, xm, wm),
                block_sizes=[1, 64, 1, 64, 128, 64, 64],
                pid_type="persistent_blocked",
                cross_loop_pipeline="dynamic",
                num_sm_multiplier=1,
                num_warps=4,
                num_stages=2,
                indexing="tensor_descriptor",
                # Roots 5 and 6, after root 3 (producers two levels deep).
                cross_loop_dry_passes=[False, True],
            )
        torch.testing.assert_close(
            e, d.sum(-1, keepdim=True).expand(2, 4), rtol=1e-4, atol=1e-1
        )
        torch.testing.assert_close(m, xm.float() @ wm.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(side, a.sum(-1, keepdim=True).expand(2, 8))
        messages = {line.split(":", 2)[2] for line in logs.output}
        self.assertTrue(
            any(message.startswith("Root 6 runs a dry pass") for message in messages),
            messages,
        )
        self.assertIn("tile_dependency_dry_trip", code)

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
                self.assertIn("ring_barriers(_helion_ring_S, 2)", code)
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
        leading box dim), two weights in one loop (the first dot takes the
        tail), a K loop inside another's body (the outer one takes the tail;
        two accumulator rings, or a nested loop's, leave fewer columns), and an
        nvfp4 dot's block scales beside a tail (10 tiles: the whole of tensor
        memory but 16 columns)."""
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
        self._check_tails(
            [
                (scale_then_ewx, (x, stacked), ring, 1, 14),
                (scale_then_w13x, (x, w, w3), ring, 1, 0),
                (scale_then_nested_wx, (x, w, v), nested_ring, 1, 0),
                (scale_then_wx_and_nvfp4, nvfp4, nvfp4_ring, 1, 10),
            ]
        )

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


if __name__ == "__main__":
    unittest.main()
