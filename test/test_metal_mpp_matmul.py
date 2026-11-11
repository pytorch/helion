"""Cooperative MPP lowering of K-loop matmul accumulators.

Every ``torch.addmm``, ``torch.mm``, ``torch.bmm`` and ``torch.baddbmm``,
and every ``hl.dot`` the planner admits (a K loop's, one over a K held as
per-thread parts, or every whole-K ``hl.dot`` of a kernel where each is
admissible), take ``metal/mpp_matmul.py``'s path.  Most kernels run their K loop under
``hl.grid`` or over free ``hl.arange`` rows (grouped_gemm's layout).
References are float64 on the CPU.
"""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.metal import mpp_matmul
from helion._compiler.metal import tile_sharing
from helion._testing import import_example
import helion.language as hl

attention = import_example("attention").attention
helion_gdn_fwd_h = import_example("gdn_fwd_h").helion_gdn_fwd_h
matmul_bf16_int4 = import_example("int4_gemm").matmul_bf16_int4
_examples_mamba2_chunk_scan = import_example("mamba2_chunk_scan")
helion_mamba2_chunk_scan_kernel = (
    _examples_mamba2_chunk_scan.helion_mamba2_chunk_scan_kernel
)
ref_chunk_scan = _examples_mamba2_chunk_scan.ref_chunk_scan
_examples_mamba2_chunk_state = import_example("mamba2_chunk_state")
helion_mamba2_chunk_state_kernel = (
    _examples_mamba2_chunk_state.helion_mamba2_chunk_state_kernel
)
ref_chunk_state = _examples_mamba2_chunk_state.ref_chunk_state
matmul = import_example("matmul").matmul

if TYPE_CHECKING:
    from collections.abc import Callable

DEVICE = "mps"
S = {"backend": "metal", "static_shapes": True}
TILE16 = helion.Config(block_sizes=[16, 16, 16], num_threads=[16, 16, 1])
TILE32 = helion.Config(block_sizes=[32, 32, 16], num_threads=[32, 32, 1])
FREE32 = helion.Config(block_sizes=[32, 32, 16])

# Metal runs on macOS only: CI's Linux jobs skip the whole file.
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Metal tests require macOS"
)

_requires_mps = unittest.skipUnless(
    sys.platform == "darwin" and torch.backends.mps.is_available(),
    "requires a Metal GPU",
)


def _full_threadgroup(test: Callable[..., None]) -> Callable[..., None]:
    """For a test that pins the layout of a full 1024-thread threadgroup:
    configs that leave ``num_warps`` unset take 32 instead of the default 4,
    whose 128-thread budget walks more of each block in lanes."""
    return patch("helion.autotuner.config_spec.DEFAULT_NUM_WARPS", 32)(test)


def _run(
    kernel: helion.Kernel, args: tuple[object, ...], config: helion.Config | None
) -> tuple[torch.Tensor, str, str]:
    """Compile and run ``kernel`` at ``config`` (None: its default config);
    return its output, its generated Python and the MSL it compiled."""
    from torch._inductor.codecache import PyCodeCache

    from helion._compiler.metal.metal_jit import _MetalKernel

    bound = kernel.bind(args)
    code = bound.to_code(config or bound.config_spec.default_config())
    module = PyCodeCache.load(code)
    out = getattr(module, kernel.fn.__name__)(*args)
    torch.mps.synchronize()
    (msl,) = [
        obj.msl_source
        for obj in vars(module).values()
        if isinstance(obj, _MetalKernel) and obj.msl_source is not None
    ]
    return out, code, msl


def _element_loops(msl: str) -> str:
    """The MSL from the first element loop on."""
    return msl[msl.index("is_valid_element") :]


def _mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.cpu().double() @ y.cpu().double()


def _close(out: torch.Tensor, ref: torch.Tensor) -> None:
    torch.testing.assert_close(out.cpu().double(), ref, atol=2e-2, rtol=2e-2)


def _tiled(values: torch.Tensor, block: int) -> torch.Tensor:
    """Each element replaced by the sum of its ``block``-wide tile."""
    sums = torch.stack([t.sum() for t in values.double().split(block)])
    return sums.repeat_interleave(block)[: values.numel()]


# --- Kernels in grouped_gemm's free layout --------------------------------


@helion.kernel(**S)
def free_rows(a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


free_rows_dynamic = helion.kernel(free_rows.fn, backend="metal", static_shapes=False)


@helion.kernel(backend="metal", static_shapes=False)
def addmm_sized(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def free_rows_if(
    a: torch.Tensor, b: torch.Tensor, sizes: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            if sizes[gm] > 0:
                rows = gm * bm + hl.arange(bm)
                cols = gn * bn + hl.arange(bn)
                rv = rows < m
                cv = cols < n
                acc = hl.zeros([bm, bn], dtype=torch.float32)
                for tk in hl.tile(k):
                    a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None])
                    b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                    acc = torch.addmm(acc, a_blk, b_blk)
                hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def masked_rows(
    a: torch.Tensor, b: torch.Tensor, lim: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """The lhs rows are masked by ``rows < lim[0]``."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            c0 = hl.load(lim, [0])
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=(rows < c0)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def masked_rows_kvary(
    a: torch.Tensor, b: torch.Tensor, lims: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """The lhs rows are masked by a bound loaded on each K tile."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                ck = hl.load(lims, [tk.begin])
                a_blk = hl.load(a, [rows, tk.index], extra_mask=(rows < ck)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def masked_rows_literal(
    a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(
                    a, [rows, tk.index], extra_mask=(rows < -(2**31) - 3)[:, None]
                )
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def masked_rows_min_literal(
    a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(
                    a, [rows, tk.index], extra_mask=(rows < -(2**63))[:, None]
                )
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def shifted_rows(
    a: torch.Tensor,
    b: torch.Tensor,
    off: torch.Tensor,
    lim: torch.Tensor,
    nbm: int,
    nbn: int,
) -> torch.Tensor:
    """The lhs rows are ``off[0] + gm * bm + arange``, masked by ``< lim[0]``;
    the store keeps the unshifted rows."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            o = hl.load(off, [0])
            c = hl.load(lim, [0])
            ar = hl.arange(bm)
            rows = gm * bm + ar
            rows_a = (o + gm * bm) + ar
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows_a, tk.index], extra_mask=(rows_a < c)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def bare_iota_rows(a: torch.Tensor, b: torch.Tensor, cnt: torch.Tensor) -> torch.Tensor:
    """The lhs index is a bare ``hl.arange``, masked by ``ar < cnt[0]``."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for _g in hl.grid(1):
        for _h in hl.grid(1):
            ar = hl.arange(bm)
            ac = hl.arange(bn)
            c0 = hl.load(cnt, [0])
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [ar, tk.index], extra_mask=(ar < c0)[:, None])
                b_blk = hl.load(b, [tk.index, ac], extra_mask=(ac < n)[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(
                out, [ar, ac], acc, extra_mask=(ar < m)[:, None] & (ac < n)[None, :]
            )
    return out


@helion.kernel(**S)
def base_if(
    a: torch.Tensor,
    b: torch.Tensor,
    off: torch.Tensor,
    flag: torch.Tensor,
    nbm: int,
    nbn: int,
) -> torch.Tensor:
    """An ``if`` reassigns the lhs base between the index and the K loop."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            base = hl.load(off, [0]) + gm * bm
            f = hl.load(flag, [0])
            rows = base + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            if f > 0:
                base = base + 8
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def bound_after(
    a: torch.Tensor, b: torch.Tensor, lim: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """A loop after the K loop reassigns the bound of the store's mask."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            bnd = hl.load(lim, [0])
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < bnd
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=(rows < m)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            for _j in hl.grid(3):
                bnd = bnd - 10
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def iota_data(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The row and column indices are epilogue data; the store is shifted."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m + 3, n + 5], dtype=torch.float32, device=x.device)
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    for gm, gn in hl.grid([(m + bm - 1) // bm, (n + bn - 1) // bn]):
        rows = gm * bm + hl.arange(bm)
        cols = gn * bn + hl.arange(bn)
        rv = rows < m
        cv = cols < n
        acc = hl.zeros([bm, bn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = hl.load(x, [rows, tk], extra_mask=rv[:, None])
            b = hl.load(y, [tk, cols], extra_mask=cv[None, :])
            acc = torch.addmm(acc, a, b)
        val = (
            acc
            + rows.to(torch.float32)[:, None] * 0.25
            - cols.to(torch.float32)[None, :]
        )
        hl.store(out, [rows + 3, cols + 5], val, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def mask_local(
    a: torch.Tensor, b: torch.Tensor, cnt: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """The lhs mask bounds the bare iota under a based index."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            ar = hl.arange(bm)
            rows = gm * bm + ar
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            c0 = hl.load(cnt, [0])
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=(ar < c0)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def mask_shifted(a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int) -> torch.Tensor:
    """The lhs mask bounds the iota with another base."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            ar = hl.arange(bm)
            rows = gm * bm + ar
            shifted = (gm * bm + 8) + ar
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=(shifted < m)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def mask_recomputed(
    a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """The lhs mask recomputes the row index from a second ``hl.arange``."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            rows2 = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=(rows2 < m)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(
                out, [rows, cols], acc, extra_mask=(rows < m)[:, None] & cv[None, :]
            )
    return out


@helion.kernel(**S)
def mask_halfrows(a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int) -> torch.Tensor:
    """The lhs mask's bound ``m / 2`` is a Python float for an odd m."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=(rows < m / 2)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def mask_le(
    a: torch.Tensor, b: torch.Tensor, cnt: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """The lhs mask is ``rows <= c0``, which keeps row c0."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            c0 = hl.load(cnt, [0])
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=(rows <= c0)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def base_vec(a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int) -> torch.Tensor:
    """The lhs reads rows ``(gm * bm + i) + i``: the base added to the iota
    is itself per-row."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            ar = hl.arange(bm)
            first = gm * bm + ar
            rows = first + ar
            cols = gn * bn + hl.arange(bn)
            fv = first < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=(rows < m)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [first, cols], acc, extra_mask=fv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def bound_vec(
    a: torch.Tensor, b: torch.Tensor, lims: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """The lhs mask's bound is a per-row tensor."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            lim = hl.load(lims, [rows], extra_mask=rv)
            rl = rows < lim
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rl[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def alpha_iota(
    a: torch.Tensor, b: torch.Tensor, off: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """The lhs rows are ``torch.add(base, ar, alpha=2)``, base + 2 * ar."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            o = hl.load(off, [0])
            ar = hl.arange(bm)
            rows = gm * bm + ar
            rows_a = torch.add(o + gm * bm, ar, alpha=2)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows_a, tk.index], extra_mask=(rows_a < m)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def alpha_base(
    a: torch.Tensor, b: torch.Tensor, off: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """The lhs rows are ``torch.add(ar, base, alpha=2)``, ar + 2 * base."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            o = hl.load(off, [0])
            ar = hl.arange(bm)
            rows = gm * bm + ar
            rows_a = torch.add(ar, o + gm * bm, alpha=2)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows_a, tk.index], extra_mask=(rows_a < m)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


# --- Kernels with tiles under hl.grid -------------------------------------


@helion.kernel(**S)
def plain_g(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def epilogue(
    x: torch.Tensor, y: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        v = torch.sigmoid(acc * 0.5 + b[tn]) + c[tm, tn]
        out[tm, tn] = v.to(torch.float16)
    return out


@helion.kernel(**S)
def two_groups(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            a1 = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                a1 = torch.addmm(a1, x[tm, tk], y[tk, tn])
            a2 = hl.full([tm, tn], 1.0, dtype=torch.float32)
            for tk in hl.tile(k):
                a2 = torch.addmm(a2, x[tm, tk], z[tk, tn])
            out1[tm, tn] = a1
            out2[tm, tn] = torch.relu(a2)
    return out1


@helion.kernel(**S)
def counter_beside(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A 0-d counter of K tiles carried beside the op."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            cnt = hl.zeros([], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                cnt = cnt + 1.0
            out[tm, tn] = acc + cnt
    return out


@helion.kernel(**S)
def stale_loop(x: torch.Tensor, y: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    """``r = acc * s``, then a loop reassigns s, then r is stored."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            s = hl.load(sc, [0])
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            r = acc * s
            for _j in hl.grid(3):
                s = s + 1.0
            out[tm, tn] = r
    return out


@helion.kernel(**S)
def stale_if(
    x: torch.Tensor, y: torch.Tensor, sc: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    """``r = acc * s``, then an ``if`` reassigns s, then r is stored."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            s = hl.load(sc, [0])
            f = hl.load(flag, [0])
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            r = acc * s
            if f > 0:
                s = s + 5.0
            out[tm, tn] = r
    return out


@helion.kernel(**S)
def stale_kcarry(
    x: torch.Tensor, y: torch.Tensor, sc: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """``e = c * s`` before the K loop, which carries s."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            s = hl.load(sc, [0])
            e = c[tm, tn] * s
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                s = s + 1.0
            out[tm, tn] = acc + e
    return out


@helion.kernel(**S)
def red_n_after(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            s = bias[tn].sum()
            out[tm, tn] = acc + s
    return out


@helion.kernel(**S)
def red_epi_before(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            s = bias[tn].sum()
            e = c[tm, tn] * s
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            for _j in hl.grid(2):
                s = s + 1.0
            out[tm, tn] = acc + e + s
    return out


@helion.kernel(**S)
def red_load_index(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, jidx: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            j = jidx[tn].sum()
            out[tm, tn] = acc + c[tm, j][:, None]
    return out


@helion.kernel(**S)
def red_jloop_acc(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([2, m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            s = hl.zeros([], dtype=torch.float32)
            for j in hl.grid(2):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                s = s + bias[tn].sum()
                out[j, tm, tn] = acc + s
    return out


@helion.kernel(**S)
def outer_epi(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([2, m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            s = bias[tn].sum()
            e = c[tm, tn] * s
            for j in hl.grid(2):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                out[j, tm, tn] = acc + e
    return out


@helion.kernel(**S)
def outer_epi_if(
    x: torch.Tensor,
    y: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    flag: torch.Tensor,
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            s = bias[tn].sum()
            e = c[tm, tn] * s
            f = hl.load(flag, [0])
            if f > 0:
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                out[tm, tn] = acc + e
    return out


@helion.kernel(**S)
def epi_ok(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    b: torch.Tensor,
    bc: torch.Tensor,
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            v = acc + c[tm, tn] + b[tn][None, :] + bc[tm][:, None] + b[tn]
            out[tm, tn] = torch.where(c[tm, tn] > 0, v, 0.0)
    return out


@helion.kernel(**S)
def k_begin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(8, k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def m_short(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m - 8, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def batch_threaded(x: torch.Tensor, y: torch.Tensor, tag: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    nb = tag.size(0)
    out = torch.zeros([nb, m, n], dtype=torch.float32, device=x.device)
    for tb, tm, tn in hl.tile([nb, m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tb, tm, tn] = acc[None, :, :]
    return out


@helion.kernel(**S)
def sibling17(x: torch.Tensor, y: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """A 17-wide sibling root: 17 x 16 threads, not whole SIMD groups."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out2 = torch.empty_like(r)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n], block_size=[16, 16]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    for tr in hl.tile(r.size(0), block_size=17):
        out2[tr] = r[tr] + 1.0
    return out


@helion.kernel(**S)
def carry_vary(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """A second, per-element carry read after the K loop."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            s = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                s = s + c[tm, tn]
            out[tm, tn] = acc + s + bias[tn]
    return out


@helion.kernel(**S)
def carry_vary_in(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """A second, per-element carry read inside the K loop."""
    m, k = x.size()
    _, n = y.size()
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            s = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                s = s + c[tm, tn]
                out[tm, tn] = acc + s
    return out


@helion.kernel(**S)
def stale_seed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A value derived from the seed before the K loop, stored after it."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            s = acc + 1.0
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc + s
    return out


@helion.kernel(**S)
def delta_in_loop(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The pre-update carry read after the addmm."""
    m, k = x.size()
    _, n = y.size()
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                old = acc
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                out[tm, tn] = acc - old
    return out


@helion.kernel(**S)
def outer_grid_carry(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """An outer loop carries the accumulator too."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for _g in hl.grid(2):
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def if_output_vary(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    """The K loop runs inside an ``if`` whose outputs are read after it."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for g in hl.grid(1):
        f = flag[g]
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            s = hl.zeros([tm, tn], dtype=torch.float32)
            if f > 0:
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                    s = s + c[tm, tn]
            out[tm, tn] = acc + s
    return out


@helion.kernel(**S)
def readback(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor, out2: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc + bias[tn]
            out2[tm, tn] = out[tm, tn] * 2.0
    return out


@helion.kernel(**S)
def prestore(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, tmp: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            tmp[tm, tn] = c[tm, tn] * 2.0
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc + tmp[tm, tn]
    return out


@helion.kernel(**S)
def gather_written_index(
    x: torch.Tensor,
    y: torch.Tensor,
    perm: torch.Tensor,
    rows: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """An epilogue operand gathered at an index read from a tensor the kernel
    writes."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            rows[tm] = perm[tm]
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc + c[rows[tm], tn]
    return out


@helion.kernel(**S)
def load_under_written_mask(
    x: torch.Tensor,
    y: torch.Tensor,
    flag_src: torch.Tensor,
    flag: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """An epilogue operand loaded under a mask read from a tensor the kernel
    writes."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            flag[tm] = flag_src[tm]
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            keep = (flag[tm] > 0)[:, None]
            out[tm, tn] = acc + hl.load(c, [tm, tn], extra_mask=keep)
    return out


@helion.kernel(**S)
def view_written(x: torch.Tensor, y: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            for tj in hl.tile(k):
                x2[tm, tj] = x[tm, tj] * 2.0
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x2[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def computed_operand(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk] * 2, y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def gathered_lhs(x: torch.Tensor, y: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[rows[tm], tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def acc_from_load(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = c[tm, tn] + 0.0
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def row_sum_after(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm in hl.tile(m):
            for tn in hl.tile(n, block_size=n):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                out[tm] = acc.sum(-1)
    return out


@helion.kernel(**S)
def k_lane_scalar(x: torch.Tensor, y: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
    """Lane-variant Scalar work in the K loop beside the op."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                cp[tm, tk] = x[tm, tk]
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def swapped_store(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([n, m], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tn, tm] = acc
    return out


@helion.kernel(**S)
def swapped_epi_load(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc + c[tn, tm]
    return out


@helion.kernel(**S)
def bias_row_swapped(x: torch.Tensor, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc + b[tm][None, :]
    return out


@helion.kernel(**S)
def where_swapped(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = torch.where(c[tn, tm] > 0, acc, 0.0)
    return out


@helion.kernel(**S)
def epi_mask_swapped(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, lim: torch.Tensor
) -> torch.Tensor:
    """An epilogue load masked by a vector over M at N's position."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            lv = lim[tm] > 0
            e = hl.load(c, [tm, tn], extra_mask=lv[None, :])
            out[tm, tn] = acc + e
    return out


# --- Staged operands and a K held as per-thread parts ---------------------


int4_gemm = helion.kernel(matmul_bf16_int4.fn, backend="metal", static_shapes=True)


@helion.kernel(**S)
def computed_rhs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], torch.exp(y[tk, tn]))
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def both_computed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, torch.sigmoid(x[tm, tk]), y[tk, tn] * 2)
            out[tm, tn] = acc + 1.0
    return out


@helion.kernel(**S)
def both_staged_kshort(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Both operands staged, under a K loop that ends 5 before the tensors'."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k - 5):
                acc = torch.addmm(
                    acc, torch.exp(x[tm, tk] * 0.1), torch.exp(y[tk, tn] * 0.1)
                )
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def staged_kbegin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A staged lhs beside a view, under a K loop that starts at 5."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(5, k):
                acc = torch.addmm(acc, torch.exp(x[tm, tk] * 0.1), y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_lhs(x3: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _, m, k = x3.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x3.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x3[2, tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_lhs_computed(x3: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _, m, k = x3.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x3.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x3[2, tm, tk] * 2.0, y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def free_rhs_computed(
    a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=(rows < m)[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=(cols < n)[None, :])
                acc = torch.addmm(acc, a_blk, b_blk * 0.5)
            mask = (rows < m)[:, None] & (cols < n)[None, :]
            hl.store(out, [rows, cols], acc, extra_mask=mask)
    return out


@helion.kernel(**S)
def counter_staged(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A 0-d counter beside a staged op: it counts K tiles, not K lanes."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            cnt = hl.zeros([], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk] * 2, y[tk, tn])
                cnt = cnt + 1.0
            out[tm, tn] = acc + cnt
    return out


@helion.kernel(**S)
def lane_carry_staged(
    x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """A carry over the K block beside a staged op, folded after the loop."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    bk = hl.register_block_size(k)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        carry = hl.zeros([bk], dtype=torch.float32)
        for tk in hl.tile(k, block_size=bk):
            acc = torch.addmm(acc, x[tm, tk], torch.exp(y[tk, tn] * 0.1))
            carry = carry + v[tk]
        out[tm, tn] = acc + carry.sum()
    return out


@helion.kernel(**S)
def lane_carry_operand(
    x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """A carry over the K block that the staged lhs reads element-wise."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    bk = hl.register_block_size(k)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        carry = hl.zeros([bk], dtype=torch.float32)
        for tk in hl.tile(k, block_size=bk):
            carry = carry + v[tk]
            acc = torch.addmm(acc, x[tm, tk] * carry[None, :], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def two_k_loops_staged(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], torch.exp(y[tk, tn]))
            out[tm, tn] = acc
            acc2 = hl.zeros([tm, tn], dtype=torch.float32)
            for tk2 in hl.tile(k):
                acc2 = torch.addmm(acc2, x[tm, tk2], z[tk2, tn] * 3)
            out2[tm, tn] = acc2
    return out, out2


@helion.kernel(**S)
def staged_sibling_cols(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """A [16, 32] sibling widens N's thread axis to 32 beside an N block of
    16: a surplus thread's column would land in the staged rhs tile's next
    K row."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out2 = torch.empty_like(z)
    for tm, tn in hl.tile([m, n], block_size=[16, 16]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k, block_size=16):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn] * 3)
        out[tm, tn] = acc
    for ti, tj in hl.tile([z.size(0), z.size(1)], block_size=[16, 32]):
        out2[ti, tj] = z[ti, tj] * 2
    return out


@helion.kernel(**S)
def stacked_into_dot(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A merged stack (staged as parts) beside a widened rhs view."""
    m, kh = x.shape
    n = w.size(1)
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    bk = hl.register_block_size(kh)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(kh, block_size=bk):
            lhs = torch.stack([x[tm, tk], x[tm, tk] * 2.0], dim=2).reshape(
                tm.block_size, tk.block_size * 2
            )
            s = 2 * tk.begin
            acc = hl.dot(lhs, w[s : s + 2 * bk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_parts_c3(
    a: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor
) -> torch.Tensor:
    """A K widened by 3: three stacked rhs parts."""
    m, k3 = a.shape
    _, n = b0.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k3 // 3)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k3 // 3, block_size=bk):
            b = torch.stack([b0[tk, tn], b1[tk, tn], b2[tk, tn] * 2], dim=1).reshape(
                tk.block_size * 3, tn.block_size
            )
            acc = hl.dot(a[tm, tk.begin * 3 : tk.begin * 3 + bk * 3], b, acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def addmm_parts(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``torch.addmm`` of two widened views."""
    m, k2 = a.shape
    _, n = w.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k2 // 2)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k2 // 2, block_size=bk):
                s = 2 * tk.begin
                acc = torch.addmm(acc, a[tm, s : s + 2 * bk], w[s : s + 2 * bk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def parts_exp_rhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A widened lhs view beside a stacked rhs that is non-zero past K."""
    m, k2 = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k2 // 2)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k2 // 2, block_size=bk):
            bt = b[tk, tn]
            b_stacked = torch.stack(
                [torch.exp(bt * 0.1), torch.sigmoid(bt)], dim=1
            ).reshape(bk * 2, tn.block_size)
            acc = hl.dot(
                a[tm, tk.begin * 2 : tk.begin * 2 + bk * 2], b_stacked, acc=acc
            )
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def parts_kbegin(a: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    """``parts_exp_rhs`` under a K loop that starts at 3."""
    m, _ = a.shape
    k, n = b0.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(3, k, block_size=bk):
            b = torch.stack(
                [torch.exp(b0[tk, tn] * 0.1), torch.sigmoid(b1[tk, tn])], dim=1
            ).reshape(bk * 2, tn.block_size)
            acc = hl.dot(a[tm, tk.begin * 2 : tk.begin * 2 + bk * 2], b, acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def parts_row_masked(
    a: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor, lim: int
) -> torch.Tensor:
    """A widened lhs load masked on its rows, so staged as parts."""
    m, _ = a.shape
    k, n = b0.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k, block_size=bk):
            idx = hl.arange(tk.begin * 2, tk.begin * 2 + bk * 2)
            lhs = hl.load(a, [tm, idx], extra_mask=(tm.index < lim)[:, None])
            b = torch.stack([b0[tk, tn], b1[tk, tn]], dim=1).reshape(
                bk * 2, tn.block_size
            )
            acc = hl.dot(lhs, b, acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_two_widened(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    _, n = w.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k2 // 2)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k2 // 2, block_size=bk):
            s = 2 * tk.begin
            acc = hl.dot(a[tm, s : s + 2 * bk], w[s : s + 2 * bk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def addmm_two_widened_root(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    _, n = w.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k2 // 2)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k2 // 2, block_size=bk):
            s = 2 * tk.begin
            acc = torch.addmm(acc, a[tm, s : s + 2 * bk], w[s : s + 2 * bk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def full_lhs_parts(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A constant lhs over a K of 2 * bk, which no thread layout holds."""
    m, k2 = a.shape
    _, n = w.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k2 // 2)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k2 // 2, block_size=bk):
            s = 2 * tk.begin
            lhs = hl.full([tm, 2 * bk], 1.0, dtype=torch.float32)
            acc = hl.dot(lhs, w[s : s + 2 * bk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_parts_noacc(
    a: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor
) -> torch.Tensor:
    m, k2 = a.shape
    _, n = b0.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k2 // 2)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k2 // 2, block_size=bk):
            b = torch.stack([b0[tk, tn], b1[tk, tn]], dim=1).reshape(
                bk * 2, tn.block_size
            )
            acc = acc + hl.dot(a[tm, tk.begin * 2 : tk.begin * 2 + bk * 2], b)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_parts_out_dtype(
    a: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor
) -> torch.Tensor:
    m, k2 = a.shape
    _, n = b0.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k2 // 2)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k2 // 2, block_size=bk):
            b = torch.stack([b0[tk, tn], b1[tk, tn]], dim=1).reshape(
                bk * 2, tn.block_size
            )
            a_tile = a[tm, tk.begin * 2 : tk.begin * 2 + bk * 2]
            acc = hl.dot(a_tile, b, acc=acc, out_dtype=torch.float16)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def parts_arange_base(
    a: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor
) -> torch.Tensor:
    """A K index ``2 * begin + hl.arange(2 * bk)``: a range over no block."""
    m, k2 = a.shape
    _, n = b0.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k2 // 2)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k2 // 2, block_size=bk):
            b = torch.stack([b0[tk, tn], b1[tk, tn]], dim=1).reshape(
                bk * 2, tn.block_size
            )
            a_tile = hl.load(a, [tm, tk.begin * 2 + hl.arange(bk * 2)])
            acc = hl.dot(a_tile, b, acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def swapped_staged(x: torch.Tensor, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A staged lhs ``x[tm, tk] + z[tk, tm]``, which meets M and K swapped."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk] + z[tk, tm], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def split_reorder(x: torch.Tensor, y: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
    """A per-lane store, then a K-lane-invariant load of what lane 0 stored:
    the split must not run the load before the stores."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                cp[tm, tk] = x[tm, tk] * 3
                first = cp[tm, tk.begin]
                acc = torch.addmm(acc, x[tm, tk] * first[:, None], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def cross_lane(x: torch.Tensor, y: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
    """A per-lane staged value, also stored after the op."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                v = x[tm, tk] * 2
                acc = torch.addmm(acc, v, y[tk, tn])
                cp[tm, tk] = v
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def two_ops_staged(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            acc2 = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk] * 2, y[tk, tn])
                acc2 = torch.addmm(acc2, x[tm, tk], y[tk, tn] * 0.5)
            out[tm, tn] = acc
            out2[tm, tn] = acc2
    return out, out2


@helion.kernel(**S)
def stacked_fp8(
    a0: torch.Tensor, a1: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor
) -> torch.Tensor:
    """Both operands staged, so no view's dtype check sees them."""
    m, k = a0.shape
    _, n = w0.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=w0.device)
    bk = hl.register_block_size(k)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k, block_size=bk):
            a = torch.stack([a0[tm, tk], a1[tm, tk]], dim=2).reshape(
                tm.block_size, bk * 2
            )
            w = torch.stack([w0[tk, tn], w1[tk, tn]], dim=1).reshape(
                bk * 2, tn.block_size
            )
            acc = hl.dot(a, w, acc=acc)
        out[tm, tn] = acc
    return out


# --- bfloat operands staged inside a K loop -------------------------------


@helion.kernel(**S)
def kloop_bcast(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``d[tm]``, loaded before the K loop, broadcast over each K tile."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (dt[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_exp(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            a = torch.exp(dt[:, None] - 1.0 + z).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_exp_pre(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        et = torch.exp(d[tm] - 1.0)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (et[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_carry_id(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A K-loop carry whose update is the identity."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        c = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (c[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
            c = c * 1.0
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_chain(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        t = torch.sigmoid(dt) * dt + 0.5
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (t[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_tile_index(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            rows = tm.index.to(torch.float32)
            a = (rows[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_outer_begin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A load the K loop reloads, plus ``tm.begin``, which it doesn't."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (x[tm, tk].to(torch.float32) * 0.0 + tm.begin).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_full_scalar(
    x: torch.Tensor, y: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        v = s[0]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = hl.full([tm, tk], v, dtype=torch.float32).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_load_exp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = torch.exp(x[tm, tk].to(torch.float32) * 0.5).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_load_chain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            xf = x[tm, tk].to(torch.float32)
            a = (torch.sigmoid(xf) * xf + 0.25).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_const(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (hl.zeros([tm, tk], dtype=torch.float32) + 1.5).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_load0(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A load the compiler may fold away, leaving a constant."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (x[tm, tk].to(torch.float32) * 0.0 + 1.5).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_load_inv(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``d[tm]`` loaded inside the K loop, at an address the loop doesn't
    change."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            a = (d[tm][:, None] + z).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_carry_zero(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A K-loop carry seeded by a constant fill."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        c = hl.zeros([tm], dtype=torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (c[:, None] + x[tm, tk].to(torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
            c = c + 1.0
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_between_loops(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """``d[tm]`` loaded in an ``hl.grid`` loop, before the K loop inside it."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for _g in hl.grid(2):
            dt = d[tm]
            for tk in hl.tile(k):
                z = hl.zeros([tm, tk], dtype=torch.float32)
                acc = torch.addmm(acc, (dt[:, None] + z).to(x.dtype), y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_arg_scale(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (x[tm, tk].to(torch.float32) * alpha).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_block_size(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (x[tm, tk].to(torch.float32) + tk.block_size).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


# --- bfloat operands staged in float32 inside a K loop --------------------


@helion.kernel(**S)
def kloop_carry_vary(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A K-loop carry halved every iteration."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        c = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (c[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
            c = c * 0.5
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_newvar(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """The conversion reaches the op through a plain assignment."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            e = torch.exp(dt[:, None] - 1.0 + z).to(x.dtype)
            a = e
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_cvt_store(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    """The converted value is also stored to a bfloat tensor."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            e = torch.exp(dt[:, None] - 1.0 + z).to(x.dtype)
            side[tm, tk] = e
            acc = torch.addmm(acc, e, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def two_cvt(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Two conversions on one chain, with float math between them."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            e = torch.exp(dt[:, None] - 1.0 + z).to(x.dtype)
            a = (e.to(torch.float32) * 3.0 + 0.25).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def half_sub(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A float16 conversion on the chain, of float16 subnormals, which
    float16 and bfloat round far apart."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            h = (dt[:, None] * x[tm, tk].to(torch.float32) * 1e-6).to(torch.float16)
            a = (h.to(torch.float32) * 1e6).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def cvt_subscript(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A conversion of a row, broadcast over the K tile by subscript."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            e = torch.exp(dt - 1.0).to(x.dtype)
            a = (e[:, None] + z).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def cvt_float_mix(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A conversion read back as float and added to a load, converted again."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            e = torch.exp(dt[:, None] - 1.0 + z).to(x.dtype)
            a = (e + x[tm, tk].to(torch.float32) * 0.5).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_symf(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, s: float
) -> torch.Tensor:
    """A float kernel argument in the float math before the conversion."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            a = (dt[:, None] * s + z).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def store_load_pre32(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A bfloat conversion stored to a float scratch and loaded back before
    the K loop: the load ends the planner's walk."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    tmp = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        tmp[tm] = torch.exp(d[tm] - 1.0).to(x.dtype).to(torch.float32)
        eb = tmp[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            a = (eb[:, None] + z).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def parts_bcast(x: torch.Tensor, w: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A merged stack, staged as per-thread parts, of a row loaded before
    the K loop."""
    m, kh = x.shape
    n = w.size(1)
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    bk = hl.register_block_size(kh)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(kh, block_size=bk):
            a = (dt[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            lhs = torch.stack([a, a * 2.0], dim=2).reshape(
                tm.block_size, tk.block_size * 2
            )
            s = 2 * tk.begin
            acc = hl.dot(lhs, w[s : s + 2 * bk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def loop_pre_cvt(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """A conversion inside an earlier loop, its float result read in the K
    loop."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        ef = hl.zeros([tm], dtype=torch.float32)
        for _tj in hl.tile(c.size(0)):
            ef = ef + torch.exp(d[tm] - 1.0).to(x.dtype).to(torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (ef[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def loop_pre_nocvt(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """``loop_pre_cvt`` with no conversion in the earlier loop."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        ef = hl.zeros([tm], dtype=torch.float32)
        for _tj in hl.tile(c.size(0)):
            ef = ef + torch.exp(d[tm] - 1.0)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (ef[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def inner_loop_cvt(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """A conversion inside a loop inside the K loop."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            ef = hl.zeros([tm], dtype=torch.float32)
            for _tj in hl.tile(c.size(0)):
                ef = ef + torch.exp(dt - 1.0).to(x.dtype).to(torch.float32)
            a = (ef[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def if_else_cvt(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    """An if before the K loop whose taken branch converts."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        if flag[0] > 0:
            ef = torch.exp(dt - 1.0).to(x.dtype).to(torch.float32)
        else:
            ef = dt * 0.0
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (ef[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def while_before_cvt(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, cnt: torch.Tensor
) -> torch.Tensor:
    """A conversion inside a while loop before the K loop."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        ef = hl.zeros([tm], dtype=torch.float32)
        i = cnt[0]
        while i > 0:
            ef = ef + torch.exp(d[tm] - 1.0).to(x.dtype).to(torch.float32)
            i = i - 1
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (ef[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def while_pre_cvt(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, cnt: torch.Tensor
) -> torch.Tensor:
    """The K loop inside a while loop, from a conversion before both."""
    m, k = x.shape
    _, n = y.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        eb = torch.exp(d[tm] - 1.0).to(x.dtype)
        i = cnt[0]
        while i > 0:
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                z = hl.zeros([tm, tk], dtype=torch.float32)
                a = (eb.to(torch.float32)[:, None] + z).to(x.dtype)
                acc = torch.addmm(acc, a, y[tk, tn])
            out[tm, tn] = acc
            i = i - 1
    return out


@helion.kernel(**S)
def maxdim_pre(x: torch.Tensor, y: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """A row maximum by ``torch.max`` over a dim, a node with two outputs."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        mx, _idx = torch.max(d2[tm, :], -1)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (mx[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_bf_exp(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``exp`` of a bfloat row loaded before the K loop, in bfloat."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = torch.exp(dt[:, None] + hl.zeros([tm, tk], dtype=x.dtype))
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_cvt_then_exp(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """``exp`` in bfloat of a conversion in the K loop."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm].to(torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            a = torch.exp((dt[:, None] * 0.5 + z).to(x.dtype))
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def bf_mul_symf(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, s: float
) -> torch.Tensor:
    """A bfloat tile times a float kernel argument."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            a = (dt[:, None] + z).to(x.dtype) * s
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S, fast_math=True)
def bf_div_fast(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A bfloat division under fast math."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            a = (dt[:, None] + z).to(x.dtype) / x[tm, tk]
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_full_bf16(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """A bfloat fill of a value computed before the K loop."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        v = torch.exp(s[0])
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = hl.full([tm, tk], v, dtype=x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_cvt_mul(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A conversion followed by a product in bfloat."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            a = torch.exp(dt[:, None] - 1.0 + z).to(x.dtype) * 3.0
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_exp_where(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A conversion selected against a load by ``torch.where`` in bfloat."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            e = torch.exp(dt[:, None] - 1.0 + z).to(x.dtype)
            a = torch.where(dt[:, None] + z > 10.0, x[tm, tk], e)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def pre_cvt_float(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A conversion before the K loop, read back as float inside it."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        eb = torch.exp(d[tm] - 1.0).to(x.dtype)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            a = (eb.to(torch.float32)[:, None] + z).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def colon_if_cvt(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """A conversion in an N loop, read by a ':' K op under an if inside it."""
    m, _ = q.shape
    n, _ = k.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        h = q[tm, :].to(torch.float32) * 0.125
        for tn in hl.tile(n):
            e = torch.exp(h - 1.0).to(q.dtype)
            if tn.begin >= 0:
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                out[tm, tn] = torch.addmm(acc, e, k[tn, :].T)
            h = h * 1.0
    return out


@helion.kernel(**S)
def carry_cvt_float(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A bfloat K-loop carry made by conversions, read back as float."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        c = torch.exp(d[tm]).to(x.dtype)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            a = (c.to(torch.float32)[:, None] + z).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
            c = (c.to(torch.float32) * 0.75).to(x.dtype)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def carry_seed_cvt(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A float K-loop carry seeded by a conversion before the loop."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        c = torch.exp(d[tm]).to(x.dtype).to(torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (c[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
            c = c * 0.75
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def carry_update_cvt(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """A float K-loop carry updated through a conversion."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        c = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (c[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
            c = (c * 0.75).to(x.dtype).to(torch.float32)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def carry_and_direct(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """One conversion read by the operand and by the update of a carry the
    operand reads: the walk reaches it directly and through the carry."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        h = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            c = (h[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, c, y[tk, tn])
            h = c.to(torch.float32).amax(-1) * 0.5
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def nest_written(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """A row loaded before an N loop that writes it back, so it is staged,
    with no conversion; the N loop runs in another loop."""
    m, _ = q.shape
    n, _ = k.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        qt = q[tm, :]
        for _g in hl.grid(2):
            for tn in hl.tile(n):
                out[tm, tn] = torch.addmm(out[tm, tn], qt, k[tn, :].T)
                q[tm, :] = qt
    return out


@helion.kernel(**S)
def gather_cvt(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """A gather whose index, like the operand, comes from a conversion."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            i = (dt[:, None] + z).to(x.dtype).to(torch.int32)
            a = (i.to(torch.float32) + t[i]).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def kloop_select(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``d[tm]``, loaded before the K loop, selected over each K tile with no
    arithmetic, so a NaN reaches the conversion with its own bits."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        dt = d[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            z = hl.zeros([tm, tk], dtype=torch.float32)
            a = torch.where(z == 0.0, dt[:, None], z).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def gather_written_cvt(
    x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, t: torch.Tensor, t2: torch.Tensor
) -> torch.Tensor:
    """``gather_cvt`` from a tensor an earlier loop writes, so the gather the
    rewrite copies runs after a device-memory barrier."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tw in hl.tile(t.size(0)):
            t2[tw] = t[tw]
        for tm, tn in hl.tile([m, n]):
            dt = d[tm]
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                z = hl.zeros([tm, tk], dtype=torch.float32)
                i = (dt[:, None] + z).to(x.dtype).to(torch.int32)
                a = (i.to(torch.float32) + t2[i]).to(x.dtype)
                acc = torch.addmm(acc, a, y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def mm_reads_cvt(
    q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """The second op's operand reads the first op's result and the first
    op's operand conversion, which the first op keeps."""
    m, dh = q.shape
    n = kt.size(1)
    out = torch.empty([m, dh], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        dt = d[tm]
        acc = hl.zeros([tm, dh], dtype=torch.float32)
        for tn in hl.tile(n):
            b = (kt[:, tn].to(torch.float32) * 2.0).to(q.dtype)
            s = torch.addmm(hl.zeros([tm, tn], dtype=torch.float32), q[tm, :], b)
            p = (s * 0.125 + dt[:, None] + b.to(torch.float32).sum(0)[None, :]).to(
                q.dtype
            )
            acc = torch.addmm(acc, p, v[tn, :])
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def upd_reads_cvt(
    x: torch.Tensor, v: torch.Tensor, w: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The first op's operand reads the second op's updated acc and a
    conversion its update reads too, which the update keeps."""
    m, k = x.shape
    n = hl.specialize(v.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out2 = torch.empty([m, k], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        dt = d[tm]
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            c = (dt * 0.5).to(x.dtype)
            acc = acc * c.to(torch.float32)[:, None]
            p = (acc + c.to(torch.float32)[:, None]).to(x.dtype)
            zero = hl.zeros([tm, tk], dtype=torch.float32)
            out2[tm, tk] = torch.addmm(zero, p, w[:, tk])
            acc = torch.addmm(acc, x[tm, tk], v[tk, :])
        out[tm, :] = acc
    return out, out2


@helion.kernel(**S)
def mat_carry(
    x: torch.Tensor, v: torch.Tensor, w: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The second op's operand reads the first op's acc carry from before
    the first op, so through the first op's materialization."""
    m, k = x.shape
    n = hl.specialize(v.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out2 = torch.empty([m, k], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        acc = d[tm][:, None] + hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            old = acc
            acc = torch.addmm(acc, x[tm, tk], v[tk, :])
            p = (old * 0.5).to(x.dtype)
            zero = hl.zeros([tm, tk], dtype=torch.float32)
            out2[tm, tk] = torch.addmm(zero, p, w[:, tk])
        out[tm, :] = acc
    return out, out2


@helion.kernel(**S)
def mat_carry_cvt(
    x: torch.Tensor, v: torch.Tensor, w: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``mat_carry`` with a bfloat conversion in the carry's seed."""
    m, k = x.shape
    n = hl.specialize(v.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out2 = torch.empty([m, k], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        seed = d[tm].to(x.dtype).to(torch.float32)
        acc = seed[:, None] + hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            old = acc
            acc = torch.addmm(acc, x[tm, tk], v[tk, :])
            p = (old * 0.5).to(x.dtype)
            zero = hl.zeros([tm, tk], dtype=torch.float32)
            out2[tm, tk] = torch.addmm(zero, p, w[:, tk])
        out[tm, :] = acc
    return out, out2


@helion.kernel(**S)
def mat_carry_cvt_rev(
    x: torch.Tensor, v: torch.Tensor, w: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``mat_carry_cvt`` with the reading op first, so it reads the carry
    itself."""
    m, k = x.shape
    n = hl.specialize(v.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out2 = torch.empty([m, k], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        seed = d[tm].to(x.dtype).to(torch.float32)
        acc = seed[:, None] + hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            p = (acc * 0.5).to(x.dtype)
            zero = hl.zeros([tm, tk], dtype=torch.float32)
            out2[tm, tk] = torch.addmm(zero, p, w[:, tk])
            acc = torch.addmm(acc, x[tm, tk], v[tk, :])
        out[tm, :] = acc
    return out, out2


_NOT_RELOADED = (
    "a bfloat MPP operand staged inside a while loop, or over a K held as "
    "per-thread parts, from a value the loop does not reload every iteration"
)
#: 64 lane copies of the staging writes in a K loop.
JAM64 = helion.Config(block_sizes=[128, 16, 16], num_threads=[2, 2, 8])
_NOT_FOLLOWED = (
    "whose bfloat conversions the planner does not follow (a loop, an if or a "
    "node with several outputs)"
)
_COMPUTED = "a bfloat MPP operand staged inside a loop from a bfloat value computed by"
_CONVERTED = (
    "a bfloat MPP operand staged inside a loop from a bfloat conversion before "
    "the loop, in another graph of the loop or in a carry"
)
#: A float -> bfloat conversion of a value that is not a literal.
_BF_CAST = re.compile(
    r"static_cast<bfloat>\((?!\s*(?:-?[\d.]+f?|static_cast<float>\(\s*-?[\d.]+f?\s*\))\s*\))"
)
#: An MPP threadgroup tile; tiles that share storage take one of their names.
_TILE = r"_mpp_(?:tile|mat|seed|vec)\w*"
#: The value a float32 tile's staging write rounds.
_FLOAT_WRITE = re.compile(
    rf"{_TILE}\[[^\]]*\] = static_cast<float>\(helion_bf16_rne\((.*)\)\);"
)


def _staged_as_float(msl: str) -> bool:
    """Whether some tile takes rounded float32 staging writes, and every such
    tile is declared float."""
    tiles = set(
        re.findall(rf"({_TILE})\[[^\]]*\] = static_cast<float>\(helion_bf16_rne", msl)
    )
    return bool(tiles) and all(f"threadgroup float {t}[" in msl for t in tiles)


#: An assignment to a name ``[type] name op= value;``, as a declaration, a
#: reference or a carry's reassignment: (name, value).
_ASSIGN = re.compile(
    r"^\s*(?:(?:thread|auto|float|half|bfloat|int|uint|long|bool)[\s&]+)*"
    r"([A-Za-z_]\w*)\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)\s*(.*);\s*$",
    re.MULTILINE,
)


def _float_slice(msl: str) -> list[str]:
    """The value of every assignment on the backward slice of the float32
    tiles' staging writes.  A read of memory ends the slice, as a load ends
    the planner's walk."""
    defs: dict[str, list[str]] = {}
    for name, value in _ASSIGN.findall(msl):
        defs.setdefault(name, []).append(value)
    work = [n for w in _FLOAT_WRITE.findall(msl) for n in re.findall(r"\w+", w)]
    seen: set[str] = set()
    out: list[str] = []
    while work:
        name = work.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        out += defs[name]
        work += [n for value in defs[name] for n in re.findall(r"\w+", value)]
    return out


def _bf(t: torch.Tensor) -> torch.Tensor:
    """``t`` rounded to bfloat, as float64 on the CPU."""
    return t.cpu().to(torch.bfloat16).double()


# --- Conversions between the cooperative and Scalar layouts ---------------


@helion.kernel(**S)
def mm_softmax(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Row softmax of the result: row reductions read a materialization."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm in hl.tile(m):
            for tn in hl.tile(n, block_size=n):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                mx = torch.amax(acc, -1)
                p = torch.exp(acc - mx[:, None])
                out[tm, tn] = p / p.sum(-1)[:, None]
    return out


@helion.kernel(**S)
def in_loop_rowsum(x: torch.Tensor, y: torch.Tensor, rs: torch.Tensor) -> torch.Tensor:
    """A row sum of the running result in every K iteration."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm in hl.tile(m):
            for tn in hl.tile(n, block_size=n):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                tot = hl.zeros([tm], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                    tot = tot + acc.sum(-1)
                rs[tm] = tot
                out[tm, tn] = acc
    return out


@helion.kernel(**S)
def mat_then_store_and_sum(
    x: torch.Tensor, y: torch.Tensor, rs: torch.Tensor
) -> torch.Tensor:
    """One result stored element-wise and row-summed."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm in hl.tile(m):
            for tn in hl.tile(n, block_size=n):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                out[tm, tn] = acc * 2.0
                rs[tm] = acc.sum(-1)
    return out


@helion.kernel(**S)
def diamond_partial(
    x: torch.Tensor, y: torch.Tensor, s: torch.Tensor, o2: torch.Tensor
) -> torch.Tensor:
    """``acc + 1`` keeps its element loop; ``d1 + acc`` reads ``d1``, which a
    reduction takes to the Scalar layout."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            d1 = acc * 2.0
            out[tm, tn] = acc + 1.0
            o2[tm, tn] = d1 + acc
            s[tm, tn] = d1 * d1.sum(-1)[:, None]
    return out


@helion.kernel(**S)
def mixed_prune(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``d1`` feeds a reduction, and ``d1 + acc`` is stored."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm in hl.tile(m):
            for tn in hl.tile(n, block_size=64):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                d1 = acc * 2.0
                out[tm, tn] = d1 + acc
                s[tm] = d1.sum(-1)
    return out


@helion.kernel(**S)
def mixed_prune2(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``mixed_prune`` over N tiles, the sum broadcast back per tile."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            d1 = acc * 2.0
            out[tm, tn] = d1 + acc
            s[tm, tn] = d1 * d1.sum(-1)[:, None]
    return out


@helion.kernel(**S)
def diamond3(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The Scalar layout of the reduced ``d1`` reaches the store two ops on."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm in hl.tile(m):
            for tn in hl.tile(n, block_size=64):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                d1 = acc * 2.0
                d2 = d1 + 1.0
                out[tm, tn] = d2 * acc
                s[tm] = d1.sum(-1)
    return out


@helion.kernel(**S)
def diamond_sibling(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``e`` is reduced and ``d1 + e`` stored, so ``d1`` has no element-loop
    use left."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            d1 = acc * 2.0
            e = acc + 1.0
            out[tm, tn] = d1 + e
            s[tm, tn] = e * e.sum(-1)[:, None]
    return out


@helion.kernel(**S)
def chain_prune(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``d1`` feeds a reduction and ``d2 = d1 + 1``, which is stored."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm in hl.tile(m):
            for tn in hl.tile(n, block_size=64):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                d1 = acc * 2.0
                d2 = d1 + 1.0
                out[tm, tn] = d2
                s[tm] = d1.sum(-1)
    return out


@helion.kernel(**S)
def mm_rowscale(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """An epilogue operand that is a per-row carry of reductions."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            rs = hl.zeros([tm], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                rs = rs + x[tm, tk].to(torch.float32).sum(-1)
            out[tm, tn] = acc * rs[:, None]
    return out


@helion.kernel(**S)
def else_reads(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    flag: torch.Tensor,
    o2: torch.Tensor,
) -> torch.Tensor:
    """The K loop in an if branch; the else branch reads the seed."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for g in hl.grid(1):
        f = flag[g]
        for tm, tn in hl.tile([m, n]):
            acc = c[tm, tn] + 0.0
            if f > 0:
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            else:
                o2[tm, tn] = acc * 3.0
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def alias_in_if(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    """A uniform if runs the K loop and binds another name to the result,
    which is read after the if."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for g in hl.grid(1):
        f = flag[g]
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            b = c[tm, tn] + 0.0
            if f > 0:
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                b = acc
            out[tm, tn] = b + 1.0
    return out


@helion.kernel(**S)
def alias_in_grid(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """An outer loop runs the K loop and binds another name to the result at
    each step."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        b = c[tm, tn] + 0.0
        for _g in hl.grid(2):
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            b = acc
        out[tm, tn] = b + 1.0
    return out


@helion.kernel(**S)
def vw_xthread(x: torch.Tensor, y: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Thread (m, n) writes ``x2[m, n]``; the K loop stages ``x2[m, k]``."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            x2[tm, tn] = x[tm, tn] * 2.0
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x2[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def staged_computed_written(
    x: torch.Tensor, y: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """``vw_xthread`` with a computed lhs, staged whatever it reads."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            x2[tm, tn] = x[tm, tn] * 2.0
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x2[tm, tk] * 1.0, y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def rhs_written(x: torch.Tensor, y: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    """An earlier tile loop writes ``y2`` on its own threads; the K loop reads
    it as the rhs."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tw, tv in hl.tile([k, n]):
            y2[tw, tv] = y[tw, tv] * 2.0
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y2[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def demote_then_view(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """The first K loop's store of ``t`` takes the Scalar layout, since the
    second K loop reads ``t`` as its lhs."""
    m, k = x.size()
    _, n = y.size()
    _, p = w.size()
    out = torch.empty([m, p], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            t[tm, tn] = acc.to(t.dtype)
            acc2 = hl.zeros([tm, tn], dtype=torch.float32)
            for tj in hl.tile(n):
                acc2 = torch.addmm(acc2, t[tm, tj], w[tj, tn])
            out[tm, tn] = acc2
    return out


@helion.kernel(**S)
def gather_idx_written(
    x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, idx2: torch.Tensor
) -> torch.Tensor:
    """A staged lhs gathered through ``idx2``, which an earlier tile loop
    writes on its own threads."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tw in hl.tile(m):
            idx2[tw] = idx[tw]
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[idx2[tm], tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def seed_expr(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = c[tm, tn] * d[tm, tn]
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def seed_bcast(x: torch.Tensor, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32) + b[tn][None, :]
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def seed_read_after(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The seed's Scalar value is read after the loop too."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            s = c[tm, tn] * 1.0
            acc = s
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc - s
    return out


@helion.kernel(**S)
def chained(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """One K loop's result seeds the next."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            a1 = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                a1 = torch.addmm(a1, x[tm, tk], y[tk, tn])
            a2 = a1 * 1.0
            for tk in hl.tile(k):
                a2 = torch.addmm(a2, x[tm, tk], z[tk, tn])
            out[tm, tn] = a2
    return out


@helion.kernel(**S)
def free_rows_seeded(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """grouped_gemm's free layout with a seed loaded from ``c``."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.load(c, [rows, cols], extra_mask=rv[:, None] & cv[None, :])
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def seed_outside_if(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for g in hl.grid(1):
        f = flag[g]
        for tm, tn in hl.tile([m, n]):
            acc = c[tm, tn] + 0.0
            if f > 0:
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def seed_outer_grid(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = c[tm, tn] * 1.0
        for _g in hl.grid(2):
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def seed_newvar(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The seed reaches the loop through a copy, and is read after it."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            a0 = c[tm, tn] * 2.0
            acc = a0
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc + a0
    return out


@helion.kernel(**S)
def grid_old_read(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The outer loop's carry read after the K loop: each pass's product."""
    m, k = x.size()
    _, n = y.size()
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for _g in hl.grid(2):
            before = acc
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc - before
    return out


@helion.kernel(**S)
def inner_tb(x: torch.Tensor, y: torch.Tensor, nb: int) -> torch.Tensor:
    """The result stored into every slice of a 3-D output from an inner loop
    over a block the op never sees."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([nb, m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            for tb in hl.tile(nb):
                out[tb, tm, tn] = acc[None, :, :]
    return out


@helion.kernel(**S)
def perm_store(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tn, tm] = acc.T
    return out


@helion.kernel(**S)
def tif_mat(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """A store under a per-element if, of a materialized value."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            s = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                s = s + c[tm, tn]
            if c[tm, tn] > 0:
                out[tm, tn] = acc + s
    return out


@helion.kernel(**S)
def store_at_row_max(
    x: torch.Tensor, y: torch.Tensor, sel: torch.Tensor
) -> torch.Tensor:
    """A store at a row index a reduction computes, which no thread can
    recompute at another element."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            row = torch.amax(sel[tm, tn], -1)
            out[row, tn] = acc + 1.0
    return out


@helion.kernel(**S)
def store_under_row_mask(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """A store under a mask a row reduction over the N tile computes."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            keep = torch.amax(c[tm, tn], -1) > 0.0
            hl.store(out, [tm, tn], acc * 2.0, extra_mask=keep[:, None])
    return out


@helion.kernel(**S)
def gather_at_row_max(
    x: torch.Tensor, y: torch.Tensor, sel: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """An epilogue operand gathered at a row index a reduction computes."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            row = torch.amax(sel[tm, tn], -1)
            out[tm, tn] = acc + c[row, tn]
    return out


@helion.kernel(**S)
def load_under_row_mask(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """An epilogue operand loaded under a mask a row reduction computes."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            keep = torch.amax(c[tm, tn], -1) > 0.0
            out[tm, tn] = acc + hl.load(c, [tm, tn], extra_mask=keep[:, None])
    return out


@helion.kernel(**S)
def mat_sibling_rowmax(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """At ``SIBLING64`` the [16, 64] sibling widens N's thread axis to 64
    beside an N block of 16, so a surplus thread's local index would read past
    the materialization tile.  Every reduced value is at most -1, so the row
    max would show a surplus thread's value if the reduction counted it."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out2 = torch.empty_like(z)
    for _g in hl.grid(1):
        for ti, tj in hl.tile([z.size(0), z.size(1)]):
            out2[ti, tj] = z[ti, tj] * 2
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc + torch.amax(-(acc * acc) - 1.0, -1)[:, None]
    return out, out2


@helion.kernel(**S)
def two_zero_seeds(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = y.size()
    o1 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    o2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            a1 = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                a1 = torch.addmm(a1, x[tm, tk], y[tk, tn])
            a2 = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                a2 = torch.addmm(a2, x[tm, tk], z[tk, tn])
            o1[tm, tn] = a1
            o2[tm, tn] = a2
    return o1, o2


@helion.kernel(**S)
def shared_load_seed(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One loaded tile seeds two K loops and is stored too."""
    m, k = x.size()
    _, n = y.size()
    o1 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    o2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    o3 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = c[tm, tn] * 1.0
            a1 = acc
            for tk in hl.tile(k):
                a1 = torch.addmm(a1, x[tm, tk], y[tk, tn])
            a2 = acc
            for tk in hl.tile(k):
                a2 = torch.addmm(a2, x[tm, tk], z[tk, tn])
            o1[tm, tn] = a1
            o2[tm, tn] = a2
            o3[tm, tn] = acc
    return o1, o2, o3


@helion.kernel(**S)
def shared_zero_seed(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One ``hl.zeros`` seeds two K loops."""
    m, k = x.size()
    _, n = y.size()
    o1 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    o2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            a1 = acc
            for tk in hl.tile(k):
                a1 = torch.addmm(a1, x[tm, tk], y[tk, tn])
            a2 = acc
            for tk in hl.tile(k):
                a2 = torch.addmm(a2, x[tm, tk], z[tk, tn])
            o1[tm, tn] = a1
            o2[tm, tn] = a2
    return o1, o2


@helion.kernel(**S)
def shared_full_seed(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One ``hl.full`` seeds two K loops."""
    m, k = x.size()
    _, n = y.size()
    o1 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    o2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.full([tm, tn], 0.5, dtype=torch.float32)
            a1 = acc
            for tk in hl.tile(k):
                a1 = torch.addmm(a1, x[tm, tk], y[tk, tn])
            a2 = acc
            for tk in hl.tile(k):
                a2 = torch.addmm(a2, x[tm, tk], z[tk, tn])
            o1[tm, tn] = a1
            o2[tm, tn] = a2
    return o1, o2


@helion.kernel(**S)
def tif_k(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The K loop under an if on a per-element predicate."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            if c[tm, tn] > 0:
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def tif_all(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The whole K loop and its seed under a per-element if."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            if c[tm, tn] > 0:
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                out[tm, tn] = acc
    return out


@helion.kernel(**S)
def while_tensor(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The K loop in a while on a per-element condition."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            v = c[tm, tn]
            while v > 0:
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                out[tm, tn] = acc
                v = v - 1.0
    return out


@helion.kernel(**S)
def else_rewrites(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    """The else branch assigns the accumulator another value."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for g in hl.grid(1):
        f = flag[g]
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            if f > 0:
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            else:
                acc = c[tm, tn] * 2.0
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def else_holds_k(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    """The K loop sits in the else branch, seeded before the if."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    side = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for g in hl.grid(1):
        f = flag[g]
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            if f > 0:
                side[tm, tn] = c[tm, tn] * 2.0
            else:
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def swapped_mat_store(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """A swapped-tile store of a materialized value."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([n, m], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            s = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                s = s + c[tm, tn]
            out[tn, tm] = acc + s
    return out


@helion.kernel(**S)
def seed_swapped(x: torch.Tensor, y: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = ct[tn, tm] + 0.0
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def seed_relabel(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, ct: torch.Tensor
) -> torch.Tensor:
    """A seed whose dims are M and N, computed from a tile that meets them
    swapped."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = c[tm, tn] + ct[tn, tm]
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def outer_rebind(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """An outer loop carries the accumulator but rebinds it after the K
    loop."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for _g in hl.grid(2):
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            acc = acc * 0.5
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def inloop_scale(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = acc * w[tm][:, None]
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def inloop_const(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = acc * 0.5
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def acc_load_in_loop(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(c[tm, tn], x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def loop_operand_written(
    x: torch.Tensor, y: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """The lhs is summed by an inner loop in the K loop from a written
    tensor, so its fence would sit inside that loop."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            x2[tm, tn] = x[tm, tn] * 2.0
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                a = hl.zeros([tm, tk], dtype=torch.float16)
                for _i in hl.grid(2):
                    a = a + x2[tm, tk] * 0.5
                acc = torch.addmm(acc, a, y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(**S)
def outer_load_written(
    x: torch.Tensor, y: torch.Tensor, s_src: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """The staged lhs is scaled by a row of ``s`` loaded before the K loop,
    after the tile writes it."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            s[tm] = s_src[tm] * 2.0
            sv = s[tm]
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk] * sv[:, None].to(x.dtype), y[tk, tn])
            out[tm, tn] = acc
    return out


def _collective_cases() -> list[
    tuple[str, helion.Kernel, tuple[object, ...], torch.Tensor]
]:
    """Kernels whose epilogues read 0-d threadgroup reductions, with their
    references at ``TILE16``: each reduction sums one 16-wide N tile."""
    torch.manual_seed(0)
    m, k, n = 48, 72, 40
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    y = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(n, device=DEVICE)
    c = torch.randn(m, n, device=DEVICE)
    jidx = (torch.arange(n, device=DEVICE) % 7 == 0).to(torch.int32)
    flag = torch.ones(1, device=DEVICE, dtype=torch.int32)
    mm, cd = _mm(x, y), c.cpu().double()
    s = _tiled(bias.cpu(), 16)[None, :]
    j = _tiled(jidx.cpu(), 16).long()
    return [
        ("red_n_after", red_n_after, (x, y, bias), mm + s),
        ("red_epi_before", red_epi_before, (x, y, bias, c), mm + cd * s + s + 2),
        ("red_load_index", red_load_index, (x, y, c, jidx), mm + cd[:, j]),
        (
            "red_jloop_acc",
            red_jloop_acc,
            (x, y, bias),
            torch.stack([mm + s, mm + 2 * s]),
        ),
        ("outer_epi_if", outer_epi_if, (x, y, bias, c, flag), mm + cd * s),
    ]


def _check_collectives() -> None:
    """Run every ``_collective_cases`` kernel; each element loop must read the
    reductions by name, never re-run them."""
    for name, kernel, args, ref in _collective_cases():
        out, _, msl = _run(kernel, args, TILE16)
        _close(out, ref)
        assert "helion_red::" in msl, name
        assert "helion_red::" not in _element_loops(msl), name


class _MppCase(unittest.TestCase):
    """Assertions shared by the test classes below."""

    def _assert_mpp(self, code: str) -> None:
        self.assertEqual(code.count("_helion_mpp_acc("), 1)
        self.assertIn("_helion_mpp_run(", code)

    def _rejects(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        message: str,
    ) -> None:
        self._rejects_at(kernel, args, (config,), message)

    def _rejects_at(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        configs: tuple[helion.Config | None, ...],
        message: str,
    ) -> None:
        for config in configs:
            with (
                self.subTest(kernel=kernel.name, config=config),
                self.assertRaisesRegex(exc.BackendUnsupported, re.escape(message)),
            ):
                _run(kernel, args, config)


@_requires_mps
class TestMppMatmul(_MppCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def _free_args(
        self, m: int, bm: int, bn: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        a = torch.randn(m, 72, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(72, 90, device=DEVICE, dtype=torch.bfloat16)
        return a, b, (m + bm - 1) // bm, (90 + bn - 1) // bn

    def test_free_rows(self) -> None:
        """grouped_gemm's layout: free rows and columns under ``hl.grid``,
        uniform mask bounds as view extents, bf16, tails (100x72 @ 72x90)."""
        for bm, bn, bk in ((32, 32, 16), (64, 16, 32)):
            a, b, nbm, nbn = self._free_args(100, bm, bn)
            config = helion.Config(block_sizes=[bm, bn, bk])
            out, code, msl = _run(free_rows, (a, b, nbm, nbn), config)
            _close(out, _mm(a, b))
            self._assert_mpp(code)
            self.assertIn("multiply_accumulate", msl)
            self.assertIn("is_valid_element", msl)
            self.assertIn("if (_mpp_a_o0 < _mpp_a_e0", msl)
            # Each handle is rebased at its tile in 64 bits, its extents
            # capped at the tile.
            self.assertRegex(msl, r"\(a \+ \(_mpp_a_o0 \* 72")
            self.assertIn(f"(long){bm})", msl)
            self.assertNotIn(".slice(", msl)
            self.assertNotIn("_mpp_mat", msl)

    def test_runtime_sizes(self) -> None:
        """Under dynamic shapes m, n and k bind as ints, so the 2-D program
        id's floor division and the K loop's bound compile; tails off the
        tile grid keep their masks."""
        config = helion.Config(block_sizes=[32, 32, 16])
        for m, k, n in ((64, 64, 64), (100, 48, 72), (64, 40, 48), (128, 256, 96)):
            with self.subTest(m=m, k=k, n=n):
                x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
                y = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
                out, code, msl = _run(addmm_sized, (x, y), config)
                _close(out, _mm(x, y))
                self._assert_mpp(code)
                self.assertRegex(msl, r"constant long& k \[\[buffer")

    def test_free_rows_under_ifs(self) -> None:
        """Data-dependent ifs on 0-d loads lower to ``else: pass``.  The if
        outputs the accumulator it defines, a slot nothing reads, so the
        result stays cooperative."""
        a, b, nbm, nbn = self._free_args(100, 32, 32)
        sizes = torch.tensor([1, 0, 1, 1], device=DEVICE, dtype=torch.int32)
        ref = _mm(a, b)
        ref[32:64] = 0
        out, code, msl = _run(free_rows_if, (a, b, sizes, nbm, nbn), FREE32)
        _close(out, ref)
        self._assert_mpp(code)
        self.assertNotIn("_mpp_mat", msl)

    def test_epilogue(self) -> None:
        """Epilogue loads are re-run per element, also with M and N on lanes
        (a 32x32 tile on 64 threads)."""
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        b = torch.randn(40, device=DEVICE)
        c = torch.randn(48, 40, device=DEVICE)
        ref = torch.sigmoid(_mm(x, y) * 0.5 + b.cpu().double()) + c.cpu().double()
        for config in (
            helion.Config(block_sizes=[16, 16, 16]),
            helion.Config(block_sizes=[32, 32, 16], num_threads=[8, 8, 1]),
        ):
            out, code, msl = _run(epilogue, (x, y, b, c), config)
            _close(out, ref)
            self._assert_mpp(code)
            self.assertNotIn("_mpp_mat", msl)
            self.assertNotIn("_mpp_vec", msl)

    def test_transposed_operands(self) -> None:
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        yt = y.T.contiguous().T
        xt = x.T.contiguous().T
        for args in ((x, yt), (xt, y)):
            out, code, _ = _run(plain_g, args, TILE16)
            _close(out, _mm(x, y))
            self._assert_mpp(code)
        a, b, nbm, nbn = self._free_args(100, 32, 32)
        out, _, _ = _run(free_rows, (a, b.T.contiguous().T, nbm, nbn), FREE32)
        _close(out, _mm(a, b))

    def test_zero_stride(self) -> None:
        """Expanded operands, with strides (0, 1) and (1, 0), are views."""
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        row = torch.randn(40, device=DEVICE, dtype=torch.float16)
        col = torch.randn(48, device=DEVICE, dtype=torch.float16)
        for args in ((x, row.expand(72, 40)), (col[:, None].expand(48, 72), y)):
            out, _, _ = _run(plain_g, args, TILE16)
            _close(out, _mm(*args))

    def test_two_groups(self) -> None:
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        z = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        out1 = torch.empty(48, 40, device=DEVICE)
        out2 = torch.empty(48, 40, device=DEVICE)
        config = helion.Config(block_sizes=[16, 16, 16, 16], num_threads=[16, 16, 1, 1])
        _run(two_groups, (x, y, z, out1, out2), config)
        _close(out1, _mm(x, y))
        _close(out2, torch.relu(1 + _mm(x, z)))

    def test_counter_beside_op(self) -> None:
        """A 0-d counter in K's lane loop runs once per K tile."""
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        out, _, _ = _run(counter_beside, (x, y), TILE16)
        _close(out, _mm(x, y) + 5)

    def test_reassigned_names(self) -> None:
        """An epilogue statement reads a 0-d value that a later loop, an if,
        or the K loop itself reassigns before the store."""
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        c = torch.randn(48, 40, device=DEVICE)
        sc = torch.tensor([1.5], device=DEVICE)
        flag = torch.ones(1, device=DEVICE, dtype=torch.int32)
        mm = _mm(x, y)
        out, _, _ = _run(stale_loop, (x, y, sc), TILE16)
        _close(out, mm * 1.5)
        out, _, _ = _run(stale_if, (x, y, sc, flag), TILE16)
        _close(out, mm * 1.5)
        out, _, _ = _run(stale_kcarry, (x, y, sc, c), TILE16)
        _close(out, mm + c.cpu().double() * 1.5)

    def test_reassigned_view_base_and_bound(self) -> None:
        a, b, nbm, nbn = self._free_args(45, 32, 32)
        off = torch.tensor([0], device=DEVICE, dtype=torch.int32)
        flag = torch.ones(1, device=DEVICE, dtype=torch.int32)
        out, _, msl = _run(base_if, (a, b, off, flag, nbm, nbn), FREE32)
        _close(out, _mm(a, b))
        self.assertIn("long _mpp_a_o0 = (long)(_mpp_snap_", msl)
        lim = torch.tensor([30], device=DEVICE, dtype=torch.int32)
        ref = _mm(a, b)
        ref[30:] = 0
        out, _, _ = _run(bound_after, (a, b, lim, nbm, nbn), FREE32)
        _close(out, ref)

    def test_collectives_read_by_name(self) -> None:
        """0-d reductions in the epilogue stay at their own statements; the
        element loop reads them by name, whatever the hash seed."""
        root = Path(__file__).resolve().parents[1]
        for seed in ("0", "1"):
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from test.test_metal_mpp_matmul import _check_collectives; "
                        "_check_collectives()"
                    ),
                ],
                cwd=root,
                env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(root)},
                check=True,
            )

    def test_rerun_backstop(self) -> None:
        """An element loop that would re-run a collective raises, here with
        the leaf rule patched out."""
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        args = (
            x,
            y,
            torch.randn(40, device=DEVICE),
            torch.randn(48, 40, device=DEVICE),
        )
        with patch.object(mpp_matmul, "_collectives", return_value=set()):
            self._rejects(outer_epi, args, TILE16, "holds a cross-thread operation")
        buffers = frozenset({"scratch_0"})
        for stmt in (
            "v = helion_red.group_sum(site, s, x, tid)",
            "v = helion_red.seg_sum(x)",
            "v = helion_red.simd_sum(x)",
            "v = scratch_0[tid]",
            "v = _metal_lane_fold(x)",
            "v = _metal_lane_reduce(x)",
            "v = _helion_mpp_run(acc, 0)",
            "v = simd_sum(x)",
            "v = quad_shuffle(x, 1)",
            "threadgroup_barrier(flags)",
            "simdgroup_barrier(flags)",
        ):
            self.assertTrue(mpp_matmul._cross_thread(ast.parse(stmt), buffers), stmt)
        self.assertFalse(
            mpp_matmul._cross_thread(ast.parse("v = simd_group + 1"), buffers)
        )

    def test_dims_met_in_place(self) -> None:
        """Operands indexed by the same tiles in the same order, broadcast and
        lower-rank operands among them, are re-run per element."""
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 48, device=DEVICE, dtype=torch.float16)
        c = torch.randn(48, 48, device=DEVICE)
        b = torch.randn(48, device=DEVICE)
        bc = torch.randn(48, device=DEVICE)
        cd, bd, bcd = c.cpu().double(), b.cpu().double(), bc.cpu().double()
        v = _mm(x, y) + cd + bd[None, :] + bcd[:, None] + bd
        out, _, _ = _run(epi_ok, (x, y, c, b, bc), TILE16)
        _close(out, torch.where(cd > 0, v, 0.0))
        out, _, _ = _run(k_begin, (x, y), TILE16)
        _close(out, _mm(x[:, 8:], y[8:]))
        out, _, _ = _run(m_short, (x, y), TILE16)
        ref = _mm(x, y)
        ref[40:] = 0
        _close(out, ref)
        a = torch.randn(45, 72, device=DEVICE, dtype=torch.bfloat16)
        bb = torch.randn(72, 50, device=DEVICE, dtype=torch.bfloat16)
        out, _, _ = _run(iota_data, (a, bb), FREE32)
        ref = torch.zeros(48, 55, dtype=torch.float64)
        rows = torch.arange(45, dtype=torch.float64)[:, None]
        cols = torch.arange(50, dtype=torch.float64)[None, :]
        ref[3:, 5:] = _mm(a, bb) + rows * 0.25 - cols
        _close(out, ref)

    def test_bare_iota_extent(self) -> None:
        """A bare ``hl.arange`` index masked on itself: rows 23..29 are zero."""
        a = torch.randn(30, 72, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(72, 30, device=DEVICE, dtype=torch.bfloat16)
        cnt = torch.tensor([23], device=DEVICE, dtype=torch.int32)
        out, _, _ = _run(bare_iota_rows, (a, b, cnt), FREE32)
        ref = _mm(a, b)
        ref[23:] = 0
        _close(out, ref)

    def test_extent_range(self) -> None:
        """Bounds and bases past both int32 edges, empty tiles beside live
        ones, and literal bounds clamped into [0, size] at codegen."""
        a, b, nbm, nbn = self._free_args(45, 32, 32)
        mm = _mm(a, b)
        zero = torch.zeros_like(mm)

        def i64(v: int) -> torch.Tensor:
            return torch.tensor([v], device=DEVICE, dtype=torch.int64)

        out, _, _ = _run(masked_rows, (a, b, i64(2**32 + 5), nbm, nbn), FREE32)
        _close(out, mm)
        out, _, _ = _run(masked_rows, (a, b, i64(-(2**32) + 5), nbm, nbn), FREE32)
        _close(out, zero)
        out, _, _ = _run(shifted_rows, (a, b, i64(2**32), i64(45), nbm, nbn), FREE32)
        _close(out, zero)
        # The last tile's rows reach past the tensor: they read as zeros.
        out, _, _ = _run(shifted_rows, (a, b, i64(5), i64(2**32 + 5), nbm, nbn), FREE32)
        ref = zero.clone()
        ref[:40] = mm[5:]
        _close(out, ref)
        lims = torch.full((72,), 2**32 + 5, device=DEVICE, dtype=torch.int64)
        lims[16:32] = -(2**32) + 5
        lims[48:64] = 20
        ref = zero.clone()
        ad, bd = a.cpu().double(), b.cpu().double()
        for k0, bound in ((0, 45), (32, 45), (48, 20), (64, 45)):
            ref[:bound] += ad[:bound, k0 : k0 + 16] @ bd[k0 : k0 + 16]
        out, _, _ = _run(masked_rows_kvary, (a, b, lims, nbm, nbn), FREE32)
        _close(out, ref)
        out, _, msl = _run(masked_rows_literal, (a, b, nbm, nbn), FREE32)
        _close(out, zero)
        # A literal bound is folded into [0, size] at codegen.
        self.assertIn("long _mpp_a_e0 = (long)(0);", msl)
        out, _, _ = _run(masked_rows_min_literal, (a, b, nbm, nbn), FREE32)
        _close(out, zero)
        # Bound 64 of m = 100 at 64-row tiles: tile 1 is empty, tile 0 not.
        a2, b2, nbm2, nbn2 = self._free_args(100, 64, 16)
        lim = torch.tensor([64], device=DEVICE, dtype=torch.int32)
        config = helion.Config(block_sizes=[64, 16, 16])
        out, _, _ = _run(masked_rows, (a2, b2, lim, nbm2, nbn2), config)
        ref = _mm(a2, b2)
        ref[64:] = 0
        _close(out, ref)
        a3, b3, nbm3, nbn3 = self._free_args(45, 64, 16)
        lim = torch.tensor([0], device=DEVICE, dtype=torch.int32)
        config = helion.Config(block_sizes=[64, 16, 32])
        out, _, _ = _run(masked_rows, (a3, b3, lim, nbm3, nbn3), config)
        _close(out, torch.zeros(45, 90, dtype=torch.float64))

    def test_wide_tiles_at_four_simdgroups(self) -> None:
        """One op covers 256 rows or columns on four SIMD groups: a K loop
        over device views with M = 256, and a ``:`` N of 256 with M and N
        tails."""
        x = torch.randn(512, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 48, device=DEVICE, dtype=torch.float16)
        config = helion.Config(block_sizes=[256, 16, 16], num_threads=[32, 16, 1])
        out, code, msl = _run(plain_g, (x, y), config)
        self.assertEqual(_launch(code), (8, 16, 1))
        self.assertIn("matmul2d_descriptor(256, 16, 16,", msl)
        _close(out, _mm(x, y))
        p = torch.randn(40, 72, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(72, 200, device=DEVICE, dtype=torch.bfloat16)
        out, _, msl = _run(colon_n_kloop, (p, w), None)
        self.assertIn("matmul2d_descriptor(32, 200,", msl)
        _close(out, _mm(p, w))


@_requires_mps
class TestMppMatmulRejects(_MppCase):
    """Forms the cooperative lowering doesn't support raise
    ``BackendUnsupported``."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        self.y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        self.c = torch.randn(48, 40, device=DEVICE)
        self.bias = torch.randn(40, device=DEVICE)
        self.a = torch.randn(45, 72, device=DEVICE, dtype=torch.bfloat16)
        self.b = torch.randn(72, 90, device=DEVICE, dtype=torch.bfloat16)

    def _free(self, *extra: object) -> tuple[object, ...]:
        return (self.a, self.b, *extra, 2, 3)

    def test_other_block_on_threads(self) -> None:
        config = helion.Config(block_sizes=[4, 16, 16, 16], num_threads=[4, 16, 16, 1])
        tag = torch.zeros(4, device=DEVICE)
        self._rejects(
            batch_threaded, (self.x, self.y, tag), config, "is spread over threads"
        )

    @_full_threadgroup
    def test_launch(self) -> None:
        message = "needs whole 32-thread SIMD groups and at most 1024 threads"
        self._rejects(sibling17, (self.x, self.y, self.bias), None, message)
        config = helion.Config(block_sizes=[32, 32, 16], num_threads=[32, 32, 2])
        self._rejects(plain_g, (self.x, self.y), config, message)

    def test_tile_over_256(self) -> None:
        x = torch.randn(1024, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        config = helion.Config(block_sizes=[512, 16, 16], num_threads=[32, 16, 1])
        self._rejects(plain_g, (x, y), config, "one op covers M, N <= 256")

    def test_wide_tile_on_few_simdgroups(self) -> None:
        """An op over 128 rows on one SIMD group, where the hardware
        miscomputes it."""
        x = torch.randn(512, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        config = helion.Config(block_sizes=[256, 32, 16], num_threads=[8, 4, 1])
        self._rejects(
            plain_g, (x, y), config, "an MPP matmul tile 256x32 on 1 SIMD groups"
        )

    def test_tile_span(self) -> None:
        """A column-major lhs whose K stride makes one tile span 2**31
        elements; its storage is reserved lazily, so nothing is touched."""
        rows = 2**24 + 2**18
        a = torch.empty_strided((rows, 128), (1, rows), dtype=torch.bfloat16)
        b = torch.empty(128, 64, dtype=torch.bfloat16)
        off = torch.zeros(1, dtype=torch.int64)
        lim = torch.full((1,), rows, dtype=torch.int64)
        bound = shifted_rows.bind((a, b, off, lim, 1, 1))
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "an MPP view whose tile spans"
        ):
            bound.to_code(helion.Config(block_sizes=[32, 32, 128]))

    def test_swapped_tiles(self) -> None:
        """A store, an epilogue operand, a ``where`` or a load mask that meets
        M and N swapped: the element loop would read another element."""
        x = self.x
        y = torch.randn(72, 48, device=DEVICE, dtype=torch.float16)
        c = torch.randn(48, 48, device=DEVICE)
        b = torch.randn(48, device=DEVICE)
        lim = torch.randn(48, device=DEVICE)
        msg = "an MPP epilogue re-run at each cooperative element would read another"
        self._rejects(swapped_store, (x, y), TILE16, msg)
        self._rejects(swapped_epi_load, (x, y, c), TILE16, msg)
        self._rejects(bias_row_swapped, (x, y, b), TILE16, msg)
        self._rejects(where_swapped, (x, y, c), TILE16, msg)
        self._rejects(epi_mask_swapped, (x, y, c, lim), TILE16, msg)


def _nan_padded(data: torch.Tensor, before: int = 0) -> torch.Tensor:
    """``data`` as a slice of a NaN-filled buffer, from column ``before`` on:
    every element before that column, past its columns or past its rows is
    NaN, so a read outside the data shows."""
    rows, cols = data.shape
    buf = torch.full(
        (rows + 8, before + cols + 24), float("nan"), dtype=data.dtype, device=DEVICE
    )
    buf[:rows, before : before + cols] = data
    return buf[:rows, : before + cols]


def _pack_int4(w: torch.Tensor) -> torch.Tensor:
    """int4_gemm's packing: rows 2i and 2i + 1 of ``w`` in one byte."""
    k, n = w.shape
    pairs = w.to(torch.int8).reshape(k // 2, 2, n)
    return ((pairs[:, 0] & 0xF) | (pairs[:, 1] << 4)).to(torch.int8)


def _rows_ref(a: torch.Tensor, b: torch.Tensor, src: list[int]) -> torch.Tensor:
    """``a'' @ b`` where row ``r`` of ``a'`` is ``a[src[r]]``, or zeros for a
    ``src[r]`` past ``a``."""
    lhs = torch.zeros(len(src), a.size(1), dtype=torch.float64)
    for r, s in enumerate(src):
        if 0 <= s < a.size(0):
            lhs[r] = a[s].cpu().double()
    return lhs @ b.cpu().double()


_BARRIER = "threadgroup_barrier(mem_flags::mem_threadgroup);"


@_requires_mps
class TestMppStaging(_MppCase):
    """Operands that are not device views are staged: the threads that hold
    them write them into a threadgroup tile, between barriers, and the op
    reads it through a handle whose K extent is the K tile's live length.
    A K held as per-thread parts is a widened view or staged parts."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def _assert_staged(self, msl: str, dtype: str, tiles: int = 1) -> None:
        """Each staged tile is written under its bounds before the first
        barrier, and the barrier pair encloses the op's run block alone."""
        self.assertEqual(
            len(re.findall(rf"threadgroup {dtype} _mpp_tile\w*\[", msl)), tiles
        )
        self.assertEqual(msl.count(_BARRIER), 2)
        _, between, _ = msl.split(_BARRIER)
        lines = [line.strip() for line in between.strip().splitlines()]
        self.assertEqual((lines[0], lines[-1]), ("{", "}"))
        self.assertIn("_mpp_op_0.run(_mpp_a, _mpp_b,", between)
        self.assertNotIn("_mpp_tile[", between)
        self.assertRegex(
            msl,
            rf"tensor<threadgroup {dtype}, dextents<int32_t, 2>, tensor_inline>"
            r"\(_mpp_tile\w*, dextents<int32_t, 2>\([^;]*metal::min\(\(long\)\("
            r"[^;]*end_\d+ - offset_\d+\)+, \(long\)\d+\)",
        )
        self.assertNotIn("_mpp_tile", _element_loops(msl))

    def test_int4_gemm(self) -> None:
        """int4_gemm's kernel: a widened view of A beside the unpacked B,
        staged as parts.  A is a slice of a NaN buffer, so a read past M or
        K shows; M = 1 rounds the descriptor up; K // 2 = 37 leaves a K tail;
        a column-major A uses the transpose flag."""
        bf16 = torch.bfloat16

        cases = [
            ((m, k, n), None, False)
            for m, k, n in (
                (1, 128, 64),
                (1, 74, 64),
                (4, 256, 64),
                (33, 128, 40),
                (16, 74, 64),
            )
        ]
        cases.append(((16, 74, 40), None, True))
        cases += [
            ((16, 128, 64), helion.Config(**config), False)
            for config in (
                {"block_sizes": [16, 16, 16], "num_threads": [4, 16, 16]},
                {"block_sizes": [64, 16, 16]},
                {"block_sizes": [16, 32, 16]},
            )
        ]
        for (m, k, n), config, column_major in cases:
            w = torch.randint(-8, 8, (k, n))
            a = torch.randn(m, k).to(bf16)
            if column_major:
                lhs = a.to(DEVICE).t().contiguous().t()
            else:
                lhs = _nan_padded(a.to(DEVICE))
            with self.subTest(m=m, k=k, n=n, config=config, column_major=column_major):
                out, _, msl = _run(int4_gemm, (lhs, _pack_int4(w).to(DEVICE)), config)
                _close(out, (a.double() @ w.double()).to(bf16).double())
                self._assert_staged(msl, "bfloat")
                self.assertRegex(msl, r"_mpp_owner\w*\)\) \{ _mpp_tile")

    def test_factored_k(self) -> None:
        """A K held as parts beyond int4_gemm: a stacked lhs, c = 3, two
        widened views through hl.dot and torch.addmm (under hl.grid and at
        the root), a stacked rhs that is
        non-zero past K beside a NaN-padded lhs, and a K loop starting at 3."""
        f16 = torch.float16
        x, w = torch.randn(32, 48, device=DEVICE), torch.randn(96, 40, device=DEVICE)
        ref = torch.stack([x, x * 2], dim=2).reshape(32, 96)
        out, _, _ = _run(stacked_into_dot, (x, w), None)
        _close(out, _mm(ref, w))
        a = torch.randn(16, 96, device=DEVICE, dtype=f16)
        bs = [torch.randn(32, 32, device=DEVICE, dtype=f16) for _ in range(3)]
        rhs = torch.stack([bs[0], bs[1], bs[2] * 2], dim=1).reshape(96, 32)
        out, _, _ = _run(dot_parts_c3, (a, *bs), None)
        _close(out, _mm(a, rhs))
        for m, k2, n in ((48, 128, 40), (16, 74, 64), (1, 128, 64)):
            a = torch.randn(m, k2, device=DEVICE, dtype=f16)
            w = torch.randn(k2, n, device=DEVICE, dtype=f16)
            for kernel, config in (
                (dot_two_widened, None),
                (addmm_two_widened_root, None),
                (
                    addmm_parts,
                    helion.Config(block_sizes=[16, 16, 16], num_threads=[2, 16, 16]),
                ),
            ):
                with self.subTest(kernel=kernel.name, m=m, k2=k2, n=n):
                    out, code, msl = _run(kernel, (a, w), config)
                    _close(out, _mm(a, w))
                    self._assert_mpp(code)
                    self.assertNotIn("_mpp_tile", msl)
        for m, k2, n in ((16, 74, 40), (1, 74, 64)):
            a = torch.randn(m, k2, device=DEVICE)
            b = torch.randn(k2 // 2, n, device=DEVICE)
            parts = torch.stack([torch.exp(b * 0.1), torch.sigmoid(b)], dim=1)
            for config in (
                None,
                helion.Config(block_sizes=[16, 16, 16], num_threads=[4, 16, 16]),
            ):
                with self.subTest(m=m, k2=k2, n=n, config=config):
                    out, _, msl = _run(parts_exp_rhs, (_nan_padded(a), b), config)
                    _close(out, _mm(a, parts.reshape(k2, n)))
                    self._assert_staged(msl, "float")
        for m, k, n in ((16, 37, 40), (1, 37, 64), (33, 64, 40)):
            a = torch.randn(m, 2 * k - 6, device=DEVICE)
            b0, b1 = torch.randn(2, k, n, device=DEVICE)
            parts = torch.stack([torch.exp(b0 * 0.1), torch.sigmoid(b1)], dim=1)
            with self.subTest(m=m, k=k, n=n):
                out, _, _ = _run(parts_kbegin, (_nan_padded(a, before=6), b0, b1), None)
                _close(out, _mm(a, parts[3:].reshape(2 * k - 6, n)))
        for m, lim in ((32, 20), (17, 9)):
            a = torch.randn(m, 64, device=DEVICE)
            b0, b1 = torch.randn(2, 32, 32, device=DEVICE)
            rows = torch.arange(m, device=DEVICE)[:, None] < lim
            rhs = torch.stack([b0, b1], dim=1).reshape(64, 32)
            with self.subTest(kernel=parts_row_masked.name, m=m, lim=lim):
                out, _, msl = _run(
                    parts_row_masked, (_nan_padded(a), b0, b1, lim), None
                )
                _close(out, _mm(torch.where(rows, a, 0), rhs))
                self._assert_staged(msl, "float", tiles=2)

    @_full_threadgroup
    def test_staged_addmm_operands(self) -> None:
        """Computed, gathered and N-D operands of torch.addmm, beside views
        that are NaN past K (a K tail of 72 % 16 = 8), with K on threads and
        on threads x lanes."""
        f16 = torch.float16
        x = torch.randn(48, 72, device=DEVICE, dtype=f16)
        y = torch.randn(72, 40, device=DEVICE, dtype=f16)
        x3 = torch.randn(3, 48, 72, device=DEVICE, dtype=f16)
        rows = torch.randperm(48, device=DEVICE)
        z = torch.randn(72, 40, device=DEVICE, dtype=f16)
        exp_y = torch.exp(y.float()).to(f16)
        cases = (
            (computed_operand, (x, _nan_padded(y)), _mm(x * 2, y)),
            (computed_rhs, (_nan_padded(x), y), _mm(x, exp_y)),
            (
                both_computed,
                (x, y),
                _mm(torch.sigmoid(x.float()).to(f16), y * 2) + 1,
            ),
            (
                both_staged_kshort,
                (x, y),
                _mm(
                    torch.exp(x[:, :67].float() * 0.1).to(f16),
                    torch.exp(y[:67].float() * 0.1).to(f16),
                ),
            ),
            (
                staged_kbegin,
                (x, _nan_padded(y)),
                _mm(torch.exp(x[:, 5:].float() * 0.1).to(f16), y[5:]),
            ),
            (gathered_lhs, (x, y, rows), _mm(x[rows], y)),
            (nd_lhs_computed, (x3, y), _mm(x3[2] * 2, y)),
            (counter_staged, (x, y), None),
            (two_k_loops_staged, (x, y, z), None),
        )
        for bk, k_threads in ((16, 1), (32, 4)):
            for kernel, args, ref in cases:
                # two_k_loops_staged's second K loop is a fourth block.
                extra = 1 if kernel is two_k_loops_staged else 0
                config = helion.Config(
                    block_sizes=[16, 16, bk] + [16] * extra,
                    num_threads=[16, 16, k_threads] + [1] * extra,
                )
                with self.subTest(kernel=kernel.name, bk=bk, k_threads=k_threads):
                    out, _, _ = _run(kernel, args, config)
                    if kernel is two_k_loops_staged:
                        _close(out[0], _mm(x, exp_y))
                        _close(out[1], _mm(x, z * 3))
                    elif kernel is counter_staged:
                        _close(out, _mm(x * 2, y) + -(-72 // bk))
                    else:
                        _close(out, ref)
        a = torch.randn(45, 72, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(72, 50, device=DEVICE, dtype=torch.bfloat16)
        out, _, _ = _run(free_rhs_computed, (a, b, 2, 2), FREE32)
        _close(out, _mm(a, b * 0.5))

    def test_surplus_threads(self) -> None:
        """A wider sibling's surplus threads write nothing: the staging write
        bounds N's local index by its block, 16, not the axis's 32."""
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        z = torch.randn(16, 32, device=DEVICE)
        out, _, msl = _run(staged_sibling_cols, (x, y, z), None)
        _close(out, _mm(x, y * 3))
        self.assertIn("required_threads_per_threadgroup(16, 32, 1)", msl)
        (write,) = [line for line in msl.splitlines() if "{ _mpp_tile[" in line]
        self.assertEqual(len(re.findall(r"- offset_\d+\) < 16\)", write)), 2)

    @_full_threadgroup
    def test_non_view_operands(self) -> None:
        """Operands that are not device views are staged: masks
        that aren't ``index < bound`` on the load's own index, per-row or
        float bounds, ``add`` with ``alpha``, per-row bases and symbolic
        sizes (rows 0..44 of 45, tiles of 32)."""
        a = torch.randn(45, 72, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(72, 90, device=DEVICE, dtype=torch.bfloat16)
        lims = torch.arange(45, device=DEVICE, dtype=torch.int32) % 7 * 7
        cnt = torch.tensor([20], device=DEVICE, dtype=torch.int32)
        off = torch.tensor([0], device=DEVICE, dtype=torch.int32)
        fbound = torch.tensor([22.5], device=DEVICE)
        lim_list = lims.tolist()
        rows = range(45)
        twice = [r // 32 * 32 + 2 * (r % 32) for r in rows]
        cases = (
            (bound_vec, (lims,), [r if r < lim_list[r] else -1 for r in rows]),
            (mask_local, (cnt,), [r if r % 32 < 20 else -1 for r in rows]),
            (mask_shifted, (), [r if r + 8 < 45 else -1 for r in rows]),
            (mask_recomputed, (), list(rows)),
            (mask_halfrows, (), [r if r < 22.5 else -1 for r in rows]),
            (masked_rows, (fbound,), [r if r < 22.5 else -1 for r in rows]),
            (mask_le, (cnt,), [r if r <= 20 else -1 for r in rows]),
            (base_vec, (), twice),
            (alpha_iota, (off,), twice),
            (alpha_base, (off,), [r % 32 + 64 * (r // 32) for r in rows]),
            (free_rows_dynamic, (), list(rows)),
        )
        for kernel, extra, src in cases:
            with self.subTest(kernel=kernel.name):
                out, _, msl = _run(kernel, (a, b, *extra, 2, 3), FREE32)
                _close(out, _rows_ref(a, b, src))
                self.assertIn("threadgroup bfloat _mpp_tile", msl)

    def test_size1_dims(self) -> None:
        """A size-1 M or N on a view rounds the descriptor up to 8; the
        view's extent stays 1 and the element loop skips the rest.  The
        size-1 operand is a slice of a NaN buffer."""
        for (m, k, n), config in (
            (
                (1, 72, 32),
                helion.Config(block_sizes=[16, 32, 16], num_threads=[1, 32, 1]),
            ),
            (
                (32, 72, 1),
                helion.Config(block_sizes=[32, 16, 16], num_threads=[32, 1, 1]),
            ),
        ):
            x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
            y = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
            args = (_nan_padded(x), y) if m == 1 else (x, _nan_padded(y))
            with self.subTest(m=m, n=n):
                out, _, msl = _run(plain_g, args, config)
                _close(out, _mm(x, y))
                desc = "(8, 32, 16," if m == 1 else "(32, 8, 16,"
                self.assertIn(f"matmul2d_descriptor{desc}", msl)
                self.assertRegex(
                    _element_loops(msl),
                    r"if \(_e_m >= \d+ \|\| _e_n >= \d+\) continue;",
                )

    def test_bfloat_staged_from_reloaded_values(self) -> None:
        """A bfloat operand staged in its K loop from loads the loop reloads,
        at a fixed address too, from a constant fill, a scalar argument or a
        block size, and a ':' K view staged in its loop because the loop
        writes its tensor, under a lane jam."""
        bf = torch.bfloat16
        x = torch.randn(48, 72, device=DEVICE, dtype=bf)
        y = torch.randn(72, 128, device=DEVICE, dtype=bf)
        d = torch.randn(48, device=DEVICE)
        xf = x.cpu().float()

        def ref(a: torch.Tensor) -> torch.Tensor:
            return _mm(a.to(bf), y)

        ones = torch.full_like(xf, 1.5)
        q = torch.randn(48, 128, device=DEVICE, dtype=bf)
        k = torch.randn(40, 128, device=DEVICE, dtype=bf)
        q0 = q.clone()
        jam32 = helion.Config(block_sizes=[64, 16], num_threads=[2, 8], num_warps=32)
        jams = (None, JAM64)
        for kernel, args, configs, expected in (
            (kloop_load_exp, (x, y), jams, ref(torch.exp(xf * 0.5))),
            (kloop_load_chain, (x, y), jams, ref(torch.sigmoid(xf) * xf + 0.25)),
            (kloop_const, (x, y), jams, ref(ones)),
            (kloop_load0, (x, y), jams, ref(ones)),
            (kloop_load_inv, (x, y, d), jams, ref(d.cpu()[:, None].expand_as(xf))),
            (kloop_arg_scale, (x, y, 0.5), jams, ref(xf * 0.5)),
            (
                kloop_block_size,
                (x, y),
                (helion.Config(block_sizes=[64, 16, 16], num_warps=32), JAM64),
                ref(xf + 16),
            ),
            (
                dem_colon_inv,
                (q, k),
                (
                    helion.Config(
                        block_sizes=[32, 32], num_threads=[8, 8], num_warps=32
                    ),
                    jam32,
                ),
                _mm(q0, k.T),
            ),
        ):
            for config in configs:
                with self.subTest(kernel=kernel.name, config=config):
                    out, _, msl = _run(kernel, args, config)
                    _close(out, expected)
                    self.assertIn("threadgroup bfloat _mpp_tile", msl)
                    if config is configs[1]:
                        self.assertIn("__lane1", msl)
        torch.testing.assert_close(q, q0)


@_requires_mps
class TestMppStagingRejects(_MppCase):
    """Staging and factored-K forms the planner can't lower raise
    ``BackendUnsupported``."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        self.y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)

    def test_staged_positional(self) -> None:
        z = torch.randn(72, 48, device=DEVICE, dtype=torch.float16)
        self._rejects(
            swapped_staged,
            (self.x, z, self.y),
            TILE16,
            "a staged MPP operand is written into its tile by block",
        )

    def test_lane_carry(self) -> None:
        """A carry over K's lane-looped block beside a staged op, folded or
        read element-wise: each lane continues the previous lane's value,
        which the split's one register per name can't hold."""
        x = torch.randn(48, 72, device=DEVICE)
        y = torch.randn(72, 40, device=DEVICE)
        v = torch.randn(72, device=DEVICE)
        msg = "a lane loop around an MPP op carries carry across lanes"
        for config in (
            None,
            helion.Config(block_sizes=[16, 16, 16], num_threads=[1, 16, 16]),
        ):
            for kernel in (lane_carry_staged, lane_carry_operand):
                self._rejects(kernel, (x, y, v), config, msg)

    def test_operands(self) -> None:
        a = torch.randn(16, 64, device=DEVICE, dtype=torch.float16)
        b0, b1 = torch.randn(2, 32, 32, device=DEVICE, dtype=torch.float16)
        self._rejects(
            dot_parts_noacc,
            (a, b0, b1),
            None,
            "an hl.dot over a K held as per-thread parts with no acc",
        )
        self._rejects(
            dot_parts_out_dtype,
            (a, b0, b1),
            None,
            "an hl.dot over per-thread parts with out_dtype=torch.float16",
        )
        self._rejects(
            computed_rhs,
            (self.x, self.y.float()),
            TILE16,
            "MPP operands of two dtypes",
        )
        self._rejects(
            parts_arange_base,
            (a, b0, b1),
            None,
            "hl.arange() of length 2*u",
        )
        self._rejects(
            full_lhs_parts,
            (a.float(), torch.randn(64, 32, device=DEVICE)),
            None,
            "a staged MPP operand over a K of 2 * a block that is not held as "
            "per-thread parts",
        )
        x = torch.randn(1, 72, device=DEVICE, dtype=torch.float16)
        config = helion.Config(block_sizes=[16, 32, 16], num_threads=[1, 32, 1])
        self._rejects(
            computed_operand,
            (x, self.y),
            config,
            "a staged MPP operand over a size-1 M or N",
        )

    def test_fp8_staged(self) -> None:
        """A staged fp8 operand; rejected at planning, so on the CPU."""
        f8 = torch.float8_e4m3fn
        args = (
            torch.randn(16, 32).to(f8),
            torch.randn(16, 32).to(f8),
            torch.randn(32, 32).to(f8),
            torch.randn(32, 32).to(f8),
        )
        bound = stacked_fp8.bind(args)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "an MPP operand of dtype torch.float8_e4m3fn"
        ):
            bound.to_code(bound.config_spec.default_config())

    def test_budget(self) -> None:
        x = torch.randn(32, 256, device=DEVICE)
        y = torch.randn(256, 64, device=DEVICE)
        config = helion.Config(block_sizes=[16, 64, 128], num_threads=[16, 64, 1])
        self._rejects(
            both_computed,
            (x, y),
            config,
            "reduction scratch and MPP staging need 40960 bytes of threadgroup memory",
        )

    def test_k_tile(self) -> None:
        """matmul2d needs K a multiple of 16: K = 1, and a K tile of 3 x 8."""
        msg = "K a multiple of 16"
        x = torch.randn(48, 1, device=DEVICE, dtype=torch.float16)
        y = torch.randn(1, 40, device=DEVICE, dtype=torch.float16)
        self._rejects(plain_g, (x, y), TILE16, msg)
        a = torch.randn(16, 96, device=DEVICE, dtype=torch.float16)
        bs = [torch.randn(32, 32, device=DEVICE, dtype=torch.float16) for _ in range(3)]
        self._rejects(
            dot_parts_c3, (a, *bs), helion.Config(block_sizes=[8, 16, 16]), msg
        )


class _FloatStagingCase(_MppCase):
    """bfloat inputs shared by the float32-staging test classes."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(48, 72, device=DEVICE, dtype=torch.bfloat16)
        self.y = torch.randn(72, 128, device=DEVICE, dtype=torch.bfloat16)
        self.d = torch.randn(48, device=DEVICE)


@_requires_mps
class TestMppFloatStaging(_FloatStagingCase):
    """A bfloat operand staged inside a for loop from values the loop doesn't
    reload is staged into a float32 tile of its bfloat values, with each
    float -> bfloat conversion on its way rounded in integer arithmetic.  The
    configs at 32 and 64 lane copies are ones where a bfloat tile, or a
    float32 tile of Metal's own conversions, read back wrong."""

    def _rows(self, values: torch.Tensor) -> torch.Tensor:
        """``values``, rounded to bfloat and broadcast over K, times y."""
        return _bf(values)[:, None] * _d(self.y).sum(0)

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        configs: tuple[helion.Config | None, ...],
        expected: Callable[[list[int]], torch.Tensor],
        *,
        whole: bool = True,
        fenced: str | None = None,
        unfenced: bool = False,
    ) -> None:
        """Run ``kernel`` at each config against ``expected(block_sizes)``.
        No float -> bfloat conversion of a computed value reaches its float32
        tile, nor, with ``whole``, appears anywhere in the kernel.  A
        device-memory barrier comes before the first read of ``fenced``; with
        ``unfenced`` the kernel has none."""
        for config in configs:
            with self.subTest(kernel=kernel.name, config=config):
                blocks = (
                    config or kernel.bind(args).config_spec.default_config()
                ).config["block_sizes"]
                out, _, msl = _run(kernel, args, config)
                _close(out, expected(list(blocks)))
                writes = _FLOAT_WRITE.findall(msl)
                self.assertTrue(writes)
                self.assertTrue(_staged_as_float(msl))
                for expr in writes + _float_slice(msl):
                    self.assertNotRegex(expr, _BF_CAST)
                if whole:
                    self.assertNotRegex(msl[msl.index("kernel void") :], _BF_CAST)
                if fenced is not None:
                    self.assertTrue(_fenced(msl, fenced))
                if unfenced:
                    self.assertNotIn("mem_flags::mem_device", msl)

    def test_values_not_reloaded(self) -> None:
        """A row loaded or computed before the K loop, a carry, a tile's
        index or offset, a fill of a scalar loaded before the K loop, a carry
        seeded by a constant fill, and a load in an outer loop before the K
        loop."""
        x, y, d = self.x, self.y, self.d
        s = torch.randn(1, device=DEVICE)

        def rows(values: torch.Tensor) -> Callable[[list[int]], torch.Tensor]:
            return lambda _: self._rows(values)

        def begins(bl: list[int]) -> torch.Tensor:
            return self._rows(torch.arange(48) // bl[0] * bl[0] * 1.0)

        def carry_zero(bl: list[int]) -> torch.Tensor:
            tiles = _k_tiles(72, bl[2])
            return sum(
                _bf(t + x[:, ks].float()) @ _d(y)[ks] for t, ks in enumerate(tiles)
            )

        e = torch.exp(d - 1.0)
        k16 = helion.Config(block_sizes=[64, 16, 16], num_threads=[2, 2, 8])
        k32 = helion.Config(block_sizes=[64, 16, 32], num_threads=[2, 2, 8])
        for kernel, args, configs, expected in (
            (kloop_bcast, (x, y, d), (None, k16), rows(d)),
            (kloop_exp, (x, y, d), (None, k32, JAM64), rows(e)),
            (kloop_exp_pre, (x, y, d), (None, k32, JAM64), rows(e)),
            (kloop_carry_id, (x, y, d), (None, k16), rows(d)),
            (
                kloop_chain,
                (x, y, d),
                (
                    None,
                    helion.Config(block_sizes=[64, 16, 32], num_threads=[2, 8, 2]),
                    helion.Config(block_sizes=[128, 16, 16], num_threads=[2, 8, 8]),
                ),
                rows(torch.sigmoid(d) * d + 0.5),
            ),
            (kloop_tile_index, (x, y), (None, k32, JAM64), rows(torch.arange(48.0))),
            (kloop_outer_begin, (x, y), (None, JAM64), begins),
            (kloop_full_scalar, (x, y, s), (None, JAM64), rows(s.expand(48))),
            (kloop_carry_zero, (x, y), (None, JAM64), carry_zero),
            (
                kloop_between_loops,
                (x, y, d),
                (None, JAM64),
                lambda _: 2 * self._rows(d),
            ),
        ):
            self._check(kernel, args, configs, expected)

    @_full_threadgroup
    def test_colon_forms(self) -> None:
        """Ops over a ':' M, N or K: a staged rhs over the N loop's own tile
        from a column loaded or computed before the loop, a carry in an
        hl.grid loop, a row loaded before the K loop, and one loaded before
        the N loop, with no conversion."""
        bf = torch.bfloat16
        # The tiles on 8 x 8 threads leave the ':' dim 16 of 1024.
        tiles8 = helion.Config(block_sizes=[32, 32], num_threads=[8, 8], num_warps=32)
        q = torch.randn(48, 128, device=DEVICE, dtype=bf)
        k = torch.randn(40, 128, device=DEVICE, dtype=bf)
        self._check(
            nest_written,
            (q, k),
            (
                helion.Config(block_sizes=[16, 16], num_threads=[16, 16], num_warps=32),
                helion.Config(block_sizes=[32, 16], num_threads=[2, 16], num_warps=32),
            ),
            lambda _: 2 * (_d(q) @ _d(k).T),
        )
        c = torch.randn(128, device=DEVICE)
        cols = torch.zeros(40, device=DEVICE)
        for kernel, scale in ((rhs_over_loop_block, 1.0), (rhs_shared_block, 0.5)):
            self._check(
                kernel,
                (q, c, cols),
                (tiles8,),
                lambda _, s=scale: (_d(q) @ _bf(c * s))[:, None].expand(48, 40),
            )
        x64 = torch.randn(48, 64, device=DEVICE, dtype=bf)
        y64 = torch.randn(64, 40, device=DEVICE, dtype=bf)
        self._check(
            colon_staged_carry,
            (x64, y64, torch.zeros(3, 48, 40, device=DEVICE)),
            (None,),
            lambda _: torch.stack([_bf(_d(x64) * 0.5**t) @ _d(y64) for t in range(3)]),
        )
        x, y, d = self.x, self.y, self.d
        self._check(
            kb_colon_n_lhs,
            (x, y, d),
            (
                None,
                helion.Config(block_sizes=[128, 16], num_threads=[2, 2], num_warps=32),
            ),
            lambda _: self._rows(d),
        )
        e = torch.randn(128, device=DEVICE)
        self._check(
            kb_colon_n_rhs,
            (x, e),
            (tiles8,),
            lambda _: _d(x).sum(1)[:, None] * _bf(e)[None, :],
        )
        xm = torch.randn(64, 72, device=DEVICE, dtype=bf)
        ym = torch.randn(72, 40, device=DEVICE, dtype=bf)
        dm = torch.randn(64, device=DEVICE)
        self._check(
            kb_colon_m_lhs,
            (xm, ym, dm),
            (
                None,
                helion.Config(block_sizes=[128, 32], num_threads=[1, 2], num_warps=32),
                helion.Config(block_sizes=[64, 32], num_threads=[1, 1], num_warps=32),
            ),
            lambda _: _bf(dm)[:, None] * _d(ym).sum(0),
        )

    def test_conversions(self) -> None:
        """Conversions copied, stored, chained, broadcast by subscript, read
        back as float and as the index of a gather from a tensor the kernel
        doesn't write (so with no device barrier), a float kernel argument
        before one, a conversion loaded back from device memory before the K
        loop, and a float16 conversion, which isn't rounded as a bfloat one."""
        x, y, d = self.x, self.y, self.d
        e = torch.exp(d - 1.0)

        def halved(bl: list[int]) -> torch.Tensor:
            tiles = _k_tiles(72, bl[2])
            return sum(
                _bf(d * 0.5**t)[:, None] * _d(y)[ks].sum(0)
                for t, ks in enumerate(tiles)
            )

        self._check(
            kloop_carry_vary,
            (x, y, d),
            (
                helion.Config(block_sizes=[64, 32, 16]),
                helion.Config(block_sizes=[64, 16, 16], num_threads=[1, 8, 8]),
            ),
            halved,
        )
        mix = _bf(_bf(e)[:, None] + _d(x) * 0.5) @ _d(y)
        for kernel, args, configs, expected in (
            (kloop_newvar, (x, y, d), (JAM64,), self._rows(e)),
            (two_cvt, (x, y, d), (JAM64,), self._rows(_bf(e) * 3.0 + 0.25)),
            (cvt_subscript, (x, y, d), (JAM64,), self._rows(e)),
            (
                cvt_float_mix,
                (x, y, d),
                (helion.Config(block_sizes=[128, 16, 16], num_threads=[4, 2, 8]),),
                mix,
            ),
            (kloop_symf, (x, y, d, 0.75), (None, JAM64), self._rows(d * 0.75)),
        ):
            self._check(kernel, args, configs, lambda _, v=expected: v)
        h = (d.cpu()[:, None] * x.cpu().float() * 1e-6).to(torch.float16).float()
        self._check(half_sub, (x, y, d), (None, JAM64), lambda _: _bf(h * 1e6) @ _d(y))
        self._check(
            store_load_pre32, (x, y, d), (JAM64,), lambda _: self._rows(e), whole=False
        )
        rows = torch.rand(48, device=DEVICE) * 7.0
        t = torch.randn(8, device=DEVICE)
        i = _bf(rows).long()
        self._check(
            gather_cvt,
            (x, y, rows, t),
            (None, JAM64),
            lambda _: self._rows(i + _d(t)[i]),
            unfenced=True,
        )
        side = torch.zeros(48, 72, device=DEVICE, dtype=torch.bfloat16)
        self._check(
            kloop_cvt_store,
            (x, y, d, side),
            (None,),
            lambda _: self._rows(e),
            whole=False,
        )
        torch.testing.assert_close(_d(side), _bf(e)[:, None].expand(48, 72))

    def test_rounding_specials(self) -> None:
        """Infinities, NaNs whose bits the rounding increment would turn
        into an infinity or, by a carry, a zero, and the largest float, which
        rounds to an infinity, each a row against a positive rhs.  A select
        keeps each NaN's bits: Metal's arithmetic makes every NaN the same."""
        bits = [0x7F800000, 0xFF800000, 0x7F800001, 0x7FFFFFFF, 0xFFFFFFFF, 0x7F7FFFFF]
        special = torch.tensor(bits, dtype=torch.int64).to(torch.int32)
        d = self.d.clone()
        d[: len(bits)] = special.view(torch.float32).to(DEVICE)
        y = (torch.rand(72, 128, device=DEVICE) + 0.5).to(torch.bfloat16)
        for config in (None, JAM64):
            with self.subTest(config=config):
                out, _, msl = _run(kloop_select, (self.x, y, d), config)
                self.assertIn("threadgroup float _mpp_tile", msl)
                torch.testing.assert_close(
                    _d(out),
                    _bf(d)[:, None] * _d(y).sum(0),
                    atol=0,
                    rtol=1e-3,
                    equal_nan=True,
                )

    @_full_threadgroup
    def test_planned_nodes_on_the_path(self) -> None:
        """Nodes the plan holds by identity on the way from a conversion to
        the operand: the rewrite's copy of a fenced gather keeps its barrier,
        and another op's matmul and updated acc, left uncopied, keep their
        own inputs."""
        x, y = self.x, self.y
        rows = torch.rand(48, device=DEVICE) * 7.0
        t = torch.randn(8, device=DEVICE)
        i = _bf(rows).long()
        self._check(
            gather_written_cvt,
            (x, y, rows, t, torch.zeros_like(t)),
            (helion.Config(block_sizes=[8, 32, 16, 16], num_threads=[8, 32, 16, 1]),),
            lambda _: self._rows(i + _d(t)[i]),
            fenced="t2",
        )
        bf = torch.bfloat16
        q = torch.randn(48, 64, device=DEVICE, dtype=bf)
        kt = torch.randn(64, 80, device=DEVICE, dtype=bf)
        v = torch.randn(80, 64, device=DEVICE, dtype=bf)

        def two_ops(bl: list[int]) -> torch.Tensor:
            acc = 0
            for ns in _k_tiles(80, bl[1]):
                b = _bf(_d(kt)[:, ns] * 2.0)
                p = _bf((_d(q) @ b) * 0.125 + _d(self.d)[:, None] + b.sum(0))
                acc = acc + p @ _d(v)[ns]
            return acc

        self._check(
            mm_reads_cvt,
            (q, kt, v, self.d),
            (
                helion.Config(block_sizes=[16, 16], num_threads=[16, 16]),
                helion.Config(block_sizes=[32, 16], num_threads=[32, 16]),
            ),
            two_ops,
            whole=False,
        )
        x80 = torch.randn(48, 80, device=DEVICE, dtype=bf)
        w = torch.randn(64, 80, device=DEVICE, dtype=bf)
        c = _bf(self.d * 0.5)[:, None]
        for config in (
            None,
            helion.Config(block_sizes=[16, 16], num_threads=[16, 16]),
            helion.Config(block_sizes=[16, 32], num_threads=[16, 32]),
        ):
            with self.subTest(kernel=upd_reads_cvt.name, config=config):
                blocks = (
                    config
                    or upd_reads_cvt.bind(
                        (x80, v, w, self.d)
                    ).config_spec.default_config()
                ).config["block_sizes"]
                acc = torch.zeros(48, 64, dtype=torch.float64)
                ref2 = torch.zeros(48, 80, dtype=torch.float64)
                for ks in _k_tiles(80, blocks[1]):
                    acc = acc * c
                    ref2[:, ks] = _bf(acc + c) @ _d(w)[:, ks]
                    acc = acc + _d(x80)[:, ks] @ _d(v)[ks]
                (out, out2), _, msl = _run(upd_reads_cvt, (x80, v, w, self.d), config)
                _close(out, acc)
                _close(out2, ref2)
                writes = _FLOAT_WRITE.findall(msl)
                self.assertTrue(writes)
                for expr in writes + _float_slice(msl):
                    self.assertNotRegex(expr, _BF_CAST)

    def test_other_ops_carry(self) -> None:
        """An operand computed from another op's acc carry as it was before
        that op, which the operand reads through the op's materialization."""
        bf = torch.bfloat16
        x = torch.randn(48, 80, device=DEVICE, dtype=bf)
        v = torch.randn(80, 64, device=DEVICE, dtype=bf)
        w = torch.randn(64, 80, device=DEVICE, dtype=bf)
        args = (x, v, w, self.d)
        for config in (
            None,
            helion.Config(block_sizes=[16, 16], num_threads=[16, 16]),
            helion.Config(block_sizes=[16, 32], num_threads=[16, 32]),
        ):
            with self.subTest(config=config):
                blocks = (
                    config or mat_carry.bind(args).config_spec.default_config()
                ).config["block_sizes"]
                acc = _d(self.d)[:, None].expand(48, 64)
                ref2 = torch.zeros(48, 80, dtype=torch.float64)
                for ks in _k_tiles(80, blocks[1]):
                    ref2[:, ks] = _bf(acc * 0.5) @ _d(w)[:, ks]
                    acc = acc + _d(x)[:, ks] @ _d(v)[ks]
                (out, out2), _, msl = _run(mat_carry, args, config)
                _close(out, acc)
                _close(out2, ref2)
                self.assertTrue(_FLOAT_WRITE.findall(msl))
                self.assertTrue(_staged_as_float(msl))
                self.assertNotRegex(msl[msl.index("kernel void") :], _BF_CAST)

    def test_attention(self) -> None:
        """``examples/attention.py`` in bfloat: p is staged in its N loop
        from the scores and the row maximum, a carry."""
        bf = torch.bfloat16
        q = torch.randn(1, 2, 64, 64, device=DEVICE, dtype=bf)
        k = torch.randn(1, 2, 80, 64, device=DEVICE, dtype=bf)
        v = torch.randn(1, 2, 80, 64, device=DEVICE, dtype=bf)
        scale = 1.0 / math.sqrt(64) * 1.44269504

        def online(b: int, bn: int) -> torch.Tensor:
            qs = _bf(_d(q)[0, b] * scale)
            m_i = torch.full((64,), -math.inf, dtype=torch.float64)
            l_i = torch.ones(64, dtype=torch.float64)
            acc = torch.zeros(64, 64, dtype=torch.float64)
            for ns in _k_tiles(80, bn):
                qk = qs @ _d(k)[0, b, ns].T
                m_ij = torch.maximum(m_i, qk.amax(-1))
                p = torch.exp2(qk - m_ij[:, None])
                alpha = torch.exp2(m_i - m_ij)
                l_i = l_i * alpha + p.sum(-1)
                acc = acc * alpha[:, None] + _bf(p) @ _d(v)[0, b, ns]
                m_i = m_ij
            return _bf(acc / l_i[:, None])

        expected = torch.stack([online(b, 16) for b in range(2)])[None]
        for bm in (16, 32):
            config = helion.Config(block_sizes=[1, bm, 16], num_threads=[1, bm, 16])
            with self.subTest(config=config):
                (out, _), _, msl = _run(attention_metal, (q, k, v), config)
                _close(out, expected)
                writes = _FLOAT_WRITE.findall(msl)
                self.assertTrue(writes)
                for expr in writes + _float_slice(msl):
                    self.assertNotRegex(expr, _BF_CAST)


@_requires_mps
class TestMppFloatStagingRejects(_FloatStagingCase):
    """bfloat operands staged inside a loop that stay rejected: a bfloat
    value on the operand's chain that isn't a conversion the planner rounds,
    or a producer it can't see."""

    def test_not_followed(self) -> None:
        """A value made by a for loop before the K loop or inside it, with or
        without a conversion, by an if, by a while loop, entering a while
        loop, or by a node with two outputs."""
        x, y, d = self.x, self.y, self.d
        one = torch.tensor([1], device=DEVICE, dtype=torch.int32)
        c = torch.zeros(1, device=DEVICE)
        for kernel, args, source in (
            (loop_pre_cvt, (x, y, d, c), "the output of _for_loop"),
            (inner_loop_cvt, (x, y, d, c), "the output of _for_loop"),
            (loop_pre_nocvt, (x, y, d, c), "the output of _for_loop"),
            (if_else_cvt, (x, y, d, one), "the output of _if"),
            (while_before_cvt, (x, y, d, one), "the output of _while_loop"),
            (while_pre_cvt, (x, y, d, one), "a while loop's input"),
        ):
            self._rejects_at(kernel, args, (None,), f"from {source}, {_NOT_FOLLOWED}")
        self._rejects_at(
            maxdim_pre,
            (x, y, torch.randn(48, 16, device=DEVICE)),
            (None,),
            _NOT_FOLLOWED,
        )

    def test_computed(self) -> None:
        """bfloat math on the operand's chain: exp of a bfloat row and of a
        conversion, a product with a float kernel argument or a constant, a
        fast-math division, a bfloat fill and a where.  The jam-64 configs
        are ones where the first three, admitted, miscompiled."""
        x, y, d = self.x, self.y, self.d
        db = d.to(torch.bfloat16)
        for kernel, args, configs, op in (
            (kloop_bf_exp, (x, y, db), (None, JAM64), "exp.default"),
            (kloop_cvt_then_exp, (x, y, db), (None, JAM64), "exp.default"),
            (bf_mul_symf, (x, y, d, 0.75), (None, JAM64), "mul.Tensor"),
            (bf_div_fast, (x, y, d), (None,), "div.Tensor"),
            (kloop_full_bf16, (x, y, torch.randn(1, device=DEVICE)), (None,), "full"),
            (kloop_cvt_mul, (x, y, d), (None,), "mul.Tensor"),
            (kloop_exp_where, (x, y, d), (None,), "where.self"),
        ):
            self._rejects_at(kernel, args, configs, f"{_COMPUTED} {op} that")

    def test_converted(self) -> None:
        """A conversion before the K loop, in another graph of the loop, or
        in a carry: a bfloat one, or in a float one's seed or update, also
        when the operand reads the update's conversion itself, and when the
        carry is another op's acc, read before that op or after it.
        The jam-64 config is one where the first, admitted, miscompiled."""
        x, y, d = self.x, self.y, self.d
        self._rejects_at(pre_cvt_float, (x, y, d), (None, JAM64), _CONVERTED)
        q = torch.randn(48, 128, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(40, 128, device=DEVICE, dtype=torch.bfloat16)
        self._rejects_at(colon_if_cvt, (q, k), (None,), _CONVERTED)
        for kernel in (carry_cvt_float, carry_seed_cvt, carry_update_cvt):
            self._rejects_at(kernel, (x, y, d), (None,), _CONVERTED)
        self._rejects_at(
            carry_and_direct,
            (x, y, d),
            (
                None,
                helion.Config(
                    block_sizes=[16, 16, 16], num_threads=[1, 128, 16], num_warps=4
                ),
            ),
            _CONVERTED,
        )
        bf = torch.bfloat16
        args = (
            torch.randn(48, 80, device=DEVICE, dtype=bf),
            torch.randn(80, 64, device=DEVICE, dtype=bf),
            torch.randn(64, 80, device=DEVICE, dtype=bf),
            d,
        )
        tile16 = helion.Config(block_sizes=[16, 16], num_threads=[16, 16])
        for kernel in (mat_carry_cvt, mat_carry_cvt_rev):
            self._rejects_at(kernel, args, (None, tile16), _CONVERTED)

    def test_factored_k(self) -> None:
        """An operand over a K held as per-thread parts."""
        w = torch.randn(144, 128, device=DEVICE, dtype=torch.bfloat16)
        self._rejects_at(parts_bcast, (self.x, w, self.d), (None,), _NOT_RELOADED)


def _layouts(msl: str) -> tuple[int, int, int]:
    """``(element-loop stores, materializations, staged seeds)`` in ``msl``:
    each is one loop over a cooperative tensor's elements."""
    mats = len(re.findall(r"threadgroup float _mpp_mat\w*\[", msl))
    seeds = len(re.findall(r"threadgroup float _mpp_seed\w*\[", msl))
    return msl.count("is_valid_element") - mats - seeds, mats, seeds


def _fenced(msl: str, tensor: str) -> bool:
    """Whether a device-memory barrier comes before the first read of
    ``tensor``."""
    lines = msl.splitlines()
    reads = [i for i, line in enumerate(lines) if re.search(rf"= .*\b{tensor}\[", line)]
    fences = [i for i, line in enumerate(lines) if "mem_flags::mem_device" in line]
    return bool(reads and fences) and fences[0] < reads[0]


def _tile_copies_ordered(msl: str) -> bool:
    """Whether every copy between a cooperative tensor and a threadgroup tile
    sits between two threadgroup barriers and skips the elements past its
    block, a rounded descriptor's padding."""
    lines = [line.strip() for line in msl.splitlines()]
    barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
    for i, line in enumerate(lines):
        if re.search(r"\b_mpp_(?:mat|seed)\w*\[_e_m \* ", line):
            if not (
                lines[i - 1].startswith("if (_e_m >= ")
                and lines[i - 6].startswith("for (auto _it = ")
                and lines[i - 7] == barrier
                and lines[i + 1] == "}"
                and lines[i + 2] == barrier
            ):
                return False
    return True


def _tile_row_sums(t: torch.Tensor, block: int) -> torch.Tensor:
    """Each element replaced by its row's sum over its ``block``-wide column
    tile."""
    return torch.cat(
        [c.sum(-1, keepdim=True).expand_as(c) for c in t.split(block, dim=1)], dim=1
    )


def _tile_row_max(t: torch.Tensor, block: int) -> torch.Tensor:
    """Each element replaced by its row's max over its ``block``-wide column
    tile."""
    return torch.cat(
        [c.amax(-1, keepdim=True).expand_as(c) for c in t.split(block, dim=1)], dim=1
    )


def _tile_row_positive(t: torch.Tensor, block: int) -> torch.Tensor:
    """Whether each element's row has a positive value in its ``block``-wide
    column tile."""
    return _tile_row_max(t, block).gt(0)


#: Row tiles over a whole N of 64.
ROWS = (
    helion.Config(block_sizes=[16, 16], num_threads=[16, 1]),
    helion.Config(block_sizes=[16, 32], num_threads=[16, 1]),
)
TILE16_2K = helion.Config(block_sizes=[16, 16, 16, 16], num_threads=[16, 16, 1, 1])
SIBLING64 = helion.Config(
    block_sizes=[16, 64, 16, 16, 16], num_threads=[16, 64, 16, 16, 1]
)


@_requires_mps
class TestMppConversions(unittest.TestCase):
    """A use an element loop can't evaluate at any element reads the
    cooperative value through a materialization into the Scalar layout, taken
    where the version it reads is defined; a seed that is not a constant fill
    is staged into the cooperative tensor."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        self.y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        self.c = torch.randn(48, 40, device=DEVICE)
        self.cd = self.c.cpu().double()
        self.rt = _mm(self.x, self.y)

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        *refs: tuple[torch.Tensor | None, torch.Tensor],
        atol: float = 2e-2,
        layouts: tuple[int, int, int] | None = None,
        fenced: str | None = None,
    ) -> None:
        """Run ``kernel`` and compare each ``(tensor, ref)``: an argument the
        kernel writes, or its output for None.  ``layouts`` pins
        :func:`_layouts`, and ``fenced`` a tensor whose first read
        :func:`_fenced` must order.  Races rarely show in results, so
        :func:`_tile_copies_ordered` checks every kernel's barriers."""
        with self.subTest(kernel=kernel.name, config=config):
            out, _, msl = _run(kernel, args, config)
            self.assertTrue(_tile_copies_ordered(msl), msl)
            for got, ref in refs:
                torch.testing.assert_close(
                    (out if got is None else got).cpu().double(),
                    ref,
                    atol=atol,
                    rtol=2e-2,
                )
            if layouts is not None:
                self.assertEqual(_layouts(msl), layouts)
            if fenced is not None:
                self.assertTrue(_fenced(msl, fenced), msl)

    @_full_threadgroup
    def test_row_reductions(self) -> None:
        """Row reductions of the result, after the K loop and in it."""
        y = torch.randn(72, 64, device=DEVICE, dtype=torch.float16)
        rt = _mm(self.x, y)
        for config, bk in zip(ROWS, (16, 32), strict=True):
            self._check(row_sum_after, (self.x, y), config, (None, rt.sum(-1)))
            self._check(
                mm_softmax,
                (self.x, y),
                config,
                (None, torch.softmax(rt, -1)),
                atol=1e-3,
            )
            rs = torch.zeros(48, device=DEVICE)
            partial = sum(
                _mm(self.x[:, : j + bk], y[: j + bk]).sum(-1) for j in range(0, 72, bk)
            )
            self._check(
                in_loop_rowsum,
                (self.x, y, rs),
                config,
                (None, rt),
                (rs, partial),
                atol=0.5,
            )
            rs = torch.zeros(48, device=DEVICE)
            self._check(
                mat_then_store_and_sum,
                (self.x, y, rs),
                config,
                (None, 2 * rt),
                (rs, rt.sum(-1)),
                atol=0.5,
                layouts=(1, 1, 0),
            )

    @_full_threadgroup
    def test_mixed_uses(self) -> None:
        """A value an element loop computes beside one in the Scalar layout:
        a use of a value that leaves the element loop leaves it too, and so
        does a value left with no element-loop use."""
        x, rt = self.x, self.rt
        y48 = torch.randn(72, 48, device=DEVICE, dtype=torch.float16)
        y64 = torch.randn(72, 64, device=DEVICE, dtype=torch.float16)
        rt48, rt64 = _mm(x, y48), _mm(x, y64)
        row = ROWS[0]
        s = torch.zeros(48, device=DEVICE)
        self._check(
            mixed_prune, (x, y48, s), row, (None, 3 * rt48), (s, (2 * rt48).sum(-1))
        )
        s = torch.zeros(48, device=DEVICE)
        self._check(
            diamond3,
            (x, y64, s),
            row,
            (None, (2 * rt64 + 1) * rt64),
            (s, (2 * rt64).sum(-1)),
        )
        s = torch.zeros(48, device=DEVICE)
        self._check(
            chain_prune,
            (x, y48, s),
            row,
            (None, 2 * rt48 + 1),
            (s, (2 * rt48).sum(-1)),
            layouts=(0, 1, 0),
        )
        for config, bn in ((TILE16, 16), (TILE32, 32)):
            s = torch.zeros(48, 64, device=DEVICE)
            self._check(
                mixed_prune2,
                (x, y64, s),
                config,
                (None, 3 * rt64),
                (s, 2 * rt64 * _tile_row_sums(2 * rt64, bn)),
            )
            s = torch.zeros(48, 40, device=DEVICE)
            self._check(
                diamond_sibling,
                (x, self.y, s),
                config,
                (None, 3 * rt + 1),
                (s, (rt + 1) * _tile_row_sums(rt + 1, bn)),
            )
            s = torch.zeros(48, 40, device=DEVICE)
            o2 = torch.zeros(48, 40, device=DEVICE)
            self._check(
                diamond_partial,
                (x, self.y, s, o2),
                config,
                (None, rt + 1),
                (o2, 3 * rt),
                (s, 2 * rt * _tile_row_sums(2 * rt, bn)),
                layouts=(1, 1, 0),
            )

    @_full_threadgroup
    def test_operands_no_thread_recomputes(self) -> None:
        """Epilogue operands that are per-element or per-row carries."""
        x, y, c, cd, rt = self.x, self.y, self.c, self.cd, self.rt
        bias = torch.randn(40, device=DEVICE)
        rowsum = x.cpu().double().sum(-1)[:, None]
        for config in (TILE16, TILE32):
            self._check(
                carry_vary,
                (x, y, c, bias),
                config,
                (None, rt + 5 * cd + bias.cpu().double()),
            )
            out = torch.zeros(48, 40, device=DEVICE)
            self._check(carry_vary_in, (x, y, c, out), config, (out, rt + 5 * cd))
            self._check(mm_rowscale, (x, y), config, (None, rt * rowsum), atol=0.5)

    def test_older_versions(self) -> None:
        """The seed read after the K loop updated it in place, and the carry
        read in the body after the op updated it."""
        x, y, rt = self.x, self.y, self.rt
        for config in (TILE16, TILE32):
            self._check(stale_seed, (x, y), config, (None, rt + 1), layouts=(0, 2, 0))
            out = torch.zeros(48, 40, device=DEVICE)
            self._check(
                delta_in_loop, (x, y, out), config, (out, _mm(x[:, 64:], y[64:]))
            )

    def test_storage_across_levels(self) -> None:
        """The accumulator carried by an outer loop, given back by a uniform
        if, read by the else branch, or bound to another name that leaves the
        if or the loop."""
        x, y, c, cd, rt = self.x, self.y, self.c, self.cd, self.rt
        for config in (TILE16, TILE32):
            self._check(outer_grid_carry, (x, y), config, (None, 2 * rt))
            self._check(alias_in_grid, (x, y, c), config, (None, 2 * rt + 1))
            for f in (1, 0):
                flag = torch.tensor([f], device=DEVICE, dtype=torch.int32)
                self._check(
                    if_output_vary,
                    (x, y, c, flag),
                    config,
                    (None, rt + 5 * cd if f else torch.zeros_like(cd)),
                )
                self._check(
                    alias_in_if, (x, y, c, flag), config, (None, (rt if f else cd) + 1)
                )
                o2 = torch.zeros(48, 40, device=DEVICE)
                self._check(
                    else_reads,
                    (x, y, c, flag, o2),
                    config,
                    (None, cd + rt if f else cd),
                    (o2, torch.zeros_like(cd) if f else 3 * cd),
                )

    def test_memory_ordering(self) -> None:
        """A store whose element loop would race with another access of a
        tensor the kernel writes takes the Scalar layout, also when it reads
        that tensor only through an epilogue load's index or mask, and an
        operand view of a written tensor is staged."""
        x, y, c, cd, rt = self.x, self.y, self.c, self.cd, self.rt
        bias = torch.randn(40, device=DEVICE)
        out2 = torch.zeros(48, 40, device=DEVICE)
        ref = rt + bias.cpu().double()
        self._check(
            readback,
            (x, y, bias, out2),
            TILE16,
            (None, ref),
            (out2, 2 * ref),
            layouts=(0, 1, 0),
        )
        tmp = torch.zeros(48, 40, device=DEVICE)
        self._check(
            prestore,
            (x, y, c, tmp),
            TILE16,
            (None, rt + 2 * self.cd),
            layouts=(0, 1, 0),
        )
        with self.subTest(kernel="view_written"):
            x2 = torch.full_like(x, float("nan"))
            out, _, msl = _run(view_written, (x, y, x2), TILE16_2K)
            _close(out, 2 * rt)
            self.assertIn("tensor<threadgroup half, dextents", msl)
            self.assertTrue(_fenced(msl, "x2"), msl)
        perm = torch.randperm(48, device=DEVICE)
        flag_src = (torch.rand(48, device=DEVICE) > 0.5).float()
        gathered = rt + cd[perm.cpu()]
        flagged = rt + torch.where(flag_src.cpu()[:, None] > 0, cd, 0.0)
        for config in (TILE16, TILE32):
            rows = torch.zeros(48, device=DEVICE, dtype=torch.int64)
            self._check(
                gather_written_index,
                (x, y, perm, rows, c),
                config,
                (None, gathered),
                layouts=(0, 1, 0),
            )
            flag = torch.zeros(48, device=DEVICE)
            self._check(
                load_under_written_mask,
                (x, y, flag_src, flag, c),
                config,
                (None, flagged),
                layouts=(0, 1, 0),
            )

    def test_fenced_staged_reads(self) -> None:
        """A staged operand read from a tensor that other threads of the
        threadgroup wrote: on the lhs, the rhs, through a computed value,
        through a gather index, or written by another K loop's store.  Each
        written tensor starts as NaN, and its first read follows a
        device-memory barrier (a race passes on the GPU too often to pin by
        results alone)."""
        f16 = torch.float16
        x = torch.randn(48, 32, device=DEVICE, dtype=f16)
        y = torch.randn(32, 32, device=DEVICE, dtype=f16)
        ref = 2 * _mm(x, y)
        for bm in (16, 32):
            config = helion.Config(block_sizes=[bm, 32, 16], num_threads=[bm, 32, 1])
            for kernel in (vw_xthread, staged_computed_written):
                x2 = torch.full_like(x, float("nan"))
                self._check(kernel, (x, y, x2), config, (None, ref), fenced="x2")
            y2 = torch.full_like(y, float("nan"))
            config = helion.Config(
                block_sizes=[16, 32, bm, 32, 16], num_threads=[16, 32, bm, 32, 1]
            )
            self._check(rhs_written, (x, y, y2), config, (None, ref), fenced="y2")
        idx = torch.randperm(48, device=DEVICE).to(torch.int32)
        config = helion.Config(
            block_sizes=[32, 16, 32, 16], num_threads=[32, 16, 32, 1]
        )
        self._check(
            gather_idx_written,
            (x, y, idx, torch.zeros_like(idx)),
            config,
            (None, _mm(x[idx.long()], y)),
            fenced="idx2",
        )
        xa = torch.randn(48, 40, device=DEVICE, dtype=f16)
        ya = torch.randn(40, 32, device=DEVICE, dtype=f16) * 0.2
        wa = torch.randn(32, 32, device=DEVICE, dtype=f16)
        ref = _mm(xa, ya).half().double() @ wa.cpu().double()
        for bj in (16, 32):
            config = helion.Config(
                block_sizes=[16, 32, 16, bj], num_threads=[16, 32, 1, 1]
            )
            t = torch.full((48, 32), float("nan"), device=DEVICE, dtype=f16)
            self._check(
                demote_then_view, (xa, ya, wa, t), config, (None, ref), fenced="t"
            )

    def test_outer_written_load(self) -> None:
        """A written tensor's load before the K loop runs in the Scalar layout,
        unfenced, and the staged operand reads its value, with M and K tails."""
        f16 = torch.float16
        for m, k, n in ((48, 40, 72), (45, 72, 40)):
            x = torch.randn(m, k, device=DEVICE, dtype=f16)
            y = torch.randn(k, n, device=DEVICE, dtype=f16)
            s_src = torch.randn(m, device=DEVICE)
            xs = (x.float() * (s_src * 2.0).half().float()[:, None]).half()
            for config in (
                helion.Config(block_sizes=[16, 16, 16], num_threads=[16, 16, 1]),
                helion.Config(block_sizes=[16, 16, 32], num_threads=[16, 16, 4]),
            ):
                with self.subTest(shape=(m, k, n), config=config):
                    s = torch.zeros(m, device=DEVICE)
                    out, _, msl = _run(outer_load_written, (x, y, s_src, s), config)
                    _close(out, _mm(xs, y))
                    _close(s, 2 * s_src.cpu().double())
                    self.assertNotIn("mem_flags::mem_device", msl)

    @_full_threadgroup
    def test_scalar_seeds(self) -> None:
        """Seeds loaded or computed before the K loop, broadcast, read after
        it, shared by two loops, or another K loop's result."""
        x, y, c, cd, rt = self.x, self.y, self.c, self.cd, self.rt
        d = torch.randn(48, 40, device=DEVICE)
        bias = torch.randn(40, device=DEVICE)
        z = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        for config in (TILE16, TILE32):
            self._check(
                acc_from_load, (x, y, c), config, (None, rt + cd), layouts=(1, 0, 1)
            )
            self._check(
                seed_expr, (x, y, c, d), config, (None, rt + cd * d.cpu().double())
            )
            self._check(
                seed_bcast, (x, y, bias), config, (None, rt + bias.cpu().double())
            )
            self._check(seed_read_after, (x, y, c), config, (None, rt))
        self._check(chained, (x, y, z), TILE16_2K, (None, rt + _mm(x, z)))
        with self.subTest(kernel="shared_load_seed"):
            outs, _, _ = _run(shared_load_seed, (x, y, z, c), TILE16_2K)
            for out, ref in zip(outs, (cd + rt, cd + _mm(x, z), cd), strict=True):
                _close(out, ref)
        a = torch.randn(45, 72, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(72, 40, device=DEVICE, dtype=torch.bfloat16)
        cab = torch.randn(45, 40, device=DEVICE)
        self._check(
            free_rows_seeded,
            (a, b, cab, 2, 2),
            FREE32,
            (None, _mm(a, b) + cab.cpu().double()),
        )

    def test_size1_rows(self) -> None:
        """M = 1 rounds the descriptor up: materializations and seeds cover
        only the row inside the block."""
        x, y = self.x[:1].contiguous(), self.y
        c = self.c[:1].contiguous()
        bias = torch.randn(40, device=DEVICE)
        rt, cd = _mm(x, y), c.cpu().double()
        self._check(
            carry_vary,
            (x, y, c, bias),
            TILE32,
            (None, rt + 5 * cd + bias.cpu().double()),
        )
        self._check(acc_from_load, (x, y, c), TILE32, (None, rt + cd))
        self._check(stale_seed, (x, y), TILE32, (None, rt + 1))

    def test_seed_levels(self) -> None:
        """A Scalar seed outside a uniform if or an outer loop that holds the
        K loop, through a copy, and an outer level's input read after it."""
        x, y, c, cd, rt = self.x, self.y, self.c, self.cd, self.rt
        for config in (TILE16, TILE32):
            for f in (1, 0):
                flag = torch.tensor([f], device=DEVICE, dtype=torch.int32)
                self._check(
                    seed_outside_if,
                    (x, y, c, flag),
                    config,
                    (None, rt + cd if f else cd),
                )
            self._check(seed_outer_grid, (x, y, c), config, (None, 2 * rt + cd))
            self._check(seed_newvar, (x, y, c), config, (None, rt + 4 * cd))
            out = torch.zeros(48, 40, device=DEVICE)
            self._check(grid_old_read, (x, y, out), config, (out, rt))

    def test_uses_in_other_graphs(self) -> None:
        """A store from an inner loop over a block the op never sees, and a
        permuted store."""
        config = helion.Config(block_sizes=[16, 16, 16, 2], num_threads=[16, 16, 1, 2])
        self._check(
            inner_tb,
            (self.x, self.y, 4),
            config,
            (None, self.rt[None].expand(4, -1, -1)),
        )
        for config in (TILE16, TILE32):
            self._check(perm_store, (self.x, self.y), config, (None, self.rt.T))

    @_full_threadgroup
    def test_store_under_tensor_if(self) -> None:
        """A store of a materialized value under a per-element if."""
        ref = torch.where(self.cd > 0, self.rt + 5 * self.cd, 0.0)
        self._check(tif_mat, (self.x, self.y, self.c), TILE16, (None, ref))

    @_full_threadgroup
    def test_uses_at_reduced_values(self) -> None:
        """A store index or mask, or an epilogue load's index or mask, that a
        row reduction computes: an element loop at another thread's element
        would read this thread's reduction, so each use reads a
        materialization."""
        x, y, c, cd, rt = self.x, self.y, self.c, self.cd, self.rt
        perm = torch.randperm(48)
        sel = perm[:, None].expand(48, 40).contiguous().to(DEVICE)
        scattered = torch.zeros_like(rt)
        scattered[perm] = rt + 1
        for config, bn in ((TILE16, 16), (TILE32, 32)):
            keep = _tile_row_positive(cd, bn)
            for kernel, args, ref in (
                (store_at_row_max, (x, y, sel), scattered),
                (store_under_row_mask, (x, y, c), torch.where(keep, 2 * rt, 0.0)),
                (gather_at_row_max, (x, y, sel, c), rt + cd[perm]),
                (load_under_row_mask, (x, y, c), rt + torch.where(keep, cd, 0.0)),
            ):
                self._check(kernel, args, config, (None, ref), layouts=(0, 1, 0))

    @_full_threadgroup
    def test_surplus_threads(self) -> None:
        """A wider sibling's surplus threads have no element of the
        materialization: the read bounds N's local index by its block, 16,
        not the axis's 64."""
        z = torch.randn(16, 64, device=DEVICE)
        (out, out2), _, msl = _run(mat_sibling_rowmax, (self.x, self.y, z), SIBLING64)
        _close(out, self.rt + _tile_row_max(-(self.rt * self.rt) - 1.0, 16))
        _close(out2, z.cpu().double() * 2)
        self.assertTrue(_tile_copies_ordered(msl), msl)
        self.assertIn("required_threads_per_threadgroup(16, 64, 1)", msl)
        (read,) = [line for line in msl.splitlines() if "auto _mpp_v" in line]
        self.assertEqual(len(re.findall(r"- offset_\d+\) < 16\)", read)), 2)


@_requires_mps
class TestMppConversionRejects(_MppCase):
    """Forms the conversions don't make correct still raise
    ``BackendUnsupported``."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        self.y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        self.c = torch.randn(48, 40, device=DEVICE)

    def test_divergent_control_flow(self) -> None:
        """A cooperative op needs every thread, so it may not sit under a
        per-element if or while."""
        msg = "under an if or a while on a per-element predicate"
        for kernel in (tif_k, tif_all, while_tensor):
            self._rejects(kernel, (self.x, self.y, self.c), TILE16, msg)

    def test_levels(self) -> None:
        """A level that doesn't give the updated accumulator back: the else
        branch rewrites it or holds the K loop, or the outer loop rebinds it
        after the K loop."""
        flag = torch.ones(1, device=DEVICE, dtype=torch.int32)
        for kernel in (else_rewrites, else_holds_k):
            self._rejects(
                kernel,
                (self.x, self.y, self.c, flag),
                TILE16,
                "doesn't give it back as the same variable",
            )
        self._rejects(
            outer_rebind,
            (self.x, self.y),
            TILE16,
            "gives back another value than the updated one",
        )

    def test_swapped_tiles(self) -> None:
        """A materialized value or a seed that meets M and N swapped, or a
        seed computed from a tile that does."""
        y = torch.randn(72, 48, device=DEVICE, dtype=torch.float16)
        c = torch.randn(48, 48, device=DEVICE)
        self._rejects(
            seed_relabel,
            (self.x, y, c, c),
            TILE16,
            "a staged MPP operand is written into its tile by block, so it would "
            "hold another element",
        )
        self._rejects(
            swapped_mat_store,
            (self.x, y, c),
            TILE16,
            "a value read from an MPP result in the Scalar layout would hold "
            "another element",
        )
        self._rejects(
            seed_swapped,
            (self.x, y, c),
            TILE16,
            "an MPP accumulator seed whose dims are not M and N",
        )

    def test_acc_loaded_in_loop(self) -> None:
        """An acc read from a tensor in the K loop is no value of the carry,
        though its tile reads the carry's size."""
        self._rejects(
            acc_load_in_loop,
            (self.x, self.y, self.c),
            TILE16,
            "a matmul whose acc is not its K loop's carry",
        )

    def test_shared_seed(self) -> None:
        """One constant fill seeding two K loops; each loop with its own fill
        lowers."""
        z = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        msg = "two MPP K loops whose accumulators start from one value"
        for kernel in (shared_zero_seed, shared_full_seed):
            self._rejects(kernel, (self.x, self.y, z), TILE16_2K, msg)
        outs, _, _ = _run(two_zero_seeds, (self.x, self.y, z), TILE16_2K)
        for out, ref in zip(outs, (_mm(self.x, self.y), _mm(self.x, z)), strict=True):
            _close(out, ref)

    def test_fence_placement(self) -> None:
        """A written tensor's load that a staged operand is computed from in a
        loop nested in the K loop, where no barrier reaches every thread."""
        x = torch.randn(48, 32, device=DEVICE, dtype=torch.float16)
        y = torch.randn(32, 32, device=DEVICE, dtype=torch.float16)
        config = helion.Config(block_sizes=[16, 32, 16], num_threads=[16, 32, 1])
        self._rejects(
            loop_operand_written,
            (x, y, torch.empty_like(x)),
            config,
            "a load of a tensor the kernel writes, inside control flow in its K "
            "loop's body",
        )

    @_full_threadgroup
    def test_backstop(self) -> None:
        """An element loop that would read a thread's own element of a
        materialization raises, here with each planner rule that prevents it
        patched out: the input half of the pruning fixpoint, and the rule that
        a value the op updates in place is never recomputable."""
        y = torch.randn(72, 48, device=DEVICE, dtype=torch.float16)
        msg = "this thread's own element of a materialized MPP result"
        s = torch.zeros(48, device=DEVICE)
        with patch.object(mpp_matmul, "_inputs_kept", return_value=True):
            self._rejects(mixed_prune, (self.x, y, s), ROWS[0], msg)
        with patch.object(
            mpp_matmul._Recomputable, "_updated_in_place", return_value=False
        ):
            self._rejects(stale_seed, (self.x, self.y), TILE16, msg)

    def test_materialized_names(self) -> None:
        """A read of a materialization tile and what is computed from it per
        thread are materialized; a collective's target and a read of another
        buffer are not."""
        body = ast.parse(
            "v = _mpp_mat_0[i]\n"
            "w = v * 2.0\n"
            "g = helion_red.group_max(site, w, _red_scratch_0, tid)\n"
            "h = g + 1.0\n"
            "u = _red_scratch_0[i]\n"
        ).body
        buffers = frozenset({"_mpp_mat_0", "_red_scratch_0"})
        self.assertEqual(mpp_matmul._materialized(body, buffers), {"v", "w"})


# --- Root K loops: every torch.addmm, and gemm, takes the planner ----------

THREE = (
    None,
    helion.Config(block_sizes=[16, 16, 16]),
    helion.Config(block_sizes=[32, 32, 32]),
)

matmul_metal = helion.kernel(matmul.fn, backend="metal", static_shapes=True)


@helion.kernel(**S)
def root_plain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_scatter(x: torch.Tensor, y: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[rows[tm], tn] = acc
    return out


@helion.kernel(**S)
def root_gathered(x: torch.Tensor, y: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    m = rows.size(0)
    k = x.size(1)
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[rows[tm], tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_m_short(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m - 8, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_m_begin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([8, 0], [m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_k_short(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k - 5):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_acc_from_load(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = c[tm, tn] + 0.0
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_const_seed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.full([tm, tn], 1.5, dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_chained(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two K loops in sequence on one acc."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_k_masked(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The lhs is masked on K past ``k - 7``."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = hl.load(x, [tm, tk], extra_mask=(tk.index < k - 7)[None, :])
            acc = torch.addmm(acc, a, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_siblings(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        acc2 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc2 = torch.addmm(acc2, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc + acc2
    return out


@helion.kernel(**S)
def root_k_count(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A second carry of the K loop counts its tiles."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        cnt = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            cnt = cnt + 1.0
        out[tm, tn] = acc + cnt
    return out


@helion.kernel(**S)
def dot_then_addmm(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """An hl.dot K loop beside an addmm K loop."""
    m, k = x.size()
    _, n = y.size()
    o1 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    o2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        a1 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a1 = hl.dot(x[tm, tk], y[tk, tn], acc=a1)
        o1[tm, tn] = a1
        a2 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a2 = torch.addmm(a2, x[tm, tk], y[tk, tn])
        o2[tm, tn] = a2 * 2.0
    return o1, o2


@helion.kernel(**S)
def dot_root_then_addmm_root(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = y.size()
    o1 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    o2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        a1 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a1 = hl.dot(x[tm, tk], y[tk, tn], acc=a1)
        o1[tm, tn] = a1
    for tm, tn in hl.tile([m, n]):
        a2 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a2 = torch.addmm(a2, x[tm, tk], y[tk, tn])
        o2[tm, tn] = a2 * 2.0
    return o1, o2


@helion.kernel(**S)
def root_store_swapped(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([n, m], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tn, tm] = acc
    return out


@helion.kernel(**S)
def root_other_loop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The matmul's K is a ``:`` slice inside a loop over another block."""
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for _tj in hl.tile(12):
            acc = torch.addmm(acc, x[tm, :], y[:, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_shared_seed(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = y.size()
    o1 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    o2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        z = hl.zeros([tm, tn], dtype=torch.float32)
        a1 = z
        for tk in hl.tile(k):
            a1 = torch.addmm(a1, x[tm, tk], y[tk, tn])
        a2 = z
        for tk in hl.tile(k):
            a2 = torch.addmm(a2, x[tm, tk], y[tk, tn])
        o1[tm, tn] = a1
        o2[tm, tn] = a2 * 2.0
    return o1, o2


@helion.kernel(**S)
def root_mm_overwrite(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Each K tile's product replaces the last one."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.mm(x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_mm_plus(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc + torch.mm(x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def root_acc_like_out(
    x: torch.Tensor, y: torch.Tensor, o: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=o.dtype)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        o[tm, tn] = acc
    return o


@helion.kernel(**S)
def root_scaled(x: torch.Tensor, y: torch.Tensor, s: float) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc * s
    return out


@helion.kernel(**S)
def root_beside_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A K loop beside a row-sum loop over its own block."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        s = hl.zeros([tm, 1], dtype=torch.float32)
        for tj in hl.tile(k):
            s = s + (x[tm, tj] * 1.0).sum(-1, keepdim=True)
        out[tm, tn] = acc + s
    return out


def _bk(
    kernel: helion.Kernel, args: tuple[object, ...], config: helion.Config | None
) -> int:
    """The K block size ``kernel`` runs at ``config`` (None: its default)."""
    config = config or kernel.bind(args).config_spec.default_config()
    return config.config["block_sizes"][2]


@_requires_mps
class TestMppRootAddmm(unittest.TestCase):
    """A root K loop's ``torch.addmm`` on the planner, where each property
    of the loop is checked rather than assumed."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        self.y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        configs: tuple[helion.Config | None, ...],
        ref: object,
    ) -> None:
        """Run ``kernel`` at each config through the planner against ``ref``,
        a tensor, a tuple of them, or a function of the config's K block."""
        for config in configs:
            with self.subTest(kernel=kernel.name, config=config):
                out, code, _ = _run(kernel, args, config)
                self.assertIn("_helion_mpp_acc(", code)
                want = ref(_bk(kernel, args, config)) if callable(ref) else ref
                outs = out if isinstance(out, tuple) else (out,)
                wants = want if isinstance(want, tuple) else (want,)
                for o, w in zip(outs, wants, strict=True):
                    _close(o, w)

    def test_gemm(self) -> None:
        """``examples/matmul.py`` with a column-major rhs, as the benchmark
        runs it, with tails on every dim (96x72 @ 72x80)."""
        x = torch.randn(96, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(80, 72, device=DEVICE, dtype=torch.float16).t()
        configs = (
            None,
            helion.Config(block_sizes=[32, 64, 128], num_threads=[1, 0, 1]),
            helion.Config(block_sizes=[32, 32, 32]),
        )
        self._check(matmul_metal, (x, y), configs, _mm(x, y))

    def test_indexed_stores_and_operands(self) -> None:
        """A scattered store and a gathered lhs re-run their index."""
        x, y = self.x, self.y
        mm = _mm(x, y)
        rows = torch.randperm(48, device=DEVICE)
        scattered = torch.zeros_like(mm)
        scattered[rows.cpu()] = mm
        configs = (
            None,
            helion.Config(
                block_sizes=[16, 16, 16], num_threads=[1, 0, 0], num_warps=32
            ),
        )
        self._check(root_scatter, (x, y, rows), configs, scattered)
        x = torch.randn(32, 72, device=DEVICE, dtype=torch.float16)
        gather = torch.randint(0, 32, (32,), device=DEVICE)
        self._check(root_gathered, (x, y, gather), (None,), _mm(x[gather], y))

    def test_loop_bounds(self) -> None:
        """An M loop that starts late or ends early, and a K mask, bound what
        is computed and stored."""
        x, y = self.x, self.y
        zeros = torch.zeros(8, 40, dtype=torch.float64)
        self._check(root_m_short, (x, y), THREE, torch.cat([_mm(x[:40], y), zeros]))
        self._check(root_m_begin, (x, y), THREE, torch.cat([zeros, _mm(x[8:], y)]))
        self._check(root_k_masked, (x, y), THREE[:2], _mm(x[:, :65], y[:65]))

    def test_seeds(self) -> None:
        """A seed loaded, filled with a nonzero constant, or left by an
        earlier K loop."""
        x, y = self.x, self.y
        mm = _mm(x, y)
        c = torch.randn(48, 40, device=DEVICE)
        self._check(root_acc_from_load, (x, y, c), THREE, c.cpu().double() + mm)
        self._check(root_const_seed, (x, y), THREE, mm + 1.5)
        self._check(root_chained, (x, y), (None, TILE16_2K), 2 * mm)

    def test_carries(self) -> None:
        """A second carry of the K loop, and sibling K loops."""
        x, y = self.x, self.y
        mm = _mm(x, y)
        self._check(root_k_count, (x, y), THREE[:2], lambda bk: mm + math.ceil(72 / bk))
        self._check(root_siblings, (x, y), (None, TILE16_2K), 2 * mm)

    def test_size1_dims(self) -> None:
        """A size-1 M or N rounds the descriptor up, at the root too."""
        x = torch.randn(1, 64, device=DEVICE)
        y = torch.randn(64, 48, device=DEVICE)
        self._check(root_plain, (x, y), (None,), _mm(x, y))
        x = torch.randn(48, 64, device=DEVICE)
        y = torch.randn(64, 1, device=DEVICE)
        self._check(root_plain, (x, y), (None,), _mm(x, y))

    def test_dot_beside_addmm(self) -> None:
        """An ``hl.dot`` K loop beside an ``addmm`` K loop, both on the
        planner, in one root and in two roots."""
        x = self.x.float()
        y = self.y.float()
        mm = _mm(x, y)
        for kernel, config in (
            (dot_then_addmm, None),
            (dot_then_addmm, helion.Config(block_sizes=[32, 32, 16, 32], num_warps=8)),
            (
                dot_root_then_addmm_root,
                helion.Config(block_sizes=[32, 32, 32, 16, 16, 16], num_warps=4),
            ),
        ):
            with self.subTest(kernel=kernel.name, config=config):
                (o1, o2), code, _ = _run(kernel, (x, y), config)
                self.assertEqual(code.count("_helion_mpp_acc("), 2)
                _close(o1, mm)
                _close(o2, 2 * mm)


@_requires_mps
class TestMppRootAddmmRejects(_MppCase):
    """Root forms the planner can't lower raise ``BackendUnsupported``, and
    only at the configs that need it."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        self.y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)

    def test_forms(self) -> None:
        x, y = self.x, self.y
        xf, yf = x.float(), y.float()
        self._rejects(
            root_store_swapped,
            (xf, torch.randn(72, 48, device=DEVICE)),
            None,
            "a store (store) meets dimension 0",
        )
        self._rejects(
            root_other_loop,
            (torch.randn(32, 24, device=DEVICE), torch.randn(24, 32, device=DEVICE)),
            None,
            "a matmul over a ':' K whose acc an enclosing loop or if passes in",
        )
        self._rejects(
            root_shared_seed,
            (xf, yf),
            None,
            "two MPP K loops whose accumulators start from one value",
        )
        for kernel in (root_mm_overwrite, root_mm_plus):
            self._rejects(
                kernel,
                (xf, yf),
                None,
                "a matmul with no acc over a tiled K, whose result is each K "
                "tile's product alone",
            )

    def test_acc_dtype(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            args = (
                self.x.to(dtype),
                self.y.to(dtype),
                torch.empty(48, 40, device=DEVICE, dtype=dtype),
            )
            for config in (None, helion.Config(block_sizes=[16, 16, 16])):
                self._rejects(
                    root_acc_like_out,
                    args,
                    config,
                    "a matmul whose acc is not float32",
                )

    def test_k_tile(self) -> None:
        """A K tile over 256 is rejected; a K tile of 256 lowers."""
        x = torch.randn(64, 1040, device=DEVICE, dtype=torch.float16)
        y = torch.randn(1040, 64, device=DEVICE, dtype=torch.float16)
        self._rejects(
            root_plain,
            (x, y),
            helion.Config(block_sizes=[32, 32, 512], num_warps=1),
            "and K <= 256",
        )
        out, _, _ = _run(root_plain, (x, y), helion.Config(block_sizes=[32, 64, 256]))
        _close(out, _mm(x, y))

    def test_launch(self) -> None:
        """A launch of less than a SIMD group is rejected; the same kernels
        at their default configs lower."""
        x, y = self.x.float(), self.y.float()
        msg = "needs whole 32-thread SIMD groups"
        self._rejects(
            root_plain,
            (x, y),
            helion.Config(
                block_sizes=[32, 32, 16], num_threads=[1, 1, 0], num_warps=32
            ),
            msg,
        )
        self._rejects(
            root_beside_sum,
            (x, y),
            helion.Config(
                block_sizes=[16, 16, 16, 32], num_threads=[0, 1, 1, 1], num_warps=32
            ),
            msg,
        )
        out, _, _ = _run(root_beside_sum, (x, y), None)
        _close(out, _mm(x, y) + x.cpu().double().sum(-1, keepdim=True))


# --- General device views -------------------------------------------------


@helion.kernel(**S)
def nd_begin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    _, _, n = y.shape
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm, tn in hl.tile([b, m, n], block_size=[1, None, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tb.begin, tm, tk], y[tb.begin, tk, tn])
        out[tb.begin, tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_gqa(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    h, m, k = x.shape
    g, _, n = y.shape
    out = torch.empty([h, m, n], dtype=torch.float32, device=x.device)
    for th, tm, tn in hl.tile([h, m, n], block_size=[1, None, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[th.begin, tm, tk], y[th.begin // (h // g), tk, tn])
        out[th.begin, tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_id(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    _, _, n = y.shape
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm, tn in hl.tile([b, m, n], block_size=[1, None, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tb.id, tm, tk], y[tb.id, tk, tn])
        out[tb.id, tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_grid(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    _, _, n = y.shape
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        for bi in hl.grid(b):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[bi, tm, tk], y[bi, tk, tn])
            out[bi, tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_literal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _, m, k = x.shape
    _, _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[2, tm, tk], y[1, tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_host(x: torch.Tensor, y: torch.Tensor, i: int) -> torch.Tensor:
    _, m, k = x.shape
    _, _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[i, tm, tk], y[i, tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_end(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    _, _, n = y.shape
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm, tn in hl.tile([b, m, n], block_size=[1, None, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tb.end - 1, tm, tk], y[tb.begin, tk, tn])
        out[tb.begin, tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_negative(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _, m, k = x.shape
    _, _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[-1, tm, tk], y[-2, tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def nt_negative(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n, _ = p.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], p[-3, tn, tk].T)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def colon_negative(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _, m, _ = x.shape
    _, _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = torch.addmm(acc, x[-1, tm, :], y[0, :, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_mid(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x is [M, B, K]: the scalar index sits between the op's dims."""
    m, b, k = x.shape
    _, _, n = y.shape
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm, tn in hl.tile([b, m, n], block_size=[1, None, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tb.begin, tk], y[tb.begin, tk, tn])
        out[tb.begin, tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_data_index(x: torch.Tensor, y: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """The outer index is a 0-d integer load."""
    _, m, k = x.shape
    _, _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        s = sel[0]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[s, tm, tk], y[s, tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def outer_reassigned(
    x: torch.Tensor, y: torch.Tensor, s0: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    """An ``if`` writes ``s + 1`` back into ``s`` between the two loads: the
    lhs reads ``s`` and the rhs ``s + 1``."""
    _, m, k = x.shape
    _, _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        s = s0[0]
        f = flag[0]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = x[s, tm, tk]
            if f > 0:
                s = s + 1
            acc = torch.addmm(acc, a, y[s, tk, tn])
            if f > 0:
                s = s - 1
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def chunked(x: torch.Tensor, y: torch.Tensor, chunk: int) -> torch.Tensor:
    """``out[c] = x[:, chunk c] @ y[chunk c, :]``, K at ``base + tk.index``."""
    chunk = hl.specialize(chunk)
    m, s = x.shape
    _, n = y.shape
    nc = (s + chunk - 1) // chunk
    out = torch.empty([nc, m, n], dtype=torch.float32, device=x.device)
    for tc, tm, tn in hl.tile([nc, m, n], block_size=[1, None, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(chunk):
            acc = torch.addmm(
                acc,
                x[tm, tk.index + tc.begin * chunk],
                y[tc.begin * chunk + tk.index, tn],
            )
        out[tc.begin, tm, tn] = acc
    return out


@helion.kernel(**S)
def m_base(
    x: torch.Tensor, y: torch.Tensor, r: torch.Tensor, chunk: int
) -> torch.Tensor:
    """M at ``base + tm.index`` in the lhs, the store and an epilogue load."""
    chunk = hl.specialize(chunk)
    s, k = x.shape
    _, n = y.shape
    out = torch.zeros([s, n], dtype=torch.float32, device=x.device)
    for tc, tm, tn in hl.tile([s // chunk, chunk, n], block_size=[1, None, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tc.begin * chunk + tm.index, tk], y[tk, tn])
        rows = tc.begin * chunk + tm.index
        out[rows, tn] = acc + r[rows, tn]
    return out


@helion.kernel(**S)
def base_reassigned(
    x: torch.Tensor, y: torch.Tensor, b0: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    """An ``if`` advances the base of ``base + tk.index`` between the two
    loads."""
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        base = b0[0]
        f = flag[0]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(16):
            a = x[tm, tk.index + base]
            if f > 0:
                base = base + 16
            acc = torch.addmm(acc, a, y[tk.index + base, tn])
            if f > 0:
                base = base - 16
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def k_short(x: torch.Tensor, y: torch.Tensor, kk: int) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(kk):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def k_window(x: torch.Tensor, y: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(lo, hi):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def causal_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """K runs to the end of the M tile."""
    m, _ = x.shape
    _, n = y.shape
    bm = hl.register_block_size(m)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n], block_size=[bm, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile((tm.id + 1) * bm):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def nt_kloop(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n, _ = p.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], p[tn, tk].T)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def tn_kloop(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    k, m = q.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, q[tk, tm].T, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_nt(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """Per-head ``q @ k.T`` with grouped heads."""
    h, m, d = q.shape
    g, n, _ = kv.shape
    out = torch.empty([h, m, n], dtype=torch.float32, device=q.device)
    for th, tm, tn in hl.tile([h, m, n], block_size=[1, None, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for td in hl.tile(d):
            acc = torch.addmm(
                acc, q[th.begin, tm, td], kv[th.begin // (h // g), tn, td].T
            )
        out[th.begin, tm, tn] = acc
    return out


@helion.kernel(**S)
def perm_outer_first(x3: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x3 is [K, B, M]: the op's dims are tensor dims 2 and 0, read
    transposed, with the scalar index between them."""
    k, _, m = x3.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x3.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x3[tk, 1, tm].T, y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def free_rows_nt(a: torch.Tensor, p: torch.Tensor, nbm: int, nbn: int) -> torch.Tensor:
    """The rhs is a transposed load whose ``hl.arange`` dim carries a mask."""
    m, k = a.shape
    n, _ = p.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 64)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None])
                p_blk = hl.load(p, [cols, tk.index], extra_mask=cv[:, None])
                acc = torch.addmm(acc, a_blk, p_blk.T)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def colon_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = torch.addmm(acc, x[tm, :], y[:, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def colon_k_t(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    n, _ = p.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = torch.addmm(acc, x[tm, :], p[tn, :].T)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def colon_tn(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _, m = q.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = torch.addmm(acc, q[:, tm].T, y[:, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def colon_k_grid(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, _ = x.shape
    _, _, n = y.shape
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        for bi in hl.grid(b):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            acc = torch.addmm(acc, x[bi, tm, :], y[bi, :, tn])
            out[bi, tm, tn] = acc
    return out


@helion.kernel(**S)
def colon_two(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        a1 = hl.zeros([tm, tn], dtype=torch.float32)
        a1 = torch.addmm(a1, x[tm, :], y[:, tn])
        a2 = hl.zeros([tm, tn], dtype=torch.float32)
        a2 = torch.addmm(a2, x[tm, :], w[:, tn])
        out[tm, tn] = a1
        out2[tm, tn] = a2
    return out, out2


@helion.kernel(**S)
def nested_colon(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            acc = torch.addmm(acc, x[tm, :], y[:, tn])
            out[tm, tn] = acc * 2.0
    return out


@helion.kernel(**S)
def chunk_scan_prev_state(c: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
    """mamba2_chunk_scan's prev-state product written with ``torch.addmm``:
    a ``:`` K over an N-D c at ``base + tm.index`` with a grouped head, and
    a transposed 5-D ``ps``, stored at ``base + tm.index``."""
    batch, seqlen, ngroups, _ = c.shape
    _, nchunks, nheads, headdim, _ = ps.shape
    chunk = seqlen // nchunks
    block_m = hl.register_block_size(chunk)
    block_n = hl.register_block_size(headdim)
    out = torch.empty(
        batch, seqlen, nheads, headdim, dtype=torch.float32, device=c.device
    )
    for th, tm, tn, tb, tc in hl.tile(
        [nheads, chunk, headdim, batch, nchunks],
        block_size=[1, block_m, block_n, 1, 1],
    ):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        c_local = c[
            tb.begin, tm.index + tc.begin * chunk, th.begin // (nheads // ngroups), :
        ]
        ps_local = ps[tb.begin, tc.begin, th.begin, tn, :]
        acc = torch.addmm(acc, c_local, ps_local.T)
        out[tb.begin, tc.begin * chunk + tm.index, th.begin, tn] = acc * 0.5
    return out


def _chunk_scan_prev_state_ref(c: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
    batch, seqlen, ngroups, dstate = c.shape
    _, nchunks, nheads, headdim, _ = ps.shape
    chunk = seqlen // nchunks
    cc = c.cpu().double().reshape(batch, nchunks, chunk, ngroups, dstate)
    cc = cc.repeat_interleave(nheads // ngroups, dim=3)
    out = torch.einsum("bclhn,bchpn->bclhp", cc, ps.cpu().double()) * 0.5
    return out.reshape(batch, seqlen, nheads, headdim)


@helion.kernel(**S)
def colon_k_bias(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = torch.addmm(c[tm, tn], x[tm, :], y[:, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def colon_k_staged(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = torch.addmm(acc, x[tm, :] * 2.0, y[:, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def colon_k_carried(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for _i in hl.grid(3):
            acc = torch.addmm(acc, x[tm, :], y[:, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def colon_m(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tn in hl.tile(n):
        acc = hl.zeros([m, tn], dtype=torch.float32)
        acc = torch.addmm(acc, x[:, :], y[:, tn])
        out[:, tn] = acc
    return out


@helion.kernel(**S)
def colon_m_kloop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tn in hl.tile(n):
        acc = hl.zeros([m, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[:, tk], y[tk, tn])
        out[:, tn] = acc
    return out


@helion.kernel(**S)
def colon_m_seeded(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tn in hl.tile(n):
        acc = c[:, tn].to(torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[:, tk], y[tk, tn])
        out[:, tn] = acc
    return out


@helion.kernel(**S)
def colon_n_kloop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, :])
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def store_base_swapped(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.zeros([n + 8, m], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tn.index + 8, tm] = acc
    return out


@helion.kernel(**S)
def base_masked(x: torch.Tensor, y: torch.Tensor, chunk: int, lim: int) -> torch.Tensor:
    """A mask on a ``base + tk.index`` dim."""
    chunk = hl.specialize(chunk)
    m, s = x.shape
    _, n = y.shape
    out = torch.empty([s // chunk, m, n], dtype=torch.float32, device=x.device)
    for tc, tm, tn in hl.tile([s // chunk, m, n], block_size=[1, None, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(chunk):
            kk = tc.begin * chunk + tk.index
            a = hl.load(x, [tm, kk], extra_mask=(kk < lim)[None, :])
            acc = torch.addmm(acc, a, y[kk, tn])
        out[tc.begin, tm, tn] = acc
    return out


@helion.kernel(**S)
def alpha_tile_base(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The lhs columns are ``torch.add(tk.index, 3, alpha=2)``, tk.index + 6."""
    m, _ = x.shape
    k, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, torch.add(tk.index, 3, alpha=2)], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def identity_permute(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk].permute(0, 1), y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def nd_written(x: torch.Tensor, y: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    _, _, n = y.shape
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tb, tm, tn in hl.tile([b, m, n], block_size=[1, None, None]):
            for tj in hl.tile(k):
                x2[tb.begin, tm, tj] = x[tb.begin, tm, tj] * 2.0
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x2[tb.begin, tm, tk], y[tb.begin, tk, tn])
            out[tb.begin, tm, tn] = acc
    return out


B16 = helion.Config(block_sizes=[16, 16, 16])
LANES = helion.Config(block_sizes=[32, 32, 32], num_threads=[8, 8, 0])


def _device_views(msl: str) -> int:
    return len(re.findall(r"tensor<device \w+, dextents", msl))


def _transposes(msl: str) -> tuple[bool, bool]:
    """The descriptor's ``(transpose_left, transpose_right)``."""
    (flags,) = re.findall(
        r"matmul2d_descriptor\(\d+, \d+, \d+, (true|false), (true|false)", msl
    )
    return flags[0] == "true", flags[1] == "true"


def _run_depths(msl: str) -> list[int]:
    """The number of MSL ``for`` loops around each ``.run(``."""
    stack: list[bool] = []
    depths: list[int] = []
    for line in (line.strip() for line in msl.splitlines()):
        if line.startswith("}"):
            stack.pop()
        if ".run(" in line:
            depths.append(sum(stack))
        if line.endswith("{"):
            stack.append(line.startswith("for ("))
    return depths


def _nan_past(
    data: torch.Tensor, cols: int | None = None, rows: int | None = None
) -> torch.Tensor:
    """``data`` with every column from ``cols`` and every row from ``rows``
    on set to NaN, so a matmul that reads past them shows."""
    out = data.clone()
    if cols is not None:
        out[:, cols:] = float("nan")
    if rows is not None:
        out[rows:] = float("nan")
    return out


@_requires_mps
class TestMppViews(unittest.TestCase):
    """Operands read straight from device memory: N-D tensors whose other
    dims take uniform indices, ``base + tile.index`` dims, transposed loads
    and ``:`` dims, with a K extent at the K loop's end.  A ``:`` K is one op
    over the whole K."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        ref: torch.Tensor | tuple[torch.Tensor, ...],
        views: int = 2,
    ) -> str:
        out, _, msl = _run(kernel, args, config)
        if isinstance(ref, tuple):
            for o, r in zip(out, ref, strict=True):
                _close(o, r)
        else:
            _close(out, ref)
        self.assertEqual(_device_views(msl), views)
        self.assertNotIn("_mpp_tile", msl)
        return msl

    def test_nd(self) -> None:
        """Every outer index is folded into the base and range-tested."""
        x3 = torch.randn(3, 40, 72, device=DEVICE, dtype=torch.float16)
        y3 = torch.randn(3, 72, 24, device=DEVICE, dtype=torch.float16)
        yg = torch.randn(2, 72, 24, device=DEVICE, dtype=torch.float16)
        xg = torch.randn(4, 40, 72, device=DEVICE, dtype=torch.float16)
        xm = torch.randn(40, 3, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 24, device=DEVICE, dtype=torch.float16)
        sel = torch.tensor([1], device=DEVICE, dtype=torch.int32)
        bmm = _mm(x3, y3)
        cases = (
            (nd_begin, (x3, y3), bmm),
            (nd_id, (x3, y3), bmm),
            (nd_grid, (x3, y3), bmm),
            (
                nd_gqa,
                (xg, yg),
                _mm(xg, yg.repeat_interleave(2, 0)),
            ),
            (
                nd_mid,
                (xm, y3),
                _mm(xm.permute(1, 0, 2), y3),
            ),
            (nd_data_index, (x3, y3, sel), _mm(x3[1], y3[1])),
            (nd_literal, (x3, y3), _mm(x3[2], y3[1])),
            (nd_host, (x3, y3, 1), _mm(x3[1], y3[1])),
            (nd_lhs, (x3, y), _mm(x3[2], y)),
        )
        for kernel, args, ref in cases:
            # nd_lhs's M/N tile loop is a device loop: one thread by default.
            for config in (TILE16 if kernel is nd_lhs else B16, LANES):
                with self.subTest(kernel=kernel.name, config=config):
                    msl = self._check(kernel, args, config, ref)
                    self.assertRegex(msl, r"_mpp_a_x0 \* \d+\)")
                    self.assertRegex(msl, r"if \(0 <= _mpp_a_x0 && _mpp_a_x0 < ")

    def test_outer_reassigned(self) -> None:
        """Each operand reads its outer index where its own load did."""
        x = torch.randn(4, 40, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 72, 24, device=DEVICE, dtype=torch.float16)
        s0 = torch.tensor([1], device=DEVICE, dtype=torch.int64)
        flag = torch.tensor([1], device=DEVICE, dtype=torch.int32)
        for config in (B16, helion.Config(block_sizes=[32, 32, 16])):
            with self.subTest(config=config):
                self._check(outer_reassigned, (x, y, s0, flag), config, _mm(x[1], y[2]))

    def test_negative_literal(self) -> None:
        """A negative literal outer index counts from the end of its dim."""
        x3 = torch.randn(3, 40, 72, device=DEVICE, dtype=torch.float16)
        y3 = torch.randn(3, 72, 24, device=DEVICE, dtype=torch.float16)
        p3 = torch.randn(3, 24, 72, device=DEVICE, dtype=torch.float16)
        cases = (
            (nd_negative, (x3, y3), _mm(x3[-1], y3[-2]), (B16, LANES)),
            (nt_negative, (x3[0], p3), _mm(x3[0], p3[0].T), (B16, LANES)),
            (
                colon_negative,
                (x3, y3),
                _mm(x3[-1], y3[0]),
                (None, helion.Config(block_sizes=[16, 16])),
            ),
        )
        for kernel, args, ref, configs in cases:
            for config in configs:
                with self.subTest(kernel=kernel.name, config=config):
                    self._check(kernel, args, config, ref)

    def test_tile_base(self) -> None:
        """``base + tile.index``: a K at it ends at ``base + end``; an M at
        it reaches the store and an epilogue load through one element loop."""
        configs = (
            B16,
            helion.Config(block_sizes=[32, 32, 32]),
            helion.Config(block_sizes=[16, 16, 32]),
        )
        for chunk, s in ((16, 96), (24, 96), (40, 120)):
            x = torch.randn(40, s, device=DEVICE, dtype=torch.float16)
            y = torch.randn(s, 24, device=DEVICE, dtype=torch.float16)
            ref = torch.stack(
                [_mm(x[:, c : c + chunk], y[c : c + chunk]) for c in range(0, s, chunk)]
            )
            for config in configs:
                with self.subTest(kernel="chunked", chunk=chunk, config=config):
                    self._check(chunked, (x, y, chunk), config, ref)
        for s, chunk in ((64, 32), (40, 20)):
            x = torch.randn(s, 72, device=DEVICE, dtype=torch.float16)
            y = torch.randn(72, 24, device=DEVICE, dtype=torch.float16)
            r = torch.randn(s, 24, device=DEVICE)
            for config in (
                B16,
                helion.Config(block_sizes=[32, 16, 32]),
                helion.Config(block_sizes=[16, 16, 16], num_threads=[4, 16, 0]),
            ):
                with self.subTest(kernel="m_base", s=s, config=config):
                    msl = self._check(
                        m_base, (x, y, r, chunk), config, _mm(x, y) + r.cpu().double()
                    )
                    self.assertEqual(_layouts(msl), (1, 0, 0))
                    self.assertIn("_mpp_a_o0 < _mpp_a_e0", msl)
        x = torch.randn(40, 96, device=DEVICE, dtype=torch.float16)
        y = torch.randn(96, 24, device=DEVICE, dtype=torch.float16)
        b0 = torch.tensor([16], device=DEVICE, dtype=torch.int64)
        flag = torch.tensor([1], device=DEVICE, dtype=torch.int32)
        for config in (B16, helion.Config(block_sizes=[32, 16, 16])):
            with self.subTest(kernel="base_reassigned", config=config):
                self._check(
                    base_reassigned,
                    (x, y, b0, flag),
                    config,
                    _mm(x[:, 16:32], y[32:48]),
                )

    def test_k_extent(self) -> None:
        """A K loop that ends before the tensor's K bounds the views' K
        extent: the columns past it are NaN and never read."""
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        for kk in (67, 9):
            args = (_nan_past(x, cols=kk), _nan_past(y, rows=kk), kk)
            for config in (B16, helion.Config(block_sizes=[32, 32, 16]), LANES):
                with self.subTest(kernel="k_short", kk=kk, config=config):
                    msl = self._check(k_short, args, config, _mm(x[:, :kk], y[:kk]))
                    self.assertIn("metal::min((long)(end_", msl)
        window = _nan_past(x, cols=61)
        window[:, :5] = float("nan")
        for config in (B16, helion.Config(block_sizes=[32, 32, 16])):
            with self.subTest(kernel="k_window", config=config):
                self._check(
                    k_window,
                    (window, _nan_past(y, rows=61), 5, 61),
                    config,
                    _mm(x[:, 5:61], y[5:61]),
                )
        xq = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        yq = torch.randn(64, 40, device=DEVICE, dtype=torch.float16)
        for bm in (16, 32):
            causal = xq.clone()
            ref = torch.empty(64, 40, dtype=torch.float64)
            for i in range(0, 64, bm):
                causal[i : i + bm, i + bm :] = float("nan")
                ref[i : i + bm] = _mm(xq[i : i + bm, : i + bm], yq[: i + bm])
            config = helion.Config(block_sizes=[bm, 16, 16])
            with self.subTest(kernel="causal_k", config=config):
                self._check(causal_k, (causal, yq), config, ref)
        with self.subTest(kernel="root_k_short"):
            self._check(
                root_k_short,
                (_nan_past(x, cols=67), _nan_past(y, rows=67)),
                None,
                _mm(x[:, :67], y[:67]),
            )

    def test_transposed(self) -> None:
        """A transposed load is a view with its dims swapped: the
        descriptor transposes the operand whose op dims are not unit-stride
        last."""
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        p = torch.randn(40, 72, device=DEVICE, dtype=torch.float16)
        q = torch.randn(72, 48, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        qh = torch.randn(4, 40, 72, device=DEVICE, dtype=torch.float16)
        kv = torch.randn(2, 24, 72, device=DEVICE, dtype=torch.float16)
        x3 = torch.randn(72, 3, 40, device=DEVICE, dtype=torch.float16)
        cases = (
            (nt_kloop, (x, p), _mm(x, p.T), (False, True)),
            (tn_kloop, (q, y), _mm(q.T, y), (True, False)),
            (
                nd_nt,
                (qh, kv),
                _mm(qh, kv.repeat_interleave(2, 0).transpose(1, 2)),
                (False, True),
            ),
            (perm_outer_first, (x3, y), _mm(x3[:, 1].T, y), (True, False)),
        )
        for kernel, args, ref, flags in cases:
            for config in (B16, LANES):
                with self.subTest(kernel=kernel.name, config=config):
                    msl = self._check(kernel, args, config, ref)
                    self.assertEqual(_transposes(msl), flags)
        a = torch.randn(45, 72, device=DEVICE, dtype=torch.bfloat16)
        pb = torch.randn(50, 72, device=DEVICE, dtype=torch.bfloat16)
        msl = self._check(free_rows_nt, (a, pb, 2, 2), FREE32, _mm(a, pb.T))
        self.assertEqual(_transposes(msl), (False, True))

    def test_colon_k(self) -> None:
        """A ``:`` K is one op over the whole K, in no loop but those around
        the matmul; its K tile pads past K, which MPP zero-fills."""
        for k in (64, 40, 12):
            x = torch.randn(48, k, device=DEVICE, dtype=torch.float16)
            y = torch.randn(k, 40, device=DEVICE, dtype=torch.float16)
            # A NaN-padded buffer of stride k + 24: the padding is never read.
            args = (_nan_padded(x), _nan_padded(y))
            for config in (
                None,
                helion.Config(block_sizes=[16, 16]),
                helion.Config(block_sizes=[32, 32]),
            ):
                with self.subTest(kernel="colon_k", k=k, config=config):
                    msl = self._check(colon_k, args, config, _mm(x, y))
                    self.assertEqual(_run_depths(msl), [0])
        x = torch.randn(48, 64, device=DEVICE, dtype=torch.float16)
        p = torch.randn(40, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 40, device=DEVICE, dtype=torch.float16)
        w = torch.randn(64, 40, device=DEVICE, dtype=torch.float16)
        x3 = torch.randn(3, 40, 64, device=DEVICE, dtype=torch.float16)
        y3 = torch.randn(3, 64, 24, device=DEVICE, dtype=torch.float16)
        q = torch.randn(256, 48, device=DEVICE, dtype=torch.float16)
        yq = torch.randn(256, 40, device=DEVICE, dtype=torch.float16)
        c = torch.randn(2, 64, 2, 32, device=DEVICE, dtype=torch.float16)
        ps = torch.randn(2, 2, 4, 32, 32, device=DEVICE, dtype=torch.float16)
        t16 = helion.Config(block_sizes=[16, 16])
        cases = (
            (colon_k_t, (x, p), _mm(x, p.T), (None, t16), [0]),
            (colon_tn, (q, yq), _mm(q.T, yq), (t16,), [0]),
            (
                colon_k_grid,
                (x3, y3),
                _mm(x3, y3),
                (t16, helion.Config(block_sizes=[32, 16])),
                [1],
            ),
            (
                colon_two,
                (x, y, w),
                (_mm(x, y), _mm(x, w)),
                (t16, helion.Config(block_sizes=[32, 32])),
                [0, 0],
            ),
            (
                nested_colon,
                (x, y),
                _mm(x, y) * 2,
                (None, helion.Config(block_sizes=[32, 16])),
                [1],
            ),
            (
                chunk_scan_prev_state,
                (c, ps),
                _chunk_scan_prev_state_ref(c, ps),
                (None, helion.Config(block_sizes=[32, 16])),
                [0],
            ),
        )
        for kernel, args, ref, configs, depths in cases:
            for config in configs:
                with self.subTest(kernel=kernel.name, config=config):
                    msl = self._check(kernel, args, config, ref, 2 * len(depths))
                    self.assertEqual(_run_depths(msl), depths)


@_requires_mps
class TestMppViewsRejects(unittest.TestCase):
    """Forms beside the device views that stay rejected or stay staged."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(48, 64, device=DEVICE, dtype=torch.float16)
        self.y = torch.randn(64, 40, device=DEVICE, dtype=torch.float16)

    def test_colon_k(self) -> None:
        x, y = self.x, self.y
        cases = (
            (
                colon_k_carried,
                (x, y),
                ("a matmul over a ':' K whose acc an enclosing loop or if passes in"),
            ),
            (
                colon_m,
                (x[:32], y),
                (
                    "an MPP matmul whose block 0 is a single root hl.tile with no "
                    "device loop inside (a flattened tile)"
                ),
            ),
            (
                colon_tn,
                (
                    torch.randn(300, 48, device=DEVICE, dtype=torch.float16),
                    torch.randn(300, 40, device=DEVICE, dtype=torch.float16),
                ),
                "and K <= 256",
            ),
            (
                store_base_swapped,
                (x[:32], y[:, :32]),
                "a store (store) meets dimension 0",
            ),
        )
        for kernel, args, message in cases:
            with (
                self.subTest(kernel=kernel.name),
                self.assertRaisesRegex(exc.BackendUnsupported, re.escape(message)),
            ):
                _run(kernel, args, None)

    def test_staged(self) -> None:
        """A masked ``base + tile.index`` dim, an ``add`` with ``alpha``, an
        identity permute, an outer index at a tile's end and a tensor the
        kernel writes stay staged."""
        x = torch.randn(40, 96, device=DEVICE, dtype=torch.float16)
        y = torch.randn(96, 24, device=DEVICE, dtype=torch.float16)
        masked = x.clone()
        masked[:, 70:] = 0
        masked_ref = torch.stack(
            [_mm(masked[:, c : c + 16], y[c : c + 16]) for c in range(0, 96, 16)]
        )
        xa = torch.randn(40, 60, device=DEVICE, dtype=torch.float16)
        xp = torch.randn(40, 24, device=DEVICE, dtype=torch.float16)
        ya = torch.randn(24, 20, device=DEVICE, dtype=torch.float16)
        x3 = torch.randn(3, 40, 72, device=DEVICE, dtype=torch.float16)
        y3 = torch.randn(3, 72, 24, device=DEVICE, dtype=torch.float16)
        for kernel, args, ref in (
            (base_masked, (x, y, 16, 70), masked_ref),
            (alpha_tile_base, (xa, ya), _mm(xa[:, 6:30], ya)),
            (identity_permute, (xp, ya), _mm(xp, ya)),
            (nd_end, (x3, y3), _mm(x3, y3)),
        ):
            with self.subTest(kernel=kernel.name):
                out, _, msl = _run(kernel, args, B16)
                _close(out, ref)
                self.assertEqual(
                    len(re.findall(r"tensor<threadgroup half, dextents", msl)), 1
                )
        config = helion.Config(block_sizes=[16, 16, 16, 16], num_threads=[16, 16, 1, 1])
        out, _, msl = _run(nd_written, (x3, y3, torch.zeros_like(x3)), config)
        _close(out, _mm(x3 * 2, y3))
        self.assertEqual(_device_views(msl), 1)
        self.assertTrue(_fenced(msl, "x2"))


# --- Lane loops around a cooperative op -----------------------------------


@helion.kernel(**S)
def row_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=n):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm] = acc.sum(-1)
    return out


@helion.kernel(**S)
def side_store(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    side = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
        side[tm, tn] = z[tm, tn] + 1.0
    return out, side


@helion.kernel(**S)
def row_carry(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A per-row Scalar carry over the K loop beside the op."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        r = hl.zeros([tm], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            r = r + w[tm, tk].sum(-1)
        out[tm, tn] = acc * r[:, None]
    return out


@helion.kernel(**S)
def row_pre(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A per-row value from a tile loop before the K loop, stored after it."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    side = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        r = hl.zeros([tm], dtype=torch.float32)
        for tj in hl.tile(k):
            r = r + w[tm, tj].sum(-1)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
        side[tm] = r
    return out, side


@helion.kernel(**S)
def grid_side(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    side = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            s = z[tm, tn] * 3.0
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk] * 2.0, y[tk, tn])
            out[tm, tn] = acc + s
            side[tm, tn] = s - 1.0
    return out, side


@helion.kernel(**S)
def next_row(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A store, then a read of the next row of what it stored: with M on one
    thread's lanes, that row belongs to the next lane."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    side = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        t[tm, tn] = z[tm, tn] * 2.0
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
        side[tm, tn] = t[tm.index + 1, tn]
    return out, side


@helion.kernel(**S)
def colon_rowsum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A ':' K op beside a row sum over the same ':' K."""
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = torch.addmm(acc, x[tm, :], y[:, tn])
        s = x[tm, :].sum(-1)
        out[tm, tn] = acc + s[:, None]
    return out


@helion.kernel(**S)
def two_into_one(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two K loops storing into one tensor."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([2, m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        a1 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a1 = torch.addmm(a1, x[tm, tk], y[tk, tn])
        a2 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a2 = torch.addmm(a2, x[tm, tk] * 2.0, y[tk, tn])
        out[0, tm, tn] = a1
        out[1, tm, tn] = a2
    return out


@helion.kernel(**S)
def outer_lanes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A batch block around the op, stored per batch row."""
    b = 4
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb in hl.tile(b):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tb, tm, tn] = acc[None, :, :] + 0.0
    return out


@helion.kernel(**S)
def kcarry_store(
    x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, cp: torch.Tensor
) -> torch.Tensor:
    """A carry over the K block beside the op and a store, so K's lane loop
    is jammed rather than split."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    bk = hl.register_block_size(k)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        carry = hl.zeros([bk], dtype=torch.float32)
        for tk in hl.tile(k, block_size=bk):
            carry = carry + v[tk]
            acc = torch.addmm(acc, x[tm, tk] * carry[None, :], y[tk, tn])
            cp[tm, tk] = x[tm, tk]
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def colon_seed_epi(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = torch.addmm(c[tm, tn] * 0.5, x[tm, :], y[:, tn])
        out[tm, tn] = acc + w[tm][:, None]
    return out


@helion.kernel(**S)
def colon_bias1d(x: torch.Tensor, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.addmm(b[tn], x[tm, :], y[:, tn])
    return out


@helion.kernel(**S)
def colon_bias_col(x: torch.Tensor, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.addmm(b[tm][:, None], x[tm, :], y[:, tn])
    return out


@helion.kernel(**S)
def colon_bias1d_expr(
    x: torch.Tensor, y: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.addmm(torch.exp(b[tn] * 0.5), x[tm, :], y[:, tn])
    return out


@helion.kernel(**S)
def colon_row_misplaced(
    x: torch.Tensor, y: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """``b[tm]`` broadcasts along N's position, not M's."""
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.addmm(b[tm], x[tm, :], y[:, tn])
    return out


@helion.kernel(**S)
def colon_col_misplaced(
    x: torch.Tensor, y: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """``b[tn][:, None]`` sits on M's position."""
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.addmm(b[tn][:, None], x[tm, :], y[:, tn])
    return out


@helion.kernel(**S)
def colon_seed_swapped(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.addmm(c[tn, tm] + 0.0, x[tm, :], y[:, tn])
    return out


@helion.kernel(**S)
def colon_seed_relabel(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """A seed whose second term meets M and N swapped."""
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.addmm(c[tm, tn] + d[tn, tm], x[tm, :], y[:, tn])
    return out


#: M on 16 threads x 2 lanes.
M_LANES = helion.Config(block_sizes=[32, 16, 16], num_threads=[16, 16, 1])
M_LANES_2K = helion.Config(block_sizes=[32, 16, 16, 16], num_threads=[16, 16, 1, 1])
#: M and N on 16 threads x 2 lanes each.
MN_LANES = helion.Config(block_sizes=[32, 32, 16], num_threads=[16, 16, 1])
MN_LANES_2K = helion.Config(block_sizes=[32, 32, 16, 16], num_threads=[16, 16, 1, 1])
#: K on 4 threads x 8 lanes.
K_LANES = helion.Config(block_sizes=[16, 16, 32], num_threads=[16, 16, 4])


@_requires_mps
class TestMppLaneJam(unittest.TestCase):
    """A lane loop that holds a cooperative op and Scalar work reading its
    lane, other than the op's own K tile beside register and staging work,
    runs each cooperative statement once where it stands and the Scalar work
    once per lane, in body order: jammed across its lanes, or distributed
    into one rolled lane loop per run of that work
    (``TestMppLaneDistribution``)."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(48, 72, device=DEVICE)
        self.y = torch.randn(72, 40, device=DEVICE)
        self.z = torch.randn(48, 40, device=DEVICE)
        self.w = torch.randn(48, 72, device=DEVICE)
        self.rt = _mm(self.x, self.y)

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        *refs: tuple[torch.Tensor | int, torch.Tensor],
        runs: int = 1,
    ) -> None:
        """Run ``kernel`` and compare each ``(got, ref)``: a tensor argument
        the kernel writes, or the index of one of its outputs.  The lanes
        run as copies or rolled loops beside one ``run`` per op, which no
        lane loop encloses."""
        with self.subTest(kernel=kernel.name, config=config):
            out, _, msl = _run(kernel, args, config)
            outs = out if isinstance(out, tuple) else (out,)
            for got, ref in refs:
                _close(outs[got] if isinstance(got, int) else got, ref)
            self.assertRegex(msl, r"__lane1|for \(int lane_\d+ = 0")
            self.assertEqual(msl.count(".run("), runs)
            for match in re.finditer(r"\.run\(", msl):
                loops = _open_loops(msl, match.start())
                self.assertFalse([h for h in loops if "lane_" in h.split("=")[0]])

    def test_scalar_work_beside_the_op(self) -> None:
        """Side stores, materialized epilogues, Scalar seeds and per-row
        Scalar state beside an op with M or N on lanes, M and N with tails."""
        x, y, z, w, rt = self.x, self.y, self.z, self.w, self.rt
        zd, wd = z.cpu().double(), w.cpu().double()
        bias = torch.randn(40, device=DEVICE)
        for config in (
            M_LANES,
            helion.Config(
                block_sizes=[64, 32, 16], num_threads=[16, 8, 1], num_warps=32
            ),
            helion.Config(
                block_sizes=[64, 64, 16], num_threads=[16, 16, 1], num_warps=32
            ),
        ):
            self._check(side_store, (x, y, z), config, (0, rt), (1, zd + 1))
        self._check(root_scaled, (x, y, 0.5), M_LANES, (0, rt * 0.5))
        self._check(acc_from_load, (x, y, z), M_LANES, (0, rt + zd))
        self._check(seed_bcast, (x, y, bias), M_LANES, (0, rt + bias.cpu().double()))
        self._check(row_carry, (x, y, w), M_LANES, (0, rt * wd.sum(-1)[:, None]))
        self._check(row_pre, (x, y, w), M_LANES_2K, (0, rt), (1, wd.sum(-1)))
        self._check(
            grid_side, (x, y, z), M_LANES, (0, _mm(2 * x, y) + 3 * zd), (1, 3 * zd - 1)
        )
        out2 = torch.zeros(48, 40, device=DEVICE)
        ref = rt + bias.cpu().double()
        self._check(readback, (x, y, bias, out2), M_LANES, (0, ref), (out2, 2 * ref))

    def test_tile_order(self) -> None:
        """A load of what another lane of the same tile stored sees it, as
        the tile program orders them."""
        t = torch.zeros(49, 40, device=DEVICE)
        zd = self.z.cpu().double()
        ref = torch.cat([zd[1:] * 2, torch.zeros(1, 40, dtype=torch.float64)])
        config = helion.Config(
            block_sizes=[64, 32, 16], num_threads=[1, 32, 1], num_warps=32
        )
        self._check(
            next_row, (self.x, self.y, self.z, t), config, (0, self.rt), (1, ref)
        )

    def test_staged_operands(self) -> None:
        """Operands staged by the lanes that hold M or K, in fp32 and fp16."""
        x, y = self.x, self.y
        xh, yh = x.half(), y.half()
        self._check(computed_operand, (x, y), M_LANES, (0, _mm(2 * x, y)))
        self._check(computed_operand, (xh, yh), M_LANES, (0, _mm(2 * xh, yh)))
        self._check(
            computed_rhs, (x, y), MN_LANES, (0, _mm(x, torch.exp(y.cpu().double())))
        )
        cp = torch.zeros(48, 72, device=DEVICE)
        for config in (M_LANES, K_LANES, TILE16):
            self._check(
                cross_lane,
                (x, y, cp),
                config,
                (0, _mm(2 * x, y)),
                (cp, 2 * x.cpu().double()),
            )
            self._check(
                k_lane_scalar, (x, y, cp), config, (0, self.rt), (cp, x.cpu().double())
            )
            self._check(
                two_ops_staged,
                (x, y),
                config,
                (0, _mm(2 * x, y)),
                (1, _mm(x, y * 0.5)),
                runs=2,
            )

    def test_two_k_loops(self) -> None:
        x, y = self.x, self.y
        z = torch.randn(72, 40, device=DEVICE)
        self._check(
            two_k_loops_staged,
            (x, y, z),
            MN_LANES_2K,
            (0, _mm(x, torch.exp(y.cpu().double()))),
            (1, _mm(x, z * 3)),
            runs=2,
        )
        self._check(
            two_into_one,
            (x, y),
            M_LANES_2K,
            (0, torch.stack([self.rt, _mm(2 * x, y)])),
            runs=2,
        )

    def test_reductions_and_store_order(self) -> None:
        """A row sum of the result with M on lanes, an epilogue with M and N
        on one thread's lanes, and a load in K's lane loop of what its first
        lane stored."""
        x, y = self.x.half(), self.y.half()
        y64 = torch.randn(72, 64, device=DEVICE, dtype=torch.float16)
        for config in (
            helion.Config(block_sizes=[16, 16], num_threads=[8, 1], num_warps=32),
            helion.Config(block_sizes=[32, 16], num_threads=[16, 1], num_warps=32),
        ):
            self._check(row_sum_after, (x, y64), config, (0, _mm(x, y64).sum(-1)))
        self._check(
            root_scaled,
            (self.x, self.y, 0.5),
            helion.Config(
                block_sizes=[32, 32, 32], num_threads=[1, 0, 1], num_warps=32
            ),
            (0, self.rt * 0.5),
        )
        cp = torch.zeros(48, 72, device=DEVICE, dtype=torch.float16)
        xd = x.cpu().double()
        col = torch.arange(72) // 32 * 32
        self._check(
            split_reorder,
            (x, y, cp),
            K_LANES,
            (0, ((xd * (xd * 3).half().double()[:, col]) @ y.cpu().double())),
        )

    def test_colon_k_seeds(self) -> None:
        """A ':' K op seeded from a Scalar value: full, rank-1 over N,
        ``[M, 1]`` and computed, with M and N tails."""
        x, y = self.x, self.y
        c, b_n, b_m = (
            self.z,
            torch.randn(40, device=DEVICE),
            torch.randn(48, device=DEVICE),
        )
        cd, bnd, bmd = c.cpu().double(), b_n.cpu().double(), b_m.cpu().double()
        w = torch.randn(48, device=DEVICE)
        for config in (
            helion.Config(block_sizes=[16, 16], num_threads=[16, 16]),
            helion.Config(block_sizes=[16, 32], num_threads=[16, 4]),
            helion.Config(block_sizes=[32, 16], num_threads=[4, 16]),
        ):
            for kernel, args, ref in (
                (colon_k_bias, (x, y, c), cd + self.rt),
                (
                    colon_seed_epi,
                    (x, y, c, w),
                    cd * 0.5 + self.rt + w.cpu().double()[:, None],
                ),
                (colon_bias1d, (x, y, b_n), bnd + self.rt),
                (
                    colon_bias1d,
                    (x.half(), y.half(), b_n),
                    bnd + _mm(x.half(), y.half()),
                ),
                (colon_bias_col, (x, y, b_m), bmd[:, None] + self.rt),
                (colon_bias1d_expr, (x, y, b_n), torch.exp(bnd * 0.5) + self.rt),
            ):
                with self.subTest(kernel=kernel.name, config=config):
                    out, _, _ = _run(kernel, args, config)
                    _close(out, ref)
        xs, ys = torch.randn(2, 64, 64, device=DEVICE)
        cs = torch.randn(64, 64, device=DEVICE)
        ws = torch.randn(64, device=DEVICE)
        rs = _mm(xs, ys) + cs.cpu().double()
        for kernel, args, ref in (
            (colon_k_bias, (xs, ys, cs), rs),
            (
                colon_seed_epi,
                (xs, ys, cs, ws),
                rs - cs.cpu().double() * 0.5 + ws.cpu().double()[:, None],
            ),
        ):
            with self.subTest(kernel=kernel.name):
                out, _, _ = _run(kernel, args, None)
                _close(out, ref)

    def test_colon_k_beside_row_sum(self) -> None:
        xs, ys = torch.randn(2, 64, 64, device=DEVICE)
        ref = _mm(xs, ys) + xs.cpu().double().sum(-1)[:, None]
        config = helion.Config(block_sizes=[16, 32], num_threads=[16, 4])
        self._check(colon_rowsum, (xs, ys), config, (0, ref))

    def test_copies(self) -> None:
        """Nested jams multiply: 4 M lanes around 16 K lanes make 64 copies,
        the most a thread may run."""
        config = helion.Config(block_sizes=[64, 16, 16], num_threads=[16, 16, 1])
        self._check(
            two_ops_staged,
            (self.x, self.y),
            config,
            (0, _mm(2 * self.x, self.y)),
            (1, _mm(self.x, self.y * 0.5)),
            runs=2,
        )


@_requires_mps
class TestMppLaneJamRejects(_MppCase):
    """Lane loops around a cooperative op that a jam can't run, and ':' K
    seeds that would stage another element, raise ``BackendUnsupported``."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(48, 72, device=DEVICE)
        self.y = torch.randn(72, 40, device=DEVICE)
        self.xs, self.ys = torch.randn(2, 64, 64, device=DEVICE)

    def test_carry_across_lanes(self) -> None:
        """A carry over K's lane-looped block: each lane would continue the
        previous lane's value."""
        v = torch.randn(72, device=DEVICE)
        cp = torch.zeros(48, 72, device=DEVICE)
        self._rejects(
            kcarry_store,
            (self.x, self.y, v, cp),
            helion.Config(block_sizes=[16, 16, 16], num_threads=[1, 16, 16]),
            "a lane loop around an MPP op carries carry across lanes, which a copy "
            "of it per lane can't hold",
        )

    def test_cooperative_reads_lane(self) -> None:
        """A batch block on lanes around the op: its element loop would store
        each batch lane's row."""
        self._rejects(
            outer_lanes,
            (self.x, self.y),
            helion.Config(block_sizes=[2, 32, 16, 16], num_threads=[1, 16, 16, 1]),
            "a cooperative MPP statement reads a lane-dependent name",
        )

    def test_copies(self) -> None:
        """Past 64 copies of a statement per thread: 128 row lanes around an
        op whose readback a rolled run can't order."""
        self._rejects(
            readback,
            (
                self.x,
                self.y,
                torch.randn(40, device=DEVICE),
                torch.zeros(48, 40, device=DEVICE),
            ),
            helion.Config(
                block_sizes=[128, 32, 16], num_threads=[1, 32, 1], num_warps=32
            ),
            "would run 128 copies of its Scalar work per thread (max 64)",
        )

    def test_lane_reduction(self) -> None:
        """A lane reduction whose lane body holds a ``for``, the op's K loop."""
        y = torch.randn(72, 64, device=DEVICE)
        self._rejects(
            row_sum,
            (self.x, y),
            None,
            "lane reduction beside a statement the split cannot schedule",
        )

    def test_colon_k_seed_positions(self) -> None:
        """A ':' K seed whose dims, or whose producers' dims, meet M and N out
        of place."""
        xs, ys = self.xs, self.ys
        b = torch.randn(64, device=DEVICE)
        c, d = torch.randn(2, 64, 64, device=DEVICE)
        self._rejects(
            colon_row_misplaced,
            (xs, ys, b),
            None,
            "a matmul accumulator (addmm) meets dimension 0 of a value of shape "
            "[u0] at the same position of [u0, u1]: block 0 meets block 1",
        )
        self._rejects(
            colon_col_misplaced,
            (xs, ys, b),
            None,
            "a matmul accumulator (addmm) meets dimension 0 of a value of shape "
            "[u1, 1] at the same position of [u0, u1]: block 1 meets block 0",
        )
        self._rejects(
            colon_seed_swapped,
            (xs, ys, c),
            None,
            "a matmul accumulator (addmm) meets dimension 0",
        )
        self._rejects(
            colon_seed_relabel,
            (xs, ys, c, d),
            None,
            "an element-wise operand (add) meets",
        )


# --- ':' slices as an op's M or N, and staged over a ':' K ----------------


@helion.kernel(**S)
def colon_staged_lhs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    n, _ = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        out[tm, tn] = torch.addmm(acc, x[tm, :] * 0.5, y[tn, :].T)
    return out


@helion.kernel(**S)
def colon_staged_rhs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        out[tm, tn] = torch.addmm(acc, x[tm, :], torch.exp(y[:, tn] * 0.1).to(y.dtype))
    return out


@helion.kernel(**S)
def colon_staged_both(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """``exp`` is 1, not 0, at a masked position past the ':' K."""
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.addmm(
            c[tm, tn], torch.exp(x[tm, :] * 0.1).to(x.dtype), y[:, tn] * 2.0
        )
    return out


@helion.kernel(**S)
def root_staged_if(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    n, _ = y.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        if s[0] > 0:
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, x[tm, :] * 0.5, y[tn, :].T)
    return out


@helion.kernel(**S)
def colon_n_seed(p: torch.Tensor, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, k = p.shape
    _, n = v.shape
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm in hl.tile(m):
        acc = c[tm, :] + 0.0
        for tk in hl.tile(k):
            acc = torch.addmm(acc, p[tm, tk], v[tk, :])
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def colon_n_epi(p: torch.Tensor, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    m, k = p.shape
    _, n = v.shape
    out = torch.empty([m, n], dtype=torch.bfloat16, device=p.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, p[tm, tk], v[tk, :])
        out[tm, :] = (acc * s[None, :] + 1.0).to(torch.bfloat16)
    return out


@helion.kernel(**S)
def colon_n_rowsum(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    m, k = p.shape
    _, n = v.shape
    out = torch.empty([m], dtype=torch.float32, device=p.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, p[tm, tk], v[tk, :])
        out[tm] = acc.sum(-1)
    return out


@helion.kernel(**S)
def colon_n_div(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention's ``acc / l_i``: the row vector is a Scalar carry."""
    m, k = p.shape
    _, n = v.shape
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        rows = hl.zeros([tm], dtype=torch.float32)
        for tk in hl.tile(k):
            pt = p[tm, tk]
            rows = rows + pt.to(torch.float32).sum(-1)
            acc = torch.addmm(acc, pt, v[tk, :])
        out[tm, :] = acc / rows[:, None]
    return out


@helion.kernel(**S)
def seed_loop_colon_n(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """A seed loaded before an hl.grid loop and staged inside it."""
    b, m, k = x.shape
    _, n = y.shape
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        ct = c[tm, :]
        for g in hl.grid(b):
            acc = ct * 1.0
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[g, tm, tk], y[tk, :])
            out[g, tm, :] = acc
    return out


@helion.kernel(**S)
def dem_kloop_store(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The K loop writes ``p`` back, so the lhs view is staged in the loop."""
    m, k = p.shape
    _, n = v.shape
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, p[tm, tk], v[tk, :])
            p[tm, tk] = p[tm, tk]
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def colon_mn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    _, _, n = y.shape
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for g in hl.grid(b):
        acc = hl.zeros([m, n], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[g, :, tk], y[g, tk, :])
        out[g, :, :] = acc
    return out


@helion.kernel(**S)
def colon_staged_inner(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    m, _ = q.shape
    n, _ = k.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        qt = q[tm, :]
        for tn in hl.tile(n):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, qt * 0.125, k[tn, :].T)
    return out


@helion.kernel(**S)
def carry_identity(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    m, _ = q.shape
    n, _ = k.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        h = q[tm, :] * 0.125
        for tn in hl.tile(n):
            out[tm, tn] = torch.addmm(
                hl.zeros([tm, tn], dtype=torch.float32), h, k[tn, :].T
            )
            h = h * 1.0
    return out


@helion.kernel(**S)
def where_dead_load(q: torch.Tensor, k: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    m, kd = q.shape
    n, _ = k.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        qt = q[tm, :]
        for tn in hl.tile(n):
            keep = hl.full([tm, kd], tn.begin, dtype=torch.int32) >= 0
            a = torch.where(keep, qt * 0.125, q2[tm, :])
            out[tm, tn] = torch.addmm(
                hl.zeros([tm, tn], dtype=torch.float32), a, k[tn, :].T
            )
    return out


@helion.kernel(**S)
def inner_loadin(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    m, _ = q.shape
    n, _ = k.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, q[tm, :] * 0.125, k[tn, :].T)
    return out


@helion.kernel(**S)
def colon_staged_carry(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    for tm, tn in hl.tile([m, n]):
        h = x[tm, :].to(torch.float32)
        for t in hl.grid(out.size(0)):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[t, tm, tn] = torch.addmm(acc, h.to(x.dtype), y[:, tn])
            h = h * 0.5
    return out


@helion.kernel(**S)
def inv_in_if(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    m, _ = q.shape
    n, _ = k.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        qt = q[tm, :]
        for tn in hl.tile(n):
            if tn.begin >= 0:
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                out[tm, tn] = torch.addmm(acc, qt * 0.125, k[tn, :].T)
    return out


@helion.kernel(**S)
def vary_in_if(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    m, _ = q.shape
    n, _ = k.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            qs = q[tm, :] * 0.125
            if tn.begin >= 0:
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                out[tm, tn] = torch.addmm(acc, qs, k[tn, :].T)
    return out


@helion.kernel(**S)
def load_in_if(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    m, _ = q.shape
    n, _ = k.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            if tn.begin >= 0:
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                out[tm, tn] = torch.addmm(acc, q[tm, :] * 0.125, k[tn, :].T)
    return out


@helion.kernel(**S)
def while_staged(x: torch.Tensor, y: torch.Tensor, cnt: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    n, _ = y.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        i = cnt[0]
        while i > 0:
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = out[tm, tn] + torch.addmm(acc, x[tm, :] * 0.5, y[tn, :].T)
            i = i - 1
    return out


@helion.kernel(**S)
def kb_colon_n_lhs(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Every K iteration stages ``d[tm]`` broadcast over the K tile."""
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        dt = d[tm]
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (dt[:, None] + hl.zeros([tm, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, :])
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def kb_colon_n_rhs(x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    (n,) = e.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        et = e[:]
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            b = (et[None, :] + hl.zeros([tk, n], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, x[tm, tk], b)
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def kb_colon_m_lhs(x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    _, k = x.shape
    _, n = y.shape
    (m,) = d.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tn in hl.tile(n):
        dt = d[:]
        acc = hl.zeros([m, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (dt[:, None] + hl.zeros([m, tk], dtype=torch.float32)).to(x.dtype)
            acc = torch.addmm(acc, a, y[tk, tn])
        out[:, tn] = acc
    return out


@helion.kernel(**S)
def colon_n_staged(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    m, k = p.shape
    _, n = v.shape
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            a = torch.exp(p[tm, tk] * 0.1).to(p.dtype)
            acc = torch.addmm(acc, a, (v[tk, :] * 2.0).to(v.dtype))
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def colon_exp2_pre(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """``exp`` computed before the N loop, negated twice inside it."""
    m, _ = q.shape
    n, _ = k.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        qe = torch.exp(q[tm, :])
        for tn in hl.tile(n):
            out[tm, tn] = torch.addmm(
                hl.zeros([tm, tn], dtype=torch.float32), torch.neg(-qe), k[tn, :].T
            )
    return out


@helion.kernel(**S)
def dem_colon_inv(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """The N loop writes ``q`` back, so the ':' K view is staged in the loop."""
    m, _ = q.shape
    n, _ = k.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            out[tm, tn] = torch.addmm(
                hl.zeros([tm, tn], dtype=torch.float32), q[tm, :], k[tn, :].T
            )
            q[tm, :] = q[tm, :]
    return out


@helion.kernel(**S)
def hoist_exp(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    m, _ = q.shape
    n, _ = k.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        qt = q[tm, :]
        for tn in hl.tile(n):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, torch.exp(qt) * qt, k[tn, :].T)
    return out


@helion.kernel(**S)
def hoist_rhs(q: torch.Tensor, kt: torch.Tensor) -> torch.Tensor:
    """A staged rhs computed in the M loop from a tile loaded before it."""
    m, _ = q.shape
    _, n = kt.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tn in hl.tile(n):
        kb = kt[:, tn]
        for tm in hl.tile(m):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, q[tm, :], kb * 0.5)
    return out


@helion.kernel(**S)
def if_late(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """``inv_in_if`` with the op skipped in the loop's first iteration."""
    m, _ = q.shape
    n, _ = k.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        qt = q[tm, :]
        for tn in hl.tile(n):
            if tn.begin > 0:
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                out[tm, tn] = torch.addmm(acc, qt * 0.125, k[tn, :].T)
    return out


@helion.kernel(**S)
def root_if_loop(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """The N loop under an ``if``, which holds the staging before it."""
    m, _ = q.shape
    n, _ = k.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        qt = q[tm, :]
        if tm.begin >= 0:
            for tn in hl.tile(n):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                out[tm, tn] = torch.addmm(acc, qt * 0.125, k[tn, :].T)
    return out


@helion.kernel(**S)
def grid_both(q: torch.Tensor, kt: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Both operands computed in an ``hl.grid`` loop that the op doesn't
    index, from tiles loaded before it."""
    m, _ = q.shape
    _, n = kt.shape
    for tm, tn in hl.tile([m, n]):
        a = q[tm, :]
        b = kt[:, tn]
        for t in hl.grid(out.size(0)):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[t, tm, tn] = torch.addmm(acc, a * 0.5, b * 2.0)
    return out


@helion.kernel(**S)
def nest_colon(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """``qt * 0.5`` doesn't change in the N loop, which runs in another loop."""
    m, _ = q.shape
    n, _ = k.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        qt = q[tm, :]
        for _g in hl.grid(2):
            for tn in hl.tile(n):
                out[tm, tn] = torch.addmm(out[tm, tn], qt * 0.5, k[tn, :].T)
    return out


@helion.kernel(**S)
def rhs_over_loop_block(
    q: torch.Tensor, c: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    """A staged rhs over the N loop's own tile, from a column loaded before
    the loop."""
    m, kd = q.shape
    (n,) = cols.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        cc = c[:]
        for tn in hl.tile(n):
            rhs = (cc[:, None] + hl.zeros([kd, tn], dtype=torch.float32)).to(q.dtype)
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, q[tm, :], rhs)
    return out


@helion.kernel(**S)
def hoist_fill(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    m, kd = q.shape
    n, _ = k.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        qt = q[tm, :]
        for tn in hl.tile(n):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            a = qt + hl.full([kd], 1.0, dtype=q.dtype)[None, :]
            out[tm, tn] = torch.addmm(acc, a, k[tn, :].T)
    return out


@helion.kernel(**S)
def rhs_shared_block(
    q: torch.Tensor, c: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    """A staged rhs over the N loop's own block, from a value computed
    before the loop over the block size the loop shares."""
    m, kd = q.shape
    (n,) = cols.shape
    bn = hl.register_block_size(16, 64)
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        cb = c[:, None] + hl.zeros([kd, bn], dtype=torch.float32)
        for tn in hl.tile(n, block_size=bn):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, q[tm, :], (cb * 0.5).to(q.dtype))
    return out


@helion.kernel(**S)
def dem_colon_store(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """A staged ':' K lhs loaded in the N loop from a tensor the loop writes."""
    m, _ = q.shape
    n, _ = k.shape
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, q[tm, :] * 0.5, k[tn, :].T)
            q[tm, :] = q[tm, :]
    return out


@helion.kernel(**S)
def colon_kn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        out[tm, :] = torch.addmm(acc, x[tm, :] * 2.0, y[:, :])
    return out


@helion.kernel(**S)
def chain_colon(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``h``, over a ':' N, is the next op's staged ':' K operand."""
    m, _ = x.shape
    _, d = w.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        h = torch.addmm(hl.zeros([tm, d], dtype=torch.float32), x[tm, :], w[:, :])
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        out[tm, tn] = torch.addmm(acc, h.to(x.dtype), y[:, tn])
    return out


def _staged_tile(dtype: torch.dtype) -> str:
    name = {torch.bfloat16: "bfloat", torch.float16: "half", torch.float32: "float"}
    return f"threadgroup {name[dtype]} _mpp_tile["


#: A staging store before the first device loop, and one inside or after it.
_STAGED_BEFORE = r"\{ _mpp_tile\w*\[[\s\S]*for \(int offset_"
_STAGED_IN = r"for \(int offset_[\s\S]*\{ _mpp_tile\w*\["


#: N on 128 threads, no slice lanes.
N_THREADS = helion.Config(block_sizes=[16, 64], num_threads=[1, 4], num_warps=32)


@_requires_mps
class TestMppColonSlices(unittest.TestCase):
    """A ':' slice as an op's M or N, and operands staged over a ':' K
    outside every loop.  A slice holds position ``e`` on thread ``e % T``
    and slice lane ``e / T`` (a tile holds element ``e`` on thread ``e / L``
    and lane ``e % L``), so the element loop binds those."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        ref: torch.Tensor,
        *patterns: str,
        absent: tuple[str, ...] = (),
        **labels: object,
    ) -> None:
        """Run ``kernel``, compare its output with ``ref``, and search its
        MSL for each regex in ``patterns`` and for none in ``absent``."""
        with self.subTest(kernel=kernel.name, config=config, **labels):
            out, _, msl = _run(kernel, args, config)
            _close(out, ref)
            for pattern in patterns:
                self.assertRegex(msl, pattern)
            for pattern in absent:
                self.assertNotRegex(msl, pattern)

    @_full_threadgroup
    def test_staged_colon_k(self) -> None:
        """Computed operands and seeds over a ':' K at the root or under an
        ``if``, with a K tail past a non-power-of-two K."""
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for k in (64, 72):
                x = torch.randn(48, k, device=DEVICE, dtype=dtype)
                y = torch.randn(40, k, device=DEVICE, dtype=dtype)
                patterns = [re.escape(_staged_tile(dtype))]
                if k == 72:
                    patterns += [
                        re.escape("matmul2d_descriptor(32, 32, 128"),
                        re.escape("(72 - 0)"),
                    ]
                self._check(
                    colon_staged_lhs,
                    (x, y),
                    None,
                    _mm(x, y.T) * 0.5,
                    *patterns,
                    dtype=dtype,
                    k=k,
                )
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.bfloat16)
        c = torch.randn(48, 40, device=DEVICE)
        staged = (re.escape(_staged_tile(torch.bfloat16)), re.escape("(72 - 0)"))
        ey = torch.exp(y.cpu().double() * 0.1).to(torch.bfloat16)
        self._check(colon_staged_rhs, (x, y), None, _mm(x, ey), *staged)
        ex = torch.exp(x.cpu().double() * 0.1).to(torch.bfloat16)
        self._check(
            colon_staged_both,
            (x, y, c),
            helion.Config(block_sizes=[32, 16], num_threads=[16, 0], num_warps=32),
            c.cpu().double() + _mm(ex, y * 2),
            *staged,
        )
        x = torch.randn(48, 64, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(40, 64, device=DEVICE, dtype=torch.bfloat16)
        for sign in (1.0, -1.0):
            ref = _mm(x, y.T) * 0.5 if sign > 0 else torch.zeros(48, 40).double()
            self._check(
                root_staged_if,
                (x, y, torch.tensor([sign], device=DEVICE)),
                None,
                ref,
                re.escape(_staged_tile(torch.bfloat16)),
                sign=sign,
            )
        x = torch.randn(48, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 40, device=DEVICE, dtype=torch.float16)
        self._check(
            colon_k_staged,
            (x, y),
            None,
            _mm(x, y) * 2,
            re.escape(_staged_tile(torch.float16)),
        )

    def test_colon_n(self) -> None:
        """A ':' N on threads and slice lanes, on threads only, and on one
        thread's slice lanes, with an N tail."""
        p = torch.randn(48, 72, device=DEVICE, dtype=torch.bfloat16)
        for n in (128, 72):
            v = torch.randn(72, n, device=DEVICE, dtype=torch.bfloat16)
            self._check(
                colon_n_kloop,
                (p, v),
                helion.Config(block_sizes=[32, 32], num_threads=[8, 8], num_warps=32),
                _mm(p, v),
                re.escape("uint3((_e_n % 16)"),
                r"slice_lane_\d+ = \(_e_n / 16\)",
                n=n,
            )
            self._check(
                colon_n_kloop,
                (p, v),
                N_THREADS,
                _mm(p, v),
                re.escape("uint3(_e_n, 0, 0)"),
                absent=(r"slice_lane_\d+ =",),
                n=n,
            )
        v = torch.randn(72, 8, device=DEVICE, dtype=torch.bfloat16)
        self._check(
            colon_n_kloop,
            (p, v),
            None,
            _mm(p, v),
            re.escape("uint3(_e_m, 0, 0)"),
            re.escape("(_e_n / 1)"),
            n=8,
        )
        x = torch.randn(48, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 32, device=DEVICE, dtype=torch.float16)
        self._check(colon_n_kloop, (x, y), None, _mm(x, y), n=32)

    def test_colon_n_uses(self) -> None:
        """A Scalar seed, an epilogue, a row sum of the materialized result,
        a division by a Scalar carry, a seed staged in an ``hl.grid`` loop and
        an lhs view demoted to staged by a store in the K loop."""
        p = torch.randn(48, 72, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(72, 40, device=DEVICE, dtype=torch.bfloat16)
        c = torch.randn(48, 40, device=DEVICE)
        s = torch.randn(40, device=DEVICE)
        pv = _mm(p, v)
        strided = re.escape("uint3((_e_n % 4), (_e_m / 2), 0)")
        # Tiles on 16 x 16 threads leave N 4 of 1024.
        tiles16 = helion.Config(
            block_sizes=[32, 32], num_threads=[16, 16], num_warps=32
        )
        materialized = r"_mpp_mat\[.*\(indices_\d+\w* - 0\)\)\]"
        self._check(colon_n_seed, (p, v, c), tiles16, c.cpu().double() + pv, strided)
        self._check(colon_n_epi, (p, v, s), tiles16, pv * s.cpu().double() + 1, strided)
        self._check(colon_n_rowsum, (p, v), None, pv.sum(-1), materialized)
        pp = torch.rand(48, 72, device=DEVICE, dtype=torch.bfloat16)
        self._check(
            colon_n_div,
            (pp, v),
            None,
            _mm(pp, v) / pp.cpu().double().sum(-1)[:, None],
            _vec("float"),
            absent=(materialized,),
        )
        x = torch.randn(3, 48, 72, device=DEVICE, dtype=torch.bfloat16)
        for config, pattern in (
            (tiles16, strided),
            (
                helion.Config(block_sizes=[16, 16], num_threads=[2, 2], num_warps=32),
                re.escape("uint3(_e_n, (_e_m / 8), 0)"),
            ),
        ):
            self._check(
                seed_loop_colon_n,
                (x, v, c),
                config,
                c.cpu().double() + _mm(x, v),
                pattern,
            )
        p0 = p.clone()
        self._check(
            dem_kloop_store,
            (p, v),
            tiles16,
            pv,
            "__lane1",
            re.escape(_staged_tile(torch.bfloat16)),
        )
        torch.testing.assert_close(p, p0)

    def test_colon_m(self) -> None:
        """A ':' M over a K loop, bare or seeded, and M and N both ':'."""
        x = torch.randn(32, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 40, device=DEVICE, dtype=torch.float16)
        c = torch.randn(32, 40, device=DEVICE)
        one_thread = (re.escape("uint3(_e_n, 0, 0)"), re.escape("(_e_m / 1)"))
        for config in (None, helion.Config(block_sizes=[32, 32], num_warps=32)):
            self._check(colon_m_kloop, (x, y), config, _mm(x, y), *one_thread)
        self._check(
            colon_m_seeded,
            (x, y, c),
            None,
            c.cpu().double() + _mm(x, y),
            *one_thread,
        )
        for m in (64, 128):
            xb = torch.randn(3, m, 72, device=DEVICE, dtype=torch.bfloat16)
            yb = torch.randn(3, 72, 32, device=DEVICE, dtype=torch.bfloat16)
            self._check(
                colon_mn,
                (xb, yb),
                helion.Config(block_sizes=[32], num_threads=[1], num_warps=32),
                _mm(xb, yb),
                r"uint3\(_e_m, \(_e_n % \d+\), 0\)",
                m=m,
            )

    def test_hoisted(self) -> None:
        """An operand computed in its outermost loop from values loaded
        before it is staged once, before the loop, in every dtype: a ':' K
        lhs, with the op under an ``if`` (skipped in the loop's first
        iteration), with the loop under an ``if``, with a constant fill, a
        rhs, and both operands.
        Staged inside the loop in bfloat, each ':' K lhs miscompiled at its
        second config, and ``hoist_rhs`` and ``if_late`` at theirs."""
        bf = torch.bfloat16
        lanes8 = helion.Config(block_sizes=[64, 32], num_threads=[8, 8], num_warps=32)
        lanes32 = helion.Config(block_sizes=[64, 16], num_threads=[2, 8], num_warps=32)
        for dtype, cases in (
            (bf, ((colon_staged_inner, lanes8), (inv_in_if, lanes8))),
            (torch.float16, ((colon_staged_inner, lanes32),)),
            (torch.float32, ((colon_staged_inner, lanes32),)),
        ):
            q = torch.randn(48, 128, device=DEVICE, dtype=dtype)
            k = torch.randn(40, 128, device=DEVICE, dtype=dtype)
            eq = torch.exp(q.cpu().float() * 0.25).to(dtype)
            for kernel, args, expected, config in (
                *((kernel, (q, k), _mm(q * 0.125, k.T), c) for kernel, c in cases),
                (colon_exp2_pre, (q * 0.25, k), _mm(eq, k.T), lanes32),
            ):
                for c in dict.fromkeys(
                    (lanes8, config) if dtype != torch.float32 else (config,)
                ):
                    self._check(
                        kernel,
                        args,
                        c,
                        expected,
                        _STAGED_BEFORE,
                        re.escape(_staged_tile(dtype)),
                        r"for \(int lane_0 = 0",
                        absent=(_STAGED_IN,),
                        dtype=dtype,
                    )
        hoisted = (_STAGED_BEFORE, re.escape(_staged_tile(bf)))
        q = torch.randn(48, 64, device=DEVICE, dtype=bf) * 0.5
        k = torch.randn(40, 64, device=DEVICE, dtype=bf)
        qe = torch.exp(q.cpu()) * q.cpu()
        kt = torch.randn(64, 40, device=DEVICE, dtype=bf)
        late = _mm(q * 0.125, k.T)
        late[:, :32] = 0
        for kernel, args, configs, expected in (
            (
                hoist_exp,
                (q, k),
                (
                    None,
                    helion.Config(
                        block_sizes=[64, 16], num_threads=[4, 8], num_warps=32
                    ),
                ),
                _mm(qe, k.T),
            ),
            (
                hoist_rhs,
                (q, kt),
                (
                    None,
                    helion.Config(
                        block_sizes=[64, 16], num_threads=[4, 16], num_warps=32
                    ),
                ),
                _mm(q, kt * 0.5),
            ),
            (
                if_late,
                (q, k),
                (
                    helion.Config(
                        block_sizes=[64, 32], num_threads=[32, 32], num_warps=32
                    ),
                ),
                late,
            ),
            (root_if_loop, (q, k), (None,), _mm(q * 0.125, k.T)),
            (hoist_fill, (q, k), (None,), _mm(q.cpu() + 1.0, k.T)),
            (
                grid_both,
                (q, kt, torch.zeros(3, 48, 40, device=DEVICE)),
                (None,),
                _mm(q * 0.5, kt * 2.0).expand(3, 48, 40),
            ),
        ):
            for config in configs:
                self._check(
                    kernel, args, config, expected, *hoisted, absent=(_STAGED_IN,)
                )

    def test_staged_in_loop_admitted(self) -> None:
        """Operands of an op over a ':' M, N or K staged inside a loop: in
        bfloat from values the loop reloads, and in float16 and float32 from
        any value, lane-jammed at the second config but for ``kb_colon_m_lhs``,
        whose second config has no jam.  None is staged before the loop: each
        is computed from a carry, a loop offset or index, or a load in the
        loop, its loop runs in another loop, or it spans the loop's tile."""
        in_loop = (_STAGED_IN,)
        bf = torch.bfloat16
        q = torch.randn(48, 128, device=DEVICE, dtype=bf)
        k = torch.randn(40, 128, device=DEVICE, dtype=bf)
        scaled = _mm(q * 0.125, k.T)
        # The tiles on 8 x 8 threads leave the ':' dim 16 of 1024.
        tiles8 = helion.Config(block_sizes=[32, 32], num_threads=[8, 8], num_warps=32)
        jam8 = helion.Config(block_sizes=[64, 32], num_threads=[8, 8], num_warps=32)
        for kernel in (inner_loadin, vary_in_if, load_in_if):
            for config in (tiles8, jam8):
                self._check(
                    kernel,
                    (q, k),
                    config,
                    scaled,
                    re.escape(_staged_tile(bf)),
                    "__lane1",
                    *in_loop,
                    absent=(_STAGED_BEFORE,),
                )
        x = torch.randn(48, 72, device=DEVICE, dtype=bf)
        v = torch.randn(72, 128, device=DEVICE, dtype=bf)
        ex = torch.exp(x.cpu().double() * 0.1).to(bf)
        self._check(
            colon_n_staged,
            (x, v),
            tiles8,
            _mm(ex, (v.cpu().double() * 2.0).to(bf)),
            re.escape(_staged_tile(bf)),
        )
        jam32 = helion.Config(block_sizes=[64, 16], num_threads=[2, 8], num_warps=32)
        for dtype, configs in (
            (torch.float16, (tiles8, jam32)),
            (torch.float32, (jam32,)),
        ):
            q = torch.randn(48, 128, device=DEVICE, dtype=dtype)
            k = torch.randn(40, 128, device=DEVICE, dtype=dtype)
            scaled = _mm(q * 0.125, k.T)
            for kernel, args in (
                (carry_identity, (q, k)),
                (where_dead_load, (q, k, torch.randn_like(q))),
            ):
                for config in configs:
                    self._check(
                        kernel,
                        args,
                        config,
                        scaled,
                        re.escape(_staged_tile(dtype)),
                        "__lane1",
                        *in_loop,
                        absent=(_STAGED_BEFORE,),
                        dtype=dtype,
                    )
        x = torch.randn(64, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        d = torch.randn(64, device=DEVICE)
        bcast = d.cpu().to(torch.float16).double()[:, None] * y.cpu().double().sum(0)
        for config, patterns in (
            (
                helion.Config(block_sizes=[32, 32], num_threads=[16, 16], num_warps=32),
                ("__lane1",),
            ),
            (helion.Config(block_sizes=[64, 32], num_threads=[1, 1], num_warps=32), ()),
        ):
            self._check(
                kb_colon_m_lhs,
                (x, y, d),
                config,
                bcast,
                re.escape(_staged_tile(torch.float16)),
                *patterns,
            )
        h = torch.float16
        q = torch.randn(48, 64, device=DEVICE, dtype=h)
        k = torch.randn(40, 64, device=DEVICE, dtype=h)
        self._check(
            nest_colon,
            (q, k),
            None,
            _mm(q, k.T),
            re.escape(_staged_tile(h)),
            *in_loop,
            absent=(_STAGED_BEFORE,),
        )
        c = torch.randn(64, device=DEVICE)
        rhs = c.cpu()[:, None].expand(64, 40)
        for kernel, half in ((rhs_over_loop_block, 1.0), (rhs_shared_block, 0.5)):
            self._check(
                kernel,
                (q, c, torch.zeros(40, device=DEVICE)),
                None,
                _mm(q, (rhs * half).to(h)),
                re.escape(_staged_tile(h)),
                *in_loop,
                absent=(_STAGED_BEFORE,),
            )
        for dtype in (torch.bfloat16, h):
            q = torch.randn(48, 64, device=DEVICE, dtype=dtype)
            k = torch.randn(40, 64, device=DEVICE, dtype=dtype)
            self._check(
                dem_colon_store,
                (q, k),
                None,
                _mm(q * 0.5, k.T),
                re.escape(_staged_tile(dtype)),
                *in_loop,
                absent=(_STAGED_BEFORE,),
                dtype=dtype,
            )


@_requires_mps
class TestMppColonSlicesRejects(_MppCase):
    """Forms beside the ':' slices that stay rejected."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_staged_in_loop(self) -> None:
        """A bfloat operand of an op over a ':' K staged inside a for or
        while loop from a value the loop doesn't reload, and not staged
        before the loop: computed in bfloat from a carry or a loop offset, in
        a loop that runs in another loop, or inside a while loop.  Each
        config but those of ``nest_colon`` and ``while_staged`` is one where
        the form, admitted with its bfloat math, miscompiled."""
        bf = torch.bfloat16
        q = torch.randn(48, 128, device=DEVICE, dtype=bf)
        k = torch.randn(40, 128, device=DEVICE, dtype=bf)
        trigger = (None, helion.Config(block_sizes=[64, 32], num_threads=[8, 8]))
        for kernel, args, configs, op in (
            (carry_identity, (q, k), trigger, "mul.Tensor"),
            (where_dead_load, (q, k, torch.randn_like(q)), trigger, "where.self"),
            (nest_colon, (q, k), (None,), "mul.Tensor"),
        ):
            self._rejects_at(kernel, args, configs, f"{_COMPUTED} {op} that")
        x = torch.randn(48, 64, device=DEVICE, dtype=bf)
        y = torch.randn(40, 64, device=DEVICE, dtype=bf)
        self._rejects_at(
            while_staged,
            (x, y, torch.tensor([2], device=DEVICE, dtype=torch.int32)),
            (None,),
            _NOT_RELOADED,
        )

    def test_flattened_tile(self) -> None:
        """A single root ``hl.tile`` with no device loop inside as M: its
        offset is each thread's own index, not the tile's start."""
        x = torch.randn(48, 64, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(64, 32, device=DEVICE, dtype=torch.bfloat16)
        self._rejects_at(
            colon_kn,
            (x, y),
            (None,),
            "an MPP matmul whose block 0 is a single root hl.tile with no device "
            "loop inside (a flattened tile)",
        )

    def test_limits(self) -> None:
        """A ':' N past the tile table, and an op's result as the next op's
        ':' K operand with that K on threads."""
        p = torch.randn(48, 72, device=DEVICE, dtype=torch.bfloat16)
        self._rejects_at(
            colon_n_kloop,
            (p, torch.randn(72, 512, device=DEVICE, dtype=torch.bfloat16)),
            (None,),
            "one op covers M, N <= 256",
        )
        self._rejects_at(
            chain_colon,
            (
                p,
                torch.randn(72, 64, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(64, 40, device=DEVICE, dtype=torch.bfloat16),
            ),
            (None,),
            "spread over threads around a cooperative MPP matmul",
        )


# --- Batched matmuls and matmuls with no acc --------------------------------


@helion.kernel(**S)
def bmm_kloop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            acc = hl.zeros([tb, tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.baddbmm(acc, x[tb, tm, tk], y[tb, tk, tn])
            out[tb, tm, tn] = acc
    return out


@helion.kernel(**S)
def bmm_kloop_seed(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            acc = c[tb, tm, tn] * 2.0
            for tk in hl.tile(k):
                acc = torch.baddbmm(acc, x[tb, tm, tk], y[tb, tk, tn])
            out[tb, tm, tn] = acc
    return out


@helion.kernel(**S)
def bmm_qk_in(q: torch.Tensor, kt: torch.Tensor) -> torch.Tensor:
    """Attention's q @ k.T, both operands loaded in the N loop."""
    b, m, d = q.shape
    n = kt.size(1)
    d = hl.specialize(d)
    out = torch.empty([b, m, n], dtype=torch.float32, device=q.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            out[tb, tm, tn] = torch.bmm(
                q[tb, tm, :], kt[tb, tn, :].transpose(1, 2), torch.float32
            )
    return out


@helion.kernel(**S)
def bmm_qk(q: torch.Tensor, kt: torch.Tensor) -> torch.Tensor:
    """``q`` loaded before the N loop, so staged in it."""
    b, m, d = q.shape
    n = kt.size(1)
    d = hl.specialize(d)
    out = torch.empty([b, m, n], dtype=torch.float32, device=q.device)
    for tb, tm in hl.tile([b, m]):
        qq = q[tb, tm, :]
        for tn in hl.tile(n):
            kk = kt[tb, tn, :]
            out[tb, tm, tn] = torch.bmm(qq, kk.transpose(1, 2), torch.float32)
    return out


@helion.kernel(**S)
def bmm_epi(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            r = torch.bmm(x[tb, tm, :], y[tb, :, tn])
            out[tb, tm, tn] = torch.relu(r * 2.0 + c[tb, tm, tn])
    return out


@helion.kernel(**S)
def bmm_staged(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm, tn in hl.tile([b, m, n], block_size=[1, None, None]):
        out[tb, tm, tn] = torch.bmm(x[tb, tm, :] * 0.5, y[tb, :, tn])
    return out


@helion.kernel(**S)
def baddbmm_colon(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            out[tb, tm, tn] = torch.baddbmm(c[tb, tm, tn], x[tb, tm, :], y[tb, :, tn])
    return out


@helion.kernel(**S)
def bmm_unit_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([1, m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[:, tm, tn] = torch.bmm(x[:, tm, :], y[:, :, tn])
    return out


@helion.kernel(**S)
def bmm_lhs_t(xt: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, k, m = xt.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.float32, device=y.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            out[tb, tm, tn] = torch.bmm(
                xt[tb, :, tm].transpose(1, 2), y[tb, :, tn], torch.float32
            )
    return out


@helion.kernel(**S)
def bmm_masked_cols(x: torch.Tensor, y: torch.Tensor, nbn: int) -> torch.Tensor:
    """A masked batched rhs, which is staged."""
    b, m, k = x.shape
    n = y.size(2)
    out = torch.zeros([b, m, n], dtype=torch.float32, device=x.device)
    bn = hl.register_block_size(16, 64)
    for tb, tm in hl.tile([b, m]):
        for gn in hl.grid(nbn):
            cols = gn * bn + hl.arange(bn)
            cv = cols < n
            acc = hl.zeros([tb, tm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                yb = hl.load(y, [tb, tk, cols], extra_mask=cv[None, None, :])
                acc = torch.baddbmm(acc, x[tb, tm, tk], yb)
            hl.store(out, [tb, tm, cols], acc, extra_mask=cv[None, None, :])
    return out


@helion.kernel(**S)
def bmm_batch_shift(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """An lhs whose batch index is the batch tile's index plus 1, which is
    staged."""
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.zeros([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b - 1, m]):
        for tn in hl.tile(n):
            out[tb, tm, tn] = torch.bmm(
                x[tb.index + 1, tm, :], y[tb, :, tn], torch.float32
            )
    return out


@helion.kernel(**S)
def bmm_gather(x: torch.Tensor, kidx: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """An lhs gathered along K, which is staged."""
    b, _ = x.shape
    m, k = kidx.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        cols = kidx[tm, :]
        for tn in hl.tile(n):
            out[tb, tm, tn] = torch.bmm(x[tb, cols], y[tb, :, tn], torch.float32)
    return out


@helion.kernel(**S)
def bmm_tb_index_epi(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            r = torch.bmm(x[tb, tm, :], y[tb, :, tn], torch.float32)
            out[tb, tm, tn] = r + tb.index.to(torch.float32)[:, None, None] * 100.0
    return out


@helion.kernel(**S)
def bmm_chain(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """A result with no acc, cast, as the next op's staged lhs."""
    b, m, d = x.shape
    n = y.size(2)
    e = z.size(2)
    d = hl.specialize(d)
    e = hl.specialize(e)
    out = torch.empty([b, m, e], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        acc = hl.zeros([tb, tm, e], dtype=torch.float32)
        for tn in hl.tile(n):
            s = torch.bmm(x[tb, tm, :], y[tb, :, tn], torch.float32)
            acc = torch.baddbmm(acc, s.to(torch.float16), z[tb, tn, :])
        out[tb, tm, :] = acc
    return out


@helion.kernel(**S)
def bmm_tb_offset(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.zeros([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([1, 0], [b, m]):
        for tn in hl.tile(n):
            out[tb, tm, tn] = torch.bmm(x[tb, tm, :], y[tb, :, tn], torch.float32)
    return out


@helion.kernel(**S)
def bmm_4d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, h, m, k = x.shape
    n = y.size(3)
    k = hl.specialize(k)
    out = torch.empty([b, h, m, n], dtype=torch.float32, device=x.device)
    for hh in hl.grid(h):
        for tb, tm in hl.tile([b, m]):
            for tn in hl.tile(n):
                out[tb, hh, tm, tn] = torch.bmm(
                    x[tb, hh, tm, :], y[tb, hh, :, tn], torch.float32
                )
    return out


@helion.kernel(**S)
def bmm_sum_scalar(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, d = x.shape
    n = y.size(2)
    d = hl.specialize(d)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            acc = hl.zeros([tb, tm, tn], dtype=torch.float32)
            for _ in hl.grid(3):
                acc = acc + torch.bmm(x[tb, tm, :], y[tb, :, tn], torch.float32)
            out[tb, tm, tn] = acc
    return out


@helion.kernel(**S)
def matmul_op3(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            out[tb, tm, tn] = x[tb, tm, :] @ y[tb, :, tn]
    return out


@helion.kernel(**S)
def matmul_op2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = y.size(1)
    k = hl.specialize(k)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = x[tm, :] @ y[:, tn]
    return out


@helion.kernel(**S)
def mm_colon(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = y.size(1)
    k = hl.specialize(k)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.mm(x[tm, :], y[:, tn]) + 1.0
    return out


@helion.kernel(**S)
def mm_loop(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    m, d = q.shape
    n = k.size(0)
    d = hl.specialize(d)
    out = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            out[tm, tn] = torch.mm(q[tm, :], k[tn, :].T)
    return out


@helion.kernel(**S)
def mm_carry_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    g, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for h in hl.grid(g):
            acc = acc + torch.mm(x[h, tm, :], y[h, :, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def mm2_sum(
    x: torch.Tensor, y: torch.Tensor, p: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    m, k = x.shape
    n = y.size(1)
    k = hl.specialize(k)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.mm(x[tm, :], y[:, tn]) + torch.mm(p[tm, :], q[:, tn])
    return out


@helion.kernel(**S)
def mm_seed_addmm(
    x: torch.Tensor, y: torch.Tensor, p: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    m, k = x.shape
    n = y.size(1)
    k = hl.specialize(k)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        c = torch.mm(x[tm, :], y[:, tn])
        out[tm, tn] = torch.addmm(c, p[tm, :], q[:, tn])
    return out


@helion.kernel(**S)
def bmm_demoted(
    x: torch.Tensor, y: torch.Tensor, o: torch.Tensor, o2: torch.Tensor
) -> torch.Tensor:
    """``o`` is read before the store, so the store takes the Scalar layout
    through a materialization."""
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            o2[tb, tm, tn] = o[tb, tm, tn]
            o[tb, tm, tn] = torch.bmm(x[tb, tm, :], y[tb, :, tn], torch.float32)
    return o


@helion.kernel(**S)
def baddbmm_acc_swapped(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            out[tb, tm, tn] = torch.baddbmm(z[tb, tn, tm], x[tb, tm, :], y[tb, :, tn])
    return out


@helion.kernel(**S)
def baddbmm_acc_swapped_store(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """The acc and the store both swap M and N: ``baddbmm``'s value takes
    the acc's shape, so the store's positional check passes."""
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, n, m], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            out[tb, tn, tm] = torch.baddbmm(z[tb, tn, tm], x[tb, tm, :], y[tb, :, tn])
    return out


@helion.kernel(**S)
def baddbmm_acc_2d(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            out[tb, tm, tn] = torch.baddbmm(c[tm, tn], x[tb, tm, :], y[tb, :, tn])
    return out


@helion.kernel(**S)
def baddbmm_acc_unit(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            out[tb, tm, tn] = torch.baddbmm(c[:, tm, tn], x[tb, tm, :], y[tb, :, tn])
    return out


@helion.kernel(**S)
def baddbmm_acc_0d(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            out[tb, tm, tn] = torch.baddbmm(c[0], x[tb, tm, :], y[tb, :, tn])
    return out


@helion.kernel(**S)
def bmm_two_batch_tiles(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m], block_size=[1, None]):
        for tb2, tn in hl.tile([b, n], block_size=[1, None]):
            r = torch.bmm(x[tb, tm, :], y[tb2, :, tn], torch.float32)
            out[tb, tb2, tm, tn] = r[:, None, :, :]
    return out


@helion.kernel(**S)
def bmm_bf16_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.shape
    n = y.size(2)
    k = hl.specialize(k)
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=x.device)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            out[tb, tm, tn] = torch.bmm(x[tb, tm, :], y[tb, :, tn])
    return out


@helion.kernel(**S)
def dot_parts_3d(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A rank-3 ``hl.dot`` over a K held as per-thread parts."""
    b, m, k2 = a.shape
    n = w.size(2)
    out = torch.zeros([b, m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k2 // 2)
    for tb, tm in hl.tile([b, m]):
        for tn in hl.tile(n):
            acc = hl.zeros([tb, tm, tn], dtype=torch.float32)
            for tk in hl.tile(k2 // 2, block_size=bk):
                s = 2 * tk.begin
                acc = hl.dot(
                    a[tb, tm, s : s + 2 * bk], w[tb, s : s + 2 * bk, tn], acc=acc
                )
            out[tb, tm, tn] = acc
    return out


#: The batch offset folded into a view's 64-bit base, and range-tested.
_BATCH_VIEW = (
    r"long _mpp_[ab]_x0 = \(long\)\(offset_\d+\);",
    r"0 <= _mpp_[ab]_x0 && _mpp_[ab]_x0 < \(long\)\(3\)",
    r"_mpp_[ab]_x0 \* \d+\)",
)
#: A matmul with no acc fills its cooperative tensor with zeros at the op.
_ZERO_FILL = r"_mpp_c_\d+\.end\(\); _it\+\+\) \*_it = 0\.0f;"


def _bmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x.cpu().double(), y.cpu().double())


@_requires_mps
class TestMppBatched(unittest.TestCase):
    """``bmm``, ``bmm.dtype``, ``baddbmm`` and ``@`` on 3-D tiles, whose
    batch block holds one index: one rank-2 op on that index's matrices.
    And ``mm``/``bmm`` with no acc over a ':' K: the op fills a fresh
    cooperative tensor with zeros."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        ref: torch.Tensor,
        *patterns: str,
        runs: int = 1,
        absent: tuple[str, ...] = (),
    ) -> None:
        """Run ``kernel``, compare its output with ``ref``, check its MSL
        runs ``runs`` ops, and search it for each regex in ``patterns`` and
        for none in ``absent``."""
        with self.subTest(kernel=kernel.name, config=config):
            out, _, msl = _run(kernel, args, config)
            _close(out, ref)
            self.assertEqual(len(re.findall(r"\.run\(", msl)), runs)
            for pattern in patterns:
                self.assertRegex(msl, pattern)
            for pattern in absent:
                self.assertNotRegex(msl, pattern)

    @_full_threadgroup
    def test_batched(self) -> None:
        h, bf = torch.float16, torch.bfloat16
        x = torch.randn(3, 48, 72, device=DEVICE, dtype=h)
        y = torch.randn(3, 72, 40, device=DEVICE, dtype=h)
        c = torch.randn(3, 48, 40, device=DEVICE)
        for config in (
            helion.Config(block_sizes=[1, 32, 32, 32]),
            helion.Config(
                block_sizes=[1, 64, 64, 16], num_threads=[0, 32, 2, 4], num_warps=32
            ),
        ):
            self._check(bmm_kloop, (x, y), config, _bmm(x, y), *_BATCH_VIEW)
        for config in (
            helion.Config(block_sizes=[1, 32, 32, 32]),
            helion.Config(block_sizes=[1, 16, 16, 16], num_threads=[1, 0, 0, 1]),
        ):
            self._check(
                bmm_kloop_seed,
                (x, y, c),
                config,
                2 * c.cpu().double() + _bmm(x, y),
                *_BATCH_VIEW,
                "_mpp_seed",
            )
        for dtype, kernel, patterns, absent in (
            (bf, bmm_qk_in, _BATCH_VIEW, ("_mpp_tile",)),
            (h, bmm_qk, (re.escape(_staged_tile(h)), _STAGED_BEFORE), (_STAGED_IN,)),
        ):
            q = torch.randn(3, 48, 64, device=DEVICE, dtype=dtype)
            kt = torch.randn(3, 40, 64, device=DEVICE, dtype=dtype)
            configs: tuple[helion.Config | None, ...] = (None,)
            if kernel is bmm_qk_in:
                configs += (
                    helion.Config(block_sizes=[1, 32, 16]),
                    helion.Config(block_sizes=[1, 64, 32]),
                )
            for config in configs:
                self._check(
                    kernel,
                    (q, kt),
                    config,
                    _bmm(q, kt.transpose(1, 2)),
                    *patterns,
                    absent=absent,
                )
        xf = torch.randn(3, 48, 64, device=DEVICE)
        yf = torch.randn(3, 64, 40, device=DEVICE)
        cf = torch.randn(3, 48, 40, device=DEVICE)
        for config in (None, helion.Config(block_sizes=[1, 32, 32])):
            self._check(
                bmm_epi,
                (xf, yf, cf),
                config,
                torch.relu(_bmm(xf, yf) * 2 + cf.cpu().double()),
                *_BATCH_VIEW,
                absent=("_mpp_mat",),
            )
        self._check(
            bmm_tb_index_epi,
            (xf, yf),
            None,
            _bmm(xf, yf)
            + 100.0 * torch.arange(3.0, dtype=torch.float64)[:, None, None],
            *_BATCH_VIEW,
            absent=("_mpp_mat",),
        )
        self._check(
            bmm_staged,
            (xf, yf),
            None,
            _bmm(xf * 0.5, yf),
            re.escape(_staged_tile(torch.float32)),
        )
        x = torch.randn(3, 48, 64, device=DEVICE, dtype=h)
        y = torch.randn(3, 64, 40, device=DEVICE, dtype=h)
        self._check(
            baddbmm_colon,
            (x, y, cf),
            None,
            cf.cpu().double() + _bmm(x, y),
            *_BATCH_VIEW,
            "_mpp_coop|_mpp_seed",
        )
        x1, y1 = xf[:1].clone(), yf[:1].clone()
        self._check(bmm_unit_batch, (x1, y1), None, _bmm(x1, y1), absent=("_mpp_tile",))
        xt = torch.randn(3, 64, 48, device=DEVICE, dtype=h)
        self._check(
            bmm_lhs_t,
            (xt, y),
            None,
            _bmm(xt.transpose(1, 2), y),
            *_BATCH_VIEW,
            absent=("_mpp_tile",),
        )
        x = torch.randn(3, 48, 32, device=DEVICE, dtype=h)
        y = torch.randn(3, 32, 64, device=DEVICE, dtype=h)
        z = torch.randn(3, 64, 32, device=DEVICE, dtype=h)
        chained = _bmm(_bmm(x, y).to(h), z)
        for config in (None, helion.Config(block_sizes=[1, 16, 32])):
            self._check(
                bmm_chain,
                (x, y, z),
                config,
                chained,
                re.escape(_staged_tile(h)),
                runs=2,
            )
        x = torch.randn(3, 48, 64, device=DEVICE, dtype=h)
        y = torch.randn(3, 64, 40, device=DEVICE, dtype=h)
        ref = _bmm(x, y)
        ref[0] = 0
        self._check(bmm_tb_offset, (x, y), None, ref, *_BATCH_VIEW)
        ref = torch.zeros(3, 48, 40, dtype=torch.float64)
        ref[:2] = _bmm(x[1:], y[:2])
        self._check(bmm_batch_shift, (x, y), None, ref, re.escape(_staged_tile(h)))
        xg = torch.randn(3, 200, device=DEVICE, dtype=h)
        kidx = torch.randint(0, 200, (48, 64), device=DEVICE)
        self._check(
            bmm_gather,
            (xg, kidx, y),
            None,
            _bmm(xg[:, kidx.cpu()], y),
            re.escape(_staged_tile(h)),
        )
        self._check(
            bmm_masked_cols,
            (x, y, 3),
            helion.Config(block_sizes=[16, 1, 16, 16], num_threads=[0, 1, 0, 1]),
            _bmm(x, y),
            re.escape(_staged_tile(h)),
        )
        x4 = torch.randn(2, 3, 48, 64, device=DEVICE, dtype=h)
        y4 = torch.randn(2, 3, 64, 40, device=DEVICE, dtype=h)
        self._check(bmm_4d, (x4, y4), None, _bmm(x4, y4), r"_mpp_[ab]_x1 \* \d+\)")
        x = torch.randn(3, 48, 32, device=DEVICE, dtype=h)
        y = torch.randn(3, 32, 40, device=DEVICE, dtype=h)
        self._check(bmm_sum_scalar, (x, y), None, 3 * _bmm(x, y), _ZERO_FILL)
        self._check(matmul_op3, (xf, yf), None, _bmm(xf, yf), *_BATCH_VIEW)

    def test_no_acc(self) -> None:
        x = torch.randn(48, 64, device=DEVICE)
        y = torch.randn(64, 40, device=DEVICE)
        for config in (None, helion.Config(block_sizes=[32, 32])):
            self._check(mm_colon, (x, y), config, _mm(x, y) + 1, _ZERO_FILL)
        k = torch.randn(40, 64, device=DEVICE)
        self._check(mm_loop, (x, k), None, _mm(x, k.T), _ZERO_FILL)
        self._check(matmul_op2, (x, y), None, _mm(x, y), _ZERO_FILL)
        xs = torch.randn(3, 48, 64, device=DEVICE)
        ys = torch.randn(3, 64, 40, device=DEVICE)
        self._check(mm_carry_sum, (xs, ys), None, _bmm(xs, ys).sum(0), _ZERO_FILL)

    def test_no_acc_results_materialized(self) -> None:
        """A result with no acc as another op's seed, or beside another
        cooperative result in one element-wise op, goes through threadgroup
        memory."""
        x, p = (torch.randn(48, 64, device=DEVICE) for _ in range(2))
        y, q = (torch.randn(64, 40, device=DEVICE) for _ in range(2))
        ref = _mm(x, y) + _mm(p, q)
        mat = r"threadgroup float _mpp_mat\w*\["
        self._check(
            mm_seed_addmm,
            (x, y, p, q),
            None,
            ref,
            mat,
            r"threadgroup float _mpp_seed\w*\[",
            runs=2,
        )
        self._check(mm2_sum, (x, y, p, q), None, ref, mat, runs=2)


@_requires_mps
class TestMppBatchedRejects(_MppCase):
    """Batched forms and matmuls with no acc that stay rejected."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_batch_block_over_1(self) -> None:
        """A batch block of 2 or 4 on lanes, with the result stored in the
        Scalar layout: the op runs for one batch index only."""
        h = torch.float16
        x = torch.randn(3, 48, 64, device=DEVICE, dtype=h)
        y = torch.randn(3, 64, 40, device=DEVICE, dtype=h)
        o = torch.randn(3, 48, 40, device=DEVICE)
        self._rejects_at(
            bmm_demoted,
            (x, y, o, torch.zeros_like(o)),
            (
                helion.Config(block_sizes=[2, 16, 16], num_threads=[1, 0, 0]),
                helion.Config(block_sizes=[4, 16, 16], num_threads=[1, 0, 0]),
            ),
            "a batched MPP matmul whose batch block 0 is over 1",
        )

    def test_acc(self) -> None:
        """A ``baddbmm`` acc read with M and N swapped, with its result
        stored swapped too or in place; a 0-d or 2-D acc; an acc whose batch
        dim is ':' over 1 beside the batch tile."""
        h = torch.float16
        x = torch.randn(3, 32, 64, device=DEVICE, dtype=h)
        y = torch.randn(3, 64, 32, device=DEVICE, dtype=h)
        z = torch.randn(3, 32, 32, device=DEVICE)
        msg = "a batched matmul whose acc's dims are not its batch, M and N blocks"
        config = (helion.Config(block_sizes=[1, 32, 32]),)
        for kernel, acc in (
            (baddbmm_acc_swapped_store, z),
            (baddbmm_acc_swapped, z),
            (baddbmm_acc_2d, z[0]),
            (baddbmm_acc_unit, z[:1]),
            (baddbmm_acc_0d, z[0, 0]),
        ):
            self._rejects_at(kernel, (x, y, acc), config, msg)

    def test_operands(self) -> None:
        """Batch dims of two blocks, a 16-bit result, and a rank-3 ``hl.dot``,
        which keeps its rank-2 reading."""
        h = torch.float16
        x = torch.randn(3, 48, 64, device=DEVICE, dtype=h)
        y = torch.randn(3, 64, 40, device=DEVICE, dtype=h)
        self._rejects_at(
            bmm_two_batch_tiles,
            (x, y),
            (None,),
            "a batched MPP matmul whose operands' batch dims are not one block",
        )
        self._rejects_at(
            bmm_bf16_out,
            (x.bfloat16(), y.bfloat16()),
            (None,),
            "a matmul whose acc is not float32",
        )
        self._rejects_at(
            dot_parts_3d,
            (x, y),
            (None,),
            "a matmul whose blocks are not M, N and the loop's K",
        )


# --- Updates of a K loop's carry before the op ----------------------------


@helion.kernel(**S)
def upd_load(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """The carry rescaled by a per-row load, recomputed at each element."""
    m, k = p.size()
    n = hl.specialize(v.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * a[tm, tk.begin][:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, :])
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def upd_where(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """A bool per-row mask from a load selects the update."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.where((a[tm, tk.begin] > 0)[:, None], acc, acc * 0.5)
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_carry(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Attention's rescale: a factor from a per-row carry, staged."""
    m, k = p.size()
    n = hl.specialize(v.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm in hl.tile(m):
        m_i = hl.zeros([tm], dtype=torch.float32)
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            m_new = m_i + a[tm, tk.begin]
            alpha = torch.exp(m_i - m_new)
            acc = acc * alpha[:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, :])
            m_i = m_new
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def upd_colcarry(p: torch.Tensor, v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A factor from a per-column carry."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        c = hl.full([tn], 1.0, dtype=torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            c = c * 0.75 + b[tk.begin, tn]
            acc = acc * c[None, :]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_tilecarry(p: torch.Tensor, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """A second [M, N] carry, held in the Scalar layout, added."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        t = hl.zeros([tm, tn], dtype=torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            t = t * 0.5 + c[tm, tn]
            acc = acc * 0.5 + t
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_colon_tile(p: torch.Tensor, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """An [M, N] Scalar carry over a ':' N, added."""
    m, k = p.size()
    n = hl.specialize(v.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm in hl.tile(m):
        t = hl.zeros([tm, n], dtype=torch.float32)
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            t = t * 0.5 + c[tm, :]
            acc = acc * 0.5 + t
            acc = torch.addmm(acc, p[tm, tk], v[tk, :])
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def upd_rowred(p: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A factor from a row reduction over the N tile."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            s = w[tm, tn].sum(-1)
            acc = acc * torch.sigmoid(s)[:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_vec_twice(p: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """One staged per-row factor read by two nodes of the update."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            s = torch.sigmoid(w[tm, tn].sum(-1) + tk.begin * 0.01)[:, None]
            acc = acc * s + s
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_int_carry(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """An int32 per-row carry added."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        c = hl.zeros([tm], dtype=torch.int32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            c = c + (a[tm, tk.begin] > 0).to(torch.int32)
            acc = acc * 0.5 + c[:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_bool_vec(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """A bool vector from a per-row carry selects the update."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        m_i = hl.zeros([tm], dtype=torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            m_new = m_i + a[tm, tk.begin]
            big = m_new > 0.0
            acc = torch.where(big[:, None], acc * 0.5, acc)
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
            m_i = m_new
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_coop_operand(
    p: torch.Tensor, v: torch.Tensor, v2: torch.Tensor
) -> torch.Tensor:
    """The update of one acc reads another op's cooperative result."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc1 = hl.zeros([tm, tn], dtype=torch.float32)
        acc2 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc1 = torch.addmm(acc1, p[tm, tk], v[tk, tn])
            acc2 = acc2 * 0.5 + acc1
            acc2 = torch.addmm(acc2, p[tm, tk], v2[tk, tn])
        out[tm, tn] = acc2
    return out


@helion.kernel(**S)
def upd_two_accs(
    p: torch.Tensor, v: torch.Tensor, v2: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two accumulators rescaled by one staged factor."""
    m, k = p.size()
    n = hl.specialize(v.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    out2 = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm in hl.tile(m):
        m_i = hl.zeros([tm], dtype=torch.float32)
        acc = hl.zeros([tm, n], dtype=torch.float32)
        acc2 = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            m_new = m_i + a[tm, tk.begin]
            alpha = torch.exp(m_i - m_new)
            acc = acc * alpha[:, None]
            acc2 = acc2 * alpha[:, None] + 1.0
            acc = torch.addmm(acc, p[tm, tk], v[tk, :])
            acc2 = torch.addmm(acc2, p[tm, tk], v2[tk, :])
            m_i = m_new
        out[tm, :] = acc
        out2[tm, :] = acc2
    return out, out2


@helion.kernel(**S)
def upd_kred(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """A factor from a reduction over the K tile."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * torch.sigmoid(a[tm, tk].sum(-1))[:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_cast_vec(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """A staged factor cast to the operands' dtype."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        m_i = hl.zeros([tm], dtype=torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            m_new = m_i + a[tm, tk.begin]
            alpha = torch.exp(m_i - m_new).to(p.dtype)
            acc = acc * alpha[:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
            m_i = m_new
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_prevec(p: torch.Tensor, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A factor in the operands' dtype, reduced before the K loop."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        w = x[tm, :].amax(-1)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * w[:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_spread(
    p: torch.Tensor, v: torch.Tensor, a: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """The update's nodes apart, with a name reassigned between them."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        m_i = hl.zeros([tm], dtype=torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            t = acc * torch.exp(-m_i)[:, None]
            m_i = m_i + a[tm, tk.begin]
            acc = t + bias[tn][None, :] * 0.25
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_copies(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Plain name copies (``u = t``, ``acc = w``) inside the update."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        m_i = hl.zeros([tm], dtype=torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            m_new = m_i + a[tm, tk.begin]
            t = acc * torch.exp(m_i - m_new)[:, None]
            u = t
            w = u + 1.0
            acc = w
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
            m_i = m_new
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_seed(
    p: torch.Tensor, v: torch.Tensor, c: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    """A Scalar seed and an update."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = c[tm, tn] + 0.0
        for tk in hl.tile(k):
            acc = acc * a[tm, tk.begin][:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_begin(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """A K loop that starts at 8, with a staged factor."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        m_i = hl.zeros([tm], dtype=torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(8, k):
            m_new = torch.maximum(m_i, a[tm, tk.begin])
            acc = acc * torch.exp(m_i - m_new)[:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
            m_i = m_new
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_old_store(
    p: torch.Tensor, v: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The carry stored before the update, and the updated value after it."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    o1 = torch.empty([m, n], dtype=torch.float32, device=p.device)
    o2 = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            o1[tm, tn] = acc
            acc = acc * a[tm, tk.begin][:, None]
            o2[tm, tn] = acc
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out, o1, o2


@helion.kernel(**S)
def upd_late_store(
    p: torch.Tensor, v: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """An element-wise use of the updated value, stored after the op."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    o2 = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * a[tm, tk.begin][:, None]
            t2 = acc * 2.0
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
            o2[tm, tn] = t2
        out[tm, tn] = acc
    return out, o2


@helion.kernel(**S)
def upd_staged_lhs(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """``upd_carry`` with a computed lhs, staged by the K tile's lanes."""
    m, k = p.size()
    n = hl.specialize(v.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm in hl.tile(m):
        m_i = hl.zeros([tm], dtype=torch.float32)
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            m_new = m_i + a[tm, tk.begin]
            alpha = torch.exp(m_i - m_new)
            acc = acc * alpha[:, None]
            acc = torch.addmm(acc, p[tm, tk] * 0.5, v[tk, :])
            m_i = m_new
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def upd_bmm(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """A ``baddbmm`` carry rescaled by a staged per-row factor."""
    b, m, k = p.size()
    n = hl.specialize(v.size(2))
    out = torch.empty([b, m, n], dtype=torch.float32, device=p.device)
    for tb, tm in hl.tile([b, m]):
        m_i = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            m_new = m_i + a[tb, tm, tk.begin]
            acc = acc * torch.exp(m_i - m_new)[:, :, None]
            acc = torch.baddbmm(acc, p[tb, tm, tk], v[tb, tk, :])
            m_i = m_new
        out[tb, tm, :] = acc
    return out


@helion.kernel(**S)
def upd_parts(
    a: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """An ``hl.dot`` over a K held as per-thread parts, its carry rescaled."""
    m, _ = a.shape
    k, n = b0.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(k)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k, block_size=bk):
            b = torch.stack([b0[tk, tn], b1[tk, tn]], dim=1).reshape(
                bk * 2, tn.block_size
            )
            acc = acc * s[tm, tk.begin][:, None]
            acc = hl.dot(a[tm, tk.begin * 2 : tk.begin * 2 + bk * 2], b, acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_attn2d(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """Online-softmax attention with rank-2 ``addmm``s."""
    m_dim = q_in.size(0)
    n_dim = k_in.size(0)
    head_dim = hl.specialize(q_in.size(1))
    out = torch.empty_like(q_in)
    for tile_m in hl.tile(m_dim):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        for tile_n in hl.tile(n_dim):
            q = q_in[tile_m, :]
            k = k_in[tile_n, :]
            qk = torch.addmm(hl.zeros([tile_m, tile_n], dtype=torch.float32), q, k.T)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v = v_in[tile_n, :]
            p = p.to(v.dtype)
            acc = torch.addmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, None]
        out[tile_m, :] = acc.to(out.dtype)
    return out


attention_metal = helion.kernel(
    attention.fn, backend="metal", static_shapes=True, fast_math=True
)


@helion.kernel(**S)
def upd_rowdep(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """An update that reduces the carry's rows."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n], block_size=[None, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * torch.sigmoid(acc.sum(-1, keepdim=True))
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_rowdep32(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``upd_rowdep`` with N one 32-wide tile."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n], block_size=[None, 32]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * torch.sigmoid(acc.sum(-1, keepdim=True))
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_intermediate(
    p: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """An intermediate value of the update stored before the op."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    out2 = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            t = acc * 0.5
            out2[tm, tn] = t
            acc = t + 1.0
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out, out2


@helion.kernel(**S)
def upd_intermediate_late(
    p: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """An intermediate value of the update stored after the op."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    out2 = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            t = acc * 0.5
            acc = t + 1.0
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
            out2[tm, tn] = t
        out[tm, tn] = acc
    return out, out2


@helion.kernel(**S)
def upd_scalar_arg(p: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    """A float kernel argument scales the carry."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * alpha
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_swapped_load(p: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A square [N, M] load added to the [M, N] carry."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * 0.5 + w[tn, tm]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_swapped_carry(
    p: torch.Tensor, v: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """An [N, M] carry added to the [M, N] carry."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        t = hl.zeros([tn, tm], dtype=torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            t = t * 0.5 + w[tn, tm]
            acc = acc * 0.5 + t
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_vec_relabel(p: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """An [M, N] carry computed from a square [N, M] load, then added."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        t = hl.zeros([tm, tn], dtype=torch.float32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            t = t * 0.5 + w[tn, tm]
            acc = acc * 0.5 + t
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_vec11(p: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A factor reduced to one element per tile."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            s = w[tm, tn].sum(-1, keepdim=True).sum(-2, keepdim=True)
            acc = acc * torch.sigmoid(s)
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_swapped_operand(
    p: torch.Tensor, v: torch.Tensor, w: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """A recomputable operand whose own product meets [N, M] by position."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * 0.5 + c[tm, tn] * w[tn, tm]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_other_tile(p: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """An [M, B] factor, B an enclosing tile of N's extent."""
    b = w.size(1)
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([b, m, n], dtype=torch.float32, device=p.device)
    for tb in hl.tile(b):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = acc * torch.sigmoid(w[tm, tb])
                acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
            out[tb.begin, tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_result_not_carry(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """An update before the op whose result is not the loop's carry."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc * 0.5, p[tm, tk], v[tk, tn]) + 1.0
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_other_carry(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The op's acc is computed from another carry than the one it replaces."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc2 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc2 = acc2 + 1.0
            acc = torch.addmm(acc2 * 0.5, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc + acc2
    return out


@helion.kernel(**S)
def upd_written_tile(p: torch.Tensor, v: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """The update reads ``o``, which the loop stores after the op."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * 0.5 + o[tm, tn]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
            o[tm, tn] = acc
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_written_load(
    p: torch.Tensor, v: torch.Tensor, a: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """The update reads ``s`` right after the Scalar layout stores it."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            s[tm, tk.begin] = a[tm, tk.begin] * 0.5 + 1.0
            acc = acc * s[tm, tk.begin][:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_written_outer(
    p: torch.Tensor, v: torch.Tensor, a: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """The update reads a load of ``s`` taken before the K loop."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        s[tm] = a[tm] * 0.5 + 1.0
        w = s[tm]
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * w[:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_written_mask(
    p: torch.Tensor, v: torch.Tensor, w: torch.Tensor, mk: torch.Tensor
) -> torch.Tensor:
    """The update reads a load masked by a load of ``mk``."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            mk[tm] = tm.index.to(torch.float32)
            wt = hl.load(w, [tm], extra_mask=mk[tm] >= 0)
            acc = acc * wt[:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def upd_written_vec(
    p: torch.Tensor, v: torch.Tensor, a: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """A written tensor reaches the update only through a staged vector."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            s[tm, tn] = a[tm, tn] * 0.5 + 1.0
            w = s[tm, tn].amax(-1)
            acc = acc * w[:, None]
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        out[tm, tn] = acc
    return out


#: The update's element loop writes each element back before the op runs.
_UPDATE = r"\*_it = [A-Za-z_]\w*;[\s\S]*\.run\("
#: A barrier right before and right after the update's element loop.
_FENCED = (
    r"threadgroup_barrier\(mem_flags::mem_threadgroup\);\n\s*for \(auto _it = "
    r"\w+\.begin\(\)[^\n]*\{\n(?:[^\n]*\n)*?\s*\*_it = \w+;\n\s*\}\n\s*"
    r"threadgroup_barrier\(mem_flags::mem_threadgroup\);"
)
#: Of the threads that hold a vector element, one writes it.
_OWNED = r"_mpp_owner\w*\)\) \{ _mpp_vec"


def _vec(dtype: str) -> str:
    """A staged vector's threadgroup tile of ``dtype``."""
    return rf"threadgroup {dtype} _mpp_vec\w*\["


def _k_tiles(k: int, block: int, begin: int = 0) -> list[slice]:
    return [slice(k0, min(k0 + block, k)) for k0 in range(begin, k, block)]


def _d(t: torch.Tensor) -> torch.Tensor:
    return t.cpu().double()


@_requires_mps
class TestMppCarryUpdate(unittest.TestCase):
    """An element-wise update of the K loop's carry before the op, run in
    place on the cooperative accumulator by an element loop.  An operand no
    element can recompute is staged into a threadgroup tile over the
    group's M and/or N blocks, which the loop reads between two barriers."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        h = torch.float16
        self.p = torch.randn(48, 72, device=DEVICE, dtype=h)
        self.v = torch.randn(72, 40, device=DEVICE, dtype=h)
        self.a = torch.randn(48, 72, device=DEVICE) * 0.3

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        ref: Callable[[list[int]], torch.Tensor | tuple[torch.Tensor, ...]],
        *patterns: str,
        absent: tuple[str, ...] = (),
        fenced: int | None = None,
    ) -> None:
        """Run ``kernel`` at ``config`` (None: its default), compare each
        output with ``ref(block_sizes)``, and check its MSL updates the
        carry before the op, matches each regex in ``patterns`` and none in
        ``absent``, and holds ``fenced`` fenced update loops if given."""
        if config is None:
            config = kernel.bind(args).config_spec.default_config()
        with self.subTest(kernel=kernel.name, config=config):
            out, _, msl = _run(kernel, args, config)
            want = ref(list(config.config["block_sizes"]))
            outs = out if isinstance(out, tuple) else (out,)
            wants = want if isinstance(want, tuple) else (want,)
            for o, w in zip(outs, wants, strict=True):
                _close(o, w)
            for pattern in (_UPDATE, *patterns):
                self.assertRegex(msl, pattern)
            for pattern in absent:
                self.assertNotRegex(msl, pattern)
            if fenced is not None:
                self.assertEqual(len(re.findall(_FENCED, msl)), fenced)

    def _pv(self, ks: slice, v: torch.Tensor | None = None) -> torch.Tensor:
        return _d(self.p)[:, ks] @ _d(self.v if v is None else v)[ks]

    def test_recomputable_operands(self) -> None:
        """Constants, per-row loads and a bool mask from a load, read at each
        element: nothing is staged, and the update's loop needs no barrier."""
        p, v, a = self.p, self.v, self.a
        w = torch.rand(48, device=DEVICE) + 0.5
        scale = _d(self.a) * 0.5 + 1

        def ref(
            bk: int, step: Callable[[torch.Tensor, slice], torch.Tensor]
        ) -> torch.Tensor:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            for ks in _k_tiles(72, bk):
                acc = step(acc, ks) + self._pv(ks)
            return acc

        cases = [
            (inloop_scale, (p, v, w), TILE16, lambda acc, ks: acc * _d(w)[:, None]),
            (inloop_const, (p, v), TILE16, lambda acc, ks: acc * 0.5),
        ]
        for config in (None, helion.Config(block_sizes=[32, 16])):
            cases.append(
                (
                    upd_load,
                    (p, v, a * 0.5 + 1),
                    config,
                    lambda acc, ks: acc * scale[:, ks.start, None],
                )
            )
        cases.append(
            (
                upd_where,
                (p, v, a),
                None,
                lambda acc, ks: torch.where(
                    (_d(a)[:, ks.start] > 0)[:, None], acc, acc * 0.5
                ),
            )
        )
        for kernel, args, config, step in cases:
            self._check(
                kernel,
                args,
                config,
                lambda bl, step=step: ref(bl[-1], step),
                absent=(_vec(r"\w+"), "threadgroup_barrier"),
            )

    def _carry_ref(
        self, a: torch.Tensor, bk: int, p: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``upd_carry``'s result: the carry scaled by ``exp(m_i - m_new)``,
        where ``m_i`` sums ``a`` at each K tile's start."""
        p = self.p if p is None else p
        acc = torch.zeros(p.size(0), 40, dtype=torch.float64)
        m_i = torch.zeros(p.size(0), dtype=torch.float64)
        for ks in _k_tiles(72, bk):
            m_new = m_i + _d(a)[:, ks.start]
            acc = acc * torch.exp(m_i - m_new)[:, None]
            acc = acc + _d(p)[:, ks] @ _d(self.v)[ks]
            m_i = m_new
        return acc

    def test_staged_vectors(self) -> None:
        """Per-row and per-column carries, a full [M, N] Scalar carry (over an
        N tile and over a ':' N), a row reduction over the N tile and over
        the K tile, a factor the update reads twice, an int and a bool
        vector, and two accumulators that share one factor: each staged once
        per K tile, read in the update's fenced loop."""
        p, v, a = self.p, self.v, self.a
        float_vec = (_vec("float"), _FENCED, _OWNED)
        m_lanes = helion.Config(
            block_sizes=[32, 32], num_threads=[16, 16], num_warps=32
        )
        for config in (m_lanes, helion.Config(block_sizes=[16, 16], num_warps=32)):
            # M on 2 lanes: the jam runs the update once.
            jam = ("__lane1",) if config is m_lanes else ()
            self._check(
                upd_carry,
                (p, v, a),
                config,
                lambda bl: self._carry_ref(a, bl[-1]),
                *float_vec,
                *jam,
                absent=("_mpp_mat",),
            )
        # One row: the vector's read index reads no thread id.
        p1, a1 = p[:1], a[:1]
        self._check(
            upd_carry,
            (p1, v, a1),
            None,
            lambda bl: self._carry_ref(a1, bl[-1], p1),
            *float_vec,
        )

        b = torch.randn(72, 40, device=DEVICE) * 0.2

        def colcarry(bl: list[int]) -> torch.Tensor:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            c = torch.ones(40, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                c = c * 0.75 + _d(b)[ks.start]
                acc = acc * c[None, :] + self._pv(ks)
            return acc

        self._check(upd_colcarry, (p, v, b), TILE16, colcarry, *float_vec)

        c = torch.randn(48, 40, device=DEVICE)

        def tilecarry(bl: list[int]) -> torch.Tensor:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            t = torch.zeros(48, 40, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                t = t * 0.5 + _d(c)
                acc = acc * 0.5 + t + self._pv(ks)
            return acc

        for config in (TILE16, TILE32):
            self._check(upd_tilecarry, (p, v, c), config, tilecarry, *float_vec)
        for config in (None, helion.Config(block_sizes=[16, 16], num_warps=32)):
            self._check(
                upd_colon_tile,
                (p, v, c),
                config,
                lambda bl: tilecarry([0, 0, bl[1]]),
                *float_vec,
            )

        w = torch.randn(48, 40, device=DEVICE) * 0.3

        def rowred(bl: list[int]) -> torch.Tensor:
            s = torch.cat(
                [
                    torch.sigmoid(_d(w)[:, ns].sum(-1, keepdim=True)).expand(
                        48, ns.stop - ns.start
                    )
                    for ns in _k_tiles(40, bl[1])
                ],
                1,
            )
            acc = torch.zeros(48, 40, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                acc = acc * s + self._pv(ks)
            return acc

        self._check(upd_rowred, (p, v, w), TILE16, rowred, *float_vec)

        def vec_twice(bl: list[int]) -> torch.Tensor:
            rs = torch.cat(
                [
                    _d(w)[:, ns].sum(-1, keepdim=True).expand(48, ns.stop - ns.start)
                    for ns in _k_tiles(40, bl[1])
                ],
                1,
            )
            acc = torch.zeros(48, 40, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                s = torch.sigmoid(rs + ks.start * 0.01)
                acc = acc * s + s + self._pv(ks)
            return acc

        # Both of the update's readers of the factor read its one tile.
        self._check(
            upd_vec_twice,
            (p, v, w),
            TILE16,
            vec_twice,
            *float_vec,
            absent=(rf"{_vec('float')}[\s\S]*{_vec('float')}",),
        )

        def kred(bl: list[int]) -> torch.Tensor:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                acc = acc * torch.sigmoid(_d(a)[:, ks].sum(-1))[:, None]
                acc = acc + self._pv(ks)
            return acc

        # K on threads: the vector is written after a cross-thread reduction.
        self._check(
            upd_kred,
            (p, v, a),
            helion.Config(
                block_sizes=[16, 32, 16], num_threads=[2, 2, 16], num_warps=32
            ),
            kred,
            *float_vec,
        )

        def int_carry(bl: list[int]) -> torch.Tensor:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            cnt = torch.zeros(48, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                cnt = cnt + (_d(a)[:, ks.start] > 0).double()
                acc = acc * 0.5 + cnt[:, None] + self._pv(ks)
            return acc

        self._check(upd_int_carry, (p, v, a), TILE16, int_carry, _vec("int"), _FENCED)

        def bool_vec(bl: list[int]) -> torch.Tensor:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            m_i = torch.zeros(48, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                m_i = m_i + _d(a)[:, ks.start]
                acc = torch.where((m_i > 0)[:, None], acc * 0.5, acc)
                acc = acc + self._pv(ks)
            return acc

        self._check(upd_bool_vec, (p, v, a), TILE16, bool_vec, _vec("bool"), _FENCED)

        v2 = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)

        def two_accs(bl: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            acc2 = torch.zeros(48, 40, dtype=torch.float64)
            m_i = torch.zeros(48, dtype=torch.float64)
            for ks in _k_tiles(72, bl[-1]):
                m_new = m_i + _d(a)[:, ks.start]
                alpha = torch.exp(m_i - m_new)[:, None]
                acc = acc * alpha + self._pv(ks)
                acc2 = acc2 * alpha + 1.0 + self._pv(ks, v2)
                m_i = m_new
            return acc, acc2

        self._check(upd_two_accs, (p, v, v2, a), None, two_accs, *float_vec, fenced=2)

    def test_written_tensor_through_vector(self) -> None:
        """A tensor the loop writes, read only through a row reduction: the
        staged vector is computed in the Scalar layout by the threads that
        wrote it, so the update's loop re-runs no load of it."""
        p, v = self.p, self.v
        a = torch.randn(48, 40, device=DEVICE) * 0.2

        def ref(bl: list[int]) -> torch.Tensor:
            w = torch.cat(
                [
                    (_d(a)[:, ns] * 0.5 + 1.0)
                    .amax(-1, keepdim=True)
                    .expand(48, ns.stop - ns.start)
                    for ns in _k_tiles(40, bl[1])
                ],
                1,
            )
            acc = torch.zeros(48, 40, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                acc = acc * w + self._pv(ks)
            return acc

        s = torch.full((48, 40), float("nan"), device=DEVICE)
        self._check(upd_written_vec, (p, v, a, s), TILE16, ref, _vec("float"), _FENCED)

    @_full_threadgroup
    def test_cooperative_operand(self) -> None:
        """An update that reads another op's cooperative result
        materializes it and stages it as an [M, N] vector, which reuses the
        materialization's tile after its last read."""
        p, v = self.p, self.v
        v2 = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)

        def ref(bl: list[int]) -> torch.Tensor:
            acc1 = torch.zeros(48, 40, dtype=torch.float64)
            acc2 = torch.zeros(48, 40, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                acc1 = acc1 + self._pv(ks)
                acc2 = acc2 * 0.5 + acc1 + self._pv(ks, v2)
            return acc2

        for config in (TILE16, TILE32):
            self._check(
                upd_coop_operand,
                (p, v, v2),
                config,
                ref,
                _FENCED,
                r"auto (_mpp_v_\d+) = [^\n]*_mpp_mat\[[\s\S]*"
                r"threadgroup_barrier\(mem_flags::mem_threadgroup\);\n[^\n]*"
                r"\{ _mpp_mat\[[^\n]*\] = static_cast<float>\(\1\)",
            )

    def test_sixteen_bit_values(self) -> None:
        """A bf16 factor computed in the loop, and one reduced before it, are
        staged as float, so no bfloat store reaches threadgroup memory."""
        bf = torch.bfloat16
        p, v, a = self.p.to(bf), self.v.to(bf), self.a

        def cast_vec(bl: list[int]) -> torch.Tensor:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            m_i = torch.zeros(48, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                m_new = m_i + _d(a)[:, ks.start]
                alpha = torch.exp(m_i - m_new).to(bf).double()
                acc = acc * alpha[:, None] + _d(p)[:, ks] @ _d(v)[ks]
                m_i = m_new
            return acc

        for config in (TILE16, helion.Config(block_sizes=[32, 32, 16], num_warps=32)):
            self._check(upd_cast_vec, (p, v, a), config, cast_vec, _vec("float"))
        x = (torch.rand(48, 24, device=DEVICE) * 0.2 + 0.9).to(bf)

        def prevec(bl: list[int]) -> torch.Tensor:
            w = _d(x).amax(-1)[:, None]
            acc = torch.zeros(48, 40, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                acc = acc * w + _d(p)[:, ks] @ _d(v)[ks]
            return acc

        self._check(
            upd_prevec,
            (p, v, x),
            helion.Config(
                block_sizes=[16, 16, 16],
                num_threads=[2, 16, 1],
                reduction_loops=[16],
                num_warps=32,
            ),
            prevec,
            _vec("float"),
        )

    def test_readers_and_names(self) -> None:
        """The update's nodes apart with a reassigned name between them, plain
        name copies inside it, a Scalar seed, a K loop from 8, and the carry
        and the updated value read by stores before and after the op,
        through snapshots."""
        p, v, a = self.p, self.v, self.a
        scale = a * 0.5 + 1
        bias = torch.randn(40, device=DEVICE)

        def spread(bl: list[int]) -> torch.Tensor:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            m_i = torch.zeros(48, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                t = acc * torch.exp(-m_i)[:, None]
                m_i = m_i + _d(a)[:, ks.start]
                acc = t + _d(bias)[None, :] * 0.25 + self._pv(ks)
            return acc

        self._check(upd_spread, (p, v, a, bias), TILE16, spread, _FENCED)

        def copies(bl: list[int]) -> torch.Tensor:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            m_i = torch.zeros(48, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                m_new = m_i + _d(a)[:, ks.start]
                acc = acc * torch.exp(m_i - m_new)[:, None] + 1.0 + self._pv(ks)
                m_i = m_new
            return acc

        self._check(upd_copies, (p, v, a), TILE16, copies, _FENCED)
        c = torch.randn(48, 40, device=DEVICE)

        def seed(bl: list[int]) -> torch.Tensor:
            acc = _d(c)
            for ks in _k_tiles(72, bl[2]):
                acc = acc * _d(scale)[:, ks.start, None] + self._pv(ks)
            return acc

        self._check(upd_seed, (p, v, c, scale), TILE16, seed)

        def begin(bl: list[int]) -> torch.Tensor:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            m_i = torch.zeros(48, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2], begin=8):
                m_new = torch.maximum(m_i, _d(a)[:, ks.start])
                acc = acc * torch.exp(m_i - m_new)[:, None] + self._pv(ks)
                m_i = m_new
            return acc

        self._check(upd_begin, (p, v, a), TILE16, begin, _FENCED)

        def stores(bl: list[int], late: bool) -> tuple[torch.Tensor, ...]:
            acc = torch.zeros(48, 40, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                o1 = acc
                acc = acc * _d(scale)[:, ks.start, None]
                o2 = acc * 2.0 if late else acc
                acc = acc + self._pv(ks)
            return (acc, o2) if late else (acc, o1, o2)

        for config in (TILE16, TILE32):
            self._check(
                upd_old_store, (p, v, scale), config, lambda bl: stores(bl, False)
            )
            self._check(
                upd_late_store, (p, v, scale), config, lambda bl: stores(bl, True)
            )

    def test_layouts(self) -> None:
        """K on lanes beside a staged lhs (the lane split runs beside the
        update), a ``baddbmm`` with a batch block of one, and an ``hl.dot``
        over a K held as per-thread parts."""
        p, v, a = self.p, self.v, self.a

        def staged_lhs(bl: list[int]) -> torch.Tensor:
            half = (_d(p) * 0.5).half().double()
            acc = torch.zeros(48, 40, dtype=torch.float64)
            m_i = torch.zeros(48, dtype=torch.float64)
            for ks in _k_tiles(72, bl[1]):
                m_new = m_i + _d(a)[:, ks.start]
                acc = acc * torch.exp(m_i - m_new)[:, None]
                acc = acc + half[:, ks] @ _d(v)[ks]
                m_i = m_new
            return acc

        self._check(
            upd_staged_lhs,
            (p, v, a),
            helion.Config(block_sizes=[16, 16], num_threads=[16, 4]),
            staged_lhs,
            _FENCED,
            "_mpp_tile",
        )
        h = torch.float16
        p3 = torch.randn(3, 48, 72, device=DEVICE, dtype=h)
        v3 = torch.randn(3, 72, 40, device=DEVICE, dtype=h)
        a3 = torch.randn(3, 48, 72, device=DEVICE) * 0.3

        def bmm(bl: list[int]) -> torch.Tensor:
            acc = torch.zeros(3, 48, 40, dtype=torch.float64)
            m_i = torch.zeros(3, 48, dtype=torch.float64)
            for ks in _k_tiles(72, bl[2]):
                m_new = m_i + _d(a3)[:, :, ks.start]
                acc = acc * torch.exp(m_i - m_new)[:, :, None]
                acc = acc + torch.matmul(_d(p3)[:, :, ks], _d(v3)[:, ks])
                m_i = m_new
            return acc

        for config in (
            helion.Config(block_sizes=[1, 16, 16]),
            helion.Config(block_sizes=[1, 32, 16]),
        ):
            self._check(upd_bmm, (p3, v3, a3), config, bmm, _vec("float"), _FENCED)
        a2 = torch.randn(48, 144, device=DEVICE, dtype=h)
        b0 = torch.randn(72, 40, device=DEVICE, dtype=h)
        b1 = torch.randn(72, 40, device=DEVICE, dtype=h)
        s = torch.rand(48, 72, device=DEVICE) * 0.2 + 0.9

        def parts(bl: list[int]) -> torch.Tensor:
            b = torch.stack([_d(b0), _d(b1)], 1).reshape(144, 40)
            acc = torch.zeros(48, 40, dtype=torch.float64)
            for ks in _k_tiles(72, bl[0]):
                ks2 = slice(2 * ks.start, 2 * ks.stop)
                acc = acc * _d(s)[:, ks.start, None] + _d(a2)[:, ks2] @ b[ks2]
            return acc

        # The registered K block comes first.
        for config in (TILE16, helion.Config(block_sizes=[32, 16, 32])):
            self._check(upd_parts, (a2, b0, b1, s), config, parts)

    def test_attention(self) -> None:
        """Online-softmax attention in f16, as rank-2 ``addmm``s and as
        ``examples/attention.py``, at configs whose N tile is on threads."""
        h = torch.float16

        def online(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bn: int
        ) -> torch.Tensor:
            m_i = torch.full((q.size(0),), -math.inf, dtype=torch.float64)
            l_i = torch.ones(q.size(0), dtype=torch.float64)
            acc = torch.zeros(q.size(0), v.size(1), dtype=torch.float64)
            for ns in _k_tiles(k.size(0), bn):
                qk = q @ k[ns].T
                m_ij = torch.maximum(m_i, qk.amax(-1))
                pr = torch.exp2(qk - m_ij[:, None])
                alpha = torch.exp2(m_i - m_ij)
                l_i = l_i * alpha + pr.sum(-1)
                acc = acc * alpha[:, None] + pr.half().double() @ v[ns]
                m_i = m_ij
            return (acc / l_i[:, None]).half().double()

        q = torch.randn(64, 32, device=DEVICE, dtype=h)
        k = torch.randn(80, 32, device=DEVICE, dtype=h)
        v = torch.randn(80, 32, device=DEVICE, dtype=h)
        self._check(
            upd_attn2d,
            (q, k, v),
            helion.Config(block_sizes=[16, 16], num_threads=[1, 32]),
            lambda bl: online(_d(q), _d(k), _d(v), bl[1]),
            _vec("float"),
            _FENCED,
        )
        q4 = torch.randn(1, 2, 64, 32, device=DEVICE, dtype=h)
        k4 = torch.randn(1, 2, 80, 32, device=DEVICE, dtype=h)
        v4 = torch.randn(1, 2, 80, 32, device=DEVICE, dtype=h)
        scale = 1.0 / math.sqrt(32) * 1.44269504

        def attn(bl: list[int]) -> torch.Tensor:
            outs = [
                online(
                    (_d(q4)[0, b] * scale).half().double(),
                    _d(k4)[0, b],
                    _d(v4)[0, b],
                    bl[2],
                )
                for b in range(2)
            ]
            return torch.stack(outs)[None]

        config = helion.Config(block_sizes=[1, 16, 16], num_threads=[1, 16, 16])
        with self.subTest(kernel="attention", config=config):
            (out, _), _, msl = _run(attention_metal, (q4, k4, v4), config)
            _close(out, attn([1, 16, 16]))
            # q * qk_scale is staged before the N loop, and p inside it.
            for pattern in (
                _UPDATE,
                _vec("float"),
                _FENCED,
                _STAGED_BEFORE,
                _STAGED_IN,
            ):
                self.assertRegex(msl, pattern)


@_requires_mps
class TestMppCarryUpdateRejects(_MppCase):
    """Updates of the carry that stay rejected, each with its reason."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        h = torch.float16
        self.p = torch.randn(48, 72, device=DEVICE, dtype=h)
        self.v = torch.randn(72, 40, device=DEVICE, dtype=h)

    def test_not_element_wise(self) -> None:
        """An update that reduces the carry's rows."""
        msg = "which is not element-wise"
        self._rejects_at(
            upd_rowdep, (self.p, self.v), (helion.Config(block_sizes=[16, 16]),), msg
        )
        self._rejects_at(
            upd_rowdep32,
            (self.p, self.v[:, :32]),
            (helion.Config(block_sizes=[16, 16], num_threads=[8, 32]),),
            msg,
        )

    def test_intermediate_readers(self) -> None:
        """An intermediate value of the update stored before or after the op:
        after it, an element loop would read the updated carry."""
        msg = "whose intermediate value t has another reader"
        configs = (TILE16, TILE32)
        for kernel in (upd_intermediate, upd_intermediate_late):
            self._rejects_at(kernel, (self.p, self.v), configs, msg)

    def test_result_not_carry(self) -> None:
        """The op's result doesn't replace the carry its acc is computed
        from, so no carry holds it in place across iterations."""
        for kernel in (upd_result_not_carry, upd_other_carry):
            self._rejects_at(
                kernel,
                (self.p, self.v),
                (TILE16,),
                "a matmul whose acc is computed from a loop carry that its result "
                "does not replace",
            )

    def test_scalar_argument(self) -> None:
        self._rejects_at(
            upd_scalar_arg,
            (self.p, self.v, 0.7),
            (TILE16,),
            "a scalar that isn't a tensor, such as a float kernel argument",
        )

    def test_positional(self) -> None:
        """A square [N, M] load or carry added to the [M, N] carry, and an
        [M, N] carry computed from one: each element would read another."""
        p = self.p
        v = torch.randn(72, 48, device=DEVICE, dtype=torch.float16)
        w = torch.randn(48, 48, device=DEVICE)
        for kernel in (upd_swapped_load, upd_swapped_carry):
            self._rejects_at(kernel, (p, v, w), (TILE16,), mpp_matmul._REBIND)
        self._rejects_at(
            upd_swapped_operand, (p, v, w, w.clone()), (TILE16,), mpp_matmul._REBIND
        )
        self._rejects_at(
            upd_vec_relabel, (p, v, w), (TILE16,), mpp_matmul._STAGE_REBIND
        )

    def test_operand_dims(self) -> None:
        """A factor of one element per tile is over neither M nor N, and one
        over M and an enclosing tile's block is over a block the op lacks."""
        w = torch.randn(48, 40, device=DEVICE)
        msg = "neither recomputable at any element nor a value over the tile's M or N"
        self._rejects_at(upd_vec11, (self.p, self.v, w), (None, TILE16), msg)
        self._rejects_at(
            upd_other_tile,
            (self.p, self.v, w),
            (None, helion.Config(block_sizes=[16, 16, 16, 16])),
            msg,
        )

    def test_written_tensors(self) -> None:
        """The update's element loop would re-run a load of a tensor the
        kernel writes on another thread than the writer's: stored after the
        op, stored just before, loaded before the K loop, or read as a
        load's mask."""
        p, v = self.p, self.v
        a = torch.randn(48, 72, device=DEVICE)
        config = (helion.Config(block_sizes=[16, 16, 16], num_threads=[16, 16, 2]),)
        msg = (
            "a load of a tensor the kernel writes, which its element loop would re-run"
        )
        for kernel, args in (
            (upd_written_tile, (p, v, torch.zeros(48, 40, device=DEVICE))),
            (upd_written_load, (p, v, a, torch.zeros(48, 72, device=DEVICE))),
            (upd_written_outer, (p, v, a[:, 0], torch.zeros(48, device=DEVICE))),
            (
                upd_written_mask,
                (p, v, a[:, 0], torch.zeros(48, device=DEVICE)),
            ),
        ):
            self._rejects_at(kernel, args, config, msg)


# --- Epilogue vectors -------------------------------------------------------


@helion.kernel(**S)
def col_amax_add(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A per-column epilogue operand: an ``amax`` over a ``:`` slice."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        c = torch.amax(w[:, tn].to(torch.float32), 0)
        out[tm, tn] = acc * 0.5 + c[None, :]
    return out


@helion.kernel(**S)
def row_col_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A per-row and a per-column operand in one epilogue, the row one read
    twice."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        r = torch.sum(x[tm, :].to(torch.float32), -1)[:, None]
        c = torch.sum(y[:, tn].to(torch.float32), 0)[None, :]
        out[tm, tn] = (acc - r) * c + r
    return out


@helion.kernel(**S)
def row_sign(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A bool per-row operand of a ``where``."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        keep = torch.sum(x[tm, :].to(torch.float32), -1) > 0
        out[tm, tn] = torch.where(keep[:, None], acc, -acc)
    return out


@helion.kernel(**S)
def kloop_rowsum_store(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A store in the K loop, scaled by a row sum of the K tile."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            xt = x[tm, tk]
            acc = torch.addmm(acc, xt, y[tk, tn])
            s = torch.sum(xt.to(torch.float32), -1)
            out[tm, tn] = acc * s[:, None]
    return out


@helion.kernel(**S)
def colon_k_rownorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A ``:`` K op scaled by the norms of its lhs's rows."""
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        xt = x[tm, :]
        acc = torch.addmm(hl.zeros([tm, tn], dtype=torch.float32), xt, y[:, tn])
        s = torch.sum(xt.to(torch.float32) * xt.to(torch.float32), -1)
        out[tm, tn] = acc * torch.rsqrt(s[:, None] + 1.0)
    return out


@helion.kernel(**S)
def row_shared_groups(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two ops in one body, each epilogue reading one row sum."""
    m, _ = x.size()
    _, n = y.size()
    o1 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    o2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        xt = x[tm, :]
        r = torch.sum(xt.to(torch.float32), -1)
        a = torch.addmm(hl.zeros([tm, tn], dtype=torch.float32), xt, y[:, tn])
        o1[tm, tn] = a * r[:, None]
        b = torch.addmm(hl.zeros([tm, tn], dtype=torch.float32), xt, w[:, tn])
        o2[tm, tn] = b - r[:, None]
    return o1, o2


@helion.kernel(**S)
def other_rownorm(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The store's row operand sums another op's materialized result over
    the N tile."""
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        xt = x[tm, :]
        a = torch.addmm(hl.zeros([tm, tn], dtype=torch.float32), xt, y[:, tn])
        r = torch.sum(a.abs(), -1)
        b = torch.addmm(hl.zeros([tm, tn], dtype=torch.float32), xt, w[:, tn])
        out[tm, tn] = b / (r[:, None] + 1.0)
    return out


@helion.kernel(**S)
def own_rownorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The store's row operand sums the result's own materialization."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, :])
        out[tm, :] = acc / (torch.sum(acc.abs(), -1, keepdim=True) + 1.0)
    return out


@helion.kernel(**S)
def store_beside_rowsum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """An element-loop store beside a row sum over a ``:`` N, which a lane
    fold reduces."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    rs = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, :])
        out[tm, :] = acc * 2.0
        rs[tm] = torch.sum(acc.abs(), -1)
    return out, rs


@helion.kernel(**S)
def tile_sum_add(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """An [M, N] epilogue operand, a sum over a third dim."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc + torch.sum(z[tm, tn, :].to(torch.float32), -1)
    return out


@helion.kernel(**S)
def fill_seed_cols(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A ``:`` K op seeded by a fill over N, which the epilogue reads too."""
    m, _ = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        z = hl.full([tn], 0.5, dtype=torch.float32)
        r = torch.addmm(z, x[tm, :], y[:, tn])
        out[tm, tn] = r * z
    return out


@helion.kernel(**S)
def fill_seed_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A ``:`` K op seeded by an [M, 1] fill, which the epilogue reads too."""
    m, _ = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        z = hl.full([tm, 1], 2.0, dtype=torch.float32)
        r = torch.addmm(z, x[tm, :], y[:, tn])
        out[tm, tn] = r - z
    return out


@helion.kernel(**S)
def epi_vec_relabel(p: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A row operand reduced from a square [N, M] load added to an [M, N]
    value: its element (i, j) reads ``w[n0 + i, m0 + j]``."""
    m, k = p.size()
    n = v.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=p.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, p[tm, tk], v[tk, tn])
        t = hl.zeros([tm, tn], dtype=torch.float32) + w[tn, tm]
        s = torch.amax(t, 1)
        out[tm, tn] = acc * s[:, None]
    return out


@helion.kernel(**S)
def epi_row_as_col(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A row sum broadcast at the N position of a square tile."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        s = torch.sum(x[tm, :].to(torch.float32), -1)
        out[tm, tn] = acc + s[None, :]
    return out


@helion.kernel(**S)
def epi_col_as_row(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A column sum broadcast at the M position of a square tile."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        c = torch.sum(y[:, tn].to(torch.float32), 0)
        out[tm, tn] = acc + c[:, None]
    return out


def _vec_stores(msl: str) -> int:
    """How many element loops store a value read from a staged vector (an
    update's loop writes its element back instead)."""
    count = 0
    for match in re.finditer(r"for \(auto _it = \w+\.begin\(\)[^{]*\{", msl):
        depth, end = 1, match.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(msl[end], 0)
            end += 1
        body = msl[match.end() : end]
        count += "_mpp_vec" in body and "*_it = " not in body
    return count


def _attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``attention``'s output: q * qk_scale rounded to bfloat, the rest in
    float64."""
    scale = 1.0 / math.sqrt(q.size(-1)) * 1.44269504
    scores = _bf(_d(q) * scale) @ _d(k).transpose(-1, -2)
    return _bf(torch.softmax(scores * math.log(2), -1) @ _d(v))


#: M on 4 threads of 4 lanes, N on 16 threads; the ``:`` slice's reduction
#: loops over chunks of 8.
LOOPED_M_LANES = helion.Config(
    block_sizes=[16, 16, 16], num_threads=[4, 16, 0], reduction_loops=[8], num_warps=2
)
#: N on 2 threads of 8 lanes, M on 16 threads.
LOOPED_N_LANES = helion.Config(
    block_sizes=[16, 16, 16], num_threads=[16, 2, 0], reduction_loops=[16], num_warps=2
)


@_requires_mps
class TestMppEpilogueVectors(unittest.TestCase):
    """A stored epilogue's row or column operand that no element can
    recompute (a reduction, a loop carry) is staged into a threadgroup tile
    over the group's M or N block, which the store's element loop reads;
    the result itself is not materialized."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        h = torch.float16
        self.x = torch.randn(48, 72, device=DEVICE, dtype=h)
        self.y = torch.randn(72, 40, device=DEVICE, dtype=h)
        self.rt = _mm(self.x, self.y)

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        *refs: torch.Tensor,
        vecs: tuple[str, ...] = ("float",),
        mats: int = 0,
        staged_in: str | None = None,
    ) -> str:
        """Run ``kernel``, compare its outputs with ``refs``, and check its
        MSL stages vectors of the dtypes ``vecs`` that a store's element
        loop reads, inside a loop whose header contains ``staged_in`` if
        given, and holds ``mats`` materializations; return the MSL."""
        with self.subTest(kernel=kernel.name, config=config):
            out, _, msl = _run(kernel, args, config)
            outs = out if isinstance(out, tuple) else (out,)
            for o, ref in zip(outs, refs, strict=True):
                _close(o, ref)
            self.assertTrue(_tile_copies_ordered(msl), msl)
            self.assertEqual(
                re.findall(r"threadgroup (\w+) _mpp_vec\w*\[", msl), [*vecs]
            )
            self.assertEqual(_vec_stores(msl) > 0, bool(vecs))
            self.assertEqual(_layouts(msl)[1], mats)
            if staged_in is not None:
                write = re.search(r"\{ _mpp_vec\w*\[", msl)
                assert write is not None
                loops = _open_loops(msl, write.start())
                self.assertTrue([h for h in loops if staged_in in h], loops)
        return msl

    def test_row_and_column_vectors(self) -> None:
        """A row carry of the K loop, over N tiles and over a ``:`` N; and
        with M or N on lanes, a column ``amax`` over a ``:`` slice, a row
        vector read twice beside a column vector, and a bool row vector."""
        x, y, rt = self.x, self.y, self.rt
        xd = _d(x)
        rows = xd.sum(-1)[:, None]
        for block, threads in ((16, [16, 16, 1]), (32, [32, 32, 1])):
            config = helion.Config(
                block_sizes=[block, block, 16], num_threads=threads, num_warps=32
            )
            self._check(mm_rowscale, (x, y), config, rt * rows)
        pp = torch.rand(48, 72, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(72, 40, device=DEVICE, dtype=torch.bfloat16)
        self._check(colon_n_div, (pp, v), None, _mm(pp, v) / _d(pp).sum(-1)[:, None])
        w = torch.randn(24, 40, device=DEVICE, dtype=torch.float16)
        for config in (LOOPED_M_LANES, LOOPED_N_LANES):
            self._check(
                col_amax_add, (x, y, w), config, rt * 0.5 + _d(w).amax(0)[None, :]
            )
            self._check(
                row_col_sums,
                (x, y),
                config,
                (rt - rows) * _d(y).sum(0)[None, :] + rows,
                vecs=("float", "float"),
            )
            self._check(
                row_sign,
                (x, y),
                config,
                torch.where((xd.sum(-1) > 0)[:, None], rt, -rt),
                vecs=("bool",),
            )

    def test_in_k_loop(self) -> None:
        """A store in the K loop stages its row vector in the K loop, at
        the default config and with every block on threads."""
        x, y = self.x, self.y
        for config in (
            kloop_rowsum_store.bind((x, y)).config_spec.default_config(),
            helion.Config(
                block_sizes=[16, 16, 16], num_threads=[32, 32, 0], num_warps=1
            ),
        ):
            last = _k_tiles(72, config.config["block_sizes"][-1])[-1]
            self._check(
                kloop_rowsum_store,
                (x, y),
                config,
                self.rt * _d(x)[:, last].sum(-1)[:, None],
                staged_in="offset_",
            )

    def test_groups(self) -> None:
        """Two ops in one body, each staging the row vector they share (the
        two vectors, never live together, share one tile), and a row sum of
        another op's or the result's own materialization read by the store
        (another op's sum reuses its materialization's tile)."""
        x = torch.randn(48, 64, device=DEVICE, dtype=torch.float16)
        y, w = (
            torch.randn(64, 40, device=DEVICE, dtype=torch.float16) for _ in range(2)
        )
        r = _d(x).sum(-1)[:, None]
        self._check(
            row_shared_groups,
            (x, y, w),
            None,
            _mm(x, y) * r,
            _mm(x, w) - r,
            vecs=("float",),
        )
        rt = _mm(x, y)
        for bn in (16, 32):
            config = helion.Config(block_sizes=[16, bn], num_threads=[128 // bn, bn])
            msl = self._check(
                other_rownorm,
                (x, y, w),
                config,
                _mm(x, w) / (_tile_row_sums(rt.abs(), bn) + 1),
                vecs=(),
                mats=1,
            )
            self.assertRegex(msl, r"_mpp_owner\)\) \{ _mpp_mat\[")
        self._check(
            own_rownorm, (x, y), None, rt / (rt.abs().sum(-1, keepdim=True) + 1), mats=1
        )

    def test_attention(self) -> None:
        """``examples/attention.py``'s ``acc / l_i`` divides in the store's
        element loop, so only the scores are materialized, and an M tile of
        32 rows over a head dim of 128 fits the threadgroup's memory."""
        for shape, config in (
            ((1, 3, 72, 64), None),
            ((1, 3, 72, 64), helion.Config(block_sizes=[1, 32, 64], num_warps=4)),
            ((1, 1, 128, 128), helion.Config(block_sizes=[1, 32, 64], num_warps=4)),
        ):
            q, k, v = (
                torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
                for _ in range(3)
            )
            with self.subTest(shape=shape, config=config):
                (out, _), _, msl = _run(attention_metal, (q, k, v), config)
                _close(out, _attention_ref(q, k, v))
                self.assertEqual(_layouts(msl)[1], 1)
                self.assertEqual(_vec_stores(msl), 1)

    def test_split_keeps_vector_order(self) -> None:
        """A ``:`` K op's lane loop holds the row vector's staging: split at
        the op, the staging would move ahead of the lane-invariant
        statements it reads, so the loop keeps body order."""
        x = torch.randn(45, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 24, device=DEVICE, dtype=torch.float16)
        xd = _d(x)
        ref = _mm(x, y) * torch.rsqrt((xd * xd).sum(-1)[:, None] + 1)
        for config in (
            None,
            helion.Config(block_sizes=[16, 16], num_threads=[4, 16], num_warps=2),
        ):
            self._check(colon_k_rownorm, (x, y), config, ref)

    def test_store_beside_lane_fold(self) -> None:
        """An element-loop store beside a row sum over a ``:`` N that a lane
        fold reduces: the store's element loop binds its own lane names."""
        x = torch.randn(48, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 40, device=DEVICE, dtype=torch.float16)
        rt = _mm(x, y)
        self._check(
            store_beside_rowsum, (x, y), None, rt * 2, rt.abs().sum(-1), vecs=(), mats=1
        )

    def test_kept_materialized(self) -> None:
        """An [M, N] operand stays in the Scalar layout: staged, it would
        take as much threadgroup memory as the materialization.  So does a
        ``:`` K op's seed over one block, which the op updates in place."""
        x = torch.randn(32, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 48, device=DEVICE, dtype=torch.float16)
        z = torch.randn(32, 48, 8, device=DEVICE, dtype=torch.float16)
        self._check(
            tile_sum_add, (x, y, z), None, _mm(x, y) + _d(z).sum(-1), vecs=(), mats=1
        )
        x = torch.randn(37, 48, device=DEVICE)
        y = torch.randn(48, 23, device=DEVICE)
        rt = _mm(x, y)
        self._check(fill_seed_cols, (x, y), None, (rt + 0.5) * 0.5, vecs=(), mats=2)
        self._check(fill_seed_rows, (x, y), None, rt, vecs=(), mats=2)


@_requires_mps
class TestMppEpilogueVectorsRejects(_MppCase):
    """Epilogue vectors that stay rejected, each with its reason."""

    def test_positional(self) -> None:
        """A row vector reduced from a square [N, M] load, and a row or a
        column vector broadcast at the other position of a square tile:
        staged by block, each element would read another's value."""
        h = torch.float16
        p = torch.randn(48, 72, device=DEVICE, dtype=h)
        v = torch.randn(72, 48, device=DEVICE, dtype=h)
        w = torch.randn(48, 48, device=DEVICE)
        configs = (None, TILE16)
        self._rejects_at(epi_vec_relabel, (p, v, w), configs, mpp_matmul._STAGE_REBIND)
        for kernel in (epi_row_as_col, epi_col_as_row):
            self._rejects_at(kernel, (p, v), configs, mpp_matmul._REBIND)


# --- Lane reductions beside a cooperative op ------------------------------


@helion.kernel(**S)
def nloop_lse(q: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
    """A row logsumexp over a materialized product's N tiles: two dependent
    lane reductions and two carries."""
    m, d = q.size()
    n = kk.size(0)
    d = hl.specialize(d)
    out = torch.empty([m], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        mi = hl.full([tm], float("-inf"), dtype=torch.float32)
        li = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            qk = torch.mm(q[tm, :], kk[tn, :].T)
            mij = torch.maximum(mi, torch.amax(qk, -1))
            li = li * torch.exp(mi - mij) + torch.sum(torch.exp(qk - mij[:, None]), -1)
            mi = mij
        out[tm] = mi + torch.log(li)
    return out


@helion.kernel(**S)
def nloop_vec_nocarry(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A row max over N scales the acc through a staged row vector and
    stages the op's operand."""
    m, n = x.size()
    d = hl.specialize(v.size(1))
    out = torch.empty([m, d], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, d], dtype=torch.float32)
        for tn in hl.tile(n):
            s = x[tm, tn]
            mx = torch.amax(s, -1)
            p = torch.exp(s - mx[:, None])
            acc = torch.addmm(acc * torch.exp(-mx)[:, None], p, v[tn, :])
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def nloop_count(q: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
    m, d = q.size()
    n = kk.size(0)
    d = hl.specialize(d)
    out = torch.empty([m], dtype=torch.int64, device=q.device)
    for tm in hl.tile(m):
        c = hl.zeros([tm], dtype=torch.int64)
        for tn in hl.tile(n):
            qk = torch.mm(q[tm, :], kk[tn, :].T)
            c = c + (qk > 0).sum(-1)
        out[tm] = c
    return out


@helion.kernel(**S)
def kloop_rowsum_staged(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A row sum over K beside a K-loop op whose lhs is staged."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    rows = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        rs = hl.zeros([tm], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk] * 2.0, y[tk, tn])
            rs = rs + x[tm, tk].sum(-1)
        out[tm, tn] = acc
        rows[tm] = rs
    return out, rows


@helion.kernel(**S)
def nloop_offset(q: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
    m, d = q.size()
    n = kk.size(0)
    d = hl.specialize(d)
    out = torch.empty([m], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        mi = hl.full([tm], -3.5, dtype=torch.float32)
        for tn in hl.tile(5, n):
            qk = torch.mm(q[tm, :], kk[tn, :].T)
            mi = torch.maximum(mi, torch.amax(qk, -1))
        out[tm] = mi
    return out


@helion.kernel(**S)
def nloop_fenced(x2: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A staged operand loaded from a tensor an earlier loop wrote, beside a
    row max."""
    m, n = x2.size()
    d = hl.specialize(v.size(1))
    w = torch.empty([m, n], dtype=torch.float32, device=x2.device)
    out = torch.empty([m, d], dtype=torch.float32, device=x2.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            w[tm, tn] = x2[tm, tn] * 2.0
        mi = hl.full([tm], float("-inf"), dtype=torch.float32)
        acc = hl.zeros([tm, d], dtype=torch.float32)
        for tn in hl.tile(n):
            s = w[tm, tn] * 1.0
            mi = torch.maximum(mi, torch.amax(s, -1))
            acc = torch.addmm(acc, s, v[tn, :])
        out[tm, :] = acc + mi[:, None]
    return out


@helion.kernel(**S)
def nloop_rmw(q: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
    m, d = q.size()
    n = kk.size(0)
    d = hl.specialize(d)
    o = torch.zeros([m], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            qk = torch.mm(q[tm, :], kk[tn, :].T)
            o[tm] = o[tm] + torch.amax(qk, -1)
    return o


@helion.kernel(**S)
def nloop_rowmax_store(
    q: torch.Tensor, kk: torch.Tensor, part: torch.Tensor
) -> torch.Tensor:
    m, d = q.size()
    n = kk.size(0)
    d = hl.specialize(d)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            qk = torch.mm(q[tm, :], kk[tn, :].T)
            part[tm, tn.id] = torch.amax(qk, -1)
    return part


_LANE_LOOP = "for (int lane_"
#: A reduction's combine across the block's threads.
_GROUP = "helion_red::group_"


def _folds(msl: str, dtype: str) -> list[bool]:
    """For each jammed lane fold of ``dtype`` in the kernel (a combine of a
    value's lane copies), whether a lane loop encloses it."""
    cast = rf"static_cast<{dtype}>\("
    fold = rf"auto lane_\w+ = .*{cast}(\w+?)__lane0(\w*)\).*{cast}\1__lane1\2\)"
    blocks: list[str] = []
    folds = []
    for line in msl[msl.index("kernel void") :].splitlines():
        if re.search(fold, line):
            folds.append(any(_LANE_LOOP in block for block in blocks))
        for char in line:
            if char == "{":
                blocks.append(line)
            elif char == "}":
                blocks.pop()
    return folds


@_requires_mps
class TestMppLaneFold(unittest.TestCase):
    """A reduction over a tile block that its device loop walks in a lane
    loop, whose lane body also holds a cooperative op's staging writes,
    fences or materializations.  Each lane reduction becomes a lane fold and
    its combine across threads, and the op's jam runs the lane loop in tile
    order."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.q = torch.randn(48, 64, device=DEVICE)
        self.kk = torch.randn(80, 64, device=DEVICE)
        self.s = self.q.cpu().double() @ self.kk.cpu().double().T

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        ref: Callable[[list[int]], torch.Tensor | tuple[torch.Tensor | None, ...]],
        *patterns: str,
        dtype: str = "float",
        combined: bool = True,
        fenced: str | None = None,
    ) -> None:
        """Run ``kernel`` at ``config`` (None: its default), compare each
        output with ``ref(block_sizes)`` (None: unchecked), and check its MSL
        has jammed folds of ``dtype``, none in a lane loop, a combine across
        threads when ``combined``, a match of each regex in ``patterns``,
        and a fence before the first read of ``fenced`` if given."""
        if config is None:
            config = kernel.bind(args).config_spec.default_config()
        with self.subTest(kernel=kernel.name, config=config):
            out, _, msl = _run(kernel, args, config)
            want = ref(list(config.config["block_sizes"]))
            outs = out if isinstance(out, tuple) else (out,)
            wants = want if isinstance(want, tuple) else (want,)
            for o, w in zip(outs, wants, strict=True):
                if w is not None:
                    _close(o, w)
            folds = _folds(msl, dtype)
            self.assertTrue(folds)
            self.assertFalse(any(folds))
            for pattern in patterns:
                self.assertRegex(msl, pattern)
            if combined:
                self.assertIn(_GROUP, msl)
            else:
                self.assertNotIn(_GROUP, msl)
            if fenced is not None:
                self.assertTrue(_fenced(msl, fenced))

    def test_materialized(self) -> None:
        """Row reductions of a materialized product, jammed with M's grid
        lanes (each fold and combine copied per M lane), and on one thread
        walking every lane, with no combine."""
        lse = torch.logsumexp(self.s, -1)
        for config in (
            helion.Config(block_sizes=[32, 32], num_threads=[16, 16], num_warps=32),
            helion.Config(block_sizes=[32, 32], num_threads=[8, 16], num_warps=32),
        ):
            self._check(
                nloop_lse,
                (self.q, self.kk),
                config,
                lambda bl: lse,
                r"helion_red::group_max\(\w+, \w+, static_cast<float>\(\w+__lane1\)",
            )
        self._check(
            nloop_lse,
            (self.q, self.kk),
            helion.Config(block_sizes=[16, 16], num_threads=[16, 1], num_warps=32),
            lambda bl: lse,
            combined=False,
        )
        self._check(
            nloop_offset,
            (self.q, self.kk),
            helion.Config(block_sizes=[16, 64], num_threads=[4, 16], num_warps=32),
            lambda bl: self.s[:, 5:].amax(-1).clamp(min=-3.5),
        )

    def test_count(self) -> None:
        """An int64 count folds in int32."""
        self._check(
            nloop_count,
            (self.q, self.kk),
            None,
            lambda bl: (self.s > 0).sum(-1).double(),
            dtype="int",
        )

    def test_staged(self) -> None:
        """Staged operands and a staged row vector, written per lane ahead of
        the op, and a fenced load, at a config whose combine runs across 32
        SIMD groups."""
        x = torch.randn(48, 80, device=DEVICE)
        v = torch.randn(80, 32, device=DEVICE)

        def vec(bl: list[int]) -> torch.Tensor:
            acc = torch.zeros(48, 32, dtype=torch.float64)
            for ns in _k_tiles(80, bl[1]):
                s = _d(x)[:, ns]
                mx = s.amax(-1)
                acc = (
                    acc * torch.exp(-mx)[:, None]
                    + torch.exp(s - mx[:, None]) @ _d(v)[ns]
                )
            return acc

        self._check(
            nloop_vec_nocarry,
            (x, v),
            helion.Config(block_sizes=[16, 32], num_threads=[16, 8]),
            vec,
        )
        y = torch.randn(80, 32, device=DEVICE)
        self._check(
            kloop_rowsum_staged,
            (x, y),
            None,
            lambda bl: (2 * _d(x) @ _d(y), _d(x).sum(-1)),
            combined=False,
        )
        w = 2 * _d(x)
        for config, combined in (
            (None, False),
            (helion.Config(block_sizes=[8, 16, 128], num_threads=[1, 1, 64]), True),
        ):
            self._check(
                nloop_fenced,
                (x, v),
                config,
                lambda bl: w @ _d(v) + w.amax(-1)[:, None],
                combined=combined,
                fenced="w",
            )

    def test_attention(self) -> None:
        """``examples/attention.py`` in bfloat at an N tail: its row max and
        row sum fold N's lanes beside both ops."""
        bf = torch.bfloat16
        q = torch.randn(1, 2, 64, 64, device=DEVICE, dtype=bf)
        k = torch.randn(1, 2, 80, 64, device=DEVICE, dtype=bf)
        v = torch.randn(1, 2, 80, 64, device=DEVICE, dtype=bf)
        scale = 1.0 / math.sqrt(64) * 1.44269504

        def online(b: int, bn: int) -> torch.Tensor:
            qs = _bf(_d(q)[0, b] * scale)
            m_i = torch.full((64,), -math.inf, dtype=torch.float64)
            l_i = torch.ones(64, dtype=torch.float64)
            acc = torch.zeros(64, 64, dtype=torch.float64)
            for ns in _k_tiles(80, bn):
                qk = qs @ _d(k)[0, b, ns].T
                m_ij = torch.maximum(m_i, qk.amax(-1))
                p = torch.exp2(qk - m_ij[:, None])
                alpha = torch.exp2(m_i - m_ij)
                l_i = l_i * alpha + p.sum(-1)
                acc = acc * alpha[:, None] + _bf(p) @ _d(v)[0, b, ns]
                m_i = m_ij
            return _bf(acc / l_i[:, None])

        for config in (
            helion.Config(block_sizes=[1, 16, 16], num_threads=[1, 2, 4], num_warps=32),
            helion.Config(
                block_sizes=[1, 16, 32], num_threads=[1, 16, 16], num_warps=32
            ),
        ):
            self._check(
                attention_metal,
                (q, k, v),
                config,
                lambda bl: (
                    torch.stack([online(b, bl[2]) for b in range(2)])[None],
                    None,
                ),
                "helion_red::group_max",
                "helion_red::group_sum",
            )

    @_full_threadgroup
    def test_device_views_keep_split(self) -> None:
        """A lane body whose only MPP statement is an op over device views
        keeps the pass split."""
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        ref = _mm(x, y) * _d(x).sum(-1)[:, None]
        with self.subTest(config=TILE16):
            out, _, msl = _run(mm_rowscale, (x, y), TILE16)
            torch.testing.assert_close(out.cpu().double(), ref, atol=0.5, rtol=2e-2)
            kernel = msl[msl.index("kernel void") :]
            self.assertIn(_LANE_LOOP, kernel)
            self.assertNotIn("__lane0", kernel)


@_requires_mps
class TestMppLaneFoldRejects(_MppCase):
    """Lane bodies beside a cooperative op that the fold route rejects, as
    the pass split does."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.q = torch.randn(48, 64, device=DEVICE)
        self.kk = torch.randn(80, 64, device=DEVICE)

    def test_memory(self) -> None:
        """A tensor stored and loaded in the lane body, whose read-modify-write
        races on the threads of the head dim, and a store into an argument
        that may share memory with another."""
        part = torch.full((48, 8), -7.0, device=DEVICE)
        for kernel, args, config, message in (
            (
                nloop_rmw,
                (self.q, self.kk),
                helion.Config(block_sizes=[16, 16], num_threads=[16, 1]),
                "lane reduction: o is stored and loaded in its lane loop",
            ),
            (
                nloop_rowmax_store,
                (self.q, self.kk, part),
                None,
                (
                    "lane reduction: part is stored in its lane loop and may "
                    "share memory with kk"
                ),
            ),
        ):
            self._rejects(kernel, args, config, message)


# --- hl.dot accumulators seeded away from zero -----------------------------


@helion.kernel(**S)
def dot_seed_zeros(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_seed_full(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.full([tm, tn], 1.5, dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_seed_load(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = c[tm, tn].to(torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


# --- Loads-first staging lane loops ---------------------------------------


@helion.kernel(**S)
def cast_rhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    n = b.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn].to(torch.bfloat16))
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def zero_point_rhs(a: torch.Tensor, w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            rhs = (w[tk, tn].to(torch.int32) - z[tk, tn].to(torch.int32)).to(
                torch.bfloat16
            )
            acc = torch.addmm(acc, a[tm, tk], rhs)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def scaled_rhs(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    n = b.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            rhs = (b[tk, tn].to(torch.float32) * s[tk, tn]).to(torch.bfloat16)
            acc = torch.addmm(acc, a[tm, tk], rhs)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def gathered_rhs(a: torch.Tensor, w: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], w[rows[tk], tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def lanes_named(a: torch.Tensor, b: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """A user variable bound to a load keeps its name in the generated code."""
    m, k = a.size()
    n = b.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            w = b[tk, tn]
            _mpp_lanes_0 = z[tk, tn]
            rhs = (w.to(torch.int32) - _mpp_lanes_0.to(torch.int32)).to(torch.bfloat16)
            acc = torch.addmm(acc, a[tm, tk], rhs)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def index_rhs(a: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    n = cols.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            rhs = tk.index[:, None] + hl.zeros([tk, tn], dtype=torch.int32)
            acc = torch.addmm(acc, a[tm, tk], rhs.to(torch.bfloat16))
        out[tm, tn] = acc
    return out


def _k_lanes(lanes: int) -> helion.Config:
    """A [16, 16, K] tile whose K threads each stage ``lanes`` K rows."""
    block_k = 64 if lanes < 32 else 128
    return helion.Config(
        block_sizes=[16, 16, block_k], num_threads=[1, 16, block_k // lanes]
    )


def _lane_arrays(msl: str) -> list[tuple[str, str, int]]:
    """Each lane array the MSL declares: its element type, name and lanes."""
    return [
        (dtype, name, int(lanes))
        for dtype, name, lanes in re.findall(r"(\w+) (_mpp_lanes_\d+)\[(\d+)\];", msl)
    ]


@_requires_mps
class TestMppLoadsFirst(unittest.TestCase):
    """A staging lane loop split at its MPP op whose device loads are all
    narrower than 32 bits, fit 64 bytes of lane arrays together and cover at
    most 32 lanes runs as two lane loops: the first issues every lane's loads
    into per-lane arrays, the second stages from them.  Any other staging
    lane loop stays whole."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def _assert_loads_first(self, msl: str, arrays: list[tuple[str, str, int]]) -> None:
        """``arrays`` are declared, and each is filled in one lane loop and
        read, under the load's mask, in a later one."""
        self.assertEqual(_lane_arrays(msl), arrays)
        for _, name, _ in arrays:
            put = re.search(rf"{name}\[(lane_\d+)\] = \w+;", msl)
            assert put is not None
            get = re.search(
                rf"for \(int {put[1]} = 0;[^{{]*\{{[^}}]*= \([^;]+ \? "
                rf"{name}\[{put[1]}\] : 0\);",
                msl[put.end() :],
            )
            self.assertIsNotNone(get, name)

    def test_narrow_loads(self) -> None:
        """An int8 rhs at 8, 16 and 32 lanes with one K tile, a K tail and
        several K tiles, and int4_gemm's packed rhs at 32 lanes."""
        a = torch.randn(5, 256).to(torch.bfloat16)
        b = torch.randint(-128, 128, (256, 24), dtype=torch.int8)
        for k in (72, 200, 256):
            ref = a[:, :k].double() @ b[:k].double()
            args = (a[:, :k].to(DEVICE), b[:k].to(DEVICE))
            for lanes in (8, 16, 32):
                with self.subTest(k=k, lanes=lanes):
                    out, _, msl = _run(cast_rhs, args, _k_lanes(lanes))
                    _close(out, ref)
                    self._assert_loads_first(msl, [("char", "_mpp_lanes_0", lanes)])
        config = helion.Config(
            block_sizes=[32, 16, 32], num_threads=[1, 1, 32], num_warps=1
        )
        for k in (64, 72):
            w = torch.randint(-8, 8, (k, 24))
            with self.subTest(kernel="int4_gemm", k=k):
                out, _, msl = _run(
                    int4_gemm,
                    (a[:, :k].to(DEVICE), _pack_int4(w).to(DEVICE)),
                    config,
                )
                _close(
                    out, (a[:, :k].double() @ w.double()).to(torch.bfloat16).double()
                )
                self._assert_loads_first(msl, [("char", "_mpp_lanes_0", 32)])

    def test_two_arrays(self) -> None:
        """Two int16 loads, and an int16 index with the bf16 rows it gathers,
        whose pointer reads the index load, fill the 64 bytes at 16 lanes."""
        a = torch.randn(5, 200).to(torch.bfloat16)
        w = torch.randint(-100, 100, (200, 24), dtype=torch.int16)
        z = torch.randint(-100, 100, (200, 24), dtype=torch.int16)
        table = torch.randn(400, 24).to(torch.bfloat16)
        rows = torch.randint(0, 400, (200,), dtype=torch.int16)
        for k in (72, 200):
            lhs = a[:, :k].double()
            for kernel, args, rhs, arrays in (
                (
                    zero_point_rhs,
                    (w[:k], z[:k]),
                    (w[:k] - z[:k]).double(),
                    [("short", "_mpp_lanes_0", 16), ("short", "_mpp_lanes_1", 16)],
                ),
                (
                    gathered_rhs,
                    (table, rows[:k]),
                    table[rows[:k].long()].double(),
                    [("short", "_mpp_lanes_0", 16), ("bfloat", "_mpp_lanes_1", 16)],
                ),
            ):
                with self.subTest(kernel=kernel.name, k=k):
                    out, _, msl = _run(
                        kernel,
                        (a[:, :k].to(DEVICE), *(t.to(DEVICE) for t in args)),
                        _k_lanes(16),
                    )
                    _close(out, lhs @ rhs)
                    self._assert_loads_first(msl, arrays)
                    if kernel is gathered_rhs:
                        # The gathered load's pointer reads the index load,
                        # which still runs once.
                        self.assertEqual(msl.count("rows["), 1)

    def test_taken_names(self) -> None:
        """A lane array's name skips one the kernel already uses."""
        a = torch.randn(5, 72).to(torch.bfloat16)
        b = torch.randint(-128, 128, (72, 24), dtype=torch.int8)
        z = torch.randint(-8, 8, (72, 24), dtype=torch.int8)
        out, _, msl = _run(
            lanes_named, (a.to(DEVICE), b.to(DEVICE), z.to(DEVICE)), _k_lanes(32)
        )
        _close(out, a.double() @ (b.double() - z.double()))
        self._assert_loads_first(
            msl, [("char", "_mpp_lanes_1", 32), ("char", "_mpp_lanes_2", 32)]
        )

    def test_whole_loops(self) -> None:
        """64 lanes, which Metal miscompiles in loads-first form when the K
        loop runs once; two int16 loads at 32 lanes, of which only the first
        fits the 64 bytes; and an int8 load beside a float32 one."""
        a = torch.randn(5, 128).to(torch.bfloat16)
        b = torch.randint(-128, 128, (128, 24), dtype=torch.int8)
        w = torch.randint(-100, 100, (128, 24), dtype=torch.int16)
        z = torch.randint(-100, 100, (128, 24), dtype=torch.int16)
        s = torch.randn(128, 24)
        cases = [
            (cast_rhs, (b[:k],), b[:k].double(), _k_lanes(64), k) for k in (72, 128)
        ]
        cases += [
            (zero_point_rhs, (w, z), (w - z).double(), _k_lanes(32), 128),
            (
                scaled_rhs,
                (b, s),
                (b.float() * s).to(torch.bfloat16).double(),
                _k_lanes(8),
                128,
            ),
        ]
        for kernel, args, rhs, config, k in cases:
            with self.subTest(kernel=kernel.name, k=k, config=config):
                out, _, msl = _run(
                    kernel,
                    (a[:, :k].to(DEVICE), *(t.to(DEVICE) for t in args)),
                    config,
                )
                _close(out, a[:, :k].double() @ rhs)
                self.assertNotIn("_mpp_lanes_", msl)

    def test_no_loads(self) -> None:
        """A staged operand computed from its indices alone keeps its one
        lane loop."""
        a = torch.randn(5, 72).to(torch.bfloat16)
        cols = torch.empty(1, 24, device=DEVICE)
        out, _, msl = _run(index_rhs, (a.to(DEVICE), cols), _k_lanes(16))
        _close(out, a.double() @ torch.arange(72.0).double()[:, None].expand(72, 24))
        self.assertEqual(msl.count("for (int lane_"), 1)

    def test_fp8_load(self) -> None:
        """An fp8 load, which has no Metal type, fails before any lane loop
        is rewritten."""
        a = torch.randn(5, 72, device=DEVICE).to(torch.bfloat16)
        b = torch.randn(72, 24, device=DEVICE).to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(
            ValueError, "Unsupported Metal dtype: torch.float8_e4m3fn"
        ):
            _run(cast_rhs, (a, b), _k_lanes(8))


@helion.kernel(**S)
def colsum_seeded(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Column sums of ``x @ y + 1``; M must fit one tile."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.full([tm, tn], 1.0, dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tn] = acc.sum(0)
    return out


@helion.kernel(**S)
def rowsum_seeded(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Row sums of ``x @ y + 1``; N must fit one tile."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.full([tm, tn], 1.0, dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm] = acc.sum(1)
    return out


@helion.kernel(**S, fast_math=True)
def colsum_fast(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Column sums of ``x @ y``, whose reduction mask fast math drops (an
    mm's masked value is 0); M must fit one tile."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tn] = torch.mm(x[tm, :], y[:, tn]).sum(0)
    return out


@helion.kernel(**S)
def seed_prod(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The column products of a 1.0 seed, whose reduction mask Helion drops
    (the fill is the product's identity), and the column sums of ``x @ y +
    1``; M must fit one tile."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([2, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.full([tm, tn], 1.0, dtype=torch.float32)
        out[0, tn] = torch.prod(acc, 0)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[1, tn] = acc.sum(0)
    return out


@helion.kernel(**S)
def seed_row_prod(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """:func:`seed_prod` over rows; N must fit one tile."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, 2], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.full([tm, tn], 1.0, dtype=torch.float32)
        out[tm, 0] = torch.prod(acc, 1)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, 1] = acc.sum(1)
    return out


@helion.kernel(**S)
def seeded_epilogue(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.full([tm, tn], 1.0, dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = torch.relu(acc) * 2
    return out


@helion.kernel(**S)
def m_block4(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n], block_size=[4, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


plain_g_dynamic = helion.kernel(plain_g.fn, backend="metal", static_shapes=False)


def _op_mn(msl: str) -> tuple[int, int]:
    """The M and N of the kernel's one MPP descriptor."""
    ((m, n),) = re.findall(r"matmul2d_descriptor\((\d+), (\d+), \d+,", msl)
    return int(m), int(n)


@_requires_mps
class TestMppOpDims(_MppCase):
    """When one tile covers an M or N loop with a static extent and nothing
    reads the cooperative value in the Scalar layout, the op's dim is that
    extent rounded up to 8, within the block: MPP computes every row of its
    descriptor, so the rest of the block would only be padding."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        ref: torch.Tensor,
        op: tuple[int, int],
    ) -> None:
        shapes = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
        with self.subTest(kernel=kernel.name, shapes=shapes, config=config, op=op):
            out, _, msl = _run(kernel, args, config)
            _close(out, ref)
            self.assertEqual(_op_mn(msl), op)

    def test_cut_dims(self) -> None:
        """An M or N extent under its block: M = 4 and 9 under 16- and
        32-row blocks, N = 4, both at 4 (the extra 8 goes to M), a
        zero-row loop, a 1.0 seed stored through an epilogue, and int4_gemm
        at M = 4, whose ``hl.dot`` keeps the M block at 16 or more."""
        t16 = helion.Config(block_sizes=[16, 16, 16])
        t32 = helion.Config(block_sizes=[32, 32, 16])
        for (m, n), config, op in (
            ((4, 40), t16, (8, 16)),
            ((4, 40), t32, (8, 32)),
            ((9, 40), t32, (16, 32)),
            ((16, 40), t32, (16, 32)),
            ((48, 4), t16, (16, 8)),
            ((4, 4), t16, (16, 8)),
            ((0, 40), t16, (8, 16)),
        ):
            x = torch.randn(m, 48, device=DEVICE, dtype=torch.float16)
            y = torch.randn(48, n, device=DEVICE, dtype=torch.float16)
            self._check(root_plain, (x, y), config, _mm(x, y), op)
        for (m, n), op in (((4, 40), (8, 16)), ((40, 5), (16, 8))):
            x = torch.randn(m, 48, device=DEVICE, dtype=torch.float16)
            y = torch.randn(48, n, device=DEVICE, dtype=torch.float16)
            ref = torch.relu(_mm(x, y) + 1) * 2
            self._check(seeded_epilogue, (x, y), t16, ref, op)
        bf16 = torch.bfloat16
        for (m, k, n), config, op in (
            ((4, 72, 24), None, (8, 16)),
            (
                (4, 72, 24),
                helion.Config(block_sizes=[32, 16, 16], num_threads=[0, 1, 16]),
                (8, 16),
            ),
            (
                (4, 128, 64),
                helion.Config(
                    block_sizes=[32, 16, 32], num_threads=[0, 1, 1024], num_warps=1
                ),
                (8, 32),
            ),
        ):
            w = torch.randint(-8, 8, (k, n))
            a = torch.randn(m, k).to(bf16)
            args = (_nan_padded(a.to(DEVICE)), _pack_int4(w).to(DEVICE))
            ref = (a.double() @ w.double()).to(bf16).double()
            self._check(int4_gemm, args, config, ref, op)

    def test_materializing_groups_keep_blocks(self) -> None:
        """A cooperative value read in the Scalar layout keeps the block's
        op, since a reduction may read the tile rows past the extent: column
        and row sums of a 1.0-seeded product (masked), of an mm under fast
        math (mask dropped), and products of the 1.0 seed itself (mask
        dropped, so a row past the op would read 0)."""
        # Each config's num_warps launches its tile's threads whole: in a
        # lane loop the column sums would reduce over a lane-strided axis.
        t16 = helion.Config(block_sizes=[16, 16, 16], num_warps=8)
        t32 = helion.Config(block_sizes=[32, 32, 16], num_warps=32)

        def plain_col_sums(p: torch.Tensor) -> torch.Tensor:
            return p.sum(0)

        def col_sums(p: torch.Tensor) -> torch.Tensor:
            return (p + 1).sum(0)

        def row_sums(p: torch.Tensor) -> torch.Tensor:
            return (p + 1).sum(1)

        def seed_cols(p: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(p[0]), col_sums(p)])

        def seed_rows(p: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(p[:, 0]), row_sums(p)], 1)

        def nt(blocks: list[int], threads: list[int]) -> helion.Config:
            launched = math.prod(t for t in threads if t)
            return helion.Config(
                block_sizes=blocks,
                num_threads=threads,
                num_warps=max(1, launched // 32),
            )

        for kernel, (m, n), config, ref, op in (
            (colsum_seeded, (4, 40), t16, col_sums, (16, 16)),
            (colsum_seeded, (9, 40), t32, col_sums, (32, 32)),
            (rowsum_seeded, (40, 5), t16, row_sums, (16, 16)),
            (colsum_fast, (4, 40), nt([16, 16], [16, 16]), plain_col_sums, (16, 16)),
            (seed_prod, (4, 40), nt([16, 16, 16], [16, 16, 0]), seed_cols, (16, 16)),
            (seed_prod, (9, 40), nt([32, 16, 16], [32, 16, 0]), seed_cols, (32, 16)),
            (seed_row_prod, (40, 5), nt([16, 16, 16], [1, 16, 0]), seed_rows, (16, 16)),
        ):
            x = torch.randn(m, 32, device=DEVICE)
            y = torch.randn(32, n, device=DEVICE)
            self._check(kernel, (x, y), config, ref(_mm(x, y)), op)
        # At the default num_warps the 16 x 16 tile runs in a lane loop, and
        # its column sum is rejected rather than miscompiled.
        x = torch.randn(4, 32, device=DEVICE)
        y = torch.randn(32, 40, device=DEVICE)
        self._rejects(
            colsum_seeded,
            (x, y),
            helion.Config(block_sizes=[16, 16, 16]),
            "sum reduction over an axis strided by a tile lane loop",
        )

    def test_unchanged_dims(self) -> None:
        """A block the extent fills, a loop of several tiles, symbolic
        extents, and a ':' M of 8 beside an N extent of 4, whose cut N takes
        the extra 8 MPP needs."""
        t16 = helion.Config(block_sizes=[16, 16, 16])
        for kernel, (m, n), config, op in (
            (root_plain, (16, 40), t16, (16, 16)),
            (root_plain, (40, 40), t16, (16, 16)),
            (plain_g_dynamic, (4, 40), TILE16, (16, 16)),
            (colon_m_kloop, (8, 4), helion.Config(block_sizes=[16, 16]), (8, 16)),
        ):
            x = torch.randn(m, 48, device=DEVICE, dtype=torch.float16)
            y = torch.randn(48, n, device=DEVICE, dtype=torch.float16)
            self._check(kernel, (x, y), config, _mm(x, y), op)


@_requires_mps
class TestMppOpDimsRejects(_MppCase):
    """A dim cut to its extent never grows past its block: a block under 8
    stays rejected."""

    def test_small_blocks(self) -> None:
        x = torch.randn(3, 32, device=DEVICE, dtype=torch.float16)
        y = torch.randn(32, 24, device=DEVICE, dtype=torch.float16)
        msg = "an MPP matmul tile 4x16x16"
        self._rejects(m_block4, (x, y), helion.Config(block_sizes=[16, 16]), msg)
        x = torch.randn(4, 32, device=DEVICE, dtype=torch.float16)
        self._rejects(colon_m_kloop, (x, y), helion.Config(block_sizes=[16, 16]), msg)


_SEED_CONFIGS = (None, helion.Config(block_sizes=[32, 32, 32]))


class TestMppDotSeedCodegen(unittest.TestCase):
    """The planner takes an ``hl.dot`` K loop with any seed: a constant fill,
    zero included, seeds its ``_helion_mpp_acc``, and a loaded seed is staged
    through ``_mpp_coop``.  Generated on the CPU."""

    def test_seeds(self) -> None:
        x = torch.randn(64, 64, dtype=torch.float16)
        y = torch.randn(64, 64, dtype=torch.float16)
        c = torch.randn(64, 64)
        for kernel, args, seed in (
            (dot_seed_zeros, (x, y), r"_helion_mpp_acc\(.*, 0\.0, \["),
            (dot_seed_full, (x, y), r"_helion_mpp_acc\(.*, 1\.5, \["),
            (
                dot_seed_load,
                (x, y, c),
                r"_mpp_coop = _helion_mpp_acc\(.*'_mpp_seed'\)",
            ),
        ):
            bound = kernel.bind(args)
            for config in _SEED_CONFIGS:
                with self.subTest(kernel=kernel.name, config=config):
                    code = bound.to_code(config or bound.config_spec.default_config())
                    body = "\n".join(
                        line
                        for line in code.splitlines()
                        if not line.lstrip().startswith("#")
                    )
                    self.assertIn("_helion_mpp_run(", body)
                    self.assertRegex(body, seed)


@_requires_mps
class TestMppDotSeed(unittest.TestCase):
    """A seeded ``hl.dot`` accumulator computes seed + x @ y."""

    def test_seeds(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        c = torch.randn(64, 64, device=DEVICE)
        mm = _mm(x, y)
        for kernel, args, ref in (
            (dot_seed_zeros, (x, y), mm),
            (dot_seed_full, (x, y), mm + 1.5),
            (dot_seed_load, (x, y, c), mm + c.cpu().double()),
        ):
            for config in _SEED_CONFIGS:
                with self.subTest(kernel=kernel.name, config=config):
                    out, _, _ = _run(kernel, args, config)
                    _close(out, ref)


# --- Whole-K hl.dot through the planner -----------------------------------


@helion.kernel(**S)
def dot_scaled_lhs(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``(x^T * exp(s)) @ w`` over a K loop: a transposed, scaled, cast lhs."""
    k, m = x.shape
    n = w.size(1)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            xt = (x[tk, tm].T.to(torch.float32) * torch.exp(s[tk])[None, :]).to(x.dtype)
            acc = hl.dot(xt, w[tk, tn], acc=acc)
        out[tm, tn] = acc.to(x.dtype)
    return out


@helion.kernel(**S)
def dot_two_stage(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """``(a @ b^T) * 0.5 + (2c) @ d``: a ``:`` K op seeds a K-loop op."""
    m, k1 = a.shape
    n = b.size(0)
    hl.specialize(k1)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = hl.dot(a[tm, :], b[tn, :].T, acc=acc)
        acc = acc * 0.5
        for tk in hl.tile(c.size(1)):
            acc = hl.dot(c[tm, tk] * 2.0, d[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_colon_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A computed lhs over a ``:`` K, with no acc."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        out[tm, tn] = hl.dot(x[tm, :] * 2.0, y[:, tn])
    return out


@helion.kernel(**S)
def dot_colon_m(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A computed lhs over a ``:`` M, so its Scalar work runs in M's lanes."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tn in hl.tile(y.size(1)):
        acc = hl.zeros([x.size(0), tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[:, tk] * 2.0, y[tk, tn], acc=acc)
        out[:, tn] = acc
    return out


@helion.kernel(**S)
def dot_colon_n(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A computed rhs over a ``:`` N."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, y.size(1)], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk], y[tk, :] * 2.0, acc=acc)
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def dot_colon_n_row_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A ``:`` N summed after the K loop."""
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, y.size(1)], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk] * 2.0, y[tk, :], acc=acc)
        out[tm] = acc.sum(-1)
    return out


@helion.kernel(**S)
def dot_both_computed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Both operands computed over a ``:`` K, so both are staged whole."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        out[tm, tn] = hl.dot(x[tm, :] * 2.0, y[:, tn] + 1.0)
    return out


@helion.kernel(**S)
def dot_beside_side_store(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    """Scalar work over another ``:`` dim beside the op: a store of ``w * 2``
    and a row sum the epilogue reads."""
    hl.specialize(x.size(1))
    hl.specialize(w.size(1))
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        side[tm, :] = w[tm, :] * 2.0
        s = w[tm, :].sum(-1)
        out[tm, tn] = hl.dot(x[tm, :], y[:, tn], out_dtype=torch.float32) + s[:, None]
    return out


@helion.kernel(**S)
def dot_after_addmm_tiles(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A ``torch.addmm`` K loop and an ``hl.dot`` K loop in two root loops."""
    m, n = a.size(0), b.size(1)
    o1 = torch.empty([m, n], dtype=torch.float32, device=a.device)
    o2 = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        o1[tm, tn] = acc
    for tm, tn in hl.tile([m, n]):
        acc2 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(c.size(1)):
            acc2 = hl.dot(c[tm, tk] * 2.0, d[tk, tn], acc=acc2)
        o2[tm, tn] = acc2
    return o1, o2


@helion.kernel(**S)
def dot_after_addmm_loop(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """A ``torch.addmm`` K loop, then an ``hl.dot`` K loop on the same acc."""
    m, n = a.size(0), b.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        for tk in hl.tile(c.size(1)):
            acc = hl.dot(c[tm, tk] * 2.0, d[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_if_seed(x: torch.Tensor, w: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    """A K loop whose seed is an ``if``'s output."""
    out = torch.empty([x.size(0), w.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), w.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        if flag[0] > 0:
            acc = acc + 1.0
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk], w[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_f16_acc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float16, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float16)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk] * 2.0, y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_bf16_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.bfloat16, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        out[tm, tn] = hl.dot(x[tm, :], y[:, tn], out_dtype=torch.bfloat16)
    return out


@helion.kernel(**S)
def dot_f16_acc_f32_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A float32 result over a float16 acc."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float16)
        out[tm, tn] = hl.dot(x[tm, :], y[:, tn], acc=acc, out_dtype=torch.float32)
    return out


@helion.kernel(**S)
def dot_plain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        out[tm, tn] = hl.dot(x[tm, :], y[:, tn])
    return out


@helion.kernel(**S)
def dot_mixed_dtypes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk] * 2.0, y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_of_dot(q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``exp(q k^T / 8) v``: one dot's result is the next dot's lhs."""
    d = hl.specialize(q.size(1))
    out = torch.empty([q.size(0), d], dtype=torch.float32, device=q.device)
    for tm in hl.tile(q.size(0)):
        acc = hl.zeros([tm, d], dtype=torch.float32)
        for tn in hl.tile(kt.size(1)):
            p = torch.exp(hl.dot(q[tm, :], kt[:, tn]) * 0.125)
            acc = hl.dot(p.to(v.dtype), v[tn, :], acc=acc)
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def dot_of_outer_dot(
    x: torch.Tensor, u: torch.Tensor, w: torch.Tensor, steps: torch.Tensor
) -> torch.Tensor:
    """``p = x @ u`` before a loop whose dot reads ``p`` each step and
    decays its acc."""
    hl.specialize(u.size(1))
    out = torch.empty([x.size(0), w.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), w.size(1)]):
        p = hl.dot(x[tm, :], u[:, :])
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for _ in hl.tile(steps.size(0)):
            acc = hl.dot(p.to(w.dtype), w[:, tn], acc=acc * 0.5)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_of_carried_dot(
    q: torch.Tensor, qt: torch.Tensor, v: torch.Tensor, x0: torch.Tensor
) -> torch.Tensor:
    """Per chunk ``q_c @ s``, then ``s += q_c^T v_c``: the first dot's rhs is
    a carry whose carried-back value is the second dot's result."""
    hl.specialize(q.size(1))
    out = torch.empty([q.size(0), v.size(1)], dtype=torch.float32, device=q.device)
    for tn in hl.tile(v.size(1)):
        s = x0[:, tn].to(torch.float32)
        for tc in hl.tile(q.size(0), block_size=16):
            out[tc, tn] = hl.dot(q[tc, :], s.to(q.dtype))
            s = hl.dot(qt[:, tc], v[tc, tn], acc=s)
    return out


@helion.kernel(**S)
def dot_of_loop_dot(x: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``(x @ u) @ w``: a K loop's result is the second dot's lhs."""
    d = hl.specialize(u.size(1))
    out = torch.empty([x.size(0), w.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), w.size(1)]):
        acc = hl.zeros([tm, d], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk], u[tk, :], acc=acc)
        out[tm, tn] = hl.dot(acc.to(w.dtype), w[:, tn])
    return out


@helion.kernel(**S)
def dot_no_acc_tiled_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Each K tile's product alone: the last tile's is stored."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1), block_size=16):
            acc = hl.dot(x[tm, tk], y[tk, tn], out_dtype=torch.float32)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_beside_bf16_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """An admissible dot beside one with a bfloat16 result."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        a = hl.dot(x[tm, :], y[:, tn], out_dtype=torch.float32)
        b = hl.dot(x[tm, :], y[:, tn], out_dtype=torch.bfloat16)
        out[tm, tn] = a + b.to(torch.float32)
    return out


@helion.kernel(**S)
def dot_scaled_by_arg(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """A lhs scaled by a float kernel argument."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk] * alpha, y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_staged_both(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Both operands computed, so both are staged."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(torch.relu(x[tm, tk]), y[tk, tn] * 0.5, acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_views_seeded(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Device views over a K loop seeded by a fill of 1."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.full([tm, tn], 1.0, dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_fixed_m(x: torch.Tensor, y: torch.Tensor, bm: torch.Tensor) -> torch.Tensor:
    """An M tile fixed to ``bm``'s length, over a staged lhs."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)], block_size=[bm.size(0), None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk] * 2.0, y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_fixed_n(x: torch.Tensor, y: torch.Tensor, bn: torch.Tensor) -> torch.Tensor:
    """An N tile fixed to ``bn``'s length, over a staged lhs."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)], block_size=[None, bn.size(0)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk] * 2.0, y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_fixed_k(x: torch.Tensor, y: torch.Tensor, bk: torch.Tensor) -> torch.Tensor:
    """A K tile fixed to ``bk``'s length, over a staged lhs."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1), block_size=bk.size(0)):
            acc = hl.dot(x[tm, tk] * 2.0, y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_result_scaled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The K loop carries the result scaled, not the result itself."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk] * 2.0, y[tk, tn], acc=acc) * 0.5
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_loaded_acc(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Each K tile's product added to a loaded acc, then summed."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = acc + hl.dot(x[tm, tk] * 2.0, y[tk, tn], acc=c[tm, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_update_read_twice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """An update of the carry whose intermediate value another carry reads."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        tot = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            t = acc * 0.5
            tot = tot + t
            acc = hl.dot(x[tm, tk] * 2.0, y[tk, tn], acc=t + 1.0)
        out[tm, tn] = acc + tot
    return out


@helion.kernel(**S)
def dot_update_by_arg(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """An update of the carry by a float kernel argument."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk] * 2.0, y[tk, tn], acc=acc * alpha)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_outer_decay(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """An outer loop that decays the K loop's carry between passes."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for _ in hl.grid(3):
            for tk in hl.tile(x.size(1)):
                acc = hl.dot(x[tm, tk] * 2.0, y[tk, tn], acc=acc)
            acc = acc * 0.5
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_colon_carried(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A ``:`` K op whose acc a loop carries."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for _ in hl.grid(3):
            acc = hl.dot(x[tm, :] * 2.0, y[:, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_shared_zero(x: torch.Tensor, y: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    """Two dots in one K loop whose accs start from one ``hl.zeros``."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        z = hl.zeros([tm, tn], dtype=torch.float32)
        a = z
        b = z
        for tk in hl.tile(x.size(1)):
            xs = x[tm, tk] * 2.0
            a = hl.dot(xs, y[tk, tn], acc=a)
            b = hl.dot(xs, y2[tk, tn], acc=b)
        out[tm, tn] = a * torch.sigmoid(b)
    return out


@helion.kernel(**S)
def dot_update_reads_output(
    x: torch.Tensor, y: torch.Tensor, o: torch.Tensor
) -> torch.Tensor:
    """An update of the carry that reads the tensor the kernel writes."""
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk] * 2.0, y[tk, tn], acc=acc + o[tm, tn])
        o[tm, tn] = acc
    return o


@helion.kernel(**S)
def dot_bf16_scale_before_loop(
    x: torch.Tensor, y: torch.Tensor, sv: torch.Tensor
) -> torch.Tensor:
    """A bfloat lhs computed in the K loop from a bfloat scale computed
    before it."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        sc = (sv[tm] * 2.0).to(torch.bfloat16)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = hl.dot(x[tm, tk] * sc[:, None], y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_flattened_root(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A ``:`` K op in a one-block root tile with no device loop inside."""
    n = hl.specialize(y.size(1))
    out = torch.empty([x.size(0), n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(x.size(0)):
        out[tm, :] = hl.dot(x[tm, :] * 2.0, y[:, :])
    return out


@helion.kernel(**S)
def dot_colon_decayed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A ``:`` K op whose acc a loop carries and decays."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for _ in hl.grid(3):
            acc = hl.dot(x[tm, :] * 2.0, y[:, tn], acc=acc * 0.5)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def lin_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention without a softmax: the first dot's scaled, rounded result
    is the second dot's lhs."""
    m, n, e = q.size(0), k.size(0), v.size(1)
    out = torch.empty([m, e], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, e], dtype=torch.float32)
        for tn in hl.tile(n):
            s = hl.dot(q[tm, :], k[tn, :].T)
            p = (s * 0.125).to(torch.bfloat16)
            acc = hl.dot(p, v[tn, :], acc=acc)
        out[tm, :] = acc
    return out


def lin_attn_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qd, kd, vd = _cpu64(q, k, v)
    return ((qd @ kd.T) * 0.125).to(torch.bfloat16).double() @ vd


@helion.kernel(**S)
def lin_attn_two_consumers(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``lin_attn``'s lhs read by two dots."""
    m, n, e = q.size(0), k.size(0), v.size(1)
    out_v = torch.empty([m, e], dtype=torch.float32, device=q.device)
    out_w = torch.empty([m, e], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        acc_v = hl.zeros([tm, e], dtype=torch.float32)
        acc_w = hl.zeros([tm, e], dtype=torch.float32)
        for tn in hl.tile(n):
            p = (hl.dot(q[tm, :], k[tn, :].T) * 0.125).to(torch.bfloat16)
            acc_v = hl.dot(p, v[tn, :], acc=acc_v)
            acc_w = hl.dot(p, w[tn, :], acc=acc_w)
        out_v[tm, :] = acc_v
        out_w[tm, :] = acc_w
    return out_v, out_w


@helion.kernel(**S)
def lin_attn_loaded_lhs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The second dot's lhs lies over the first op's M and N but is computed
    from a load alone."""
    m, n, e = q.size(0), k.size(0), v.size(1)
    out = torch.empty([m, e], dtype=torch.float32, device=q.device)
    scores = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, e], dtype=torch.float32)
        for tn in hl.tile(n):
            scores[tm, tn] = hl.dot(q[tm, :], k[tn, :].T)
            acc = hl.dot((x[tm, tn] * 0.5).to(torch.bfloat16), v[tn, :], acc=acc)
        out[tm, :] = acc
    return out, scores


@helion.kernel(**S)
def lin_attn_row_sums(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``lin_attn`` that also stores each chunk's row sums of the lhs, a use
    of it in the Scalar layout."""
    m, n, e = q.size(0), k.size(0), v.size(1)
    out = torch.empty([m, e], dtype=torch.float32, device=q.device)
    sums = torch.empty([m, n // 16], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, e], dtype=torch.float32)
        for tn in hl.tile(n, block_size=16):
            p = (hl.dot(q[tm, :], k[tn, :].T) * 0.125).to(torch.bfloat16)
            sums[tm, tn.id] = p.to(torch.float32).sum(-1)
            acc = hl.dot(p, v[tn, :], acc=acc)
        out[tm, :] = acc
    return out, sums


@helion.kernel(**S)
def lin_attn_normalized(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``lin_attn`` with its rows normalized after the loop."""
    m, n, e = q.size(0), k.size(0), v.size(1)
    out = torch.empty([m, e], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, e], dtype=torch.float32)
        for tn in hl.tile(n):
            p = (hl.dot(q[tm, :], k[tn, :].T) * 0.125).to(torch.bfloat16)
            acc = hl.dot(p, v[tn, :], acc=acc)
        out[tm, :] = acc / acc.sum(-1, keepdim=True)
    return out


@helion.kernel(**S)
def lin_attn_side_rowmax(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``lin_attn`` that also stores the first dot's result less its row
    max, a reduction over the first op's lanes beside the second op."""
    m, n, e = q.size(0), k.size(0), v.size(1)
    out = torch.empty([m, e], dtype=torch.float32, device=q.device)
    side = torch.empty([m, n], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, e], dtype=torch.float32)
        for tn in hl.tile(n):
            s = hl.dot(q[tm, :], k[tn, :].T)
            side[tm, tn] = s - torch.amax(s, dim=-1, keepdim=True)
            acc = hl.dot((s * 0.125).to(torch.bfloat16), v[tn, :], acc=acc)
        out[tm, :] = acc
    return out, side


@helion.kernel(**S)
def lin_attn_reloaded(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """The second dot's lhs re-reads ``c``, which the kernel writes."""
    m, n, e = q.size(0), k.size(0), v.size(1)
    out = torch.empty([m, e], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, e], dtype=torch.float32)
        for tn in hl.tile(n):
            s = hl.dot(q[tm, :], k[tn, :].T)
            c[tm, tn] = s
            acc = hl.dot((s * c[tm, tn]).to(torch.bfloat16), v[tn, :], acc=acc)
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def lin_attn_transposed(
    q: torch.Tensor, k: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """The second dot reads the first's result transposed."""
    m, n, e = q.size(0), k.size(0), w.size(1)
    out = torch.empty([n, e], dtype=torch.float32, device=q.device)
    for tn in hl.tile(n):
        acc = hl.zeros([tn, e], dtype=torch.float32)
        for tm in hl.tile(m):
            s = hl.dot(q[tm, :], k[tn, :].T)
            acc = hl.dot(s.to(torch.bfloat16).T, w[tm, :], acc=acc)
        out[tn, :] = acc
    return out


@helion.kernel(**S)
def carry_stored_each_step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A K loop's carry stored at each step, before the dot updates it."""
    m, k = x.shape
    n = y.size(1)
    snap = torch.empty([k // 16, m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k, block_size=16):
            snap[tk.id, tm, tn] = acc
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
    return snap


@helion.kernel(**S)
def state_scan(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """Per 32-step chunk ``o = q @ bf16(S)``, then ``S = S * exp(g) + k^T v``:
    a carried state read before its update."""
    t, d = q.shape
    e = v.size(1)
    out = torch.empty([t, e], dtype=torch.float32, device=q.device)
    for tile_e in hl.tile(e):
        state = hl.zeros([d, tile_e], dtype=torch.float32)
        for tt in hl.tile(t, block_size=32):
            out[tt, tile_e] = hl.dot(q[tt, :], state.to(torch.bfloat16))
            state = state * torch.exp(g[tt.begin])
            state = hl.dot(k[tt, :].T, v[tt, tile_e], acc=state)
    return out


@helion.kernel(**S)
def state_read_late(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """``state_scan`` whose first dot reads the state after the update and
    the second dot write it, through a value cast before them."""
    t, d = q.shape
    e = v.size(1)
    out = torch.empty([t, e], dtype=torch.float32, device=q.device)
    for tile_e in hl.tile(e):
        state = hl.zeros([d, tile_e], dtype=torch.float32)
        for tt in hl.tile(t, block_size=32):
            old = state.to(torch.bfloat16)
            state = state * torch.exp(g[tt.begin])
            state = hl.dot(k[tt, :].T, v[tt, tile_e], acc=state)
            out[tt, tile_e] = hl.dot(q[tt, :], old)
    return out


@helion.kernel(**S)
def state_read_after_decay(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """``state_scan`` whose first dot casts the state from before the decay
    after the decay, which updates it in place, though before the second
    dot."""
    t, d = q.shape
    e = v.size(1)
    out = torch.empty([t, e], dtype=torch.float32, device=q.device)
    for tile_e in hl.tile(e):
        state = hl.zeros([d, tile_e], dtype=torch.float32)
        for tt in hl.tile(t, block_size=32):
            old = state
            state = state * torch.exp(g[tt.begin])
            out[tt, tile_e] = hl.dot(q[tt, :], old.to(torch.bfloat16))
            state = hl.dot(k[tt, :].T, v[tt, tile_e], acc=state)
    return out


@helion.kernel(**S)
def carry_delta_scaled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A K loop's carry scaled before the dot, the scaled value read after
    it."""
    m, k = x.shape
    n = y.size(1)
    out = torch.empty([k // 16, m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k, block_size=16):
            twice = acc * 2.0
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
            out[tk.id, tm, tn] = acc - twice
    return out


@helion.kernel(**S)
def carry_stored_after_its_update(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A K loop's carry stored at each step after the dot updates it, from
    a copy taken before."""
    m, k = x.shape
    n = y.size(1)
    snap = torch.empty([k // 16, m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.full([tm, tn], 0.5, dtype=torch.float32)
        for tk in hl.tile(k, block_size=16):
            old = acc
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
            snap[tk.id, tm, tn] = old
    return snap


@helion.kernel(**S)
def lin_attn_reloaded_nested(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """The second dot's lhs re-reads ``c``, which the kernel writes, inside
    a function of it."""
    m, n, e = q.size(0), k.size(0), v.size(1)
    out = torch.empty([m, e], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, e], dtype=torch.float32)
        for tn in hl.tile(n):
            s = hl.dot(q[tm, :], k[tn, :].T)
            c[tm, tn] = s
            p = (s * torch.sigmoid(c[tm, tn])).to(torch.bfloat16)
            acc = hl.dot(p, v[tn, :], acc=acc)
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def lin_attn_reloaded_outer(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """The second dot's lhs re-reads ``c``, which the kernel writes outside
    the loop the dots run in."""
    m, n, e = q.size(0), k.size(0), v.size(1)
    out = torch.empty([m, e], dtype=torch.float32, device=q.device)
    for tm in hl.tile(m):
        c[tm, :] = c[tm, :] + 1.0
        acc = hl.zeros([tm, e], dtype=torch.float32)
        for tn in hl.tile(n):
            s = hl.dot(q[tm, :], k[tn, :].T)
            acc = hl.dot((s * c[tm, tn]).to(torch.bfloat16), v[tn, :], acc=acc)
        out[tm, :] = acc
    return out


@helion.kernel(**S)
def lin_attn_rhs(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The first dot, over the K loop's block and the output's columns, is
    the second dot's rhs; the output's rows are none of its blocks."""
    m, k = x.shape
    n = b.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            s = hl.dot(a[tk, :], b[:, tn])
            acc = hl.dot(x[tm, tk], (s * 0.5).to(torch.bfloat16), acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def lin_attn_rhs_nested(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, tag: torch.Tensor
) -> torch.Tensor:
    """``lin_attn_rhs`` with its output tile a device loop inside a grid."""
    m, k = x.shape
    n = b.size(1)
    out = torch.empty([tag.size(0), m, n], dtype=torch.float32, device=x.device)
    for tb in hl.tile(tag.size(0), block_size=1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                s = hl.dot(a[tk, :], b[:, tn])
                acc = hl.dot(x[tm, tk], (s * 0.5).to(torch.bfloat16), acc=acc)
            out[tb, tm, tn] = acc[None, :, :]
    return out


def lin_attn_rhs_ref(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    xd, ad, bd = _cpu64(x, a, b)
    return xd @ _bf(0.5 * (ad @ bd))


@helion.kernel(**S)
def lin_attn_column_tiles(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``lin_attn`` with the output's columns tiled, a block of the second
    dot only."""
    m, n, e = q.size(0), k.size(0), v.size(1)
    out = torch.empty([m, e], dtype=torch.float32, device=q.device)
    for tm, tile_e in hl.tile([m, e]):
        acc = hl.zeros([tm, tile_e], dtype=torch.float32)
        for tn in hl.tile(n):
            s = hl.dot(q[tm, :], k[tn, :].T)
            acc = hl.dot((s * 0.125).to(torch.bfloat16), v[tn, tile_e], acc=acc)
        out[tm, tile_e] = acc
    return out


@helion.kernel(**S)
def lin_attn_sibling_loop(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``lin_attn`` beside a loop over a block that no op runs in."""
    m, n, e = q.size(0), k.size(0), v.size(1)
    out = torch.empty([m, e], dtype=torch.float32, device=q.device)
    twice = torch.empty_like(x)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, e], dtype=torch.float32)
        for tn in hl.tile(n):
            s = hl.dot(q[tm, :], k[tn, :].T)
            acc = hl.dot((s * 0.125).to(torch.bfloat16), v[tn, :], acc=acc)
        out[tm, :] = acc
        for tj in hl.tile(x.size(1)):
            twice[tm, tj] = x[tm, tj] * 2
    return out, twice


@helion.kernel(**S)
def dot_side_block(x: torch.Tensor, y: torch.Tensor, tag: torch.Tensor) -> torch.Tensor:
    """A ``:`` K dot beside a grid block it doesn't use."""
    m, n = x.size(0), y.size(1)
    out = torch.empty([tag.size(0), m, n], dtype=torch.float32, device=x.device)
    for tb, tm, tn in hl.tile([tag.size(0), m, n]):
        out[tb, tm, tn] = hl.dot(x[tm, :], y[:, tn])[None, :, :]
    return out


def state_scan_ref(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    qd, kd, vd, gd = _cpu64(q, k, v, g)
    state = torch.zeros(qd.size(1), vd.size(1), dtype=torch.float64)
    out = torch.empty(qd.size(0), vd.size(1), dtype=torch.float64)
    for c in range(0, qd.size(0), 32):
        rows = slice(c, c + 32)
        out[rows] = qd[rows] @ state.to(torch.bfloat16).double()
        state = state * torch.exp(gd[c]) + kd[rows].T @ vd[rows]
    return out


gdn_fwd_h = helion.kernel(helion_gdn_fwd_h.fn, **S)


def _gdn_inputs(t: int, dk: int, dv: int) -> tuple[object, ...]:
    """gdn_fwd_h's bfloat16 inputs for one batch and head, chunks of 64,
    with ``w`` orthonormal per chunk so the carried state stays bounded."""
    gen = torch.Generator().manual_seed(0)
    chunks = -(-t // 64)
    wu, _, wv = torch.linalg.svd(
        torch.randn(chunks, 64, dk, generator=gen), full_matrices=False
    )
    w = (wu @ wv).reshape(chunks * 64, dk)[:t]
    k, u = torch.randn(t, dk, generator=gen), torch.randn(t, dv, generator=gen)
    g = torch.cumsum(math.log(1 / 16) * torch.rand(t, generator=gen), dim=0)
    rows = [x.reshape(1, t, 1, -1).to(torch.bfloat16).to(DEVICE) for x in (k, w, u)]
    return (*rows, g.reshape(1, t, 1).to(DEVICE), 64)


def _gdn_states(*args: object) -> torch.Tensor:
    """gdn_fwd_h's float32 state at the start of each chunk, in float64,
    rounding to bfloat16 where the kernel does; a tail chunk takes its live
    rows only."""
    k, w, u, g = (a[0, :, 0].cpu().double() for a in args[:4])

    def rounded(x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.bfloat16).double()

    state = torch.zeros(k.size(1), u.size(1), dtype=torch.float64)
    out = []
    for c in range(0, k.size(0), 64):
        rows = slice(c, c + 64)
        out.append(state)
        gc = g[rows]
        new_v = (u[rows] - w[rows] @ rounded(state)) * torch.exp(gc[-1] - gc)[:, None]
        state = state * torch.exp(gc[-1]) + k[rows].T @ rounded(new_v)
    return torch.stack(out)


def gdn_ref(*args: object) -> torch.Tensor:
    return _gdn_states(*args).to(torch.bfloat16).double()[None, :, None]


@helion.kernel(**S)
def gdn_fwd_h_column_sums(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """gdn_fwd_h that also stores the column sums of each chunk's starting
    state, a use of the carry that needs the Scalar layout."""
    batch, seqlen, nheads, dhead = k.shape
    dhead = hl.specialize(dhead)
    chunk_size = hl.specialize(chunk_size)
    dstate = u.shape[-1]
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    h = torch.empty(
        batch, nchunks, nheads, dhead, dstate, dtype=k.dtype, device=k.device
    )
    sums = torch.empty(
        batch, nchunks, nheads, dstate, dtype=torch.float32, device=k.device
    )
    block_v = hl.register_block_size(dstate)
    for tile_b, tile_h, tile_v in hl.tile(
        [batch, nheads, dstate], block_size=[1, 1, block_v]
    ):
        i_b = tile_b.id
        i_h = tile_h.id
        b_h = hl.zeros([dhead, tile_v], dtype=torch.float32)
        for t_i in hl.tile(seqlen, block_size=chunk_size):
            h[i_b, t_i.id, i_h, :, tile_v] = b_h.to(k.dtype)
            sums[i_b, t_i.id, i_h, tile_v] = b_h.sum(0)
            b_v = hl.dot(w[i_b, t_i, i_h, :], b_h.to(k.dtype), out_dtype=torch.float32)
            b_v = u[i_b, t_i, i_h, tile_v].to(torch.float32) - b_v
            t_i_last = min(t_i.begin + chunk_size, seqlen) - 1
            b_g_last = g[i_b, t_i_last, i_h].to(torch.float32)
            b_g = g[i_b, t_i, i_h].to(torch.float32)
            decay = torch.where(t_i.index < seqlen, torch.exp(b_g_last - b_g), 0)
            b_v *= decay[:, None]
            b_h *= torch.exp(b_g_last)
            b_h = hl.dot(k[i_b, t_i, i_h, :].T, b_v.to(k.dtype), acc=b_h)
    return h, sums


mamba2_chunk_scan = helion.kernel(helion_mamba2_chunk_scan_kernel.fn, **S)
mamba2_chunk_state = helion.kernel(helion_mamba2_chunk_state_kernel.fn, **S)


def _randn(*shape: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return torch.randn(*shape).to(dtype).to(DEVICE)


def _decays(*shape: int) -> torch.Tensor:
    return torch.cumsum(-torch.rand(*shape), dim=-1).half().to(DEVICE)


def _scan_args(
    b: int, nc: int, chunk: int, h: int, p: int, n: int
) -> tuple[torch.Tensor, ...]:
    seq = nc * chunk
    return (
        _randn(b, nc, 1, chunk, chunk),
        _randn(b, seq, h, p),
        torch.rand(b, h, nc, chunk).half().to(DEVICE),
        _decays(b, h, nc, chunk),
        _randn(b, seq, 1, n),
        _randn(b, nc, h, p, n),
        _randn(h),
    )


def _state_args(
    b: int, nc: int, chunk: int, h: int, p: int, n: int
) -> tuple[torch.Tensor, ...]:
    seq = nc * chunk
    return (
        torch.rand(b, seq, 1, n).half().to(DEVICE),
        torch.rand(b, seq, h, p).half().to(DEVICE),
        torch.rand(b, h, nc, chunk).half().to(DEVICE),
        _decays(b, h, nc, chunk),
    )


def _cpu64(*args: torch.Tensor) -> list[torch.Tensor]:
    return [a.cpu().double() for a in args]


def _ops(msl: str) -> int:
    return msl.count("matmul2d_descriptor(")


def _inputs(args: tuple[object, ...]) -> str:
    """The tensor arguments' shapes and dtypes, to tell subtests apart."""
    return ", ".join(
        f"{list(a.shape)} {a.dtype}" for a in args if isinstance(a, torch.Tensor)
    )


class _WholeKDotCase(_MppCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def _mpp(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        ref: torch.Tensor,
        ops: int = 1,
    ) -> tuple[str, str]:
        code = msl = ""
        with self.subTest(kernel=kernel.name, config=config, inputs=_inputs(args)):
            out, code, msl = _run(kernel, args, config)
            self.assertIn("_helion_mpp_run(", code)
            self.assertNotIn("dot_product", code)
            self.assertEqual(_ops(msl), ops)
            if isinstance(out, tuple):
                out = torch.cat([o.reshape(-1) for o in out])
            _close(out, ref)
        return code, msl

    def _scalar(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        ref: torch.Tensor,
    ) -> None:
        with self.subTest(kernel=kernel.name, config=config, inputs=_inputs(args)):
            out, code, _ = _run(kernel, args, config)
            self.assertIn("dot_product", code)
            self.assertNotIn("_helion_mpp_run(", code)
            if isinstance(out, tuple):
                out = torch.cat([o.reshape(-1) for o in out])
            _close(out, ref)


@_requires_mps
class TestMppWholeKDot(_WholeKDotCase):
    """An ``hl.dot`` over a whole K (a tile K loop or a ``:`` slice) is
    planned like ``torch.addmm``, or ``torch.mm`` with no acc, when the
    planner takes every such ``hl.dot`` of the kernel: none rounds its
    product to another ``out_dtype``, an operand computed from a matmul's
    result is staged from that matmul's element loop
    (``TestMppElementStaging``), each dim whose tile no config changes fits
    one op, each has an acc unless K is a ``:`` slice, and planning them
    raises nothing."""

    def test_staged_k_loop(self) -> None:
        """A transposed, scaled, cast lhs is staged per K tile; M = 24 and
        K = 72 have tails."""
        for dtype in (torch.float16, torch.bfloat16):
            x, w = _randn(72, 24, dtype=dtype), _randn(72, 40, dtype=dtype)
            s = torch.randn(72, device=DEVICE) * 0.1
            xd, wd, sd = _cpu64(x, w, s)
            # The kernel rounds the scaled lhs to the operand dtype.
            xt = (xd.T * sd.exp()[None, :]).to(dtype).double()
            for config in (None, helion.Config(block_sizes=[32, 16, 16])):
                self._mpp(dot_scaled_lhs, (x, w, s), config, xt @ wd)

    def test_colon_k_op_seeds_k_loop(self) -> None:
        """A ``:`` K op over device views, one transposed, seeds a K-loop op
        over a staged lhs."""
        a, b = _randn(40, 32), _randn(56, 32)
        c, d = _randn(40, 72), _randn(72, 56)
        ad, bd, cd, dd = _cpu64(a, b, c, d)
        ref = (ad @ bd.T) * 0.5 + (2 * cd) @ dd
        for config in (None, helion.Config(block_sizes=[32, 32, 32])):
            self._mpp(dot_two_stage, (a, b, c, d), config, ref, ops=2)

    def test_colon_dims_within_one_op(self) -> None:
        """A ``:`` K of 9 rounds up to 16 and one of 256 is the table's
        largest; a ``:`` M or N of 8 is the table's smallest, and one of 64
        the most lanes the jam holds."""
        for k in (9, 256):
            x, y = _randn(40, k), _randn(k, 24)
            xd, yd = _cpu64(x, y)
            self._mpp(dot_colon_k, (x, y), None, (2 * xd) @ yd)
        for length in (8, 64):
            x, y = _randn(length, 48), _randn(48, 40)
            xd, yd = _cpu64(x, y)
            self._mpp(dot_colon_m, (x, y), None, (2 * xd) @ yd)
            x, y = _randn(40, 48), _randn(48, length)
            xd, yd = _cpu64(x, y)
            self._mpp(dot_colon_n, (x, y), None, xd @ (2 * yd))

    def test_replica_threads_on_a_side_dim(self) -> None:
        """Threads on a ``:`` slice that none of the group's values has as a
        dim hold the same element of each: a side store and a row sum."""
        x, y = _randn(48, 64, dtype=torch.float32), _randn(64, 40, dtype=torch.float32)
        w = _randn(48, 24, dtype=torch.float32)
        xd, yd, wd = _cpu64(x, y, w)
        ref = torch.cat(
            [(xd @ yd + wd.sum(-1, keepdim=True)).reshape(-1), (2 * wd).reshape(-1)]
        )
        for config in (None, helion.Config(block_sizes=[16, 16])):
            with self.subTest(config=config):
                side = torch.zeros(48, 24, device=DEVICE)
                out, code, msl = _run(dot_beside_side_store, (x, y, w, side), config)
                self.assertIn("_helion_mpp_run(", code)
                self.assertEqual(_ops(msl), 1)
                _close(torch.cat([out.reshape(-1), side.reshape(-1)]), ref)

    def test_beside_addmm(self) -> None:
        """An ``hl.dot`` group beside a ``torch.addmm`` group, in another
        root loop or on the same acc."""
        a, b = _randn(48, 64), _randn(64, 40)
        c, d = _randn(48, 32), _randn(32, 40)
        ad, bd, cd, dd = _cpu64(a, b, c, d)
        both = torch.cat([(ad @ bd).reshape(-1), ((2 * cd) @ dd).reshape(-1)])
        self._mpp(dot_after_addmm_tiles, (a, b, c, d), None, both, ops=2)
        self._mpp(
            dot_after_addmm_loop, (a, b, c, d), None, ad @ bd + (2 * cd) @ dd, ops=2
        )

    def test_if_seed(self) -> None:
        x, w = _randn(64, 32), _randn(32, 48)
        xd, wd = _cpu64(x, w)
        flag = torch.ones(1, device=DEVICE)
        for config in (None, helion.Config(block_sizes=[32, 32, 16])):
            self._mpp(dot_if_seed, (x, w, flag), config, xd @ wd + 1)

    def test_staging_budget(self) -> None:
        """Both float32 operands staged over a ``:`` K of 256 fit threadgroup
        memory at 16 x 16 tiles, not at the default 32 x 32."""
        x = _randn(64, 256, dtype=torch.float32)
        y = _randn(256, 64, dtype=torch.float32)
        xd, yd = _cpu64(x, y)
        self._mpp(
            dot_both_computed,
            (x, y),
            helion.Config(block_sizes=[16, 16]),
            (2 * xd) @ (yd + 1),
        )
        self._rejects(
            dot_both_computed,
            (x, y),
            None,
            "reduction scratch and MPP staging need 65536 bytes of threadgroup memory",
        )

    def test_float_kernel_argument(self) -> None:
        """A float kernel argument in a staged operand has no block."""
        x, y = _randn(40, 72), _randn(72, 48)
        xd, yd = _cpu64(x, y)
        for config in (None, helion.Config(block_sizes=[32, 16, 16])):
            self._mpp(dot_scaled_by_arg, (x, y, 0.5), config, (0.5 * xd) @ yd)

    def test_operand_computed_from_a_matmul(self) -> None:
        """A dot's result is the next dot's lhs, and a carry whose
        carried-back value is the next dot's result is a dot's rhs."""
        q, kt, v = _randn(32, 32) / 4, _randn(32, 48) / 4, _randn(48, 32)
        qd, ktd, vd = _cpu64(q, kt, v)
        p = torch.exp(qd @ ktd * 0.125).half().double()
        self._mpp(dot_of_dot, (q, kt, v), None, p @ vd, ops=2)
        q, qt, v = _randn(64, 16), _randn(16, 64), _randn(64, 32)
        x0 = _randn(16, 32, dtype=torch.float32)
        qd, qtd, vd, s = _cpu64(q, qt, v, x0)
        chunks = []
        for c in range(0, 64, 16):
            chunks.append(qd[c : c + 16] @ s.half().double())
            s = s + qtd[:, c : c + 16] @ vd[c : c + 16]
        self._mpp(dot_of_carried_dot, (q, qt, v, x0), None, torch.cat(chunks), ops=2)

    def test_tile_of_one_over_views(self) -> None:
        """An M or N tile that every config makes 1 widens to 8 over device
        views."""
        for x, y in ((_randn(1, 64), _randn(64, 48)), (_randn(40, 64), _randn(64, 1))):
            xd, yd = _cpu64(x, y)
            self._mpp(dot_views_seeded, (x, y), None, 1 + xd @ yd)

    def test_fixed_tiles_within_one_op(self) -> None:
        """Fixed M and N tiles of 8 and 256 and K tiles of 16 and 256, the
        table's bounds."""
        x, y = _randn(300, 64), _randn(64, 300)
        xd, yd = _cpu64(x, y)
        for size in (8, 256):
            fixed = torch.empty(size, device=DEVICE)
            self._mpp(dot_fixed_m, (x, y, fixed), None, (2 * xd) @ yd)
            self._mpp(dot_fixed_n, (x, y, fixed), None, (2 * xd) @ yd)
        x, y = _randn(40, 512), _randn(512, 48)
        xd, yd = _cpu64(x, y)
        for size in (16, 256):
            fixed = torch.empty(size, device=DEVICE)
            self._mpp(dot_fixed_k, (x, y, fixed), None, (2 * xd) @ yd)

    def test_mamba2_chunk_scan(self) -> None:
        """A ``:`` K op seeding a causal K loop that ends at the M tile's end,
        at a small shape with tails."""
        args = _scan_args(2, 3, 64, 3, 24, 32)
        ref = ref_chunk_scan(*_cpu64(*args))
        for config in (None, helion.Config(block_sizes=[32, 32, 64])):
            code, _ = self._mpp(mamba2_chunk_scan, args, config, ref, ops=2)
            with self.subTest(config=config, owner="preamble"):
                # Owner tests are thread-only, so they are assigned before
                # the first loop, outside every lane nest.
                body = code[code.index("def _helion_") :]
                preamble = body[: body.index("\n    for ")]
                self.assertRegex(
                    preamble, r"_mpp_owner\w* = tid\[0\] == 0 and tid\[2\]"
                )

    def test_mamba2_chunk_scan_dstate_equal_to_headdim(self) -> None:
        """At the default config dstate's ':' block holds threads around the K
        loop's matmul, whose operands load x: x's host dim of dstate's length
        is no block of theirs."""
        args = _scan_args(1, 2, 64, 2, 16, 16)
        self._mpp(mamba2_chunk_scan, args, None, ref_chunk_scan(*_cpu64(*args)), ops=2)

    def test_mamba2_chunk_state(self) -> None:
        """A staged, transposed, decayed lhs, at a small shape with tails."""
        args = _state_args(2, 3, 48, 3, 24, 40)
        ref = ref_chunk_state(*_cpu64(*args))
        for config in (None, helion.Config(block_sizes=[32, 32, 32])):
            self._mpp(mamba2_chunk_state, args, config, ref)


@_requires_mps
class TestMppWholeKDotRejects(_WholeKDotCase):
    """Every whole-K ``hl.dot`` of a kernel with one the planner can't take
    keeps the scalar contraction."""

    def test_result_and_operand_dtypes(self) -> None:
        x, y = _randn(32, 48), _randn(48, 40)
        xd, yd = _cpu64(x, y)
        self._scalar(dot_f16_acc, (x, y), None, (2 * xd) @ yd)
        xb, yb = (
            _randn(32, 48, dtype=torch.bfloat16),
            _randn(48, 48, dtype=torch.bfloat16),
        )
        self._scalar(dot_bf16_out, (xb, yb), None, _mm(xb, yb))
        y32 = _randn(48, 40, dtype=torch.float32)
        self._scalar(dot_mixed_dtypes, (x, y32), None, (2 * xd) @ y32.cpu().double())

    def test_scalar_rejections_keep_their_messages(self) -> None:
        """A float16 acc under a float32 result, float8 operands and a dim
        that is no block, which the scalar contraction rejects with its own
        message."""
        x, y = torch.randn(32, 48).half(), torch.randn(48, 40).half()
        bound = dot_f16_acc_f32_out.bind((x, y))
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "pass an out_dtype equal to the acc's dtype"
        ):
            bound.to_code(bound.config_spec.default_config())
        f8 = torch.float8_e4m3fn
        bound = dot_plain.bind((x.to(f8), y.to(f8)))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "outside an MPP region; only float16, bfloat16 and float32",
        ):
            bound.to_code(bound.config_spec.default_config())
        bound = dot_constant_rows.bind((torch.randn(16, 24),))
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "hl.dot dimension 8 is not a tile or a `:` slice"
        ):
            bound.to_code(bound.config_spec.default_config())

    def test_operand_computed_from_a_matmul(self) -> None:
        """A matmul's result that reaches an operand other than from that
        matmul's element loop: from the enclosing graph and through a loop's
        output."""
        x, u, w = _randn(32, 16), _randn(16, 32), _randn(32, 16)
        xd, ud, wd = _cpu64(x, u, w)
        self._scalar(
            dot_of_outer_dot,
            (x, u, w, torch.zeros(4, device=DEVICE)),
            helion.Config(block_sizes=[16, 16, 4], num_warps=32),
            1.875 * ((xd @ ud).half().double() @ wd),
        )
        x, u, w = _randn(48, 64), _randn(64, 32), _randn(32, 40)
        xd, ud, wd = _cpu64(x, u, w)
        # With its output columns on one thread the thread rule doesn't
        # apply, so only the one-graph rule keeps it scalar.
        one = helion.Config(
            block_sizes=[16, 16, 16], num_threads=[0, 1, 0], num_warps=4
        )
        for config in (None, one):
            ref = (xd @ ud).half().double() @ wd
            self._scalar(dot_of_loop_dot, (x, u, w), config, ref)

    def test_colon_dims_past_one_op(self) -> None:
        """A ``:`` K of 8 is under the table's 16 and one of 257 over its
        256; a ``:`` M or N of 4 is under the table's 8, and one of 65 past
        the 64 lanes the jam holds, even where a row sum contracts it."""
        for k in (8, 257):
            x, y = _randn(40, k), _randn(k, 24)
            xd, yd = _cpu64(x, y)
            self._scalar(dot_colon_k, (x, y), None, (2 * xd) @ yd)
        for length in (4, 65):
            x, y = _randn(length, 48), _randn(48, 40)
            xd, yd = _cpu64(x, y)
            self._scalar(dot_colon_m, (x, y), None, (2 * xd) @ yd)
            x, y = _randn(40, 48), _randn(48, length)
            xd, yd = _cpu64(x, y)
            self._scalar(dot_colon_n, (x, y), None, xd @ (2 * yd))
        x, y = _randn(48, 64), _randn(64, 128)
        xd, yd = _cpu64(x, y)
        self._scalar(dot_colon_n_row_sums, (x, y), None, ((2 * xd) @ yd).sum(-1))

    def test_no_acc_over_a_tiled_k(self) -> None:
        x, y = _randn(32, 64, dtype=torch.float32), _randn(64, 48, dtype=torch.float32)
        xd, yd = _cpu64(x, y)
        self._scalar(dot_no_acc_tiled_k, (x, y), None, xd[:, 48:] @ yd[48:])

    def test_one_inadmissible_dot_keeps_every_dot(self) -> None:
        x, y = (
            _randn(32, 48, dtype=torch.bfloat16),
            _randn(48, 48, dtype=torch.bfloat16),
        )
        mm = _mm(x, y)
        self._scalar(dot_beside_bf16_dot, (x, y), None, mm + mm.bfloat16().double())

    def test_planner_rejections(self) -> None:
        """A kernel the planner rejects with its whole-K dots, one per kind
        of rejection: the K loop's carry, the carry's update, an enclosing
        loop, two groups on one value, memory order and bfloat staging."""
        x, y, y2 = _randn(40, 64), _randn(64, 48), _randn(64, 48)
        c = _randn(40, 48, dtype=torch.float32)
        xd, yd, y2d, cd = _cpu64(x, y, y2, c)
        tiles = [((2 * xd[:, k : k + 16]) @ yd[k : k + 16]) for k in range(0, 64, 16)]
        k16 = helion.Config(block_sizes=[16, 16, 16])
        scaled = torch.zeros(40, 48, dtype=torch.float64)
        update = torch.zeros(40, 48, dtype=torch.float64)
        tot = torch.zeros(40, 48, dtype=torch.float64)
        for p in tiles:
            scaled = (scaled + p) * 0.5
            tot = tot + update * 0.5
            update = update * 0.5 + 1 + p
        self._scalar(dot_result_scaled, (x, y), k16, scaled)
        self._scalar(dot_loaded_acc, (x, y, c), k16, 4 * cd + 2 * xd @ yd)
        self._scalar(dot_update_read_twice, (x, y), k16, update + tot)
        self._scalar(dot_update_by_arg, (x, y, 1.0), k16, 2 * xd @ yd)
        self._scalar(dot_outer_decay, (x, y), None, 0.875 * 2 * xd @ yd)
        self._scalar(dot_colon_carried, (x, y), None, 6 * xd @ yd)
        self._scalar(
            dot_shared_zero,
            (x, y, y2),
            None,
            (2 * xd @ yd) * torch.sigmoid(2 * xd @ y2d),
        )
        o = torch.zeros(40, 48, device=DEVICE)
        self._scalar(dot_update_reads_output, (x, y, o), None, 2 * xd @ yd)
        xb, yb = (
            _randn(40, 64, dtype=torch.bfloat16),
            _randn(64, 48, dtype=torch.bfloat16),
        )
        sv = _randn(40, dtype=torch.float32)
        sc = (sv.cpu() * 2).to(torch.bfloat16)
        lhs = (xb.cpu() * sc[:, None]).double()
        self._scalar(
            dot_bf16_scale_before_loop, (xb, yb, sv), None, lhs @ _cpu64(yb)[0]
        )

    def test_staged_operand_over_a_tile_of_one(self) -> None:
        """An operand staged over an M or N tile that every config makes 1,
        over a K loop and over a ``:`` K."""
        for x, y in ((_randn(1, 64), _randn(64, 48)), (_randn(40, 64), _randn(64, 1))):
            xd, yd = _cpu64(x, y)
            self._scalar(dot_staged_both, (x, y), None, xd.relu() @ (0.5 * yd))
        x, y = _randn(1, 64), _randn(64, 48)
        xd, yd = _cpu64(x, y)
        self._scalar(dot_colon_k, (x, y), None, (2 * xd) @ yd)

    def test_fixed_tiles_past_one_op(self) -> None:
        """Fixed M and N tiles of 4 and 512, and K tiles of 8 and 512, one
        step past the table's bounds, and a K tile of 1, which no op widens."""
        x, y = _randn(600, 64), _randn(64, 600)
        xd, yd = _cpu64(x, y)
        for size in (4, 512):
            fixed = torch.empty(size, device=DEVICE)
            self._scalar(dot_fixed_m, (x, y, fixed), None, (2 * xd) @ yd)
            self._scalar(dot_fixed_n, (x, y, fixed), None, (2 * xd) @ yd)
        x, y = _randn(40, 1024), _randn(1024, 48)
        xd, yd = _cpu64(x, y)
        for size in (1, 8, 512):
            fixed = torch.empty(size, device=DEVICE)
            self._scalar(dot_fixed_k, (x, y, fixed), None, (2 * xd) @ yd)

    def test_rejected_where_the_scalar_contraction_compiled(self) -> None:
        """Forms the planner takes that codegen then rejects at configs the
        scalar contraction compiled: a one-block root tile with no device
        loop inside, and Scalar work beside a ``:`` K op that a lane jam
        would copy more than 64 times."""
        x, y = _randn(40, 72), _randn(72, 32)
        self._rejects(
            dot_flattened_root,
            (x, y),
            helion.Config(block_sizes=[64], num_threads=[1]),
            "is a single root hl.tile with no device loop inside",
        )
        x, y = _randn(40, 72), _randn(72, 48)
        self._rejects(
            dot_colon_decayed,
            (x, y),
            None,
            "a lane loop around an MPP op would run 160 copies",
        )


def _element_loops_with(msl: str, pattern: str) -> list[int]:
    """Where each element loop over a cooperative tensor whose body matches
    ``pattern`` starts in ``msl``."""
    starts = []
    head = r"for \(auto _it = (\w+)\.begin\(\); _it != \1\.end\(\); _it\+\+\) \{"
    for loop in re.finditer(head, msl):
        depth, end = 1, loop.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(msl[end], 0)
            end += 1
        if re.search(pattern, msl[loop.end() : end]):
            starts.append(loop.start())
    return starts


_TILE_WRITE = r"\{ _mpp_tile\w* *\["


@_requires_mps
class TestMppElementStaging(_WholeKDotCase):
    """An operand that is an element-wise function of another op's
    cooperative value, over that op's M and N, is written into its tile from
    that op's element loop: each thread writes the elements MPP gives it,
    with no owner test and no materialization.  The K loop's carry is
    cooperative for the uses that run before the op writes it."""

    def _staged_from_elements(self, msl: str) -> None:
        loops = _element_loops_with(msl, _TILE_WRITE)
        self.assertTrue(loops)
        self.assertFalse(_element_loops_with(msl, r"_mpp_owner|tid\[\d\] == 0"))

    def test_operand_staged_from_another_ops_elements(self) -> None:
        """``lin_attn``'s lhs, with tails in M and in the K loop, and at a
        persistent launch whose threads cover each tile exactly, where the
        staging write needs no in-tile check."""
        persistent = helion.Config(
            block_sizes=[16, 16],
            num_threads=[0, 128],
            num_warps=2,
            pid_type="persistent_interleaved",
            num_sm_multiplier=128,
        )
        for m, n, config in ((48, 80, None), (37, 53, None), (48, 80, persistent)):
            args = (_randn(m, 64, dtype=torch.bfloat16) * 0.3,)
            args += (_randn(n, 64, dtype=torch.bfloat16) * 0.3,)
            args += (_randn(n, 32, dtype=torch.bfloat16),)
            with self.subTest(m=m, n=n, config=config):
                ref = lin_attn_ref(*args)
                code, msl = self._mpp(lin_attn, args, config, ref, ops=2)
                self.assertNotIn("_helion_mpp_materialize", code)
                self._staged_from_elements(msl)

    def test_one_value_staged_into_two_ops(self) -> None:
        """Both dots that read ``lin_attn``'s lhs stage it from the first
        op's element loop."""
        q = _randn(48, 64, dtype=torch.bfloat16) * 0.3
        k = _randn(80, 64, dtype=torch.bfloat16) * 0.3
        v, w = (
            _randn(80, 32, dtype=torch.bfloat16),
            _randn(80, 32, dtype=torch.bfloat16),
        )
        ref = torch.cat(
            [lin_attn_ref(q, k, v).reshape(-1), lin_attn_ref(q, k, w).reshape(-1)]
        )
        code, msl = self._mpp(lin_attn_two_consumers, (q, k, v, w), None, ref, ops=3)
        self.assertNotIn("_helion_mpp_materialize", code)
        self.assertEqual(len(_element_loops_with(msl, _TILE_WRITE)), 2)
        self._staged_from_elements(msl)

    def test_carried_state_read_before_its_update(self) -> None:
        """gdn_fwd_h: the state is a cooperative carry of a 256-row op on
        four SIMD groups; it is stored, and staged as the first dot's rhs in
        bfloat16, before the update and the ops change it, and the first
        dot's computed result is staged as the second dot's rhs."""
        args = _gdn_inputs(256, 256, 32)
        code, msl = self._mpp(gdn_fwd_h, args, None, gdn_ref(*args), ops=2)
        self.assertEqual(_launch(code), (32, 4, 1))
        self.assertIn("matmul2d_descriptor(256, 32, 64,", msl)
        self.assertIn("threadgroup bfloat _mpp_tile", msl)
        self.assertNotIn("_helion_mpp_materialize", code)
        first_run = msl.index("_mpp_op_0.run(")
        (store,) = _element_loops_with(msl, r"\bh\[")
        self.assertLess(store, first_run)
        self.assertLess(_element_loops_with(msl, _TILE_WRITE)[0], first_run)
        self._staged_from_elements(msl)

    def test_carried_state_with_tails(self) -> None:
        """gdn_fwd_h with a last chunk of 8 steps, with a V tail, and with
        both at a 64-wide V tile, where both ops' N is cut to the 40 live
        columns."""
        v_tile = helion.Config(block_sizes=[64], num_warps=4)
        for t, dk, dv, config in (
            (200, 256, 32, None),
            (256, 256, 40, None),
            (200, 64, 40, v_tile),
        ):
            args = _gdn_inputs(t, dk, dv)
            self._mpp(gdn_fwd_h, args, config, gdn_ref(*args), ops=2)

    def test_carry_stored_before_its_update(self) -> None:
        """A K loop's carry stored at each step from its element loop, not
        from a snapshot."""
        x, y = _randn(48, 64), _randn(64, 40)
        xd, yd = _cpu64(x, y)
        ref = torch.stack([xd[:, :c] @ yd[:c] for c in range(0, 64, 16)])
        code, msl = self._mpp(carry_stored_each_step, (x, y), None, ref)
        self.assertNotIn("_helion_mpp_materialize", code)
        self.assertTrue(_element_loops_with(msl, r"\bsnap\["))

    def test_threads_only_on_the_ops_blocks(self) -> None:
        """At configs whose threads around each op lie on its own blocks:
        ``lin_attn_rhs`` with its output rows on one thread,
        ``lin_attn_column_tiles`` at its default, and threads on a block that
        only a sibling loop runs over."""
        x = _randn(40, 72, dtype=torch.bfloat16)
        a = _randn(72, 32, dtype=torch.bfloat16)
        b = _randn(32, 56, dtype=torch.bfloat16)
        config = helion.Config(
            block_sizes=[16, 16, 16], num_threads=[1, 8, 0], num_warps=4, pid_type="xyz"
        )
        self._mpp(lin_attn_rhs, (x, a, b), config, lin_attn_rhs_ref(x, a, b), ops=2)
        q = _randn(48, 64, dtype=torch.bfloat16) * 0.3
        k = _randn(80, 64, dtype=torch.bfloat16) * 0.3
        v = _randn(80, 72, dtype=torch.bfloat16)
        ref = lin_attn_ref(q, k, v)
        self._mpp(lin_attn_column_tiles, (q, k, v), None, ref, ops=2)
        v, w = _randn(80, 32, dtype=torch.bfloat16), _randn(48, 40, dtype=torch.float32)
        ref = torch.cat(
            [lin_attn_ref(q, k, v).reshape(-1), _cpu64(2 * w)[0].reshape(-1)]
        )
        config = helion.Config(
            block_sizes=[16, 16, 16], num_threads=[1, 1, 8], num_warps=4
        )
        self._mpp(lin_attn_sibling_loop, (q, k, v, w), config, ref, ops=2)

    def test_state_scan(self) -> None:
        """A carried state read before its update as a dot's rhs, beside a
        device-view rhs, with tails in T and in the state's columns."""
        for t, e in ((96, 64), (83, 40)):
            args = (
                _randn(t, 64, dtype=torch.bfloat16) * 0.3,
                _randn(t, 64, dtype=torch.bfloat16) * 0.3,
                _randn(t, e, dtype=torch.bfloat16),
                -torch.rand(t, device=DEVICE) * 0.1,
            )
            with self.subTest(t=t, e=e):
                ref = state_scan_ref(*args)
                code, msl = self._mpp(state_scan, args, None, ref, ops=2)
                self.assertNotIn("_helion_mpp_materialize", code)
                self._staged_from_elements(msl)


@_requires_mps
class TestMppElementStagingRejects(_WholeKDotCase):
    """Operands computed from a matmul's result that are not staged from
    its element loop keep the scalar contraction, or a rejection."""

    def test_state_read_after_its_update(self) -> None:
        """The carry is cast before the update, but the dot that reads the
        cast runs after the update and the op write the carry."""
        args = (
            _randn(96, 64, dtype=torch.bfloat16) * 0.3,
            _randn(96, 64, dtype=torch.bfloat16) * 0.3,
            _randn(96, 64, dtype=torch.bfloat16),
            -torch.rand(96, device=DEVICE) * 0.1,
        )
        q, k, v, g = _cpu64(*args)
        state = torch.zeros(64, 64, dtype=torch.float64)
        ref = torch.empty(96, 64, dtype=torch.float64)
        for c in range(0, 96, 32):
            rows = slice(c, c + 32)
            ref[rows] = q[rows] @ state.to(torch.bfloat16).double()
            state = state * torch.exp(g[c]) + k[rows].T @ v[rows]
        self._scalar(state_read_late, args, None, ref)
        self._scalar(state_read_after_decay, args, None, ref)

    def test_scaled_carry_read_after_its_update(self) -> None:
        """The carry scaled before the dot and read after the dot updates
        the carry reads a snapshot taken before the update."""
        x, y = _randn(48, 64), _randn(64, 40)
        xd, yd = _cpu64(x, y)
        ref = torch.stack(
            [
                xd[:, c : c + 16] @ yd[c : c + 16] - xd[:, :c] @ yd[:c]
                for c in range(0, 64, 16)
            ]
        )
        out, code, msl = _run(carry_delta_scaled, (x, y), None)
        self.assertIn("_helion_mpp_materialize", code)
        self.assertEqual(_ops(msl), 1)
        _close(out, ref)

    def test_transposed_result(self) -> None:
        """The second dot reads the first's result transposed."""
        q = _randn(64, 64, dtype=torch.bfloat16) * 0.3
        k = _randn(48, 64, dtype=torch.bfloat16) * 0.3
        w = _randn(64, 32, dtype=torch.bfloat16)
        qd, kd, wd = _cpu64(q, k, w)
        ref = (qd @ kd.T).to(torch.bfloat16).double().T @ wd
        config = helion.Config(block_sizes=[32, 32], num_warps=32)
        self._scalar(lin_attn_transposed, (q, k, w), config, ref)

    def test_chain_rereads_a_written_tensor(self) -> None:
        """The second dot's lhs re-reads a tensor the kernel writes, in the
        dots' loop or outside it, which an element loop would read on another
        thread than the writer."""
        q = _randn(48, 64, dtype=torch.bfloat16) * 0.3
        k = _randn(80, 64, dtype=torch.bfloat16) * 0.3
        v = _randn(80, 32, dtype=torch.bfloat16)
        c = torch.zeros(48, 80, device=DEVICE)
        for kernel in (lin_attn_reloaded, lin_attn_reloaded_nested):
            self._rejects(
                kernel,
                (q, k, v, c),
                None,
                "reduction over persistent dim 2 in a lane loop beside other "
                "reduction dims",
            )
        # The tensor may be written in any graph, the root one included.
        q = _randn(37, 64, dtype=torch.bfloat16) * 0.3
        k = _randn(53, 64, dtype=torch.bfloat16) * 0.3
        v = _randn(53, 40, dtype=torch.bfloat16)
        self._rejects(
            lin_attn_reloaded_outer,
            (q, k, v, torch.rand(37, 53, device=DEVICE) + 0.5),
            helion.Config(block_sizes=[16, 16], num_threads=[16, 16], num_warps=4),
            "in a lane loop beside other reduction dims",
        )

    def test_carry_stored_after_its_update(self) -> None:
        """A copy of the carry stored after the dot updates the carry reads
        a snapshot taken before the update."""
        x, y = _randn(40, 80), _randn(80, 48)
        xd, yd = _cpu64(x, y)
        ref = torch.stack([0.5 + xd[:, :c] @ yd[:c] for c in range(0, 80, 16)])
        out, code, msl = _run(carry_stored_after_its_update, (x, y), None)
        self.assertIn("_helion_mpp_materialize", code)
        self.assertEqual(_ops(msl), 1)
        _close(out, ref)

    def test_threads_on_a_block_outside_the_ops(self) -> None:
        """Where a block that is none of an op's own holds threads around
        the op, a kernel staging one op's result into another keeps the
        scalar contraction: ``lin_attn_rhs``'s output rows at its default, a
        column tile beside ``lin_attn``, and an output tile that a device
        loop holds.  With no operand computed from a matmul, the op stays on
        MPP, which rejects it."""
        x = _randn(40, 72, dtype=torch.bfloat16)
        a = _randn(72, 32, dtype=torch.bfloat16)
        b = _randn(32, 56, dtype=torch.bfloat16)
        ref = lin_attn_rhs_ref(x, a, b)
        self._scalar(lin_attn_rhs, (x, a, b), None, ref)
        tag = torch.zeros(2, device=DEVICE)
        config = helion.Config(
            block_sizes=[16, 16, 16], num_threads=[16, 0, 64], num_warps=4
        )
        self._scalar(lin_attn_rhs_nested, (x, a, b, tag), config, ref.expand(2, -1, -1))
        q = _randn(48, 64, dtype=torch.bfloat16) * 0.3
        k = _randn(80, 64, dtype=torch.bfloat16) * 0.3
        v = _randn(80, 72, dtype=torch.bfloat16)
        config = helion.Config(
            block_sizes=[16, 16, 16],
            num_threads=[8, 4, 0],
            num_warps=16,
            pid_type="xyz",
        )
        self._scalar(lin_attn_column_tiles, (q, k, v), config, lin_attn_ref(q, k, v))
        self._rejects(
            dot_side_block,
            (_randn(48, 64), _randn(64, 40), torch.zeros(4, device=DEVICE)),
            None,
            "block 0 is spread over threads around a cooperative MPP matmul",
        )

    def test_operand_not_computed_from_the_other_op(self) -> None:
        """An operand over the first op's M and N that a load alone computes
        is no element-wise function of the first op's value."""
        chains: list[object] = []
        real = mpp_matmul._element_chain

        def spy(*args: object) -> object:
            chains.append(chain := real(*args))
            return chain

        q = _randn(48, 64, dtype=torch.bfloat16) * 0.3
        k = _randn(80, 64, dtype=torch.bfloat16) * 0.3
        v, x = _randn(80, 32, dtype=torch.bfloat16), _randn(48, 80, dtype=torch.float32)
        bound = lin_attn_loaded_lhs.bind((q, k, v, x))
        with patch.object(mpp_matmul, "_element_chain", spy):
            code = bound.to_code(bound.config_spec.default_config())
        self.assertIn("_helion_mpp_run(", code)
        self.assertTrue(chains)
        self.assertEqual([c for c in chains if c is not None], [])

    def test_value_with_a_scalar_use(self) -> None:
        """The second dot's lhs also feeds a row sum, so it is computed in
        the Scalar layout and the kernel keeps the scalar contraction."""
        q = _randn(48, 64, dtype=torch.bfloat16) * 0.3
        k = _randn(80, 64, dtype=torch.bfloat16) * 0.3
        v = _randn(80, 32, dtype=torch.bfloat16)
        qd, kd, vd = _cpu64(q, k, v)
        p = ((qd @ kd.T) * 0.125).to(torch.bfloat16).double()
        ref = torch.cat(
            [(p @ vd).reshape(-1), p.reshape(48, 5, 16).sum(-1).reshape(-1)]
        )
        config = helion.Config(block_sizes=[32], num_warps=32)
        self._scalar(lin_attn_row_sums, (q, k, v), config, ref)

    def test_source_op_cut_below_the_consumer(self) -> None:
        """The first op's 72-row K loop takes one 64-row tile of a 128-row
        block, so its element loop writes fewer rows of the operand tile
        than the second op reads."""
        args = (
            _randn(24, 64, dtype=torch.bfloat16) * 0.3,
            _randn(72, 64, dtype=torch.bfloat16) * 0.3,
            _randn(72, 48, dtype=torch.bfloat16),
        )
        self._rejects(
            lin_attn_normalized,
            args,
            helion.Config(block_sizes=[32, 32], num_warps=4),
            "an MPP operand staged from another op's elements over a tile that "
            "op cuts to its loop's extent",
        )

    def test_materialized_carry_beside_a_cut_source(self) -> None:
        """A Scalar use of gdn_fwd_h's carry makes the second op materialize
        it, so that op's N keeps the whole 64-wide V tile while the first
        op's N is cut to the 40 live columns: the staged rhs would miss
        columns the second op reads.  At the default 32-wide tile neither op
        is cut."""
        args = _gdn_inputs(200, 64, 40)
        self._rejects(
            gdn_fwd_h_column_sums,
            args,
            helion.Config(block_sizes=[64], num_warps=4),
            "an MPP operand staged from another op's elements over a tile that "
            "op cuts to its loop's extent",
        )
        states = _gdn_states(*args)
        ref = torch.cat(
            [
                states.to(torch.bfloat16).double().reshape(-1),
                states.sum(1).reshape(-1),
            ]
        )
        code, _ = self._mpp(gdn_fwd_h_column_sums, args, None, ref, ops=2)
        self.assertIn("_helion_mpp_materialize", code)

    def test_fold_over_another_lane_in_a_jam(self) -> None:
        """A row max of the first dot's result folds the lanes of its N
        beside the second op, inside the lane loop the second op is jammed
        across."""
        args = (
            _randn(24, 64, dtype=torch.bfloat16) * 0.3,
            _randn(72, 64, dtype=torch.bfloat16) * 0.3,
            _randn(72, 48, dtype=torch.bfloat16),
        )
        self._rejects(
            lin_attn_side_rowmax,
            args,
            helion.Config(block_sizes=[32, 32], num_warps=4),
            "a lane loop around an MPP op holds a reduction over another lane loop",
        )

    def test_batched_chain(self) -> None:
        """A batched op's result reaches the next op through a
        materialization."""
        h = torch.float16
        x, y, z = _randn(3, 48, 32), _randn(3, 32, 64), _randn(3, 64, 32)
        xd, yd, zd = _cpu64(x, y, z)
        out, code, msl = _run(bmm_chain, (x, y, z), None)
        self.assertIn("_helion_mpp_materialize", code)
        self.assertEqual(_ops(msl), 2)
        _close(out, (xd @ yd).to(h).double() @ zd)

    def test_wide_tile_needs_four_simdgroups(self) -> None:
        """gdn_fwd_h's 256-row op on one or two SIMD groups."""
        args = _gdn_inputs(256, 256, 32)
        for warps in (1, 2):
            self._rejects(
                gdn_fwd_h,
                args,
                helion.Config(block_sizes=[32], num_warps=warps),
                f"an MPP matmul tile 256x32 on {warps} SIMD groups; over 128 rows "
                "or columns it needs 4",
            )


@helion.kernel(**S)
def addmm_reduced_seed(
    x: torch.Tensor, y: torch.Tensor, w3: torch.Tensor
) -> torch.Tensor:
    """A seed reduced over a ``:`` dim."""
    hl.specialize(w3.size(2))
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = w3[tm, tn, :].sum(-1)
        for tk in hl.tile(x.size(1)):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def addmm_reduced_update(
    x: torch.Tensor, y: torch.Tensor, w3: torch.Tensor
) -> torch.Tensor:
    """A carry update by a factor reduced over a ``:`` dim."""
    hl.specialize(w3.size(2))
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1), block_size=16):
            acc = acc * w3[tm, tn, :].sum(-1)
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def addmm_reduced_epilogue(
    x: torch.Tensor, y: torch.Tensor, w3: torch.Tensor
) -> torch.Tensor:
    """A stored epilogue that reads a value reduced over a ``:`` dim."""
    hl.specialize(w3.size(2))
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc * w3[tm, tn, :].sum(-1)
    return out


@helion.kernel(**S)
def addmm_side_dim_epilogue(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """An epilogue over the ``:`` dim a row sum spreads over threads, which
    reads the result through its materialization."""
    j = hl.specialize(w.size(1))
    out = torch.empty([x.size(0), y.size(1), j], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([x.size(0), y.size(1)]):
        s = w[tm, :].sum(-1)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x.size(1)):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn, :] = (acc + s[:, None])[:, :, None] * w[tm, :][:, None, :]
    return out


@helion.kernel(**S)
def addmm_reduced_operand(x3: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A staged operand reduced over a ``:`` dim."""
    hl.specialize(x3.size(2))
    out = torch.empty([x3.size(0), y.size(1)], dtype=torch.float32, device=y.device)
    for tm, tn in hl.tile([x3.size(0), y.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(x3.size(1)):
            acc = torch.addmm(acc, x3[tm, tk, :].sum(-1), y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(**S)
def dot_constant_rows(y: torch.Tensor) -> torch.Tensor:
    """An ``hl.dot`` whose M is no block."""
    out = torch.empty([5, y.size(1)], dtype=torch.float32, device=y.device)
    for tn in hl.tile(y.size(1)):
        s = hl.full([5, y.size(0)], 1.0, dtype=torch.float32)
        out[:, tn] = hl.dot(s, y[:, tn])
    return out


_REPLICA_CONFIGS = (None, helion.Config(block_sizes=[16, 16, 16], num_warps=8))


@_requires_mps
class TestMppReplicaThreads(_MppCase):
    """A ``:`` slice may hold threads around a cooperative op when none of
    the group's values, nor a value staged into an operand, has it as a dim:
    threads that differ only on it hold the same element of each."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def _replicas(
        self,
        kernel: helion.Kernel,
        args: tuple[torch.Tensor, ...],
        ref: torch.Tensor,
        configs: tuple[helion.Config | None, ...] = _REPLICA_CONFIGS,
    ) -> None:
        for config in configs:
            with self.subTest(kernel=kernel.name, config=config):
                out, code, _ = _run(kernel, args, config)
                self.assertIn("_helion_mpp_run(", code)
                _close(out, ref)

    def test_reduced_seed_update_and_epilogue(self) -> None:
        x = torch.randn(32, 64, device=DEVICE)
        y = torch.randn(64, 48, device=DEVICE)
        w3 = torch.rand(32, 48, 24, device=DEVICE) / 12
        xd, yd, w3d = (t.cpu().double() for t in (x, y, w3))
        s = w3d.sum(-1)
        self._replicas(addmm_reduced_seed, (x, y, w3), s + xd @ yd)
        self._replicas(addmm_reduced_epilogue, (x, y, w3), (xd @ yd) * s)
        acc = torch.zeros(32, 48, dtype=torch.float64)
        for k in range(0, 64, 16):
            acc = acc * s + xd[:, k : k + 16] @ yd[k : k + 16]
        self._replicas(
            addmm_reduced_update,
            (x, y, w3),
            acc,
            (
                helion.Config(block_sizes=[16, 16], num_warps=8),
                helion.Config(block_sizes=[32, 32], num_warps=32),
            ),
        )

    def test_epilogue_over_the_side_dim(self) -> None:
        x = torch.randn(32, 64, device=DEVICE)
        y = torch.randn(64, 48, device=DEVICE)
        w = torch.randn(32, 24, device=DEVICE)
        xd, yd, wd = (t.cpu().double() for t in (x, y, w))
        acc = xd @ yd + wd.sum(-1, keepdim=True)
        self._replicas(
            addmm_side_dim_epilogue, (x, y, w), acc[:, :, None] * wd[:, None, :]
        )


@_requires_mps
class TestMppReplicaThreadsRejects(_MppCase):
    def test_reduced_operand(self) -> None:
        """A staged operand reduced over a ``:`` dim on threads keeps that dim
        among the group's."""
        x3 = torch.randn(32, 64, 24, device=DEVICE)
        y = torch.randn(64, 48, device=DEVICE)
        self._rejects_at(
            addmm_reduced_operand,
            (x3, y),
            _REPLICA_CONFIGS,
            "is spread over threads around a cooperative MPP matmul",
        )


# --- Threadgroup budget: num_warps sizes the launch -----------------------


@helion.kernel(**S)
def free_rows_wide(
    a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """grouped_gemm's layout with its 32 to 128 row and column blocks."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(32, 128)
    bn = hl.register_block_size(32, 128)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def free_rows_wide_scalar_store(
    a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    first = torch.zeros([m], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(32, 128)
    bn = hl.register_block_size(32, 128)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
            hl.store(first, [rows], hl.load(a, [rows, 0], extra_mask=rv), extra_mask=rv)
    return out, first


@helion.kernel(**S)
def free_rows_wide_staged(
    a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(32, 128)
    bn = hl.register_block_size(32, 128)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None]) * 2
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(**S)
def free_rows_wide_dead_carry(
    a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(32, 128)
    bn = hl.register_block_size(32, 128)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = torch.addmm(acc, a_blk, b_blk)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
            s = hl.zeros([bm], dtype=torch.float32)
            for j in hl.grid(3):
                s = s + hl.load(a, [rows, j], extra_mask=rv)
    return out


def _launch(code: str) -> tuple[int, ...]:
    """The launch's threadgroup shape, from the host code's ``_block_dims``."""
    dims = re.search(r"_block_dims=\((\d+), (\d+), (\d+)\)", code)
    assert dims is not None, code
    return tuple(int(d) for d in dims.groups())


def _open_loops(msl: str, pos: int) -> list[str]:
    """The headers of the MSL ``for`` blocks open at ``pos``."""
    stack: list[str] = []
    for match in re.finditer(r"for \(([^)]*)\) \{|\{|\}", msl[:pos]):
        if match.group(0) == "}":
            stack.pop()
        else:
            stack.append(match.group(1) or "")
    return [header for header in stack if header]


def _distribute(source: str) -> str | None:
    """``_distribute_lane_loop`` of the lane loop *source*, unparsed."""
    loop = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(loop, ast.For)
    out = mpp_matmul._distribute_lane_loop(loop, "slice_lane_0", set())
    if out is None:
        return None
    return ast.unparse(ast.fix_missing_locations(ast.Module(out, [])))


@_requires_mps
class TestMppThreadgroupBudget(unittest.TestCase):
    """``num_warps`` sizes the threadgroup around an MPP op: blocks beyond it
    walk lanes, and a lane loop around the op runs once per run of the
    statements that depend on its lane."""

    def _attention_args(self) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        torch.manual_seed(0)
        bf = torch.bfloat16
        q, k, v = (torch.randn(1, 2, 64, 64, dtype=bf) for _ in range(3))
        scale = 1.0 / math.sqrt(64) * 1.44269504
        scores = (q.double() * scale).to(bf).double() @ k.double().transpose(-1, -2)
        expected = torch.softmax(scores * math.log(2), -1) @ v.double()
        return (q.to(DEVICE), k.to(DEVICE), v.to(DEVICE)), _bf(expected)

    def test_attention_head_dim_walks_lanes(self) -> None:
        """At the default num_warps=4, attention's ``:`` head dim takes what
        the tiles leave, one thread of 64 lanes, and its lane loop is split
        around the K loop so no lane loop holds an MPP op."""
        args, expected = self._attention_args()
        (out, _), code, msl = _run(attention_metal, args, None)
        _close(out, expected)
        self.assertEqual(math.prod(_launch(code)), 128)
        self.assertEqual(msl.count("for (int slice_lane_"), 1)
        runs = [m.start() for m in re.finditer(r"_mpp_op_\d+\.run\(", msl)]
        self.assertTrue(runs)
        for pos in runs:
            self.assertFalse(
                [h for h in _open_loops(msl, pos) if "lane_" in h.split("=")[0]]
            )

    def test_attention_at_wider_budgets(self) -> None:
        args, expected = self._attention_args()
        default = attention_metal.bind(args).config_spec.default_config().config
        for num_warps in (8, 32):
            with self.subTest(num_warps=num_warps):
                config = helion.Config(**{**default, "num_warps": num_warps})
                (out, _), code, _ = _run(attention_metal, args, config)
                _close(out, expected)
                self.assertEqual(math.prod(_launch(code)), 32 * num_warps)

    def test_k_loop_tile_shares_the_budget(self) -> None:
        """int4_gemm at M = 1: the 16 x 16 output tile fits 128 threads by
        giving the N tile 8 threads of 2 lanes."""
        torch.manual_seed(0)
        w = torch.randint(-8, 8, (128, 64))
        a = torch.randn(1, 128).to(torch.bfloat16)
        out, code, _ = _run(int4_gemm, (a.to(DEVICE), _pack_int4(w).to(DEVICE)), None)
        _close(out, (a.double() @ w.double()).to(torch.bfloat16).double())
        self.assertEqual(_launch(code), (16, 8, 1))

    def test_free_blocks_walk_lanes(self) -> None:
        """grouped_gemm's 128 x 128 tile fits any budget: each free block
        takes threads within it and walks the rest in lanes, which only the
        element loops bind.  Explicit counts are taken in block order, each
        within what the earlier ones left."""
        bf = torch.bfloat16
        a = torch.randn(200, 64, dtype=bf)
        b = torch.randn(64, 300, dtype=bf)
        args = (a.to(DEVICE), b.to(DEVICE), 2, 3)
        for num_warps, num_threads, launch, laned in (
            (32, [], (32, 32, 1), "_free_lane_0"),
            (8, [], (16, 16, 1), "_free_lane_0"),
            (4, [128, 128, 0], (128, 1, 1), "_free_lane_1"),
        ):
            with self.subTest(num_warps=num_warps, num_threads=num_threads):
                config = helion.Config(
                    block_sizes=[128, 128, 32],
                    num_threads=num_threads,
                    num_warps=num_warps,
                )
                out, code, msl = _run(free_rows_wide, args, config)
                _close(out, _mm(a, b))
                self.assertEqual(_launch(code), launch)
                self.assertIn(laned, msl)
                self.assertNotIn("_free_lane_", msl[: msl.index("is_valid_element")])

    def test_free_lanes_read_by_scalar_code_reject(self) -> None:
        bf = torch.bfloat16
        a = torch.randn(200, 64, dtype=bf)
        b = torch.randn(64, 300, dtype=bf)
        config = helion.Config(block_sizes=[128, 128, 32], num_warps=8)
        for kernel, reader in (
            (free_rows_wide_scalar_store, "store_1"),
            (free_rows_wide_staged, "a staged MPP operand"),
            (free_rows_wide_dead_carry, "s"),
        ):
            with (
                self.subTest(kernel=kernel.name),
                self.assertRaisesRegex(
                    exc.BackendUnsupported,
                    rf"block 0 on 16 threads x 8 lanes, but {reader} reads it in "
                    r"Scalar code",
                ),
            ):
                kernel.bind((a, b, 2, 3)).to_code(config)
        # A ``_mpp_coop`` seed is staged from the Scalar layout, by thread.
        a = torch.randn(45, 72, dtype=bf)
        b = torch.randn(72, 40, dtype=bf)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"block 0 on 8 threads x 4 lanes, but _mpp_coop reads it in Scalar code",
        ):
            free_rows_seeded.bind((a, b, torch.randn(45, 40), 2, 2)).to_code(
                helion.Config(block_sizes=[32, 32, 16], num_warps=4)
            )

    def test_former_jam_cap_rejections_compile(self) -> None:
        """A ':' K that MPP contracts takes what the tiles leave, so 64 x 64
        tiles run on 32 x 32 threads, not 4 x 4 threads around 256 jammed
        lanes, past the 64-copy cap; two staged ops' 64-row tile likewise
        fits without a jam."""
        xs, ys = torch.randn(2, 64, 64, device=DEVICE)
        c = torch.randn(64, 64, device=DEVICE)
        w = torch.randn(64, device=DEVICE)
        out, code, _ = _run(
            colon_seed_epi,
            (xs, ys, c, w),
            helion.Config(block_sizes=[64, 64], num_warps=32),
        )
        self.assertEqual(_launch(code), (32, 32, 1))
        _close(out, c.cpu().double() * 0.5 + _mm(xs, ys) + w.cpu().double()[:, None])
        x = torch.randn(48, 72, device=DEVICE)
        y = torch.randn(72, 40, device=DEVICE)
        (out, out2), code, _ = _run(
            two_ops_staged,
            (x, y),
            helion.Config(
                block_sizes=[64, 16, 32], num_threads=[16, 16, 1], num_warps=32
            ),
        )
        self.assertEqual(_launch(code), (16, 16, 1))
        _close(out, _mm(x * 2, y))
        _close(out2, _mm(x, y * 0.5))

    def test_row_sum_tile_beside_the_k_loop_takes_threads(self) -> None:
        """The row sum's nested tile takes threads beside a 16-thread M tile,
        so the op gets two whole SIMD groups."""
        x = torch.randn(48, 72, device=DEVICE)
        y = torch.randn(72, 40, device=DEVICE)
        out, code, _ = _run(
            root_beside_sum,
            (x, y),
            helion.Config(
                block_sizes=[16, 16, 16, 32], num_threads=[0, 1, 0, 0], num_warps=32
            ),
        )
        self.assertEqual(_launch(code), (16, 4, 1))
        _close(out, _mm(x, y) + x.cpu().double().sum(-1, keepdim=True))

    def test_distribution_repeats_a_cast_index(self) -> None:
        """A pure index that casts (the signed-index printer's
        ``static_cast(long, ...)``) is repeated like one that doesn't."""
        self.assertEqual(
            _distribute(
                """
                for slice_lane_0 in range(4):
                    i = static_cast(long, slice_lane_0) * 16
                    tl.store(y + i, 1.0)
                    _mpp_acc_0 = _helion_mpp_acc(0, None)
                    tl.store(out + i, 2.0)
                """
            ),
            textwrap.dedent(
                """\
                for slice_lane_0 in range(4):
                    i = static_cast(long, slice_lane_0) * 16
                    tl.store(y + i, 1.0)
                _mpp_acc_0 = _helion_mpp_acc(0, None)
                for slice_lane_0 in range(4):
                    i = static_cast(long, slice_lane_0) * 16
                    tl.store(out + i, 2.0)"""
            ),
        )

    def test_distribution_positions(self) -> None:
        """Each run of lane-dependent statements becomes its own lane loop,
        with the pure index it reads repeated; lane-invariant statements run
        once, in order, a run's own invariant assignment ahead of its loop."""
        self.assertEqual(
            _distribute(
                """
                for slice_lane_0 in range(4):
                    i = slice_lane_0 * 16
                    t = n * 2
                    tl.store(y + i, tl.load(x + i) * t)
                    _mpp_acc_0 = _helion_mpp_acc(0, None)
                    tl.store(w, t)
                    tl.store(out + i, tl.load(z + i))
                """
            ),
            textwrap.dedent(
                """\
                t = n * 2
                for slice_lane_0 in range(4):
                    i = slice_lane_0 * 16
                    tl.store(y + i, tl.load(x + i) * t)
                _mpp_acc_0 = _helion_mpp_acc(0, None)
                tl.store(w, t)
                for slice_lane_0 in range(4):
                    i = slice_lane_0 * 16
                    tl.store(out + i, tl.load(z + i))"""
            ),
        )
        # ``t = n * 2`` can't move ahead of the store deferred before it,
        # which reads ``t``.
        self.assertEqual(
            _distribute(
                """
                for slice_lane_0 in range(4):
                    _mpp_acc_0 = _helion_mpp_acc(0, None)
                    tl.store(y + slice_lane_0, 1.0)
                    tl.store(w, t)
                    t = n * 2
                    tl.store(out + slice_lane_0, t)
                """
            ),
            textwrap.dedent(
                """\
                _mpp_acc_0 = _helion_mpp_acc(0, None)
                for slice_lane_0 in range(4):
                    tl.store(y + slice_lane_0, 1.0)
                tl.store(w, t)
                t = n * 2
                for slice_lane_0 in range(4):
                    tl.store(out + slice_lane_0, t)"""
            ),
        )

    def test_distribution_declines(self) -> None:
        for name, source in (
            (
                "lane-dependent cooperative statement",
                """
                for slice_lane_0 in range(4):
                    i = slice_lane_0 * 16
                    _mpp_acc_0 = _helion_mpp_acc(0, None)
                    _mpp_acc_1 = _helion_mpp_acc(i, None)
                    tl.store(out + i, tl.load(z + i))
                """,
            ),
            (
                "deferred store's tensor read later in its run",
                """
                for slice_lane_0 in range(4):
                    _mpp_acc_0 = _helion_mpp_acc(0, None)
                    tl.store(y + slice_lane_0, 1.0)
                    tl.store(w, 2.0)
                    q = tl.load(w + slice_lane_0)
                    tl.store(out + slice_lane_0, q)
                """,
            ),
            (
                "self-reading value read by a later run",
                """
                for slice_lane_0 in range(4):
                    i = slice_lane_0 * 16
                    i = i + 1
                    tl.store(y + i, 1.0)
                    _mpp_acc_0 = _helion_mpp_acc(0, None)
                    tl.store(out + i, 2.0)
                """,
            ),
            (
                "value carried across lanes",
                """
                for slice_lane_0 in range(4):
                    tl.store(y + c, tl.load(x + slice_lane_0))
                    c = slice_lane_0 + 1
                    _mpp_acc_0 = _helion_mpp_acc(0, None)
                """,
            ),
            (
                "loaded value read by a later run",
                """
                for slice_lane_0 in range(4):
                    q = tl.load(x + slice_lane_0)
                    _mpp_acc_0 = _helion_mpp_acc(0, None)
                    tl.store(out + slice_lane_0, q)
                """,
            ),
            (
                "no cooperative statement",
                """
                for slice_lane_0 in range(4):
                    tl.store(y + slice_lane_0, tl.load(x + slice_lane_0))
                """,
            ),
        ):
            with self.subTest(name):
                self.assertIsNone(_distribute(source))

    def test_lane_start_value_stays_in_its_lane_loop(self) -> None:
        """A per-row sum's ``s = 0.0`` reads no lane but holds a value per
        lane, so it stays in the lane loop that accumulates it."""
        self.assertEqual(
            _distribute(
                """
                for slice_lane_0 in range(2):
                    tl.store(y + slice_lane_0, 1.0)
                    _mpp_acc_0 = _helion_mpp_acc(0, None)
                    s = 0.0
                    for j in range(3):
                        s = s + tl.load(x + slice_lane_0 * 3 + j)
                    tl.store(out + slice_lane_0, s)
                """
            ),
            textwrap.dedent(
                """\
                for slice_lane_0 in range(2):
                    tl.store(y + slice_lane_0, 1.0)
                _mpp_acc_0 = _helion_mpp_acc(0, None)
                for slice_lane_0 in range(2):
                    s = 0.0
                    for j in range(3):
                        s = s + tl.load(x + slice_lane_0 * 3 + j)
                    tl.store(out + slice_lane_0, s)"""
            ),
        )
        # root_beside_sum's row sum beside its K loop, its rows on 2 lanes.
        x = torch.randn(48, 72, device=DEVICE, dtype=torch.float16)
        y = torch.randn(72, 40, device=DEVICE, dtype=torch.float16)
        xf, yf = x.float(), y.float()
        out, code, _ = _run(root_beside_sum, (xf, yf), None)
        self.assertEqual(_launch(code), (8, 16, 1))
        _close(out, _mm(xf, yf) + xf.cpu().double().sum(-1, keepdim=True))


# --- Lane distribution around a cooperative statement ---------------------


@helion.kernel(**S)
def dist_loop_reset(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A device loop beside the K loop that resets a per-lane value and
    accumulates one that depends on no lane."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        s = x[tm, tn] * 1.0
        t = hl.zeros([tn], dtype=torch.float32)
        for tj in hl.tile(k):
            s = hl.zeros([tm, tn], dtype=torch.float32)
            t = t + y[tj, tn].sum(0)
        w = t * 2.0
        out[tm, tn] = acc + s + w[None, :]
    return out


@helion.kernel(**S)
def dist_if_reset(x: torch.Tensor, y: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    """The ``if`` form of dist_loop_reset."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for g in hl.grid(1):
        f = flag[g]
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            s = x[tm, tn] * 1.0
            t = y[0, tn] * 1.0
            if f > 0:
                s = hl.zeros([tm, tn], dtype=torch.float32)
                t = t + 1.0
            w = t * 2.0
            out[tm, tn] = acc + s + w[None, :]
    return out


@helion.kernel(**S)
def dist_stale_index(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A call-free per-lane value that a device loop updates before the K
    loop and that is read after it."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        s = tm.index * 2
        for tj in hl.tile(k):
            s = s + y[tj, 0].sum(0).to(torch.int32)
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc + s[:, None]
    return out


@helion.kernel(**S)
def dist_rmw(x: torch.Tensor, y: torch.Tensor, cnt: torch.Tensor) -> torch.Tensor:
    """A per-lane reset beside a read-modify-write of one row per tile."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        s = x[tm, tn] * 1.0
        for _tj in hl.tile(k):
            s = hl.zeros([tm, tn], dtype=torch.float32)
            cnt[tm.id, tn] = cnt[tm.id, tn] + 1.0
        out[tm, tn] = acc + s
    return out


@helion.kernel(**S)
def dist_rb_loop(x: torch.Tensor, y: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
    """Each iteration reads the tile's first row of ``out`` and stores every
    row from it."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        s = x[tm, tn] * 1.0
        for _tj in hl.tile(k):
            s = hl.zeros([tm, tn], dtype=torch.float32)
            a = out[tm.begin, tn]
            out[tm, tn] = a[None, :] + s + 1.0
        out2[tm, tn] = acc + s
    return out


@helion.kernel(**S)
def dist_readback(x: torch.Tensor, y: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
    """Store a row, then read back its pair row (2i <-> 2i + 1)."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc + x[tm, tn]
        rows = tm.index + 1 - 2 * (tm.index % 2)
        out2[tm, tn] = out[rows, tn]
    return out


@helion.kernel(**S)
def dist_swap_load(x: torch.Tensor, y: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
    """Load the pair row of ``buf``, then store this row: every load reads
    the old values."""
    m, k = x.size()
    _, n = y.size()
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        rows = tm.index + 1 - 2 * (tm.index % 2)
        b = buf[rows, tn]
        buf[tm, tn] = b + acc
    return buf


@helion.kernel(**S)
def dist_two_stores(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Store every row, then overwrite the pair rows: the second store
    wins."""
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        v = acc + x[tm, tn]
        rows = tm.index + 1 - 2 * (tm.index % 2)
        out[tm, tn] = v
        out[rows, tn] = v * 2.0
    return out


@helion.kernel(**S)
def dist_store_then_load(
    x: torch.Tensor, y: torch.Tensor, buf: torch.Tensor
) -> torch.Tensor:
    """A store and a load of ``buf`` that read no lane, after a per-lane
    statement."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        v = x[tm, tn] * 2.0
        buf[tm.id, tn] = y[0, tn] * 3.0
        z = buf[tm.id, tn]
        out[tm, tn] = acc + v + z[None, :]
    return out


@helion.kernel(**S)
def dist_store_then_load_acc(
    x: torch.Tensor, y: torch.Tensor, buf: torch.Tensor
) -> torch.Tensor:
    """dist_store_then_load whose per-lane statement reads the result."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        v = acc + x[tm, tn]
        buf[tm.id, tn] = y[0, tn] * 3.0
        z = buf[tm.id, tn]
        out[tm, tn] = v + z[None, :]
    return out


def _lane_looped(msl: str, pattern: str) -> bool:
    """Whether the first match of ``pattern`` in ``msl`` sits in a rolled
    ``lane_`` loop."""
    match = re.search(pattern, msl)
    assert match is not None, pattern
    return any(
        re.match(r"int lane_\d+ = 0", header)
        for header in _open_loops(msl, match.start())
    )


def _pair_rows(t: torch.Tensor) -> torch.Tensor:
    """``t`` with rows 2i and 2i + 1 swapped."""
    idx = torch.arange(t.size(0))
    return t[idx + 1 - 2 * (idx % 2)]


@_requires_mps
class TestMppLaneDistribution(unittest.TestCase):
    """A lane loop around a cooperative statement that reads no lane is
    split into one rolled lane loop per run of the statements that depend on
    its lane, and runs in the order the jam would give it; where a run's
    lane-by-lane order could differ from the jam's, the jam takes the loop."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(48, 72, device=DEVICE)
        self.y = torch.randn(72, 40, device=DEVICE)
        self.xy = _mm(self.x, self.y)
        self.xd = self.x.cpu().double()[:, :40]

    def _jammed(self, msl: str) -> None:
        """The rows' lane loop (block 0's) is jammed, not rolled."""
        self.assertIn("__lane1", msl)
        self.assertNotIn("for (int lane_0 = 0", msl)

    def test_per_lane_start_stays_in_its_lane_loop(self) -> None:
        """``s = 0.0`` starting a per-row sum reads no lane but holds a value
        per lane."""
        _, _, msl = _run(root_beside_sum, (self.x, self.y), None)
        self.assertTrue(_lane_looped(msl, re.escape("auto s = float(0.0);")))

    def test_element_loop_definitions_stay_private(self) -> None:
        """Names an element loop defines don't leave it, so upd_seed's seed
        staging, 16 x 16 on 128 threads, keeps one rolled lane loop."""
        h = torch.float16
        p = torch.randn(48, 72, device=DEVICE, dtype=h)
        v = torch.randn(72, 40, device=DEVICE, dtype=h)
        c = torch.randn(48, 40, device=DEVICE)
        a = torch.randn(48, 72, device=DEVICE) * 0.3
        out, _, msl = _run(upd_seed, (p, v, c, a), TILE16)
        expected = _d(c)
        for ks in _k_tiles(72, 16):
            expected = expected * _d(a)[:, ks.start, None] + _d(p)[:, ks] @ _d(v)[ks]
        _close(out, expected)
        self.assertEqual(len(re.findall(r"for \(int lane_\d+ = 0", msl)), 1)
        self.assertNotIn("__lane1", msl)

    def test_fold_combine_stays_after_its_fold(self) -> None:
        """A lane fold's partial is lane-dependent, so the combine across
        threads that reads it stays after the fold."""
        q = torch.randn(48, 64, device=DEVICE)
        kk = torch.randn(80, 64, device=DEVICE)
        p = torch.randn(48, 72, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(72, 40, device=DEVICE, dtype=torch.bfloat16)
        lse = torch.logsumexp(_d(q) @ _d(kk).T, -1)
        for kernel, args, expected, config in (
            (nloop_lse, (q, kk), lse, None),
            (
                nloop_lse,
                (q, kk),
                lse,
                helion.Config(block_sizes=[16, 16], num_threads=[16, 1], num_warps=4),
            ),
            (
                nloop_lse,
                (q, kk),
                lse,
                helion.Config(block_sizes=[32, 32], num_threads=[8, 16], num_warps=4),
            ),
            (colon_n_rowsum, (p, v), _mm(p, v).sum(-1), None),
        ):
            with self.subTest(kernel=kernel.name, config=config):
                out, _, msl = _run(kernel, args, config)
                _close(out, expected)
                for part in re.findall(r"auto (lane_\w*part\w*) =", msl):
                    combine = re.search(rf"group_\w+\([^;]*\b{part}\b", msl)
                    assert combine is not None
                    self.assertLess(msl.index(f"auto {part} ="), combine.start())

    def test_compound_assignments_follow_the_lane(self) -> None:
        """A device loop or ``if`` that resets a per-lane value runs per lane,
        so every other name it assigns is per lane too: its start and its
        readers stay in the lane loop."""
        ysum = self.y.cpu().double().sum(0)
        for config in (
            None,
            helion.Config(block_sizes=[32, 16, 16, 16], num_warps=4),
        ):
            with self.subTest(kernel="loop", config=config):
                out, _, msl = _run(dist_loop_reset, (self.x, self.y), config)
                _close(out, self.xy + 2.0 * ysum[None, :])
                self.assertTrue(_lane_looped(msl, re.escape("auto t = float(0.0);")))
                self.assertTrue(_lane_looped(msl, r"\(t \* v_\d+\)"))
        flag = torch.ones(1, dtype=torch.int32, device=DEVICE)
        config = helion.Config(
            block_sizes=[16, 16, 16], num_threads=[8, 16, 0], num_warps=4
        )
        with self.subTest(kernel="if"):
            out, _, msl = _run(dist_if_reset, (self.x, self.y, flag), config)
            y0 = self.y.cpu().double()[0]
            _close(out, self.xy + 2.0 * (y0 + 1.0)[None, :])
            self.assertTrue(_lane_looped(msl, r"if \(v_\d+\) \{"))

    def test_updated_value_is_not_recomputed(self) -> None:
        """A later run can't recompute a per-lane value from its first
        definition once a device loop has updated it, so the jam takes the
        loop."""
        y = self.y.clone()
        y[:, 0] = torch.randint(0, 4, (72,), device=DEVICE).float()
        rows = 2.0 * torch.arange(48, dtype=torch.float64)[:, None]
        expected = _mm(self.x, y) + rows + y.cpu().double()[:, 0].sum()
        for config in (
            None,
            helion.Config(block_sizes=[32, 16, 16, 16], num_warps=4),
            helion.Config(
                block_sizes=[16, 16, 16, 16], num_threads=[8, 16, 0, 0], num_warps=4
            ),
        ):
            with self.subTest(config=config):
                out, _, msl = _run(dist_stale_index, (self.x, y), config)
                _close(out, expected)
                self._jammed(msl)

    def test_runs_that_meet_their_own_stores_are_jammed(self) -> None:
        """A run executes lane by lane where the jam runs every lane's copy
        of a statement before the next: when a tensor the run stores is
        touched by another of its statements, or loaded by the statement
        that stores it, the jam takes the loop."""
        x, y, xy, xd = self.x, self.y, self.xy, self.xd
        lanes_32 = [
            None,
            helion.Config(block_sizes=[16, 32, 16, 16], num_threads=[1, 32, 0, 0]),
        ]
        for config in [
            *lanes_32,
            helion.Config(block_sizes=[16, 32, 16, 16], num_threads=[2, 32, 0, 0]),
        ]:
            with self.subTest(kernel="rmw", config=config):
                cnt = torch.zeros(48, 40, device=DEVICE)
                out, _, msl = _run(dist_rmw, (x, y, cnt), config)
                bound = dist_rmw.bind((x, y, cnt))
                blocks = (config or bound.config_spec.default_config()).block_sizes
                expected = torch.zeros(48, 40, dtype=torch.float64)
                expected[: math.ceil(48 / blocks[0])] = math.ceil(72 / blocks[3])
                _close(out, xy)
                _close(cnt, expected)
                self._jammed(msl)
        for config in lanes_32:
            with self.subTest(kernel="rb_loop", config=config):
                out2 = torch.zeros(48, 40, device=DEVICE)
                out, _, msl = _run(dist_rb_loop, (x, y, out2), config)
                blocks = (
                    config
                    or dist_rb_loop.bind((x, y, out2)).config_spec.default_config()
                ).block_sizes
                _close(
                    out,
                    torch.full(
                        (48, 40), float(math.ceil(72 / blocks[3])), dtype=torch.float64
                    ),
                )
                _close(out2, xy)
                self._jammed(msl)
        rows_16 = [
            None,
            helion.Config(block_sizes=[16, 16, 16], num_threads=[8, 16, 0]),
        ]
        two_warps = helion.Config(
            block_sizes=[16, 16, 16], num_threads=[4, 16, 0], num_warps=2
        )
        for config in rows_16:
            with self.subTest(kernel="swap_load", config=config):
                buf = torch.randn(48, 40, device=DEVICE)
                expected = _pair_rows(buf.cpu().double()) + xy
                out, _, msl = _run(dist_swap_load, (x, y, buf), config)
                _close(out, expected)
                self._jammed(msl)
        for config in [*rows_16, two_warps]:
            with self.subTest(kernel="two_stores", config=config):
                out, _, msl = _run(dist_two_stores, (x, y), config)
                _close(out, 2.0 * _pair_rows(xy + xd))
                self._jammed(msl)
            with self.subTest(kernel="readback", config=config):
                out2 = torch.zeros(48, 40, device=DEVICE)
                out, _, msl = _run(dist_readback, (x, y, out2), config)
                _close(out, xy + xd)
                _close(out2, _pair_rows(xy + xd))
                self._jammed(msl)

    def test_run_storing_only_its_own_tensor_stays_distributed(self) -> None:
        """A nested slice loop that stores ``out``, which nothing else in the
        run touches, keeps its rolled lane loop."""
        p = torch.randn(48, 72, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(72, 40, device=DEVICE, dtype=torch.bfloat16)
        config = helion.Config(block_sizes=[32, 32], num_warps=4)
        out, _, msl = _run(colon_n_rowsum, (p, v), config)
        _close(out, _mm(p, v).sum(-1))
        self.assertRegex(msl, r"for \(int lane_\d+ = 0")

    def test_invariant_load_stays_after_the_store_it_reads(self) -> None:
        """A store that reads no lane moves after its run's loop, so a load
        of the same tensor that reads no lane can't move ahead of it."""
        x, y = self.x, self.y
        y0 = 3.0 * y.cpu().double()[0][None, :]
        for kernel, expected, config in (
            (dist_store_then_load, self.xy + 2.0 * self.xd + y0, None),
            (
                dist_store_then_load,
                self.xy + 2.0 * self.xd + y0,
                helion.Config(
                    block_sizes=[16, 16, 16], num_threads=[1, 16, 0], num_warps=4
                ),
            ),
            (dist_store_then_load_acc, self.xy + self.xd + y0, None),
        ):
            with self.subTest(kernel=kernel.name, config=config):
                buf = torch.full((48, 40), -100.0, device=DEVICE)
                out, _, msl = _run(kernel, (x, y, buf), config)
                _close(out, expected)
                self.assertLess(msl.index("buf[(("), msl.index("? buf[(("))

    def test_staging_loop_inside_a_distributed_lane_loop(self) -> None:
        """A staged ``:`` K splits its staging loop inside the row lane loop
        that E then distributes; the staging keeps its split form there."""
        h = torch.float16
        x = torch.randn(48, 64, device=DEVICE, dtype=h)
        y = torch.randn(64, 40, device=DEVICE, dtype=h)
        xb = torch.randn(48, 72, device=DEVICE, dtype=torch.bfloat16)
        yb = torch.randn(40, 72, device=DEVICE, dtype=torch.bfloat16)
        x3 = torch.randn(3, 48, 64, device=DEVICE)
        y3 = torch.randn(3, 64, 40, device=DEVICE)
        for kernel, args, expected in (
            (colon_k_staged, (x, y), _mm(x, y) * 2),
            (colon_staged_lhs, (xb, yb), _mm(xb, yb.T) * 0.5),
            (
                bmm_staged,
                (x3, y3),
                torch.bmm(x3.cpu().double() * 0.5, y3.cpu().double()),
            ),
        ):
            with self.subTest(kernel=kernel.name):
                out, _, msl = _run(kernel, args, None)
                _close(out, expected)
                self.assertTrue(_lane_looped(msl, r"for \(int slice_lane_\d+"))
                self.assertNotIn("_mpp_lanes_", msl)


# --- Uniform launches: masks the shared per-tile rule proves true ----------


@helion.kernel(**S)
def bias_relu(x: torch.Tensor, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = torch.relu(acc + b[tn]).to(torch.float16)
    return out


@helion.kernel(**S)
def add_into(x: torch.Tensor, y: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        o[tm, tn] = acc + o[tm, tn]
    return o


@helion.kernel(**S)
def colsum_one_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n], block_size=[m, None]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tn] = acc.sum(0)
    return out


#: 256 threads as 16 x 16, every tile loop on 16 of them.
_UNIFORM16 = helion.Config(
    block_sizes=[16, 16, 16], num_threads=[16, 16, 1], num_warps=8
)
_INT4_UNIFORM = (
    (
        helion.Config(block_sizes=[32, 16, 64], num_threads=[0, 1, 0], num_warps=2),
        (64, 1, 1),
    ),
    (helion.Config(block_sizes=[128, 16, 16], num_warps=8), (16, 16, 1)),
)


def _int4_args(m: int, k: int, n: int) -> tuple[torch.Tensor, ...]:
    """int4_gemm's arguments (A a slice of a NaN buffer) and its result."""
    w = torch.randint(-8, 8, (k, n))
    a = torch.randn(m, k).to(torch.bfloat16)
    ref = (a.double() @ w.double()).to(torch.bfloat16).double()
    return _nan_padded(a.to(DEVICE)), _pack_int4(w).to(DEVICE), ref


def _attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = _d(q) @ _d(k).transpose(-1, -2) / math.sqrt(q.size(-1))
    return torch.softmax(scores, -1) @ _d(v)


@_requires_mps
class TestMppUniformLaunch(_MppCase):
    """When every strategy of a planner MPP kernel has one thread extent, the
    launch has no surplus threads: a block whose extent divides it drops its
    masks, and a tile loop's staging writes and materialized-tile reads
    drop their in-tile checks.  Each config's launch is asserted, since only
    a uniform one takes this path."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def _launches(self, code: str, launch: tuple[int, int, int]) -> None:
        self.assertIn(f"_block_dims={launch}", code)

    def test_int4_gemm(self) -> None:
        """int4_gemm's staged B: no select on its load or on its lane
        array's read-back, and an unguarded staging write."""
        for m in (1, 16):
            lhs, rhs, ref = _int4_args(m, 256, 128)
            for config, launch in _INT4_UNIFORM:
                with self.subTest(m=m, config=config):
                    out, code, msl = _run(int4_gemm, (lhs, rhs), config)
                    _close(out, ref)
                    self._launches(code, launch)
                    self.assertNotRegex(msl, r"\bmask_\d+")
                    self.assertNotIn("? B[", msl)
                    self.assertRegex(msl, r"if \(_mpp_owner\w*\) \{ _mpp_tile\[")
                    if launch == (64, 1, 1):
                        self.assertRegex(
                            msl, r"auto b_tile = _mpp_lanes_\d+\[lane_\d+\];"
                        )

    def test_epilogues(self) -> None:
        """A bias, relu and f16 cast, and an epilogue that reads its own
        destination through the materialized tile, 50 times with fresh
        arguments: no mask and no select on the tile read."""
        x = torch.randn(32, 128, device=DEVICE, dtype=torch.float16)
        y = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
        b = torch.randn(64, device=DEVICE)
        with self.subTest(kernel="bias_relu"):
            out, code, msl = _run(bias_relu, (x, y, b), _UNIFORM16)
            _close(out, torch.relu(_mm(x, y) + _d(b)))
            self._launches(code, (16, 16, 1))
            self.assertNotRegex(msl, r"\bmask_\d+")
        for _ in range(50):
            o = torch.randn(32, 64, device=DEVICE)
            ref = _mm(x, y) + _d(o)
            with self.subTest(kernel="add_into"):
                out, code, msl = _run(add_into, (x, y, o), _UNIFORM16)
                _close(out, ref)
                self._launches(code, (16, 16, 1))
                self.assertNotRegex(msl, r"\bmask_\d+")
                self.assertRegex(msl, r"= _mpp_mat\w*\[")
                self.assertNotIn("? _mpp_mat", msl)

    def test_dot_k_loop(self) -> None:
        """An ``hl.dot`` K loop drops its masks like an ``addmm`` one."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        out, code, msl = _run(dot_seed_zeros, (x, y), _UNIFORM16)
        _close(out, _mm(x, y))
        self.assertIn("_helion_mpp_acc(", code)
        self._launches(code, (16, 16, 1))
        self.assertNotRegex(msl, r"\bmask_\d+")

    def test_reductions(self) -> None:
        """Column sums of one M tile, and attention's row max and sum, read
        the cooperative result unguarded and reduce it with no identity
        fill.  Attention's ':' head dim keeps its in-tile checks, which
        only a tile loop's layout proves true."""
        x = torch.randn(32, 128, device=DEVICE, dtype=torch.float16)
        y = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
        config = helion.Config(block_sizes=[32, 16], num_threads=[32, 0], num_warps=1)
        with self.subTest(kernel="colsum_one_tile"):
            out, code, msl = _run(colsum_one_tile, (x, y), config)
            _close(out, _mm(x, y).sum(0))
            self._launches(code, (32, 1, 1))
            self.assertNotRegex(msl, r"\bmask_\d+")
            self.assertNotIn("? _mpp_mat", msl)
        q, k, v = (
            torch.randn(1, 2, 64, 64, device=DEVICE, dtype=torch.float16)
            for _ in range(3)
        )
        config = helion.Config(block_sizes=[1, 16, 16], num_threads=[1, 16, 16])
        with self.subTest(kernel="attention"):
            (out, _), code, msl = _run(attention_metal, (q, k, v), config)
            _close(out, _attention_ref(q, k, v))
            self._launches(code, (16, 16, 1))
            self.assertNotRegex(msl, r"\bmask_\d+")
            self.assertNotIn("_mask_to", msl)
            self.assertRegex(msl, r"auto _mpp_v_\d+ = _mpp_mat\w*\[")
            self.assertRegex(msl, r"if \(\(\(indices_\d+ - 0\) < 64\)\) \{ _mpp_tile\[")


@_requires_mps
class TestMppUniformLaunchKeepsMasks(_MppCase):
    """What a uniform launch doesn't prove stays checked: an extent that
    doesn't divide its block, a launch with two thread extents, and kernels
    whose free ``hl.arange`` claims threads no strategy owns."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_extent_not_dividing(self) -> None:
        """N = 100 under 16-column tiles and int4_gemm at M = 3 keep those
        masks."""
        x = torch.randn(32, 128, device=DEVICE, dtype=torch.float16)
        y = torch.randn(128, 100, device=DEVICE, dtype=torch.float16)
        b = torch.randn(100, device=DEVICE)
        with self.subTest(kernel="bias_relu"):
            out, code, msl = _run(bias_relu, (x, y, b), _UNIFORM16)
            _close(out, torch.relu(_mm(x, y) + _d(b)))
            self.assertIn("_block_dims=(16, 16, 1)", code)
            self.assertRegex(msl, r"\bindices_\d+ < 100\)")
        lhs, rhs, ref = _int4_args(3, 256, 128)
        for config, launch in _INT4_UNIFORM:
            with self.subTest(kernel="int4_gemm", config=config):
                out, code, msl = _run(int4_gemm, (lhs, rhs), config)
                _close(out, ref)
                self.assertIn(f"_block_dims={launch}", code)
                self.assertRegex(msl, r"\bindices_\d+ < 3\)")

    def test_two_thread_extents(self) -> None:
        """int4_gemm launched as 16 x 8 or 8 x 16 keeps its tile masks and
        its staging write's in-tile checks."""
        config = helion.Config(block_sizes=[128, 16, 16], num_warps=4)
        for m, launch in ((1, (16, 8, 1)), (16, (8, 16, 1))):
            lhs, rhs, ref = _int4_args(m, 256, 128)
            with self.subTest(m=m):
                out, code, msl = _run(int4_gemm, (lhs, rhs), config)
                _close(out, ref)
                self.assertIn(f"_block_dims={launch}", code)
                self.assertIn("? B[", msl)
                self.assertRegex(msl, r"if \(\(\(\(indices_\d+ - offset_\d+\) < 128\)")

    def test_threads_no_strategy_owns(self) -> None:
        """Free ``hl.arange`` rows keep their masks at a launch whose
        strategies share one extent."""
        a = torch.randn(64, 48, device=DEVICE, dtype=torch.float16)
        b = torch.randn(48, 64, device=DEVICE, dtype=torch.float16)
        out, code, _ = _run(free_rows, (a, b, 4, 4), _UNIFORM16)
        _close(out, _mm(a, b))
        self.assertIn("_block_dims=(16, 16, 1)", code)


# --- Staging lane loops dealt out over the threads that share them ---------


@helion.kernel(**S)
def scaled_colon_lhs(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A ':' K lhs computed once per M tile, before the N loop holding the
    op."""
    m, _ = x.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        a = x[tm, :] * 0.5
        for tn in hl.tile(n):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, a, w[:, tn])
    return out


@helion.kernel(**S)
def scaled_colon_rhs(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A ':' K rhs computed once per N tile, before the M loop holding the
    op."""
    m, _ = x.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tn in hl.tile(n):
        b = w[:, tn] * 2.0
        for tm in hl.tile(m):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, x[tm, :], b)
    return out


@helion.kernel(**S)
def gated_lhs(
    x: torch.Tensor, g: torch.Tensor, bias: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """A lhs from a row gate and a column bias, before the N loop."""
    m, _ = x.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        a = (x[tm, :] * g[tm][:, None] + bias[None, :]).to(x.dtype)
        for tn in hl.tile(n):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, a, w[:, tn])
    return out


@helion.kernel(**S)
def two_pins(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A lhs that N's threads share and a rhs that M's threads share, staged
    by one lane loop over the ':' K."""
    m, _ = x.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        a = x[tm, :] * 2.0
        b = w[:, tn] * 0.5
        for _k in hl.tile(1):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            out[tm, tn] = torch.addmm(acc, a, b)
    return out


def _spread_loops(msl: str) -> list[tuple[int, int, int, str]]:
    """Each lane loop that starts at a thread index: its thread axis, extent,
    step and body."""
    loops = []
    for match in re.finditer(
        r"for \(int (\w*lane_\d+) = tid\[(\d)\]; \1 < (\d+); \1 \+= (\d+)\) \{", msl
    ):
        end, depth = match.end(), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(msl[end], 0)
            end += 1
        axis, extent, step = (int(g) for g in match.groups()[1:])
        loops.append((axis, extent, step, msl[match.end() : end]))
    return loops


#: A staging write that an owner test pins, in a lane loop from 0.
_OWNED_LANES = (
    r"for \(int \w*lane_\d+ = 0;[^{]*\{[^}]*_mpp_owner\w*\)\) \{ _mpp_(tile|seed)"
)


@_requires_mps
class TestMppStagingSpread(_MppCase):
    """A lane loop staging an operand that the threads of another block's
    axis all hold, whose writes an owner test gives the thread at 0 of that
    axis, deals its lanes out over the axis instead: thread ``t`` runs lanes
    ``t``, ``t + n``, ... and writes them without the owner test."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def _spreads(self, code: str, msl: str, axis: int, extent: int) -> str:
        """The one dealt lane loop runs over ``extent`` lanes from
        ``tid[axis]`` by the launch's threads on that axis and stages
        without an owner test; return its MSL position."""
        (loop,) = _spread_loops(msl)
        self.assertEqual(loop[:3], (axis, extent, _launch(code)[axis]))
        self.assertRegex(loop[3], r"\{ _mpp_(tile|seed)\w*\[")
        self.assertNotIn("_mpp_owner", loop[3])
        return msl.index(loop[3])

    def test_attention(self) -> None:
        """attention's ``q * qk_scale`` over the ':' head dim, dealt out over
        the N threads, at a sequence length no block divides."""
        bf = torch.bfloat16
        q, k, v = (torch.randn(2, 3, 72, 64, dtype=bf) for _ in range(3))
        scale = 1.0 / math.sqrt(64) * 1.44269504
        scores = (q.double() * scale).to(bf).double() @ k.double().transpose(-1, -2)
        expected = _bf(torch.softmax(scores * math.log(2), -1) @ v.double())
        args = (q.to(DEVICE), k.to(DEVICE), v.to(DEVICE))
        for config in (
            None,
            helion.Config(block_sizes=[1, 32, 64], num_threads=[1, 0, 0], num_warps=8),
            helion.Config(block_sizes=[1, 16, 64], num_threads=[1, 0, 0], num_warps=4),
        ):
            with self.subTest(config=config):
                (out, _), code, msl = _run(attention_metal, args, config)
                _close(out, expected)
                self._spreads(code, msl, 0, 64)

    def test_hoisted_operands(self) -> None:
        """A lhs over 40 lanes, which 16 N threads don't divide, a rhs
        dealt out over the M threads, and a lhs computed from a row gate and
        a column bias, run 20 times."""
        h = torch.float16
        x, w = torch.randn(48, 40, dtype=h), torch.randn(40, 48, dtype=h)
        for config in (
            None,
            helion.Config(block_sizes=[16, 16], num_threads=[0, 16], num_warps=1),
        ):
            with self.subTest(kernel="scaled_colon_lhs", config=config):
                out, code, msl = _run(
                    scaled_colon_lhs, (x.to(DEVICE), w.to(DEVICE)), config
                )
                _close(out, _mm(x * 0.5, w))
                self._spreads(code, msl, 1, 40)
        x, w = torch.randn(40, 64, dtype=h), torch.randn(64, 48, dtype=h)
        for config in (
            None,
            helion.Config(block_sizes=[16, 16], num_threads=[16, 0], num_warps=1),
        ):
            with self.subTest(kernel="scaled_colon_rhs", config=config):
                out, code, msl = _run(
                    scaled_colon_rhs, (x.to(DEVICE), w.to(DEVICE)), config
                )
                _close(out, _mm(x, w * 2.0))
                self._spreads(code, msl, 1, 64)
        x, g, bias = (
            torch.randn(37, 72, dtype=h),
            torch.randn(37, dtype=h),
            torch.randn(72, dtype=h),
        )
        w = torch.randn(72, 40, dtype=h)
        a = (x * g[:, None] + bias[None, :]).to(h)
        config = helion.Config(block_sizes=[32, 32], num_warps=4)
        for _ in range(20):
            with self.subTest(kernel="gated_lhs"):
                args = tuple(t.to(DEVICE) for t in (x, g, bias, w))
                out, code, msl = _run(gated_lhs, args, config)
                _close(out, _mm(a, w))
                self._spreads(code, msl, 1, 72)

    def test_inside_a_lane_loop(self) -> None:
        """A staging loop nested in the M tile's lane loop: ``if_late``'s 64
        lanes over 32 threads, and ``hoist_exp``'s 2 lanes over 8 threads,
        where the threads past the extent run none."""
        bf = torch.bfloat16
        q = torch.randn(48, 64, dtype=bf) * 0.5
        k = torch.randn(40, 64, dtype=bf)
        late = _mm(q * 0.125, k.T)
        late[:, :32] = 0
        for kernel, config, expected, axis, extent in (
            (
                if_late,
                helion.Config(block_sizes=[64, 32], num_threads=[32, 32], num_warps=32),
                late,
                1,
                64,
            ),
            (
                hoist_exp,
                helion.Config(block_sizes=[64, 16], num_threads=[4, 8], num_warps=32),
                _mm(torch.exp(q) * q, k.T),
                2,
                2,
            ),
        ):
            with self.subTest(kernel=kernel.name):
                out, code, msl = _run(kernel, (q.to(DEVICE), k.to(DEVICE)), config)
                _close(out, expected)
                pos = self._spreads(code, msl, axis, extent)
                self.assertTrue(
                    [h for h in _open_loops(msl, pos) if h.startswith("int lane_")]
                )

    def test_keeps_the_owner(self) -> None:
        """Kept on the owner path: a lane loop staging a lhs that N's threads
        share beside a rhs that M's share, which no one axis deals out, and
        attention at a config whose N axis has one thread."""
        h = torch.float16
        x, w = torch.randn(48, 64, dtype=h), torch.randn(64, 40, dtype=h)
        with self.subTest(kernel="two_pins"):
            out, _, msl = _run(two_pins, (x.to(DEVICE), w.to(DEVICE)), None)
            _close(out, _mm(x * 2.0, w * 0.5))
            self.assertRegex(msl, _OWNED_LANES)
            self.assertFalse(_spread_loops(msl))
        q, k, v = (torch.randn(1, 2, 64, 64, device=DEVICE, dtype=h) for _ in range(3))
        config = helion.Config(
            block_sizes=[1, 32, 16], num_threads=[1, 32, 1], num_warps=1
        )
        with self.subTest(kernel="attention"):
            (out, _), code, msl = _run(attention_metal, (q, k, v), config)
            _close(out, _attention_ref(q, k, v))
            self.assertIn("_block_dims=(32, 1, 1)", code)
            self.assertRegex(msl, _OWNED_LANES)
            self.assertFalse(_spread_loops(msl))


_SPREAD_OWNER = "_mpp_owner = tid[0] == 0 and tid[2] == 0\n"
_SPREAD_VALUE = "v_1 = float(slice_lane_0) * 2.0"
_SPREAD_STAGE = (
    "_helion_mpp_stage(_mpp_tile, tid[1] * 16 + slice_lane_0, v_1,"
    " slice_lane_0 < 16 and _mpp_owner, (0, 2))"
)


def _spread_body(
    pre: str = "",
    run: tuple[str, ...] = (_SPREAD_VALUE, _SPREAD_STAGE),
    extent: str = "16",
    wrap: str | None = None,
) -> str:
    """A kernel body whose lane loop holds the staging statements ``run``
    beside a K loop holding an op, after ``pre``, under ``wrap``."""
    loop = "\n".join(
        [
            f"for slice_lane_0 in range({extent}):",
            *(f"    {s}" for s in run),
            "    for offset_3 in tl.range(0, 64, 16):",
            "        acc = _helion_mpp_run(0)",
        ]
    )
    if wrap is not None:
        loop = f"{wrap}\n{textwrap.indent(loop, '    ')}"
    return f"{_SPREAD_OWNER}{pre}{loop}\n"


def _spread_value(scale: str) -> tuple[str, str]:
    return (f"v_1 = float(slice_lane_0) * {scale}", _SPREAD_STAGE)


def _spread_owner(test: str) -> str:
    """A body whose staging write's last term is a name bound to ``test``."""
    stage = _SPREAD_STAGE.replace("_mpp_owner,", "_mpp_owner_9,")
    return _spread_body(f"_mpp_owner_9 = {test}\n", (_SPREAD_VALUE, stage))


class TestMppStagingSpreadRule(unittest.TestCase):
    """``_run_lane_loops_once`` on hand-built bodies at threadgroup
    (8, 32, 1), where the staging writes pin ``tid[0]`` (8 threads) and
    ``tid[2]``: which lane loops are dealt out over ``tid[0]``."""

    def _rewrite(
        self, source: str, tg_size: tuple[int, int, int] = (8, 32, 1)
    ) -> list[ast.stmt]:
        return mpp_matmul._run_lane_loops_once(
            ast.parse(source).body,
            set(),
            {},
            frozenset({"_mpp_tile", "_mpp_tile_1"}),
            tg_size,
        )

    def _lanes(self, source: str, tg_size: tuple[int, int, int] = (8, 32, 1)) -> str:
        """The header of the staging lane loop after the rewrite."""
        out = self._rewrite(source, tg_size)
        text = ast.unparse(ast.fix_missing_locations(ast.Module(out, [])))
        (header,) = re.findall(r"for slice_lane_0 in range\(.*\):", text)
        return header

    def test_deals_lanes(self) -> None:
        loop = self._rewrite(_spread_body())[1]
        self.assertEqual(
            ast.unparse(ast.fix_missing_locations(loop)).splitlines(),
            [
                "for slice_lane_0 in range(tid[0], 16, 8):",
                "    v_1 = float(slice_lane_0) * 2.0",
                (
                    "    _helion_mpp_stage(_mpp_tile, tid[1] * 16 + slice_lane_0, v_1,"
                    " slice_lane_0 < 16, ())"
                ),
            ],
        )
        self.assertEqual(
            self._lanes(_spread_body(), (2, 32, 1)),
            "for slice_lane_0 in range(tid[0], 16, 2):",
        )

    def test_uniform_inputs_deal_lanes(self) -> None:
        """Names and control every thread of the threadgroup computes alike,
        a write index through a name, and a per-element name of an element
        loop, which declares its own variable."""
        for name, source in {
            "uniform_if_assign": _spread_body(
                "s = 1.0\nif tgid[0] < 4:\n    s = 2.0\n", _spread_value("s")
            ),
            "uniform_augassign": _spread_body(
                "s = 1.0\ns += 2.0\n", _spread_value("s")
            ),
            "under_uniform_if": _spread_body(wrap="if tgid[0] < 4:"),
            "under_uniform_for": _spread_body(wrap="for i in tl.range(0, 4):"),
            "index_through_a_name": _spread_body(
                run=(
                    "i_1 = tid[1] * 16 + slice_lane_0",
                    _SPREAD_VALUE,
                    _SPREAD_STAGE.replace("tid[1] * 16 + slice_lane_0,", "i_1,"),
                )
            ),
            "element_loop_bind": _spread_body(
                "s = 2.0\nfor _e in _helion_mpp_elements(acc):\n    s = float(tid[0])\n",
                _spread_value("s"),
            ),
        }.items():
            with self.subTest(name=name):
                self.assertEqual(
                    self._lanes(source), "for slice_lane_0 in range(tid[0], 16, 8):"
                )

    def _spread_alone(self, *run: str) -> None:
        """Assert ``_Replicas.spread`` keeps a lane loop holding ``run``."""
        body = ast.parse(
            _SPREAD_OWNER
            + "for slice_lane_0 in range(16):\n"
            + "".join(f"    {s}\n" for s in run)
        ).body
        loop = body[1]
        assert isinstance(loop, ast.For)
        replicas = mpp_matmul._Replicas(body, (8, 32, 1), frozenset({"_mpp_tile"}))
        self.assertIs(replicas.spread(loop), loop)

    def test_keeps_a_collective(self) -> None:
        """A collective in the run must run as often on every thread, even
        one whose result no write reads."""
        self._spread_alone(
            "s_1 = helion_red.simd_sum(float(slice_lane_0))",
            _SPREAD_VALUE,
            _SPREAD_STAGE,
        )

    def test_keeps_a_carry_to_a_later_lane(self) -> None:
        """A name a lane reads before a statement assigns it holds the
        previous lane's value.  ``_distribute_lane_loop`` hands such a loop
        to the jam, so only a direct call reaches ``spread`` with it."""
        self._spread_alone(*_spread_value("s"), "s = float(slice_lane_0)")

    def test_keeps(self) -> None:
        """Kept: one case per condition of the rule, per form that binds a
        name the run reads, per owner check and per kind of thread-varying
        control around the loop."""
        stage = _SPREAD_STAGE
        tile_1 = stage.replace("_mpp_tile,", "_mpp_tile_1,")
        cases = {
            "range_not_constant": _spread_body(extent="n"),
            "range_with_start": _spread_body(extent="0, 16"),
            "reduction_in_run": _spread_body(
                run=(
                    "s_1 = helion_red.simd_sum(float(slice_lane_0))",
                    "v_1 = s_1 * 2.0",
                    stage,
                )
            ),
            "threadgroup_read_in_run": _spread_body(
                run=("v_1 = _mpp_tile_1[slice_lane_0] * 2.0", stage)
            ),
            "device_store_in_run": _spread_body(
                run=(_SPREAD_VALUE, "tl.store(out + slice_lane_0, v_1, None)", stage)
            ),
            **{
                f"reduction_in_write_{arg}": _spread_body(
                    run=(_SPREAD_VALUE, stage.replace(old, new))
                )
                for arg, old, new in (
                    (
                        "index",
                        "slice_lane_0,",
                        "slice_lane_0 + helion_red.simd_max(0),",
                    ),
                    ("value", " v_1,", " helion_red.simd_max(v_1),"),
                    ("predicate", " < 16 ", " < helion_red.simd_max(16) "),
                )
            },
            "lane_jammed_name": _SPREAD_OWNER
            + "for slice_lane_0 in range(16):\n"
            + "    for lane_1 in range(2):\n"
            + "        t_1 = foo(tid[0], lane_1)\n"
            + "        for offset_3 in tl.range(0, 64, 16):\n"
            + "            acc = _helion_mpp_run(0)\n"
            + "        v_1 = float(slice_lane_0) * t_1\n"
            + f"        {stage}\n",
            "different_pins": _spread_body(
                "_mpp_owner_1 = tid[0] == 0\n",
                (
                    _SPREAD_VALUE,
                    stage,
                    tile_1.replace("_mpp_owner, (0, 2)", "_mpp_owner_1, (0,)"),
                ),
            ),
            "reads_tid": _spread_body(
                run=("v_1 = float(slice_lane_0 + tid[0])", stage)
            ),
            "index_without_lane": _spread_body(
                run=(_SPREAD_VALUE, stage.replace(" + slice_lane_0,", ","))
            ),
            "index_rebound": _spread_body(
                run=(
                    "i_1 = tid[1] * 16 + slice_lane_0",
                    "i_1 = tid[1] * 16",
                    _SPREAD_VALUE,
                    stage.replace("tid[1] * 16 + slice_lane_0,", "i_1,"),
                )
            ),
            "self_carry": _spread_body(
                "s = 0.0\n",
                ("s = s + float(slice_lane_0)", stage.replace(" v_1,", " s,")),
            ),
            "self_carry_two": _spread_body(
                "s = 0.0\n",
                (
                    "v_1 = float(slice_lane_0)",
                    "s = s + v_1",
                    stage.replace(" v_1,", " s,"),
                ),
            ),
            **{
                name: _spread_body(pre, _spread_value(scale))
                for name, pre, scale in (
                    ("if_tid", "s = 1.0\nif tid[0] < 4:\n    s = 2.0\n", "s"),
                    ("for_tid", "s = 1.0\nfor i in range(tid[0]):\n    s = 2.0\n", "s"),
                    ("while_tid", "s = 1.0\nwhile tid[0] < 4:\n    s = 2.0\n", "s"),
                    ("walrus_in_test", "if (s := float(tid[0])) > 0:\n    pass\n", "s"),
                    ("augassign", "s = 1.0\ns += float(tid[0])\n", "s"),
                    ("tuple_target", "s, t = (float(tid[0]), 1.0)\n", "s"),
                    ("annassign", "s: float = float(tid[0])\n", "s"),
                    (
                        "for_target",
                        "for j in range(tid[0], tid[0] + 1):\n    pass\n",
                        "float(j)",
                    ),
                    ("whole_tid", "s = helper(tid)\n", "s"),
                    ("coop_value", "s = _helion_mpp_acc(0)\n", "s"),
                    ("subscript_store", "arr[0] = float(tid[0])\n", "arr[0]"),
                    ("attribute_store", "o.x = float(tid[0])\n", "o.x"),
                    ("nested_store", "o.x[0][0] = float(tid[0])\n", "o.x[0][0]"),
                    (
                        "bound_later_in_a_loop",
                        "s = 1.0\nfor i in tl.range(0, 4):\n    u = s\n    s = float(tid[0])\n",
                        "u",
                    ),
                    ("tid_named_axis", "j = 0\n", "tid[j]"),
                )
            },
            "last_term_not_a_name": _spread_body(
                run=(_SPREAD_VALUE, stage.replace("_mpp_owner,", "tgid[0] < 12,"))
            ),
            "owner_axes_mismatch": _spread_owner("tid[0] == 0"),
            "owner_tests_tgid": _spread_owner("tgid[0] == 0 and tid[2] == 0"),
            "owner_nonzero": _spread_owner("tid[0] == 1 and tid[2] == 0"),
            "owner_not_equality": _spread_owner("tid[0] <= 0 and tid[2] == 0"),
            "owner_chained": _spread_owner("tid[0] == 0 == tgid[0] and tid[2] == 0"),
            "owner_named_axis": _spread_owner("tid[j] == 0 and tid[2] == 0"),
            "owner_named_value": _spread_owner("tid[0] == k and tid[2] == 0"),
            "owner_rebound": _spread_body(
                "if tgid[0] < 1:\n    _mpp_owner = tid[0] == 0 and tid[2] == 0\n"
            ),
            "under_tid_if": _spread_body(wrap="if tid[0] < 4:"),
            "under_tid1_if": _spread_body(wrap="if tid[1] < 4:"),
            "under_tid1_else": _spread_body(wrap="if tid[1] < 4:\n    pass\nelse:"),
            "under_tid_for": _spread_body(wrap="for i in range(tid[1]):"),
        }
        for name, source in cases.items():
            with self.subTest(name=name):
                self.assertNotIn("tid[", self._lanes(source))
        for name, tg_size in (("two_threaded_pins", (8, 32, 4)), ("none", (1, 32, 1))):
            with self.subTest(name=name):
                self.assertNotIn("tid[", self._lanes(_spread_body(), tg_size))


# --- MPP tiles that share storage ------------------------------------------


@helion.kernel(**S)
def mm_softmax_mm(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Per H tile: ``s = x @ w1``, ``p = exp(s - rowmax(s))``, ``acc += p @ w2``."""
    b, m, _ = x.size()
    h = w1.size(2)
    n = hl.specialize(w2.size(2))
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        acc = hl.zeros([tb, tm, n], dtype=torch.float32)
        for th in hl.tile(h):
            s = torch.bmm(x[tb, tm, :], w1[tb, :, th], torch.float32)
            p = torch.exp(s - torch.amax(s, -1)[:, :, None])
            acc = torch.baddbmm(acc, p.to(x.dtype), w2[tb, th, :])
        out[tb, tm, :] = acc
    return out


@helion.kernel(**S)
def two_mats(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Two materialized products in one H step."""
    b, m, _ = x.size()
    h = w1.size(2)
    out = torch.empty([b, m], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        r = hl.zeros([tb, tm], dtype=torch.float32)
        for th in hl.tile(h):
            s1 = torch.bmm(x[tb, tm, :], w1[tb, :, th], torch.float32)
            s2 = torch.bmm(x[tb, tm, :], w2[tb, :, th], torch.float32)
            r = r + torch.amax(s1 * s2, -1)
        out[tb, tm] = r
    return out


@helion.kernel(**S)
def p_twice(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor
) -> torch.Tensor:
    """One staged P read by two ops, before the next H step's scores."""
    b, m, _ = x.size()
    h = w1.size(2)
    n = hl.specialize(w2.size(2))
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        acc1 = hl.zeros([tb, tm, n], dtype=torch.float32)
        acc2 = hl.zeros([tb, tm, n], dtype=torch.float32)
        for th in hl.tile(h):
            s = torch.bmm(x[tb, tm, :], w1[tb, :, th], torch.float32)
            p = torch.exp(s - torch.amax(s, -1)[:, :, None]).to(x.dtype)
            acc1 = torch.baddbmm(acc1, p, w2[tb, th, :])
            acc2 = torch.baddbmm(acc2, p, w3[tb, th, :])
        out[tb, tm, :] = acc1 - 0.5 * acc2
    return out


@helion.kernel(**S)
def mat_then_seed(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """A product seeded from a value computed from an earlier product's
    materialization, in one H step."""
    b, m, _ = x.size()
    h = w1.size(2)
    out = torch.empty([b, m], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([b, m]):
        r = hl.zeros([tb, tm], dtype=torch.float32)
        for th in hl.tile(h):
            s = torch.bmm(x[tb, tm, :], w1[tb, :, th], torch.float32)
            p = torch.exp(s - torch.amax(s, -1)[:, :, None])
            o = torch.baddbmm(p, x[tb, tm, :], w2[tb, :, th])
            r = r + torch.amax(o, -1)
        out[tb, tm] = r
    return out


@helion.kernel(**S)
def two_staged(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Two computed lhs operands, each staged for its own op in one K step,
    with no materialization in the step."""
    m, k = x.size()
    n = w1.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc1 = hl.zeros([tm, tn], dtype=torch.float32)
        acc2 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc1 = torch.addmm(acc1, torch.relu(x[tm, tk]), w1[tk, tn])
            acc2 = torch.addmm(acc2, x[tm, tk] * 0.5 + 1.0, w2[tk, tn])
        out[tm, tn] = acc1 - acc2
    return out


def _h_tiles(h: int, block: int) -> list[slice]:
    return [slice(h0, min(h0 + block, h)) for h0 in range(0, h, block)]


def _softmax_tile(s: torch.Tensor) -> torch.Tensor:
    return torch.exp(s - s.amax(-1, keepdim=True))


def _tiles_declared(msl: str) -> list[tuple[str, str, int]]:
    """The ``(dtype, name, slots)`` of every MPP threadgroup tile."""
    return [
        (dtype, name, int(slots))
        for dtype, name, slots in re.findall(
            r"threadgroup (\w+) (_mpp_\w+)\[(\d+)\];", msl
        )
    ]


def _first_writes_fenced(msl: str, tile: str) -> bool:
    """Whether every run of consecutive Scalar writes to ``tile`` comes right
    after a threadgroup barrier."""
    lines = [line.strip() for line in msl.splitlines()]
    write = re.compile(rf"\{{ {tile}\[")
    return all(
        lines[i - 1] == "threadgroup_barrier(mem_flags::mem_threadgroup);"
        for i, line in enumerate(lines)
        if write.search(line) and not write.search(lines[i - 1])
    )


_ATTENTION_SHARED = (
    helion.Config(block_sizes=[1, 16, 128], num_warps=8, pid_type="xyz"),
    helion.Config(block_sizes=[1, 16, 128], num_warps=4),
)


@_requires_mps
class TestMppTileSharing(unittest.TestCase):
    """MPP threadgroup tiles that are never live together share one tile,
    with a threadgroup barrier before each later user's first access and,
    in a loop, after the last user's last access."""

    def _attention(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(0)
        return tuple(torch.randn(shape, device=DEVICE, dtype=dtype) for _ in range(3))

    def test_attention_scores_and_p_share(self) -> None:
        """The scores' materialization, the ``acc *= alpha`` row vector and
        P's staged tile, all float and written and read within one K step,
        share the scores' tile: P's staging and the vector's come after a
        barrier, and the K step ends with one after the second op.  The
        hoisted bfloat Q tile and the ``acc / l_i`` vector after the loop
        keep their own."""
        args = self._attention((1, 2, 200, 128), torch.bfloat16)
        for config in _ATTENTION_SHARED:
            with self.subTest(config=config):
                (out, _), _, msl = _run(attention_metal, args, config)
                _close(out, _attention_ref(*args))
                self.assertEqual(
                    _tiles_declared(msl),
                    [
                        ("float", "_mpp_mat", 2048),
                        ("bfloat", "_mpp_tile", 2048),
                        ("float", "_mpp_vec_1", 16),
                    ],
                )
                self.assertRegex(
                    msl,
                    r"\{ _mpp_mat\[[^\n]*\] = static_cast<float>\(helion_bf16_rne\(",
                )
                self.assertTrue(_first_writes_fenced(msl, "_mpp_mat"))
                self.assertRegex(
                    msl,
                    r"_mpp_op_1\.run\([^\n]*\n\s*\}\n\s*\}\n"
                    r"(\s*threadgroup_barrier\(mem_flags::mem_threadgroup\);\n){2}"
                    r"\s*m_i = ",
                )

    def test_configs_over_the_budget_fit(self) -> None:
        """Configs whose separate tiles exceed the 32 KB of threadgroup
        memory fit once they share.  In f32 attention, the float Q tile
        hoisted at kernel scope also takes in the ``acc / l_i`` vector after
        the K loop."""
        for name, kernel, args, config, ref, tiles in (
            (
                "bf16",
                attention_metal,
                self._attention((1, 2, 200, 128), torch.bfloat16),
                helion.Config(block_sizes=[1, 32, 128], num_warps=8),
                _attention_ref,
                [
                    ("float", "_mpp_mat", 4096),
                    ("bfloat", "_mpp_tile", 4096),
                    ("float", "_mpp_vec_1", 32),
                ],
            ),
            (
                "bf16_d64",
                attention_metal,
                self._attention((1, 2, 96, 64), torch.bfloat16),
                helion.Config(block_sizes=[1, 64, 64], num_warps=8),
                _attention_ref,
                [
                    ("float", "_mpp_mat", 4096),
                    ("bfloat", "_mpp_tile", 4096),
                    ("float", "_mpp_vec_1", 64),
                ],
            ),
            (
                "f32",
                attention_metal,
                self._attention((1, 2, 200, 128), torch.float32),
                helion.Config(block_sizes=[1, 32, 64], num_warps=8),
                _attention_ref,
                [("float", "_mpp_mat", 2048), ("float", "_mpp_tile", 4096)],
            ),
        ):
            with self.subTest(kernel=name, config=config):
                (out, _), _, msl = _run(kernel, args, config)
                _close(out, ref(*args))
                self.assertEqual(_tiles_declared(msl), tiles)
        torch.manual_seed(0)
        x, w1, w2 = (
            torch.randn(shape, device=DEVICE)
            for shape in ((2, 64, 128), (2, 128, 160), (2, 160, 64))
        )
        config = helion.Config(block_sizes=[1, 32, 128], num_warps=8)
        with self.subTest(kernel="mm_softmax_mm", config=config):
            out, _, msl = _run(mm_softmax_mm, (x, w1, w2), config)
            ref = torch.zeros(2, 64, 64, dtype=torch.float64)
            for hs in _h_tiles(160, 128):
                ref += _softmax_tile(_d(x) @ _d(w1)[:, :, hs]) @ _d(w2)[:, hs]
            _close(out, ref)
            self.assertEqual(_tiles_declared(msl), [("float", "_mpp_mat", 4096)])

    def test_staged_seed_shares(self) -> None:
        """A seed staged from values computed from the scores is read by
        ``_helion_mpp_acc`` under its tile's name, a string: the read counts in its
        live range, and the name follows the tile into shared storage."""
        torch.manual_seed(0)
        bf = torch.bfloat16
        x, w1, w2 = (
            torch.randn(shape, device=DEVICE, dtype=bf)
            for shape in ((2, 64, 64), (2, 64, 96), (2, 64, 96))
        )
        config = helion.Config(block_sizes=[1, 16, 32], num_warps=4)
        out, _, msl = _run(mat_then_seed, (x, w1, w2), config)
        ref = torch.zeros(2, 64, dtype=torch.float64)
        for hs in _h_tiles(96, 32):
            seed = _softmax_tile(_d(x) @ _d(w1)[:, :, hs])
            ref += (seed + _d(x) @ _d(w2)[:, :, hs]).amax(-1)
        _close(out, ref)
        self.assertEqual(_tiles_declared(msl), [("float", "_mpp_mat", 512)])
        self.assertNotIn("_mpp_seed", msl)
        self.assertRegex(msl, r"\*_it = _mpp_mat\[_e_m \* 32 \+ _e_n\];")

    def test_staged_runs_fence_a_home(self) -> None:
        """A K step that materializes nothing still shares: the ops reading
        its two staged operands fence it, so the second operand takes the
        first one's tile.  Each run is repeated 20 times."""
        torch.manual_seed(0)
        x, w1, w2 = (
            torch.randn(shape, device=DEVICE)
            for shape in ((40, 200), (200, 56), (200, 56))
        )
        config = helion.Config(block_sizes=[32, 32, 32], num_warps=4)
        ref = torch.relu(_d(x)) @ _d(w1) - (0.5 * _d(x) + 1.0) @ _d(w2)
        out, _, msl = _run(two_staged, (x, w1, w2), config)
        _close(out, ref)
        self.assertEqual(
            _tiles_declared(msl),
            [
                ("float", "_mpp_mat", 1024),
                ("float", "_mpp_mat_1", 1024),
                ("float", "_mpp_tile", 1024),
            ],
        )
        (k_step,) = re.findall(
            r"\n    for \(int offset_2 .*?\n    \}\n", msl, re.DOTALL
        )
        self.assertNotIn("_mpp_mat", k_step)
        bound = two_staged.bind((x, w1, w2))
        bound.set_config(config)
        for _ in range(20):
            _close(bound(x, w1, w2), ref)

    def test_element_types_never_share(self) -> None:
        """f16 attention stages P as half: it keeps its own tile beside the
        float scores, so a 32-row tile stays over the budget."""
        args = self._attention((1, 2, 200, 128), torch.float16)
        config = helion.Config(block_sizes=[1, 32, 128], num_warps=8)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "reduction scratch and MPP staging need 32904 bytes of threadgroup memory",
        ):
            _run(attention_metal, args, config)

    def test_repeated_runs(self) -> None:
        """Each shared config, run 20 times: attention, two materializations
        in one step, and one staged P read by two ops."""
        args = self._attention((1, 2, 200, 128), torch.bfloat16)
        cases: list[
            tuple[helion.Kernel, tuple[torch.Tensor, ...], helion.Config, str]
        ] = [
            (attention_metal, args, config, "_mpp_tile_1")
            for config in _ATTENTION_SHARED
        ]
        torch.manual_seed(0)
        bf = torch.bfloat16
        x, w1, w2 = (
            torch.randn(shape, device=DEVICE, dtype=bf)
            for shape in ((2, 48, 64), (2, 64, 200), (2, 64, 200))
        )
        w3, w4 = (torch.randn(2, 200, 64, device=DEVICE, dtype=bf) for _ in range(2))
        for config in (
            helion.Config(block_sizes=[1, 16, 64], num_warps=4),
            helion.Config(block_sizes=[1, 32, 64], num_warps=8),
        ):
            cases += [
                (two_mats, (x, w1, w2), config, "_mpp_mat_1"),
                (p_twice, (x, w1, w3, w4), config, "_mpp_tile_1"),
            ]

        def reference(kernel: helion.Kernel, bh: int) -> torch.Tensor:
            if kernel is attention_metal:
                return _attention_ref(*args)
            s = _d(x) @ _d(w1)
            if kernel is two_mats:
                s = s * (_d(x) @ _d(w2))
                return sum(s[:, :, hs].amax(-1) for hs in _h_tiles(200, bh))
            acc1 = acc2 = torch.zeros(2, 48, 64, dtype=torch.float64)
            for hs in _h_tiles(200, bh):
                p = _bf(_softmax_tile(s[:, :, hs]))
                acc1 = acc1 + p @ _d(w3)[:, hs]
                acc2 = acc2 + p @ _d(w4)[:, hs]
            return acc1 - 0.5 * acc2

        for kernel, kargs, config, merged in cases:
            with self.subTest(kernel=kernel.name, config=config):
                ref = reference(kernel, config.config["block_sizes"][2])
                _, _, msl = _run(kernel, kargs, config)
                self.assertNotIn(merged, msl)
                bound = kernel.bind(kargs)
                bound.set_config(config)
                for _ in range(20):
                    out = bound(*kargs)
                    _close(out[0] if isinstance(out, tuple) else out, ref)


#: An op whose staged operand ``{t}`` the walker fences with barriers.
_STAGED_RUN = "acc = _helion_mpp_run(acc, {t}, v, spaces=[1, 0])"
_TILES = [("A", "float", 1024), ("B", "float", 1024), ("H", "bfloat", 1024)]


def _share(
    source: str, tiles: list[tuple[str, str, int | str]] | None = None
) -> tuple[str, list[tuple[str, str, int | str]]]:
    """``share_tiles`` on the body ``source``: the body it returns, unparsed,
    and the declarations it keeps."""
    body = ast.parse(textwrap.dedent(source)).body
    body, kept = tile_sharing.share_tiles(body, list(tiles or _TILES))
    return ast.unparse(ast.fix_missing_locations(ast.Module(body, []))), kept


def _unparse(source: str) -> str:
    return ast.unparse(ast.parse(textwrap.dedent(source)))


class TestTileSharingRules(unittest.TestCase):
    """``tile_sharing.share_tiles`` on hand-built bodies: which tiles may
    share, how groups form, and where barriers go."""

    def _shares(self, source: str, expected: str) -> None:
        body, kept = _share(source)
        self.assertEqual(body, _unparse(expected))
        self.assertEqual(kept, [("A", "float", 1024), ("H", "bfloat", 1024)])

    def _keeps(self, source: str) -> None:
        body, kept = _share(source)
        self.assertEqual(body, _unparse(source))
        self.assertEqual(kept, _TILES)

    def test_disjoint_tiles_in_a_loop(self) -> None:
        """B takes A's storage, after a barrier, and the loop body ends with
        one before the next iteration writes A again."""
        self._shares(
            f"""
            for k in range(4):
                _helion_mpp_materialize(qk, A, 64)
                s = A[i]
                _helion_mpp_stage(B, i, s, True, ())
                {_STAGED_RUN.format(t="B")}
            """,
            f"""
            for k in range(4):
                _helion_mpp_materialize(qk, A, 64)
                s = A[i]
                _helion_mpp_barrier()
                _helion_mpp_stage(A, i, s, True, ())
                {_STAGED_RUN.format(t="A")}
                _helion_mpp_barrier()
            """,
        )

    def test_disjoint_tiles_at_kernel_scope(self) -> None:
        """At kernel scope a home needs no fenced statement, and no barrier
        follows the last user: there is no next iteration."""
        self._shares(
            """
            _helion_mpp_stage(A, i, 1.0, True, ())
            s = A[i]
            _helion_mpp_stage(B, i, s, True, ())
            t = B[i]
            """,
            """
            _helion_mpp_stage(A, i, 1.0, True, ())
            s = A[i]
            _helion_mpp_barrier()
            _helion_mpp_stage(A, i, s, True, ())
            t = A[i]
            """,
        )

    def test_home_under_an_if(self) -> None:
        """A home under an ``if`` gets a barrier after its last user only
        inside a loop."""
        for loop in (False, True):
            outer = "for k in range(4):" if loop else "if d:"
            with self.subTest(loop=loop):
                self._shares(
                    f"""
                    {outer}
                        if c:
                            _helion_mpp_materialize(qk, A, 64)
                            s = A[i]
                            _helion_mpp_stage(B, i, s, True, ())
                            {_STAGED_RUN.format(t="B")}
                    """,
                    f"""
                    {outer}
                        if c:
                            _helion_mpp_materialize(qk, A, 64)
                            s = A[i]
                            _helion_mpp_barrier()
                            _helion_mpp_stage(A, i, s, True, ())
                            {_STAGED_RUN.format(t="A")}
                            {"_helion_mpp_barrier()" if loop else ""}
                    """,
                )

    def test_staged_runs_fence_a_home(self) -> None:
        """An op whose operand is staged fences its home, as a
        materialization does: B takes A's storage in a loop body that
        materializes nothing."""
        self._shares(
            f"""
            for k in range(4):
                _helion_mpp_stage(A, i, 1.0, True, ())
                {_STAGED_RUN.format(t="A")}
                _helion_mpp_stage(B, i, 2.0, True, ())
                {_STAGED_RUN.format(t="B")}
            """,
            f"""
            for k in range(4):
                _helion_mpp_stage(A, i, 1.0, True, ())
                {_STAGED_RUN.format(t="A")}
                _helion_mpp_barrier()
                _helion_mpp_stage(A, i, 2.0, True, ())
                {_STAGED_RUN.format(t="A")}
                _helion_mpp_barrier()
            """,
        )

    def test_storage_is_the_first_users_tile(self) -> None:
        """A group's storage is the tile its first access names, whatever
        the names' order: B, written first, keeps its storage and A takes
        it, after a barrier before A's write."""
        body, kept = _share(
            f"""
            for k in range(4):
                _helion_mpp_materialize(qk, B, 64)
                s = B[i]
                _helion_mpp_stage(A, i, s, True, ())
                {_STAGED_RUN.format(t="A")}
            """
        )
        self.assertEqual(
            body,
            _unparse(
                f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, B, 64)
                    s = B[i]
                    _helion_mpp_barrier()
                    _helion_mpp_stage(B, i, s, True, ())
                    {_STAGED_RUN.format(t="B")}
                    _helion_mpp_barrier()
                """
            ),
        )
        self.assertEqual(kept, [("B", "float", 1024), ("H", "bfloat", 1024)])

    def test_accesses_in_a_lane_loop(self) -> None:
        """Accesses in a ``for`` inside the home count at that ``for``, whose
        body may run no iteration."""
        self._shares(
            f"""
            for k in range(4):
                _helion_mpp_materialize(qk, A, 64)
                s = A[i]
                for lane in range(tid[0], 64, 8):
                    _helion_mpp_stage(B, lane, s, True, ())
                {_STAGED_RUN.format(t="B")}
            """,
            f"""
            for k in range(4):
                _helion_mpp_materialize(qk, A, 64)
                s = A[i]
                _helion_mpp_barrier()
                for lane in range(tid[0], 64, 8):
                    _helion_mpp_stage(A, lane, s, True, ())
                {_STAGED_RUN.format(t="A")}
                _helion_mpp_barrier()
            """,
        )

    def test_a_staged_seed_is_read_by_name(self) -> None:
        """``_helion_mpp_acc`` reads its seed tile by name, a string: the read ends
        B's live range, and the string is renamed with the tile."""
        acc = "acc = _helion_mpp_acc(0, 16, 16, 16, (), 0, 0, None, [], 1, 0, 16, 16, {t!r})"
        self._shares(
            f"""
            for k in range(4):
                _helion_mpp_materialize(qk, A, 64)
                s = A[i]
                _helion_mpp_stage(B, i, s, True, ())
                {acc.format(t="B")}
            """,
            f"""
            for k in range(4):
                _helion_mpp_materialize(qk, A, 64)
                s = A[i]
                _helion_mpp_barrier()
                _helion_mpp_stage(A, i, s, True, ())
                {acc.format(t="A")}
                _helion_mpp_barrier()
            """,
        )
        self._keeps(
            f"""
            for k in range(4):
                _helion_mpp_stage(B, i, 1.0, True, ())
                _helion_mpp_materialize(qk, A, 64)
                s = A[i]
                {acc.format(t="B")}
            """
        )

    def test_unsharable_tiles_kept(self) -> None:
        """Overlapping live ranges, two homes (one tile read in two sibling
        loops too), a read before the first write (in the same statement too,
        or by an op), an access under an ``if``, a ``while`` or a loop's
        ``else``, a use other than a marker's tile argument, an unfenced home
        in a loop (one fenced only inside a nested loop too), and different
        element types."""
        run = _STAGED_RUN.format(t="B")
        for name, source in {
            "overlap": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    _helion_mpp_stage(B, i, 1.0, True, ())
                    s = A[i]
                    {run}
                """,
            "two_homes": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                t = s
                _helion_mpp_stage(B, i, s, True, ())
                {run}
                """,
            "two_loops": """
                for i in range(2):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    _helion_mpp_materialize(qk, B, 64)
                    t = B[i]
                for j in range(2):
                    u = A[j]
                """,
            "same_statement": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    _helion_mpp_stage(B, i, A[i], True, ())
                    {run}
                """,
            "read_first": f"""
                for k in range(4):
                    s = A[i]
                    _helion_mpp_materialize(qk, A, 64)
                    _helion_mpp_stage(B, i, s, True, ())
                    {run}
                """,
            "run_reads_first": f"""
                for k in range(4):
                    {run}
                    _helion_mpp_stage(B, i, 1.0, True, ())
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                """,
            "under_if": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    if c:
                        _helion_mpp_stage(B, i, s, True, ())
                    {run}
                """,
            "under_while": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    while c:
                        _helion_mpp_stage(B, i, s, True, ())
                    {run}
                """,
            "loop_else": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    for j in range(4):
                        pass
                    else:
                        _helion_mpp_stage(B, i, s, True, ())
                    {run}
                """,
            "other_use": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = helper(A)
                    _helion_mpp_stage(B, i, s, True, ())
                    {run}
                """,
            "marker_value": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    _helion_mpp_stage(B, i, s, B, ())
                    {run}
                """,
            "mat_source": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    _helion_mpp_materialize(B, B, 64)
                    {run}
                """,
            "index": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = X[A]
                    _helion_mpp_stage(B, i, s, True, ())
                    {run}
                """,
            "seed_elsewhere": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    _helion_mpp_stage(B, i, s, True, ())
                    acc = _helion_mpp_acc(0, 16, 16, 16, (), 0, 0, None, [], 1, 0, 16, "B", None)
                    {run}
                """,
            "read_by_own_write": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    _helion_mpp_stage(B, i, B[j], True, ())
                    {run}
                """,
            "read_in_later_loop_else": f"""
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    _helion_mpp_stage(B, i, s, True, ())
                    {run}
                    for j in range(4):
                        pass
                    else:
                        t = A[i]
                """,
            "keyword_use": """
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    _helion_mpp_stage(B, i, s, True, ())
                    acc = _helion_mpp_run(acc, a, v, spaces=[1, 0], lhs=B)
                """,
            "unfenced_loop": """
                for k in range(4):
                    _helion_mpp_stage(A, i, 1.0, True, ())
                    s = A[i]
                    _helion_mpp_stage(B, i, s, True, ())
                    acc = _helion_mpp_run(acc, B, v, spaces=[0, 0])
                """,
            "fenced_below": """
                for k in range(4):
                    _helion_mpp_stage(A, i, 1.0, True, ())
                    s = A[i]
                    for j in range(2):
                        acc = _helion_mpp_run(acc, a, v, spaces=[1, 0])
                    _helion_mpp_stage(B, i, s, True, ())
                    t = B[i]
                """,
            "element_types": """
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    _helion_mpp_stage(H, i, s, True, ())
                    acc = _helion_mpp_run(acc, H, v, spaces=[1, 0])
                """,
        }.items():
            with self.subTest(name=name):
                self._keeps(source)

    def test_reduction_scratch_never_shares(self) -> None:
        """Reduction scratch, sized per site, keeps its own storage."""
        source = f"""
            for k in range(4):
                _helion_mpp_materialize(qk, A, 64)
                s = A[i]
                _helion_mpp_stage(B, i, s, True, ())
                {_STAGED_RUN.format(t="B")}
            """
        tiles: list[tuple[str, str, int | str]] = [
            ("A", "float", 1024),
            ("B", "float", "_red_site"),
        ]
        body, kept = _share(source, tiles)
        self.assertEqual(body, _unparse(source))
        self.assertEqual(kept, tiles)

    def test_largest_saving_first(self) -> None:
        """A can share with B or C, which overlap each other: it takes B,
        the larger, which also sets the storage's slots.  The saving is the
        smaller tile's slots: C, larger than A and B, which overlap each
        other, takes B, saving 1024 slots rather than A's 512."""
        source = """
            for k in range(4):
                _helion_mpp_materialize(qk, A, 64)
                s = A[i]
                _helion_mpp_stage(B, i, s, True, ())
                _helion_mpp_stage(C, i, s, True, ())
                acc = _helion_mpp_run(acc, B, C, spaces=[1, 1])
            """
        tiles: list[tuple[str, str, int | str]] = [
            ("A", "float", 512),
            ("B", "float", 1024),
            ("C", "float", 64),
        ]
        body, kept = _share(source, tiles)
        self.assertEqual(kept, [("A", "float", 1024), ("C", "float", 64)])
        self.assertEqual(
            body,
            _unparse(
                """
                for k in range(4):
                    _helion_mpp_materialize(qk, A, 64)
                    s = A[i]
                    _helion_mpp_barrier()
                    _helion_mpp_stage(A, i, s, True, ())
                    _helion_mpp_stage(C, i, s, True, ())
                    acc = _helion_mpp_run(acc, A, C, spaces=[1, 1])
                    _helion_mpp_barrier()
                """
            ),
        )
        source = """
            for k in range(4):
                _helion_mpp_materialize(qk, A, 64)
                _helion_mpp_materialize(qk, B, 64)
                s = A[i] + B[i]
                _helion_mpp_materialize(qk, C, 64)
                u = C[i]
            """
        tiles = [("A", "float", 512), ("B", "float", 1024), ("C", "float", 2048)]
        _, kept = _share(source, tiles)
        self.assertEqual(kept, [("A", "float", 512), ("B", "float", 2048)])


if __name__ == "__main__":
    unittest.main()
