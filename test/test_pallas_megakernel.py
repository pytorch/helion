"""Tests for the Pallas megakernel: several top-level loops in one program."""

from __future__ import annotations

import ast
import contextlib
import itertools
import re
import sys
import types
from typing import TYPE_CHECKING
from typing import Any
from unittest import mock

import torch

import helion
from helion import exc
from helion._compiler.pallas import megakernel
from helion._compiler.pallas.lane_dense import lane_dense_perm
from helion._compiler.pallas.lane_dense import tpu_default_layout
from helion._compiler.tile_dependency import StorageRole
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessPallas
import helion.language as hl
from helion.runtime.pallas import launcher

if TYPE_CHECKING:
    from collections.abc import Iterator

_JAX_FN = helion.OutputCodeOptions(allow_helion_deps=False, jax_fn=True)

# XLA's default layouts of TPU arrays, probed on TPU7x with jax 0.10: the dim
# order (major to minor) | the dtypes probed | the shapes laid out that way.
_TPU_DEFAULT_LAYOUTS = """
1 0 | bf16 f32 i8 | 5120x1 5120x2 5120x4 5120x6 5120x8 5120x16 5120x32 5120x64
    5120x96 5120x130 5120x192 130x6 100x6 6x4 64x6 128x6 256x6
0 1 | bf16 f32 i8 | 5120x127 5120x128 5120x256
0 2 1 | bf16 f32 i8 | 6x5120x6 6x1280x4 6x5120x64
1 3 0 2 | bf16 f32 i8 | 8x6x5120x6
1 2 0 | bf16 i8 | 5120x1x6
2 1 0 | f32 | 5120x1x6
2 1 0 | bf16 i8 | 1280x4x1
1 2 0 | f32 | 1280x4x1
0 2 1 | bf16 | 1x5120x6
2 0 1 | f32 | 1x5120x6
0 2 1 | bf16 f32 | 1x1280x4 1x5120x64 1x2048x8 2x1280x4 2x5120x64 2x2048x8
    3x5120x6 3x1280x4 3x5120x64 3x2048x8 4x1280x4 4x5120x64 4x2048x8 6x2048x8
    7x1280x4 7x5120x64 7x2048x8 8x1280x4 8x5120x64 8x2048x8 12x5120x6 12x1280x4
    12x5120x64 12x2048x8 16x1280x4 16x5120x64 16x2048x8 24x1280x4 24x5120x64
    24x2048x8 48x1280x4 48x5120x64 48x2048x8 64x1280x4 64x5120x64 64x2048x8
2 0 1 | bf16 f32 | 2x5120x6 4x5120x6 7x5120x6 8x5120x6 16x5120x6 24x5120x6
    48x5120x6 64x5120x6
1 0 | bf16 f32 | 48x6
0 1 | bf16 f32 | 48x128 6x6 64x5120
0 1 2 3 | bf16 f32 | 16x1x2048x256 48x6x128x128
0 3 1 2 | bf16 f32 | 48x2x5120x6
0 1 2 | bf16 f32 | 6x48x5120
1 3 0 2 | bf16 f32 | 4x3x5120x6
"""


@helion.kernel(backend="pallas", static_shapes=True)
def two_independent_roots(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.empty_like(x)
    b = torch.empty_like(y)
    for tile in hl.tile(x.size()):
        a[tile] = x[tile] + 1.0
    for tile in hl.tile(y.size()):
        b[tile] = y[tile] * 2.0
    return a, b


@helion.kernel(backend="pallas", static_shapes=True)
def two_dependent_roots(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        tmp[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    # Each row tile reads several row and column tiles of the previous root.
    for tile_m in hl.tile(m):
        row = tmp[tile_m, :]
        out[tile_m, :] = row - row.sum(-1, keepdim=True)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def qwen3_dense_mlp(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_w: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    eps: hl.constexpr = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, h = x.shape
    inter = w_gate.shape[1]
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    normed = torch.empty([m, h], dtype=x.dtype, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        s = x[tm, :].float() + residual[tm, :].float()
        hidden[tm, :] = s
        rms = torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps)
        normed[tm, :] = (s * rms * norm_w[None, :].float()).to(x.dtype)
    for tm, tn in hl.tile([m, inter]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            a = normed[tm, tk]
            g = torch.addmm(g, a, w_gate[tk, tn])
            u = torch.addmm(u, a, w_up[tk, tn])
        gate = torch.nn.functional.silu(g.to(x.dtype).float()).to(x.dtype)
        act[tm, tn] = gate * u.to(x.dtype)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(inter):
            acc = torch.addmm(acc, act[tm, tk], w_down[tk, tn])
        out[tm, tn] = (acc + hidden[tm, tn]).to(x.dtype)
    return out, hidden


@helion.kernel(backend="pallas", static_shapes=True)
def qwen3_dense_mlp_linear_layout(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_w: torch.Tensor,
    w_gate: torch.Tensor,  # [inter, h], like nn.Linear
    w_up: torch.Tensor,  # [inter, h]
    w_down: torch.Tensor,  # [inter, h]
    eps: hl.constexpr = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, h = x.shape
    inter = w_gate.shape[0]
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    normed = torch.empty([m, h], dtype=x.dtype, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        s = x[tm, :].float() + residual[tm, :].float()
        hidden[tm, :] = s
        rms = torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps)
        normed[tm, :] = (s * rms * norm_w[None, :].float()).to(x.dtype)
    for tm, tn in hl.tile([m, inter]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            a = normed[tm, tk]
            g = torch.addmm(g, a, w_gate[tn, tk].T)
            u = torch.addmm(u, a, w_up[tn, tk].T)
        gate = torch.nn.functional.silu(g.to(x.dtype).float()).to(x.dtype)
        act[tm, tn] = gate * u.to(x.dtype)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(inter):
            acc = torch.addmm(acc, act[tm, tk], w_down[tk, tn])
        out[tm, tn] = (acc + hidden[tm, tn]).to(x.dtype)
    return out, hidden


@helion.kernel(backend="pallas", static_shapes=True)
def colsum_scratch(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        tmp[tm, tn] = x[tm, tn] + 1.0
    for tn in hl.tile(n):
        out[tn] = tmp[:, tn].sum(0)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def colmax_input(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    y = torch.empty_like(x)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        y[tm, tn] = x[tm, tn] * 2.0
    for tn in hl.tile(n):
        out[tn] = torch.amax(x[:, tn] + y[:, tn], 0)
    return y, out


@helion.kernel(backend="pallas", static_shapes=True)
def tiled_colsum_scratch(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        tmp[tm, tn] = x[tm, tn] + 1.0
    for tn in hl.tile(n):
        acc = hl.zeros([tn], dtype=torch.float32)
        for tk in hl.tile(m):
            acc = acc + tmp[tk, tn].sum(0)
        out[tn] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def offset_read_scratch(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    tmp = torch.empty_like(x)
    out = torch.zeros_like(x)
    for tm, tn in hl.tile([m, n]):
        tmp[tm, tn] = x[tm, tn] + 1.0
    for tm in hl.tile(8, m):
        out[tm, :] = tmp[tm, :] * 2.0
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def offset_write_scratch(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for tm in hl.tile(0, 8):
        tmp[tm, :] = x[tm, :] * 3.0
    for tm in hl.tile(8, m):
        tmp[tm, :] = x[tm, :] + 1.0
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = tmp[tm, tn] * 2.0
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def barrier_roots(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        tmp[tm, tn] = x[tm, tn] + 1.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = tmp[tm, tn] * 2.0
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def grid_root(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        tmp[tm, tn] = x[tm, tn] + 1.0
    for i in hl.grid(m // 8):
        out[i * 8 : i * 8 + 8, :] = tmp[i * 8 : i * 8 + 8, :] * 2.0
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def inplace_input(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        x[tm, tn] = x[tm, tn] + 1.0
    for tm in hl.tile(m):
        row = x[tm, :]
        out[tm, :] = row - row.sum(-1, keepdim=True)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def int_and_none_subscripts(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    _, m, n = x.size()
    tmp = torch.empty([m, n], dtype=x.dtype, device=x.device)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        tmp[tm, :] = x[0, tm, :] + x[1, tm, :]
    for tm in hl.tile(m):
        out[tm, :] = tmp[tm, :] * w[None, :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def view_of_intermediate(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    tmp = torch.empty_like(x)
    tmp2 = tmp.view(m, n)
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        tmp[tm, tn] = x[tm, tn] + 1.0
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = tmp2[tm, tn] * 2.0
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def split_loops_mlp(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    m, h = x.shape
    inter = w_gate.shape[1]
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty([m, h], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, inter]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            g = torch.addmm(g, x[tm, tk], w_gate[tk, tn])
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            u = torch.addmm(u, x[tm, tk], w_up[tk, tn])
        act[tm, tn] = (torch.nn.functional.silu(g) * u).to(x.dtype)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(inter):
            acc = torch.addmm(acc, act[tm, tk], w_down[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def project_then_glu(
    x: torch.Tensor, w_in: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    m, h = x.shape
    inter = w_gate.shape[1]
    y = torch.empty([m, h], dtype=torch.bfloat16, device=x.device)
    out = torch.empty([m, inter], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = torch.addmm(acc, x[tm, tk].to(torch.bfloat16), w_in[tk, tn])
        y[tm, tn] = acc.to(torch.bfloat16)
    for tm, tn in hl.tile([m, inter]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            a = y[tm, tk]
            g = torch.addmm(g, a, w_gate[tk, tn])
            u = torch.addmm(u, a, w_up[tk, tn])
        out[tm, tn] = torch.nn.functional.silu(g) * u
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def weight_read_by_root(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = w.size(1)
    tmp = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out = torch.empty([n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[tk, tn])
        tmp[tm, tn] = acc
    for tn in hl.tile(n):
        out[tn] = tmp[:, tn].sum(0) + w[:, tn].sum(0)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def list_store_then_stream(x: torch.Tensor, bufs: list[torch.Tensor]) -> torch.Tensor:
    m, h = x.shape
    out = torch.empty([m, h], dtype=torch.float32, device=x.device)
    for tm, th in hl.tile([m, h]):
        bufs[1][tm, th] = x[tm, th] * 2.0
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = torch.addmm(acc, bufs[1][tm, tk], bufs[0][tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def list_atomic_then_stream(x: torch.Tensor, bufs: list[torch.Tensor]) -> torch.Tensor:
    m, h = x.shape
    out = torch.empty([m, h], dtype=torch.float32, device=x.device)
    for tm, th in hl.tile([m, h]):
        hl.atomic_add(bufs[1], [tm, th], x[tm, th] * 2.0)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = torch.addmm(acc, bufs[1][tm, tk], bufs[0][tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def grid_root_stream(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, h = x.shape
    e_count, _, n = w.shape
    y = torch.empty([m, h], dtype=torch.float32, device=x.device)
    out = torch.empty([e_count, m, n], dtype=torch.float32, device=x.device)
    for tm, th in hl.tile([m, h]):
        y[tm, th] = x[tm, th] * 2.0
    for e in hl.grid(e_count):
        acc = hl.zeros([m, n], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = torch.addmm(acc, y[:, tk], w[e, tk, :])
        out[e, :, :] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def glu_then_down(
    x: torch.Tensor, w_gate: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    m, h = x.shape
    inter = w_gate.shape[1]
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty([m, h], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, inter]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            g = torch.addmm(g, x[tm, tk], w_gate[tk, tn])
        act[tm, tn] = g.to(x.dtype)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(inter):
            acc = torch.addmm(acc, act[tm, tk], w_down[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def norm_then_indexed_matmul(
    x: torch.Tensor, norm_w: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    m, h = x.shape
    n = w1.shape[1]
    y = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out = torch.empty([m, 128], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            a = x[tm, tk] * norm_w[tk][None, :]
            acc = torch.addmm(acc, a, w1[tk, tn])
        y[tm, tn] = acc
    for tm in hl.tile(m):
        acc = hl.zeros([tm, 128], dtype=torch.float32)
        for tk in hl.tile(n):
            acc = torch.addmm(acc, y[tm, tk], w2[0, tk, :])
        out[tm, :] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def root_matmul_then_matmul(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    m, h = x.shape
    n = w1.shape[1]
    y = torch.empty([m, n], dtype=x.dtype, device=x.device)
    out = torch.empty([m, h], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        y[tm, tn] = (x[tm, :] @ w1[:, tn]).to(x.dtype)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(n):
            acc = torch.addmm(acc, y[tm, tk], w2[tk, tn])
        out[tm, tn] = acc
    return out


def _streamed_shapes(bound: Any) -> list[torch.Size]:
    model = bound.config_spec.pallas_stream_model
    if model is None:
        return []
    return [load.fake.shape for site in model.sites for load in site.loads]


@helion.kernel(backend="pallas", static_shapes=True)
def mlp_stack(
    x: torch.Tensor,
    norm_w: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    eps: hl.constexpr = 1e-6,
) -> torch.Tensor:
    m, h = x.shape
    layers, _, inter = w_gate.shape
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    normed = torch.empty([m, h], dtype=x.dtype, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for tm in hl.tile(m):
            s = hidden[tm, :]
            rms = torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps)
            normed[tm, :] = (s * rms * norm_w[i, :].float()[None, :]).to(x.dtype)
        for tm, tn in hl.tile([m, inter]):
            g = hl.zeros([tm, tn], dtype=torch.float32)
            u = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = normed[tm, tk]
                g = torch.addmm(g, a, w_gate[i, tk, tn])
                u = torch.addmm(u, a, w_up[i, tk, tn])
            gate = torch.nn.functional.silu(g.to(x.dtype).float()).to(x.dtype)
            act[tm, tn] = gate * u.to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                acc = torch.addmm(acc, act[tm, tk], w_down[i, tk, tn])
            hidden[tm, tn] = acc + hidden[tm, tn]
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def gemv_chain(
    x: torch.Tensor,
    w_in: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w_out: torch.Tensor,
    start: hl.constexpr = 0,
) -> torch.Tensor:
    m, h = x.shape
    layers, _, r = a.shape
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    t = torch.empty([m, r], dtype=x.dtype, device=x.device)
    out = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = torch.addmm(acc, x[tm, tk], w_in[tk, tn])
        hidden[tm, tn] = acc
    for i in range(start, layers):
        for tm, tr in hl.tile([m, r]):
            acc = hl.zeros([tm, tr], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = torch.addmm(acc, hidden[tm, tk].to(x.dtype), a[i, tk, tr])
            t[tm, tr] = acc.to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(r):
                acc = torch.addmm(acc, t[tm, tk], b[i, tk, tn])
            hidden[tm, tn] = hidden[tm, tn] + acc / (i + 1)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = torch.addmm(acc, hidden[tm, tk].to(x.dtype), w_out[tk, tn])
        out[tm, tn] = acc.to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def tp_mlp_stack(
    x: torch.Tensor,
    norm_w: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    peers: torch.Tensor,
    rank: torch.Tensor,
    eps: hl.constexpr = 1e-6,
) -> torch.Tensor:
    """``mlp_stack`` with the intermediate dim sharded over ranks.  Each layer
    all-reduces its partial down projection: push it to every peer, then sum.
    Layers alternate between two receive buffers, so a fast rank never
    overwrites a partial that a peer has not summed yet."""
    m, h = x.shape
    layers, _, inter = w_gate.shape
    world = peers.size(1) + 1
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    normed = torch.empty([m, h], dtype=x.dtype, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    recv = torch.empty([2, world, m, h], dtype=torch.float32, device=x.device)
    out = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for _ in hl.grid(1):
        hl.remote_barrier(peers[0, :])
    for p in range(layers // 2):
        for par in hl.static_range(2):
            for tm in hl.tile(m):
                s = hidden[tm, :]
                rms = torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps)
                w = norm_w[2 * p + par, :].float()[None, :]
                normed[tm, :] = (s * rms * w).to(x.dtype)
            for tm, tn in hl.tile([m, inter]):
                g = hl.zeros([tm, tn], dtype=torch.float32)
                u = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(h):
                    a = normed[tm, tk]
                    g = torch.addmm(g, a, w_gate[2 * p + par, tk, tn])
                    u = torch.addmm(u, a, w_up[2 * p + par, tk, tn])
                gate = torch.nn.functional.silu(g.to(x.dtype).float()).to(x.dtype)
                act[tm, tn] = gate * u.to(x.dtype)
            for tm, tn in hl.tile([m, h]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(inter):
                    acc = torch.addmm(acc, act[tm, tk], w_down[2 * p + par, tk, tn])
                recv[par, rank[0], tm, tn] = acc
            for _ in hl.grid(1):
                me = rank[0]
                for j in hl.tile(world - 1, block_size=1):
                    copy = hl.make_async_remote_copy(
                        recv,
                        [par, me],
                        peers[0, j.begin],
                        dst=recv,
                        dst_index=[par, me],
                    )
                    copy.start()
                    if j.begin == world - 2:
                        for _w in hl.static_range(world - 1):
                            copy.wait()
            for tm, tn in hl.tile([m, h]):
                hidden[tm, tn] = hidden[tm, tn] + torch.sum(recv[par, :, tm, tn], 0)
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def tp_gemv_chain(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    peers: torch.Tensor,
    rank: torch.Tensor,
) -> torch.Tensor:
    """A low-rank residual chain with the rank dim sharded: each layer pushes
    its partial to every peer and sums the partials of all ranks.  The receive
    buffer alternates with the layer's parity."""
    m, h = x.shape
    layers, _, r = a.shape
    world = peers.size(1) + 1
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    t = torch.empty([m, r], dtype=x.dtype, device=x.device)
    recv = torch.empty([2, world, m, h], dtype=torch.float32, device=x.device)
    out = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, h]):
        hidden[tm, tn] = x[tm, tn].float()
    for _ in hl.grid(1):
        hl.remote_barrier(peers[0, :])
    for i in range(layers):
        for tm, tr in hl.tile([m, r]):
            acc = hl.zeros([tm, tr], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = torch.addmm(acc, hidden[tm, tk].to(x.dtype), a[i, tk, tr])
            t[tm, tr] = acc.to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(r):
                acc = torch.addmm(acc, t[tm, tk], b[i, tk, tn])
            recv[i % 2, rank[0], tm, tn] = acc
        for _ in hl.grid(1):
            me = rank[0]
            for j in hl.tile(world - 1, block_size=1):
                copy = hl.make_async_remote_copy(
                    recv,
                    [i % 2, me],
                    peers[0, j.begin],
                    dst=recv,
                    dst_index=[i % 2, me],
                )
                copy.start()
                if j.begin == world - 2:
                    for _w in hl.static_range(world - 1):
                        copy.wait()
        for tm, tn in hl.tile([m, h]):
            hidden[tm, tn] = hidden[tm, tn] + torch.sum(recv[i % 2, :, tm, tn], 0)
    for tm, tn in hl.tile([m, h]):
        out[tm, tn] = hidden[tm, tn].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def tp_gemv_pair_stack(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    peers: torch.Tensor,
    rank: torch.Tensor,
) -> torch.Tensor:
    """``tp_gemv_chain`` with the low-rank activation summed over ranks too:
    two exchanges per layer, each between the gemvs of two weight rings."""
    m, h = x.shape
    layers, _, r = a.shape
    world = peers.size(1) + 1
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    t = torch.empty([m, r], dtype=x.dtype, device=x.device)
    recv_t = torch.empty([2, world, m, r], dtype=torch.float32, device=x.device)
    recv = torch.empty([2, world, m, h], dtype=torch.float32, device=x.device)
    out = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, h]):
        hidden[tm, tn] = x[tm, tn].float()
    for _ in hl.grid(1):
        hl.remote_barrier(peers[0, :])
    for i in range(layers):
        for tm, tr in hl.tile([m, r]):
            acc = hl.zeros([tm, tr], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = torch.addmm(acc, hidden[tm, tk].to(x.dtype), a[i, tk, tr])
            recv_t[i % 2, rank[0], tm, tr] = acc
        for _ in hl.grid(1):
            me = rank[0]
            for j in hl.tile(world - 1, block_size=1):
                copy = hl.make_async_remote_copy(
                    recv_t,
                    [i % 2, me],
                    peers[0, j.begin],
                    dst=recv_t,
                    dst_index=[i % 2, me],
                )
                copy.start()
                if j.begin == world - 2:
                    for _w in hl.static_range(world - 1):
                        copy.wait()
        for tm, tr in hl.tile([m, r]):
            t[tm, tr] = torch.sum(recv_t[i % 2, :, tm, tr], 0).to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(r):
                acc = torch.addmm(acc, t[tm, tk], b[i, tk, tn])
            recv[i % 2, rank[0], tm, tn] = acc
        for _ in hl.grid(1):
            me = rank[0]
            for j in hl.tile(world - 1, block_size=1):
                copy = hl.make_async_remote_copy(
                    recv,
                    [i % 2, me],
                    peers[0, j.begin],
                    dst=recv,
                    dst_index=[i % 2, me],
                )
                copy.start()
                if j.begin == world - 2:
                    for _w in hl.static_range(world - 1):
                        copy.wait()
        for tm, tn in hl.tile([m, h]):
            hidden[tm, tn] = hidden[tm, tn] + torch.sum(recv[i % 2, :, tm, tn], 0)
    for tm, tn in hl.tile([m, h]):
        out[tm, tn] = hidden[tm, tn].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def tp_all_gather_sum(
    x: torch.Tensor, peers: torch.Tensor, rank: torch.Tensor
) -> torch.Tensor:
    """The sum over ranks of ``x``, gathered in ``x``'s dtype."""
    m, h = x.shape
    world = peers.size(1) + 1
    gather = torch.empty([world, m, h], dtype=x.dtype, device=x.device)
    out = torch.empty([m, h], dtype=torch.float32, device=x.device)
    for _ in hl.grid(1):
        hl.remote_barrier(peers[0, :])
    for tm, tn in hl.tile([m, h]):
        gather[rank[0], tm, tn] = x[tm, tn]
    for _ in hl.grid(1):
        me = rank[0]
        for j in hl.tile(world - 1, block_size=1):
            copy = hl.make_async_remote_copy(
                gather, [me], peers[0, j.begin], dst=gather, dst_index=[me]
            )
            copy.start()
            if j.begin == world - 2:
                for _w in hl.static_range(world - 1):
                    copy.wait()
    for tm, tn in hl.tile([m, h]):
        out[tm, tn] = torch.sum(gather[:, tm, tn].float(), 0)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def tp_all_reduce_residual(
    x: torch.Tensor,
    w: torch.Tensor,
    peers: torch.Tensor,
    rank: torch.Tensor,
    algorithm: hl.constexpr = None,
) -> torch.Tensor:
    """A residual stream of row-parallel GEMVs, each all-reduced by
    ``hl.all_reduce``: one exchange per layer."""
    m, h = x.shape
    res = torch.empty_like(x)
    partial = torch.empty([m, h], dtype=torch.float32, device=x.device)
    red = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, h]):
        res[tm, tn] = x[tm, tn]
    for _ in hl.grid(1):
        hl.remote_barrier(peers[0, :])
    for i in range(w.size(0)):
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(res[tm, tk], w[i, tk, tn], acc=acc)
            partial[tm, tn] = acc
        hl.all_reduce(red, partial, peers[0, :], rank[0], algorithm=algorithm)
        for tm, tn in hl.tile([m, h]):
            res[tm, tn] = res[tm, tn] + red[tm, tn]
    return res


@helion.kernel(backend="pallas", static_shapes=True)
def tp_all_reduce_pairs(
    x: torch.Tensor, w: torch.Tensor, peers: torch.Tensor, rank: torch.Tensor
) -> torch.Tensor:
    """``tp_all_reduce_residual`` with two GEMVs and all-reduces per layer."""
    m, h = x.shape
    res = torch.empty_like(x)
    partial = torch.empty([m, h], dtype=torch.float32, device=x.device)
    red = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, h]):
        res[tm, tn] = x[tm, tn]
    for _ in hl.grid(1):
        hl.remote_barrier(peers[0, :])
    for i in range(w.size(0)):
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(res[tm, tk], w[i, 0, tk, tn], acc=acc)
            partial[tm, tn] = acc
        hl.all_reduce(red, partial, peers[0, :], rank[0])
        for tm, tn in hl.tile([m, h]):
            res[tm, tn] = res[tm, tn] + red[tm, tn]
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(res[tm, tk], w[i, 1, tk, tn], acc=acc)
            partial[tm, tn] = acc
        hl.all_reduce(red, partial, peers[0, :], rank[0])
        for tm, tn in hl.tile([m, h]):
            res[tm, tn] = res[tm, tn] + red[tm, tn]
    return res


@helion.kernel(backend="pallas", static_shapes=True)
def shape_chosen_roots(x: torch.Tensor) -> torch.Tensor:
    """Device loops chosen by a host ``if`` on a shape."""
    out = torch.empty_like(x)
    if x.size(0) > 8:
        for tile in hl.tile(x.size()):
            out[tile] = x[tile] * 2
    else:
        for tile in hl.tile(x.size()):
            out[tile] = x[tile] + 1
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def tp_gemv_head(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w_out: torch.Tensor,
    peers: torch.Tensor,
    rank: torch.Tensor,
) -> torch.Tensor:
    """Two gemvs with their outputs summed over ranks, then a gemv of the
    second sum."""
    m, h = x.shape
    r = b.size(1)
    n = w_out.size(1)
    world = peers.size(1) + 1
    recv = torch.empty([world, m, h], dtype=torch.float32, device=x.device)
    recv_t = torch.empty([world, m, r], dtype=torch.float32, device=x.device)
    hidden = torch.empty([m, h], dtype=x.dtype, device=x.device)
    t = torch.empty([m, r], dtype=x.dtype, device=x.device)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for _ in hl.grid(1):
        hl.remote_barrier(peers[0, :])
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = torch.addmm(acc, x[tm, tk], a[tk, tn])
        recv[rank[0], tm, tn] = acc
    for _ in hl.grid(1):
        me = rank[0]
        for j in hl.tile(world - 1, block_size=1):
            copy = hl.make_async_remote_copy(
                recv, [me], peers[0, j.begin], dst=recv, dst_index=[me]
            )
            copy.start()
            if j.begin == world - 2:
                for _w in hl.static_range(world - 1):
                    copy.wait()
    for tm, tn in hl.tile([m, h]):
        hidden[tm, tn] = torch.sum(recv[:, tm, tn], 0).to(x.dtype)
    for tm, tr in hl.tile([m, r]):
        acc = hl.zeros([tm, tr], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = torch.addmm(acc, hidden[tm, tk], b[tk, tr])
        recv_t[rank[0], tm, tr] = acc
    for _ in hl.grid(1):
        me = rank[0]
        for j in hl.tile(world - 1, block_size=1):
            copy = hl.make_async_remote_copy(
                recv_t, [me], peers[0, j.begin], dst=recv_t, dst_index=[me]
            )
            copy.start()
            if j.begin == world - 2:
                for _w in hl.static_range(world - 1):
                    copy.wait()
    for tm, tr in hl.tile([m, r]):
        t[tm, tr] = torch.sum(recv_t[:, tm, tr], 0).to(x.dtype)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(r):
            acc = torch.addmm(acc, t[tm, tk], w_out[tk, tn])
        out[tm, tn] = acc.to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def scale_stack(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, h = x.shape
    layers = w.size(0)
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :]
    for i in range(layers):
        for tm in hl.tile(m):
            hidden[tm, :] = hidden[tm, :] * w[i, :][None, :] + i
        for tm in hl.tile(m):
            hidden[tm, :] = hidden[tm, :] + 1.0
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def periodic_stack(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    norm_w: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    period: hl.constexpr = 3,
) -> torch.Tensor:
    """Periods of ``period - 1`` low-rank layers then one MLP layer."""
    m, h = x.shape
    rank = a.size(2)
    periods, _, inter = w_up.shape
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    t = torch.empty([m, rank], dtype=x.dtype, device=x.device)
    normed = torch.empty([m, h], dtype=x.dtype, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for p in range(periods):
        for r in hl.static_range(period):
            if r == period - 1:
                for tm in hl.tile(m):
                    s = hidden[tm, :]
                    rms = torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + 1e-6)
                    normed[tm, :] = (s * rms * norm_w[p, :][None, :]).to(x.dtype)
                for tm, tn in hl.tile([m, inter]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(h):
                        acc = torch.addmm(acc, normed[tm, tk], w_up[p, tk, tn])
                    act[tm, tn] = torch.nn.functional.silu(acc).to(x.dtype)
                for tm, tn in hl.tile([m, h]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(inter):
                        acc = torch.addmm(acc, act[tm, tk], w_down[p, tk, tn])
                    hidden[tm, tn] = hidden[tm, tn] + acc
            else:
                for tm, tr in hl.tile([m, rank]):
                    acc = hl.zeros([tm, tr], dtype=torch.float32)
                    for tk in hl.tile(h):
                        a_tile = a[(period - 1) * p + r, tk, tr]
                        acc = torch.addmm(acc, hidden[tm, tk].to(x.dtype), a_tile)
                    t[tm, tr] = acc.to(x.dtype)
                for tm, tn in hl.tile([m, h]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(rank):
                        b_tile = b[(period - 1) * p + r, tk, tn]
                        acc = torch.addmm(acc, t[tm, tk], b_tile)
                    hidden[tm, tn] = hidden[tm, tn] + acc
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def write_and_attend(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    """Decode attention that writes row ``pos`` of the caches, then scans
    every cache row and masks those past ``pos``."""
    nkv, group, d = q.shape
    ctx = k_cache.size(1)
    qs = torch.empty_like(q)
    out = torch.empty_like(q)
    for _ in hl.grid(1):
        qs[:, :, :] = (q[:, :, :].float() * d**-0.5).to(q.dtype)
    for _ in hl.grid(1):
        p = pos[0]
        k_cache[:, p, :] = k_new[:, :]
        v_cache[:, p, :] = v_new[:, :]
        qv = qs[:, :, :]
        m_i = hl.full([nkv, group], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([nkv, group], dtype=torch.float32)
        acc = hl.zeros([nkv, group, d], dtype=torch.float32)
        for tt in hl.tile(ctx):
            s = hl.dot(qv, k_cache[:, tt, :].transpose(1, 2), out_dtype=torch.float32)
            s = torch.where(tt.index[None, None, :] <= p, s, -1e30)
            m_new = torch.maximum(m_i, torch.amax(s, -1))
            alpha = torch.exp(m_i - m_new)
            probs = torch.exp(s - m_new[:, :, None])
            l_i = l_i * alpha + torch.sum(probs, -1)
            pv = hl.dot(probs.to(q.dtype), v_cache[:, tt, :], out_dtype=torch.float32)
            acc = acc * alpha[:, :, None] + pv
            m_i = m_new
        out[:, :, :] = (acc / l_i[:, :, None]).to(q.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def write_and_attend_bounded(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    """``write_and_attend`` with a scan that stops at row ``pos``."""
    nkv, group, d = q.shape
    qs = torch.empty_like(q)
    out = torch.empty_like(q)
    for _ in hl.grid(1):
        qs[:, :, :] = (q[:, :, :].float() * d**-0.5).to(q.dtype)
    for _ in hl.grid(1):
        p = pos[0]
        k_cache[:, p, :] = k_new[:, :]
        v_cache[:, p, :] = v_new[:, :]
        qv = qs[:, :, :]
        m_i = hl.full([nkv, group], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([nkv, group], dtype=torch.float32)
        acc = hl.zeros([nkv, group, d], dtype=torch.float32)
        for tt in hl.tile(0, p + 1):
            s = hl.dot(qv, k_cache[:, tt, :].transpose(1, 2), out_dtype=torch.float32)
            s = torch.where(tt.index[None, None, :] <= p, s, -1e30)
            m_new = torch.maximum(m_i, torch.amax(s, -1))
            alpha = torch.exp(m_i - m_new)
            probs = torch.exp(s - m_new[:, :, None])
            l_i = l_i * alpha + torch.sum(probs, -1)
            pv = hl.dot(probs.to(q.dtype), v_cache[:, tt, :], out_dtype=torch.float32)
            acc = acc * alpha[:, :, None] + pv
            m_i = m_new
        out[:, :, :] = (acc / l_i[:, :, None]).to(q.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def write_then_attend(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    """``write_and_attend_bounded`` with the cache write in a root of its own."""
    nkv, group, d = q.shape
    out = torch.empty_like(q)
    for _ in hl.grid(1):
        p = pos[0]
        k_cache[:, p, :] = k_new[:, :]
        v_cache[:, p, :] = v_new[:, :]
    for _ in hl.grid(1):
        p = pos[0]
        qv = (q[:, :, :].float() * d**-0.5).to(q.dtype)
        m_i = hl.full([nkv, group], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([nkv, group], dtype=torch.float32)
        acc = hl.zeros([nkv, group, d], dtype=torch.float32)
        for tt in hl.tile(0, p + 1):
            s = hl.dot(qv, k_cache[:, tt, :].transpose(1, 2), out_dtype=torch.float32)
            s = torch.where(tt.index[None, None, :] <= p, s, -1e30)
            m_new = torch.maximum(m_i, torch.amax(s, -1))
            alpha = torch.exp(m_i - m_new)
            probs = torch.exp(s - m_new[:, :, None])
            l_i = l_i * alpha + torch.sum(probs, -1)
            pv = hl.dot(probs.to(q.dtype), v_cache[:, tt, :], out_dtype=torch.float32)
            acc = acc * alpha[:, :, None] + pv
            m_i = m_new
        out[:, :, :] = (acc / l_i[:, :, None]).to(q.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def gated_stack(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    """Per layer: a narrow gate projection whose root reads its [H, G] weight
    whole, then a low-rank update scaled by the gates."""
    m, h = x.shape
    layers, _, g = w_gate.shape
    rank = w_up.size(2)
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    gates = torch.empty([m, g], dtype=torch.float32, device=x.device)
    t = torch.empty([m, rank], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for tm in hl.tile(m):
            gate = hl.dot(
                hidden[tm, :].to(x.dtype), w_gate[i, :, :], out_dtype=torch.float32
            )
            gates[tm, :] = torch.sigmoid(gate)
        for tm, tr in hl.tile([m, rank]):
            acc = hl.zeros([tm, tr], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(hidden[tm, tk].to(x.dtype), w_up[i, tk, tr], acc=acc)
            t[tm, tr] = (acc * gates[tm, :].mean(-1, keepdim=True)).to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(rank):
                acc = hl.dot(t[tm, tk], w_down[i, tk, tn], acc=acc)
            hidden[tm, tn] = hidden[tm, tn] + acc
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def write_and_read_row(
    k_new: torch.Tensor, k_cache: torch.Tensor, pos: torch.Tensor
) -> torch.Tensor:
    """Writes row ``pos`` of a cache, then reads it in a loop and at root scope."""
    nkv, ctx, d = k_cache.shape
    out = torch.empty_like(k_new)
    for _ in hl.grid(1):
        p = pos[0]
        k_cache[:, p, :] = k_new[:, :]
    for _ in hl.grid(1):
        p = pos[0]
        acc = hl.zeros([nkv, d], dtype=torch.float32)
        for tt in hl.tile(ctx):
            acc = acc + torch.sum(k_cache[:, tt, :].float(), 1)
        out[:, :] = (acc + k_cache[:, p, :].float()).to(out.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def periodic_tap_stack(
    x: torch.Tensor,
    taps: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    period: hl.constexpr = 3,
) -> torch.Tensor:
    """Periods of ``period - 1`` tap layers, whose ``hl.grid`` root reads a
    per-layer [H, K] tap matrix whole to rescale the rows, then one MLP
    layer."""
    m, h = x.shape
    periods, _, inter = w_up.shape
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for p in range(periods):
        for r in hl.static_range(period):
            if r == period - 1:
                for tm, tn in hl.tile([m, inter]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(h):
                        a = hidden[tm, tk].to(x.dtype)
                        acc = hl.dot(a, w_up[p, tk, tn], acc=acc)
                    act[tm, tn] = torch.nn.functional.silu(acc).to(x.dtype)
                for tm, tn in hl.tile([m, h]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(inter):
                        acc = hl.dot(act[tm, tk], w_down[p, tk, tn], acc=acc)
                    hidden[tm, tn] = hidden[tm, tn] + acc
            else:
                for _ in hl.grid(1):
                    w = taps[(period - 1) * p + r, :, :]
                    g = hl.dot(hidden[:, :].to(x.dtype), w, out_dtype=torch.float32)
                    scale = torch.sigmoid(g).mean(-1, keepdim=True)
                    hidden[:, :] = hidden[:, :] * scale
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def write_and_sum_before(
    k_new: torch.Tensor, k_cache: torch.Tensor, pos: torch.Tensor
) -> torch.Tensor:
    """Writes row ``pos`` of a cache, then sums the rows before it."""
    nkv, d = k_new.shape
    k_twice = torch.empty_like(k_new)
    out = torch.empty_like(k_new, dtype=torch.float32)
    for _ in hl.grid(1):
        k_twice[:, :] = k_new[:, :] * 2
    for _ in hl.grid(1):
        p = pos[0]
        k_cache[:, p, :] = k_twice[:, :]
        acc = hl.zeros([nkv, d], dtype=torch.float32)
        for tt in hl.tile(0, p):
            acc = acc + torch.sum(k_cache[:, tt, :].float(), 1)
        out[:, :] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def narrow_rank_stack(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Per layer: a rank-``R`` update whose [H, R] down-projection streams in
    (tk, R) tiles of its inner loop."""
    m, h = x.shape
    layers, _, rank = a.shape
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    t = torch.empty([m, rank], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for tm in hl.tile(m):
            acc = hl.zeros([tm, rank], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(hidden[tm, tk].to(x.dtype), a[i, tk, :], acc=acc)
            t[tm, :] = acc.to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            update = hl.dot(t[tm, :], b[i, :, tn], out_dtype=torch.float32)
            hidden[tm, tn] = hidden[tm, tn] + update
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def gqa_decode_layer(
    hidden: torch.Tensor,
    norm_w: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
    eps: hl.constexpr = 1e-6,
) -> torch.Tensor:
    """One decode token of a GQA layer: RMSNorm, q/k/v projections,
    attention over in-place KV caches, output projection plus residual."""
    m, h = hidden.shape
    nkv, _, d = k_cache.shape
    heads = wq.size(1) // d
    dt = hidden.dtype
    x = torch.empty_like(hidden)
    q = torch.empty([m, heads * d], dtype=dt, device=hidden.device)
    k = torch.empty([m, nkv * d], dtype=dt, device=hidden.device)
    v = torch.empty([m, nkv * d], dtype=dt, device=hidden.device)
    attn = torch.empty([m, heads * d], dtype=dt, device=hidden.device)
    out = torch.empty_like(hidden)
    for tm in hl.tile(m):
        s = hidden[tm, :].float()
        rms = torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps)
        x[tm, :] = (s * rms * norm_w[None, :].float()).to(dt)
    for tm, tn in hl.tile([m, heads * d]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = torch.addmm(acc, x[tm, tk], wq[tk, tn])
        q[tm, tn] = acc.to(dt)
    for tm, tn in hl.tile([m, nkv * d]):
        acc_k = hl.zeros([tm, tn], dtype=torch.float32)
        acc_v = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            acc_k = torch.addmm(acc_k, x[tm, tk], wk[tk, tn])
            acc_v = torch.addmm(acc_v, x[tm, tk], wv[tk, tn])
        k[tm, tn] = acc_k.to(dt)
        v[tm, tn] = acc_v.to(dt)
    for _ in hl.grid(1):
        p = pos[0]
        k_cache[:, p, :] = k[0, :].reshape(nkv, d)
        v_cache[:, p, :] = v[0, :].reshape(nkv, d)
        qv = (q[0, :].reshape(nkv, heads // nkv, d).float() * d**-0.5).to(dt)
        m_i = hl.full([nkv, heads // nkv], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([nkv, heads // nkv], dtype=torch.float32)
        acc_o = hl.zeros([nkv, heads // nkv, d], dtype=torch.float32)
        for tt in hl.tile(0, p + 1):
            s = hl.dot(qv, k_cache[:, tt, :].transpose(1, 2), out_dtype=torch.float32)
            s = torch.where(tt.index[None, None, :] <= p, s, -1e30)
            m_new = torch.maximum(m_i, torch.amax(s, -1))
            alpha = torch.exp(m_i - m_new)
            probs = torch.exp(s - m_new[:, :, None])
            l_i = l_i * alpha + torch.sum(probs, -1)
            pv = hl.dot(probs.to(dt), v_cache[:, tt, :], out_dtype=torch.float32)
            acc_o = acc_o * alpha[:, :, None] + pv
            m_i = m_new
        attn[0, :] = (acc_o / l_i[:, :, None]).to(dt).reshape(heads * d)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(heads * d):
            acc = torch.addmm(acc, attn[tm, tk], wo[tk, tn])
        out[tm, tn] = hidden[tm, tn] + acc.to(dt)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def scale_then_narrow(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, h = x.shape
    g = w.size(1)
    y = torch.empty([m, h], dtype=x.dtype, device=x.device)
    out = torch.empty([m, g], dtype=torch.float32, device=x.device)
    for tm, th in hl.tile([m, h]):
        y[tm, th] = x[tm, th] * 2.0
    for tm in hl.tile(m):
        acc = hl.zeros([tm, g], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = hl.dot(y[tm, tk], w[tk, :], acc=acc)
        out[tm, :] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def embed_head_step(
    token: torch.Tensor,
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    vocab: hl.constexpr,
    eps: hl.constexpr = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The non-layer parts of a decode step: the token's embedding row, and
    the top-1 of the LM head over the final-normed hidden state, a running
    max over the vocab tiles (the padded columns from ``vocab`` on masked)."""
    m, h = hidden.shape
    vp = lm_head.size(1)
    next_input = torch.empty([m, h], dtype=embedding.dtype, device=hidden.device)
    normed = torch.empty_like(hidden)
    top_idx = torch.empty([m], dtype=torch.int32, device=hidden.device)
    top_val = torch.empty([m], dtype=torch.float32, device=hidden.device)
    for _ in hl.grid(1):
        t = token[0]
        next_input[0, :] = embedding[t, :]
    for tm in hl.tile(m):
        x = hidden[tm, :].float()
        rms = torch.rsqrt(torch.mean(x * x, -1, keepdim=True) + eps)
        normed[tm, :] = (x * rms * final_norm[None, :].float()).to(hidden.dtype)
    for tm in hl.tile(m):
        best = hl.full([tm], float("-inf"), dtype=torch.float32)
        best_i = hl.zeros([tm], dtype=torch.int32)
        for tn in hl.tile(vp):
            logits = hl.dot(normed[tm, :], lm_head[:, tn], out_dtype=torch.float32)
            logits = torch.where(tn.index[None, :] < vocab, logits, float("-inf"))
            tile_max = torch.amax(logits, -1)
            tile_arg = torch.argmax(logits, -1).to(torch.int32) + tn.begin
            # Strictly greater: a tie keeps the earlier tile's index.
            better = tile_max > best
            best_i = torch.where(better, tile_arg, best_i)
            best = torch.where(better, tile_max, best)
        top_idx[tm] = best_i
        top_val[tm] = best
    return next_input, top_idx, top_val


@helion.kernel(backend="pallas", static_shapes=True)
def logits_then_top1(
    hidden: torch.Tensor, lm_head: torch.Tensor, vocab: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    """The LM head's logits of every vocab tile into scratch, then the top-1
    of the whole row in a root of its own."""
    m = hidden.size(0)
    vp = lm_head.size(1)
    logits = torch.empty([m, vp], dtype=torch.float32, device=hidden.device)
    top_idx = torch.empty([m], dtype=torch.int32, device=hidden.device)
    top_val = torch.empty([m], dtype=torch.float32, device=hidden.device)
    for tm, tn in hl.tile([m, vp]):
        acc = hl.dot(hidden[tm, :], lm_head[:, tn], out_dtype=torch.float32)
        logits[tm, tn] = torch.where(tn.index[None, :] < vocab, acc, float("-inf"))
    for tm in hl.tile(m):
        row = logits[tm, :]
        top_val[tm] = torch.amax(row, -1)
        top_idx[tm] = torch.argmax(row, -1).to(torch.int32)
    return top_idx, top_val


@helion.kernel(backend="pallas", static_shapes=True)
def gathered_top1(
    vals: torch.Tensor, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The top-1 of per-shard ``(value, index)`` maxima, the lowest index on
    ties, one column at a time: a tensor-parallel decode step's final
    reduce, after a root that scales the values."""
    shards, m = vals.shape
    scaled = torch.empty_like(vals)
    top_idx = torch.empty([m], dtype=torch.int32, device=vals.device)
    top_val = torch.empty([m], dtype=torch.float32, device=vals.device)
    for ts, tm in hl.tile([shards, m]):
        scaled[ts, tm] = vals[ts, tm] * 2
    for tm in hl.tile(m, block_size=1):
        v = scaled[:, tm]
        best = torch.amax(v, 0)
        cand = torch.where(v == best[None, :], idx[:, tm], 1 << 30)
        top_idx[tm] = torch.amin(cand, 0)
        top_val[tm] = best
    return top_idx, top_val


@helion.kernel(backend="pallas", static_shapes=True)
def scaled_token_pair(
    x: torch.Tensor, tokens: torch.Tensor, table: torch.Tensor
) -> torch.Tensor:
    """Rows of a table at two runtime indices, the second wrapped into
    range, times ``x`` normalized in an earlier root."""
    m, h = x.shape
    v = table.size(0)
    normed = torch.empty([m, h], dtype=torch.float32, device=x.device)
    out = torch.empty([2, h], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        s = x[tm, :].float()
        normed[tm, :] = s * torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + 1e-6)
    for _ in hl.grid(1):
        first = tokens[0]
        second = (tokens[1] + 1) % v
        out[0, :] = table[first, :].float() * normed[0, :]
        out[1, :] = table[second, :].float() * normed[0, :]
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def dense_step(
    token: torch.Tensor,
    embedding: torch.Tensor,
    norm_w: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    vocab: hl.constexpr,
    eps: hl.constexpr = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A dense decode step: embedding row, L pre-norm MLP blocks, final
    norm, then the top-1 of the LM head."""
    m = token.size(0)
    h = embedding.size(1)
    layers, _, inter = w_gate.shape
    vp = lm_head.size(1)
    dt = embedding.dtype
    hidden = torch.empty([m, h], dtype=torch.float32, device=embedding.device)
    normed = torch.empty([m, h], dtype=dt, device=embedding.device)
    act = torch.empty([m, inter], dtype=dt, device=embedding.device)
    top_idx = torch.empty([m], dtype=torch.int32, device=embedding.device)
    top_val = torch.empty([m], dtype=torch.float32, device=embedding.device)
    for _ in hl.grid(1):
        t = token[0]
        hidden[0, :] = embedding[t, :].float()
    for i in range(layers):
        for tm in hl.tile(m):
            s = hidden[tm, :]
            rms = torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps)
            normed[tm, :] = (s * rms * norm_w[i, :].float()[None, :]).to(dt)
        for tm, tn in hl.tile([m, inter]):
            g = hl.zeros([tm, tn], dtype=torch.float32)
            u = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = normed[tm, tk]
                g = torch.addmm(g, a, w_gate[i, tk, tn])
                u = torch.addmm(u, a, w_up[i, tk, tn])
            gate = torch.nn.functional.silu(g.to(dt).float()).to(dt)
            act[tm, tn] = gate * u.to(dt)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                acc = torch.addmm(acc, act[tm, tk], w_down[i, tk, tn])
            hidden[tm, tn] = acc + hidden[tm, tn]
    for tm in hl.tile(m):
        s = hidden[tm, :]
        rms = torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps)
        normed[tm, :] = (s * rms * final_norm[None, :].float()).to(dt)
    for tm in hl.tile(m):
        best = hl.full([tm], float("-inf"), dtype=torch.float32)
        best_i = hl.zeros([tm], dtype=torch.int32)
        for tv in hl.tile(vp):
            logits = hl.dot(normed[tm, :], lm_head[:, tv], out_dtype=torch.float32)
            logits = torch.where(tv.index[None, :] < vocab, logits, float("-inf"))
            tile_max = torch.amax(logits, -1)
            tile_arg = torch.argmax(logits, -1).to(torch.int32) + tv.begin
            better = tile_max > best
            best_i = torch.where(better, tile_arg, best_i)
            best = torch.where(better, tile_max, best)
        top_idx[tm] = best_i
        top_val[tm] = best
    return top_idx, top_val


@helion.kernel(backend="pallas", static_shapes=True)
def recurrent_state_stack(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    rec_state: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Per layer: an ``hl.grid`` root that shifts the first row into the
    layer's [H, K] conv window and decays the layer's [R, H] recurrent state
    by the window's sums, both read and rewritten whole, then an MLP whose
    weights the ring streams."""
    m, h = x.shape
    layers, _, inter = w_up.shape
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for _ in hl.grid(1):
            window = conv_state[i, :, :]
            row = hidden[0, :].to(x.dtype)
            window = torch.cat([window[:, 1:], row[:, None]], dim=-1)
            conv_state[i, :, :] = window
            mixed = torch.sum(window.float(), -1)
            state = rec_state[i, :, :] * 0.5 + mixed[None, :]
            rec_state[i, :, :] = state
            hidden[:, :] = hidden[:, :] + torch.tanh(state).mean(0)[None, :]
        for tm, tn in hl.tile([m, inter]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = hidden[tm, tk].to(x.dtype)
                acc = hl.dot(a, w_up[i, tk, tn], acc=acc)
            act[tm, tn] = torch.nn.functional.silu(acc).to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                acc = hl.dot(act[tm, tk], w_down[i, tk, tn], acc=acc)
            hidden[tm, tn] = hidden[tm, tn] + acc
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def batched_state_stack(
    x: torch.Tensor,
    rec_state: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """``recurrent_state_stack``'s recurrence for a batch of sequences, each
    with a [R, H] state of its own per layer: the root reads and rewrites
    sequence ``b``'s state at ``rec_state[i, b]``, one slice per sequence."""
    m, h = x.shape
    layers, _, inter = w_up.shape
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for _ in hl.grid(1):
            for b in hl.static_range(m):
                state = rec_state[i, b, :, :] * 0.5 + hidden[b, :][None, :]
                rec_state[i, b, :, :] = state
                hidden[b, :] = hidden[b, :] + torch.tanh(state).mean(0)
        for tm, tn in hl.tile([m, inter]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = hidden[tm, tk].to(x.dtype)
                acc = hl.dot(a, w_up[i, tk, tn], acc=acc)
            act[tm, tn] = torch.nn.functional.silu(acc).to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                acc = hl.dot(act[tm, tk], w_down[i, tk, tn], acc=acc)
            hidden[tm, tn] = hidden[tm, tn] + acc
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def paired_state_stack(
    x: torch.Tensor,
    rec_state: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """``batched_state_stack`` with two layers per loop iteration, so the
    loop body holds two roots that read state slices of one shape."""
    m, h = x.shape
    layers, _, inter = w_up.shape
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for p in range(layers // 2):
        for r in hl.static_range(2):
            for _ in hl.grid(1):
                for b in hl.static_range(m):
                    state = rec_state[2 * p + r, b, :, :] * 0.5 + hidden[b, :][None, :]
                    rec_state[2 * p + r, b, :, :] = state
                    hidden[b, :] = hidden[b, :] + torch.tanh(state).mean(0)
            for tm, tn in hl.tile([m, inter]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(h):
                    a = hidden[tm, tk].to(x.dtype)
                    acc = hl.dot(a, w_up[2 * p + r, tk, tn], acc=acc)
                act[tm, tn] = torch.nn.functional.silu(acc).to(x.dtype)
            for tm, tn in hl.tile([m, h]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(inter):
                    acc = hl.dot(act[tm, tk], w_down[2 * p + r, tk, tn], acc=acc)
                hidden[tm, tn] = hidden[tm, tn] + acc
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def state_snapshot_stack(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    rec_state: torch.Tensor,
    conv_snap: torch.Tensor,
    rec_snap: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Per layer: an ``hl.grid`` root that advances the layer's [H, K] conv
    window and [R, H] recurrent state by each row in turn, writing a
    snapshot of both after every row, then an MLP whose weights the ring
    streams."""
    m = hl.specialize(x.size(0))
    h = x.size(1)
    layers, _, inter = w_up.shape
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for _ in hl.grid(1):
            window = conv_state[i, :, :]
            state = rec_state[i, :, :]
            for b in hl.static_range(m):
                row = hidden[b, :].to(x.dtype)
                window = torch.cat([window[:, 1:], row[:, None]], dim=-1)
                state = state * 0.5 + torch.sum(window.float(), -1)[None, :]
                conv_snap[i, b, :, :] = window
                rec_snap[i, b, :, :] = state
            conv_state[i, :, :] = window
            rec_state[i, :, :] = state
            hidden[:, :] = hidden[:, :] + torch.tanh(state).mean(0)[None, :]
        for tm, tn in hl.tile([m, inter]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = hidden[tm, tk].to(x.dtype)
                acc = hl.dot(a, w_up[i, tk, tn], acc=acc)
            act[tm, tn] = torch.nn.functional.silu(acc).to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                acc = hl.dot(act[tm, tk], w_down[i, tk, tn], acc=acc)
            hidden[tm, tn] = hidden[tm, tn] + acc
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def transposed_snapshot_stack(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_w: torch.Tensor,
    conv_snap: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Per layer: an ``hl.grid`` root that lays the layer's [H, K] conv
    window and weights out time major, the window ahead of the block's
    rows, and writes the window after each row back transposed, then an MLP
    whose weights the ring streams."""
    m = hl.specialize(x.size(0))
    h = x.size(1)
    taps = hl.specialize(conv_state.size(2))
    layers, _, inter = w_up.shape
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for _ in hl.grid(1):
            state_in = conv_state[i, :, :].transpose(0, 1)
            inputs = torch.cat([state_in[1:, :], hidden[:, :].to(x.dtype)], dim=0)
            for b in hl.static_range(m):
                conv_snap[i, b, :, :] = inputs[b : b + taps, :].transpose(0, 1)
            conv_state[i, :, :] = inputs[m - 1 : m - 1 + taps, :].transpose(0, 1)
            w_conv = conv_w[i, :, :].float().transpose(0, 1)
            mixed = inputs[:m, :].float() * w_conv[taps - 1 : taps, :]
            hidden[:, :] = hidden[:, :] + mixed
        for tm, tn in hl.tile([m, inter]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = hidden[tm, tk].to(x.dtype)
                acc = hl.dot(a, w_up[i, tk, tn], acc=acc)
            act[tm, tn] = torch.nn.functional.silu(acc).to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                acc = hl.dot(act[tm, tk], w_down[i, tk, tn], acc=acc)
            hidden[tm, tn] = hidden[tm, tn] + acc
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def block_kv_rows(
    x: torch.Tensor, w_k: torch.Tensor, k_cache: torch.Tensor, pos: torch.Tensor
) -> None:
    """Per layer: a key projection of a block of rows, then each row in
    turn at row ``pos`` plus its index of the layer's [KV, T, D] cache."""
    m = hl.specialize(x.size(0))
    h = x.size(1)
    layers, nkv, _, d = k_cache.shape
    k = torch.empty([m, nkv * d], dtype=x.dtype, device=x.device)
    for i in range(layers):
        for tm, tn in hl.tile([m, nkv * d]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(x[tm, tk], w_k[i, tk, tn], acc=acc)
            k[tm, tn] = acc.to(x.dtype)
        for _ in hl.grid(1):
            p = pos[0]
            for b in hl.static_range(m):
                k_cache[i, :, p + b, :] = k[b, :].reshape(nkv, d)


@helion.kernel(backend="pallas", static_shapes=True)
def fused_kv_rows(
    x: torch.Tensor,
    w_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
) -> None:
    """Per layer: one fused key-value projection, then row ``pos`` of each
    cache from slices of the projection's first row; the key's halves are
    swapped by slicing the key slice again."""
    m, h = x.shape
    layers, nkv, ctx, d = k_cache.shape
    half = d // 2
    kv = torch.empty([m, 2 * nkv * d], dtype=x.dtype, device=x.device)
    for i in range(layers):
        for tm, tn in hl.tile([m, 2 * nkv * d]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(x[tm, tk], w_kv[i, tk, tn], acc=acc)
            kv[tm, tn] = acc.to(x.dtype)
        for _ in hl.grid(1):
            p = pos[0]
            row = kv[0, :]
            k = row[: nkv * d].reshape(nkv, d)
            k_cache[i, :, p, :] = torch.cat([-k[:, half:], k[:, :half]], dim=-1)
            v_cache[i, :, p, :] = row[nkv * d :].reshape(nkv, d)


@helion.kernel(backend="pallas", static_shapes=True)
def cache_stack(
    x: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    """Per layer: key and value projections of the first row, decode
    attention that writes them at row ``pos`` of the layer's [KV, T, D]
    caches and scans every cache row in an inner loop, then an output
    projection; the ring streams the weights."""
    m, h = x.shape
    layers, nkv, ctx, d = k_cache.shape
    group = h // (nkv * d)
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    k = torch.empty([m, nkv * d], dtype=x.dtype, device=x.device)
    v = torch.empty([m, nkv * d], dtype=x.dtype, device=x.device)
    attn = torch.empty([m, h], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for tm, tn in hl.tile([m, nkv * d]):
            acc_k = hl.zeros([tm, tn], dtype=torch.float32)
            acc_v = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = hidden[tm, tk].to(x.dtype)
                acc_k = hl.dot(a, w_k[i, tk, tn], acc=acc_k)
                acc_v = hl.dot(a, w_v[i, tk, tn], acc=acc_v)
            k[tm, tn] = acc_k.to(x.dtype)
            v[tm, tn] = acc_v.to(x.dtype)
        for _ in hl.grid(1):
            p = pos[0]
            k_cache[i, :, p, :] = k[0, :].reshape(nkv, d)
            v_cache[i, :, p, :] = v[0, :].reshape(nkv, d)
            q = (hidden[0, :] * d**-0.5).to(x.dtype).reshape(nkv, group, d)
            m_i = hl.full([nkv, group], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([nkv, group], dtype=torch.float32)
            acc_o = hl.zeros([nkv, group, d], dtype=torch.float32)
            for tt in hl.tile(ctx):
                s = hl.dot(
                    q, k_cache[i, :, tt, :].transpose(1, 2), out_dtype=torch.float32
                )
                s = torch.where(tt.index[None, None, :] <= p, s, -1e30)
                m_new = torch.maximum(m_i, torch.amax(s, -1))
                alpha = torch.exp(m_i - m_new)
                probs = torch.exp(s - m_new[:, :, None])
                l_i = l_i * alpha + torch.sum(probs, -1)
                pv = hl.dot(
                    probs.to(x.dtype), v_cache[i, :, tt, :], out_dtype=torch.float32
                )
                acc_o = acc_o * alpha[:, :, None] + pv
                m_i = m_new
            attn[0, :] = (acc_o / l_i[:, :, None]).to(x.dtype).reshape(h)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(attn[tm, tk], w_o[i, tk, tn], acc=acc)
            hidden[tm, tn] = hidden[tm, tn] + acc
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def cache_stack_bounded(
    x: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    """``cache_stack`` with a scan that stops at row ``pos``."""
    m, h = x.shape
    layers, nkv, ctx, d = k_cache.shape
    group = h // (nkv * d)
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    k = torch.empty([m, nkv * d], dtype=x.dtype, device=x.device)
    v = torch.empty([m, nkv * d], dtype=x.dtype, device=x.device)
    attn = torch.empty([m, h], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for tm, tn in hl.tile([m, nkv * d]):
            acc_k = hl.zeros([tm, tn], dtype=torch.float32)
            acc_v = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = hidden[tm, tk].to(x.dtype)
                acc_k = hl.dot(a, w_k[i, tk, tn], acc=acc_k)
                acc_v = hl.dot(a, w_v[i, tk, tn], acc=acc_v)
            k[tm, tn] = acc_k.to(x.dtype)
            v[tm, tn] = acc_v.to(x.dtype)
        for _ in hl.grid(1):
            p = pos[0]
            k_cache[i, :, p, :] = k[0, :].reshape(nkv, d)
            v_cache[i, :, p, :] = v[0, :].reshape(nkv, d)
            q = (hidden[0, :] * d**-0.5).to(x.dtype).reshape(nkv, group, d)
            m_i = hl.full([nkv, group], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([nkv, group], dtype=torch.float32)
            acc_o = hl.zeros([nkv, group, d], dtype=torch.float32)
            for tt in hl.tile(p + 1):
                s = hl.dot(
                    q, k_cache[i, :, tt, :].transpose(1, 2), out_dtype=torch.float32
                )
                s = torch.where(tt.index[None, None, :] <= p, s, -1e30)
                m_new = torch.maximum(m_i, torch.amax(s, -1))
                alpha = torch.exp(m_i - m_new)
                probs = torch.exp(s - m_new[:, :, None])
                l_i = l_i * alpha + torch.sum(probs, -1)
                pv = hl.dot(
                    probs.to(x.dtype), v_cache[i, :, tt, :], out_dtype=torch.float32
                )
                acc_o = acc_o * alpha[:, :, None] + pv
                m_i = m_new
            attn[0, :] = (acc_o / l_i[:, :, None]).to(x.dtype).reshape(h)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(attn[tm, tk], w_o[i, tk, tn], acc=acc)
            hidden[tm, tn] = hidden[tm, tn] + acc
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def narrow_conv_stack(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    conv_state: torch.Tensor,
    conv_w: torch.Tensor,
) -> torch.Tensor:
    """Per layer, an ``hl.grid`` root: a gate projection of the first row by
    the layer's narrow [H, G] weight, and a causal conv step that shifts the
    row into the layer's narrow [H, K] window state in place and applies the
    [H, K] taps to it."""
    m, h = x.shape
    layers = w_gate.size(0)
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for _ in hl.grid(1):
            row = hidden[:, :]
            gate = hl.dot(row.to(x.dtype), w_gate[i, :, :], out_dtype=torch.float32)
            window = conv_state[i, :, :]
            window = torch.cat([window[:, 1:], row[0, :].to(x.dtype)[:, None]], dim=-1)
            conv_state[i, :, :] = window
            mixed = torch.sum(window.float() * conv_w[i, :, :].float(), -1)
            scale = torch.sigmoid(gate).mean(-1, keepdim=True)
            hidden[:, :] = row * scale + mixed[None, :]
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def narrow_column_scale(
    x: torch.Tensor, w: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    """Per layer, scales the first row by column ``cols[i]`` of the layer's
    narrow [H, G] weight: a gather over its narrow dim."""
    m, h = x.shape
    layers = w.size(0)
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for _ in hl.grid(1):
            c = cols[i]
            hidden[:, :] = hidden[:, :] * w[i, :, c].float()[None, :]
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def narrow_gate_stack(
    x: torch.Tensor,
    w_a: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Per layer, an ``hl.grid`` root: a gate projection of the first row by
    the layer's narrow [H, G] weight of an [L, H, G] stack, biased and decayed
    by the layer's rows of the [L, G] vectors ``a_log`` and ``dt_bias``, then
    a scale of the row by the mean decay."""
    m, h = x.shape
    layers = w_a.size(0)
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for _ in hl.grid(1):
            row = hidden[:, :]
            gate = hl.dot(row.to(w_a.dtype), w_a[i, :, :], out_dtype=torch.float32)
            decay = torch.exp(a_log[i, :].float()) * torch.sigmoid(
                gate + dt_bias[i, :].float()[None, :]
            )
            hidden[:, :] = row * (0.5 + decay.mean(-1, keepdim=True))
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


def _mlp_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_w: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference with the kernel's bf16 rounding points."""
    s = x.float() + residual.float()
    rms = torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps)
    normed = (s * rms * norm_w.float()).to(x.dtype)
    g = (normed.float() @ w_gate.float()).to(x.dtype)
    u = (normed.float() @ w_up.float()).to(x.dtype)
    act = torch.nn.functional.silu(g.float()).to(x.dtype) * u
    out = ((act.float() @ w_down.float()) + s).to(x.dtype)
    return out, s


def _mlp_args(m: int, h: int = 256, inter: int = 512) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return (
        torch.randn(m, h, device=DEVICE, dtype=dtype),
        torch.randn(m, h, device=DEVICE, dtype=dtype),
        torch.randn(h, device=DEVICE, dtype=dtype),
        torch.randn(h, inter, device=DEVICE, dtype=dtype) * 0.05,
        torch.randn(h, inter, device=DEVICE, dtype=dtype) * 0.05,
        torch.randn(inter, h, device=DEVICE, dtype=dtype) * 0.05,
    )


def _vmem_refs(
    kernel: helion.Kernel, args: tuple[object, ...], **config: object
) -> list[tuple[tuple[int, ...], str]]:
    """``(shape, dtype)`` of every VMEM ref the kernel body sees under
    ``config`` over the default config.

    Runs the host wrapper with a launcher that records the launch instead of
    executing it, then applies the launcher's own padding to the VMEM arguments.
    """
    bound = kernel.bind(args)
    captured: dict[str, Any] = {}

    def capture(
        pallas_kernel: object, grid: object, *launch_args: object, **kw: Any
    ) -> object:
        captured["args"] = launch_args
        captured["kw"] = kw
        output_only = [
            launch_args[i]
            for i in kw["_output_indices"]
            if i not in kw["_inplace_indices"]
        ]
        return output_only[0] if len(output_only) == 1 else tuple(output_only)

    default = bound.config_spec.default_config()
    bound.compile_config(helion.Config(**{**default.config, **config}))(
        *args, _launcher=capture
    )
    launch_args, kw = captured["args"], captured["kw"]
    shapes = {
        i: list(arg.shape)
        for i, arg in enumerate(launch_args)
        if isinstance(arg, torch.Tensor) and i not in kw.get("_hbm_arg_indices", [])
    }
    for arg_idx, dim, block_size, extra in kw.get("_ds_pad_dims", []):
        if arg_idx in shapes:
            shapes[arg_idx][dim] += (-shapes[arg_idx][dim]) % block_size + extra
    refs = [
        (tuple(shape), str(launch_args[i].dtype).removeprefix("torch."))
        for i, shape in shapes.items()
    ]
    for shape, dtype, kind in kw.get("_scratch_shapes", []):
        if kind == "vmem":
            refs.append((tuple(shape), str(dtype).removeprefix("jnp.")))
    return refs


_MLP_BLOCKS = [16, 16, 128, 128, 16, 128, 128]


def _ring_depths(code: str) -> list[int]:
    """Slots of each weight ring (one DMA semaphore array per ring)."""
    return [
        int(depth)
        for depth in re.findall(r"\(\((\d+),\), None, 'dma_semaphore'\)", code)
    ]


def _ring_tile(name: str) -> str:
    """A regex for ``name``'s tile read from a ring slot or an arena slot."""
    return rf"{name}_tile = (ring(_\d+)?\.at\[|_helion_ring_tile\(ring(_\d+)?, )"


@contextlib.contextmanager
def _per_weight_rings() -> Iterator[None]:
    """Per-weight rings instead of an arena (``pallas_stream_arena``) in
    default configs, for tests of what only they do: slots shared by shape,
    per-ring depths and floors, copies held at an exchange.  A bound kernel
    keeps the default config it was bound with, so the bound kernels of this
    module are dropped on entry and exit."""
    kernels = [v for v in globals().values() if isinstance(v, helion.Kernel)]
    for kernel in kernels:
        kernel.reset()
    with mock.patch.object(megakernel, "_ARENA_BY_DEFAULT", False):
        yield
    for kernel in kernels:
        kernel.reset()


def _glu_reference(
    y: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    g = y.float() @ w_gate.float()
    u = y.float() @ w_up.float()
    return torch.nn.functional.silu(g) * u


def _import_module(code: str, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(code, name, "exec"), module.__dict__)
    return module


_STACK_BLOCKS = [16, 16, 16, 128, 128, 16, 128, 128, 16]


def _mlp_stack_args(
    m: int, layers: int, h: int = 256, inter: int = 512
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return (
        torch.randn(m, h, device=DEVICE, dtype=dtype),
        torch.randn(layers, h, device=DEVICE, dtype=dtype),
        torch.randn(layers, h, inter, device=DEVICE, dtype=dtype) * 0.05,
        torch.randn(layers, h, inter, device=DEVICE, dtype=dtype) * 0.05,
        torch.randn(layers, inter, h, device=DEVICE, dtype=dtype) * 0.05,
    )


def _mlp_stack_reference(
    x: torch.Tensor,
    norm_w: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    s = x.float()
    for i in range(w_gate.size(0)):
        rms = torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps)
        normed = (s * rms * norm_w[i].float()).to(x.dtype)
        g = (normed.float() @ w_gate[i].float()).to(x.dtype)
        u = (normed.float() @ w_up[i].float()).to(x.dtype)
        act = torch.nn.functional.silu(g.float()).to(x.dtype) * u
        s = act.float() @ w_down[i].float() + s
    return s.to(x.dtype)


def _gemv_chain_args(m: int, layers: int, h: int = 256) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return (
        torch.randn(m, h, device=DEVICE, dtype=dtype),
        torch.randn(h, h, device=DEVICE, dtype=dtype) * 0.06,
        torch.randn(layers, h, h, device=DEVICE, dtype=dtype) * 0.06,
        torch.randn(layers, h, h, device=DEVICE, dtype=dtype) * 0.06,
        torch.randn(h, h, device=DEVICE, dtype=dtype) * 0.06,
    )


def _gemv_chain_reference(
    x: torch.Tensor,
    w_in: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w_out: torch.Tensor,
    start: int = 0,
) -> torch.Tensor:
    hidden = x.float() @ w_in.float()
    for i in range(start, a.size(0)):
        t = (hidden.to(x.dtype).float() @ a[i].float()).to(x.dtype)
        hidden = hidden + (t.float() @ b[i].float()) / (i + 1)
    return (hidden.to(x.dtype).float() @ w_out.float()).to(x.dtype)


def _tp_ranks(world: int, rank: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``peers`` ([1, world - 1], the other ranks) and ``rank`` ([1]) inputs."""
    others = [(rank + off) % world for off in range(1, world)]
    return (
        torch.tensor([others], device=DEVICE, dtype=torch.int32),
        torch.tensor([rank], device=DEVICE, dtype=torch.int32),
    )


def _periodic_stack_args(
    periods: int, period: int, h: int = 256, rank: int = 128, inter: int = 512
) -> tuple[Any, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    low_rank = periods * (period - 1)
    return (
        torch.randn(16, h, device=DEVICE, dtype=dtype),
        torch.randn(low_rank, h, rank, device=DEVICE, dtype=dtype) * 0.06,
        torch.randn(low_rank, rank, h, device=DEVICE, dtype=dtype) * 0.06,
        torch.randn(periods, h, device=DEVICE),
        torch.randn(periods, h, inter, device=DEVICE, dtype=dtype) * 0.05,
        torch.randn(periods, inter, h, device=DEVICE, dtype=dtype) * 0.05,
        period,
    )


def _periodic_stack_reference(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    norm_w: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    period: int,
) -> torch.Tensor:
    s = x.float()
    for p in range(w_up.size(0)):
        for r in range(period - 1):
            k = (period - 1) * p + r
            t = (s.to(x.dtype).float() @ a[k].float()).to(x.dtype)
            s = s + t.float() @ b[k].float()
        rms = torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + 1e-6)
        normed = (s * rms * norm_w[p]).to(x.dtype)
        up = normed.float() @ w_up[p].float()
        s = s + torch.nn.functional.silu(up).to(x.dtype).float() @ w_down[p].float()
    return s.to(x.dtype)


def _attention_args(
    pos: int, nkv: int = 2, group: int = 2, d: int = 128, ctx: int = 256
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return (
        torch.randn(nkv, group, d, device=DEVICE, dtype=dtype),
        torch.randn(nkv, d, device=DEVICE, dtype=dtype),
        torch.randn(nkv, d, device=DEVICE, dtype=dtype),
        torch.randn(nkv, ctx, d, device=DEVICE, dtype=dtype),
        torch.randn(nkv, ctx, d, device=DEVICE, dtype=dtype),
        torch.tensor([pos], device=DEVICE, dtype=torch.int32),
    )


def _attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    p: int,
) -> torch.Tensor:
    """Decode attention over the cache rows up to ``p``, with the kernels'
    bf16 rounding points."""
    qs = (q.float() * q.size(-1) ** -0.5).to(q.dtype)
    s = qs.float() @ k_cache[:, : p + 1].float().transpose(1, 2)
    probs = torch.softmax(s, -1).to(q.dtype)
    return (probs.float() @ v_cache[:, : p + 1].float()).to(q.dtype)


def _gqa_layer_args(
    pos: int, h: int = 256, heads: int = 4, nkv: int = 2, d: int = 128, ctx: int = 256
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return (
        torch.randn(1, h, device=DEVICE, dtype=dtype),
        torch.randn(h, device=DEVICE, dtype=dtype),
        torch.randn(h, heads * d, device=DEVICE, dtype=dtype) * 0.06,
        torch.randn(h, nkv * d, device=DEVICE, dtype=dtype) * 0.06,
        torch.randn(h, nkv * d, device=DEVICE, dtype=dtype) * 0.06,
        torch.randn(heads * d, h, device=DEVICE, dtype=dtype) * 0.06,
        torch.randn(nkv, ctx, d, device=DEVICE, dtype=dtype),
        torch.randn(nkv, ctx, d, device=DEVICE, dtype=dtype),
        torch.tensor([pos], device=DEVICE, dtype=torch.int32),
    )


def _gqa_layer_reference(
    hidden: torch.Tensor,
    norm_w: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The layer output and the written caches."""
    dt = hidden.dtype
    nkv, _, d = k_cache.shape
    p = int(pos[0])
    s = hidden.float()
    x = (
        s * torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps) * norm_w.float()
    ).to(dt)
    q = (x.float() @ wq.float()).to(dt)
    k_cache, v_cache = k_cache.clone(), v_cache.clone()
    k_cache[:, p] = (x.float() @ wk.float()).to(dt).reshape(nkv, d)
    v_cache[:, p] = (x.float() @ wv.float()).to(dt).reshape(nkv, d)
    attn = _attention_reference(q.reshape(nkv, -1, d), k_cache, v_cache, p)
    out = hidden + (attn.reshape(1, -1).float() @ wo.float()).to(dt)
    return out, k_cache, v_cache


def _gated_stack_args(
    gates: int, layers: int = 3, h: int = 2048, rank: int = 128
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return (
        torch.randn(16, h, device=DEVICE, dtype=dtype),
        torch.randn(layers, h, gates, device=DEVICE, dtype=dtype) * 0.02,
        torch.randn(layers, h, rank, device=DEVICE, dtype=dtype) * 0.02,
        torch.randn(layers, rank, h, device=DEVICE, dtype=dtype) * 0.05,
    )


def _gated_stack_reference(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    s = x.float()
    for i in range(w_gate.size(0)):
        a = s.to(x.dtype).float()
        gates = torch.sigmoid(a @ w_gate[i].float())
        t = ((a @ w_up[i].float()) * gates.mean(-1, keepdim=True)).to(x.dtype)
        s = s + t.float() @ w_down[i].float()
    return s.to(x.dtype)


def _periodic_tap_stack_args(
    periods: int, period: int, h: int = 2048, taps: int = 4, inter: int = 256
) -> tuple[Any, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tap_layers = periods * (period - 1)
    return (
        torch.randn(16, h, device=DEVICE, dtype=dtype),
        torch.randn(tap_layers, h, taps, device=DEVICE, dtype=dtype) * 0.03,
        torch.randn(periods, h, inter, device=DEVICE, dtype=dtype) * 0.02,
        torch.randn(periods, inter, h, device=DEVICE, dtype=dtype) * 0.05,
        period,
    )


def _periodic_tap_stack_reference(
    x: torch.Tensor,
    taps: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    period: int,
) -> torch.Tensor:
    s = x.float()
    for p in range(w_up.size(0)):
        for r in range(period - 1):
            g = s.to(x.dtype).float() @ taps[(period - 1) * p + r].float()
            s = s * torch.sigmoid(g).mean(-1, keepdim=True)
        act = torch.nn.functional.silu(s.to(x.dtype).float() @ w_up[p].float())
        s = s + act.to(x.dtype).float() @ w_down[p].float()
    return s.to(x.dtype)


def _recurrent_state_stack_args(
    layers: int, m: int = 8, h: int = 256, taps: int = 4, rank: int = 8
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return (
        torch.randn(m, h, device=DEVICE, dtype=dtype),
        torch.randn(layers, h, taps, device=DEVICE, dtype=dtype),
        torch.randn(layers, rank, h, device=DEVICE, dtype=torch.float32),
        torch.randn(layers, h, 512, device=DEVICE, dtype=dtype) * 0.05,
        torch.randn(layers, 512, h, device=DEVICE, dtype=dtype) * 0.05,
    )


def _recurrent_state_stack_reference(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    rec_state: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The output and the updated states, from copies of the states."""
    conv_state, rec_state = conv_state.clone(), rec_state.clone()
    s = x.float()
    for i in range(w_up.size(0)):
        row = s[0].to(x.dtype)
        conv_state[i] = torch.cat([conv_state[i][:, 1:], row[:, None]], dim=-1)
        rec_state[i] = rec_state[i] * 0.5 + conv_state[i].float().sum(-1)[None, :]
        s = s + torch.tanh(rec_state[i]).mean(0)[None, :]
        act = torch.nn.functional.silu(s.to(x.dtype).float() @ w_up[i].float())
        s = s + act.to(x.dtype).float() @ w_down[i].float()
    return s.to(x.dtype), conv_state, rec_state


def _state_snapshot_stack_reference(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    rec_state: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """The output, the updated states and the per-row snapshots, from
    copies of the states."""
    conv_state, rec_state = conv_state.clone(), rec_state.clone()
    m = x.size(0)
    conv_snap = conv_state[:, None].repeat(1, m, 1, 1)
    rec_snap = rec_state[:, None].repeat(1, m, 1, 1)
    s = x.float()
    for i in range(w_up.size(0)):
        for b in range(m):
            row = s[b].to(x.dtype)
            conv_state[i] = torch.cat([conv_state[i][:, 1:], row[:, None]], dim=-1)
            rec_state[i] = rec_state[i] * 0.5 + conv_state[i].float().sum(-1)[None, :]
            conv_snap[i, b], rec_snap[i, b] = conv_state[i], rec_state[i]
        s = s + torch.tanh(rec_state[i]).mean(0)[None, :]
        act = torch.nn.functional.silu(s.to(x.dtype).float() @ w_up[i].float())
        s = s + act.to(x.dtype).float() @ w_down[i].float()
    return s.to(x.dtype), conv_state, rec_state, conv_snap, rec_snap


def _transposed_snapshot_stack_reference(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_w: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """The output, the updated conv state and the per-row snapshots, from a
    copy of the state."""
    conv_state = conv_state.clone()
    m, taps = x.size(0), conv_state.size(2)
    conv_snap = []
    s = x.float()
    for i in range(w_up.size(0)):
        inputs = torch.cat([conv_state[i].transpose(0, 1)[1:], s.to(x.dtype)], 0)
        conv_snap.append(
            torch.stack([inputs[b : b + taps].transpose(0, 1) for b in range(m)])
        )
        conv_state[i] = inputs[m - 1 : m - 1 + taps].transpose(0, 1)
        s = s + inputs[:m].float() * conv_w[i, :, taps - 1].float()
        act = torch.nn.functional.silu(s.to(x.dtype).float() @ w_up[i].float())
        s = s + act.to(x.dtype).float() @ w_down[i].float()
    return s.to(x.dtype), conv_state, torch.stack(conv_snap)


def _narrow_conv_stack_args(
    layers: int, h: int = 1280, gates: int = 6, taps: int = 4
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return (
        torch.randn(1, h, device=DEVICE, dtype=dtype),
        torch.randn(layers, h, gates, device=DEVICE, dtype=dtype) * 0.03,
        torch.randn(layers, h, taps, device=DEVICE, dtype=dtype),
        torch.randn(layers, h, taps, device=DEVICE, dtype=dtype) * 0.5,
    )


def _narrow_conv_stack_reference(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    conv_state: torch.Tensor,
    conv_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The output and the updated conv state, from a copy of the state."""
    conv_state = conv_state.clone()
    s = x.float()
    for i in range(w_gate.size(0)):
        gate = s.to(x.dtype).float() @ w_gate[i].float()
        row = s[0].to(x.dtype)
        conv_state[i] = torch.cat([conv_state[i][:, 1:], row[:, None]], dim=-1)
        mixed = (conv_state[i].float() * conv_w[i].float()).sum(-1)
        s = s * torch.sigmoid(gate).mean(-1, keepdim=True) + mixed[None, :]
    return s.to(x.dtype), conv_state


def _narrow_gate_stack_args(
    layers: int, h: int = 1280, gates: int = 6, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return (
        torch.randn(1, h, device=DEVICE, dtype=dtype),
        torch.randn(layers, h, gates, device=DEVICE, dtype=dtype) * 0.03,
        torch.randn(layers, gates, device=DEVICE, dtype=dtype) * 0.5,
        torch.randn(layers, gates, device=DEVICE, dtype=dtype),
    )


def _narrow_gate_stack_reference(
    x: torch.Tensor, w_a: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor
) -> torch.Tensor:
    s = x.float()
    for i in range(w_a.size(0)):
        gate = s.to(w_a.dtype).float() @ w_a[i].float()
        decay = torch.exp(a_log[i].float()) * torch.sigmoid(
            gate + dt_bias[i].float()[None, :]
        )
        s = s * (0.5 + decay.mean(-1, keepdim=True))
    return s.to(x.dtype)


def _cache_stack_args(
    layers: int, p: int, nkv: int = 2, group: int = 2, d: int = 128, ctx: int = 256
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    h = nkv * group * d
    return (
        torch.randn(1, h, device=DEVICE, dtype=dtype),
        torch.randn(layers, h, nkv * d, device=DEVICE, dtype=dtype) * 0.05,
        torch.randn(layers, h, nkv * d, device=DEVICE, dtype=dtype) * 0.05,
        torch.randn(layers, h, h, device=DEVICE, dtype=dtype) * 0.05,
        torch.randn(layers, nkv, ctx, d, device=DEVICE, dtype=dtype),
        torch.randn(layers, nkv, ctx, d, device=DEVICE, dtype=dtype),
        torch.tensor([p], device=DEVICE, dtype=torch.int32),
    )


def _cache_stack_reference(
    x: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The output and the updated caches, from copies of the caches."""
    k_cache, v_cache = k_cache.clone(), v_cache.clone()
    p = int(pos[0])
    _, nkv, _, d = k_cache.shape
    s = x[0].float()
    for i in range(w_k.size(0)):
        a = s.to(x.dtype).float()
        k_cache[i, :, p] = (a @ w_k[i].float()).to(x.dtype).view(nkv, d)
        v_cache[i, :, p] = (a @ w_v[i].float()).to(x.dtype).view(nkv, d)
        q = (s * d**-0.5).to(x.dtype).float().view(nkv, -1, d)
        scores = q @ k_cache[i].float().transpose(1, 2)
        scores[:, :, p + 1 :] = float("-inf")
        probs = torch.softmax(scores, -1)
        attn = (probs @ v_cache[i].float()).to(x.dtype).view(-1)
        s = s + attn.float() @ w_o[i].float()
    return s.to(x.dtype)[None], k_cache, v_cache


def _fused_kv_rows_reference(
    x: torch.Tensor,
    w_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The updated caches, from copies of the caches."""
    k_cache, v_cache = k_cache.clone(), v_cache.clone()
    p = int(pos[0])
    _, nkv, _, d = k_cache.shape
    for i in range(w_kv.size(0)):
        row = (x[0].float() @ w_kv[i].float()).to(x.dtype)
        k = row[: nkv * d].view(nkv, d)
        k_cache[i, :, p] = torch.cat([-k[:, d // 2 :], k[:, : d // 2]], dim=-1)
        v_cache[i, :, p] = row[nkv * d :].view(nkv, d)
    return k_cache, v_cache


def _lane_dense_off(kernel: Any, args: tuple[Any, ...]) -> dict[str, Any]:
    """The default config of ``kernel`` for ``args``, with every input passed
    in its logical layout."""
    config = kernel.bind(args).config_spec.default_config()
    return {**config.config, "pallas_lane_dense": False}


def _site_shapes(
    bound: Any, **config: object
) -> list[tuple[int | None, tuple[int, ...]]]:
    model = bound.config_spec.pallas_stream_model_for(config)
    assert model is not None
    return [
        (site.block_id, tuple(load.fake.shape))
        for site in model.sites
        for load in site.loads
    ]


def _rms_norm(s: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    s = s.float()
    return s * torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps) * w.float()


def _masked_logits(
    normed: torch.Tensor, lm_head: torch.Tensor, vocab: int
) -> torch.Tensor:
    logits = normed.float() @ lm_head.float()
    logits[:, vocab:] = float("-inf")
    return logits


def _embed_head_args(
    token: int, v: int, h: int, vp: int, vocab: int
) -> tuple[object, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    lm_head = torch.randn(h, vp, device=DEVICE, dtype=dtype) * 0.05
    # Large padded columns, which only the vocab mask hides.
    lm_head[:, vocab:] = 1.0
    return (
        torch.tensor([token], device=DEVICE, dtype=torch.int32),
        torch.randn(v, h, device=DEVICE, dtype=dtype),
        torch.randn(1, h, device=DEVICE, dtype=dtype),
        (1.0 + 0.1 * torch.randn(h, device=DEVICE)).to(dtype),
        lm_head,
        vocab,
    )


def _planted_top1_args(
    h: int, vp: int, vocab: int, first: int, second: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A hidden state of ones and an LM head whose columns ``first`` and
    ``second`` are equal and give the largest logit, with a larger one in a
    padded column: the top-1 is ``first``."""
    torch.manual_seed(0)
    lm_head = torch.randn(h, vp, device=DEVICE) * 2e-3
    lm_head[:, first] = 1e-2
    lm_head[:, second] = 1e-2
    lm_head[:, vocab:] = 2e-2
    return (
        torch.ones(1, h, device=DEVICE, dtype=torch.bfloat16),
        lm_head.to(torch.bfloat16),
    )


def _dense_step_args(
    layers: int, v: int = 4000, h: int = 256, inter: int = 512, vp: int = 2048
) -> tuple[object, ...]:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    lm_head = torch.randn(h, vp, device=DEVICE, dtype=dtype) * 0.05
    lm_head[:, vp - 48 :] = 1.0
    return (
        torch.tensor([v - 7], device=DEVICE, dtype=torch.int32),
        torch.randn(v, h, device=DEVICE, dtype=dtype),
        (1.0 + 0.1 * torch.randn(layers, h, device=DEVICE)).to(dtype),
        torch.randn(layers, h, inter, device=DEVICE, dtype=dtype) * 0.05,
        torch.randn(layers, h, inter, device=DEVICE, dtype=dtype) * 0.05,
        torch.randn(layers, inter, h, device=DEVICE, dtype=dtype) * 0.05,
        (1.0 + 0.1 * torch.randn(h, device=DEVICE)).to(dtype),
        lm_head,
        vp - 48,
    )


def _dense_step_logits(
    token: torch.Tensor,
    embedding: torch.Tensor,
    norm_w: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    vocab: int,
) -> torch.Tensor:
    dt = embedding.dtype
    s = embedding[int(token[0])][None, :].float()
    for i in range(w_gate.size(0)):
        normed = _rms_norm(s, norm_w[i]).to(dt)
        g = (normed.float() @ w_gate[i].float()).to(dt)
        u = (normed.float() @ w_up[i].float()).to(dt)
        act = torch.nn.functional.silu(g.float()).to(dt) * u
        s = act.float() @ w_down[i].float() + s
    return _masked_logits(_rms_norm(s, final_norm).to(dt), lm_head, vocab)


@helion.kernel(backend="pallas", static_shapes=True)
def routed_experts(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """A token's top-k experts, whose ids and weights are inputs."""
    m, h = x.shape
    inter = w_gate.size(2)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.zeros([m, h], dtype=torch.float32, device=x.device)
    for k in range(topk_idx.size(1)):
        for tm, tn in hl.tile([m, inter]):
            e = topk_idx[0, k]
            g = hl.zeros([tm, tn], dtype=torch.float32)
            u = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = x[tm, tk]
                g = torch.addmm(g, a, w_gate[e, tk, tn])
                u = torch.addmm(u, a, w_up[e, tk, tn])
            act[tm, tn] = (torch.nn.functional.silu(g) * u).to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            e = topk_idx[0, k]
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                acc = torch.addmm(acc, act[tm, tk], w_down[e, tk, tn])
            out[tm, tn] = out[tm, tn] + topk_w[0, k] * acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def routed_experts_in_kernel_topk(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    topk: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Router GEMV and top-k (argmax and mask) in the kernel, then the
    experts it selects, which read their ids from its output."""
    m, h = x.shape
    n_exp = w_router.size(1)
    inter = w_gate.size(2)
    logits = torch.empty([m, n_exp], dtype=torch.float32, device=x.device)
    sel_idx = torch.empty([m, topk], dtype=torch.int32, device=x.device)
    sel_w = torch.empty([m, topk], dtype=torch.float32, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.zeros([m, h], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        logits[tm, :] = hl.dot(x[tm, :], w_router[:, :], out_dtype=torch.float32)
    for tm in hl.tile(m):
        row = torch.sigmoid(logits[tm, :])
        lane = hl.arange(n_exp)
        total = hl.zeros([tm], dtype=torch.float32)
        for k in hl.static_range(topk):
            best = torch.amax(row, -1)
            arg = torch.argmax(row, -1).to(torch.int32)
            sel_idx[tm, k] = arg
            sel_w[tm, k] = best
            total = total + best
            row = torch.where(lane[None, :] == arg[:, None], -1.0, row)
        for k in hl.static_range(topk):
            sel_w[tm, k] = sel_w[tm, k] / total
    for k in range(topk):
        for tm, tn in hl.tile([m, inter]):
            e = sel_idx[0, k]
            g = hl.zeros([tm, tn], dtype=torch.float32)
            u = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = x[tm, tk]
                g = torch.addmm(g, a, w_gate[e, tk, tn])
                u = torch.addmm(u, a, w_up[e, tk, tn])
            act[tm, tn] = (torch.nn.functional.silu(g) * u).to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            e = sel_idx[0, k]
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                acc = torch.addmm(acc, act[tm, tk], w_down[e, tk, tn])
            out[tm, tn] = out[tm, tn] + sel_w[0, k] * acc
    return out, sel_idx, sel_w


@helion.kernel(backend="pallas", static_shapes=True)
def shared_and_routed_experts(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    s_gate: torch.Tensor,
    s_up: torch.Tensor,
    s_down: torch.Tensor,
) -> torch.Tensor:
    """A shared expert and a residual, then the routed experts."""
    m, h = x.shape
    inter = w_gate.size(2)
    s_inter = s_gate.size(1)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    s_act = torch.empty([m, s_inter], dtype=x.dtype, device=x.device)
    out = torch.empty([m, h], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, s_inter]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            a = x[tm, tk]
            g = torch.addmm(g, a, s_gate[tk, tn])
            u = torch.addmm(u, a, s_up[tk, tn])
        s_act[tm, tn] = (torch.nn.functional.silu(g) * u).to(x.dtype)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(s_inter):
            acc = torch.addmm(acc, s_act[tm, tk], s_down[tk, tn])
        out[tm, tn] = x[tm, tn].float() + acc
    for k in range(topk_idx.size(1)):
        for tm, tn in hl.tile([m, inter]):
            e = topk_idx[0, k]
            g = hl.zeros([tm, tn], dtype=torch.float32)
            u = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = x[tm, tk]
                g = torch.addmm(g, a, w_gate[e, tk, tn])
                u = torch.addmm(u, a, w_up[e, tk, tn])
            act[tm, tn] = (torch.nn.functional.silu(g) * u).to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            e = topk_idx[0, k]
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                acc = torch.addmm(acc, act[tm, tk], w_down[e, tk, tn])
            out[tm, tn] = out[tm, tn] + topk_w[0, k] * acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def batch_routes(
    x: torch.Tensor,
    route_idx: torch.Tensor,
    route_coef: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Several tokens' routes, combined densely: each route's expert runs on
    every row, weighted by a per-row coefficient (zero for the rows the
    route does not serve)."""
    m, h = x.shape
    inter = w_gate.size(2)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.zeros([m, h], dtype=torch.float32, device=x.device)
    for r in range(route_idx.size(0)):
        for tm, tn in hl.tile([m, inter]):
            e = route_idx[r]
            g = hl.zeros([tm, tn], dtype=torch.float32)
            u = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                a = x[tm, tk]
                g = torch.addmm(g, a, w_gate[e, tk, tn])
                u = torch.addmm(u, a, w_up[e, tk, tn])
            act[tm, tn] = (torch.nn.functional.silu(g) * u).to(x.dtype)
        for tm, tn in hl.tile([m, h]):
            e = route_idx[r]
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                acc = torch.addmm(acc, act[tm, tk], w_down[e, tk, tn])
            out[tm, tn] = out[tm, tn] + route_coef[r, tm][:, None] * acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def root_level_experts(
    x: torch.Tensor, topk_idx: torch.Tensor, w_gate: torch.Tensor
) -> torch.Tensor:
    """Expert weights read whole along K in each root tile."""
    m = x.size(0)
    inter = w_gate.size(2)
    out = torch.zeros([m, inter], dtype=torch.float32, device=x.device)
    for k in range(topk_idx.size(1)):
        for tm, tn in hl.tile([m, inter]):
            e = topk_idx[0, k]
            out[tm, tn] = out[tm, tn] + hl.dot(
                x[tm, :], w_gate[e, :, tn], out_dtype=torch.float32
            )
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def experts_and_routes_out(
    x: torch.Tensor, topk_idx: torch.Tensor, w_gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert weights by id, from ids that a root also copies out as a tile,
    so they stay in VMEM."""
    m = x.size(0)
    inter = w_gate.size(2)
    routes = torch.empty_like(topk_idx)
    out = torch.zeros([m, inter], dtype=torch.float32, device=x.device)
    for tr in hl.tile(topk_idx.size(0)):
        routes[tr, :] = topk_idx[tr, :]
    for k in range(topk_idx.size(1)):
        for tm, tn in hl.tile([m, inter]):
            e = topk_idx[0, k]
            out[tm, tn] = out[tm, tn] + hl.dot(
                x[tm, :], w_gate[e, :, tn], out_dtype=torch.float32
            )
    return out, routes


@helion.kernel(backend="pallas", static_shapes=True)
def grid_slot_experts(
    x: torch.Tensor, topk_idx: torch.Tensor, w_gate: torch.Tensor
) -> torch.Tensor:
    """Expert ids read at a grid index, which the ring cannot follow."""
    m, h = x.shape
    inter = w_gate.size(2)
    y = torch.empty([m, h], dtype=x.dtype, device=x.device)
    out = torch.empty(
        [topk_idx.size(1), m, inter], dtype=torch.float32, device=x.device
    )
    for tm, th in hl.tile([m, h]):
        y[tm, th] = x[tm, th] * 2.0
    for k in hl.grid(topk_idx.size(1)):
        e = topk_idx[0, k]
        acc = hl.zeros([m, inter], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = torch.addmm(acc, y[:, tk], w_gate[e, tk, :])
        out[k, :, :] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def permuted_experts(
    x: torch.Tensor, topk_idx: torch.Tensor, perm: torch.Tensor, w_gate: torch.Tensor
) -> torch.Tensor:
    """Expert ids read at a runtime index (two levels of indirection)."""
    m, h = x.shape
    inter = w_gate.size(2)
    out = torch.zeros([m, inter], dtype=torch.float32, device=x.device)
    for k in range(topk_idx.size(1)):
        for tm, tn in hl.tile([m, inter]):
            e = perm[topk_idx[0, k]]
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = torch.addmm(acc, x[tm, tk], w_gate[e, tk, tn])
            out[tm, tn] = out[tm, tn] + acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def routed_layer_stack(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    topk: hl.constexpr,
) -> torch.Tensor:
    """Folded layers that each route the hidden state in the kernel and run
    the experts it picks: every layer reads ids the layer itself writes."""
    m, h = x.shape
    layers, _, n_exp = w_router.shape
    inter = w_gate.size(3)
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    logits = torch.empty([m, n_exp], dtype=torch.float32, device=x.device)
    sel_idx = torch.empty([m, topk], dtype=torch.int32, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for tm in hl.tile(m):
            logits[tm, :] = hl.dot(
                hidden[tm, :].to(x.dtype), w_router[i, :, :], out_dtype=torch.float32
            )
        for tm in hl.tile(m):
            row = logits[tm, :]
            lane = hl.arange(n_exp)
            for k in hl.static_range(topk):
                arg = torch.argmax(row, -1).to(torch.int32)
                sel_idx[tm, k] = arg
                row = torch.where(lane[None, :] == arg[:, None], float("-inf"), row)
        for k in hl.static_range(topk):
            for tm, tn in hl.tile([m, inter]):
                e = sel_idx[0, k]
                g = hl.zeros([tm, tn], dtype=torch.float32)
                u = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(h):
                    a = hidden[tm, tk].to(x.dtype)
                    g = torch.addmm(g, a, w_gate[i, e, tk, tn])
                    u = torch.addmm(u, a, w_up[i, e, tk, tn])
                act[tm, tn] = (torch.nn.functional.silu(g) * u).to(x.dtype)
            for tm, tn in hl.tile([m, h]):
                e = sel_idx[0, k]
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(inter):
                    acc = torch.addmm(acc, act[tm, tk], w_down[i, e, tk, tn])
                hidden[tm, tn] = hidden[tm, tn] + acc
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def projected_layer_stack(
    x: torch.Tensor,
    w_proj: torch.Tensor,
    w_router: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    topk: hl.constexpr,
) -> torch.Tensor:
    """``routed_layer_stack`` with a dense projection in front of each
    layer's router: weights no root's ids gate, beside experts that the
    layer's top-k gates."""
    m, h = x.shape
    layers, _, n_exp = w_router.shape
    inter = w_gate.size(3)
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)
    mixed = torch.empty([m, h], dtype=x.dtype, device=x.device)
    logits = torch.empty([m, n_exp], dtype=torch.float32, device=x.device)
    sel_idx = torch.empty([m, topk], dtype=torch.int32, device=x.device)
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)
    out = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        hidden[tm, :] = x[tm, :].float()
    for i in range(layers):
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = torch.addmm(acc, hidden[tm, tk].to(x.dtype), w_proj[i, tk, tn])
            mixed[tm, tn] = (hidden[tm, tn] + acc).to(x.dtype)
        for tm in hl.tile(m):
            logits[tm, :] = hl.dot(
                mixed[tm, :], w_router[i, :, :], out_dtype=torch.float32
            )
        for tm in hl.tile(m):
            row = logits[tm, :]
            lane = hl.arange(n_exp)
            for k in hl.static_range(topk):
                arg = torch.argmax(row, -1).to(torch.int32)
                sel_idx[tm, k] = arg
                row = torch.where(lane[None, :] == arg[:, None], float("-inf"), row)
        for k in hl.static_range(topk):
            for tm, tn in hl.tile([m, inter]):
                e = sel_idx[0, k]
                g = hl.zeros([tm, tn], dtype=torch.float32)
                u = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(h):
                    a = mixed[tm, tk]
                    g = torch.addmm(g, a, w_gate[i, e, tk, tn])
                    u = torch.addmm(u, a, w_up[i, e, tk, tn])
                act[tm, tn] = (torch.nn.functional.silu(g) * u).to(x.dtype)
            for tm, tn in hl.tile([m, h]):
                e = sel_idx[0, k]
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(inter):
                    acc = torch.addmm(acc, act[tm, tk], w_down[i, e, tk, tn])
                hidden[tm, tn] = hidden[tm, tn] + acc
    for tm in hl.tile(m):
        out[tm, :] = hidden[tm, :].to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def remapped_routes(
    x: torch.Tensor, ids: torch.Tensor, w_in: torch.Tensor, w_gate: torch.Tensor
) -> torch.Tensor:
    """A projection, then a multi-tile root that remaps the expert ids, then
    steps that read them last row first."""
    m, h = x.shape
    n_exp, _, inter = w_gate.shape
    steps = ids.size(0)
    hidden = torch.empty([m, h], dtype=x.dtype, device=x.device)
    route = torch.empty_like(ids)
    out = torch.zeros([m, inter], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, h]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            acc = torch.addmm(acc, x[tm, tk], w_in[tk, tn])
        hidden[tm, tn] = acc.to(x.dtype)
    for tr in hl.tile(steps):
        route[tr, :] = (ids[tr, :] + 1) % n_exp
    for r in range(steps):
        for tm, tn in hl.tile([m, inter]):
            e = route[steps - 1 - r, 0]
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = torch.addmm(acc, hidden[tm, tk], w_gate[e, tk, tn])
            out[tm, tn] = out[tm, tn] + acc
    return out


def _expert_weights(
    experts: int, h: int, inter: int, layers: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.bfloat16
    lead = (experts,) if layers is None else (layers, experts)
    return (
        (torch.randn(*lead, h, inter, device=DEVICE) / 16).to(dtype),
        (torch.randn(*lead, h, inter, device=DEVICE) / 16).to(dtype),
        (torch.randn(*lead, inter, h, device=DEVICE) / 22).to(dtype),
    )


def _swiglu(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    g = x.float() @ w_gate.float()
    u = x.float() @ w_up.float()
    act = (torch.nn.functional.silu(g) * u).to(x.dtype)
    return act.float() @ w_down.float()


def _routed_experts_args(
    ids: list[int], experts: int = 8, h: int = 256, inter: int = 256
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    x = torch.randn(1, h, device=DEVICE, dtype=torch.bfloat16)
    topk_idx = torch.tensor([ids], device=DEVICE, dtype=torch.int32)
    topk_w = torch.rand(1, len(ids), device=DEVICE)
    return (x, topk_idx, topk_w, *_expert_weights(experts, h, inter))


def _routed_experts_reference(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros(x.size(0), x.size(1), device=DEVICE)
    for k, e in enumerate(topk_idx[0].tolist()):
        out += topk_w[0, k] * _swiglu(x, w_gate[e], w_up[e], w_down[e])
    return out


def _routed_layer_stack_reference(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, list[list[int]]]:
    """The output, and the experts each layer picks."""
    hidden = x.float()
    picks = []
    for i in range(w_router.size(0)):
        logits = hidden.to(x.dtype).float() @ w_router[i].float()
        ids = torch.topk(logits[0], topk).indices.tolist()
        picks.append(ids)
        for e in ids:
            hidden = hidden + _swiglu(
                hidden.to(x.dtype), w_gate[i, e], w_up[i, e], w_down[i, e]
            )
    return hidden.to(x.dtype), picks


def _projected_layer_stack_reference(
    x: torch.Tensor,
    w_proj: torch.Tensor,
    w_router: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, list[list[int]]]:
    """The output, and the experts each layer picks."""
    hidden = x.float()
    picks = []
    for i in range(w_router.size(0)):
        proj = hidden.to(x.dtype).float() @ w_proj[i].float()
        mixed = (hidden + proj).to(x.dtype)
        logits = mixed.float() @ w_router[i].float()
        ids = torch.topk(logits[0], topk).indices.tolist()
        picks.append(ids)
        for e in ids:
            hidden = hidden + _swiglu(mixed, w_gate[i, e], w_up[i, e], w_down[i, e])
    return hidden.to(x.dtype), picks


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
@skipIfRefEager("megakernel codegen is not used in ref eager mode")
class TestPallasMegakernel(TestCase):
    def assertOneSequentialProgram(self, code: str) -> None:
        self.assertEqual(code.count("def _helion_"), 1)
        self.assertEqual(code.count("_launcher(_helion_"), 1)
        self.assertRegex(code, r"_launcher\(_helion_\w+, \(1,\)")
        self.assertNotIn("program_id", code)

    def test_two_independent_roots(self) -> None:
        """Roots with no cross-root hazard keep the multi-program lowering."""
        x = torch.randn(64, 256, device=DEVICE)
        y = torch.randn(32, 128, device=DEVICE)
        bound = two_independent_roots.bind((x, y))
        self.assertFalse(bound.config_spec.pallas_sequential_roots)
        self.assertIsNone(bound.config_spec.pallas_stream_model)
        code = bound.to_triton_code(helion.Config(block_sizes=[32, 128, 16, 128]))
        self.assertIn("pid_shared = pl.program_id(0)", code)
        self.assertNotIn("pl.loop", code)

    def test_two_dependent_roots(self) -> None:
        x = torch.randn(64, 256, device=DEVICE)
        code, out = code_and_output(two_dependent_roots, (x,), block_sizes=[8, 128, 16])
        tmp = x * 2.0
        torch.testing.assert_close(
            out, tmp - tmp.sum(-1, keepdim=True), rtol=1e-4, atol=1e-4
        )
        self.assertOneSequentialProgram(code)
        self.assertIn("@pl.loop(0, 16)", code)
        self.assertIn("@pl.loop(0, 4)", code)
        # tmp is never returned, so it lives in VMEM scratch instead of HBM.
        launch = re.search(r"_launcher\((.*)\)", code)
        assert launch is not None
        self.assertNotIn("tmp", launch.group(1).split("_output_indices")[0])
        self.assertIn("_scratch_shapes=[((64, 256), 'jnp.float32', 'vmem')", code)

    def _check_mlp(self, m: int, **config: object) -> str:
        args = _mlp_args(m)
        code, (out, hidden) = code_and_output(qwen3_dense_mlp, args, **config)
        expected_out, expected_hidden = _mlp_reference(*args)
        torch.testing.assert_close(hidden, expected_hidden)
        torch.testing.assert_close(out, expected_out, rtol=2e-2, atol=2e-2)

        self.assertOneSequentialProgram(code)
        launch = re.search(r"_launcher\((.*)\)", code)
        assert launch is not None
        launch_args = launch.group(1).split("_output_indices")[0]
        for name in ("normed", "act"):
            self.assertNotIn(f" {name},", launch_args)
            # The host only keeps a metadata placeholder for scratch.
            self.assertRegex(code, rf"{name} = .*torch.empty\(.*device='meta'\)")
        rows = -(-m // 16) * 16
        self.assertIn(
            f"_scratch_shapes=[(({rows}, 256), 'jnp.bfloat16', 'vmem'), "
            f"(({rows}, 512), 'jnp.bfloat16', 'vmem')",
            code,
        )
        # Mosaic only slices bf16 refs in whole (16, 128) tiles.
        for shape, dtype in _vmem_refs(qwen3_dense_mlp, args, **config):
            if dtype == "bfloat16" and len(shape) >= 2:
                self.assertEqual(shape[-2] % 16, 0, shape)
        return code

    def test_dense_mlp_m1(self) -> None:
        code = self._check_mlp(1, block_sizes=_MLP_BLOCKS)
        # The one-tile RMSNorm root is straight-line code; the others loop.
        self.assertIn("    pid_shared = 0\n", code)
        self.assertEqual(code.count("@pl.loop("), 2)
        # The single row is padded to a whole bf16 sublane tile.
        self.assertIn("_BLOCK_SIZE_0 = int(16)", code)
        self.assertIn("pl.ds(pl.multiple_of(offset_0, 16), _BLOCK_SIZE_0)", code)

    def test_dense_mlp_m8(self) -> None:
        # The default tiles are the whole weights, one per slot of an arena
        # that holds all three: 64 native tiles each.
        code = self._check_mlp(8)
        self.assertIn("_BLOCK_SIZE_2 = int(512)", code)
        self.assertIn(
            "((192, 16, 128), 'jnp.bfloat16', 'vmem'), ((3,), None, 'dma_semaphore')",
            code,
        )
        # Per-weight rings: one per tile shape.
        with _per_weight_rings():
            code = self._check_mlp(8)
        self.assertIn(
            "((2, 256, 512), 'jnp.bfloat16', 'vmem'), ((2,), None, 'dma_semaphore'), "
            "((1, 512, 256), 'jnp.bfloat16', 'vmem'), ((1,), None, 'dma_semaphore')",
            code,
        )

    def test_dense_mlp_jax_fn(self) -> None:
        import jax
        import jax.numpy as jnp
        import numpy as np

        args = _mlp_args(8)
        bound = qwen3_dense_mlp.bind(args)
        # A shallow ring refills slots, across roots too, inside the kernel.
        config = helion.Config(block_sizes=_MLP_BLOCKS, pallas_stream_depth=4)
        code = bound.to_code(config, options=_JAX_FN)
        self.assertEqual(_ring_depths(code), [4])
        self.assertNotIn("import helion", code)
        self.assertIn("return (results[1], results[0])", code)
        name = "megakernel_dense_mlp_jax_fn"
        try:
            module = _import_module(code, name)
            jax_args = [
                jnp.asarray(a.float().cpu().numpy()).astype(jnp.bfloat16) for a in args
            ]
            out, hidden = jax.jit(module.qwen3_dense_mlp)(*jax_args)
        finally:
            sys.modules.pop(name, None)
        expected_out, expected_hidden = _mlp_reference(*args)
        self.assertEqual(out.dtype, jnp.bfloat16)
        self.assertEqual(hidden.dtype, jnp.float32)
        np.testing.assert_allclose(
            np.asarray(hidden), expected_hidden.cpu().numpy(), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(out).astype(np.float32),
            expected_out.float().cpu().numpy(),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_fori_loop_jax_fn(self) -> None:
        """A fori_loop kernel with DMA scratch and HBM operands runs via jax_fn."""
        import jax
        import jax.numpy as jnp
        import numpy as np

        @helion.kernel(
            backend="pallas",
            static_shapes=True,
            config=helion.Config(
                block_sizes=[128, 128, 128], pallas_loop_type="fori_loop"
            ),
        )
        def fori_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        x = torch.randn(256, 384, device=DEVICE)
        y = torch.randn(384, 256, device=DEVICE)
        code = fori_matmul.bind((x, y)).to_code(options=_JAX_FN)
        self.assertIn("jax.lax.fori_loop", code)
        self.assertIn("_HBM_ARG_INDICES = [0, 1]", code)
        self.assertIn("'dma_semaphore'", code)
        name = "megakernel_fori_loop_jax_fn"
        try:
            module = _import_module(code, name)
            out = jax.jit(module.fori_matmul)(
                jnp.asarray(x.cpu().numpy()), jnp.asarray(y.cpu().numpy())
            )
        finally:
            sys.modules.pop(name, None)
        np.testing.assert_allclose(
            np.asarray(out), (x @ y).cpu().numpy(), rtol=1e-4, atol=1e-3
        )

    def test_full_slice_over_padded_dim(self) -> None:
        """A ``:`` over a padded dim reads only the tensor, not the padding."""
        x = torch.randn(1, 256, device=DEVICE)
        code, out = code_and_output(colsum_scratch, (x,), block_sizes=[8, 128, 128])
        torch.testing.assert_close(out, (x + 1.0).sum(0))
        self.assertIn("load_1 = tmp[0:1, pl.ds(", code)
        x = torch.randn(60, 256, device=DEVICE)
        _, out = code_and_output(colsum_scratch, (x,), block_sizes=[16, 128, 128])
        torch.testing.assert_close(out, (x + 1.0).sum(0), rtol=1e-4, atol=1e-4)
        # The launcher zero-pads x; zeros must not win a max over negatives.
        x = -torch.rand(1, 256, device=DEVICE) - 1.0
        code, (y, out) = code_and_output(colmax_input, (x,), block_sizes=[8, 128, 128])
        torch.testing.assert_close(y, x * 2.0)
        torch.testing.assert_close(out, (x * 3.0).amax(0))
        self.assertIn("x[0:1, pl.ds(", code)

    def test_scratch_padding_reads_zero(self) -> None:
        # Rows 40:64 of tmp are only read by the 32-row tiles: zeroed at start.
        x = torch.randn(40, 256, device=DEVICE)
        code, out = code_and_output(
            tiled_colsum_scratch, (x,), block_sizes=[8, 128, 128, 32]
        )
        torch.testing.assert_close(out, (x + 1.0).sum(0), rtol=1e-4, atol=1e-4)
        self.assertIn("_scratch_shapes=[((64, 256), 'jnp.float32', 'vmem')", code)
        self.assertIn("tmp[40:64, :] = jnp.zeros((24, 256), jnp.float32)", code)
        # The 16-row producer tiles overhang m=36 and must not store x+1 there.
        x = torch.randn(36, 256, device=DEVICE)
        code, out = code_and_output(
            tiled_colsum_scratch, (x,), block_sizes=[16, 128, 128, 32]
        )
        torch.testing.assert_close(out, (x + 1.0).sum(0), rtol=1e-4, atol=1e-4)
        self.assertIn(
            "= jnp.where(offset_0 + lax.broadcasted_iota(jnp.int32, "
            "(_BLOCK_SIZE_0, 1), 0) < 36, v_1, tmp[",
            code,
        )

    def test_begin_offset_tiles_over_scratch(self) -> None:
        x = torch.randn(20, 256, device=DEVICE)
        _, out = code_and_output(offset_read_scratch, (x,), block_sizes=[16, 128, 16])
        expected = torch.zeros_like(x)
        expected[8:] = (x[8:] + 1.0) * 2.0
        torch.testing.assert_close(out, expected)
        # Tiles of 16 rows from row 8 reach row 40 > m=32.
        x = torch.randn(32, 256, device=DEVICE)
        code, out = code_and_output(
            offset_write_scratch, (x,), block_sizes=[8, 16, 8, 128]
        )
        expected = x + 1.0
        expected[:8] = x[:8] * 3.0
        torch.testing.assert_close(out, expected * 2.0)
        self.assertIn("_scratch_shapes=[((48, 256), 'jnp.float32', 'vmem')", code)

    def test_barrier(self) -> None:
        x = torch.randn(64, 256, device=DEVICE)
        code, out = code_and_output(barrier_roots, (x,), block_sizes=[8, 128, 8, 128])
        torch.testing.assert_close(out, (x + 1.0) * 2.0)
        self.assertOneSequentialProgram(code)
        self.assertEqual(code.count("@pl.loop(0, 16)"), 2)

    def test_grid_root(self) -> None:
        x = torch.randn(64, 256, device=DEVICE)
        code, out = code_and_output(grid_root, (x,), block_sizes=[8, 128])
        torch.testing.assert_close(out, (x + 1.0) * 2.0)
        self.assertOneSequentialProgram(code)
        self.assertIn("@pl.loop(0, 8)", code)

    def test_inplace_input(self) -> None:
        x = torch.randn(64, 256, device=DEVICE)
        expected_x = x + 1.0
        _, out = code_and_output(inplace_input, (x,), block_sizes=[8, 128, 16])
        torch.testing.assert_close(x, expected_x)
        torch.testing.assert_close(
            out, expected_x - expected_x.sum(-1, keepdim=True), rtol=1e-4, atol=1e-4
        )

    def test_int_and_none_subscripts(self) -> None:
        x = torch.randn(2, 1, 256, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(256, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(int_and_none_subscripts, (x, w))
        torch.testing.assert_close(out, (x[0] + x[1]) * w[None, :])
        # x[0, tm, :] tiles a second-minor bf16 dim: a whole 16-row tile.
        self.assertIn("_BLOCK_SIZE_0 = int(16)", code)

    def test_weight_ring_depths(self) -> None:
        """Every weight tile streams through one ring, at any depth >= c."""
        for m in (1, 8):
            args = _mlp_args(m)
            expected_out, expected_hidden = _mlp_reference(*args)
            # c=2 tiles per gate/up iteration; G=24 streamed tiles in total.
            for depth in (2, 3, 4, 8, 24):
                with self.subTest(m=m, depth=depth):
                    code, (out, hidden) = code_and_output(
                        qwen3_dense_mlp,
                        args,
                        block_sizes=_MLP_BLOCKS,
                        pallas_stream_depth=depth,
                    )
                    torch.testing.assert_close(hidden, expected_hidden)
                    torch.testing.assert_close(out, expected_out, rtol=2e-2, atol=2e-2)
                    self.assertEqual(_ring_depths(code), [depth])

    def test_stream_unroll(self) -> None:
        """Ring-consumer loops trace ``pallas_stream_unroll`` iterations per
        fori_loop step, the remainder after the loop, and a loop of one step
        in Python.  ``pallas_stream_wait_group`` takes the ring waits of that
        many iterations, at most what every ring holds, before their bodies."""
        args = _mlp_args(8, inter=2048)
        expected_out, expected_hidden = _mlp_reference(*args)
        # gate/up: 2 iterations of 2 tiles; down: 16 iterations of 1 tile.
        # Each case lists the fori_loop trip counts, the step calls of one
        # iteration at a time (range, index), and the groups of step calls
        # (index, range).
        cases = [
            # 32 KiB tiles: 8 iterations per step, so gate/up runs in Python.
            ({}, ["2"], [("2", "_u"), ("8", "8 * _jo + _u_1")], []),
            (
                {"pallas_stream_unroll": 1, "pallas_stream_wait_group": 4},
                [
                    "(256 - 0 + _BLOCK_SIZE_3 - 1) // _BLOCK_SIZE_3",
                    "(2048 - 0 + _BLOCK_SIZE_6 - 1) // _BLOCK_SIZE_6",
                ],
                [],
                [],
            ),
            (
                {"pallas_stream_unroll": 3, "pallas_stream_wait_group": 4},
                ["5"],
                [],
                [("_u", "2"), ("3 * _jo + _u_1", "3"), ("_u_2", "15, 16")],
            ),
            # Two slots hold one gate/up iteration and two down iterations.
            (
                {
                    "pallas_stream_unroll": 3,
                    "pallas_stream_wait_group": 4,
                    "pallas_stream_depth": 2,
                },
                ["5"],
                [("2", "_u")],
                [
                    ("3 * _jo + _u_1", "2"),
                    ("3 * _jo + _u_1", "2, 3"),
                    ("_u_2", "15, 16"),
                ],
            ),
        ]
        for config, trips, steps, groups in cases:
            with self.subTest(**config):
                code, (out, hidden) = code_and_output(
                    qwen3_dense_mlp, args, block_sizes=_MLP_BLOCKS, **config
                )
                torch.testing.assert_close(hidden, expected_hidden)
                torch.testing.assert_close(out, expected_out, rtol=2e-2, atol=2e-2)
                self.assertEqual(
                    re.findall(r"jax\.lax\.fori_loop\(0, ([^,]+), _fori_body", code),
                    trips,
                )
                self.assertEqual(
                    re.findall(
                        r"for _u\w* in range\(([^)]*)\):\n\s*_fori_step\w*\(([^)]*)\)",
                        code,
                    ),
                    steps,
                )
                self.assertEqual(
                    re.findall(
                        r"\[_fori_step\w*\(([^)]*)\) for _u\w* in range\(([^)]*)\)\]",
                        code,
                    ),
                    groups,
                )

    def test_weight_ring_layout(self) -> None:
        args = _mlp_args(8)
        code, _ = code_and_output(
            qwen3_dense_mlp, args, block_sizes=_MLP_BLOCKS, pallas_stream_depth=4
        )
        # One ring and one semaphore array replace the per-tensor buffers.
        self.assertIn(
            "((4, 128, 128), 'jnp.bfloat16', 'vmem'), ((4,), None, 'dma_semaphore')",
            code,
        )
        self.assertEqual(_ring_depths(code), [4])
        self.assertNotIn("_buf", code)
        self.assertNotRegex(code, r"\bw_(gate|up|down)_sem\b")
        # The prologue issues stream indices 0..3 before root 0.
        prologue = re.findall(r"\n    pltpu\.make_async_copy\(.*\.start\(\)", code)
        self.assertEqual(len(prologue), 4)
        self.assertLess(code.index(prologue[-1]), code.index("pid_shared = 0"))
        self.assertIn(
            "pltpu.make_async_copy(w_up.at[pl.ds(128, 128), pl.ds(0, 128)], "
            "ring.at[3], ring_sem.at[3]).start()",
            code,
        )
        # Consumers wait on their slot, then refill it D indices ahead; the
        # gate/up loop's refills cross into the down projection's stream.
        self.assertIn("_g = pid_shared_1 * 4 + _j * 2", code)
        self.assertIn("_slot_1 = (_g + 1) % 4", code)
        self.assertIn("w_up_tile = ring.at[_slot_1]", code)
        self.assertIn("load_5 = w_up_tile[:, :]", code)
        self.assertIn("@pl.when(_gn >= 16)", code)
        self.assertIn("_g_1 = 16 + pid_shared_2 * 4 + _j_1", code)
        self.assertIn("@pl.when(_gn_2 < 24)", code)
        # The default depth holds the whole small stream (G=24).
        code, _ = code_and_output(qwen3_dense_mlp, args, block_sizes=_MLP_BLOCKS)
        self.assertEqual(_ring_depths(code), [24])
        self.assertNotIn("@pl.when", code)

    def test_weight_ring_sub_windows(self) -> None:
        """Tile shapes that fill 90% of a common slot share a ring; the smaller
        tiles use a sub-window of it."""
        for h, inter, blocks, window in (
            # The down projection's (1280, 128) tiles under (1408, 128) slots.
            (
                1408,
                1280,
                [16, 16, 128, 1408, 16, 128, 1280],
                "ring.at[_slot_1, pl.ds(0, 1280), :]",
            ),
            # The down projection's (128, 1280) tiles under (128, 1408) slots.
            (
                1280,
                1408,
                [16, 16, 1408, 128, 16, 1280, 128],
                "ring.at[_slot_1, :, pl.ds(0, 1280)]",
            ),
        ):
            args = _mlp_args(8, h=h, inter=inter)
            expected_out, _ = _mlp_reference(*args)
            with self.subTest(h=h, inter=inter):
                code, (out, _) = code_and_output(
                    qwen3_dense_mlp, args, block_sizes=blocks, pallas_stream_depth=3
                )
                torch.testing.assert_close(out, expected_out, rtol=2e-2, atol=2e-2)
                self.assertEqual(_ring_depths(code), [3])
                self.assertIn(window, code)

    def test_weight_ring_tile_decode(self) -> None:
        """Refills decode the target tile like the root decodes its pid."""
        args = _mlp_args(32)
        expected_out, _ = _mlp_reference(*args)
        for loop_orders in ([[0, 1], [0, 1]], [[1, 0], [1, 0]]):
            with self.subTest(loop_orders=loop_orders):
                code, (out, _) = code_and_output(
                    qwen3_dense_mlp,
                    args,
                    block_sizes=_MLP_BLOCKS,
                    loop_orders=loop_orders,
                    pallas_stream_depth=3,
                )
                torch.testing.assert_close(out, expected_out, rtol=2e-2, atol=2e-2)
                self.assertIn("@pl.when(_gn_1 >= 32)", code)

    def test_weight_ring_loops_in_one_root(self) -> None:
        """Two streamed loops in one root share a ring; refills pick the loop."""
        torch.manual_seed(0)
        x = torch.randn(32, 256, device=DEVICE, dtype=torch.bfloat16)
        w_gate, w_up = (
            torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16) * 0.05
            for _ in range(2)
        )
        w_down = torch.randn(512, 256, device=DEVICE, dtype=torch.bfloat16) * 0.05
        act = (
            torch.nn.functional.silu(x.float() @ w_gate.float())
            * (x.float() @ w_up.float())
        ).to(torch.bfloat16)
        expected = act.float() @ w_down.float()
        for depth in (2, 3, 7):
            with self.subTest(depth=depth):
                code, out = code_and_output(
                    split_loops_mlp,
                    (x, w_gate, w_up, w_down),
                    block_sizes=[16, 128, 128, 128, 16, 128, 128],
                    pallas_stream_depth=depth,
                )
                torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
                # x is read-only too: its (16, 128) tiles take a ring of their
                # own, whose far smaller slots let it hold its whole stream.
                self.assertEqual(_ring_depths(code), [32, depth])
                self.assertIn("x_tile = ring.at[_slot]", code)
                self.assertIn("w_gate_tile = ring_1.at[_slot_1]", code)
                # A gate/up refill picks w_up's loop for the second half.
                self.assertRegex(code, r"@pl.when\(_q(_\d+)? >= 2\)")

    def test_weight_ring_per_dtype(self) -> None:
        """Each streamed dtype gets its own ring; loads decode dynamically
        when a consumer's stride does not fix the target load.  The f32 ring's
        slots are small enough to hold its whole stream."""
        torch.manual_seed(0)
        x = torch.randn(32, 256, device=DEVICE)
        w_in = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16) * 0.05
        w_gate, w_up = (
            torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16) * 0.05
            for _ in range(2)
        )
        y = (x.to(torch.bfloat16).float() @ w_in.float()).to(torch.bfloat16)
        expected = _glu_reference(y, w_gate, w_up)
        for depth in (2, 3, 5):
            with self.subTest(depth=depth):
                code, out = code_and_output(
                    project_then_glu,
                    (x, w_in, w_gate, w_up),
                    block_sizes=[16, 128, 128, 16, 128, 128],
                    pallas_stream_depth=depth,
                )
                torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
                self.assertEqual(_ring_depths(code), [8, depth])
                self.assertIn("((8, 16, 128), 'jnp.float32', 'vmem')", code)
                self.assertIn(f"(({depth}, 128, 128), 'jnp.bfloat16', 'vmem')", code)
                self.assertRegex(code, r"@pl.when\(_q_\d+ % 2 == 1\)")

    def test_weight_ring_per_tile_shape(self) -> None:
        """Each tile shape streams through a ring of its own, so no slot is
        padded to a larger tile.  ``pallas_stream_depth`` is the depth of the
        ring that streams the most bytes; the VMEM left lets the down ring
        hold its whole stream, and each ring refills only its own stream.  The
        down ring, read only after the gate/up ring, starts its prologue after
        the gate/up ring's last refill."""
        args = _mlp_args(8)
        expected_out, expected_hidden = _mlp_reference(*args)
        # 16 (128, 128) gate/up tiles, then 4 (256, 128) down tiles: half the
        # bytes in twice the slot size.
        blocks = [16, 16, 128, 128, 16, 128, 256]
        for depth, depths in ((None, [16, 4]), (6, [6, 4]), (4, [4, 4])):
            with self.subTest(depth=depth):
                code, (out, hidden) = code_and_output(
                    qwen3_dense_mlp,
                    args,
                    block_sizes=blocks,
                    pallas_stream_depth=depth,
                )
                torch.testing.assert_close(hidden, expected_hidden)
                torch.testing.assert_close(out, expected_out, rtol=2e-2, atol=2e-2)
                self.assertEqual(_ring_depths(code), depths)
                self.assertIn(
                    f"(({depths[0]}, 128, 128), 'jnp.bfloat16', 'vmem'), "
                    f"(({depths[0]},), None, 'dma_semaphore'), "
                    f"(({depths[1]}, 256, 128), 'jnp.bfloat16', 'vmem'), "
                    f"(({depths[1]},), None, 'dma_semaphore')",
                    code,
                )
                self.assertIn("w_up_tile = ring.at[_slot_1]", code)
                self.assertIn("w_down_tile = ring_1.at[_slot_2]", code)
                prime = code.index("ring_1.at[0], ring_sem_1.at[0]).start()")
                if depth is None:
                    # The gate/up ring holds its whole stream and never
                    # refills: the down ring starts at kernel start too.
                    self.assertNotIn("_prime", code)
                    self.assertLess(prime, code.index("@pl.loop"))
                else:
                    # Reading gate/up index 16 - D, the first without a refill,
                    # starts the down ring.
                    self.assertIn(f"@pl.when(_g == {16 - depth})", code)
                    self.assertGreater(prime, code.index("def _prime():"))
                    self.assertIn("@pl.when(_gn_1 < 16)", code)
                    self.assertNotIn("_gn_2", code)

    def test_weight_ring_arena(self) -> None:
        """``pallas_stream_arena`` streams every tile shape through one ring of
        native VMEM tiles, in program order: a slot holds the largest tile's
        16 (16, 128) tiles, and each tile reads a view of the slot's first
        tiles.  The dependency-free code embeds the views' helper."""
        args = _mlp_args(8)
        expected_out, expected_hidden = _mlp_reference(*args)
        blocks = [16, 16, 128, 128, 16, 128, 256]
        # 16 (128, 128) gate/up tiles, then 4 (256, 128) down tiles.
        for depth, ring_depth in ((None, 20), (6, 6)):
            with self.subTest(depth=depth):
                code, (out, hidden) = code_and_output(
                    qwen3_dense_mlp,
                    args,
                    block_sizes=blocks,
                    pallas_stream_depth=depth,
                    pallas_stream_arena=True,
                )
                torch.testing.assert_close(hidden, expected_hidden)
                torch.testing.assert_close(out, expected_out, rtol=2e-2, atol=2e-2)
                self.assertEqual(_ring_depths(code), [ring_depth])
                self.assertIn(
                    f"(({16 * ring_depth}, 16, 128), 'jnp.bfloat16', 'vmem')", code
                )
                self.assertIn(
                    "w_gate_tile = _helion_ring_tile(ring, _slot * 16, (128, 128))",
                    code,
                )
                self.assertRegex(
                    code,
                    r"w_down_tile = _helion_ring_tile\(ring, _slot_\d+ \* 16, "
                    r"\(256, 128\)\)",
                )
        bound = qwen3_dense_mlp.bind(args)
        free = bound.to_code(
            helion.Config(block_sizes=blocks, pallas_stream_arena=True),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
        self.assertIn("def ring_tile(", free)
        self.assertNotIn("import helion", free)
        ast.parse(free)

    def test_weight_ring_linear_layout(self) -> None:
        """Input projections stored [out, in] (nn.Linear) tile like the output
        projections, so every weight tile shares one ring; the transposed tiles
        feed their dots directly, with no transpose of the weight tile."""
        x, residual, norm_w, w_gate, w_up, w_down = _mlp_args(16)
        expected_out, expected_hidden = _mlp_reference(
            x, residual, norm_w, w_gate, w_up, w_down
        )
        # gate/up tiles w[tn, tk] and down tiles w_down[tk, tn] are all (128, 256).
        blocks = [16, 16, 128, 256, 16, 256, 128]
        args = (x, residual, norm_w, w_gate.T.contiguous(), w_up.T.contiguous(), w_down)
        code, (out, hidden) = code_and_output(
            qwen3_dense_mlp_linear_layout, args, block_sizes=blocks
        )
        torch.testing.assert_close(hidden, expected_hidden)
        torch.testing.assert_close(out, expected_out, rtol=2e-2, atol=2e-2)
        (depth,) = _ring_depths(code)
        self.assertIn(f"(({depth}, 128, 256), 'jnp.bfloat16', 'vmem')", code)
        self.assertNotIn("jnp.transpose", code)
        self.assertEqual(code.count("dimension_numbers=(((1,), (1,)), ((), ()))"), 2)
        # The [in, out] layout's (256, 128) gate/up tiles take a second ring.
        code = code_and_output(qwen3_dense_mlp, _mlp_args(16), block_sizes=blocks)[0]
        self.assertEqual(len(_ring_depths(code)), 2)

    def test_ring_depths_follow_traffic(self) -> None:
        """Past two iterations' loads per ring, the VMEM for the rings goes
        slot by slot to the ring whose roots it speeds up the most.  Each
        ring sits full while the other streams, so a ring deeper than one
        layer's tiles gains nothing: the folded stack's rings stop where the
        single layer's do."""
        from helion._compiler.pallas.megakernel import _STREAM_RESERVE_BYTES
        from helion._compiler.pallas.megakernel import resolve_ring_depths

        # Per layer: 16 (128, 128) gate/up tiles, two per iteration, then 4
        # (256, 128) down tiles, one per iteration.
        for kernel, args, blocks in (
            (qwen3_dense_mlp, _mlp_args(8), [16, 16, 128, 128, 16, 128, 256]),
            (
                mlp_stack,
                _mlp_stack_args(8, 3),
                [16, 16, 16, 128, 128, 16, 128, 256, 16],
            ),
        ):
            with self.subTest(kernel=kernel.name):
                spec = kernel.bind(args).config_spec
                model = spec.pallas_stream_model
                assert model is not None
                sizes = {
                    block.block_id: size
                    for block, size in zip(spec.block_sizes, blocks, strict=True)
                }
                # 4 gate/up and 2 down slots: 256 KiB.
                floor = (
                    model.resident(False)
                    + model.overhang_bytes(sizes)
                    + model.carried_bytes(sizes)
                    + _STREAM_RESERVE_BYTES
                    + 256 * 1024
                )
                for extra_kib, requested, depths in (
                    (0, None, [4, 2]),
                    # The gate/up ring streams 4x the down ring's bytes, in
                    # slots half the size: it gets the first slots.
                    (64, None, [6, 2]),
                    (128, None, [6, 3]),
                    (256, None, [10, 3]),
                    (512, None, [16, 4]),
                    (4096, None, [16, 4]),
                    # pallas_stream_depth pins the gate/up ring (down to one
                    # iteration's loads); the down ring takes what is left.
                    (0, 2, [2, 3]),
                    (128, 8, [8, 2]),
                    (256, 8, [8, 4]),
                ):
                    resolved = resolve_ring_depths(
                        model, sizes, requested, floor + extra_kib * 1024, False
                    )
                    self.assertEqual([depth for _, depth in resolved], depths)
                with self.assertRaisesRegex(
                    exc.InvalidConfig, r"depths \[4, 2\] need 262144"
                ):
                    resolve_ring_depths(model, sizes, None, floor - 1, False)

    def test_weight_ring_invalid_configs(self) -> None:
        from helion._compiler.pallas.megakernel import _STREAM_RESERVE_BYTES
        from helion._compiler.pallas.megakernel import resolve_ring_depths

        args = _mlp_args(8)
        bound = qwen3_dense_mlp.bind(args)
        spec = bound.config_spec
        backend = bound.env.backend
        # The ring must hold one gate/up iteration's two tiles.
        with self.assertRaisesRegex(exc.InvalidConfig, "smaller than the 2"):
            code_and_output(
                qwen3_dense_mlp, args, block_sizes=_MLP_BLOCKS, pallas_stream_depth=1
            )
        with self.assertRaisesRegex(exc.InvalidConfig, "positive int"):
            code_and_output(
                qwen3_dense_mlp, args, block_sizes=_MLP_BLOCKS, pallas_stream_depth=0
            )
        self.assertFalse(
            backend.autotune_config_is_viable(
                spec, helion.Config(block_sizes=_MLP_BLOCKS, pallas_stream_depth=1)
            )
        )
        self.assertTrue(
            backend.autotune_config_is_viable(
                spec, helion.Config(block_sizes=_MLP_BLOCKS, pallas_stream_depth=2)
            )
        )
        self.assertEqual(
            spec._flat_fields()["pallas_stream_depth"].choices,
            (None, *range(4, 17)),
        )
        # The ring has to fit next to the resident tensors, as far as their
        # tiles reach, the loop-carried scratch and the reserve.
        model = spec.pallas_stream_model
        assert model is not None
        sizes = dict(enumerate(_MLP_BLOCKS))
        # The launcher's two copies of the f32 hidden state, [8, 256] read in
        # 16-row tiles, each take 8 rows past one sublane tile.
        self.assertEqual(model.overhang_bytes(sizes), 2 * 8 * 256 * 4)
        # In 8-row tiles nothing overhangs.
        self.assertEqual(model.overhang_bytes({**sizes, 0: 8, 1: 8, 4: 8}), 0)
        capacity = (
            model.resident_bytes
            + model.overhang_bytes(sizes)
            + model.carried_bytes(sizes)
            + _STREAM_RESERVE_BYTES
            + 4 * 128 * 128 * 2
        )
        ((_, depth),) = resolve_ring_depths(model, sizes, None, capacity, False)
        self.assertEqual(depth, 4)
        with self.assertRaisesRegex(exc.InvalidConfig, r"depths \[5\] need 163840"):
            resolve_ring_depths(model, sizes, 5, capacity, False)
        # By default a ring holds at least two iterations' loads.
        with self.assertRaisesRegex(exc.InvalidConfig, r"depths \[4\] need 131072"):
            resolve_ring_depths(model, sizes, None, capacity - 1, False)
        # A streamed loop needs a K tile that divides it: no padded tiles.
        args = _mlp_args(8, h=384)
        with self.assertRaisesRegex(exc.InvalidConfig, "extent 384"):
            code_and_output(
                qwen3_dense_mlp, args, block_sizes=[16, 16, 128, 256, 16, 128, 128]
            )

    def test_weight_ring_tuned_fields(self) -> None:
        """Depths past the longest stream clamp to the same ring, so they are
        not tuned; neither are load buffer counts when the rings serve every
        inner-loop load."""
        x = torch.randn(16, 16, device=DEVICE)
        w = torch.randn(2, 16, 128, device=DEVICE)
        fields = grid_root_stream.bind((x, w)).config_spec._flat_fields()
        # Two programs of one 16-row tile: a stream of two.
        self.assertEqual(fields["pallas_stream_depth"].choices, (None, 2))
        self.assertNotIn("pallas_load_buffer_count", fields)
        fields = qwen3_dense_mlp.bind(_mlp_args(8)).config_spec._flat_fields()
        self.assertEqual(fields["pallas_stream_depth"].choices, (None, *range(4, 17)))
        self.assertNotIn("pallas_load_buffer_count", fields)
        # w's inner-loop load is not streamed: it keeps its buffer count.
        x = torch.randn(16, 256, device=DEVICE)
        w = torch.randn(256, 256, device=DEVICE)
        fields = weight_read_by_root.bind((x, w)).config_spec._flat_fields()
        self.assertIn("pallas_load_buffer_count", fields)

    @_per_weight_rings()
    def test_stream_tile_defaults(self) -> None:
        """A megakernel that streams weights through per-weight rings
        defaults to large tiles that divide their dims, and searches the
        non-power-of-two sizes that tile a whole axis.  One that streams
        through an arena defaults to tiles of about 2.5 MiB."""
        spec = qwen3_dense_mlp.bind(_mlp_args(8)).config_spec
        self.assertIn("pallas_megakernel_stream_tiles", spec.autotuner_heuristics)
        self.assertEqual(
            spec.default_config().config["block_sizes"],
            [16, 16, 512, 256, 16, 256, 512],
        )

        def qwen3_mlp_args(layers: int) -> tuple[torch.Tensor, ...]:
            # 2176 = 17 * 128: no power of two above 128 divides it.
            h, inter, dtype = 5120, 2176, torch.bfloat16
            return (
                torch.empty(1, h, device=DEVICE, dtype=dtype),
                torch.empty(layers, h, device=DEVICE, dtype=dtype),
                torch.empty(layers, h, inter, device=DEVICE, dtype=dtype),
                torch.empty(layers, h, inter, device=DEVICE, dtype=dtype),
                torch.empty(layers, inter, h, device=DEVICE, dtype=dtype),
            )

        with mock.patch.object(launcher, "_CACHED_VMEM_LIMIT_BYTES", 11 << 20):
            spec = mlp_stack.bind(qwen3_mlp_args(1)).config_spec
        self.assertEqual(spec.block_sizes[3].extra_search_values, (128, 2176))
        self.assertEqual(
            spec.block_sizes[4].extra_search_values,
            (128, 256, 512, 640, 1024, 1280, 2560, 5120),
        )
        # 11 MiB of VMEM leaves ~7 MiB for the rings, too little for four
        # iterations of (256, 2176) gate/up tiles, and at the default depths
        # for four of (128, 2176): (1280, 128) gate/up and (128, 2560) down.
        self.assertEqual(
            spec.default_config().config["block_sizes"],
            [16, 16, 16, 128, 1280, 16, 2560, 128, 16],
        )
        # 64 MiB: the same tiles.  Larger ones would fit, but the same ring
        # bytes in larger slots leave fewer bytes in flight.
        with mock.patch.object(launcher, "_CACHED_VMEM_LIMIT_BYTES", 64 << 20):
            spec = mlp_stack.bind(qwen3_mlp_args(2)).config_spec
        self.assertEqual(
            spec.default_config().config["block_sizes"],
            [16, 16, 16, 2176, 128, 16, 5120, 128, 16],
        )
        # An arena's tiles come closest to 2.5 MiB without going over:
        # (512, 2176) gate/up and (2176, 512) down tiles, 2.1 MiB each.
        with (
            mock.patch.object(launcher, "_CACHED_VMEM_LIMIT_BYTES", 64 << 20),
            mock.patch.object(megakernel, "_ARENA_BY_DEFAULT", True),
        ):
            # Three layers: a new bound kernel.
            spec = mlp_stack.bind(qwen3_mlp_args(3)).config_spec
            config = spec.default_config()
        self.assertEqual(
            config.config["block_sizes"], [16, 16, 16, 2176, 512, 16, 512, 2176, 16]
        )
        self.assertIs(config.config["pallas_stream_arena"], True)

    def test_weight_ring_ineligible(self) -> None:
        """A weight also read by a load the ring cannot serve (a small root-level
        read) is not streamed anywhere: it stays a whole VMEM block."""
        torch.manual_seed(0)
        x = torch.randn(16, 256, device=DEVICE)
        w = torch.randn(256, 256, device=DEVICE)
        bound = weight_read_by_root.bind((x, w))
        self.assertEqual(_streamed_shapes(bound), [torch.Size([16, 256])])
        code, out = code_and_output(
            weight_read_by_root, (x, w), block_sizes=[16, 128, 128, 128]
        )
        torch.testing.assert_close(out, (x @ w).sum(0) + w.sum(0), rtol=1e-4, atol=1e-2)
        self.assertNotIn("make_async_copy(w.at[", code)

    def test_weight_ring_grid_root(self) -> None:
        """An hl.grid root streams its weights; the autotuner's viability check
        resolves the grid axis to its fixed block size."""
        torch.manual_seed(0)
        x = torch.randn(16, 256, device=DEVICE)
        w = torch.randn(4, 256, 128, device=DEVICE)
        bound = grid_root_stream.bind((x, w))
        spec = bound.config_spec
        self.assertEqual(_streamed_shapes(bound), [torch.Size([4, 256, 128])])
        for depth in (None, 1, 3):
            config = helion.Config(
                block_sizes=[16, 128, 128], pallas_stream_depth=depth
            )
            self.assertTrue(bound.env.backend.autotune_config_is_viable(spec, config))
        code, out = code_and_output(
            grid_root_stream, (x, w), block_sizes=[16, 128, 128], pallas_stream_depth=3
        )
        torch.testing.assert_close(
            out, torch.einsum("mk,ekn->emn", x * 2.0, w), rtol=1e-4, atol=1e-3
        )
        self.assertEqual(_ring_depths(code), [3])
        self.assertIn("ring.at[_slot]", code)

    def test_weight_ring_decode_activation(self) -> None:
        """A decode activation x[tm, tk] has fewer rows than a sublane tile:
        it takes the regular path while the weights stream."""
        torch.manual_seed(0)
        wg = torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16) * 0.05
        wd = torch.randn(512, 256, device=DEVICE, dtype=torch.bfloat16) * 0.05
        for m in (1, 8):
            with self.subTest(m=m):
                x = torch.randn(m, 256, device=DEVICE, dtype=torch.bfloat16)
                bound = glu_then_down.bind((x, wg, wd))
                self.assertEqual(
                    _streamed_shapes(bound),
                    [torch.Size([256, 512]), torch.Size([512, 256])],
                )
                code, out = code_and_output(glu_then_down, (x, wg, wd))
                expected = (x.float() @ wg.float()).to(torch.bfloat16).float()
                torch.testing.assert_close(
                    out, expected @ wd.float(), rtol=1e-3, atol=1e-3
                )
                self.assertNotIn("x_tile", code)
                self.assertRegex(code, _ring_tile("w_gate"))

    def test_weight_ring_unsupported_loads_fall_back(self) -> None:
        """A load the ring cannot serve (a rank-1 norm weight) takes the
        regular path; the other tensors still stream, including one whose
        leading dim is indexed by a constant."""
        torch.manual_seed(0)
        x = torch.randn(16, 256, device=DEVICE)
        norm_w = torch.randn(256, device=DEVICE)
        w1 = torch.randn(256, 256, device=DEVICE)
        w2 = torch.randn(2, 256, 128, device=DEVICE)
        args = (x, norm_w, w1, w2)
        bound = norm_then_indexed_matmul.bind(args)
        self.assertEqual(
            _streamed_shapes(bound),
            [torch.Size([16, 256]), torch.Size([256, 256]), torch.Size([2, 256, 128])],
        )
        code, out = code_and_output(
            norm_then_indexed_matmul, args, block_sizes=[16, 128, 128, 16, 128]
        )
        expected = ((x * norm_w) @ w1) @ w2[0]
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-2)
        self.assertIn("w1_tile = ring_1.at[", code)
        self.assertIn("w2.at[0, pl.ds(", code)
        self.assertIn("norm_w_buf[:]", code)

    def test_weight_ring_root_level_weight(self) -> None:
        """A large weight read by a root's own tile loop streams through a
        ring of its own: each tile waits on its slot and refills it after the
        tile's compute.  The two-axis root walks its tiles in either order."""
        torch.manual_seed(0)
        x = torch.randn(32, 1024, device=DEVICE, dtype=torch.bfloat16)
        w1 = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16) * 0.05
        w2 = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16) * 0.05
        args = (x, w1, w2)
        bound = root_matmul_then_matmul.bind(args)
        model = bound.config_spec.pallas_stream_model
        assert model is not None
        self.assertEqual(
            [
                (site.block_id, load.fake.shape)
                for site in model.sites
                for load in site.loads
            ],
            [(None, torch.Size([1024, 1024])), (4, torch.Size([1024, 1024]))],
        )
        y = (x.float() @ w1.float()).to(torch.bfloat16)
        expected = y.float() @ w2.float()
        for loop_orders in ([[0, 1], [0, 1]], [[1, 0], [1, 0]]):
            with self.subTest(loop_orders=loop_orders):
                code, out = code_and_output(
                    root_matmul_then_matmul,
                    args,
                    block_sizes=[16, 256, 16, 128, 128],
                    loop_orders=loop_orders,
                    pallas_stream_depth=3,
                )
                torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
                # w2's ring holds its whole stream: (128, 128) tiles, each read
                # once per M tile.
                self.assertEqual(_ring_depths(code), [3, 128])
                self.assertIn("w1_tile = ring.at[_slot]", code)
                self.assertIn("load_1 = w1_tile[:, :]", code)
                # Root 0 has 2 x 4 tiles of w1's (1024, 256) column blocks; the
                # one that reads index 5 starts w2's ring after its refill.
                self.assertIn("@pl.when(_gn < 8)", code)
                self.assertIn("@pl.when(_g == 5)", code)
                self.assertGreater(
                    code.index("ring_1.at[0], ring_sem_1.at[0]).start()"),
                    code.index("def _prime():"),
                )

    def test_weight_ring_vmem_guard(self) -> None:
        """A tensor that must stay whole in VMEM but does not fit is rejected."""
        x = torch.randn(2048, 128, device=DEVICE, dtype=torch.bfloat16)
        w1 = torch.randn(128, 4096, device=DEVICE, dtype=torch.bfloat16)
        w2 = torch.randn(4096, 128, device=DEVICE, dtype=torch.bfloat16)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"the largest, y \(16777216 bytes\), is written"
        ):
            root_matmul_then_matmul.bind((x, w1, w2))

    def test_weight_ring_list_argument_roles(self) -> None:
        """Tensors of one list argument get their own roles: the one written
        by root 0 (by a store or an atomic) is not streamed."""
        torch.manual_seed(0)
        x = torch.randn(16, 256, device=DEVICE)
        w = torch.randn(256, 256, device=DEVICE)
        expected = (x * 2.0) @ w
        for kernel in (list_store_then_stream, list_atomic_then_stream):
            with self.subTest(kernel=kernel.name):
                y = torch.zeros(16, 256, device=DEVICE)
                bound = kernel.bind((x, [w, y]))
                roles = bound.host_function.device_ir.storage_roles
                _, w_fake, y_fake = bound.env.input_sources
                self.assertIs(
                    roles[id(w_fake.untyped_storage())], StorageRole.READ_ONLY
                )
                self.assertIs(roles[id(y_fake.untyped_storage())], StorageRole.OUTPUT)
                model = bound.config_spec.pallas_stream_model
                assert model is not None
                self.assertEqual(
                    [load.fake.shape for site in model.sites for load in site.loads],
                    [torch.Size([256, 256])],
                )
                code, out = code_and_output(
                    kernel, (x, [w, y]), block_sizes=[16, 128, 16, 128, 128]
                )
                torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)
                self.assertIn("make_async_copy(bufs_item_0.at[", code)
                self.assertNotIn("make_async_copy(bufs_item_1.at[", code)

    def test_folded_mlp_stack(self) -> None:
        """A host loop around the per-layer roots folds into one ``pl.loop``
        whose weight tiles stream through one ring across iterations."""
        for layers, depth in ((1, 3), (3, 2), (3, 5), (3, None)):
            args = _mlp_stack_args(8, layers)
            with self.subTest(layers=layers, depth=depth):
                code, out = code_and_output(
                    mlp_stack,
                    args,
                    block_sizes=_STACK_BLOCKS,
                    pallas_stream_depth=depth,
                )
                torch.testing.assert_close(
                    out, _mlp_stack_reference(*args), rtol=2e-2, atol=2e-2
                )
                self.assertOneSequentialProgram(code)
                self.assertIn(f"@pl.loop(0, {layers})\n    def _i_loop(i):", code)
                # The stream holds 24 tiles per layer.  Every root reads the
                # one ring, so it never sits full: by default it holds the
                # whole stream.
                self.assertEqual(_ring_depths(code), [depth or 24 * layers])
        # Stacked weights stay in HBM; the [L, H] norm weight is loaded whole.
        self.assertIn("_hbm_arg_indices=[1, 2, 3]", code)
        self.assertIn("norm_w[:, :]", code)
        args = _mlp_stack_args(8, 3)
        code, _ = code_and_output(
            mlp_stack, args, block_sizes=_STACK_BLOCKS, pallas_stream_depth=5
        )
        # 2-D ring slots serve the 3-D stacked weights; the prologue runs once,
        # before the folded loop.
        self.assertIn(
            "((5, 128, 128), 'jnp.bfloat16', 'vmem'), ((5,), None, 'dma_semaphore')",
            code,
        )
        prologue = re.findall(r"\n    pltpu\.make_async_copy\(.*\.start\(\)", code)
        self.assertEqual(len(prologue), 5)
        self.assertLess(code.index(prologue[-1]), code.index("def _i_loop(i):"))
        self.assertIn(
            "pltpu.make_async_copy(w_gate.at[0, pl.ds(0, 128), pl.ds(0, 128)], "
            "ring.at[0], ring_sem.at[0]).start()",
            code,
        )
        # Consumers add the iteration's offset and select the layer with i.
        self.assertIn("_g = i * 24 + pid_shared_2 * 4 + _j * 2", code)
        self.assertIn("_g_1 = 16 + i * 24 + pid_shared_3 * 4 + _j_1", code)
        self.assertIn(
            "pltpu.make_async_copy(w_gate.at[i, pl.ds(pl.multiple_of(_j * 128, 128), "
            "128), pl.ds(pl.multiple_of(offset_3, 128), 128)], w_gate_tile, "
            "ring_sem.at[_slot]).wait()",
            code,
        )
        self.assertIn("load_3 = w_gate_tile[:, :]", code)
        # Refills decode the iteration, which selects the layer, so the last
        # tiles of a layer prefetch the next layer's first tiles.
        self.assertIn("_layer = _gn // 24", code)
        self.assertIn("w_up.at[_layer, pl.ds(", code)
        self.assertIn("@pl.when(_gn_4 < 72)", code)
        self.assertIn("w_gate.at[_layer_2, pl.ds(", code)

    def test_roots_share_carried_scratch(self) -> None:
        """Roots run one after another, so the down projection's loop-carried
        accumulator takes the buffer of the gate/up root's ``g`` (same shape
        and dtype) instead of a scratch of its own."""
        args = _mlp_stack_args(8, 3)
        code, out = code_and_output(mlp_stack, args, block_sizes=_STACK_BLOCKS)
        torch.testing.assert_close(
            out, _mlp_stack_reference(*args), rtol=2e-2, atol=2e-2
        )
        refs = _vmem_refs(mlp_stack, args, block_sizes=_STACK_BLOCKS)
        self.assertEqual(refs.count(((16, 128), "float32")), 2)
        (alias,) = re.findall(r"\n    (scratch_\w+ = scratch_\w+)\n", code)
        # The kernel binds the name before any root runs.
        self.assertLess(code.index(alias), code.index("def _i_loop(i):"))

    def test_rings_count_carried_scratch(self) -> None:
        """The rings' fit counts the loop-carried scratch at the config's
        block sizes, as the roots share it: of each shape and dtype, as many
        buffers as the root that carries the most."""
        args = _mlp_stack_args(8, 3)
        spec = mlp_stack.bind(args).config_spec
        model = spec.pallas_stream_model
        assert model is not None
        for blocks, carried in (
            # g and u of the gate/up root; the down root's accumulator takes
            # one of their buffers.
            (_STACK_BLOCKS, [((16, 128), "float32")] * 2),
            # With wider gate/up tiles, it differs in shape and takes a buffer
            # of its own.
            (
                [16, 16, 16, 256, 128, 16, 128, 128, 16],
                [((16, 256), "float32")] * 2 + [((16, 128), "float32")],
            ),
        ):
            with self.subTest(blocks=blocks):
                tuned = {
                    block.block_id: size
                    for block, size in zip(spec.block_sizes, blocks, strict=True)
                }
                refs = _vmem_refs(mlp_stack, args, block_sizes=blocks)
                f32 = [ref for ref in refs if ref[1] == "float32"]
                # The f32 hidden state, kept whole in VMEM, is not carried.
                f32.remove(((16, 256), "float32"))
                self.assertEqual(sorted(f32), sorted(carried))
                self.assertEqual(
                    model.carried_bytes(tuned),
                    sum(rows * cols * 4 for (rows, cols), _ in carried),
                )

    def test_folded_stack_ring_per_tile_shape(self) -> None:
        """Across a folded loop each tile shape keeps its own ring, whose
        refills decode the layer from that ring's per-layer stride.  Over
        more than one layer their reads interleave, so every ring starts at
        kernel start.  Each ring sits full while the other streams, so by
        default it holds one layer's tiles, which then land before its root
        starts."""
        args = _mlp_stack_args(8, 3)
        # Per layer: 16 (128, 128) gate/up tiles and 4 (256, 128) down tiles.
        blocks = [16, 16, 16, 128, 128, 16, 128, 256, 16]
        for depth, depths in ((None, [16, 4]), (4, [4, 4]), (6, [6, 4])):
            with self.subTest(depth=depth):
                code, out = code_and_output(
                    mlp_stack, args, block_sizes=blocks, pallas_stream_depth=depth
                )
                torch.testing.assert_close(
                    out, _mlp_stack_reference(*args), rtol=2e-2, atol=2e-2
                )
                self.assertOneSequentialProgram(code)
                self.assertEqual(_ring_depths(code), depths)
                self.assertNotIn("_prime", code)
        self.assertIn("_g = i * 16 + pid_shared_2 * 4 + _j * 2", code)
        self.assertIn("_layer = _gn // 16", code)
        self.assertIn("_g_1 = i * 4 + pid_shared_3 * 2 + _j_1", code)
        self.assertIn("@pl.when(_gn_4 < 12)", code)
        self.assertIn("_layer_2 = _gn_4 // 4", code)
        # A single layer reads the rings one after the other again: the down
        # ring starts after the gate/up ring's last refill.
        args = _mlp_stack_args(8, 1)
        code, out = code_and_output(
            mlp_stack, args, block_sizes=blocks, pallas_stream_depth=6
        )
        torch.testing.assert_close(
            out, _mlp_stack_reference(*args), rtol=2e-2, atol=2e-2
        )
        self.assertEqual(_ring_depths(code), [6, 4])
        self.assertIn("@pl.when(_g == 10)", code)
        self.assertGreater(
            code.index("ring_1.at[0], ring_sem_1.at[0]).start()"),
            code.index("def _prime():"),
        )

    def test_folded_gemv_chain(self) -> None:
        """The ring flows from roots before the loop through its iterations into
        the roots after it; the loop may start at any constant."""
        for layers, start, depth in ((3, 0, 3), (3, 0, 5), (4, 1, 5), (2, 1, 4)):
            args = _gemv_chain_args(16, layers)
            with self.subTest(layers=layers, start=start, depth=depth):
                code, out = code_and_output(
                    gemv_chain,
                    (*args, start),
                    block_sizes=[16, 128, 128] * 4,
                    pallas_stream_depth=depth,
                )
                torch.testing.assert_close(
                    out, _gemv_chain_reference(*args, start), rtol=2e-2, atol=2e-2
                )
                # x's 4 (16, 128) tiles stream through a ring of their own.
                self.assertEqual(_ring_depths(code), [4, depth])
                self.assertIn(f"@pl.loop({start}, {layers})", code)
        args = _gemv_chain_args(16, 4)
        code, _ = code_and_output(
            gemv_chain,
            (*args, 1),
            block_sizes=[16, 128, 128] * 4,
            pallas_stream_depth=5,
        )
        # Before the loop: w_in's 4 tiles refill into its first iteration.
        self.assertIn("_e = _gn - 4", code)
        self.assertIn("_layer = _e // 8", code)
        self.assertIn("a.at[1 + _layer, pl.ds(", code)
        # In the loop: the iteration offset counts from the loop's start, and
        # the last iteration refills the root after the loop.
        self.assertIn("_g_2 = 4 + (i - 1) * 8 + pid_shared_1 * 2 + _j_1", code)
        self.assertIn("a.at[i, pl.ds(", code)
        self.assertIn("@pl.when(_gn_2 >= 28)", code)
        self.assertIn("_g_4 = 28 + pid_shared_3 * 2 + _j_3", code)

    def test_tp_all_reduce_gemv_chain(self) -> None:
        """A per-layer all-reduce over remote copies inside a folded loop.  The
        receive buffer is scratch that every root addresses in VMEM, a rank
        read from SMEM selects one sender slot like an integer, and each push
        moves only the logical rows of the row-padded buffer."""
        codes = {}
        for m in (1, 16):
            torch.manual_seed(0)
            dtype = torch.bfloat16
            args = (
                torch.randn(m, 256, device=DEVICE, dtype=dtype),
                torch.randn(3, 256, 256, device=DEVICE, dtype=dtype) * 0.06,
                torch.randn(3, 256, 256, device=DEVICE, dtype=dtype) * 0.06,
                *_tp_ranks(8),
            )
            bound = tp_gemv_chain.bind(args)
            codes[m] = code = bound.to_code(bound.config_spec.default_config())
            with self.subTest(m=m):
                self.assertOneSequentialProgram(code)
                self.assertIn("@pl.loop(0, 3)\n    def _i_loop(i):", code)
                self.assertIn("_hbm_arg_indices=[1, 2]", code)
                self.assertIn("_smem_arg_indices=[4]", code)
                # The weight stream starts before the barrier, under its latency.
                self.assertLess(
                    code.index("ring_sem.at[0]).start()"),
                    code.index("pltpu.get_barrier_semaphore()"),
                )
                # [parity, sender, rows padded to 16, h], summed in place.
                self.assertIn("((2, 8, 16, 256), 'jnp.float32', 'vmem')", code)
                self.assertNotIn("recv_load", code)
                self.assertRegex(code, r"load_\d+ = recv\[mod_\d+, :, pl\.ds\(")
        # One decode row: the store masks the padded rows of its 2-D block and
        # the pushes trim them, to the row since the f32 scratch slices to any.
        self.assertRegex(
            codes[1],
            r"recv\[mod, load_\d+, pl\.ds\([^\n]*\] = jnp\.where\(offset_\d+ \+ "
            r"lax\.broadcasted_iota\(jnp\.int32, \(_BLOCK_SIZE_\d+, 1\), 0\) < 1, ",
        )
        self.assertIn(
            "make_async_remote_copy(recv.at[mod_1, me_copy_0, pl.ds(0, 1)], "
            "recv.at[mod_1, me_copy_0, pl.ds(0, 1)], remote_send_sem, "
            "remote_recv_sem, ",
            codes[1],
        )
        # Sixteen rows fill the tile: the whole sender slot moves.
        self.assertIn(
            "make_async_remote_copy(recv.at[mod_1, me_copy_0], "
            "recv.at[mod_1, me_copy_0], remote_send_sem, remote_recv_sem, ",
            codes[16],
        )

    def test_tp_exchange_trim_by_dtype(self) -> None:
        """A push of the logical rows of a row-padded buffer trims them to the
        8-row tiles Mosaic slices, or to the row of a one-row f32 buffer;
        bf16 packs two rows to a sublane, so its one row still moves 8."""
        for dtype, m, rows in (
            (torch.float32, 1, 1),
            (torch.float32, 2, 8),
            (torch.bfloat16, 1, 8),
        ):
            with self.subTest(dtype=dtype, m=m):
                args = (torch.randn(m, 256, device=DEVICE, dtype=dtype), *_tp_ranks(8))
                bound = tp_all_gather_sum.bind(args)
                code = bound.to_code(bound.config_spec.default_config())
                self.assertIn(
                    f"make_async_remote_copy(gather.at[me_copy_0, pl.ds(0, {rows})], "
                    f"gather.at[me_copy_0, pl.ds(0, {rows})], ",
                    code,
                )

    def test_tp_all_reduce_algorithm_by_payload(self) -> None:
        """``hl.all_reduce`` expands to a one-shot exchange (push the f32
        partial to every peer, then sum) for a small payload and to reduce-
        scatter + all-gather (bf16 chunks of h / world) for a large one, or to
        the algorithm given.  Alone in its loop, a site alternates two
        buffer slots by the loop index."""
        torch.manual_seed(0)
        h = 1024
        w = torch.randn(3, h, h, device=DEVICE, dtype=torch.bfloat16) * 0.03
        for m, algorithm, one_shot in (
            (1, None, True),
            (16, None, False),
            (16, "one_shot", True),
            (1, "reduce_scatter", False),
        ):
            with self.subTest(m=m, algorithm=algorithm):
                x = torch.randn(m, h, device=DEVICE, dtype=torch.bfloat16)
                args = (x, w, *_tp_ranks(8), algorithm)
                bound = tp_all_reduce_residual.bind(args)
                code = bound.to_code(bound.config_spec.default_config())
                self.assertOneSequentialProgram(code)
                self.assertIn("mod = i % 2", code)
                if one_shot:
                    self.assertIn("((2, 8, 16, 1024), 'jnp.float32', 'vmem')", code)
                    self.assertIn("make_async_remote_copy(partial", code)
                    self.assertNotIn("_all_reduce0_scatter", code)
                else:
                    # [slot, sender, rows, h / world]: f32 scatter (rows padded
                    # to 8), bf16 gather (rows padded to 16).
                    self.assertIn(
                        f"((2, 8, {max(m, 8)}, 128), 'jnp.float32', 'vmem')", code
                    )
                    self.assertIn("((2, 8, 16, 128), 'jnp.bfloat16', 'vmem')", code)
                    self.assertNotIn("_all_reduce0_recv", code)

    def test_tp_all_reduce_sites_share_slots(self) -> None:
        """Two all-reduces per loop body each keep one buffer slot: a peer
        reaches a site's next execution only after the other site's exchange,
        which this rank joins after reading the buffer."""
        torch.manual_seed(0)
        args = (
            torch.randn(1, 1024, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(3, 2, 1024, 1024, device=DEVICE, dtype=torch.bfloat16),
            *_tp_ranks(8),
        )
        bound = tp_all_reduce_pairs.bind(args)
        code = bound.to_code(bound.config_spec.default_config())
        self.assertOneSequentialProgram(code)
        self.assertIn("_all_reduce1_recv", code)
        self.assertNotIn("i % 2", code)
        self.assertEqual(code.count("((1, 8, 16, 1024), 'jnp.float32', 'vmem')"), 2)

    def test_tp_all_reduce_fuses_pointwise_consumer(self) -> None:
        """A tile loop after a one-shot site that reads the destination only
        at its own tile runs in the sum's loop: each summed tile feeds the
        residual add without a root (and a reload) in between.  Reduce-
        scatter keeps the loop after its gather."""
        torch.manual_seed(0)
        h = 1024
        w = torch.randn(3, h, h, device=DEVICE, dtype=torch.bfloat16) * 0.03
        x = torch.randn(1, h, device=DEVICE, dtype=torch.bfloat16)
        for algorithm, fused in (("one_shot", True), ("reduce_scatter", False)):
            with self.subTest(algorithm=algorithm):
                bound = tp_all_reduce_residual.bind((x, w, *_tp_ranks(8), algorithm))
                code = bound.to_code(bound.config_spec.default_config())
                self.assertOneSequentialProgram(code)
                (residual,) = [
                    root
                    for root in code.split("def _root")
                    if "res[tm, tn] = res[tm, tn] + red[tm, tn]" in root
                ]
                self.assertEqual("_all_reduce0_recv[" in residual, fused)
                self.assertEqual("= red[pl.ds(" in residual, not fused)

    def test_host_if_on_shape_chooses_roots(self) -> None:
        """A host ``if`` on a static shape around device loops keeps only the
        taken branch's loops."""
        for m, expected in ((16, lambda x: x * 2), (8, lambda x: x + 1)):
            with self.subTest(m=m):
                x = torch.randn(m, 128, device=DEVICE)
                code, out = code_and_output(shape_chosen_roots, (x,))
                torch.testing.assert_close(out, expected(x))
                self.assertEqual(code.count("def _helion_shape_chosen_roots"), 1)

    def test_tp_all_reduce_mlp_stack(self) -> None:
        """Each unrolled parity of a folded TP MLP stack has its own exchange
        site and semaphores; its partial lands in its sender slot of the
        receive scratch, and the sum root reads all slots in place."""
        args = (*_mlp_stack_args(1, 4, inter=256), *_tp_ranks(8))
        bound = tp_mlp_stack.bind(args)
        code = bound.to_code(bound.config_spec.default_config())
        self.assertOneSequentialProgram(code)
        self.assertIn("@pl.loop(0, 2)\n    def _p_loop(p):", code)
        self.assertIn("_hbm_arg_indices=[1, 2, 3]", code)
        self.assertLess(
            code.index("ring_sem.at[0]).start()"),
            code.index("pltpu.get_barrier_semaphore()"),
        )
        self.assertIn("((2, 8, 16, 256), 'jnp.float32', 'vmem')", code)
        self.assertNotIn("recv_load", code)
        for par, me, sems in (
            (0, "me_copy_0", "remote_send_sem, remote_recv_sem"),
            (1, "me_1_copy_0", "remote_send_sem_1, remote_recv_sem_1"),
        ):
            self.assertRegex(
                code,
                rf"recv\[{par}, load_\d+, pl\.ds\([^\n]*\] = jnp\.where\("
                r"offset_\d+ \+ lax\.broadcasted_iota\(jnp\.int32, "
                r"\(_BLOCK_SIZE_\d+, 1\), 0\) < 1, ",
            )
            self.assertIn(
                f"make_async_remote_copy(recv.at[{par}, {me}, pl.ds(0, 1)], "
                f"recv.at[{par}, {me}, pl.ds(0, 1)], {sems}, ",
                code,
            )
            self.assertRegex(code, rf"load_\d+ = recv\[{par}, :, pl\.ds\(")
        self.assertEqual(code.count("remote_copy.wait()"), 7)
        self.assertEqual(code.count("remote_copy_1.wait()"), 7)

    def _tp_gemv_pair_code(self, **constants: int) -> str:
        """``tp_gemv_pair_stack``'s code, with megakernel ``constants``."""
        torch.manual_seed(0)
        dtype = torch.bfloat16
        args = (
            torch.randn(1, 1024, device=DEVICE, dtype=dtype),
            torch.randn(4, 1024, 512, device=DEVICE, dtype=dtype),
            torch.randn(4, 512, 1024, device=DEVICE, dtype=dtype),
            *_tp_ranks(8),
        )
        with (
            _per_weight_rings(),
            mock.patch.multiple(
                "helion._compiler.pallas.megakernel",
                _STREAM_TARGET_BYTES=1 << 20,
                **constants,
            ),
        ):
            bound = tp_gemv_pair_stack.bind(args)
            code = bound.to_code(bound.config_spec.default_config())
        # a's ring and b's ring: 4 slots, one layer's copies each.
        self.assertIn("((4, 256, 512), 'jnp.bfloat16', 'vmem')", code)
        self.assertIn("((4, 128, 1024), 'jnp.bfloat16', 'vmem')", code)
        return code

    def test_exchange_defers_ring_copies(self) -> None:
        """The ring copies read first after an exchange, whose slots free up
        before the exchange ahead of it, start behind its last remote copy,
        where they fill the round trip.  The first exchange of the folded
        loop's body starts those of the gemv after it; the last one, those of
        the next iteration's first gemv, and none in the last iteration."""
        code = self._tp_gemv_pair_code()
        # Iteration 0's first gemv still primes at kernel start.
        self.assertLess(
            code.index("ring.at[3], ring_sem.at[3]).start()"),
            code.index("pltpu.get_barrier_semaphore()"),
        )
        self.assertNotIn("ring_1.at[0]", code)
        for copy, catch_up in (
            ("remote_copy", "_catch_up"),
            ("remote_copy_1", "_catch_up_1"),
        ):
            start = code.index(f"{copy}.start()\n")
            match = re.compile(
                rf"{copy}\.start\(\)\n\n\s+@pl\.when\(_j_\d+ == \(7 - 0 \+ "
                rf"_BLOCK_SIZE_\d+ - 1\) // _BLOCK_SIZE_\d+ - 1\)\n"
                rf"\s+def {catch_up}\(\):\n"
            ).match(code, start)
            assert match is not None, code[start : start + 400]
            body = code[match.end() : code.index(f"{copy}.wait()", start)]
            self.assertEqual(body.count(".start()"), 4)
        first_catch_up = code[code.index("def _catch_up():") :]
        self.assertIn(
            "make_async_copy(b.at[i, pl.ds(384, 128), pl.ds(0, 1024)], "
            "ring_1.at[(3 + i * 4) % 4], ring_sem_1.at[(3 + i * 4) % 4]).start()",
            first_catch_up,
        )
        last_catch_up = code[code.index("def _catch_up_1():") :]
        layer = re.search(r"(_layer_\d+) = i \+ 1\n", last_catch_up)
        assert layer is not None
        self.assertIn(f"@pl.when({layer[1]} < 4)", last_catch_up)
        self.assertIn(
            f"make_async_copy(a.at[{layer[1]}, pl.ds(0, 256), pl.ds(0, 512)], "
            "ring.at[(4 + i * 4) % 4], ring_sem.at[(4 + i * 4) % 4]).start()",
            last_catch_up,
        )
        # The refills into the deferred copies do not start them.
        self.assertRegex(code, r"@pl\.when\(_gn_\d+ >= 4\)")
        self.assertRegex(code, r"@pl\.when\(\(_gn_\d+ >= 4\) \| \(_layer < 1\)\)")

    def test_exchange_hold_budget(self) -> None:
        """An exchange holds only the bytes its round trip covers: the copies
        whose slots free up first, which would otherwise stream while the
        roots before it run.  The rest start as their slots free up."""
        code = self._tp_gemv_pair_code(_EXCHANGE_HOLD_BYTES=512 << 10)
        for copy in ("remote_copy", "remote_copy_1"):
            start = code.index(f"{copy}.start()\n")
            body = code[start : code.index(f"{copy}.wait()", start)]
            self.assertEqual(body.count(".start()"), 3)
        first_catch_up = code[code.index("def _catch_up():") :]
        for index in range(2):
            self.assertIn(
                f"make_async_copy(b.at[i, pl.ds({index * 128}, 128), "
                f"pl.ds(0, 1024)], ring_1.at[({index} + i * 4) % 4], "
                f"ring_sem_1.at[({index} + i * 4) % 4]).start()",
                first_catch_up,
            )
        last_catch_up = code[code.index("def _catch_up_1():") :]
        layer = re.search(r"(_layer_\d+) = i \+ 1\n", last_catch_up)
        assert layer is not None
        self.assertIn(
            f"make_async_copy(a.at[{layer[1]}, pl.ds(256, 256), pl.ds(0, 512)], "
            "ring.at[(5 + i * 4) % 4], ring_sem.at[(5 + i * 4) % 4]).start()",
            last_catch_up,
        )
        # Iteration 0's later copies of b prime at kernel start, and the
        # refills skip only the held copies.
        self.assertLess(
            code.index(
                "make_async_copy(b.at[0, pl.ds(384, 128), pl.ds(0, 1024)], "
                "ring_1.at[3], ring_sem_1.at[3]).start()"
            ),
            code.index("pltpu.get_barrier_semaphore()"),
        )
        self.assertRegex(code, r"@pl\.when\(_gn_\d+ >= 2\)")
        self.assertRegex(code, r"@pl\.when\(\(_gn_\d+ >= 2\) \| \(_layer < 1\)\)")

    def test_exchange_holds_nothing_streaming_into_it(self) -> None:
        """An exchange holds no copies whose slots free up after the exchange
        ahead of it: the rings stream those into its round trip anyway.  In
        an MLP stack, every window is read right after the exchange before."""
        args = (*_mlp_stack_args(1, 4, h=512, inter=512), *_tp_ranks(8))
        with mock.patch(
            "helion._compiler.pallas.megakernel._STREAM_TARGET_BYTES", 1 << 20
        ):
            bound = tp_mlp_stack.bind(args)
            code = bound.to_code(bound.config_spec.default_config())
        self.assertIn("remote_copy_1.start()", code)
        self.assertNotIn("_catch_up", code)
        self.assertNotRegex(code, r"_gn_\d+ >= ")

    def test_exchange_arena_holds_nothing(self) -> None:
        """An arena holds no copies at an exchange: it streams in program
        order, so the copies in flight at an exchange are the ones the roots
        right after it read."""
        torch.manual_seed(0)
        dtype = torch.bfloat16
        args = (
            torch.randn(1, 1024, device=DEVICE, dtype=dtype),
            torch.randn(4, 1024, 512, device=DEVICE, dtype=dtype),
            torch.randn(4, 512, 1024, device=DEVICE, dtype=dtype),
            *_tp_ranks(8),
        )
        bound = tp_gemv_pair_stack.bind(args)
        config = {**bound.config_spec.default_config(), "pallas_stream_depth": 4}
        code = bound.to_code(helion.Config(**config))
        self.assertEqual(_ring_depths(code), [4])
        self.assertIn("_helion_ring_tile(ring, ", code)
        self.assertIn("def _refill", code)
        self.assertIn("remote_copy_1.start()", code)
        self.assertNotIn("_catch_up", code)
        self.assertNotRegex(code, r"_gn_\d+ >= ")

    def test_exchange_arena_other_rings_hold_nothing(self) -> None:
        """Beside an arena, other rings hold no copies at an exchange either:
        a held copy would queue behind the arena's copies in flight and land
        after the roots right after the exchange read it."""
        torch.manual_seed(0)
        dtype = torch.bfloat16
        args = (
            torch.randn(1, 1024, device=DEVICE, dtype=dtype),
            torch.randn(4, 1024, 64, device=DEVICE, dtype=dtype),
            torch.randn(4, 64, 1024, device=DEVICE, dtype=dtype),
            *_tp_ranks(8),
        )
        bound = tp_gemv_pair_stack.bind(args)
        code = bound.to_code(bound.config_spec.default_config())
        self.assertEqual(_ring_depths(code), [1, 2])
        self.assertIn("_helion_ring_tile(ring_1, ", code)
        self.assertIn("ring.at[pl.ds(_slot * 1024, 1024), :]", code)
        self.assertIn("remote_copy_1.start()", code)
        self.assertNotIn("_catch_up", code)

    def test_exchange_defers_ring_copies_unfolded(self) -> None:
        """Outside folded loops, the second exchange starts the first ring
        copies of the root after it, into the slots the root before the first
        exchange read.  The first exchange has none ahead of it to hold."""
        dtype = torch.bfloat16
        args = (
            torch.randn(1, 1024, device=DEVICE, dtype=dtype),
            torch.randn(1024, 1024, device=DEVICE, dtype=dtype),
            torch.randn(1024, 512, device=DEVICE, dtype=dtype),
            torch.randn(512, 1024, device=DEVICE, dtype=dtype),
            *_tp_ranks(8),
        )
        with (
            _per_weight_rings(),
            mock.patch(
                "helion._compiler.pallas.megakernel._STREAM_TARGET_BYTES", 1 << 20
            ),
        ):
            bound = tp_gemv_head.bind(args)
            code = bound.to_code(bound.config_spec.default_config())
        # One ring of 4 slots streams a's 8 row blocks, then w_out's 4.
        self.assertIn("((4, 128, 1024), 'jnp.bfloat16', 'vmem')", code)
        self.assertIn(
            "make_async_copy(a.at[pl.ds(384, 128), pl.ds(0, 1024)], ring.at[3], "
            "ring_sem.at[3]).start()",
            code,
        )
        self.assertIn("@pl.when(_gn >= 12)", code)
        start = code.index("remote_copy.start()\n")
        self.assertNotIn(
            "_catch_up", code[start : code.index("remote_copy.wait()", start)]
        )
        start = code.index("remote_copy_1.start()\n")
        catch_up = code[start : code.index("remote_copy_1.wait()", start)]
        self.assertIn("def _catch_up():", catch_up)
        for tile in range(4):
            self.assertIn(
                f"make_async_copy(w_out.at[pl.ds({tile * 128}, 128), "
                f"pl.ds(0, 1024)], ring.at[{tile}], ring_sem.at[{tile}]).start()",
                catch_up,
            )
        self.assertEqual(catch_up.count(".start()"), 5)

    def test_folded_loop_scalar_uses(self) -> None:
        """The loop variable indexes tensors and enters arithmetic in roots."""
        x = torch.randn(16, 256, device=DEVICE)
        w = torch.randn(3, 256, device=DEVICE)
        code, out = code_and_output(scale_stack, (x, w), block_sizes=[8, 8, 8, 8])
        expected = x.clone()
        for i in range(3):
            expected = (expected * w[i] + i) + 1.0
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        self.assertOneSequentialProgram(code)
        self.assertIn("@pl.loop(0, 3)\n    def _i_loop(i):", code)
        self.assertIn("lax.convert_element_type(i, jnp.float32)", code)
        self.assertEqual(code.count("def _root"), 4)

    def test_folded_single_root(self) -> None:
        """A folded loop around one root still runs as a sequential program."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def accumulate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            n = w.size(2)
            out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
            for i in range(w.size(0)):
                for tm, tn in hl.tile([m, n]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(k):
                        acc = torch.addmm(acc, x[tm, tk], w[i, tk, tn])
                    out[tm, tn] = out[tm, tn] + acc
            return out

        x = torch.randn(16, 256, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(3, 256, 256, device=DEVICE, dtype=torch.bfloat16) * 0.1
        expected = sum(x.float() @ w[i].float() for i in range(3))
        for depth in (2, 3):
            with self.subTest(depth=depth):
                code, out = code_and_output(
                    accumulate,
                    (x, w),
                    block_sizes=[16, 128, 128],
                    pallas_stream_depth=depth,
                )
                torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
                self.assertOneSequentialProgram(code)
                self.assertIn("@pl.loop(0, 3)\n    def _i_loop(i):", code)
                # x and w stream through rings of their own; x's small one
                # holds its whole stream.
                self.assertEqual(_ring_depths(code), [12, depth])
                self.assertIn("_g_1 = i * 4 + pid_shared * 2 + _j", code)
                self.assertIn("_layer = _gn // 4", code)

    def test_folded_periodic_stack(self) -> None:
        """A host ``hl.static_range`` in a folded body is unrolled at the AST
        level, and ``if``s on its variable keep one branch per copy: one loop
        iteration runs a whole period of two layer types with differently
        shaped weights, indexed by affine functions of the loop variable."""
        codes = {}
        for periods, period, depth in ((2, 3, 2), (2, 3, 5), (3, 2, 4)):
            args = _periodic_stack_args(periods, period)
            bound = periodic_stack.bind(args)
            # 2 roots per low-rank layer, 3 per MLP layer, and 2 outside.
            roots = 2 * (period - 1) + 3 + 2
            self.assertEqual(len(bound.host_function.device_ir.root_ids), roots)
            reference = bound.config_spec.autotune_reference_config()
            block_sizes = reference.config["block_sizes"]
            with self.subTest(periods=periods, period=period, depth=depth):
                code, out = code_and_output(
                    periodic_stack,
                    args,
                    block_sizes=block_sizes,
                    pallas_stream_depth=depth,
                )
                torch.testing.assert_close(
                    out, _periodic_stack_reference(*args), rtol=2e-2, atol=2e-2
                )
                self.assertOneSequentialProgram(code)
                self.assertEqual(_ring_depths(code), [depth])
                self.assertIn(f"@pl.loop(0, {periods})\n    def _p_loop(p):", code)
                stride = "" if period == 2 else f"{period - 1} * "
                self.assertIn(f"a.at[{stride}p, pl.ds(", code)
                codes[periods, period, depth] = code
        # Refills decode the iteration, then the copy within the period.
        self.assertIn("_layer = _gn // 24", codes[2, 3, 5])
        self.assertIn("b.at[1 + 2 * _layer, pl.ds(", codes[2, 3, 5])

    def test_folded_root_level_stream_sites(self) -> None:
        """Inside a folded host loop, a root that reads a per-layer weight
        whole streams it through a ring of its own: the window moves with the
        loop's layer index, and refills decode the layer of the copy.  (Lane
        dense, the narrow weights would be kept whole in VMEM.)"""
        for gates in (6, 128):
            args = _gated_stack_args(gates)
            bound = gated_stack.bind(args)
            self.assertEqual(
                _site_shapes(bound, pallas_lane_dense=False),
                [(None, (3, 2048, gates)), (4, (3, 2048, 128)), (7, (3, 128, 2048))],
            )
            block_sizes = bound.config_spec.autotune_reference_config().config[
                "block_sizes"
            ]
            with self.subTest(gates=gates):
                code, out = code_and_output(
                    gated_stack,
                    args,
                    block_sizes=block_sizes,
                    pallas_stream_depth=3,
                    pallas_lane_dense=False,
                )
                torch.testing.assert_close(
                    out, _gated_stack_reference(*args), rtol=2e-2, atol=2e-2
                )
                self.assertOneSequentialProgram(code)
                self.assertIn("@pl.loop(0, 3)", code)
                if gates == 128:
                    self.assertIn(
                        "make_async_copy(w_gate.at[i, :, :], w_gate_tile, ", code
                    )
                    self.assertIn("make_async_copy(w_gate.at[_layer, :, :], ", code)
        codes = {}
        for periods, period in ((2, 3), (3, 2)):
            args = _periodic_tap_stack_args(periods, period)
            bound = periodic_tap_stack.bind(args)
            taps = (periods * (period - 1), 2048, 4)
            self.assertEqual(
                [shape for _, shape in _site_shapes(bound, pallas_lane_dense=False)],
                [taps] * (period - 1) + [(periods, 2048, 256), (periods, 256, 2048)],
            )
            block_sizes = bound.config_spec.autotune_reference_config().config[
                "block_sizes"
            ]
            with self.subTest(periods=periods, period=period):
                code, out = code_and_output(
                    periodic_tap_stack,
                    args,
                    block_sizes=block_sizes,
                    pallas_stream_depth=4,
                    pallas_lane_dense=False,
                )
                torch.testing.assert_close(
                    out, _periodic_tap_stack_reference(*args), rtol=2e-2, atol=2e-2
                )
                self.assertIn(f"@pl.loop(0, {periods})", code)
                codes[period] = code
        # The two tap layers of period p read taps 2p and 2p + 1; the copies
        # read rows of a view of the taps, whose minor dim is ragged.
        self.assertIn("taps.reshape(8192, 4).at[pl.ds(2 * _layer * 2048,", codes[3])
        self.assertIn(
            "taps.reshape(8192, 4).at[pl.ds((1 + 2 * _layer_1) * 2048, 2048), :]",
            codes[3],
        )
        self.assertIn("taps.reshape(6144, 4).at[pl.ds(p * 2048, 2048), :]", codes[2])

    def test_ragged_minor_dim_stream(self) -> None:
        """A tile that spans a narrow, unaligned minor dim streams: its ring
        stacks the slots as rows, and a tensor of more than two dims is
        copied from a view of it as rows of that dim.  The slots are counted
        padded to whole VMEM tiles."""
        torch.manual_seed(0)
        dtype = torch.bfloat16
        x = torch.randn(16, 1024, device=DEVICE, dtype=dtype)
        for rank in (6, 128):
            a = torch.randn(3, 1024, rank, device=DEVICE, dtype=dtype) * 0.03
            b = torch.randn(3, rank, 1024, device=DEVICE, dtype=dtype) * 0.1
            args = (x, a, b)
            bound = narrow_rank_stack.bind(args)
            self.assertEqual(_site_shapes(bound), [(2, (3, 1024, rank))])
            with self.subTest(rank=rank):
                code, out = code_and_output(
                    narrow_rank_stack,
                    args,
                    block_sizes=[16, 16, 128, 16, 128, 16],
                    pallas_stream_depth=3,
                )
                s = x.float()
                for i in range(3):
                    t = (s.to(dtype).float() @ a[i].float()).to(dtype)
                    s = s + t.float() @ b[i].float()
                torch.testing.assert_close(out, s.to(dtype), rtol=2e-2, atol=2e-2)
            if rank == 6:
                self.assertIn("((384, 6), 'jnp.bfloat16', 'vmem')", code)
                self.assertIn(
                    "make_async_copy(a.reshape(3072, 6).at[pl.ds(128, 128), :], "
                    "ring.at[pl.ds(128, 128), :], ring_sem.at[1]).start()",
                    code,
                )
                self.assertIn(
                    "make_async_copy(a.reshape(3072, 6).at[pl.ds(i * 1024 + "
                    "pl.multiple_of(_j * 128, 128), 128), :], a_tile, ",
                    code,
                )
                self.assertIn("a_tile = ring.at[pl.ds(_slot * 128, 128), :]", code)
            else:
                self.assertIn("((3, 128, 128), 'jnp.bfloat16', 'vmem')", code)
                self.assertIn(
                    "a.at[i, pl.ds(pl.multiple_of(_j * 128, 128), 128), :]", code
                )
        # A 2-D tensor slices its rows directly.
        for gates in (6, 4):
            w = torch.randn(1024, gates, device=DEVICE, dtype=dtype) * 0.03
            bound = scale_then_narrow.bind((x, w))
            self.assertEqual(_site_shapes(bound), [(3, (1024, gates))])
            with self.subTest(gates=gates):
                code, out = code_and_output(
                    scale_then_narrow,
                    (x, w),
                    block_sizes=[16, 128, 16, 128],
                    pallas_stream_depth=3,
                )
                expected = (x * 2.0).float() @ w.float()
                torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
                self.assertIn(f"(({3 * 128}, {gates}), 'jnp.bfloat16', 'vmem')", code)
                self.assertIn(
                    "make_async_copy(w.at[pl.ds(pl.multiple_of(_gn * 128, 128), "
                    "128), :], ring.at[pl.ds(_slot * 128, 128), :], ",
                    code,
                )
        # Mosaic cannot slice a tile of more than two dims from a ragged minor
        # dim, so a load whose leading dim is not scalar is not streamed.
        w = torch.randn(4, 256, 6, device=DEVICE)
        y = torch.randn(16, 256, device=DEVICE)
        self.assertEqual(_streamed_shapes(grid_root_stream.bind((y, w))), [])
        code, out = code_and_output(
            grid_root_stream, (y, w), block_sizes=[16, 128, 128]
        )
        torch.testing.assert_close(
            out, torch.einsum("mk,ekn->emn", y * 2.0, w), rtol=1e-4, atol=1e-3
        )
        self.assertNotIn("make_async_copy(w", code)

    @_per_weight_rings()
    def test_ragged_tiled_dim_stream(self) -> None:
        """A tile over a dim that no smaller block copies in whole VMEM tiles
        (an intermediate size of 192, which no multiple of 128 divides)
        streams when its block can be the full size of the dim: the config
        searches that size only, the tile over the minor dim copies like a
        whole ragged one, from a view of the tensor as rows, and the launcher
        pads no weight."""
        args = _mlp_stack_args(8, 2, inter=192)
        bound = mlp_stack.bind(args)
        spec = bound.config_spec
        # gate/up stream in the gate/up root's K loop (block 4), tiled over
        # their 192 lanes by block 3; down in the down root's K loop over its
        # 192 rows (block 7).
        self.assertEqual(
            _site_shapes(bound),
            [(4, (2, 256, 192)), (4, (2, 256, 192)), (7, (2, 192, 256))],
        )
        for block_id in (3, 7):
            fragment = spec.block_sizes[block_id]._fragment(spec)
            self.assertEqual(fragment.search_values(), [192])
            self.assertEqual(fragment.default(), 192)
        self.assertEqual(
            spec.default_config().config["block_sizes"],
            [16, 16, 16, 192, 256, 16, 256, 192, 16],
        )
        for block_sizes in (None, [16, 16, 16, 192, 128, 16, 128, 192, 16]):
            with self.subTest(block_sizes=block_sizes):
                config = {} if block_sizes is None else {"block_sizes": block_sizes}
                code, out = code_and_output(mlp_stack, args, **config)
                torch.testing.assert_close(
                    out, _mlp_stack_reference(*args), rtol=2e-2, atol=2e-2
                )
                self.assertOneSequentialProgram(code)
                # Only the 8-row activations are padded, to 16-row tiles.
                self.assertIn("_ds_pad_dims=[(0, 0, 16, 0), (5, 0, 16, 0)]", code)
                self.assertIn("_hbm_arg_indices=[1, 2, 3]", code)
                rows = 256 if block_sizes is None else 128
                self.assertIn(f"(({4 * rows}, 192), 'jnp.bfloat16', 'vmem')", code)
                self.assertIn(f"((2, 192, {rows}), 'jnp.bfloat16', 'vmem')", code)
                self.assertIn(
                    "make_async_copy(w_gate.reshape(512, 192).at[pl.ds(i * 256 + "
                    f"pl.multiple_of(_j * {rows}, {rows}), {rows}), :], w_gate_tile, ",
                    code,
                )
                self.assertIn(
                    "w_down.at[i, pl.ds(pl.multiple_of(_j_1 * 192, 192), 192), ",
                    code,
                )
        # No smaller block divides the dim, so the stream rejects any other.
        bad = [16, 16, 16, 128, 128, 16, 128, 192, 16]
        with self.assertRaisesRegex(
            exc.InvalidConfig, "block size 128 does not divide 192"
        ):
            code_and_output(mlp_stack, args, block_sizes=bad)
        self.assertFalse(
            bound.env.backend.autotune_config_is_viable(
                spec, helion.Config(block_sizes=bad)
            )
        )
        # A power of two that divides the dim keeps the regular search.
        spec = mlp_stack.bind(_mlp_stack_args(8, 2, inter=256)).config_spec
        model = spec.pallas_stream_model
        assert model is not None
        self.assertEqual(model.whole_block_sizes, {})
        self.assertEqual(
            spec.block_sizes[3]._fragment(spec).search_values(), [128, 256]
        )

    @_per_weight_rings()
    def test_narrow_ring_floor_follows_runs(self) -> None:
        """A ring of tiles narrower than the 128 lanes keeps no more slots
        than its longest run of consecutive reads: a narrow tile read once
        per root, between roots that stream other rings, gets one slot.
        Wide rings keep two iterations' loads.  (Lane dense, the narrow
        weights would be read whole.)"""
        for gates, depths in ((6, [1, 2, 2]), (128, [2, 2, 2])):
            args = _gated_stack_args(gates)
            with self.subTest(gates=gates):
                code, out = code_and_output(
                    gated_stack, args, **_lane_dense_off(gated_stack, args)
                )
                torch.testing.assert_close(
                    out, _gated_stack_reference(*args), rtol=2e-2, atol=2e-2
                )
                self.assertEqual(_ring_depths(code), depths)
                if gates == 6:
                    self.assertIn("((2048, 6), 'jnp.bfloat16', 'vmem')", code)
        # Two tap layers in a row read the taps ring in one run of two.
        for periods, period, depths in ((2, 3, [2, 2, 2]), (3, 2, [1, 2, 2])):
            args = _periodic_tap_stack_args(periods, period)
            with self.subTest(periods=periods, period=period):
                code, out = code_and_output(
                    periodic_tap_stack,
                    args,
                    **_lane_dense_off(periodic_tap_stack, args),
                )
                torch.testing.assert_close(
                    out, _periodic_tap_stack_reference(*args), rtol=2e-2, atol=2e-2
                )
                self.assertEqual(_ring_depths(code), depths)

    def test_seed_shrinks_to_fit(self) -> None:
        """Each root sizes its default tiles alone; when the rings of all the
        roots do not fit together, or the one that streams the most holds
        fewer than four iterations at its default depth, every root's share
        halves until they do."""
        dtype = torch.bfloat16
        x = torch.empty(16, 5120, device=DEVICE, dtype=dtype)
        cases = [
            (
                gated_stack,
                (
                    x,
                    torch.empty(2, 5120, 6, device=DEVICE, dtype=dtype),
                    torch.empty(2, 5120, 512, device=DEVICE, dtype=dtype),
                    torch.empty(2, 512, 5120, device=DEVICE, dtype=dtype),
                ),
                8,
                # The first seed: (512, 512) up and (128, 5120) down tiles.
                [16, 16, 16, 512, 512, 16, 5120, 128, 16],
                [16, 16, 16, 512, 512, 16, 512, 512, 16],
            ),
            (
                periodic_tap_stack,
                (
                    x,
                    torch.empty(4, 5120, 4, device=DEVICE, dtype=dtype),
                    torch.empty(2, 5120, 2176, device=DEVICE, dtype=dtype),
                    torch.empty(2, 2176, 5120, device=DEVICE, dtype=dtype),
                    3,
                ),
                8,
                # The first seed: (128, 2176) up and (128, 2560) down tiles.
                # The taps, passed lane dense, are kept whole in VMEM, so its
                # rings fit, but too shallow.
                [16, 16, 2176, 128, 16, 2560, 128, 16],
                [16, 16, 128, 1280, 16, 1280, 128, 16],
            ),
        ]
        for kernel, args, vmem_mib, first, expected in cases:
            with self.subTest(kernel=kernel.name):
                with mock.patch.object(
                    launcher, "_CACHED_VMEM_LIMIT_BYTES", vmem_mib << 20
                ):
                    spec = kernel.bind(args).config_spec
                model = spec.pallas_stream_model
                assert model is not None
                self.assertEqual(spec.default_config().config["block_sizes"], expected)
                for sizes, fits in ((first, False), (expected, True)):
                    tuned = {
                        block.block_id: size
                        for block, size in zip(spec.block_sizes, sizes, strict=True)
                    }
                    error = model.config_error(tuned, None, False)
                    self.assertEqual(
                        error is None and not model.rings_shallow(tuned, False),
                        fits,
                        error,
                    )

    def test_tpu_default_layout(self) -> None:
        """The layout predictor reproduces every probed TPU default layout,
        and ``lane_dense_perm`` takes exactly the narrow-minor ones that are
        not row major."""
        dtypes = {"bf16": torch.bfloat16, "f32": torch.float32, "i8": torch.int8}
        groups = re.sub(r"\n\s+", " ", _TPU_DEFAULT_LAYOUTS).strip().splitlines()
        rows = 0
        for group in groups:
            perm, names, shapes = (part.split() for part in group.split("|"))
            expected = tuple(map(int, perm))
            for name, shape in itertools.product(names, shapes):
                sizes = [int(size) for size in shape.split("x")]
                with self.subTest(shape=shape, dtype=name):
                    self.assertEqual(tpu_default_layout(sizes, dtypes[name]), expected)
                    lane_dense = (
                        expected != tuple(range(len(sizes))) and sizes[-1] < 128
                    )
                    self.assertEqual(
                        lane_dense_perm(sizes, dtypes[name]),
                        expected if lane_dense else None,
                    )
                rows += 1
        self.assertEqual(rows, 186)

    def test_lane_dense_operands(self) -> None:
        """Narrow-minor inputs XLA lays out minor-swapped are passed lane
        dense: the launcher swaps their last two dims, the kernel indexes
        them in that order, transposes at loads and before stores, and a dot
        contracts a weight's physical minor dim instead of transposing it.
        The results match the logical layout's (``pallas_lane_dense=False``)."""
        for layers in (1, 3):
            args = _narrow_conv_stack_args(layers)
            expected, state_expected = _narrow_conv_stack_reference(*args)
            for lane_dense in (True, False):
                with self.subTest(layers=layers, lane_dense=lane_dense):
                    state_args = (args[0], args[1], args[2].clone(), args[3])
                    config = narrow_conv_stack.bind(
                        state_args
                    ).config_spec.default_config()
                    code, out = code_and_output(
                        narrow_conv_stack,
                        state_args,
                        **{**config.config, "pallas_lane_dense": lane_dense},
                    )
                    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
                    torch.testing.assert_close(
                        state_args[2], state_expected, rtol=1e-2, atol=1e-2
                    )
                    lane_dense_code = (
                        "_lane_dense_perms={1: (0, 2, 1), 2: (0, 2, 1), 3: (0, 2, 1)}",
                        "dimension_numbers=(((1,), (1,)), ((), ()))",
                        "window = jnp.swapaxes(conv_state[i, :, :], -1, -2)",
                        "conv_state[i, :, :] = jnp.swapaxes(window_1, -1, -2)",
                        "jnp.swapaxes(conv_w[i, :, :], -1, -2)",
                    )
                    for snippet in lane_dense_code:
                        if lane_dense:
                            self.assertIn(snippet, code)
                        else:
                            self.assertNotIn(snippet, code)
        self.assertEqual(
            list(
                narrow_conv_stack.bind(args).config_spec.pallas_lane_dense({}).values()
            ),
            [(0, 2, 1)] * 3,
        )

        # Weights too large to keep whole in VMEM stream through rings of
        # lane-dense slots stacked along the lanes, their narrow rows padded
        # to whole sublane tiles; the state stays in HBM, each layer's slice
        # copied in physical order and transposed.
        args = _narrow_conv_stack_args(12, h=4096)
        expected, state_expected = _narrow_conv_stack_reference(*args)
        state_args = (args[0], args[1], args[2].clone(), args[3])
        code, out = code_and_output(narrow_conv_stack, state_args)
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state_args[2], state_expected, rtol=1e-2, atol=1e-2)
        for rows, name in ((6, "w_gate"), (4, "conv_w")):
            self.assertRegex(code, rf"\(\({rows}, \d+\), 'jnp\.bfloat16', 'vmem'\)")
            self.assertIn(f"make_async_copy({name}.at[i, :, :], {name}_tile", code)
        self.assertIn("dimension_numbers=(((1,), (1,)), ((), ()))", code)
        self.assertIn("window = jnp.swapaxes(conv_state_slice[:, :], -1, -2)", code)
        self.assertIn("conv_state_rows[...] = jnp.swapaxes(window_1, -1, -2)", code)
        self.assertIn(
            "_lane_dense_perms={1: (0, 2, 1), 2: (0, 2, 1), 3: (0, 2, 1)}", code
        )

    def test_lane_dense_jax_fn(self) -> None:
        """The standalone JAX entrypoint swaps the lane-dense inputs, and an
        in-place one back, around the kernel."""
        import jax
        import jax.numpy as jnp
        import numpy as np

        args = _narrow_conv_stack_args(3)
        expected, state_expected = _narrow_conv_stack_reference(*args)
        bound = narrow_conv_stack.bind(args)
        code = bound.to_code(bound.config_spec.default_config(), options=_JAX_FN)
        self.assertIn(
            "_LANE_DENSE_PERMS = {1: (0, 2, 1), 2: (0, 2, 1), 3: (0, 2, 1)}", code
        )
        # Return the in-place conv state too.
        code = code.replace(
            "ds_pad_dims=_DS_PAD_DIMS)",
            "ds_pad_dims=_DS_PAD_DIMS, return_all_outputs=True)",
        )
        name = "megakernel_lane_dense_jax_fn"
        try:
            module = _import_module(code, name)
            jax_args = [
                jnp.asarray(a.float().cpu().numpy()).astype(jnp.bfloat16) for a in args
            ]
            state, out = jax.jit(module.narrow_conv_stack)(*jax_args)
        finally:
            sys.modules.pop(name, None)
        for actual, reference in ((out, expected), (state, state_expected)):
            self.assertEqual(actual.shape, reference.shape)
            np.testing.assert_allclose(
                np.asarray(actual).astype(np.float32),
                reference.float().cpu().numpy(),
                rtol=2e-2,
                atol=2e-2,
            )

    def test_lane_dense_untouched(self) -> None:
        """Inputs whose minor dim fills the 128 lanes, or that a load
        gathers over the narrow dim, are passed as they are."""
        args = _gated_stack_args(128)
        code, out = code_and_output(gated_stack, args)
        torch.testing.assert_close(
            out, _gated_stack_reference(*args), rtol=2e-2, atol=2e-2
        )
        self.assertNotIn("_lane_dense_perms", code)
        self.assertNotIn("swapaxes", code)
        self.assertNotIn("(((1,), (1,)), ((), ()))", code)

        torch.manual_seed(0)
        x = torch.randn(1, 1280, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(3, 1280, 6, device=DEVICE, dtype=torch.bfloat16)
        cols = torch.tensor([5, 0, 3], device=DEVICE, dtype=torch.int32)
        self.assertEqual(lane_dense_perm(list(w.shape), w.dtype), (0, 2, 1))
        code, out = code_and_output(narrow_column_scale, (x, w, cols))
        expected = x.float()
        for i, col in enumerate(cols.tolist()):
            expected = expected * w[i, :, col].float()[None, :]
        torch.testing.assert_close(out, expected.to(x.dtype), rtol=0, atol=0)
        self.assertNotIn("_lane_dense_perms", code)
        self.assertNotIn("swapaxes", code)
        self.assertIn(
            ((3, 1280, 6), "bfloat16"), _vmem_refs(narrow_column_scale, (x, w, cols))
        )

    def test_lane_dense_permuted(self) -> None:
        """Inputs XLA lays out in any other non-row-major dim order are passed
        lane dense in it too: layer ``i`` of an [L, H, G] stack laid out
        (2, 0, 1) is row ``i`` of each [L, H] plane, read directly in float32
        and through 32-bit words in bfloat16, and row ``i`` of an [L, G]
        vector laid out (1, 0) is rolled out of the lanes.  A stack with an
        odd count of bfloat16 rows per plane is passed as it is."""
        perms = "_lane_dense_perms={1: (2, 0, 1), 2: (1, 0), 3: (1, 0)}"
        rows = {
            torch.bfloat16: "lax.bitcast_convert_type(w_a.bitcast(jnp.uint32)"
            "[:, i // 2, :] >> lax.convert_element_type(i % 2 * 16, jnp.uint32) "
            "<< 16, jnp.float32).astype(jnp.bfloat16)",
            torch.float32: "w_a[:, i, :]",
        }
        for dtype, row in rows.items():
            args = _narrow_gate_stack_args(16, dtype=dtype)
            expected = _narrow_gate_stack_reference(*args)
            for lane_dense in (True, False):
                with self.subTest(dtype=dtype, lane_dense=lane_dense):
                    config = narrow_gate_stack.bind(args).config_spec.default_config()
                    code, out = code_and_output(
                        narrow_gate_stack,
                        args,
                        **{**config.config, "pallas_lane_dense": lane_dense},
                    )
                    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
                    lane_dense_code = (
                        perms,
                        row,
                        "dimension_numbers=(((1,), (1,)), ((), ()))",
                        "((0, 0), (0, 112))), -i % 128, axis=1)[:, 0]",
                    )
                    for snippet in lane_dense_code:
                        if lane_dense:
                            self.assertIn(snippet, code)
                        else:
                            self.assertNotIn(snippet, code)

        # Layer ``i`` of a stack too large to keep whole in VMEM streams at
        # root level through a ring of 32-bit slots, each holding the words
        # of rows ``i // 2 * 2`` and the next, stacked along the lanes since
        # Mosaic cannot slice a slot of 6 rows out of a 3-D ring.
        args = _narrow_gate_stack_args(16, h=8192)
        code, out = code_and_output(narrow_gate_stack, args)
        torch.testing.assert_close(
            out, _narrow_gate_stack_reference(*args), rtol=2e-2, atol=2e-2
        )
        self.assertIn(perms, code)
        self.assertRegex(code, r"\(\(6, \d+\), 'jnp\.uint32', 'vmem'\)")
        self.assertIn(
            "w_a_tile = ring.at[:, pl.ds(pl.multiple_of(_slot * 8192, 8192), 8192)]",
            code,
        )
        self.assertIn(
            "make_async_copy(w_a.bitcast(jnp.uint32).at[:, i // 2, :], w_a_tile", code
        )
        self.assertIn(
            "lax.bitcast_convert_type(w_a_tile[:, :] >> "
            "lax.convert_element_type(i % 2 * 16, jnp.uint32) << 16",
            code,
        )

        args = _narrow_gate_stack_args(7)
        self.assertEqual(
            tpu_default_layout(list(args[1].shape), args[1].dtype), (2, 0, 1)
        )
        code, out = code_and_output(narrow_gate_stack, args)
        torch.testing.assert_close(
            out, _narrow_gate_stack_reference(*args), rtol=2e-2, atol=2e-2
        )
        self.assertNotIn("_lane_dense_perms", code)
        self.assertNotIn("bitcast", code)

    def test_lane_dense_permuted_jax_fn(self) -> None:
        """The standalone JAX entrypoint permutes the lane-dense inputs into
        their dim orders, and in-place ones back by the inverse orders."""
        import jax
        import jax.numpy as jnp
        import numpy as np

        args = _narrow_gate_stack_args(16)
        expected = _narrow_gate_stack_reference(*args)
        bound = narrow_gate_stack.bind(args)
        code = bound.to_code(bound.config_spec.default_config(), options=_JAX_FN)
        self.assertIn("_LANE_DENSE_PERMS = {1: (2, 0, 1), 2: (1, 0), 3: (1, 0)}", code)
        name = "megakernel_lane_dense_permuted_jax_fn"
        try:
            module = _import_module(code, name)
            jax_args = [
                jnp.asarray(a.float().cpu().numpy()).astype(jnp.bfloat16) for a in args
            ]
            out = jax.jit(module.narrow_gate_stack)(*jax_args)
        finally:
            sys.modules.pop(name, None)
        self.assertEqual(out.shape, expected.shape)
        np.testing.assert_allclose(
            np.asarray(out).astype(np.float32),
            expected.float().cpu().numpy(),
            rtol=2e-2,
            atol=2e-2,
        )

        w = jnp.arange(24).reshape(2, 3, 4)
        v = jnp.arange(6).reshape(2, 3)
        fn = launcher._lane_dense_jit_fn(
            lambda w, v: (w + 1, v * 2), {0: (2, 0, 1), 1: (1, 0)}, {0: (2, 0, 1)}
        )
        w_out, v_out = fn(w, v)
        np.testing.assert_array_equal(w_out, w + 1)
        np.testing.assert_array_equal(v_out, v.T * 2)

    def test_host_static_range(self) -> None:
        """Host ``hl.static_range`` loops unroll into one copy of their roots
        per value; host ``if``s on constants keep one branch."""

        @helion.kernel(backend="pallas", static_shapes=True)
        def scaled_sum(
            x: torch.Tensor, w: torch.Tensor, odd: hl.constexpr = True
        ) -> torch.Tensor:
            out = torch.zeros_like(x)
            for r in hl.static_range(3):
                if r % 2 == 0 or odd:
                    for tile in hl.tile(x.size()):
                        out[tile] = out[tile] + x[tile] * w[r, 0]
            return out

        x = torch.randn(16, 128, device=DEVICE)
        w = torch.randn(4, 128, device=DEVICE)
        for odd, rows in ((True, (0, 1, 2)), (False, (0, 2))):
            with self.subTest(odd=odd):
                bound = scaled_sum.bind((x, w, odd))
                self.assertEqual(len(bound.host_function.device_ir.root_ids), len(rows))
                code, out = code_and_output(scaled_sum, (x, w, odd))
                expected = x * sum(w[row, 0] for row in rows)
                torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
                self.assertOneSequentialProgram(code)

    def test_folded_loop_compile_time_flat(self) -> None:
        """The body roots are traced once: the program does not grow with L."""
        codes = []
        for layers in (2, 16):
            args = _mlp_stack_args(8, layers)
            bound = mlp_stack.bind(args)
            self.assertEqual(len(bound.host_function.device_ir.root_ids), 5)
            config = helion.Config(block_sizes=_STACK_BLOCKS, pallas_stream_depth=6)
            codes.append(bound.to_code(config))
        short, long = codes
        self.assertIn("@pl.loop(0, 16)", long)
        self.assertEqual(short.count("\n"), long.count("\n"))
        self.assertEqual(short.count("make_async_copy"), long.count("make_async_copy"))

    def test_folded_loop_invalid_configs(self) -> None:
        args = _mlp_stack_args(8, 3)
        with self.assertRaisesRegex(exc.InvalidConfig, "smaller than the 2"):
            code_and_output(
                mlp_stack, args, block_sizes=_STACK_BLOCKS, pallas_stream_depth=1
            )
        with self.assertRaisesRegex(exc.InvalidConfig, "extent 384"):
            code_and_output(
                mlp_stack,
                _mlp_stack_args(8, 3, h=384),
                block_sizes=[16, 16, 16, 128, 256, 16, 128, 128, 16],
            )

        @helion.kernel(backend="pallas", static_shapes=True)
        def next_layer(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            n = w.size(2)
            out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
            for i in range(w.size(0)):
                for tm, tn in hl.tile([m, n]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(k):
                        acc = torch.addmm(acc, x[tm, tk], w[i + 1, tk, tn])
                    out[tm, tn] = out[tm, tn] + acc
            return out

        x = torch.randn(16, 256, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(2, 256, 128, device=DEVICE, dtype=torch.bfloat16)
        with self.assertRaisesRegex(exc.BackendUnsupported, "out of bounds"):
            code_and_output(next_layer, (x, w), block_sizes=[16, 128, 128])

    def test_folded_loop_errors(self) -> None:
        x = torch.randn(16, 128, device=DEVICE)
        w = torch.randn(4, 128, device=DEVICE)

        def folded(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for i in range(3):
                for tile in hl.tile(x.size()):
                    out[tile] = out[tile] + x[tile] * w[i, 0]
            return out

        def strided(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for i in range(0, 4, 2):
                for tile in hl.tile(x.size()):
                    out[tile] = out[tile] + x[tile] * w[i, 0]
            return out

        def static_range(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for i in hl.static_range(w.size(0)):
                for tile in hl.tile(x.size()):
                    out[tile] = out[tile] + x[tile] * w[i, 0]
            return out

        def for_else(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for i in range(3):
                for tile in hl.tile(x.size()):
                    out[tile] = out[tile] + x[tile] * w[i, 0]
            else:
                out = out + 1
            return out

        def host_compute(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for i in range(3):
                y = i * 2
                for tile in hl.tile(x.size()):
                    out[tile] = out[tile] + x[tile] * y
            return out

        def body_allocation(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for i in range(3):
                out = torch.zeros_like(x)
                for tile in hl.tile(x.size()):
                    out[tile] = x[tile] * w[i, 0]
            return out

        def nested(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for i in range(2):
                for j in range(2):
                    for tile in hl.tile(x.size()):
                        out[tile] = out[tile] + x[tile] * w[i, j]
            return out

        def while_loop(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            while x.size(0) > 32:
                for tile in hl.tile(x.size()):
                    out[tile] = out[tile] + x[tile]
            return out

        # Other backends keep requiring top-level device loops.
        with self.assertRaises(exc.NestedGridLoop):
            helion.kernel(backend="triton", static_shapes=True)(folded).bind((x, w))
        for fn, error, message in (
            (strided, exc.UnsupportedFoldedHostLoop, "step 1"),
            (static_range, exc.DeviceAPIOnHost, "static_range"),
            (for_else, exc.UnsupportedFoldedHostLoop, "for/else"),
            (host_compute, exc.UnsupportedFoldedHostLoop, "may only hold"),
            (body_allocation, exc.UnsupportedFoldedHostLoop, "may only hold"),
            (nested, exc.UnsupportedFoldedHostLoop, "cannot be nested"),
            (while_loop, exc.NestedGridLoop, ""),
        ):
            with self.subTest(kernel=fn.__name__):
                kernel = helion.kernel(backend="pallas", static_shapes=True)(fn)
                with self.assertRaisesRegex(error, message):
                    code_and_output(kernel, (x, w))

    def _check_cache_writes(
        self,
        kernel: helion.Kernel,
        args: tuple[torch.Tensor, ...],
        out: torch.Tensor,
        p: int,
    ) -> None:
        q, k_new, v_new, k_cache, v_cache, _ = args
        self.assertTrue(torch.equal(k_cache[:, p], k_new))
        self.assertTrue(torch.equal(v_cache[:, p], v_new))
        expected = _attention_reference(q, k_cache, v_cache, p)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    def assertCachesInHbm(
        self,
        code: str,
        kernel: helion.Kernel,
        args: Any,
        *,
        forwarded: bool,
        **config: object,
    ) -> None:
        """The caches stay in HBM under ``config``, read through a 2-slot DMA
        buffer.  The row write of a store in the scan's root is forwarded
        into the scan's tile and waited for after the scan, otherwise it is
        waited for before the scan's first read."""
        cache = (tuple(args[-2].shape), "bfloat16")
        self.assertNotIn(cache, _vmem_refs(kernel, args, **config))
        self.assertRegex(code, r"k_cache_buf\.at\[\(_j(_\d+)? \+ 1\) % 2\]")
        self.assertIn("k_cache_rows, k_cache.at[:, pl.ds(row_base", code)
        if forwarded:
            self.assertRegex(code, r"@pl.when\(_rows_iteration == _j(_\d+)?\)")
            self.assertGreater(
                code.rindex("_rows_out"), code.index("jax.lax.fori_loop(0, _num_")
            )
        else:
            self.assertNotIn("_forward_rows", code)
            self.assertLess(code.rindex("_rows_out"), code.index("_prime_fori_loads"))

    def test_hbm_resident_cache(self) -> None:
        """A cache written at a runtime row and read in an inner loop can
        stay in HBM: the row's sublane tile is patched and written back with
        a DMA."""
        for kernel, forwarded in (
            (write_and_attend, True),
            (write_and_attend_bounded, True),
            (write_then_attend, False),
        ):
            for p in (0, 15, 16, 255):
                with self.subTest(kernel=kernel.name, p=p):
                    args = _attention_args(p)
                    # Two scan iterations, rather than the default one.
                    code, out = code_and_output(
                        kernel, args, block_sizes=[128], pallas_hbm_resident=True
                    )
                    self._check_cache_writes(kernel, args, out, p)
            self.assertCachesInHbm(
                code, kernel, args, forwarded=forwarded, pallas_hbm_resident=True
            )

    def test_hbm_resident_cache_row_outside_scan(self) -> None:
        """A row that the scan of its root does not reach is written ahead
        of the scan, one it reaches from the scan's tile."""
        torch.manual_seed(0)
        k_new = torch.randn(2, 128, device=DEVICE, dtype=torch.bfloat16)
        for p in (0, 16, 17, 255):
            with self.subTest(p=p):
                k_cache = torch.randn(2, 256, 128, device=DEVICE, dtype=torch.bfloat16)
                pos = torch.tensor([p], device=DEVICE, dtype=torch.int32)
                expected = k_cache[:, :p].float().sum(1)
                expected_cache = k_cache.clone()
                expected_cache[:, p] = k_new * 2
                code, out = code_and_output(
                    write_and_sum_before,
                    (k_new, k_cache, pos),
                    block_sizes=[16],
                    pallas_hbm_resident=True,
                )
                torch.testing.assert_close(k_cache, expected_cache, rtol=0, atol=0)
                torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)
        self.assertIn("def _write_rows", code)
        self.assertIn("def _forward_rows", code)

    def test_hbm_resident_cache_data_dependent_trips(self) -> None:
        """A scan bounded by the decode position runs only the tiles up to it."""
        args = _attention_args(37)
        code, _ = code_and_output(
            write_and_attend_bounded, args, block_sizes=[128], pallas_hbm_resident=True
        )
        self.assertIn("_num_iterations = (v_", code)
        self.assertIn("@pl.when(_num_iterations > 0)", code)

    def test_hbm_resident_cache_root_read(self) -> None:
        """A cache also read at root scope stays a whole VMEM buffer."""
        torch.manual_seed(0)
        k_new = torch.randn(2, 128, device=DEVICE, dtype=torch.bfloat16)
        k_cache = torch.randn(2, 256, 128, device=DEVICE, dtype=torch.bfloat16)
        pos = torch.tensor([20], device=DEVICE, dtype=torch.int32)
        expected_cache = k_cache.clone()
        expected_cache[:, 20] = k_new
        expected = expected_cache.float().sum(1) + k_new.float()
        args = (k_new, k_cache, pos)
        _, out = code_and_output(write_and_read_row, args)
        torch.testing.assert_close(k_cache, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=0.5)
        self.assertIn(((2, 256, 128), "bfloat16"), _vmem_refs(write_and_read_row, args))

    def test_hbm_resident_cache_layer(self) -> None:
        """A GQA decode layer streams its weights through the ring.  By
        default it keeps its caches in HBM when their row stages take less
        VMEM than the whole caches, which leaves the rings more: with 16 MiB
        of VMEM and with 64."""
        for p in (0, 100):
            for hbm in (None, False):
                with self.subTest(p=p, hbm=hbm):
                    args = _gqa_layer_args(p)
                    expected, k_expected, v_expected = _gqa_layer_reference(*args)
                    config = gqa_decode_layer.bind(args).config_spec.default_config()
                    if hbm is not None:
                        config = helion.Config(**config, pallas_hbm_resident=hbm)
                    code, out = code_and_output(gqa_decode_layer, args, **config)
                    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
                    torch.testing.assert_close(args[6], k_expected, rtol=0, atol=0)
                    torch.testing.assert_close(args[7], v_expected, rtol=0, atol=0)
                    self.assertRegex(code, _ring_tile("wq"))
                    if hbm is None:
                        self.assertCachesInHbm(
                            code, gqa_decode_layer, args, forwarded=True
                        )
                    else:
                        cache = (tuple(args[6].shape), "bfloat16")
                        self.assertIn(
                            cache,
                            _vmem_refs(
                                gqa_decode_layer, args, pallas_hbm_resident=False
                            ),
                        )
        spec = gqa_decode_layer.bind(_gqa_layer_args(0)).config_spec
        self.assertIs(spec.pallas_hbm_resident_default, True)
        with mock.patch.object(launcher, "_CACHED_VMEM_LIMIT_BYTES", 64 << 20):
            spec = gqa_decode_layer.bind(_gqa_layer_args(0, ctx=512)).config_spec
        self.assertIs(spec.pallas_hbm_resident_default, True)

    def assertTop1(
        self,
        top_idx: torch.Tensor,
        top_val: torch.Tensor,
        logits: torch.Tensor,
        tol: float = 1e-3,
    ) -> None:
        """``top_idx`` picks a largest logit of each row (up to ``tol``, as
        the summation order differs) and ``top_val`` is its value."""
        best = logits.amax(-1)
        picked = logits.gather(-1, top_idx.long()[:, None])[:, 0]
        torch.testing.assert_close(picked, best, rtol=tol, atol=tol)
        torch.testing.assert_close(top_val, best, rtol=tol, atol=tol)

    def test_row_gather_embedding(self) -> None:
        """An embedding table read only at the token's row stays in HBM: the
        sublane tile holding the row is copied at kernel start, ahead of the
        LM head's ring, and the row selected from it at the load."""
        for token in (0, 17, 31999):
            with self.subTest(token=token):
                args = _embed_head_args(token, 32000, 512, 2048, 2000)
                _, embedding, hidden, final_norm, lm_head, vocab = args
                code, (next_input, top_idx, top_val) = code_and_output(
                    embed_head_step, args
                )
                torch.testing.assert_close(
                    next_input, embedding[token][None, :], rtol=0, atol=0
                )
                normed = _rms_norm(hidden, final_norm).to(hidden.dtype)
                self.assertTop1(
                    top_idx, top_val, _masked_logits(normed, lm_head, vocab)
                )
        self.assertNotIn(((32000, 512), "bfloat16"), _vmem_refs(embed_head_step, args))
        self.assertRegex(code, _ring_tile("lm_head"))
        body = code[code.index("def _helion_") :]
        start = body.index("embedding_row_copy.start()")
        self.assertLess(start, body.index("lm_head.at["))
        self.assertLess(start, body.index("\n    t = token[0]"))
        self.assertLess(
            body.index("\n    t = token[0]"), body.index("embedding_row_copy.wait()")
        )
        self.assertIn("pl.multiple_of(row // 16 * 16, 16)", code)
        self.assertIsNone(
            embed_head_step.bind(args).config_spec.pallas_hbm_resident_default
        )

    def test_row_gather_at_load(self) -> None:
        """Rows read at indices the kernel computes, two of one table, copy
        their sublane tiles at the load: f32 rows are selected with a
        runtime sublane index, packed rows by rotating the tile.  A table
        of at most 1 MiB stays whole in VMEM."""
        torch.manual_seed(0)
        x = torch.randn(1, 384, device=DEVICE)
        normed = _rms_norm(x, torch.ones(384, device=DEVICE))[0]
        for v, dtype in (
            (4000, torch.float32),
            (4000, torch.bfloat16),
            (256, torch.float32),
        ):
            table = torch.randn(v, 384, device=DEVICE).to(dtype)
            for first, second in ((0, v - 1), (17, 18), (v - 10, 5)):
                with self.subTest(v=v, dtype=dtype, tokens=(first, second)):
                    tokens = torch.tensor(
                        [first, second], device=DEVICE, dtype=torch.int32
                    )
                    code, out = code_and_output(scaled_token_pair, (x, tokens, table))
                    rows = table[[first, (second + 1) % v]].float()
                    torch.testing.assert_close(out, rows * normed, rtol=1e-6, atol=1e-6)
            refs = _vmem_refs(scaled_token_pair, (x, tokens, table))
            if v == 256:
                self.assertIn(((256, 384), "float32"), refs)
                self.assertNotIn("table_rows", code)
                continue
            self.assertNotIn(((v, 384), str(dtype).removeprefix("torch.")), refs)
            for copy in ("table_row_copy", "table_row_copy_1"):
                self.assertRegex(code, rf"{copy}\.start\(\)\n\s*{copy}\.wait\(\)")
            if dtype == torch.float32:
                self.assertIn("table_rows[row - row_base, :]", code)
            else:
                self.assertIn("pltpu.roll(", code)

    def test_streamed_top1(self) -> None:
        """The top-1 over the LM head streamed through the ring, as a
        running max over vocab tiles or over logits kept in scratch, takes
        the first of equal maxima, as torch.argmax does, and skips the
        padded columns, at power-of-two and other vocab sizes."""
        for kernel in (embed_head_step, logits_then_top1):
            for vp, vocab, first, second in (
                (2048, 2000, 5, 1900),
                (4992, 4900, 130, 131),
                (4992, 4900, 1700, 4800),
            ):
                with self.subTest(kernel=kernel.name, vp=vp, first=first):
                    hidden, lm_head = _planted_top1_args(384, vp, vocab, first, second)
                    if kernel is embed_head_step:
                        args = (
                            torch.tensor([3], device=DEVICE, dtype=torch.int32),
                            torch.randn(4000, 384, device=DEVICE, dtype=torch.bfloat16),
                            hidden,
                            torch.ones(384, device=DEVICE, dtype=torch.bfloat16),
                            lm_head,
                            vocab,
                        )
                        code, (_, top_idx, top_val) = code_and_output(kernel, args)
                    else:
                        code, (top_idx, top_val) = code_and_output(
                            kernel, (hidden, lm_head, vocab)
                        )
                    logits = _masked_logits(hidden, lm_head, vocab)
                    self.assertEqual(top_idx.tolist(), [first])
                    torch.testing.assert_close(top_val, logits[:, first])
                    self.assertRegex(code, _ring_tile("lm_head"))
                    # Interpret mode's jnp.argmax takes the first of ties but
                    # Mosaic's does not: the index comes from an iota instead.
                    self.assertNotIn("jnp.argmax", code)
                    self.assertIn("lax.broadcasted_iota(jnp.int32", code)

    def test_one_element_outputs_padded(self) -> None:
        """A one-element output, like a decode step's top-1, is padded to a
        lane row along its minor dim and sliced back: on TPU, a DMA out of a
        one-element VMEM buffer can miss the store just before it.  An
        output of two elements keeps its shape."""
        for m in (1, 2):
            with self.subTest(m=m):
                torch.manual_seed(0)
                vals = torch.randn(8, m, device=DEVICE)
                vals[[2, 5]] = 9.0
                idx = torch.arange(8 * m, device=DEVICE, dtype=torch.int32)
                idx = idx.reshape(8, m)
                args = (vals, idx)
                _, (top_idx, top_val) = code_and_output(gathered_top1, args)
                self.assertEqual(top_idx.tolist(), idx[2].tolist())
                self.assertEqual(top_val.tolist(), [18.0] * m)
                refs = _vmem_refs(gathered_top1, args)
                lanes = 128 if m == 1 else m
                self.assertIn(((lanes,), "int32"), refs)
                self.assertIn(((lanes,), "float32"), refs)

    def test_dense_step(self) -> None:
        """A dense decode step in one program: the embedding row, folded
        MLP layers, the final norm and the LM head's top-1.  With rings that
        refill, the LM head's ring starts at the last layer's last refill,
        so the stream does not drain between them."""
        shallow = {
            "block_sizes": [16, 16, 256, 128, 16, 128, 256, 16, 16, 512],
            "pallas_stream_depth": 3,
        }
        for layers in (1, 3):
            for config in ({}, shallow):
                with self.subTest(layers=layers, config=config):
                    args = _dense_step_args(layers)
                    code, (top_idx, top_val) = code_and_output(
                        dense_step, args, **config
                    )
                    logits = _dense_step_logits(*args)
                    self.assertTop1(top_idx, top_val, logits, tol=2e-2)
                    self.assertIn("embedding_row_copy.start()", code)
                    self.assertRegex(code, _ring_tile("lm_head"))
        prime = code.index("def _prime")
        self.assertLess(code.index("_refill_2():"), prime)
        self.assertLess(prime, code.index("lm_head.at["))

    def test_default_loop_blocks(self) -> None:
        """An inner loop that streams tiles of a tensor kept in HBM defaults
        to the smallest block whose tiles fill 256 KB; one that reads only
        tensors kept whole in VMEM, to its whole extent."""
        p = 1000
        # 1 MiB caches fit whole in the 16 MiB of VMEM of interpret mode.
        args = _attention_args(p, nkv=1, group=8, d=256, ctx=2048)
        bound = write_and_attend_bounded.bind(args)
        self.assertIs(bound.config_spec.pallas_hbm_resident_default, False)
        self.assertEqual(bound.config_spec.default_config().block_sizes, [2048])
        code, out = code_and_output(write_and_attend_bounded, args)
        self._check_cache_writes(write_and_attend_bounded, args, out, p)
        self.assertIn(
            ((1, 2048, 256), "bfloat16"), _vmem_refs(write_and_attend_bounded, args)
        )

        # 4 MiB caches do not, so they stay in HBM.
        args = _attention_args(p, nkv=1, group=8, d=256, ctx=8192)
        bound = write_and_attend_bounded.bind(args)
        self.assertIs(bound.config_spec.pallas_hbm_resident_default, True)
        # A [1, 512, 256] bf16 tile of each cache.
        self.assertEqual(bound.config_spec.default_config().block_sizes, [512])
        code, out = code_and_output(write_and_attend_bounded, args)
        self._check_cache_writes(write_and_attend_bounded, args, out, p)
        self.assertCachesInHbm(code, write_and_attend_bounded, args, forwarded=True)

        x = torch.randn(1024, 256, device=DEVICE)
        bound = tiled_colsum_scratch.bind((x,))
        # The inner loop over the rows reads scratch kept whole in VMEM; the
        # tile loops keep their generic defaults.
        reference = bound.config_spec.autotune_reference_config().block_sizes
        self.assertEqual(
            bound.config_spec.default_config().block_sizes, [*reference[:3], 1024]
        )
        _, out = code_and_output(tiled_colsum_scratch, (x,))
        torch.testing.assert_close(out, (x + 1.0).sum(0), rtol=1e-4, atol=1e-3)

    @_per_weight_rings()
    def test_hbm_resident_layer_state(self) -> None:
        """Each layer's recurrent state of a folded stack, read and rewritten
        whole, stays in HBM by default.  A slice is copied into a buffer of
        its own among the ring copies, once the previous layer's root is
        done with the buffer: layer 0's in the prologue, a later one's in a
        refill, or at its root when no ring copy follows.  Its write stays
        in flight through the next root, which does not touch the state.
        Passed lane dense, the conv state stays in HBM too, its slices
        copied in physical order and transposed."""
        args = _recurrent_state_stack_args(3)
        expected, conv_expected, rec_expected = _recurrent_state_stack_reference(*args)
        self.assertIs(
            recurrent_state_stack.bind(args).config_spec.pallas_hbm_resident_default,
            True,
        )
        refs = _vmem_refs(recurrent_state_stack, args, pallas_lane_dense=False)
        self.assertNotIn(((3, 256, 4), "bfloat16"), refs)
        self.assertNotIn(((3, 8, 256), "float32"), refs)
        self.assertIn(((256, 4), "bfloat16"), refs)
        self.assertIn(((8, 256), "float32"), refs)
        conv = "conv_state.reshape(768, 4).at[pl.ds({}, 256), :], conv_state_slice"
        default = recurrent_state_stack.bind(args).config_spec.default_config()
        for depth in (None, 3):
            with self.subTest(depth=depth):
                state_args = (args[0], args[1].clone(), args[2].clone(), *args[3:])
                config = {
                    **default.config,
                    "pallas_stream_depth": depth,
                    "pallas_lane_dense": False,
                }
                code, out = code_and_output(recurrent_state_stack, state_args, **config)
                torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)
                torch.testing.assert_close(state_args[1], conv_expected, rtol=0, atol=0)
                torch.testing.assert_close(state_args[2], rec_expected, rtol=0, atol=0)
                # Layer 0's slices, ahead of the ring's first copy.
                self.assertLess(
                    code.index(f"{conv.format(0)}, conv_state_slice_sem).start()"),
                    code.index("make_async_copy(w_up"),
                )
                self.assertLess(
                    code.index("rec_state.at[0, :, :], rec_state_slice"),
                    code.index("make_async_copy(w_up"),
                )
                # The writes are waited for after the up projection's root.
                self.assertGreater(
                    code.index("_rows_out_1.wait()"),
                    code.index("_fori_body_0, None)"),
                )
                # Layer 1's slices, in a refill of a ring (the down
                # projection's, which the up projection's root does not
                # read), before the copy that refill issues; layer 2's, at
                # its root, since no ring copy follows.
                early = code.index(conv.format(256))
                self.assertLess(code.index("def _fori_body_0"), early)
                self.assertLess(early, code.index("def _refill():"))
                self.assertIn("rec_state.at[1, :, :], rec_state_slice", code)
                self.assertIn("@pl.when(i == 2)", code)
        state_args = (args[0], args[1].clone(), args[2].clone(), *args[3:])
        code, out = code_and_output(recurrent_state_stack, state_args)
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)
        torch.testing.assert_close(state_args[1], conv_expected, rtol=0, atol=0)
        torch.testing.assert_close(state_args[2], rec_expected, rtol=0, atol=0)
        self.assertRegex(code, r"_lane_dense_perms=\{\d: \(0, 2, 1\)\}")
        self.assertIn("jnp.swapaxes(conv_state_slice[:, :], -1, -2)", code)
        self.assertIn("rec_state.at[0, :, :], rec_state_slice", code)

    def test_hbm_resident_batched_state(self) -> None:
        """A root that reads and rewrites one slice of a layer's state per
        sequence keeps the state in HBM: each slice load copies into a
        buffer of its own, and the write of one sequence's slice stays in
        flight while the next sequences' are read."""
        layers, m, h, rank = 3, 4, 256, 8
        torch.manual_seed(0)
        args = (
            torch.randn(m, h, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(layers, m, rank, h, device=DEVICE, dtype=torch.float32),
            torch.randn(layers, h, 512, device=DEVICE, dtype=torch.bfloat16) * 0.05,
            torch.randn(layers, 512, h, device=DEVICE, dtype=torch.bfloat16) * 0.05,
        )
        x, rec_state, w_up, w_down = args
        rec_expected = rec_state.clone()
        s = x.float()
        for i in range(layers):
            rec_expected[i] = rec_expected[i] * 0.5 + s[:, None, :]
            s = s + torch.tanh(rec_expected[i]).mean(1)
            act = torch.nn.functional.silu(s.to(x.dtype).float() @ w_up[i].float())
            s = s + act.to(x.dtype).float() @ w_down[i].float()
        self.assertNotIn(
            ((layers, m, rank, h), "float32"), _vmem_refs(batched_state_stack, args)
        )
        state_args = (x, rec_state.clone(), w_up, w_down)
        code, out = code_and_output(batched_state_stack, state_args)
        torch.testing.assert_close(out, s.to(x.dtype), rtol=2e-2, atol=5e-2)
        torch.testing.assert_close(state_args[1][0], rec_expected[0], rtol=0, atol=0)
        torch.testing.assert_close(state_args[1], rec_expected, rtol=2e-2, atol=1e-3)
        for b in range(m):
            self.assertRegex(
                code, rf"rec_state\.at\[[^\n]*, {b}, :, :\], rec_state_slice"
            )
        self.assertIn("rec_state_slice_3", code)
        self.assertNotRegex(code, r"load_\d+ = rec_state\[")
        # The first sequence's write: its start, then its wait.
        first_write = re.findall(
            r"(_rows_out\w*) = pltpu.make_async_copy\(rec_state_rows\w*, "
            r"rec_state\.at\[[^\n]*, 0, :, :\]",
            code,
        )
        self.assertLess(
            code.index("rec_state_slice_3, rec_state_slice_sem_3).wait()"),
            code.index(f"{first_write[-1]}.wait()"),
        )

    def test_hbm_resident_slices_shared(self) -> None:
        """Two roots of a loop body that read state slices of one shape share
        the slice buffers when VMEM bounds the ring, so the ring takes what
        the second root's buffers would, and each copy starts ahead of a
        ring copy once the root before it that reads the buffer is done;
        with VMEM to spare, each load keeps a buffer of its own."""
        layers, m, h, rank = 4, 4, 256, 8
        torch.manual_seed(0)
        x = torch.randn(m, h, device=DEVICE, dtype=torch.bfloat16)
        rec_state = torch.randn(layers, m, rank, h, device=DEVICE)
        w_up = torch.randn(layers, h, 512, device=DEVICE, dtype=torch.bfloat16)
        w_down = torch.randn(layers, 512, h, device=DEVICE, dtype=torch.bfloat16)
        w_up, w_down = w_up * 0.05, w_down * 0.05
        rec_expected = rec_state.clone()
        s = x.float()
        for i in range(layers):
            rec_expected[i] = rec_expected[i] * 0.5 + s[:, None, :]
            s = s + torch.tanh(rec_expected[i]).mean(1)
            act = torch.nn.functional.silu(s.to(x.dtype).float() @ w_up[i].float())
            s = s + act.to(x.dtype).float() @ w_down[i].float()
        for vmem, shared in ((64 << 20, False), (3400 << 10, True)):
            with self.subTest(shared=shared):
                paired_state_stack.reset()
                state = rec_state.clone()
                with mock.patch.object(launcher, "_CACHED_VMEM_LIMIT_BYTES", vmem):
                    code, out = code_and_output(
                        paired_state_stack, (x, state, w_up, w_down)
                    )
                torch.testing.assert_close(out, s.to(x.dtype), rtol=2e-2, atol=5e-2)
                torch.testing.assert_close(state, rec_expected, rtol=2e-2, atol=1e-3)
                self.assertIn("rec_state_slice_3,", code)
                if not shared:
                    self.assertIn("rec_state_slice_7,", code)
                    continue
                self.assertNotIn("rec_state_slice_4,", code)
                # The second root's copy of sequence 0, into the first root's
                # buffer, starts in a refill of the first root's MLP.
                second = (
                    "rec_state.at[1 + 2 * _layer_1, 0, :, :], rec_state_slice, "
                    "rec_state_slice_sem_4).start()"
                )
                self.assertLess(code.index("def _refill"), code.index(second))
                self.assertLess(
                    code.index(second),
                    code.index("rec_state_slice, rec_state_slice_sem_4).wait()"),
                )

    def test_hbm_resident_snapshots(self) -> None:
        """Per-row snapshots of each layer's states stay in HBM: the stores
        of one shape to a tensor take turns in a ring of stages, so none
        waits for the write before it, and a snapshot passed lane dense is
        staged transposed into its physical dim order and written whole."""
        layers, m, h, taps, rank = 3, 8, 256, 4, 8
        x, conv_state, rec_state, w_up, w_down = _recurrent_state_stack_args(
            layers, m=m, h=h, taps=taps, rank=rank
        )
        expected = _state_snapshot_stack_reference(
            x, conv_state, rec_state, w_up, w_down
        )
        conv_snap = torch.empty(layers, m, h, taps, device=DEVICE, dtype=x.dtype)
        rec_snap = torch.empty(layers, m, rank, h, device=DEVICE)
        args = (x, conv_state, rec_state, conv_snap, rec_snap, w_up, w_down)
        self.assertIs(
            state_snapshot_stack.bind(args).config_spec.pallas_hbm_resident_default,
            True,
        )
        refs = _vmem_refs(state_snapshot_stack, args)
        self.assertNotIn(((layers, m, taps, h), "bfloat16"), refs)
        self.assertNotIn(((layers, m, rank, h), "float32"), refs)
        code, out = code_and_output(state_snapshot_stack, args)
        torch.testing.assert_close(out, expected[0], rtol=2e-2, atol=5e-2)
        # Layers past the first take rows of the hidden after an MLP, which
        # the kernel rounds differently.
        for got, want in zip(
            (conv_state, rec_state, conv_snap, rec_snap), expected[1:], strict=True
        ):
            torch.testing.assert_close(got, want, rtol=2e-2, atol=5e-2)
        self.assertIn("5: (0, 1, 3, 2)}", code)
        # One stage per row, and no write waited for until the root ends.
        self.assertIn("rec_snap_rows_7", code)
        self.assertIn("conv_snap_rows_7", code)
        self.assertNotIn("rec_snap_rows_8", code)
        rows = code[
            code.index("conv_snap.at[i, 0, :, :]") : code.index(
                "rec_snap.at[i, 7, :, :]"
            )
        ]
        self.assertNotIn(".wait()", rows)
        self.assertIn("conv_snap_rows[...] = jnp.swapaxes(", code)

    def test_lane_dense_transposes_cancel(self) -> None:
        """Values of tensors passed lane dense that the kernel transposes
        stay in physical dim order: a snapshot stored as the transpose of a
        time-major window, into a tensor kept in HBM, is staged from the
        window itself, and a state or weight read transposed (after a dtype
        conversion, too) is read as laid out: the state, kept in HBM, from
        the buffer its slice is copied into."""
        layers, m, h, taps = 3, 8, 256, 4
        x, conv_state, _, w_up, w_down = _recurrent_state_stack_args(
            layers, m=m, h=h, taps=taps
        )
        conv_w = torch.randn(layers, h, taps, device=DEVICE, dtype=x.dtype)
        expected = _transposed_snapshot_stack_reference(
            x, conv_state, conv_w, w_up, w_down
        )
        conv_snap = torch.empty(layers, m, h, taps, device=DEVICE, dtype=x.dtype)
        args = (x, conv_state, conv_w, conv_snap, w_up, w_down)
        code, out = code_and_output(transposed_snapshot_stack, args)
        torch.testing.assert_close(out, expected[0], rtol=2e-2, atol=5e-2)
        for got, want in zip((conv_state, conv_snap), expected[1:], strict=True):
            torch.testing.assert_close(got, want, rtol=2e-2, atol=5e-2)
        self.assertIn(
            "_lane_dense_perms={3: (0, 2, 1), 4: (0, 1, 3, 2), 5: (0, 2, 1)}", code
        )
        self.assertIn("conv_snap_rows_7", code)
        self.assertNotIn("jnp.swapaxes(", code)
        self.assertIn("conv_snap_rows[...] = subscript_1.astype(", code)
        self.assertIn("load_1 = conv_state_slice[:, :]", code)
        self.assertIn("conv_state_rows[...] = subscript", code)
        self.assertIn("subscript = load_1[1:4, :]", code)
        self.assertIn("v_2 = lax.convert_element_type(load_3, jnp.float32)", code)
        self.assertIn("subscript_11 = v_2[3:4, :]", code)

    def test_hbm_resident_block_rows(self) -> None:
        """A block of rows stored one at a time to a cache kept in HBM, each
        patched into the sublane tile holding it, shares one stage: every
        store waits for the write before it, so a tile it reads back holds
        the rows already written to it, across a tile boundary too."""
        torch.manual_seed(0)
        layers, m, nkv, d = 2, 8, 2, 128
        for p in (0, 12, 120):
            with self.subTest(p=p):
                x = torch.randn(m, 256, device=DEVICE, dtype=torch.bfloat16)
                w_k = (
                    torch.randn(layers, 256, nkv * d, device=DEVICE, dtype=x.dtype)
                    * 0.05
                )
                k_cache = torch.randn(layers, nkv, 128, d, device=DEVICE, dtype=x.dtype)
                pos = torch.tensor([p], device=DEVICE, dtype=torch.int32)
                expected = k_cache.clone()
                k = (x.float() @ w_k.float()).to(x.dtype)
                expected[:, :, p : p + m, :] = k.reshape(layers, m, nkv, d).transpose(
                    1, 2
                )
                code, _ = code_and_output(block_kv_rows, (x, w_k, k_cache, pos))
                torch.testing.assert_close(k_cache, expected, rtol=1e-2, atol=1e-2)
                self.assertNotIn(
                    ((layers, nkv, 128, d), "bfloat16"),
                    _vmem_refs(block_kv_rows, (x, w_k, k_cache, pos)),
                )
                self.assertNotIn("k_cache_rows_1", code)
                # The run of row stores reads and writes the window of two
                # sublane tiles that holds its rows once.
                self.assertIn("pl.ds(row_base_0, 32)", code)
                self.assertNotIn("row_base_1", code)
                self.assertEqual(code.count("_rows_in.start()"), 1)
                self.assertEqual(code.count("_rows_out.start()"), 1)
                self.assertLess(
                    code.index("row_block_7"), code.index("_rows_out.start()")
                )

    @_per_weight_rings()
    def test_hbm_resident_layer_caches(self) -> None:
        """Each layer's KV caches of a folded stack stay in HBM by default:
        a row store whose row the scan forwards into its tiles, and the
        scan's first tile copied among the ring copies.  Behind the rings,
        the scan's default block spans the whole context."""
        default = cache_stack.bind(_cache_stack_args(3, 0)).config_spec.default_config()
        for p in (0, 100, 255):
            args = _cache_stack_args(3, p)
            expected, k_expected, v_expected = _cache_stack_reference(*args)
            for depth in (None, 2):
                with self.subTest(p=p, depth=depth):
                    cache_args = (*args[:4], args[4].clone(), args[5].clone(), args[6])
                    config = dict(default.config)
                    if depth is not None:
                        config["pallas_stream_depth"] = depth
                    code, out = code_and_output(cache_stack, cache_args, **config)
                    torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)
                    # The new rows round the projections differently.
                    torch.testing.assert_close(
                        cache_args[4], k_expected, rtol=1e-2, atol=1e-2
                    )
                    torch.testing.assert_close(
                        cache_args[5], v_expected, rtol=1e-2, atol=1e-2
                    )
        self.assertIs(
            cache_stack.bind(args).config_spec.pallas_hbm_resident_default, True
        )
        cache = ((3, 2, 256, 128), "bfloat16")
        self.assertNotIn(cache, _vmem_refs(cache_stack, args))
        # 16 MiB of VMEM leaves the scan two 128-row tiles; with 64, it spans
        # the context, rather than the 512 rows that fill 256 KB.
        self.assertEqual(default.block_sizes[4], 128)
        self.assertIn("def _forward_rows", code)
        with mock.patch.object(launcher, "_CACHED_VMEM_LIMIT_BYTES", 64 << 20):
            spec = cache_stack.bind(_cache_stack_args(3, 0, ctx=1024)).config_spec
        self.assertEqual(spec.default_config().block_sizes[4], 1024)
        # Layer 0's first tiles, between the ring copies of its projections
        # and of its output projection.
        prime = "k_cache.at[pl.ds({}, 1), :, pl.ds(0, 128), :], k_cache_buf.at[0]"
        self.assertLess(code.index("w_v.at[0,"), code.index(prime.format(0)))
        self.assertLess(code.index(prime.format(0)), code.index("w_o.at[0,"))
        # Layer 1's, in a refill of the output projection's ring, once layer
        # 0's scan is done with the buffer; layer 2's, at the scan.
        early = code.index(prime.format(1))
        self.assertLess(code.index("def _fori_body_2"), early)
        self.assertLess(early, code.index("_fori_body_2, None)"))
        self.assertIn("v_cache.at[pl.ds(1, 1), :, pl.ds(0, 128), :]", code)
        self.assertIn("@pl.when(i == 2)", code)

    @_per_weight_rings()
    def test_hbm_resident_bounded_scan_prime(self) -> None:
        """The first tiles of a scan that stops at a runtime row start early
        too, among the ring copies: always, and a scan of no tile, which the
        bound allows, waits for them after the loop."""
        spec = cache_stack_bounded.bind(_cache_stack_args(3, 0)).config_spec
        config = spec.default_config()
        for p in (0, 100, 255):
            with self.subTest(p=p):
                args = _cache_stack_args(3, p)
                expected, k_expected, v_expected = _cache_stack_reference(*args)
                code, out = code_and_output(cache_stack_bounded, args, **config)
                torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)
                torch.testing.assert_close(args[4], k_expected, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(args[5], v_expected, rtol=1e-2, atol=1e-2)
        prime = "k_cache.at[pl.ds({}, 1), :, pl.ds(0, 128), :], k_cache_buf.at[0]"
        self.assertLess(code.index("w_v.at[0,"), code.index(prime.format(0)))
        self.assertLess(code.index(prime.format(0)), code.index("w_o.at[0,"))
        self.assertIn("@pl.when(_num_iterations == 0)", code)
        self.assertNotIn("_prime_fori_loads", code)

    def test_rings_leave_loop_buffers(self) -> None:
        """The rings deepen only into the VMEM the inner loops' own DMA
        buffers leave: two per streamed tile, at the config's block size."""
        from helion._compiler.pallas.megakernel import _STREAM_RESERVE_BYTES
        from helion._compiler.pallas.megakernel import resolve_ring_depths

        with mock.patch.object(launcher, "_CACHED_VMEM_LIMIT_BYTES", 64 << 20):
            spec = cache_stack.bind(_cache_stack_args(3, 0, ctx=1024)).config_spec
        model = spec.pallas_stream_model
        assert model is not None
        tuned = {
            block.block_id: size
            for block, size in zip(
                spec.block_sizes, spec.default_config().block_sizes, strict=True
            )
        }
        scan = spec.block_sizes[4].block_id
        self.assertEqual(tuned[scan], 1024)
        # Two buffers each of the key and the value tile, [2, 1024, 128] bf16.
        loops = model.loop_buffer_bytes(tuned, True)
        self.assertEqual(loops, 4 * 2 * 1024 * 128 * 2)
        self.assertEqual(
            model.loop_buffer_bytes({**tuned, scan: 128}, True), loops // 8
        )
        # With the caches whole in VMEM the scan streams nothing.
        self.assertEqual(model.loop_buffer_bytes(tuned, False), 0)

        def ring_bytes(capacity: int) -> int:
            resolved = resolve_ring_depths(model, tuned, None, capacity, True)
            return sum(depth * layout.slot_bytes for layout, depth in resolved)

        # The rings fit next to the loops' buffers, the resident tensors, the
        # loop-carried scratch and the reserve; without the loops' share of
        # the VMEM they do not.
        full = ring_bytes(model.vmem_capacity)
        fits = (
            _STREAM_RESERVE_BYTES
            + model.resident(True)
            + model.overhang_bytes(tuned)
            + model.carried_bytes(tuned)
            + loops
            + full
        )
        self.assertEqual(ring_bytes(fits), full)
        with self.assertRaisesRegex(exc.InvalidConfig, "inner-loop buffer"):
            ring_bytes(fits - loops)

    def test_fused_projection_row_slices(self) -> None:
        """Static slices of a row of a buffer written on device, sliced
        again (a fused key-value projection, rotate-half on the key), index
        the row's value."""
        torch.manual_seed(0)
        layers, nkv, d = 2, 2, 128
        args = (
            torch.randn(1, 256, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(layers, 256, 2 * nkv * d, device=DEVICE, dtype=torch.bfloat16)
            * 0.05,
            torch.randn(layers, nkv, 64, d, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(layers, nkv, 64, d, device=DEVICE, dtype=torch.bfloat16),
            torch.tensor([5], device=DEVICE, dtype=torch.int32),
        )
        k_expected, v_expected = _fused_kv_rows_reference(*args)
        code_and_output(fused_kv_rows, args)
        torch.testing.assert_close(args[2], k_expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(args[3], v_expected, rtol=1e-2, atol=1e-2)

    def test_aliased_intermediate_rejected(self) -> None:
        x = torch.randn(64, 256, device=DEVICE)
        with self.assertRaisesRegex(exc.BackendUnsupported, "aliased tensors"):
            code_and_output(view_of_intermediate, (x,), block_sizes=[8, 128, 8, 128])

    def test_dynamic_shapes_rejected(self) -> None:
        @helion.kernel(backend="pallas", static_shapes=False)
        def dynamic_roots(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.empty_like(x)
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                tmp[tile] = x[tile] + 1.0
            for tile in hl.tile(x.size()):
                out[tile] = tmp[tile] * 2.0
            return out

        x = torch.randn(64, 128, device=DEVICE)
        with self.assertRaisesRegex(exc.BackendUnsupported, "dynamic shapes"):
            code_and_output(dynamic_roots, (x,))

    def _check_experts(
        self,
        kernel: object,
        args: tuple[object, ...],
        expected: tuple[torch.Tensor, ...],
        depth: int | None = None,
        block_sizes: list[int] | None = None,
    ) -> str:
        """Run ``kernel`` with its default config, at stream depth ``depth``
        and with ``block_sizes`` if given, and check its outputs."""
        bound = kernel.bind(args)  # pyrefly: ignore[missing-attribute]
        config = dict(bound.config_spec.default_config())
        if depth is not None:
            config["pallas_stream_depth"] = depth
        if block_sizes is not None:
            config["block_sizes"] = block_sizes
        code, out = code_and_output(kernel, args, **config)
        outs = out if isinstance(out, tuple) else (out,)
        for actual, reference in zip(outs, expected, strict=True):
            torch.testing.assert_close(
                actual.float(), reference.float(), rtol=2e-2, atol=2e-2
            )
        self.assertOneSequentialProgram(code)
        return code

    def test_indexed_weight_load_double_buffers(self) -> None:
        """A read-only inner-loop load at a runtime index the ring cannot
        follow (a grid index, an index read at a runtime index) takes two
        buffers, so the next tile's copy overlaps this one's compute."""
        torch.manual_seed(0)
        x = torch.randn(16, 256, device=DEVICE, dtype=torch.bfloat16)
        idx = torch.tensor([[7, 0]], device=DEVICE, dtype=torch.int32)
        w_gate, _, _ = _expert_weights(8, 256, 512)
        perm = torch.randperm(8, device=DEVICE).to(torch.int32)
        y = (x * 2.0).to(x.dtype)
        cases = [
            (
                grid_slot_experts,
                (x, idx, w_gate),
                torch.stack([y.float() @ w_gate[e].float() for e in idx[0]]),
            ),
            (
                permuted_experts,
                (x, idx, perm, w_gate),
                sum(x.float() @ w_gate[int(perm[e])].float() for e in idx[0]),
            ),
        ]
        for kernel, args, expected in cases:
            with self.subTest(kernel=kernel.name):
                bound = kernel.bind(args)
                model = bound.config_spec.pallas_stream_model
                self.assertTrue(
                    model is None
                    or all(
                        load.fake.dim() != 3
                        for site in model.sites
                        for load in site.loads
                    )
                )
                code = self._check_experts(kernel, args, (expected,))
                self.assertIn("w_gate_buf.at[(_j + 1) % 2]", code)
                self.assertIn("w_gate_buf.at[_j % 2]", code)

    def test_routed_experts_stream_by_id(self) -> None:
        """Expert weights indexed by ids read from a read-only index vector
        join the ring; a copy that runs ahead reads the id again, clamped."""
        args = _routed_experts_args([5, 2])
        bound = routed_experts.bind(args)
        shapes = [shape for _, shape in _site_shapes(bound)]
        self.assertEqual(shapes, [(8, 256, 256)] * 3)
        # Repeated experts and the first and last ids, with refills across
        # the folded loop over k.
        for ids in ([5, 2], [3, 3], [0, 7], [7, 0]):
            with self.subTest(ids=ids):
                args = _routed_experts_args(ids)
                code = self._check_experts(
                    routed_experts, args, (_routed_experts_reference(*args),), depth=2
                )
                self.assertEqual(_ring_depths(code), [2])
                self.assertIn("w_down.at[jnp.clip(topk_idx[0, _layer], 0, 7)", code)
                # The waits only need the copy's shape.
                self.assertIn("w_gate.at[0, pl.ds(", code)

    def test_shared_and_routed_experts_stream(self) -> None:
        """A shared expert and routed experts of the same tile shape share
        a ring: the routed tiles follow the shared ones in program order."""
        torch.manual_seed(0)
        args = _routed_experts_args([6, 1])
        s_gate, s_up, s_down = (w[0] for w in _expert_weights(1, 256, 512))
        x = args[0]
        expected = (
            x.float()
            + _swiglu(x, s_gate, s_up, s_down)
            + _routed_experts_reference(*args)
        )
        full = (*args, s_gate, s_up, s_down)
        bound = shared_and_routed_experts.bind(full)
        shapes = [shape for _, shape in _site_shapes(bound)]
        self.assertEqual(
            shapes, [(256, 512), (256, 512), (512, 256), *[(8, 256, 256)] * 3]
        )
        code = self._check_experts(shared_and_routed_experts, full, (expected,))
        self.assertIn("jnp.clip(topk_idx[0, ", code)

    def test_batch_routes_dense_combine(self) -> None:
        """Several tokens' routes (B > 1) stream their experts by id and
        combine densely."""
        torch.manual_seed(0)
        tokens, topk, experts = 8, 2, 8
        x = torch.randn(tokens, 256, device=DEVICE, dtype=torch.bfloat16)
        ids = torch.randint(0, experts, (tokens, topk), device=DEVICE)
        ids[0] = torch.tensor([0, experts - 1])
        ids[1] = torch.tensor([3, 3])
        weights = torch.rand(tokens, topk, device=DEVICE)
        route_idx = ids.reshape(-1).to(torch.int32)
        coef = torch.zeros(tokens * topk, tokens, device=DEVICE)
        for t in range(tokens):
            coef[t * topk : (t + 1) * topk, t] = weights[t]
        w_gate, w_up, w_down = _expert_weights(experts, 256, 256)
        expected = torch.zeros(tokens, 256, device=DEVICE)
        for r, e in enumerate(route_idx.tolist()):
            expected += coef[r][:, None] * _swiglu(x, w_gate[e], w_up[e], w_down[e])
        args = (x, route_idx, coef, w_gate, w_up, w_down)
        bound = batch_routes.bind(args)
        shapes = [shape for _, shape in _site_shapes(bound)]
        self.assertEqual(shapes, [(8, 256, 256)] * 3)
        code = self._check_experts(batch_routes, args, (expected,), depth=4)
        self.assertIn("jnp.clip(route_idx[_layer], 0, 7)", code)

    def test_root_level_experts_stream(self) -> None:
        """Expert weights read whole along K in a root's own tiles stream
        by id instead of staying whole in VMEM."""
        torch.manual_seed(0)
        x = torch.randn(1, 256, device=DEVICE, dtype=torch.bfloat16)
        idx = torch.tensor([[7, 0]], device=DEVICE, dtype=torch.int32)
        w_gate, _, _ = _expert_weights(8, 256, 512)
        args = (x, idx, w_gate)
        bound = root_level_experts.bind(args)
        self.assertEqual(_site_shapes(bound), [(None, (8, 256, 512))])
        expected = sum(x.float() @ w_gate[e].float() for e in idx[0])
        code = self._check_experts(root_level_experts, args, (expected,))
        self.assertIn("w_gate.at[jnp.clip(topk_idx[0, 0], 0, 7)", code)

    def test_vmem_ids_mirror_to_smem(self) -> None:
        """Read-only ids kept in VMEM (a root also reads them as a tile):
        the prologue copies them to SMEM, where the expert copies read
        them."""
        torch.manual_seed(0)
        x = torch.randn(1, 256, device=DEVICE, dtype=torch.bfloat16)
        w_gate, _, _ = _expert_weights(8, 256, 512)
        for ids in ([7, 0], [3, 3]):
            with self.subTest(ids=ids):
                idx = torch.tensor([ids], device=DEVICE, dtype=torch.int32)
                args = (x, idx, w_gate)
                expected = sum(x.float() @ w_gate[e].float() for e in ids)
                code = self._check_experts(
                    experts_and_routes_out, args, (expected, idx)
                )
                fill = code.index("index_smem[0, 1] = topk_idx[0, 1]")
                self.assertLess(fill, code.index("make_async_copy("))
                self.assertIn("w_gate.at[jnp.clip(index_smem[0, 0], 0, 7)", code)
                self.assertNotIn(".at[jnp.clip(pltpu.roll(", code)

    def test_in_kernel_topk_gates_expert_copies(self) -> None:
        """Expert ids a root writes in the kernel: their copies start only
        after it ends, the prologue's included."""
        torch.manual_seed(0)
        x = torch.randn(1, 256, device=DEVICE, dtype=torch.bfloat16)
        w_router = (torch.randn(256, 8, device=DEVICE) / 16).to(torch.bfloat16)
        w_gate, w_up, w_down = _expert_weights(8, 256, 256)
        logits = torch.sigmoid(x.float() @ w_router.float())
        weights, ids = torch.topk(logits, 2, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        ids = ids.to(torch.int32)
        expected = (
            _routed_experts_reference(x, ids, weights, w_gate, w_up, w_down),
            ids,
            weights,
        )
        args = (x, w_router, w_gate, w_up, w_down, 2)
        bound = routed_experts_in_kernel_topk.bind(args)
        shapes = [shape for _, shape in _site_shapes(bound)]
        self.assertEqual(shapes, [(8, 256, 256)] * 3)
        for depth in (2, 5):
            with self.subTest(depth=depth):
                code = self._check_experts(
                    routed_experts_in_kernel_topk, args, expected, depth=depth
                )
                # The copies read the ids from an SMEM mirror that the
                # top-k's root fills when it ends.
                fill = code.index("index_smem[0, 1] = sel_idx[0, 1]")
                self.assertGreater(fill, code.rindex("sel_idx[pl.ds("))
                first_copy = code.index("w_gate.at[jnp.clip(index_smem[0, 0], 0, 7)")
                self.assertGreater(first_copy, fill)
                self.assertNotIn(".at[jnp.clip(pltpu.roll(", code)

    def test_layer_stack_gates_next_layer_copies(self) -> None:
        """Folded layers that each write the ids their experts read: a copy
        never reads the ids before this layer's producer writes them."""
        torch.manual_seed(1)
        layers, experts = 3, 8
        x = torch.randn(1, 256, device=DEVICE, dtype=torch.bfloat16)
        w_router = (torch.randn(layers, 256, experts, device=DEVICE) / 4).to(
            torch.bfloat16
        )
        w_gate, w_up, w_down = _expert_weights(experts, 256, 256, layers)
        expected, picks = _routed_layer_stack_reference(
            x, w_router, w_gate, w_up, w_down, 2
        )
        # Every layer picks different experts.
        self.assertEqual(len({tuple(p) for p in picks}), layers)
        args = (x, w_router, w_gate, w_up, w_down, 2)
        for depth in (2, 4):
            with self.subTest(depth=depth):
                code = self._check_experts(
                    routed_layer_stack, args, (expected,), depth=depth
                )
                self.assertEqual(_ring_depths(code), [depth])
                # The deferred copies start in the layer, after the top-k,
                # which refills the ids' SMEM mirror in each layer.
                self.assertIn(f"ring_sem.at[(0 + i * 6) % {depth}]", code)
                self.assertIn(f" >= {depth})", code)
                self.assertIn("\n        index_smem[0, 1] = sel_idx[0, 1]\n", code)
                self.assertIn("w_gate.at[i, jnp.clip(index_smem[0, 0], 0, 7)", code)

    def test_gated_copies_take_their_own_arena(self) -> None:
        """Expert copies that wait for the layer's top-k stream through an
        arena of their own, so the dense weights' arena runs ahead through
        the router and the top-k."""
        torch.manual_seed(3)
        layers, experts = 3, 8
        x = torch.randn(1, 256, device=DEVICE, dtype=torch.bfloat16)
        w_proj = (torch.randn(layers, 256, 256, device=DEVICE) / 16).to(torch.bfloat16)
        w_router = (torch.randn(layers, 256, experts, device=DEVICE) / 4).to(
            torch.bfloat16
        )
        w_gate, w_up, w_down = _expert_weights(experts, 256, 256, layers)
        expected, picks = _projected_layer_stack_reference(
            x, w_proj, w_router, w_gate, w_up, w_down, 2
        )
        self.assertEqual(len({tuple(p) for p in picks}), layers)
        args = (x, w_proj, w_router, w_gate, w_up, w_down, 2)
        code = self._check_experts(projected_layer_stack, args, (expected,))
        proj = re.search(r"w_proj_tile = _helion_ring_tile\((ring(_\d+)?),", code)
        gate = re.search(r"w_gate_tile = _helion_ring_tile\((ring(_\d+)?),", code)
        down = re.search(r"w_down_tile = _helion_ring_tile\((ring(_\d+)?),", code)
        assert proj is not None and gate is not None and down is not None
        self.assertNotEqual(proj.group(1), gate.group(1))
        self.assertEqual(gate.group(1), down.group(1))

    @_per_weight_rings()
    def test_multi_tile_producer_gates_catch_up(self) -> None:
        """The deferred copies start after the last tile of a producer of
        several tiles, including those an earlier root's refills skip (in a
        ring shared by shape: an arena gives the gated copies their own)."""
        torch.manual_seed(2)
        steps, experts = 16, 8
        x = torch.randn(1, 256, device=DEVICE, dtype=torch.bfloat16)
        w_in = (torch.randn(256, 256, device=DEVICE) / 16).to(torch.bfloat16)
        w_gate, _, _ = _expert_weights(experts, 256, 256)
        hidden = (x.float() @ w_in.float()).to(x.dtype)
        # Routes too large for an SMEM mirror, then small enough for one,
        # which the producer's last tile fills.
        for width, route in ((128, "route[15, 0]"), (8, "index_smem[15, 0]")):
            with self.subTest(width=width):
                ids = torch.randint(
                    0, experts, (steps, width), device=DEVICE, dtype=torch.int32
                )
                ids[:, 0] = torch.tensor(
                    [3, 7, 7, 0, 6, 2, 5, 1, 4, 7, 6, 3, 0, 2, 5, 6],
                    dtype=torch.int32,
                )
                expected = sum(
                    hidden.float() @ w_gate[(e + 1) % experts].float()
                    for e in ids[:, 0].tolist()
                )
                args = (x, ids, w_in, w_gate)
                code = self._check_experts(
                    remapped_routes,
                    args,
                    (expected,),
                    depth=3,
                    block_sizes=[16, 256, 256, 8, 16, 256, 256],
                )
                self.assertRegex(
                    code, r"@pl\.when\(pid_shared_\d+ == 1\)\n\s*def _catch_up"
                )
                self.assertIn(f"w_gate.at[jnp.clip({route}, 0, 7)", code)
                # w_in's refill would start the third step's copy: the gate
                # skips it.
                self.assertIn(" >= 4)", code)
                self.assertEqual("def _mirror" in code, width == 8)

    def assertStackMatches(
        self,
        out: torch.Tensor,
        expected: torch.Tensor,
        states: dict[str, torch.Tensor],
        expected_states: dict[str, torch.Tensor],
    ) -> None:
        # bf16 rounding differences compound over the layers.
        torch.testing.assert_close(out, expected, rtol=5e-2, atol=1e-1)
        for name, state in states.items():
            torch.testing.assert_close(
                state, expected_states[name], rtol=2e-2, atol=5e-2
            )

    def test_gemma3_stack_example(self) -> None:
        """Local and global layers in a static pattern, picked by a constant
        ``if`` on the ``hl.static_range`` index inside the attention root,
        fold into one device loop over periods, with the tanh GeGLU lowered
        in f32."""
        mod = import_path(EXAMPLES_DIR / "gemma3_decode_stack.py")
        torch.manual_seed(0)
        period, window = 2, 64
        hidden, w, s, pos = mod.make_gemma3_stack_inputs(
            layers=4,
            dim=256,
            intermediate=384,
            head_dim=128,
            context=256,
            position=150,
            device=DEVICE,
        )
        s_ref = {k: v.clone() for k, v in s.items()}
        expected = mod.gemma3_decode_stack_ref(hidden, w, s_ref, pos, period, window)
        code, out = code_and_output(
            mod.gemma3_decode_stack,
            mod.gemma3_stack_args(hidden, w, s, pos, period, window),
        )
        self.assertStackMatches(out, expected, s, s_ref)
        self.assertOneSequentialProgram(code)
        self.assertIn("@pl.loop(0, 2)", code)
        self.assertIn("lax.tanh(", code)

    def test_gemma3_step_example(self) -> None:
        """The Gemma-3 stack between a scaled embedding row and a
        soft-capped top-1 over the LM head."""
        mod = import_path(EXAMPLES_DIR / "gemma3_decode_step.py")
        torch.manual_seed(0)
        period, window, vocab = 2, 64, 1000
        inputs = mod.make_decode_step_inputs(
            layers=2,
            dim=256,
            embedding_rows=4000,
            padded_vocab=1024,
            vocab=vocab,
            intermediate=384,
            head_dim=128,
            context=256,
            position=150,
            device=DEVICE,
        )
        token, embedding, w, s, pos, final_norm, lm_head = inputs
        s_ref = {k: v.clone() for k, v in s.items()}
        logits = mod.gemma3_decode_step_logits(
            token, embedding, w, s_ref, pos, final_norm, lm_head, period, window, vocab
        )
        code, (top_idx, top_val) = code_and_output(
            mod.gemma3_decode_step,
            mod.decode_step_args(
                token, embedding, w, s, pos, final_norm, lm_head, period, window, vocab
            ),
        )
        self.assertTop1(top_idx, top_val, logits, tol=1e-1)
        for name in mod.STATE_NAMES:
            torch.testing.assert_close(s[name], s_ref[name], rtol=2e-2, atol=5e-2)
        self.assertOneSequentialProgram(code)
        self.assertIn("embedding_row_copy.start()", code)

    def test_mla_stack_example(self) -> None:
        """Multi-head latent attention with absorbed weights: batched dots of
        per-head rows with stacked 4-D weights, and two latent caches
        written at ``pos``."""
        mod = import_path(EXAMPLES_DIR / "mla_decode_stack.py")
        torch.manual_seed(0)
        hidden, w, s, pos = mod.make_mla_stack_inputs(
            layers=2,
            dim=256,
            intermediate=384,
            heads=4,
            latent_dim=256,
            context=256,
            position=150,
            device=DEVICE,
        )
        s_ref = {k: v.clone() for k, v in s.items()}
        expected = mod.mla_decode_stack_ref(hidden, w, s_ref, pos)
        code, out = code_and_output(
            mod.mla_decode_stack, mod.mla_stack_args(hidden, w, s, pos)
        )
        self.assertStackMatches(out, expected, s, s_ref)
        self.assertOneSequentialProgram(code)

    def test_parallel_residual_stack_example(self) -> None:
        """Attention and the MLP read one LayerNorm and add into the residual
        from one root with two reduction loops."""
        mod = import_path(EXAMPLES_DIR / "parallel_residual_stack.py")
        torch.manual_seed(0)
        hidden, w, s, pos = mod.make_parallel_stack_inputs(
            layers=2,
            dim=256,
            intermediate=384,
            head_dim=128,
            context=256,
            position=150,
            device=DEVICE,
        )
        s_ref = {k: v.clone() for k, v in s.items()}
        expected = mod.parallel_residual_stack_ref(hidden, w, s_ref, pos)
        code, out = code_and_output(
            mod.parallel_residual_stack, mod.parallel_stack_args(hidden, w, s, pos)
        )
        self.assertStackMatches(out, expected, s, s_ref)
        self.assertOneSequentialProgram(code)
        self.assertIn("@pl.loop(0, 2)", code)
