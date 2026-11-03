"""The flattened-rows attention form in the CuTe flash matcher.

``examples/blackwell_attention.py`` tiles ``B*H*M`` query rows on one root
axis, reads K/V at ``tile_n.index + tile_m.begin // M * N`` from 2-D
``[rows, D]`` tensors, applies a runtime softmax scale after the row max
(``maximum(m, amax(qk) * s)``, ``qk * s - m``), carries ``(m, acc, l)`` and
rescales the accumulator with a packed ``mul.f32x2`` inline asm.  Each of
those is a relaxation of the matcher (``attention_match``) and of the output
plan (``cute_flash``); each has a positive and a negative pin here, as do the
other admitted runtime-scale forms (the raw argument, the ``exp`` base, the
scale applied before the row max) and the chain's indifference to the loop's
invariant inputs.  CPU only.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any

import pytest
import sympy
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test.cute_population_contracts import tcgen05_mma_support

import helion
from helion._compiler.cute import cute_flash
from helion._compiler.cute.analysis import attention_match as _attention_match
from helion._compiler.cute.analysis import match_common
from helion._compiler.cute.analysis.match_common import FlattenedRowOffset
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
from helion._testing.cute_aux import as_sm100
import helion.language as hl
from helion.language.inline_asm_ops import inline_asm_elementwise

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


_MUL_F32X2 = """
    {
        .reg .b64 ra, rb, rc;
        mov.b64 ra, { $2, $3 };
        mov.b64 rb, { $4, $5 };
        mul.f32x2 rc, ra, rb;
        mov.b64 { $0, $1 }, rc;
    }
"""
_ADD_F32X2 = _MUL_F32X2.replace("mul.f32x2", "add.f32x2")
_PACKED_CONSTRAINTS = "=r,=r,r,r,r,r"


def _packed(asm: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        asm, _PACKED_CONSTRAINTS, [a, b], dtype=torch.float32, is_pure=True, pack=2
    )


# ---------------------------------------------------------------------------
# Kernels.
# ---------------------------------------------------------------------------
@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def flattened_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, qk_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The example's form without its subtiling: a runtime scale applied
    after the row max, the statistics carried as ``(m, acc, l)``."""
    B, H, M, D = q_in.shape
    N = k_in.shape[2]
    D = hl.specialize(D)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, D)
    MM = q.shape[0]
    o = q.new_empty(MM, D)
    lse = q.new_empty(MM, dtype=torch.float32)
    block_m = hl.register_block_size(M)
    block_n = hl.register_block_size(N)
    qk_scale = qk_scale * 1.44269504
    for tile_m in hl.tile(MM, block_size=block_m):
        m_i = hl.zeros([tile_m]) - float("inf")
        acc = hl.zeros([tile_m, D])
        l_i = hl.zeros([tile_m]) + 1.0
        q_i = q[tile_m, :]
        start_n = tile_m.begin // M * N
        for tile_n in hl.tile(N, block_size=block_n):
            k_j = k[tile_n + start_n, :]
            v_j = v[tile_n + start_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp2(qk)
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            acc = acc * alpha[:, None]
            acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    return o.reshape(B, H, M, D), lse.reshape(B, H, M)


@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def flattened_attention_raw_scale(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, qk_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The deferred form over the raw argument (no host multiply): the
    argument itself is the exp2-domain scale."""
    B, H, M, D = q_in.shape
    N = k_in.shape[2]
    D = hl.specialize(D)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, D)
    MM = q.shape[0]
    o = q.new_empty(MM, D)
    lse = q.new_empty(MM, dtype=torch.float32)
    for tile_m in hl.tile(MM):
        m_i = hl.zeros([tile_m]) - float("inf")
        acc = hl.zeros([tile_m, D])
        l_i = hl.zeros([tile_m]) + 1.0
        q_i = q[tile_m, :]
        start_n = tile_m.begin // M * N
        for tile_n in hl.tile(N):
            k_j = k[tile_n + start_n, :]
            v_j = v[tile_n + start_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp2(qk)
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            acc = acc * alpha[:, None]
            acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    return o.reshape(B, H, M, D), lse.reshape(B, H, M)


@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def flattened_attention_exp_base(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, qk_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The deferred form in the ``exp`` base with a natural-log LSE: the
    body's exp2 multiplier is ``log2(e)``, the LSE is rescaled by ``ln 2``."""
    B, H, M, D = q_in.shape
    N = k_in.shape[2]
    D = hl.specialize(D)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, D)
    MM = q.shape[0]
    o = q.new_empty(MM, D)
    lse = q.new_empty(MM, dtype=torch.float32)
    for tile_m in hl.tile(MM):
        m_i = hl.zeros([tile_m]) - float("inf")
        acc = hl.zeros([tile_m, D])
        l_i = hl.zeros([tile_m]) + 1.0
        q_i = q[tile_m, :]
        start_n = tile_m.begin // M * N
        for tile_n in hl.tile(N):
            k_j = k[tile_n + start_n, :]
            v_j = v[tile_n + start_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp(qk)
            alpha = torch.exp(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            acc = acc * alpha[:, None]
            acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    return o.reshape(B, H, M, D), lse.reshape(B, H, M)


@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def flattened_attention_scaled_scores(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, qk_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The runtime scale applied to the scores before the row max
    (``maximum(m, amax(qk * s))``): the post-dot form without the deferral."""
    B, H, M, D = q_in.shape
    N = k_in.shape[2]
    D = hl.specialize(D)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, D)
    MM = q.shape[0]
    o = q.new_empty(MM, D)
    lse = q.new_empty(MM, dtype=torch.float32)
    qk_scale = qk_scale * 1.44269504
    for tile_m in hl.tile(MM):
        m_i = hl.zeros([tile_m]) - float("inf")
        acc = hl.zeros([tile_m, D])
        l_i = hl.zeros([tile_m]) + 1.0
        q_i = q[tile_m, :]
        start_n = tile_m.begin // M * N
        for tile_n in hl.tile(N):
            k_j = k[tile_n + start_n, :]
            v_j = v[tile_n + start_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32) * qk_scale
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, None]
            p = torch.exp2(qk)
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            acc = acc * alpha[:, None]
            acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    return o.reshape(B, H, M, D), lse.reshape(B, H, M)


@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def flattened_attention_row_bias(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    bias: torch.Tensor,
    qk_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``flattened_attention_scaled_scores`` plus a per-row bias read
    outside the KV loop and added inside it: a loop-invariant input of the
    loop body beside Q and the three carried statistics."""
    B, H, M, D = q_in.shape
    N = k_in.shape[2]
    D = hl.specialize(D)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, D)
    MM = q.shape[0]
    o = q.new_empty(MM, D)
    lse = q.new_empty(MM, dtype=torch.float32)
    qk_scale = qk_scale * 1.44269504
    for tile_m in hl.tile(MM):
        m_i = hl.zeros([tile_m]) - float("inf")
        acc = hl.zeros([tile_m, D])
        l_i = hl.zeros([tile_m]) + 1.0
        q_i = q[tile_m, :]
        row_bias = bias[tile_m]
        start_n = tile_m.begin // M * N
        for tile_n in hl.tile(N):
            k_j = k[tile_n + start_n, :]
            v_j = v[tile_n + start_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32) * qk_scale
            qk = qk + row_bias[:, None]
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, None]
            p = torch.exp2(qk)
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            acc = acc * alpha[:, None]
            acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    return o.reshape(B, H, M, D), lse.reshape(B, H, M)


@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def flattened_attention_literal_scale(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The deferred form with the standard literal scale (``log2(e) /
    sqrt(64)``) and the ``(m, l, acc)`` carried order."""
    B, H, M, D = q_in.shape
    N = k_in.shape[2]
    D = hl.specialize(D)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, D)
    MM = q.shape[0]
    o = q.new_empty(MM, D)
    lse = q.new_empty(MM, dtype=torch.float32)
    qk_scale = 0.18033688
    for tile_m in hl.tile(MM):
        m_i = hl.zeros([tile_m]) - float("inf")
        l_i = hl.zeros([tile_m]) + 1.0
        acc = hl.zeros([tile_m, D])
        q_i = q[tile_m, :]
        start_n = tile_m.begin // M * N
        for tile_n in hl.tile(N):
            k_j = k[tile_n + start_n, :]
            v_j = v[tile_n + start_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp2(qk)
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            acc = acc * alpha[:, None]
            acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    return o.reshape(B, H, M, D), lse.reshape(B, H, M)


@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def flattened_attention_negative_scale(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The deferred form with a negative literal scale: ``amax(qk) * s`` is
    no longer the scaled scores' row max."""
    B, H, M, D = q_in.shape
    N = k_in.shape[2]
    D = hl.specialize(D)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, D)
    MM = q.shape[0]
    o = q.new_empty(MM, D)
    lse = q.new_empty(MM, dtype=torch.float32)
    qk_scale = -0.18033688
    for tile_m in hl.tile(MM):
        m_i = hl.zeros([tile_m]) - float("inf")
        l_i = hl.zeros([tile_m]) + 1.0
        acc = hl.zeros([tile_m, D])
        q_i = q[tile_m, :]
        start_n = tile_m.begin // M * N
        for tile_n in hl.tile(N):
            k_j = k[tile_n + start_n, :]
            v_j = v[tile_n + start_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp2(qk)
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            acc = acc * alpha[:, None]
            acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    return o.reshape(B, H, M, D), lse.reshape(B, H, M)


@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def flattened_attention_packed_add(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, qk_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The example's packed rescale with ``add.f32x2`` instead of
    ``mul.f32x2``: not the accumulator rescale."""
    B, H, M, D = q_in.shape
    N = k_in.shape[2]
    D = hl.specialize(D)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, D)
    MM = q.shape[0]
    o = q.new_empty(MM, D)
    lse = q.new_empty(MM, dtype=torch.float32)
    qk_scale = qk_scale * 1.44269504
    for tile_m in hl.tile(MM):
        m_i = hl.zeros([tile_m]) - float("inf")
        acc = hl.zeros([tile_m, D])
        l_i = hl.zeros([tile_m]) + 1.0
        q_i = q[tile_m, :]
        start_n = tile_m.begin // M * N
        for tile_n in hl.tile(N):
            k_j = k[tile_n + start_n, :]
            v_j = v[tile_n + start_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp2(qk)
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            acc = _packed(_ADD_F32X2, acc, alpha[:, None])
            acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    return o.reshape(B, H, M, D), lse.reshape(B, H, M)


@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def flattened_attention_half_groups(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, qk_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """A K/V offset over half-head groups (``tile_m.begin // (M // 2) * N``):
    a valid offset form whose query length is not the KV length."""
    B, H, M, D = q_in.shape
    N = k_in.shape[2]
    D = hl.specialize(D)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, D)
    MM = q.shape[0]
    half = M // 2
    o = q.new_empty(MM, D)
    lse = q.new_empty(MM, dtype=torch.float32)
    qk_scale = qk_scale * 1.44269504
    for tile_m in hl.tile(MM):
        m_i = hl.zeros([tile_m]) - float("inf")
        acc = hl.zeros([tile_m, D])
        l_i = hl.zeros([tile_m]) + 1.0
        q_i = q[tile_m, :]
        start_n = tile_m.begin // half * N
        for tile_n in hl.tile(N):
            k_j = k[tile_n + start_n, :]
            v_j = v[tile_n + start_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp2(qk)
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            acc = acc * alpha[:, None]
            acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    return o.reshape(B, H, M, D), lse.reshape(B, H, M)


@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def single_head_rows_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qk_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """One root axis over the rows of one head, K/V read without an offset:
    not the flattened form (no batch head is named)."""
    M, D = q.shape
    N = k.shape[0]
    D = hl.specialize(D)
    o = q.new_empty(M, D)
    lse = q.new_empty(M, dtype=torch.float32)
    qk_scale = qk_scale * 1.44269504
    for tile_m in hl.tile(M):
        m_i = hl.zeros([tile_m]) - float("inf")
        acc = hl.zeros([tile_m, D])
        l_i = hl.zeros([tile_m]) + 1.0
        q_i = q[tile_m, :]
        for tile_n in hl.tile(N):
            k_j = k[tile_n, :]
            v_j = v[tile_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp2(qk)
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            acc = acc * alpha[:, None]
            acc = hl.dot(p.to(v.dtype), v_j, acc=acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    return o, lse


@helion.kernel(static_shapes=True, backend="cute", autotune_accuracy_check=False)
def rows_attention_runtime_scale_causal(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qk_scale: float
) -> torch.Tensor:
    """The 2-D ``q[bh, tile_m, :]`` form with a runtime scale and a causal
    mask: a runtime scale admits no modifier."""
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = hl.specialize(q.size(2))
    out = torch.empty_like(q)
    qk_scale = qk_scale * 1.44269504
    for bh, tile_m in hl.tile([batch_heads, seq_len], block_size=[1, None]):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q_tile = q[bh.begin, tile_m, :]
        for tile_n in hl.tile(seq_len):
            k_tile = k[bh.begin, tile_n, :]
            qk = hl.dot(q_tile, k_tile.transpose(0, 1)) * qk_scale
            qk = torch.where(
                tile_m.index[:, None] >= tile_n.index[None, :], qk, float("-inf")
            )
            m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
            p = torch.exp2(qk - m_new[:, None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=-1)
            acc = acc * alpha[:, None]
            v_tile = v[bh.begin, tile_n, :]
            acc = hl.dot(p.to(v.dtype), v_tile, acc=acc)
            m_i = m_new
        out[bh.begin, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _bind(kernel: Kernel, args: tuple[Any, ...]) -> BoundKernel:
    with as_sm100():
        return kernel._bind_isolated(args)


def _qkv(
    shape: tuple[int, int, int, int] = (1, 4, 256, 64),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.empty(shape, dtype=torch.bfloat16) for _ in range(3))  # type: ignore[return-value]


def _kv_loop_graph(bound: BoundKernel) -> ForLoopGraphInfo:
    device_ir = bound.host_function.device_ir
    root = {bid for ids in device_ir.grid_block_ids for bid in ids}
    graphs = [
        graph_info
        for graph_info in device_ir.graphs
        if isinstance(graph_info, ForLoopGraphInfo)
        and len(graph_info.block_ids) == 1
        and not (set(graph_info.block_ids) & root)
    ]
    assert len(graphs) == 1, [type(g).__name__ for g in device_ir.graphs]
    return graphs[0]


def _pattern(bound: BoundKernel) -> _attention_match.AttentionSoftmaxPattern | None:
    graph_info = _kv_loop_graph(bound)
    with bound.env, bound.host_function:
        return _attention_match._attention_softmax_pattern_head_dim(
            graph_info.graph, kv_block_id=graph_info.block_ids[0]
        )


def _surface(bound: BoundKernel) -> _attention_match.FlashSearchSurface | None:
    with as_sm100(), bound.env, bound.host_function:
        return _attention_match.detect_flash_search_surface(
            bound.host_function.device_ir
        )


def _output_plan(bound: BoundKernel) -> cute_flash.FlashGraphOutputPlan | None:
    pattern = _pattern(bound)
    assert pattern is not None
    graph_info = _kv_loop_graph(bound)
    device_ir = bound.host_function.device_ir
    with bound.env, bound.host_function:
        return cute_flash._flash_graph_output_plan_from_graphs(
            device_ir.graphs,
            root_block_ids=device_ir.grid_block_ids[0],
            kv_block_id=graph_info.block_ids[0],
            score_plan=pattern.score_plan,
        )


def _example_kernel() -> Kernel:
    return import_path(
        EXAMPLES_DIR / "blackwell_attention.py"
    ).blackwell_attention_kernel


# ---------------------------------------------------------------------------
# The matcher.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("head_dim", [64, 128])
def test_runtime_scale_flattened_form_matches(head_dim: int) -> None:
    bound = _bind(flattened_attention, (*_qkv((1, 4, 256, head_dim)), 0.125))
    pattern = _pattern(bound)
    assert pattern is not None
    assert pattern.score_rank == 2
    assert pattern.flattened == FlattenedRowOffset(0, 256, 256)
    assert pattern.batch_block_id == 0
    assert pattern.head_dim == head_dim
    plan = pattern.score_plan
    # The traced ``qk_scale * 1.44269504`` is the exp2-domain scale itself.
    assert isinstance(plan.runtime_scale, sympy.Expr)
    assert plan.qk_scale_log2 == 1.0
    assert plan.lse_scale == 1.0
    assert not plan.modifiers
    assert not cute_flash._standard_dense_score_plan_supported(plan)


@pytest.mark.parametrize(
    ("kernel", "raw", "qk_scale_log2", "lse_scale", "lse_log_base"),
    [
        pytest.param(
            flattened_attention_raw_scale, True, 1.0, 1.0, "log2", id="raw_scale"
        ),
        pytest.param(
            flattened_attention_exp_base,
            True,
            math.log2(math.e),
            math.log(2.0),
            "log",
            id="exp_base",
        ),
        pytest.param(
            flattened_attention_scaled_scores,
            False,
            1.0,
            1.0,
            "log2",
            id="scaled_scores",
        ),
    ],
)
def test_other_runtime_scale_forms_match(
    kernel: Kernel,
    raw: bool,
    qk_scale_log2: float,
    lse_scale: float,
    lse_log_base: str,
) -> None:
    bound = _bind(kernel, (*_qkv(), 0.125))
    pattern = _pattern(bound)
    assert pattern is not None
    assert pattern.flattened == FlattenedRowOffset(0, 256, 256)
    plan = pattern.score_plan
    # The raw argument is the scale symbol itself; the host multiply is a
    # product over it.
    assert isinstance(plan.runtime_scale, sympy.Expr)
    assert isinstance(plan.runtime_scale, sympy.Symbol) == raw
    assert math.isclose(plan.qk_scale_log2, qk_scale_log2, rel_tol=1e-6)
    assert math.isclose(plan.lse_scale, lse_scale, rel_tol=1e-6)
    assert not plan.modifiers
    assert _surface(bound) is not None
    output_plan = _output_plan(bound)
    assert output_plan is not None
    assert output_plan.flattened_rows
    assert output_plan.lse_log_base == lse_log_base


def test_example_forward_matches_with_the_packed_rescale() -> None:
    bound = _bind(_example_kernel(), (*_qkv(), 0.125))
    pattern = _pattern(bound)
    assert pattern is not None
    assert pattern.flattened == FlattenedRowOffset(0, 256, 256)
    assert pattern.score_plan.runtime_scale is not None


def test_literal_deferred_scale_is_the_standard_scale() -> None:
    bound = _bind(flattened_attention_literal_scale, _qkv())
    pattern = _pattern(bound)
    assert pattern is not None
    plan = pattern.score_plan
    assert plan.runtime_scale is None
    assert math.isclose(plan.qk_scale_log2, math.log2(math.e) / 8.0, rel_tol=1e-6)
    assert cute_flash._standard_dense_score_plan_supported(plan)


def test_negative_deferred_scale_is_declined() -> None:
    assert _pattern(_bind(flattened_attention_negative_scale, _qkv())) is None


def test_packed_asm_other_than_the_multiply_is_declined() -> None:
    assert _pattern(_bind(flattened_attention_packed_add, (*_qkv(), 0.125))) is None


def test_rows_without_an_offset_are_not_the_flattened_form() -> None:
    q, k, v = (torch.empty((256, 64), dtype=torch.bfloat16) for _ in range(3))
    bound = _bind(single_head_rows_attention, (q, k, v, 0.125))
    assert _pattern(bound) is None
    assert _surface(bound) is None


def test_runtime_scale_admits_no_modifier() -> None:
    q, k, v = (torch.empty((4, 256, 64), dtype=torch.bfloat16) for _ in range(3))
    assert (
        _pattern(_bind(rows_attention_runtime_scale_causal, (q, k, v, 0.125))) is None
    )


def test_online_softmax_chain_ignores_loop_invariant_inputs() -> None:
    bias = torch.empty(1024, dtype=torch.float32)
    bound = _bind(flattened_attention_row_bias, (*_qkv(), bias, 0.125))
    graph = _kv_loop_graph(bound).graph
    # Q, the row bias and the three statistics: the bias is read, never
    # written, so the loop carries three of its four inputs after Q.
    assert sum(node.op == "placeholder" for node in graph.nodes) == 5
    (centered,) = (
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten.sub.Tensor
        and node.meta["val"].ndim == 2
    )
    (pv_node,) = (
        node
        for node in graph.nodes
        if node.op == "call_function"
        and _attention_match._attention_pv_operands(node) is not None
    )
    score_node = centered.args[0]
    assert isinstance(score_node, torch.fx.Node)
    with bound.env, bound.host_function:
        assert _attention_match._attention_exact_online_softmax_chain(
            score_node, pv_node
        )


# ---------------------------------------------------------------------------
# The surface and the output plan.
# ---------------------------------------------------------------------------
def test_surface_is_the_dense_lse_surface_over_one_root_axis() -> None:
    bound = _bind(flattened_attention, (*_qkv((2, 8, 1024, 64)), 0.125))
    surface = _surface(bound)
    assert surface is not None
    assert (surface.head_dim, surface.num_kv, surface.num_bh) == (64, 8, 16)
    assert surface.block_size_targets == {0: 128, 1: 128}
    assert surface.dense_lse_output
    assert not surface.standard_dense_output
    assert surface.tensor_4d_heads == 8
    assert not surface.has_row_epilogue and surface.plain_row_body
    assert bound.config_spec.cute_flash_search_enabled
    assert [s.structural_pin for s in bound.config_spec.block_sizes] == [128, 128]


def test_half_group_offset_is_not_square_attention() -> None:
    bound = _bind(flattened_attention_half_groups, (*_qkv(), 0.125))
    pattern = _pattern(bound)
    assert pattern is not None
    assert pattern.flattened == FlattenedRowOffset(0, 128, 256)
    assert _surface(bound) is None
    assert _output_plan(bound) is None


def test_output_plan_reads_the_2d_operands_and_the_1d_lse() -> None:
    bound = _bind(flattened_attention, (*_qkv((2, 8, 1024, 64)), 0.125))
    plan = _output_plan(bound)
    assert plan is not None
    assert plan.flattened_rows
    assert (plan.q_name, plan.k_name, plan.v_name, plan.o_name) == (
        "q",
        "k",
        "v",
        "o",
    )
    assert (plan.batch, plan.seq, plan.head_dim) == (16, 1024, 64)
    assert (plan.lse_name, plan.lse_log_base) == ("lse", "log2")
    assert plan.output_epilogue == "identity"
    assert not plan.v_k_major and not plan.v_storage_k_major


@pytest.mark.cpu_render(
    mma_support=tcgen05_mma_support(),
    fake_device_sm_count=148,
    reduction_smem_budget_bytes=232448,
)
def test_example_default_config_renders_the_fused_body() -> None:
    # Bound in isolation: the example module is shared with the SIMT render
    # pin of ``test_cute_staged_matmul_reductions`` (``HELION_CUTE_FLASH=0``),
    # whose bind of the same shapes must not be read from, nor published
    # to, the kernel's bound-kernel cache.
    kernel = _example_kernel()
    bound = kernel._bind_isolated((*_qkv(), 0.125))
    config = bound.config_spec.default_config()
    assert config.config["block_sizes"] == [128, 128]
    assert config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4"
    code = bound.to_code(config)
    assert "'kind': 'helion_flash'" in code
    # The runtime scale reaches the body as a kernel scalar; the host passes
    # the traced ``qk_scale * 1.44269504``.
    assert "_flash_scale" in code
    assert "qk_scale = qk_scale * 1.44269504" in code
    # The fused body takes the row max of the raw scores and scales it: the
    # host function asserts the scale positive before the launch.
    assert "assert qk_scale > 0" in code
    assert "pack=2" not in code


# ---------------------------------------------------------------------------
# The primitives.
# ---------------------------------------------------------------------------
def _packed_call(asm: str, constraints: str = _PACKED_CONSTRAINTS, **kwargs: Any):
    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    args = [asm, constraints, [a, b], torch.float32, True, 2]
    for index, key in enumerate(
        ("asm", "constraints", "operands", "dtype", "pure", "pack")
    ):
        if key in kwargs:
            args[index] = kwargs[key]
    return a, b, graph.call_function(inline_asm_elementwise, tuple(args))


def test_packed_f32x2_multiply_reads_the_idiom_only() -> None:
    a, b, node = _packed_call(_MUL_F32X2)
    assert match_common.packed_f32x2_multiply(node) == (a, b)
    assert match_common.packed_f32x2_multiply(_packed_call(_ADD_F32X2)[2]) is None
    assert (
        match_common.packed_f32x2_multiply(_packed_call(_MUL_F32X2, pack=1)[2]) is None
    )
    assert (
        match_common.packed_f32x2_multiply(_packed_call(_MUL_F32X2, pure=False)[2])
        is None
    )
    assert (
        match_common.packed_f32x2_multiply(_packed_call(_MUL_F32X2, "=r,r,r")[2])
        is None
    )
    assert (
        match_common.packed_f32x2_multiply(
            _packed_call(_MUL_F32X2, dtype=torch.float16)[2]
        )
        is None
    )
