"""Helion kernels for the attention-gym variant suite.

Output-only flash-attention kernels in the canonical ``examples/attention.py``
envelope (collapsed [B*H, S, D] views, single static KV loop, online softmax in
exp2 space) so the CuTe flash path can pattern-match them. Each variant's
score/mask modifier is a pure module-level "stage" function passed to a shared
kernel template as an argument (Helion inlines callable args at trace time;
kernel functions themselves cannot be closures).

Semantics match FlexAttention exactly: softmax(mask(score_mod(q@k*sm_scale)))
in natural-log space. The online softmax runs in base-2 space, so
scale-equivariant modifiers (linear biases, softcap) use constants
pre-multiplied by log2(e); the non-equivariant sigmoid-act modifier uses the
natural-space template that rescales after the modifier.

Row-max accumulators initialize to a finite ``-1e30`` sentinel instead of
-inf: masks like document/natten/flamingo leave some (row, kv-tile) pairs
fully masked, and ``-inf - -inf = nan`` would poison the online softmax when
the full KV loop visits them. Rows with no visible KV produce 0, matching
FlexAttention.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

import helion
import helion.language as hl

LOG2E = 1.44269504
NEG_SENTINEL = -1e30


def _shared_baseline(*args: object) -> torch.Tensor:
    """True-output baseline for autotune accuracy checks (any variant)."""
    from attngym_variants import VARIANTS  # pyrefly: ignore [missing-import]
    from attngym_variants import reference_output  # pyrefly: ignore [missing-import]

    stage = next(a for a in args if callable(a))
    name = stage.__variant__  # type: ignore[attr-defined]
    q, k, v = args[0], args[1], args[2]
    variant = VARIANTS[name]
    score_mod, mask_mod = variant.make_mods(q.device)  # type: ignore[union-attr]
    return reference_output(variant, q, k, v, score_mod, mask_mod)  # type: ignore[arg-type]


_BASELINE_KWARGS: dict[str, object] = {
    "static_shapes": True,
    "autotune_baseline_fn": _shared_baseline,
    "autotune_baseline_atol": 5e-2,
    "autotune_baseline_rtol": 2e-2,
}


@helion.kernel(**_BASELINE_KWARGS)  # pyrefly: ignore [no-matching-overload]
def attn_stage0(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    stage: Callable[..., torch.Tensor],
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], -1e30, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = stage(
                qk,
                tile_b.index[:, None, None],
                tile_m.index[None, :, None],
                tile_n.index[None, None, :],
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(**_BASELINE_KWARGS)  # pyrefly: ignore [no-matching-overload]
def attn_stage0_nat(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    stage: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Natural-space template: stage sees q@k*sm_scale, rescaled afterwards."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], -1e30, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * sm_scale, kt.transpose(1, 2), torch.float32)
            qk = stage(
                qk,
                tile_b.index[:, None, None],
                tile_m.index[None, :, None],
                tile_n.index[None, None, :],
            )
            qk = qk * 1.44269504
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(**_BASELINE_KWARGS)  # pyrefly: ignore [no-matching-overload]
def attn_stage0_gqa(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    stage: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """GQA template: K/V have fewer heads; kv batch = q batch // group."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    heads_q = hl.specialize(q_in.size(1))
    heads_kv = hl.specialize(k_in.size(1))
    group = heads_q // heads_kv
    assert heads_kv * group == heads_q
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], -1e30, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kv_b = tile_b.index // group
            kt = k_view[kv_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = stage(
                qk,
                tile_b.index[:, None, None],
                tile_m.index[None, :, None],
                tile_n.index[None, None, :],
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[kv_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(**_BASELINE_KWARGS)  # pyrefly: ignore [no-matching-overload]
def attn_stage1(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    t0: torch.Tensor,
    stage: Callable[..., torch.Tensor],
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], -1e30, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = stage(
                qk,
                tile_b.index[:, None, None],
                tile_m.index[None, :, None],
                tile_n.index[None, None, :],
                t0,
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(**_BASELINE_KWARGS)  # pyrefly: ignore [no-matching-overload]
def attn_stage2(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    stage: Callable[..., torch.Tensor],
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], -1e30, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = stage(
                qk,
                tile_b.index[:, None, None],
                tile_m.index[None, :, None],
                tile_n.index[None, None, :],
                t0,
                t1,
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(**_BASELINE_KWARGS)  # pyrefly: ignore [no-matching-overload]
def attn_stage3(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    stage: Callable[..., torch.Tensor],
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], -1e30, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = stage(
                qk,
                tile_b.index[:, None, None],
                tile_m.index[None, :, None],
                tile_n.index[None, None, :],
                t0,
                t1,
                t2,
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


# ---------------------------------------------------------------------------
# Stage functions (pure, module-level, constants baked as literals).
# Broadcast shapes: qk [b, m, n], b_idx [b, 1, 1], q_idx [1, m, 1], kv_idx [1, 1, n].
# All linear-bias / softcap constants are pre-multiplied by log2(e) because
# qk arrives in base-2 space (except the *_nat template).
# ---------------------------------------------------------------------------


def stage_alibi(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    slopes2: torch.Tensor,
) -> torch.Tensor:
    qk = qk + (kv_idx - q_idx) * hl.load(slopes2, [b_idx % 16])
    return torch.where(q_idx >= kv_idx, qk, float("-inf"))


def stage_softcap30(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    qk = 43.2808512 * torch.tanh(qk / 43.2808512)  # 30 * log2(e)
    return torch.where(q_idx >= kv_idx, qk, float("-inf"))


def stage_sandwich(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    rel_bias2: torch.Tensor,
    head_scale: torch.Tensor,
) -> torch.Tensor:
    qk = qk + hl.load(rel_bias2, [q_idx - kv_idx + 8191]) * hl.load(
        head_scale, [b_idx % 16]
    )
    return torch.where(q_idx >= kv_idx, qk, float("-inf"))


def stage_sigmoid_act(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    return torch.log(torch.sigmoid(qk) + 1.0)


def stage_sliding_window1024(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    delta = q_idx - kv_idx
    return torch.where((delta >= 0) & (delta <= 1024), qk, float("-inf"))


def stage_dilated_sw512x2(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    diff = torch.abs(q_idx - kv_idx)
    return torch.where((diff <= 512) & ((diff % 2) == 0), qk, float("-inf"))


def stage_prefix_lm1024(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    return torch.where((kv_idx < 1024) | (q_idx >= kv_idx), qk, float("-inf"))


def stage_global_sw512(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    is_global: torch.Tensor,
) -> torch.Tensor:
    keep = (
        (torch.abs(q_idx - kv_idx) <= 512)
        | hl.load(is_global, [q_idx])
        | hl.load(is_global, [kv_idx])
    )
    return torch.where(keep, qk, float("-inf"))


def stage_document(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    doc_ids: torch.Tensor,
) -> torch.Tensor:
    keep = (hl.load(doc_ids, [q_idx]) == hl.load(doc_ids, [kv_idx])) & (q_idx >= kv_idx)
    return torch.where(keep, qk, float("-inf"))


def stage_natten2d_128_13(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    q_x = q_idx // 128
    q_y = q_idx % 128
    kv_x = kv_idx // 128
    kv_y = kv_idx % 128
    center_x = q_x.clamp(6, 121)
    center_y = q_y.clamp(6, 121)
    hori = (center_x - 6 <= kv_x) & (kv_x <= center_x + 6)
    vert = (center_y - 6 <= kv_y) & (kv_y <= center_y + 6)
    return torch.where(hori & vert, qk, float("-inf"))


def stage_sta2d_128_64_16(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    # canvas 128x128, tile 16x16 (numel 256, 8x8 tile grid), kernel 64x64
    # (4x4 tiles): left border 2, right border 1.
    q_tile = q_idx // 256
    kv_tile = kv_idx // 256
    q_tile_h = q_tile // 8
    q_tile_w = q_tile % 8
    kv_tile_h = kv_tile // 8
    kv_tile_w = kv_tile % 8
    center_h = q_tile_h.clamp(2, 6)
    center_w = q_tile_w.clamp(2, 6)
    h_mask = (kv_tile_h >= center_h - 2) & (kv_tile_h <= center_h + 1)
    w_mask = (kv_tile_w >= center_w - 2) & (kv_tile_w <= center_w + 1)
    return torch.where(h_mask & w_mask, qk, float("-inf"))


def stage_block_diffusion4096_128(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    q_noised = q_idx < 4096
    kv_noised = kv_idx < 4096
    q_block = (q_idx % 4096) // 128
    kv_block = (kv_idx % 4096) // 128
    block_diagonal = (q_block == kv_block) & (q_noised == kv_noised)
    offset_block_causal = (q_block > kv_block) & q_noised & ~kv_noised
    block_causal = (q_block >= kv_block) & ~q_noised & ~kv_noised
    keep = block_diagonal | offset_block_causal | block_causal
    return torch.where(keep, qk, float("-inf"))


def stage_causal(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    return torch.where(q_idx >= kv_idx, qk, float("-inf"))


def stage_gemma2(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    qk = 72.134752 * torch.tanh(qk / 72.134752)  # 50 * log2(e)
    delta = q_idx - kv_idx
    return torch.where((delta >= 0) & (delta <= 1024), qk, float("-inf"))


def stage_shared_prefix(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    doc_start_tok: torch.Tensor,
    prefix_start_tok: torch.Tensor,
    prefix_end_tok: torch.Tensor,
) -> torch.Tensor:
    same_doc_causal = (kv_idx >= hl.load(doc_start_tok, [q_idx])) & (q_idx >= kv_idx)
    prefix = (kv_idx >= hl.load(prefix_start_tok, [q_idx])) & (
        kv_idx < hl.load(prefix_end_tok, [q_idx])
    )
    return torch.where(same_doc_causal | prefix, qk, float("-inf"))


def stage_flamingo(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    start_tok: torch.Tensor,
    end_tok: torch.Tensor,
) -> torch.Tensor:
    keep = (q_idx >= hl.load(start_tok, [kv_idx])) & (
        q_idx < hl.load(end_tok, [kv_idx])
    )
    return torch.where(keep, qk, float("-inf"))


for _stage_fn, _variant in [
    (stage_alibi, "alibi"),
    (stage_softcap30, "softcap"),
    (stage_sandwich, "sandwich"),
    (stage_sigmoid_act, "sigmoid-act"),
    (stage_sliding_window1024, "sliding-window"),
    (stage_dilated_sw512x2, "dilated-sw"),
    (stage_prefix_lm1024, "prefix-lm"),
    (stage_global_sw512, "global-sw"),
    (stage_document, "document"),
    (stage_natten2d_128_13, "natten2d"),
    (stage_sta2d_128_64_16, "sta2d"),
    (stage_block_diffusion4096_128, "block-diffusion"),
    (stage_causal, "gqa-causal"),
    (stage_gemma2, "gemma2"),
    (stage_shared_prefix, "shared-prefix"),
    (stage_flamingo, "flamingo"),
]:
    _stage_fn.__variant__ = _variant  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Host-side wiring: kernel template + stage + extra tensors per variant
# ---------------------------------------------------------------------------


def _alibi_slopes2(num_heads: int, device: torch.device) -> torch.Tensor:
    h = torch.arange(num_heads, device=device, dtype=torch.float32)
    return torch.exp2(-((h + 1) * 8.0 / num_heads)) * LOG2E


def make_extras(name: str, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Host-side extra tensor args for the variant's Helion kernel."""
    from attngym_variants import _document_layout  # pyrefly: ignore [missing-import]
    from attngym_variants import _sandwich_tables  # pyrefly: ignore [missing-import]

    if name == "alibi":
        return (_alibi_slopes2(16, device),)
    if name == "sandwich":
        rel_bias, head_scale, _offset = _sandwich_tables(device, 16, 8192)
        return (rel_bias * LOG2E, head_scale)
    if name == "global-sw":
        is_global = torch.zeros(8192, dtype=torch.bool, device=device)
        is_global[:64] = True
        is_global[::1024] = True
        return (is_global,)
    if name == "document":
        document_id, _offsets = _document_layout(device, 32768, 12, seed=0)
        return (document_id.to(torch.int32),)
    if name == "shared-prefix":
        lengths = torch.full((8,), 2048, dtype=torch.int32, device=device)
        offsets = torch.zeros(9, dtype=torch.int32, device=device)
        offsets[1:] = torch.cumsum(lengths, dim=0)
        doc_of_tok = torch.repeat_interleave(
            torch.arange(8, device=device, dtype=torch.int64), 2048
        )
        prefix_doc = torch.tensor(
            [0, 0, 0, 0, 4, 4, 4, 4], device=device, dtype=torch.int64
        )
        doc_start_tok = offsets[doc_of_tok].to(torch.int32)
        own_prefix = prefix_doc[doc_of_tok]
        is_continuation = own_prefix != doc_of_tok
        zeros = torch.zeros_like(doc_start_tok)
        prefix_start_tok = torch.where(is_continuation, offsets[own_prefix], zeros).to(
            torch.int32
        )
        prefix_end_tok = torch.where(
            is_continuation, offsets[own_prefix + 1], zeros
        ).to(torch.int32)
        return (doc_start_tok, prefix_start_tok, prefix_end_tok)
    if name == "flamingo":
        num_images = 16
        image_tokens = 256
        starts = torch.arange(num_images, dtype=torch.int32, device=device) * 512
        ends = torch.clamp(starts + 2048, max=8192)
        image_of_tok = torch.repeat_interleave(
            torch.arange(num_images, device=device, dtype=torch.int64), image_tokens
        )
        return (starts[image_of_tok].contiguous(), ends[image_of_tok].contiguous())
    return ()


_KERNEL_AND_STAGE: dict[str, tuple[object, Callable[..., torch.Tensor]]] = {
    "alibi": (attn_stage1, stage_alibi),
    "softcap": (attn_stage0, stage_softcap30),
    "sandwich": (attn_stage2, stage_sandwich),
    "sigmoid-act": (attn_stage0_nat, stage_sigmoid_act),
    "sliding-window": (attn_stage0, stage_sliding_window1024),
    "dilated-sw": (attn_stage0, stage_dilated_sw512x2),
    "prefix-lm": (attn_stage0, stage_prefix_lm1024),
    "global-sw": (attn_stage1, stage_global_sw512),
    "document": (attn_stage1, stage_document),
    "natten2d": (attn_stage0, stage_natten2d_128_13),
    "sta2d": (attn_stage0, stage_sta2d_128_64_16),
    "block-diffusion": (attn_stage0, stage_block_diffusion4096_128),
    "gqa-causal": (attn_stage0_gqa, stage_causal),
    "gemma2": (attn_stage0_gqa, stage_gemma2),
    "shared-prefix": (attn_stage3, stage_shared_prefix),
    "flamingo": (attn_stage2, stage_flamingo),
}


def kernel_call_args(
    name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
) -> tuple[object, tuple[object, ...]]:
    """Return (kernel, full args tuple) for the named variant."""
    kernel, stage = _KERNEL_AND_STAGE[name]
    extras = make_extras(name, device)
    return kernel, (q, k, v, *extras, stage)
