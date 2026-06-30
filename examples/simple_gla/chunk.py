"""
Simple GLA - chunked-parallel
=============================

Linear attention with one scalar log-decay g_t per head per timestep applied to
the whole KV state, chunked-parallel form.

Recurrence (per timestep):
    S_t = exp(g_t) * S_{t-1} + k_t^T @ v_t
    o_t = (1/sqrt(D)) * q_t @ S_t

g is [B, T, H] negative log-decay; exp(g_t) scales the [D, DV] state before the
rank-1 k_t^T @ v_t is added. The decay enters each kernel as a per-chunk
cumulative log-decay d = cumsum(g) within a chunk.
"""

from __future__ import annotations

import math
from typing import Any

import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

# 1 / ln(2), to apply decays with exp2 instead of exp. The backward uses plain
# exp (its dg path is precision-sensitive).
RCP_LN2 = 1.4426950408889634

# %%
# Helion Kernels
# --------------
# Kernels split along the chunked-parallel pipeline:
#   forward:   fwd_h  (serial state pass)  -> fwd_o  (parallel output)
#   backward:  bwd_dh (reverse state pass) -> bwd_dqk + bwd_dv (parallel)
#
# gc[BH, N, C] is the per-chunk cumulative log-decay (cumsum of g within a chunk).


@helion.kernel()
def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    gc: torch.Tensor,
    g_last: torch.Tensor,
) -> torch.Tensor:
    """Serial state accumulation across chunks, with scalar decay.

    h_all[i] holds the state going into chunk i, seeded from a zero state:
        v_decay[t] = exp(g_last[i-1] - gc[i-1, t])
        h_all[0]   = 0
        h_all[i]   = exp(g_last[i-1]) * h_all[i-1]
                     + k[i-1]^T @ (v_decay[:, None] * v[i-1])

    Args:
        k:      [BH, N, C, D]
        v:      [BH, N, C, DV]
        gc:     [BH, N, C] per-chunk cumulative log-decay
        g_last: [BH, N] per-chunk total decay (gc[:, :, -1])
    Returns:
        h_all: [BH, N, D, DV]
    """
    BH = k.size(0)
    N = k.size(1)
    D = k.size(3)
    DV = v.size(3)

    # The stored state rides at the input dtype; only the accumulator is fp32.
    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)

    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)

        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc.to(h_all.dtype)
            g_i = gc[idx, i_t, :].float()  # [C]
            gl = g_last[idx, i_t].float()  # scalar
            k_i = k[idx, i_t, :, tile_d]
            v_i = (
                v[idx, i_t, :, tile_dv].float()
                * torch.exp2((gl - g_i) * RCP_LN2)[:, None]
            ).to(v.dtype)
            h_acc = torch.exp2(gl * RCP_LN2) * h_acc
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)

    return h_all


@helion.kernel()
def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gc: torch.Tensor,
    h: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Per-chunk parallel output with scalar decay.

        d           = gc                          # [C], cumulative log-decay
        L[i,j]      = exp(d_i - d_j)              # i >= j else 0
        attn    = (q @ k^T) * L                  # [C, D] @ [D, C] -> [C, C]
        o_cross = (q @ h) * exp(d_i)             # [C, D] @ [D, DV] -> [C, DV]
        o_intra = attn @ v                       # [C, C] @ [C, DV] -> [C, DV]
        out     = scale * (o_cross + o_intra)    # [C, DV]

    The output is linear in q, so scale factors out and is applied once at the
    end; q is passed unscaled.

    Args:
        q, k: [BHN, C, D]
        v:    [BHN, C, DV]
        gc:   [BHN, C] per-chunk cumulative log-decay
        h:    [BHN, D, DV] state entering each chunk
        scale: applied to the output.
    Returns:
        out: [BHN, C, DV]
    """
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)

    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)

    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        attn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)

        d = gc[tile_bhn, :].float() * RCP_LN2                      # [tile_bhn, C]
        exp_d = torch.exp2(d)[:, :, None]                          # [tile_bhn, C, 1]
        dmask = torch.exp2(d[:, :, None] - d[:, None, :])          # [tile_bhn, C, C]

        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qt, ht.to(qt.dtype), acc=o_cross)
            attn = hl.dot(qt, kt.transpose(-2, -1), acc=attn)

        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        attn = attn * dmask * causal
        o_cross = o_cross * exp_d

        vt = v[tile_bhn, :, tile_dv]
        o_intra = hl.dot(attn.to(vt.dtype), vt)
        out[tile_bhn, :, tile_dv] = ((o_cross + o_intra) * scale).to(out.dtype)

    return out


@helion.kernel()
def chunk_bwd_dh(
    q: torch.Tensor,
    gc: torch.Tensor,
    g_last: torch.Tensor,
    do: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Serial reverse state-gradient pass, with scalar decay.

    dh_all[i] holds the gradient coming out of chunk i, seeded from a zero
    final-state gradient:
        q_decay[t]  = exp(gc[i+1, t])
        dh_all[N-1] = 0
        dh_all[i]   = exp(g_last[i+1]) * dh_all[i+1]
                      + ((scale * q_decay)[:, None] * q[i+1])^T @ do[i+1]

    q is passed unscaled; scale folds into the q-row term here.

    Args:
        q:       [BH, N, C, D]
        gc:      [BH, N, C] per-chunk cumulative log-decay
        g_last:  [BH, N] per-chunk total decay
        do:      [BH, N, C, DV] output gradient
        scale:   folded into the q-row scaling.
    Returns:
        dh_all: [BH, N, D, DV]
    """
    BH = q.size(0)
    N = q.size(1)
    D = q.size(3)
    DV = do.size(3)

    # dh rides in fp32: it feeds dg's near-cancelling state correction (the
    # <h, dh> carry and the dk_state * k term), which loses too much to bf16
    # rounding at large T. FLA keeps it fp32 for the same reason.
    dh_all = torch.empty([BH, N, D, DV], dtype=torch.float32, device=q.device)

    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        dh_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)

        for i_t in hl.grid(N):
            i = N - 1 - i_t
            dh_all[idx, i, tile_d, tile_dv] = dh_acc
            g_i = gc[idx, i, :].float()  # [C]
            gl = g_last[idx, i].float()  # scalar
            q_i = (q[idx, i, :, tile_d].float() * (scale * torch.exp(g_i))[:, None]).to(
                q.dtype
            )
            do_i = do[idx, i, :, tile_dv]
            dh_acc = torch.exp(gl) * dh_acc
            dh_acc = hl.dot(q_i.transpose(-2, -1), do_i, acc=dh_acc)

    return dh_all


@helion.kernel()
def chunk_bwd_dqk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gc: torch.Tensor,
    g_last: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-chunk parallel dQ, dK, and dg with scalar decay.

        d               = gc                  # [C], cumulative log-decay
        decay_mask[i,j] = exp(d_i - d_j)       # i >= j else 0
        exp_d[i]        = exp(d_i)
        exp_dk[j]       = exp(g_last - d_j)
        dA      = (do @ v^T) * decay_mask                 # [C, DV] @ [DV, C] -> [C, C]
        dq = scale * (dA @ k + exp_d[:, None] * (do @ h^T))     # -> [C, D]
        dk = scale * (dA^T @ q) + exp_dk[:, None] * (v @ dh^T)  # -> [C, D]

    q is passed unscaled; scale folds in here. dq is linear in the q-free terms,
    so scale multiplies the whole dq; dk's intra term uses q, so scale folds
    into that matmul while the state term carries scale through dh.

    dd is the per-position decay-gradient term, built in registers from the
    fp32 dq/dk (a near-cancelling difference, so it stays fp32 with natural
    exp):
        D_t      = sum_D(dq_t * q_t) - sum_D(dk_t * k_t)        (every position)
        D_last  += exp(g_last) * <h, dh>_F + sum_D(dk_state * k) (last only)
    where dk_state is the state-write part of dk (exp_dk * (v @ dh^T)). dd reads
    the already-scaled dq against raw q (scale lives in dq, not in q). The
    caller finishes dg by a reverse cumsum dg_t = sum_{j>=t} D_j (kept on the
    host: it is a negligible scan and its fp32 rounding order matters for the
    near-cancelling sum). dq/dk are emitted at the input dtype.

    Args:
        q, k:   [BHN, C, D]
        v:      [BHN, C, DV]
        gc:     [BHN, C] per-chunk cumulative log-decay
        g_last: [BHN] per-chunk total decay
        h, dh:  [BHN, D, DV] forward state and its gradient
        do:     [BHN, C, DV] output gradient
        scale:  folded into dq and the dk intra term.
    Returns:
        dq, dk: [BHN, C, D] (input dtype); dd: [BHN, C] (fp32, pre-cumsum)
    """
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)

    dq_out = torch.empty([BHN, C, D], dtype=q.dtype, device=q.device)
    dk_out = torch.empty([BHN, C, D], dtype=k.dtype, device=k.device)
    dd_out = torch.empty([BHN, C], dtype=torch.float32, device=q.device)

    for tile_bhn in hl.tile(BHN):
        dA_raw = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        dq_cross_acc = hl.zeros([tile_bhn, C, D], dtype=torch.float32)
        dk_state_acc = hl.zeros([tile_bhn, C, D], dtype=torch.float32)
        # exp(g_last) * <h, dh>_F, accumulated over the value dim.
        carry = hl.zeros([tile_bhn], dtype=torch.float32)

        for tile_dv in hl.tile(DV):
            dot = do[tile_bhn, :, tile_dv]
            vt = v[tile_bhn, :, tile_dv]
            ht = h[tile_bhn, :, tile_dv]
            dht = dh[tile_bhn, :, tile_dv]

            dA_raw = hl.dot(dot, vt.transpose(-2, -1), acc=dA_raw)
            dq_cross_acc = hl.dot(
                dot, ht.transpose(-2, -1).to(dot.dtype), acc=dq_cross_acc
            )
            dk_state_acc = hl.dot(
                vt, dht.transpose(-2, -1).to(vt.dtype), acc=dk_state_acc
            )
            carry += (ht.float() * dht.float()).sum(dim=-1).sum(dim=-1)

        d = gc[tile_bhn, :].float()  # [tile_bhn, C]
        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        decay_mask = torch.exp(d[:, :, None] - d[:, None, :]) * causal
        dA = dA_raw * decay_mask

        qt = q[tile_bhn, :, :]
        kt = k[tile_bhn, :, :]
        exp_d = torch.exp(d)[:, :, None]                 # [tile_bhn, C, 1]
        gl = g_last[tile_bhn].float()                    # [tile_bhn]
        exp_dk = torch.exp(gl[:, None] - d)[:, :, None]  # [tile_bhn, C, 1]
        # the cross/state terms are decay-scaled before the add, so the matmul
        # fuses into the pre-scaled dot accumulator.
        dk_state = dk_state_acc * exp_dk
        dq_acc = hl.dot(dA.to(kt.dtype), kt, acc=dq_cross_acc * exp_d) * scale
        dk_acc = hl.dot(dA.transpose(-2, -1).to(qt.dtype), qt) * scale + dk_state

        # dd reads the fp32 dq/dk accumulators in-register, before they are cast
        # down to the input dtype for storage, so dg keeps full fp32 precision.
        dd = (dq_acc * qt.float()).sum(dim=-1) - (dk_acc * kt.float()).sum(dim=-1)
        dq_out[tile_bhn, :, :] = dq_acc.to(dq_out.dtype)
        dk_out[tile_bhn, :, :] = dk_acc.to(dk_out.dtype)

        # last-position state-write/state-carry correction, also fp32.
        dg_last = torch.exp(gl) * carry + (dk_state * kt.float()).sum(dim=-1).sum(
            dim=-1
        )
        is_last = (idx == C - 1).float()  # [C]
        dd_out[tile_bhn, :] = dd + is_last[None, :] * dg_last[:, None]

    return dq_out, dk_out, dd_out


@helion.kernel()
def chunk_bwd_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    g_last: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Per-chunk parallel dV with scalar decay.

        d               = gc                  # [C], cumulative log-decay
        decay_mask[i,j] = exp(d_i - d_j)       # i >= j else 0
        exp_dk[j]       = exp(g_last - d_j)
        attn    = scale * (q @ k^T) * decay_mask          # [C, D] @ [D, C] -> [C, C]
        dv      = attn^T @ do + exp_dk[:, None] * (k @ dh)  # + [C, C] @ [C, DV]

    q is passed unscaled; scale folds into the intra attention. The state term
    k @ dh already carries scale through dh.

    Args:
        q, k:   [BHN, C, D]
        gc:     [BHN, C] per-chunk cumulative log-decay
        g_last: [BHN] per-chunk total decay
        do:     [BHN, C, DV] output gradient
        dh:     [BHN, D, DV] state gradient
        scale:  folded into the intra attention.
    Returns:
        dv: [BHN, C, DV]
    """
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = do.size(2)

    dv_out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)

    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        dv_state = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        attn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)

        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            dht = dh[tile_bhn, tile_d, tile_dv]

            attn = hl.dot(qt, kt.transpose(-2, -1), acc=attn)
            dv_state = hl.dot(kt, dht.to(kt.dtype), acc=dv_state)

        d = gc[tile_bhn, :].float()  # [tile_bhn, C]
        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        decay_mask = torch.exp(d[:, :, None] - d[:, None, :]) * causal
        attn = attn * (decay_mask * scale)

        gl = g_last[tile_bhn].float()
        exp_dk = torch.exp(gl[:, None] - d)[:, :, None]
        dv_state = dv_state * exp_dk

        dot = do[tile_bhn, :, tile_dv]
        dv_acc = hl.dot(attn.transpose(-2, -1).to(dot.dtype), dot, acc=dv_state)
        dv_out[tile_bhn, :, tile_dv] = dv_acc.to(dv_out.dtype)

    return dv_out


# %%
# Autograd Wiring
# ---------------


def _chunk_cumsum_g(g: torch.Tensor, B: int, H: int, T: int, C: int) -> torch.Tensor:
    """Per-chunk cumulative log-decay.

    Args:
        g: [B, T, H] log-decay (host layout)
    Returns:
        gc: [B*H, N, C] cumsum of g within each chunk, fp32.

    Kept as a 3-op PyTorch path (permute -> reshape/float -> cumsum): g is tiny,
    so a hand-tiled Helion scan is launch-bound and loses to a batched cumsum.
    """
    N = T // C
    gh = g.permute(0, 2, 1).reshape(B * H, N, C).float()  # [BH, N, C]
    return gh.cumsum(dim=-1)


class _SimpleGLAFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        C: int,
        scale: float,
    ) -> torch.Tensor:
        # q is passed unscaled; scale is folded into the output.
        B, H, T, D = q.shape
        DV = v.shape[-1]
        N = T // C
        BH = B * H
        BHN = BH * N

        qf = q.reshape(BH, N, C, D)
        kf = k.reshape(BH, N, C, D)
        vf = v.reshape(BH, N, C, DV)

        gc = _chunk_cumsum_g(g, B, H, T, C)  # [BH, N, C]
        g_last = gc[:, :, -1].contiguous()  # [BH, N] per-chunk total decay
        h_all = chunk_fwd_h(kf, vf, gc, g_last)
        o = chunk_fwd_o(
            qf.reshape(BHN, C, D),
            kf.reshape(BHN, C, D),
            vf.reshape(BHN, C, DV),
            gc.reshape(BHN, C),
            h_all.reshape(BHN, D, DV),
            scale,
        )

        ctx.save_for_backward(q, k, v, g, h_all, gc, g_last)
        ctx.C = C
        ctx.scale = scale
        return o.reshape(B, H, T, DV)

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v, g, h_all, gc, g_last = ctx.saved_tensors
        C = ctx.C
        scale = ctx.scale
        B, H, T, D = q.shape
        DV = v.shape[-1]
        N = T // C
        BH = B * H
        BHN = BH * N

        # q is passed unscaled; the backward kernels fold scale in themselves.
        # dg is a near-cancelling difference (dq*q - dk*k), so its reduction stays
        # fp32 in-kernel (chunk_bwd_dqk); inputs are bf16 with fp32 accumulators.
        qf4 = q.reshape(BH, N, C, D)
        do4 = grad_output.reshape(BH, N, C, DV)

        qf2 = q.reshape(BHN, C, D)
        kf2 = k.reshape(BHN, C, D)
        vf2 = v.reshape(BHN, C, DV)
        hf = h_all.reshape(BHN, D, DV)
        dof = do4.reshape(BHN, C, DV)
        gcf = gc.reshape(BHN, C)
        g_last_f = g_last.reshape(BHN)

        dh_all = chunk_bwd_dh(qf4, gc, g_last, do4, scale)
        dhf = dh_all.reshape(BHN, D, DV)
        dq, dk, dd = chunk_bwd_dqk(qf2, kf2, vf2, gcf, g_last_f, hf, dof, dhf, scale)
        dv = chunk_bwd_dv(qf2, kf2, gcf, g_last_f, dof, dhf, scale)

        # dd holds the per-position decay-gradient term (plus the last-position
        # state correction) from chunk_bwd_dqk; dg is its reverse cumsum within
        # each chunk. The scan stays on the host: it is negligible (a single
        # cumsum over [BH, N, C]) and its fp32 rounding order matters for the
        # near-cancelling sum.
        dd = dd.reshape(BH, N, C)
        dg = dd.flip(-1).cumsum(-1).flip(-1)
        dg = dg.reshape(B, H, T).permute(0, 2, 1).contiguous()  # [B, T, H]

        # dq comes out already scaled from the kernel; dg stays fp32 to match
        # g's layout. Trailing Nones: C, scale.
        return (
            dq.reshape(B, H, T, D).to(q.dtype),
            dk.reshape(B, H, T, D).to(k.dtype),
            dv.reshape(B, H, T, DV).to(v.dtype),
            dg,
            None,
            None,
        )


# %%
# Public API
# ----------


def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    C: int = 64,
) -> torch.Tensor:
    """Simple GLA chunked linear attention with scalar decay.

    Args:
        q, k: [B, H, T, D]
        v:    [B, H, T, DV]
        g:    [B, T, H] per-head scalar log-decay (negative), token-first layout.
        scale: applied to q. Defaults to 1/sqrt(D).
        C:     chunk size. T must be divisible by C (no padding here).
    Returns:
        out: [B, H, T, DV]
    """
    D = q.size(-1)
    if scale is None:
        scale = D**-0.5
    assert q.size(-2) % C == 0, f"T={q.size(-2)} must be divisible by C={C}"
    return _SimpleGLAFn.apply(q, k, v, g, C, scale)


# %%
# References for Validation
# -------------------------


def naive_recurrent_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Step-by-step recurrent reference. Slow but obviously correct.

    Args:
        q, k: [B, H, T, D]; v: [B, H, T, DV]; g: [B, T, H] scalar log-decay.
    """
    B, H, T, D = q.shape
    DV = v.shape[-1]
    if scale is None:
        scale = D**-0.5
    S = q.new_zeros(B, H, D, DV, dtype=torch.float32)
    out = torch.empty_like(v, dtype=torch.float32)
    qf = q.float() * scale
    kf = k.float()
    vf = v.float()
    gf = g.float().permute(0, 2, 1)  # [B, H, T]
    for t in range(T):
        gate = torch.exp(gf[:, :, t])  # [B, H]
        S = S * gate[:, :, None, None] + torch.einsum(
            "bhd,bhv->bhdv", kf[:, :, t], vf[:, :, t]
        )
        out[:, :, t] = torch.einsum("bhd,bhdv->bhv", qf[:, :, t], S)
    return out.to(v.dtype)


def fla_chunk_simple_gla_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """FLA's chunk_simple_gla on its NATIVE token-first [B, T, H, D] layout.

    No transpose: callers that already hold token-first tensors use this directly
    so the layout conversion is not charged to FLA's measured time. g is already
    token-first [B, T, H]. Unwraps FLA's (output, final_state) tuple.
    """
    from fla.ops.simple_gla import chunk_simple_gla as _fla  # pyrefly: ignore

    o = _fla(q, k, v, g=g, scale=scale)
    return o[0] if isinstance(o, tuple) else o


def fla_chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    C: int = 64,
) -> torch.Tensor:
    """FLA's chunk_simple_gla wrapped to accept head-first [B, H, T, D].

    Transposes q/k/v to FLA's native layout and back; used by ``test()`` for
    correctness. For timing, use ``fla_chunk_simple_gla_native`` and transpose
    outside the timed region. g is already token-first [B, T, H], no transpose.
    """
    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()
    o = fla_chunk_simple_gla_native(qt, kt, vt, g, scale=scale)
    return o.transpose(1, 2).contiguous()


# %%
# Testing Function
# ----------------


def test(
    b: int,
    h: int,
    t: int,
    d: int,
    dv: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
) -> None:
    """Forward + backward correctness vs a naive recurrent reference and FLA.

    Args:
        b: Batch size
        h: Number of heads
        t: Sequence length (must be divisible by chunk size 64)
        d: Query/key head dim
        dv: Value head dim
    """
    try:
        from fla.ops.simple_gla import chunk_simple_gla as _fla_fn  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "fla is required for this example's correctness check; "
            "install flash-linear-attention"
        ) from e

    torch.manual_seed(42)
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, h, t, d, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(b, h, t, d, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(b, h, t, dv, dtype=dtype, device=device, requires_grad=True)
    # scalar log-decay per head per timestep, token-first [B, T, H], negative.
    g = torch.nn.functional.logsigmoid(
        torch.randn(b, t, h, dtype=torch.float32, device=device)
    ).requires_grad_(True)

    def helion_fn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        return chunk_simple_gla(q, k, v, g, scale=scale)

    def fla_fn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        return fla_chunk_simple_gla(q, k, v, g, scale=scale)

    def naive_fn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        return naive_recurrent_simple_gla(q, k, v, g, scale=scale)

    run_example(
        helion_fn,
        {"naive_recurrent": naive_fn, "fla": fla_fn},
        (q, k, v, g),
        kernel_name="helion",
        baseline_name="naive_recurrent",
        bwd=True,
    )


# %%
# Main Function
# -------------


def main() -> None:
    """Run a single modest shape (B=1, H=2, T=512, D=DV=64, bf16)."""
    test(1, 2, 512, 64, 64, HALF_DTYPE, device=DEVICE)


if __name__ == "__main__":
    main()
