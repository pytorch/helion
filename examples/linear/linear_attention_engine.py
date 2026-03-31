"""
Linear Attention Engine
=======================

Generalized chunked linear attention with Helion kernels for forward and backward.

Covers: Simple GLA, Full GLA, DeltaNet, Gated DeltaNet, KDA, Vanilla LinAttn,
Retention, Mamba-2 SSD, RWKV-6, RWKV-7.

Parameterized by:
  - decay type: scalar or diagonal
  - correction: none / rank-1 shared key / rank-1 separate key
  - elementwise modifiers: q_mod, k_mod, v_mod, decay_mod, beta_mod, a_mod, output_mod
"""

from __future__ import annotations

from typing import Callable

import torch

from .linear_attention_utils import prepare_wy_repr_bwd
from .linear_attention_utils import solve_tril_inv
import helion.experimental
import helion.language as hl

# ════════════════════════════════════════════════════════════════════════════════
# Interface
# ════════════════════════════════════════════════════════════════════════════════


class LinearAttentionEngine:
    """
    Recurrence:
        S_t = Decay(alpha_t) * S_{t-1} + beta_t * (k'_t x v'_t^T - a_t (a_t^T S_{t-1}))
        o_t = output_mod(q'_t^T * S_t)
    """

    def __init__(
        self,
        q_mod: Callable | None = None,
        k_mod: Callable | None = None,
        v_mod: Callable | None = None,
        decay_mod: Callable | None = None,
        beta_mod: Callable | None = None,
        a_mod: Callable | None = None,
        output_mod: Callable | None = None,
        chunk_size: int = 64,
    ):
        self.q_mod = q_mod
        self.k_mod = k_mod
        self.v_mod = v_mod
        self.decay_mod = decay_mod
        self.beta_mod = beta_mod
        self.a_mod = a_mod
        self.output_mod = output_mod
        self.C = chunk_size

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
        **cio: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q_p = self.q_mod(q, cio) if self.q_mod else q
        k_p = self.k_mod(k, cio) if self.k_mod else k
        v_p = self.v_mod(v, cio) if self.v_mod else v
        g = self.decay_mod(decay, cio) if self.decay_mod else decay
        b = self.beta_mod(beta, cio) if self.beta_mod and beta is not None else beta
        a_p = self.a_mod(a, cio) if self.a_mod and a is not None else a

        result = chunked_linear_attn(
            q_p,
            k_p,
            v_p,
            g,
            beta=b,
            a=a_p,
            C=self.C,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )

        if return_final_state:
            o, final_state = result
        else:
            o = result

        if self.output_mod:
            o = self.output_mod(o, cio)

        if return_final_state:
            return o, final_state
        return o


# ════════════════════════════════════════════════════════════════════════════════
# Helion kernels
# ════════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════════
# Fused recurrent kernels (single-step, for autoregressive decoding)
# ════════════════════════════════════════════════════════════════════════════════


@helion.experimental.aot_kernel()
def recurrent_step_fused(
    q: torch.Tensor,  # [BH, D]
    k: torch.Tensor,  # [BH, D]
    v: torch.Tensor,  # [BH, DV]
    state: torch.Tensor,  # [BH, D, DV]  (mutated in-place)
    alpha: torch.Tensor,  # [BH] scalar or [BH, D] diagonal
) -> torch.Tensor:  # [BH, DV] output
    """Fused recurrent step for no-correction linear attention.

    In one kernel launch:
      state = alpha * state + k^T @ v
      output = q^T @ state

    Supports both scalar decay (alpha: [BH]) and diagonal decay (alpha: [BH, D]).
    state is updated in-place.
    """
    BH = q.size(0)
    D = q.size(1)
    DV = v.size(1)
    diagonal = alpha.dim() == 2

    out = torch.empty([BH, DV], dtype=v.dtype, device=v.device)

    for tile_bh, tile_dv in hl.tile([BH, DV], block_size=[1, None]):
        idx = tile_bh.id

        # Load state slice: [D, dv_tile]
        s = state[idx, :, tile_dv].float()

        # Apply decay
        if diagonal:
            a = alpha[idx, :]  # [D]
            s = s * a[:, None]
        else:
            a = alpha[idx]  # scalar
            s = s * a

        # State update: s += k^T v = outer(k, v)
        k_vec = k[idx, :].float()  # [D]
        v_vec = v[idx, tile_dv].float()  # [dv_tile]
        s = s + k_vec[:, None] * v_vec[None, :]

        # Write state back
        state[idx, :, tile_dv] = s.to(state.dtype)

        # Output: o = q^T s = (q . each col of s)
        q_vec = q[idx, :].float()  # [D]
        o_vec = (q_vec[:, None] * s).sum(0)  # [dv_tile]
        out[idx, tile_dv] = o_vec.to(out.dtype)

    return out


@helion.experimental.aot_kernel()
def recurrent_step_correction_fused(
    q: torch.Tensor,  # [BH, D]
    k: torch.Tensor,  # [BH, D]   (correction direction, or key)
    v: torch.Tensor,  # [BH, DV]
    state: torch.Tensor,  # [BH, D, DV]  (mutated in-place)
    alpha: torch.Tensor,  # [BH] scalar or [BH, D] diagonal
    beta: torch.Tensor,  # [BH]  correction strength
) -> torch.Tensor:  # [BH, DV] output
    """Fused recurrent step with rank-1 delta-rule correction.

    In one kernel launch:
      state = alpha * state
      kts = k^T @ state
      state -= beta * k @ kts^T
      state += beta * k @ v^T
      output = q^T @ state

    state is updated in-place.
    """
    BH = q.size(0)
    D = q.size(1)
    DV = v.size(1)
    diagonal = alpha.dim() == 2

    out = torch.empty([BH, DV], dtype=v.dtype, device=v.device)

    for tile_bh, tile_dv in hl.tile([BH, DV], block_size=[1, None]):
        idx = tile_bh.id

        # Load state slice: [D, dv_tile]
        s = state[idx, :, tile_dv].float()

        # Decay
        if diagonal:
            a = alpha[idx, :]
            s = s * a[:, None]
        else:
            a = alpha[idx]
            s = s * a

        k_vec = k[idx, :].float()  # [D]
        v_vec = v[idx, tile_dv].float()  # [dv_tile]
        b = beta[idx].float()  # scalar

        # kts = k^T @ s: contract over D → [dv_tile]
        kts = (k_vec[:, None] * s).sum(0)

        # Delta rule: erase then write
        s = s - b * k_vec[:, None] * kts[None, :]
        s = s + b * k_vec[:, None] * v_vec[None, :]

        # Write state back
        state[idx, :, tile_dv] = s.to(state.dtype)

        # Output
        q_vec = q[idx, :].float()
        o_vec = (q_vec[:, None] * s).sum(0)
        out[idx, tile_dv] = o_vec.to(out.dtype)

    return out


def recurrent_step(
    q: torch.Tensor,  # [B, H, 1, D]
    k: torch.Tensor,  # [B, H, 1, D]
    v: torch.Tensor,  # [B, H, 1, DV]
    state: torch.Tensor,  # [B, H, D, DV]
    alpha: float | torch.Tensor = 1.0,
    beta_val: torch.Tensor | None = None,
    a_val: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step recurrent update using fused Helion kernels.

    Returns (output [B,H,1,DV], new_state [B,H,D,DV]).
    State is updated in-place for efficiency.
    """
    B, H, _, D = q.shape
    DV = v.shape[-1]
    BH = B * H

    q_f = q.squeeze(2).reshape(BH, D)
    k_f = k.squeeze(2).reshape(BH, D)
    v_f = v.squeeze(2).reshape(BH, DV)
    state_f = state.reshape(BH, D, DV)

    if isinstance(alpha, torch.Tensor):
        alpha_sq = alpha.squeeze(2)
        if alpha_sq.dim() == 3:
            # Diagonal: [B, H, D] → [BH, D]
            alpha_f = alpha_sq.reshape(BH, D)
        else:
            # Scalar: [B, H] → [BH]
            alpha_f = alpha_sq.reshape(BH)
    else:
        alpha_f = torch.full([BH], alpha, device=q.device, dtype=q.dtype)

    if beta_val is not None:
        b_f = beta_val.squeeze(2).reshape(BH)
        a_f = a_val.squeeze(2).reshape(BH, D) if a_val is not None else k_f
        o_f = recurrent_step_correction_fused(
            q_f,
            a_f,
            v_f,
            state_f,
            alpha_f,
            b_f,
        )
    else:
        o_f = recurrent_step_fused(
            q_f,
            k_f,
            v_f,
            state_f,
            alpha_f,
        )

    return o_f.reshape(B, H, 1, DV), state.reshape(B, H, D, DV)


# ════════════════════════════════════════════════════════════════════════════════
# Chunked Helion kernels
# ════════════════════════════════════════════════════════════════════════════════


@helion.experimental.aot_kernel()
def chunk_fwd_prescale_diag(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused pre-scaling for diagonal decay forward."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)

    q_scaled = torch.empty([BHN, C, D], dtype=torch.float32, device=q.device)
    k_intra = torch.empty([BHN, C, D], dtype=torch.float32, device=q.device)
    k_state = torch.empty([BHN, C, D], dtype=torch.float32, device=q.device)
    g_last_out = torch.empty([BHN, D], dtype=torch.float32, device=q.device)

    for tile_bhn, tile_d in hl.tile([BHN, D]):
        g_acc = hl.zeros([tile_bhn, tile_d], dtype=torch.float32)
        for c in range(C):
            g_acc = g_acc + g[tile_bhn, c, tile_d].float()
            exp_gc = torch.exp(g_acc)
            q_scaled[tile_bhn, c, tile_d] = q[tile_bhn, c, tile_d].float() * exp_gc
            k_intra[tile_bhn, c, tile_d] = k[tile_bhn, c, tile_d].float() / exp_gc
        g_last_out[tile_bhn, tile_d] = g_acc

        g_acc2 = hl.zeros([tile_bhn, tile_d], dtype=torch.float32)
        for c in range(C):
            g_acc2 = g_acc2 + g[tile_bhn, c, tile_d].float()
            k_state[tile_bhn, c, tile_d] = k[tile_bhn, c, tile_d].float() * torch.exp(
                g_acc - g_acc2
            )

    return q_scaled, k_intra, k_state, g_last_out


@helion.experimental.aot_kernel()
def chunk_fwd_h_diag_fused(
    k_state: torch.Tensor,
    v: torch.Tensor,
    g_last: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Fused state accumulation over N chunks (diagonal decay)."""
    BH = k_state.size(0)
    N = k_state.size(1)
    D = k_state.size(3)
    DV = v.size(3)

    h_all = torch.empty([BH, N, D, DV], dtype=h0.dtype, device=h0.device)

    for tile_bh, tile_dv in hl.tile([BH, DV], block_size=[1, None]):
        idx = tile_bh.id
        h_acc = h0[idx, :, tile_dv].float()

        for i_t in hl.grid(N):
            h_all[idx, i_t, :, tile_dv] = h_acc.to(h_all.dtype)
            gl_d = g_last[idx, i_t, :]
            h_acc = h_acc * torch.exp(gl_d)[:, None]
            k_i = k_state[idx, i_t, :, :]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = h_acc + torch.mm(k_i.T, v_i.float())

    return h_all


@helion.experimental.aot_kernel()
def chunk_fwd_wy_diag_helion(
    a: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cs_d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """WY decomposition with precomputed scalar A matrix."""
    BHN = a.size(0)
    C = hl.specialize(a.size(1))
    D = a.size(2)
    DV = v.size(2)

    w = torch.empty([BHN, C, D], dtype=a.dtype, device=a.device)
    u = torch.empty([BHN, C, DV], dtype=v.dtype, device=v.device)
    A_buf = torch.empty([BHN, C, C], dtype=a.dtype, device=a.device)

    for tile_bhn, tile_dv in hl.tile([BHN, DV], block_size=[1, None]):
        idx = tile_bhn.id

        a_fwd = a[idx, :, :] * torch.exp(g_cs_d[idx, :, :])
        a_bwd = a[idx, :, :] * torch.exp(-g_cs_d[idx, :, :])
        A_raw = torch.mm(a_fwd, a_bwd.T)
        A_buf[idx, :, :] = A_raw.to(A_buf.dtype)

        for i in range(C):
            bi = beta[idx, i]
            ba_i = bi * a[idx, i, :] * torch.exp(g_cs_d[idx, i, :])
            bv_i = bi * v[idx, i, tile_dv].float()

            for j in range(C):
                if j < i:
                    A_ij = bi * A_buf[idx, i, j]
                    ba_i = ba_i - A_ij * w[idx, j, :].float()
                    bv_i = bv_i - A_ij * u[idx, j, tile_dv].float()

            w[idx, i, :] = ba_i.to(w.dtype)
            u[idx, i, tile_dv] = bv_i.to(u.dtype)

    return w, u, A_buf


@helion.experimental.aot_kernel()
def chunk_fwd_phase1_diag_fused(
    w: torch.Tensor,
    u: torch.Tensor,
    a_scaled: torch.Tensor,
    g_last_d: torch.Tensor,
    h0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Phase 1 for correction: WY correction + state propagation + v_new."""
    BH = w.size(0)
    N = w.size(1)
    D = w.size(3)
    DV = u.size(3)

    h_all = torch.empty([BH, N, D, DV], dtype=h0.dtype, device=h0.device)
    v_new_all = torch.empty([BH, N, w.size(2), DV], dtype=u.dtype, device=u.device)

    for tile_bh, tile_dv in hl.tile([BH, DV], block_size=[1, None]):
        idx = tile_bh.id
        h_acc = h0[idx, :, tile_dv].float()

        for i_t in hl.grid(N):
            h_all[idx, i_t, :, tile_dv] = h_acc.to(h_all.dtype)
            h_orig = h_acc

            gl_d = g_last_d[idx, i_t, :]
            h_acc = h_orig * torch.exp(gl_d)[:, None]

            w_i = w[idx, i_t, :, :]
            u_i = u[idx, i_t, :, tile_dv]
            v_corr = torch.mm(w_i, h_orig)
            v_new = u_i.float() - v_corr
            v_new_all[idx, i_t, :, tile_dv] = v_new.to(v_new_all.dtype)

            a_sc_i = a_scaled[idx, i_t, :, :]
            h_acc = h_acc + torch.mm(a_sc_i.T, v_new)

    return h_all, v_new_all


@helion.experimental.aot_kernel()
def chunk_fwd_o_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Output computation for all chunks in parallel (no correction)."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)
    head_dim = hl.specialize(D)

    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)

    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        attn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)

        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = torch.baddbmm(o_cross, qt, ht)
            attn = torch.baddbmm(attn, qt, kt.transpose(-2, -1))

        gc = g_cs[tile_bhn, :]
        decay_ij = torch.exp(gc[:, :, None] - gc[:, None, :])
        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        attn = attn * decay_ij * causal
        o_cross = o_cross * torch.exp(gc)[:, :, None]

        vt = v[tile_bhn, :, tile_dv]
        o_intra = torch.bmm(attn.to(vt.dtype), vt)
        out[tile_bhn, :, tile_dv] = (o_cross + o_intra).to(out.dtype)

    return out


@helion.experimental.aot_kernel()
def chunk_bwd_dh_diag_fused(
    q_scaled: torch.Tensor,
    do: torch.Tensor,
    g_last: torch.Tensor,
    dh_init: torch.Tensor,
) -> torch.Tensor:
    """Fused state gradient propagation over N chunks in reverse (diagonal decay)."""
    BH = q_scaled.size(0)
    N = q_scaled.size(1)
    D = q_scaled.size(3)
    DV = do.size(3)

    dh_all = torch.empty([BH, N, D, DV], dtype=dh_init.dtype, device=dh_init.device)

    for tile_bh, tile_dv in hl.tile([BH, DV], block_size=[1, None]):
        idx = tile_bh.id
        dh_acc = dh_init[idx, :, tile_dv].float()

        for i_t in hl.grid(N):
            i = N - 1 - i_t
            dh_all[idx, i, :, tile_dv] = dh_acc.to(dh_all.dtype)
            gl_d = g_last[idx, i, :]
            dh_acc = dh_acc * torch.exp(gl_d)[:, None]
            q_i = q_scaled[idx, i, :, :]
            do_i = do[idx, i, :, tile_dv]
            dh_acc = dh_acc + torch.mm(q_i.T, do_i.float())

    return dh_all


@helion.experimental.aot_kernel()
def chunk_bwd_dh_correction_diag_fused(
    w: torch.Tensor,
    a_scaled: torch.Tensor,
    q_scaled: torch.Tensor,
    do: torch.Tensor,
    dv_new_intra: torch.Tensor,
    g_last_d: torch.Tensor,
    dh_init: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused corrected reverse dH propagation + dv_new computation."""
    BH = w.size(0)
    N = w.size(1)
    D = w.size(3)
    DV = do.size(3)

    dh_all = torch.empty([BH, N, D, DV], dtype=dh_init.dtype, device=dh_init.device)
    dv_new_all = torch.empty([BH, N, w.size(2), DV], dtype=do.dtype, device=do.device)

    for tile_bh, tile_dv in hl.tile([BH, DV], block_size=[1, None]):
        idx = tile_bh.id
        dh_acc = dh_init[idx, :, tile_dv].float()

        for i_t in hl.grid(N):
            i = N - 1 - i_t
            dh_all[idx, i, :, tile_dv] = dh_acc.to(dh_all.dtype)
            dh_orig = dh_acc

            a_sc_i = a_scaled[idx, i, :, :]
            dvi = dv_new_intra[idx, i, :, tile_dv]
            dv_new = dvi.float() + torch.mm(a_sc_i, dh_orig)
            dv_new_all[idx, i, :, tile_dv] = dv_new.to(dv_new_all.dtype)

            gl_d = g_last_d[idx, i, :]
            dh_acc = dh_orig * torch.exp(gl_d)[:, None]

            q_sc_i = q_scaled[idx, i, :, :]
            do_i = do[idx, i, :, tile_dv]
            dh_acc = dh_acc + torch.mm(q_sc_i.T, do_i.float())

            w_i = w[idx, i, :, :]
            dh_acc = dh_acc - torch.mm(w_i.T, dv_new)

    return dh_all, dv_new_all


@helion.experimental.aot_kernel()
def chunk_bwd_dqkg_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: torch.Tensor,
    g_last: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dQ, dK, dg for all chunks in parallel (no correction)."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)

    dq_out = torch.empty([BHN, C, D], dtype=q.dtype, device=q.device)
    dk_out = torch.empty([BHN, C, D], dtype=k.dtype, device=k.device)
    dg_out = torch.empty([BHN, C], dtype=g_cs.dtype, device=g_cs.device)

    for tile_bhn, tile_d in hl.tile([BHN, D]):
        dq_acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
        dk_acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)

        gc = g_cs[tile_bhn, :]
        exp_gc = torch.exp(gc)[:, :, None]
        gl = g_last[tile_bhn]
        exp_gl_minus_gc = torch.exp(gl[:, None] - gc)[:, :, None]

        for tile_dv in hl.tile(DV):
            dot = do[tile_bhn, :, tile_dv]
            vt = v[tile_bhn, :, tile_dv]
            ht = h[tile_bhn, tile_d, tile_dv]
            dht = dh[tile_bhn, tile_d, tile_dv]
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]

            dA = torch.bmm(dot, vt.transpose(-2, -1))
            decay_ij = torch.exp(gc[:, :, None] - gc[:, None, :])
            idx = hl.arange(C)
            causal = (idx[:, None] >= idx[None, :]).float()
            dA = dA * decay_ij * causal

            dq_acc = torch.baddbmm(dq_acc, dA, kt)
            dk_acc = torch.baddbmm(dk_acc, dA.transpose(-2, -1), qt)

            dq_cross = torch.bmm(dot, ht.transpose(-2, -1))
            dq_acc = dq_acc + dq_cross * exp_gc

            dk_state = torch.bmm(vt, dht.transpose(-2, -1))
            dk_acc = dk_acc + dk_state * exp_gl_minus_gc

        dq_out[tile_bhn, :, tile_d] = dq_acc.to(q.dtype)
        dk_out[tile_bhn, :, tile_d] = dk_acc.to(k.dtype)
        dg_out[tile_bhn, :] = hl.zeros([tile_bhn, C], dtype=torch.float32)

    return dq_out, dk_out, dg_out


@helion.experimental.aot_kernel()
def chunk_bwd_dv_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    k_state: torch.Tensor,
    g_cs: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
) -> torch.Tensor:
    """Compute dV for all chunks in parallel (no correction)."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = do.size(2)

    dv_out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)

    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        dv_acc = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)

        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            kst = k_state[tile_bhn, :, tile_d]
            dot = do[tile_bhn, :, tile_dv]
            dht = dh[tile_bhn, tile_d, tile_dv]

            attn = torch.bmm(qt, kt.transpose(-2, -1))
            gc = g_cs[tile_bhn, :]
            decay_ij = torch.exp(gc[:, :, None] - gc[:, None, :])
            idx = hl.arange(C)
            causal = (idx[:, None] >= idx[None, :]).float()
            attn = attn * decay_ij * causal

            dv_acc = torch.baddbmm(dv_acc, attn.transpose(-2, -1), dot)
            dv_state = torch.bmm(kst, dht)
            dv_acc = dv_acc + dv_state

        dv_out[tile_bhn, :, tile_dv] = dv_acc.to(dv_out.dtype)

    return dv_out


@helion.experimental.aot_kernel()
def chunk_bwd_dqkg_diag_helion(
    q: torch.Tensor,
    k_intra: torch.Tensor,
    k_state: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dQ, dK_intra, dK_state for diagonal decay with pre-scaling."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)

    dq_out = torch.empty([BHN, C, D], dtype=q.dtype, device=q.device)
    dk_intra_out = torch.empty([BHN, C, D], dtype=q.dtype, device=q.device)
    dk_state_out = torch.empty([BHN, C, D], dtype=q.dtype, device=q.device)

    for tile_bhn, tile_d in hl.tile([BHN, D]):
        dq_acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
        dk_intra_acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
        dk_state_acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)

        for tile_dv in hl.tile(DV):
            dot = do[tile_bhn, :, tile_dv]
            vt = v[tile_bhn, :, tile_dv]
            ht = h[tile_bhn, tile_d, tile_dv]
            dht = dh[tile_bhn, tile_d, tile_dv]
            qt = q[tile_bhn, :, tile_d]
            kit = k_intra[tile_bhn, :, tile_d]

            dA = torch.bmm(dot, vt.transpose(-2, -1))
            idx = hl.arange(C)
            causal = (idx[:, None] >= idx[None, :]).float()
            dA = dA * causal

            dq_acc = torch.baddbmm(dq_acc, dA, kit)
            dk_intra_acc = torch.baddbmm(dk_intra_acc, dA.transpose(-2, -1), qt)
            dq_cross = torch.bmm(dot, ht.transpose(-2, -1))
            dq_acc = dq_acc + dq_cross
            dk_state = torch.bmm(vt, dht.transpose(-2, -1))
            dk_state_acc = dk_state_acc + dk_state

        dq_out[tile_bhn, :, tile_d] = dq_acc.to(q.dtype)
        dk_intra_out[tile_bhn, :, tile_d] = dk_intra_acc.to(q.dtype)
        dk_state_out[tile_bhn, :, tile_d] = dk_state_acc.to(q.dtype)

    return dq_out, dk_intra_out, dk_state_out


@helion.experimental.aot_kernel()
def chunk_bwd_dg_scalar_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: torch.Tensor,
    g_last: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
) -> torch.Tensor:
    """Compute FINAL dg for all chunks (no correction, scalar decay)."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)

    dg_out = torch.empty([BHN, C], dtype=g_cs.dtype, device=g_cs.device)

    for (tile_bhn,) in hl.tile([BHN]):
        gc = g_cs[tile_bhn, :]
        gl = g_last[tile_bhn]
        decay_ij = torch.exp(gc[:, :, None] - gc[:, None, :])
        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        exp_gc = torch.exp(gc)
        exp_gl_gc = torch.exp(gl[:, None] - gc)

        dg_acc = hl.zeros([tile_bhn, C], dtype=torch.float32)

        attn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            attn = torch.baddbmm(attn, qt, kt.transpose(-2, -1))
        attn = attn * decay_ij * causal

        dA_raw = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_dv in hl.tile(DV):
            dot = do[tile_bhn, :, tile_dv]
            vt = v[tile_bhn, :, tile_dv]
            dA_raw = torch.baddbmm(dA_raw, dot, vt.transpose(-2, -1))

        dA_A = dA_raw * attn
        dg_acc = dg_acc + dA_A.sum(-1) - dA_A.sum(-2)

        for tile_dv in hl.tile(DV):
            dot = do[tile_bhn, :, tile_dv]
            qh = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
            for tile_d in hl.tile(D):
                qt = q[tile_bhn, :, tile_d]
                ht = h[tile_bhn, tile_d, tile_dv]
                qh = torch.baddbmm(qh, qt, ht)
            dg_acc = dg_acc + (dot * qh * exp_gc[:, :, None]).sum(-1)

        dg_last_acc = hl.zeros([tile_bhn], dtype=torch.float32)
        for tile_d in hl.tile(D):
            kt = k[tile_bhn, :, tile_d]
            v_dh = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
            dh_h = hl.zeros([tile_bhn, tile_d], dtype=torch.float32)
            for tile_dv in hl.tile(DV):
                vt = v[tile_bhn, :, tile_dv]
                dht = dh[tile_bhn, tile_d, tile_dv]
                ht = h[tile_bhn, tile_d, tile_dv]
                v_dh = torch.baddbmm(v_dh, vt, dht.transpose(-2, -1))
                dh_h = dh_h + (dht * ht).sum(-1)
            dg_acc = dg_acc - (v_dh * kt * exp_gl_gc[:, :, None]).sum(-1)
            dg_last_acc = dg_last_acc + dh_h.sum(-1) * torch.exp(gl)
            dg_last_acc = dg_last_acc + (v_dh * kt * exp_gl_gc[:, :, None]).sum(-1).sum(
                -1
            )

        rev_cs_mat = (idx[:, None] >= idx[None, :]).float()
        dg_rev = torch.mm(dg_acc, rev_cs_mat)
        dg_out[tile_bhn, :] = (dg_rev + dg_last_acc[:, None]).to(dg_out.dtype)

    return dg_out


@helion.experimental.aot_kernel()
def chunk_bwd_dg_diag_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    dq_raw: torch.Tensor,
    dk_intra_raw: torch.Tensor,
    dk_state_raw: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused post-scaling + dg computation for diagonal decay backward."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = h.size(2)

    dq_out = torch.empty([BHN, C, D], dtype=torch.float32, device=q.device)
    dk_out = torch.empty([BHN, C, D], dtype=torch.float32, device=q.device)
    dg_out = torch.empty([BHN, C, D], dtype=torch.float32, device=q.device)

    for tile_bhn, tile_d in hl.tile([BHN, D]):
        dh_h_sum = hl.zeros([tile_bhn, tile_d], dtype=torch.float32)
        for tile_dv in hl.tile(DV):
            dht = dh[tile_bhn, tile_d, tile_dv].float()
            ht = h[tile_bhn, tile_d, tile_dv].float()
            dh_h_sum = dh_h_sum + (dht * ht).sum(-1)

        g_acc = hl.zeros([tile_bhn, tile_d], dtype=torch.float32)
        for c in range(C):
            g_acc = g_acc + g[tile_bhn, c, tile_d].float()
            dq_out[tile_bhn, c, tile_d] = dq_raw[
                tile_bhn, c, tile_d
            ].float() * torch.exp(g_acc)
        g_last = g_acc

        dg_last_state = dh_h_sum * torch.exp(g_last)

        g_acc2 = hl.zeros([tile_bhn, tile_d], dtype=torch.float32)
        dg_last_ks = hl.zeros([tile_bhn, tile_d], dtype=torch.float32)
        for c in range(C):
            g_acc2 = g_acc2 + g[tile_bhn, c, tile_d].float()
            exp_neg_gc = torch.exp(-g_acc2)
            exp_gl_gc = torch.exp(g_last - g_acc2)

            dkir = dk_intra_raw[tile_bhn, c, tile_d].float()
            dksr = dk_state_raw[tile_bhn, c, tile_d].float()
            dk_out[tile_bhn, c, tile_d] = dkir * exp_neg_gc + dksr * exp_gl_gc

            k_c = k[tile_bhn, c, tile_d].float()
            dg_last_ks = dg_last_ks + dksr * k_c * exp_gl_gc

        dg_last_total = dg_last_state + dg_last_ks

        g_cs_cur = g_last
        suffix = dg_last_total
        for c_rev in range(C):
            c = C - 1 - c_rev
            exp_gc = torch.exp(g_cs_cur)
            exp_gl_gc = torch.exp(g_last - g_cs_cur)

            q_c = q[tile_bhn, c, tile_d].float()
            k_c = k[tile_bhn, c, tile_d].float()
            dqr = dq_raw[tile_bhn, c, tile_d].float()
            dkir = dk_intra_raw[tile_bhn, c, tile_d].float()
            dksr = dk_state_raw[tile_bhn, c, tile_d].float()

            dg_cs_c = dqr * q_c * exp_gc - dkir * k_c / exp_gc - dksr * k_c * exp_gl_gc

            suffix = suffix + dg_cs_c
            dg_out[tile_bhn, c, tile_d] = suffix

            if c > 0:
                g_cs_cur = g_cs_cur - g[tile_bhn, c, tile_d].float()

    return dq_out, dk_out, dg_out


@helion.experimental.aot_kernel()
def chunk_fwd_correction_helion(
    q: torch.Tensor,
    a: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correction forward for one chunk: sequential delta-rule recurrence on GPU."""
    BH, D, DV = h.size()
    C = hl.specialize(q.size(1))

    out = torch.empty([BH, C, DV], dtype=q.dtype, device=q.device)
    h_new = torch.empty([BH, D, DV], dtype=h.dtype, device=h.device)

    for tile_bh, tile_dv in hl.tile([BH, DV], block_size=[1, None]):
        idx_bh = tile_bh.id
        s = h[idx_bh, :, tile_dv].float()

        for j in range(C):
            gj = g[idx_bh, j, :]
            s = s * torch.exp(gj)[:, None]

            aj = a[idx_bh, j, :]
            vj = v[idx_bh, j, tile_dv]
            bj = beta[idx_bh, j]

            kts = (aj[:, None] * s).sum(0, keepdim=True)
            s = s - bj * aj[:, None] * kts
            s = s + bj * aj[:, None] * vj[None, :]

            qj = q[idx_bh, j, :]
            oj = (qj[:, None] * s).sum(0)
            out[idx_bh, j, tile_dv] = oj.to(out.dtype)

        h_new[idx_bh, :, tile_dv] = s.to(h.dtype)

    return out, h_new


# ════════════════════════════════════════════════════════════════════════════════
# Autograd integration
# ════════════════════════════════════════════════════════════════════════════════


class ChunkedLinearAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, a, C, initial_state, return_final_state):
        tensors = [q, k, v, g]
        ctx.has_beta = beta is not None
        ctx.has_a = a is not None
        if beta is not None:
            tensors.append(beta)
        if a is not None:
            tensors.append(a)
        ctx.save_for_backward(*tensors)
        ctx.C = C

        input_dtype = q.dtype
        if beta is not None:
            o, h_all, v_new_all, A_inv, w_wy = _helion_chunked_fwd_correction(
                q, k, v, g, beta, a, C, initial_state=initial_state
            )
            ctx.h_all = h_all
            ctx.v_new_all = v_new_all
            ctx.A_inv = A_inv
            ctx.w_wy = w_wy
            final_state = None
            if return_final_state:
                B, H, T_pad, D = q.shape
                DV = v.shape[-1]
                N = T_pad // C
                BH = B * H
                a_use = a if a is not None else k
                g_corr = g.float()
                if g_corr.dim() == 3:
                    g_corr = g_corr.unsqueeze(-1).expand(-1, -1, -1, D).contiguous()
                gc_last = g_corr.reshape(BH, N, C, D)[:, -1]
                _, h_new = chunk_fwd_correction_helion(
                    q.float().reshape(BH, N, C, D)[:, -1],
                    a_use.float().reshape(BH, N, C, D)[:, -1],
                    v.float().reshape(BH, N, C, DV)[:, -1],
                    gc_last,
                    beta.float().reshape(BH, N, C)[:, -1],
                    h_all[:, -1],
                )
                final_state = h_new.reshape(B, H, D, DV)
        else:
            o, h_all, final_state = _helion_chunked_fwd(
                q,
                k,
                v,
                g,
                C,
                initial_state=initial_state,
                return_final_state=return_final_state,
            )
            ctx.h_all = h_all
            ctx.v_new_all = None

        return o.to(input_dtype), final_state

    @staticmethod
    def backward(ctx, grad_output, _grad_final_state):
        tensors = ctx.saved_tensors
        q, k, v, g = tensors[:4]
        idx = 4
        beta = tensors[idx] if ctx.has_beta else None
        if ctx.has_beta:
            idx += 1
        a = tensors[idx] if ctx.has_a else None
        C = ctx.C
        h_all = ctx.h_all
        v_new_all = ctx.v_new_all

        if not ctx.has_beta:
            dq, dk, dv, dg = _helion_chunked_bwd(
                q, k, v, g, grad_output, C, h_all=h_all
            )
            return dq, dk, dv, dg, None, None, None, None, None

        A_inv = ctx.A_inv
        w_wy = ctx.w_wy
        dq, dk, dv, dg, dbeta, da = _helion_chunked_bwd_correction(
            q,
            k,
            v,
            g,
            beta,
            a,
            grad_output,
            C,
            h_all=h_all,
            v_new_all=v_new_all,
            A_inv=A_inv,
            w_wy=w_wy,
        )
        return dq, dk, dv, dg, dbeta, da, None, None, None


# ════════════════════════════════════════════════════════════════════════════════
# Forward / backward pipelines
# ════════════════════════════════════════════════════════════════════════════════


def _init_state(initial_state, BH, D, DV, ref_tensor):
    if initial_state is not None:
        return initial_state.reshape(BH, D, DV).float().contiguous()
    return ref_tensor.new_zeros(BH, D, DV, dtype=torch.float32)


def _final_state_from_h_all(h_all, k_state_4d, v_flat, g_last_4d, B, H):
    h_last = h_all[:, -1].float()
    gl = g_last_4d[:, -1]
    h_final = h_last * torch.exp(gl).unsqueeze(-1)
    h_final = h_final + torch.bmm(
        k_state_4d[:, -1].float().transpose(-2, -1),
        v_flat[:, -1].float(),
    )
    return h_final.reshape(B, H, h_final.shape[1], h_final.shape[2])


def _helion_chunked_fwd(q, k, v, g, C, initial_state=None, return_final_state=False):
    B, H, T, D = q.shape
    DV = v.shape[-1]
    N = T // C
    BH = B * H
    BHN = BH * N

    if g.dim() == 3:
        g = g.unsqueeze(-1).expand(-1, -1, -1, D).contiguous()

    qf = q.reshape(BHN, C, D)
    kf = k.reshape(BHN, C, D)
    gf = g.reshape(BHN, C, D)

    q_scaled, k_intra, k_state, g_last_flat = chunk_fwd_prescale_diag(qf, kf, gf)

    v_flat = v.reshape(BH, N, C, DV).float()
    k_state_4d = k_state.reshape(BH, N, C, D)
    g_last_4d = g_last_flat.reshape(BH, N, D)

    state = _init_state(initial_state, BH, D, DV, q)
    h_all = chunk_fwd_h_diag_fused(k_state_4d, v_flat, g_last_4d, state)

    vf2 = v_flat.reshape(BHN, C, DV)
    zero_g_cs = q.new_zeros(BHN, C, dtype=torch.float32)
    hf2 = h_all.reshape(BHN, D, DV)

    o = chunk_fwd_o_helion(q_scaled, k_intra, vf2, zero_g_cs, hf2)

    final_state = None
    if return_final_state:
        final_state = _final_state_from_h_all(
            h_all, k_state_4d, v_flat, g_last_4d, B, H
        )

    return o.reshape(B, H, T, DV), h_all, final_state


def _helion_chunked_fwd_correction(q, k, v, g, beta, a, C, initial_state=None):
    B, H, T, D = q.shape
    DV = v.shape[-1]
    N = T // C
    assert T % C == 0
    BH = B * H
    BHN = BH * N

    q, k, v, g = q.float(), k.float(), v.float(), g.float()
    beta = beta.float()
    if a is not None:
        a = a.float()
    else:
        a = k

    diagonal_decay = g.dim() == 4

    if not diagonal_decay:
        g_scalar = g.reshape(BH, N, C)
        g = g.unsqueeze(-1).expand(-1, -1, -1, D).contiguous()

    qc = q.reshape(BH, N, C, D)
    ac = a.reshape(BH, N, C, D)
    vc = v.reshape(BH, N, C, DV)
    gc = g.reshape(BH, N, C, D)
    bc = beta.reshape(BH, N, C)

    g_cs_d = gc.cumsum(-2)
    g_last_d = g_cs_d[:, :, -1, :]

    ac_flat = ac.reshape(BHN, C, D)
    vc_flat = vc.reshape(BHN, C, DV)
    g_cs_d_flat = g_cs_d.reshape(BHN, C, D)
    bc_flat = bc.reshape(BHN, C)

    w, u, A_buf = chunk_fwd_wy_diag_helion(ac_flat, vc_flat, bc_flat, g_cs_d_flat)
    A_inv = None
    w = w.reshape(BH, N, C, D)
    u = u.reshape(BH, N, C, DV)

    a_scaled_4d = ac * torch.exp(g_last_d[:, :, None, :] - g_cs_d)

    state = _init_state(initial_state, BH, D, DV, q)
    h_all, v_new_all = chunk_fwd_phase1_diag_fused(w, u, a_scaled_4d, g_last_d, state)

    if diagonal_decay:
        qf2 = (qc * torch.exp(g_cs_d)).reshape(BHN, C, D)
        af2 = (ac * torch.exp(-g_cs_d)).reshape(BHN, C, D)
        g_cs_for_output = q.new_zeros(BHN, C, dtype=torch.float32)
    else:
        qf2 = qc.reshape(BHN, C, D)
        af2 = ac.reshape(BHN, C, D)
        g_cs_for_output = g_scalar.cumsum(-1).reshape(BHN, C)

    o = chunk_fwd_o_helion(
        qf2,
        af2,
        v_new_all.reshape(BHN, C, DV),
        g_cs_for_output,
        h_all.reshape(BHN, D, DV),
    )

    return o.reshape(B, H, T, DV), h_all, v_new_all, A_inv, w


def _helion_chunked_bwd(q, k, v, g, grad_output, C, h_all=None):
    B, H, T, D = q.shape
    DV = v.shape[-1]
    N = T // C
    BH = B * H
    BHN = BH * N

    # Upcast to float32 for numerical stability
    q, k, v = q.float(), k.float(), v.float()
    g = g.float()
    grad_output = grad_output.float()

    diagonal_decay = g.dim() == 4
    g_diag = g if diagonal_decay else g.unsqueeze(-1).expand(-1, -1, -1, D).contiguous()

    qf = q.reshape(BHN, C, D)
    kf = k.reshape(BHN, C, D)
    gf = g_diag.reshape(BHN, C, D)
    q_scaled, k_intra, k_state, g_last_flat = chunk_fwd_prescale_diag(qf, kf, gf)

    vf = v.reshape(BH, N, C, DV).float()
    k_state_4d = k_state.reshape(BH, N, C, D)
    g_last_4d = g_last_flat.reshape(BH, N, D)

    if h_all is None:
        state = q.new_zeros(BH, D, DV, dtype=torch.float32)
        h_all = chunk_fwd_h_diag_fused(k_state_4d, vf, g_last_4d, state)

    q_scaled_4d = q_scaled.reshape(BH, N, C, D)
    do_flat = grad_output.reshape(BH, N, C, DV).float()
    dstate = q.new_zeros(BH, D, DV, dtype=torch.float32)
    dh_all = chunk_bwd_dh_diag_fused(q_scaled_4d, do_flat, g_last_4d, dstate)

    if diagonal_decay:
        ki_f2 = k_intra.reshape(BHN, C, D)
        ks_f2 = k_state.reshape(BHN, C, D)
        vf2 = vf.reshape(BHN, C, DV)
        hf2 = h_all.reshape(BHN, D, DV)
        dhf2 = dh_all.reshape(BHN, D, DV)
        dof2 = do_flat.reshape(BHN, C, DV)
        qf2 = q_scaled.reshape(BHN, C, D)

        dq_raw, dk_intra_raw, dk_state_raw = chunk_bwd_dqkg_diag_helion(
            qf2, ki_f2, ks_f2, vf2, hf2, dof2, dhf2
        )

        zero_g_cs = q.new_zeros(BHN, C, dtype=torch.float32)
        dv_raw = chunk_bwd_dv_helion(qf2, ki_f2, ks_f2, zero_g_cs, dof2, dhf2)
        dv = dv_raw.reshape(B, H, T, DV).to(v.dtype)

        dq, dk, dg_f32 = chunk_bwd_dg_diag_helion(
            qf,
            kf,
            gf,
            dq_raw,
            dk_intra_raw,
            dk_state_raw,
            hf2,
            dhf2,
        )

        dq = dq.reshape(B, H, T, D).to(q.dtype)
        dk = dk.reshape(B, H, T, D).to(k.dtype)
        dg = dg_f32.reshape(B, H, T, D).to(g.dtype)

        return dq, dk, dv, dg
    gc = g.reshape(B, H, N, C)
    g_cs = gc.cumsum(-1)
    g_last = g_cs[..., -1]

    qf2 = q.reshape(BHN, C, D)
    kf2 = k.reshape(BHN, C, D)
    vf2 = vf.reshape(BHN, C, DV)
    g_csf2 = g_cs.reshape(BHN, C)
    g_lastf2 = g_last.reshape(BHN)
    hf2 = h_all.reshape(BHN, D, DV)
    dhf2 = dh_all.reshape(BHN, D, DV)
    dof2 = do_flat.reshape(BHN, C, DV)

    dq_raw, dk_raw, _ = chunk_bwd_dqkg_helion(
        qf2, kf2, vf2, g_csf2, g_lastf2, hf2, dof2, dhf2
    )

    kf2_scaled = kf2 * torch.exp(g_lastf2[:, None, None] - g_csf2.unsqueeze(-1))
    dv_raw = chunk_bwd_dv_helion(qf2, kf2, kf2_scaled, g_csf2, dof2, dhf2)

    dg_final = chunk_bwd_dg_scalar_helion(
        qf2, kf2, vf2, g_csf2, g_lastf2, hf2, dof2, dhf2
    )
    dg = dg_final.reshape(B, H, T).to(g.dtype)

    return (
        dq_raw.reshape(B, H, T, D),
        dk_raw.reshape(B, H, T, D),
        dv_raw.reshape(B, H, T, DV),
        dg,
    )


def _recompute_wy(ac_flat, vc_flat, bc_flat, g_cs_d_flat, BH, N, C, D, DV, A_inv, w_wy):
    w_recomp, u_recomp, A_buf = chunk_fwd_wy_diag_helion(
        ac_flat, vc_flat, bc_flat, g_cs_d_flat
    )
    if A_inv is None:
        idx_C = torch.arange(C, device=ac_flat.device)
        mask = (idx_C.unsqueeze(-1) > idx_C.unsqueeze(-2)).float()
        A = bc_flat.unsqueeze(-1) * A_buf * mask
        A_inv = solve_tril_inv(A)
    if w_wy is not None:
        w = w_wy.reshape(BH, N, C, D)
    else:
        w = w_recomp.reshape(BH, N, C, D)
    u_val = u_recomp.reshape(BH, N, C, DV)
    return w, u_val, A_inv


def _helion_chunked_bwd_correction(
    q,
    k,
    v,
    g,
    beta,
    a,
    grad_output,
    C,
    h_all=None,
    v_new_all=None,
    A_inv=None,
    w_wy=None,
):
    B, H, T, D = q.shape
    DV = v.shape[-1]
    N = T // C
    BH = B * H
    BHN = BH * N

    q, k, v, g = q.float(), k.float(), v.float(), g.float()
    beta = beta.float()
    has_a = a is not None
    if a is not None:
        a = a.float()
    else:
        a = k

    diagonal_decay = g.dim() == 4

    if not diagonal_decay:
        g_scalar = g.reshape(BH, N, C)
        g_cs_scalar = g_scalar.cumsum(-1)
        g_last_scalar = g_cs_scalar[:, :, -1]
        g = g.unsqueeze(-1).expand(-1, -1, -1, D).contiguous()

    qc = q.reshape(BH, N, C, D)
    ac = a.reshape(BH, N, C, D)
    vc = v.reshape(BH, N, C, DV)
    gc = g.reshape(BH, N, C, D)
    bc = beta.reshape(BH, N, C)
    doc = grad_output.float().reshape(BH, N, C, DV)

    g_cs_d = gc.cumsum(-2)
    g_last_d = g_cs_d[:, :, -1, :]

    ac_flat = ac.reshape(BHN, C, D)
    vc_flat = vc.reshape(BHN, C, DV)
    g_cs_d_flat = g_cs_d.reshape(BHN, C, D)
    bc_flat = bc.reshape(BHN, C)

    w, u_val, A_inv = _recompute_wy(
        ac_flat, vc_flat, bc_flat, g_cs_d_flat, BH, N, C, D, DV, A_inv, w_wy
    )

    if h_all is None or v_new_all is None:
        a_scaled_4d = ac * torch.exp(g_last_d[:, :, None, :] - g_cs_d)
        h_all, v_new_all = chunk_fwd_phase1_diag_fused(
            w,
            u_val,
            a_scaled_4d,
            g_last_d,
            _init_state(None, BH, D, DV, q),
        )

    exp_g_cs_d = torch.exp(g_cs_d)
    exp_neg_g_cs_d = torch.exp(-g_cs_d)
    exp_gl_gc_d = torch.exp(g_last_d[:, :, None, :] - g_cs_d)

    q_scaled = q.reshape(BH, N, C, D) * exp_g_cs_d
    a_intra = ac * exp_neg_g_cs_d
    a_scaled_c = ac * exp_gl_gc_d

    qf = q_scaled.reshape(BHN, C, D)
    af_intra = a_intra.reshape(BHN, C, D)
    a_scaled_f = a_scaled_c.reshape(BHN, C, D)
    v_new_fwd_f = v_new_all.reshape(BHN, C, DV).float()
    zero_g_cs = q.new_zeros(BHN, C, dtype=torch.float32)
    hf = h_all.reshape(BHN, D, DV)
    dof = doc.reshape(BHN, C, DV)

    zero_dh = q.new_zeros(BHN, D, DV, dtype=torch.float32)
    dv_new_intra_flat = chunk_bwd_dv_helion(
        qf, af_intra, a_scaled_f, zero_g_cs, dof, zero_dh
    )
    dv_new_intra_c = dv_new_intra_flat.reshape(BH, N, C, DV)

    dstate = q.new_zeros(BH, D, DV, dtype=torch.float32)
    dh_all, dv_new_bwd = chunk_bwd_dh_correction_diag_fused(
        w,
        a_scaled_c,
        q_scaled,
        doc,
        dv_new_intra_c,
        g_last_d,
        dstate,
    )

    dv_new_bwd_f = dv_new_bwd.reshape(BHN, C, DV).float()
    dhf = dh_all.reshape(BHN, D, DV)

    dw = -torch.bmm(dv_new_bwd_f, hf.transpose(-2, -1))
    du = dv_new_bwd_f

    da_wy, dv_out_flat, dbeta_flat, dg_cs_d_wy = prepare_wy_repr_bwd(
        ac_flat, vc_flat, bc_flat, g_cs_d_flat, A_inv, dw, du
    )

    if diagonal_decay:
        dq_raw, da_intra_raw, da_state_raw = chunk_bwd_dqkg_diag_helion(
            qf,
            af_intra,
            a_scaled_f,
            v_new_fwd_f,
            hf,
            dof,
            dhf,
        )

        dq_raw_r = dq_raw.float().reshape(B, H, N, C, D)
        da_intra_raw_r = da_intra_raw.float().reshape(B, H, N, C, D)
        da_state_raw_r = da_state_raw.float().reshape(B, H, N, C, D)
        qc_r = qc.reshape(B, H, N, C, D)
        ac_r = ac.reshape(B, H, N, C, D)
        exp_g_cs_d_r = exp_g_cs_d.reshape(B, H, N, C, D)
        exp_neg_g_cs_d_r = exp_neg_g_cs_d.reshape(B, H, N, C, D)
        exp_gl_gc_d_r = exp_gl_gc_d.reshape(B, H, N, C, D)

        dg_cs_d_total = (
            dq_raw_r * qc_r * exp_g_cs_d_r
            - da_intra_raw_r * ac_r * exp_neg_g_cs_d_r
            - da_state_raw_r * ac_r * exp_gl_gc_d_r
        )

        dg_cs_d_combined = dg_cs_d_total + dg_cs_d_wy.reshape(B, H, N, C, D)
        dg_per_step = dg_cs_d_combined.flip(-2).cumsum(-2).flip(-2)

        h_all_r = h_all.float().reshape(B, H, N, D, DV)
        dh_all_r = dh_all.float().reshape(B, H, N, D, DV)
        g_last_d_r = g_last_d.reshape(B, H, N, D)
        dg_last_d = (dh_all_r * h_all_r * torch.exp(g_last_d_r).unsqueeze(-1)).sum(
            -1
        ) + (da_state_raw_r * ac_r * exp_gl_gc_d_r).sum(-2)
        dg_per_step = dg_per_step + dg_last_d.unsqueeze(-2)

        dq = (
            (dq_raw.float() * exp_g_cs_d.reshape(BHN, C, D)).reshape(B, H, T, D).float()
        )
        da_par = da_intra_raw.float() * exp_neg_g_cs_d.reshape(
            BHN, C, D
        ) + da_state_raw.float() * exp_gl_gc_d.reshape(BHN, C, D)
        da_total_flat = da_par + da_wy
        dg_out = dg_per_step.reshape(B, H, T, D).float()
    else:
        g_csf = g_cs_scalar.reshape(BHN, C)
        g_lastf = g_last_scalar.reshape(BHN)
        qf_raw = qc.reshape(BHN, C, D)
        af_raw = ac_flat

        dq_raw, da_par, _ = chunk_bwd_dqkg_helion(
            qf_raw,
            af_raw,
            v_new_fwd_f,
            g_csf,
            g_lastf,
            hf,
            dof,
            dhf,
        )

        dg_attn_state = chunk_bwd_dg_scalar_helion(
            qf_raw,
            af_raw,
            v_new_fwd_f,
            g_csf,
            g_lastf,
            hf,
            dof,
            dhf,
        )

        dg_cs_wy = dg_cs_d_wy.sum(-1)
        dg_wy_per_step = (
            dg_cs_wy.float()
            .reshape(BH, N, C)
            .flip(-1)
            .cumsum(-1)
            .flip(-1)
            .reshape(BHN, C)
        )
        dg_chunks = dg_attn_state.float() + dg_wy_per_step

        da_total_flat = da_par + da_wy
        dq = dq_raw.reshape(B, H, T, D).float()
        dg_out = dg_chunks.reshape(B, H, T).float()

    dv_out = dv_out_flat.reshape(B, H, T, DV).float()
    dbeta_out = dbeta_flat.reshape(B, H, T).float()

    if has_a:
        return (
            dq,
            None,
            dv_out,
            dg_out,
            dbeta_out,
            da_total_flat.reshape(B, H, T, D).float(),
        )
    return (
        dq,
        da_total_flat.reshape(B, H, T, D).float(),
        dv_out,
        dg_out,
        dbeta_out,
        None,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════════════


def chunked_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
    C: int = 64,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
    head_first: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Public entry point for chunked linear attention."""
    if not head_first:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        g = g.transpose(1, 2).contiguous()
        if beta is not None:
            beta = beta.transpose(1, 2).contiguous() if beta.dim() >= 3 else beta
        if a is not None:
            a = a.transpose(1, 2).contiguous()

    B, H, T, D = q.shape
    H_kv = k.shape[1]

    if H_kv < H:
        assert H % H_kv == 0
        n_rep = H // H_kv
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    T_cur = q.shape[2]
    pad = (C - T_cur % C) % C
    if pad > 0:
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        if g.dim() == 3:
            g = torch.nn.functional.pad(g, (0, pad))
        else:
            g = torch.nn.functional.pad(g, (0, 0, 0, pad))
        if beta is not None:
            if beta.dim() == 3:
                beta = torch.nn.functional.pad(beta, (0, pad))
            else:
                beta = torch.nn.functional.pad(beta, (0, pad))
        if a is not None:
            a = torch.nn.functional.pad(a, (0, 0, 0, pad))

    o, final_state = ChunkedLinearAttnFn.apply(
        q, k, v, g, beta, a, C, initial_state, return_final_state
    )

    if pad > 0:
        o = o[:, :, :T_cur]

    if not head_first:
        o = o.transpose(1, 2).contiguous()

    if return_final_state:
        return o, final_state
    return o
