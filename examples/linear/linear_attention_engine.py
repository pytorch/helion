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

from enum import Enum
from typing import Any
from typing import Callable
from typing import Literal
from typing import Protocol
from typing import overload

import torch

import helion
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
    ) -> None:
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

        result = chunked_linear_attn(  # pyrefly: ignore
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

        final_state = None
        if return_final_state:
            o, final_state = result
        else:
            o = result

        if self.output_mod:
            o = self.output_mod(o, cio)

        if return_final_state:
            return o, final_state  # pyrefly: ignore
        return o


# ════════════════════════════════════════════════════════════════════════════════
# Helion kernels
# ════════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════════
# Fused recurrent kernels (single-step, for autoregressive decoding)
# ════════════════════════════════════════════════════════════════════════════════


@helion.kernel()
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


@helion.kernel()
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

    q_f = q.squeeze(2).contiguous().reshape(BH, D)
    k_f = k.squeeze(2).contiguous().reshape(BH, D)
    v_f = v.squeeze(2).contiguous().reshape(BH, DV)
    state_f = state.contiguous().reshape(BH, D, DV)

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

# 1 / ln(2), to apply decays with exp2 (one hardware instruction) instead of exp.
RCP_LN2 = 1.4426950408889634


@helion.kernel()
def chunk_fwd_h_diag_fused(
    k: torch.Tensor,
    v: torch.Tensor,
    g_last: torch.Tensor,
    h0: torch.Tensor,
    gc: torch.Tensor | None = None,
    use_g: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
    scalar_decay: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    diag_anchored: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """Fused state accumulation over N chunks.

    Tiles [BH, D, DV] to parallelize over both key and value dimensions, matching
    FLA's grid=(K_blocks, V_blocks, BH). Compile-time decay modes (specialized, no
    runtime branch):
      - use_g=False:        no decay; k is passed raw.
      - diagonal (default): k is pre-scaled by the state decay on host; g_last is
        a per-channel [BH, N, D] state decay applied as h_acc * exp(g_last).
      - scalar_decay=True:  k is raw and gc [BH, N, C] decays v in-kernel; g_last
        is a scalar [BH, N] state decay. With g in natural-log space:
            h_all[i] = exp(g_last[i-1]) * h_all[i-1]
                       + k[i-1]^T @ (v[i-1] * exp(g_last[i-1] - gc[i-1]))
        which the kernel computes as exp2(RCP_LN2 * x) == exp(x), since exp2 is
        one hardware instruction:
            h_all[i] = exp2(RCP_LN2 * g_last[i-1]) * h_all[i-1]
                       + k[i-1]^T @ (v[i-1] * exp2(RCP_LN2 * (g_last[i-1] - gc[i-1])))
      - diag_anchored=True: k is raw and gc [BH, N, C, D] is the per-channel
        cumsum; the decay rides k per channel instead of v (exp2 of RCP_LN2 * gc):
            h_all[i] = exp2(RCP_LN2 * gc_last[i-1]) * h_all[i-1]
                       + (k[i-1] * exp2(RCP_LN2 * (gc_last[i-1] - gc[i-1])))^T @ v[i-1]
    """
    BH = k.size(0)
    N = k.size(1)
    C = hl.specialize(k.size(2))
    D = k.size(3)
    DV = v.size(3)

    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)

    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = h0[idx, tile_d, tile_dv].float()

        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc.to(h_all.dtype)
            k_i = k[idx, i_t, :, tile_d]
            if scalar_decay:
                g_i = gc[idx, i_t, :].float()  # [C]
                gl = g_last[idx, i_t].float()  # scalar
                v_i = (
                    v[idx, i_t, :, tile_dv].float()
                    * torch.exp2((gl - g_i) * RCP_LN2)[:, None]
                ).to(v.dtype)
                h_acc = torch.exp2(gl * RCP_LN2) * h_acc
            elif diag_anchored:
                gc_i = gc[idx, i_t, :, tile_d]
                gc_last = gc[idx, i_t, C - 1, tile_d]
                k_i = (
                    k_i.float() * torch.exp2((gc_last[None, :] - gc_i) * RCP_LN2)
                ).to(k.dtype)
                h_acc = h_acc * torch.exp2(gc_last * RCP_LN2)[:, None]
                v_i = v[idx, i_t, :, tile_dv]
            else:
                if use_g:
                    gl_d = g_last[idx, i_t, tile_d]
                    h_acc = h_acc * torch.exp(gl_d)[:, None]
                v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)

    return h_all


@helion.kernel()
def chunk_fwd_wy_delta_helion(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cs: torch.Tensor | None = None,
    Akk: torch.Tensor | None = None,
    scalar_decay: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    diag_anchored: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """WY / UT transform for the delta rule (beta correction). Builds the
    unit-lower-triangular inverse T by Neumann-series doubling, then the WY
    factors w, u; T is returned as A_inv so the backward reuses it. Decay modes:
      - scalar_decay=False, diag_anchored=False (default): no decay (delta, g=0):
            kk = k @ k.T                    # [C, C]
            A  = -(beta * kk) * strict      # [C, C], strict lower (s < t)
            T  = (I - A)^-1                 # Neumann doubling
            w  = T @ (beta * k)             # [C, D]
            u  = T @ (beta * v)             # [C, DV]
      - scalar_decay=True: g_cs [BHN, C] scalar cumulative log-decay (gated_delta);
        the decay folds into A and the w-key (u unchanged), gc = g_cs:
            A = -(beta * (k @ k.T) * exp(gc - gc^T)) * strict
            w = T @ (beta * exp(gc) * k)
      - diag_anchored=True: g_cs [BHN, C, D] per-channel cumulative log-decay (kda).
        The per-channel L-mask exp(gc - gc^T) is [C,C,D] and unbounded, so A reads
        the precomputed anchored k-gram Akk (already carries the decay + strict
        mask); the w-key rides per-channel exp(gc):
            A = -beta * Akk
            w = T @ (beta * exp(gc) * k);  u = T @ (beta * v)
    """
    BHN = k.size(0)
    C = hl.specialize(k.size(1))
    D = k.size(2)
    DV = v.size(2)
    n_doublings = C.bit_length() - 1

    w = torch.empty([BHN, C, D], dtype=k.dtype, device=k.device)
    u = torch.empty([BHN, C, DV], dtype=v.dtype, device=v.device)
    A_inv = torch.empty([BHN, C, C], dtype=torch.float32, device=k.device)

    for tile_bhn in hl.tile(BHN, block_size=1):
        beta_i = beta[tile_bhn, :].to(torch.float32)  # [1, C]

        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            kt = k[tile_bhn, :, tile_d]
            kk = hl.dot(kt, kt.transpose(-2, -1), acc=kk)
        idx = hl.arange(C)
        strict_lower = (idx[:, None] > idx[None, :]).to(torch.float32)
        if diag_anchored:
            # The anchored k-gram Akk already carries the per-channel decay and
            # the strict-lower mask (kk above is unused on this path).
            A = -(beta_i[:, :, None] * Akk[tile_bhn, :, :])
        elif scalar_decay:
            decay = g_cs[tile_bhn, :].to(torch.float32)  # [1, C]
            L = torch.exp2((decay[:, :, None] - decay[:, None, :]) * RCP_LN2)
            A = -(beta_i[:, :, None] * kk * L) * strict_lower
        else:
            A = -(beta_i[:, :, None] * kk) * strict_lower

        eye = (idx[:, None] == idx[None, :]).to(torch.float32)
        eye = eye[None, :, :].broadcast_to([tile_bhn, C, C])
        T = eye + A
        Apow = A
        for _ in range(n_doublings - 1):
            Apow = hl.dot(Apow, Apow)
            T = hl.dot(Apow, T, acc=T)

        A_inv[tile_bhn, :, :] = T

        for tile_d in hl.tile(D):
            kt = k[tile_bhn, :, tile_d].to(torch.float32) * beta_i[:, :, None]
            if scalar_decay:
                kt = kt * torch.exp2(decay * RCP_LN2)[:, :, None]
            elif diag_anchored:
                gc_d = g_cs[tile_bhn, :, tile_d].to(torch.float32)
                kt = kt * torch.exp2(gc_d * RCP_LN2)
            w[tile_bhn, :, tile_d] = hl.dot(T, kt).to(w.dtype)
        for tile_dv in hl.tile(DV):
            vt = v[tile_bhn, :, tile_dv].to(torch.float32) * beta_i[:, :, None]
            u[tile_bhn, :, tile_dv] = hl.dot(T, vt).to(u.dtype)

    return w, u, A_inv


@helion.kernel()
def chunk_fwd_h_delta_helion(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    h0: torch.Tensor,
    g_cs: torch.Tensor | None = None,
    decay_last: torch.Tensor | None = None,
    scalar_decay: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    diag_anchored: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serial state pass for the delta rule. Walk i = 0 -> N-1 carrying the state
    S; h_all[i] is the state entering chunk i. The chunk walk is serial; DV is
    split across programs (D kept whole), so each holds a [D, tile_dv] slice of S.
    Decay modes:
      - scalar_decay=False, diag_anchored=False (default): no decay (delta, g=0):
            h_all[i] = S                       # [D, DV]
            v_new[i] = u[i] - w[i] @ S         # [C, DV]  delta correction
            S        = S + k[i].T @ v_new[i]   # [D, DV]  write corrected values
      - scalar_decay=True: g_cs [BH, N, C] scalar cumulative log-decay, decay_last
        [BH, N] per-chunk total (gated_delta); the carry decays and the key rides
        the anchored decay (v_new unchanged), gc = g_cs:
            S = exp(decay_last) * S + (k[i] * exp(decay_last - gc[i])).T @ v_new[i]
      - diag_anchored=True: g_cs [BH, N, C, D] per-channel cumulative log-decay
        (kda); gc_last = g_cs[.,C-1,.] is a length-D vector, so the carry decays
        per-channel over D and the key rides the per-channel anchored decay:
            S = exp(gc_last)[:, None] * S
                + (k[i] * exp(gc_last[None, :] - gc[i])).T @ v_new[i]
    """
    BH = k.size(0)
    N = k.size(1)
    D = k.size(3)
    DV = u.size(3)

    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)
    v_new = torch.empty([BH, N, w.size(2), DV], dtype=k.dtype, device=u.device)

    for tile_bh, tile_dv in hl.tile([BH, DV], block_size=[1, None]):
        idx = tile_bh.id
        h_acc = h0[idx, :, tile_dv].float()  # [D, bv]

        for i_t in hl.grid(N):
            h_all[idx, i_t, :, tile_dv] = h_acc.to(h_all.dtype)
            h_orig = h_acc
            w_i = w[idx, i_t, :, :]  # [C, D]
            u_i = u[idx, i_t, :, tile_dv]  # [C, bv]
            vnew_i = u_i.float() - hl.dot(w_i, h_orig.to(w_i.dtype)).float()
            v_new[idx, i_t, :, tile_dv] = vnew_i.to(v_new.dtype)
            k_i = k[idx, i_t, :, :]  # [C, D]
            if scalar_decay:
                decay_i = g_cs[idx, i_t, :].float()  # [C]
                dl = decay_last[idx, i_t].float()  # scalar
                k_i = (
                    k_i.float() * torch.exp2((dl - decay_i) * RCP_LN2)[:, None]
                ).to(k_i.dtype)
                h_orig = h_orig * torch.exp2(dl * RCP_LN2)
            elif diag_anchored:
                gc_i = g_cs[idx, i_t, :, :].float()  # [C, D]
                gl = decay_last[idx, i_t, :].float()  # [D]
                k_i = (
                    k_i.float() * torch.exp2((gl[None, :] - gc_i) * RCP_LN2)
                ).to(k_i.dtype)
                h_orig = h_orig * torch.exp2(gl * RCP_LN2)[:, None]
            h_acc = hl.dot(k_i.transpose(-2, -1), vnew_i.to(k_i.dtype), acc=h_orig)

    return h_all, v_new


@helion.kernel()
def chunk_fwd_o_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: torch.Tensor,
    h: torch.Tensor,
    use_g: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
    scale: float = 1.0,
) -> torch.Tensor:
    """Output computation for all chunks in parallel (no correction). With
    use_g=False the decay is compiled out for variants with no decay. The output
    is linear in q, so scale is applied to it and q can be passed unscaled.
    """
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)
    hl.specialize(D)

    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)

    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        attn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)

        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qt, ht, acc=o_cross)
            attn = hl.dot(qt, kt.transpose(-2, -1), acc=attn)

        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        if use_g:
            gc = g_cs[tile_bhn, :]
            decay_ij = torch.exp2((gc[:, :, None] - gc[:, None, :]) * RCP_LN2)
            attn = attn * decay_ij * causal
            o_cross = o_cross * torch.exp2(gc * RCP_LN2)[:, :, None]
        else:
            attn = attn * causal

        vt = v[tile_bhn, :, tile_dv]
        # o_intra = attn @ v accumulated onto o_cross
        o = hl.dot(attn.to(vt.dtype), vt, acc=o_cross)
        out[tile_bhn, :, tile_dv] = (o * scale).to(out.dtype)

    return out


@helion.kernel()
def chunk_bwd_dstate_delta_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dh0: torch.Tensor,
    g_cs: torch.Tensor | None = None,
    decay_last: torch.Tensor | None = None,
    scalar_decay: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse state pass for the delta rule. Walk i = N-1 -> 0 carrying the
    reverse state dS; emits the full value grad dvni so the per-chunk pass need
    not recompute it. The chunk walk is serial; DV is split across programs (D
    kept whole), so each holds a [D, tile_dv] slice of dS. Decay modes:
      - scalar_decay=False (default): no decay (delta_rule, g=0):
            dS_future[i] = dS
            attn   = causal(q_i @ k_i.T)                 # [C, C]  s <= t
            dvni_i = attn.T @ do_i + k_i @ dS            # [C, DV]
            dS     = dS + q_i.T @ do_i - w_i.T @ dvni_i  # [D, DV]
      - scalar_decay=True: g_cs [BH, N, C] cumulative log-decay, decay_last
        [BH, N] per-chunk total (gated_delta); gc = g_cs[i], dl = decay_last[i]:
            attn   = causal(q_i @ k_i.T) * exp(gc - gc^T)
            dvni_i = attn.T @ do_i + exp(dl - gc)[:, None] * (k_i @ dS)
            dS     = exp(dl) * dS + q_i.T @ (exp(gc) * do_i) - w_i.T @ dvni_i
    """
    BH = q.size(0)
    N = q.size(1)
    C = hl.specialize(q.size(2))
    D = q.size(3)
    DV = do.size(3)

    dS_future = torch.empty([BH, N, D, DV], dtype=q.dtype, device=q.device)
    dvni = torch.empty([BH, N, C, DV], dtype=q.dtype, device=q.device)

    for tile_bh, tile_dv in hl.tile([BH, DV], block_size=[1, None]):
        idx = tile_bh.id
        dS = dh0[idx, :, tile_dv].float()  # [D, bv]

        for i_rev in hl.grid(N):
            i = N - 1 - i_rev
            dS_future[idx, i, :, tile_dv] = dS.to(dS_future.dtype)
            q_i = q[idx, i, :, :]  # [C, D]
            k_i = k[idx, i, :, :]  # [C, D]
            do_i = do[idx, i, :, tile_dv]  # [C, bv]

            jdx = hl.arange(C)
            causal = (jdx[:, None] >= jdx[None, :]).to(torch.float32)
            attn = hl.dot(q_i, k_i.transpose(-2, -1)).float()
            if scalar_decay:
                gc = g_cs[idx, i, :].float()  # [C]
                dl = decay_last[idx, i].float()  # scalar
                attn = attn * torch.exp2((gc[:, None] - gc[None, :]) * RCP_LN2)
            attn = (attn * causal).to(q.dtype)

            dv_i = hl.dot(attn.transpose(-2, -1), do_i).float()
            if scalar_decay:
                dv_kh = hl.dot(k_i, dS.to(k_i.dtype)).float()
                dv_i = dv_i + dv_kh * torch.exp2((dl - gc) * RCP_LN2)[:, None]
            else:
                dv_i = hl.dot(k_i, dS.to(k_i.dtype), acc=dv_i)  # [C, bv]
            dvni[idx, i, :, tile_dv] = dv_i.to(dvni.dtype)

            w_i = w[idx, i, :, :]  # [C, D]
            if scalar_decay:
                dog = (do_i.float() * torch.exp2(gc * RCP_LN2)[:, None]).to(q.dtype)
                dS = dS * torch.exp2(dl * RCP_LN2)
                dS = hl.dot(q_i.transpose(-2, -1), dog, acc=dS)
            else:
                dS = hl.dot(q_i.transpose(-2, -1), do_i.to(q_i.dtype), acc=dS)
            dS = dS - hl.dot(w_i.transpose(-2, -1), dv_i.to(w_i.dtype)).float()

    return dS_future, dvni


@helion.kernel()
def chunk_bwd_dqkw_delta_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    v_new: torch.Tensor,
    do: torch.Tensor,
    dvni: torch.Tensor,
    dS_future: torch.Tensor,
    g_cs: torch.Tensor | None = None,
    decay_last: torch.Tensor | None = None,
    scalar_decay: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-chunk parallel adjoint of the delta output and state carry. dattn sums
    over DV in its own loop up front (formed once); dW/dq/dk_as then tile over D
    with an inner DV loop, reusing it. Decay modes:
      - scalar_decay=False (default): no decay (delta_rule, g=0):
            dattn = causal(do @ v_new.T)               # [C, C]
            dW    = -dvni @ S.T                         # [C, D]  grad w.r.t. w
            dq    = dattn @ k + do @ S.T                # [C, D]  intra + inter
            dk_as = dattn.T @ q + v_new @ dS_future.T   # [C, D]  attn + state
        dg_p, dg_last are unwritten (delta has no decay gradient).
      - scalar_decay=True: g_cs [BHN, C] cumulative log-decay, decay_last [BHN]
        per-chunk total (gated_delta); gc = g_cs, dl = decay_last:
            dattn = causal(do @ v_new.T) * exp(gc - gc^T)
            dq    = dattn @ k + exp(gc)[:, None] * (do @ S.T)
            dk_as = dattn.T @ q + exp(dl - gc)[:, None] * (v_new @ dS_future.T)
        and the decay gradient (o-kernel + state-carry parts):
            dg_p    = (dq * q).sum(-1) - (dk_as * k).sum(-1)
            dg_last = exp(dl) * (h * dh_out).sum() + (dk_state * k).sum()
    """
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = hl.specialize(do.size(2))

    dW_out = torch.empty([BHN, C, D], dtype=torch.float32, device=q.device)
    dq_out = torch.empty([BHN, C, D], dtype=q.dtype, device=q.device)
    dk_as_out = torch.empty([BHN, C, D], dtype=torch.float32, device=k.device)
    dg_p_out = torch.empty([BHN, C], dtype=torch.float32, device=q.device)
    dg_last_out = torch.empty([BHN], dtype=torch.float32, device=q.device)

    hdt = q.dtype
    for tile_bhn in hl.tile(BHN, block_size=1):
        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).to(torch.float32)
        if scalar_decay:
            decay_b = g_cs[tile_bhn, :].to(torch.float32)  # [1, C]
            dl = decay_last[tile_bhn].to(torch.float32)  # [1]
            decay_exp = torch.exp2(decay_b * RCP_LN2)[:, :, None]  # [1, C, 1]
            kdec_w = torch.exp2((dl[:, None] - decay_b) * RCP_LN2)[:, :, None]

        dattn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_dv in hl.tile(DV):
            do_dv = do[tile_bhn, :, tile_dv].to(hdt)
            vnew_h = v_new[tile_bhn, :, tile_dv].to(hdt)
            dattn = hl.dot(do_dv, vnew_h.transpose(-2, -1), acc=dattn)
        if scalar_decay:
            L = torch.exp2((decay_b[:, :, None] - decay_b[:, None, :]) * RCP_LN2)
            dattn = dattn * L
        dattn = (dattn * causal).to(hdt)  # [1, C, C]

        dg_p = hl.zeros([tile_bhn, C], dtype=torch.float32)
        dg_last_dk = hl.zeros([tile_bhn], dtype=torch.float32)
        for tile_d in hl.tile(D):
            q_d = q[tile_bhn, :, tile_d].to(hdt)  # [1, C, bd]
            k_d = k[tile_bhn, :, tile_d].to(hdt)  # [1, C, bd]
            dW = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
            dq_cross = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
            dk_state = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
            for tile_dv in hl.tile(DV):
                S_h = h[tile_bhn, tile_d, tile_dv].to(hdt)  # [1, bd, dv]
                do_dv = do[tile_bhn, :, tile_dv].to(hdt)  # [1, C, dv]
                dSf_dv = dS_future[tile_bhn, tile_d, tile_dv].to(hdt)  # [1, bd, dv]
                dvni_h = dvni[tile_bhn, :, tile_dv].to(hdt)  # [1, C, dv]
                vnew_h = v_new[tile_bhn, :, tile_dv].to(hdt)  # [1, C, dv]
                dW = dW - hl.dot(dvni_h, S_h.transpose(-2, -1)).float()
                dq_cross = hl.dot(do_dv, S_h.transpose(-2, -1), acc=dq_cross)
                dk_state = hl.dot(vnew_h, dSf_dv.transpose(-2, -1), acc=dk_state)

            if scalar_decay:
                dq_cross = dq_cross * decay_exp
                dk_state = dk_state * kdec_w

            dW_out[tile_bhn, :, tile_d] = dW
            dq = hl.dot(dattn, k_d, acc=dq_cross)  # [1, C, bd]
            dq_out[tile_bhn, :, tile_d] = dq.to(dq_out.dtype)
            dk_as = hl.dot(dattn.transpose(-2, -1), q_d, acc=dk_state)
            dk_as_out[tile_bhn, :, tile_d] = dk_as

            if scalar_decay:
                qf = q[tile_bhn, :, tile_d].to(torch.float32)
                kf = k[tile_bhn, :, tile_d].to(torch.float32)
                dg_p = dg_p + (dq.float() * qf).sum(-1) - (dk_as * kf).sum(-1)
                dg_last_dk = dg_last_dk + (dk_state * kf).sum(-1).sum(-1)

        if scalar_decay:
            dg_p_out[tile_bhn, :] = dg_p
            hd = hl.zeros([tile_bhn], dtype=torch.float32)
            for tile_d, tile_dv in hl.tile([D, DV]):
                h_t = h[tile_bhn, tile_d, tile_dv].to(torch.float32)
                dhot = dS_future[tile_bhn, tile_d, tile_dv].to(torch.float32)
                hd = hd + (h_t * dhot).sum(-1).sum(-1)
            dg_last_out[tile_bhn] = torch.exp2(dl * RCP_LN2) * hd + dg_last_dk

    return dW_out, dq_out, dk_as_out, dg_p_out, dg_last_out


@helion.kernel()
def chunk_bwd_wy_dL_delta_helion(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A_inv: torch.Tensor,
    dW: torch.Tensor,
    dvni: torch.Tensor,
    g_cs: torch.Tensor | None = None,
    scalar_decay: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backprop through the WY solve for dv, dbeta, and the dL/d_ba that
    chunk_bwd_dk_delta_helion needs to finish dk (a = k shared key). T = A_inv is
    the saved inverse. The decay scalings are folded into dL and d_ba here, so
    chunk_bwd_dk is unchanged (its dL @ k matmuls stay off the triangular-inverse
    dots, which miscompile at C = 64). Decay modes:
      - scalar_decay=False (default): no decay (delta_rule, g=0):
            dAinv = dvni @ (beta*v).T + dW @ (beta*k).T
            dv    = beta * (T.T @ dvni)
            d_ba  = T.T @ dW
            dA    = -strict_lower(T.T @ dAinv @ T.T)
            dL    = beta * dA
            dbeta = sum(T.T@dvni * v) + sum(d_ba * k) + sum(dA * k@k.T)
        dg_wy is unwritten (delta has no decay gradient).
      - scalar_decay=True: g_cs [BHN, C] cumulative log-decay (gated_delta);
        gc = g_cs, decay_exp = exp(gc), L = exp(gc - gc^T). The value path is
        unchanged; the key path and dA pick up the decay, and d_ba/dL absorb it
        so chunk_bwd_dk stays identical:
            dAinv += dW @ (beta*decay_exp*k).T
            d_ba   = (T.T @ dW) * decay_exp
            dL     = beta * dA * L
            dbeta += sum(T.T@dW * k * decay_exp) + sum(dA * k@k.T * L)
            dg_wy  = sum(T.T@dW * beta*decay_exp*k) + rowsum(dLL) - colsum(dLL)
                     with dLL = dA * beta * (k@k.T) * L
    """
    BHN = k.size(0)
    C = hl.specialize(k.size(1))
    D = k.size(2)
    DV = hl.specialize(v.size(2))

    dL_out = torch.empty([BHN, C, C], dtype=k.dtype, device=k.device)
    dv_out = torch.empty([BHN, C, DV], dtype=v.dtype, device=v.device)
    dbeta_out = torch.empty([BHN, C], dtype=torch.float32, device=k.device)
    d_ba_out = torch.empty([BHN, C, D], dtype=torch.float32, device=k.device)
    dg_wy_out = torch.empty([BHN, C], dtype=torch.float32, device=k.device)

    hdt = k.dtype
    for tile_bhn in hl.tile(BHN, block_size=1):
        beta_i = beta[tile_bhn, :].to(torch.float32)  # [1, C]
        T_i = A_inv[tile_bhn, :, :]
        T_t_h = T_i.transpose(-2, -1).to(hdt)
        idx = hl.arange(C)
        strict_lower = (idx[:, None] > idx[None, :]).to(torch.float32)
        if scalar_decay:
            decay = g_cs[tile_bhn, :].to(torch.float32)  # [1, C]
            decay_exp = torch.exp2(decay * RCP_LN2)[:, :, None]  # [1, C, 1]
            causal = (idx[:, None] >= idx[None, :]).to(torch.float32)
            L = causal * torch.exp2((decay[:, :, None] - decay[:, None, :]) * RCP_LN2)

        dAinv = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        dbeta = hl.zeros([tile_bhn, C], dtype=torch.float32)
        dg_wy = hl.zeros([tile_bhn, C], dtype=torch.float32)
        for tile_dv in hl.tile(DV):
            v_dv = v[tile_bhn, :, tile_dv].float()
            dvni_dv = dvni[tile_bhn, :, tile_dv].float()
            betaV = (beta_i[:, :, None] * v_dv).to(hdt)
            dAinv = hl.dot(dvni_dv.to(hdt), betaV.transpose(-2, -1), acc=dAinv)
            d_bv = hl.dot(T_t_h, dvni_dv.to(hdt)).float()
            dv_out[tile_bhn, :, tile_dv] = (d_bv * beta_i[:, :, None]).to(dv_out.dtype)
            dbeta = dbeta + (d_bv * v_dv).sum(-1)

        for tile_d in hl.tile(D):
            k_d = k[tile_bhn, :, tile_d].float()
            dW_d = dW[tile_bhn, :, tile_d]
            if scalar_decay:
                kbg = (beta_i[:, :, None] * decay_exp * k_d).to(hdt)
                dAinv = hl.dot(dW_d.to(hdt), kbg.transpose(-2, -1), acc=dAinv)
                d_ba = hl.dot(T_t_h, dW_d.to(hdt)).float()
                d_ba_out[tile_bhn, :, tile_d] = d_ba * decay_exp
                dbeta = dbeta + (d_ba * k_d * decay_exp).sum(-1)
                dg_wy = dg_wy + (d_ba * beta_i[:, :, None] * decay_exp * k_d).sum(-1)
            else:
                betaK = (beta_i[:, :, None] * k_d).to(hdt)
                dAinv = hl.dot(dW_d.to(hdt), betaK.transpose(-2, -1), acc=dAinv)
                d_ba = hl.dot(T_t_h, dW_d.to(hdt)).float()
                d_ba_out[tile_bhn, :, tile_d] = d_ba
                dbeta = dbeta + (d_ba * k_d).sum(-1)

        dAinv_h = dAinv.to(hdt)
        dA = -hl.dot(hl.dot(T_t_h, dAinv_h).to(hdt), T_t_h).float()
        dA = dA * strict_lower

        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            k_dh = k[tile_bhn, :, tile_d].to(hdt)
            kk = hl.dot(k_dh, k_dh.transpose(-2, -1), acc=kk)

        if scalar_decay:
            dL_out[tile_bhn, :, :] = (dA * beta_i[:, :, None] * L).to(dL_out.dtype)
            dbeta = dbeta + (dA * kk * L).sum(-1)
            dLL = dA * beta_i[:, :, None] * kk * L
            dg_wy = dg_wy + dLL.sum(-1) - dLL.sum(-2)
            dg_wy_out[tile_bhn, :] = dg_wy
        else:
            dL_out[tile_bhn, :, :] = (dA * beta_i[:, :, None]).to(dL_out.dtype)
            dbeta = dbeta + (dA * kk).sum(-1)
        dbeta_out[tile_bhn, :] = dbeta

    return dL_out, dv_out, dbeta_out, d_ba_out, dg_wy_out


@helion.kernel()
def chunk_bwd_dk_delta_helion(
    k: torch.Tensor,
    dL: torch.Tensor,
    d_ba: torch.Tensor,
    beta: torch.Tensor,
    dk_as: torch.Tensor,
) -> torch.Tensor:
    """Assemble the full dk from the WY-backward outputs (a = k shared key). dk_as
    (attn + state) and d_ba/dL come from chunk_bwd_wy_dL_delta_helion; keeping the
    dL @ k matmuls in their own kernel avoids fusing them with the triangular-
    inverse dots (which miscompile at C = 64):
        dk = dk_as + d_ba * beta + dL @ k + dL.T @ k   # [C, D]"""
    BHN = k.size(0)
    C = hl.specialize(k.size(1))
    D = k.size(2)

    dk_out = torch.empty([BHN, C, D], dtype=k.dtype, device=k.device)

    hdt = k.dtype
    for tile_bhn in hl.tile(BHN, block_size=1):
        beta_i = beta[tile_bhn, :].to(torch.float32)
        dL_h = dL[tile_bhn, :, :].to(hdt)
        dLt_h = dL[tile_bhn, :, :].transpose(-2, -1).to(hdt)
        for tile_d in hl.tile(D):
            k_dh = k[tile_bhn, :, tile_d].to(hdt)
            dk = (
                dk_as[tile_bhn, :, tile_d]
                + d_ba[tile_bhn, :, tile_d] * beta_i[:, :, None]
            )
            dk = hl.dot(dL_h, k_dh, acc=dk)
            dk = hl.dot(dLt_h, k_dh, acc=dk)
            dk_out[tile_bhn, :, tile_d] = dk.to(dk_out.dtype)
    return dk_out


@helion.kernel()
def chunk_bwd_o_kda_helion(
    q: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    Aqk: torch.Tensor,
    gc: torch.Tensor,
    do: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adjoint of the kda anchored output o = (q * exp(gc)) @ h + Aqk @ v_new,
    with per-channel decay gc [BHN, C, D]:
        dqg    = do @ h.T                       # [C, D]
        dq_o   = dqg * exp(gc)                  # [C, D]
        dgc_o  = dqg * (scale*q) * exp(gc)      # [C, D]  per-channel decay grad
        dh     = (q * exp(gc)).T @ do           # [D, DV]
        dAqk   = causal(do @ v_new.T)           # [C, C]  s <= t
        dv_new = Aqk.T @ do                     # [C, DV]
    The q/dgc half mirrors the GLA diag output backward; dAqk and dv_new are the
    delta-correction score/value grads."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    K = q.size(2)
    V = v_new.size(2)

    dq_o = torch.empty([BHN, C, K], dtype=torch.float32, device=q.device)
    dgc_o = torch.empty([BHN, C, K], dtype=torch.float32, device=q.device)
    dh = torch.empty([BHN, K, V], dtype=torch.float32, device=q.device)
    dAqk = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    dv_new = torch.empty([BHN, C, V], dtype=torch.float32, device=q.device)

    hdt = q.dtype
    for tile_bhn in hl.tile(BHN, block_size=1):
        idx = hl.arange(C)
        incl = (idx[:, None] >= idx[None, :])[None, :, :]

        Atb = Aqk[tile_bhn, :, :].transpose(-2, -1).to(hdt)
        dA = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_v in hl.tile(V):
            dob = do[tile_bhn, :, tile_v].to(hdt)
            vb = v_new[tile_bhn, :, tile_v].to(hdt)
            dA = dA + hl.dot(dob, vb.transpose(-2, -1))
            dv_new[tile_bhn, :, tile_v] = hl.dot(Atb, dob)
        dAqk[tile_bhn, :, :] = torch.where(incl, dA, 0.0)

        for tile_k in hl.tile(K):
            qt = q[tile_bhn, :, tile_k].to(torch.float32) * scale
            gct = gc[tile_bhn, :, tile_k].to(torch.float32)
            egc = torch.exp2(gct * RCP_LN2)
            qg = (qt * egc).to(hdt)
            qgt_b = qg.transpose(-2, -1)

            dqg = hl.zeros([tile_bhn, C, tile_k], dtype=torch.float32)
            for tile_v in hl.tile(V):
                dob = do[tile_bhn, :, tile_v].to(hdt)
                hb = h[tile_bhn, tile_k, tile_v].to(hdt)
                dqg = hl.dot(dob, hb.transpose(-2, -1), acc=dqg)
                dh[tile_bhn, tile_k, tile_v] = hl.dot(qgt_b, dob)

            dq_o[tile_bhn, :, tile_k] = dqg * egc
            dgc_o[tile_bhn, :, tile_k] = dqg * qt * egc

    return dq_o, dgc_o, dh, dAqk, dv_new


@helion.kernel()
def chunk_bwd_state_du_kda_helion(
    k: torch.Tensor,
    w: torch.Tensor,
    gc: torch.Tensor,
    dv_new_out: torch.Tensor,
    dh_all: torch.Tensor,
    dS_scratch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serial reverse pass of the kda state recurrence's adjoint. Walk i = N-1 -> 0
    carrying dS [K, DV]; per chunk (gc_last = gc[C-1] is the per-channel [K] total):
        du_i       = (k_i * exp(gc_last - gc_i)) @ dS + dv_new_out_i   # [C, DV]
        dS_save[i] = dS                                               # incoming snapshot
        dS         = exp(gc_last)[:, None] * dS - w_i.T @ du_i + dh_all[i]  # [K, DV]
    du feeds the parallel dw/dk/dgc pass; dS_save is the state-grad snapshot."""
    N = k.size(1)
    C = hl.specialize(k.size(2))
    K = k.size(3)
    V = dv_new_out.size(3)
    BH = k.size(0)

    du = torch.empty([BH, N, C, V], dtype=torch.float32, device=k.device)
    dS_save = torch.empty([BH, N, K, V], dtype=torch.float32, device=k.device)

    for tile_bh, tile_v in hl.tile([BH, V], block_size=[1, None]):
        idx = tile_bh.id
        dS = dS_scratch[idx, :, tile_v].to(torch.float32)  # [K, bv]

        for i_rev in hl.grid(N):
            i = N - 1 - i_rev
            dS_save[idx, i, :, tile_v] = dS.to(dS_save.dtype)

            gc_i = gc[idx, i, :, :].to(torch.float32)  # [C, K]
            gc_last = gc[idx, i, C - 1, :].to(torch.float32)  # [K]
            k_i = k[idx, i, :, :].to(torch.float32)  # [C, K]
            w_i = w[idx, i, :, :].to(torch.float32)  # [C, K]
            k_scaled = k_i * torch.exp2((gc_last[None, :] - gc_i) * RCP_LN2)

            dv_new_i = hl.dot(k_scaled.to(w.dtype), dS.to(w.dtype)).float()
            dv_new_i = dv_new_i + dv_new_out[idx, i, :, tile_v].to(torch.float32)
            du[idx, i, :, tile_v] = dv_new_i

            dS = dS * torch.exp2(gc_last * RCP_LN2)[:, None]
            dS = dS - hl.dot(
                w_i.transpose(-2, -1).to(w.dtype), dv_new_i.to(w.dtype)
            ).float()
            dS = dS + dh_all[idx, i, :, tile_v].to(torch.float32)

    return du, dS_save


@helion.kernel()
def chunk_bwd_state_dwk_kda_helion(
    k: torch.Tensor,
    gc: torch.Tensor,
    h_all: torch.Tensor,
    v_new: torch.Tensor,
    du: torch.Tensor,
    dS_save: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parallel phase-2 of the kda state adjoint: dw, dk (state part), dgc (state
    part). Per chunk, gc_last = gc[C-1] is the per-channel [K] total,
    k_scaled = k * exp(gc_last - gc):
        dw        = -du @ S.T                              # [C, K]
        dk_scaled = v_new @ dS_save.T                      # [C, K]
        dk_state  = dk_scaled * exp(gc_last - gc)          # [C, K]  state part of dk
        dgc_pos   = -(dk_scaled * k_scaled)                # [C, K]
        dgc_last  = (dS_save * S).sum(V) * exp(gc_last)    # [K]  last-row anchor term
                    + (dk_scaled * k_scaled).sum(C)
        dgc_state = dgc_pos + (row == C-1) * dgc_last      # [C, K]  per-channel
    """
    BH = k.size(0)
    N = k.size(1)
    BHN = BH * N
    C = hl.specialize(k.size(2))
    K = k.size(3)
    V = v_new.size(3)

    kf = k.reshape(BHN, C, K)
    gcf = gc.reshape(BHN, C, K)
    Sf = h_all.reshape(BHN, K, V)
    vnf = v_new.reshape(BHN, C, V)
    duf = du.reshape(BHN, C, V)
    dSf = dS_save.reshape(BHN, K, V)

    dw = torch.empty([BHN, C, K], dtype=torch.float32, device=k.device)
    dk_s = torch.empty([BHN, C, K], dtype=torch.float32, device=k.device)
    dgc_s = torch.empty([BHN, C, K], dtype=torch.float32, device=k.device)

    for tile_bhn, tile_k in hl.tile([BHN, K]):
        rows = hl.arange(C)
        is_last = (rows == C - 1).to(torch.float32)[None, :, None]

        gc_i = gcf[tile_bhn, :, tile_k].to(torch.float32)  # [C, bk]
        gc_last = gcf[tile_bhn, C - 1, tile_k].to(torch.float32)  # [bk]
        k_i = kf[tile_bhn, :, tile_k].to(torch.float32)
        elast = torch.exp2(gc_last * RCP_LN2)
        escaled = torch.exp2((gc_last[:, None, :] - gc_i) * RCP_LN2)
        k_scaled = k_i * escaled

        dw_acc = hl.zeros([tile_bhn, C, tile_k], dtype=torch.float32)
        dk_scaled = hl.zeros([tile_bhn, C, tile_k], dtype=torch.float32)
        delast = hl.zeros([tile_bhn, tile_k], dtype=torch.float32)
        for tile_v in hl.tile(V):
            S_i = Sf[tile_bhn, tile_k, tile_v].to(torch.float32)
            dS = dSf[tile_bhn, tile_k, tile_v].to(torch.float32)
            v_new_i = vnf[tile_bhn, :, tile_v].to(torch.float32)
            du_i = duf[tile_bhn, :, tile_v]
            dw_acc = dw_acc - hl.dot(du_i.to(k.dtype), S_i.transpose(-2, -1).to(k.dtype))
            dk_scaled = hl.dot(
                v_new_i.to(k.dtype), dS.transpose(-2, -1).to(k.dtype), acc=dk_scaled
            )
            delast = delast + torch.sum(dS * S_i, dim=-1)

        dw[tile_bhn, :, tile_k] = dw_acc
        dk_s[tile_bhn, :, tile_k] = dk_scaled * escaled
        dgc_pos = -(dk_scaled * k_scaled)
        dgc_last = delast * elast + torch.sum(dk_scaled * k_scaled, dim=1)
        dgc_s[tile_bhn, :, tile_k] = dgc_pos + is_last * dgc_last[:, None, :]

    return (
        dw.reshape(BH, N, C, K),
        dk_s.reshape(BH, N, C, K),
        dgc_s.reshape(BH, N, C, K),
    )


@helion.kernel()
def chunk_bwd_wu_kda_helion(
    Tinv: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gc: torch.Tensor,
    Akk: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adjoint of the kda WY / UT transform w = Tinv @ (beta*exp(gc)*k),
    u = Tinv @ (beta*v), reusing the saved inverse Tinv. Per-channel decay
    egc = exp(gc), kbg = beta*egc*k:
        dTinv = dw @ kbg.T + du @ (beta*v).T
        dkbg  = Tinv.T @ dw -> dk_w = dkbg*beta*egc, dgc_kbg = dkbg*kbg
        dvb   = Tinv.T @ du -> dv = dvb*beta
        dN    = strict(Tinv.T @ dTinv @ Tinv.T);  dAkk = -dN*beta
        dbeta = sum(dkbg*k*egc) + sum(dvb*v) + sum(-dN*Akk)
    dAkk defers the k-gram backprop to the anchored gram2 kernel (the anchored
    gram can't be reconstructed with a [C,C] L-mask); the Tinv.T dots stay off any
    dL @ k fusion, preserving the C = 64 split."""
    BHN = k.size(0)
    C = hl.specialize(k.size(1))
    K = k.size(2)
    V = v.size(2)

    dk_w = torch.empty([BHN, C, K], dtype=torch.float32, device=k.device)
    dgc_kbg = torch.empty([BHN, C, K], dtype=torch.float32, device=k.device)
    dv = torch.empty([BHN, C, V], dtype=torch.float32, device=k.device)
    dbeta = torch.empty([BHN, C], dtype=torch.float32, device=k.device)
    dAkk = torch.empty([BHN, C, C], dtype=torch.float32, device=k.device)

    hdt = k.dtype
    for tile_bhn in hl.tile(BHN, block_size=1):
        idx = hl.arange(C)
        strict = (idx[:, None] > idx[None, :])[None, :, :]
        b_beta = beta[tile_bhn, :].to(torch.float32)
        Ttt = Tinv[tile_bhn, :, :].transpose(-2, -1)
        Tttb = Ttt.to(hdt)

        dTinv = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        db = hl.zeros([tile_bhn, C], dtype=torch.float32)
        for tile_k in hl.tile(K):
            kt = k[tile_bhn, :, tile_k].to(torch.float32)
            egc = torch.exp2(gc[tile_bhn, :, tile_k].to(torch.float32) * RCP_LN2)
            kbg = kt * b_beta[:, :, None] * egc
            dwb = dw[tile_bhn, :, tile_k].to(hdt)
            dTinv = hl.dot(dwb, kbg.to(hdt).transpose(-2, -1), acc=dTinv)
            dkbg = hl.dot(Tttb, dwb).to(torch.float32)
            dk_w[tile_bhn, :, tile_k] = dkbg * b_beta[:, :, None] * egc
            dgc_kbg[tile_bhn, :, tile_k] = dkbg * kbg
            db = db + torch.sum(dkbg * kt * egc, dim=-1)
        for tile_v in hl.tile(V):
            vt = v[tile_bhn, :, tile_v].to(torch.float32)
            vb = vt * b_beta[:, :, None]
            dub = du[tile_bhn, :, tile_v].to(hdt)
            dTinv = hl.dot(dub, vb.to(hdt).transpose(-2, -1), acc=dTinv)
            dvb = hl.dot(Tttb, dub).to(torch.float32)
            dv[tile_bhn, :, tile_v] = dvb * b_beta[:, :, None]
            db = db + torch.sum(dvb * vt, dim=-1)

        dN = hl.dot(hl.dot(Ttt, dTinv), Ttt)
        dN = torch.where(strict, dN, 0.0)
        dAkk[tile_bhn, :, :] = -dN * b_beta[:, :, None]
        db = db + torch.sum(-dN * Akk[tile_bhn, :, :], dim=-1)
        dbeta[tile_bhn, :] = db

    return dk_w, dgc_kbg, dv, dbeta, dAkk


@helion.kernel()
def chunk_bwd_gram2_kda_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Anchored adjoint of both intra-chunk grams (Aqk q-rows, Akk k-rows) in one
    pass, mirroring the forward chunk_fwd_A_diag_anchored_helion(build_kk=True).
    For each BC_DIAG=16 sub-block pair (row block r, col block c), anchor gc_n at
    the row block's first row (off-diagonal) or its midpoint (diagonal block); the
    gated row/col operands are the forward's:
        qg  = q_r * exp(gc_r - gc_n)          # [BC, D]
        kgr = k_r * exp(gc_r - gc_n)          # [BC, D]
        kg  = k_c * exp(gc_n - gc_c)          # [BC, D]  shared column operand
    Then, from the incoming score grads dAqk_b, dAkk_b for the block:
        d_qg  = dAqk_b @ kg                    # [BC, D]
        d_kgr = dAkk_b @ kg                    # [BC, D]
        d_kg  = dAqk_b.T @ qg + dAkk_b.T @ kgr # [BC, D]
        dq[r]  += d_qg * exp(gc_r - gc_n)
        dk[r]  += d_kgr * exp(gc_r - gc_n)
        dk[c]  += d_kg * exp(gc_n - gc_c)
        dgc[r] += d_qg * qg + d_kgr * kgr
        dgc[c] += -d_kg * kg                   # per-channel decay grad
    No committed backward exists for the anchored gram, so this is dedicated to kda."""
    BC = BC_DIAG
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    K = q.size(2)
    NC = C // BC

    dq = torch.zeros([BHN, C, K], dtype=torch.float32, device=q.device)
    dk = torch.zeros([BHN, C, K], dtype=torch.float32, device=q.device)
    dgc = torch.zeros([BHN, C, K], dtype=torch.float32, device=q.device)

    block_k = hl.register_block_size(K)
    for tile_bhn, tile_k in hl.tile([BHN, K], block_size=[1, block_k]):
        rows = hl.arange(16)  # BC_DIAG; literal required by hl.arange
        incl = (rows[:, None] >= rows[None, :])[None, :, :]
        strict = (rows[:, None] > rows[None, :])[None, :, :]

        for i_i in range(1, NC):
            r0 = i_i * BC
            gn = gc[tile_bhn, r0, tile_k].to(torch.float32)
            q_i = q[tile_bhn, r0 : r0 + BC, tile_k].to(torch.float32) * scale
            k_i = k[tile_bhn, r0 : r0 + BC, tile_k].to(torch.float32)
            gc_i = gc[tile_bhn, r0 : r0 + BC, tile_k].to(torch.float32)
            erow = torch.exp2((gc_i - gn[:, None, :]) * RCP_LN2)
            qg = q_i * erow
            kgr = k_i * erow
            dq_i = hl.zeros([tile_bhn, BC, tile_k], dtype=torch.float32)
            dk_i = hl.zeros([tile_bhn, BC, tile_k], dtype=torch.float32)
            dgc_i = hl.zeros([tile_bhn, BC, tile_k], dtype=torch.float32)
            for i_j in range(i_i):
                c0 = i_j * BC
                k_j = k[tile_bhn, c0 : c0 + BC, tile_k].to(torch.float32)
                gc_j = gc[tile_bhn, c0 : c0 + BC, tile_k].to(torch.float32)
                ecol = torch.exp2((gn[:, None, :] - gc_j) * RCP_LN2)
                kg = k_j * ecol
                dAqk_b = dAqk[tile_bhn, r0 : r0 + BC, c0 : c0 + BC]
                dAkk_b = dAkk[tile_bhn, r0 : r0 + BC, c0 : c0 + BC]
                d_qg = hl.dot(dAqk_b, kg)
                d_kgr = hl.dot(dAkk_b, kg)
                dq_i = dq_i + d_qg * erow
                dk_i = dk_i + d_kgr * erow
                dgc_i = dgc_i + d_qg * qg + d_kgr * kgr
                d_kg = hl.dot(dAqk_b.transpose(-2, -1), qg) + hl.dot(
                    dAkk_b.transpose(-2, -1), kgr
                )
                dk[tile_bhn, c0 : c0 + BC, tile_k] += d_kg * ecol
                dgc[tile_bhn, c0 : c0 + BC, tile_k] += -d_kg * kg
            dq[tile_bhn, r0 : r0 + BC, tile_k] += dq_i
            dk[tile_bhn, r0 : r0 + BC, tile_k] += dk_i
            dgc[tile_bhn, r0 : r0 + BC, tile_k] += dgc_i

        for i_d in range(NC):
            d0 = i_d * BC
            gn = gc[tile_bhn, d0 + BC // 2, tile_k].to(torch.float32)
            q_d = q[tile_bhn, d0 : d0 + BC, tile_k].to(torch.float32) * scale
            k_d = k[tile_bhn, d0 : d0 + BC, tile_k].to(torch.float32)
            gc_d = gc[tile_bhn, d0 : d0 + BC, tile_k].to(torch.float32)
            erow = torch.exp2((gc_d - gn[:, None, :]) * RCP_LN2)
            ecol = torch.exp2((gn[:, None, :] - gc_d) * RCP_LN2)
            qg = q_d * erow
            kgr = k_d * erow
            kg = k_d * ecol
            dAqk_b = torch.where(incl, dAqk[tile_bhn, d0 : d0 + BC, d0 : d0 + BC], 0.0)
            dAkk_b = torch.where(
                strict, dAkk[tile_bhn, d0 : d0 + BC, d0 : d0 + BC], 0.0
            )
            d_qg = hl.dot(dAqk_b, kg)
            d_kgr = hl.dot(dAkk_b, kg)
            d_kg = hl.dot(dAqk_b.transpose(-2, -1), qg) + hl.dot(
                dAkk_b.transpose(-2, -1), kgr
            )
            dq[tile_bhn, d0 : d0 + BC, tile_k] += d_qg * erow
            dk[tile_bhn, d0 : d0 + BC, tile_k] += d_kgr * erow + d_kg * ecol
            dgc[tile_bhn, d0 : d0 + BC, tile_k] += d_qg * qg + d_kgr * kgr - d_kg * kg

    return dq, dk, dgc


@helion.kernel()
def chunk_bwd_dh_diag_fused(
    q: torch.Tensor,
    do: torch.Tensor,
    g_last: torch.Tensor,
    dh_init: torch.Tensor,
    gc: torch.Tensor | None = None,
    use_g: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
    scalar_decay: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    diag_anchored: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    scale: float = 1.0,
) -> torch.Tensor:
    """Fused state gradient propagation over N chunks in reverse.

    Tiles [BH, D, DV] to parallelize over both key and value dimensions.
    Compile-time decay modes (specialized, no runtime branch):
      - use_g=False:        no decay; q is unscaled so scale folds into the q-row.
      - diagonal (default): q is pre-scaled by exp(gc); g_last is a per-channel
        [BH, N, D] state decay applied as dh_acc * exp(g_last).
      - scalar_decay=True:  q is raw and gc [BH, N, C] scales it in-kernel;
        g_last is a scalar [BH, N] state decay. With g in natural-log space:
            dh_all[i] = exp(g_last[i+1]) * dh_all[i+1]
                        + (scale * exp(gc[i+1]))[:, None] * q[i+1])^T @ do[i+1]
        which the kernel computes as exp2(RCP_LN2 * x) == exp(x), since exp2 is
        one hardware instruction:
            dh_all[i] = exp2(RCP_LN2 * g_last[i+1]) * dh_all[i+1]
                        + (scale * exp2(RCP_LN2 * gc[i+1]))[:, None] * q[i+1])^T @ do[i+1]
      - diag_anchored=True: q is raw and gc [BH, N, C, D] is the per-channel
        cumsum; the decay rides q per channel (exp2 of RCP_LN2 * gc):
            dh_all[i] = exp2(RCP_LN2 * gc_last[i+1]) * dh_all[i+1]
                        + (scale * exp2(RCP_LN2 * gc[i+1]) * q[i+1])^T @ do[i+1]
    """
    BH = q.size(0)
    N = q.size(1)
    C = hl.specialize(q.size(2))
    D = q.size(3)
    DV = do.size(3)

    dh_all = torch.empty([BH, N, D, DV], dtype=dh_init.dtype, device=dh_init.device)

    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        dh_acc = dh_init[idx, tile_d, tile_dv].float()

        for i_t in hl.grid(N):
            i = N - 1 - i_t
            dh_all[idx, i, tile_d, tile_dv] = dh_acc.to(dh_all.dtype)
            if scalar_decay:
                g_i = gc[idx, i, :].float()
                dh_acc = torch.exp2(g_last[idx, i].float() * RCP_LN2) * dh_acc
                q_i = (
                    q[idx, i, :, tile_d].float()
                    * (scale * torch.exp2(g_i * RCP_LN2))[:, None]
                ).to(q.dtype)
            elif diag_anchored:
                gc_i = gc[idx, i, :, tile_d].float()
                gc_last = gc[idx, i, C - 1, tile_d].float()
                dh_acc = dh_acc * torch.exp2(gc_last * RCP_LN2)[:, None]
                q_i = (
                    q[idx, i, :, tile_d].float()
                    * (scale * torch.exp2(gc_i * RCP_LN2))
                ).to(q.dtype)
            elif use_g:
                gl_d = g_last[idx, i, tile_d]
                dh_acc = dh_acc * torch.exp(gl_d)[:, None]
                q_i = q[idx, i, :, tile_d]
            else:
                q_i = q[idx, i, :, tile_d] * scale
            do_i = do[idx, i, :, tile_dv]
            dh_acc = hl.dot(q_i.transpose(-2, -1), do_i, acc=dh_acc)

    return dh_all


@helion.kernel()
def chunk_bwd_dqk_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: torch.Tensor,
    g_last: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    use_g: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dQ, dK for all chunks in parallel (no correction). With
    use_g=False the decay is compiled out, for variants with no decay; q is
    then unscaled, so scale folds into dq and the dk q-term here."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)

    dq_out = torch.empty([BHN, C, D], dtype=q.dtype, device=q.device)
    dk_out = torch.empty([BHN, C, D], dtype=k.dtype, device=k.device)

    for tile_bhn, tile_d in hl.tile([BHN, D]):
        # Accumulate dA (raw, no decay) and cross/state terms across DV tiles
        dA_raw = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        dq_cross_acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
        dk_state_acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)

        for tile_dv in hl.tile(DV):
            dot = do[tile_bhn, :, tile_dv]
            vt = v[tile_bhn, :, tile_dv]
            ht = h[tile_bhn, tile_d, tile_dv]
            dht = dh[tile_bhn, tile_d, tile_dv]

            dA_raw = hl.dot(dot, vt.transpose(-2, -1), acc=dA_raw)
            dq_cross_acc = hl.dot(
                dot, ht.transpose(-2, -1).to(dot.dtype), acc=dq_cross_acc
            )
            dk_state_acc = hl.dot(
                vt, dht.transpose(-2, -1).to(vt.dtype), acc=dk_state_acc
            )

        # Apply decay mask, then combine cross/state terms (decay compiled out
        # when use_g=False).
        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        qt = q[tile_bhn, :, tile_d]
        kt = k[tile_bhn, :, tile_d]
        if use_g:
            gc = g_cs[tile_bhn, :]
            decay_ij = torch.exp(gc[:, :, None] - gc[:, None, :])
            dA = dA_raw * decay_ij * causal
            gl = g_last[tile_bhn]
            exp_gc = torch.exp(gc)[:, :, None]
            exp_gl_minus_gc = torch.exp(gl[:, None] - gc)[:, :, None]
            # Decay the cross/state terms, then fold the add into the accumulator.
            dq_acc = hl.dot(dA.to(kt.dtype), kt, acc=dq_cross_acc * exp_gc)
            dk_acc = hl.dot(
                dA.transpose(-2, -1).to(qt.dtype),
                qt,
                acc=dk_state_acc * exp_gl_minus_gc,
            )
        else:
            dA = dA_raw * causal
            dq_acc = hl.dot(dA.to(kt.dtype), kt, acc=dq_cross_acc) * scale
            dk_acc = hl.dot(
                dA.transpose(-2, -1).to(qt.dtype),
                (qt * scale).to(qt.dtype),
                acc=dk_state_acc,
            )

        dq_out[tile_bhn, :, tile_d] = dq_acc.to(q.dtype)
        dk_out[tile_bhn, :, tile_d] = dk_acc.to(k.dtype)

    return dq_out, dk_out


@helion.kernel()
def chunk_bwd_dqkg_scalar_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    g_last: torch.Tensor | None = None,
    diag_anchored: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    compute_dg: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dQ, dK, and the per-position dg_raw. The DV loop accumulates, shared:
        dA_raw   = do @ v^T          # [C, C]
        dq_cross = do @ h^T          # [C, D]
        dk_state = (v @ dh^T) * exp2(gc_last - gc)
    Then two compile-time decay modes (specialized, no runtime branch):
      - scalar (default): g_cs [BHN, C], g_last [BHN]; decay is a [C, C] mask on dA:
            dA = dA_raw * exp2(gc_t - gc_s) * tril
            dq = scale * (dA @ k + exp2(gc) * dq_cross)
            dk = scale * (dA^T @ q) + dk_state
      - diag_anchored=True: g_cs [BHN, C, D] per channel; decay folds into operands:
            dA = dA_raw * tril
            dq = scale * exp2(gc) * (dA @ (exp2(-gc) * k) + dq_cross)
            dk = scale * exp2(-gc) * (dA^T @ (exp2(gc) * q)) + dk_state
        dg_raw = q * dq - k * dk
    The state-carry decay gradient is folded into dg_raw's last position, then a
    reverse cumsum over the chunk finishes dg in-kernel (scalar callers sum over D).
    compute_dg=False skips the dg work entirely (dg_by_d is left unwritten) for
    variants whose decay needs no gradient, e.g. retention's constant decay.
    """
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)

    dq_out = torch.empty([BHN, C, D], dtype=q.dtype, device=q.device)
    dk_out = torch.empty([BHN, C, D], dtype=k.dtype, device=k.device)
    dg_by_d = torch.empty([BHN, C, D], dtype=torch.float32, device=q.device)

    for tile_bhn, tile_d in hl.tile([BHN, D]):
        dA_raw = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        dq_cross_acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
        dk_state_acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
        dh_h_acc = hl.zeros([tile_bhn, tile_d], dtype=torch.float32)

        for tile_dv in hl.tile(DV):
            dot = do[tile_bhn, :, tile_dv]
            vt = v[tile_bhn, :, tile_dv]
            ht = h[tile_bhn, tile_d, tile_dv]
            dht = dh[tile_bhn, tile_d, tile_dv]

            dA_raw = hl.dot(dot, vt.transpose(-2, -1), acc=dA_raw)
            dq_cross_acc = hl.dot(
                dot, ht.transpose(-2, -1).to(dot.dtype), acc=dq_cross_acc
            )
            dk_state_acc = hl.dot(
                vt, dht.transpose(-2, -1).to(vt.dtype), acc=dk_state_acc
            )
            if compute_dg:
                dh_h_acc += (ht.float() * dht.float()).sum(dim=-1)

        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        qt = q[tile_bhn, :, tile_d]
        kt = k[tile_bhn, :, tile_d]

        if diag_anchored:
            gct = g_cs[tile_bhn, :, tile_d].float()
            gc_last = g_cs[tile_bhn, C - 1, tile_d].float()
            exp_gc = torch.exp2(gct * RCP_LN2)
            exp_neg_gc = torch.exp2(-gct * RCP_LN2)
            exp_gl = torch.exp2(gc_last * RCP_LN2)
            exp_gl_minus_gc = torch.exp2((gc_last[:, None, :] - gct) * RCP_LN2)

            dA = dA_raw * causal[None, :, :]
            dk_state = dk_state_acc * exp_gl_minus_gc
            dq_acc = hl.dot(dA, kt * exp_neg_gc, acc=dq_cross_acc) * exp_gc * scale
            dk_acc = (
                hl.dot(dA.transpose(-2, -1), qt * exp_gc) * exp_neg_gc * scale
                + dk_state
            )
        else:
            gc = g_cs[tile_bhn, :].float()
            gl = g_last[tile_bhn].float()
            exp_gc = torch.exp2(gc * RCP_LN2)[:, :, None]
            exp_gl = torch.exp2(gl * RCP_LN2)[:, None]
            exp_gl_minus_gc = torch.exp2((gl[:, None] - gc) * RCP_LN2)[:, :, None]

            dA = dA_raw * torch.exp2((gc[:, :, None] - gc[:, None, :]) * RCP_LN2) * causal
            dk_state = dk_state_acc * exp_gl_minus_gc
            dq_acc = hl.dot(dA.to(kt.dtype), kt, acc=dq_cross_acc * exp_gc) * scale
            dk_acc = hl.dot(dA.transpose(-2, -1).to(qt.dtype), qt) * scale + dk_state

        dq_out[tile_bhn, :, tile_d] = dq_acc.to(dq_out.dtype)
        dk_out[tile_bhn, :, tile_d] = dk_acc.to(dk_out.dtype)

        if compute_dg:
            dg_raw = dq_acc * qt.float() - dk_acc * kt.float()
            dg_last = exp_gl * dh_h_acc + (dk_state * kt.float()).sum(dim=1)
            is_last = (idx == C - 1).float()
            dg = dg_raw + is_last[None, :, None] * dg_last[:, None, :]
            dg_by_d[tile_bhn, :, tile_d] = hl.cumsum(dg, dim=1, reverse=True)

    return dq_out, dk_out, dg_by_d


@helion.kernel()
def chunk_bwd_dv_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    k_state: torch.Tensor,
    g_cs: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    g_last: torch.Tensor | None = None,
    A: torch.Tensor | None = None,
    use_g: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
    scalar_decay: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    diag_anchored: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute dV for all chunks in parallel (no correction). Compile-time decay
    modes (specialized, no runtime branch):
      - use_g=False:        no decay; intra attention uses q' = scale*q.
      - diagonal (default): k_state is pre-scaled by the state decay on host, and
        the intra attention is masked by exp(gc - gc^T).
      - scalar_decay=True:  k is raw; the state decay exp(g_last - gc) multiplies
        the k @ dh accumulator in-kernel, so no host-materialized k_state is
        needed. With g in natural-log space:
            dv[i] = (scale * exp(gc - gc^T) * tril * (q @ k^T))^T @ do
                    + exp(g_last - gc)[:, None] * (k @ dh)
        computed as exp2(RCP_LN2 * x) == exp(x), since exp2 is one instruction:
            dv[i] = (scale * exp2(RCP_LN2 * (gc - gc^T)) * tril * (q @ k^T))^T @ do
                    + exp2(RCP_LN2 * (g_last - gc))[:, None] * (k @ dh)
      - diag_anchored=True: g_cs [BHN, C, D] is the per-channel cumsum; the state
        decay exp2(RCP_LN2 * (gc_last - gc)) rides k, and the intra term reuses the
        forward's pre-scaled, pre-masked score matrix A instead of recomputing it:
            dv[i] = A^T @ do + exp2(RCP_LN2 * (gc_last - gc)) * k) @ dh
    """
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = do.size(2)

    dv_out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)

    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        dv_acc = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        attn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)

        for tile_d in hl.tile(D):
            kt = k[tile_bhn, :, tile_d]
            dht = dh[tile_bhn, tile_d, tile_dv]

            if scalar_decay:
                attn = hl.dot(q[tile_bhn, :, tile_d], kt.transpose(-2, -1), acc=attn)
                dv_acc = hl.dot(kt, dht.to(kt.dtype), acc=dv_acc)
            elif diag_anchored:
                gct = g_cs[tile_bhn, :, tile_d].float()
                gc_last = g_cs[tile_bhn, C - 1, tile_d].float()
                kg = (
                    kt.float() * torch.exp2((gc_last[:, None, :] - gct) * RCP_LN2)
                ).to(kt.dtype)
                dv_acc = hl.dot(kg, dht.to(kg.dtype), acc=dv_acc)
            else:
                attn = hl.dot(q[tile_bhn, :, tile_d], kt.transpose(-2, -1), acc=attn)
                kst = k_state[tile_bhn, :, tile_d]
                dv_acc = hl.dot(kst, dht.to(kst.dtype), acc=dv_acc)

        # Apply decay mask once after accumulating attn across D tiles
        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        if scalar_decay:
            gc = g_cs[tile_bhn, :].float()
            decay_mask = (
                torch.exp2((gc[:, :, None] - gc[:, None, :]) * RCP_LN2) * causal
            )
            attn = attn * (decay_mask * scale)
            exp_dk = torch.exp2((g_last[tile_bhn].float()[:, None] - gc) * RCP_LN2)
            dv_acc = dv_acc * exp_dk[:, :, None]
        elif diag_anchored:
            attn = A[tile_bhn, :, :]
        elif use_g:
            gc = g_cs[tile_bhn, :]
            decay_ij = torch.exp(gc[:, :, None] - gc[:, None, :])
            attn = attn * decay_ij * causal
        else:
            attn = attn * (causal * scale)

        dot = do[tile_bhn, :, tile_dv]
        dv_acc = hl.dot(attn.transpose(-2, -1).to(dot.dtype), dot, acc=dv_acc)

        dv_out[tile_bhn, :, tile_dv] = dv_acc.to(dv_out.dtype)

    return dv_out


# Sub-block size for the anchored intra-chunk score matmul, matching FLA's
BC_DIAG = 16


@helion.kernel()
def chunk_fwd_A_diag_anchored_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    scale: float,
    build_kk: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchored intra-chunk score matrix. Rows split into BC_DIAG-row sub-blocks;
    gc_n is gc at the sub-block's first row (the decay anchor):
        qg = q_blk * exp2(gc_blk - gc_n)        # [BC_DIAG, D]
        kg = k     * exp2(gc_n   - gc)          # [C, D]
        a  = (qg @ kg.T) * scale                # [BC_DIAG, C]
        A[tile_row] = a * causal                # [BC_DIAG, C], keep t >= s
    Anchoring at gc_n bounds each exponent to BC_DIAG rows, so exp2 stays in range.
    With build_kk=True (kda) it also emits the strictly-lower k-rows Gram Akk,
    kg_row = k_blk * exp2(gc_blk - gc_n), Akk[tile_row] = (kg_row @ kg.T) * strict,
    sharing the anchored kg column operand; Akk feeds the WY/UT transform (fp32).
    Akk is always returned (zeros when build_kk=False; callers ignore it)."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))

    A = torch.zeros([BHN, C, C], dtype=torch.float32, device=q.device)
    Akk = torch.zeros([BHN, C, C], dtype=torch.float32, device=q.device)

    for tile_bhn in hl.tile(BHN, block_size=1):
        kt = k[tile_bhn, :, :].float()
        gct = gc[tile_bhn, :, :].float()
        idx = hl.arange(C)

        for tile_row in hl.tile(C, block_size=BC_DIAG):
            row_begin = tile_row.begin
            row_end = row_begin + BC_DIAG
            q_blk = q[tile_bhn, tile_row, :].float()
            gc_blk = gc[tile_bhn, tile_row, :].float()
            gc_n = gc[tile_bhn, row_begin, :].float()

            in_range = (idx < row_end)[None, :]
            erow = torch.exp2((gc_blk - gc_n[:, None, :]) * RCP_LN2)
            qg = q_blk * erow
            expo_k = torch.where(
                in_range[:, :, None],
                (gc_n[:, None, :] - gct) * RCP_LN2,
                torch.zeros_like(gct),
            )
            kg = kt * torch.exp2(expo_k)
            kgt = kg.transpose(-2, -1)
            a = hl.dot(qg, kgt) * scale

            causal = tile_row.index[:, None] >= idx[None, :]
            A[tile_bhn, tile_row, :] = a * causal[None, :, :].to(a.dtype)

            if build_kk:
                kg_row = k[tile_bhn, tile_row, :].float() * erow
                akk = hl.dot(kg_row, kgt)
                strict = tile_row.index[:, None] > idx[None, :]
                Akk[tile_bhn, tile_row, :] = akk * strict[None, :, :].to(akk.dtype)

    return A, Akk


@helion.kernel()
def chunk_fwd_o_diag_anchored_helion(
    q: torch.Tensor,
    v: torch.Tensor,
    gc: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Per-chunk parallel output. A is the pre-masked intra-chunk score matrix:
        qg = q * exp2(gc)               # [C, D]
        o_cross = (qg @ h) * scale      # [C, DV]
        o_intra = A @ v                 # [C, DV]
        out = o_cross + o_intra         # [C, DV]"""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)

    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)

    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)

        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d].float()
            gct = gc[tile_bhn, :, tile_d]
            qg = (qt * torch.exp2(gct * RCP_LN2)).to(q.dtype)
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qg, ht.to(qg.dtype), acc=o_cross)
        o_cross = o_cross * scale

        vt = v[tile_bhn, :, tile_dv]
        At = A[tile_bhn, :, :]
        o_intra = hl.dot(At.to(vt.dtype), vt)
        out[tile_bhn, :, tile_dv] = (o_cross + o_intra).to(out.dtype)

    return out


# Autograd integration
# ════════════════════════════════════════════════════════════════════════════════


class ChunkedLinearAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor | None,
        a: torch.Tensor | None,
        C: int,
        initial_state: torch.Tensor | None,
        return_final_state: bool,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tensors = [q, k, v]
        ctx.has_g = g is not None
        ctx.has_beta = beta is not None
        ctx.has_a = a is not None
        ctx.scale = scale
        if g is not None:
            tensors.append(g)
        if beta is not None:
            tensors.append(beta)
        if a is not None:
            tensors.append(a)
        ctx.save_for_backward(*tensors)
        ctx.C = C

        input_dtype = q.dtype
        if beta is not None:
            # Delta rule (beta correction): the decay mode is read from g
            # inside the shared host (None / scalar / per-channel). A separate
            # a-tensor is only supported without decay.
            if g is not None and a is not None:
                raise NotImplementedError(
                    "beta correction with decay requires a=None"
                )
            o, h_all, v_new_all, A_inv, w_wy, final_state = (
                _helion_chunked_fwd_delta(
                    q, k, v, beta, a, C, g=g,
                    initial_state=initial_state,
                    return_final_state=return_final_state,
                )
            )
            ctx.h_all = h_all
            ctx.v_new_all = v_new_all
            ctx.A_inv = A_inv
            ctx.w_wy = w_wy
        else:
            o, h_all, final_state = _helion_chunked_fwd(
                q,
                k,
                v,
                g,
                C,
                initial_state=initial_state,
                return_final_state=return_final_state,
                scale=scale,
            )
            ctx.h_all = h_all
            ctx.v_new_all = None

        return o.to(input_dtype), final_state

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
        _grad_final_state: object,
    ) -> tuple[  # type: ignore[override]
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
    ]:
        tensors = ctx.saved_tensors
        q, k, v = tensors[:3]
        idx = 3
        g = tensors[idx] if ctx.has_g else None
        if ctx.has_g:
            idx += 1
        beta = tensors[idx] if ctx.has_beta else None
        if ctx.has_beta:
            idx += 1
        a = tensors[idx] if ctx.has_a else None
        C = ctx.C
        h_all = ctx.h_all
        v_new_all = ctx.v_new_all

        if not ctx.has_beta:
            dq, dk, dv, dg = _helion_chunked_bwd(
                q,
                k,
                v,
                g,
                grad_output,
                C,
                h_all=h_all,
                scale=ctx.scale,
                needs_dg=ctx.needs_input_grad[3],
            )
            return dq, dk, dv, dg, None, None, None, None, None, None

        A_inv = ctx.A_inv
        w_wy = ctx.w_wy

        # Delta rule: the decay mode is read from g, mirroring the forward. KDA
        # (diagonal decay) has its own kernels; DeltaNet and Gated DeltaNet share
        # one host that returns grads in q,k,v,g,beta,a order.
        if g is not None and g.dim() == 4:
            dq, dk, dv, dg, dbeta = _helion_chunked_bwd_kda(
                q,
                k,
                v,
                g,
                beta,  # pyrefly: ignore
                grad_output,
                C,
                h_all=h_all,
                v_new_all=v_new_all,
                A_inv=A_inv,
                w_wy=w_wy,
            )
            return dq, dk, dv, dg, dbeta, None, None, None, None, None

        dq, dk, dv, dg, dbeta, da = _helion_chunked_bwd_delta(
            q,
            k,
            v,
            beta,  # pyrefly: ignore
            a,
            grad_output,
            C,
            h_all=h_all,
            v_new_all=v_new_all,
            A_inv=A_inv,
            w_wy=w_wy,
            g=g,
        )
        return dq, dk, dv, dg, dbeta, da, None, None, None, None


# ════════════════════════════════════════════════════════════════════════════════
# Forward / backward pipelines
# ════════════════════════════════════════════════════════════════════════════════


def _init_state(
    initial_state: torch.Tensor | None,
    BH: int,
    D: int,
    DV: int,
    ref_tensor: torch.Tensor,
) -> torch.Tensor:
    if initial_state is not None:
        return initial_state.reshape(BH, D, DV).float().contiguous()
    return ref_tensor.new_zeros(BH, D, DV, dtype=torch.float32)


def _final_state_from_h_all(
    h_last: torch.Tensor,
    k_state_last: torch.Tensor,
    v_last: torch.Tensor,
    g_last: torch.Tensor | None,
    B: int,
    H: int,
    use_g: bool = True,
) -> torch.Tensor:
    h_final = h_last.float()
    if use_g:
        assert g_last is not None
        h_final = h_final * torch.exp(g_last).unsqueeze(-1)
    h_final = h_final + torch.bmm(
        k_state_last.float().transpose(-2, -1),
        v_last.float(),
    )
    return h_final.reshape(B, H, h_final.shape[1], h_final.shape[2])


def _helion_chunked_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    C: int,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, H, T, D = q.shape
    DV = v.shape[-1]
    N = T // C
    BH = B * H
    BHN = BH * N

    if g is None:
        # No-decay path: the in-kernel decay ops are skipped and compiled out via use_g=False.
        k_state_4d = k.reshape(BH, N, C, D)
        v_flat = v.reshape(BH, N, C, DV)
        state = _init_state(initial_state, BH, D, DV, q)
        h_all = chunk_fwd_h_diag_fused(k_state_4d, v_flat, None, state, use_g=False)

        qf = q.reshape(BHN, C, D)
        kf = k.reshape(BHN, C, D)
        vf2 = v_flat.reshape(BHN, C, DV)
        hf2 = h_all.reshape(BHN, D, DV)
        o = chunk_fwd_o_helion(qf, kf, vf2, None, hf2, use_g=False, scale=scale)

        h_all._scalar_bwd_cache = None  # pyrefly: ignore
        final_state = None
        if return_final_state:
            final_state = _final_state_from_h_all(
                h_all[:, -1], k_state_4d[:, -1], v_flat[:, -1], None, B, H, use_g=False
            )
        return o.reshape(B, H, T, DV), h_all, final_state

    scalar_decay = g.dim() == 3

    if scalar_decay:
        # Scalar decay path.
        gc = g.float().reshape(BH, N, C)
        g_cs = gc.cumsum(-1)  # [BH, N, C]
        g_last = g_cs[:, :, -1]  # [BH, N]

        k_4d = k.reshape(BH, N, C, D)
        v_flat = v.reshape(BH, N, C, DV)
        state = _init_state(initial_state, BH, D, DV, q)
        h_all = chunk_fwd_h_diag_fused(
            k_4d, v_flat, g_last, state, gc=g_cs, scalar_decay=True
        )

        # Output kernel: pass raw q, k with g_cs for decay; scale folds in here.
        qf = q.reshape(BHN, C, D)
        kf = k.reshape(BHN, C, D)
        vf2 = v_flat.reshape(BHN, C, DV)
        g_csf = g_cs.reshape(BHN, C)
        hf2 = h_all.reshape(BHN, D, DV)

        o = chunk_fwd_o_helion(qf, kf, vf2, g_csf, hf2, scale=scale)

        # Attach cached data to h_all for the backward to use
        g_last_4d = g_last.unsqueeze(-1).expand(-1, -1, D)
        h_all._scalar_bwd_cache = (  # pyrefly: ignore
            g_cs,
            g_last,
            g_last_4d,
        )

        final_state = None
        if return_final_state:
            # Decay the last chunk's keys: k * exp(g_last - g_cs).
            k_state_last = (
                k_4d[:, -1].float()
                * torch.exp(g_last[:, -1, None, None] - g_cs[:, -1, :, None])
            ).to(k.dtype)
            final_state = _final_state_from_h_all(
                h_all[:, -1], k_state_last, v_flat[:, -1], g_last_4d[:, -1], B, H
            )

        return o.reshape(B, H, T, DV), h_all, final_state

    # Diagonal decay path.
    gc4 = g.reshape(BH, N, C, D).float().cumsum(-2)
    gc = gc4.reshape(BHN, C, D)

    k_4d = k.reshape(BH, N, C, D)
    v_4d = v.reshape(BH, N, C, DV)
    g_last_4d = gc4[:, :, -1, :]
    state = _init_state(initial_state, BH, D, DV, q)
    h_all = chunk_fwd_h_diag_fused(
        k_4d, v_4d, g_last_4d, state, gc=gc4, diag_anchored=True
    )

    qf = q.reshape(BHN, C, D)
    kf = k.reshape(BHN, C, D)
    vf = v.reshape(BHN, C, DV)
    hf = h_all.reshape(BHN, D, DV)
    A, _ = chunk_fwd_A_diag_anchored_helion(qf, kf, gc, scale)
    o = chunk_fwd_o_diag_anchored_helion(qf, vf, gc, hf, A, scale)

    # Attach cached data to h_all for the backward to use.
    h_all._diag_bwd_cache = (gc, A)  # pyrefly: ignore

    final_state = None
    if return_final_state:
        k_state_last = (
            k_4d[:, -1].float() * torch.exp(gc4[:, -1, -1:, :] - gc4[:, -1])
        ).to(k.dtype)
        final_state = _final_state_from_h_all(
            h_all[:, -1], k_state_last, v_4d[:, -1], g_last_4d[:, -1], B, H
        )

    return o.reshape(B, H, T, DV), h_all, final_state


def _helion_chunked_fwd_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor | None,
    C: int,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor | None,
]:
    """Chunked delta-rule forward (beta correction), dispatched by the decay in g:
      - g is None    -> DeltaNet: beta correction, no decay.
      - g.dim() == 3 -> Gated DeltaNet: beta correction + scalar decay.
      - g.dim() == 4 -> KDA: beta correction + diagonal per-channel decay.

    One pipeline in every mode: the WY / UT transform (Neumann doubling), the
    serial state pass (v_new = u - w S; decayed carry), and the output kernel.
    The decay folds into the shared kernels via scalar_decay / diag_anchored;
    KDA additionally builds the anchored grams (Aqk, Akk) and uses the anchored
    output kernel. q is pre-scaled by the caller. Returns the triangular inverse
    T (A_inv) and w for the backward, plus the final state when requested (which
    uses the corrected values v_new, not v).
    """
    B, H, T, D = q.shape
    DV = v.shape[-1]
    N = T // C
    BH = B * H
    BHN = BH * N

    scalar_decay = g is not None and g.dim() == 3
    diag_anchored = g is not None and g.dim() == 4

    # Per-chunk cumulative log-decay (natural log; kernels apply RCP_LN2 at exp2).
    # g_cs / decay_last are None with no decay, [.,C] scalar, or [.,C,D] diagonal.
    g_cs = decay_last = g_cs_flat = None
    if scalar_decay:
        g_cs = g.float().reshape(BH, N, C).cumsum(-1)
        decay_last = g_cs[:, :, -1]
        g_cs_flat = g_cs.reshape(BHN, C)
    elif diag_anchored:
        g_cs = g.float().reshape(BH, N, C, D).cumsum(-2)
        decay_last = g_cs[:, :, -1, :]
        g_cs_flat = g_cs.reshape(BHN, C, D)

    a_use = a if a is not None else k
    qf = q.reshape(BHN, C, D)
    kf = k.reshape(BHN, C, D)
    af = a_use.reshape(BHN, C, D)
    vf = v.reshape(BHN, C, DV)
    bf = beta.reshape(BHN, C)

    # KDA needs the anchored k-gram Akk to feed the WY transform (its q-gram Aqk
    # weights the output below); the scalar / no-decay modes form k @ k in-kernel.
    Aqk = Akk = None
    if diag_anchored:
        Aqk, Akk = chunk_fwd_A_diag_anchored_helion(qf, kf, g_cs_flat, 1.0, build_kk=True)

    w, u, A_inv = chunk_fwd_wy_delta_helion(
        af, vf, bf, g_cs_flat, Akk,
        scalar_decay=scalar_decay, diag_anchored=diag_anchored,
    )

    k4 = k.reshape(BH, N, C, D)
    w4 = w.reshape(BH, N, C, D)
    u4 = u.reshape(BH, N, C, DV)
    state = _init_state(initial_state, BH, D, DV, q).to(k.dtype)
    h_all, v_new_all = chunk_fwd_h_delta_helion(
        k4, w4, u4, state, g_cs, decay_last,
        scalar_decay=scalar_decay, diag_anchored=diag_anchored,
    )

    hf = h_all.reshape(BHN, D, DV)
    vnewf = v_new_all.reshape(BHN, C, DV)

    if diag_anchored:
        # o = (q * exp2(gc)) @ h + Aqk @ v_new (anchored, per-channel decay).
        o = chunk_fwd_o_diag_anchored_helion(qf, vnewf, g_cs_flat, hf, Aqk, 1.0)
    else:
        o = chunk_fwd_o_helion(qf, kf, vnewf, g_cs_flat, hf, use_g=scalar_decay)

    final_state = None
    if return_final_state:
        # Add the last chunk's writes to its entering state; keys are decayed to
        # the chunk end (per-channel when diagonal), values are the corrected v_new.
        if diag_anchored:
            k_state_last = (
                k4[:, -1].float() * torch.exp(g_cs[:, -1, -1:, :] - g_cs[:, -1])
            ).to(k.dtype)
            g_last_arg = decay_last[:, -1]
        elif scalar_decay:
            k_state_last = (
                k4[:, -1].float()
                * torch.exp(decay_last[:, -1, None, None] - g_cs[:, -1, :, None])
            ).to(k.dtype)
            g_last_arg = decay_last.unsqueeze(-1).expand(-1, -1, D)[:, -1]
        else:
            k_state_last = a_use.reshape(BH, N, C, D)[:, -1]
            g_last_arg = None
        final_state = _final_state_from_h_all(
            h_all[:, -1], k_state_last, v_new_all[:, -1], g_last_arg, B, H,
            use_g=g is not None,
        )

    return o.reshape(B, H, T, DV), h_all, v_new_all, A_inv, w, final_state


def _helion_chunked_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    grad_output: torch.Tensor,
    C: int,
    h_all: torch.Tensor | None = None,
    scale: float = 1.0,
    needs_dg: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, H, T, D = q.shape
    DV = v.shape[-1]
    N = T // C
    BH = B * H
    BHN = BH * N

    if g is None:
        # No-decay backward: no g_cs/q_scaled prescale, decay compiled out of
        # the kernels, and no dg gradient.
        v_flat = v.reshape(BH, N, C, DV)
        do_flat = grad_output.reshape(BH, N, C, DV)
        q_4d = q.reshape(BH, N, C, D)

        if h_all is None:
            k_state_4d = k.reshape(BH, N, C, D)
            state = q.new_zeros(BH, D, DV, dtype=torch.float32)
            h_all = chunk_fwd_h_diag_fused(k_state_4d, v_flat, None, state, use_g=False)

        dstate = q.new_zeros(BH, D, DV, dtype=torch.float32)
        dh_all = chunk_bwd_dh_diag_fused(
            q_4d, do_flat, None, dstate, use_g=False, scale=scale
        )

        g_csf2 = q.new_zeros(BHN, C, dtype=torch.float32)
        g_lastf2 = q.new_zeros(BHN, dtype=torch.float32)
        dhf2 = dh_all.reshape(BHN, D, DV)
        qb = q.reshape(BHN, C, D)
        kb = k.reshape(BHN, C, D)
        vb = v.reshape(BHN, C, DV)
        dob = grad_output.reshape(BHN, C, DV)
        hb = h_all.reshape(BHN, D, DV)

        dq_raw, dk_raw = chunk_bwd_dqk_helion(
            qb, kb, vb, g_csf2, g_lastf2, hb, dob, dhf2, use_g=False, scale=scale
        )
        dv_raw = chunk_bwd_dv_helion(
            qb, kb, kb, g_csf2, dob, dhf2, use_g=False, scale=scale
        )

        return (
            dq_raw.reshape(B, H, T, D),
            dk_raw.reshape(B, H, T, D),
            dv_raw.reshape(B, H, T, DV),
            None,
        )

    diagonal_decay = g.dim() == 4

    if not diagonal_decay:
        # Scalar decay path — use cached data from forward when available,
        # and defer float conversion to the kernels that need it.
        bwd_cache = (
            getattr(h_all, "_scalar_bwd_cache", None) if h_all is not None else None
        )
        if bwd_cache is not None:
            g_cs, g_last_scalar, g_last_4d = bwd_cache
        else:
            gc = g.float().reshape(BH, N, C)
            g_cs = gc.cumsum(-1)
            g_last_scalar = g_cs[:, :, -1]
            g_last_4d = g_last_scalar.unsqueeze(-1).expand(-1, -1, D)

        if h_all is None:
            k_state_4d = k.float().reshape(BH, N, C, D) * torch.exp(
                g_last_scalar[:, :, None, None] - g_cs[:, :, :, None]
            )
            state = q.new_zeros(BH, D, DV, dtype=torch.float32)
            vf = v.float().reshape(BH, N, C, DV)
            h_all = chunk_fwd_h_diag_fused(k_state_4d, vf, g_last_4d, state)

        q_4d = q.reshape(BH, N, C, D)
        do_4d = grad_output.reshape(BH, N, C, DV)
        dstate = q.new_zeros(BH, D, DV, dtype=torch.float32)
        dh_all = chunk_bwd_dh_diag_fused(
            q_4d, do_4d, g_last_scalar, dstate, gc=g_cs, scalar_decay=True, scale=scale
        )

        g_csf2 = g_cs.reshape(BHN, C)
        g_lastf2 = g_last_scalar.reshape(BHN)
        dhf2 = dh_all.reshape(BHN, D, DV)

        qb = q.reshape(BHN, C, D)
        kb = k.reshape(BHN, C, D)
        vb = v.reshape(BHN, C, DV)
        dob = grad_output.reshape(BHN, C, DV)
        hb = h_all.reshape(BHN, D, DV)

        dq_raw, dk_raw, dg_by_d = chunk_bwd_dqkg_scalar_helion(
            qb, kb, vb, g_csf2, hb, dob, dhf2, g_last=g_lastf2,
            compute_dg=needs_dg, scale=scale,
        )

        dv_raw = chunk_bwd_dv_helion(
            qb,
            kb,
            kb,
            g_csf2,
            dob,
            dhf2,
            g_last=g_lastf2,
            scalar_decay=True,
            scale=scale,
        )

        dg = dg_by_d.sum(-1).reshape(B, H, T).to(g.dtype) if needs_dg else None

        return (
            dq_raw.reshape(B, H, T, D),
            dk_raw.reshape(B, H, T, D),
            dv_raw.reshape(B, H, T, DV),
            dg,
        )

    # Diagonal decay path — reuse the forward's cached gc and A when available.
    bwd_cache = getattr(h_all, "_diag_bwd_cache", None) if h_all is not None else None
    if bwd_cache is not None:
        gc, A = bwd_cache
    else:
        gc4 = g.reshape(BH, N, C, D).float().cumsum(-2)
        gc = gc4.reshape(BHN, C, D)
        A, _ = chunk_fwd_A_diag_anchored_helion(
            q.reshape(BHN, C, D), k.reshape(BHN, C, D), gc, scale
        )

    gc4 = gc.reshape(BH, N, C, D)
    g_last_4d = gc4[:, :, -1, :]
    v_4d = v.reshape(BH, N, C, DV)

    if h_all is None:
        state = q.new_zeros(BH, D, DV, dtype=torch.float32)
        h_all = chunk_fwd_h_diag_fused(
            k.reshape(BH, N, C, D), v_4d, g_last_4d, state, gc=gc4, diag_anchored=True
        )

    q_4d = q.reshape(BH, N, C, D)
    do_4d = grad_output.reshape(BH, N, C, DV)
    dstate = q.new_zeros(BH, D, DV, dtype=torch.float32)
    dh_all = chunk_bwd_dh_diag_fused(
        q_4d, do_4d, g_last_4d, dstate, gc=gc4, diag_anchored=True, scale=scale
    )

    qb = q.reshape(BHN, C, D)
    kb = k.reshape(BHN, C, D)
    vb = v.reshape(BHN, C, DV)
    dob = grad_output.reshape(BHN, C, DV)
    hb = h_all.reshape(BHN, D, DV)
    dhf2 = dh_all.reshape(BHN, D, DV)

    dq_raw, dk_raw, dg_by_d = chunk_bwd_dqkg_scalar_helion(
        qb, kb, vb, gc, hb, dob, dhf2, diag_anchored=True, scale=scale
    )

    dv_raw = chunk_bwd_dv_helion(
        qb, kb, kb, gc, dob, dhf2, A=A, diag_anchored=True, scale=scale
    )

    dg = dg_by_d.reshape(B, H, T, D).to(g.dtype)

    return (
        dq_raw.reshape(B, H, T, D),
        dk_raw.reshape(B, H, T, D),
        dv_raw.reshape(B, H, T, DV),
        dg,
    )


def _helion_chunked_bwd_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor | None,
    grad_output: torch.Tensor,
    C: int,
    h_all: torch.Tensor,
    v_new_all: torch.Tensor,
    A_inv: torch.Tensor,
    w_wy: torch.Tensor,
    g: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
]:
    """DeltaNet / Gated DeltaNet backward (beta correction), q pre-scaled.

    Both share the same four kernels, reusing the saved triangular inverse
    A_inv = T; scalar decay (g.dim() == 3) folds in via scalar_decay=True:
      - dstate: reverse state pass -> dS_future, dvni (grad of v_new)
      - dqkw:   per-chunk dW, dq, dk_as, and (with decay) dg_p, dg_last
      - wy_dL:  backprop the WY solve -> dv, dbeta, dL/d_ba, and (with decay) dg_wy
      - dk:     assemble the full key grad = dk_as + d_ba*beta + dL @ k
    With decay, dg = reverse-cumsum over each chunk of (dg_p + dg_wy + dg_last).
    Returns (dq, dk, dv, dg, dbeta, da); dk/da/dg are None when they do not apply.
    """
    B, H, T, D = q.shape
    DV = v.shape[-1]
    N = T // C
    BH = B * H
    BHN = BH * N

    scalar_decay = g is not None
    has_a = a is not None
    a_use = a if a is not None else k

    g_cs = decay_last = decay_lastf = None
    if scalar_decay:
        g_cs = g.float().reshape(BH, N, C).cumsum(-1)
        decay_last = g_cs[:, :, -1].contiguous()  # [BH, N]
        decay_lastf = decay_last.reshape(BHN)
        g_cs = g_cs.reshape(BH, N, C)
    g_cs_flat = g_cs.reshape(BHN, C) if scalar_decay else None

    a4 = a_use.reshape(BH, N, C, D)
    w4 = w_wy.reshape(BH, N, C, D)
    q4 = q.reshape(BH, N, C, D)
    do4 = grad_output.reshape(BH, N, C, DV)

    dh0 = q.new_zeros(BH, D, DV, dtype=k.dtype)
    dS_future, dvni4 = chunk_bwd_dstate_delta_helion(
        q4, a4, w4, do4, dh0, g_cs, decay_last, scalar_decay=scalar_decay
    )

    qf = q.reshape(BHN, C, D)
    af = a_use.reshape(BHN, C, D)
    vf = v.reshape(BHN, C, DV)
    bf = beta.reshape(BHN, C)
    hf = h_all.reshape(BHN, D, DV)
    vnewf = v_new_all.reshape(BHN, C, DV)
    dof = do4.reshape(BHN, C, DV)
    dSf = dS_future.reshape(BHN, D, DV)
    dvnif = dvni4.reshape(BHN, C, DV)

    dW, dq, dk_as, dg_p, dg_last = chunk_bwd_dqkw_delta_helion(
        qf, af, hf, vnewf, dof, dvnif, dSf, g_cs_flat, decay_lastf,
        scalar_decay=scalar_decay,
    )

    dL, dv, dbeta, d_ba, dg_wy = chunk_bwd_wy_dL_delta_helion(
        af, vf, bf, A_inv, dW, dvnif, g_cs_flat, scalar_decay=scalar_decay
    )
    dk_full = chunk_bwd_dk_delta_helion(af, dL, d_ba, bf, dk_as)

    dq_out = dq.float().reshape(B, H, T, D)
    dv_out = dv.float().reshape(B, H, T, DV)
    dbeta_out = dbeta.reshape(B, H, T)

    if not scalar_decay:
        # DeltaNet: dk_full is the grad for a (== k when a is None); no dg.
        da_total = dk_full.float().reshape(B, H, T, D)
        if has_a:
            return dq_out, None, dv_out, None, dbeta_out, da_total
        return dq_out, da_total, dv_out, None, dbeta_out, None

    # Gated DeltaNet: a is k, so dk_full is dk; assemble dg.
    dg_total = (dg_p + dg_wy).reshape(BH, N, C)
    idx = torch.arange(C, device=q.device)
    is_last = (idx == C - 1).to(dg_total.dtype)
    dg_total = dg_total + is_last[None, None, :] * dg_last.reshape(BH, N)[:, :, None]
    dg = dg_total.flip(-1).cumsum(-1).flip(-1)

    dk_out = dk_full.float().reshape(B, H, T, D)
    dg_out = dg.reshape(B, H, T)
    return dq_out, dk_out, dv_out, dg_out, dbeta_out, None


def _helion_chunked_bwd_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    grad_output: torch.Tensor,
    C: int,
    h_all: torch.Tensor,
    v_new_all: torch.Tensor,
    A_inv: torch.Tensor,
    w_wy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """KDA backward (beta correction + diagonal decay), q pre-scaled.

    Chains the kda backward kernels, reusing the saved Tinv (A_inv), w, h_all,
    v_new from the forward and recomputing the anchored grams Aqk/Akk:
      - bwd_o     -> dq_o, dgc_o, dh, dAqk, dv_new
      - bwd_state -> dw, du, dk_state, dgc_state  (serial du pass + parallel dwk)
      - bwd_wu    -> dk_w, dgc_kbg, dv, dbeta, dAkk
      - bwd_gram2 -> dq_gram, dk_gram, dgc_gram
    dq/dk/dgc sum the contributions; dg = reverse-cumsum over each chunk of dgc.
    """
    B, H, T, D = q.shape
    DV = v.shape[-1]
    N = T // C
    BH = B * H
    BHN = BH * N

    # Cumulative per-chunk log-decay: the state kernels (fed gc4) and the WY /
    # output / gram kernels (fed g_cs_flat) all expect the cumsum, not raw g.
    gc4 = g.float().reshape(BH, N, C, D).cumsum(-2)
    g_cs_flat = gc4.reshape(BHN, C, D)

    qf = q.reshape(BHN, C, D)
    kf = k.reshape(BHN, C, D)
    vf = v.reshape(BHN, C, DV)
    bf = beta.reshape(BHN, C)
    hf = h_all.reshape(BHN, D, DV)
    vnf = v_new_all.reshape(BHN, C, DV)
    Tinv = A_inv
    dof = grad_output.reshape(BHN, C, DV)

    # Recompute the anchored grams (cheap; needed for dv_new/dAqk and dbeta/gram2).
    Aqk, Akk = chunk_fwd_A_diag_anchored_helion(qf, kf, g_cs_flat, 1.0, build_kk=True)

    dq_o, dgc_o, dh, dAqk, dv_new = chunk_bwd_o_kda_helion(
        qf, vnf, hf, Aqk, g_cs_flat, dof, 1.0
    )

    k4 = k.reshape(BH, N, C, D)
    w4 = w_wy.reshape(BH, N, C, D)
    h4 = h_all.reshape(BH, N, D, DV)
    vn4 = v_new_all.reshape(BH, N, C, DV)
    dvnew4 = dv_new.reshape(BH, N, C, DV)
    dh4 = dh.reshape(BH, N, D, DV)
    dS_scratch = q.new_zeros(BH, D, DV, dtype=torch.float32)
    du4, dS_save = chunk_bwd_state_du_kda_helion(k4, w4, gc4, dvnew4, dh4, dS_scratch)
    dw4, dk_s4, dgc_s4 = chunk_bwd_state_dwk_kda_helion(
        k4, gc4, h4, vn4, du4, dS_save
    )
    dw = dw4.reshape(BHN, C, D)
    du = du4.reshape(BHN, C, DV)
    dk = dk_s4.reshape(BHN, C, D)
    dgc = dgc_o + dgc_s4.reshape(BHN, C, D)

    dk_w, dgc_kbg, dv, dbeta, dAkk = chunk_bwd_wu_kda_helion(
        Tinv, kf, vf, bf, g_cs_flat, Akk, dw, du
    )
    dk = dk + dk_w
    dgc = dgc + dgc_kbg

    dq_gram, dk_gram, dgc_gram = chunk_bwd_gram2_kda_helion(
        qf, kf, g_cs_flat, dAqk, dAkk, 1.0
    )
    dq = dq_o + dq_gram
    dk = dk + dk_gram
    dgc = dgc + dgc_gram

    # decay = cumsum(g) over C, so dg = reverse-inclusive-cumsum of dgc per chunk.
    dgc4 = dgc.reshape(BH, N, C, D)
    dg = dgc4.flip(-2).cumsum(-2).flip(-2)

    dq_out = dq.float().reshape(B, H, T, D)
    dk_out = dk.float().reshape(B, H, T, D)
    dv_out = dv.float().reshape(B, H, T, DV)
    dg_out = dg.reshape(B, H, T, D)
    dbeta_out = dbeta.reshape(B, H, T)

    return dq_out, dk_out, dv_out, dg_out, dbeta_out


# Public entry point
# ════════════════════════════════════════════════════════════════════════════════


@overload
def chunked_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None = ...,
    a: torch.Tensor | None = ...,
    C: int = ...,
    initial_state: torch.Tensor | None = ...,
    return_final_state: Literal[False] = ...,
    head_first: bool = ...,
    scale: float = ...,
) -> torch.Tensor: ...


@overload
def chunked_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None = ...,
    a: torch.Tensor | None = ...,
    C: int = ...,
    initial_state: torch.Tensor | None = ...,
    return_final_state: Literal[True] = ...,
    head_first: bool = ...,
    scale: float = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def chunked_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
    C: int = 64,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
    head_first: bool = True,
    scale: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Public entry point for chunked linear attention. g=None means no decay.

    scale folds into the kernels (no-decay path only); pass q unscaled with it.
    Defaults to 1.0 so callers that pre-scale q are unaffected."""
    if not head_first:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        if g is not None:
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
        q = torch.nn.functional.pad(q, [0, 0, 0, pad])
        k = torch.nn.functional.pad(k, [0, 0, 0, pad])
        v = torch.nn.functional.pad(v, [0, 0, 0, pad])
        if g is not None:
            if g.dim() == 3:
                g = torch.nn.functional.pad(g, [0, pad])
            else:
                g = torch.nn.functional.pad(g, [0, 0, 0, pad])
        if beta is not None:
            if beta.dim() == 3:
                beta = torch.nn.functional.pad(beta, [0, pad])
            else:
                beta = torch.nn.functional.pad(beta, [0, pad])
        if a is not None:
            a = torch.nn.functional.pad(a, [0, 0, 0, pad])

    o, final_state = ChunkedLinearAttnFn.apply(
        q, k, v, g, beta, a, C, initial_state, return_final_state, scale
    )

    if pad > 0:
        o = o[:, :, :T_cur]

    if not head_first:
        o = o.transpose(1, 2).contiguous()

    if return_final_state:
        return o, final_state
    return o


# ════════════════════════════════════════════════════════════════════════════════
# Kernel variants
# ════════════════════════════════════════════════════════════════════════════════


class LinearAttentionVariant(Enum):
    """A named chunked-linear-attention variant."""

    VANILLA = "vanilla_linear_attn"
    SIMPLE_GLA = "simple_gla"
    RETENTION = "retention"
    FULL_GLA = "full_gla"
    DELTA_RULE = "delta_rule"
    GATED_DELTA_RULE = "gated_delta_rule"
    KDA = "kda"
    MAMBA2_SSD = "mamba2_ssd"


class HelionForwardKernel(Protocol):
    """Shared callable type for Helion-native linear-attention wrappers."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        *,
        C: int = 64,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


# ────────────────────────────────────────────────────────────────────────────────
# Helion forward kernels, one per variant, sharing a Helion-native signature.
#
# Each takes head-first [B, H, T, *] inputs and returns chunked_linear_attn's native
# result: a bare output unless final state is requested.
# ────────────────────────────────────────────────────────────────────────────────


def helion_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    *,
    C: int = 64,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return chunked_linear_attn(
        q,
        k,
        v,
        None,
        C=C,
        scale=scale,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


def helion_chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    *,
    C: int = 64,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert g is not None
    return chunked_linear_attn(
        q * scale,
        k,
        v,
        g,
        C=C,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


def helion_chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    *,
    C: int = 64,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert g is not None
    return chunked_linear_attn(
        q * scale,
        k,
        v,
        g,
        C=C,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


def helion_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    *,
    C: int = 64,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert g is not None
    return chunked_linear_attn(
        q,
        k,
        v,
        g,
        C=C,
        scale=scale,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


def helion_chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    *,
    C: int = 64,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert beta is not None
    return chunked_linear_attn(
        q * scale,
        k,
        v,
        None,
        beta=beta,
        C=C,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


def helion_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    *,
    C: int = 64,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert g is not None
    assert beta is not None
    return chunked_linear_attn(
        q * scale,
        k,
        v,
        g,
        beta=beta,
        C=C,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


def helion_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    *,
    C: int = 64,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert g is not None
    assert beta is not None
    return chunked_linear_attn(
        q * scale,
        k,
        v,
        g,
        beta=beta,
        C=C,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


def helion_chunk_mamba2_ssd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    *,
    C: int = 64,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert g is not None
    return chunked_linear_attn(
        q * scale,
        k,
        v,
        g,
        C=C,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


_HELION_FWD: dict[LinearAttentionVariant, HelionForwardKernel] = {
    LinearAttentionVariant.VANILLA: helion_chunk_linear_attn,
    LinearAttentionVariant.SIMPLE_GLA: helion_chunk_simple_gla,
    LinearAttentionVariant.RETENTION: helion_chunk_retention,
    LinearAttentionVariant.FULL_GLA: helion_chunk_gla,
    LinearAttentionVariant.DELTA_RULE: helion_chunk_delta_rule,
    LinearAttentionVariant.GATED_DELTA_RULE: helion_chunk_gated_delta_rule,
    LinearAttentionVariant.KDA: helion_chunk_kda,
    LinearAttentionVariant.MAMBA2_SSD: helion_chunk_mamba2_ssd,
}


def get_helion_fwd_kernel(
    variant: LinearAttentionVariant,
) -> HelionForwardKernel:
    """Return the Helion forward kernel for a variant.

    kernel = get_helion_fwd_kernel(LinearAttentionVariant.SIMPLE_GLA)
    o = kernel(q, k, v, g, scale=scale)
    """
    return _HELION_FWD[variant]
