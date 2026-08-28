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
import functools
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
def l2norm_fwd_helion(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit-norm each row over the D axis, in fp32.
    y[n, d] = x[n, d] / sqrt(sum_d x[n]^2 + eps)     # [N, D]"""
    N, D = x.size()
    y = torch.empty_like(x)
    for tile_n in hl.tile(N):
        xt = x[tile_n, :].to(torch.float32)
        rstd = torch.rsqrt((xt * xt).sum(dim=-1, keepdim=True) + eps)
        y[tile_n, :] = (xt * rstd).to(x.dtype)
    return y


@helion.kernel()
def chunk_cumsum_gc_helion(
    g: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    H: int = 1,
    N: int = 1,
    use_gate: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    has_bias: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
    use_lower_bound: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """Per-chunk cumulative log-decay over the chunk (C) axis.
        gc[i] = sum_{j <= i} g[j]        # [BHN, C, D], fp32
    Computed as the tensor-core matmul gc = L @ g with L the [C, C]
    lower-triangular ones matrix, instead of a strided scan over the C axis
    (dim -2 of a [BH, N, C, D] tensor) whose steps stride by D. The bf16 g is
    exactly representable in tf32, and the sum accumulates in fp32, so gc keeps
    the same precision as an fp32 scan while reading g once through the MMA.

    With use_gate=True, g arrives pre-activation and the gate is applied before the
    sum, over rows flattened as [B, H, N]:
        h  = (r // N) % H                # head of row r
        gb = g + dt_bias[h]              # has_bias=True, else gb = g
      - use_lower_bound=True (default):
            g = lower_bound * sigmoid(exp(A_log[h]) * gb)
      - use_lower_bound=False:
            g = -exp(A_log[h]) * softplus(gb)"""
    BHN = g.size(0)
    C = hl.specialize(g.size(1))
    D = g.size(2)
    gc = torch.empty([BHN, C, D], dtype=torch.float32, device=g.device)
    for tile_bhn, tile_d in hl.tile([BHN, D]):
        idx = hl.arange(C)
        ltri = (idx[:, None] >= idx[None, :]).to(torch.float32)  # [C, C] incl. diag
        L = ltri[None, :, :].broadcast_to([tile_bhn, C, C])  # pyrefly: ignore[no-matching-overload]
        gt = g[tile_bhn, :, tile_d].to(torch.float32)  # [b, C, d]
        if use_gate:
            h_idx = (tile_bhn.index // N) % H
            a = torch.exp(A_log[h_idx].to(torch.float32))[:, None, None]  # pyrefly: ignore[unsupported-operation]
            if has_bias:
                gt = gt + dt_bias[h_idx, tile_d][:, None, :]  # pyrefly: ignore[unsupported-operation]
            if use_lower_bound:
                gt = lower_bound * torch.sigmoid(a * gt)  # pyrefly: ignore[unsupported-operation]
            else:
                sp = torch.clamp(gt, min=0.0) + torch.log1p(torch.exp(-torch.abs(gt)))
                gt = -a * sp
        gc[tile_bhn, :, tile_d] = hl.dot(L, gt)
    return gc


@helion.kernel()
def chunk_cumsum_gc_varlen_helion(
    g: torch.Tensor,
    token_base: torch.Tensor,
    valid_len: torch.Tensor,
    gc: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_gate: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    has_bias: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
    use_lower_bound: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
) -> None:
    """chunk_cumsum_gc_helion over a varlen batch, g read token-major.

    Row r of the flat [H * NT] chunk axis addresses its own tokens, the four lines
    every varlen kernel here opens with:
        h     = r // NT                  # head
        j     = r  % NT                  # chunk, over all sequences
        rows  = token_base[j] + i        # its tokens, i in [0, C)
        valid = i < valid_len[j]         # False past this sequence's end
    so g is read where it lies, never copied into chunk-major order:
        gt = g[rows, h] * valid          # [C, D] fp32, tail zeroed
    With use_gate=True gt arrives pre-activation and the gate applies before the sum:
      - use_lower_bound=True (default):
            gt = lower_bound * sigmoid(exp(A_log[h]) * (gt + dt_bias[h]))
      - use_lower_bound=False:
            gt = -exp(A_log[h]) * softplus(gt + dt_bias[h])
    The gate maps 0 to lower_bound * sigmoid(0) != 0, so the tail is masked again
    after it, then the running total is a prefix sum along the chunk:
        gt = gt * valid                  # [C, D]
        gc[r] = cumsum(gt, dim=0)        # [C, D]
    The dense kernel's [C, C] lower-triangular matmul reaches the same values, but it
    sums on the tensor cores in their reduced input precision and costs C times the
    arithmetic.
    A zeroed tail holds the running sum flat, leaving the chunk total gc[C-1]
    unchanged. Separate from the dense kernel because the chunk axis must be
    block_size=1 for a scalar row, and hl.tile cannot sit inside a branch."""
    NT = token_base.size(0)
    C = hl.specialize(gc.size(1))
    for tile_r, tile_d in hl.tile([gc.size(0), g.size(2)], block_size=[1, None]):
        idx = hl.arange(C)
        j = tile_r.begin % NT
        h = tile_r.begin // NT
        valid = idx < valid_len[j]
        gt = hl.load(g, [token_base[j] + idx, h, tile_d], extra_mask=valid[:, None]).to(
            torch.float32
        )  # [C, d]
        gt = torch.where(valid[:, None], gt, 0.0)
        if use_gate:
            a = torch.exp(A_log[h].to(torch.float32))  # pyrefly: ignore[unsupported-operation]
            if has_bias:
                gt = gt + dt_bias[h : h + 1, tile_d]  # pyrefly: ignore[unsupported-operation]
            if use_lower_bound:
                gt = lower_bound * torch.sigmoid(a * gt)  # pyrefly: ignore[unsupported-operation]
            else:
                sp = torch.clamp(gt, min=0.0) + torch.log1p(torch.exp(-torch.abs(gt)))
                gt = -a * sp
            gt = torch.where(valid[:, None], gt, 0.0)
        gc[tile_r.begin, :, tile_d] = hl.cumsum(gt, dim=0)


@helion.kernel()
def chunk_fwd_h_diag_fused(
    k: torch.Tensor,
    v: torch.Tensor,
    g_last: torch.Tensor,
    h0: torch.Tensor | None,
    gc: torch.Tensor | None = None,
    use_g: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
    scalar_decay: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    diag_anchored: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    has_h0: hl.constexpr = True,  # pyrefly: ignore[bad-function-definition]
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
        if has_h0:
            h_acc = h0[idx, tile_d, tile_dv].float()  # pyrefly: ignore[unsupported-operation]
        else:
            h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)

        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc.to(h_all.dtype)
            k_i = k[idx, i_t, :, tile_d]
            if scalar_decay:
                g_i = gc[  # pyrefly: ignore[unsupported-operation]
                    idx, i_t, :
                ].float()  # [C]
                gl = g_last[idx, i_t].float()  # scalar
                v_i = (
                    v[idx, i_t, :, tile_dv].float()
                    * torch.exp2((gl - g_i) * RCP_LN2)[:, None]
                ).to(v.dtype)
                h_acc = torch.exp2(gl * RCP_LN2) * h_acc
            elif diag_anchored:
                gc_i = gc[idx, i_t, :, tile_d]  # pyrefly: ignore[unsupported-operation]
                gc_last = gc[idx, i_t, C - 1, tile_d]  # pyrefly: ignore[unsupported-operation]
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
    k_state_out: torch.Tensor | None = None,
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
        This path also writes the serial state pass's key operand, which shares
        the k read and the per-channel gc:
            k_state_out = k * exp(gc[C-1] - gc)     # [C, D]
    """
    BHN = k.size(0)
    C = hl.specialize(k.size(1))
    D = k.size(2)
    DV = v.size(2)
    # The Neumann-series inverse is exact in n_doublings = log2(C) steps only
    # when C is a power of two.
    assert C & (C - 1) == 0, f"chunk size C must be a power of two, got {C}"
    n_doublings = C.bit_length() - 1

    w = torch.empty([BHN, C, D], dtype=k.dtype, device=k.device)
    u = torch.empty([BHN, C, DV], dtype=v.dtype, device=v.device)
    A_inv = torch.empty([BHN, C, C], dtype=torch.float32, device=k.device)

    for tile_bhn in hl.tile(BHN, block_size=1):
        beta_i = beta[tile_bhn, :].to(torch.float32)  # [1, C]

        idx = hl.arange(C)
        strict_lower = idx[:, None] > idx[None, :]
        if diag_anchored:
            # The anchored k-gram Akk already carries the per-channel decay and
            # the strict-lower mask; the in-kernel k @ k gram is compiled out.
            A = -(beta_i[:, :, None] * Akk[tile_bhn, :, :])  # pyrefly: ignore[unsupported-operation]
        else:
            kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
            for tile_kk in hl.tile(D):
                kt_kk = k[tile_bhn, :, tile_kk]
                kk = hl.dot(kt_kk, kt_kk.transpose(-2, -1), acc=kk)
            if scalar_decay:
                decay = g_cs[tile_bhn, :].to(  # pyrefly: ignore[unsupported-operation]
                    torch.float32
                )  # [1, C]
                L = torch.exp2((decay[:, :, None] - decay[:, None, :]) * RCP_LN2)
                A = torch.where(strict_lower, -(beta_i[:, :, None] * kk * L), 0.0)
            else:
                A = torch.where(strict_lower, -(beta_i[:, :, None] * kk), 0.0)

        eye = (idx[:, None] == idx[None, :]).to(torch.float32)
        eye = eye[None, :, :].broadcast_to([tile_bhn, C, C])  # pyrefly: ignore[no-matching-overload]
        T = eye + A
        Apow = A
        for _ in range(n_doublings - 1):
            Apow = hl.dot(Apow, Apow)
            T = hl.dot(Apow, T, acc=T)

        A_inv[tile_bhn, :, :] = T

        for tile_d in hl.tile(D):
            raw_k = k[tile_bhn, :, tile_d].to(torch.float32)
            kt = raw_k * beta_i[:, :, None]
            if scalar_decay:
                kt = kt * torch.exp2(decay * RCP_LN2)[:, :, None]  # pyrefly: ignore[unbound-name]
            elif diag_anchored:
                gc_d = g_cs[tile_bhn, :, tile_d].to(torch.float32)  # pyrefly: ignore[unsupported-operation]
                kt = kt * torch.exp2(gc_d * RCP_LN2)
                gc_last = g_cs[tile_bhn, C - 1, tile_d].to(torch.float32)  # pyrefly: ignore[unsupported-operation]
                k_state_out[tile_bhn, :, tile_d] = (  # pyrefly: ignore[unsupported-operation]
                    raw_k * torch.exp2((gc_last[:, None, :] - gc_d) * RCP_LN2)
                ).to(k.dtype)
            w[tile_bhn, :, tile_d] = hl.dot(T, kt).to(w.dtype)
        for tile_dv in hl.tile(DV):
            vt = v[tile_bhn, :, tile_dv].to(torch.float32) * beta_i[:, :, None]
            u[tile_bhn, :, tile_dv] = hl.dot(T, vt).to(u.dtype)

    return w, u, A_inv


@helion.kernel()
def chunk_fwd_wy_delta_varlen_helion(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cs: torch.Tensor,
    Akk: torch.Tensor,
    token_base: torch.Tensor,
    valid_len: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    k_state_out: torch.Tensor,
    scalar_gate: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> None:
    """chunk_fwd_wy_delta_helion's diag_anchored path over a varlen batch.

    k, v and beta are read token-major ([T_total, H, D], [T_total, H, DV],
    [T_total, H]); g_cs, Akk and every output are per-chunk and stay chunk-major
    [H * NT, C, *]. Row r addresses its tokens as in chunk_cumsum_gc_varlen_helion:
        beta_i = beta[rows, h] * valid                # [C] fp32, tail zeroed
        k_i    = k[rows, h]    * valid                # [C, D]
        v_i    = v[rows, h]    * valid                # [C, DV]
    then the transform is the dense diag_anchored one, with the anchored k-gram Akk
    already carrying the per-channel decay and the strict mask:
        A   = -(beta_i * Akk)                         # [C, C] strict lower
        T   = (I - A)^-1                              # log2(C) Neumann doublings
        w   = T @ (beta_i * exp2(RCP_LN2 * gc) * k_i)         # [C, D]
        u   = T @ (beta_i * v_i)                              # [C, DV]
        k_state_out = k_i * exp2(RCP_LN2 * (gc[C-1] - gc))    # [C, D]
    A zero beta_i row zeros that row of A, so T keeps an identity row there and w, u
    and k_state_out are all zero on it: the tail contributes nothing downstream.

    scalar_gate=True takes g_cs one channel wide, the cumulative sum of a gate that is
    one scalar per token. Both decays are then read once for the whole row rather than
    per D tile, and each broadcasts over the channel axis.

    The dense kernel also returns T as A_inv for its backward; this path is forward
    only, so T is not stored. With DV == D the value loop rides the key loop, so u is
    written in the same pass as w and k_state_out."""
    NT = token_base.size(0)
    C = hl.specialize(g_cs.size(1))
    D = k.size(2)
    DV = v.size(2)
    assert C & (C - 1) == 0, f"chunk size C must be a power of two, got {C}"
    n_doublings = C.bit_length() - 1
    fuse_v = D == DV

    for tile_r in hl.tile(w.size(0), block_size=1):
        j = tile_r.begin % NT
        h = tile_r.begin // NT
        base = token_base[j]
        idx = hl.arange(C)
        valid = idx < valid_len[j]

        beta_i = torch.where(
            valid, hl.load(beta, [base + idx, h], extra_mask=valid), 0.0
        ).to(torch.float32)  # [C]

        A = -(beta_i[:, None] * Akk[tile_r.begin, :, :])
        eye = (idx[:, None] == idx[None, :]).to(torch.float32)
        T = eye + A
        Apow = A
        for _ in range(n_doublings - 1):
            Apow = hl.dot(Apow, Apow)
            T = hl.dot(Apow, T, acc=T)

        if scalar_gate:
            g_col = g_cs[tile_r.begin, :, 0].to(torch.float32)  # [C]
            g_last = g_cs[tile_r.begin, C - 1, 0].to(torch.float32)  # scalar
            kt_gate = (beta_i * torch.exp2(g_col * RCP_LN2))[:, None]
            k_state_gate = torch.exp2((g_last - g_col) * RCP_LN2)[:, None]

        for tile_d in hl.tile(D):
            raw_k = torch.where(
                valid[:, None],
                hl.load(k, [base + idx, h, tile_d], extra_mask=valid[:, None]),
                0,
            ).to(torch.float32)
            if scalar_gate:
                kt = raw_k * kt_gate
                k_state_out[tile_r.begin, :, tile_d] = (raw_k * k_state_gate).to(
                    k.dtype
                )
            else:
                gc_d = g_cs[tile_r.begin, :, tile_d].to(torch.float32)
                gc_last = g_cs[tile_r.begin, C - 1, tile_d].to(torch.float32)
                kt = raw_k * beta_i[:, None] * torch.exp2(gc_d * RCP_LN2)
                k_state_out[tile_r.begin, :, tile_d] = (
                    raw_k * torch.exp2((gc_last[None, :] - gc_d) * RCP_LN2)
                ).to(k.dtype)
            w[tile_r.begin, :, tile_d] = hl.dot(T, kt).to(w.dtype)
            if fuse_v:
                vt = (
                    torch.where(
                        valid[:, None],
                        hl.load(v, [base + idx, h, tile_d], extra_mask=valid[:, None]),
                        0,
                    ).to(torch.float32)
                    * beta_i[:, None]
                )
                u[tile_r.begin, :, tile_d] = hl.dot(T, vt).to(u.dtype)
        if not fuse_v:
            for tile_dv in hl.tile(DV):
                vt = (
                    torch.where(
                        valid[:, None],
                        hl.load(v, [base + idx, h, tile_dv], extra_mask=valid[:, None]),
                        0,
                    ).to(torch.float32)
                    * beta_i[:, None]
                )
                u[tile_r.begin, :, tile_dv] = hl.dot(T, vt).to(u.dtype)


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
    k_pre_scaled: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
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
        With k_pre_scaled=True the key arrives with exp(gc_last - gc) already
        applied, so this serial walk only decays the carry.
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
                decay_i = g_cs[  # pyrefly: ignore[unsupported-operation]
                    idx, i_t, :
                ].float()  # [C]
                dl = decay_last[  # pyrefly: ignore[unsupported-operation]
                    idx, i_t
                ].float()  # scalar
                k_i = (k_i.float() * torch.exp2((dl - decay_i) * RCP_LN2)[:, None]).to(
                    k_i.dtype
                )
                h_orig = h_orig * torch.exp2(dl * RCP_LN2)
            elif diag_anchored:
                gl = decay_last[  # pyrefly: ignore[unsupported-operation]
                    idx, i_t, :
                ].float()  # [D]
                if not k_pre_scaled:
                    gc_i = g_cs[  # pyrefly: ignore[unsupported-operation]
                        idx, i_t, :, :
                    ].float()  # [C, D]
                    k_i = (k_i.float() * torch.exp2((gl[None, :] - gc_i) * RCP_LN2)).to(
                        k_i.dtype
                    )
                h_orig = h_orig * torch.exp2(gl * RCP_LN2)[:, None]
            h_acc = hl.dot(k_i.transpose(-2, -1), vnew_i.to(k_i.dtype), acc=h_orig)

    return h_all, v_new


@helion.kernel()
def chunk_fwd_h_delta_varlen_helion(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    h0: torch.Tensor,
    decay_last: torch.Tensor,
    chunk_offsets: torch.Tensor,
    NT: int,
    H: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """chunk_fwd_h_delta_helion's diag_anchored serial pass, walked per sequence.

    Sequence n owns chunks chunk_offsets[n] : chunk_offsets[n + 1], a count that
    differs per sequence, so the walk is one program per (head, sequence) and the
    trip count is a tensor load. All operands are chunk-major, row h * NT + j, and k
    arrives with the anchored per-channel decay already applied. With decay_last =
    gc[C-1] the per-channel chunk total, in natural-log space:
        S = h0[h, n]                                 # [D, DV]
        for j in chunk_offsets[n] : chunk_offsets[n + 1]:
            h_all[h, j] = S                          # state entering chunk j
            v_new[h, j] = u[h, j] - w[h, j] @ S      # [C, DV] delta correction
            S = exp(decay_last[h, j])[:, None] * S
                + k[h, j].T @ v_new[h, j]            # [D, DV]
        ht[h, n] = S                                 # this sequence's final state
    computed as exp2(RCP_LN2 * x) == exp(x):
            S = exp2(RCP_LN2 * decay_last[h, j])[:, None] * S + ...
    Initializing S inside the per-sequence loop is the boundary reset: no path can
    carry it across one. A ragged chunk's tail rows are zero, so they add nothing to
    either matmul, and ht is the final state directly, with no host-side last-chunk
    arithmetic.

    DV leads the tile axes, so a chunk row's DV slices differ by 1 in the flat program
    id where an (H * N)-major order separates them by H * N. Matches FLA's
    grid=(cdiv(V, BV), N * HV)."""
    D = k.size(2)
    DV = u.size(2)
    N = chunk_offsets.size(0) - 1
    C = k.size(1)
    HNT = k.size(0)

    h_all = torch.empty([HNT, D, DV], dtype=k.dtype, device=k.device)
    v_new = torch.empty([HNT, C, DV], dtype=k.dtype, device=u.device)
    ht = torch.empty([H * N, D, DV], dtype=torch.float32, device=k.device)

    for tile_dv, tile_hn in hl.tile([DV, H * N], block_size=[None, 1]):
        hn = tile_hn.begin
        n = hn % N
        h_off = (hn // N) * NT  # first chunk row of this head
        h_acc = h0[hn, :, tile_dv].float()  # [D, bv]

        for tile_j in hl.tile(chunk_offsets[n], chunk_offsets[n + 1], block_size=1):
            j = h_off + tile_j.begin
            h_all[j, :, tile_dv] = h_acc.to(h_all.dtype)
            w_j = w[j, :, :]  # [C, D]
            u_j = u[j, :, tile_dv]  # [C, bv]
            vnew_j = u_j.float() - hl.dot(w_j, h_acc.to(w_j.dtype)).float()
            v_new[j, :, tile_dv] = vnew_j.to(v_new.dtype)
            gl = decay_last[j, :].float()  # [D]
            h_acc = h_acc * torch.exp2(gl * RCP_LN2)[:, None]
            k_j = k[j, :, :]  # [C, D]
            h_acc = hl.dot(k_j.transpose(-2, -1), vnew_j.to(k_j.dtype), acc=h_acc)

        ht[hn, :, tile_dv] = h_acc

    return h_all, v_new, ht


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
        causal = idx[:, None] >= idx[None, :]
        if use_g:
            gc = g_cs[tile_bhn, :]
            decay_ij = torch.exp2((gc[:, :, None] - gc[:, None, :]) * RCP_LN2)
            attn = torch.where(causal, attn * decay_ij, 0.0)
            o_cross = o_cross * torch.exp2(gc * RCP_LN2)[:, :, None]
        else:
            attn = torch.where(causal, attn, 0.0)

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
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse state pass for the delta rule. Walk i = N-1 -> 0 carrying the
    reverse state dS; emits the full value grad dvni so the per-chunk pass need
    not recompute it. The chunk walk is serial; DV is split across programs (D
    kept whole), so each holds a [D, tile_dv] slice of dS. The forward scales o,
    so scale folds into the do_i load once; dvni/dS_future carry it to dqkw, which
    scales only its own do load. Decay modes:
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
            do_i = (do[idx, i, :, tile_dv].float() * scale).to(do.dtype)  # [C, bv]

            jdx = hl.arange(C)
            causal = jdx[:, None] >= jdx[None, :]
            attn = hl.dot(q_i, k_i.transpose(-2, -1)).float()
            if scalar_decay:
                gc = g_cs[  # pyrefly: ignore[unsupported-operation]
                    idx, i, :
                ].float()  # [C]
                dl = decay_last[  # pyrefly: ignore[unsupported-operation]
                    idx, i
                ].float()  # scalar
                attn = attn * torch.exp2((gc[:, None] - gc[None, :]) * RCP_LN2)
            attn = torch.where(causal, attn, 0.0).to(q.dtype)

            dv_i = hl.dot(attn.transpose(-2, -1), do_i).float()
            if scalar_decay:
                dv_kh = hl.dot(k_i, dS.to(k_i.dtype)).float()
                dv_i = dv_i + dv_kh * torch.exp2((dl - gc) * RCP_LN2)[:, None]  # pyrefly: ignore[unbound-name]
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
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-chunk parallel adjoint of the delta output and state carry. dattn sums
    over DV in its own loop up front (formed once); dW/dq/dk_as then tile over D
    with an inner DV loop, reusing it. scale folds into this kernel's own do loads
    (dattn and the dq_cross term); dvni/dS_future already carry it from dstate and
    are not re-scaled. Decay modes:
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
    dg_p_out = torch.zeros([BHN, C], dtype=torch.float32, device=q.device)
    dg_last_out = torch.zeros([BHN], dtype=torch.float32, device=q.device)

    hdt = q.dtype
    for tile_bhn in hl.tile(BHN, block_size=1):
        idx = hl.arange(C)
        causal = idx[:, None] >= idx[None, :]
        if scalar_decay:
            decay_b = g_cs[tile_bhn, :].to(  # pyrefly: ignore[unsupported-operation]
                torch.float32
            )  # [1, C]
            dl = decay_last[tile_bhn].to(  # pyrefly: ignore[unsupported-operation]
                torch.float32
            )  # [1]
            decay_exp = torch.exp2(decay_b * RCP_LN2)[:, :, None]  # [1, C, 1]
            kdec_w = torch.exp2((dl[:, None] - decay_b) * RCP_LN2)[:, :, None]

        dattn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_dv in hl.tile(DV):
            do_dv = (do[tile_bhn, :, tile_dv].to(torch.float32) * scale).to(hdt)
            vnew_h = v_new[tile_bhn, :, tile_dv].to(hdt)
            dattn = hl.dot(do_dv, vnew_h.transpose(-2, -1), acc=dattn)
        if scalar_decay:
            L = torch.exp2((decay_b[:, :, None] - decay_b[:, None, :]) * RCP_LN2)  # pyrefly: ignore[unbound-name]
            dattn = dattn * L
        dattn = torch.where(causal, dattn, 0.0).to(hdt)  # [1, C, C]

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
                do_dv = (do[tile_bhn, :, tile_dv].to(torch.float32) * scale).to(
                    hdt
                )  # [1, C, dv]
                dSf_dv = dS_future[tile_bhn, tile_d, tile_dv].to(hdt)  # [1, bd, dv]
                dvni_h = dvni[tile_bhn, :, tile_dv].to(hdt)  # [1, C, dv]
                vnew_h = v_new[tile_bhn, :, tile_dv].to(hdt)  # [1, C, dv]
                dW = dW - hl.dot(dvni_h, S_h.transpose(-2, -1)).float()
                dq_cross = hl.dot(do_dv, S_h.transpose(-2, -1), acc=dq_cross)
                dk_state = hl.dot(vnew_h, dSf_dv.transpose(-2, -1), acc=dk_state)

            if scalar_decay:
                dq_cross = dq_cross * decay_exp  # pyrefly: ignore[unbound-name]
                dk_state = dk_state * kdec_w  # pyrefly: ignore[unbound-name]

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
            dg_last_out[tile_bhn] = torch.exp2(dl * RCP_LN2) * hd + dg_last_dk  # pyrefly: ignore[unbound-name]

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
    dg_wy_out = torch.zeros([BHN, C], dtype=torch.float32, device=k.device)

    hdt = k.dtype
    for tile_bhn in hl.tile(BHN, block_size=1):
        beta_i = beta[tile_bhn, :].to(torch.float32)  # [1, C]
        T_i = A_inv[tile_bhn, :, :]
        T_t_h = T_i.transpose(-2, -1).to(hdt)
        idx = hl.arange(C)
        strict_lower = (idx[:, None] > idx[None, :]).to(torch.float32)
        if scalar_decay:
            decay = g_cs[tile_bhn, :].to(  # pyrefly: ignore[unsupported-operation]
                torch.float32
            )  # [1, C]
            decay_exp = torch.exp2(decay * RCP_LN2)[:, :, None]  # [1, C, 1]
            causal = idx[:, None] >= idx[None, :]
            L = torch.where(
                causal,
                torch.exp2((decay[:, :, None] - decay[:, None, :]) * RCP_LN2),
                0.0,
            )

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
                kbg = (beta_i[:, :, None] * decay_exp * k_d).to(hdt)  # pyrefly: ignore[unbound-name]
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
            dL_out[tile_bhn, :, :] = (dA * beta_i[:, :, None] * L).to(dL_out.dtype)  # pyrefly: ignore[unbound-name]
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
    with per-channel decay gc [BHN, C, D]. The forward scales o, so scale folds
    into the do load once here and flows to every output (dh feeds the state
    kernels; dAqk feeds gram2):
        dqg    = (scale*do) @ h.T               # [C, D]
        dq_o   = dqg * exp(gc)                  # [C, D]
        dgc_o  = dqg * q * exp(gc)              # [C, D]  per-channel decay grad
        dh     = (q * exp(gc)).T @ (scale*do)   # [D, DV]
        dAqk   = causal((scale*do) @ v_new.T)   # [C, C]  s <= t
        dv_new = Aqk.T @ (scale*do)             # [C, DV]
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
            dof32 = do[tile_bhn, :, tile_v].to(torch.float32)
            dob = (dof32 * scale).to(hdt)
            vb = v_new[tile_bhn, :, tile_v].to(hdt)
            dA = dA + hl.dot(dob, vb.transpose(-2, -1))
            # Aqk carries scale, so dv_new pairs it with the unscaled do.
            dv_new[tile_bhn, :, tile_v] = hl.dot(Atb, dof32.to(hdt))
        dAqk[tile_bhn, :, :] = torch.where(incl, dA, 0.0)

        for tile_k in hl.tile(K):
            qt = q[tile_bhn, :, tile_k].to(torch.float32)
            gct = gc[tile_bhn, :, tile_k].to(torch.float32)
            egc = torch.exp2(gct * RCP_LN2)
            qg = (qt * egc).to(hdt)
            qgt_b = qg.transpose(-2, -1)

            dqg = hl.zeros([tile_bhn, C, tile_k], dtype=torch.float32)
            for tile_v in hl.tile(V):
                dob = (do[tile_bhn, :, tile_v].to(torch.float32) * scale).to(hdt)
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
            dS = (
                dS
                - hl.dot(
                    w_i.transpose(-2, -1).to(w.dtype), dv_new_i.to(w.dtype)
                ).float()
            )
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
            dw_acc = dw_acc - hl.dot(
                du_i.to(k.dtype), S_i.transpose(-2, -1).to(k.dtype)
            )
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
                g_i = gc[idx, i, :].float()  # pyrefly: ignore[unsupported-operation]
                dh_acc = torch.exp2(g_last[idx, i].float() * RCP_LN2) * dh_acc
                q_i = (
                    q[idx, i, :, tile_d].float()
                    * (scale * torch.exp2(g_i * RCP_LN2))[:, None]
                ).to(q.dtype)
            elif diag_anchored:
                gc_i = gc[idx, i, :, tile_d].float()  # pyrefly: ignore[unsupported-operation]
                gc_last = gc[idx, i, C - 1, tile_d].float()  # pyrefly: ignore[unsupported-operation]
                dh_acc = dh_acc * torch.exp2(gc_last * RCP_LN2)[:, None]
                q_i = (
                    q[idx, i, :, tile_d].float() * (scale * torch.exp2(gc_i * RCP_LN2))
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
    g_cs: torch.Tensor | None,
    g_last: torch.Tensor | None,
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
            gc = g_cs[tile_bhn, :]  # pyrefly: ignore[unsupported-operation]
            decay_ij = torch.exp(gc[:, :, None] - gc[:, None, :])
            dA = dA_raw * decay_ij * causal
            gl = g_last[tile_bhn]  # pyrefly: ignore[unsupported-operation]
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
    dg_by_d = torch.zeros([BHN, C, D], dtype=torch.float32, device=q.device)

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
        causal = idx[:, None] >= idx[None, :]
        qt = q[tile_bhn, :, tile_d]
        kt = k[tile_bhn, :, tile_d]

        if diag_anchored:
            gct = g_cs[tile_bhn, :, tile_d].float()
            gc_last = g_cs[tile_bhn, C - 1, tile_d].float()
            exp_gc = torch.exp2(gct * RCP_LN2)
            exp_neg_gc = torch.exp2(-gct * RCP_LN2)
            exp_gl = torch.exp2(gc_last * RCP_LN2)
            exp_gl_minus_gc = torch.exp2((gc_last[:, None, :] - gct) * RCP_LN2)

            dA = torch.where(causal[None, :, :], dA_raw, 0.0)
            dk_state = dk_state_acc * exp_gl_minus_gc
            dq_acc = hl.dot(dA, kt * exp_neg_gc, acc=dq_cross_acc) * exp_gc * scale
            dk_acc = (
                hl.dot(dA.transpose(-2, -1), qt * exp_gc) * exp_neg_gc * scale
                + dk_state
            )
        else:
            gc = g_cs[tile_bhn, :].float()
            gl = g_last[tile_bhn].float()  # pyrefly: ignore[unsupported-operation]
            exp_gc = torch.exp2(gc * RCP_LN2)[:, :, None]
            exp_gl = torch.exp2(gl * RCP_LN2)[:, None]
            exp_gl_minus_gc = torch.exp2((gl[:, None] - gc) * RCP_LN2)[:, :, None]

            dA = torch.where(
                causal,
                dA_raw * torch.exp2((gc[:, :, None] - gc[:, None, :]) * RCP_LN2),
                0.0,
            )
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
    g_cs: torch.Tensor | None,
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
                gct = g_cs[tile_bhn, :, tile_d].float()  # pyrefly: ignore[unsupported-operation]
                gc_last = g_cs[tile_bhn, C - 1, tile_d].float()  # pyrefly: ignore[unsupported-operation]
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
        causal = idx[:, None] >= idx[None, :]
        if scalar_decay:
            gc = g_cs[tile_bhn, :].float()  # pyrefly: ignore[unsupported-operation]
            decay_ij = torch.exp2((gc[:, :, None] - gc[:, None, :]) * RCP_LN2)
            attn = torch.where(causal, attn * decay_ij * scale, 0.0)
            exp_dk = torch.exp2((g_last[tile_bhn].float()[:, None] - gc) * RCP_LN2)  # pyrefly: ignore[unsupported-operation]
            dv_acc = dv_acc * exp_dk[:, :, None]
        elif diag_anchored:
            attn = A[tile_bhn, :, :]  # pyrefly: ignore[unsupported-operation]
        elif use_g:
            gc = g_cs[tile_bhn, :]  # pyrefly: ignore[unsupported-operation]
            decay_ij = torch.exp(gc[:, :, None] - gc[:, None, :])
            attn = torch.where(causal, attn * decay_ij, 0.0)
        else:
            attn = torch.where(causal, attn * scale, 0.0)

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
    gc_n is gc at the sub-block's first row (the decay anchor), and cols is the live
    width 0 : (i + 1) * BC_DIAG the causal mask leaves nonzero:
        qg = q_blk * exp2(gc_blk - gc_n)        # [BC_DIAG, D]
        kg = k[cols] * exp2(gc_n - gc[cols])    # [cols, D]
        a  = (qg @ kg.T) * scale                # [BC_DIAG, cols]
        A[rows, cols] = a * causal              # keep t >= s
    Anchoring at gc_n bounds each exponent to BC_DIAG rows, so exp2 stays in range.
    With build_kk=True (kda) it also emits the strictly-lower k-rows Gram Akk,
    kg_row = k_blk * exp2(gc_blk - gc_n), Akk[rows, cols] = (kg_row @ kg.T) * strict,
    sharing the anchored kg column operand; Akk feeds the WY/UT transform (fp32).
    Akk is always returned (zeros when build_kk=False; callers ignore it), and both
    arrive zeroed so the columns past cols already hold what a full-width form would
    store there.

    The sub-blocks are written out, one per tier, each guarded on the chunk size:
        tier 0:  rows  0..15,  cols  0..15      always
        tier 1:  rows 16..31,  cols  0..31      C >= 2 * BC_DIAG
        tier 2:  rows 32..47,  cols  0..47      C >= 3 * BC_DIAG
        tier 3:  rows 48..63,  cols  0..63      C >= 4 * BC_DIAG
    so C = 32 takes tiers 0-1 and C = 64 all four. Unrolled rather than tiled because
    each width must be a literal: hl.arange rejects arithmetic on a specialized size.
    FLA's KDA unrolls the same way in chunk_intra.py."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    assert C in (2 * BC_DIAG, 4 * BC_DIAG), (
        f"chunk size C must be {2 * BC_DIAG} or {4 * BC_DIAG}, got {C}"
    )

    A = torch.zeros([BHN, C, C], dtype=q.dtype, device=q.device)
    Akk = torch.zeros([BHN, C, C], dtype=q.dtype, device=q.device)

    for tile_bhn in hl.tile(BHN, block_size=1):
        rows0 = 0 + hl.arange(16)
        cols0 = hl.arange(16)
        q0 = q[tile_bhn, rows0, :].float()
        kc0 = k[tile_bhn, cols0, :].float()
        g0 = gc[tile_bhn, rows0, :].float()
        gcol0 = gc[tile_bhn, cols0, :].float()
        n0 = gc[tile_bhn, 0, :].float()
        e0 = torch.exp2((g0 - n0[:, None, :]) * RCP_LN2)
        kgt0 = (kc0 * torch.exp2((n0[:, None, :] - gcol0) * RCP_LN2)).transpose(-2, -1)
        a0 = hl.dot(q0 * e0, kgt0) * scale
        causal0 = (rows0[:, None] >= cols0[None, :])[None, :, :]
        A[tile_bhn, rows0, cols0] = a0 * causal0.to(a0.dtype)
        if build_kk:
            akk0 = hl.dot(k[tile_bhn, rows0, :].float() * e0, kgt0)
            strict0 = (rows0[:, None] > cols0[None, :])[None, :, :]
            Akk[tile_bhn, rows0, cols0] = akk0 * strict0.to(akk0.dtype)

        if C >= 2 * BC_DIAG:
            rows1 = 16 + hl.arange(16)
            cols1 = hl.arange(32)
            q1 = q[tile_bhn, rows1, :].float()
            kc1 = k[tile_bhn, cols1, :].float()
            g1 = gc[tile_bhn, rows1, :].float()
            gcol1 = gc[tile_bhn, cols1, :].float()
            n1 = gc[tile_bhn, 16, :].float()
            e1 = torch.exp2((g1 - n1[:, None, :]) * RCP_LN2)
            kgt1 = (kc1 * torch.exp2((n1[:, None, :] - gcol1) * RCP_LN2)).transpose(
                -2, -1
            )
            a1 = hl.dot(q1 * e1, kgt1) * scale
            causal1 = (rows1[:, None] >= cols1[None, :])[None, :, :]
            A[tile_bhn, rows1, cols1] = a1 * causal1.to(a1.dtype)
            if build_kk:
                akk1 = hl.dot(k[tile_bhn, rows1, :].float() * e1, kgt1)
                strict1 = (rows1[:, None] > cols1[None, :])[None, :, :]
                Akk[tile_bhn, rows1, cols1] = akk1 * strict1.to(akk1.dtype)

        if C >= 3 * BC_DIAG:
            rows2 = 32 + hl.arange(16)
            cols2 = hl.arange(48)
            q2 = q[tile_bhn, rows2, :].float()
            kc2 = k[tile_bhn, cols2, :].float()
            g2 = gc[tile_bhn, rows2, :].float()
            gcol2 = gc[tile_bhn, cols2, :].float()
            n2 = gc[tile_bhn, 32, :].float()
            e2 = torch.exp2((g2 - n2[:, None, :]) * RCP_LN2)
            kgt2 = (kc2 * torch.exp2((n2[:, None, :] - gcol2) * RCP_LN2)).transpose(
                -2, -1
            )
            a2 = hl.dot(q2 * e2, kgt2) * scale
            causal2 = (rows2[:, None] >= cols2[None, :])[None, :, :]
            A[tile_bhn, rows2, cols2] = a2 * causal2.to(a2.dtype)
            if build_kk:
                akk2 = hl.dot(k[tile_bhn, rows2, :].float() * e2, kgt2)
                strict2 = (rows2[:, None] > cols2[None, :])[None, :, :]
                Akk[tile_bhn, rows2, cols2] = akk2 * strict2.to(akk2.dtype)

        if C >= 4 * BC_DIAG:
            rows3 = 48 + hl.arange(16)
            cols3 = hl.arange(64)
            q3 = q[tile_bhn, rows3, :].float()
            kc3 = k[tile_bhn, cols3, :].float()
            g3 = gc[tile_bhn, rows3, :].float()
            gcol3 = gc[tile_bhn, cols3, :].float()
            n3 = gc[tile_bhn, 48, :].float()
            e3 = torch.exp2((g3 - n3[:, None, :]) * RCP_LN2)
            kgt3 = (kc3 * torch.exp2((n3[:, None, :] - gcol3) * RCP_LN2)).transpose(
                -2, -1
            )
            a3 = hl.dot(q3 * e3, kgt3) * scale
            causal3 = (rows3[:, None] >= cols3[None, :])[None, :, :]
            A[tile_bhn, rows3, cols3] = a3 * causal3.to(a3.dtype)
            if build_kk:
                akk3 = hl.dot(k[tile_bhn, rows3, :].float() * e3, kgt3)
                strict3 = (rows3[:, None] > cols3[None, :])[None, :, :]
                Akk[tile_bhn, rows3, cols3] = akk3 * strict3.to(akk3.dtype)

    return A, Akk


@helion.kernel()
def chunk_fwd_A_diag_anchored_varlen_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    token_base: torch.Tensor,
    valid_len: torch.Tensor,
    A: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    l2norm_q: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> None:
    """chunk_fwd_A_diag_anchored_helion over a varlen batch, q/k read token-major.

    Row r addresses its tokens as in chunk_cumsum_gc_varlen_helion; gc, A and Akk
    are per-chunk and stay chunk-major [H * NT, C, *]. Per BC_DIAG-row sub-block,
    with gc_n = gc at the sub-block's first row (the decay anchor) and cols the live
    width 0 : (i + 1) * BC_DIAG:
        q_blk = q[rows_blk, h] * valid_blk             # [BC_DIAG, D], tail zeroed
        k_col = k[rows_col, h] * valid_col             # [cols, D]
        qg    = q_blk * exp2(RCP_LN2 * (gc_blk - gc_n))       # [BC_DIAG, D]
        kg    = k_col * exp2(RCP_LN2 * (gc_n - gc[cols]))     # [cols, D]
        A[blk]   = (qg @ kg.T) * scale * causal        # [BC_DIAG, cols], t >= s
        Akk[blk] = (k_blk * exp2(RCP_LN2 * (gc_blk - gc_n))) @ kg.T * strict
    Anchoring at gc_n bounds each exponent to BC_DIAG rows, so exp2 stays in range,
    exactly as in the dense kernel. Both grams always build; kda is the only varlen
    variant.

    A zeroed k column gives kg[s] = 0, hence A[:, s] = 0, so no key from the next
    sequence enters the gram; zeroed rows are masked again by the output store.

    The sub-blocks are written out, one per tier, each guarded on the chunk size:
        tier 0:  rows  0..15,  cols  0..15      always
        tier 1:  rows 16..31,  cols  0..31      C >= 2 * BC_DIAG
        tier 2:  rows 32..47,  cols  0..47      C >= 3 * BC_DIAG
        tier 3:  rows 48..63,  cols  0..63      C >= 4 * BC_DIAG
    so C = 32 takes tiers 0-1 and C = 64 all four; the caller rejects any other C.
    cols is the width the causal mask leaves nonzero, so both matmuls span
    BC_DIAG * (1 + 2 + ... + NC) = C * (NC + 1) / 2 columns instead of NC * C, and
    the caller's zeroed A and Akk already hold what the full-width form would store
    past it.
    Unrolled rather than tiled because each width must be a literal: hl.arange
    rejects arithmetic on a specialized size. FLA's KDA unrolls the same way in
    chunk_intra.py, on the same NC >= 3 / NC >= 4 guards."""
    NT = token_base.size(0)
    C = hl.specialize(gc.size(1))
    D = hl.specialize(gc.size(2))
    for tile_r in hl.tile(A.size(0), block_size=1):
        r = tile_r.begin
        j = r % NT
        h = r // NT
        base = token_base[j]
        vlen = valid_len[j]
        dcols = hl.arange(D)

        # Sub-block 0: its live columns are its own rows, so the row operands serve
        # as the column operands and k is read once.
        rows0 = hl.arange(16)
        cols0 = hl.arange(16)
        m0 = rows0 < vlen
        q0 = torch.where(
            m0[:, None],
            hl.load(q, [base + rows0, h, dcols], extra_mask=m0[:, None]),
            0,
        ).float()  # [BC_DIAG, D]
        if l2norm_q:
            q0 = q0 * torch.rsqrt((q0 * q0).sum(dim=-1, keepdim=True) + 1e-6)
        k0 = torch.where(
            m0[:, None],
            hl.load(k, [base + rows0, h, dcols], extra_mask=m0[:, None]),
            0,
        ).float()
        g0 = gc[r, rows0, :].float()
        n0 = gc[r, 0, :].float()
        e0 = torch.exp2((g0 - n0[None, :]) * RCP_LN2)
        kgt0 = (k0 * torch.exp2((n0[None, :] - g0) * RCP_LN2)).transpose(-2, -1)
        a0 = hl.dot(q0 * e0, kgt0) * scale
        A[r, rows0, cols0] = a0 * (rows0[:, None] >= cols0[None, :]).to(a0.dtype)
        akk0 = hl.dot(k0 * e0, kgt0)
        Akk[r, rows0, cols0] = akk0 * (rows0[:, None] > cols0[None, :]).to(akk0.dtype)

        if C >= 2 * BC_DIAG:
            rows1 = 16 + hl.arange(16)
            cols1 = hl.arange(32)
            m1 = rows1 < vlen
            mc1 = cols1 < vlen
            q1 = torch.where(
                m1[:, None],
                hl.load(q, [base + rows1, h, dcols], extra_mask=m1[:, None]),
                0,
            ).float()
            if l2norm_q:
                q1 = q1 * torch.rsqrt((q1 * q1).sum(dim=-1, keepdim=True) + 1e-6)
            k1 = torch.where(
                m1[:, None],
                hl.load(k, [base + rows1, h, dcols], extra_mask=m1[:, None]),
                0,
            ).float()
            kc1 = torch.where(
                mc1[:, None],
                hl.load(k, [base + cols1, h, dcols], extra_mask=mc1[:, None]),
                0,
            ).float()
            n1 = gc[r, 16, :].float()
            e1 = torch.exp2((gc[r, rows1, :].float() - n1[None, :]) * RCP_LN2)
            ec1 = torch.exp2((n1[None, :] - gc[r, cols1, :].float()) * RCP_LN2)
            kgt1 = (kc1 * ec1).transpose(-2, -1)
            a1 = hl.dot(q1 * e1, kgt1) * scale
            A[r, rows1, cols1] = a1 * (rows1[:, None] >= cols1[None, :]).to(a1.dtype)
            akk1 = hl.dot(k1 * e1, kgt1)
            Akk[r, rows1, cols1] = akk1 * (rows1[:, None] > cols1[None, :]).to(
                akk1.dtype
            )

        if C >= 3 * BC_DIAG:
            rows2 = 32 + hl.arange(16)
            cols2 = hl.arange(48)
            m2 = rows2 < vlen
            mc2 = cols2 < vlen
            q2 = torch.where(
                m2[:, None],
                hl.load(q, [base + rows2, h, dcols], extra_mask=m2[:, None]),
                0,
            ).float()
            if l2norm_q:
                q2 = q2 * torch.rsqrt((q2 * q2).sum(dim=-1, keepdim=True) + 1e-6)
            k2 = torch.where(
                m2[:, None],
                hl.load(k, [base + rows2, h, dcols], extra_mask=m2[:, None]),
                0,
            ).float()
            kc2 = torch.where(
                mc2[:, None],
                hl.load(k, [base + cols2, h, dcols], extra_mask=mc2[:, None]),
                0,
            ).float()
            n2 = gc[r, 32, :].float()
            e2 = torch.exp2((gc[r, rows2, :].float() - n2[None, :]) * RCP_LN2)
            ec2 = torch.exp2((n2[None, :] - gc[r, cols2, :].float()) * RCP_LN2)
            kgt2 = (kc2 * ec2).transpose(-2, -1)
            a2 = hl.dot(q2 * e2, kgt2) * scale
            A[r, rows2, cols2] = a2 * (rows2[:, None] >= cols2[None, :]).to(a2.dtype)
            akk2 = hl.dot(k2 * e2, kgt2)
            Akk[r, rows2, cols2] = akk2 * (rows2[:, None] > cols2[None, :]).to(
                akk2.dtype
            )

        if C >= 4 * BC_DIAG:
            rows3 = 48 + hl.arange(16)
            cols3 = hl.arange(64)
            m3 = rows3 < vlen
            mc3 = cols3 < vlen
            q3 = torch.where(
                m3[:, None],
                hl.load(q, [base + rows3, h, dcols], extra_mask=m3[:, None]),
                0,
            ).float()
            if l2norm_q:
                q3 = q3 * torch.rsqrt((q3 * q3).sum(dim=-1, keepdim=True) + 1e-6)
            k3 = torch.where(
                m3[:, None],
                hl.load(k, [base + rows3, h, dcols], extra_mask=m3[:, None]),
                0,
            ).float()
            kc3 = torch.where(
                mc3[:, None],
                hl.load(k, [base + cols3, h, dcols], extra_mask=mc3[:, None]),
                0,
            ).float()
            n3 = gc[r, 48, :].float()
            e3 = torch.exp2((gc[r, rows3, :].float() - n3[None, :]) * RCP_LN2)
            ec3 = torch.exp2((n3[None, :] - gc[r, cols3, :].float()) * RCP_LN2)
            kgt3 = (kc3 * ec3).transpose(-2, -1)
            a3 = hl.dot(q3 * e3, kgt3) * scale
            A[r, rows3, cols3] = a3 * (rows3[:, None] >= cols3[None, :]).to(a3.dtype)
            akk3 = hl.dot(k3 * e3, kgt3)
            Akk[r, rows3, cols3] = akk3 * (rows3[:, None] > cols3[None, :]).to(
                akk3.dtype
            )


@helion.kernel()
def chunk_fwd_A_intra_scalar_gate_varlen_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    token_base: torch.Tensor,
    valid_len: torch.Tensor,
    A: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    l2norm_q: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> None:
    """Both intra-chunk grams in full, for a gate that is one scalar per token.

    Replaces chunk_fwd_A_diag_anchored_varlen_helion on that gate, which is the one
    gated_delta_rule has. Row r addresses its tokens as in
    chunk_cumsum_gc_varlen_helion; gc, A and Akk are per-chunk and stay chunk-major
    [H * NT, C, *], with gc one channel wide.

    A per-channel gate has to carry its decay inside the reduction over D, which is
    what forces the anchored kernel's sub-blocks: each folds exp2(gc - gc_n) into its
    own operands, so the gram is written a tier at a time and the tail of each row is
    left to the caller's zeroed buffers. A scalar gate is constant over D, so it
    leaves the sum entirely:
        A[i, j]   = scale * (q[i] @ k[j]) * exp2(RCP_LN2 * (gc[i] - gc[j]))
    which is one matmul for the whole C-square gram and an elementwise [C, C] decay.
    Every entry is written, the masked ones as zero, so this path needs no cleared
    buffer to write into.

    gc falls with the token index, so the exponent is non-positive everywhere the
    causal mask keeps, and exp2 is bounded by 1 at any gate scale. It is clamped
    there rather than masked afterwards, since rows past a sequence's end hold a gc
    that does not follow the fall and would otherwise overflow before the mask.

    A zeroed row of q or k gives a zero row and column, so no token from the next
    sequence enters either gram."""
    NT = token_base.size(0)
    C = hl.specialize(gc.size(1))
    D = hl.specialize(q.size(2))
    for tile_r in hl.tile(A.size(0), block_size=1):
        r = tile_r.begin
        j = r % NT
        h = r // NT
        base = token_base[j]
        vlen = valid_len[j]
        rows = hl.arange(C)
        # A gram's column index must be a separate arange from its row index.
        cols = hl.arange(C)
        m = (rows < vlen)[:, None]
        dcols = hl.arange(D)
        qi = torch.where(
            m, hl.load(q, [base + rows, h, dcols], extra_mask=m), 0
        ).float()
        ki = torch.where(
            m, hl.load(k, [base + rows, h, dcols], extra_mask=m), 0
        ).float()
        if l2norm_q:
            qi = qi * torch.rsqrt((qi * qi).sum(dim=-1, keepdim=True) + 1e-6)
        gr = hl.load(gc, [r, rows, 0]).float()
        gk = hl.load(gc, [r, cols, 0]).float()
        # The grams are stored in q's dtype, so the products are taken there, with
        # hl.dot accumulating in fp32. The decay stays fp32 and applies after.
        kt = ki.transpose(-2, -1).to(A.dtype)
        gram_qk = hl.dot(qi.to(A.dtype), kt).float()
        gram_kk = hl.dot(ki.to(A.dtype), kt).float()
        diff = gr[:, None] - gk[None, :]
        dec = torch.exp2(torch.where(diff < 0, diff, 0.0) * RCP_LN2)
        # Akk excludes the diagonal, which the wy transform takes as zero; A keeps it,
        # where the decay is exp2(0) = 1.
        A[r, rows, cols] = torch.where(
            rows[:, None] >= cols[None, :], gram_qk * dec * scale, 0.0
        ).to(A.dtype)
        Akk[r, rows, cols] = torch.where(
            rows[:, None] > cols[None, :], gram_kk * dec, 0.0
        ).to(Akk.dtype)


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


@helion.kernel()
def chunk_fwd_o_diag_anchored_varlen_helion(
    q: torch.Tensor,
    v: torch.Tensor,
    gc: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    token_base: torch.Tensor,
    valid_len: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    l2norm_q: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    scalar_gate: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> None:
    """chunk_fwd_o_diag_anchored_helion over a varlen batch, q read and out written
    token-major.

    Row r addresses its tokens as in chunk_cumsum_gc_varlen_helion; v (the corrected
    v_new), gc, h and A are per-chunk and stay chunk-major. A is the pre-masked
    anchored score matrix, so the output is the dense one:
        q_i     = q[rows, h] * valid                 # [C, D], tail zeroed
        qg      = q_i * exp2(RCP_LN2 * gc)           # [C, D]
        o_cross = (qg @ h) * scale                   # [C, DV] state term
        o_intra = A @ v                              # [C, DV] intra-chunk term
        out[rows, h] = o_cross + o_intra   where valid    # [C, DV]
    Storing under valid is what protects the boundary: a row past a sequence's end is
    never written, so it cannot overwrite the next sequence's first tokens. That makes
    the write the inverse of the token-major reads, with no scatter pass.

    scalar_gate=True takes gc one channel wide, the cumulative sum of a gate that is
    one scalar per token, and broadcasts it over the channel axis.

    A chunk owns one program and takes D and DV whole, so the body has no loop over
    either; a DV loop would re-read q, gc and A per iteration."""
    NT = token_base.size(0)
    C = hl.specialize(gc.size(1))
    D = hl.specialize(q.size(2))
    DV = hl.specialize(out.size(2))

    for tile_r in hl.tile(A.size(0), block_size=1):
        j = tile_r.begin % NT
        head = tile_r.begin // NT
        base = token_base[j]
        idx = hl.arange(C)
        valid = idx < valid_len[j]
        dcols = hl.arange(D)
        vcols = hl.arange(DV)

        qt = torch.where(
            valid[:, None],
            hl.load(q, [base + idx, head, dcols], extra_mask=valid[:, None]),
            0,
        ).float()  # [C, D]
        if l2norm_q:
            qt = qt * torch.rsqrt((qt * qt).sum(dim=-1, keepdim=True) + 1e-6)
        if scalar_gate:
            g_col = gc[tile_r.begin, :, 0].float()  # [C]
            qg = (qt * torch.exp2(g_col * RCP_LN2)[:, None]).to(q.dtype)
        else:
            gct = gc[tile_r.begin, :, :]
            qg = (qt * torch.exp2(gct * RCP_LN2)).to(q.dtype)
        ht = h[tile_r.begin, :, :]
        o_cross = hl.dot(qg, ht.to(qg.dtype)) * scale

        vt = v[tile_r.begin, :, :]
        At = A[tile_r.begin, :, :]
        o_intra = hl.dot(At.to(vt.dtype), vt)
        hl.store(
            out,
            [base + idx, head, vcols],
            (o_cross + o_intra).to(out.dtype),
            extra_mask=valid[:, None],
        )


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
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        lower_bound: float | None = None,
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
                raise NotImplementedError("beta correction with decay requires a=None")
            o, h_all, v_new_all, A_inv, w_wy, final_state = _helion_chunked_fwd_delta(
                q,
                k,
                v,
                beta,
                a,
                C,
                g=g,
                initial_state=initial_state,
                return_final_state=return_final_state,
                scale=scale,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
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
            return dq, dk, dv, dg, None, None, None, None, None, None, None, None, None

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
                scale=ctx.scale,
            )
            return dq, dk, dv, dg, dbeta, None, None, None, None, None, None, None, None

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
            scale=ctx.scale,
        )
        return dq, dk, dv, dg, dbeta, da, None, None, None, None, None, None, None


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
        has_h0 = initial_state is not None
        state = _init_state(initial_state, BH, D, DV, q) if has_h0 else None
        h_all = chunk_fwd_h_diag_fused(
            k_state_4d, v_flat, None, state, use_g=False, has_h0=has_h0
        )

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
        g_cs = g.reshape(BH, N, C).cumsum(-1, dtype=torch.float32)  # [BH, N, C]
        g_last = g_cs[:, :, -1]  # [BH, N]

        k_4d = k.reshape(BH, N, C, D)
        v_flat = v.reshape(BH, N, C, DV)
        has_h0 = initial_state is not None
        state = _init_state(initial_state, BH, D, DV, q) if has_h0 else None
        h_all = chunk_fwd_h_diag_fused(
            k_4d, v_flat, g_last, state, gc=g_cs, scalar_decay=True, has_h0=has_h0
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
    gc = chunk_cumsum_gc_helion(g.reshape(BHN, C, D))
    gc4 = gc.reshape(BH, N, C, D)

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
    scale: float = 1.0,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
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
        g_cs_flat = chunk_cumsum_gc_helion(
            g.reshape(BHN, C, D),
            A_log,
            dt_bias,
            lower_bound,
            H,
            N,
            use_gate=A_log is not None,
            has_bias=dt_bias is not None,
            use_lower_bound=lower_bound is not None,
        )
        g_cs = g_cs_flat.reshape(BH, N, C, D)
        decay_last = g_cs[:, :, -1, :]

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
        Aqk, Akk = chunk_fwd_A_diag_anchored_helion(
            qf, kf, g_cs_flat, scale, build_kk=True
        )

    # The anchored key the serial state pass consumes shares the WY transform's
    # k read and per-channel decay, so the WY kernel emits it.
    k_state_flat = torch.empty_like(kf) if diag_anchored else None
    w, u, A_inv = chunk_fwd_wy_delta_helion(
        af,
        vf,
        bf,
        g_cs_flat,
        Akk,
        k_state_flat,
        scalar_decay=scalar_decay,
        diag_anchored=diag_anchored,
    )

    k4 = k.reshape(BH, N, C, D)
    k_state4 = k_state_flat.reshape(BH, N, C, D) if k_state_flat is not None else k4
    w4 = w.reshape(BH, N, C, D)
    u4 = u.reshape(BH, N, C, DV)
    state = _init_state(initial_state, BH, D, DV, q).to(k.dtype)
    h_all, v_new_all = chunk_fwd_h_delta_helion(
        k_state4,
        w4,
        u4,
        state,
        g_cs,
        decay_last,
        scalar_decay=scalar_decay,
        diag_anchored=diag_anchored,
        k_pre_scaled=diag_anchored,
    )

    hf = h_all.reshape(BHN, D, DV)
    vnewf = v_new_all.reshape(BHN, C, DV)

    if diag_anchored:
        # o = (q * exp2(gc)) @ h + Aqk @ v_new (anchored, per-channel decay).
        # Aqk already carries scale; the kernel applies scale to the q@h cross-term.
        o = chunk_fwd_o_diag_anchored_helion(qf, vnewf, g_cs_flat, hf, Aqk, scale)
    else:
        o = chunk_fwd_o_helion(
            qf, kf, vnewf, g_cs_flat, hf, use_g=scalar_decay, scale=scale
        )

    if diag_anchored:
        # Attach cached data to h_all for the backward to use.
        h_all._kda_bwd_cache = (g_cs_flat, Aqk, Akk)  # pyrefly: ignore

    final_state = None
    if return_final_state:
        # Add the last chunk's writes to its entering state; keys are decayed to
        # the chunk end (per-channel when diagonal), values are the corrected v_new.
        if diag_anchored:
            assert g_cs is not None
            assert decay_last is not None
            k_state_last = (
                k4[:, -1].float() * torch.exp(g_cs[:, -1, -1:, :] - g_cs[:, -1])
            ).to(k.dtype)
            g_last_arg = decay_last[:, -1]
        elif scalar_decay:
            assert g_cs is not None
            assert decay_last is not None
            k_state_last = (
                k4[:, -1].float()
                * torch.exp(decay_last[:, -1, None, None] - g_cs[:, -1, :, None])
            ).to(k.dtype)
            g_last_arg = decay_last.unsqueeze(-1).expand(-1, -1, D)[:, -1]
        else:
            k_state_last = a_use.reshape(BH, N, C, D)[:, -1]
            g_last_arg = None
        final_state = _final_state_from_h_all(
            h_all[:, -1],
            k_state_last,
            v_new_all[:, -1],
            g_last_arg,
            B,
            H,
            use_g=g is not None,
        )

    return o.reshape(B, H, T, DV), h_all, v_new_all, A_inv, w, final_state


@functools.lru_cache(maxsize=4)
def _kda_varlen_chunk_tables(
    cu_seqlens: torch.Tensor,
    C: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Place every chunk: sequence n owns chunk_offsets[n] : chunk_offsets[n + 1], and
    chunk j of it starts at token_base[j] with valid_len[j] rows of its own.
        lens          = diff(cu_seqlens)                    # [N]
        chunk_offsets = pad(cumsum(ceil(lens / C)), 1, 0)   # [N + 1]
        token_base[j] = cu_seqlens[n] + i * C               # [NT], i within seq n
        valid_len[j]  = min(lens[n] - i * C, C)             # [NT], rows in 1..C

    Every table is a pure function of cu_seqlens and C, so a caller that reuses one
    cu_seqlens across forwards, as a stack of layers does, builds them once. NT comes
    from int(chunk_offsets[-1]), a device-to-host sync, so rebuilding them per call
    also drains the launch queue between forwards.

    Keyed by identity, which is what a tensor hashes as, so refilling one cu_seqlens
    in place returns that tensor's first tables: a new batch needs a new tensor, not
    an edited one. Comparing values instead would need the same device-to-host sync
    this exists to skip. FLA holds its own varlen tables under the same rule, in
    fla.utils.tensor_cache.
    """
    N = cu_seqlens.numel() - 1
    device = cu_seqlens.device
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_offsets = torch.nn.functional.pad(((lens + C - 1) // C).cumsum(0), (1, 0))
    counts = chunk_offsets[1:] - chunk_offsets[:-1]
    seq_id = torch.repeat_interleave(torch.arange(N, device=device), counts)
    NT = int(chunk_offsets[-1])
    local = torch.arange(NT, device=device) - chunk_offsets[seq_id]
    token_base = cu_seqlens[seq_id] + local * C
    valid_len = (lens[seq_id] - local * C).clamp(max=C)
    return chunk_offsets, token_base, valid_len, NT


def _helion_chunked_fwd_kda_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    C: int,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
    scale: float = 1.0,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    l2norm_q: bool = False,
    scalar_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """KDA forward over a varlen batch (cu_seqlens), forward only.

    Inputs are token-major [T_total, H, *] with cu_seqlens [N+1] marking the
    boundaries. The host tables place every chunk, and the five kernels mirror the
    dense diag_anchored pipeline:
        chunk_offsets [N+1], token_base [NT], valid_len [NT]
        NT = sum(ceil(len_n / C))            # total chunks, ragged
        gc            = cumsum_gc_varlen(g)              # [H*NT, C, D] fp32
        Aqk, Akk      = A_diag_anchored_varlen(q, k, gc) # [H*NT, C, C]
        w, u, k_state = wy_delta_varlen(k, v, beta, gc, Akk)
        h_all, v_new, ht = h_delta_varlen(k_state, w, u, h0, gc[C-1], chunk_offsets)
        o             = o_diag_anchored_varlen(q, v_new, gc, h_all, Aqk)

    scalar_gate=True is the gated_delta_rule case, where the gate is one scalar per
    (token, head) and g arrives [T_total, H, 1]. Broadcast over the key dimension that
    gate is diag(exp(g)) == exp(g) * I, so this same pipeline computes it: gc is one
    channel wide throughout, and the gram pass is
    A_intra_scalar_gate_varlen(q, k, gc), which takes each gram in a single matmul
    because a decay constant over D leaves the reduction.

    q/k/v/g/beta are never copied: each kernel reads them where they lie and zeros
    the rows past a sequence's end. Only the intermediates above are materialized, as
    the dense path materializes its own. The state pass walks one sequence at a time,
    so the recurrence resets at every boundary and ht is the final state per sequence
    with no host-side last-chunk arithmetic. The output kernel stores token-major
    under the same mask, so there is no scatter pass.

    T_total < C is padded up to C, since a chunk addresses C rows and the store needs
    them in bounds; valid_len still marks only the real tokens, so the padding reads
    as zero and o is returned at the true T_total.
    """
    T_total, H, D = q.shape
    if T_total < C:
        extra = C - T_total
        pad = lambda x: torch.cat([x, x.new_zeros(extra, *x.shape[1:])])  # noqa: E731
        q, k, v, g, beta = pad(q), pad(k), pad(v), pad(g), pad(beta)
    DV = v.shape[-1]
    N = cu_seqlens.numel() - 1

    chunk_offsets, token_base, valid_len, NT = _kda_varlen_chunk_tables(cu_seqlens, C)

    HNT = H * NT
    # One channel on the scalar-gate path, where g arrives [T_total, H, 1], D on the
    # per-channel one. Everything downstream reads g_cs at whichever width it lands.
    g_cs = torch.empty(HNT, C, g.size(2), dtype=torch.float32, device=q.device)
    chunk_cumsum_gc_varlen_helion(
        g,
        token_base,
        valid_len,
        g_cs,
        A_log,
        dt_bias,
        lower_bound,
        use_gate=A_log is not None,
        has_bias=dt_bias is not None,
        use_lower_bound=lower_bound is not None,
    )
    decay_last = g_cs[:, C - 1, :]

    # The anchored kernel writes only the causal blocks of each gram and leaves the
    # rest to these zeros; the scalar-gate one writes every entry, so it needs none.
    alloc_A = torch.empty if scalar_gate else torch.zeros
    Aqk = alloc_A(HNT, C, C, dtype=q.dtype, device=q.device)
    Akk = alloc_A(HNT, C, C, dtype=q.dtype, device=q.device)
    if scalar_gate:
        chunk_fwd_A_intra_scalar_gate_varlen_helion(
            q, k, g_cs, token_base, valid_len, Aqk, Akk, scale, l2norm_q=l2norm_q
        )
    else:
        chunk_fwd_A_diag_anchored_varlen_helion(
            q, k, g_cs, token_base, valid_len, Aqk, Akk, scale, l2norm_q=l2norm_q
        )

    w = torch.empty(HNT, C, D, dtype=k.dtype, device=k.device)
    u = torch.empty(HNT, C, DV, dtype=v.dtype, device=v.device)
    k_state = torch.empty(HNT, C, D, dtype=k.dtype, device=k.device)
    chunk_fwd_wy_delta_varlen_helion(
        k,
        v,
        beta,
        g_cs,
        Akk,
        token_base,
        valid_len,
        w,
        u,
        k_state,
        scalar_gate=scalar_gate,
    )

    # The chunk operands are head-major (row h * NT + j), so the state pass indexes
    # its own [H * N] axis the same way: row h * N + n. initial_state arrives in
    # FLA's [N, H, D, DV] order, hence the transpose in and back out.
    if initial_state is not None:
        h0 = (
            initial_state.reshape(N, H, D, DV)
            .transpose(0, 1)
            .reshape(H * N, D, DV)
            .float()
            .contiguous()
        )
    else:
        h0 = q.new_zeros(H * N, D, DV, dtype=torch.float32)

    h_all, v_new, ht = chunk_fwd_h_delta_varlen_helion(
        k_state, w, u, h0, decay_last, chunk_offsets, NT, H
    )

    # Sized from the padded q so the masked store stays in bounds, then sliced back
    # to the real token count on return.
    o = q.new_zeros(q.size(0), H, DV)
    chunk_fwd_o_diag_anchored_varlen_helion(
        q,
        v_new,
        g_cs,
        h_all,
        Aqk,
        token_base,
        valid_len,
        o,
        scale,
        l2norm_q=l2norm_q,
        scalar_gate=scalar_gate,
    )
    o = o[:T_total]

    final_state = None
    if return_final_state:
        final_state = ht.reshape(H, N, D, DV).transpose(0, 1).contiguous()

    return o, final_state


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

        dhf2 = dh_all.reshape(BHN, D, DV)
        qb = q.reshape(BHN, C, D)
        kb = k.reshape(BHN, C, D)
        vb = v.reshape(BHN, C, DV)
        dob = grad_output.reshape(BHN, C, DV)
        hb = h_all.reshape(BHN, D, DV)

        dq_raw, dk_raw = chunk_bwd_dqk_helion(
            qb, kb, vb, None, None, hb, dob, dhf2, use_g=False, scale=scale
        )
        dv_raw = chunk_bwd_dv_helion(
            qb, kb, kb, None, dob, dhf2, use_g=False, scale=scale
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
            g_cs = g.reshape(BH, N, C).cumsum(-1, dtype=torch.float32)
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
            qb,
            kb,
            vb,
            g_csf2,
            hb,
            dob,
            dhf2,
            g_last=g_lastf2,
            compute_dg=needs_dg,
            scale=scale,
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
    scale: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
]:
    """DeltaNet / Gated DeltaNet backward (beta correction), q unscaled.

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
        assert g is not None
        g_cs = g.float().reshape(BH, N, C).cumsum(-1)
        decay_last = g_cs[:, :, -1].contiguous()  # [BH, N]
        decay_lastf = decay_last.reshape(BHN)
        g_cs = g_cs.reshape(BH, N, C)
    g_cs_flat = None
    if scalar_decay:
        assert g_cs is not None
        g_cs_flat = g_cs.reshape(BHN, C)

    a4 = a_use.reshape(BH, N, C, D)
    w4 = w_wy.reshape(BH, N, C, D)
    q4 = q.reshape(BH, N, C, D)
    do4 = grad_output.reshape(BH, N, C, DV)

    dh0 = q.new_zeros(BH, D, DV, dtype=k.dtype)
    # do enters the backward through two independent kernels (dstate and dqkw);
    # each scales its own do load once. dstate's outputs dS_future/dvni already
    # carry scale, so dqkw must scale only its direct do load, not those.
    dS_future, dvni4 = chunk_bwd_dstate_delta_helion(
        q4, a4, w4, do4, dh0, g_cs, decay_last, scalar_decay=scalar_decay, scale=scale
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
        qf,
        af,
        hf,
        vnewf,
        dof,
        dvnif,
        dSf,
        g_cs_flat,
        decay_lastf,
        scalar_decay=scalar_decay,
        scale=scale,
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
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """KDA backward (beta correction + diagonal decay), q unscaled.

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

    qf = q.reshape(BHN, C, D)
    kf = k.reshape(BHN, C, D)
    vf = v.reshape(BHN, C, DV)
    bf = beta.reshape(BHN, C)
    hf = h_all.reshape(BHN, D, DV)
    vnf = v_new_all.reshape(BHN, C, DV)
    Tinv = A_inv
    dof = grad_output.reshape(BHN, C, DV)

    # Reuse the forward's cached cumsum and anchored grams when available.
    bwd_cache = getattr(h_all, "_kda_bwd_cache", None)
    if bwd_cache is not None:
        g_cs_flat, Aqk, Akk = bwd_cache
    else:
        g_cs_flat = chunk_cumsum_gc_helion(g.reshape(BHN, C, D))
        Aqk, Akk = chunk_fwd_A_diag_anchored_helion(
            qf, kf, g_cs_flat, scale, build_kk=True
        )
    gc4 = g_cs_flat.reshape(BH, N, C, D)

    # scale folds into the do load here; dh/dAqk/dv_new carry it downstream, so the
    # state kernels and gram2 (fed dAqk/dAkk) stay at 1.0 to avoid double-counting.
    dq_o, dgc_o, dh, dAqk, dv_new = chunk_bwd_o_kda_helion(
        qf, vnf, hf, Aqk, g_cs_flat, dof, scale
    )

    k4 = k.reshape(BH, N, C, D)
    w4 = w_wy.reshape(BH, N, C, D)
    h4 = h_all.reshape(BH, N, D, DV)
    vn4 = v_new_all.reshape(BH, N, C, DV)
    dvnew4 = dv_new.reshape(BH, N, C, DV)
    dh4 = dh.reshape(BH, N, D, DV)
    dS_scratch = q.new_zeros(BH, D, DV, dtype=torch.float32)
    du4, dS_save = chunk_bwd_state_du_kda_helion(k4, w4, gc4, dvnew4, dh4, dS_scratch)
    dw4, dk_s4, dgc_s4 = chunk_bwd_state_dwk_kda_helion(k4, gc4, h4, vn4, du4, dS_save)
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
    A_log: torch.Tensor | None = ...,
    dt_bias: torch.Tensor | None = ...,
    lower_bound: float | None = ...,
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
    A_log: torch.Tensor | None = ...,
    dt_bias: torch.Tensor | None = ...,
    lower_bound: float | None = ...,
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
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
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
        q,
        k,
        v,
        g,
        beta,
        a,
        C,
        initial_state,
        return_final_state,
        scale,
        A_log,
        dt_bias,
        lower_bound,
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
        q,
        k,
        v,
        g,
        C=C,
        scale=scale,
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
        q,
        k,
        v,
        g,
        C=C,
        scale=scale,
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
        q,
        k,
        v,
        None,
        beta=beta,
        C=C,
        scale=scale,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


def _varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    C: int,
    cu_seqlens: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    return_final_state: bool,
    state_v_first: bool,
    l2norm_q: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    scalar_gate: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """The varlen entry both KDA and gated_delta_rule take: check, then dispatch.

    Inputs are token-major [1, T_total, H, *] and initial_state is already in
    [N, H, D, DV] order, whatever layout the caller was handed; state_v_first here
    applies to the returned state alone.
    """
    if q.size(0) != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.size(0)} when "
            "using `cu_seqlens`. Please flatten variable-length inputs before "
            "processing."
        )
    # The gram kernels assert the same bound; this raises the caller-facing error
    # before a kernel trace does.
    if C not in (2 * BC_DIAG, 4 * BC_DIAG):
        raise ValueError(
            f"chunk size C must be {2 * BC_DIAG} or {4 * BC_DIAG} on the "
            f"cu_seqlens path, got {C}"
        )
    # Every kernel indexes tokens through cu_seqlens, so a vector that does not
    # partition [0, T_total) reads the wrong rows rather than failing: an end
    # below T_total leaves that much of the output at its zero initialization.
    if int(cu_seqlens[-1]) != q.size(1):
        raise ValueError(
            f"cu_seqlens must end at T_total={q.size(1)}, got {int(cu_seqlens[-1])}"
        )
    if int(cu_seqlens[0]) != 0 or not bool((cu_seqlens[1:] > cu_seqlens[:-1]).all()):
        raise ValueError("cu_seqlens must start at 0 and strictly increase")
    # Grouped-query: give every query head its own key/value head, matching what
    # chunked_linear_attn does for the dense path.
    H_kv = k.size(2)
    if H_kv < q.size(2):
        assert q.size(2) % H_kv == 0
        n_rep = q.size(2) // H_kv
        k = k.repeat_interleave(n_rep, dim=2)
        v = v.repeat_interleave(n_rep, dim=2)
    o, final_state = _helion_chunked_fwd_kda_varlen(
        q[0],
        k[0],
        v[0],
        g[0],
        beta[0],
        C,
        cu_seqlens,
        initial_state=initial_state,
        return_final_state=return_final_state,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        l2norm_q=l2norm_q,
        scalar_gate=scalar_gate,
    )
    o = o.unsqueeze(0)
    if return_final_state:
        assert final_state is not None
        return o, final_state.transpose(-2, -1) if state_v_first else final_state
    return o


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
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    state_v_first: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet, the delta rule under one scalar decay per token and head.

    use_qk_l2norm_in_kernel=True takes q and k raw and norms them per head over D,
    the transform the model applies before the call:
            q = q / ||q||,  k = k / ||k||
    It has no backward, so setting it on an input with requires_grad=True raises
    NotImplementedError.

    cu_seqlens [N+1] switches to a variable-length batch, as it does for KDA.
    Inputs are then token-major [1, T_total, H, D] with g [1, T_total, H], one
    scalar per token and head, rather than the head-first layout of the dense path:
    sequence n spans tokens cu_seqlens[n] : cu_seqlens[n+1] and the recurrence
    restarts at each boundary. The output matches its inputs, [1, T_total, H, DV],
    and initial_state / the returned final state are [N, H, D, DV], one per
    sequence. This path is forward only.

    Broadcast over the key dimension the gate is diag(exp(g)) == exp(g) * I, so the
    varlen KDA pipeline computes this variant exactly, reading g one channel wide;
    _helion_chunked_fwd_kda_varlen documents what that saves.

    state_v_first holds the state as [N, H, DV, D] instead of [N, H, D, DV], matching
    what vLLM uses; the flag name is FLA's. It applies to initial_state and the
    returned final state alike, so a returned state feeds straight back in.
    """
    assert g is not None
    assert beta is not None
    needs_grad = any(
        t is not None and t.requires_grad for t in (q, k, v, g, beta, initial_state)
    )
    if use_qk_l2norm_in_kernel and needs_grad:
        raise NotImplementedError(
            "the in-kernel l2 norm is forward-only; call with the flag off and "
            "norm the inputs outside the kernel to keep gradients"
        )

    if state_v_first and initial_state is not None:
        initial_state = initial_state.transpose(-2, -1).contiguous()

    # q's norm rides the gram kernel on the varlen path, so only k is normed here.
    fuse_q_l2norm = use_qk_l2norm_in_kernel and cu_seqlens is not None
    if use_qk_l2norm_in_kernel:
        norm = lambda t: l2norm_fwd_helion(t.reshape(-1, t.size(-1))).view_as(t)  # noqa: E731
        k = norm(k)
        if not fuse_q_l2norm:
            q = norm(q)

    if cu_seqlens is not None:
        if needs_grad:
            raise NotImplementedError(
                "the cu_seqlens path is forward-only; pass a dense batch "
                "to keep gradients"
            )
        return _varlen_forward(
            q,
            k,
            v,
            g.unsqueeze(-1),
            beta,
            C,
            cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            return_final_state=return_final_state,
            state_v_first=state_v_first,
            l2norm_q=fuse_q_l2norm,
            scalar_gate=True,
        )

    out = chunked_linear_attn(
        q,
        k,
        v,
        g,
        beta=beta,
        C=C,
        scale=scale,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )
    if return_final_state and state_v_first:
        o, final_state = out
        return o, final_state.transpose(-2, -1)
    return out


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
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    state_v_first: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """KDA, optionally applying the model's input transforms on the way in.

    All three flags default off, meaning the caller has already transformed its
    inputs. Turning one on moves that transform into the kernels:
      - use_qk_l2norm_in_kernel=True: q, k arrive raw, normed per head over D:
            q = q / ||q||,  k = k / ||k||
      - use_gate_in_kernel=True: g arrives pre-activation, A_log [H] required,
        dt_bias [H, D] and lower_bound optional:
            g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))   lower_bound set
            g = -exp(A_log) * softplus(g + dt_bias)                 otherwise
      - use_beta_sigmoid_in_kernel=True: beta arrives as logits:
            beta = sigmoid(beta)
    None of the three transforms has a backward, so a flag set on an input with
    requires_grad=True raises NotImplementedError.

    cu_seqlens [N+1] switches to a variable-length batch. Inputs are then
    token-major [1, T_total, H, D], the layout FLA, vLLM and FlashKDA all use for
    a varlen batch, rather than the head-first layout of the dense path: sequence n
    spans tokens cu_seqlens[n] : cu_seqlens[n+1] and the recurrence restarts at
    each boundary. The output matches its inputs, [1, T_total, H, DV], and
    initial_state / the returned final state are [N, H, D, DV], one per sequence.
    This path is forward only.

    state_v_first holds the state as [N, H, DV, D] instead of [N, H, D, DV], matching
    what vLLM and FlashKDA use; the flag name is FLA's. It applies to initial_state
    and the returned final state alike, so a returned state feeds straight back in.
    """
    assert g is not None
    assert beta is not None
    any_flag = (
        use_qk_l2norm_in_kernel or use_gate_in_kernel or use_beta_sigmoid_in_kernel
    )
    needs_grad = any(
        t is not None and t.requires_grad for t in (q, k, v, g, beta, initial_state)
    )
    if any_flag and needs_grad:
        raise NotImplementedError(
            "the in-kernel KDA preamble is forward-only; call with the flags off "
            "and transform the inputs outside the kernel to keep gradients"
        )

    if state_v_first and initial_state is not None:
        initial_state = initial_state.transpose(-2, -1).contiguous()

    fuse_q_l2norm = use_qk_l2norm_in_kernel and cu_seqlens is not None
    if use_qk_l2norm_in_kernel:
        norm = lambda t: l2norm_fwd_helion(t.reshape(-1, t.size(-1))).view_as(t)  # noqa: E731
        k = norm(k)
        if not fuse_q_l2norm:
            q = norm(q)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta)
    if use_gate_in_kernel:
        assert A_log is not None
    else:
        A_log = None
        dt_bias = None
        lower_bound = None

    if cu_seqlens is not None:
        if needs_grad:
            raise NotImplementedError(
                "the cu_seqlens path is forward-only; pass a dense batch "
                "to keep gradients"
            )
        return _varlen_forward(
            q,
            k,
            v,
            g,
            beta,
            C,
            cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            return_final_state=return_final_state,
            state_v_first=state_v_first,
            l2norm_q=fuse_q_l2norm,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )

    out = chunked_linear_attn(
        q,
        k,
        v,
        g,
        beta=beta,
        C=C,
        scale=scale,
        initial_state=initial_state,
        return_final_state=return_final_state,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    if return_final_state and state_v_first:
        o, final_state = out
        return o, final_state.transpose(-2, -1)
    return out


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
