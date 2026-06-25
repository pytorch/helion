"""
Gated Linear Attention (GLA) - chunked-parallel
===============================================

Linear attention with a data-dependent per-channel (vector) forget gate on the
key dimension, chunked-parallel form.

Recurrence (per timestep, g_t is a [D]-vector log-gate, b_t = sum_{u<=t} g_u):
    S_t = diag(exp(g_t)) @ S_{t-1} + k_t^T @ v_t
    o_t = scale * q_t @ S_t

Parallel form replaces the causal mask with a per-channel decay (gate cancels
inside the d-sum, so it cannot factor out of q @ k^T):
    o_n = scale * sum_{m<=n} (sum_d q[n,d] k[m,d] exp(b_n,d - b_m,d)) v_m.
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

# 1 / ln(2), to apply decays with exp2 instead of exp.
RCP_LN2 = 1.4426950408889634


# %%
# Host-side gate cumsum
# ---------------------


@helion.kernel()
def _chunk_local_cumsum_kernel(g: torch.Tensor) -> torch.Tensor:
    """Per-chunk inclusive cumsum of g over the chunk axis, scaled by RCP_LN2.

    The inclusive cumsum over the chunk axis C is a lower-triangular ones-matmul
    L @ g (L[t,u] = 1 iff u <= t) in fp32 on tensor cores.

    Args:
        g: [BHN, C, D] log-gate (chunks already flattened into the leading dim)
    Returns:
        gc: [BHN, C, D] fp32, gc[i, t, :] = RCP_LN2 * sum_{u<=t} g[i, u, :]
    """
    BHN = g.size(0)
    C = hl.specialize(g.size(1))
    D = g.size(2)

    gc_out = torch.empty([BHN, C, D], dtype=torch.float32, device=g.device)

    for tile_bhn, tile_d in hl.tile([BHN, D]):
        gt = g[tile_bhn, :, tile_d].float()  # [tile_bhn, C, tile_d]
        idx = hl.arange(C)
        # L[t,u] = 1 iff u <= t (lower-triangular incl. diagonal): inclusive cumsum.
        lower = (idx[:, None] >= idx[None, :]).to(torch.float32)  # [C, C]
        gc = hl.dot(lower[None, :, :].broadcast_to(tile_bhn, C, C), gt) * RCP_LN2
        gc_out[tile_bhn, :, tile_d] = gc

    return gc_out


def _chunk_local_cumsum_base2(g: torch.Tensor, C: int) -> torch.Tensor:
    """Per-chunk inclusive cumsum of g along time, scaled by RCP_LN2.

    Args:
        g: [BH, N, C, D] log-gate (already reshaped into chunks)
    Returns:
        gc: [BH, N, C, D] fp32, gc[..., t, :] = RCP_LN2 * sum_{u<=t} g[..., u, :]
    """
    BH, N, C2, D = g.shape
    gc = _chunk_local_cumsum_kernel(g.reshape(BH * N, C2, D))
    return gc.reshape(BH, N, C2, D)


# %%
# Helion Kernels
# --------------
# Forward splits along the chunked-parallel pipeline:
#   fwd_h (serial state pass) -> fwd_A (parallel intra scores) -> fwd_o (output)


@helion.kernel()
def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    gc: torch.Tensor,
) -> torch.Tensor:
    """Serial state accumulation across chunks, with per-channel gate.

    h_all[i] holds the state going into chunk i, from a zero initial state:
        h_all[0] = 0
        h_all[i] = diag(exp2(gc_last[i-1])) h_all[i-1] + kg[i-1]^T @ v[i-1]
    where kg[s,k] = k[s,k] * exp2(gc_last[k] - gc[s,k]).

    Args:
        k:  [BH, N, C, D]
        v:  [BH, N, C, DV]
        gc: [BH, N, C, D] base-2 per-chunk cumsum of the gate
    Returns:
        h_all: [BH, N, D, DV]
    """
    BH = k.size(0)
    N = k.size(1)
    C = hl.specialize(k.size(2))
    D = k.size(3)
    DV = v.size(3)

    # The stored state rides at the input dtype; only the accumulator is fp32.
    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)

    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)

        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc.to(h_all.dtype)
            gc_i = gc[idx, i_t, :, tile_d]         # [C, D]
            gc_last = gc[idx, i_t, C - 1, tile_d]  # [D]
            v_i = v[idx, i_t, :, tile_dv]
            kg = (
                k[idx, i_t, :, tile_d].float()
                * torch.exp2(gc_last[None, :] - gc_i)
            ).to(v_i.dtype)
            h_acc = h_acc * torch.exp2(gc_last)[:, None]
            h_acc = hl.dot(kg.transpose(-2, -1), v_i, acc=h_acc)

    return h_all


# Secondary-chunk sub-block size: C is split into BC-row sub-blocks for the
# anchored intra-chunk matmul.
BC = 16


@helion.kernel()
def chunk_fwd_A(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Intra-chunk score matrix A, anchored secondary-chunking (tensor cores).

    A[t,s] = scale * sum_k q[t,k] k[s,k] exp2(gc[t,k] - gc[s,k]) for t >= s, else
    0. Each chunk of C rows is split into BC-row sub-blocks; per row sub-block one
    anchored matmul covers all causal columns. With the gate at the row sub-block
    start as the anchor gc_n,
        A[i_blk, :] = scale * (q exp2(gc - gc_n)) @ (k exp2(gc_n - gc))^T.
    Anchoring keeps both exponents bounded so exp2 cannot overflow.

    Args:
        q, k: [BHN, C, D]
        gc:   [BHN, C, D] base-2 per-chunk cumsum of the gate
        scale: applied to the scores.
    Returns:
        A: [BHN, C, C] fp32, lower-triangular.
    """
    BHN = q.size(0)
    C = hl.specialize(q.size(1))

    A = torch.zeros([BHN, C, C], dtype=torch.float32, device=q.device)

    for tile_bhn in hl.tile(BHN, block_size=1):
        k_all = k[tile_bhn, :, :].float()    # [1, C, D]
        gc_all = gc[tile_bhn, :, :].float()  # [1, C, D]
        col_idx = hl.arange(C)

        for tile_row in hl.tile(C, block_size=BC):
            row_begin = tile_row.begin
            row_end = row_begin + BC  # one-past the diagonal sub-block
            q_blk = q[tile_bhn, tile_row, :].float()    # [1, BC, D]
            gc_blk = gc[tile_bhn, tile_row, :].float()  # [1, BC, D]
            gc_n = gc[tile_bhn, row_begin, :].float()   # [1, D] anchor (block start)

            # Anchored matmul over all causal columns. The k-side exponent for
            # columns past the diagonal sub-block (s >= row_end) would overflow,
            # so it is masked to 0; those scores are zeroed by the lower-tri mask.
            in_range = (col_idx < row_end)[None, :]             # [1, C]
            qg = q_blk * torch.exp2(gc_blk - gc_n[:, None, :])  # [1, BC, D]
            expo_k = torch.where(
                in_range[:, :, None],
                gc_n[:, None, :] - gc_all,
                torch.zeros_like(gc_all),
            )  # [1, C, D]
            kg = k_all * torch.exp2(expo_k)               # [1, C, D]
            a = hl.dot(qg, kg.transpose(-2, -1)) * scale  # [1, BC, C]

            # Lower-triangular causal mask over the full [BC, C] row band:
            # keep entry (t, s) iff s <= global row index t.
            lt = tile_row.index[:, None] >= col_idx[None, :]  # [BC, C]
            A[tile_bhn, tile_row, :] = a * lt[None, :, :].to(a.dtype)

    return A


@helion.kernel()
def chunk_fwd_o(
    q: torch.Tensor,
    v: torch.Tensor,
    gc: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Per-chunk parallel output: o = scale * (q exp2(gc)) @ h + A.tril @ v.

    The cross term gates q by exp2(gc) per key channel before the state matmul;
    the output is linear in q, so scale factors out and is applied once to the
    fp32 cross accumulator; q is passed unscaled. The intra term is the lower-
    triangular score matrix A (already masked by chunk_fwd_A) times v.

    Args:
        q:  [BHN, C, D]
        v:  [BHN, C, DV]
        gc: [BHN, C, D] base-2 per-chunk cumsum of the gate
        h:  [BHN, D, DV] state entering each chunk
        A:  [BHN, C, C] fp32 lower-triangular intra-chunk scores
        scale: applied to the cross term.
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

        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d].float()
            gct = gc[tile_bhn, :, tile_d]
            qg = (qt * torch.exp2(gct)).to(q.dtype)
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qg, ht.to(qg.dtype), acc=o_cross)
        o_cross = o_cross * scale

        # A is already lower-triangular out of chunk_fwd_A, so no re-mask here.
        vt = v[tile_bhn, :, tile_dv]
        At = A[tile_bhn, :, :]
        o_intra = hl.dot(At.to(vt.dtype), vt)
        out[tile_bhn, :, tile_dv] = (o_cross + o_intra).to(out.dtype)

    return out


# %%
# Backward Kernels
# ----------------


@helion.kernel()
def chunk_bwd_dh(
    q: torch.Tensor,
    gc: torch.Tensor,
    do: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Serial reverse state-gradient pass, with per-channel gate.

    dh_all[i] holds the state gradient coming out of chunk i, from a zero
    final-state gradient:
        q_decay[s,k] = exp2(gc[s,k])
        dh_all[N-1]  = 0
        dh_all[i]    = diag(exp2(gc_last[i+1])) dh_all[i+1]
                       + ((scale * q_decay[i+1]) * q[i+1])^T @ do[i+1]

    q is passed unscaled; scale folds into the q-side decay here.

    Args:
        q:  [BH, N, C, D]
        gc: [BH, N, C, D] base-2 per-chunk cumsum of the gate
        do: [BH, N, C, DV]
        scale: folded into the q-side decay.
    Returns:
        dh_all: [BH, N, D, DV]
    """
    BH = q.size(0)
    N = q.size(1)
    C = hl.specialize(q.size(2))
    D = q.size(3)
    DV = do.size(3)
    mm_dt = do.dtype  # bf16 matmul operand dtype; fp32 state accumulation

    # The stored gradient rides at the input dtype; only the accumulator is fp32.
    dh_all = torch.empty([BH, N, D, DV], dtype=q.dtype, device=q.device)

    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        dh_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)

        for i_t in hl.grid(N):
            i = N - 1 - i_t
            dh_all[idx, i, tile_d, tile_dv] = dh_acc.to(dh_all.dtype)

            gc_i = gc[idx, i, :, tile_d].float()
            gc_last = gc[idx, i, C - 1, tile_d].float()
            # q weighted by scale exp2(gc); state decays by exp2(gc_last).
            q_i = (q[idx, i, :, tile_d].float() * (torch.exp2(gc_i) * scale)).to(mm_dt)
            do_i = do[idx, i, :, tile_dv]
            dh_acc = dh_acc * torch.exp2(gc_last)[:, None]
            dh_acc = hl.dot(q_i.transpose(-2, -1), do_i, acc=dh_acc)

    return dh_all


@helion.kernel()
def chunk_bwd_dqk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gc: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    do: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-chunk parallel dQ, dK, plus the state-carry gate gradient dgk.

        exp_gc[t,k]      = exp2(gc[t,k])
        exp_neg_gc[j,k]  = exp2(-gc[j,k])
        exp_gc_last[j,k] = exp2(gc_last[k] - gc[j,k])
        dA      = (do @ v^T) * causal                     # [C, DV] @ [DV, C] -> [C, C]
        dq      = scale * exp_gc * (dA @ (exp_neg_gc * k) + do @ h^T)
        dk_intra     = scale * exp_neg_gc * (dA^T @ (exp_gc * q))
        dk_state_term = exp_gc_last * (v @ dh^T)          # [C, DV] @ [DV, D] -> [C, D]
        dk      = dk_intra + dk_state_term

    q is passed unscaled; scale folds into dq and dk_intra via the gate
    scalings. The dk state term reads dh, which already carries scale.

    The state-carry gate gradient dgk is produced here too, fused into the DV
    loop (h = state into chunk i):
        hdh[k]  = (h * dh).sum_v                          # [D, DV] -> [D]
        dgk[k]  = exp2(gc_last[k]) * hdh[k] + (k * dk_state_term).sum_j
    Only the state part of dk enters dgk; the intra part of (k * dk) is carried
    by the q*dq - k*dk reverse-cumsum in chunk_bwd_dg.

    Outputs are fp32: dg is built from the near-cancelling difference q*dq - k*dk
    reverse-cumsummed over the chunk, which is precision-sensitive.

    Args:
        q, k: [BHN, C, D]
        v:    [BHN, C, DV]
        gc:   [BHN, C, D] base-2 per-chunk cumsum of the gate
        h, dh: [BHN, D, DV] forward state and its gradient
        do:   [BHN, C, DV] output gradient
        scale: folded into dq/dk via the gate scalings.
    Returns:
        dq, dk: [BHN, C, D] fp32; dgk: [BHN, D] fp32 state-carry gate gradient
    """
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)
    dt = q.dtype  # native (bf16) matmul operand dtype; fp32 accumulators throughout

    dq_out = torch.empty([BHN, C, D], dtype=torch.float32, device=q.device)
    dk_out = torch.empty([BHN, C, D], dtype=torch.float32, device=k.device)
    dgk_out = torch.empty([BHN, D], dtype=torch.float32, device=q.device)

    # Tile D so each D-tile's fp32 working set fits smem at K=V=256. The intra
    # matmuls stay fp32 to feed the near-cancelling q*dq - k*dk difference dg.
    for tile_bhn, tile_d in hl.tile([BHN, D], block_size=[1, None]):
        dA_raw = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        dq_cross = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
        dk_state = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
        hdh = hl.zeros([tile_bhn, tile_d], dtype=torch.float32)
        for tile_dv in hl.tile(DV):
            dot = do[tile_bhn, :, tile_dv]
            vt = v[tile_bhn, :, tile_dv]
            ht = h[tile_bhn, tile_d, tile_dv]
            dht = dh[tile_bhn, tile_d, tile_dv]

            dA_raw = hl.dot(dot, vt.transpose(-2, -1), acc=dA_raw)
            dq_cross = hl.dot(dot, ht.transpose(-2, -1).to(dt), acc=dq_cross)
            dk_state = hl.dot(vt, dht.transpose(-2, -1).to(dt), acc=dk_state)
            hdh = hdh + (ht.float() * dht.float()).sum(dim=-1)

        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        dA = dA_raw * causal  # [tile_bhn, C, C] fp32, t >= j

        qt = q[tile_bhn, :, tile_d].float()
        kt = k[tile_bhn, :, tile_d].float()
        gct = gc[tile_bhn, :, tile_d].float()
        gc_last = gc[tile_bhn, C - 1, tile_d].float()

        # One exp2 over the gate; the other two scalings come from reciprocal and
        # multiply (cheaper than three exp2, the dominant elementwise cost at D256).
        exp_gc = torch.exp2(gct)
        exp_neg_gc = 1.0 / exp_gc
        exp_gc_last = torch.exp2(gc_last)[:, None, :] * exp_neg_gc

        # Fuse the cross add into the dot accumulator, then scale the whole sum.
        kg = kt * exp_neg_gc
        dq_acc = hl.dot(dA, kg, acc=dq_cross) * exp_gc * scale

        # The intra dk matmul is scaled before the add, which a dot accumulator
        # can't express (no alpha), so it stays a separate dot + add.
        qg = qt * exp_gc
        dk_intra = hl.dot(dA.transpose(-2, -1), qg) * exp_neg_gc * scale
        dk_state_term = dk_state * exp_gc_last
        dk_acc = dk_intra + dk_state_term

        dgk = torch.exp2(gc_last) * hdh + (kt * dk_state_term).sum(dim=1)

        dq_out[tile_bhn, :, tile_d] = dq_acc
        dk_out[tile_bhn, :, tile_d] = dk_acc
        dgk_out[tile_bhn, tile_d] = dgk

    return dq_out, dk_out, dgk_out


@helion.kernel()
def chunk_bwd_dv(
    k: torch.Tensor,
    gc: torch.Tensor,
    A: torch.Tensor,
    dh: torch.Tensor,
    do: torch.Tensor,
) -> torch.Tensor:
    """Per-chunk parallel dV with per-channel gate.

        k_decay[j,k] = exp2(gc_last[k] - gc[j,k])
        dv_state     = (k_decay * k) @ dh                 # [C, D] @ [D, DV] -> [C, DV]
        dv           = (A * causal)^T @ do + dv_state     # [C, C] @ [C, DV] -> [C, DV]

    A is the lower-triangular forward score matrix. Neither term takes scale: A
    carries it from the forward and dh carries it from the state pass.

    Args:
        k:  [BHN, C, D]
        gc: [BHN, C, D] base-2 per-chunk cumsum of the gate
        A:  [BHN, C, C] fp32 forward score matrix
        dh: [BHN, D, DV] state gradient
        do: [BHN, C, DV] output gradient
    Returns:
        dv: [BHN, C, DV]
    """
    BHN = k.size(0)
    C = hl.specialize(k.size(1))
    D = k.size(2)
    DV = do.size(2)
    dt = k.dtype  # native (bf16) operand dtype for the state matmul

    dv_out = torch.empty([BHN, C, DV], dtype=do.dtype, device=k.device)

    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        dv_state = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)

        for tile_d in hl.tile(D):
            kt = k[tile_bhn, :, tile_d].float()  # [tile_bhn, C, D]
            gct = gc[tile_bhn, :, tile_d].float()
            gc_last = gc[tile_bhn, C - 1, tile_d].float()  # [tile_bhn, D]
            dht = dh[tile_bhn, tile_d, tile_dv].to(dt)  # dh is fp32; bf16 operand

            kg = (kt * torch.exp2(gc_last[:, None, :] - gct)).to(dt)
            dv_state = hl.dot(kg, dht, acc=dv_state)

        # Intra term kept fp32: A is the fp32 forward score matrix.
        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        At = A[tile_bhn, :, :] * causal  # [tile_bhn, C, C]
        dot = do[tile_bhn, :, tile_dv].float()
        dv_acc = hl.dot(At.transpose(-2, -1), dot, acc=dv_state)
        dv_out[tile_bhn, :, tile_dv] = dv_acc.to(dv_out.dtype)

    return dv_out


@helion.kernel()
def chunk_bwd_dg(
    q: torch.Tensor,
    k: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dgk: torch.Tensor,
    g_dtype: torch.dtype,
) -> torch.Tensor:
    """Per-channel gate gradient dg, fused.

    dg_raw[t,k] = q[t,k] dq[t,k] - k[t,k] dk[t,k]   (fp32)
    dg[t,k]     = sum_{u>=t} dg_raw[u,k] + dgk[k]    (reverse cumsum over the
                                                      chunk + state-carry term)

    The reverse cumsum is an upper-triangular ones matmul R @ dg_raw (R[t,u] = 1
    iff u >= t) in fp32, which keeps the near-cancelling q*dq - k*dk difference
    exact.

    Args:
        q, k:  [BHN, C, D] native (bf16) forward activations
        dq, dk: [BHN, C, D] fp32 gradients from chunk_bwd_dqk
        dgk:   [BHN, D] fp32 per-chunk state-carry gate gradient
        g_dtype: output dtype for dg (matches the input gate dtype)
    Returns:
        dg: [BHN, C, D] in g_dtype
    """
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)

    dg_out = torch.empty([BHN, C, D], dtype=g_dtype, device=q.device)

    for tile_bhn, tile_d in hl.tile([BHN, D]):
        qt = q[tile_bhn, :, tile_d].float()  # [tile_bhn, C, tile_d]
        kt = k[tile_bhn, :, tile_d].float()
        dqt = dq[tile_bhn, :, tile_d]  # fp32
        dkt = dk[tile_bhn, :, tile_d]
        dg_raw = qt * dqt - kt * dkt  # [tile_bhn, C, tile_d] fp32

        idx = hl.arange(C)
        # R[t,u] = 1 iff u >= t (upper-triangular incl. diagonal): reverse cumsum.
        rev = (idx[:, None] <= idx[None, :]).to(torch.float32)  # [C, C]
        dgk_t = dgk[tile_bhn, tile_d]  # [tile_bhn, tile_d]
        dg = hl.dot(
            rev[None, :, :].broadcast_to(tile_bhn, C, C),
            dg_raw,
            acc=dgk_t[:, None, :].broadcast_to(tile_bhn, C, dgk_t.size(-1)),
        )
        dg_out[tile_bhn, :, tile_d] = dg.to(g_dtype)

    return dg_out


# %%
# Autograd Wiring
# ---------------


class _GLAFn(torch.autograd.Function):
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
        B, H, T, D = q.shape
        DV = v.shape[-1]
        N = T // C
        BH = B * H
        BHN = BH * N

        kf = k.reshape(BH, N, C, D)
        vf = v.reshape(BH, N, C, DV)
        gf = g.reshape(BH, N, C, D)
        gc = _chunk_local_cumsum_base2(gf, C)  # [BH, N, C, D] fp32

        qf = q.reshape(BHN, C, D)
        kf2 = kf.reshape(BHN, C, D)
        vf2 = vf.reshape(BHN, C, DV)
        gcf = gc.reshape(BHN, C, D)

        # q is passed unscaled; scale is folded into the intra scores A and the
        # cross term of the output.
        A = chunk_fwd_A(qf, kf2, gcf, scale)

        h_all = chunk_fwd_h(kf, vf, gc)
        hf = h_all.reshape(BHN, D, DV)
        o = chunk_fwd_o(qf, vf2, gcf, hf, A, scale)

        # Save A (the lower-triangular fp32 score matrix) so dv reuses it instead
        # of recomputing chunk_fwd_A in the backward.
        ctx.save_for_backward(q, k, v, gc, h_all, A)
        ctx.scale = scale
        ctx.C = C
        ctx.g_dtype = g.dtype
        return o.reshape(B, H, T, DV)

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v, gc, h_all, A = ctx.saved_tensors
        scale = ctx.scale
        C = ctx.C
        B, H, T, D = q.shape
        DV = v.shape[-1]
        N = T // C
        BH = B * H
        BHN = BH * N

        # dq/dk come out fp32 because dg is built from the near-cancelling
        # difference q*dq - k*dk and needs the precision.
        qf4 = q.reshape(BH, N, C, D)
        do4 = grad_output.reshape(BH, N, C, DV)
        gc4 = gc.reshape(BH, N, C, D)

        qf2 = q.reshape(BHN, C, D)
        kf2 = k.reshape(BHN, C, D)
        vf2 = v.reshape(BHN, C, DV)
        hf = h_all.reshape(BHN, D, DV)
        dof = grad_output.reshape(BHN, C, DV)
        gcf = gc.reshape(BHN, C, D)

        # Zero terminal-state gradient is seeded in-register inside the kernel.
        dh_all = chunk_bwd_dh(qf4, gc4, do4, scale)
        dhf = dh_all.reshape(BHN, D, DV)

        # dgk [BHN, D] is the state-carry gate gradient, fused into this kernel.
        dq, dk, dgk = chunk_bwd_dqk(qf2, kf2, vf2, gcf, hf, dhf, dof, scale)
        # A is the saved forward score matrix, reused for dv's intra term.
        dv = chunk_bwd_dv(kf2, gcf, A, dhf, dof)

        # dg per channel: revcumsum_C(q*dq - k*dk) + dgk, emitted in g_dtype. The
        # RCP_LN2 in gc and the ln2 from d/dx exp2 cancel, so no extra factor here.
        dg = chunk_bwd_dg(qf2, kf2, dq, dk, dgk, ctx.g_dtype)

        dq = dq.reshape(B, H, T, D)
        dk = dk.reshape(B, H, T, D)
        dv = dv.reshape(B, H, T, DV)
        dg = dg.reshape(B, H, T, D)

        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dg,
            None,
            None,
        )


# %%
# Public API
# ----------


def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    C: int = 64,
) -> torch.Tensor:
    """Gated linear attention, chunked-parallel form.

    Args:
        q, k: [B, H, T, D]
        v:    [B, H, T, DV]
        g:    [B, H, T, D] per-channel log-decay (exp(g) in (0,1)^D)
        scale: applied to q. Defaults to 1/sqrt(D).
        C:     chunk size. T must be divisible by C (no padding here).
    Returns:
        out: [B, H, T, DV]
    """
    D = q.size(-1)
    if scale is None:
        scale = D**-0.5
    assert q.size(-2) % C == 0, f"T={q.size(-2)} must be divisible by C={C}"
    return _GLAFn.apply(q, k, v, g, C, scale)


# %%
# References for Validation
# -------------------------


def naive_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Step-by-step recurrent reference. Slow but obviously correct.

    Args: q, k, g: [B, H, T, D]; v: [B, H, T, DV]
    """
    B, H, T, D = q.shape
    DV = v.shape[-1]
    if scale is None:
        scale = D**-0.5
    S = q.new_zeros(B, H, D, DV, dtype=torch.float32)
    out = torch.empty(B, H, T, DV, dtype=torch.float32, device=q.device)
    qf = q.float() * scale
    kf = k.float()
    vf = v.float()
    gf = g.float()
    for t in range(T):
        decay = gf[:, :, t].exp()  # [B, H, D]
        S = S * decay[..., None] + torch.einsum(
            "bhd,bhv->bhdv", kf[:, :, t], vf[:, :, t]
        )
        out[:, :, t] = torch.einsum("bhd,bhdv->bhv", qf[:, :, t], S)
    return out.to(v.dtype)


def fla_chunk_gla_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """FLA's chunk_gla on its NATIVE token-first [B,T,H,D] layout.

    No transpose: callers that already hold token-first tensors use this directly
    so the layout conversion is not charged to FLA's measured time. Unwraps FLA's
    (output, final_state) tuple.
    """
    from fla.ops.gla import chunk_gla as _fla  # pyrefly: ignore

    o = _fla(q, k, v, g, scale=scale)
    return o[0] if isinstance(o, tuple) else o


def fla_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    C: int = 64,
) -> torch.Tensor:
    """FLA's chunk_gla wrapped to accept head-first [B,H,T,D].

    Transposes to FLA's native layout and back; used by ``test()`` for
    correctness. For timing, use ``fla_chunk_gla_native`` and transpose outside
    the timed region.
    """
    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()
    gt = g.transpose(1, 2).contiguous()
    o = fla_chunk_gla_native(qt, kt, vt, gt, scale=scale)
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
        from fla.ops.gla import chunk_gla as _fla_fn  # noqa: F401
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
    # g = logsigmoid_clamp(randn): per-channel log-decay, exp(g) in (0,1)^D.
    g = (
        torch.nn.functional.logsigmoid(torch.randn(b, h, t, d, device=device))
        .clamp_min(-5.0)
        .to(dtype)
        .requires_grad_(True)
    )

    def helion_fn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        return chunk_gla(q, k, v, g, scale=scale)

    def fla_fn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        return fla_chunk_gla(q, k, v, g, scale=scale)

    def naive_fn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        return naive_recurrent_gla(q, k, v, g, scale=scale)

    # 1. fwd+bwd vs the naive recurrence.
    run_example(
        helion_fn,
        {"naive_recurrent": naive_fn},
        (q, k, v, g),
        kernel_name="helion",
        baseline_name="naive_recurrent",
        bwd=True,
    )

    # 2. fwd+bwd vs FLA's chunk_gla.
    run_example(
        helion_fn,
        {"fla": fla_fn},
        (q, k, v, g),
        kernel_name="helion",
        baseline_name="fla",
        bwd=True,
    )


# %%
# Main Function
# -------------


def main() -> None:
    """Run a modest representative shape (B=1, H=2, T=512, D=DV=64, bf16)."""
    test(1, 2, 512, 64, 64, HALF_DTYPE, device=DEVICE)


if __name__ == "__main__":
    main()
