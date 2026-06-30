"""
Linear Attention - chunked-parallel
===================================

Linear attention with no decay and no correction term, chunked-parallel form.

Recurrence (per timestep):
    S_t = S_{t-1} + k_t^T @ v_t
    o_t = q_t @ S_t

Chunked form parallelizes intra-chunk attention while sequentializing the
state pass across chunks. Validated against a naive recurrent reference and
FLA's `chunk_linear_attn` (which dispatches through `simple_gla` with g=0).
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

# %%
# Helion Kernels
# --------------
# Kernels split along the chunked-parallel pipeline:
#   forward:   fwd_h  (serial state pass)  -> fwd_o  (parallel output)
#   backward:  bwd_dh (reverse state pass) -> bwd_dqk + bwd_dv (parallel)
# plus fwd_fused, a single-launch forward that fuses fwd_h + fwd_o (see below).


@helion.kernel()
def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Serial state accumulation across chunks (zero initial state).

    h_all[i] holds the state going *into* chunk i:
        h_all[0] = 0
        h_all[i] = h_all[i-1] + k[i-1]^T @ v[i-1]

    The initial state is always zero in the forward pass, so we seed the
    accumulator with hl.zeros in-register instead of reading an h0 buffer from
    HBM. This matches FLA's USE_INITIAL_STATE=False path and removes a host-side
    new_zeros fill kernel that would otherwise launch every forward.

    Args:
        k:  [BH, N, C, D]
        v:  [BH, N, C, DV]
    Returns:
        h_all: [BH, N, D, DV]
    """
    BH = k.size(0)
    N = k.size(1)
    D = k.size(3)
    DV = v.size(3)

    # State buffer rides at the INPUT dtype (bf16), matching FLA's
    # states_in_fp32=False: chunk_fwd_o casts h down to q.dtype anyway, so an
    # fp32 store wastes HBM bandwidth on both the write here and the per-chunk
    # read in the output pass. The accumulator h_acc stays fp32.
    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)

    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        idx = tile_bh.id

        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc.to(h_all.dtype)
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = torch.addmm(h_acc, k_i.transpose(-2, -1), v_i)

    return h_all


@helion.kernel()
def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Per-chunk parallel output: o = scale * (q @ h + (q @ k^T).tril @ v).

    The output is linear in q, so the 1/sqrt(D) scale factors out of the whole
    expression and is applied once here on the fp32 accumulator. This lets the
    caller pass q unscaled and avoids a full-tensor `q * scale` elementwise
    kernel on every forward (FLA likewise folds scale into the output dot).

    Args:
        q, k: [BHN, C, D]
        v:    [BHN, C, DV]
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

        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = torch.baddbmm(o_cross, qt, ht.to(qt.dtype))
            attn = torch.baddbmm(attn, qt, kt.transpose(-2, -1))

        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        attn = attn * causal

        vt = v[tile_bhn, :, tile_dv]
        o_intra = torch.bmm(attn.to(vt.dtype), vt)
        out[tile_bhn, :, tile_dv] = ((o_cross + o_intra) * scale).to(out.dtype)

    return out


@helion.kernel()
def chunk_fwd_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass fused forward: serial chunk scan with inline output.

    One program per (bh, dv-block). The state going *into* each chunk is held in
    registers (h_acc, fp32) across the serial chunk loop; the chunk output is
    emitted inline before the state is advanced. This fuses chunk_fwd_h +
    chunk_fwd_o into ONE launch, killing the h_all HBM round-trip (write in
    fwd_h, read in fwd_o) and a kernel launch.

    Per chunk i (zero initial state):
        o[i]   = scale * (q[i] @ h_acc + tril(q[i] @ k[i]^T) @ v[i])
        h_acc += k[i]^T @ v[i]    (cross term reads h_acc BEFORE this update)

    The per-chunk state h_all is also written to HBM as it scans, so the
    backward gets it for free -- the store is the only added cost over a
    state-free scan, and the backward needs it anyway (FLA recomputes it).

    Args:
        q, k: [BH, N, C, D]
        v:    [BH, N, C, DV]
        scale: applied to the output.
    Returns:
        out:   [BH, N, C, DV]
        h_all: [BH, N, D, DV] state entering each chunk (bf16, FLA parity)
    """
    BH = q.size(0)
    N = q.size(1)
    C = hl.specialize(q.size(2))
    D = hl.specialize(q.size(3))
    DV = v.size(3)

    out = torch.empty([BH, N, C, DV], dtype=q.dtype, device=q.device)
    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)

    for tile_bh, tile_dv in hl.tile([BH, DV], block_size=[1, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([D, tile_dv], dtype=torch.float32)

        for i_t in hl.grid(N):
            q_i = q[idx, i_t, :, :]  # [C, D]
            k_i = k[idx, i_t, :, :]  # [C, D]
            v_i = v[idx, i_t, :, tile_dv]  # [C, bv]

            # state entering chunk i (chunks < i) -> save for the backward.
            h_all[idx, i_t, :, tile_dv] = h_acc.to(h_all.dtype)

            o_cross = torch.mm(q_i, h_acc.to(q_i.dtype)).float()  # [C, bv]

            attn = torch.mm(q_i, k_i.transpose(-2, -1)).float()  # [C, C]
            cidx = hl.arange(C)
            causal = (cidx[:, None] >= cidx[None, :]).float()
            attn = attn * causal
            o_intra = torch.mm(attn.to(v_i.dtype), v_i).float()  # [C, bv]

            out[idx, i_t, :, tile_dv] = ((o_cross + o_intra) * scale).to(out.dtype)

            h_acc = torch.addmm(h_acc, k_i.transpose(-2, -1), v_i)

    return out, h_all


@helion.kernel()
def chunk_bwd_dh(
    q: torch.Tensor,
    do: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Serial reverse state-gradient pass (zero final-state gradient).

    dh_all[i] holds the gradient *coming out of* chunk i:
        dh_all[N-1] = 0
        dh_all[i]   = dh_all[i+1] + scale * q[i+1]^T @ do[i+1]

    q is the *unscaled* query; the forward folds scale into the output, so the
    state gradient carries that scale (q' = scale*q). The zero final-state
    gradient is seeded in-register, dropping a host new_zeros fill.

    Args:
        q:  [BH, N, C, D]
        do: [BH, N, C, DV] output gradient
        scale: folded into the q-row scaling.
    Returns:
        dh_all: [BH, N, D, DV]
    """
    BH = q.size(0)
    N = q.size(1)
    D = q.size(3)
    DV = do.size(3)

    dh_all = torch.empty([BH, N, D, DV], dtype=q.dtype, device=q.device)

    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        dh_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        idx = tile_bh.id

        for i_t in hl.grid(N):
            i = N - 1 - i_t
            dh_all[idx, i, tile_d, tile_dv] = dh_acc.to(dh_all.dtype)
            q_i = q[idx, i, :, tile_d] * scale
            do_i = do[idx, i, :, tile_dv]
            dh_acc = torch.addmm(dh_acc, q_i.transpose(-2, -1), do_i)

    return dh_all


@helion.kernel()
def chunk_bwd_dh_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused reverse state-gradient scan that ALSO emits dV inline.

    One program per (bh, dv-block) carries the full-D state gradient dh_acc
    [D, bv] in registers across the reverse chunk scan (mirrors the forward
    fusion). dv[i] consumes the state coming *out of* chunk i, which is exactly
    dh_acc before it absorbs chunk i, so dv is emitted in the same pass that
    writes dh_all -- killing the separate chunk_bwd_dv launch and its re-read of
    dh_all + recompute of k @ dh.

    Per chunk i (reverse order, zero final-state gradient):
        dh_all[i] = dh_acc                              (state out of chunk i)
        dv[i]     = (k[i] @ dh_acc) + tril(scale*q[i] @ k[i]^T)^T @ do[i]
        dh_acc   += scale * q[i]^T @ do[i]

    Args:
        q, k: [BH, N, C, D]
        do:   [BH, N, C, DV] output gradient
        scale: folded into the q-row scaling and the intra attention.
    Returns:
        dh_all: [BH, N, D, DV] state gradient entering each chunk's consumers
        dv:     [BH, N, C, DV]
    """
    BH = q.size(0)
    N = q.size(1)
    C = hl.specialize(q.size(2))
    D = hl.specialize(q.size(3))
    DV = do.size(3)

    dh_all = torch.empty([BH, N, D, DV], dtype=q.dtype, device=q.device)
    dv_out = torch.empty([BH, N, C, DV], dtype=do.dtype, device=do.device)

    for tile_bh, tile_dv in hl.tile([BH, DV], block_size=[1, None]):
        dh_acc = hl.zeros([D, tile_dv], dtype=torch.float32)
        idx = tile_bh.id

        for i_t in hl.grid(N):
            i = N - 1 - i_t
            q_i = q[idx, i, :, :]  # [C, D]
            k_i = k[idx, i, :, :]  # [C, D]
            do_i = do[idx, i, :, tile_dv]  # [C, bv]

            # state coming out of chunk i (chunks > i) -> dh_all and dv_state.
            dh_all[idx, i, :, tile_dv] = dh_acc.to(dh_all.dtype)
            dv_state = torch.mm(k_i, dh_acc.to(k_i.dtype)).float()  # [C, bv]

            # intra: dv_intra = tril(scale*q@k^T)^T @ do.
            attn = torch.mm(q_i, k_i.transpose(-2, -1)).float()  # [C, C]
            cidx = hl.arange(C)
            causal = (cidx[:, None] >= cidx[None, :]).float()
            attn = attn * (causal * scale)
            dv_acc = torch.addmm(dv_state, attn.transpose(-2, -1).to(do_i.dtype), do_i)
            dv_out[idx, i, :, tile_dv] = dv_acc.to(dv_out.dtype)

            # advance state gradient for the next (earlier) chunk.
            q_s = q_i * scale
            dh_acc = torch.addmm(dh_acc, q_s.transpose(-2, -1), do_i)

    return dh_all, dv_out


@helion.kernel()
def chunk_bwd_dqk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-chunk parallel dQ, dK (q passed unscaled; scale folded here).

    With q' = scale*q the forward query, dq is the gradient w.r.t. the unscaled
    q, i.e. scale * dq'. dq' has no q dependence, so we scale it once at the
    end. dk uses q' = scale*q directly; the state term carries scale through dh.

    Args:
        q, k: [BHN, C, D]
        v:    [BHN, C, DV]
        h, dh: [BHN, D, DV] forward state and its gradient
        do:   [BHN, C, DV] output gradient
        scale: folded into dA and the dq cross term.
    Returns:
        dq, dk: [BHN, C, D]
    """
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)

    dq_out = torch.empty([BHN, C, D], dtype=q.dtype, device=q.device)
    dk_out = torch.empty([BHN, C, D], dtype=k.dtype, device=k.device)

    for tile_bhn, tile_d in hl.tile([BHN, D]):
        dA_raw = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        dq_cross_acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
        dk_state_acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)

        for tile_dv in hl.tile(DV):
            dot = do[tile_bhn, :, tile_dv]
            vt = v[tile_bhn, :, tile_dv]
            ht = h[tile_bhn, tile_d, tile_dv]
            dht = dh[tile_bhn, tile_d, tile_dv]

            dA_raw = torch.baddbmm(dA_raw, dot, vt.transpose(-2, -1))
            dq_cross_acc = torch.baddbmm(
                dq_cross_acc, dot, ht.transpose(-2, -1).to(dot.dtype)
            )
            dk_state_acc = torch.baddbmm(
                dk_state_acc, vt, dht.transpose(-2, -1).to(vt.dtype)
            )

        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        dA = dA_raw * causal

        qt = q[tile_bhn, :, tile_d]
        kt = k[tile_bhn, :, tile_d]
        # dq' (q-independent) -> dq = scale * dq'. Fuse the cross-state add into
        # the matmul (baddbmm), then apply scale to the whole sum.
        dq_acc = torch.baddbmm(dq_cross_acc, dA.to(kt.dtype), kt) * scale
        # dk uses q' = scale*q; the state term carries scale via dh. The matmul
        # is scaled before the add, which baddbmm can't express (no alpha), so
        # this stays an explicit bmm + add.
        dk_acc = torch.bmm(dA.transpose(-2, -1).to(qt.dtype), qt) * scale + dk_state_acc

        dq_out[tile_bhn, :, tile_d] = dq_acc.to(q.dtype)
        dk_out[tile_bhn, :, tile_d] = dk_acc.to(k.dtype)

    return dq_out, dk_out


@helion.kernel()
def chunk_bwd_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Per-chunk parallel dV (q passed unscaled; scale folded here).

    The intra attention uses q' = scale*q, so the masked attention carries
    scale; the state term k @ dh already carries scale through dh.

    Args:
        q, k: [BHN, C, D]
        do:   [BHN, C, DV] output gradient
        dh:   [BHN, D, DV] state gradient
        scale: folded into the intra attention.
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

            attn = torch.baddbmm(attn, qt, kt.transpose(-2, -1))
            dv_state = torch.baddbmm(dv_state, kt, dht.to(kt.dtype))

        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        attn = attn * (causal * scale)

        dot = do[tile_bhn, :, tile_dv]
        dv_acc = torch.baddbmm(dv_state, attn.transpose(-2, -1).to(dot.dtype), dot)
        dv_out[tile_bhn, :, tile_dv] = dv_acc.to(dv_out.dtype)

    return dv_out


# %%
# Autograd Wiring
# ---------------


class _LinearAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
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

        # Two forward structures: the single-pass fused kernel (one launch, no
        # h_all HBM round-trip; parallel over (BH, DV-block), serial N-chunk
        # scan) and the chunk-parallel split (chunk_fwd_h + chunk_fwd_o, parallel
        # over BH*N). Fused is the default; LINATT_NO_FUSED_FWD=1 selects split.
        use_fused = os.environ.get("LINATT_NO_FUSED_FWD") != "1"

        if use_fused:
            qf4 = q.reshape(BH, N, C, D)
            # Fused scan emits o and the per-chunk state h_all in one launch
            # (state already in registers), so the backward gets h_all without a
            # separate state pass. FLA pays that recompute in its backward; we
            # hand the state over from the forward.
            o, h_all = chunk_fwd_fused(qf4, kf, vf, scale)
            ctx.save_for_backward(q, k, v, h_all)
            ctx.C = C
            ctx.scale = scale
            return o.reshape(B, H, T, DV)

        # h_all stays fp32 (accumulator); inputs ride at native dtype. Zero
        # initial state is seeded inside the kernel, no h0 buffer needed.
        h_all = chunk_fwd_h(kf, vf)

        qf = q.reshape(BHN, C, D)
        kf2 = kf.reshape(BHN, C, D)
        vf2 = vf.reshape(BHN, C, DV)
        hf = h_all.reshape(BHN, D, DV)

        # scale is folded into the output (q passed unscaled) so we skip a
        # full-tensor q*scale elementwise launch on the forward.
        o = chunk_fwd_o(qf, kf2, vf2, hf, scale)

        ctx.save_for_backward(q, k, v, h_all)
        ctx.C = C
        ctx.scale = scale
        return o.reshape(B, H, T, DV)

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, h_all = ctx.saved_tensors
        C = ctx.C
        scale = ctx.scale
        B, H, T, D = q.shape
        DV = v.shape[-1]
        N = T // C
        BH = B * H
        BHN = BH * N

        # The forward folds scale into the output, equivalent to using q' =
        # scale * q. The backward kernels are written for that scaled query, so
        # we scale q here (and dq is the gradient w.r.t. the *unscaled* q, which
        # is scale * dq'). Scale is threaded into the dh pass so q never gets
        # materialized scaled in HBM.
        qf4 = q.reshape(BH, N, C, D)
        kf4 = k.reshape(BH, N, C, D)
        do4 = grad_output.reshape(BH, N, C, DV)

        qf2 = q.reshape(BHN, C, D)
        kf2 = k.reshape(BHN, C, D)
        vf2 = v.reshape(BHN, C, DV)
        hf = h_all.reshape(BHN, D, DV)
        dof = do4.reshape(BHN, C, DV)

        # The reverse dh scan can emit dv inline (dv consumes the state coming
        # out of each chunk, held in registers), fusing chunk_bwd_dh +
        # chunk_bwd_dv into one launch and killing the dh_all->dv round-trip;
        # dqk always stays split. Fused is the default; LINATT_NO_FUSED_BWD=1
        # selects the split dh + dv path.
        use_fused_bwd = os.environ.get("LINATT_NO_FUSED_BWD") != "1"

        if use_fused_bwd:
            dh_all, dv = chunk_bwd_dh_dv(qf4, kf4, do4, scale)
            dhf = dh_all.reshape(BHN, D, DV)
            dq, dk = chunk_bwd_dqk(qf2, kf2, vf2, hf, dof, dhf, scale)
        else:
            dh_all = chunk_bwd_dh(qf4, do4, scale)
            dhf = dh_all.reshape(BHN, D, DV)
            dq, dk = chunk_bwd_dqk(qf2, kf2, vf2, hf, dof, dhf, scale)
            dv = chunk_bwd_dv(qf2, kf2, dof, dhf, scale)

        return (
            dq.reshape(B, H, T, D),
            dk.reshape(B, H, T, D),
            dv.reshape(B, H, T, DV),
            None,
            None,
        )


# %%
# Public API
# ----------


def chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    C: int = 64,
) -> torch.Tensor:
    """Vanilla chunked linear attention.

    Args:
        q, k: [B, H, T, D]
        v:    [B, H, T, DV]
        scale: applied to q. Defaults to 1/sqrt(D).
        C:     chunk size. T must be divisible by C (no padding here).
    Returns:
        out: [B, H, T, DV]
    """
    D = q.size(-1)
    if scale is None:
        scale = D**-0.5
    assert q.size(-2) % C == 0, f"T={q.size(-2)} must be divisible by C={C}"
    return _LinearAttnFn.apply(q, k, v, C, scale)


# %%
# References for Validation
# -------------------------


def naive_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Step-by-step recurrent reference. Slow but obviously correct.

    Args: q, k: [B, H, T, D]; v: [B, H, T, DV]
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
    for t in range(T):
        S = S + torch.einsum("bhd,bhv->bhdv", kf[:, :, t], vf[:, :, t])
        out[:, :, t] = torch.einsum("bhd,bhdv->bhv", qf[:, :, t], S)
    return out.to(v.dtype)


def fla_chunk_linear_attn_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """FLA's chunk_linear_attn on its NATIVE token-first [B,T,H,D] layout.

    No transpose: callers that already hold token-first tensors use this directly
    so the layout conversion is not charged to FLA's measured time. Unwraps FLA's
    (output, final_state) tuple.
    """
    from fla.ops.linear_attn import chunk_linear_attn as _fla  # pyrefly: ignore

    o = _fla(q, k, v, scale=scale, normalize=False)
    return o[0] if isinstance(o, tuple) else o


def fla_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    C: int = 64,
) -> torch.Tensor:
    """FLA's chunk_linear_attn wrapped to accept head-first [B,H,T,D].

    Transposes to FLA's native layout and back; used by ``test()`` for
    correctness. For timing, use ``fla_chunk_linear_attn_native`` and transpose
    outside the timed region.
    """
    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()
    o = fla_chunk_linear_attn_native(qt, kt, vt, scale=scale)
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
        from fla.ops.linear_attn import chunk_linear_attn as _fla_fn  # noqa: F401
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

    def helion_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return chunk_linear_attn(q, k, v, scale=scale)

    def fla_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return fla_chunk_linear_attn(q, k, v, scale=scale)

    def naive_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return naive_recurrent_linear_attn(q, k, v, scale=scale)

    run_example(
        helion_fn,
        {"fla": fla_fn, "naive_recurrent": naive_fn},
        (q, k, v),
        kernel_name="helion",
        baseline_name="fla",
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
