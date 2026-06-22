"""
Linear Attention - chunked-parallel
===================================

Linear attention with no decay and no correction term, chunked-parallel form.

Recurrence (per timestep):
    S_t = S_{t-1} + k_t^T @ v_t
    o_t = q_t @ S_t

The chunked form parallelizes intra-chunk attention and sequentializes the
state pass across chunks.
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

# %%
# Helion Kernels
# --------------
# Kernels split along the chunked-parallel pipeline:
#   forward:   fwd_h  (serial state pass)  -> fwd_o  (parallel output)
#   backward:  bwd_dh (reverse state pass) -> bwd_dqk + bwd_dv (parallel)


@helion.kernel()
def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Serial state accumulation across chunks, from a zero initial state.

    h_all[i] holds the state going into chunk i:
        h_all[0] = 0
        h_all[i] = h_all[i-1] + k[i-1]^T @ v[i-1]

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

    # The stored state rides at the input dtype; only the accumulator is fp32.
    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)

    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        idx = tile_bh.id

        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc.to(h_all.dtype)
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)

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

    The output is linear in q, so scale factors out and is applied once on the
    fp32 accumulator, letting the caller pass q unscaled.

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
            o_cross = hl.dot(qt, ht.to(qt.dtype), acc=o_cross)
            attn = hl.dot(qt, kt.transpose(-2, -1), acc=attn)

        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        attn = attn * causal

        vt = v[tile_bhn, :, tile_dv]
        o_intra = hl.dot(attn.to(vt.dtype), vt)
        out[tile_bhn, :, tile_dv] = ((o_cross + o_intra) * scale).to(out.dtype)

    return out


@helion.kernel()
def chunk_bwd_dh(
    q: torch.Tensor,
    do: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Serial reverse state-gradient pass, from a zero final-state gradient.

    dh_all[i] holds the gradient coming out of chunk i:
        dh_all[N-1] = 0
        dh_all[i]   = dh_all[i+1] + scale * q[i+1]^T @ do[i+1]

    q is passed unscaled; scale folds into the q-row term here.

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
            dh_acc = hl.dot(q_i.transpose(-2, -1), do_i, acc=dh_acc)

    return dh_all


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

    dq is the gradient w.r.t. the unscaled q, so it is scaled once at the end;
    dk carries scale through the q-term and the state gradient dh.

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

            dA_raw = hl.dot(dot, vt.transpose(-2, -1), acc=dA_raw)
            dq_cross_acc = hl.dot(
                dot, ht.transpose(-2, -1).to(dot.dtype), acc=dq_cross_acc
            )
            dk_state_acc = hl.dot(
                vt, dht.transpose(-2, -1).to(vt.dtype), acc=dk_state_acc
            )

        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        dA = dA_raw * causal

        qt = q[tile_bhn, :, tile_d]
        kt = k[tile_bhn, :, tile_d]
        # dq' (q-independent) -> dq = scale * dq'. Fuse the cross-state add into
        # the dot accumulator, then apply scale to the whole sum.
        dq_acc = hl.dot(dA.to(kt.dtype), kt, acc=dq_cross_acc) * scale
        # dk uses q' = scale*q; the state term carries scale via dh. The matmul
        # is scaled before the add, so it stays a separate dot plus the add.
        dk_acc = hl.dot(dA.transpose(-2, -1).to(qt.dtype), qt) * scale + dk_state_acc

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

            attn = hl.dot(qt, kt.transpose(-2, -1), acc=attn)
            dv_state = hl.dot(kt, dht.to(kt.dtype), acc=dv_state)

        idx = hl.arange(C)
        causal = (idx[:, None] >= idx[None, :]).float()
        attn = attn * (causal * scale)

        dot = do[tile_bhn, :, tile_dv]
        dv_acc = hl.dot(attn.transpose(-2, -1).to(dot.dtype), dot, acc=dv_state)
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

        h_all = chunk_fwd_h(kf, vf)

        qf = q.reshape(BHN, C, D)
        kf2 = kf.reshape(BHN, C, D)
        vf2 = vf.reshape(BHN, C, DV)
        hf = h_all.reshape(BHN, D, DV)

        # q is passed unscaled; scale is folded into the output.
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

        # q is passed unscaled; the backward kernels fold scale in themselves.
        qf4 = q.reshape(BH, N, C, D)
        do4 = grad_output.reshape(BH, N, C, DV)

        qf2 = q.reshape(BHN, C, D)
        kf2 = k.reshape(BHN, C, D)
        vf2 = v.reshape(BHN, C, DV)
        hf = h_all.reshape(BHN, D, DV)
        dof = do4.reshape(BHN, C, DV)

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
        {"naive_recurrent": naive_fn, "fla": fla_fn},
        (q, k, v),
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
