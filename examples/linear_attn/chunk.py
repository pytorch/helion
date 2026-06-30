"""
Linear Attention — chunked-parallel
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
# Five kernels split along the chunked-parallel pipeline:
#   forward:   fwd_h  (serial state pass)  → fwd_o  (parallel output)
#   backward:  bwd_dh (reverse state pass) → bwd_dqk + bwd_dv (parallel)


@helion.kernel()
def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Serial state accumulation across chunks.

    h_all[i] holds the state going *into* chunk i:
        h_all[0] = h0
        h_all[i] = h_all[i-1] + k[i-1]^T @ v[i-1]

    Args:
        k:  [BH, N, C, D]
        v:  [BH, N, C, DV]
        h0: [BH, D, DV] initial state
    Returns:
        h_all: [BH, N, D, DV]
    """
    BH = k.size(0)
    N = k.size(1)
    D = k.size(3)
    DV = v.size(3)

    h_all = torch.empty([BH, N, D, DV], dtype=h0.dtype, device=h0.device)

    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = h0[idx, tile_d, tile_dv].float()

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
) -> torch.Tensor:
    """Per-chunk parallel output: o = q @ h + (q @ k^T).tril @ v."""
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
        out[tile_bhn, :, tile_dv] = (o_cross + o_intra).to(out.dtype)

    return out


@helion.kernel()
def chunk_bwd_dh(
    q: torch.Tensor,
    do: torch.Tensor,
    dh_init: torch.Tensor,
) -> torch.Tensor:
    """Serial reverse state-gradient pass.

    dh_all[i] holds the gradient *coming out of* chunk i:
        dh_all[N-1] = dh_init
        dh_all[i]   = dh_all[i+1] + q[i+1]^T @ do[i+1]
    """
    BH = q.size(0)
    N = q.size(1)
    D = q.size(3)
    DV = do.size(3)

    dh_all = torch.empty([BH, N, D, DV], dtype=dh_init.dtype, device=dh_init.device)

    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        dh_acc = dh_init[idx, tile_d, tile_dv].float()

        for i_t in hl.grid(N):
            i = N - 1 - i_t
            dh_all[idx, i, tile_d, tile_dv] = dh_acc.to(dh_all.dtype)
            q_i = q[idx, i, :, tile_d]
            do_i = do[idx, i, :, tile_dv]
            dh_acc = torch.addmm(dh_acc, q_i.transpose(-2, -1), do_i)

    return dh_all


@helion.kernel()
def chunk_bwd_dqk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-chunk parallel dQ, dK."""
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
        dq_acc = torch.bmm(dA.to(kt.dtype), kt) + dq_cross_acc
        dk_acc = torch.bmm(dA.transpose(-2, -1).to(qt.dtype), qt) + dk_state_acc

        dq_out[tile_bhn, :, tile_d] = dq_acc.to(q.dtype)
        dk_out[tile_bhn, :, tile_d] = dk_acc.to(k.dtype)

    return dq_out, dk_out


@helion.kernel()
def chunk_bwd_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
) -> torch.Tensor:
    """Per-chunk parallel dV."""
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
        attn = attn * causal

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
    ) -> torch.Tensor:
        B, H, T, D = q.shape
        DV = v.shape[-1]
        N = T // C
        BH = B * H
        BHN = BH * N

        kf = k.reshape(BH, N, C, D)
        vf = v.reshape(BH, N, C, DV)
        h0 = q.new_zeros(BH, D, DV, dtype=torch.float32)

        # h_all stays fp32 (accumulator); inputs ride at native dtype.
        h_all = chunk_fwd_h(kf, vf, h0)

        qf = q.reshape(BHN, C, D)
        kf2 = kf.reshape(BHN, C, D)
        vf2 = vf.reshape(BHN, C, DV)
        hf = h_all.reshape(BHN, D, DV)

        o = chunk_fwd_o(qf, kf2, vf2, hf)

        ctx.save_for_backward(q, k, v, h_all)
        ctx.C = C
        return o.reshape(B, H, T, DV)

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, h_all = ctx.saved_tensors
        C = ctx.C
        B, H, T, D = q.shape
        DV = v.shape[-1]
        N = T // C
        BH = B * H
        BHN = BH * N

        qf4 = q.reshape(BH, N, C, D)
        do4 = grad_output.reshape(BH, N, C, DV)
        dh0 = q.new_zeros(BH, D, DV, dtype=torch.float32)
        dh_all = chunk_bwd_dh(qf4, do4, dh0)

        qf2 = q.reshape(BHN, C, D)
        kf2 = k.reshape(BHN, C, D)
        vf2 = v.reshape(BHN, C, DV)
        hf = h_all.reshape(BHN, D, DV)
        dhf = dh_all.reshape(BHN, D, DV)
        dof = do4.reshape(BHN, C, DV)

        dq, dk = chunk_bwd_dqk(qf2, kf2, vf2, hf, dof, dhf)
        dv = chunk_bwd_dv(qf2, kf2, dof, dhf)

        return (
            dq.reshape(B, H, T, D),
            dk.reshape(B, H, T, D),
            dv.reshape(B, H, T, DV),
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
    return _LinearAttnFn.apply(q * scale, k, v, C)


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
    """Run a single representative shape (B=1, H=32, T=2048, D=DV=128, bf16)."""
    test(1, 32, 2048, 128, 128, HALF_DTYPE, device=DEVICE)


if __name__ == "__main__":
    main()
