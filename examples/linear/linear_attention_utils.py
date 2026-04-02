"""
Linear Attention Utilities
==========================

Utility functions for the linear attention engine:
- Pure PyTorch reference implementation (for validation only)
- WY decomposition helpers for correction paths
- Input generators for each attention variant
- Recurrent step for autoregressive decoding
- Kernel config / caching infrastructure
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════════
# Shared test helpers
# ════════════════════════════════════════════════════════════════════════════════


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error between two tensors."""
    return (a.float() - b.float()).norm().item() / b.float().norm().clamp(
        min=1e-8
    ).item()


def head_to_time_first(x: torch.Tensor) -> torch.Tensor:
    """Head-first [B,H,T,...] -> time-first [B,T,H,...] for FLA."""
    return x.transpose(1, 2).contiguous()


# ════════════════════════════════════════════════════════════════════════════════
# Reference implementation (pure PyTorch, for validation only)
# ════════════════════════════════════════════════════════════════════════════════


def chunked_linear_attn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
    C: int = 64,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pure PyTorch reference for chunked linear attention forward.

    Supports both scalar decay (g: [B,H,T]) and diagonal decay (g: [B,H,T,D]).
    For correction case (beta provided): uses sequential recurrence within chunks.
    For no-correction case: uses parallel matmul-based intra-chunk computation.
    """
    B, H, T, D = q.shape
    DV = v.shape[-1]
    N = T // C
    assert T % C == 0

    input_dtype = q.dtype
    q, k, v, g = q.float(), k.float(), v.float(), g.float()
    if beta is not None:
        beta = beta.float()
    if a is not None:
        a = a.float()

    diagonal_decay = g.dim() == 4

    qc = q.reshape(B, H, N, C, D)
    kc = k.reshape(B, H, N, C, D)
    vc = v.reshape(B, H, N, C, DV)
    if diagonal_decay:
        gc = g.reshape(B, H, N, C, D)
    else:
        gc = g.reshape(B, H, N, C)

    bc = beta.reshape(B, H, N, C) if beta is not None else None
    ac = a.reshape(B, H, N, C, D) if a is not None else None

    outputs = []
    if initial_state is not None:
        state = initial_state.float().clone()
    else:
        state = q.new_zeros(B, H, D, DV, dtype=torch.float32)

    for i in range(N):
        qi = qc[:, :, i]
        ki = kc[:, :, i]
        vi = vc[:, :, i]
        gi = gc[:, :, i]

        bi = bc[:, :, i] if bc is not None else None
        ai = ac[:, :, i] if ac is not None else ki

        if diagonal_decay and bi is not None:
            chunk_out = []
            s = state.clone()
            for j in range(C):
                alpha = torch.exp(gi[:, :, j])
                s = alpha.unsqueeze(-1) * s
                kj = ai[:, :, j]
                vj = vi[:, :, j]
                bj = bi[:, :, j].unsqueeze(-1).unsqueeze(-1)
                kts = torch.einsum("bhd,bhdv->bhv", kj, s)
                s = s - bj * torch.einsum("bhd,bhv->bhdv", kj, kts)
                s = s + bj * torch.einsum("bhd,bhv->bhdv", kj, vj)
                oj = torch.einsum("bhd,bhdv->bhv", qi[:, :, j], s)
                chunk_out.append(oj)
            outputs.append(torch.stack(chunk_out, dim=2).to(input_dtype))
            state = s

        elif diagonal_decay:
            g_cs = gi.cumsum(-2)
            g_last = g_cs[:, :, -1]

            q_scaled = qi * torch.exp(g_cs)
            k_scaled_intra = ki * torch.exp(-g_cs)

            o_cross = torch.einsum("bhcd,bhdv->bhcv", q_scaled, state)

            attn = torch.einsum("bhid,bhjd->bhij", q_scaled, k_scaled_intra)
            causal_mask = torch.tril(torch.ones(C, C, device=q.device))
            attn = attn * causal_mask
            o_intra = torch.einsum("bhij,bhjv->bhiv", attn, vi)

            outputs.append((o_cross + o_intra).to(input_dtype))

            k_for_state = ki * torch.exp(g_last[:, :, None, :] - g_cs)
            state = state * torch.exp(g_last).unsqueeze(-1) + torch.einsum(
                "bhcd,bhcv->bhdv", k_for_state, vi
            )

        elif bi is not None:
            chunk_out = []
            s = state.clone()
            for j in range(C):
                alpha = torch.exp(gi[:, :, j]).unsqueeze(-1).unsqueeze(-1)
                s = alpha * s
                kj = ai[:, :, j]
                vj = vi[:, :, j]
                bj = bi[:, :, j].unsqueeze(-1).unsqueeze(-1)
                kts = torch.einsum("bhd,bhdv->bhv", kj, s)
                s = s - bj * torch.einsum("bhd,bhv->bhdv", kj, kts)
                s = s + bj * torch.einsum("bhd,bhv->bhdv", kj, vj)
                oj = torch.einsum("bhd,bhdv->bhv", qi[:, :, j], s)
                chunk_out.append(oj)
            outputs.append(torch.stack(chunk_out, dim=2).to(input_dtype))
            state = s

        else:
            g_cs = gi.cumsum(-1)
            g_last = g_cs[:, :, -1]

            o_cross = torch.einsum("bhcd,bhdv->bhcv", qi, state)
            o_cross = o_cross * torch.exp(g_cs).unsqueeze(-1)

            attn = torch.einsum("bhid,bhjd->bhij", qi, ki)
            decay_mask = torch.exp(g_cs.unsqueeze(-1) - g_cs.unsqueeze(-2))
            causal_mask = torch.tril(torch.ones(C, C, device=q.device))
            attn = attn * decay_mask * causal_mask
            o_intra = torch.einsum("bhij,bhjv->bhiv", attn, vi)

            outputs.append((o_cross + o_intra).to(input_dtype))

            k_scaled = ki * torch.exp(g_last[..., None, None] - g_cs.unsqueeze(-1))
            state = state * torch.exp(g_last).unsqueeze(-1).unsqueeze(
                -1
            ) + torch.einsum("bhcd,bhcv->bhdv", k_scaled, vi)

    return torch.stack(outputs, dim=2).reshape(B, H, T, DV).to(input_dtype)


# ════════════════════════════════════════════════════════════════════════════════
# Naive recurrent reference (step-by-step, for correctness validation)
# ════════════════════════════════════════════════════════════════════════════════


def naive_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor | None = None,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Step-by-step recurrent computation. Slow but correct."""
    B, H, T, D = q.shape
    DV = v.shape[-1]
    S = q.new_zeros(B, H, D, DV, dtype=torch.float32)
    outputs = []
    diagonal = decay.dim() == 4

    for t in range(T):
        qt = q[:, :, t].float() * q_scale
        kt = k[:, :, t].float()
        vt = v[:, :, t].float()
        bt = beta[:, :, t] if beta is not None else None

        if diagonal:
            alpha = torch.exp(decay[:, :, t])
            S = alpha.unsqueeze(-1) * S
        else:
            alpha = torch.exp(decay[:, :, t])
            S = alpha.unsqueeze(-1).unsqueeze(-1) * S

        if bt is not None:
            kts = torch.einsum("bhd,bhdv->bhv", kt, S)
            S = S - bt.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
                "bhd,bhv->bhdv", kt, kts
            )
            S = S + bt.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
                "bhd,bhv->bhdv", kt, vt
            )
        else:
            S = S + torch.einsum("bhd,bhv->bhdv", kt, vt)

        ot = torch.einsum("bhd,bhdv->bhv", qt, S)
        outputs.append(ot)

    return torch.stack(outputs, dim=2)


# ════════════════════════════════════════════════════════════════════════════════
# WY decomposition helpers
# ════════════════════════════════════════════════════════════════════════════════


def solve_tril_inv(A: torch.Tensor) -> torch.Tensor:
    """Compute (I + A)^{-1} where A is strictly lower triangular."""
    BHN, C, _ = A.shape
    I = torch.eye(C, device=A.device, dtype=A.dtype).unsqueeze(0).expand(BHN, -1, -1)
    return torch.linalg.solve_triangular(I + A, I, upper=False, unitriangular=True)


def prepare_wy_repr_bwd(
    a: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cs_d: torch.Tensor,
    A_inv: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backprop through WY decomposition (unified scalar/diagonal)."""
    C = a.shape[1]

    exp_g = torch.exp(g_cs_d)
    exp_neg_g = torch.exp(-g_cs_d)
    a_fwd = a * exp_g
    a_bwd = a * exp_neg_g
    ba = beta.unsqueeze(-1) * a_fwd
    bv = beta.unsqueeze(-1) * v

    A_inv_T = A_inv.transpose(-2, -1)

    d_ba = torch.bmm(A_inv_T, dw)
    d_bv = torch.bmm(A_inv_T, du)

    dAinv = torch.bmm(dw, ba.transpose(-2, -1)) + torch.bmm(du, bv.transpose(-2, -1))
    dA = -torch.bmm(A_inv_T, torch.bmm(dAinv, A_inv_T))

    idx = torch.arange(C, device=a.device)
    mask = (idx.unsqueeze(-1) > idx.unsqueeze(-2)).float()
    dA = dA * mask

    dL_dots = dA * beta.unsqueeze(-1)

    da_fwd_from_A = torch.bmm(dL_dots, a_bwd)
    da_bwd_from_A = torch.bmm(dL_dots.transpose(-2, -1), a_fwd)

    da_from_A = da_fwd_from_A * exp_g + da_bwd_from_A * exp_neg_g
    dg_d_from_A = da_fwd_from_A * a_fwd - da_bwd_from_A * a_bwd

    dots = torch.bmm(a_fwd, a_bwd.transpose(-2, -1))
    dbeta_from_A = (dA * dots).sum(-1)

    da_from_ba = d_ba * beta.unsqueeze(-1) * exp_g
    dbeta_from_ba = (d_ba * a_fwd).sum(-1)
    dg_d_from_ba = d_ba * ba

    dv_out = d_bv * beta.unsqueeze(-1)
    dbeta_from_bv = (d_bv * v).sum(-1)

    da = da_from_A + da_from_ba
    dbeta = dbeta_from_A + dbeta_from_ba + dbeta_from_bv
    dg_cs_d = dg_d_from_A + dg_d_from_ba

    return da, dv_out, dbeta, dg_cs_d


# ════════════════════════════════════════════════════════════════════════════════
# Recurrent step (autoregressive decoding)
# ════════════════════════════════════════════════════════════════════════════════


def recurrent_step_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    alpha: float | torch.Tensor = 1.0,
    beta_val: torch.Tensor | None = None,
    a_val: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step recurrent update (pure PyTorch reference).

    Returns (output, new_state).
    """
    k_sq = k.squeeze(2)
    v_sq = v.squeeze(2)
    q_sq = q.squeeze(2)

    if isinstance(alpha, torch.Tensor):
        alpha_sq = alpha.squeeze(2)
        if alpha_sq.dim() == 3:
            state = alpha_sq.unsqueeze(-1) * state
        else:
            state = alpha_sq.unsqueeze(-1).unsqueeze(-1) * state
    else:
        state = alpha * state

    if beta_val is not None:
        b = beta_val.squeeze(2)
        a_sq = a_val.squeeze(2) if a_val is not None else k_sq

        if b.dim() == 3:
            r = b.shape[-1]
            for j in range(r):
                bj = b[:, :, j].unsqueeze(-1).unsqueeze(-1)
                aj = a_sq[:, :, j]
                kts = torch.einsum("bhd,bhdv->bhv", aj, state)
                state = state - bj * torch.einsum("bhd,bhv->bhdv", aj, kts)
                state = state + bj * torch.einsum("bhd,bhv->bhdv", aj, v_sq)
        else:
            b_exp = b.unsqueeze(-1).unsqueeze(-1)
            kts = torch.einsum("bhd,bhdv->bhv", a_sq, state)
            state = state - b_exp * torch.einsum("bhd,bhv->bhdv", a_sq, kts)
            state = state + b_exp * torch.einsum("bhd,bhv->bhdv", a_sq, v_sq)
    else:
        state = state + torch.einsum("bhd,bhv->bhdv", k_sq, v_sq)

    o = torch.einsum("bhd,bhdv->bhv", q_sq, state)
    return o.unsqueeze(2), state


# ════════════════════════════════════════════════════════════════════════════════
# Input generators for each attention variant
# ════════════════════════════════════════════════════════════════════════════════


def make_simple_gla_inputs(
    B: int,
    H: int,
    T: int,
    D: int,
    DV: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create inputs for simple GLA (scalar decay)."""
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(
        B, H, T, DV, device=device, dtype=dtype, requires_grad=requires_grad
    )
    g = F.logsigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
    scale = 1.0 / math.sqrt(D)
    return q, k, v, g, scale


def make_full_gla_inputs(
    B: int,
    H: int,
    T: int,
    D: int,
    DV: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create inputs for full GLA (diagonal decay)."""
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(
        B, H, T, DV, device=device, dtype=dtype, requires_grad=requires_grad
    )
    g = F.logsigmoid(torch.randn(B, H, T, D, device=device, dtype=dtype))
    scale = 1.0 / math.sqrt(D)
    return q, k, v, g, scale


def make_delta_rule_inputs(
    B: int,
    H: int,
    T: int,
    D: int,
    DV: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create inputs for DeltaNet (correction, no decay)."""
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = F.normalize(torch.randn(B, H, T, D, device=device, dtype=dtype), dim=-1)
    if requires_grad:
        k = k.detach().requires_grad_(True)
    v = torch.randn(
        B, H, T, DV, device=device, dtype=dtype, requires_grad=requires_grad
    )
    beta = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
    g = torch.zeros(B, H, T, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, g, beta, scale


def make_gated_delta_rule_inputs(
    B: int,
    H: int,
    T: int,
    D: int,
    DV: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create inputs for Gated DeltaNet (correction + scalar decay)."""
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = F.normalize(torch.randn(B, H, T, D, device=device, dtype=dtype), dim=-1)
    if requires_grad:
        k = k.detach().requires_grad_(True)
    v = torch.randn(
        B, H, T, DV, device=device, dtype=dtype, requires_grad=requires_grad
    )
    beta = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
    g = F.logsigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
    scale = 1.0 / math.sqrt(D)
    return q, k, v, g, beta, scale


def make_vanilla_linear_attn_inputs(
    B: int,
    H: int,
    T: int,
    D: int,
    DV: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create inputs for vanilla linear attention (no decay)."""
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(
        B, H, T, DV, device=device, dtype=dtype, requires_grad=requires_grad
    )
    g = torch.zeros(B, H, T, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, g, scale


def make_retention_inputs(
    B: int,
    H: int,
    T: int,
    D: int,
    DV: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create inputs for RetNet (fixed per-head decay)."""
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(
        B, H, T, DV, device=device, dtype=dtype, requires_grad=requires_grad
    )
    g_gamma = (
        1 - 2.0 ** (-5 - torch.arange(H, dtype=torch.float32, device=device))
    ).log()
    g = g_gamma[None, :, None].expand(B, H, T).contiguous().to(dtype)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, g, scale


def make_mamba2_inputs(
    B: int,
    H: int,
    T: int,
    D: int,
    DV: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create inputs for Mamba-2 SSD (scalar decay, no correction)."""
    dt = F.softplus(torch.randn(B, T, H, device=device, dtype=dtype))
    A = -torch.rand(H, device=device, dtype=dtype) - 0.5
    B_mat = torch.randn(B, T, H, D, device=device, dtype=dtype)
    C_mat = torch.randn(B, T, H, D, device=device, dtype=dtype)
    x = torch.randn(B, T, H, DV, device=device, dtype=dtype)

    q = C_mat.transpose(1, 2).contiguous()
    k = B_mat.transpose(1, 2).contiguous()
    v = (x * dt.unsqueeze(-1)).transpose(1, 2).contiguous()
    g = (A[None, None, :] * dt).transpose(1, 2).contiguous()

    if requires_grad:
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)

    scale = 1.0
    return q, k, v, g, scale
