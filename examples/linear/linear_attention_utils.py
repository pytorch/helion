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

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ════════════════════════════════════════════════════════════════════════════════
# Shared test helpers
# ════════════════════════════════════════════════════════════════════════════════


def rel_error(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    """Relative L2 error between two tensors."""
    assert a is not None and b is not None
    return (a.float() - b.float()).norm().item() / b.float().norm().clamp(
        min=1e-8
    ).item()


# Thresholds for the dashboard's helion_accuracy gate: Helion's relative L2
# error vs the fp32 PyTorch reference.
ACC_FWD_TOL = 0.02
ACC_BWD_TOL = 0.05


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

    def correction_chunk(
        qi: torch.Tensor,
        ki: torch.Tensor,
        vi: torch.Tensor,
        gi: torch.Tensor,
        bi: torch.Tensor,
        ai: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One chunk of the sequential beta-correction recurrence, returning
        (chunk_output, new_state); tensors are explicit args so it can be
        gradient-checkpointed."""
        s = state
        chunk_out = []
        for j in range(C):
            g_j = gi[:, :, j]
            alpha = (
                torch.exp(g_j).unsqueeze(-1)
                if diagonal_decay
                else torch.exp(g_j).unsqueeze(-1).unsqueeze(-1)
            )
            s = alpha * s
            kj = ai[:, :, j]
            vj = vi[:, :, j]
            bj = bi[:, :, j].unsqueeze(-1).unsqueeze(-1)
            kts = torch.einsum("bhd,bhdv->bhv", kj, s)
            s = s - bj * torch.einsum("bhd,bhv->bhdv", kj, kts)
            s = s + bj * torch.einsum("bhd,bhv->bhdv", kj, vj)
            chunk_out.append(torch.einsum("bhd,bhdv->bhv", qi[:, :, j], s))
        return torch.stack(chunk_out, dim=2), s

    for i in range(N):
        qi = qc[:, :, i]
        ki = kc[:, :, i]
        vi = vc[:, :, i]
        gi = gc[:, :, i]

        bi = bc[:, :, i] if bc is not None else None
        ai = ac[:, :, i] if ac is not None else ki

        if bi is not None:
            # Sequential beta-correction recurrence, checkpointed per chunk to
            # avoid OOMs.
            chunk_out, state = checkpoint(  # pyrefly: ignore[not-iterable]
                correction_chunk,
                qi,
                ki,
                vi,
                gi,
                bi,
                ai,
                state,
                use_reentrant=False,
            )
            outputs.append(chunk_out.to(input_dtype))

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

        out_t = torch.einsum("bhd,bhdv->bhv", qt, S)
        outputs.append(out_t)

    return torch.stack(outputs, dim=2)


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


def make_mamba2_inputs(
    B: int,
    H: int,
    T: int,
    D: int,
    DV: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
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
