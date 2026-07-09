"""
Jagged HSTU Attention (v2) Example -- Forward + Backward, Target Aware
=====================================================================

Jagged (unpadded) multi-head HSTU attention using data-dependent tile loops,
with forward *and* backward kernels and optional "target awareness".

Unlike :mod:`jagged_hstu_attn`, which tiles over a fixed ``max_seq_len``
and guards with ``if tile_q.begin < seq_len``, this version iterates
directly over each sequence's ``[start, end)`` range via
``hl.tile(start, end)``.  The approach avoids wasted work on padding and
is compatible with both the Triton and Pallas backends (including TPU).

Forward (per head, per sequence)::

    S   = alpha * (Q @ K^T)          # pre-activation scores
    P   = silu(S) * attn_scale       # SiLU-gated, unnormalized (no softmax)
    P   = P * keep_mask              # causal (+ target-aware) masking
    O   = P @ V

Backward (given dO)::

    dP  = dO @ V^T
    dV  = P^T @ dO
    dS  = dP * keep_mask * attn_scale * silu'(S)
    dQ  = alpha * (dS @ K)
    dK  = alpha * (dS^T @ Q)

where ``silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))``.

Target awareness
----------------
The last ``num_targets[b]`` tokens of each sequence are "targets" (ranking
candidates).  Targets attend to the whole user-interaction history (UIH) and
to themselves, but *not* to each other.  This is realized by clamping token
positions at ``c = seq_len - num_targets`` before the causal comparison, which
is equivalent to the per-column rule::

    keep(m, n) = (m == n) or ((m >= n) and (n < c))   # m = query, n = key

i.e. a "target" key column (``n >= c``) is only ever attended to by itself,
while a history column (``n < c``) uses ordinary causal attention.  When
``num_targets`` is ``None`` this reduces to plain lower-triangular causal
attention (``keep(m, n) = m >= n``).

Tensor shapes::

    q, k, v      : [L, H, D]    (L = total tokens across all sequences)
    seq_offsets  : [B + 1]      (int32 cumulative offsets)
    num_targets  : [B] or None  (int32, per-sequence target count)
    output       : [L, H, D]
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Reference Implementation
# ------------------------


# %%
def _jagged_to_padded(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """[L, D] jagged -> [B, max_len, D] padded with zeros."""
    B = offsets.shape[0] - 1
    D = values.shape[1]
    out = values.new_zeros(B, max_len, D)
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        n = min(e - s, max_len)
        out[i, :n] = values[s : s + n]
    return out


def _padded_to_jagged(
    padded: torch.Tensor,
    offsets: torch.Tensor,
    total_L: int,
) -> torch.Tensor:
    """[B, max_len, D] padded -> [L, D] jagged."""
    D = padded.shape[2]
    out = padded.new_zeros(total_L, D)
    B = offsets.shape[0] - 1
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        out[s:e] = padded[i, : e - s]
    return out


def _hstu_keep_mask(
    max_seq_len: int,
    seq_lengths: torch.Tensor,
    num_targets: torch.Tensor | None,
) -> torch.Tensor:
    """Build the per-sequence [B, N, N] boolean *keep* mask (True == attend).

    ``keep(m, n) = (m == n) or ((m >= n) and (n < c))`` with ``c = seq_len -
    num_targets`` (``c = N`` when ``num_targets`` is None, giving plain causal).
    """
    device = seq_lengths.device
    N = max_seq_len
    B = seq_lengths.size(0)
    ids = torch.arange(N, device=device)
    if num_targets is None:
        cap = seq_lengths.new_full((B, 1, 1), N)
    else:
        cap = (seq_lengths - num_targets).view(B, 1, 1)
    row = ids.view(1, N, 1)  # query position m
    col = ids.view(1, 1, N)  # key position n
    causal = row >= col
    n_is_history = col < cap
    diag = row == col
    return (causal & n_is_history) | diag


def reference_jagged_hstu_attention(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad-then-compute reference matching the production HSTU pattern.

    Differentiable w.r.t. ``q``/``k``/``v`` so it can serve as the autograd
    ground truth for the backward kernels.
    """
    L, H, D = q.shape

    padded_q = (
        _jagged_to_padded(q.reshape(L, H * D), seq_offsets, max_seq_len)
        .view(-1, max_seq_len, H, D)
        .transpose(1, 2)
    )  # [B, H, N, D]
    padded_k = (
        _jagged_to_padded(k.reshape(L, H * D), seq_offsets, max_seq_len)
        .view(-1, max_seq_len, H, D)
        .transpose(1, 2)
    )  # [B, H, N, D]
    padded_v = (
        _jagged_to_padded(v.reshape(L, H * D), seq_offsets, max_seq_len)
        .view(-1, max_seq_len, H, D)
        .transpose(1, 2)
    )  # [B, H, N, D]

    qk_attn = torch.einsum("bhxa,bhya->bhxy", padded_q, padded_k) * alpha
    qk_attn = F.silu(qk_attn) * attn_scale

    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    keep_mask = _hstu_keep_mask(max_seq_len, seq_lengths, num_targets)
    qk_attn = qk_attn * keep_mask.unsqueeze(1).to(qk_attn.dtype)

    attn_out = torch.einsum("bhxd,bhdv->bhxv", qk_attn.to(v.dtype), padded_v)
    return _padded_to_jagged(
        attn_out.transpose(1, 2).flatten(2, 3),
        seq_offsets,
        L,
    ).view(L, H, D)


# %%
# Forward Kernel
# --------------


# %%
@helion.kernel(
    static_shapes=True,
    autotune_baseline_fn=reference_jagged_hstu_attention,
)
def jagged_hstu_attention(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Jagged HSTU attention forward pass.

    For each sequence defined by ``seq_offsets``, computes causal SiLU-gated
    attention with optional target awareness::

        scores = silu(q @ k^T * alpha) * attn_scale
        scores = where(keep_mask, scores, 0)
        out = scores @ v
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    out = torch.empty_like(v)

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        # First target index (global). With no targets, cap == end so every
        # key is "history" and the mask is plain causal.
        if num_targets is not None:
            cap = end - num_targets[seq_idx]
        else:
            cap = end

        for tile_q in hl.tile(start, end):
            # [tile_q, H, D] -> [H, tile_q, D] for batched matmul
            q_blk = q[tile_q, :, :].transpose(0, 1)
            acc = hl.zeros([H, tile_q, D], dtype=torch.float32)

            for tile_kv in hl.tile(start, end):
                k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
                v_blk = v[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]

                # [H, tile_q, D] @ [H, D, tile_kv] -> [H, tile_q, tile_kv]
                scores = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * alpha
                scores = F.silu(scores) * attn_scale

                # target-aware causal mask: [tile_q, tile_kv] broadcast over H
                q_idx = tile_q.index.unsqueeze(1)
                kv_idx = tile_kv.index.unsqueeze(0)
                keep = ((q_idx >= kv_idx) & (kv_idx < cap)) | (q_idx == kv_idx)
                scores = torch.where(keep[None, :, :], scores, 0.0)

                # [H, tile_q, tile_kv] @ [H, tile_kv, D] -> [H, tile_q, D]
                acc = acc + torch.bmm(scores.to(v.dtype), v_blk)

            # [H, tile_q, D] -> [tile_q, H, D]
            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)

    return out


# %%
# Backward Kernel - dQ
# --------------------


# %%
@helion.kernel(
    static_shapes=True,
)
def jagged_hstu_attention_bwd_dq(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dO: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Jagged HSTU backward: compute dQ.

    For each q tile, accumulates over all kv tiles::

        S  = alpha * Q @ K^T
        dP = dO @ V^T
        dS = dP * keep_mask * attn_scale * silu'(S)
        dQ = alpha * (dS @ K)
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    dq = torch.empty_like(q)

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        if num_targets is not None:
            cap = end - num_targets[seq_idx]
        else:
            cap = end

        for tile_q in hl.tile(start, end):
            q_blk = q[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]
            do_blk = dO[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]
            acc_dq = hl.zeros([H, tile_q, D], dtype=torch.float32)

            for tile_kv in hl.tile(start, end):
                k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
                v_blk = v[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]

                # S = alpha * Q @ K^T  [H, tile_q, tile_kv]
                s = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * alpha
                # dP = dO @ V^T  [H, tile_q, tile_kv]
                dp = torch.bmm(do_blk, v_blk.transpose(-2, -1))

                # dS = dP * keep_mask * attn_scale * silu'(S)
                sig = torch.sigmoid(s)
                silu_grad = sig * (1.0 + s * (1.0 - sig))
                ds = dp * attn_scale * silu_grad

                q_idx = tile_q.index.unsqueeze(1)
                kv_idx = tile_kv.index.unsqueeze(0)
                keep = ((q_idx >= kv_idx) & (kv_idx < cap)) | (q_idx == kv_idx)
                ds = torch.where(keep[None, :, :], ds, 0.0)

                # dQ += alpha * dS @ K  [H, tile_q, tile_kv] @ [H, tile_kv, D]
                acc_dq = acc_dq + torch.bmm((ds * alpha).to(k.dtype), k_blk)

            dq[tile_q, :, :] = acc_dq.transpose(0, 1).to(dq.dtype)

    return dq


# %%
# Backward Kernel - dK
# --------------------


# %%
@helion.kernel(
    static_shapes=True,
)
def jagged_hstu_attention_bwd_dk(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dO: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Jagged HSTU backward: compute dK.

    For each kv tile, accumulates over all q tiles (transposed layout)::

        S^T  = alpha * K @ Q^T
        dP^T = V @ dO^T
        dS^T = dP^T * keep_mask^T * attn_scale * silu'(S^T)
        dK   = alpha * (dS^T @ Q)
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    dk = torch.empty_like(k)

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        if num_targets is not None:
            cap = end - num_targets[seq_idx]
        else:
            cap = end

        for tile_kv in hl.tile(start, end):
            k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
            v_blk = v[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
            acc_dk = hl.zeros([H, tile_kv, D], dtype=torch.float32)

            for tile_q in hl.tile(start, end):
                q_blk = q[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]
                do_blk = dO[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]

                # S^T = alpha * K @ Q^T  [H, tile_kv, tile_q]
                s_t = torch.bmm(k_blk, q_blk.transpose(-2, -1)) * alpha
                # dP^T = V @ dO^T  [H, tile_kv, tile_q]
                dp_t = torch.bmm(v_blk, do_blk.transpose(-2, -1))

                sig = torch.sigmoid(s_t)
                silu_grad = sig * (1.0 + s_t * (1.0 - sig))
                ds_t = dp_t * attn_scale * silu_grad

                # keep_mask^T: row = kv (n), col = q (m); keep(m, n)
                n_idx = tile_kv.index.unsqueeze(1)
                m_idx = tile_q.index.unsqueeze(0)
                keep_t = ((m_idx >= n_idx) & (n_idx < cap)) | (m_idx == n_idx)
                ds_t = torch.where(keep_t[None, :, :], ds_t, 0.0)

                # dK += alpha * dS^T @ Q  [H, tile_kv, tile_q] @ [H, tile_q, D]
                acc_dk = acc_dk + torch.bmm((ds_t * alpha).to(q.dtype), q_blk)

            dk[tile_kv, :, :] = acc_dk.transpose(0, 1).to(dk.dtype)

    return dk


# %%
# Backward Kernel - dV
# --------------------


# %%
@helion.kernel(
    static_shapes=True,
)
def jagged_hstu_attention_bwd_dv(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    dO: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Jagged HSTU backward: compute dV.

    For each kv tile, accumulates over all q tiles (transposed layout)::

        P^T = silu(alpha * K @ Q^T) * attn_scale * keep_mask^T
        dV  = P^T @ dO
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    dv = torch.empty_like(k)

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        if num_targets is not None:
            cap = end - num_targets[seq_idx]
        else:
            cap = end

        for tile_kv in hl.tile(start, end):
            k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
            acc_dv = hl.zeros([H, tile_kv, D], dtype=torch.float32)

            for tile_q in hl.tile(start, end):
                q_blk = q[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]
                do_blk = dO[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]

                # P^T = silu(alpha * K @ Q^T) * attn_scale  [H, tile_kv, tile_q]
                s_t = torch.bmm(k_blk, q_blk.transpose(-2, -1)) * alpha
                p_t = F.silu(s_t) * attn_scale

                n_idx = tile_kv.index.unsqueeze(1)
                m_idx = tile_q.index.unsqueeze(0)
                keep_t = ((m_idx >= n_idx) & (n_idx < cap)) | (m_idx == n_idx)
                p_t = torch.where(keep_t[None, :, :], p_t, 0.0)

                # dV += P^T @ dO  [H, tile_kv, tile_q] @ [H, tile_q, D]
                acc_dv = acc_dv + torch.bmm(p_t.to(dO.dtype), do_blk)

            dv[tile_kv, :, :] = acc_dv.transpose(0, 1).to(dv.dtype)

    return dv


# %%
# Autograd wrapper (forward + backward)
# -------------------------------------


# %%
class HSTUAttentionFunction(torch.autograd.Function):
    """Differentiable jagged HSTU attention, ready to drop into an nn module.

    Composes the forward and the three backward kernels so that a training
    graph tracing ``fwd + bwd + optimizer`` can use it directly.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx: Any,  # noqa: ANN401
        max_seq_len: int,
        alpha: float,
        attn_scale: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        num_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        out = jagged_hstu_attention(
            max_seq_len, alpha, attn_scale, q, k, v, seq_offsets, num_targets
        )
        ctx.save_for_backward(q, k, v)
        ctx.seq_offsets = seq_offsets
        ctx.num_targets = num_targets
        ctx.max_seq_len = max_seq_len
        ctx.alpha = alpha
        ctx.attn_scale = attn_scale
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        dO: torch.Tensor,
    ) -> tuple[None, None, None, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v = ctx.saved_tensors
        seq_offsets = ctx.seq_offsets
        num_targets = ctx.num_targets
        max_seq_len = ctx.max_seq_len
        alpha = ctx.alpha
        attn_scale = ctx.attn_scale
        dO = dO.contiguous()

        dq = jagged_hstu_attention_bwd_dq(
            max_seq_len, alpha, attn_scale, q, k, v, dO, seq_offsets, num_targets
        )
        dk = jagged_hstu_attention_bwd_dk(
            max_seq_len, alpha, attn_scale, q, k, v, dO, seq_offsets, num_targets
        )
        dv = jagged_hstu_attention_bwd_dv(
            max_seq_len, alpha, attn_scale, q, k, dO, seq_offsets, num_targets
        )
        return None, None, None, dq, dk, dv, None, None


def hstu_attention_fwd_bwd(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autograd-enabled entry point (forward + backward)."""
    return HSTUAttentionFunction.apply(  # type: ignore[no-any-return]
        max_seq_len, alpha, attn_scale, q, k, v, seq_offsets, num_targets
    )


# %%
# Main
# ----


# %%
def _make_inputs(
    num_sequences: int,
    max_seq_len: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    with_targets: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    lengths = torch.randint(
        max_seq_len // 2,
        max_seq_len + 1,
        (num_sequences,),
        dtype=torch.int32,
    )
    seq_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(lengths, dim=0).to(torch.int32),
        ]
    ).to(device)
    total_l = int(seq_offsets[-1].item())

    q = torch.randn(total_l, heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_l, heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_l, heads, head_dim, dtype=dtype, device=device)

    num_targets = None
    if with_targets:
        # A quarter of every sequence are ranking targets (>= 1, < seq_len).
        num_targets = torch.clamp(lengths // 4, min=1).to(device)

    return q, k, v, seq_offsets, num_targets


def _check_backward(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None,
    atol: float,
    rtol: float,
) -> None:
    """Compare each backward kernel (and the fused autograd path) against the
    autograd gradients of the pure-PyTorch reference forward."""
    dO = torch.randn_like(q)

    # Autograd ground truth from the reference forward.
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = reference_jagged_hstu_attention(
        max_seq_len, alpha, attn_scale, q_ref, k_ref, v_ref, seq_offsets, num_targets
    )
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dO)

    # Standalone backward kernels.
    dq = jagged_hstu_attention_bwd_dq(
        max_seq_len, alpha, attn_scale, q, k, v, dO, seq_offsets, num_targets
    )
    dk = jagged_hstu_attention_bwd_dk(
        max_seq_len, alpha, attn_scale, q, k, v, dO, seq_offsets, num_targets
    )
    dv = jagged_hstu_attention_bwd_dv(
        max_seq_len, alpha, attn_scale, q, k, dO, seq_offsets, num_targets
    )

    for name, got, ref in (
        ("dQ", dq, dq_ref),
        ("dK", dk, dk_ref),
        ("dV", dv, dv_ref),
    ):
        max_diff = (got.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol)
        print(f"  {name}: max abs diff = {max_diff:.3e}  OK")

    # Fused autograd path (exercises HSTUAttentionFunction end to end).
    q_g = q.detach().clone().requires_grad_(True)
    k_g = k.detach().clone().requires_grad_(True)
    v_g = v.detach().clone().requires_grad_(True)
    out = hstu_attention_fwd_bwd(
        max_seq_len, alpha, attn_scale, q_g, k_g, v_g, seq_offsets, num_targets
    )
    out.backward(dO)
    for name, grad, ref in (
        ("autograd dQ", q_g.grad, dq_ref),
        ("autograd dK", k_g.grad, dk_ref),
        ("autograd dV", v_g.grad, dv_ref),
    ):
        assert grad is not None
        torch.testing.assert_close(grad.float(), ref.float(), atol=atol, rtol=rtol)
        print(f"  {name}: OK")


def main() -> None:
    torch.manual_seed(0)
    num_sequences = 16
    max_seq_len = 512
    # Heads/head_dim are kept modest so the dense pad-then-compute reference (and
    # its autograd graph, used as the backward ground truth) stays within memory:
    # it materializes [B, H, max_seq_len, max_seq_len] score matrices.
    heads = 4
    head_dim = 128
    alpha = 1.0 / head_dim**2
    attn_scale = 1.0 / max_seq_len
    dtype = torch.float32
    device = torch.device(DEVICE)

    for with_targets in (False, True):
        tag = "target-aware" if with_targets else "causal-only"
        print(f"\n===== Forward ({tag}) =====")
        q, k, v, seq_offsets, num_targets = _make_inputs(
            num_sequences, max_seq_len, heads, head_dim, dtype, device, with_targets
        )
        fwd_args = (max_seq_len, alpha, attn_scale, q, k, v, seq_offsets, num_targets)
        run_example(
            lambda *a: jagged_hstu_attention(*a),
            lambda *a: reference_jagged_hstu_attention(*a),
            fwd_args,
            atol=1e-2,
            rtol=1e-2,
        )

        print(f"===== Backward ({tag}) =====")
        _check_backward(
            max_seq_len,
            alpha,
            attn_scale,
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            atol=2e-2,
            rtol=2e-2,
        )


if __name__ == "__main__":
    main()
