"""
Jagged Q Dense KV GDPA Example
================================

GDPA (Generalized Dot Product Attention) forward and backward passes with
jagged Q (variable-length queries per sequence) and fixed-length KV.

The KV dimension is small enough to load in full (no tiling), while Q is
tiled. This kernel iterates over sequences via ``hl.grid(num_sequences)``
and uses ``hl.tile(q_start, q_end)`` for the query dimension.

Forward::

    P = Q @ K ^ T * qk_scale
    O = P @ V

Backward (given dO)::

    dP = dO @ V^T
    dV = P^T @ dO
    dS = activation_grad(Q @ K^T) * dP  (identity for no activation)
    dQ = dS * qk_scale @ K
    dK = dS^T * qk_scale @ Q

Tensor shapes::

    q            : [L_q, H, D]   (L_q = total query tokens across all sequences)
    k, v         : [B, N, H, D]  (N = fixed kv length per sequence)
    q_offsets    : [B + 1]       (int32 cumulative offsets for queries)
    output       : [L_q, H, D]
"""

# %%
# Imports
# -------

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Reference Implementations
# -------------------------


def reference_jagged_q_dense_kv_gdpa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    qk_scale: float,
) -> torch.Tensor:
    """PyTorch reference for dense GDPA forward (no activation).

    q: [L_q, H, D], k: [B, N, H, D], v: [B, N, H, D]
    """
    B = q_offsets.size(0) - 1
    batch_outputs = []

    for b in range(B):
        q_start, q_end = int(q_offsets[b]), int(q_offsets[b + 1])

        q_batch = q[q_start:q_end].clone()  # [seq_len_q, H, D]
        k_batch = k[b].clone()  # [N, H, D]
        v_batch = v[b].clone()  # [N, H, D]

        q_b = q_batch.permute(1, 0, 2)  # [H, seq_len_q, D]
        k_b = k_batch.permute(1, 0, 2)  # [H, N, D]
        v_b = v_batch.permute(1, 0, 2)  # [H, N, D]

        # [H, seq_len_q, N]
        attn_weight = torch.bmm(q_b, k_b.transpose(-2, -1)) * qk_scale

        # [H, seq_len_q, D]
        out_batch = torch.bmm(attn_weight.to(v.dtype), v_b)

        batch_outputs.append(out_batch.permute(1, 0, 2).contiguous())

    return torch.cat(batch_outputs, dim=0)


def reference_jagged_q_dense_kv_gdpa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dO: torch.Tensor,
    q_offsets: torch.Tensor,
    qk_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference for dense GDPA backward (no activation).

    q: [L_q, H, D], k: [B, N, H, D], v: [B, N, H, D], dO: [L_q, H, D]
    """
    B = q_offsets.size(0) - 1
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for b in range(B):
        q_start, q_end = int(q_offsets[b]), int(q_offsets[b + 1])

        q_b = q[q_start:q_end].clone().permute(1, 0, 2).float()  # [H, M, D]
        k_b = k[b].clone().permute(1, 0, 2).float()  # [H, N, D]
        v_b = v[b].clone().permute(1, 0, 2).float()  # [H, N, D]
        do_b = dO[q_start:q_end].clone().permute(1, 0, 2).float()  # [H, M, D]

        # P = Q @ K^T * qk_scale  [H, M, N]
        P = torch.bmm(q_b, k_b.transpose(-2, -1)) * qk_scale

        # dP = dO @ V^T  [H, M, N]
        dP = torch.bmm(do_b, v_b.transpose(-2, -1))

        # dV = P^T @ dO  [H, N, D]
        dv_b = torch.bmm(P.transpose(-2, -1), do_b)

        # dS = dP (identity activation gradient)
        dS = dP

        # dQ = dS * qk_scale @ K  [H, M, D]
        dq_b = torch.bmm(dS * qk_scale, k_b)

        # dK = (dS * qk_scale)^T @ Q  [H, N, D]
        dk_b = torch.bmm((dS * qk_scale).transpose(-2, -1), q_b)

        dq[q_start:q_end] = dq_b.permute(1, 0, 2).to(dq.dtype)
        dk[b] = dk_b.permute(1, 0, 2).to(dk.dtype)
        dv[b] = dv_b.permute(1, 0, 2).to(dv.dtype)

    return dq, dk, dv


# %%
# Forward Kernel
# --------------


@helion.kernel(
    config=helion.Config(block_sizes=[32]),
    static_shapes=True,
)
def jagged_q_dense_kv_gdpa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    qk_scale: float,
) -> torch.Tensor:
    """Dense GDPA forward pass.

    q: [L_q, H, D], k: [B, N, H, D], v: [B, N, H, D]

    For each sequence, computes::

        scores = q @ k ^ T * qk_scale
        out = scores @ v
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = q_offsets.size(0) - 1
    out = torch.empty_like(q)

    for seq_idx in hl.grid(num_sequences):
        q_start = q_offsets[seq_idx]
        q_end = q_offsets[seq_idx + 1]

        # [N, H, D] -> [H, N, D]
        k_blk = k[seq_idx, :, :, :].transpose(0, 1)
        v_blk = v[seq_idx, :, :, :].transpose(0, 1)

        for tile_q in hl.tile(q_start, q_end):
            # [tile_q, H, D] -> [H, tile_q, D]
            q_blk = q[tile_q, :, :].transpose(0, 1)

            # [H, tile_q, D] @ [H, D, N] -> [H, tile_q, N]
            scores = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * qk_scale

            # [H, tile_q, N] @ [H, N, D] -> [H, tile_q, D]
            acc = torch.bmm(scores.to(v.dtype), v_blk)

            # [H, tile_q, D] -> [tile_q, H, D]
            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)

    return out


# %%
# Backward Kernel - dK
# ---------------------


@helion.kernel(
    config=helion.Config(block_sizes=[32]),
    static_shapes=True,
)
def jagged_q_dense_kv_gdpa_bwd_dk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dO: torch.Tensor,
    q_offsets: torch.Tensor,
    qk_scale: float,
) -> torch.Tensor:
    """Dense GDPA backward: compute dK.

    For each sequence, accumulates contributions from all q tiles::

        dP^T = V @ dO^T                 [H, N, M]
        dK = dP^T * qk_scale @ Q        [H, N, D]
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    N = hl.specialize(k.size(1))
    num_sequences = q_offsets.size(0) - 1
    dk = torch.zeros_like(k)

    for seq_idx in hl.grid(num_sequences):
        q_start = q_offsets[seq_idx]
        q_end = q_offsets[seq_idx + 1]

        # [N, H, D] -> [H, N, D]
        v_blk = v[seq_idx, :, :, :].transpose(0, 1)

        for tile_q in hl.tile(q_start, q_end):
            # [tile_q, H, D] -> [H, tile_q, D]
            q_blk = q[tile_q, :, :].transpose(0, 1)
            do_blk = dO[tile_q, :, :].transpose(0, 1)

            # dP^T = V @ dO^T  [H, N, tile_q]
            dpT = torch.bmm(v_blk, do_blk.transpose(-2, -1))

            # dK += dP^T * qk_scale @ Q  [H, N, D]
            dk_contrib = torch.bmm((dpT * qk_scale).to(q.dtype), q_blk)

            # [H, N, D] -> [N, H, D]
            dk[seq_idx, :, :, :] += dk_contrib.transpose(0, 1).to(dk.dtype)

    return dk


# %%
# Backward Kernel - dV
# ---------------------


@helion.kernel(
    config=helion.Config(block_sizes=[32]),
    static_shapes=True,
)
def jagged_q_dense_kv_gdpa_bwd_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    dO: torch.Tensor,
    q_offsets: torch.Tensor,
    qk_scale: float,
) -> torch.Tensor:
    """Dense GDPA backward: compute dV.

    For each sequence, accumulates contributions from all q tiles::

        P^T = K @ Q^T * qk_scale        [H, N, M]
        dV = P^T @ dO                    [H, N, D]
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    N = hl.specialize(k.size(1))
    num_sequences = q_offsets.size(0) - 1
    dv = torch.zeros_like(k)

    for seq_idx in hl.grid(num_sequences):
        q_start = q_offsets[seq_idx]
        q_end = q_offsets[seq_idx + 1]

        # [N, H, D] -> [H, N, D]
        k_blk = k[seq_idx, :, :, :].transpose(0, 1)

        for tile_q in hl.tile(q_start, q_end):
            # [tile_q, H, D] -> [H, tile_q, D]
            q_blk = q[tile_q, :, :].transpose(0, 1)
            do_blk = dO[tile_q, :, :].transpose(0, 1)

            # P^T = K @ Q^T * qk_scale  [H, N, tile_q]
            pT = torch.bmm(k_blk, q_blk.transpose(-2, -1)) * qk_scale

            # dV += P^T @ dO  [H, N, D]
            dv_contrib = torch.bmm(pT.to(dO.dtype), do_blk)

            # [H, N, D] -> [N, H, D]
            dv[seq_idx, :, :, :] += dv_contrib.transpose(0, 1).to(dv.dtype)

    return dv


# %%
# Backward Kernel - dQ
# ---------------------


@helion.kernel(
    config=helion.Config(block_sizes=[32]),
    static_shapes=True,
)
def jagged_q_dense_kv_gdpa_bwd_dq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dO: torch.Tensor,
    q_offsets: torch.Tensor,
    qk_scale: float,
) -> torch.Tensor:
    """Dense GDPA backward: compute dQ.

    For each q tile, computes::

        dP = dO @ V^T                    [H, tile_q, N]
        dQ = dP * qk_scale @ K           [H, tile_q, D]
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = q_offsets.size(0) - 1
    dq = torch.empty_like(q)

    for seq_idx in hl.grid(num_sequences):
        q_start = q_offsets[seq_idx]
        q_end = q_offsets[seq_idx + 1]

        # [N, H, D] -> [H, N, D]
        k_blk = k[seq_idx, :, :, :].transpose(0, 1)
        v_blk = v[seq_idx, :, :, :].transpose(0, 1)

        for tile_q in hl.tile(q_start, q_end):
            # [tile_q, H, D] -> [H, tile_q, D]
            do_blk = dO[tile_q, :, :].transpose(0, 1)

            # dP = dO @ V^T  [H, tile_q, N]
            dP = torch.bmm(do_blk, v_blk.transpose(-2, -1))

            # dQ = dP * qk_scale @ K  [H, tile_q, D]
            acc_dq = torch.bmm((dP * qk_scale).to(k.dtype), k_blk)

            # [H, tile_q, D] -> [tile_q, H, D]
            dq[tile_q, :, :] = acc_dq.transpose(0, 1).to(dq.dtype)

    return dq


# %%
# Main
# ----


def main() -> None:
    torch.manual_seed(0)
    num_sequences = 64
    max_seq_len_q = 128
    seq_len_kv = 128
    heads = 8
    head_dim = 256
    qk_scale = 1.0
    dtype = torch.bfloat16
    device = torch.device(DEVICE)

    lengths_q = torch.randint(
        max_seq_len_q // 2,
        max_seq_len_q + 1,
        (num_sequences,),
        dtype=torch.int32,
    )
    q_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(lengths_q, dim=0).to(torch.int32),
        ]
    ).to(device)
    L_q = int(q_offsets[-1].item())

    q = torch.randn(L_q, heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(num_sequences, seq_len_kv, heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(num_sequences, seq_len_kv, heads, head_dim, dtype=dtype, device=device)
    dO = torch.randn(L_q, heads, head_dim, dtype=dtype, device=device)

    # Test forward
    print("=== Forward ===")
    fwd_args = (q, k, v, q_offsets, qk_scale)
    run_example(
        lambda *a: jagged_q_dense_kv_gdpa_fwd(*a),
        lambda *a: reference_jagged_q_dense_kv_gdpa_fwd(*a),
        fwd_args,
        atol=1e-4,
        rtol=1e-4,
    )

    # Test backward dK
    print("\n=== Backward dK ===")
    bwd_args = (q, k, v, dO, q_offsets, qk_scale)

    def ref_dk(*a):
        _, dk, _ = reference_jagged_q_dense_kv_gdpa_bwd(*a)
        return dk

    run_example(
        lambda *a: jagged_q_dense_kv_gdpa_bwd_dk(*a),
        lambda *a: ref_dk(*a),
        bwd_args,
        atol=1e-4,
        rtol=1e-4,
    )

    # Test backward dV
    print("\n=== Backward dV ===")
    dv_args = (q, k, dO, q_offsets, qk_scale)

    def ref_dv(q_t, k_t, do_t, qo, qks):
        _, _, dv_out = reference_jagged_q_dense_kv_gdpa_bwd(q_t, k_t, v, do_t, qo, qks)
        return dv_out

    run_example(
        lambda *a: jagged_q_dense_kv_gdpa_bwd_dv(*a),
        lambda *a: ref_dv(*a),
        dv_args,
        atol=1e-4,
        rtol=1e-4,
    )

    # Test backward dQ
    print("\n=== Backward dQ ===")

    def ref_dq(*a):
        dq, _, _ = reference_jagged_q_dense_kv_gdpa_bwd(*a)
        return dq

    run_example(
        lambda *a: jagged_q_dense_kv_gdpa_bwd_dq(*a),
        lambda *a: ref_dq(*a),
        bwd_args,
        atol=1e-4,
        rtol=1e-4,
    )


if __name__ == "__main__":
    main()
