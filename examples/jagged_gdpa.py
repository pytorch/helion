"""
Jagged GDPA Example
====================

Jagged (unpadded) multi-head GDPA (Generalized Dot Product Attention) forward
and backward passes using data-dependent tile loops.

This kernel iterates over sequences via ``hl.grid(num_sequences)`` and accesses
dynamic start/end ranges using ``hl.tile(start, end)``, matching the pattern
from :mod:`jagged_hstu_attn_2`. This approach works on both Triton and Pallas
backends (including TPU).

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
    k, v         : [L_kv, H, D]  (L_kv = total key/value tokens across all sequences)
    q_offsets    : [B + 1]       (int32 cumulative offsets for queries)
    kv_offsets   : [B + 1]       (int32 cumulative offsets for keys/values)
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


def reference_jagged_gdpa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    qk_scale: float,
) -> torch.Tensor:
    """PyTorch reference for jagged GDPA forward (no activation)."""
    B = q_offsets.size(0) - 1
    batch_outputs = []

    for b in range(B):
        q_start, q_end = int(q_offsets[b]), int(q_offsets[b + 1])
        k_start, k_end = int(kv_offsets[b]), int(kv_offsets[b + 1])

        # https://github.com/google-pytorch/torch_tpu/issues/1066 requires us to use clone
        q_batch = q[q_start:q_end].clone()  # [seq_len_q, H, D]
        k_batch = k[k_start:k_end].clone()  # [seq_len_kv, H, D]
        v_batch = v[k_start:k_end].clone()  # [seq_len_kv, H, D]

        q_b = q_batch.permute(1, 0, 2)  # [H, seq_len_q, D]
        k_b = k_batch.permute(1, 0, 2)  # [H, seq_len_kv, D]
        v_b = v_batch.permute(1, 0, 2)  # [H, seq_len_kv, D]

        # [H, seq_len_q, seq_len_kv]
        attn_weight = torch.bmm(q_b, k_b.transpose(-2, -1)) * qk_scale

        # [H, seq_len_q, D]
        out_batch = torch.bmm(attn_weight.to(v.dtype), v_b)

        batch_outputs.append(out_batch.permute(1, 0, 2).contiguous())

    return torch.cat(batch_outputs, dim=0)


def reference_jagged_gdpa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dO: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    qk_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference for jagged GDPA backward (no activation).

    Given dO (gradient of loss w.r.t. output), computes dQ, dK, dV.

    P = Q @ K^T * qk_scale          [H, M, N]
    O = P @ V                        [H, M, D]

    dP = dO @ V^T                    [H, M, N]
    dV = P^T @ dO                    [H, N, D]
    dS = dP  (identity activation grad)
    dQ = dS * qk_scale @ K           [H, M, D]
    dK = (dS * qk_scale)^T @ Q       [H, N, D]
    """
    B = q_offsets.size(0) - 1
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for b in range(B):
        q_start, q_end = int(q_offsets[b]), int(q_offsets[b + 1])
        k_start, k_end = int(kv_offsets[b]), int(kv_offsets[b + 1])

        # https://github.com/google-pytorch/torch_tpu/issues/1066 requires us to use clone
        q_b = q[q_start:q_end].clone().permute(1, 0, 2).float()  # [H, M, D]
        k_b = k[k_start:k_end].clone().permute(1, 0, 2).float()  # [H, N, D]
        v_b = v[k_start:k_end].clone().permute(1, 0, 2).float()  # [H, N, D]
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
        dk[k_start:k_end] = dk_b.permute(1, 0, 2).to(dk.dtype)
        dv[k_start:k_end] = dv_b.permute(1, 0, 2).to(dv.dtype)

    return dq, dk, dv


# %%
# Forward Kernel
# --------------


@helion.kernel(
    config=helion.Config(block_sizes=[32, 32]),
    static_shapes=True,
)
def jagged_gdpa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    qk_scale: float,
) -> torch.Tensor:
    """Jagged GDPA forward pass.

    For each sequence defined by offsets, computes::

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
        k_start = kv_offsets[seq_idx]
        k_end = kv_offsets[seq_idx + 1]

        for tile_q in hl.tile(q_start, q_end):
            # [tile_q, H, D] -> [H, tile_q, D]
            q_blk = q[tile_q, :, :].transpose(0, 1)
            acc = hl.zeros([H, tile_q, D], dtype=torch.float32)

            for tile_kv in hl.tile(k_start, k_end):
                k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
                v_blk = v[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]

                # [H, tile_q, D] @ [H, D, tile_kv] -> [H, tile_q, tile_kv]
                scores = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * qk_scale

                # [H, tile_q, tile_kv] @ [H, tile_kv, D] -> [H, tile_q, D]
                acc = acc + torch.bmm(scores.to(v.dtype), v_blk)

            # [H, tile_q, D] -> [tile_q, H, D]
            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)

    return out


# %%
# Backward Kernel - dK
# ---------------------


@helion.kernel(
    config=helion.Config(block_sizes=[32, 32]),
    static_shapes=True,
)
def jagged_gdpa_bwd_dk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dO: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    qk_scale: float,
) -> torch.Tensor:
    """Jagged GDPA backward: compute dK.

    For each kv tile, accumulates contributions from all q tiles::

        dP^T = V @ dO^T                 [H, N, M]
        dK = dP^T * qk_scale @ Q        [H, N, D]
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = q_offsets.size(0) - 1
    dk = torch.empty_like(k)

    for seq_idx in hl.grid(num_sequences):
        q_start = q_offsets[seq_idx]
        q_end = q_offsets[seq_idx + 1]
        k_start = kv_offsets[seq_idx]
        k_end = kv_offsets[seq_idx + 1]

        for tile_kv in hl.tile(k_start, k_end):
            v_blk = v[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
            acc_dk = hl.zeros([H, tile_kv, D], dtype=torch.float32)

            for tile_q in hl.tile(q_start, q_end):
                q_blk = q[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]
                do_blk = dO[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]

                # dP^T = V @ dO^T  [H, tile_kv, tile_q]
                dpT = torch.bmm(v_blk, do_blk.transpose(-2, -1))

                # dK += dP^T * qk_scale @ Q  [H, tile_kv, tile_q] @ [H, tile_q, D]
                acc_dk = acc_dk + torch.bmm((dpT * qk_scale).to(q.dtype), q_blk)

            dk[tile_kv, :, :] = acc_dk.transpose(0, 1).to(dk.dtype)

    return dk


# %%
# Backward Kernel - dV
# ---------------------


@helion.kernel(
    config=helion.Config(block_sizes=[32, 32]),
    static_shapes=True,
)
def jagged_gdpa_bwd_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    dO: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    qk_scale: float,
) -> torch.Tensor:
    """Jagged GDPA backward: compute dV.

    For each kv tile, accumulates contributions from all q tiles::

        P^T = K @ Q^T * qk_scale        [H, N, M]
        dV = P^T @ dO                    [H, N, D]
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = q_offsets.size(0) - 1
    dv = torch.empty_like(k)

    for seq_idx in hl.grid(num_sequences):
        q_start = q_offsets[seq_idx]
        q_end = q_offsets[seq_idx + 1]
        k_start = kv_offsets[seq_idx]
        k_end = kv_offsets[seq_idx + 1]

        for tile_kv in hl.tile(k_start, k_end):
            k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
            acc_dv = hl.zeros([H, tile_kv, D], dtype=torch.float32)

            for tile_q in hl.tile(q_start, q_end):
                q_blk = q[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]
                do_blk = dO[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]

                # P^T = K @ Q^T * qk_scale  [H, tile_kv, tile_q]
                pT = torch.bmm(k_blk, q_blk.transpose(-2, -1)) * qk_scale

                # dV += P^T @ dO  [H, tile_kv, tile_q] @ [H, tile_q, D]
                acc_dv = acc_dv + torch.bmm(pT.to(dO.dtype), do_blk)

            dv[tile_kv, :, :] = acc_dv.transpose(0, 1).to(dv.dtype)

    return dv


# %%
# Backward Kernel - dQ
# ---------------------


@helion.kernel(
    config=helion.Config(block_sizes=[32, 32]),
    static_shapes=True,
)
def jagged_gdpa_bwd_dq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dO: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    qk_scale: float,
) -> torch.Tensor:
    """Jagged GDPA backward: compute dQ.

    For each q tile, accumulates contributions from all kv tiles::

        dP = dO @ V ^ T[H, tile_q, tile_kv]
        dQ = dP * qk_scale @ K[H, tile_q, D]
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = q_offsets.size(0) - 1
    dq = torch.empty_like(q)

    for seq_idx in hl.grid(num_sequences):
        q_start = q_offsets[seq_idx]
        q_end = q_offsets[seq_idx + 1]
        k_start = kv_offsets[seq_idx]
        k_end = kv_offsets[seq_idx + 1]

        for tile_q in hl.tile(q_start, q_end):
            do_blk = dO[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]

            acc_dq = hl.zeros([H, tile_q, D], dtype=torch.float32)

            for tile_kv in hl.tile(k_start, k_end):
                k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
                v_blk = v[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]

                # dP = dO @ V^T  [H, tile_q, tile_kv]
                dP = torch.bmm(do_blk, v_blk.transpose(-2, -1))

                # dQ += dP * qk_scale @ K  [H, tile_q, tile_kv] @ [H, tile_kv, D]
                acc_dq = acc_dq + torch.bmm((dP * qk_scale).to(k.dtype), k_blk)

            dq[tile_q, :, :] = acc_dq.transpose(0, 1).to(dq.dtype)

    return dq


# %%
# Main
# ----


def main() -> None:
    torch.manual_seed(0)
    num_sequences = 64
    max_seq_len_q = 128
    max_seq_len_kv = 128
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
    lengths_kv = torch.randint(
        max_seq_len_kv // 2,
        max_seq_len_kv + 1,
        (num_sequences,),
        dtype=torch.int32,
    )
    q_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(lengths_q, dim=0).to(torch.int32),
        ]
    ).to(device)
    kv_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(lengths_kv, dim=0).to(torch.int32),
        ]
    ).to(device)
    L_q = int(q_offsets[-1].item())
    L_kv = int(kv_offsets[-1].item())

    q = torch.randn(L_q, heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(L_kv, heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(L_kv, heads, head_dim, dtype=dtype, device=device)
    dO = torch.randn(L_q, heads, head_dim, dtype=dtype, device=device)

    # Test forward
    print("=== Forward ===")
    fwd_args = (q, k, v, q_offsets, kv_offsets, max_seq_len_q, max_seq_len_kv, qk_scale)
    run_example(
        lambda *a: jagged_gdpa_fwd(*a),
        lambda *a: reference_jagged_gdpa_fwd(*a),
        fwd_args,
        atol=1e-4,
        rtol=1e-4,
    )

    # Test backward dK
    print("\n=== Backward dK ===")
    bwd_args = (
        q,
        k,
        v,
        dO,
        q_offsets,
        kv_offsets,
        max_seq_len_q,
        max_seq_len_kv,
        qk_scale,
    )

    def ref_dk(*a):
        _, dk, _ = reference_jagged_gdpa_bwd(*a)
        return dk

    run_example(
        lambda *a: jagged_gdpa_bwd_dk(*a),
        lambda *a: ref_dk(*a),
        bwd_args,
        atol=1e-4,
        rtol=1e-4,
    )

    # Test backward dV
    print("\n=== Backward dV ===")
    dv_args = (
        q,
        k,
        dO,
        q_offsets,
        kv_offsets,
        max_seq_len_q,
        max_seq_len_kv,
        qk_scale,
    )

    def ref_dv(q_t, k_t, do_t, qo, kvo, msq, mskv, qks):
        _, _, dv_out = reference_jagged_gdpa_bwd(
            q_t, k_t, v, do_t, qo, kvo, msq, mskv, qks
        )
        return dv_out

    run_example(
        lambda *a: jagged_gdpa_bwd_dv(*a),
        lambda *a: ref_dv(*a),
        dv_args,
        atol=1e-4,
        rtol=1e-4,
    )

    # Test backward dQ
    print("\n=== Backward dQ ===")

    def ref_dq(*a):
        dq, _, _ = reference_jagged_gdpa_bwd(*a)
        return dq

    run_example(
        lambda *a: jagged_gdpa_bwd_dq(*a),
        lambda *a: ref_dq(*a),
        bwd_args,
        atol=1e-4,
        rtol=1e-4,
    )


if __name__ == "__main__":
    main()
