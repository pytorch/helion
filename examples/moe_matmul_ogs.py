"""
Mixture-of-Experts (MoE) matmul with Outer-Gather-Scatter (OGS)
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def moe_matmul_ogs(
    A: torch.Tensor,  # [B, K]
    W: torch.Tensor,  # [E, K, N]
    expert_token_counts: torch.Tensor,  # [E]
    expert_token_offsets: torch.Tensor,  # [E + 1]
    sorted_to_orig_token_idx: torch.Tensor,  # [B]
    T_max_tensor: torch.Tensor,  # [T_max]
):
    B, K = A.shape
    E, _, N = W.shape
    T_max = T_max_tensor.numel()

    C = torch.zeros(
        B,
        N,
        dtype=torch.promote_types(A.dtype, W.dtype),
        device=A.device,
    )

    row_ids = torch.arange(T_max, device=A.device, dtype=torch.int32)
    k_ids = torch.arange(K, device=A.device, dtype=torch.int32)

    for e_idx in hl.grid(E):
        start = expert_token_offsets[e_idx]
        num_tokens = expert_token_counts[e_idx]

        if num_tokens != 0:
            for tile_t, tile_n in hl.tile([T_max, N]):
                local_rows = row_ids[tile_t].squeeze(0).squeeze(0)  # [BLOCK_T]
                row_valid = (
                    (local_rows < num_tokens).squeeze(0).squeeze(0)
                )  # bool[BLOCK_T]

                # For invalid rows, use 0 as a dummy index (will be masked out)
                safe_offset = torch.where(
                    row_valid,
                    local_rows,
                    0,
                )

                orig_rows_idxes = start + safe_offset  # [1, BLOCK_T]
                orig_rows = sorted_to_orig_token_idx[
                    orig_rows_idxes.squeeze(0)
                ]  # [BLOCK_T]

                acc = hl.zeros([tile_t, tile_n], dtype=torch.float32)

                for tile_k in hl.tile(K):
                    col_ids = k_ids[tile_k].squeeze(0)
                    col_valid = col_ids < K  # bool[BLOCK_K]

                    load_mask = row_valid[:, None] & col_valid[None, :]

                    A_frag = A[orig_rows, tile_k]

                    W_frag = W[e_idx, tile_k, tile_n]  # [BLOCK_K, BLOCK_N]

                    acc = torch.addmm(acc, A_frag, W_frag)

                block_T = acc.size(0)
                block_N = acc.size(1)
                existing = C[orig_rows, tile_n]
                mask_2d = row_valid.view(block_T, 1).expand(block_T, block_N)
                C[orig_rows, tile_n] = torch.where(mask_2d, acc.to(C.dtype), existing)

    return C


def moe_matmul_ogs_helion_kernel_args_gen(
    A: torch.Tensor,  # [B, K]
    W: torch.Tensor,  # [E, K, N]
    top1_expert_per_token: torch.Tensor,  # [B]
) -> torch.Tensor:
    B = A.size(0)
    E = W.size(0)
    device = A.device

    sorting = torch.argsort(top1_expert_per_token, stable=True).to(torch.int32)  # [B]
    expert_token_counts = torch.bincount(top1_expert_per_token, minlength=E).to(
        torch.int32
    )  # [E]

    expert_token_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)  # [E+1]
    expert_token_offsets[0] = 0
    expert_token_offsets[1:] = torch.cumsum(expert_token_counts, 0, dtype=torch.int32)

    T_max = int(expert_token_counts.max().item())

    return (
        A,
        W,
        expert_token_counts,
        expert_token_offsets,
        sorting,
        torch.empty(T_max, device=device),
    )


def moe_matmul_ogs_reference(
    A: torch.Tensor, W: torch.Tensor, top1_expert_per_token: torch.Tensor
) -> torch.Tensor:
    B, K = A.shape
    N = W.size(2)
    device, dtype = A.device, torch.promote_types(A.dtype, W.dtype)

    C = torch.empty(B, N, device=device, dtype=dtype)
    n_experts = W.size(0)

    for e in range(n_experts):
        token_idx = (top1_expert_per_token == e).nonzero(as_tuple=True)[0]
        if token_idx.numel() == 0:
            continue
        C[token_idx] = A[token_idx] @ W[e]  # [Ne, K] @ [K, N]

    return C


def check() -> None:
    from triton.testing import do_bench

    B = 1000  # tokens / rows
    K = 500  # hidden size
    N = 200  # output size
    n_experts = 30
    dtype = torch.float16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    A = torch.randn(B, K, device=device, dtype=dtype)
    W = torch.randn(n_experts, K, N, device=device, dtype=dtype)
    top1_expert_per_token = torch.randint(n_experts, (B,), device=device)

    helion_kernel_args = moe_matmul_ogs_helion_kernel_args_gen(
        A, W, top1_expert_per_token
    )

    C_helion = moe_matmul_ogs(*helion_kernel_args)
    C_ref = moe_matmul_ogs_reference(A, W, top1_expert_per_token)
    torch.testing.assert_close(C_helion, C_ref, atol=1e-2, rtol=1e-2)

    sec = do_bench(lambda: moe_matmul_ogs(*helion_kernel_args))
    baseline_sec = do_bench(
        lambda: moe_matmul_ogs_reference(A, W, top1_expert_per_token)
    )
    print(
        f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}s, speed-up: {baseline_sec / sec:.2f}x"
    )


if __name__ == "__main__":
    check()
