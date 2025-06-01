"""
Mixture-of-Experts (MoE) matmul with Outer-Gather-Scatter (OGS)
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def moe_matmul_ogs(
    A: torch.Tensor,  # [T, K] - Input activations (T tokens, K features)
    W: torch.Tensor,  # [E, K, N] - Expert weights (E experts, K input features, N output features)
    expert_token_counts: torch.Tensor,  # [E] - Number of tokens assigned to each expert
    expert_token_offsets: torch.Tensor,  # [E + 1] - Starting position of each expert's tokens in sorted order
    sorted_to_orig_token_idx: torch.Tensor,  # [T] - Maps sorted token positions back to original positions
    max_T_per_expert_tensor: torch.Tensor,  # [max_T_per_expert] - Dummy tensor whose size indicates max tokens per expert
) -> torch.Tensor:  # [T, N] - Output activations
    # Extract dimensions from input tensors
    T, K = A.shape
    E, _, N = W.shape
    max_T_per_expert = (
        max_T_per_expert_tensor.numel()
    )  # Maximum number of tokens any expert processes

    C = torch.zeros(
        T,
        N,
        dtype=torch.promote_types(A.dtype, W.dtype),
        device=A.device,
    )

    # Iterate over each expert
    for e_idx in hl.grid(E):
        # Get the global range of tokens assigned to this expert
        start = expert_token_offsets[e_idx]  # Starting index in sorted token array
        num_tokens = expert_token_counts[e_idx]  # Number of tokens for this expert

        # Skip experts with no assigned tokens
        if num_tokens != 0:
            # Tile over tokens and output features for this expert
            for tile_t, tile_n in hl.tile([max_T_per_expert, N]):
                # Get local token offsets for this tile
                # (i.e. the tile's corresponding chunk in [0 .. max_T_per_expert-1] token range)
                local_token_offsets = tile_t.index  # [BLOCK_T]

                # Create mask for valid tokens (some tiles may be partially filled)
                token_valid = local_token_offsets < num_tokens  # bool[BLOCK_T]

                # For invalid tokens, use 0 as a dummy index (will be masked out later)
                local_token_offsets_valid = torch.where(
                    token_valid,
                    local_token_offsets,
                    0,
                )  # [BLOCK_T]

                # Convert local offsets to global sorted indices
                expert_sorted_token_indices = (
                    start + local_token_offsets_valid
                )  # [1, BLOCK_T]

                # Map sorted indices back to global original token positions
                expert_orig_token_indices = sorted_to_orig_token_idx[
                    expert_sorted_token_indices.squeeze(0)
                ]  # [BLOCK_T]

                acc = hl.zeros([tile_t, tile_n], dtype=torch.float32)

                # Perform tiled matrix multiplication: A[tokens, :] @ W[expert, :, :]
                for tile_k in hl.tile(K):
                    A_frag = A[expert_orig_token_indices, tile_k]  # [BLOCK_T, BLOCK_K]
                    W_frag = W[e_idx, tile_k, tile_n]  # [BLOCK_K, BLOCK_N]
                    acc = torch.addmm(acc, A_frag, W_frag)

                # Write results back to output tensor, masking out invalid tokens
                block_T = acc.size(0)
                block_N = acc.size(1)
                existing_values = C[expert_orig_token_indices, tile_n]
                mask_2d = token_valid.view(block_T, 1).expand(block_T, block_N)
                # Write results only for valid tokens, preserve existing values for invalid ones
                C[expert_orig_token_indices, tile_n] = torch.where(
                    mask_2d, acc.to(C.dtype), existing_values
                )

    return C


def moe_matmul_ogs_helion_kernel_args_gen(
    A: torch.Tensor,  # [T, K] - Input activations
    W: torch.Tensor,  # [E, K, N] - Expert weights
    top1_expert_per_token: torch.Tensor,  # [T] - Expert assignment for each token (0 to E-1)
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    E = W.size(0)  # Number of experts
    device = A.device

    # Sort tokens by their assigned expert to group tokens for the same expert together
    sorted_to_orig_token_idx = torch.argsort(top1_expert_per_token, stable=True).to(
        torch.int32
    )  # [T] - Maps position in sorted array to original token index

    # Count how many tokens are assigned to each expert
    expert_token_counts = torch.bincount(top1_expert_per_token, minlength=E).to(
        torch.int32
    )  # [E] - Number of tokens per expert

    # Compute starting offset for each expert's tokens in the sorted array
    # This allows O(1) lookup of where each expert's tokens begin
    expert_token_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)  # [E+1]
    expert_token_offsets[0] = 0
    expert_token_offsets[1:] = torch.cumsum(expert_token_counts, 0, dtype=torch.int32)

    # Find the maximum tokens assigned to any single expert
    # This determines the tile size for token dimension in the kernel
    max_T_per_expert = int(expert_token_counts.max().item())

    return (
        A,
        W,
        expert_token_counts,
        expert_token_offsets,
        sorted_to_orig_token_idx,
        torch.empty(
            max_T_per_expert, device=device
        ),  # Dummy tensor to pass max_T_per_expert as a tensor dimension
    )


def moe_matmul_ogs_reference(
    A: torch.Tensor, W: torch.Tensor, top1_expert_per_token: torch.Tensor
) -> torch.Tensor:
    T, K = A.shape
    N = W.size(2)
    device, dtype = A.device, torch.promote_types(A.dtype, W.dtype)

    C = torch.empty(T, N, device=device, dtype=dtype)
    n_experts = W.size(0)

    for e in range(n_experts):
        token_idx = (top1_expert_per_token == e).nonzero(as_tuple=True)[0]
        if token_idx.numel() == 0:
            continue
        C[token_idx] = A[token_idx] @ W[e]  # [Ne, K] @ [K, N]

    return C


def check() -> None:
    from triton.testing import do_bench

    T = 1000  # number of tokens
    K = 500  # hidden size
    N = 200  # output size
    n_experts = 30
    dtype = torch.float16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    A = torch.randn(T, K, device=device, dtype=dtype)
    W = torch.randn(n_experts, K, N, device=device, dtype=dtype)
    top1_expert_per_token = torch.randint(n_experts, (T,), device=device)

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
