"""
Mixture-of-Experts (MoE) Matmul with Outer-Gather-Scatter (OGS)
===============================================================
This example demonstrates a Helion kernel implementation of a Mixture-of-Experts
matrix multiplication using an Outer-Gather-Scatter approach. It efficiently
handles token routing to multiple experts with variable token counts per expert.
The example includes:

- The Helion kernel performing tiled matmul per expert with masking for variable token counts.

- Helper functions to generate kernel arguments by sorting tokens by expert.

- A reference PyTorch implementation for correctness comparison.

- A check function to validate the Helion kernel against the reference.
"""

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel(static_shapes=False)
def moe_matmul_ogs(
    A: torch.Tensor,  # [T, K] - Input activations (T tokens, K features)
    W: torch.Tensor,  # [E, K, N] - Expert weights (E experts, K input features, N output features)
    expert_token_counts: torch.Tensor,  # [E] - Number of tokens assigned to each expert
    expert_token_offsets: torch.Tensor,  # [E + 1] - Starting position of each expert's tokens in sorted order
    sorted_to_orig_token_idx: torch.Tensor,  # [T] - Maps sorted token positions back to original positions
    max_T_per_expert: int,  # Maximum number of tokens per expert
) -> torch.Tensor:  # [T, N] - Output activations
    """
    Helion kernel implementing MoE matmul with Outer-Gather-Scatter.
    Args:
        A (torch.Tensor): Input activations of shape [T, K].
        W (torch.Tensor): Expert weights of shape [E, K, N].
        expert_token_counts (torch.Tensor): Number of tokens per expert [E].
        expert_token_offsets (torch.Tensor): Starting offsets of tokens per expert [E+1].
        sorted_to_orig_token_idx (torch.Tensor): Maps sorted token indices to original token indices [T].
        max_T_per_expert (int): Maximum number of tokens per expert.
    Returns:
        torch.Tensor: Output activations of shape [T, N].
    """
    T, K = A.shape
    E, _, N = W.shape
    C = torch.zeros(
        T,
        N,
        dtype=torch.promote_types(A.dtype, W.dtype),
        device=A.device,
    )
    for e_idx in hl.grid(E):
        start = expert_token_offsets[e_idx]
        num_tokens = expert_token_counts[e_idx]
        if num_tokens != 0:
            for tile_t, tile_n in hl.tile([max_T_per_expert, N]):
                local_token_offsets = tile_t.index
                token_valid = local_token_offsets < num_tokens
                local_token_offsets_valid = torch.where(
                    token_valid, local_token_offsets, 0
                )
                expert_sorted_token_indices = start + local_token_offsets_valid
                expert_orig_token_indices = sorted_to_orig_token_idx[
                    expert_sorted_token_indices.squeeze(0)
                ]
                acc = hl.zeros([tile_t, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(K):
                    A_frag = A[expert_orig_token_indices, tile_k]
                    W_frag = W[e_idx, tile_k, tile_n]
                    acc = torch.addmm(acc, A_frag, W_frag)
                block_T, block_N = acc.size()
                existing_values = C[expert_orig_token_indices, tile_n]
                mask_2d = token_valid.view(block_T, 1).expand(block_T, block_N)
                C[expert_orig_token_indices, tile_n] = torch.where(
                    mask_2d, acc.to(C.dtype), existing_values
                )
    return C


# %%
def moe_matmul_ogs_helion_kernel_args_gen(
    A: torch.Tensor,  # [T, K] - Input activations
    W: torch.Tensor,  # [E, K, N] - Expert weights
    top1_expert_per_token: torch.Tensor,  # [T] - Expert assignment for each token (0 to E-1)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Generates arguments for the Helion MoE matmul OGS kernel.
    Sorts tokens by expert, computes token counts and offsets per expert,
    and calculates max tokens per expert.
    Args:
        A (torch.Tensor): Input activations [T, K].
        W (torch.Tensor): Expert weights [E, K, N].
        top1_expert_per_token (torch.Tensor): Expert assignment per token [T].
    Returns:
        Tuple of tensors to be passed as kernel arguments.
    """
    E = W.size(0)
    device = A.device
    sorted_to_orig_token_idx = torch.argsort(top1_expert_per_token, stable=True).to(
        torch.int32
    )
    expert_token_counts = torch.bincount(top1_expert_per_token, minlength=E).to(
        torch.int32
    )
    expert_token_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    expert_token_offsets[0] = 0
    expert_token_offsets[1:] = torch.cumsum(expert_token_counts, 0, dtype=torch.int32)
    max_T_per_expert = int(expert_token_counts.max().item())
    return (
        A,
        W,
        expert_token_counts,
        expert_token_offsets,
        sorted_to_orig_token_idx,
        max_T_per_expert,
    )


# %%
def moe_matmul_ogs_reference(
    A: torch.Tensor, W: torch.Tensor, top1_expert_per_token: torch.Tensor
) -> torch.Tensor:
    """
    Reference PyTorch implementation of MoE matmul with OGS.
    Performs matmul per expert by selecting tokens assigned to each expert.
    Args:
        A (torch.Tensor): Input activations [T, K].
        W (torch.Tensor): Expert weights [E, K, N].
        top1_expert_per_token (torch.Tensor): Expert assignment per token [T].
    Returns:
        torch.Tensor: Output activations [T, N].
    """
    T, K = A.shape
    N = W.size(2)
    device, dtype = A.device, torch.promote_types(A.dtype, W.dtype)
    C = torch.empty(T, N, device=device, dtype=dtype)
    n_experts = W.size(0)
    for e in range(n_experts):
        token_idx = (top1_expert_per_token == e).nonzero(as_tuple=True)[0]
        if token_idx.numel() == 0:
            continue
        C[token_idx] = A[token_idx] @ W[e]
    return C


# %%
def check(T: int, K: int, N: int, n_experts: int) -> None:
    """
    Validates the Helion MoE matmul OGS kernel against the reference implementation.
    Generates random inputs and expert assignments, runs both implementations,
    and compares their outputs.
    Args:
        T (int): Number of tokens.
        K (int): Number of input features.
        N (int): Number of output features.
        n_experts (int): Number of experts.
    """
    dtype = torch.float16
    device = DEVICE if torch.accelerator.is_available() else "cpu"
    A = torch.randn(T, K, device=device, dtype=dtype)
    W = torch.randn(n_experts, K, N, device=device, dtype=dtype)
    top1_expert_per_token = torch.randint(n_experts, (T,), device=device)
    helion_kernel_args = moe_matmul_ogs_helion_kernel_args_gen(
        A, W, top1_expert_per_token
    )

    def helion_fn() -> torch.Tensor:
        return moe_matmul_ogs(*helion_kernel_args)

    def reference_fn() -> torch.Tensor:
        return moe_matmul_ogs_reference(A, W, top1_expert_per_token)

    run_example(helion_fn, reference_fn, ())


# %%
def main() -> None:
    """
    Main entry point to run the MoE matmul OGS kernel check with example parameters.
    """
    check(1000, 500, 200, 30)


# %%
if __name__ == "__main__":
    main()
