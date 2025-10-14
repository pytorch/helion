"""
Helion Gather GEMV Kernel Example
=================================
This example demonstrates a Helion kernel implementation of a gather operation
followed by general matrix-vector multiplication (GEMV). The operation is:
w[idx].to(x.dtype) @ x, where w is a 3D tensor, idx contains indices to gather,
and x is a vector.

Based on the tritonbench gather_gemv operator that is motivated by Mixtral performance
where gather + gemv is the primary kernel.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
# Gather GEMV Kernel
# ------------------


# %%
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def gather_gemv(w: Tensor, idx: Tensor, x: Tensor) -> Tensor:
    """
    Performs a gather operation on w using idx, then matrix-vector multiplication with x.

    Args:
        w (Tensor): Weight matrix of shape [B, S, S] where B is batch size, S is sequence length.
        idx (Tensor): Index tensor of shape [N] containing indices to gather from dimension 0 of w.
        x (Tensor): Vector of shape [S] to multiply with the gathered matrices.

    Returns:
        Tensor: Result of shape [N, S] where each row i is w[idx[i]] @ x.
    """
    B, S1, S2 = w.size()
    N = idx.size(0)
    S = x.size(0)
    assert S1 == S2, f"Weight matrix must be square, got {S1} != {S2}"
    assert S == S1, f"Vector size {S} must match matrix size {S1}"

    # Rearrange shapes for matrix-vector multiplication
    w_view = w.contiguous().view(B * S, S).to(x.dtype)  # Shape: [N, S, S]
    x = x.view(S, 1)

    # Create output tensor
    out = torch.empty([N * S, 1], dtype=x.dtype, device=x.device)

    # Perform matrix-vector multiplication for each gathered matrix
    for tile_n_s in hl.tile(N * S):
        acc = hl.zeros([tile_n_s, 1], dtype=torch.float32)
        idx_id = tile_n_s.index // S
        idx_gather = idx[idx_id]
        for tile_k in hl.tile(S):
            # Matrix-vector multiplication
            gathered = w_view[idx_gather * S + tile_n_s.index % S, tile_k]
            acc += hl.dot(gathered, x[tile_k, :])
        out[tile_n_s, :] = acc

    return out.contiguous().view(N, S)


# %%
# Verification Function
# ---------------------


# %%
def check(B: int, S: int, N: int) -> None:
    """
    Verify the gather_gemv kernel implementation against PyTorch's baseline.

    Args:
        B (int): Batch size for weight matrix.
        S (int): Sequence length (matrix size).
        N (int): Number of indices to gather.
    """
    # Create test tensors matching tritonbench format
    w = torch.randn((B, S, S), device=DEVICE, dtype=torch.float16)
    idx = torch.randint(0, B, [N], device=DEVICE, dtype=torch.int32)
    x = torch.randn((S), device=DEVICE, dtype=torch.float16)

    def baseline_gather_gemv(w: Tensor, idx: Tensor, x: Tensor) -> Tensor:
        """PyTorch baseline implementation."""
        outputs = []
        for idx_val in idx.tolist():
            outputs.append(w[idx_val].to(x.dtype) @ x)
        return torch.stack(outputs, dim=0)

    run_example(gather_gemv, baseline_gather_gemv, (w, idx, x))


# %%
# Tritonbench Integration
# -----------------------


# %%
def gather_gemv_tritonbench(
    tb_op: object, w: Tensor, idx: Tensor, x: Tensor
) -> Callable:
    """
    Wrapper for tritonbench that matches its interface.

    Args:
        w (Tensor): Weight matrix of shape [B, S, S].
        idx (Tensor): Index tensor of shape [N].
        x (Tensor): Vector of shape [S].

    Returns:
        Callable: A callable that runs the gather_gemv kernel.
    """
    return lambda: gather_gemv(w, idx, x)


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the gather_gemv kernel verification.
    Uses sizes similar to tritonbench for consistency.
    """
    # Test with sizes from tritonbench
    B = 8  # Batch size, could be number of experts in MoE
    N = 2  # Number of indices, experts selected
    for i in range(11, 15):
        S = 2**i
        print(f"Testing with B={B}, S={S}, N={N}")
        check(B, S, N)


# %%
if __name__ == "__main__":
    main()
