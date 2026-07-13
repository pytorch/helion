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
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
# Gather GEMV Kernel
# ------------------


# %%
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper], autotune_ignore_errors=True, autotune_effort="full")
def gather_gemv(w: Tensor, idx: Tensor, x: Tensor) -> Tensor:
    B, S1, S2 = w.size()
    N = idx.size(0)
    S = x.size(0)
    assert S1 == S2
    assert S == S1

    w_view = w.contiguous().view(B * S, S).to(x.dtype)  # [B*S, S]
    x = x.view(S, 1)  # [S, 1]
    out = torch.empty([N * S, 1], dtype=x.dtype, device=x.device)

    for tile_n_s in hl.tile(N * S):
        acc = hl.zeros([tile_n_s, 1], dtype=torch.float32)

        n_idx = tile_n_s.index
        which_idx = (n_idx / S).to(torch.int32)

        idx_gather = idx[which_idx]  # [tile_n_s]

        row_offset = idx_gather * S + (n_idx % S)  # [tile_n_s]

        for tile_k in hl.tile(S):
            # gathered: [tile_n_s, tile_k] -
            gathered = w_view[row_offset, tile_k]  # [tile_n_s, tile_k]

            # x_val: [tile_k, 1] -
            x_val = x[tile_k, :]  # [tile_k, 1]

            # [tile_n_s, tile_k] @ [tile_k, 1] = [tile_n_s, 1]
            #  sum replace dot
            acc += (gathered.to(torch.float32) * x_val.to(torch.float32).T).sum(dim=1, keepdim=True)

        out[tile_n_s, :] = acc.to(x.dtype)

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
    w = torch.randn((B, S, S), device=DEVICE, dtype=HALF_DTYPE)
    idx = torch.randint(0, B, [N], device=DEVICE, dtype=torch.int32)
    x = torch.randn((S), device=DEVICE, dtype=HALF_DTYPE)

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