"""
Fused SwiGLU with Pre-Reordered Weights
========================================
Fuses gate_proj + up_proj + SwiGLU activation into a single GEMM pass.

Key insight: Concatenate W_gate and W_up into a single [K, 2N] matrix.
Perform one wider GEMM, then bisect the accumulator in the epilogue to
apply SiLU(gate) * up — all in registers, zero HBM round-trips for
intermediate tensors.

This eliminates:
- 2 separate GEMM kernel launches (gate + up)
- 1 separate SwiGLU activation kernel launch
- 2 × M × N × 2 bytes of HBM traffic for intermediate tensors
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
# Fused SwiGLU GEMM Kernel
# ------------------------


# %%
@helion.kernel()
def fused_swiglu_gemm(X: Tensor, W_gate_up: Tensor) -> Tensor:
    """
    Fused gate_proj + up_proj + SwiGLU in a single GEMM.

    The weight matrix W_gate_up is [K, 2*N] where:
    - W_gate_up[:, :N] = gate_proj weights
    - W_gate_up[:, N:] = up_proj weights

    Computes: SiLU(X @ W_gate) * (X @ W_up)
    But in one GEMM pass with epilogue fusion.

    Args:
        X: Input activations [M, K].
        W_gate_up: Concatenated weights [K, 2*N].

    Returns:
        SwiGLU output [M, N].
    """
    M, K = X.shape
    _, N2 = W_gate_up.shape
    N = N2 // 2

    out = torch.empty(M, N, dtype=X.dtype, device=X.device)

    for tile_m, tile_n in hl.tile([M, N]):
        # Accumulator for gate projection (first N columns)
        acc_gate = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        # Accumulator for up projection (last N columns)
        acc_up = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        for tile_k in hl.tile(K):
            x_tile = X[tile_m, tile_k]

            # Load gate weights (first N columns)
            w_gate_tile = W_gate_up[tile_k, tile_n]
            acc_gate = torch.addmm(acc_gate, x_tile, w_gate_tile)

            # Load up weights (offset by N)
            n_begin = tile_n.begin + N
            n_end = n_begin + tile_n.block_size
            w_up_tile = W_gate_up[tile_k, n_begin:n_end]
            acc_up = torch.addmm(acc_up, x_tile, w_up_tile)

        # Epilogue: SiLU(gate) * up — all in registers
        silu_gate = acc_gate * torch.sigmoid(acc_gate)
        result = silu_gate * acc_up

        out[tile_m, tile_n] = result.to(out.dtype)

    return out


# %%
# SwiGLU Elementwise Kernel
# -------------------------


# %%
@helion.kernel()
def swiglu_elementwise(a: Tensor, b: Tensor) -> Tensor:
    """
    Standalone SwiGLU elementwise kernel: SiLU(a) * b.
    Used as a component and for backward compatibility.

    Args:
        a: Gate tensor (output of gate_proj).
        b: Up tensor (output of up_proj).

    Returns:
        SwiGLU output.
    """
    assert a.shape == b.shape
    out = torch.empty_like(a, dtype=torch.promote_types(a.dtype, b.dtype))
    total = a.numel()
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    out_flat = out.view(-1)

    for tile_idx in hl.tile(total):
        a_vals = a_flat[tile_idx].to(torch.float32)
        b_vals = b_flat[tile_idx]
        sigmoid_a = torch.sigmoid(a_vals)
        silu_a = a_vals * sigmoid_a
        result = silu_a.to(b_vals.dtype) * b_vals
        out_flat[tile_idx] = result

    return out


# %%
# Module
# ------


# %%
class FusedSwiGLUMLP(nn.Module):
    """
    LLaMA-style FFN MLP with fused SwiGLU.

    Instead of separate gate_proj, up_proj, then SwiGLU:
    - Concatenates gate + up weights offline into [K, 2N]
    - Runs a single fused GEMM with SwiGLU epilogue
    - Then runs down_proj normally
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Separate projections for weight initialization
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Will be populated by prepare_fused_weights()
        self._fused_weight: Tensor | None = None

    def prepare_fused_weights(self) -> None:
        """Concatenate gate and up weights for fused execution."""
        # gate_proj.weight is [N, K], need to transpose to [K, N] for GEMM
        gate_w = self.gate_proj.weight.data.T  # [K, N]
        up_w = self.up_proj.weight.data.T  # [K, N]
        self._fused_weight = torch.cat([gate_w, up_w], dim=1).contiguous()  # [K, 2N]

    def forward(self, x: Tensor) -> Tensor:
        if self._fused_weight is None:
            self.prepare_fused_weights()
        assert self._fused_weight is not None
        swiglu_out = fused_swiglu_gemm(x, self._fused_weight)
        return self.down_proj(swiglu_out)


class BaselineSwiGLUMLP(nn.Module):
    """Standard unfused SwiGLU MLP for comparison."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swiglu = nn.functional.silu(gate) * up
        return self.down_proj(swiglu)


# %%
# TritonBench Wrapper
# -------------------


# %%
def fused_swiglu_tritonbench(tb_op: object, x: Tensor) -> Callable:
    """Tritonbench-compatible wrapper."""
    hidden_size = x.shape[-1]
    # pyrefly: ignore [missing-attribute]
    intermediate_size = tb_op.intermediate_size

    mlp = FusedSwiGLUMLP(hidden_size, intermediate_size).to(x.device).to(x.dtype)

    # pyrefly: ignore [missing-attribute]
    baseline = tb_op.baseline_op
    mlp.gate_proj.weight.data.copy_(baseline.gate_proj.weight.data)
    mlp.up_proj.weight.data.copy_(baseline.up_proj.weight.data)
    mlp.down_proj.weight.data.copy_(baseline.down_proj.weight.data)

    mlp.prepare_fused_weights()
    return lambda: mlp(x)


# %%
# Verification
# ------------


# %%
def check_fused_swiglu(m: int, k: int, n: int) -> None:
    """Test fused SwiGLU against unfused baseline."""
    X = torch.randn(m, k, dtype=HALF_DTYPE, device=DEVICE)

    # Create concatenated weight matrix
    W_gate = torch.randn(k, n, dtype=HALF_DTYPE, device=DEVICE)
    W_up = torch.randn(k, n, dtype=HALF_DTYPE, device=DEVICE)
    W_gate_up = torch.cat([W_gate, W_up], dim=1).contiguous()

    def fused_fn(X: Tensor, W: Tensor) -> Tensor:
        return fused_swiglu_gemm(X, W)

    def baseline_fn(X: Tensor, W: Tensor) -> Tensor:
        n = W.shape[1] // 2
        gate_out = torch.matmul(X, W[:, :n])
        up_out = torch.matmul(X, W[:, n:])
        return nn.functional.silu(gate_out) * up_out

    run_example(fused_fn, baseline_fn, (X, W_gate_up), atol=1.0, rtol=2e-1)
    print(f"Fused SwiGLU: M={m}, K={k}, N={n} PASSED")


def check_fused_mlp(
    batch_size: int, seq_len: int, hidden_size: int, intermediate_size: int
) -> None:
    """Test fused MLP module against baseline."""
    x = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE, dtype=HALF_DTYPE)

    fused_mlp = FusedSwiGLUMLP(hidden_size, intermediate_size).to(DEVICE).to(HALF_DTYPE)
    baseline_mlp = (
        BaselineSwiGLUMLP(hidden_size, intermediate_size).to(DEVICE).to(HALF_DTYPE)
    )

    # Copy weights
    baseline_mlp.gate_proj.weight.data.copy_(fused_mlp.gate_proj.weight.data)
    baseline_mlp.up_proj.weight.data.copy_(fused_mlp.up_proj.weight.data)
    baseline_mlp.down_proj.weight.data.copy_(fused_mlp.down_proj.weight.data)

    fused_mlp.prepare_fused_weights()

    run_example(
        lambda x: fused_mlp(x),
        lambda x: baseline_mlp(x),
        (x,),
        atol=1.0,
        rtol=2e-1,
    )
    print(
        f"Fused SwiGLU MLP: B={batch_size}, T={seq_len}, "
        f"H={hidden_size}, I={intermediate_size} PASSED"
    )


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """Run correctness checks."""
    print("Testing Fused SwiGLU GEMM...")
    check_fused_swiglu(64, 4096, 11008)
    check_fused_swiglu(128, 4096, 11008)

    print("\nTesting Fused SwiGLU MLP Module...")
    check_fused_mlp(4, 2048, 4096, 11008)


# %%
# Run Example
# -----------

# %%
if __name__ == "__main__":
    main()
