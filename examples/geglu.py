"""
Helion GEGLU MLP Example
========================
This example demonstrates a Helion kernel implementation of GEGLU MLP (GELU-Gated Linear Unit MLP).
GEGLU MLP is a common pattern in transformer architectures like Gemma, where:

1. Input x is projected through gate_proj and up_proj

2. GEGLU operation: GELU(gate_proj(x)) * up_proj(x)

3. Result is projected through down_proj

GELU uses tanh approximation: 0.5 * a * (1 + tanh(sqrt(2/π) * (a + 0.044715 * a³)))

Based on liger_kernel's GEGLU implementation used in Gemma and other gated feedforward networks.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
# GEGLU Kernel
# ------------


# %%
@helion.kernel()
def geglu(a: Tensor, b: Tensor) -> Tensor:
    """
    Performs GEGLU operation: GELU(a) * b using tanh approximation for GELU.

    GELU(a) = 0.5 * a * (1 + tanh(sqrt(2/π) * (a + 0.044715 * a³)))
    GEGLU(a, b) = GELU(a) * b

    Args:
        a (Tensor): Input tensor for GELU activation of any shape.
        b (Tensor): Input tensor for multiplication, must have same shape as a.

    Returns:
        Tensor: Result of GEGLU operation with same shape as inputs.
    """
    # Ensure tensors have the same shape
    assert a.shape == b.shape, (
        f"Input tensors must have same shape, got {a.shape} != {b.shape}"
    )

    # Create output tensor
    out = torch.empty_like(a, dtype=torch.promote_types(a.dtype, b.dtype))

    # Get the total number of elements and process in tiles
    total_elements = a.numel()

    # Flatten tensors for easier processing
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    out_flat = out.view(-1)

    # Process elements in tiles
    for tile_idx in hl.tile(total_elements):
        # Load input values and convert to float32 for computation
        a_vals = a_flat[tile_idx].to(torch.float32)
        b_vals = b_flat[tile_idx]

        # GELU computation using tanh approximation
        # Constants for tanh approximation
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / π)

        # Compute a cubed
        a_cubed = a_vals * a_vals * a_vals

        # Compute tanh argument: sqrt(2/π) * (a + 0.044715 * a^3)
        tanh_arg = sqrt_2_over_pi * (a_vals + 0.044715 * a_cubed)

        # Compute tanh and GELU
        tanh_result = torch.tanh(tanh_arg)
        gelu_a = 0.5 * a_vals * (1.0 + tanh_result)

        # GEGLU: GELU(a) * b
        result = gelu_a.to(b_vals.dtype) * b_vals

        # Store result
        out_flat[tile_idx] = result

    return out


# %%
# GEGLU MLP Module (matches liger_kernel structure)
# -------------------------------------------------


# %%
@dataclass
class Config:
    """
    Configuration class for MLP.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "gelu_pytorch_tanh",
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act


class HelionGEGLUMLP(nn.Module):
    """
    Helion implementation of GEGLU MLP matching liger_kernel.LigerGEGLUMLP structure.

    This implements the complete MLP used in transformer architectures:
    down_proj(GEGLU(gate_proj(x), up_proj(x)))
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: down_proj(GEGLU(gate_proj(x), up_proj(x)))
        """
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        geglu_output = geglu(gate_output, up_output)
        return self.down_proj(geglu_output)


# %%
# Verification Function
# ---------------------


# %%
def check_geglu_kernel(shape: tuple[int, ...]) -> None:
    """
    Verify the GEGLU kernel implementation against PyTorch's baseline.

    Args:
        shape: Shape of the input tensors to test.
    """
    # Create test tensors
    a = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    b = torch.randn(shape, device=DEVICE, dtype=torch.float16)

    def baseline_geglu(a: Tensor, b: Tensor) -> Tensor:
        """
        PyTorch baseline implementation using tanh approximation GELU.
        This matches the liger_kernel implementation.
        """
        return nn.functional.gelu(a, approximate="tanh").to(b.dtype) * b

    run_example(geglu, baseline_geglu, (a, b))


class BaselineMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: down_proj(GEGLU(gate_proj(x), up_proj(x)))
        """
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        geglu_output = (
            nn.functional.gelu(gate_output, approximate="tanh").to(up_output.dtype)
            * up_output
        )
        return self.down_proj(geglu_output)


def check_geglu_mlp(
    batch_size: int, seq_len: int, hidden_size: int, intermediate_size: int
) -> None:
    """
    Verify the GEGLU MLP implementation against PyTorch's baseline MLP.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size
    """

    config = Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="gelu_pytorch_tanh",
    )

    # Create test input
    x = torch.randn(
        batch_size, seq_len, hidden_size, device=DEVICE, dtype=torch.float16
    )

    # Create models
    helion_mlp = HelionGEGLUMLP(config).to(DEVICE).to(torch.float16)
    baseline_mlp = BaselineMLP(config).to(DEVICE).to(torch.float16)

    # Copy weights to ensure same parameters
    baseline_mlp.gate_proj.weight.data = helion_mlp.gate_proj.weight.data.clone()
    baseline_mlp.up_proj.weight.data = helion_mlp.up_proj.weight.data.clone()
    baseline_mlp.down_proj.weight.data = helion_mlp.down_proj.weight.data.clone()

    # Run comparison
    run_example(lambda x: helion_mlp(x), lambda x: baseline_mlp(x), (x,))


# %%
# Tritonbench Integration
# -----------------------


# %%
def geglu_tritonbench(tb_op: object, x: Tensor) -> Callable:
    """
    Wrapper for tritonbench that matches its interface.
    Copies weights from tritonbench operator models to ensure fair comparison.

    Args:
        tb_op: TritonBench operator instance with baseline_model and liger_model
        x (Tensor): Input tensor for the GEGLU MLP.

    Returns:
        Callable: A callable that runs the GEGLU kernel with copied weights.
    """

    # Extract configuration from tritonbench operator
    config = Config(
        hidden_size=tb_op.hidden_size,  # pyright: ignore[reportAttributeAccessIssue]
        intermediate_size=tb_op.intermediate_size,  # pyright: ignore[reportAttributeAccessIssue]
        hidden_act=tb_op.hidden_act,  # pyright: ignore[reportAttributeAccessIssue]
    )

    # Create Helion model
    helion_mlp = HelionGEGLUMLP(config).to(x.device).to(x.dtype)

    # Copy weights from tritonbench baseline model (LlamaMLP) to ensure fairness
    # LlamaMLP has: gate_proj, up_proj, down_proj (same structure as our HelionGEGLUMLP)
    baseline_model = tb_op.baseline_model  # pyright: ignore[reportAttributeAccessIssue]

    # Copy gate projection weights
    helion_mlp.gate_proj.weight.data.copy_(baseline_model.gate_proj.weight.data)

    # Copy up projection weights
    helion_mlp.up_proj.weight.data.copy_(baseline_model.up_proj.weight.data)

    # Copy down projection weights
    helion_mlp.down_proj.weight.data.copy_(baseline_model.down_proj.weight.data)

    return lambda: helion_mlp(x)


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the GEGLU kernel and MLP verification.
    Tests various shapes including typical transformer sizes.
    """
    print("Testing GEGLU kernel...")

    # Test GEGLU kernel with different shapes
    kernel_test_shapes = [(8, 2048, 4096), (8, 4096, 8192)]

    for shape in kernel_test_shapes:
        print(f"Testing GEGLU kernel shape: {shape}")
        check_geglu_kernel(shape)
        print(f"✓ GEGLU kernel shape {shape} passed")

    print("\nTesting GEGLU MLP...")

    # Test GEGLU MLP with transformer-typical sizes
    mlp_test_configs = [
        (8, 2048, 4096, 11008),
        (8, 4096, 8192, 11008),
    ]

    for batch_size, seq_len, hidden_size, intermediate_size in mlp_test_configs:
        print(
            f"Testing GEGLU MLP: B={batch_size}, T={seq_len}, H={hidden_size}, I={intermediate_size}"
        )
        check_geglu_mlp(batch_size, seq_len, hidden_size, intermediate_size)
        print("✓ GEGLU MLP config passed")


# %%
if __name__ == "__main__":
    main()
