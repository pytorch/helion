"""
Helion SwiGLU MLP Example
=========================
This example demonstrates a Helion kernel implementation of SwiGLU MLP (Swish-Gated Linear Unit MLP).
SwiGLU MLP is a common pattern in transformer architectures like LLaMA, where:

1. Input x is projected through gate_proj and up_proj

2. SwiGLU operation: SiLU(gate_proj(x)) * up_proj(x)

3. Result is projected through down_proj

SiLU (Swish) activation: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

Based on liger_kernel's SwiGLU implementation used in LLaMA and other gated feedforward networks.
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
    from typing import Any


# %%
# SwiGLU Kernel
# -------------


# %%
@helion.kernel()
def swiglu_fwd(a: Tensor, b: Tensor) -> Tensor:
    """
    Performs SwiGLU operation: SiLU(a) * b where SiLU is the Swish activation.

    SiLU(a) = a * sigmoid(a) = a / (1 + exp(-a))
    SwiGLU(a, b) = SiLU(a) * b

    Args:
        a (Tensor): Input tensor for SiLU activation of any shape.
        b (Tensor): Input tensor for multiplication, must have same shape as a.

    Returns:
        Tensor: Result of SwiGLU operation with same shape as inputs.
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

        # SiLU computation: x * sigmoid(x)
        sigmoid_a = torch.sigmoid(a_vals)
        silu_a = a_vals * sigmoid_a

        # SwiGLU: SiLU(a) * b
        result = silu_a.to(b_vals.dtype) * b_vals

        # Store result
        out_flat[tile_idx] = result

    return out


@helion.kernel()
def swiglu_bwd(gout: Tensor, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
    """
    Implement the backward formula for swiglu.
    """
    dx1 = torch.empty_like(x1)
    dx2 = torch.empty_like(x2)

    gout_flat = gout.view(-1)
    x1_flat = x1.view(-1)
    x2_flat = x2.view(-1)
    dx1_flat = dx1.view(-1)
    dx2_flat = dx2.view(-1)

    for tile in hl.tile(x1.numel()):
        x1_vals = x1_flat[tile].to(torch.float32)
        gout_vals = gout_flat[tile].to(torch.float32)

        # compute dx2
        dx2_vals = x1_vals * torch.sigmoid(x1_vals) * gout_vals
        dx2_flat[tile] = dx2_vals.to(x2.dtype)

        # compute dx1
        x2_vals = x2_flat[tile].to(torch.float32)
        x1_exp = torch.exp(x1_vals)
        x1_exp_plus1 = x1_exp + 1
        dextra = x1_exp / x1_exp_plus1 + x1_vals * x1_exp / x1_exp_plus1 / x1_exp_plus1
        dx1_vals = gout_vals * x2_vals * dextra
        dx1_flat[tile] = dx1_vals.to(x1.dtype)

    return dx1, dx2


class SwigluFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401
        x1: Tensor,
        x2: Tensor,
    ) -> Tensor:
        out = swiglu_fwd(x1, x2)
        ctx.save_for_backward(x1, x2)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        grad_out: Tensor,
    ) -> tuple[Tensor, Tensor]:
        x1, x2 = ctx.saved_tensors
        dx1, dx2 = swiglu_bwd(grad_out, x1, x2)
        return dx1, dx2


def swiglu(a: Tensor, b: Tensor) -> Tensor:
    """swiglu with forward + backward support."""
    return SwigluFunction.apply(a, b)  # type: ignore[no-any-return]


# %%
# SwiGLU MLP Module (matches liger_kernel structure)
# --------------------------------------------------


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
        hidden_act: str = "silu",
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act


class HelionSwiGLUMLP(nn.Module):
    """
    Helion implementation of SwiGLU MLP matching liger_kernel.LigerSwiGLUMLP structure.

    This implements the complete MLP used in transformer architectures:
    down_proj(SwiGLU(gate_proj(x), up_proj(x)))
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Validate activation function
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: down_proj(SwiGLU(gate_proj(x), up_proj(x)))
        """
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        swiglu_output = swiglu(gate_output, up_output)
        return self.down_proj(swiglu_output)


# %%
# Verification Function
# ---------------------


# %%
def check_swiglu_kernel(shape: tuple[int, ...]) -> None:
    """
    Verify the SwiGLU kernel implementation against PyTorch's baseline.

    Args:
        shape: Shape of the input tensors to test.
    """
    # Create test tensors
    a = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    b = torch.randn(shape, device=DEVICE, dtype=torch.float16)

    def baseline_swiglu(a: Tensor, b: Tensor) -> Tensor:
        """
        PyTorch baseline implementation using SiLU activation.
        This matches the liger_kernel implementation.
        """
        return nn.functional.silu(a).to(b.dtype) * b

    run_example(swiglu, baseline_swiglu, (a, b))


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
        Forward pass: down_proj(SwiGLU(gate_proj(x), up_proj(x)))
        """
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        swiglu_output = nn.functional.silu(gate_output).to(up_output.dtype) * up_output
        return self.down_proj(swiglu_output)


def check_swiglu_mlp(
    batch_size: int, seq_len: int, hidden_size: int, intermediate_size: int
) -> None:
    """
    Verify the SwiGLU MLP implementation against PyTorch's baseline MLP.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size
    """

    config = Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
    )

    # Create test input
    x = torch.randn(
        batch_size, seq_len, hidden_size, device=DEVICE, dtype=torch.float16
    )

    # Create models
    helion_mlp = HelionSwiGLUMLP(config).to(x.device).to(torch.float16)
    baseline_mlp = BaselineMLP(config).to(x.device).to(torch.float16)

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
def swiglu_tritonbench(tb_op: object, x: Tensor) -> Callable:
    """
    Wrapper for tritonbench that matches its interface.

    Args:
        tb_op: TritonBench operator instance
        x (Tensor): Input tensor for the MLP.

    Returns:
        Callable: A callable that runs the SwiGLU MLP.
    """

    # Extract configuration from tritonbench operator
    config = Config(
        hidden_size=tb_op.hidden_size,  # pyright: ignore[reportAttributeAccessIssue]
        intermediate_size=tb_op.intermediate_size,  # pyright: ignore[reportAttributeAccessIssue]
        hidden_act=tb_op.hidden_act,  # pyright: ignore[reportAttributeAccessIssue]
    )

    # Create Helion model
    helion_mlp = HelionSwiGLUMLP(config).to(x.device).to(x.dtype)

    # Copy weights from tritonbench baseline model (LlamaMLP) to ensure fairness
    # LlamaMLP has: gate_proj, up_proj, down_proj (same structure as our HelionGEGLUMLP)
    baseline_model = tb_op.baseline_op  # pyright: ignore[reportAttributeAccessIssue]

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
    Main entry point that runs the SwiGLU kernel and MLP verification.
    Tests various shapes including typical transformer sizes.
    """
    print("Testing SwiGLU kernel...")

    # Test SwiGLU kernel with different shapes
    kernel_test_shapes = [(4, 8192, 4096), (8, 8192, 4096)]

    for shape in kernel_test_shapes:
        print(f"Testing SwiGLU kernel shape: {shape}")
        check_swiglu_kernel(shape)
        print(f"✓ SwiGLU kernel shape {shape} passed")

    print("\nTesting SwiGLU MLP...")

    # Test SwiGLU MLP with transformer-typical sizes
    mlp_test_configs = [
        (4, 8192, 4096, 11008),
        (8, 8192, 4096, 11008),
    ]

    for batch_size, seq_len, hidden_size, intermediate_size in mlp_test_configs:
        print(
            f"Testing SwiGLU MLP: B={batch_size}, T={seq_len}, H={hidden_size}, I={intermediate_size}"
        )
        check_swiglu_mlp(batch_size, seq_len, hidden_size, intermediate_size)
        print("✓ SwiGLU MLP config passed")


# %%
if __name__ == "__main__":
    main()
