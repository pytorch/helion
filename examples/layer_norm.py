"""
Helion Layer Normalization Forward Example
==========================================
This example demonstrates a Helion kernel implementation of 1D layer normalization
using FP16 inputs and compares it against PyTorch's built-in layer_norm function.
"""

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel
def layer_norm_fwd(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Performs 1D layer normalization on the input tensor using Helion.
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim], expected to be FP16.
        normalized_shape (list[int]): List containing the dimension to normalize over (should be length 1).
        weight (torch.Tensor): Learnable scale parameter of shape [dim].
        bias (Optional[torch.Tensor]): Learnable bias parameter of shape [dim].
        eps (float, optional): Small value added to variance for numerical stability. Default is 1e-5.
    Returns:
        torch.Tensor: The layer-normalized output tensor of shape [batch_size, dim], in FP16.
    """
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {m}"
    if bias is not None:
        assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {m}"
    assert len(normalized_shape) == 1, (
        "Helion layer norm only supports 1D layer norm currently"
    )
    assert normalized_shape[0] == n, (
        f"normalized shape mismatch {normalized_shape[0]} != {n}"
    )
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        var, mean = torch.var_mean(acc, dim=-1, keepdim=True, correction=0)
        normalized = (acc - mean) * torch.rsqrt(var + eps)
        if bias is not None:
            acc = normalized * (weight[:].to(torch.float32)) + (
                bias[:].to(torch.float32)
            )
        else:
            acc = normalized * (weight[:].to(torch.float32))
        out[tile_m, :] = acc.to(x.dtype)
    return out


# %%
def main() -> None:
    """
    Main execution function for the layer normalization example.
    - Generates random input, weight, and bias tensors.
    - Runs the Helion layer normalization kernel and compares its output to PyTorch's
      built-in layer_norm function using the run_example utility.
    - Prints comparison results and checks for correctness within specified tolerances.
    """
    batch_size = 32
    dim = 64
    device = "cuda"
    x = torch.randn([batch_size, dim], device=device, dtype=torch.float16)
    weight = torch.randn([dim], device=device, dtype=torch.float16)
    bias = torch.randn([dim], device=device, dtype=torch.float16)
    eps = 1e-4
    for b in [bias, None]:
        run_example(
            layer_norm_fwd,
            torch.nn.functional.layer_norm,
            (x, [dim], weight, b, eps),
            kernel_name="helion",
            baseline_name="torch",
            rtol=1e-3,
            atol=1e-3,
        )


# %%
if __name__ == "__main__":
    main()
