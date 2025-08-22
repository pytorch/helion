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
def layer_norm_fwd_with_bias(
    x: torch.Tensor,
    nomralized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Performs 1D layer normalization with bias on the input tensor using Helion.
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim], expected to be FP16.
        nomralized_shape (list[int]): List containing the dimension to normalize over (should be length 1).
        weight (torch.Tensor): Learnable scale parameter of shape [dim].
        bias (torch.Tensor): Learnable bias parameter of shape [dim].
        eps (float, optional): Small value added to variance for numerical stability. Default is 1e-5.
    Returns:
        torch.Tensor: The layer-normalized output tensor of shape [batch_size, dim], in FP16.
    """
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {m}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {m}"
    assert len(nomralized_shape) == 1, (
        "Helion layer norm only supports 1D layer norm currently"
    )
    assert nomralized_shape[0] == n, (
        f"normalized shape mismatch {nomralized_shape[0]} != {n}"
    )
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        var, mean = torch.var_mean(acc, dim=-1, keepdim=True, correction=0)
        normalized = (acc - mean) * torch.rsqrt(var + eps)
        acc = normalized * (weight[:].to(torch.float32)) + (bias[:].to(torch.float32))
        out[tile_m, :] = acc
    return out


@helion.kernel
def layer_norm_fwd_no_bias(
    x: torch.Tensor,
    nomralized_shape: list[int],
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Performs 1D layer normalization without bias on the input tensor using Helion.
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim], expected to be FP16.
        nomralized_shape (list[int]): List containing the dimension to normalize over (should be length 1).
        weight (torch.Tensor): Learnable scale parameter of shape [dim].
        eps (float, optional): Small value added to variance for numerical stability. Default is 1e-5.
    Returns:
        torch.Tensor: The layer-normalized output tensor of shape [batch_size, dim], in FP16.
    """
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {m}"
    assert len(nomralized_shape) == 1, (
        "Helion layer norm only supports 1D layer norm currently"
    )
    assert nomralized_shape[0] == n, (
        f"normalized shape mismatch {nomralized_shape[0]} != {n}"
    )
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        var, mean = torch.var_mean(acc, dim=-1, keepdim=True, correction=0)
        normalized = (acc - mean) * torch.rsqrt(var + eps)
        acc = normalized * (weight[:].to(torch.float32))
        out[tile_m, :] = acc
    return out


def layer_norm_fwd_tritonbench(
    x: torch.Tensor,
    nomralized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Wrapper function that dispatches to the appropriate layer normalization kernel.
    Compatible with tritonbench which may pass None for bias.
    """
    if bias is None:
        return layer_norm_fwd_no_bias(x, nomralized_shape, weight, eps)
    else:
        return layer_norm_fwd_with_bias(x, nomralized_shape, weight, bias, eps)


# %%
def main() -> None:
    """
    Main execution function for the layer normalization example.
    - Generates random input, weight, and bias tensors.
    - Runs the Helion layer normalization kernel and compares its output to PyTorch's
      built-in layer_norm function using the run_example utility.
    - Tests both with bias and without bias (no-bias mode).
    - Prints comparison results and checks for correctness within specified tolerances.
    """
    batch_size = 32
    dim = 64
    device = "cuda"
    x = torch.randn([batch_size, dim], device=device, dtype=torch.float16)
    weight = torch.randn([dim], device=device, dtype=torch.float16)
    bias = torch.randn([dim], device=device, dtype=torch.float16)
    eps = 1e-4
    
    # Test with bias
    print("Testing layer_norm WITH bias:")
    run_example(
        layer_norm_fwd_with_bias,
        torch.nn.functional.layer_norm,
        (x, [dim], weight, bias, eps),
        kernel_name="helion_with_bias",
        baseline_name="torch",
        rtol=1e-3,
        atol=1e-3,
    )
    
    # Test without bias (no-bias mode)
    print("\nTesting layer_norm WITHOUT bias (no-bias mode):")
    run_example(
        layer_norm_fwd_no_bias,
        lambda x, shape, w, e: torch.nn.functional.layer_norm(x, shape, w, None, e),
        (x, [dim], weight, eps),
        kernel_name="helion_no_bias",
        baseline_name="torch",
        rtol=1e-3,
        atol=1e-3,
    )
    
    # Test wrapper function with bias
    print("\nTesting wrapper function WITH bias:")
    result_with_bias = layer_norm_fwd_tritonbench(x, [dim], weight, bias, eps)
    expected_with_bias = torch.nn.functional.layer_norm(x, [dim], weight, bias, eps)
    print(f"  Wrapper with bias matches torch: {torch.allclose(result_with_bias, expected_with_bias, rtol=1e-3, atol=1e-3)}")
    
    # Test wrapper function without bias
    print("\nTesting wrapper function WITHOUT bias:")
    result_no_bias = layer_norm_fwd_tritonbench(x, [dim], weight, None, eps)
    expected_no_bias = torch.nn.functional.layer_norm(x, [dim], weight, None, eps)
    print(f"  Wrapper without bias matches torch: {torch.allclose(result_no_bias, expected_no_bias, rtol=1e-3, atol=1e-3)}")


# %%
if __name__ == "__main__":
    main()
