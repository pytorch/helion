from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl

# TritonBench configuration
TRITONBENCH_ARGS = {"primals_1": None, "primals_2": None, "primals_3": None}


@helion.kernel()
def welford_layer_norm(
    weight: torch.Tensor,
    bias: torch.Tensor,
    input: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Welford algorithm for computing layer norm.
    
    Args:
        weight: Scale parameter (gamma) with shape [D]
        bias: Shift parameter (beta) with shape [D]  
        input: Input tensor with shape [S, D]
        eps: Small value to avoid division by zero
    
    Returns:
        Tuple of (output tensor, mean, inv_std)
    """
    S, D = input.shape
    
    # Output tensors
    out = torch.empty_like(input)
    mean_out = torch.empty((S, 1), dtype=torch.float32, device=input.device)
    inv_std_out = torch.empty((S, 1), dtype=torch.float32, device=input.device)
    
    # Process rows in tiles
    for tile_s in hl.tile(S):
        # Compute mean using simple reduction first
        row_sum = hl.zeros([tile_s], dtype=torch.float32)
        
        for tile_d in hl.tile(D):
            x = input[tile_s, tile_d].to(torch.float32)
            row_sum = row_sum + x.sum(dim=-1)
        
        mean = row_sum / D
        
        # Store mean
        mean_out[tile_s, 0] = mean
        
        # Compute variance using the mean
        var_sum = hl.zeros([tile_s], dtype=torch.float32)
        
        for tile_d in hl.tile(D):
            x = input[tile_s, tile_d].to(torch.float32)
            diff = x - mean[:, None]
            var_sum = var_sum + (diff * diff).sum(dim=-1)
        
        variance = var_sum / D
        
        # Compute inverse standard deviation
        inv_std = torch.rsqrt(variance + eps)
        
        # Store inv_std
        inv_std_out[tile_s, 0] = inv_std
        
        # Apply normalization
        for tile_d in hl.tile(D):
            x_orig = input[tile_s, tile_d]
            x_normalized = (x_orig - mean[:, None]) * inv_std[:, None]
            
            # Apply scale and bias
            out[tile_s, tile_d] = x_normalized * weight[tile_d] + bias[tile_d]
    
    return out, mean_out, inv_std_out


def welford_tritonbench(primals_1, primals_2, primals_3):
    """
    Wrapper for tritonbench that matches the expected interface.
    
    Args:
        primals_1: weight (gamma) parameter
        primals_2: bias (beta) parameter
        primals_3: input tensor
    
    Returns:
        Tuple of (output, input, mean, inv_std) to match tritonbench interface
    """
    # Run the welford layer norm kernel
    output, mean, inv_std = welford_layer_norm(primals_1, primals_2, primals_3)
    
    return (output, primals_3, mean, inv_std)


def reference_layer_norm(weight, bias, input, eps=1e-5):
    """PyTorch reference implementation for layer normalization."""
    return torch.nn.functional.layer_norm(input, (input.shape[-1],), weight, bias, eps)


def check(S: int, D: int) -> None:
    # Create input tensors
    weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    input = torch.randn([S, D], device="cuda", dtype=torch.bfloat16)
    
    # Run comparison - just compare the output tensor
    run_example(
        {"helion": lambda w, b, x: welford_layer_norm(w, b, x)[0]},
        lambda w, b, x: reference_layer_norm(w, b, x),
        (weight, bias, input),
    )


def main() -> None:
    # Test with various shapes
    check(262144, 1024)
    check(262144, 2048)
    check(512, 768)
    check(1024, 1024)


if __name__ == "__main__":
    main()