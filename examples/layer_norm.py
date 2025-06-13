from __future__ import annotations

import torch

import helion
import helion.language as hl

"""
    NOTE: layer_norm_fwd_ideal does not work! I am keeping this around as a reference
    to what I believed should have worked in Helion when I first began without debugging.

    The user experience should be pushed this direction
"""
@helion.kernel(static_shapes=True)
def layer_norm_fwd_ideal(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Layer normalization forward pass.
    
    Args:
        x: Input tensor of shape [batch_size, hidden_size]
        weight: Scale parameter of shape [hidden_size]
        bias: Bias parameter of shape [hidden_size]
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized tensor of shape [batch_size, hidden_size]
    """
    m = x.size(0)
    out = torch.empty_like(x)

    for tile_b in hl.tile(m):
        row = x[tile_b]
        mean, var = torch.var_mean(row)
        
        layer_norm_out = (row - mean) / torch.sqrt(var + eps)
        layer_norm_out = layer_norm_out * weight + bias
        out[tile_b, :] = layer_norm_out

    return out

@helion.kernel(static_shapes=True, use_default_config=True)
def layer_norm_fwd(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {m}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {m}"
    out = torch.empty(
        [m, n], dtype=torch.float16, device=x.device
    )

    eps = 1e-5

    for tile_m in hl.tile(m):
        # acc = x[tile_m, :].to(torch.float32) works! We should not have to do this cast
        acc = x[tile_m, :]
        
        var, mean = torch.var_mean(acc, dim=-1, keepdim=True, correction=0)

        normalized = (acc - mean) * torch.rsqrt(var + eps)
        acc = normalized * (weight[:].to(torch.float32)) + (bias[:].to(torch.float32))
    
        out[tile_m, :] = acc
    return out


def check(batch_size: int, hidden_size: int) -> None:
    from triton.testing import do_bench
    
    # Create random input tensors
    x = torch.randn([batch_size, hidden_size], device="cuda", dtype=torch.float16)
    weight = torch.randn([hidden_size], device="cuda", dtype=torch.float16)
    bias = torch.randn([hidden_size], device="cuda", dtype=torch.float16)
    
    # Run Helion kernel
    result = layer_norm_fwd(x, weight, bias)
    
    # # Run PyTorch layer norm for comparison
    torch_result = torch.nn.functional.layer_norm(
        x, [hidden_size], weight, bias, eps=1e-5
    )
    
    # # Check correctness
    torch.testing.assert_close(result, torch_result, rtol=1e-2, atol=1e-1)
    
    # Benchmark Helion implementation
    helion_sec = do_bench(lambda: layer_norm_fwd(x, weight, bias))
    
    # Benchmark PyTorch implementation
    torch_sec = do_bench(lambda: torch.nn.functional.layer_norm(
        x, [hidden_size], weight, bias, eps=1e-5
    ))
    
    print(
        f"Helion time: {helion_sec:.4f}ms, torch time: {torch_sec:.4f}, speedup: {torch_sec / helion_sec:.2f}x"
    )


def main() -> None:
    # Test with different sizes
    print("Testing batch_size=128, hidden_size=768")
    check(128, 768)
    
    print("\nTesting batch_size=32, hidden_size=1024")
    check(32, 1024)


if __name__ == "__main__":
    main()
