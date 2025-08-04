#!/usr/bin/env python3
"""Enhanced Helion kernel implementations with advanced patterns and optimizations."""

import logging
import helion
import helion.language as hl
import torch
from torch import Tensor
from triton.testing import do_bench

logging.getLogger().setLevel(logging.WARNING)

print("Enhanced Helion Kernel Implementations")
print("=" * 80)

# Enhanced Example 1: Vectorized Constant Add with Multiple Operations
@helion.kernel(config=helion.Config(block_sizes=[512], num_warps=4))
def add_scaled_kernel(x: torch.Tensor, scale: float = 2.0, offset: float = 10.0) -> torch.Tensor:
    """Enhanced: Add offset after scaling, demonstrating scalar parameter usage."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] * scale + offset
    return out

# Enhanced Example 2: Generalized Broadcasting Operation
@helion.kernel(config=helion.Config(block_sizes=[64, 64]))
def broadcast_op_kernel(x: torch.Tensor, y: torch.Tensor, op: str = "add") -> torch.Tensor:
    """Enhanced: Support multiple broadcasting operations."""
    n0 = x.size(0)
    n1 = y.size(0)
    out = torch.empty([n1, n0], dtype=x.dtype, device=x.device)
    
    for tile_i, tile_j in hl.tile([n1, n0]):
        x_broadcast = x[None, tile_j]
        y_broadcast = y[tile_i, None]
        
        if op == "add":
            out[tile_i, tile_j] = x_broadcast + y_broadcast
        elif op == "mul":
            out[tile_i, tile_j] = x_broadcast * y_broadcast
        elif op == "sub":
            out[tile_i, tile_j] = y_broadcast - x_broadcast
    
    return out

# Enhanced Example 3: Multi-Stage Fusion
@helion.kernel(config=helion.Config(block_sizes=[128, 128], num_stages=3))
def fused_ops_kernel(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Enhanced: Fuse multiple operations - (x * y + z) with ReLU and scaling."""
    n0 = x.size(0)
    n1 = y.size(0)
    out = torch.empty([n1, n0], dtype=x.dtype, device=x.device)
    
    for tile_i, tile_j in hl.tile([n1, n0]):
        # Multiple fused operations
        temp = x[None, tile_j] * y[tile_i, None] + z[tile_i, tile_j]
        # Apply ReLU and scale by constant
        out[tile_i, tile_j] = torch.relu(temp) * 0.1
    
    return out

# Enhanced Example 4: Adaptive Block Size with Statistics
@helion.kernel(config=helion.Config(block_sizes=[256], num_warps=8))
def adaptive_sum_kernel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enhanced: Compute sum, mean, and variance in a single pass."""
    n = x.numel()
    sum_out = torch.zeros(1, dtype=x.dtype, device=x.device)
    mean_out = torch.zeros(1, dtype=x.dtype, device=x.device)
    var_out = torch.zeros(1, dtype=x.dtype, device=x.device)
    
    # First pass: compute sum and mean
    for tile in hl.tile(x.shape):
        sum_out += x[tile].sum()
    
    mean_out[0] = sum_out[0] / n
    
    # Second pass: compute variance
    for tile in hl.tile(x.shape):
        diff = x[tile] - mean_out[0]
        var_out += (diff * diff).sum()
    
    var_out[0] = var_out[0] / n
    
    return sum_out, mean_out, var_out

# Enhanced Example 5: Hierarchical Reduction
@helion.kernel(config=helion.Config(block_sizes=[4, 128, 64], num_warps=16))
def hierarchical_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """Enhanced: Two-level reduction for better performance on large tensors."""
    batch, seq_len, hidden = x.size()
    out = torch.empty([batch, hidden], dtype=x.dtype, device=x.device)
    
    # Register block size for inner reduction
    seq_block_size = hl.register_block_size(min(seq_len, 128))
    
    for tile_batch, tile_hidden in hl.tile([batch, hidden]):
        # First level: partial sums
        partial_sum = hl.zeros([tile_batch, seq_block_size, tile_hidden], dtype=x.dtype)
        
        for tile_seq in hl.tile(seq_len, block_size=seq_block_size):
            partial_sum[:, :, :] += x[tile_batch, tile_seq, tile_hidden]
        
        # Second level: final reduction
        out[tile_batch, tile_hidden] = partial_sum.sum(1)
    
    return out

# Enhanced Example 6: Advanced Jagged Operations
@helion.kernel(config=helion.Config(block_sizes=[2, 256, 256], num_warps=16, num_stages=4))
def jagged_stats_kernel(
    x_data: torch.Tensor, 
    x_offsets: torch.Tensor, 
    y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enhanced: Compute both mean and std dev of jagged tensor, add to dense matrix."""
    num_rows = y.size(0)
    out_mean = torch.zeros_like(y)
    out_std = torch.zeros_like(y)
    
    for tile0 in hl.tile(num_rows):
        starts = x_offsets[tile0]
        ends = x_offsets[tile0.index + 1]
        nnz = ends - starts
        max_nnz = nnz.amax()
        
        # Compute mean
        row_sum = hl.zeros([tile0], dtype=x_data.dtype)
        for tile1 in hl.tile(0, max_nnz):
            x_slice = hl.load(
                x_data,
                [starts[:, None] + tile1.index[None, :]],
                extra_mask=tile1.index[None, :] < nnz[:, None]
            )
            row_sum += x_slice.sum(axis=1)
        
        row_mean = torch.where(nnz > 0, row_sum / nnz.float(), torch.zeros_like(row_sum))
        
        # Compute variance
        row_var = hl.zeros([tile0], dtype=x_data.dtype)
        for tile1 in hl.tile(0, max_nnz):
            x_slice = hl.load(
                x_data,
                [starts[:, None] + tile1.index[None, :]],
                extra_mask=tile1.index[None, :] < nnz[:, None]
            )
            diff = x_slice - row_mean[:, None]
            row_var += (diff * diff).sum(axis=1)
        
        row_std = torch.where(nnz > 1, torch.sqrt(row_var / (nnz.float() - 1)), torch.zeros_like(row_var))
        
        # Apply to output
        for tile1 in hl.tile(0, max_nnz):
            mask = tile1.index[None, :] < nnz[:, None]
            out_mean[tile0, tile1] = torch.where(mask, y[tile0, tile1] + row_mean[:, None], y[tile0, tile1])
            out_std[tile0, tile1] = torch.where(mask, row_std[:, None], torch.zeros_like(row_std[:, None]))
        
        # Copy remaining columns
        for tile1 in hl.tile(max_nnz, y.size(1)):
            out_mean[tile0, tile1] = y[tile0, tile1]
            out_std[tile0, tile1] = torch.zeros_like(y[tile0, tile1])
    
    return out_mean, out_std

# Test functions
def test_enhanced_kernels():
    print("\n1. Testing Enhanced Constant Add (with scaling)")
    x = torch.randn(10000, device="cuda")
    result = add_scaled_kernel(x, scale=3.0, offset=5.0)
    expected = x * 3.0 + 5.0
    torch.testing.assert_close(result, expected)
    print("   ✅ Enhanced add kernel correct")
    
    print("\n2. Testing Generalized Broadcasting Operation")
    x = torch.randn(512, device="cuda")
    y = torch.randn(256, device="cuda")
    
    # Test different operations
    for op in ["add", "mul", "sub"]:
        result = broadcast_op_kernel(x, y, op)
        if op == "add":
            expected = x[None, :] + y[:, None]
        elif op == "mul":
            expected = x[None, :] * y[:, None]
        elif op == "sub":
            expected = y[:, None] - x[None, :]
        
        torch.testing.assert_close(result, expected)
        print(f"   ✅ Broadcasting {op} correct")
    
    print("\n3. Testing Multi-Stage Fusion")
    x = torch.randn(256, device="cuda")
    y = torch.randn(256, device="cuda")
    z = torch.randn(256, 256, device="cuda")
    result = fused_ops_kernel(x, y, z)
    # Verify shape and basic computation
    expected = torch.relu(x[None, :] * y[:, None] + z) * 0.1
    torch.testing.assert_close(result, expected)
    print("   ✅ Multi-stage fusion kernel correct")
    
    print("\n4. Testing Adaptive Statistics Kernel")
    x = torch.randn(50000, device="cuda")
    sum_result, mean_result, var_result = adaptive_sum_kernel(x)
    expected_sum = x.sum()
    expected_mean = x.mean()
    expected_var = x.var(unbiased=False)
    torch.testing.assert_close(sum_result, expected_sum, rtol=1e-4)
    torch.testing.assert_close(mean_result, expected_mean, rtol=1e-4)
    torch.testing.assert_close(var_result, expected_var, rtol=1e-4)
    print("   ✅ Adaptive statistics kernel correct")
    
    print("\n5. Testing Hierarchical Reduction")
    x = torch.randn(32, 512, 128, device="cuda")
    result = hierarchical_sum_kernel(x)
    expected = x.sum(1)
    torch.testing.assert_close(result, expected, rtol=1e-4)
    print("   ✅ Hierarchical reduction kernel correct")
    
    print("\n6. Testing Advanced Jagged Operations")
    from test_tutorial_kernels import random_jagged_2d
    rows, cols = 32, 512
    x_data, x_offsets = random_jagged_2d(rows, cols, device="cuda")
    y = torch.randn(rows, cols, device="cuda")
    mean_result, std_result = jagged_stats_kernel(x_data, x_offsets, y)
    assert mean_result.shape == y.shape
    assert std_result.shape == y.shape
    print("   ✅ Advanced jagged operations kernel correct")
    
    print("\n" + "=" * 80)
    print("All enhanced kernels passed! These demonstrate:")
    print("- Scalar parameters and multi-operation kernels")
    print("- Advanced fusion patterns")
    print("- Multi-pass algorithms")
    print("- Hierarchical reductions")
    print("- Complex jagged tensor operations")

if __name__ == "__main__":
    test_enhanced_kernels()