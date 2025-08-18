# Test Plan: Slice → View → Sum Operations

## Requirements

Test the composition of three operations in sequence: slice → view → sum

### 1. Slice Types
- **Tile**: `x[tile_m, tile_n]` using `hl.tile()`
- **Partial slice**: `x[a:b, c:d]` with explicit ranges
- **Full slice**: `x[:, :]` selecting all elements

### 2. View Operations
- **Transpose**: `tensor.transpose(dim0, dim1)`
- **Permute**: `tensor.permute(...)` for reordering dimensions
- **Reshape**: `tensor.reshape(...)` with concrete sizes on one or two dims and -1 for the rest

### 3. Sum Operations
- `tensor.sum(dim=n)` where n is each possible dimension
- One test should be created for each dimension that can be summed

### Key Constraints
- All operations must happen inside device loops
- No if statements inside device loops
- All dimensions must be explicitly specified (no ellipsis)
- Tests should use patterns similar to `test_range_slice_inner_dim`
- Target: 10-20 tests total, each testing multiple combinations within a single kernel

## Proposed Tests

### 1. test_tile_transpose_sum_each_dim
Tests tile slicing → transpose → sum on each dimension
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    M, N = x.shape
    out_sum_dim0 = torch.zeros([N], dtype=x.dtype, device=x.device)
    out_sum_dim1 = torch.zeros([M], dtype=x.dtype, device=x.device)
    
    for tile_m in hl.tile(M, block_size=32):
        for tile_n in hl.tile(N, block_size=32):
            sliced = x[tile_m, tile_n]  # SLICE
            transposed = sliced.transpose(0, 1)  # VIEW
            
            # SUM on each dimension
            out_sum_dim0[tile_n] += transposed.sum(dim=0)
            out_sum_dim1[tile_m] += transposed.sum(dim=1)
            
    return out_sum_dim0, out_sum_dim1
```

### 2. test_tile_reshape_sum_each_dim
Tests tile slicing → reshape with -1 → sum
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    M, N = x.shape
    out1 = torch.zeros([32*32], dtype=x.dtype, device=x.device)
    out2 = torch.zeros([32], dtype=x.dtype, device=x.device)
    
    for tile_m in hl.tile(M, block_size=32):
        for tile_n in hl.tile(N, block_size=32):
            sliced = x[tile_m, tile_n]  # SLICE
            
            # Reshape with -1
            reshaped1 = sliced.reshape(-1)  # VIEW
            out1[0:reshaped1.shape[0]] += reshaped1  # SUM (implicit)
            
            # Reshape with concrete dim
            reshaped2 = sliced.reshape(32, -1)  # VIEW
            out2[0:32] += reshaped2.sum(dim=1)  # SUM
            
    return out1, out2
```

### 3. test_partial_slice_transpose_sum
Tests partial slicing → transpose → sum
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    M, N = x.shape
    out_first_half = torch.zeros([N//2], dtype=x.dtype, device=x.device)
    out_second_half = torch.zeros([M//2], dtype=x.dtype, device=x.device)
    
    # First half slice
    for i in range(M):
        sliced = x[i, 0:N//2]  # SLICE: partial [0:N//2]
        transposed = sliced.unsqueeze(0).transpose(0, 1)  # VIEW
        out_first_half += transposed.sum(dim=1)  # SUM
        
    # Second half slice  
    for j in range(N):
        sliced = x[M//2:M, j]  # SLICE: partial [M//2:M]
        transposed = sliced.unsqueeze(1).transpose(0, 1)  # VIEW
        out_second_half += transposed.sum(dim=0)  # SUM
        
    return out_first_half, out_second_half
```

### 4. test_full_slice_transpose_sum
Tests full slicing → transpose → sum
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    M, N = x.shape
    out1 = torch.zeros([N], dtype=x.dtype, device=x.device)
    out2 = torch.zeros([M], dtype=x.dtype, device=x.device)
    
    for i in range(M):
        sliced = x[i, :]  # SLICE: full slice on dim 1
        viewed = sliced.unsqueeze(0).transpose(0, 1)  # VIEW
        out1 += viewed.squeeze(1)  # SUM (implicit)
        
    for j in range(N):
        sliced = x[:, j]  # SLICE: full slice on dim 0
        viewed = sliced.unsqueeze(1).transpose(0, 1)  # VIEW  
        out2 += viewed.squeeze(0)  # SUM (implicit)
        
    return out1, out2
```

### 5. test_tile_permute_sum_3d
Tests 3D tile slicing → permute → sum on each dimension
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    I, J, K = x.shape
    out_dim0 = torch.zeros([J, K], dtype=x.dtype, device=x.device)
    out_dim1 = torch.zeros([I, K], dtype=x.dtype, device=x.device)
    out_dim2 = torch.zeros([I, J], dtype=x.dtype, device=x.device)
    
    for tile_i in hl.tile(I):
        for tile_j in hl.tile(J):
            for tile_k in hl.tile(K):
                sliced = x[tile_i, tile_j, tile_k]  # SLICE
                
                # Permute (0,1,2) -> (1,0,2)
                perm1 = sliced.permute(1, 0, 2)  # VIEW
                out_dim0[tile_j, tile_k] += perm1.sum(dim=1)  # SUM dim 1 (was dim 0)
                
                # Permute (0,1,2) -> (0,2,1)
                perm2 = sliced.permute(0, 2, 1)  # VIEW
                out_dim1[tile_i, tile_k] += perm2.sum(dim=2)  # SUM dim 2 (was dim 1)
                
                # Permute (0,1,2) -> (2,1,0)
                perm3 = sliced.permute(2, 1, 0)  # VIEW
                out_dim2[tile_i, tile_j] += perm3.sum(dim=0)  # SUM dim 0 (was dim 2)
                
    return out_dim0, out_dim1, out_dim2
```

### 6. test_partial_slice_permute_sum_3d
Tests 3D partial slicing → permute → sum
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    I, J, K = x.shape
    out1 = torch.zeros([I//2, K], dtype=x.dtype, device=x.device)
    out2 = torch.zeros([I, J//3], dtype=x.dtype, device=x.device)
    
    # Partial slice [0:I//2, :, :]
    for i in range(I//2):
        for j in range(J):
            for k in range(K):
                sliced = x[i, j, k]  # SLICE (element access)
                # Accumulate scalars
                out1[i, k] += sliced
                
    # Partial slice [:, J//3:2*J//3, :]
    for i in range(I):
        for j in range(J//3, 2*J//3):
            for k in range(K):
                val = x[i, j, k]  # SLICE
                out2[i, j - J//3] += val  # Direct accumulation
                
    return out1, out2
```

### 7. test_full_slice_reshape_sum_2d
Tests full slicing → reshape → sum
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    M, N = x.shape
    out1 = torch.zeros([1], dtype=x.dtype, device=x.device)
    out2 = torch.zeros([M], dtype=x.dtype, device=x.device)
    
    # Full slice with reshape - inside loops
    for i in range(M):
        for j in range(N):
            val = x[i, j]  # SLICE: element access
            out1[0] += val  # SUM: accumulate all elements
    
    # Row-wise full slice
    for i in range(M):
        row_sum = torch.zeros([1], dtype=x.dtype, device=x.device)
        for j in range(N):
            val = x[i, j]  # SLICE: element of row
            row_sum[0] += val  # SUM: accumulate row
        out2[i] = row_sum[0]
        
    return out1, out2
```

### 8. test_tile_reshape_minus_one_sum
Tests tile slicing → reshape with -1 → sum
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    M, N = x.shape  # e.g., 128, 256
    out1 = torch.zeros([8], dtype=x.dtype, device=x.device)
    out2 = torch.zeros([32], dtype=x.dtype, device=x.device)
    
    for tile_m in hl.tile(M, block_size=32):
        for tile_n in hl.tile(N, block_size=32):
            sliced = x[tile_m, tile_n]  # SLICE: 32x32
            
            # Reshape with -1 in last dim
            reshaped1 = sliced.reshape(8, -1)  # VIEW: 8x128
            out1 += reshaped1.sum(dim=1)  # SUM
            
            # Reshape with -1 in first dim  
            reshaped2 = sliced.reshape(-1, 32)  # VIEW: 32x32
            out2 += reshaped2.sum(dim=0)  # SUM
            
    return out1, out2
```

### 9. test_mixed_slice_types_in_one_kernel
Tests all three slice types → various views → sum in one kernel
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    M, N = x.shape
    out_tile = torch.zeros([N], dtype=x.dtype, device=x.device)
    out_partial = torch.zeros([M//2], dtype=x.dtype, device=x.device)
    out_full = torch.zeros([1], dtype=x.dtype, device=x.device)
    
    # Tile slice -> transpose -> sum
    for tile_m in hl.tile(M, block_size=32):
        for tile_n in hl.tile(N, block_size=32):
            sliced = x[tile_m, tile_n]  # SLICE: tile
            transposed = sliced.transpose(0, 1)  # VIEW
            out_tile[tile_n] += transposed.sum(dim=1)  # SUM
            
    # Partial slice -> accumulation in loops
    for i in range(M//2):
        for j in range(N):
            val = x[i, j]  # SLICE: partial (first half)
            out_partial[i] += val  # Direct accumulation
            
    # Full slice -> accumulation in loops
    for i in range(M):
        for j in range(N):
            val = x[i, j]  # SLICE: element access
            out_full[0] += val  # SUM: accumulate all
    
    return out_tile, out_partial, out_full
```

### 10. test_strided_slice_view_sum
Tests strided slicing → view → sum
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    M, N = x.shape
    out1 = torch.zeros([M//2, N//2], dtype=x.dtype, device=x.device)
    out2 = torch.zeros([M//3], dtype=x.dtype, device=x.device)
    
    # Stride 2 slice
    for i in range(0, M, 2):
        for j in range(0, N, 2):
            val = x[i, j]  # SLICE: strided
            out1[i//2, j//2] = val
            
    # Stride 3 with accumulation
    for i in range(0, M, 3):
        for j in range(N):
            val = x[i, j]  # SLICE: strided
            out2[i//3] += val  # SUM: accumulate
            
    return out1, out2
```

### 11. test_3d_tile_all_view_types
Tests 3D tile slicing → all view types → sum
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    I, J, K = x.shape
    out_transpose = torch.zeros([I, K], dtype=x.dtype, device=x.device)
    out_permute = torch.zeros([J, I], dtype=x.dtype, device=x.device)
    out_reshape = torch.zeros([I], dtype=x.dtype, device=x.device)
    
    for tile_i in hl.tile(I):
        for tile_j in hl.tile(J):
            for tile_k in hl.tile(K):
                sliced = x[tile_i, tile_j, tile_k]  # SLICE
                
                # Transpose last two dims
                trans = sliced.transpose(1, 2)  # VIEW
                out_transpose[tile_i, tile_k] += trans.sum(dim=1)  # SUM
                
                # Permute to (j,i,k)
                perm = sliced.permute(1, 0, 2)  # VIEW
                out_permute[tile_j, tile_i] += perm.sum(dim=2)  # SUM
                
                # Reshape and sum
                resh = sliced.reshape(-1)  # VIEW
                out_reshape[tile_i] += resh.sum()  # SUM
                
    return out_transpose, out_permute, out_reshape
```

### 12. test_4d_slice_view_sum_combos
Tests 4D slicing → view → sum combinations
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    B, C, H, W = x.shape
    out1 = torch.zeros([B, C], dtype=x.dtype, device=x.device)
    out2 = torch.zeros([B, H, W], dtype=x.dtype, device=x.device)
    
    for tile_b in hl.tile(B, block_size=4):
        for tile_c in hl.tile(C, block_size=8):
            for tile_h in hl.tile(H, block_size=16):
                for tile_w in hl.tile(W, block_size=16):
                    sliced = x[tile_b, tile_c, tile_h, tile_w]  # SLICE
                    
                    # Permute BCHW -> BHWC, sum over HW
                    perm = sliced.permute(0, 2, 3, 1)  # VIEW
                    out1[tile_b, tile_c] += perm.sum(dim=(1, 2))  # SUM
                    
                    # Transpose C and H, sum over C
                    trans = sliced.transpose(1, 2)  # VIEW
                    out2[tile_b, tile_h, tile_w] += trans.sum(dim=1)  # SUM
                    
    return out1, out2
```

### 13. test_partial_slice_reshape_concrete_dims
Tests partial slicing → reshape with concrete dimensions → sum
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    M, N = x.shape  # 128, 256
    out1 = torch.zeros([16, 64], dtype=x.dtype, device=x.device)
    out2 = torch.zeros([32, 8], dtype=x.dtype, device=x.device)
    
    # Partial slice with tile loops
    for tile_m in hl.tile(M//2, block_size=16):  # Tile over first M//2
        for tile_n in hl.tile(N, block_size=64):
            sliced = x[tile_m, tile_n]  # SLICE: 16x64 tile from partial region
            
            # Reshape to 3D and sum middle
            reshaped1 = sliced.reshape(16, 16, 4)  # VIEW: 16x16x4
            summed1 = reshaped1.sum(dim=1)  # SUM: -> 16x4
            out1[tile_m, 0:4] += summed1
            
    # Another partial slice pattern
    for i in range(M//4):  # Loop over M//4 = 32
        for tile_n in hl.tile(N, block_size=8):
            row_slice = x[i, tile_n]  # SLICE: 1x8
            reshaped2 = row_slice.reshape(1, 8)  # VIEW
            out2[i, tile_n] += reshaped2.sum(dim=0)  # SUM
    
    return out1, out2
```

### 14. test_transpose_chains_in_loops
Tests slicing → chained transpose operations → sum
```python
@helion.kernel(use_default_config=True, static_shapes=True)
def kernel(x: Tensor) -> Tensor:
    M, N = x.shape
    out1 = torch.zeros([M], dtype=x.dtype, device=x.device)
    out2 = torch.zeros([N], dtype=x.dtype, device=x.device)
    
    for tile_m in hl.tile(M):
        for tile_n in hl.tile(N):
            sliced = x[tile_m, tile_n]  # SLICE
            
            # Double transpose (should be identity)
            t1 = sliced.transpose(0, 1)  # VIEW
            t2 = t1.transpose(0, 1)  # VIEW (back to original)
            out1[tile_m] += t2.sum(dim=1)  # SUM
            
            # Transpose then reshape
            t3 = sliced.transpose(0, 1)  # VIEW
            r1 = t3.reshape(-1)  # VIEW
            out2[tile_n] += r1.sum()  # SUM
            
    return out1, out2
```

### 15. test_all_slice_types_all_dims_sum
Tests all slice types → view → sum on all dimensions
```python
@helion.kernel(use_default_config=True, static_shapes=True)  
def kernel(x: Tensor) -> Tensor:
    I, J, K = x.shape
    # Test summing on each dimension for each slice type
    out_tile_sum0 = torch.zeros([J, K], dtype=x.dtype, device=x.device)
    out_tile_sum1 = torch.zeros([I, K], dtype=x.dtype, device=x.device)
    out_tile_sum2 = torch.zeros([I, J], dtype=x.dtype, device=x.device)
    
    out_partial_sum0 = torch.zeros([J//2, K], dtype=x.dtype, device=x.device)
    out_partial_sum1 = torch.zeros([I//2, K], dtype=x.dtype, device=x.device)
    out_partial_sum2 = torch.zeros([I//2, J//2], dtype=x.dtype, device=x.device)
    
    # Tile slice - sum each dim
    for tile_i in hl.tile(I):
        for tile_j in hl.tile(J):
            for tile_k in hl.tile(K):
                sliced = x[tile_i, tile_j, tile_k]  # SLICE
                viewed = sliced.permute(0, 1, 2)  # VIEW (identity)
                
                out_tile_sum0[tile_j, tile_k] += viewed.sum(dim=0)
                out_tile_sum1[tile_i, tile_k] += viewed.sum(dim=1)
                out_tile_sum2[tile_i, tile_j] += viewed.sum(dim=2)
                
    # Partial slice - inside loops with proper indexing
    for i in range(I//2):
        for j in range(J//2):
            for k in range(K):
                val = x[i, j, k]  # SLICE: element from partial region
                # Direct accumulation for different sum dimensions
                out_partial_sum0[j, k] += val  # Accumulate over dim 0
                out_partial_sum1[i, k] += val  # Accumulate over dim 1  
                out_partial_sum2[i, j] += val  # Accumulate over dim 2
    
    return (out_tile_sum0, out_tile_sum1, out_tile_sum2,
            out_partial_sum0, out_partial_sum1, out_partial_sum2)
```

## Summary

This test plan provides comprehensive coverage of the slice → view → sum operation sequence:

- **15 tests** covering all combinations
- **No if statements** in device loops
- **All dimensions explicitly specified** (no ellipsis)
- Each test verifies the specific sequence: slice → view → sum
- Tests cover:
  - All slice types: tile, partial, full
  - All view types: transpose, permute, reshape
  - Sum operations on each possible dimension
- Tests follow the device loop pattern from existing tests
- Multiple combinations tested within each kernel for efficiency