# GEMM + Reduce-Scatter Implementation Analysis

## Overview

This document analyzes the implementation of fused GEMM + reduce-scatter kernels in Helion, including what works, what doesn't, and why.

## Files Created

| File | Status | Description |
|------|--------|-------------|
| `examples/gemm_reduce_scatter_fused.py` | ✅ Working | Full all-reduce + slice extraction |
| `examples/gemm_reduce_scatter_true.py` | ❌ WIP | True reduce-scatter (blocked by Helion limitations) |

---

## Working Implementation: `gemm_reduce_scatter_fused.py`

### Approach
1. Each rank computes its local GEMM tile and writes to symmetric memory buffer
2. Barrier sync ensures all ranks have written the tile
3. Reduce the tile from all remote buffers (full all-reduce per tile)
4. Write full reduced result to output buffer (M x N)
5. Final barrier sync
6. **In Python wrapper**: Extract this rank's slice (M x N//world_size)

### Key Code Pattern
```python
for tile_m, tile_n in hl.tile([M, N]):
    # GEMM
    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    for tile_k in hl.tile(K, block_size=BLOCK_SIZE_K):
        acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
    local_buf[tile_m, tile_n] = acc

    # Sync
    hl.triton_kernel("symm_mem_sync", ...)

    # All-reduce (not reduce-scatter)
    reduced = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    for buf in buf_tuple:
        reduced = reduced + buf[tile_m, tile_n].to(torch.float32)
    output[tile_m, tile_n] = reduced  # Full M x N output

    # Final sync
    hl.triton_kernel("symm_mem_sync", ...)

# In wrapper: extract slice
return output[:, my_slice_start:my_slice_end].contiguous()
```

### Trade-offs
- **Pro**: Works correctly, simple implementation
- **Con**: Computes and stores full reduced result, then discards 3/4 of it (for 4 ranks)
- **Con**: Extra memory for full M x N output buffer

---

## Non-Working Implementation: `gemm_reduce_scatter_true.py`

### Goal
Have each rank only compute and store its 1/world_size portion of the reduced result, avoiding the full all-reduce overhead.

### Attempted Approaches

#### Approach 1: Two Separate Tile Loops
```python
# Phase 1: GEMM for all tiles
for tile_m, tile_n in hl.tile([M, N]):
    # GEMM + sync
    ...

# Phase 2: Reduce only this rank's slice
for tile_m, tile_out_n in hl.tile([M, size_per_rank]):
    # Map output coords to buffer coords
    buf_col_start = my_slice_start + tile_out_n.begin
    reduced = reduced + buf[tile_m, buf_col_start:buf_col_end]
    output[tile_m, tile_out_n] = reduced
```

**Result**: ❌ Failed with type error
```
helion.exc.TorchOpTracingError: RuntimeError: The size of tensor a (u7) must match
the size of tensor b (u10) at non-singleton dimension 1
```

Helion's symbolic shape system cannot unify `tile_out_n` (from iterating over `size_per_rank`) with the sliced buffer shape (computed from `my_slice_start + tile_out_n.begin`).

#### Approach 2: Single Loop with Conditional Write
```python
for tile_m, tile_n in hl.tile([M, N]):
    # GEMM + sync + reduce (same as before)
    ...

    # Conditional write based on tile ownership
    output_tile_id = tile_n.id - my_first_tile_n
    output[tile_m, output_tile_id * BLOCK_SIZE_N : (output_tile_id + 1) * BLOCK_SIZE_N] = reduced
```

**Result**: ❌ Hangs/crashes

For tiles outside this rank's slice, `output_tile_id` becomes negative or >= num_tiles_per_rank, causing out-of-bounds memory writes.

---

## Root Cause Analysis

### The Fundamental Problem

In Helion, `hl.tile([M, N])` generates a 2D grid of blocks where **every block executes**. There is no mechanism to:

1. **Skip blocks conditionally**: All blocks in the grid run regardless of any condition
2. **Mask stores at block level**: Triton's `tl.store` with mask works at element level, but Helion's `output[tile_m, tile_n] = value` generates unmasked stores
3. **Use different grids for different phases**: Each `hl.tile()` call contributes to the same kernel's block configuration

### Illustration

For N=256, world_size=4, block_size_n=64:
- Grid has 4 tiles along N dimension (tile_n.id = 0, 1, 2, 3)
- Rank 0 owns tile_n.id = 0
- Rank 1 owns tile_n.id = 1
- etc.

When rank 0 executes:
- Tile 0: `output_tile_id = 0 - 0 = 0` → writes to output[:, 0:64] ✅
- Tile 1: `output_tile_id = 1 - 0 = 1` → writes to output[:, 64:128] ❌ OUT OF BOUNDS
- Tile 2: `output_tile_id = 2 - 0 = 2` → writes to output[:, 128:192] ❌ OUT OF BOUNDS
- Tile 3: `output_tile_id = 3 - 0 = 3` → writes to output[:, 192:256] ❌ OUT OF BOUNDS

---

## What Would Be Needed to Fix This

### Option 1: Helion Support for Conditional Stores
Add `tl.where`-style masking to Helion's store operations:
```python
# Hypothetical syntax
if tile_n.id >= my_first_tile_n and tile_n.id < my_last_tile_n:
    output[tile_m, output_col] = reduced
```

### Option 2: Persistent Kernel Style
Allow launching only a subset of blocks:
```python
# Hypothetical: iterate only over this rank's tiles
for tile_m, tile_n in hl.tile([M, N],
                               n_range=(my_first_tile_n, my_last_tile_n)):
    ...
```

### Option 3: Separate Kernels
Split into two kernels:
1. GEMM kernel: All ranks compute all tiles
2. Reduce-scatter kernel: Each rank only processes its slice

```python
# Kernel 1: GEMM (all tiles)
@helion.jit
def gemm_kernel(a, b, local_buf):
    for tile_m, tile_n in hl.tile([M, N]):
        local_buf[tile_m, tile_n] = gemm(a, b)

# Kernel 2: Reduce-scatter (only this rank's slice)
@helion.jit
def reduce_scatter_kernel(buf_tuple, output, my_slice_start, size_per_rank):
    for tile_m, tile_out_n in hl.tile([M, size_per_rank]):
        # Read from all buffers at offset my_slice_start
        ...
```

### Option 4: Use Full Output Buffer (Current Workaround)
The current working solution: allocate M x N output, write everything, extract slice in Python.

---

## Performance Implications

| Approach | Memory | Compute | Communication |
|----------|--------|---------|---------------|
| Full all-reduce + slice | O(M × N) output | Computes all tiles | Same |
| True reduce-scatter | O(M × N/world_size) output | Only reduces owned tiles | Same |

For large models with many ranks, the true reduce-scatter approach would save:
- Memory: `(world_size - 1) / world_size` of output buffer
- Compute: `(world_size - 1) / world_size` of reduce operations

---

## Conclusion

The true reduce-scatter pattern is blocked by Helion's execution model where all blocks in a tile grid must execute. The working implementation uses the full all-reduce approach with post-kernel slicing, which is functionally correct but not optimal for memory usage.

Future Helion enhancements to support conditional block execution or masked stores would enable the more efficient true reduce-scatter pattern.
