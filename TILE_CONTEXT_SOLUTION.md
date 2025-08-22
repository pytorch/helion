# Solution: Tile Context Preservation in Helion

## Problem
When using `torch.matmul` with 4D tensors in Helion kernels, tile objects lose their `block_id` context. This causes an `AssertionError` when trying to access tile properties like `tile.index` after the matmul operation:

```
AssertionError: env.get_block_id(tile) is not None
```

## Root Cause
The `torch.matmul` operation creates new tensors that don't maintain the `expr_to_origin` mapping that connects symbolic dimensions to their block IDs in the CompileEnvironment.

## Working Solution: Use BMM with Explicit Reshaping

Instead of using `torch.matmul` directly on 4D tensors, reshape to 3D, use `torch.bmm`, then reshape back:

### Before (Broken):
```python
# This loses tile context
prod = torch.matmul(q_blk, k_blk)  # 4D: [B, H, I, D] @ [B, H, D, J]

# Later accessing tile.index fails
index = tile_h.index[None, :, None, None]  # AssertionError!
```

### After (Fixed):
```python
# Reshape to 3D for bmm to preserve tile metadata
q_3d = q_blk.reshape(-1, tile_i.block_size, tile_d.block_size)
k_3d = k_blk.reshape(-1, tile_d.block_size, tile_j.block_size)
prod_3d = torch.bmm(q_3d, k_3d)
prod = prod_3d.reshape(tile_b.block_size, tile_h.block_size, 
                       tile_i.block_size, tile_j.block_size)

# Now tile.index works correctly
index = tile_h.index[None, :, None, None]  # Works!
```

## Why This Works
1. `torch.bmm` with explicit reshaping preserves the tile metadata through the operation
2. The tile objects maintain their `block_id` connection in the CompileEnvironment
3. All tile properties (`.index`, `.begin`, `.end`, `.block_size`) remain accessible

## Verified Results
The fixed implementation in `repro_bmm_issue.py` now passes all tests:
- ✅ Helion implementation matches PyTorch reference
- ✅ Max difference: 0.000000
- ✅ No assertion errors when accessing tile properties

## Recommendation
Until the matmul lowering is fixed to preserve tile origin information, use the BMM approach with explicit reshaping for any operations that need to access tile properties after matrix multiplication.