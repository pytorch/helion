# Tile Context Loss Through Operations Analysis

## Problem Summary

When using `torch.matmul` in Helion kernels, tile objects lose their `block_id` context, causing an `AssertionError` when trying to access tile properties like `tile.index` after the operation. This doesn't happen with `torch.bmm`.

## Root Cause

The issue occurs in `/home/willfeng/local/helion/helion/language/tile_ops.py`:

```python
@_decorators.register_fake(tile_index)
def _(tile: torch.SymInt) -> torch.Tensor:
    assert isinstance(tile, torch.SymInt)
    env = CompileEnvironment.current()
    assert env.get_block_id(tile) is not None  # <-- This assertion fails
    return torch.empty([tile], dtype=env.settings.index_dtype, device=env.device)
```

## How Tile Context is Tracked

1. **Tile Creation**: When `hl.tile()` creates tiles, each tile gets a unique `block_id`
2. **Block ID Tracking**: The `CompileEnvironment` tracks the mapping between symbolic integers (SymInt) and their block IDs through the `expr_to_origin` dictionary
3. **Origin Preservation**: Operations need to preserve this origin information for tiles to remain usable

## Why BMM Works but Matmul Fails

### BMM Approach (Works)
```python
# Explicit reshaping preserves tile metadata
scores.reshape(-1, tile_i.block_size, tile_j.block_size)
v_blk.reshape(-1, tile_j.block_size, tile_v.block_size)
result = torch.bmm(scores_reshaped, v_blk_reshaped)
```

### Matmul Approach (Fails)
```python
# Direct matmul on 4D tensors loses tile metadata
prod = torch.matmul(q_blk, k_blk)
# Later accessing tile_h.index fails because block_id is lost
```

## The Issue in Detail

1. When `torch.matmul` is called on tensors derived from tiles, it creates new tensors
2. These new tensors don't maintain the `expr_to_origin` mapping that connects symbolic dimensions to their block IDs
3. When later code tries to access `tile.index`, it needs to look up the block_id, but can't find it
4. The assertion `env.get_block_id(tile) is not None` fails

## Code Flow

1. **Tile operations preserve context**:
   - `tile[...]` indexing operations maintain tile metadata
   - Simple arithmetic operations preserve tile context

2. **Some operations lose context**:
   - `torch.matmul` creates new tensors without preserving origin tracking
   - The reshape operations in the matmul lowering don't propagate the block_id mappings

## Solution Approaches

1. **Use BMM with explicit reshaping** (current workaround):
   - Manually reshape to 3D before bmm
   - This preserves tile metadata through the operation

2. **Fix matmul lowering** (proper fix):
   - The `codegen_matmul` function in `inductor_lowering.py` needs to preserve tile origin information
   - When creating reshaped tensors, it should propagate the `expr_to_origin` mappings

3. **Alternative pattern**:
   - Store tile indices before the matmul operation
   - Use stored indices instead of accessing `tile.index` after matmul

## Example Workaround

Instead of:
```python
prod = torch.matmul(q_blk, k_blk)
# ... later ...
index = tile_h.index[None, :, None, None]  # Fails
```

Use:
```python
# Store index before matmul
h_index = tile_h.index
prod = torch.matmul(q_blk, k_blk)
# ... later ...
index = h_index[None, :, None, None]  # Works
```

## Recommendation

For now, use the BMM approach with explicit reshaping or store tile indices before matmul operations. A proper fix would require modifying the matmul lowering to preserve tile origin information through the reshape operations.