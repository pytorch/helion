# Jagged Iteration Patterns in GPU Kernels

When a kernel operates on variable-length rows (jagged/ragged tensors), there are three fundamental strategies for mapping the work onto GPU threads.

## Pattern 1: Masked Batched

Process **multiple rows together** in a single tile. The inner loop runs to `max(lengths)` across all rows in the tile, and a per-lane mask discards out-of-bounds elements.

```
tile_b = [row0, row1, row2, row3]   # lengths = [3, 7, 2, 5]
for tile_k in range(0, max=7):       # loop to max across tile
    mask = [k<3, k<7, k<2, k<5]     # per-lane validity
    vals = load(data, indices, mask)
    acc += vals
```

**Best when:** Rows have similar lengths (high density = avg_length / max_length_in_tile).

**Helion examples:** `jagged_sum`, `jagged_mean`, `jagged_softmax`, `jagged_layer_norm`, `jagged_dense_bmm`, `jagged_dense_add`.

## Pattern 2: Maskless Per-Entity

Process **one variable-length entity at a time** (a group, a sequence, a batch element). The inner loop runs to that entity's exact length. No masking needed — every loaded element is valid.

```
for g in grid(num_groups):              # one group per grid element
    start = group_offsets[g]
    end = group_offsets[g + 1]
    M_g = end - start
    for tile_m, tile_n in tile([M_g, N]): # exact bounds, no waste
        acc = zeros()
        for tile_k in tile(K):
            acc += A[start + tile_m, tile_k] @ B[tile_k, tile_n]
        out[start + tile_m, tile_n] = acc
```

**Best when:** Entities are large enough to fill a thread block, or lengths vary wildly (avoids worst-case entity dominating an entire tile).

**Helion examples:** `grouped_gemm_jagged` (`hl.grid(G)` + `hl.tile([M_g, N])`), `jagged_hstu_attn` (`block_size=[1,1,None]`).

## Pattern 3: Persistent Dynamic

A **fixed number of workers** (typically = number of SMs) stay resident and stride through tiles across all groups. Each worker takes every `num_workers`-th tile, distributing work evenly regardless of group sizes.

```
num_workers = num_SMs
for worker_id in grid(num_workers):
    for g in grid(num_groups):
        num_group_tiles = ceil(M_g / BLOCK_M) * ceil(N / BLOCK_N)
        for local_tile in grid(num_group_tiles):
            tile_in_group = local_tile * num_workers + worker_id
            if tile_in_group < num_group_tiles:
                # decode 2D tile coords from linear index
                m_tile = tile_in_group % num_m_tiles
                n_tile = tile_in_group // num_m_tiles
                row_idx = group_start + m_tile * BLOCK_M + arange(BLOCK_M)
                col_idx = n_tile * BLOCK_N + arange(BLOCK_N)
                # boundary masks for partial tiles
                acc = matmul_tile(A[row_idx, :], B[:, col_idx])
                store(out, row_idx, col_idx, acc, mask=valid)
```

**Best when:** Length distribution is highly skewed or bimodal, and the imbalance would leave SMs idle under the other two patterns.

**Helion examples:** `grouped_gemm_jagged_persistent`.

## Summary

| | Masked Batched | Maskless Per-Entity | Persistent Dynamic |
|---|---|---|---|
| Entities per tile | Many | One | Varies (work-stealing) |
| Masking | Per-lane mask | None | Per-tile mask |
| Wasted compute | Yes (padding to max) | None | Minimal |
| Load balance | Good if lengths similar | Poor if lengths vary | Excellent |
| Complexity | Low | Low | High |
| Best density regime | High (>0.7) | Any | Low / skewed |

## Implications for Autotuning

The key insight is that **these are lowering strategies, not user-visible code patterns**. A user writes a single jagged loop:

```python
for tile_b in hl.tile(B):
    for tile_k in hl.tile(jagged_size(tile_b)):
        acc += x[tile_b, tile_k]
```

The compiler can lower this to any of the three strategies and let the autotuner benchmark them on the actual data distribution and hardware. The user doesn't choose — the autotuner measures.
