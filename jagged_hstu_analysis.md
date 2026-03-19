# HSTU Attention: `hl.jagged_tile` Conversion Analysis

## Goal

Rewrite `examples/jagged_hstu_attn.py` to use `hl.jagged_tile` for the sequence
dimension, replacing the current manual masking pattern:

```python
# Current: manual mask with hl.tile over max_seq_len
for tile_b, tile_h, tile_q in hl.tile(
    [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
):
    starts = seq_offsets[tile_b.begin]       # scalar
    ends = seq_offsets[tile_b.begin + 1]     # scalar
    seq_len = ends - starts                  # scalar
    if tile_q.begin < seq_len:               # manual guard
        mask_q = tile_q.index < seq_len      # manual mask
        q_blk = q[tile_q.index + starts, tile_h.begin, :]
        ...
        for tile_kv in hl.tile(0, tile_q.end, block_size=None):  # causal bound
```

```python
# Target: jagged_tile handles variable seq_len automatically
for tile_b in hl.tile(num_batches):          # block_size=1 via config
    starts = seq_offsets[tile_b]             # [tile_b] tensor
    ends = seq_offsets[tile_b.index + 1]
    seq_len = ends - starts                  # [tile_b] tensor
    for tile_h in hl.tile(num_heads):        # block_size=1 via config
        for tile_q in hl.jagged_tile(seq_len):  # auto mask + auto loop bound
            q_row = starts[:, None] + tile_q.index[None, :]  # [tile_b, tile_q]
            ...
            for tile_kv in hl.tile(0, tile_q.end):  # causal bound
```

This would enable Pattern 1→2 degradation: with `batch block_size > 1`, multiple
sequences are batched with an outer mask (Pattern 1); with `batch block_size = 1`,
the mask is eliminated (Pattern 2).

## Four Compiler Blockers

### Blocker 1: `torch.matmul` rejects mismatched ndims

**Severity:** Low (5-line fix)

**File:** `helion/_compiler/matmul_utils.py:28-31`

```python
def torch_matmul_replacement(a, b, ...):
    if a.dim() != b.dim():
        raise NotImplementedError(
            "torch.matmul with different input tensor dims is not supported"
        )
```

**Problem:** The jagged_tile version produces 3D tiles `[tile_b, tile_q, dim]` but
the B-side tensor `k_blk.T` may be 2D `[dim, tile_kv]`. PyTorch's native
`torch.matmul` broadcasts 3D @ 2D automatically.

**Fix:** Unsqueeze the lower-dim operand:
```python
if a.dim() == 3 and b.dim() == 2:
    return torch.bmm(a, b.unsqueeze(0))
if a.dim() == 2 and b.dim() == 3:
    return torch.bmm(a.unsqueeze(0), b)
```

Note: `hl.dot` in `matmul_ops.py:114-131` already has batch-dimension broadcasting
logic, so this fix is consistent with the existing design.

### Blocker 2: `starts[:, None]` subscript on scalar-like tensor

**Severity:** Medium

**File:** `helion/language/view_ops.py:76-89`

```python
@_decorators.register_fake(subscript)
def _(tensor, index):
    input_size = collections.deque(tensor.size())
    for val in index:
        if val is None:
            output_size.append(1)
        elif isinstance(val, slice) and repr(val) == "slice(None, None, None)":
            output_size.append(input_size.popleft())  # ← fails on 0D tensor
```

**Problem:** When `tile_b` has `block_size=1`, `starts = seq_offsets[tile_b]` may be
traced as a 0D scalar during fake-mode propagation. Then `starts[:, None]` tries to
consume a dimension from an empty deque.

**Root cause:** The tile indexing `seq_offsets[tile_b]` collapses the dimension when
the tile has exactly one element (block_size=1), producing a scalar rather than a
1-element 1D tensor.

**Fix options:**
1. Ensure tile indexing always preserves a dimension (even with block_size=1), so
   `seq_offsets[tile_b]` returns shape `[1]` not `[]`.
2. Handle the 0D case in the subscript handler: allow `None` indexing on scalars
   (PyTorch supports `scalar_tensor[None]` → shape `[1]`).

### Blocker 3: `tile_q.end` not supported on jagged tiles

**Severity:** Medium-high

**File:** `helion/language/tile_ops.py:89-97, 122-151`

```python
def _disable_flatten_get_tile(tile):
    index = env.get_block_id(tile)
    assert index is not None          # ← fails for jagged tile
    ...

@_decorators.codegen(tile_end, "common")
def _(state):
    # Assumes scalar end: offset + block_size
    # Jagged tiles have per-lane ends (one per parent element)
```

**Problem:** The causal attention inner loop needs `hl.tile(0, tile_q.end)`.
`tile_end` calls `_disable_flatten_get_tile` which looks up the tile's block_id via
`env.get_block_id(tile)`. For jagged tiles, this lookup returns `None` because the
jagged tile's proxy doesn't resolve to a registered block_id.

Even if the lookup succeeded, the codegen assumes `.end` is a scalar
(`offset + block_size`), but a jagged tile's end is per-lane — each parent element
has a different sequence length.

**Fix options:**
- Add a `jagged_tile_end` API that returns the max end across lanes (i.e.,
  `offset + block_size` clamped by `amax(lengths)`). This would be correct as an
  upper bound for the causal kv loop — the causal mask handles the per-lane
  boundaries.
- Or: make `tile_end` aware of jagged tiles by checking the block_id type and
  emitting `tl.minimum(offset + block_size, amax)`.

**Workaround:** Use `hl.tile(0, max_seq_len)` for the kv loop with explicit causal +
jagged masks. Less efficient (iterates past actual sequence end) but semantically
correct.

### Blocker 4: Mixed 2D-expression + tile indexing on device tensors

**Severity:** Medium-high

**Files:** `helion/_compiler/type_propagation.py:462-547`, `indexing_strategy.py`

**Problem:** The expression `q[q_row, tile_h, :]` uses:
- `q_row`: a 2D computed expression `[tile_b, tile_q]` (from `starts[:, None] + tile_q.index[None, :]`)
- `tile_h`: a 1D tile variable
- `:`: full last dimension

The type propagation's `_device_indexing_size` partially supports multi-dimensional
tensor indexers, but the codegen (`SubscriptIndexing.create`) for combining a 2D
tensor indexer with a tile index and a full-dim slice is untested. The pointer
arithmetic generation would need to broadcast the 2D indexer dimensions with the
tile dimensions.

**Workaround:** Use `hl.load` with flat 1D indexing:
```python
q_flat = q.view(-1)
q_blk = hl.load(q_flat, [q_row[:,:,None] * stride_seq + tile_h.index[None,None,:] * dimV + hl.arange(dimV)[None,None,None,:]])
```
This works but is verbose and produces 4D tensors with size-1 dimensions that
complicate downstream matmul/permute operations.

## Recommended Fix Order

1. **Blocker 1** (matmul ndim) — simple 5-line fix, unblocks many patterns
2. **Blocker 2** (subscript 0D) — focused fix in one handler
3. **Blocker 3** (tile_q.end) — needs design decision on jagged tile semantics
4. **Blocker 4** (mixed indexing) — can be worked around with `hl.load`; fix later

With blockers 1-3 fixed and blocker 4 worked around via `hl.load`, a jagged_tile
HSTU kernel should be compilable, though the generated code would be more verbose
than the original due to flat indexing and extra dimensions.

## Appendix: The jagged_tile Parent Constraint

`hl.jagged_tile` enforces that any expression involving `tile_q.index` must also
involve the parent tile (`tile_b`). This means you cannot write:

```python
causal = tile_q.index[:, None] > tile_kv.index[None, :]  # ERROR: no parent
```

Instead, use global positions (which naturally include the parent):

```python
q_global = starts[:, None] + tile_q.index[None, :]     # includes tile_b via starts
kv_global = starts[:, None] + tile_kv.index[None, :]
causal = q_global[:, :, None] > kv_global[:, None, :]   # OK: involves parent
```

This is semantically equivalent since `starts` cancels out in the comparison, but
satisfies the compiler constraint.
