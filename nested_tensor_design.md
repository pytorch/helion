# Native NestedTensor Support in Helion

## Problem

Every jagged kernel in Helion requires users to manually:
1. Unwrap `nested_tensor._values` and `nested_tensor._offsets` before calling the kernel
2. Compute `starts`, `ends`, `lengths`, `max_length` from offsets (4 lines, repeated in every kernel)
3. Construct flat indices into the packed buffer (`starts[:, None] + tile_k.index[None, :]`)
4. Construct validity masks (`tile_k.index[None, :] < lengths[:, None]`)
5. Pass masks to `hl.load(..., extra_mask=...)`

This is error-prone, verbose, and obscures the actual algorithm.

## Design Principle

**No new user-facing types.** The user passes a `torch.nested` tensor (PyTorch's standard NestedTensor with `layout=torch.jagged`) and indexes it with tiles. The compiler handles offset arithmetic and masking internally — the same way it already handles dense tensor indexing.

## PyTorch NestedTensor Background

PyTorch core provides `NestedTensor` with `layout=torch.jagged`:
- It's a real `torch.Tensor` subclass with autograd, compile support, and shape semantics
- Stores `_values` (packed data) and `_offsets` (cumulative offsets)
- Shape reports a symbolic `*` for the jagged dimension, e.g. `(B, *, D)`
- TorchRec is migrating from its own `JaggedTensor`/`KeyedJaggedTensor` to this
- This is the canonical representation going forward

## User-Facing API

### Kernel Signature

The user just writes `torch.Tensor` in the type annotation. The compiler detects that the actual argument is nested at specialization time.

```python
@helion.kernel()
def jagged_sum(x: torch.Tensor) -> torch.Tensor:
    B = x.size(0)       # batch size (concrete)
    M = x.size(2)       # trailing dense dim (concrete)
    out = torch.zeros([B, M], dtype=x.dtype, device=x.device)

    for tile_b in hl.tile(B):
        for tile_m in hl.tile(M):
            acc = hl.zeros([tile_b, tile_m], dtype=torch.float32)
            for tile_k in hl.tile(x.jagged_size(tile_b)):
                acc = acc + x[tile_b, tile_k, tile_m]
            out[tile_b, tile_m] = acc.to(out.dtype)
    return out

# Caller passes a NestedTensor directly:
result = jagged_sum(nested_tensor)
```

### Jagged Size

`x.jagged_size(tile)` returns the dynamic extent of the jagged dimension for the given batch tile. The compiler lowers this to:

```python
# Generated code (what the compiler produces):
lengths = offsets[tile_b.index + 1] - offsets[tile_b]
max_length = lengths.amax()
# hl.tile(max_length) uses max_length as the loop bound
```

When called without a tile argument, `x.jagged_size()` returns the global `max_length` across all rows (useful for output allocation).

### Indexing

`x[tile_b, tile_k, tile_m]` on a NestedTensor generates the flat-index + mask code that users currently write by hand:

```python
# Generated code (what the compiler produces):
starts = offsets[tile_b]
flat_idx = starts[:, None] + tile_k.index[None, :]
flat_idx_2d = flat_idx[:, :, None] * M + tile_m.index[None, None, :]
mask = tile_k.index[None, :] < lengths[:, None]
result = hl.load(values_flat, [flat_idx_2d], extra_mask=mask[:, :, None])
```

The user never sees this. They just write `x[tile_b, tile_k, tile_m]`.

### Stores

Writing to a NestedTensor output works symmetrically:

```python
out[tile_b, tile_k, tile_m] = values
# Generates: hl.store(out_values_flat, [flat_idx_2d], values, extra_mask=mask[:, :, None])
```

## Examples: Before and After

### jagged_sum

**Before (25 lines of kernel logic):**
```python
@helion.kernel()
def jagged_sum(x_data: torch.Tensor, x_offsets: torch.Tensor) -> torch.Tensor:
    M = x_data.shape[1]
    num_rows = x_offsets.size(0) - 1
    out = torch.zeros([num_rows, M], dtype=x_data.dtype, device=x_data.device)
    x_flat = x_data.view(-1)
    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts
        max_nnz = nnz.amax()
        for tile_m in hl.tile(M):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)
            for tile_k in hl.tile(0, max_nnz):
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = base_indices[:, :, None] * M + tile_m.index[None, None, :]
                row_mask = tile_k.index[None, :] < nnz[:, None]
                x_slice = hl.load(x_flat, [flat_indices], extra_mask=row_mask[:, :, None])
                row_sums = row_sums + x_slice.sum(dim=1)
            out[tile_b, tile_m] = row_sums
    return out

result = jagged_sum(nt._values, nt._offsets)
```

**After (12 lines of kernel logic):**
```python
@helion.kernel()
def jagged_sum(x: torch.Tensor) -> torch.Tensor:
    B, M = x.size(0), x.size(2)
    out = torch.zeros([B, M], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(B):
        for tile_m in hl.tile(M):
            acc = hl.zeros([tile_b, tile_m], dtype=torch.float32)
            for tile_k in hl.tile(x.jagged_size(tile_b)):
                acc = acc + x[tile_b, tile_k, tile_m].sum(dim=1)
            out[tile_b, tile_m] = acc.to(out.dtype)
    return out

result = jagged_sum(nt)
```

### jagged_dense_add

**Before:**
```python
@helion.kernel()
def jagged_dense_add(x_data, x_offsets, y):
    num_rows = y.size(0)
    out = torch.zeros_like(y)
    for tile0 in hl.tile(num_rows):
        starts = x_offsets[tile0]
        ends = x_offsets[tile0.index + 1]
        nnz = ends - starts
        max_nnz = nnz.amax()
        for tile1 in hl.tile(0, max_nnz):
            x_slice = hl.load(x_data,
                [starts[:, None] + tile1.index[None, :]],
                extra_mask=tile1.index[None, :] < nnz[:, None])
            out[tile0, tile1] = y[tile0, tile1] + x_slice
        for tile1 in hl.tile(max_nnz, out.size(1)):
            out[tile0, tile1] = y[tile0, tile1]
    return out

result = jagged_dense_add(nt._values, nt._offsets, dense)
```

**After:**
```python
@helion.kernel()
def jagged_dense_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B, N = y.size(0), y.size(1)
    out = y.clone()
    for tile_b in hl.tile(B):
        for tile_col in hl.tile(x.jagged_size(tile_b)):
            out[tile_b, tile_col] = out[tile_b, tile_col] + x[tile_b, tile_col]
    return out

result = jagged_dense_add(nt, dense)
```

### jagged_dense_bmm

**Before (35 lines):**
```python
@helion.kernel()
def jagged_dense_bmm(seq_offsets, jagged, dense, bias):
    L, D = jagged.shape
    B, D, K = dense.shape
    dtype = torch.promote_types(jagged.dtype, dense.dtype)
    jagged = jagged.view(-1)
    output = torch.empty((L, K), dtype=dtype, device=device).view(-1)
    for tile_b in hl.tile(B):
        starts = seq_offsets[tile_b]
        ends = seq_offsets[tile_b.index + 1]
        seq_len = ends - starts
        max_seq_len = seq_len.amax()
        for tile_len in hl.tile(0, max_seq_len):
            mask = tile_len.index[None, :] < seq_len[:, None]
            jagged_indices = starts[:, None] + tile_len.index[None, :]
            for tile_k in hl.tile(0, K):
                acc = hl.zeros([tile_b, tile_len, tile_k], ...)
                for tile_d in hl.tile(0, D):
                    jagged_data = hl.load(jagged,
                        [jagged_indices[:, :, None] * D + tile_d.index[None, None, :]],
                        extra_mask=mask[:, :, None] & (tile_d.index[None, None, :] < D))
                    dense_data = dense[tile_b, tile_d, tile_k]
                    acc = acc + torch.matmul(jagged_data, dense_data)
                if bias is not None:
                    acc = acc + bias[tile_b, tile_k].unsqueeze(1)
                hl.store(output,
                    [jagged_indices[:, :, None] * K + tile_k.index[None, None, :]],
                    acc, extra_mask=mask[:, :, None])
    return output.reshape(L, K)
```

**After (18 lines):**
```python
@helion.kernel()
def jagged_dense_bmm(jagged: torch.Tensor, dense: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    B = jagged.size(0)
    D = jagged.size(2)
    K = dense.size(2)
    dtype = torch.promote_types(jagged.dtype, dense.dtype)
    # Output is a new NestedTensor with same jagged structure but different trailing dim
    output = torch.nested.nested_tensor_from_jagged(
        torch.empty(jagged._values.size(0), K, dtype=dtype, device=jagged.device),
        jagged._offsets)
    for tile_b in hl.tile(B):
        for tile_len in hl.tile(jagged.jagged_size(tile_b)):
            for tile_k in hl.tile(K):
                acc = hl.zeros([tile_b, tile_len, tile_k], dtype=dtype)
                for tile_d in hl.tile(D):
                    acc = acc + torch.matmul(jagged[tile_b, tile_len, tile_d], dense[tile_b, tile_d, tile_k])
                if bias is not None:
                    acc = acc + bias[tile_b, tile_k].unsqueeze(1)
                output[tile_b, tile_len, tile_k] = acc
    return output
```

### grouped_gemm (maskless pattern)

The maskless pattern (one group at a time, exact bounds) works naturally:

**Before:**
```python
@helion.kernel(static_shapes=False)
def grouped_gemm(A_packed, B, group_offsets):
    G = group_offsets.size(0) - 1
    out = torch.empty(total_M, N, ...)
    for g in hl.grid(G):
        start = group_offsets[g]
        end = group_offsets[g + 1]
        M_g = end - start
        if M_g != 0:
            for tile_m, tile_n in hl.tile([M_g, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(K):
                    acc = torch.addmm(acc, A_packed[start + tile_m.index, tile_k], B[tile_k, tile_n])
                out[start + tile_m.index, tile_n] = acc.to(out.dtype)
```

**After:**
```python
@helion.kernel(static_shapes=False)
def grouped_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    G, N, K = A.size(0), B.size(1), B.size(0)
    # Output is a new NestedTensor with same structure, different trailing dim
    output = torch.nested.nested_tensor_from_jagged(
        torch.empty(A._values.size(0), N, dtype=..., device=A.device), A._offsets)
    for g in hl.grid(G):
        for tile_m, tile_n in hl.tile([A.jagged_size(g), N]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(K):
                acc = torch.addmm(acc, A[g, tile_m, tile_k], B[tile_k, tile_n])
            output[g, tile_m, tile_n] = acc.to(output.dtype)
    return output
```

Note: when `g` is a scalar from `hl.grid()` (not a tile), `A.jagged_size(g)` returns the exact length for that single group — no masking needed. The compiler sees the loop bound matches the jagged extent exactly and skips mask generation.

## Compiler Implementation

### 1. NestedTensorType in type_propagation.py

When the compiler sees a kernel argument where `arg.is_nested` is true:
- Create a `NestedTensorType` that tracks: values dtype/shape, offsets tensor, ragged_idx
- At the kernel boundary, decompose the NestedTensor into `_values` and `_offsets` as separate Triton kernel arguments
- The user-visible type is still `torch.Tensor`

### 2. Lowering `x.jagged_size(tile)`

```python
# Generates:
starts = offsets[tile.index]
ends = offsets[tile.index + 1]
lengths = ends - starts
max_length = lengths.amax()
```

When `tile` is a scalar (from `hl.grid`), skip the `.amax()` — the length is exact.

### 3. Lowering `x[tile_b, tile_k, tile_m]`

When the compiler sees a subscript on a NestedTensor:
1. Identify which index corresponds to the jagged dimension (from `ragged_idx`)
2. Generate flat-index computation from offsets
3. Generate validity mask for the jagged dimension
4. Emit `hl.load(values_flat, [flat_indices], extra_mask=jagged_mask)`

For stores, emit `hl.store(values_flat, [flat_indices], data, extra_mask=jagged_mask)`.

### 4. Specialization

NestedTensors specialize on:
- Values dtype and device
- Number of dense dimensions and their sizes
- Ragged dimension index
- NOT on specific offsets values (those are runtime data)

## Autotuning Over Jagged Iteration Strategies

### The Question

Given the same user code:
```python
for tile_b in hl.tile(B):
    for tile_k in hl.tile(x.jagged_size(tile_b)):
        ... = x[tile_b, tile_k, ...]
```

Can the compiler autotune between different lowering strategies for the jagged loop + indexing?

### Strategy 1: Masked Batched (current default)

Multiple batch rows processed together. The jagged loop uses `max_length` across the batch tile as the bound, with per-row masks.

```
tile_b processes [row0, row1, row2, row3] together
tile_k iterates up to max(len0, len1, len2, len3)
mask = tile_k < [len0, len1, len2, len3]  (per-row)
```

- Pros: Good memory coalescing, high occupancy, amortizes loop overhead
- Cons: Wasted work when lengths vary widely within a batch tile
- Best when: density (avg_length / max_length within tile) is high

### Strategy 2: Maskless Per-Row (like grouped_gemm)

Each batch row processed independently with exact bounds. No masking overhead.

```
for b in hl.grid(B):         # one row at a time
    for tile_k in hl.tile(x.jagged_size(b)):  # exact length
        ...                    # no mask needed
```

- Pros: Zero wasted work, no mask overhead
- Cons: Less coalescing, potentially lower occupancy, more kernel launches or serial iteration
- Best when: lengths vary widely, or per-row work is large enough to fill the GPU

### Strategy 3: Persistent Work-Stealing

Fixed number of workers (= SM count) pull tiles from a global worklist.

- Pros: Perfect load balance for highly skewed distributions
- Cons: Complex scheduling, overhead for small workloads
- Best when: extreme length imbalance (e.g., 1 row has 10K elements, rest have 10)

### How Autotuning Integrates

The key insight: **the loop structure determines the strategy, and the loop structure is what gets autotuned.**

When the compiler sees `hl.tile(x.jagged_size(tile_b))` where `tile_b` is a tile (not a scalar), it knows the user wants to iterate over a jagged dimension in a batched context. It can add a tunable:

```python
# In device_ir.py, when lowering a jagged tile loop:
if is_jagged_tile_loop(loop):
    env.config_spec.user_defined_tunables["jagged_iteration"] = EnumFragment(
        choices=("masked_batched", "maskless_per_row")
    )
```

Then during code generation:
- **`masked_batched`**: Lower as-is — `hl.tile(max_length)` with mask. The batch tile size for `tile_b` is autotuned as usual.
- **`maskless_per_row`**: Rewrite the outer tile to `hl.grid(B)` and the inner tile to `hl.tile(exact_length)`. No mask generated.

The autotuner benchmarks both variants and picks the faster one. No heuristics — just measurement.

**Persistent work-stealing** is a more invasive transformation and probably shouldn't be auto-selected initially. Users who need it can write the persistent pattern explicitly (as in `grouped_gemm_jagged_persistent`). It can be added as a third autotuning choice later.

### What the User Sees

Nothing. The user writes:
```python
for tile_b in hl.tile(B):
    for tile_k in hl.tile(x.jagged_size(tile_b)):
        acc = acc + x[tile_b, tile_k, tile_m]
```

The autotuner decides whether to batch rows (masked) or process them individually (maskless). The user can override:

```python
@helion.kernel(jagged_strategy="maskless_per_row")
def my_kernel(x: torch.Tensor): ...
```

But most users never need to.

## Summary

| Aspect | This design |
|--------|-------------|
| New user-facing types | None — uses `torch.Tensor` (NestedTensor) |
| New user-facing APIs | `x.jagged_size(tile)` only |
| Indexing | `x[tile_b, tile_k, tile_m]` — same syntax as dense |
| Compiler work | NestedTensorType + jagged indexing lowering + jagged size lowering |
| Autotuning | Transparent — compiler generates masked vs maskless variants, autotuner picks |
| PyTorch interop | Perfect — it IS a PyTorch tensor |
| TorchRec interop | Direct — TorchRec is migrating to NestedTensor |
| Lines of code saved | ~50% per jagged kernel |
| Breaking changes | None |
