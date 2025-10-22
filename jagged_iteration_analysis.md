# Jagged Iteration Patterns in Helion Example Kernels

## Kernels that uses jagged tensor: jagged_dense_add, jagged_mean, moe_matmul_ogs, jagged_hstu_attn, jagged_softmax, jagged_sum, grouped_gemm

## Jagged Dense Add (`examples/jagged_dense_add.py`)
- Per-row tile over `num_rows`; compute `nnz` span and its tile-wise maximum (`max_nnz`).
- First loop `hl.tile(0, max_nnz)` iterates masked jagged columns using `extra_mask=tile1.index[None, :] < nnz[:, None]` to gate loads.
- Second loop over the dense tail (`hl.tile(max_nnz, out.size(1))`) writes the remaining columns without masks, illustrating a hybrid masked + dense traversal strategy.

## Jagged Mean (`examples/jagged_mean.py`)
- Tiles rows, then features (`max_M`), while keeping a per-row jagged span (`max_nnz`).
- Flattens storage to 1D for gather simplicity; computes `flat_indices` by combining row offsets and feature indices.
- Intersects two jagged dimensions (row length mask and feature-count mask) before each load; accumulates and normalizes only where both masks are valid.

## Mixture-of-Experts Matmul OGS (`examples/moe_matmul_ogs.py`)
- Iterates experts with `hl.grid(E)` and tiles `[max_T_per_expert, N]` regardless of the exact token count.
- Uses `token_valid = local_token_offsets < num_tokens` and `torch.where` to rewrite invalid offsets prior to gathers.
- Writes outputs conditionally with an expanded 2D validity mask, so inactive lanes keep their previous values.

## Jagged HSTU Attention (`examples/jagged_hstu_attn.py`)
- Tiles over batches, heads, and sequence positions up to `max_seq_len`; skips tiles whose start exceeds the real sequence length.
- Inside each query tile, keys/values are iterated with `hl.tile(0, tile_q.end)` so the loop range shrinks as the causal window advances.
- Applies combined masks for sequence length and causal ordering when accumulating attention scores, effectively managing two jagged axes (length and causal prefix).

## Jagged Softmax (`examples/jagged_softmax.py`)
- Similar to `jagged_mean`, but performs two full passes over the jagged dimension: first to compute log-sum-exp statistics, second to normalize.
- Each pass tiles `[max_seqlen]` with row-length masks, while feature columns are tiled separately; flattened addressing keeps mask composition scalar.

## Jagged Sum (`examples/jagged_sum.py`)
- Mirrors the data-dependent tiling pattern of `jagged_mean` but omits feature masking (single jagged axis).
- Accumulates sums across `hl.tile(0, max_nnz)` with a row-length mask before writing dense feature tiles.

## Grouped GEMM (`examples/grouped_gemm.py`)
- `grouped_gemm_jagged`: loops groups explicitly; per-group jagged extent `M_g` feeds directly into `hl.tile([M_g, N])`, so no mask is required along the row dimension.
- `grouped_gemm_jagged_persistent`: persistent workers stride over per-group tile grids. Uses index arithmetic (`row_idx`, `col_idx`) plus `rows_valid`/`cols_valid` masks to guard loads/stores for partial tiles.

## Shared Techniques
- Flattened addressing for jagged+feature indexing (`jagged_mean`, `jagged_sum`, `jagged_softmax`).
- Hybrid masked + dense loops to reduce masking work (`jagged_dense_add`).
- Explicit token/row validity masks to avoid out-of-bound gathers (`moe_matmul_ogs`, `grouped_gemm_jagged_persistent`).
- Dual jagged-dimension handling when data has independent sparsity patterns (`jagged_mean`, `_helion_jagged_attention_kernel`).

## Proposal: Auto-tuning Jagged Tensor Abstraction

### Problem Statement
Users currently must manually:
1. Choose iteration strategy (masked, flattened, skip-based, etc.)
2. Construct validity masks correctly for each jagged dimension
3. Handle edge cases and partial tiles
4. Optimize for their specific data distribution without guidance

This creates a high barrier to entry and leads to suboptimal kernel implementations.

### Proposed Solution: `JaggedTensor` Type with Compiler Auto-tuning

#### 1. First-Class Jagged Tensor Type

Introduce a `JaggedTensor` data structure that encapsulates:

```python
class JaggedTensor:
    """
    A tensor with one or more jagged (variable-length) dimensions.

    Storage formats:
    - CSR-like: values (1D), offsets (per outer dimension)
    - Padded: dense tensor + lengths array
    - Nested: list of variable-length tensors (for analysis/conversion)
    """
    values: Tensor           # Flattened data storage
    offsets: Tensor          # Cumulative offsets (CSR-style) OR
    lengths: Tensor          # Per-element lengths (if padded format)
    max_length: int          # Maximum length across jagged dimension
    total_elements: int      # Total number of valid elements

    # Metadata for autotuning
    jagged_dims: List[int]   # Which dimensions are jagged (e.g., [1] for ragged rows)
    sparsity: float          # avg_length / max_length (density metric)
    variance: float          # Variance in lengths (uniformity metric)
    format: str              # 'csr', 'padded', 'nested'
```

**Key design principles:**
- Multiple storage formats for different access patterns
- Rich metadata enables compiler heuristics and profiling-guided optimization
- Interoperable with PyTorch/Triton ecosystem

#### 2. Iteration Strategy Taxonomy

The compiler can autotune over these strategies:

| Strategy | Masking? | Best For | Characteristics |
|----------|----------|----------|-----------------|
| **Data-Dependent Tiling** ⭐ | **NO** | Per-group/per-row processing | `hl.tile(M_g)` - Perfect efficiency, zero masking overhead (grouped_gemm.py:93) |
| **Dense Masked Batched** | Yes | High density (>0.7), batch processing | `hl.tile(max_nnz)` - Process multiple rows together, wasted lanes (jagged_dense_add.py:73) |
| **Tile Skipping** | Yes | Low density (<0.3), clustered invalids | Early-exit tiles, control flow overhead (jagged_hstu_attn.py:19) |
| **Hybrid Dense-Sparse** | Partial | Mixed regions | Dense iteration for common case, masking for tail |
| **Persistent Dynamic** | Yes | Highly variable, load balancing | Work stealing across SMs (grouped_gemm_persistent) |

**⭐ Maskless data-dependent tiling is optimal when feasible** - it avoids all masking overhead by using exact jagged extents in `hl.tile(M_i)` or `hl.grid()` loops.

#### 3. Compiler Auto-tuning Framework

**Approach: Leverage Helion's Existing Autotuner**

Helion already has a powerful autotuning infrastructure with:
- `ConfigSpec` with user-definable tunables (EnumFragment, IntegerFragment, etc.)
- Multiple search strategies (PatternSearch, DifferentialEvolution, RandomSearch)
- Benchmarking infrastructure that measures actual performance
- Persistent caching of optimal configurations

**For JaggedTensor, we simply:**

1. **Detect JaggedTensor usage** (compile-time):
   ```python
   # Simple check: does kernel have JaggedTensor argument?
   if has_jagged_tensor_arg(kernel_func):
       env.config_spec.user_defined_tunables["jagged_strategy"] = EnumFragment(
           choices=("maskless", "masked", "persistent_dynamic")
       )
   ```

2. **Generate 3 kernel variants** (compile-time):
   - Each variant uses one of the three core iteration strategies
   - All compiled variants use the SAME user code

3. **Autotune on first call** (runtime):
   - Helion's autotuner benchmarks all strategies
   - Measures actual wall-clock time
   - NO heuristics, NO cost models - just real measurements

4. **Cache the winner** (runtime):
   - Best strategy cached by (sparsity, variance, shape)
   - Future calls with similar JaggedTensors reuse cached choice
   - No re-benchmarking unless input characteristics change significantly

**User Code (unchanged regardless of strategy):**

```python
@helion.kernel()
def jagged_kernel(jt: JaggedTensor, dense: Tensor) -> Tensor:
    out = torch.zeros_like(dense)
    # User writes standard hl.tile() loops
    for tile_row in hl.tile(jt.num_rows):
        for tile_col in hl.tile(jt.max_length):
            # Simple indexing - compiler handles the rest
            out[tile_row, tile_col] = jt[tile_row, tile_col] + dense[tile_col]
    return out

# First call: autotuner benchmarks maskless vs masked vs persistent
result1 = jagged_kernel(jt1, dense1)  # ~10ms extra for autotuning

# Subsequent calls: uses cached best strategy
result2 = jagged_kernel(jt2, dense2)  # Fast! Uses cached choice
```

**Key Benefits:**
- No guesswork - autotuner measures real performance
- Works across different GPUs (H100, MI300, etc.)
- Adapts to actual workload characteristics
- Reuses all existing Helion autotuning infrastructure

**The Three Strategies (All Generated and Benchmarked):**

The compiler generates code for all three strategies and lets the autotuner measure which is fastest:

```python
# User-level code (what user writes) - SAME for all strategies
for tile_row in hl.tile(jt.num_rows):
    for tile_col in hl.tile(jt.max_length):
        out[tile_row, tile_col] = jt[tile_row, tile_col] + dense[tile_col]

# Strategy 1: MASKLESS data-dependent tiling (like grouped_gemm.py:93)
# Used when: each row processed independently, known sizes at runtime
for row_id in hl.grid(jt.num_rows):
    M_row = jt.lengths[row_id]  # Exact length for this row
    if M_row > 0:
        # NO MASKING! Tile exactly M_row columns
        for tile_col in hl.tile(M_row):
            # No extra_mask needed - tile range is exact valid data
            values = jt.values[jt.offsets[row_id] + tile_col.index]
            out[row_id, tile_col] = values + dense[tile_col]

# Strategy 2: Dense masked batched (like jagged_dense_add.py:73)
# Used when: batch processing rows together, high density
for tile_row in hl.tile(jt.num_rows):
    lengths = jt.lengths[tile_row]
    max_len = lengths.amax()  # Max across batch
    for tile_col in hl.tile(0, max_len):
        # Compiler auto-inserts extra_mask for invalid elements
        values = hl.load(jt.values,
                        [jt.offsets[tile_row][:, None] + tile_col.index[None, :]],
                        extra_mask=tile_col.index[None, :] < lengths[:, None])
        out[tile_row, tile_col] = values + dense[tile_col]

# Strategy 3: Persistent dynamic (like grouped_gemm_jagged_persistent)
# Used when: workloads are highly skewed or require intra-kernel work stealing
num_workers = hl.device_num_sms()
for worker_id in hl.grid(num_workers):
    tile_idx = worker_id
    total_tiles = jt.total_tiles
    while tile_idx < total_tiles:
        tile = jt.tile_metadata[tile_idx]
        # Compiler injects dynamic bounds + masks per tile
        values = hl.load(jt.values, [tile.rows, tile.cols], extra_mask=tile.mask)
        out[tile.rows, tile.cols] = values + dense[tile.cols]
        tile_idx += num_workers

```

The persistent variant still uses existing `hl.grid` loops; the runtime forces
`pid_type` to a `persistent_*` mode so only `num_workers` threads launch and each
worker strides through the global tile worklist with the manual `while` loop above.

**Key insight:** Strategy 1 (maskless) often has **zero masking overhead** and perfect efficiency. Strategy 2 (masked) can be faster for batched processing with high density. Strategy 3 (persistent) shines when lengths are extremely imbalanced and SM-level load balancing matters. **The autotuner measures all three and picks the fastest** - no guessing!

#### 4. Key Design Principle: No New Language APIs Required

**The proposal reuses ALL existing Helion APIs** - `hl.tile()`, `hl.grid()`, `hl.load()`, `hl.arange()`, etc.

The compiler automatically recognizes when you're working with a `JaggedTensor` type and:
- Inserts appropriate `extra_mask` parameters into existing `hl.load()` calls
- Selects optimal iteration strategies while preserving the same `hl.tile()` loop structure
- Handles index transformations (offset calculations, flattening) transparently

**This means:**
- Users write standard `hl.tile()` loops, just like they do now
- No new `hl.jagged_*()` functions to learn
- Existing Helion kernels can be incrementally adopted
- Zero breaking changes to the language

#### 5. API Design

**Construction:**
```python
# From list of tensors (irregular)
jt = hl.JaggedTensor.from_list([tensor1, tensor2, ...])

# From padded tensor + lengths
jt = hl.JaggedTensor.from_padded(padded_data, lengths)

# From CSR-like format
jt = hl.JaggedTensor.from_csr(values, offsets)
```

**Operations:**
```python
# High-level iteration using EXISTING hl.tile/hl.grid (compiler handles strategy)
@helion.kernel()
def kernel(jt: JaggedTensor):
    for tile_i in hl.tile(jt.num_rows):     # Use existing hl.tile()
        for tile_j in hl.tile(jt.max_length): # Use existing hl.tile()
            # Compiler recognizes JaggedTensor and inserts masks automatically
            x = jt[tile_i, tile_j]  # Safe access, auto-masked

# Manual strategy override (for experts)
@helion.kernel(jagged_strategy="persistent_dynamic")
def kernel_expert(jt: JaggedTensor):
    # User can still override if they have domain knowledge
    for tile_i in hl.tile(jt.num_rows):
        for tile_j in hl.tile(jt.max_length):
            x = jt[tile_i, tile_j]
```

**Auto-tuning hints:**
```python
# Provide runtime hints for better autotuning
jt = jt.with_hint(
    distribution="uniform",     # uniform, exponential, bimodal
    access_pattern="row_major", # row_major, random, streaming
    reuse="high"                # cache hint
)
```

#### 6. Implementation Strategy

**Stage 1: Core Infrastructure** ✅ **COMPLETE**
- ✅ Implement `JaggedTensor` type with multiple storage formats
- ✅ Add CSR ↔ padded ↔ nested format conversions
- ✅ Compute and store metadata (max_length, sparsity, variance)

**Files Created:**
- `helion/language/jagged_tensor.py` - Core JaggedTensor class
- `examples/jagged_dense_add.py` - Updated with JaggedTensor usage

**Stage 2: Compiler Type Integration** ✅ **COMPLETE**
- ✅ Add JaggedTensorType to Helion's type system (type_propagation.py)
- ✅ Fake tensor conversion for compilation (compile_environment.py)
- ✅ Kernel specialization and caching (kernel.py)
- ✅ Direct JaggedTensor kernel arguments (no wrapper needed)

**Files Modified:**
- `helion/_compiler/type_propagation.py` - Added JaggedTensorType class
- `helion/_compiler/compile_environment.py` - Added to_fake support
- `helion/runtime/kernel.py` - Added specialization key generation
- `helion/language/jagged_tensor.py` - Added __getitem__ for compiler-generated code

**Achievement:** Users can now pass JaggedTensor directly to kernels:
```python
@helion.kernel()
def my_kernel(jt: hl.JaggedTensor, dense: torch.Tensor):
    # JaggedTensor passed directly - no wrapper needed! ✅
    for tile0 in hl.tile(jt.num_rows):
        # Access jt.offsets, jt.values directly
```

**Stage 3: Multi-Variant Autotuning** 🚧 **IN PROGRESS**

**Approach:** Leverage Helion's existing autotuning infrastructure (EnumFragment + benchmarking)

**Detection (Simple):**
- Check if kernel has `JaggedTensor` argument → add tunable
- No complex pattern matching - just presence of JaggedTensor type

**Implementation Steps:**
1. **Detect JaggedTensor usage** in `lower_to_device_ir()`:
   ```python
   def has_jagged_tensor_arg(device_ir: DeviceIR) -> bool:
       # Check if any kernel argument is JaggedTensor
       for arg_type in func.arg_types:
           if isinstance(arg_type, JaggedTensorType):
               return True
       return False
   ```

2. **Add tunable to ConfigSpec** when detected:
   ```python
   if has_jagged_tensor_arg(device_ir):
       env.config_spec.user_defined_tunables["jagged_strategy"] = EnumFragment(
           choices=("maskless", "masked", "persistent_dynamic")
       )
   ```

3. **Multi-variant codegen** - check config during code generation:
   ```python
   strategy = state.config.user_defined_tunables.get("jagged_strategy", "masked")

   if strategy == "maskless":
       # Generate: for tile in hl.tile(0, nnz[i]) - exact bounds per row
   elif strategy == "masked":
       # Generate: for tile in hl.tile(0, max_nnz) + extra_mask - batched
   elif strategy == "persistent_dynamic":
       # Generate: persistent PID loop with dynamic tile scheduling
       state.config.pid_type = choose_persistent_pid_type()
   ```

   Here `choose_persistent_pid_type()` can consult hardware metadata to flip between
   `persistent_blocked` and `persistent_interleaved` depending on whether worker
   striding or interleaving yields better cache reuse.

4. **Autotuner benchmarks all strategies** on first call, caches the winner

**Key Insight:** We're NOT doing heuristic-based selection. The autotuner measures actual performance and picks the fastest variant.

**Files to Modify:**
- `helion/_compiler/device_ir.py` - Add detection + tunable registration
- `helion/_compiler/inductor_lowering.py` or similar - Multi-variant codegen
- Leverage existing: `helion/autotuner/benchmarking.py`, `config_fragment.py`


#### 7. Example: Before & After

**Current (Manual):**
```python
@helion.kernel()
def jagged_dense_add(x_data, x_offsets, dense):
    num_rows = dense.size(0)
    out = torch.zeros_like(dense)
    for tile_row in hl.tile(num_rows):
        starts = x_offsets[tile_row]
        ends = x_offsets[tile_row.index + 1]
        nnz = ends - starts
        max_nnz = nnz.amax()

        # User must manually stitch together flat offsets + masks
        for tile_col in hl.tile(0, max_nnz):
            flat_idx = starts[:, None] + tile_col.index[None, :]
            valid = tile_col.index[None, :] < nnz[:, None]
            jagged_vals = hl.load(x_data, [flat_idx], extra_mask=valid)
            out[tile_row, tile_col] = dense[tile_row, tile_col] + jagged_vals

        # User must remember to copy the dense tail
        for tile_col in hl.tile(max_nnz, out.size(1)):
            out[tile_row, tile_col] = dense[tile_row, tile_col]
    return out
```

**Proposed (Automatic):**
```python
@helion.kernel()
def jagged_dense_add(jt: JaggedTensor, dense):
    out = torch.zeros_like(dense)
    for tile_row in hl.tile(jt.num_rows):
        # Compiler inserts masks + flattened indexing for jagged accesses
        for tile_col in hl.tile(jt.max_length):
            out[tile_row, tile_col] = dense[tile_row, tile_col] + jt[tile_row, tile_col]

        # Compiler still spots the dense tail opportunity
        for tile_col in hl.tile(jt.max_length, out.size(1)):
            out[tile_row, tile_col] = dense[tile_row, tile_col]
    return out
```

#### 8. Concrete Examples: Before & After Transformations

##### 8.1. Maskless Iteration: Transforming `grouped_gemm.py`

**Current implementation (lines 87-102 in `examples/grouped_gemm.py`):**
```python
@helion.kernel(static_shapes=False)
def grouped_gemm_jagged(A_packed, B, group_offsets):
    G = group_offsets.size(0) - 1

    # User manually iterates groups and computes M_g
    for g in hl.grid(G):
        start = group_offsets[g]
        end = group_offsets[g + 1]
        M_g = end - start  # Manual computation
        if M_g != 0:
            # User explicitly tiles with M_g - maskless!
            for tile_m, tile_n in hl.tile([M_g, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(K):
                    # User manually computes indices with start offset
                    a_blk = A_packed[start + tile_m.index, tile_k]
                    b_blk = B[tile_k, tile_n]
                    acc = torch.addmm(acc, a_blk, b_blk)
                # User manually computes output indices
                out[start + tile_m.index, tile_n] = acc.to(out.dtype)
    return out
```

**With JaggedTensor (proposed):**
```python
@helion.kernel()
def grouped_gemm_jagged(A_jt: JaggedTensor, B):
    N = B.size(1)
    out = torch.zeros([A_jt.num_groups, A_jt.max_length, N],
                      dtype=torch.promote_types(A_jt.dtype, B.dtype),
                      device=A_jt.device)

    for tile_g in hl.tile(A_jt.num_groups):
        for tile_m, tile_n in hl.tile([A_jt.max_length, N]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(K):
                acc = torch.addmm(acc,
                                  A_jt[tile_g, tile_m, tile_k],
                                  B[tile_k, tile_n])
            out[tile_g, tile_m, tile_n] = acc.to(out.dtype)
    return out
```

**Benefits:**
- **Zero masking overhead** - compiler uses exact M_g for each group
- Simpler indexing - no manual `start + tile_m.index` calculations
- Cleaner abstraction - algorithm separated from memory layout

##### 8.2. Masked Iteration: Transforming `jagged_mean.py`

**Current implementation (lines 60-106 in `examples/jagged_mean.py`):**
```python
@helion.kernel()
def jagged_mean_kernel(x_data, x_offsets, x_feature_counts, max_M):
    num_rows = x_offsets.size(0) - 1
    out = torch.zeros([num_rows, max_M], dtype=x_data.dtype, device=x_data.device)
    x_flat = x_data.view(-1)

    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts
        max_nnz = nnz.amax()
        feature_counts = x_feature_counts[tile_b]

        for tile_m in hl.tile(max_M):
            feature_valid = tile_m.index < feature_counts[:, None]
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)

            for tile_k in hl.tile(0, max_nnz):
                # User manually computes flat indices
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = base_indices[:, :, None] * max_M + tile_m.index[None, None, :]

                # User manually constructs combined mask
                row_mask = tile_k.index[None, :] < nnz[:, None]
                combined_mask = row_mask[:, :, None] & feature_valid[:, None, :]

                x_slice = hl.load(x_flat, [flat_indices], extra_mask=combined_mask)
                row_sums = row_sums + x_slice.sum(dim=1)

            result = torch.where(nnz[:, None] > 0, row_sums / nnz[:, None], 0.0)
            out[tile_b, tile_m] = torch.where(feature_valid, result, 0.0)
    return out
```

**With JaggedTensor (proposed):**
```python
@helion.kernel()
def jagged_mean_kernel(jt: JaggedTensor):  # Single argument!
    num_rows = jt.num_rows
    max_M = jt.size(1)  # Feature dimension
    out = torch.zeros([num_rows, max_M], dtype=jt.dtype, device=jt.device)

    for tile_b in hl.tile(num_rows):
        for tile_m in hl.tile(max_M):
            row_sums = hl.zeros([tile_b, tile_m], dtype=jt.dtype)

            # Compiler handles: offsets, flat indices, and masking!
            for tile_k in hl.tile(jt.max_length):
                # Simple indexing - compiler inserts all the complex logic
                x_slice = jt[tile_b, tile_k, tile_m]
                row_sums = row_sums + x_slice.sum(dim=1)

            # Lengths accessible as jt.lengths property
            result = torch.where(jt.lengths[tile_b][:, None] > 0,
                               row_sums / jt.lengths[tile_b][:, None], 0.0)
            out[tile_b, tile_m] = result
    return out
```

**Key improvements:**
- 50% fewer lines of code
- No manual offset/index calculations
- No manual mask construction
- Compiler auto-selects between flattened/padded storage strategies
- Same or better performance through auto-tuning

#### 9. Summary: Autotuning Over Multiple Strategies

The `JaggedTensor` abstraction enables the compiler to **generate and benchmark** multiple iteration strategies:

1. **Maskless data-dependent tiling** (like `grouped_gemm.py`)
   - `for g in hl.grid(G): for tile in hl.tile(M_g): ...`
   - Zero masking overhead, perfect efficiency
   - Best for: per-group/per-row processing with independent sizes
   - Compiler generates: exact `hl.tile(jt.lengths[i])` loops

2. **Masked batched iteration** (like `jagged_dense_add.py`, `jagged_mean.py`)
   - `for tile in hl.tile(max_nnz): load(..., extra_mask=...)`
   - Process multiple rows together, some wasted lanes
   - Best for: high-density data, batch processing benefits, coalescing
   - Compiler generates: `hl.load()` with automatic `extra_mask` insertion

3. **Persistent dynamic scheduling** (like `grouped_gemm_jagged_persistent`)
   - Workers stay resident on SMs and pull tiles dynamically
   - Best for: highly skewed jagged lengths, load-imbalanced workloads
   - Compiler generates: persistent PID loops with dynamic tile metadata and masks

**The key insight:** Users write the same high-level code with `hl.tile()`, and the **autotuner benchmarks all strategies** on first call:
- Measures actual wall-clock time on real GPU
- No heuristics or cost models - just empirical performance
- Caches the winner based on (sparsity, variance, shape, GPU)
- Works across different hardware (H100, MI300, etc.)

#### 10. Expected Benefits

1. **Productivity**: 3-5x reduction in code complexity for jagged kernels
   - No manual offset calculations
   - No manual mask construction
   - Same simple code works for all data distributions

2. **Performance**: Empirical autotuning finds optimal strategy
   - Measures actual wall-clock time on target GPU
   - No guessing based on heuristics
   - Within 5-10% of hand-tuned specialist kernels
   - Chooses maskless when it's faster, masked when batching helps, persistent when load balance dominates

3. **Correctness**: Automatic masking/indexing eliminates common bugs
   - Compiler handles offset calculations
   - Automatic mask generation prevents out-of-bounds access

4. **Portability**: Same code, different optimal strategies per GPU
   - Autotuned separately for H100, MI300, future GPUs
   - Adapts to memory bandwidth, SM count, cache hierarchy
   - No manual tuning per architecture

5. **Maintainability**: Algorithm separate from optimization strategy
   - User code is strategy-agnostic
   - Compiler can add new strategies without code changes

This proposal provides a path toward making jagged tensor operations as easy to write as dense operations, while **matching or exceeding** the performance of hand-optimized kernels through empirical autotuning.
