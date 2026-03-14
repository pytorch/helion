# Helion Metal Backend: Apple GPU Code Generation

## Overview

The Metal backend for Helion generates Metal Shading Language (MSL) kernels that run on Apple Silicon GPUs via the MPS device. It uses Apple's Metal Performance Primitives (MPP) `cooperative_tensor` API for matmul and fused attention, and SIMD shuffles with per-reduction threadgroup shared memory for standalone reductions.

The backend supports four kernel types — elementwise, softmax/reduction, matrix multiplication, and fused attention — and includes an autotuner that searches over tile sizes, simdgroup counts, and threadgroup cache usage.

**Key results** (per-call GPU timing, TritonBench methodology):
- **Matmul**: 1.00-1.05x vs MLX at sizes ≥ 512
- **Matmul + ReLU**: 1.05-1.10x vs MLX at sizes ≥ 256 (fused epilogue)
- **RMSNorm**: 0.94-1.06x vs MLX
- **Softmax**: 0.92-1.02x vs MLX
- **LayerNorm**: 0.92-1.00x vs MLX
- **Fused attention (GQA)**: 0.73-0.97x vs MLX
- **Two-layer MLP**: 1.00-1.06x vs MLX at sizes ≥ 128

## Architecture

### End-to-End Compilation Pipeline

```
Helion DSL kernel (Python)
    │
    ▼  torch.fx tracing + decomposition
FX Graph (aten ops: mm, addmm, amax, exp, ...)
    │
    ▼  GraphInterpreter.run_node() — per-node lowering
    │  For each node:
    │    1. lowering = n.meta["lowering"]  (AtenLowering or ReductionLowering)
    │    2. result = lowering.codegen(ctx, node)  — dispatches to backend-specific handler
    │    3. n.meta["codegen"] = result  (AST expression)
    │
    ▼  Metal-specific codegen handlers (aten_lowering.py)
    │  - codegen_mm_metal:     emits _metal_mm(lhs, rhs)     + records MetalMatmulOp
    │  - codegen_addmm_metal:  emits _metal_addmm(acc,l,r)   + records MetalMatmulOp
    │  - codegen_bmm_metal:    emits _metal_bmm(lhs, rhs)    + records MetalMatmulOp
    │  - codegen_baddbmm_metal: emits _metal_baddbmm(acc,l,r) + records MetalMatmulOp
    │  - reduction_expr():     emits _metal_max/sum/min(x,d)  + records MetalReductionOp
    │
Body AST with sentinels (_metal_mm, _metal_max, etc.)
  + MetalMatmulOp / MetalReductionOp structured IR in backend._metal_ops
    │
    ▼  DeviceFunction.codegen_function_call()
    │  Calls backend.build_launcher_args() — classifies kernel, resolves
    │  grid/thread dimensions, scratch sizes from structured IR
    │
    ▼  DeviceFunction.codegen_function_def()
    │  Detects backend.name == "metal", calls backend.generate_msl_function(device_fn)
    │
    ▼  MetalBackend.generate_msl_function()
    │  1. _classify_kernel() from _metal_ops → MetalKernelKind
    │  2. MslWalker(backend, device_fn, env, metal_ops).generate()
    │     Dispatches to _generate_walked / _generate_reduction / _generate_elementwise
    │  3. Wraps MSL in Python function: return (msl_source, kernel_name)
    │
MSL source string
    │
    ▼  default_metal_launcher()  [runtime]
    │  1. metal_kernel() → (msl_source, kernel_name)
    │  2. torch.mps.compile_shader(msl_source) → Metal library (cached)
    │  3. Dispatch with appropriate thread/threadgroup config per kernel kind
    │
GPU dispatch via MPS
```

### Phase 1: FX Tracing and Decomposition

Helion traces the user's Python kernel through `torch.fx`, producing an FX graph of aten-level operations. The Helion compiler decomposes high-level PyTorch ops into primitives (e.g., `torch.nn.functional.softmax` → `amax` + `exp` + `sum` + `div`). The FX graph is then lowered node-by-node via `GraphInterpreter.run_node()` in `inductor_lowering.py`.

For each `call_function` node, the interpreter looks up the registered `Lowering` object from `n.meta["lowering"]` and calls `lowering.codegen(ctx, node)`. The `codegen()` method dispatches to the backend-specific handler registered via `@lowering.register_codegen("metal")`.

### Phase 2: Metal-Specific Lowering (Sentinel + Structured IR)

The Metal backend uses a **dual-channel lowering** strategy. Each Metal codegen handler produces:

1. **Sentinel AST expression**: A placeholder function call (e.g., `_metal_mm(load, load_1)`) inserted into the body AST. These sentinels are recognized later by the MslWalker during MSL generation.

2. **Structured IR dataclass**: A `MetalMatmulOp` or `MetalReductionOp` appended to `backend._metal_ops`. These carry metadata (operand names, dtypes, batched flag) that the MSL emitter and launcher use for dimension resolution and kernel classification.

The sentinel generation and IR recording happen in `aten_lowering.py`:

```python
# Simplified from _metal_dot() and _record_metal_matmul()
def codegen_mm_metal(ctx, node):
    lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    # Record structured IR
    backend._metal_ops.append(MetalMatmulOp(
        lhs_name=_ast_to_name(lhs), rhs_name=_ast_to_name(rhs),
        has_acc=False, acc_name=None, is_batched=False, dtype=lhs_fake.dtype,
    ))
    # Emit sentinel AST
    return expr_from_string("_metal_mm({lhs}, {rhs})", lhs=lhs, rhs=rhs)
```

For reductions, `MetalBackend.reduction_expr()` records `MetalReductionOp` and returns a sentinel string like `_metal_max(x, 1)`.

| PyTorch op | Sentinel AST node | Structured IR |
|---|---|---|
| `torch.mm(a, b)` | `_metal_mm(a, b)` | `MetalMatmulOp(lhs, rhs)` |
| `torch.addmm(acc, a, b)` | `_metal_addmm(acc, a, b)` | `MetalMatmulOp(lhs, rhs, acc)` |
| `torch.bmm(a, b)` | `_metal_bmm(a, b)` | `MetalMatmulOp(lhs, rhs, batched=True)` |
| `torch.baddbmm(acc, a, b)` | `_metal_baddbmm(acc, a, b)` | `MetalMatmulOp(lhs, rhs, acc, batched=True)` |
| `torch.amax(x, dim)` | `_metal_max(x, dim)` | `MetalReductionOp(input, "max", dim)` |
| `torch.sum(x, dim)` | `_metal_sum(x, dim)` | `MetalReductionOp(input, "sum", dim)` |

### Phase 3: Launcher Argument Resolution

`DeviceFunction.codegen_function_call()` calls `backend.build_launcher_args()` **before** MSL generation. This method:

1. Classifies the kernel from `_metal_ops` via `_classify_kernel()`.
2. Resolves physical dimensions (M, N, K, batch) from tensor `fake_value` shapes via `_resolve_matmul_dims()`.
3. Computes grid dimensions, thread counts, and scratch buffer sizes.
4. Appends dispatch parameters as keyword arguments to the launcher call.

| Kernel Kind | Dispatch Parameters |
|---|---|
| `FUSED_ATTENTION` | `_block_size`, `_composed_grid`, `_scratch_size`, `_num_simdgroups`, `_batch_size` |
| `MATMUL` | `_block_size`, `_matmul_grid=(grid_m, grid_n)`, `_num_simdgroups` |
| `SOFTMAX` | `_block_size`, `_nrows` |
| `ELEMENTWISE` | `_block_size` |

### Phase 4: MSL Generation

`DeviceFunction.codegen_function_def()` detects `backend.name == "metal"` and calls `backend.generate_msl_function(device_fn)`, which:

1. Consumes the recorded `_metal_ops`.
2. Classifies the kernel via `_classify_kernel()`.
3. Creates an `MslWalker` and calls `walker.generate()`.
4. Wraps the MSL string in a Python function: `def kernel_name(): return (msl_source, kernel_name)`.

### Phase 5: Runtime Dispatch

`default_metal_launcher()` in `helion/runtime/__init__.py`:

1. Calls `metal_kernel()` to get `(msl_source, kernel_name)`.
2. Compiles MSL via `torch.mps.compile_shader()` (cached on the function object).
3. Filters to tensor args only — scalar/constexpr values are baked into MSL constants.
4. Dispatches based on keyword arguments:
   - `_composed_grid`: Per-simdgroup 2D/3D dispatch with scratch allocation (attention).
   - `_matmul_grid`: 2D dispatch for matmul.
   - `_nrows`: One threadgroup per row (reduction).
   - Default: one thread per element (elementwise).

### MslWalker (Unified MSL Emitter)

The `MslWalker` class is the central MSL emitter. It walks the body AST statement-by-statement and dispatches to one of three code paths based on kernel classification:

- **`_generate_walked()`** — All matmul-containing kernels (MATMUL and FUSED_ATTENTION). Uses MPP `cooperative_tensor` + `matmul2d`.
- **`_generate_reduction()`** — SOFTMAX-classified kernels (softmax, LayerNorm, RMSNorm, CrossEntropy). Uses SIMD shuffle + threadgroup shared memory.
- **`_generate_elementwise()`** — Simple per-element kernels.

For attention kernels, `_generate_walked()` resolves matmul dimensions from the structured IR, determines RHS transposition, resolves the V tensor and output tensor for the second matmul, then calls `_emit_attention_body()` to emit the tiled softmax attention sequence.

For matmul-only kernels, it emits the MPP `matmul2d` call with optional epilogue fusion (relu, cast, binary ops detected by `_scan_for_epilogue()`).

### Kernel Classification

The backend classifies each kernel into one of four types using `MetalKernelKind`:

| `_metal_ops` contents | Classification |
|---|---|
| MatmulOp + ReductionOp (2+ matmuls, matmul→reduction→matmul ordering) | `FUSED_ATTENTION` |
| MatmulOp only | `MATMUL` |
| ReductionOp only | `SOFTMAX` |
| Neither | `ELEMENTWISE` |

Classification is centralized in `_classify_kernel()`. Both the MSL emitter (`generate_msl_function`) and the launcher argument builder (`build_launcher_args`) call it, ensuring they agree on the kernel type and dispatch strategy.

### AST-to-MSL Conversion

The `MetalBackend._ast_expr_to_msl()` method recursively converts Python AST nodes to MSL C++ strings. Key mappings:

| Python AST | MSL C++ |
|---|---|
| `libdevice.exp(x)` | `exp2(x * 1.4426950408889634f)` |
| `tl.cast(x, tl.float32)` | `static_cast<float>(x)` |
| `tl.reshape(x, shape)` | `x` (stripped) |
| `tl.full([], val, dtype)` | `((metal_type)(val))` |
| `_metal_max/sum/min(x, d)` | preserved as sentinel (consumed by classifier) |
| `x.to(dtype)` | `x` (stripped) |
| `x ** 0.5` | `sqrt(x)` |

### Constexpr Resolution

Scalar parameters (like `eps: float = 1e-5`) are embedded as MSL compile-time constants. The `_resolve_constexpr_define()` method handles:

1. **Bare identifier resolution**: `host_str = "eps"` is resolved to the actual value via `HostFunction.constexpr_args` or `SymbolArgument` var_hints in the shape env.
2. **Type-correct emission**: Float values emit `constant float eps = 1e-05f;` (not `constant int`).
3. **Float hint storage**: `CompileEnvironment.to_fake()` stores `sympy.Float(obj)` in the shape env var_hints for float parameters, paralleling the existing int hint storage.

## Kernel Implementations

### Elementwise Kernels

The simplest path. Each thread processes one element:

```msl
kernel void add(device float* x [[buffer(0)]],
                device float* y [[buffer(1)]],
                device float* out [[buffer(2)]],
                uint _gid [[thread_position_in_grid]]) {
    out[_gid] = x[_gid] + y[_gid];
}
```

Dispatch: `total_threads = grid[0] * block_size`, one thread per element.

### Reduction Kernels (Softmax, LayerNorm, RMSNorm, CrossEntropy)

One threadgroup per row. Threads within the group stride over columns and cooperate via SIMD shuffles + threadgroup shared memory for cross-simdgroup reduction.

#### Statement Classification

Before emitting MSL, `_classify_reduction_stmts()` walks the body AST and classifies each statement:

| Kind | Example | Meaning |
|---|---|---|
| `"col"` | `auto x = buf[_row * _RDIM + _j];` | Column-strided computation (inside `for _j` loop) |
| `"scalar"` | `float mean = sum / _RDIM;` | Scalar computation (outside column loop) |
| `"reduce"` | `("sum_var", "sum")` | SIMD reduction (shuffle + shared memory) |
| `"store"` | `out[_row * _RDIM + _j] = val;` | Output store (inside column loop) |

The classifier tracks "column-live" variables — those produced by row/column loads (`x[idx, :]`) or expressions referencing them. It also handles 1D broadcast patterns (`x[:, :]`) and `tl.full([], val)` scalar inits.

#### SIMD Reduction Architecture

Each reduction uses its own `threadgroup float _shared_N[32]` array:

```msl
// Per-thread partial sum
float sum = 0.0f;
for (uint _jv = _tid; _jv < _RDIM4 / 4; _jv += _tg_size) {
    float4 _ld = ((device float4*)(x + _row * _RDIM))[_jv];
    sum += _ld.x + _ld.y + _ld.z + _ld.w;
}

// Intra-simdgroup reduction via shuffle
for (uint _off = 16; _off > 0; _off >>= 1)
    sum += simd_shuffle_down(sum, _off);

// Cross-simdgroup reduction via shared memory
if (_lane == 0) _shared_0[_sg] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (_sg == 0) {
    float _v = (_lane < _num_sg) ? _shared_0[_lane] : 0.0f;
    for (uint _off = 16; _off > 0; _off >>= 1)
        _v += simd_shuffle_down(_v, _off);
    _shared_0[0] = _v;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
sum = _shared_0[0];  // broadcast to all threads
```

**Critical: per-reduction shared memory.** When a kernel has multiple reductions (e.g. LayerNorm: mean + variance), each reduction MUST use its own shared memory array (`_shared_0`, `_shared_1`, etc.). Reusing a single `_shared[32]` causes a race condition on Apple GPUs where threads can read stale first-reduction values when the second reduction writes to `_shared[_sg]`, despite threadgroup barriers. This manifests as non-deterministic accuracy failures at RDIM ≥ 4096 with 4+ simdgroups.

#### Vectorized Column Loops

Column loops use `float4` vectorization for the main body and a scalar tail for remainder:

```msl
// Vectorized main loop (4 elements per iteration)
for (uint _jv = _tid; _jv < _RDIM4 / 4; _jv += _tg_size) {
    float4 _ld = ((device float4*)(x + _row * _RDIM))[_jv];
    // ... per-lane computation on _ld.x, _ld.y, _ld.z, _ld.w
}
// Scalar tail for _RDIM % 4 remainder
for (uint _j = _RDIM4 + _tid; _j < _RDIM; _j += _tg_size) {
    // ... scalar computation
}
```

#### Threadgroup Row Cache (`use_tg_cache`)

The `use_tg_cache` autotuner knob caches the input row in threadgroup memory (32KB limit). When enabled, the input is loaded once from device memory into `threadgroup float _tg_row[_RDIM]`, and subsequent passes read from threadgroup memory instead of device memory:

```msl
threadgroup float _tg_row[_RDIM];
for (uint _ci = _tid; _ci < _RDIM; _ci += _tg_size)
    _tg_row[_ci] = x[_row * _RDIM + _ci];
threadgroup_barrier(mem_flags::mem_threadgroup);
```

Guard: `RDIM * 4 <= 32768` (RDIM ≤ 8192 for float32).

#### Reduction Optimization: `_optimize_reduction_entries`

Eliminates redundant recomputation across reduction passes:

**Single-reduction (softmax)**: Transforms 3-pass (max → exp+store → normalize) into 2-pass by writing `exp(x - max)` to the output buffer during the sum pass, then reading it back for normalization. Uses multiply-by-reciprocal (`1.0f / sum`) for the final pass.

**Two-reduction (LayerNorm) with `use_tg_cache`**: Stores centered values `(x - mean)` to the output buffer during the variance pass. The final normalization pass reads from the output buffer instead of recomputing centering from the threadgroup cache.

### Matrix Multiplication (MPP matmul2d)

Uses Apple's Metal Performance Primitives `matmul2d` API with `cooperative_tensor`:

```msl
constexpr auto desc = matmul2d_descriptor(
    TILE_M, TILE_N, dynamic_length_v<int>,  // K is dynamic
    false, false, false,
    matmul2d_descriptor::mode::multiply);
matmul2d<desc, execution_simdgroups<NUM_SG>> op;

auto As = A.slice(0, tgid.y * TILE_M);
auto Bs = B.slice(tgid.x * TILE_N, 0);
auto Cs = C.slice(tgid.x * TILE_N, tgid.y * TILE_M);
op.run(As, Bs, Cs);
```

Key details:
- `dynamic_length_v<int>` means MPP handles the K-reduction loop internally — the autotuner's K-tile parameter has no effect, so it is pinned.
- `execution_simdgroups<NUM_SG>` controls how many simdgroups cooperate on one output tile.
- Input tensors are wrapped as `tensor_inline` 2D tensors with `dextents<int32_t, 2>(cols, rows)` (MSL uses column-major extent ordering).
- **Epilogue fusion**: The `_scan_for_epilogue()` scanner detects relu, cast, and binary ops after matmul and fuses them into the `cooperative_tensor` `begin()/end()` iteration.
- **L2 cache swizzle**: Threadgroup grid indices are swizzled for better spatial locality.

#### Dimension Resolution Flow

Matmul dimensions are resolved through a multi-step process:

1. **Structured IR**: `MetalMatmulOp` stores operand names (`lhs_name`, `rhs_name`) — these are SSA variable names from the lowered AST (e.g., `"load"`, `"load_1"`).
2. **`_build_arg_maps()`**: Walks the full AST tree (including loop bodies) to build `load_to_arg` — a map from SSA load variables to their source buffer names (e.g., `"load" → "q"`, `"load_1" → "kt"`).
3. **`_resolve_name()`**: Resolves an SSA name to a `TensorArg` via `arg_map` and `load_to_arg` indirection.
4. **`_resolve_matmul_dims()`**: Given a `MetalMatmulOp`, resolves the full `TensorArg` objects, then reads M, N, K from `fake_value.size()` via the compile environment's shape hints.

```python
# Simplified from _resolve_matmul_dims:
lhs_arg = _resolve_name(mm_op.lhs_name, arg_map, load_to_arg)  # e.g. TensorArg for "q"
M = env.size_hint(lhs_arg.fake_value.size(-2))  # M from Q's second-to-last dim
K = env.size_hint(lhs_arg.fake_value.size(-1))  # K from Q's last dim
rhs_d0 = env.size_hint(rhs_arg.fake_value.size(-2))
rhs_d1 = env.size_hint(rhs_arg.fake_value.size(-1))
N = rhs_d1 if rhs_d0 == K else rhs_d0  # handles transpose
```

### Fused Attention (Tiled Softmax + Composable cooperative_tensor)

Computes `softmax(Q @ K^T / sqrt(d)) @ V` using composable `cooperative_tensor` operations with an inner loop over N-dimension tiles.

#### AST-Driven Scale Extraction

The `_extract_attention_scale()` method scans the inner for-loop body for `tl.full([], float_val, tl.float32)` assignments to extract the precomputed `qk_scale` constant (typically `(1/sqrt(d)) * log2(e) ≈ 0.255`). This avoids hardcoding the scale and adapts to any head dimension.

#### Generated MSL Structure

```msl
// Q slice for this tile row
auto _q_slice = _t_q.slice(0, _tile_row * _TILE_M);

// Scores matmul descriptor (TILE_M × TILE_N tiles, K dynamic)
constexpr auto _scoreDesc = matmul2d_descriptor(
    _TILE_M, _TILE_N, _K,
    false, transpose_rhs, false,
    matmul2d_descriptor::mode::multiply);
matmul2d<_scoreDesc, execution_simdgroup> _scoreOp;

// Output matmul descriptor (TILE_M × OUT_D, N dynamic)
constexpr auto _outDesc = matmul2d_descriptor(
    _TILE_M, _OUT_D, dynamic_length_v<int>,
    false, false, false,
    matmul2d_descriptor::mode::multiply);
matmul2d<_outDesc, execution_simdgroup> _outOp;

for (int _tn = 0; _tn < _N; _tn += _TILE_N) {
    // 1. Scores = Q_tile @ K_tile → cooperative_tensor
    auto _k_tile = _t_kt.slice(_tn, 0);
    auto _cScores = _scoreOp.get_destination_cooperative_tensor<...>();
    _scoreOp.run(_q_slice, _k_tile, _cScores);

    // 2. Scale scores by qk_scale
    for (auto _it = _cScores.begin(); _it != _cScores.end(); _it++)
        *_it *= 0.255f;

    // 3. Row max via reduce_rows
    auto _cMaxRow = _scoreOp.get_row_reduction_destination_cooperative_tensor<...>();
    reduce_rows(_cScores, _cMaxRow, reduction_operation::max, -FLT_MAX);

    // 4. Subtract max and exponentiate via map_iterator
    for (auto _it = _cScores.begin(); _it != _cScores.end(); _it++) {
        auto _max_it = _cMaxRow.map_iterator(_it);
        *_it = exp2(*_it - *_max_it);
    }

    // 5. Row sum and normalize
    auto _cSumRow = _scoreOp.get_row_reduction_destination_cooperative_tensor<...>();
    reduce_rows(_cScores, _cSumRow, reduction_operation::sum, 0.0f);
    for (auto _it = _cScores.begin(); _it != _cScores.end(); _it++) {
        auto _sum_it = _cSumRow.map_iterator(_it);
        *_it *= (1.0f / *_sum_it);
    }

    // 6. Store normalized P tile to scratch
    _cScores.store(_scratch_tile);
    simdgroup_barrier(mem_flags::mem_device);
}

// 7. Output matmul: P @ V → output
_outOp.run(_scratch_row, _v_slice, _o_slice);
```

Key design decisions:

- **`execution_simdgroup`** scope (1 SG per tile) enables `reduce_rows()` for row-wise reductions within the cooperative_tensor. Multiple tiles per threadgroup via `sg_idx` indexing.
- **Scratch buffer**: The full P matrix (`M × N` per head) is materialized in device memory between the scores and output matmuls. This is necessary because MPP's `cooperative_tensor` `map_iterator` only works between coops from the same matmul descriptor — cross-matmul-op iteration is not supported.
- **TILE_N = N**: Currently forced for correctness. Per-tile independent softmax produces incorrect results when TILE_N < N because each tile only sees a subset of keys. True online softmax (with running max/sum and accumulator rescaling) would require cross-descriptor `map_iterator` support or explicit threadgroup memory for the correction factors.
- **Batched dispatch**: For 3D inputs `[B*H, M, D]`, the grid uses `tgid.z` for head indexing. Per-head pointer offsets are computed in the kernel preamble.
- **GQA (Grouped-Query Attention)**: K/V expanded via `repeat_interleave` before the kernel call. The standard batched attention kernel handles GQA naturally.

#### MPP cooperative_tensor API Patterns Used

| Pattern | API | Where Used |
|---|---|---|
| Matmul into registers | `op.get_destination_cooperative_tensor<>()` + `op.run(A, B, coop)` | Scores, output |
| Row reduction | `op.get_row_reduction_destination_cooperative_tensor<>()` + `reduce_rows(coop, row_coop, op, identity)` | Max, sum |
| Element-wise in registers | `for (auto _it = coop.begin(); ...) *_it = expr;` | Scale, exp, normalize |
| Row broadcast | `coop_max.map_iterator(_it)` inside 2D iteration | Subtract max, normalize |
| Materialize to device | `coop.store(tensor_slice)` + barrier | P matrix |

## Autotuner Integration

The Metal backend integrates with Helion's autotuner to search for optimal kernel configurations. The autotuner explores tile sizes, simdgroup counts, and memory caching strategies, benchmarking each candidate on the actual hardware.

### How the Autotuner Works

Helion's autotuner uses **Likelihood-Free Bayesian Optimization (LFBO)** with a tree-based surrogate model. The search proceeds in phases:

1. **Initial population**: Random configs are generated and benchmarked (30 for `"quick"`, 100 for `"full"` effort).
2. **Surrogate fitting**: A Random Forest classifier is trained on (config → fast/slow) labels.
3. **Guided search**: Pattern search with neighborhood exploration proposes new configs based on the surrogate's predictions. Each generation explores neighbors of the best-so-far configs by mutating one knob at a time (e.g., doubling TILE_M, halving num_warps).
4. **Convergence**: Search terminates after a fixed number of generations (5 for `"quick"`, 20 for `"full"`) or when no improvement is found.

Effort levels control search intensity:

| Effort | Initial Pop | Generations | Random Budget | Typical Time |
|--------|------------|-------------|---------------|--------------|
| `"none"` | 0 | 0 | 0 | 0s (use default) |
| `"quick"` | 30 | 5 | 100 | 5-15s |
| `"full"` | 100 | 20 | 1000 | 20-60s |

Results are cached to disk (`torchinductor_<user>/helion/`). Subsequent runs with the same kernel and input shapes skip autotuning entirely.

### Metal Search Space

The Metal backend restricts the search space to three config keys via `supports_config_key()`. All other Triton/Pallas-specific keys (loop_orders, num_stages, pid_type, indexing, maxnreg, etc.) are excluded, preventing wasted search budget.

| Knob | Type | Range | Default | What It Controls |
|------|------|-------|---------|-----------------|
| `block_sizes` | list of powers-of-2 | [32, max] per dim | varies | Tile dimensions for threadgroup work partitioning |
| `num_warps` | power-of-2 | [2, 16] | 4 | Number of simdgroups per threadgroup (`threads = num_warps × 32`) |
| `use_tg_cache` | boolean | {False, True} | False | Cache input row in threadgroup memory for reduction kernels |

### How Config Maps to MSL

Each config knob directly affects the generated MSL constants and descriptors:

**`block_sizes`** maps to physical tile dimensions depending on kernel type:

| Kernel Type | block_sizes[0] | block_sizes[1] | block_sizes[2] |
|-------------|---------------|---------------|---------------|
| Elementwise | threads per threadgroup | — | — |
| Reduction (softmax, LayerNorm) | rows per threadgroup | — | — |
| Matmul | TILE_M (output rows) | TILE_N (output cols) | K-tile (reduction) |
| Attention (unbatched) | TILE_M | TILE_N (forced to N) | — |
| Attention (batched) | batch tile | TILE_M | TILE_N (forced to N) |

**`num_warps`** maps to `execution_simdgroups<NUM_SG>` for matmul/attention, and to the threadgroup size (`num_warps × 32` threads) for reduction/elementwise kernels. More simdgroups means more hardware parallelism within a tile but also higher register pressure.

**`use_tg_cache`** controls whether reduction kernels cache the input row in 32KB threadgroup memory. When enabled and the row fits (`RDIM × 4 ≤ 32768`), multi-pass reductions (e.g., LayerNorm: mean pass + variance pass) read from fast threadgroup memory instead of device memory on the second pass.

### Static vs Dynamic K for Matmul

The K-tile (`block_sizes[2]`) controls the MPP matmul descriptor's K-dimension strategy:

- **`TILE_K >= K`** (e.g., K=1024, TILE_K=1024): Static K — `matmul2d_descriptor(TILE_M, TILE_N, 1024, ...)`. MPP can unroll the K-reduction loop at compile time.
- **`TILE_K < K`** (e.g., K=1024, TILE_K=256): Dynamic K — `matmul2d_descriptor(TILE_M, TILE_N, dynamic_length_v<int>, ...)`. MPP pipelines the K-reduction loop internally.

The autotuner searches both strategies. For smaller K, static K with full unrolling often wins. For larger K, dynamic K with MPP's internal pipelining can be faster due to reduced register pressure.

### Block Size Constraints

`adjust_block_size_constraints()` enforces a minimum block size of 32 (Apple GPU SIMD width) for all dimensions. The maximum is `next_power_of_2(size_hint)` — for a 1024×1024 matmul, TILE_M can range from 32 to 1024.

### Metal-Specific Autotuner Behavior

- **No precompilation**: Metal does not support subprocess-based precompilation (`supports_precompile() → False`). All compilation happens in-process via `torch.mps.compile_shader()`.
- **Generic benchmarking**: Uses `do_bench_generic` (wall-clock with `torch.mps.synchronize()`) instead of Triton's GPU event timing.
- **Silent error handling**: `classify_autotune_exception()` classifies all exceptions as `"debug"`, so failed configs are silently skipped rather than aborting the search. This is important because some tile size combinations may produce invalid MSL (e.g., tiles larger than the matrix).
- **`num_stages` pinned to 1**: Metal has no software pipelining, so this dimension is eliminated from the search space.

### Example: Autotuner in Action

For a 1024×1024 matmul + ReLU kernel, the autotuner explores configs like:

```
block_sizes=[32, 32, 32],  num_warps=2   →  0.62ms  (small tiles, few SGs)
block_sizes=[64, 32, 64],  num_warps=4   →  0.55ms  (larger M-tile)
block_sizes=[64, 128, 256], num_warps=16 →  0.58ms  (too many SGs, register pressure)
block_sizes=[256, 32, 256], num_warps=8  →  0.54ms  ← best
```

The winning config uses 256-row tiles with 8 simdgroups — each threadgroup handles a large output chunk with high hardware parallelism. The generated MSL is structurally identical across all configs; only the constants (`_TILE_M`, `_TILE_N`, `_NUM_SG`) change. This is the key insight: the autotuner finds the optimal balance between tile size (data reuse), simdgroup count (parallelism), and register pressure for the specific problem size and hardware.

## Benchmarks

All benchmarks on Apple Silicon (MPS), float32. Per-call GPU timing following TritonBench methodology (one sync per call, adaptive warmup/repeat, IQR outlier removal). Compared against PyTorch eager, `torch.compile(inductor)`, and MLX.

### Vector Add

| Size | Eager | Inductor | Helion | MLX | vs Eager | vs Ind | vs MLX |
|------|-------|----------|--------|-----|----------|--------|--------|
| 1K | 0.092ms | 0.111ms | 0.102ms | 0.096ms | 0.90x | 1.08x | 0.94x |
| 16K | 0.106ms | 0.107ms | 0.105ms | 0.097ms | 1.01x | 1.02x | 0.93x |
| 262K | 0.130ms | 0.115ms | 0.112ms | 0.104ms | **1.16x** | 1.03x | 0.93x |
| 1.0M | 0.249ms | 0.256ms | 0.131ms | 0.124ms | **1.91x** | **1.96x** | 0.95x |
| 4.2M | 0.329ms | 0.341ms | 0.320ms | 0.315ms | 1.03x | 1.06x | 0.98x |
| 16.8M | 1.048ms | 1.092ms | 1.058ms | 1.053ms | 0.99x | 1.03x | 1.00x |

### Matrix Multiplication (MPP matmul2d)

| Size | Eager | Inductor | Helion | MLX | vs Eager | vs Ind | vs MLX |
|------|-------|----------|--------|-----|----------|--------|--------|
| 128×128 | 0.140ms | 0.129ms | 0.104ms | 0.097ms | **1.34x** | **1.24x** | 0.93x |
| 256×256 | 0.165ms | 0.140ms | 0.109ms | 0.108ms | **1.52x** | **1.29x** | 0.99x |
| 512×512 | 0.281ms | 0.214ms | 0.157ms | 0.165ms | **1.80x** | **1.36x** | **1.05x** |
| 1024² | 0.559ms | 0.651ms | 0.541ms | 0.548ms | 1.03x | **1.20x** | **1.01x** |
| 2048² | 3.282ms | 3.318ms | 3.300ms | 3.389ms | 0.99x | 1.01x | **1.03x** |

### Matmul + ReLU (fused epilogue)

| Size | Eager | Inductor | Helion | MLX | vs Eager | vs Ind | vs MLX |
|------|-------|----------|--------|-----|----------|--------|--------|
| 128×128 | 0.174ms | 0.138ms | 0.103ms | 0.099ms | **1.69x** | **1.34x** | 0.96x |
| 256×256 | 0.204ms | 0.149ms | 0.111ms | 0.116ms | **1.84x** | **1.34x** | **1.05x** |
| 512×512 | 0.309ms | 0.274ms | 0.159ms | 0.175ms | **1.94x** | **1.72x** | **1.10x** |
| 1024² | 1.006ms | 0.709ms | 0.541ms | 0.564ms | **1.86x** | **1.31x** | **1.04x** |
| 2048² | 3.531ms | 3.598ms | 3.313ms | 3.498ms | **1.07x** | **1.09x** | **1.06x** |

### RMSNorm

| (M, N) | Eager | Inductor | Helion | MLX | vs Eager | vs Ind | vs MLX |
|--------|-------|----------|--------|-----|----------|--------|--------|
| (256, 1024) | 0.300ms | 0.110ms | 0.103ms | 0.103ms | **2.92x** | **1.07x** | 1.00x |
| (1024, 1024) | 0.377ms | 0.136ms | 0.126ms | 0.134ms | **2.99x** | **1.08x** | **1.06x** |
| (1024, 4096) | 1.197ms | 0.291ms | 0.251ms | 0.244ms | **4.76x** | **1.16x** | 0.97x |
| (4096, 2560) | 3.022ms | 0.546ms | 0.520ms | 0.487ms | **5.81x** | **1.05x** | 0.94x |
| (4096, 4096) | 4.636ms | 0.827ms | 0.784ms | 0.754ms | **5.91x** | **1.05x** | 0.96x |

### Softmax

| (M, N) | Eager | Inductor | Helion | MLX | vs Eager | vs Ind | vs MLX |
|--------|-------|----------|--------|-----|----------|--------|--------|
| (256, 1024) | 0.186ms | 0.117ms | 0.104ms | 0.104ms | **1.79x** | **1.13x** | 1.00x |
| (1024, 1024) | 0.151ms | 0.151ms | 0.128ms | 0.130ms | **1.18x** | **1.18x** | **1.02x** |
| (1024, 4096) | 0.282ms | 0.245ms | 0.241ms | 0.241ms | **1.17x** | 1.02x | 1.00x |
| (4096, 2560) | 0.540ms | 0.533ms | 0.523ms | 0.483ms | 1.03x | 1.02x | 0.92x |

### LayerNorm

| (M, N) | Eager | Inductor | Helion | MLX | vs Eager | vs Ind | vs MLX |
|--------|-------|----------|--------|-----|----------|--------|--------|
| (256, 1024) | 0.130ms | 0.119ms | 0.106ms | 0.097ms | **1.23x** | **1.13x** | 0.92x |
| (1024, 1024) | 0.129ms | 0.143ms | 0.125ms | 0.119ms | 1.03x | **1.14x** | 0.95x |
| (1024, 4096) | 0.276ms | 0.263ms | 0.274ms | 0.273ms | 1.01x | 0.96x | 1.00x |
| (4096, 2560) | 0.547ms | 0.574ms | 0.543ms | 0.503ms | 1.01x | **1.06x** | 0.93x |

### Cross Entropy

| (N, V) | Eager | Inductor | Helion | MLX | vs Eager | vs Ind | vs MLX |
|--------|-------|----------|--------|-----|----------|--------|--------|
| (256, 1024) | 0.183ms | 0.122ms | 0.135ms | 0.115ms | **1.36x** | 0.91x | 0.85x |
| (1024, 1024) | 0.187ms | 0.141ms | 0.145ms | 0.134ms | **1.29x** | 0.97x | 0.93x |
| (1024, 4096) | 0.392ms | 0.163ms | 0.185ms | 0.155ms | **2.12x** | 0.88x | 0.84x |
| (4096, 2560) | 0.863ms | 0.343ms | 0.327ms | 0.300ms | **2.64x** | **1.05x** | 0.92x |

### Two-Layer MLP

| Size | Eager | Inductor | Helion | MLX | vs Eager | vs Ind | vs MLX |
|------|-------|----------|--------|-----|----------|--------|--------|
| 128×512→1024→512 | 0.257ms | 0.238ms | 0.200ms | 0.200ms | **1.29x** | **1.19x** | 1.00x |
| 256×1024→2048→1024 | 0.616ms | 0.634ms | 0.558ms | 0.592ms | **1.10x** | **1.14x** | **1.06x** |
| 512×1024→2048→1024 | 1.032ms | 1.070ms | 0.996ms | 1.048ms | 1.04x | **1.07x** | **1.05x** |

### GQA (Grouped-Query Attention, B×H8×K2)

| Config | Eager | Inductor | Helion | MLX | vs Eager | vs Ind | vs MLX |
|--------|-------|----------|--------|-----|----------|--------|--------|
| B1 S=64 | 0.171ms | 0.165ms | 0.122ms | 0.118ms | **1.40x** | **1.35x** | 0.97x |
| B1 S=128 | 0.192ms | 0.171ms | 0.173ms | 0.127ms | **1.11x** | 0.99x | 0.73x |
| B1 S=256 | 0.292ms | 0.269ms | 0.213ms | 0.163ms | **1.37x** | **1.26x** | 0.76x |
| B2 S=64 | 0.187ms | 0.159ms | 0.134ms | 0.123ms | **1.39x** | **1.19x** | 0.91x |
| B2 S=128 | 0.240ms | 0.199ms | 0.188ms | 0.141ms | **1.27x** | **1.05x** | 0.75x |
| B2 S=256 | 0.525ms | 0.354ms | 0.306ms | 0.228ms | **1.72x** | **1.16x** | 0.75x |

## Known Limitations

1. **Fused attention uses scratch buffer**: The composable `cooperative_tensor` path materializes P to device memory between the two matmuls. A register-based flash attention path would be faster but requires cross-matmul-op `map_iterator` support or explicit threadgroup memory for running state, which MPP does not currently provide.

2. **TILE_N forced to N for attention**: The inner N-loop currently forces TILE_N = N for correctness. Per-tile independent softmax produces incorrect results when TILE_N < N. True online softmax with running max/sum tracking would need a way to rescale the output accumulator (which is a cooperative_tensor from a different matmul descriptor) by the correction factor alpha (which is a row-reduction coop from the scores descriptor). MPP's `map_iterator` does not support cross-descriptor iteration.

3. **Float32 only**: All kernel paths use `float` throughout. Half-precision would need `simdgroup_matrix<half, 8, 8>` and half-precision threadgroup memory.

4. **Non-contiguous input overhead**: The launcher calls `.contiguous()` on non-contiguous tensors, which allocates a copy.

5. **Elementwise path still uses regex**: The `_py_to_msl()` method is retained for the simple elementwise (1D per-thread) path. Other paths use the AST-walking pipeline.

## File Index

| File | Description |
|------|-------------|
| `helion/_compiler/backend.py` | `MetalBackend` + `MslWalker`: classification, unified walker, reduction emitter, matmul/attention emitter, epilogue fusion, autotuner constraints |
| `helion/_compiler/aten_lowering.py` | Metal codegen for mm/addmm/bmm/baddbmm sentinel + structured IR emission |
| `helion/_compiler/inductor_lowering.py` | `GraphInterpreter.run_node()`: per-node lowering dispatch |
| `helion/_compiler/device_function.py` | `codegen_function_def` (Metal dispatch) + `codegen_function_call` (launcher args) |
| `helion/_compiler/compile_environment.py` | Float hint storage for Metal constexpr resolution |
| `helion/runtime/__init__.py` | `default_metal_launcher`: compile, cache, dispatch |
| `test/test_metal.py` | 29 tests: vec-add, softmax, matmul (7 shapes), rmsnorm (3), layernorm (2), cross-entropy (2), naive/fused/batched/GQA attention |
| `benchmarks/bench_all_metal.py` | Full benchmark: 9 sections, Helion vs eager vs inductor vs MLX |
