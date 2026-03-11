# Helion Metal Backend: Apple GPU Code Generation via MPP

## Overview

The Metal backend for Helion generates Metal Shading Language (MSL) kernels that run on Apple Silicon GPUs via the MPS (Metal Performance Shaders) device. It uses Apple's Metal Performance Primitives (MPP) TensorOps API (Metal 4+, macOS 26) for hardware-accelerated matrix multiplication and leverages SIMD shuffle instructions for efficient reductions.

The backend supports four kernel types — elementwise, softmax, matrix multiplication, and fused attention — and includes an autotuner that searches over tile sizes and simdgroup counts to find optimal configurations per problem size.

**Key result**: The Helion Metal backend beats both PyTorch eager and `torch.compile(inductor)` on Apple Silicon for matmul, softmax, and fused attention across all tested sizes.

## Architecture

### Compilation Pipeline

```
Helion DSL kernel (Python)
    ↓  [Helion tracing + FX graph]
Body AST with sentinels (_metal_mm, _reduce_max_val, etc.)
    ↓  [MetalBackend.generate_msl_function()]
MSL source string
    ↓  [torch.mps.compile_shader()]
Compiled Metal shader library
    ↓  [default_metal_launcher()]
GPU dispatch via MPS
```

The compilation has three phases:

1. **Tracing**: Helion traces the Python kernel through `torch.fx`, producing an FX graph. Each operation is lowered via backend-specific codegen handlers in `aten_lowering.py`, which emit **sentinel expressions** — placeholder function calls like `_metal_mm(lhs, rhs)` that mark where Metal-specific operations should be emitted.

2. **MSL Generation**: `MetalBackend.generate_msl_function()` unparses the body AST to text, classifies the kernel from the sentinels present, and dispatches to the appropriate emitter method which generates the complete MSL kernel string.

3. **Dispatch**: The launcher compiles the MSL via `torch.mps.compile_shader()`, caches the result, and dispatches to the GPU with the appropriate thread/threadgroup configuration.

### Kernel Classification

The backend classifies each kernel into one of four types using `MetalKernelKind`:

```python
class MetalKernelKind:
    ELEMENTWISE = "elementwise"      # 1D flat dispatch
    SOFTMAX = "softmax"              # 2D row-parallel SIMD reduction
    MATMUL = "matmul"                # MPP matmul2d
    FUSED_ATTENTION = "fused_attention"  # matmul + softmax + matmul
```

Classification is based on which sentinels appear in the body text:

| Sentinels in body | Classification |
|---|---|
| `_metal_mm` + `_RDIM` | `FUSED_ATTENTION` |
| `_metal_mm` only | `MATMUL` |
| `_RDIM` only | `SOFTMAX` |
| Neither | `ELEMENTWISE` |

Both the MSL emitter and the launcher argument builder use this classification, ensuring they agree on the kernel type and dispatch strategy.

### Sentinel System

Operations in the Helion kernel body are lowered to sentinel expressions by `aten_lowering.py`:

| PyTorch op | Sentinel | Args |
|---|---|---|
| `torch.mm(a, b)` | `_metal_mm(a, b)` | 2-arg (no accumulator) |
| `torch.addmm(acc, a, b)` | `_metal_addmm(acc, a, b)` | 3-arg (with accumulator) |
| `torch.bmm(a, b)` | `_metal_bmm(a, b)` | 2-arg batched |
| `torch.baddbmm(acc, a, b)` | `_metal_baddbmm(acc, a, b)` | 3-arg batched |
| `torch.amax(x, dim)` | `_metal_max(x, dim)` → `_reduce_max_val` | Row max |
| `torch.sum(x, dim)` | `_metal_sum(x, dim)` → `_reduce_sum_val` | Row sum |

The `_py_to_msl()` method transforms remaining Python AST text to MSL C++ via regex substitutions (e.g., `libdevice.exp` → `exp`, `x[row, :]` → `x[_gid * _RDIM + _j]`).

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

### Softmax Kernels (2-Pass Optimized)

One threadgroup per row. Threads within the group stride over columns and cooperate via SIMD shuffles + threadgroup shared memory for cross-simdgroup reduction.

**Pass 1** — Row max: Each thread computes a local max over its strided columns. SIMD shuffle reduces within each simdgroup (32 threads), then threadgroup shared memory collects per-simdgroup results for a final reduction.

**Pass 2** — Exp + sum: Each thread computes `exp(x - row_max)`, writes to the output buffer, and accumulates a local sum. The sum is reduced the same way as the max. Writing exp values to output avoids re-reading input in pass 3.

**Pass 3** — Normalize: Each thread multiplies its output elements by `1/sum` (multiplication is faster than per-element division).

```msl
// Pass 2: exp + sum (write exp to output for reuse)
float sum_exp = 0.0f;
for (uint j = tid; j < RDIM; j += tg_size) {
    float e = exp((float)x[row * RDIM + j] - amax);
    out[row * RDIM + j] = e;
    sum_exp += e;
}
// [SIMD + threadgroup reduction for sum_exp]

// Pass 3: normalize (reuse from output, multiply not divide)
float inv_sum = 1.0f / sum_exp;
for (uint j = tid; j < RDIM; j += tg_size) {
    out[row * RDIM + j] *= inv_sum;
}
```

### Matrix Multiplication (MPP matmul2d)

Uses Apple's Metal Performance Primitives `matmul2d` API, which provides hardware-accelerated tiled matrix multiplication with configurable tile sizes and simdgroup counts.

```msl
constexpr auto desc = matmul2d_descriptor(
    TILE_M, TILE_N, dynamic_length_v<int>,  // K is dynamic
    false, false, false,                     // no transpose
    matmul2d_descriptor::mode::multiply);
matmul2d<desc, execution_simdgroups<NUM_SG>> op;

auto As = A.slice(0, tgid.y * TILE_M);
auto Bs = B.slice(tgid.x * TILE_N, 0);
auto Cs = C.slice(tgid.x * TILE_N, tgid.y * TILE_M);
op.run(As, Bs, Cs);
```

Key details:
- `dynamic_length_v<int>` means MPP handles the K-reduction loop internally — the autotuner's K-tile parameter has no effect, so it is pinned to avoid wasting search budget.
- `execution_simdgroups<NUM_SG>` controls how many simdgroups (each 32 threads) cooperate on one output tile.
- Input tensors are wrapped as `tensor_inline` 2D tensors with `dextents<int32_t, 2>(cols, rows)` (note: MSL uses column-major extent ordering).
- The autotuner searches over `TILE_M`, `TILE_N`, and `NUM_SG`.

### Fused Attention (Batched Single-Dispatch)

The most complex kernel type. Computes `softmax(Q @ K^T / sqrt(d)) @ V` in a single GPU dispatch for all B×H attention heads, avoiding the overhead of multiple kernel launches.

**Architecture**: Three stages in one kernel, all using `execution_simdgroups<NUM_SG>`:

```
Stage 1: Scores = Q_tile @ K^T  →  scratch buffer (multi-SG matmul2d)
Stage 2: Softmax(scores)        →  scratch buffer (threadgroup SIMD shuffles)
Stage 3: Output = Weights @ V   →  output buffer  (multi-SG matmul2d)
```

**Why device-memory softmax instead of cooperative_tensor?**

MPP provides `reduce_rows()` which can operate directly on `cooperative_tensor` (the register-file abstraction that chains matmul2d outputs). However, `reduce_rows` has a hard constraint: `static_assert("reduce_rows requires a single SIMD group")`. Using 1 simdgroup makes the matmuls ~4x slower. The workaround stores the matmul output to a scratch buffer in device memory, performs softmax there with all threads cooperating, then feeds the result to the second matmul.

**Batched dispatch**: For 3D tensor inputs `[B*H, M, D]`, the emitter generates a 3D grid:
- `tgid.x`: thread within threadgroup (used for SIMD shuffles)
- `tgid.y`: row tile index (M dimension)
- `tgid.z`: head index (batch dimension)

Per-head pointer offsets are computed from `tgid.z * head_stride`, and each threadgroup works on one `(head, tile_m)` combination independently.

```msl
// Per-head pointer offsets
uint head = tgid.z;
device float* hq = q + head * HEAD_STRIDE;
device float* hkt = kt + head * KT_STRIDE;
// ... wrap as tensor_inline, run matmul, softmax, matmul
```

**Scale factor**: The emitter uses `rsqrt((float)K)` for the standard `1/sqrt(d_k)` attention scaling, applied during the softmax phase.

## Autotuner Integration

The Metal backend integrates with Helion's autotuner to search for optimal configurations:

**Supported config keys**: Only `block_sizes` and `num_warps`. The backend explicitly excludes `loop_orders`, `flatten_loops`, and `reduction_loops` because the Metal emitter ignores them — including them would waste autotuner search budget on no-op dimensions.

**K-tile pinning**: For matmul kernels with 3+ block size dimensions, the inner reduction (K) dimension is pinned to `next_power_of_2(size_hint)` since MPP's `dynamic_length_v` handles K-reduction internally. This prevents the autotuner from exploring different K-tile values that have zero effect, effectively doubling the search budget for the parameters that matter (TILE_M, TILE_N, NUM_SG).

**Block size constraints**: Minimum block size is set to 32 (the Apple GPU SIMD width).

**Impact**: Pruning unused config dimensions + pinning K-tile turned matmul 512×512 from 0.66x to 1.06x vs eager — the autotuner was previously wasting ~75% of its budget exploring no-op dimensions.

## Launcher

`default_metal_launcher()` in `helion/runtime/__init__.py` handles MSL compilation, caching, and GPU dispatch:

1. **Compilation**: `torch.mps.compile_shader(msl_source)` compiles MSL to a Metal shader library. Results are cached on the kernel function object.

2. **Contiguity**: Non-contiguous tensor inputs are made contiguous before dispatch — `tensor_inline` wraps raw pointers assuming contiguous memory layout.

3. **Device validation**: All tensor args are validated to be on the MPS device.

4. **Dispatch modes**:
   - `ELEMENTWISE`: `threads = grid[0] * block_size`, `group_size = block_size`
   - `SOFTMAX`: `threads = nrows * block_size`, `group_size = block_size` (one threadgroup per row)
   - `MATMUL`: 2D dispatch `threads = [grid_n * tpg, grid_m]`, `group_size = [tpg, 1]`
   - `FUSED_ATTENTION`: 2D or 3D dispatch `threads = [tpg, grid_m, batch]`, `group_size = [tpg, 1, 1]`, with scratch buffer allocation

## Benchmarks

All benchmarks on Apple Silicon (MPS), float32. Compared against PyTorch eager and `torch.compile` with the inductor backend.

### Matrix Multiplication (MPP matmul2d)

| Size | Eager | Inductor | Helion | vs Eager | vs Inductor |
|------|-------|----------|--------|----------|-------------|
| 128×128 | 0.131ms | 0.193ms | 0.144ms | 0.91x | **1.34x** |
| 256×256 | 0.181ms | 0.198ms | 0.148ms | **1.23x** | **1.34x** |
| 512×512 | 0.169ms | 0.179ms | 0.159ms | **1.06x** | **1.12x** |
| 1024×1024 | 0.544ms | 0.570ms | 0.528ms | **1.03x** | **1.08x** |

### Softmax (2-Pass SIMD Reduction)

| (M, N) | Eager | Inductor | Helion | vs Eager | vs Inductor |
|--------|-------|----------|--------|----------|-------------|
| (128, 256) | 0.135ms | 0.163ms | 0.125ms | **1.08x** | **1.30x** |
| (256, 1024) | 0.160ms | 0.203ms | 0.111ms | **1.44x** | **1.84x** |
| (1024, 1024) | 0.267ms | 0.229ms | 0.148ms | **1.81x** | **1.55x** |
| (4096, 2560) | 0.572ms | 0.676ms | 0.536ms | **1.07x** | **1.26x** |

### Fused Attention (B=2, H=8, D=64, Single Dispatch)

| Seq Len | SDPA Eager | SDPA Inductor | Helion | vs Eager | vs Inductor |
|---------|------------|---------------|--------|----------|-------------|
| 64 | 0.293ms | 0.302ms | 0.276ms | **1.06x** | **1.09x** |
| 128 | 0.221ms | 0.245ms | 0.201ms | **1.10x** | **1.22x** |
| 256 | 0.363ms | 0.380ms | 0.328ms | **1.11x** | **1.16x** |
| 512 | 0.967ms | 0.976ms | 0.516ms | **1.87x** | **1.89x** |

## Known Limitations

1. **MPP `reduce_rows` single-simdgroup constraint**: `reduce_rows` requires `execution_simdgroups<1>`, and produces incorrect results for N > 128 columns. This forces the fused attention kernel to use device-memory softmax instead of cooperative_tensor chaining. See `benchmarks/mpp_reduce_rows_repro.metal` for a minimal reproduction.

2. **Softmax emitter is pattern-specific**: Only handles the exact softmax pattern (max → exp → sum → normalize). Other reduction patterns (standalone sum, argmax, layer norm) raise `BackendUnsupported`.

3. **Fused attention emitter is pattern-specific**: Only handles the matmul → softmax → matmul pattern. Other matmul + reduction combinations raise `BackendUnsupported`.

4. **Regex-based AST parsing**: The `_py_to_msl()` method and sentinel extraction use regex on unparsed AST text, which is fragile. A production-quality implementation would walk the FX graph directly.

5. **float32 only tested**: Half-precision and bfloat16 paths exist in the type mappings but are untested.

6. **Non-contiguous input overhead**: The launcher calls `.contiguous()` on non-contiguous tensors, which allocates a copy. Ideally the emitter would handle strides natively.

## File Index

| File | Lines | Description |
|------|-------|-------------|
| `helion/_compiler/backend.py` | +1229 | `MetalBackend` class: classification, 4 emitters, `_py_to_msl`, autotuner constraints |
| `helion/_compiler/aten_lowering.py` | +51 | Metal codegen for mm/addmm/bmm/baddbmm sentinel emission |
| `helion/runtime/__init__.py` | +115 | `default_metal_launcher`: compile, cache, dispatch |
| `test/test_metal.py` | +422 | 20 tests: vec-add, softmax, matmul, naive/fused/batched attention |
| `benchmarks/metal_attention_bench.py` | +125 | Fused attention benchmark vs SDPA |
| `benchmarks/metal_matmul_bench.py` | +97 | Matmul benchmark vs eager + inductor |
| `benchmarks/metal_softmax_bench.py` | +99 | Softmax benchmark vs eager + inductor |
| `benchmarks/mpp_reduce_rows_repro.metal` | +82 | MPP `reduce_rows` bug reproduction |
