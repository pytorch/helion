# v0i0 (Markus Hoehnerbach) — TPU/Pallas Work in Helion: Comprehensive Analysis

## Overview

Markus (GitHub: v0i0, `mhoehnerbach@meta.com`) contributed **295 commits** to this repo. His TPU/Pallas work can be categorized into **7 major architectural areas**, each reflecting fundamental differences between TPU and GPU hardware.

---

## 1. The Three Launcher Types (Memory Model & Data Movement)

**This is the single most important architectural difference from GPU.**

On GPU (Triton/CUDA), data movement is implicit — you load from global memory through caches, and the hardware manages L1/L2. On TPU, data movement is **explicit** — you must DMA data from HBM to VMEM before computing.

Markus built three distinct launcher/codegen paths reflecting three TPU programming patterns:

### a) Default Launcher (`default_pallas_launcher`)
- Uses `pl.pallas_call` with `BlockSpec`s (tiled access patterns)
- Pallas runtime automatically tiles data from HBM -> VMEM based on BlockSpecs
- Analogous to Triton's `tl.load` with block pointers, but the tiling is declared upfront rather than inside the kernel
- **Key commits**: `c4aa089` (remove shape-based block specs), `95c9b6d` (fix BlockSpec regressions)

### b) Pipeline Launcher (`default_pallas_pipeline_launcher`)
- Uses `pltpu.emit_pipeline` for **software-pipelined DMA**
- All tensors passed as `memory_space=pl.ANY` (HBM refs)
- Scratch buffers allocated as `pltpu.VMEM` shapes
- The kernel internally pipelines: while computing on tile N in VMEM, it prefetches tile N+1 from HBM
- **Key commits**: `bd38873` (emit_pipeline with run_scoped for rolled reductions)

### c) Fori Loop Launcher (`default_pallas_fori_launcher`)
- Uses `jax.lax.fori_loop` + `pltpu.make_async_copy` for **manual DMA control**
- Adds `pltpu.SemaphoreType.DMA` for async copy synchronization
- Gives finest-grained control over data movement
- **Key commits**: `6e0aad0` (fori_loop carry state), `ee60794` (carry-as-vmem scratch refs)

**GPU equivalent understanding**: On GPU, you don't think about DMA at all — the cache hierarchy handles it. On TPU, you must explicitly:
1. Allocate scratch in VMEM (like allocating shared memory, but for ALL data)
2. DMA from HBM -> VMEM (like `cp.async` in CUDA, but mandatory)
3. Synchronize with semaphores (like CUDA's `__syncthreads` but for DMA completion)
4. Pipeline these operations for throughput

---

## 2. Block Size Constraints (Lane/Sublane Architecture)

**This directly reflects TPU's 2D register tile architecture.**

From `PallasBackend.adjust_block_size_constraints()` (backend.py:1108-1183):

```
TPU Pallas requires:
- 1D last dim: multiple of 128 * (32 // dtype_bits)
  -> 128 for f32, 256 for bf16, 512 for int8
- 2D+ last dim: multiple of 128 (lane dimension)
- 2D+ second-to-last dim: multiple of 8 (sublane dimension)
```

**Why these numbers?** TPU's vector unit processes data in **2D tiles**:
- **128 lanes** (columns) — the SIMD width of the VPU. Think of it as 128 parallel ALUs
- **8 sublanes** (rows) — allows processing 8 rows simultaneously

This is fundamentally different from GPU where:
- GPU has **32 threads per warp** (1D SIMD), each processing one element
- TPU processes a **128x8 = 1024 element 2D tile** in one instruction

**Key commits**:
- `58c1148` — treat sublane constraint like lane constraint (most recent, fixing that second-to-last dim also needs power-of-2 clamping when tensor dim is small)
- `f4d4dfc` — Fix TPU block size alignment for 1D tensors in reductions

---

## 3. Looped Reductions (Carry State & Functional Semantics)

**TPU requires functional-style programming — no mutable state.**

On GPU/Triton:
```python
acc = tl.zeros(...)
for k in range(0, K, BLOCK_K):
    acc += tl.dot(a, b)  # mutate acc in-place
```

On TPU/Pallas, this doesn't work because JAX uses **functional transformations**. Markus implemented two solutions:

### a) fori_loop carry state (`6e0aad0`)
Thread the accumulator through `jax.lax.fori_loop` as carry state:
```python
def body(i, carry):
    acc = carry
    acc = acc + compute(...)
    return acc
result = lax.fori_loop(0, N, body, init_acc)
```

### b) VMEM scratch refs (`ee60794`, `bd38873`)
Use `pl.run_scoped` to allocate VMEM scratch for accumulators:
```python
def _reduction_body(acc_scratch, result_ref, sem):
    acc_scratch[...] = jnp.full(...)
    def _pipeline_body(x_vmem):
        acc = acc_scratch[...]
        acc = acc + compute(x_vmem)
        acc_scratch[...] = acc
    pltpu.emit_pipeline(_pipeline_body, ...)(...)
    result = reduce(acc_scratch[...])
    # DMA result back to HBM
    copy = pltpu.make_async_copy(result_ref, out, sem)
    copy.start(); copy.wait()
pl.run_scoped(_reduction_body, pltpu.VMEM(...), ...)
```

The key insight: Markus implemented **automatic phi-node detection** — analyzing the FX graph to find which variables are modified across loop iterations (carry variables) and promoting them to scratch refs automatically.

---

## 4. Conditional Execution (lax.cond vs. Python if)

**TPU cannot do Python `if` on traced values.**

On GPU:
```python
if mask:  # fine, each thread evaluates independently
    do_thing()
```

On TPU, Markus implemented `lax.cond`-based conditionals:
- **Key commits**: `4612b33` (fix phi-merged values), `baffff1` (refactor), `4b9e05d` (0-d tensor predicates)

Critical constraint: `lax.cond` requires a **scalar predicate**. Tensor-derived predicates from loads produce vectors on TPU due to hardware tiling -> unsupported. This is because every "element" access on TPU actually loads a 128x8 tile.

The phi-merge problem was particularly tricky: `lax.cond(pred, true_fn, false_fn)` must return the same structure from both branches. Markus had to:
1. Analyze which variables are modified in the true branch
2. Create matching identity functions for the false branch
3. Unpack results and merge them back (phi nodes)

---

## 5. Matmul / Dot Product (MXU Constraints)

On GPU, matmul maps to Tensor Cores with flexible tile sizes. On TPU, matmul goes through the **MXU (Matrix Multiply Unit)**, a 128x128 systolic array.

**Key commits**: `edc5385` (make dot work with jnp.matmul + f32 accumulator), `d8638ce` (f32 accumulator for sub-32-bit matmul)

Key differences Markus had to handle:
- Use `jnp.matmul` instead of `jnp.dot` for correct BMM support
- Always promote to f32 accumulator for sub-32-bit types (bf16/f16) — the MXU computes in f32 internally
- No warp-level matmul (Tensor Core `wmma`) — the MXU is a chip-level resource, not per-thread
- Block sizes for matmul must align to MXU dimensions

---

## 6. Scalar Arguments & Type System

**TPU Mosaic lowering requires rank >= 1 for all BlockSpecs.**

From `PallasBackend.transform_host_arg()` (backend.py:966-993):
```python
# Scalars are passed as 1-dim tensors (shape [1]) rather than
# 0-dim tensors (shape []) because TPU Pallas Mosaic lowering
# requires rank >= 1 for all block specs.
return f"torch.tensor([{host_str}], ...)"
```

On GPU, scalars are just kernel parameters (constexpr or passed by value). On TPU, they must be wrapped as 1-element tensors and dereferenced with `[0]` inside the kernel.

Other type system differences:
- No int64 on TPU — RNG seeds must be `torch.int32` (backend.py:1212-1216)
- `range()` bounds must be plain Python ints, not traced values (`range_requires_python_int = True`)
- `lax.convert_element_type()` instead of Triton's `.to()` for casts
- float16 is poorly supported — Markus added `HALF_DTYPE` constant for portable tests (`4c4b33b`)

---

## 7. Buffer Donation & Output Tracking

On GPU, you write outputs with `tl.store()` — any address, any time. On TPU with `pallas_call`, you must declare upfront which tensors are outputs via `input_output_aliases` and `out_shape`.

From `PallasBackend.build_launcher_args()` (backend.py:1341-1404):
- Identifies output tensors: created inside function body OR mutated in-place
- Builds `_output_indices` for the launcher
- Only output tensors get their buffers donated (avoiding "Buffer has been deleted" errors)

**Key fix by Theotime** but directly in Markus's domain: `5dc8b65` (fix Pallas buffer donation: only donate in-place mutated tensors)

---

## Config & Autotuning Differences

The `PallasBackend` only supports these config keys:
```python
_PALLAS_SUPPORTED_KEYS = {
    "block_sizes",      # tile sizes (constrained to lane/sublane multiples)
    "loop_orders",      # iteration order
    "flatten_loops",    # whether to flatten multi-dim loops
    "reduction_loops",  # rolled vs unrolled reductions
    "pallas_loop_type", # "default" | "emit_pipeline" | "fori_loop"
}
```

Everything GPU-specific is excluded: `num_warps`, `num_stages`, `maxnreg`, `indexing`, `pid_type`.

Autotuning uses `do_bench_generic` (not Triton's event-based timing), and precompilation is disabled (`supports_precompile = False`) since Pallas doesn't support subprocess precompilation.

---

## Commit Distribution Summary

| Category | Count | Description |
|----------|-------|-------------|
| Pallas/TPU codegen | ~40 | Core backend, launchers, emit_pipeline, fori_loop |
| Test management | ~44 | xfailIfPallas, skipIfPallas, test enablement |
| CI/Build/torch_tpu | ~53 | Bazel builds, torch_tpu pins, TPU wheels |
| Bug fixes | ~72 | Crashes, segfaults, xfail reclassification |
| Block specs | ~15 | Lane/sublane constraints, BlockSpec computation |
| Conditionals | ~8 | lax.cond, phi nodes, 0-d predicates |
| Reductions | ~10 | carry state, VMEM scratch, run_scoped |
| Non-TPU (Blackwell, autotuner, examples) | ~53 | Also worked on GPU-side features |

---

## What You Need to Learn: TPU Architecture & Pallas

### TPU Hardware Architecture

#### 1. Memory Hierarchy (vs GPU)

| Level | TPU | GPU Equivalent | Notes |
|-------|-----|---------------|-------|
| HBM | 16-96 GB, ~1-3 TB/s | Global Memory | Same concept, different bandwidth |
| VMEM | ~16-32 MB per core | Shared Memory + L1 | But MUCH larger than GPU SMEM (48-228 KB) |
| SMEM | Small scalar memory | Registers | For scalar values only |
| CMEM | Constant memory | Constant cache | Small, fast |
| Registers | 2D tile registers (128x8) | 32-bit registers per thread | Fundamentally different shape |

#### 2. Compute Units

| Unit | TPU | GPU Equivalent |
|------|-----|---------------|
| MXU | 128x128 systolic array, bfloat16 | Tensor Cores (per SM) |
| VPU | 128 lanes x 8 sublanes vector unit | CUDA cores (per SM) |
| SFU | Scalar functional unit | Special function units |

Key conceptual shift: **TPU is SIMD, not SIMT.** There are no threads, no warps, no thread blocks. Instead, you write code that operates on 2D tiles. The 128 lanes and 8 sublanes execute the SAME instruction on different data positions simultaneously.

#### 3. Data Movement

TPU uses **explicit DMA** rather than implicit caching:
- You issue `pltpu.make_async_copy(src, dst, semaphore)` to start a DMA transfer
- You call `copy.wait()` to synchronize
- `pltpu.emit_pipeline` automates this by double-buffering tiles
- `pl.run_scoped` allocates scratch VMEM with automatic lifetime management

#### 4. MXU Systolic Array Operation

The MXU computes: `bfloat16[8,128] @ bfloat16[128,128] -> float32[8,128]` every 8 cycles. Weights flow downward, activations enter from the left.

#### 5. Register Tile Sizes by Dtype

| dtype | Tile shape (sublanes x lanes) |
|-------|------|
| float32/int32 | 8 x 128 |
| bfloat16/float16 | 16 x 128 |
| int8 | 32 x 128 |

#### 6. Reduction Performance on TPU

- **Along sublanes** (dim -2, size 8): fast via shuffle
- **Along lanes** (dim -1, size 128): **slow**, requires Cross-Lane Unit (XLU)
- Only float sum/max/min and bool any/all supported; **no integer reductions**

#### 7. Grid Execution Model

On TPU, grid iterations execute **lexicographically and sequentially** on a single core (unlike GPU where blocks run in parallel). This means consecutive iterations can skip redundant HBM transfers when accessing the same slice.

### Pallas Programming Model

**Key APIs you need to understand:**

1. **`pl.pallas_call(kernel, out_shape, grid, in_specs, out_specs)`** — The entry point. Like `triton.jit` but with upfront tiling declaration.

2. **`pl.BlockSpec(block_shape, index_map)`** — Declares how each tensor is tiled. The `index_map` maps grid indices to tile offsets. No equivalent in Triton (Triton computes offsets inside the kernel).

3. **`pltpu.emit_pipeline(body_fn, grid=...)(refs...)`** — Software-pipelined loop. Automatically prefetches next tile while computing current. Like Triton's `num_stages` but explicit.

4. **`jax.lax.fori_loop(lower, upper, body_fn, carry)`** — Functional for-loop. No Python for-loops on traced values. Carry state threads accumulators.

5. **`pl.run_scoped(fn, *scratch_shapes)`** — Allocates VMEM scratch with RAII lifetime. Like `__shared__` allocation but scoped.

6. **`pltpu.make_async_copy(src_ref, dst_ref, sem)`** — Manual DMA. Like CUDA `cp.async` but for full tiles.

7. **`lax.cond(pred, true_fn, false_fn)`** — Conditional execution. Both branches must return same structure. No runtime branching.

### Key Architectural Differences from GPU (Concise)

| Aspect | GPU (Triton/CUDA) | TPU (Pallas) |
|--------|-------------------|-------------|
| Threading model | SIMT: 32 threads/warp | SIMD: 128 lanes x 8 sublanes |
| Data movement | Implicit (cache hierarchy) | Explicit (DMA HBM<->VMEM) |
| Tile shape | 1D per thread block axis | 2D register tiles (128x8) |
| Block size constraints | Powers of 2, <=1024 threads | Lane (128) and sublane (8) multiples |
| Matmul unit | Per-warp Tensor Cores | Chip-level 128x128 MXU |
| Reductions | Warp shuffles + shared mem | jnp.sum/max on tile, functional carry |
| Branching | Warp divergence | lax.cond (no divergence) |
| Atomics | Yes (atomicAdd, etc.) | No atomics |
| Shared memory | 48-228 KB per SM | 16-32 MB VMEM per core |
| Scheduling | Hardware warp scheduler | Static schedule from compiler |
| Synchronization | __syncthreads, barriers | DMA semaphores |

### TPU Generation Comparison

| Aspect | v4 | v5e | v5p | v6e (Trillium) | v7 (Ironwood) |
|--------|-----|------|------|------|------|
| MXU Array | 128x128 | 128x128 | 128x128 | 256x256 | Not disclosed |
| MXUs/TensorCore | 4 | 4 | 4 | 2 | Not disclosed |
| TensorCores/chip | 2 | 1 | 2 | 1 | 2 (dual-chiplet) |
| HBM Capacity | 32 GB | 16 GB | 95 GB | 32 GB | 192 GB |
| HBM Bandwidth | 1200 GB/s | 800 GB/s | 2765 GB/s | 1600 GB/s | 7370 GB/s |
| BF16 TFLOPs/chip | 275 | 197 | 459 | 918 | 2307 |
| ICI Bandwidth | - | 400 GB/s | 1200 GB/s | 800 GB/s | 1200 GB/s |
| Topology | 3D torus | 2D torus | 3D torus | 2D torus | 3D torus |

---

## Current Pallas Feature Coverage

### Still Unsupported on Pallas (what you'll likely need to work on)

**Language features not yet implemented:**
- Atomic operations (`atomic_add`, `atomic_cas`) — TPU has no atomics
- `while` loops — would need `lax.while_loop`
- `hl.split` / `hl.join` — tensor split/concatenation in tile ops
- Block pointer indexing — Triton-specific optimization
- Tensor gather indexing — scatter/gather patterns
- `flex_attention` — requires closures/torch.compile integration
- Persistent kernel barriers — `hl.barrier()` not yet for Pallas
- `hl.dot` operation — missing on Pallas (vs aten matmul which works)

**Hardware/dtype limitations:**
- **float16** — TPU Mosaic does NOT support float16; use `bfloat16` instead (this is why `HALF_DTYPE` was created)
- **int64** — causes MLIR type mismatch on TPU
- **Non-power-of-2 reduction dims** — TPU alignment requirement
- **Large 4D tensors** — may exceed VMEM

### Test Coverage Summary
- **107 xfail/skip decorators** across 11 test files
- **~30-40% of Helion features** work on Pallas
- The biggest gaps: atomics, distributed ops, advanced tiling patterns, and some control flow

---

## Key Codebase Files to Read

| File | Purpose |
|------|---------|
| `helion/_compiler/backend.py:857` | `PallasBackend` class — the entire TPU backend interface |
| `helion/runtime/__init__.py` | The 3 launcher functions (default, pipeline, fori) |
| `helion/language/_tracing_ops.py` | Pallas-specific codegen for loops and conditionals |
| `test/test_pallas.py` | Concrete examples of what works |
| `helion/_compiler/device_function.py:207` | `ScratchArg` and VMEM scratch management |
| `helion/_compiler/tile_strategy.py` | `EmitPipelineLoopState`, `ForiLoopState` |
| `helion/_compiler/generate_ast.py` | `add_emit_pipeline_loop()`, `add_fori_loop()` |
| `helion/_compiler/aten_lowering.py` | Operation lowering: `_pallas_argreduce()`, `_pallas_dot()` |
| `helion/_compiler/matmul_utils.py` | `_emit_pallas_matmul()` with f32 accumulator support |

---

## Essential Learning Resources (Ranked)

1. **"How to Think About TPUs"** (JAX Scaling Book) — https://jax-ml.github.io/scaling-book/tpus/ — the single best resource for understanding TPU hardware
2. **"Writing TPU Kernels with Pallas"** — https://docs.jax.dev/en/latest/pallas/tpu/details.html — essential reference for the programming model
3. **"When XLA Isn't Enough: From Pallas to VLIW"** — https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas — deep-dive into the full compilation pipeline (Pallas -> Mosaic -> LLO -> VLIW)
4. **Pallas Quickstart** — https://docs.jax.dev/en/latest/pallas/quickstart.html
5. **Grid and BlockSpec tutorial** — https://docs.jax.dev/en/latest/pallas/grid_blockspec.html
6. **Scalar Prefetch and Block-Sparse** — https://docs.jax.dev/en/latest/pallas/tpu/sparse.html
7. **TPU v4 paper** (Jouppi et al., ISCA 2023) — https://arxiv.org/abs/2304.01433
8. **PyTorch/XLA Pallas Integration** — https://docs.pytorch.org/xla/master/features/pallas.html
9. **Flash Attention in Pallas tutorial** — https://blog.vikrampawar.com/pallas-flash-attn.html
