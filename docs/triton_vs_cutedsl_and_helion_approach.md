# Triton DSL vs CuTe DSL: Key Differences and Helion's Approach

**Date**: 2026-04-13
**Context**: Analysis of how Helion should bridge the gap between user-friendly DSL and performant CuTe DSL code generation, informed by deep study of both Triton (cloned and read source) and CuTe DSL (CUTLASS examples, API inspection).

---

## 1. Abstraction Level

### Triton DSL — Block-level, compiler-managed

The programmer writes in terms of **blocks/tiles** (`tl.load`, `tl.dot`, `tl.store`). The compiler decides how threads map to data, how registers are laid out, and how data moves between memory levels. The programmer never sees threads, warps, or registers.

```python
# Triton: block-level abstraction
@triton.jit
def matmul_kernel(A, B, C, ...):
    offs_m = tl.arange(0, BLOCK_M)
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])  # block load
    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.dot(a, b)                                       # block MMA
    tl.store(C + ..., acc)                                    # block store
```

### CuTe DSL — Thread-level, programmer-managed

The programmer writes in terms of **threads and their partitions** — `thr_mma.partition_A(smem)`, `cute.copy(tma_atom, ...)`, explicit `for _i in range(cute.size(frag))` loops over fragment elements. SMEM allocation, TMA atoms, pipeline barriers, and warp synchronization are all explicitly managed.

```python
# CuTe DSL: thread-level abstraction
@cutlass.kernel
def matmul_kernel(A, B, C, tma_a: cute.CopyAtom, tiled_mma: cute.TiledMma, ...):
    smem_a = alloc_smem(...)
    thr_mma = tiled_mma.get_slice(thread_idx)
    acc = make_fragment(thr_mma.partition_shape_C(...), Float32)
    cute.copy(tma_a, global_partition, smem_partition, tma_bar_ptr=barrier)
    cute.gemm(tiled_mma, acc, partition_A(smem_a), partition_B(smem_b), acc)
    for _i in range(cute.size(acc)):
        # explicit per-element access
        ...
```

---

## 2. Layout System — The Deepest Difference

### Triton: Z2 LinearLayout (Matrices over GF(2))

- All dimensions **must be powers of 2** (`assert(isPowerOf2_32(size))` in `LinearLayout.h`).
- Layout conversions are matrix operations over GF(2) — elegant but restrictive.
- **Cannot represent GB300 NVFP4 tensor core shapes** (non-PO2). As Vijay noted: *"Triton PO2 tile shapes limitation from their usage of Z2 layouts is absurd in my mind. This has real world downsides like not being able to describe programs that map to GB300 NVFP4 tensor core."*
- Triton's own source acknowledges this: `LinearLayout.h` lines 281-313 explicitly compare to CuTe and note CuTe's superiority for non-PO2 shapes.

### CuTe: (Shape, Stride) Tuples — Hierarchical Layout Algebra

- **No power-of-2 restriction**. Any integer shape works.
- Supports hierarchical nesting (e.g., `((4,8), (2,4))` for warpgroup decomposition).
- Rich algebra with operations not available in Z2:
  - `composition(A, B)` — compose two layouts
  - `complement(L, N)` — find the "missing" layout that tiles the rest of the space
  - `logical_divide(L, tile)` — decompose a layout by a tiling
  - `left_inverse(L)` / `right_inverse(L)` — layout inverses
  - `make_layout_tv(thread_layout, value_layout)` — construct thread-value decomposition
- Swizzle is a separate concern via `ComposedLayout`, not baked into the representation.

### Comparison Table

| Aspect | Triton (Z2 LinearLayout) | CuTe Layout |
|--------|-------------------------|-------------|
| **Shape restriction** | **PO2 only** | **Any integer** |
| **GB300 NVFP4** | **Cannot represent** | **Native support** |
| **Representation** | Matrices over GF(2) | (Shape, Stride) tuples |
| **Nesting** | Flat only | Hierarchical |
| **Composition** | Matrix multiply over GF(2) | `cute.composition(A, B)` |
| **Inverse** | Matrix inverse over GF(2) | `cute.left_inverse` / `right_inverse` |
| **Complement** | Not directly available | `cute.complement(L, N)` |
| **Swizzle** | Built into Z2 layout | Separate `ComposedLayout` |

---

## 3. Layout Propagation & Conversion

### Triton: Compiler-driven

The compiler does layout propagation automatically. Elementwise ops inherit the MMA encoding from their inputs (`RemoveLayoutConversions.cpp`). When layouts conflict, the compiler inserts conversion operations and chooses the cheapest strategy by analyzing LinearLayout matrices:
- **Register permutation** — if the conversion is a simple register rename
- **Warp shuffle** — if data needs to move between threads in the same warp
- **SMEM roundtrip** — fallback for arbitrary layout conversions

This is ~thousands of lines of C++ in `lib/Conversion/TritonGPUToLLVM/`.

### CuTe DSL: Generated code handles it

Layout conversion is handled through CuTe's partition API in the generated code. `partition_C()` gives the accumulator layout, `partition_A()` gives the operand layout. When conversion is needed (e.g., GEMM-I output feeds GEMM-II input):

```python
# Write fragments to SMEM using accumulator layout
tCsP = thr_mma.partition_C(smem_p)
for _i in range(cute.size(tCsP)):
    tCsP[_i] = p_frag[_i]

sync_threads()

# Read from SMEM using operand layout
tAsP = thr_mma.partition_A(smem_p)
```

CuTe handles the thread-to-element mapping internally. The compiler doesn't need to understand the layout details.

---

## 4. Multi-MMA / Attention Support

### Triton: No special-casing needed

You just write two `tl.dot` calls in the same loop body (see `06-fused-attention.py`). The compiler's layout propagation and conversion machinery handles everything generically:

```python
# Triton attention — just two dots, no ceremony
qk = tl.dot(q, k)          # GEMM-I
p = tl.math.exp2(qk - m)   # pointwise on MMA output
acc += tl.dot(p, v)         # GEMM-II
```

### CuTe DSL: Explicit management required

You must explicitly manage the data flow between GEMMs — allocate separate accumulators, write fragments to SMEM, read back as next MMA's input, manage barriers:

```python
# CuTe DSL attention — explicit orchestration
cute.gemm(tiled_mma, s_frag, sQ, sK, s_frag)     # GEMM-I
for _i in range(cute.size(s_frag)):                 # softmax on fragments
    s_frag[_i] = exp2(s_frag[_i] - row_max)
tCsP = thr_mma.partition_C(smem_p)                 # fragment → SMEM
for _i in range(cute.size(tCsP)):
    tCsP[_i] = s_frag[_i]
sync_threads()
tAsP = thr_mma.partition_A(smem_p)                 # SMEM → next MMA input
cute.gemm(tiled_mma, o_frag, tAsP, sV, o_frag)    # GEMM-II
```

More code, but more control over every detail.

---

## 5. Memory & Pipelining

### Triton: Compiler-driven software pipelining

Pipelining is handled by `SoftwarePipeliner.cpp`. The programmer writes simple sequential code; the compiler restructures into prologue/main/epilogue with multi-buffering automatically.

### CuTe DSL: Explicit pipeline management

You create `PipelineTmaAsync`, call `producer_acquire`, `producer_commit`, `consumer_wait`, `consumer_release` yourself. More verbose, but you control stage count, buffer placement, and barrier semantics precisely.

```python
pipeline = PipelineTmaAsync.create(num_stages=3, ...)
pipeline.producer_acquire(stage)
cute.copy(tma_atom, gmem, smem[stage], tma_bar_ptr=barrier)
pipeline.producer_commit(stage)
# ...
pipeline.consumer_wait(stage)
cute.gemm(...)
pipeline.consumer_release(stage)
```

---

## 6. Hardware Portability vs Peak Performance

| | Triton | CuTe DSL |
|---|--------|---------|
| **Portability** | Same code runs on SM80/SM90/SM100 (compiler retargets) | Must explicitly choose MMA impl, TMA vs scalar loads, pipeline type |
| **Peak perf** | Compiler may miss optimizations | Full control — can match hand-tuned CUTLASS |
| **New hardware** | Compiler team must add backend support | New atoms/pipelines available immediately via CUTLASS updates |
| **Learning curve** | Low (block-level thinking) | High (thread-level, layout algebra) |

---

## 7. Helion's Approach: Transpiler, Not Compiler

### The key architectural insight

Helion doesn't need to rebuild Triton's compiler. The compilation targets are fundamentally different:

```
Triton:   user DSL → [layout algebra, register mapping, PTX emission] → machine code
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                      ~100K lines of C++ (MLIR passes, LLVM codegen)

Helion:   user DSL → [codegen planning] → CuTe DSL Python → (CUTLASS handles the rest)
                      ~~~~~~~~~~~~~~~~~
                      ~600-800 lines of Python
```

Triton compiles `tl.dot` all the way down to PTX. Along the way it must **invent layouts** (choose MMA encoding, propagate through IR, insert conversions), **lower to registers** (map every block-level op to per-thread register ops using Z2 math), **software pipeline** (analyze memory patterns, restructure control flow), and **emit PTX** (actual machine instructions).

Helion compiles to **CuTe DSL Python code**, not PTX. CuTe DSL is already a high-level target that handles the hard parts. Helion decides **what** CuTe DSL code to emit, but doesn't need to solve **how** layouts map to registers — CuTe does that.

### What Helion actually needs to build

| Triton compiler feature | Helion equivalent | Complexity |
|---|---|---|
| Z2 layout algebra (~5K lines C++) | **Skip** — delegate to CuTe's layout algebra in generated code | None |
| Layout propagation across IR (~3K lines C++) | **Minimal** — classify FX nodes as fragment-compatible or not | ~50 lines |
| Register-level lowering (~10K lines C++) | **Skip** — CuTe DSL's `for _i in range(cute.size(frag))` handles it | None |
| Layout conversion (shuffle/SMEM) (~2K lines C++) | **Simple** — emit `partition_C → SMEM → partition_A` when needed | ~80 lines |
| Reduction on MMA layouts (~1K lines C++) | **Delegate** — query CuTe layout for row structure, emit warp shuffles | ~150 lines |
| Software pipelining (~2K lines C++) | **Explicit** — emit `PipelineTmaAsync` calls (structured, not analysis-based) | ~150 lines |
| TMA load codegen | **Template** — emit `cute.copy(tma_atom, ...)` instead of scalar loops | ~200 lines |

### The analogy

Triton built a **full compiler backend** (MLIR → PTX). Helion is building a **transpiler** (FX graph → CuTe DSL Python). The CuTe DSL runtime + CUTLASS compiler is Helion's backend.

The hard parts Triton solves in C++ (layout math, register allocation, instruction selection) — Helion gets for free from CuTe. What Helion builds is the **codegen planner**: given an FX graph with MMA ops, pointwise ops, and reductions, emit the right sequence of CuTe DSL calls with proper SMEM management and pipelining.

### Why this works — CuTe layout as the secret weapon

1. **No PO2 restriction** — CuTe layouts handle any MMA atom shape, including future GB300 NVFP4
2. **Already the compilation target** — Helion generates CuTe DSL code, so CuTe layouts are native
3. **Simplifies the compiler** — Instead of Helion understanding all MMA layout details, it delegates to `partition_A()`, `partition_C()`, `logical_divide()` etc. in the generated code
4. **Fragment-level operations need no layout knowledge** — `for _i in range(cute.size(frag)): frag[_i] = exp2(frag[_i])` works regardless of the underlying CuTe layout
5. **Reduction uses CuTe layout queries** — The compiler queries the MMA atom's partition layout at codegen time to determine `rows_per_thread`, `cols_per_thread`, `threads_per_row` — works for any atom

### Bottom line

Helion is a **code generator** (~600-800 lines of Python), not a **compiler backend** (~100K lines of C++). It generates expert-level CuTe DSL code from user-friendly Helion DSL, leveraging CuTe's layout algebra as the universal layout system. The user writes `torch.bmm` and `torch.exp2`; Helion emits `cute.gemm`, `partition_C → SMEM → partition_A`, and `PipelineTmaAsync` — matching what a CUTLASS expert would write by hand.
