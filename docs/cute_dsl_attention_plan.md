# Plan: Helion Lowering to Performant CuTe DSL Attention

**Date**: 2026-04-13
**Scope**: End-to-end plan for generating performant fused attention kernels in CuTe DSL from Helion's `@helion.kernel` DSL, targeting H100 (SM90a / Hopper).

---

## 1. The User's Code

A Helion user writes attention like this (from `examples/attention.py`):

```python
@helion.kernel
def attention(q, k, v):
    out = torch.empty_like(q)
    batch, m_dim = q.shape[0], q.shape[1]
    for tile_b, tile_m in hl.tile([batch, m_dim]):
        m_i = hl.full(..., float("-inf"))
        l_i = torch.full_like(..., 1.0)
        acc = hl.zeros(..., dtype=float32)
        q_tile = q[tile_b, tile_m, :]

        for tile_n in hl.tile(v.size(1)):         # KV sequence loop
            k_tile = k[tile_b, :, tile_n]
            qk = torch.bmm(q_tile, k_tile)        # GEMM-I: Q @ K^T
            m_ij = torch.maximum(m_i, amax(qk))   # online softmax
            p = torch.exp2(qk - m_ij)
            alpha = torch.exp2(m_i - m_ij)
            acc = acc * alpha                      # rescale previous O
            l_i = l_i * alpha + torch.sum(p, -1)
            v_tile = v[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p, v_tile)    # GEMM-II: O += P @ V
            m_i = m_ij

        out[tile_b, tile_m, :] = acc / l_i
    return out
```

Today this works with the **Triton backend**. The goal: make it generate performant **CuTe DSL** code.

---

## 2. Target Generated CuTe DSL Code

Working backwards from SOTA kernels (CUTLASS FMHA, FlashAttention-4), the generated CuTe DSL kernel should look like:

```python
@cutlass.kernel
def attention_kernel(
    Q, K, V, O,
    tma_atom_k: cute.CopyAtom,
    tma_atom_v: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    # ... layouts, pipeline params
):
    # ---- Setup ----
    smem_q = alloc_smem(...)     # persistent Q tile (loaded once)
    smem_k = alloc_smem(...)     # streamed K tiles (multi-buffered for pipeline)
    smem_v = alloc_smem(...)     # streamed V tiles (multi-buffered for pipeline)
    smem_p = alloc_smem(...)     # staging buffer: GEMM-I output → GEMM-II input

    thr_mma = tiled_mma.get_slice(thread_idx)
    o_frag = make_fragment(thr_mma.partition_shape_C((bm, bd)), Float32)
    for _i in range(cute.size(o_frag)):
        o_frag[_i] = Float32(0.0)

    row_max = Float32(-inf)      # per-thread, carried across iterations
    row_sum = Float32(0.0)       # per-thread, carried across iterations

    # Load Q tile once (persistent)
    # ... TMA load Q → smem_q ...

    # ---- Main KV Loop ----
    for kv_tile in range(num_kv_tiles):

        # ---- TMA load K[kv_tile] → smem_k ----
        # ... pipeline producer: acquire, TMA copy, commit ...

        # ---- GEMM-I: S = Q @ K^T ----
        s_frag = make_fragment(thr_mma.partition_shape_C((bm, bn)), Float32)
        for _i in range(cute.size(s_frag)):
            s_frag[_i] = Float32(0.0)
        cute.gemm(tiled_mma, s_frag, partition_A(smem_q), partition_B(smem_k), s_frag)

        # ---- Online softmax (fragment-level, no SMEM roundtrip) ----
        # Step 1: row-max of s_frag
        new_max = Float32(-inf)
        for _i in range(cute.size(s_frag)):
            new_max = max(new_max, s_frag[_i])
        new_max = cute.arch.warp_reduction_max(new_max, threads_per_row)

        # Step 2: rescale previous accumulator
        scale = cute.exp2((row_max - new_max) * Float32(1.4426950408889634))
        for _i in range(cute.size(o_frag)):
            o_frag[_i] = o_frag[_i] * scale
        row_sum = row_sum * scale

        # Step 3: P = exp2((S - new_max) * log2e)
        for _i in range(cute.size(s_frag)):
            s_frag[_i] = cute.exp2((s_frag[_i] - new_max) * Float32(1.4426950408889634))

        # Step 4: row-sum update
        local_sum = Float32(0.0)
        for _i in range(cute.size(s_frag)):
            local_sum = local_sum + s_frag[_i]
        row_sum = row_sum + cute.arch.warp_reduction_sum(local_sum, threads_per_row)

        row_max = new_max

        # ---- Stage P: fragment → SMEM (one roundtrip) ----
        tCsP = thr_mma.partition_C(smem_p)
        for _i in range(cute.size(tCsP)):
            tCsP[_i] = Float16(s_frag[_i])       # cast f32 → f16 for GEMM-II input
        cute.arch.sync_threads()

        # ---- TMA load V[kv_tile] → smem_v ----
        # ... pipeline producer: acquire, TMA copy, commit ...

        # ---- GEMM-II: O += P @ V ----
        cute.gemm(tiled_mma, o_frag, partition_A(smem_p), partition_B(smem_v), o_frag)

        cute.arch.sync_threads()

    # ---- Final normalization ----
    for _i in range(cute.size(o_frag)):
        o_frag[_i] = o_frag[_i] / row_sum

    # ---- Write O to SMEM → global ----
    tCsO = thr_mma.partition_C(smem_o)
    for _i in range(cute.size(tCsO)):
        tCsO[_i] = o_frag[_i]
    cute.arch.sync_threads()
    # ... store to global memory ...
```

---

## 3. The Three Hard Problems

### 3.1 Problem 1: Multi-MMA Loop Body (Architectural)

The current `_emit_mma_pipeline()` is a monolithic 800-line function (`cute_mma.py` lines 515-1343) that assumes it owns the entire loop body. It generates `load → sync → MMA → readback` as one atomic unit. For attention, we need two MMAs with softmax between them.

**The right solution is NOT to extend `_emit_mma_pipeline()`** — that function is at its complexity limit. Instead, build a **FragmentCodegenWalker** that processes the FX graph node-by-node:

```python
class FragmentCodegenWalker:
    """Walk FX graph nodes, emitting fragment-level CuTe DSL code."""

    def __init__(self, cg, tiled_mma, thr_mma, ...):
        self.fragments = {}       # node → fragment variable name
        self.smem_buffers = {}    # for inter-MMA staging

    def emit_node(self, node):
        if is_mma_node(node):         → emit cute.gemm()
        if is_load_node(node):        → emit TMA copy or scalar load to SMEM
        if is_pointwise(node):        → emit element-wise fragment loop
        if is_reduction(node):        → emit warp shuffle reduction
        if needs_staging(node):       → emit partition_C → SMEM → partition_A
```

This is analogous to how Helion's existing scalar codegen walks the graph — but operating on fragments instead of scalars.

**Fragment state propagation**: When a node's input comes from an MMA (or from another fragment-mode node), it stays in fragment mode. When it needs to feed a store or an incompatible op, it transitions to scalar mode via SMEM readback. This decision is made by checking:
- Is the input in the `self.fragments` dict? → fragment mode
- Is the target in the `FRAGMENT_OPS` registry? → stays in fragment mode
- Otherwise → SMEM readback to scalar

The existing `_emit_mma_pipeline()` stays for single-MMA loops (130 tests pass, well-tested). The `FragmentCodegenWalker` handles multi-MMA graphs.

### 3.2 Problem 2: Fragment-Level Reductions (CuTe Layout)

Softmax needs `row_max(S)` and `row_sum(P)` — reductions along axis 1 of the MMA fragment. MMA fragments have non-obvious thread-to-element mappings that depend on the atom.

**Using CuTe's layout algebra** (not Triton's Z2 LinearLayout), we avoid hardcoding per-atom mappings AND avoid the power-of-2 shape restriction. The generated code queries the MMA's partition layout:

```python
# At codegen time, the Helion compiler knows:
#   - MMA impl (universal/warp/tcgen05) from _choose_mma_impl()
#   - Tile shape (bm, bn) from config
#   - The MMA atom determines the partition layout

# For warp MMA (16×8×16):
#   - Each warp (32 threads) holds a 16×8 output tile
#   - Thread t holds elements at known (row, col) positions
#   - CuTe's partition_shape_C() gives the per-thread value shape
#   - From value shape, we derive: elements_per_row, threads_per_row

# For universal MMA:
#   - Each thread holds 1 element → trivial reduction (all cross-thread)

# Generated CuTe DSL:
# Within-thread: reduce fragment elements sharing a row
local_max = s_frag[_row_start]
for _col in range(1, _cols_per_thread):
    local_max = max(local_max, s_frag[_row_start + _col])

# Cross-thread: warp shuffle for threads sharing the same row
row_max = cute.arch.warp_reduction_max(local_max, threads_in_group=_threads_per_row)
```

**Key point**: `_cols_per_thread` and `_threads_per_row` are computed at codegen time from the MMA atom's partition properties. The Helion compiler queries CuTe (via `thr_mma.partition_shape_C()`) rather than maintaining per-atom lookup tables.

This works for any MMA atom — warp (16×8), warpgroup (64×N), TCGEN05 (128×8), or future GB300 NVFP4 atoms with non-PO2 shapes — because CuTe's layout algebra has no power-of-2 restriction.

### 3.3 Problem 3: Loop-Carried Fragment State

The attention loop carries state across iterations:

| Variable | Type | Lifetime |
|----------|------|----------|
| `o_frag` | MMA fragment (many elements/thread) | Entire KV loop |
| `row_max` | Scalar (one per thread per row) | Entire KV loop |
| `row_sum` | Scalar (one per thread per row) | Entire KV loop |
| `s_frag` | MMA fragment | One iteration only (temporary) |

The compiler needs to:
1. **Identify loop-carried values** in the FX graph — nodes whose definitions reach the loop back-edge
2. **Distinguish fragment carries** (`o_frag`) from **scalar carries** (`row_max`, `row_sum`)
3. **Emit initialization** in `outer_prefix` and **readback** in `outer_suffix`

This generalizes the current K-loop accumulator handling (which already carries `acc_frag` across K iterations) to support multiple carried values of different types.

---

## 4. Implementation Phases

### Phase 1: Fragment Mode Codegen (Correctness)

**Goal**: Attention produces correct output with CuTe backend. Scalar loads, no pipelining, but correct.

| Component | What | Est. Lines |
|-----------|------|-----------|
| Lift exclusivity | Replace `_mma_loop_is_exclusive()` with `_classify_loop_ops()` | ~50 |
| FragmentCodegenWalker | Node-by-node codegen: MMA, pointwise, reduction, staging | ~300 |
| Fragment pointwise | `for _i in range(frag_size): frag[_i] = op(frag[_i])` | ~60 |
| Fragment reduction | Within-thread reduce + `warp_reduction_max/sum` | ~120 |
| SMEM staging | `partition_C` → SMEM → `sync` → `partition_A` for inter-MMA | ~50 |
| Loop-carried state | Multiple fragment + scalar carries across iterations | ~60 |
| **Total** | | **~640** |

**Key design decisions**:

- **FragmentCodegenWalker lives alongside `_emit_mma_pipeline()`**, not replacing it. Single-MMA loops continue using the existing path. Multi-MMA loops use the walker.
- **Fragment op registry** is extensible: `{aten.add, aten.sub, aten.mul, aten.div, aten.exp2, aten.neg, aten.maximum, aten.minimum, ...}`. Adding a new op = one line.
- **Reduction uses CuTe layout algebra**: `threads_per_row` and `elements_per_row` derived from `thr_mma.partition_shape_C()` at codegen time. No hardcoded per-atom tables.
- **SMEM staging between MMAs**: One roundtrip (fragment → SMEM via `partition_C`, next MMA reads via `partition_A`). CuTe handles the layout conversion through the partition operations.

**Testing**: Run attention example with `HELION_BACKEND=cute`. Compare output to Triton backend result. All 130 existing CuTe tests must still pass (single-MMA path unchanged).

### Phase 2: TMA + Pipelining (Performance)

**Goal**: Replace scalar loads with TMA, add multi-stage pipelining. Benefits all MMA kernels (single-MMA GEMM and multi-MMA attention).

| Component | What | Est. Lines |
|-----------|------|-----------|
| TMA atom creation | In `@cute.jit` wrapper, passed as `cute.CopyAtom` kernel params | ~100 |
| TMA kernel args | New `TmaAtomArg` type in `device_function.py` | ~50 |
| Swizzled SMEM | `hopper_helpers.make_smem_layout_a/b()` for SM90 | ~50 |
| PipelineTmaAsync | Producer/consumer pipeline with configurable `num_stages` | ~100 |
| Prologue/drain | Pre-fill stages in prefix, drain in suffix | ~80 |
| num_stages config | Add to CuteBackend `supports_config_key()` | ~10 |
| Runtime wrapper | Generate TMA atom creation in `_create_cute_wrapper()` | ~80 |
| **Total** | | **~470** |

**Attention-specific pipelining**:
- Q: persistent in SMEM (loaded once before KV loop, not pipelined)
- K: multi-buffered, TMA-loaded per KV iteration (pipelined)
- V: multi-buffered, TMA-loaded per KV iteration (pipelined)
- P staging buffer: single buffer (produced and consumed within one iteration, no pipelining needed)

**Testing**: First verify on existing single-MMA GEMM tests with `HELION_CUTE_MMA_IMPL=warp` + TMA. Then verify attention correctness + measure speedup.

### Phase 3: Warpgroup Specialization (Peak Performance)

**Goal**: Separate TMA and MMA into different warp groups for maximum compute/memory overlap. Needed to match FA4's 71% GPU utilization.

| Component | What | Est. Lines |
|-----------|------|-----------|
| WarpGroup abstraction | Role assignment (compute, TMA, softmax) | ~100 |
| Warp-ID switch codegen | Different code paths per warp group | ~150 |
| Named barriers | Generalize TCGEN05's NamedBarrier usage | ~80 |
| Register budget | `warpgroup_reg_alloc/dealloc` per role | ~40 |
| Warpgroup MMA | Enable `_choose_mma_impl()` to select "warpgroup" | ~100 |
| Pipeline coordination | PipelineTmaAsync with separate producer/consumer groups | ~100 |
| **Total** | | **~570** |

**Not required for correctness** — Phases 1+2 produce correct, reasonably fast attention. Phase 3 is for competitive performance.

---

## 5. SMEM Budget Analysis

For attention on H100 (228KB SMEM limit):

| Buffer | Size (bm=128, D=128, bN=64, f16) | Notes |
|--------|-----------------------------------|-------|
| Q (persistent) | 128 × 128 × 2 = 32 KB | Loaded once |
| K (2 stages) | 64 × 128 × 2 × 2 = 32 KB | Pipelined TMA |
| V (2 stages) | 64 × 128 × 2 × 2 = 32 KB | Pipelined TMA |
| P (staging) | 128 × 64 × 2 = 16 KB | GEMM-I → GEMM-II |
| O (readback) | 128 × 128 × 4 = 64 KB | f32 accumulator → SMEM |
| **Total** | **176 KB** | **Fits in 228 KB** |

With 3 pipeline stages: 176 + 32 = **208 KB** (fits).
With 4 pipeline stages: 176 + 64 = **240 KB** (does NOT fit).

**Autotuner range for `num_stages`**: [2, 3] for typical attention tile sizes on H100.

Note: O readback (64 KB) only happens in `outer_suffix` after the KV loop. During the loop, O lives in registers (the `o_frag` fragment). So the loop-body SMEM footprint is 176 - 64 = **112 KB**, leaving room for larger tiles or more pipeline stages.

---

## 6. Attention Loop Structure Detail

The KV loop has a specific structure that differs from GEMM's K-loop:

```
GEMM K-loop:                           Attention KV-loop:
  Load A[k], B[k] → SMEM                 Load K[n] → SMEM
  S += A × B (accumulate)                S = Q × K^T (fresh each iter)
  (single MMA, single accumulator)       softmax(S) → P
                                         Load V[n] → SMEM
                                         O += P × V (accumulate)
                                         (two MMAs, two fragments, softmax between)
```

Key differences:
1. **Two GEMMs per iteration** with different operands and reduction dims
2. **Intermediate computation** (softmax) between GEMMs, operating on fragment values
3. **GEMM-I produces a temporary** (`s_frag`), GEMM-II accumulates into a persistent carry (`o_frag`)
4. **Q is persistent** — loaded once, reused every iteration
5. **K and V are streamed** — new tile each iteration (pipelineable)

The GEMM-I inner K-loop (over head dimension D) may exist if D > bk:
```
for kv_tile in range(num_kv_tiles):     # Outer: KV sequence
    s_frag = zeros(...)
    for d_tile in range(D // bk):       # Inner: head dimension (if D > bk)
        cute.gemm(..., s_frag, ...)     # Accumulate S
    softmax(s_frag)
    cute.gemm(..., o_frag, ...)         # GEMM-II (bN is the K-dim here)
```

For typical head_dim=128 and bk=16, there are 8 inner K iterations for GEMM-I. The FragmentCodegenWalker needs to handle this nested loop structure.

---

## 7. File Modification Map

### Phase 1 (Fragment Mode)

| File | Lines Changed | What |
|------|--------------|------|
| `helion/_compiler/cute/cute_mma.py` | ~400 | FragmentCodegenWalker, classify_loop_ops, fragment ops |
| `helion/_compiler/cute/fragment_ops.py` | ~150 (new) | Fragment pointwise + reduction codegen helpers |
| `helion/_compiler/backend.py` | ~30 | Detect multi-MMA loops in `_detect_mma_loop()` |
| `test/test_cute_backend.py` | ~60 | Attention test, multi-MMA tests |

### Phase 2 (TMA + Pipelining)

| File | Lines Changed | What |
|------|--------------|------|
| `helion/_compiler/cute/cute_mma.py` | ~200 | TMA load path, pipeline management |
| `helion/_compiler/cute/tma_support.py` | ~100 (new) | TMA atom helpers, capability detection |
| `helion/_compiler/device_function.py` | ~50 | TmaAtomArg type |
| `helion/runtime/__init__.py` | ~80 | TMA atom creation in cute wrapper |
| `helion/_compiler/backend.py` | ~10 | `supports_config_key("num_stages")` |
| `test/test_cute_backend.py` | ~40 | TMA + pipeline tests |

### Phase 3 (Warpgroup Specialization)

| File | Lines Changed | What |
|------|--------------|------|
| `helion/_compiler/cute/cute_mma.py` | ~300 | Warp-ID switch, role assignment |
| `helion/_compiler/cute/mma_support.py` | ~30 | Enable warpgroup selection |
| `helion/_compiler/cute/warp_specialization.py` | ~200 (new) | WarpGroup abstraction, barriers |
| `test/test_cute_backend.py` | ~40 | Warpgroup tests |

---

## 8. Why CuTe Layout (Not Triton's Z2 LinearLayout)

The fragment-level reduction (Problem 2) requires knowing which fragment elements belong to the same row. Triton solves this with `LinearLayout` — matrices over GF(2). But LinearLayout has a fundamental limitation: **all dimensions must be powers of 2** (`assert(isPowerOf2_32(size))` throughout `LinearLayout.cpp`).

CuTe layouts use `(Shape, Stride)` tuples with hierarchical nesting and standard integer arithmetic. No PO2 restriction. Since Helion already generates CuTe DSL code, CuTe's layout algebra is free:

- `thr_mma.partition_C(tensor)` → CuTe layout encoding thread-to-element mapping
- `thr_mma.partition_shape_C(tile_shape)` → per-thread value shape (gives elements_per_row)
- `cute.composition(A, B)` → compose layouts for conversion
- `cute.right_inverse(L)` → invert a layout mapping

The Helion compiler emits these calls in the generated kernel. Layout questions are answered by CuTe at kernel compile time, not by Helion's Python compiler.

**Advantages over Triton's Z2**:
- No power-of-2 restriction → future-proof for GB300 NVFP4
- Layout algebra is richer (complement, logical_divide, hierarchical nesting)
- Lower Helion compiler complexity (delegates to CuTe)
- MMA fragment layout knowledge comes from CuTe's own partition operations

---

## 9. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Fragment reduction correctness (row mapping) | Start with universal MMA (1 elem/thread, trivial) before warp MMA |
| SMEM budget overflow | Autotuner constrains tile sizes; O readback only in suffix |
| CuTe DSL API changes | Pin `nvidia-cutlass-dsl` version; TCGEN05 path already proves API stability |
| Phase 1 performance too slow (scalar loads) | Acceptable for correctness validation; Phase 2 adds TMA |
| Warpgroup MMA complexity | Deferred to Phase 3; warp MMA + TMA is sufficient for Phase 2 |
| Nested K-loop for GEMM-I (D > bk) | Existing K-loop handling in `_emit_mma_pipeline()` provides template |

---

## 10. Success Criteria

| Phase | Criterion |
|-------|-----------|
| Phase 1 | `HELION_BACKEND=cute python examples/attention.py` produces correct output (matches Triton within atol=1e-2). All 130 existing CuTe tests still pass. |
| Phase 2 | Attention kernel uses TMA loads (verified by `HELION_PRINT_OUTPUT_CODE=1` showing no scalar load loops). Performance within 3x of FA4. |
| Phase 3 | Performance within 1.5x of FA4. Warpgroup MMA + TMA + named barriers visible in generated code. |
