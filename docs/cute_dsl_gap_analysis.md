Helion CuTe DSL Backend vs SOTA CuTe DSL Kernels: Gap Analysis

# Helion CuTe DSL Backend vs SOTA CuTe DSL Kernels: Gap Analysis

**Date**: 2026-04-10
**Scope**: Comparison of Helion's CuTe DSL backend code generation capabilities against state-of-the-art CuTe DSL-based attention and normalization kernels (FlashAttention-4, CUTLASS FMHA examples, CUTLASS normalization examples).

---

## 1. Executive Summary

Helion's CuTe backend successfully generates correct GEMM kernels with simple epilogues using universal, warp-level, and TCGEN05 (Blackwell) MMA atoms. However, it **cannot currently generate code patterns comparable to SOTA CuTe DSL kernels** for attention or normalization workloads. The gap spans 6-7 critical features that SOTA kernels depend on, with 4 being structural/architectural blockers that would require significant design changes to address.

---

## 2. SOTA Landscape: What Exists Today

### 2.1 CUTLASS Official CuTe DSL Examples

NVIDIA ships complete CuTe DSL Python implementations in `examples/python/CuTeDSL/`:

| Example | Target | Key Patterns |
|---------|--------|-------------|
| `blackwell/fmha.py` | SM100 | 15 specialized warps, TMA, TMEM, multi-pipeline, online softmax |
| `blackwell/fmha_bwd.py` | SM100 | Backward pass (dQ/dK/dV), same architecture |
| `hopper/fmha.py` | SM90 | 3 warpgroups, 5-stage K/V pipeline, WGMMA |
| `ampere/flash_attention_v2.py` | SM80 | CpAsync, manual SMEM management, swizzled layouts |
| `blackwell/rmsnorm.py` | SM100 | Cluster-based reduction, async pipelining |
| `hopper/cta_norm.py` | SM90 | LayerNorm + RMSNorm, warp-level butterfly shuffle |
| `blackwell/mla/` | SM100 | Multi-head Latent Attention decode (FP16/FP8) |
| `blackwell/mamba2_ssd/` | SM100 | Mamba2 state space model |
| `blackwell/mixed_input_fmha/` | SM100 | Decode + prefill d256/d512 |

### 2.2 FlashAttention-4 (Tri Dao)

FA4 is implemented **entirely in CuTe DSL Python** (46 files in `flash_attn/cute/`):

- Achieves **1613 TFLOPS/s BF16** on B200 (71% GPU utilization)
- **1.3x** over cuDNN 9.13, **2.7x** over Triton
- **20-30x faster compile times** than C++ template-based FA3 (~2.5s vs ~55s)
- Supports Hopper (SM90), Blackwell (SM100), and SM120
- Forward + backward passes, variable-length, paged KV cache, GQA, block sparsity
- Integrated into PyTorch via FlexAttention: `torch.compile(partial(flex_attention, kernel_options={"BACKEND": "FLASH"}))`

### 2.3 FlashAttention-3 (Hopper, C++ CUTLASS)

FA3 uses C++ CUTLASS templates (not CuTe DSL Python) but established the architectural patterns that FA4 and CUTLASS FMHA examples follow:

- Multi-stage async TMA pipelines (5 stages for K/V)
- Producer/consumer warpgroup specialization with dynamic register reallocation
- Intra-warpgroup 2-stage pipelining (GEMM-II overlaps with softmax)
- Online softmax with `exp2` and `log2(e)` scaling for FMA fusion

---

## 3. Key Patterns in SOTA Kernels

### 3.1 Software Pipelining

SOTA kernels overlap data movement with computation using multi-stage async pipelines:

```python
# CuTe DSL pattern (from CUTLASS Blackwell FMHA)
ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
    num_stages=self.num_ab_stage,          # typically 2-5
    producer_group=make_thread_cooperative_group([self.tma_warp_id]),
    consumer_group=make_thread_cooperative_group([self.mma_warp_id]),
    barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
    tx_count=self.num_tma_load_bytes,
)

# Producer: issue TMA load into next stage while consumer computes current stage
handle = ab_producer.acquire_and_advance()
cute.copy(tma_atom_a, src, dst, tma_bar_ptr=handle.barrier)
handle.commit()

# Consumer: wait for data, compute, release stage
handle = ab_consumer.wait_and_advance()
cute.gemm(tiled_mma, acc, fragA, fragB, acc)
handle.release()
```

Pipeline types available in CuTe DSL: `PipelineTmaAsync` (Hopper), `PipelineTmaUmma` (Blackwell), `PipelineAsyncUmma`, `PipelineUmmaAsync`, `PipelineClcFetchAsync`, `PipelineTmaMultiConsumersAsync`.

**Helion CuTe status**: No software pipelining for universal/warp MMA. TCGEN05 path has a 2-stage A/B buffer (`ab_stage_count = 2`) and `PipelineUmmaAsync` with `num_stages=1` for accumulators, but this is far from the multi-stage overlapping pattern.

### 3.2 TMA (Tensor Memory Accelerator) Bulk Copies

SOTA kernels use TMA for all global→shared memory transfers:

```python
# CuTe DSL TMA pattern
tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
    a_op, a, a_smem_layout, mma_tiler, tiled_mma, cluster_layout.shape
)

# Partition and issue bulk copy (entire tile in one instruction)
tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
    tma_atom_a, cta_coord, cluster_layout,
    cute.group_modes(sA, 0, 3), cute.group_modes(gA, 0, 3)
)
cute.copy(tma_atom_a, tAgA[None, coord], tAsA[None, handle.index],
          tma_bar_ptr=handle.barrier, mcast_mask=multicast_mask)
```

Key TMA features used:
- **Bulk transfer**: Entire tile (e.g., 128x64 elements) in a single instruction
- **Multicast**: Deliver same tile to multiple CTAs in a cluster
- **Descriptor prefetch**: `prefetch_descriptor()` for latency hiding
- **Cache hints**: `EVICT_LAST` for temporal reuse patterns

**Helion CuTe status**: All global→shared memory loads use scalar element-by-element access with Python if/else masking:
```python
# Helion's current pattern (cute_mma.py)
smem_a[m_local, _k] = A[m_global, _gk] if m_global < M and _gk < K else 0.0
```
This generates ~N instructions per tile element vs 1 instruction for the entire tile with TMA. The instruction count difference is roughly 100-1000x for typical tile sizes.

### 3.3 Warpgroup Specialization

SOTA kernels assign distinct roles to different warps/warpgroups:

**FlashAttention-4 (Blackwell, 15 warps):**
| Warp(s) | Role | Register Budget |
|---------|------|-----------------|
| 0-7 | Softmax (two groups) | Low |
| 8-11 | Correction (exponential rescaling) | Low |
| 12 | MMA (tensor core operations) | High (240+) |
| 13 | TMA (data movement) | Minimal (24-56) |
| 14 | Epilogue (output storage) | Low |

Register reallocation: `warpgroup_reg_dealloc<24>()` / `warpgroup_reg_alloc<240>()` dynamically moves registers between producer and consumer warps.

**Helion CuTe status**: All threads execute the same code path. The `ThreadLayout` model assigns one scalar value per thread with identical roles. `mma_support.py` probes warpgroup MMA capability but `_choose_mma_impl()` never selects it. No mechanism for heterogeneous warp code paths.

### 3.4 Online Softmax

The online softmax algorithm maintains running statistics across K tiles:

```
For each new K tile:
  1. S = Q @ K^T                      # GEMM-I
  2. local_max = rowmax(S)            # warp shuffle reduction
  3. scale = exp2((old_max - new_max) * log2e)
  4. O *= scale                        # rescale previous accumulator
  5. row_sum *= scale                  # rescale previous sum
  6. P = exp2((S - new_max) * log2e)  # softmax numerator
  7. row_sum += rowsum(P)
  8. O += P @ V                        # GEMM-II (accumulate)
  9. old_max = new_max
Final: O *= 1/row_sum                  # normalize
```

Key implementation details:
- All arithmetic in float32 for numerical stability
- Uses `exp2` (not `exp`) with `log2(e)` pre-multiplied, enabling `ffma` instead of separate `fadd` + `fmul`
- FA4 adds **conditional rescaling**: only rescale when `max_new - max_old > tau` (tau=8.0), with final correction pass
- FA4 uses **software-emulated exponentials**: polynomial approximation via Cody-Waite range reduction + Horner's method (degree-3, max error 8.77e-5). Only 10-25% of entries use hardware MUFU.

**Helion CuTe status**: No online softmax in the CuTe MMA pipeline. The `_mma_loop_is_exclusive()` constraint in `cute_mma.py` requires the K-loop body to contain *only* the MMA dataflow — no room for softmax computation between K tiles. The `examples/attention.py` in Helion exists but uses the Triton backend, not CuTe.

### 3.5 Swizzled Shared Memory Layouts

SOTA kernels use swizzled SMEM layouts to avoid bank conflicts:

```python
# CuTe DSL swizzle pattern
a_smem_layout_staged = sm100_utils.make_smem_layout_a(
    tiled_mma, mma_tiler, dtype, num_stages
)
sA = smem.allocate_tensor(
    element_type=dtype,
    layout=a_smem_layout_staged.outer,    # spatial structure
    byte_alignment=128,
    swizzle=a_smem_layout_staged.inner    # bank conflict avoidance
)
```

Swizzle patterns: `SW128` (128-byte), `SW64` (64-byte), `SW32` (32-byte). Standard for all WGMMA operands.

**Helion CuTe status**: Zero references to swizzle/Swizzle/bank_conflict in the cute/ directory. All shared memory uses simple row-major layouts: `cute.make_layout((bm, bk), stride=(bk, 1))`.

### 3.6 Named Barriers and Async Synchronization

SOTA kernels use 15+ named barriers for fine-grained synchronization:

```python
# CuTe DSL barrier patterns
barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=256)
barrier.arrive()   # non-blocking signal
barrier.wait()     # block until barrier satisfied
barrier.sync()     # arrive + wait

# Mbarrier arrays for multi-stage pipelines
mbar = pipeline.MbarrierArray(...)
mbar.arrive(index, dst, cta_group)
mbar.wait(index, phase)
```

FA3 named barriers: `QueryEmpty/QueryFull`, `WarpSchedulerWG1`, `PFull/PEmpty`, `StreamkBarrier0/1`, `MathWgOrderBarrier`, cluster barriers.

**Helion CuTe status**: Only `sync_threads()` for universal/warp paths. TCGEN05 path has `NamedBarrier` and `PipelineUmmaAsync` with producer/consumer acquire/release, but this is limited to the MMA accumulator pipeline, not general-purpose.

### 3.7 Concurrent/Back-to-Back MMAs

Attention requires two MMAs with softmax between them: S = Q@K^T, then O = softmax(S) @ V. SOTA kernels express this as:

```python
# Simplified pattern from CUTLASS FMHA
for k_tile in range(num_k_tiles):
    # GEMM-I: S = Q @ K^T
    cute.gemm(tiled_mma, s_frag, q_frag, k_frag, s_frag)

    # Online softmax on S (in registers, warp shuffle reductions)
    online_softmax(s_frag, row_max, row_sum, o_frag)

    # GEMM-II: O += P @ V (P = softmax result, in accumulators/SMEM)
    cute.gemm(tiled_mma, o_frag, p_frag, v_frag, o_frag)
```

FA4 further overlaps these: while one tile does GEMM-II, another tile computes softmax (ping-pong schedule using TMEM for both tiles' state).

**Helion CuTe status**: `_mma_loop_is_exclusive()` in `cute_mma.py` requires the loop body to contain *only* the MMA dataflow. Returns `False` if any other `call_function` nodes are present in the loop. This structurally prevents expressing attention's Q@K → softmax → P@V pattern.

---

## 4. Helion CuTe Backend: Current Capabilities

### 4.1 What Works Well

| Capability | Status | Evidence |
|-----------|--------|---------|
| Universal MMA (scalar-per-thread) | **Working** | All tile sizes, F32 accumulation |
| Warp MMA (16x8x16 F16→F32) | **Working** | MmaF16BF16Op atom |
| TCGEN05 MMA (128x8x16 Blackwell) | **Working** | TMEM allocation, async pipeline |
| Simple epilogues | **Working** | Bias add, dtype cast, activation |
| Layout propagation | **Working** | 4-phase seed/fwd/bwd/insert pipeline |
| SMEM layout change (shuffle) | **Working** | For reshape/permute/layout mismatch |
| K-loop reduction | **Working** | Accumulator carried across iterations |
| Pointwise ops via Inductor | **Working** | CuteDSLOpOverrides for 40+ operations |
| 132 dedicated tests pass | **Working** | test_cute_backend + test_cute_lowerings + test_cute_layout |

### 4.2 Architecture: How Memory Hierarchy is Currently Handled

Helion's CuTe backend uses a **layout-driven** approach:

1. **Layout Planning** (`layout_propagation.py`): 4-phase pass decides ThreadLayout per FX graph node
2. **Layout Change Insertion**: Where producer/consumer layouts disagree, insert `_cute_layout_change` (SMEM round-trip)
3. **MMA Pipeline** (`cute_mma.py`): Three code sections:
   - `outer_prefix`: Allocate SMEM for A, B, C; create MMA atoms and accumulators
   - `loop_body`: Global→SMEM (scalar loads) → SMEM→Register (partition+cast) → MMA compute
   - `outer_suffix`: Fragment→SMEM→per-thread-scalar for epilogue

The model is fundamentally **synchronous** and **homogeneous** (all threads do the same thing).

---

## 5. Feature-by-Feature Gap Matrix

| Feature | SOTA (FA4/CUTLASS FMHA) | Helion CuTe | Gap |
|---------|------------------------|-------------|-----|
| Software pipelining | 2-5 stage async overlapping | None (univ/warp); 2-stage partial (tcgen05) | **Critical** |
| TMA bulk copy | All loads, multicast, descriptors | Scalar indexing with if/else masking | **Critical** |
| Warpgroup specialization | 4+ distinct roles, dynamic reg realloc | All threads same role; probed not used | **Critical** |
| Concurrent MMAs | Q@K → softmax → P@V fused | Exclusive loop body constraint | **Critical** |
| Swizzled SMEM | SW128/SW64/SW32 on all operands | Row-major stride only | **High** |
| Online softmax | Running max/sum, exp2, conditional rescale | Not in CuTe pipeline | **High** |
| Named barriers | 15+ barriers for fine-grained sync | sync_threads only (tcgen05 partial) | **High** |
| Multi-stage buffering | Triple/quad buffer with phase bits | Single-stage (univ/warp); 2-stage (tcgen05) | **High** |
| Register pressure control | setmaxnreg per warpgroup | No control; tile-size determined | **Medium** |
| Epilogue fusion (hardware) | TMA store, EVT visitor pattern | Scalar-only after SMEM readback | **Medium** |
| Cluster-level operations | Multicast, cluster barriers | Not supported | **Medium** |
| TMEM (Blackwell) | Accumulator + intermediates | Supported in tcgen05 path | **OK** |
| Basic MMA atoms | WGMMA, TCGEN05 | Universal, Warp, TCGEN05 | **OK** |

---

## 6. The 4 Architectural Blockers

These are structural limitations that prevent Helion from expressing SOTA patterns even if individual features were added incrementally:

### 6.1 Single-MMA-per-Loop Constraint

**Location**: `cute_mma.py:_mma_loop_is_exclusive()` (line ~112)

The function checks that the loop body contains *only* the MMA dataflow. If any other `call_function` nodes are present, it returns `False` and the MMA optimization is rejected.

**Impact**: Attention requires Q@K → softmax → P@V — two MMAs with softmax between them. This cannot be expressed in Helion's current model. Each MMA would need to be a separate kernel, with a global memory round-trip for the intermediate result.

**What's needed**: Support for multi-MMA kernel graphs where the compiler can schedule multiple MMA operations with intermediate pointwise/reduction operations.

### 6.2 Homogeneous Thread Model

**Location**: `layout.py:ThreadLayout` class

All threads are assigned the same role: each thread holds one scalar value and executes the same code. The `ThreadLayout` abstraction encodes thread-to-element mapping but has no concept of thread roles or specialization.

**Impact**: SOTA kernels assign TMA, MMA, softmax, and epilogue to different warps running different code paths simultaneously. Without this, the kernel serializes all operations.

**What's needed**: A mechanism to partition threads into groups with different code paths, coordinated by barriers.

### 6.3 Synchronous Data Movement

**Location**: `cute_mma.py` loop body — scalar loads followed by `sync_threads()` followed by compute

The entire memory hierarchy is synchronous:
```
load A[i] → load B[i] → sync → partition → copy to regs → MMA → sync → repeat
```

**Impact**: The compute units are idle during loads and vice versa. SOTA kernels achieve 70%+ utilization by overlapping these phases.

**What's needed**: Integration with CuTe DSL's `Pipeline*` classes to express async producer/consumer patterns.

### 6.4 Scalar Global Memory Access

**Location**: `cute_mma.py` — element-by-element loads with Python conditionals

```python
smem_a[m_local, _k] = A[m_global, _gk] if m_global < M and _gk < K else 0.0
```

**Impact**: For a 128x64 tile of float16, this generates ~8192 load instructions. TMA does it in 1. The instruction overhead alone can dominate execution time, and the access pattern prevents the memory subsystem from coalescing efficiently.

**What's needed**: TMA descriptor creation and bulk copy primitives, or at minimum vectorized loads (128-bit) with predicated masking.

---

## 7. Normalization Kernel Gap

### 7.1 What SOTA CuTe DSL Normalization Looks Like

**Blackwell RMSNorm** (`blackwell/rmsnorm.py`):
- Cluster-based reduction: 1-16 CTAs cooperate on a single row for large hidden dims
- Async pipelining: weight load hidden behind G2S wait
- 128-bit vectorized loads via `autovec_copy`
- Cross-CTA mbarrier synchronization

**Hopper CTA Norm** (`hopper/cta_norm.py`):
- Both LayerNorm and RMSNorm via compile-time dispatch
- Two-level reduction: warp-level butterfly shuffle + CTA-level via shared memory
- FP32 accumulation regardless of input dtype

### 7.2 Helion's Position

Helion can express simple reductions through its existing reduction support (both Triton and CuTe backends), but **standalone normalization kernels in CuTe** would need:
- Warp-level shuffle reductions (not currently in CuTe backend)
- Cross-CTA cluster reductions (not supported)
- Vectorized loads (not in CuTe backend)
- Two-pass or Welford's algorithm primitives

The more impactful gap is **GEMM + normalization fusion** (CUTLASS example 37): fusing LayerNorm/RMSNorm into GEMM epilogue avoids a separate kernel launch and global memory round-trip. This requires the hardware epilogue fusion (EVT visitor pattern) that Helion's CuTe backend doesn't support.

---

## 8. Roadmap: What Would Be Needed to Close the Gap

Ordered by estimated impact and dependency:

### Phase 1: Foundation (enables basic performance parity for GEMM)

1. **TMA integration for global→SMEM loads**
   - Replace scalar loads with `cute.nvgpu.make_tiled_tma_atom_A/B`
   - Add TMA descriptor management and `prefetch_descriptor()`
   - Estimated impact: 5-20x improvement in load efficiency

2. **Swizzled SMEM layouts**
   - Use `make_smem_layout_a/b()` helpers from cutlass.utils
   - Pass `swizzle=` parameter to `smem.allocate_tensor()`
   - Estimated impact: 20-40% improvement from bank conflict elimination

3. **Multi-stage software pipelining**
   - Integrate `cutlass.pipeline.PipelineTma*` classes
   - Express producer/consumer acquire/wait/release pattern
   - Estimated impact: 30-60% improvement from compute/memory overlap

### Phase 2: Expressiveness (enables attention kernels)

4. **Relax single-MMA-per-loop constraint**
   - Support multiple MMA operations per kernel
   - Allow intermediate operations (softmax) between MMAs
   - This is the prerequisite for any attention kernel

5. **Online softmax primitive**
   - Running max/sum across K tiles with warp shuffle reductions
   - exp2 with log2(e) scaling for FMA fusion
   - Rescale accumulator on max update

6. **Warpgroup specialization**
   - Thread group partitioning with distinct code paths
   - Named barrier coordination between groups
   - Dynamic register reallocation (`setmaxnreg`)

### Phase 3: Peak Performance (for competitive performance)

7. **Register pressure management**
   - Per-warpgroup register budgets
   - Tile size selection informed by register pressure estimates

8. **Hardware epilogue fusion**
   - EVT (Epilogue Visitor Tree) pattern integration
   - TMA store for output writeback
   - GEMM + normalization fusion

9. **Cluster-level operations**
   - TMA multicast across cluster
   - Cross-CTA barriers and reductions
   - Cooperative kernel launch

---

## 9. Alternative Approach: Direct CuTe DSL Kernel Authoring

Rather than extending Helion's compiler to generate all these patterns, an alternative approach is to allow users to write performance-critical kernels directly in CuTe DSL Python (as FlashAttention-4 does) and integrate them into Helion as external templates.

This is analogous to how PyTorch uses:
- `torch._C._nn.scaled_dot_product_attention` → dispatches to cuDNN/FlashAttention/memory-efficient backends
- CUTLASS templates for specific GEMM configurations

Helion could:
1. Detect attention/normalization patterns at the graph level
2. Dispatch to pre-written CuTe DSL kernel templates (like FA4)
3. Use Helion's CuTe backend only for "glue" pointwise/simple-reduction operations
4. Gradually expand the backend's capabilities to subsume more patterns

This hybrid approach would provide SOTA performance immediately while the compiler catches up.

---

## 10. Conclusion

Helion's CuTe backend is a solid foundation for basic GEMM with simple epilogues, but it is **fundamentally insufficient** for generating SOTA attention or normalization kernels. The gap is not a collection of missing features that can be added incrementally — it includes 4 structural blockers (single-MMA, homogeneous threads, synchronous data movement, scalar loads) that require architectural changes to the compiler.

The CUTLASS ecosystem now provides complete CuTe DSL Python implementations of attention (Ampere/Hopper/Blackwell), normalization, and MLA kernels. FlashAttention-4 demonstrates that CuTe DSL Python can achieve 71% GPU utilization and beat cuDNN by 1.3x. The patterns and APIs needed are well-established — the question is how much of this Helion should generate automatically vs. integrate as pre-built kernel templates.

**Recommended next step**: Evaluate the hybrid approach — integrate FA4/CUTLASS FMHA as a dispatch target for attention patterns while pursuing Phase 1 improvements (TMA, swizzle, pipelining) to close the GEMM performance gap.

---

## 11. Implementation Analysis: Code-Level Deep Dive

This section provides a code-level analysis of the Helion CuTe backend based on a thorough reading of every relevant source file. It maps the gap analysis features to concrete implementation strategies, identifies hidden dependencies between features, and proposes specific code changes.

### 11.1 Codebase Architecture Summary

The CuTe backend's code generation pipeline flows through these key files:

```
helion/_compiler/
├── backend.py (CuteBackend class, lines 1992-2513)
│   ├── pre_codegen() → calls plan_layouts()
│   ├── HelionCuteDSLOpOverrides → wraps PyTorch Inductor's CuteDSLOpOverrides
│   ├── launcher_keyword_args() → thread block dimensions
│   └── MMA detection helpers (_detect_mma_loop, etc.)
│
├── cute/
│   ├── cute_mma.py (1690 lines) — THE critical file
│   │   ├── _mma_loop_is_exclusive() (lines 112-124) — single-MMA gate
│   │   ├── _emit_mma_pipeline() (lines 515-1343) — core codegen
│   │   │   ├── outer_prefix: MMA atom + SMEM alloc + acc init
│   │   │   ├── loop_body: G→S loads → sync → S→R → gemm
│   │   │   └── outer_suffix: fragment → SMEM → scalar readback
│   │   ├── _choose_mma_impl() (lines 1394-1427) — universal/warp/tcgen05
│   │   └── _Tcgen05LayoutPlan (lines 56-82) — Blackwell-specific state
│   │
│   ├── layout.py (182 lines)
│   │   ├── ThreadLayout dataclass — thread-to-element mapping
│   │   ├── LayoutTag enum — COALESCED/REDUCTION/MMA_*/INHERITED/IDENTITY
│   │   └── LayoutConstraint — preferred/layout/required triple
│   │
│   ├── layout_propagation.py — 5-phase pipeline
│   │   ├── _seed_constraints() — loads=COALESCED, reduces=REDUCTION
│   │   ├── _forward_propagate() — pointwise inherits from inputs
│   │   ├── _backward_propagate() — producers adopt consumer layout
│   │   ├── _resolve_layouts() — preferred → layout
│   │   └── _insert_layout_changes() — SMEM round-trip nodes
│   │
│   ├── layout_change.py — SMEM write(src_stride) → sync → read(dst_stride)
│   ├── mma_support.py — hardware probing (warp, warpgroup, tcgen05)
│   └── thread_budget.py — 1024 thread limit enforcement
```

### 11.2 Revised Feature Dependency Graph

After studying the code, the dependencies between features are tighter than originally described:

```
                     ┌──────────────────────────────────────────────────┐
                     │ Swizzled SMEM must pair with TMA, not separate  │
                     └───────────────────┬──────────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │ TMA + Swizzled SMEM (P0) │
                            └────────────┬────────────┘
                                         │
                                         ▼
                          ┌───────────────────────────────┐
                          │ Multi-Stage Pipelining (P0)   │
                          └──────────────┬────────────────┘
                                         │
                                         ▼
                          ┌───────────────────────────────┐
                          │ Warpgroup Specialization (P3) │
                          └───────────────────────────────┘

  Independent track (can start in parallel):

  ┌─────────────────────────────┐     ┌──────────────────────────┐
  │ Relax Single-MMA (P1)      │     │ Warp Shuffle Max/Sum (P1)│
  │  Phase A: post-MMA ops     │     │ (extends reduction_expr) │
  │  Phase B: multi-MMA chains │     └────────────┬─────────────┘
  └──────────────┬──────────────┘                  │
                 │                                 │
                 └──────────┬──────────────────────┘
                            │
                            ▼
               ┌──────────────────────────────┐
               │ Attention Pipeline (P2)      │
               │ _emit_attention_pipeline()   │
               │ + Online Softmax             │
               └──────────────────────────────┘
```

**Critical insight: Swizzled SMEM and TMA must be implemented together.** The current scalar load pattern (`smem_a[m_local, _k] = A[m_global, _gk]`) assumes row-major SMEM indexing. Swizzle remaps coordinates, which would break these explicit scalar writes. TMA handles swizzled addressing internally. Therefore: add swizzle only on paths that use TMA; keep non-swizzled SMEM for scalar-load paths (universal MMA on SM80).

### 11.3 Feature Implementation: TMA + Swizzled SMEM

**What changes**: `cute_mma.py` lines 800-1006 (SMEM allocation and load codegen)

**Key realization**: In CuTe DSL, TMA atoms are created *inside the kernel* using the global tensor reference. The CuTe DSL compiler handles host-side descriptor creation automatically when it sees this pattern. This means **no changes to Helion's kernel argument passing** — the change is entirely within codegen.

The TCGEN05 path already demonstrates the pattern — it uses `_tcgen05_smem_layout_expr()` (line ~1530) which calls `cutlass.utils.blackwell_helpers.make_smem_layout_a()` for swizzled SMEM, and `cute.nvgpu.tcgen05.make_umma_smem_desc()` (line 1068) for descriptor-based MMA. We can follow this pattern for SM90.

#### 11.3.1 TMA Capability Detection

Add to `mma_support.py` or as a helper in `cute_mma.py`:

```python
def _can_use_tma(mma_impl: str, capability: tuple[int, int] | None) -> bool:
    """TMA requires SM90+ (Hopper or newer)."""
    if capability is None:
        return False
    return capability[0] >= 9 and mma_impl in ("warp", "tcgen05")
```

The SM version is available via `CuteMmaSupport.capability` (already probed in `mma_support.py` lines 108-145).

#### 11.3.2 Swizzled SMEM Allocation

**Current code** (`cute_mma.py` lines 841-862, universal/warp path):
```python
prefix.append(f"{smem_a_ptr} = cute.arch.alloc_smem({input_dtype_str}, {bm * bk})")
prefix.append(f"{smem_a} = cute.make_tensor({smem_a_ptr}, "
              f"cute.make_layout(({bm}, {bk}), stride=({bk}, 1)))")
```

**Proposed change** (for `use_tma=True`):
```python
# Compute swizzled layout using CUTLASS utilities
smem_a_layout_var = df.new_var("smem_a_layout")
prefix.append(f"{smem_a_layout_var} = cutlass.utils.hopper.make_smem_layout_a("
              f"{tiled_mma}, ({bm}, {bk}), {input_dtype_str}, {num_stages})")
# Allocate with alignment and swizzle
total_smem_a = bm * bk * num_stages
prefix.append(f"{smem_a_ptr} = cute.arch.alloc_smem("
              f"{input_dtype_str}, {total_smem_a}, alignment=128)")
prefix.append(f"{smem_a} = cute.make_tensor({smem_a_ptr}, {smem_a_layout_var})")
```

The TCGEN05 path already does this at lines 813-839 — the SM90 version follows the same pattern with `cutlass.utils.hopper.*` instead of `cutlass.utils.blackwell_helpers.*`.

#### 11.3.3 TMA Atom Creation

Add to `outer_prefix` (new code, ~30 lines):
```python
# Create TMA load atoms for A and B operands
tma_atom_a = df.new_var("tma_atom_a")
tma_atom_b = df.new_var("tma_atom_b")
prefix.append(f"{tma_atom_a} = cute.make_tma_copy("
              f"cute.nvgpu.TmaLoad(), {lhs_arg_name}, "
              f"{smem_a}.layout, ({bm}, {bk}), {tiled_mma})")
prefix.append(f"{tma_atom_b} = cute.make_tma_copy("
              f"cute.nvgpu.TmaLoad(), {rhs_arg_name}, "
              f"{smem_b}.layout, ({bn}, {bk}), {tiled_mma})")
# Prefetch descriptors to hide latency
prefix.append(f"{tma_atom_a}.prefetch_descriptor()")
prefix.append(f"{tma_atom_b}.prefetch_descriptor()")
```

#### 11.3.4 TMA Load Codegen (Replacing Scalar Loads)

**Current scalar load pattern** (`cute_mma.py` lines 904-928, ~25 lines per operand):
```python
# Universal path: each thread loads elements with bounds checking
if n_local == 0:
    for _k in range(bk):
        _gk = k_offset + _k
        smem_a[m_local, _k] = A[m_global, _gk] if m_global < M and _gk < K else 0.0
```

**Proposed TMA replacement** (~10 lines total):
```python
if use_tma:
    # Only one elected thread issues the TMA load (hardware handles the rest)
    cg.add_statement(f"if cute.arch.elect_one():\n"
                     f"    cute.copy({tma_atom_a}, {tma_global_a}[k_stage], "
                     f"{tma_smem_a}[k_stage], tma_bar_ptr={barrier})\n"
                     f"    cute.copy({tma_atom_b}, {tma_global_b}[k_stage], "
                     f"{tma_smem_b}[k_stage], tma_bar_ptr={barrier})")
else:
    # Keep existing scalar load path for SM80 / universal MMA
    ... # existing code unchanged
```

**Key benefits:**
- TMA handles out-of-bounds automatically (fills with zeros), eliminating `if m_global < M` masking
- One instruction per tile vs ~8192 scalar instructions for a 128x64 f16 tile
- Proper coalescing and cache management

#### 11.3.5 Barrier Replacement

**Current** (`cute_mma.py` line 1006):
```python
cute.arch.sync_threads()  # global barrier — all threads wait
```

**With TMA + pipeline**:
```python
consumer_handle = pipeline.consumer_wait_and_advance(consumer_state)
# ... MMA compute ...
consumer_handle.release()
```

The `sync_threads()` is replaced by the pipeline's acquire/wait/release protocol, which only blocks when data isn't ready (enabling overlap with the next stage's load).

#### 11.3.6 Scope and Fallback Strategy

- **SM90+ with warp/tcgen05 MMA**: Use TMA + swizzled SMEM + pipeline barriers
- **SM80 or universal MMA**: Keep existing scalar load path with non-swizzled SMEM
- Gate with `if use_tma:` / `else:` in the codegen, similar to the existing `if mma_impl == "universal":` / `else:` branches

**Estimated scope**: ~200 lines new/modified in `cute_mma.py`, ~30 lines in a new `tma_support.py` helper module.

### 11.4 Feature Implementation: Multi-Stage Software Pipelining

**Depends on**: TMA + Swizzled SMEM (Section 11.3)

**What changes**: Loop structure in `cute_mma.py` lines 863-1099

#### 11.4.1 The Loop Structure Challenge

Helion's current loop model: `outer_prefix → [loop_body] → outer_suffix`

Pipelining requires a 4-part structure: `prefix → prologue → [main_loop] → drain → suffix`

Helion provides `device_loop.outer_prefix` and `device_loop.outer_suffix` as AST statement lists. The loop body is emitted via `cg.add_statement()`. This gives us a natural mapping:

```python
# outer_prefix: MMA setup + SMEM alloc + pipeline creation + PROLOGUE
prefix = device_loop.outer_prefix
prefix.append(...)  # MMA atom, SMEM, pipeline setup

# Prologue: pre-fill (N-1) stages before main loop starts
for stage in range(num_stages - 1):
    prefix.append(f"p_handle = producer.acquire_and_advance(producer_state)")
    prefix.append(f"tma_load(A, B, stage={stage}, barrier=p_handle.barrier)")
    prefix.append(f"p_handle.commit()")

# Main loop body: overlap load(next_stage) with compute(current_stage)
cg.add_statement(f"c_handle = consumer.wait_and_advance(consumer_state)")
cg.add_statement(f"cute.gemm(..., smem_a[current_stage], ...)")
cg.add_statement(f"c_handle.release()")
# Issue next load (guarded by remaining iterations)
cg.add_statement(f"if k_offset + {bk} < {k_total}:\n"
                 f"    p_handle = producer.acquire_and_advance(producer_state)\n"
                 f"    tma_load(A, B, next_stage, p_handle.barrier)\n"
                 f"    p_handle.commit()")

# outer_suffix: drain + fragment readback (unchanged from current)
suffix = device_loop.outer_suffix
```

#### 11.4.2 Pipeline Type Selection

```python
# SM90 (Hopper): PipelineTmaAsync
pipeline, pipeline_state = cutlass.pipeline.PipelineTmaAsync.create(
    num_stages=num_stages,
    producer_group=cute.arch.make_thread_cooperative_group([tma_warp_id]),
    consumer_group=cute.arch.make_thread_cooperative_group(compute_warp_ids),
    barrier_storage=mbar_ptr,
    tx_count=bytes_per_stage,
)

# SM100 (Blackwell): PipelineTmaUmma
# The TCGEN05 path already uses PipelineUmmaAsync (lines 753-757)
# — extend to PipelineTmaUmma for full TMA integration
```

#### 11.4.3 Stage Count as Autotunable Parameter

The TCGEN05 path hardcodes `ab_stage_count = 2` (line 933). This should become configurable:

```python
# In config spec:
num_stages: int = 2  # default, autotunable range [2, 5]

# SMEM budget constraint:
# For bm=128, bk=64, f16: 1 stage = 16KB
# SM90 SMEM limit: 228KB → max ~14 stages (but 3-5 is optimal)
# SM100 SMEM limit: 232KB → similar
max_stages = min(5, total_smem_budget // stage_smem_size)
```

#### 11.4.4 SMEM Multi-Buffering

Current (single buffer):
```python
smem_a = alloc_smem(dtype, bm * bk)
```

Multi-stage:
```python
smem_a = alloc_smem(dtype, bm * bk * num_stages)  # N copies
# Stage indexing:
smem_a_stage = smem_a[(None, 0, 0, stage_idx)]    # select active stage
```

The TCGEN05 path already does 4D stage indexing at lines 946-964:
```python
mma_stage = (k_offset // bk) % ab_stage_count
smem_a_mma = smem_a[(None, 0, 0, mma_stage)]
```

**Estimated scope**: ~150 lines new/modified in `cute_mma.py`, ~20 lines to make stage count configurable in `_new_tcgen05_layout_plan()`.

### 11.5 Lessons from Triton: Layout-Driven Generic Multi-MMA

A deep study of OpenAI Triton's upstream codebase (`triton-lang/triton`) reveals that Triton solves the multi-MMA problem **generically through layout propagation**, not through pattern detection. This section describes Triton's approach and how it informs Helion's implementation strategy.

#### 11.5.1 Triton's Core Architecture: LinearLayout Abstraction

Triton represents every thread-to-element mapping (MMA fragments, blocked tensors, shared memory) as a `LinearLayout` — a linear transformation from thread indices to tensor element positions.

**Key files:**
- `include/triton/Tools/LinearLayout.h` — core abstraction
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` — layout conversion algebra
- `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td` — layout encoding definitions

Layout types in the hierarchy:
```
DistributedEncodingTrait
├── BlockedEncodingAttr       (standard tiled distribution)
├── LinearEncodingAttr        (LinearLayout-based)
├── NvidiaMmaEncodingAttr     (MMA fragment layout)
├── DotOperandEncodingAttr    (MMA input operand layout)
├── SliceEncodingAttr         (dimension removal)
└── ...

SharedEncodingTrait
├── SwizzledSharedEncodingAttr (bank conflict avoidance)
├── NVMMASharedEncodingAttr    (Hopper+ SMEM for WGMMA)
└── ...
```

#### 11.5.2 How Triton Keeps Operations in MMA Register Layout

The critical insight: **pointwise operations on MMA outputs happen directly in MMA register layout with NO layout conversion.**

From `RemoveLayoutConversions.cpp` (lines 269-333):
```cpp
if (user->hasTrait<OpTrait::Elementwise>()) {
    setEncoding(user->getResults(), info, changed, user);
    // Elementwise ops INHERIT the MMA layout from their input
}
```

When Triton compiles attention:
```python
qk = tl.dot(q, k)            # Output: MMA register layout
qk = qk * qk_scale           # STILL in MMA register layout (direct register multiply)
m_ij = tl.max(qk, 1)         # Reduction IN MMA layout (warp shuffle)
p = tl.math.exp2(qk - m_ij)  # STILL in MMA register layout
acc = tl.dot(p, v, acc)       # P consumed by second dot
```

**Result:** 0-1 SMEM roundtrips between MMAs, vs the 2 roundtrips Helion's scalar model would require.

#### 11.5.3 How Triton Handles Reductions on MMA Fragments

For `tl.max(qk, axis=1)` where `qk` has MMA layout (`ReduceOpToLLVM.cpp`):

1. **Within-thread:** Each thread reduces its own fragment elements that belong to the same row. The `LinearLayout` tells the compiler which fragment elements map to which row.
2. **Within-warp:** Warp shuffles combine values across threads that share the same row.
3. **Cross-warp:** Only if reduction spans warps (uses shared memory + barrier).

The `LinearLayout` abstraction makes this generic — the compiler doesn't need hardcoded MMA atom knowledge because the layout encodes the thread-to-element mapping.

#### 11.5.4 How Triton Converts Between Dots

When one dot's output feeds another dot's input (`ConvertLayoutOpToLLVM.cpp` lines 38-84), Triton picks the cheapest conversion:

| Conversion Type | Cost | When Used |
|-----------------|------|-----------|
| Register reorder | Free | Layouts differ only in register dimension |
| Warp shuffle | Fast | Layouts differ in lane dimension |
| SMEM roundtrip | Expensive | Layouts differ in warp/block dimension |

The `minimalCvtLayout()` function computes the minimal conversion by comparing source and destination `LinearLayout` representations.

#### 11.5.5 Multi-Dot Detection and Optimization

`AccelerateMatmul.cpp` (lines 85-147) detects chained dots and adjusts warp allocation:
```cpp
bool hasChainedDot = false;
for (Operation *op : slices) {
    if (isa<DotOp>(op) && (op != dotOp)) {
        hasChainedDot = true;
    }
}
if (hasChainedDot) {
    // Bias warp allocation for efficient within-warp reductions between dots
    if (shape[0] >= shape[1]) return {numWarps, 1};
    else return {1, numWarps};
}
```

This is NOT pattern detection for attention specifically — it optimizes ANY chained dot pattern.

#### 11.5.6 Online Softmax Falls Out Naturally

From `06-fused-attention.py` (lines 48-110), the online softmax is NOT special-cased. It works because:
1. Elementwise ops on MMA accumulators are direct register operations
2. Reductions use warp shuffles respecting MMA layout
3. Loop-carried values (`m_i`, `l_i`, `acc`) maintain MMA layout across iterations
4. The third operand `c` in `tl.dot(a, b, c)` enables direct accumulation

### 11.6 Why CuTe Layouts, Not Triton's LinearLayout

Triton uses `LinearLayout` — linear functions over GF(2) (the binary Galois field) — to represent thread-to-element mappings. This has a fundamental limitation: **all dimensions must be powers of 2.** Triton's own source code acknowledges this (`LinearLayout.h` lines 281-313):

> "CuTe layouts support non-power-of-two shapes; LLs do not. In particular this means that LLs cannot represent padded layouts."

The code enforces this with `assert(llvm::isPowerOf2_32(size))` throughout `LinearLayout.cpp` and `LinearLayoutConversions.cpp`.

#### 11.6.1 The Z2 Limitation Is Fundamental, Not Fixable

The PO2 constraint is mathematically necessary for GF(2):
- Bases are defined at power-of-2 indices: L(1), L(2), L(4), L(8), ...
- Any index x decomposes as x = b₀·1 ⊕ b₁·2 ⊕ b₂·4 ⊕ ... (binary representation)
- L(x) = b₀·L(1) ⊕ b₁·L(2) ⊕ b₂·L(4) ⊕ ... (linearity over GF(2))
- This only covers all indices in [0, N) when N is a power of 2

**Real-world impact** (Vijay's point): Triton cannot describe programs that map to GB300 NVFP4 tensor cores, which use non-PO2 shapes. This restricts both researchers and hardware designers from parametrizing freely.

#### 11.6.2 CuTe Layout: Shape + Stride Algebra (No PO2 Restriction)

CuTe layouts are `(Shape, Stride)` tuples with **hierarchical nesting**:

```
Layout = (Shape, Stride)
  Shape  = int | (Shape, Shape, ...)     # arbitrary integers, any nesting depth
  Stride = int | (Stride, Stride, ...)   # matching structure
```

The mapping is standard linear algebra: `index(coord) = Σ(coord_i × stride_i)`. Any integer shape works — 3, 5, 7, 127, whatever the hardware needs.

**CuTe's layout algebra** (all available in the CuTe DSL Python API):

| Operation | API | Purpose |
|-----------|-----|---------|
| `composition(A, B)` | Compose two layouts | Layout A applied after B |
| `complement(L, N)` | Layout complement | Find the "rest" of a tiling |
| `logical_divide(L, T)` | Tile a layout | Split into tile + remainder |
| `logical_product(L, T)` | Blocked tiling | Distribute tiles across threads |
| `left_inverse(L)` | Left inverse | Undo a layout mapping |
| `right_inverse(L)` | Right inverse | Find pre-image of a mapping |
| `make_layout_tv(thr, val)` | Thread-Value decomposition | Map threads to tile elements |
| `coalesce(L)` | Simplify | Merge contiguous dimensions |
| `Swizzle(B, M, S)` | Bank conflict avoidance | XOR-based address remapping |

**Key advantage over Triton**: These operations work on arbitrary integer shapes. A `(3, 5)` tile, a `(7, 13)` reduction, or a GB300 NVFP4 shape are all first-class citizens.

#### 11.6.3 Helion's Advantage: CuTe Layout Is Already the Target

Helion generates CuTe DSL Python code. This means:

1. **No need to build a layout algebra** — CuTe's `composition`, `complement`, `left_inverse` etc. are code-generation targets. Helion emits calls to them.

2. **MMA fragment layout is already a CuTe layout** — `thr_mma.partition_C(tensor)` returns a CuTe tensor whose layout encodes which thread holds which elements. No reverse-engineering needed.

3. **Layout conversion is a CuTe operation** — `composition(src_layout, right_inverse(dst_layout))` gives the permutation between any two layouts. CuTe computes this at kernel compile time.

4. **Swizzle is native** — `ComposedLayout` pairs a logical layout with a `Swizzle` function. Already used in the TCGEN05 path via `cute.recast_ptr(ptr, layout.inner)`.

**Comparison to Triton's approach**:

| Aspect | Triton (Z2 LinearLayout) | Helion + CuTe Layout |
|--------|-------------------------|---------------------|
| Shape restriction | **PO2 only** | **Any integer** |
| GB300 NVFP4 support | **Cannot represent** | **Native** |
| Layout analysis timing | In compiler (Python/C++) | In generated code (CuTe DSL) |
| Automatic conversion | Compiler inserts ConvertLayoutOp | Codegen emits `composition`/`right_inverse` |
| Swizzle | Built into Z2 layout | Separate `Swizzle` + `ComposedLayout` |
| Optimization search | Compiler can search for bank-conflict-free layouts | Relies on CUTLASS utilities (`make_smem_layout_a/b`) |
| Compiler complexity | High (must understand all layouts) | Lower (delegates to CuTe runtime) |

The one thing Triton's approach does better is **automatic search** for bank-conflict-free shared memory layouts. But CUTLASS provides utility functions (`make_smem_layout_a/b`, swizzle pattern selection) that solve this for known MMA atoms. Helion already uses these in the TCGEN05 path.

### 11.7 Generic Multi-MMA via Fragment Mode with CuTe Layouts

Combining the lessons from Triton (keep values in fragment layout) with CuTe's layout algebra (no PO2 restriction, richer operations), Helion should implement **fragment-level codegen that delegates layout questions to CuTe DSL**.

#### 11.7.1 Current vs. Proposed Data Flow

**Current Helion (scalar model):**
```
MMA output (fragments)
  → SMEM (partition_C) → scalar (smem[m_local, n_local])    # SMEM roundtrip 1
  → pointwise ops (scalar, via ThreadLayout)
  → SMEM → fragments                                         # SMEM roundtrip 2
  → next MMA
```

**Proposed Helion (fragment mode + CuTe layout):**
```
MMA output (fragments in CuTe layout)
  → pointwise ops (element-wise loop over fragment)           # NO roundtrip
  → reduction (CuTe layout tells us row membership → warp shuffle)  # NO roundtrip
  → SMEM (one partition_C write for next MMA input)           # 1 roundtrip
  → next MMA (reads via partition_A/B)
```

#### 11.7.2 Step 1: Lift the Exclusivity Constraint (~50 lines)

Replace `_mma_loop_is_exclusive()` (lines 112-124) with operation classification:

```python
def _classify_loop_ops(mma_node: Node) -> dict[str, set[Node]]:
    """Classify loop operations relative to the MMA node."""
    mma_deps = _collect_node_dependencies(mma_node)
    downstream = _collect_downstream_users(mma_node)
    
    fragment_compatible = set()
    for node in downstream:
        if node.op == "call_function" and _is_fragment_op(node.target):
            fragment_compatible.add(node)
    
    return {
        "mma_deps": mma_deps,
        "fragment_ops": fragment_compatible,
        "incompatible": ...,
    }

def _is_fragment_op(target) -> bool:
    """Check if operation can work element-wise on MMA fragments."""
    FRAGMENT_OPS = {
        torch.ops.aten.add, torch.ops.aten.sub,
        torch.ops.aten.mul, torch.ops.aten.div,
        torch.ops.aten.neg, torch.ops.aten.exp2,
        torch.ops.aten.maximum, torch.ops.aten.minimum,
        # ... extensible registry
    }
    return target in FRAGMENT_OPS
```

#### 11.7.3 Step 2: Fragment-Level Pointwise Codegen (~100 lines)

CuTe DSL already supports element-wise fragment access. Helion's existing code does this for dtype casts (`cute_mma.py` lines 1028-1036):
```python
for _mma_i in range(cute.size(rA)):
    rA[_mma_i] = acc_dtype(tAsA[_mma_i])
```

Generalize to arbitrary pointwise operations on MMA accumulators:
```python
# Generated CuTe DSL code:
for _i in range(cute.size(acc_frag)):
    acc_frag[_i] = acc_frag[_i] * cutlass.Float32(scale)
```

No layout knowledge needed — element-wise loops work regardless of the fragment's CuTe layout.

#### 11.7.4 Step 3: Fragment-Level Reduction via CuTe Layout Algebra (~150 lines)

For row-wise reductions (needed for softmax's `row_max` and `row_sum`), the Helion compiler needs to know which fragment elements belong to the same row. **Instead of hardcoding MMA atom mappings**, delegate to CuTe's layout algebra in the generated code:

```python
# Generated CuTe DSL code:

# Get the accumulator's thread-value layout from the MMA atom
tv_layout = thr_mma.partition_C(smem_c)  # CuTe layout: thread → (row, col)

# CuTe's layout tells us how many values per thread, and which row each belongs to.
# Use logical_divide to decompose into row-groups:
row_val_layout = cute.logical_divide(tv_layout.layout, (bm, 1))

# Within-thread: reduce fragment elements sharing a row
# (CuTe layout gives us the element indices per row — no PO2 restriction)
for _row in range(_rows_per_thread):
    row_max = acc_frag[_row_start]
    for _col in range(1, _cols_per_thread):
        row_max = max(row_max, acc_frag[_row_start + _col])
    
    # Cross-thread: warp shuffle for threads sharing the same row
    row_max = cute.arch.warp_reduction_max(row_max, threads_in_group=_threads_per_row)
```

**Key point**: The `_rows_per_thread`, `_cols_per_thread`, and `_threads_per_row` values come from the CuTe layout at codegen time. The Helion compiler queries the MMA atom's partition layout (which CuTe computes) rather than hardcoding per-atom mappings.

This works for any MMA atom — warp (16×8), warpgroup (64×N), TCGEN05 (128×8), or future GB300 NVFP4 atoms with non-PO2 shapes.

#### 11.7.5 Step 4: Fragment → SMEM → Next MMA Input Staging (~80 lines)

When fragment output from GEMM-I feeds into GEMM-II, one SMEM roundtrip is needed. CuTe's partition operations handle layout conversion automatically:

```python
# Generated CuTe DSL code:

# Write P fragments to SMEM (accumulator layout → SMEM layout)
tCsP = thr_mma.partition_C(smem_p)
for _i in range(cute.size(tCsP)):
    tCsP[_i] = p_frag[_i]

cute.arch.sync_threads()

# Read from SMEM as next MMA's A operand (SMEM layout → operand layout)
tAsP = thr_mma.partition_A(smem_p)
```

`partition_C` and `partition_A` use CuTe's layout composition internally to compute the correct thread-to-element mapping for writes and reads respectively. The layouts can differ (accumulator layout ≠ operand layout), and CuTe handles the conversion through the SMEM intermediary.

This is one SMEM roundtrip (vs two in the scalar model).

#### 11.7.6 Step 5: Multiple Accumulator Management (~60 lines)

For attention, two accumulators live simultaneously:

```python
# Generated CuTe DSL code:

# In outer_prefix:
s_frag = cute.make_fragment(tiled_mma.partition_shape_C((bm, bn)), acc_dtype)
o_frag = cute.make_fragment(tiled_mma.partition_shape_C((bm, bd)), acc_dtype)

# In loop body:
cute.gemm(tiled_mma, s_frag, sQ, sK, s_frag)    # GEMM-I: Q@K^T
# ... fragment-level softmax on s_frag (steps 2-3 above) ...
# ... write P to SMEM (step 4 above) ...
cute.gemm(tiled_mma, o_frag, sP, sV, o_frag)    # GEMM-II: P@V

# In outer_suffix:
# Only o_frag needs readback (s_frag is consumed within loop)
```

### 11.8 Warpgroup Specialization (P2, Performance Only)

A performance optimization that assigns different warps to different roles. Not required for correctness. The TCGEN05 path already uses `NamedBarrier` and `PipelineUmmaAsync` — generalizing to SM90 with `PipelineTmaAsync` follows the same pattern.

### 11.9 Final Priority Matrix

| Priority | Feature | Complexity | Est. Lines | Dependencies | Benefit |
|----------|---------|-----------|------------|-------------|---------|
| **P0** | TMA + Swizzled SMEM | Medium | ~200 | None | 5-20x load efficiency |
| **P0** | Multi-stage pipelining | Medium-High | ~150 | TMA | 30-60% compute/mem overlap |
| **P1** | Lift MMA exclusivity + classify ops | Low | ~50 | None | Enable fragment mode |
| **P1** | Fragment-level pointwise codegen | Medium | ~100 | P1 gate lift | Direct register ops |
| **P1** | Fragment-level reduction (CuTe layout) | Medium-High | ~150 | P1 gate lift | Enable softmax |
| **P1** | Fragment→SMEM→next MMA staging | Medium | ~80 | P1 above | Multi-MMA support |
| **P1** | Multiple accumulator management | Medium | ~60 | P1 above | Attention kernels |
| **P2** | Warpgroup specialization | Very High | ~600+ | P0 | Peak performance |

### 11.10 Comparison: Helion (CuTe Layout) vs Triton (Z2 LinearLayout)

| Aspect | Triton (Z2 LinearLayout) | Helion (CuTe Layout) |
|--------|-------------------------|---------------------|
| **Shape restriction** | **PO2 only** | **Any integer** |
| **GB300 NVFP4** | **Cannot represent** | **Native support** |
| **Layout representation** | Matrices over GF(2) | (Shape, Stride) tuples |
| **Nesting** | Flat only | Hierarchical nesting |
| **Composition** | Matrix multiply over GF(2) | `cute.composition(A, B)` |
| **Inverse** | Matrix inverse over GF(2) | `cute.left_inverse(L)` / `cute.right_inverse(L)` |
| **Complement** | Not directly available | `cute.complement(L, N)` |
| **Swizzle** | Built into Z2 layout | Separate `Swizzle` via `ComposedLayout` |
| **Fragment layout source** | Compiler-internal NvidiaMmaEncodingAttr | `thr_mma.partition_C()` (runtime CuTe layout) |
| **Pointwise on fragments** | Automatic (compiler layout propagation) | Element-wise loops (no layout knowledge needed) |
| **Reduction on fragments** | LinearLayout-driven warp shuffle | CuTe layout-driven row decomposition |
| **Layout conversion** | Compiler analyzes register/shuffle/SMEM cost | `partition_C` → SMEM → `partition_A` (CuTe handles it) |
| **Pattern detection needed** | No | No |
| **Compiler complexity** | High (must understand all layouts) | Lower (delegates to CuTe) |
| **Optimization search** | Automatic bank-conflict-free search | Uses CUTLASS utilities |

**Bottom line**: Helion should use CuTe's layout algebra directly rather than reimplementing Triton's Z2 system. CuTe layouts are strictly more expressive (no PO2 restriction), already available as the compilation target, and simplify the compiler by delegating layout questions to the generated CuTe DSL code.
