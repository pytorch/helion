# Helion → CuTe DSL Transpiler Architecture

**Date**: 2026-04-13
**Scope**: Generic transpiler design that converts Helion FX graphs to performant CuTe DSL code — no pattern detection, handles arbitrary op chains including attention.

---

## 1. Design Philosophy

Helion is a **transpiler** (FX graph → CuTe DSL Python), not a **compiler backend** (IR → PTX). CuTe DSL + CUTLASS handles register mapping, instruction selection, and layout math. Helion decides **what** CuTe DSL code to emit, not **how** layouts map to hardware.

The entire transpiler reduces to one question: **for each FX node, what domain is its data in, and what CuTe DSL code handles domain transitions?**

---

## 2. The Core Abstraction: Data Domains

Every value in the FX graph lives in one of three domains:

| Domain | Where data lives | Examples |
|---|---|---|
| **GLOBAL** | HBM (device DRAM) | Kernel inputs/outputs |
| **SHARED** | SMEM (per-CTA) | Tiles loaded for MMA, staging buffers |
| **FRAGMENT** | Registers (per-thread, MMA layout) | MMA accumulators, pointwise results on MMA output |

---

## 3. The Generic Pipeline

```
FX Graph → [1. Classify] → [2. Propagate Domains] → [3. Insert Transitions] → [4. Emit CuTe DSL]
```

### Stage 1: Node Classification

Every FX node gets a **node kind** from a fixed, small registry — not pattern detection:

```python
NODE_KIND_MAP = {
    # MMA operations
    torch.ops.aten.mm: NodeKind.MMA,
    torch.ops.aten.bmm: NodeKind.MMA,
    torch.ops.aten.matmul: NodeKind.MMA,
    torch.ops.aten.addmm: NodeKind.MMA,

    # Pointwise operations (element-wise on fragments)
    torch.ops.aten.add: NodeKind.POINTWISE,
    torch.ops.aten.sub: NodeKind.POINTWISE,
    torch.ops.aten.mul: NodeKind.POINTWISE,
    torch.ops.aten.div: NodeKind.POINTWISE,
    torch.ops.aten.neg: NodeKind.POINTWISE,
    torch.ops.aten.exp2: NodeKind.POINTWISE,
    torch.ops.aten.log2: NodeKind.POINTWISE,
    torch.ops.aten.maximum: NodeKind.POINTWISE,
    torch.ops.aten.minimum: NodeKind.POINTWISE,
    torch.ops.aten.where: NodeKind.POINTWISE,

    # Reductions
    torch.ops.aten.sum: NodeKind.REDUCTION,
    torch.ops.aten.max: NodeKind.REDUCTION,
    torch.ops.aten.amax: NodeKind.REDUCTION,

    # Memory operations
    # (loads/stores identified by graph structure, not op target)
}
```

~20 op mappings. Extensible by adding one line per new op.

### Stage 2: Domain Propagation

Walk the graph in **topological order**. Each node's output domain is determined by 6 rules:

```python
def propagate_domain(node, input_domains):
    kind = classify(node)

    # Rule 1: MMA output is always FRAGMENT
    if kind == NodeKind.MMA:
        return Domain.FRAGMENT

    # Rule 2: Pointwise on all-FRAGMENT inputs stays FRAGMENT
    if kind == NodeKind.POINTWISE:
        if all(d == Domain.FRAGMENT for d in input_domains):
            return Domain.FRAGMENT
        # Mixed domains: needs materialization (rare)
        return Domain.SHARED

    # Rule 3: Reduction on FRAGMENT stays FRAGMENT (warp shuffle)
    if kind == NodeKind.REDUCTION:
        if any(d == Domain.FRAGMENT for d in input_domains):
            return Domain.FRAGMENT
        return Domain.SHARED

    # Rule 4: Memory read → SHARED (loaded into SMEM for MMA consumption)
    if kind == NodeKind.MEMORY_READ:
        return Domain.SHARED

    # Rule 5: Memory write → GLOBAL (destination)
    if kind == NodeKind.MEMORY_WRITE:
        return Domain.GLOBAL

    # Rule 6: Constants/scalars → FRAGMENT (broadcast)
    return Domain.FRAGMENT
```

**The key discovery**: When a FRAGMENT-domain value feeds into an MMA node as an **operand** (not accumulator), domain propagation detects a FRAGMENT→SHARED→FRAGMENT transition is needed. This is where SMEM staging gets inserted — **automatically, without pattern matching**.

### Stage 3: Insert Domain Transitions

At every domain boundary, insert the appropriate CuTe DSL transition code:

| Transition | When it happens | CuTe DSL Code |
|---|---|---|
| GLOBAL → SHARED | Load tile from HBM | `cute.copy(tma_atom, gmem_part, smem_part, tma_bar_ptr=bar)` (SM90+) or scalar loop (fallback) |
| SHARED → FRAGMENT (MMA) | MMA reads operands | Implicit: `cute.gemm(mma, acc, partition_A(smem), partition_B(smem), acc)` |
| FRAGMENT → SHARED | Fragment feeds next MMA input | `tCsX = thr_mma.partition_C(smem_buf)` → element-wise write → `sync_threads()` |
| SHARED → FRAGMENT (non-MMA) | Rare: read back as scalars | `tCsX = thr_mma.partition_C(smem_buf)` → element-wise read |
| FRAGMENT → SHARED → GLOBAL | Store results | Write fragments to SMEM via `partition_C`, then TMA store or scalar store |

### Stage 4: CuTe DSL Emitters

Each node kind has a small emitter function. The emitter takes (node, domain context, variable names) and returns CuTe DSL code:

#### 4a. MMA Emitter

```python
def emit_mma(node, acc_var, smem_a, smem_b, mma_var):
    return dedent(f"""
        # Zero accumulator
        {acc_var} = cute.make_fragment(thr_mma.partition_shape_C(tile_mn), Float32)
        for _i in range(cute.size({acc_var})):
            {acc_var}[_i] = Float32(0.0)

        # GEMM: accumulate over K tiles
        cute.gemm({mma_var}, {acc_var},
                  thr_mma.partition_A({smem_a}),
                  thr_mma.partition_B({smem_b}),
                  {acc_var})
    """)
```

#### 4b. Pointwise Emitter (fragment domain)

```python
CUTE_OP_MAP = {
    torch.ops.aten.add: "+",
    torch.ops.aten.mul: "*",
    torch.ops.aten.exp2: "cute.exp2",
    torch.ops.aten.neg: "-",
    # ...
}

def emit_pointwise_fragment(node, result_var, input_vars):
    op = CUTE_OP_MAP[node.target]
    if is_unary(node.target):
        return f"for _i in range(cute.size({result_var})):\n    {result_var}[_i] = {op}({input_vars[0]}[_i])"
    else:
        return f"for _i in range(cute.size({result_var})):\n    {result_var}[_i] = {input_vars[0]}[_i] {op} {input_vars[1]}[_i]"
```

#### 4c. Reduction Emitter (fragment domain, CuTe layout-driven)

```python
def emit_reduction_fragment(node, frag_var, result_var, reduce_op, mma_layout_info):
    """
    Uses CuTe layout to determine fragment row structure.
    mma_layout_info provides: rows_per_thread, cols_per_thread, threads_per_row
    — queried from thr_mma.partition_C() at codegen time.
    """
    return dedent(f"""
        # Within-thread reduction (element-wise over fragment)
        for _row in range({mma_layout_info.rows_per_thread}):
            {result_var}[_row] = {frag_var}[_row * {mma_layout_info.cols_per_thread}]
            for _col in range(1, {mma_layout_info.cols_per_thread}):
                {result_var}[_row] = {reduce_op}({result_var}[_row],
                    {frag_var}[_row * {mma_layout_info.cols_per_thread} + _col])

        # Cross-thread warp shuffle reduction
        for _row in range({mma_layout_info.rows_per_thread}):
            {result_var}[_row] = cute.arch.warp_reduction_{reduce_op}(
                {result_var}[_row], threads_in_group={mma_layout_info.threads_per_row})
    """)
```

#### 4d. Domain Transition Emitter

```python
def emit_fragment_to_shared(frag_var, smem_buf, mma_var):
    """FRAGMENT → SHARED: write MMA fragments to SMEM for next MMA's input."""
    return dedent(f"""
        tCsX = {mma_var}.get_slice(thread_idx).partition_C({smem_buf})
        for _i in range(cute.size(tCsX)):
            tCsX[_i] = {frag_var}[_i]
        cute.arch.sync_threads()
    """)

def emit_shared_to_mma_input(smem_buf, mma_var, operand="A"):
    """SHARED → next MMA input: read from SMEM as operand A or B."""
    partition_fn = f"partition_{operand}"
    return f"t{operand}sX = {mma_var}.get_slice(thread_idx).{partition_fn}({smem_buf})"
```

---

## 4. Pipeline Wrapping (Orthogonal to Domain Propagation)

Pipelining is a **mechanical transformation** on GLOBAL→SHARED transitions. It doesn't need to understand what the compute does.

### Without pipelining (baseline):

```python
for tile_idx in range(num_tiles):
    # GLOBAL → SHARED
    cute.copy(tma_atom, gmem[tile_idx], smem)
    sync()
    # Compute on SHARED
    cute.gemm(mma, acc, partition_A(smem), ...)
```

### With N-stage pipelining:

```python
pipeline = PipelineTmaAsync.create(num_stages=N, ...)

# Prologue: fill first N-1 stages
for stage in range(N - 1):
    pipeline.producer_acquire(stage)
    cute.copy(tma_atom, gmem[stage], smem[stage], tma_bar_ptr=barrier[stage])
    pipeline.producer_commit(stage)

# Main loop: overlap compute with next load
for tile_idx in range(num_tiles):
    stage = tile_idx % N

    # Consumer: compute on current stage
    pipeline.consumer_wait(stage)
    cute.gemm(mma, acc, partition_A(smem[stage]), ...)
    pipeline.consumer_release(stage)

    # Producer: load next tile
    next_tile = tile_idx + N - 1
    if next_tile < num_tiles:
        next_stage = next_tile % N
        pipeline.producer_acquire(next_stage)
        cute.copy(tma_atom, gmem[next_tile], smem[next_stage], ...)
        pipeline.producer_commit(next_stage)

# Drain: consume remaining stages
for stage in range(remaining):
    pipeline.consumer_wait(stage)
    cute.gemm(mma, acc, partition_A(smem[stage]), ...)
    pipeline.consumer_release(stage)
```

This wrapping is applied to **any** GLOBAL→SHARED load, regardless of what the compute phase does.

---

## 5. Worked Example: Attention (No Pattern Detection)

Given Helion FX graph for attention:

```
load_q → SHARED
load_k → SHARED
bmm(q, k) → FRAGMENT (s_frag)              # Rule 1: MMA → FRAGMENT
amax(s_frag) → FRAGMENT (row_max)           # Rule 3: reduction on FRAGMENT
sub(s_frag, row_max) → FRAGMENT             # Rule 2: pointwise on FRAGMENT
exp2(...) → FRAGMENT (p_frag)               # Rule 2: pointwise on FRAGMENT
sum(p_frag) → FRAGMENT (row_sum)            # Rule 3: reduction on FRAGMENT
    ↓
p_frag feeds bmm as operand_A
    → Rule 4 triggers! FRAGMENT→SHARED transition needed
    → Insert: partition_C(smem_p) ← write p_frag; sync; partition_A(smem_p)
    ↓
load_v → SHARED
bmm(p_staged, v) → FRAGMENT (o_frag)       # Rule 1: MMA → FRAGMENT
div(o_frag, row_sum) → FRAGMENT             # Rule 2: pointwise on FRAGMENT
store(o_frag) → GLOBAL                      # Rule 5/6: FRAGMENT→SHARED→GLOBAL
```

The transpiler **discovers** the SMEM staging point between GEMM-I and GEMM-II automatically via Rule 4. No attention pattern detector needed.

### Generated CuTe DSL code (what the transpiler produces):

```python
@cutlass.kernel
def attention_kernel(
    Q, K, V, O,
    tma_atom_q: cute.CopyAtom,
    tma_atom_k: cute.CopyAtom,
    tma_atom_v: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    smem_layout_q, smem_layout_k, smem_layout_v, smem_layout_p,
    bm: int, bn: int, bd: int, num_kv_tiles: int,
):
    # ──── SMEM Allocation ────
    smem_q = cute.alloc_smem(smem_layout_q)      # persistent Q tile
    smem_k = cute.alloc_smem(smem_layout_k)      # streamed K (multi-buffered)
    smem_v = cute.alloc_smem(smem_layout_v)      # streamed V (multi-buffered)
    smem_p = cute.alloc_smem(smem_layout_p)      # staging: GEMM-I out → GEMM-II in

    thr_mma = tiled_mma.get_slice(cute.arch.thread_idx())

    # ──── Accumulator Init (FRAGMENT domain) ────
    o_frag = cute.make_fragment(thr_mma.partition_shape_C((bm, bd)), cutlass.Float32)
    for _i in range(cute.size(o_frag)):
        o_frag[_i] = cutlass.Float32(0.0)

    # Row-max and row-sum: per-thread scalars carried across iterations
    row_max = cutlass.Float32(float("-inf"))
    row_sum = cutlass.Float32(0.0)

    # ──── Load Q (persistent, loaded once) ────
    # GLOBAL → SHARED transition
    cute.copy(tma_atom_q, Q_gmem_partition, smem_q)
    cute.arch.sync_threads()

    # ──── Main KV Loop ────
    for kv_tile in range(num_kv_tiles):

        # ──── GLOBAL → SHARED: Load K[kv_tile] ────
        cute.copy(tma_atom_k, K_gmem_partition[kv_tile], smem_k)
        cute.arch.sync_threads()

        # ──── MMA: S = Q @ K^T (FRAGMENT domain) ────
        s_frag = cute.make_fragment(thr_mma.partition_shape_C((bm, bn)), cutlass.Float32)
        for _i in range(cute.size(s_frag)):
            s_frag[_i] = cutlass.Float32(0.0)
        cute.gemm(tiled_mma, s_frag,
                  thr_mma.partition_A(smem_q),
                  thr_mma.partition_B(smem_k),
                  s_frag)

        # ──── REDUCTION: row_max of s_frag (FRAGMENT domain, Rule 3) ────
        # Within-thread max over fragment elements in same row
        new_row_max = cutlass.Float32(float("-inf"))
        for _row in range(rows_per_thread):
            for _col in range(cols_per_thread):
                new_row_max = max(new_row_max, s_frag[_row * cols_per_thread + _col])
        # Cross-thread warp shuffle
        new_row_max = cute.arch.warp_reduction_max(new_row_max, threads_per_row)

        # ──── POINTWISE: rescale previous O accumulator (FRAGMENT domain, Rule 2) ────
        scale = cute.exp2((row_max - new_row_max) * cutlass.Float32(1.4426950408889634))
        for _i in range(cute.size(o_frag)):
            o_frag[_i] = o_frag[_i] * scale
        row_sum = row_sum * scale

        # ──── POINTWISE: P = exp2((S - new_max) * log2e) (FRAGMENT domain, Rule 2) ────
        for _i in range(cute.size(s_frag)):
            s_frag[_i] = cute.exp2(
                (s_frag[_i] - new_row_max) * cutlass.Float32(1.4426950408889634)
            )

        # ──── REDUCTION: row_sum += sum(P) (FRAGMENT domain, Rule 3) ────
        for _row in range(rows_per_thread):
            row_sum_local = cutlass.Float32(0.0)
            for _col in range(cols_per_thread):
                row_sum_local = row_sum_local + s_frag[_row * cols_per_thread + _col]
            row_sum_local = cute.arch.warp_reduction_sum(row_sum_local, threads_per_row)
            row_sum = row_sum + row_sum_local

        row_max = new_row_max

        # ──── DOMAIN TRANSITION: FRAGMENT → SHARED (Rule 4 triggered!) ────
        # s_frag (GEMM-I output) feeds GEMM-II as operand A
        # Must stage through SMEM to convert accumulator layout → operand layout
        tCsP = thr_mma.partition_C(smem_p)
        for _i in range(cute.size(tCsP)):
            tCsP[_i] = s_frag[_i]
        cute.arch.sync_threads()

        # ──── GLOBAL → SHARED: Load V[kv_tile] ────
        cute.copy(tma_atom_v, V_gmem_partition[kv_tile], smem_v)
        cute.arch.sync_threads()

        # ──── MMA: O += P @ V (FRAGMENT domain) ────
        cute.gemm(tiled_mma, o_frag,
                  thr_mma.partition_A(smem_p),    # P from SMEM staging
                  thr_mma.partition_B(smem_v),
                  o_frag)

    # ──── POINTWISE: O = O / row_sum (FRAGMENT domain, Rule 2) ────
    for _i in range(cute.size(o_frag)):
        o_frag[_i] = o_frag[_i] / row_sum

    # ──── DOMAIN TRANSITION: FRAGMENT → SHARED → GLOBAL (Rule 6) ────
    tCsO = thr_mma.partition_C(smem_o)
    for _i in range(cute.size(tCsO)):
        tCsO[_i] = o_frag[_i]
    cute.arch.sync_threads()
    cute.copy(tma_atom_o, smem_o, O_gmem_partition)
```

Every line above was generated by the 6 domain rules + 4 emitters. No attention-specific logic in the transpiler.

---

## 6. SMEM Budget Planning

The transpiler must allocate SMEM buffers. Budget computation is mechanical:

```python
def compute_smem_budget(tile_sizes, dtypes, num_stages, staging_buffers):
    budget = 0
    # Persistent tiles (loaded once, kept for entire kernel)
    budget += tile_sizes["Q"] * dtype_size(dtypes["Q"])  # e.g., 128×128 × 2B = 32KB
    # Streamed tiles (multi-buffered by pipeline)
    budget += tile_sizes["K"] * dtype_size(dtypes["K"]) * num_stages
    budget += tile_sizes["V"] * dtype_size(dtypes["V"]) * num_stages
    # Staging buffers (FRAGMENT → SHARED transitions)
    for buf in staging_buffers:
        budget += buf.size * dtype_size(buf.dtype)
    return budget
```

For attention with bm=128, bn=64, bd=128, f16, 2 pipeline stages:
- Q: 128×128 × 2B = 32KB (persistent)
- K: 128×64 × 2B × 2 stages = 32KB
- V: 64×128 × 2B × 2 stages = 32KB
- P staging: 128×64 × 4B = 32KB (f32 accumulator)
- **Total: ~128KB** — fits in H100's 228KB SMEM

---

## 7. Component Size Estimates

| Component | Purpose | Est. Lines |
|---|---|---|
| Node classifier | op → NodeKind registry | ~30 |
| Domain propagator | 6 rules, topological walk | ~80 |
| Transition inserter | detect domain boundaries, insert SMEM staging | ~100 |
| MMA emitter | emit `cute.gemm()` + accumulator init | ~40 |
| Pointwise emitter | emit element-wise fragment loops | ~40 |
| Reduction emitter | CuTe layout query + warp shuffle | ~80 |
| Domain transition emitter | `partition_C → SMEM → partition_A` codegen | ~60 |
| Pipeline wrapper | prologue/main/drain template | ~120 |
| SMEM budget planner | allocation sizes from tile shapes + domains | ~50 |
| **Total** | | **~600** |

---

## 8. Why This Design is Correct

1. **No pattern detection** — Domain rules handle arbitrary op chains. Attention, GEMM, MLP, or any future fusion falls out of the same 6 rules.

2. **CuTe handles the hard parts** — Layout math, register mapping, bank-conflict-free swizzling, and instruction selection are all delegated to CuTe DSL and CUTLASS.

3. **Minimal compiler surface** — ~600 lines of Python vs Triton's ~100K lines of C++. Each component is independently testable.

4. **No PO2 restriction** — CuTe layouts work with any MMA atom shape, including future GB300 NVFP4.

5. **Pipelining is orthogonal** — Pipeline wrapping transforms GLOBAL→SHARED loads without understanding the compute graph.

6. **Fragment operations need no layout knowledge** — `for _i in range(cute.size(frag)): frag[_i] = op(frag[_i])` works regardless of underlying CuTe layout.

7. **Reductions use CuTe layout queries** — `rows_per_thread`, `cols_per_thread`, `threads_per_row` come from the MMA atom's partition layout at codegen time, not hardcoded per-atom.
