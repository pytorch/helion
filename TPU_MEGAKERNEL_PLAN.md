# Compiling TPU Megakernels in Helion

Status: design plan v2. It incorporates the compiler-feasibility and
TPU-dataflow reviews. Branch `tpu-megakernel-compiler`, based on `main` at
554daa4f3.

**Scope:** megakernel (cross-kernel) optimizations only. These are the
optimizations that exist *because* several ops share one TPU program: where
data lives between ops, and how DMA is scheduled across op boundaries.
Single-op concerns are out of scope: quantization, MXU shapes, numerics,
batching, attention internals.

## 0. Thesis

On GPU, a megakernel is a scheduling problem. Persistent workers on many SMs
start consumer tiles as soon as tile-level readiness counters say they can.
That is Helion's `cross_loop_pipeline`.

On TPU (v7: one JAX device is one TensorCore) a megakernel is **one sequential
program per core**. It is HBM-bandwidth bound and has about 64 MiB of
software-managed VMEM. The wins are dataflow wins:

1. Intermediates never round-trip HBM. They stay in VMEM across op (root)
   boundaries.
2. **The weight DMA ring never drains.** The next op's or layer's weights are in
   flight while the current op computes, across op, layer and collective
   boundaries.
3. VMEM is a pool whose bytes are reused as values die.

Tile-dependency analysis is the foundation, but it answers different questions
than on GPU. Program order is fixed, so the question is not "when may this tile
run?". The questions are:

* **Which tensors are cross-root intermediates?** These become VMEM-resident.
* **Which loads have no in-kernel producer, or which producer, and which reads
  free which buffers?** This bounds how early each DMA may be issued, by RAW for
  sources and WAR for destinations.
* **What is each buffer's live range?** This drives VMEM aliasing.

No new language features are needed. The user writes an ordinary Helion kernel
with several top-level `hl.tile` loops, just like the GPU megakernels in
`pretuned_kernels/megakernels/`. The compiler selects the TPU megakernel
lowering whenever the roots are dependent (today that is an error on non-CUDA
backends, `device_ir.py:3582`). All knobs are autotunable `Config` keys.

---

## 1. Catalogue of cross-kernel optimizations in the hand-written megakernels

Sources:

* `~/tpu-megakernels/qwen/decode_megakernel.py` (Q), the whole Qwen3.5 decode
  stack in one call;
* `qwen/dflash.py` (DF) and `qwen/load.py` (LD);
* `kimi/decode_megakernel.py` (K), Kimi K3, one call named `kimi_decoder_stack`;
* `pool_alias.py` and `collectives32.py`.

### A. Residency across op boundaries

| # | What | Where | Compiler derivation |
|---|------|-------|---------------------|
| A1 | **Every inter-op activation stays in VMEM** for the whole stack: normed x, q/k/v, attention out, gate_up, act and residual. Small epilogues (embedding fetch, RoPE tables, argmax) are folded into the same program, so there is no launch per op. HBM sees only weights, caches and final outputs. | Q scratch list ~L921, RoPE/embedding L816-847, argmax L861-905; K `stack` L3618 | A tensor allocated in the kernel body, written by root p, read by root c ≥ p and not returned becomes VMEM scratch (dependency graph plus allocation analysis). |
| A2 | **Small next-layer parameters staged as soon as their slot frees.** Qwen has one single-buffered slot per norm kind: wait for row `li`, then start `li+1`. Kimi stages norms, attention weights and router in single slots refilled from `after_experts`, and its metadata uses layer parity. | Q `norm()` L569-580; K L3894-3988, parity L3967-3986, refill L4725 | Stacked `[L, …]` read-only parameter: refill at the earliest **WAR-legal** point (slot consumed). This is the same mechanism as B1. |
| A3 | **Liveness-based VMEM pool aliasing.** One byte pool is reinterpreted per phase; the expert pool doubles as the dense MLP's slab ring. | K `pool_views` L3811; `pool_alias.py` | A live interval per buffer from the dependency graph, then interval coloring onto a pool. |
| A4 | **Phase-scoped transient scratch** | K `pl.run_scoped` L1679, L2404, L3039, L4269 | Root-local temporaries become `pl.run_scoped` around the root. |

### B. One weight stream across all ops

| # | What | Where | Compiler derivation |
|---|------|-------|---------------------|
| B1 | **A global weight-tile ring with a linear counter across ops, layers and the LM head** (`t_total = t_layers + lm_tiles`). `fetch(g)` decodes g into (layer, op, tile) and starts one DMA into bank `g % 12`. `gemv` waits on tile g, computes, then fetches `g + 12`. The ring never drains at a boundary. | Q `fetch` L468, `gemv` L517, `BANK_COUNT` L27, L369-370, L512-515 | All streamable loads in program order give a static DMA sequence. Refills past the end of a site decode into later sites. Legal because no root writes the streamed tensors (RAW). |
| B2 | **One ring of max-size slots, shared by differently-shaped tiles.** Qwen uses banks `[12, 1024, 1280]` and DMAs into a sub-window. Kimi's dense MLP uses 4 slots sized `max(gate_up slab, down slab)`, reinterpreted per block, so the down slabs start while gate_up computes. **Neither reference uses separate rings per shape.** | Q L462-466, L922; K `run_dense_mlp` L4803-4880, L4819 | One ring with slot = max tile. Tile shapes are chosen *jointly across ops* so slots are not wasted (B3). |
| B3 | **Stream tile shapes chosen across ops.** Qwen pads widths so every matrix tiles into the same slot. Kimi uses contiguous full-width K-row slabs ("a column block would be 7168 short DMA segments"). | LD `tile_hw` L270, `schedule` L229, `tile_matrix` L281, width padding; K L4806-4810 | The autotuner searches `block_sizes`. The compiler seeds them with one shared slot shape built from contiguous row slabs. |
| B4 | **The ring is deep enough to cover the longest refill-free gap.** About 22-24 MiB in flight in Qwen and about 21 MiB in Kimi, sized for attention, collectives and routing, not just DMA latency. | Q `BANK_COUNT=12` × up to 2.5 MiB; K 4 × 5.25 MiB | Bytes in flight ≥ BW × (DMA latency + longest refill-free gap). This is the `pallas_stream_depth` heuristic and search range. |
| B5 | **Next-layer weights prefetched into the freed pool while a collective is in flight.** `after_experts()` stages layer l+1, then `_all_reduce_rows` runs. | K L4704, L4878-4881, L1977 | Follows from B1 plus A3. The stream keeps running across a collective root once the pool is dead. |
| B6 | **Issue before compute with paired slots, and speculative starts.** The next expert's copies start before the current compute, and expert DMAs start during routing. | K L868-885, L1600 | Refill placement within an iteration is a codegen choice: issue first when depth ≥ 2c. |

### C. Cross-op fusion

| # | What | Where | Compiler derivation |
|---|------|-------|---------------------|
| C1 | **Chunked MLP:** gate/up chunk c, then act c, then down chunk c accumulates into the output. The two producer/consumer ops are interleaved at chunk granularity, so `act` never fully materializes and VMEM stays bounded at large M. | DF `make_draft_kernel` L629-1190 (host-interleaved gate/up per chunk, L1038-1041) | **Written in source** as one root (§1.5). The dependency edges carry no coordinate relation (`TileDependency`, `tile_dependency.py:4665`), so automatic fusion is not on the table. |

### D. Cross-op synchronization and structure

| # | What | Where | Compiler derivation |
|---|------|-------|---------------------|
| D1 | **Hoist independent loads ahead of the previous op's compute.** MLA primes KV blocks before the projections. Qwen issues the ring prologue before the barrier and embedding all-reduce. DFlash starts layer-0 weight DMAs before the barrier and psum. | K `mla` L2485; Q L829-838; DF | Issue at the earliest point after the last in-kernel writer, which is kernel start for weights. This is the ring prologue. |
| D2 | **Write-behind across ops.** State and cache writes are started after the producing op and waited at the next layer or drained at exit. | Q L599-602, L851-853; DF | An output produced by root p is copied `VMEM → HBM` asynchronously after p. The wait happens before buffer reuse (WAR) or at exit. |
| D3 | **In-kernel collectives between ops** over remote DMA with double-buffered slots: no launch boundary at TP reductions. | `collectives32.py`; K `_all_reduce_rows` | The user writes them in source (`pallas/distributed_ops.py`). The compiler orders them and overlaps them with B5. |
| D4 | **Layer loop.** Qwen runs `fori_loop(0, layers)` with a `lax.cond` on layer type, without peeling. Kimi peels layer 0 because "a control-flow region blocks overlap". | Q L808-810, L850; K L4015-4018, L4967-4968 | Fold isomorphic root sequences into `pl.loop` (the stream decode becomes affine in the layer, Q L797-801). Peel non-isomorphic layers. Needs the source gap closed (§1.5). |

---

## 1.5 Ranking and ownership

**Ownership rule.** These are the owners:

* **Compiler:** semantics-preserving placement and movement decisions that can
  be derived from the tile-dependency graph, shapes and dtypes. Litmus test: if
  expressing it in Helion source would need DMA, semaphore, VMEM-placement or
  slot APIs, which Helion deliberately does not expose, it belongs here.
* **Autotuner:** tile shapes and depths.
* **Source:** anything that restructures *what is computed*.
* **Host:** load-time weight layout.

Ranked by expected impact on an HBM-bound TPU decode stack:

| Rank | Optimization | Owner | Notes |
|---|---|---|---|
| 1 | **The ring never drains: B1 + B4 + D1 (+ B2 sharing)** | Compiler + autotuner (depth) | This is the core of the Qwen design. For a single MLP the boundary cost is a few %; at stack and TP scale it is one bubble per op × about 8 ops/layer × layers. Not expressible in source. |
| 2 | **B3 stream tile shapes chosen across ops** (contiguous slabs, one slot shape) | Autotuner + compiler seed (host padding optional) | DMA efficiency **and** no slot waste in the shared ring. The size of the gain must be measured; the references give no number. |
| 3 | **A1 inter-op VMEM residency plus single launch** | Compiler | The direct gain is modest (KB-sized activations, µs launches). Its real value is that it makes rank 1 possible. |
| 4 | **B5 + D3 collective overlap** (TP) | Compiler (overlap) + source (collective) | Follows from 1 and 5. |
| 5 | **A3 VMEM liveness aliasing** | Compiler | Enables deep rings once many buffers are live (attention, multi-layer). |
| 6 | **D4 layer loop** (code size, compile time) | Source + compiler | Feasibility requirement for 30-60 layers. |
| 7 | **A2 small-param staging** | Compiler | A small per-layer latency. |
| 8 | **D2 write-behind** | Compiler | Matters for cache and state writes. |
| 9 | **B6 issue-before-compute** | Compiler (tunable) | |
| 10 | **A4 scoped scratch** | Compiler | |
| 11 | **C1 chunked MLP** | **Source + host** | Only matters at large M. |

**What this means for sequencing.** Ranks 1-3 are milestones M1 and M2.

**C1 as source:** write it as one root with a full-width down accumulator. The
naive `w_gate[:, tc]` column chunks break B3 (they become short DMA segments).
DFlash gets away with it only because its weights are small and its gate/up are
host-interleaved per chunk. So it needs a host layout too, e.g. `[N, K]`
gate/up so a chunk is contiguous rows.

**Expressibility gap for D4.** A host `for layer in range(L)` around roots
raises `NestedGridLoop` (`type_propagation.py:1068`). Options:

* **(a, recommended)** allow static host-level `range`/`hl.static_range` loops
  around roots, unrolled at trace time and folded back by the compiler. This
  relaxes an existing restriction rather than adding an API.
* (b) Python metaprogramming outside the kernel.
* (c) One launch per layer.

This needs user sign-off and does not affect M1 or M2.

---

## 2. What Helion has today

Verified against the code by the feasibility review.

**Pallas launcher.** It is `pl.kernel` on `TensorCoreMesh(num_cores=1)` with
`pltpu.emit_pipeline` over `grid or (1,)`, not `pallas_call`. Grid steps run
in order on one core; there are no `dimension_semantics`. `vmem_limit_bytes` is
already set to device capacity (`runtime/pallas/launcher.py:165`, ~L1616-1800).
The grid comes from `pid.codegen_grid()` (`device_function.py:1960-1990`).

**Multi-root kernels.** `visit_For` (`generate_ast.py:1408-1580`) uses
`ForEachProgramID` (`program_id.py:542`), giving a summed grid plus an if/else
on `pid_shared`. **On Pallas this is already broken:**

* `ForEachProgramID.pid_info` is empty (`program_id.py:550`);
* so `_compute_block_spec_info` gives full blocks (`backend.py` ~L1005-1011);
* while `_tile_pattern_code` emits `":"` for tileable dims
  (`pallas/codegen.py:824-870`);
* and no `test_pallas*` test has two or more roots.

**Dependency gate.** `device_ir.py:3570-3598`:
`build_tile_dependency_graph` runs, then `_install_dependency_phases` (3109-3155,
which rewrites phases per stage), then `LoopDependencyError` on non-CUDA.
`CrossLoopSchedulingError` is also raised for barriers mixed with edges
(`device_ir.py:3115`) and for `allocation_id < 0` (aliases/views) in a phase
with several roots (`tile_dependency.py:6119`).

**Dependency data.**

* `TileDependency` (`tile_dependency.py:4665`) carries producer and consumer
  roots, `allocation_id`, `tensor_names` and `access_dependencies`.
* `TileDependencyRelation` (L4652) carries only incidence; **there is no
  coordinate relation.**
* Per-dim access shape comes from `TileAccess` (L4541).
* TileAccess is built once on the original graphs
  (`device_ir_analysis.py:1615-1780`), while codegen uses per-config graph
  copies (`device_ir.py:1534`).

**Body-created tensors** become launcher outputs today: in-place VMEM in/out if
they are read, HBM outputs otherwise (`backend.py:1350-1390`). Internal scratch
exists, but only for storages that are read ∩ written ∩ remote
(`pallas/internal_scratch.py` ~L118). `sorted_args` exclusion
(`device_function.py:1036-1045`) and `_load_route` (`codegen.py:38`) already
handle internal scratch.

**DMA streaming.** `_codegen_fori_loop` (`pallas/tracing_ops.py:4716-5585`)
streams per inner loop with 1-2 slots (`_j % 2`). The classifier
`_classify_pipelined_tensors` (L4182) is per-tensor and conservative. An
excluded tensor **silently falls back to a full VMEM BlockSpec.** Slices are
built from the current grid's `offset_var` (`_build_dma_slices`, L5090).

**`pl.program_id` uses that break under one sequential program:**

* emit_pipeline loop type (`tracing_ops.py:3443-3457`);
* ordered_carry (`ordered_carry.py:169-175, 268`);
* copy guards (`launcher.py:460-497, 1180-1186`).

### Lessons from the earlier attempt (`../helion-tpu-megakernel`)

* Do not do post-codegen AST surgery.
* Do not handle dependencies only between adjacent root pairs.
* Do not use prefetch depth 1.
* Do not gate it behind an opt-in boolean setting.

Instead: plan from device IR before codegen, cover the full dependency graph,
use a global stream with tunable depth, and select the lowering automatically.

---

## 3. Design

### 3.1 Overview

```
hl kernel (N dependent top-level roots, static_shapes=True)
  device IR → TileAccess → TileDependencyGraph                 (existing)
  gate (device_ir.py:3570): Pallas + implicit deps → megakernel mode
     roles per storage: INTERMEDIATE | READ_ONLY | OUTPUT | ...   (from graph)
  backend.pre_codegen: PallasMegakernelPlan (pallas/megakernel.py, per config)
     RootSchedule, Placement, StreamPlan (sites, order, ring), VMEM budget
  codegen: roots → sequential pl.loop bodies; streamed loads → ring
  launcher: grid=(1,); weights HBM; intermediates scratch; small args VMEM
```

### 3.2 Gate (M1)

At `device_ir.py:3570`: if the backend is Pallas and
`implicit_dependency_starts` is non-empty, enter megakernel mode instead of
raising. That means:

* keep the graph;
* skip `require_persistent_blocked` and `enable_cross_loop_pipeline`;
* classify storage roles from the graph and allocation analysis here, keyed by
  storage or name rather than fx node, so they survive per-config graph copies.

Program order on one core satisfies every RAW, WAR and WAW edge by
construction. The graph's job is classification and DMA legality, not ordering.

The following are rejected with clear errors:

* aliases and views (`allocation_id < 0`);
* dynamic shapes;
* `hl.barrier` mixed with implicit edges;
* the `emit_pipeline` loop type and ordered_carry (both use `pl.program_id`).

### 3.3 Root sequencing (M1)

New `SequentialRootsProgramIDs(ForEachProgramID)`:

* `codegen_pid_init` returns `[]`;
* no `pid_shared -= …`;
* `codegen_grid` returns `(1,)`.

In `visit_For`, each root's body is wrapped in place of the if/else chain:

```python
@pl.loop(0, N_r)          # N_r static; N_r == 1 → straight-line
def _root_r(pid_shared_r):
    <root body; per-dim pids decoded from pid_shared_r by FlatProgramIDs, unchanged>
```

`shared_pid_var` is set per root in `tile_strategy.py` (~L6014, 6075,
6437-6545, 7471, 8060). Watch scoping inside the closure: hoisted parent
statements, `flush_deferred_rdim_defs`, DCE and nonlocal names.

### 3.4 Placement (M1)

| Placement | Rule | Lowering |
|---|---|---|
| `SCRATCH` | Body-allocated, not returned | `plan_internal_remote_scratch` **without the `remote` requirement**. Rows rounded up to the native sublane tile (Kimi L5090 hit corruption with a 2-row scratch). |
| `WHOLE_VMEM` | Kernel arg or returned tensor ≤ threshold (default 1 MiB) | Full-array VMEM block. One DMA before the body, writeback after. |
| `HBM_STREAM` | Read-only (no store on its `allocation_id`), block-aligned affine access, divisible extents | HBM ref (`_hbm_arg_indices`), fed by the ring (§3.5). **Bypasses `_classify_pipelined_tensors`. Failing eligibility is a compile error, never a silent full-VMEM fallback.** |
| `HBM` | Large outputs or written inputs | HBM ref with explicit copies (write-behind in M3) |

In megakernel mode, `can_tile=False` everywhere (`plan_tiling.py:759`), so every
access is an explicit `_ds_expr` slice using `offset_var` (`codegen.py:1039`).
That also sidesteps the existing multi-root BlockSpec bug. VMEM budget
(scratch + whole + ring + reserve ≥ 8 MiB for temporaries and Mosaic internals)
is checked in `autotune_config_is_viable`.

### 3.5 The global weight stream (M2)

**Sites and order.** A *site* is the innermost loop of a root that contains
`HBM_STREAM` loads: an inner `hl.tile` fori loop, or the root tile loop itself.
It has `c` loads per iteration (e.g. gate and up) and `S` static iterations.
Global DMA index for load `l` at site iteration `it`:
`g = T_site + it*c + l`, where `it = pid_shared_r * NK + fori_index`.

**One ring.** A single ring `ring: (D, *slot_shape)` per dtype, with
`slot_shape = max` over streamed tiles (rounded to (16|8|32, 128) by dtype),
plus a semaphore array `(D,)`. Each DMA targets the sub-window
`ring.at[g % D, :rows, :cols]` (Qwen style). The default-config seed chooses
block sizes that give **all sites one tile shape**, so there is no waste.
Byte-pool reinterpretation (Kimi style) comes with M3 aliasing.

**Protocol** (soundness rules from review):

1. **Prologue (D1, B1).** Issue DMAs `0 … D-1` at the earliest legal point. For
   weights that is kernel start, before root 0. In general it is after the last
   root writing the source (RAW), and after the last read of whatever
   previously occupied the destination (WAR, once aliasing exists).
2. **Consume.** `wait` on load `l` immediately before its use. The wait
   descriptor is built from **the consumer's own static source slice and the
   same slot sub-window**, so there is no decode on the wait path. Semaphores
   count bytes, so a wait must match the started copy's destination size, and
   copies of different sizes never share a slot semaphore. Per-load waits give
   split waits (compute starts on the first arrival).
3. **Refill.** After the iteration's compute, issue `g + D` for each load. Its
   slot is the one just freed, so there is no WAR hazard on the ring. A refill
   index past the site's end decodes into the next site, possibly in a later
   root or layer. **That is the cross-op stream.** Require `D ≥ c` and seed
   `D ≥ 2c` (with D = c nothing is in flight during compute).
   Issue-before-compute (B6) is a tunable once `D ≥ 2c`.
4. **Decode cost.** In M2a, the refill decode is a `pl.when` chain over site
   ranges. It was measured at ≤ 2 µs over the whole MLP (`target_mk.py`).
   M2b, deprioritized and gated on profiling, splits each site into a steady-state part
   (refill target in the same site, so no branch) and a statically unrolled
   tail of `ceil(D/c)` iterations that refills into the next site. This avoids
   control-flow regions in the hot loop (Kimi peels layer 0 for this reason).
5. **Exit.** `g < G_total` guards the refill. The compiler asserts statically
   that every index is consumed exactly once (any skipped load leaves a
   semaphore nonzero) and drains write-behind at exit.

**Depth heuristic.**
`ring bytes ≈ BW × (DMA latency + longest refill-free gap)`, clamped to the VMEM
budget. Measured on v7, the MLP saturates at D = 4c and 16 MiB, while D = 2c
loses 15%. Seed `max(4c, ceil(16 MiB / slot))` and search
`pallas_stream_depth ∈ {2c … 16}` slots. Longer refill-free gaps (attention,
collectives) will need more.

### 3.6 VMEM liveness and write-behind (M3)

* **Live intervals.** For each SCRATCH tensor: first writer root to last reader
  root. The ring is live from the prologue to the last consume.
* **Packing.** Interval coloring. Stage 1 aliases identical `(shape, dtype)`
  only. Stage 2 uses byte-pool reinterpretation (upstream `ref.bitcast`/reshape,
  or a supported equivalent of `pool_alias.py`).
* **Scoped scratch.** Root-local temporaries become `pl.run_scoped` (A4).
* **Write-behind (D2).** An output produced by root p gets an async
  `VMEM → HBM` copy after p. The wait is placed before buffer reuse or at exit.

### 3.7 Decoder layer (M3): scalar operands, dynamic slices, cross-root HBM hazards

A draft, to be revised with the M3-prep findings (`/tmp/mk/m3/FINDINGS.md`).

* **Scalar operands (F9).** A small integer input tensor (decode position,
  token id) whose elements are only used as scalars (index arithmetic, masks,
  loop bounds) is placed in SMEM. Use `pltpu.SMEM` operand placement, or scalar
  prefetch when the grid needs it. Reads become `ref[0]` scalars. The rule is
  generic: element reads of an int tensor that feed index or mask expressions.
* **Dynamic slices.** `cache[h, p, :] = row` and `cache[h, 0:ctx, :]` with a
  traced `p` lower to DMAs whose start is a scalar expression. These are
  ordinary HBM refs with `.at[pl.ds(p, 1)]`, so they bypass the global ring
  because they are not read-only.
* **Cross-root HBM RAW/WAR.** A tensor that is both written and read in HBM
  (the KV cache: write row p in root a, read rows `0..ctx` in root b) needs the
  write DMA to be complete before b's first read DMA. The rule:
  * a write-DMA's wait sits before the first later DMA whose region may overlap
    (conservative: same tensor);
  * otherwise it sits at exit (write-behind, D2).
* **Dynamic trip counts are not on the M3 critical path.** Reading the whole
  static context with a `t ≤ p` mask streams 2 MB of KV per GQA layer at
  ctx = 2048. That is about 0.6 µs per GQA layer, or about 9 µs over the 16
  GQA layers of a step. A `hl.tile(0, p + 1)` with data-dependent `end` (F9b)
  is deferred until long contexts matter. Then it becomes a fori loop with a
  traced trip count, and its local stream (F4) is sized for the max.
* **Ring underflow during non-streaming roots.** Roots without ring sites
  (norms, RoPE, attention over a VMEM-resident cache) issue no refills. The
  ring stays full (D DMAs in flight) and drains only once its consumers resume,
  so no stall occurs as long as each such gap is shorter than the ring's
  ~4.5 µs of buffered bandwidth (16 MiB at 3.7 TB/s). The depth heuristic adds
  the longest gap.

**Revision after the branch survey (`/tmp/mk/f4/SURVEY.md`).** This is
needed for M4 and M5: 16 GQA layers × 2 MiB of KV do not fit in VMEM.

* **Mosaic constraint.** A dynamic single-row DMA on the tiled (second-minor)
  dim does not compile: "offsets along tiled dimensions must be aligned". A
  row write into `[kvh, ctx, d]` therefore always moves an aligned block of S
  rows (16 for bf16) that contains `p`.
* **F4, HBM-resident read/write tensors.** `plan_hbm_resident()` runs from
  `plan_megakernel`. A tensor qualifies when every access is one of:
  * a tiled load inside a fori loop;
  * a root-scope store with a scalar index.

  Qualifying tensors:
  * stay in HBM (aliased in place);
  * are exempt from the outer-access rejection in
    `_classify_pipelined_tensors`;
  * get a 2-slot local stream;
  * leave the resident-VMEM budget.

  Main already has a data-dependent fori trip count (a traced `cdiv`, with
  prime and prefetch guarded by `_num_iterations`). F9b only needs the
  megakernel gate split into symbolic vs data-dependent bounds. Ring loads
  stay restricted to loops with a static end.
* **F7, aligned-block write with a deferred wait.**
  * **Stage 1:**
    * read block `p//S` into a staging tile (the read can be hoisted to
      root start, because nothing earlier writes it);
    * patch row `p%S` with an iota `where`;
    * start the write asynchronously and register it as pending on that
      storage;
    * the wait goes before the next read DMA of the same storage, or at the
      end of the root body.
  * **Stage 2:** when a later stream in the same layer reads the block that
    contains `p`, forward the row into that stream slot at `t == p//bt` and
    write the block back behind the stream. This removes the staging read.
    It reuses the write-behind wait and drain logic from
    `pallas-store-double-buffering` (wait-previous, drain under
    `num_iterations > 0`).
* **Tests.**
  * Single-root attention with `pos` ∈ {0, 15, 16, ctx-1}.
  * A two-root write-then-attend layer.
  * GDN conv-state write-behind, which is structurally different.

**F3 revision (slot shapes), from the DMA probe in `/tmp/mk/f3/RESULTS.md`.**

* **Slot size, not row width, decides bandwidth.** Bandwidth depends only on
  slot bytes: below ~256 KB a fixed per-DMA cost dominates. Row width does
  not matter.
* **`_ring_layouts` today groups by (dtype, rank)** and sizes slots to the
  elementwise maximum tile. That is wasteful as soon as tiles differ in
  shape, e.g. (128, 2176) gate/up next to (2176, 256) down.
* **One ring per distinct tile shape instead.** Shapes within ~10% waste of
  each other may be clustered.
* **Depth per ring.** `pallas_stream_depth` is the depth of the ring with the
  most traffic; by default it holds ~16 MiB, with at least 4× its loads per
  iteration. Every other ring gets about as many bytes in no more slots,
  with at least 2× its loads per iteration, capped at its stream. All of this
  is clamped to the VMEM budget. A traffic-proportional split was measured
  first and underfed rings with large slots: on M2, 107 µs vs 98 µs.
* **Prologue handoff.** A ring read only after another ring's reads end
  starts its prologue at that ring's first index without a refill, instead
  of at kernel start. Folded loops of more than one layer interleave the
  rings, so they are excluded. Without the handoff the DMA queue drains
  between sequential rings, about 3% on the M2 dense MLP.
* **Unchanged.** Refill chains and `_check_ring` are already per ring.
* **Default tile choice for streamed loads under F12.** Pick a tile that:
  * divides the dims;
  * is ≥ 512 KB;
  * has an N width ≥ 256 or the full dim;
  * prefers fewer, larger tiles under the ring budget.

  The autotuner still explores, with the opt-in non-pow2 search enabled in
  megakernel mode.

### 3.8 Folded host loops (M4, F5) and periodic branches (M5)

Decision 1 of the roadmap: a host `for i in range(L)` around roots is legal in
megakernel mode.

* **Front end.** Relax `NestedGridLoop` (`type_propagation.py:1071`) when the
  loop is a host `range` loop and the kernel is a megakernel candidate. Type the
  body once with `i` as a `SymIntType` (merged literals already produce a
  SymInt). Roots inside receive `i` like any host scalar: a sympy symbol in the
  device graphs (`device_ir.py` `_get_symnode`). Record a
  `HostLoopInfo(var, start, stop, step, roots=[r0..rk))` on `DeviceIR`. Roots
  are traced **once per body**, not once per layer, so compile time stays
  flat in L.
* **Codegen.** `SequentialRootsProgramIDs` wraps roots `[r0..rk)` in a
  `lax.fori_loop(0, L, body)` / `pl.loop`. The symbol for `i` binds to the
  induction variable instead of a kernel parameter.
* **Stacked weights.** `w[i, tn, tk]` is an affine slice with a traced leading
  index, so it stays a ring site. Global stream index:
  `g = (i - start) * T_body + T_site + it * c + l`. The refill `g + D` decodes
  as `layer = g' // T_body` (offset `g' % T_body`), followed by the per-body site
  `pl.when` chain, issuing from `w.at[layer, …]`. The ring flows from layer i
  into layer i + 1, and from the last layer into the roots after the loop.
  Plan §3.5's decode becomes two-level, but the site chain still has only
  about 8 entries.
* **Small params (F6).** `norm_w[i]` (`[L, H]`, read-only) gets a 2-slot
  buffer. Iteration i+1's slice is issued after slot (i+1) % 2's last reader
  of iteration i - 1.
* **Periodic branches (M5).** If a host `if` around roots has a condition that
  is a function of `i mod P` for a static P (the Qwen 3:1 pattern; Gemma 5:1
  local/global), the compiler unrolls the folded loop by P: an outer
  `fori(L // P)` holds P static copies of the body, each with the branch
  resolved statically, plus a static remainder. No `lax.cond` sits in the hot
  path, so the stream schedule per period is static. Kimi's comment, "a
  control-flow region blocks overlap", is the reason. A non-periodic
  data-dependent branch falls back to `lax.cond`, and the ring index advances
  by the branch-specific count on both sides (needs a statically known count;
  otherwise reject).
* **Per-type stacks.** A model with two layer types indexes per-type stacks
  with affine expressions of the period index, for example
  `w_gdn[3 * p + r]` and `w_attn[p]`. The user can write `for p in range(L // 4):
  for r in hl.static_range(4): if r == 3: … else: …`, which needs no mod
  analysis. Both spellings are supported.
* **Tests (≥ 2 structurally different).** These kernels exercise it:
  * N-layer dense MLP stack;
  * N-layer GQA decoder;
  * a synthetic two-type periodic stack (e.g. P = 3) with different weight
    shapes per type.

**As implemented (F5, worktree `helion-f5-loops`).** Differences from the
design above:

* **Front end.** `NestedGridLoop` is relaxed in `_check_root_parents`, with
  classification in `_foldable_loop_var` (`type_propagation.py`).
  * Bounds: literal or constexpr `range`, non-empty, **step 1 only**, Name
    target, no `else`.
  * `i` is an unbacked SymInt constrained to `[start, stop-1]`.
  * The body may hold only roots and `i`-independent `empty` allocations
    (hoisted). Folded loops cannot nest.
  * Violations raise `UnsupportedFoldedHostLoop` with the reason.
  * `HostLoopInfo` holds `var, symbol, start, stop, first_root, end_root`.
* **Codegen.** Emits `@pl.loop(start, stop)`, not `fori_loop`.
  * The ring prime and scratch zeroing go before the loop
    (`codegen_root(prologue_body=...)`).
  * A single-root kernel with a folded loop also uses sequential-roots mode.
* **Stream index.** `g = B + (i - start) * T_body + T_site + it * c + l`
  (`RingLoop`).
  * The refill decodes `_layer = _gn // T_body` and `_gn' = _gn % T_body`, then
    runs the body's site `pl.when` chain.
  * The leading index is any affine function of `i` (`a.at[1 + 2 * _layer]`).
  * 2-D slots serve 3-D stacked weights.
  * `_check_ring` simulates the first period, the steady state and the loop
    exit.
* **F6 not done.** `norm_w[i, :]` is loaded whole each iteration. Measured
  per-layer time is still flat in L.
* **Periodic branches.** Only the explicit spelling is supported. Host
  `hl.static_range` loops are unrolled before type propagation, and host `if`s
  that hold device loops and whose test is a compile-time constant keep one
  branch.
  * Mod analysis of `if i % P == k` inside one folded loop is **not** done.
  * The `lax.cond` fallback is **not** done.
  * A runtime `if` on `i` around roots is rejected.
* **Measured** (M=1, against an XLA `lax.scan` over the same weights; `w1` writes
  gate/up full width):

  | Kernel | L | Megakernel | XLA |
  |---|---|---|---|
  | H=4096, I=12288 | 64 | 87.2 µs/layer, 3.46 TB/s | 95.2 |
  | H=5120, I=2176, `w1` | 64 | 25.3 µs/layer, 2.64 TB/s | 30.8 |
  | H=5120, I=2176, tiled | 8 | 38.2 µs/layer | 31.2 |

  * The tiled H=5120 kernel loses because I = 17·128 forces 128-wide MXU tiles.
  * Compile time is flat in L.

### 3.9 Later

* B5/D3 collective overlap (M7).
* C1 stays a source pattern.

---

## 4. Config surface (no language changes)

| Key | Meaning |
|---|---|
| `block_sizes` (existing) | Tile shapes. The seed picks one shared contiguous slot shape across sites. |
| `pallas_loop_type` (existing) | Restricted to `fori_loop` in megakernel mode. |
| `pallas_stream_depth` (new) | Ring slots, i.e. prefetch distance in DMAs. Only in megakernel mode. |

---

## 5. Milestone: Qwen3-8B dense MLP (M1 → M2)

**Shape** (`pretuned_kernels/megakernels/qwen3_decode_layer`): M=1, H=4096,
I=12288, eps=1e-6, bf16. Weights use the `x @ W` layout: `w_gate, w_up: [H, I]`
and `w_down: [I, H]`. That is 302 MB (288 MiB) of weights, so ~82 µs at
~3.7 TB/s per TensorCore. The gate is measured against a pure DMA copy of the
same bytes.

```python
@helion.kernel(backend="pallas", static_shapes=True)
def qwen3_dense_mlp(x, residual, norm_w, w_gate, w_up, w_down, eps=1e-6):
    m, h = x.shape
    inter = w_gate.shape[1]
    hidden = torch.empty([m, h], dtype=torch.float32, device=x.device)  # returned
    normed = torch.empty([m, h], dtype=x.dtype, device=x.device)        # SCRATCH
    act = torch.empty([m, inter], dtype=x.dtype, device=x.device)       # SCRATCH
    out = torch.empty([m, h], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):                                  # root 0: residual + RMSNorm
        s = x[tm, :].float() + residual[tm, :].float()
        hidden[tm, :] = s
        normed[tm, :] = (s * torch.rsqrt(torch.mean(s * s, -1, keepdim=True) + eps)
                         * norm_w[None, :].float()).to(x.dtype)
    for tm, tn in hl.tile([m, inter]):                     # root 1: gate/up + SiLU*mul
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(h):
            a = normed[tm, tk]
            g = torch.addmm(g, a, w_gate[tk, tn])
            u = torch.addmm(u, a, w_up[tk, tn])
        # rounding points as in the Qwen reference: projections → bf16, silu → bf16
        act[tm, tn] = torch.nn.functional.silu(g.to(x.dtype).float()).to(x.dtype) * u.to(x.dtype)
    for tm, tn in hl.tile([m, h]):                         # root 2: down + residual
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(inter):
            acc = torch.addmm(acc, act[tm, tk], w_down[tk, tn])
        out[tm, tn] = (acc + hidden[tm, tn]).to(x.dtype)
    return out, hidden
```

### Measured baselines

TPU v7, one TensorCore, measured 2026-10-02 on pod `eche-helion-mk`.

| Variant | Time | Bandwidth | Notes |
|---|---|---|---|
| (d) DMA-only ring roofline | **91.7 µs** | 3.29 TB/s | Same order and shapes. Saturates once ≥ 8 MiB is in flight (tk=256, D=4). D=2 at tk=256 gives 127 µs. |
| (c) XLA `jax.jit` MLP | **100.9 µs** | 2.99 TB/s | 91% of (d) |
| (b) three separate Helion kernels | **365 µs** | 0.83 TB/s | M padded to 16, fori_loop, `[16, 4096, 256]`, run through jax_fn. The existing fori DMA keeps one buffer per weight with no double-buffering. |

The gate (≥ 85% of (d) **and** faster than (c)) therefore means **< 100.9 µs**,
i.e. ≥ 91% of the DMA roofline.

How these were run, and what they exposed:

* The pod has no `torch_tpu`, so kernels run through
  `to_code(jax_fn=True)` + `jax.jit`.
* That needed lifting the jax_fn guard on scratch/HBM kwargs for kernels
  without remote copies.
* jax_fn returns multiple outputs in launcher order.
* A row-1 bf16 block (M=1) fails real Mosaic compilation (tiling 2) but passes
  interpret mode.

Scripts are in `~/torus-helion/tpu-megakernel-scripts/`: `baselines.py`,
`baseline_helion3.py`, `target_mk.py`.

### Hand-written target lowering (`target_mk.py`)

This is exactly what §3.4-3.5 says M2 should emit:

* one `pallas_call`;
* prologue, then root 0, then root 1 (3 × fori over 16 k-steps, c=2), then
  root 2 (fori over 48);
* `normed` and `act` as VMEM scratch;
* one ring of `[tk, 4096]` bf16 slots with byte-matched waits;
* refills decoded with a `pl.when` chain.

M padded to 16. Correct against the XLA reference within bf16 rounding.

| Variant | Time | Bandwidth | % of (d) |
|---|---|---|---|
| tk=256, D=4 (8 MiB) | 108.0 µs | 2.80 TB/s | 85% |
| **tk=256, D=8 (16 MiB), cross-root refill** | **93.7 µs** | 3.22 TB/s | **98%** |
| tk=256, D=12 (24 MiB), cross | 93.7 µs | 3.22 TB/s | 98% |
| tk=256, D=8, **drain at root boundary** | 95.5 µs | 3.16 TB/s | 96% |
| tk=256, D=8, accumulators in VMEM scratch instead of carried | 93.6 µs | 3.23 TB/s | 98% |
| tk=512, D=6 (24 MiB), cross | 94.0 µs | 3.21 TB/s | 98% |

**Conclusions**

* **The design clears the gate:** 93.7 µs, versus 100.0 µs for XLA in the same
  run.
* **Cross-root refill (B1) is worth about 1.8 µs per boundary.** That is 2% of
  this kernel, with a single boundary (gate_up → down). It scales with
  boundaries per layer.
* **Depth rule.** D = 2c is not enough (108 µs). The seed is
  **D ≥ 4c and ring ≥ 16 MiB**: D=8 for tk=256, c=2. Deeper rings give no
  further gain here, because the MLP has no long refill-free gap.
* **The `pl.when` refill-decode chain in the hot loop costs ≤ 2 µs.** So
  **M2b (steady-state/tail split) is deprioritized**; M2a is enough for the
  milestone.
* **Carried accumulators are fine.** There is no need to force VMEM scratch
  accumulators.

### Seed config

All three weights use one tile shape, `[256, 4096]` bf16. Each tile is 2 MiB
of 8 KiB contiguous row segments, so there is one ring with no waste.

| Root | Block sizes | Tiles | Streamed DMAs |
|---|---|---|---|
| Root 1 | `tn=4096`, `tk=256` | 3 tiles × 16 k-steps, c=2 | 96 |
| Root 2 | `tn=4096`, `tk=256` | 1 tile × 48 k-steps, c=1 | 48 |

That is 144 DMAs × 2 MiB = 288 MiB. The seed depth is 8 slots, i.e. 16 MiB in
flight; this was measured as saturating (see the target lowering above). Accumulators `g` and `u` are each `[16, 4096]` f32 =
256 KiB. Scratch, whole-VMEM and reserve add a few MiB, well under budget.

### Expected lowering

* One kernel, `grid=(1,)`.
* `normed` and `act` placed as SCRATCH.
* `x`, `residual`, `norm_w`, `hidden` and `out` placed as WHOLE_VMEM.
* The ring prologue is issued before root 0, so RMSNorm overlaps the first
  24 MiB.
* Root 1's last 6 iterations refill into root 2's down tiles. The boundary is
  crossed with no drain.

### Steps and gates

1. **Pre-M1 test.** A two-root, independent Pallas kernel (currently broken).
   It should pass once megakernel mode handles all multi-root Pallas kernels,
   or at least once `can_tile=False` is used for them.
2. **M1.**
   * Gate, sequential roots, placement.
   * Weights are streamed by the existing per-inner-loop fori DMA, or by a
     depth-2 ring with no cross-site refill.
   * Interpret mode (`HELION_PALLAS_INTERPRET=1`) at H=256, I=512, M∈{1,8}.
   * Code assertions: one kernel, `grid=(1,)`, no `program_id`, `normed`/`act`
     scratch.
3. **M2.**
   * Ring, protocol, `pallas_stream_depth`, seed heuristic.
   * Interpret-mode tests: DMA count = 144-equivalent, refill crosses the root
     boundary, depths {2c, 8, 12}, unequal tile shapes (sub-window), and static
     consumption.
4. **TPU at the Qwen shape.** Compare:
   * (a) the megakernel;
   * (b) three separate Helion kernels;
   * (c) XLA (`jax.jit`/torch reference);
   * (d) a measured DMA-copy roofline.

   **Gate: (a) ≥ 85% of (d), and faster than (b) and (c).**
5. **Deliverables.** `test/test_pallas_megakernel.py` and
   `examples/tpu_dense_mlp_megakernel.py` (with `main()`).

### Implementation notes (needed for the milestone, not cross-kernel optimizations)

* **MXU row padding for M < sublane count.** Pad LHS rows to the dtype's
  native count (16 for bf16), broadcasting for M=1. Use
  `preferred_element_type=f32` and slice `[:M]` (Kimi `_dot` L44-58, Qwen L527).
  First check what the current Pallas dot lowering already does for M=1.
* **Dynamic HBM row offsets must be tile-aligned** (Qwen L826-847). Stream
  eligibility checks this.
* **Large accumulators.** Watch for extra per-step copies of the 256 KiB
  accumulators in the fori carried-state path.

## 6. Phases

| Phase | Content | Catalogue |
|---|---|---|
| M1 | Gate, sequential roots, placement, VMEM budget, multi-root Pallas fix | A1 |
| M2 | Global ring with shared slots, protocol, depth, seed shapes | B1, B2, B3, B4, D1, B6 |
| M3 | Liveness pool, run_scoped, write-behind | A3, A4, D2 |
| M4 | Multi-layer (source gap, root folding), small-param staging, collective overlap | D4, A2, B5, D3 |
| source | Chunked MLP pattern | C1 |

## 7. Risks

1. **Untested ground.** Multi-root Pallas has no tests today, and closure
   scoping inside `pl.loop` (hoisting, DCE, nonlocals) is unverified.
2. **Semaphore byte-count semantics.** A mismatched wait hangs or races
   silently, and interpret mode may not catch it. Mitigations: the static
   consumption assertion, and waits built from the consumer's own slice.
3. **Control-flow regions** in the refill path blocking overlap (M2b split),
   plus accumulator copies.
4. **Analysis versus config mismatch.** Roles come from the original graphs
   while placement uses per-config copies. Key everything by storage or name.

## 8. Implementation map

**M1**

* `device_ir.py:3570`: megakernel branch.
* `program_id.py`: `SequentialRootsProgramIDs`.
* `generate_ast.py` `visit_For`: `pl.loop` wrapping.
* `tile_strategy.py`: per-root `shared_pid_var`.
* `pallas/backend.py`
  * `pre_codegen`: plan, `can_tile=False`, scratch.
  * `build_launcher_args`: scratch and HBM indices.
* `pallas/internal_scratch.py`: drop the remote requirement.
* `autotuner/config_spec.py`: loop-type restriction.

**M2**

* New `pallas/megakernel.py`: plan, sites, order, ring.
* `pallas/dma.py`: ring resource.
* `pallas/tracing_ops.py`
  * `_codegen_fori_loop`: ring branch with its own slice builder, not
    `_build_dma_slices`.
  * `_classify_pipelined_tensors`: exempt streamed tensors.
* Prologue emission at kernel start.
* `pallas_stream_depth` in `config_spec.py`, `runtime/config.py` and the
  backend's `supports_config_key`.
