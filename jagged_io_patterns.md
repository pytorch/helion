# Jagged / ragged kernel I/O patterns

Survey of the I/O shapes used by `examples/jagged_*.py` and `examples/segment_reduction.py`.
All concrete sizes come from each file's `main()`.

The jagged dim is always dim 0 (leftmost) in the concatenated layout.

In the "padded" column, `<=` marks the variable / padded dim — i.e. the dim
whose true per-row size is between 1 and the listed bound, and the concatenated
buffer stores only the valid entries (sum = L) rather than the padded total.

## Shape table

| kernel | B | max_seq_len | trailing dims | padded `[B, S, …]` | concat `[L, …]` | offsets | output |
|---|---|---|---|---|---|---|---|
| `jagged_sum` | 8 | 64 | M = 128 | `[8, <=64, 128]` | `[L, 128]` | `[9]` | `[8, 128]` dense |
| `jagged_mean` | 32 | 64 | max_M = 8 | `[32, <=64, 8]` | `[L, 8]` | `[33]` | `[32, 8]` dense |
| `jagged_softmax` | 512 | 64 | M = 128 | `[512, <=64, 128]` | `[L, 128]` | `[513]` | `[L, 128]` jagged |
| `jagged_layer_norm` | swept ¹ | 128 | M (swept) ¹ | `[B, <=128, M]` | `[L, M]` | `[B+1]` | `[L, M]` jagged |
| `jagged_dense_bmm` | 23 | 37 | D = 34 (in), K = 24 (out) | `[23, <=37, 34]` | `[L, 34]` | `[24]` | `[L, 24]` jagged |
| `jagged_hstu_attn` | 1024 | 1024 | H = 4, Dh = 128 | `[1024, <=1024, 4, 128]` | `[L, 4, 128]` | `[1025]` | `[L, 4, 128]` jagged |
| `jagged_dense_add` | 256 | 5000 | — (no trailing feature dim) | `[256, <=5000]` 2-D | `[nnz]` **1-D** | `[257]` | `[256, 5000]` dense (= `y`) |
| `segment_reduction` | — ² | — ² | F = 128 | n/a (never padded) | `[2000, 128]` | n/a — `indices: [2000]` | `[100, 128]` dense |

¹ `jagged_layer_norm` `main()` sweeps `B ∈ {32, 256, 2048, 16384}` and
  `M ∈ {32, 256}` via `itertools.product`, with `max_seqlen` fixed at 128.

² `segment_reduction` has no `B` or `max_seq_len` in its API. It takes
  `input_data: [E, F]`, `indices: [E]` (sorted segment IDs), and `num_nodes`.
  Segment boundaries are inferred at runtime from `indices`, not declared via
  offsets. The closest analog to B is `num_nodes = 100`, but that's the
  *output* bucket count, not a sequence count; any single segment can have
  size 0..E.

## Notes

- `L` is the *sum* of per-row lengths, not `B × max_seq_len`. Per-row lengths
  are random in `[1, max_seq_len]`, so `L ≤ B · max_seq_len`, usually strictly
  less.
- Output is **dense `[B, …]`** only for the reducers (`jagged_sum`,
  `jagged_mean`). For per-element / per-row ops (`jagged_softmax`,
  `jagged_layer_norm`, `jagged_dense_bmm`, `jagged_hstu_attn`) the output stays
  jagged and is written back into a packed buffer keyed by the same offsets.
- `jagged_dense_add` is the odd one shape-wise: the concatenated buffer is
  truly 1-D (`[nnz]`), and the dense output's column count `N` is independent
  of `max_seq_len` (it only needs `N ≥ max per-row length`; `main()` happens to
  set both to 5000).
- `jagged_hstu_attn` is the only kernel with two trailing dense dims (`H, Dh`).
- `segment_reduction` does not match the offsets-based jagged shape pattern at
  all — it's a sorted-segment-id + scatter primitive.

## Hardware evaluation

Grouping by I/O structure (which also lines up with TPU placement):

### Group 1 — flat-buffer + `jagged_tile`, regular per-row work

Kernels: `jagged_sum`, `jagged_mean`, `jagged_softmax`, `jagged_layer_norm`,
`jagged_dense_bmm`.

Shared idiom: outer tile over `B`, inner `for tile_k in hl.jagged_tile(seq_len)`,
loads from a flat (`.view(-1)`) buffer with
`(starts[:, None] + tile_k.index[None, :])[:, :, None] * stride + tile_inner.index`.
The five kernels differ only in:
- number of passes over the jagged loop (1 for sum/mean, 2 for softmax,
  3 for layer_norm, 1 with inner matmul for bmm)
- whether the output is dense `[B, M]` (sum, mean) or jagged `[L, …]`
  (softmax, layer_norm, bmm)
- `jagged_mean` also uses `hl.jagged_tile(feature_counts)` for the feature
  dim → doubly jagged
- `jagged_dense_bmm` adds an inner matmul against a per-batch dense weight

TPU placement: TensorCore. The work inside each row is regular (reduction or
matmul). The TPU/Pallas pain point here is `hl.jagged_tile`'s dynamic trip
count, not the inner op. Fixing the lowering once should cover all five.

### Group 2 — masked-tile, jagged-in / dense-out

Kernel: `jagged_dense_add`.

Outer tile over `B`, inner `for tile1 in hl.tile(0, max_nnz)` where
`max_nnz = nnz.amax()` (runtime-bounded regular tile, **not** `jagged_tile`),
plus `extra_mask=tile1.index < nnz[:, None]`. Mixes jagged input with a dense
operand `y` and writes to a dense output.

TPU placement: borderline. The input side is irregular (scatter-like reads),
the output side is dense. Could fit either core depending on how the
masked-load is lowered.

### Group 3 — direct-indexed attention

Kernel: `jagged_hstu_attn`.

No flattening, no `jagged_tile`. Tiles over `[B, H, max_seq_len]` with
block_size `[1, 1, None]` — one program per (batch, head, q-block). Indexes
the 3-D `q/k/v` tensors directly at row positions `tile_q.index + starts`
with a runtime mask `tile_q.index < seq_len`. Causal mask is the usual
`q_idx > kv_idx`.

TPU placement: TensorCore. The inner work is matmul; raggedness is a
masking / early-exit concern, not a structural one.

### Group 4 — sorted-segment-id scatter

Kernel: `segment_reduction`.

Different op entirely. No offsets, no `jagged_tile`. Dense `[E, F]` reads
plus a sorted `indices[E]` segment-id vector. Uses `hl.associative_scan`
to collapse runs of equal IDs within a tile, then `hl.atomic_add` to
scatter the run totals into the dense `[num_segments, F]` output at
segment boundaries.

TPU placement: SparseCore. Data-dependent destination addresses, irregular
access, atomic accumulation. This is the canonical SC workload and
probably warrants a separate backend dispatch rather than being lowered
through the same Pallas path as Groups 1–3.

## Per-group lowering discussion

Constraints in effect (saved to project memory): no SparseCore work this round,
and the lowering has to be general (no per-kernel branching, no rewriting the
Helion source — GPU lowering must keep working).

### Group 1 — `hl.jagged_tile`, TC

**Why all five kernels use `hl.jagged_tile`.** It's the convenience macro for
"iterate one variable-length inner dim and emit the per-lane mask for me."
Group 1's inner loops are clean jagged-only loops, so the sugar applies.
(Group 2's `jagged_dense_add` doesn't use it because its inner column iteration
is jagged-then-dense in sequence: the first loop runs `0..max_nnz` and the
second runs `max_nnz..N` filling the dense tail. You can't easily extract
`max_nnz` from `hl.jagged_tile`, so the author uses the manual `amax + tile +
extra_mask` form. We can confirm with Meta but the reasoning is visible from
the code.)

**Current lowering is an amax-bounded loop + mask — but with a caveat on TPU.**
From `helion/language/loops.py:594-608`, `hl.jagged_tile(parent)` lowers to:

```python
end = parent.amax()
for tile_k in hl.tile(end):
    mask = tile_k.index[None, :] < parent[:, None]
    ...
```

This amax-bounded dense loop + mask is handled natively by Triton. On TPU the
same lowering goes through Pallas, but with **two problems**:

1. **Amax-padding within the parent tile.** When the parent tile has
   block_size > 1, `parent.amax()` is the max over all sequences in that tile,
   and every lane runs that many iterations with masking. Sequences shorter
   than the amax do wasted masked work. On GPU this is cheap; on TPU/Mosaic
   the masked sublanes still flow through the MXU/VPU.

2. **Scatter-form writeback.** Jagged-output kernels (softmax, layer_norm,
   bmm) emit `hl.store(out, [flat_indices], …)` where
   `flat_indices = (starts + tile_k.index) * stride + tile_m.index`. Each
   output element has an explicitly computed int address. The scatter form
   obscures any underlying contiguity from the backend. Mosaic prefers
   slice-based stores (`pl.ds(start, sz)`); arbitrary scatters fall back to
   slow paths.

**Both problems are addressed by the compile-flow plan below, not by a single
knob.** The L-space transformation (change (3)) eliminates the N (item) axis
entirely — there is no parent tile, so the amax-over-lanes padding simply does
not arise; the grid iterates fixed-size L-blocks. The scatter-form writeback is
resolved by the indexing-classification change (change (1)): once the access
is recognized as a per-axis `pl.ds` slice rather than a computed tensor index,
the store lowers as a slice, not a scatter. Earlier framing pinned this on
"parent-tile block_size = 1" — that helps locally (a scalar `starts`, a scalar
trip count) but it is not the fix; `block_size > 1` is not structurally broken
on TPU, it just does not buy throughput, and the L-space form makes the
question moot.

**Memory-hierarchy mapping (the careful version):**

Pallas explicitly manages three layers (HBM workload → VMEM blocks → VREG
vectorization). Helion's source primitives only name **two** of them, and
the mapping to Triton confirms which one is which.

Reading `helion/_compiler/tile_strategy.py:codegen_grid` /
`codegen_device_loop`: a top-level `hl.tile` becomes a grid axis with
`offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`; an inner `hl.tile`
becomes a Python for-loop with the same arange inside. In both cases
`block_size` is Triton's `BLOCK_SIZE` constexpr, and the per-tile data is a
register vector. Global loads (`x[tile]`) go DRAM-to-register directly;
Triton has no user-visible SRAM primitive.

So:

| layer | Pallas | Triton | Helion source |
|---|---|---|---|
| DRAM (HBM / GMEM) | `jax.Array` refs | global pointers | source tensors (`x_data`, `x_offsets`) — *not* a tile |
| SRAM (VMEM / SMEM) | **explicit** — BlockSpec / `scratch_shapes` VMEM allocs | **implicit** — compiler decides | **not represented** in the surface language |
| compute (VREG / registers) | VMEM↔VREG crossed by `load_p` / `swap_p` (`pl.load` / `pl.store`, = `ref[idx]`) — the **only** memory primitives Pallas exposes; VREG residency itself is LLO-managed | `BLOCK_SIZE` register vector | **`tile.block_size` lives here** — same as Triton's `BLOCK_SIZE` |

**On the Pallas VMEM↔VREG boundary**:

- Pallas exposes exactly one pair of memory primitives — `load_p` / `swap_p`,
  surfaced as `pl.load` / `pl.store`. Plain `ref[idx]` indexing binds the *same*
  primitives: `pl.load`'s docstring says "if neither `mask` nor `other` is
  specified, this has the same semantics as `x_ref_or_view[idx]`." So unmasked
  access auto-binds load/store — you only reach for the explicit `pl.load` /
  `pl.store` call to pass a `mask`.
- Lowering:
  - No-stride → MLIR `vector.load` / `tpu.vector_store`;
  - Strided (`pl.ds(start, size, step)`) → `tpu.strided_load` / `tpu.strided_store`;
- Mask support is **asymmetric**:
  - **Masked store works** — `pl.store(…, mask=mask)` lowers to
    `tpu.vector_store(…, mask=mask)`, a genuine hardware-predicated store:
    masked-off lanes are *not written*, no read-modify-write. Limited to
    32-bit data, non-strided.
  - **Masked load does not exist.** `_load_lowering_rule` does
    `if mask is not None: raise NotImplementedError` — `pl.load(mask=…)` does
    not lower on TPU at all. **Mosaic always loads the full tile.** If the dev
    wants masked-load semantics they must write it themselves: `ref[idx]`
    (full load) + `jnp.where(mask, val, other)`. Helion's Pallas backend does
    exactly this on the dev's behalf (PR #2214 multiplies the loaded slice by
    the mask) — so `hl.load(extra_mask=…)` works at the Helion level, but the
    masking is the framework's load-then-`where`, *not* a masked-load
    primitive.
  - This costs nothing on TPU: the VPU is tile-granular, so the full tile is
    read regardless; the mask is just one cheap `where`. And it is always
    memory-safe — the VMEM block is fully allocated, so reading all of it
    never faults (unlike an out-of-bounds *global* load on GPU, where a real
    masked load is needed for safety). The masked-*store* primitive exists
    because a store has a side effect a load doesn't — "store full" would
    clobber values that should be preserved.

Key consequences of this asymmetry:

- **`hl.tile` is the compute-granule layer, not the SRAM layer.** It maps
  1:1 to Triton's register-vector `BLOCK_SIZE`. On GPU this works because
  Triton conflates SRAM and registers in its programming model. On TPU it's
  a gap: Pallas separates VMEM and VREG explicitly, but Helion has no
  primitive at the VMEM-block level. The TPU backend has to bridge —
  either by treating `hl.tile.block_size` as both the VMEM block size *and*
  the vectorization granule (letting Mosaic vectorize below), or by
  introducing a compiler-internal VMEM-block notion distinct from
  `tile.block_size`.

- **`hl.jagged_tile` is at the same layer as `hl.tile`.** It is *not* a
  separate memory tier. It's the same compute-granule loop primitive plus
  two added behaviors: (a) data-dependent trip count via `parent.amax()`,
  and (b) implicit per-lane mask on loads/stores at that granule. The
  masking is an operation at the compute granule, not a layer below it.
  Storage doesn't grow; compute does (the masked-out iterations execute).

- **Pallas's VMEM block size does not have a true 1:1 in Helion.** A Pallas
  kernel sizes its VMEM blocks explicitly (BlockSpec block shape /
  `scratch_shapes`); that knob looks like `tile_k.block_size`, but they live
  at different layers. They coincide on GPU because the SRAM layer is implicit
  there; on TPU they need not be the same number.

**Helion-specific extras that sit on top of the layer model:**

- **amax-padding** (from `hl.jagged_tile`'s `end = parent.amax()`, when
  parent tile block_size > 1): not a memory tier — it widens the *trip
  count* of the compute-granule loop so all lanes within one program align
  to the longest lane. The L-space form (change (3)) has no parent tile, so
  this padding does not arise at all.

- **Scatter-form writeback** (current `hl.store(out, [flat_indices], …)`
  pattern): an operation choice at the compute granule, not a layer. Once the
  access is classified as a per-axis `pl.ds` slice (change (1)) rather than a
  computed tensor index, the store lowers as a slice, not a scatter.

## Compile-flow plan: three coupled changes

Supporting `hl.jagged_tile` on Pallas decomposes into three changes in the
compile flow. They are ordered **(3) → (1) → (2)**: the IR representation
comes first because the other two operate on it.

### (3) L-space IR node — `RaggedTileInfo`

Today the frontend produces a `JaggedTileIndexType` — an inner jagged loop,
nested under parent item-tiles (N-space). The Pallas-friendly form is L-space:
a flat iteration over the concatenated `L` dim, with the offsets table carried
so item-membership is derivable.

**Form.** `JaggedTileIndexType` is a `TypeInfo` *only because* `hl.jagged_tile`
is a source primitive — its type-propagation handler creates it. There is no
`hl.ragged_tile` source primitive (we picked the compiler-pass route, not a
source rewrite), and the loop-fusion pass runs post-type-propagation on the
device-IR graphs (like `plan_tiling` does at `backend.pre_codegen`). So the
L-space node is **not** a `TypeInfo` — it is an **env-registry entry keyed by
block_id**, mirroring `jagged_tile_parent_ids`: a `RaggedTileInfo` dataclass
(offsets-tensor reference + the collapsed-from block_ids + feature/`M` info),
plus `env.register_ragged_tile()` / `env.is_ragged_tile()` / `env.ragged_tile_info`.

The loop-fusion / grid-promotion pass *produces* a fused L-space block_id by
collapsing the `tile_b (TileIndexType) → hl.jagged_tile(nnz) (JaggedTileIndexType)`
nest, and registers it as a `RaggedTileInfo`.

The pass owns **two** rewrites, not one:

1. **The loop nest** — collapse `tile_b → hl.jagged_tile(nnz)` into a single
   L-space iteration over the fused (ragged) block_id.
2. **The load/store subscripts** — rewrite `x_flat[flat_indices]` (the flat
   `.view(-1)` + computed-tensor-index form) into a clean ragged-tile-indexed
   access, e.g. `x_data[ragged_tile, col_tile]`. The flat-stride address
   arithmetic does not survive the pass.

The point of registering the L-space form is that the Pallas backend then
**checks `env.is_ragged_tile(block_id)` directly** — it never has to
reverse-engineer a flattened `a*x + b` expression. Because the pass rewrites
the subscripts too, change (1) sees a ragged block_id, not a computed index
tensor.

First, because (1) and (2) both operate on this representation. Note this is
*not* a leak: the `RaggedTileInfo` registry is the substrate (1) and (2) work
on — it is meant to survive into them and is only consumed at final codegen.
What the pass dissolves is the `JaggedTileIndexType`-under-`TileIndexType`
N-space nest, not the `RaggedTileInfo` itself.

### (1) Indexing classification — `TilePattern`, not `TensorIndexPattern`

Today the jagged load `hl.load(x_flat, [flat_indices])` carries a computed
tensor index (from `.view(-1)` + offset arithmetic). The Pallas `plan_tiling`
pass classifies it as `TensorIndexPattern` → `IndirectGatherPattern` →
`one_hot(idx, V) @ table` (catastrophic for a reduction; the load path is also
guarded by a 16 MiB VMEM threshold). Stores with a tensor index are rejected
outright at `plan_tiling` time — so the jagged-output kernels can't even
compile.

Fix: because change (3)'s pass already rewrote the subscript into a
ragged-tile-indexed access, change (1) is **registry-driven, not
arithmetic-driven** — extend `_detect_indexing_pattern` to recognize a block_id
for which `env.is_ragged_tile(block_id)` holds and classify it as a
`TilePattern` with a data-dependent bound, exactly analogous to how it already
recognizes a `TileIndexType` block_id today. No `a*x + b` pattern-matching, no
un-flattening at this layer — that work belongs to (3)'s subscript rewrite. The
dim then
lowers via the existing `pl.ds` machinery — per-axis slices — and never touches
`gather.py`. This also unblocks the store side, since `pl.ds`-shaped stores are
not rejected the way tensor-index scatters are.

Second, because it needs (3): it classifies the ragged block_id that (3)
produces and registers.

### (2) Loop construct — `fori_loop` / `unroll`, not `emit_pipeline`

`pallas_loop_type` defaults to `emit_pipeline`, an HBM↔VMEM streaming construct
(double-buffering, DMA scheduling). For a jagged inner loop whose data is
already VMEM-resident (staged once by the grid BlockSpec), `emit_pipeline` is
the wrong wrapper — its DMA machinery has nothing to schedule.

Fix: select `fori_loop` (or `unroll`) for the jagged inner loop, with every
tensor marked non-pipelined (the per-tensor pipelining decision, PR #2093).
The loop body is then just `pl.ds` slices + compute — a pure VMEM→VREG loop,
no DMA APIs. The construct already exists; the backend has to be taught to
pick it for jagged inner loops instead of the `emit_pipeline` default.

Last, because it only matters once (1) makes the access a `pl.ds` slice —
then (2) chooses the loop wrapper around those slices.

### Downstream lowering: `hl.atomic_add` for L-space writeback

Change (3)'s pass emits `hl.atomic_add(out, [item_idx, m_tile], partial)` in
the per-item inner body. The atomic is structurally required, not a
performance hedge: when an item's rows are split across multiple L-blocks,
those L-blocks execute as separate grid programs and concurrently update
`out[item_idx, m_tile]`. The read-modify-write must serialize; the order of
the partial sums does not matter (sum is commutative/associative).

Single-threadedness inside one program does not help here — the race is
across programs in the Mosaic grid.

Lowering contract the Pallas-side atomic_add codegen can assume:

- `target` is a 2-D HBM-resident output (`out[num_rows, M]` for
  `jagged_sum` / `jagged_mean`).
- `index` is `[scalar_item_idx, m_tile_slice]` — scalar in dim 0, slice
  (`pl.ds`-shaped) in dim 1.
- `value` is a 1-D `(BLOCK_M,)` register vector — the per-item partial sum
  for this L-block and this M-tile.
- Memory semantics: `"relaxed"`. No ordering requirements between programs;
  the final value at each `out[item_idx, m_tile]` element is the sum of all
  partials contributed by any grid program.

The per-lane mask is already folded into `partial` (masked elements are 0
from the prior `hl.where`), so the atomic_add itself does **not** need an
additional lane mask — it adds the full `(BLOCK_M,)` vector.

This is a separate lowering work-item; change (3) emits the FX node, and
this section documents the contract a coworker is implementing for Pallas
codegen. Until that lowering lands, kernels using the L-space form will
fail at codegen rather than at IR construction.

### What is *not* needed

- **No new VMEM↔VREG loop construct** — `fori_loop` with all tensors
  non-pipelined already is one.
- **No change to `hl.jagged_tile`'s loop machinery** — it correctly reuses
  the shared `TileStrategy` path; the only jagged-specific bit
  (`_setup_mask`'s per-lane `index < parent` mask) stays.
- **BlockSpec at the grid level stays** — something has to stage HBM→VMEM,
  and grid-level BlockSpec is the right tool. (Caveat: BlockSpec needs static
  block shapes; the variable per-item size may force an explicit `async_copy`
  instead — still the grid-level 1↔2 step.)

## Remaining caveats for Group 1

- **`jagged_mean`'s nested `hl.jagged_tile`.** The feature dim is also
  jagged via `hl.jagged_tile(feature_counts)`. The lowering needs to compose
  correctly when one jagged loop is nested in another. Worth a separate
  sanity check.
- **Lane underutilization when most-minor < 128.** `jagged_mean` has M=8 →
  only 8/128 lanes used. Fix is workspace-side lane padding inside the
  lowering (allocate a `[BLOCK, 128]` workspace tile, mask the unused
  lanes) — independent of the three compile-flow changes.
- **Load imbalance from skewed `seq_len`.** With one L-block per program,
  wall time is set by the longest. Inherent; mitigation would be host-side
  sort-by-length, but that's out of scope for the lowering itself.
