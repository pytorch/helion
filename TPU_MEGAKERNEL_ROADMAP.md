# Roadmap: Qwen3.8 decode as one Helion-compiled TPU megakernel

Living document. It is updated at every milestone and whenever a measurement
changes the plan. The design rationale is in `TPU_MEGAKERNEL_PLAN.md`; this file
covers direction, gates and status.

## North star

Compile the full Qwen3.8-27B TP8 decode step from ordinary Helion source into
one Pallas program per TPU v7 core, and match the hand-written megakernel
(`~/tpu-megakernels/qwen/decode_megakernel.py`, "the reference") in ms/token on
the same 8 cores. The source covers embedding, 64 hybrid layers (3 gated-delta-net
: 1 GQA), MLP, final norm, LM head, argmax and TP all-reduce.

**Generality is the constraint, not a nice-to-have.** Other models will be
tested. Every optimization must:

* be an analysis or transformation on Helion device IR, keyed on generic
  properties: read-only, written-once, affine slice, loop structure, live
  range. It must never key on op names, model names or specific shapes;
* be exercised by at least two structurally different kernels in
  `test/test_pallas_megakernel.py` before it counts as done;
* come with no reference code copied into Helion. Hand-written targets
  (`target_*.py`) live outside the repo and are used only to measure what is
  achievable.

## Where the time goes (reference, measured on our pod)

Measured with `tpu-megakernel-scripts/bench_qwen_ref.py`: random weights,
context 2048, position 100, 8 cores, TP8. The pod has jax 0.10, while the
reference pins 0.11.1, so the pod copy patches only I/O layouts:

* small per-layer params are stored f32 or lane-padded;
* conv state is stored transposed;
* the argmax pair buffers are padded.

The kernel structure is unchanged.

| Config | ms/token | Achieved weight BW | Notes |
|---|---|---|---|
| Reference, 8 layers + LM head | 0.431 | 2.69 TB/s | 1.156 GB/rank streamed |
| Reference, 8 layers, `noaux` | 0.432 | 2.67 TB/s | aux param DMAs are free |
| **Reference, 64 layers, full TP8** | **2.330** | **2.99 TB/s** | 6.96 GB/rank (padded); floor 1.88 ms at 3.7 TB/s |
| **Reference, 64 layers, `noexchange`** | **2.290** | 3.04 TB/s | TP all-reduce costs only 2% |
| XLA per-rank, 64 layers + LM head + argmax, no TP | 4.117 | 1.57 TB/s | 6.45 GB (unpadded); `bench_xla_layers.py` |
| XLA per-rank, 8 layers + LM head | 0.600 | 1.81 TB/s | |
| XLA per-rank, 1 GQA layer + MLP | 0.0679 | 1.41 TB/s | 95.7 MB |
| XLA per-rank, 1 GDN layer + MLP | 0.0687 | 1.39 TB/s | 95.8 MB |

The reference is 1.8× faster than XLA at the same per-rank work. Its per-layer
cost is about 34 µs: 2.29 ms / 64, including the LM head share. XLA's is about
68 µs.

Per-rank bytes per token are about 96% streamed weights:

* 48 × (qkvz 26.2 + outw 7.9 + MLP 70.8) MB;
* 16 × (qwkv 21 + ow 7.9 + MLP 70.8) MB;
* LM head 328 MB.

States and KV are about 40 MB. The step is a weight stream with compute and 128
all-reduces hidden underneath it. **Everything below serves "the stream never
stalls".**

## Generic compiler features (what we are building)

Each feature has an ID used in the milestones. "Plan §" points to the design.

| ID | Feature | Generic rule | Plan § |
|---|---|---|---|
| F1 | Sequential roots, VMEM-resident intermediates | Body-allocated, non-returned tensor written by root p and read by root c ≥ p → scratch | 3.3, 3.4 |
| F2 | **Global weight ring** | All read-only, affine, block-aligned loads in program order form one static DMA sequence through one ring of max-size slots. Refill g+D decodes into later sites. | 3.5 |
| F3 | Per-shape rings and stream tile selection | One ring per distinct tile shape, each with depth budgeted in bytes and proportional to its traffic (the DMA probe shows bandwidth depends on slot bytes, with the knee at ~256 KB). Default tiles are ≥ 512 KB with N ≥ 256 or the full dim; the autotuner searches around them, non-pow2 included. | 3.5 + F3 revision |
| F4 | Local (dynamic) streams | Loads with data-dependent trip count or address (KV cache up to `pos`, expert weights, embedding row) get their own small multi-buffered stream. The global ring keeps flowing underneath; its depth must cover the longest gap. | new |
| F5 | Folded host loops (layer loop) | A host `for i in range(L)` around roots becomes `pl.loop(0, L)`. The stream index is affine in i: `g = i·T_body + T_site + …`. `hl.static_range` loops are unrolled. Branches on loop variables (`if i % 4 == 3`) become `pl.when`/`lax.cond`, with the stream schedule computed per static period. | 1.5 (D4) |
| F6 | Cross-iteration prefetch of small read-only params | A slice `w[i]` of a stacked `[L, …]` read-only tensor in a folded loop gets a per-load slot (×2 if needed). Iteration i+1's slice is issued as soon as slot i's last reader is done (WAR). | A2 |
| F7 | State prefetch and write-behind | A read-modify-write slice `s[i]` (recurrent state, conv state, KV row) is read early (after the last in-kernel writer of that slice, at the latest one root ahead) and written back async. The wait sits before buffer reuse (next iteration) or at exit. | 3.6 (D2) |
| F8 | VMEM liveness pool | Interval coloring of scratch, local-stream and state buffers. The ring gets what is left. | 3.6 (A3) |
| F9 | Scalar operands | Scalars read from small int tensors (token, position) are lowered to SMEM/scalar prefetch and usable as indices, loop bounds and masks. | new |
| F10 | In-kernel collectives | TP all-reduce over remote DMA, expressed with the existing Helion distributed API if there is one (under investigation). The compiler overlaps the ring with the collective because the ring is pre-issued. | 1 (D3, B5) |
| F11 | Small-M matmul lowering | M < sublane count: pad LHS rows (broadcast) and use `preferred_element_type=f32`. | 5 notes |
| F12 | **Stream tile shape freedom** | Pallas block sizes may be non-powers of two when they are hardware-tile multiples or the full dim (2176, 1536, 5120). F3 then picks large contiguous slabs. Without it the M3 GQA layer streams through (128,128) slots at 1.1 TB/s. | new |
| F13 | **Compute-root efficiency** | Every non-streaming root must finish within the ring's buffered time (~4.5 µs at 16 MiB). Generic codegen work: 2-D reductions, unit batch-dim squeeze, no spills (loop over heads instead of whole-state values), hoisted masks. Done so far: launcher whole-block staging, `put_full_operand_first` (GDN core 69 → 3.3 µs). | new |
| F14 | Local streams in the global DMA order | Generated code today primes a local stream (e.g. `_prime_fori_loads` for the KV cache) at the head of its loop. By then the ring has ~16 MiB of later sites' weights in flight, so the first KV tile waits ~5 µs; forcing the cache to HBM costs +7.8 µs on the GQA layer. Rule: issue the prime (the first `min(slots, trips)` tiles; addresses depend only on scalars) just before the first ring refill whose decoded site comes after the loop in program order. This is its place in program-ordered DMA order. RAW on the row written in this kernel is already handled by forwarding into the slot. Then HBM state costs what VMEM-whole state does, which stacks need, since per-layer caches and states don't all fit in VMEM. **✅ Done (with F15):** `_plan_early_copies` starts each whole-slice state load and each local-stream prime of a tensor kept in HBM, per folded iteration, just before the first ring copy of a stream index read after it. Conditions: the address depends only on constants, the layer and scalars; the destination is free (ring copies started before that are passed over); and every store to the source in between is disjoint by an integer index or is a forwarded row write. Otherwise the copy starts where the program reaches it. GQA layer with caches in HBM: 32.7 µs body, VMEM-whole 33.0. | new |
| F15 | **Per-layer state in HBM** | A written tensor whose root-level accesses are whole slices at scalar leading indices (constants, scalars, or affine in the folded layer loop), plus inner-loop tile streams, stays in HBM. A load is a DMA into a VMEM slot of its own plus a wait. A store is an async DMA kept as a pending write: waited at the end of its root, or, for a single-tile root, at the end of the next root when that root does not use the tensor. A ragged minor dim (e.g. conv state `[L, C, 4]`) copies through a `[rows, cols]` view. Default: with rings, HBM when the slots and stages take less VMEM than the whole tensors, so the rings get the rest. **✅ Done:** hybrid stack 67.7 → 31.2 µs/layer at L=16; L=32/64 compile and run at 30.9/32.7. | new |
| F16 | **Depth-first stream tiles and ring budget** | Weight-DMA bandwidth follows Little's law. During a root only its own ring has copies in flight, because every other ring sits full of prefetched tiles. So a root streams at about `min(3.35 TB/s, (D − loads)·slot / λ)` with λ ≈ 1.25 µs, and saturates at about 6–7 MB in flight per ring (MLP-stack sweep in the 2026-10-02 log). Two policy changes follow. (a) **Seed:** among tiles that fit, are wide and fill `_STREAM_MIN_SLOT_BYTES`, prefer contiguous (whole minor dim, which also lets same-N sites share a slot) and then the **smallest** slot, so a ring of the same bytes has more slots in flight. Today's seed prefers the largest tiles, which gives depth-2 rings whenever VMEM is ample. (b) **Depths:** give each ring bytes for the time it saves, i.e. greedy marginal allocation with the bandwidth model above, weighted by traffic per period. A ring holding a whole root's traffic makes that root free; saturated rings get nothing more. Today every ring gets about the primary's bytes regardless of traffic. | ✅ (a)+(b) in main: hybrid L=8 38.2 → 33.0 µs/layer (2.90 TB/s), L=4 40.2 → 33.3; MLP stacks neutral |
| F17 | Narrow-minor-dim (ragged) rings | Tiles whose minor dim is far below 128 lanes (GDN `b`/`a` (5120, 6), conv (·, 4)) take 128-lane-padded VMEM: 7.2 MB for ~0.3 MB of weights at L=8. Budget them in padded bytes at minimum depth, or stream them through a lane-dense view when the layout allows it. | ✅ minimum depth (`_ring_floor`); F17b lane-dense operands merged into main 2026-10-03 (`pallas_lane_dense`, report /tmp/mk/f17b/REPORT.md). L=8 body 363.7→358.1 µs, and 5 layout copies (16.8 µs per call) are gone. L=64 body 2163.7→2145.0. ✅ F17c (any XLA default-layout permutation, merged 2026-10-03, /tmp/mk/f17c/REPORT.md): TP8 L=64 device 2356.9 → 2243.5 µs/call (the `[48,5120,6]` relayouts, 2 × 38.5 µs, are gone), body 2209.8 → 2171.7; SC step body 1928.2 → 1896.5, hybrid 1834.2 → 1799.3. Open: (0,2,1) stacks streamed per layer don't compile on TPU (6-row second-minor slice), bf16 rows copy two rows' words |
| F18 | Shared ring arena / global DMA schedule | The long-term fix for ring fragmentation. All rings draw slots from one VMEM arena, so the bytes in flight are the whole arena whichever root runs. It needs a layout every slot shape can view, which Mosaic only allows over leading dims, so first find out what reinterpretation Mosaic supports. | ✅ default on since 2026-10-03 (`_ARENA_BY_DEFAULT = True`; knob `pallas_stream_arena`, raw views via `helion/runtime/pallas/ring_view.py`, seed tiles ≤ 2.5 MiB). **TP8 L=64: body 2617.3 → 2209.8 µs, 2356.9 µs/token device (reference 2330), exposed exchange 7.9 → 4.4 µs/layer.** Single core L=64: step body 2044.5 → 1928.2 µs (device 2063.2 → 1947.8), hybrid 1899.8 → 1834.2 µs (3.34 TB/s). Needs arena and big tiles together. Tests of per-weight rings pin them with `_per_weight_rings()` |
| F19 | Program-order prologue | The prologue primes every ring at once (about 40 MB), so the first root waits about 10 µs for its tiles behind everyone else's. Prime in global program order: the first rings now, later rings at an earlier ring's refill, like the existing ring handoff. That costs 10 µs per call, 0.16 µs/layer at L=64 but 1.25 µs/layer at L=8. | ⬜ |
| F20-0 | Double-buffered indirect inner loads | Read-only tile load in an inner loop whose scalar subscript is not affine in folded loops (expert id read from a tensor). Today these go through the ordinary per-loop DMA path with `pallas_load_buffer_count=1`, i.e. a synchronous wait on every tile. Default to 2, same rule as HBM-resident outputs. Interim baseline for MoE. | ✅ M8a (`is_indexed_weight_load`) |
| F20a | **Index-vector ring sites** | A scalar subscript that is the value of a scalar load from a read-only integer tensor X, whose own subscripts are constants or folded-loop affine (`e = sel_idx[0, k]; w_gate[e, tk, tn]`). The load joins the global ring at its program-order slot; refill code reads `X[sub]` at issue time and clamps it to the dim. Same-shape static tiles (shared expert) share slots automatically because ring keys drop scalar dims. Also removes the "kept whole in VMEM" budget error for root-level expert reads. | ✅ M8a; X ≤ 256 elements in VMEM is mirrored into SMEM (`IndexMirror`) |
| F20b | Refills gated on an in-kernel producer | X is written by root P earlier in the same folded iteration (in-kernel router + top-k). A refill of an X-addressed site that would issue before P ends (prologue prime, cross-iteration refill, earlier consumer) moves to a catch-up batch at P's last tile. It is free when ≥ D static tiles sit between P and the first expert site (shared expert placed after top-k), and `_simulate_refill` measures the bubble otherwise. Next-layer expert refills never read this layer's X. | ✅ M8a (`RingGate`, catch-up batch at the producer's last tile) |
| F20c | Element-granular early issue | P is a single-tile root whose unrolled `static_range` iteration k stores X at constant subscripts. Issue the held refills for slot k right after that store, i.e. during the remaining selections. Worth 1–2 µs/layer only when no static tiles cover routing; build only if measured. | ⬜ (conditional) |
| F21 | Device-side trip counts in folded loops | A folded loop `for j in range(Nmax)` whose roots are all guarded by `j < n` with n read on device (distinct experts at B > 1, local route count under EP). It becomes its own ring primed and refilled under `pl.when(g < n)`; the static ring's indices skip the segment. The guard must lower to `pl.when` around roots, not `lax.cond` with synchronous copies inside. Generalizes F4. | ⬜ |
| F22 | VMEM accounting | The rings get `capacity − reserve − resident − inner-loop buffers − loop-carried scratch`. The loop-carried scratch is sized per config from the block-size symbols and shared across roots like `share_root_scratch` (per (shape, dtype), the max count over roots). The reserve, which covers only what the plan does not size (Mosaic internal scratch and spills), is 3 MiB instead of 8. | ✅ in main; TPU: step L=64 body −27 µs (2127.2), hybrid L=32 fits (989.9 µs). `_WHOLE_RESERVE_BYTES` (8 MiB) keeps the config-independent whole-in-VMEM choice conservative |
| F23 | Remote copies in DMA order | A remote copy (TP exchange) queues behind every ring copy issued before it, so deeper rings make the exchange slower (M7: 4.85 → 5.99 µs). Issue it ahead of the trailing refills of the root that produces its source, or per producer tile, and keep the ring flowing during the wait. | ✅ M7b (merged 2026-10-03): held copies start right behind the last remote start; bubble-aware holds (12 MiB, earliest-freed prefix). Hybrid exposed exchange 8.3 → 6.7 µs/layer; the rest needs F18 |
| F22b | Tile overhang in the ring budget | A tensor kept whole in VMEM is allocated to the extent its tiles reach (`_ref_extents`: rows padded to the tile, +1 block for noncanonical starts), not to its shape. `TileOverhang` rebuilds that extent at each config's block sizes and the rings budget the excess (× launcher copies). | ✅ main worktree (unit tested; TP8 L=8: 2.5 MiB that caused the 64.59/64 MiB OOM) |
| F24 | Ragged tiled minor dims in rings | A tiled weight dim whose size is not a multiple of 128 (MLA `w_kv [L, 2048, 576]`) is rejected by `_stream_load`, so each tile becomes a synchronous copy and the launcher pads 576 → 640 on the host every call. Rule: a tiled dim whose block can be the full size is streamed like a whole narrow minor dim (the F17 ring path); its search space is the full size or 128-aligned divisors. Same for a second-minor dim the smallest block does not divide. | ✅ merged into main 2026-10-03 (search limited to the full size, `StreamModel.whole_block_sizes`). TPU: MLA L=8 59.74 → 53.43 µs/layer (2.72 → 3.05 TB/s), kv-down root 9.43 → 0.84 µs, host pad of `w_kv` (7.6 µs/call) gone; ragged repro N=576 149.25 → 14.50 µs. Report /tmp/mk/f24/REPORT.md |
| F25 | Ring-consumer GEMV loop shape | A VMEM-fed M=16 dot costs ~0.15 µs fixed plus ~8 TB/s streaming, so a ring-consumer loop of one small dot per `fori_loop` iteration is compute-bound below HBM rate ((128, 2176) tiles: 2.55 TB/s). Pick the loop shape (unroll, accumulator in values vs scratch, minimum bytes per dot in the seed) from the probe. | ✅ v82: step L=8 body 357.7 µs at U=1 → 348.9 at the 2 MiB default (−2.5%). Implemented as `pallas_stream_unroll` (body replication). With the old 4 MiB target, step L=8 body was 360.6 µs at U=1 vs ~393 at U=2/4/8/default. Timeline: gate/up −10.5, out −3.5, qkv −2 µs, but LM head +40 µs (a replicated dot+argmax body is scheduled worse at any U≥2; refill placement disproven by probe). Default target lowered to 2 MiB so steps ≥ 2 MiB stay rolled |
| F26 | Grouped ring waits | A DMA wait orders every later vector op after it, so a wait per tile exposes the MXU fill/drain on every dot (probe 5: 6.1 → 2.7 TB/s). In a stream-unrolled step, run the ring waits of G iterations before their bodies, G = min(U, depth // loads per iteration) per ring so every waited copy has started. Each step returns its body as a closure (`_fori_run`). Only offsets and ring waits are hoisted; immediate loads, stage waits and refills stay in the body. Follow-up F26b: fewer refill branches (`pl.when` ≈ 30 ns/tile). | 🟡 main worktree, knob `pallas_stream_wait_group` (default off). TPU with the whole unroll grouped: step L=64 2044.5 → 2146.1 µs, hybrid 1899.8 → 1932.5 (fewer copies in flight in a DMA-bound loop). Useful only where the consumer is compute-bound |
| F27 | Packed runtime rows through 32-bit words | A runtime row of a bfloat16 VMEM tensor with every lane kept (`w[layer, :]`, `table[ids[b], :]`) is read through the ref's 32-bit words (`ref.bitcast(u32)[..., i // 2, :]`, shift by `16 * (i % 2)`), not by loading the 16-row window, widening it and rotating it with `pltpu.roll`. Applies to any leading-dim stack (`row_packing(dtype, rows) == 2`). | ✅ main worktree (`_packed_row_load_applies` in `vmem_scalar_load.py`; tests `test_runtime_row_gather_packed`, `test_runtime_row_packed_stacked`). TP8: all 14 norm-weight rolls gone; exch 2072.1 / noex 1896.5 µs (v88 2071.7 / 1899.8), neutral on TPU because the step is not consumer-bound alone |
| F28 | Gated loads take their own arena | An arena streams in program order, so a slot whose refill waits for an in-kernel producer (F20b, `StreamLoad.producer`) holds back every later copy, including the next layer's weights. Producer-gated raw loads get a raw arena of their own (one more key in `_ring_layouts`), so the ungated copies keep streaming through the router, top-k and the first expert tile. Dense models have no gated loads, so it is a no-op for them. | ✅ main worktree 2026-10-03 (`test_gated_copies_take_their_own_arena`; local suites green). Harness version on the MoE stack: 39.2 → 37.6–37.9 µs/layer. **Compiler version on TPU (v90, `/root/tp8/moe90.out`): MoE stack kernel body 39.26 → 37.85 µs/layer (L=8), 39.16 → 37.62 (L=16), numerics PASS; TP8 L=64 2072.5 µs (neutral)** |
| F29 | Shared write-behind stages for HBM stores | Each store to a tensor kept in HBM (F15) staged its rows in a VMEM buffer of its own, so a block verify's 8 per-row snapshots per layer took 8 stages (24 × 384 KB at small shapes: 11.9 MB, over budget). Stores of one shape to one tensor now share a stage and semaphore: every store already waits for the write before it to the same storage, so the stage is free by then. `_hbm_slot_bytes` counts one stage per (tensor, shape). | ✅ main worktree 2026-10-03 (`test_hbm_resident_snapshots`, `test_hbm_resident_block_rows`). Verify kernel at small shapes: HBM slot bytes 11.86 → 1.97 MB. Next: a pool of 2 stages per tensor so a write is not waited on at the next row (the reference waits at the next GDN layer). |
| F29b | Lane-dense snapshot outputs in HBM | A narrow state snapshot (`conv_snap [L, B, C, 4]`) is passed lane dense, laid out (0, 1, 3, 2) by XLA, and lane-dense tensors were always kept whole in VMEM: about 15.7 MB at L=64. An input that is only stored, each time as a whole slice at scalar leading indices, with a dim order that swaps the last two dims into a 128-lane minor dim, now stays in HBM. Each slice is staged transposed into physical order and written whole by a DMA. | ✅ main worktree 2026-10-03 (`test_hbm_resident_snapshots`) |
| F29c | Stable static_range induction types | Type propagation iterates loops to a fixed point and merges AST node types across passes. A loop over `hl.static_range` gave its induction variable a fresh unbacked symbol on every pass, so `v[j : j + n]` merged bounds from two passes, lost its static length, and failed with ShapeSpecializingAllocation. The induction variable type is now cached per For node. Frontend, backend-agnostic. | ✅ main worktree 2026-10-03 (`test_static_range_slices_keep_their_length`) |
| F30 | Combined row-store runs to HBM tensors | A block of rows written one at a time at runtime rows `base + c` (a verify block's 8 KV rows per cache per layer) did a read-patch-write DMA round trip of the sublane tile per row, serialized: 8 per cache per layer, about 380 µs at L64. `row_store_runs` finds runs in the FX graph: stores with equal subscripts except the row, rows `base + c` of one base node with static offsets at most a tile apart, and no other use of the tensor, nested loop or branch between them. A run reads the window of two sublane tiles holding its rows once (clamped to the tensor's end), patches each row in VMEM, and writes it once, leaving the wait pending as before. `_hbm_slot_bytes` counts the 2-tile stage. Generic: any per-row scatter of a static block (KV append of a speculative block, chunked prefill rows). | ✅ main worktree 2026-10-03 (`test_hbm_resident_block_rows` asserts one window read and write). Verify kernel ALL OK in interpret mode. TPU timing pending (v92). |
| F31 | Write-behind ring of stages for HBM stores | F29 shared one stage per (tensor, shape), so each of a verify block's 8 conv snapshots per GDN layer waited for the previous one's DMA (~0.45 µs each, serial). Whole-slice stores of one shape to one tensor in a root's graph now take turns in a ring of `K = min(#stores, _STORE_RING_BYTES // stage)` stages (`hbm_store_rings`, cached per graph). A store waits only for the previous write from its own slot, provided every pending write to the tensor is a ring write to a provably disjoint slice (some index differing as integer literals). Otherwise it flushes all of them, as before. `_hbm_slot_bytes` counts K stages. Generic: any repeated per-step snapshot or scatter of whole slices. | ✅ main worktree 2026-10-03 (`test_hbm_resident_snapshots`: no wait between snapshot rows). **v94 TPU: verify GDN conv-snapshot phase 5.46 → 0.08 µs/layer (with F32).** |
| F32 | Inverse transposes cancel at lane-dense boundaries | A value that the source transposes into a lane-dense tensor's physical order (`snap[i, b] = window.T`, `state[i] = window.T`) was swapaxed back to logical and then transposed to physical again on store. That was 2 XLU transposes per store, ~1.2 µs/layer for the verify conv snapshots. The store now writes the permute's input (`physical_store_value`). Load side (F32b): a lane-dense load whose only users are permutes that undo the physical transpose, directly or through dtype conversions, stays physical (`pallas_physical_values`), and those permutes become no-ops. | Store side ✅ main worktree 2026-10-03, in v94. Load side ✅ dev tree (`test_lane_dense_transposes_cancel`), porting. |
| F33 | Multi-dim narrowing of device values | `v[b, :, :k]` on a device value (one int and one slice narrowing) was rejected ("only one narrowing index"). `device_ir._split_narrowing` splits it into one subscript per narrowed dim, right to left, then a final subscript that adds the `None` dims. Frontend, backend-agnostic. | ✅ dev tree 2026-10-04 (`test_multi_dim_narrowing_of_value`) |
| F34 | Several slice loads of one HBM tensor per root | A root that reads and rewrites one slice per sequence (`rec_state[l, b]` for b in `static_range(B)`) disqualified the tensor from HBM residency, because loads in one graph were assumed to be one inner-loop tile stream. The repeated-graph check now applies only to inner-loop loads. `MegakernelPlan.slice_loads` is keyed by (root, storage, ordinal among the root's loads of the tensor), since codegen graphs are copies of the planned ones. Each slice gets its own buffer and semaphore, and `_early_stores_safe` already proves the per-b slices disjoint. | ✅ dev tree 2026-10-04 (`test_hbm_resident_batched_state`). M10 L8 B8: rec_state 37.7 MB whole in VMEM → 8 slice buffers |
| F35 | Lane-dense read-write state in HBM | F29b kept only store-only lane-dense tensors in HBM. A narrow state that is read too (`conv_state [L, B, C, 4]`, lane dense (0, 1, 3, 2)) stayed whole in VMEM: about 31 MB counted at L64 B8. Now every access may be a whole slice (last two dims whole, leading dims whole or integer) in a root body. A load copies its slice in physical order (`HbmCopy.perm`) and `logical_load` transposes the buffer, which cancels against a source transpose (F32b). If the lane-dense-off config cannot fit in VMEM, it is no longer offered, instead of failing the whole compile (`pallas_lane_dense_off_models = None`). | ✅ dev tree 2026-10-04 (`test_hbm_resident_layer_state`, `test_lane_dense_operands`, `test_lane_dense_transposes_cancel`). M10 L64 B8 codegen now fits |
| F36 | Tile-aligned exchange trims | `logical_region_parts` trimmed remote copies of padded scratch to the logical rows, e.g. `pl.ds(0, 1)` for a decode row. Mosaic rejects DMA slices of tiled dims that are not tile multiples ("Slice sizes along tiled dimensions must be aligned to tiles"). That broke M10 B1 on TPU. Trims now round up to 8 rows / 128 lanes, and the padding is zero on every device. | ✅ dev tree 2026-10-04 (`test_tp_all_reduce_gemv_chain` / `_mlp_stack` expect `pl.ds(0, 8)`) |
| F37 | Slice reads wait only for overlapping writes | A slice copy of an HBM-resident tensor (`root_slice_load`) flushed every pending write to the tensor first. So the per-sequence state write of sequence b was waited on right before sequence b+1's slice was read, which exposed the write DMA for every sequence and layer. `slice_read_waits` now waits only for the ring writes to slices the copy may overlap (an integer index differs → disjoint), and flushes everything when a non-ring write is pending. The ring-write helpers (`RingWrite`, `ring_waits`) moved to `megakernel.py`. | ✅ dev tree 2026-10-04 (`test_hbm_resident_batched_state` checks write b=0 is still in flight at slice b=3's wait). TPU A/B in v99 |
| F38 | `hl.all_reduce`: the exchange algorithm is a compiler choice | Every TP kernel hand-wrote its exchange, and the best one depends on the payload. v101 M10: one-shot B1 2163.9 vs RS+AG 2419.5 µs, but B8 4567.7 vs 3113.0. `hl.all_reduce(dst, src, peers, rank, *, algorithm=None)` is a host statement. An AST pass (`collective_expansion.py`) expands it, before the host `static_range` unroll, into top-level device loops of either algorithm. The choice is a host `if` on a shape expression: a cost model with 3.9 µs per round and 61 GB/s per peer (fitted from v101), break-even ≈ 42 KB of f32 at world 8. RS+AG also needs `cols % (128 · world) == 0`, and world ≤ 2 always uses one-shot. Under static shapes the test is a literal. Type propagation resolves it, and a new generic pass, `fold_static_host_ifs`, keeps the taken branch. That pass also lets any kernel choose device loops by a host `if` on shapes. Both algorithms sum in rank order before the cast, so they agree bitwise. Buffers are per site. A site that is alone in its loop alternates 2 slots by loop-index parity; ≥2 sites per loop body use 1 slot each. | ✅ dev tree 2026-10-04 (`test_tp_all_reduce_algorithm_by_payload`, `test_tp_all_reduce_sites_share_slots`, `test_host_if_on_shape_chooses_roots`). TPU `tpb102` (v102, M10 with `hl.all_reduce` at every exchange): the compiler picks one-shot at B1 and RS+AG at B8. L8 numerics are identical to v101. L64 bodies: B1 2197.5 vs 2163.9 µs hand-written (+1.5%), B8 3144.3 vs 3113.0 (+1.0%). The gap is the extra partial copy and residual root. |
| F39 | Early first tiles of a scan with a runtime bound | `_loop_prime_copies` started a scan's first K/V tiles among the ring copies (one layer ahead) only for static bounds. A bounded scan `hl.tile(t + 1)` primed its first tiles right before its loop, so every sequence of M10's per-b attention waited out one DMA latency serially. The first copy's address depends only on the loop's start and block size, so dynamic-end loops qualify now. The copy always starts (`LoopPrime.dynamic`, no `num_iterations > 0` guard), and a loop of no iteration waits for it after (`_drain_fori_primes`). | ✅ dev tree 2026-10-04 (`test_hbm_resident_bounded_scan_prime`). M10 L64 B8: all 8 sequences' first tiles start during the previous layer. TPU: `tpb100` (v100) |
| F40 | Exchange trims by dtype (F36 regression fix) | F36 rounded every remote-copy row trim up to 8 rows. On f32 buffers that moved 8x the payload: 160 KB instead of 20 KB per peer in M7 (`recv [2, 8, 16, 5120]`), 128 exchanges per call. Probe `probetrim` (`probe_trim.py`, DMA of `ref.at[i, pl.ds(0, rows)]`): f32 accepts 1, 8 and 16 rows but rejects 2 and 4. bf16 rejects 1, 2 and 4. Lanes need multiples of 128. So the rule is whole 8-row tiles, plus a single row of a 32-bit ref. M7's 1-row f32 pushes (v88–v90) used that exception. | ✅ dev tree 2026-10-04 (`test_tp_exchange_trim_by_dtype`: f32 m=1 → 1, f32 m=2 → 8, bf16 m=1 → 8; gemv_chain/mlp_stack expect `pl.ds(0, 1)` again). bf16 16-row tiles: probe `probetrim2`. TPU: `tpb101` (v101: 1 row for every f32 trim; the same as the final rule at m = 1 and 8) Packed-dtype follow-up (`probetrim3`): local VMEM→VMEM DMAs of bf16 rows below the native 16-row tile (int8: 32) crash the Mosaic compiler with SIGFPE. Remote copies accept 8-row bf16 trims: tpb102 B8 runs and passes with RS+AG bf16 gathers trimmed to 8 rows. So the rule applies only to remote copies, which are its only caller (`distributed_ops`). |
| F41 | Roots share slice buffers when VMEM bounds the rings | Each whole-slice load of a tensor kept in HBM had a buffer of its own. M10 B8 reads 8 rec_state slices (393 KB each) in each of the 3 GDN roots of a loop body: 24 buffers, 9.4 MB, and the arena ring got only 7680 tiles (gate/up alone is 20 ring tiles per layer). A root reads a slice once its copy lands, so roots can share: the n-th load of a (shape, dtype) in each root uses the n-th buffer of that class. A shared buffer's early copy (`_plan_early_copies`) must wait until the previous reader root is done. With a deep ring, every refill may be issued before then, which leaves the copy no anchor (seen in `paired_state_stack`: late starts). So sharing is a per-config choice, `slices_pooled`: share only when it lets the rings fit or deepens them. The model resolves the rings both ways (`hbm_slot_bytes` vs `hbm_pooled_slot_bytes`), and codegen uses the same decision. M10 L8 B8: ring 7680 → 9600 tiles, slice buffers 26 → 10, copy starts unchanged. | ✅ dev tree 2026-10-04 (`test_hbm_resident_slices_shared`). TPU `tpb103`: L8 B8 numerics as v102; L64 B8 body 3144.3 → 3137.9 µs with the ring at 5120 tiles instead of 3840. Ring depth is not what holds B8 back |
| F42 | One-shot all-reduce sums its own term from the source | The one-shot expansion copied this rank's partial into `recv[slot, me]` so the rank-ordered sum could read every term from one buffer: a local copy per site (128 per M10 call). The sum root now selects `where(arange(world) == me, src, recv)` per tile, so the order (and bits) stay identical across ranks and the copy is gone. Reduce-scatter keeps its chunk copies (B8 payloads are 10 KB per chunk). | TPU `tpb104` (v104): L8 B1 numerics as v103, but the L64 B1 body **regressed 2197.7 → 2589.8 µs** (+3 µs per all-reduce) with the sum as `where(arange(world)[:, None, None] == me, src, recv)` + `sum(0)`. F42b (v105) sums in a static loop over ranks with a scalar `where` per term: `tpb106` L64 B1 2212.4 µs, still +14.7 over v103. **Reverted** (the copy is cheaper than any per-term select in the sum root) |
| F44 | A residual add runs in the one-shot sum's loop | Every `hl.all_reduce` is followed by its consumer, a pointwise tile loop (`res[tm, tn] += red[tm, tn]`): a root of its own, which reloads the sum the root before stored. When the statement after a site is a 2-D `hl.tile` loop that reads the destination only at its own tile and never writes it, the expansion moves it into the one-shot sum's loop. Reads of `dst[tm, tn]` become the summed value; the `dst` store stays for later readers. The consumer's tiles must cover the destination. That is a static host `if` on the shapes, folded like F38's, with the unfused pair as the fallback. Reduce-scatter keeps the consumer after its gather. | ✅ dev tree 2026-10-04 (`test_tp_all_reduce_fuses_pointwise_consumer`). M10 L8 B1: 24 → 16 roots. TPU `tpb113` (v107): L8 B1/B8 numerics identical to v103; L64 B1 2197.7 → **2165.2 µs** (−1.5%); B8 3138.2 (RS+AG, unaffected) |

## Milestone ladder

Every milestone has a Helion source kernel written the way a model author would
write it, a measured reference, and a perf gate. Status: ✅ done, 🔄 in
progress, ⏳ next, ⬜ later.

| # | Milestone | New features | Reference to beat / match | Gate | Status |
|---|---|---|---|---|---|
| M1 | Dense MLP, sequential roots | F1, F11 | 3 separate Helion kernels: 365 µs | correct; one launch | ✅ 290 µs (sync DMA) |
| M2 | Dense MLP with global ring | F2, F3 | XLA 100.1 µs; hand target 95.0 µs | **< XLA** | ✅ 96.0 µs at M=1 (XLA 99.8); 94.5 µs at M=16 (XLA 103.0); hand target 94.6 |
| M3 | One attention-plus-MLP decoder layer (Qwen3.8 full-attention layer, per-rank TP8 shapes, no TP) | F4 (KV stream), F7 (KV row write-behind), F9 (`pos`, dynamic trip count), several sites with different shapes in one ring | XLA 67.9 µs; reference about 34 µs/layer | < 67.9 µs, then ≤ 37 µs | 🔄 first gate ✅ with default configs: GQA layer 44.7 µs chained (32.9 µs kernel body), GDN layer 44.5 µs (32.9 µs body; XLA 68.7). The rest is per-call overhead; the honest per-layer number comes from M5 stacks |
| M4 | N-layer dense stack (host layer loop) | F5, F6, F8 | XLA `scan`; reference scaled | per-layer time flat in N; ≤ 1.10× reference | 🔄 F5 ✅ merged: 86.8 µs/layer at L=64 vs XLA scan 95.2 (H=4096, I=12288); tiled Qwen shape (H=5120, I=2176) 20.7 µs/layer at L=8 vs scan 31.3 with F3 defaults; F6/F8 open |
| M5 | Qwen3.8 hybrid stack (GDN + GQA, LLLF) | GDN recurrent and conv state (F7), static layer pattern (F5 branches) | reference with `noexchange` | ≤ 1.10× reference `noexchange` | ✅ with F15 (merged into main 2026-10-03): kernel body 33.3 / 32.1 / 31.2 / 30.9 / 32.7 µs/layer at L = 4/8/16/32/64 (F15 worktree numbers; re-check of merged main queued), vs ≈ 34 µs/layer for the reference `noexchange` (2.29 ms − head, / 64) |
| M6 | Full step without TP: embedding, final norm, LM head, argmax | F4 (row gather), ring flows from last layer into LM head, argmax (missing on Pallas today) | XLA 4.12 ms; reference `noexchange` 2.29 ms | ≤ 2.40 ms | ✅ **full step (`examples/qwen38_decode_step.py`, embedding + 64 hybrid layers + final norm + LM head shard + argmax) on one core: 2.183 ms/token chained device time, kernel body 2.164 ms (2.99 TB/s), top-1 matches XLA** (2026-10-03, main before F17-lite) vs reference `noexchange` 2.29 ms. L=8: 377.6 µs device, 363.7 body (was 396.4 before F15 and the VMEM fixes). Earlier pieces: HBM row gather, early row reads, first-index argmax fix; embed+head 101.7 µs body vs XLA 125.5 |
| M7 | TP8 | F10 + overlap (source uses the existing `hl.make_async_remote_copy` / `hl.remote_barrier`) | reference 2.33 ms | **≤ 2.45 ms/token** | ✅ **gate met with F18 (knob, 2026-10-03): TP8 step L=64 at 2356.9 µs/token device, body 2209.8 µs, top-1 OK (reference 2330 µs).** History: merged into main 2026-10-03 (partial): in-kernel all-reduce correct on 8 devices (one-shot push into `recv[parity, sender, m, h]`, computed peer). Exchange overhead: TP MLP L=16 21.48 vs 19.93 µs/layer (+7.8%), TP hybrid L=8 43.08 vs 34.71 (+24%). Exchange is a ~4.3 µs/layer latency floor; it hides only behind weight DMA that waits during it, and the hybrid rings are below full bandwidth. Budget at L=64: ≤ ~3.7 µs/layer exposed. M7b merged (F23 + bubble-aware holds): hybrid exchange 8.3 → 6.7 µs/layer, hybrid L16 extrapolates to ~2.51 ms for 64 layers (/tmp/mk/m7b/REPORT.md). TP8 zero top-1 = Mosaic one-element store→DMA hazard; fixed by padding one-element outputs. **TP8 step L=8 on TPU: top-1 OK, 424.6 µs device (noex 361.2), so the exchange costs ~7.9 µs/layer → ~2.65 ms at L=64.** Next: DMA priority probe, F18 arena, exchange latency |
| M8 | Generality pass | MoE: F20-0, F20a (index-vector ring sites), F20b (refills gated on in-kernel top-k), F21 (device trip counts); no speculative issue (Kimi doesn't guess experts either). Then MLA | XLA layer and scan on Qwen3-30B-A3B-like (H 2048, E 128, K 8, I 768) and DeepSeek-V2-Lite-like (E 64, K 6, shared expert) layers; Llama/Qwen3 dense stack | no model-specific code paths; MoE layer < XLA and ≥ 85 % of the bandwidth model; top-k bubble ≤ 1 µs | 🔄 M8a merged into main 2026-10-03 (F20-0, F20a, F20b, SMEM index mirror): Qwen3-30B-A3B-like layer with router + top-k in kernel 29.29 µs (69.8 % of model) vs XLA 47.82; ids as inputs 25.77 (78.7 %) vs 41.22; DeepSeek-V2-Lite-like 55.21 (67.5 %) vs 67.18. Beats XLA everywhere, misses the 85 % gate: at M=1 the in-kernel compute floor (24.54) is above the gate, and the down ring streams at 2.67 TB/s. Arena default (F18), v86 tree: Qwen MoE 28.84 → 29.30 µs (ids 25.31 → 25.76), DeepSeek-V2-Lite-like 50.85 → 47.22 µs (`/tmp/mk/m8/arena_m8.log`). Next: F20c, F21, MLA (/tmp/mk/m8/IMPL_REPORT.md) |

| M9 | **Block verify (DFlash, 8 rows)** — the reference's headline mode (1,515 tok/s at acceptance 6) | Multi-row step over one sequence: 8-row GEMVs (free under F11's 16-row padding), causal block attention over cache + block, conv and recurrent state advanced over 8 tokens with per-position snapshots for rollback, per-row argmax, hidden taps. Expected compiler work: multi-row single-tile roots (sequential `static_range` over rows), per-row state snapshot writes (F7/F15 with 8 slices), block KV row writes (F7 with an 8-row block) | reference `make_verify_block(block=8)` on our pod: **3.044 ms/call** L=64 (2.437 noexchange; same-day decode 2.327 / 2.289), `bench_qwen_ref_verify.py`, design note `/tmp/mk/verify-ref/VERIFY_DESIGN.md` | ≤ 1.0× reference | ✅ **1.8% faster, same session: 2988 vs 3043 µs/call** (16-call loop, the reference's methodology; body 2945.8 µs) with `/tmp/mk/tp8/tp_verify_rsag_b16.py` + `fast_math=True` on v97 (reduction distribution + lane-reduction-aware operand order; F31/F32/F32b). Without fast_math 3070.7. Numerics: against an f32 oracle the kernel's error matches the bf16 reference oracle's (hidden max rel 0.0203 vs 0.0205; top-1 7/8 each), so the strict kernel-vs-bf16-oracle hidden check fails only from bf16 drift. Standalone recurrence matches hand Pallas in the same harness (Helion 618 vs hand 609 ns/step). Next: M10 |
| M10 | Batch decode B = 2–8 (separate sequences) | Per-sequence caches and states (batched leading dim on every state access), B-row GEMVs | XLA; GB200-class targets from the reference's chart (1.4–2× at B 1–8) | per-token time flat in B up to 8 | 🟡 numerics OK (bf16 drift vs f32 oracle). L64 v101: RS+AG B1 2419.5, B8 3113.0; one-shot B1 2163.9 (B1 gate ✅), B8 4567.7 µs body. Algorithm choice by payload = F38; XLA loop B1 5818.2, B8 6257.3 µs/call. Next: overlap per-b attention/recurrence chains |

Why M9 next: decode parity is met (loop methodology: ours 2102 µs/token vs the
reference 2328 with the TP exchange), but the reference repo's headline numbers
are speculative decoding, whose verify kernel streams the same 6.5 GB/rank once
for 8 tokens. Its extra work is compute (8 query rows, 8 sequential state
updates), so it stresses consumer slack, which is exactly where TP8 is tight.

## Model zoo for generality checks

Each compiler feature is tested on several of these, never only on Qwen3.8.

| Kernel | Exercises |
|---|---|
| Dense MLP (Qwen3-8B 4096/12288; Llama-3-8B 4096/14336) | F1-F3, uneven widths |
| GEMV chain with mismatched shapes (synthetic) | F2/F3 slot sharing, refill decode across sites |
| Llama/Qwen3 dense decoder layer and stack | F4-F9 without GDN. ✅ 16.3 µs/layer at L=32 (3.35 TB/s; XLA 35.0) |
| Qwen3.8 hybrid stack | F5 branches, F7 states. ✅ 30.9 µs/layer at L=64 |
| MoE layer (top-k routed experts) | data-dependent streams (M8 agent) |
| Gemma-3-like stack: 5:1 local/global attention, qk-norm, four norms, GeGLU, soft-capped head | periodic branches with a different period, masks. ✅ L=12 stack 18.86 µs/layer (2.97 TB/s; XLA 39.50); full step top-1 matches (327.9 µs, 2.82 TB/s) |
| DeepSeek-V2-Lite-like MLA stack (absorbed, latent cache), dense MLP | narrow per-head absorb matmuls, latent caches. ✅ 59.64 µs/layer (2.73 TB/s; XLA 75.93); 54.03 (3.01) with a rows-only `w_kv` tiling, so F24 |
| Parallel-residual stack (GPT-J/Falcon/Cohere) | one input feeding two branches. ✅ TP8 shard 18.38 µs/layer (2.74 TB/s weights, 2.96 with KV; XLA 93.75 / scan 80.04); gap = whole-cache read (F4 `pos` bound) and a 32-slot ring that can't fill before `fc_in` (F16) |

## Decisions taken (veto welcome)

1. **Host layer loops (D4) use option (a).** A Python `range` loop at host
   level around `hl.tile` roots is allowed in megakernel mode and folded by the
   compiler (F5). `hl.static_range` unrolls. This relaxes `NestedGridLoop` but
   adds no new API. Needed from M4.
2. **The per-rank comparison until M7** is the reference with the `noexchange`
   ablation, so TP is not on the critical path for single-core features.
3. **Weight layout is the model author's choice.** The compiler does not
   re-tile weights in HBM. F3 picks contiguous row-slab tiles for row-major
   weights, measured at 3.20 TB/s on the MLP. If contiguous pre-tiled layouts
   turn out to be required for peak, that becomes a host-side utility, not a
   compiler pass.

## Open risks

* **jax 0.10 vs 0.11.1 on the pod.** Older Mosaic rejects some unaligned
  full-dim slices, and our codegen must stay within what 0.10 accepts.
* **Mosaic compile time and code size** for a 64-layer program. F5 folding is
  mandatory; full unrolling is not an option.
* **Ring depth vs. VMEM** once attention and GDN buffers are live. F8 is needed
  by M4-M5.
* **Collectives API gap** (F10). This may need backend work; it is
  investigated before M7.
* **Refill decode cost** grows with the number of sites per folded body (about
  8 for Qwen3.8). A `pl.when` chain over about 8 sites per refill must stay off
  the critical path.

## Status log

* **2026-10-02:**
  * M1 merged into the worktree: sequential roots, VMEM intermediates, sublane
    padding, jax_fn fixes. Measured 290 µs (sync DMA).
  * Hand-written MLP target measured at 94.3 µs device time; XLA at 99.5 µs.
  * Reference Qwen3.8 megakernel runs on the pod: 8 layers = 0.431 ms/token.
  * M2 (global ring) is in progress with a subagent.
  * Goal widened to full Qwen3.8 parity; this roadmap was created.
  * M3-prep agent started in worktree `helion-m3-prep`. It writes the GQA and
    GDN layers as plain Helion source, runs them on TPU and catalogues
    blockers.
  * Plan §3.7 (M3) and §3.8 (F5 folded loops, periodic branches) drafted.
  * Re-prioritized F9b (data-dependent trip count): masked full-context
    attention costs only about 9 µs/step at ctx = 2048, so it is deferred.
    Scalar `pos` for the KV row write and the mask (F9a) stays in M3.
  * **M2 done.** The global ring (one ring per dtype/ndim, prologue, per-site
    waits, `pl.when` refill chain across roots) measures 98.6 µs at M=1 vs XLA
    100.1 µs and the hand target 95.0 µs (same run). Fixed a real-Mosaic
    failure: masked scratch stores at M=1 reshaped a 1-D `i1` mask. The mask is
    now built from a broadcasted 2-D iota. The remaining 3.6 µs gap to the
    target is unexplained so far: refill decode, the M=1 masked stores, or
    slot shape. A review agent is checking the ring for hazards.
  * Depth sweep at M=1, device time:

    | Depth | 4 | 8 | 12 | 16 | tk512, D=6 |
    |---|---|---|---|---|---|
    | µs | 113.2 | 98.6 | 98.4 | 98.7 | 98.8 |

    Depth saturates at 8 slots (16 MiB), matching the hand target's behavior.
  * F5 (folded host layer loops, M4) started in worktree `helion-f5-loops`,
    from an M2 snapshot (base kept at `/tmp/mk/snap-m2-base` for 3-way merges).
    Pod etiquette for agents: `flock /root/tpu.lock`.
  * M2 review (`/tmp/mk/m2review/REVIEW.md`). The semaphore, ordering and
    drain protocol was verified sound. Found:
    * a wrong result: roles keyed by name, so tensor lists collapse and
      atomic writes go unseen;
    * an autotune KeyError on `hl.grid` roots;
    * three generality gaps:
      * activations streamed and then rejected at M=1;
      * root-level weights kept whole in VMEM;
      * one unsupported load fails the whole kernel;
    * all multi-root Pallas kernels forced sequential;
    * a self-referential ring check.
    A fixer agent is working on these in the main worktree.
  * **M3-prep finished** (`/tmp/mk/m3/FINDINGS.md`):
    * Both the GQA and GDN layers are written as ordinary Helion source, as
      several single-root kernels and as one multi-root kernel each. All are
      correct on TPU against the qwen reference.
    * Five generic fixes:
      * half-open slices on values;
      * bf16 single-row store;
      * jax_fn with scratch and SMEM;
      * root-local scalars;
      * packed-dtype row gather.
    * 13 blockers catalogued with repros (B1-B13).
    * Timings that matter:
      * GQA layer multi-root: **84.3 µs** (XLA 67.9). Weights stream at only
        ~1.1 TB/s because the block sizes must be powers of two, which forces
        (128,128) slots with 256-byte rows.
      * Decode attention core: **28.7 µs** against a 0.6 µs KV floor.
      * GDN core: **79 µs** (48.9 MB of spills).
  * **Re-prioritized M3.** The cores have to fit under the ring's ~4.5 µs of
    buffer, or each stalls the stream. Two new feature lines:
    * F12, non-power-of-two block sizes plus the F3 seed: agent in worktree
      `helion-pallas-blocksizes`;
    * F13, attention and GDN core codegen efficiency: agent in
      `helion-m3-prep`. Targets ≤ 5 µs and ≤ 8 µs.
    KV in HBM (F4) and the row DMA write (F7) come after these, because
    whole-VMEM caches still fit for a single layer.
  * **F5 done in worktree `helion-f5-loops`, not merged yet.**
    * A host `for i in range(L)` around roots becomes `@pl.loop`. The ring
      index is affine in `i`, refills decode `(layer, site)`, and 3-D
      stacked weights stream through 2-D slots.
    * `hl.static_range` plus a constant host `if` gives periodic stacks
      (interpret mode only so far).
    * TPU, M=1, µs per layer:

      | Kernel | L | Megakernel | XLA `scan` |
      |---|---|---|---|
      | H=4096, I=12288 | 64 | **87.2** (3.46 TB/s) | 95.2 |
      | H=5120, I=2176, full-width gate/up | 64 | **25.3** | 30.8 |
      | H=5120, I=2176, 128-wide tiles | 8 | 38.2 | 31.2 |

    * Compile time and code size stay flat in L: about 0.4 s and 271
      generated lines.
    * The tiled H=5120 case loses because the tiles are 128 wide (F12 should
      fix it); the ring alone runs at 23.3 µs.
    * Gaps:
      * loops must have step 1 and cannot nest;
      * F6 (small per-layer params) is not done;
      * no `i % P` analysis.
    * Full report: `/tmp/mk/f5/REPORT.md`.
    * Next: once the M2 review fixer is done, 3-way merge against the
      `/tmp/mk/snap-m2-base` snapshot.
  * **F4/F7/F9b branch survey done** (`/tmp/mk/f4/SURVEY.md`).
    * Main already has fori loops whose trip count depends on data, with
      double-buffered prime and prefetch.
    * Missing everywhere:
      * a stream over a tensor that is also written at root scope;
      * an HBM write whose wait sits before the next overlapping read;
      * an aligned block write.
    * Mosaic forbids a dynamic one-row DMA on the tiled dim, so the row write
      becomes an aligned 16-row block. In stage 2 it is forwarded into the
      attention stream's slot.
    * Reusable pieces: write-behind wait and drain from
      `pallas-store-double-buffering`, tail zero-fill from
      `pallas-row-slab-boundary-dma`.
    * Design in plan §3.7 (revision). Implementation starts after the
      F5/M2-fix merge, to avoid a five-way merge.
  * **F12 non-pow2 block sizes done** in `helion-pallas-blocksizes`
    (`/tmp/mk/b6/REPORT.md`, patch `/tmp/mk/b6/nonpow2.patch`).
    * It applies cleanly to both megakernel worktrees.
    * A block size is legal when it is a power of two, the full extent, or a
      tile multiple (lane 128; sublane 16 for bf16).
    * bf16 [16,5120]×[5120,2176]: 16.35 µs at (16,2176,512) vs 41.6 µs for
      the best pow2 (XLA 6.0).
    * Existing generated code is byte-identical (822/822). Autotuner search
      of non-pow2 sizes is opt-in.
  * **DMA probe for F3** (`/tmp/mk/f3/RESULTS.md`).
    * Ring bandwidth depends on **slot bytes, not row width.** 256-byte rows
      reach 2.9 TB/s.
    * 32 KB slots ((128,128) bf16) reach only 0.69 TB/s, from a fixed cost of
      30-48 ns per DMA. That explains the M3 GQA layer's 1.1 TB/s.
    * The knee is around 128-256 KB.
    * A 4 MiB ring loses only ~4% against 16 MiB.
    * **F3 decision:** one ring per tile shape, each with depth budgeted in
      bytes, plus a megakernel default tile heuristic: slot ≥ 512 KB, N ≥ 256
      for the v7 MXU or the full dim. No byte-uniform reinterpreted slots,
      i.e. no `pool_alias` hack. Qwen's dims have gcd 128, so a single shared
      slot width would force 128-wide MXU tiles.

  * **F5 and the M2 review fixes merged** into this worktree. The merge was
    3-way per file against `/tmp/mk/snap-m2-base`, with the merged state
    snapshotted in `/tmp/mk/snap-merged`.
    * Conflicts were resolved by union. `StreamModel` and `RingRoot` carry
      both sides' fields.
    * `_stream_load` takes the folded loops. It raises an error for a
      scalar index that is provably out of bounds, instead of falling back.
    * Sequential-roots mode is also entered when there are folded host
      loops.
    * Root-level stream sites are skipped inside folded loops. Open: they
      need a layer term in `_root_site_index`.
    * Behaviour change: `w2[0, tk, tn]` (a constant leading index) now
      streams via scalars.
    * Results:
      * test_pallas_megakernel: 39 passed.
      * The other suites match their pre-merge counts.
      * The TPU jax_fn modules (M2 M=1 and M=16; F5 b/w1 stacks at L=8 and
        64; small numerics modules) are byte-identical to the measured ones,
        so 98.6/96.9 µs and 87.2 µs/layer carry over.
    * The F12 patch is applied on top.
  * **F3 agent launched** in the isolated worktree `helion-f3-rings` (a copy
    of the merged state plus F12). Scope:
    * per-shape rings with byte-budgeted depth;
    * a megakernel default tile heuristic;
    * non-pow2 candidates on by default in megakernel mode.
    * Gate: the tiled Qwen MLP stack (H=5120, I=2176, L=8) beats XLA scan,
      with no regression on M2/F5.
  * **F13 done; m3-prep ported into this worktree.**
    * Generic M3 fixes ported: #1 half-open slices of values, #2 packed
      dynamic-row RMW store, #4 root-local scalars, #5 packed runtime-row
      gather. Also the jax_fn fixes and `examples/qwen38_decode.py`.
    * F13 adds two generic fixes (report: `/tmp/mk/m3core/REPORT.md`):
      * **Launcher staging of whole-array blocks.** All input copies start
        before any wait, and an in-place output reuses its input's buffer.
      * **`put_full_operand_first`.** Mosaic lays out an elementwise op
        like its first operand, so add/mul now put the full-shape operand
        first. This took the GDN core from 69.05 to 3.30 µs.
    * Attention core: 4.35 µs at block 2048 (≤ 5 needs block ≥ 1024; the
      default of 32 gives 36 µs). The remaining ~2.6 µs is KV cache
      whole-in/whole-out traffic, which F4/F7 removes.
    * Two conflicts resolved: the megakernel scratch mask now runs before
      the packed row store, and an import.
    * Suites: test_pallas 240, load_store 118, megakernel 39, block_sizes
      13, bound_kernel 11, loop_dependencies 5. All pass.
    * **TPU after the port (launcher staging helps the MLP too):**

      | Kernel | Time | XLA | Hand target |
      |---|---|---|---|
      | M2 M=1 | 96.0 µs | 99.8 | 94.6 |
      | M2 M=16 | 94.5 µs | 103.0 | 94.6 |
      | F5 stack b, L=64 | 87.2 µs/layer, 3.46 TB/s | 95.2 (scan) | |
      | F5 stack w1, L=8 (H=5120, I=2176, ring-only wide variant) | 26.5 µs/layer | 31.4 (scan) | |

      Numerics checks pass.
    * Base snapshot for merging F3 back: `/tmp/mk/snap-f3-base` (before the
      port).
  * **F4/F7/F9b agent launched** in the isolated worktree `helion-f4-kv`
    (base `/tmp/mk/snap-f4-base`, i.e. the state after the m3-prep port).
    Scope:
    * data-dependent loop bounds in megakernel roots;
    * `plan_hbm_resident` (the KV cache stays in HBM with a 2-slot local
      stream);
    * aligned 16-row block writes with a deferred wait (stage 1), and the
      forward-into-slot variant (stage 2) if time allows.
    * Gate: attention-core and GQA-layer deltas on TPU.
  * **Fused GDN layer (`qwen38_gdn_layer`, 9 roots incl. MLP) on the merged
    tree: it compiles and fits now.** Blocker B9 is gone, thanks to
    `put_full_operand_first`.
    * Numerics are exact: out 0, conv_state 0, rec_state 6e-8.
    * Default config: 329 µs chained (was 388; XLA-sized weights are about
      96 MB, i.e. ≈ 32 µs at 3 TB/s).
    * The weights stream through one (128,128) ring. The current
      (dtype, rank) grouping would size a wide-tile slot at 2176×2176, and
      2176 = 17·128 forces 128-wide shared slots, so this waits on F3's
      per-shape rings.
    * Block-size index map for later CFG overrides:
      `[rms m, qkv m/tn/tk, z m/tn/tk, b/a m, out m/tn/tk, post m,
      mlp m/tn/tk, down m/tn/tk]`.
  * **F3 merged into main (per-shape rings plus default stream tiles).**
    * Report: `/tmp/mk/f3/REPORT.md`. The merge was clean except for the
      roadmap, where main's copy was kept.
    * Suites: megakernel 42, test_pallas 240, load_store 118, block_sizes
      13, bound_kernel 11, loop_dependencies 5. All pass.
    * **TPU, default configs only (no CFG overrides):**

      | Kernel | Helion | XLA |
      |---|---|---|
      | M2 M=1 | 96.4 µs | 100.0 |
      | M2 M=16 | 94.4 µs | 102.6 |
      | Qwen-shape MLP stack (H=5120, I=2176), L=8 | 20.7 µs/layer | 31.3 (scan) |
      | F5 stack (H=4096, I=12288), L=64 | 86.8 µs/layer | 95.2 (scan) |
      | Fused GDN layer + MLP | **44.5 µs** (was 329) | 68.7 |
      | Fused GQA layer + MLP, default | 108.2 µs | 67.9 |
      | Fused GQA layer + MLP, ctx block 512 / 1024 / 2048 | **44.6** / 44.7 / 44.7 µs | 67.9 |

      Numerics pass; caches are exact. All times are chained device time,
      so each includes ~5-11 µs of per-call overhead.
    * **M3's first gate (< XLA) is met for both layer types.** Next gate:
      ≤ 37 µs.
    * Open issue: the default block size for fori loops over VMEM-resident
      or local-stream tensors (e.g. the attention ctx loop, block_sizes
      index 10 of `qwen38_gqa_layer`) is 16, which costs 64 µs. This was
      handed to the F4 agent as a generic heuristic: full extent when
      everything is resident, else slots ≥ 256 KB within the VMEM budget.
    * Other gaps to ≤ 37 µs:
      * per-call overhead;
      * whole-KV-cache copy in and out (F4/F7);
      * read-only activations get their own small rings;
      * interpret mode assumes 16 MiB VMEM, so it picks different default
        tiles than the device;
      * root-level stream sites inside folded loops are unsupported (M5).
  * **M5 agent launched** in the isolated worktree `helion-m5-hybrid` (base
    `/tmp/mk/snap-m5-base`, i.e. main after the F3 merge).
    * The kernel: a hybrid Qwen3.8 stack in Helion source. Each period of
      4 runs 3 GDN layers then 1 GQA layer, with per-type stacked weights
      and per-layer caches and states.
    * Generic fixes in scope:
      * root-level stream sites inside folded loops;
      * per-layer state indexed by an affine expression of the folded loop
        variable;
      * several tile shapes per period.
    * Gate: per-layer µs at L=4/8/16. A stack amortizes the per-call
      overhead, so this is the honest number to compare with the
      reference's ~34 µs/layer.
  * **F4/F7/F9b merged into main, plus the `pallas_hbm_resident` knob.**
    * Report: `/tmp/mk/f4/REPORT.md`. Merge base: `/tmp/mk/snap-f4-base`.
    * What landed:
      * HBM-resident KV cache, streamed through a 2-slot local buffer;
      * the row write is a read-modify-write of an aligned 16-row block, with
        a deferred wait, forwarded into the scan's stream slot;
      * data-dependent trip counts (`hl.tile(pos + 1)`);
      * a default loop-block rule (`LoopTileModel`): the whole extent when
        nothing streams; otherwise ≥ 256 KB slots within the VMEM budget.
    * New config key **`pallas_hbm_resident`** (bool | None):
      * None means the compiler default: a tensor stays whole in VMEM when
        it fits next to the other whole tensors, the 8 MiB reserve and
        16 MiB of rings; otherwise it goes to HBM;
      * the autotuner also tries the non-default choice;
      * why: in the layer, HBM KV tiles queue behind the ring's in-flight
        DMAs, while whole-VMEM caches are staged at kernel start and overlap
        the ring prime;
      * the generic fix is F14.
    * Suites: megakernel 48, test_pallas 241, load_store 118, block_sizes 13,
      bound_kernel 11, loop_dependencies 5. All pass. Pyrefly is clean on the
      touched files.
    * **TPU, default configs, main after F4:**

      | Kernel | Chained | Kernel body (profiler) | XLA |
      |---|---|---|---|
      | M2 M=1 / M=16 | 96.2 / 94.4 µs | | 99.8 / 103.0 |
      | Qwen-shape MLP stack, L=8 | 20.8 µs/layer | | 31.4 (scan) |
      | F5 stack, L=64 | 86.8 µs/layer | | 95.2 (scan) |
      | GQA layer + MLP (picks ctx block 2048, caches in VMEM) | **44.7 µs** (was 108.2) | **32.9 µs** | 67.9 |
      | GQA layer, `pallas_hbm_resident=True` | 52.4 µs | | |
      | GDN layer + MLP | 44.5 µs | **32.9 µs** | 68.7 |

    * **The kernel body is already at the reference's per-layer pace.** The
      reference's ~34 µs/layer includes the LM-head share, so its layer is
      about 32.3 µs; 95.7 MB in 32.9 µs is 2.91 TB/s.
    * What remains of the 44.7 µs is per-call harness overhead:
      * pad and slice of the `[1, H]` hidden to 16 rows outside the
        kernel: 3 µs;
      * GDN only: XLA layout copies of `[5120, 6]` and `[1280, 4]` operands
        (the TPU picks transposed layouts for small minor dims): about
        10 µs;
      * gaps between calls in the chained loop.
    * Stacks amortize all of these. Next: the M5 per-layer kernel-body time at
      L = 4/8/16, then M6 (LM head/argmax) and M7 (TP all-reduce, the
      reference's 2%).
  * **M7 survey (read-only): the existing distributed API covers the
    source side of TP.**
    * `hl.make_async_remote_copy(src, src_index, device_id, dst, dst_index)`:
      a one-sided push with `start` / `wait` / `wait_send` / `wait_recv`.
    * `hl.remote_barrier(peers)`: two-phase, with an auto `collective_id`.
    * Both lower to `pltpu.make_async_remote_copy` with mesh device ids and
      compiler-allocated DMA semaphores.
    * Tests: `test/test_remote_copy.py` (one-shot copy, pipelined reuse,
      ring all-gather, writes to remote HBM).
    * So an all-reduce is a Helion-source pattern: push partials to peers,
      wait, sum. No new language feature is needed.
    * In megakernel mode, the root dependency analysis treats a remote copy
      conservatively as a write (`megakernel.py` ~L356). Nothing rejects it.
    * M7 compiler work:
      1. keep the global ring flowing across the exchange: the ring is
         pre-issued, so this is mostly not draining at the root that waits;
      2. place the send/receive buffers;
      3. check that `jax_fn` export composes with `shard_map` over the 8-device
         mesh on the pod.
* 2026-10-02 — **M5 merged into main; F14 validated; F15 launched.**
  * Merge: per-file 3-way of the M5 worktree against its base. Conflicts were
    in the seed heuristic (`seed_limit`/`seed_block_sizes` now take the
    `keep_hbm` budget) and in tests. Suites: 51 / 241 / 118 / 13 / 11 / 5
    passed, all as before plus M5's three tests.
  * TPU, merged main, hybrid stack kernel body (default config, compiler-picked
    ctx block):

    | L | µs/layer | TB/s |
    |---|---|---|
    | 4 | 40.0 | 2.39 |
    | 8 | 38.2 | 2.51 |
    | 16 | 67.7 | 1.41 |

    * `pallas_hbm_resident=True` at L=16: 67.8. The generated code is
      identical, because none of the stack's per-layer state qualifies for
      the HBM path. This is the gap F15 closes.
    * A CTXBLK=512 override at L=16 gave 68.1, so the ctx tile is not the
      issue.
  * F14 probe (GQA layer + MLP, kernel body):

    | config | µs |
    |---|---|
    | caches whole in VMEM | 32.91 |
    | caches in HBM | 36.39 |
    | caches in HBM, KV prime hand-moved into the prologue at its program-order slot | **32.66** |

    * So issuing in DMA order makes HBM state free.
    * The extra 2 × 2.3 µs XLA `copy` ops in device_time in HBM mode come
      from the harness not donating the in-place cache args. They are not
      kernel cost.
  * Launched the F15 + F14 agent (`helion-f15-state`, base
    `/tmp/mk/snap-f15-base`). Targets:
    * L=16/32/64 at ≤ 40 µs/layer;
    * single layers with HBM state at ≤ 33 µs.

* 2026-10-02 — **In-kernel timeline; M6 merged; F16 (stream tiles and ring depths) done.**
  * Timeline tool: `jax.named_scope` around each root, with
    `LIBTPU_INIT_ARGS=--xla_enable_custom_call_region_trace=true`. The scopes
    show up as events on the "XLA TraceMe" line of `/device:TPU:0`, giving
    per-root µs inside the megakernel (`/tmp/mk/m5/timeline.py`,
    `run_tl.sh L CFGFILE`).
  * Finding: weight-DMA bandwidth follows Little's law, per ring.
    * During a root only its own ring has copies in flight, because every
      other ring sits full of prefetched tiles.
    * Qwen MLP stack, gate/up root, by ring depth (slot 0.56 MB):

      | depth | 4 | 6 | 8 | 12 | 16 | 32 |
      |---|---|---|---|---|---|---|
      | µs | 31.85 | 21.86 | 16.73 | 13.14 | 12.26 | 12.02 |

    * Fit: `BW(x) ≈ 3.72 TB/s · (1 − exp(−x / 2.4 MB))`, with
      `x = (depth − loads) · slot`.
  * M6 merged into main: 3-way per file, 2 add/add conflicts in tests.
    Suites 55 / 241 / 118 / 13 / 11 / 5 passed.
  * F16(a), seed: prefer contiguous tiles, then the smallest slot that
    reaches `_STREAM_MIN_SLOT_BYTES`. The same ring bytes in smaller slots
    leave more in flight, and same-width sites (down / out / o) share a
    ring. Hybrid L=4: 40.2 → 34.4 µs/layer.
  * F16(b), depths: greedy marginal allocation.
    * Every ring starts at two iterations' loads. Each further slot goes to
      the ring with the largest stall reduction per byte, where
      `stall = Σ_runs max(T_run − depth·slot, 0) / expm1(x / 2.4 MB)`.
    * A run is a maximal stretch of consecutive roots that read the ring.
      The ring sits full between runs, so its first `depth` slots land
      before the run starts.
    * Cap per ring: `max(4·loads, 16 MiB)`. `pallas_stream_depth` pins the
      primary ring and the rest is allocated greedily.
    * The run grouping matters for rings shared by consecutive roots. A
      single 128×128 ring read by every root of a folded MLP stack never
      sits full, so it keeps its whole stream. Split per tile shape, each
      ring stops at one layer's tiles.
  * TPU, merged main, kernel body:

    | kernel | before F16 | F16(a) | F16(a)+(b) |
    |---|---|---|---|
    | hybrid L=4 | 40.2 | 34.4 | **33.3** µs/layer |
    | hybrid L=8 | 38.2 | 36.6 | **33.0** µs/layer (2.90 TB/s) |
    | Qwen MLP stack L=8 | 20.1 | 20.1 | 20.09 |
    | M2-shape stack L=8 | 87.7 | 87.6 | 87.86 |

    * Hybrid L=8 rings: (8,256,1280), (6,512,768), (6,128,5120),
      (17,128,2176), (5,256,1536), (4,1024,256), plus the ragged b/a and
      conv rings.
    * Per root: gate/up 15.5 → 13.1 µs, down 7.4 → 6.0 µs.
    * L=8 includes about 1.25 µs/layer of cold start (F19), so the steady
      state is ≈ 32 µs/layer. That is the reference's 32.3 µs/layer at TP8
      byte volume.
  * Projection: with F15 freeing state VMEM (~50 MB of rings), the model
    gives a stream time of ≈ 25.9 µs/layer. Add compute and handoffs and the
    total is ≈ 30 µs/layer.
  * Next:
    * merge F15+F14 when the agent lands, then hybrid L=16/32/64;
    * F17: padded accounting, or transposed storage for narrow-minor
      weights;
    * F19: program-order prologue.

* **2026-10-03:**
  * **M6 full step compiles and is correct on TPU.**
    `examples/qwen38_decode_step.py` runs as one kernel: embedding row →
    hybrid layers (folded period loop) → final RMSNorm → streamed LM-head
    top-1.
    * Interpret mode at small shapes: top-1 matches the reference.
    * TPU kernel body:

      | L | body | top-1 | sum of parts |
      |---|---|---|---|
      | 4 | 235.2 µs (3.02 TB/s) | OK | 4 × 33.3 + 101.7 = 235 µs, exact |
      | 8 | 396.4 µs (2.76 TB/s) | OK | 366 µs; the step is ≈ 30 µs over it |

    * Timeline at L=8: the LM-head root takes 113.8 µs, against 101.7
      standalone, and gate/up is 13.55 µs against 13.1. Every ring is
      shallower than in the stack alone. The head ring is (3, 5120, 256),
      which is only 5.2 MB in flight, and the stack rings each lose 1–4
      slots. The cause is the VMEM budget: whole-VMEM per-layer state grows
      with L, so this is F15's problem, not a new one.
  * **M8 design note** (`/tmp/mk/m8/REPORT.md`).
    * Kimi never speculates on experts. Every compute gap is covered by DMAs
      that were known statically, plus one exact early issue inside top-k.
    * Our probes show that tensor-indexed expert weights fall back to
      synchronous per-tile DMAs. The shared expert already streams, and
      top-k and routing-weighted accumulation already work.
    * New feature rows F20-0, F20a, F20b, F20c and F21.
    * The M8a implementation agent (F20-0, F20a, F20b, with a TPU layer
      against XLA) runs in worktree `helion-m8-moe`.
  * Tried a narrower LM-head tile, (5120, 128) instead of (5120, 256), so the
    same VMEM gives a deeper ring: 6 slots, 6.5 MB in flight. It is **worse**:
    the head goes from 113.8 to 153.3 µs (2.14 TB/s). Strided tiles with
    256-byte rows lose DMA efficiency no matter how much is in flight, which
    confirms F3's rule of N ≥ 256 (512-byte rows). The bandwidth model
    should take row bytes into account, not only slot bytes. Recorded for
    F16 follow-up.
  * **F15 + F14 (worktree `helion-f15-state`, base `/tmp/mk/snap-f15-base`):
    per-layer state in HBM, issued in the global DMA order.**
    * What landed:
      * `_hbm_resident_tensors` now accepts:
        * whole-slice loads and stores in a root's own body at scalar leading
          indices, which may be affine in the folded layer loop;
        * ragged minor dims (conv state `[L, C, 4]`) when every access is a
          whole slice; these copy through a `[rows, cols]` reshape view
          (`ragged_rows_ref`). Mosaic cannot slice the 3-D ref.
      * Loads (`MegakernelPlan.root_slice_load`, `_plan_hbm_copies`): a DMA
        into a VMEM slot of their own plus a wait.
      * Stores (`_emit_hbm_resident_store`): an async DMA kept as a pending
        write. `MegakernelPlan.finish_root` carries the writes of a
        single-tile root, to tensors the next root does not use, to the end
        of that next root.
      * F14 (`_plan_early_copies`, `_early_stores_safe`,
        `_loop_prime_copies`, `LoopPrime`, `_early_statements`): each slice
        load and each local-stream prime starts, per folded iteration, just
        before the first ring copy of a stream index read after it.
        * Conditions:
          * the address depends only on constants, the layer and scalars;
          * the slot is free (ring copies started before that are passed
            over);
          * every store to the source in between, including writes still
            pending from the previous root, is disjoint by an integer index
            or is a row write that the inner loop forwards into its tile.
        * Otherwise the copy starts where the program reaches it (loop-head
          prime, or the load itself). The iterations are emitted as
          `pl.when` runs in the refills and in the prologue.
      * Defaults:
        * `_default_hbm_resident`: with rings, HBM when the slices' slots
          and store stages (`_hbm_slot_bytes`) take less VMEM than the whole
          tensors would (twice their bytes). The `StreamModel`'s
          `resident`, `stream_budget`, `seed_limit` and `config_error`
          count the slots, not the whole tensors.
        * The seed (`get_seed_config`) also halves while
          `StreamModel.rings_shallow`, i.e. while the deepest-traffic ring
          holds fewer than four iterations' loads at its default depth.
        * `LoopTiles.behind_rings`: an inner static-extent loop that streams
          while the rings stream later sites defaults to its whole extent
          within the VMEM budget. Its later copies would queue behind the
          rings, so it gets one early prime per layer (ctx 2048 for the
          GQA cache).
    * Tests (interpret): `test_hbm_resident_layer_state` (folded stack with
      per-layer conv and recurrent state, ragged minor dim) and
      `test_hbm_resident_layer_caches` (folded stack with per-layer KV caches
      written at `pos` and scanned in a ctx loop). Each runs depths None and 2
      and checks numerics, exact states, and where the early copies land.
    * Suites: megakernel 53 (51 + 2 new), test_pallas 241 / 33 skipped / 10
      xfailed, load_store 118 / 1 / 60, block_sizes 13, bound_kernel 11 / 8,
      loop_dependencies 5 / 15. All match the baselines.
    * **TPU, hybrid stack, default configs (numerics PASS at every L):**

      | L | Before: kernel body | After: kernel body | After: device time |
      |---|---|---|---|
      | 4 | 40.0 µs/layer | **33.3** | 37.4 |
      | 8 | 38.2 | **32.1** | 34.2 |
      | 16 | 67.7 | **31.2** | 32.4 |
      | 32 | not measured | **30.9** | 31.5 |
      | 64 | not measured | **32.7** | 33.1 |

      * The gain has two sources:
        * HBM state (forced, old tiles) at L=8 is 32.6 vs 38.2 for VMEM;
        * at L=16 the whole state no longer squeezes the rings.
      * Before `finish_root`, the run was 34.3 / 33.1 / 32.3 / 31.9 / 33.9
        µs/layer.
    * **TPU, single layers (kernel body, chained device time):**

      | Layer | State in HBM | State whole in VMEM |
      |---|---|---|
      | GQA + MLP (default: HBM, ctx block 2048) | **32.7 µs**, 48.7 | 33.0, 44.4 |
      | GDN + MLP (default: VMEM; slots = whole) | **32.8 µs**, 48.2 | 32.9, 44.4 |

      * In HBM, the bodies match VMEM-whole; before F14, HBM cost +7.8 µs.
      * Chained device time is ~4 µs higher in HBM because the harness does
        not donate the aliased cache and state buffers, so XLA copies them
        around the call. A decode loop that donates them avoids this.
      * With the old ctx block of 512 (T = 4 tiles per layer, 2 slots), the
        GQA layer in HBM is 37.6. The new default streams one tile per
        layer, so more local-stream slots are not needed by default. They
        would matter where the cache does not fit as one tile.
    * Open:
      * L=64 is 1.8 µs/layer above L=32 (not yet profiled);
      * with deep default rings in small kernels, later layers' copies
        cannot find a free slot early and fall back to loop-head primes;
      * interpret mode cannot DMA into a reshape view, so ragged stores are
        written through `.at[...]` there. The TPU-only view path is covered
        only on TPU;
      * GDN `SPLIT=1` hits a Mosaic error (pre-existing);
      * more slots for multi-tile local streams (T > 1).
  * **F15 merged into main.** Merged hybrid stack at L=64: 30.9 µs/layer
    kernel body (1976 µs, 3.10 TB/s), numerics PASS. That is below the
    reference's ≈ 34 µs/layer `noexchange` (head share included).
  * **Zoo: Llama/Qwen3 dense stack** (`examples/llama_decode_stack.py`,
    harness `/tmp/mk/zoo/run`). It was written as plain source, with no
    Qwen-specific code. It needed one generic fix: nested subscripts on a
    view (`x[i][:, j]`) in `view_ops.py`. TPU results (H 4096, I 1792,
    GQA 32/8 per-rank shapes, ctx 2048, numerics PASS):

    | L | kernel body | device time | XLA unrolled | XLA scan |
    |---|---|---|---|---|
    | 8 | 16.98 µs/layer (3.21 TB/s) | 18.91 | 47.55 | 35.53 |
    | 32 | **16.29 µs/layer (3.35 TB/s)** | 16.81 | 35.02 | 37.35 |

    It is 2.1× faster than XLA, at the HBM roofline, with nothing tuned
    for this model. That is the first evidence that the features
    generalize beyond Qwen3.8.
  * **Full step OOM at L=8 after the F15 merge** ("Used 65.09M of 64.00M").
    The generated scratch summed to 63.25 MiB. Two VMEM users were
    invisible to the ring budget:
    * **Inner-loop DMA buffers.** A `behind_rings` loop (the KV ctx scan)
      takes `2 × tile` per streamed tensor: 4 × 2 MiB at ctx 2048. The
      rings were sized as if that VMEM were free, so the 8 MiB reserve
      covered it only by luck. Fix: `LoopTileModel.buffer_bytes` and
      `StreamModel.loop_buffer_bytes(tuned, keep_hbm)` count it at the
      config's block sizes, next to the resident tensors.
      `resolve_ring_depths` now takes every tuned block size, not only the
      stream ids. Test: `test_rings_leave_loop_buffers`.
    * **Loop-carried accumulators per root.** Every root registered its own
      `scratch_i` for each loop-carried value, though roots run one after
      another and a carried value dies with its loop. Fix (F8-lite):
      `MegakernelPlan.share_root_scratch` runs after codegen. A root's
      k-th carried scratch of a given shape and dtype aliases the k-th one
      any earlier root registered (`scratch_7 = scratch_2` in the
      preamble). Test: `test_roots_share_carried_scratch`.
    * Result (interpret codegen, L=8 step): 63.25 → 55.81 MiB of scratch.
      Accumulators went from 4.3 to 0.9 MiB, and the rings gave 3.5 MiB to
      the KV buffers. TPU check pending.
  * **F17-lite: narrow rings keep only their longest run** (`_ring_runs`,
    `_ring_floor`). A ring whose tiles are narrower than 128 lanes is
    mostly lane padding, e.g. GDN `b`/`a` (5120, 6), 61 KB real in a
    1.25 MiB slot. Its floor is now the longest run of consecutive reads,
    not two iterations' loads: conv weights go 2 → 1 slot, `b`/`a` 4 → 2.
    At L=8 that gives 2.8 MiB back to the wide rings: gate/up 16 → 17
    slots, and qkvz, out-proj and GQA rings one slot each.
    * A first version applied this to every ring. In `gated_stack`
      (whole-tile roots everywhere) it dropped all three rings to depth 1.
      The F16 cost model calls that free, since each ring's run fits.
      But when every run is short and the program is DMA-bound, the
      second iteration's slots are what keep tiles in flight. So wide
      rings keep the 2× floor. This is a known limit of the model: it
      treats rings separately and does not count the bytes in flight
      across rings over a period. Recorded for F18.
    * Test: `test_narrow_ring_floor_follows_runs` (gated stack with narrow
      vs wide gates; tap stack with one vs two tap layers in a run).
      `test_seed_shrinks_to_fit`'s gated case moves to 13 MiB, because the
      first seed now fits at 14.
    * Interpret suites green after F17-lite: megakernel 62, test_pallas 241,
      load_store 118, views 12, memory_access 4.
  * **Finding: narrow-minor operands cost ~20 µs per call outside the
    kernel** (old step profile, L=8: device total 416.45 µs vs kernel body
    396.4). TPU's default layout for an array whose minor dim is tiny is
    minor-swapped (`bf16[6,5120,6]{1,2,0}`, `bf16[6,1280,4]{1,2,0:T(4,128)}`).
    Pallas needs row-major operands, so XLA copies GDN `b`/`a` (6.7 + 5.4 µs),
    the conv weights and conv state in (2.6 µs each), and the conv state
    back out (2.6 µs), on every call. This grows with L: about 160 µs per
    token at 64 layers. The row-major copy also pads the minor dim to 128
    lanes in HBM, so each DMA of a `(5120, 6)` tile moves 1.3 MB for 61 KB
    of data. That is about 2.6 MB of extra HBM traffic per GDN layer.
    * The patched reference we time already keeps conv state and conv
      buffers lane-dense (`(taps, conv_dim)`), and it folds `a`/`b` into the
      big projection.
    * Plan, F17b (lane-dense operands): for a kernel operand whose minor dim
      is narrow and whose second-minor dim is lane-aligned, the launcher
      passes `swapaxes(x, -1, -2)`. Under the default layout that is a
      bitcast, not a copy. In-place outputs are swapped back the same way.
      Inside the kernel every access uses the swapped index, and DMA slots
      and VMEM buffers take the lane-dense shape. Loaded values are
      transposed back at the load, and stored values at the store. A dot
      whose RHS comes straight from such a load contracts on the RHS minor
      dim instead (an NT matmul), so no transpose is emitted. Transpose
      sinking through elementwise ops and reductions (the conv window) is
      left for later. TPU probe `/tmp/mk/lane/lane_probe.py` checks the
      bitcast, the NT dot and the in-kernel transpose.
  * **M6 gate met (TPU, main before F17-lite, 2026-10-03).** Full decode step
    at L=64: 2183.2 µs chained device time, kernel body 2163.7 µs, top-1
    token matches XLA. The gate is ≤ 2.40 ms; the reference `noexchange` is
    2.29 ms. L=8: device 377.6 µs, body 363.7 µs (was 396.4 before F15 and
    the two VMEM fixes; no OOM). Hybrid stack: 32.4 µs/layer at L=8 and
    31.3 µs/layer at L=64, numerics PASS. Logs: `/tmp/mk/m5/merged_prefloor.log`;
    on the pod, `/root/m5/m{step,hyb}_L{8,64}_prefloor.log`. The remaining
    work toward the reference's 2.33 ms is TP8 (M7). Its exchange costs the
    reference 40 µs.
    * TPU probe results (`/tmp/mk/lane/probe.log`). The default layouts of
      `(6,5120,6)`, `(6,1280,4)` and `(6,5120,64)` bf16 are all
      `major_to_minor=(0,2,1)`. Calling the kernel directly gives 4 copies
      (2 in, 2 out for the aliased state). Calling it through `swapaxes`
      gives 0 copies, with identical results. Both the NT `dot_general` and
      `jnp.dot(x, w_t.T)` lower in Mosaic and are exact.
  * **Merged-F15 TPU run, before the VMEM fixes** (`/tmp/mk/merge-f15/tpu.log`,
    `tl_32_64.log`; tree /root/helion-merged):
    * Step at L=8 and at L=64: VMEM OOM at compile (65.09M of 64M at L=8).
      Fixed afterwards by counting inner-loop buffers and sharing root
      scratch; the fixed step runs at 2183 µs at L=64 (entry above).
    * Hybrid stack at L=64: 1976.2 µs kernel body (30.9 µs/layer,
      3.10 TB/s), numerics PASS.
    * Hybrid stack at L=32: VMEM OOM (64.18M of 64M, over by 181 KiB). It
      had no timeline, so the earlier open item "L=64 is 1.8 µs/layer
      above L=32" is still unexplained. Re-check L=32 on current main,
      which counts loop buffers and shares root scratch.
    * Hybrid L=64 timeline: kernel 1978.2 µs over 581 scope events. Only
      1.34 µs of gaps between scopes plus a 1.41 µs tail, so the roots
      run back to back. Per stage (lines of `examples/qwen38_hybrid_stack.py`):

      | stage | total | count | mean per layer |
      |---|---|---|---|
      | MLP gate/up (L244) | 874.3 µs | 64 | 13.66 µs |
      | MLP down (L253) | 360.3 | 64 | 5.63 |
      | GDN qkv proj (L179) | 225.6 | 48 | 4.70 |
      | GDN z proj (L185) | 107.0 | 48 | 2.23 |
      | GDN out proj (L232) | 86.9 | 48 | 1.81 |
      | GQA q proj (L104) | 81.4 | 16 | 5.09 |
      | GDN conv + gated delta rule (L199) | 71.8 | 48 | 1.50 |
      | GQA out proj (L172) | 28.9 | 16 | 1.81 |
      | GDN `a`/`b` (L191) | 27.0 | 48 | 0.56 |
      | MLP norm (L239) | 26.3 | 64 | 0.41 |
      | attention norm (L97) | 24.3 | 64 | 0.38 |
      | GQA v proj (L114) | 23.4 | 16 | 1.46 |
      | GQA attention over the cache (L121) | 21.3 | 16 | 1.33 |
      | GQA k proj (L109) | 15.6 | 16 | 0.97 |

    * Reading it: the projections are weight streams, and per-stage time
      tracks bytes, not FLOPs. Gate/up is 44.6 MB per layer in 13.66 µs
      (3.26 TB/s). Down appears faster than peak (22.3 MB in 5.63 µs)
      because its first slots are primed during the preceding norm and
      gate/up. The non-streaming work is small: conv plus recurrence is
      1.50 µs per GDN layer, attention over the cache 1.33 µs per GQA
      layer. GDN `a`/`b` costs 0.56 µs per layer for 122 KB of real
      weights, because its DMAs move 2.6 MB of 128-lane padding. F17b
      targets that.
    * Default-layout probe (`/tmp/mk/lane/layout.log`). XLA TPU picks the
      layout with the smallest tiled footprint, and ties go to row-major.
      * 2D `(5120, n)` is swapped for n ≤ 96 and also for n = 130 and 192.
        It is row-major for n = 127, 128 and 256.
      * Row tiles shrink with the second-minor size. For bf16 they are
        (2,128), (4,128) and (8,128) with (2,1) packing.
      * The choice is a full permutation, not only a swap of the last two
        dims. `(8,6,5120,6)` becomes `(1,3,0,2)`, physical `[6,6,8,5120]`:
        the leading 8 fills the sublane tile.
      * So F17b has to mirror XLA's permutation: a function from shape and
        dtype to a perm. At the call the launcher passes
        `jnp.transpose(x, perm)`, which is a bitcast. Inside the kernel,
        index components follow the perm. Applying it when the predicted
        layout differs from XLA's would introduce a copy instead of
        removing one.
      * Layer-stacked shapes `[L, K, n]` for L = 1…64 are being probed
        (`layout_probe2.py`).
  * **F17-lite on TPU (main, 2026-10-03; `/tmp/mk/m5/merged_floor.log`).**
    Full step: L=8 at 374.9 µs device / 361.0 µs body (before F17-lite:
    377.6 / 363.7). L=64 at 2173.8 / 2154.6 (before: 2183.2 / 2163.7).
    Hybrid stack: L=8 at 258.3 µs (32.3 µs/layer), L=64 at 1994.5 µs
    (31.2 µs/layer). Numerics PASS everywhere, top-1 OK.
    * **Reserve sweep** (step L=64, `_STREAM_RESERVE_BYTES` changed by
      hand): 8 MiB gives a 2154.6 µs body; 4 MiB gives 2117.4 µs (device
      2136.3), −37 µs (−1.7%); 3 MiB gives 2121.6 µs (device 2141.6).
      The reserve is VMEM that no ring gets. At L=64 with 3 MiB, sized
      scratch is 59.12 MiB, and only about 1.6 MiB of that is
      accumulators and intermediates (`/tmp/mk/step/scratch_sum.py` on
      `gen_L64_floor_r3.py`). So most of the 8 MiB was unused slack.
      Below 4 MiB the extra slots stop helping: the gate/up ring is
      already past the depth where the stream is saturated.
    * Next (F22, VMEM accounting): count the scratch the plan does
      allocate but does not size today (loop-carried accumulators, after
      F8-lite sharing, and small intermediates) inside the budget, and
      keep only a fixed margin for Mosaic-internal scratch. The candidate
      margin is 4 MiB; check that no current TPU kernel OOMs with it.
    * Unshared HBM-resident state buffers: each root that touches a
      `keep_hbm` state gets its own VMEM staging buffer. The L=64 step has
      6× the recurrent state ([6,128,128] f32, 2.25 MiB in total), 7× the
      [1280,4] conv buffer (2.19 MiB) and 2× KV (4 MiB). A staging ring
      shared across roots would free about 3 MiB. F17b also shrinks the
      [1280,4] and (10240,6) buffers by about 4 MiB.
* 2026-10-03 — **F22 (VMEM accounting); M7 and M8a merged; step timeline; layout rule.**
  * **Step L=64 timeline** (main before F22, `/tmp/mk/m5/step_tl64.log`):
    kernel 2154.8 µs, gaps 1.31 µs, tail 1.21 µs.

    | Stage (line in `qwen38_decode_step.py`) | µs total | µs per layer |
    |---|---|---|
    | gate/up (L258) | 880.9 | 13.76 |
    | down (L267) | 383.8 | 6.00 |
    | GDN qkv (L193) | 224.9 | 4.69 |
    | GDN z (L199) | 128.0 | 2.67 |
    | LM head (L277) | 114.1 | (once) |
    | GDN out (L246) | 86.9 | 1.81 |
    | GQA q (L118) | 86.8 | 5.43 |
    | GDN conv + recurrence (L213) | 71.3 | 1.49 |
    | GQA v (L128) | 30.1 | 1.88 |
    | GQA out (L186) | 29.2 | 1.83 |
    | GDN a/b (L205) | 26.5 | 0.55 |
    | MLP norm (L253) | 26.3 | 0.41 |
    | attention norm (L111) | 24.4 | 0.38 |
    | attention over the cache (L135) | 21.4 | 1.33 |
    | GQA k (L123) | 15.6 | 0.97 |

    Against the hybrid stack at L=64 (1994.5 µs), the step pays about
    46 µs on top of the embedding and LM head. The extra time is in down
    (+23.5 µs), z (+21), v (+6.7), gate/up (+6.6) and q (+5.4), all weight
    streams. The step has more state (embedding row, LM head shard) in
    VMEM, so its rings are shallower. The reserve sweep (−37 µs at 4 MiB)
    fits that, and F22 is the generic version of the sweep.
  * **F22 done in main (interpret).** `CarriedScratch` (built in
    `enter_sequential_roots_mode` from each root's `_for_loop` carried
    values) is sized per config and counted next to the inner-loop buffers
    in `_resolve_stream`. `_STREAM_RESERVE_BYTES` is 3 MiB. Tests:
    `test_rings_count_carried_scratch`. The depth goldens use the
    constant. `test_seed_shrinks_to_fit` now runs at 8 MiB of VMEM (the
    same 5 MiB of rings as 13 MiB with the old reserve). Known risk: the
    config-independent uses of the reserve (`_default_hbm_resident`, the
    "does not fit" check) do not count carried scratch. Kernels without
    rings now get 5 MiB more and rely on the 3 MiB margin.
  * **Default-layout rule, confirmed on stacked shapes**
    (`/tmp/mk/lane/layout2.log`). Over every permutation, pad the minor
    dim to 128 and the second-minor dim n to `max(min(8, next_pow2(n)),
    2 for bf16 / 1 for f32)` rows, and take the smallest footprint. Ties
    go to the permutation with the fewest inversions. It predicts every
    probe:
    * `(L, 5120, 6)` bf16: `(2,0,1)` for L ∈ {2,4,7,8,16,24,48,64},
      `(0,2,1)` for L ∈ {1,3,6,12}.
    * `(L,1280,4)`, `(L,5120,64)`, `(L,2048,8)`: always `(0,2,1)`.
    * `(48,2,5120,6)`: `(0,3,1,2)`. `(4,3,5120,6)`: `(1,3,0,2)`.

    F17b mirrors this rule. It is unit-tested against the whole table and
    passes operands unchanged when the prediction is uncertain.
  * **M8a merged** (F20-0, F20a, F20b, SMEM index mirror; numbers in the
    M8 row). The 6 conflicts with main were in `MegakernelPlan` (new
    `index_mirrors` next to main's early-copy fields), `prologue` (mirror
    fills after the early row reads), `root_refills` (mirror refreshes
    first) and `_scalar_index` (the `IndexRead` branch, then main's
    `_layer_index`).
  * **M7 merged (partial).** Compiler side: a scratch store indexed by a
    0-d tensor drops the dim; all megakernel scratch is VMEM
    (`_scratch_lives_in_vmem`); remote copies move only the logical rows
    (`logical_region_parts`, `_region_expr`); a transposed tile used only
    as a dot's rhs folds into `dimension_numbers`. Findings: the exchange
    costs a ~4.3 µs/layer latency floor. Deeper rings delay it, because
    remote copies queue behind refills (now F23). Transposed (NT) dots on
    tpu7x cost 1.25–1.7× in a throughput microbenchmark, so an [N, H]
    weight layout makes gate/up compute-bound. That matters for F17b:
    keep NT dots to the narrow GDN `a`/`b` and conv operands.
  * **TPU check of the merged main (F22 + M8a + M7; `/tmp/mk/m5/v78.log`,
    pod tree `/root/helion-v78`).** Everything passes numerics, and top-1
    matches. F22 is a win wherever VMEM is tight:

    | Workload | Before F22 (body µs) | Merged main (body µs) |
    |---|---|---|
    | Step L=64 | 2154.6 (device 2173.8) | **2127.2 (device 2148.2)** |
    | Step L=8 | 361.0 | 360.2 (device 376.1) |
    | Hybrid L=8 | 258.3 | 256.7 (32.1 µs/layer) |
    | Hybrid L=32 | OOM (merged-F15) | 989.9 (30.9 µs/layer) |
    | Hybrid L=64 | 1994.5 | 1981.3 (31.0 µs/layer, 3.09 TB/s) |
    | Llama zoo L=8 | — | 136.0 (17.0 µs/layer, 3.21 TB/s; device 151.3) |
    | MoE Qwen-30B-like, in kernel | 29.29 (M8a tree) | 29.40 |
    | MoE, ids as inputs | 25.77 | 25.76 |
    | MoE DSV2-Lite-like | 55.21 | 55.01 |

    The step at L=64 is now 2.148 ms device time on one core, against the
    reference's 2.29 ms `noexchange`. The step costs 146 µs over the hybrid
    stack, of which the LM head is 114. The rest of the TP gap is the
    exchange (M7b).
  * **Bandwidth accounting at L=64 (step timeline, per-rank weights).** A
    GDN layer streams 95.8 MB in 31.8 µs, 3.02 TB/s, the same as the
    reference's `noexchange` (3.04 TB/s). Per-stage bandwidth varies
    (qkv 2.79, z 2.94, GQA q 2.90, GQA v 1.39, gate/up 3.24, down/out
    3.7–4.3). That only shows which stage's copies wait behind which:
    refills are issued in program order, so a stage's refills queue behind
    the refills its predecessor issued. What matters is when the DMA queue
    runs dry.
    * **Why rings lose time at every non-streaming root.** Each consumed
      tile issues one refill, D tiles ahead, into its own ring. Every ring
      except the one just consumed is full of landed tiles waiting for
      their roots. So during a compute-only root (norm, conv + recurrence,
      attention, the TP exchange) the only queued DMA is the in-flight part
      of the previous ring (≈ its depth × slot bytes). A bubble longer than
      that queue idles the engine.
    * Most ring VMEM therefore holds tiles that have landed but whose root
      has not started; little of it is queue. A program-order ring over one
      arena (F18) turns all of its free slots into queue. That is the
      reference's design, and it hides the 4.3 µs TP exchange behind it
      (M7b).
* 2026-10-03 (cont.) — **Single-core loss budget, TP8 step harness, DMA-only probes.**
  * **Where the 2127 µs of the L=64 step body goes** (per-rank bytes at
    3.35 TB/s would be about 1.93 ms):
    * GDN layer: 31.8 µs against an ideal 28.6. The per-stage excess
      over 3.35 TB/s sums to the gap:

      | Stage | Excess per layer |
      |---|---|
      | qkv | +0.78 µs |
      | z | +0.32 |
      | a/b (padded DMA) | +0.55 |
      | conv + recurrence | +1.49 |
      | norms | +0.79 |
      | gate/up | +0.45 |
      | out-proj | −0.54 |
      | down | −0.66 |
      | **Total** | **≈ 3.2 µs** |

      The DMA is idle for nearly the whole of each non-streaming root.
    * LM head: 114 µs against ~98 ideal.
  * **Out-proj tiles arrive before out-proj starts.** Its 6 tiles are
    refilled at the end of the previous layer's `down`, which shares the
    `(128, 5120)` ring with 7 slots. Yet out-proj still takes 1.81 µs for
    7.9 MB, so with no DMA wait the GEMV itself runs at about 4.3 TB/s.
    * **Hypothesis:** M=16 GEMV on tpu7x (VMEM → vreg → MXU weight push)
      is only about 30% faster than HBM. A streaming root whose tiles are
      already in VMEM is then far from free, and DMA/compute overlap
      matters, not only DMA bubbles.
    * **Probes queued:**
      * `patch_nodot_all.py`: every 2-D dot becomes a 16-row read, so the
        run measures the DMA schedule alone.
      * `patch_lmh_nodot.py`: the same for the LM head only.
      * `lmhead_sweep.sh`: LM head vocab tile 256/512/640/1280.
      * Logs: `/tmp/mk/m5/{nodot_all,lmh_nodot,lmh}.log`.
      * `gemv_probe.py` (`/tmp/mk/gemv`): weights fully in VMEM, swept
        1000 times, for MXU M=16, MXU M=8 and VPU multiply-reduce. It gives
        the VMEM-fed GEMV rate directly.
    * **DMA-only result** (`nodot_all`: every 2-D GEMV dot replaced by a
      16-row read of its weight tile; attention and the other compute
      kept):

      | Step body | Normal | No GEMV dots | Delta |
      |---|---|---|---|
      | L=8 | 360.2 µs | 340.7 µs | −19.5 µs |
      | L=64 | 2127.2 µs | 1988.3 µs (3.25 TB/s) | −139 µs, ≈2.2 µs/layer |
      | L=8, LM head dot only (`lmh_nodot`) | 360.2 µs | 357.0 µs | −3.2 µs |

      * **The ring/DMA schedule is nearly at roofline.** Alone it is
        ~58 µs off 3.35 TB/s at L=64.
      * **The main loss is GEMV compute not hidden behind DMA**, about
        139 µs. About 90% of the ~22 µs/layer of GEMV work (at the
        ~4.3 TB/s VMEM-fed rate) already overlaps. What remains fits the
        tail of each streaming root: its last tile's compute,
        ~0.3–0.6 µs × ~8 roots/layer.
      * This reorders the single-core priorities:
        * **GEMV rate and the tail come first.** Faster M=16 GEMV
          codegen, e.g. the weight as the streamed MXU operand, not the
          stationary one; or a smaller last tile per root.
        * F18 (program-order arena) is worth at most ~58 µs single-core.
          It still matters for hiding the TP exchange.
      * The LM head GEMV costs only 3 µs. Its 114 µs against 98 ideal is
        DMA, the ring squeeze and prologue below.
    * **LM head tile sweep result (step L=8 body):**

      | Vocab tile | Body |
      |---|---|
      | 256 (default) | 360.2 µs |
      | 512 | rejected: must divide 32000 |
      | 640 | 372.4 µs |
      | 1280 | 390.8 µs |

      Wider, more contiguous slabs are monotonically *worse*, so strided
      `(5120, tv)` DMA is not what limits the LM head.
      * Likely cause: the LM head ring is allocated for the whole kernel
        but used only in the tail. A wider tile takes VMEM away from the
        layer rings for all L layers (+30 µs at L=8 is ~3.8 µs/layer).
      * Generic follow-up: **phase-scoped ring VMEM**. A ring whose
        consumers all come after a loop should reuse the arena of rings
        that are dead by then. This is F8-lite applied to rings: the
        allocator needs live ranges in program order, not
        whole-kernel-lifetime slots.
      * **Confirmed by the compiler's own ring choice**
        (`/tmp/mk/depths/depths.py`: codegen only, hooks
        `resolve_ring_depths`; step at L=8, default config):

        | Vocab tile | LM head ring | qkv | out/down | gate/up | k/v | Layer rings total |
        |---|---|---|---|---|---|---|
        | 128 | (5120,128) ×8, 10.5 MB | ×10 | ×7 | ×19 | ×6 | 42.6 MB |
        | 256 | (5120,256) ×5, 13.1 MB | ×9 | ×6 | ×18 | ×6 | 40.1 MB |
        | 1280 | (5120,1280) ×2, 26.2 MB | ×6 | ×4 | ×12 | ×3 | 26.7 MB |

        The LM head ring takes 25% of ring VMEM at the default tile. It
        does need about 4 slots in flight *during its phase* (F16: 2.6 MB
        in flight is only ~2.1 TB/s). But it holds them for the whole
        kernel. The greedy `_ring_depths` prices that VMEM as if it were
        free.
      * Blocker: Mosaic's `memref_reshape` is a logical row-major
        reshape. Reinterpreting one byte arena as rings with different
        minor dims is not a byte identity in the (sublane, 128) tiled
        layout. Rings of equal minor dim can share one arena: merging
        tile-aligned second-minor dims, e.g. `(S, 128, C)` as
        `(S/2, 256, C)`, is a byte identity. Shares across minor dims
        would need the DMA to write a canonical `(n, 16, 128)` tile
        layout, plus a vreg-renumbering load. Parked as second-order:
        about 16 µs of LM head excess plus a few µs/layer of squeeze at
        L=8. At L=64 the layer rings are mostly at their caps already.
  * **TP8 end-to-end step** (`/tmp/mk/tp8`). `tp_step.py` is the decode
    step with the M7 all-reduce after the mixer and after the MLP, plus
    an all-gather of each rank's top-1 (`tops[world, 2, m]`, one remote
    copy per peer) and a global argmax with the lowest index on ties.
    Plain Helion source. `run_tp_step.py` checks it against the qwen
    oracle with a psum per residual add. Codegen works in interpret mode,
    but interpret mode cannot execute remote DMA under shard_map. First
    TPU run (L=8/64, exchange vs no exchange) queued.
    * **First TPU run fixes:**
      * The `tops` remote copy was rejected by Mosaic: a copy's tiled dims
        must be whole (8, 128) tiles. `tops` is now `[world, 8, 128]` f32
        (value in row 0, global index in row 1), and each copy moves one
        whole `tops.at[me]` tile.
      * All four runs then hit a VMEM OOM (64.59 of 64 MiB). Root cause:
        a compiler accounting bug, fixed below. Workaround until the fix
        reaches the pod: `RESERVE_MIB=6`. Rerun (L=8/64, exchange vs no
        exchange) queued, `/tmp/mk/tp8/rerun8.log`.
  * **F22b: tile overhang in the ring budget** (main worktree, unit
    tested).
    * Bug: `_whole_vmem_bytes` counted each tensor kept whole in VMEM at
      its own shape, padded to one sublane tile. Codegen allocates it to
      the extent its tiles reach (`_ref_extents`). Example: the TP8
      `recv` buffer, f32 `[2, 8, 1, 5120]` read in 16-row tiles, is
      allocated `[2, 8, 16, 5120]` = 5 MiB but was counted as 2.5 MiB.
    * Fix: `TileOverhang` / `StreamModel.overhang_bytes(tuned)`. It records
      per tensor the `(block id, scale, offset, noncanonical)` of each
      tile that indexes each dim, rebuilds `_ref_extents` at a config's
      block sizes, and adds the excess (× the launcher's copies) to the
      bytes the rings are budgeted against.
    * TP8 L=8 (`/tmp/mk/depths/tp_budget.py`): overhang 2.50 MiB. The
      rings drop from ~48.2 to 45.7 MiB, so the estimated allocation is
      ~62.1 MiB, leaving room for the 1.68 MiB of register spill slots
      that only the 3 MiB reserve covers.
  * **GEMV probes: VMEM-fed M=16 GEMV has a fixed cost per dot**
    (`/tmp/mk/gemv/gemv.log`; W already in VMEM, 1000 sweeps; one
    `jnp.dot` per `fori_loop` iteration, accumulating into a VMEM ref):

    | W tile (tk, tn) | Bytes per dot | Rate | Time per dot |
    |---|---|---|---|
    | (128, 2176), gate/up today | 557 KB | 2.55 TB/s | 0.218 µs |
    | (256, 1280), GDN qkv today | 655 KB | 2.82 TB/s | 0.232 µs |
    | (128, 5120), out/down | 1.31 MB | 4.20 TB/s | 0.312 µs |
    | (5120, 256), LM head | 2.62 MB | 4.88 TB/s | 0.537 µs |

    * Fit: about 0.15 µs (~280 cycles) fixed per dot, plus ~8 TB/s
      streaming.
    * So the gate/up and qkv tiles chosen today are *compute-bound*:
      slower than HBM's 3.35 TB/s, even with the weight already in VMEM.
      This explains why the streaming roots' tails do not hide.
    * In the generated loop (`step_L64.py`, `_fori_body_3/4`), each
      iteration does: wait for the slot; load the f32 accumulator
      `(16, tn)` from scratch; one or two dots; store it back; issue
      refills. It is a plain `jax.lax.fori_loop`, so iterations cannot
      overlap across the loop-carried scratch dependency.
    * Probe 2 queued (`gemv_probe2.py`, `/tmp/mk/gemv/gemv2.log`): the
      accumulator through scratch vs as a `fori_loop` carry, unroll 2/4,
      two split accumulators, and the gate/up pair, at tk 128/256/512.
      It picks the generic codegen change. Candidates:
      * unrolling ring-consumer loops;
      * carrying the accumulator in values, not scratch;
      * a `_stream_min_tile` floor of about 1 MB per dot in the
        block-size seeding.
  * **Model zoo pass done** (zoo agent, `/tmp/mk/zoo2/REPORT.md`). All
    four kernels compile, match the reference, and beat XLA: Gemma-3
    stack and full step, MLA stack, GPT-J parallel-residual TP8 shard.
    Results are in the zoo table above.
    * Generic fixes from the pass:
      * Pallas lowering for `gelu(approximate="tanh")`.
      * Row-split reshapes: a view between rank-1 and a static 2–64-row
        shape becomes a stack of row slices. Mosaic rejects
        rank-1 → `[r, 1, w]` and merges chained reshapes, but never
        merges a stack.
    * New blocker: F24 (ragged tiled minor dims in rings), MLA's only
      shortfall.
    * Usability notes for model authors:
      * folded layer loops cannot hold host scalar assignments, so write
        per-layer indices inline;
      * int loop-bound parameters need `hl.constexpr`;
      * embedding row gather needs a multiple-of-16 row count.
    * Merge into main after the F22b suites.
* 2026-10-03, F25 (ring-consumer loop shape):
  * **GEMV probe 2** (`/tmp/mk/gemv/gemv2.log`), VMEM-fed:
    * Carrying the accumulator in values instead of scratch makes no
      difference (0.241 vs 0.244 µs/dot).
    * Mosaic (jax 0.10) rejects `fori_loop(unroll=k)` unless k is 1 or
      the full trip count.
    * Two dots per iteration help: split accumulators gain +36%; the
      gate/up pair is 0.158 µs/dot vs 0.244.
    * So the fixed cost (~0.2 µs) is per loop *step*, not per dot.
  * **Probe 3** (`/tmp/mk/gemv/gemv3.log`): manual body replication,
    VMEM-fed, W[5120, 2176] tk=128:

    | Iterations per step | Rate |
    |---|---|
    | 1 | 2.28 TB/s |
    | 2 | 3.12 TB/s |
    | 4 | 3.75 TB/s |
    | full Python unroll | 4.71 TB/s |

    * qkv tile (W[5120, 1280] tk=256): 2.20 → 2.95 → 3.47 → 3.99.
    * Summing two dots before one accumulator update is no better than
      two updates.
  * The probe's ring-fed variant (one ring, W in HBM) is DMA-bound:
    1.2 TB/s at depth 3, ~2.5 at depth 8, 2.8 at depth 12. One ring's
    in-flight DMAs cannot fill HBM; the real kernel overlaps many rings
    (DMA-only step: 3.25 TB/s).
  * **Implementation (generic):**
    * `_codegen_fori_loop` turns a ring-consumer loop body into a step
      function of the loop index, called `U` times per `fori_loop`
      step, plus a tail for `trips % U`.
    * A loop of one step is unrolled in Python, so its ring slots are
      static.
    * `U` = `pallas_stream_unroll` (autotuned over None/1/2/4/8). The
      default is the smallest power of two ≤ 8 whose iterations stream
      4 MiB (`megakernel.stream_unroll`).
    * Interpret outputs are bit-identical to the rolled loop.
    * Test: `test_stream_unroll`.
* 2026-10-03, F25 on TPU (step L=8, /tmp/mk/f25/sweep_L8.log): unroll is a regression.

  | U | Body (µs) |
  |---|---|
  | 1 | 360.6 |
  | 2 | 393.2 |
  | 4 | 392.8 |
  | 8 | 396.9 |
  | default | 393.8 |

  * Top-1 OK everywhere.
  * The probe's win does not transfer to ring-fed loops. Each real step waits on a
    DMA semaphore and issues refills under `pl.when` (scf.if), which split the
    unrolled body into basic blocks.
  * The regression is roughly independent of U, so the loop structure itself
    (e.g. losing Mosaic's handling of a simple rolled loop) is the suspect.
  * Timelines U=1 vs U=4 at L=8: /tmp/mk/f25/tl_L8.log.
* 2026-10-03, F17b merged into main (3-way against /tmp/mk/snap-f17b-base).
  * 13 conflicts:
    * matmul: `rhs_transposed` = M7's folded-`.T` OR the lane-dense RHS load.
    * megakernel: M8a `IndexRead` scalars plus F17b `perm`; `index_reads`
      are reordered with the subscripts; F22b overhang kept.
  * Pre-merge backup: /tmp/mk/snap-f25.
* 2026-10-03, TP8 debug (v80 L=8, exchange):
  * Per-rank embedding row, final residual and final x all match the psum
    oracle (max err 0, 0.5 at |res| 24, 0.08 at |x| 4.26).
  * So the layers and both all-reduces per layer are correct. The zero output
    (idx 0, val 0.0 on every rank) comes from the LM head top-1, the tops
    all-gather, or the final reduction.
  * Next debug run dumps the local best and `tops` before and after the
    gather (/tmp/mk/tp8/v80_L8_dbg2.log).

* 2026-10-03, F25 timelines (step L=8, U=1 vs U=4, /tmp/mk/f25/tl_L8.log). Per-root
  sums over the call, with the ring step (bytes per iteration, trips, ring depth)
  from `/tmp/mk/f25b/sites.py`:

  | Root | Step | U=1 (µs) | U=4 (µs) | Δ |
  |---|---|---|---|---|
  | L258 gate/up (2× (128,2176), 40 trips, D=18 shared) | 1.06 MiB | 109.78 | 99.29 | −10.5 |
  | L246 GDN/attn out ((128,5120), 6 trips, D=6) | 1.25 MiB | 10.88 | 7.38 | −3.5 |
  | L193 GDN qkv ((256,1280), 20 trips, D=9) | 0.62 MiB | 37.54 | 35.58 | −2.0 |
  | L199 ((512,768), 10 trips) | 0.75 MiB | 14.43 | 15.40 | +1.0 |
  | L267 w_down ((128,5120), 17 trips, D=6) | 1.25 MiB | 48.02 | 54.21 | +6.2 |
  | L277 LM head ((5120,256), 125 trips, D=5) | 2.50 MiB | 98.67 | 138.63 | +40.0 |

  * The unroll does what the probe predicted for the compute-bound GEMV
    loops (gate/up −10%). The loss is in loops that were already at the DMA
    rate: the LM head runs at 3.32 TB/s rolled and 2.36 unrolled.
  * The two (128,5120) loops differ only in trips: 6 trips (1 rolled step of
    4 plus a 2-step tail) gains, and 17 trips (4 rolled steps) loses. So the
    loss grows with how many unrolled bodies run back to back on a shallow
    ring.
  * Working hypothesis: inside one unrolled body, the scheduler no longer
    issues each `pl.when` refill before the next step's wait and compute.
    The refills drift toward the end of the body, so a D=5 ring has as few
    as D−U tiles in flight.
  * Probe `/tmp/mk/f25b/ring_probe.py` tests this on one ring loop
    (LM-head body with argmax, w_down, gate/up) with three refill
    placements: `when` (today), `clamp` (unconditional, clamped
    index, drain waits) and `early` (refill the previous slot before the
    wait).
* 2026-10-03, TP8 dbg2 (/tmp/mk/tp8/v80_L8_dbg2.log): each rank's local best is
  right, and after the all-gather `tops` holds every rank's (val, idx) with the
  reference winner (rank 3, 4.4166, idx 108774). So the LM head and the gather
  are correct; the final reduce root is wrong. That root runs at block size 1:
  it loads `tops[:, 0, pl.ds(0, 1)]` from an (8,8,128) f32 VMEM scratch,
  reduces over dim 0 and stores (1,) outputs, and returns 0 / 0.0 on TPU.
  Mosaic repro: `/tmp/mk/f25b/top1_repro.py`.
* 2026-10-03, F24 merged into main (3-way against /tmp/mk/snap-f24-base, 4 conflicts
  in megakernel.py resolved against F17b: physical `shape` in place of
  `fake.shape`; `ragged_cols` is now "tile over the whole physical minor dim";
  `search_only` takes the union of `whole_block_sizes` over the lane-dense on
  and off models). Interpret suites all pass (megakernel 85). Snapshots:
  /tmp/mk/snap-pre-f24, /tmp/mk/snap-f24-merged.
* 2026-10-03, F25 ring probe (/tmp/mk/f25b/run1.log): **refill placement is not
  the cause.** All three placements give the same rates at each U:
  * LM head with argmax (D=5): 2.26 TB/s at U=1, 1.83 at U=2 or 4.
  * w_down (D=6): flat at ~0.41.
  * gate/up (D=9): 0.38 at U=1, 0.40–0.41 at U=2 or 4.

  So the LM-head regression is Mosaic's schedule of a replicated
  dot + argmax body. It is not ring depth starving the DMAs. A 2.5 MiB step
  already amortizes the ~0.2 µs step cost. The default target drops from
  4 MiB to 2 MiB, so steps ≥ 2 MiB stay rolled (LM head U=1); gate/up gets
  U=2, qkv/z/attn-q/kv get U=4, and GDN out / w_down get U=2. Validation:
  pod tree v82, /tmp/mk/m5/v82.log.
* 2026-10-03, TP8 top-1 follow-up: the Mosaic final-reduce construct is correct
  in `pallas_call` (all top1_repro variants). The jax_fn entry mapping is
  correct too: results = output-only slots [2, 33, 34, 35, 36], so
  results[3]/[4] are top_idx/top_val. Two hypotheses remain:
  * the launcher's `pl.kernel` staging copy-out of a (1,) VMEM buffer
    (`/tmp/mk/tp8b/stage_repro.py`);
  * the in-kernel compute. dbg3 also writes the final best/argmin into
    `dbg[0:2, 0]`.
* 2026-10-03, v82 on TPU (main = F25 at 2 MiB + F17b + F24, /tmp/mk/m5/v82.log), all
  numerics OK:

  | Run | Before | v82 |
  |---|---|---|
  | step L=8 body | 360.6 (v81 U=1), 358.1 (F17b) | **348.7 µs**, 3.14 TB/s |
  | step L=64 body | 2127 | **2044.6 µs**, 3.16 TB/s |
  | hybrid L=8 | | 245.8 µs, 30.7 µs/layer |
  | hybrid L=64 | | **1900.0 µs**, 29.7 µs/layer, 3.23 TB/s |
  | F25 A/B, step L=8 | U=1: 357.7 | default: 348.9 (−8.8 µs) |
* 2026-10-03, TP8 top-1 root cause (/tmp/mk/tp8b/dbg3.log): the kernel's final
  reduce is right on every rank (dbg: 4.4166 @ 108774). The value is lost in
  the copy-out. A (1,) VMEM buffer DMA'd to a (1,) HBM output in `pl.kernel`
  reads back 0 even for constants, while (128,) and (1, 1) buffers work
  (stage_repro.py). The single-core step's (1,) staged outputs do work, so the
  failure depends on context. stage_repro2.py isolates store vs DMA, the
  length threshold, and the jit output layout.
* 2026-10-03, TP8 hazard pinned down (stage_repro3/4, /tmp/mk/tp8b/stage3b.log,
  stage4.log). The trigger is **any one-element VMEM buffer** that a vector store
  writes and a DMA then copies out. It fails for (1,) and (1, 1), in i32, f32 and
  bf16, reading back 0 or stale data. This corrects the entry above: (1, 1) also
  fails, and it only looked fine when the stale value happened to match. Padding
  the minor dim fixes it in every dtype: (8,), (16,), (128,), (1, 8) and (1, 128)
  are correct with only element 0 stored. DMA-in followed by a load of a
  one-element buffer is correct, so only outputs need the fix. Mosaic has no fence
  primitive. Fix: pad one-element VMEM outputs to a lane row via `_ds_pad_dims`
  (backend), and slice them back in the launcher.
* 2026-10-03, M7b done (agent report /tmp/mk/m7b/REPORT.md); merged into main
  3-way against /tmp/mk/snap-m7b-base with no conflicts.
  * F23: held ring copies start right behind the root's last remote start.
  * Bubble-aware holds: earliest-freed prefix, global `_EXCHANGE_HOLD_BYTES` =
    12 MiB. A copy is held only if its slot frees before the previous exchange.
  * Exposed exchange (µs/layer): hybrid L8 8.31 → 6.68, hybrid L16 8.15 → 6.89.
    TP MLP stays at 1.48 (the rule holds nothing there).
  * Hybrid L16 extrapolates to ~2.51 ms for 64 layers (layer stack only), vs the
    2.45 gate.
  * Hold sweep: best at 12–16 MiB. At 20 MiB, holds queue behind the remote copies
    and stretch the exchange.
  * Per-shape rings are full when the exchange starts. The reference keeps
    ~31 MB in flight in one global ring through the all-reduce (its noex
    ablation saves only ~0.6 µs/layer). **F18 is the next lever**; the
    exchange latency itself is next after that (XOR pairing, per-peer sems,
    rank skew).
* 2026-10-03, one-element output fix verified.
  * Interpret suites on main with the fix: megakernel 90 (+1:
    `test_one_element_outputs_padded`), pallas 245, load_store 118,
    block_sizes 13, bound_kernel 11, loop_dependencies 5, views 12,
    memory_access 4, remote_copy 1. All pass.
  * TPU, TP8 step L=8 (pod tree `/root/helion-v83`, `/tmp/mk/tp8b/v83_8.log`):
    top-1 108774 on every rank, matching the oracle.

    | Variant | Device µs | Body µs |
    |---|---|---|
    | exchange | 424.6 | 416.3 (52.0 µs/layer, 2.63 TB/s) |
    | noex | 361.2 | 352.9 (44.1 µs/layer, 3.10 TB/s) |

    The exchange costs ~7.9 µs/layer, including the top-1 all-gather. At L=64
    that extrapolates to ~2.65 ms (2.148 single-core + 64 × 7.9 µs), against
    the 2.45 gate. To reach the reference's 2.33, exposed exchange must drop to
    ~2.8 µs/layer. The exchange is now the whole TP gap: single-core we are
    already 0.14 ms ahead of the reference's `noexchange`.
  * Mosaic facts found while scoping F18 (jax 0.10):
    * `AsyncCopyDescriptor.start(priority=...)` is lowered to
      `tpu.enqueue_dma(priority=...)`. If the DMA engine has a second queue,
      remote exchange copies could overtake queued ring refills, which is the
      FIFO effect behind "holds queue behind remote copies". Probe:
      `/tmp/mk/tp8/prio_probe.py`.
    * `start(add=True)`, an accumulating DMA (a hardware reduce for
      reduce-scatter), exists in the API, but its lowering raises
      NotImplementedError.
    * `tpu.reinterpret_cast` (a raw typed view of VMEM with an explicit tiled
      layout) exists in the dialect, but no public Pallas transform reaches
      it. The reference gets to it by monkeypatching `_reshape_memref`.
      Options for F18: (a) a uniform minor dim across rings (one `(rows, W)`
      arena, slots as row ranges, needs a ragged last N tile for 2176 = 17 ×
      128); (b) Helion's own raw-view transform (fragile: jax internals).
  * TPU, TP8 step L=64 (v83, `/tmp/mk/tp8b/v83_64.log`): top-1 216429 on
    every rank, matching the oracle.

    | Variant | Device µs | Body µs |
    |---|---|---|
    | exchange | 2732.5 | 2616.0 (40.9 µs/layer, 2.47 TB/s) |
    | noex | 2226.8 | 2110.3 (33.0 µs/layer, 3.06 TB/s) |

    * Exchange: (2616.0 − 2110.3) / 64 = 7.9 µs/layer, the same as at L=8.
    * Device time is ~117 µs above the body at L=64, against 8 µs at L=8 and
      21 µs single-core. The TP harness does not donate the per-layer states
      (rec_state, KV caches, ~52 MB/rank), so XLA copies them before the
      kernel aliases them. That is a harness artifact. The comparable number is
      the body plus ~20 µs: **~2.64 ms with exchange vs the 2.45 gate.**
  * Correction to the harness note above. Unfiltered op list for TP8 L=64
    (pod `/root/tp8/tp_step_L64_.log`): the ~117 µs is
    * 2 × 38.5 µs relayouts of GDN `b`/`a` (`bf16[1,48,5120,6]`);
    * 15.9 + 15.2 µs of KV-cache copies and 6.3 µs of rec_state copy.

    The cause is the harness, not the kernel:
    * Per-rank operands are sliced `t[0]` out of `[W, ...]` stacks. The
      default TPU layout of the 4-D stack (`{2,1,3,0}`) is not the 3-D layout
      F17b's lane-dense swap expects, so XLA relayouts every call.
    * The jax_fn entrypoint returns only the output-only results, so in-place
      states are never donated, and XLA copies each aliased state every call.

    Harness fix (`run_tp_step.py`; the old version is kept as
    `run_tp_step_v83.py`):
    * fold the rank dim into dim 0, so shards have single-device shapes and
      layouts;
    * call `_pallas_jax_call(return_all_outputs=True)`, donate the states and
      thread them call to call like a decode loop.

    Generic follow-up: the jax_fn entrypoint should return in-place outputs,
    so a JAX caller can donate states. Today it silently drops the updates.
  * TPU, v83 single-core revalidation (`/tmp/mk/tp8b/v83_runs.log`), no
    regression from the pad fix. All top-1 and numerics PASS.

    | Run | Device µs | Body µs |
    |---|---|---|
    | step L=8 | 363.0 | 348.8 |
    | step L=64 | 2064.9 | 2044.5 (3.16 TB/s) |
    | hybrid L=8 | | 246.1 (30.8 µs/layer) |
    | hybrid L=64 | | 1899.8 (29.7 µs/layer) |

    Single-core L=64 is 2.065 ms vs the reference's 2.29 ms noexchange.
  * TPU probes of the exchange and DMA queue (`/tmp/mk/prio/probes2.log`,
    probes `/tmp/mk/tp8/{prio,xchg}_probe.py`):
    * DMA priority. One 4 KiB local copy queued behind 24 × 1 MiB local
      copies costs 16.26 µs/rep at priority 0 and 11.79 µs/rep at priority 1
      (spin alone 9.65, copies alone 8.56). A priority-1 local copy overtakes
      the queue. Raising the big copies to priority 1 does not let a
      priority-0 copy through.
    * Remote DMA rejects non-zero priority (`tpu.enqueue_dma`: "non-zero
      priority is not supported for remote DMA"). Priority cannot speed up
      the exchange.
    * Intrinsic 8-way all-reduce latency inside one kernel, with no other
      work:

      | Payload | One-shot | One-shot, per-peer sems | XOR (3 rounds) |
      |---|---|---|---|
      | 20 KiB | 4.15 µs | 4.16 µs | 6.99 µs |
      | 160 KiB | 15.02 µs | | 12.56 µs |

      One-shot is right for the decode payload (20 KiB). XOR wins only for
      large payloads.
    * Remote copies do not queue behind local DMA. A one-shot exchange with
      12 MiB of local copies issued just before it costs 4.80 µs/rep, against
      4.82 for the local copies alone. That contradicts the M7 "FIFO" claim.
    * Conclusion. The 7.9 µs/layer of exposed exchange is not queueing and
      not the algorithm. It is that the ring DMA idles during the exchange.
      Per-shape rings are full when the exchange starts, so there is little
      queued work to overlap with. The lever is F18: one program-order arena
      across all weight shapes, so ~12 MB of the next sites' tiles are in
      flight during every exchange.
  * F18 direction: raw views. One VMEM arena of native `(16, 128)` bf16
    tiles. A tile slot is a range of native tiles viewed as any `(bk, bn)`
    with `bn % 128 == 0` via `tpu.reinterpret_cast` + `erase_memref_layout`
    (helper `/tmp/mk/f18/rawview.py`: a `ReshapeTransform` subclass
    dispatched through a wrapped `lowering._reshape_memref`). Tiles keep
    their tuned shapes, slots are sized by the largest tile, and fill is
    ≥ 85% for every Qwen tile at 640 KB slots. Rejected:
    * one max-box slot with sub-windows (poor fill across 1280/2176/5120);
    * a uniform minor dim across rings (forces 128-wide rows or a ragged
      2176 tile).

    Probes:
    * `/tmp/mk/f18/raw_probe.py`: correctness of mixed-shape tiles at
      dynamic, unaligned arena offsets, and GEMV TB/s of raw vs plain rings.
    * `/tmp/mk/f18/arena_probe.py` (8 devices): a synthetic 80 MB/rank layer
      (`(256,1280)`, `(128,2176)`, `(128,2560)` tiles, two one-shot
      exchanges), with one 34-slot arena vs three per-shape rings of the same
      VMEM (22 MB), exch vs noex. It measures how much exchange the arena
      hides.
  * TPU, TP8 with the harness fix (`/tmp/mk/tp8c/fair.log`). Top-1 OK at
    L=8 and L=64. The kernel body is unchanged: L=64 2617.0 µs with exchange,
    2112.6 noex. The fix did NOT remove the per-call copies: device total
    2764.0 / 2259.4, and the op list still has the `b`/`a` relayouts
    (`bf16[48,5120,6]{1,2,0}` → `{2,1,0}`, 2 × 38.5 µs) and the state copies.
    So folding the rank dim and donating are not enough. Inside
    `shard_map`, the lane-dense swap (F17b) and the in-place aliasing do not
    take effect the way they do single-core, where device minus body is
    20 µs. Open (harness/entrypoint, not kernel): find where those copies
    come from. Until then, the TP number to compare is the body plus ~20 µs
    (L=64: ~2.64 ms with exchange).
  * TPU, F18 probes (`/tmp/mk/f18/raw1.log`, `arena1.log`, `arena2.log`):
    * Raw views work on jax 0.10.0. Mixed-shape tiles at dynamic, unaligned
      arena offsets give exact GEMV results. Raw-view rings stream exactly
      as fast as plain rings for `(128,2176)`, `(256,1280)` and `(128,5120)`.
    * Synthetic 8-device layer (80 MB/rank, 134 tiles, two one-shot
      exchanges), µs/layer, at the same 22 MB of VMEM:

      | Variant | rings noex | rings exch | arena noex | arena exch |
      |---|---|---|---|---|
      | DMA only (no dot) | 28.67 | 35.60 | 28.52 | **29.42** |
      | dot only (no DMA) | 26.15 | 27.17 | 25.97 | 27.99 |
      | both | 31.59 | 38.82 | 29.35 | 37.14 |

    * The arena does what F18 promised for the DMA. With DMA alone it hides
      both exchanges (+0.9 µs/layer against +6.9 for per-shape rings), and it
      is 7% faster noex with compute (29.35 vs 31.59).
    * But with compute, the exchange is exposed again (+7.8 µs/layer). The
      GEMV consumer is nearly as slow as the DMA: dot only takes 26.0 µs vs
      28.5 for DMA only, so its headroom is ~9%. During an exchange the
      DMA keeps filling the arena. Afterwards the consumer drains the
      backlog only 9% faster than new tiles land, so it cannot catch up
      before the next exchange. The arena fills and the DMA stalls.
    * Hiding E µs of exchange per T µs layer needs a consumer about E/T
      faster than the DMA: 8/30 ≈ 27% here. **The lever is now GEMV compute
      rate (consumer headroom), with F18 as the enabler.** The reference
      streams more slowly (2.82 TB/s noex vs our 3.06 at TP8 L=64), so its
      consumer has headroom, and it loses only ~0.6 µs/layer to exchanges.
    * Next probe: GEMV compute rate with VMEM-resident tiles, by formulation
      (`x @ w` with a VMEM acc, a register-accumulated K loop, `wᵀx`), tile
      shape and rows (`/tmp/mk/gemv/gemv_probe.py`).
  * **GEMV compute ceiling, measured** (single core, weights resident in
    VMEM, B=16):
    * `fori_loop` over tiles (`/tmp/mk/gemv/g1.log`): ~0.15 µs fixed per
      step. 557 KB tiles reach 2.4 TB/s, 1 MiB tiles 3.6, 1.3 MB tiles 4.0.
      `wᵀx` (`dot_general` contracting dim 0 of w) is 40% slower in every
      case. Summing K dots in a value before one acc update is no faster.
    * Python-unrolled tiles, 16 MiB per `fori_loop` step
      (`/tmp/mk/gemv/g4.log`): **6.0–6.5 TB/s for every tile shape** from
      (128,2176) to (1024,1280). The order (n-major, k-major, value
      accumulation, one dot over all of K) makes no difference.
    * The MXU absorbs a skinny GEMV at ~2.1× the DMA rate (~2.9–3.1 TB/s).
      What costs is *dependency points*. Probe 3's "full unroll" zeroed the
      acc and flushed it into the output every 40 dots, and got only
      4.7 TB/s: ~1 µs per flush.
    * So consumer headroom is available in principle. The arena probe's
      dot-only variant (3.08 TB/s) loses it somewhere in the per-tile
      code. Probe 5 (`gemv_probe5.py`) adds the megakernel features one at
      a time: dynamic slots, exchange data dependencies, DMA wait/start
      per tile, `pl.when` refills.
  * **Probe 5/6: the DMA wait is the consumer's barrier**
    (`/tmp/mk/gemv/g5.log`, `g6.log`). Single core, the arena probe's 134
    tiles/layer from VMEM-resident slots. Per-tile DMA work uses 4 KiB
    stub copies so HBM never limits:

    | Per-tile code | µs/layer | TB/s |
    |---|---|---|
    | dot only, static slots | 13.11 | 6.10 |
    | + dynamic slots | 13.08 | 6.11 |
    | + exchange data dependency after R0/R2 | 13.17 | 6.07 |
    | + wait, start per tile | 29.6 | 2.70 |
    | waits grouped by 2 / 4 / 8, starts per tile | 20.9 / 17.1 / 16.2 | 3.83 / 4.69 / 4.95 |
    | starts grouped by 4 / 8, waits per tile | 29.5 | 2.71 |
    | waits + starts grouped by 4, each start under `pl.when` | 21.4 | 3.73 |

    * A DMA wait orders every later vector op after it. So a wait per tile
      exposes the MXU fill/drain (~120 ns) on every tile, and the
      consumer falls to the DMA rate.
    * Grouping K waits before K dots pays that cost once per group.
    * Each `pl.when` around a refill costs another ~30 ns per tile.
    * **F26 (new): grouped ring waits.** In a stream-unrolled step, issue
      the waits of all U iterations before their bodies. Refills stay
      after each body. Then cut the refill branches (F26b: one branch per
      group, or clamped branch-free refills drained at exit).
    * Cost: during the group's waits, D−U slots are in flight instead of
      D−1, so it needs depth. F18 provides it.
  * **The 8-device "slow consumer" was the dispatch floor**
    (`/tmp/mk/gemv/g7.log`, `/tmp/mk/f18/arena3.log`, `arena4.log`).
    * Probe 7 runs probe 5 under `shard_map` on 1 / 2 (same chip) / 2
      (different chips) / 4 / 8 devices. The dot-only consumer runs at
      6.15 / 5.76 / 5.74 / 5.50 / 4.96 TB/s (L=16): more devices cost a
      little, and a chip's second core costs the same as a second chip.
    * The arena probe's dot-only variant across L:

      | L | µs/layer noex | total µs | µs/layer exch |
      |---|---|---|---|
      | 8 | 26.31 | 210 | 28.02 |
      | 16 | 14.50 | 232 | 22.54 |
      | 32 | 12.82 | 410 | 21.09 |

      An 8-device `shard_map` call takes ~200 µs to dispatch from the host.
      So the L=8 probe numbers (arena 1–3) measured dispatch, not the
      kernel. **Probe at L ≥ 32.** The real TP8 L=64 runs (≥ 2 ms) are
      unaffected.
    * At L=32 the dot-only consumer runs at 6.24 TB/s, and the exchange
      adds 8.3 µs/layer. With no DMA to overlap, that is the latency of the
      two exchanges per layer (~4 µs each), and it matches the 7.9
      µs/layer exposed in the real TP8 L=64 kernel.
    * Both variant, rings, L=32: 26.31 µs/layer noex (3.04 TB/s), 34.33
      exch. The whole exchange latency is exposed: the rings hold ~5 MB
      (~1.6 µs of DMA), so they fill early in each exchange and the DMA
      stalls.
    * The arena3 conclusions (grouping doesn't help, dot-only at 3.0
      TB/s) are void. Rerun at L=32 (`/tmp/mk/f18/arena5.log`), µs/layer:

      | Variant | rings noex | rings exch | arena noex | arena exch |
      |---|---|---|---|---|
      | DMA only (no dots) | 27.78 | 33.50 | 24.58 | **25.24** |
      | dots only (resident) | 12.82 | 21.09 | 12.79 | 21.07 |
      | both, wait per tile | 26.32 | 34.18 | 24.75 | 32.75 |
      | both, waits grouped by 4 | 26.98 | 34.94 | 24.72 | 32.81 |
      | both, waits grouped by 8 | 29.29 | 37.35 | 24.78 | 32.81 |

    * The arena streams at 3.23 TB/s (7% faster than rings) and hides
      all the compute when there is no exchange. Without dots it also
      hides the exchange (+0.66 µs/layer).
    * **With dots the whole exchange is exposed (+8.0), even though the
      consumer is 2× faster than the DMA** (12.8 vs 24.6 µs/layer).
      Grouped waits change nothing. So the consumer-headroom model
      (backlog drains at 1 − DMA/consumer rate) is wrong, or something
      else stalls during the exchange.
    * Grouping by 8 makes rings *slower* (29.3 vs 26.3 noex). Waiting for
      8 tiles at once leaves D−8 slots in flight in 8/16/12-slot rings.
    * Exchange variants, arena, waits grouped by 8, L=32
      (`/tmp/mk/f18/arena6.log`), exch µs/layer (noex 24.7):

      | Payload | Result feeds next root's x | exch | exposed |
      |---|---|---|---|
      | acc (after R0/R2 dots) | yes | 32.81 | 8.0 |
      | acc | no | 32.79 | 8.0 |
      | zeros (no dot dependency) | no | 29.70 | 5.0 |
      | zeros | yes | 29.72 | 5.0 |

      - The next root's dependency on the result costs nothing.
      - The payload's dependency on the producing dots costs 3 µs/layer,
        ~1.5 per exchange: the MXU drains before the sends start.
      - A blocking exchange with dots still exposes 5 µs, vs 0.66 without
        dots.
    * Model: after a blocking exchange of E µs, the consumer has an E-µs
      DMA backlog, which it drains at rate C − D.
      - With grouped waits, C ≈ 4.5–5 TB/s (multi-device), so a 4 µs
        backlog takes ~8 µs to drain and needs ~37 MB of tiles. But R0
        between the exchanges is only 13 MB.
      - Per layer the consumer has 24.6 − 80 MB/C ≈ 7–8 µs of slack, and
        two exchanges need 8 µs. So the consumer barely catches up, the
        arena (21.8 MB) fills, and the DMA stalls.
    * The reference (`/tmp/mk/ref/exchange_design.md`) has no exchange
      trick: a blocking one-shot all-to-all and one wait per tile.
      - But its tiles are ~2.6 MB (12 banks ≈ 31 MB), so the per-wait
        barrier is paid ~4× less often per byte and the consumer runs near
        the dot-only rate.
      - **Hypothesis: large tiles plus a deep global ring are what hide the
        exchange.** Probes: deeper arena with 640 KB tiles (`arena7.sh`,
        depth 50/70), and ~2.5 MB tiles in a 12–16-slot arena
        (`arena8.sh`).
  * **F26 TPU A/B (whole unroll grouped)**, pre-F26 v84a vs v84. Single
    core: step L=8 348.7 → 361.7 µs, L=64 2044.5 → 2146.1, hybrid L=64
    1899.8 → 1932.5. TP8 L=64: exch 2617.3 → 2755.5, noex 2112.4 →
    2239.8 (exposed exchange unchanged, 505 → 516 µs).
    * A group's refills start only after its last tile lands, so G − 1
      fewer copies are in flight. The step is DMA-bound, so that costs
      more than the barriers save.
    * F26 is now knob `pallas_stream_wait_group`: default None (a wait per
      iteration), autotuner choices 2/4/8, capped by the unroll and by ring
      depth.
  * **Tile size, not ring depth, decides whether the exchange hides**
    (arena probe, L=32, µs/layer; `/tmp/mk/f18/arena7.log`, `arena8.log`):

    | Tiles (per layer) | Buffering | noex | exch | exposed |
    |---|---|---|---|---|
    | 640 KB (134) | arena 34 (22 MB) | 24.75 | 32.75 | 8.0 |
    | 640 KB | arena 50 (32 MB) | 24.79 | 32.76 | 8.0 |
    | 640 KB | arena 70 (45 MB) | 25.00 | 32.89 | 7.9 |
    | 640 KB | arena 70, waits by 4 | 25.19 | 32.79 | 7.6 |
    | ~2.5 MB (33) | rings 4/6/4 (35 MB) | 24.81 | 31.22 | 6.4 |
    | ~2.5 MB | **arena 12 (33 MB)** | 24.65 | **26.10** | **1.45** |
    | ~2.5 MB | arena 16 (44 MB) | 24.75 | 26.21 | 1.5 |
    | ~2.5 MB, DMA only | arena 16 | 24.69 | 24.97 | 0.3 |
    | ~2.5 MB, dots only | arena 12 | 12.67 | 20.91 | 8.2 |

    * With 640 KB tiles, a deeper arena never helps, even at twice the
      bytes in flight. With 2.5 MB tiles, a 12-slot arena hides all but
      1.45 µs/layer of the exchange (payload MXU drain included). That
      is the reference's shape (12 banks of ~2.6 MB).
    * Big tiles in per-weight rings help less (6.4 exposed). During R0
      only R0's ring refills: the next roots' rings are already full, so
      when the exchange blocks the consumer only R0's 4 slots are in
      flight. An arena keeps program order, so all 12 slots are always in
      flight toward the next tiles.
    * Hypothesis for why small tiles fail at any depth: a cap on
      outstanding DMAs (or a per-copy issue cost) makes bytes in flight
      scale with tile size, not slot count. Not yet tested directly.
    * Plan: (1) bigger default tiles (`seed_block_sizes` prefers the
      smallest tile ≥ `_STREAM_MIN_SLOT_BYTES` = 512 KB, on the theory
      that smaller slots keep more in flight; the probe contradicts it).
      Test the TP8 step with a 2.5 MB minimum first. (2) F18 arena, or a
      cheaper equivalent: deeper rings for the roots after an exchange.
  * **F18 implemented (2026-10-03), knob `pallas_stream_arena`.**
    * With the knob, every whole-VMEM-tile 2-D streamed tile of a dtype
      shares one ring (root-level loads get a second one). A slot is
      `(n, sublanes, 128)` native tiles, where `n` fits the largest tile;
      the buffer is `(depth * n, sublanes, 128)`. Each tile reads the
      slot through `ring_tile(arena, slot * n, (rows, cols))`, a
      `ReshapeTransform` subclass. Mosaic lowers it to
      `tpu.reinterpret_cast` + `erase_memref_layout` (patched
      `_reshape_memref`); interpret mode reshapes, with a patched
      `transform_swap_array` for writes.
    * Depth: the arena's cap is its whole stream, so the stall model
      grows it into all the VMEM left. It holds no copies at exchanges
      (`_hold_for_exchanges` skips it): program order already keeps the
      next roots' tiles in flight.
    * Seed: with `_ARENA_BY_DEFAULT`, `seed_block_sizes(arena=True)`
      picks the tiles closest to `_ARENA_SLOT_BYTES` (2.5 MiB) without
      going over, with no four-iterations limit. Qwen MLP: (512, 2176)
      gate/up and (2176, 512) down. TP8 step: tiles (1024, 1280),
      (512, 2176), (2176, 512), (256, 5120), (5120, 256); the arena is
      18 slots of 2.6 MB (47 MB).
    * The loop-tile heuristic used to rebuild the default config from its
      block sizes alone, which dropped other seeded knobs. It now keeps
      them.
    * **TP8 L=64 on TPU (helion-v85, kernel body µs, 2026-10-03):**

      | config | exch | noex | exposed |
      |---|---|---|---|
      | main (per-weight rings, default tiles) | 2617.3 | 2112.4 | 7.9 µs/layer |
      | rings + 2.5 MiB tiles (`_STREAM_MIN_SLOT_BYTES`) | 3267.8 | — | — |
      | arena + default tiles | 2705.4 | 2108.9 | 9.0 µs/layer |
      | **arena + 2.5 MiB tiles (`_ARENA_BY_DEFAULT`)** | **2209.8** | **1928.1** | **4.4 µs/layer** |

      Device time per token: **2356.9 µs** (main 2764.2), vs the reference's
      2330 µs, so M7's ≤ 2.45 ms gate passes; top-1 matches. Both halves
      are needed. Big tiles in per-weight rings leave each ring a couple of
      slots, and the arena with small tiles stays latency-bound around an
      exchange. With both, the noex body streams at 3.35 TB/s (HBM peak), and
      the exchange is 4.4 µs/layer exposed: about its latency floor
      (~4.3 µs, M7).

    * **Single core L=64 (helion-v85, `/tmp/mk/m5/f18_sc.log`):** step body
      2044.5 → 1928.2 µs (device 2063.2 → 1947.8), top-1 OK; hybrid body
      1899.8 → 1834.2 µs (28.7 µs/layer, 3.34 TB/s), numerics PASS. The
      arena wins everywhere measured, so **it is the default since
      2026-10-03** (`_ARENA_BY_DEFAULT = True`).
    * Tests: 33 structural checks assumed per-weight rings. Tests of what only
      per-weight rings do (slots shared by shape, per-ring depths and
      floors, copies held at an exchange) pin them with the
      `_per_weight_rings()` context manager / decorator. It also resets the
      module's bound kernels, because a bound kernel keeps the default config
      it was bound with. Ring-or-arena tile reads match `_ring_tile(name)`.
      New: `test_exchange_arena_holds_nothing`. Suites: megakernel +
      autotuner heuristics 186 passed / 56 skipped; test_pallas 245 passed;
      test_examples 80 passed after dropping a stale xfail
      (`test_gather_gemv_half` passes in interpret mode since the launcher's
      whole-block staging).
    * **What is left of the TP8 per-call time (device 2356.9 − body 2209.8 =
      147 µs):**
      - 2 × 38.5 µs: XLA relayouts the GDN a/b weights `bf16[48,5120,6]`.
        Their default layout is the dim order (2, 0, 1), and F17b serves
        only the last-two swap. Plan: F17c.
      - ~67 µs: the in-place states (recurrent state f32[48,6,128,128]
        16.5 + 6.5 µs, KV caches 15.7 + 15.3 + 6.7 + 5.8 µs) are copied
        once before the kernel and once after (`/tmp/mk/tp8/hlo_peek.log`).
        This happens even though the jit donates them and the custom call
        aliases each one (`output_to_operand_aliasing` {0}: state, {1},
        {2}: caches). JAX's lowering has no copies, so an XLA pass adds
        them. A standalone `new_ref` + donation toy under shard_map has
        none (`/tmp/mk/tp8/alias_toy.py`), so the cause is specific to the
        step. **Cause (found 2026-10-03): the kernel communicates.** XLA's
        copy insertion removes every copy, then its special-case step adds a
        copy of each in-place operand and of each kernel output at the root
        (`hlo_l8p` dumps 0057 → 0058). A toy reproduces it with a remote
        barrier and `collective_id` and not without them, for `pl.kernel` +
        `new_ref` and for `pallas_call` + `input_output_aliases` alike
        (`alias_toy4.py`, `alias_toy5.py`). Remote DMAs need symmetric
        buffers, so XLA keeps a communicating kernel's buffers out of
        runtime-allocated entry parameters and outputs. Our output structure
        doesn't matter: dropping idx/val, reordering outputs, not donating
        conv: all 4 variants keep the copies (`copy_bisect.py`). The
        reference kernel has the same structure (a communicating
        `pallas_call` aliasing its states), so it pays these copies too, per
        call. Its bench runs 16 steps in one jitted `fori_loop`, where the
        states are loop carries in compiler-allocated buffers and the entry
        copies are paid once per loop. `run_tp_step.py` now has the same
        mode (`STEPS=16`), for an apples-to-apples number.
      - conv state: 2.3 µs (lane-dense in, fresh output buffer, copied
        back).
* 2026-10-03 — **TP8: apples-to-apples loop number; where the last ~230 µs go.**
  * **Loop methodology (`STEPS=16`, one jitted `fori_loop`, like the
    reference's bench): 2250.5 µs/token with the exchange** (device per call
    2362.1, body 2215.6; `/tmp/mk/tp8/loop_vs_ref.log`). Reference: 2330
    µs/token. Same session (`/tmp/mk/tp8/waitgroup_looprest.out`):

    | µs/token, 16-step loop | ours (v86) | reference |
    |---|---|---|
    | with exchange | **2250.5** | 2329 |
    | no exchange | **2048.4** | 2293 |

    Ours is 3.4% faster with the exchange and 10.7% without. The reference
    exposes only 36 µs of exchange; we expose ~200, so the exchange is now
    the whole TP8 lever.
  * **Consumer critical path** (`stub_ring.py`: every ring copy and wait
    replaced by one 4 KiB tile, so only compute and exchanges remain; L=64
    body, `/tmp/mk/tp8/stub_ab.log`):

    | | consumer only | real (with DMA) |
    |---|---|---|
    | noex | 1412.5 µs (22.07 µs/layer) | 1928 (30.1/layer, DMA-bound) |
    | exch | 1977.8 µs (30.90 µs/layer) | 2210–2216 |

    * With both exchanges the consumer alone (30.9 µs/layer) is longer than
      the weight DMA (30.1). Perfect overlap would give ~1978 µs; we lose
      ~230 µs to imperfect overlap on top of that.
    * The exchanges cost 8.8 µs/layer of consumer time (4.4 each), all of it
      latency.
  * **Exchange latency is intrinsic, not layout** (`xchg_layout_probe.py`,
    8-way one-shot of one f32 row, µs/exchange incl. ~1.0 µs of reduction):
    a row of a padded (16, 5120) buffer 4.32, of an (8, 5120) buffer 4.32, a
    lane-dense (1, 5120) buffer 4.28, the dense (8, 640) block 4.13; no
    exchange 1.0–1.1. Unpadded receive buffers are not worth a compiler
    change.
  * Hypotheses for the overlap loss, being tested on helion-v86:
    1. **Writeback waits drain the ring.** The state writebacks (rec_state,
       k/v cache rows) are waited right before each exchange, and they queue
       behind ~44 MB of ring refills, so the exchange starts with the ring
       nearly empty. The reference waits for the previous layer's
       writebacks at the next layer. Probe: `sink_wb.py` (M3_PATCH) defers
       each wait to the staging buffer's next write (`pl.when(p > 0)`) plus
       exit waits after the period loop.
    2. **Per-tile wait barriers** lengthen the consumer chain: retest
       `pallas_stream_wait_group` 2/4 with the 18-slot arena (pre-arena it
       lost because fewer copies were in flight).
    3. **Chunked exchange:** send each GEMV output chunk as soon as it is
       produced, so the exchange overlaps the GEMV's tail
       (`xchg_chunk_probe.py`).
  * Smaller items seen in the generated code: the peer index is computed
    with a vector `roll` + extract per send (7 per exchange) and `rank` is
    read from VMEM; the post-exchange reduction is a 40-step loop over
    128-column blocks of 16 padded rows.
* 2026-10-03 — **F17c merged (general-permutation lane-dense inputs;
  /tmp/mk/f17c/REPORT.md).**
  * An input whose minor dim is under 128 and whose XLA default layout is not
    row-major is passed as `jnp.transpose(x, perm)` (a bitcast); the kernel
    reads it through `_lane_dense_access` (static slices; integer reads on
    the lane dim, or on rows for 32-bit / even-row bf16 via u32 words
    unpacked by shift). Streamed per layer through a root-level ring whose
    slots sit side by side along the lanes (`(rows, depth * cols)`): Mosaic
    rejects 6-row slot slices of a 3-D ring, which interpret mode can't see.
  * Launcher kwarg `_lane_dense_perms={arg: perm}`; in-place outputs go back
    by the inverse perm.
  * TPU: TP8 exch device 2356.9 → **2243.5 µs/call** (body 2209.8 →
    2171.7), noex 2074.7 → 1970.7 (body 1928.1 → 1899.1); SC step body
    1928.2 → 1896.5 (device total 2044.1 → 1932.8); hybrid body 1834.2 →
    1799.3. The body gain (~30–38 µs) is the 6-row slot (~160 KB) replacing
    6 columns padded to 128 lanes (1.3 MB) per layer.
  * Remaining TP8 per-call copies: rec state 16.2 + 6.5 µs, KV caches 9.7 +
    5.8 + 14.2 + 15.6 µs (the communicating-kernel copies; amortized by the
    loop methodology), `bf16[48,1280,4]{1,2,0}` 2.0 µs (cause unknown).
  * Follow-ups: (0,2,1) stacks streamed per layer won't compile on TPU (needs
    a whole-VMEM fallback); bf16 integer rows copy two rows' words (layers
    i, i+1 could share one copy); no f16/int8 integer rows, no integer-row
    stores.
  * Merged-tree suites (serial, interpret; /tmp/mk/merge-f17c/): megakernel
    94 passed / 342 subtests, autotuner heuristics 94 passed / 56 skipped,
    test_pallas 245 passed, load_store 118 passed, memory_access 4 passed,
    test_examples 80 passed. Pyrefly: 396 errors, the environment's import
    baseline. Pod tree /root/helion-v87 (snapshot /tmp/mk/snap-v87).
  * **Probe results (v86, L=64 body µs):**
    * Grouped ring waits with the arena are neutral: wait group 2 → exch
      2213.8 / noex 1930.3, group 4 → 2214.9 / 1931.1 (base 2210–2216 /
      1928). Per-tile barriers are not the TP8 loss. Keep the knob off.
    * Chunked exchange does not help (`xchg_chunk_probe.py`, 10 dots
      (16, 2176) @ (2176, 512) then an 8-way all-reduce of the (1, 5120)
      row): compute only 3.63 µs/rep, exchange after the last chunk 7.65,
      each chunk sent as produced 7.81. The ~4 µs is completion latency
      after the last send, not transfer time.
    * Deferred writeback waits (`sink_wb.py`, fixed semaphore names) are
      neutral on v87: exch body 2177.6 vs 2173.2 µs. Writeback waits do not
      drain the ring.
  * **Per-root timeline, v87 L=64** (`scope_roots.py` + TLV=1;
    `/tmp/mk/tp8/tl87_{exch,noex}.log`; mean µs per instance):

    | root (src line) | exch | noex |
    |---|---|---|
    | L271 down proj + exchange 2 | 9.12 | 6.14 |
    | L260 gate/up | 7.45 | 11.22 |
    | L241 exchange 1 | 5.10 | — |
    | L191 GDN a/b projection | **1.78** | **0.33** |
    | L179 | 2.37 | 2.85 |
    | L100 | 2.88 | 3.56 |
    | L298 LM head | 103 | 98 |

    * gate/up recovers ~3.8 µs/layer of what the down/exchange root adds:
      with the exchange, the ring keeps filling during the round trip.
    * **L191 stalls 1.45 µs/layer (~70 µs/token) with the exchange.** Its
      a/b weights come through F17c's lane-dense ring (`ring_1`, not in the
      arena). `_hold_for_exchanges` holds that ring's next-layer copies at
      the MLP exchange and issues them in `_catch_up`, behind the remote
      copies and ~44 MB of arena copies already in flight; local DMA is
      roughly FIFO, so they land after L191 starts waiting.
* 2026-10-03 — **F23 fix: no exchange holds beside an arena (v88).**
  `_hold_for_exchanges` returns the rings unchanged when any ring is raw.
  The arena keeps the DMA engines busy through the round trip, and a copy
  held at the exchange queues behind the arena's copies in flight, landing
  after the roots right after the exchange read it. Test:
  `test_exchange_arena_other_rings_hold_nothing` (tp_gemv_pair_stack with
  r=64: `a` in a plain ring next to `b`'s arena; it had `_catch_up` refills
  of `a` before the fix).
  * TPU, TP8 L=64, same session (`/tmp/mk/tp8/ab88.sh`): exch body 2170.0
    (v87) → **2071.7 µs** (−98 µs, −4.5%), device 2241.7 → 2143.5; noex
    1899.8 (unchanged); top-1 OK. Timeline: L191 1.78 → 0.33 µs.
  * Suites (serial, interpret; `/tmp/mk/hold-fix/`): megakernel 95 passed,
    autotuner heuristics 94, test_pallas 245, load_store 118, memory_access
    4, test_examples 80. Pyrefly at the 396 baseline. Pod tree
    /root/helion-v88 (snapshot /tmp/mk/snap-v88).
  * Scalar-unit peers (`scalar_peer.py`: `peers[0, j]` vector roll/extract
    → `rank ^ (j + 1)` on the scalar unit): 2072.7 → 2070.4, neutral. Not
    worth an SMEM placement change.
  * **Where v88 stands** (`/tmp/mk/tp8/tl88_summary.txt`): real exch body
    2068 vs consumer-only (stub_ring) 1976.5, so ~92 µs of overlap loss is
    left, almost all in the down proj + exchange 2 root (8.80 vs 7.70
    µs/layer). The step is consumer-bound with exchanges (1976 > DMA-bound
    noex ~1900). Consumer per layer (stub exch, µs): exch1 4.99, exch2
    ~3.58, GEMVs ~16.7 (gate/up 7.16, down 4.12, GDN qkv 2.35, z 1.40, out
    1.43), GDN core 1.85, norms + reductions ~1.96. Levers: shorter
    exchange roots, fewer/fused small roots, GEMV rate.
  * **Loop methodology** (16 chained steps, same session, `v88b.sh`): ours
    v88 **2102.1 µs/token** exch / 2008.8 noex, vs the reference 2328 /
    2291 (its streamed bytes are 6.96 GB/rank padded, ours 6.46). That is
    **9.7% faster than the reference** with the TP exchange.
  * Probes on v88 (L=64 body µs, base exch 2071.7 / noex 1899.8):
    * `sink_wb` (writeback waits deferred to the buffer's next write):
      real exch 2072.6, consumer-only 1981.5 (base 1976.5). Neutral.
    * Arena slot 4 MiB (`MK_CONST=_ARENA_SLOT_BYTES=4194304`): exch 2169.9
      (+98). Bigger tiles mean fewer slots in flight; keep 2.5 MiB.
    * Arena slot 5 MiB: exch 2072.6 / noex 1901.0. Neutral; 4 MiB noex
      1895.8. The default stays.
    * Vector-only root loops fully unrolled (`unroll_vec.py`, 8 `pl.loop(0,
      40)` reduction roots): consumer-only 1976.5 → **1917.7** (L251/L290
      0.58 → 0.13 µs each), but real exch 2071.7 → 2070.7. Exchange send
      loops unrolled (`unroll_xchg.py`): 2072.0. Neutral.
    * **So the real step is bound by neither side alone.** Consumer-only
      (1918 with unrolled reductions) and DMA-only (noex 1900) are both
      ~150 µs under real exch (2071). Time taken off consumer roots outside
      the down proj just becomes waiting at the down proj (L271, +1.1
      µs/layer vs stub). Next probe: GEMV dots stubbed to zeros, which keeps
      DMA + exchange + small roots (`stub_dot.py`, `dma89.sh`).
    * `late_wb.py` (rec_state writeback started after exchange 1) crashed
      the core on exit (nonzero semaphore); not pursued, since sink_wb
      already showed writeback waits are not the cost.
* 2026-10-03 — **v89 (F27 packed rows) and the dot-stub probe.**
  * v89 = v88 + F27 (`/root/helion-v89`, snapshot `/tmp/mk/snap-v89`): exch
    body 2072.1 µs, noex 1896.5, top-1 OK. Neutral; kept as generically
    better code (bitcast row reads instead of window + roll).
  * **Dot stub** (`stub_dot.py`: the 26 GEMV `dot_general`s → zeros, so DMA,
    exchanges and the small roots remain): exch 1968.8 vs noex 1900.1 µs.
    With the consumer far below the DMA, the two exchanges per layer
    (~8.6 µs/layer of latency) hide down to **1.08 µs/layer exposed**.
    Timeline (µs/layer, exch vs noex): down + exchange 2 (L271) 14.42 vs
    6.33, gate/up (L260) 4.18 vs 11.64, exchange 1 (L241) 5.44, and the GDN
    roots after an exchange are free (L179 2.67 → 0.24, L185 2.28 → 0.14):
    the ring fills during the round trip as designed.
  * So real exch (2072) = DMA (1900) + 69 µs residual exposure + ~104 µs
    from the GEMV consumer having no slack (consumer alone 1976 with
    exchanges, 1418 without). Next: per-rank body times (`prof_all.py`,
    `skew89.sh`): the profiler only ever read device 0, and with an
    exchange every rank runs at the slowest rank's pace.
  * Reference exchange design (read only): synchronous start-all/wait-all
    one-shot; the newer version reduce-scatters in f32 and all-gathers in
    bf16. It costs the reference ~2% because its DMA-bound loop (2291 µs)
    leaves its consumer slack, not because its exchange is faster.
* 2026-10-03 — **Per-rank skew, unroll on v89, the reference verify, the MoE stack.**
  * Per-rank body times (`prof_all.py`, `/root/tp8/skew89.out`): noex, TPU:3
    runs 1971.1 µs mean, every other rank 1897.5–1906.9 (TPU:0 1898.3). exch:
    all ranks 2068.8–2071.9. With exchanges every rank runs at the slowest
    rank's pace, so real exch 2070 ≈ 1898 (DMA) + 73 (TPU:3, matches the
    stub_dot 69 µs exposure) + ~100 (GEMV consumer without slack). Our
    profiler only ever read device 0, which hid the slow chip.
  * v89 + unroll_vec (`/root/tp8/unr89.out`): real exch 2070.0 (neutral),
    consumer only (stub_ring + unroll) 1889.3 (v88 stub_ring 1976.5). The
    consumer is faster but the slow chip and the barrier still bind.
  * **Reference verify** (agent, `/root/verifyref/vr1.out`, jax 0.10): L=64
    block 8 **3.044 ms/call** (2627.8 processed tok/s), noexchange 2.437;
    same-day decode 2.327 ms/token (noex 2.289). The verify exchanges (129
    RS(f32)+AG(bf16) all-reduces of [8, H]) cost 0.61 ms against decode's
    0.04, so the reference verify is half exchange-bound; the 8-row compute
    adds 148 µs over noex decode. Design note: `/tmp/mk/verify-ref/VERIFY_DESIGN.md`
    (structure only, 15 generic capabilities, section 4).
  * **MoE decoder stack** (agent, `/root/moestack/moe{1,2,3}.out`; H 2048,
    E 128, K 8, I 768, 32/4 heads, ctx 2048; model 31.71 µs/layer at 3.72
    TB/s): mk 39.26 (L=8) / 39.16 (L=16) µs/layer, **XLA unrolled 90.66 /
    92.24 (2.3× slower)**. Fixed expert ids (`mk_ids`) 34.39 / 35.36, so the
    in-kernel gate (router → top-k → expert DMA) costs 3.8–4.9 µs/layer.
    No-DMA 25.86 (consumer has slack); no-compute 38.71 (the DMA schedule
    alone is the bound). No arena 38.01, no arena depth 12 39.27. **Split
    arenas** (a separate raw ring for producer-gated loads, harness patch
    `split_gated_arenas` in `/tmp/mk/moe-stack/run/run_moe_stack.py`): 37.87 /
    37.63, −1.5 µs/layer. Candidate F28 (generic; dense models have no gated
    loads so it is neutral for them). Timelines failed (`root_names`
    StopIteration in the harness), not rerun.
* 2026-10-03 — **F28 (gated loads take their own arena) in the compiler.**
  * v90 on TPU: MoE stack (H 2048, E 128, K 8) kernel body 37.85 / 37.62
    µs/layer at L=8 / 16 (v89 39.26 / 39.16, harness patch 37.87 / 37.63),
    3.12–3.14 TB/s active; numerics PASS. TP8 L=64 exch 2072.5 µs (neutral:
    dense, no gated loads). Local suites all green.
  * Per-device probe (`devspeed.py`): every core 3.11 TB/s HBM and 907
    TFLOP/s MXU, so TPU:3's 73 µs (noex 1971 vs ~1898) is not slower hardware.
    It is something about our kernel on that core: open item.
  * MoE agent backlog: bound the attention scan by `pos` (~1.2 µs/layer; also
    M9 capability 10); expert tiles (1.5 MB) sit in 2.6 MB arena slots.

* 2026-10-03 — **M9 verify kernel compiles and is correct in interpret mode; F29, F29b, F29c.**
  * `examples/qwen38_verify_step.py` (dev tree `/tmp/mk/m9/wt`; plain Helion source, no reference code) runs an 8-row block: embedding gather, GQA attention over the cache plus a causal block, block causal conv with per-row conv snapshots, a sequential delta rule over the rows with per-row recurrent snapshots, MLP, per-layer hidden taps, and per-row top-1. At small shapes (L 4, H 256) every output matches the torch reference (`/tmp/mk/m9/run_small.py`: ALL OK).
  * Three generic gaps fixed on the way: F29c (static_range slices), F29 (one stage per store site was 11.9 MB of VMEM), and F29b (the lane-dense conv snapshot was pinned whole in VMEM).
  * Seen in the generated code: a block of 8 KV rows written one row at a time does 8 serialized read-patch-write DMA round trips per cache. Candidate F30: combine the row stores at `base + c` into one 2-tile window read, in-VMEM patches, and one write.
* 2026-10-03 — **M9 verify kernel runs on TPU (single core); F30 and narrow packed rows.**
  * F30 (row-store runs) in main: the verify block's 8 KV rows per cache per
    layer are one 2-tile window read, 8 VMEM patches and one write.
  * TPU gap found and fixed (generic): Mosaic reads a bf16 32-bit word row at
    a runtime index only from refs of ≥ 128 lanes, at any row count (probes
    `/tmp/mk/tp8/probe_rows{,2}.py`). Narrow refs (per-layer `[L, heads]`
    tables kept row major, e.g. 6 GDN layers × 6 heads at L=8) are now read
    whole and the row masked-selected (`packed_row_load`).
    Test `test_runtime_row_gather_packed_narrow`.
  * v93, one core, DFlash block 8: **L=8 body 400.3 µs (50.0 µs/layer),
    NUMERICS PASS; L=64 body 2608.0 µs (40.75 µs/layer)** vs the reference's
    2437 µs noexchange (38.1 µs/layer): 7% behind on compute.
  * L=64 numerics: top-1 values pass (the 2 index mismatches are near-ties).
    Taps, conv/KV states exceed the fixed atol, but the relative error grows
    smoothly with depth (0.3% at layer 0 → 3.5% at 63) and evenly over rows:
    bf16 drift (~√64 × 0.4%), not a bug. The runner prints the breakdown.
  * Next: per-root timeline of the verify body to find the 171 µs; then TP8
    verify with the RS+AG exchange against the 3.044 ms gate.
* 2026-10-03 — **Verify gap closed on one core: F31 (stage ring) + F32 (store transpose cancellation).**
  * Per-root timeline of v93 (L8, region-trace flag on): the GDN conv +
    recurrence + snapshot root is 17.3 µs/layer. Inside it (scoped phases):
    - conv snapshots 5.46 µs: 8 stores serialized on one stage, plus
      transpose pairs;
    - recurrence 16 half-steps × 0.44 µs = 7.1 µs;
    - the rest about 1 µs.

    Timing-only ablations (`scope_gdn_ab.py`):
    - no transposes: 4.20 µs;
    - no waits: 1.89 µs;
    - both: 0.66 µs.
  * Scopes act as LLO scheduling barriers only with
    `--xla_enable_custom_call_region_trace=true`: 402 → 379 µs with the flag,
    no change without it. In jax 0.10 Mosaic, named_scope lowers to
    `tpu.trace_start/stop`. There is no optimization_barrier rule, and
    reloading the carried state from VMEM (`carry`) does not help.
  * **v94 (F31 + F32 store side), one core, DFlash block 8, no flag:**
    - **L=8: 400.3 → 357.9 µs, NUMERICS PASS.**
    - **L=64: 2608.0 → 2141.4 µs (33.5 µs/layer), vs the reference's 2437 µs
      noexchange: 12% ahead.**
    - L=64 numerics are bit-identical to v92/v93 (the same drift-only
      maxerrs and near-tie top-1 swaps).
    - GDN conv-snapshot phase: 0.08 µs/layer.
    - The GDN root is now ~8.1 µs/layer (was 17.3). The recurrence's 16
      half-steps at 0.44 µs each are most of it.
  * The barrier gain is gone. On v94, flag + scopes runs 358.8 µs and no
    flag + no scopes runs 357.9 µs. So the scopes' 23 µs on v93 came from the
    scheduler working around the serialized snapshot waits, which F31 removed.
    No barrier mechanism is needed.
  * Next:
    - F32b (load-side cancellation) to main;
    - TP8 verify against the 3.044 ms gate (fork agent: `/tmp/mk/tp8/tp_verify.py`,
      `run_tp_verify.py`).
  * F32b (load-side cancellation) is ported to main and dev. A lane-dense
    load whose users are all permutes undoing the transpose, directly or
    through `convert_element_type`, stays physical
    (`DeviceFunction.pallas_physical_values`), and those permutes return
    their input. Test `test_lane_dense_transposes_cancel`. Suites green on
    both trees (megakernel 99, pallas 246, load_store 120, examples 80,
    heuristics 94); pyrefly 396.
  * Bounded attention scan: `hl.tile(t + m)` with `t = pos[0]` already
    compiles to a dynamic `fori_loop` (`_num_iterations` from the bound,
    `@pl.when` priming, double-buffered K/V DMA). This needs a source change
    only, no compiler change, and the reference bounds its scan the same way.
    At pos=100, ctx=2048 the scan drops from 16 blocks to 1–2. Measured as
    v95 (v94 + F32b + bounded verify example).
* 2026-10-03 — **TP8 verify, first numbers (fork agent; `/tmp/mk/tp8/tp_verify.py`,
  `tp_verify_rsag.py`, `run_tp_verify.py`).** Both kernels are plain Helion
  source generated from the verify example (unbounded scan), on v94/v95
  compilers. They compiled first time: no compiler gaps.

  | exchange | L8 exch | L8 noex | L64 exch | L64 noex |
  |---|---|---|---|---|
  | all-to-all (1.15 MB/rank) | 612.0 | 365.2 | 4351.0 | 2240.3 |
  | RS f32 + AG f32 (287 KB/rank) | — | 363.5 | **3233.8** | 2178.5 |

  * RS+AG is 6% over the 3.044 ms gate. The exchange exposes 16.5 µs/layer,
    ~8 µs per all-reduce, against the reference's ~4.7 (3044 − 2437 over 128
    exchanges).
  * Numerics: ranks agree exactly; top-1 picks are within 0.1 of the best
    logit. The strict hidden check fails, but the layer-0 error is one bf16 ulp
    (0.009), the error is uniform across rows, and the states pass at L8: drift.
  * Generated code shows the GDN snapshot/state write-behind (~1.2 MB) drained
    at the end of the out-proj root, right before the reduce-scatter. The arena
    is 19 × 2.56 MB (48.6 MB, ~16 µs of HBM).
  * Next: bounded-scan TP kernel (`tp_verify_rsag_b.py`, `tpvb95.sh`) with a
    per-root L64 timeline (`TL=1` hook in `run_tp_verify.py`) to see where the
    exchange is exposed; AG in bf16 like the reference (half the AG bytes).
* 2026-10-03 — **v95 + bounded TP8 verify: 2% from the gate.**
  * v95 single core (v94 + F32b + `hl.tile(t + m)`): L8 356.7 µs PASS, **L64
    2080.6 µs** (v94 2141.4; ref noex 2437), with the same drift pattern. The
    bounded scan saves ~3.8 µs per attention layer: single core is HBM-bound,
    so only the skipped KV bytes count.
  * TP8 RS+AG with the bounded scan (`tp_verify_rsag_b.py`): **L64 exch
    3104.6 µs vs the 3.044 ms gate**, noex 2088.2 (3.09 TB/s). L8 exch 473.3.
  * Per-root timeline at L64 (µs, one GDN layer, 47 µs total):

    | qkv | z | b/a | conv + recurrence | out-proj | RS+AG | norm | gate/up | down + RS+AG |
    |---|---|---|---|---|---|---|---|---|
    | 2.44 | 1.50 | 0.32 | 7.31 | 1.48 | 8.83 | 0.17 | 8.71 | 16.07 |

    Gate/up eats 44.6 MB in 8.7 µs (5.1 TB/s): the arena fills during the
    exchange, so the DMA is not what is exposed. Chain estimate per GDN layer:
    dots ~15.5 + recurrence 7.3 + exchanges 17 + small ~2 ≈ 42 µs, against
    31 µs of DMA. TP8 verify is **chain-bound**, like the reference (47.6
    µs/layer average vs our 48.5).
  * Exchange rounds are ~4.3 µs of completion latency each; earlier probes
    found layout and chunking don't help. RS+AG = 2 rounds. The levers are
    therefore the rest of the chain:
    1. the GDN recurrence (7.3 µs × 48 layers ≈ 350 µs/call): VPU codegen of
       the unrolled delta rule. Generic: other recurrent models benefit. Probe
       `/tmp/mk/rec/rec_probe.py` tests reduce-then-re-expand (`sum(axis=-1)`
       then `[..., None]`, today's codegen) against keepdims.
    2. AG in bf16 (`tp_verify_rsag_b16.py`, source).
    3. TPU:3 skew (~73 µs on decode; every rank waits for the slowest).
  * **M9 gate met (body time): TP8 verify with RS(f32) + AG(bf16) and the
    bounded scan (`tp_verify_rsag_b16.py`): L64 3029.5 µs vs 3044.** L8 464.7
    (f32 AG 473.3). Numerics identical to the f32-AG run (the reduced chunk
    was rounded to bf16 before the add anyway). Source-only change. Caveat:
    our number is kernel body; check the reference's 3.044 ms methodology
    (`bench_qwen_ref_verify.py`) and run a STEPS loop for device time per
    call, as with the decode gate.
* 2026-10-04 — **M9 under the reference's methodology: 0.8% short; the
  recurrence is the lever.**
  * `run_tp_verify.py STEPS=16`: 16 chained calls in a jitted `fori_loop`
    (states donated and threaded, top-1 fed back as tokens, pos += B per
    call), best of 3, the same as `bench_qwen_ref_verify.py`. Same session
    (`tpvloop16.sh`):

    | L64 µs/call | Helion TP8 (`tp_verify_rsag_b16`) | reference |
    |---|---|---|
    | exchange | 3073.5 | 3048 |
    | noexchange | 2284.5 | 2437 (2026-10-03) |

    Loop vs profiler body: +44 µs/call with exchange (decode: +35). The noex
    loop is 196 µs above the rank-0 body: with no exchange, the call waits
    for the slowest rank (TPU:3 skew), whereas with exchange the skew is
    absorbed into the waits. We are 6% faster than the reference without
    exchange and lose it all in the exchanges (789 vs 611 µs/call).
  * Recurrence probe (`rec_probe.py`, [6,128,128] f32 state, ns/step):
    A today's codegen 692; B keepdims 668; **C 455** — both reductions on
    the *old* state, `pred = g·Σ(k⊙S)`, `o = g·Σ(q⊙S) + δ·(k·q)`, then
    `S' = g·S + δ⊗k`; D (C + keepdims) 684. C and A issue the same number of
    VPU ops, but C makes 2 sweeps over the 96-vreg state instead of 3 (it
    does not fit the 64-vreg file, so each sweep is vld/vst traffic), and `o`
    leaves the state chain. Worth ~90 µs/call at 8 steps × 48 GDN layers.
  * Plan: (1) measure C in the real kernel as a source variant
    (`tp_verify_rsag_b16c.py`); (2) if it holds, implement it as a generic
    FX rewrite — **reduction distribution over an affine state update**: for
    `S' = a⊙S + u⊗w` and a reduction `Σ_d(x⊙S')` with `a`, `u` invariant
    along `d`, emit `a⊙Σ_d(x⊙S) + u⊙Σ_d(x⊙w)` when `Σ_d(·⊙S)` already
    sweeps `S` (shares the sweep). It applies to every delta-rule /
    linear-attention / SSM recurrence (GDN, Mamba2, RetNet, mLSTM, RWKV7),
    not just Qwen3.8.
* 2026-10-04 (cont.) — **Reduction distribution pass implemented; the C form
  is slower in the real kernel (investigating).**
  * Source variant C (`tp_verify_rsag_b16c.py`) in TP8 is *slower*: L8 body
    473.8 vs 464.7; L64 body 3101.2 vs 3029.5; L64 loop 3147.6 vs 3073.5.
    Numerics: same bf16 drift pattern (top_idx 6/8 vs ref, ranks match).
  * Snapshot store is not why: probe with a per-step f32 snapshot store:
    As 693, Cs 461 ns/step. The Helion-generated b16c recurrence matches
    probe C op for op (same reductions on the old state, `g_b[:, :, None]`,
    small `k·q`). Remaining differences: the real kernel unrolls the steps
    (no `fori_loop`), casts each snapshot to bf16 into double-buffered VMEM
    rows plus an HBM DMA, and the recurrence is scheduled alongside the rest
    of the layer. Next: probe variants Ad/Cd (bf16 + DMA snapshot) and the
    standalone Helion kernel (`rec_bench.py`: off / depth1 / full).
  * Generic pass `helion/_compiler/pallas/distribute_reductions.py`
    (fast_math only, runs in `lower_to_device_ir` before
    `put_full_operand_first`): `sum(x·a_b·w, D) → a·sum(x·w, D)` and
    `sum((x + c_b·y)·w, D) → sum(x·w, D) + c·sum(y·w, D)` when `a_b`, `c_b`
    are `[..., None]` views invariant along `D` and `y·w` is smaller than
    the product. A rewrite-depth limit (`_MAX_DEPTH = 1`) stops at the
    step's input: from the unmodified source it produces exactly variant C;
    unbounded depth gives the chunked (WY-like) form with O(m²) cross terms.
    Tests: `test/test_pallas_distribute_reductions.py` (5 graph-level + 1
    e2e). All local Pallas suites pass.
* 2026-10-04 (cont. 2) — **Cause found: operand order of a lane reduction.
  v97 (lane-reduction-aware operand order + reduction distribution under
  fast_math): TP8 verify L64 loop 2988.9 µs/call vs ref 3048 (−1.9%).**
  * Bisect of the standalone recurrence (`rec_kernel.py`, M=32 unrolled,
    timed chained through the state with `chain.state_chain_time`, since
    `prof.chained_time` lets XLA hoist pure kernels): Helion depth1 992
    ns/step vs probe J 487 with the same jaxpr, except `sum[6,128] * g[6,1]`
    (Helion) vs `g * sum` (probe). Swapping those 64 products: 601 ns/step
    (pass off: 666). Mosaic gives an elementwise op the layout of its
    first operand; a lane reduction's result is laid out down the sublanes
    (16 vregs per head here instead of 1/8 of one), so the whole tail of the
    step ran in that layout. `put_full_operand_first` (ours, F-series) had
    flipped the source's `g_b * sum(...)` — which is also why the b16c
    source variant lost.
  * Fix (generic, `pallas/operand_order.py`): first operand = not
    lane-reduced (a reduction over the last dim, propagated through the
    first operand of pointwise ops), then full-shape. Test
    `test/test_pallas_operand_order.py`.
  * TP8 verify, v97, L64 (`tpv97.sh`):

    | | body µs | loop µs/call |
    |---|---|---|
    | b16 | 3029.6 | 3070.7 |
    | **b16 + fast_math** | **2945.8** | **2988.9** |
    | reference | — | 3048 (earlier session; same-session bracket queued) |

    L8 bodies: 464.5 / 454.0. The operand rule alone leaves b16 unchanged.
    Numerics with fast_math match the baseline: every check passes but the
    strict hidden one (failing since the first TP8 kernel: drift 0.9% →
    2.5% over 8 layers, maxerr 0.562 vs 0.578), rec_snap 2.0e-3 vs 2.4e-3,
    the one extra top-1 difference is a near-tie (picked logit within 0.022
    of the best). Todo: an f32 oracle to bound bf16 drift for both us and
    the reference.
  * Remaining gap to probe C/G in the standalone kernel: 601 vs ~410–487
    ns/step; next lever.
  * Same-session bracket (`tpvcmp.sh`): Helion b16 + fast_math 2987.9,
    reference 3043, Helion 2988.4 µs/call. **M9 met: 1.8% faster.**
  * The b16c source variant on v97: body 2938.1, loop 2980.7. The pass on
    the unmodified source recovers all but 8 µs of the hand rewrite.

### 2026-10-04 — standalone recurrence gap was a harness artifact

* The earlier "601 vs ~410–487 ns/step" compared the probe at 640 chained
  reps with Helion at 50 reps. 50 reps include ~120 ns/step of jit
  dispatch at M=32. In one script (`rec_hack.py`, `PROBE=J,Jo,A`,
  50 reps): Helion depth1 bare 618 (vmem out 638), off bare 683,
  hand probe J 609, Jo 610, A 797. Helion is within 1.5% of the best
  hand-written Pallas form, and its off form beats hand A.
* CSE of the duplicated view assignments in the generated code: 613, so
  it is noise. There is no codegen gap left in the recurrence.
* Rule: compare kernels only within the same harness and rep count.
* Next: an f32 oracle for M9 numerics (bound bf16 drift for ours and the
  reference), then M10 batch decode.


### 2026-10-04 — M9 numerics bounded by an f32 oracle

`run_tp_verify.py F32REF=1` reruns the qwen oracle with every bf16 input
upcast to f32 and `jax.default_matmul_precision("highest")`, then reports the
error of the bf16 oracle and of the kernel against it (relative to the f32
oracle's max magnitude). L8, b16 + fast_math:

| tensor | bf16 oracle max / mean | kernel max / mean |
|---|---|---|
| top_val | 0.0067 / 0.0038 | 0.0058 / 0.0028 |
| hidden | 0.0205 / 0.0017 | 0.0203 / 0.0020 |
| rec_state | 0.0115 / 0.00025 | 0.0105 / 0.00031 |
| rec_snap | 0.0052 / 0.00010 | 0.0072 / 0.00013 |

Top-1 vs the f32 oracle: the kernel and the bf16 oracle both get 7 of 8 rows,
but miss different rows. The strict "hidden" check (kernel vs bf16 oracle,
maxerr 0.56) is two bf16 computations drifting apart; each is equally far
from f32.

L64 (full depth): hidden max/mean 0.0507/0.0043 for the bf16 oracle vs
0.0542/0.0049 for the kernel with fast_math (0.0588/0.0049 without);
top_val 0.0218 vs 0.0165. Top-1 vs f32: bf16 oracle 7/8, kernel 6/8 with
or without fast_math. Per-layer hidden drift grows at the same rate, with
the kernel 5–15% above the oracle. fast_math does not add error. M9
numerics: ✅ (same order as the reference oracle; a ~10% higher mean error
is a watch item, likely from bf16 rounding of intermediates the oracle keeps
in f32).

### 2026-10-04 — M10 plan: batch decode (B separate sequences)

The public reference has only a B=1 decode megakernel (`transformer_stack`).
Its batch chart (1.4–2× GB200 at B 1–8) has no public kernel, so the gate is
our own scaling: **per-call time at B=8 ≤ 1.15× B=1, and B=1 ≤ the
reference's 2330 µs**, plus beating the XLA TP8 oracle loop.

* Source: `/tmp/mk/tp8/tp_decode_batch.py`, the M9 rsag kernel with a batch
  dim on every state (`conv_state [Lg, B, ...]`, `rec_state [Lg, B, ...]`,
  `k/v_cache [La, B, kv, ctx, d]`, `pos [B]`). Attention and the delta rule
  run per sequence (`hl.static_range(B)`), and the GEMVs take B rows.
  Harness: `run_tp_batch.py` (vmapped qwen oracle, `XLA=1` for its loop
  time).
* Compiler gaps found while writing it:
  * **F33 ✅** — multi-dim narrowing of a device value (`v[b, :, :k]`):
    `device_ir._split_narrowing` splits the subscript into one per narrowed
    dim (right to left), then adds the `None` dims. Test:
    `test_multi_dim_narrowing_of_value`.
  * **F34 ✅** — several slice loads of one HBM tensor per root
    (`rec_state[l, b]` per sequence). M10 L8 B8 first light had failed with
    rec_state (37.7 MB) whole in VMEM.
  * **F35 ✅** — lane-dense read-write narrow state (`conv_state`) in HBM.
    L64 B8 had failed with conv_state whole in VMEM. The lane-dense-off
    config is now dropped when it cannot fit, instead of failing the compile.
  * **F36 ✅** — exchange trims round up to whole tiles. M10 B1 had failed
    Mosaic compile on `pl.ds(0, 1)` of a 16-row bf16 gather buffer.
  * Harness: the XLA-only path of `run_tp_batch.py` now adds
    `/root/tpu-megakernels` to `sys.path` itself.
  * **First light (v98, `tpbatch2`, FM=1):**
    * L8 B8: every row's top-1 matches the reference and the states are within
      tolerance. The hidden check fails (max err 0.56, rel err 1.3% → 2.0% over
      8 layers). **f32 oracle (`tpbatchdrift.sh`, `F32REF=1`): bf16 drift,
      not a bug.** Against f32, the kernel's hidden error equals the bf16
      oracle's: max rel 0.0158 vs 0.015, mean 0.0017 vs 0.0015, the same
      growth per layer. States agree too, and top-1 is 8/8 for f32, the
      bf16 oracle and the kernel. Same with fast_math off.
    * L64 body: B1 2570.2 µs (loop 2598.5 µs/call), B2 2758.7 (2794.1), B4
      2997.3 (3040.1), **B8 3472.6 (3531.4 = 441 µs/token): 1.35× B1, so the
      gate (≤ 1.15×) is missed**. Each extra sequence costs ~1.5–2 µs/layer.
      XLA TP8 loop B1: 5818.2 µs/call (2.2× slower than ours); B8 rerun
      pending (harness print bug).
      B1 is ~500 µs above the M7 decode kernel. Likely cause: the batch source
      took RS+AG from the M9 verify kernel, i.e. 2 exchange rounds (~4.3 µs
      each) per all-reduce instead of M7's one-shot round. One-shot source
      variant: `tp_decode_batch_os.py` (generated by `gen_batch_os.py`), job
      `tpbos`.
    * **v99 (F37, `tpbatch99`):** slice copies wait only for ring writes that
      overlap them, so sequence b's state write stays in flight while b+1..
      are read. B8 body 3472.6 → **3282.4 µs** (loop 3340.9 = 417.6
      µs/token); B1 unchanged at 2570.4. B8/B1 = 1.277, still above 1.15.
    * Per-stage timeline, B1 → B8 (µs per occurrence; `tp_decode_batch.py`
      lines). The stages that grow with B:

      | Stage | B1 | B8 | × layers | Δ per call |
      |---|---|---|---|---|
      | L135 attention (per-b bounded scan) | 3.13 | 18.41 | 16 | ≈ +245 |
      | L218 conv + delta-rule recurrence | 0.92 | 5.33 | 48 | ≈ +210 |
      | L306 gate/up GEMV | ~7.8 | ~11.6 | 64 | ≈ +245 |
      | prologue | 5.98 | 19.06 | 1 | +13 |
      | L268 / L315 exchanges | ~8–15 | ~8–15 | — | flat |

      Attention and the recurrence are serial per-b chains: each sequence's
      DMA starts only after the previous sequence's compute. Gate/up grows
      with B rows (8-row padding means B8 should be free; check the
      tile/row layout).
    * **XLA TP8 loop B8: 6257.3 µs/call** (782 µs/token). Ours (v99) is 1.87×
      faster at B8 and 2.2× at B1.
    * **One-shot variant (`tpbos`, v99) is much slower, not faster:** B1 body
      3819.0 (RS+AG 2570.4), B8 4717.7 (RS+AG 3282.4). Same numerics as RS+AG.
      Together with the M7 step below (also one-shot), this points at the
      one-shot exchange path on recent trees, not at RS+AG's extra round.
    * **v100 (F39, `tpb100`):** the per-sequence attention scans' first
      K/V tiles start during the previous layer. B8 body 3282.4 →
      **3111.7 µs** (loop 3173.4 = 396.7 µs/token), B1 2570.4 → 2532.1.
      B8/B1 = 1.229. L8 B8 numerics unchanged (top-1 8/8, the same bf16
      hidden drift). B8 timeline: attention L135 18.41 → **7.88 µs** per
      layer (B1 3.13). Unchanged: recurrence L218 5.3, gate/up L306 11.5,
      exchanges L268 8.0 and L315 15.0–15.6, prologue 19.3.
    * **Bisect (`tpstepbisect`, M7 tp_step L64 body):** v90 2071.3, v95
      2071.8, v98 3572.0, v99 3571.5. The regression lands in v95..v98
      (F34–F36).
    * **Root cause (F40):** the v90 → v99 diff of the generated tp_step
      code shows the recv pushes going from `pl.ds(0, 1)` to `pl.ds(0, 8)`:
      F36's round-up gives 8x the f32 exchange payload. The f32 RS chunks of
      M10 B1 (and one-shot's f32 pushes) paid the same. Fixed by F40; v101
      job `tpb101` re-measures M7 and M10 B1/B8 RS+AG and one-shot.
    * **v101 (F40, `tpb101`) confirms it.** M7 tp_step L64 body back to
      **2071.9 µs** (v99 3571.5). M10 L64 bodies (loop µs/call in
      parentheses):

      | all-reduce | B1 | B8 | B8 µs/token |
      |---|---|---|---|
      | RS+AG (`tp_decode_batch`) | 2419.5 (2449.7) | 3113.0 (3169.9) | 396.2 |
      | one-shot (`tp_decode_batch_os`) | **2163.9** (2194.8) | 4567.7 (4628.1) | 578.5 |

      One-shot wins at B1 by 256 µs: one exchange round instead of two,
      about 2 µs × 128 all-reduces. It loses at B8 by 1455 µs: each rank
      pushes its whole f32 partial (8 × 5120 × 4 B = 160 KB) to 7 peers,
      ~11 µs more per all-reduce, so it is bandwidth-bound. RS+AG pushes 20
      KB f32 chunks and gathers in bf16. **The B1 gate (≤ 2330) is met by
      one-shot.** No single source is best at both sizes, which is the case
      for F38 (all-reduce algorithm as a compiler/autotuner choice by payload
      bytes). Best per B: 2163.9 / 3113.0 = 1.44×. L8 B1 numerics: top-1
      differs (53116 vs 204229) with top value within 0.012 and the hidden
      drift curve of B8. Likely a near-tie under bf16 drift; f32-oracle check
      pending.
    * **Possible regression:** the M7 decode step (`one.sh 64 0`,
      `tp_step.py`) on v99 measured a 3571.7 µs body, against 2071.7 on
      v88 and 2072.5 on v90. Bisect job `tpstepbisect` (v90/v95/v98/v99).
    * **Reference scaling for context:** the reference's Qwen decode
      megakernel is single-sequence (scalar token and position). Its README
      batch chart (end to end, server included) reads 249 / 392 / 515 / 865
      tok/s aggregate at B 1 / 2 / 4 / 8, i.e. B8 step ≈ 2.3× the B1 step.
      Our kernel body is at 1.28× (v99). The 1.15× gate is our own stricter
      target, not a parity requirement, so M10 work should favour
      optimizations that also help other models (per-sequence chain overlap,
      exchange choice) over Qwen-only tuning.
    * Candidate F38: an all-reduce primitive whose algorithm (one-shot vs
      RS+AG, by payload bytes) is a compiler/autotuner choice instead of
      source. Today every model hand-writes its exchange.
    * **Where B1 → B8 grows (v101 RS+AG timelines, `tptl101`).** Body 2426.7
      vs 3116.3 µs; one period (4 layers) 147.5 vs 187.4 µs. Per-root deltas
      (B8 − B1) × count per call:
      | root | B1 | B8 | Δ × count |
      |---|---|---|---|
      | L306 gate/up GEMV | 7.5 | 11.7–11.9 | +4.3 × 64 ≈ 275 |
      | L218 GDN recurrence | 0.92 | 5.34 | +4.4 × 48 ≈ 212 |
      | L135 attention | 0.95 | 7.89 | +6.9 × 16 ≈ 111 |
      | L268 / L315 out-proj + exchanges | 7.3 / 14.3 | 8.0 / 15.1 | ≈ +0.8 × 128 ≈ 100 |
      | prologue | 6.6 | 20.1 | +13.5 |
      The exchanges barely grow (RS+AG). Gate/up is the biggest. The weight
      stream is the same at B8, so 4.3 µs per layer of compute, or of stalls,
      grows with the rows. The next thing to check is the B8 gate/up codegen
      (MXU vs VPU, tile config, f32 activation work on padded rows).
    * **F38 (`hl.all_reduce`, v102).** M10's 128 hand-written exchanges are
      now `hl.all_reduce(red, partial, peers[0, :], rank[0])` plus a residual
      root. Local codegen picks one-shot at B1 (18 recv refs) and RS+AG at B8
      (16 scatter refs). TPU `tpb102` checks numerics (L8) and L64 bodies
      against the v101 hand-written ones (one-shot B1 2163.9, RS+AG B8
      3113.0). Each all-reduce may cost a little more: one local copy of the
      partial and one extra root (sum, then residual). A generic
      copy-forwarding / root-fusion pass would recover it.
    * **F41 (shared slice buffers, v103).** B8's per-layer recurrent state
      took 24 slice buffers (9.4 MB) and capped the arena ring. Roots now
      share them when that deepens the rings: L8 B8 has 8 rec_state
      buffers and a ring of 9600 tiles instead of 7680 (L64: 5120 vs
      3840). TPU `tpb103`: L64 B8 3144.3 → 3137.9 µs. A third more ring
      gives 0.2%, so B8 is not ring-capacity bound and sharing the K/V
      loop-prime buffers (8 MB) waits. `tpb105` checks the other way
      (48 MB limit, ring 2560 tiles) and takes a v104 B8 timeline.
    * **F42 (one-shot without the self copy, v104).** The sum root picks
      this rank's term from `src`, so the per-site local copy into `recv` is
      gone. The first form (iota `where` over a 3-D block, then `sum(0)`)
      cost +392 µs at L64 B1 on TPU: lesson, keep selects scalar-predicated
      and 2-D in hot roots. F42b (v105, a static loop over ranks) is still +14.7 µs
      (`tpb106` 2212.4): reverted to the v103 copy. Next on the all-reduce
      overhead: fuse the residual root into the sum root (generic
      adjacent-root fusion when tiles match).
    * **B8 ring sensitivity (`tpb105`, v104 L64 B8).** Body vs ring slots
      (2.6 MB each): 4 slots (48 MB limit) 4470.5 µs, 6 slots (v102)
      3144.3, 8 slots (v103+) 3137.9. A knee at 6, flat after. At a 40 MB
      limit the layout splits into several rings and the body is 10558.5
      µs: the default config falls off a cliff under VMEM pressure (open
      risk; the autotuner's viability check should reject it). v104 B8
      timeline vs v101: recurrence 5.36 (same), gate/up 10.65–11.3 (was
      11.7–11.9), down + RS+AG 15.1–15.7, and two new residual roots of
      0.23–0.29 µs per layer from `hl.all_reduce` (≈ 32 µs per call; a
      root-fusion candidate). `tpb107` runs B1 with B8's ring depth.
    * **Lane reductions (`lrp`, Pallas probe, 8 seqs × 64 steps).** The
      delta-rule update per sequence (6 × 128 × 128 f32): sums over the
      lane dim as M10 writes them 0.938 µs/step; one in-kernel transpose
      of the state so both sums run over sublanes 0.663 (−29%); the state
      stored transposed 0.404 (−57%). Candidate generic lowering: a value
      reduced over lanes in several products with broadcast rows is
      transposed once and reduced over sublanes. `lrp2` adds a transpose
      per reduction and the two sums as one batched MXU einsum (HIGHEST).
    * **Ring depth at B1 (`tpb107`, v105).** B1 body vs VMEM limit: 64 MB
      (7040-tile ring) 2212.4, 56 MB (B8's 8 slots) 2341.6, 52 MB 2394.1.
      B1 does feel the ring; B8 (6 → 8 slots flat) is bound elsewhere, but
      B8's ring is 2 MB short of B1's (5120 vs 7040 tiles), because of
      per-sequence buffers: 16 K/V loop-prime buffers of 512 positions
      (8 MB, vs 1 MB at B1) and 10 state slices (3.75 MB). `tpb109` (80/96
      MB limits): Mosaic rejects them, VMEM capacity is 64 MiB, so a deeper
      B8 ring needs smaller per-sequence buffers (prime only what the next
      loop needs, share slots across sequential loops). `tpb110` (v106,
      attention block): B8 512 → 3211.4, 256 → 3175.5 (−36), 128 → 3214.0;
      B1 128 → 2209.7 (= 512). Freeing 6 MB of K/V buffers (128) does not
      help B8, so the ring is not its limiter; 256 trades rows loaded
      against loop iterations at these positions. A position-dependent
      tuning choice (autotuner), not a compiler change; K/V prime sharing
      stays low priority. `tpb110`: the attention block at 128/256 instead of
      the default 512 (positions are 100 + 37 b, so 512 loads up to 4x the
      visible rows and sizes those buffers).
    * **F43 (`reduce_over_sublanes`, v106).** Generic FX pass
      (`pallas/sublane_reductions.py`, after `distribute_affine_reductions`):
      a 32-bit tile whose last two dims are static multiples of 128 and
      that feeds ≥2 `sum(tile * row, -1)` with `row` invariant along the
      second-minor dim is transposed once; each sum becomes
      `sum(tile_t * row_t, -2)` (`row_t` is the same subscript with the
      `None` moved last, else a permute). M10's recurrence now matches
      probe variant B. Tests: delta step, two mat-vecs, single-sum and
      unaligned/bf16 negatives. `tpb108`: L8 B1 numerics, L64 B1 vs
      `tpb106`, L64 B8 vs 3137.9. **Result: B1 2211.9 (= v105 2212.4), B8
      3211.4 (+73 vs 3137.9)**, numerics as before: the probe's −29% does
      not carry over inside M10. `tptl106`: the B8 recurrence root went
      5.36 → 6.89 µs (0.67 → 0.86 µs per sequence; B1 0.91 unchanged).
      **F43 removed** (code kept in `/tmp/mk/f43_removed`). Lesson: the
      probe chained each sequence on the last (`c = o * 1e-3`), so it
      timed latency; M10's sequences are independent and the scheduler
      overlaps them, so throughput decides, and there two lane sums beat
      a transpose plus sublane sums. Probes must keep the kernel's
      independence structure. `lrp2` (µs/step): A 0.940, B 0.665,
      B2 (a transpose per product) 0.678, E (batched MXU einsum, HIGHEST)
      1.494, C 0.405. The transpose is cheap and the lane reduction is the
      cost, so `lrp3` asks whether a single reduction pays too (2-D and
      3-D tiles); if so `_MIN_REDUCTIONS` drops to 1.
    * **B8 vs B1 per GDN layer (`tptl106`, v106; recurrence as v104).**
      down + AR 15.3 vs 10.4 (+4.9), recurrence 5.36 vs 0.91 (+4.45),
      out-proj + AR 8.0 vs 5.3 (+2.7), gate/up 11.3 vs 8.7 (+2.6); the
      rest equal. ≈ +13.4 µs × 48 GDN layers + attention (7.9 per layer at
      B8). Both AR roots grow: B8 auto-picks reduce-scatter + all-gather
      (f32 [8, 5120] partials, 160 KB), B1 one-shot. `tpb111` (v103 = the
      worktree after the F42/F43 reverts) forces one-shot at B8 to check
      the cost model's break-even. (`tpb109`/`tpb110` ran on v106: compare
      within each job only.)
    * **B8 gate/up root codegen equals B1's** (v106 L64, B8 with the
      128 attention block so both rings are 7040 tiles): the L263 root
      differs only in the row bound (`< 8` vs `< 1` masks, grid of 8 vs
      1 rows); the matmuls are [16, 512] × [512, 2176] in both. So its
      +2.6 µs per layer is time spent waiting on its ring tiles, not
      compute. One cause is inherent: the B8 recurrent-state traffic
      (6.3 MB per GDN layer, ≈ 2.3 µs at 2.69 TB/s) shares HBM with the
      weight stream. The reference has no batch-decode kernel (its
      `rec` is [nl, vh, vd, kd], the same layout as M10's), so the 1.15×
      gate is our own target.
    * **`tpb111` (v103): the cost model's B8 pick is right.** L64 B8
      forced one-shot 4303.3 µs vs reduce-scatter + all-gather 3138.7
      (auto = RS+AG, 3137.9 in tpb103). L8 B8 one-shot numerics equal
      RS+AG bit for bit (top_val maxerr 0.0419, hidden 0.562 vs the bf16
      oracle, as tpb102/tpb103). So the two AR roots' +7.6 µs per layer
      is RS+AG's second round (≈ 3.9 µs each), not a wrong choice.
    * **Where B8's +941 µs goes (`tpb112`, v103 L64 B8, timing-only
      ablations that cut a per-sequence loop to one sequence).** Full
      3139.2; recurrence loop × 1: 2845.7 (−293.5, 6.1 µs per GDN layer,
      0.87 µs per extra sequence: about the root's compute plus its state
      traffic, both exposed); attention loop × 1: 2892.8 (−246.4, 15.4 µs
      per attention layer, 2.2 µs per extra sequence); both: 2591.0
      (−548, additive). The rest, 2591.0 − 2197.7 (B1) = 393 µs, is row
      growth: RS+AG's second round and the GEMV roots' waits. Ring depth
      does not matter at B8, and the ablations add up, so B8 is bound by
      the TensorCore's serial time, not by the weight stream. Each
      sequence's attention is one `fori_loop` (dynamic trip count, mostly
      1 here: POS 100 + 37 b < the 512 block) reached through scratch
      carries, and each loop is a scheduling barrier, so the 8 chains
      (wait K/V, forward the new row, QK, softmax, PV) run back to back.
      Candidates: (1) horizontal fusion of the sibling scans of an
      unrolled `static_range` into one loop over the longest trip count,
      with each sequence's carries kept by a scalar-predicated select past
      its own count (generic: per-sequence/per-expert loops); (2) the
      recurrence's lane reductions on the MXU (`rp1` probe: batched matvec
      of the state against [k, q] at HIGH/HIGHEST/split-bf16 vs lane sums,
      8 independent sequences per step).
    - `rp1` (Mosaic probe, 8 sequences × 6 heads × [128, 128] f32 state,
      µs per sequence step): lane sums 0.695; MXU HIGHEST 1.376 (rel err
      2.5e-7); split-bf16 ×3 1.182 (4.6e-6); bf16 DEFAULT 1.008 (2.4e-3,
      too coarse anyway); HIGH is not lowered. **Candidate (2) rejected**:
      the matvec against a 2-column operand wastes the MXU, and the lane
      sums M10 emits are the fastest form.
    - `sp1` (Mosaic probe, 8 online-softmax scans over VMEM K/V, block
      512, µs per step of 8 sequences): back-to-back loops 3.575, fused
      loop with predicated carries 2.754, straight-line one tile each
      2.677 (lower bound); at POS 900+ (2 trips each) 8.266 vs 7.340.
      Fusion saves ~0.1 µs per sequence per layer (≈ 13 µs per token at
      L64 B8), far below the 2.2 µs per sequence the kernel pays, so the
      loop barrier is not the cost. `tpb114` ablates the rest on the
      real kernel: the scan as a static one-trip loop (`abl_static`) and
      the cache row stores dropped (`abl_nowrite`; the 8 sequences share
      one row stage, so each waits for the previous one's HBM write).
    - `tpb114` (v107 L64 B8, baseline 3138.2): `abl_static` 3119.5
      (−19 µs: a static one-trip scan, i.e. no loop barrier, is worth
      0.15 µs per sequence per attention layer, as `sp1` said);
      `abl_nowrite` cannot launch (an input never written breaks the
      `jax_fn` arg map), so `tpb115` runs `abl_write1` (only sequence 0
      stores its row).
    - `rp2` (Mosaic probe): the recurrence step with the state stored
      [h, k, v], both reductions over sublanes and the update an outer
      product with k on sublanes, takes **0.307 µs per sequence step vs
      0.695** for lane sums over [h, v, k] (rel err 4.6e-7). The layout
      of a state the host owns is not the compiler's to change (F17c takes
      only XLA's default layouts, never a copy), so `rp3` measures the
      generic form: transpose the [v, k] state in the kernel around a step
      that reduces over sublanes (`TT`).
* Expected cost model at B=8: weights are the same as B=1. Per call and
  rank, recurrent state moves 48 × 8 × 393 KB × 2 ≈ 302 MB (≈ 100 µs at
  3 TB/s), versus 38 MB (≈ 13 µs) at B=1. KV reads are
  8 separate caches. Likely compiler work:
  * prefetch each sequence's state slice during the previous sequence's
    update (pipelining a static_range over HBM slices);
  * overlap the per-sequence attention DMA with compute.
