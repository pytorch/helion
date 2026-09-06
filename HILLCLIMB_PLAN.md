# Hillclimb: dense attention 2x32x32768x64 fp16, HELION_BACKEND=cute, GB300

Run name: `attn-dense32768`
Start (T0): 2026-09-04T04:25:20Z (epoch 1788495920)

## Target

Single variant, as requested by the user:
`z=2 h=32 seq_len=32768 head_dim=64 dtype=float16 causal=0 biased=0`
(`num_kv = 32768/128 = 256` KV tiles, i.e. the `num_kv=256` dense tuning policy.)

Hardware: NVIDIA GB300 (sm_103), 1400 W power limit, 4 GPUs, otherwise idle.
Env: `conda run -n helion-gb300-reference` (torch 2.15.0.dev20260824+cu130,
CuTe/nvidia-cutlass-dsl 4.7.0, triton 3.8.0, pytest available).

FA4 baseline needs:
`HELION_FA4_ROOT=/home/shangdiy/flash-attention PYTHONPATH=/home/shangdiy/quack-cutlass-4.7`

## Baselines

Every number below is the pooled median of all do_bench samples from an
interleaved run on one idle GPU. See the measurement hazard note in
`benchmarks/cute/hillclimb/attn_dense32768/README.md`.

| impl | median TFLOP/s | artifact |
| --- | ---: | --- |
| torch SDPA (cuDNN 9.20.0.48) | 485.3 | inline, GPU 0 |
| FA4 `fa4-v4.0.0.beta23` | 1244.9 | `final_fa4_*.json` |
| helion-cute (compiler seed) | 1269.5 | `tri_at1_*.json` |

Best baseline = FA4. torch SDPA is 2.6x behind and never competitive here.

## Iteration log

- (init) First baselines looked catastrophic (helion 571 vs FA4 1229). That was
  a **measurement artifact**: GB300 GPUs on one board share a power/cooling
  envelope, so a busy sibling GPU roughly halves the clocks of the GPU under
  test. Same kernel, same config: 1294 TFLOP/s on a quiet board, 564 with a
  neighbour at 826 W. All later numbers interleave Helion and FA4 on one GPU.
- step00 (T+1h05m) baseline on GPU 3, interleaved x3: helion-cute 1269.0,
  FA4 1245.1 -> **ratio 1.019**. Artifacts `g3base_*.json`.
- at1 (T+2h04m) cold full autotune, fresh seed, cold caches, 7407 s,
  ~1800 candidates: converged onto *exactly* the compiler seed, 1266 TFLOP/s.
  Initial population median 17.3 ms vs 13.3 ms best -> the space is a cliff.
- Diagnosed the cliff: the resident/packed dense softmax lowerings were gated
  on a byte-exact 22-field match against the promoted seed, so every neighbour
  silently fell back to the standard body (~14% slower, all perturbations
  landing on the same 16.0 ms).
- step01 (commit c7ba4dad) replaced the seed match with the lowering's real
  structural preconditions. Neighbours now hold 13.34-13.85 ms; all six
  promoted seeds are byte-identical in generated source.
- at2 (T+9h05m) cold full autotune re-run against the smooth space, 6528 s:
  this time the search *moved*, landing on a config 3 fields off the seed
  (`e2e_offset` 5->4, `other_regs` 40->48, `precompute_qk_desc` True->False).
  Interleaved 3-way vs the seed and FA4 (n=36 each): seed 1269.5, new 1274.0,
  FA4 1245.0 -- the new config is even with the seed on this shape, so the
  change buys reachability, not speed here.
- Chased the remaining 3.8% spread: Helion is **bimodal** at 13.30 / 13.85 ms.
  `nvidia-smi` during a run shows throttle reason `0x4` (SW power cap), power
  pinned at ~1365 W of a 1400 W cap, and the SM clock oscillating 1935 <-> 1725
  MHz. FA4 hits the same cap but only sags to 1912 MHz and stays single-moded.
  Helion draws more power per clock and still wins on wall time. This is a
  machine power-state effect, not a compiler lever.
- step02 (commit 77af2e41) found and fixed a second cliff of the same family:
  `rescale_threshold=12` (a value the autotuner searches) overflowed the fixed
  probability shift of 7 and dropped the lowering -- 15.89 ms vs 13.30 ms.
  Fitting the shift to the threshold keeps the lowering at every threshold.
- Final measurement (T+10h30m), 5 interleaved rounds, n=45 per impl:
  helion-cute 1274.5 TFLOP/s vs FA4 1244.9 -> **ratio 1.024** (1.060 comparing
  each kernel's fast mode). `check_goal.py`: GOAL MET.

## Baseline correction (T+13h, user-supplied)

The user pointed at P2489069734: FA4 autotuned for this exact shape. Its plan
is m_block 128, **n_block 160**, q_stage 2, **persistent**, **one CTA**, no
CLC -- quite unlike what `flash_attn_func`'s built-in heuristic picks. Added it
to the harness as the `fa4-tuned` impl with `--fa4-config` for ablation.

Interleaved on one idle GPU 3, n=36 samples per impl:

| impl | TFLOP/s |
| --- | ---: |
| FA4 default heuristic | 1244.9 |
| FA4 tuned, n_block=128 | 1349.5 |
| FA4 tuned, n_block=160 | **1375.8** |
| helion-cute (2-CTA, step02 config) | 1276.1 |

So the real baseline is 1375.8, not 1244.9, and the honest ratio at step02 is
**0.928**. Two lessons: my earlier "Helion beats FA4 at n=128" claim came from
comparing across batches and was wrong; and the baseline was under-tuned, which
the goal contract explicitly warns about.

Ablating the tuned plan one field at a time (each vs the tuned plan, same
batch): two-CTA instructions **-9.8%**, non-persistent -1.7%, n_block 128
-1.9%, CLC scheduler +0.0%. FA4's edge is mostly **one CTA**.

## Step 03: the resident softmax on the one-CTA pipeline

Helion's best config is two-CTA, and its one-CTA family was stuck on the
standard softmax body because the specialized lowerings were gated to
`fa4_2cta`. Nothing in the bodies needs two CTAs -- the *causal* resident
lowering already runs one-CTA -- so the gate was arbitrary.

| config | ms (median of 4) |
| --- | ---: |
| fa4 (1 CTA), before | 16.70 |
| fa4 (1 CTA), after | 13.45 |
| fa4 (1 CTA), kv_stage=11 | **13.41** |
| fa4_2cta (unchanged) | 13.81 |

The one-CTA path is also far steadier (0.5% spread vs over 4%), because the
two-CTA path oscillates against the 1400 W cap.

Knobs swept on top of the one-CTA path and found neutral or worse:
`other_regs` (32 costs 16%, 64 illegal), `softmax_regs=192`, `role_map=fa4`,
`first_load_order=4`, `s_stage=1`, `persistent` (16.5 ms, much worse).
`kv_stage` caps at 12 on shared memory; 10-11 are best.

## Backlog

0. **KV tile size.** FA4 gets 1.9% from `n_block=160`. Helion's flash emitter
   hardcodes a 128-wide KV tile: the MMA tiler is 128x128 ("the only legal
   one"), the TMEM column layout is built around it (S0 @ 0, S1 @ 128, O @
   256), `num_kv = (seq + 127) // 128`, and the exp2 packet schedules and
   P-store repetitions all count 128 columns. Making this a real dimension is
   the single largest remaining structural gap, and it is a multi-day change.
1. Promote the lowering choice itself to a `cute_flash_dense_softmax` knob
   (`standard`/`resident`/`packed_xu`) plus `cute_flash_prob_log2_shift`.
   Working patch saved at `artifacts/attn-dense32768/opt2_dense_softmax_knob.patch`;
   it needs the implied schedule fields canonicalized when the knob is
   non-standard so the structural-coverage design can witness the values.
   Measured on this shape: resident 13.39 ms, packed_xu 14.74 ms, standard
   15.99 ms -- so the value here is generality (unseeded KV sizes), not speed.
2. Persistent grid / CLC scheduler: 8192 CTAs over 148 SMs is 55.4 waves, so
   the tail is worth <1%. Low priority.
3. The shape is power-limited, not latency-limited: both kernels sit on the
   1400 W cap. Further wins need less energy per FLOP (fewer TMEM round trips
   or register-file accesses), not better scheduling. Two independent cold
   full-effort searches over ~3700 candidates found nothing better than the
   seed's arithmetic.
4. The two `test_cute_flash_length_invariance.py` causal failures are
   pre-existing on `origin/main` (verified at 2de2ad0f) and unrelated.


---

# Phase 2: all 8 shapes vs tuned FA4 (from 2026-09-05)

Baseline is now `fa4-tuned`: FA4 at the per-shape winning configs from
`helion_paper/data/attention_fa4_tuned_gb300.csv`. All eight recorded plans use
`tile_mn=[128, 160]`; dense 64K adds `FA_CLC=1`, dense 64K/128K
`disable_scheduler_metadata`, dense 256K `num_splits=2`, causal 512K
`num_splits=-1`. `seqlen_k_per_split` is beta26-only and is dropped on this
beta23 checkout (recorded in the result notes).

## Step 1 baseline, interleaved on one idle GPU 3, n=18 per impl

| shape | helion (seed) | fa4-tuned | ratio |
| --- | ---: | ---: | ---: |
| dense_32K | 1268.9 | 1367.5 | 0.928 |
| dense_64K | 1323.2 | 1371.8 | 0.965 |
| dense_128K | 1280.1 | 1362.4 | 0.940 |
| dense_256K | 1223.5 | 1365.1 | **0.896** |
| causal_64K | 1326.4 | 1347.7 | 0.984 |
| causal_128K | 1353.9 | 1367.1 | 0.990 |
| causal_256K | 1325.4 | 1372.4 | 0.966 |
| causal_512K | 1286.8 | 1372.2 | 0.938 |
| geomean | | | **0.950** |

Artifacts `artifacts/attn8/base8_*`. Worst = dense_256K.

## Observations

- Dense is uniformly weaker than causal. Helion's causal path already runs
  one-CTA; dense ran two-CTA until this session's change.
- Applying one-CTA to the dense seeds, one field on top of each shape's own
  seed, splits sharply: 32K +2.5%, 128K +2.7%, 64K -6%, 256K -22%. It is a
  genuine per-shape choice, which is what the autotuner is for -- and it is
  only reachable at all because of the resident-lowering change.
- Every tuned FA4 plan uses a 160-wide KV tile; Helion's emitter is hardwired
  to 128. FA4's own ablation puts that at 1.9%, so it is real but not the
  whole 5% deficit.
- Cost note: a cold full autotune on dense_256K runs at ~0.1 configs/s, so
  ~5 h for that shape alone; eight shapes is ~40 GPU-hours on one GPU.

## Phase 2 iteration log

- (base8) 8-shape baseline recorded, geomean 0.950, checker GOAL NOT MET 7/8.
- at_d256: cold full autotune on the worst shape ran at ~0.088 configs/s
  (~6.7 h projected) and was stopped in favour of breadth. A quick search on
  the same shape moved it 1223.5 -> 1242.5 (0.896 -> 0.910): no big win in the
  neighbourhood of the seed.
- **Root cause found.** Every tuned FA4 plan uses tile_mn=[128,160], and that
  one field carries the whole advantage. Same harness, same path, dense 32K,
  nothing else changed: FA4 at [128,128] = 1248.1, at [128,160] = 1367.5
  (+9.6%). Helion at its hardwired 128 = 1268.9, i.e. **Helion already beats
  FA4 at a matched tile width**. An earlier ablation put the tile at 1.9%; that
  was measured on a direct-op construction that also changed persistence and
  CTA count, and is superseded.
  Width sweep (FA4, dense 32K): 128 -> 1248, 144 -> 1332 FAIL, 160 -> 1368,
  176 -> 1141 FAIL, 192 -> 1022. 160 is the peak and the only wide width FA4
  validates. It is also the largest tile keeping 2 score + 2 output
  accumulators inside 512 TMEM columns (2*160 + 2*64 = 448); Helion's 128-wide
  layout uses only 384 of 512.
- step04 (commit 10208610) made the KV tile width a real dimension: MMA tilers,
  TMA/smem layouts, TMEM column offsets, P-tile width, K/V staging. Byte-
  identical at 128; at 160 on a divisible sequence (163840), interleaved n=18:
  989.9 -> 1027.6 TFLOP/s (+3.8%), correctness PASS.
- Not yet in the search: 160 does not divide a power-of-two sequence, so the
  eight benchmark shapes need a **masked tail KV tile** the dense path has
  never had. Gating search choices on divisibility would make the search
  surface length-dependent, which the backend forbids. The tail is the next
  step and is what unlocks this on the real shapes.


## Step 05: the masked tail unlocks the wide tile on real shapes

`160` does not divide a power-of-two sequence, so the eight benchmark shapes
needed a masked trailing KV tile. TMA zero-fills past the tensor extent and a
zero score is not a no-op (`exp2(0 - max)` joins the row sum), so the partial
tile's columns must be forced to -inf. The hardware TMEM row reduction folds
the max into the load and cannot see the mask, so that one tile discards the
hardware max and reduces in software -- the pair the causal diagonal already
uses. Descending KV order visits the partial tile first, so it peels off as a
one-iteration prefix.

Commits: `cea10ba3` masked tail, `fe855284` dense-256K resident seed,
`90d66f36` width in the search + sm_103 dense seeds at 160.

Interleaved against tuned FA4, one idle GPU 3:

| shape | before | after | fa4-tuned | ratio |
| --- | ---: | ---: | ---: | ---: |
| dense_32K | 1268.9 | 1338.8 | 1365.1 | 0.928 -> 0.981 |
| dense_64K | 1323.2 | 1350.5 | 1364.5 | 0.965 -> 0.990 |
| dense_128K | 1280.1 | 1361.3 | 1361.3 | 0.940 -> 1.000 |
| dense_256K | 1223.5 | 1331.4 | 1361.1 | 0.896 -> 0.978 |
| geomean (8 shapes) | | | | 0.950 -> 0.978 |

dense_256K needed a second change: it was the only seed on the packed f16x2
exp2 lowering, whose packet schedule is written against 128 columns, so the
wider tile did nothing for it (1223.5 -> 1226.0). Switching that seed to the
resident value graph, which every other dense seed already used, gave
1332.9 at width 160.

## Remaining

1. **Causal has no wide tile.** All four causal shapes (0.938-0.990) are still
   at 128. The blocker is the diagonal split proof, which assumes square
   128x128 tiles: with N != M the masked/unmasked boundary becomes
   `floor(m*128 / kv_n)` and the active count `ceil((m+1)*128 / kv_n)`. This is
   the largest remaining item and the worst shapes are here.
2. **Dense knobs are still tuned for a 128-wide tile.** e2e offsets, register
   splits and staging were all searched at 128. Now that the width is in the
   search surface, a cold full autotune can re-tune around 160.
3. The autotuner should choose the softmax lowering per shape rather than
   taking it from the policy; the 256K case was found by hand-editing the seed.
   Patch for the knob is at `artifacts/attn-dense32768/opt2_dense_softmax_knob.patch`.
