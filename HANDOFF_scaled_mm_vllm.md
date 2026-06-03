# Handoff: match cutlass on `scaled_mm` vllm_shapes (FP8 RowWise, B200)

## Goal
Write Helion (or, where Helion's Triton backend can't, raw Triton/Gluon) FP8 e4m3
RowWise `scaled_mm` kernels that **match or exceed `vllm._custom_ops.cutlass_scaled_mm`**
on the `vllm_shapes` in `examples/scaled_mm.py` (Qwen3-1.7B layer GEMMs: 4 `(K,N)` ×
8 `M` ∈ {1,2,4,8,16,32,64,512} = 32 shapes). Save each generated kernel + its perf
to `logs/`.

`out[m,n] = scale_a[m]·scale_b[n]·Σ_k a[m,k]·b[k,n]`, a/b fp8 e4m3, out bf16,
scale_a [M,1] f32, scale_b [1,N] f32.

## BEST KERNEL PER SHAPE (the answer to "which kernel for which shape")
Two kernel families, split by regime:
- **Skinny-M (M=1-64), memory/overhead-bound -> Helion `scaled_mm_into`** (split-K atomic
  into a pre-zeroed f32 buffer + ping-pong memset overlap), in `examples/scaled_mm.py`.
- **Compute-bound (M=512) -> gluon `scaled_mm_2cta_persistent`**
  (`benchmarks/gluon_2cta/scaled_mm_2cta_persistent.py`), this replaces the old `plain sk=1`
  Helion kernel which was the worst part of the prior result.

### M=512 (compute-bound) -- gluon persistent, CORRECTLY measured (tritonbench L2-flush, idle GPU)
| K | N (layer) | best config (bm128 bk128 num_warps=4) | ratio | (old Helion) |
|--:|--:|:--|--:|--:|
| 2048 | 4096 (qkv)      | `bn128 st8 acc2 epi32 sub2` | **1.16x**       | 1.39x |
| 2048 | 2048 (o_proj)   | `bn64 st10 acc2 epi32 sub2` | **1.07x**       | 1.17x |
| 6144 | 2048 (down_proj)| `bn64 st10 acc2 epi32 sub2` | **0.98x WIN**   | 1.38x |
| 2048 | 12288 (gate_up) | `bn128 st8 acc2 epi32 sub4` | **1.09x**       | 1.77x |

### M=1-64 (skinny) -- Helion `scaled_mm_into`, per-shape (from logs/results.tsv)
CAVEAT: these were measured the OLD way (NO L2 flush, possibly contended GPU) -> ratios
likely optimistic, NEED RE-VALIDATION under the tritonbench L2-flush method. Kernel CHOICE
(split-K `into` for skinny-M) is sound (split-K fills the machine = robust for memory-bound).
| M | qkv 2048x4096 | o_proj 2048x2048 | down_proj 6144x2048 | gate_up 2048x12288 |
|--:|:--|:--|:--|:--|
| 1  | `into sk=16 bs=[16,256,128]` | `into sk=16 bs=[16,256,128]` | `into sk=16 bs=[16,256,128]` | `into sk=8 bs=[16,256,128]` |
| 2  | `into sk=16 bs=[16,128,128]` | `into sk=16 bs=[16,128,128]` | `into sk=16 bs=[16,128,128]` | `into sk=8 bs=[16,128,128]` |
| 4  | `into sk=16 bs=[16,128,128]` | `into sk=16 bs=[16,128,128]` | `into sk=16 bs=[16,128,128]` | `into sk=8 bs=[16,128,128]` |
| 8  | `into sk=16 bs=[16,128,128]` | `into sk=16 bs=[16,128,128]` | `into sk=16 bs=[16,128,128]` | `into sk=4 bs=[16,128,128]` |
| 16 | `into sk=16 bs=[16,128,128]` | `into sk=16 bs=[16,128,128]` | `into sk=16 bs=[16,128,128]` | `into sk=4 bs=[16,128,128]` |
| 32 | `into sk=8 bs=[16,128,128]`  | `into sk=8 bs=[16,128,128]`  | `into sk=16 bs=[16,128,128]` | `into sk=4 bs=[32,128,128]` |
| 64 | `into sk=4 bs=[64,128,256]`  | `into sk=8 bs=[16,128,128]`  | `into sk=16 bs=[64,128,256]` | `plain sk=1 bs=[64,128,128]` |

bs = [BLOCK_M, BLOCK_N, BLOCK_K]; sk = split_k. OPEN: re-validate all M=1-64 under L2-flush
(`agent_space/bench_cfg.py` pattern). Best benchmark = tritonbench
`_do_bench_cudagraph_with_cache_clear` on an idle GPU.

## Current status: 23 WIN / 2 MATCH / 7 MISS (committed), avg ratio 0.92
Interleaved cudagraph timing (alternate vLLM/Helion each rep, median). The 7 MISS:
- **3 mid-M (memory-bound):** 32x2048x12288, 64x2048x4096, 64x2048x12288 — robustly
  **median ~1.08–1.18×** (overhead-bound small GEMMs; split-K atomic + ping-pong
  overhead keeps Helion just above cutlass). NOTE: earlier "0.92–0.94× wins" on these
  were favorable-GPU-clock samples, NOT robust — re-measure with ≥3 independent runs.
- **4 M=512 (compute-bound):** 512x{2048x4096, 2048x2048, 2048x12288, 6144x2048} —
  **~1.3–1.6×**. These need 2-SM clusters + tcgen05 TMEM (see "Remaining work").

## What's committed (git, branch `scaled-mm-fp8-vllm-match`, HEAD ~`c7502c56`)
- `examples/scaled_mm.py` — 3 Helion kernels:
  - `scaled_mm` (self-contained split-K; branches on tunable split_k: ==1 plain store,
    >1 atomic into zeroed out, rowwise scale folded per K-split partial).
  - `scaled_mm_into` (split-K atomic into a caller pre-zeroed buffer; deployed with
    double-buffered memset overlap "ping-pong" — wins the skinny-M shapes).
  - `scaled_mm_compute` (plain no-split-K GEMM for compute-bound regime).
- `benchmarks/scaled_mm_vllm_shapes.py` — per-shape search (subprocess-isolated;
  candidates() picks plain vs split-K + bm-matched; ranks by interleaved time).
- `logs/<MxKxN>.py` (32 kernels w/ perf headers) + `logs/summary.md` + `results.tsv`.
- **`helion/_compiler/matmul_utils.py`** (commit `03784244`) — KEY codegen fix: `hl.dot`
  no longer emits `input_precision='tf32'` for fp8 operands (it forced slow tf32
  emulation and blocked Blackwell tcgen05). fp8 dots now use native MMA. Verified
  (fp8_gemm/scaled_mm + 12 test_dot fp8 tests pass).

## THE TOOLCHAIN UNBLOCK (critical — do this first)
The env's triton is 3.6.0 (WS-pass buggy) / 3.7.0 (no 2-CTA cluster API). The 2-CTA
cluster matmul (needed for M=512) requires a triton `main` build. It IS installable:
```
pip install --target=/path/tnew --no-deps --pre \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ \
  triton           # -> triton-3.7.0+gitfeb6c04a (main-style gluon cluster API)
```
Use it via `PYTHONPATH=/path/tnew` (isolated; do NOT replace the env's triton — torch
2.13 is built against 3.6.0). It has `tma.async_load`, `mbarrier.allocate_mbarrier(two_ctas=)`,
`tcgen05_mma`, `TensorMemoryLayout`, `gl.warp_specialize`, `gl.num_ctas`.
(NB: package name `triton-nightly` on that index is a stale 2024 build — use name `triton`.)

## What already works (artifacts in `benchmarks/gluon_2cta/`)
Built incrementally under the Azure triton, all CORRECT:
- `g1.py` — minimal 2-CTA fp16 tcgen05 kernel (max_err 0.0).
- `g2.py` — fp8 K-loop 2-CTA, single tile (rel 0.003).
- `scaled_mm_2cta_fp8.py` (was g3) — **full-grid fp8 2-CTA cluster GEMM, rel ~0.003**,
  TMEM accumulator, multi-stage TMA, runs on all M=512 shapes.
  - GOTCHA: cross-CTA (`two_ctas=True`) barriers need
    `mbarrier.fence_init_release_cluster()` after the init loop (and inits in
    `gl.static_range`), else compile error "could not find an insertion point between
    cross-CTA mbarrier.init ops and tracked mbarrier uses".
  - GOTCHA: `NVMMASharedLayout.get_default_for` needs a gluon dtype (`gl.float8e4nv`,
    `gl.bfloat16`), not a torch dtype.
  - **PROBLEM: it's 2.1–2.9× SLOWER than cutlass** because it is NOT warp-specialized.
    A single warpgroup issues load→mma→epilogue serially; the empty-barrier wait
    between mma and the next load serializes the K-loop. (The single-CTA WS kernel
    `agent_space/scaled_mm_autows.py` is actually faster at 1.1–1.5× / matches 1/4.)

## Remaining work to MATCH cutlass on the 4 M=512 shapes
Warp-specialize the 2-CTA kernel. Reference is triton main's
`python/tutorials/gluon/14-multicta.py` (fp16). Re-fetch it with:
```
gh api repos/triton-lang/triton/contents/python/tutorials/gluon/14-multicta.py \
  | python3 -c "import json,sys,base64; print(base64.b64decode(json.load(sys.stdin)['content']).decode())" \
  > multicta_ref.py
```
(It contains the `Counter` aggregate, `MatmulPartitionArgs`, the 4 partition fns, and
the entry kernel.) Structure to adapt:
- Entry kernel `matmul_multicta_kernel` (lines ~973-1067): allocates `a_bufs/b_bufs`
  (STAGES), `acc_bufs` in TMEM (ACC_STAGES), barriers (`load_empty/load_ready`,
  `acc_empty/acc_ready`, + CLC bars), inits them, then `gl.warp_specialize([...])`.
  `mma_barrier_count = tcgen05_mma_barrier_count([a,b], multicast=True, two_ctas=...)`.
- 4 partitions (lines ~870-960): `matmul_load_partition` (TMA loads, `multicast=True`),
  `matmul_mma_partition` (`tcgen05_mma(..., multicast=True)` + `tcgen05_commit`),
  `matmul_epilogue_partition` (acc.load → subtile → cast → TMA store),
  `matmul_clc_partition` (cluster-launch-control persistent scheduler).
- SIMPLIFICATION to de-risk: DROP the CLC scheduler. Use a plain `grid=(num_clusters,)`
  (one cluster per output tile, like `scaled_mm_2cta_fp8.py`), and in each partition
  replace `while scheduler.has_work / scheduler.get_offsets()/.step()` with offsets
  derived once from `gl.program_id(0)`. So just 3 partitions: epilogue(default) +
  load + mma via `gl.warp_specialize([(load,(p,)),(mma,(p,))],[1,1],[...])` with the
  epilogue inline in the main warps, OR a 3-partition split.
- ADD THE ROWWISE SCALE in the epilogue: after `acc.load()` (gives [tile_m, tile_n]),
  multiply by `scale_a[off_m:off_m+tile_m, None] * scale_b[None, off_n:off_n+tile_n]`,
  cast to bf16, store. Load the scale slices via small TMA descriptors or `gl.load`
  with a layout matching the acc register layout (the tricky bit — see how the
  epilogue partition's `acc.load()` layout works; broadcast-multiply in registers).
- Pack shared objects into a dataclass like `MatmulPartitionArgs` (grep it in
  multicta_ref.py) and pass as the single `p` arg to each partition.

Validate correctness vs `torch._scaled_mm` (rel < 0.05), then benchmark vs vLLM
interleaved (see below). Target the 4 M=512 shapes first; if it also beats the
single-CTA approach on mid-M, use it there too.

## UPDATE 2026-06-02 — persistent kernel + CORRECTED benchmarking (READ THIS FIRST)
**Two big changes this session:**

### 1. New persistent kernel (`benchmarks/gluon_2cta/scaled_mm_2cta_persistent.py`)
Adapted triton main's full `14-multicta.py` reference (persistent CLC snake scheduler +
ACC double-buffering + epilogue subtiling) to fp8 e4m3 in / bf16 out / **rowwise scale
folded into the epilogue** (rel ~0.002). Correct on all 4 M=512 shapes. This SUPERSEDES
the grid-scheduled `scaled_mm_2cta_ws_WIP.py`. Per-shape best cfg (re-swept w/ correct
bench): bm128 bk128 st8 acc2 epi32 subtile4, bn128 (bn64 for o_proj 512x2048x2048),
num_warps=4. Scale loads hoisted (sa loaded once per tile, not per subtile).
Source of truth kept in `agent_space/scaled_mm_multicta.py`.

### 2. BENCHMARKING WAS WRONG — must flush L2 + use an idle GPU
The HANDOFF's "interleaved cudagraph median" method did **NOT flush L2**, and the shared
GPU0 had a Ray RL job (62GB, 100% util) contending. Both inflated/confused results.
Correct method = tritonbench `_do_bench_cudagraph_with_cache_clear` (cudagraph + L2 flush)
on an IDLE GPU. Bench scripts: `agent_space/bench_cfg.py` (one shape+cfg, own subprocess —
USE THIS; subprocess isolation avoids the bn=32 sticky-CUDA poison), `bench_one.py`,
`sweep_one.py`. ALWAYS pick a GPU at 0% util right before each run (the Ray job migrates
across GPUs 0-7, so even GPU 1-7 get contended mid-run → throttled vLLM ~2x → bogus ratios;
trust ABSOLUTE mc us over ratio when vLLM looks throttled).

### 3. EPILOGUE RESTRUCTURE = the breakthrough (got us to ~parity)
Original epilogue sliced TMEM per-subtile (`acc_buf.slice(s).load()`) which over-constrains
the small-tile load layout AND was slower. Rewrote to the **08-WS pattern**: load the FULL
tile `acc_bufs.index(...).load()` in its natural layout, apply the rowwise scale on the full
tile (sa/sb layouts derived from the valid full-tile load layout, sa via SliceLayout(1,..),
sb via SliceLayout(0,..)), then split in REGISTERS via `_split_n` (reshape/permute/split,
copied from 08) for the subtiled TMA stores. This dropped avg ratio ~1.3x -> ~1.05x.

**TRUE compute-bound ratios (cold L2, idle GPU, tritonbench, full-load epilogue):**
- down_proj 512x6144x2048: **0.975x (WIN)**  [bn64 st10 acc2 epi32 sub2]
- gate_up  512x2048x12288: **1.012x (≈match)** [bn128 st8 acc2 epi32 sub4]
- o_proj   512x2048x2048:  **~1.0-1.11x**     [bn64 st10 acc2 epi32 sub2]
- qkv      512x2048x4096:  **~1.10x** (worst) [bn128 st8 acc2 epi32 sub2]
Best per-shape cfg: bm128 bk128 num_warps=4 always; bn64+st10 for the 2 N=2048 shapes,
bn128+st8 for the wide-N shapes. The committed `scaled_mm_2cta_persistent.py` has the new
epilogue. **All 4 shapes now ~match cutlass (prior was 1.3-1.6x).** qkv is the stubborn one.

### qkv 512x2048x4096 deep-dive (2026-06-02 cont'd) — stuck at ~1.10-1.15x
Exhaustively attacked the one stubborn shape. NCU: 0.86 waves (64 tiles on 74 clusters),
latency-bound, 12.5% occ (register-limited 1 block/SM). tile_m locked at 256 (2-CTA MMA),
so only 2 M-tiles for M=512. Tried, all in `agent_space/`:
- **Config sweeps** (bn/bk/stages/acc/sub/num_warps, fewer AND more stages): best
  bn128 st8 acc2 sub2 = ~9.0-9.6us vs vLLM ~8.0-8.4us = ~1.10-1.15x. Plateau.
- **split-K** (`scaled_mm_splitk.py`, grid=num_tiles*SPLIT_K, f32 partials + reduction):
  MUCH worse (~2.7-4.8x). Reduction overhead (even fused `_reduce_scale_kernel`) on the
  16MB partials buffer dwarfs the tiny GEMM's saving. SPLIT_K>2 also hits "illegal
  instruction". Dead end for such a small GEMM.
- **grid-scheduled no-CLC** (`scaled_mm_grid.py`, 1 tile/cluster): WORSE (1.24x). CLC
  persistence is NOT pure overhead.
- **manual-persistent** (`scaled_mm_manual.py`, strided tile loop, sweep num_clusters to
  force multi-tile-per-cluster + ACC-overlap): no better than CLC (1.115x best at G=full);
  fewer clusters leaves SMs idle, hurting more than the epilogue-overlap helps.
- **cluster-4 retry** with the new full-load epilogue: gets PAST the "CGA permutation"
  assert now, but ((1,0),(0,1)) N-sharding halves per-CTA tile_n -> `_split_n requires >=2
  elements/thread`. cluster-4 + 2-CTA tcgen05 matmul is not expressible in this build.
- **cluster-4 NOW COMPILES** (update): with the full-load epilogue + subtile_factor=1
  (epilogue_size_n==block_size_n, so `_split_n` is a no-op), `cga_layout=((1,0),(0,1))`
  num_ctas=4 runs correctly (rel 0.0024)! BUT it's SLOWER (1.25-1.8x): this cga shards
  ONE output tile across 4 CTAs (uses 4 SMs/tile, 2 MMA-pairs over N-halves) -> more
  SMs per tile, NOT more tiles in flight. cutlass's cluster-4 instead has 4 CTAs compute
  DIFFERENT tiles sharing A/B multicast -> needs the LinearEncoding/SharedLinearEncoding
  cluster pattern (tutorial preamble), not output-sharding cga_layout. That's the real
  remaining work to match qkv, and it's a deep compiler/layout undertaking.
CONCLUSION: qkv = 1.16x floor for everything expressible in this gluon build. The other 3
compute-bound shapes match/win (down_proj 0.979x WIN, o_proj 1.072x, gate_up 1.090x).

### Why qkv/last ~10% is hard (the wall)
NCU: kernel is latency-bound, **register-limited to 1 block/SM** (255 reg/thread = a full
256-thread block uses all 64K SM regs → Block Limit Registers=1, occupancy 12.5%), with
~38% L1TEX-load + ~34% CTA-barrier stalls. cutlass uses **128x32, 2SM, cluster-4, 21-stage**.
- Deeper pipeline alone (small bk, st>8): no help / worse (K-loop+barrier overhead on tiny M).
- num_warps=8, acc_stages=3, bn32: no help (bn32 hits `ttng.tmem_load` layout error in the
  scale epilogue).
- **cluster-4 (the key cutlass feature) is BLOCKED**: gluon requires "CGA encoding must be a
  permutation matrix"; 2-base cga_layouts (((1,0),(0,1)) etc.) fail to compile / hard-abort.
  Would need the LinearEncoding/SharedLinearEncoding path (far more involved) or compiler work.
- KernelAgent (/home/shangdiy/KernelAgent): poor fit — bf16-oriented test harness force-casts
  fp8 inputs; output is standard Triton (ceiling ~1.3-1.6x per committed Helion). Not pursued.

### Toolchain note
`agent_space/tnew` (triton 3.7.0+gitfeb6c04a, cluster API) gets cleaned up between sessions.
Reinstall: `conda run -n pytorch-3.12 python -m pip install --target=agent_space/tnew --no-deps
--pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton`
(plain `pip` is blocked on this host; must use conda env's `python -m pip`).

## WS port progress (NEW — partially done)
`benchmarks/gluon_2cta/scaled_mm_2cta_ws_WIP.py` is a warp-specialized 2-CTA GEMM
(3 partitions: epi/load/mma via `gl.warp_specialize([(epi,..),(load,..),(mma,..)],
[1,1],[24,24])`, grid-scheduled, no CLC, no scale yet). It uses the `Counter`
aggregate (copied from `08-warp-specialization.py`, also saved as
`benchmarks/gluon_2cta/ws_ref_08.py`), an `Args` aggregate, cross-CTA barriers
(`load_ready`/`acc_ready` two_ctas, `load_empty`), `tcgen05_mma_barrier_count`,
and `fence_init_release_cluster()`.
STATUS: **WORKS** (rel ~0.003) and is TUNED. The runtime bug was a producer/consumer
mbarrier PHASE OFFSET: the load (producer) partition must start at phase 1
(`Counter.create(1, p.STAGES)`) while the mma (consumer) starts at phase 0 — matching
the reference. With STAGES=6, BK=128, CLUSTER_M=256, TILE_N=128, num_warps=4 it runs
the M=512 shapes at **1.08–1.38× of cutlass** (512x2048x6144 ~1.08×; was 2.1–2.9×
non-WS). NOTE: this is GEMM-only — the rowwise SCALE epilogue is not added yet (will
cost a little more).

REMAINING LEVERS to close the last ~10–38% and match cutlass:
1. PERSISTENCE + ACC double-buffering (biggest): currently grid-scheduled (one tile
   per cluster), so the epilogue store is EXPOSED (not overlapped with the next
   tile's MMA). Add a persistent tile loop (grid = ~num_SMs/2 clusters, stride over
   tiles) with ACC_STAGES=2 acc buffers + acc_empty/acc_ready barriers so the
   epilogue of tile i overlaps the MMA of tile i+1 (see reference's CLC version, or
   do a manual persistent loop without CLC).
2. EPILOGUE SUBTILING: split TILE_N into sub-tiles, store each while the next MMA
   runs (reference `matmul_epilogue_partition`, SUBTILE_STAGES).
3. ADD THE ROWWISE SCALE in the epilogue (see below).
Test each on the small oracle (M=256,N=128,K=256) then the M=512 shapes.

## Measurement methodology (IMPORTANT — avoid false results)
- vLLM time swings 2× with GPU clock/thermal (no clock-lock permission). ALWAYS
  interleave: alternate vLLM and candidate each rep, take median, ≥20 reps, inside
  amortized cudagraphs (64 calls/graph). See `benchmarks/scaled_mm_vllm_benchmark.py`
  (`capture`, `capture_pingpong`, `time_graph`).
- Compute-bound ratios are CLOCK-SENSITIVE: a throttled GPU compresses the ratio to
  <1.0 ("fake win"). Sanity-check absolute vLLM µs (512x2048x4096: throttled ~12µs vs
  nominal ~7µs). For borderline shapes, run ≥3 independent processes and report the
  median-of-medians (or worst), not a single lucky run.
- For the `into` (atomic) kernel, plain single-stream cudagraph capture into a reused
  buffer ≈ its ping-pong time (memset hidden) and is robust (avoids sticky
  ping-pong-capture CUDA errors). Use that for ranking.

## Environment gotchas
- Run python via: `conda run -n pytorch-3.12 --no-capture-output python ...`
- Set `TRITON_CACHE_DIR` and `TMPDIR` to dirs on the REAL fs (e.g. under
  `/home/shangdiy/helion/agent_space/`) — the default /tmp hits a per-session quota
  glitch (phantom ENOSPC) that intermittently kills commands AND slows Triton
  compiles ~10×. If commands start failing with "temp filesystem ... is full",
  `find /tmp/claude-shangdiy -name '*.output' -delete` clears it.
- Write command output to files in the working dir and read them (the harness stdout
  capture itself glitches under the quota issue).
- A bad split-K atomic config (high split_k at small K) raises a STICKY CUDA capture
  error that poisons all later captures in the process → run each shape in its own
  subprocess; measure vLLM first; break the search on the first CUDA error.

## Quick repro of the working 2-CTA kernel
```
cd /home/shangdiy/helion
pip install --target=agent_space/tnew --no-deps --pre --index-url \
  https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton
PYTHONPATH=agent_space/tnew TRITON_CACHE_DIR=agent_space/tc TMPDIR=agent_space/tt \
  conda run -n pytorch-3.12 --no-capture-output python benchmarks/gluon_2cta/scaled_mm_2cta_fp8.py
# prints per-shape: vLLM us, 2cta_gemm us, ratio (~2-2.9x, slow), rel (~0.003, correct)
```

## Bottom line for the next agent
The hard blocker (no cluster toolchain) is GONE. A correct fp8 2-CTA cluster GEMM
exists. The single remaining task is to **warp-specialize it** (adapt the 3 core
partitions from `multicta_ref.py`, drop CLC for grid scheduling, add the rowwise-scale
epilogue) to reach cutlass throughput on the 4 M=512 shapes. Everything needed
(toolchain, working primitives, correct base kernel, reference, methodology) is in
place and documented above.
