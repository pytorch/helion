# attention-gym variants hillclimb (run: attngym-2026-09-01)

Goal: 16 NEW attention variants from meta-pytorch/attention-gym (not covered by the
existing dense/causal/biased benchmarks), each reaching parity (>=0.99 ratio) with the
best baseline under HELION_BACKEND=cute, geomean >= 1.0.

Helion implementation vehicle: `examples/flex_attention.py::helion_flex_attention`
(score_mod + BlockMask block sparsity + GQA), benchmarked output-only.

## Selected variants (16)

Shapes follow attention-gym's canonical benchmark (B16 H16 S8192 D64 fp16) with
diversity in dtype/head-dim/GQA where noted. "gather" = mask/score mod reads a
closure tensor.

| # | name | kind | definition | shape |
|---|------|------|-----------|-------|
| 1 | alibi | score mod (h-dep) | causal + exp2(-(h+1)*8/H)*(kv-q) | B16 H16 S8192 D64 fp16 |
| 2 | softcap | score mod (tanh) | causal + 30*tanh(s/30) | B16 H16 S8192 D64 fp16 |
| 3 | sandwich | score mod (1D table gather + head scale) | causal + rel_bias[q-kv+off]*head_scale[h] | B8 H16 S8192 D64 fp16 |
| 4 | sigmoid-act | score mod (log/sigmoid), dense | log(sigmoid(s)+1) | B8 H16 S4096 D128 bf16 |
| 5 | sliding-window | mask | causal AND q-kv<=1024 | B16 H16 S8192 D64 fp16 |
| 6 | dilated-sw | mask (no full blocks) | abs(d)<=512 AND d%2==0 | B8 H16 S8192 D64 fp16 |
| 7 | prefix-lm | mask | causal OR kv<1024 | B16 H16 S8192 D64 fp16 |
| 8 | global-sw | mask + gather | abs(d)<=512 OR is_global[q] OR is_global[kv] | B8 H16 S8192 D64 fp16 |
| 9 | document | mask + gather (jagged) | same-doc AND causal, 12 docs in 32768 | B1 H16 S32768 D64 fp16 |
| 10 | natten2d | mask (div/mod 2D) | 128x128 canvas, 13x13 kernel | B4 H16 S16384 D64 fp16 |
| 11 | sta2d | mask (block-aligned tiles) | canvas 128x128, tile 16x16, kernel 48x48 | B4 H16 S16384 D64 fp16 |
| 12 | block-diffusion | mask (block diag+causal) | S=4096 blk128, seq 8192 | B4 H16 S8192 D64 fp16 |
| 13 | gqa-causal | structure | causal, Hq32 Hkv8 | B4 H32/8 S8192 D128 bf16 |
| 14 | gemma2 | combo (softcap+SWA+GQA) | 50*tanh(s/50), causal AND d<=1024, Hq16 Hkv8 | B4 H16/8 S8192 D128 bf16 |
| 15 | shared-prefix | mask + gather | shared system prompt across docs | B1 H16 S16384 D64 fp16 |
| 16 | flamingo-xattn | rectangular M!=N + gather | text->image cross attn, M8192 N4096 | B4 H16 D64 fp16 |

Diversity axes covered: transcendental score mods (2,4,14), gathers in mods/masks
(3,8,9,15,16), div/mod index math (10,11,12), no-full-block sparsity (6), GQA (13,14),
bf16+D128 (4,13,14), rectangular M!=N (16), jagged/data-dependent (9,15).

## Baselines

- flex-triton: torch.compile(flex_attention) default mode (attention-gym methodology)
- flex-triton-ma: max-autotune-no-cudagraphs (stronger; use best of the two)
- flex-cute: FLASH (CuTeDSL) flex backend where supported
- sdpa / fa4: only where semantics representable (13: GQA causal; 5/14 windows if FA4 supports)
- helion-triton: same helion kernel, triton backend (sanity reference, not the bar)

Best baseline per variant = max throughput among the above (same FLOP accounting:
4*D*unmasked_pairs, identical for all impls of a variant).

## Architecture decision

Vehicle = per-variant Helion kernels in the CANONICAL examples/attention.py
envelope (collapsed [B*H,S,D], single static KV loop, where-masks + inline
score mods), NOT the flex BlockMask kernel: the cute backend's FA4 flash path
(cute_flash.py) pattern-matches exactly this envelope via AttentionScorePlan
(already supports causal/tensor-bias/alibi/sliding-window/prefix-lm/document/
softcap kinds; sparse masks route to ws_overlap). Data-dependent BlockMask
loops are unsupported there. Generalization work = new score-plan kinds +
GQA kv indexing + M!=N + block-skipping for arithmetic/gather masks.

Frontend constraints found empirically (kernels in benchmarks/cute/attngym_kernels.py):
- kernel fns cannot be closures; stage callables passed as kernel args are
  inlined at trace time (flex-example pattern).
- host-tensor gathers inside stage callables must use hl.load (subscript
  gathers only get rewritten in the kernel's own AST).
- tile_b.index[:, None, None] built in kernel source, passed to stages.

## Status log

- 2026-09-01: run started. Found: examples/flex_attention.py exists (score_mod +
  BlockMask + GQA). Under HELION_BACKEND=cute effort=none default config, BOTH
  examples/attention.py and examples/flex_attention.py produce WRONG results
  (~0.5% mismatch, naive SIMT path suspected). test_cute_flash_schedule passes (61).
  Old goal.json (matmul, complete) archived to artifacts/matmul-2026-08-31/.
- 2026-09-01: found+fixed shared codegen bug: allocate_reduction_dimension
  compared rdim.size == size with a GUARDING SymInt eq; mixed unbacked/int pair
  (tile_m.index[None,:,None] slice extent vs head_dim rdim) specialized the
  tile_m block symbol to head_dim (u1:=64), corrupting downstream shapes.
  Symptom: attention kernels failed to compile (or were silently WRONG on cute)
  whenever block_m != head_dim. Fix: known_equal (non-guarding) whenever either
  side is symbolic (helion/_compiler/compile_environment.py). This most likely
  also explains the pre-existing cute-backend attention numerics failures at
  default config. NOTE: shared-compiler change, affects all backends.
- 2026-09-01: all 16 variant kernels PASS correctness on triton AND cute at
  block_sizes=[1,128,128]. Flash path fires for: alibi, softcap, sliding-window,
  prefix-lm. Generic (slow) path: sandwich, sigmoid-act, dilated-sw, global-sw,
  document(!), natten2d, sta2d, block-diffusion, gqa-causal, gemma2,
  shared-prefix, flamingo. Harness: compare_attngym_backends.py.

## Iteration log

(append: worst variant | idea | cold-autotune result | checker verdict)
