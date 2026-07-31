# KDA Prefill: Helion and SGLang Benchmarking Notes

This note records the Kimi Delta Attention (KDA) prefill implementation, the
baseline used for comparison, and the commands used on two GB200 GPUs. The
`sitecustomize.py` integration is an A/B harness, not a proposed SGLang API.

## Baseline and dispatch

For `moonshotai/Kimi-Linear-48B-A3B-Instruct`, SGLang's default linear-attention
backend is `triton`. The prefill/extend call chain is:

```text
KDAAttnBackend.forward_extend
  -> KDAKernelDispatcher.extend
  -> TritonKDAKernel.extend
  -> sglang.kernels.ops.attention.fla.kda.chunk_kda
```

The dispatcher and wrapper live in:

```text
python/sglang/srt/layers/attention/linear/kda_backend.py
python/sglang/srt/layers/attention/linear/kernels/kda_triton.py
```

The kernels used by the default path are in-tree under:

```text
python/sglang/kernels/ops/attention/fla/kda.py
python/sglang/kernels/ops/attention/fla/chunk_intra.py
python/sglang/kernels/ops/attention/fla/chunk_delta_h.py
```

This code was copied from Flash Linear Attention and adapted through vLLM, but
it is vendored into SGLang and does not import an external FLA package at
runtime.

SGLang also has two non-default KDA prefill implementations:

- `cutedsl`: in-tree Blackwell kernels, selected explicitly with
  `--linear-attn-prefill-backend cutedsl`. It requires SM100 and K=128, and the
  current wrapper cannot return intermediate states for the default
  `extra_buffer` radix-cache strategy.
- `flashkda`: an optional wrapper around the external MoonshotAI `flash_kda`
  CUTLASS package. Unsupported shapes, gates, speculative extend, and requests
  for intermediate states fall back to the Triton baseline.

FlashInfer is not a KDA prefill baseline. Its SGLang wrapper implements KDA
decode and verify only and explicitly leaves extend/prefill to Triton or CuTe
DSL.

## Production contract

The TP=2 shape for Kimi Linear 48B is:

```text
global heads:       32
local heads/GPU:    16
K:                  128
V:                  128
chunk size:         64
activation/output:  bfloat16
recurrent state:    float32
```

SGLang's normal Kimi prefill/extend path concatenates each request's new tokens
and always passes `query_start_loc` as `cu_seqlens`. The default hot-path
specialization is therefore **packed varlen + forward substitution + FP32
state**. Fixed-length input remains supported for direct callers and isolated
microbenchmarks. Newton-Schulz is opt-in and can be combined with either input
layout.

The Helion public entry point has the same parameter order and defaults as
SGLang's `chunk_kda`:

```python
chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    initial_state_indices=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    output_intermediate_states=False,
    **kwargs,
)
```

It preserves the corresponding numerical and mutation contract:

- fixed-length and packed variable-length input;
- raw gates with `A_log`, optional `dt_bias`, and optional safe-gate
  `lower_bound`, or already activated gates when `A_log` is absent;
- optional in-kernel Q/K L2 normalization;
- indexed FP32 or BF16 state pools in `[slot, H, V, K]` layout;
- in-place output through `v` and in-place final-state updates;
- optional intermediate states with the same fixed/packed layouts;
- partial chunks, FP16 or BF16 activations, K up to 256, and arbitrary V.

The implementation and focused tests are:

```text
examples/linear/kda_prefill.py
test/test_kda_prefill.py
```

## Implementation

The production path launches six Helion-generated kernels:

1. Q/K L2 normalization.
2. Gate activation and chunk-local cumulative sum.
3. Chunk-local QK and KKT matrices.
4. Fused triangular solve plus U, W, and gated-K recomputation.
5. Recurrent chunk-state update.
6. Output projection and in-place output store.

The important fusion is step 4. Keeping solve and recomputation separate was
slower for every production sequence length tested.

Each numerical path has one selected configuration for all observed sequence
lengths. Fixed and packed state propagation also use separate configurations
because they have materially different load patterns:

| Stage/path | Blocks | Loop order | Warps | Stages | Indexing |
|---|---|---|---:|---:|---|
| matrices, forward substitution | 32 | `[1, 2, 0]` | 1 | 2 | pointer |
| matrices, Newton-Schulz | 32 | `[2, 1, 0]` | 1 | 2 | pointer |
| solve/recompute, forward substitution | 64, 64 | `[0, 1]` | 1 | 3 | pointer |
| solve/recompute, Newton-Schulz | 64, 64 | `[0, 1]` | 2 | 3 | pointer |
| state, fixed length | 16 | default | 8 | 3 | pointer |
| state, packed varlen | 16 | `[1, 2, 0]` | 4 | 3 | pointer |
| output | 128 | `[1, 2, 0]` | 2 | 4 | pointer |

All kernels use `static_shapes=False`. H, K, V, inner layouts, dtypes,
fixed/varlen mode, and optional numerical modes remain specialized. Total
sequence length and every leading stride derived from it stay dynamic, matching
the Triton baseline's `do_not_specialize=["T"]` behavior. Generated launch
code takes runtime T, grid sizes, and leading strides while retaining constexpr
H/K/V.

## Tuning findings

Multi-shape autotuning and focused sweeps used one geometric-mean objective,
with production points at T=512 and T=8192 and validation at T=2048.

- The prologue now emits the cumulative gate, rounded Q/K operands, gated K,
  and each chunk's final decay in one pass. This removes repeated exponentials
  and avoids rereading the last gate in state propagation.
- Forward substitution distributes the four independent 16x16 diagonal
  inversions over the matrix CTAs. The solve consumes those pre-inverted
  blocks. This was bitwise identical for the matrix, W, and U intermediates and
  reduced full-pipeline latency by 13.5%, 12.3%, and 3.4% at
  T=512/2048/8192 relative to doing the same inversions in the solve CTA.
- Moving Newton-Schulz inversion into the matrix stage was neutral or slower,
  so that path keeps inversion in the solve CTA. Three BF16 MMA-style
  refinements match the CuTe DSL structure and outperform FP32 refinements
  while remaining within the recurrent-reference tolerance.
- Separate solve sweeps selected one warp for forward substitution and two for
  Newton-Schulz. Both use 64x64 registered blocks and three pipeline stages.
- Fixed state propagation benefits from eight warps: 11.320, 36.134, and
  162.226 us at T=512/2048/8192, versus 11.552, 37.091, and 172.232 us with four
  warps. Packed varlen instead prefers four warps and loop order `[1, 2, 0]`.
- Matrix tuning selected block 32, one warp, and two stages, with distinct loop
  orders for the two inverse paths.
- Tensor descriptors were emitted for the contiguous normalization loads, but
  that variant was 3-4x slower. Gathered matrix/output addresses did not lower
  to descriptors. Pointer indexing is retained.
- L2 grouping was useful only for the now-unused split W helper. Eviction hints,
  persistent PID, `maxnreg`, range unrolling, and range staging were neutral or
  slower on the production kernels. Four range stages regressed performance.
- Warp specialization reached the Triton operation but failed in the SM100
  compiler for the matrix kernel, so it is not enabled.
- A broad random fused search found configurations with minute-long compile
  outliers and a worse geometric mean. Those were rejected because compile
  behavior is part of the acceptance criteria.

The autotuning entry point is:

```text
benchmarks/kda_prefill_autotune.py
```

For example:

```bash
python benchmarks/kda_prefill_autotune.py \
  --kernel fused \
  --varlen \
  --sequence-lengths 512 8192 \
  --cache-tag kda-prefill-fused-gb200
```

## Microbenchmark results

Environment:

```text
Helion:  7ba4dc7070252d110a03d3f4bcb575d53f5e9699 + this worktree
SGLang:  d0b9689805232d8ab37789121cbc3b766b5c723e + benchmark changes
Torch:   2.11.0+cu130
Triton:  3.6.0
GPU:     NVIDIA GB200, SM100, 152 SMs
```

SGLang's existing CuTe DSL prefill benchmark was extended in place to include
Helion. With BF16, H=16, K=V=128, one fixed sequence, and CUDA-graph timing:

| T | Triton (ms) | CuTe DSL (ms) | Helion forward-substitution (ms) | Triton/Helion | Helion Newton-Schulz (ms) | Triton/Helion-NS |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.067 | 0.031 | 0.042 | 1.62x | 0.039 | 1.74x |
| 2048 | 0.176 | 0.099 | 0.107 | 1.64x | 0.103 | 1.68x |
| 8192 | 0.674 | 0.363 | 0.379 | 1.78x | 0.367 | 1.82x |

The default forward-substitution path retains the Triton baseline's inverse
operation order. Newton-Schulz is algebraically equivalent and passes the same
recurrent-reference tolerances, but changes floating-point ordering. It is
therefore opt-in through the ``newton_schulz=True`` keyword and is reported as a
separate result rather than as a replacement for the default baseline.

CuTe DSL remains faster on this fixed-shape GB200 microbenchmark. The default
benchmark command is:

```bash
cd /path/to/sglang
python benchmark/bench_linear_attention/bench_kda_prefill_cutedsl.py \
  --mode bench \
  --dtype bfloat16 \
  --num-heads 16 \
  --seq-lens 512 2048 8192 \
  --helion-root /path/to/helion-multi-autotune
```

Add ``--helion-newton-schulz`` to select and clearly label the experimental
Newton-Schulz specialization.

A paired packed-varlen benchmark using raw gates, internal Q/K normalization,
H=16, K=V=128, and ragged non-aligned lengths produced:

| Total T | Triton (ms) | Helion forward (ms) | Triton/Helion | Helion NS (ms) | Triton/Helion-NS |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.067957 | 0.052078 | 1.305x | 0.047767 | 1.423x |
| 2048 | 0.157140 | 0.126784 | 1.239x | 0.122787 | 1.280x |
| 8192 | 0.532932 | 0.438449 | 1.215x | 0.430456 | 1.238x |

### BF16 decode tuning

The decode kernel now has separate fixed configurations for FP32 and BF16
recurrent state. The public parameter order, return tuple, aliases, in-place
state/output mutations, padding behavior, and numerical operations are shared
by both paths. The FP32 configuration is unchanged.

One multi-shape search evaluated 1,752 configurations on B=1 and B=256 with an
equal-weight geometric mean of latency ratios. The selected BF16 configuration
uses a V tile of 16, one warp, four stages, flat PID, L2 grouping 16, and the
exact indexing and eviction fields recorded in
`examples/linear/kda_packed_decode.py`. It improved the raw kernel as follows:

| B | Prior config (us) | BF16 config (us) | Ratio |
|---:|---:|---:|---:|
| 1 | 11.360 | 8.288 | 0.730x |
| 256 | 72.224 | 51.680 | 0.716x |

The joint objective was `0.7225x`, or a `1.384x` raw-kernel speedup. The search
command was:

```bash
cd /home/eche/local/helion-multi-autotune
python -m examples.linear.kda_packed_decode \
  --sglang-root /home/eche/local/sglang \
  --mode correctness \
  --batch-sizes 1 \
  --tp-sizes 2 \
  --activation-dtype bfloat16 \
  --state-dtype bfloat16 \
  --multi-autotune \
  --tune-batch-sizes 1 256 \
  --tune-aggregation geomean
```

SGLang's existing FlashInfer decode benchmark was extended with an optional
paired Helion baseline. It validates both Helion variants against packed
Triton, alternates timing order for three rounds, and reports medians. Direct
timing includes Python validation, output allocation, and dispatch, which hide
most of the raw gain at small B. The tuned kernel remained `1.68-1.83x` faster
than FlashInfer and was 8% faster than the old Helion config at B=256:

```bash
cd /home/eche/local/sglang
python benchmark/bench_linear_attention/bench_kda_flashinfer_mtp.py \
  --mode bench \
  --task decode \
  --dtype bfloat16 \
  --batch-sizes 1 4 16 32 64 128 256 \
  --num-q-heads 16 \
  --num-v-heads 16 \
  --helion-root /home/eche/local/helion \
  --helion-baseline-root /home/eche/local/helion-multi-autotune
```

Generated Triton confirmed the selected block, PID, stage, L2, and eviction
choices. The requested tensor-descriptor indexing entries were downgraded to
pointer loads because these accesses are gathered through runtime state indices;
no descriptor instructions were emitted. The exact autotuned config is retained.

A follow-up inline Triton experiment isolated only the recurrent-state access.
The pointer and TMA specializations shared launch geometry and all arithmetic;
TMA used a `[1, 1, block_v, 128]` descriptor box with the state slot loaded from
`ssm_state_indices` on device. Descriptor load/store was bitwise identical to
the pointer variant, confirming that dynamic slot coordinates are supported.
It was not profitable:

| Block V | Warps | B=1 TMA/pointer | B=256 TMA/pointer | Geomean |
|---:|---:|---:|---:|---:|
| 16 | 1 | 1.796x | 1.433x | 1.605x |
| 32 | 1 | 1.856x | 1.052x | 1.398x |
| 32 | 2 | 2.016x | 1.273x | 1.602x |
| 64 | 8 | 1.522x | 1.255x | 1.382x |
| 128 | 8 | 1.675x | 1.638x | 1.656x |

Issuing the transfer before the independent gate exponentials improved overlap,
but the best matched result still regressed by 26.8% at B=1 and 9.5% at B=256.
The production kernel therefore retains coalesced pointer state access. The
reproducer is `benchmarks/kda_decode_tma_probe.py`.

The fresh TP=2 BF16 end-to-end comparison used default FlashInfer decode and
Triton prefill as the baseline. The Helion row selected Triton only as the
packed-decode integration carrier, then substituted the tuned Helion decode;
prefill remained Triton. Each server ran one 20-request warmup followed by three
100-request samples of the workload documented below.

| Configuration | Input tok/s samples | Median | Geomean | vs baseline |
|---|---:|---:|---:|---:|
| Default FlashInfer decode | 26,952 / 25,937 / 28,201 | 26,952 | 27,014 | baseline |
| Tuned Helion decode | 27,275 / 27,874 / 27,286 | 27,286 | 27,477 | +1.24% median, +1.71% geomean |

Every matched run generated identical text and token lengths, processed 247,525
input and 485 output tokens, and had zero request errors. Logs, JSONL outputs,
the paired microbenchmark, and generated code are under:

```text
/home/eche/results/kda-bf16-decode-tuning-20260724
/home/eche/results/kda-bf16-decode-tuned-e2e-20260724
```

## Dynamic-shape and startup behavior

After one T=512 compile, previously unseen T=2048 and T=8192 calls reused the
same six bound kernels in 0.00225 and 0.00209 seconds and emitted no additional
code. A TP=2 server emitted exactly 12 output-code events, six per worker, all
before readiness and none during the serving benchmarks.

Cold compilation remains a cost. A local first compile took about 13.35
seconds. Across comparable TP=2 launches, Helion reached readiness roughly
10-12 seconds later than Triton. This is substantially better than specializing
and compiling for every new T, but it is not compile-time parity with the
existing Triton path.

## End-to-end A/B result

The serving workload used two GB200 GPUs, TP=2, FP32 recurrent state, Triton
decode, 100 deterministic random prompts derived with the SGLang benchmarker,
target input length 4096 with range ratio 0.25, output length 8, concurrency 4,
seed 4242, and a cache flush before each measured run. Each run processed
247,525 input tokens and 485 output tokens with zero errors.

| Backend | Run 1 input tok/s | Run 2 input tok/s | Geomean |
|---|---:|---:|---:|
| Triton prefill | 26,575 | 27,597 | 27,081 |
| Helion prefill | 27,542 | 28,957 | 28,241 |

The clean-run geomean improvement is **4.28%**. One earlier Helion sample had a
non-reproducing 4.38-second scheduler stall and is preserved in the result
directory as `helion-prefill-n100.jsonl`; there was no Helion code generation
during the stall. It is excluded from the clean-run geomean rather than hidden.

Results and logs from this machine are under:

```text
/home/eche/results/kda-prefill-ab-20260723
```

### Path-isolated follow-up

After tuning the forward-substitution and Newton-Schulz paths separately, a
fresh TP=2 run kept FP32 state and Triton decode fixed. Each clean row is one
100-request sample after a 20-request warmup:

| Prefill path | Input tok/s | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | vs Triton |
|---|---:|---:|---:|---:|---:|
| Triton | 26,874.99 | 9.21 | 218.15 | 349.65 | baseline |
| Helion forward substitution | 26,487.65 | 9.34 | 252.91 | 367.74 | -1.44% |
| Helion Newton-Schulz | 25,926.43 | 9.55 | 237.83 | 402.78 | -3.53% |

The first forward and first Newton measured samples each hit the same isolated
1.7-second p99 TTFT scheduler stall. They are preserved as the non-`r2` files
and excluded from the table. With only one clean serving sample per path, the
small differences above are not robust enough to override the paired kernel
measurements or the earlier multi-run serving result.

All rows processed the same 247,525 input and 485 output tokens with no request
errors. Forward substitution generated byte-identical text to Triton.
Newton-Schulz preserved every output length but changed 90 of 100 generated
texts, which is consistent with its intentionally different floating-point
order. Both Helion servers emitted 12 generated-code events before readiness
and none during measurement. The complete artifacts are under:

```text
/home/eche/results/kda-prefill-structural-20260724
```

## End-to-end reproduction

Set paths and offline mode:

```bash
export HELION_ROOT=/path/to/helion-multi-autotune
export SGLANG_ROOT=/path/to/sglang
export MODEL=/path/to/Kimi-Linear-48B-A3B-Instruct/snapshot
export DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
export RESULTS=/path/to/results/kda-prefill-ab
export PYTHONPATH="${HELION_ROOT}/scripts/kda_sglang_ab:${HELION_ROOT}:${SGLANG_ROOT}/python"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${RESULTS}"
```

Launch the default Triton baseline:

```bash
cd "${SGLANG_ROOT}"
unset SGLANG_KDA_HELION_PREFILL
python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --trust-remote-code \
  --tp-size 2 \
  --linear-attn-backend triton \
  --linear-attn-decode-backend triton \
  --linear-attn-prefill-backend triton \
  --mamba-ssm-dtype float32 \
  --disable-custom-all-reduce \
  --port 30000 \
  2>&1 | tee "${RESULTS}/triton-server.log"
```

For the numerical-contract Helion path, use the same command with:

```bash
export SGLANG_KDA_HELION_PREFILL=1
unset HELION_KDA_PREFILL_NEWTON_SCHULZ
```

Set `HELION_KDA_PREFILL_NEWTON_SCHULZ=1` in addition to the substitution flag
to benchmark the separately tuned Newton-Schulz path.

With only this flag, the hook in `scripts/kda_sglang_ab/sitecustomize.py`
replaces only `kda_triton.chunk_kda`; decode and all other model kernels remain
unchanged. `SGLANG_KDA_HELION_DECODE=1` independently enables the packed-decode
substitution. Confirm the corresponding two `first ... call` markers, one per
TP worker, before using a Helion result.

After the server reports readiness, run one short workload to settle the full
serving path, then measure:

```bash
python -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --dataset-path "${DATASET}" \
  --model "${MODEL}" \
  --num-prompts 100 \
  --random-input-len 4096 \
  --random-output-len 8 \
  --random-range-ratio 0.25 \
  --max-concurrency 4 \
  --seed 4242 \
  --temperature 0 \
  --flush-cache \
  --warmup-requests 1 \
  --output-details \
  --disable-tqdm \
  --output-file "${RESULTS}/result.jsonl"
```

## Verification

Run the focused contract tests before benchmarking:

```bash
cd "${HELION_ROOT}"
pytest test/test_kda_prefill.py -x -vv -s
```

The final run passed all four tests in 36.01 seconds. A wider direct packed test
at T=73, H=2, K=256, and V=80 had maximum output error `3.052e-5`, maximum
intermediate-state error `1.707e-4`, finite outputs, and exact preservation of
untouched state-pool rows. A Newton-Schulz packed-varlen safe-gate check at
K=V=32 preserved the in-place and untouched-slot contracts with maximum output
and state errors of `1.831e-4` and `7.321e-4`.

SGLang's benchmark correctness mode also passed both Helion paths and CuTe DSL
against `fused_recurrent_kda` at T=128, 192, 256, 512, and 1024 with H=32 and
K=V=128. Both Helion paths had maximum output error `4.88e-4`; the largest
final-state errors were `4.39e-3` for forward substitution and `4.71e-3` for
Newton-Schulz.

## Default-backend FP32/BF16 matrix

On 2026-07-24, the TP=2 serving benchmark was repeated without the base or
prefill backend arguments. SGLang resolved the default backends as follows:

```text
FP32 state: decode=triton,     prefill=triton
BF16 state: decode=flashinfer, prefill=triton
```

Helion decode requires `TritonKDAKernel` as its integration point. Therefore,
the BF16 decode-only and combined rows explicitly selected
`--linear-attn-decode-backend triton`, then replaced its packed-decode callable
with Helion. Their performance comparison is still against the flag-free
FlashInfer baseline. BF16 prefill-only retained default FlashInfer decode.

Each configuration ran one 20-request warmup followed by three deterministic
100-request samples. Each sample processed 247,525 input and 485 output tokens
with no request errors. The primary statistic is the three-run median because
the serving benchmark occasionally contains multi-second scheduling stalls.

| State | Configuration | Input tok/s samples | Median | vs default | Geomean |
|---|---|---|---:|---:|---:|
| FP32 | default Triton/Triton | 24,129 / 26,784 / 25,279 | 25,279 | baseline | 25,374 |
| FP32 | Helion decode | 27,449 / 29,436 / 28,704 | 28,704 | +13.55% | 28,518 |
| FP32 | Helion prefill | 16,588 / 26,729 / 25,833 | 25,833 | +2.19% | 22,541 |
| FP32 | Helion decode + prefill | 29,113 / 29,148 / 29,636 | 29,148 | +15.31% | 29,298 |
| BF16 | default FlashInfer/Triton | 25,618 / 27,369 / 27,437 | 27,369 | baseline | 26,795 |
| BF16 | Helion decode | 25,594 / 27,795 / 27,175 | 27,175 | -0.71% | 26,838 |
| BF16 | Helion prefill | 27,169 / 28,700 / 27,496 | 27,496 | +0.46% | 27,781 |
| BF16 | Helion decode + prefill | 29,196 / 28,937 / 29,420 | 29,196 | +6.67% | 29,183 |

The first FP32 Helion-prefill sample reproduced the previously observed
one-time long-run stall. It emitted no new Helion code and subsequent samples
were normal. It is retained in both the raw samples and geomean; the median is
reported as the robust primary result. The small BF16 prefill-only difference
is within observed run-to-run noise, while the combined improvements are much
larger and stable across all three samples.

BF16 state also nearly doubled recurrent-state capacity at the same memory
budget: `max_mamba_cache_size` increased from 2,605 to 5,040 and
`max_running_requests` from 521 to 1,008. The complete logs and JSONL outputs
are saved under:

```text
/home/eche/results/kda-default-matrix-20260724
```

## ShareGPT FP32-state occupancy sweep

A flag-free default baseline and the three Helion substitutions were measured
with 128 fixed-seed ShareGPT prompts, 64 output tokens per prompt, FP32 state,
TP=2, and concurrency 1/4/16/32/64/128. No linear-attention backend arguments
were supplied; SGLang resolved the baseline to Triton decode and Triton prefill.
Each cache-flushed cell processed 31,609 input and 8,192 output tokens with 128
successful requests and no nonempty errors.

| Configuration | C=1 | C=4 | C=16 | C=32 | C=64 | C=128 | Ratio geomean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Triton baseline | 848.3 | 2,427.6 | 7,206.0 | 12,780.0 | 19,239.6 | 30,858.0 | 1.0000x |
| Helion decode | 876.7 | 2,535.2 | 7,339.8 | 12,992.4 | 19,886.1 | 30,915.2 | 1.0247x |
| Helion prefill | 948.6 | 2,543.5 | 7,562.7 | 13,067.4 | 19,592.1 | 31,292.5 | 1.0445x |
| Helion decode + prefill | 942.0 | 2,469.7 | 7,296.1 | 12,964.2 | 19,299.5 | 30,430.8 | 1.0233x |

Values are total input-plus-output tokens per second. The geomean speedups were
2.47% for decode, 4.45% for prefill, and 2.33% for both. Prefill-only reduced
mean-TTFT geomean by 9.95%; at concurrency 1 it improved total throughput by
11.82% and reduced mean TTFT from 156.1 to 117.7 ms. The combined row was not
additive in this one-run sweep, so its small differences at higher occupancy
should be treated as serving variance rather than a kernel interaction.

The first baseline C=128 sample and a later baseline C=64 repeat each hit an
isolated scheduler stall. Both are preserved; the table uses the clean original
C=64 and clean repeated C=128. Full logs, JSONL files, per-cell percentages, and
the outlier policy are under:

```text
/home/eche/results/kda-sharegpt-occupancy-fp32-20260724
```
