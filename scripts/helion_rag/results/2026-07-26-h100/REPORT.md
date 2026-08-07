# Final Report: LFBO, LLM, Hybrid, and Contextual RAG-LLM Kernel Autotuning

- **Campaign:** `head_to_head_4arm_shapes`
- **Finalized:** 2026-07-26
- **Updated:** 2026-07-27 (added per-approach regret against the LFBO baseline)
- **Scope:** 33 GPU-kernel workloads, four search approaches, five paired repetitions, 660 completed runs

## Poster-ready abstract

GPU-kernel autotuning must balance the quality of the selected kernel configuration against the time and cost required to find it. We compare four approaches—learning-from-best-observations (LFBO), direct LLM-guided search, LLM-seeded LFBO (Hybrid), and contextual retrieval-augmented LLM search (RAG-LLM)—in a paired campaign covering 33 kernel shapes and five seeds per approach. Outcomes include selected latency, end-to-end readiness time, evaluated configurations, correctness, regret to the best observed campaign result, and provider-token usage. Across kernels, Hybrid produced the lowest geometric-mean selected latency (0.0743 ms) and improved performance by 17.1% relative to LFBO (ratio 0.829, 95% bootstrap CI 0.748–0.910, Holm-adjusted *p*=0.012), but increased readiness time by 54.3%. LLM achieved the lowest readiness time (62.8 s geometric mean), 42.1% below LFBO, while improving selected latency by 10.5%; the performance effect did not remain significant after family-wise correction. RAG-LLM improved selected latency by 1.9% relative to the context-free LLM (ratio 0.981, CI 0.964–0.994, adjusted *p*=0.034), at 14.3% greater readiness time and 17.3% more provider tokens. All four approaches completed every run and achieved 164/165 correct results. The results expose a Pareto frontier: Hybrid favors final kernel quality, LLM favors tuning speed, and RAG-LLM offers a modest quality gain over direct LLM search at additional time and token cost.

## Problem statement and motivation

Autotuners search a discrete kernel-configuration space in which compilation failures, invalid candidates, and expensive benchmarks make exhaustive search impractical. A useful tuner must therefore answer two questions jointly:

1. How fast is the kernel configuration it ultimately selects?
2. How quickly and reliably does it reach that configuration?

LFBO provides a strong non-LLM search baseline. LLM guidance may inject code and hardware priors early in the search, contextual retrieval may make those priors workload-specific, and a Hybrid policy may combine LLM initialization with the broader exploration of LFBO. The experiment tests whether those additions improve the performance–time trade-off under one controlled, paired campaign.

## Hypotheses

| Hypothesis | Final assessment |
|---|---|
| LLM guidance reduces time-to-ready relative to LFBO. | **Supported.** LLM was 42.1% faster and RAG-LLM was 33.5% faster than LFBO in geometric-mean readiness time. |
| LLM-seeded LFBO improves final selected performance, but costs more search time. | **Supported.** Hybrid improved performance by 17.1% versus LFBO and was the only LFBO performance contrast significant after Holm correction; readiness increased by 54.3%. |
| Contextual retrieval improves direct LLM search. | **Supported for selected performance, not cost.** RAG-LLM was 1.9% faster than LLM after correction, but readiness was 14.3% longer and provider-token use was 17.3% higher. |
| The four approaches retain comparable reliability. | **Supported in this campaign.** Each arm produced 164 correct results from 165 completed runs and covered all 33 workloads with at least 3/5 correct repetitions. |

## Compared approaches

| Approach | Search policy | LLM request | Retrieval | Typical attempted configurations |
|---|---|---:|---:|---:|
| LFBO | `LFBOTreeSearch` | 0 | No | 80 |
| LLM | `LLMGuidedSearch` | 1 | No | 26 |
| Hybrid | `LLMSeededLFBOTreeSearch` | 1 | No | 80 |
| RAG-LLM | `LLMGuidedSearch` | 1 | Yes | 26 |

## Methodology and experimental setup

- **Paired design:** every workload–repetition block used the same seed across all four approaches. Five seeds (1000–1004) yielded 165 scheduled runs per arm.
- **Workload coverage:** 33 shapes spanning MatMul, Split-K MatMul, attention, FP8 attention, grouped GEMM, SwiGLU, softmax, GDN, RMSNorm, RoPE, and Mamba-2 scan.
- **Search controls:** candidate-attempt cap 80, early trajectory stopping disabled, cache reads and writes disabled, and a 1,800 s per-run timeout.
- **LLM controls:** Vertex provider, `claude-opus-4-8`, one round, 15 proposed configurations, 10 initial random configurations, and `best_of_k=1`.
- **Primary outcomes:** stabilized selected latency in milliseconds and readiness wall time in seconds; lower is better.
- **Configuration count:** the detailed table reports median `evaluation_count` across correct repetitions. Attempted counts are summarized separately because compilation/validation causes evaluations and attempts to differ.
- **Regret (two baselines):** *Regret vs oracle* = `(arm median latency / bounded campaign oracle − 1) × 100%`, where the bounded oracle is the fastest correct stabilized latency observed across all arms and repetitions for that kernel (an observed campaign reference, not a true exhaustive hardware oracle). *Regret vs LFBO* = `(arm median latency / LFBO median latency − 1) × 100%`, i.e. how much each approach beats (negative) or trails (positive) the default LFBO autotuner, with LFBO at 0% by definition. Both are per-kernel and aggregated as geometric means across kernels; the per-kernel values are in `analysis/per_kernel_arm.csv` (`regret_pct`, `regret_vs_lfbo_pct`).
- **Aggregation:** per-kernel medians are combined with geometric means. Pairwise inference uses the median matched-repetition ratio for each kernel (33 statistical units), a 200,000-resample percentile kernel bootstrap for 95% CIs, a two-sided Wilcoxon signed-rank test on log ratios, and Holm correction across all 12 performance/readiness contrasts.
- **Validity rule:** only `status="completed"` and `correct=true` records contribute performance, readiness, configuration, regret, and paired-ratio statistics.

The campaign applies identical workloads, seeds, scheduling controls, and maximum attempt limits. Actual search effort is policy-dependent—LLM and RAG-LLM normally stop at 26 attempts, while LFBO and Hybrid normally use 80—so selected quality and wall time must be interpreted jointly rather than as a fixed-evaluation-budget comparison.

## Headline aggregate results

| Arm | Completed | Correct | Workloads ≥3/5 correct | Performance GM (ms) ↓ | Readiness GM (s) ↓ | Regret vs oracle GM ↓ | Regret vs LFBO GM ↓ | Median attempts | Total elapsed (h) | Provider tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LFBO | 165/165 | 164/165 | 33/33 | 0.090913 | 108.93 | 43.20% | 0.0% | 80 | 6.14 | 0 |
| LLM | 165/165 | 164/165 | 33/33 | 0.082492 | **62.84** | 29.94% | −9.3% | 26 | **3.38** | 746,156 |
| Hybrid | 165/165 | 164/165 | 33/33 | **0.074310** | 175.09 | **17.05%** | **−18.3%** | 80 | 9.21 | 746,596 |
| RAG-LLM | 165/165 | 164/165 | 33/33 | 0.080849 | 71.82 | 27.35% | −11.1% | 26 | 3.82 | 875,243 |

The best median selected performance was achieved on 8 kernels by LFBO, 3 by LLM, 10 by Hybrid, and 11 by RAG-LLM, with one exact LLM/RAG-LLM tie. Fractional tie accounting gives 8.0, 3.5, 10.0, and 11.5 wins, respectively. LLM had the lowest median readiness on 32 kernels; RAG-LLM won the remaining kernel.

### Regret by approach

Regret is reported against two baselines (geometric mean across the 33 kernels; lower is better; negative = faster than LFBO):

| Arm | Regret vs campaign oracle | Regret vs LFBO baseline |
|---|---:|---:|
| LFBO | 43.20% | 0.0% (baseline) |
| LLM | 29.94% | −9.3% |
| Hybrid | **17.05%** | **−18.3%** |
| RAG-LLM | 27.35% | −11.1% |

The LFBO-baseline view answers "how much does each approach beat the default LFBO autotuner." Hybrid leads on both baselines; all three LLM-assisted arms improve on LFBO overall, but the two large Split-K MatMul shapes are strong positive-regret outliers for the retrieval-free LLM and RAG-LLM (LFBO/Hybrid recover there). Per-kernel regret against LFBO (`(arm/LFBO − 1) × 100%`) follows.

| Kernel type | Kernel identifier | LFBO | LLM | Hybrid | RAG-LLM |
|---|---|---:|---:|---:|---:|
| MatMul | `matmul-1024x1024x1024` | +0.0% | -48.8% | -48.8% | -49.3% |
| MatMul | `matmul-4096x4096x4096` | +0.0% | -59.8% | -59.7% | -60.4% |
| MatMul | `matmul-8192x8192x8192` | +0.0% | -55.4% | -55.8% | -55.6% |
| Split-K MatMul | `matmul_split_k-64x1024x64` | +0.0% | +10.4% | -3.9% | +12.7% |
| Split-K MatMul | `matmul_split_k-64x16384x64` | +0.0% | +432.7% | +0.4% | +433.4% |
| Split-K MatMul | `matmul_split_k-64x65536x64` | +0.0% | +159.9% | +9.1% | +159.2% |
| Attention | `attention-2x8x512x64` | +0.0% | -37.5% | -37.5% | -38.0% |
| Attention | `attention-2x8x4096x64` | +0.0% | -41.9% | -48.5% | -48.5% |
| Attention | `attention-2x8x8192x64` | +0.0% | -18.9% | -19.9% | -20.1% |
| FP8 Attention | `fp8_attention-2x4x512x64` | +0.0% | +0.0% | +0.0% | +0.0% |
| FP8 Attention | `fp8_attention-2x4x2048x64` | +0.0% | -9.5% | -9.6% | -9.8% |
| FP8 Attention | `fp8_attention-2x4x8192x64` | +0.0% | +9.9% | -5.8% | +10.0% |
| Grouped GEMM | `grouped_gemm-g2m1024` | +0.0% | -44.3% | -45.4% | -44.9% |
| Grouped GEMM | `grouped_gemm-g4m512` | +0.0% | -54.6% | -53.4% | -54.9% |
| Grouped GEMM | `grouped_gemm-g8m512` | +0.0% | -44.2% | -43.6% | -46.3% |
| SwiGLU | `swiglu-2048x2048` | +0.0% | +21.7% | +19.7% | +20.8% |
| SwiGLU | `swiglu-4096x4096` | +0.0% | +2.4% | +2.6% | +2.6% |
| SwiGLU | `swiglu-8192x8192` | +0.0% | -0.2% | -0.3% | -0.2% |
| Softmax | `softmax-4096x1024` | +0.0% | -0.3% | +0.7% | -0.3% |
| Softmax | `softmax-4096x8192` | +0.0% | +0.1% | -0.9% | -0.7% |
| Softmax | `softmax-4096x32768` | +0.0% | -0.0% | +0.0% | -0.1% |
| GDN forward | `gdn_fwd_h-b1h4s2048ds128` | +0.0% | -9.2% | -16.0% | -14.6% |
| GDN forward | `gdn_fwd_h-b1h4s4096ds64` | +0.0% | +8.1% | -4.6% | -0.4% |
| GDN forward | `gdn_fwd_h-b1h4s8192ds128` | +0.0% | +0.6% | -21.0% | -18.1% |
| RMSNorm | `rms_norm-4096x1024` | +0.0% | -13.4% | -13.0% | -13.0% |
| RMSNorm | `rms_norm-4096x8192` | +0.0% | -1.4% | -1.2% | -1.4% |
| RMSNorm | `rms_norm-4096x32768` | +0.0% | -1.2% | -1.2% | -1.3% |
| RoPE | `rope-1x4x2x512x128` | +0.0% | +11.9% | +8.0% | +14.6% |
| RoPE | `rope-1x4x2x2048x128` | +0.0% | +9.4% | +8.7% | +9.7% |
| RoPE | `rope-1x4x2x8192x128` | +0.0% | +6.7% | +6.2% | +7.1% |
| Mamba-2 scan | `mamba2_chunk_scan-b1h4s2048ds128` | +0.0% | -18.1% | -16.6% | -17.3% |
| Mamba-2 scan | `mamba2_chunk_scan-b1h4s4096ds256` | +0.0% | -20.1% | -20.6% | -23.9% |
| Mamba-2 scan | `mamba2_chunk_scan-b1h4s8192ds256` | +0.0% | -21.1% | -21.6% | -27.5% |

### Contrasts against LFBO

| Outcome | Arm/LFBO ratio GM | Relative change | 95% bootstrap CI | W/T/L | Holm-adjusted *p* | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Performance | LLM: 0.895 | −10.5% | 0.772–1.057 | 19/1/13 | 0.130 | Directionally better; not significant after correction |
| Performance | Hybrid: **0.829** | **−17.1%** | 0.748–0.910 | 22/1/10 | **0.012** | Significant improvement |
| Performance | RAG-LLM: 0.887 | −11.3% | 0.764–1.049 | 24/0/9 | 0.066 | Directionally better; misses corrected threshold |
| Readiness | LLM: **0.579** | **−42.1%** | 0.527–0.635 | 33/0/0 | 2.79×10⁻⁹ | Significant reduction |
| Readiness | Hybrid: 1.543 | +54.3% | 1.505–1.584 | 0/0/33 | 2.79×10⁻⁹ | Significantly slower |
| Readiness | RAG-LLM: 0.665 | −33.5% | 0.600–0.738 | 30/0/3 | 2.23×10⁻⁷ | Significant reduction |

### Direct LLM-family contrasts

| Contrast | Performance ratio GM | Performance adjusted *p* | Readiness ratio GM | Readiness adjusted *p* | Main implication |
|---|---:|---:|---:|---:|---|
| Hybrid / LLM | 0.915 (−8.5%) | 0.027 | 2.724 (+172.4%) | 2.79×10⁻⁹ | Better kernels at substantially greater search time |
| RAG-LLM / LLM | 0.981 (−1.9%) | 0.034 | 1.143 (+14.3%) | 3.73×10⁻⁹ | Small quality gain with measurable time/token overhead |
| RAG-LLM / Hybrid | 1.073 (+7.3%) | 0.819 | 0.420 (−58.0%) | 2.79×10⁻⁹ | Similar final quality statistically; RAG is much faster |

## Detailed analysis and observations

### 1. Suite-wide statistical interpretation

The aggregate results separate final kernel quality from tuning readiness. Hybrid achieved the lowest geometric-mean selected latency, improving on LFBO by **17.1%**. Its 95% bootstrap interval (0.748–0.910) lies entirely below parity, and the result remains significant after correction across the 12-test family (Holm-adjusted *p*=0.012). Hybrid is therefore the clearest evidence that LLM information can improve a conventional optimizer: using the LLM only as a seed, then continuing LFBO exploration, produced a material and statistically supported quality gain.

The LLM and RAG-LLM arms also improved selected latency relative to LFBO by 10.5% and 11.3%, respectively, but their LFBO contrasts do not survive the full Holm correction. The LLM interval crosses 1.0 and its adjusted *p*=0.130. RAG-LLM's raw direction is strong—24 kernel wins and only 9 losses against LFBO—but its adjusted *p*=0.066 narrowly misses the conventional 0.05 threshold. These results should be reported as directional suite-wide improvements, not definitive LFBO superiority claims.

The retrieval ablation is more precise because RAG-LLM and LLM share the same 26-attempt search structure. RAG-LLM improved selected latency by **1.9%** relative to LLM (ratio 0.981, CI 0.964–0.994, adjusted *p*=0.034), winning 21, tying 2, and losing 10 kernels. This is statistically supported, but small in magnitude. Hybrid was 8.5% faster than LLM (adjusted *p*=0.027), whereas RAG-LLM and Hybrid were statistically indistinguishable in final quality (ratio 1.073, CI 0.994–1.215, adjusted *p*=0.819). Thus the data support retrieval as a marginal improvement to direct LLM search, while extended LFBO exploration remains the larger quality lever.

### 2. Quality–readiness Pareto frontier

No approach simultaneously minimizes selected latency and tuning time. LLM had the lowest geometric-mean readiness (62.84 s), followed by RAG-LLM (71.82 s), LFBO (108.93 s), and Hybrid (175.09 s). Relative to LFBO, LLM and RAG-LLM reduced readiness by 42.1% and 33.5%, both with strongly significant corrected tests. Hybrid moved in the opposite direction: its readiness was 54.3% greater than LFBO and 172.4% greater than LLM.

A per-kernel Pareto analysis using median selected latency and median readiness reinforces this trade-off. LLM is nondominated on **32/33** kernels, RAG-LLM on **22/33**, Hybrid on **10/33**, and LFBO on **10/33**. This does not mean LLM produces the fastest kernel most often—it does not—but its low tuning time makes it difficult for another arm to beat it on both axes. Hybrid survives on the frontier where its additional search produces sufficiently better kernels; LFBO remains relevant on families where LLM guidance selects poor regions of the configuration space.

For production systems, the correct policy depends on reuse. In the 21 kernels where Hybrid is faster than LLM but takes longer to tune, the median simple break-even point is approximately **76.5 million kernel invocations**: additional tuning seconds divided by per-call latency savings. For RAG-LLM versus LLM, the corresponding median across 21 quality-improving kernels is approximately **10.1 million invocations**. These are illustrative amortization estimates assuming stable latency and one tuned kernel invocation per use; fused graphs, batching, and repeated launches can change the economics substantially.

### 3. Kernel-family heterogeneity

The suite-wide mean hides strong operator dependence. The table below reports the **actual family-level geometric means** of selected performance in milliseconds and readiness wall time in seconds. Each cell is `performance (ms) / wall time (s)`; lower is better. These family summaries are descriptive because each contains only three workloads.

| Kernel family | LFBO perf (ms) / time (s) | LLM perf (ms) / time (s) | Hybrid perf (ms) / time (s) | RAG-LLM perf (ms) / time (s) | Main observation |
|---|---:|---:|---:|---:|---|
| MatMul | 0.362500 / 72.75 | 0.163498 / 58.14 | 0.163192 / 124.01 | **0.162016** / 70.89 | All LLM-assisted methods roughly halve latency; RAG wins two shapes and Hybrid one. |
| Attention | 0.210637 / 146.41 | 0.140110 / **74.94** | 0.134008 / 239.10 | **0.133601** / 79.49 | Hybrid and RAG deliver similar quality; RAG retains near-LLM readiness. |
| Grouped GEMM | 0.147527 / 102.05 | 0.076798 / **54.72** | 0.077255 / 173.07 | **0.075401** / 64.10 | Large, consistent gains for all assisted methods; RAG has the best family quality. |
| Mamba-2 | 0.020513 / 78.29 | 0.016458 / **68.00** | 0.016484 / 139.74 | **0.015787** / 74.39 | Retrieval becomes more valuable at the larger sequence/state shapes. |
| GDN | 0.039709 / 136.94 | 0.039540 / **75.78** | **0.034111** / 202.01 | 0.035207 / 89.58 | Continued LFBO exploration is important; Hybrid wins all three GDN shapes. |
| RMSNorm | 0.106582 / 87.91 | **0.100724** / **57.24** | 0.100913 / 163.15 | 0.100834 / 66.65 | Final quality is nearly tied; LLM gives the best time-quality balance. |
| Softmax | 0.054322 / 137.04 | 0.054285 / **60.74** | 0.054284 / 193.40 | **0.054116** / 69.12 | Search changes readiness much more than final latency. |
| FP8 Attention | 3.563886 / 156.24 | 3.557487 / **73.55** | **3.378568** / 234.62 | 3.554495 / 78.78 | Hybrid helps the largest shape, but the family is otherwise close to parity. |
| SwiGLU | **0.050961** / 108.48 | 0.054810 / **52.61** | 0.054523 / 173.15 | 0.054701 / 61.78 | LFBO remains better in quality, especially at the smallest shape. |
| RoPE | **0.010629** / 248.89 | 0.011621 / **110.61** | 0.011438 / 391.27 | 0.011738 / 127.12 | LFBO wins all three shapes; assisted methods trade latency for faster readiness, except Hybrid. |
| Split-K MatMul | **0.034196** / 43.25 | 0.084868 / **32.98** | 0.034793 / 65.25 | 0.085406 / 38.21 | Direct LLM and RAG guidance fail badly at large K; LFBO/Hybrid exploration is essential. |

Standard MatMul, attention, grouped GEMM, and Mamba-2 are the strongest evidence for learned guidance. Their configuration landscapes appear to contain transferable structure that the LLM or retrieved examples can exploit. Softmax and normalization kernels are already near a broad performance plateau: search policy changes wall time substantially but selected latency only marginally. RoPE, SwiGLU, and Split-K MatMul demonstrate the opposite failure mode—learned priors can be confidently wrong when a family's optimal configuration requires specialized tiling or reduction behavior.

### 4. Robustness and outliers

The ±2.5% empirical parity band gives a more conservative view than exact win counts. Against LFBO, LLM produced **15 clear wins, 9 ties, and 9 losses**; Hybrid produced **19/8/6**; and RAG-LLM produced **16/8/9**. Hybrid has the most favorable distribution and the safest tail: none of its 33 kernel medians is more than 25% slower than LFBO, and its worst ratio is 1.197. LLM and RAG-LLM each have two losses exceeding 25%, both in large Split-K MatMul.

The most important outliers are `matmul_split_k-64x16384x64` and `matmul_split_k-64x65536x64`. At K=16,384, LLM and RAG-LLM are both approximately **5.33×** slower than LFBO; at K=65,536 they are approximately **2.60×** slower. Hybrid stays near LFBO on both because the LFBO phase can recover from an unhelpful LLM seed. These outliers also explain why the arithmetic impression from individual kernels can differ from the suite geometric mean. They should be highlighted as evidence for a guarded or adaptive policy rather than removed as inconvenient cases.

### 5. Search effort and convergence behavior

The four policies consume substantially different search effort. Across correct runs, median attempted/evaluated/benchmarked configurations were approximately **80/41/29 for LFBO**, **26/33/24 for LLM**, **80/71/50 for Hybrid**, and **26/33/24 for RAG-LLM**. Evaluation counts can exceed attempts because they include the full event accounting used by the harness; benchmark counts are lower because invalid, duplicate, or non-benchmarkable candidates are filtered.

The trajectory plots show that LLM-proposed and retrieval-conditioned candidates improve the incumbent early. On the representative 8192³ MatMul, LLM, Hybrid, and RAG-LLM reach their displayed best median internal latencies at about 25.4 s, compared with 39.7 s for LFBO. Across the full suite, once all workloads are observable, trajectory-oracle median regret is 42.3% for LFBO, 10.7% for LLM, 10.3% for Hybrid, and 4.9% for RAG-LLM. The final displayed medians are 42.3%, 6.4%, 5.0%, and 3.8%, respectively.

The ≤5% time-to-hit CDF provides a stricter production view. At approximately 26.4 s, LFBO has reached the target on 0.0% of workloads, LLM on 36.4%, Hybrid on 39.4%, and RAG-LLM on 42.4%. Final coverage rises to 24.2%, 36.4%, 51.5%, and 54.5%. Retrieval therefore leads both early and final strict-target coverage, but no method reaches the target universally. Claims of complete cold-start elimination would not be supported by this campaign.

### 6. Retrieval value and provider cost

RAG-LLM consumed 875,243 provider tokens, compared with 746,156 for LLM and 746,596 for Hybrid. This is **17.3% more tokens than direct LLM search**, or approximately 5,305 versus 4,522 tokens per run. The additional context buys a statistically supported 1.9% geometric-mean latency improvement over LLM, 21/2/10 kernel-level wins/ties/losses, and higher ≤5% target coverage. It also increases geometric-mean readiness by 14.3%.

The result is best interpreted as a measurable but non-free retrieval benefit. RAG is attractive when kernels will be reused enough to amortize a small runtime improvement, when early near-oracle coverage matters, or when the retrieval corpus closely matches the target family. Direct LLM search is preferable when tuning latency and provider cost dominate. Hybrid uses essentially the same token budget as LLM but gains quality by spending local search time rather than additional context tokens.

### 7. Correctness and reliability

All 660 scheduled runs completed. Every arm produced 164/165 correct results and achieved the predefined reliability coverage of at least three correct repetitions on all 33 workloads. The only residual correctness issue is `gdn_fwd_h-b1h4s4096ds64`, for which every approach achieved 4/5 correct repetitions. Because the failure is symmetric across arms, it does not create an obvious comparative reliability advantage, but it indicates a workload- or validation-specific issue that should be investigated before expanding the benchmark.

The matched statistical analysis uses 164/165 valid paired blocks (99.4% joint-success coverage). This high overlap is important: the reported ratios largely compare the same workload and seed under both policies rather than different subsets of successful runs.

### 8. Overall interpretation

The campaign does not identify one universally superior autotuner. Instead, it identifies three practical operating points:

- **LLM is the readiness-first policy:** lowest geometric-mean tuning time, 32/33 Pareto appearances, and moderate quality improvement without retrieval overhead.
- **Hybrid is the quality-first and risk-controlled policy:** best aggregate selected latency, strongest corrected LFBO result, most clear wins under the noise band, and protection against catastrophic LLM mis-seeding—at the cost of the longest tuning time.
- **RAG-LLM is the cold-start/coverage policy:** best strict-target trajectory coverage and a small but significant improvement over direct LLM search, with moderate time and token overhead.
- **LFBO remains an essential fallback:** it wins RoPE and the two largest Split-K shapes and requires no provider access, demonstrating that learned guidance should augment rather than eliminate classical exploration.

A production autotuner should therefore route policies by workload family and reuse horizon. A sensible design would use LLM or RAG-LLM for fast initial readiness, retain a family-aware escape hatch to LFBO for Split-K/RoPE-like cases, and continue into Hybrid exploration only when expected kernel reuse justifies the additional tuning cost.

## Selectable poster figures

### A. Aggregate effects versus LFBO


Points are geometric-mean ratios and error bars are 95% kernel-bootstrap intervals. The dashed line at 1 denotes parity with LFBO.

### B. Performance trend over wall time


Lower normalized incumbent is better. LLM and RAG-LLM reach near-best configurations earlier; LFBO and especially Hybrid continue searching longer.

### C. Performance trend over benchmarked configurations


This view separates candidate efficiency from wall-clock overhead and shows how quickly each policy reduces incumbent latency as configurations are benchmarked.

### D. Search overhead across representative kernels


The 15-workload poster subset spans every kernel family and is ordered by LFBO readiness difficulty. Geometric-mean evaluated configurations were **40.9 / 32.4 / 69.8 / 32.5** for LFBO / LLM / Hybrid / RAG-LLM; corresponding readiness geometric means were **120.9 / 68.3 / 188.6 / 75.9 s**. Hybrid explores most deeply, while LLM and RAG-LLM use the smallest search budgets.

### E. Cold-start step trajectory


For `matmul-8192x8192x8192`, the best median internal latencies for LFBO / LLM / Hybrid / RAG-LLM were **3.754 / 1.662 / 1.665 / 1.573 ms**. The three LLM-assisted approaches reached those displayed minima at approximately 25.4 s, while LFBO reached its minimum at 39.7 s. The post-step rendering does not imply an unmeasured result at time zero.

### F. Per-kernel RAG-LLM/LFBO performance ratios


Each kernel shows LLM/LFBO, Hybrid/LFBO, and RAG-LLM/LFBO; LFBO is the solid 1.0 reference. Under the ±2.5% band, LLM recorded **15 wins / 9 ties / 9 losses**, Hybrid **19/8/6**, and RAG-LLM **16/8/9**. Dashed lines mark the empirical parity band, and the logarithmic axis keeps the large Split-K regressions visible without hiding near-parity kernels.

### G. Suite-wide normalized trajectory regret


The heavy lines are suite medians and ribbons are workload-level 25th–75th percentiles for all four approaches. Once every workload was observable, median regrets were **42.3% LFBO, 10.7% LLM, 10.3% Hybrid, and 4.9% RAG-LLM**; final displayed medians were **42.3%, 6.4%, 5.0%, and 3.8%**, respectively. The reference is the best internal trajectory incumbent observed for each workload, distinct from the stabilized final-outcome oracle used in the result table.

### H. Time to hit the ≤5% regret target


This CDF keeps all 33 workloads in the denominator. By approximately 26.4 s, coverage was **0.0% / 36.4% / 39.4% / 42.4%** for LFBO / LLM / Hybrid / RAG-LLM. Final coverage was **24.2% / 36.4% / 51.5% / 54.5%**. Thus all LLM-assisted approaches accelerate cold start, with contextual RAG attaining the highest strict-target coverage; the data do not support the hypothetical claim that every workload is optimized within two seconds.

### I. Fifteen-workload small-multiple grid


The 3×5 grid exposes workload heterogeneity hidden by suite aggregates. Every panel contains all four approaches, retains its own latency scale, and shares a common wall-clock range and color convention. It is intended as the granular poster panel for attendees interested in specific operators.

Figures are not checked in; regenerate them from a campaign directory with `scripts/helion_rag/analyze_head_to_head.py`.

### J. Selected performance by kernel


Performance is normalized within each workload to LFBO=1 so kernels spanning several orders of magnitude remain comparable. Learned guidance produces large improvements on conventional attention, MatMul, grouped GEMM, and Mamba-2, while most saturated elementwise kernels cluster near parity. The two large Split-K cases are the dominant exceptions: LLM and RAG-LLM regress sharply, whereas Hybrid stays close to LFBO.

### K. Readiness wall time by kernel


Wall time has a much more stable policy ordering than performance. LLM is fastest or nearly fastest across the suite, RAG-LLM is slightly slower, LFBO is generally next, and Hybrid is consistently slowest. Workload complexity changes the magnitude of readiness time—especially for RoPE and large attention—but rarely changes the ordering.

Both plots regenerate from a campaign directory with `scripts/helion_rag/analyze_head_to_head.py`.

## Expanded per-kernel metrics

This table separates every approach and metric into its own column. Performance is median selected latency in milliseconds, wall time is median readiness in seconds, and Config is the median number of evaluated configurations. All values use correct completed repetitions.

| Kernel Type | Dimensions | LFBO Perf (ms) | LLM Perf (ms) | Hybrid Perf (ms) | RAG-LLM Perf (ms) | LFBO Wall Time (s) | LLM Wall Time (s) | Hybrid Wall Time (s) | RAG-LLM Wall Time (s) | LFBO Config | LLM Config | Hybrid Config | RAG-LLM Config |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MatMul (`matmul-1024x1024x1024`) | 1024×1024×1024 (M×K×N) | 0.023584 | 0.012064 | 0.012064 | 0.011968 | 57.48 | 52.51 | 103.35 | 72.48 | 42 | 34 | 72 | 34 |
| MatMul (`matmul-4096x4096x4096`) | 4096×4096×4096 (M×K×N) | 0.524480 | 0.210800 | 0.211488 | 0.207888 | 69.98 | 55.37 | 116.98 | 66.24 | 42 | 34 | 72 | 34 |
| MatMul (`matmul-8192x8192x8192`) | 8192×8192×8192 (M×K×N) | 3.851040 | 1.718592 | 1.703408 | 1.709312 | 95.72 | 67.58 | 157.74 | 74.22 | 42 | 34 | 72 | 34 |
| Split-K MatMul (`matmul_split_k-64x1024x64`) | 64×1024×64 (M×K×N) | 0.009856 | 0.010880 | 0.009472 | 0.011104 | 63.06 | 37.89 | 97.34 | 45.31 | 42 | 30 | 68 | 30 |
| Split-K MatMul (`matmul_split_k-64x16384x64`) | 64×16384×64 (M×K×N) | 0.038720 | 0.206272 | 0.038880 | 0.206528 | 35.58 | 30.20 | 53.22 | 34.76 | 36 | 28 | 59 | 28 |
| Split-K MatMul (`matmul_split_k-64x65536x64`) | 64×65536×64 (M×K×N) | 0.104784 | 0.272368 | 0.114368 | 0.271648 | 36.05 | 31.35 | 53.62 | 35.43 | 36 | 28 | 59 | 28 |
| Attention (`attention-2x8x512x64`) | 2×8×512×64 (B×H×S×D) | 0.020544 | 0.012832 | 0.012832 | 0.012736 | 100.51 | 64.53 | 189.72 | 68.81 | 41 | 34 | 72 | 34 |
| Attention (`attention-2x8x4096x64`) | 2×8×4096×64 (B×H×S×D) | 0.430976 | 0.250368 | 0.221888 | 0.222048 | 151.52 | 80.61 | 228.65 | 84.79 | 42 | 34 | 72 | 33 |
| Attention (`attention-2x8x8192x64`) | 2×8×8192×64 (B×H×S×D) | 1.055520 | 0.856112 | 0.845216 | 0.843232 | 206.08 | 80.89 | 315.09 | 86.08 | 42 | 34 | 72 | 34 |
| FP8 Attention (`fp8_attention-2x4x512x64`) | 2×4×512×64 (B×H×S×D) | 0.212384 | 0.212480 | 0.212480 | 0.212416 | 89.69 | 57.65 | 151.42 | 68.63 | 41 | 31 | 69 | 31 |
| FP8 Attention (`fp8_attention-2x4x2048x64`) | 2×4×2048×64 (B×H×S×D) | 3.711104 | 3.356976 | 3.354864 | 3.345648 | 152.59 | 71.59 | 233.19 | 71.27 | 41 | 30 | 66 | 29 |
| FP8 Attention (`fp8_attention-2x4x8192x64`) | 2×4×8192×64 (B×H×S×D) | 57.431025 | 63.119520 | 54.100977 | 63.192608 | 278.68 | 96.40 | 365.73 | 99.96 | 41 | 30 | 68 | 30 |
| Grouped GEMM (`grouped_gemm-g2m1024`) | G=2, M=1024, K=256, N=128 | 0.149216 | 0.083152 | 0.081520 | 0.082144 | 102.88 | 55.45 | 176.47 | 65.64 | 43 | 34 | 72 | 33 |
| Grouped GEMM (`grouped_gemm-g4m512`) | G=4, M=512, K=256, N=128 | 0.156688 | 0.071136 | 0.073024 | 0.070720 | 102.74 | 54.53 | 170.85 | 63.05 | 42 | 34 | 71 | 33 |
| Grouped GEMM (`grouped_gemm-g8m512`) | G=8, M=512, K=256, N=128 | 0.137328 | 0.076576 | 0.077456 | 0.073792 | 100.55 | 54.18 | 171.94 | 63.62 | 42 | 33 | 71 | 34 |
| SwiGLU (`swiglu-2048x2048`) | 2048×2048 (M×N) | 0.014144 | 0.017216 | 0.016928 | 0.017088 | 90.34 | 49.36 | 165.67 | 58.10 | 42 | 33 | 71 | 33 |
| SwiGLU (`swiglu-4096x4096`) | 4096×4096 (M×N) | 0.050528 | 0.051744 | 0.051840 | 0.051840 | 114.46 | 51.18 | 168.79 | 60.05 | 42 | 33 | 71 | 33 |
| SwiGLU (`swiglu-8192x8192`) | 8192×8192 (M×N) | 0.185184 | 0.184832 | 0.184704 | 0.184768 | 123.44 | 57.66 | 185.63 | 67.59 | 41 | 33 | 71 | 33 |
| Softmax (`softmax-4096x1024`) | 4096×1024 (M×N) | 0.009568 | 0.009536 | 0.009632 | 0.009536 | 121.35 | 61.56 | 182.10 | 67.32 | 40 | 33 | 71 | 33 |
| Softmax (`softmax-4096x8192`) | 4096×8192 (M×N) | 0.066304 | 0.066400 | 0.065728 | 0.065824 | 151.41 | 54.72 | 224.62 | 63.68 | 41 | 33 | 71 | 33 |
| Softmax (`softmax-4096x32768`) | 4096×32768 (M×N) | 0.252672 | 0.252640 | 0.252672 | 0.252480 | 140.07 | 66.53 | 176.84 | 77.04 | 41 | 33 | 71 | 33 |
| GDN forward (`gdn_fwd_h-b1h4s2048ds128`) | B=1, H=4, S=2048, Dh=64, Ds=128, C=128 | 0.023872 | 0.021664 | 0.020064 | 0.020384 | 137.40 | 75.98 | 203.09 | 89.48 | 41 | 33 | 71 | 33 |
| GDN forward (`gdn_fwd_h-b1h4s4096ds64`) | B=1, H=4, S=4096, Dh=64, Ds=64, C=128 | 0.034080 | 0.036848 | 0.032520 | 0.033960 | 136.13 | 70.79 | 193.73 | 84.36 | 41 | 33.5 | 71 | 33.5 |
| GDN forward (`gdn_fwd_h-b1h4s8192ds128`) | B=1, H=4, S=8192, Dh=64, Ds=128, C=128 | 0.076960 | 0.077440 | 0.060832 | 0.063040 | 137.31 | 80.90 | 209.53 | 95.24 | 41 | 33 | 71 | 33 |
| RMSNorm (`rms_norm-4096x1024`) | 4096×1024 (M×N) | 0.018688 | 0.016192 | 0.016256 | 0.016256 | 69.05 | 53.73 | 120.00 | 62.63 | 41 | 33 | 71 | 33 |
| RMSNorm (`rms_norm-4096x8192`) | 4096×8192 (M×N) | 0.130048 | 0.128256 | 0.128480 | 0.128288 | 74.23 | 56.08 | 178.52 | 67.38 | 40 | 33 | 71 | 33 |
| RMSNorm (`rms_norm-4096x32768`) | 4096×32768 (M×N) | 0.498176 | 0.492064 | 0.492032 | 0.491616 | 132.53 | 62.25 | 202.74 | 70.15 | 40 | 33 | 71 | 33 |
| RoPE (`rope-1x4x2x512x128`) | 1×4×2×512×128 (B×Hq×Hk×S×D) | 0.007232 | 0.008096 | 0.007808 | 0.008288 | 207.64 | 94.89 | 310.59 | 102.26 | 43 | 32 | 70 | 32 |
| RoPE (`rope-1x4x2x2048x128`) | 1×4×2×2048×128 (B×Hq×Hk×S×D) | 0.009216 | 0.010080 | 0.010016 | 0.010112 | 266.69 | 109.86 | 411.97 | 141.43 | 42 | 32 | 70 | 32 |
| RoPE (`rope-1x4x2x8192x128`) | 1×4×2×8192×128 (B×Hq×Hk×S×D) | 0.018016 | 0.019232 | 0.019136 | 0.019296 | 278.40 | 129.81 | 468.13 | 142.02 | 40 | 32 | 70 | 32 |
| Mamba-2 scan (`mamba2_chunk_scan-b1h4s2048ds128`) | B=1, H=4, S=2048, Dh=64, Ds=128, C=128 | 0.013472 | 0.011040 | 0.011232 | 0.011136 | 77.55 | 65.93 | 135.95 | 72.46 | 41 | 32 | 72 | 33 |
| Mamba-2 scan (`mamba2_chunk_scan-b1h4s4096ds256`) | B=1, H=4, S=4096, Dh=64, Ds=256, C=128 | 0.019552 | 0.015616 | 0.015520 | 0.014880 | 79.84 | 69.11 | 140.82 | 74.78 | 42 | 33 | 72 | 34 |
| Mamba-2 scan (`mamba2_chunk_scan-b1h4s8192ds256`) | B=1, H=4, S=8192, Dh=64, Ds=256, C=128 | 0.032768 | 0.025856 | 0.025696 | 0.023744 | 77.51 | 69.00 | 142.53 | 75.96 | 41 | 33 | 71 | 34 |

## Statistical pattern analysis of the expanded table

This analysis treats the 33 workloads as paired blocks and the four approaches as repeated measurements. It does not treat the 132 workload-arm cells as independent samples. Performance, wall time, and configuration statistics are the per-workload medians shown in the expanded table. Lower values and lower ranks are better.

### Omnibus differences among the four approaches

| Outcome | Friedman χ²(3) | *p* | Kendall's W | Mean ranks: LFBO / LLM / Hybrid / RAG-LLM | Interpretation |
|---|---:|---:|---:|---|---|
| Selected performance | 12.771 | 0.00516 | 0.129 | 3.015 / 2.758 / **2.091** / 2.136 | Approaches differ, but the small W shows strong kernel dependence. |
| Readiness wall time | 96.636 | 8.22×10⁻²¹ | 0.976 | 2.970 / **1.030** / 4.000 / 2.000 | Policy almost completely determines the readiness ordering. |
| Evaluated configurations | 96.403 | 9.22×10⁻²¹ | 0.974 | 3.000 / **1.500** / 4.000 / **1.500** | LLM and RAG use nearly identical budgets; Hybrid evaluates the most. |

The key hidden pattern is the difference in effect size. Search policy has a nearly deterministic effect on cost—wall time and configurations have Kendall's W≈0.97—but only a modest effect on final performance rank (W=0.129). Kernel identity and shape therefore matter far more for selected latency than for tuning overhead. This explains why a single global “winner” is inappropriate even though the overall performance test is significant.

The Friedman result is an omnibus test. The corrected paired contrasts reported earlier identify where the performance differences lie: Hybrid/LFBO, Hybrid/LLM, and RAG-LLM/LLM are significant after Holm correction, whereas LLM/LFBO and RAG-LLM/LFBO are not.

### Configuration count, wall time, and quality

Across the 99 non-LFBO workload-arm contrasts, the relative number of evaluated configurations is strongly associated with relative wall time (Spearman ρ=0.734, *p*=5.25×10⁻¹⁸). More configurations are also associated with a better latency ratio to LFBO (ρ=−0.318, *p*=0.00135), while longer relative wall time has a weaker association with better latency (ρ=−0.248, *p*=0.0133).

These pooled correlations are descriptive, not causal. They primarily separate the policies: Hybrid evaluates roughly 70–72 configurations and is slowest, whereas LLM and RAG usually evaluate 28–34 and finish earlier. Within each individual arm, configuration ratio is not significantly correlated with wall-time ratio (*p*=0.124–0.612). Thus the overhead is driven mainly by the policy's search regime, not by small workload-to-workload changes in configuration count.

Median wall time per evaluated configuration was 2.81 s for LFBO, 1.98 s for LLM, 2.49 s for Hybrid, and 2.18 s for RAG-LLM. RAG's per-evaluation time is about 9.9% above LLM even though their evaluated-configuration counts are almost identical, consistent with retrieval/context-processing overhead rather than extra benchmarking.

### Does learned guidance help more difficult kernels?

Using LFBO selected latency as a proxy for baseline difficulty, slower kernels tend to show larger relative gains from learned guidance. Spearman correlations between LFBO latency and the arm/LFBO latency ratio are −0.346 for LLM (*p*=0.048), −0.390 for Hybrid (*p*=0.0247), and −0.335 for RAG-LLM (*p*=0.0565). A negative coefficient means the relative latency ratio generally falls—improves—as baseline latency rises.

This is an exploratory trend rather than confirmatory evidence: after Holm correction across these three correlations, adjusted *p* values are 0.0968, 0.0741, and 0.0968. The pattern is nevertheless consistent with the family results, where standard MatMul and attention show large gains while very small or already-saturated elementwise kernels show near parity.

### Family-composition sensitivity

The following sensitivity analysis recomputes the geometric mean of the **table's per-kernel median latency ratios**. This ratio-of-medians estimand is used only to diagnose suite composition; it is distinct from the primary matched-repetition estimator reported earlier.

| Included workloads | LLM/LFBO | Hybrid/LFBO | RAG-LLM/LFBO | Hidden implication |
|---|---:|---:|---:|---|
| All 33 kernels | 0.907 | **0.817** | 0.889 | All assisted approaches improve the table-derived suite geomean. |
| Excluding Split-K MatMul | 0.821 | **0.800** | 0.802 | Two large Split-K failures conceal much stronger LLM/RAG gains elsewhere. |
| Excluding standard MatMul | 0.973 | **0.868** | 0.953 | Standard MatMul contributes a large share of LLM/RAG's aggregate benefit. |
| Leave-one-family-out range | 0.821–0.973 | **0.795–0.868** | 0.802–0.953 | Improvement direction is robust, but magnitude depends strongly on family mix. |

Hybrid is least vulnerable to suite composition: its leave-one-family-out ratio remains between 0.795 and 0.868. LLM and RAG remain below parity in every leave-one-family-out analysis, but their apparent gain varies from only 2.7–4.7% without standard MatMul to about 18–20% without Split-K. Benchmark composition must therefore be reported alongside any headline geomean.

### Retrieval isolates a quality effect rather than a search-budget effect

LLM and RAG-LLM evaluate effectively the same number of configurations: their table-derived configuration ratio has a geometric mean of 1.000 and a median of 1.000. Nevertheless, RAG/LLM selected latency has a geometric-mean ratio of 0.980, while readiness has a ratio of 1.143. Retrieval therefore changes **which configurations are selected**, not how many are evaluated.

Under the ±2.5% practical-equivalence band, RAG-LLM records 7 clear wins, 26 ties, and no clear losses against LLM. The median RAG/LLM latency ratio is 0.997 and its interquartile range is 0.986–1.001. This reveals a nuanced pattern behind the statistically significant 1.9% aggregate gain: most kernels move only slightly, seven improve materially, and none regress by more than the empirical noise threshold. The cost of this favorable shift is a 14.3% geometric-mean readiness increase and 17.3% more provider tokens.

### Repeatability across the five seeds

The coefficient of variation (CV) of selected latency across repetitions measures search-outcome stability, not microbenchmark measurement error alone. Median workload-level CVs and workload-bootstrap intervals are:

| Approach | Median CV | 95% bootstrap interval | Workloads with CV >5% | Clear W/T/L vs LFBO | Losses >25% vs LFBO |
|---|---:|---:|---:|---:|---:|
| LFBO | 18.9% | 9.8–26.6% | 25/33 | Reference | — |
| LLM | **2.5%** | 1.0–5.5% | 12/33 | 15/9/9 | 2 |
| Hybrid | 4.4% | 1.5–6.2% | 13/33 | **19/8/6** | **0** |
| RAG-LLM | 3.3% | 1.1–4.1% | **10/33** | 16/8/9 | 2 |

The LLM-assisted approaches are substantially more repeatable at the median than LFBO. LFBO's selected latency varies widely across seeds on Split-K MatMul, attention, and grouped GEMM; this suggests sensitivity to the random search trajectory. Hybrid combines much lower variability than LFBO with the safest performance tail: none of its kernel medians is more than 25% slower than LFBO. LLM and RAG each have two severe losses, both at the large Split-K shapes.

All arms show their maximum CV on a Split-K workload, reaching approximately 0.99–1.05. Consequently, Split-K is not merely an average-performance outlier; it is also the least stable family across seeds and should receive dedicated policy logic or additional repetitions.

### Patterns exposed by performance, time, and configurations jointly

1. **Search cost is predictable; selected quality is conditional.** LLM is almost always fastest to ready, Hybrid almost always slowest, and their configuration ordering barely changes. Performance rankings change substantially across families.
2. **Hybrid converts extra configurations into robustness.** It evaluates roughly twice as many configurations as LLM/RAG, achieves the best average performance rank, has no >25% LFBO regressions, and is least sensitive to removing any one family.
3. **RAG improves proposal quality rather than budget.** Its configuration count is indistinguishable from LLM, but its latency distribution shifts slightly downward with no practical-equivalence-band losses; time and token overhead are the price of context.
4. **Large conventional kernels are the best learned-guidance targets.** MatMul, attention, and grouped GEMM drive much of the positive effect. Already-saturated kernels offer little performance headroom, while specialized Split-K and RoPE landscapes can contradict learned priors.
5. **A routed policy is statistically better motivated than one global tuner.** Use LLM for readiness, RAG when early near-oracle coverage justifies context cost, Hybrid for high-reuse or risk-sensitive kernels, and LFBO/Hybrid fallbacks for Split-K and RoPE.

## Complete per-kernel four-arm table

Each approach cell is **median performance (ms) / median readiness wall time (s) / median evaluated configurations / regret**. All metrics use correct completed repetitions. Lower is better. All cells represent 5/5 correct repetitions except `gdn_fwd_h-b1h4s4096ds64`, which is 4/5 for every approach.

| Kernel type | Kernel identifier | Dimensions | LFBO | LLM | Hybrid | RAG-LLM |
|---|---|---|---:|---:|---:|---:|
| MatMul | `matmul-1024x1024x1024` | 1024×1024×1024 (M×K×N) | 0.023584 / 57.5 / 42 / +102.5% | 0.012064 / 52.5 / 34 / +3.6% | 0.012064 / 103.4 / 72 / +3.6% | 0.011968 / 72.5 / 34 / +2.7% |
| MatMul | `matmul-4096x4096x4096` | 4096×4096×4096 (M×K×N) | 0.524480 / 70.0 / 42 / +157.1% | 0.210800 / 55.4 / 34 / +3.3% | 0.211488 / 117.0 / 72 / +3.7% | 0.207888 / 66.2 / 34 / +1.9% |
| MatMul | `matmul-8192x8192x8192` | 8192×8192×8192 (M×K×N) | 3.851040 / 95.7 / 42 / +144.8% | 1.718592 / 67.6 / 34 / +9.2% | 1.703408 / 157.7 / 72 / +8.3% | 1.709312 / 74.2 / 34 / +8.6% |
| Split-K MatMul | `matmul_split_k-64x1024x64` | 64×1024×64 (M×K×N) | 0.009856 / 63.1 / 42 / +17.6% | 0.010880 / 37.9 / 30 / +29.8% | 0.009472 / 97.3 / 68 / +13.0% | 0.011104 / 45.3 / 30 / +32.4% |
| Split-K MatMul | `matmul_split_k-64x16384x64` | 64×16384×64 (M×K×N) | 0.038720 / 35.6 / 36 / +20.8% | 0.206272 / 30.2 / 28 / +543.3% | 0.038880 / 53.2 / 59 / +21.3% | 0.206528 / 34.8 / 28 / +544.1% |
| Split-K MatMul | `matmul_split_k-64x65536x64` | 64×65536×64 (M×K×N) | 0.104784 / 36.0 / 36 / +171.1% | 0.272368 / 31.4 / 28 / +604.6% | 0.114368 / 53.6 / 59 / +195.9% | 0.271648 / 35.4 / 28 / +602.7% |
| Attention | `attention-2x8x512x64` | 2×8×512×64 (B×H×S×D) | 0.020544 / 100.5 / 41 / +63.4% | 0.012832 / 64.5 / 34 / +2.0% | 0.012832 / 189.7 / 72 / +2.0% | 0.012736 / 68.8 / 34 / +1.3% |
| Attention | `attention-2x8x4096x64` | 2×8×4096×64 (B×H×S×D) | 0.430976 / 151.5 / 42 / +96.8% | 0.250368 / 80.6 / 34 / +14.3% | 0.221888 / 228.7 / 72 / +1.3% | 0.222048 / 84.8 / 33 / +1.4% |
| Attention | `attention-2x8x8192x64` | 2×8×8192×64 (B×H×S×D) | 1.055520 / 206.1 / 42 / +34.5% | 0.856112 / 80.9 / 34 / +9.1% | 0.845216 / 315.1 / 72 / +7.7% | 0.843232 / 86.1 / 34 / +7.4% |
| FP8 Attention | `fp8_attention-2x4x512x64` | 2×4×512×64 (B×H×S×D) | 0.212384 / 89.7 / 41 / +43.9% | 0.212480 / 57.6 / 31 / +44.0% | 0.212480 / 151.4 / 69 / +44.0% | 0.212416 / 68.6 / 31 / +44.0% |
| FP8 Attention | `fp8_attention-2x4x2048x64` | 2×4×2048×64 (B×H×S×D) | 3.711104 / 152.6 / 41 / +94.4% | 3.356976 / 71.6 / 30 / +75.9% | 3.354864 / 233.2 / 66 / +75.8% | 3.345648 / 71.3 / 29 / +75.3% |
| FP8 Attention | `fp8_attention-2x4x8192x64` | 2×4×8192×64 (B×H×S×D) | 57.431025 / 278.7 / 41 / +105.4% | 63.119520 / 96.4 / 30 / +125.7% | 54.100977 / 365.7 / 68 / +93.5% | 63.192608 / 100.0 / 30 / +126.0% |
| Grouped GEMM | `grouped_gemm-g2m1024` | G=2, M=1024, K=256, N=128 | 0.149216 / 102.9 / 43 / +87.5% | 0.083152 / 55.4 / 34 / +4.5% | 0.081520 / 176.5 / 72 / +2.4% | 0.082144 / 65.6 / 33 / +3.2% |
| Grouped GEMM | `grouped_gemm-g4m512` | G=4, M=512, K=256, N=128 | 0.156688 / 102.7 / 42 / +135.2% | 0.071136 / 54.5 / 34 / +6.8% | 0.073024 / 170.9 / 71 / +9.6% | 0.070720 / 63.0 / 33 / +6.1% |
| Grouped GEMM | `grouped_gemm-g8m512` | G=8, M=512, K=256, N=128 | 0.137328 / 100.6 / 42 / +92.6% | 0.076576 / 54.2 / 33 / +7.4% | 0.077456 / 171.9 / 71 / +8.6% | 0.073792 / 63.6 / 34 / +3.5% |
| SwiGLU | `swiglu-2048x2048` | 2048×2048 (M×N) | 0.014144 / 90.3 / 42 / +11.1% | 0.017216 / 49.4 / 33 / +35.2% | 0.016928 / 165.7 / 71 / +32.9% | 0.017088 / 58.1 / 33 / +34.2% |
| SwiGLU | `swiglu-4096x4096` | 4096×4096 (M×N) | 0.050528 / 114.5 / 42 / +1.2% | 0.051744 / 51.2 / 33 / +3.6% | 0.051840 / 168.8 / 71 / +3.8% | 0.051840 / 60.1 / 33 / +3.8% |
| SwiGLU | `swiglu-8192x8192` | 8192×8192 (M×N) | 0.185184 / 123.4 / 41 / +0.5% | 0.184832 / 57.7 / 33 / +0.3% | 0.184704 / 185.6 / 71 / +0.2% | 0.184768 / 67.6 / 33 / +0.2% |
| Softmax | `softmax-4096x1024` | 4096×1024 (M×N) | 0.009568 / 121.3 / 40 / +2.4% | 0.009536 / 61.6 / 33 / +2.1% | 0.009632 / 182.1 / 71 / +3.1% | 0.009536 / 67.3 / 33 / +2.1% |
| Softmax | `softmax-4096x8192` | 4096×8192 (M×N) | 0.066304 / 151.4 / 41 / +1.0% | 0.066400 / 54.7 / 33 / +1.2% | 0.065728 / 224.6 / 71 / +0.1% | 0.065824 / 63.7 / 33 / +0.3% |
| Softmax | `softmax-4096x32768` | 4096×32768 (M×N) | 0.252672 / 140.1 / 41 / +0.1% | 0.252640 / 66.5 / 33 / +0.1% | 0.252672 / 176.8 / 71 / +0.1% | 0.252480 / 77.0 / 33 / +0.1% |
| GDN forward | `gdn_fwd_h-b1h4s2048ds128` | B=1, H=4, S=2048, Dh=64, Ds=128, C=128 | 0.023872 / 137.4 / 41 / +31.8% | 0.021664 / 76.0 / 33 / +19.6% | 0.020064 / 203.1 / 71 / +10.8% | 0.020384 / 89.5 / 33 / +12.5% |
| GDN forward | `gdn_fwd_h-b1h4s4096ds64` | B=1, H=4, S=4096, Dh=64, Ds=64, C=128 | 0.034080 / 136.1 / 41 / +10.1% | 0.036848 / 70.8 / 33.5 / +19.1% | 0.032520 / 193.7 / 71 / +5.1% | 0.033960 / 84.4 / 33.5 / +9.7% |
| GDN forward | `gdn_fwd_h-b1h4s8192ds128` | B=1, H=4, S=8192, Dh=64, Ds=128, C=128 | 0.076960 / 137.3 / 41 / +43.1% | 0.077440 / 80.9 / 33 / +44.0% | 0.060832 / 209.5 / 71 / +13.1% | 0.063040 / 95.2 / 33 / +17.2% |
| RMSNorm | `rms_norm-4096x1024` | 4096×1024 (M×N) | 0.018688 / 69.0 / 41 / +16.3% | 0.016192 / 53.7 / 33 / +0.8% | 0.016256 / 120.0 / 71 / +1.2% | 0.016256 / 62.6 / 33 / +1.2% |
| RMSNorm | `rms_norm-4096x8192` | 4096×8192 (M×N) | 0.130048 / 74.2 / 40 / +1.5% | 0.128256 / 56.1 / 33 / +0.1% | 0.128480 / 178.5 / 71 / +0.2% | 0.128288 / 67.4 / 33 / +0.1% |
| RMSNorm | `rms_norm-4096x32768` | 4096×32768 (M×N) | 0.498176 / 132.5 / 40 / +1.4% | 0.492064 / 62.2 / 33 / +0.1% | 0.492032 / 202.7 / 71 / +0.1% | 0.491616 / 70.1 / 33 / +0.1% |
| RoPE | `rope-1x4x2x512x128` | 1×4×2×512×128 (B×Hq×Hk×S×D) | 0.007232 / 207.6 / 43 / +8.1% | 0.008096 / 94.9 / 32 / +21.1% | 0.007808 / 310.6 / 70 / +16.7% | 0.008288 / 102.3 / 32 / +23.9% |
| RoPE | `rope-1x4x2x2048x128` | 1×4×2×2048×128 (B×Hq×Hk×S×D) | 0.009216 / 266.7 / 42 / +11.6% | 0.010080 / 109.9 / 32 / +22.1% | 0.010016 / 412.0 / 70 / +21.3% | 0.010112 / 141.4 / 32 / +22.5% |
| RoPE | `rope-1x4x2x8192x128` | 1×4×2×8192×128 (B×Hq×Hk×S×D) | 0.018016 / 278.4 / 40 / +11.5% | 0.019232 / 129.8 / 32 / +19.0% | 0.019136 / 468.1 / 70 / +18.4% | 0.019296 / 142.0 / 32 / +19.4% |
| Mamba-2 scan | `mamba2_chunk_scan-b1h4s2048ds128` | B=1, H=4, S=2048, Dh=64, Ds=128, C=128 | 0.013472 / 77.6 / 41 / +28.0% | 0.011040 / 65.9 / 32 / +4.9% | 0.011232 / 135.9 / 72 / +6.7% | 0.011136 / 72.5 / 33 / +5.8% |
| Mamba-2 scan | `mamba2_chunk_scan-b1h4s4096ds256` | B=1, H=4, S=4096, Dh=64, Ds=256, C=128 | 0.019552 / 79.8 / 42 / +76.1% | 0.015616 / 69.1 / 33 / +40.6% | 0.015520 / 140.8 / 72 / +39.8% | 0.014880 / 74.8 / 34 / +34.0% |
| Mamba-2 scan | `mamba2_chunk_scan-b1h4s8192ds256` | B=1, H=4, S=8192, Dh=64, Ds=256, C=128 | 0.032768 / 77.5 / 41 / +53.8% | 0.025856 / 69.0 / 33 / +21.3% | 0.025696 / 142.5 / 71 / +20.6% | 0.023744 / 76.0 / 34 / +11.4% |

## Limitations and interpretation

- The bounded campaign oracle is the best observed result, not an exhaustive oracle; reported regret should not be interpreted as distance from the hardware optimum.
- Only five repetitions were run per workload-arm. Paired blocking and kernel-level aggregation reduce noise, but family-specific conclusions remain exploratory.
- The approaches do not consume equal numbers of candidates. This study compares end-to-end policies under a common cap, not performance at a strictly fixed evaluation budget.
- RAG-LLM token totals include retrieval-conditioned prompts and are therefore larger by design; token counts are reported, but monetary cost depends on provider pricing.
- Hardware identity, driver, framework, compiler, and clock-control metadata are not serialized in `study_manifest.json`. These details must be recovered and added before external submission because they are essential for reproducibility.
- One common correctness failure remains in each arm for `gdn_fwd_h-b1h4s4096ds64`; conclusions use the four matched correct repetitions.

## Conclusion

The experiment supports a joint performance-and-readiness framing rather than declaring a single winner. Hybrid is the strongest choice when final kernel latency dominates tuning cost. Direct LLM search is the strongest choice when rapid readiness matters. Contextual RAG improves direct LLM search by a small but statistically supported amount, while requiring additional wall time and tokens. A practical system should therefore select or adapt the search policy according to the workload's tuning-time budget, expected reuse, and value of marginal runtime improvement.

## Reproducibility and source artifacts

Final-state audit:

- `runs.jsonl`: 660 unique workload–arm–repetition records
- `results/`: 660 terminal JSON records
- `events/`: 660 trajectory logs
- `summary.json`: 660 completed, 0 failed, 132 workload-arm cells
- `analysis/per_run.csv`: 660 analyzed runs
- `analysis/trajectory_long.csv`: 29,052 candidate-evaluation records

Detailed machine-readable outputs are in `analysis/all_arm_table.csv`, `analysis/per_kernel_arm.csv`, `analysis/aggregate_statistics.csv`, `analysis/reliability.csv`, and `analysis/cost.csv`. The complete expanded Markdown arm table is `analysis/all_arm_table.md`.

The final analysis can be regenerated from the repository root with:

```bash
PYTHONPATH=scripts/helion_rag .venv/bin/python \
  scripts/helion_rag/analyze_head_to_head.py \
  --campaign .helion-rag/head_to_head_4arm_shapes
```
