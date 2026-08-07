# Geometric-mean perf & autotune time by kernel (rag_lfbo vs lfbo)

Reps collapsed by geomean per workload (equal weight), then geomean across each
kernel's held-out workloads — the same weighting the verdict statistics use.
`perf` = `perf_ms` (latency of chosen config); `tune` = `autotune_time_s` (search time).

| Kernel | perf_ms lfbo | perf_ms rag | perf ratio | tune_s lfbo | tune_s rag | time ratio |
|---|--:|--:|--:|--:|--:|--:|
| attention | 0.7362 | 0.6462 | 0.878 | 1661.0 | 1347.6 | 0.811 |
| cross_entropy | 0.8271 | 0.8260 | 0.999 | 768.6 | 617.6 | 0.804 |
| helion_mamba2_chunk_scan | 0.3790 | 0.3652 | 0.964 | 1756.0 | 1241.4 | 0.707 |
| helion_mamba2_chunk_state | 0.1538 | 0.1582 | 1.029 | 1185.6 | 1074.8 | 0.907 |
| jsd_forward | 0.7698 | 0.7663 | 0.996 | 629.5 | 620.5 | 0.986 |
| kl_div_forward | 0.4051 | 0.4046 | 0.999 | 476.4 | 412.4 | 0.866 |
| layer_norm_bwd | 0.1762 | 0.1781 | 1.011 | 1328.1 | 988.4 | 0.744 |
| layer_norm_fwd | 0.0888 | 0.0906 | 1.021 | 541.5 | 419.3 | 0.774 |
| matmul | 0.0533 | 0.0524 | 0.984 | 411.8 | 570.4 | 1.385 |
| matmul_bf16_int4 | 0.3127 | 0.3150 | 1.007 | 1275.1 | 1087.3 | 0.853 |
| rms_norm_bwd | 0.1438 | 0.1429 | 0.994 | 1221.9 | 1244.6 | 1.019 |
| rms_norm_fwd | 0.0552 | 0.0547 | 0.991 | 519.8 | 475.0 | 0.914 |
| rope_fwd \* | 0.0703 | 0.0973 | 1.078 | 1691.0 | 1959.5 | 0.843 |
| softmax_two_pass | 0.0350 | 0.0341 | 0.976 | 499.4 | 548.5 | 1.098 |
| welford | 1.5933 | 1.5934 | 1.000 | 834.2 | 784.4 | 0.940 |
| **OVERALL (geomean of workloads)** | **0.2194** | **0.2162** | **0.9923** | **861.9** | **801.8** | **0.8985** |

\* `rope_fwd` baseline geomean excludes one workload the cold LFBO search timed out on (RAG completed it),
so its two arms cover different workload sets and are not directly comparable.

The overall geomeans reproduce the report's pooled cluster-bootstrap point estimates
(perf 0.9923; autotune-time 0.8985 ≈ 0.8948 with resampling).

Chart: `perf_autotune_by_kernel.html` (grouped bars, both arms, per kernel — perf log-scale, tune linear).
