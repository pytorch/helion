# Linear Attention Kernels: Helion vs FLA (ported from #1941)

Forward and forward+backward benchmarks for the linear-attention variants, Helion against FLA's hand-written Triton.

`%FLA = 100 * fla_ms / helion_ms`. Above 100% means Helion is faster.

Shapes are FLA's six production shapes (`flash-linear-attention/benchmarks/ops/registry.py:184`), bf16, `D = DV`.

`acc` is the harness's per-shape pass/fail (`fwd/bwd`): each side's Helion output and gradients are checked against the engine's own fp32 reference within tolerance.

To reproduce (full autotune by default; drop `--kernel` for all variants):

```bash
python -m benchmarks.run_linattn --kernel full_gla --output helionbench.json
```

# B200 (sm_100)

## vanilla_linear_attn

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 0.319ms | 0.432ms | 135.4% | 1.602ms | 2.778ms | 173.3% |
| B2_T16384_H16_D128 | ok/ok | 0.223ms | 0.319ms | 143.4% | 1.125ms | 2.194ms | 195.0% |
| B4_T2048_H16_D128 | ok/ok | 0.106ms | 0.088ms | 83.6% | 0.977ms | 1.081ms | 110.7% |
| B4_T4096_H64_D128 | ok/ok | 0.429ms | 0.578ms | 134.8% | 2.089ms | 3.827ms | 183.2% |
| B8_T2048_H32_D256 | ok/ok | 0.659ms | 0.796ms | 120.8% | 3.128ms | 63.667ms | 2035.4% |
| B8_T1024_H8_D64 | ok/ok | 0.098ms | 0.086ms | 87.7% | 0.918ms | 1.099ms | 119.8% |

## simple_gla

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 1.467ms | 0.610ms | 41.6% | 6.512ms | 3.653ms | 56.1% |
| B2_T16384_H16_D128 | ok/ok | 1.008ms | 0.554ms | 54.9% | 5.369ms | 2.877ms | 53.6% |
| B4_T2048_H16_D128 | ok/ok | 0.294ms | 0.137ms | 46.6% | 1.349ms | 1.340ms | 99.3% |
| B4_T4096_H64_D128 | ok/ok | 1.921ms | 0.736ms | 38.3% | 8.841ms | 4.738ms | 53.6% |
| B8_T2048_H32_D256 | ok/ok | 2.184ms | 1.129ms | 51.7% | 12.644ms | 72.729ms | 575.2% |
| B8_T1024_H8_D64 | ok/ok | 0.192ms | 0.131ms | 68.2% | 1.296ms | 1.279ms | 98.7% |

## retention

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 1.508ms | 0.621ms | 41.1% | 6.543ms | 3.280ms | 50.1% |
| B2_T16384_H16_D128 | ok/ok | 1.010ms | 0.501ms | 49.7% | 5.254ms | 2.593ms | 49.4% |
| B4_T2048_H16_D128 | ok/ok | 0.294ms | 0.235ms | 79.9% | 1.244ms | 1.180ms | 94.8% |
| B4_T4096_H64_D128 | ok/ok | 1.925ms | 0.742ms | 38.6% | 8.841ms | 4.254ms | 48.1% |
| B8_T2048_H32_D256 | ok/ok | 2.198ms | 1.052ms | 47.9% | 12.873ms | 62.873ms | 488.4% |
| B8_T1024_H8_D64 | ok/ok | 0.199ms | 0.199ms | 99.9% | 1.196ms | 1.182ms | 98.8% |

## full_gla

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 3.306ms | 3.602ms | 109.0% | 13.217ms | 16.486ms | 124.7% |
| B2_T16384_H16_D128 | ok/ok | 2.366ms | 2.704ms | 114.3% | 9.636ms | 11.990ms | 124.4% |
| B4_T2048_H16_D128 | ok/ok | 0.712ms | 0.666ms | 93.4% | 2.588ms | 2.973ms | 114.9% |
| B4_T4096_H64_D128 | ok/ok | 4.226ms | 4.597ms | 108.8% | 17.670ms | 21.878ms | 123.8% |
| B8_T2048_H32_D256 | ok/ok | 5.682ms | 5.692ms | 100.2% | 23.440ms | 29.011ms | 123.8% |
| B8_T1024_H8_D64 | ok/ok | 0.270ms | 0.206ms | 76.1% | 1.322ms | 1.276ms | 96.5% |

## rwkv6

The two timed ops are not identical. Both share the state recurrence

```
S_t = exp(g_t) * S_{t-1} + k_t^T @ v_t
```

but differ in the output:

```
Helion (timed):  o_t = (q_t @ S_t) * gate_t     # output gate applied
FLA   (timed):   o_t =  q_t @ S_t               # no output gate
```

So the Helion side computes an extra elementwise `* gate_t` that the timed FLA `chunk_rwkv6` never does, yet is still faster.

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 3.337ms | 4.367ms | 130.9% | 13.273ms | 17.466ms | 131.6% |
| B2_T16384_H16_D128 | ok/ok | 2.394ms | 3.233ms | 135.0% | 9.679ms | 12.907ms | 133.4% |
| B4_T2048_H16_D128 | ok/ok | 0.717ms | 0.812ms | 113.2% | 2.604ms | 3.157ms | 121.2% |
| B4_T4096_H64_D128 | ok/ok | 4.268ms | 5.743ms | 134.6% | 17.740ms | 23.151ms | 130.5% |
| B8_T2048_H32_D256 | ok/ok | 5.717ms | 7.575ms | 132.5% | 23.516ms | 29.528ms | 125.6% |
| B8_T1024_H8_D64 | ok/ok | 0.282ms | 0.236ms | 83.8% | 1.251ms | 1.215ms | 97.1% |

## delta_rule

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 12.308ms | 1.313ms | 10.7% | 39.603ms | 4.572ms | 11.5% |
| B2_T16384_H16_D128 | ok/ok | 5.240ms | 0.981ms | 18.7% | 21.754ms | 3.355ms | 15.4% |
| B4_T2048_H16_D128 | ok/ok | 2.394ms | 0.330ms | 13.8% | 7.445ms | 1.733ms | 23.3% |
| B4_T4096_H64_D128 | ok/ok | 9.345ms | 1.712ms | 18.3% | 38.972ms | 5.968ms | 15.3% |
| B8_T2048_H32_D256 | ok/FAIL | 13.773ms | 1.822ms | 13.2% | 51.030ms | 8.639ms | 16.9% |
| B8_T1024_H8_D64 | ok/FAIL | 0.357ms | 0.326ms | 91.3% | 2.283ms | 1.665ms | 72.9% |

## gated_delta_rule

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 12.325ms | 1.204ms | 9.8% | 39.790ms | 5.390ms | 13.5% |
| B2_T16384_H16_D128 | ok/ok | 5.241ms | 0.926ms | 17.7% | 21.875ms | 4.073ms | 18.6% |
| B4_T2048_H16_D128 | ok/ok | 2.393ms | 0.323ms | 13.5% | 7.447ms | 1.801ms | 24.2% |
| B4_T4096_H64_D128 | ok/ok | 9.357ms | 1.516ms | 16.2% | 39.006ms | 6.939ms | 17.8% |
| B8_T2048_H32_D256 | ok/FAIL | 13.780ms | 1.955ms | 14.2% | 51.231ms | 9.162ms | 17.9% |
| B8_T1024_H8_D64 | ok/FAIL | 0.358ms | 0.324ms | 90.6% | 2.225ms | 1.815ms | 81.6% |

## kda

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 13.067ms | 2.241ms | 17.2% | 43.868ms | 11.201ms | 25.5% |
| B2_T16384_H16_D128 | ok/ok | 5.759ms | 1.609ms | 27.9% | 23.774ms | 7.784ms | 32.7% |
| B4_T2048_H16_D128 | ok/ok | 2.509ms | 0.514ms | 20.5% | 8.183ms | 2.131ms | 26.0% |
| B4_T4096_H64_D128 | ok/ok | 10.395ms | 2.923ms | 28.1% | 44.345ms | 14.610ms | 32.9% |
| B8_T2048_H32_D256 | ok/ok | 14.793ms | 3.772ms | 25.5% | 55.760ms | 19.116ms | 34.3% |
| B8_T1024_H8_D64 | ok/ok | 0.386ms | 0.498ms | 129.0% | 2.412ms | 2.062ms | 85.5% |
