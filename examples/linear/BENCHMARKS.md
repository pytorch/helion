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

Caveats from the run:

- `simple_gla` and `delta_rule` are absent. GPU 2's shard crashed during `delta_rule` autotuning (`NoConfigFound` on the `chunk_fwd_wy_diag` correction kernel, 288/697 configs failing to compile), so `simple_gla`'s already-completed results were lost with the crash. Being recollected now.

## vanilla_linear_attn

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 2.862ms | 0.433ms | 15.1% | 8.371ms | 2.781ms | 33.2% |
| B2_T16384_H16_D128 | ok/ok | 2.032ms | 0.319ms | 15.7% | 6.654ms | 2.196ms | 33.0% |
| B4_T2048_H16_D128 | ok/ok | 0.530ms | 0.092ms | 17.4% | 1.479ms | 0.980ms | 66.2% |
| B4_T4096_H64_D128 | ok/ok | 3.817ms | 0.580ms | 15.2% | 12.760ms | 3.833ms | 30.0% |
| B8_T2048_H32_D256 | ok/ok | 4.496ms | 0.795ms | 17.7% | 14.212ms | 63.624ms | 447.7% |
| B8_T1024_H8_D64 | ok/ok | 0.228ms | 0.089ms | 39.0% | 1.322ms | 1.081ms | 81.8% |

## retention

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 2.874ms | 0.613ms | 21.3% | 9.704ms | 3.300ms | 34.0% |
| B2_T16384_H16_D128 | ok/ok | 2.022ms | 0.496ms | 24.5% | 6.708ms | 2.605ms | 38.8% |
| B4_T2048_H16_D128 | ok/ok | 0.539ms | 0.230ms | 42.6% | 1.769ms | 1.209ms | 68.3% |
| B4_T4096_H64_D128 | ok/ok | 3.827ms | 0.740ms | 19.3% | 12.957ms | 4.288ms | 33.1% |
| B8_T2048_H32_D256 | ok/ok | 4.631ms | 1.053ms | 22.7% | 15.129ms | 62.601ms | 413.8% |
| B8_T1024_H8_D64 | ok/ok | 0.208ms | 0.194ms | 93.3% | 1.276ms | 1.009ms | 79.0% |

## full_gla

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 3.288ms | 3.602ms | 109.5% | 13.258ms | 16.491ms | 124.4% |
| B2_T16384_H16_D128 | ok/ok | 2.394ms | 2.703ms | 112.9% | 9.703ms | 11.989ms | 123.6% |
| B4_T2048_H16_D128 | ok/ok | 0.715ms | 0.666ms | 93.1% | 2.609ms | 2.978ms | 114.1% |
| B4_T4096_H64_D128 | ok/ok | 4.185ms | 4.598ms | 109.9% | 17.689ms | 21.875ms | 123.7% |
| B8_T2048_H32_D256 | ok/ok | 5.804ms | 5.694ms | 98.1% | 22.944ms | 29.019ms | 126.5% |
| B8_T1024_H8_D64 | ok/ok | 0.275ms | 0.218ms | 79.2% | 1.396ms | 1.208ms | 86.5% |

## gated_delta_rule

`bwd` fails on the two shapes below where the engine's gradients exceed the reference tolerance; the f+b timings are still reported.

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 7.378ms | 1.204ms | 16.3% | 31.159ms | 5.388ms | 17.3% |
| B2_T16384_H16_D128 | ok/ok | 5.290ms | 0.925ms | 17.5% | 22.024ms | 4.072ms | 18.5% |
| B4_T2048_H16_D128 | ok/ok | 1.368ms | 0.347ms | 25.4% | 5.721ms | 1.820ms | 31.8% |
| B4_T4096_H64_D128 | ok/ok | 9.348ms | 1.517ms | 16.2% | 40.695ms | 6.945ms | 17.1% |
| B8_T2048_H32_D256 | ok/FAIL | 13.798ms | 1.958ms | 14.2% | 50.098ms | 9.168ms | 18.3% |
| B8_T1024_H8_D64 | ok/FAIL | 0.355ms | 0.335ms | 94.5% | 2.207ms | 1.762ms | 79.9% |

## kda

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 15.058ms | 2.241ms | 14.9% | 47.752ms | 11.204ms | 23.5% |
| B2_T16384_H16_D128 | ok/ok | 10.467ms | 1.609ms | 15.4% | 33.167ms | 7.777ms | 23.4% |
| B4_T2048_H16_D128 | ok/ok | 2.507ms | 0.503ms | 20.1% | 8.174ms | 2.111ms | 25.8% |
| B4_T4096_H64_D128 | ok/ok | 19.834ms | 2.925ms | 14.7% | 62.914ms | 14.615ms | 23.2% |
| B8_T2048_H32_D256 | ok/ok | 24.053ms | 3.771ms | 15.7% | 73.766ms | 19.095ms | 25.9% |
| B8_T1024_H8_D64 | ok/ok | 0.383ms | 0.514ms | 134.0% | 2.482ms | 2.124ms | 85.6% |

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
| B1_T8192_H96_D128 | ok/ok | 3.317ms | 4.372ms | 131.8% | 13.316ms | 17.468ms | 131.2% |
| B2_T16384_H16_D128 | ok/ok | 2.417ms | 3.235ms | 133.8% | 9.739ms | 12.903ms | 132.5% |
| B4_T2048_H16_D128 | ok/ok | 0.714ms | 0.812ms | 113.7% | 2.623ms | 3.156ms | 120.3% |
| B4_T4096_H64_D128 | ok/ok | 4.219ms | 5.741ms | 136.1% | 17.752ms | 23.156ms | 130.4% |
| B8_T2048_H32_D256 | ok/ok | 5.841ms | 7.579ms | 129.7% | 23.003ms | 29.507ms | 128.3% |
| B8_T1024_H8_D64 | ok/ok | 0.284ms | 0.236ms | 83.1% | 1.272ms | 1.030ms | 81.0% |
