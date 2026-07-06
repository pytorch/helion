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
| B1_T8192_H96_D128 | ok/ok | 0.318ms | 0.429ms | 134.9% | 1.538ms | 2.736ms | 177.9% |
| B2_T16384_H16_D128 | ok/ok | 0.222ms | 0.317ms | 142.8% | 1.107ms | 2.155ms | 194.7% |
| B4_T2048_H16_D128 | ok/ok | 0.106ms | 0.091ms | 85.8% | 0.948ms | 1.039ms | 109.6% |
| B4_T4096_H64_D128 | ok/ok | 0.428ms | 0.576ms | 134.6% | 2.002ms | 3.753ms | 187.5% |
| B8_T2048_H32_D256 | ok/ok | 0.658ms | 0.803ms | 122.0% | 3.007ms | 62.303ms | 2071.9% |
| B8_T1024_H8_D64 | ok/ok | 0.098ms | 0.089ms | 90.8% | 0.852ms | 1.103ms | 129.5% |

## simple_gla

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 1.429ms | 0.599ms | 41.9% | 6.255ms | 3.584ms | 57.3% |
| B2_T16384_H16_D128 | ok/ok | 0.984ms | 0.548ms | 55.7% | 5.228ms | 2.834ms | 54.2% |
| B4_T2048_H16_D128 | ok/ok | 0.287ms | 0.141ms | 49.1% | 1.248ms | 1.273ms | 102.0% |
| B4_T4096_H64_D128 | ok/ok | 1.876ms | 0.728ms | 38.8% | 8.737ms | 4.653ms | 53.3% |
| B8_T2048_H32_D256 | ok/ok | 2.147ms | 1.033ms | 48.1% | 12.670ms | 70.873ms | 559.4% |
| B8_T1024_H8_D64 | ok/ok | 0.199ms | 0.135ms | 67.8% | 1.291ms | 1.227ms | 95.0% |

## retention

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 1.426ms | 0.615ms | 43.1% | 6.242ms | 3.233ms | 51.8% |
| B2_T16384_H16_D128 | ok/ok | 0.982ms | 0.495ms | 50.4% | 5.218ms | 2.534ms | 48.6% |
| B4_T2048_H16_D128 | ok/ok | 0.286ms | 0.232ms | 81.1% | 1.327ms | 1.293ms | 97.4% |
| B4_T4096_H64_D128 | ok/ok | 1.871ms | 0.735ms | 39.3% | 8.571ms | 4.186ms | 48.8% |
| B8_T2048_H32_D256 | ok/ok | 2.146ms | 1.047ms | 48.8% | 12.649ms | 61.341ms | 484.9% |
| B8_T1024_H8_D64 | ok/ok | 0.189ms | 0.195ms | 103.2% | 1.274ms | 1.265ms | 99.3% |

## full_gla

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 1.833ms | 1.870ms | 102.0% | 7.660ms | 8.554ms | 111.7% |
| B2_T16384_H16_D128 | ok/ok | 1.346ms | 1.447ms | 107.5% | 5.662ms | 6.340ms | 112.0% |
| B4_T2048_H16_D128 | ok/ok | 0.395ms | 0.348ms | 88.1% | 1.493ms | 1.540ms | 103.1% |
| B4_T4096_H64_D128 | ok/ok | 2.363ms | 2.399ms | 101.5% | 10.394ms | 11.349ms | 109.2% |
| B8_T2048_H32_D256 | ok/ok | 3.151ms | 2.926ms | 92.9% | 13.532ms | 14.882ms | 110.0% |
| B8_T1024_H8_D64 | ok/ok | 0.167ms | 0.189ms | 113.2% | 1.216ms | 1.227ms | 100.9% |

## delta_rule

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 11.902ms | 1.286ms | 10.8% | 38.415ms | 4.510ms | 11.7% |
| B2_T16384_H16_D128 | ok/ok | 5.126ms | 0.963ms | 18.8% | 21.407ms | 3.310ms | 15.5% |
| B4_T2048_H16_D128 | ok/ok | 2.304ms | 0.321ms | 13.9% | 7.231ms | 1.658ms | 22.9% |
| B4_T4096_H64_D128 | ok/ok | 9.092ms | 1.678ms | 18.5% | 38.001ms | 5.877ms | 15.5% |
| B8_T2048_H32_D256 | ok/FAIL | 13.470ms | 1.805ms | 13.4% | 49.957ms | 8.559ms | 17.1% |
| B8_T1024_H8_D64 | ok/ok | 0.341ms | 0.316ms | 92.7% | 2.217ms | 1.665ms | 75.1% |

## gated_delta_rule

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 11.887ms | 1.192ms | 10.0% | 38.403ms | 5.318ms | 13.8% |
| B2_T16384_H16_D128 | ok/ok | 5.119ms | 0.917ms | 17.9% | 21.352ms | 4.018ms | 18.8% |
| B4_T2048_H16_D128 | ok/ok | 2.294ms | 0.331ms | 14.4% | 7.196ms | 1.814ms | 25.2% |
| B4_T4096_H64_D128 | ok/ok | 9.075ms | 1.503ms | 16.6% | 38.246ms | 6.843ms | 17.9% |
| B8_T2048_H32_D256 | ok/FAIL | 13.435ms | 1.939ms | 14.4% | 49.875ms | 9.057ms | 18.2% |
| B8_T1024_H8_D64 | ok/FAIL | 0.345ms | 0.331ms | 95.9% | 2.153ms | 1.813ms | 84.2% |

## kda

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 12.617ms | 2.196ms | 17.4% | 42.645ms | 11.924ms | 28.0% |
| B2_T16384_H16_D128 | ok/ok | 5.634ms | 1.576ms | 28.0% | 23.295ms | 8.304ms | 35.6% |
| B4_T2048_H16_D128 | ok/ok | 2.429ms | 0.505ms | 20.8% | 7.929ms | 2.138ms | 27.0% |
| B4_T4096_H64_D128 | ok/ok | 10.114ms | 2.864ms | 28.3% | 43.334ms | 15.649ms | 36.1% |
| B8_T2048_H32_D256 | ok/ok | 14.426ms | 3.702ms | 25.7% | 54.617ms | 20.076ms | 36.8% |
| B8_T1024_H8_D64 | ok/ok | 0.366ms | 0.500ms | 136.6% | 2.343ms | 2.109ms | 90.0% |

