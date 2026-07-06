# Linear Attention Kernels: Helion vs FLA (ported from #1941)

Forward and forward+backward benchmarks for the linear-attention variants, Helion against FLA's hand-written Triton.

`%FLA = 100 * fla_ms / helion_ms`. Above 100% means Helion is faster.

Shapes are FLA's six production shapes (`flash-linear-attention/benchmarks/ops/registry.py:184`), bf16, `D = DV`.

`acc` is the harness's per-shape pass/fail (`fwd/bwd`): each side's Helion output and gradients are checked against the engine's own fp32 reference within tolerance.

To reproduce (full autotune by default; drop `--kernel` for all variants):

```bash
python -m benchmarks.run_linattn --kernel full_gla --output helionbench.json
```

# H100 (sm_90) — NVIDIA H100 80GB HBM3

## vanilla_linear_attn

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 0.690ms | 0.731ms | 105.9% | 2.767ms | 4.364ms | 157.7% |
| B2_T16384_H16_D128 | ok/ok | 0.464ms | 0.620ms | 133.6% | 1.879ms | 3.336ms | 177.6% |
| B4_T2048_H16_D128 | ok/ok | 0.215ms | 0.176ms | 81.9% | 0.797ms | 0.821ms | 103.0% |
| B4_T4096_H64_D128 | ok/ok | 0.904ms | 0.956ms | 105.8% | 3.659ms | 5.923ms | 161.9% |
| B8_T2048_H32_D256 | ok/ok | 1.317ms | 1.341ms | 101.8% | 5.437ms | 10.780ms | 198.3% |
| B8_T1024_H8_D64 | ok/ok | 0.217ms | 0.168ms | 77.4% | 0.776ms | 0.779ms | 100.4% |

## simple_gla

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 1.291ms | 0.804ms | 62.3% | 9.474ms | 5.698ms | 60.1% |
| B2_T16384_H16_D128 | ok/ok | 0.866ms | 0.546ms | 63.1% | 5.126ms | 3.667ms | 71.5% |
| B4_T2048_H16_D128 | ok/ok | 0.322ms | 0.254ms | 79.0% | 1.514ms | 1.740ms | 114.9% |
| B4_T4096_H64_D128 | ok/ok | 1.698ms | 1.017ms | 59.9% | 9.385ms | 6.718ms | 71.6% |
| B8_T2048_H32_D256 | ok/ok | 2.097ms | 1.426ms | 68.0% | 15.214ms | 10.514ms | 69.1% |
| B8_T1024_H8_D64 | ok/ok | 0.304ms | 0.247ms | 81.4% | 1.715ms | 1.307ms | 76.2% |

## retention

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 1.291ms | 0.991ms | 76.8% | 9.494ms | 4.501ms | 47.4% |
| B2_T16384_H16_D128 | ok/ok | 0.865ms | 0.889ms | 102.7% | 5.166ms | 3.553ms | 68.8% |
| B4_T2048_H16_D128 | ok/ok | 0.328ms | 0.397ms | 121.1% | 1.706ms | 1.721ms | 100.9% |
| B4_T4096_H64_D128 | ok/ok | 1.689ms | 1.247ms | 73.8% | 9.456ms | 6.121ms | 64.7% |
| B8_T2048_H32_D256 | ok/ok | 2.094ms | 1.579ms | 75.4% | 15.192ms | 10.552ms | 69.5% |
| B8_T1024_H8_D64 | ok/ok | 0.359ms | 0.370ms | 103.0% | 1.533ms | 1.472ms | 96.0% |

## full_gla

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 2.914ms | 2.416ms | 82.9% | 12.278ms | 11.134ms | 90.7% |
| B2_T16384_H16_D128 | ok/ok | 2.004ms | 1.837ms | 91.7% | 8.342ms | 8.088ms | 97.0% |
| B4_T2048_H16_D128 | ok/ok | 0.541ms | 0.447ms | 82.7% | 2.174ms | 1.986ms | 91.3% |
| B4_T4096_H64_D128 | ok/ok | 3.792ms | 3.199ms | 84.3% | 15.965ms | 15.074ms | 94.4% |
| B8_T2048_H32_D256 | ok/ok | 4.690ms | 3.622ms | 77.2% | 19.436ms | 17.273ms | 88.9% |
| B8_T1024_H8_D64 | ok/ok | 0.333ms | 0.374ms | 112.1% | 1.022ms | 1.369ms | 133.9% |

## delta_rule

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 10.241ms | 1.806ms | 17.6% | 45.160ms | 6.567ms | 14.5% |
| B2_T16384_H16_D128 | ok/ok | 7.070ms | 1.176ms | 16.6% | 29.285ms | 4.310ms | 14.7% |
| B4_T2048_H16_D128 | ok/ok | 1.546ms | 0.438ms | 28.3% | 7.191ms | 1.864ms | 25.9% |
| B4_T4096_H64_D128 | ok/ok | 12.344ms | 2.336ms | 18.9% | 53.548ms | 8.587ms | 16.0% |
| B8_T2048_H32_D256 | ok/ok | 14.561ms | 2.519ms | 17.3% | 61.937ms | 11.189ms | 18.1% |
| B8_T1024_H8_D64 | FAIL/ok | 0.469ms | 0.405ms | 86.3% | 2.475ms | 1.414ms | 57.1% |

## gated_delta_rule

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 10.195ms | 1.623ms | 15.9% | 44.968ms | 6.925ms | 15.4% |
| B2_T16384_H16_D128 | ok/ok | 7.062ms | 1.062ms | 15.0% | 29.307ms | 4.595ms | 15.7% |
| B4_T2048_H16_D128 | ok/ok | 1.552ms | 0.491ms | 31.7% | 7.204ms | 1.746ms | 24.2% |
| B4_T4096_H64_D128 | ok/ok | 12.124ms | 2.046ms | 16.9% | 53.758ms | 8.925ms | 16.6% |
| B8_T2048_H32_D256 | ok/ok | 13.955ms | 2.441ms | 17.5% | 61.294ms | 13.717ms | 22.4% |
| B8_T1024_H8_D64 | FAIL/ok | 0.487ms | 0.479ms | 98.4% | 2.966ms | 2.562ms | 86.4% |

## kda

| Shape | acc (fwd/bwd) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok/ok | 11.878ms | 3.184ms | 26.8% | 52.319ms | 17.387ms | 33.2% |
| B2_T16384_H16_D128 | ok/ok | 8.194ms | 2.223ms | 27.1% | 35.501ms | 11.719ms | 33.0% |
| B4_T2048_H16_D128 | ok/ok | 1.857ms | 0.600ms | 32.3% | 8.811ms | 2.998ms | 34.0% |
| B4_T4096_H64_D128 | ok/ok | 14.445ms | 4.072ms | 28.2% | 66.470ms | 22.650ms | 34.1% |
| B8_T2048_H32_D256 | ok/ok | 16.203ms | 4.376ms | 27.0% | 73.040ms | 24.834ms | 34.0% |
| B8_T1024_H8_D64 | FAIL/ok | 0.517ms | 0.602ms | 116.3% | 2.675ms | 1.905ms | 71.2% |
