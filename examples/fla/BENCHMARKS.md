# Linear Attention Kernels: Helion vs FLA (implemented from scratch)

Forward and forward+backward benchmarks for the chunked linear-attention kernels, Helion against FLA's hand-written Triton.

`%FLA = 100 * fla_ms / helion_ms`. Above 100% means Helion is faster.

Shapes are FLA's six production shapes (`flash-linear-attention/benchmarks/ops/registry.py:184`), bf16, `D = DV`.

To reproduce (full autotune by default; drop `--kernel` for all four):

```bash
python -m benchmarks.run_linattn --kernel linear_attn --output helionbench.json
```

Caveats:

- These times were collected with the earlier per-example `bench.py` scripts (since removed); the current `run_linattn.py` harness times the same kernels, so the numbers should be unchanged.
- The accuracy metric below is the old `bench.py` one and no longer matches the harness: `acc` is the worst `get_err_ratio` (relative RMS error) of Helion / FLA against the fp32 naive recurrence across the output and all gradients (`o`, `dq`, `dk`, `dv`), `ok` when Helion's worst ratio is within 2x FLA's, and `bwd:oom` means the fp32 reference backward OOM'd so the gradient ratios were skipped and the shown number is the forward (`o`) ratio only. The current harness instead reports a 0/1 pass/fail against absolute tolerances (`ACC_FWD_TOL`, `ACC_BWD_TOL`); the tables will switch to that on the next run.
- Both the H100 and B200 tables use `quick`-autotune Helion configs; CI runs a full autotune, so the next run's numbers may differ.

# H100 (sm_90)

## linear_attn (vanilla)

| Shape | acc (h/fla) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok 3.3e-03/3.3e-03 | 0.682ms | 0.726ms | 106.4% | 2.010ms | 3.367ms | 167.5% |
| B2_T16384_H16_D128 | ok 3.3e-03/3.3e-03 | 0.457ms | 0.609ms | 133.4% | 1.355ms | 2.585ms | 190.7% |
| B4_T2048_H16_D128 | ok 3.3e-03/3.3e-03 | 0.149ms | 0.148ms | 99.7% | 0.956ms | 1.547ms | 161.9% |
| B4_T4096_H64_D128 | ok 3.3e-03/3.3e-03 | 0.900ms | 0.950ms | 105.6% | 2.659ms | 4.530ms | 170.4% |
| B8_T2048_H32_D256 | ok 2.3e-03/2.3e-03 (bwd:oom) | 1.271ms | 1.338ms | 105.3% | 3.994ms | 8.981ms | 224.8% |
| B8_T1024_H8_D64 | ok 2.9e-03/2.9e-03 | 0.143ms | 0.137ms | 95.3% | 0.755ms | 0.796ms | 105.3% |

## retention

| Shape | acc (h/fla) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok 2.9e-03/3.3e-03 | 0.691ms | 0.961ms | 139.1% | 2.057ms | 3.426ms | 166.5% |
| B2_T16384_H16_D128 | ok 3.0e-03/3.4e-03 | 0.466ms | 0.868ms | 186.3% | 1.380ms | 2.776ms | 201.1% |
| B4_T2048_H16_D128 | ok 3.1e-03/3.5e-03 | 0.164ms | 0.358ms | 217.8% | 0.758ms | 1.225ms | 161.6% |
| B4_T4096_H64_D128 | ok 2.9e-03/3.3e-03 | 0.905ms | 1.147ms | 126.7% | 6.260ms | 4.524ms | 72.3% |
| B8_T2048_H32_D256 | ok 2.4e-03/2.4e-03 (bwd:oom) | 1.271ms | 1.598ms | 125.7% | 3.909ms | 9.079ms | 232.3% |
| B8_T1024_H8_D64 | ok 3.3e-03/3.3e-03 | 0.151ms | 0.316ms | 209.5% | 0.824ms | 1.299ms | 157.7% |

# B200 (sm_100)

## linear_attn (vanilla)

| Shape | acc (h/fla) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok 3.3e-03/3.3e-03 | 0.319ms | 0.433ms | 135.7% | 1.228ms | 1.929ms | 157.1% |
| B2_T16384_H16_D128 | ok 3.3e-03/3.3e-03 | 0.250ms | 0.320ms | 128.1% | 0.969ms | 1.571ms | 162.1% |
| B4_T2048_H16_D128 | ok 3.3e-03/3.3e-03 | 0.089ms | 0.090ms | 101.9% | 0.607ms | 0.947ms | 156.1% |
| B4_T4096_H64_D128 | ok 3.3e-03/3.3e-03 | 0.451ms | 0.579ms | 128.4% | 1.624ms | 2.587ms | 159.4% |
| B8_T2048_H32_D256 | ok 2.9e-03/2.9e-03 | 0.703ms | 0.800ms | 113.8% | 2.611ms | 62.367ms | 2388.4% |
| B8_T1024_H8_D64 | ok 2.9e-03/2.9e-03 | 0.081ms | 0.088ms | 108.8% | 0.677ms | 1.031ms | 152.2% |

## retention

| Shape | acc (h/fla) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok 2.9e-03/3.3e-03 | 0.435ms | 0.611ms | 140.6% | 1.515ms | 2.431ms | 160.5% |
| B2_T16384_H16_D128 | ok 3.0e-03/3.4e-03 | 0.304ms | 0.479ms | 157.5% | 1.190ms | 1.938ms | 162.9% |
| B4_T2048_H16_D128 | ok 3.1e-03/3.5e-03 | 0.097ms | 0.230ms | 237.7% | 2.157ms | 1.137ms | 52.7% |
| B4_T4096_H64_D128 | ok 2.9e-03/3.3e-03 | 0.560ms | 0.737ms | 131.6% | 1.966ms | 3.044ms | 154.9% |
| B8_T2048_H32_D256 | ok 3.0e-03/3.0e-03 | 0.812ms | 1.116ms | 137.4% | 2.940ms | 61.566ms | 2094.2% |
| B8_T1024_H8_D64 | ok 3.3e-03/3.3e-03 | 0.088ms | 0.188ms | 214.4% | 0.461ms | 0.995ms | 215.7% |

## simple_gla

| Shape | acc (h/fla) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok 2.9e-03/2.9e-03 | 0.432ms | 0.615ms | 142.6% | 1.964ms | 2.804ms | 142.7% |
| B2_T16384_H16_D128 | ok 2.9e-03/3.1e-02 | 0.355ms | 0.556ms | 156.8% | 1.473ms | 2.255ms | 153.1% |
| B4_T2048_H16_D128 | ok 2.9e-03/2.9e-02 | 0.128ms | 0.128ms | 100.0% | 0.890ms | 1.216ms | 136.7% |
| B4_T4096_H64_D128 | ok 2.9e-03/3.3e-02 | 0.571ms | 0.791ms | 138.5% | 2.453ms | 3.511ms | 143.1% |
| B8_T2048_H32_D256 | ok 2.9e-03/9.8e-03 | 0.884ms | 1.154ms | 130.6% | 4.331ms | 71.789ms | 1657.7% |
| B8_T1024_H8_D64 | ok 2.9e-03/2.9e-03 | 0.121ms | 0.118ms | 97.3% | 0.686ms | 0.956ms | 139.4% |

## gla

| Shape | acc (h/fla) | helion fwd | fla fwd | fwd %FLA | helion f+b | fla f+b | f+b %FLA |
|---|---|---|---|---|---|---|---|
| B1_T8192_H96_D128 | ok 4.1e-03/2.4e-03 | 1.265ms | 1.916ms | 151.4% | 3.561ms | 7.854ms | 220.5% |
| B2_T16384_H16_D128 | ok 4.1e-03/2.4e-03 | 0.828ms | 1.478ms | 178.4% | 2.559ms | 5.798ms | 226.6% |
| B4_T2048_H16_D128 | ok 4.0e-03/2.4e-03 | 0.236ms | 0.358ms | 151.8% | 0.895ms | 1.414ms | 158.0% |
| B4_T4096_H64_D128 | ok 4.1e-03/2.4e-03 | 1.570ms | 2.464ms | 157.0% | 4.431ms | 10.308ms | 232.6% |
| B8_T2048_H32_D256 | ok 4.1e-03/2.4e-03 | 2.466ms | 2.990ms | 121.3% | 6.413ms | 13.802ms | 215.2% |
| B8_T1024_H8_D64 | ok 4.1e-03/2.4e-03 | 0.169ms | 0.194ms | 114.5% | 0.836ms | 1.237ms | 147.9% |
