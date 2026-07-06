# Linear Attention Kernels: Accuracy

Forward/backward correctness of each variant vs the fp32 reference kernel, at autotune effort `none`. Cells are `fwd/bwd`: `ok` within tolerance, `FAIL` over it, `n/a` not measured.

| variant | B1_T8192_H96_D128 | B2_T16384_H16_D128 | B4_T2048_H16_D128 | B4_T4096_H64_D128 | B8_T2048_H32_D256 | B8_T1024_H8_D64 |
|---|---|---|---|---|---|---|
| vanilla_linear_attn | ok/ok | ok/ok | ok/ok | ok/ok | ok/ok | ok/ok |
| simple_gla | ok/ok | ok/ok | ok/ok | ok/ok | ok/ok | ok/ok |
| retention | ok/ok | ok/ok | ok/ok | ok/ok | ok/ok | ok/ok |
| full_gla | ok/ok | ok/ok | ok/ok | ok/ok | ok/ok | ok/ok |
| delta_rule | ok/ok | ok/ok | ok/ok | ok/ok | ok/FAIL | ok/ok |
| gated_delta_rule | ok/ok | ok/ok | ok/ok | ok/ok | ok/FAIL | ok/ok |
| kda | ok/ok | ok/ok | ok/ok | ok/ok | ok/ok | ok/ok |
