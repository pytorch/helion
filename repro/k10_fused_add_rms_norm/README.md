# [no land] Fused residual RMSNorm configs and reproducer

This artifact keeps the canonical Helion kernel and the twelve selected
GB300 CuTe configs. It normalizes the FP32 input/residual sum and returns
the normalized output plus the input-dtype rounded residual. Inputs remain
unchanged. BF16 uses relative L2 limits of 0.005 globally and per row;
FP16 uses 0.001. The runner also checks exact residual rounding, changed
inputs and weights, and CUDA graph replay/restoration.

Run from a Helion checkout or environment with the CuTe dependencies:

```bash
CUDA_VISIBLE_DEVICES=0 python repro/k10_fused_add_rms_norm/run.py --case 9 --fixed
# Or select a checkout explicitly:
CUDA_VISIBLE_DEVICES=0 python repro/k10_fused_add_rms_norm/run.py --case 9 --fixed --core /path/to/helion
```

Portable runs label source identity `UNVERIFIED_PORTABLE`. The optional
`--verify-frozen-source` mode retains the original exact-source guards and
requires the archived snapshot paths recorded in `provenance/source_pins.json`.
Case 9 needs the affine and block-dimension prefetch commits in the parent
stack; main alone emits a different program for that config. Cases 7–8
use legitimate main FULL configs verified on the stack. Their original
private FULL choices remain in the linked provenance.

All twelve configs passed 396 native correctness checks. The portable
bootstrap passed 99 additional checks covering frozen and portable modes.
The parent five-commit source passed lint, code review, the full default
suite (17,811 tests and 3,215 subtests), and the full CuTe suite (27,164
tests and 4,648 subtests), with no failures or errors. Those tests validate
the unchanged source tree recorded in `provenance/final_validation.json`.

The native paired campaign has geometric-mean target/CuTe ratio 1.051409.
Ten shapes beat their strongest validated target; the largest BF16 and
FP16 shapes are within 1% of QuACK. This runner performs correctness checks
with fixed configs and does not run autotuning or performance measurement.
Configs, original FULL winners, canonical inputs, precision policy and
source/measurement links are kept for reproducibility. This commit is
intended as an experiment artifact and should not land with the compiler PRs.
