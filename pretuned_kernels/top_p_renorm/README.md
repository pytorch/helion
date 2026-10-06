# Dense top-p renormalization

This is the measured ordinary AIR pipeline for contiguous FP32 probabilities
of shape `(16, 128512)`, scalar `p=0.1`, and 32 histogram partitions on GB300.
The fourteen saved stage configurations form a **fixed-config diagnostic**, not
a full cold-autotune winner. Run the included benchmark to measure the
seventeen-launch callable with the current compiler.

`stages.py` retains the ordinary stage bodies and their explicit FP32/FTZ
arithmetic. Identical pass bodies share seven kernel definitions, with
`pass_id` distinguishing the histogram, choose, and selected-count calls.
The three group-reduction stages have identical bodies and configurations.
Three histogram calls each issue a zeroing launch followed by a processing
launch. The remaining eleven stage calls each issue one launch.
The adjacent `_helion_aot_stages_...py` contains the exact fourteen configurations
and fixed structural policy. Runtime AOT discovery uses the defining source
filename (`stages.py`), then selects the function-specific key and config.
Selectors match every tensor's shape, stride, and dtype plus all scalar
arguments. They reject unknown signatures without tuning.

Run from the repository root with the repository's NVIDIA dependencies:

```bash
HELION_AOT_MODE=evaluate python -m pretuned_kernels.top_p_renorm.top_p_renorm
# Equivalently:
HELION_AOT_MODE=evaluate python pretuned_kernels/top_p_renorm/top_p_renorm.py
```

`check_case(SHAPES[0])` checks the ordinary callable and an independent Torch
sort/CDF reference before timing, using seeded FP32 standard-normal logits
followed by softmax, as in the measured case. Both keep whole equal-value
threshold groups.
The AIR cutoff is centered at `p * input_row_mass`; the checker retains the
existing eight-neighbor FP32 cutoff bound, `8 * float32_eps` relative output
normalization tolerance, and `2e-6` relative row-mass tolerance, with zero
absolute tolerance. Cutoff ambiguity never relaxes normalization or tie checks.
This bounded contract does not assert identical support across floating atomic
orders or reproduce a particular FlashInfer execution trajectory.

The full public calls, including intermediate allocations and every launch,
are captured. Checks verify input preservation and poison the output with NaNs
before graph replay. The shared pretuned benchmark timer measures the captured
whole call with L2 clearing outside the timed events; validation stays outside
the timed interval. `main()` reports fresh comparisons against Torch.
