# Reusable prompt: Helion CuTe hillclimb

Copy everything below the line. Replace the bracketed parts.

---

`/cute-hillclimb` I want to hill-climb the **[KERNEL]** kernel from
`benchmarks/cute/[BENCHMARK].py`, on the **[SHAPE]** shape only. Work in this
worktree.

Record every optimization you tried, the performance you reached, and the
elapsed time from the start of the session. Create a separate commit per
optimization containing the best kernel config for that step plus its
timestamp, so I can re-benchmark any step later.

Follow these rules — each one cost me real time in a previous run of this task:

**Baselines**

1. Do not trust a baseline at its library defaults. Before you conclude
   anything, check whether the baseline has an autotuned or hand-tuned
   configuration for this exact shape, and ask me for one if you cannot find
   it. In a previous run FA4's built-in heuristic measured 1245 TFLOP/s while
   FA4 autotuned for the same shape measured 1376 — the whole conclusion
   flipped. Add the tuned baseline to the harness as its own impl with a
   per-(GPU, shape) plan table and a `--<impl>-config KEY=VALUE` flag.
2. Once you have the best baseline, **ablate it one field at a time** and
   report which fields carry its advantage. That tells you what capability to
   go build, and it is much cheaper than guessing.

**Measurement**

3. Never compare numbers taken in different batches. Interleave the
   implementations (ABAB) inside one batch on one GPU, and report the pooled
   median over every `do_bench` sample, not a median of medians.
4. Before every measurement, check that sibling GPUs are idle. GB300 GPUs on
   one board share a power/cooling envelope: a neighbour drawing 826 W roughly
   halves the clocks of the GPU under test. In a previous run this produced a
   convincing 2.3x "regression" that did not exist.
5. If a kernel's samples are bimodal, check `nvidia-smi
   --query-gpu=clocks_throttle_reasons.active,power.draw,clocks.sm` during a
   run before attributing it to the code. Throttle reason `0x4` is the power
   cap. Bimodality that is fixed *per process* is usually the clock ramp, not
   the kernel: in a previous run 5 of 6 processes read 50.6 ms and the one
   launched right after the GPU went idle read 53.8 ms, while the within-
   process spread was 0.3%. Discard the first process after an idle period.
6. Report the checker's verdict, not your summary of it. If a number moves by
   less than the run-to-run spread, say it is even rather than claiming a win.

**Finding real wins**

7. If a cold full autotune converges onto exactly its own seed config, treat
   that as evidence of a **cliff in the search space**, not evidence that the
   seed is optimal. Test it: perturb the seed by one field at a time. If every
   single-field perturbation lands on the same worse number, some gate is
   requiring a byte-exact config match and silently disabling a lowering.
8. Grep the backend for gates of the form "this optimization applies only if
   the config equals the promoted seed". Replace them with the structural
   preconditions the optimization actually has. Prove the replacement is safe
   by dumping generated source for every seed the target policy promotes and
   diffing it against the unmodified tree — it should be byte-identical.
9. When a lowering is restricted to one pipeline family / CTA count / staging
   mode, check whether the restriction is real or inherited. In a previous run
   the dense resident softmax was gated to the 2-CTA pipeline while the
   equivalent causal lowering already ran on 1 CTA, and lifting it was worth
   20% for the whole 1-CTA family.
9b. When a deficit survives every knob and seed you can think of, stop tuning
   and compare the baseline's *tile shape* against ours. In a previous run the
   entire 8-shape gap came down to one number: every tuned FA4 plan used a
   160-wide KV tile and Helion had 128 spelled as a literal in the MMA tiler,
   the flat_divide tilers, the TMEM offsets, the ring sizing and the launcher.
   Making it a real searchable dimension was the whole win.
9c. A wider tile usually does not divide the sequence. Mask the trailing
   partial tile rather than restricting the width to divisors — restricting it
   makes the search surface depend on the sequence length, which breaks the
   length-invariance tests, and it is the long shapes you most want it on.
9d. After removing a cliff, re-run a cold autotune on *every* shape, not just
   the one you were chasing. The cliff was freezing all of them, and in a
   previous run six of eight shapes moved to a different config.

**Hygiene**

10. Do compiler edits in a separate `git worktree` while long autotunes are
    running; background runs re-read the working tree.
11. Keep `[cutedsl]` commits free of planning files. Plan docs, logs and step
    records go in `[noland]` commits.
12. Tell me plainly when a change is performance-neutral on the target shape
    and its value is generality. Do not dress it up as a win.

Environment for this box:

- `conda run -n helion-gb300-reference` (torch nightly + pytest; the
  `pytorch-3.12` env has no pytest and an older torch).
- FA4 needs `HELION_FA4_ROOT=/home/shangdiy/flash-attention` and
  `PYTHONPATH=/home/shangdiy/quack-cutlass-4.7`. Strict full-autotune runs
  reject `PYTHONPATH`, so use `env -u PYTHONPATH` for those.
- `ruff` is at `~/.conda/envs/crucible_grade/bin/ruff` (`./lint.sh` cannot find
  one).
- `test_examples.py` cannot be collected — `expecttest` is missing and I do not
  want packages installed. The `test_cute_flash_*.py` suites are the coverage
  to use. `test_cute_flash_length_invariance.py` has 2 pre-existing causal
  failures on `origin/main`; verify rather than assume they are yours.
