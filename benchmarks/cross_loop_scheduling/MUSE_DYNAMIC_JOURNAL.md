# Muse/Glimmer dynamic-batch scheduler probe

## Scope

This probe preserves the BF16 Muse/Glimmer FFN's five-stage boundary:

1. gate/up split-K
2. gate/up reduction
3. SiLU and multiply
4. down-projection split-K
5. down-projection reduction

The persistent kernel contains the same five Helion bodies used by the
five-launch standalone baseline.  Fusion, numerics, and kernel boundaries are
unchanged.  Only the batch dimension is dynamic (`1 <= B <= 4`); scratch
allocations have a fixed capacity of four batches.

The probe is
[`muse_glimmer_ffn_dynamic.py`](muse_glimmer_ffn_dynamic.py).

## Environment and command

- GPU: physical GPU 1, NVIDIA B200
- CUDA isolation: `CUDA_VISIBLE_DEVICES=1`
- Helion base commit: `ecbcfbb89ace815d18f913fd5b1b3046fc50d049`
  (`Pack parameterized root schedules across waves`)
- The run also included the uncommitted generic multi-axis root-order,
  symbolic projection, symbolic relation-coverage, and symbolic full-bound
  composition changes in the shared worktree.
- Python: `/home/eche/local/helion-clc/.venv/bin/python`

```bash
CUDA_VISIBLE_DEVICES=1 /home/eche/local/helion-clc/.venv/bin/python \
  benchmarks/cross_loop_scheduling/muse_glimmer_ffn_dynamic.py \
  --batch-sizes 1,2 --exemplar-batch 2 --multiplier 8 \
  --repetitions 100 --warmup-ms 4000 --skip-static \
  --output /tmp/muse_all_exact_b1_b2.json
```

All timings below are cold-L2 medians measured from pre-captured CUDA graphs.

## Configuration

- gate/up main: N32 x K128
- gate/up reduction: N64
- activation: N256
- down main: N32 x K128
- down reduction: N64
- persistent: one warp, two stages, `num_sm_multiplier=8`
- indexing: pointer for all bodies
- PID type: `persistent_blocked`
- cross-loop schedule: `static_pipeline`
- natural multi-axis root loop orders: identity

Multiplier 8 was the best of the tested values:

| multiplier | B1 persistent (us) | B1 standalone (us) |
| ---: | ---: | ---: |
| 1 | 2631.68 | 1664.99 |
| 8 | 1606.62 | 1647.70 |
| 12 | 1616.69 | 1649.02 |

## Correctness and cold-L2 performance after multi-axis counter support

All five active outputs were bit-exact against the matched standalone bodies
at both tested batch sizes (maximum absolute error 0).

| batch | parameterized persistent (us) | five-launch standalone (us) | persistent speedup |
| ---: | ---: | ---: | ---: |
| 1 | 1579.104 | 1649.600 | 1.0446x |
| 2 | 3110.080 | 3244.000 | 1.0431x |

This final timing run has exact counters on all four inter-root edges and no
root barriers.  B4 still needs to be rerun with the all-exact lowering.

Before symbolic multi-axis quotient/converse support, the all-root-barrier
lowering measured 1606.560/3162.048/6304.640 us for B1/B2/B4 versus
1647.696/3242.976/6321.088 us standalone.  That older run was bit-exact and
also reused one cubin across all three sizes.  B4 still needs to be rerun after
the exact-counter change.

These are deliberately matched pointer-body measurements, not the separately
pretuned tensor-descriptor Muse kernels.  They answer whether the
parameterized persistent lowering retains the performance of the exact
constituent bodies; they are not a claim about the production Muse absolute
latency.

## One-cubin evidence

Running B1 and B2 through the same compiled callable produced exactly one
specialization and one cubin:

- specialization count: 1
- Triton hash: `10685351a5ef1bcdf85ddbcc93428adc9bd3662619e06f512f3b9cb1fc113d2a`
- cubin SHA-256: `cc92edacc53059f4b39d5477a5445abedd6aac88f1d5c27fbf9d32b10b41ce03`

## Resources

| kernel | warps | registers/thread | spills/thread | shared bytes |
| --- | ---: | ---: | ---: | ---: |
| parameterized persistent | 1 | 168 | 2 | 768 |
| standalone gate main | 1 | 167 | 0 | 512 |
| standalone gate reduction | 1 | 48 | 0 | 0 |
| standalone activation | 1 | 64 | 0 | 0 |
| standalone down main | 1 | 144 | 0 | 768 |
| standalone down reduction | 1 | 47 | 0 | 0 |

## Scheduler path and remaining compiler gap

The natural source has five symbolic multi-axis root domains:

| root | axes | symbolic task count |
| --- | --- | --- |
| gate/up main | 5 | `19968 * B` |
| gate/up reduction | 4 | `640 * B` |
| activation | 2 | `16 * B` |
| down main | 3 | `3328 * B` |
| down reduction | 2 | `104 * B` |

Instrumentation showed that parameterized root-major scheduling was called
and emitted four resident segments; the final reduction is a final-arrival
continuation.  The parameterized event-frontier proposal was called but
declined.  After adjacent-target relation coalescing and exact nonuniform
fan-in bounds, all four edges use exact counters:

| edge | producer rank | consumer rank | event-key rank/size | fan-in | continuation |
| --- | ---: | ---: | --- | ---: | --- |
| gate/up main -> gate/up reduction | 5 | 4 | rank 4, `640 * B` | 32, tail 16 | no |
| gate/up reduction -> activation | 4 | 2 | rank 2, `16 * B` | 40 | no |
| activation -> down | 2 | 3 | rank 2, `16 * B` | 1 | no |
| down -> final reduction | 3 | 2 | rank 2, `104 * B` | 32 | yes |

The lowered kernel contains readiness waits, atomic exchanges, and atomic
adds, but no root barrier, dispatch ticket, or grid barrier.  Code generation
took 9.850 seconds and binary compilation took 9.462 seconds, 19.312 seconds
total for the final timed two-size run after moving union coalescing into
readiness construction and removing size-hint proofs.

The final 0->1 result required preserving a nonuniform static tail:

- Gate/up main -> gate/up reduction is one exact relation.  A reduction task
  over `(B, slice, half, N64)` maps to the same `B/slice/half`, producer N32
  tiles `[2*n, 2*n+2)`, and all 16 K splits.  There are 39 producer N tiles,
  so the last N64 tile is clipped and `39 != 2 * 20`.
  Factoring the dynamic B axis around the static inner relation derives exact
  publication and arrival-count relations.  Correlation-aware bounds prove
  that the per-key arrival expression is in `[16, 32]`; the epoch stride uses
  32 while each wait reads the exact per-key count.
Before adjacent-target coalescing, gate/up reduction -> activation was the
union of two exact relations, one for each gate/up half.  Each mapped
`(B, slice)` to all 20 N64 tiles in one half.  The union remained two pieces,
so `_separable_fixed_width_partition()` rejected `len(pieces) != 1` and no
publication/count proof was available.  Coalescing the adjacent half ranges
into one `[0, 2) x [0, 20)` producer box now proves uniform fan-in 40 and emits
the counter without changing source code.

For B1, multiplier 8 means 1184 resident workers.  The packed root-major
slots are root0 `[0,19968)`, root1 `[19968,20608)`, root2 `[20608,20624)`, and
root3 `[20624,23952)`.  The exact 0->1 counter lets 160 gate-reduction CTAs
placed beside the final 1024 gate CTAs in wave 16 proceed per tile when their
32 (or tail 16) arrivals complete.  The exact 1->2 counter similarly lets 16
activation CTAs placed beside the final 480 reduction CTAs in wave 17 proceed
per slice after 40 arrivals.  The probe has no device-side timing markers, so
this is an exact schedule-level overlap statement, not a measured Gantt claim.

## Static-reference limitation

The same source also defines a static-shape persistent kernel with the same
geometry and resources.  Its compilation did not complete in more than 150
seconds, even after disabling the global-list proposal.  The stack remained
in:

```text
_root_schedule_traversal
  -> task_order.converse
  -> _piecewise_single_source_mixed_radix_converse
  -> _simplify_logical_expression
  -> sympy.simplify
```

Consequently, this run has a direct dynamic-persistent versus matched
standalone comparison, but no honest static-persistent number.  The
parameterized path itself avoids this per-CTA/static converse compile-time
failure; a static comparison should be rerun after the canonical multi-axis
converse is made structural rather than expansion-heavy.

## Fixed-capacity tracing detail

Fixed-capacity dimensions and the assertion `B <= 4` must be literal in the
traced body.  Using a Python global for either caused TileDependency to record
an unbacked allocation hint `(8192,)`, mark the layout non-static, and lose the
exact dependency relation.  This is a tracing/canonicalization limitation,
not a scheduling heuristic.

## 2026-09-11 fixed-B1 retirement-frontier check

The current fixed-capacity compiler was measured on physical GPU 1 with the
same five mathematical stages and BF16/FP32 numerics. Cold-L2 timings use 500
samples after a 10-second warmup.

With gate N32, K128, down K128, one warp, two stages, multiplier eight, and
pipeline depth two, global placement and forced canonical placement both
measured about 188.3 us. The best independently configured standalone in that
run was 171.94 us, so N32 is not at parity.

With gate N64 and otherwise identical settings, event-frontier placement
measured 180.00 us versus 188.38 us for forced canonical placement. It splits
the 16-task activation root as 14 early tasks and 2 late tasks without
fragmenting the 10,240-task gate producer, then retains the down and final
reduction roots. This recovers 8.38 us from scheduling alone. The production
standalone measured 180.16 us (parity), but the best matched standalone was
171.94 us, leaving a 4.7% resource/configuration gap. A worker-width and
register/range-parameter sweep is therefore required; this result must not be
presented as final Muse parity.

Artifact: `/tmp/muse_fixed_depth2_gn64_gpu1.json`.

A follow-up worker-width sweep on physical GPU 5 kept gate N64, K128,
down K128, one warp, two stages, depth two, and all kernel bodies fixed:

| multiplier | workers | persistent latency |
| ---: | ---: | ---: |
| 6 | 888 | 198.560 us |
| 8 | 1184 | 177.984 us |
| 10 | 1480 | 171.968 us |
| 12 | 1776 | **161.728 us** |

The best standalone in the same paired run was 169.888 us, so multiplier 12
is 8.160 us (4.80%) faster. Multiplier 16 was correctly rejected because its
2,368 requested resident programs exceed the launcher's 1,924-block capacity.
Every valid persistent candidate is bit-exact against the matched standalone,
uses R126 with zero spills, 16,640 bytes shared, and one warp. All use the same
four exact counters and root-1 continuation; no root barrier, transient source,
or Muse-specific scheduler rule appears. Artifact:
`/tmp/muse_fixed_multiplier_sweep_gpu5.json`.

## 2026-09-12 unified-scheduler guard

The current source was rerun stably on physical GPU 1 with the established
B1 N64/K128/down-K128, one-warp, two-stage, multiplier-12, depth-two point.
Persistent latency is **161.632 us** versus **169.920 us** for the best matched
standalone boundary (4.88% lower latency). All five outputs are bit-exact, and
the persistent cubin remains R126/zero-spill/16,640-byte shared. The accepted
plan has five segments and four exact counters; no root barrier, transient
source, or workload-specific scheduler predicate is involved. Artifact:
`/tmp/muse_post_source_ticket_m12_gpu1.json`.
