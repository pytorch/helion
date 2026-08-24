# Gemma 4 A4B Helion Megakernel Stress Test

This note treats the A4B benchmark files as a source of representative kernels,
not as an R1-R10 checklist. The initial scheduling probe made no compiler
changes; its counter-layout result subsequently motivated the generic
cache-line-isolation fix described below.

## Best batch-1 result

On an uncontended NVIDIA B200, with route skew 2:

| implementation | median latency |
| --- | ---: |
| Helion megakernel | **34.22 us** |
| separately tuned Helion, matched decomposition | 39.05 us |
| hand-written Triton megakernel | 30.94 us |

The Helion megakernel is 12.4% faster than its in-process standalone baseline.
The hand Triton result and its Helion control were measured in a second process,
because that harness owns both implementations; its matched Helion control was
37.15 us, confirming the known several-percent cross-process drift.

The best probe configuration is:

```text
source_mode=assignment_hierarchical_topk
router_block=8
gate_block=16
gate_block_k=256
gate_stages=3
down_block=64
down_block_k=64
down_stages=5
reduce_block=256
num_warps=4
maxnreg=128
num_sm_multiplier=4
cross_loop_num_workers=0
```

Run it with:

```bash
PYTHONPATH=$PWD:/home/eche/local/helion-gemma4/benchmarks/gemma4 \
CUDA_VISIBLE_DEVICES=0 \
/home/eche/.conda/envs/helion-cu13/bin/python \
  benchmarks/gemma4/helion_gemma4_a4b_moe_megakernel.py \
  --batch 1 --route-skew 2 \
  --source-mode assignment_hierarchical_topk --config-mode matched \
  --workers 0 --worker-multiplier 4 --num-warps 4 --maxnreg 128 \
  --reduce-block 256 --benchmark --repeats 50 --batch-replays 100
```

Add `--print-lowered` to emit the complete lowered Triton kernel.

## Best grouped batch-8 result

The production grouped formulation required one additional source-layout
change: keep the gate/down intermediate in `(expert tile, row, column)` order
and forward each tile's expert, assignment, validity, and route-weight metadata.
This turns the important gate-to-down edge from an indirect allocation relation
into an ordinary 64-key event with fan-in 11.

With route skew 2, the best interleaved result was:

| implementation | median latency |
| --- | ---: |
| Helion task-major grouped megakernel | **69.78 us** |
| separately tuned grouped Helion | 92.86 us |
| hand-written Triton megakernel | approximately 60.97 us |

At route skew 0, the same configuration measured 146.67 us versus 147.15 us
for separate grouped Helion: effectively tied rather than regressing. These
measurements were taken while another process retained GPU memory but showed no
compute utilization, so the interleaved ratios are useful but should be
repeated on a completely uncontended GPU.

The best grouped configuration is:

```text
source_mode=grouped_task_major
group_gate_block=64
group_gate_block_k=128
group_down_block=256
group_down_block_k=64
group_reduce_block=64
gate_stages=3
down_stages=3
num_warps=4
maxnreg=256                 # compiles to 238 registers, no spills
num_sm_multiplier=4
cross_loop_num_workers=288 # normalized to the legal 286-worker candidate
```

Run it with:

```bash
PYTHONPATH=$PWD:/home/eche/local/helion-gemma4/benchmarks/gemma4 \
CUDA_VISIBLE_DEVICES=0 \
/home/eche/.conda/envs/helion-cu13/bin/python \
  benchmarks/gemma4/helion_gemma4_a4b_moe_megakernel.py \
  --batch 8 --route-skew 2 \
  --source-mode grouped_task_major --config-mode matched \
  --workers 288 --worker-multiplier 4 --num-warps 4 --maxnreg 256 \
  --group-down-block 256 --gate-stages 3 --down-stages 3 \
  --benchmark --repeats 30 --batch-replays 100
```

## Source changes relative to the production kernels

The performant batch-1 source deliberately changes the Helion program while
leaving compiler codegen untouched:

- Router projection is split into expert blocks and rematerializes the small
  router RMS input.
- Exact top-k is hierarchical: four independent 32-expert top-8 selections are
  merged by one 32-candidate top-8 selection. Tests over several seeds match the
  reference IDs and weights.
- Pre-MoE RMSNorm is rematerialized inside each assignment-local gate/up task.
- Gate/up and GeGLU are fused in source.
- The routing weight is applied in the down projection, so the final reduction
  is an unweighted sum.
- Selected route IDs and weights are forwarded through the gate output boundary.
  This is currently needed because the event lowerer cannot publish one coarse
  producer task into several fine-grained joined event keys.

The first five changes are computation/layout choices appropriate for a kernel
author or autotuner. The final metadata-forwarding change is a compiler
workaround and should disappear once multi-key event contributions lower
directly.

## Generated schedule

For batch 1, the lowered kernel contains:

1. Router projection: 16 tasks.
2. Grouped candidate top-k: 4 tasks after router family completion.
3. Candidate merge: one final-arrival task after all four candidate tasks.
4. Gate/up plus GeGLU: 352 tasks after routing completion.
5. Down projection: 352 statically scheduled tasks, each waiting on one of 8
   route-slot events with fan-in 44.
6. Expert reduction: 11 final-arrival tasks, each triggered after 32 down tiles.
7. Post-RMSNorm: one task after completion of all 11 reductions.

This is produced from the event DAG. There is no A4B, MoE, FFN, or root-number
matcher in the scheduler.

## NCU comparison

One-pass targeted NCU measurements:

| metric | Helion | hand Triton |
| --- | ---: | ---: |
| profiled duration | 73.18 us | 65.44 us |
| DRAM bytes | 98.83 MB | 98.53 MB |
| DRAM peak utilization | 17.63% | 19.65% |
| active warps | 15.05% | 17.25% |
| eligible warps/cycle | 0.371 | 0.449 |
| long-scoreboard stall | 48.18% | 51.21% |
| registers/thread | 109 | 117 |
| grid | 592 | 444 |

The remaining gap is not excess memory traffic. The hand kernel sustains more
active and eligible warps. Forcing a 444-CTA Helion diagnostic did not improve
latency, so the gap is not explained by worker count alone.

For the grouped batch-8 kernels, targeted NCU likewise shows essentially equal
DRAM traffic:

| metric | Helion task-major | hand Triton |
| --- | ---: | ---: |
| profiled duration | 124.32 us | 112.54 us |
| DRAM bytes | 159.44 MB | 159.61 MB |
| DRAM peak utilization | 16.73% | 18.50% |
| active warps | 11.95% | 11.27% |
| eligible warps/cycle | 0.088 | 0.099 |
| long-scoreboard stall | 72.90% | 68.53% |
| registers/thread | 238 | 138 |
| grid | 286 | 296 |

The remaining hand-Triton advantage is therefore execution efficiency and
register pressure, not avoidable expert-weight traffic.

## Batch-1 gap localization

An interleaved same-process comparison removes the earlier cross-process
uncertainty: with identical tensors and 444 requested workers, the retained
Helion source measured 34.96 us and the hand Triton probe measured 31.01 us.
Router fission was already present in both. The Helion source uses sixteen
8-expert projection tasks and a four-way hierarchical top-k; a direct
128-expert `torch.topk` spills 138 registers and measures 39.70 us, while a
no-spill iterative singleton top-k measures 45.37 us.

The remaining gap came from event-state layout. The hand probe places each
gate-to-down readiness key on a distinct 128-byte cache line. The previous
Helion lowering packed the eight counters into adjacent `uint32` words.
Changing only the hand probe from a 32-word stride to a one-word stride slows
it from 31.00 us to 32.54 us while its matched baseline remains
37.03--37.05 us.

The initial compiler diagnostic that cache-line-aligns only the eight
gate-to-down keys gives 31.07 us for Helion versus 31.02 us for hand Triton.
Padding only the eleven down-to-reduction keys does not recover the
performance. Focused NCU measurements reinforce the causal result:

| metric | packed gate keys | padded gate keys |
| --- | ---: | ---: |
| profiled duration | 70.05 us | 66.05 us |
| L2 atomic-input active cycles | 4,259 | 2,975 |
| L2 atomic requests | 526 | 526 |
| long-scoreboard stall | 49.19% | 45.89% |

This is false sharing between independently updated readiness keys, not a
missing MoE schedule. The compiler now gives every counted-event key
cache-line-separated storage. This is a generic event-layout invariant based
on the target cache-line size, not an autotuned or model-specific decision.
Matched checks show Qwen unchanged at 79.10 versus 79.07 us and Gemma E4B
improving from 74.13 to 73.56 us, with unchanged compiled resources. The A4B
diagnostic lowering is saved as
`/tmp/a4b_helion_padded_gate_counters_lowered.txt`.

### Preserving the vLLM GeGLU boundary

The production vLLM Triton MoE path materializes the W13 gate/up result, runs
`gelu_tanh_and_mul` as a separate kernel, and then runs W2. A second Helion
source variant preserves that gate/up-to-GeGLU boundary while retaining router
fission and pre-norm rematerialization.

The generic dependency pass accepts the extra root without a special case. It
derives six gate/up-to-GeGLU readiness milestones with arrivals
`(64, 64, 64, 64, 64, 32)`, followed by eight GeGLU/metadata-to-down joins.
With today's packed counters, this extra event exaggerates the cost: 37.09 us
unfused versus 35.18 us fused. With cache-line-separated keyed counters, the
same interleaved comparison is 32.88 us unfused versus 32.68 us fused. A
128-wide GeGLU tile is best among the tested 64, 128, and 256 widths.

Therefore source-level gate/up-plus-GeGLU fusion is not required for competitive
performance. Once event storage is fixed, Helion can preserve vLLM's ordinary
materialization boundary for only about 0.2 us in this experiment. The complete
lowering is saved as
`/tmp/a4b_helion_unfused_geglu_padded_counters_lowered.txt`.

## Robustness results

- Batches 1, 2, and 8 are numerically correct with explicit
  `(batch, route-slot, tile)` task coordinates.
- Batch 2 remains correct with gate and down L2 groupings of 8 and 4. The
  lowered code uses logical coordinates for event keys and distinct remapped
  physical coordinates for root execution.
- A 512-wide reduction produces six keys with exact fan-in
  `(64, 64, 64, 64, 64, 32)` and remains correct.
- Aliases and views used by the hierarchical candidate merge and expert-weight
  flattening are normalized sufficiently for correctness.
- Several conditional stores and data-dependent masks conservatively lift
  publication to guaranteed enclosing scopes.

## Stress findings

### Multi-key joined events

Without forwarding route metadata, the gate-to-down and down-to-reduce memory
relations are exact, but a singleton top-k producer contributes to every joined
key. Current counted-event lowering rejects any producer task that maps to more
than one key, causing the whole boundary to fall back to family completion.

The event IR already permits the required relation. Lowering should either
publish the task to each key or factor the coarse prerequisite from the
fine-grained event.

### Nonuniform final-arrival execution

The 512-wide reduction tail proves piecewise expected counts correctly, but it
uses a statically scheduled reduction rather than letting the final contributor
execute it. The final-arrival comparison can use the expected count for the
current key, so nonuniform fan-in should not require another executor policy.

### Multi-wave schedule expansion

At batch 8, the static down-task permutation expands into nested `tl.where`
expressions up to roughly 12 KiB on one source line. Batch 16 lowering completed,
but the end-to-end probe took 174.6 seconds and peaked at about 2.2 GiB RSS.
The worker schedule needs a factored Cartesian/periodic lowering rather than a
piecewise scalar select over flattened task intervals.

### Grouped/ragged kernels

The assignment-local batch-8 probe is 142.33 us versus 94.21 us for tuned
grouped Helion because it reloads expert weights per assignment. This is the
wrong source formulation for a shared-expert batch.

Directly stacking the production grouped kernels initially selected the
cooperative-grid fallback because `max_active_tiles` remained a runtime value.
Adding `max_active_tiles = hl.specialize(max_active_tiles)` restores the normal
static event scheduler. This is a source annotation, not a need for a dynamic
queue.

The verbatim grouped gate and down roots use family completion because both
index the intermediate through routing-dependent `active_tiles` and `order`
arrays. With standalone-derived tilings that form reports 128 registers, 232
spills, and roughly 138 us versus roughly 93 us separately.

Changing only the Helion source to store the intermediate by logical
expert-tile coordinate lets the existing compiler derive exact gate-to-down
readiness. Forwarding the route metadata through the same task-major boundary
also avoids an unrelated coarse top-k dependency. Replacing the fused router
with the hierarchical router removes all spills. No compiler or schedule
matcher is involved.

Per-range staging is critical: the unstaged task-major form measured about
97.8 us, while `gate_stages=3` reduced it to roughly 72 us. Widening the down
output tile from 64 to 256 and allowing 238 registers selected a 286-worker
schedule and reached 69.8 us. Copying the standalone tensor-descriptor choices
positionally into this changed source regressed the comparable configuration
from about 72 us to 100 us; indexing choices must be tuned for the actual access
form rather than inherited by kernel name.

### Ordinary affine-vector indexing

The candidate top-k load uses the contiguous expression
`group * group_size + arange(group_size)`, but it is currently represented as
an unknown subscript. Consequently the router-to-candidate edge uses family
completion even though four disjoint producer regions are directly provable.
The normalized one-sided access representation should retain this expression.

## Design conclusion

The stress test supports the planned architecture: one scope-aware memory/event
graph, one outer worker schedule, and local or static execution per exact event.
The failures are missing relation forms and compact lowering, not missing
model-specific schedules. The next compiler work should therefore prioritize:

1. cache-line isolation for independently updated keyed-event counters;
2. compact factored worker/task mappings;
3. multi-key event contribution lowering;
4. per-key expected counts for final-arrival execution;
5. ordinary affine-vector normalization; and
6. task-major source layouts for truly data-dependent grouped intermediates.
