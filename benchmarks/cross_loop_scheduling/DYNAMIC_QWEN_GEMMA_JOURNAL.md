# Dynamic Qwen3 and Gemma 4 scheduler validation

All CUDA commands in this investigation use physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 /home/eche/local/helion-cross-kernel/.venv/bin/python ...
```

The comparison preserves the checked-in pretuned math, fusion boundary,
numerics, and tile/resource configuration.  Probe-side source construction
changes the Helion decorator from `static_shapes=True` to
`static_shapes=False`, unifies equivalent batch extents under one backed
symbol, specializes invariant input layout metadata, and enables Triton's
`do_not_specialize` lowering.  The latter is required before claiming that one
cubin is reused across runtime sizes.

## 2026-09-09: initial admission audit

Neither dynamic kernel reached `build_static_pipeline_plan` at commit
`ecbcfbb8`.  Both failed earlier in
`cross_loop_codegen._root_task_orders()` with:

```text
InvalidConfig: cross_loop_schedule='static_pipeline' requires a representable
root PID task order
```

The exact rejected subset is simpler than the error suggests.  Every
parameterized root has configured PID axis order equal to its logical-domain
axis order, and every case uses `PersistentBlockedProgramIDs`; no effective L2
grouping is present.  The sole rejecting condition is the existing
`len(pid_axis_order) != 1` guard for parameterized domains.

Gemma's first rejected root is router projection:

```text
root 0: axes/order (0, 1), counts (B, 16), blocks (1, 8)
```

Its complete parameterized rank sequence is `2, 2, 1, 3, 2, 3, 2, 1`.

Qwen's first rejected root in the B2 exemplar is activation/FP8 quantization:

```text
root 12: axes/order (35, 36), counts (B, ceildiv(I, 16)), blocks (1, 16)
```

Roots 13 and 14 are also rank two.  Roots 0--11 were specialized to the B2
max-domain during tracing even under `static_shapes=False`; direct B1 replay
may be valid through bounds masking, but no broader batch-polymorphism claim
is justified until generated code is inspected and B1/B2 correctness passes.

No Triton kernel was emitted in this checkpoint, so there are no latency,
resource, or cubin-reuse results yet.

## 2026-09-09: canonical multi-axis admission and Qwen provenance

After the generic compiler accepted canonical multi-axis PID orders, Gemma
lowered directly.  Qwen additionally needed its configured split-attention
and O-projection axis permutations admitted.  Its first attempted dynamic
source still specialized early batch dimensions and produced an illegal B1
replay from a B2 exemplar.  The cause was not scheduler masking: independent
batch-shaped mutable buffers acquired unrelated shape symbols and fake-tensor
operations specialized them while resolving equality/broadcast constraints.

The minimal source-level solution is to use `context_lens.size(0)` as the one
authoritative backed batch symbol and assign every stage's existing
`*_num_tokens`/`*_m` variable from it.  This does not add work or max-domain
masking.  The two attention-merge dimensions containing batch must also not be
passed to `hl.specialize`.  The resulting 15 root sizes are:

```text
32B, 32B, 768B, 10B, 8B, 1024B, 512B, 32B, 32B, 512B,
32B, 32B, 1536B, 96B, 512B
```

All depend on the same symbol and no other runtime symbol.  Invariant model
sizes and all contiguous strides are specialized explicitly.  This both
preserves one binary across B1/B2 and avoids turning fixed layout information
into runtime arithmetic.

Qwen results on physical GPU 0, 20--30 cold-L2 samples after a 1 s warmup:

| Variant | B1 S8192 | B2 S2048/S8192 | binaries | resources |
| --- | ---: | ---: | ---: | --- |
| all sizes/strides non-specialized | 464.83 us | 579.41 us | 1 | R255, 22 B spill, 2,048 B shared |
| ordinary Triton specialization | 147.44 us | 186.32 us | 2 | R255, 2 B spill, 17,408 B shared |
| batch-only dynamic, invariant layout specialized | **170.06 us** | **186.34 us** | **1** | R255, 8 B spill, 17,408 B shared |

The final cubin SHA256 is
`f319dda85776237d6ebc5c4818f957e6050cc210b4f6362a48924a0053165c66`.
The dynamic schedule is a 15-segment parameterized root-major schedule;
event-frontier recurrence declines, no readiness waits lower, and root
barriers remain.  B2 has effectively zero generic-binary cost versus ordinary
Triton specialization, while B1 pays about 23 us for retaining a generic
batch path.  The much larger gap from the established approximately 94.05-us
pretuned B1 control is therefore the loss of the static continuation and
fine-grained synchronization plan, not merely runtime stride arithmetic.

The same-source static ragged probe fails on clean `ecbcfbb8` as well as the
working tree: root 12's retained local schedule has one 1,536-task segment,
but neither `_root_task_placement_relation` nor `_root_schedule_traversal`
constructs its certificate.  Disabling the global proposal does not change
that failure, so it is not attributed to the multi-axis patch.

## 2026-09-09: initial Gemma result

Gemma successfully compiles one dynamic cubin and replays B2 -> B1 -> B2 with
runtime-sized outputs.  It selects the eight-segment parameterized root-major
schedule; event-frontier recurrence declines.  Both batches route to distinct
experts (B2 has 15 distinct experts across 16 assignments).

Initial `m3/W4/R128` cold-L2 medians (20 samples, 1 s warmup):

| Batch | Dynamic persistent | Static persistent | Matched 8-launch standalone |
| --- | ---: | ---: | ---: |
| B1 | 137.02 us | 51.23 us | 55.15 us |
| B2 | 202.62 us | 73.58 us | 69.47 us |

Correctness passes the production tolerances.  The dynamic cubin SHA256 is
`985892c6acd486d65483d2c1911f67698596687325832bc2cf1434e51ccc1480`.
Its R128/72-byte-spill/512-byte-shared resource profile is qualitatively worse
than the static cubins (zero spill and 34,816 bytes shared), making resource
specialization the first tuning target rather than schedule order.

### Exact readiness-event audit

The dynamic dependency graph itself is not missing edges.  Qwen retains all
26 edges and 45 obligations, while Gemma retains all 11 edges and 14
obligations.  The loss happens when those obligations are quotiented into
readiness events:

- Qwen's dynamic graph yields 14 scalar-key events.  Each has key shape `()`,
  one producer, `fan_in=None`, and no producer publication/converse relation.
  All consumers are otherwise canonical and total.  Every event therefore
  fails `_parameterized_uniform_counter_fan_in`.
- Gemma's dynamic graph yields seven events with the same scalar/P1/FNone
  producer property.  Their consumer counts are `1, 1, 1, 2, 1, 1, 1` for
  roots `0->1`, `1->2`, `2->3`, `3->{4,5}`, `4->5`, `5->6`, and `6->7`.

The static B2 graphs prove that scalar barriers are not inherent.  Qwen has 17
fine events before its seven whole-root events, including rank-two counters
with fan-ins 8, 17 (two producer roots), and 16, as well as rank-two nested
consumers.  Gemma has rank-one `(2)/F1`, rank-two `(2,11)/F32`, and rank-one
`(2)/F11` events before its five scalar barriers.  Merely admitting rank-two
key tensors is therefore insufficient: parameterized event construction must
preserve the producer-set quotient/publication relation, multi-producer and
nested-consumer cases, and fan-in-one continuations.

Post-patch instrumentation sharpened where the coarsening occurs.  Generalized
multi-axis quotient/converse code is not reached by either workload:

- Qwen code generation invokes `producer_set_quotient` zero times.
- Gemma instantiates 14 symbolic dependency records, but all 14 already have
  `producers_by_consumer=None`; source-axis analysis and quotienting therefore
  both have zero calls.

The common gate is `TileAccess.layout_is_static`.  With `static_shapes=False`,
`device_ir_analysis.tile_accesses()` marks an access static only when every
shape, stride, and storage-offset value has Python `int` type.  A symbolic
leading batch extent consequently invalidates the whole access layout even
when contiguous strides and storage offset are compile-time invariants.  The
same routine then records `tensor_shape` through `env.size_hint`, losing the
symbolic extent needed by a later proof.  Dynamic fine-grained readiness thus
requires preserving symbolic extents in the existing `TileAccess` facts and
separating a symbolic shape from a genuinely unknown layout before the new
quotient proof can have any effect.

### Symbolic-layout result

After `TileAccess` retained backed symbolic shape/stride/offset expressions,
Gemma recovered the same fine event geometry as static B2: `(B)/F1` from root
2 to 3, `(B, 11)/F32` from root 5 to 6, and `(B)/F11` from root 6 to 7.
Finalization retains the latter two and makes root 7 a final-arrival
continuation; barriers remain only through root 5.

Qwen recovers 13 of 45 exact dependency records, including the FFN tail's
`(B, 96)/F16` and `(B)/F96` counters.  Attention roots 5 through 8 remain
whole-root-barriered because their masked/conditional access relations are not
yet exact.  Consequently its cold-L2 result is unchanged within noise:

| Shape | Before symbolic layouts | After symbolic layouts |
| --- | ---: | ---: |
| B1, S8192 | 170.03 us | 169.82 us |
| B2, S2048/S8192 | 186.35 us | 188.22 us |

The post-patch Qwen compile takes 85.46 s, reuses cubin SHA256
`6da52d448ddc136313dbfce4cbc398674509be2fa7f9d177cc6a396f7489a083`
across B2 -> B1 -> B2, and uses R255, 8 bytes of spill, 17,408 bytes shared,
and one warp.  A reset B2 replay is bit-exact.

Gemma's counters lower correctly, but global Triton `do_not_specialize` still
dominates.  At multiplier two its one-cubin path takes 169.94 us (B1) and
239.58 us (B2), versus matched standalone at 55.28 and 69.60 us.  Compilation
takes 22.90 s; cubin SHA256 is
`69be05ba21e31a6c4b7c8d28789376fa9eab5b36076415d6908fb5d843cb015f`;
resources are R128, 2 bytes of spill, 4,096 bytes shared, and four warps.  B1
is bit-exact against standalone; B2 passes production tolerances with maximum
absolute error 0.001953125.

### Gemma occupancy and specialization isolation

With ordinary Triton value/layout specialization, a multiplier sweep gives:

| multiplier | B1 | B2 | resources |
| ---: | ---: | ---: | --- |
| 1 | 65.50 us | 92.11 us | R128, no spill, 34,816 B shared |
| 2 | **63.46 us** | **83.90 us** | R128, no spill, 34,816 B shared |
| 3 | 71.62 us | 92.10 us | R128, no spill, 34,816 B shared |
| 4 | 88.00 us | 112.58 us | R128, no spill, 34,816 B shared |

Multiplier six is invalid: it requests 888 concurrently resident programs,
above the 592-program capacity of this compiled kernel.  The best multiplier
two control is within 8.3 us (B1) and 14.4 us (B2) of the matched standalone
results, so the remaining specialized gap is compatible with the retained
whole-root barriers rather than a gross resource mismatch.

The global `triton_do_not_specialize=True` path marks 40 strides of internally
allocated contiguous buffers, `eps`, and two equivalent batch expressions as
non-specialized.  The strides and `eps` are invariant across B1/B2, but neither
can currently be published through source: `hl.specialize(eps)` raises
`SpecializeArgType`, while `hl.specialize(router_logits.stride(0))` fails type
propagation because the internal allocation has `SourceOrigin` rather than an
input-tensor source expression.  Both failed source experiments were removed.
Selective Triton specialization of only batch (or compiler-side constant
propagation for internal contiguous strides) is needed to close the remaining
one-cubin resource gap without shape recompilation.

## Required result table

The completed run will record cold-L2 medians for each shape and all three
matched variants:

| Workload | Shape | Dynamic persistent | Static persistent | Standalone Helion |
| --- | --- | ---: | ---: | ---: |
| Qwen3 decode | B1, S8192 | 169.82 us | ~94.05 us historical pretuned | unavailable for ragged probe |
| Qwen3 decode | B2, S2048/S8192 | 188.22 us | unavailable: existing traversal-proof failure | unavailable for ragged probe |
| Gemma 4 A4B MoE | B1 | 169.94 us | 51.23 us | 55.28 us |
| Gemma 4 A4B MoE | B2 | 239.58 us | 73.58 us | 69.60 us |

For each dynamic callable the report must additionally include exact output
agreement, the parameterized schedule geometry selected by lowering, register,
spill, shared-memory, and warp counts, and an unchanged singleton cubin hash
after both runtime shapes execute.

## 2026-09-09: symbolic flattened attention chain

The remaining Qwen attention barriers were not caused by masks or unknown
storage layouts.  All accesses are unmasked and have exact symbolic contiguous
layouts.  The missing provenance was in the flattened consumer indices:

- roots 5 -> 6 write shaped partials but root 6 reads one-dimensional views;
- roots 6 -> 7 write shaped chunk outputs but root 7 reads one-dimensional
  views; and
- each flattened offset contains a host-backed symbolic factor
  `32 * batch`.

The FX ancestors otherwise use only the existing bounded affine subset:
fixed tile indices, static iotas, broadcast-only subscripts/loads, integer
addition, and multiplication.  The first unsupported node was the
`_get_symnode("32*attention_q_size0")` operand of `aten.mul.Tensor`, because
the evaluator previously accepted only a Python integer multiplier.  Keeping
that backed symbolic scalar in the existing affine-index representation
recovers all four flattened loads without adding a dependency abstraction.

The resulting exact and emitted attention counters are:

| edge | readiness-key domain | fan-in |
| --- | --- | ---: |
| root 5 -> 6 | `(16, 8B)` | 8 |
| root 6 -> 7 | `(32B)` | 16 |
| root 7 -> 8 | `(32B)` | 1 |

All three former root barriers disappear.  The whole kernel still uses the
15-segment parameterized root-major schedule because barriers elsewhere in the
layer prevent the closed event-frontier recurrence.  Generated Triton contains
readiness waits plus atomic add/exchange publication.

On physical GPU 0, multiplier 8, request-major attention order, 40 cold-L2
samples after 2 seconds of thermal warmup:

| Shape | before attention proof | exact attention counters |
| --- | ---: | ---: |
| B1, S8192 | 167.87 us | **159.65 us** |
| B2, S2048/S8192 | 183.74 us | **176.10 us** |

Compilation took 82.30 seconds.  The same cubin was present before and after
B2 -> B1 -> B2 replay (SHA256
`ec047a656267461969a42ab25464dae40806754fac086c3dbb06e5cffbd96ebd`).
Resources are R255, 6 bytes of spill, 17,408 bytes of shared memory, and one
warp.  Reset repeated executions were bit-exact and every output was finite at
both sizes.  This is a scheduler-proof validation; the stronger numerical
comparison remains the previously established matched baseline check.

### Final stable-tree validation after deterministic relation normalization

The final compiler snapshot (source SHA256
`4de88673654dc83bbcbb5f0e2aa675001c7a6ff3db56e1a738b66515aa0e8c36`)
was compiled and measured on physical GPU 0. Compilation took 84.34 seconds.
The emitted parameterized root-major schedule has 15 segments and retains the
same exact attention chain:

| edge | readiness-key domain | fan-in |
| --- | --- | ---: |
| root 5 -> 6 | `(16, 8B)` | 8 |
| root 6 -> 7 | `(32B)` | 16 |
| root 7 -> 8 | `(32B)` | 1 |

B2 -> B1 -> B2 replay retained one cubin (SHA256
`f0e15318562efecefa4f6578b83bea7fc3c6aa663452d33c71a68db17cdad44c`),
with R255, 6 bytes of spill, 17,408 bytes of shared memory, and one warp.
Every output was finite and deterministic. More strongly, the ragged B2 run
matched two independent B1 executions bit-for-bit across all 16 returned
tensors (maximum absolute difference zero).

Cold-L2 medians over 40 samples after a 2-second warmup:

| Shape | Dynamic persistent |
| --- | ---: |
| B1, S8192 | **157.74 us** |
| B2, S2048/S8192 | **176.06 us** |

The deterministic target-box normalization preserves exact adjacency-only
merging and returns the exact unmerged relation when its structural comparison
budget is exhausted. Focused coalescing, flattened-access, and Qwen relation
tests pass (9/9); the full tile-dependency and scheduler suites pass (212 tests
plus 35 subtests).

## 2026-09-09: final Gemma B1/B2 validation and internal-layout fix

The remaining large Gemma regression was not caused by a missing MoE-tail
dependency.  The final symbolic graph emits the expected two exact counters:

| edge | readiness-key domain | fan-in | lowering |
| --- | --- | ---: | --- |
| root 5 -> 6 | `(B, 11)` | 32 | ordinary readiness counter |
| root 6 -> 7 | `(B)` | 11 | final-arrival continuation |

The remaining root barriers are exactly `0->1`, `1->2`, `2->3`, `3->4`, and
`4->5`.  The parameterized root-major schedule contains seven segments for
roots 0 through 6; root 7 is continuation-owned.  Event-frontier scheduling
declines, as expected for this mixed barrier/counter graph.

The actual bottleneck was a generic dynamic-layout codegen issue.  Although
all intermediate buffers are allocated by the generated host wrapper with
fixed contiguous layouts, their 25 concrete stride values were passed to
Triton as fully generic runtime scalars.  With
`triton_do_not_specialize=True`, this disabled the same staging optimization
used by the static kernel:

| one-cubin dynamic path | registers | spills | shared | B1 | B2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 128 | 2 B | 4,096 B | 124.98 us | 192.27 us |
| internal integer strides literal | 128 | 0 B | 34,816 B | 67.55 us | 83.94 us |

`DeviceFunction.tensor_stride` now emits a `StaticShape` only when the stride
is a Python integer and the tensor is neither a replayable input nor present
in the input-source map.  User/aliased input strides and symbolically varying
internal strides remain runtime arguments.  The generated do-not-specialize
list consequently shrinks from 40 intermediate-stride entries plus the three
real runtime symbols to only `router_project_m`, `eps`, and `expert_geglu_m`.
This is an ownership/layout invariant, not a Gemma-specific rule.

Focused validation:

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .../python -m pytest -q \
  test/test_indexing.py::TestIndexing::test_dynamic_internal_strides_remain_literal \
  test/test_indexing.py::TestIndexing::test_symbolic_internal_stride_remains_runtime \
  test/test_indexing.py::TestIndexing::test_triton_do_not_specialize_emits_do_not_specialize
3 passed
```

The dynamic multiplier sweep, with the fixed source/config and one cubin per
row, was:

| `num_sm_multiplier` | B1 | B2 |
| ---: | ---: | ---: |
| 1 | 67.62 us | 94.18 us |
| **2** | **63.46 us** | **81.89 us** |
| 3 | 67.55 us | 83.94 us |
| 4 | 75.74 us | 100.24 us |

Two and eight warps were both worse than four (67.47/92.03 us and
94.10/110.46 us respectively).  The retained configuration is therefore the
pretuned arithmetic configuration with `num_sm_multiplier=2`, four warps,
`maxnreg=128`, gate range stages 3, and down range stages 5.
Raising the register cap to 256 lets PTXAS use 156 registers but is neutral
(63.36/81.81 us), confirming that register pressure is no longer the limiter.
Cold-L2 per-root diagnostics identify gate-up and down as the expected heavy
roots (B1: 22.40/14.29 us; B2: 34.66/22.43 us), and the recovered persistent
kernel now has the same shared-memory staging as those standalone roots.

Authoritative matched cold-L2 command on physical GPU 0:

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .../python \
  benchmarks/cross_loop_scheduling/gemma4_a4b_moe_dynamic.py \
  --batch-sizes 1,2 --exemplar-batch 2 --multiplier 2 \
  --static-multiplier 3 --repetitions 100 --warmup-ms 5000
```

| shape | dynamic persistent m2 | static persistent m3 | matched 8-launch standalone |
| --- | ---: | ---: | ---: |
| B1 | **65.472 us** | 51.296 us | 55.264 us |
| B2 | **81.168 us** | 73.664 us | 67.712 us |

The dynamic B2 -> B1 replay retains exactly one cubin: Triton hash
`ba1842e458a8db711b95749722999a2c0a0a1e20dc2181d8710042d57fb26476`,
SHA256 `e5e4f64f0817f9b55b7f4014beb2ace1a2dca48732fd9f53595786d25dc402c4`.
Compilation took 22.43 seconds.  Resources are R128, zero spill, 34,816 bytes
shared, and four warps.  All seven returned tensors are bit-exact against both
the same-shape static kernel and the matched standalone pipeline at B1 and B2.
B2 routes 16 assignments to 15 distinct experts, so this is not a degenerate
same-expert case.

The remaining 10-14 us gap is now scheduling overhead, not constituent-kernel
resource loss.  At multiplier two every parameterized root segment admits the
fixed 296-worker cohort.  The five barriers therefore receive 1,480 arrivals
per launch even though the corresponding static B2 active widths are only
`32 + 8 + 2 + 296 + 96 = 434`.  A runtime `min(task_count, worker_count)`
cohort cannot use the current `epoch * arrival_count` replay protocol because
the arrival count changes between B1 and B2.  The general follow-up is bounded
epoch framing for root barriers (the same principle already used for dynamic
readiness counters), after which parametric segments can use runtime-minimal
worker cohorts without recompilation.

## 2026-09-09: Qwen causal synchronization ablations

The current-tree exact-shape control is now compileable after repairing the
symbolic inverse for its reflected mixed-radix root-12 placement. These are
same-source ragged kernels, request-major attention order, multiplier eight,
and unchanged FP8 arithmetic/fusion boundaries. Representative matched
cold-L2 results on physical GPU 0 are:

| path | B1, S8192 | B2, S2048/S8192 | resources |
| --- | ---: | ---: | --- |
| exact-shape persistent | 104.35 us | 128.90 us | B1 R255/10 B spill; B2 R255/12 B spill; 17,408 B shared, W1 |
| production one-cubin dynamic | 128.86 us | 155.65 us | R255/12 B spill/17,408 B/W1 |
| dynamic + supported intermediate continuations | 129.55 us | 157.73 us | R255/12 B spill/17,408 B/W1 |
| dynamic + weighted active-participant barriers | **108.45 us** | **137.09 us** | R255/8 B spill/17,408 B/W1 |
| dynamic + bounded-epoch unit barriers | **108.42 us** | **139.02 us** | R255/8 B spill/17,408 B/W1 |

Every dynamic variant reused one cubin across B1 and B2 and was bit-exact
against its same-shape static control across all 16 outputs. Both
active-participant variants additionally exercised an explicit B2 -> B1 -> B2
replay sequence. The production dynamic cubin SHA256 was
`2506e4d6eba4ab366297be4834996dd44af26a54c29390c490288814f2623f70`;
the weighted and bounded-unit diagnostic cubins were respectively
`6f2f59e9fa48ac047fa6cde9f7f950673f96ca06365ae40b5c3bc15595e38647`
and `15c6d5de129fa0cb48bb2d6b3a6d669743b708f9cd690aa39d0a0d0453413f7c`.

The synchronization plans explain the gap. Static has seven root-barrier
edges and four final-arrival continuations (`6->7`, `7->8`, `(0,9)->10`, and
`12->13`). Production dynamic has zero continuations and ten root-barrier
edges. More importantly, each of nine dynamic barrier-producing roots makes
all 1,184 resident workers wait and publish even when only a few workers own
tasks: 10,656 B1 root-barrier arrivals versus roughly 914 publications in the
static schedule.

Selecting the two intermediate continuations already carrying complete
parameterized certificates (`6->7` and `12->13`, both fan-in 16) removes roots
7 and 13 from the resident schedule but leaves the ten barriers and their
10,656 publications unchanged. Its small regression rules this admission
policy out as the primary cause.

The active-participant diagnostic derives participation directly from the
existing parameterized root-major `WorkerScheduleSegment.task_order`. For
fixed worker count `W`, symbolic first slot `F`, symbolic task count `T`, and
worker `w`, define:

```text
j = (w + W - (F mod W)) mod W
A = min(W, T)
```

Exactly the workers with `j < A` own at least one root task. The first
diagnostic lets only those workers wait/publish and gives worker `j` weight
`floor(W/A) + [j < W mod A]`, whose sum is the shape-stable value `W`. The
cleaner protocol uses uint64 counters with static epoch stride `W`: each active
worker first applies `atomic_max(epoch*W)`, contributes one, and consumers wait
for `epoch*W + A`. A positive sentinel generalizes this to `T=0` by using
`max(A,1)` participants; Qwen also carries an explicit `B >= 1` host guard.
Host-side checks cover both wrapped cohorts (`F mod W + A > W`) and multi-wave
roots. Both protocols preserve the same 15 root-major segments, five exact
readiness counters, ten barrier edges, and kernel bodies. Dynamic logical
barrier arrivals fall from 10,656 to 978 at B1 and 1,604 at B2. The comparable
seven-producer static plans use 914 and 1,476 arrivals; the remaining 64/128
arrivals are exactly the dynamic plan's additional root-8 and root-9 barriers.

This recovers 16--20 us without changing placement, fusion, or numerics, so
the dominant regression is full-cohort barrier participation rather than
dynamic indexing, attention masking, or missing intermediate continuations.
The bounded-unit protocol is within 0.03 us of weighted arrivals at B1 and
about 1.94 us slower at B2, while avoiding large per-thread contributions and
supporting shape-changing replay directly.

Two secondary proof gaps remain independently measurable. The exact `7->8`
consumer map is row-major flattening from `(b,g) in [0,B)x[0,32)` to
`k=32*b+g in [0,32B)`: it is total and single-valued, but the relation algebra
does not yet derive the generic inverse `b=floor(k/32), g=Mod(k,32)`, so it
cannot become a continuation. For `8->9`, an exact fan-in-32 event exists but
does not cover every dynamic access obligation, leaving a coarse barrier that
subsumes it. The exact `(0,9)->10` fan-in-17 event is a valid continuation
candidate, but current parameterized counter admission explicitly permits only
one producer. None of these secondary gaps explains the roughly 20 us
recovered by active barrier participation.

The diagnostic commands were:

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .../python \
  benchmarks/cross_loop_scheduling/qwen3_decode_layer_dynamic.py \
  --request-major --multiplier 8 \
  --supported-intermediate-continuations \
  --repetitions 40 --warmup-ms 2000

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .../python \
  benchmarks/cross_loop_scheduling/qwen3_decode_layer_dynamic.py \
  --request-major --multiplier 8 --active-participant-barriers \
  --repetitions 60 --warmup-ms 3000

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .../python \
  benchmarks/cross_loop_scheduling/qwen3_decode_layer_dynamic.py \
  --request-major --multiplier 8 --bounded-unit-barriers \
  --repetitions 60 --warmup-ms 3000
```

Post-change symbolic validation is clean:

```text
test/test_tile_dependency.py + test/test_cross_loop_scheduler.py:
217 passed, 35 subtests passed

focused dynamic internal-stride/indexing tests:
3 passed
```

### Compiler conclusion

Active barrier support is a general property of the existing
`WorkerScheduleSegment.task_order`, not a Qwen heuristic. A production change
should extend `RootBarrierPublicationPlan` so that it remains the sole owner of
participant support and arrival count for both concrete and parameterized
schedules; codegen should consume that plan. The bounded uint64/unit-arrival
protocol is the preferred rendering. The weighted protocol and all root-ID
selection in this probe are diagnostic only and should not enter the compiler.

The current `build_static_pipeline_plan` branch on parameterized domains does
more than select a symbolic renderer. It also imposes different synchronization
policy: sink-only, fan-in-greater-than-one continuations; a single-producer
parameterized-counter gate; full-cohort root barriers; and a separate
event-frontier attempt. The end state should share readiness construction,
counter/continuation selection, coverage finalization, and barrier-publication
semantics. Concrete and symbolic cases may retain different certified
`WorkerSchedule` renderings, but those renderings must consume the same
semantic plan.

The generic pieces worth keeping are exact symbolic `TileAccess` layouts,
literal compiler-owned fixed internal strides, symbolic row-major
flatten/unflatten, and the reflected mixed-radix converse needed to prove the
ordinary Qwen root-12 task order. Potential duplicate truth to remove is
codegen recomputation of cumulative root offsets and parameterized barrier
participants when the recognized worker-schedule geometry already contains
`F` and `T`. Model-specific continuation admission, weighted arrivals, and AST
rewrites stay confined to this diagnostic probe.

## 2026-09-09: original checked-in Gemma B1 control on GPU 2

This control measures the unmodified checked-in source and AOT configuration,
not the dynamic-batch diagnostic. Both `gemma4_a4b_moe.py` and its SM100
config are byte-identical to local `main`. The config is the checked-in
multiplier-4 static pipeline with W4, R128, gate/down stages 3/5, and block
sizes `[8,16,256,128,64,64,256]`.

Balanced CUDA-graph timing clears L2 before every replay. Five 200-sample
rounds on the current compiler gave medians of **49.024 us** for the exact AOT
kernel and **53.168 us** for the matched eight-launch standalone control
(7.8% lower latency). Five 300-sample rounds on clean local main `f1d0c423`
gave **49.120 us** and **53.248 us**, respectively. Thus the redesign changes
this control by less than 0.1 us. All seven returned tensors were bit-exact for
seed 0. The route was `[52,14,44,125,121,25,37,45]`: eight assignments to eight
distinct experts.

The exact persistent resource profile is R128, zero spills, 34,816 bytes of
shared memory, and four warps. The matched standalone stages use:

| stage | registers | spill bytes | shared bytes | warps |
| --- | ---: | ---: | ---: | ---: |
| router | 48 | 0 | 128 | 4 |
| group top-k | 28 | 0 | 512 | 4 |
| global top-k | 30 | 0 | 512 | 4 |
| gate/up | 51 | 0 | 34,816 | 4 |
| GeGLU | 16 | 0 | 0 | 2 |
| down | 32 | 0 | 33,280 | 4 |
| expert reduction | 29 | 0 | 2,048 | 2 |
| post norm | 31 | 0 | 32 | 8 |

This source exercises the fully concrete/static scheduler branch: B=1 and all
root domains are specialized. With 148 SMs and multiplier 4, the plan has 592
workers and seven resident segments for roots `0,1,2,3,4,5,7`. Root 6 is not
resident: one exact root-5-to-root-6 counter has 11 readiness keys and fan-in
32, and each final arrival executes the corresponding reduction tile as a
continuation. Whole-root barriers remain on `0->1`, `1->2`, `2->3`, `3->4`,
`4->5`, and `6->7`; transient-source admission is not used. Lowering contains
counter atomic exchange/add and root-barrier code, with no parameterized or
event-frontier loop, dispatch ticket, or grid-barrier helper.

The global-list proposal is accepted, but changes only root 7's one-task
dispatch offset from 7 to 6. A paired 300-sample same-source ablation measured
51.296 us with that proposal and 51.136 us with it disabled, so there is no
meaningful list-scheduling gain here. The original B1 speedup instead comes
from the persistent boundary plus the established root-5-to-root-6
final-arrival continuation. Clean local main selects the same seven roots,
counter, continuation, and barrier set (its root-7 offset remains 7).

`gemma4_a4b_moe_batched.py` does not invent a different B1 persistent body:
its static constructor copies the checked-in function verbatim and replaces
only the AOT decorator with a non-autotuning kernel decorator. Its standalone
control mechanically emits the same eight roots as eight kernels, preserving
math, intermediate dtypes, and outputs while allowing independent resource
configs. Its dynamic mode is structurally different only in making batch a
runtime symbol and specializing invariant sizes/strides. Routing remains
data-dependent in all cases, but it changes weight addresses rather than the
static task topology: there are always eight expert-assignment lanes per token.

All commands for this section used `CUDA_VISIBLE_DEVICES=2`. No compiler or
pretuned-source changes were made.

## 2026-09-09: untouched pretuned Qwen3 production-source control

The checked-in source and AOT configuration were used without transformation
or override. A separate public-entry sanity run invoked
`qwen3_decode_layer(*args)` itself (rather than a rewritten kernel) and selected
the same cubin and resource envelope reported below; its 40-sample isolated
cold-L2 median was 100.096 us:

```text
pretuned_kernels/megakernels/qwen3_decode_layer/qwen3_decode_layer.py
  sha256 3bc49516f74b09100a2f3cca964c2204e5d304fecd0f08dd9eb68357152ad8dc
pretuned_kernels/megakernels/qwen3_decode_layer/_helion_aot_qwen3_decode_layer_cuda_sm100.py
  sha256 0c64239b383d6ae9b5c593d9c27f79e656bb2c8f6a882e18fbfd818555cd1987
```

This is the native B1/S8192/Q1, 128-attention-split, 15-root FP8 decode
kernel with the checked-in multiplier-eight static-pipeline config. It was
bound and compiled directly as `qwen3_decode_layer`; none of the source
rewrites in `qwen3_decode_layer_batched.py` were applied. The latter adds a
runtime batch symbol, `context_lens`, split guards and tail masks, invariant
layout specializations, and an optional request-major root-5 order. The
untouched source instead has fixed B=1/S=8192, no ragged branch, and root-5
order `[2,1,0]` (equivalent in its batch-one domain).

The authoritative comparison used physical GPU 6 after confirming no compute
processes, a 10-second thermal warmup, CUDA-graph replay, mutable residual/cache
reset before every observation, a verified 256 MiB L2 flush before every
timed replay, and 120 randomized/interleaved samples:

| implementation | cold-L2 median |
| --- | ---: |
| current-tree untouched persistent | **104.272 us** |
| tuned 12-launch standalone, optimized 32 attention splits | **100.160 us** |
| tuned 12-launch standalone, same 128 attention splits | **112.416 us** |
| clean local-main untouched persistent | **124.672 us** |
| clean local-main 32-split standalone | **100.016 us** |
| clean local-main 128-split standalone | **110.464 us** |

Thus current-tree persistent is 4.11 us (4.1%) behind the independently tuned
standalone optimum but 8.14 us (7.2%) ahead of the same-128-split standalone.
It is 20.40 us faster than the fresh clean-main persistent build under the
same interleaved protocol. Two isolated one-entry current-tree runs measured
100.096--100.256 us, but that is not the primary matched number: unlike the
interleaved comparison, they continuously keep only the persistent instruction
path hot.

Current-tree persistent resources are 255 registers/thread, zero spill bytes,
17,408 bytes shared memory, and one warp. Its cubin SHA256 is
`ddcb239fab6433af509178bcd5c3608ddbfa07cde5daf8952968cc76dc61732b`.
Clean local main uses 253 registers, zero spills, the same 17,408 bytes shared,
and one warp; its cubin SHA256 is
`03a51cd239ed4790a5f49903c5f6de73e8ecec06f1de163035d68d03d3aa762d`.
Standalone stage kernels use 16--164 registers at 32 splits and 16--95 at 128
splits, no spills, and independently chosen one/four/eight/16-warp launch
geometries; their largest shared allocations are 69,632 and 14,268 bytes at
32 and 128 splits respectively.

This kernel exercises only the concrete scheduler policy. The parameterized
root-major and parameterized event-frontier builders are not called. The
concrete global-list proposal is attempted but rejected, so the retained
schedule is the existing concrete worker schedule over 1,184 workers. It has
seven root barriers:

```text
0->1, 1->2, 2->3, 3->4, 4->5, 10->11, 11->12
```

It retains seven exact counter plans: `5->6` (128 keys, fan-in 8), `6->7`
(32, fan-in 16, continuation), `7->8` (32, fan-in 1, continuation),
`(0,9)->10` (32, fan-in 17, continuation), `12->13` (96, fan-in 16,
continuation), `8->9` (one key, fan-in 32), and `13->14` (one key, fan-in 96).
Roots 7, 8, 10, and 13 therefore execute as final-arrival continuations; the
resident segments are roots `0,1,2,3,4,5,6,9,11,12,14`. Lowered Triton has
ordinary readiness waits/adds and root barriers, with no parameterized-root,
event-frontier, dispatch-ticket, or grid-barrier loop.

The standalone controls use the established tuned 12-launch Helion boundary
and FP32 activation scales. The 32-split result is the optimized baseline; the
128-split result matches the persistent attention partition and reduction
order. Final output, residual, and KV-cache agree with the 128-split control at
the production tolerances (maximum absolute errors 0.05078125, 0.015625, and
0.015625 respectively). A still stricter helper that launches the exact
source-visible granular RMS body currently fails before timing because its
generated wrapper computes `_RDIM_SIZE_7` from `_BLOCK_SIZE_4` before defining
`_BLOCK_SIZE_4`. That is a separate standalone-lowering defect, not a
scheduler result, and was not patched for this measurement. The built-in vLLM
comparison could not be run because vLLM is not installed in the available
environment.

All measurements in this section used `CUDA_VISIBLE_DEVICES=6`; physical GPU
0 was occupied by unrelated autotuning processes. No compiler or pretuned
source was changed.

## 2026-09-09: resolving the historical 94.048 us Qwen result

The historical result was not a warm-L2 or clock artifact. The exact command
was recovered from rollout trace command 79. It compiled the untouched
pretuned source/config, captured only the persistent graph (resetting mutable
inputs during capture), ran `thermal_warmup(2000)`, then called
`bench_pre_captured_cudagraph(graph.replay, rep=50)`. That timer clears the
default 256 MiB Triton benchmark buffer before every replay. It does not reset
the residual/cache during timed observations.

The generated Triton file from that run still exists at:

```text
/tmp/torchinductor_eche/7t/c7taikrtls7p2yjvrzmphihxfjvtipksv3mqgeociurc2ihfidov.py
sha256 fcc085466b97dfac26b1cb1e741ee54d6687aa55db20623848a445a41ca81baa
146,140 bytes, 1,825 lines, mtime 2026-09-08 05:37:17 -0700
```

Replaying that exact generated module today on physical GPU 6 reproduced
94.080--94.224 us after clocks reached 1,845 MHz. Its current JIT
specialization used 254 registers/thread, zero spills, 17,408 bytes shared,
and one warp. One captured binary was PTX SHA256
`2c58e94412ecef222cad40a80796df1f1b93eef376a9ccfb07c13d5409cbcfa1`
and cubin SHA256
`fb707c073248d0ca9c211d78f94d83ac84d97d76b4072261c5cdef4d1802b7c3`.
The original 94.048 us trace did not print a cubin hash, so this is a fresh JIT
of the byte-exact historical Triton, not an unsupported claim about the
unrecorded original cubin.

Protocol controls rule out the obvious measurement explanations:

| experiment at 1,845 MHz | historical lowering | current untouched lowering |
| --- | ---: | ---: |
| isolated single graph, cold L2 | 94.080--94.224 us | 98.112--98.272 us |
| balanced two-kernel interleave, reset each sample | 98.272 us | 100.256 us |
| current interleaved with the full 12-launch standalone control | n/a | 104.272 us |

Resetting mutable inputs outside the timed interval changed the current median
by at most 0.16 us, and both kernels were measured at the same steady SM/memory
clocks. Interleaving large, different instruction paths does impose a real
instruction/cache-state penalty; that explains why 104.272 us is not directly
comparable to the historical one-entry 94.048 us. It does not explain the
whole difference: under identical isolated and two-way-interleaved protocols,
the older lowering remains approximately 4.0 us and 2.0 us faster,
respectively.

The source and AOT-config hashes are unchanged. The material lowering differs:
the historical program has nine root barriers and only the `7->8` and
`(0,9)->10` final-arrival continuations. In particular, `5->6` and `6->7`
remain coarse barriers and `12->13` remains a resident counter consumer. The
current program has seven root barriers, exact counters for `5->6` and `6->7`,
and continuations for roots 7, 8, 10, and 13. The historical/current generated
Triton and PTX diffs were retained during the investigation as
`/tmp/qwen_historical_vs_current.ptx.diff` and the corresponding generated
modules; the PTX diff is 20,972 lines. Thus the 94 us result was a real older
synchronization/code-generation result, while roughly four microseconds of the
104.272 us comparison is measurement-context overhead.

Module-level `_BLOCK_SIZE_* = tl.constexpr(1)` declaration order is
nondeterministic because `DeviceFunction.register_triton_outlined_helper`
iterates the `ReadWrites.reads` set and records those names in insertion order.
Multiple fresh compiles produced source files differing only in those
declarations while selecting the same cubin and performance envelope. This is
a reproducibility cleanup opportunity, not the performance regression.

### Production active-owner barrier smoke failure

After the production bounded-unit barrier implementation landed, the fresh
dynamic B2 request-major/multiplier-eight smoke compiled but hung in its first
kernel launch. It was terminated after more than seven minutes at 100% GPU
utilization; consequently it produced no valid B1/B2 timing.

The failure is an exact lowering mismatch, not a schedule-graph cycle.
`RootBarrierPublicationPlan.participant_order` proves Euclidean modulo support,
but production generated Triton renders a wrapped local worker ordinal as
`(pid + -offset) % W`. PTX confirms signed remainder semantics. The successful
probe-only bounded-unit implementation instead emitted
`(pid + W - (offset % W)) % W`.

For B2, root 2 starts at worker slot 128 and has 1,536 tasks, so all 1,184
workers must participate and publish. The production predicate rejects workers
0--127, allowing only 1,056 publications, while root 3 waits for
`epoch * 1184 + 1184`; the target is unreachable. This is the first hang. The
same defect would drop the wrapped part of B1 root 5 (first slot 850, 1,024
active owners) before the exact `5->6` readiness handoff. Barrier addresses,
uint64 epoch framing, unit publication sites, and target counts otherwise
match the successful diagnostic. The required generic repair is to preserve
nonnegative modulo semantics when rendering the certified participant
relation; no workload-specific scheduling rule is involved.

The generic Euclidean-modulo renderer repair was then validated on GPU 6.
Compile-only inspection showed every potentially wrapped membership predicate
as `(((x % W) + W) % W) < A`; a B2 launch subsequently completed. Under a
10-second warmup and 120 balanced cold-L2 observations per pair, the production
parameterized kernel produced:

| shape | same-source static | one-cubin dynamic |
| --- | ---: | ---: |
| B1, S8192 | 106.368 us | 109.456 us |
| B2, S[2048,8192] | 126.976 us | 133.024 us |

The dynamic results are bit-exact with each corresponding same-source static
kernel and reuse one cubin across B1 and B2 (SHA256
`ea6ee9aa15c6844da786b0a9c1f40c77cb43211b056ca043b0a5a9f5440bca16`).
The dynamic resource envelope is 255 registers/thread, 10 spill bytes, 17,408
bytes shared, and one warp; static B1/B2 use the same registers/shared/warp
counts with 10/12 spill bytes. The retained parameterized plan is one
15-segment root-major schedule with five counters (`5->6`, `6->7`, `7->8`,
`12->13`, `13->14`) and ten barriers (`0->1`, `0->10`, `1->2`, `2->3`,
`3->4`, `4->5`, `8->9`, `9->10`, `10->11`, `11->12`).

A separate strict B1 comparison fed identical tensors to the untouched
pretuned static source and the dynamic ragged source, interleaved the two CUDA
graphs, reset mutable state before every observation, and used the same
10-second warmup/120-sample/256-MiB-flush protocol. Untouched measured
100.160 us and dynamic measured 106.368 us; outputs and residual were
bit-exact. The untouched cubin is
`ddcb239fab6433af509178bcd5c3608ddbfa07cde5daf8952968cc76dc61732b`
at R255/zero spills/17,408 B/W1, versus the dynamic cubin above at R255/10 B
spills/17,408 B/W1. This isolates a 6.208 us parameterization/source-lowering
gap at B1 after correctness was restored.

The fresh dynamic compilation used for the fixed run took **111.517 s**.  This
is cold compile wall time rather than kernel latency; B1 and B2 then selected
the single cubin hash above without recompilation.

### Untouched-Qwen synchronization ablation

The following probe changes only compiler-selected synchronization for the
unchanged checked-in B1/S8192/Q1 source and AOT config:

```text
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/eche/local/helion-scheduler-redesign \
  /home/eche/local/helion-clc/.venv/bin/python -u \
  benchmarks/cross_loop_scheduling/qwen3_decode_layer_sync_ablation.py \
  --repetitions 120 --warmup-ms 10000
```

Every variant cloned one common input set.  This detail is essential because
the production helper deliberately allocates `cos_sin` and several scratch
tensors uninitialized pending vLLM setup; independently regenerated inputs are
not a correctness reference.  Output, residual, and the written KV-cache slot
were bit-exact across the common-input variants.  Each reported isolated
median uses a ten-second thermal warmup, 120 observations, reset outside the
timed interval, and a 256 MiB L2 flush before every replay.

| synchronization variant | barriers/counter change | isolated cold-L2 | paired with current |
| --- | --- | ---: | ---: |
| current | exact `5->6`; continuations `6->7`, `12->13` | 100.336 us | -- |
| `5->6` counter disabled | add `5->6` root barrier | 104.224 us | 104.256 vs 100.192 us |
| `6->7` continuation disabled | retain ordinary exact 32-key/fan-in-16 counter and resident root 7 | 100.176 us | 100.416 vs 100.224 us |
| `6->7` counter also disabled | add `6->7` root barrier | 104.160 us | 104.384 vs 100.176 us |
| `12->13` continuation disabled | retain ordinary exact 96-key/fan-in-16 counter and resident root 13 | 100.432 us | 102.176 vs 100.160 us |
| historical synchronization topology | barriers `5->6`,`6->7`; resident counter consumer 13 | 100.128 us | 102.208 vs 100.160 us |

All variants use R255, 17,408 bytes shared, and one warp.  Current,
`5->6`-barrier, no-`6->7`-continuation, and no-`12->13`-continuation spill zero
bytes; the `6->7`-barrier and full historical-topology recompiles spill two
bytes.  The two 120-sample run medians agreed within 0.18 us for each isolated
variant.

The result rules out the newly selected synchronization as the historical
94-to-100 us regression.  Exact `5->6` is worth about four microseconds, and
coarsening `6->7` all the way to a barrier also costs about four microseconds.
Executing root 7 as a continuation rather than an ordinary exact-counter
consumer is neutral within roughly 0.2 us.  Root 13 continuation is neutral in
the isolated deployment-like protocol and saves roughly two microseconds when
interleaved with another large cubin.  Most importantly, reproducing the old
nine-barrier/two-continuation topology with the current compiler still runs at
100.128 us, not 94 us.  The remaining gap is therefore in another generated
code difference in the historical compiler/cubin, not in these three
synchronization decisions.

The graph-structural conclusion is also consistent with event-frontier
scheduling.  Retain an exact event when it releases downstream critical work;
both the 128-key `5->6` reduction frontier and the 32-key `6->7` frontier meet
that criterion, and their barrier ablations lose materially.  Final-arrival
continuation is an execution-placement optimization, not a different
dependency: when it does not shorten the unit-weight critical path or remove a
meaningful tail (roots 7 and 13 here), either rendering is effectively tied.
No root ID, model name, measured task latency, or resource-cost exception is
needed to explain these outcomes.

## 2026-09-09: Gemma production active-owner validation on GPU 2

The dynamic Gemma probe was rerun at `a9df42d1` with B2 as the compile
exemplar and the same callable replayed B2 -> B1 -> B2.  Timing used a
10-second thermal warmup and 120 balanced CUDA-graph observations, with a
256-MiB L2 clear before every replay.  Physical GPU 2 was idle before the run.

```text
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  /home/eche/local/helion-cross-kernel/.venv/bin/python \
  benchmarks/cross_loop_scheduling/gemma4_a4b_moe_dynamic.py \
  --batch-sizes 1,2 --exemplar-batch 2 --multiplier 2 \
  --static-multiplier 3 --repetitions 120 --warmup-ms 10000
```

| shape | one-cubin dynamic m2 | same-source static m3 | matched 8-launch standalone |
| --- | ---: | ---: | ---: |
| B1 | **141.152 us** | 51.040 us | 55.168 us |
| B2 | **190.432 us** | 71.552 us | 69.504 us |

The dynamic callable retained one specialization.  Its Triton hash is
`054d18744c10d32d82e49e27cc6cfa1ebe8986b877eb0a50f5a5fa2ee84e772d`
and its cubin SHA256 is
`fb616b3bdb730ef5bdc2335a1d46a899dcbb73f5871cf4235189dac0b6b6d2ce`.
Compilation took 29.113 seconds.  The dynamic resource envelope is R128,
12 spill bytes, 4,096 bytes shared, and four warps.  The same-source static B1
and B2 controls compiled in 2.072 and 12.019 seconds and used respectively
R128 and R116, zero spills, 34,816 bytes shared, and four warps.

The selected dynamic path is the seven-segment parameterized root-major
schedule over resident roots 0 through 6; event-frontier scheduling was
attempted and declined.  It retains the `(B,11)`/fan-in-32 `5->6` counter and
the `(B)`/fan-in-11 `6->7` final-arrival continuation.  Whole-root barriers
remain exactly `0->1`, `1->2`, `2->3`, `3->4`, and `4->5`.

B1 is bit-exact against both controls across all seven outputs.  At B2 the
same-source static control remains bit-exact with standalone, while dynamic
passes the production tolerances with maximum absolute errors by output of
`[0.001953125, 0, 0, 0, 7.6293945e-06, 6.1035156e-05,
3.0517578e-05]`; routing IDs are exact.  The B2 tokens select 15 distinct
experts across 16 assignments, with different expert sets, so the replay
exercises data-dependent routing rather than a degenerate route.

This checkpoint therefore restores forward progress after the signed-modulo
hang, but it does not retain the earlier dynamic shared-memory staging
envelope.  The observed R128/12-B-spill/4,096-B-shared binary explains why the
active-owner result remains far above the static controls; no compiler change
was made as part of this measurement.

As a regression control, the untouched checked-in B1 source and AOT config
were also compiled from a clean detached `a9df42d1` worktree and timed on
physical GPU 2 with the same 10-second/120-sample cold-L2 protocol.  The
pretuned persistent kernel measured **49.120 us**, versus **55.200 us** for
the matched eight-launch standalone pipeline.  All seven outputs were
bit-exact, all eight assignments selected distinct experts, compilation took
2.076 seconds, and resources remained R128, zero spills, 34,816 bytes shared,
and four warps.  Its cubin SHA256 was
`5ee97ae96304e90663dbebea1f5db272c36dbfccbbe2c842b9427e01513bce45`.

## 2026-09-09: Gemma root-6 continuation ablation on GPU 2

A probe-only option changes exactly one scheduling decision in the dynamic
Gemma plan: the existing exact `(B,11)`/fan-in-32 `5->6` counter executes root
6 as its final-arrival continuation. The pre-existing `(B)`/fan-in-11 `6->7`
continuation remains selected, and the parameterized root-major worker
schedule is rebuilt without roots 6 and 7. Arithmetic, routing, fusion,
configuration, and source-visible kernel bodies are unchanged.

The comparison used clean detached `a9df42d1` plus the positive
specialized-input-layout provenance change. Physical GPU 2 was idle. Both
batches used B2 compilation followed by B1 and B2 replay, a 10-second thermal
warmup, and 120 balanced observations of all four CUDA graphs with a 256-MiB
L2 clear before every replay.

| shape | baseline dynamic m2 | root-6 continuation m2 | same-source static m3 | matched standalone |
| --- | ---: | ---: | ---: | ---: |
| B1 | **61.312 us** | 65.408 us | 51.040 us | 55.168 us |
| B2 | **77.776 us** | 85.856 us | 73.600 us | 69.504 us |

Forcing the intermediate continuation loses 4.096 us (6.7%) at B1 and
8.080 us (10.4%) at B2. It therefore does not close the residual dynamic gap.
Both dynamic variants use R126, zero spills, 34,816 bytes shared, and four
warps, so the loss is not a resource-envelope change.

The baseline and ablation each retain one cubin across B2 -> B1 -> B2. Their
SHA256 values are respectively
`7d07ab14638895f21c8e5ecec8df631123148e1f2b72c161d0485bcf9085f3e8`
and
`71433cc294559193b7b2b6192083174b9cfb2c607715d20833afc440162808cc`.
Compile times were 28.689 and 27.475 seconds; static B1/B2 compiled in 2.058
and 11.433 seconds.

All seven outputs are bit-exact among baseline dynamic, forced continuation,
same-source static, and standalone at both batch sizes. B2 routes 16
assignments to 15 distinct experts with different per-token expert sets.

Generated Triton proves the intended nested ownership. In
`/tmp/torchinductor_eche/xf/cxf5xplkbpr4rmzwc374eqis3ttrrgkv4aafznmr3gdkn3fcc5f5.py`
(source SHA256
`b97b77ad417c791666c2dff848225b9ce3132abc000b96b23b30d4dd9442e97b`),
the root-5 scheduled helper's fan-in-32 final-arrival branch calls root 6,
publishes root 6's fan-in-11 arrival, and then calls root 7 from the nested
final-arrival branch. The entry function contains no resident root-6 call and
the module contains no readiness-wait loop. The ablated worker schedule has
six resident segments, roots 0 through 5, while both counter plans have
`continuation_consumer_index=0`; the five root barriers are unchanged.

### In-progress Stage-1a compile regression

An initial attempt against the concurrently modified Stage-1a workspace did
not reach the ablation. The baseline dynamic compile was interrupted after
more than five minutes with scheduler file SHA256
`8db6e4dae6795beb4d54087271f8c8a24f0ddd2b0f0bc73504df1cb7c751ef65`.
Its stack was
`root_barrier_publication_plan ->
_parametric_root_major_schedule_geometry ->
_parametric_root_major_schedule_geometry_from_parts ->
_parametric_root_major_relation`, spending the time in
`sympy.simplify(end - begin)` while reconstructing relation pieces. This is a
baseline geometry-recognition compile-time regression, not a continuation-plan
rejection or GPU hang.

## 2026-09-09: Gemma continuation ownership causal controls on GPU 2

The follow-up used the same clean detached `a9df42d1` worktree and the same
positive specialized-input-layout proof as the preceding Gemma experiment.
Only probe-time plan selection changed. Physical GPU 2 was idle. The primary
matrix used B2 compilation followed by B1/B2 replay, a 10-second thermal
warmup, 120 balanced observations of all six CUDA graphs, and a 256-MiB L2
clear before each replay. Static controls now use the same multiplier-two,
296-worker geometry as dynamic.

| shape | standalone | static m2 | dynamic resident root 6 | 5->6 continuation, root 7 resident | full 5->6->7 chain | dynamic u32 counter storage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 55.264 us | 57.280 us | **61.408 us** | 63.568 us | 65.472 us | 61.376 us |
| B2 | 69.584 us | 81.888 us | **77.792 us** | 83.936 us | 84.064 us | 77.792 us |

All seven outputs are bit-exact among every variant and both controls at both
batch sizes. B1 selects eight distinct experts for eight assignments; B2
selects 15 distinct experts for 16 assignments and the token expert sets
differ. Every dynamic callable retained exactly one specialization across
B2 -> B1 -> B2. Baseline and full-chain cubin SHA256 values remain
`7d07ab14638895f21c8e5ecec8df631123148e1f2b72c161d0485bcf9085f3e8`
and `71433cc294559193b7b2b6192083174b9cfb2c607715d20833afc440162808cc`.
Baseline/full compilation took 28.702/27.291 seconds.

The root-6-only result isolates the expensive decision. Relative to resident
root 6 it loses 2.160 us (3.5%) at B1 and 6.144 us (7.9%) at B2. Chaining root
7 as well adds a further 1.904 us at B1 but only 0.128 us at B2. Root-6-only
retains R126, zero spills, 34,816 bytes shared, and four warps, exactly the
baseline/full-chain resource envelope. Its generated source is
`/tmp/torchinductor_eche/25/c25jvn7hxhy5pynpt4j6vkqwx2c73oy6tfopej4fyw3ss4smml7u.py`
(SHA256 `d75356fcf73e3afc35f3e27d5542d7d0bfb763d32b9e44f0b8b6e52e498c5fe9`).
The root-5 scheduled helper calls root 6 only inside the fan-in-32 final-arrival
branch; the entry function has no resident root-6 call. Root 7 remains a
resident exact-counter waiter, and no root-7 call occurs in the root-5 helper.

The emitted dynamic schedule supplies an exact placement proof. With W=296,
root 5 occupies packed global slots `[421,773)` at B1. Its last global wave is
slots `[592,773)`, physical workers 0--180; resident root 6 immediately fills
slots `[773,784)`, workers 181--191. At B2 root 5 occupies `[842,1546)`; its
last global wave tail is `[1480,1546)`, workers 0--65, and resident root 6
fills `[1546,1568)`, workers 66--87. Thus the production resident plan runs
the reductions on blocks otherwise idle in the producer tail wave. With root
6 removed, the root-6-only schedule gives resident root 7 the first packed
slots after root 5: worker 181 at B1 and workers 66--67 at B2.

A continuation cannot preserve that ownership. Generated Triton appends root
6 directly to `tile_dependency_root_5_scheduled_task` on the thread block whose
atomic arrival observes `epoch * 32 + 31`. The actual winner is runtime- and
data-dependent among the 32 producers of each key, so static inspection cannot
name its physical worker. The lexicographically last producer-task owners are
`[156,188,220,252,284,20,52,84,116,148,180]` at B1 and
`[16,17,80,81,144,145,208,209,272,273,40,41,104,105,168,169,232,233,0,1,64,65]`
at B2, but completion order need not equal task order. What the code proves is
that continuation serializes each reduction onto a producer strand and denies
it to the complementary resident blocks; it does not reduce the number of
logical root-6 tasks. The measured loss follows that ownership restriction.

The same-m2 static controls separate this effect from dynamic shape lowering.
Static B1 chooses resident root 6 and measures 57.280 us (R128/0 spills/34,816
bytes/W4), leaving a 4.128-us dynamic parameterization gap. Static B2 instead
chooses the full 5->6->7 chain and measures 81.888 us (R116/0/34,816/W4), so
it is not a resident-placement comparison. A second three-way, 10-second/120-
sample cold-L2 control disabled only static B2's 5->6 continuation while
retaining 6->7:

| B2 static m2 control | time | resources |
| --- | ---: | --- |
| selected full 5->6->7 chain | 83.808 us | R116 / 0 spill / 34,816 B / W4 |
| root 6 resident, 6->7 continuation | **75.648 us** | R116 / 0 spill / 34,816 B / W4 |
| standalone anchor | 69.472 us | eight launches |

The outputs are bit-exact. The resident control source is
`/tmp/torchinductor_eche/tj/ctjler7swrlg5kofkirva5h5uqzpnmktml4bewvmohgsw46w3esm.py`
(SHA256 `9a5648fe568acdd538aa446a0e9fb4865ed62a6c5f024b558e39a56e7adb2499`).
It emits root 6 as a 22-worker resident loop and retains root 7 inside root 6;
the normal source emits both consumers inside root 5. Normal/resident compile
times were 14.038/5.208 seconds. The static global-list refinement accepts the
full-chain schedule but declines the forced-resident one, which therefore uses
the valid conservative six-segment placement. Even without dynamic packed-tail
placement, resident root 6 wins by 8.160 us, independently confirming that
producer-strand continuation is causal.

Finally, a bounded-run counter-width ablation changed dynamic parameterized
counter storage, acquire loads, and atomics from u64 to u32 while retaining
128-byte counter spacing. PTX changes all 14 synchronization atomics and all
12 acquire loads from u64 to u32; resources stay R126/0/34,816/W4. Latency is
unchanged within 0.032 us at B1 and exactly tied at B2. The generated source is
`/tmp/torchinductor_eche/kc/ckc4m6aqr6xl7x6zesvbqy4wz5exwfvjzcm45smkphg3qv2llff6.py`
(SHA256 `50b8cf0211f5d7fbfb2495430c72d9e92f62d5391339d9314f39b70ae96b297d`).
This bounded probe is not replay-wrap safe and intentionally is not a compiler
proposal; it rules out u64 synchronization memory operations as the measured
continuation penalty. No non-perturbing root-level timestamp facility is
available in this fused kernel: external profiling sees one launch, while
injecting global-timer stores into the exact-counter critical path would alter
the short tail being measured. The schedule mapping and the two independent
resident-versus-continuation timing controls are therefore the causal evidence.

## 2026-09-09: Gemma worker-width recovery sweep on GPU 2

A final probe-only sweep kept root 6 resident and varied only the persistent
worker count. It again used clean detached `a9df42d1` plus the positive input-
layout provenance proof on physical GPU 2. No compiler or kernel-body source
was edited. The dynamic variants compile once from B2 and replay B2 -> B1 ->
B2 with one specialization. Fixed-shape controls use a probe-local exact
packed schedule: each root's existing task order is split only at global
worker-wave boundaries, and root 6 occupies the immediately following idle
workers. Root 7 retains the ordinary exact planner's choice.

The four-second/30-observation balanced cold-L2 screen was:

| shape | dynamic m1 | dynamic m2 | dynamic m3 | dynamic m4 |
| --- | ---: | ---: | ---: | ---: |
| B1 | 69.504 us | 61.312 us | **53.120 us** | 53.280 us |
| B2 | 96.128 us | 79.744 us | **73.616 us** | 79.776 us |

| shape | exact packed static m2 | exact packed static m3 | exact packed static m4 |
| --- | ---: | ---: | ---: |
| B1 | 59.360 us | **51.152 us** | 51.168 us |
| B2 | 79.808 us | **71.664 us** | 77.792 us |

Thus m3 wins both shapes. Relative to m2 in the same screen, m3 saves 8.192
us/6.128 us for dynamic B1/B2 and 8.208 us/8.144 us for exact static. Moving
to m4 is neutral at B1 and loses 6.160/6.128 us at B2 for dynamic/static. This
isolates worker width from the root-6 ownership effect measured above.

Every packed placement is complementary by construction. With root-5/root-6
global intervals `[421B,773B)` and `[773B,784B)`, the final root-5 owners and
root-6 owners are:

| shape | workers | final root-5 owners | resident root-6 owners |
| --- | ---: | --- | --- |
| B1 m2 | 296 | 0--180 | 181--191 |
| B1 m3 | 444 | 0--328 | 329--339 |
| B1 m4 | 592 | 0--180 | 181--191 |
| B2 m2 | 296 | 0--65 | 66--87 |
| B2 m3 | 444 | 0--213 | 214--235 |
| B2 m4 | 592 | 0--361 | 362--383 |

All six exact packed controls compiled and launched. Static B1 compile times
for m2/m3/m4 were 9.256/6.028/4.513 seconds; B2 took
66.340/64.230/68.506 seconds because the larger exact task-order slices are
more expensive to prove. All use R128, zero spills, 34,816 bytes shared, and
four warps. All four dynamic widths use R126, zero spills, 34,816 bytes, and
four warps; their compile times were 28.988/29.277/30.412/29.267 seconds.

The m3 winners were then reloaded from their exact generated sources and
confirmed together with the standalone control using a 10-second warmup and
120 balanced cold-L2 observations:

| shape | standalone | one-cubin dynamic m3 | exact packed static m3 |
| --- | ---: | ---: | ---: |
| B1 | 55.264 us | **53.216 us** | **51.168 us** |
| B2 | 69.888 us | **75.712 us** | **71.712 us** |

The dynamic m3 source is
`/tmp/torchinductor_eche/pb/cpbq4gxnrkxr2vuf35hpsm6ccxl4abfeyviz7yya7gcs3mbsr7ob.py`
(source SHA256
`7861c35db155e3aad0bbe9df267842baf8009498aa33fc601f30a5b60651fb82`,
Triton hash
`1c2781fc06dada9886183ef6754bf6991e45554b2a73ecff82340162aa8f9fa6`,
cubin SHA256
`4eb331e110ad1760944ea008ad47ef627c0a9053bf1d8939f6403882ac16fe9e`).
The packed static B1/B2 m3 sources are respectively
`/tmp/torchinductor_eche/wu/cwurmtsumsi36hwut77yn2bkkqk5ofrssrdhsh3jc7pdql2o4vte.py`
and
`/tmp/torchinductor_eche/dz/cdz5s2hikj2m3t66nndjtvyi7ergyhirixmxu5w7pzejxu4v3nnb.py`,
with source SHA256 values
`b522c9ca8c9237e3da93fff0dd054a82bae2c65288cf23ed22fbc705e9dcacc8`
and `1e7b2d1d0a4e99b9fbcd68d33ae11f244d83a228bb2f4edbefc9137a72bb6b42`.

All seven outputs are bit-exact across the confirmed variants and standalone.
B2 again selects 15 distinct experts across 16 assignments with different
per-token expert sets. The dynamic callable still has exactly one cubin after
B2/B1/B2 replay. There is one unavoidable B1 topology distinction: fixed B1
collapses `6->7` to a one-key whole-root barrier, so the exact planner's
`root_barrier_producer_root` rule retains resident root 7 and exposes no legal
final-arrival candidate. Dynamic retains the parameterized `6->7`
continuation. The invariant was not weakened. At B2 both exact packed static
and dynamic use resident root 6 plus the `6->7` continuation, making their
confirmed 4.000-us difference the clean residual symbolic-rendering/state
overhead at identical W444 placement. The B1 2.048-us dynamic/static difference
also includes the unavoidable root-7 topology difference.

The historical untouched B1 result of 49.120 us is therefore not recovered:
the best exact packed resident-root6 control is 51.168 us, 2.048 us slower,
and the deployable one-cubin dynamic winner is 53.216 us, 4.096 us slower.
Ownership accounts for the earlier continuation regression, m3 recovers the
worker-width loss while preserving the best B2 result, and approximately four
microseconds of B2 remains attributable to symbolic dynamic lowering/state
rather than placement or resource envelope.

## 2026-09-11: post-schedule quotient and current fixed-capacity controls

The scheduler now feeds exact per-iteration nested readiness into root-local
ordering and the event-frontier proposal. It derives a compact quotient only
after accepting a `WorkerSchedule`; an unsafe or unlowerable compact result
falls back to exact keys on that same schedule without rerunning ownership or
placement. The combined scheduler/codegen suite passed 223 tests and 94
subtests before the additional real multi-wave regression fixture.

On physical GPU 2, Qwen B1/Q1/S8192 remains bit-exact across all 16 outputs and
the full KV cache. Exact 96-key lowering exposed the cost of per-iteration waits
at 210.800 us. The generic earliest-admission quotient safely collapses that
event to one fan-in-96 key and restores:

| implementation | cold-L2 median |
| --- | ---: |
| old fixed-context persistent control | 96.096 us |
| current fixed-capacity depth 1 | 106.400 us |
| current fixed-capacity depth 2 | 106.336 us |
| matched full-layer standalone | 110.400 us |

Thus the current generic persistent kernel beats standalone by 3.68%, but it
does not recover the old persistent schedule. Depth two places roots 12/13/14
at global slots `[2930,4466)`, `[4466,4562)`, and `[4562,5074)` and derives
8->9 as 30/2 plus 13->14 as one fan-in-96 key. Since all 96 root-13 tasks have
one schedule rank, post-placement lowering cannot manufacture the historical
74/22 overlap. The next generic experiment is retirement-frontier placement:
reserve the complete producer relation, but let complete downstream cohorts
use each worker once after that worker's final reserved producer occurrence.

On physical GPU 6, the mechanically identical fixed-capacity
`static_shapes=False` Gemma source at multiplier three/depth two improved over
the previous checkpoint:

| shape | persistent | standalone | result |
| --- | ---: | ---: | ---: |
| B1 | 50.976 us | 55.104 us | 7.49% lower latency |
| B2 | 69.440 us | 69.408 us | 0.05% difference |

Three distinct route tensors per capacity are bit-exact across all seven
outputs; B2 uses 15 distinct experts across 16 assignments, and each capacity
retains one cubin. Depth two changes the resident placement from 7 to 2 waves
at B1 and 9 to 4 waves at B2. The checked-in Gemma decorator remains
`static_shapes=True`; this fixed-capacity runtime-routing contract is still a
probe-local mechanical source conversion.

Artifacts:

- `/tmp/qwen_production_post_quotient_current_tree_depth1_gpu2.json`
- `/tmp/qwen_production_post_quotient_current_tree_depth2_gpu2.json`
- `/tmp/qwen_standalone_native_cold_l2_gpu2.json`
- `/tmp/gemma_current_fixed_capacity_gpu6_pair500.json`

## 2026-09-11: retirement-frontier first GPU signal

The first retirement implementation was deliberately conservative: it kept
the complete prepared schedule fixed and filled only literal holes after each
worker's final producer occurrence. Five generic CPU tests passed, including a
rotated producer interval and same-rank nested admission, but the production
Qwen run was unchanged at **106.336 us**. Roots 12/13/14 were already densely
packed at global slots `[2930,4466)`, `[4466,4562)`, and `[4562,5074)`, so the
terminal absolute-wave complement was occupied rather than idle. The final
counter therefore remained one fan-in-96 key. Artifact:
`/tmp/qwen_retirement_depth2_gpu2.json`.

This rejects hole filling as the scheduler abstraction. The generic successor
is part of the existing committed-run transition. For a reserved dense run of
`N=qW+r` tasks in `[F,E)`, ownership advances through all `N`, while readiness
is temporarily held after the first `qW` tasks. The linear interval
`[E,E+W-r)` is exactly one next slot for every low-count producer lane, even
when it crosses an absolute wave. The ordinary complete-cohort chooser may use
that interval once, after which the final `r` producer tasks become visible and
normal placement resumes. For Qwen this derives an 832-slot interval from
`1536=1*1184+352`; it should place root-13's 74-task ready prefix, then the
512-task nested root-14 family, and leave root-13's 22-task suffix afterward.
This must replace, not coexist with, the experimental retirement pre-pass.

## 2026-09-11: integrated retirement-frontier result

The literal-hole experiment above was removed. The production implementation
is now a transition in the one event-frontier selector: a fully admissible
dense producer suffix remains an immutable ownership reservation, while its
admission cursor is held at the last complete worker round. Each lane that has
already retired may host at most one complete downstream cohort before the
producer tail is revealed. Final schedule coverage, segment-DAG progress, and
counter lowering are still proved from the same `WorkerSchedule` and
`ReadinessGraph`; there is no second task DAG or Qwen-specific path.

For the Qwen B1/Q1/S8192 case (`W=1184`), the compiler derives:

```text
P       [2930,4466)
R first [4466,4540)   # 74 tasks
C       [4736,5248)   # 512 tasks, next complete worker rank
R final [5298,5320)   # 22 tasks
```

The producer traversal's certified inverse proves the exact readiness frontier
`key k -> 16*k + 15`; the post-schedule quotient consequently emits 74/22
rather than a literal or model match. Outputs remain bit-exact.

A same-process three-way cold-L2 ablation on physical GPU 2 isolated the
mechanism:

| placement / nested wait | latency |
| --- | ---: |
| packed / one fan-in-96 wait | 104.384 us |
| gapped / one fan-in-96 wait | 108.512 us |
| gapped / derived 74/22 waits | 104.416 us |

The frontier recovers 4.096 us, bringing the new general schedule to parity
with the prior packed compiler result. Repeating packed and retirement in both
measurement orders for three trials gave stable packed 104.320 us and
retirement 104.384--104.448 us after one first-order warmup outlier. The
matched full-layer standalone remains 110.400 us.

Cross-workload checks after this change:

| workload | persistent | standalone | result |
| --- | ---: | ---: | --- |
| ragged FlashMLA B4/Q4/H16 | 63.360 us | 69.472 us | 9.65% faster, bit-exact |
| Gemma A4B fixed-capacity B1 | 51.040 us | 55.168 us | 7.48% faster, bit-exact |
| Gemma A4B fixed-capacity B2 | 69.504 us | 69.504 us | parity, bit-exact |

The final combined scheduler/codegen suite passes 233 tests and 94 subtests.
The gapped relation renderer is enabled only for a nonempty repeated root;
ordinary one-shot root-major schedules retain their previous compact lowering.
An independent architecture review found no soundness blocker and confirmed
that retirement is integrated into the sole selector with no model literals or
obsolete hole-filling path.

### Qwen runtime-mask static analysis and ablation

The fixed-capacity Qwen source differs from the old exact-context source only
inside attention scheduling semantics, not in task capacity: every attention
CTA loads `context_lens`, executes a runtime outer split guard, forms an
elementwise valid-key mask, performs masked block-table/K/V loads, and applies
a score `where`. At B1/S8192 every one of those predicates is true, but Triton
cannot eliminate them because `context_lens` deliberately remains runtime.

A temporary full-context-only source ablation replaced exactly that block with
the old unmasked attention body while retaining the current compiler, tensor
capacity, configuration, schedule, and all other kernel bodies. On physical
GPU 3 it was bit-exact for all 16 outputs and the full KV cache. The plans were
identical. Paired cold-L2 timing measured 106.432 us runtime-masked versus
102.448 us unmasked, so masking costs 3.984 us in this case. It also changes
the persistent binary from R255/zero spills/450,192-byte cubin to R255/22
spills/470,344-byte cubin. This explains a material part, but not all, of the
gap to the historical roughly 96-us exact-context result. It is a semantic
cost of runtime raggedness, not evidence for another scheduler path. Artifact:
`/tmp/qwen_full_unmasked_ablation_gpu3.json`.

## 2026-09-12 current unified-scheduler rollout

The post-retirement, post-source-ticket compiler was remeasured from stable
sources on physical GPU 7. Qwen uses the checked-in pretuned B1/Q1 source with
`static_shapes=False`: fixed capacity/model extents and strides are explicit
specializations, while context lengths, positions, block-table entries, slot
mappings, KV contents, and activations remain runtime values. The same cubin
was replayed at S8192 and S2051 with bit-exact outputs and KV state.

| Qwen B1/S8192 path | cold-L2 median | resources |
| --- | ---: | --- |
| persistent depth 1 | **104.352 us** | R255, 22-byte spill, 17,408 B shared, W1 |
| persistent depth 2 | 106.224 us | R255, 26-byte spill, 17,408 B shared, W1 |
| matched 128-split standalone | 106.224 us | constituent kernels |

Depth two still derives the exact root-13 `74/22` split and advances root 14,
but its larger lowering and four extra spill bytes erase that structural gain.
Depth one is the autotuned winner and is 1.76% faster than standalone. No
profitability exception is needed: `cross_loop_pipeline_depth` already exposes
both legal choices.

The fixed-capacity Gemma `static_shapes=False` probe also retained runtime
routing and exact outputs over all-same, negated, rolled/scaled, and diverse
expert assignments:

| Gemma path | persistent depth 2 | standalone | result |
| --- | ---: | ---: | --- |
| B1, m3/W444 | **51.008 us** | 55.200 us | 7.59% lower latency |
| B2, m3/W444 | 69.664 us | 69.536 us | 0.18% higher latency; practical parity |

B1 is a pass. A clean ordinary-knob sweep resolved the apparent B2 regression
without a compiler policy change: at multiplier three, depth one was 71.680 us
and depths two through four were the same schedule/cubin at 69.536 us in the
screen; multipliers two and four were slower, while multiplier five was
correctly rejected because 740 resident programs exceed the 592-program
capacity. The balanced 500-sample measurement above is therefore the tuned
B2 result, not a selected outlier. Artifacts:
`/tmp/qwen_current_rollout_gpu7.json`,
`/tmp/qwen_standalone_rollout_gpu7.json`,
`/tmp/gemma_current_fixed_capacity_gpu7_pair500.json`, and
`/tmp/gemma_b2_tuning_gpu7_summary.json`.

## 2026-09-12: production-source attribution and same-GPU controls

The checked-in Qwen B1/Q1 source was compiled on physical GPU 0 with only its
decorator changed between `static_shapes=False` and `static_shapes=True`.
The two variants have identical plans, normalized generated Triton, TTIR,
TTGIR, and executable SASS instruction streams.  Both use R255, 22 spill
bytes, 17,408 bytes shared memory, and one warp.  Explicit specialization of
the fixed tensor capacities and strides therefore completely removes
`static_shapes` as a scheduler or body-lowering variable while leaving the
intended attention metadata as runtime tensor loads.

The only device-body source difference from the old fixed-context control is
the required ragged-attention work: runtime `context_lens`, the split guard,
the per-element tail predicate, masked page/K/V loads, and the score select.
Removing only those operations leaves the plan/counters/barriers identical,
removes the 22 spill bytes, and shortens the binary by 344 SASS instructions
(including 21 spill stores and 21 spill loads).  Timing of two large cubins in
one process is unusually sensitive to allocation and code placement by about
2 us: reversing allocation/capture order can erase or invert the apparent
masked/unmasked difference.  The earlier statement that masks cost exactly
3.984 us is therefore too strong.  A favorable isolated comparison is 100.320
us masked versus 96.288 us unmasked, but the robust conclusion is structural:
runtime raggedness creates the extra instructions and register pressure;
`static_shapes=False` and the unified scheduler do not.

On that same GPU, the current production persistent depth-one kernel is
**100.320 us** versus **108.384 us** for the matched standalone boundary
(7.44% lower latency).  Depth two's derived 74/22 frontier is also real: with
an identical `WorkerSchedule`, replacing it by one root-entry fan-in-96 wait
regresses 104.320 to 106.464 us.  Depth one remains the end-to-end autotuning
winner because depth two's larger lowering uses 26 rather than 22 spill bytes.
No Qwen-shaped scheduling exception follows from either result.

The checked-in Gemma B1 source/config was then rerun on an otherwise idle
physical GPU 2.  The exact public entry measured 51.01 us versus 124.93 us for
vLLM's production FlashInfer CUTLASS path.  The mechanically matched
eight-root Helion comparison measured **51.136 us persistent versus 55.200 us
standalone**.  A clean older scheduler checkpoint measured 49.44 us in a
separate public-entry run.  A direct compilation from current Helion `main`
at `cfb135ef` also selected the exact same generated module and configuration
as this branch and the preserved older control: generated-source SHA256
`0e9560efa9aefa58b77c1f9d9e331e1de024f08e4152143eb20d5fc850d84bdd`,
multiplier four, W4, R128, zero spills, and 34,816 bytes shared.  The 49--51 us
spread is consequently a measurement/code-placement mode, not a compiler
regression.  The checked-in pretuned form and the fixed-capacity probe both
remain performance passes through the unified compiler path.
