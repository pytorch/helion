# Static/dynamic cross-loop scheduler unification

Status: canonical implementation plan after the 2026-09-12 phase-0 executor
ablation rejected the first physical design.  Architecture, Qwen/Gemma, and
execution/progress reviewers signed off on the revised one-shot packet design.

This document supersedes the global event-frontier/list-scheduling roadmap in
`parametric_event_frontier_scheduler_plan.md`.  That document remains useful
history for the symbolic relation work already merged into this branch, but it
is no longer the implementation target for cross-root placement.

The experiments that motivated this pivot live on the reference-only branch
and worktree `helion-dynamic-dispatch-probes`.  They are evidence and
reproduction material.  They will not be merged or cherry-picked into the
compiler implementation.

## Decision

Build one compiler scheduler with two execution choices, not two scheduling
algorithms:

1. Construct one canonical, depth-one logical schedule from the existing
   `TileDependencyGraph`, `ReadinessGraph`, readiness counters, final-arrival
   continuations, and root-local ordering pass.
2. Represent every non-continuation root exactly once in the existing
   `WorkerSchedule`, in source/root order.
3. Select `static` or `dynamic` execution independently for each root through
   one root-indexed autotuning field.
4. Lower the final segment sequence into one ordered packet stream.  Static
   regions contribute coarse worker-strand packets; dynamic regions contribute
   one packet per logical task.
5. Delete the global event-frontier/list placer, its pipeline-depth knob, and
   the special one-source ticket path after the general executor passes the
   performance gates below.

The scheduler still makes important decisions: it derives exact readiness,
selects continuations, chooses a root-local task permutation, and proves
progress.  What disappears is speculative cross-root splitting and placement.
Dynamic ownership supplies the load balancing that list placement previously
tried to approximate with fixed workers.

The north star is simplification.  The final compiler must have one semantic
DAG, one readiness graph, one worker schedule, one continuation identity, and
one code-generation path.  Static versus dynamic is a physical execution
property of a root in that schedule, not another scheduler or plan IR.

## Evidence for the pivot

All numbers in this section are cold-L2 B200 measurements from the archived
probes.  They must be refreshed from the implementation branch before final
acceptance.

| workload | canonical dynamic | list-order dynamic | relevant control |
| --- | ---: | ---: | ---: |
| FlashMLA B4/Q4/H16 | 61.312 us | 61.280 us | 67.456 us standalone |
| FlashMLA B9 ragged | **88.064 us** | 90.048 us | 100.288 us standalone |
| Muse FFN, m8 | **159.744 us** | 161.728 us | 172.000 us standalone |
| Muse FFN, m12 | **161.392 us** | 163.648 us | 173.888 us standalone |
| DeepSeek-V3 MoE | 167.872 us | 167.872 us | 157.728 us standalone |
| Nemotron MoE | 77.856 us | 77.856 us | 61.504 us standalone |

Current production validation after the unified lowering uses 500 cold-L2
samples after a 10-second warmup on GPU4 (GPU1 for Qwen/Gemma and the MoE
endpoints):

| workload | unified persistent | same-run control | preserved facts |
| --- | ---: | ---: | --- |
| FlashMLA B4/Q4/H16 | **61.280 us** | 67.552 us standalone | bit-exact replay, K22/F16, R80/spill6/49,160B |
| FlashMLA B9 ragged | **91.936 us** | 106.224 us standalone | bit-exact replay, all 18 counters, R80/spill6/49,160B |
| FlashMLA Q1, B1/S65536/H16 | **38.688 us** | **38.688 us standalone** | bit-exact replay, static/static, fan-in-128 root barrier, R192/spill0/169,984B |
| FlashMLA Q2, B64/S512/H16 | **32.640 us** | **32.640 us standalone** | bit-exact replay, static/static, direct continuation, R255/spill24/169,984B |
| Muse FFN m8 | **161.632 us** | 173.840 us standalone | five outputs bit-exact, R160/spill0/16,640B |
| Qwen3 pretuned B1/S8192 | **102.400 us** | 102.400 us pre-refactor golden | byte-identical Triton/SASS, R255/spill22/17,408B |
| Qwen3 ragged B2, S=[2048,8192] | **126.944 us** | 128.960 us same-source static control | bit-exact, identical cubin/resources, R255/spill22/17,408B |
| Gemma4 A4B pretuned B1 | **49.056 us** | 55.168 us standalone | byte-identical Triton/SASS, R128/spill0/34,816B |
| Gemma4 A4B B2, 15 routed experts | 69.568 us | 67.552 us standalone | bit-exact, tuned m3, R116/spill0/34,816B |
| DeepSeek-V3 MoE, all dynamic | 170.016 us | 159.552 us standalone | 11 outputs/replay correct, R150/spill0/52,224B |
| Nemotron MoE, all dynamic | 79.680 us | 63.216 us standalone | output/replay correct, R96/spill0/34,816B |

The B4/B9 kernels keep their TMEM attention root inline and outline one
helper-safe packet suffix.  B4 has one 786-packet stream; B9 has one
1,448-packet stream.  Both execute one ticket claim per CTA and no dispatch
loop.  Muse applies the same capability rule: its GEMM roots remain inline and
only the final reduction suffix is outlined.  Same-run absolute latency drifts
between measurements, so paired persistent-minus-standalone margins are the
primary regression check.  Recorded ThunderKittens timings (71.456 us B4 and
88.128 us B9) produced nonfinite/invalid outputs under the required numerics;
B4 nevertheless beats the raw timing, while B9 is compared primarily against
the valid matched standalone and the prior Helion result.
DeepSeek's relative gap is unchanged from its archived one-shot control
(approximately 6.6%), and Nemotron reproduces its archived dynamic latency.
The same current source measured DeepSeek all-static at 182.112 us and
Nemotron all-static at 104.320 us, so dynamic improves those canonical static
endpoints by 12.096 us and 24.640 us respectively.  Their remaining gaps to
standalone are constituent-body/resource work; the scheduler does not hide
them by changing numerics or fusion.

The canonical Q1 and Q2 cases deliberately exercise different synchronization
mechanisms under the same `static/static` root policy.  Q1 publishes a
fan-in-128 root barrier; Q2 contracts its second root into a direct same-CTA
continuation.  Both exactly match their two-launch standalone boundary and
pass 20 alternating-input CUDA Graph replays.  This rules out selecting an
executor merely because one synchronization form happened to favor it.

In all six comparisons, dynamic execution erased the list order's benefit.
Canonical order tied it at B4/DeepSeek/Nemotron and beat it at B9/Muse.  The
negative control is Qwen full decode: same-source static depth-one ownership
measured 104.384 us while unchanged-order dynamic FIFO measured 112.608 us.
The older approximately 94-us Qwen number used a fixed-context/unmasked source
and is historical context, not a valid dynamic-source acceptance baseline.
Current `static_shapes=False` controls are approximately 100.320 us on GPU0
and 104.352 us on GPU7; acceptance uses a fresh same-device, same-source static
golden.  Gemma A4B likewise has a strong checked-in static control around
49--51 us at B1.

The experiments used two physical implementations:

- FlashMLA and Muse launched one ticket-owning CTA per logical task and relied
  on hardware CTA retirement/admission.
- DeepSeek and Nemotron launched a fixed resident cohort whose CTAs repeatedly
  claimed tickets from an atomic cursor.

The first signed-off draft proposed using the fixed resident claim loop for
every dynamic region.  Phase 0 rejected that design.  On canonical FlashMLA
B4, 500 cold-L2 samples after a 10-second warmup measured:

| physical executor | latency | resources |
| --- | ---: | --- |
| one-shot root-major task packets | **61.440 us** | R80, 54-byte spill, 49,160-byte shared |
| fixed W148 `T+W` claim loop | 104.416 us | R80, 104-byte spill, 49,160-byte shared |
| matched standalone | 67.456 us | constituent kernels |

All outputs were bit-exact and both persistent variants retained the required
K22/fan-in16 compact counter.  Moving the complete packet body behind a common
noinline boundary was not a valid control: Triton rejects the Blackwell TMEM
attention body in a non-kernel function during `lowerTensorMemoryAlloc`.
Therefore the 43-us gap is an executor/code-shape limitation, not missing
readiness or an untuned knob.  The fixed-loop design failed its written kill
gate and is abandoned.

The replacement is a single **one-shot ordered packet stream**, which
generalizes the successful source-ticket mechanism without retaining a source
special case:

- a maximal static run contributes exactly `W` packets; packet `w` executes
  worker `w`'s unchanged static strand through every root in that run;
- a maximal dynamic run contributes one packet per logical root task in
  canonical root/local-ordinal order; and
- one monotone atomic ticket assigns those packet roles in actual CTA admission
  order.  Each launched CTA executes one packet completely and retires.

This handles arbitrary `static -> dynamic -> static` sequences because the
static unit is the complete resident cohort, not an individual root task.  It
also preserves both measured endpoints: an all-dynamic graph has exactly the
MLA/Muse one-shot form, while an all-static graph has one `W`-packet run that
strength-reduces to the current `program_id` worker mapping with no ticket
atomic and byte-identical lowering.

Muse m12 remains evidence about logical order and dynamic load balancing, not
about a feasible fixed pool of 1,776 simultaneously resident CTAs.  The
one-shot stream does not require all dynamic packets to be resident; any
static run still requires its `W` worker packets to fit concurrently.

## Scope and non-goals

This redesign supports a fixed, positive compiled task capacity per root with
runtime data inside that capacity.  A statically zero-capacity root is rejected
before scheduling; runtime masks may still turn every capacity task into a
no-op, and those packets retain their normal publications.  In particular, the
following remain runtime values:

- `seq_lens[B]` and active-request masks;
- KV page/block metadata;
- per-request attention work and ragged tails; and
- MoE routing, expert IDs, and expert occupancy.

Batch/query capacity, root count, block sizes, and root task-capacity formulas
remain specialized when they determine the compiled task universe.  Supporting
one cubin across different capacities is future work and is not required for
this simplification.

The redesign does not add:

- a host-generated schedule or host/device synchronization;
- a device ready queue;
- a cost or latency model;
- a new schedule/program/packet/event class;
- CLC lowering; or
- cyclic dependency support.

CLC remains a future implementation of dynamic claiming.  It must consume the
same accepted plan and proof; it is not a replacement scheduler.

## Final sources of truth

### `TileDependencyGraph`

This remains the sole semantic dependency graph.  No scheduler-side CTA DAG is
built for proof or code generation.

### `ReadinessGraph`

This remains the sole source of readiness keys, producer publications,
consumer waits, exact fan-in, and fallback root-barrier obligations.

### `StaticPipelinePlan`

Keep the existing fields unchanged:

```python
StaticPipelinePlan(
    worker_schedule,
    root_task_orders,
    readiness_counters,
    root_barrier_edges,
)
```

Continuation identity remains solely in
`ReadinessCounterPlan.continuation_consumer_index`.  Dispatch choice is encoded
by the retained root segment, not copied into another plan field.

### `WorkerSchedule` and `WorkerScheduleSegment`

The final invariant is deliberately narrower than the current list-scheduler
representation:

- every non-continuation root has exactly one segment;
- continuation roots have no segment;
- segments occur in increasing source-root order;
- each segment's target is an exact, bijective cover of that root's tasks;
- the segment's normalized relation plus its exact dense support define the
  only task traversal used by both static and dynamic rendering; and
- one scalar field on the existing segment says whether that traversal is
  statically owned or dynamically claimed.

Add `dispatch_mode: Literal["static", "dynamic"] = "static"` to the existing
`WorkerScheduleSegment`.  This is one field on the authority we already have,
not a new abstraction or parallel plan.  All segments remain in the current
resident launch stage while the old source-ticket path is removed.  The old
stage-zero meaning, its extra launch grid, and its singular-source rules are
then deleted.  The resulting degenerate launch-stage axis may be removed in a
later mechanical cleanup; it must not be repurposed as dispatch mode.

Cross-root chronology is now the source-root order, not an ordering induced by
dispatch mode or by splitting a root into several segment occurrences.  The
worker/wave coordinates retain their exact role within a static segment and
provide the dense ordinal certificate used to decode a dynamic ticket.  A
representable flattened relation is an optional strength reduction; otherwise
codegen substitutes the proved dense slot directly into the authoritative
normalized relation.  The compiler rejects a segment whose exact support is
not dense or bijective.

This is the single-source-of-truth rule:

```text
root task order
    -> one normalized WorkerScheduleSegment with one dispatch_mode
    -> either static striding or dynamic ticket decoding
```

Neither renderer may recreate or alter the logical PID order.

## The one scheduling pipeline

The final pipeline is:

```text
TileDependencyGraph
  -> ReadinessGraph
  -> exact counters + conservative root-barrier fallback
  -> canonical all-static one-segment-per-root seed
  -> transactional root-local producer ordering of the seed
  -> final-arrival continuation selection/assignment
  -> transactional schedule rebuild + final root-local ordering
  -> apply per-root static/dynamic choices
  -> exact coverage + progress + publication validation
  -> optional nested-counter quotient as an emitted strength reduction
  -> freeze StaticPipelinePlan
```

Details:

1. Counter and root-barrier edge selection is independent of dispatch mode.
   Every
   `TileDependencyGraph` obligation must remain represented by an emitted exact
   counter or root-barrier fallback.  Static worker/wave order is never allowed
   to discharge an obligation, because that implication would disappear when
   either endpoint becomes dynamic.  Root-barrier participant and arrival
   metadata is frozen only after dispatch modes are applied.
2. `build_baseline_worker_schedule` produces one root-major segment per root.
3. `_consumer_major_producer_order` may change the task permutation inside one
   root, transactionally.  It may not split, interleave, or reorder roots.  Its
   current `_source_ticket_frontiers` branch must be removed: the retained
   ordering is derived only from mode-independent readiness relations.  Phase
   0 compares the exact root-local traversal with and without that branch; if
   ordinary readiness cannot express a required ordering, generalize that
   readiness rule rather than querying dispatch mode.
4. Continuations are selected exactly as today from this accepted all-static
   schedule.  The dispatch vector may not create a new continuation.  A
   continuation is executed by the CTA that observes the final readiness
   arrival; this rule is independent of whether that producer task was
   statically owned or dynamically claimed.
5. Continuation ownership is applied by transactionally rebuilding and
   renormalizing the final one-segment schedule from the frozen all-resident
   seed.  Never delete segments in place and leave stale offsets, support, or
   publication geometry.  Remaining roots retain the identical local
   traversal and source order.  Mirror today's two transactions unless an
   equivalence test proves they can be collapsed: first order the all-resident
   seed used for continuation choice, then rebuild and run the same root-local
   transaction under the final continuation counters.
6. The configured root dispatch vector changes only each remaining segment's
   `dispatch_mode` field.
7. Nested counter compaction remains a post-selection lowering optimization.
   It cannot feed back into schedule construction.  There are two generic
   proofs, both derived from the final schedule:

   - A root-entry quotient is legal when every contributing producer
     recursively contracts to a strictly earlier root and the mixed-mode
     progress theorem proves that its work is resident/claimed before the
     consumer can block.  Static producers use stable resident ownership;
     dynamic producers use ticket-interval precedence.  Completion remains
     gated by the compact counter.
   - A finer segmented quotient retains the existing worker-rank proof only
     when every relevant occurrence is static.

   Dynamic same-root quotients decline unless exact ordinal precedence is
   added later.  Static frontiers such as Qwen's historical 74/22 split may
   never be carried into dynamic execution by assumption.  On any failed
   proof, retain the exact semantic counter on the identical schedule.

There is no list-schedule proposal, candidate cascade, priority queue,
criticality class, release credit, affine-repeat policy, or pipeline-depth
search in this pipeline.

## Autotuning surface

Replace `cross_loop_pipeline_depth` with one field:

```python
cross_loop_root_dispatch: list[Literal["static", "dynamic"]]
```

Its length is the number of task-family roots known when
`ConfigSpec.enable_cross_loop_schedule(root_count)` is called.  Implement it
with the existing `ListOf(EnumFragment(...), length=root_count)` machinery.
Default every entry to `"static"` to preserve current-main behavior.

This is one config field but has one binary coordinate per root.  Use ordinary
`ListOf` neighbor generation, including its uniform candidates.  Do not add a
topology-pruned search space or a custom candidate generator.

The field is indexed by source root, not by schedule segment.  That makes its
meaning stable when continuation selection removes a root.  The entry for a
continuation root is ignored; duplicate autotuner candidates are acceptable
and preferable to topology-specific knob shapes.

Expected useful settings are evidence, not compiler heuristics:

- FlashMLA B4/B9: all non-continuation roots dynamic;
- Muse FFN: all non-continuation roots dynamic;
- Qwen3 decode/FFN: all static;
- Gemma A4B: initially all static;
- DeepSeek-V3 and Nemotron MoE: initially all dynamic, then tune mixed vectors
  if individual heavy roots prefer resident ownership.

Do not infer a mode from a model name, root number, fan-in literal, task count
threshold, or a topology-shaped candidate list.  The only compiler policy is
legality; profitability belongs to the ordinary autotuner.

## Static execution

Static lowering is the existing persistent-worker behavior.  For a segment
with `W` workers, worker `w` executes the exact segment ordinals assigned to
its strided slice.  It performs the segment's incoming waits, task-level waits,
body, publications, and root-barrier publication using the frozen plan.

Consecutive static roots form one maximal static run.  Its `W` packet roles
are exactly the current persistent worker bodies restricted to those roots:
static packet `w` visits every segment in the run and executes worker `w`'s
slice before retiring.  This preserves cross-root strand chronology,
continuations, and the current static progress proof inside the run.

All-static code generation is a strict compatibility gate:

- same root-local traversal;
- same continuations and counters;
- same root-body inlining/outlining decisions;
- same launch grid and persistent state; and
- byte-identical lowered Triton whenever unrelated naming cleanup does not
  prevent it.

This is how Qwen and Gemma keep their proven resident behavior while the
global list machinery is removed.

## Dynamic execution

### Derived packet stream

Codegen walks source roots and coalesces adjacent roots with the same mode.  The
segments remain the only task schedule; maximal runs and prefix sums are local
rendering facts, not stored schedule state and not a new abstraction.

For each run:

- a static run contributes `W` packets.  Local packet `w` executes worker
  `w`'s exact static slices for all segments in that run, in segment order;
- a dynamic run with root task counts `T0, T1, ...` contributes
  `T = sum(Ti)` packets.  Its local packet ordinal is decoded by prefix sums
  into one root and one root-local ordinal, then mapped through that segment's
  normalized relation and exact dense-support certificate.

Concatenating the run ranges gives one fixed packet count

```text
P = sum(W for each static run)
    + sum(Ti for every dynamic root).
```

No packet table or run object is stored in `StaticPipelinePlan`.  The emitted
decoder directly uses segment-derived constant prefix ranges and the existing
`scheduled_root_task_body`.  Kernel-scoped/TMEM roots stay in the kernel body;
legal root helpers retain their current inlining/outlining decisions.
When the proved segment traversal matches the configured canonical PID order,
the dynamic local ordinal is that PID directly; codegen omits the otherwise
redundant logical-coordinate round trip.  Permuted or nonrepresentable
traversals still query the authoritative normalized relation.

There is one generic code-shape boundary for heterogeneous Blackwell kernels.
If a packet stream contains a kernel-scoped/TMEM branch, keep all branches
through the final such branch inline and place the remaining contiguous,
helper-safe packet suffix behind one guarded noinline helper.  This preserves
TMEM legality while isolating the suffix's live ranges from the kernel-scoped
body.  If every branch is helper-safe, leave the selector inline; if the final
branch requires kernel scope, there is no suffix to outline.  This is derived
solely from the existing kernel-scope legality classification, not a workload,
root-count, or task-count heuristic, and it does not change packet order.

In a non-folded packet stream, a static packet's decoded logical worker `w` is
the sole worker identity used for task slicing, epoch/state indexing, waits,
and publication.  Physical `program_id`/SM identity must not leak into its
semantics.  Static bodies therefore need the same role-relocatability audit as
dynamic bodies, even though they execute a coarser strand.

### Ordered one-shot admission

Launch exactly `P` CTAs.  Every CTA performs one relaxed atomic increment on a
single persistent `uint64` packet cursor, decodes the returned packet role,
executes that complete role through waits, body, inline continuation, and
publications, then retires.  The cursor determines logical admission order;
CUDA block IDs do not.

One invocation consumes exactly `P` cursor values.  With
`base(epoch) = (epoch - 1) * P`, every claim in that invocation lies in
`[base(epoch), base(epoch) + P)`, `packet = raw % P`, and
`epoch = raw // P + 1`.  `P` is invariant for the cubin.  Runtime sequence
lengths, routing, and masks may turn a capacity packet into a no-op, but it
still executes all required publications.  Different runtime metadata can
replay without reset; changing `P` requires another compilation.  Concurrent
launches may not share the same persistent state lease.  Overflow follows the
existing persistent-state lifetime contract.

When every retained segment is static, the complete schedule is one all-static
run, `P == W`, and every packet is the corresponding ordinary worker strand.
Codegen must strength-reduce
`packet` to `program_id`, retain the existing per-worker epoch protocol, and
omit the cursor state and atomic.  This is an optimization of the identical
packet semantics, not a second scheduler.  It is the strict Qwen/Gemma
compatibility path.  Test the semantic all-static predicate directly; never
fold merely because an unrelated dynamic/mixed packet count happens to equal
`W`.  Ignored continuation config entries do not prevent the fold.

For any mixed schedule containing a static run, prove that the compiled kernel
can simultaneously residently support all `W` static worker packets.  The grid
may contain more than `W` total packets; prior dynamic packets drain and make
room until the complete static cohort is active.  Never silently clamp or
reinterpret `W`.  An all-dynamic schedule needs no static-cohort residency
proof beyond the backend's ordinary launch/resource constraints.

### Root-barrier publication

Generalize the existing source-ticket publication rule rather than inventing
a new barrier path:

- a static root publishes once from each proved final participating worker;
- a continuation root publishes through its continuation instances; and
- a dynamic root with an outgoing root barrier publishes once per completed
  fixed-capacity task, including a masked/no-op task, with exact arrival count
  equal to that root's task capacity.

Rename `source_stage_arrival_count` to `dynamic_task_arrival_count`.  Codegen
consumes the frozen `RootBarrierPublicationPlan`; it does not recompute mode or
arrival mass.  Per-task publication is used only for root-barrier fallback;
ordinary exact readiness events retain their existing publication sites.
Every accepted root has positive compiled capacity, so every root-barrier
publisher is an ordinary static owner, continuation, or dynamic task.  Runtime
masked/no-op tasks still publish exactly like active tasks.

## Progress and correctness proof

Correctness and liveness are separate obligations.

### Exact-once correctness

For every non-continuation root:

1. the segment traversal is a total function from dense ordinals to logical
   tasks;
2. its inverse is single-valued and total over the root domain;
3. a static run's `W` strand packets partition every static segment exactly
   once, while a dynamic run contributes exactly one packet for every root
   ordinal; and
4. all waits/publications are derived from the unchanged `ReadinessGraph`.

Continuation contraction must cover its complete consumer root exactly once.
The union of ordinary and continuation-owned roots must equal the configured
root set.

### Progress invariant

After recursively contracting continuation chains, the one-segment root tuple
must be a strict topological order for every cross-root root-entry wait,
nested-checkpoint wait, and root-barrier prerequisite.  Reject a backward or
unsupported cross-root edge.
A dynamic root with same-root inter-CTA waits is ineligible unless exact
producer-ticket-before-consumer-ticket precedence is proved; the initial
implementation rejects dynamic dispatch for every such root.  Static execution is
accepted only if its existing strand/rank proof independently proves progress;
otherwise the configuration is rejected rather than silently forced static.

For a consumer packet that has been issued:

- a producer in the same static run is covered by the existing exact
  static-strand/rank proof and the bounded `W`-packet cohort;
- every one of the `W` owner packets in an earlier static run was issued before
  the cursor crossed that run's packet range; or
- every producer in an earlier dynamic root has already been claimed before a
  packet cursor can cross that root's interval.

Claimed does not mean completed.  Exact counters still gate completion and
visibility.  An earlier packet is active on a resident CTA, completed, or
waiting only on a still-earlier packet/root.  A CTA retires only after its
complete dynamic task or static worker strand, inline continuation, and all
publications finish.  A minimal-unfinished-packet induction therefore reaches
runnable work and rules out a wait cycle.

This proof covers all four transitions:

- static -> static: consecutive roots share one static run and use the existing
  `W`-strand proof;
- static -> dynamic: every static owner packet is issued before the cursor
  reaches a dynamic consumer packet, although some owners may still run;
- dynamic -> static: every producer task packet is issued before the first
  packet of the `W`-owner static cohort; and
- dynamic -> dynamic: the monotone packet cursor crosses a root interval only
  after every earlier-root task packet is issued.

A static cohort may initially be admitted behind unfinished dynamic packets.
Those earlier packets cannot depend backward and therefore drain.  Because the
kernel is proved capable of residently holding all `W` static packets, the
complete cohort eventually becomes active; no subset of waiting static owners
can permanently exclude an unissued peer.

Nested waits use the owning consumer root in the same induction.  A
final-arrival continuation executes only after its exact event completes, so
contracting it into the final producer preserves strict order.  After dispatch
modes are applied, revalidate that each continuation consumer is covered once,
its trigger event is exact, and that trigger's covered obligations contain
every semantic dependency obligation of the continuation root.  This last
condition is mandatory for a dynamic trigger: the continuation executes
inside its producer packet before the cursor necessarily crosses the producer
root, so it may not wait for an unissued sibling packet through a second
prerequisite.  Every continuation chain must terminate in ordinary resident
work.  The continuation body and all of its publications finish before the
producer CTA retires.  No validator may weaken a wait or appeal
to likely CTA launch order.

Dynamic eligibility also requires an exact dense root traversal and a root body
whose semantics use logical PID coordinates rather than a physical worker/SM
identity.  A TMEM or other kernel-scoped body may remain inline in the entry
kernel; it need not be outlineable.  An unsupported physical-identity
dependency or same-root coordination makes the dynamic choice illegal.  An
explicitly requested illegal `dynamic` mode rejects the config; only the
field's default selects static implicitly.  There is no partial-root dynamic
escape hatch.

The launch grid may exceed resident capacity, but a mixed schedule's `W`-packet
static cohort must fit simultaneously.  The proof never assumes that an
unissued later packet will rescue progress; it relies only on already-issued
earlier packets and the eventual admission of the bounded `W`-packet static
cohort as preceding acyclic work drains.

## Why root is the mode boundary

A list-schedule segment is merely a contiguous occurrence created by
cross-root placement; it is not a semantic task family.  Choosing execution
mode on those occurrences would make the knob unstable and would require the
list scheduler to exist before the mode could be interpreted.

After list placement is removed, every root has exactly one segment.  Root and
segment boundaries therefore coincide.  Root is the safe semantic boundary
because it has:

- one exact logical task domain and traversal;
- one set of incoming readiness obligations;
- one publication policy; and
- one stable source-level identity for autotuning.

Maximal same-mode runs are derived only to choose packet granularity and reduce
dispatch overhead.  They never change this root-level meaning.

## Implementation roadmap

### Phase 0: reject the fixed resident loop — complete

- The reference-only B4 probe measured 104.416 us fixed-loop versus 61.440 us
  one-shot and 67.456 us standalone, bit-exact with K22/F16.
- The loop doubled spills from 54 to 104 bytes at the same R80/49,160-byte
  shared-memory envelope.
- A common noinline packet helper is illegal for the TMEM attention root, so
  the resource failure cannot be isolated away without changing the kernel
  body boundary.
- The negative probe and result are archived at reference-only commit
  `d17e2db9`; they are not merged into production.
- Therefore delete the fixed-loop/T+W design from the roadmap rather than
  retaining it beside the fast path.

### Phase 0b: validate the one-shot packet executor — complete

- Treat the archived FlashMLA B4/B9 and Muse m8 one-shot results as the positive
  all-dynamic controls, then reproduce them through production packet lowering.
- Reconfirm that the all-static strength reduction emits the pre-refactor
  Qwen/Gemma plan and code with no cursor/atomic.  The fresh GPU1 Qwen golden is
  102.304 us for the checked-in AOT entrypoint, with identical plan/cubin across
  the compared source wrappers, R255/22-byte spill/17,408-byte shared, and
  continuations `{7, 8, 10}`.
- As an isolation-only probe, force that unchanged Qwen worker body through
  `W` one-shot CTAs with one atomic role assignment and cursor-derived epoch,
  without changing any root/task mode.  Compare full-KV correctness, plan,
  resources, and latency to direct `program_id`; run the same control for Gemma
  if practical.  Production still folds all-static execution to no atomic.
  This GPU1 control is complete: all 16 outputs and full KV were bit-exact,
  resources remained R255/22-byte spill/17,408-byte shared, and the atomic role
  assignment cost about 6.3 us (104.416--106.368 us direct versus
  110.880--112.480 us atomic).  Mixed dispatch is therefore mechanically sound
  but must recover more than that generic admission tax to be profitable; the
  ordinary autotuner makes that decision.
- The source-ticket-frontier branch has been deleted.  Qwen's all-static
  traversal remains byte-identical, while production MLA and Muse retain their
  dynamic wins with root-local order derived only from ordinary readiness.
- For FlashMLA, retain and report compact K22/fan-in16.  The exact K352/F1
  fallback previously measured about 71.52 us versus about 63.33 us compact
  and 67.49 us standalone; production is not accepted without the generic
  root-entry quotient.
- Add a synthetic or real `static -> dynamic -> static` correctness/progress
  case and a case with multiple same-mode runs.
- Run one-shot and mixed-mode ablations for DeepSeek and Nemotron, whose
  archived dynamic evidence used fixed resident loops.  Tune only the generic
  root dispatch vector and existing resource knobs.
  The all-dynamic one-shot controls are now complete: DeepSeek measured
  165.920 us versus 165.888 us fixed-loop and 155.648 us standalone; Nemotron
  measured 79.904 us versus 77.824 us fixed-loop and 61.440 us standalone.
  Both were bit-exact to the fixed-loop outputs and reference-valid.  DeepSeek
  is exact performance parity; Nemotron's 2.080-us/2.7% one-shot cost is a
  bounded performance gate for production tuning, not a second executor.

### Phase 1: install the root dispatch surface — complete

- Change `ConfigSpec.enable_cross_loop_schedule()` to accept root count.
- Add `cross_loop_root_dispatch` with the existing `ListOf(EnumFragment)`.
- Add runtime `Config` parsing, defaults, validation, serialization, and
  backend-key support.
- Remove `cross_loop_pipeline_depth` from public/runtime config and update
  configs/tests that intentionally exercise the new scheduler.
- Default to all static.

### Phase 2: collapse schedule construction — complete

- Make baseline construction produce one segment per non-continuation root.
- Retain transactional `_consumer_major_producer_order` only as a
  mode-independent root-local permutation; remove its source-ticket frontier.
- Preserve the two-step continuation transaction: order the all-static seed,
  select and assign continuations from it, then rebuild/renormalize and apply
  the same local ordering under final counters.
- Apply dispatch modes from `cross_loop_root_dispatch` after continuation
  contraction.
- Replace `_try_finalize_pipeline_proposal`'s placement-candidate cascade with
  one validation/freeze transaction plus the exact-counter fallback for nested
  quotient lowering.
- Add explicit invariants for unique segment per root and source-root order.
- Revalidate selected continuations structurally after modes are applied; mode
  selection cannot alter continuation identity.
- Generalize root-entry nested quotient progress from the singular source
  exception to the same contracted-root static/dynamic proof.  Keep finer
  static segmented quotients on their existing rank proof.

### Phase 3: lower the ordered packet stream — complete

- Reuse `scheduled_logical_task_expression` and
  `scheduled_root_task_body`; do not clone root bodies or decode PIDs again.
- Derive maximal adjacent same-mode runs locally in codegen.
- Give every static run exactly `W` worker-strand packet roles and every
  dynamic run one packet per exact task ordinal.
- Concatenate those ranges into fixed `P`, allocate one persistent `uint64`
  cursor, launch `P` CTAs, and emit one claim/one complete role per CTA.
- Decode dynamic ticket ranges through each authoritative segment traversal;
  render a static packet with the unchanged worker-strand code over its run.
- Isolate a helper-safe packet suffix after the final kernel-scoped/TMEM branch
  with one guarded noinline helper; leave all-helper-safe streams inline.
- Generalize frozen root-barrier publication from source tickets to dynamic
  tasks.
- Strength-reduce the semantic all-static case to the exact current
  `program_id`/per-worker-epoch path with no cursor or atomic.

### Phase 4: replace the progress proof — complete

- Prove exact segment traversal and coverage once.
- Prove strict source-root dependency order after recursive continuation
  contraction, plus exact ticket precedence for any admitted same-root wait.
- Prove mode-independent synchronization coverage: every semantic obligation
  remains an exact counter or root-barrier edge even when static wave order
  would have been sufficient.
- Prove root-entry counter quotient liveness from the same mixed-mode root
  theorem; do not add a dynamic/source special case.
- Prove global packet-prefix order, complete static-gang packet coverage, and
  `W`-packet resident capacity whenever a static run exists.
- Cover nested waits and root barriers from the same emitted prerequisite view.
- Delete wave/list-priority proofs that are no longer semantic.

### Phase 5: delete superseded machinery — complete

Delete, rather than leave dormant:

- `_event_frontier_list_schedule`;
- `_global_unit_list_schedule`;
- global-list criticality/slack/release-credit and affine-repeat helpers;
- `_source_ticket_candidate` and strict-partial-source detection;
- singular `_source_ticket_schedule_segment` assumptions;
- `_source_segment_ticket_order`, `_with_source_ticket_schedule_segment`, and
  `_has_valid_source_ticket_schedule`;
- `_source_ticket_frontiers` inside root-local ordering;
- source-ticket-only launch-grid/epoch branches, replacing them with the
  generic packet-prefix renderer; and
- `cross_loop_pipeline_depth` and its autotuner/config/backend plumbing.

Retain only generally required relation utilities.  Before deleting a helper,
verify whether exact readiness, continuation contraction, root-local ordering,
or nested-counter compaction still calls it.

### Phase 6: replace obsolete tests — complete

Remove tests whose contract is list priority, multi-segment root placement,
pipeline depth, or one special source root.  Preserve relation algebra tests
that exercise generally useful exactness.

Add focused tests for:

- one segment per non-continuation root and source-root ordering;
- root-indexed config defaulting/normalization;
- exact static and dynamic traversal equivalence;
- adjacent same-mode run derivation without stored run state;
- replay frames with fixed `P` and multiple runs;
- all four static/dynamic transitions;
- nested waits and continuations in both modes;
- dynamic producer -> continuation -> downstream execution (use DeepSeek root
  6 -> 7 as the real-workload gate);
- exact dynamic root-barrier arrival counts;
- conservative rejection of backward/unsupported dependencies;
- all-static lowered-code compatibility; and
- absence of the deleted config keys and special-source symbols.

### Phase 7: performance and observability validation — complete

Run every case with same-source standalone, correctness, resources, compile
time, and cold-L2 medians.  Generate standalone-on-top/persistent-on-bottom
Gantt charts for MLA B4 and B9.

| probe | required comparison |
| --- | --- |
| FlashMLA B4/Q4/H16 | all-dynamic vs archived 61.31 us, matched standalone 67.46 us, and the recorded ThunderKittens baseline |
| FlashMLA B9 ragged | all-dynamic vs archived 88.06 us and matched standalone 100.29 us |
| FlashMLA Q1/Q2 canonical endpoints | static/static against the matched two-launch boundary, covering both a root barrier and a direct continuation |
| Qwen3 checked-in pretuned decode | all-static vs a pre-refactor golden from the identical entrypoint/source; compare clean-main's different fixed-context source only as context, not as a hash/timing identity gate |
| Qwen3 batched/dynamic probe | all-static vs the fresh same-source `static_shapes=False` golden (~100.32 us GPU0 or ~104.35 us GPU7, not the historical 94-us source); same cubin across expected runtime metadata |
| Gemma4 A4B pretuned B1 | all-static vs clean-main ~49--51-us control |
| Gemma4 A4B B2 routed case | canonical all-static, all-dynamic, and tuned mixed modes vs the current ~69.54-us selected control and standalone; confirm multiple experts route.  A roughly 2-us/3% loss from retiring list placement may be accepted only if documented as the simplification tradeoff |
| Muse FFN m8 | all-dynamic vs 159.74-us archived result and 172.00-us matched standalone |
| Muse FFN m12 | informational only after retuning to a physically resident `W`; do not require the infeasible archived 1,776-worker result |
| DeepSeek-V3 MoE | all-dynamic and tuned mixed modes vs 167.87-us archived dynamic, static controls, and standalone |
| Nemotron MoE | all-dynamic and tuned mixed modes vs 77.86-us archived dynamic, static controls, and standalone; use the non-tensor-descriptor probe |

For every retained binary record registers, spills, shared memory, warps,
stages, worker count, and dispatch vector.  A schedule speedup caused by changed
fusion or numerics is invalid.

## Acceptance criteria

The redesign is complete only when all of the following hold:

1. There is one scheduler construction pipeline and one frozen
   `StaticPipelinePlan`.
2. Every scheduled root has exactly one authoritative segment.
3. Static and dynamic render the same segment traversal.
4. Arbitrary legal root-mode mixtures lower to one ordered packet stream and
   one progress proof; all-static is only its identity strength reduction.
5. `cross_loop_pipeline_depth`, global list placement, and transient/source
   ticket policy are absent from production code.
6. No workload name, root literal, fan-in literal, or task-count threshold
   selects a schedule.
7. All-static Qwen/Gemma retain main performance and lowering.
8. All-dynamic MLA/Muse retain their measured wins over standalone and do not
   materially regress the best archived executor.
9. DeepSeek/Nemotron retain at least the dynamic-root-major signal; remaining
   gaps to standalone are reported as body/resource work, not hidden.
10. Correctness, replay, root barriers, nested waits, and continuations pass
    in static, dynamic, and mixed synthetic tests; statically zero-capacity
    roots are rejected explicitly.

The all-static golden additionally requires: omitted dispatch field defaults to
all static; no cursor allocation or atomic claim appears; plan/counter/barrier
facts match the current control; Qwen retains continuations at roots 7, 8, and
10 with root 13 resident; Gemma retains root 6 resident and root 7 as its
continuation; and static nested counters remain unchanged.

The failed fixed resident claim loop is not retained.  If the ordered packet
stream cannot preserve all-dynamic MLA/Muse, all-static Qwen/Gemma, and a
correct mixed schedule, stop and revisit unification rather than adding a
second permanent scheduler behind a workload predicate.
