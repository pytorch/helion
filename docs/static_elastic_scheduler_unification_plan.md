# Static/elastic cross-loop scheduler unification

Status: canonical implementation plan.  Architecture, Qwen/Gemma, and
execution/progress reviewers signed off on 2026-09-12.  The phase-0 executor
ablation remains the deliberate performance kill gate.

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
3. Select `static` or `elastic` execution independently for each root through
   one root-indexed autotuning field.
4. Lower consecutive elastic roots with a dynamically claimed task stream over
   the same fixed resident CTA cohort used by static roots.
5. Delete the global event-frontier/list placer, its pipeline-depth knob, and
   the special one-source ticket path after the general executor passes the
   performance gates below.

The scheduler still makes important decisions: it derives exact readiness,
selects continuations, chooses a root-local task permutation, and proves
progress.  What disappears is speculative cross-root splitting and placement.
Elastic ownership supplies the load balancing that list placement previously
tried to approximate with fixed workers.

The north star is simplification.  The final compiler must have one semantic
DAG, one readiness graph, one worker schedule, one continuation identity, and
one code-generation path.  Static versus elastic is a physical execution
property of a root in that schedule, not another scheduler or plan IR.

## Evidence for the pivot

All numbers in this section are cold-L2 B200 measurements from the archived
probes.  They must be refreshed from the implementation branch before final
acceptance.

| workload | canonical elastic | list-order elastic | relevant control |
| --- | ---: | ---: | ---: |
| FlashMLA B4/Q4/H16 | 61.312 us | 61.280 us | 67.456 us standalone |
| FlashMLA B9 ragged | **88.064 us** | 90.048 us | 100.288 us standalone |
| Muse FFN, m8 | **159.744 us** | 161.728 us | 172.000 us standalone |
| Muse FFN, m12 | **161.392 us** | 163.648 us | 173.888 us standalone |
| DeepSeek-V3 MoE | 167.872 us | 167.872 us | 157.728 us standalone |
| Nemotron MoE | 77.856 us | 77.856 us | 61.504 us standalone |

In all six comparisons, elastic execution erased the list order's benefit.
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

Muse m12 is evidence about logical order and elastic load balancing, not about
a feasible fixed pool of 1,776 simultaneously resident CTAs.  That probe
overlaunched 13,688 one-shot CTAs while requiring residency for 1,184.  The
production fixed-pool gate therefore uses Muse m8; m12 must be retuned to a
physically resident width rather than introducing a second width concept.

That distinction matters.  One-task-per-CTA overlaunch is excellent for a
fully elastic graph but cannot safely express an arbitrary
`static -> elastic -> static` sequence without a second phase/ownership
protocol.  A fixed resident claim loop can express every mixture with one
progress argument.  It is therefore the intended production mechanism, but
it is not accepted on elegance alone: phase 0 must show that it preserves the
MLA/Muse benefit and the all-static Qwen/Gemma behavior.  If it cannot, stop
and revisit the design rather than retaining two permanent executors.

## Scope and non-goals

This redesign supports a fixed compiled task capacity with runtime data inside
that capacity.  In particular, the following remain runtime values:

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

CLC remains a future implementation of elastic claiming.  It must consume the
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

- every non-continuation, nonempty root has exactly one segment;
- continuation roots have no segment;
- segments occur in increasing source-root order;
- each segment's target is an exact, bijective cover of that root's tasks;
- the segment's dense logical ordinal is the only task traversal used by both
  static and elastic rendering; and
- one scalar field on the existing segment says whether that traversal is
  statically owned or elastically claimed.

Add `dispatch_mode: Literal["static", "elastic"] = "static"` to the existing
`WorkerScheduleSegment`.  This is one field on the authority we already have,
not a new abstraction or parallel plan.  All segments remain in the current
resident launch stage while the old source-ticket path is removed.  The old
stage-zero meaning, its extra launch grid, and its singular-source rules are
then deleted.  The resulting degenerate launch-stage axis may be removed in a
later mechanical cleanup; it must not be repurposed as dispatch mode.

Cross-root chronology is now the source-root order, not an ordering induced by
dispatch mode or by splitting a root into several segment occurrences.  The
worker/wave coordinates retain their exact role within a static segment and
provide the dense ordinal certificate used to decode an elastic ticket.  The
compiler must reject a segment that cannot expose that one exact traversal.

This is the single-source-of-truth rule:

```text
root task order
    -> one normalized WorkerScheduleSegment with one dispatch_mode
    -> either static striding or elastic ticket decoding
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
  -> apply per-root static/elastic choices
  -> exact coverage + progress + publication validation
  -> optional nested-counter quotient as an emitted strength reduction
  -> freeze StaticPipelinePlan
```

Details:

1. Counter and root-barrier selection is independent of dispatch mode.  Every
   `TileDependencyGraph` obligation must remain represented by an emitted exact
   counter or root-barrier fallback.  Static worker/wave order is never allowed
   to discharge an obligation, because that implication would disappear when
   either endpoint becomes elastic.
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
   statically owned or elastically claimed.
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
     elastic producers use ticket-interval precedence.  Completion remains
     gated by the compact counter.
   - A finer segmented quotient retains the existing worker-rank proof only
     when every relevant occurrence is static.

   Elastic same-root quotients decline unless exact ordinal precedence is
   added later.  Static frontiers such as Qwen's historical 74/22 split may
   never be carried into elastic execution by assumption.  On any failed
   proof, retain the exact semantic counter on the identical schedule.

There is no list-schedule proposal, candidate cascade, priority queue,
criticality class, release credit, affine-repeat policy, or pipeline-depth
search in this pipeline.

## Autotuning surface

Replace `cross_loop_pipeline_depth` with one field:

```python
cross_loop_root_dispatch: list[Literal["static", "elastic"]]
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
continuation or statically empty root is ignored; duplicate autotuner
candidates are acceptable and preferable to topology-specific knob shapes.

Expected useful settings are evidence, not compiler heuristics:

- FlashMLA B4/B9: all non-continuation roots elastic;
- Muse FFN: all non-continuation roots elastic;
- Qwen3 decode/FFN: all static;
- Gemma A4B: initially all static;
- DeepSeek-V3 and Nemotron MoE: initially all elastic, then tune mixed vectors
  if individual heavy roots prefer resident ownership.

Do not infer a mode from a model name, root number, fan-in literal, task count
threshold, or a topology-shaped candidate list.  The only compiler policy is
legality; profitability belongs to the ordinary autotuner.

## Static execution

Static lowering is the existing persistent-worker behavior.  For a segment
with `W` workers, worker `w` executes the exact segment ordinals assigned to
its strided slice.  It performs the segment's incoming waits, task-level waits,
body, publications, and root-barrier publication using the frozen plan.

All-static code generation is a strict compatibility gate:

- same root-local traversal;
- same continuations and counters;
- same root-body inlining/outlining decisions;
- same launch grid and persistent state; and
- byte-identical lowered Triton whenever unrelated naming cleanup does not
  prevent it.

This is how Qwen and Gemma keep their proven resident behavior while the
global list machinery is removed.

## Elastic execution

### Physical pool

Launch exactly the existing `W` persistent CTAs, and prove that schedule width,
launch-grid width, and the simultaneously resident cohort are the same `W`.
Do not silently clamp or reinterpret `W` in mixed mode.  Every CTA keeps its stable
worker ID and epoch for the whole kernel.  At an elastic region, those same
CTAs repeatedly claim logical task ordinals from an atomic cursor.  No extra
CTA is launched per task, and no static worker role is replaced.

This one physical pool is what makes arbitrary mode sequences composable.
Static and elastic are two ways for the same resident workers to consume the
same segment traversal.

### Derived elastic runs

Codegen coalesces adjacent elastic root segments into a maximal elastic run.
This is a local rendering optimization, not stored schedule state and not a
new abstraction.  Run boundaries are derived by scanning the final segment
tuple; static roots and the ends of the schedule delimit runs.

For a run with root task counts `T0, T1, ...`, concatenate their exact segment
ordinals in root order.  A ticket is decoded by prefix sums into one root and
one root-local ordinal, then passed through that segment's existing
`logical_task_order`.  The ordinary `scheduled_root_task_body` emits waits,
the unchanged root body, readiness publication, and continuation handling.

Adjacent roots share a cursor so a worker can begin a ready downstream root as
soon as all earlier-root tickets have been claimed.  This both preserves MLA
overlap and avoids `W` terminal atomic operations per small root.  A static
root intentionally terminates the run.

### Replay-safe cursor protocol

For a run containing the compile-time-invariant capacity
`T = sum(Ti)` tasks, one invocation consumes exactly
`T + W` cursor values:

```text
0 .. T-1       task tickets
T .. T+W-1     one terminal ticket for each resident CTA
```

Each CTA loops until it claims one terminal ticket, then proceeds to the next
segment/run.  Because a CTA exits after its first terminal claim, exactly `W`
terminal claims occur and the next invocation begins on the next complete
frame.  If `base(epoch) = (epoch - 1) * (T + W)`, every claim in that invocation
must fall in `[base(epoch), base(epoch) + T + W)`.  Store one persistent
`uint64` cursor per maximal elastic run.  Use the existing per-worker epoch as
the sole readiness epoch; cursor arithmetic must agree with that epoch but
must not become a second epoch source.

Every one of the `W` CTAs must visit every elastic run exactly once, even if it
owned no tasks in the preceding static root.  There may be no early return or
conditional path around the claim loop.  The protocol is legal only while `T`
is invariant for the cubin and every invocation consumes a complete frame.
Runtime sequence lengths, routing, and masks may make a fixed-capacity task a
no-op, but that task must still claim its ticket and perform every required
publication.  Different runtime metadata may replay without resetting the
cursor; changing `T` requires another compilation.  Overlapping launches may
not share the same persistent state lease.  Integer overflow follows the
existing persistent epoch/state lifetime contract and must be covered by an
explicit test or bound.

The cursor atomic is relaxed.  Readiness publication/waits retain their
existing release/acquire semantics.

### Root-barrier publication

Generalize the existing source-ticket publication rule rather than inventing
a new barrier path:

- a static root publishes once from each proved final participating worker;
- a continuation root publishes through its continuation instances; and
- an elastic root with an outgoing root barrier publishes once per completed
  fixed-capacity task, including a masked/no-op task, with exact arrival count
  equal to that root's task capacity.

Rename `source_stage_arrival_count` to `elastic_task_arrival_count`.  Codegen
consumes the frozen `RootBarrierPublicationPlan`; it does not recompute mode or
arrival mass.  Per-task publication is used only for root-barrier fallback;
ordinary exact readiness events retain their existing publication sites.
For a statically empty root, retain the existing vacuous/synthetic single
arrival.  Because an empty root has no segment or elastic terminal, the frozen
publication plan assigns that arrival once to resident worker 0 at the root's
canonical position in the top-level schedule loop (or to an exactly equivalent
initialization site).  Empty-root ownership is explicit publication metadata,
not an invented task segment.

## Progress and correctness proof

Correctness and liveness are separate obligations.

### Exact-once correctness

For every non-continuation root:

1. the segment traversal is a total function from dense ordinals to logical
   tasks;
2. its inverse is single-valued and total over the root domain;
3. static striding partitions those ordinals exactly once, or the elastic
   cursor issues every ordinal exactly once; and
4. all waits/publications are derived from the unchanged `ReadinessGraph`.

Continuation contraction must cover its complete consumer root exactly once.
The union of ordinary and continuation-owned roots must equal the configured
root set.

### Progress invariant

After recursively contracting continuation chains, the one-segment root tuple
must be a strict topological order for every root-entry wait, nested-checkpoint
wait, and root-barrier prerequisite.  Reject a backward or unsupported edge.
An elastic root with same-root inter-CTA waits is ineligible unless exact
producer-ticket-before-consumer-ticket precedence is proved; the initial
implementation rejects elasticity for every such root.  Static execution is
accepted only if its existing strand/rank proof independently proves progress;
otherwise the configuration is rejected rather than silently forced static.

For a consumer that can begin:

- every producer in an earlier static root has a stable owner among the same
  resident `W` CTAs; or
- every producer in an earlier elastic root has already been claimed before a
  cursor can cross that root's ticket interval.

Claimed does not mean completed.  Exact counters still gate completion and
visibility.  A claimed producer is active on a resident CTA, completed, or
waiting only on a strictly earlier contracted root; the minimal-unfinished-root
induction supplies eventual progress.  A worker finishes the task and its
publications before making its next claim.  A CTA
waiting in a later root cannot evict or prevent the resident CTA that owns an
earlier producer from running.  Induction over strict source-root order then
rules out a wait cycle.

This proof covers all four transitions:

- static -> static: existing persistent strand proof;
- static -> elastic: early workers may claim consumers while slower static
  producer owners continue;
- elastic -> static: a worker exits the elastic interval only after every
  producer ticket has been claimed; and
- elastic -> elastic: the shared monotone cursor crosses the root boundary
  only after every earlier-root ticket has been claimed.

Nested waits use the owning consumer root in the same induction.  A
final-arrival continuation executes only after its exact event completes, so
contracting it into the final producer preserves strict order.  After dispatch
modes are applied, revalidate that each continuation consumer is covered once,
its trigger event is exact, and that trigger's covered obligations contain
every semantic dependency obligation of the continuation root.  This last
condition is mandatory for an elastic trigger: the continuation executes
inside its producer task before the cursor necessarily crosses the producer
root, so it may not wait for an unclaimed sibling ticket through a second
prerequisite.  Every continuation chain must terminate in ordinary resident
work.  The continuation body and all of its publications finish before the
producer CTA claims another ticket.  No validator may weaken a wait or appeal
to likely CTA launch order.

Elastic eligibility also requires an exact dense root traversal and a
relocatable root body expressed through logical PID coordinates rather than a
physical worker/SM identity.  Unsupported kernel-scoped identity or same-root
coordination makes the elastic choice illegal.  An explicitly requested
illegal `elastic` mode rejects the config; only the field's default selects
static implicitly.  There is no partial-root elastic escape hatch.

The launch grid must remain no larger than the compiled-kernel resident
capacity, exactly as required by the current static persistent path.  The
proof never assumes that a later, not-yet-launched block will rescue progress.

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

Maximal elastic runs are derived only to reduce dispatch overhead.  They never
change this root-level meaning.

## Implementation roadmap

### Phase 0: settle the physical executor

- Add or adapt a reference-only generated-Triton ablation that runs the exact
  MLA B4/B9 and Muse root-major task streams with a fixed `W`-CTA claim loop.
- Compare it directly with the archived one-task-per-CTA executor, using the
  same source/config, numerics, cold-L2 protocol, logical order, and outlined
  per-ticket helper boundary.  The loop may change registers/spills, so record
  and compare actual cubin resources rather than requiring equality.
- Use Muse m8 as the fixed-pool gate.  Retune m12 only within a physically
  resident `W`; do not add a second physical-width abstraction.
- Include a synthetic or real `static -> elastic -> static` case and a case
  with two elastic runs, so endpoint-only results cannot mask a mixed-mode
  progress flaw.
- Reconfirm all-static Qwen/Gemma controls.
- Compare exact root-local traversals with and without the current
  source-ticket-frontier branch before deleting that branch.
- For FlashMLA, report and benchmark both the compact K22/fan-in16 root-entry
  counter and exact K352/fan-in1 fallback under the fixed pool.  The archived
  elastic win used K22/F16; prior exact K352/F1 measurements were about
  71.52 us versus about 63.33 us compact and 67.49 us standalone.  Phase 0 is
  not passed unless the generic mixed-mode quotient recovers K22/F16.
- Accept the fixed-pool design only if MLA B4/B9 and Muse retain their
  standalone wins and remain within roughly 2 us (or 3%, whichever is larger)
  of the best archived elastic result.  Otherwise stop and redesign; do not
  preserve both executors as permanent policy.

### Phase 1: install the root dispatch surface

- Change `ConfigSpec.enable_cross_loop_schedule()` to accept root count.
- Add `cross_loop_root_dispatch` with the existing `ListOf(EnumFragment)`.
- Add runtime `Config` parsing, defaults, validation, serialization, and
  backend-key support.
- Remove `cross_loop_pipeline_depth` from public/runtime config and update
  configs/tests that intentionally exercise the new scheduler.
- Default to all static.

### Phase 2: collapse schedule construction

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
  exception to the same contracted-root static/elastic proof.  Keep finer
  static segmented quotients on their existing rank proof.

### Phase 3: lower elastic runs

- Reuse `scheduled_logical_task_expression` and
  `scheduled_root_task_body`; do not clone root bodies or decode PIDs again.
- Derive maximal adjacent elastic runs locally in codegen.
- Allocate one persistent `uint64` cursor per run.
- Emit the `T + W` replay frame and worker claim loop.
- Decode ticket ranges through each authoritative segment traversal.
- Generalize frozen root-barrier publication from source tickets to elastic
  tasks.
- Preserve the exact all-static generated path.

### Phase 4: replace the progress proof

- Prove exact segment traversal and coverage once.
- Prove strict source-root dependency order after recursive continuation
  contraction, plus exact ticket precedence for any admitted same-root wait.
- Prove mode-independent synchronization coverage: every semantic obligation
  remains an exact counter or root-barrier edge even when static wave order
  would have been sufficient.
- Prove root-entry counter quotient liveness from the same mixed-mode root
  theorem; do not add an elastic/source special case.
- Prove cursor interval order and fixed resident capacity.
- Cover nested waits and root barriers from the same emitted prerequisite view.
- Delete wave/list-priority proofs that are no longer semantic.

### Phase 5: delete superseded machinery

Delete, rather than leave dormant:

- `_event_frontier_list_schedule`;
- `_global_unit_list_schedule`;
- global-list criticality/slack/release-credit and affine-repeat helpers;
- `_source_ticket_candidate` and strict-partial-source detection;
- singular `_source_ticket_schedule_segment` assumptions;
- `_source_segment_ticket_order`, `_with_source_ticket_schedule_segment`, and
  `_has_valid_source_ticket_schedule`;
- `_source_ticket_frontiers` inside root-local ordering;
- source-ticket-only launch-grid and epoch branches; and
- `cross_loop_pipeline_depth` and its autotuner/config/backend plumbing.

Retain only generally required relation utilities.  Before deleting a helper,
verify whether exact readiness, continuation contraction, root-local ordering,
or nested-counter compaction still calls it.

### Phase 6: replace obsolete tests

Remove tests whose contract is list priority, multi-segment root placement,
pipeline depth, or one special source root.  Preserve relation algebra tests
that exercise generally useful exactness.

Add focused tests for:

- one segment per non-continuation root and source-root ordering;
- root-indexed config defaulting/normalization;
- exact static and elastic traversal equivalence;
- adjacent elastic-run derivation without stored run state;
- replay frames with `T`, `W`, empty roots, and multiple runs;
- all four static/elastic transitions;
- nested waits and continuations in both modes;
- elastic producer -> continuation -> downstream execution (use DeepSeek root
  6 -> 7 as the real-workload gate);
- exact elastic root-barrier arrival counts;
- conservative rejection of backward/unsupported dependencies;
- all-static lowered-code compatibility; and
- absence of the deleted config keys and special-source symbols.

### Phase 7: performance and observability validation

Run every case with same-source standalone, correctness, resources, compile
time, and cold-L2 medians.  Generate standalone-on-top/persistent-on-bottom
Gantt charts for MLA B4 and B9.

| probe | required comparison |
| --- | --- |
| FlashMLA B4/Q4/H16 | all-elastic vs archived 61.31 us, matched standalone 67.46 us, and the recorded ThunderKittens baseline |
| FlashMLA B9 ragged | all-elastic vs archived 88.06 us and matched standalone 100.29 us |
| Qwen3 checked-in pretuned decode | all-static vs a pre-refactor golden from the identical entrypoint/source; compare clean-main's different fixed-context source only as context, not as a hash/timing identity gate |
| Qwen3 batched/dynamic probe | all-static vs the fresh same-source `static_shapes=False` golden (~100.32 us GPU0 or ~104.35 us GPU7, not the historical 94-us source); same cubin across expected runtime metadata |
| Gemma4 A4B pretuned B1 | all-static vs clean-main ~49--51-us control |
| Gemma4 A4B B2 routed case | canonical all-static, all-elastic, and tuned mixed modes vs the current ~69.54-us selected control and standalone; confirm multiple experts route.  A roughly 2-us/3% loss from retiring list placement may be accepted only if documented as the simplification tradeoff |
| Muse FFN m8 | all-elastic vs 159.74-us archived result and 172.00-us matched standalone |
| Muse FFN m12 | informational only after retuning to a physically resident `W`; do not require the infeasible archived 1,776-worker result |
| DeepSeek-V3 MoE | all-elastic and tuned mixed modes vs 167.87-us archived dynamic, static controls, and standalone |
| Nemotron MoE | all-elastic and tuned mixed modes vs 77.86-us archived dynamic, static controls, and standalone; use the non-tensor-descriptor probe |

For every retained binary record registers, spills, shared memory, warps,
stages, worker count, and dispatch vector.  A schedule speedup caused by changed
fusion or numerics is invalid.

## Acceptance criteria

The redesign is complete only when all of the following hold:

1. There is one scheduler construction pipeline and one frozen
   `StaticPipelinePlan`.
2. Every scheduled root has exactly one authoritative segment.
3. Static and elastic render the same segment traversal.
4. Arbitrary legal root-mode mixtures share one fixed resident CTA pool and
   one progress proof.
5. `cross_loop_pipeline_depth`, global list placement, and transient/source
   ticket policy are absent from production code.
6. No workload name, root literal, fan-in literal, or task-count threshold
   selects a schedule.
7. All-static Qwen/Gemma retain main performance and lowering.
8. All-elastic MLA/Muse retain their measured wins over standalone and do not
   materially regress the best archived executor.
9. DeepSeek/Nemotron retain at least the dynamic-root-major signal; remaining
   gaps to standalone are reported as body/resource work, not hidden.
10. Correctness, replay, root barriers, nested waits, and continuations pass
    in static, elastic, and mixed synthetic tests.

The all-static golden additionally requires: omitted dispatch field defaults to
all static; no cursor allocation or atomic claim appears; plan/counter/barrier
facts match the current control; Qwen retains continuations at roots 7, 8, and
10 with root 13 resident; Gemma retains root 6 resident and root 7 as its
continuation; and static nested counters remain unchanged.

If the fixed resident claim loop cannot satisfy both the elastic and static
performance gates, the implementation pauses at phase 0.  The answer is not a
second permanent scheduler hidden behind an MLA predicate.
