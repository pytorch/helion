# Minimal cross-loop scheduler redesign

## Status

This is the living design and implementation record for
`/home/eche/local/helion-scheduler-redesign`, based on Helion `origin/main` at
`f28a94dc`. The ordered-segment lowering, bounded list-schedule proposer,
symbolic acceptance proofs, per-worker publication plan, and transient-source
lowering are implemented. The B4 and B9 FlashMLA gates pass; broader Qwen and
MoE rollout now includes a positive Nemotron result plus Qwen3, Gemma A4B, and
DeepSeek-V3 no-regression controls.

The plan intentionally does **not** introduce a new hierarchy of persistent
schedule IRs. It extends the existing `WorkerSchedule` contract so that the
schedule the compiler computes is the schedule codegen executes.

## Summary

The minimum general change is:

```text
TileDependencyGraph                         existing
        |
ReadinessGraph                              existing
        |
global, topology-only ready-list placement  replace root-local placement
        |
WorkerSchedule.segments                     existing type, tuple order is now executable
        |
cross_loop_codegen                          emit segments in tuple order, not root order
```

Per-worker strands are implicit: a worker's strand is the subsequence of the
globally ordered segments whose worker range contains it. We do not need a
separate `SymbolicStrandSchedule`, `StaticRunProgram`, or literal instruction
array per worker.

This preserves the functionality needed for FlashMLA, Qwen3 decode, Gemma
A4B, DeepSeek MoE, and Nemotron while making the implementation substantially
less invasive.

## Applicability and gating

Nothing is gated on FlashMLA or a model name.

There are three progressively narrower generic decisions:

1. **Ordered-segment lowering** applies to every compatible
   `static_pipeline` plan. A source-ordered schedule lowers identically; an
   interleaved schedule finally retains its chronology.
2. **Global ready-list movement** is proposed from the exact emitted
   prerequisite descriptors for counters and root barriers. Acceptance proves
   every producer arm independently on its exact support; it never hides
   disjoint partial arms behind a synthetic total frontier. Unsupported
   relations retain the conservative schedule.
3. **Transient-source execution** applies only when the graph has one
   oversubscribed wait-free source, a safe ticket order, and exactly one
   requested resident strand per physical visible SM. Source tickets execute
   outside the resident `WorkerSchedule`; exact counters decide when resident
   work becomes useful. There is no modeled admission width or source-tail
   placement. FlashMLA is the first known positive case, not a compiler
   special case.

Plans with final-arrival continuations currently retain their existing
placement instead of entering global list scheduling. Graphs above the current
4,096-CTA proposal bound, including the roughly 5,000-CTA Qwen3 decode probe,
also retain the conservative plan. These are explicit fallbacks, not model
gates. The tested Nemotron routed/shared graph and Gemma A4B unfused graph use
the generic path; DeepSeek-V3 and full Qwen3 retain their existing continuation
plans.

## Non-negotiable constraints

- One persistent Triton kernel.
- No change to user-visible fusion boundaries.
- No change to numerical algorithms or dtypes.
- No latency or resource cost model.
- No measured/profile-guided scheduling inputs.
- No FLOP, byte, register, shared-memory, or bandwidth estimates.
- No CLC or runtime ready queue in the initial implementation.
- No schedule catalogue or public policy selector.
- No public per-stage admission-width knob.
- No root IDs, model names, or shape constants in compiler logic.
- Exact dependency counters remain authoritative at runtime.
- Unsupported relations monotonically fall back to conservative barriers.
- If any optimized-plan proof fails, rebuild the complete current-main plan;
  do not keep partially changed ownership or synchronization decisions.

The existing kernel configuration still selects worker count, warps, stages,
and register limits. The scheduler consumes the selected worker count but does
not try to predict the performance of those settings.

## Why even this smaller refactor is necessary

### Current mismatch

Before this redesign, `WorkerScheduleSegment` mapped tasks to
`(worker, worker_step)`, but `cross_loop_codegen.py` grouped segments back by
root and emitted all roots in source order:

```text
for phase:
    for root in numerical order:
        emit every segment for root
```

`_family_placements_at_worker_step` consequently forbade a later root from
passing an earlier root because that lowerer could not represent it. The
ordered-segment implementation removes this limitation for the global path;
the helper remains only for conservative fallback placement.

### B9 evidence

The named representative B9 trace measured 97.344 us persistent versus 95.136
us standalone. It showed real overlap:

- attention ended at 66.976 us;
- radix work began at 36.448 us;
- 184 of 188 radix tasks were producer-ready before attention ended;
- 170 started and 107 finished before attention ended; and
- request 7's late reduction was ready at 63.2 us but started at
  77.952--91.424 us.

The 28.224-us worst admission delay was head-of-line blocking in fixed
root-major strands, not missing readiness information. The scheduler already
knew enough to do better; the pre-redesign codegen could not express its
desired chronology.

## What stays unchanged

### `TileDependencyGraph`

It remains the semantic source of producer/consumer ordering. Scheduling may
choose where legal work runs, but may not invent or weaken dependencies.

### `ReadinessGraph`

It remains the symbolic representation of:

- readiness-key domains;
- producer tasks and nested publication sites;
- exact producer-to-key and consumer-to-key relations;
- exact arrival cardinalities where derivable; and
- covered `DependencyObligation`s.

### Synchronization mechanisms

Keep the existing mechanisms:

- exact readiness counters;
- final-arrival continuations; and
- root-barrier fallback.

The redesign changes task placement and executable order. It does not invent a
new synchronization primitive.

### `WorkerScheduleSegment`

Keep the current compact affine mapping:

```python
WorkerScheduleSegment(
    root,
    task_order,
    worker_begin,
    worker_count,
    dispatch_offset,
)
```

It already represents a root slice without materializing one entry per CTA.
The important contract change is that the tuple position of a segment in
`WorkerSchedule.segments` becomes executable program order.

## The revised `WorkerSchedule` contract

`WorkerSchedule` remains the only schedule representation passed to codegen:

```python
@dataclasses.dataclass(frozen=True)
class WorkerSchedule:
    worker_count: int
    segments: tuple[WorkerScheduleSegment, ...]
```

The contract becomes:

1. `segments` is a global topological sequence of executable runs.
2. Worker `w` executes exactly the subsequence containing `w`.
3. That subsequence is worker `w`'s strand and its order is authoritative.
4. Workers skip segments in which they do not participate; skipping is not a
   barrier.
5. A segment may contain several strided tasks for each participating worker.
6. If another segment must occur between two waves, the first segment must be
   split.
7. Root number never reconstructs chronology.

For one segment and dense task-order index `i`, the existing mapping remains:

```text
dispatch = dispatch_offset + i
worker = worker_begin + dispatch % worker_count
worker_step = dispatch // worker_count
```

`worker_step` remains useful to the topology-only scheduler, but codegen order
comes from the segment tuple. The validator requires worker steps to be
monotone along every worker's tuple subsequence. A schedule that cannot satisfy
both contracts is rejected.

This single representation provides the semantics of per-worker instruction
strands without adding a second persistent IR or generating 148 literal worker
programs.

## Detailed compiler pipeline

### 1. Port only the dependency-analysis prerequisites

Clean main does not yet derive every relation needed by the FlashMLA probes.
Before changing scheduling, port and independently test only these proven
relation-analysis deltas from `helion-compiler-mla`:

- bounded affine/modulo simplification;
- guarded singleton-target-axis inverse normalization;
- projection of task-local axes before taking a nested-event converse;
- partial-source readiness relations;
- strict-subset one-key counters; and
- exact fixed-scalar alias regions.

These are semantic relation improvements, not scheduling policy. Each must
retain conservative failure when its preconditions are not proved.

### 2. Build `ReadinessGraph` as current main does

Continue deriving exact events from `TileDependencyGraph`. Retain every
dependency obligation not covered by an exact event so it can become a root
barrier later.

No worker placement or priority belongs in this stage.

### 3. Select existing ownership mechanisms

Run current continuation analysis first:

- eligible narrow consumers may remain final-arrival continuations;
- continuation-owned tasks are removed from ordinary worker placement; and
- all other roots remain candidates for resident scheduling.

The first implementation does not globally reorder roots containing unresolved
or unprojectable nested-site waits. It leaves their current placement and
internal loop-split counter lowering intact. These roots become fixed anchors
in the ordered segment list. A nested-wait root whose complete prerequisites
can be projected exactly to the owning CTA remains movable at owning-CTA
granularity; the scheduler moves the CTA, not pieces of its live execution.

This restriction preserves existing nested-loop functionality without adding
a preemptive task/micro-op IR. A live CTA cannot be split into separately
scheduled pieces merely because it waits inside a loop.

### 4. Build one emitted-prerequisite view

`ReadinessGraph` describes semantic possibilities; the selected counter and
barrier plans describe what codegen will actually emit. The implementation
canonicalizes those mechanisms into one cached sequence of prerequisite
descriptors. A descriptor is either:

- one whole-root barrier edge; or
- one exact-counter consumer together with all of that counter's independently
  proved producer arms.

The bounded candidate generator and the symbolic acceptance proof consume the
same descriptor sequence. Continuation consumers are excluded in exactly one
place. This prevents proposal and proof from silently interpreting the emitted
dependency set differently.

### 5. Generate a bounded topology-only candidate

There is no cost model. Every logical CTA has unit structural weight. For
graphs with at most 4,096 CTAs, the current proposer materializes the emitted
prerequisite DAG, computes unit-depth bottom levels and earliest starts, and
uses this deterministic priority:

```text
least structural slack
then greatest bottom level
then stable source root/task order
```

At each abstract worker step it selects up to `W` ready tasks across all roots.
Selected adjacent tasks from the same source segment are compressed into a
`WorkerScheduleSegment`; compatible runs are merged. For transient-source
candidates, source tasks are external producers and are not placed in the
resident program. Resident tasks whose only unsatisfied producer is that
source enter the proposal queue immediately; their emitted runtime waits still
guard actual execution.

This CTA enumeration is only a proposal mechanism. Graphs above the bound,
relations outside the materializer, continuation-bearing plans, and candidates
that exceed the segment limit retain the conservative schedule. There is no
`_ReadySlice`, affine-front generator, or post-hoc segment-DAG linearizer in
the current implementation.

### 6. Accept only a symbolic certificate

Materialized candidate facts are not trusted for acceptance. The candidate
segments are retained, but every legality fact is re-derived from their
ordered tuple and symbolic relations:

1. Every segment task order is a total function into its root domain.
2. Segment cardinalities sum to the exact root-domain cardinality for every
   resident root. A transient source has no resident segment.
3. The union of segment converses is a total logical-CTA-to-scheduled-ordinal
   function. Together with equal cardinality, this proves that every CTA is
   executed exactly once: no duplicate and no omitted computation.
4. Worker support and worker-step monotonicity are proved with interval
   arithmetic, without enumerating workers or tasks.
5. Producer ranks come from that one accepted traversal. Every producer arm
   is checked independently on its exact partial support.
6. Resident semantic edges strictly increase worker-step rank. In a mixed
   producer join, every resident arm is checked independently.
7. A transient source has a symbolic bijection from logical tasks to tickets
   `[0, P)`, has no incoming prerequisite, and has no resident segments.
   Source-to-resident arms alone use earlier ticket-role order for progress;
   their emitted counters or barriers still prove completion and visibility.

Tests make `CoordinateRelation.materialize/targets` and diagnostic worker
enumerators raise while acceptance runs. Enumeration is therefore
mechanically excluded from the proof boundary.

### 7. Keep one source of truth

The ordered `WorkerSchedule.segments` tuple is the program. All other schedule
objects are cached symbolic views of that tuple:

- `_ScheduledRootTraversal` owns segment ordinal ranges, the forward
  scheduled-ordinal-to-logical-task mapping when representable as one
  relation, and the exact inverse used for ownership and rank proofs. Codegen
  consumes this certificate directly; for an existing PID permutation that
  cannot be collapsed into one relation, it renders the certificate's segment
  ranges rather than rebuilding prefixes.
- `RootBarrierPublicationPlan` performs one reverse interval scan. Both the
  resident arrival count and every publication site consume that same exact
  partition.
- The emitted-prerequisite descriptors feed both candidate DAG construction
  and symbolic progress validation. Transient-source selection and legality
  consume that same descriptor sequence rather than rebuilding graph edges.

The list scheduler appends segments in executable chronology and merges only
adjacent compatible runs. There is no second schedule, no per-worker task
array, and no later topological sort that can change the order. Failure of any
symbolic ownership, rank, readiness, or transient proof discards the candidate
and leaves the conservative plan intact.

### 8. Finalize synchronization against the chosen order

Every concrete producer-consumer ordering in every
`DependencyObligation` must be covered by at least one proved mechanism:

- existing exact counter wait/arrival;
- existing final-arrival continuation;
- conservative root barrier.

Redundant coverage is legal. Partial discharge is legal only if relation
algebra proves that the selected mechanisms completely partition the
obligation. Initially, do **not** remove an existing exact counter merely
because some producer/consumer pairs share a strand. Same-worker order helps
prove progress, but it does not replace dependency synchronization.

Coverage is recomputed from what codegen will actually emit.

#### Exact counters

Keep the current key domains, arrival cardinalities, waits, publications, and
epoch semantics. Reordering changes when a consumer reaches its wait, not what
makes the wait complete.

#### Root barriers with split roots

Global segment order may split and interleave a root. Root completion therefore
cannot be published from every segment.

For each root, scan its segments backwards using symbolic worker intervals:

1. At each occurrence, subtract workers that participate in a later occurrence
   of the same root.
2. Immediately after that segment, only the remaining strict-subset workers
   publish root completion.
3. Each worker therefore publishes exactly once, immediately after its own
   final root task, even when the root is fragmented or worker supports differ.
4. Count transient tasks and continuation-owned tasks separately when either
   category owns tasks from that root.

The expected root-barrier arrival count is:

```text
participating resident workers
+ transient logical tasks
+ continuation-owned logical tasks
```

Each category publishes exactly once. A nonparticipant does not publish.

This avoids delaying a worker's publication to another worker's later segment,
while retaining a compact interval representation.

#### Incoming root barriers

Initially emit an incoming root-barrier wait at every segment of the consumer
root. Repeated waits are safe and keep the implementation simple. A later pass
may prove and retain only the first local wait on each worker.

For FlashMLA B9, source-to-radix edges are exact counters and final roots have
one ordinary segment, so this conservative choice does not add repeated waits
to its critical path.

#### Explicit `hl.barrier()` phases

Current main does not support mixing explicit source barriers with implicit
tile-dependency scheduling. Keep that restriction. This redesign does not add
a new phase-barrier IR.

An explicit multi-phase kernel stays on current main's `_emit_phase_loops`
path, including its real backend grid barrier. The optimized global segment
path declines such a kernel. Tuple ordering alone is never treated as
cross-CTA synchronization.

### 9. Lower segments directly

Replace `static_root_body(root)` plus the numerical root loop with segment-order
lowering:

```text
worker = resident worker id
epoch = current replay epoch

for segment in worker_schedule.segments:     # codegen-time unrolling
    if worker is in segment.worker range:
        perform incoming root-barrier wait
        for task in affine strided slice assigned to worker:
            map through segment.task_order
            perform exact task/nested-site waits
            execute the root task body
            publish exact readiness arrivals

    if this is worker's final occurrence of the root:
        publish one local root-completion arrival

store resident worker replay completion
```

There is no barrier between segments. Each CTA advances independently and
skips segments outside its worker range.

#### Root-body reuse

Do not clone a large root body for every segment. Build one reusable outlined
task body per root when the root appears in multiple segments, using the
existing cross-loop outlining machinery when the backend permits it. Roots
whose operations must remain kernel-scoped, including Blackwell dot roots,
reuse the one kernel-scoped body construction rather than forcing an illegal
device helper. Each segment supplies logical task coordinates and the event
epoch.

Inlining or outlining remains a code-generation decision. Schedule correctness
must not depend on it.

#### Non-dense slices

Port the proven non-dense multi-segment lowering from `helion-compiler-mla`.
Each segment iterates its own dense slice and composes through `task_order`.
Codegen must not assume that the second root segment begins at the next
root-local PID after the first.

### 10. Add the proven transient-source launch mode

FlashMLA B4 needs one-task attention CTAs to retire while a fixed resident
cohort executes the downstream schedule. Add only one field to the existing
plan:

```python
@dataclasses.dataclass(frozen=True)
class StaticPipelinePlan:
    worker_schedule: WorkerSchedule
    readiness_counters: tuple[ReadinessCounterPlan, ...]
    root_barrier_edges: frozenset[tuple[int, int]]
    transient_source_root: int | None
```

No separate launch-plan IR is needed while only one inferred source is
supported.

#### Eligibility

Infer a transient source only when all of these hold:

1. exactly one producer-only source among the emitted-prerequisite endpoints;
2. that source has no incoming emitted counter or barrier prerequisite;
3. `P > W`, so the wait-free source is genuinely oversubscribed;
4. at least one emitted exact-counter wait has a source contribution proved to
   be a nonempty strict subset of source tasks, after unioning all source arms
   for that wait, and no source root barrier also gates it;
5. every source task can publish before retirement;
6. the source does not own an incompatible final-arrival continuation;
7. no explicit `hl.barrier()` phase is present;
8. `W` equals the physical visible SM count, with
   `persistent_reserved_sms == 0`;
9. the backend is initially restricted to CUDA Triton, where device-wide
   occupancy is an integral `C = bW` CTAs and the existing compiled-kernel
   residency check applies; and
10. the ticket-role, exact-ownership, progress, and residency proofs below
   succeed.

`P > W` and the strict-partial source signal are conservative topology-only
selection filters, not legality conditions or performance estimates. The
signal does not claim the complete mixed-producer consumer is ready. If the
complete source fits in the resident cohort, the existing static mechanisms
already have all source work admitted and the ticket role supplies no new
admission capability. There is deliberately no condition on `P % W`.

#### Launch and replay

For `P` source tasks and `W` resident workers, launch `P + W` physical CTAs.
A monotonic ticket assigns roles:

```text
raw = atomicAdd(launch_cursor, 1)
role = raw % (P + W)
epoch = raw // (P + W) + 1
```

Roles `[0, P)` execute one source task, publish, and retire. Roles
`[P, P + W)` enter the resident segment program.

No resident role can receive a ticket until every transient source ticket has
been issued. Therefore every incomplete source task is already running or
runnable rather than queued behind a resident CTA that may wait on it. The
complete `W`-CTA resident cohort must fit concurrently after it begins waiting
on collective state.

Current main's launcher proves `W <= C`, where `C` is the compiled
specialization's device-wide CTA capacity. Source-first ticketing plus this
check proves progress: resident CTAs cannot strand unissued source work, and
the complete resident cohort can eventually coexist. The schedule does not
assume `C == W`.

Resident-only plans retain current main's per-worker epoch load/store protocol.
Transient plans derive epochs from the monotonic `P + W` ticket sequence. One
synchronization-state allocation may not be shared by concurrently overlapping
invocations.

#### Two ticket roles, one executable resident program

The source is not a `WorkerSchedule` segment. Its execution is defined solely
by `transient_source_root`, the configured PID task order, and source tickets
`[0, P)`. The resident list scheduler sees only non-source tasks and assigns
them local steps beginning at zero.

This distinction matches the lowering. A physical CTA atomically obtains one
ticket. Source tickets execute one source task and retire; resident tickets
enter the resident program. Segment offsets do not delay physical admission,
so inserting a modeled source wave or an "early" seed into resident strands is
both misleading and performance-sensitive. Exact counters, not an admission
width, determine when an already-admitted resident task may proceed.

The proof boundary therefore establishes:

- the configured source PID order is a total bijection with `[0, P)`;
- the source owns no resident segment and has no incoming prerequisite;
- resident segments exactly and exclusively cover every non-source task;
- source-to-resident prerequisites are emitted launch-stage edges;
- every resident-to-resident prerequisite strictly increases resident rank;
- mixed joins exempt only the source arm; and
- source and root-barrier publications occur before the source CTA retires.

### 11. Validate progress and ownership

Before codegen, validate:

- every resident-owned logical task appears exactly once;
- every transient task has exactly one source ticket and appears nowhere in
  resident segments;
- every continuation-owned task has exactly one dynamic owner;
- no two segments occupy the same `(worker, worker_step)`;
- worker steps are monotone in tuple order on every strand;
- semantic dependency edges plus strand-order edges are acyclic;
- no wait on a strand precedes every possible executor of its producer;
- exact and barrier arrival counts match all ownership categories;
- the complete waiting resident cohort fits concurrently;
- transient mode has one worker per physical visible SM, no reserved SMs, and
  a supported CUDA Triton backend;
- transient ticket ordering satisfies the producer-progress invariant; and
- replay state is not shared by overlapping invocations.

On failure, rebuild and revalidate the complete current-main plan. Never weaken
a dependency to make an optimized schedule pass.

## Why this is enough for the target workloads

All historical timings below are directional evidence from experimental
worktrees. Every implementation phase must establish fresh same-process,
same-source, current-main-based controls with identical numerics and cache
handling.

### FlashMLA B4

The required behavior is expressible with existing types plus ordered
segments:

- inferred transient attention source;
- exact attention-to-radix counters;
- a resident-only global radix/final schedule beginning at local step zero;
- hardware admission as attention source tickets retire; and
- root barriers/final work after each radix root completes.

Nothing requires a new task IR, admission width, or runtime queue. The
successful generated program is exactly the no-modeled-seed program: resident
CTAs enter through the ticket branch and their counter waits expose dynamic
readiness.

Measured gate: **61.248--61.312 us** cold-L2 versus
**67.360--67.424 us** matched standalone, with unchanged BF16/FP32 numerical
behavior.

### FlashMLA B9

The resident ready queue sees all first-level radix roots together. Direct
segment-order lowering preserves that decision, while runtime counters let
each resident strand begin useful work when its actual producer group retires.

Gate:

- request 7 starts close to its approximately 63.2-us readiness point rather
  than 77.952--91.424 us;
- post-attention tail and counter waiting shrink;
- B4 does not regress; and
- persistent beats the fresh matched standalone control.

Measured gate: **89.984--90.016 us** cold-L2 versus
**100.128--100.144 us** matched standalone radix-tree and
**93.984--94.016 us** matched standalone serial reduction.

### FlashMLA Q1 and Q2

B1/S65536/Q1 has `P = 128 <= W = 148`, so it declines transient scheduling
before any performance judgment. B64/S512/Q2 already benefits from
final-arrival behavior and should retain it. These guard against applying the
B4 ticket role merely because a source exists.

### Qwen3 decode

Qwen3 is chain-dominated, and exact tile-level handoffs can expose useful
upstream/downstream overlap. The representative full decode layer has roughly
5,000 CTAs, however, so it exceeds the current 4,096-CTA bounded proposer and
falls back unchanged. A future scalable proposer could use the same segment
contract and symbolic acceptance proof; the present implementation must not
claim this benefit.

Gate: no regression against a fresh current-main tuned static result. The
historical approximately 96-us persistent and 108.5-us standalone
measurements are directional only. Fresh matched lowering produced the same
plan and both versions measured 155.680 us cold-L2 with this probe's current
high-spill configuration.

### Qwen3 FFN

The isolated graph is a dense three-stage chain:

```text
1,536 W13 -> 96 activation -> 512 W2 tasks
```

Unit-depth policies previously had the same abstract makespan. It is a compile
time, code-size, and performance negative control. The new queue should not
invent useful choices where the readiness graph has none.

### Gemma 4 26B-A4B MoE

Gemma provides statically known fork/join branches. Ready tasks from an
underfilled branch can be followed immediately by ready work from another
branch on the same strands. The chosen order no longer collapses back to root
order in codegen.

Gate: no regression against fresh current-main static and standalone controls.
Historical approximately 49.0-us persistent and 55.3-us standalone results are
directional targets.

### DeepSeek-V3 MoE

The router releases independent shared and routed expert branches that later
join. Prior explicit task-list work showed value from filling otherwise idle
slots and interleaving branch tails, but the pre-redesign static codegen could
not retain that CTA order.

For specialization-known task domains, ordered segments can encode the same
kind of task interleaving statically:

1. router tasks release routed work;
2. independent shared work backfills free strands;
3. shared and routed tails follow global readiness rather than root number;
4. join work starts after its real prerequisites; and
5. no device claim/cancellation loop is needed.

The historical 149--151-us CLC result versus approximately 157.6 us standalone
is a hypothesis-generating target, not proof that static strands will match it.
Fresh current-main and redesigned results are equivalent after normalization
by their paired standalone controls (1.188x and 1.189x respectively). The
continuation-bearing graph conservatively retains its established placement.
Runtime-varying routing remains outside this initial static design.

### Nemotron-3 Nano MoE

Structural longest-path ordering previously reproduced a manually interleaved
routed/shared source and improved source order by roughly 10.6%. The new
lowering should preserve the scheduler's chosen interleaving at task-segment
granularity rather than merely reordering whole roots.

Gate: improve on a fresh current-main source-order control. This passes:
98.304 us cold-L2 versus 118.816 us on clean main and 120.864 us with only the
global proposal disabled in the redesigned compiler. The historical
102--104-us result was therefore directionally accurate.

### Muse/Glimmer

Muse is a negative control: legal overlap between bandwidth-heavy branches can
hurt due to contention, which a topology-only scheduler cannot predict.

The scheduler may decline reordering only through a predeclared structural
predicate, such as the absence of provable idle capacity or an underfilled
tail. Runtime benchmark measurements may accept or reject the compiler change,
but they must never silently become workload-specific scheduling policy.

### Full DeepSeek MLA layer

The full layer's main limitation was the shared register/shared-memory/warp
envelope across heterogeneous operations, not missing root chronology. This
redesign does not claim to solve that. The positive MLA target is the
attention-plus-reduction FlashMLA boundary; the full layer remains a resource
negative control.

## Fresh cross-workload validation

All measurements below were taken on the same NVIDIA B200 (visible device 6)
with a 256 MiB L2 flush before each of 50 randomized single-replay samples.
The comparison preserves source, fusion, numerics, tile configuration, and
worker count. The older probe drivers required harness-only compatibility
adapters for their removed `cross_loop_num_workers` key and renamed
`execution_scopes` field; those adapters do not change generated work or
scheduling policy.

| workload | redesigned scheduler | control | result |
| --- | ---: | ---: | --- |
| FlashMLA B4 | 61.312 us | 67.424 us standalone | positive transient-source case |
| FlashMLA B9 | 90.016 us | 100.128 us standalone radix tree | positive head-of-line case |
| Nemotron routed-first MoE, iterative top-k, m4 | **98.304 us** | 118.816 us clean main; 120.864 us same compiler with only the global proposal disabled | **17.3% faster than main and 18.7% faster than the targeted ablation** |
| Gemma A4B, hierarchical top-k, unfused GeGLU, m2 | 57.376 us | 57.328 us clean main | parity within timer quantization |
| Qwen3 full decode, m8 | 155.680 us | 155.680 us clean main | exact latency parity; conservative continuation/size fallback |
| DeepSeek-V3 MoE, m4 | 180.256 us | 182.304 us clean main | parity after normalizing by the paired standalone controls (1.189x versus 1.188x) |

The Nemotron plan is the non-MLA positive case. Its executable root order is
`norm, router, shared-up, top-k, shared-activation, routed-up[0],
routed-up[1] + shared-down, routed-activation, routed-down, final-add`.
The scheduler places the 168 shared-down tasks into the underfilled second
wave of the 928-task routed-up family. This is precisely the generic fork/tail
opportunity the list scheduler was intended to expose; there is no Nemotron or
MoE matcher.

Gemma's accepted non-root-major proposal does not measurably change makespan
at the tested geometry. Qwen3 exceeds the bounded proposal size and retains
its continuation plan. DeepSeek-V3 also retains continuations; a speculative
legacy ready-family placement initially violated the now-authoritative tuple
chronology. Such a proposal is now rejected locally while the valid
conservative plan is retained, rather than rejecting the entire configuration.

## Implementation sequence

### Phase 0: relation prerequisites — implemented

The required relation-analysis deltas and focused tests are present. Existing
FlashMLA benchmark/Gantt tooling remains in the experiment worktree; a unified
compiler plan dump is still future observability work.

### Phase 1: authoritative segment order — implemented

- Document and validate the new `WorkerSchedule.segments` contract.
- Translate current-main root order into the same segment tuple it already
  produces.
- Change codegen to iterate segments directly.
- Reuse one root task body across multiple occurrences.
- Keep current dense assignments and current ownership selection.

Codegen now consumes segment tuple order and one cached traversal certificate;
it no longer owns chronology or independently reconstructs ordinal prefixes.

### Phase 2: split/interleaved roots — implemented

- Port symbolic task-order slicing and non-dense segment lowering.
- Merge adjacent compatible list-scheduled runs.
- Publish barriers after each worker's final root occurrence.
- Validate arrivals from resident, transient, and continuation owners.
- Add complete-plan fallback.

Permuted multi-segment CUDA codegen and strict-subset publication partitions
have focused tests.

### Phase 3: bounded global ready queue — implemented

- Materialize only the bounded emitted-prerequisite CTA DAG used to propose a
  schedule.
- Implement unit-depth structural slack and stable priority.
- Convert selected runs directly to symbolic task-order slices.
- Reject the proposal unless exact-once, rank, and progress proofs succeed
  without enumeration.

There is no `_ReadySlice` layer or segment-DAG linearizer. Plans with
continuations and graphs above 4,096 CTAs currently retain their baseline
placement.

### Phase 4: transient-source lowering — implemented

- Port only the proven source inference and monotonic ticket mechanism.
- Gate on one strand per physical visible SM, an oversubscribed source, and a
  symbolically proved strict-partial source signal.
- Keep source tasks entirely outside the resident segment program.
- Validate replay and resident-cohort progress.

The current generic B4 schedule is correct, byte-identical to the best
zero-modeled-seed control, and matches the hand-shaped research oracle within
timer noise.

### Phase 5: workload rollout — implemented for required gates

Run gates in this order:

1. FlashMLA B4 regression.
2. FlashMLA B9 head-of-line positive case.
3. FlashMLA Q1/Q2 negative cases.
4. Qwen3 FFN and full decode.
5. Gemma A4B MoE.
6. Nemotron MoE.
7. DeepSeek-V3 MoE.
8. Muse and full-MLA negative controls remain optional extended coverage.

Do not add a heuristic for one benchmark without a structural explanation and
at least one positive and one negative cross-workload check.

### Phase 6: remove obsolete constraints — partially implemented

Root-major emission has been removed from the global path and the global queue
is implemented. Retain `_family_placements_at_worker_step` and other legacy
helpers only for anchored or conservative fallback paths; remove them only
after those callers migrate and cross-workload gates pass.

## File-level change map

### `helion/_compiler/tile_dependency.py`

Only the six independently tested relation-normalization changes. No schedule
policy enters this file.

### `helion/_compiler/cross_loop_scheduler.py`

- symbolic task-order slicing and exact traversal certificates;
- authoritative segment tuple order and symbolic schedule validators;
- one emitted-prerequisite view shared by proposal and proof;
- bounded unit-task list placement;
- one inferred transient source with disjoint source-ticket/resident roles;
- symbolic oversubscription and strict-partial-source selection; and
- one root-publication interval plan shared by counting and codegen.

The public compiler contract remains `StaticPipelinePlan` plus
`WorkerSchedule`; there is no parallel schedule representation.

### `helion/_compiler/cross_loop_codegen.py`

- lower segments in tuple order;
- lower non-dense task slices;
- reuse root bodies;
- publish root completion after each worker's final root occurrence;
- add transient ticket dispatch; and
- preserve current exact-counter, continuation, and replay machinery.

### Tests

Extend the existing tile-dependency, scheduler, and codegen tests. Do not add a
parallel test-only schedule representation to production code.

## Required tests

### Relation tests

- bounded shifted modulo positive and boundary cases;
- singleton-target inverse guards;
- project-before-inverse for nested sites;
- partial-source readiness;
- strict-subset one-key event lowering;
- fixed-scalar alias precision; and
- unsupported cases retain conservative dependencies.

### Schedule tests

- tuple order, not root number, defines each worker strand;
- later lexical ready work passes an earlier blocked root;
- segment splitting makes shared-worker order uniform;
- a worker never owns two tasks at one abstract step;
- every resident task appears exactly once;
- arbitrary representable nonidentity traversals are proved bijective, while
  existing L2 PID traversals remain correctly rendered from the same segment
  certificate even when their inverse is outside the proof algebra;
- a fine-grained pipeline preserves overlap;
- a chain without partial release remains stable;
- unresolved or unprojectable nested-wait roots remain anchored, while exact
  projectable nested waits remain movable at owning-CTA granularity; and
- code-size overflow rebuilds the complete current-main plan.

### Synchronization tests

- every concrete dependency ordering has at least one complete proof;
- existing exact counters are retained even when some pairs share a strand;
- a fragmented producer root publishes once per participating resident worker;
- transient and continuation-owned tasks contribute exactly once to root
  completion;
- incoming root-barrier waits guard every consumer segment initially;
- mixed counter/barrier/continuation coverage remains valid; and
- combined semantic and worker-order edges are acyclic;
- acceptance succeeds with all task/worker materialization helpers disabled;

### Transient tests

- `P <= W` and full-fan-in sources decline selection;
- multiple indegree-zero sources decline initially;
- a source owning an incompatible continuation declines;
- `W != physical_visible_sm_count` declines transient mode;
- nonzero `persistent_reserved_sms` declines transient mode;
- unsupported/non-integral-occupancy backends decline transient mode;
- correctness of the two ticket roles is independent of remainder for
  `P < W`, `P = W`, `P = W + 1`, `P = kW`, and `P >> W`;
- `P > W` with only a root barrier declines;
- an exact counter requiring the complete source declines;
- individually partial source arms are unioned before strict-subset proof;
- the source has no resident segments;
- mixed source/resident joins exempt only the source arm from resident rank;
- all source tickets precede every resident ticket;
- all source tasks publish before retirement;
- the full resident cohort satisfies residency; and
- repeated graph replay produces distinct correct epochs.

### Codegen tests

- segment AST order follows `WorkerSchedule.segments` exactly;
- a genuinely permuted, split-root traversal executes the certificate's
  logical mapping correctly;
- nonparticipating workers skip without synchronizing;
- split roots reuse one task body rather than cloning it;
- per-worker final-occurrence barrier publication follows all root segments;
- a transient source contributes exactly `P` root-barrier arrivals, one from
  each source ticket;
- `P <= W` emits no transient dispatch-ticket path;
- explicit `hl.barrier()` kernels retain current-main phase lowering and never
  enter the optimized segment path; and
- resident-only replay remains unchanged.

### Runtime and performance tests

- exact outputs versus same-source standalone Helion;
- unchanged BF16 inputs/outputs and FP32 attention/reduction state;
- pre-captured CUDA Graph medians with L2 flushed before every sample;
- forward/reverse measurement ordering and thermal warmup;
- compiled registers, spills, shared memory, warps, stages, and occupancy
  recorded only as diagnostics; and
- standalone above persistent in both all-SM and root-level Gantt charts.

## Observability

Add a deterministic plan dump containing:

- readiness events and exact producer frontiers;
- each segment's root, task slice, worker range, and abstract steps;
- the authoritative global segment order;
- the induced strand sequence for selected workers;
- counter waits and publications;
- root-barrier expected counts by ownership category;
- continuation and transient ownership;
- source ticket range and resident-local step range;
- oversubscription and strict-partial-source selection evidence;
- structural bottom level/slack; and
- fallback reason.

Runtime instrumentation should label root, segment, logical task, worker,
readiness key, wait interval, and work interval. The resulting charts must make
it possible to distinguish:

1. work that was not legally ready;
2. ready work delayed by schedule order;
3. time spent polling; and
4. a task body that became slower in the shared kernel envelope.

Only the first three can be affected by this scheduler design.

## Explicitly deferred work

- A second persistent schedule IR.
- Literal per-worker instruction arrays.
- Dynamic CLC/work stealing.
- Runtime instruction tensors.
- Multiple transient source groups.
- Reordering roots with unsupported nested waits.
- Optimizing across explicit `hl.barrier()` phases.
- Same-strand elimination of otherwise valid exact counters.
- Latency, resource, contention, or profile-driven cost models.
- ILP scheduling.
- Joint tile-size or reduction-tree search.
- Per-operation register or warp specialization.
- Fusion-boundary changes.

These are not required to fix root-major head-of-line blocking in the target
static workloads.

## Acceptance criteria

The redesign is complete when:

1. `WorkerSchedule.segments` is the documented executable chronology.
2. Codegen iterates segments rather than roots.
3. Every worker's segment subsequence is valid and monotone.
4. Split roots have correct completion accounting.
5. Every dependency remains covered by at least one proved mechanism.
6. Failure rebuilds the entire current-main plan.
7. Transient execution uses disjoint source-ticket and resident roles, with no
   source segments or admission-width policy.
8. Transient selection requires only one oversubscribed wait-free source and a
   strict-partial source signal derived from emitted prerequisites.
9. Generated code remains affine-compressed.
10. There is one topology-only scheduler and no public schedule/admission knob.
11. No latency/resource cost model is introduced.
12. FlashMLA B4 is preserved and B9's ready-work delay is materially reduced.
13. Qwen3 does not regress and at least one static MoE branch case improves.

The central invariant is:

> `WorkerSchedule.segments` is the resident program. The sole external role is
> the explicitly identified transient source, whose canonical task order is
> executed exactly once by source tickets. Root order is syntax and readiness
> is legality; neither may silently replace these executable orders.
