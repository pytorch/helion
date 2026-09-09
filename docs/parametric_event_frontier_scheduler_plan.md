# Parameterizing the existing cross-loop scheduler

## Status

This is the end-state plan and implementation ledger for Helion's cross-loop
scheduler. It incorporates the experiments from FlashMLA, Qwen3 decode, Gemma
4 A4B MoE, DeepSeek-V3 MoE, Nemotron MoE, and Muse/Glimmer FFN.

Implementation checkpoint (2026-09-08):

- Concrete schedule ownership is normalized into the existing
  `WorkerScheduleSegment.task_order` relation.
- Concrete event-frontier proposal and symbolic ownership/progress proofs are
  implemented without a production CTA DAG.
- Independent schedules bypass proposal, and a same-coverage resident
  proposal is retained only when its unit-task final wave does not regress.
- The generic source-ticket frontier participates in resident priority, while
  the proven transient-source execution mechanism is retained for now.
- The first parameterized-extents vertical slice is implemented for canonical
  rank-one roots: symbolic domains lower through the same `WorkerSchedule`,
  conservative root barriers, and runtime-bounded loops, with one cubin reused
  across changing extents.
- Parametric exact-event recurrence, cross-workload rollout, and final
  source/local-path consolidation remain to be implemented and measured.

The hard architectural constraint is:

> Add no new compiler abstractions.

The implementation may generalize fields and semantics of existing objects,
but it must not add a parallel schedule IR, a task-stream IR, an event-family
IR, a runtime instruction IR, or a model-specific scheduling path.

The existing `cross_loop_scheduler_redesign_plan.md` remains the record of the
static experiments and historical timings. This document replaces its
long-term architectural proposal.

## Executive decision

Keep and reconfigure the abstractions Helion already has:

```text
TileDependencyGraph             semantic task dependencies
        |
ReadinessGraph                  exact key/counter semantics
        |
StaticPipelinePlan              complete selected lowering plan
        |
WorkerSchedule                  authoritative task ownership and chronology
        |
WorkerScheduleSegment           one root's schedule relation
        |
cross_loop_codegen              direct execution of that relation
```

`CoordinateDomain` and `CoordinateRelation` become parameter-aware. For one
invocation, let `p` be the fixed tuple of runtime shape parameters. A
`WorkerScheduleSegment` is generalized from a concrete contiguous dispatch
slice into a root-labelled family of relations:

```text
R_p : (launch stage, worker, wave) -> D_root(p)
```

The parameters select the relation member and its bounds; they are not
enumerated schedule coordinates.

The union of segment relations in `WorkerSchedule` is the schedule. A
repeated pattern is represented by affine, floor-division, and modulo pieces
in the existing `CoordinateRelation`; it is not represented by a new Run,
Repeat, Region, or ScheduleProgram object.

For example, a repeated two-root pipeline can be represented conceptually as:

```text
partial relation:   even schedule waves -> partial key wave//2
reduction relation: odd schedule waves  -> reduction key wave//2
```

The real mappings include worker lanes, fan-in, and guards, but the principle
is the same. Repetition is a property of the relation.

## Why this is the minimum necessary change

The current `WorkerScheduleSegment` already contains:

- a root identity;
- a symbolic task-order `CoordinateRelation`;
- worker support;
- dispatch position; and
- the information needed to derive worker steps.

Its current restriction is that each segment occupies one concrete contiguous
dispatch interval and tuple order supplies chronology. That cannot compactly
express a runtime-sized multi-root pattern such as:

```text
for key in runtime_key_count:
    finish the key's producers
    admit its reduction
```

Adding another schedule hierarchy would duplicate `WorkerSchedule`. Instead,
make schedule coordinates explicit in the existing segment relation and make
their wave coordinate authoritative. A single relation piece can then cover
an arbitrary runtime number of translation-equivalent repetitions.

This is more invasive inside `WorkerScheduleSegment` than wrapping it in a new
Region type, but it leaves Helion with fewer concepts and one source of truth.

## Non-negotiable constraints

- One persistent Triton kernel for the scheduled boundary.
- No change to fusion boundaries.
- No change to numerical algorithms, accumulator types, or reduction order.
- No model names, root IDs, or benchmark shapes in compiler policy.
- No latency, register, bandwidth, or profile-derived cost model.
- No catalogue of schedules and no public scheduling-policy selector.
- No public admission-width knob.
- No host-generated schedule or instruction tensor.
- No host/device synchronization to construct a schedule.
- No production materialization of all CTAs.
- No correctness proof based on sampled concrete shapes.
- No duplicate task, dependency, ownership, or schedule graphs.
- No requirement that `num_sm_multiplier` be a power of two.
- Existing range-level resource settings remain visible in lowered Triton.
- Failure to prove an optimized order retains a conservative order in the
  same `WorkerSchedule`; it does not select another scheduler.

## Scope of dynamic reuse

### Schedule-polymorphic dimensions

A dimension is schedule-polymorphic when it changes task-domain extents or
counter fan-in but does not enter:

- kernel-body specialization;
- tile configuration;
- allocation layout;
- backend legality; or
- numerical algorithm selection.

All values in its proved guard domain reuse one compiled binary and one
symbolic `WorkerSchedule`. Typical candidates are batch size, query count,
token count, and a uniform sequence length.

The runtime value appears in relation bounds, `ceildiv` expressions, event
arrival targets, and generated loop limits. It does not enter the schedule
cache key as a concrete value.

### Body-specialized dimensions

A dimension used by `hl.specialize`, a constexpr backend choice, or an
autotuned tile configuration may require another cubin. It should still reuse
the same dependency and scheduling construction where possible, but the plan
does not falsely promise one binary across incompatible backend bodies.

### Ragged and data-dependent metadata

Per-request context arrays, page tables, expert histograms, and device-created
expert offsets are not ordinary shape scalars. A fixed masked maximum-domain
schedule can reuse one binary, but globally compacting or reprioritizing that
work requires runtime information.

The initial plan does not introduce a new ready-queue abstraction to solve
that problem. It first takes the existing relation/counter machinery as far as
possible. Runtime-created compact schedules remain deferred unless they can be
expressed through existing Helion loops, atomics, counters, and tensors without
another scheduler IR.

Computing launch grid and scratch size on the host from already-known scalar
tensor shapes is explicitly allowed. That is launch preparation, not a
host-generated schedule, and requires no device-to-host synchronization.

## Existing sources of truth

### `TileDependencyGraph`

This remains the only semantic source for producer-consumer obligations. The
scheduler may reorder tasks only within the partial order it defines.

### `ReadinessGraph`

This remains the only source for:

- readiness-key domains;
- producer-to-key relations;
- consumer-to-required-key relations;
- publication sites;
- exact arrival cardinalities; and
- obligations not discharged by exact events.

The compiler must build one cached emitted-prerequisite view after choosing
counters, continuations, and root-barrier fallbacks. Proposal, proof, codegen,
and diagnostics all consume that same view.

There is no new EventFamily object. An event family is simply an existing
`ReadinessGraph` event considered over parameterized domains.

### `StaticPipelinePlan`

This remains the complete plan consumed by codegen. It continues to own the
selected `WorkerSchedule`, `ReadinessCounterPlan`s, root-barrier edges, and
existing continuation/source-role decisions.

Do not add `ScheduleProgram` beside it.

### `WorkerSchedule`

This remains the sole authority for non-continuation task placement and
chronology. Its segment relations must form an exact disjoint cover of all
resident and launch-prefix logical tasks.

The global `wave` coordinate replaces tuple position as semantic chronology.
Tuple position is only a stable code-generation and diagnostic order.

### `WorkerScheduleSegment`

Reconfigure the existing class so its `CoordinateRelation` describes the
support and mapping of one root in global schedule coordinates.

The target state is conceptually:

```text
WorkerScheduleSegment:
    root
    task_order: CoordinateRelation

task_order source:
    parameterized launch-stage/worker/wave domain, possibly partial

task_order target:
    exact logical task coordinates for root
```

Every segment uses the appropriate common global schedule domain for its
launch stage. The relation may be undefined at points where that root does not
own the slot. Consequently:

- `task_count` becomes the symbolic cardinality of relation support, not the
  size of the complete source domain;
- undefined `(worker, wave)` points are valid and mean that this segment does
  not own the slot;
- worker support, first/last wave, slicing, and root-publication support are
  derived from relation support;
- source supports of resident segments are pairwise disjoint; and
- the union of their target images is proved to cover every resident task
  exactly once.

The migration must audit every current call site that assumes
`task_count == source_domain.size`, total source support, or contiguous
dispatch arithmetic. Those assumptions may be used only after a relation has
been proved to have the old concrete form.

The existing `worker_begin`, `worker_count`, and `dispatch_offset` fields can
remain during migration as a compact concrete form. They must eventually be
derived from, or normalized into, the same schedule relation rather than
forming an independent placement truth.

There is no new TaskStream object. When the scheduling algorithm needs a
cursor for one candidate root/order, it stores that state in local maps keyed
by existing root and relation IDs.

## Parameterizing existing coordinate relations

`CoordinateDomain.axis_counts_items` currently stores positive concrete
integers. Reusing schedules across shapes requires these existing objects to
support integer expressions over Helion's current symbolic shape environment.

### Domain changes

- Axis extents may be concrete integers or proved nonnegative integer
  expressions.
- `size_expr` returns a symbolic product.
- Operations that enumerate must explicitly request and validate a concrete
  size.
- A runtime-empty domain is represented by a guard; it is not rejected because
  an extent cannot be proved positive at compile time.
- Block sizes remain compile-time configuration values.

### Relation changes

- Piece bounds may contain parameter expressions.
- Piece predicates may contain Presburger-compatible guards.
- Target expressions may use affine terms and guarded quasi-affine forms.
- Projection, converse, composition, cardinality, and implication return an
  exact symbolic result or decline.
- No operation samples runtime values and promotes the samples into a proof.

Runtime mixed-radix flattening is a deliberately restricted extension. A
variable divisor such as `wave % fan_in(p)` and a product such as
`index * extent(p)` are not ordinary Presburger-affine expressions. Support
them only through existing-relation canonical constructors with dedicated
bijection, range, converse, and parameter-substitution lemmas. Arbitrary
nonlinear products between schedule coordinates and runtime parameters, or
arbitrary variable-divisor expressions, decline.

### Concrete compatibility

Substituting concrete parameters into the generalized relation must recover
the current concrete `CoordinateRelation`. Existing concrete callers continue
to use that specialization during the migration.

Random small-shape enumeration is a regression oracle only. Universal guarded
normalization and implication checking form the proof.

## Schedule coordinates and lowering

### Resident schedule domain

For resident work, use existing coordinate-domain machinery to describe:

```text
0 <= worker < worker_count
0 <= wave < wave_count(runtime_parameters)
```

For each `(worker, wave)`, at most one resident segment relation is defined.
Its target is the logical task executed in that slot. An undefined slot is
idle.

At the current concrete checkpoint, normalized relations occupy only the
resident value of the launch-stage axis; source execution still uses the
existing transient ticket lowering. The otherwise-redundant axis is retained
deliberately for Phase 5, where that same `WorkerSchedule` relation must own
the source prefix. It must be removed if source ownership is not migrated;
the current implementation does not yet claim unified source/resident
execution.

`worker_count` remains a selected configuration/hardware value. `wave_count`
is derived symbolically from the union of segment support; it is not stored in
a second schedule object.

The current concrete migration still propagates the largest enclosing wave
domain from normalized inputs, so that envelope may conservatively exceed
actual support after a rewrite. Exact support-derived sizing is part of the
parameterized-domain work, not a property claimed by the present checkpoint.

### Repetition without Run/Repeat

Suppose one schedule cycle consumes `c` waves and advances logical key by one.
A segment relation may use:

```text
cycle = wave // c
phase = wave % c
logical_key = cycle
active when phase belongs to this root
```

The runtime loop bound is parameterized by key count. This is exactly the
information a Repeat node would contain, represented in the relation algebra
Helion already uses.

If `c` is parameter-dependent, this mapping is admitted only through the
canonical runtime mixed-radix constructor and its exact lemmas described
above. It is not treated as a generic affine expression.

The compiler emits such a piece only after proving translation equivalence:

- the same winning structural priority;
- the same readiness signature;
- the same root sequence;
- the same worker mapping;
- the same event-contribution pattern; and
- the same affine state delta for every repetition under the guard.

Comparisons are partitioned only at symbolically derived affine or
quasi-affine crossings. Piece growth is bounded. If translation equivalence is
not proved, retain a compact conservative root order; never emit one piece per
runtime task.

### Code generation

The canonical direct lowering is conceptually:

```text
for wave in range(wave_count(runtime_parameters)):
    for segment in worker_schedule.segments:   # compile-time bounded
        if segment.task_order is defined at (resident, worker, wave):
            logical_task = segment.task_order(resident, worker, wave)
            perform exact waits
            execute the existing root body
            publish exact arrivals
```

Segment predicates must be mutually exclusive for a resident slot. There is
no implicit barrier between waves. Each persistent CTA advances along its own
worker strand, and existing readiness waits protect data visibility.

This loop evaluates only `launch_stage=resident`. Source-ticket lowering
queries the same indexed segment relation at `launch_stage=source`; it does not
own another task mapping.

Codegen may strength-reduce or restructure this loop after proving equivalence
to the authoritative relations. Such optimization must not create a second
schedule. Large root bodies are constructed once, and schedule relations must
not cause repeated inlining or alter range-level resource configuration.

For concrete schedules, codegen may retain today's segment-ordered fast form
when a proof shows it is the same relation. This is a lowering optimization,
not a second scheduling path.

A scheduled-task wrapper with one segment and at most one task per
participating worker may remain inline when it only consumes readiness. There
is no duplicated call site or loop-carried scheduled state to isolate.
Wrappers used by repeated segments or by per-task readiness-counter or nested
loop publications remain outlined. Caller-owned root-barrier publication does
not affect wrapper inlining: codegen emits it after the task dispatch. This
changes neither the schedule relation nor the selected counter; it only avoids
an artificial device-call boundary around a one-trip wait and body.

Any concrete flattening is migration and diagnostic machinery only. Production
dynamic lowering must never expand runtime waves into a compile-time segment
list.

## Event-frontier list-scheduling policy

The scheduling algorithm operates on the finite set of existing roots and
readiness events. Its per-root cursor and event progress values are temporary
compiler state, not new IR objects.

### Quotient dependency graph

Derive a finite bipartite graph directly from existing roots and readiness
events:

```text
root -> readiness event -> root
```

This graph contains dependency kinds, not runtime CTA instances. It is built
from the canonical emitted prerequisites and discarded after scheduling and
diagnostic metadata are computed.

If repeated source structure makes this quotient cyclic, prove an affine
progress rank inside each SCC. If no such rank is proved, retain the existing
proved sequential worker order for that SCC. Never turn a valid recurrence
such as `A_i -> B_i -> A_(i+1)` into mutually dependent whole-root barriers.

### Schema criticality

Assign root nodes unit structural weight and readiness-event nodes zero weight.
On the SCC-condensed quotient graph, compute:

```text
top(v)    = longest weighted path from an entry to v
bottom(v) = longest weighted path from v to an exit
horizon   = max_v(top(v) + bottom(v))
slack(v)  = horizon - top(v) - bottom(v)
base(v)   = (slack(v), -top(v))
```

This is a quotient-graph criticality class. It is not concrete-task slack,
elapsed-time slack, or a latency estimate. Canonical identity is not part of
the class; it appears only as the final tie-break.

Because the quotient structure is shape-independent, ordinary changes to B,
Q, sequence length, or token count do not recompute it or require recompiling
the kernel.

### Deriving a readiness-major producer order

Event-completion scheduling depends on grouping producer tasks by the consumer
cohort they unblock. Derive that order only by composing existing relations:

```text
consumer scheduled ordinal
    -> consumer logical task
    -> every required emitted readiness key
    -> every static producer logical task for those keys
    -> enumerated producer ordinal
```

Union all producer arms after applying the same continuation contraction used
by codegen. Flatten the final consumer/key/local-producer coordinates through
the canonical mixed-radix constructors; do not enumerate concrete tasks.

Accept this producer order only when:

- the composition is symbolically representable;
- its forward mapping and converse are total functions over exactly the
  producer root domain;
- it contains no duplicate or omitted producer task;
- multiple informative consumers induce pointwise-equivalent producer orders;
  and
- every nested requirement and partial producer arm is included.

Unresolved nested, overlapping, partial, nonuniform, or conflicting fibers
retain the root's canonical order. A root barrier supplies readiness but no
fine-grained ordering candidate.

This is a local composition of cached `CoordinateRelation`s, not another graph
or task-order abstraction.

### Root frontiers

For each existing root task order, the scheduler tracks symbolic expressions
for:

```text
cursor
task-domain end
admissible end
next relation-piece boundary
all outgoing event-key boundaries
```

There may be multiple outgoing readiness events. Candidate chunks end at the
earliest boundary across every affected event relation, not at a single
invented current key.

The scheduler distinguishes:

- **assigned**: the task has a worker/wave coordinate;
- **admissible**: every predecessor has an earlier ownership rank, so entering
  its wait cannot block admission of an unassigned producer; and
- **runnable**: completed readiness counters permit executing the body.

A predetermined resident relation may assign an admissible consumer before
its producers have physically completed; the emitted wait gates its body.

### Candidate interval

For each admissible root frontier, propose the largest interval ending at:

```text
min(
    admissible_end,
    every outgoing event-key boundary,
    next relation-piece boundary,
    cursor + remaining worker slots,
    task-domain end,
)
```

This is the largest interval over which readiness, priority, event
contributions, and task mapping are unchanged and affine. It is not a tunable
chunk-size heuristic.

### Multi-event release lookahead

For each candidate interval, derive which consumer cohorts become admissible
after applying all of its claimed-event contributions. A consumer cohort is
newly admissible only if every claimed prerequisite frontier required by that
cohort is then satisfied.

Define:

```text
release_class = min(base(consumer) for newly admissible consumer cohorts)
effective_class = min(base(candidate root), release_class)
```

If no cohort becomes admissible, use the candidate root's base class. This lets
the final producer interval inherit the class of the downstream work it
unlocks without confusing claims with physical completion.

### Priority

Choose candidate intervals lexicographically by:

```text
1. effective structural slack
2. immediate inlet from an exact earlier launch stage
3. effective downstream depth
4. already-admissible work at that class before merely prospective release
5. closes a readiness event at that class
6. earliest fixed launch-stage producer frontier
7. canonical root, key, and task order
```

The fourth field is computed mechanically:

```text
prospective = 0 if effective_class == base(candidate root) else 1
```

Thus a candidate whose own currently admissible work already has the winning
class precedes a producer interval that can only create work at that class.
When no such downstream candidate is yet admissible, the event-closing
producer inherits its consumer's better class and wins over ordinary ancestor
work.

The second and sixth fields apply only when an earlier launch stage has an
exact ticket order. At equal slack, immediate consumers of that stage precede
deeper resident work. Otherwise a statically admissible downstream wait can
occupy a fixed worker strand while an independent inlet task that could make
progress is placed behind it. This is a launch-stage property, not a named
root or model rule; when there is no earlier stage, every candidate has the
same inlet class and ordinary downstream pipelining is unchanged.

Within those inlet tasks, the sixth field is the maximum source ticket
required by the candidate cohort, derived through the emitted readiness
relations. It is a structural release order, not a completion-time estimate:
it affects priority only, while runtime counters remain the sole permission to
execute a consumer body. Candidate intervals end whenever this frontier
changes.

Highest bottom-level is not the primary rule. It can keep issuing FlashMLA
partials because every partial contains the reduction in its suffix, delaying
an already-admissible reduction indefinitely.

### Work conservation

Select intervals until every worker slot in the abstract wave is filled or no
admissible work remains. Preferring a ready downstream root does not create a
barrier: after assigning its available tasks, remaining workers receive other
admissible roots.

For schedules with identical resident task coverage, selection rejects a
proposal whose final occupied unit-task wave is later than the input
schedule's. This is a symbolic no-regression certificate, not a latency cost
model. A source-ticket proposal is compared separately because moving a source
root out of resident ownership intentionally changes the compared task set.
With no emitted prerequisite there is no scheduling opportunity, so the input
schedule is returned without stepping through its waves.

### Symbolic recurrence extraction

Advance the temporary root/event frontiers symbolically. When the complete
frontier state changes by the same affine delta under an invariant priority
decision, summarize the recurrence directly in the affected
`WorkerScheduleSegment.task_order` relation using wave `floor`/modulo pieces.

If a finite guarded comparison changes the winner, split relation pieces at
the proved crossing. If neither recurrence nor a bounded piecewise form can be
proved, retain canonical compact ordering with a conservative admissible
frontier. Do not unroll the runtime extent.

There is one scheduler. "Local order" is simply the conservative result when
only one root frontier is movable or a finer frontier cannot be proved.

## Synchronization and execution ownership

### Claimed versus completed readiness

Compile-time scheduling tracks claimed producer coverage to determine
admissibility and event-closing candidates. Runtime counter completion remains
the only permission to execute a consumer body.

The selected counter target may be a runtime expression. Initialization and
epoch handling must be replay-safe.

Counter layout, byte size, target initialization, and replay reset are derived
from the same parameterized readiness-key relations. The host may allocate or
select scratch capacity from already-known scalar shapes, but it may not read
device metadata or construct task order. Device-derived ragged keys use the
conservative maximum-domain storage permitted by their existing source.

### Predetermined resident progress

Every wait in a resident schedule must have all possible producers at a
strictly earlier symbolic ownership rank:

```text
(launch stage, wave, worker-local step)
```

The proof uses all required nested keys and contracts continuations exactly as
codegen does. Runtime waits provide visibility; earlier ownership rank prevents
a logical cycle, but rank alone does not prevent an unlaunched producer worker
from being excluded by waiting resident CTAs.

Every blocking resident schedule therefore also requires a backend capacity
certificate. Initially:

- `worker_count == visible_sms * required_blocks_per_sm`;
- no SMs are reserved from that count; and
- post-compilation occupancy proves at least `required_blocks_per_sm` resident
  CTAs per visible SM.

All producer worker strands covered by the rank proof can then become resident
concurrently. This is a legality requirement, not a performance cost model.

### Final-arrival continuations

Retain the existing `FinalArrivalContinuation` abstraction. A continuation is
selected only when its current exact-once and body-safety proofs succeed.
Outgoing dependencies are contracted through the same cached helper used by
proposal, proof, and codegen.

Do not introduce a generic continuation IR.

### Generalizing the existing transient source

Retain the current source-ticket mechanism during the first scheduling phases.
Every non-continuation task relation, including a source-ticket relation, is
stored exactly once in `WorkerSchedule.segments` and distinguished by its
launch-stage support. `StaticPipelinePlan` may cache the index of that existing
segment for lowering convenience; it must never store or reconstruct another
source mapping. Continuation ownership remains authoritative in the existing
selected continuation/counter plan.

Generalize source selection by replacing the current model-shaped root
assumption with predicates over that existing source segment relation.

The model-independent eligibility conjunction is:

- one wait-free source order with an exact task-to-ticket bijection;
- source task count `P > W`;
- a nonempty strict-partial readiness relation from source work into resident
  work;
- no incoming source wait;
- every source task publishes before completion;
- source tickets are allocated globally before resident tickets;
- resident worker count is no greater than proved concurrent capacity; and
- prefix, resident, and continuation domains form an exact disjoint cover.

CUDA does not guarantee increasing CTA admission from PID order alone.
Therefore correctness requires the existing global source-first ticket
allocator; disjoint PID ranges are insufficient. Replay epochs must not alias.

Excluding the source root from resident ownership and admissibility does not
exclude it from scheduling priority.  Its exact ticket relation induces a
priority-only external frontier for each dependent resident cohort.  This
preserves source-release order without pretending that ticket issue proves
physical completion.

Once B4/B9 parity is established, rename or reshape the existing field if
needed, but do not add a separate launch-prefix plan object.

### Root barriers

The semantic root-complete event counts logical root tasks. For predetermined
relations, the compiler may prove one aggregated publication per participating
worker after that worker's final root task. The aggregate count and publication
sites must be derived from the same worker/task relation used by codegen.

No dynamic-claim root aggregation is introduced in the initial design.

## Symbolic proof contract

No optimized schedule reaches codegen unless all properties below hold for
every runtime parameter satisfying the schedule guard.

### Exact ownership

- Each resident logical task has exactly one `(worker, wave)` preimage.
- No resident slot maps to more than one root task.
- Source-ticket, resident, and continuation task domains are pairwise disjoint.
- Their union covers every logical task exactly once.
- Task-order and ticket-order converses are total and unique on their support.

### Dependency coverage

Every `DependencyObligation` is covered by at least one emitted mechanism:

- exact readiness counter;
- final-arrival continuation; or
- conservative root barrier.

Coarsening scheduler admission does not replace or weaken the authoritative
counter relation. Coverage is computed from what codegen emits.

### Resident rank progress

For every predetermined wait edge, prove:

```text
max(producer ownership rank) < consumer ownership rank
```

The rank is derived once from `WorkerScheduleSegment.task_order`. Nested waits
use all required keys. This replaces the quadratic segment-pair precedence
validator for global schedules.

### Source progress and capacity

Source-to-resident progress uses lexicographic launch-stage/ticket rank plus a
backend capacity certificate. Initially that certificate requires:

- `W == visible SM count`;
- no reserved SMs;
- compiled occupancy of at least one resident CTA per SM; and
- the proved source-first ticket allocator.

These facts establish progress, not predicted performance.

### Proof restrictions

- Proofs use relations, parameter guards, and interval frontiers.
- Production proof never enumerates tasks, workers, keys, or shape values.
- Unsupported operations decline conservatively.
- Concrete CTA DAGs exist only in tests as differential oracles.
- Random substitutions test implementation correctness but are not proofs.

## Avoiding duplicate computation inside the compiler

The following values must each be derived once and cached:

- emitted prerequisite descriptors;
- root task-order relations and exact converses;
- continuation contraction;
- symbolic worker/wave rank;
- root participation and final publication support;
- quotient-graph `top`, `bottom`, and criticality class; and
- schedule relation support per root.

Proposal, proof, codegen, and diagnostics consume these caches. No component
reconstructs a CTA DAG or independently interprets nested readiness.

## Probe requirements

Historical timings below are directional evidence from experimental
worktrees. Every implementation phase must remeasure same-source controls with
identical numerics, fusion boundaries, resource settings, and cache handling.

### FlashMLA attention/reduction boundary

This boundary contains attention partial and reduction work. It does not stand
for the complete DeepSeek MLA layer with projections, normalization, RoPE,
cache operations, and O projection.

For uniform shapes, batch and query count may be schedule-polymorphic. A
sequence-derived `num_splits` may be polymorphic only when every guarded value
uses the same reduction topology and summation order; otherwise it remains a
body-specialized regime. Head count normally remains model/TP-static. A
key-major partial relation should make producer-fiber boundaries symbolic in
eligible `num_splits` regimes.

Required behavior:

- preserve source-first ticket admission for the B4 case;
- finish useful partial fibers instead of distributing equal progress over
  every key;
- assign the newly admissible reduction at the earliest safe wave;
- preserve exact BF16/FP32 numerics and reduction order;
- decline source-ticket mode for cases such as B1/S65536/Q1 when its structural
  eligibility conjunction fails; and
- retain the established Q2 behavior.

Required cases:

- canonical ThunderKittens B4;
- B9 random sequence lengths as a concrete static/ragged validation;
- B1, S65536, Q1, H16; and
- the established Q2 case.

B9 random lengths do not count as schedule-polymorphic reuse until ragged
device metadata is supported. Earlier phases compile or mask that case
conservatively.

Historical evidence is approximately 61.3 us persistent versus 67.4 us
matched standalone for B4, and approximately 90.0 us versus 100.1 us matched
standalone radix-tree for B9.

Compare against same-boundary standalone Helion and production
ThunderKittens. Report ThunderKittens kernel latency separately from any host
schedule construction.

### Full DeepSeek MLA path

Keep this separate from FlashMLA. It includes projection, normalization/RoPE,
cache, attention, reduction, and O-projection work according to the established
Helion/vLLM boundary.

The prior main limitation was the shared register/shared-memory/warp envelope,
especially O projection, rather than missing schedule chronology. Use it as a
resource negative control. A scheduling change may reduce gaps but must not be
credited for a body slowdown or a changed fusion/numerical path.

### Qwen3 decode

Use the checked-in pretuned full-decode kernel and current attention boundary.
Its graph is mostly chain-dominated, so parity is acceptable; a structural
scheduler should not invent concurrency that dependencies do not expose.

Required validation:

- B1 no regression;
- B greater than one with one short and one long context;
- uniform B/Q/maximum-context changes across declared polymorphic guards;
- exact QKV/attention/reduction/O-projection handoffs;
- compilation through the same event-frontier scheduler; and
- no ready downstream attention work delayed behind equal-class ancestors.

Before ragged metadata scheduling, mixed contexts use a canonical max-domain
masked schedule. They do not claim globally optimal per-request ordering.

Historical pretuned parity was approximately 94.0 us on the redesigned branch
versus 94.1 us on clean main.

### Gemma 4 A4B MoE

Use the checked-in pretuned fused hierarchical kernel. The current slot-dense
boundary has work domains determined by token slots/top-k; expert IDs primarily
affect addresses. It is therefore suitable for parameterized relations even
when inputs route to different experts.

Required validation:

- B1 no regression;
- B2 and larger token counts;
- inputs routing tokens to distinct experts;
- exact woven mixed-radix down-projection inverses;
- final-arrival ownership; and
- one binary across dimensions explicitly declared schedule-polymorphic.

This is mainly a generality/parity case. Historical B1 parity was approximately
49.06 us versus 49.12 us on clean main. The B2 diagnostic was effectively tied
at approximately 67.5 us across list, local, and standalone.

### DeepSeek-V3 MoE

The scheduling opportunity is a router fork into independent shared and routed
expert branches followed by a join. Ready shared work can occupy otherwise
underfilled routed waves.

The current routed probe contains a `tokens == 1` specialization. Do not claim
token-polymorphic reuse until that source is made parameterized without
changing its numerical/fusion boundary.

Required validation:

- exact routed/shared branch and join dependencies;
- relation-level branch interleaving;
- comparison with a canonical-order ablation produced by the same scheduler;
- same-source standalone Helion; and
- a later token-dynamic slot-dense source before claiming binary reuse.

Historical evidence was approximately 171.8 us event-scheduled versus 182.2 us
with the global proposal disabled. Standalone remained faster at approximately
154.0 us, so body/resource work remains separate.

### Nemotron MoE

Nemotron provides both a static branch-packing win and a boundary on purely
symbolic scheduling.

For the specialization-known routed-first probe, existing relation pieces
should reproduce the generic opportunity of placing shared-down work into an
underfilled routed-up wave. Historical evidence was approximately 98.3 us
versus 118.8 us on clean main.

Production routing may generate expert histograms, sorted offsets, and work
ranges inside the kernel. Under the no-new-abstraction constraint, the initial
implementation uses existing source loops/tensors with a proved masked maximum
domain or canonical expert order. It must not pretend to have compacted an
unknown runtime worklist.

If that is insufficient, pause and demonstrate why existing loops, atomics,
and counter plans cannot express exact claims before proposing any runtime
scheduler extension.

### Muse/Glimmer FFN

Use the established five stages:

```text
gate split-K main
    -> keyed gate reduction
    -> activation
    -> down split-K main
    -> keyed down reduction
```

The critical structural behavior is to finish one activation slice's gate
fan-in, assign its reduction/activation, and expose its down projection before
spreading equal-priority gate progress across every slice.

Use matched bodies and resource settings:

- split-K 16 for gate and down/activation;
- gate/down N32/K128, reduction N64, activation block 256;
- W1, range stages 2, no `maxnreg`, and a common safe multiplier initially;
- `standalone_matched` with identical body parameters; and
- freshly tuned standalone as the final performance target.

The previous normalization reduced a concrete proposal from thousands of
fragments to 77 segments, but validation still took minutes. The parameterized
schedule relation should express the repeated key pattern with a bounded
number of pieces, and proof time must scale with those pieces rather than CTA
or segment-pair count.

Muse is also a resource-contention control. Earlier legal overlap can lose
when two bandwidth-heavy stages contend. Record that as a body/resource result;
do not add a Muse-specific scheduling exception or silently tune priorities
from timings.

## Measurement and observability

### Performance protocol

For every probe:

- preserve source, fusion, numerical operations, and dtypes;
- use production or explicitly matched tile/range configurations;
- verify outputs before timing;
- use pre-captured CUDA Graph replay where applicable;
- flush at least 256 MiB of L2 before every measured sample;
- thermally warm the device;
- randomize forward/reverse comparison order;
- report median and dispersion over enough samples for stability;
- record registers, spills, shared memory, warps, stages, and occupancy; and
- use uninstrumented execution for pass/fail timing.

Cold-L2 flushed latency is the primary metric.

### Gantt charts

For every required shape, produce aligned charts with standalone on top and
persistent below at both all-SM and root/event level. Mark:

- task execution;
- readiness publication;
- consumer admission;
- readiness wait/poll time;
- ready-but-unscheduled delay; and
- post-final-producer tail.

The chart must distinguish a scheduling delay from a root body that became
slower inside the shared resource envelope.

### Compiler diagnostics

Extend existing plan dumps; do not add a diagnostic IR. Report:

- parameter guards and schedule-polymorphic dimensions;
- root/event quotient edges;
- `top`, `bottom`, slack, and base class;
- each segment relation's schedule support and task mapping;
- event-closing/release-class decisions;
- source-ticket and continuation ownership;
- symbolic ownership/progress certificates;
- relation-piece count and compilation time; and
- conservative-order reasons.

A requested small concrete substitution may be printed after symbolic
acceptance for debugging.

## Required ablations

Temporary internal test controls may disable:

- event-completion tie-breaking;
- source-ticket selection;
- noncanonical root interleaving; and
- direct parametric lowering in favor of a concrete specialization.

These controls all use the same scheduler and relations. They are not separate
user-visible schedule modes and are removed after rollout.

For each workload separate:

1. schedule ordering gain over an identical persistent body;
2. launch/tail gain over matched standalone;
3. body slowdown from the shared resource envelope; and
4. compile-time/generated-code overhead.

## Implementation sequence

The ordering below incorporates the reviewer's recommendation to validate the
scheduling representation before generalizing the entire relation algebra.

### Phase 0: freeze evidence

- Preserve all six probe families as benchmark-only assets.
- Record source hashes, exact shapes, configurations, numerical checks, and
  cold-L2 commands.
- Save lowered Triton and aligned Gantt charts.
- Add small concrete CTA-DAG oracles in tests only.

Exit gate: every historical timing and structural claim has a reproducible
matched control.

### Phase 1: re-express current concrete schedules

- Teach existing `WorkerScheduleSegment` to expose its current placement as a
  `(worker, wave) -> logical task` relation.
- Prove that the union of those relations matches current tuple semantics.
- Test that `task_count` is relation-support cardinality for partial-support
  and runtime-empty relations, rather than complete source-domain size.
- Lower the relation through current codegen for concrete extents.
- Keep concrete fields only as a normalized compatibility form.

Exit gate: all current static schedule/codegen tests are unchanged, with no
new compiler data type.

### Phase 2: event-frontier ordering on concrete extents

- Replace concrete CTA-node scheduling with root/event frontier intervals.
- Compute schema criticality and multi-event release lookahead.
- Emit the result directly as existing segment relation pieces.
- Compare against the small concrete oracle.
- Validate FlashMLA, Muse, and Nemotron before dynamic-domain work.

Exit gate: scheduling quality matches the current successful experiments,
piece counts remain bounded, and no production CTA DAG exists.

### Phase 3: symbolic rank proof

- Derive worker/wave rank from the existing segment relations.
- Prove exact ownership and every wait edge symbolically.
- Use all nested keys and the one continuation contraction cache.
- Bypass and then delete the quadratic segment-pair validator.
- Make materialization raise if called inside acceptance.

Exit gate: proof time scales with relation-piece count and canonical Muse N32
compiles within the agreed budget.

### Phase 4: parameterized extents

- Generalize existing CoordinateDomain/CoordinateRelation bounds.
- Derive symbolic event counts, fan-in, and schedule wave counts.
- Detect translation-equivalent frontier recurrence and express it with
  floor/modulo relation pieces.
- Lower runtime-bounded wave loops.
- Permit host launch/scratch sizing from known scalar shapes without schedule
  generation.
- Verify cubin identity, not merely Python kernel/cache identity, while varying
  polymorphic shapes, scratch sizes, and counter targets.

Exit gate: the dynamic implicit-dependency test that currently rejects the
static pipeline passes and reuses one binary across its declared shape guard.

Implementation checkpoint (2026-09-08): the exit gate is satisfied for the
minimal conservative slice. `CoordinateDomain` and `CoordinateRelation`
preserve symbolic integer bounds while concrete enumeration remains explicit.
Canonical rank-one roots are represented by two exact relation pieces per
root (full waves plus a partial tail), and codegen strength-reduces that proved
relation into runtime-bounded cyclic loops. Dynamic memory layouts are not
specialized from hints: dependencies coarsen to root barriers, and every
resident worker publishes once per producer root so epoch targets stay fixed
while shapes vary between graph replays. Exact parameterized readiness events
and recurrence extraction remain later Phase 4 work; parameterized roots with
rank greater than one, L2-permuted orders, continuations, and transient-source
admission still decline rather than relying on a shape hint.
The binary-reuse regression explicitly opts its runtime extent out of Triton
specialization; parameterizing the schedule does not override backend
specialization policy for ordinary scalar arguments.

### Phase 5: source-ticket generalization

- Port only the proved source-first ticket behavior.
- Replace shape/root assumptions with exact source relation predicates.
- Preserve the existing plan and counter abstractions.
- Prove source/resident/continuation disjoint coverage and capacity.
- Delete redundant transient-specific logic only after B4/B9 parity.

Exit gate: FlashMLA B4/B9 retain their gains without an MLA matcher or an
admission-width knob.

### Phase 6: cross-workload rollout

Roll out in this order:

1. FlashMLA B4, Q1/Q2, and B9;
2. Muse/Glimmer FFN;
3. Qwen3 B1 and mixed-context B greater than one;
4. Gemma A4B B1/B2 and varied expert IDs;
5. DeepSeek-V3 routed/shared probe; and
6. Nemotron routed-first probe.

Each step requires correctness, compilation-time, code-size, cold-L2, and
Gantt gates before proceeding.

### Phase 7: dynamic metadata audit

- Measure how far max-domain masking and canonical relation order go for
  ragged MLA/Qwen and production Nemotron routing.
- Identify concrete missing operations in existing loops, atomics, counters,
  and relations.
- Do not add an abstraction merely because a dynamic queue would be familiar.
- If a runtime worklist is proven necessary, return with measured evidence and
  a separate minimal proposal subject to the no-new-abstraction constraint.

### Phase 8: consolidation

- Remove production concrete CTA materialization.
- Remove the old independent local/global scheduler choice.
- Remove duplicate task-order inversions and prerequisite traversals.
- Remove obsolete transient-source special cases after source-ticket parity.
- Keep concrete oracles only in tests.

Exit gate: one existing schedule representation, one dependency view, one
ownership proof, and one scheduler remain.

## File-level change map

### `helion/_compiler/tile_dependency.py`

- Generalize existing domain extents and relation guards.
- Preserve exact converse/projection/composition semantics.
- Add no scheduling or runtime-execution object.

### `helion/_compiler/cross_loop_scheduler.py`

- Reconfigure existing `WorkerScheduleSegment` relation semantics.
- Derive temporary root/event frontiers from existing graph objects.
- Compute criticality and event-release priority.
- Build one `WorkerSchedule` and prove it.
- Generalize existing source/continuation decisions without new plan types.

### `helion/_compiler/cross_loop_codegen.py`

- Lower global worker/wave relations directly.
- Emit parameterized wave bounds and relation predicates.
- Reuse existing root bodies, counters, barriers, and continuation lowering.
- Preserve concrete fast lowering only as a proved rendering optimization.

### Tests

- Concrete schedule relation equivalence.
- Parameter substitution and exact inverses.
- Event-completion priority and multi-event fan-in.
- No duplicate/omitted tasks.
- Nested all-key rank proofs.
- Source-ticket admission and replay.
- Dynamic binary reuse.
- Small concrete oracle comparison.
- All six performance probe families.

## Risks and honest boundaries

### Relation expressiveness

Some useful schedules may not have a compact affine/quasi-affine worker/wave
relation. The correct response is conservative order, not runtime-sized piece
generation or an unproved formula.

### Generated control flow

A global wave loop with root predicates may increase branch overhead or extend
resource lifetimes. Compare lowered Triton, registers, spills, and body timing.
Proved codegen strength reduction is allowed; an independent schedule is not.

### Structural versus resource optimization

Event-aware order can reduce tail and readiness delay but cannot guarantee that
overlap wins under a shared register, warp, shared-memory, or bandwidth
envelope. Muse and full DeepSeek MLA must expose this distinction.

### Makespan optimality

No topology-only greedy algorithm is optimal for every precedence graph. The
chosen policy has one explicit objective: preserve constrained branches while
advancing completed dependency frontiers. Cross-workload ablations determine
whether it is worth keeping.

### Runtime-created work

A compiler cannot symbolically rank expert groups whose sizes do not exist
until a router runs. Under this plan such cases use existing masked/canonical
execution first. Any later compaction mechanism requires separate evidence and
must reuse existing abstractions.

## Final acceptance criteria

The redesign is complete when:

1. no new compiler abstraction has been introduced;
2. `TileDependencyGraph` and `ReadinessGraph` remain semantic truth;
3. `StaticPipelinePlan` remains the overall lowering plan;
4. `WorkerSchedule` remains the only execution schedule;
5. `WorkerScheduleSegment` relations exactly cover task ownership and global
   worker/wave chronology;
6. repeated patterns are relation pieces, not Run/Repeat/Region objects;
7. declared schedule-polymorphic dimensions reuse one compiled binary;
8. production scheduling and proof never materialize all CTAs;
9. every logical task executes exactly once;
10. every dependency remains covered and every wait has a progress proof;
11. source tickets use a source-first allocator and capacity certificate;
12. there is one event-frontier scheduler and no local/global policy split;
13. priority contains no latency/resource estimate or model-specific rule;
14. FlashMLA B4/B9 retain performance and Q1/Q2 do not regress;
15. Qwen3 and Gemma retain pretuned performance;
16. DeepSeek-V3 and Nemotron retain branch-packing wins;
17. Muse compiles compactly and exposes event-completion behavior;
18. all primary timings are cold-L2 flushed against matched standalone Helion;
    and
19. standalone-above/persistent-below Gantt charts explain every remaining
    performance gap.

The governing invariant is:

> Existing dependency relations define legality. Existing worker-schedule
> relations define exact ownership and order. Existing synchronization plans
> define runtime readiness. Dynamic shapes parameterize those same objects;
> they do not create another scheduler.
