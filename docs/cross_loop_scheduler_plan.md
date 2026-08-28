# Cross-Loop Scheduler Architecture and Implementation Plan

This is the living design document for Helion's cross-loop scheduler. Update the
status checklist and progress log as implementation decisions change or work is
completed.

## Goals

The scheduler should:

- Make the Qwen3 layer megakernel faster than the equivalent separate kernels.
  The current target is approximately 74 microseconds versus approximately 80
  microseconds for the separate-kernel baseline.
- Be correct for broader shapes, batch sizes, loop grids, indexing layouts, and
  producer/consumer topologies without recognizing Qwen-specific source shapes.
- Preserve each root's existing code generation. Cross-loop scheduling may add
  only the dispatch, waits, publications, fences, and bookkeeping required to
  order existing work.
- Prefer a conservative, demonstrably correct schedule over a specialized fast
  path whose producer-to-consumer mapping is not proven.
- Become simpler by deriving scheduling from producer and consumer accesses,
  rather than accumulating topology matchers and exceptional cases.

## Static-first submission cleanup (2026-08-27)

The first upstream PR is based on `helion-cross-kernel` and contains only the
static persistent dispatcher. CLC remains a separate follow-up built over the
same dependency and readiness products. The cleanup must preserve the current
Qwen3, Gemma4 A4B, GPT-OSS, and DeepSeek-V3 static schedules while reducing
API surface and keeping task-count-sized materialization out of production.

Required cleanup, in order:

- [x] Expose the synchronization policy as the kernel-local autotuner field
  `cross_loop_schedule = "barrier" | "static_pipeline"`. Admit the field only
  on NVIDIA Triton kernels whose DeviceIR contains at least two top-level task
  families and a compiler-inferred cross-loop dependency. Do not require
  statically instantiable task geometry merely to expose the choice. The safe
  default is `"barrier"`; `"static_pipeline"` is an ordinary autotuner
  candidate and raises `InvalidConfig` when the selected tile geometry cannot
  produce a concrete legal schedule. Single-root kernels, independent roots,
  explicit-barrier-only kernels, and unsupported backends do not expose the
  field. A future CLC dispatcher may become a third value without changing the
  dependency graph or event representation.
- [ ] Follow-up: establish a runtime residency contract for static kernels containing
  cross-loop waits. The current launch-time occupancy calculation proves only
  theoretical capacity on an otherwise idle device; concurrent work on another
  stream can occupy slots after that calculation and leave a producer CTA
  queued behind resident consumers that are polling. Cooperative launch was
  implemented and validated, but is deliberately reverted from this PR because
  it regresses the canonical cold-L2 workloads. A follow-up must provide a
  concurrency-safe execution contract, then add a bounded concurrent-launch
  regression. This is explicitly deferred from the static-first PR; its safe
  default remains `cross_loop_schedule="barrier"`, which launches
  cooperatively because a grid barrier intrinsically requires simultaneous
  residency.
- [x] Preserve accesses from DeviceIR graphs reachable through more than one
  top-level root. Instantiate access facts per `(root, execution callsite)` or
  reject unsupported graph sharing explicitly; never map ambiguous ownership
  to `-1` and silently omit the hazard. Add a source-level shared-callsite test.
- [x] Localize symbolic-proof fallback. Failure to merge, quotient, or lower
  one consumer dependency must coarsen only that dependency point or event to
  `FamilyDone`; it must not discard exact events elsewhere in the graph. Add a
  mixed regular/irregular graph test proving that unrelated readiness survives.
- [x] Preserve nested-scope readiness when a consumer family cannot be moved
  earlier. Coarsen the exact relation to one scope-entry wait per owning strand
  instead of falling back to whole-family completion. Implemented in
  `18499db5`.
- [x] Honor an explicitly selected `cross_loop_num_workers` exactly. Progress
  validation, rather than event-boundary snapping, determines whether that
  worker count is legal. Implemented in `18499db5` as a migration step before
  consolidating the worker-grid tuning surface.
- [x] Remove `cross_loop_num_workers` and derive the static persistent grid
  solely as `W = device_sm_count * num_sm_multiplier`. Make
  `num_sm_multiplier` the sole worker-grid knob. Retain its existing
  power-of-two autotuner domain for this first PR so ordinary persistent-kernel
  search spaces do not change. Remove the exact-worker field atomically from
  Config, ConfigSpec, DeviceIR/scheduler APIs, tests, and canonical probes;
  there must be no second exact-worker escape hatch. Readiness keys, local
  triggers, placement, and liveness remain deterministic products of the event
  graph and the selected `W`. Relaxing the multiplier to a bounded integer
  domain is a follow-up PR.
- [x] Remove `preordered_edges`. Every production and test caller supplies an
  empty set, so root-completion selection should depend only on emitted event
  coverage and the root-completion edges selected by the scheduler.
- [x] Make nested-scope geometry explicit. After proving that the renderer has
  exactly one nested axis, use that axis's declared count rather than inferring
  actions per strand from `scope_domain.size // root_domain.size`.
- [x] Move the exact task-level schedule oracle out of production code.
  `validate_worker_schedule()`, `_local_trigger_predecessors()`,
  `_static_ancestors()`, `WorkerSchedule.task_order()`, and the
  `EventGraph.materialized_*` helpers are test-only. Relocate them to test
  support so production scheduling contains no `LogicalRelation.materialize()`
  calls and compile-time work scales with relation pieces rather than task
  count.
- [x] Extract cross-loop emission from `program_id.py` into
  `cross_loop_codegen.py`. Keep one shared event/body lowering and one static
  dispatch backend; this is a mechanical module boundary, not a second
  scheduler or event representation. Require structurally identical generated
  Triton before and after the move.
- [x] Ablate the remaining generic scheduling policies in the canonical probes:
  automatic final-arrival execution, key-major producer reordering, greedy
  complete-family placement, and one-axis milestone segmentation. Keep them as
  compiler policy only when they are broadly useful and performance-neutral or
  beneficial; do not replace them with model-specific toggles. The 128-byte
  event-counter separation remains an ordinary hardware-layout policy.
- [ ] Reduce submission scope. Keep canonical runnable Qwen3, Gemma4 A4B,
  GPT-OSS, and DeepSeek-V3 benchmarks, but exclude generated PTX/results and
  exploratory probes from the compiler PR. Condense this living log into a
  short architecture document or move the historical record out of the PR.
- [ ] Rebase onto current `origin/main` before final validation and resolve the
  overlapping compiler changes there before judging the final diff size.
- [ ] Run the final acceptance battery: complete dependency tests, Ruff and
  formatting, affine-chain compile scaling, structural lowering comparison,
  the shared-callsite regression, and cold-L2
  Qwen3/Gemma4/GPT-OSS/DeepSeek-V3 measurements. Concurrent-launch coverage is
  deferred with the static residency contract above and is not a blocker for
  this static-first PR.

### Worker-grid consolidation decision (2026-08-27)

The cross-loop scheduler will expose one worker-grid tuning dimension:
`num_sm_multiplier`. Static dispatch does not need an independently tuned
absolute worker count. For a device with `S` SMs, the selected grid is exactly
`W = S * num_sm_multiplier`; the ordinary scheduler then derives all task
placement and synchronization from that `W` and the symbolic event DAG.

The first PR retains the existing power-of-two multiplier domain. This keeps
the change focused and avoids expanding the autotuning space of every ordinary
persistent kernel. The multiplier composes with `cross_loop_schedule`; it
controls the grid size for either admitted strategy but does not select the
strategy. A follow-up may make the multiplier a bounded integer-valued choice,
preferably only for cross-loop schedules, after its search cost and
normalization behavior are reviewed independently.

The current workload battery supports removal of the exact-worker knob, while
also documenting why the integer-domain follow-up matters:

| Workload | SM multiplier | Workers on B200 | Cold-L2 observation |
| --- | ---: | ---: | --- |
| Qwen3 decode | 8 | 1,184 | 93.29 us non-cooperative versus 95.40 us cooperative; exact 1,024-worker parent schedule was 89.74 us |
| Gemma4 A4B MoE | 4 | 592 | 47.63 us non-cooperative versus 52.35 us cooperative for the boundary-preserving unfused-GeGLU source |
| Gemma4 E4B decode | 4 | 592 | 102.56 us versus 92.12 us separate; key-major ordering is material here |
| DeepSeek-V3 MoE | 4 | 592 | 182.02 us versus 158.43 us separate; static scheduling remains a known performance gap |
| GPT-OSS MoE | 8 (12 follow-up) | 1,184 (1,776) | Multiplier 8 is about 42 us; the manually admitted multiplier 12 point is 36.35 us versus 37.02 us separate |

Compiled occupancy remains a legality constraint, not a second tuning input.
For example, GPT-OSS cannot residently support multiplier 16 with its selected
root codegen, while multiplier 8 regresses to approximately 42.08 us and the
future multiplier 12 is both legal and slightly faster than the separate
baseline. Therefore GPT-OSS performance parity is a known
limitation of the first power-of-two-only PR, not evidence for restoring an
independent exact-worker knob. Invalid multiplier candidates are discarded
through the same configuration-validity path used for other resource
constraints. This decision supersedes historical sections below that prescribe
graph-snapped exact worker counts or treat `num_sm_multiplier` only as a
capacity hint.

### Cleanup implementation checkpoint (2026-08-27)

The required first-PR simplification work is implemented, with the static
concurrent-residency contract explicitly deferred above:

- Static cross-loop polling kernels use ordinary launches. The launcher rejects
  a configuration when post-ptxas occupancy cannot support its complete grid on
  an otherwise idle device, but this does not reserve capacity against other
  streams. CUDA Graph capture remains covered; concurrent submission is not yet
  guaranteed. Kernels selecting `cross_loop_schedule="barrier"` still launch
  cooperatively.
- The launch-policy A/B holds the generated schedule and grid constant. Gemma
  A4B and GPT-OSS lowerings differ only by removal of
  `launch_cooperative_grid=True`; Qwen additionally has nondeterministic
  constexpr declaration ordering. The event waits, publications, task ranges,
  outlined root bodies, state layout, and minimum-residency check are otherwise
  unchanged.
- Shared DeviceIR graphs retain every owning root, and one failed symbolic
  quotient now coarsens only the affected dependency rather than the complete
  event graph.
- `cross_loop_num_workers`, `preordered_edges`, inferred nested-axis geometry,
  and production task materialization have been removed. The exhaustive
  materialized schedule validator now lives only in test support.
- Cross-loop emission moved out of `program_id.py` into
  `cross_loop_codegen.py`; the scheduler and event relation remain the sole
  semantic inputs to lowering.
- Relative to `18499db5`, this cleanup removes about 355 net production lines:
  `cross_loop_scheduler.py` shrinks by roughly 300 lines, while the large
  `program_id.py` block is moved rather than duplicated.
- The affine-chain compile check remains flat: 65,536 elements lower in
  0.178 seconds to 82 lines, while 1,048,576 elements lower in 0.148 seconds to
  82 lines, with no `tl.where` expansion.

The generic-policy ablation does not justify new user knobs:

- Final-arrival execution and one-axis nested milestones each save about
  10--11 microseconds on Qwen and remain enabled.
- Greedy complete-family placement is neutral on Qwen and Gemma A4B, so it
  remains an internal deterministic policy for this PR.
- Key-major producer ordering costs Qwen roughly 2--3 microseconds and is
  neutral on Gemma A4B, but removing it regresses Gemma E4B from about 103 to
  123 microseconds. It therefore remains as the generic readiness-exposure
  transform rather than becoming a model switch or autotuner knob.
- The 128-byte counter spacing remains a target hardware-layout invariant.

The static schedule is no longer selected implicitly merely because static task
counts happen to be available. `"barrier"` keeps the ordinary phased lowering
and cooperative launch; `"static_pipeline"` opts into the event-based static
dispatcher and its occupancy validation. This makes schedule selection visible
to configuration search and prevents an unsupported static shape from silently
changing semantics or falling back to another strategy.

After the schedule-knob migration, the focused dependency, lowering, runtime,
and Config validation suite passes 191 tests, 4 skips, and 55 subtests. The
symbolic affine-chain probe still lowers in 0.143 seconds to 82 lines, and the
explicit static configurations retain their cold-L2 behavior: Qwen3 is 93.16
microseconds versus 104.73 for separate kernels, Gemma4 A4B is 47.80 versus
53.33, Gemma4 E4B is 96.60 versus 91.04, and GPT-OSS MoE is 42.22 versus
36.72. DeepSeek-V3 MoE is 174.66 versus 158.13. GPT-OSS is part of every
subsequent cross-loop scheduler validation battery. The remaining
PR-preparation work is scope reduction, rebase onto current `origin/main`, and
final validation after conflict resolution.

### Reviewer conclusions and PR boundaries (2026-08-27)

The independent whole-branch review found no model names, model-specific task
counts, or explicit FFN/attention/MoE topology matchers in the compiler path.
Its required correctness work is captured above: runtime residency, callsite-
aware graph ownership, edge-local fallback, explicit nested geometry, and
removal of production task materialization. The review also confirmed that
`preordered_edges` removal, codegen extraction, probe reduction, and worker-grid
consolidation are appropriate first-PR cleanup.

The current quotient discovery is intentionally narrower than a general
polyhedral quotient: it primarily projects source axes already exposed by the
logical tiling and does not always discover derived keys such as `axis // k`.
Safe fallback makes this a non-blocking generalization gap. Documentation must
describe the supported restricted relation algebra accurately; richer quotient
discovery belongs in a follow-up driven by a concrete workload rather than in
the first PR.

Relevant conclusions from the `helion-clc` submission review:

- Symbolic event cardinality and mixed-radix predecessor quotients are already
  addressed by the newer `helion-cross-kernel` relation implementation. Do not
  port CLC's enumerating `uniform_preimage_cardinality()` fallback.
- CLC residency decoupling, CLC compatibility normalization, cancellation
  usefulness policy, ticket state, and command-range construction are not part
  of this PR.
- The current one-nested-axis renderer remains a conservative supported subset.
  Multi-axis rendering should be generalized only when a concrete exact
  relation requires it; unsupported scopes continue to lift to an enclosing
  event or `FamilyDone`.
- Partial placement of large consumer families, such as batched Qwen W2, is a
  future scheduling extension rather than submission cleanup. The current PR
  must preserve compact symbolic expressions and safe fallback for those
  shapes without adding another placement policy.
- Static scheduling must never rewrite a root's indexing, epilogue, or compute
  configuration. It may add only dispatch, waits, publications, fences, and
  state required by the proven dependency schedule.

## Authoritative target architecture (2026-08-25)

This section is the current design contract. All other descriptions of
separate root-task, ordered-action, access-cohort, readiness-frontier, or
counted-event IRs in this document are implementation history and do not
define the target.

The compiler should have one coordinate model and one relation algebra:

```text
DeviceIR execution scopes + LogicalDomain
                    ↓
allocation-coordinate access relations
                    ↓
source-ordered reaching definitions
                    ↓
symbolic LogicalRelation dependency edges
                    ↓
relation quotient into KeyedEvent objects
                    ↓
WorkerSchedule over non-preemptive outer strands
                    ↓
direct relation and worker-stream lowering
```

The design intentionally does not introduce a separate abstraction for nested
"actions." Root tasks and nested loop instances are both instances of an
execution scope over a `LogicalDomain`. The distinction that remains is an
execution constraint: only an outer root strand is independently placeable.
A nested scope projects onto its owning strand and executes there in lexical
program order with its live values intact.

### Minimal retained IR

1. **`LogicalDomain`** describes a bounded Cartesian integer domain: stable
   axis identity, extent expressions that become positive integers for a
   selected configuration, and optional tile geometry. It does not own a
   traversal. Root scopes, nested scopes, event-key spaces, allocation
   coordinates, and worker-stream spaces all use this same value type. Each
   domain carries a semantic kind so equal-rank allocation, event, task, and
   worker domains cannot be composed accidentally.
2. **`ExecutionScope`** identifies a DeviceIR callsite and carries only facts
   not expressible as an access relation: parent and lexical order, owning root
   strand, its logical domain, the projection to that strand, guaranteed-
   execution multiplicity relative to its parent, stable entry and completion
   insertion points, access order, dominance/post-dominance facts, the lowered
   loop and induction-variable mapping, and backend/configuration-dependent
   wait/publication/segmentation capabilities. Stable scope identity remains
   necessary for code insertion and repeated callsites. It is not a second
   dependency graph.
3. **`LogicalRelation`** is the single binary, set-valued relation used for
   accesses, dependency edges, event contribution/use maps, scope-to-strand
   projection, and compact schedule maps. Its direction is always
   `source instance -> set[target instances]`, and its domain and codomain
   kinds must match every composition.
4. **`KeyedEvent`** contains a logical key domain, producer-to-key relations,
   consumer-to-key relations, and covered hazard/program-point provenance.
   Final-arrival execution is a scheduler-owned `LocalTrigger`, not part of
   dependency identity. `FamilyDone` is the same event with a unit key domain.
5. **`WorkerSchedule`** remains the only placement abstraction. Its segments
   map a logical outer-task subdomain to worker-stream positions. Nested scopes
   inherit placement through their scope-to-strand relation.

There should be no separate semantic and lowered event IRs, no action-only
dependency graph, and no schedule type for continuation, direct readiness,
milestones, attention, FFN, or MoE.

### Restricted relation algebra

`LogicalRelation` is deliberately smaller than a general polyhedral system. It
is a normalized finite disjoint union of pieces supporting:

- bounded Cartesian integer domains;
- affine expressions with static integer coefficients;
- floor division and modulo by positive static constants;
- rectangular and residue-class guards;
- target points or Cartesian target ranges `(begin, end, step)` whose
  endpoints and static strides use those expressions;
- exact union and deduplication for equal, disjoint, contained, or
  rectilinearly subtractable pieces;
- restriction to a source or target subdomain;
- composition when the result remains in this grammar;
- inversion for projections, permutations, constant scaling, div/mod tiling,
  and bounded range mappings; and
- symbolic cardinality for disjoint target ranges.

The implementation may use small internal expression/piece records, but they
must remain implementation details of `LogicalRelation`, not parallel compiler
IRs. There is no general satisfiability or equivalence solver: expressions are
normalized affine-div forms with static coefficients and positive static
divisors; guards come only from understood indexing operations; inversion and
composition use explicit closed rewrite rules; and relation-piece growth must
remain proportional to source access pieces rather than runtime points. A
relation that cannot be proved closed in this grammar lifts monotonically to
an enclosing scope or `FamilyDone`. Flatten/unflatten uses affine linearization
and static div/mod; a tile crossing a reshape boundary may become a small finite
union, but a decomposition whose piece count scales with tensor extent is
rejected.

Enumeration is permitted as a differential test oracle during migration. It
must not be the normal compiler representation, an intermediate passed to the
scheduler, or a fallback whose cost scales with a large runtime task domain.

### Direct symbolic dependency construction

Every load and store is interpreted as a `LogicalRelation` from its execution
scope to allocation-coordinate ranges. Producer and consumer access relations
are composed through allocation coordinates to obtain the exact dependency
relation directly. The compiler must not enumerate access footprints and then
attempt to rediscover their structure.

For the one-dimensional producer-16/consumer-32 example:

```text
consumer c -> producer range [2*c, 2*c + 2)
```

For Qwen gate/up to activation:

```text
(batch, group) ->
    (batch, [8*group, 8*group + 8))
  union
    (batch, [768 + 8*group, 768 + 8*group + 8))
```

For the W2 reduction:

```text
(batch, output_tile, k_group) -> activation(batch, k_group)
```

The absent `output_tile` coordinate is therefore dropped by event quotienting
without recognizing an FFN. Unequal tiles and boundary fragments produce
piecewise ranges and symbolic cardinalities. Uniform fan-in is not a legality
requirement.

Views and reshapes are handled before dependency construction:

```text
scope coordinate
    -> logical tensor interval
    -> allocation-coordinate interval
    -> overlapping producer-coordinate range
```

Source-ordered reaching definitions remain authoritative for partial and
in-place writes. They restrict the allocation support on which each composed
relation is valid. Multiple accesses are unioned and deduplicated as relations;
fan-in counts distinct `(producer instance, event key)` pairs rather than
access edges.

### One event construction path

Event formation operates directly on symbolic dependency relations:

1. Form a consumer signature from the normalized producer-range expressions
   required at that program point.
2. Build the semantic quotient from the normalized consumer expressions that
   determine those predecessor ranges. Axes absent from every predecessor
   expression are dropped. The quotient need not be mathematically minimal;
   when equality cannot be proved, retain a finer exact consumer key.
3. Use the inverse relation to derive producer contributions and symbolic
   fanout.
4. Union source-tagged relations for joins and derive per-key fan-in from
   symbolic cardinality after deduplication.
5. Represent root completion by coarsening to a unit key domain.

The event object produced here is also the object consumed by scheduling and
lowering. Eliminate the current `InstantiatedEvent*` versus `CountedEvent*`
duplication and the later task-to-key adaptation pass. Backend artifacts such
as counter offsets, counter storage layout, and selected prefix/count encoding
may exist transiently, but they reference this event and do not copy its
relations.

A monotonic completed-producer-prefix counter is an optional lowering of a
range relation, not part of dependency semantics. It is legal only when the
selected producer traversal proves that the required range is a completed
prefix. Otherwise the same relation lowers to counted keys, bounded fanout, or
`FamilyDone`.

### Nested scopes without an action-specific scheduler

The same relation graph handles root and nested boundaries. A nested scope's
domain includes its enclosing root coordinates and local loop coordinates. Its
scope-to-strand projection drops the local coordinates.

Program-order edges between scopes are also relations. They allow the graph
to prove that a wait at an earlier checkpoint covers a later access and allow
milestones to compose through arbitrary chains. The scheduler does not build
an `ActionDependencyRelation`, `_OrderedActionReadiness`, or a separate action
event graph.

This unification does not make nested iterations movable tasks. A reduction or
matmul loop may carry accumulators, descriptors, and pipeline state across
iterations. Scheduling may insert waits or publications at proved checkpoints
and may mechanically split a representable loop, but it may move only the
complete owning outer strand unless a future transformation explicitly
materializes continuation state.

Relations also cannot replace execution guarantees. `ExecutionScope` must
retain whether a checkpoint executes exactly once and whether synchronization
may legally be emitted there. Conditional branches, dynamic or zero-trip
loops, asynchronous stores without a proved completion point, and unsupported
control flow lift to an enclosing legal scope.

### Symbolic milestones and worker scheduling

Milestones are a schedule-dependent quotient of the same dependency relation:

```text
consumer scope -> required producer region
               -> producer completion round under WorkerSchedule
               -> one admission frontier in the consumer loop
```

The selected worker position partitions a one-dimensional nested loop into at
most two contiguous regions: the prefix whose producers are already complete
at admission and the remaining suffix. This derives Qwen's 64/32 and Gemma's
36/4 handoffs without a cohort or FFN abstraction, while avoiding one cloned
loop per later producer-completion level. If the whole loop has one readiness
condition, lowering inserts one wait around the original loop and does not
split it. When lowering cannot represent the frontier exactly, it coarsens the
dependency.

The semantic quotient above and this admission-frontier quotient are distinct
operations implemented by the same relation algebra. The first groups equal
exact predecessor fibers; the second groups the exact fibers on either side of
the selected schedule frontier.

The implementation must not materialize those fibers. Scheduling needs three
exact reductions over a relation, all kept as private operations of the
restricted relation algebra rather than as another compiler IR:

- project or invert a contribution to obtain `event key -> producer strand`;
- count the distinct producer instances in that preimage for event fan-in; and
- map those strands through `WorkerSchedule` and take the maximum stream
  position needed for readiness.

The result of the last operation is a single-valued symbolic relation from an
event key to a completion position. It is a scheduling query, not a new
semantic graph node or a public "piecewise scalar" abstraction. If an exact
result would require enumerating a runtime-sized domain or modulo period, the
query declines and the event is conservatively coarsened.

`WorkerScheduleSegment` remains the sole placement record, but it becomes a
compact relation from a logical outer-task subdomain to worker-stream
positions. It must not encode a recovered sequence of flattened task IDs.
One root may have several disjoint segments, enabling batch fibers or other
readiness-equivalent regions to begin at different times. The relation is
bijective over non-local tasks and efficiently usable in both directions:
scheduling queries task-to-position, while code generation queries
position-to-task.

Code generation emits segments in the exact global worker-stream order checked
by liveness, rather than regrouping all segments by source root. The following
are mandatory invariants:

- every non-local outer task executes exactly once;
- scheduled task subdomains are disjoint and cover the required domain;
- a waiting worker never owns an unmet producer later in its stream;
- liveness validates the same order code generation emits;
- `FamilyDone` is published once per participating worker after that worker's
  final segment for the root;
- each scope receives the milestone plan associated with its scheduled root
  subdomain;
- fanout lowers as a compact bounded range where possible, not an unrolled
  select forest; and
- nonuniform tail fan-in lowers as a symbolic cardinality formula or a small
  number of pieces, never a key-indexed table.

A completion position alone is not a liveness proof because it discards worker
identity. The production liveness proof is therefore mutation-local and
worker-aware:

1. Start from the source-ordered schedule, which is valid by construction.
2. Before moving or inserting a consumer region, compute its transitive
   prerequisite strand relation symbolically.
3. For every occupied destination worker, reject the mutation if any required
   strand would remain later than the new blocking point on that worker.
4. Recompute the compact prerequisite relation after each accepted mutation.

This also detects mutual cross-worker waits that have the same scalar
completion round. The existing expanded task/event-node validator remains only
as a small-domain differential oracle. It must not run in production on a
domain whose size scales with runtime task or event-key count. Final-arrival
local execution remains a contraction of this graph only when there is one
complete movable consumer task per key, a sole complete prerequisite, and no
nested wait in the consumer strand.

Do not pay for a generic root-tag dispatcher when the final schedule is
root-monotone and contiguous. Code generation should mechanically canonicalize
such a `WorkerSchedule` back to the current simple root loops. Use a compact
component/position dispatcher only for genuinely interleaved schedules. This
is an emitter simplification of one schedule IR, not another schedule type.

L2 traversal is composed only after selecting a logical task. Its transform
must be bijective, including partial final groups. Event-key traversal is also
explicit and independent of key identity; first-seen integer numbering must
not influence contributor order, worker-count breakpoints, or placement.

### Required deletions

The completed migration should remove or subsume:

- `InstantiatedTaskFamily`'s duplicate logical-domain implementation;
- `InstantiatedActionDomain` as a distinct domain abstraction;
- `ActionDependencyRelation.predecessors_by_consumer_action`;
- `_access_predecessor_sets`, `dependency_predecessor_sets`,
  `instantiate_root_predecessor_sets`, and `instantiate_action_relations` as
  enumeration-based APIs;
- `InstantiatedEventContribution`, `InstantiatedEventUse`, and
  `InstantiatedKeyedEvent` duplication with `CountedEventContribution`,
  `CountedEventUse`, and `CountedEventPlan`;
- `instantiate_event_graph`, `canonicalize_ready_events`, and
  `lower_counted_events` as separate expand/recompress stages;
- action/cohort-specific readiness and placement abstractions;
- `TaskToKeySegment`, `_compress_task_to_key`, `_fit_task_sequence`, and the
  balanced `tl.where` diagnostic stopgap;
- materialized predecessor, task-to-key, expected-arrival, and physical-
  traversal tables; and
- action-specific or cohort-specific lowering branches.

`ExecutionScope`, the allocation reaching-definition analysis, stable loop
metadata, mechanical loop segmentation, and synchronization capability checks
remain necessary correctness machinery.

### Migration and validation order

1. Finalize `LogicalDomain` and the restricted `LogicalRelation` grammar with
   exhaustive small-domain differential tests.
2. Construct symbolic access and dependency relations beside the old
   enumerated proof. Verify identity, unequal tile sizes, tails, batch axes,
   offsets/views, joins, fanout, and L2-independent logical identity.
3. Build symbolic `KeyedEvent` objects beside the old event graph. Compare
   dependency relations, semantic quotienting, fan-in, joins, fanout, and
   fallback decisions exhaustively on small domains.
4. Switch event scheduling and direct event lowering to the symbolic graph,
   including per-use final-arrival annotations and symbolic cardinality.
5. Delete predecessor, event, fan-in, and task/key tables together with their
   compressors only after the symbolic event path no longer consumes them.
6. Replace materialized physical traversal with a compact logical-to-physical
   transform.
7. Generalize `WorkerScheduleSegment` while reproducing current batch-one
   schedule structure and resource envelope.
8. Emit and validate global worker-stream order, including split-root
   `FamilyDone` aggregation and per-region milestones.
9. Enable partial root-domain placement and symbolic milestone formation.
10. Delete the remaining legacy event IR and enumeration paths after
    differential, correctness, and performance parity.

### Active implementation task list (2026-08-25)

- [x] Give every event a canonical event-local coordinate chart before event
  identity is assigned. Group equivalent fanout by the complete producer
  contribution signature; never alpha-rename an already-created event.
- [x] Add exact relation operations for source projection, inversion/preimage,
  distinct fiber cardinality, compact worker placement, and fiber maximum,
  with small-domain differential tests.
- [x] Make readiness-frontier construction, direct placement, final-arrival
  selection, key-major ordering, and worker-count snapping consume symbolic
  relations without task/key arrays. Nested scopes now use one symbolic
  admission frontier rather than enumerating every later readiness level.
- [x] Complete the incremental worker-aware mutation proof. Static contributor
  closure follows local triggers and earlier blocking dependencies, same-round
  producers remain eligible on other workers, and adversarial transitive-worker
  tests retain the expanded graph validator only as a small-domain oracle.
- [x] Lower key expressions, event membership, fan-in, and compact fanout
  directly from `KeyedEvent` relations.
- [x] Delete `relation_fibers`, `_materialize_keyed_events`,
  `InstantiatedActionDomain`, `InstantiatedTaskFamily`, `InstantiatedEvent*`,
  the enumerated predecessor/action APIs, `_compress_task_to_key`, and the
  duplicate legacy event/canonicalization passes. `CountedEventPlan` remains
  only as a lowering-selection record referencing canonical event relations;
  it is not a second semantic event IR.
- [x] Replace flattened `WorkerSchedule._placements` and materialized physical
  traversal tables with compact logical-domain schedule relations. Per-task
  lookup remains only for unit tests and the small-domain validator.
- [x] Remove or rename remaining migration-shaped wrappers and stale comments
  where doing so reduces concepts. Dependency-bearing nested loops are scopes,
  not a parallel action/cohort scheduling hierarchy.
- [x] Verify correctness and bounded lowering resources for Qwen batch
  1/2/8/16, Gemma E4B, and Gemma A4B/MoE batch 1/8/16. Batched A4B uses the
  unfused-GeGLU source that preserves the vLLM kernel boundaries; the batch-one
  matched source now rejects unsupported batched use explicitly.
- [x] Verify affine-chain compile-time scaling through 19,968 producer tasks
  and 624 consumer tasks without compile time or generated-code size scaling
  with task count.
- [x] Add uneven/uniform Muse relation coverage, rerun cold-L2 Qwen/Gemma
  benchmarks, and retain representative lowered Triton and NCU evidence.

Latest validation checkpoint (2026-08-25): the Qwen batch-scaling stress source
passes correctness at batch 1/2/8/16 and lowers to 1,286--1,394 lines with only
18--32 `tl.where` sites, independent of the runtime task count. That source
keeps each complete RMS reduction and all 32 quant groups inside one opaque
root; it compiles at 255 registers with 336--456 spill bytes and is not the
Qwen performance source. The intended granular Qwen source exposes the RMS
partial and finalize loops as ordinary roots and compiles at 252 registers
with no spills. Gemma E4B passes at 162 registers with no spilling. The
vLLM-boundary-preserving Gemma A4B/MoE source passes at batch 1/8/16 with
96/80/84 registers and 0/2/0 spill bytes.

Qwen lowering wall time is also flat across batch: batch 1/2/8/16 take
14.35/14.58/14.43/14.50 seconds in fresh Python processes, including roughly
11 seconds of import, tracing, and source setup. The symbolic scheduling and
code-generation increment therefore remains a few seconds rather than scaling
with the 8,192-task batched producer domain. This was achieved by composing
relation pieces directly instead of globally canonicalizing both operands,
grouping equal source guards during fiber reductions, and proving constant or
dominant expressions from finite bounds before asking SymPy to simplify them.

Cold-L2 B200 measurement of the intended granular Qwen source is 89.59
microseconds versus 98.39 microseconds for separate Helion. The saved
pre-refactor lowering `/tmp/qwen3_multi_scope_final_lowered.py` measures 88.86
microseconds under the same cold harness, so the symbolic refactor is at
performance parity within run-to-run noise. Warm medians are 78.04 versus
85.29 microseconds, matching the earlier 78.20-versus-85.52 checkpoint.

The previously recorded 156.50-microsecond Qwen result came from the different
opaque fused-RMS source, not from a scheduler regression. Its lowered Triton
has two 101/102-line noinline roots in which one CTA serially performs the
4,096-element reduction and all 32 quant groups. Every QKV/W13 worker waits on
those singleton roots, and the full kernel spills 336--456 bytes. The granular
source instead emits 32 partial tasks and 32 finalize/quant tasks per RMS
boundary, retains 252 registers with no spills, and recovers the old schedule.
This is a source visibility issue: nested work hidden inside an opaque task
cannot be scheduled independently by the current cross-loop scheduler.

For completeness, the granular Qwen source at batch 2 and 8 currently measures
130.03/384.38 microseconds versus separate Helion at 121.24/300.76; those larger
batches remain performance work rather than a compile-scaling or correctness
failure. Gemma E4B is 107.21 versus 91.21. Gemma A4B batch 1 is 54.83 versus
53.53 for the matched separate baseline; the earlier batch 1/8/16 sweep was
55.21/139.16/236.83 versus 53.16/108.94/223.99. These cold-cache numbers
supersede warm-cache figures when assessing the performance objective.

The affine compile-scaling probe remains flat: 65,536 through 1,048,576
elements lower in 0.186--0.198 seconds to approximately 4.8 KB/82 lines, and
the 19,968-producer/624-consumer Muse-sized case lowers in 0.210 seconds to
4.7 KB/82 lines. No case emits a task-count-dependent `tl.where` forest.
Representative final artifacts are
`/tmp/qwen3_granular_symbolic_final_lowered.py` (the current granular Qwen
lowering), `/tmp/qwen3_multi_scope_final_lowered.py` (its known-good
pre-refactor counterpart), `/tmp/qwen_b1_symbolic_final_lowered.py` (the
opaque fused-RMS diagnostic),
`/tmp/gemma_e4b_symbolic_final_lowered.py`,
`/tmp/gemma_a4b_symbolic_final_lowered.py`, and the matching NCU CSV files
`/tmp/ncu_qwen_symbolic_final.csv`,
`/tmp/ncu_gemma_e4b_symbolic_final.csv`, and
`/tmp/ncu_gemma_a4b_symbolic_final.csv`.

The liveness proof deliberately keeps worker identity. The symbolic scalar
query `event key -> maximum producer stream position` is sufficient to derive
frontiers, but it is not sufficient to prove progress: equal maximum positions
can hide a producer later on the waiting worker or a mutual cross-worker wait.
Before every placement mutation the scheduler must therefore preserve the
correlated `(worker, stream position)` prerequisite relation. This is a
relation reduction over the existing graph, not a new schedule IR.

Relation "fibers" are not retained compiler objects. A fiber is simply the set
of relation targets for one fixed source coordinate. The only required fiber
operations are exact cardinality (event fan-in) and reduction after composing
with worker placement (frontier position). Both return ordinary
`LogicalRelation` values over canonical value domains; no public piecewise-
scalar abstraction is introduced.

Required stress tests are:

- the 65,536-element producer-16/consumer-32 affine chain;
- Muse's approximately 25,500-task small-tile configuration, including the
  uneven 16-by-1,248 and uniform 13-by-1,536 cases;
- Qwen FFN and full layer at batch 1, 2, 8, and 16 with cold-L2 flushing,
  saved lowered Triton, and NCU timelines;
- Gemma E4B schedule/performance parity; and
- affine task-major Gemma A4B at batch 1, 8, and 16 with generated-code size
  and compiler peak memory bounded by relation complexity rather than task
  count.

The existing dense-factorization prototype is diagnostic only: it demonstrates
that direct coordinate rendering removes the `tl.where` explosion, but it
still expands the relation first. Replace it during steps 1--3 rather than
building more compiler behavior on top of it.

### Symbolic migration status (2026-08-25)

- `LogicalDomain` is implemented without an embedded traversal. Scope identity
  and semantic domain kind are part of its type, while physical/L2 order is a
  separate caller choice.
- The dependency and scheduling APIs accept `TileDependencyGraph`, configured
  logical domains, and compact traversal relations. Production no longer has
  `InstantiatedTaskFamily` or `InstantiatedActionDomain`.
- Supported accesses are first lowered independently to
  `scope -> allocation-coordinate` relations. Dependency construction uses a
  generic overlap operation on those relations, rather than a second
  producer/consumer axis-matching proof.
- Ancestor and earlier-sibling program-order coverage is expressed as the same
  `LogicalRelation` projection; the flattened compatibility implementation has
  been deleted.
- The final-form `KeyedEvent` path now performs a consumer-signature
  quotient directly from symbolic relations. It supports unequal tile sizes,
  tail tiles, batch axes, dropped irrelevant consumer axes, disjoint producer
  ranges, multi-producer joins, nested scope endpoints, and unit-key
  `FamilyDone` fallback without per-task predecessor or key tables. Production
  scheduling and lowering consume these relations directly; the materialized
  event adapter and duplicate event canonicalization pipeline are deleted.
- `DependencyPoint` identifies the memory hazard plus both producer and
  consumer callsites. Equivalent exact and family-completion certificates can
  therefore be normalized without conflating independently executed producer
  scopes. Final-arrival eligibility is based on complete obligation coverage,
  not an incidental count of event objects.
- Family-completion fallback is derived as the complete graph obligation set
  minus obligations represented by usable exact events. This covers omitted
  conditional/unknown-domain callsites and failed root projections; fallback
  is no longer tied to whichever affine attempts happened to return `None`.
- Root counted-event selection and emitter-capability filtering are one pass;
  the former `lower_counted_events` adapter has been deleted.
- Final-arrival selection and counted-event emission share the same canonical
  relation-lowerability predicate, so scheduling cannot remove a consumer that
  the emitter later rejects.
- Traversal bijectivity is a tested constructor invariant of
  `physical_traversal_relation` and `inverse().fiber_enumeration()`, rather than
  an expensive generic theorem re-proved for every configuration. Production
  keeps only constant-time typed-domain, size, and nonempty-relation checks;
  small round-trip tests cover axis permutations, L2 full and tail groups with
  outer axes, and multi-piece fiber enumeration. Dead schedule-reporting state
  (`task_ready_edges`, stored local-trigger worker sets, unused endpoint
  wrappers, and the unused `excluded_roots` API argument) has been removed.
- Relation containment now checks arithmetic-progression phase as well as
  bounds and stride. Shifted strided source or target sets cannot be absorbed
  by `union()` or used as false program-order coverage certificates.
- Literal integer subscripts and omitted trailing full slices are preserved in
  the DeviceIR access map. Injective layouts are normalized after removing
  size-one view dimensions, so ordinary `unsqueeze`/size-one aliases retain
  exact allocation-coordinate dependencies.
- Program-order implication is now relation composition: a later access is
  covered by an earlier nested checkpoint only when the preceding-scope
  relation composed with the checkpoint's acquired predecessors contains the
  later dependency relation. Fallback is tracked per dependency point rather
  than coarsening every access between a root pair.
- Counted-event uses preserve partial source domains. Consumer tasks outside
  the use relation perform no wait instead of forcing the entire edge to root
  completion.
- Point-relation composition distributes directly over existing pieces and
  does not first construct a global source-cell canonicalization. Fiber maxima
  group identical source guards and use finite expression bounds to select a
  dominant value before constructing a symbolic `Max`. This removed the final
  model-scale expand/recompress compile-time path while retaining the same
  relation semantics.
- The affine-chain compile probe remains flat at approximately 0.19--0.21
  seconds for 65,536 through 1,048,576 elements, including the 19,968-producer/
  624-consumer configuration. Generated source remains approximately 4.7 KB
  with no task-count-dependent `tl.where` forest.
- The expanded predecessor/event graph remains available only in
  `validate_worker_schedule` as a small-domain oracle. Production scheduling
  has no task-count-dependent relation materialization.
- Qwen batch 1/2/8/16, Gemma E4B, and Gemma A4B batch 1/8/16 lower and pass
  correctness checks. Batched A4B preserves the vLLM-style gate/up, GeGLU, and
  down-projection boundaries.
- Adversarial worker-correlated liveness, Muse compile scaling, bounded batched
  lowering size, uncontended cold-L2 timing, and representative NCU validation
  are complete. Further work is performance optimization, not completion of
  the symbolic dependency/event refactor.

All following architecture sections are dated rationale, experiments, and
migration history. If their prescriptive wording conflicts with this section,
this section takes precedence.

### Current checkpoint

The graph-derived scheduler now lowers the Qwen attention and FFN components
and the Gemma FFN component with canonical logical task coordinates, including
roots whose physical traversal uses L2 grouping. Region-aware reaching
definitions preserve partially overwritten allocation regions, and keyed
events support multiple producer roots, repeated consumer predecessor sets,
counted readiness, and access-local waits. There is no model-name, root-number,
or flattened-task-ID matcher in that path; the custom partitioned-attention
scheduler has been deleted.

The scheduler now plans the complete event DAG. `ReadinessFrontierPlan`, its
single-component topology matcher, and the `tile_dependency_frontier` config
have been deleted. One `WorkerSchedule` records all static placement, including
direct ready-family placement, while `LocalTrigger` records final-arrival
execution.
Access-ready consumer placement recursively contracts arbitrary chains of
local tasks to their static ancestors; it no longer searches for a particular
three-root FFN topology. Multiple independent components and distinct
readiness timelines for batch or other outer coordinates use the same pass.

The only dependency-scheduling tuning input is now
`cross_loop_num_workers`: a requested kernel-wide worker count. Positive values
are snapped to complete event-key boundaries, after which local triggers,
static placement, staged access waits, and root-completion fallback are derived
deterministically. Zero retains the ordinary persistent grid and conservative
access-local fallback. Counted-event lowering is use-oriented and supports one
event feeding multiple consumer roots. Every uniformly keyed root-entry wait
now uses that path, while irregular exact predecessor sets retain the generic
per-task-event fallback. Whole-family completion is represented by the same
one-key event IR; lowering aggregates its logical task completions into one
publication per finishing worker when that is equivalent.

The region-aware dependency proof is the retained foundation. Dynamic task
counts use the existing cooperative phase fallback, while large static task
domains are planned without a task-product cutoff: task-region proofs use a
sorted allocation-interval sweep rather than a producer-by-consumer Cartesian
scan. The former frontier selector and config field have been removed.

The remaining architectural gap is now explicit: the semantic DAG ends at
whole outer-loop tasks, while the most important FFN handoff is recovered by a
parallel `AccessCohortPlan` path that discovers and splits a consumer's nested
loop after ordinary code generation. This is not the intended final design.
Dependency-bearing nested loop iterations must become ordinary ordered work in
the same DAG. Gemma's shared-KV variants remain a root-codegen/resource-envelope
problem: their current fused kernels use roughly 70 KiB of shared memory and
support only two resident CTAs per SM. Do not add dependency-scheduler cases to
disguise that limitation.

The latest uncontended measurements for the retained implementation are about
77.79 microseconds for Qwen versus 84.90 microseconds for its same-source
separate baseline, and about 74.31 microseconds for Gemma sliding non-shared
versus 78.87 microseconds separately. The nested-action refactor should first
preserve these lowerings and resource envelopes; recovering the older Qwen
76.9-microsecond result is not a reason to retain a special scheduler path.

### Gemma A4B MoE stress-test checkpoint

The batch-1 A4B MoE probe validates the full-DAG design on a substantially
different seven-root chain. With source-level assignment-local GEMVs and an
exact hierarchical top-k, the current generic lowering reaches 34.22
microseconds versus 39.05 microseconds for the separately tuned Helion kernels
on an uncontended B200. The hand-written Triton megakernel remains faster at
30.94 microseconds. The generated schedule uses ordinary family completion for
the routing prefix, keyed gate-to-down readiness with fan-in 44, final-arrival
down-to-reduction execution with fan-in 32, and family completion into the
post-norm. No MoE topology or root IDs are recognized by the compiler.

The same source is correct for batches 2 and 8, including nontrivial L2
remapping. Logical event keys retain the batch and route-slot coordinates while
physical task traversal changes independently. A 512-wide reduction tail is
also proved exactly with piecewise fan-in `(64, 64, 64, 64, 64, 32)`. These are
positive evidence that logical task identity, allocation-coordinate overlap,
and the keyed-event algebra are the right semantic foundation.

The stress test also exposes concrete implementation gaps. They should be
fixed within the existing architecture rather than by adding schedule types:

- Every keyed-event counter is now cache-line isolated, rather than applying
  that layout only to ordered-action events. On batch-1 A4B, the eight
  gate-to-down keys receive 44 concurrent publications apiece. The original
  diagnostic improved Helion from roughly 33--35 microseconds to 31.07
  microseconds, matching the 31.02-microsecond hand Triton control. NCU reports
  the same 526 atomic requests but 30% fewer L2 atomic-input active cycles.
  Matched compiler A/B checks show no Qwen regression (79.10 to 79.07
  microseconds) and improve Gemma E4B from 74.13 to 73.56 microseconds, with
  identical register, spill, and shared-memory envelopes. Cache-line isolation
  is therefore a storage invariant, not a model matcher or schedule knob.
- A singleton/coarse producer that contributes to several joined event keys
  still makes counted-event lowering reject an otherwise exact relation. The
  batch-1 probe currently forwards route metadata through the gate/down roots
  to avoid that limitation. The canonical event graph must preserve and lower
  multi-key contributions or factor an already-satisfied coarse prerequisite
  from the fine-grained event.
- Piecewise arrival counts lower correctly for static consumers, but the
  final-arrival executor is not selected for the nonuniform-tail case. A local
  trigger should compare against the proved expected count for its key; a
  nonuniform count is not by itself a reason to change executors.
- Multi-wave worker schedules are represented compactly in Python but can
  expand into long nested `tl.where` expressions when translated back from a
  virtual schedule position to a logical task. Batch 8 emitted a 12-KiB
  expression, and an end-to-end batch-16 lowering probe took about 175 seconds
  and 2.2 GiB RSS. Preserve factored Cartesian coordinates and periodic worker
  mappings through lowering so code size and compile work scale with schedule
  factors, not task count or wave count.
- The grouped MoE kernels initially selected the cooperative dynamic-grid
  fallback solely because the static `max_active_tiles` capacity was not
  specialized in source. Explicitly specializing that bound restores the
  ordinary static event scheduler. This is a source annotation issue today;
  the compiler may later infer such allocation-shape bounds, but it does not
  justify a dynamic-queue schedule.
- The grouped gate and down roots index their intermediate through routing-
  dependent `active_tiles` and `order` arrays. The conservative indirect-access
  rule therefore uses family completion. Prefer a task-major intermediate
  layout in Helion source so both roots expose the shared logical group key.
  Do not add an MoE matcher or a general speculative indirect-index proof. A
  future generic extension may prove structurally identical, immutable index
  expressions, but only if that proof remains local and exact.
- A computed contiguous router slice such as
  `group * group_size + arange(group_size)` is currently recorded as an
  unknown subscript, so router-to-candidate readiness becomes whole-family
  completion. Normalized one-sided access maps should retain this ordinary
  affine vector expression.

The batch-8 assignment-local source is intentionally not a performance target:
it reloads an expert's weights for every routed assignment and measures 142.33
microseconds versus 94.21 microseconds for the grouped standalone kernels. A
verbatim composition of the grouped Helion kernels is also correct but falls
back to family completion across its indirectly indexed intermediate.

A source-level task-major intermediate resolves that ambiguity without a new
compiler policy. The gate root writes activation and routing metadata by
logical expert-tile key; the down root consumes the same key and scatters only
its final weighted result. The existing graph then derives 64 gate-to-down
event keys with fan-in 11. Combined with the hierarchical router, a 256-wide
down tile, per-range staging, and the compiler-derived 286-worker candidate,
the probe reaches about 69.8 microseconds versus 92.9 microseconds for separate
grouped Helion at batch 8 / skew 2. At skew 0 it is approximately tied with the
separate baseline. These measurements were interleaved on a GPU whose foreign
process retained memory but was quiescent; repeat them on a fully uncontended
GPU before treating the exact values as release numbers.

This result strengthens rather than changes the design conclusion: source code
should expose the logical task key when an intermediate is otherwise indexed
through data-dependent routing, and the compiler should schedule the resulting
ordinary event graph. The autotuner owns block widths, range pipeline depths,
register cap, and worker count. It does not need an MoE-specific schedule.

The batch-1 source need not fuse gate/up with GeGLU. Preserving vLLM's explicit
W13-output materialization adds a normal intermediate root; the graph derives
piecewise gate/up-to-GeGLU milestones and a keyed GeGLU-to-down join. Once
keyed counters are cache-line isolated, this source measures 32.88 microseconds
versus 32.68 microseconds for the manually fused source. Keep the boundary in
the source when fidelity is preferred; do not require epilogue fusion as a
compiler scheduling pattern.

### Ordered action domains inside task strands

This section supersedes the earlier temporary rule that only a complete opaque
root task may appear as work in the event DAG. Worker placement remains at
outer-task granularity, but dependency-bearing nested loop work is first-class
for readiness, publication, ordering, and liveness.

The compiler distinguishes two concepts:

```text
task strand     one outer-loop task and its persistent worker assignment
ordered action  one dependency-visible region executed within that strand
```

An ordered action is not an independently movable task. It inherits its
strand's worker, executes in source program order, and carries live values to
the next action. Arithmetic inside the action remains opaque. Only the control
skeleton needed for cross-root memory ordering is visible to the scheduler.

For example, a natural RMSNorm followed by a matmul should be represented as:

```text
RMS strand(row):
    opaque reduction and inverse-RMS prefix
    -> produce normalized group 0
    -> produce normalized group 1
    -> ...

matmul strand(row, output tile):
    initialize accumulator
    -> consume K group 0
    -> consume K group 1
    -> ...
    -> store output

produce(row, group) -> keyed event(row, group) -> consume(row, output, group)
```

This exposes each inner RMS output tile without recognizing RMSNorm and lets a
downstream reduction consume it without recognizing a matmul. It enables
streaming but does not replicate the RMS reduction or move its group actions to
other workers. Recomputation remains a separate source or computation-transform
choice.

The current FFN access-cohort lowering is the consumer-only prototype of this
model. It is driven by a dependent load, not by `hl.dot`: it finds the load's
enclosing loop, maps loop iterations to producer keys, derives their readiness
positions through the existing event graph, combines adjacent iterations with
the same effective readiness frontier, and splits the complete loop body. The
dot and accumulator update are preserved only because they reside in that loop
body. The current path cannot attach publication to a nested producer store,
which is why it cannot express the natural RMSNorm example.

#### One semantic work graph

Build one compressed bipartite graph:

```text
ordered work-action family --contributes--> keyed event
keyed event --required-at--> ordered work-action family

same-strand program order:
action[i] -> action[i + 1]
```

A root with no dependency-bearing nested loop still has exactly one action per
outer task, preserving today's behavior. A nested action domain is identified
by a stable DeviceIR loop callsite path, not merely a graph ID or block ID. A
graph body may have several callsites, and rolling, cloning, branches, or while
loops may change execution multiplicity without changing that body. Each scope
therefore records its owner root, parent scope, lexical position, local axes,
bounds and step, plus whether it is guaranteed to execute and can be segmented.
Its action coordinates consist of the outer task axes plus the relevant
enclosing loop axes. Loads require events at action entry; stores publish events
at action completion. Multiple loads in one action form an ordinary joined
predecessor signature.

The complete Qwen FFN then follows from ordinary graph composition:

```text
W13 tile actions
    -> gate/up keyed event
    -> activation task action
    -> activation keyed event
    -> ordered W2 K actions
```

There is no query for a three-node path and no FFN object. Attention,
reductions, scans, producer-side tiled epilogues, and longer chains use the same
relations.

#### Maximal ordered segments

The semantic graph retains exact action-level dependencies. After choosing the
static worker schedule, propagate readiness through the complete DAG and along
same-strand program-order edges. For each ordered loop domain:

1. Compute every action's exact predecessor-event signature.
2. Include readiness already guaranteed by preceding actions in the strand.
3. Derive the effective producer schedule frontier for each action.
4. Quotient adjacent actions with the same frontier into a schedule-level
   milestone key. Its contributors are the deduplicated union of the exact
   predecessor actions for the complete interval.
5. Combine the maximal contiguous action interval for that milestone into one
   emitted loop segment.
6. Insert one acquire before the segment and retain publications at the exact
   producer action boundaries required by downstream uses.

Qwen's 64/32 and Gemma's 36/4 regions are results of this pass, not fixed stage
counts. They require schedule-derived event aggregation rather than merely
exposing 96 or 40 exact K actions: the two milestone keys have expected arrival
counts `(64, 32)` or `(36, 4)`. Counted-event lowering must therefore support
piecewise/nonuniform expected counts. Without this quotient, the generic graph
would emit dozens of individual polls or lose the existing overlap. More waves,
batch coordinates, several inner axes, and longer dependency chains use the
same prefix-readiness calculation. If a loop domain, program order, carried
state, or event relation cannot be represented exactly, move the wait
monotonically outward to an enclosing action or `FamilyDone`.

#### Simplify `tile_dependency.py` around one relation

Do not add an action-domain layer alongside the current affine and cohort
types. Refactor the existing dependency representation so there is one source
of truth:

```text
DeviceIR loop path and logical coordinates
    -> normalized allocation-coordinate access domain
    -> exact producer-action / consumer-action overlap relation
    -> predecessor-signature quotient
    -> KeyedEvent contributor and use relations
```

Each access should be normalized independently into an allocation-coordinate
mapping over its complete action domain. Producer and consumer axes should not
be paired by a second bespoke proof. The normalized form must retain exact
contiguous intervals and injective coordinate or strided boxes; an address hull
alone is insufficient for noncontiguous views. For a selected static
configuration, evaluate exact access regions, discover overlaps with the
existing sorted interval sweep, and compress the resulting regular relation
for storage and lowering. Affine structure is a one-sided access/address map or
an efficient encoding of the resulting relation, not an independent pairwise
semantic proof.

This should permit deletion or substantial collapse of:

- `AffinePredecessorAxis` and `AffinePredecessorMap` as a separate pairwise
  proof language;
- `ReadinessRequirement`, whose task/root distinction becomes a property of
  the proved action relation or its conservative fallback;
- duplicate `predecessor_task_ids()` and `_edge_predecessor_sets()` relation
  evaluation;
- `UniformTaskPartition` as a second proof of dependency membership; any
  uniform partition needed for compact lowering should be derived only from
  the canonical action-to-key relation;
- dimension-pairing special cases for size-one views and storage offsets after
  those views have been normalized into allocation coordinates; and
- the parallel raw-wait selection path currently needed to recover information
  discarded by root-task canonicalization.

This does not require a general symbolic-set solver. Exact affine rectangular
or strided regions are supported; unknown, indirect, masked, dynamic, or
non-injective cases conservatively use an enclosing action or whole-family
completion. Batch and every other non-singleton outer axis remain part of the
logical action coordinates and therefore of event identity. L2 remapping is
applied only after those logical relations are built. Configuration-time
enumeration may validate a concrete relation, but the retained IR must compress
regular domains rather than store one object per runtime action; W2 alone can
otherwise create tens of thousands of action instances.

#### Scheduler changes should be focused

The existing event algebra and most scheduling policy remain:

- `KeyedEvent` still represents joins, fanout, exact readiness, and
  `FamilyDone`.
- `WorkerSchedule` still places outer task strands only.
- A local final-arrival executor still applies only to a complete movable task
  start, not to arbitrary inner actions.
- Inner actions inherit their strand's worker and contribute same-worker order
  edges to readiness and deadlock analysis.
- Completing any action may publish outgoing events, so chains compose through
  any number of root or nested-loop boundaries.
- Direct placement and liveness operate on task strands while consulting all
  action-level waits and publications.

The event algebra and outer placement policy survive, but this is more than a
mechanical type substitution. Event contribution/use relations, readiness
grouping, liveness, cycle validation, and counted-event lowering must all
understand action endpoints and same-strand order. The current event
canonicalization should be generalized rather than duplicated.
`canonicalize_task_readiness()` and `canonicalize_ready_events()` should
converge on one predecessor-signature quotient over action program points. The
current recursive local-ancestor and worker-cycle proofs should consume that
graph with same-strand edges instead of learning about a new scheduling mode.

A locally triggered root that later blocks at a nested action may occupy one of
the workers required to produce that action's event. Initially forbid local
execution for such a strand unless the action-level liveness proof establishes
progress. This is a generic non-preemptive-worker rule, not a topology case.

#### Producer publication and lowering safety

A nested producer action may publish only at a boundary guaranteed to execute
exactly once for each counted logical action. Conditional branches, while
loops, zero-trip or dynamic loops, duplicated callsites, and ambiguous control
flow must initially lift publication to a guaranteed enclosing scope or
`FamilyDone`.

Publication must follow CTA-wide completion of every store represented by the
action and use release ordering. For pipelined, unrolled, or flattened
`tl.range` loops, publication belongs between emitted loop segments after any
required pipeline completion; inserting an atomic immediately after a
syntactic store may publish while asynchronous work remains in flight. Segment
boundaries must align with the selected physical inner-loop traversal while
event identity remains in logical coordinates.

Several stores from one action that satisfy the same key contribute once, not
once per store. Later overlapping stores remain reaching definitions and must
not be hidden by an earlier publication. The generic lowering may reuse the
existing loop-cloning and body-fingerprint helpers, but loop selection is by
stable scope identity and all waits/publications come from the final action
graph.

#### Required deletion outcome

The refactor is not complete while any of these remain as a parallel policy:

- `AccessCohortPlan`;
- `_derive_access_cohorts`;
- `place_access_ready_consumers`;
- `EventUse.placement == "access"`;
- cohort-specific counter allocation and publication;
- cohort-specific exclusions from counted-event lowering;
- late access-marker-based loop discovery; or
- cohort-specific AST splitting.

The final lowering may still segment a generated loop, but it must do so as a
mechanical rendering of generic ordered action segments identified by stable
DeviceIR loop IDs. It must not search the lowered AST for a characteristic
load, dot, reduction, model, or root topology.

#### Implementation sequence and gates

1. Build the reachable DeviceIR execution-scope tree from root callsites.
   Record stable callsite paths, parent scopes, logical action axes, bounds,
   steps, source order, and execution/segmentability proofs for every
   cross-root load and store. Add graph-only tests for sibling loops sharing
   block IDs, repeated graph bodies, nested loops, branches, and rolled graphs.
2. Replace the pairwise affine-predecessor proof with one normalized
   action-domain overlap relation. Preserve conservative fallback before
   deleting the old proof.
3. First use the new relation to reconstruct the existing root-only event graph
   and differentially compare every current affine test, Qwen, and Gemma. Only
   then delete the old pairwise proof and duplicate scheduler-side relation
   evaluation.
4. Generalize event contributors and uses to action domains. Verify joins,
   fanout, partial writes, outer batch axes, and arbitrary-length chains before
   changing scheduling.
5. Add same-strand order/affinity to readiness, liveness, and cycle checking.
   Keep outer task placement and existing local triggers unchanged initially.
6. Add generic schedule-derived event quotienting and maximal ordered loop
   segments. Support piecewise arrival counts and verify that this reproduces
   Qwen 64/32 and Gemma 36/4 without a cohort object.
7. Lower generic action waits and publications using stable callsite identity,
   preserving loop bodies, range attributes, accumulator state, and outlined
   root ABIs.
8. Delete the complete cohort path and the old affine/event duplication only
   after structural lowered-code equivalence and correctness tests pass.
9. Validate the natural non-recomputed RMSNorm-to-matmul source as a producer-
   side streaming case, while treating any decision to replicate its reduction
   as an independent computation optimization.
10. Re-run Qwen and Gemma correctness, lowered Triton inspection, resource
   usage, repeated CUDA Graph replay, and uncontended performance. Do not add a
   model-specific escape hatch to recover a small latency difference.

### Canonical full-DAG event scheduler

This section, together with the ordered-action-domain section above, is
authoritative for the next compiler refactor. The compiler is not an FFN
scheduler, an attention scheduler, or a collection of multi-root patterns. It
constructs one semantic event DAG and one static cross-loop schedule. The
familiar three outcomes are represented by two orthogonal choices rather than
three unrelated schedule types:

```text
readiness: exact keyed event | whole-family completion event
executor:  final arrival     | static worker schedule

local            = exact event + final-arrival executor
direct           = exact event + static worker schedule
root completion  = family-completion event + static worker schedule
```

`Continuation` is not a fourth mode. It is the historical name for the local
case, where the final contributor executes the ready task. A readiness
frontier is likewise not a mode or a separately selected schedule. It is a set
of event keys that happens to become ready before later keys under the chosen
worker count and deterministic task order.

#### Architectural layers

Keep correctness, scheduling, and lowering separate:

```text
DeviceIR task strands, ordered action domains, and opaque action bodies
    ↓
Allocation-coordinate accesses and reaching definitions
    ↓
Pairwise producer-action -> consumer-action relations
    ↓
Full keyed-event DAG and event uses
    ↓
CrossLoopSchedule with ordered WorkerSchedule relations and local triggers
    ↓
One persistent Triton kernel with local/direct/root-completion execution
```

The dependency analyzer remains pairwise. Full-DAG behavior comes from
composition: completing a consumer task contributes to its outgoing events,
which may enable later tasks. There is no search for a path of length two, no
fixed lookahead depth, and no matcher for `projection -> activation ->
projection` or `attention partial -> merge -> projection`.

The planner operates on the complete DAG. Whole-family completion is an
explicit event, not an assumption that deleting one coarse edge disconnects
the graph. A consumer may require both an exact event from one reaching
definition and whole-family completion from another, and an exact path may
still connect the endpoints of a coarse dependency. Worker-reuse intervals or
independent scheduling regions are derived only after the schedule is stable.

"Full DAG" does not mean materializing one compiler object for every runtime
task. Task families, key domains, contributor/use relations, and worker
schedules remain symbolic or run-length compressed. Configuration-time task
enumeration is only a bounded proof technique for access overlap; the retained
schedule IR must scale with regular relation segments rather than tensor
element counts.

#### Minimal dependency IR

Retain only three semantic concepts:

1. `TaskFamily`

   - Root ID and source order.
   - Logical coordinate domain, including batch and other outer axes.
   - Stable ordered action domains for dependency-bearing nested loops.
   - A projection from every action coordinate to its owning outer task strand.
   - Reads and writes in underlying allocation coordinates.
   - Stable DeviceIR program points for action entry and completion.

   Physical PID mappings, opaque root bodies, and their generated ABI remain
   attached to DeviceIR and are bound by the scheduler/lowering for a selected
   configuration. They are not dependency semantics.

2. `KeyedEvent`

   - An independent logical key domain; it is not owned by one root.
   - Contributor relations mapping a producer task to zero, one, or multiple
     keys.
   - An exact expected-arrival count per key, represented compactly when it is
     uniform or piecewise uniform.
   - Required publication/acquire ordering.

   Whole-family completion is not another event type. `FamilyDone(family)` is
   the canonical one-key `KeyedEvent` to which every logical task in the family
   contributes exactly once. Lowering may aggregate those logical completions
   by worker schedule when that is proven equivalent.

3. `EventUse`

   - The consuming action domain and logical action coordinates.
   - The stable DeviceIR action-entry program point at which readiness is
     needed.
   - A relation mapping event keys to the action instances that require them.

The execution planner produces one `CrossLoopSchedule` containing:

```text
WorkerSchedule
    (worker, symbolic schedule position) -> logical task instance

LocalTrigger
    (event, key) -> logical task instance
    possible executor-worker set

waits and publications attached to stable task program points
```

`WorkerSchedule` is the single source of truth for statically assigned work.
The inverse `TaskPlacement(task) -> (worker, position)` is a query, not a
second stored IR. A schedule position may be an affine or lexicographic tuple,
and worker schedules may contain compressed loops; no per-runtime-task list is
required. Dispatch rounds are projections of this order, not an independent
schedule representation.

Local and direct retain separate zero-overhead emission. Local consumes the
return value of the producer-side atomic and invokes the task on the final
arrival. Direct is simply a task in `WorkerSchedule` with an exact acquire wait
at its required program point. Ordinary producer traversal and direct
consumers therefore share one static scheduling representation.

A logical task publishes the same outgoing contributions regardless of how it
was executed. In particular, `FamilyDone` receives exactly one semantic
completion from every task in its family even when some tasks are statically
scheduled and others are local. Physical lowering may coarsen this to one
arrival per worker only after proving that the worker's assigned tasks are
complete; active-worker counts are not the semantic definition.

The only independently movable executable unit is one complete logical root
task. Dependency-bearing nested actions are nevertheless first-class graph
nodes: they inherit the owning task strand's worker and carry explicit
same-worker, program-order, and live-state continuity. Making an action visible
to the dependency graph does not permit moving it to another worker.

This deliberately eliminates separate `ReadyAction`, `DirectAction`,
`TaskAction`, root-completion-plan, task-placement, and dispatch-round records.
Their information already lives in `TaskFamily`, `KeyedEvent`, `EventUse`, or
the ordered `WorkerSchedule`. `LocalTrigger` remains separate because its
executor worker is selected by the runtime final arrival; hiding that
difference behind a common executor wrapper would not simplify the proof or
the lowering.

Do not put `key_root`, `on_ready_root`, a downstream root, a frontier size, or
worker IDs in the semantic event. Those are properties of uses and the static
execution plan. `UniformTaskPartition` may remain as a compact proof format,
but it must not define which topology the scheduler recognizes.

#### Dependency construction remains local and exact

For each direct reaching-definition edge, determine the producer tasks needed
by every consumer task or access:

```text
write_region(producer_task) intersects read_region(consumer_task)
```

Construct these relations in logical task coordinates, not lowered PID order.
Normalize ordinary views and flatten/unflatten operations into allocation
coordinates first. Preserve indirect dimensions as unknown rather than
inventing an affine relation.

Then quotient consumer uses by identical predecessor signatures. This creates
the smallest natural event key space without model knowledge. The relation may
contain:

- one-to-one tasks;
- many producers per key;
- one producer contributing to several keys;
- several producer roots contributing to one key;
- several consumer tasks sharing one key;
- partial producer domains; and
- different outer coordinates such as batch, head, expert, or sequence tile.

If predecessor membership, coverage, or expected arrivals cannot be proved,
require the relevant `FamilyDone` event at that use. WAR and WAW hazards remain
ordering facts; local execution is permitted only when all prerequisites of
the invoked task are represented and the task may be reordered safely.

#### Concrete graph-pass algorithm

Represent the schedulable graph as a symbolic bipartite DAG:

```text
ordered action-family instance --contributes--> KeyedEvent
KeyedEvent --required-at--> ordered action-family instance
```

An action instance is a coordinate in a logical action domain, not a separately
allocated compiler object. Root actions are independently movable; nested
actions inherit their task strand's worker and source order. The graph remains
compressed by action domains, affine or run-length relations, and key domains;
it must not allocate one compiler node per runtime action.

Build and plan this graph with the following semantic and scheduling passes.

##### Pass 1: direct reaching-definition relations

Scan accesses by underlying allocation and source order. For every consumer
access, retain the exact reaching definitions of the regions it may read or
overwrite. Emit only direct relations:

```text
TileDependency
    producer task domain
    consumer task/access domain
    producer-task -> consumer-use relation
    RAW, WAR, or WAW kind
```

`TileDependency` is a symbolic relation over task domains, not one materialized
runtime edge. `AccessDependency` may remain the lower-level source-ordered
memory hazard from which these task relations are derived.

For `A -> B -> C`, this pass creates `A -> B` and `B -> C`. It creates
`A -> C` only when C directly accesses a definition still supplied by A.
Transitive root reachability is never treated as a substitute for a region
dependency.

Partial writes naturally retain multiple reaching definitions. If A supplies
one region and B overwrites another region later, a C task that reads both has
contributors from A and B rather than a fictitious whole-allocation latest
writer.

##### Pass 2: event canonicalization

For each consumer action program point, collect its complete direct predecessor
signature across all incoming relations. Quotient structurally equivalent
signatures into an event key domain and create:

```text
KeyedEvent
    contributor relations
    expected arrivals per key

EventUse
    event-key -> consumer-action relation
```

This canonicalization may join contributors from several roots, share one
event among several uses, or leave a producer task contributing to several
keys. If all prerequisites of one movable root-task action are exact, combine
them into one joined event so local execution has a single final arrival.
Nested actions use the same joined events but retain their parent worker. If
any required relation is unknown, require the relevant canonical `FamilyDone`
event at the nearest safe enclosing action instead of inventing a partial
trigger. Exact and whole-family prerequisites may coexist at one use when
different reaching definitions have different proof precision.

Completing a consumer action contributes to its outgoing events in exactly the
same way as an original source root action. This is the only mechanism needed
to compose dependencies across several root or nested-loop boundaries.

##### Pass 3: instantiate the baseline worker schedule for one `W`

Bind logical task families to their selected physical PID mappings and opaque
bodies. Seed `WorkerSchedule` with the existing deterministic persistent
schedule. A task's schedule position records its complete order on one worker,
not merely its worker ID:

```text
schedule_position = (phase, family order, local round, intra-round order)
```

The tuple may be simplified when the existing schedule already supplies one
affine rank. For the usual strided distribution, `worker = rank % W` and the
local round begins with `rank // W`. The retained representation stays
symbolic or run-length compressed.

##### Pass 4: form legal local triggers

Contract an exact event use into `LocalTrigger` only when it enables one
complete movable task per key, represents all of that task's prerequisites,
and preserves its ordering constraints. Remove those task instances from their
static worker schedule and record every worker that may make the final arrival.

Local work is not free. Its possible executor-worker set remains live through
the task and its outgoing publications. Fanout, competing uses, or access-local
state remain ordinary waits in static worker schedules unless a stronger proof
is added later.

##### Pass 5: construct and validate static worker schedules

Place remaining task instances into ordered `WorkerSchedule` positions. An
exact direct use is a static task whose acquire wait occurs after every
progress-critical task that the same worker must perform to satisfy that or
another wait. Several tasks may share one key, one worker may execute several
ordered tasks, and a task family may span any number of rounds.

For a candidate schedule, propagate readiness forward over event relations and
worker positions, then propagate worker liveness backward. Track:

- unfinished producer positions needed by future events;
- static consumers from their first blocking wait through final publication;
- the possible executor sets of local triggers;
- stable arithmetic mappings that must retain worker affinity; and
- positions after which worker IDs may be reused.

This is schedule dataflow, not path matching. In particular:

```text
P -> event A -> M -> event B -> C
```

is handled because M consumes event A and its completion contributes to event
B. The pass never asks whether P, M, and C form a known three-root pattern.
Dispatch rounds and readiness frontiers are derived views of the completed
worker order; exact counters, not predicted task latency, decide runtime
readiness.

##### Pass 6: monotonic fallback, stabilization, and lowering

If exact local or static placement cannot prove coverage, ordering, progress,
or residency, replace that use's exact materialization with the relevant
`FamilyDone` event and rebuild the affected worker schedule. Each use can only
move from exact to whole-family readiness, so this process is finite and does
not require a profitability heuristic.

After no use changes, derive independent reuse intervals from the finalized
worker schedules. Do not derive regions by deleting coarse dependency edges:
an exact path may still connect the same families. Validate that every waiter
and every worker required to satisfy it fit within compiled resident capacity,
then lower common event-state allocation, release publication, acquire waits,
local final-arrival dispatch, and static worker schedules mechanically.

##### Key-level transitive simplification

The graph pass may remove a direct wait only through readiness dominance at the
same logical key granularity. If waiting for event B proves that the exact A
keys required by a consumer were prerequisites of every contribution that made
B ready, the A wait is redundant. A root-level path `A -> B -> C` alone is not
such a proof.

Conceptually, propagate a monotone set of guaranteed readiness facts. A task
completion inherits the exact keys it waited for. An all-contributor event
inherits the facts guaranteed by its required contributor tasks. A fact may
dominate a later use only when every execution that can make the prerequisite
event ready includes that fact.

Implement this over compact event/key relations rather than materialized task
sets. Root-completion facts may dominate a complete producer root; partial
events dominate only the keys and regions they actually cover.

A useful implementation skeleton is:

```text
relations = build_tile_dependencies(reaching_definitions)
event_graph = canonicalize_events(relations)

for W in legal_worker_counts(event_graph, compiled_capacity):
    materialization = initial_exact_and_family_done_uses(event_graph)
    while True:
        worker_schedule = seed_existing_schedule(event_graph, W)
        local_triggers = form_local_triggers(
            event_graph, materialization, worker_schedule
        )
        worker_schedule = place_static_tasks(
            event_graph, materialization, worker_schedule, W
        )
        readiness = forward_readiness(event_graph, worker_schedule)
        liveness = backward_worker_liveness(
            event_graph, worker_schedule, local_triggers, readiness
        )
        failed_uses = validate_progress_and_residency(
            worker_schedule, local_triggers, liveness
        )
        if not failed_uses:
            break
        materialization = downgrade_to_family_done(
            materialization, failed_uses
        )
    lower(worker_schedule, local_triggers, materialization)
```

#### Full-DAG composition, not multi-root pattern matching

An event use enables an opaque action. A root-entry action may begin a movable
task; a nested action resumes its existing task strand on the same worker.
Completing either publishes its outgoing event contributions. Transitive
pipelines therefore emerge by ordinary graph propagation:

```text
producer actions
    -> event A
    -> intermediate actions
    -> event B
    -> downstream actions
```

The analyzer never asks whether this is an FFN, attention, RMSNorm, or a
matmul. It also never asks for exactly one intermediate root or one nesting
depth. Chains, diamonds, joins, fanout, and producer-to-consumer inner-loop
pipelines all use the same representation.

For a selected resident worker count `W`, assign each independently reorderable
producer task a deterministic schedule rank. The default is its existing
physical traversal. A key-major permutation is legal only when all affected
tasks are independent and the outgoing event relations define one compatible
order; otherwise preserve the existing traversal. A task's dispatch round is:

```text
round(task) = schedule_rank(task) // W
```

An event key's producer round is the maximum round of its contributors. This
is a static precedence class, not a prediction that equal-cost work finishes at
the same clock time; exact counters still determine runtime readiness. Every
intermediate task remains real work in either `WorkerSchedule` or a
`LocalTrigger`, and only its completion publishes outgoing events. The planner
must never treat it as zero-cost merely to propagate a round number.

Adjacent iterations of an existing consumer reduction loop with identical
prerequisite event classes may be coalesced into one wait region inside the
same scheduled task. This is how the Qwen 64/32 and Gemma 36/4 reduction
regions arise. They are not separately movable actions, candidates, or model
constants:

```text
Qwen:  1024 first-wave producer tasks / 16 arrivals per key = 64 keys
Gemma:  576 first-wave producer tasks / 16 arrivals per key = 36 keys
```

The lowering may insert waits and preserve accumulator state at such a proved
access boundary. It must preserve the same worker, program order, loop body,
tile sizes, range attributes, loads, dots, and stores. It may not turn those
regions into independently movable tasks or introduce a different computation
kernel merely because a different schedule would be convenient.

#### DAG-wide worker-schedule liveness

The scheduler must see all live task families and local-trigger executor sets
in the complete DAG, because a purely edge-local placement cannot reserve
downstream workers while upstream tail work continues. This is global resource
allocation over ordered worker schedules, not dependency inference across
multiple roots.

For every dispatch round or event interval, compute:

```text
occupied workers = workers with unfinished progress-critical schedules
slack workers    = resident workers - occupied workers
```

Existing arithmetic worker mappings are anchors. Long-running task families
retain stable assignments; no global cursor may dynamically rank workers and
destroy that affinity. Worker IDs are reusable after a family's liveness
interval ends.

The placement rule is deliberately conservative:

- `local`: the final arrival executes the ready task. This is the default and
  reserves no separate worker ID, but it extends liveness over every worker
  that may be the final arrival. Initially require exactly one complete opaque
  consumer task per key, complete representation of its other prerequisites,
  and permission to reorder that task. Repeated fanout uses are not silently
  bundled onto the final producer.
- `direct`: place the ready task instances at predetermined positions in
  `WorkerSchedule`. Several tasks may share one key, one worker may execute
  several ordered tasks, and there is no queue or runtime task-ID load.
- `root completion`: replace the exact prerequisite with the canonical
  `FamilyDone` event when exact readiness, ordering, liveness, or residency
  cannot be proved. The task remains statically scheduled.

If several exact uses compete for the same schedule positions and cannot all
fit with valid liveness intervals, preserve eligible local triggers and
downgrade the unsafe uses to `FamilyDone`. Do not choose among graph branches
with a model-sensitive priority or cost heuristic. All waiting workers, all
possible local executors, and every worker needed to make their prerequisites
progress must fit concurrently under the driver's compiled-kernel occupancy
result; otherwise reject the plan or use whole-family completion.

For the two measured FFNs, the familiar formula is only a closed-form summary
of this liveness calculation:

```text
I = W - T - C
```

where `T` is unfinished producer demand and `C` is reserved downstream demand.
The implementation must operate on worker liveness sets and arbitrary numbers
of waves, not contain this formula or assume a three-node chain.

#### Concrete traces through the general policy

Qwen FFN:

```text
1536 projection tasks -> 96 activation keys, fan-in 16
first 1024 tasks       -> 64 keys complete in dispatch round zero
remaining demand       -> 512 producer tasks
downstream demand      -> 512 output tasks
slack                  -> zero
```

Activation therefore executes locally. The 512 downstream workers process the
ready reduction region while the other 512 workers finish projection, then
finish the remaining region when the final 32 keys become ready.

Gemma FFN:

```text
640 projection tasks -> 40 activation keys, fan-in 16
first 576 tasks      -> 36 keys complete in dispatch round zero
remaining demand     -> 64 producer tasks
downstream demand    -> 160 output tasks
slack                -> 352 workers
```

The complete 40-task activation family fits in stable slack, so tasks use
direct keyed execution. The first 36 activations enable the first reduction
region; the last four enable the final region. This is the same graph policy as
Qwen with a different liveness result.

Attention partials and merge:

```text
partial task: (batch, kv_head, split)
merge task:   (batch, kv_head, query)
predecessors: every split for that batch/head/query
```

The event fan-in is the split count. Multiple query tasks may share the same
producer signature. A one-task-per-key merge may run locally. Repeated uses of
one key form a direct task family when that family fits; otherwise they retain
root-completion ordering. The first implementation must not make one final
partial task execute an arbitrary fanout serially.

Attention merge and output projection:

```text
merge writes: (batch, query_head, head_dimension)
O reads:      flattened reduction intervals
```

Allocation-coordinate normalization maps each O reduction interval back to
the merge keys that supply it. Equal-readiness intervals may become scheduled
reduction regions. If that flatten/unflatten relation is not exact, O waits for
merge root completion.

Output projection and post-attention RMS:

```text
event key:    batch and other outer coordinates
contributors: all O output tiles for that key
ready task:   one post-attention normalization/residual task
```

This is an ordinary many-to-one event, usually executed locally by the final
arrival. No reduction-specific scheduler primitive is needed.

KV-cache update and paged attention remain an important conservative case.
Runtime block tables and slot mappings may prevent proving which attention
split reads a newly written cache location. Until an exact effect relation is
available, that use requires the cache writer's canonical `FamilyDone` event
rather than an attention-specific exception. Other exact paths in the same DAG
remain available and are not discarded merely because this use is coarse.

#### Autotuning surface

The dependency graph determines keys, fan-in, legal task orderings, readiness
classes, liveness, and whether local/direct execution is safe. These are not
tuning knobs.

The only dependency-related performance input should be the total persistent
worker count `W`, represented as an ordinary kernel grid choice and constrained
by compiled occupancy. `num_sm_multiplier` may supply a capacity target or
autotuner seed, but it need not equal the final grid size. Generate legal `W`
values by snapping that capacity and task-count breakpoints to graph-derived
alignment requirements: complete event keys, consumer reduction-tile
boundaries, and forward-progress constraints. This produces 1,024 for the
measured Qwen schedule and 576 rather than the unaligned 592-worker capacity
for Gemma.

Given `W`, ready groups and worker placement are derived deterministically.
`G` is a result of `W` and the contributor mapping, not another knob. Remove
`tile_dependency_frontier`; it has now been replaced by
`cross_loop_num_workers`. Do not add local/direct toggles, producer-order knobs,
raw cohort sizes, or model-specific schedule choices.
For a DAG with several independent or partially connected subgraphs, form one
kernel-wide finite set from the capacity endpoint and every task/event
relation's legal aligned breakpoints. Each `W` still uses the same planner; a
use that cannot form a resident exact schedule at that `W` uses `FamilyDone`.
There are no per-subgraph schedule indices.

If future evidence shows a legal local/direct choice that cannot be resolved
by complete-family slack, the only acceptable extension is a small per-event
placement tuning field over compiler-proved alternatives. Do not introduce it
for Qwen or Gemma, where the deterministic rule is already validated.

#### Required compiler restructuring

The existing code has the right access and logical-coordinate foundation but
mixes event semantics with one topology. Refactor it as follows:

##### Naming and module boundaries

Use `tile` for the semantic unit whose dependencies are being represented, and
`cross_loop` for the compiler phase that schedules those tasks across top-level
loops:

```text
tile_dependency.py         semantic dependency/event graph
cross_loop_scheduler.py    configuration-time planning and Triton lowering
program_id.py              generic PID/body services delegated to the scheduler
```

Prefer `cross_loop_scheduler.py` over `tile_scheduler.py`. Helion already has
`tile_strategy.py` and `tile_dispatch.py`, which select tilings and loop
execution strategies. A `tile_scheduler` name would make the cross-root event
scheduler sound responsible for those existing concerns.

Keep the module split small initially. `cross_loop_scheduler.py` may contain a
pure `build_cross_loop_schedule()` phase and a separate
`lower_cross_loop_schedule()` phase in the same file. Extract a third lowering
module only if the implementation remains large after the frontier-specific
code is deleted.

Recommended symbol renames:

| Current name | Target name |
|---|---|
| `cross_loop_dependencies.py` | `tile_dependency.py` |
| `CrossLoopAccess` | `TileAccess` |
| `CrossLoopDependencyEdge` | `TileDependency` |
| `CrossLoopDependencyPlan` | `TileDependencyGraph` |
| `build_cross_loop_dependency_plan` | `build_tile_dependency_graph` |
| Lowered access markers | delete; use stable DeviceIR loop/action identity |
| `tile_dependency_planner.py` | `cross_loop_scheduler.py` |
| `GenericSchedulePlan` | `CrossLoopSchedule` |
| `build_generic_schedule_plan` | `build_cross_loop_schedule` |
| `KeyedEventPlan` | `KeyedEvent` |
| `KeyedEventContributorPlan` | `EventContribution` |
| `WaitSpec` | `EventUse` once it carries the independent use relation |
| Separate ordinary/direct traversals | one symbolic `WorkerSchedule` relation |
| Local continuation schedule records | `LocalTrigger` |
| `ReadinessFrontierPlan` and `ReadinessFrontierSelection` | delete |
| `TILE_DEPENDENCY_FRONTIER_CONFIG` | delete |

`TileDependencyKind`, logical task geometry, and allocation identity remain
useful. Refactor `TaskFamily` to own or reference its ordered action domains
rather than adding a parallel inner-loop hierarchy. `UniformTaskPartition` may
survive only as a compact lowering view derived from the canonical action/key
relation; it must not remain a second legality proof. `EventUse` names a stable
DeviceIR action endpoint directly. Delete `AccessProgramPoint` and lowered AST
access-marker discovery so there is only one notion of source program position.

The generated runtime state should use event terminology rather than the
deleted schedule topology: for example `cross_loop_event_state`,
`cross_loop_epoch`, and `cross_loop_event_wait`. Do not preserve names such as
`continuation`, `frontier`, or `pipeline` in the new path except when referring
to historical tests or measurements.

Tests should follow the same boundary:

```text
test_tile_dependency.py         access, reaching-definition, and relation proofs
test_cross_loop_scheduler.py    readiness, liveness, and placement
test_cross_loop_codegen.py      emitted Triton and runtime integration
```

Perform these renames while introducing the new graph records rather than as a
standalone mechanical change followed by another rewrite. Temporary internal
re-exports are acceptable during migration, but remove them with the old
frontier planner so the final compiler has one vocabulary.

- `tile_dependency.py` (currently `cross_loop_dependencies.py`)
  - Retain allocation regions and reaching definitions, but replace the
    pairwise producer/consumer-axis proof with one normalized access-domain
    representation over root and nested-loop coordinates.
  - Replace producer-owned `EventSpec` plus consumer `WaitSpec` as the final
    representation with event relations that support independent contributor
    and use maps.
  - Keep root-completion requirements for unknown relations.

- `cross_loop_scheduler.py` (replacing `tile_dependency_planner.py`)
  - Bind selected configuration geometry with `LogicalDomain`; keep physical
    PID order in a separate `LogicalRelation` traversal rather than duplicating
    either concept in a scheduler-specific task-family type.
  - Make `KeyedEvent` independent of `key_root` and remove `on_ready_root`.
  - Replace `AccessCohortPlan`, `ReadinessFrontierPlan`, and
    `ReadinessFrontierSelection` with generic action event uses, one ordered
    `WorkerSchedule`, local triggers, and worker-liveness analysis.
  - Remove `_select_readiness_frontier` and the assumption that exactly one
    counted event feeds exactly one downstream cohort.
  - Retain compressed task/action-to-key segments only as derived relation
    encodings.
  - Treat `FamilyDone` as a canonical one-key `KeyedEvent`, not a second
    synchronization hierarchy.
  - Derive task placement and dispatch rounds from `WorkerSchedule`; do not
    store duplicate inverse mappings or round lists.

- `program_id.py`
  - Stop discovering and lowering a special structural pipeline inside
    `_emit_tile_dependency_stage_loops`.
  - Replace `readiness_frontier_body`, `consumed_on_ready_roots`, and
    producer-specific dispatch branches with common event-state,
    wait/publication helpers, one local-trigger emitter, and one static
    `WorkerSchedule` emitter. `FamilyDone` uses the same event lowering rather
    than a second synchronization hierarchy.
  - Keep logical/physical PID conversion and opaque root outlining.
  - Move event-specific lowering into a focused helper/module so ProgramID
    machinery supplies task bodies and PID maps rather than owning the graph
    scheduler.

- Configuration and tests
  - Delete `tile_dependency_frontier` after performance parity.
  - Continue tuning ordinary root codegen and total worker count normally.
  - Rewrite codegen tests to assert event relations, placements, and preserved
    opaque bodies rather than frontier indices or continuation names.

#### Migration sequence and deletion gates

1. Introduce `tile_dependency.py` and the full-DAG event/use representation,
   translating existing dependency facts without changing codegen.
2. Introduce `cross_loop_scheduler.py`, initially delegating to the retained
   lowering where necessary, and make `program_id.py` call its narrow entry
   points.
3. Represent whole-family completion as the canonical one-key `KeyedEvent` and
   permit exact and whole-family prerequisites to coexist at one use.
4. Build the symbolic `WorkerSchedule` from the existing persistent traversal,
   including a complete schedule position rather than only a worker mapping.
5. Implement generic readiness propagation and worker liveness over arbitrary
   chain lengths, fanout, joins, partial domains, and numbers of producer
   rounds. Initially keep access-local uses as waits inside their owning tasks.
6. Port local triggers and compare lowered Qwen code against the retained
   immediate probe.
7. Port exact statically scheduled execution and compare lowered Gemma code
   against the retained direct-keyed probe.
8. Add monotonic exact-to-`FamilyDone` fallback, validate residency and
   progress over finalized worker schedules, and only then derive worker-reuse
   intervals.
9. Generalize access-loop wait regions without making them movable, and
   validate attention
   partial/merge, merge/O, and O/post-normalization boundaries.
10. Re-run full-layer correctness, repeated graph replay, NCU, and interleaved
   tuned standalone Helion controls for Qwen and all Gemma variants.
11. Delete the old modules, `ReadinessFrontierPlan`,
    `_select_readiness_frontier`, the frontier config, compatibility re-exports,
    and old local-execution/continuation branches only after lowered code and
    performance parity.
12. Keep any computation replication or recomputation transform outside this
    scheduler. It needs its own purity and profitability justification.

Current status:

- Full-DAG event uses, recursive local-trigger ancestry, ordered worker
  placement, and monotonic root-completion fallback are implemented.
- The old frontier planner, its AST emitter, fixed-depth candidate search, old
  config field, compatibility modules, and single-consumer counted-event
  assumptions have been removed.
- Outer-coordinate readiness patterns are refined into common access stages,
  so batch size greater than one does not require identical absolute producer
  timelines.
- Remaining cleanup is to make the access-local event materialization less
  separately named, move more lowering mechanics out of `program_id.py`, and
  decide whether canonical `FamilyDone` events can directly replace the
  root-edge projection without obscuring the simple lowering.
- Final gates are broader shape tests, fresh Qwen/Gemma lowered-code and
  resource comparison, uncontended latency runs, and NCU confirmation.

#### Findings from reviewing the probes and current compiler

The review found concrete topology assumptions that must be deleted rather
than generalized with more cases:

1. `ReadinessFrontierPlan` contains exactly one counted intermediate event and
   exactly one downstream access cohort. `_select_readiness_frontier` requires
   one matching cohort, one downstream producer root, one stream axis, and one
   unique candidate family. This is the old FFN shape expressed generically in
   its names but not in its structure.
2. `tail_producer_tasks = producer_tasks - worker_count` and
   `downstream_worker_begin = worker_count - downstream_tasks` encode one
   initial round followed by one tail round. The probes happened to have two
   rounds; the scheduler must instead represent arbitrary ordered worker
   schedules and three or more rounds.
3. `KeyedEventPlan.key_root` and `on_ready_root` combine event identity,
   consumer identity, and local placement. They prevent one event from serving
   several uses cleanly and make local execution look like a dependency fact.
4. `AccessCohortPlan` admits only one coarsened relation per consumer root and
   identifies one producer stream axis. General reduction readiness may have
   several incoming events, axes, or access program points.
5. `readiness_frontier_body` in `program_id.py` emits a bespoke producer-first,
   optional-tail, downstream-worker sequence. Full-DAG scheduling should emit
   `WorkerSchedule` relations produced by liveness allocation, not recognize a
   producer/downstream pair during code generation.
6. `tile_dependency_frontier` exists because the current planner enumerates
   different values of `G` independently from worker count. The probes show
   that `G` is the number of complete event keys in the relevant dispatch
   rounds once `W` and task order are fixed; it should be derived.

The probe behavior also supplies general constraints:

- Qwen and Gemma require the same events but different placement. Therefore
  local/direct belongs to worker planning, not event construction.
- The dynamic completion-rank and FIFO probes were slower. Therefore a full
  DAG does not imply a runtime DAG scheduler; dense lowering remains static.
- The isolated Gemma FFN is slower than separate kernels while the full layer
  is faster. Therefore placement cannot be selected from an isolated edge's
  latency; the planner must preserve region-wide overlap and launch savings.
- Qwen's 64/32 and Gemma's 36/4 splits are consequences of dispatch rounds,
  not a universal two-region design. Tests must include producers spanning at
  least three rounds and consumers with more than two readiness regions.
- Attention partial/merge has several consumer tasks sharing one predecessor
  set, while QKV/cache joins have several producer roots. Both invalidate an
  event representation permanently owned by one producer/consumer pair.

Two questions remain deliberately conservative rather than heuristic:

- If a producer has several outgoing event relations that imply incompatible
  key-major orders, preserve its existing traversal. Do not choose an edge by
  model identity or guessed benefit.
- If several exact statically scheduled task families have overlapping
  liveness and do not all fit, keep individually eligible tasks local and use
  root completion for the rest. Do not introduce a priority rule until a real
  non-FFN workload demonstrates that a small per-event placement knob is
  needed.

#### Overfitting review

The design is acceptable only if all of the following remain true:

- It contains no model names, root numbers, projection/activation/GEMV tests,
  or fixed graph depth.
- It handles chains of length one through many, diamonds, fanout, joins,
  disconnected components, and mixtures of exact and whole-family events.
- It does not assume one intermediate root, one downstream root, one varying
  axis, a constant task count, a two-wave producer, or `P <= 2W`.
- Batch, sequence, head, expert, and other outer coordinates remain in event
  keys and worker-liveness domains.
- A producer may contribute to multiple keys, and multiple consumers may share
  a predecessor signature.
- Expected arrivals may vary by key semantically; unsupported nonuniform
  representations fall back rather than being forced into a uniform counter.
- Ordinary aliases, flatten/unflatten views, partial in-place updates, and
  multiple reaching definitions are handled before scheduling.
- L2 remapping changes only physical task conversion, never event identity.
- Multiple direct-action families are accepted only when their liveness
  intervals and worker sets are jointly valid; otherwise individually eligible
  actions stay local and other uses fall back to root completion.
- Dynamic or indirect relations, zero-task domains, oversized grids, and
  insufficient residency have an explicit root-completion path.
- Dense static tasks never pay a runtime queue or global completion-rank
  protocol merely to recover information already present in the DAG.
- Root computation bodies and their codegen configuration remain unchanged;
  only scheduling boundaries, waits, publications, and task dispatch may be
  added.

The Qwen and Gemma FFNs are therefore validation cases, not structural
templates. Attention partial/merge tests exercise shared predecessor sets;
merge/O exercises flatten normalization and reduction-region readiness;
O/RMS exercises many-to-one singleton action; generic unit tests must add
longer chains, diamonds, joins, multiple simultaneous ready families,
nonuniform fan-in fallback, batches larger than one, and producer domains
spanning three or more waves. If those tests require a new scheduler type, the
IR is still too specialized.

Topology sanity review:

| Generic graph | Expected representation and policy |
|---|---|
| One-to-one elementwise chain of arbitrary length | One key per logical tile; eligible tasks may execute locally and publish the next event. |
| Diamond fanout followed by a join | One publication may have several `EventUse` records; the join event has contributors from both branches. Repeated uses are direct or root-complete, not serialized implicitly on one producer. |
| Stencil or halo exchange | One producer may contribute to several neighboring keys; exact bounded predecessor sets use direct waits, otherwise root completion. |
| Multi-stage reduction | Consumer access regions are grouped by identical prerequisite events, with any number of dispatch rounds rather than two fixed stages. |
| Several disconnected pipelines | Finalized worker schedules reuse worker IDs after the relevant liveness intervals end. |
| Batch or outer Cartesian axes | The axes remain in event keys, preventing readiness from one batch/item satisfying another. |
| Runtime scatter, routing, or indirect alias | Root completion in the dense static backend; a future irregular backend may consume the same semantic relation with a queue. |
| Zero-size or dynamic task domain | Preserve the existing safe phase fallback without allocating invalid event state. |

This review found no need for an FFN- or attention-specific semantic object.
The unresolved work is algorithmic—compatible task ordering and conservative
worker-liveness allocation—not additional model patterns.

### Earlier keyed-event rationale and experimental evidence

The scheduler must represent readiness of a logical value region, not merely
completion of the most recent root that touched its allocation. Use three
separate layers:

```text
Allocation footprints and reaching definitions    correctness
    ↓
Exact logical task-dependency relations            legality
    ↓
Keyed events and static dispatch plans             execution policy
```

The central synchronization IR is a compiler-derived, restricted form of the
Event Tensor abstraction:

```text
KeyedEvent
    key domain: logical readiness coordinates
    updates:
        completion source
        proven participating task domain
        producer coordinates -> zero, one, or bounded event keys
    uses:
        consumer task, access, or cohort
        event keys -> consumer coordinates
    expected arrivals per key: derived from the update relations
```

`KeyedEvent` records dependency semantics, not a chosen execution strategy. A
separate static dispatch plan decides whether a use becomes a root wait, exact
event wait, coarsened milestone, last-arrival opaque-task invocation, or
readiness frontier. In particular, `on_ready` is a lowering choice rather than
a property of dependency legality.

This is one compiler mechanism, not separate attention and FFN lowerings:

- root completion is one event key receiving contributions from every active
  producer worker;
- an exact task event has one key per producer task and fan-in one;
- a uniform relation has a consumer key and a proved constant fan-in;
- a partial relation contributes only from a proved producer subdomain;
- a multi-producer join receives contributions from several roots; and
- a readiness frontier admits the cohort of tasks associated with ready keys.

The same generic event lowering performs notifications and waits. The static
dispatch lowering may elect the last arrival and either invoke one existing
opaque task or publish readiness to a consumer cohort. Region boundaries affect
only update-domain metadata and guards. They must not create new model-specific
emitters or rewrite root computation.

#### Why this refactor is necessary

The simpler Qwen source experiments exposed the abstraction gap directly:

- Exact RAW+WAW task readiness correctly proved that each Q/K head consumes 16
  contiguous eight-element QKV projection tiles. Lowering that relation as 768
  task publications and 40 consumers each polling 16 epochs measured 83.50
  microseconds versus 80.31 microseconds separately.
- A partial counted continuation eliminated those polling loops and correctly
  let the first 640 Q/K-producing tasks participate while the remaining 128 V
  tasks ran normally. With conservative downstream completion it measured
  82.42 microseconds versus 86.35 microseconds for the same-source separate
  graph.
- The partial in-place update revealed that a single `latest_writer` per
  allocation is insufficient: Q/K is supplied by the normalization root while
  V is still supplied by the projection root. Treating normalization as a
  whole-allocation definition can release cache or attention before V exists.
- Canonical out-of-place or head-shaped QKV representations made the dependency
  easier to express but measured roughly 80.2--81.0 microseconds because they
  changed materialization or projection codegen. Dependency scheduling should
  not require that compromise.

The partial-prefix proof and lowering remain useful as the first implementation
of a contributor domain. The temporary conservative pass-through root wait is
not the final abstraction; it must disappear once allocation reaching
definitions and multi-producer keyed joins are represented directly.

#### Event Tensor paper assessment

The paper's event abstraction is a strong match for this synchronization IR:
an event is a tensor-shaped family of counters, producer task coordinates map
to event coordinates, and event coordinates map independently to consumer
tasks. This is more general and cleaner than attaching an event permanently to
one producer/consumer root pair.

Adopt these ideas:

- Event keys have their own logical shape, including symbolic batch or sequence
  axes, rather than borrowing one root's flattened task ID.
- Producer-update and consumer-use maps are independent and may be many-to-one,
  one-to-many, or restricted to a participating task domain.
- An event may combine contributions from multiple roots, and its expected
  count is derived per key.
- The dependency representation is independent of whether the compiler later
  chooses static waits, a last-arrival continuation, or a different scheduler.
- The same abstraction can eventually represent data-dependent mappings, such
  as MoE routing, without changing the dense-layer event model.

Do not adopt these parts yet:

- Do not require users to annotate Event Tensors. The paper starts from an
  explicitly annotated graph; Helion must derive safe mappings from ordinary
  kernel accesses.
- Do not adopt the paper's centralized dynamic work queue for dense low-batch
  layers. Its own dense-Qwen results show substantial dynamic-scheduler
  overhead, consistent with our earlier global-cursor experiments. Keep the
  structured static persistent scheduler.
- Do not require decrement-to-zero counters or per-launch reinitialization.
  Helion's epoch-relative counters are already capture- and replay-friendly.
- Do not assume the event abstraction solves aliases, partial writes,
  residency, deadlock, memory ordering, or root ABI preservation. Those remain
  independent compiler obligations.

Thus `KeyedEvent` is best understood as an internal, compiler-derived Event
Tensor. A future irregular/MoE backend may lower the same IR to a dynamic queue,
but Qwen and Gemma should first use a static schedule generated from it.

#### Ready-queue re-evaluation

The ready-queue alternative was re-tested with current tuned root bodies rather
than inferred from the older prototypes. The result strengthens the Event
Tensor design above but rejects a FIFO as the default dense lowering.

- The old Qwen FIFO was rerun first. With its older four-warp, 197 KB shared
  memory envelope it measured 199.50 microseconds versus 185.00 for its matched
  three-kernel graph. The queued W2-slice variant measured 256.14 microseconds.
  These probes identified the important failure modes—per-tile cursor traffic,
  head-of-line waits, and changed W2 codegen—but are not performance controls
  for the current one-warp bodies.
- A new Qwen probe keeps the current W13 and W2 bodies and compares in one
  interleaved run against the tuned standalone Helion kernels. Three fresh
  GPU-0 processes, each with 50 samples of 20 graph replays, measured
  39.08--40.64 microseconds for standalone Helion and 37.96--39.38 for the
  local-on-ready policy, a 1.07--3.11% reduction in every run. A separate
  100-sample control measured 41.32 microseconds when all activation tasks were
  transported through FIFO slots versus 41.09 for standalone Helion.
  Dynamically ranking every worker added a larger penalty because it destroyed
  the stable dense-family assignment. The local-on-ready kernel uses 74
  registers with no spills.
- A new exact-root Gemma FFN probe compares the same policy against the tuned
  standalone Helion gate, GeGLU, and down kernels. The standalone graph measured
  34.96 microseconds and the immediate fused FFN 39.13. FIFO-offloading the four
  first-frontier tasks improved the fused FFN to 38.59, while direct keyed
  handoff improved it further to 37.91. NCU measured 67.52 microseconds for the
  instrumented immediate kernel and 66.02 for keyed handoff, with unchanged
  achieved occupancy and higher sustained memory throughput for the keyed form.
- In the complete Gemma layer on GPU 0, three fresh processes with 50 samples
  of 20 graph replays measured 80.11--80.16 microseconds for tuned standalone
  Helion, 75.56--75.68 for immediate activation, and 74.72--74.78 for direct
  keyed scheduling of all 40 activation tasks. The keyed policy therefore
  improves the best separate-kernel control by 6.68--6.72%. Earlier 8-, 16-,
  and 36-task handoffs measured in the same performance band, so this is not a
  sharp fitted count. After removing the rejected FIFO path from the full-layer
  probe, a final 50-by-20 run measured 74.74, 75.41, and 80.07 microseconds for
  keyed, immediate, and standalone execution respectively.
- The isolated Gemma FFN is a necessary control: tuned standalone Helion
  measured 34.98 microseconds, while direct keyed handoff measured 37.85 and
  immediate execution measured 39.08. Thus the full-layer speedup comes from
  cross-root overlap and launch elimination; it does not rely on claiming that
  the monolithic GEMV code is intrinsically faster.
- The same frozen Gemma lowering measured 74.79 microseconds on GPU 0 but 79.81
  on GPU 6, despite both devices reporting idle. All schedule comparisons must
  therefore remain same-process and interleaved; cross-GPU absolute numbers are
  not suitable for selecting a policy.
- The A4B MoE probes independently reach the same conclusion: once ragged work
  is tiled into approximately uniform tasks, a dynamic cursor loses 2--7% to a
  static mapping in five of six measured cells. A queue pays only when task
  duration remains genuinely runtime-irregular.

The resulting execution policy is one rule over the Event Tensor, not separate
model schedules:

1. A last producer makes an event key ready. Its default `on_ready` action is
   local execution of the consumer task.
2. For a one-wave frontier, derive the producer tail `T`, downstream dense
   demand `C`, and unused resident cohort `I = W - T - C` from the task graph
   and selected worker count `W`.
3. If `I` covers a complete ready task family, those otherwise-idle workers may
   execute that family by direct event key. Gemma has `W=576`, `T=64`, `C=160`,
   and `I=352`, so all 40 activation tasks fit. Qwen has `W=1024`, `T=512`,
   `C=512`, and `I=0`, so activation remains a local continuation.
4. Preserve arithmetic assignment and affinity for dense GEMV families. Do not
   rank all workers through a global cursor merely to rediscover an already
   balanced mapping.
5. Use FIFO transport only when a task's ready key or duration is genuinely
   runtime-irregular and static tiling cannot make task costs uniform.

This removes the need for an FFN-specific three-node scheduling concept. The
dependency graph supplies event keys and fan-in; the resident task demands
determine whether an `on_ready` action runs locally or on a proven-idle worker.
No cost model or new tuning knob is required for the Qwen/Gemma cases. If a
future graph has only a partial idle cohort, keep local execution as the
conservative default and expose a bounded per-event offload amount only after a
real workload demonstrates value.

All headline comparisons use an idle B200, the same process and input tensors,
captured CUDA graphs, rotating execution order, and the checked-in tuned Helion
configs. The Gemma config file is byte-identical to the original Gemma
exploration worktree. Every measured candidate is checked against the Helion or
Torch reference before timing and again after repeated graph replay. The probes
retain their candidate source or PTX, and the FFN-only controls also write each
standalone Helion kernel's lowered Triton.

#### Dependency construction

Replace the single allocation-level `latest_writer` with conservative reaching
definitions. A later partial or in-place write kills an earlier definition only
for regions that it is proven to cover. Consumer readiness is formed from all
reaching definitions needed by its accesses.

Keep this implementation bounded and concrete:

1. DeviceIR supplies logical task axes and allocation-coordinate access facts.
2. Represent each footprint dimension as an exact interval or a wildcard and
   retain both may-overlap and must-cover information. An unknown write adds a
   reaching definition but kills no older definition.
3. After a candidate kernel configuration instantiates tile sizes, enumerate
   finite task footprints—not tensor elements and not an unconditional
   producer/consumer Cartesian product—for supported affine accesses.
4. Derive exact predecessor sets and contributor ownership from those concrete
   logical task coordinates. Preserve known axes, such as KV head, even when a
   different address dimension is indirect.
5. Compress only proven regular relations into affine intervals, domains, and
   constant-fan-in events.
6. Bound compile-time enumeration with an explicit planning-work limit. If the
   limit is exceeded, task counts remain symbolic, or disjointness, coverage,
   and contribution counts are not proven, use root completion.

This avoids both a general symbolic-set solver and reconstruction from lowered
PID arithmetic. Enumeration is a configuration-time proof technique; the
lowered representation remains compact.

#### Qwen and FFN through the same IR

The intended Qwen event graph is:

```text
qk_input[batch, q_or_k_head], count 16
    <- QKV projection tiles 16*h ... 16*h+15
    -> Q/K norm+RoPE task[batch, h]

cache_input[batch, kv_group], count 17
    <- normalized K head[batch, group]
    <- 16 projected V tiles[batch, group]
    -> cache task[batch, group]

attention_ready[batch, kv_group], count 5
    <- four normalized Q heads[batch, group]
    <- cache task completion[batch, group]
    -> admit that group's split-attention cohort

chunk_ready[batch, chunk, q_head], count 8
    <- eight split-attention tasks
    -> chunk merge task

head_ready[batch, q_head], count 16
    <- sixteen chunk merge tasks
    -> final merge task and attention quantization readiness
```

For the current batch-one shape, the first map covers Q/K projection tasks 0
through 639 while V projection tasks 640 through 767 update `cache_input`
instead. The event mapping therefore preserves both reaching definitions
without pretending that Q/K normalization overwrote the V region. A static
schedule should traverse the backward closure of the first `G` attention keys
in key-major order, reproducing the useful Q/K/V grouping without a named
attention lowering.

The FFN is a smaller instance of the same representation:

```text
gate tiles[group] + up tiles[group]
    -> activation-ready[group]
    -> activation task[group]
    -> downstream readiness frontier
```

Gemma4 uses the same event IR, with a simpler attention prefix and two explicit
residual joins. For the non-shared-KV variant:

```text
input RMSNorm completion -> QKV projection tasks

projected_head[batch, head]
    <- all projection tiles belonging to that head
    -> fused Q/K/V norm + RoPE + cache task[batch, head]

attention_ready[batch, kv_group], count 6
    <- four normalized Q-head tasks
    <- one K-normalize/cache task
    <- one V-normalize/cache task
    -> attention split cohort for that group

attention split completion -> attention merge
attention merge completion -> output projection
output projection completion -> post-attention residual/pre-FF norm

gate/up projection contributions[activation group]
    -> GeGLU task[activation group]
    -> selected FFN readiness frontier
    -> down-projection cohort

down completion + saved post-attention residual
    -> post-FF residual/RMSNorm

post-FF output -> PLE gate -> PLE projection
PLE projection completion + saved post-FF output
    -> final PLE norm/residual/scale
```

For the shared-KV variant, the cache is an already-available external input, so
the attention prefix becomes:

```text
input RMSNorm completion -> Q projection tasks
projected_q_head[batch, q_head]
    <- all Q-projection tiles belonging to that head
    -> Q norm/RoPE task[batch, q_head]

attention_ready[batch, kv_group], count 4
    <- four normalized Q-head tasks
    -> attention split cohort using the external KV cache
```

With the retained projection tile size, a projected head receives 32
contributions for the 256-wide sliding-attention head and 64 contributions for
the 512-wide full-attention head. Non-shared attention readiness has fan-in six;
shared-KV readiness has fan-in four. Sliding attention admits 16 split tasks per
KV group and full attention admits 64. These counts are derived from task and
event-coordinate maps, not encoded as Gemma constants.

Sliding and full attention change only task extents, split counts, resource
usage, and therefore which legal frontier fits residency. They do not require
different event kinds or a different lowering. Existing measurements also show
that finer attention and output-projection continuations are not profitable for
Gemma; the same event graph may therefore select root-completion or coarse
cohort publication on those edges while retaining the FFN counted continuation.
This is a policy choice within the common event representation, not a separate
Gemma schedule.

Different models therefore instantiate different event graphs, but use the
same planner, event state, contribution lowering, last-arrival operation, and
worker dispatcher.

#### Policy boundary

The compiler derives task domains, predecessor membership, fan-in, and event
keys as legality facts. After selecting a legal readiness frontier, the static
scheduler derives its worker-to-task mapping mechanically from the event graph;
that mapping is policy, but not an independent tuning dimension. The only
intended scheduling autotuning surface remains the small set of legal readiness
frontiers (and therefore the derived launch grid). Do not add a static
model-sensitive cost model or independent producer/consumer worker-assignment
knobs.

#### Refactor and deletion order

1. Restore conservative root ordering for any partial WAW/WAR relation not yet
   represented by complete reaching-definition semantics. The experimental
   partial-prefix lowering remains test scaffolding, not the production default.
2. Make allocation history retain the reaching definitions of partially
   overwritten regions and derive each consumer key's complete predecessor
   set. Change transitive coverage checks from root-pair reachability to
   readiness facts, because a partial task relation does not order an entire
   producer root.
3. Introduce `KeyedEvent` with independent update and use mappings, then port
   root completion, exact task events, existing counted continuations, and
   access cohorts without changing their generated code.
4. Add multi-contributor joins and lower every contribution through one
   notification helper. Keep last-arrival opaque-task execution and downstream
   cohort publication in the separate static dispatch plan.
5. Express the Qwen QKV -> Q/K -> cache -> attention prefix using those events,
   preserving every existing root body and its configured PID decomposition.
   Derive a key-coalesced producer traversal from the selected readiness
   frontier so event semantics alone do not regress to physical-order polling.
6. Revalidate FFN continuation and all four Gemma variants through the same
   representation; their generated arithmetic must remain unchanged.
7. Delete the temporary partial-write pass-through workaround and
   `_match_partitioned_dependency_pipeline` only after correctness, lowered
   code, resource use, and performance match the retained fast path. This is
   complete: the generic Qwen lowering matches the retained custom schedule.
8. Keep one-wave reduction replication separate because it duplicates
   computation and is not a dependency protocol.

## Recommended architecture

```text
DeviceIR roots
    ↓
Memory accesses grouped by allocation
    ↓
Region-aware reaching definitions and task relations
    ↓
Keyed events with one or more contributor domains
    ↓
Generic persistent execution with waits/publications
```

### 1. Treat each top-level loop as an opaque task family

For every root, retain only:

- Root ID and source order.
- Grid axes and symbolic task counts.
- Existing PID decomposition.
- Reads and writes to underlying allocations.
- Existing root body, unchanged.

The scheduler should not identify a layer, an FFN, a reduction chain, or a
particular number of adjacent roots. A root is a family of tasks whose existing
body is dispatched with a set of task coordinates.

This boundary is important: the scheduler decides *when* an existing task or
part of a task may run. It does not decide how its arithmetic, loads, stores,
reductions, or loop nests are generated.

### 2. Build dependency edges from DeviceIR memory accesses

Create a small scheduler-specific access fact for every relevant load and
store. It should identify:

- The owning root and graph.
- The underlying allocation, so aliases and views are grouped correctly.
- Load versus store and a stable access ID.
- Tensor shape, strides, and storage offset.
- Subscript tensor dimensions.
- Affine block IDs, scales, constant offsets, and accessed extents where these
  can be proven.
- Whether an explicit mask or unsupported/indirect address expression prevents
  a fine-grained proof.

Use these facts to derive cross-root hazards in source order:

- RAW: a later root reads an allocation written by an earlier root.
- WAW: a later root writes an allocation written by an earlier root.
- WAR: a later root writes an allocation read by an earlier root.

RAW, WAW, and WAR edges may use fine-grained readiness only when every relevant
access relation is proved. Otherwise they use conservative root-completion
ordering. Unsupported side effects, atomics, or source-level conflicts that
cannot be reconciled with DeviceIR facts must remain errors rather than being
guessed away.

Dependency discovery is based on allocation identity, not source variable
names. The former source-level memory-dependency analysis has been deleted.
The ordinary host walker now contributes only explicit barrier boundaries, and
DeviceIR name lookup diagnoses unsupported cross-root value capture; neither
constructs a second dependency graph.

### 3. Prove a readiness mapping for each edge

For every producer-store/consumer-load pair on a RAW edge, classify the
strongest readiness relation that can be proved:

1. **Task-ready**: consumer coordinates map to an exact set or interval of
   producer task coordinates using compatible views and affine subscripts.
2. **Root-ready**: a memory dependency exists, but a safe task mapping cannot be
   proved. The consumer waits for the producer root to complete.
3. **Unsupported**: the effect cannot safely be represented by memory
   readiness and compilation must retain the existing legality error.

The proof must check the actual producer-store and consumer-load mapping. Equal
task counts, matching topology, or a familiar number of loads are not a proof.
For example, reversing the consumer's activation-group order must either produce
the corresponding reversed predecessor map or fall back to root completion.

The initial affine proof should be intentionally conservative:

- Require compatible allocation views: shape, strides, and storage offset.
- Require one unambiguous producer store for task-ready publication.
- Require understood affine dimensions, scales, offsets, and masks.
- Include every relevant coordinate in the event key. Batch is therefore a
  normal key dimension: `(batch, activation_group)` rather than a batch-1
  special case.
- Compute the producer tiles whose written regions overlap the consumer load's
  region. Unequal producer and consumer tile sizes may yield one or several
  predecessors.
- Fall back to root-ready for indirect gathers, ambiguous aliases, masks that
  change coverage, multiple unresolved stores, or other unproved cases.

A conservative fallback is part of the design, not another special case.

### 3a. Separate dependency semantics, legal schedules, and policy

The implementation should have three deliberately small layers:

1. **Dependency semantics** compute the exact predecessor relation
   `P(consumer_key)` from memory overlap. This layer answers only which producer
   tasks must finish before a consumer operation is safe.
2. **Schedule construction** recognizes a few generic properties of that
   relation, such as identity, a uniform partition, or contiguous nested-loop
   cohorts. It may construct an optimized schedule only after proving the
   required property.
3. **Policy** exposes only a small compiler-generated set of legal candidates
   to measurement. It must not predict performance with a cost model or expose
   arbitrary worker assignments.

The dependency graph is therefore authoritative. A continuation analysis must
not independently rediscover memory dependencies from root numbers, task
counts, or flattened PID patterns. It consumes `P(consumer_key)` and proves
only the additional property needed by its lowering.

### 3b. Fan-in is derived, not guessed or tuned

For a consumer key `c`, the raw fan-in is `|P(c)|`: the number of producer tasks
whose writes that consumer needs. If every producer task publishes one arrival,
the last-arrival counter target is exactly that cardinality. Choosing another
number would be incorrect.

An optimization may legally combine several producer tasks into one
publication. In that case the *arrival* fan-in changes, but it is still derived
from the proven grouping; the autotuner may choose the grouping, not the
counter target independently.

The last-arrival continuation is useful only for a particularly simple
relation. For the first generic implementation, require that the predecessor
sets:

- are nonempty;
- are pairwise disjoint;
- cover the producer task domain;
- have one constant cardinality; and
- can be enumerated by a compact coordinate mapping.

These conditions make the producer-to-consumer ownership function total and
single-valued. Each producer task increments exactly one consumer counter, and
the last producer for that key may execute the unchanged consumer task.

Disjointness and coverage are not already guaranteed by the dependency graph.
For example, a producer tile can feed two consumer tiles, or some producer tiles
can be irrelevant to this particular edge. Both are valid dependency graphs,
but neither fits the minimal one-counter continuation primitive. They should
use exact task events or root completion instead of making the continuation
primitive more complicated.

The current power-of-two fan-in restriction is an implementation artifact, not
a semantic requirement. Prefer an epoch-relative exact target comparison so
any statically proved positive fan-in is representable. Retain a power-of-two
restriction only if generated-code measurements demonstrate a material need.

### 3c. Separate readiness proofs from execution strategies

“Continuation” is not a third kind of dependency legality. Dependency analysis
proves one of two readiness levels:

```text
Dependency proof                 Possible execution strategy
────────────────────────────────────────────────────────────
Root-ready              →        root-completion counter
Task-ready P(c)         →        exact task events
Task-ready P(c)         →        proven event coarsening
Task-ready P(c)         →        last-arrival continuation
```

For each consumer task `c`, the dependency graph defines:

```text
P(c) = producer tasks whose writes are required by c
```

Root completion conservatively replaces `P(c)` with the entire producer root.
It is valid even when a finer mapping cannot be represented and usually has low
synchronization overhead, but exposes no producer/consumer overlap.

Exact task events implement `P(c)` directly. Each producer task publishes its
completion, and each consumer waits only for the predecessors it needs. This is
the faithful fine-grained lowering, but event storage, publication, polling,
and poorly placed waits can make it slower than root completion.

A continuation is an optimization of an already-proven task-ready relation.
When the predecessor sets are disjoint, have a constant fan-in, and admit a
compact mapping, each participating producer contributes one arrival to its
consumer key and the final arrival executes the unchanged consumer task
directly. The participating domain may cover the complete producer family or a
separately proved subdomain; unowned tasks execute normally and retain their
own reaching definitions. Continuation does not discover new dependencies or
make an otherwise illegal edge legal.

For the Gemma gate-up to GeGLU edge, there are 640 producer tasks and 40
consumer tasks, with 16 exact predecessors per consumer. The partition proof is
valid. An early compiler experiment made the edge legal by removing L2
remapping and measured about 153 us, while forcing a larger three-stage
pipeline measured about 263 us. The later exact-codegen probe separated those
effects: preserving the generated gate/up root and its L2 mapping, root
completion measured about 85.52 us and consumer-major continuation about
87.22 us. Thus logical task identity fixes legality, but continuation alone is
not profitable for this graph. The activation-to-down edge still requires the
entire activation domain unless the down reduction is admitted in proven K
segments. Therefore:

```text
legal continuation != profitable continuation
```

The dependency graph remains the single source of truth for legality. The
compiler should enumerate only proved alternatives among root completion,
exact events, coarsened events, and continuation; it should not use a cost
model to predict which legal alternative is fastest. Alignment, residency, and
forward progress are proof obligations. Event granularity, continuation, and
legal worker frontiers belong to a small autotuning surface attached to an edge
or interacting dependency component.

The controlled Qwen and Gemma experiments narrow the continuation-pipeline
surface further. Define:

- `Np`: total producer tasks;
- `Nc`: total downstream consumer tasks;
- `F`: proved continuation fan-in;
- `G`: a legal readiness frontier in continuation-consumer coordinates; and
- `W`: physical persistent workers.

The useful schedule family uses one complete initial producer wave:

```text
W = F * G
```

`G` must also place every split consumer access on an existing loop-tile
boundary. The compiler then derives the assignment rather than tuning it:

- all `W` workers execute one initial producer task;
- one worker is assigned to each downstream consumer task when `Nc < W`;
- the remaining producer tasks are assigned to the non-consumer cohort; and
- the candidate is admitted only when those remaining tasks can make progress
  within the proved resident grid.

Thus producer and consumer worker counts are not independent choices. The only
measured structural choice for this family is `G`, equivalently the exact
worker count `W`. Do not couple it to `num_sm_multiplier`: that global,
power-of-two multiplier is too coarse and would create a redundant tuning
cross-product. Compile each graph-derived `(G, W)` candidate directly and
discard it if its exact compiled resource usage cannot keep all `W` workers
resident. If no such frontier is legal, use exact events or root completion.

### 3d. Make logical task geometry authoritative

Logical task coordinates must be first-class DeviceIR metadata. Preserve each
root's axis extents, block sizes, coordinate order, and logical-to-physical PID
mapping before lowering. Do not reconstruct task counts or identities from
generated PID expressions.

This directly addresses two Gemma findings:

- A specialized `q_heads // kv_heads` expression originally remained a string
  in the lowered PID representation, causing static task-count recovery to
  fail and silently select the cooperative-grid fallback. Task families now
  preserve the specialized logical extent, and the benchmark again uses the
  natural quotient expression without a source workaround.
- Nontrivial L2 grouping previously disabled task events because readiness knew
  only flattened physical PIDs. The scheduler now retains logical axis order
  and converts between logical coordinates and each root's existing physical
  traversal only at dispatch.

The dependency proof should operate entirely in canonical logical coordinates.
The execution layer should then apply the root's existing PID mapping when it
dispatches that task. This keeps readiness identity stable across loop-order
and L2-remapping choices without making either transformation a special case.

### 3e. Normalize views before dependency proof

Ordinary aliases, slices, and reshapes should be normalized into coordinates of
their underlying allocation before overlap is analyzed. A projection written
as `[token, flat_feature]` and consumed as `[token, head, head_dim]` describes
the same allocation regions and should not fall back merely because the source
ranks differ.

The normalized form should retain allocation identity, storage offset, strides,
extents, and affine expressions. Fine-grained readiness remains unavailable for
indirect or ambiguous mappings, but simple view algebra should not hide an
otherwise exact predecessor relation.

The first implemented normalization removes inserted or removed size-one view
dimensions and records full slices explicitly. This already proves the Qwen
attention map-to-quantization edge across `(32, 128)` and `(1, 32, 128)` views.
Nontrivial flatten/unflatten relations remain conservative root-completion
edges until their allocation-coordinate intervals are represented directly.

### 4. Represent scheduling with generic events and waits

The plan should contain only generic scheduling concepts:

- An event identifies a logical key domain and one or more producer
  contribution domains.
- A wait identifies a consumer root/access and the readiness keys it requires.
- Multiple incoming edges become multiple waits.
- Fanout is multiple waits on the same publication.
- Chains arise naturally because a root can both wait and publish.

There should be no planner types for “grouped continuation”, “ordered input
singleton”, “one-wave reduction fanout”, or a fixed multi-root pipeline. Those
are emergent dependency graphs, not separate scheduling primitives.

Readiness keys are tuples over the coordinates required by the proven mapping.
They should not assume a single nontrivial producer axis. This makes rank,
batching, and Cartesian grids ordinary cases.

### 5. Use generic persistent execution without rewriting root bodies

Keep the current worktree's static persistent worker traversal and exact
co-residency validation as the execution foundation:

- Launch only a number of workers proven to be simultaneously resident.
- Let resident workers visit existing root task families in source order.
- Insert acquire waits only where a proven dependency requires them.
- Insert release publication only after the corresponding producer writes are
  complete and visible.
- Keep stream-local/capture-aware runtime state and monotonic launch epochs so
  stale readiness values cannot satisfy a later launch.

Full residency is the deadlock-safety invariant. A worker may wait only when all
workers needed to produce the awaited readiness can remain resident and make
progress. If this cannot be validated for a configuration, reject that
configuration or use a coarser safe schedule.

Do not copy the older experimental global atomic work cursor as the default
admission mechanism. It adds dispatch overhead and previously exposed Triton
dynamic-dispatch limitations. A cursor can be reconsidered only if a workload
demonstrates that static traversal cannot express a necessary schedule.

Outlined roots must remain opaque functions with a complete generated ABI.
Their live-ins include both source values and compiler-created values such as
tensor descriptors, dynamic strides, and scheduling state. Root extraction
must compute and thread this ABI mechanically. A scheduling transformation
must never leave a helper referring to a descriptor or other generated value
that exists only in the parent kernel.

Root completion should also support generic multi-predecessor joins. A
singleton consumer with several incoming root-ready edges simply waits for all
of their completion events and then executes its unchanged body. It should not
require output-tiling the consumer, duplicating a reduction, or recognizing a
residual-specific topology.

### 6. Place readiness at the narrowest proven program point

Root-level dependencies can wait at root admission and publish at root
completion. Fine-grained dependencies may require access-aware placement:

- Publish task readiness after all stores contributing to that task key.
- Wait immediately before the first dependent load when root admission has all
  required coordinates.
- If a consumer coordinate is introduced by an existing nested tile loop, wait
  at that loop iteration boundary before the dependent load, while leaving the
  loop body unchanged.

This is the one intentional interaction with existing root codegen: stable
access and loop IDs provide insertion points for waits/publications. The
scheduler must not reconstruct, reorder, or specialize the root's computation.

The older planner's access-ID concept is useful here, but its current behavior
of placing all waits at root entry is insufficient for Qwen's nested K-loop
consumption.

### 7. Coarsen events generically and only for performance

Exact readiness keys are the correctness model. The implementation may coarsen
them into contiguous slices or cohorts when this is proven equivalent:

- A coarse event is published only after every exact producer key in the cohort
  is ready.
- A consumer waits on every cohort intersecting its exact predecessor set.
- Coarsening must be derived from the affine mapping and worker traversal, not
  from root numbers or model dimensions.

For the Qwen FFN, this can recover the useful behavior of activation-group
cohorts without encoding an FFN matcher. Batch and shape changes simply add or
resize key dimensions.

Coarsening policy should be internal and preferably autotuned or selected from
simple occupancy/work-amortization facts. It should not become a collection of
public Qwen-oriented knobs.

### 7a. Keep the continuation primitive intentionally narrow

The scheduler does not need to reproduce every detail of the hand-written FFN
probe. The minimal useful continuation primitive is:

```text
exact predecessor relation P(c)
    ↓ prove a uniform partition
consumer-major producer traversal
    ↓ one arrival per producer task
last arrival executes the opaque consumer task
```

For Qwen, `c` is `(batch, activation_group)` and `P(c)` happens to contain the
gate and up producer tiles for that key. The scheduler should not know those
names or require two regions. It should consume the exact predecessor set and
compress its enumeration into a coordinate-space mapping.

The coordinate representation is needed only for cheap runtime dispatch. The
compiler can validate disjointness, coverage, and constant fan-in directly from
the exact compile-time predecessor sets for static shapes. It then accepts a
continuation only if those sets factor into a small uniform coordinate form.
This avoids both a general symbolic set solver and brittle pattern matching on
flattened task IDs.

Producer ordering is a performance policy, not another dependency proof. A
consumer-major order makes each proved key a contiguous logical task interval.
For the one-wave continuation pipeline this order is derived, not tuned: it is
the order that makes `W = F * G` complete exactly the first `G` keys while the
opaque producer body still receives its original physical task coordinates.
Outside this pipeline the compiler preserves the root's configured traversal.

Continuation should remain in the design, but not as a separate schedule kind.
Represent it as an optional completion action on a counted event:

```text
producer contributes to event(key)
    ↓ final required contribution
execute opaque consumer_task(key)
    ↓ publish any readiness produced by that task
```

The compiler attaches this action automatically when a selected readiness
frontier requires the intermediate consumer to materialize and publish keys
before the producer root completes. This is the Qwen and Gemma FFN case. Do not
apply continuation merely because a uniform partition is legal: Gemma's
standalone gate/up-to-activation continuation measured about 87.22 us versus
85.52 us for root completion because it unlocked no useful downstream work.
Conversely, disabling the continuation in the Qwen pipeline regressed the
whole layer from about 74.9 us to about 86.7 us. Its value belongs to the
connected dependency component.

There is therefore no independent continuation toggle in the initial tuning
surface. A frontier-pipeline candidate includes the proved last-arrival action;
an edge outside such a candidate uses ordinary counted events or root
completion. Fan-in, continuation ownership, producer traversal, and worker IDs
remain compiler-derived.

The execution policy is already performance-confirmed. The retained generic
Qwen lowering uses last-arrival execution and most recently measures about
74.66 us versus 85.14 us for separate kernels; disabling that continuation
path measured about 86.7 us in the earlier controlled ablation. The best Gemma
probe uses the same operation sequence--the final one of 16 gate/up producer
arrivals executes the activation task and then publishes its downstream
readiness--and measures about 72.2 us versus 79.8 us separately. The compiler-
generated Gemma lowering now measures 73.16 us versus 78.74 us in a paired
same-process run.
Gemma's 87.22 us standalone-continuation result is not contradictory: without
the downstream frontier, early activation execution unlocks no useful work.

Representation invariance is now validated in the Triton probes. Factoring the
Gemma arrival election into a reusable counted-event helper while leaving the
opaque `on_ready` body unchanged produced byte-identical final SASS, the same
198 registers, zero spills, and 34,816 bytes of shared memory. A paired
same-process run measured 73.061 us for the direct spelling and 73.143 us for
the counted-event spelling. The established Qwen probe already expresses the
same factored arrival followed by an inline `on_ready` body; it revalidated at
37.046 us for the persistent FFN core versus 38.948 us for its matched
three-kernel graph. The compiler refactor should therefore preserve this exact
lowering, with lowered-code equivalence and full-layer timing retained as
regression gates while the old plan hierarchy is removed.

### 7b. What may be autotuned

The autotuner may choose only among schedules whose legality is already proven.
For the initial continuation-pipeline implementation, expose one dependent
fragment: the legal readiness frontier `G`. Each value determines the exact
worker count, producer traversal, continuation action, tail-producer cohort,
consumer cohort, and two consumer access stages. These are not independent
knobs.

Other edges may later expose event coarsening when they have several proved
milestone sets, but that is separate from producer/consumer worker allocation.
Do not add a continuation toggle, native-versus-consumer-major toggle, or raw
worker-count knob preemptively.

The autotuner must not choose dependency membership, readiness-key dimensions,
fan-in independently of publication grouping, fence placement, or whether an
unproved mapping is considered safe.

The fragment should store a candidate index rather than a model-shaped group
count in the public configuration. Its values are regenerated from the current
task graph and tile sizes. No user-visible Qwen/Gemma schedule parameters are
introduced.

Root completion is represented by the stable internal candidate value `-1`.
It is the safe default and remains available to tuning beside any fine-grained
frontiers. A nonnegative index selects one of the graph-derived frontiers; an
out-of-range index is rejected rather than aliased to another schedule.

### 8. Keep the public interface small

The durable public contract should be an opt-in cross-loop/tile-dependency
schedule, plus explicit barriers where the user requires phase ordering.

Epoch replication, producer order, stage count, and continuation split are no
longer public policy parameters. Worker order and synchronization layout are
compiler-derived from the proved graph and selected readiness frontier.

Singleton placement is likewise compiler-derived. Ordinary root traversal
assigns work to low worker IDs first, so a singleton is assigned outside the
active ranges of its direct predecessors when such a resident worker exists.
Stable singleton order spreads several such roots without introducing a knob;
when every worker participated, high worker IDs are preferred because they do
not receive a partial-wave tail task.

Dynamic shapes must not reach internal assertions. If a task count is symbolic,
the planner should either emit symbolic expressions or select a documented safe
fallback.

## Correctness invariants

Every implementation stage must preserve these invariants:

1. A consumer can pass a wait only after every producer store that may overlap
   its dependent load is globally visible.
2. No task-ready edge is emitted unless the producer-to-consumer mapping is
   proved from DeviceIR access facts.
3. Unknown mapping means root-ready, never an inferred identity mapping.
4. Aliases and views are compared by allocation identity and compatible layout.
5. All relevant grid coordinates participate in task identity; singleton
   dimensions may simplify away, but non-singleton batch/rank dimensions may
   not.
6. Multiple producers, consumers, stores, and loads are represented as graph
   edges and event updates/uses, not topology-specific cases.
7. Every declared update executes exactly once per key and launch epoch, and
   the derived expected count equals the number of guaranteed contributions.
8. A last-arrival action is wait-free: every dependency of the invoked task is
   included in the triggering event. Event/action dependencies are acyclic, and
   every logical task has exactly one executor.
9. Full residency is necessary but not sufficient for progress. The selected
   static queues or structured traversal must retain a nonwaiting path to every
   outstanding contribution; validate this with a bounded configuration-time
   progress simulation where formulas alone are insufficient.
10. Producer writes complete CTA-wide before release publication; the final
    arrival acquires prior contributions before executing a consumer, and any
    downstream publication follows the same barrier/release discipline.
11. Launch epochs and stream-local state prevent cross-launch and cross-stream
    readiness reuse.
12. Disabling the schedule leaves ordinary Helion codegen unchanged.
13. Enabling the schedule changes root bodies only by inserting the required
    synchronization instrumentation at stable boundaries.

## What to reuse

### From the current worktree

- The validated Qwen benchmark and known approximately 75 microsecond result.
- Static persistent worker traversal.
- Exact resident-program capacity validation.
- Stream-local and capture-aware persistent state.
- Launch epoch handling.
- The generic whole-root fallback, after revalidating its mapping and progress
  assumptions.
- Existing root body and PID generation.

### From `/home/eche/local/helion-megakernel`

Selectively port concepts from
`helion/_compiler/cross_loop_dependencies.py`:

- Ownership mapping from nested DeviceIR graphs to top-level roots.
- Allocation-based grouping of memory facts.
- Stable DeviceIR loop identity and source order.
- Conservative allocation-coordinate normalization and overlap checking.
- Conservative task-ready versus root-ready classification.
- Tests for unequal tiles, Cartesian grids, joins, fanout, indirect-index
  fallback, and CUDA graph replay.

Improve these pieces while porting:

- Use a dedicated scheduler access fact rather than continuing to expand the
  broad autotuner `MemoryOpFact` interface if that keeps the boundary cleaner.
- Extend affine mapping to coordinates introduced by nested loops.
- Place waits at the dependent access/loop boundary rather than always at root
  entry.
- Use current runtime state and residency validation rather than the older
  workspace/reset and global-cursor implementation.

### Do not carry forward

- Source-AST/topology recognizers as correctness gates.
- Fixed adjacent-root patterns.
- Assumptions that one nontrivial axis equals the total task count.
- Inferred group identity from counts alone.
- The older global atomic cursor as the default dispatcher.
- Unrelated global codegen changes justified only by the megakernel experiment.

## Why use this worktree instead of starting from scratch

Use `/home/eche/local/helion-tile-dependency-schedule` as the integration base.
It contains the working performance result and the stronger runtime/residency
pieces. The older worktree is an ancestor with a large uncommitted experimental
stack; using it as the base would require re-solving already validated runtime
and performance work.

“Start from scratch” should apply to the planner abstraction, not the branch:
introduce the small edge-based planner alongside the current implementation,
make it authoritative for legality, migrate codegen to its generic plan, and
then delete the specialized machinery.

## Post-Qwen/Gemma refactor decision

The Qwen and Gemma experiments separate three concerns that the current
implementation still partially mixes:

1. dependency legality: which producer tasks must complete before a consumer
   task or access may run;
2. synchronization representation: root counters, exact task epochs, or a
   counted event indexed by a consumer key; and
3. execution policy: ordinary worker traversal, last-arrival continuation, or
   overlap between producer and consumer worker cohorts.

The compiler should model these independently. A legal continuation is not
automatically profitable, and a profitable root tile configuration is not a
new scheduling strategy.

The intended end state is:

```text
Opaque root + logical task space
              |
              v
Normalized allocation-coordinate accesses
              |
              v
Exact producer-to-consumer task relations
              |
              v
Generic readiness events and legal execution choices
              |
              v
Small deterministic global policy
              |
              v
Generic persistent lowering around unchanged root calls
```

### First-class logical task families

Every top-level root will have one authoritative task-family description:

- source root and body graph;
- logical axes, extents, origins, steps, and block-size references;
- symbolic and instantiated task counts;
- logical-to-physical PID traversal, including L2 remapping;
- normalized allocation accesses and stable program points; and
- the complete outlined-root ABI, including compiler-created descriptors.

Dependency proofs use logical coordinates only. Physical PID traversal is an
execution choice and must not change event identity.

### One dependency representation

The allocation-derived `CrossLoopDependencyEdge` is the only dependency edge
stored in DeviceIR. The earlier source-AST dependency and access records are no
longer copied into DeviceIR or consulted by lowering. Each graph edge carries:

- producer and consumer task families;
- RAW, WAR, or WAW kind;
- normalized allocation-coordinate regions;
- a conservative consumer-key-to-producer-task relation when provable; and
- root completion otherwise.

The supported exact relation is intentionally small: a union of affine
Cartesian producer-coordinate intervals. This represents unequal tile sizes,
outer batch axes, and the two disjoint gate/up regions without requiring a
general symbolic-set solver. Bounded configuration-time enumeration validates
the concrete task relation; successful relations are compressed into this
small production representation before lowering.

### Counted readiness as the common synchronization primitive

Root completion, exact task events, joins, and readiness cohorts are all
instances of a counted event:

- an index/key space;
- a mapping from completed work to event keys;
- an exact expected contribution count;
- publication program points; and
- consumer wait program points.

Root completion has one key and one contribution per active producer worker.
An exact task event uses the producer task as its key and count one. A join uses
the consumer logical key and derives its fan-in from the predecessor relation.
A cohort event adds a monotone milestone to that key.

Continuation is an execution choice over a keyed event: the producer that
makes the final contribution may execute the now-ready opaque consumer task.
Disjointness, exact contributor-domain coverage, and fan-in are proof
obligations, never tuning knobs.
The continuation action may itself publish another counted event after the
consumer store, allowing a chain such as producer projection -> activation ->
ordered down-projection to remain one graph-derived component.

Do not materialize a separate continuation plan hierarchy. The dependency IR
contains only keyed events and their update/use relations. The static dispatch
plan may attach a last-arrival opaque-task action and refers to the same events
for its frontiers and milestones. This keeps root completion, exact events,
joins, and continuation on one mechanism without mixing legality and policy.

### Local proof, small measured policy

Dependency legality remains pairwise over producer and consumer task families.
A thin global planner is still required for joins, worker ownership, progress,
and the shared resource envelope. It should prove and enumerate candidates,
not rank them with a static performance model.

For the one-wave continuation family above, the compiler can derive a compact
candidate interval. Requiring one consumer wave and at most one tail-producer
wave gives the lower bound

```text
W >= ceil((Np + Nc) / 2)
```

and candidate-specific compiled residency gives the upper bound. Only values
representable as complete continuation keys and complete downstream reduction
tiles enter the autotuner. For every accepted `W`, the compiler fixes
`G = W / F`, assigns all `Nc` consumer tasks one worker each, and assigns the
`Np - W` tail tasks to the low worker IDs. There is no worker-map search and no
`num_sm_multiplier` dependency.

This produces six legal Gemma candidates rather than an arbitrary assignment
space: 416, 448, 480, 512, 544, and 576 workers. For Qwen, the same proof makes
1024 workers the first legal candidate because 1536 producer tasks and 512
consumer tasks must fit in two overlapping waves. The autotuner measures the
small frontier set because the best endpoint depends on the generated kernel:
Qwen selects the first legal frontier, while Gemma selects the largest legal
frontier below its four-CTA-per-SM resident budget.

The launch grid is kernel-wide. Initially, admit one frontier-driving
dependency component, or several components only when they derive the same
`W`. Other components use ordinary counted-event traversal on that grid. If
future workloads require incompatible profitable frontiers, collect evidence
before introducing a joint component search; do not begin with a Cartesian
product of per-edge worker counts.

### Keep codegen tuning outside the scheduler

The scheduler invokes each outlined root unchanged except for waits,
publications, and task dispatch. Fusion-aware root configuration is a separate
autotuning concern. Gemma's gain came from ordinary choices such as QKV N=8,
output K=512, range stages, and the gate-weight `evict_first` annotation. The
same schedule regressed from roughly 72.25 to 84.4 microseconds when that load
annotation was omitted.

The fused autotuner may jointly select existing root tile/range/indexing
choices using the compiled global resource envelope. It must not encode those
choices as model-specific scheduler logic.

### Simplification guardrails

The final planner and lowering must not depend on:

- model, operator, or tensor names;
- adjacency of roots in source order;
- recognizing a particular number of roots;
- scanning lowered Triton AST for characteristic loads or reductions;
- reconstructing logical tasks from flattened PID expressions; or
- public knobs for fan-in, producer order, event replication, or continuation
  splits.

Any failed proof selects exact task events, root completion, or the existing
cooperative barrier fallback. The cross-loop scheduler must not duplicate,
split, reassociate, or otherwise rewrite opaque computation bodies.

### Refactor sequence and deletion gates

1. Introduce `TaskFamily` geometry and explicit program points in DeviceIR
   without changing emitted code.
2. Build the unified dependency graph from normalized allocation accesses and
   retain the current graph as a checked compatibility view temporarily.
3. Add a small schedule IR for counted events, waits, publications, optional
   last-arrival opaque task calls, and exact logical-worker dispatch; move pure
   planning out of `program_id.py`.
4. Migrate root completion, exact task events, and uniform last-arrival
   continuation onto that single counted-event representation.
5. Add generic multi-producer joins and monotone nested-loop milestones. Use
   these to derive legal frontier candidates and express the Qwen/Gemma FFN
   stream without a separate pipeline matcher.
6. Add an internal dependent autotuning fragment that selects a legal frontier
   candidate. Derive the exact launch grid and both worker cohorts from that
   candidate, then reject it after compilation if full residency is not
   satisfied.
7. Express the useful attention/reduction composition from these primitives.
   If a computation-replication optimization remains desirable, place it in a
   separate transformation pass rather than the dependency scheduler.
8. Delete `_match_ordered_input_singletons`,
   `_match_one_wave_reduction_fanouts`, and
   `_match_partitioned_dependency_pipeline` only after equivalent generic
   plans pass correctness, codegen-invariance, and performance gates.
9. Remove the migration-only `producer_order` and `epoch_replicas` API knobs.
   This is complete: the compiler owns both choices and the default generated
   synchronization sequence is unchanged.

The legacy top-level source dependency analysis has been deleted. DeviceIR's
allocation graph now performs both unscheduled-hazard rejection and scheduled
stage construction. Explicit `hl.barrier()` calls are recorded directly by the
ordinary host-AST walker and partition the allocation history into independent
epochs. A device value accidentally captured by a later root is rejected at
the exact DeviceIR name lookup that previously asserted; it does not require a
second memory-dependency analysis.

The resolved schedule stores only phase boundaries and user policy. Its
duplicate edge/access objects have been removed from DeviceIR and from the
schedule consumed by code generation. Diagnostics and probes inspect
`CrossLoopDependencyPlan` directly. Atomic read-modify-write operations also
enter the allocation graph conservatively as writes, so removing the source
scanner does not create a synchronization hole.

## Implementation plan

Status values: **pending**, **in progress**, **complete**, or **blocked**.

| Phase | Status | Work | Exit criterion |
| --- | --- | --- | --- |
| 0 | complete | Record the architecture and migration plan. | This living document exists and is kept current. |
| 1 | complete | Add scheduler-specific DeviceIR access facts and an allocation-based dependency graph without changing emitted kernels. | Unit tests inspect correct RAW/WAR/WAW edges, multidimensional keys, and conservative fallback. |
| 2 | complete | Make the edge proof gate all existing fine-grained schedules. | The reversed-group adversarial kernel is correct by using a proven reversed map or safe root fallback; Qwen still takes a valid fast path. |
| 3 | complete | Add generic root-ready and task-ready events to the current static persistent traversal. | Simple chains, fanout, joins, unequal tiles, and Cartesian grids use one planner/codegen path. |
| 4 | complete | Support readiness coordinates introduced by existing nested loops and access-aware wait placement. | Nested-loop dependencies are correct without changing computation codegen. |
| 5 | complete | Add two narrow optimizations over exact readiness: nested-loop cohorts and uniform-partition last-arrival continuations. | Qwen's FFN chain is represented without a topology matcher; failures fall back to exact events or root completion. |
| 6 | complete | Replace the one-producer event model and partitioned-attention matcher with region-aware, multi-contributor keyed events. | Qwen QKV/QK/cache/attention and both FFN variants use the same contributor/event lowering; the custom matcher and temporary partial-write pass-through are deleted. |
| 7 | complete | Audit and separate unrelated codegen changes; handle dynamic and large task domains without assertions or arbitrary planning cutoffs. | Schedule-off codegen is unchanged, dynamic shapes use the cooperative phase fallback, and static region matching scales with task domains plus actual overlaps rather than their Cartesian product. |
| 8 | in progress | Final correctness, performance, lint, and design review, including a second-model Gemma4 probe. | Test matrix passes; Qwen meets the agreed performance range; remaining limitations are explicit and structural. |
| 9 | in progress | Finish the full-DAG `CrossLoopSchedule` migration with canonical `FamilyDone` events, `WorkerSchedule`, and `LocalTrigger`. | Qwen and Gemma reach lowered-code and performance parity through one graph/scheduling path, adversarial DAG tests pass, and the old topology records and tuning field are deleted. |
| 10 | in progress | Promote dependency-bearing nested loops to ordered action domains in the same DAG and delete the access-cohort side path. | Natural producer-side and consumer-side streaming lower through generic events; Qwen 64/32 and Gemma 36/4 segments are graph-derived; no cohort-specific planner or emitter remains. |
| 11 | in progress | Preserve logical-coordinate relations through dependency analysis, scheduling, and lowering, and permit one root family to occupy several graph-derived worker-stream segments. | Batch-scaled Qwen and affine Gemma lower through compact logical relations and segmented worker streams; emitted worker order matches the validated schedule; fanout, joins, nonuniform tails, and L2 remapping remain exact; no dependency, event, fan-in, or dispatch mapping is reconstructed from a flattened task table. |

### Current migration boundary

The dependency graph now owns ordinary root-entry readiness, exact task
readiness, generic multi-producer joins, repeated-predecessor ready groups,
and last-arrival continuations. Nested-loop consumer readiness is still routed
through `AccessCohortPlan`; this is the remaining parallel representation and
must be absorbed into the ordered action-domain DAG. The old
flattened-ID grouped-continuation, ordered-input-singleton, and partitioned
attention matchers and their emitters have been deleted. The Qwen QKV/QK/cache
join, attention admission, two merge levels, quantization handoff, and FFN all
use the same keyed-event and root-dispatch machinery.

The compiler-side one-wave reduction-fanout matcher and emitter have been
deleted. Dependency scheduling never replicates an opaque root. The retained
Gemma benchmark source explicitly output-tiles and recomputes its short RMS
reductions; that is a kernel-author computation choice and is intentionally
outside the scheduler.

Root completion is no longer stored as a parallel schedule topology. It is a
canonical one-key `FamilyDone` event in `CrossLoopSchedule`; the generated
kernel retains the efficient physical implementation of one publication per
finishing worker rather than one atomic per logical task.
Region-aware reaching definitions now split exact partial writes, retain
unknown earlier definitions conservatively, and derive complete predecessor
sets for multi-root joins. Ordinary size-one views, storage offsets, and the
task-aligned Qwen layouts normalize into allocation coordinates. More general
flatten/unflatten support remains deliberately deferred until a second real
kernel requires it.

A source-only Qwen experiment narrowed that requirement. The attention split
can write directly to `(split, query_head, dimension)` storage and both merge
levels can index their multidimensional tensors directly, with no performance
loss and simpler lowered address arithmetic. This removes ordinary alias/view
normalization from that path, so a general reshape solver is not a prerequisite
for replacing the compatibility emitter. It does not, however, prove the
partitioned reduction dependency by itself: the producer query-head coordinate
is `kv_head * q_per_kv + query_in_group`, and each first-level merge task reads
the bounded split interval `chunk * splits_per_chunk + [0, splits_per_chunk)`.
Those are intrinsic logical task relations, not artifacts of a tensor view.

The implementation should therefore prefer canonical source layouts where
they are natural and express predecessor intervals with ordinary tile ranges.
Add affine logical-coordinate facts only for a real relation that cannot be
written that way. Defer a general allocation-region or in-place relation solver
until another real kernel requires it. Indirect cache relations fall back to
root completion unless source-level logical task identity proves the required
join; Qwen's task-aligned cache source now provides that proof.

An even simpler source form expresses both attention merge reductions as
ordinary `hl.tile` ranges: eight split tasks per chunk and all sixteen chunks
for the final merge. With the natural four-dimensional partial tensors, the
existing DeviceIR proof then classifies split-to-chunk-merge and
chunk-merge-to-final-merge as exact task-ready edges without any new alias
analysis. This is the preferred source shape for the generic replacement.
The query-head merge block is an ordinary kernel tiling choice. A block of four
makes the predecessor sets a disjoint partition. A block of one makes four
consumer tasks share each predecessor set; the keyed-event planner represents
that as one ready-group counter and four independent consumers, rather than
executing four consumers serially on the last producer. That distinction closes
the performance gap: the one-head source with generic ready groups matches the
deleted custom schedule. Let the existing block-size tuner choose task
granularity; do not add a scheduler-specific fanout knob.

### Remaining execution plan

The module split, full event DAG, symbolic `WorkerSchedule`, `LocalTrigger`,
canonical `FamilyDone`, and worker-count tuning migration are complete. The
former frontier/two-wave records, topology matchers, compatibility modules,
and compiler-side reduction replication have been deleted.

Remaining, in order:

1. Refactor `tile_dependency.py` around DeviceIR callsite-scoped ordered action
   domains and one normalized allocation-overlap relation, as specified above.
   Do not add another compatibility plan beside `AccessCohortPlan`.
2. Differentially prove that the new relation preserves batch/outer axes,
   unequal tiles and tails, offsets and ordinary views, multi-producer joins,
   and logical identity under L2 traversal before deleting the old affine
   path.
3. Generalize the existing keyed-event graph and worker-schedule validation to
   action endpoints with inherited same-worker order. Preserve root-task local
   and static placement semantics.
4. Move schedule-derived event quotienting and maximal contiguous loop-segment
   formation into ordinary DAG readiness propagation. Support piecewise event
   arrival counts, then reproduce the current Qwen and Gemma FFN lowerings.
5. Add producer-side nested publication and validate a natural,
   non-recomputed RMSNorm-to-matmul source.
6. Delete the complete access-cohort path, late access-marker discovery, and
   duplicate affine/readiness representations.
7. Finish adversarial coverage for mixed exact/whole-family paths, multiple
   event uses, nested producers and consumers, batches and other outer axes,
   partial in-place regions, and worker-schedule progress.
8. Re-run full-layer correctness, repeated graph replay, lowered-code diffs,
   NCU, and interleaved tuned standalone Helion controls for Qwen and all Gemma
   variants on uncontended GPUs.
9. Remove stale names and dead migration helpers only when generated code is
   unchanged; keep outlined Triton helpers because they are proven register-
   lifetime boundaries inside the one launched kernel.

Do not add a strategy cost model or another scheduler-specific tuning index.
The dependency graph derives events, fan-in, legal order, and progress facts;
the scheduler derives local triggers and worker schedules. The only retained
dependency-related performance input is the graph-aligned resident worker
count `W`, measured through the ordinary autotuner configuration.

Continue broadening relation normalization only when a real workload needs it.
Dynamic task domains already use the cooperative phase fallback. Large static
domains use interval sweeps rather than hardcoded task-product limits, and
unknown address relations select `FamilyDone`.

### Current performance evidence

- The current generic graph-derived Qwen schedule is 77.79 microseconds versus
  84.90 microseconds for separate kernels, with 253 registers, zero spills, and
  17,408 bytes of shared memory. The approximately 0.9-microsecond gap from the
  older 76.87-microsecond lowering is not a reason to retain a specialized
  schedule.
- The current tuned Gemma sliding non-shared schedule is 74.31 microseconds
  versus 78.87 microseconds separately, with 203 registers, zero spills, and
  34,816 bytes of shared memory.
- NCU reports 141.44 microseconds for the final Qwen kernel, compared with
  150.02 microseconds before ready-group coalescing and 139.14 microseconds for
  the deleted custom path. The final run transfers 233.4 MB, reaches 21.5% of
  peak DRAM throughput, and spends 43.9% of active-warp issue cycles on long
  scoreboard stalls, closely matching the former custom path.
- Exact access-local task polling for the generic path was approximately 262
  microseconds and caused heavy register spilling.
- Hoisting exact polls into a preflight traversal improved this only to roughly
  174--180 microseconds and retained the spills.
- Replacing downstream exact polling with two readiness cohorts improved the
  generic path to roughly 144 microseconds, but the producer-to-activation edge
  still used expensive exact polling.
- Outlining the scheduled task reduced spills only slightly and did not recover
  the performance gap.
- The uniform-partition last-arrival continuation plus the derived one-wave
  producer/consumer split recovered the full performance target without an
  autotuner knob or a Qwen-specific matcher.
- The compiler-generated Gemma schedule is approximately 73.16 microseconds
  versus approximately 78.74 microseconds for separate kernels in a paired
  same-process run, with 206 registers, zero spills, and 35,200 bytes of shared
  memory. Moving singleton work away from worker zero supplied the latest gain.
- Gemma's fused root configuration remains ordinary benchmark-side codegen
  tuning, not scheduler special-casing. The retained Triton probe remains about
  2.5 microseconds faster and is the comparison target for lowering cleanup.

These results validate the narrow generic continuation primitive. They do not
justify reproducing every specialized ordering or staging detail from the
probe.

### Gemma4 generalization probe

The first full-layer probe outside Qwen stacked the existing Gemma4 E4B Helion
roots before the graph-derived FFN frontier and fused-root retuning landed.
Those historical B200 results were:

| Variant | Megakernel | Separate Helion | Delta |
| --- | ---: | ---: | ---: |
| sliding, non-shared | 105.42 us | 79.73 us | +25.69 us |
| full, non-shared | 155.58 us | 131.57 us | +24.00 us |
| sliding, shared | 100.19 us | 78.07 us | +22.12 us |
| full, shared | 144.46 us | 129.22 us | +15.25 us |

The measurements are interleaved CUDA Graph medians from the same process.
They should be compared separately from the older Gemma README numbers, whose
full-attention separate-kernel results were materially faster on the earlier
software state.

The initial probe exposed the following general boundaries; later entries note
which have since been resolved:

- The first stacked kernel lowered most Gemma roots through whole-value
  completion. The current compiler now derives the FFN continuation/frontier
  component generically; unrelated edges still use root completion when finer
  readiness has no proved or performant representation.
- Passing `q_heads // kv_heads` through the composed source initially left the
  shared-attention task extent symbolic during scheduler construction. Task
  family metadata now consumes the specialized quotient directly, so the
  natural source expression and L2-remapped dispatch select the same schedule.
- The initial outlined-root implementation failed to thread tensor descriptors
  into generated helpers. Outlined ABI discovery now includes compiler-created
  preamble values, and both a focused descriptor test and a Gemma diagnostic
  configuration compile correctly. The retained benchmark still uses pointer
  indexing for direct comparison with the tuned probe.
- Nontrivial L2 PID remaps initially forced root completion. Logical event keys
  are now canonical and mapped to each root's physical L2 traversal only at
  dispatch, so Gemma takes the same proved FFN continuation/frontier path as
  Qwen. The earlier 153 and 263 microsecond failures remain useful evidence
  that legality alone does not choose a profitable frontier.
- The full-attention root demonstrates the resource cost of a single kernel
  configuration. A 512-by-128 attention tile exceeds B200 shared memory even
  at two stages. A 32-token context tile with four warps and three stages is
  legal and fastest among the tested simple choices, but the complete kernel
  still reaches 254--255 registers. This is a configuration/resource issue,
  not evidence for a model-specific scheduler case.
- The two residual RMSNorm joins are singleton consumers with two producer
  roots. Generic multi-input root-completion joins are now supported. The probe
  still output-tiles these roots at width 1024; reverting that source adaptation
  is a performance comparison, no longer a missing legality primitive.

The remaining general improvements suggested by Gemma are therefore small and
orthogonal: normalize affine accesses across simple views/reshapes and decide
whether to retain the output-tiled residual roots as a performance choice. The
natural singleton forms now compile and run correctly at approximately 75.17
microseconds with the same FFN frontier, versus approximately 73.16
microseconds for the retained tiled forms. They are no longer legality
workarounds. Neither issue requires a Gemma-specific continuation or a new
public tuning knob.

The later exact-codegen probe recovered a performant lowering without changing
that architecture. With the actual Helion-generated root bodies, the retained
576-worker schedule measures approximately 72.2 microseconds versus
79.8--80.0 microseconds for separate Helion. The winning changes are ordinary
root configuration choices made for the fused envelope: QKV N=8/K=256 at four
stages, output projection N=16/K=512 at three stages, gate/up N=32/K=256,
activation width 256, streamed down K=512 at three stages, and PLE-gate K=256.
The standalone gate-weight `evict_first` annotation must also be retained:
omitting it measured 84.4 microseconds versus 72.25 microseconds with it in a
matched post-copy run. This is an existing load-codegen choice, not a new
scheduling strategy.

Attention retiling did not expose a missing scheduler primitive. Q blocks 1
and 2 were slower than the existing Q=4 shape, a smaller context tile was
approximately neutral, and more attention splits were slower. Attention and
output-projection continuations also regressed. Relaxed polling and alternate
worker placement did not help.

The controlled worker/frontier ablation replaces an earlier incomparable
592-worker result that also changed activation and down-projection tiling. With
all root codegen held fixed, the aligned one-wave Gemma candidates measured:

| Workers | Ready groups | Megakernel | Separate Helion |
| ---: | ---: | ---: | ---: |
| 416 | 26 | 81.07 us | 80.06 us |
| 448 | 28 | 82.57 us | 80.16 us |
| 480 | 30 | 81.20 us | 80.00 us |
| 512 | 32 | 80.82 us | 80.16 us |
| 544 | 34 | 80.25 us | 79.97 us |
| 576 | 36 | **75.77 us** | 80.03 us |
| 608 | 38 | 83.48 us | 80.05 us |

The two independent-cohort ablations were much worse. Holding the 36-group
frontier fixed but launching only 560 workers measured 91.00 us because 16
producer tasks entered a second pre-readiness wave. Holding 576 workers and
the 36-group frontier fixed but reducing the consumer cohort from 160 to 144
measured 88.53 us because 16 output tiles entered a second consumer wave; 80
consumer workers measured about 87.48 us. These results support one structural
frontier knob, not separate producer/consumer allocation knobs.

Trace timestamps explain the frontier cliff. The 576-worker/36-group schedule
published its first and final FFN readiness at 23.62 and 35.42 us and finished
the down tasks at 39.84 us. The exact-wave 608-worker/38-group schedule did not
publish until 29.79 and 41.44 us and finished at 45.60 us. Targeted NCU measured
157.4 versus 170.5 us under profiler replay, 1.21 versus 1.12 TB/s DRAM traffic,
0.104 versus 0.098 eligible warps per scheduler, and 9.23 versus 10.04 long-
scoreboard-stalled warps per issue cycle. More active warps in the 608-worker
case did not create more eligible work. Full residency is a legality condition,
not a performance ordering.

## Validation matrix

### Dependency and legality tests

- Identity producer/consumer mapping.
- Reversed consumer group order.
- Offset and sliced views.
- Unequal producer and consumer tile sizes.
- Two-dimensional and higher-rank Cartesian grids.
- Multiple producer stores to one allocation.
- Multiple consumer loads from one allocation.
- Four-or-more-stage chains.
- Diamond fanout followed by a multi-input join.
- One event key used by several consumer tasks.
- One producer task contributing to several keys.
- More ready consumer tasks than resident workers.
- Partial in-place definitions requiring an exact event and `FamilyDone` at
  the same consumer.
- A local trigger whose completed task later contributes to `FamilyDone`.
- Access-local waits around a carried reduction accumulator without moving the
  loop regions to different workers.
- Alias/view agreement and incompatible-view fallback.
- Masked stores and loads.
- Indirect/gather indexing fallback.
- WAR and WAW root-completion ordering.
- Opaque or atomic effects retain legality errors.
- Conditional root-local temporaries do not become false cross-root state.

### Shape tests

- Batch sizes 1, 2, and 4 where the source kernel supports them.
- Short and long sequence/context lengths.
- Smaller and larger hidden/intermediate widths.
- Non-divisible extents and partial edge tiles.
- Singleton axes becoming non-singleton and vice versa.
- `static_shapes=False` compilation and execution.

### Runtime tests

- Repeated launches without stale event reuse.
- Multiple CUDA streams.
- CUDA graph capture and replay.
- Exact launch grids derived from graph-aligned worker-count choices,
  independent of
  `num_sm_multiplier`.
- Insufficient-residency configuration rejection.
- Configuration-specific rejection when changing the worker schedule changes
  compiled registers or shared memory.
- `FamilyDone` fallback when an exact schedule exceeds compiled resident
  capacity.
- Zero-task roots and symbolic task counts.

### Performance checks

- Qwen3 full-layer megakernel versus separate kernels.
- FFN subgraph at batch 1, 2, and 4.
- Synchronization overhead on simple chains.
- Register count and spill count.
- Generated-code audit confirming unchanged root computation.
- Lowered-Triton comparison proving that a counted-event `LocalTrigger`
  preserves the existing fast producer order, last-arrival test, consumer
  call, fences, and downstream publication.
- Worker-count sweeps that include the Qwen first legal point and all aligned
  Gemma points between the graph-derived lower bound and compiled residency
  limit.

Performance is a required constraint, but correctness proofs may not be weakened
to recover a few microseconds. Prefer removing overhead from the generic event
representation or traversal over adding a model-specific fast path.

## Known evidence and current baseline

As of 2026-08-22:

- The current full Qwen3 granular schedule runs at 76.09 microseconds versus
  86.27 microseconds for separate kernels on the latest cache-bypassed run and
  matches the retained custom scheduler within measurement noise.
- Shorter context configurations tested so far remain correct and faster than
  the corresponding separate-kernel runs.
- The generic FFN continuation is selected for batch sizes 1, 2, and 4 in the
  focused grouped-chain test. Batch remains part of the readiness key.
- The former ordered-singleton FFN stream is now selected from the same task
  dependency graph and access program points. No lowered-AST pointer/range
  recognizer or flattened producer-ID detector remains.
- A reversed activation-group consumer now declines continuation and uses safe
  root completion. The unsafe topology/count matcher that previously accepted
  it has been deleted.
- A repeated-launch end-to-end test exercises fan-in 3, confirming that the
  epoch-relative exact arrival target does not require power-of-two fan-in.
- Explicit `static_shapes=False` scheduling now selects the existing safe
  cooperative phase-barrier fallback when task counts are unavailable.
- The exact-codegen Gemma4 E4B probe now reaches approximately 72.2
  microseconds versus 79.8--80.0 microseconds for separate Helion, with no
  compiler changes and unchanged graph-derived legality.
- The compiler-generated Gemma4 E4B sliding non-shared schedule reaches 74.09
  microseconds versus 78.87 microseconds in the latest paired benchmark. It
  uses the same keyed-event, continuation, and frontier machinery as Qwen.
- Full non-shared remains correct and slightly faster than separate Helion at
  128.51 versus 130.82 microseconds. Sliding-shared and full-shared are correct
  but currently slow because the fused kernel uses 69--70 KiB shared memory and
  249--255 registers, leaving only two resident CTAs per SM and no legal FFN
  frontier. This is a codegen/resource-envelope issue, not evidence for a new
  dependency schedule.
- Gemma attention retiling and additional legal continuations did not improve
  the result. Joint fused-envelope tuning of existing root tile and range
  settings supplied the gain.

These observations motivate the edge-based design and define regressions that
the migration must prevent.

## Design decisions and open questions

This section preserves decisions from the preceding implementation phase.
Items that treat continuation or a selected frontier as first-class policy are
superseded by the canonical full-DAG section above.

Resolved:

- Use the current worktree as the integration base.
- Treat the older worktree as a source of planner concepts and tests only.
- Keep scheduler access facts in a dedicated `CrossLoopAccess` dataclass rather
  than expanding the autotuner's `MemoryOpFact` contract.
- Derive correctness from DeviceIR allocation/access facts.
- Preserve existing root code generation.
- Keep a conservative root-ready fallback.
- Keep static persistent traversal, exact residency validation, and current
  runtime state unless evidence requires a different dispatcher.
- Represent the FFN continuation as an exact uniform partition with one varying
  producer axis and affine outer coordinates.
- Derive cohort boundaries and worker geometry from each proved frontier. The
  autotuner selects only a legal frontier candidate index; it does not tune raw
  worker assignments or synchronization facts.
- Remove the now-unused `tile_dependency_stages` and `continuation_split`
  policy parameters.

Open, to be answered with implementation evidence:

- How to avoid compile-time enumeration for very large static task domains
  without introducing a broad symbolic-set solver.
- Whether shared-KV Gemma root tilings can reduce the fused resource envelope
  enough to admit a legal frontier; this should be explored as ordinary root
  codegen tuning, not scheduler special-casing.

## Progress log

### 2026-08-23

- Re-read the current Qwen admission/queue probe, the exact-root Gemma FFN
  probe, the full Gemma layer probe, and their lowered Triton. The common
  mechanism is a keyed event with local or predetermined direct execution;
  dynamic completion ranking and FIFO transport are both unnecessary for
  dense static task families.
- Identified the remaining topology specialization in the compiler:
  `ReadinessFrontierPlan`, `_select_readiness_frontier`, the single-cohort
  restriction, the two-round tail formula, and `readiness_frontier_body`
  together reconstruct one counted-intermediate/downstream pipeline.
- Made the full event DAG the canonical target. Dependency construction stays
  pairwise, while event composition crosses an arbitrary number of exact
  boundaries. Whole-family completion is an explicit event and does not imply
  a graph cut; exact and whole-family paths may coexist.
- Specified the graph/schedule pipeline as direct reaching-definition
  relations, event canonicalization, baseline worker-schedule construction,
  local-trigger contraction, joint readiness/liveness validation, and
  monotonic exact-to-`FamilyDone` fallback. Redundant waits require key-level
  readiness dominance rather than root-pair reachability.
- Reduced the execution vocabulary to local, direct, and root completion.
  Continuation is only the old name for local final-arrival execution; a
  frontier is a derived ready-key set, not a schedule kind or tuning choice.
- Unified ordinary traversal and direct execution in one symbolic
  `WorkerSchedule`; task placement and dispatch rounds are derived views rather
  than duplicate stored records. Local execution remains a `LocalTrigger` with
  a possible executor-worker set and a separate zero-overhead emitter.
- Replaced the FFN-specific `I = W - T - C` conception with conservative
  worker-liveness allocation over arbitrary worker schedules, local-trigger
  executor sets, and dispatch rounds.
  The formula remains only an explanation of the measured Qwen/Gemma cases.
- Audited the design against arbitrary-length chains, diamonds, fanout, joins,
  stencils, multi-stage reductions, disconnected components, outer Cartesian
  axes, nonuniform fan-in, indirect accesses, and three-or-more-round producer
  domains. Ambiguous ordering or competing direct families retain existing
  traversal and local/root-completion execution rather than adding heuristics.
- Defined the migration and deletion gates: introduce independent event/use
  relations and generic liveness first, reproduce Qwen local and Gemma direct
  lowerings, validate attention boundaries, then remove the frontier planner
  and `tile_dependency_frontier`.
- Chose `tile_dependency.py` for the semantic graph and
  `cross_loop_scheduler.py` for planning/lowering. This distinguishes logical
  tile dependencies from the existing tile-shape strategy machinery while
  making the cross-root scheduling scope explicit. The old module names and
  compatibility imports are removed only after parity.
- Kept the initial executable unit deliberately small: one complete logical
  root task. Access-local readiness remains a wait inside that task until an
  explicit same-worker and live-state continuation contract exists.
- Represented `FamilyDone` as the canonical one-key `KeyedEvent` whose
  contributors are the complete logical task family. This preserves one event
  abstraction and permits physical per-worker aggregation during lowering.
- Removed the stored root-completion topology from `CrossLoopSchedule`.
  Selected coarse relations now lower to canonical one-key event plans, while
  the emitter derives the same efficient per-worker arrival protocol from the
  final `WorkerSchedule`; the Gemma lowered kernel is unchanged.
- Routed every uniformly keyed root-entry relation through the counted-event
  lowering, including ordinary one-to-one direct waits. Excluding an access-
  local consumer now removes only that event use rather than discarding other
  consumers of the same event. Irregular predecessor sets deliberately retain
  the exact per-task fallback.

### 2026-08-22

- Removed the hardcoded task-product budget and the 64-segment acceptance
  cutoff from ready-group construction. The planner now indexes each task's
  graph-derived allocation interval, sorts producer and consumer intervals,
  and visits only possible overlaps. This preserves conservative
  flatten/unflatten support without a quadratic Cartesian scan; unknown
  address geometry still falls back to root completion. A 2,052-by-2,052
  flattened-view regression verifies that a valid relation above the former
  cutoff remains a compact 513-key event, and a strided-column regression
  verifies that exact coordinates disambiguate overlapping address hulls.
- Regenerated both final kernels after that change. Their Triton bodies are
  unchanged apart from constexpr declaration order. Fresh B200 measurements
  are 75.82 microseconds versus 86.17 separately for Qwen and 74.02
  microseconds versus 78.98 separately for Gemma sliding non-shared. The
  regenerated lowerings are
  `/tmp/qwen3_generic_keyed_interval_sweep_lowered.txt` and
  `/tmp/gemma4_generic_keyed_interval_sweep_lowered.py`.
- Added region-aware reaching definitions for partial and in-place writes.
  Exact writes split earlier reaching regions; unknown writes remain
  conservative and do not erase definitions they cannot prove they cover.
- Added multi-contributor keyed events with independent producer task-to-key
  and consumer task-to-key maps. Uniform fan-in is derived from those maps;
  repeated maps are compressed with an affine period rather than emitted as
  large tables.
- Generalized repeated predecessor-set coalescing to a single producer. This
  turns Qwen's 4,096 attention-to-merge task polls into 128 ready-group counters
  with fan-in eight and one wait per merge task.
- Added direct fan-in-one chaining for nested on-ready actions. A uniquely
  owned key executes its next opaque task in program order without allocating
  or polling an unnecessary atomic counter.
- Replaced Qwen's custom partitioned-attention lowering with the generic event
  graph: projection-to-Q/K fan-in 16, Q/K plus V-to-cache fan-in 17,
  Q/cache-to-attention fan-in five, split-to-chunk ready groups, chunk-to-final
  fan-in 16, and direct final-to-quant chaining. Deleted
  `_match_partitioned_dependency_pipeline` and its custom state and emitter.
- Retained the task-aligned Qwen source adaptations used to expose logical
  dependencies without changing scheduler codegen: one cache task per KV head,
  flat Q/K normalization, task-aligned split/merge loops, and head-shaped
  attention quantization. These are benchmark/source changes and remain
  explicitly separate from the compiler scheduler.
- Found and removed a redundant exact task edge in Gemma. The direct
  `post_ff -> final` wait was already ordered by the whole-root path through
  PLE gate and projection. The new generic transitive-elision pass trusts only
  root-completion/preordered edges, never a partial fine-grained edge. This
  improved Gemma from 76.09 to 74.09 microseconds.
- Revalidated Qwen at 76.09 microseconds versus 86.27 separately and Gemma
  sliding non-shared at 74.09 versus 78.87. Saved final lowered Triton at
  `/tmp/qwen3_generic_keyed_final_lowered.txt` and
  `/tmp/gemma4_generic_keyed_final_lowered.py`, with NCU CSVs at
  `/tmp/ncu_qwen_generic_keyed_final.csv` and
  `/tmp/ncu_gemma4_generic_final.csv`.
- Added focused coverage for single-producer ready-group fanout, counter-free
  fan-in-one nested continuation, and exact task waits dominated by a proven
  whole-root path. The focused result is 73 tests and 13 subtests passing, with
  four expected skips; Ruff and `git diff --check` are clean.
- Fixed two remaining places where a uniform event partition implicitly
  assumed an `on_ready` continuation. Producer reordering and reverse
  consumer-coordinate reconstruction now occur only when the dispatch plan
  actually attaches an on-ready task; the same event can instead lower to an
  ordinary keyed counter and wait. The continuation-disabled Qwen ablation now
  compiles and runs again.
- Revalidated shape changes after the matcher deletion. Context 4096 with 64
  attention splits measured 72.39 microseconds versus 82.66 separately. A
  consistent 16-query-head, 4-KV-head, hidden-2048, intermediate-6144 shape
  measured 40.08 versus 45.99 microseconds. The failed 16-head/hidden-4096
  invocation was rejected by the probe's model identity
  `hidden == q_heads * head_dim`, before scheduler comparison.
- Audited the frontier tuning API. Exact candidate count depends on the chosen
  block sizes and is therefore unavailable when the static `ConfigSpec`
  fragment is created. Keep the bounded internal index and reject invalid
  values during lowering until Helion has a general dependent-fragment API.

### 2026-08-21

- Built a 14-root Gemma4 E4B layer megakernel from the existing Helion kernels
  without changing compiler source.
- Audited the emitted Triton for all four representative layer variants rather
  than relying only on timing and resource summaries.
- Initially worked around a specialized-expression leak by passing `q_per_kv`
  directly; later logical-task work in this log removed that workaround.
- Initially found that root outlining failed to thread tensor-descriptor
  handles used by a root body; retained pointer indexing in the benchmark while
  fixing and validating the generated outlined-root ABI later in this log.
- Confirmed that making the Gemma FFN continuation legal by removing L2 remaps
  is not sufficient for performance: the ordinary continuation regressed to
  roughly 153 us, and the fully admitted three-stage pipeline regressed to
  roughly 263 us. No Gemma-specific scheduler path was added.
- Reduced full-attention shared-memory pressure with a 32-token inner attention
  tile and selected simple head-width-based launch defaults.
- Validated correctness and interleaved performance for layers 0, 5, 24, and
  29. The weighted 42-layer total is approximately 4.667 ms for the current
  megakernel versus 3.680 ms for current separate Helion kernels.

### 2026-08-20

- Selected the current worktree as the integration base.
- Compared the current implementation with the older
  `cross_loop_dependencies.py` experiment.
- Identified allocation grouping, access IDs, generic event/wait records, and
  affine predecessor proofs as the reusable parts of the older design.
- Identified the older global cursor, root-entry-only waits, and workspace reset
  scheme as pieces not to port wholesale.
- Recorded the reversed-group unsoundness, batch-shape fallback, and dynamic
  shape assertion as concrete migration tests.
- Created this living architecture and implementation plan before further
  compiler changes.
- Added a scheduler-specific `CrossLoopAccess` fact, allocation hazard graph,
  strict affine predecessor proof, and evaluator for concrete predecessor task
  sets. This metadata is collected only for explicitly scheduled kernels and
  does not yet replace emitted scheduling code.
- Lowered allocation edges into the planner's durable event/wait vocabulary.
  One producer task or root event is shared across fanout consumers, and a
  root-completion wait subsumes finer waits for the same root pair.
- Classified waits by placement: top-level coordinates use root admission;
  coordinates introduced by nested loops remain access-local work for the next
  phase rather than being guessed at root entry.
- Added focused tests for identity and unequal-tile mappings, multidimensional
  batch keys, reversed-index fallback, masked and multiple-store fallback,
  RAW/WAR/WAW ordering, and fanout.
- Made the existing grouped-continuation fast path require an exact match
  between its assumed producer cohorts and the DeviceIR-derived predecessor
  sets. A normal grouped chain retains the fast path; the reversed-group
  adversarial chain now uses the safe whole-value schedule and is correct.
- Distinguished tile-width subscripts from scalar `tile.id` subscripts in the
  readiness model, so activation values and per-group scales use the same
  coordinate relation without pretending they cover the same number of tensor
  elements.
- Added an end-to-end grouped-chain test covering batch 1 in normal and reversed
  group order, plus batch 2. Batch 1 retains grouped scheduling only for the
  proven normal mapping; reversed order and batch 2 use the safe fallback and
  remain correct.
- Added a generic root-admission task-event path to the existing static
  persistent traversal. It derives exact predecessor tasks from affine access
  maps and leaves each root body opaque. Cartesian batch grids, unequal tile
  sizes in both directions, partial edge tiles, batch sizes 2 and 4, chains,
  fanout, joins, and repeated launches now use this common path in focused
  tests.
- Made event indexing follow each root's configured PID-axis order rather than
  assuming source-axis order. Producer and consumer roots may therefore use
  different loop-order permutations without changing the dependency proof.
- Made nonzero/dynamic grid starts and non-unit/dynamic grid steps an explicit
  proof boundary. Until loop origins are represented in the affine map, such
  axes use root completion; this prevents local task IDs from being mistaken
  for absolute tensor coordinates.
- Kept roots with a nontrivial L2 PID remap on root completion for now. Exact
  task events are indexed in logical Cartesian PID order, so using them across
  an unmodeled remap would be an unsound identity assumption. The eventual
  solution is to represent the remap as part of task identity, not to add an
  L2-specific dependency pattern.
- Verified that exact task events survive repeated CUDA Graph replay and remain
  isolated across two concurrent CUDA streams through the existing
  stream-local persistent-state allocator.
- Replaced the dynamic-task-count assertion with the existing cooperative
  phase-barrier fallback. An explicit `static_shapes=False` regression test now
  compiles and runs correctly.
- Simplified source legality analysis to recognize names assigned on every arm
  of an `if` as root-local, eliminating the previously observed false
  cross-root dependency without adding a scheduler-specific exception.
- Revalidated the Qwen3 granular probe with its matched configuration after the
  generic task-event integration: 250 registers, zero spills, approximately
  76.4 microseconds versus approximately 80.7 microseconds for the
  separate-kernel control in the latest run.
- Revalidated the shorter 2048-token / 32-split case after the generic
  continuation landed: 247 registers, zero spills, approximately 71.63
  microseconds versus approximately 76.18 microseconds for the separate-kernel
  control.
- Revalidated the standalone FFN after the generic continuation landed: 78
  registers, zero spills, approximately 37.35 microseconds versus approximately
  38.01 microseconds for the matched three-kernel control.
- Re-ran whole-layer ablations to determine which migration shims are currently
  performance-critical. With the same matched configuration, disabling grouped
  continuation regressed to approximately 86.7 microseconds, disabling the
  partitioned prefix pipeline to approximately 80.1 microseconds, and disabling
  reduction fanout to approximately 78.1 microseconds. Disabling every
  specialized family produced approximately 93.8 microseconds. This makes
  nested FFN continuation the first generic replacement target and argues
  against deleting the existing emitters before equivalent event coarsening is
  available.
- Added `UniformTaskPartition`, derived solely from exact predecessor sets. Its
  proof requires nonempty constant-cardinality sets that are disjoint, cover
  the producer domain, and factor into one varying producer axis plus affine
  outer-coordinate mappings.
- Lowered accepted partitions in consumer-major coordinate order while
  preserving the producer root's configured PID decomposition at final
  dispatch. Fan-in is the proved set cardinality, and last arrival uses an
  epoch-relative exact target rather than modulo arithmetic.
- Composed the continuation with the graph-derived nested-loop cohort relation.
  A deterministic occupancy-bounded worker split overlaps the producer tail
  and downstream consumer only when both fit in one resident wave. No new
  public or autotuner knob was added.
- Deleted the flattened-task-ID grouped continuation detector,
  completion-prefix heuristic, custom plan type, and custom emitter. The FFN
  fast path is now selected from dependency edges and exact predecessor sets.
- Validated grouped FFN lowering at batch sizes 1, 2, and 4, reversed-group
  fallback, differing producer/consumer PID orders, overlap and incomplete
  coverage rejection, and non-power-of-two fan-in 3 across repeated launches.
- Revalidated the granular Qwen3 layer after deleting the custom continuation:
  252 registers, zero spills, 74.92 microseconds median versus 80.39
  microseconds for separate kernels (10 measurement repeats with 10 replays).
- During review, moved continuation proof ahead of cohort wait removal so a
  root with mixed root-entry and nested access dependencies cannot be accepted
  from only a subset of its predecessor relation.
- Routed whole-root completion counts through the actual capped launch-worker
  count, preventing fallback edges from waiting for non-launched workers when a
  composed continuation pipeline reduces the grid.
- Removed the dead preflight-wait cloning experiment and the unused
  `tile_dependency_stages` and `continuation_split` public parameters.
- The final focused validation covers 62 passing tests, 5 expected skips, and 9
  subtests; Ruff, focused Pyrefly checks, and `git diff --check` pass.
- Retiled the exact generated Gemma4 roots while keeping scheduling primitives
  unchanged. The best retained B200 result is approximately 72.2 microseconds
  versus 79.8--80.0 microseconds for separate Helion.
- Verified from lowered Triton and NCU that QKV N=8/K=256 and output projection
  N=16/K=512 improve the combined resource envelope. Attention Q retiling,
  attention/output continuations, relaxed polling, and worker spreading were
  negative.
- Reproduced the retained Gemma configuration from the integration worktree at
  72.25 microseconds versus 80.04 microseconds for separate Helion. The
  gate-weight `evict_first` annotation is required; omitting it regressed to
  84.4 microseconds without changing the schedule.
- Replaced the earlier confounded 592-worker ablation with a controlled
  worker/frontier matrix that holds root codegen fixed. The winning Gemma point
  is 576 workers and 36 ready activation groups at 75.77 microseconds versus
  80.03 microseconds for separate Helion. A 592-worker launch at the same
  36-group frontier measured 76.99 microseconds, while an exact 608-worker,
  38-group wave measured 83.48 microseconds. Full residency remains a legality
  invariant, not a performance ordering.
- Verified that the successful worker assignment need not be independently
  tuned. `initial_producer_tasks == workers` is critical: 560 workers at the
  576-task frontier measured 91.00 microseconds. One worker per downstream
  consumer tile is likewise important: reducing 160 consumers to 144 or 80
  measured 88.53 and 87.48 microseconds. The compiler can derive both cohorts
  after the autotuner selects a legal dependency frontier.
- Replaced the ordered-input singleton matcher with graph-derived
  per-coordinate counted readiness at explicit access program points. The
  lowered Qwen kernel remains structurally identical to the retained fast
  kernel apart from constexpr declaration order.
- Made the outlined-root ABI include live compiler-created preamble values.
  This mechanically threads tensor descriptors (and analogous generated
  values) into opaque helpers instead of leaving parent-scope free names.
- Validated that ABI change with an actual scheduled two-matmul kernel using a
  tensor descriptor and with the Gemma4 E4B layer using a descriptor for its
  outlined down-projection root. The latter compiled and ran correctly with
  the descriptor present in the helper signature (150 registers, zero spills
  in the diagnostic configuration).
- Revalidated the current Qwen3 granular lowering after the planner migration:
  its 1,596-line Triton body is unchanged from the committed fast lowering
  except for constexpr declaration order, with 252 registers, zero spills, and
  a 75.8 microsecond median in the latest run.
- Revalidated the retained Gemma4 pointer configuration after the planner
  migration: all checked outputs and cache writes pass, with 168 registers and
  four spills. Timing was not repeated because unrelated processes occupied
  the GPU; the last idle-GPU result remains approximately 72.2 microseconds.
- Deleted the legacy pre-DeviceIR dependency analysis, its compiler passes,
  copied access/edge state, and the duplicate edge list in the resolved
  schedule. Every multi-root kernel now derives legality and stage boundaries
  from allocation-identity accesses collected from DeviceIR.
- Kept only two orthogonal checks outside that graph: the normal host walker
  records explicit barrier boundaries, and DeviceIR name lookup reports an
  attempted cross-root device-value capture. Neither constructs memory edges.
- Added atomic operations to the allocation graph as conservative writes and
  verified both unscheduled rejection and scheduled execution.
- Revalidated the exact Qwen3 and Gemma4 lowerings after deleting source
  analysis. Both generated files retain the same line counts and differ from
  the prior fast versions only in constexpr declaration order; correctness and
  resource use remain unchanged.
- Validated the proposed counted-event `on_ready` representation before the
  compiler refactor. In the Gemma Triton probe, direct inline election and a
  reusable counted-event helper generated byte-identical SASS with identical
  resources; a paired run measured 73.061 and 73.143 microseconds respectively.
  The saved lowered source is `/tmp/gemma4_counted_event_ab_lowered.py`, with
  full compiler dumps under `/tmp/gemma4_direct_ir` and
  `/tmp/gemma4_generic_ir`.
- Revalidated the existing Qwen Triton probe, which already factors the
  counted arrival from its inline ready action. The persistent FFN core measured
  37.046 microseconds versus 38.948 microseconds for the matched three-kernel
  graph, with exact outputs and compiler dumps under
  `/tmp/qwen_counted_event_ir`.
- Made logical task identity authoritative in planner proofs and readiness
  keys. Existing physical PID order and L2 grouping are translated only at the
  dispatch boundary, so root bodies retain their original code generation.
- Enabled the graph-derived continuation/frontier pipeline through later root-
  completion dependencies; a continuation consumer can now publish the root
  completion needed by downstream roots.
- Added compiler-derived legal frontier candidates and an internal candidate
  index. Gemma selects 576 workers at frontier index 5; the 608-worker candidate
  is rejected by exact post-compilation residency at a measured capacity of
  592.
- Replaced expensive exact task events for generic multi-producer joins with
  root-completion waits. In the Gemma layer this reduced the timing-only result
  from about 96.57 microseconds with exact access-local waits to about 73.69
  microseconds while retaining correctness.
- Revalidated Qwen at about 74.92 microseconds versus 85.10 microseconds
  separately, and Gemma at about 74.75 microseconds versus 78.72 microseconds
  separately in a paired run. The corresponding lowered Triton is saved at
  `/tmp/qwen3_helion_logical_frontier_lowered.py` and
  `/tmp/gemma4_helion_latest_lowered.py`.
- Replaced the named continuation-plan hierarchy with a counted event carrying
  an optional `on_ready` opaque task and a separate readiness-frontier plan.
  Qwen and Gemma lowered bodies are unchanged apart from constexpr declaration
  order.
- Removed the public `producer_order` and `epoch_replicas` migration knobs and
  deleted their inactive replicated-counter branches. The compiler retains the
  previous default producer-order decision, and both model lowerings changed
  only by algebraic simplification of one-element state indexing.
- Routed all four last-arrival actions in the partitioned-attention
  compatibility emitter through the same epoch-relative counted-event helper.
  The exact Qwen layer remains at about 74.92 microseconds versus 85.10
  microseconds separately, with 252 registers and no spills.
- Added root completion as the stable `-1` member of the internal frontier
  choice. A lowering default mismatch that still interpreted a missing value
  as frontier zero was fixed by defining the default once in the planner. A
  dedicated integration test now verifies that an otherwise frontier-capable
  graph defaults to root completion unless a nonnegative candidate is selected.
- Swept representative Gemma specializations. The tuned sliding non-shared
  layer still selects frontier index 5, `G=36`, and `W=576`. The full
  non-shared specialization requires at least `W=416` for the same FFN graph,
  but its compiled 255-register, 70,016-byte-shared kernel supports only 296
  resident CTAs. Candidate zero is therefore rejected before launch, while
  root completion is correct and measured 131.13 microseconds versus 131.23
  microseconds for separate Helion in a short paired run. Sliding-shared and
  full-shared representative layers also compiled and ran correctly with the
  first fine-grained frontier.
- Removed the Gemma `q_per_kv` source workaround. The natural
  `q_heads // kv_heads` expression now produces the concrete logical task
  extent and the same lowered attention dispatch.
- Derived singleton worker affinity from predecessor participation instead of
  assigning every singleton to worker zero. Each singleton uses an otherwise
  idle predecessor-external worker when available and stable distinct high
  workers otherwise. This changed only scheduling placement, kept Qwen at
  74.66 microseconds versus 85.14 separately, and improved the compiler Gemma
  result to 73.16 microseconds versus 78.74 separately.
- Verified that Gemma's two natural singleton RMSNorm/residual roots now lower
  and execute correctly through generic multi-producer joins. With the same
  fused root settings they measured approximately 75.17 microseconds, versus
  73.16 microseconds for the retained output-tiled forms. The two adaptations
  remain explicitly documented performance choices rather than compiler
  legality requirements.
- Generalized every structural-plan entry wait from one predecessor to an
  arbitrary ordered set of graph-derived root-completion dependencies.
- Added conservative allocation-view normalization for inserted or removed
  size-one dimensions and explicit full slices. This upgrades the real Qwen
  attention root 8-to-9 edge to task readiness while leaving nontrivial
  flatten/unflatten views on root completion. Singleton producer events are
  canonicalized back to the equivalent root-completion representation, so the
  final Qwen and Gemma lowerings remain unchanged apart from constexpr
  declaration order.
- Revalidated the unchanged end-to-end kernels on an idle B200: Qwen measured
  74.95 microseconds versus 85.12 microseconds separately; Gemma measured
  74.43 microseconds versus 78.51 microseconds separately, with 206 registers,
  zero spills, and 35,200 bytes of shared memory.
- Tested the simpler source-first alternative for Qwen attention. The split
  writes canonical `(split, query_head, dimension)` partials and the merge uses
  direct multidimensional indexing. It passes strict end-to-end validation and
  measured 74.16 microseconds versus 84.39 microseconds separately, compared
  with 74.23 versus 85.08 microseconds for the retained source in an adjacent
  run. Both use 252 registers, zero spills, and 17,408 bytes of shared memory.
  The saved lowered Triton is
  `/tmp/qwen3_canonical_attention_source_lowered.py`.
- Disabling the partitioned-attention compatibility emitter for that canonical
  source remains correct but measured 78.84 microseconds versus 84.34
  microseconds separately, with 248 registers. Its lowering is saved at
  `/tmp/qwen3_canonical_attention_root_completion_lowered.py`. The remaining
  roughly 4.7-microsecond gap is therefore scheduling overlap, not view
  codegen. The allocation graph still classifies split-to-merge and the first
  merge-to-final-merge edges as root-ready because it does not retain the
  linearized query-head expression or bounded split interval.
- Expressing the merge intervals directly as `hl.tile` ranges removes that
  proof gap without compiler changes: split-to-chunk-merge and
  chunk-merge-to-final-merge both become task-ready. Keeping one query head per
  merge task preserves the original root resource shape (the full kernel has
  no spills), but the generic exact-event traversal measures 83.29
  microseconds versus 84.58 microseconds separately.
- Grouping all four query heads into one merge task makes the predecessor sets
  a disjoint uniform partition. Generic counted events now compose both merge
  levels and publish the final root-completion event correctly. This path
  measures 81.30 microseconds versus 87.27 microseconds separately, with 255
  registers and 26 spills; its no-wait floor is 74.56 microseconds. The saved
  lowering is `/tmp/qwen3_task_aligned_counted_chain_lowered.py`. The remaining
  gap comes primarily from root-completion barriers before split attention,
  not from missing merge dependency information.
- A direct producer-fan-out continuation was also tested and rejected. With
  one query head per merge task, the last split arrival executed four ready
  merge tasks serially and measured 88.27 microseconds. Two heads measured
  84.27 microseconds. The ordinary four-head merge tile is therefore the
  simpler current choice; a future worker-ready queue may support fan-out, but
  it is not required for the first generic replacement.
- The task-aligned one-head source also exposed a correctness defect in the
  remaining compatibility matcher: it accepted the familiar task counts while
  assuming the old flattened logical axis order and produced incorrect
  attention output. The probe now disables that matcher automatically. The
  generic replacement must consume the dependency graph's logical coordinates
  directly; this is further evidence for deleting rather than extending the
  matcher.
- Generalized exact readiness to include proved RAW, WAW, and WAR hazards. The
  flat in-place Q/K source then proved the expected 16 QKV-tile predecessors
  per Q/K-head task, but exact task publication and polling regressed the layer
  to 83.50 microseconds versus 80.31 microseconds separately.
- Added a conservative partial-prefix counted partition: disjoint uniform
  predecessor sets may cover an exact physical prefix while the unowned suffix
  executes normally. CUDA tests verify both the continuation and preservation
  of an in-place allocation's untouched suffix. The emitted Qwen Triton uses
  40 counters with fan-in 16 and no QKV-to-Q/K polling loop.
- The safe partial-continuation Qwen path measured 82.42 microseconds versus
  86.35 microseconds for the same-source separate graph. It still cannot expose
  the useful cache/attention overlap because the current single-writer
  allocation history loses the fact that V remains defined by the projection
  root. A temporary conservative pass-through wait preserves correctness but
  is explicitly a deletion target, not the final design.
- Adopted the holistic keyed-event refactor above. Multi-contributor event IR
  and region-aware reaching definitions now reconstruct the Qwen attention
  pipeline from the dependency graph; the old partitioned-attention matcher
  and its separate lowering have been deleted.

### 2026-08-22 cleanup

- Removed the public `TileDependencySchedule` opt-in and its resolved-schedule
  mirror. Multi-root dependencies are now detected from DeviceIR allocation
  facts, and source-order phase boundaries are installed directly.
- Promoted `tile_dependency_frontier` to an ordinary `Config`/autotuner field.
  Its search bound is derived from the actual potential stream axes; the former
  fixed maximum of 63 has been deleted.
- Collapsed worker epochs, exact task epochs, counted events, access cohorts,
  singleton completion, root completion, and reduction-fanout bookkeeping into
  one cache-line-aligned persistent event-state allocation. Logical event kinds
  retain distinct offsets, but the launcher now receives one state tensor.
- Canonicalized singleton completion onto the same root-completion event state.
  A singleton is simply a root with one active contributor, not a separate
  synchronization protocol.
- Removed duplicate ownership from `TaskFamily`, `InstantiatedTaskFamily`,
  `CrossLoopDependencyPlan`, `AccessProgramPoint`, and keyed-event mappings.
  Source-order tuple position identifies roots; the dependency plan owns
  accesses; and compact task-to-key segments are derived from the authoritative
  mapping rather than stored independently.
- Unified the three opaque-loop AST cloning/rewrite walkers behind one helper
  while retaining structural fingerprints that prove source computation was
  not changed.
- Kept Triton device-function outlining. Fully splicing every helper into the
  launch body increased Gemma registers from 204 to 230 and regressed roughly
  74.0 to 77.3 microseconds. Keeping only explicitly noinline regions caused
  Qwen to rise from 251 registers with no spills to 255 registers with 20-byte
  spills and regress roughly 76.1 to 78.3 microseconds. These helpers remain
  internal to one launched kernel and are a register-lifetime boundary, not a
  separate-kernel schedule.
- The cleanup removes 238 net production lines relative to the preceding
  keyed-event commit. Current production code remains about 7,001 net lines
  above `main`; halving that safely would require deleting substantive
  dependency proofs or the still-performance-critical reduction-replication
  path rather than merely consolidating abstractions.
- Final focused validation passes 77 tests, 4 expected skips, and 13 subtests.
  The unified event state preserves 251 registers/no spills for Qwen and 204
  registers/no spills for Gemma. Adjacent cache-bypassed B200 runs measured
  76.38 versus 76.40 microseconds for Qwen and 74.94 versus 75.10 microseconds
  for Gemma before/after the state-layout change, within measurement noise.

### Final review follow-up

- Replaced the opaque counter stride `32` with a 128-byte cache-line alignment
  and derive its word count from the `uint32` event-state dtype. This remains a
  hardware-layout policy, not a model-shaped limit. Removing the separation
  regressed Qwen from 76.41 to 78.25 microseconds and Gemma from roughly 74.9
  to 79.70 microseconds.
- Made dependency legality conservative for dynamic tensor layouts,
  multidimensional nonzero storage offsets, and non-injective layouts such as
  stride-zero views. Unproved cases now fall back to root completion rather
  than using size hints or coordinate disjointness as proofs.
- Removed the unused partitioned-materialization geometry helper, redundant IR
  fields, and the midpoint access-loop split. A direct cache-bypassed Qwen
  split/no-split comparison measured 75.49 and 75.48 microseconds. The complete
  review patch measured 75.59 versus 75.69 microseconds for Qwen and 74.45
  versus 74.27 microseconds for Gemma, with unchanged resources.
- Keep `_OneWaveReductionFanout` temporarily. A stricter closed-subgraph
  matcher removed Qwen's useful upstream continuation and regressed the layer
  to roughly 84 microseconds. Its replacement therefore remains a distinct
  planned computation transform: prove exact external boundaries and the
  upstream partition from the dependency graph, and prove that replicating the
  singleton body is pure or idempotent. Do not extend the current shape-based
  matcher with additional model cases.
- Final focused validation passes 80 tests, 4 expected skips, and 15 subtests.
  Qwen strict end-to-end validation and Gemma output/cache validation also
  pass. Ruff and formatting pass; targeted Pyrefly reports only the two
  pre-existing duplicate-SymPy-module errors in `device_ir.py`.

## 2026-08-23 scope-aware action DAG refactor

### Architectural decision

The target architecture is **one scope-aware memory/event graph plus
non-preemptive outer task strands**. This completes the full-DAG design rather
than adding another scheduler beside it.

Nested loop iterations are dependency-visible action checkpoints, but they are
not independently movable tasks. A worker still enters one outer root task and
runs its program-order strand. The graph may place waits and publications at
proved nested boundaries while accumulators, descriptors, and other live state
remain local to that strand. This avoids continuation-state transport and keeps
the original root body unchanged apart from synchronization and mechanical loop
segmentation.

The architecture is:

```text
reachable DeviceIR callsite/scope tree
                ↓
logical action domains on non-preemptive root strands
                ↓
one-sided allocation-coordinate access maps
                ↓
exact producer-action → consumer-action overlap relations
                ↓
canonical keyed-event DAG
                ↓
schedule-derived milestone quotient + outer worker placement
                ↓
generic waits/publications at stable scope boundaries
```

### Dependency representation

1. Build execution scopes by traversing reachable DeviceIR callsites. Scope
   identity is `(root, callsite path)`, not merely `graph_id`: sibling loops may
   reuse a graph body, transformed graphs may be copied, and one graph body may
   have multiple callsites.
2. Keep the root `TaskFamily` as the only worker-placement domain. Each nested
   segment has an `ActionDomain` containing its complete logical coordinates:
   outer/root axes first, then enclosing nested-loop axes.
3. Associate every load and store with its stable execution scope and lexical
   operation order. Normalize each access independently into an exact map from
   logical action coordinates to an allocation footprint.
4. Build producer-action to consumer-action relations solely from allocation
   footprint overlap and reaching definitions. Do not maintain a second
   producer/consumer dimension-pairing proof language in the scheduler.
5. Preserve exact boxes, intervals, and the small strided forms required for
   ordinary views. Do not reduce accesses to address hulls. Batch and all other
   outer axes stay in the logical coordinates; L2 remapping remains a later
   physical traversal decision.
6. Unknown, indirect, conditional, non-injective, or dynamically shaped cases
   lift monotonically to an enclosing proved scope, ultimately `FamilyDone`.

`tile_dependency.py` owns memory facts, action domains, configuration-time
relation instantiation, and the canonical event graph. The scheduler consumes
only action/event relations. It should not import `TileAccess`,
`AllocationRegion`, or a separate pairwise affine predecessor proof.

### Event and scheduling model

- Root entry/exit and nested loop entry/exit are instances of the same action
  endpoint abstraction. Root completion is the one-key family-completion event
  over logical action completions, including work executed through local or
  direct readiness.
- Exact predecessor signatures are the source of truth. Repeated signatures
  become keyed events; multi-producer joins union and deduplicate the producing
  actions.
- The scheduler assigns only outer task strands to ordered worker-program
  positions. Nested waits/publications remain inside their owning strands.
- Local final-arrival execution is valid only for a complete movable root-task
  start. A locally triggered strand with a later blocking nested wait is
  conservatively disallowed until action-level liveness proves progress.
- Progress validation combines event dependencies with scope/program-order
  edges. A root-completion edge is a component cut only when no exact event path
  still connects the same components.
- Multiple action instances with one logical key map to ordered worker-program
  positions, not to one ambiguous `key → worker` assignment.

### Generic milestone quotient

The Qwen 64/32 and Gemma 36/4 FFN handoffs are not separate scheduling kinds.
They are the same graph rewrite:

1. Start with exact predecessor sets for every nested consumer action.
2. Account for the actual worker-program position at which each predecessor
   becomes complete.
3. Coalesce adjacent consumer actions with the same effective readiness into
   maximal representable loop segments.
4. Replace each segment's individual keys with one milestone key whose
   predecessor set is the deduplicated union of those exact keys.
5. Derive each milestone's arrival count from that union. Arrival counts may be
   nonuniform across segments.

This is a schedule-derived quotient of the exact action DAG, not an FFN
matcher, an access cohort, or a new schedule policy. The same operation applies
to any contiguous nested consumer range whose exact dependencies admit the
quotient. The lowering emits one wait before each original loop segment and
preserves the loop body and range attributes.

The quotient also has a deliberately narrow lowering contract: each consumer
action waits on one derived readiness key. A raw relation that would require a
consumer to poll several independent keys must first be quotiented into one
deduplicated predecessor signature. If that cannot be represented, it lifts to
an enclosing action or `FamilyDone`; the code generator does not synthesize an
arbitrary multi-key polling loop. This keeps scheduling policy in the graph
pass and prevents a merely legal, overly fine-grained event encoding from
silently becoming the chosen schedule.

### Publication and lowering constraints

- A nested publication is legal only at a boundary proved to execute exactly
  once per counted action. Conditional branches, while loops, zero-trip or
  dynamic loops, and duplicated callsites lift to an enclosing guaranteed
  boundary unless separately proved.
- Multiple stores within one strand are ordered reaching definitions. Later
  overlapping stores remain prerequisites; repeated stores by one action are
  deduplicated before counting.
- Publication requires CTA-wide completion and release semantics. For
  pipelined or unrolled `tl.range`, publication occurs only between emitted
  segments where all relevant asynchronous work is complete, not naively after
  a source-level iteration.
- Segment boundaries must be representable in the configured physical inner
  traversal. Logical coordinates prove dependencies; lowering order determines
  which exact relations can be rendered as contiguous segments.
- Stable DeviceIR scope metadata locates the lowered loop. Mechanical AST
  cloning/splitting may remain initially, but marker-based discovery and
  cohort-specific emitters must not return.

### Generality check

- **Qwen FFN:** exact W13 producer actions quotient into the existing 64/32 K
  milestones for activation/W2 while retaining batch in every key.
- **Gemma FFN:** the same quotient produces 36/4 milestones; no model, tensor
  name, or task-count matcher is involved.
- **Attention:** ordinary exact keyed readiness remains an edge-local result.
  Multi-stage attention chains compose through the same event DAG without a
  special attention lowering.
- **RMSNorm → matmul:** future nested producer publication can expose tiles as
  the reduction/matmul strand reaches proved checkpoints. It does not require
  making inner iterations independently schedulable.
- **Unknown future chains:** arbitrary DAG paths compose by event reachability
  and program order. The scheduler does not search for a named three-node FFN
  shape or a named two-node attention shape.

### Implementation checkpoint

Completed in the current scheduler checkpoint:

- Added stable DeviceIR callsite scopes and propagated their identity to
  lowered loops.
- Added configured action domains with full outer and nested logical axes.
- Moved configured allocation-overlap relation construction into
  `tile_dependency.py`; root and nested relations now use the same proof.
- Added generic nested-consumer keyed events and schedule-derived milestone
  quotienting with nonuniform arrival counts.
- Made `tile_dependency.py` instantiate one canonical predecessor relation per
  allocation hazard and identify every hazard independently. Root and nested
  keyed events consume those relations instead of rerunning a second pairwise
  affine proof in the scheduler.
- Deleted the legacy semantic root-event graph, affine predecessor maps,
  uniform-partition proof, task-readiness builder, exact task-event buffers,
  and raw per-task polling path. Configured allocation-overlap relations and
  their keyed-event quotient are now the single source of dependency truth.
- Track schedule coverage by `(hazard, consumer callsite)` dependency points
  rather than by root pair or source-level hazard alone. This lets exact action
  events and family completion coexist for distinct regions and repeated
  callsites between the same roots without treating an unrelated exact path as
  proof that a coarse edge was covered.
- Recompute dependency coverage from the counted events and waits that will
  actually be emitted. Redundant fine-grained uses are removed only after the
  selected root-completion order proves them unnecessary.
- Kept counted-event lowering intentionally scalar at the consumer: arrival
  counts may differ by key, but every consumer action waits on one quotient
  key. Unquotientable multi-key relations safely fall back to family
  completion. This preserves the useful batch/group FFN handoff without
  selecting Gemma's legal but slower QKV-to-attention polling schedule.
- Deleted `AccessCohortPlan`, `_derive_access_cohorts`,
  `place_access_ready_consumers`, access-marker discovery, cohort counter
  allocation, cohort lowering, and the access-specific event placement mode.
- Generalized event contributions to root or nested producer scopes and added
  mechanical publication at stable nested-loop scope boundaries. Nested
  actions remain on their owning non-preemptive task strand; worker placement
  still applies only to outer root tasks. A focused nested-producer streaming
  test now lowers to keyed waits and publications without root completion.
- Added generic same-strand program-order reasoning. For every later access,
  the dependency pass computes which earlier scope actions have necessarily
  executed in the same strand. A prior wait suppresses the later fallback only
  when its acquired producer-action set covers that access's exact predecessor
  set for every action. A scalar read after a streamed reduction is therefore
  covered, while the same read before the reduction correctly retains family
  completion.
- Exclude any root containing a nested blocking wait from local final-arrival
  execution. This is the conservative non-preemptive liveness rule until a
  future action-level proof can establish that such a strand cannot occupy a
  worker needed for its own progress.
- Canonical action relations now directly name producer root, producer scope,
  and producer action. The scheduler no longer reconstructs producer access
  overlap from allocation regions. Structurally identical keyed events are
  merged before scheduling, eliminating duplicate waits while retaining the
  original dependency points.
- Generalized one outer task strand to contain any number of supported sibling
  action scopes. Placement is solved once from the union of their exact event
  constraints; each scope then receives the same generic milestone quotient.
  This removes the one-scope scheduler and lowering assertions. A focused
  two-producer/two-scope kernel starts its first reduction while the second
  producer is still running, waits at both stable scope boundaries, and uses no
  family-completion barrier.
- Preserve action-scope identity when cloning untouched AST loops. This makes
  repeated mechanical segmentation compositional rather than silently losing
  the metadata needed to find later sibling scopes.
- Instrument nested producer publication before consumer segmentation. Split
  loop clones therefore retain the original logical action-coordinate origin;
  they cannot rebase a later segment to action zero and republish the wrong
  event keys. A three-root nested load/store chain now waits, computes, and
  publishes through the same scope while preserving exact downstream keys.
- Verified nested publication with both `tl.range(num_stages=4)` and
  `tl.range(loop_unroll_factor=2)`. The original range attributes remain in the
  lowered kernel, and the CTA-wide completion plus release publication remains
  ordered after the loop body's stores.
- Made nested publication an explicit lowering capability. The dependency DAG
  still records every nested store, but a configured tensor-descriptor store is
  not offered as an early-publication endpoint until its asynchronous
  completion protocol is proved. Its dependency monotonically falls back to an
  enclosing root event or family completion.
- Added an explicit two-axis nested-consumer test. The scope and accesses remain
  visible in the action DAG, while the current one-axis renderer declines the
  optimization and emits the proven family-completion fallback without a
  topology-specific branch.
- Reproduced Qwen's 64/32 and Gemma's 36/4 loop segmentation through stable
  action scopes. The current saved lowerings are
  `/tmp/qwen3_multi_scope_final_lowered.py` and
  `/tmp/gemma4_multi_scope_final_lowered.py`; their event topology and resource
  use match the prior best lowerings.
- Revalidated Qwen at 78.20 microseconds versus 85.52 microseconds separately
  (253 registers, no spills, 17,408 bytes shared) and Gemma at 73.18
  microseconds versus 80.09 microseconds separately (203 registers, no spills,
  34,816 bytes shared). These are within the observed run-to-run envelope of
  the prior 77.86 and 72.27 microsecond measurements.
- The current checkpoint passes 83 dependency tests plus 18 subtests, and 14
  loop-dependency tests with 4 expected skips. Both strict Qwen validation and
  Gemma output/cache validation pass.

### Batch scaling: preserve logical coordinates through lowering

The cold-L2 batch-size stress test exposed one deeper architectural gap. The
dependency proof retains batch and other outer axes, but configuration-time
event construction flattens those coordinates into enumerated action IDs too
early. Event lowering, physical task traversal, and worker placement then each
try to recover structure from unrelated one-dimensional tables.

For batch-two Qwen FFN, the exact W13-to-activation contribution is simply:

```text
(batch, output_tile) -> (batch, (output_tile % 768) // 8)
```

Materializing that function as 3,072 key entries and passing it through the
current one-dimensional run compressor produces 382 segments and 381
`tl.where` selections. Balancing the selection tree avoids Python parser-depth
failure, but it does not reduce generated instructions or solve the underlying
problem. The same premature flattening affects physical worker dispatch.

The placement failure is distinct. The action dependency and milestone proofs
succeed, but `place_ordered_action_consumers` can currently move only a whole
root family. At batch two the complete 1,024-task W2 family does not fit into
the useful holes in the final W13 wave. At batches eight and sixteen, the W2
family is larger than the resident worker grid and is rejected before a
partial pipeline can be considered. Compact event keys fix code size; they do
not by themselves restore overlap.

The authoritative architecture and migration sequence are specified at the
top of this document. This stress test adds two concrete acceptance criteria:

- The generated event and dispatch expressions must scale with normalized
  relation pieces rather than the 3,072 batch-two W13 tasks or the much larger
  batch-eight/sixteen domains.
- Partial placement must retain the W2 nested-loop milestones for the exact
  root subdomains moved into producer-tail capacity; compact key expressions
  alone are insufficient.

Dense, regular Qwen batching remains a static scheduling problem. Truly
data-dependent ragged MoE routing may still select `FamilyDone` or a future
runtime dispatcher, but that is independent of this coordinate-preservation
failure.

Deferred extensions, in priority order:

1. Generalize the current conservative one-nested-axis renderer only when an
   exact relation produces representable segments; unsupported cases continue
   to lift safely.
2. Extend compact logical relations beyond the initial rectilinear/div-mod
   form only when a real access requires it. Batch sixteen already makes
   enumerated nested-action relations materially expensive, so the initial
   compact representation is required; a broader symbolic relation language
   is not.
3. Expand adversarial coverage for repeated callsites, nested control flow,
   tails, and multi-axis scopes while preserving monotone fallback.
4. Continue saving and structurally comparing every Qwen and Gemma lowering,
   and benchmark on an uncontended GPU after each semantic deletion.

### Mixed-radix predecessor quotients

The DeepSeekV3 routed W13-to-SwiGLU boundary exposes an implementation gap in
the relation algebra, not a new scheduling policy. The dependency pass already
proves the exact relation from a consumer action to its producer tasks. For
consumer coordinates `(slot, activation_block)`, the two predecessor pieces
are:

```text
[256 * slot + 16 * activation_block,
 256 * slot + 16 * activation_block + 16)

[256 * slot + 16 * activation_block + 128,
 256 * slot + 16 * activation_block + 144)
```

This is a regular partition of the 2,048 producer tasks into 64 logical event
keys. Each key owns 16 gate tiles and 16 up tiles, so its fan-in is 32. The
current implementation loses this structure because `LogicalRelation.inverse`
examines each range piece independently and recognizes only intervals whose
address expression uses one source axis. It therefore cannot derive the
producer-to-key map from the combined two-piece, two-axis relation. Event
construction retains one key per producer task, the scalar consumer emitter
cannot wait on 32 independent keys, and coverage safely falls back to
`FamilyDone`.

The fix is a generic predecessor-quotient factorization. Event construction should
make the following three related maps explicit:

```text
C -> K   consumer action to semantic event key
K -> P   exact producer predecessors owned by each key
P -> K   unique event key published by each participating producer
```

Here `C` is the consumer action domain, `K` is the consumer-signature quotient,
and `P` is the producer action domain. The original dependency relation factors
through `K`; equivalently, consumer actions with the same key have the same
producer predecessor set. Disjointness and producer coverage are properties of
that factorization, not prerequisites for constructing it. Overlapping fibers
are semantically valid producer fanout to several event keys. Partial producer
coverage is also valid when the converse has a compact exact membership guard.
Specific lowering and scheduling strategies may impose stronger requirements.

For the DeepSeekV3 example, keep the semantic key multidimensional:

```text
K = (slot, activation_block)

C -> K:
    (slot, activation_block) -> (slot, activation_block)

K -> P:
    the two 16-wide ranges above

P -> K:
    producer -> (
        producer // 256,
        (producer // 16) % 8,
    )
```

The omitted producer digits are the 16 positions within a producer group and
the gate/up half. They affect fan-in, but not event identity. The key is
flattened to `slot * 8 + activation_block` only by lowering; flattened storage
must not replace the semantic `(slot, activation_block)` coordinates in the
proof.

#### Relation operation and ownership

Separate compiler discovery from relation-algebra verification:

```text
q = discover_predecessor_quotient({R_i})
F_i = R_i.factor_through(q)
```

Quotient discovery is a deterministic compiler operation over every merged
incoming relation at one consumer program point. It constructs the natural
`C -> K` map from the coordinates that can change any predecessor fiber. In
the initial axis-projection form, the key axes are the union of all source axes
used by any `R_i`. It must not discover a separate, potentially incompatible
key domain for each producer. Formally, it approximates the equivalence:

```text
c1 ~ c2  iff  for every producer endpoint i, R_i(c1) == R_i(c2)
```

The discovered `q` must be total and single-valued, and `K` must be its
represented image rather than a larger Cartesian domain containing unused
keys.

`factor_through(q)` is the generic relation-algebra operation. It proves
`R_i == q ; F_i` exactly and returns `F_i: K -> P_i`; it does not choose a
schedule. Before discovery, accesses belonging to the same
`(producer_root, producer_scope, producer_domain)` endpoint are unioned and
deduplicated. Relations with different producer domains remain separate and
factor independently through the common `q`.

Add one closed mixed-radix converse operation to `LogicalRelation`, rather
than a DeepSeek-, MoE-, dimension-, or constant-specific matcher. Given an
exact `K -> P` predecessor relation, it must:

1. Factor the predecessor relation into `K -> P`, proving that dropping the
   non-key consumer axes does not change a predecessor fiber.
2. Analyze all pieces together. Complementary gate/up ranges are one producer
   partition; they must not be inverted independently.
3. Recognize dense static mixed-radix affine layouts with any number of logical
   source axes, positive static strides, axis permutations, and a bounded union
   of contiguous ranges.
4. Construct the exact relational converse `P -> K`. It may be multivalued
   when one producer contributes to several keys and partial when some producer
   tasks do not participate.
5. Derive whether different keys have disjoint producer fibers and whether the
   producer domain is fully covered. A total disjoint quotient permits an
   unguarded single-key publication function; partial coverage requires a
   compact exact membership guard.
6. Derive producer key coordinates with ordinary floor/mod expressions. Piece
   growth must be proportional to relation/access structure, never producer or
   consumer task count.
7. Derive fan-in directly as `K -> P` fiber cardinality. This may be a symbolic
   per-key relation when boundary keys are smaller; do not reconstruct it by
   inverting `P -> K` after the fact.
8. Use `K -> P` directly for key-major producer enumeration and validation;
   use `P -> K` only where producer publication needs key expressions.

Do not introduce a persistent `RegularPartition` beside `KeyedEvent`. It would
duplicate the event key domain, uses, contributors, cardinality, and legality
facts. `KeyedEvent` is the persistent single source of truth:

```text
KeyedEvent.key_domain                 K
EventUse.keys                         C -> K
EventContribution.predecessors        K -> P
```

The publication relation `P -> K` is obtained through a dedicated cached
`publication_converse()` query on `EventContribution.predecessors`, not stored
as independent semantics and not repeatedly reconstructed by unrelated
callers. Arrival counts are derived from
`predecessors.fiber_cardinality()`. Fiber disjointness is derived from whether
the converse is single-valued. Producer coverage is queried directly from the
`K -> P` relation when possible, or conservatively through a representable
converse; an unavailable proof means "unknown," not "false." An ephemeral
relation-level factorization result may construct and validate the views
atomically, but it must not become a parallel scheduler IR.

This reverses the current persistent orientation of `EventContribution.keys`:
the authoritative relation becomes `K -> P`, which is the dependency fact the
compiler originally proved. Existing simple inversions remain valid. The new
joint-piece mixed-radix converse is a closed relation-algebra operation, not a
general symbolic equation solver.

The recognizer is deliberately not an arbitrary multivariate integer solver.
It accepts regular mixed-radix layouts whose coefficients and static domain
extents establish a unique digit decomposition. It declines when it sees:

- overlapping producer fibers or one producer contributing to several keys;
- uncovered periodic producer regions whose membership cannot be represented
  compactly;
- non-affine or data-dependent indexing and masks;
- incompatible or dynamic radices;
- tail fragments whose exact support or fan-in cannot be expressed with a
  bounded number of pieces; or
- a quotient that drops a consumer axis which actually changes predecessor
  membership.

Declining preserves the current monotonic behavior: retain a finer exact event
when its lowering is representable, otherwise use the canonical
family-completion event. An exact quotient with overlapping fibers may remain
in dependency analysis even when today's scalar publication emitter cannot
lower its producer fanout. No guessed key, fan-in, or partial continuation is
permitted.

Lowering requirements remain separate from semantic legality:

- ordinary waits permit several consumer actions to share one key;
- the current one-publication-per-producer emitter requires a single-valued
  `P -> K` relation, although bounded producer fanout remains semantically
  legal for a future emitter;
- final-arrival execution additionally requires one consumer action per key,
  equivalently a total single-valued converse of `C -> K`;
- whole-family key-major reordering requires complete producer coverage, while
  ordinary guarded publication and continuation do not inherently require it;
- local continuation may initially retain uniform fan-in, while ordinary waits
  use the existing symbolic per-key arrival counts.

#### Migration and acceptance criteria

1. Add relation-level tests for the supplied `(8, 4096)` example. Verify the
   exact `K -> P` fibers, `P -> K` key map, total/disjoint quotient proof, and
   constant fan-in 32.
2. Split event construction into joint `C -> K` discovery and per-endpoint
   `R_i.factor_through(q)` verification. Then derive publication through the
   cached mixed-radix converse. Remove the current
   `relation.inverse().project_target(key_domain)` dependency from this path.
3. Change `EventContribution` to retain only the authoritative `K -> P`
   predecessor relation. Route fan-in, key-major ordering, and small-domain
   validation through it; derive and cache `P -> K` for publication. Do not
   duplicate relation semantics in scheduler-specific tables.
4. Keep the current generic inverse for other relation operations. Extend it
   only with closed mixed-radix rewrites that are independently useful; do not
   make arbitrary symbolic inversion a prerequisite for event formation.
5. Verify that slot counts 1, 2, 8, and 64 produce constant relation-piece and
   generated-code complexity. Vary producer and activation block sizes where
   the same regular partition exists.
6. Add tests showing that overlap and partial support remain semantically
   legal when representable, but strategies requiring single publication or
   full-family reordering decline them. A single gate/up half with periodic
   holes, incompatible tails, and non-affine indexing must reject or coarsen
   safely when no compact converse is available.
7. Add nested-scope and multi-producer-join variants, and verify that batch,
   expert, head, and other outer coordinates remain semantic key axes while L2
   traversal remains purely physical.
8. Run the end-to-end reproducer and require 64 event keys, fan-in 32,
   final-arrival execution, no W13-to-SwiGLU root-completion edge, and no
   task-count-sized `tl.where` cascade. Check repeated-launch correctness.
9. Revalidate Qwen3, Gemma4 E4B, Gemma4 A4B MoE, and full DeepSeekV3 MoE
   lowering, resource use, cold-L2 latency, and SM overlap. The new relation
   operation must not select a different schedule when the existing proof was
   already sufficient.

#### Implementation checkpoint (2026-08-25)

The predecessor-quotient migration is complete for the supplied mixed-radix
failure:

- `EventContribution.predecessors: K -> P` is now the sole stored producer
  dependency relation. Fan-in and key-major enumeration use it directly;
  `P -> K` is a cached derived publication converse.
- Event construction discovers one consumer quotient from all incoming
  producer endpoints, factors every endpoint independently through it, and
  unions/deduplicates accesses from the same producer root and execution
  scope before deriving cardinality.
- The mixed-radix converse analyzes all range pieces jointly and recovers the
  exact `(slot, activation_block)` key for the routed W13-to-SwiGLU example.
  Its current closed-form recognizer covers multidimensional key domains over
  a one-dimensional producer task domain; higher-rank producer domains remain
  on the existing exact fallback until an independently useful relational
  rewrite is justified.
  The end-to-end reproducer lowers to 64 keys with fan-in 32, executes the
  consumer on the final arrival, emits one structural `tl.where`, and has no
  root-completion wait on that boundary.
- Unsupported overlap, periodic partial support, and nonrepresentable tails
  remain exact in the tile-dependency relation. They do not enter the emitted
  event candidate set unless a supported publication strategy exists; the
  scheduler instead retains the finer producer-key fallback or root
  completion.
- Nested milestone coarsening remains a derived scheduling operation. It starts
  from the cached exact converse of the authoritative predecessor relation,
  composes the stage map, and inverts the exact result back to `K_stage -> P`.
  This preserves Gemma's existing 36/4 down-projection milestone split without
  introducing general range-image machinery into `LogicalRelation.then()`.
- Worker-count snapping now ignores semantic frontiers that cannot be emitted
  by the counted-event lowering.

The implementation was reviewed twice. The final review approved the
factorization and single-source-of-truth design and rejected a broader general
range-composition extension as unnecessary. It also identified an existing
large-batch compile-time hotspot: total-function checking repeatedly
canonicalized thousands of already-disjoint L2 traversal pieces. A generic
fast proof now recognizes a partition of the source domain with one valid
target point per piece, reducing the check from a quadratic cell-by-piece scan
to a sweep over the existing partition. Canonical single-valued forms are also
cached per immutable relation.

Validation results:

- The dependency, scheduler, and codegen coverage is split by compiler module:
  115 focused tests and 31 subtests pass. Together with config and legacy loop
  dependency coverage, 191 tests pass, 4 skip, and 55 subtests pass.
- Qwen lowering completes in 13.6 seconds at batch 1 and 14.3 seconds at batch
  8 in fresh processes. Stress cases batch 64 and 128 complete in 18.1 and
  19.3 seconds with the granular source; no multi-minute compile remains.
- The normalized Qwen, Gemma E4B, and Gemma A4B lowered Triton sources are
  byte-identical to commit `572f3c62`, where their earlier dependency proofs
  already succeeded.
- Cold-L2 B200 measurements remain approximately 89.8 microseconds versus
  97.9 for the granular Qwen layer, 105.1 versus 91.4 for Gemma E4B, and 48.0
  versus 53.7 for the vLLM-boundary-preserving Gemma A4B MoE source. The
  155.7-microsecond Qwen result is the intentionally opaque fused-RMS stress
  source, which spills 456 bytes; it is not the performance source.

### 2026-08-28 lowering ownership and duplication audit

- [x] Move cross-loop-only AST staging, relation rendering, synchronization,
  state allocation, traversal binding, and outlining helpers out of
  `program_id.py` and into `cross_loop_codegen.py`. `program_id.py` now retains
  the generic persistent-loop implementation and one cross-loop dispatch hook;
  its branch delta is 26 additions and 6 deletions relative to `main`.
- [x] Compute configured root geometry once in the emitter. Task counts,
  offsets, root-domain geometry, and physical traversals are derived from the
  same local tuple; nested-only axes are filled once from their block specs.
- [x] Remove an empty publication path and a tautological local AST check.
  Make the small-test arrival-count oracle enumerate predecessor fibers rather
  than invoking the production cardinality proof it is meant to check.
- [x] Simplify `WorkerScheduleSegment` to its production relation-based task
  ordering. The test-only scalar/periodic task-order language and its dead
  lowering branches are gone; a segment now contains one ordinal-to-task
  relation, one worker interval, and one dense schedule offset. This removed
  159 net production lines. The 191-test compiler battery passes, affine-chain
  lowering remains compact (0 `tl.where`), and cold-L2 Qwen, Gemma A4B, and
  GPT-OSS probes retain correct lowering and representative performance. The
  GPT-OSS integer-domain validation at multiplier 12 measures 35.68 us versus
  37.07 us for the separate Helion graph; multiplier 8 remains the known
  slower 42-us point.
- [x] Remove duplicate uniform-fan-in calculations at their source instead of
  broadly caching relation queries. `CountedEventPlan.uniform_arrivals()` is
  the sole derivation, and codegen evaluates it once for each retained plan;
  speculative nested-placement candidates do no fan-in work. Its key domain
  is likewise derived from the contributor relations instead of stored a
  second time. Generic `LogicalRelation` queries remain uncached until
  profiling demonstrates an independent need.
- [x] Remove copied edge summaries from `TileDependency`. Hazard kinds now
  derive from the authoritative `AccessDependency` records, and the unused
  producer/consumer access tuples are gone. Remove test-only event lookup and
  counted-event convenience APIs rather than maintaining parallel query
  surfaces in production.
- [x] Stop reconstructing root-completion edges in codegen. Family-completion
  events remain part of semantic analysis, but the final schedule carries the
  selected root-completion edges directly rather than manufacturing counted
  events that codegen must decode again.
- [x] Remove the exact task-placement oracle from production. Static placement
  queries now ask only which root occupies a worker-stream position; exhaustive
  task inversion and materialization live in `test/_cross_loop_schedule_oracle.py`.
- [x] Bind configured domains once. `instantiate_logical_domains()` creates one
  scope-indexed table and root families reuse their root-scope objects. Codegen
  passes those same domains through dependency instantiation and scheduling;
  the scheduler no longer reconstructs them from `axis_geometry`.
- [x] Remove duplicated event identity and domain state. `KeyedEvent` derives
  its key domain from its contributor relation and its ID from that domain;
  `EventGraph` derives root domains from root traversals; `CountedEventPlan`
  likewise derives its key domain. Event-local IDs are assigned when an event
  candidate first enters the canonical table, eliminating the later relation
  retargeting pass.
- [x] Share mixed-radix fiber analysis at its source. Each event contribution
  computes cardinality once and passes it into publication-converse discovery,
  rather than rerunning the same source-cell proof for arrival counts.
- [x] Share event and readiness-derived state rather than copying it. Semantic
  and lowering events now use the same `contributions` vocabulary and one key-
  domain validator. Event IDs, key domains, and root domains are derived from
  their owning relations/traversals. Completion-frontier analysis computes the
  local-trigger index, static contributors, total-support proof, and transitive
  prerequisite roots once per query instead of rebuilding them in each caller.
- [x] Make the canonical Qwen benchmark configuration the probe default. The
  raw Helion default selects the barrier schedule and measures roughly 4 ms,
  not the static-pipeline megakernel. Use `--default-config` only for that
  explicit ablation; normal Qwen validation no longer depends on remembering
  `--probe-config`.

Current post-cleanup cold-L2 checks retain the intended schedules: the granular
Qwen3 layer is 93.38 us versus 104.48 us separate, Gemma4 A4B is 47.58 us
versus 53.60 us separate, and GPT-OSS at the separately validated multiplier
12 point is 35.06 us versus 36.93 us separate. The GPT-OSS lowering matches
the prior 1,776-worker static schedule after normalizing the now-symbolic
`_NUM_SM * 12` launch expression.

Relative to the preceding cleanup commit `7867e5a1`, this round currently
removes 517 net production lines across `tile_dependency.py`,
`cross_loop_scheduler.py`, and `cross_loop_codegen.py`. The larger suggested
event/placement merger is intentionally left as a follow-up: the remaining
semantic-event and selected-lowering objects represent genuinely different
compiler phases, and collapsing them now would trade a small amount of surface
area for less explicit ownership.
- [ ] Consider unifying `KeyedEvent`, `CountedEventPlan`, local triggers, and
  root-completion publication into one final lowering-event representation.
  This is a larger follow-up because it changes ownership across scheduling
  and lowering.
- [x] Centralize the existing L2 applicability rule and ordinary coordinate
  flattening used by traversal construction and lowering. This removes three
  independently maintained no-op checks and three copies of row-major
  flattening without changing the emitted expressions.
- [x] Use the DeviceIR `TaskFamily` as the sole source of top-level axis
  extents during static-schedule admission. A missing family or axis now makes
  that configuration invalid instead of silently falling back to the lowered
  `PIDInfo` geometry and maintaining two potentially divergent coordinate
  descriptions.
- [x] Derive each root's active worker support from the tasks actually present
  in every cyclic schedule segment, not from the segment's maximum worker
  capacity. This prevents short or reordered families from over-counting
  root-completion arrivals when `task_count < worker_count`.
- [x] Remove configured scope domains from `EventGraph`. Each producer and
  consumer relation already owns its exact scope domain, so retaining the
  complete scope table duplicated identity and geometry after event
  construction. Exhaustive source-order and strand-projection helpers now
  live only in the test oracle.
- [x] Centralize dense root-assignment derivation in `WorkerSchedule`. Scheduler
  interval queries and codegen now share one linear validation rather than
  independently sorting segments and rebuilding prefix sums. Derive root task
  counts and case offsets from the instantiated root domains, and consume
  uniform fan-in directly from each plan instead of maintaining a second
  codegen dictionary.
- [x] Move positional coordinate renaming into `LogicalRelation`. Event
  canonicalization now uses the same general source/target axis-isomorphism
  operations instead of maintaining separate scheduler-local rewrites. Source
  renaming substitutes symbols simultaneously; target renaming changes only
  codomain labels, and both reject positionally incompatible shapes.
- [x] Perform counted-lowering admission after assigning canonical event-local
  coordinates and identity. The accepted `EventContribution` objects now own
  the proof reused by trigger selection and lowering instead of proving the
  pre-canonical relation and then repeating the work on a rebuilt event. In a
  profiled Qwen lowering this reduced readiness analyses from 43 to 25 and
  cross-loop emission from 4.91 s to 4.31 s without adding a scheduler cache.
- [ ] Replace the remaining symbolic and hand-written L2 forward/inverse
  formulas only after introducing a compact bidirectional traversal relation.
  A one-piece forward relation is renderable, but its `Min`/`Mod` inverse is
  not representable by the current relation grammar; using it now would disable
  completion-frontier proofs or expand back into a `tl.where` cascade. Retain
  the differential tail/permutation tests until that representation exists.
