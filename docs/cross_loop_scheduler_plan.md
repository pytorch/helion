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

## Recommended architecture

```text
DeviceIR roots
    ↓
Memory accesses grouped by allocation
    ↓
Producer → consumer dependency edges
    ↓
Proven readiness mapping per edge
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

RAW edges are candidates for fine-grained readiness. WAW and WAR edges should
initially use conservative root-completion ordering. Unsupported side effects,
atomics, or source-level conflicts that cannot be reconciled with DeviceIR facts
must remain errors rather than being guessed away.

Dependency discovery must be based on allocation identity, not only source
variable names. Source-level dependency analysis remains useful for diagnostics
and for detecting effects that the memory-fact layer cannot represent.

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
3. **Policy** chooses among the legal schedules. Initially this should be a
   simple deterministic choice. The autotuner may later choose among a small
   set of already-proven alternatives.

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

### 4. Represent scheduling with generic events and waits

The plan should contain only generic scheduling concepts:

- An event identifies a producer root and a readiness domain, such as task or
  root completion.
- A wait identifies a consumer root/access and the producer readiness keys it
  requires.
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
consumer-major order makes all producers for one key arrive close together and
allows useful work to continue early. Native producer order remains correct,
but may delay every continuation until nearly the entire producer root has
finished.

### 7b. What may be autotuned

The autotuner may choose only among schedules whose legality is already proven.
Reasonable internal candidates are:

- exact task events versus a proven partition continuation;
- native versus consumer-major producer traversal when both lower compactly;
- the number and boundaries of proven nested-loop readiness cohorts; and
- existing worker-count or occupancy choices subject to full-residency checks.

The autotuner must not choose dependency membership, readiness-key dimensions,
fan-in independently of publication grouping, fence placement, or whether an
unproved mapping is considered safe.

Do not add these knobs preemptively. First implement the smallest deterministic
partition continuation and measure it. Add a bounded internal tuning choice
only when two general, legal strategies have a meaningful workload-dependent
tradeoff.

### 8. Keep the public interface small

The durable public contract should be an opt-in cross-loop/tile-dependency
schedule, plus explicit barriers where the user requires phase ordering.

Epoch replication and producer order remain migration-time implementation
details. Remove or internalize them when the remaining structural schedulers no
longer require controlled experiments. The obsolete stage-count and
continuation-split parameters have already been removed.

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
   edges and sets of waits/publications, not topology-specific cases.
7. Every waiting launch configuration satisfies the full-residency progress
   requirement.
8. Launch epochs and stream-local state prevent cross-launch and cross-stream
   readiness reuse.
9. Disabling the schedule leaves ordinary Helion codegen unchanged.
10. Enabling the schedule changes root bodies only by inserting the required
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
- Stable access IDs.
- `EventSpec`, `WaitSpec`, and affine predecessor-map concepts.
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

## Implementation plan

Status values: **pending**, **in progress**, **complete**, or **blocked**.

| Phase | Status | Work | Exit criterion |
| --- | --- | --- | --- |
| 0 | complete | Record the architecture and migration plan. | This living document exists and is kept current. |
| 1 | complete | Add scheduler-specific DeviceIR access facts and an allocation-based dependency graph without changing emitted kernels. | Unit tests inspect correct RAW/WAR/WAW edges, multidimensional keys, and conservative fallback. |
| 2 | in progress | Make the edge proof gate all existing fine-grained schedules. | The reversed-group adversarial kernel is correct by using a proven reversed map or safe root fallback; Qwen still takes a valid fast path. |
| 3 | complete | Add generic root-ready and task-ready events to the current static persistent traversal. | Simple chains, fanout, joins, unequal tiles, and Cartesian grids use one planner/codegen path. |
| 4 | complete | Support readiness coordinates introduced by existing nested loops and access-aware wait placement. | Nested-loop dependencies are correct without changing computation codegen. |
| 5 | complete | Add two narrow optimizations over exact readiness: nested-loop cohorts and uniform-partition last-arrival continuations. | Qwen's FFN chain is represented without a topology matcher; failures fall back to exact events or root completion. |
| 6 | in progress | Remove specialized matcher/emitter families and internalize or remove specialized public knobs. | Planner and codegen contain only root/task/access/event abstractions. |
| 7 | pending | Audit and separate unrelated codegen changes; handle dynamic task counts without assertions. | Schedule-off codegen is unchanged, and dynamic-shape scheduling selects a valid symbolic path or safe fallback. |
| 8 | pending | Final correctness, performance, lint, and design review. | Test matrix passes; Qwen meets the agreed performance range; remaining limitations are explicit and structural. |

### Current migration boundary

The dependency graph now owns ordinary root-entry readiness, nested-loop access
cohorts, and the FFN producer-to-map continuation. The old flattened-ID grouped
continuation matcher and its emitter have been deleted. Remaining structural
emitters cover the attention partition pipeline, ordered singleton inputs, and
reduction fanout. They are compatibility shims during migration, not templates
for additional matcher families.

### Immediate next steps after the generic FFN continuation

1. Audit each remaining structural planner against the dependency graph and
   make explicit which additional schedule property it proves beyond edge
   legality.
2. Replace the ordered-input singleton and reduction-fanout matchers with small
   graph-derived composition primitives where that makes the implementation
   simpler; retain conservative root completion otherwise.
3. Isolate the attention partition pipeline's useful primitives before trying
   to replace it wholesale. Do not force its reduction topology into the
   uniform-partition continuation abstraction.
4. Remove or internalize migration-only policy knobs after their remaining
   users disappear. Do not introduce new knobs without measured evidence of
   two materially different legal schedules.
5. Expand shape testing to shorter contexts, changed intermediate widths,
   partial tiles, and dynamic task counts, keeping unsupported relations on the
   exact-event or root-completion paths.
6. Audit schedule-off generated code and remove dead access-wait experiments or
   helpers that no longer serve a fallback path.

### Current performance evidence

- The generic graph-derived Qwen schedule is approximately 74.9 microseconds
  versus approximately 80.4 microseconds for separate kernels on the current
  setup, with 252 registers and zero spills.
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

These results validate the narrow generic continuation primitive. They do not
justify reproducing every specialized ordering or staging detail from the
probe.

## Validation matrix

### Dependency and legality tests

- Identity producer/consumer mapping.
- Reversed consumer group order.
- Offset and sliced views.
- Unequal producer and consumer tile sizes.
- Two-dimensional and higher-rank Cartesian grids.
- Multiple producer stores to one allocation.
- Multiple consumer loads from one allocation.
- Fanout, joins, and longer chains.
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
- Insufficient-residency configuration rejection.
- Zero-task roots and symbolic task counts.

### Performance checks

- Qwen3 full-layer megakernel versus separate kernels.
- FFN subgraph at batch 1, 2, and 4.
- Synchronization overhead on simple chains.
- Register count and spill count.
- Generated-code audit confirming unchanged root computation.

Performance is a required constraint, but correctness proofs may not be weakened
to recover a few microseconds. Prefer removing overhead from the generic event
representation or traversal over adding a model-specific fast path.

## Known evidence and current baseline

As of 2026-08-20:

- The current full Qwen3 granular schedule runs at approximately 75
  microseconds versus approximately 81 microseconds for separate kernels on the
  existing test setup.
- Shorter context configurations tested so far remain correct and faster than
  the corresponding separate-kernel runs.
- The generic FFN continuation is selected for batch sizes 1, 2, and 4 in the
  focused grouped-chain test. Batch remains part of the readiness key.
- A reversed activation-group consumer now declines continuation and uses safe
  root completion. The unsafe topology/count matcher that previously accepted
  it has been deleted.
- A repeated-launch end-to-end test exercises fan-in 3, confirming that the
  epoch-relative exact arrival target does not require power-of-two fan-in.
- Explicit `static_shapes=False` scheduling now selects the existing safe
  cooperative phase-barrier fallback when task counts are unavailable.

These observations motivate the edge-based design and define regressions that
the migration must prevent.

## Design decisions and open questions

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
- Use deterministic cohort boundaries and worker geometry for now; measured
  performance does not justify an autotuner choice.
- Remove the now-unused `tile_dependency_stages` and `continuation_split`
  policy parameters.

Open, to be answered with implementation evidence:

- How to express the remaining attention partition and reduction schedules as
  compositions of graph-derived primitives without losing performance.
- Whether `producer_order` and `epoch_replicas` can be internalized after those
  remaining structural schedulers are migrated.
- How to avoid compile-time enumeration for very large static task domains
  without introducing a broad symbolic-set solver.

## Progress log

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
