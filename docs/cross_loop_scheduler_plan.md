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

### Current checkpoint

The graph-derived scheduler now lowers the Qwen attention and FFN components
and the Gemma FFN component with canonical logical task coordinates, including
roots whose physical traversal uses L2 grouping. Region-aware reaching
definitions preserve partially overwritten allocation regions, and keyed
events support multiple producer roots, repeated consumer predecessor sets,
counted last-arrival actions, and readiness frontiers. There is no model-name,
root-number, or flattened-task-ID matcher in that path; the custom partitioned
attention scheduler has been deleted.

Fresh cache-bypassed B200 measurements are 76.09 microseconds for Qwen versus
86.27 microseconds for its separate-kernel baseline, and 74.09 microseconds for
the tuned Gemma sliding non-shared layer versus 78.87 microseconds separately.
The Qwen kernel uses 251 registers, no spills, and 17,408 bytes of shared
memory. The Gemma kernel uses 204 registers, no spills, and 34,816 bytes of
shared memory. Qwen also matches the retained custom scheduler within noise:
76.08 versus 76.16 microseconds in adjacent fresh runs.

The main dependency-IR migration is complete. Dynamic task counts use the
existing cooperative phase fallback, while large static task domains are
planned without a task-product cutoff: region-derived ready groups use a
sorted allocation-interval sweep rather than a producer-by-consumer Cartesian
scan. Remaining work is validation and cleanup: broaden shape coverage,
replace the fixed-range frontier tuning domain if the autotuner API permits it,
and keep one-wave reduction replication clearly separate from dependency
scheduling. Gemma's shared-KV variants remain a root-codegen/resource-envelope
problem: their current fused kernels use roughly 70 KiB of shared memory and
support only two resident CTAs per SM, so no legal FFN frontier fits. Do not
add dependency-scheduler cases to disguise that limitation.

### Holistic keyed-event refactor

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

### Current migration boundary

The dependency graph now owns ordinary root-entry readiness, exact task
readiness, generic multi-producer joins, repeated-predecessor ready groups,
nested-loop access cohorts, and last-arrival continuations. The old
flattened-ID grouped-continuation, ordered-input-singleton, and partitioned
attention matchers and their emitters have been deleted. The Qwen QKV/QK/cache
join, attention admission, two merge levels, quantization handoff, and FFN all
use the same keyed-event and root-dispatch machinery.

Only one structural emitter remains: one-wave reduction fanout. It consumes
the unified allocation-based dependency graph for legality but deliberately
replicates computation, so it is not a dependency protocol and should remain a
separate optimization pass.

The review classifies that remaining path as follows:

- One-wave reduction fanout deliberately replicates an opaque singleton
  computation in every consumer CTA. That is an optional computation-
  replication transform, not a dependency-scheduling primitive. Keep it
  outside the generic scheduler until DeviceIR can prove the root is pure and
  idempotent; do not teach the dependency graph about reductions by inspecting
  lowered Triton.
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

### Next execution plan

Completed in the current implementation:

1. Compiler-generated legal frontier candidates derive `G`, exact `W`,
   consumer-major producer traversal, the tail-producer cohort, and the
   one-task-per-consumer cohort.
2. An internal frontier fragment selects among those candidates. The launch
   uses the exact derived grid and post-compilation residency rejects an
   oversized candidate; Gemma's 608-worker candidate was correctly rejected at
   a measured capacity of 592. `num_sm_multiplier` is not part of this choice.
3. Logical event identity is independent of physical loop order and L2
   grouping. Batch and other outer logical axes remain in readiness keys.
4. The counted-event last-arrival operation sequence was validated in Triton
   probes and is active in compiler-generated Qwen and Gemma lowerings.
5. Generic multi-producer root joins no longer force exact per-task polling.
   Gemma showed why this matters: exact access-local events measured about
   96.57 us, while the root-join policy recovered approximately 73.69 us in the
   timing-only run.
6. Every structural-plan entry accepts the same ordered set of upstream
   root-completion dependencies. This removes the former singular-input
   assumption without adding topology cases.
7. Singleton producer task events are canonicalized to root completion. A
   one-task event and root completion are identical, and retaining only one
   representation avoids duplicate counters and preserves compact lowering.

Remaining, in order:

1. Finish separating one-wave reduction replication from dependency
   scheduling. Treat
   it as an independently justified pure-computation transform rather than a
   dependency primitive.
2. Canonicalize ordinary source views where that is performance-neutral and
   express reduction intervals as tile ranges. Preserve additional affine
   logical-coordinate expressions only when source-level tiles cannot express
   a demonstrated relation. Do not build a general reshape/in-place region
   solver without a second demonstrated need. Specialized scalar quotient task
   extents, including Gemma's natural `q_heads // kv_heads`, are already
   preserved.
3. Retain the provisional bounded frontier-index fragment for now. The current
   fragment API is fixed when `ConfigSpec` is built, while the exact legal
   candidate count depends on the selected block sizes and is known only during
   lowering. Invalid indices already fail as invalid configurations. Revisit a
   dependent fragment only as a general autotuner capability; do not couple
   schedule discovery back into DeviceIR construction for this feature alone.
4. Expand full-layer shape testing and rerun Qwen/Gemma correctness and
   performance on uncontended GPUs after each structural deletion gate.
5. Continue broadening the compact relation representation only when a real
   workload needs it. Dynamic task domains already use the cooperative phase
   fallback. Large static domains no longer use a hardcoded task-product
   cutoff: the region proof sorts task address intervals and visits only actual
   overlap candidates. Unknown address intervals still select root completion.

Do not add a strategy cost model. The Triton Gemma probe showed that logical
task identity fixes continuation legality but not profitability by itself:
root completion was about 85.52 us, consumer-major continuation about 87.22 us,
an aligned 416-worker frontier about 81.07 us, a 512-worker frontier about
80.82 us, and the aligned 576-worker frontier about 75.77 us in the same
experiment. The compiler proves the small legal frontier family and the
autotuner measures only its candidate index.

The compiler must continue to have one general scheduler. The tuning surface
belongs to an individually proven dependency edge or interacting graph
component; it must not select named Qwen, Gemma, FFN, or attention schedules.
The compiler derives event keys, exact counts, legal split alignment, residency,
and progress constraints automatically, and the autotuner measures only the
remaining legal choices.

### Current performance evidence

- The final generic graph-derived Qwen schedule is 75.82 microseconds versus
  86.17 microseconds for separate kernels in the latest fresh paired run, with
  251 registers, zero spills, and 17,408 bytes of shared memory. A fresh run of
  the retained custom scheduler measured 76.16 microseconds, so the generic
  replacement is at parity within noise. Earlier software/environment snapshots
  measured the same designs near 74--75 microseconds.
- The final tuned Gemma sliding non-shared schedule is 74.02 microseconds versus
  78.98 microseconds separately, with 204 registers, zero spills, and 34,816
  bytes of shared memory. Eliminating an exact task wait already dominated by a
  whole-root dependency path recovered about 2 microseconds without changing
  any scheduling policy.
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
- Exact launch grids derived from frontier candidates, independent of
  `num_sm_multiplier`.
- Insufficient-residency configuration rejection.
- Candidate-specific rejection when changing the frontier changes compiled
  registers or shared memory.
- Root-completion fallback when every fine-grained frontier exceeds compiled
  resident capacity.
- Zero-task roots and symbolic task counts.

### Performance checks

- Qwen3 full-layer megakernel versus separate kernels.
- FFN subgraph at batch 1, 2, and 4.
- Synchronization overhead on simple chains.
- Register count and spill count.
- Generated-code audit confirming unchanged root computation.
- Lowered-Triton comparison proving that counted-event `on_ready` continuation
  preserves the existing fast producer order, last-arrival test, consumer call,
  fences, and downstream publication.
- Frontier sweeps that include the Qwen first legal point and all aligned Gemma
  points between the one-tail-wave lower bound and compiled residency limit.

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
- Adopted the holistic keyed-event refactor above. The next implementation
  step is multi-contributor event IR plus region-aware reaching definitions,
  followed by reconstruction of the Qwen attention pipeline and deletion of
  `_match_partitioned_dependency_pipeline`.
