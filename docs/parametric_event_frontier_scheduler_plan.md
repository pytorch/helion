# Unified parametric event-frontier scheduler

## Status

This is the end-state plan and implementation ledger for Helion's cross-loop
scheduler. It incorporates the experiments from FlashMLA, Qwen3 decode, Gemma
4 A4B MoE, DeepSeek-V3 MoE, Nemotron MoE, and Muse/Glimmer FFN.

Implementation checkpoint (2026-09-09):

- Architecture review is reopened. The current tree has one dependency and
  schedule representation, but it does **not** yet have one scheduling policy:
  `build_static_pipeline_plan` still chooses different continuation, counter,
  barrier, list-scheduling, and transient-source behavior according to whether
  a root domain has parameter symbols. The redesign is not complete until that
  policy split is removed.
- A source-identical Qwen ablation now proves that the old concrete scheduler's
  useful behavior is not just counter granularity. The historical schedule
  places complementary producer and consumer cohorts in the same wave and
  derives a two-key nested frontier with fan-ins 74 and 22. Collapsing only
  the frontier costs about 2 us; removing that placement as well costs about
  6.2 us and reproduces the current regression. Both facts must be recovered
  by the unified symbolic policy, with neither constant encoded in policy.
- A controlled Gemma continuation ablation disproves the earlier assumption
  that symbolic lowering should copy the exact-shape continuation choice.
  With identical R126/zero-spill/34,816-byte-shared/W4 resources, keeping the
  expert reduction resident measures 61.408/77.792 us at B1/B2; forcing only
  that reduction onto the final producer measures 63.568/83.936 us, and the
  full reduction/post-norm chain measures 65.472/84.064 us. The packed
  resident schedule already places reductions on otherwise-idle lanes in the
  producer tail wave. Continuation ownership and resident placement must
  therefore be compared atomically; continuation is not an unconditional
  optimization.
- The tuned one-cubin Gemma schedule at multiplier three measures
  53.216/75.712 us at B1/B2 versus 51.168/71.712 us for the matched exact
  packed schedule and 55.264/69.888 us standalone. This recovers the worker-
  width loss and isolates 2.048/4.000 us of symbolic/state rendering overhead.
  The historical exact B1 control remains 49.120 us. Preserving that exact
  case and removing generic rendering overhead are separate exit gates.
- Concrete schedule ownership is normalized into the existing
  `WorkerScheduleSegment.task_order` relation.
- Concrete event-frontier proposal and symbolic ownership/progress proofs are
  implemented without a production CTA DAG.
- Independent schedules bypass proposal, and a same-coverage resident proposal
  is retained only when its final occupied unit-task wave does not regress.
  This is part of the explicit unit-weight structural policy, not a measured
  latency estimate.
- Transient source work is represented by an ordinary
  `WorkerScheduleSegment` at launch stage zero. Its relation is now the sole
  authority for source ticket order, task count, external readiness frontiers,
  and codegen mapping; resident work occupies launch stage one.
- Parameterized multi-axis roots now lower through the same `WorkerSchedule`,
  conservative root barriers, and runtime-bounded loops, with one cubin reused
  across changing extents. PID-axis order remains an exact existing
  `CoordinateRelation`, not a second schedule representation.
- `TileAccess` now retains canonical symbolic shape, stride, and storage-offset
  expressions. Backed shape parameters remain exact; unbacked values and
  unsupported indirect access relations decline independently without using
  size hints as correctness facts.
- Exact parameterized root-entry readiness supports rank-one positional
  fan-in one and a proved fixed-width producer partition whose one consumer
  per key executes as the existing final-arrival continuation. Parameterized
  counters use one 64-bit epoch-framed state allocation across changing
  extents and counter-section offsets.
- A root-level fixed-width readiness event may now subsume equivalent accesses
  inside fixed-trip nested loops. The proof projects only statically bounded
  inner axes, retains runtime outer axes symbolically, and requires the root
  event to cover every nested memory obligation before removing nested waits.
  Fixed-capacity scratch is accepted only when the existing shape environment
  proves its dominating source guard; unknown bounds conservatively decline.
  A dynamic-batch Q1/H16 MLA probe reuses one cubin across B=1,2,4,9 and lowers
  its nested C4 reducer as an exact fan-in-16 final-arrival continuation.
- Parameterized root-entry counters now also support an exact repeated-fiber
  fan-out: `F*K` producer tasks publish to `K` keys and `C*K` ordinary
  consumers acquire `floor(c/C)`. Direct unshifted `tile.id // C` flattened
  accesses remain in the existing relation algebra; integer rounding,
  bounded modulo elimination, adjacent target coalescing, and proved
  out-of-domain pruning keep the dependency compact. A production-shaped
  F64/C16 MLA probe lowers with no root barrier or continuation and reuses one
  cubin across B=1,2,4,9.
- The former equal-size parametric event-frontier recurrence has been removed.
  It occupied `L*ceil(N/W)` waves, whereas the existing packed root-major
  relation occupies `ceil(L*N/W)` for the same work; its separate recognizer
  and renderer therefore added policy surface without a proved benefit.
  Event-aware locality must now be proposed by the one scheduler and certified
  from the accepted segment relations and readiness plans.
- Exact bounded nonuniform fan-in now lowers through per-key arrival-count
  expressions and a proved maximum epoch stride. Data-dependent ragged fan-in,
  masked producer publication, unequal non-continuation recurrences, and the
  remaining cross-workload rollout still require work.

## Active implementation checklist

This is the ordered handoff list after the 2026-09-09 interruption. Complete
Priority 1 before resuming scheduler-policy or performance work. A checked box
means the code exists and its focused unit tests pass; it does not replace the
cross-workload exit gates below.

### Current implementation marker: close the symbolic proof layer

Commits through `7e7e44ac` are the current compiler checkpoint. The existing immutable
`CoordinateRelation` remains the only schedule/dependency truth. It now retains
derived exact converses through proved construction and transformation,
represents ragged grouped L2 order with one forward piece and a two-piece
full-group/tail inverse, recovers those proofs after deepcopy/pickle, supports
multiple uniquely ordered symbolic outer radix axes, and validates concrete
and symbolic `WorkerSchedule`s through the same ownership/disjointness path.
Ambiguous symbolic radix order declines conservatively.

The old factorial mixed-radix inverse search has also been removed. One shared
structural recognizer infers `(stride, radix)` digits, verifies the unique
complete chain, and declines malformed or ambiguous chains. A six-digit case
that exceeded 60 seconds now proves in about 0.12 seconds. The representative
`[5, 3, B]`, group-size-2 schedule after an unaligned prefix builds in about
7.1 seconds rather than 15.9 seconds; no runtime extent is enumerated.

Phase 4A.2 has started without waiting for policy redesign: event construction
now always records the finest exact semantic event, producer-set quotienting is
derived later only as a counter-lowering strength reduction, and the compiler
retains one `ReadinessGraph`. A reviewed correction removed raw nested-
obligation exclusions from readiness traversal because an uncommitted counter
must not erase a semantic prerequisite. Continuation candidates are derived
once from the graph; ownership selection remains deliberately unchanged. One
parameter-independent, root-domain-aware predicate now validates exact counter
publication, consumer shape, and continuation shape. Temporary parameterized
renderer limits and plan-wide epoch bounds remain explicitly separate from
semantic legality and are applied before a plan is frozen.

Symbolic traversal helpers now consume `size_expr` and
`axis_count_expressions`, support exact empty traversals, and validate legacy
scalar dispatch metadata against an ordinalization of the authoritative
schedule relation. Unsupported symbolic offsets decline cleanly rather than
falling back to concrete counts.

The flat-converse attempt is now cached once per relation instance and the
packed `[5, 3, B, Q]` unaligned-prefix regression passes for direct,
deepcopy/pickle, and either-axis-zero cases. Its schedule build is about 2.6
seconds and cache-free proof recovery is about 5.5 seconds. The transform audit
now covers rename, reorder, projection, lift, coalescing, substitution, and
copying. The integrated Phase 4A.1 suites pass 142 tile/ragged tests with 216
subtests and 154 exact/scheduler tests with 78 subtests. Phase 4A.1 is closed.

The first Phase 4A.2/4A.4 prerequisites are also complete. Final-arrival
candidates now validate the canonical lowered event with the same exact
counter predicate used by plan validation. A quotient-lowered continuation is
recovered through the event identity already carried by its readiness-key
domain and accepted only after exact equality with that event's canonical
lowering; no second candidate identity or filtered graph is retained.

Dense task-order slicing now accepts proved symbolic begin/count expressions by
composing a translated ordinal relation with the existing flat task-order
relation. It retains exact converses, remains one piece for the representative
unaligned `B*Q` prefix, handles substitutions to zero, and leaves the bounded
concrete/manual implementation as a compatibility fallback. The frontier
extremum operation now proves the Qwen mixed-radix permutation followed by the
fixed-width worker quotient without enumerating `B` or its 96 nested
iterations. Every returned extremum carries a local attaining-corner witness;
possibly empty symbolic producer fibers decline, and a per-query memo bounds
nested-expression work by expression-DAG size. The resulting frontier remains
the derived 74/22 partition with neither value encoded in scheduler policy.
Independent review signs off on these proof primitives after adversarial
correlated-expression, runtime-zero, wrapping-modulo, and positive-stride
checks. The current full tile suite passes 136 tests with 57 subtests.

Schedule-derived nested counter partitions now retain their exact inverse at
the point where full coverage, ordered positive segments, and symbolic endpoint
identities are proved. Existing generic converses remain canonical whenever
they are representable; the retained partition proof is used only for an
individual producer arm whose ordinary converse declines. This preserves the
historical concrete 3/1 normalization while allowing symbolic outer B/Q axes,
and tests forbid concretizing those runtime axes. In other words, bijectivity
is proved once as relation provenance rather than rediscovered independently
by the scheduler and code generator.

The first post-placement nested-frontier derivation is also implemented. It
factors only outer axes whose removal and lift-back reproduce the original
readiness relation exactly, maximizes any retained fibers with the existing
relation extremum, and takes the exact preimage of waves strictly before the
consumer's admission wave. It accepts only one contiguous prefix. The
Qwen-shaped proof derives `(0, 74, 96)` without materialization or endpoint
sampling and takes about 0.60 seconds cold and 0.29 seconds warm independent of
the nested extent. Independent brute-force and randomized review found exact
agreement and conservative decline. This is not yet all of Phase 4A.4: the
current caller still supplies one scalar consumer wave, and the concrete
monotone binary search remains only as a migration fallback until symbolic
task-varying admission reaches parity.

The common list-schedule acceptance gate now derives exact per-root task mass
and makespan horizon from authoritative schedule relations. A minimum-wave
capacity proof handles runtime-empty packed schedules; nonminimal schedules use
the ordinary placement/frontier relation and decline if an exact maximum is
not representable. Exact ties accept the candidate, an unproved comparison
retains the resident schedule, and no recurrence-specific bypass remains.
Independent review confirmed that candidate coverage validation precedes this
cardinality comparison and that sparse/trailing-wave cases match the concrete
oracle.

A reviewed relation-derived progress certificate now maps each resident
logical task to its scalar global slot `wave * worker_count + worker` from the
authoritative `WorkerScheduleSegment.task_order`. It composes exact root-entry
counter dependencies into consumer-task-to-producer-task relations, takes the
producer-slot maximum, and accepts only strict slot increase. It handles
symbolic empty domains and multi-arm fork/join events without enumerating
workers or tasks, while conservatively declining barriers, nested sites,
continuations, source-stage work, and proof-budget exhaustion. This rank is a
topological/deadlock certificate for admitted same-wave work; it is explicitly
not the completion-time objective used to choose continuations. Independent
review found and fixed missing per-relation and Cartesian-product budget gates.

Code generation also no longer reruns counter legality or continuation
ownership checks after `StaticPipelinePlan` construction. The plan validates
those facts once and codegen renders them. Independent review found one
production construction path and no serialization path that could bypass this
validation; any future plan serialization must add validated restoration.
The integrated tile-dependency and scheduler suites at this checkpoint pass
289 tests with 119 subtests.

A proposed shortcut that routed the old equal-width repeated recurrence through
the common list-scheduler entry was reviewed and rejected; it is not part of
this branch. For `L=3`, `W=4`, and `N=5`, that recurrence occupies
`L*ceil(N/W)=6` waves while the valid packed root-major schedule occupies
`ceil(L*N/W)=4`. It therefore cannot bypass the common same-coverage horizon
comparison merely because its relation has a compact rendering. The review
also showed that schedule geometry alone can falsely classify independent or
root-barrier roots as a recurrence. Any retained recurrence rendering must be
certified from both the final schedule relation and its actual readiness plans.
The symbolic, relation-based no-regression comparison is now complete.

A subsequent tail-packing prototype was also rejected. It preserved translated
fan-in-one producer/consumer cohorts and packed only incomplete tails, but it
had the same four-wave horizon as the existing packed root-major relation for
the representative `L=3`, `N=5`, `W=4` case. It changed only locality, supplied
no measured runtime win, and raised representative schedule-build time from
about 10 seconds to 28.5 seconds. The prototype also exposed avoidable nested
`Mod` proof hazards. Keep it out of production: locality transformations must
instead be proposed by the one event-frontier policy, pass the common progress
proof, and demonstrate a runtime benefit.

The obsolete six-wave recurrence is now gone, and the existing packed
root-major relation is the sole conservative fallback. Exact extrema with all
attainers, occupied same-strand precedence, and the generic weighted scalar
pullback are also complete. Same-strand chronology is derived only from the
authoritative `WorkerScheduleSegment.task_order` relations; a review-found
partial/strided clipping bug and a second mixed full/partial-piece clipping bug
were fixed before integration. The combined tile-dependency and scheduler
suites now pass 335 tests with 342 subtests; the separate max-plus oracle adds
12 tests with 4 subtests. Occupied strand ordinal `q(s)` is
also complete for concrete, holed, and ordinary symbolic packed schedules.
The acyclic resident root quotient and the first test-only concrete max-plus
oracle are now complete. The oracle is independently differential-tested and
already supplies topology-shaped FlashMLA, Qwen, Gemma, and Muse positive and
negative controls. Those compact fixtures are an early objective check, not a
substitute for extracting the real model graphs; the real-graph and symbolic-
substitution gates below remain mandatory before production ownership changes.
The active implementation point is therefore the acyclic relation-level
max-plus evaluator, compared exhaustively against the concrete oracle without
yet changing scheduling policy. Then continue with the real-model gate, joint
continuation/resident choice, translated recurrence closure, post-placement
nested quotients, source-ticket actions through that same policy, one final
validation/lowering pass, and deletion of the top-level constant/parameterized
branch.

Then continue Phase 4A.2 by replacing the remaining parameter-only admission
filters with one exact action-legality decision, followed by the single joint
event-frontier policy in Phase 4A.3.

### Priority 1: finish the symbolic dependency refactor

- [x] Canonicalize every `TileAccess` shape, stride, and storage-offset value
  once at construction into a SymPy integer expression, including
  `sympy.Integer` for constants.
- [x] Remove the concrete size-hint fallback from dependency `TileAccess`
  construction. Keep size hints available only to existing tuning paths.
- [x] Give `layout_is_symbolically_exact` its narrow meaning: the stored
  layout expressions contain only guarded host-backed parameters. Test that
  indirect indices and explicit masks leave this flag true but separately
  make access-relation construction decline.
- [x] Preserve multi-axis symbolic domains in the existing
  `CoordinateRelation` and factor exact positional products without
  enumerating runtime extents.
- [x] Derive fixed-width producer-set quotients, their converse publication
  relation, and arrival cardinality from one partition certificate.
- [x] Preserve and bound nonuniform static tails in ordinary readiness
  counters rather than widening them or replacing them with a root barrier.
- [x] Finish and audit nonuniform counter lowering end to end: decode keys
  using symbolic axis counts, evaluate the per-key arrival expression, use
  its proved static maximum for epoch framing, and keep continuations limited
  to uniform exact-once cases.
- [ ] Re-audit that dependency construction, readiness selection, schedule
  proof, diagnostics, and codegen all consume the same relation/count facts;
  remove duplicate codegen barrier/continuation derivation and any hinted
  layout, fan-in inference, or CTA-DAG truth.
- [ ] After exact symbolic layouts are wired through, audit every helper,
  field, and fallback added by the earlier parametric work. Delete or fold any
  mechanism whose only purpose was to recover information lost by size-hint
  concretization; retain a compatibility fast path only when it is a proved
  rendering of the same `CoordinateRelation`, never a second truth.
- [x] Run focused tests, the full tile-dependency and cross-loop-scheduler
  suites, formatting/lint checks, and `git diff --check`.
- [ ] Re-obtain final independent reviewer sign-off on the complete compiler
  diff. Reviewers must explicitly check the single-source-of-truth invariant,
  absence of unnecessary abstractions and model cases, symbolic proof
  soundness, nonuniform-counter replay safety, cleanup completeness, and that
  constants versus parameter expressions do not select different scheduling
  policy.

Priority 1 exit gate: dynamic layout values are never concretized for
correctness, ordinary nonuniform counters pass replay/tail tests, unsupported
indirect or conditional accesses decline conservatively, and the compiler has
one symbolic dependency representation. The refactor is not complete until
the cleanup audit and final reviewer sign-off are recorded.

### Priority 2: complete the generic relation coverage

- [x] Bridge exact contiguous symbolic multidimensional and flat layouts for
  direct affine tiles such as `(B, 8) <-> 8*B` without specializing `B`.
- [x] Preserve exact index provenance for flattened gathers whose `tile.index`
  lane width is configuration-selected. The same affine representation now
  recovers the Qwen 5→6, 6→7, and 7→8 relations.
- [x] Feed that provenance into the existing linear `CoordinateRelation` and
  complete the mixed-radix 5→6 and 6→7 proofs without a Qwen-specific rule or
  compile-time enumeration proportional to a runtime extent.
- [ ] Support exact nested-consumer and multi-producer join obligations using
  the existing relation union/composition operations and one emitted-
  prerequisite view.
- [ ] Define and prove the safe publication rule for masked or conditional
  producer bodies. Never count an inactive producer unless every scheduled
  CTA is independently proved to publish.
- [ ] Keep normalization cost bounded by rank and relation-piece count; add a
  structural piece budget and conservative fallback for pathological unions
  or compositions. The existing coalescing comparison budget is not enough:
  union, source-cell products, composition, and L2 order construction must
  check the common budget before materializing an intermediate product.

### Priority 2A: remove the static/parameterized policy split

This is a blocking architecture milestone, not optional cleanup. Every kernel
must traverse one semantic pipeline regardless of whether a domain extent is a
constant or a guarded host-backed expression:

```text
TileDependencyGraph
    -> one ReadinessGraph
    -> one event-frontier scheduling and ownership policy
       (each early-admission action proves its existing synchronization lowering)
       with conservative root-major fallback
    -> one WorkerSchedule
    -> schedule-frontier quotient for nested waits
    -> one counter/root-barrier finalization pass
    -> one RootBarrierPublicationPlan derivation
    -> one symbolic ownership/progress/replay validation
    -> renderer-only codegen and backend compilation
    -> post-compile residency gate before cache acceptance or launch
```

The ordering is deliberate. Continuation eligibility must be known before
scheduling, but ownership is selected by the event-frontier scheduler when an
event releases its consumer and before that consumer receives any resident
slot. A final-arrival continuation is therefore an execution action of the
same scheduling policy, not an independent pre-pass or a later rewrite. Exact
semantic readiness is already available from `ReadinessGraph`, so scheduling
does not need emitted counters to be finalized. Conversely, a nested counter
partition may depend on the final producer and consumer wave relation. It
therefore cannot be coarsened or finalized before placement. Qwen's useful
74/22 frontier is the concrete counterexample to the old ordering.

- [x] Build `ReadinessGraph` once before any concrete/parameterized rendering
  choice.
- [ ] Derive final-arrival continuation eligibility from the one
  `ReadinessGraph` before comparing scheduling actions.
  Remove the parameterized sink-only and `fan_in > 1` policy. Eligibility must
  depend on exact-once event/consumer mapping, publishability, body safety, and
  progress—not on whether an extent contains a free symbol. This does not
  choose ownership or contract the graph.
- [ ] Before accepting any early-admission action, prove from that same event
  and its existing relations that the required counter/publication lowering is
  available. This is an action-legality check inside the one scheduler, not a
  filtered graph, capability object, or alternate policy. A schedule-dependent
  nested mechanism is attempted transactionally: if its final quotient cannot
  lower, reject the entire optimized proposal once.
- [ ] From one complete provisional all-resident schedule, let the one
  event-frontier scheduler compare each eligible continuation with resident
  ownership. Charge each inline body, including a chain, as one unit after its
  final producer on every possible final-producer strand. Compare the
  guard-wide lexicographic objective `(unit completion makespan,
  critical-path resident handoff depth)` and commit at most one continuation
  globally.
  Inline may win a proved primary tie only by strictly reducing the maximum
  handoff depth among primary-critical terminal paths; prefer resident on an
  unproved comparison or a full objective tie. The decision is
  atomic for the whole event family and compiled guard. Never run a separate
  continuation-selection pass before or after scheduling. The selected
  candidate remains the existing ephemeral `FinalArrivalContinuation` until
  the final `ReadinessCounterPlan` records it.
- [ ] Replace the permanent single-producer parameterized-counter gate with
  the same exact publication/cardinality/replay certificate used for all
  domains. Keep stricter continuation requirements only where execution
  ownership genuinely requires them.
- [ ] Run one symbolic event-frontier proposal implementation for constants and
  parameters alike. It consumes the exact `ReadinessGraph`, not a prematurely
  coarsened counter plan, and produces the authoritative schedule relation.
  Only afterward may codegen choose a concrete loop or compact recurrence as a
  certified rendering. Concrete task-list placement remains a test oracle, not
  an alternate production decision procedure.
- [ ] After the final `WorkerSchedule` is selected, derive each nested
  consumer's maximal schedule-frontier quotient. For a fixed owning task, group
  adjacent nested iterations exactly when their latest required producer wave
  has the same relation to the consumer's admission wave. Keep the exact
  per-iteration event as semantic truth; the quotient is only the emitted
  synchronization plan. If an optimized placement relied on a quotient that
  cannot lower, discard that proposal. A root-entry counter or root barrier is
  used only when the placement was already proved safe with that coarser wait,
  including in the rebuilt all-resident fallback.
- [ ] Finalize ordinary counters, continuation counters, and root-barrier
  fallbacks once after placement, then validate the final result against the
  original dependency obligations. No late step may silently change ownership.
  A rejected optimized proposal is discarded in full and may fall back once to
  root-major placement; placement and synchronization must not repeatedly
  mutate one another.
- [ ] Make conservative root-major order the fallback of that same policy.
  Concrete wave-aligned and symbolic slot-packed schedules must not remain
  competing semantic baselines.
- [ ] Keep transient source execution as an ordinary launch-stage-zero
  `WorkerScheduleSegment`; derive its identity from that relation and remove
  separate selection policy or mapping truth.
- [ ] Generalize the existing `RootBarrierPublicationPlan` for every schedule.
  It alone derives publication support, final occurrence, exact runtime arrival
  count, effective empty-root count, static bound, per-site contribution, and
  empty-root completion ownership from authoritative `task_order` and retained
  continuation plans. One plan is shared by every outgoing edge of a producer
  root.
- [ ] Use unit arrivals with bounded uint64 epochs as the common barrier
  protocol for constant and parameterized schedules. Weighted fixed-mass arrivals remain a diagnostic-only
  strength-reduction candidate unless identical-body measurements establish a
  material need and the same plan proves a dense publisher rank.
- [ ] Remove codegen-side reconstruction of schedule geometry, continuation
  task counts, root arrival counts, and counter legality. Recognizers may only
  render a certificate already stored in the selected plan.
- [ ] Validate ownership, dependency coverage, and progress against the final
  original graph without task-, worker-, wave-, or runtime-extent enumeration.
  Strict-before is the fast sufficient proof for any wait. Root-entry and
  nested waits may instead depend on same-wave producers when all are assigned,
  the symbolic segment-precedence relation is acyclic, no required producer
  lies later on the waiting consumer's own strand, and the resident capacity
  certificate holds. Requiring every producer to occupy an earlier wave would
  reject both Gemma's packed reduction and Qwen's final 22-task cohort.
- [ ] Inventory every class/dataclass added since the branch merge-base. Fold
  or delete `_PlacedRun`, `_EmittedPrerequisite`, `_NestedLoopReadiness`,
  `_ScheduledRootTraversal`, `_AffineIndexScalar`, `_AffineIndexTensor`, and
  any other helper that duplicates approved plan/relation state. Ephemeral
  analysis helpers may remain only when they are not semantic IR and cannot be
  represented cleanly by existing `TileAccess`/`CoordinateRelation` objects;
  no retained/cached dataclass or graph-shaped state outside the approved
  sources of truth may use this exception.
- [ ] Remove the parameterized early return in `WorkerSchedule.__post_init__`,
  replace both hard-coded `(0, W)` publication branches, eliminate codegen's
  root-arrival reconstruction/schedule-policy validation/counter-legality
  import, and extend `_forbid_schedule_enumeration` to catch direct loops over
  domain size, workers, and waves.
- [ ] Retain the old concrete scheduler only as a differential test oracle until
  the unified policy passes all performance gates; then remove it from
  production.

Priority 2A exit gate: constant and symbolic extents traverse the same policy.
Identical decisions are required only when normalized readiness relations,
guards, configured task orders, worker count, body-safety facts, and capacity
facts are extensionally identical. A broad symbolic guard and a narrower
constant compile may expose different provable facts, but one polymorphic
binary must make one event-family continuation choice over its whole guard;
the existing scalar `continuation_consumer_index` cannot express a guarded
per-shape ownership choice. No compiler path contains a model, root-ID
predicate, sampled shape, or benchmark-specific exception.

The equality required here is semantic rather than syntactic: constant codegen
may strength-reduce a bounded epoch counter, a participant relation, or a
symbolic recurrence. Such rendering differences are allowed only after their
equivalence to the same finalized plan is proved.

Implementation checkpoint (commit `a9df42d1`): the first correctness slice is
complete. `RootBarrierPublicationPlan` now owns exact resident, continuation,
and source-stage arrival counts; packed root-major schedules derive their
runtime active-worker relation from the authoritative `task_order`; and
bounded unit-arrival barriers reuse one cubin across changing shapes, including
empty stages. The lowering also preserves SymPy's Euclidean `Mod` semantics
when emitting Triton. This matters for wrapped worker cohorts: signed remainder
previously omitted 128 required Qwen B2 publishers and deadlocked the following
root. The fixed B2 replay is bit-exact. This checkpoint does not yet remove the
top-level constant/parameter policy split or `transient_source_root`.

The same checkpoint replaces negative provenance (“not recognized as an
input”) with positive storage provenance for exact wrapper allocations.
Input aliases, views, and `torch.empty(..., out=input)` therefore cannot make
runtime strides look static. This is a prerequisite for comparing symbolic and
exact-shape schedules without silently changing the generated kernel body.

### Priority 2B: validate the scheduling objective before policy wiring

Do not wait for the complete symbolic continuation lowering to discover
whether the unit-work objective makes the right decisions. Add a test-only
concrete max-plus oracle over the actual model-shaped `ReadinessGraph` and
`WorkerSchedule` relations. Materialization is allowed only inside this oracle;
production scheduling must remain bounded by roots, relation pieces, and
symbolic expressions.

- [x] Evaluate the exact task DAG with one unit of completion cost per body and
  a secondary cross-worker handoff count. Check the implementation against
  exhaustive tiny DAGs before using model-shaped cases.
- [ ] Score the current all-resident schedule, the previously successful
  FlashMLA event-aware schedule, and each single continuation alternative over
  the complete downstream horizon. The oracle must count an inline body once,
  use the maximum over all event producers, and retain all possible final-
  publisher alternatives.
- [ ] Run the oracle on the checked-in source structures for ragged FlashMLA
  B4, both Qwen forms, both Gemma 4 A4B forms, and Muse/Glimmer FFN. It must
  predict the known qualitative decisions: immediate FlashMLA reduction
  release improves the topology; Qwen retains its good attention ordering and
  resident root 13; Gemma keeps the expert reduction resident when inlining
  lengthens the critical path; and Muse completes a fan-in group and releases
  its consumer instead of spreading equal-priority producers across every
  group. Add DeepSeek and Nemotron once their exact graphs are available.
- [ ] For every symbolic case accepted by the production evaluator, substitute
  `0`, `1`, boundary-minus-one, boundary, boundary-plus-one, and representative
  larger extents and require exact agreement with this oracle. Constants and
  runtime shapes must not select different policies merely because one path
  uses Python integers and the other uses SymPy expressions.
- [ ] If the objective fails to rank any already measured win or negative
  control correctly, revise the objective before wiring it into production.
  Do not compensate with a model/root-ID heuristic.

  Implementation checkpoint (2026-09-10): the test-only oracle materializes
  exact root-level readiness edges and immediate occupied same-strand edges,
  then evaluates the lexicographic longest path. It rejects missing ownership
  and cycles, does not impose global slot order, shares production continuation
  eligibility, retains all causally maximal singleton publishers as mutually
  exclusive alternatives, and counts the inline body once. Independent checks
  covered 1,000 randomized resident graphs, 300 multi-key graphs, 500 singleton
  continuation cases, and a multi-axis domain. Compact topology-shaped motifs
  rank FlashMLA early release, Qwen ordering/resident join, Gemma's resident
  reduction, and multiplicity-preserving Muse group completion correctly. A
  deliberately over-serialized Muse caricature remains as a negative control.
  Correlated multi-body continuations, extraction of the actual model graphs,
  and symbolic-substitution parity remain explicit follow-up work.

This is an early scheduling signal, not a GPU performance model. It validates
dependency overlap, critical-path completion, ownership, and handoff logic; it
does not predict register pressure, shared resource contention, instruction
mix, or constituent body speed. Those remain the later cold-L2 performance
gate rather than inputs to scheduling policy.

### Priority 3: resume the paused validation and performance work

- [ ] Revalidate canonical ragged FlashMLA B4 against matched standalone
  Helion and ThunderKittens, including cold-L2 latency and aligned Gantt
  charts; retain Q1/Q2 and random B9 coverage.
- [ ] Revalidate both Qwen3 source forms: the checked-in pretuned production
  kernel at its native fixed shape and the mechanically related ragged B1/B2
  source with one cubin. Record schedule/counter/barrier structure, resources,
  matched standalone, and current-main latency for each; neither source may be
  substituted for the other.
- [ ] Revalidate both Gemma 4 A4B source forms: the checked-in pretuned fused
  kernel and the B1/B2 symbolic probe. Preserve routed expert diversity and
  report schedule/counter/barrier structure, resources, matched standalone,
  and current-main latency.
- [x] Revalidate Muse/Glimmer FFN B1/B2, including its nonuniform 32/16 tail,
  and preserve the current persistent-over-standalone result.
- [ ] Complete Muse B4, identical-source exact-shape, event-aware-order A/B,
  and compile-time gates.
- [ ] Audit DeepSeek-V3 MoE and Nemotron MoE for remaining static-factor,
  nested-wait, and multi-producer relation gaps.
- [ ] Treat Triton specialization/resource-envelope regressions as a separate
  body-lowering problem; do not encode them as dependency or scheduler
  heuristics. A compiler-owned literal stride requires positive allocation and
  layout provenance; `tensor_input_source(...) is None` alone is insufficient.
  Test noncontiguous inputs, views/aliases, and `empty_like(input)` before
  landing the internal-stride optimization.
- [ ] After the performance gates pass, compare cold compile time against
  current main for the identical source, configuration, and cache state.
  Report Helion analysis/code-generation time separately from Triton binary
  compilation. The current roughly 85--100 second Qwen diagnostic compile is
  not an accepted end state if it materially regresses main; any repair must
  remain bounded by graph and relation structure and must not enumerate runtime
  `B`, sequence length, workers, waves, or CTAs. Compile-time work is secondary to
  restoring runtime parity, so tune it only after the generated schedule and
  kernel performance are correct.

The compile gate is numeric. For each frozen source/configuration, record a
clean current-main baseline and require:

| quantity | limit |
| --- | --- |
| dependency plus scheduling/proof time | `max(10 s, 2.0 * main)` |
| generated Python/Triton source bytes | `max(main + 64 KiB, 1.25 * main)` |
| backend/Triton compilation time | `max(main + 15 s, 1.25 * main)` |
| total cold compile time | `max(main + 20 s, 1.25 * main)` |
| retained pieces per relation | 4,096 |
| prospective pair/product states per relation operation | 65,536 |

These deliberately generous limits make compile time secondary to runtime
parity while rejecting the observed 98-second relation proof and greater-than-
five-minute Gemma reconstruction. A workload exceeding a limit blocks rollout
until the structural operation is bounded; it is not waived by a fast kernel.
- [ ] Split final compiler commits from historical benchmark/probe commits and
  keep new probes and journals out of the compiler change.

### 2026-09-09 validation checkpoint; architecture review reopened

- Fresh full-diff and workload reviews agree with the event-frontier direction
  but block the prior draft. They require joint continuation/placement
  selection, proof that each early action has a lowerable wait, a common
  same-wave progress proof, explicit empty-root chronology and capacity gates,
  and bounded relation operations. Final plan and implementation sign-off both
  remain open until those points and Priority 2A are complete.

- The combined tile-dependency, cross-loop-scheduler, and Triton lowering
  suite passes: 241 tests, 35 subtests, and one expected skip on GPU 6.
  Checked-in coverage includes zero-sized replay, moving counter-section
  offsets, exact bounded nonuniform fan-in, symbolic flattened gathers, and
  conservative decline for unsupported symbolic layouts and relations.
- Dynamic Qwen reuses one cubin for B1/B2 and retains exact 5→6, 6→7, 7→8,
  12→13, and 13→14 counters. After production active-participant barriers and
  Euclidean-modulo lowering, it measures 109.456 us for B1/S8192 and
  133.024 us for ragged B2/S2048+S8192 versus identical-source exact-shape
  106.368/126.976 us. The earlier 128.86/155.65-us and probe-only
  108.42/139.02-us results are superseded. The current dynamic compilation is
  111.517 seconds and fails the compile budget; B2 also remains outside the 3%
  runtime parity gate.
- A later source-identical bisect isolated the remaining Qwen B1 regression:
  `f8271993` measures 94.08 us and retains a two-key nested 13→14 frontier
  (74 producers, then 22), while `f6134a11` and the current symbolic path
  coarsen it to one fan-in-96 wait. Reverting only continuation ownership does
  not recover performance. The unification work must therefore preserve
  nested release frontiers derived from the exact dependency and final worker
  schedule; it must not treat every nested wait as a root-entry barrier.
- Positive layout provenance now has a measured second requirement: an
  integer-strided view of a sole input storage is exact when every stride of
  that replayable input was explicitly specialized. Without this propagation,
  four Gemma flattened-weight strides became runtime arguments and changed the
  kernel from R126/0-byte spill/34,816-byte shared to R128/12-byte spill/
  4,096-byte shared. The positive proof restores dynamic B1/B2 latency from
  141.15/190.43 us to 61.31/77.74 us, versus static 51.07/71.58 us and
  standalone 55.20/69.50 us, while preserving one cubin and bit-exact replay.
- Gemma continuation and worker-width controls supersede that first post-fix
  timing. At m2, resident root 6 measures 61.408/77.792 us; forcing only root 6
  inline measures 63.568/83.936 us; the full 5→6→7 chain measures
  65.472/84.064 us; and a u32 synchronization diagnostic differs by at most
  0.032 us. At tuned m3, one-cubin dynamic measures 53.216/75.712 us versus
  exact packed 51.168/71.712 us and standalone 55.264/69.888 us. The remaining
  work is symbolic/state rendering and the B1 ownership tradeoff, not resource
  loss or uint64 atomics.
- Dynamic Muse/Glimmer retains its exact 32/16 tail counter and one cubin.
  Measured persistent versus matched standalone latency is 1579.10/1649.60 us
  at B1 and 3110.08/3244.00 us at B2.
- The completed dependency-analysis cleanup removed size-hint-based correctness
  proofs and retained affine access provenance, positional-product and
  fixed-width relation proofs, exact allocation sizing, and bounded exact
  target-box normalization. The broader scheduling/helper cleanup remains open
  under Priority 2A.
- Earlier reviews approved the symbolic dependency representation, replay
  safety of exact readiness counters, and absence of model-name dispatch. A
  subsequent full-diff review found that the early concrete/parameterized
  branch still duplicates scheduling policy and that codegen reconstructs
  some barrier facts. Those findings supersede the earlier architecture
  sign-off; final review remains open until Priority 2A is complete.

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
RootBarrierPublicationPlan      derived barrier support/count/contribution
        |
cross_loop_codegen              renderer of finalized relations and plans
```

There is one semantic policy for both constant and parameterized extents.
Code may branch on a proved relation form only after synchronization and
scheduling decisions are complete, in order to emit a compact concrete loop
or symbolic recurrence. Such a branch is a rendering choice, never a second
continuation, barrier, source-admission, or ordering policy.

## Current implementation diagnosis

The compiler already has most of the right mechanisms. The architectural
problem is that their selection and ordering are split across two policies.
The current constant-extent path is approximately:

```text
baseline WorkerSchedule
    -> choose continuations
    -> reorder continuation producers
    -> choose continuations again
    -> place nested consumers and derive counter splits
    -> place newly ready root families
    -> propose a global list schedule
    -> finalize counters and root barriers
```

The current parameterized-extent path is approximately:

```text
packed root-major WorkerSchedule
    -> choose only sink continuations with uniform fan-in greater than one
    -> discard counters outside a separate parameterized subset
    -> keep packed root-major order
    -> finalize its counters and barriers separately
```

This split was a conservative implementation sequence, not a semantic
requirement of dynamic shapes. The original concrete algorithms use
`CoordinateDomain.size`, Python `range`, materialized worker intervals, and
search over concrete waves or loop iterations. Applying them literally to a
runtime `B`, `Q`, sequence length, or token count would either specialize the
binary, build a host schedule for every shape, or perform compile work
proportional to a runtime extent. The initial parameterized path therefore
accepted only relations with simple closed forms and lowering protocols whose
replay safety was already proved.

That temporary safety boundary now causes observable policy divergence:

- Gemma exposes that ownership cannot be copied between the two paths. The old
  exact B1/m4 path uses an expert-down-to-reduction continuation, while packed
  B1/B2 schedules benefit from a resident reduction in the producer's partial-
  wave holes. The current paths reach those outcomes through unrelated ordering
  and selection passes instead of one structural comparison.
- Qwen's constant path places complementary producer and nested-consumer
  cohorts in one wave and derives a 74/22 readiness frontier. The parameterized
  path retains the exact semantic event but fails the mixed-radix maximum proof,
  so it moves the consumer and emits one fan-in-96 entry wait.
- FlashMLA's source-first execution is represented correctly as a launch-stage
  segment, but `transient_source_root` still duplicates its identity and the
  policy is not yet reached through the same selection flow.
- Parameterized root barriers now have correct active-owner and replay-safe
  epoch lowering, but constants and parameters still select different barrier
  renderings before a common semantic plan exists.

The constant path is not the desired architecture either. It contains a
circular sequence: continuation choice changes placement, placement can change
continuation viability, nested placement creates synchronization plans, and a
later global proposal can change the schedule again. Consequently:

- continuation selection runs twice;
- `place_nested_loop_consumers` both schedules work and invents emitted
  synchronization;
- `place_ready_families` may reverse an earlier ownership choice;
- nested counter partitions can describe a schedule that a later pass changed;
- synchronization can be finalized, discarded, and recomputed; and
- concrete enumeration and quadratic segment checks limit compile-time scale.

The redesign therefore does not make the symbolic branch imitate this control
flow. It preserves the useful decisions of the concrete path while replacing
both paths with one acyclic decision pipeline.

### Optimization landscape and ownership

The following distinctions are normative:

| Layer | Question answered | Authoritative existing object | May affect |
| --- | --- | --- | --- |
| Dependency analysis | Which producer instances feed a consumer instance? | `TileDependencyGraph` and `CoordinateRelation` | correctness |
| Readiness formation | Which exact producer set completes one logical event? | `ReadinessGraph` | synchronization granularity |
| Continuation ownership | Does the final producer execute the consumer body, or does it remain resident? | finalized `ReadinessCounterPlan.continuation_consumer_index` | task ownership and handoff latency |
| Resident scheduling | Which worker/wave owns each remaining task? | `WorkerScheduleSegment.task_order` | overlap and makespan |
| Nested frontier quotient | Which adjacent loop iterations share one emitted wait for this schedule? | finalized `ReadinessCounterPlan` derived from the exact event and `WorkerSchedule` | synchronization overhead without weakening readiness |
| Root completion | Which owners publish, and what target is replay-safe? | `RootBarrierPublicationPlan` | barrier correctness and overhead |
| Rendering | Which proved loop/recurrence/constant form emits the plan? | `cross_loop_codegen` | instruction count only |
| Body/resource tuning | How are individual root bodies tiled and pipelined? | existing block/range/warp/register configuration | constituent-kernel speed and occupancy |

A final-arrival continuation and list scheduling are intentionally separate.
List scheduling orders tasks that remain resident: a producer publishes, a
resident worker reaches the consumer, waits, and executes it. A continuation
changes ownership: the producer observing the final arrival immediately calls
the consumer body, so that consumer has no resident slot to schedule. They are
separate execution mechanisms selected by one scheduling policy: when an event
closes, the scheduler either takes the eligible inline action or assigns the
consumer to a resident frontier. Both consume the same readiness event and
neither defines a second dependency graph.

Body/resource tuning is also separate from scheduling. `num_warps`,
`maxnreg`, block sizes, range staging, flattening, loop order, L2 grouping, and
`num_sm_multiplier` may alter throughput or feasible residency, but they must
not change dependency truth or introduce model-specific schedule policy. The
scheduler consumes the final worker count and a backend capacity certificate;
it does not predict resource cost.

## Unified end-state pipeline

The single production pipeline is:

```text
1. TileDependencyGraph
      exact memory/dataflow obligations
2. ReadinessGraph
      exact producer sets and consumer requirements
3. Event-frontier scheduling and ownership
      every early-admission action proves its existing synchronization lowering;
      atomically compare resident placement with charged inline execution;
      produce one WorkerSchedule plus at most one selected continuation
4. Schedule-frontier quotient
      coarsen nested waits against the final schedule, when exact
5. Synchronization finalization
      retained counters, continuation identity, and root-barrier fallback
6. RootBarrierPublicationPlan
      exact publishers, arrival counts, bounds, and empty-root owner
7. Symbolic validation
      ownership, coverage, progress, configured-capacity bound, and replay safety
8. Code generation and backend compilation
      renderer and proved strength reductions only
9. Post-compile residency gate
      verify actual cubin occupancy before cache acceptance or launch
```

There is no capability graph or preliminary synchronization plan. The
scheduler reads the one `ReadinessGraph`; when considering early admission or
inline ownership, it proves that the event's existing relations can lower the
required counter/publication. From a complete provisional resident placement,
it then makes ownership and placement one atomic decision: inline execution is
charged on every causally possible final-publisher strand and resident
execution retains its proved slot. Final synchronization is constructed only
after placement supplies the information needed to quotient nested waits. No
provisional counter plan is a source of truth.

There is one bounded fallback, not a second policy. If the optimized schedule,
schedule-frontier quotient, or progress proof declines, discard that proposal
and construct the all-resident conservative root-major `WorkerSchedule`, derive
synchronization for that schedule, and validate it. Its same-wave boundary
waits use the common assigned-producer/no-same-strand-future/acyclic/full-
residency proof; if that proof declines, wave-align the affected boundary.
Do not alternate between placement and mechanism selection until a heuristic
fixed point is reached.

For a constant domain, SymPy simplification may reduce every bound to an
integer and codegen may emit today's compact loops. For a parameterized domain,
the same relations retain guarded expressions and codegen emits runtime-bounded
loops. Substitution parity is required when the full normalized policy inputs
are extensionally identical. A narrower constant guard may prove an ownership
choice that cannot be made uniformly over a broader polymorphic guard; both
must still be outputs of this one policy.

## Guiding scheduling principle

> Among all provably admissible work, protect the unit-weight precedence
> critical path; at equal criticality, finish the readiness event closest to
> completion and immediately admit its consumers; never leave a worker idle
> while admissible work exists.

This principle has three ordered parts:

1. **Legality:** `TileDependencyGraph` and the exact `ReadinessGraph`
   determine which work may be assigned and which runtime wait protects it.
2. **Priority:** unit-weight `top`/`bottom`/slack protect structurally critical
   chains. Event-closing lookahead and remaining-producer count break ties so a
   nearly complete fan-in is finished instead of spreading equal-priority work
   across many keys. For ownership choices with equal proved completion
   makespan, minimize cross-owner readiness handoff depth on the
   primary-critical terminal paths.
3. **Work conservation:** every worker slot receives admissible work when any
   exists; newly released consumers enter the same ready frontier immediately.

Unit weight is intentional. It supplies a stable, shape-polymorphic precedence
priority without pretending to predict instruction latency, register pressure,
bandwidth, or occupancy. Those resource effects remain empirical performance
gates, not scheduler inputs.

The dynamic-shape refactor must preserve this decision rule under symbolic
substitution. It may summarize repeated decisions with affine/floor/modulo
pieces or decline to canonical order when a comparison is unprovable, but it
must not replace the rule with a parameterized-only recurrence, sink-only
continuation policy, or sampled-shape schedule.

`CoordinateDomain` and `CoordinateRelation` are parameter-aware on this branch.
For one
invocation, let `p` be the fixed tuple of runtime shape parameters. A
normalized `WorkerScheduleSegment` is a root-labelled family of relations:

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

The original main implementation restricted each segment to one concrete
contiguous dispatch interval and used tuple order as chronology. This branch
already normalizes that form into global schedule coordinates, but its
parameterized validator still admits only hand-recognized root-major and
event-frontier shapes. That recognizer split cannot support a runtime-sized
multi-root pattern such as:

```text
for key in runtime_key_count:
    finish the key's producers
    admit its reduction
```

Adding another schedule hierarchy would duplicate `WorkerSchedule`. Instead,
finish making the existing relation's schedule coordinates authoritative and
validate them through common `CoordinateRelation` support operations. A single
relation piece can then cover an arbitrary runtime number of translation-
equivalent repetitions.

This is more invasive inside `WorkerScheduleSegment` than wrapping it in a new
Region type, but it leaves Helion with fewer concepts and one source of truth.

## Non-negotiable constraints

- One persistent Triton kernel for the scheduled boundary.
- No change to fusion boundaries.
- No change to numerical algorithms, accumulator types, or reduction order.
- No model names, root IDs, or benchmark shapes in compiler policy.
- No measured, profiled, shape-specific latency, register, bandwidth, or
  resource cost model. Unit root/task weight is the explicit structural model
  used for critical-path priority and no-regression comparison.
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

Helion's existing tracing and specialization policy decides which values are
constant before dependency analysis. A specialized dimension, stride, or
configuration value enters the common pipeline as a constant expression,
canonically `sympy.Integer`; an unspecialized host-backed shape enters as a
guarded symbolic expression. `tile_dependency` and the scheduler never invoke
a static alternative or independently decide to specialize. Constants simply
allow the same relation operations to simplify further and permit codegen to
strength-reduce the finalized plan.

For the decode probes, `B` and host-backed sequence-length extents are the
canonical runtime expressions. They must remain symbolic through dependency
analysis, scheduling, and Triton lowering. Model dimensions, fixed tile
geometry, and invariant layouts are not made dynamic merely because `B` or a
sequence extent is dynamic.

### Runtime shapes versus layout specialization

Dynamic shape reuse must not erase layout facts that the compiler already
knows. Apply this one rule consistently:

- a concrete stride of a compiler-created allocation is emitted as a literal;
- a stride that symbolically depends on `B`, sequence length, or another
  schedule-polymorphic value remains a runtime expression;
- a user-input stride remains runtime unless the source explicitly applies
  `hl.specialize`; and
- `B` and sequence-length expressions themselves are never specialized merely
  to recover performance.

`hl.specialize(input.stride(i))` is the correct source mechanism for an
invariant input layout. It is not the mechanism for compiler-created
temporaries: Helion chose those allocations and must preserve their known
concrete strides automatically. Triton's `do_not_specialize` set must contain
only values whose runtime variation the binary promises to support; it must
not turn compiler-known layout constants into runtime parameters.

This is body lowering rather than scheduling policy. It may change generated
address arithmetic, shared-memory selection, registers, and spills, but it
must not change `TileDependencyGraph`, `ReadinessGraph`, or `WorkerSchedule`.

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
- exact readiness-event publication sites;
- exact arrival cardinalities; and
- obligations not discharged by exact events.

The finalized `StaticPipelinePlan` exposes emitted prerequisites as a derived
iterator/property over its retained counter plans and root-barrier edges. Its
result may be cached against the immutable final plan, but no independently
constructed or mutable descriptor collection may disagree with those fields.
Proposal, proof, codegen, and diagnostics consume that same derived view.

There is no new EventFamily object. An event family is simply an existing
`ReadinessGraph` event considered over parameterized domains.

### `StaticPipelinePlan`

This remains the complete plan consumed by codegen. It continues to own the
selected `WorkerSchedule`, `ReadinessCounterPlan`s, and root-barrier edges.
Continuation identity lives only in
`ReadinessCounterPlan.continuation_consumer_index`; source identity is a
derived query over the unique launch-stage-zero schedule segment, never an
independently mutable fact.

The existing `RootBarrierPublicationPlan` for every producer root is exposed
through a derived, frozen property of this plan and forced during final
validation. It is not a new stored source of truth: it is cached solely from
the final worker schedule, retained continuation plans, and root-barrier edges.
Codegen consumes that property and may not derive or revalidate publication
policy independently.

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

`CoordinateDomain.axis_counts_items` now accepts integer expressions over
Helion's symbolic shape environment. The remaining work is to make every
consumer use that representation without falling back to concrete-only
enumeration or parameterized-form recognizers.

### Symbolic memory-layout contract

This is a refinement of the existing `TileAccess`; it is not a replacement
for it and does not introduce a parallel access representation.

`TileAccess.tensor_shape`, `tensor_strides`, and `storage_offset` retain their
exact integer expressions whenever those expressions contain only guarded,
host-backed shape symbols. These fields have one canonical internal spelling:
every value is a SymPy integer expression and constants are `sympy.Integer`,
not a mixture of Python `int` and SymPy values. Conversion happens once when
the access fact is built; proof operations do not repeatedly normalize the
same value.

The flag is named `layout_is_symbolically_exact` and means only:

> Shape, stride, and storage offset are exact expressions over guarded,
> host-backed shape parameters.

It does not describe the subscript. An indirect expert lookup, explicit mask,
or conditional access can have a symbolically exact tensor layout while its
access-to-allocation relation remains unsupported. Those access properties
independently cause relation construction to decline. In particular,
ordinary layouts such as these remain exact:

```text
shape  = (B, 4096)       stride = (4096, 1)
shape  = (8*B, 1408)     stride = (1408, 1)
shape  = (B, 8, 2816)    stride = (22528, 2816, 1)
```

Unbacked/data-dependent layout symbols and layout expressions whose equality
cannot be proved remain unknown. An unknown layout may conservatively create a
dependency edge, but it cannot contribute a fine-grained relation. A size hint
is permitted to choose or tune a configuration; it must never substitute for
an extent, stride, offset, alias fact, relation bound, or schedule proof. There
is no concrete hinted mirror of the symbolic layout inside the dependency
model.

The exact-dataflow pipeline remains the existing one:

```text
TileAccess(shape/stride/offset expressions)
    -> access-to-allocation CoordinateRelation
    -> producers_by_consumer on TileDependencyGraph edges
    -> exact readiness events in ReadinessGraph
    -> joint scheduling/ownership in WorkerSchedule
    -> schedule-derived nested frontier quotient
    -> selected counters/barriers in StaticPipelinePlan
    -> root publication derivation in RootBarrierPublicationPlan
    -> direct cross_loop_codegen lowering
```

`TileDependencyGraph` remains the only semantic DAG. The scheduler does not
reconstruct a CTA or root/event DAG. Scheduling traverses the already-selected
`ReadinessGraph` relations directly using only ephemeral cursors, claimed
frontiers, and adjacency maps. Correctness and code generation consume the
original relations and covered dependency obligations.

### Normalization of one dependency relation

For a dependency from consumer tasks to producer tasks, construct exactly one
`CoordinateRelation` over symbolic domains. Normalize that relation using a
small set of exact, composable identities:

1. Factor a Cartesian product `Identity(dynamic_axes) × inner_relation`.
   The dynamic axes may be renamed or permuted, but their extents and point
   coordinates must be symbolically equal under the current shape guards.
2. Extend that rule to exact static-factor reshapes, for example
   `(B, 8) <-> 8*B` and `assignment -> floor(assignment / 8)`. The factor is a
   compile-time integer and both domain-size equations must hold exactly.
3. Recognize affine projections and fixed-width producer-set quotients. A
   consumer-to-key relation and its producer publication relation are derived
   together from the same partition certificate.
4. Preserve fixed inner tails as bounded piecewise relations. For example, a
   39-tile producer dimension consumed in width-two groups has arrival counts
   32 for the first 19 keys and 16 for the final key; it must not be widened to
   32 or collapsed into a root barrier.
5. Coalesce adjacent exact boxes canonically before deriving the converse and
   cardinality. Coalescing is semantic normalization, not a scheduling
   heuristic.
6. Decline masked, indirect, ragged, or otherwise unsupported relations
   conservatively. Never infer an unconditional producer publication from a
   conditional memory store unless every scheduled CTA is separately proved
   to publish the corresponding readiness event.

The normalization result is still a `CoordinateRelation`; there is no new
normalized-relation class. Converse, per-key arrival count, totality,
publication, waiting, continuation selection, and diagnostics all consume
that same relation. In particular, ordinary parameterized counters may use an
exact nonuniform `arrival_count_by_key`; final-arrival continuations retain
their stricter uniform/exact-once requirements.

Parameterized nonuniform counter lowering follows that relation exactly:

- decode readiness keys with symbolic `CoordinateDomain` axis counts;
- evaluate the exact per-key `arrival_count_by_key` expression for each wait;
- prove a positive static lower bound and a static maximum across all keys;
- use that maximum as the replay-safe epoch stride;
- initialize a key for the current epoch with the existing atomic-max step;
- publish one arrival per proved producer and wait for that key's exact target;
  and
- decline if the lower bound, maximum, publication relation, or key decoding
  cannot be proved without runtime-domain enumeration.

The epoch stride is allocation/replay framing, not the wait target. Thus a
width-two tail may use stride 32 while ordinary keys wait for 32 and the final
key waits for 16. A final-arrival continuation remains eligible only when its
fan-in is uniform and its existing exact-once consumer bijection is proved.

Normalization work is bounded by relation structure. It may inspect and
coalesce relation pieces and axes, but must never enumerate runtime extents,
readiness keys, workers, waves, or CTAs. Use one shared limit of 4,096 retained
pieces and 65,536 prospective pair/product states per relation operation.
Union, source-cell Cartesian products, composition, L2-order construction, and
coalescing check the budget before materializing the product and decline
conservatively when it would be exceeded. The intended cost is linear or
near-linear in admitted IR size, not in `B`, `Q`, token count, or sequence
length.

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

### Schedule domain

Use existing coordinate-domain machinery to describe:

```text
launch_stage in {source, resident}
0 <= worker < worker_count
0 <= wave < wave_count(runtime_parameters)
```

For each `(launch_stage, worker, wave)`, at most one segment relation is
defined. Its target is the logical task executed in that slot. An undefined
slot is idle. Launch stage zero represents source tickets in dense ticket
order; launch stage one represents persistent resident worker chronology.
Both are owned by the same `WorkerSchedule`. The source-first global ticket
allocator remains the execution mechanism that gives the launch-stage order
its runtime meaning.

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

## One readiness-driven scheduling policy

The scheduling algorithm operates directly on the existing roots and
`ReadinessGraph` events. It uses unit-weight precedence criticality but does not
construct or retain another dependency graph or estimate hardware latency.
Per-root cursors, per-event claimed frontiers, and temporary adjacency maps are
local implementation state over the existing graph, not another semantic
object.

Constants and guarded symbolic expressions enter this same algorithm. When a
comparison between symbolic frontiers cannot be proved, the algorithm uses the
canonical conservative order; it does not invoke a different scheduler.

If repeated source structure induces an apparent cycle between root and event
relations, prove an affine progress rank directly from those relations. If no
rank is proved, retain the proved sequential order for the affected frontiers.
Never widen a valid recurrence such as `A_i -> B_i -> A_(i+1)` into mutually
dependent whole-root barriers merely because its runtime extent is symbolic.

### Unit-weight structural criticality

Traverse the root/event adjacency already present in `ReadinessGraph`. Root
stages have structural weight one and readiness events weight zero. On its
acyclic condensation, compute:

```text
top(v)    = longest unit-weight path from an entry to v
bottom(v) = longest unit-weight path from v to an exit
horizon   = max_v(top(v) + bottom(v))
slack(v)  = horizon - top(v) - bottom(v)
base(v)   = (slack(v), -top(v))
```

This is the explicit unit-duration model used by the list scheduler. It is not
a learned, profiled, shape-specific, or resource-aware latency model. These
classes are derived properties of `ReadinessGraph` topology and may be cached
only against that graph; they are not another DAG or correctness truth.
Ordinary changes to B, Q, sequence length, or token count therefore reuse the
same structural classes when topology is unchanged.

### Deriving a readiness-major producer order

Event-completion scheduling depends on grouping producer tasks by the consumer
cohort they unblock. Derive that order only by composing existing relations:

```text
consumer scheduled ordinal
    -> consumer logical task
    -> every required readiness key
    -> every producer logical task for those keys
    -> producer-order ordinal
```

Union all producer arms directly from the uncontracted readiness graph. At
this point ownership candidates are ephemeral; neither a continuation
contraction nor a final counter plan exists. When the scheduler evaluates one
event-family action, derive the resident and inline alternatives from these
same orders and commit any selected contraction atomically with placement.
Flatten the consumer/key/local-producer coordinates through the canonical
mixed-radix constructors; do not enumerate concrete tasks.

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

The canonical order is the configured/autotuned intra-root order for that
kernel. The scheduler does not erase it merely to make source forms look alike;
it replaces it only when the relation composition above proves an exact
readiness-major permutation. Different source/configuration forms may therefore
start from different canonical orders while still using the same scheduling
policy.

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
- **admissible**: every required producer is already assigned. A strict-earlier
  rank is sufficient; a same-wave entry or nested wait instead requires no
  producer after the wait on its consumer's own strand, acyclic segment
  precedence, and the resident-capacity proof, so waiting cannot block an
  unassigned producer; and
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

### Event-aware critical-path priority

For each candidate interval, apply its exact claimed contributions to the
existing readiness events and derive:

- which events become complete;
- which consumer cohorts consequently become admissible; and
- for each incomplete outgoing event, the exact remaining producer frontier.

Let an interval that newly releases consumer cohorts inherit their best
structural class:

```text
release_class = min(base(consumer) for newly admissible consumers)
effective_class = min(base(candidate root), release_class)
```

If no consumer becomes admissible, use the candidate root's base class. Choose
candidate intervals lexicographically by:

1. lowest effective structural slack and greatest structural depth;
2. work already admissible at that class before work that can only release it;
3. completion of a readiness event;
4. fewest exact producer claims remaining for an outgoing event;
5. immediate inlet from an exact earlier launch stage and earliest required
   source ticket; and
6. canonical root, key, and task order.

The criticality class protects the structural critical path. Event inheritance
and remaining-claim tie-breaking prevent equal-class producer work from being
spread across many fan-in groups while an almost-ready consumer waits. Neither
uses measured cycles. A symbolic comparison is used only when current guards
prove its ordering; otherwise the canonical tie-break applies. Candidate
intervals end whenever a readiness contribution or source-ticket frontier
changes.

This is the single policy for concrete and symbolic inputs. A newly completed
event releases its consumer immediately into the same ready set; no separate
parameterized recurrence policy decides whether that consumer deserves to run.

### Work conservation

Select intervals until every worker slot in the abstract wave is filled or no
admissible work remains. Preferring a ready downstream root does not create a
barrier: after assigning its available tasks, remaining workers receive other
admissible roots.

For schedules with identical resident task coverage, reject a proposal whose
final occupied unit-task wave is later than the conservative schedule's. For a
continuation candidate, compare complete logical completion ranks after adding
one unit for every inline body on the final-producer strand against the
earliest admissible resident completion. Include the whole continuation chain
and every possible final-arrival winner, not only its first consumer or one
chosen strand. On a proved primary tie, compare the maximum number of resident
readiness handoffs along primary-critical terminal paths; continuation may win
only by strictly reducing that depth. Prefer resident ownership when either
comparison is unproved or the full pair ties. This remains a structural
topology objective, not a measured makespan or latency estimate. A source-ticket
proposal is compared separately because it intentionally changes resident
coverage. With no provably lowerable
fine-grained prerequisite there is no early-admission opportunity, so the
canonical compact order is returned without stepping through its frontiers.

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

### Schedule-derived nested frontiers

Nested-loop readiness is the one synchronization decision that must follow
placement. `ReadinessGraph` retains the exact per-iteration relation throughout
scheduling; the compiler does not first replace it with an entry barrier or a
sampled partition.

For each nested consumer site, derive with existing relation operations:

```text
producer_frontier(key) = maximum producer wave contributing to key
consumer_wave(task)    = wave owning the consumer CTA
ready_prefix(task)     = iterations whose producer frontier is strictly
                         before consumer admission
later_frontiers(task)  = subsequent changes in producer frontier
```

The concrete implementation sequence is:

1. derive `task -> wave` once from the producer root's final
   `WorkerScheduleSegment.task_order`;
2. take the exact converse of `task -> readiness key` to obtain
   `key -> producer tasks`;
3. use `CoordinateRelation.max_target_value_by_source` to maximize the wave
   over each producer fiber;
4. compose the nested consumer's `iteration -> key` relation with that
   `key -> latest wave` result;
5. compare it with the owning consumer task's admission wave; and
6. form maximal adjacent iteration intervals at the resulting piece
   boundaries, then feed those boundaries to the existing segmented-counter
   constructor.

The missing Qwen proof is specifically step 3 for a bijective mixed-radix task
permutation followed by a fixed-width quotient. Extend the existing converse,
composition, and extremum routines for that algebraic form; do not add a Qwen
matcher or enumerate its 96 iterations.

Factor any schedule-polymorphic outer coordinates first. For example, when a
dependency and schedule are `Identity(B) x inner_relation`, compute the finite
piecewise frontier over `inner_relation` once and lift it through `B`. The work
must scale with relation rank and piece count, never with `B`, CTA count,
worker count, or the numerical nested-loop extent.

Use the maximal adjacent iteration intervals on which the frontier relation is
identical. These intervals become the existing
`ReadinessCounterPlan` key partition through
`_segmented_nested_loop_counter`; no new nested-schedule object is introduced.
Qwen's mixed-radix relation yields two intervals, `[0, 74)` and `[74, 96)`,
because those are precisely the points at which the latest producer wave
changes relative to the consumer. The policy contains neither 74 nor 96.

The first emitted wait must admit the consumer safely. Later waits may be
satisfied by producer tasks in the same global wave on different resident
worker strands. Such a schedule is legal only when the existing segment
relations prove all of the following:

- the first required prefix is complete before consumer admission;
- every producer needed by a later checkpoint is already assigned;
- no such producer is scheduled after the wait on the waiting consumer's own
  worker strand;
- the symbolic segment-precedence quotient is acyclic; and
- the backend resident-capacity certificate covers every involved strand.

If an optimized placement relies on a nested frontier whose maximum/composition
is not representable, reject that complete proposal. The single all-resident
root-major fallback then derives a root-entry counter or root barrier from the
unchanged semantic event. Do not retain the placement while silently
coarsening the wait it relied on.

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

A strictly earlier producer ownership rank is the simple sufficient case for
any resident wait:

```text
(launch stage, wave, worker-local step)
```

It is not a universal requirement. Both a root-entry wait and a later nested
checkpoint may consume work completed in the same wave by another resident
strand. The common proof uses the exact `ReadinessGraph`, selected ownership,
and final `WorkerSchedule`, and requires every producer to be assigned, no
required producer after the wait on its consumer's own strand, an acyclic
symbolic segment-precedence relation, and resident capacity for all involved
strands. Runtime waits provide visibility. This is required both for Qwen's
final 22 producers and for Gemma reductions placed in unused lanes of the
down-projection tail wave.

Every blocking resident schedule therefore requires two capacity gates. A
conservative pre-codegen certificate uses configured resource bounds; after
backend compilation and before cache acceptance or launch, actual cubin
metadata must confirm the promised residency. Initially both gates require:

- `worker_count == visible_sms * required_blocks_per_sm`;
- no SMs are reserved from that count; and
- occupancy of at least `required_blocks_per_sm` resident CTAs per visible SM.

All producer worker strands covered by the rank proof can then become resident
concurrently. This is a legality requirement, not a performance cost model.

### Final-arrival continuations

A continuation is selected only when exact-once and body-safety proofs succeed.
Its sole finalized identity is the existing
`ReadinessCounterPlan.continuation_consumer_index`; any candidate object used
during selection is ephemeral and must not become another plan or IR.
A continuation action is legal under the same rule for constant and
parameterized extents and requires:

- an exactly lowerable producer-to-key publication relation;
- a bounded positive arrival count for every nonempty key;
- an exact one-consumer-task-per-key mapping;
- a body that may execute safely in the final producer's context;
- no nested consumer wait or ownership cycle; and
- exact downstream publication after contracting this consumer into its
  producer chain.

After constructing a complete provisional all-resident schedule, compare each
eligible continuation action against that placement using the lexicographic
pair of complete unit-weight completion makespan and critical-path resident
handoff depth. A possible final publisher is a causal maximum of the event's
producer tasks, not merely the producer with greatest wave or completion.
Every inline alternative receives the event maximum over all required
producers and then charges one unit for the consumer body on that publisher's
strand; an inline chain costs one unit per body, including all later work on
the affected strand. A resident cross-owner readiness wait adds one handoff at
its point on a primary-critical terminal path. Inline may win only when its
primary makespan is proved no worse and, on a primary tie, its maximum handoff
depth is strictly lower. Prefer resident ownership on an unproved comparison
or a full pair tie. At most one consumer globally may own a final arrival, and
one choice must hold for the whole event family and compiled guard. If a
continuation wins, omit the consumer from resident placement and contract
downstream readiness in that same atomic scheduling action. Otherwise it
remains ordinary resident work. A parameter symbol, sink status, sampled task
count, or `fan_in > 1` is not an eligibility or priority rule.

The secondary handoff objective is necessary rather than optional decoration:
for a fixed provisional placement, resident and inline ownership can have the
same unit-weight makespan even though inline execution removes a cross-owner
synchronization boundary. A rule that gave resident ownership every such tie
could therefore miss a structural benefit. Handoff depth is derived solely
from readiness topology and ownership, contains no measured latency, and
distinguishes intermediate work that would delay a producer strand from
terminal work that removes a synchronization edge at equal makespan. This is
not a claim that arbitrary or greedily constructed resident placement always
dominates inline execution.

The selected identity is copied exactly once into the final counter plan after
the schedule is accepted. Proposal, proof, and diagnostics consume that same
selection; codegen only renders it. There is no second continuation pass.

This is distinct from resident placement but not from the scheduling decision.
Resident placement reserves a future worker/wave slot and emits a wait before
the body. A continuation uses no resident slot: the producer observing the
final arrival calls the consumer immediately. Gemma demonstrates why these
alternatives must be compared jointly: its packed resident reductions occupy
free tail-wave lanes, while forcing them inline adds serial work to producer
strands.

### Generalizing the existing transient source

Retain the current source-ticket mechanism during the first scheduling phases.
Every non-continuation task relation, including a source-ticket relation, is
stored exactly once in `WorkerSchedule.segments` and distinguished by its
launch-stage support. Source identity is derived from the unique
launch-stage-zero segment; an independent `transient_source_root` field is
migration debt and must be removed once callers consume that derived property.
Continuation ownership remains authoritative in the selected counter plan.

Generalize source selection by replacing the current model-shaped root
assumption with predicates over that existing source segment relation.

The model-independent eligibility conjunction is:

- one wait-free source order with an exact task-to-ticket bijection;
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

Static schedules derive concrete participant intervals. Commit `a9df42d1`
also derives exact rotated active-worker support for the recognized packed
parameterized root-major form instead of publishing from every worker. The
remaining requirement is to derive that support from arbitrary authoritative
schedule-relation projection, not from a root-major recognizer or codegen
reconstruction.

Generalize the existing `RootBarrierPublicationPlan`; do not add a symbolic-
barrier object. For every root, the plan derives from authoritative
`WorkerScheduleSegment.task_order`:

- each worker or continuation execution that owns root work;
- that owner's final publication occurrence;
- exact real arrival count `A_real(p)`;
- exact effective arrival count `A_eff(p)` after empty-root handling;
- a static maximum `M`;
- the unit contribution at every publication site; and
- one synthetic, correctly ordered completion owner when the real task support
  is empty.

The common semantic protocol for constant and parameterized barriers uses
uint64 bounded epochs. At launch epoch `e`, each exact owner advances the
counter to at least `e*M`, then contributes one release arrival; consumers
acquire-wait for `e*M + A_eff(p)`. Constant cases may strength-reduce this only
after proving equivalence. The proof must establish
`A_eff = max(A_real, 1)` and `1 <= A_eff(p) <= M`. Alternating large/small
shapes must remain replay-safe without a reset kernel.

For the packed root-major certificate with first slot `F`, task count `T`, and
worker count `W`, the compact rendering may use rotated ordinal
`j = (worker + W - (F mod W)) mod W` and membership
`j < min(T, W)`. This formula is accepted only after proving it equals the
segment relation's projected support; codegen does not rediscover it.

When `T=0`, that predicate has no real owner. The same plan must derive and
store the synthetic owner's launch-stage/wave/strand occurrence from the
authoritative schedule relation and root ordering; `F mod W` alone identifies
a worker but not chronology. That occurrence performs every incoming wait,
executes no body, and publishes completion. Until this derivation exists,
continuation- or source-only ownership is rejected for a root that may be empty
and has downstream obligations.

Participant support and contribution are represented by existing
`CoordinateRelation` and symbolic-expression fields generalized within
`RootBarrierPublicationPlan`; concrete `participant_intervals` are only a
certified compact form. Final occurrence is selected by launch-stage/wave
chronology, never segment tuple order. Resident, continuation, source-stage,
and synthetic publication arms must be disjoint, their union must yield
`A_eff`, and `M` must bound that complete union. Every required bound is an
action-legality condition proved before joint scheduling. If it is unavailable, that
candidate is not admitted; no late publication-plan failure mutates ownership.
A failure while validating a complete proposal discards it and rebuilds the
all-resident fallback once.

There is exactly one publication plan and counter per unique producer root,
regardless of how many outgoing barrier edges consume it. Shared counter state
is not reused by overlapping launches. The implementation must document and
test the serialized-launch contract plus the uint64 epoch overflow/reset
horizon.

A weighted fixed-total publication is only an optional strength reduction of
that same plan. It is not a second barrier policy and is not retained merely
because a diagnostic probe used it.

No dynamic-claim root aggregation is introduced in the initial design.

Required relation-level tests cover rotated `F`, wrapping support,
`T={0,1,W-1,W,W+1}`, multiple root occurrences, multiple outgoing edges,
source-stage and continuation publication arms, and
`nonzero -> zero -> nonzero` replay. Production proof may manipulate symbolic
worker-count expressions but never enumerates workers, waves, tasks, or runtime
shape values.

## Symbolic proof contract

No optimized schedule reaches codegen unless every relation-level property
below holds for every runtime parameter satisfying the schedule guard. The
backend-residency portion is necessarily checked after compilation but before
the artifact is accepted into the executable cache or launched.

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
counter relation. Coverage is computed from mechanisms in the finalized plan;
separate tests verify that codegen renders those mechanisms exactly.

### Resident rank progress

For every wait edge, first try the sufficient strict-order proof:

```text
max(producer ownership rank) < consumer ownership rank
```

The rank is derived once from `WorkerScheduleSegment.task_order`. When a
root-entry or nested wait shares a wave with producers, use the common stronger
proof instead: all required producer tasks are assigned, no producer lies
after the wait on the consumer's own worker strand, the symbolic segment-
precedence quotient is acyclic, and resident capacity covers all strands.
Nested proofs use all required keys, not merely the first checkpoint. These
bounded relation proofs replace the quadratic segment-pair validator for
production schedules.

### Source progress and capacity

Source-to-resident progress uses lexicographic launch-stage/ticket rank plus
the same two-part capacity gate. The pre-codegen symbolic certificate requires:

- `W == visible SM count`;
- no reserved SMs;
- configured resource bounds permitting at least one resident CTA per SM; and
- the proved source-first ticket allocator.

The backend then verifies actual compiled occupancy before accepting the
artifact for launch. These facts establish progress, not predicted
performance.

### Proof restrictions

- Proofs use relations, parameter guards, and interval frontiers.
- Production proof never enumerates tasks, workers, keys, or shape values.
- Unsupported operations decline conservatively.
- Concrete CTA DAGs exist only in tests as differential oracles.
- Random substitutions test implementation correctness but are not proofs.

## Avoiding duplicate computation inside the compiler

The following values must each be derived once and cached:

- root task-order relations and exact converses;
- the derived emitted-prerequisite iteration over final plan fields;
- symbolic worker/wave rank;
- structurally selected continuation ownership and its single downstream
  contraction, copied without reselection into retained counter plans;
- each nested producer-wave frontier and its final schedule quotient;
- root participation, final publication support, exact/effective arrival
  counts, epoch bounds, and per-site contribution;
- unit-weight `top`, `bottom`, and slack derived directly from
  `ReadinessGraph`; and
- schedule relation support per root.

Proposal, proof, codegen, and diagnostics consume these caches. No component
reconstructs a CTA DAG or independently interprets nested readiness.

## Probe requirements

Historical timings below are directional evidence from experimental
worktrees. Every implementation phase must remeasure same-source controls with
identical numerics, fusion boundaries, resource settings, and cache handling.

For every required source form, record its source hash, direct invocation
command, numerical comparison, cubin hash/reuse claim, selected schedule,
counters and barriers, register/spill/shared-memory/warp envelope, matched
standalone, and current-main result. An untouched pretuned control must invoke
the checked-in function directly; a rewritten decorator or body is a separate
source.

Before implementation, freeze a no-regression band for each control. Unless a
workload-specific target below is stricter, the untouched pretuned kernel may
not regress by more than `max(2 us, 2%)` versus its pre-refactor current-tree
and current-main controls. A symbolic replay kernel must come within
`max(2 us, 3%)` of its identical-source exact-shape control while retaining one
cubin. Any exception requires a documented causal profile and explicit review;
it cannot be hidden by comparing a different source or configuration.

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

Keep the two established source/topology families separate:

| source/topology | shape | persistent | Helion serial | Helion radix | ThunderKittens raw |
| --- | --- | ---: | ---: | ---: | ---: |
| canonical Q4 | B4 | 61.184 us | 83.744 us | 67.360 us | 71.456 us (invalid output) |
| canonical Q4 | B9 ragged | 90.016 us | 94.080 us | 100.224 us | 88.128 us (invalid output) |
| dynamic F64/Q1 | B1/B2/B4/B9 | 57.12/61.22/98.08/171.81 us | 95.84/98.05/136.96/225.41 us | n/a | n/a |

The long-context Q1 control is 38.656 us persistent versus 36.768 us
standalone, and Q2 is 30.464 versus 30.496 us. Do not substitute the dynamic
F64/Q1 source for the canonical Q4 ThunderKittens comparison. The current
ThunderKittens output is nonfinite, so its numbers are performance-only until
numerical validation is repaired.

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

Two source forms are mandatory and complementary:

1. the untouched checked-in pretuned kernel, with its native fixed shape,
   source-level specializations, and production configuration; and
2. the mechanically related ragged B1/B2 probe, whose batch and context
   expressions exercise schedule reuse and one-cubin lowering.

The first guards the established production writing style and fixed-shape
performance under the unified policy. The second validates symbolic reuse.
Both must pass through the same semantic readiness/synchronization/scheduling
policy; a win on one does not excuse a regression or different policy on the
other. Invoke the pretuned function directly and record its source hash; a
rewritten decorator or body is not the untouched control.

Required validation:

- B1 no regression;
- B greater than one with one short and one long context;
- uniform B/Q/maximum-context changes across declared polymorphic guards;
- exact QKV/attention/reduction/O-projection handoffs;
- compilation through the same event-frontier scheduler; and
- no ready downstream attention work delayed behind equal-class ancestors;
- readiness-major producer ordering either proves a replacement from existing
  task/event relations or preserves the canonical configured root order. The
  untouched fixed-B1 kernel uses root-5 order `[2,1,0]`; the ragged B2 probe
  intentionally uses `[2,0,1]`, and those orders coincide when the batch axis
  is degenerate at B1; and
- compare exact B2 task order plus continuation/counter/barrier decisions after
  symbolic substitution; require equality when the full normalized guard,
  configured order, worker count, and capacity facts are equivalent, and
  otherwise explain the differing proof fact rather than merely listing the
  same counters.

Before ragged metadata scheduling, mixed contexts use a canonical max-domain
masked schedule. They do not claim globally optimal per-request ordering.

The historical approximately 94-us result has now been reproduced from its
byte-identical generated Triton at 94.080--94.224 us under cold-L2 timing. The
current untouched lowering measures approximately 100.256 us in the matched
isolated protocol; a larger multi-kernel interleave adds instruction/cache
pressure and is not the causal comparison. Preserve both standalone split
conventions explicitly: the 128-split standalone is the numerically matched
control, while the 32-split path is retained only as the production-performance
reference.

A direct synchronization/placement ablation identifies the entire regression.
The historical plan places root 13 on workers 576:672 and root 14 on
672:1184, then derives two nested keys with fan-ins 74 and 22. Collapsing only
those keys to one fan-in-96 wait while retaining placement measures 96.128 us.
Removing nested placement as well moves root 13 to 1088:1184 and root 14 to
0:512, increases spills, and measures 100.288 us. Thus approximately 2 us is
the streaming frontier and approximately 4 us is complementary same-wave
placement/code shape. The unified scheduler must preserve both from the exact
relation; restoring only the counter after choosing a coarse schedule is not
sufficient.

Scheduler A/B controls retain the identical persistent fusion boundary.
Standalone controls intentionally change launch fusion while preserving each
stage's arithmetic and reduction order; they measure launch/tail benefit, not
schedule-policy equivalence.

Do not introduce a Qwen-specific "parameterized local schedule." First
separate three effects using the identical ragged source and configuration:

1. exact-shape Helion lowering;
2. parameterized Helion scheduling with ordinary Triton specialization; and
3. the same parameterized schedule with one-cubin Triton specialization.

The reflected mixed-radix converse repair now lets the exact-shape ragged
control compile at 104.35/128.90 us for B1/B2. The original approximately
94-us fixed-context kernel remains a separate production guard because the
ragged source contains runtime `context_lens` guards and tail masks.

`Identity(runtime_axes) x inner_relation` may be used as a common relation
normalization for both constant and symbolic extents. It must never select a
“parameterized local schedule.” Any resulting task order is chosen by the one
event-frontier policy, stored directly in
`WorkerScheduleSegment.task_order`, and tested on non-Qwen relations. If the
common transformation cannot be proved, retain the common conservative order
rather than adding a model matcher, schedule catalogue, sampled-size template,
or parameter-only escape hatch.

### Gemma 4 A4B MoE

Use the checked-in pretuned fused hierarchical kernel. The current slot-dense
boundary has work domains determined by token slots/top-k; expert IDs primarily
affect addresses. It is therefore suitable for parameterized relations even
when inputs route to different experts.

As with Qwen, validate two source forms rather than treating the symbolic
batch rewrite as a substitute for production source:

1. the untouched checked-in pretuned fused kernel and configuration; and
2. the mechanically related symbolic B1/B2 probe with varied expert routing.

They must traverse the same structural policy. Identical decisions are required
only when normalized relations, guards, configured task order, worker count,
body-safety facts, and capacity facts are equivalent. Differences in emitted
loops are acceptable only as proved renderings of a finalized plan. Invoke the
pretuned function directly and record its source hash. Compile an identical-
source exact-shape control for each symbolic B1/B2 case.

Required validation:

- B1 no regression;
- B2 and larger token counts;
- inputs routing tokens to distinct experts;
- exact woven mixed-radix down-projection inverses;
- final-arrival ownership; and
- one binary across dimensions explicitly declared schedule-polymorphic.

This is mainly a generality/parity case. The untouched pretuned B1 source on
the current tree measures 49.024 us persistent versus 53.168 us matched
eight-launch standalone and 49.120 us persistent on clean main; all seven
outputs are bit-exact and all eight routed assignments select distinct experts.
Its global proposal is neutral (51.296 us versus 51.136 us disabled), so it is
a negative control for global reordering. The old continuation result is not a
general requirement: its benefit depends on the worker schedule around it.

A controlled same-resource ablation establishes the actual ownership effect.
At m2, the one-cubin dynamic schedule measures:

| ownership | B1 | B2 |
| --- | ---: | ---: |
| resident root 6, root 7 continuation | **61.408 us** | **77.792 us** |
| root 6 continuation, root 7 resident | 63.568 us | 83.936 us |
| full 5→6→7 continuation chain | 65.472 us | 84.064 us |

Every variant is R126, zero spill, 34,816 bytes shared, W4, one-cubin, and
bit-exact. Root 6 occupies complementary tail-wave lanes in the resident plan:
workers 181–191 after root-5 workers 0–180 at B1, and workers 66–87 after
root-5 workers 0–65 at B2. Inline ownership serializes each reduction onto the
runtime final producer. A separate exact B2/m2 control recovers
83.808→75.648 us by retaining root 6 resident with identical R116 resources.
Changing all synchronization atomics and loads from u64 to u32 changes latency
by at most 0.032 us, ruling out epoch width as the cause.

Worker-width tuning then recovers most of the historical gap without changing
ownership. Under the 10-second warmup/120-sample cold-L2 protocol, m3 gives:

| shape | one-cubin dynamic | exact packed | standalone |
| --- | ---: | ---: | ---: |
| B1 | 53.216 us | 51.168 us | 55.264 us |
| B2 | 75.712 us | 71.712 us | 69.888 us |

The dynamic binary is R126/zero-spill/34,816-byte-shared/W4 and all outputs are
bit-exact with 15 distinct B2 experts across 16 assignments. At B2 the exact
and dynamic plans have identical W444 placement and ownership, so their
4.000-us difference is symbolic/state rendering overhead. At B1 the historical
49.120-us exact control remains 2.048 us faster than exact packed and 4.096 us
faster than the polymorphic binary. The unified compiler must preserve the
historical exact case when its narrow guard proves inline execution better,
keep resident ownership for the B1/B2 guard when that is the uniform safe
choice, and separately remove avoidable symbolic rendering overhead. It must
not force constant/symbolic ownership parity or add a Gemma rule.

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
154.0 us, so body/resource work remains separate. The current dynamic probe is
a negative control at 542.752/886.816 us versus 340.000/564.704 us for its
matched serial dynamic standalone and spills 354 bytes at R255. A same-source
static persistent control does not compile. Do not attribute that body/code-
shape failure to scheduling or claim dynamic parity from the static branch-
packing win.

### Nemotron MoE

Nemotron provides both a static branch-packing win and a boundary on purely
symbolic scheduling.

For the specialization-known routed-first probe, existing relation pieces
should reproduce the generic opportunity of placing shared-down work into an
underfilled routed-up wave. Historical evidence was approximately 98.3 us
versus 118.8 us on clean main. The current one-cubin dynamic probe measures
260.128/337.952 us versus 188.448/243.712 us for its matched overlapping
dynamic standalone; a same-source static persistent control is unavailable.
It is likewise a dynamic body/lowering negative control, not evidence that
branch packing failed or that dynamic parity has been reached.

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

Current dynamic B1/B2 measurements are 1579.104/3110.080 us persistent versus
1649.600/3244.000 us matched standalone with the exact 32/16 tail and one
cubin. This is positive evidence, not a completed gate: retain an exact-shape
static control, run B4 with the exact counter, and A/B the event-aware order.
The earlier static compile exceeding 150 seconds fails the numeric compile
budget even if runtime performance is retained.

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
- root/event adjacency read directly from `ReadinessGraph`;
- unit-weight `top`, `bottom`, slack, and base class;
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
- final-arrival continuation ownership while retaining the same exact event;
- nested frontier coarsening while retaining the same final placement;
- nested consumer placement while retaining a conservative entry wait;
- source-ticket selection;
- noncanonical root interleaving; and
- direct parametric lowering in favor of a concrete specialization.

These controls all use the same scheduler and relations. They are not separate
user-visible schedule modes and are removed after rollout.

For each workload separate:

1. schedule ordering gain over an identical persistent body;
2. continuation-versus-resident ownership with producer order, worker count,
   and bodies held fixed and each consumer placed at its earliest legal slot;
3. nested frontier gain with consumer placement held fixed;
4. launch/tail gain over matched standalone;
5. body slowdown from the shared resource envelope; and
6. compile-time/generated-code overhead.

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
- Compute unit-weight structural criticality directly from `ReadinessGraph` and
  combine it with event-release and remaining-claim priority.
- Emit the result directly as existing segment relation pieces.
- Compare against the small concrete oracle.
- Validate FlashMLA, Muse, and Nemotron before dynamic-domain work.

Exit gate: scheduling quality matches the current successful experiments,
piece counts remain bounded, and no production CTA DAG exists.

### Phase 3: symbolic rank proof

- Derive worker/wave rank from the existing segment relations.
- Prove exact ownership and resident admission symbolically. Root-entry and
  nested waits use the same strict-before fast path or same-wave strand/
  precedence/capacity proof.
- Use every exact nested key and the structurally selected continuation
  contraction; validate the finalized counter identity after placement.
- Bypass and then delete the quadratic segment-pair validator.
- Make materialization raise if called inside acceptance.

Exit gate: proof time scales with relation-piece count and canonical Muse N32
compiles within the numeric budget above.

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
Canonical rank-one roots are represented by exact relation pieces for an
initial partial wave, full waves, and a final partial wave, and codegen
strength-reduces that proved relation into runtime-bounded cyclic loops.
Dynamic memory layouts are not
specialized from hints: dependencies coarsen to root barriers, and every
resident worker publishes once per producer root so epoch targets stay fixed
while shapes vary between graph replays. At this checkpoint, exact
parameterized readiness events and recurrence extraction remained later Phase
4 work; subsequent checkpoints below implement the first exact-event and
recurrence slices. Parameterized roots with rank greater than one, L2-permuted
orders, continuations, and transient-source admission still decline rather
than relying on a shape hint.
The binary-reuse regression explicitly opts its runtime extent out of Triton
specialization; parameterizing the schedule does not override backend
specialization policy for ordinary scalar arguments.

Implementation checkpoint (2026-09-09): the same parameterized root-major
schedule now consumes ordinary `ReadinessGraph` events when the existing
relations prove a rank-one positional bijection on both sides and constant
fan-in one. Those events lower through the existing `ReadinessCounterPlan`,
release `atomic_xchg(counter[key], epoch)`, and acquire wait for that epoch;
unsupported layouts, partial relations, nested waits, and wider fan-in retain
root barriers. Fixed cumulative root-barrier sections precede parameter-sized
epoch-counter sections, so changing exact-event sizes cannot relocate barrier
state across replay. Tests cover zero, shrink/grow, worker-count boundaries,
two moving symbolic counter sections with equal aggregate storage, and one
compiled cubin. This established parameterized exact synchronization; the next
checkpoint adds the first cross-root repeating wave relation for immediate
per-key overlap.

Historical checkpoint (2026-09-09, removed 2026-09-10): a unique topological
chain of two or more equal-size canonical rank-one roots was once closed into
the following parametric event-frontier recurrence when every emitted
prerequisite was a positional fan-in-one counter. For `L` roots, `W` workers,
and runtime task count `N`, phase `p` owned:

```text
task = (wave // L) * W + worker
wave % L = p
```

with `L * ceildiv(N, W)` symbolic waves. Review later established that the
already-supported packed root-major relation holds the same tasks in
`ceildiv(L*N, W)` waves (six versus four for `L=3`, `N=5`, `W=4`) and compiles
at the prior baseline cost. No runtime benefit justified the separate
recognizer or renderer, so production selection, lowering, and their private
helpers were deleted. This history is retained only to prevent reintroducing
the same schedule as a special case.

Implementation checkpoint (2026-09-09): parameterized readiness also accepts
the exact rank-one partition

```text
key k -> producer tasks [F*k, F*k + F)
```

when `F` is a positive compile-time integer and the producer extent is proved
to be exactly `F*K`. The same relation derivation produces both the converse
publication `producer p -> floor(p/F)` and constant arrival count `F`; the
scheduler does not re-match the affine formula. For `F > 1`, one positional
sink task per key is removed from resident placement and executed through the
current candidate-continuation helper by the producer observing the final
arrival. Priority 2A folds that candidate into
`ReadinessCounterPlan.continuation_consumer_index`, the sole finalized
continuation identity. The unequal `F*K -> K` topology deliberately remains
root-major and does not introduce another scheduling or ownership policy.

All parameterized exact counters share a 64-bit state allocation containing
fixed per-worker epochs followed by aligned readiness sections. If `M` is the
maximum compiler-proved static fan-in in the plan, launch epoch `e` uses base
`e*M` and event target `e*M + F`. Fan-in one publishes the target directly;
wider fan-in first raises a possibly stale or relocated word to the epoch base
and then adds one arrival. This is replay-safe across empty, shrink/grow, and
moving counter sections with different fan-ins without a reset kernel or a
host-generated schedule. At that historical checkpoint root barriers remained
in separate 32-bit state; commit `a9df42d1` superseded it with the common
bounded-uint64 root-barrier protocol. The serialized-launch and overflow/reset
contract must explicitly bound `epoch*M`; “effectively unreachable” is not a
proof. Tests cover `K=0/1`, worker-count
boundaries, alternating CUDA graphs, multiple moving sections, different
fan-ins, and one-cubin reuse.

Implementation checkpoint (2026-09-09): the same parameterized counter path
now accepts multiple ordinary consumers per readiness key. The exact symbolic
shape is `F*K` producers, `C*K` consumers, producer partition
`k -> [F*k,F*k+F)`, and consumer quotient `c -> floor(c/C)`, with static
positive `F` and `C` and exact extent identities. Final-arrival continuation
admission remains one-consumer-per-key, and the equal-size event-frontier
recurrence remains fan-in-one only. Thus this broadens synchronization without
silently broadening either execution-ownership optimization.

Flattened scratch analysis was extended in place to carry a static divisor on
an otherwise existing affine term. Only direct, unshifted scalar
`tile.id // constant` is admitted. The resulting `CoordinateRelation` remains
the sole dependency truth. Existing relation simplification now canonicalizes
integer floor/ceiling endpoints, removes a modulo only when symbolic bounds
prove its dividend lies in range, coalesces adjacent target boxes through one
shared helper, and discards only pieces proved wholly outside the target
domain. These operations reduce the dynamic F64/C16 MLA dependency to one
root relation without enumerating B or introducing another graph.

The self-contained Q1/H16/D512 probe preserves F64 H16xN128 partials, the
H1xD512 two-pass C8 reducer, FP32 scratch, BF16 output, and identical
standalone arithmetic. On physical GPU 6, cold-L2 medians for B=1,2,4,9 were
67.36/69.38/112.42/200.50 us persistent versus
96.03/98.08/136.99/227.04 us matched two-launch Helion. All outputs were
bit-exact and all four shapes reused one cubin. This validates the generic
fan-out synchronization slice. At that checkpoint its worker schedule was the
wave-aligned parameterized root-major relation, not a symbolic event-frontier
recurrence for unequal root extents.

Implementation checkpoint (2026-09-09): parameterized root-major ownership
is now globally slot-packed instead of rounding every root to a fresh worker
wave. For root `r`, task `t` owns the single global slot

```text
slot(r, t) = sum(task_count(i), i < r) + t
worker = slot % W
wave = slot // W
```

This orders global slots but does not by itself prove completion order across
different workers. It fills lanes that would otherwise be idle in the
preceding root's final partial wave only when the common same-wave proof shows
that all producers are assigned, none lies later on the waiting strand, the
segment-precedence relation is acyclic, and every strand is resident. The
existing `WorkerScheduleSegment.task_order` relation is the sole ownership
truth; a compact rotated worker-strided loop is merely a proved rendering of
that relation. If the same-wave proof declines, the fallback wave-aligns that
boundary. There is no new schedule object, model gate, admission width,
latency estimate, or host schedule.

A more aggressive key-major ablation was rejected despite greater overlap: it
measured 120.54/266.02 us at B4/B9 because reducers displaced later producer
work and stretched the producer. Slot-packed root-major instead measured
57.12/61.22/98.08/171.81 us at B1/B2/B4/B9 on physical GPU 6 versus
95.84/98.05/136.96/225.41 us for matched standalone. GPU-7 Gantt traces show
38.78 us of B4 overlap and 39.46 us of B9 overlap while preserving producer
throughput; outputs are bit-exact and all shapes reuse one cubin.

### Phase 4A: unify semantic policy before wider rollout

The preceding checkpoints proved useful representations but left a temporary
policy split in `build_static_pipeline_plan`. Remove it before claiming the
redesign complete:

#### Phase 4A.1: normalize schedule construction

The 2026-09-09 architecture re-review approves moving proofs earlier, with
one important constraint: the forward `CoordinateRelation` remains the sole
semantic truth. An exact converse is only a derived memo on that same immutable
relation. It is not a constructor argument, caller-provided certificate,
boolean assertion, second relation IR, or schedule representation. Presence of
the memo never constitutes acceptance by itself; validation still checks the
required forward and converse properties.

The actual proof boundaries are:

| semantic boundary | required property |
| --- | --- |
| configured root task order `Q` in `_validate_root_task_orders` | true bijection from configured task coordinates to every logical CTA |
| one `WorkerScheduleSegment` | point-valued exact partial placement with an exact converse; it need not cover the whole root |
| union of all segments for one root in `_validate_normalized_worker_schedule` | true bijection from represented schedule support to every logical CTA, plus separately disjoint schedule support across segments/roots |
| `_logical_task_to_order_ordinal` for one segment | an exact single-valued partial logical-task-to-segment-ordinal converse; totality is not required until segments are combined |
| `_root_task_placement_relation` and `_root_schedule_traversal` | unique total logical-task-to-slot/ordinal lookup after the complete per-root union |
| `_source_segment_ticket_order` | true bijection for the complete source-ticket relation |
| selected final-arrival continuation | true bijection between consumer tasks and readiness keys; producer fan-in may still be many-to-one |
| `RootBarrierPublicationPlan.participant_order` | true bijection from participating worker support to every effective arrival slot, including the synthetic single slot for an empty root |
| readiness publication/count derivation, `_maximum_value_by_key`, producer-frontier calculation, and nested-counter coarsening | exact relational converse and cardinality, which may be set-valued; no bijection requirement |
| task-to-wave maps, dense normalization, overlap/progress checks, cardinality queries, and rendering | only the specific totality, point-valuedness, support, range, or disjointness property consumed at that site |
| positional-bijection recognizers | lowering/strength-reduction eligibility only; never the universal semantic acceptance rule |

Implement the earlier proof without changing those boundaries:

- [x] Add private derived exact-converse memoization to the existing
  `CoordinateRelation`. Exclude it from constructor arguments, equality,
  hashing, serialization, and codegen truth. Only proof-producing relation
  operations may seed a forward/converse pair, with reversed-domain assertions.
- [x] Make `converse()`, cardinality, and
  `is_bijection_from_source_support()` consume an already-derived converse and
  cheap structural lemmas before attempting source-support factorization.
  A cached converse is still accepted as a bijection only when the forward map
  is point-valued and the converse is a total function over the target.
  Likewise, target cardinality may be inferred from a support bijection only
  after those two facts are proved; an exact converse alone is insufficient.
- [x] Construct packed `P` and `P_converse` together, prove/cache configured
  `Q_converse` once before scheduling, and let existing relation composition
  retain `(P;Q)_converse = Q_converse;P_converse`. Do this for both the direct
  packed path and the existing piece-aligned/sliced fallback. Prevent converse
  propagation from recursively invoking itself while building the reverse
  composition or reverse union; propagation may inspect only already-memoized
  converses and must not force a new converse derivation.
- [x] Make existing `then()` and `union()` retain exact converses when both
  operands have proved converses. For union, use
  `(A union B)_converse = A_converse union B_converse`; individual segments
  may be partial. Missing tasks must make the combined converse non-total,
  duplicate ownership must make it non-functional, and overlapping schedule
  support remains an independent rejection.
- [x] Audit every existing transform (`rename`, projection, lift, slice,
  substitution, coalescing, and `dataclasses.replace`). It must either derive
  the transformed exact converse or drop the memo and use the bounded generic
  proof. No result may inherit a converse merely because its Python object was
  copied.
- [x] Keep an extensional, bounded generic fallback for manually constructed
  relations and extensionally equal copies with no memo. Run it only after the
  memo and cheap structural rules, never implicitly from ordinary support-
  cardinality calculation. Construction provenance must affect compile time,
  not which extensionally equal normalized relation within the supported proof
  grammar is accepted.
- [x] Prove canonical/permuted dynamic task orders with coordinate and
  positional-product rules. Prove reflected and woven orders on their bounded
  concrete inner relation, then lift exact dynamic positional axes. Construct
  concrete-radix ragged L2 orders as one algebraic point map with its derived
  two-piece full/tail inverse, then lift dynamic outer axes. Do not retain the
  old `group_count * second_count` construction limit: it budgets an obsolete
  proof witness rather than actual work. Symbolic grouped axes and fully
  symbolic woven/reflected radices may decline until a deterministic bounded
  lemma exists; remove the factorial production search rather than treating
  permutations as a schedule space.
- [x] Preserve runtime-empty roots. Proofs require nonnegative extents, not
  strictly positive symbolic extents, and must validate the synthetic one-slot
  root-barrier participant order without sampling a nonzero shape.
- [x] Update the exact-converse consumers and true-bijection validators to
  reuse the one proof: `_validate_root_task_orders`, per-root `WorkerSchedule`
  validation, per-segment partial ordinal lookup, complete-root
  placement/traversal/ticket lookup, continuation selection/finalization, and
  root-barrier participant validation. Codegen consumes the finalized relation
  and plan and performs no duplicate bijection discovery.
- [x] Add negative tests for omitted tasks, duplicate targets, overlapping
  segment support, out-of-domain maps, padded tails, stale converse memos after
  transforms, and non-bijective participant order. Add positive tests for zero
  roots, multi-segment roots, canonical/permuted/reflected/woven orders, and
  supported L2 with dynamic positional outer axes.
- [x] Once the ordinary packed path uses retained converses, inventory callers
  of `_factored_source_support_converse`, source-support ordinalization, and the
  symbolic mixed-radix search. Demote them to the bounded manual-relation
  fallback or delete them when no independent semantic caller remains.
  `_factored_source_support_converse` remains only as the bounded fallback for
  extensionally constructed parameterized relations and support cardinality;
  `_source_support_ordinalization` remains the shared support/factorization
  primitive; the factorial mixed-radix search has been deleted.
- [x] Rerun focused semantic tests, both full suites, `git diff --check`, and
  the compile-time gate before resuming Phase 4A.2. Record relation/scheduling
  proof time separately. The representative ragged schedule build is 7.1
  seconds, B×Q is 2.6 seconds, and cache-free B×Q proof recovery is 5.5
  seconds; the old six-digit factorial case is 0.12 seconds.

- Move exact symbolic support cardinality, support projection, disjointness,
  coverage, and semantic equality into the existing `CoordinateRelation`.
  Scheduler and barrier code consume those operations; they do not reconstruct
  a canonical root-major relation to compare against.
- Generalize the existing root-major builder to consume configured task-order
  relations for both constant and symbolic domains.
- Normalize legacy concrete segment fields immediately into
  `WorkerScheduleSegment.task_order`; keep the fields only as an input adapter.
- Replace the parameterized early return in `WorkerSchedule.__post_init__`
  with bounded relation validation. Do not accept overlap merely because a
  root-major or event-frontier recognizer matches.
- Handle runtime-empty roots through relation support and the existing
  synthetic root-completion owner rather than sampling a nonzero extent. Store
  its derived occurrence in the existing publication plan.
- Apply the common 4,096-piece/65,536-product budget before union,
  composition, source-cell product, and L2 construction allocate intermediates.

Exit gate: for canonical, permuted, reflected, woven, and supported L2 task
orders, symbolic substitution at `N={0,1,W-1,W,W+1}` is semantically identical
to direct constant construction. Unsupported orders decline before codegen.

#### Phase 4A.2: build readiness once

- Build `ReadinessGraph` once above any constant/parameterized branch.
- [x] Derive structural final-arrival candidates without consulting sampled
  domain size or a provisional worker placement. Validate their canonical
  counter lowering with the common exact-plan predicate, and recover a
  quotient-lowered emitted continuation through the readiness-key domain's
  existing event identity plus exact lowered-relation equality. Do not select
  or contract candidates yet.
- [ ] Keep schedule-dependent progress, final-publication occurrence, and
  synthetic empty-root ownership as transaction-local action-acceptance proofs
  in Phase 4A.3, not candidate properties. Remove the remaining parameterized
  sink-only and `fan_in > 1` selection policy there.
- Reuse the existing relation operations to answer whether a proposed early
  admission has exact publication/cardinality/replay lowering. Do not create a
  capability wrapper or filtered readiness graph. Unsupported events remain
  semantic truth but admit work only through conservative root ordering.
- Validate the emitted counter shape once: consumers in one counter are all
  root-level or all part of the supported nested-site form, and a final-arrival
  continuation is always root-level. Codegen must not repartition a mixed plan
  and silently lose a consumer arm.
- Reject external ownership for a possibly empty root with downstream work
  until its exact synthetic occurrence is proved.
- Delete the parameterized sink-only, `fan_in > 1`, and separate counter-
  support filters. The exact publication/cardinality/replay proof is the
  common action-legality test.

Exit gate: the legality result is invariant under extensionally equal
normalized relations and guards, no ownership has yet changed, and scheduling
cannot accept an action whose eventual publication is unproved.

#### Phase 4A.3: jointly schedule resident and inline work once

Complete these relation-level prerequisites before making continuation
ownership a production decision. They are derived views over the existing
`WorkerSchedule` and `ReadinessGraph`, not new plan or IR state:

- [x] Derive occupied same-strand precedence from
  `WorkerScheduleSegment.task_order`. Construct occupied slot identity from
  each segment and relate slots with the same `(launch_stage, worker)` and a
  strictly earlier wave. It is acceptable initially to retain all strict
  predecessors; immediate-predecessor selection is only a later strength
  reduction. Iterate over roots and relation pieces, never workers, waves, or
  logical CTAs.
- [x] Extend the existing exact extremum machinery to return both the extremal
  value and the exact relation containing every target coordinate that attains
  it. Preserve value/witness correlation across ties, empty domains, dominance
  proofs, and parameter-dependent crossings for the whole compiled guard. A
  single attaining corner or a bare `sympy.Max` is insufficient. Implement
  this as an exact-or-decline operation on `CoordinateRelation`, returning
  `(value_by_source, attaining_targets_by_source)` and sharing the candidate
  collection used by `max_target_value_by_source`; do not add an extrema IR.
  Constant plateaus retain the whole target box, affine extrema retain the
  exact extremal face, and floor/static-quotient extrema retain the complete
  preimage plateau. Cross-piece winners require a bounded exact partition of
  the existing source domain. If a winner changes only with an unresolved
  parameter, or its level set is non-rectangular in the current relation
  grammar, decline for the whole guard rather than sampling or dropping tied
  witnesses. Charge candidate comparison and partition products to the common
  relation budgets.

  Implementation checkpoint (2026-09-10):
  `CoordinateRelation.extreme_target_value_and_attainers_by_source` now shares
  candidate collection with the existing scalar maximum and returns the exact
  extremal value plus every attaining target. It handles constant plateaus,
  affine faces, bounded common-axis winner partitions, and static quotient
  plateaus while declining unrepresentable crossings. Candidate boxes are
  clipped with exactly the same clamp-then-stride semantics as relation
  materialization before either legacy or joint extrema are evaluated.
  Independent differential review covered oversized and negative bounds,
  strides, symbolic substitutions, empty intersections, and proof budgets.
- [x] Add one generic exact weighted scalar pullback to the existing relation
  algebra. Optional target and source potentials are composed around the
  shared exact-extrema implementation, so candidate collection, source-cell
  partitioning, clipping, and all-attainer preservation remain single-source.
  Independent review covered 456,976 concrete lattice intersections, 50,850
  symbolic substitutions, and randomized pointwise-add and end-to-end
  weighted-extrema oracles with no mismatches.
- [x] Derive the occupied strand ordinal
  `q(s) = 1 + |{t: t <strand s}|` from the exact same-strand relation. Holes
  and unoccupied waves contribute nothing. This is an ephemeral scalar
  `CoordinateRelation`, not another schedule field.

  Implementation checkpoint (2026-09-10): arbitrary schedules count the
  exact occupied-predecessor relation. A schedule already proved to be the
  canonical dense packed prefix uses the full same-strand predecessor relation
  as an algebraic strength reduction, but still delegates cardinality to
  `target_count_by_source`; it does not introduce another scheduling policy.
  Symbolic `N`, `4*N+1`, and `(3,N,2)` schedules produce three-piece ordinal
  relations, and boundary substitutions plus randomized packed, holed, split,
  and staged schedules agree with exhaustive materialization.
- [ ] Add one bounded exact-or-decline ranked prefix/recurrence proof to the
  existing relation algebra. A finite number of ordinary compositions is not
  sufficient: even the affine pipeline `A_i -> B_i -> A_(i+1)` has a cyclic
  root quotient and completion growing with runtime `i`. First support an
  acyclic relation-piece quotient. Then support a cyclic quotient only when a
  common well-founded rank, a static translation period, base coverage, tail
  coverage, and the complete Bellman equality are all proved. Encode the
  resulting scalar maps with existing affine/floor/modulo relation pieces;
  do not add a recurrence, state, or schedule IR. Decline parameter-dependent
  periods, unbounded state, nonrectangular partitions, or proof-budget
  overflow.

  Implementation checkpoint (2026-09-10): the acyclic all-resident root
  quotient is complete. It includes every possibly nonempty root-level
  readiness edge and every required cross-root same-strand ordering edge,
  rejects nested/nonresident/cyclic cases, and uses the same deterministic
  topological helper as existing acyclicity and criticality checks. The
  canonical symbolic `(3, N, 2)` packed schedule now proves order `(0, 1, 2)`
  in about 0.08 seconds rather than declining after about 5 seconds. Source
  support is clipped to its domain before nonemptiness testing, fixing phantom
  edges. Independent randomized review found no unsound accepts; the packed
  strength reduction is intentionally conservative and may decline a reverse
  readiness edge between roots that occupy disjoint workers in one wave. The
  scalar acyclic prefix evaluator and cyclic affine recurrence proof are still
  outstanding.

- [x] Let the existing extrema implementation consume a partial scalar-value
  relation when, and only when, its value support provably covers every target
  reachable by the relation being reduced. Keep the public extrema API and its
  candidate/winner/attainer machinery unchanged: normalize the value support
  once, intersect it with each clipped reachable target box, prove those
  intersections disjoint, and prove their exact cardinality equals the whole
  reachable box. Missing coverage, conditional nonemptiness, overlap that
  cannot be normalized, or budget overflow must decline. This is required
  because schedule scores are defined exactly on occupied slots; filling the
  rectangular placement domain or rebuilding `q` from wave arithmetic would
  introduce a second source of truth.
- [ ] Reuse that same exact support-cover proof in
  `CoordinateRelation.pointwise_add_scalar`: a partial right operand is legal
  exactly when its canonical single-valued support covers the left operand's
  support; extra right-hand support is irrelevant. Preserve the left support,
  validate both values and the sum on every intersection, and retain the
  existing total-right-operand fast path. This lets the Bellman evaluator do
  shifted `q` arithmetic directly on occupied slots, rather than adding more
  symbolic `task -> slot -> q` preimage cases or filling holes with fabricated
  values.
- [x] Add one private exact-or-decline partition of a resident slot-to-slot
  dependency relation into same-owner and cross-owner edges. Ownership is the
  existing `(launch_stage, worker)` projection of the two endpoints. Preserve
  symbolic guards and strides, require the two outputs to be disjoint and to
  cover the original clipped edge relation, and charge all splits to the
  common relation budgets. This pair-dependent partition is the only extra
  operation needed for the handoff component; separable source/target scalar
  potentials cannot represent endpoint inequality.

  Implementation checkpoint (2026-09-10): extrema now canonicalize a partial
  scalar map once and accept it only after the disjoint intersections with
  each clipped reachable target box have exactly the same cardinality as that
  box. Scalar values are validated against their carrier even when the partial
  map is already canonical. Independent review covered 6,000 randomized
  max/min calls plus symbolic zero and tail substitutions. The private
  resident-handoff partition proves both endpoints are in the resident stage,
  classifies uniform owner relations, and splits only the exact dense full-
  worker mixed form into `{w}`, `[0,w)`, and `(w,W)`. It does not enumerate
  workers or CTAs; unsupported modular or conditional partitions decline.
  Independent review covered 5,000 concrete relations and symbolic tails with
  no partition, coverage, or owner mismatches.

- [ ] Evaluate the lexicographic objective `(completion, handoffs)` with that
  prefix proof over same-strand precedence and semantic readiness edges. For
  each body `s`, let `R(s)` be the greatest readiness-predecessor score,
  `I(s) = R(s) + (1, 0) - (q(s), 0)`, and
  `D(s) = max((0, 0), max_{t <=strand s} I(t))`; then the body score is
  `(q(s), 0) + D(s)`. Every body contributes `(1, 0)`. A resident readiness
  edge contributes `(0, 1)` exactly when producer and consumer owners differ,
  and only among predecessors tied for maximum completion. Existing extremum
  and all-attainer relations perform each Bellman pullback; the new prefix
  proof supplies only the runtime-length closure they cannot express. The
  evaluator is ephemeral and adds no serialized schedule representation.
  For an acyclic piece quotient, the pair may be encoded as the scalar
  `K*completion + handoffs`, with static `K = quotient_node_count + 1`, only
  after proving a path cannot revisit a charged node and therefore
  `handoffs < K`. A cyclic translation proof keeps two scalar relations and
  performs completion extrema before handoff extrema unless a larger symbolic
  radix is itself proved to stay within the supported expression grammar.
- [ ] Before this evaluator changes production ownership, run the Priority 2B
  concrete oracle on the real FlashMLA, Qwen, Gemma, and Muse graphs. Require
  the objective to reproduce the known positive and negative scheduling
  choices, then require symbolic-substitution parity on every accepted case.
- [ ] Represent an inline consumer body exactly once. Its possible
  final-producer winners are the causally maximal producer on each strand,
  followed by removal of producers proved to precede another candidate for
  the same key. They are mutually exclusive ownership alternatives, not
  simultaneous copies of the body on every producer strand and not simply
  the producers with maximum wave or completion. Each alternative receives
  the event maximum over all required producers, executes after its publisher,
  and contributes the body once. Propagate each alternative through the
  complete downstream graph and compare the guard-wide worst result with
  resident placement.

The scalar global-slot relation used by tail-packing is a sufficient acyclic
progress rank, not this objective. In particular, a producer and waiting
consumer on different workers in the same wave occupy one resident wave but
take two unit completion steps. Root-schema depth and occupied-wave horizon
therefore cannot substitute for the max-plus evaluator.

- Feed the one exact `ReadinessGraph` into one event-frontier policy; reject an
  early-admission candidate unless its required synchronization lowering is
  proved.
- From one complete provisional all-resident schedule, compare each legal
  continuation alternative after charging one unit per inline body on every
  possible final-producer strand. This is not an online decision made before
  downstream placement exists. Compare complete completion horizon first and
  critical-path resident handoff depth second; prefer resident on an unproved
  comparison or a full objective tie, and require one choice for the whole
  compiled guard.
- Commit at most one continuation consumer globally, remove its resident
  coverage, and contract downstream readiness atomically. Score the resident
  plan once and each single-continuation proposal independently; this avoids
  an implicit exponential assignment problem. No other pass reselects it.
- Preserve the configured intra-root order unless an exact readiness-major
  permutation is proved.
- Emit the result directly as `WorkerScheduleSegment.task_order` relations.
- Use root-major ordering as the sole conservative fallback.
- Treat source-ticket admission as another action of this same policy, derived
  from the launch-stage-zero schedule relation. Do not retain a concrete-only
  transient-source selector as a second production policy.
- Keep the old concrete placement algorithm only as a small differential test
  oracle; remove it from production after parity.

The unit-body objective treats a root body atomically. A nested-site wait is a
legality/frontier constraint, but it is not a separate timed body phase until
the compiler has an exact phase model. Any ownership comparison that would
depend on nested-site timing declines rather than silently assigning it a
cost.

Exit gate: the policy contains no parameter-presence branch, model/root ID,
sampled shape, latency estimate, or host-generated runtime schedule. Gemma B2
chooses resident expert reduction and recovers the measured continuation loss;
an exact B1 compile may choose inline only when its narrower guard and final
schedule prove the structural comparison. Qwen retains resident root 13 and
the post-placement 74/22 frontier.

#### Phase 4A.4: derive nested frontiers from the accepted schedule

- [x] Extend existing relation composition/extremum support for a bijective
  mixed-radix task order followed by a fixed-width quotient.
- Factor schedule-polymorphic outer axes such as `Identity(B)` and derive the
  finite inner piecewise frontier once.
- Replace `_split_nested_loop_at_readiness`'s concrete binary search with the
  equivalent symbolic boundary derivation.
- Make `place_nested_loop_consumers` placement-only, then remove or fold it
  once the common event-frontier scheduler owns that placement.
- Retain `_nested_loop_entry_counter` only when the accepted placement was
  already proved safe with an entry wait, or in the rebuilt all-resident
  fallback. If an optimized placement relied on a finer quotient that declines,
  reject the complete proposal rather than changing its wait afterward.

Exit gate: unchanged Qwen source/config recovers both its complementary
producer/consumer placement and the derived 74/22 frontier, with no literal
74, 22, or 96 in scheduling policy. A forced one-key ablation remains slower.

#### Phase 4A.5: finalize and prove once

- Construct final `ReadinessCounterPlan`s only after schedule selection and
  nested-frontier derivation.
- Select root-barrier fallback from uncovered original obligations.
- Derive root-barrier ownership for all schedule forms through the existing
  `RootBarrierPublicationPlan` and bounded epoch protocol.
- Prove exact task ownership, obligation coverage, continuation contraction,
  root-entry admission, same-wave segment progress, the conservative
  pre-codegen capacity bound, and replay safety from the final relations.
- On proof failure, reject the optimized proposal once and validate the
  all-resident conservative root-major result. Same-wave root boundaries use
  the common segment-precedence proof; wave-align only an affected boundary
  that cannot pass it. Never repair a plan by silently dropping an obligation
  or repeatedly changing ownership and placement.

Exit gate: one finalization and validation implementation serves constant and
symbolic domains and is invoked at most once for each of the optimized and
fallback proposals; production proof cost is bounded by graph rank and
relation pieces.

#### Phase 4A.6: make codegen a renderer and delete migration paths

- Remove codegen reconstruction of root arrival counts, continuation task
  counts, event order, and counter legality.
- After backend compilation, verify actual cubin occupancy against the promised
  resident capacity before accepting the artifact into the executable cache or
  launching it. This gate validates the selected plan; it does not reschedule.
- Permit concrete loops, compact recurrences, interval publishers, and fixed-
  count uint32 barriers only as proved strength reductions of the final plan.
- Lower every accepted `WorkerScheduleSegment.task_order` relation through the
  generic relation renderer. Root-major and event-frontier recognizers may
  remain only as equivalent fast renderings, never as admission requirements.
- Derive source-stage identity from the launch-stage-zero schedule relation and
  delete `transient_source_root`.
- Delete the production constant/parameterized and local/global policy split,
  redundant continuation selection, stale nested-plan path, and obsolete
  schedule validators.
- Inventory and fold retained helper dataclasses that duplicate relation or
  final-plan truth.

Exit gate: there is one dependency graph, one readiness graph, one ownership
selection, one scheduler, one finalization pass, one schedule, and one
publication plan per root.

During migration, the current concrete scheduler remains a test oracle and
performance control. It is not an acceptable permanent production alternative
selected by absence of free symbols.

Exit gate: extensionally identical normalized inputs yield identical decisions.
A constant compile with a narrower guard may prove a different choice than a
polymorphic compile, but both traverse the same algorithm and all choices are
uniform over their declared guard. Differences in generated code are explained
by finalized relation rendering or unavoidable body specialization, and both
original pretuned and symbolic probe forms of Qwen and Gemma meet their
performance gates.

### Phase 5: source-ticket generalization after policy unification

- Extend the relation-based source-ticket action already admitted by the Phase
  4A scheduler; do not introduce it here as a separate scheduling path.
- Replace shape/root assumptions with exact source relation predicates.
- Preserve the existing plan and counter abstractions.
- Prove source/resident/continuation disjoint coverage and capacity.
- Delete redundant transient-specific logic only after B4/B9 parity.

Exit gate: FlashMLA B4/B9 retain their gains without an MLA matcher or an
admission-width knob.

Implementation checkpoint (2026-09-09): the existing transient source is now
an exact launch-stage-zero `WorkerScheduleSegment`; there is no parallel source
task-order mapping. Ticket order, count, external source frontiers, source-body
mapping, and root-barrier arrival targets are derived from that relation.
`transient_source_root` currently remains a cached role identity and codegen
rejects disagreement with the unique stage-zero segment. Priority 2A supersedes
that migration state: callers must derive the role from the segment and delete
the duplicate field. The resident stage is unchanged. This checkpoint covers
the proved static source-first behavior; parameterized wider fan-in and ragged
source extents remain outside the accepted subset.

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
- Verify that the old independent concrete/parameterized and local/global
  policy branches were removed in Phase 4A; do not defer their removal here.
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
- Compute unit-weight criticality and event-release priority directly from
  `ReadinessGraph` without another graph object.
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
- Constant/symbolic continuation-selection parity under extensionally equal
  guards/orders/worker facts, plus a narrower-guard case that legitimately
  proves a different choice, a charged chain, and an ineligible competitor.
- No duplicate/omitted tasks.
- Root-entry and nested first-checkpoint admission, same-wave cross-strand
  progress, and same-strand future-producer rejection.
- Schedule-frontier quotient substitution tests, including Qwen's
  mixed-radix/fixed-width form and unsupported nonmonotone fallback.
- Source-ticket admission and replay, including a `P < W` partial-fanout case
  to prove that source size is not an eligibility rule.
- Zero-sized resident, continuation-candidate, and source-candidate roots with
  downstream obligations, including nonzero→zero→nonzero replay and consecutive
  empty roots.
- Dynamic binary reuse.
- Relation-budget rejection before intermediate Cartesian products are built.
- Post-compile residency rejection before cache acceptance or launch.
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
11. continuation ownership and resident placement are selected atomically by
    the same structural rule for constant and symbolic domains; inline bodies
    are charged on every possible producer strand, handoff depth breaks only a
    proved makespan tie, and resident wins unproved or full-objective ties;
12. nested counter partitions are exact quotients of the accepted schedule and
    are never derived from a schedule that a later pass mutates;
13. source tickets use a source-first allocator and capacity certificate;
14. there is one event-frontier scheduler and no concrete/parameterized or
    local/global policy split; extensionally identical normalized policy inputs
    yield the same decisions, while a narrower guard may expose additional
    proofs through the same algorithm;
15. priority uses the documented unit-weight structural model and contains no
    measured/profiled latency, resource estimate, or model-specific rule;
16. FlashMLA B4/B9 retain performance and Q1/Q2 do not regress;
17. both untouched pretuned and mechanically related symbolic B1/B2 source
    forms of Qwen3 and Gemma retain performance through the same policy;
18. DeepSeek-V3 and Nemotron retain branch-packing wins;
19. Muse compiles compactly and exposes event-completion behavior;
20. all primary timings are cold-L2 flushed against matched standalone Helion;
21. standalone-above/persistent-below Gantt charts explain every remaining
    performance gap;
22. `RootBarrierPublicationPlan` is the sole derivation of participant support,
    publication sites, per-site contribution, arrival counts, epoch bounds,
    and empty-root completion; and
23. cold compile-time comparison with current main shows analysis, relation
    normalization, code generation, and Triton compilation remain within the
    numeric budget and do not enumerate runtime extents, workers, waves, or
    CTAs.

The governing invariant is:

> Existing dependency relations define legality. Existing worker-schedule
> relations define exact ownership and order. Existing synchronization plans
> define runtime readiness. Dynamic shapes parameterize those same objects;
> they do not create another scheduler.
