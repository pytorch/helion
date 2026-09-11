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
  masked producer publication, and the remaining cross-workload rollout still
  require work. Cyclic dependency scheduling is outside the current milestone;
  those cases retain the conservative fallback.

Architecture decision (2026-09-10):

- Production scheduling will use one **parametric cohort list scheduler**. It
  is a genuine readiness-driven list scheduler, but its list items are exact
  affine event cohorts rather than materialized CTAs.
- Priority is a finite, shape-independent property of the existing root/event
  topology. Runtime shape expressions determine cohort existence, ready-prefix
  length, packed offsets, and tails; they never participate in a comparison
  that chooses one root over another.
- The production symbolic max-plus evaluator is abandoned. The concrete
  max-plus implementation remains a test-only diagnostic oracle. The generic
  exact relation primitives built while investigating it remain because they
  also serve ownership, progress, frontier, and counter proofs.
- Resident execution is the default. Final-arrival continuation is admitted
  only after a separate local dominance proof covers every possible publisher
  strand and proves that downstream frontiers cannot move later.
- Static extents are constants flowing through this same policy. There will be
  no constant/parameterized or local/global scheduling-policy split.
- The first milestone is acyclic and dynamic-first. Cyclic/SCC scheduling,
  host-generated schedules, runtime instruction tensors, and device work
  queues are not prerequisites.

## Active implementation checklist

This is the authoritative next roadmap. Later historical phase descriptions
record how the current branch was reached, but an unchecked symbolic max-plus
task in those records is not active work.

### Next roadmap: parametric cohort list scheduling

- [ ] Freeze the exact current controls for canonical and dynamic FlashMLA,
  both Qwen source forms, both Gemma source forms, Muse B1/B2/B4,
  DeepSeek-V3 MoE, Nemotron MoE, Qwen FFN, and the full DeepSeek MLA negative
  control. Record worker count, task/cohort counts, exact schedule relations,
  counters/barriers, resources, cold-L2 latency, and aligned Gantts.
- [ ] Remove the abandoned symbolic max-plus evaluator tasks and production
  hooks. Retain the concrete oracle only in tests and retain generic relation
  operations only where another correctness proof consumes them.
- [ ] Derive one finite root/event priority table directly from the existing
  `ReadinessGraph`: least unit-weight slack, greatest structural depth, best
  released-consumer class, event closure, active-cohort continuation,
  launch-stage order, and canonical root/cohort order. No priority component
  may contain a runtime extent or symbolic remaining-work comparison.
- [ ] Derive exact readiness-equivalent cohorts and maximal candidate runs from
  the existing `CoordinateRelation`s. A per-wave candidate/emission interval
  ends at the ready prefix, event-key boundary, relation-piece boundary,
  worker-wave capacity, or root end. Selecting an eligible whole cohort commits
  its complete run; that run may span several emission intervals without
  reranking or interleaving. Do not invent a smaller cohort to fill a lane
  tail.
- [ ] Build the deterministic list policy over maximal affine cohort runs.
  Correctness and proof-budget checks are hard constraints; how far otherwise
  legal ready work may move ahead of canonical order is controlled only by the
  cross-loop pipeline-stage knob below, not by a growing collection of
  profitability rules.
- [ ] Lift guard-uniform repeated decisions by affine translation induction and
  emit them directly as existing `WorkerScheduleSegment.task_order` relation
  pieces. Use bounded prefix/full-group/tail pieces; add no Run, Repeat,
  Region, instruction-stream, or schedule-program abstraction.
- [ ] Make canonical packed root-major behavior the conservative outcome of
  the same scheduler. An unresolved priority tie uses canonical order; unknown
  readiness, ownership, or progress rejects the optimized proposal.
- [ ] Expose one scheduler-specific autotune knob:
  `cross_loop_pipeline_depth`, an integer in the fixed range 1--4.
  Depth one preserves the validated canonical all-resident schedule, including
  any wave alignment required by the progress proof. A cohort
  pulled ahead of canonical order starts depth two; work whose early admission
  depends on pulled work is one plus the maximum predecessor depth. Depth
  resets only when placement exactly rejoins the canonical frontier. The knob
  changes eligibility, never readiness, priority, or counter semantics, and
  does not directly select ownership; one value applies to the whole
  polymorphic guard. Always
  offer the same four values; duplicate schedules are acceptable, just as for
  existing Triton knobs.
- [ ] Complete the resident-only scheduler first. Then add the narrow local
  continuation-dominance proof; do not revive global symbolic completion
  scoring to choose ownership.
- [ ] Derive nested frontiers, counters, root barriers, active participants,
  and replay bounds exactly once from the accepted `WorkerSchedule`. Codegen
  renders those decisions and does not rediscover them.
- [ ] Remove the top-level constant/parameterized policy split and the
  concrete/local versus symbolic/global policy split. Constants are normalized
  as `sympy.Integer` and traverse the same scheduler.
- [ ] Re-express source-first FlashMLA admission as an ordinary launch-stage
  relation and delete the duplicate `transient_source_root` identity only
  after B4/B9 parity is demonstrated.
- [ ] Run substitution and bounded concrete-oracle tests before GPU work. Proof
  and construction time must scale with roots, events, relation pieces, and a
  finite motif, never with `B`, `Q`, task count, worker count, or wave count.
- [ ] Roll out GPU validation in this order: FlashMLA; Qwen; Gemma; Muse;
  Nemotron; DeepSeek-V3. Require numerical parity, one-cubin reuse where
  declared, cold-L2 controls, resource reports, and Gantts at each step.
- [ ] Keep resource tuning separate from scheduling policy. Tune worker count,
  `num_sm_multiplier`, warps, range stages, register limits, and tile choices,
  but do not feed measured latency or a model name into priority.
- [ ] Audit dynamic metadata last. Host-backed `B`, `Q`, and sequence extents
  participate in relation bounds. Device `seq_lens`, expert histograms, and
  compacted worklists remain canonical masked work unless a later measurement
  justifies a separate device-claiming project.
- [ ] Finish with a cleanup audit and independent architecture review: one
  dependency graph, one readiness graph, one scheduler, one worker schedule,
  one finalization pass, and one source of truth for every counter/barrier.

### Completed foundation and remaining proof prerequisites

A checked box below means the code exists and its focused unit tests pass; it
does not replace the roadmap or cross-workload gates above.

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
The active implementation point is now the parametric cohort list scheduler in
the roadmap above. The generic relation primitives remain available, but the
symbolic max-plus scorer and its ranked-prefix prerequisite are no longer
production work. Continue Phase 4A.2 by replacing parameter-only action gates,
then implement resident cohort placement, post-placement nested quotients,
source-stage actions through that same policy, and one final
validation/lowering pass before deleting the top-level
constant/parameterized branch.

The first Phase 4A.2 transaction is now implemented locally. Constant and
parameterized proposals share one finalization boundary for renderer
retention, coverage, emitted continuation identity, exact task ownership,
progress, and final publication occurrence. A rejected optimized proposal is
retried exactly once from the all-resident schedule without reapplying the old
global/transient scheduler. During this migration, the historical dynamic
sink/fan-in/positional continuation policy remains unchanged: candidate
legality is common, but ownership must not broaden until the post-placement
dominance proof exists. Kernel-scope roots are explicitly ineligible for
continuation ownership.

The Phase 4A.3 entry now accepts the authoritative all-resident schedule `C`,
derives guard-uniform semantic root criticality directly from
`ReadinessGraph`, and proves that depth one returns the identical schedule
object for concrete, symbolic, permuted multi-axis, runtime-empty, and tail
shapes without enumeration. Higher depths still conservatively alias `C`;
the public autotune field is intentionally deferred until a production call
can consume it and at least one higher depth can change the accepted schedule.
The next implementation step is exact readiness-equivalent cohort-order
candidate derivation from semantic relations, followed by bounded whole-cohort
placement. No benchmark or codegen path depends on this incomplete entry yet.

That candidate-derivation step is now implemented locally. Configured task
order is composed with semantic readiness in its native multidimensional
coordinates; it is never flattened before the dependency proof. One-key
fibers and separable set-valued fan-out are represented as an exact
`order -> cohort -> event keys` factorization, with the converse of the first
relation remaining the only `cohort -> order` truth. Dynamic full-domain
fibers now use a symbolic fast path in the existing target enumerator, and the
same clipped range drives both its cardinality and decoding.

These relations are ordering candidates only. Original per-arm readiness
relations remain authoritative for admission, arrival multiplicity, closure,
and progress. A root is kept canonical if any of its candidate relations is
unsupported; whole-root barriers, repeated same-root producer arms, nested
producer sites, ambiguous multi-piece intra-cohort order, nonuniform
readiness-major tails, and aggregate proof-budget exhaustion all decline. The
first bounded placement slice is now implemented locally. Every exact cohort
view of a root must agree extensionally on one full readiness-major traversal;
that traversal replaces only the logical-task map over the root's identical
canonical packed slot support. Depth one returns the identical baseline;
depths two through four currently select the same first noncanonical action.
The candidate is accepted only after exact ownership and progress proofs over
the original uncontracted semantic events. Conflicting views or one unsupported
event keep the complete root canonical.

The parameterized renderer now evaluates the accepted
`WorkerScheduleSegment.task_order` relation directly at each proved packed
slot. It does not reconstruct a second dense traversal. Exact symbolic
partitions that cannot be canonicalized into disjoint boxes are renderable only
when the relation has already proved a support bijection. Runtime divisors are
allowed only when they are target-domain parameter expressions that exactly
factor the target-domain size; the bijection then proves they are positive on
every executed point. Empty-domain evaluation is clamped safely, and signed
division uses Euclidean-floor semantics rather than Triton's truncation.

CPU substitution tests cover zero, aligned, and unaligned dynamic extents,
depth aliasing, conflicting cohort views, and unsupported-event veto. A CUDA
regression forces a `W-1` producer prefix and a rank-two dynamic key so the
consumer traversal crosses a worker-wave boundary, exercises both a negative
floor-division numerator and dynamic divisors, and checks numerical output.
It validates rendering at one positive rank-two shape; it does not yet prove
one-cubin substitution across `Q`. A zero-`Q` replay currently faults in the
pre-existing canonical parameterized lowering as well, so runtime-empty
rank-two execution remains an explicit later gate rather than a claim of this
slice.

The first bounded placement change is now implemented locally. At a completed
root boundary with a statically known partial wave, the canonical successor
and every later ready, incomparable candidate are scored by the same
`(effective criticality, releases consumer at that class)` prefix. The
event-closure field remains neutral in this root-only strength reduction:
closure is a property of consumer-relevant event keys, not merely of a set of
producer roots, and will be enabled by the next cohort-frontier slice. A later
root moves only when it is the unique strict winner, its whole root is proved
to be one exact cohort in every semantic readiness view, and it fits the
remaining lanes. A root absent from every semantic event is one neutral
whole-root cohort; a root hidden by an unsupported event is not.
Unclassified or symbolically sized ready competitors veto the proposal rather
than disappearing from the priority set. Static empty roots are the only safe
exception. Root edges and criticality are derived once, reachability is
precomputed once, and rebuilding is bounded by the same aggregate relation
budget used by `WorkerSchedule`.

The lowerable CPU controls now distinguish a real pull, an equal-class tie,
the case where the canonical successor inherits the class of a newly released
join consumer, and independent optimistic priority floors for both the
canonical action and every competing ready cohort. Paired controls remove only
the downstream release responsible for each floor and then require the pull;
this prevents a generic tie or conservative decline from passing accidentally.
A disjoint-key control proves that an unrelated producer arm cannot receive
event-closure credit. A CUDA test executes packed whole-root order `(1, 0, 2)`
through the parameterized relation renderer and final publication bookkeeping.

The first exact sub-root placement slice is also implemented in the scheduler
and relation tests. Every semantic view must agree on the same cohort
partition; its uniform first-fiber cardinality is the only width fact retained.
At a proved static partial-wave boundary, one complete cohort may move ahead of
incomparable roots when its pessimistic known priority strictly beats the
optimistic lower bound of the canonical successor and every other ready
competitor. Width controls eligibility only and never ranking. Root-local
readiness-major permutations are independently progress-validated, then
composed transactionally into the tail refinement; if that refinement fails,
the accepted root-local proposal remains instead of rolling all the way back
to `C`.

The resulting order is represented only as repeated existing
`WorkerScheduleSegment.task_order` relations, for example
`P ; C[0:Wc] ; U ; C[Wc:]`. Exact aligned slices retain explicit inverse
support, adjacent packed intervals are proved from their authoritative
relations, and concrete specialization at `B={0,1,3,4}` selects the same
semantic order after pruning a zero-support suffix. That runtime-empty suffix
is covered by CPU substitution; the compiled CUDA fixture covers B={1,3,4,75}
and worker-wave wraparound. Relation substitution now removes such concretely
empty pieces. Composition clips raw boxes to their
declared domain before taking a preimage; the previous hand-built partial
packed inverse was rejected because copying an implicitly clipped inverse into
a larger target domain would falsely give the segment ownership of the whole
root. No CTA, worker, wave, key, or shape enumeration is used for acceptance.

The relation-driven repeated-root renderer is now implemented locally. It
derives every segment's dense packed slot interval from the authoritative
`task_order`, strides that interval by resident worker count, maps each live
slot back through that same relation, and invokes the existing shared root
body. One compiled kernel has executed the injected `P ; C[0:2] ; U ; C[2:]`
schedule correctly at B={1,3,4,75}; this covers worker-wave wraparound without
shape specialization. Fine-grained counters remain the readiness authority. A
repeated parameterized plan that still needs a root barrier is rejected before
code generation until publication ownership can be derived from the same
relations.

Packed construction now retains private proof certificates for exact dense
source support, cardinality, and single-valuedness. These are caches, not
semantic fields: equality, hashing, and correctness do not depend on them.
Cache loss falls back to bounded relation proof and may conservatively decline
an optional optimized schedule; finalized schedules remain compiler-local
between selection and rendering rather than crossing a serialization boundary.
This avoids
re-expanding the constructor's three `Min`/`Mod`/`FloorDiv` boxes during every
pairwise ownership check; the first renderer test fell from an unbounded
multi-minute proof to an 86-second full generate/compile/run test. The exact
support, disjointness, inverse, total-cardinality, and relation-budget checks
remain mandatory.

`_task_order_slice` now also proves the leading fixed-width cohort and its
remainder for arbitrary-rank mixed-radix traversals. The remainder is an
O(rank) partition by the slowest nonzero outer digit, with an explicit exact
native-coordinate inverse; no runtime extent, CTA, worker, or wave is
enumerated. CPU substitution tests cover an H-fastest `H2 x B x Q` PID-order
traversal. The same construction now covers grouped rank-N orders after
strengthening the shared nonnegativity proof for bounded polynomials over
positive integer shape parameters. The proof rewrites each parameter as
`p0 + 1`, first bounds the possible expansion structurally, and accepts only
when SymPy proves the shifted expression nonnegative. It is used consistently
by coordinate domains and derived support-cardinality certificates. An exact
group-size-two `H5 x 3 x B x Q` control now slices and packs `[0,15)` plus
`[15,15BQ)`, with the latter empty at B=Q=1; merely nonnegative B/Q still
decline because a fixed 15-task prefix would be invalid at zero. This closes
the grouped-rank proof prerequisite without an L2- or model-specific branch.

The first production-wiring foothold is now implemented. Before either legacy
constant/parameterized policy branch, `build_static_pipeline_plan` can run the
same all-resident cohort proposal at an internal depth and finalize a changed
schedule once through the existing counter, coverage, ownership, and progress
transaction. It is currently gated to the semantic subset with no eligible
continuation or nested endpoint, and to callers that have disabled the legacy
transient-source proposal. That last gate is deliberately common to constant
and symbolic inputs: the current transient implementation is concrete-only,
so silently treating a symbolic proof failure as permission to reorder would
reintroduce the policy split. A CUDA integration test reaches the repeated-root
renderer through this production entry by explicitly disabling that competing
migration proposal; it is not yet evidence that ordinary codegen selects the
foothold. Default depth remains one and the field is not exposed publicly while
the legacy ownership/placement split remains reachable.

The next transaction replaces the one-shot tail helper with one finite,
ephemeral cursor/run walk over `C`. It can make several whole-cohort decisions,
recompute newly ready roots after each committed run, propagate causal depth,
and repack exactly once into the existing `WorkerSchedule`; it introduces no
retained state or second graph. A leading sub-root cohort remains terminal
until consumer-key-scoped frontier state is implemented. Post-placement local
continuation dominance follows that resident transaction so the temporary
capability gate and old shape-specific branches can be removed. Only after a
higher depth changes a real accepted plan under the unified path should
`cross_loop_pipeline_depth` become a public autotune field.
Consumer-key-scoped closure, affine recurrence lifting, and the public depth
knob remain subsequent milestones. Exact segment tuple shape,
depth-2/3/4 aliasing, one-root-per-parameterized-schedule, and the old
pointwise multi-cohort decline are migration details rather than final policy
requirements.

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
    -> one parametric cohort list policy
       (each early-admission action proves its existing synchronization lowering)
       with conservative root-major fallback
    -> one WorkerSchedule
    -> optional local continuation dominance over that resident placement
    -> schedule-frontier quotient for nested waits
    -> one counter/root-barrier finalization pass
    -> one RootBarrierPublicationPlan derivation
    -> one symbolic ownership/progress/replay validation
    -> renderer-only codegen and backend compilation
    -> post-compile residency gate before cache acceptance or launch
```

The ordering is deliberate. Continuation eligibility may be derived before
scheduling, but every consumer first receives a resident placement. Only then
can a local proof compare every possible final-publisher strand with that exact
resident occurrence and remove the consumer when inline ownership dominates.
A final-arrival continuation is therefore an ownership strength reduction of
the accepted schedule, not a second scheduler. Exact semantic readiness is
already available from `ReadinessGraph`, so scheduling does not need emitted
counters to be finalized. Conversely, a nested counter partition may depend on
the final producer and consumer wave relation. It therefore cannot be
coarsened or finalized before placement. Qwen's useful 74/22 frontier is the
concrete counterexample to the old ordering.

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
- [ ] Keep every eligible consumer resident while constructing the first
  unified cohort schedule. After resident placement is final, admit a
  continuation only when one local dominance proof covers every possible
  final-publisher strand, proves strand-injective ownership with no accumulated
  inline chain, completes the body no later than its resident occurrence, and
  leaves every downstream release frontier no later. Resident wins any
  unproved case. The selected candidate remains the existing ephemeral
  `FinalArrivalContinuation` until the final `ReadinessCounterPlan` records it.
- [ ] Replace the permanent single-producer parameterized-counter gate with
  the same exact publication/cardinality/replay certificate used for all
  domains. Keep stricter continuation requirements only where execution
  ownership genuinely requires them.
- [ ] Run one symbolic event-frontier proposal implementation for constants and
  parameters alike. It consumes the exact `ReadinessGraph`, not a prematurely
  coarsened counter plan, and produces the authoritative schedule relation.
  Only afterward may codegen choose an equivalent concrete or runtime-bounded
  loop rendering. Concrete task-list placement remains a test oracle, not an
  alternate production decision procedure.
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
runtime-bounded loop. Such rendering differences are allowed only after their
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

### Post-mortem: symbolic max-plus

Symbolic max-plus was attempted for a principled reason. A unit-duration
longest-path objective over readiness and same-strand edges can distinguish two
legal schedules that have the same occupied worker-wave horizon. That matters
for a resident consumer waiting on another worker and for a continuation body
serialized after the final publisher. It also supplied a clean secondary
handoff objective without introducing measured kernel latencies or a
model-specific rule. The concrete oracle correctly ranked small FlashMLA,
Qwen, Gemma, and Muse motifs and found real mistakes in early ownership
reasoning.

The production symbolic formulation was nevertheless the wrong abstraction:

- Bellman evaluation requires an inclusive prefix maximum along every occupied
  worker strand. With a runtime extent, the set of active prefix fibers is
  itself parameter-conditional.
- The exact relation grammar can represent the final packed schedule compactly
  while still being unable to partition every intermediate max winner without
  nonrectangular or rapidly multiplying pieces.
- The canonical symbolic `(3, N, 2)` case matched all 192 tested concrete
  substitutions, yet the relation-level scorer spent roughly 42--48 seconds
  on conditional prefix fibers and declined. Rewriting the recurrence in a
  packed global-rank domain encountered the same semantic boundary.
- Continuations make the recurrence harder: every causally possible final
  publisher is a mutually exclusive owner alternative, and an inline chain
  changes the same-strand recurrence downstream.
- Even a successful exact unit-work score would not predict instruction
  latency, register pressure, shared-memory residency, bandwidth contention,
  or the cost of a heterogeneous root body. It would be expensive exactness
  for an intentionally incomplete performance model.

Therefore production must not add a generic fiber-emptiness splitter, ranked-
prefix operation, symbolic Bellman evaluator, or another schedule IR merely to
finish this score. The useful work is retained selectively:

- the concrete max-plus evaluator remains a bounded test oracle and debugging
  aid;
- exact extrema/attainers, partial-support coverage, same-strand precedence,
  and scalar pullback remain only where ordinary ownership, readiness,
  frontier, or progress proofs consume them; and
- disagreements between the concrete oracle and the structural scheduler are
  recorded as diagnostics and GPU ablation targets, not repaired with another
  symbolic objective or a model/root-ID exception.

The replacement is the parametric cohort list policy below: static topology
chooses priority, exact readiness determines which cohorts exist, and symbolic
expressions determine only run boundaries and placement. This deliberately
trades exact unit-work optimality for bounded compilation, one-cubin reuse,
and a policy that can actually run on dynamic shapes.

#### Reuse and cleanup inventory

Keep as production foundation:

- symbolic `TileAccess`, `CoordinateDomain`, and `CoordinateRelation` layout
  and dependency semantics, including exact converses and multi-axis forms;
- one semantic `ReadinessGraph`, exact fan-in/fan-out, bounded nonuniform
  arrival counts, replay-safe epochs, and active-owner barrier publication;
- authoritative `WorkerScheduleSegment.task_order`, globally packed symbolic
  placement, runtime-bounded codegen, and post-compile residency checks;
- exact coverage, disjointness, global-slot/same-wave progress, nested-frontier
  derivation, and Qwen's generic mixed-radix/fixed-width proof; and
- launch-stage source relations, performance probes, Gantt instrumentation,
  and the concrete list/max-plus test oracles.

Reconfigure rather than replace:

- reuse root-schema criticality and concrete ready-list semantics at affine
  cohort granularity;
- make canonical root-major packing the stable ordering outcome of the same
  scheduler rather than a parameter-only path;
- make continuation eligibility common but ownership resident-first; and
- derive source identity from launch-stage support instead of retaining
  `transient_source_root` as duplicate truth.

Delete or quarantine after a call-site audit:

- symbolic-score-only occupied-strand ordinals, handoff partitions, resident
  max-plus quotients, and their tests when no ordinary progress/frontier proof
  consumes them;
- any scorer-specific partial-support operation with no other correctness
  caller; and
- obsolete parameterized recognizers, duplicated continuation selection,
  codegen-side counter/barrier reconstruction, and model-shaped scheduling
  escape hatches.

No production symbolic Bellman evaluator was committed. The cleanup is
therefore a narrow removal of unused proof scaffolding; the dependency,
readiness, placement, synchronization, and lowering refactors remain the bulk
of the implementation.

### Acyclic go/no-go gate

Cyclic event-graph optimization is not on the immediate roadmap. Finish the
single acyclic cohort scheduler and use it to select schedules for the real
FlashMLA, Qwen, Gemma, and Muse graphs. Then run the existing correctness checks,
cubin-reuse checks, cold-L2 benchmarks, and standalone comparisons. Proceed to
the remaining lowering cleanup only if this acyclic path preserves the known
FlashMLA result and demonstrates at least one reproducible end-to-end win; if
it does not, remove or quarantine the unused experimental scheduler rather
than extending it with cyclic/SCC machinery.

Implementation checkpoint (2026-09-10): the performance half of this gate is
positive, and the symbolic objective evaluator has been rejected. On the
canonical static ragged B4/Q4/H16 FlashMLA boundary, the existing acyclic
event-frontier schedule measures 61.312 us cold-L2 versus 67.456 us for the
matched three-launch Helion baseline. The dynamic F64/C16 fan-out path reuses
one cubin across B1/B2/B4/B9 and measures 57.216/59.424/100.112/173.968 us
versus 100.192/100.224/145.248/245.600 us for matched standalone. Thus the
acyclic machinery has real end-to-end value and should not be abandoned.

The first relation-only max-plus scorer prototype was removed
before production wiring. It matched all 192 small concrete oracle cases, but
the canonical symbolic `(3, N, 2)` schedule took about 48 seconds and declined
while reducing its inclusive same-strand prefix. Re-expressing that prefix in
the proved packed global-rank domain did not solve the problem: ordinary
extrema still encountered parameter-conditionally active value pieces. Do not
add a generic fiber-emptiness splitter, a ranked-prefix operation solely for
that scorer, or revive the discarded scaffolding. Static topology priority and
exact affine cohort boundaries replace this optimization slice.

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
or runtime-bounded symbolic loop. Such a branch is a rendering choice, never a
second continuation, barrier, source-admission, or ordering policy.

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
| Rendering | Which proved runtime-bounded or constant loop form emits the plan? | `cross_loop_codegen` | instruction count only |
| Body/resource tuning | How are individual root bodies tiled and pipelined? | existing block/range/warp/register configuration | constituent-kernel speed and occupancy |

A final-arrival continuation and list scheduling are intentionally separate.
List scheduling orders tasks that remain resident: a producer publishes, a
resident worker reaches the consumer, waits, and executes it. A continuation
changes ownership: the producer observing the final arrival immediately calls
the consumer body, so that consumer has no resident slot to schedule. They are
separate execution mechanisms in one finalization pipeline. The cohort
scheduler first assigns every consumer to a resident frontier; afterward, the
local dominance proof may remove one resident occurrence and replace it with
the eligible inline action. Both consume the same readiness event and neither
defines a second dependency graph or priority policy.

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
3. Parametric cohort list scheduling
      rank root/event kinds by finite topology;
      let symbolic extents determine only ready prefixes and run boundaries;
      produce one resident WorkerSchedule
4. Optional local continuation dominance
      retain resident ownership unless every possible publisher proves that
      inline execution cannot delay this body or any downstream frontier
5. Schedule-frontier quotient
      coarsen nested waits against the final schedule, when exact
6. Synchronization finalization
      retained counters, continuation identity, and root-barrier fallback
7. RootBarrierPublicationPlan
      exact publishers, arrival counts, bounds, and empty-root owner
8. Symbolic validation
      ownership, coverage, progress, configured-capacity bound, and replay safety
9. Code generation and backend compilation
      renderer and proved strength reductions only
10. Post-compile residency gate
      verify actual cubin occupancy before cache acceptance or launch
```

There is no capability graph or preliminary synchronization plan. The
scheduler reads the one `ReadinessGraph`; when considering early admission or
cohort movement, it proves that the event's existing relations can lower the
required counter/publication. It constructs the complete resident schedule
first. A later continuation candidate may remove a resident occurrence only
through the narrow local dominance proof; otherwise ownership stays resident.
Final synchronization is constructed only after placement supplies the
information needed to quotient nested waits. No provisional counter plan is a
source of truth.

There is one bounded conservative outcome, not a second policy. If an optimized
cohort move, schedule-frontier quotient, or progress proof declines, the same
scheduler retains canonical packed root-major order for that proposal, derives
synchronization, and validates it. Its same-wave boundary
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

> Among all provably admissible cohorts, protect the shape-independent
> unit-weight precedence critical path; at equal criticality, close the active
> readiness event and immediately admit its consumers without displacing
> unfinished ancestors; never leave a worker idle while an eligible complete
> candidate fits the remaining lanes.

This principle has three ordered parts:

1. **Legality:** `TileDependencyGraph` and the exact `ReadinessGraph`
   determine which work may be assigned and which runtime wait protects it.
2. **Priority:** unit-weight `top`/`bottom`/slack protect structurally critical
   chains. A run that releases a better structural class inherits that class;
   event closure and active-cohort continuation then break equal-class ties.
   Runtime extents, symbolic remaining-work counts, measured latency, and
   continuation handoff scores never choose between roots.
3. **Work conservation:** every worker slot receives eligible work when a
   complete candidate fits. Newly released consumers enter the ready frontier
   immediately, but dependent work uses only proved tail capacity unless every
   unfinished ancestor is already assigned. Independent roots may backfill one
   another. Exact cohort and committed-run constraints may deliberately leave
   an otherwise unusable lane idle.

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
  used for critical-path priority; runtime measurements select only ordinary
  autotune configuration values.
- No catalogue of schedules and no public scheduling-policy selector. The one
  permitted scheduler-specific scalar is `cross_loop_pipeline_depth` in the
  fixed range 1--4.
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

If root/event relations induce a cyclic quotient, this milestone declines the
optimized proposal and retains the proved conservative schedule and barriers.
Do not add an affine recurrence proof merely to broaden coverage: first require
the acyclic scheduler to demonstrate end-to-end performance value.

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
this point every consumer remains resident and no final counter plan exists.
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

The canonical schedule `C` is the validated configured/autotuned all-resident
root and intra-root order for that kernel, including any wave-aligned boundary
required when same-wave progress cannot be proved, and before this scheduler
derives any new readiness-major permutation. Depth 1 reproduces `C` exactly.
The scheduler does not erase configured order merely to make source forms look alike. The
relation composition above proposes an exact readiness-major permutation as a
noncanonical depth-2-or-higher move, and only when the resulting placement
does not interrupt a committed ancestor run. The common fixed priority orders
the remaining legal candidates, and configured depth is the only profitability
cap. Different source/configuration forms may therefore start from different
`C` schedules while traversing the same scheduling policy.

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

A readiness-equivalent cohort is not a set of tasks that merely happen to have
the same scalar release rank. It is the maximal affine family in one existing
relation piece whose members have the same exact prerequisite event family and
key fiber, including every nested wait. Affine translations of that family
across keys may share one relation piece, but each key fiber remains an atomic
cohort for noncanonical dependent movement.

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
chunk-size heuristic. Canonical placement and incomparable ready roots may end
at worker-wave capacity. If that capacity cuts a dependency-released cohort, a
noncanonical pull rounds back to the last exact cohort boundary; if no complete
cohort fits, that candidate is ineligible while a blocking ancestor remains.
After every ancestor that it could displace is assigned, the scheduler may
select the whole cohort as one committed multi-wave run. Per-wave relation
pieces may clip its representation, but no other candidate may interleave
before the cohort is fully assigned. The compiler never invents a
shape-specific partial cohort just to fill a hole.

### Event-aware critical-path priority

For each candidate interval, apply its exact claimed contributions to the
existing readiness events and derive:

- which events become complete;
- which consumer cohorts consequently become admissible; and
- whether the candidate continues or closes the currently active canonical
  event cohort.

Let an interval that newly releases consumer cohorts inherit their best
structural class:

```text
release_class = min(base(consumer) for newly admissible consumers)
effective_class = min(base(candidate root), release_class)
```

If no consumer becomes admissible, use the candidate root's base class. Choose
candidate intervals lexicographically by:

1. lowest effective structural slack and greatest structural depth;
2. a newly admissible consumer at that class;
3. completion of a readiness event;
4. continuation of the active canonical event cohort before opening an
   untouched equal-class cohort;
5. immediate inlet from an exact earlier launch stage; and
6. canonical root, event key, and configured task order.

The criticality class protects the structural critical path. Event inheritance
and active-cohort closure prevent equal-class producer work from being spread
across many fan-in groups while an already-started event waits. Every priority
field is a finite topology integer or an exact boolean. `B`, `Q`, sequence
extent, symbolic root size, symbolic remaining claims, and worker-tail length
never rank two candidates. A guard-dependent priority winner uses the
canonical tie-break unless one outcome is proved over the complete guard.
Candidate intervals still end whenever a readiness contribution or
source-ticket frontier changes; those expressions bound the run but do not
rank it.

This is the single priority policy for concrete and symbolic inputs. A newly
completed event releases its consumer into the same ready set with pull depth
one greater than the moved work needed to release it. The configured
`cross_loop_pipeline_depth` decides whether that candidate is eligible; no
separate parameterized policy decides whether it deserves to run.

### Work conservation

Select intervals until every worker slot in the abstract wave is filled or no
eligible complete candidate fits. An admissible dependent cohort that would be
split by the remaining lanes is deferred while a blocking ancestor remains;
once all such ancestors are assigned, it may begin only as an atomic committed
run that continues across subsequent waves.
Preferring a ready downstream root does not create a barrier: after assigning
its complete available cohorts, remaining workers receive other eligible roots
or retain the canonical tail.

First construct the validated configured all-resident baseline `C`, including
every progress-required wave-aligned boundary and before any newly derived
readiness-major permutation. At depth one, return `C` exactly. At larger
configured depths, walk its root-run frontiers in order. Once a run has been
selected, finish its maximal affine portion before opening another run; this
is the **committed run**. At the committed run's
terminal hole or boundary, the canonical successor competes with legal
ready-list alternatives. Assign the first noncanonical winner pull depth two,
propagate `max(2, 1 + max(predecessor pull depth))` through work made early by
that pull, and choose the highest-priority candidate whose depth is within the
configured bound. Independent work advanced past canonical order also begins
at depth two. Pull depth persists across worker-wave boundaries and resets to
one only when all affected root frontiers exactly rejoin `C`. Depth is
ephemeral scheduler state and is not stored as another schedule
representation.

For every configured depth, this produces one deterministic proposal:

1. derive the static priority table once from the existing readiness graph;
2. start from the same canonical packed schedule and exact symbolic frontiers;
3. discard candidates that fail readiness, progress, ownership, residency,
   relation-budget, committed-run, or structural non-displacement checks;
4. discard otherwise legal candidates whose pull depth exceeds the configured
   value;
5. include the eligible canonical successor in that same fixed-priority set at
   depth 1, choose the highest-priority candidate, and let canonical order win
   an exact tie; and
6. emit a noncanonical run only when it strictly outranks the canonical
   successor (or the successor is ineligible); otherwise emit the canonical
   run.

There is no second compiler decision asking whether that legal pull is likely
to be fast. In particular, predicted makespan, occupied-wave horizon, handoff
count, locality estimates, body/resource estimates, and model-specific
thresholds do not accept or reject the proposal. Comparing the resulting
depths is the autotuner's job.

Dependent movement is non-displacing with respect to the committed run. A
complete consumer cohort may occupy lanes proved unused by that run, including
its final partial wave, and may beat the canonical successor at that boundary.
It may not interrupt the committed run or push any unfinished ancestor into a
later occupied wave merely to create earlier visible overlap. Interleaving
beyond such holes is allowed only between roots proved incomparable in the
root/event quotient; different keys inside ancestor-related roots do not
suffice. Never split one readiness-equivalent consumer cohort solely to fill a
smaller hole. Once all blocking ancestors are assigned, a larger selected
cohort may span waves, but it becomes the next committed run and remains atomic
until fully placed.

Occupied-wave horizon, first-admission frontiers, handoff counts, and the
concrete max-plus score are diagnostics across depth values, not production
profitability gates. Exact coverage, readiness, progress, residency, and
relation budgets remain hard. The autotuner decides whether the extra legal
overlap at a larger depth repays added waits, fragmentation, locality loss, or
body contention.

Resident ownership is the default. A continuation may be selected only after
the resident schedule is complete and one local proof establishes exact
final-publisher alternatives, strand-injective ownership, no accumulated
inline chain, execution no later than the resident occurrence for every
winner, and no later downstream frontier. An unproved case remains resident.
A source-stage action is compared separately because it intentionally changes
resident coverage. With no provably lowerable fine-grained prerequisite there
is no early-admission opportunity, so the scheduler retains canonical compact
order without stepping through its frontiers.

### Parametric repetition without symbolic priority

The scheduler may execute a bounded concrete control trace over relation cells,
not runtime tasks. When the normalized state repeats by an affine translation,
lift the repeated trace only after proving for every repetition:

- identical active relation pieces and static priority outcome;
- identical readiness and event-cohort structure;
- identical worker-lane ownership pattern;
- affine positive deltas for root cursors and event frontiers; and
- exact progress to the next symbolic boundary.

Emit the lifted prefix, full repetitions, and tail directly as partial
`WorkerScheduleSegment.task_order` relations using existing affine,
floor-division, modulo, and clipping operations. A runtime-dependent period,
unresolved winner, nonlinear cursor update, or proof-budget overflow retains
canonical packed order. This is finite-motif induction, not cyclic dependency-
graph support and not a Run/Repeat IR.

### Deferred: cyclic recurrence extraction

Cyclic event-graph support is deliberately not part of the immediate roadmap.
The acyclic scheduler must first pass its correctness and performance gates; if
it does not produce worthwhile wins, abandon the redesign rather than adding
recurrence machinery. Until a separate future decision revisits this scope, a
cyclic quotient retains the conservative schedule and synchronization.

If cyclic support is reconsidered later, it must still be expressed through the
existing `WorkerScheduleSegment.task_order` relation and proved from the same
readiness graph; this note is not an implementation task or an exit criterion
for the current work.

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

After constructing the complete all-resident cohort schedule, keep resident
ownership unless a local dominance proof succeeds for the entire event family.
A possible final publisher is a causal maximum of the event's producer tasks,
not merely the producer with greatest wave. For every possible winner, prove:

- publisher ownership is strand-injective across simultaneously live keys;
- the same strand cannot accumulate an inline chain;
- the consumer completes no later than its resident occurrence; and
- every contracted downstream release frontier is no later.

Only then may the consumer be removed from resident placement and contracted
into its producer chain. Any missing alternative, correlated winner, nested
timing dependence, or unproved comparison keeps the consumer resident. A
parameter symbol, sink status, sampled task count, or `fan_in > 1` is not an
eligibility or priority rule. The concrete max-plus oracle may report whether
an interesting continuation would have helped at a substituted shape, but it
does not select production ownership.

The selected identity is copied exactly once into the final counter plan after
the schedule is accepted. Proposal, proof, and diagnostics consume that same
selection; codegen only renders it. There is no second continuation pass.

This is distinct from resident placement but remains part of finalizing the
same schedule.
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

The earlier hypothesis was to finish one activation slice's gate fan-in and
immediately pipeline reduction/activation/down before spreading gate progress.
The measured 77-segment N64 experiment disproves that as a general policy: it
interleaved downstream work through an unfinished gate root and regressed
235.376 versus 175.984 us local. The safe structural behavior is narrower:
preserve the committed gate producer run, then place complete reduction and
activation cohorts only in its genuine terminal tail. Event closure chooses
among already-safe tail candidates; it does not manufacture capacity by
postponing later gate work.

Use matched bodies and resource settings:

- split-K 16 for gate and down/activation;
- gate/down N32/K128, reduction N64, activation block 256;
- W1, range stages 2, no `maxnreg`, and a common safe multiplier initially;
- `standalone_matched` with identical body parameters; and
- freshly tuned standalone as the final performance target.

The previous normalization reduced a concrete proposal from thousands of
fragments to 77 segments, but that aggressive schedule was also materially
slower. The parameterized schedule relation should preserve the compact
five-root pattern with a bounded number of pieces, and proof time must scale
with those pieces rather than CTA or segment-pair count.

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

## Expected scheduler traces and cross-probe conflict audit

This section de-risks the policy before implementation. **Recorded** facts are
present in a probe, generated schedule, or timing journal. **Derived** geometry
is arithmetic from those facts. **Expected** behavior is what the proposed
scheduler should produce and remains subject to CPU schedule dumps and GPU
measurement.

### One constraint hierarchy, not per-kernel priorities

The probes do pull in different directions if “priority” is allowed to move
arbitrary ready CTAs. FlashMLA and the MoEs benefit from earlier downstream
work; Qwen, Muse, and full MLA show that legal overlap can lose when it
fragments or narrows a producer gang. The unifying rule is to constrain the
move before consulting priority:

1. Exact readiness, ownership, acyclicity, publication lowering, and resident
   capacity are mandatory.
2. Preserve the configured task order and every committed ancestor/root run.
   A hole is unused capacity in that committed run's terminal wave; it is not
   capacity created by postponing producer tasks.
3. Any noncanonical pull-forward of dependent work may enter such a hole only
   as a complete **readiness-equivalent cohort**: one exact event-key fiber
   with the same prerequisite event family across every nested wait. Merely
   reaching the same scalar release rank does not merge two cohorts. Canonical
   packed placement may retain an already-proved partial tail; the scheduler
   may not manufacture a new partial cohort merely to occupy a smaller hole.
4. Broader displacement is allowed only between roots proved incomparable in
   the root/event quotient. Different keys inside an ancestor-related root do
   not create an exception; this distinction rejects Muse's failed deep
   pipeline.
5. Among the candidates left by rules 1--4, use static structural criticality,
   released-consumer class, event closure, active-cohort continuation,
   launch-stage order, and canonical order.

This hierarchy is consistent with every measured **resident-placement** win.
Continuations are not a priority exception: they change execution ownership
rather than resident ordering and remain behind the separate local dominance
gate. Canonical source order still carries useful unmodeled information in
FlashMLA and Qwen. The scheduler preserves it unless an exact readiness-major
permutation and the hierarchy above prove a replacement.

Three tensions remain explicit rather than becoming hidden tie-breakers:

- Canonical request/task order encodes useful locality in FlashMLA, Qwen, and
  Muse. The compiler can preserve or exactly transform it, but a topology-only
  priority cannot rediscover “long request first” from device metadata.
- Continuation profitability is not resident list priority. If the local
  dominance proof cannot preserve fixed Qwen, Gemma, Muse, and DeepSeek
  controls, stop and reconsider ownership separately rather than weakening the
  placement hierarchy.
- Topology cannot predict shared-resource or bandwidth contention. If Muse or
  full MLA regresses at a deeper pipeline setting, the autotuner must select
  depth 1 for that configuration or polymorphic guard. Do not narrow the
  candidate domain, change priority, or add a workload-specific exception in
  response to that measurement.

If a future measured resident-placement winner genuinely violates this
hierarchy, the alternatives are: (a) a stronger general dominance proof over
the existing schedule relations, or (b) a later device-side ready-work claimer
for data-dependent tasks. A catalogue of model-specific priorities, sampled
shape schedules, or host-generated per-invocation instruction tensors is not
an acceptable intermediate fix.

### One generic autotuned scheduling knob

The scheduler exposes exactly one new scheduling knob:
`cross_loop_pipeline_depth`. It is analogous to Triton's `num_stages`, not an
admission width, priority selector, or catalogue index.

Start from the validated configured all-resident canonical `WorkerSchedule`,
including every progress-required wave alignment and before newly derived
readiness-major movement or continuations:

- depth 1 permits no noncanonical pull-forward and preserves all natural
  canonical tail packing;
- a cohort moved earlier than its canonical occurrence has pull depth 2;
- if that early admission depends on another moved cohort, its depth is one
  plus the maximum moved-predecessor depth;
- incomparable branch work moved ahead of canonical order starts at depth 2;
- joins take the maximum incoming depth, and depth is not reset merely because
  an intermediate moved root was assigned;
- pull depth persists across worker waves and resets to 1 only when every
  affected root cursor and packed lane position exactly rejoins the canonical
  schedule; and
- conditionally empty symbolic roots still belong to the finite schema; their
  runtime emptiness affects fit, never the configured depth or priority.

“Ahead of canonical order” is defined against `C`'s stable packed ordinal
`wave * worker_count + worker`, not against physical chronology. A different
same-wave worker position is therefore still a noncanonical transformation and
requires depth at least 2, even though it does not execute earlier in time.
Only the global `wave` coordinate and same-strand order carry execution
chronology. The packed ordinal exists solely to identify deviation from and
exact rejoining with `C`.

Readiness, progress, committed-run non-displacement, whole-cohort movement,
and exact lowering are checked before this cap. The knob changes only how far
an otherwise legal list-scheduling proposal may pull work forward. One
compile-time value is part of the configuration/cache identity and applies to
the entire dynamic-shape guard; it is never selected separately for each `B`
or `Q` at runtime.

This is the only new scheduler-specific tuning dimension. Existing Triton and
persistent-resource knobs such as block sizes, `num_warps`, range stages,
`maxnreg`, and `num_sm_multiplier` remain independent. The fixed autotune
domain is always the fixed set `{1, 2, 3, 4}`, with default one when tuning
is disabled. Do not derive or prune this range from topology, and do not remove
duplicate outcomes: as with `num_stages`, two values are allowed to lower to
the same program. Do not add admission width, priority weights, chunk size, or
per-root knobs.

This fixed-domain rule is deliberate. `num_stages` is represented as an
ordinary `IntegerFragment`: the autotuner is allowed to measure values that
later prove equivalent or unhelpful for a particular loop. Pipeline depth must
behave the same way. The readiness topology determines the schedule produced
*at* a value; it does not determine which values exist in the search space.
The compiler therefore needs no “useful maximum depth” heuristic and no
schedule-equivalence pass in config generation.

For an ordinary fixed-shape specialization, the normal autotuner offers all
four values alongside the kernel's existing resource configuration; its search
strategy need not exhaust the Cartesian product. For a
declared polymorphic kernel, one value must serve the whole guard. Use the
existing `autotune_multi` mechanism over representative shapes, with a
normalized worst-case objective (`aggregation="max"`, normally
`relative_to="baseline"`) for acceptance, so B1/B2/B4 cannot silently select
different schedule depths while claiming one-cubin reuse. A pretuned kernel
records the chosen scalar in its ordinary `Config`; the compiler never
contains a model-to-depth table.

The division of responsibility is intentionally sharp. The compiler still
proves semantic legality and preserves the structural envelope that makes one
depth denote one stable family of schedules. The autotuner decides whether to
open any noncanonical event pipeline at all and how many causally successive
pulls are worthwhile. Thus the knob absorbs the uncertain decisions exposed by
the probes—extra synchronization, locality loss, and body/resource contention—
without turning priorities, widths, or per-root policies into tunables.
The knob does not select continuation ownership directly. Because the common
continuation-dominance proof runs after resident placement, different depths
may expose different legal ownership outcomes through that same proof; there
is still no second ownership policy or ownership knob.

Committed-run non-displacement and readiness-cohort atomicity are deliberately
conservative search-language restrictions, not claims that all other orders
would be semantically incorrect. They remain hard at every depth because
relaxing either would introduce a second admission-width/fragmentation policy
that one depth scalar cannot express. The known winning schedules obey both;
the autotuner chooses only among the legal depths inside this fixed envelope.

The expected settings below are hypotheses to validate, not defaults or
special cases:

| probe | likely depth | reason |
| --- | ---: | --- |
| canonical FlashMLA B4 | 1 | source admission supplies the win; resident radix/final placement is already canonical |
| canonical FlashMLA B9 | likely 3 | moved request-local work can release one additional early final cohort |
| dynamic FlashMLA F64/C16 | 1 | canonical packed C16 tails already win; deeper/key-major pulling lost |
| FlashMLA Q1 | 1 | one full producer wave exposes no useful readiness frontier |
| FlashMLA Q2 | 1 for resident scheduling | its benefit is continuation ownership, not list depth |
| Qwen3 pretuned/dynamic | 1, with 2 as the only plausible challenger | preserve the chain and 74/22 placement; broader dependent repacking lost |
| Gemma 4 A4B | 1 | canonical root-6 reductions already fit the terminal root-5 tail; ownership remains separate |
| Muse/Glimmer | 1 | larger depths should alias while the known harmful fragmentation remains illegal |
| Nemotron MoE | 4 | shared up (2) → activation (3) → shared down (4) fills the routed-up tail |
| DeepSeek-V3 MoE | likely 4 | analogous three-level branch preparation; resident-only confirmation remains required |
| full DeepSeek MLA / dense FFN | 1 | preserve full producer gangs; scheduling has little useful slack |

The autotuner, not the table, chooses the value. The table only demonstrates
that one scalar describes the observed cross-kernel tension. Cold-L2 rollout
must confirm the expected choice and record when several values alias.

### Canonical FlashMLA Q4, B4

Recorded geometry on B200 is `W=148`, one resident CTA per SM, with sequence
lengths `[1696, 1730, 4641, 45118]`:

```text
attention/source tasks by request: 14 + 14 + 37 + 353 = 418
radix-16 groups by request:          1 +  1 +  3 +  23 = 28
first-stage radix tasks:             28 * Q4 = 112
final tasks:                          4 * Q4 = 16
resident tasks:                     112 + 16 = 128 <= W
physical ticket roles:              418 source + 148 resident
```

The conceptual source packing is two full 148-task waves plus a 122-task tail,
but the successful program must **not** treat the 26 apparent lanes as a
shared resident wave. Source tickets `[0,418)` execute one wait-free attention
task and retire. Only after all source tickets have been allocated do tickets
`[418,566)` enter the 148-strand resident program at resident wave zero.

Expected scheduler steps:

1. Preserve the exact source ticket order and launch stage. No resident action
   may delay issuance of an unassigned source ticket.
2. Preserve the configured radix order, which begins with the long request's
   22 full groups and then covers the short-request and tail roots. This order
   is source/configuration information; topology priority does not infer which
   device `seq_len` is longest.
3. Build exact attention-to-radix cohorts. Resident tasks are assigned once;
   their waits make each radix body runnable when its own producers finish.
4. Use event closure only to order already-safe resident cohorts. Do not model
   an “early width” of 26 or move a resident task onto a source strand.
5. Accept only if source admission and every previously earlier event frontier
   are preserved.

This predicts the clean two-role schedule that measured about 61.31 us versus
67.46 us matched standalone. The older policy that explicitly filled all 26
conceptual source-tail lanes measured about 71.5 us; the hierarchy rejects it
before priority is considered.

### Canonical FlashMLA Q4, B9

Recorded geometry uses the same `W=148`:

```text
source tasks: [92,111,80,67,50,56,119,33,76] = 684
radix groups: [ 6,  7, 5, 5, 4, 4,  8, 3, 5] = 47
radix tasks: 47 * Q4 = 188
final tasks: 9 * Q4 = 36
resident tasks: 224 = 148 + 76
```

Source tickets have four full groups of 148 and a 92-task tail. The resident
program has one full wave and a 76-task tail. Its request-specific radix roots
are mutually independent until their final request reductions.

Expected scheduler steps:

1. Protect all 684 source tickets exactly as for B4.
2. Put the request-local first-stage radix roots in one resident ready list.
   Split runs only at exact request/query readiness boundaries.
3. A cohort enters the ready list only when its required source-ticket
   frontier is satisfied. Among simultaneously admissible cohorts, use the
   static structural class, event closure, and canonical request/key order;
   never rank two cohorts by their numeric ticket-frontier values.
4. Honor the resulting global segment order in codegen, so a ready request is
   not hidden behind all tasks of a blocked numerical root.
5. Never alter source issue order to improve resident order.

The measured trace had 184 of 188 radix tasks ready, 170 started, and 107
finished before attention ended. The four request-7 tail reducers previously
waited 14.8--28.2 us after becoming ready; the successful order removes that
head-of-line delay and measures about 90.0 us versus 100.13 us standalone.

### Dynamic FlashMLA F64/C16

This is the cleanest symbolic placement case. With `W=148`, each runtime
request contributes 64 producer tasks and one complete 16-task consumer
cohort. `B` controls only repetition and packed offsets:

| B | producer placement | safe consumer tail placement |
| ---: | --- | --- |
| 1 | wave 0 workers 0--63 | cohort 0 on 64--79 |
| 2 | wave 0 workers 0--127 | canonical packing uses workers 128--147, then wave 1 workers 0--11 |
| 4 | wave 0 full; wave 1 producers 0--107 | canonical packing uses all 40 tail lanes, then continues consumers in wave 2 |
| 9 | three full waves; wave 3 producers 0--131 | one cohort on 132--147; eight cohorts next wave |

The scheduler preserves producer order and cuts candidates at every
64-producer event boundary. Depth-one canonical packing retains its already-
proved partial consumer tails. Any additional pull-forward must move a complete
16-consumer cohort into actual tail lanes. The rejected key-major order
`[64 producers, 16 consumers]` per request
moves the last B4 producer from wave 1 to wave 2 and the last B9 producer from
wave 3 to wave 4; it measured roughly 120.5/266.0 us. Non-displacement rejects
that proposal immediately.

Current one-cubin B1/B2/B4/B9 results are
57.216/59.424/100.112/173.968 us versus
100.192/100.224/145.248/245.600 us standalone.

The B1/S65536/Q1 control has 128 producers in one `W=148` wave and one
fan-in-128 consumer frontier. There is no early cohort to schedule; source
tickets add overhead and must decline. Q2 already has a useful local
final-arrival pattern and is a continuation gate, not evidence for broader
resident reordering.

### Qwen3 decode

At multiplier eight, `W=1184`; the fifteen root sizes are:

```text
[32B, 32B, 768B, 10B, 8B, 1024B, 512B, 32B, 32B,
 512B, 32B, 32B, 1536B, 96B, 512B]
```

The recorded B1 graph has 5,170 logical tasks and 5,106 resident tasks after
two existing continuations. It is almost entirely a chain. Consequently the
priority table has little useful freedom and the correct result is mostly the
configured local order plus exact handoffs.

Expected scheduler steps:

1. Preserve the configured root-5 attention order (`[2,1,0]` in the pretuned
   source and request-major `[2,0,1]` in the ragged source) unless relation
   composition proves an exact readiness-major replacement.
2. Advance the exact root 5→6 fan-in-8, 6→7 fan-in-16, and 7→8 fan-in-1
   cohorts without interrupting unfinished producer runs.
3. Pack roots from their symbolic prefix offsets. In the historical B1
   relation, that arithmetic puts root 13's 96 tasks on workers 576--671 and
   root 14's 512 tasks on workers 672--1183 in the same wave.
4. After placement, derive root 14's nested wait partition from the exact
   schedule. It yields 74 earlier producers followed by 22 later producers;
   neither number belongs in priority policy.
5. Keep root 13 resident unless continuation dominance is proved. Preserve
   existing safe continuations only through the same ownership proof.

Collapsing 74/22 to one fan-in-96 wait costs about 2 us. Moving the placement
as well costs roughly another 4 us. Conversely, a global proposal reduced 14
abstract waves to 13 but measured about 98.2 us versus 94.1 us local because it
only repacked dependent roots. The non-displacement/incomparability rules
reject that proposal despite its smaller wave count.

For the one-cubin B1/B2 source, the same topology table is reused while `B`
scales all root/cohort bounds. Device `context_lens` masks bodies; it cannot
change priority or task counts. The current 109.456/133.024 us versus
106.368/126.976 us exact-shape controls and 111.5-second compile remain
separate rendering/proof gates, not a reason for a Qwen priority.

### Gemma 4 A4B MoE

At tuned multiplier three, `W=444`; root counts and packed intervals are:

```text
r0 router       16B   [0,16B)
r1 group top-k   4B   [16B,20B)
r2 route merge    B   [20B,21B)
r3 gate/up      352B  [21B,373B)
r4 activation   48B   [373B,421B)
r5 down         352B  [421B,773B)
r6 reduction     11B  [773B,784B)
r7 post-norm      B   [784B,785B)
```

The graph is effectively a chain, so topology priority again makes no broad
reordering choice. Packing and exact readiness do the useful work:

```text
B1 wave 0: r0 16 | r1 4 | r2 1 | r3 352 | r4 48 | r5 23
   wave 1: r5 329 | r6 11 | idle 104

B2 wave 0: r0 32 | r1 8 | r2 2 | r3 402
   wave 1: r3 302 | r4 96 | r5 46
   wave 2: r5 444
   wave 3: r5 214 | r6 22 | idle 208
```

The exact `(B,11)` root 5→6 event has fan-in 32. Every resident root-6 task
fits the genuine terminal root-5 tail without moving a down-projection task.
The scheduler therefore keeps root 6 resident. At W296 the same causal A/B
measured 61.408/77.792 us resident versus 63.568/83.936 us inline; the full
inline chain was slower still. Root 7 remains a separate continuation decision.

The fixed B1 pretuned kernel currently chooses a root-6 continuation and
measures 49.120 us, while the best exact packed resident-root-6 control is
51.168 us, but those are not a clean ownership-only A/B. A generic terminal-
publisher proof may legitimately choose differently under the narrower B1/W592
guard; this remains an explicit validation gate rather than a static/dynamic
policy exception.

### Muse/Glimmer FFN

For the current dynamic N32 configuration, `W=1184`, one warp, two stages:

```text
r0 gate split-K       19968B
r1 gate reduction       640B   fan-in 32, final N64 keys fan-in 16
r2 SiLU/multiply         16B   fan-in 40
r3 down split-K        3328B   fan-in 1 from r2
r4 final reduction      104B   fan-in 32
```

The current four-resident-root plus final-continuation packing is:

```text
B1 waves 0--15: r0 full
   wave 16: r0 0--1023 | r1 1024--1183
   wave 17: r1 0--479 | r2 480--495 | r3 496--1183
   waves 18--19: r3 full
   wave 20: r3 0--271

B2 waves 0--32: r0 full
   wave 33: r0 0--863 | r1 864--1183
   wave 34: r1 0--959 | r2 960--991 | r3 992--1183
   waves 35--39: r3 full
   wave 40: r3 0--543

B4 waves 0--66: r0 full
   wave 67: r0 0--543 | r1 544--1183
   wave 68: r1 full
   wave 69: r1 0--735 | r2 736--799 | r3 800--1183
   waves 70--79: r3 full
   wave 80: r3 0--1087
```

The scheduler should preserve the complete r0 producer run, admit r1 only in
its terminal tail, then admit r2/r3 only in r1's terminal tail. `B` changes
only the interval endpoints. It must not pipeline one completed slice's r2/r3
through later r0 slices: the measured 77-segment N64 schedule did exactly that
and regressed 235.376 versus 175.984 us local. A conservative five-segment
event plan measured 172.064 versus 173.968 us local.

Current all-exact dynamic B1/B2 is 1579.104/3110.080 us versus
1649.600/3244.000 us matched standalone, but this does not prove that event-
priority beats root-major. B4, exact-shape, event-order, and resident-r4 A/Bs
remain required. Muse is the key proof that different event keys do not by
themselves license fragmentation of an ancestor root.

### Nemotron MoE

For the recorded B1/m4 schedule, `W=592` and root task counts are:

```text
norm 1, router 128, top-k 1,
routed-up 928, routed activation 64, routed-down 168,
shared-up 116, shared activation 15, shared-down 168, final-add 11
```

The winning order is:

```text
norm -> router + shared-up -> top-k + shared activation
     -> routed-up wave 0: 592
     -> routed-up wave 1: 336 | shared-down 168 | idle 88
     -> routed activation -> routed-down -> final add
```

The shared and routed branches are root-level incomparable before final-add.
Advancing shared-up/activation makes the complete 168-task shared-down family
available for the routed-up tail's 256 holes. No routed-up task moves later.
This is exactly the hierarchy's broad-interleaving case and measured 98.304 us
versus 120.864 us with the proposal disabled.

At B2 the routed-up family is `1856 = 3*592 + 80`, leaving 512 terminal lanes;
the scaled 336-task shared-down family still fits. The same symbolic priority
and B-scaled bounds should therefore recover the placement. Current dynamic
260.128/337.952 us versus 188.448/243.712 us standalone is dominated by
128-byte spills/body lowering, not the absence of a structural opportunity.

### DeepSeek-V3 MoE

For the recorded static B1/m4 schedule, `W=592`:

```text
router 128, top-k 1,
routed W13 2048, routed SwiGLU 64, routed W2 1792, reduction 28,
shared W13 256, shared SwiGLU 8, shared W2 224, final add 28
```

Useful capacity is deterministic:

```text
routed W13: 2048 = 3*592 + 272, leaving 320 lanes
shared W13: 256
shared W2:  224
routed W2:  1792 = 3*592 + 16, leaving 576 lanes
```

Expected steps are to place router work beside independent shared-W13 work,
advance shared SwiGLU when ready, put the complete shared-W2 family in a routed
W13 terminal hole, then run the routed suffix and final join. The recorded
ten-segment plan with three continuations measured 171.808 us versus 182.176 us
with global scheduling disabled. Exact segment offsets were not retained, so
the specific shared-W2 hole above is a geometry-derived hypothesis that must be
checked in the new schedule dump.

At dynamic W1184, the same holes exist for B1 and grow with B2, but the current
kernel is R255 with 354-byte spills and measures 542.752/886.816 us versus
340.000/564.704 us standalone. The list policy can recover branch placement;
it cannot repair that resource/code-shape loss. Device-created expert
histograms likewise remain outside static scheduling.

### Full DeepSeek MLA and dense-chain controls

The full MLA B1/context-8192 control uses `W=256`. Important masses are Q-B
1536 (six full waves), q-down 512 (two waves), attention partial 256 (one full
wave), attention reduction 64, V-up 256 (one full wave), and O projection 896
(three waves plus 128). The scheduler should retain exact q-down readiness,
which saves about 2 us, but otherwise preserve full producer gangs:

- attention's 256 tasks release within roughly 0.288 us, so fine-grained
  reduction admission adds overhead rather than useful overlap;
- V-up exactly fills W256, so placing 64 reducers first and narrowing V-up to
  192 lanes creates a second V wave and loses about 2 us; and
- aggressive HLFET measures 128.960 us versus 128.928 us source/non-delay.

The remaining full-MLA gap is resource-driven: cache preference alone moved
157.50→127.06 us, and padding O-projection shared allocation from 128 to
56,320/94,208/169,984 bytes moved 36.672→40.768/46.912/69.440 us. The priority
policy must decline rather than claim a scheduling fix.

The isolated Qwen FFN is `1536 W13 -> 96 activation -> 512 W2`; it has no
independent branch or terminal hole for a complete downstream gang. Canonical
order is therefore the expected list result. These negative controls are as
important as the wins: they ensure “ready” is not mistaken for “profitable to
move.”

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

## Historical implementation sequence and detailed current phases

Phases 0--4 record the implementation sequence that produced the current
foundation. Phase 4A expands the active cohort-list work. The ordered checklist
at the top of this document governs what happens next if wording here appears
to conflict with it.

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
  combine it with released-consumer, event-closing, active-cohort, and stable
  canonical priority.
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
- Use every exact nested key and any local-dominance-selected continuation
  contraction; validate the finalized counter identity after placement.
- Bypass and then delete the quadratic segment-pair validator.
- Make materialization raise if called inside acceptance.

Exit gate: proof time scales with relation-piece count and canonical Muse N32
compiles within the numeric budget above.

### Phase 4: parameterized extents

- Generalize existing CoordinateDomain/CoordinateRelation bounds.
- Derive symbolic event counts, fan-in, and schedule wave counts.
- Lower runtime-bounded wave loops.
- Let cyclic event quotients retain the conservative schedule; cyclic
  event-frontier optimization is explicitly deferred until after the acyclic
  performance decision.
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
strength-reduces that proved relation into runtime-bounded wave loops.
Dynamic memory layouts are not
specialized from hints: dependencies coarsen to root barriers, and every
resident worker publishes once per producer root so epoch targets stay fixed
while shapes vary between graph replays. At this checkpoint, exact
parameterized readiness events remained later Phase 4 work; subsequent
checkpoints below implement the first exact-event slices. Parameterized roots
with rank greater than one, L2-permuted
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
admission remains one-consumer-per-key. At that historical checkpoint, the
subsequently removed equal-size event-frontier recurrence remained fan-in-one.
Thus the counter change broadened synchronization without silently broadening
either execution-ownership optimization.

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

#### Phase 4A.3: build the parametric cohort list schedule

This phase replaces the discarded symbolic max-plus evaluator. It operates on
the existing roots, readiness events, and coordinate relations; it introduces
no retained graph, schedule, region, or instruction abstraction.

- [x] Compute finite unit-weight `top`, `bottom`, slack, and canonical
  priority classes directly from the possibly-nonempty acyclic root/event
  topology. Cache them only as derived properties of `ReadinessGraph`.
- [x] Derive each root's exact readiness-equivalent cohort-order candidates by
  composing consumer order, readiness keys, producer arms, and the configured
  producer task order. Keep configured order in `C`; an exact-bijection
  readiness-major replacement is a depth-2-or-higher proposal and must also
  pass the committed-run/non-displacement gate.
- [ ] Track only symbolic root cursors, exact admissible prefixes, active cohort
  identity, event frontiers, relation-piece boundaries, and remaining lanes.
  These are ephemeral local variables, not another semantic object.
- [ ] Form maximal affine candidate intervals and rank them only by static
  structural class, released-consumer class, event closure, active-cohort
  continuation, launch stage, and canonical order. Runtime expressions may
  bound a run but may not rank candidates.
- [ ] Add `cross_loop_pipeline_depth` as the only scheduler-specific
  `IntegerFragment`, with the unconditional candidate set `{1,2,3,4}` and
  default 1. Do not inspect topology to prune or deduplicate its values.
- [ ] Construct depth 1 as the validated configured all-resident schedule,
  including any progress-required wave alignment and before any derived
  readiness-major transformation. Assign
  depth 2 to the first noncanonical pull, propagate one plus the maximum pulled
  predecessor depth through newly early work, preserve depth across waves, and
  reset it only at an exact rejoin with the canonical frontiers. Reject a pull
  deeper than the configured value. One depth is uniform over the complete
  symbolic guard.
- [ ] Fill an abstract wave while a complete eligible candidate fits. Place a
  dependent cohort in an earlier wave only when its complete exact event-key
  fiber fits proved idle committed-run tail lanes or every ancestor it could
  block is already assigned. In the latter case, make the whole fiber one
  committed multi-wave run; per-wave relation pieces may clip its encoding but
  no candidate may interleave before it finishes. Permit broader interleaving
  only between semantically incomparable roots in the root/event quotient.
- [ ] Detect a guard-uniform affine translation of the finite control state and
  lift it into bounded prefix/full-repeat/tail relation pieces. Prove the
  priority winner, readiness pattern, lane pattern, and positive cursor/frontier
  delta for every repetition. Decline runtime-dependent periods or winners.
- [ ] Emit the result directly as existing
  `WorkerScheduleSegment.task_order` relations. Require exact disjoint
  coverage, exact converses, acyclic progress, resident capacity, and bounded
  relation complexity. Record occupied horizon and admission frontiers as
  diagnostics; autotuning, not a compiler profitability proof, compares them.
- [ ] Keep all consumers resident for the first complete implementation.
  Evaluate continuation only afterward with the local all-publishers dominance
  rule. Never call a symbolic completion scorer.
- [ ] Keep the concrete CTA list scheduler and max-plus evaluator in tests only:
  substitute small dynamic bounds, compare coverage/readiness and qualitative
  choices, and use disagreements to select GPU ablations.
- [ ] Enforce the common 4,096-piece and 65,536-product budgets before
  constructing intermediates. Production compilation must not enumerate
  runtime tasks, keys, workers, waves, or shapes.
- [ ] Test exact-or-decline behavior on FlashMLA fan-out, Qwen's mixed-radix
  74/22 frontier, Gemma's producer-tail reduction, Muse's 32/16 tail,
  Nemotron/DeepSeek fork-join packing, an independent graph, a dense chain,
  runtime-empty roots, and deliberately unresolved symbolic winners. Exercise
  all four depth values even when multiple values produce the same relation.

Implementation checkpoint (2026-09-11): the former one-shot tail pull has
been replaced by one bounded walk over the finite unique-root schema. The walk
keeps only local root cursors/assigned roots, active causal depths, emitted
slot mass, and a run list; it repacks once into the existing
`WorkerSchedule`. Whole-root cohorts may span waves. Independent pulls start
at depth 2, dependent pulls use one plus the maximum active predecessor depth,
and depth resets only when the assigned-root cursor vector exactly equals the
corresponding canonical prefix. Tests cover the depth-1--4 chain, independent
pulls, a join, a full-wave boundary, a true rejoin, exact canonical ties, and
positive symbolic whole-root substitution. Construction is bounded by the
finite root schema and the common relation budgets; no task, wave, shape, or
runtime extent is enumerated.

This is deliberately an incremental checkpoint, not completion of Phase
4A.3. A strict sub-root cohort is still one terminal action. An oversized
strict cohort vetoes the boundary instead of disappearing or becoming a
multi-wave committed run. Event closure and active-cohort continuation remain
neutral unless their omission is provably irrelevant (for example, a whole
root with no producer role); ties involving producer roles retain `C` until
key-scoped frontiers exist. A possibly empty moved root also retains `C` until
the optimized proposal carries its synthetic publication occurrence. These
tests encode safe migration declines and must be revised when the missing
proofs land; they are not the final scheduling policy.

The supporting relation/lowering fixes remain general. A total
`CoordinateRelation` now derives its exact converse and symbolic target count
once, enabling a positive symbolic whole root to be both an intermediate
producer and consumer in a depth-3 chain. Root-barrier lowering uses the same
packed schedule relation: a unique symbolic producer gets the existing
participant-order relation even when another root repeats; a parameter-free
split producer uses the exact final-worker interval partition; a symbolically
split producer still declines. Invalid constant fibers with out-of-domain or
conditionally empty targets are rejected before a converse is memoized.

Immediate continuation of this phase:

1. carry exact consumer-key frontiers through the finite walk and use them for
   event closure and active-cohort continuation;
2. commit a complete strict sub-root cohort across waves, rather than treating
   it as a tail-only action, while preserving its atomicity;
3. retain conditionally empty moved roots together with their synthetic
   publication occurrence;
4. detect guard-uniform affine repetition and lift it into bounded relation
   pieces; and
5. only then expose the public depth knob and remove the legacy policy split.

Exit gate: constant and symbolic instances with extensionally equal guards,
orders, readiness, worker count, and capacity facts select the same cohort
policy; dynamic repetitions remain bounded relation pieces; Qwen and Gemma
retain configured orders where a readiness-major replacement is not proved;
and no max-plus, model name, sampled shape, or measured latency is consulted.

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
- Permit concrete loops, runtime-bounded loops, interval publishers, and fixed-
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
2. Qwen3 B1 and mixed-context B greater than one;
3. Gemma A4B B1/B2 and varied expert IDs;
4. Muse/Glimmer FFN;
5. Nemotron routed-first probe; and
6. DeepSeek-V3 routed/shared probe.

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

### `helion/runtime/config.py`

- Add `cross_loop_pipeline_depth` as one ordinary integer config field with
  default 1 and document that it affects only `static_pipeline` scheduling.
- Include it in normal config serialization, equality, display, cache identity,
  and pretuned configs; add no per-root or runtime-shape form.

### `helion/autotuner/config_spec.py`

- Register the field in the existing valid/configurable key tables.
- When cross-loop static-pipeline scheduling is available, expose the fixed
  `IntegerFragment(1, 4, 1)` unconditionally. Do not inspect topology, prune
  aliases, or synthesize a different bound for a particular kernel.
- Reuse normal candidate generation and `autotune_multi`; add no scheduler-
  specific search or objective implementation.

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
- Constant/symbolic local continuation-dominance parity under extensionally
  equal guards/orders/worker facts, plus a narrower-guard case that
  legitimately proves a different choice, a correlated publisher chain that
  declines, and an ineligible competitor.
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
- Config default/range, serialization/cache identity, pretuned round trip, and
  all four pipeline-depth values even when several lower identically.
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
11. resident placement is selected first for constant and symbolic domains;
    continuation ownership changes it only through the same local dominance
    proof over every possible publisher strand, and resident wins every
    unproved case;
12. nested counter partitions are exact quotients of the accepted schedule and
    are never derived from a schedule that a later pass mutates;
13. source tickets use a source-first allocator and capacity certificate;
14. there is one event-frontier scheduler and no concrete/parameterized or
    local/global policy split; extensionally identical normalized policy inputs
    yield the same decisions, while a narrower guard may expose additional
    proofs through the same algorithm;
15. priority uses the documented unit-weight structural model and exact
    booleans only; runtime extents, symbolic remaining-work comparisons,
    measured/profiled latency, resource estimates, and model-specific rules
    are absent; `cross_loop_pipeline_depth` in the fixed range 1--4 is the only
    scheduler-specific autotune dimension; all four values are offered without
    topology pruning or alias removal, depth 1 reproduces canonical placement,
    and one value governs a complete polymorphic guard;
16. FlashMLA B4/B9 retain performance and Q1/Q2 do not regress;
17. both untouched pretuned and mechanically related symbolic B1/B2 source
    forms of Qwen3 and Gemma retain performance through the same policy;
18. DeepSeek-V3 and Nemotron retain branch-packing wins;
19. Muse compiles compactly, retains its 32/16 tail, and rejects fragmentation
    of its committed gate producer run;
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
