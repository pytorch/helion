# Local-order `WorkerSchedule` simplification

Status: implemented and validated on 2026-09-13 as a follow-up to the
`cross_loop_pipeline = barrier | static | dynamic` design.

## Goal

Remove placement generality left behind by the retired cross-root list
scheduler. Preserve the current schedule, generated code, correctness, and
performance while making each resident segment describe only its root-local
task order.

This refactor must add no new compiler abstraction and must not change fusion,
numerics, continuation policy, or scheduling policy.

## Final representation

`WorkerScheduleSegment` stores only:

- `root`; and
- the existing compact, possibly multidimensional, dense-source
  `task_order -> logical task` relation.

`WorkerSchedule` stores the ordered segments and the kernel-wide worker count
`W`. There remains exactly one segment for every non-continuation root, in
source order. The local task-order relation must be a total bijection onto that
root's fixed-capacity logical domain.

Physical placement is derived, never stored:

```text
static_wave_base(r) = sum(ceil(T[j] / W) for prior resident roots j)
static_slot(r, o)   = W * static_wave_base(r) + o
static_worker       = static_slot % W
static_wave         = static_slot // W

dynamic_ticket_base(r) = sum(T[j] for prior resident roots j)
dynamic_ticket(r, o)   = dynamic_ticket_base(r) + o
```

Static roots must remain wave-aligned. For example, with `W=4` and root sizes
`(3, 2)`, root 0 uses wave 0/workers 0--2 and root 1 starts at wave 1/workers
0--1. Densely packing root 1 into worker 3 of wave 0 would be a scheduling
change and would invalidate existing continuation proofs.

The local task order stays multidimensional when that is its compact form.
Flattening to an ordinal is a derived view used by codegen and proofs; it is
not a second authoritative relation. This is important for Qwen's woven/L2
orders.

## Implementation stages

### 1. Remove algebra that has no production caller

Delete these abandoned max-plus/parametric-scheduler operations and their
dedicated tests:

- `CoordinateRelation.pointwise_add_scalar`;
- `CoordinateRelation.weighted_max_target_value_and_attainers_by_source`;
- `CoordinateRelation.reorder_source_axes`; and
- `CoordinateRelation.is_pointwise_strictly_less_than`.

Keep `is_pointwise_strictly_less_than_where_defined`, ordinary max/argmax,
symbolic substitution, and materialization used as a differential-test oracle.
This stage should remove about 346 production lines without changing lowering.

### 2. Make local order authoritative

- Reduce `WorkerScheduleSegment` to `root` and `task_order`.
- Validate each local order once: dense source, exact total function, exact
  total converse, and full coverage of the configured root domain.
- Derive one reusable local ordinal view from that relation when needed.
- Build static wave bases and dynamic ticket bases as simple prefix folds over
  the same ordered segment tuple.
- Preserve the current root-local reorder eligibility rule `T >= W` initially;
  broadening it is a separate scheduling experiment.

### 3. Move every consumer to derived placement

- Static codegen derives worker/wave from the padded static prefix.
- Dynamic codegen derives packet ranges from the dense ticket prefix.
- Same-root progress derives `logical task -> local ordinal -> static wave`.
- Static root-barrier participation is `[0, min(W, T))`; dynamic publication
  has exactly `T` arrivals.
- Continuation selection constructs its task-to-worker/wave relation
  temporarily from the local inverse and padded static base. It must not store
  a second placement truth.
- Continuation contraction removes the consumer root and recomputes both
  prefix folds transactionally.

### 4. Delete obsolete placement machinery

After all consumers use the derived formulas, remove:

- `worker_begin`, per-segment `worker_count`, and `dispatch_offset`;
- normalized worker-domain schedule relations and their round-trip validator;
- normalized placement reconstruction and geometry classifiers;
- multi-segment traversal recovery and compatibility methods;
- relation-valued root-barrier participant ordering when simple counts suffice;
- test-only `WorkerSchedule.without_roots`; and
- any relation helper whose final production caller disappeared, including
  `has_same_source_support`.

Keep the compact packed-interval constructors used to derive exact static
placement for continuations and woven/L2 orders. Also keep
`rebase_source_domain`: continuation selection uses it to compare resident and
virtual ownership in one widened ambient worker domain without introducing a
second placement representation.

Retain the core dependency algebra: exact composition/converse/union,
projection and coalescing, source support and bijection proofs,
`producer_set_quotient`, max/argmax with attainers, partial strict ordering,
and mixed-radix/woven inverse support.

Expected reduction is roughly 1.3--1.8k scheduler/codegen lines in addition to
the initial dead-algebra deletion, with further relation cleanup determined by
the post-refactor call graph.

## Required gates

Each stage must pass before deleting its compatibility implementation:

1. CPU relation, scheduler, exact-converse, and configuration suites.
2. The `W=4`, root-size `(3, 2)` wave-alignment test.
3. Qwen woven-order forward/converse and 74/22 frontier tests.
4. Qwen continuations `{7, 8, 10}` with root 13 resident.
5. Gemma root 6 as a continuation and root 7 resident.
6. Static Qwen/Gemma generated-source hashes, SASS/resources, correctness, and
   cold-L2 latency remain equivalent.
7. Dynamic MLA B4/B9 and Muse correctness and cold-L2 performance remain
   equivalent; FlashMLA Q1/Q2 continue covering root-barrier and continuation
   behavior.
8. DeepSeek and Nemotron retain the existing dynamic-over-static signal.

If preserving an old arbitrary-offset, rotated-worker, repeated-root, or
interleaved-root test requires restoring general placement state, delete or
rewrite that test: those schedule forms are outside the final compiler design.

## Implementation result

- `WorkerScheduleSegment` now stores only `root` and `task_order`.
- The obsolete plural/multi-segment lookup was replaced by the invariant-aware
  `segment_for_root` view.
- Static placement uses a derived wave-padded prefix; dynamic dispatch uses a
  derived dense ticket prefix over the same segment tuple.
- Codegen walks that authoritative segment tuple directly. It does not retain
  or reconstruct a second schedule geometry.
- Five unused `CoordinateRelation` operations and the tests dedicated only to
  retired placement forms were removed.
- The compiler/test diff deletes over 4,300 lines (about 3,700 net) while
  adding no new compiler abstraction.

Validation:

- 459 scheduler, dependency, loop, configuration, and exact-converse tests
  passed (plus 242 subtests; 3 skipped).
- 40 cross-loop codegen tests passed (plus 18 subtests).
- Representative static and dynamic generated Triton are byte-identical to
  the pre-refactor compiler (8,008-byte and 7,075-byte fixtures respectively).
- FlashMLA B4 dynamic: 61.264 us persistent versus 69.408 us standalone,
  bit-exact.
- FlashMLA B9 ragged dynamic: 89.984 us persistent versus 102.336 us
  standalone, bit-exact.
- Pretuned Gemma4 A4B B1 static: 51.136 us persistent versus 55.296 us
  standalone, all outputs bit-exact.
- Pretuned Qwen dynamic-shape and same-source static-shape compilation produce
  identical normalized device code and the same R255/spill22/17,408-byte
  resource envelope; the dynamic-shape isolated cold-L2 result was 100.480 us.
