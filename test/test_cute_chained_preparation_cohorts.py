from __future__ import annotations

import ast
from collections import Counter
from dataclasses import FrozenInstanceError
import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from helion._compiler.cute.chained_preparation_cohorts import PreparationCohorts
from helion._compiler.cute.chained_preparation_cohorts import plan_preparation_cohorts


def _plan(count=3, *, cohort_threads=128, recurrence_threads=128, has_tma=True):
    plan = plan_preparation_cohorts(
        count * cohort_threads + recurrence_threads,
        recurrence_threads,
        count,
        has_tma=has_tma,
    )
    assert plan is not None
    return plan


@pytest.mark.parametrize("cta_threads", range(128, 1025, 128))
@pytest.mark.parametrize("recurrence_threads", range(128, 1025, 128))
def test_all_physical_team_partitions_and_one_team_fallback(
    cta_threads, recurrence_threads
):
    for count in range(1, 9):
        plan = plan_preparation_cohorts(cta_threads, recurrence_threads, count)
        expected = (
            count > 1
            and recurrence_threads < cta_threads
            and (cta_threads - recurrence_threads) % (128 * count) == 0
        )
        assert (plan is not None) is expected
        if plan is not None:
            assert plan.cohort_threads >= 128
            assert plan.slots == count
            assert plan.cohort_threads * count + recurrence_threads == cta_threads
            assert len(set(plan.named_barrier_ids)) == count
            assert min(plan.named_barrier_ids) >= 5
            assert max(plan.named_barrier_ids) < 16


@pytest.mark.parametrize(
    "field", ("cta_threads", "recurrence_threads", "count", "named_barrier_base")
)
@pytest.mark.parametrize("value", (True, False, None, 3.0, "3", -1, 0))
def test_integer_fields_are_strict_in_planner_and_record(field, value):
    args = {
        "cta_threads": 512,
        "recurrence_threads": 128,
        "count": 3,
        "named_barrier_base": 5,
    }
    args[field] = value
    assert plan_preparation_cohorts(**args) is None
    with pytest.raises(ValueError, match="cohort resources"):
        PreparationCohorts(**args)


@pytest.mark.parametrize("value", (0, 1, None, "True", [], {}))
def test_tma_presence_requires_an_actual_boolean(value):
    assert plan_preparation_cohorts(512, 128, 3, has_tma=value) is None
    with pytest.raises(ValueError, match="cohort resources"):
        PreparationCohorts(512, 128, 3, value)


@pytest.mark.parametrize(
    "args", ((512, 128, 2), (384, 64, 2), (1152, 128, 2), (1024, 128, 8), (480, 128, 2))
)
def test_partial_warpgroups_unequal_teams_and_cta_overflow_reject(args):
    assert plan_preparation_cohorts(*args) is None


@pytest.mark.parametrize("count", range(2, 8))
def test_named_barrier_capacity_reserves_existing_protocol_ids(count):
    cta = (count + 1) * 128
    assert plan_preparation_cohorts(cta, 128, count, named_barrier_base=4) is None
    exact = plan_preparation_cohorts(cta, 128, count, named_barrier_base=16 - count)
    assert exact is not None and exact.named_barrier_ids[-1] == 15
    assert (
        plan_preparation_cohorts(cta, 128, count, named_barrier_base=17 - count) is None
    )


@pytest.mark.parametrize("count", range(2, 8))
@pytest.mark.parametrize("has_tma", (False, True))
def test_slot_barriers_are_private_disjoint_and_account_all_aligned_bytes(
    count, has_tma
):
    plan = _plan(count, has_tma=has_tma)
    indices = [
        value
        for slot in range(count)
        for value in plan.barrier_indices(slot)
        if value is not None
    ]
    assert sorted(indices) == list(range(count * (2 + int(has_tma))))
    assert plan.slot_mbarrier_count == len(indices)
    assert plan.slot_mbarrier_bytes == len(indices) * 8
    assert plan.slot_mbarrier_allocated_bytes % 128 == 0
    assert (
        plan.slot_mbarrier_bytes
        <= plan.slot_mbarrier_allocated_bytes
        < plan.slot_mbarrier_bytes + 128
    )
    pointers = plan.barrier_pointers()
    for slot in range(count):
        actual = tuple(
            eval(value, {"chain_slot_bars": 0, "chain_slot": slot})
            if value is not None
            else None
            for value in pointers
        )
        assert actual == plan.barrier_indices(slot)
    if count == 7 and has_tma:
        assert plan.slot_mbarrier_allocated_bytes == 256


@pytest.mark.parametrize("slot", (-1, 3, True, None, 1.0))
def test_invalid_slot_cannot_address_other_protocol_regions(slot):
    with pytest.raises(ValueError, match="cohort slot"):
        _plan().barrier_indices(slot)


@pytest.mark.parametrize("step", (True, None, 0, -1, 1.0))
def test_invalid_loop_step_rejects_before_source_emission(step):
    for method in (
        _plan().producer_header,
        _plan().consumer_header,
        _plan().iteration_bindings,
    ):
        with pytest.raises(ValueError, match="positive integer"):
            method(step)


def _range(header, bindings):
    loop = ast.parse(header + "\n    pass").body[0]
    assert isinstance(loop, ast.For) and isinstance(loop.iter, ast.Call)
    assert ast.unparse(loop.iter.func) == "cutlass.range"
    assert [
        (keyword.arg, ast.literal_eval(keyword.value)) for keyword in loop.iter.keywords
    ] == [("unroll", 1)]
    return range(
        *(
            eval(compile(ast.Expression(arg), "header", "eval"), bindings)
            for arg in loop.iter.args
        )
    )


@pytest.mark.parametrize(
    "count,cohort_threads,recurrence_threads",
    ((2, 128, 128), (3, 128, 128), (2, 256, 256), (2, 384, 256), (7, 128, 128)),
)
def test_emitted_coordinates_cover_zero_ragged_and_nonzero_origin_loops_once(
    count, cohort_threads, recurrence_threads
):
    plan = _plan(
        count, cohort_threads=cohort_threads, recurrence_threads=recurrence_threads
    )
    for begin in (0, 7, 128):
        for step in (1, 3, 16):
            for chunks in (0, 1, count - 1, count, count + 1, 2 * count + 1):
                stop = begin + max(0, chunks * step - (step - 1))
                base = {"chain_loop_begin": begin, "chain_loop_end": stop}
                expected = list(range(begin, stop, step))
                assert list(_range(plan.consumer_header(step), base)) == expected
                actual = []
                for cohort in range(count):
                    for index in _range(
                        plan.producer_header(step), base | {"chain_cohort": cohort}
                    ):
                        values = base | {"chain_loop_index": index}
                        exec("\n".join(plan.iteration_bindings(step)), values)
                        assert values["chain_slot"] == cohort
                        assert values["chain_iteration"] == expected.index(index)
                        assert (
                            eval(plan.completion_phase, values)
                            == (expected.index(index) // count) % 2
                        )
                        actual.append(index)
                assert Counter(actual) == Counter(expected)


@pytest.mark.parametrize(
    "count,cohort_threads", ((2, 128), (3, 128), (2, 256), (2, 384), (7, 128))
)
def test_role_local_threads_warps_and_whole_team_barrier_participation(
    count, cohort_threads
):
    plan = _plan(count, cohort_threads=cohort_threads)
    barriers = Counter()
    owners = []
    pipeline = SimpleNamespace(NamedBarrier=lambda **kwargs: kwargs)
    for thread in range(plan.preparation_threads):
        values: dict[str, Any] = {
            "chain_thread": thread,
            "chain_pipeline": pipeline,
        }
        exec("\n".join(plan.preparation_bindings()), values)
        assert values["chain_prep_warp"] == values["chain_prep_thread"] // 32
        owners.append((values["chain_cohort"], values["chain_prep_thread"]))
        barrier = values["chain_prep_barrier"]
        assert barrier["num_threads"] == cohort_threads
        barriers[barrier["barrier_id"]] += 1
    assert len(set(owners)) == plan.preparation_threads
    assert barriers == Counter(dict.fromkeys(plan.named_barrier_ids, cohort_threads))
    for thread in range(plan.preparation_threads, plan.cta_threads):
        values = {"chain_thread": thread}
        exec("\n".join(plan.recurrence_bindings()), values)
        assert values["chain_recurrence_thread"] == thread - plan.preparation_threads
        assert (
            values["chain_recurrence_warp"] == (thread - plan.preparation_threads) // 32
        )
    execution = plan.preparation_execution()
    assert execution.threads == cohort_threads
    assert execution.sync == "chain_prep_barrier.arrive_and_wait()"
    assert "sync_threads" not in "\n".join(plan.preparation_bindings())
    with pytest.raises(FrozenInstanceError):
        plan.count = 1  # pyrefly: ignore [read-only]


def _simulate(plan, chunks, *, wrong_phase=False, shared_tma=False):
    """Explore all interleavings; predicates observe parity, assertions versions.

    Slot states: empty, preparing, published, consumer reading. Producer steps:
    acquire, async issue, async completion, publish. Consumer steps: await,
    async reads in flight, reads complete/release. EMPTY cannot precede the last
    step, even if the last MMA has already issued.
    """
    count = plan.count
    initial = (
        (0,) * count,
        (0,) * count,
        (0,) * count,
        0,
        0,
        (-1,) * count,
        (-1,) * count,
    )
    pending, seen = [initial], {initial}
    terminals = 0
    while pending:
        progress, stages, slots, consumed, consumer_stage, ready, empty = pending.pop()
        successors = []
        for cohort in range(count):
            iteration = progress[cohort] * count + cohort
            if iteration >= chunks:
                continue
            generation, stage = progress[cohort], stages[cohort]
            p, s, f, r = list(progress), list(stages), list(slots), list(ready)
            if stage == 0:
                phase = eval(plan.reuse_phase, {"chain_generation": generation})
                if generation and empty[cohort] & 1 != phase:
                    continue
                assert slots[cohort] == 0, "overwrite before consumer completion"
                assert not generation or empty[cohort] == generation - 1
                f[cohort], s[cohort] = 1, 1
            elif stage == 1:
                assert slots[cohort] == 1
                if plan.has_tma:
                    barrier = (
                        2 * count if shared_tma else plan.barrier_indices(cohort)[2]
                    )
                    for other in range(count):
                        other_barrier = (
                            2 * count if shared_tma else plan.barrier_indices(other)[2]
                        )
                        assert (
                            other == cohort
                            or stages[other] != 2
                            or other_barrier != barrier
                        ), "concurrent transactions share one TMA barrier"
                s[cohort] = 2
            elif stage == 2:
                values = {"chain_generation": generation, "chain_iteration": iteration}
                phase = eval(
                    "chain_iteration & 1" if wrong_phase else plan.completion_phase,
                    values,
                )
                assert phase == generation & 1, "wrong slot-generation TMA parity"
                assert slots[cohort] == 1
                s[cohort] = 3
            else:
                assert slots[cohort] == 1
                f[cohort], r[cohort], p[cohort], s[cohort] = (
                    2,
                    generation,
                    generation + 1,
                    0,
                )
            successors.append(
                (
                    tuple(p),
                    tuple(s),
                    tuple(f),
                    consumed,
                    consumer_stage,
                    tuple(r),
                    empty,
                )
            )
        if consumed < chunks:
            slot, generation = consumed % count, consumed // count
            f, e = list(slots), list(empty)
            phase = eval(plan.completion_phase, {"chain_generation": generation})
            if consumer_stage == 0 and ready[slot] & 1 == phase:
                assert slots[slot] == 2 and ready[slot] == generation
                f[slot] = 3
                successors.append(
                    (progress, stages, tuple(f), consumed, 1, ready, empty)
                )
            elif consumer_stage == 1:
                assert slots[slot] == 3 and ready[slot] == generation
                successors.append((progress, stages, slots, consumed, 2, ready, empty))
            elif consumer_stage == 2:
                assert slots[slot] == 3 and ready[slot] == generation
                f[slot], e[slot] = 0, generation
                successors.append(
                    (progress, stages, tuple(f), consumed + 1, 0, ready, tuple(e))
                )
        if not successors:
            assert consumed == chunks
            assert all(slot == 0 for slot in slots)
            assert all(stage == 0 for stage in stages)
            assert all(
                progress[j] == len(range(j, chunks, count)) for j in range(count)
            )
            terminals += 1
        for state in successors:
            if state not in seen:
                seen.add(state)
                pending.append(state)
    assert terminals == 1
    return len(seen)


@pytest.mark.parametrize(
    "count,chunks",
    [(count, chunks) for count in (2, 3) for chunks in (*range(10), 16, 32)]
    + [(4, chunks) for chunks in range(6)]
    + [(7, 0), (7, 1), (7, 2)],
)
@pytest.mark.parametrize("has_tma", (False, True))
def test_exhaustive_async_protocol_keeps_order_and_prevents_early_slot_reuse(
    count, chunks, has_tma
):
    assert _simulate(_plan(count, has_tma=has_tma), chunks) >= 1


@pytest.mark.parametrize("count", (2, 3, 4))
def test_wrong_global_iteration_phase_and_shared_tma_barrier_fail(count):
    plan = _plan(count)
    with pytest.raises(AssertionError, match="slot-generation TMA parity"):
        _simulate(plan, count + 1, wrong_phase=True)
    with pytest.raises(AssertionError, match="share one TMA barrier"):
        _simulate(plan, count, shared_tma=True)


@pytest.mark.parametrize("dtype_name", ("BFloat16", "Float16"))
def test_actual_cute_four_warp_cohort_preserves_each_eight_warp_atom_input(dtype_name):
    """Original N-warp ownership becomes ordered N repeats, not a new K sum."""
    cutlass = pytest.importorskip("cutlass")
    cute = importlib.import_module("cutlass.cute")
    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = vars(cutlass)[dtype_name]
    plan = _plan()
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            atom = cute.make_mma_atom(
                cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16))
            )
            old = cute.make_tiled_mma(atom, atom_layout_mnk=(1, 8, 1))
            new = cute.make_tiled_mma(
                atom, atom_layout_mnk=(1, plan.cohort_threads // 32, 1)
            )

            def coordinates(tensor):
                return [
                    tuple(int(x) for x in tensor[i]) for i in range(cute.size(tensor))
                ]

            for tid in range(256):
                previous, current = old.get_slice(tid), new.get_slice(tid % 128)
                repetition = tid // 128
                old_c = previous.partition_C(cute.make_identity_tensor((32, 64)))
                new_c = current.partition_C(cute.make_identity_tensor((32, 64)))
                assert coordinates(old_c[None, None, 0]) == coordinates(
                    new_c[None, None, repetition]
                )
                for k in (32, 128):
                    old_a = previous.partition_A(cute.make_identity_tensor((32, k)))
                    new_a = current.partition_A(cute.make_identity_tensor((32, k)))
                    assert coordinates(old_a) == coordinates(new_a)
                    old_b = previous.partition_B(cute.make_identity_tensor((64, k)))
                    new_b = current.partition_B(cute.make_identity_tensor((64, k)))
                    for kk in range(k // 16):
                        assert coordinates(old_b[None, 0, kk]) == coordinates(
                            new_b[None, repetition, kk]
                        )
        assert module.operation.verify()
