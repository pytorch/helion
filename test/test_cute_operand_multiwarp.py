"""Typed roles, ownership, lifetime and resource proofs for dormant RNA plans."""

from __future__ import annotations

import ast
import ctypes
from dataclasses import asdict
from dataclasses import replace
import json
import pickle
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable

from .test_cute_operand_pipeline import _run_stage
from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandDescriptorPlan
from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandResourceUsage
from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandRolePlan
from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandTransformKind
from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandTransformPlan
from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandType
from helion._compiler.cute.tcgen05_operand_transform import prove_operand_stage_span
from helion._compiler.cute.tcgen05_pipeline import make_pipeline_state
from helion.runtime.cute.launcher import _cute_disk_cache_key

if TYPE_CHECKING:
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
pipeline = pytest.importorskip("cutlass.pipeline")
ir = pytest.importorskip("cutlass._mlir.ir")


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with _mock_cuda_unavailable():

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("GPU forbidden in the multiwarp operand proof")

        monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
        yield


@pytest.mark.parametrize(
    "warps,threads,padding",
    ((1, 256, ()), (2, 384, (9, 10, 11)), (4, 384, (11,)), (8, 512, (15,))),
)
@pytest.mark.parametrize("self_consumer", (False, True))
def test_roles_partition_the_whole_block_and_keep_arrival_units_distinct(
    warps: int, threads: int, padding: tuple[int, ...], self_consumer: bool
) -> None:
    roles = Tcgen05OperandRolePlan(warps, self_consumer)
    assert roles.threads_per_cta == threads and threads % 128 == 0
    assert roles.thread_block_dims == (threads, 1, 1)
    assert roles.padding_warp_ids == padding
    consumers = (
        *roles.epilogue_warp_ids,
        roles.mma_warp,
        roles.raw_tma_warp,
        *roles.converter_warp_ids,
        roles.scheduler_warp,
    )
    assert consumers == tuple(range(7 + warps))
    assert (*consumers, *padding) == tuple(range(threads // 32))
    assert roles.scheduler_empty_warp_arrivals == len(consumers) - int(
        not self_consumer
    )
    assert roles.scheduler_consumer_warp_ids == tuple(
        range(6 + warps + int(self_consumer))
    )
    assert (roles.scheduler_warp in roles.scheduler_consumer_warp_ids) == self_consumer
    assert roles.converted_full_lane_arrivals == warps * 32
    assert roles.converted_full_lane_arrivals != roles.scheduler_empty_warp_arrivals
    assert [roles.converter_rank(warp) for warp in roles.converter_warp_ids] == list(
        range(warps)
    )
    for warp in (
        *roles.epilogue_warp_ids,
        roles.mma_warp,
        roles.raw_tma_warp,
        roles.scheduler_warp,
        *padding,
        -1,
        threads // 32,
    ):
        with pytest.raises(ValueError, match="conversion role"):
            roles.converter_rank(warp)


@pytest.mark.parametrize("warps", (0, -1, True, False, 1.0, 3, 6, 16, 32))
def test_only_proved_integer_converter_choices_are_configurable(warps: int) -> None:
    with pytest.raises(ValueError, match="proved converter warp choices"):
        Tcgen05OperandRolePlan(warps)


@pytest.mark.parametrize("self_consumer", (0, 1, "false", None))
def test_scheduler_participation_requires_a_typed_fact(self_consumer: bool) -> None:
    with pytest.raises(ValueError, match="explicit bool"):
        Tcgen05OperandRolePlan(4, self_consumer)


@pytest.mark.parametrize("warps", (1, 2, 4, 8))
def test_role_and_resource_records_serialize_without_losing_types(warps: int) -> None:
    roles = Tcgen05OperandRolePlan(warps)
    usage = Tcgen05OperandResourceUsage(roles, 101, 1024, 230400)
    payload = json.loads(json.dumps(asdict(usage)))
    restored = Tcgen05OperandResourceUsage(
        Tcgen05OperandRolePlan(**payload.pop("roles")), **payload
    )
    assert restored == usage
    assert pickle.loads(pickle.dumps(usage)) == usage
    assert Tcgen05OperandRolePlan(**json.loads(json.dumps(asdict(roles)))) == roles


@pytest.mark.parametrize(
    "warps,registers,allocated",
    ((1, 93, 24576), (2, 93, 36864), (4, 93, 36864), (8, 101, 53248)),
)
def test_physical_padding_and_both_shared_storage_parts_are_charged(
    warps: int, registers: int, allocated: int
) -> None:
    usage = Tcgen05OperandResourceUsage(
        Tcgen05OperandRolePlan(warps), registers, 1024, 230400
    )
    assert usage.allocated_registers_per_cta == allocated
    assert usage.shared_bytes == 231424
    assert usage.fits(shared_capacity_bytes=232448)
    assert usage.fits(shared_capacity_bytes=usage.shared_bytes)
    assert not usage.fits(shared_capacity_bytes=usage.shared_bytes - 1)
    if usage.roles.padding_warp_ids:
        unpadded = ((registers + 7) // 8) * 8 * usage.roles.active_warps * 32
        assert unpadded < usage.allocated_registers_per_cta
    # Actual whole-object usage is authoritative even when operands alone fit.
    overflow = replace(usage, static_shared_bytes=3073)
    assert not overflow.fits(shared_capacity_bytes=232448)


def test_register_granularity_can_reject_an_apparently_fitting_w8_schedule() -> None:
    usage = Tcgen05OperandResourceUsage(Tcgen05OperandRolePlan(8), 128, 1024, 230400)
    assert usage.allocated_registers_per_cta == 65536
    assert usage.fits(shared_capacity_bytes=232448)
    over = replace(usage, registers_per_thread=129)
    assert over.allocated_registers_per_cta == 69632
    assert not over.fits(shared_capacity_bytes=232448)
    assert usage.roles == over.roles  # Rejection never changes the schedule.


@pytest.mark.parametrize("registers", (0, -1, 256, True, 93.0))
def test_unproved_register_facts_decline(registers: int) -> None:
    with pytest.raises(ValueError, match="register count"):
        Tcgen05OperandResourceUsage(Tcgen05OperandRolePlan(4), registers, 0, 0)


@pytest.mark.parametrize("redistribution", (True, 0, None))
def test_uniform_resource_proof_does_not_admit_runtime_redistribution(
    redistribution: bool,
) -> None:
    with pytest.raises(ValueError, match="separate accounting"):
        Tcgen05OperandResourceUsage(
            Tcgen05OperandRolePlan(4), 93, 1024, 230400, redistribution
        )


@pytest.mark.parametrize("bytes_", (-1, True, 1024.0))
def test_unproved_shared_byte_facts_decline(bytes_: int) -> None:
    roles = Tcgen05OperandRolePlan(4)
    for static, dynamic in ((bytes_, 0), (0, bytes_)):
        with pytest.raises(ValueError, match="byte counts"):
            Tcgen05OperandResourceUsage(roles, 93, static, dynamic)
    with pytest.raises(ValueError, match="positive byte limit"):
        Tcgen05OperandResourceUsage(roles, 93, 1024, 230400).fits(
            shared_capacity_bytes=bytes_
        )


def _plan(
    warps: int, a_words: int, b_words: int, stride: int | None = None
) -> Tcgen05OperandTransformPlan:
    a = prove_operand_stage_span(
        cute.make_layout(
            (a_words, 2), stride=(1, a_words if stride is None else stride)
        ),
        stages=2,
        alignment_bytes=16,
    )
    b = prove_operand_stage_span(
        cute.make_layout((b_words, 2), stride=(1, b_words)),
        stages=2,
        alignment_bytes=16,
    )
    assert a is not None and b is not None
    return Tcgen05OperandTransformPlan(
        Tcgen05OperandTransformKind.TF32_RNA,
        Tcgen05OperandDescriptorPlan(
            Tcgen05OperandType.FLOAT32,
            Tcgen05OperandType.FLOAT32,
            Tcgen05OperandType.TFLOAT32,
        ),
        a,
        b,
        warps,
    )


@pytest.mark.parametrize("warps", (2, 4, 8))
def test_each_operand_requires_its_own_complete_multiwarp_packets(warps: int) -> None:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            quantum = 128 * warps
            plan = _plan(warps, quantum * 3, quantum * 5)
            assert plan.a.words != plan.b.words
            assert (
                plan.converted_full_arrivals == plan.roles.converted_full_lane_arrivals
            )
            assert plan.scheduler_consumer_warps == warps
            for a_words, b_words in (
                (quantum - 128, quantum),
                (quantum, quantum - 128),
            ):
                with pytest.raises(ValueError, match="stage span"):
                    _plan(warps, a_words, b_words)


@pytest.mark.parametrize("warps", (1, 2, 4, 8))
def test_last_packet_address_is_proved_before_int32_arithmetic(warps: int) -> None:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            quantum = 128 * warps
            plan = _plan(warps, quantum, quantum, (1 << 31) - quantum)
            last = plan.a.stride_words + (warps * 32 - 1) * 4 + 3
            assert last == (1 << 31) - 1 and ctypes.c_int32(last).value == last
            with pytest.raises(ValueError, match="Int32 domain"):
                _plan(warps, quantum, quantum, (1 << 31) - quantum + 4)


def test_wrapper_cache_identity_separates_counts_roles_and_legacy_schema() -> None:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            plans = [_plan(warps, 2048, 4096) for warps in (1, 2, 4, 8)]
    identities = [plan.cache_identity() for plan in plans]
    assert len(set(identities)) == 4
    assert len({json.dumps(identity) for identity in identities}) == 4
    assert all(identity[0] == "tcgen05_operand_transform_v2" for identity in identities)
    assert all(pickle.loads(pickle.dumps(plan)) == plan for plan in plans)

    def key(
        plan: Tcgen05OperandTransformPlan, identity: tuple[object, ...]
    ) -> str | None:
        wrapper = {"kind": "tcgen05_ab_tma", "operand_transform": identity}
        kernel = SimpleNamespace(
            _helion_cute_source_hash="unchanged generated source",
            _helion_cute_wrapper_plans=[wrapper],
        )
        return _cute_disk_cache_key(
            kernel,
            (),
            (plan.roles.threads_per_cta, 1, 1),
            (repr(wrapper),),
            (1, 1, 1),
            "--enable-tvm-ffi",
            148,
        )

    keys = [key(plan, plan.cache_identity()) for plan in plans]
    assert all(result is not None for result in keys) and len(set(keys)) == 4
    self_consuming = replace(plans[2], scheduler_self_consumer=True)
    assert key(self_consuming, self_consuming.cache_identity()) not in keys
    legacy = ("tcgen05_operand_transform_v1", *identities[0][1:-2], "cta_group_one")
    assert key(plans[0], legacy) not in keys


@pytest.mark.parametrize("warps", (2, 4, 8))
@pytest.mark.parametrize("padded", (False, True))
def test_actual_multiwarp_helper_covers_both_spans_and_preserves_other_stages(
    warps: int, padded: bool
) -> None:
    quantum = 128 * warps
    a_words, b_words = quantum * 3, quantum * 5
    for stage in range(3):
        _run_stage(
            a_words=a_words,
            b_words=b_words,
            a_stride=a_words + (128 if padded else 0),
            b_stride=b_words + (256 if padded else 0),
            stages=3,
            stage=stage,
            converter_warps=warps,
        )


@pytest.mark.parametrize("warps", (2, 4, 8))
@pytest.mark.parametrize("fault", ("duplicate_rank", "missing_fence", "early_release"))
def test_actual_multiwarp_helper_exposes_ownership_and_publication_faults(
    warps: int, fault: str
) -> None:
    def mutate(tree: ast.Module) -> None:
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "convert_and_publish_operand_stage"
        )
        if fault == "duplicate_rank":
            function.body.insert(0, ast.parse("converter_warp_rank = 0").body[0])
        elif fault == "missing_fence":
            function.body.pop(-2)
        else:
            function.body.append(ast.parse("converted.consumer_release(state)").body[0])

    with pytest.raises(AssertionError):
        _run_stage(
            a_words=128 * warps,
            b_words=256 * warps,
            a_stride=128 * warps,
            b_stride=256 * warps,
            stages=3,
            stage=1,
            converter_warps=warps,
            mutate=mutate,
        )


@cute.jit
def _converter_phases(
    warps: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    counts: cutlass.Constexpr[tuple[int, ...]],
    address: cutlass.Int64,
) -> None:
    output = cute.make_tensor(
        cute.make_ptr(
            cutlass.Int32, address, cute.AddressSpace.generic, assumed_align=4
        ),
        cute.make_layout(warps * len(counts) * 3),
    )
    for rank in cutlass.range_constexpr(warps):
        state = make_pipeline_state(pipeline.PipelineUserType.Consumer, stages)
        for item in cutlass.range_constexpr(len(counts)):
            for _step in cutlass.range_constexpr(counts[item]):
                state.advance()
            slot = (rank * len(counts) + item) * 3
            output[slot] = state.index
            output[slot + 1] = state.phase
            output[slot + 2] = state.count


@pytest.mark.parametrize("warps", (1, 2, 4, 8))
@pytest.mark.parametrize("stages", (1, 2, 3, 6))
def test_actual_helion_sdk_has_one_independent_cursor_per_converter_warp(
    warps: int, stages: int
) -> None:
    counts = (0, 1, 0, 2, 7, 0, 19, 0)
    storage = (ctypes.c_int32 * (warps * len(counts) * 3))()
    address = ctypes.addressof(storage)
    compiled = cute.compile(
        _converter_phases,
        warps,
        stages,
        counts,
        cutlass.Int64(address),
        options="--gpu-arch sm_100a",
    )
    compiled(address)  # CPU host JIT over ordinary CPU memory; no GPU launch.
    for rank in range(warps):
        total = 0
        for item, count in enumerate(counts):
            total += count
            slot = (rank * len(counts) + item) * 3
            assert list(storage[slot : slot + 3]) == [
                total % stages,
                (total // stages) % 2,
                total,
            ]


def _interleaved_ring(
    warps: int, stages: int, seed: int, fault: str | None = None
) -> None:
    counts = (0, 1, 0, 7, 2, 0, 19)
    total = sum(counts)
    slots = [
        SimpleNamespace(kind="empty", ordinal=-1, published=set())
        for _ in range(stages)
    ]
    producer = mma = completed = 0
    converters = [0] * warps
    rng = random.Random(seed)
    arrivals = warps if fault == "warp_count" else 32 * warps
    while completed < total:
        actions = []
        if producer < total and slots[producer % stages].kind == "empty":
            actions.append(("raw", producer, 0))
        for slot in slots:
            if slot.kind == "dma":
                actions.append(("dma_complete", slot.ordinal, 0))
            if slot.kind == "mma":
                actions.append(("mma_complete", slot.ordinal, 0))
        for rank, ordinal in enumerate(converters):
            slot = slots[ordinal % stages]
            if ordinal < total and slot.ordinal == ordinal and slot.kind == "raw":
                actions.append(("convert", ordinal, rank))
        next_slot = slots[mma % stages]
        if (
            mma < total
            and next_slot.ordinal == mma
            and next_slot.kind == "raw"
            and len(next_slot.published) >= arrivals
        ):
            actions.append(("mma", mma, 0))
        assert actions, "pipeline stalled after an invalid early recycle"
        action, ordinal, rank = (
            next((a for a in actions if a[0] == "mma"), actions[0])
            if fault == "warp_count"
            else rng.choice(actions)
        )
        slot = slots[ordinal % stages]
        if action == "raw":
            slot.kind, slot.ordinal, slot.published = "dma", ordinal, set()
            producer += 1
        elif action == "dma_complete":
            slot.kind = "raw"
        elif action == "convert":
            lanes = set(range(rank * 32, (rank + 1) * 32))
            assert lanes.isdisjoint(slot.published)
            slot.published.update(lanes)
            converters[rank] += 1
            if fault == "early_recycle" and len(slot.published) == 32 * warps:
                slot.kind = "empty"
        elif action == "mma":
            assert slot.published == set(range(32 * warps))
            slot.kind = "mma"
            mma += 1
        else:
            assert action == "mma_complete" and slot.kind == "mma"
            slot.kind = "empty"
            completed += 1
        assert completed <= mma <= min(converters) <= max(converters) <= producer
        assert max(converters) - min(converters) <= stages
    assert producer == mma == total and converters == [total] * warps


@pytest.mark.parametrize("warps", (1, 2, 4, 8))
@pytest.mark.parametrize("stages", (1, 2, 3, 6))
def test_conversion_warps_may_skew_but_only_umma_completion_recycles(
    warps: int, stages: int
) -> None:
    for seed in range(4):
        _interleaved_ring(warps, stages, seed)


@pytest.mark.parametrize("warps", (2, 4, 8))
@pytest.mark.parametrize("fault", ("warp_count", "early_recycle"))
def test_arrival_count_and_empty_lifetime_faults_fail_the_async_model(
    warps: int, fault: str
) -> None:
    with pytest.raises(AssertionError):
        _interleaved_ring(warps, 1, 0, fault)
