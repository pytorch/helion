"""Independent byte ownership for a dormant native shared-storage proposal."""

from __future__ import annotations

import dataclasses
import json
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest

from test.test_cute_helper_cache_dependencies import _kernel
from test.test_cute_helper_cache_dependencies import _key
from test.test_cute_operand_pipeline import _actual_functions

from helion._compiler.cute.tcgen05_storage import NativeSharedAllocation
from helion._compiler.cute.tcgen05_storage import NativeSharedPlacement
from helion._compiler.cute.tcgen05_storage import NativeSharedStoragePlan
from helion.runtime.cute import source_dependencies

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("seed", range(12))
def test_packed_storage_keeps_every_simultaneously_live_byte(seed: int) -> None:
    rng = random.Random(seed)
    allocations = tuple(
        NativeSharedAllocation(
            f"object_{i}", rng.randint(1, 3072), 1 << rng.randrange(0, 11)
        )
        for i in range(24)
    )
    plan = NativeSharedStoragePlan.pack(allocations)
    memory = bytearray([231] * plan.bytes)
    owner = [-1] * plan.bytes
    for i, placement in enumerate(plan.placements):
        assert placement.offset % placement.allocation.alignment == 0
        for position in range(placement.offset, placement.end):
            assert owner[position] == -1
            owner[position] = i
            memory[position] = i
    for i, placement in enumerate(plan.placements):
        assert (
            memory[placement.offset : placement.end]
            == bytes([i]) * placement.allocation.bytes
        )
    assert all(memory[i] == 231 for i, item in enumerate(owner) if item == -1)
    assert sum(item >= 0 for item in owner) == sum(a.bytes for a in allocations)
    assert (
        NativeSharedStoragePlan.pack(allocations).cache_identity == plan.cache_identity
    )
    encoded = json.dumps(plan.cache_identity)
    assert json.loads(encoded) == [list(item) for item in plan.cache_identity]


@pytest.mark.parametrize("bad", (0, -1, True, 1.5))
def test_invalid_allocation_size_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        NativeSharedAllocation("bad", cast("int", bad), 16)


@pytest.mark.parametrize("bad", (0, -1, True, 3, 24, 1 << 31))
def test_invalid_alignment_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        NativeSharedAllocation("bad", 16, cast("int", bad))


def test_direct_plan_construction_cannot_forge_disjointness_or_alignment() -> None:
    allocation = NativeSharedAllocation("a", 128, 128)
    other = NativeSharedAllocation("b", 16, 8)
    for placements in (
        (),
        (NativeSharedPlacement(allocation, 1),),
        (NativeSharedPlacement(allocation, -128),),
        (NativeSharedPlacement(allocation, 0), NativeSharedPlacement(other, 120)),
        (NativeSharedPlacement(allocation, 0), NativeSharedPlacement(allocation, 128)),
        (NativeSharedPlacement(other, 0), NativeSharedPlacement(allocation, 128)),
        (NativeSharedPlacement(allocation, 1 << 31),),
    ):
        with pytest.raises(ValueError):
            NativeSharedStoragePlan(placements)


def test_complete_storage_cannot_borrow_reserved_capacity_or_prune_a_ring() -> None:
    allocations = (
        NativeSharedAllocation("output", 2 * 128 * 32 * 4, 1024),
        NativeSharedAllocation("raw_a", 6 * 128 * 32 * 4, 128),
        NativeSharedAllocation("raw_b", 6 * 128 * 32 * 4, 128),
        NativeSharedAllocation("metadata", 256 * 4, 16),
        NativeSharedAllocation("other", 636, 4),
    )
    plan = NativeSharedStoragePlan.pack(allocations)
    assert plan.fits(capacity=232448, reserved=1024)
    enlarged = NativeSharedStoragePlan.pack(
        tuple(
            dataclasses.replace(item, bytes=item.bytes + 1024)
            if item.name == "metadata"
            else item
            for item in allocations
        )
    )
    assert not enlarged.fits(capacity=232448, reserved=1024)
    assert (
        next(
            p for p in enlarged.placements if p.allocation.name == "raw_a"
        ).allocation.bytes
        == 98304
    )
    assert not plan.fits(capacity=232448, reserved=-1)
    with pytest.raises(ValueError):
        NativeSharedStoragePlan.pack((NativeSharedAllocation("a", 1 << 31, 8),))


@pytest.mark.parametrize("stages", (3, 6))
@pytest.mark.parametrize("warps", (1, 2, 4, 8))
@pytest.mark.parametrize(
    "storage_kind", ("aligned", "wrong_dtype", "global", "misaligned")
)
def test_supplied_converted_full_storage_keeps_raw_empty_and_lane_arrivals(
    stages: int, warps: int, storage_kind: str
) -> None:
    one, empty, shared = object(), object(), object()
    supplied = SimpleNamespace(dtype="i64", memspace=shared, alignment=8)
    if storage_kind == "wrong_dtype":
        supplied.dtype = "i32"
    elif storage_kind == "global":
        supplied.memspace = object()
    elif storage_kind == "misaligned":
        supplied.alignment = 4
    events = []

    def forbidden_allocation(*args: object, **kwargs: object) -> None:
        raise AssertionError("caller-supplied storage must not allocate another ring")

    def full(storage: object, count: int, agent: object, **kwargs: object) -> object:
        assert storage is supplied and count == stages
        assert agent == ("async", ("thread", 32 * warps))
        return storage

    fake_cute = SimpleNamespace(
        AddressSpace=SimpleNamespace(smem=shared),
        arch=SimpleNamespace(
            alloc_smem=forbidden_allocation,
            mbarrier_init_fence=lambda: events.append("fence"),
            sync_threads=lambda: events.append("sync"),
        ),
        nvgpu=SimpleNamespace(
            tcgen05=SimpleNamespace(CtaGroup=SimpleNamespace(ONE=one))
        ),
    )
    fake_pipeline = SimpleNamespace(
        PipelineAsync=SimpleNamespace(_make_sync_object=full),
        PipelineOp=SimpleNamespace(AsyncThread="async"),
        CooperativeGroup=lambda agent, count: (agent, count),
        Agent=SimpleNamespace(Thread="thread"),
        PipelineAsyncUmma=lambda *args: args,
    )
    function = _actual_functions(
        fake_cute, SimpleNamespace(const_expr=bool, Int64="i64"), fake_pipeline
    )["make_converted_operand_pipeline"]
    raw = SimpleNamespace(
        num_stages=stages,
        cta_group=one,
        producer_mask=None,
        consumer_mask=None,
        sync_object_empty=empty,
    )
    if storage_kind == "aligned":
        result = function(raw, stages, warps, True, supplied)
        assert result == (supplied, empty, stages, None, None, one)
    else:
        with pytest.raises(AssertionError):
            function(raw, stages, warps, True, supplied)
    assert events == []


@pytest.mark.parametrize(
    "edited",
    (
        "tcgen05_grouped_prefix",
        "tcgen05_operand_pipeline",
        "tcgen05_operand_transform",
        "tcgen05_storage",
    ),
)
def test_native_helper_dependency_edit_changes_only_native_disk_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edited: str
) -> None:
    paths = set(source_dependencies._COMMON_DEPENDENCIES)
    for dependencies in source_dependencies._WRAPPER_DEPENDENCIES.values():
        paths.update(dependencies)
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n")
    monkeypatch.setattr(source_dependencies, "_PACKAGE_ROOT", tmp_path)
    native, scaled, ordinary = (
        _kernel("tcgen05_grouped_rna"),
        _kernel("block_scaled_mma"),
        _kernel("tcgen05_ab_tma"),
    )
    before = [_key(kernel) for kernel in (native, scaled, ordinary)]
    assert all(key is not None for key in before)
    (tmp_path / f"_compiler/cute/{edited}.py").write_text("# changed helper\n")
    after = [_key(kernel) for kernel in (native, scaled, ordinary)]
    assert before[0] != after[0]
    assert before[1:] == after[1:]
