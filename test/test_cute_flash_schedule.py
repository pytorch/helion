from __future__ import annotations

import dataclasses

import pytest

from helion._compiler.cute.flash_schedule import FlashEdge
from helion._compiler.cute.flash_schedule import FlashOutputOrder
from helion._compiler.cute.flash_schedule import FlashSchedule
from helion._compiler.cute.flash_schedule import FlashScheduleError
from helion._compiler.cute.flash_schedule import FlashScheduleSpec
from helion._compiler.cute.flash_schedule import FlashSyncScope
from helion._compiler.cute.flash_schedule import build_fa4_schedule
from helion._compiler.cute.flash_schedule import verify_flash_schedule


def _replace_edge(
    schedule: FlashSchedule,
    *,
    source: str,
    target: str,
    barrier: str | None = None,
    arrival_count: int | None = None,
    scope: FlashSyncScope | None = None,
    multicast: bool | None = None,
) -> FlashSchedule:
    matches = [
        index
        for index, edge in enumerate(schedule.edges)
        if edge.source == source
        and edge.target == target
        and (barrier is None or edge.barrier == barrier)
    ]
    assert len(matches) == 1
    edges = list(schedule.edges)
    edge = edges[matches[0]]
    edges[matches[0]] = dataclasses.replace(
        edge,
        arrival_count=edge.arrival_count if arrival_count is None else arrival_count,
        scope=edge.scope if scope is None else scope,
        multicast=edge.multicast if multicast is None else multicast,
    )
    return dataclasses.replace(schedule, edges=tuple(edges))


def _replace_region_alias(
    schedule: FlashSchedule,
    name: str,
    alias_group: str | None,
) -> FlashSchedule:
    matches = [
        index
        for index, region in enumerate(schedule.memory_regions)
        if region.name == name
    ]
    assert len(matches) == 1
    regions = list(schedule.memory_regions)
    regions[matches[0]] = dataclasses.replace(
        regions[matches[0]], alias_group=alias_group
    )
    return dataclasses.replace(schedule, memory_regions=tuple(regions))


@pytest.mark.parametrize(
    ("head_dim", "kv_depth", "shared_memory_bytes", "tmem_columns"),
    (
        (64, 2, 101376, 384),
        (128, 3, 232448, 512),
    ),
)
def test_canonical_fa4_resources(
    head_dim: int,
    kv_depth: int,
    shared_memory_bytes: int,
    tmem_columns: int,
) -> None:
    schedule = build_fa4_schedule(FlashScheduleSpec(head_dim, kv_depth))

    verified = verify_flash_schedule(schedule)

    assert verified.schedule is schedule
    assert schedule.shared_memory_bytes == shared_memory_bytes
    assert schedule.tmem_columns == tmem_columns


@pytest.mark.parametrize(
    ("kv_depth", "shared_memory_bytes"),
    (
        (2, 134144),
        (3, 166912),
        (4, 199680),
    ),
)
def test_separate_kv_depth_resources(
    kv_depth: int,
    shared_memory_bytes: int,
) -> None:
    schedule = build_fa4_schedule(FlashScheduleSpec(64, kv_depth, separate_kv=True))

    verify_flash_schedule(schedule)

    assert schedule.shared_memory_bytes == shared_memory_bytes


def test_separate_kv_capacity_failure() -> None:
    schedule = build_fa4_schedule(FlashScheduleSpec(64, 6, separate_kv=True))

    with pytest.raises(FlashScheduleError, match="shared-memory capacity"):
        verify_flash_schedule(schedule)


def test_two_cta_memory_is_rank_local_and_output_is_interleaved() -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(
            64,
            2,
            cta_count=2,
            multicast_kv=True,
            cooperative_mma=True,
        )
    )

    verify_flash_schedule(schedule)

    regions = {region.name: region for region in schedule.memory_regions}
    assert regions["Q_r0_q0"].offset == regions["Q_r1_q0"].offset
    assert regions["Q_r0_q0"].cta_rank == 0
    assert regions["Q_r1_q0"].cta_rank == 1
    assert {
        (owner.cta_rank, owner.query_slot): owner.output_tile
        for owner in schedule.output_owners
    } == {
        (0, 0): 0,
        (1, 0): 1,
        (0, 1): 2,
        (1, 1): 3,
    }
    barriers = {barrier.name: barrier for barrier in schedule.barriers}
    for rank in range(2):
        assert barriers[f"k_reuse_r{rank}"].expected_arrivals == 1
        assert barriers[f"v_reuse_r{rank}"].expected_arrivals == 1
        assert {
            edge.source for edge in schedule.edges if edge.barrier == f"k_reuse_r{rank}"
        } == {"qk_r0_q1"}
        assert {
            edge.source for edge in schedule.edges if edge.barrier == f"v_reuse_r{rank}"
        } == {"pv_r0_q1"}
    assert barriers["pfor_q0"].scope is FlashSyncScope.CLUSTER_LEADER


def test_independent_cga_output_is_cta_contiguous() -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(
            64,
            2,
            cta_count=2,
            multicast_kv=True,
            output_order=FlashOutputOrder.CTA_CONTIGUOUS,
        )
    )

    verify_flash_schedule(schedule)

    assert {
        (owner.cta_rank, owner.query_slot): owner.output_tile
        for owner in schedule.output_owners
    } == {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    barriers = {barrier.name: barrier for barrier in schedule.barriers}
    for rank in range(2):
        for slot in range(2):
            assert barriers[f"pfor_r{rank}_q{slot}"].expected_arrivals == 256
            assert barriers[f"pfor_r{rank}_q{slot}"].scope is FlashSyncScope.CTA
            assert barriers[f"pfor2_r{rank}_q{slot}"].expected_arrivals == 128
            assert barriers[f"pfor2_r{rank}_q{slot}"].scope is FlashSyncScope.CTA


def test_multicast_reuse_waits_for_both_ctas() -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(64, 2, cta_count=2, multicast_kv=True)
    )

    verify_flash_schedule(schedule)

    barriers = {barrier.name: barrier for barrier in schedule.barriers}
    for rank in range(2):
        for operand, consumer in (("k", "qk"), ("v", "pv")):
            barrier = f"{operand}_reuse_r{rank}"
            assert barriers[barrier].expected_arrivals == 2
            assert {
                edge.source for edge in schedule.edges if edge.barrier == barrier
            } == {f"{consumer}_r0_q1", f"{consumer}_r1_q1"}


def test_missing_cross_cta_reuse_release_is_rejected() -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(64, 2, cta_count=2, multicast_kv=True)
    )
    schedule = dataclasses.replace(
        schedule,
        edges=tuple(
            edge
            for edge in schedule.edges
            if not (edge.source == "qk_r1_q1" and edge.barrier == "k_reuse_r0")
        ),
    )

    with pytest.raises(
        FlashScheduleError,
        match="barrier k_reuse_r0 expected 2 arrivals, got 1",
    ):
        verify_flash_schedule(schedule)


def test_duplicate_same_cta_reuse_release_is_rejected() -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(64, 2, cta_count=2, multicast_kv=True)
    )
    schedule = dataclasses.replace(
        schedule,
        edges=tuple(
            dataclasses.replace(edge, source="qk_r0_q1")
            if edge.source == "qk_r1_q1" and edge.barrier == "k_reuse_r0"
            else edge
            for edge in schedule.edges
        ),
    )

    with pytest.raises(
        FlashScheduleError,
        match="multicast K/V reuse has the wrong consumer releases",
    ):
        verify_flash_schedule(schedule)


def test_local_reuse_barriers_remain_single_arrival() -> None:
    schedule = build_fa4_schedule(FlashScheduleSpec(64, 2, separate_kv=True))

    verify_flash_schedule(schedule)

    barriers = {barrier.name: barrier for barrier in schedule.barriers}
    assert barriers["k_reuse_r0"].expected_arrivals == 1
    assert barriers["v_reuse_r0"].expected_arrivals == 1


def test_aliased_kv_is_one_physical_ring_without_independent_depths() -> None:
    schedule = build_fa4_schedule(FlashScheduleSpec(64, 3))

    verify_flash_schedule(schedule)

    regions = {region.name: region for region in schedule.memory_regions}
    k_region = regions["K_r0"]
    v_region = regions["V_r0"]
    assert k_region.physical
    assert not v_region.physical
    assert k_region.offset == v_region.offset
    assert k_region.extent == v_region.extent == 3 * 128 * 64 * 2
    assert k_region.alias_group == v_region.alias_group == "shared_kv_ring_r0"
    assert k_region.reuse_distance is None
    assert v_region.reuse_distance is None


def test_aliased_kv_cannot_claim_independent_v_storage() -> None:
    schedule = build_fa4_schedule(FlashScheduleSpec(64, 2))
    regions = tuple(
        dataclasses.replace(region, physical=True) if region.name == "V_r0" else region
        for region in schedule.memory_regions
    )
    schedule = dataclasses.replace(schedule, memory_regions=regions)

    with pytest.raises(
        FlashScheduleError,
        match="aliased K/V must use one physical shared ring",
    ):
        verify_flash_schedule(schedule)


def test_barrier_arrival_count_corruption_is_rejected() -> None:
    schedule = build_fa4_schedule(FlashScheduleSpec(64, 2))
    schedule = _replace_edge(
        schedule,
        source="softmax_r0_q0",
        target="pv_r0_q0",
        barrier="pfor_q0",
        arrival_count=127,
    )

    with pytest.raises(FlashScheduleError, match="pfor_q0 expected 256"):
        verify_flash_schedule(schedule)


def test_unsplit_probability_publication_has_no_pfor2_dependency() -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(64, 2, separate_kv=True, split_p_arrive=False)
    )

    verify_flash_schedule(schedule)

    assert all(not barrier.name.startswith("pfor2") for barrier in schedule.barriers)
    assert all(
        edge.barrier is None or not edge.barrier.startswith("pfor2")
        for edge in schedule.edges
    )


def test_barrier_scope_corruption_is_rejected() -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(
            64,
            2,
            cta_count=2,
            multicast_kv=True,
            cooperative_mma=True,
        )
    )
    schedule = _replace_edge(
        schedule,
        source="correction_r0_q0",
        target="pv_r0_q0",
        scope=FlashSyncScope.CTA,
    )

    with pytest.raises(FlashScheduleError, match="barrier scope mismatch for pfor_q0"):
        verify_flash_schedule(schedule)


def test_missing_k_reuse_edge_is_rejected() -> None:
    schedule = build_fa4_schedule(FlashScheduleSpec(64, 2, separate_kv=True))
    schedule = dataclasses.replace(
        schedule,
        edges=tuple(edge for edge in schedule.edges if edge.barrier != "k_reuse_r0"),
        barriers=tuple(
            barrier for barrier in schedule.barriers if barrier.name != "k_reuse_r0"
        ),
    )

    with pytest.raises(FlashScheduleError, match="K_r0 is reused before all consumers"):
        verify_flash_schedule(schedule)


def test_score_probability_alias_corruption_is_rejected() -> None:
    schedule = build_fa4_schedule(FlashScheduleSpec(64, 2))
    schedule = _replace_region_alias(schedule, "P_r0_q0", "wrong_alias")

    with pytest.raises(
        FlashScheduleError, match="S_r0_q0 and P_r0_q0 overlap illegally"
    ):
        verify_flash_schedule(schedule)


def test_causal_multicast_does_not_invent_a_private_diagonal_path() -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(
            64,
            2,
            cta_count=2,
            causal=True,
            multicast_kv=True,
        )
    )

    verify_flash_schedule(schedule)

    assert not _region_names(schedule, prefix="causal_diagonal_")
    assert not {
        node.name for node in schedule.nodes if node.name.startswith("diagonal_")
    }


def test_explicit_causal_tail_requires_private_diagonal_path() -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(
            64,
            2,
            cta_count=2,
            causal=True,
            multicast_kv=True,
            private_causal_tail=True,
        )
    )
    region_name = "causal_diagonal_K_r0"
    writer = "diagonal_k_load_r0"
    schedule = dataclasses.replace(
        schedule,
        edges=tuple(
            edge
            for edge in schedule.edges
            if not (edge.resource == region_name and edge.source == writer)
        ),
        memory_regions=tuple(
            region for region in schedule.memory_regions if region.name != region_name
        ),
    )

    with pytest.raises(FlashScheduleError, match="private diagonal path"):
        verify_flash_schedule(schedule)


def test_causal_private_diagonal_cannot_be_multicast() -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(
            64,
            2,
            cta_count=2,
            causal=True,
            multicast_kv=True,
            private_causal_tail=True,
        )
    )
    schedule = _replace_edge(
        schedule,
        source="diagonal_k_load_r0",
        target="qk_r0_q0",
        multicast=True,
    )

    with pytest.raises(
        FlashScheduleError, match="CTA-private data cannot be multicast"
    ):
        verify_flash_schedule(schedule)


@pytest.mark.parametrize(
    "spec",
    (
        FlashScheduleSpec(64, 2, private_causal_tail=True),
        FlashScheduleSpec(
            64,
            2,
            cta_count=2,
            multicast_kv=True,
            private_causal_tail=True,
        ),
    ),
)
def test_private_causal_tail_requires_causal_multicast(
    spec: FlashScheduleSpec,
) -> None:
    with pytest.raises(
        FlashScheduleError,
        match="private causal tail requires causal K/V multicast",
    ):
        build_fa4_schedule(spec)


@pytest.mark.parametrize(
    ("kv_iterations", "middle_phase"),
    (
        (4, 0),
        (3, 1),
    ),
)
def test_persistent_phase_continuity_at_work_boundary(
    kv_iterations: int,
    middle_phase: int,
) -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(
            64,
            2,
            persistent=True,
            kv_iterations=kv_iterations,
        )
    )

    verify_flash_schedule(schedule)

    cycles = {cycle.barrier: cycle for cycle in schedule.phase_cycles}
    assert cycles["s_full_r0_q0"].uses_per_work == kv_iterations
    assert cycles["s_full_r0_q0"].phases == (0, middle_phase, 0)
    assert cycles["pfor_q0"].phases == (0, middle_phase, 0)


def test_persistent_phase_corruption_is_rejected() -> None:
    schedule = build_fa4_schedule(
        FlashScheduleSpec(64, 2, persistent=True, kv_iterations=3)
    )
    cycles = list(schedule.phase_cycles)
    index = next(
        index for index, cycle in enumerate(cycles) if cycle.barrier == "s_full_r0_q0"
    )
    cycles[index] = dataclasses.replace(cycles[index], phases=(0, 0, 0))
    schedule = dataclasses.replace(schedule, phase_cycles=tuple(cycles))

    with pytest.raises(FlashScheduleError, match="phase is discontinuous"):
        verify_flash_schedule(schedule)


@pytest.mark.parametrize("mode", ("duplicate", "missing"))
def test_output_ownership_corruption_is_rejected(mode: str) -> None:
    schedule = build_fa4_schedule(FlashScheduleSpec(64, 2))
    if mode == "duplicate":
        owners = (*schedule.output_owners, schedule.output_owners[0])
        error = "multiple owners"
    else:
        owners = schedule.output_owners[:-1]
        error = "ownership is incomplete"
    schedule = dataclasses.replace(schedule, output_owners=owners)

    with pytest.raises(FlashScheduleError, match=error):
        verify_flash_schedule(schedule)


def test_zero_delta_cycle_is_rejected() -> None:
    schedule = build_fa4_schedule(FlashScheduleSpec(64, 2))
    schedule = dataclasses.replace(
        schedule,
        edges=(
            *schedule.edges,
            FlashEdge("output_store_r0_q0", "q_load_r0_q0"),
        ),
    )

    with pytest.raises(
        FlashScheduleError, match="same-iteration schedule contains a cycle"
    ):
        verify_flash_schedule(schedule)


def test_direct_and_staged_output_resources() -> None:
    staged = build_fa4_schedule(FlashScheduleSpec(64, 2, stage_output=True))
    direct = build_fa4_schedule(FlashScheduleSpec(64, 2, stage_output=False))

    verify_flash_schedule(staged)
    verify_flash_schedule(direct)

    assert staged.shared_memory_bytes == 101376
    assert direct.shared_memory_bytes == 68608
    assert staged.shared_memory_bytes - direct.shared_memory_bytes == 32768
    assert _region_names(staged, prefix="O_stage_") == {
        "O_stage_r0_q0",
        "O_stage_r0_q1",
    }
    assert not _region_names(direct, prefix="O_stage_")


def _region_names(schedule: FlashSchedule, *, prefix: str) -> set[str]:
    return {
        region.name
        for region in schedule.memory_regions
        if region.name.startswith(prefix)
    }
