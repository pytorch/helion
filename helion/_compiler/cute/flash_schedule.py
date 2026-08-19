from __future__ import annotations

from collections import defaultdict
import dataclasses
import enum


class FlashScheduleError(ValueError):
    """A structural flash schedule is unsafe or exceeds device resources."""


class FlashNodeKind(enum.Enum):
    Q_LOAD = "q_load"
    K_LOAD = "k_load"
    V_LOAD = "v_load"
    DIAGONAL_LOAD = "diagonal_load"
    QK_MMA = "qk_mma"
    SOFTMAX = "softmax"
    STAT_PUBLISH = "stat_publish"
    FINAL_STAT_PUBLISH = "final_stat_publish"
    FINAL_STAT_CONSUME = "final_stat_consume"
    CORRECTION = "correction"
    PV_MMA = "pv_mma"
    OUTPUT_STAGE = "output_stage"
    OUTPUT_STORE = "output_store"


class FlashMemorySpace(enum.Enum):
    SMEM = "smem"
    TMEM = "tmem"


class FlashVisibility(enum.Enum):
    CTA_PRIVATE = "cta_private"
    CLUSTER_VISIBLE = "cluster_visible"


class FlashSyncScope(enum.Enum):
    CTA = "cta"
    CLUSTER = "cluster"
    CLUSTER_LEADER = "cluster_leader"


class FlashOutputOrder(enum.Enum):
    INTERLEAVED = "interleaved"
    CTA_CONTIGUOUS = "cta_contiguous"


class FlashStatReleaseMapping(enum.Enum):
    CROSS_SLOT = "cross_slot"
    SAME_SLOT = "same_slot"


@dataclasses.dataclass(frozen=True)
class FlashScheduleSpec:
    head_dim: int
    kv_depth: int
    cta_count: int = 1
    query_slots_per_cta: int = 2
    separate_kv: bool = False
    causal: bool = False
    multicast_kv: bool = False
    persistent: bool = False
    kv_iterations: int | None = None
    stage_output: bool = True
    dtype_bytes: int = 2
    private_causal_tail: bool = False
    output_order: FlashOutputOrder = FlashOutputOrder.INTERLEAVED
    cooperative_mma: bool = False
    split_p_arrive: bool = True
    stat_depth: int = 2
    pipelined_stat_handoff: bool = False
    final_only_stat_handoff: bool = False
    stat_release_mapping: FlashStatReleaseMapping = FlashStatReleaseMapping.CROSS_SLOT
    query_slots_have_equal_kv_iterations: bool = False


@dataclasses.dataclass(frozen=True)
class FlashNode:
    name: str
    kind: FlashNodeKind
    cta_rank: int | None = None
    query_slot: int | None = None


@dataclasses.dataclass(frozen=True)
class FlashEdge:
    source: str
    target: str
    iteration_delta: int = 0
    barrier: str | None = None
    arrival_count: int = 0
    scope: FlashSyncScope = FlashSyncScope.CTA
    resource_scope: FlashSyncScope = FlashSyncScope.CTA
    resource: str | None = None
    multicast: bool = False


@dataclasses.dataclass(frozen=True)
class FlashBarrier:
    name: str
    expected_arrivals: int
    scope: FlashSyncScope


@dataclasses.dataclass(frozen=True)
class FlashMemoryRegion:
    name: str
    space: FlashMemorySpace
    offset: int
    extent: int
    alignment: int
    visibility: FlashVisibility
    writer: str
    consumers: tuple[str, ...]
    reuse_distance: int | None = None
    reuse_barrier: str | None = None
    alias_group: str | None = None
    cta_rank: int | None = None
    physical: bool = True


@dataclasses.dataclass(frozen=True)
class FlashOutputOwner:
    output_tile: int
    cta_rank: int
    query_slot: int


@dataclasses.dataclass(frozen=True)
class FlashPhaseCycle:
    barrier: str
    phases: tuple[int, ...]
    uses_per_work: int = 1


@dataclasses.dataclass(frozen=True)
class FlashSchedule:
    spec: FlashScheduleSpec
    nodes: tuple[FlashNode, ...]
    edges: tuple[FlashEdge, ...]
    barriers: tuple[FlashBarrier, ...]
    memory_regions: tuple[FlashMemoryRegion, ...]
    output_owners: tuple[FlashOutputOwner, ...]
    phase_cycles: tuple[FlashPhaseCycle, ...]
    shared_memory_bytes: int
    tmem_columns: int


@dataclasses.dataclass(frozen=True)
class FlashScheduleLimits:
    shared_memory_bytes: int = 232448
    tmem_columns: int = 512


@dataclasses.dataclass(frozen=True)
class VerifiedFlashSchedule:
    schedule: FlashSchedule

    @property
    def spec(self) -> FlashScheduleSpec:
        return self.schedule.spec


def _node_name(kind: str, cta_rank: int, query_slot: int) -> str:
    return f"{kind}_r{cta_rank}_q{query_slot}"


def _kv_node_name(kind: str, cta_rank: int, multicast: bool) -> str:
    if multicast:
        return f"{kind}_multicast"
    return f"{kind}_r{cta_rank}"


def _shared_memory_bytes(spec: FlashScheduleSpec) -> int:
    tile_bytes = 128 * spec.head_dim * spec.dtype_bytes
    q_bytes = spec.query_slots_per_cta * tile_bytes
    kv_rings = 2 if spec.separate_kv else 1
    kv_bytes = kv_rings * spec.kv_depth * tile_bytes
    output_bytes = spec.query_slots_per_cta * tile_bytes if spec.stage_output else 0
    # Scale/stat transport plus aligned barrier storage in the current FA4 layout.
    return q_bytes + kv_bytes + output_bytes + 3072


def _output_tile(spec: FlashScheduleSpec, cta_rank: int, query_slot: int) -> int:
    if spec.output_order is FlashOutputOrder.CTA_CONTIGUOUS:
        return cta_rank * spec.query_slots_per_cta + query_slot
    return query_slot * spec.cta_count + cta_rank


def _pipelined_stat_release(
    spec: FlashScheduleSpec, source_slot: int
) -> tuple[int, int]:
    if spec.stat_release_mapping is FlashStatReleaseMapping.SAME_SLOT:
        return source_slot, 1
    # The acknowledged correction loop visits slot 0 before slot 1. Slot 0
    # releases slot 1's held prologue value in the current loop iteration;
    # slot 1 then releases slot 0 for the next iteration.
    return source_slot ^ 1, source_slot


def _pipelined_stat_releaser(
    spec: FlashScheduleSpec, target_slot: int
) -> tuple[int, int]:
    source_slot = (
        target_slot
        if spec.stat_release_mapping is FlashStatReleaseMapping.SAME_SLOT
        else target_slot ^ 1
    )
    mapped_target, iteration_delta = _pipelined_stat_release(spec, source_slot)
    assert mapped_target == target_slot
    return source_slot, iteration_delta


def build_fa4_schedule(spec: FlashScheduleSpec) -> FlashSchedule:
    """Build the structural graph for an FA4-style local schedule.

    Score storage is always two local query slots. With two CTAs this becomes
    four aggregate CTA-local slices; it is not a temporal four-stage score ring.
    """
    if spec.head_dim not in (64, 128):
        raise FlashScheduleError("FA4 schedule requires head_dim 64 or 128")
    if spec.kv_depth < 2:
        raise FlashScheduleError("FA4 schedule requires at least two K/V stages")
    if spec.cta_count not in (1, 2):
        raise FlashScheduleError("FA4 schedule supports one or two CTAs")
    if spec.query_slots_per_cta != 2:
        raise FlashScheduleError("FA4 schedule requires two local score slots")
    if spec.multicast_kv and spec.cta_count != 2:
        raise FlashScheduleError("K/V multicast requires a two-CTA schedule")
    if spec.cooperative_mma and spec.cta_count != 2:
        raise FlashScheduleError("a cooperative MMA requires two CTAs")
    if spec.cooperative_mma and spec.output_order is not FlashOutputOrder.INTERLEAVED:
        raise FlashScheduleError("a cooperative MMA requires interleaved output")
    if spec.private_causal_tail and not (spec.causal and spec.multicast_kv):
        raise FlashScheduleError("a private causal tail requires causal K/V multicast")
    if not isinstance(spec.output_order, FlashOutputOrder):
        raise FlashScheduleError("output order is invalid")
    if not isinstance(spec.stat_release_mapping, FlashStatReleaseMapping):
        raise FlashScheduleError("stat release mapping is invalid")
    if spec.dtype_bytes <= 0:
        raise FlashScheduleError("dtype size must be positive")
    if spec.stat_depth not in (1, 2):
        raise FlashScheduleError("stat transport depth must be one or two")
    if spec.pipelined_stat_handoff and spec.stat_depth != 1:
        raise FlashScheduleError("pipelined stat handoff requires depth one")
    if (
        spec.pipelined_stat_handoff
        and spec.causal
        and spec.stat_release_mapping is FlashStatReleaseMapping.CROSS_SLOT
        and not spec.query_slots_have_equal_kv_iterations
    ):
        raise FlashScheduleError(
            "causal cross-slot stat releases require proof of equal query-slot "
            "K/V iteration counts"
        )
    if (
        not spec.pipelined_stat_handoff
        and spec.stat_release_mapping is FlashStatReleaseMapping.SAME_SLOT
    ):
        raise FlashScheduleError(
            "same-slot stat releases require pipelined stat handoff"
        )
    if spec.final_only_stat_handoff and (spec.stat_depth != 1 or spec.causal):
        raise FlashScheduleError(
            "final-only stat handoff requires depth one and a noncausal schedule"
        )
    if spec.pipelined_stat_handoff and spec.final_only_stat_handoff:
        raise FlashScheduleError("stat handoff modes are mutually exclusive")

    nodes: list[FlashNode] = []
    edges: list[FlashEdge] = []
    barriers: list[FlashBarrier] = []
    regions: list[FlashMemoryRegion] = []
    owners: list[FlashOutputOwner] = []

    kv_visibility = (
        FlashVisibility.CLUSTER_VISIBLE
        if spec.multicast_kv
        else FlashVisibility.CTA_PRIVATE
    )
    kv_scope = FlashSyncScope.CLUSTER if spec.multicast_kv else FlashSyncScope.CTA
    kv_reuse_arrivals = (
        1 if spec.cooperative_mma else spec.cta_count if spec.multicast_kv else 1
    )
    tile_bytes = 128 * spec.head_dim * spec.dtype_bytes
    q_bytes = spec.query_slots_per_cta * tile_bytes
    kv_ring_bytes = spec.kv_depth * tile_bytes
    k_offset = q_bytes
    v_offset = k_offset + (kv_ring_bytes if spec.separate_kv else 0)
    output_offset = q_bytes + (2 if spec.separate_kv else 1) * kv_ring_bytes
    overhead_offset = output_offset + (
        spec.query_slots_per_cta * tile_bytes if spec.stage_output else 0
    )
    stat_bytes_per_query = spec.stat_depth * 128 * 4

    load_nodes: set[str] = set()
    for rank in range(spec.cta_count):
        k_load = _kv_node_name("k_load", rank, spec.multicast_kv)
        v_load = _kv_node_name("v_load", rank, spec.multicast_kv)
        for name, kind in (
            (k_load, FlashNodeKind.K_LOAD),
            (v_load, FlashNodeKind.V_LOAD),
        ):
            if name not in load_nodes:
                nodes.append(FlashNode(name, kind, None if spec.multicast_kv else rank))
                load_nodes.add(name)

        barriers.extend(
            [
                FlashBarrier(f"k_ready_r{rank}", 1, kv_scope),
                FlashBarrier(f"v_ready_r{rank}", 1, kv_scope),
                FlashBarrier(
                    f"k_reuse_r{rank}",
                    kv_reuse_arrivals,
                    kv_scope,
                ),
                FlashBarrier(
                    f"v_reuse_r{rank}",
                    kv_reuse_arrivals,
                    kv_scope,
                ),
            ]
        )
        # Legacy FA4 alternates K and V acquisitions through one physical
        # pipeline ring. K names that backing allocation and V is a logical
        # view; only the separate-ring family has per-operand depth proofs.
        regions.extend(
            [
                FlashMemoryRegion(
                    f"K_r{rank}",
                    FlashMemorySpace.SMEM,
                    k_offset,
                    kv_ring_bytes,
                    1024,
                    kv_visibility,
                    k_load,
                    tuple(
                        _node_name("qk", rank, slot)
                        for slot in range(spec.query_slots_per_cta)
                    ),
                    reuse_distance=spec.kv_depth if spec.separate_kv else None,
                    alias_group=(
                        None if spec.separate_kv else f"shared_kv_ring_r{rank}"
                    ),
                    cta_rank=rank,
                ),
                FlashMemoryRegion(
                    f"V_r{rank}",
                    FlashMemorySpace.SMEM,
                    v_offset,
                    kv_ring_bytes,
                    1024,
                    kv_visibility,
                    v_load,
                    tuple(
                        _node_name("pv", rank, slot)
                        for slot in range(spec.query_slots_per_cta)
                    ),
                    reuse_distance=spec.kv_depth if spec.separate_kv else None,
                    alias_group=(
                        None if spec.separate_kv else f"shared_kv_ring_r{rank}"
                    ),
                    cta_rank=rank,
                    physical=spec.separate_kv,
                ),
            ]
        )

        if spec.private_causal_tail:
            diagonal_k = f"diagonal_k_load_r{rank}"
            diagonal_v = f"diagonal_v_load_r{rank}"
            nodes.extend(
                [
                    FlashNode(diagonal_k, FlashNodeKind.DIAGONAL_LOAD, rank),
                    FlashNode(diagonal_v, FlashNodeKind.DIAGONAL_LOAD, rank),
                ]
            )
            regions.extend(
                [
                    FlashMemoryRegion(
                        f"causal_diagonal_K_r{rank}",
                        FlashMemorySpace.SMEM,
                        0,
                        tile_bytes,
                        1024,
                        FlashVisibility.CTA_PRIVATE,
                        diagonal_k,
                        tuple(
                            _node_name("qk", rank, slot)
                            for slot in range(spec.query_slots_per_cta)
                        ),
                        cta_rank=rank,
                        physical=False,
                    ),
                    FlashMemoryRegion(
                        f"causal_diagonal_V_r{rank}",
                        FlashMemorySpace.SMEM,
                        0,
                        tile_bytes,
                        1024,
                        FlashVisibility.CTA_PRIVATE,
                        diagonal_v,
                        tuple(
                            _node_name("pv", rank, slot)
                            for slot in range(spec.query_slots_per_cta)
                        ),
                        cta_rank=rank,
                        physical=False,
                    ),
                ]
            )

    for rank in range(spec.cta_count):
        k_load = _kv_node_name("k_load", rank, spec.multicast_kv)
        v_load = _kv_node_name("v_load", rank, spec.multicast_kv)
        for slot in range(spec.query_slots_per_cta):
            q_load = _node_name("q_load", rank, slot)
            qk = _node_name("qk", rank, slot)
            softmax = _node_name("softmax", rank, slot)
            stat = _node_name("stat", rank, slot)
            final_stat_publish = _node_name("final_stat_publish", rank, slot)
            final_stat_consume = _node_name("final_stat_consume", rank, slot)
            correction = _node_name("correction", rank, slot)
            pv = _node_name("pv", rank, slot)
            output_stage = _node_name("output_stage", rank, slot)
            output_store = _node_name("output_store", rank, slot)
            nodes.extend(
                [
                    FlashNode(q_load, FlashNodeKind.Q_LOAD, rank, slot),
                    FlashNode(qk, FlashNodeKind.QK_MMA, rank, slot),
                    FlashNode(softmax, FlashNodeKind.SOFTMAX, rank, slot),
                    FlashNode(stat, FlashNodeKind.STAT_PUBLISH, rank, slot),
                    FlashNode(correction, FlashNodeKind.CORRECTION, rank, slot),
                    FlashNode(pv, FlashNodeKind.PV_MMA, rank, slot),
                    FlashNode(output_stage, FlashNodeKind.OUTPUT_STAGE, rank, slot),
                    FlashNode(output_store, FlashNodeKind.OUTPUT_STORE, rank, slot),
                ]
            )
            if spec.final_only_stat_handoff:
                nodes.extend(
                    [
                        FlashNode(
                            final_stat_publish,
                            FlashNodeKind.FINAL_STAT_PUBLISH,
                            rank,
                            slot,
                        ),
                        FlashNode(
                            final_stat_consume,
                            FlashNodeKind.FINAL_STAT_CONSUME,
                            rank,
                            slot,
                        ),
                    ]
                )

            q_ready = f"q_ready_r{rank}_q{slot}"
            s_full = f"s_full_r{rank}_q{slot}"
            stat_ready = f"stat_ready_r{rank}_q{slot}"
            stat_empty = f"stat_empty_r{rank}_q{slot}"
            o_full = f"o_full_r{rank}_q{slot}"
            barriers.extend(
                [
                    FlashBarrier(q_ready, 1, FlashSyncScope.CTA),
                    FlashBarrier(s_full, 1, FlashSyncScope.CTA),
                    FlashBarrier(stat_ready, 1, FlashSyncScope.CTA),
                    FlashBarrier(stat_empty, 1, FlashSyncScope.CTA),
                    FlashBarrier(o_full, 1, FlashSyncScope.CTA),
                ]
            )
            edges.extend(
                [
                    FlashEdge(
                        q_load,
                        qk,
                        barrier=q_ready,
                        arrival_count=1,
                        resource=f"Q_r{rank}_q{slot}",
                    ),
                    FlashEdge(
                        k_load,
                        qk,
                        barrier=f"k_ready_r{rank}" if slot == 0 else None,
                        arrival_count=1 if slot == 0 else 0,
                        scope=kv_scope,
                        resource_scope=kv_scope,
                        resource=f"K_r{rank}",
                        multicast=spec.multicast_kv,
                    ),
                    FlashEdge(
                        qk,
                        softmax,
                        barrier=s_full,
                        arrival_count=1,
                        resource=f"S_r{rank}_q{slot}",
                    ),
                    FlashEdge(softmax, stat),
                    FlashEdge(
                        stat,
                        correction,
                        barrier=stat_ready,
                        arrival_count=1,
                        resource=f"STAT_r{rank}_q{slot}",
                    ),
                    FlashEdge(
                        v_load,
                        pv,
                        barrier=f"v_ready_r{rank}" if slot == 0 else None,
                        arrival_count=1 if slot == 0 else 0,
                        scope=kv_scope,
                        resource_scope=kv_scope,
                        resource=f"V_r{rank}",
                        multicast=spec.multicast_kv,
                    ),
                    FlashEdge(
                        pv,
                        output_stage,
                        barrier=o_full,
                        arrival_count=1,
                        resource=f"O_r{rank}_q{slot}",
                    ),
                    FlashEdge(
                        output_stage,
                        output_store,
                        resource=(
                            f"O_stage_r{rank}_q{slot}" if spec.stage_output else None
                        ),
                    ),
                    FlashEdge(pv, qk, iteration_delta=1),
                    FlashEdge(
                        pv,
                        correction,
                        iteration_delta=1,
                        resource=f"O_r{rank}_q{slot}",
                    ),
                ]
            )
            if not spec.final_only_stat_handoff:
                stat_release_slot = slot
                stat_release_delta = spec.stat_depth
                if spec.pipelined_stat_handoff:
                    stat_release_slot, stat_release_delta = _pipelined_stat_release(
                        spec, slot
                    )
                edges.append(
                    FlashEdge(
                        correction,
                        _node_name("stat", rank, stat_release_slot),
                        iteration_delta=stat_release_delta,
                        barrier=f"stat_empty_r{rank}_q{stat_release_slot}",
                        arrival_count=1,
                    )
                )
            else:
                edges.extend(
                    [
                        FlashEdge(
                            correction,
                            final_stat_publish,
                            barrier=stat_empty,
                            arrival_count=1,
                        ),
                        FlashEdge(
                            final_stat_publish,
                            final_stat_consume,
                            resource=f"FINAL_STAT_r{rank}_q{slot}",
                        ),
                    ]
                )
                if spec.persistent:
                    edges.append(
                        FlashEdge(
                            final_stat_consume,
                            stat,
                            iteration_delta=1,
                        )
                    )
            if slot + 1 == spec.query_slots_per_cta:
                if spec.cooperative_mma and rank != 0:
                    reuse_ranks: tuple[int, ...] | range = ()
                elif spec.multicast_kv:
                    reuse_ranks = range(spec.cta_count)
                else:
                    reuse_ranks = (rank,)
                for reuse_rank in reuse_ranks:
                    edges.extend(
                        [
                            FlashEdge(
                                qk,
                                k_load,
                                iteration_delta=spec.kv_depth,
                                barrier=f"k_reuse_r{reuse_rank}",
                                arrival_count=1,
                                scope=kv_scope,
                            ),
                            FlashEdge(
                                pv,
                                v_load,
                                iteration_delta=spec.kv_depth,
                                barrier=f"v_reuse_r{reuse_rank}",
                                arrival_count=1,
                                scope=kv_scope,
                            ),
                        ]
                    )
            else:
                edges.extend(
                    [
                        FlashEdge(
                            qk,
                            _node_name("qk", rank, slot + 1),
                        ),
                        FlashEdge(
                            pv,
                            _node_name("pv", rank, slot + 1),
                        ),
                    ]
                )
            if spec.private_causal_tail:
                edges.extend(
                    [
                        FlashEdge(
                            f"diagonal_k_load_r{rank}",
                            qk,
                            resource=f"causal_diagonal_K_r{rank}",
                        ),
                        FlashEdge(
                            f"diagonal_v_load_r{rank}",
                            pv,
                            resource=f"causal_diagonal_V_r{rank}",
                        ),
                    ]
                )

            owners.append(FlashOutputOwner(_output_tile(spec, rank, slot), rank, slot))
            regions.append(
                FlashMemoryRegion(
                    f"Q_r{rank}_q{slot}",
                    FlashMemorySpace.SMEM,
                    slot * tile_bytes,
                    tile_bytes,
                    1024,
                    FlashVisibility.CTA_PRIVATE,
                    q_load,
                    (qk,),
                    cta_rank=rank,
                )
            )
            score_offset = slot * 128
            regions.extend(
                [
                    FlashMemoryRegion(
                        f"S_r{rank}_q{slot}",
                        FlashMemorySpace.TMEM,
                        score_offset,
                        128,
                        1,
                        FlashVisibility.CTA_PRIVATE,
                        qk,
                        (softmax,),
                        alias_group=f"score_p_r{rank}_q{slot}",
                        cta_rank=rank,
                    ),
                    FlashMemoryRegion(
                        f"P_r{rank}_q{slot}",
                        FlashMemorySpace.TMEM,
                        score_offset,
                        128,
                        1,
                        FlashVisibility.CTA_PRIVATE,
                        softmax,
                        (pv,),
                        reuse_distance=1,
                        alias_group=f"score_p_r{rank}_q{slot}",
                        cta_rank=rank,
                    ),
                    FlashMemoryRegion(
                        f"O_r{rank}_q{slot}",
                        FlashMemorySpace.TMEM,
                        256 + slot * spec.head_dim,
                        spec.head_dim,
                        1,
                        FlashVisibility.CTA_PRIVATE,
                        pv,
                        (correction, output_stage),
                        cta_rank=rank,
                    ),
                ]
            )
            regions.append(
                FlashMemoryRegion(
                    f"STAT_r{rank}_q{slot}",
                    FlashMemorySpace.SMEM,
                    overhead_offset + slot * stat_bytes_per_query,
                    stat_bytes_per_query,
                    16,
                    FlashVisibility.CTA_PRIVATE,
                    stat,
                    (correction,),
                    reuse_distance=(
                        1 if spec.final_only_stat_handoff else spec.stat_depth
                    ),
                    reuse_barrier=(
                        None if spec.final_only_stat_handoff else stat_empty
                    ),
                    alias_group=(
                        f"stat_terminal_r{rank}_q{slot}"
                        if spec.final_only_stat_handoff
                        else None
                    ),
                    cta_rank=rank,
                )
            )
            if spec.final_only_stat_handoff:
                regions.append(
                    FlashMemoryRegion(
                        f"FINAL_STAT_r{rank}_q{slot}",
                        FlashMemorySpace.SMEM,
                        overhead_offset + slot * stat_bytes_per_query,
                        stat_bytes_per_query,
                        16,
                        FlashVisibility.CTA_PRIVATE,
                        final_stat_publish,
                        (final_stat_consume,),
                        reuse_distance=1 if spec.persistent else None,
                        alias_group=f"stat_terminal_r{rank}_q{slot}",
                        cta_rank=rank,
                    )
                )
            if spec.stage_output:
                regions.append(
                    FlashMemoryRegion(
                        f"O_stage_r{rank}_q{slot}",
                        FlashMemorySpace.SMEM,
                        output_offset + slot * tile_bytes,
                        tile_bytes,
                        1024,
                        FlashVisibility.CTA_PRIVATE,
                        output_stage,
                        (output_store,),
                        cta_rank=rank,
                    )
                )

    for slot in range(spec.query_slots_per_cta):
        for rank in range(spec.cta_count):
            if spec.cooperative_mma:
                pfor = f"pfor_q{slot}"
                pfor2 = f"pfor2_q{slot}"
                pfor_scope = FlashSyncScope.CLUSTER_LEADER
                if rank == 0:
                    barriers.append(
                        FlashBarrier(pfor, 256 * spec.cta_count, pfor_scope)
                    )
                    if spec.split_p_arrive:
                        barriers.append(
                            FlashBarrier(pfor2, 128 * spec.cta_count, pfor_scope)
                        )
            else:
                rank_suffix = f"_r{rank}" if spec.cta_count > 1 else ""
                pfor = f"pfor{rank_suffix}_q{slot}"
                pfor2 = f"pfor2{rank_suffix}_q{slot}"
                pfor_scope = FlashSyncScope.CTA
                barriers.append(FlashBarrier(pfor, 256, pfor_scope))
                if spec.split_p_arrive:
                    barriers.append(FlashBarrier(pfor2, 128, pfor_scope))
            softmax = _node_name("softmax", rank, slot)
            correction = _node_name("correction", rank, slot)
            pv = _node_name("pv", rank, slot)
            edges.extend(
                [
                    FlashEdge(
                        softmax,
                        pv,
                        barrier=pfor,
                        arrival_count=128,
                        scope=pfor_scope,
                        resource=f"P_r{rank}_q{slot}",
                    ),
                    FlashEdge(
                        correction,
                        pv,
                        barrier=pfor,
                        arrival_count=128,
                        scope=pfor_scope,
                    ),
                ]
            )
            if spec.split_p_arrive:
                edges.append(
                    FlashEdge(
                        softmax,
                        pv,
                        barrier=pfor2,
                        arrival_count=128,
                        scope=pfor_scope,
                    )
                )

    phase_cycles: tuple[FlashPhaseCycle, ...] = ()
    if spec.persistent:
        if spec.kv_iterations is None or spec.kv_iterations <= 0:
            raise FlashScheduleError(
                "persistent schedules require a positive K/V iteration count"
            )
        cycles = []
        for barrier in barriers:
            stat_barrier = barrier.name.startswith(("stat_ready", "stat_empty"))
            uses_per_work = (
                1
                if spec.final_only_stat_handoff
                and barrier.name.startswith("stat_empty")
                else spec.kv_iterations + 1
                if stat_barrier and spec.pipelined_stat_handoff
                else spec.kv_iterations
                if barrier.name.startswith(
                    (
                        "k_ready",
                        "v_ready",
                        "k_reuse",
                        "v_reuse",
                        "s_full",
                        "stat_ready",
                        "stat_empty",
                        "pfor",
                    )
                )
                else 1
            )
            initial_phase = (
                1
                if spec.pipelined_stat_handoff and barrier.name.startswith("stat_empty")
                else 0
            )
            cycles.append(
                FlashPhaseCycle(
                    barrier.name,
                    (
                        initial_phase,
                        initial_phase ^ (uses_per_work & 1),
                        initial_phase,
                    ),
                    uses_per_work,
                )
            )
        phase_cycles = tuple(cycles)
    tmem_columns = 256 + spec.query_slots_per_cta * spec.head_dim
    return FlashSchedule(
        spec=spec,
        nodes=tuple(nodes),
        edges=tuple(edges),
        barriers=tuple(barriers),
        memory_regions=tuple(regions),
        output_owners=tuple(owners),
        phase_cycles=phase_cycles,
        shared_memory_bytes=_shared_memory_bytes(spec),
        tmem_columns=tmem_columns,
    )


def _reachable(
    edges_by_source: dict[str, list[FlashEdge]],
    source: str,
    target: str,
    iteration_delta: int,
) -> bool:
    pending = [(source, 0)]
    seen: set[tuple[str, int]] = set()
    while pending:
        node, delta = pending.pop()
        state = (node, delta)
        if state in seen or delta > iteration_delta:
            continue
        seen.add(state)
        if node == target and delta == iteration_delta:
            return True
        pending.extend(
            (edge.target, delta + edge.iteration_delta)
            for edge in edges_by_source.get(node, ())
        )
    return False


def _validate_zero_delta_acyclic(
    node_names: set[str], edges: tuple[FlashEdge, ...]
) -> None:
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = dict.fromkeys(node_names, 0)
    for edge in edges:
        if edge.iteration_delta == 0:
            adjacency[edge.source].append(edge.target)
            indegree[edge.target] += 1
    pending = [name for name, degree in indegree.items() if degree == 0]
    visited = 0
    while pending:
        source = pending.pop()
        visited += 1
        for target in adjacency.get(source, ()):
            indegree[target] -= 1
            if indegree[target] == 0:
                pending.append(target)
    if visited != len(node_names):
        raise FlashScheduleError("same-iteration schedule contains a cycle")


def verify_flash_schedule(
    schedule: FlashSchedule,
    limits: FlashScheduleLimits | None = None,
) -> VerifiedFlashSchedule:
    if limits is None:
        limits = FlashScheduleLimits()
    if limits.shared_memory_bytes <= 0 or limits.tmem_columns <= 0:
        raise FlashScheduleError("schedule limits must be positive")

    node_by_name = {node.name: node for node in schedule.nodes}
    node_names = set(node_by_name)
    if len(node_names) != len(schedule.nodes):
        raise FlashScheduleError("schedule node names must be unique")
    for node in schedule.nodes:
        if (
            node.cta_rank is not None
            and not 0 <= node.cta_rank < schedule.spec.cta_count
        ):
            raise FlashScheduleError(f"node {node.name} has an invalid CTA rank")
        if node.query_slot is not None:
            if node.cta_rank is None or not (
                0 <= node.query_slot < schedule.spec.query_slots_per_cta
            ):
                raise FlashScheduleError(f"node {node.name} has an invalid query slot")
    for edge in schedule.edges:
        if edge.source not in node_names or edge.target not in node_names:
            raise FlashScheduleError("schedule edge references an unknown node")
        if edge.iteration_delta < 0:
            raise FlashScheduleError("iteration deltas must be nonnegative")
    _validate_zero_delta_acyclic(node_names, schedule.edges)

    barrier_by_name = {barrier.name: barrier for barrier in schedule.barriers}
    if len(barrier_by_name) != len(schedule.barriers):
        raise FlashScheduleError("barrier names must be unique")
    if any(barrier.expected_arrivals <= 0 for barrier in schedule.barriers):
        raise FlashScheduleError("barrier arrivals must be positive")
    arrivals: dict[str, int] = defaultdict(int)
    for edge in schedule.edges:
        if edge.barrier is None:
            if edge.arrival_count != 0:
                raise FlashScheduleError("arrival count requires a barrier")
            continue
        barrier = barrier_by_name.get(edge.barrier)
        if barrier is None:
            raise FlashScheduleError("edge references an unknown barrier")
        if edge.scope is not barrier.scope:
            raise FlashScheduleError(f"barrier scope mismatch for {barrier.name}")
        if edge.arrival_count <= 0:
            raise FlashScheduleError("barrier arrivals must be positive")
        arrivals[barrier.name] += edge.arrival_count
    for barrier in schedule.barriers:
        if arrivals[barrier.name] != barrier.expected_arrivals:
            raise FlashScheduleError(
                f"barrier {barrier.name} expected {barrier.expected_arrivals} "
                f"arrivals, got {arrivals[barrier.name]}"
            )
    if schedule.spec.pipelined_stat_handoff:
        expected_releases = {
            FlashEdge(
                _node_name("correction", rank, source_slot),
                _node_name("stat", rank, target_slot),
                iteration_delta,
                f"stat_empty_r{rank}_q{target_slot}",
                1,
            )
            for rank in range(schedule.spec.cta_count)
            for source_slot in range(schedule.spec.query_slots_per_cta)
            for target_slot, iteration_delta in (
                (_pipelined_stat_release(schedule.spec, source_slot),)
            )
        }
        release_edges = [
            edge
            for edge in schedule.edges
            if edge.barrier is not None and edge.barrier.startswith("stat_empty_")
        ]
        if (
            len(release_edges) != len(expected_releases)
            or set(release_edges) != expected_releases
        ):
            raise FlashScheduleError(
                "pipelined stat releases do not match the selected "
                f"{schedule.spec.stat_release_mapping.value} mapping"
            )
    if schedule.spec.final_only_stat_handoff:
        edge_keys = {
            (edge.source, edge.target, edge.iteration_delta, edge.barrier)
            for edge in schedule.edges
        }
        for rank in range(schedule.spec.cta_count):
            for slot in range(schedule.spec.query_slots_per_cta):
                correction = _node_name("correction", rank, slot)
                stat = _node_name("stat", rank, slot)
                publish = _node_name("final_stat_publish", rank, slot)
                consume = _node_name("final_stat_consume", rank, slot)
                if (
                    (correction, publish, 0, f"stat_empty_r{rank}_q{slot}")
                    not in edge_keys
                    or (publish, consume, 0, None) not in edge_keys
                    or (
                        schedule.spec.persistent
                        and (consume, stat, 1, None) not in edge_keys
                    )
                ):
                    raise FlashScheduleError(
                        "final-only stat handoff is missing terminal ordering"
                    )
    if schedule.spec.multicast_kv:
        for rank in range(schedule.spec.cta_count):
            for operand, consumer_kind in (("k", "qk"), ("v", "pv")):
                barrier_name = f"{operand}_reuse_r{rank}"
                release_edges = [
                    edge for edge in schedule.edges if edge.barrier == barrier_name
                ]
                source_ranks = (
                    (0,)
                    if schedule.spec.cooperative_mma
                    else range(schedule.spec.cta_count)
                )
                expected_sources = {
                    _node_name(
                        consumer_kind,
                        source_rank,
                        schedule.spec.query_slots_per_cta - 1,
                    )
                    for source_rank in source_ranks
                }
                if (
                    len(release_edges) != len(expected_sources)
                    or {edge.source for edge in release_edges} != expected_sources
                    or any(
                        edge.target != f"{operand}_load_multicast"
                        or edge.iteration_delta != schedule.spec.kv_depth
                        or edge.arrival_count != 1
                        or edge.scope is not FlashSyncScope.CLUSTER
                        for edge in release_edges
                    )
                ):
                    raise FlashScheduleError(
                        "multicast K/V reuse has the wrong consumer releases"
                    )

    region_by_name = {region.name: region for region in schedule.memory_regions}
    if len(region_by_name) != len(schedule.memory_regions):
        raise FlashScheduleError("memory-region names must be unique")
    edges_by_source: dict[str, list[FlashEdge]] = defaultdict(list)
    resource_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for edge in schedule.edges:
        edges_by_source[edge.source].append(edge)
        if edge.multicast and edge.resource is None:
            raise FlashScheduleError("multicast edge requires a memory resource")
        if edge.resource is None:
            continue
        region = region_by_name.get(edge.resource)
        if region is None:
            raise FlashScheduleError("edge references an unknown memory region")
        if edge.source != region.writer or edge.target not in region.consumers:
            raise FlashScheduleError(
                f"edge does not match the lifetime of memory region {region.name}"
            )
        resource_edges[region.name].add((edge.source, edge.target))
        if region.visibility is FlashVisibility.CLUSTER_VISIBLE:
            if edge.resource_scope not in (
                FlashSyncScope.CLUSTER,
                FlashSyncScope.CLUSTER_LEADER,
            ):
                raise FlashScheduleError("cluster-visible data uses a CTA-local edge")
            if not edge.multicast:
                raise FlashScheduleError("cluster-visible load is not multicast")
        elif edge.resource_scope is not FlashSyncScope.CTA:
            raise FlashScheduleError("CTA-private data uses a cluster-scoped edge")
        if edge.multicast and region.visibility is FlashVisibility.CTA_PRIVATE:
            raise FlashScheduleError("CTA-private data cannot be multicast")
        if edge.multicast and not schedule.spec.multicast_kv:
            raise FlashScheduleError("multicast edge is disabled by the schedule")
        if region.visibility is FlashVisibility.CTA_PRIVATE:
            source_rank = node_by_name[edge.source].cta_rank
            target_rank = node_by_name[edge.target].cta_rank
            if region.cta_rank is None:
                raise FlashScheduleError(
                    f"CTA-private memory region {region.name} has no owner"
                )
            if any(
                rank is not None and rank != region.cta_rank
                for rank in (source_rank, target_rank)
            ):
                raise FlashScheduleError(
                    f"CTA-private memory region {region.name} crosses CTA ranks"
                )

    for region in schedule.memory_regions:
        if region.offset < 0 or region.extent <= 0:
            raise FlashScheduleError("memory regions require positive bounds")
        if region.alignment <= 0 or region.offset % region.alignment != 0:
            raise FlashScheduleError(f"memory region {region.name} is misaligned")
        if region.writer not in node_names or any(
            consumer not in node_names for consumer in region.consumers
        ):
            raise FlashScheduleError("memory lifetime references an unknown node")
        if not region.consumers:
            raise FlashScheduleError(f"memory region {region.name} has no consumer")
        if region.cta_rank is not None and not (
            0 <= region.cta_rank < schedule.spec.cta_count
        ):
            raise FlashScheduleError(
                f"memory region {region.name} has an invalid CTA rank"
            )
        expected_edges = {(region.writer, consumer) for consumer in region.consumers}
        if not expected_edges.issubset(resource_edges[region.name]):
            raise FlashScheduleError(
                f"memory region {region.name} has an uncovered consumer"
            )
        if region.reuse_distance is not None:
            if region.reuse_distance <= 0:
                raise FlashScheduleError("reuse distance must be positive")
            for consumer in region.consumers:
                if not _reachable(
                    edges_by_source,
                    consumer,
                    region.writer,
                    region.reuse_distance,
                ):
                    raise FlashScheduleError(
                        f"memory region {region.name} is reused before all consumers"
                    )
            if region.reuse_barrier is not None:
                reuse_releases = tuple(
                    (consumer, region.reuse_distance) for consumer in region.consumers
                )
                writer = node_by_name[region.writer]
                if (
                    schedule.spec.pipelined_stat_handoff
                    and writer.kind is FlashNodeKind.STAT_PUBLISH
                ):
                    assert writer.cta_rank is not None
                    assert writer.query_slot is not None
                    source_slot, iteration_delta = _pipelined_stat_releaser(
                        schedule.spec, writer.query_slot
                    )
                    reuse_releases = (
                        (
                            _node_name(
                                "correction",
                                writer.cta_rank,
                                source_slot,
                            ),
                            iteration_delta,
                        ),
                    )
                for release_source, iteration_delta in reuse_releases:
                    if not any(
                        edge.source == release_source
                        and edge.target == region.writer
                        and edge.iteration_delta == iteration_delta
                        and edge.barrier == region.reuse_barrier
                        for edge in schedule.edges
                    ):
                        raise FlashScheduleError(
                            f"memory region {region.name} is reused before all consumers"
                        )
        elif region.reuse_barrier is not None:
            raise FlashScheduleError("reuse barrier requires a reuse distance")

    for rank in range(schedule.spec.cta_count):
        k_region = region_by_name.get(f"K_r{rank}")
        v_region = region_by_name.get(f"V_r{rank}")
        if k_region is None or v_region is None:
            raise FlashScheduleError("K/V ring representation is incomplete")
        if schedule.spec.separate_kv:
            if (
                not k_region.physical
                or not v_region.physical
                or k_region.reuse_distance != schedule.spec.kv_depth
                or v_region.reuse_distance != schedule.spec.kv_depth
            ):
                raise FlashScheduleError(
                    "separate K/V rings must retain independent depth"
                )
        elif (
            not k_region.physical
            or v_region.physical
            or k_region.offset != v_region.offset
            or k_region.extent != v_region.extent
            or k_region.alias_group is None
            or k_region.alias_group != v_region.alias_group
            or k_region.reuse_distance is not None
            or v_region.reuse_distance is not None
        ):
            raise FlashScheduleError(
                "aliased K/V must use one physical shared ring without "
                "independent reuse claims"
            )

    for index, lhs in enumerate(schedule.memory_regions):
        for rhs in schedule.memory_regions[index + 1 :]:
            if (
                not lhs.physical
                or not rhs.physical
                or lhs.space is not rhs.space
                or lhs.cta_rank != rhs.cta_rank
            ):
                continue
            overlaps = lhs.offset < rhs.offset + rhs.extent and rhs.offset < (
                lhs.offset + lhs.extent
            )
            if not overlaps:
                continue
            if lhs.alias_group is None or lhs.alias_group != rhs.alias_group:
                raise FlashScheduleError(
                    f"memory regions {lhs.name} and {rhs.name} overlap illegally"
                )
            lhs_before_rhs = all(
                _reachable(edges_by_source, consumer, rhs.writer, 0)
                for consumer in lhs.consumers
            )
            rhs_before_lhs = all(
                _reachable(edges_by_source, consumer, lhs.writer, 0)
                for consumer in rhs.consumers
            )
            if not lhs_before_rhs and not rhs_before_lhs:
                raise FlashScheduleError(
                    f"aliased regions {lhs.name} and {rhs.name} have live overlap"
                )

    output_count = schedule.spec.cta_count * schedule.spec.query_slots_per_cta
    output_tiles = [owner.output_tile for owner in schedule.output_owners]
    if len(output_tiles) != len(set(output_tiles)):
        raise FlashScheduleError("an output tile has multiple owners")
    if set(output_tiles) != set(range(output_count)):
        raise FlashScheduleError("output ownership is incomplete")
    for owner in schedule.output_owners:
        if not 0 <= owner.cta_rank < schedule.spec.cta_count:
            raise FlashScheduleError("output owner has an invalid CTA rank")
        if not 0 <= owner.query_slot < schedule.spec.query_slots_per_cta:
            raise FlashScheduleError("output owner has an invalid query slot")
        expected_tile = _output_tile(
            schedule.spec,
            owner.cta_rank,
            owner.query_slot,
        )
        if owner.output_tile != expected_tile:
            raise FlashScheduleError("output owner has an invalid tile mapping")
        output_store = _node_name("output_store", owner.cta_rank, owner.query_slot)
        if output_store not in node_names:
            raise FlashScheduleError("output owner has no output-store node")

    if schedule.spec.private_causal_tail:
        for rank in range(schedule.spec.cta_count):
            for kind, consumer_kind in (("K", "qk"), ("V", "pv")):
                name = f"causal_diagonal_{kind}_r{rank}"
                region = region_by_name.get(name)
                if (
                    region is None
                    or region.visibility is not FlashVisibility.CTA_PRIVATE
                    or region.cta_rank != rank
                    or region.physical
                ):
                    raise FlashScheduleError(
                        "causal multicast requires a private diagonal path"
                    )
                for slot in range(schedule.spec.query_slots_per_cta):
                    consumer = _node_name(consumer_kind, rank, slot)
                    if (region.writer, consumer) not in resource_edges[name]:
                        raise FlashScheduleError(
                            "causal multicast diagonal dependency is incomplete"
                        )

    expected_shared_memory = _shared_memory_bytes(schedule.spec)
    if schedule.shared_memory_bytes != expected_shared_memory:
        raise FlashScheduleError("shared-memory accounting does not match the schedule")
    expected_tmem_columns = (
        256 + schedule.spec.query_slots_per_cta * schedule.spec.head_dim
    )
    if schedule.tmem_columns != expected_tmem_columns:
        raise FlashScheduleError("TMEM accounting does not match the schedule")
    if schedule.shared_memory_bytes > limits.shared_memory_bytes:
        raise FlashScheduleError("schedule exceeds shared-memory capacity")
    if schedule.tmem_columns > limits.tmem_columns:
        raise FlashScheduleError("schedule exceeds TMEM column capacity")
    for region in schedule.memory_regions:
        if not region.physical:
            continue
        capacity = (
            schedule.shared_memory_bytes
            if region.space is FlashMemorySpace.SMEM
            else schedule.tmem_columns
        )
        if region.offset + region.extent > capacity:
            raise FlashScheduleError(f"memory region {region.name} exceeds capacity")

    phase_by_barrier = {cycle.barrier: cycle for cycle in schedule.phase_cycles}
    if len(phase_by_barrier) != len(schedule.phase_cycles):
        raise FlashScheduleError("barrier phase cycles must be unique")
    if schedule.spec.persistent:
        if set(phase_by_barrier) != set(barrier_by_name):
            raise FlashScheduleError("persistent barrier phase coverage is incomplete")
        for cycle in schedule.phase_cycles:
            if cycle.uses_per_work <= 0:
                raise FlashScheduleError("barrier use count must be positive")
            if len(cycle.phases) < 2:
                raise FlashScheduleError("barrier phase cycle has the wrong length")
            if any(phase not in (0, 1) for phase in cycle.phases):
                raise FlashScheduleError("barrier phases must be binary")
            for current, following in zip(
                cycle.phases[:-1],
                cycle.phases[1:],
                strict=True,
            ):
                if following != (current ^ (cycle.uses_per_work & 1)):
                    raise FlashScheduleError(
                        "persistent barrier phase is discontinuous across work items"
                    )
    elif schedule.phase_cycles:
        raise FlashScheduleError("nonpersistent schedule has barrier phase cycles")

    return VerifiedFlashSchedule(schedule)
