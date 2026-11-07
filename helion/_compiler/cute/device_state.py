"""``CuteDeviceFunctionState``: the per-``DeviceFunction`` state of the CuTe backend.

Role: state.  ``Backend.new_device_state(config)`` builds one per kernel
render (``state.new_device_state``); the fields are grouped into the
namespaces defined in ``state.py`` (``tcgen05``, ``flash``, ``lanes``,
``root_plans``); every reader names the namespace (the flat-name
``AttributeAlias`` layer each field had after the grouping is retired).  The
tcgen05 plan records the pipeline emitters read live here too:
``CuteTcgen05MatmulPlan`` (the kernel-wide collective contract),
``CuteTcgen05GroupedPlan``, ``CuteTcgen05StoreValue`` and the vloop-sink facts.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING
from typing import Literal
from typing import Protocol
from typing import cast

from ... import exc
from .state import CuteFlashState
from .state import CuteLaneState
from .state import CuteRootPlans
from .state import CuteTcgen05State

if TYPE_CHECKING:
    import ast
    from collections.abc import Sequence

    import torch
    from torch.fx.node import Node

    from .aux_tensor import Tcgen05AuxTensorDescriptor
    from .cute_epilogue import Tcgen05GroupedTailEpilogueMatch
    from .cute_mma import _Tcgen05AuxPipelinePlan
    from .cute_mma import _Tcgen05SchedPipelinePlan
    from .epilogue_fanout import FanoutStore
    from .fragment_epilogue import Tcgen05FragmentEpiloguePlan
    from .grouped_balanced_workers import GroupedBalancedWorkersPlan
    from .grouped_full_coverage import Tcgen05GroupedFullCoveragePlan
    from .grouped_row_union import GroupedRowUnionPlan
    from .published_scalars import PublishedScalarRequest
    from .register_producers import RegisterProducerRequest
    from .signed_bitfield import SignedBytePacket
    from .tcgen05_lifecycle import Tcgen05LifecycleContext


@dataclasses.dataclass(frozen=True)
class Tcgen05AbStartupPrefill:
    """Exactly S initial packets, carried into the normal producer state.

    The startup guard uses a private clone. Only the later producer role
    advances the original state, before consuming its first scheduler record.
    """

    stages: int
    producer_state: str
    first_record: str

    def role_prelude(self) -> str:
        return (
            f"{self.first_record} = cutlass.Boolean(True)\n"
            f"for _tcgen05_prefill_resume in cutlass.range({self.stages}, unroll_full=True):\n"
            f"    {self.producer_state}.advance()"
        )


class Tcgen05Orientation(enum.Enum):
    MN = enum.auto()
    NM = enum.auto()


class Tcgen05GroupedDMode(enum.Enum):
    NONE = enum.auto()
    ALL_TILES = enum.auto()
    EDGE_ONLY = enum.auto()


class Tcgen05GroupedSchedulerMode(enum.Enum):
    """How a grouped persistent kernel selects its next logical tile."""

    DEVICE_GROUP_SEARCH = "device_group_search"
    RUNTIME_DIRECT = "runtime_direct"
    RUNTIME_CLC = "runtime_clc"


@dataclasses.dataclass(frozen=True)
class CuteTcgen05GroupedPlan:
    orientation: Tcgen05Orientation
    layout: str
    count: str
    sched_params: str
    problem_sizes: str
    starts: str
    metadata_idx: str
    group_idx: str
    cta_tile_idx_m: str
    cta_tile_idx_n: str
    problem_m: str
    problem_n: str
    problem_k: str
    global_m_start: str
    scheduler_mode: Tcgen05GroupedSchedulerMode = (
        Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH
    )
    # Host-expanded per-logical-tile records for the runtime N,M worklist
    # direct scheduler.  Unlike ``static_problem_shapes``, this table is built
    # from the current worklist values by the launcher and is therefore reusable
    # by one compiled kernel across routing vectors.  ``total_clusters`` is a
    # runtime scalar because the table row count is launch metadata, not an AOT
    # problem signature.
    runtime_tile_records: str | None = None
    runtime_total_clusters: str | None = None
    static_problem_shapes: tuple[tuple[int, int, int], ...] | None = None
    static_group_quota_args: tuple[str, ...] = ()
    real_groups: str | None = None
    valid_m: str | None = None
    store_m: str | None = None
    direct_pointers: str | None = None
    direct_strides: str | None = None
    d_mode: Tcgen05GroupedDMode = Tcgen05GroupedDMode.NONE
    d_tensormap: str | None = None
    fixed_tensormaps: bool = False
    # N,M worklists carry their source-row tile explicitly so runtime metadata
    # validation and launch bounds consume the exact schedule selected by the
    # compiler.  For compact device layouts, ``layout`` names either the
    # ``split_sizes[G]`` or ``offsets[G + 1]`` tensor while ``problem_sizes``
    # and ``starts`` are kernel-local SMEM tensors.  ``m_size`` lets the
    # launcher derive a safe static cluster bound without reading layout
    # values on the host.
    source_m_tile: int | None = None
    m_size: int | None = None
    # The typed device-layout initializer writes this guarded input feature
    # extent into every group's N column. None keeps the generic panel decoder.
    device_n_size: int | None = None
    device_layout_kind: Literal["split_sizes", "offsets"] | None = None
    # The source program drops negative-address rows before applying its
    # static M cap; its original signed offset difference remains unchanged.
    clipped_negative_start: bool = False
    # A fixed rank-3 S2G descriptor encodes the clipped row interval through
    # coordinates. It retains ALL_TILES tail semantics without a mutable map.
    bounded_output_shift: bool = False
    worker_balance: GroupedBalancedWorkersPlan | None = None
    full_coverage: Tcgen05GroupedFullCoveragePlan | None = None

    def __post_init__(self) -> None:
        assert (self.valid_m is None) == (self.store_m is None)
        assert (self.orientation is Tcgen05Orientation.NM) == (self.valid_m is not None)
        assert (
            self.orientation is Tcgen05Orientation.NM or self.device_split_sizes
        ) == (self.source_m_tile is not None)
        assert (self.direct_pointers is None) == (self.direct_strides is None)
        assert (
            self.d_mode is Tcgen05GroupedDMode.NONE or self.bounded_output_shift
        ) == (self.d_tensormap is None)
        if self.bounded_output_shift:
            assert self.device_split_sizes
            assert self.d_mode is Tcgen05GroupedDMode.ALL_TILES
            assert not self.fixed_tensormaps
            assert self.direct_pointers is None
        if self.worker_balance is not None:
            assert self.orientation is Tcgen05Orientation.MN
            assert self.device_split_sizes
            assert (
                self.scheduler_mode is Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH
            )
            assert self.direct_pointers is None
            assert not self.fixed_tensormaps
            assert self.full_coverage is None
        assert not self.device_split_sizes or self.source_m_tile is not None
        assert self.device_n_size is None or (
            self.device_split_sizes and self.device_n_size > 0
        )
        assert self.device_layout_kind in (None, "split_sizes", "offsets")
        assert not self.clipped_negative_start or self.device_layout_kind == "offsets"
        assert (self.device_layout_kind is not None) == self.device_split_sizes
        assert (self.runtime_tile_records is None) == (
            self.runtime_total_clusters is None
        )
        if self.scheduler_mode is Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH:
            assert self.runtime_tile_records is None
            assert self.runtime_total_clusters is None
        elif self.scheduler_mode in (
            Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT,
            Tcgen05GroupedSchedulerMode.RUNTIME_CLC,
        ):
            assert self.runtime_tile_records is not None
            assert self.runtime_total_clusters is not None
            assert self.orientation is Tcgen05Orientation.NM
            assert not self.device_split_sizes
            assert self.static_problem_shapes is None
            if self.scheduler_mode is Tcgen05GroupedSchedulerMode.RUNTIME_CLC:
                assert self.fixed_tensormaps
        else:
            raise AssertionError(
                f"unhandled grouped scheduler mode: {self.scheduler_mode!r}"
            )
        if self.orientation is Tcgen05Orientation.NM:
            assert (
                self.uses_runtime_tile_table
                or self.real_groups is not None
                or self.device_split_sizes
            )
            assert self.fixed_tensormaps == (self.d_mode is Tcgen05GroupedDMode.NONE)
        if self.fixed_tensormaps:
            assert self.orientation is Tcgen05Orientation.NM
            assert not self.device_split_sizes
            assert self.direct_pointers is None
        if self.full_coverage is not None:
            assert self.orientation is Tcgen05Orientation.NM
            assert self.device_layout_kind == "offsets"
            assert self.clipped_negative_start
            assert not self.fixed_tensormaps
            assert self.direct_pointers is None
            assert self.d_mode is Tcgen05GroupedDMode.ALL_TILES
            assert (
                self.scheduler_mode is Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH
            )
            assert self.source_m_tile == self.full_coverage.tile_m
            assert self.m_size == self.full_coverage.m
            assert int(self.count) == self.full_coverage.groups
        if self.static_problem_shapes is not None:
            assert self.orientation is Tcgen05Orientation.MN
            assert self.real_groups is None
        if self.static_group_quota_args:
            assert self.static_problem_shapes is not None
            assert len(self.static_group_quota_args) == len(self.static_problem_shapes)

    @property
    def device_split_sizes(self) -> bool:
        return self.m_size is not None

    @property
    def uses_runtime_tile_table(self) -> bool:
        return self.scheduler_mode in (
            Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT,
            Tcgen05GroupedSchedulerMode.RUNTIME_CLC,
        )


class _CuteTcgen05Orientation(Protocol):
    @property
    def bm(self) -> int: ...

    @property
    def bn(self) -> int: ...

    @property
    def orientation(self) -> Tcgen05Orientation: ...


class _CuteTcgen05OrientationMixin:
    def _orientation(self) -> _CuteTcgen05Orientation:
        return cast("_CuteTcgen05Orientation", self)

    def _is_nm(self) -> bool:
        return self._orientation().orientation is Tcgen05Orientation.NM

    @property
    def source_tile_m(self) -> int:
        orientation = self._orientation()
        return orientation.bn if self._is_nm() else orientation.bm

    @property
    def source_tile_n(self) -> int:
        orientation = self._orientation()
        return orientation.bm if self._is_nm() else orientation.bn

    @property
    def accumulator_view(self) -> str:
        return "nm" if self._is_nm() else "mn"

    @property
    def output_view(self) -> str:
        return self.accumulator_view

    @property
    def d_store_view(self) -> str:
        return "nm_transposed" if self._is_nm() else "normal"

    @property
    def d_store_layout(self) -> str:
        layout = "COL_MAJOR" if self._is_nm() else "ROW_MAJOR"
        return f"cutlass.utils.layout.LayoutEnum.{layout}"


@dataclasses.dataclass(frozen=True)
class CuteTcgen05StoreValue(_CuteTcgen05OrientationMixin):
    lifecycle_context: Tcgen05LifecycleContext
    output_block_ids: tuple[int, ...]
    output_stores: tuple[Node, ...] | None = None
    bm: int = 0
    bn: int = 0
    bk: int = 0
    thr_mma: str = ""
    # Epilogue warps per warp group (the four TMEM datapath warps the CUTLASS
    # partition helpers address through ``epi_tidx``).
    epi_warp_count: int = 0
    # Epilogue warp groups (``tcgen05_epi_warp_groups``): ``epi_group`` /
    # ``epi_leader`` are the per-warp group index and leader-warp predicate
    # the role prefix defines when there are several groups ("" otherwise).
    epi_warp_groups: int = 1
    epi_group: str = ""
    epi_leader: str = ""
    epi_acc_frag_base: str = ""
    epi_tidx: str = ""
    warp_idx: str = ""
    epi_tile: str = ""
    # Stages of each output's TMA-store C ring: ``c_stages_per_group`` per
    # epilogue warp group, ``c_stage_count`` in total.
    c_stage_count: int = 0
    c_stages_per_group: int = 0
    epilog_sync_barrier_id: int = 0
    tmem_load_atom: str = ""
    epilogue_rest_mode: str = ""
    tma_store_atom: str = ""
    tma_store_tensor: str = ""
    tail_tma_store_atom: str = ""
    tail_tma_store_tensor: str = ""
    role_local_tile_counter: str = ""
    use_role_local_epi: bool = False
    use_tma_store_epilogue: bool = False
    tma_store_full_tiles_only: bool = False
    partial_output_tma_store: bool = False
    # Output element dtype (cutlass type string, e.g. "cutlass.BFloat16")
    # used when computing the tcgen05 epilogue tile.
    epi_elem_dtype_str: str = ""
    explicit_epi_tile_m: int | None = None
    explicit_epi_tile_n: int | None = None
    explicit_d_store_box_n: int | None = None
    segment_store_m_offset: str = ""
    segment_store_start: str = ""
    segment_store_actual_m: str = ""
    segment_store_valid_m_bound: str = ""
    segment_store_node: Node | None = None
    segment_store_row_index: Node | None = None
    segment_store_valid_m: Node | None = None
    orientation: Tcgen05Orientation = Tcgen05Orientation.MN
    output_column_major: bool = False
    row_union: GroupedRowUnionPlan | None = None
    # ``tcgen05_split_k``: CTA-uniform predicate naming the tile's last
    # arriving split (the only CTA that runs the output stores).
    split_k_last: str = ""

    @property
    def d_store_layout(self) -> str:
        if self.output_column_major:
            return "cutlass.utils.layout.LayoutEnum.COL_MAJOR"
        return super().d_store_layout

    def __post_init__(self) -> None:
        explicit_tile_values = (
            self.explicit_epi_tile_m,
            self.explicit_epi_tile_n,
            self.explicit_d_store_box_n,
        )
        assert all(value is None for value in explicit_tile_values) or all(
            value is not None for value in explicit_tile_values
        )
        if self.explicit_d_store_box_n is not None:
            assert self.explicit_epi_tile_n == self.explicit_d_store_box_n
        if self.c_stages_per_group == 0:
            object.__setattr__(self, "c_stages_per_group", self.c_stage_count)
        assert self.epi_warp_groups >= 1
        assert self.c_stage_count == self.c_stages_per_group * self.epi_warp_groups
        assert (
            (self.epi_warp_groups > 1) == bool(self.epi_group) == bool(self.epi_leader)
        )

    @property
    def has_explicit_epilogue_tile(self) -> bool:
        return self.explicit_epi_tile_m is not None


@dataclasses.dataclass(frozen=True)
class CuteTcgen05MatmulPlan(_CuteTcgen05OrientationMixin):
    """Kernel-wide tcgen05 collective contract selected by CuTe matmul codegen.

    The warp-role order is part of the codegen contract: epilogue warps occupy
    the low warp IDs, then one MMA-exec warp, AB/TMA-load warps, optional
    scheduler warps, and optional C-input/aux warps. Predicate builders,
    pipeline arrive counts, and generated launch metadata all derive from this
    layout, so changes here are behavioral changes.
    """

    bm: int
    bn: int
    bk: int
    k_tile_count: int
    cluster_m: int
    is_two_cta: bool
    uses_role_local_persistent_body: bool
    cta_thread_count: int
    physical_m_threads: int
    acc_stage_count: int
    ab_stage_count: int
    c_stage_count: int
    epi_warp_count: int
    # The actual logical M/N offset names, independent of block numbering
    # and launch-axis ordering. Auxiliary producers share these coordinates
    # with the MMA and epilogue roles after PID remapping.
    output_offsets: tuple[str, str] | None = None
    ab_load_warp_count: int = 1
    one_shot_role_scheduler: bool = False
    # Dedicated scheduler warp count for ROLE_LOCAL_WITH_SCHEDULER. Default
    # zero keeps MONOLITHIC's historical role IDs; one adds a scheduler warp
    # after the AB-load warp that publishes work-tile metadata through the
    # scheduler pipeline.
    scheduler_warp_count: int = 0
    sched_stage_count: int = 0
    # Optional C-input / auxiliary-tensor warp. WITH_SCHEDULER may lift this
    # to one warp; launched_warp_count still rounds to a warpgroup-aligned
    # envelope, so the lifted warp occupies the previous inert padding slot.
    # The scheduler-pipeline arrive count includes this warp only when the
    # productive aux-body gate is open.
    c_input_warp_count: int = 0
    # Optional epilogue store/drain warp (Workstream A Stage 3, cycle 91).
    # WITH_SCHEDULER may lift this to one warp; like ``c_input_warp_count`` the
    # lifted warp occupies the previous inert padding slot so
    # ``launched_warp_count`` stays warpgroup-aligned (no extra warp launched).
    # The store warp's body is inert in cycle 91 (the TMA-D store still runs on
    # warp 0); Stage 4 moves the R2S->TMA-D drain onto it. It sits at the END of
    # the warp-id order (after the C-input warp) so existing warp ids do not
    # shift.
    store_warp_count: int = 0
    persistence_model: str = "static_persistent"
    cluster_n: int = 1
    l2_swizzle_size: int = 1
    tma_store_full_tiles_only: bool = False
    # M-paired tiles: number of 256-row UMMA subtiles per work tile along M
    # (block_m // mma tile M). 2 stages B once per K stage and shares it
    # across both subtiles, with one TMEM accumulator stage per subtile.
    m_subtile_count: int = 1
    flat_role_launch_warp_count: int | None = None
    grouped: CuteTcgen05GroupedPlan | None = None
    row_union: GroupedRowUnionPlan | None = None
    # Per-anchor auxiliary descriptors discovered by the forward FX walker. This
    # is store-fusion metadata, not a collective compatibility field:
    # two matmuls with identical collective parameters but different downstream
    # aux tensors must share one collective plan. compare=False is also required
    # because descriptor equality reaches tensor-valued fields whose ``==`` does
    # not produce a scalar bool suitable for dataclass equality.
    aux_tensor_descriptors: tuple[Tcgen05AuxTensorDescriptor, ...] = dataclasses.field(
        default=(), compare=False
    )

    def __post_init__(self) -> None:
        if self.row_union is not None:
            assert self.grouped is None
            assert self.uses_role_local_persistent_body
            if self.row_union.linear_record_clc:
                assert self.has_scheduler_warp and self.is_clc_persistent
                protocol = self.row_union.paired_protocol
                assert protocol is not None
                assert self.sched_stage_count == protocol.scheduler_stages
                assert self.c_input_warp_count == self.store_warp_count == 0
                assert self.epi_warp_count == 4 and self.ab_load_warp_count == 1
            else:
                assert not self.has_scheduler_warp
                assert not self.is_clc_persistent
            if self.row_union.schedule is None:
                assert not self.is_two_cta and self.cluster_m == self.cluster_n == 1
            else:
                schedule = self.row_union.schedule
                assert self.is_two_cta
                assert (self.bm, self.bn, self.bk) == (
                    schedule.mma_m,
                    schedule.mma_n,
                    schedule.block_k,
                )
                assert (self.cluster_m, self.cluster_n) == (
                    schedule.cluster_m,
                    schedule.cluster_n,
                )
        if self.grouped is None:
            return
        scheduler_mode = self.grouped.scheduler_mode
        if scheduler_mode is Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH:
            if self.grouped.orientation is Tcgen05Orientation.NM:
                assert self.uses_role_local_persistent_body
                assert self.has_scheduler_warp
                assert not self.is_clc_persistent
            return
        if scheduler_mode is Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT:
            assert self.uses_role_local_persistent_body
            assert not self.has_scheduler_warp
            assert not self.is_clc_persistent
            return
        if scheduler_mode is Tcgen05GroupedSchedulerMode.RUNTIME_CLC:
            assert self.uses_role_local_persistent_body
            assert self.has_scheduler_warp
            assert self.is_clc_persistent
            return
        raise AssertionError(f"unhandled grouped scheduler mode: {scheduler_mode!r}")

    @property
    def orientation(self) -> Tcgen05Orientation:
        if self.row_union is not None and self.row_union.schedule is not None:
            return Tcgen05Orientation.NM
        return self.grouped.orientation if self.grouped else Tcgen05Orientation.MN

    @property
    def c_input_aux_tensor_descriptors(self) -> tuple[Tcgen05AuxTensorDescriptor, ...]:
        """Aux descriptors staged by the C-input warp.

        Exact-shape rank-2 MxN aux tensors use the SMEM-ring producer.
        Broadcast row vectors and leading-passthrough rank-3 residuals stay on
        the direct per-thread load path; the producer scheduler is 2-D only.
        """
        return tuple(
            d
            for d in self.aux_tensor_descriptors
            if d.broadcast_axis is None and d.host_tensor_val.ndim == 2
        )

    @property
    def is_clc_persistent(self) -> bool:
        # Lazy import keeps this CuTe state module out of strategy import cycles
        # and compares against the enum value, so an enum rename does not leave a
        # stale string literal silently evaluating false.
        from .strategies import Tcgen05PersistenceModel

        return self.persistence_model == Tcgen05PersistenceModel.CLC_PERSISTENT.value

    @property
    def exec_warp_id(self) -> int:
        # Epilogue warps are first so the historical MONOLITHIC role predicates
        # and accumulator partitioning stay stable.
        return self.epi_warp_count

    @property
    def ab_load_warp_begin(self) -> int:
        return self.exec_warp_id + 1

    @property
    def ab_load_warp_end(self) -> int:
        return self.ab_load_warp_begin + self.ab_load_warp_count

    @property
    def tma_warp_id(self) -> int:
        return self.ab_load_warp_begin

    @property
    def has_scheduler_warp(self) -> bool:
        return self.scheduler_warp_count > 0

    @property
    def scheduler_warp_id(self) -> int:
        # Dedicated scheduler warps sit after AB/TMA-load warps. Callers must
        # guard this read because MONOLITHIC has no scheduler warp.
        assert self.has_scheduler_warp, (
            "scheduler_warp_id is only valid when scheduler_warp_count > 0"
        )
        return self.ab_load_warp_end

    @property
    def persistent_scheduler_owner_warp_id(self) -> int:
        # MONOLITHIC scheduling rides on the TMA warp. WITH_SCHEDULER moves
        # ownership to the dedicated scheduler warp that broadcasts work tiles.
        if self.has_scheduler_warp:
            return self.scheduler_warp_id
        return self.tma_warp_id

    @property
    def has_c_input_warp(self) -> bool:
        return self.c_input_warp_count > 0

    @property
    def c_input_warp_id(self) -> int:
        # The C-input warp is only valid in scheduler-backed role-local kernels;
        # it sits after the scheduler warp and consumes the aux pipeline.
        assert self.has_c_input_warp, (
            "c_input_warp_id is only valid when c_input_warp_count > 0"
        )
        return self.scheduler_warp_id + self.scheduler_warp_count

    @property
    def has_store_warp(self) -> bool:
        return self.store_warp_count > 0

    @property
    def store_warp_id(self) -> int:
        # The store warp is only valid in scheduler-backed role-local kernels;
        # it sits at the END of the warp-id order — after the C-input warp (if
        # present) — so adding it never shifts existing warp ids. With sched=1
        # and no C-input that puts it at warp id 7 (warpgroup 1, warps 4-7),
        # which is the former inert padding slot.
        assert self.has_store_warp, (
            "store_warp_id is only valid when store_warp_count > 0"
        )
        return (
            self.scheduler_warp_id + self.scheduler_warp_count + self.c_input_warp_count
        )

    @property
    def role_warp_count(self) -> int:
        return (
            self.epi_warp_count
            + 1
            + self.ab_load_warp_count
            + self.scheduler_warp_count
            + self.c_input_warp_count
            + self.store_warp_count
        )

    @property
    def launched_warp_count(self) -> int:
        # ``setmaxregister`` is warpgroup-uniform on Blackwell: all 4 warps in a
        # warpgroup must request a compatible register budget. Scheduler-backed
        # kernels therefore round the role count up to a full warpgroup. With
        # the default C-input / store counts this pads 7 role warps to 8; with
        # one C-input OR one store warp (cycle 91, Workstream A Stage 3), that
        # warp occupies the former padding slot and the launch stays at 8.
        # MONOLITHIC keeps its historical 6-warp launch because generated-code
        # golden pins and validated runtime behavior depend on that shape.
        if self.has_scheduler_warp:
            warpgroup = 4
            return (self.role_warp_count + warpgroup - 1) // warpgroup * warpgroup
        return self.role_warp_count

    @property
    def block_shape(self) -> tuple[int, int, int]:
        if self.flat_role_launch_warp_count is not None:
            assert self.flat_role_launch_warp_count >= self.role_warp_count
            return (self.physical_m_threads * self.flat_role_launch_warp_count, 1, 1)
        return (self.physical_m_threads, self.launched_warp_count, 1)


@dataclasses.dataclass(frozen=True)
class CuteVloopWrapperFact:
    """A grid tile axis partitioned into an outer lane loop x constexpr V-loop.

    Recorded by ``PerThreadNDTileStrategy.codegen_grid`` for every
    ``VecLaneWrapper`` so the late vector-loop sinking pass can identify the
    V-loop by its constexpr lane variable and reason about the per-element
    index (``index_var = base_index_var + vec_lane_var``) and bounds mask.
    """

    block_id: int
    vec_width: int
    lane_var: str
    vec_lane_var: str
    base_index_var: str
    index_var: str
    # The block's bounds mask variable, or None when the strategy elided it.
    mask_var: str | None
    # The tile extent is a proven multiple of ``vec_width`` and the tile
    # starts at zero, so every V-wide chunk lies entirely inside or entirely
    # outside the extent: a bound on the per-element index is equivalent to
    # the same bound on the chunk base.
    uniform_vector_mask: bool
    # The tile starts at zero with a unit step, so every per-thread chunk
    # base is a multiple of ``vec_width`` (the extent may still be dynamic:
    # a jagged tile's chunks can straddle it).
    chunk_base_aligned: bool = False


@dataclasses.dataclass(frozen=True)
class CuteVloopLoadFact:
    """A scalar tile load that a sunk V-loop may turn into one vector load.

    Recorded by the load lowering (keyed by the scalar pointer expression)
    when the site addresses the vectorized grid axis at its stride-1 dim with
    the plain per-element index but sits inside loops nested in the V-loop,
    where the ordinary hoist above the V-loop cannot reach it.
    """

    vec_lane_var: str
    tensor_name: str
    vec_width: int
    dtype: torch.dtype
    # The site's cache-policy marker, applied to the vector form of the load.
    eviction_suffix: str
    # Every V-aligned chunk whose first element the program reads lies
    # inside the tensor (its extent along the lane dim is a multiple of V
    # and the chunk bases are V-aligned), so a chunk straddling a dynamic
    # tile extent may be loaded whenever its first lane is in bounds.
    whole_chunks_in_tensor: bool = False


@dataclasses.dataclass(frozen=True)
class CuteVloopStoreFact:
    """A scalar tile store that a sunk V-loop may turn into one vector store.

    The store-side twin of ``CuteVloopLoadFact``: recorded by the store
    lowering (keyed by the scalar pointer expression) when the site
    addresses the vectorized axis at its stride-1 dim with the plain
    per-element index inside loops nested in the V-loop.  The sink pass
    collects the V per-lane values and flushes them after the V-loop with
    one ``_cute_store_u32_vec`` / ``_cute_store_u16_vec``.
    """

    vec_lane_var: str
    tensor_name: str
    vec_width: int
    dtype: torch.dtype


# Stand-in codegen result for a node the tcgen05 fragment epilogue emits later
# (``Backend.deferred_node_codegen``); the committed store checks for it.
DEFERRED_TCGEN05_FRAGMENT_EPILOGUE = object()


def is_deferred_tcgen05_fragment_epilogue(value: object) -> bool:
    return value is DEFERRED_TCGEN05_FRAGMENT_EPILOGUE


class CuteDeviceFunctionState:
    """CuTe-owned state for one DeviceFunction codegen instance.

    The fields are grouped by the emitter family that owns them
    (``cute/state.py``): ``tcgen05`` (the GEMM pipeline), ``flash`` (the
    attention bodies), ``lanes`` (lane loops, thread axes, the launch shape)
    and ``root_plans`` (the whole-root plan slots); a few kernel-wide node
    facts stay flat.
    """

    def __init__(self) -> None:
        self.tcgen05 = CuteTcgen05State()
        self.flash = CuteFlashState()
        self.lanes = CuteLaneState()
        self.root_plans = CuteRootPlans()
        # Kernel-wide per-node lowering facts.
        self.explicit_rng_seed_exprs: set[str] = set()
        self.uniform_comparison_marker: str | None = None
        self.signed_byte_packets: dict[Node, SignedBytePacket] = {}
        # The load_eviction_policies slot each load node took
        # (``memory_ops._cute_load_eviction_slot``): a load lowered again (a
        # matmul operand re-lowered under a block remap) keeps its slot.
        self.load_eviction_slots: dict[Node, int] = {}
        # Merging / splitting reshapes lowered so far: True when each thread
        # holds the element at its own coordinates (proven by
        # ``cute_reshape._chain_keeps_thread_elements``), False when the
        # elements sit on other threads; only a store path that re-reads them
        # where they sit may store such a value, the generic store refuses
        # (``cute_reshape.is_cute_thread_moving_reshape_value``).
        self.reshape_elements_in_place: dict[Node, bool] = {}
        self.published_scalar_requests: list[PublishedScalarRequest] = []
        self.register_producer_requests: list[RegisterProducerRequest] = []

    # -- tcgen05 registries -----------------------------------------------------

    def register_tcgen05_fragment_epilogue_plan(
        self, plan: Tcgen05FragmentEpiloguePlan
    ) -> None:
        """Atomically commit one fully validated live-FX fragment plan."""
        if self.tcgen05.fragment_epilogue_plan is not None:
            raise exc.BackendUnsupported(
                "cute", "tcgen05 fragment epilogue plan must be unique"
            )
        self.tcgen05.fragment_epilogue_plan = plan

    def reject_tcgen05_fragment_epilogue_plan(
        self, anchor: Node, *, bm: int, bn: int, bk: int
    ) -> None:
        """Memoize a failed thread-locality proof for this config."""
        self.tcgen05.rejected_fragment_epilogue_plans.add((anchor, bm, bn, bk))

    def tcgen05_fragment_epilogue_plan_was_rejected(
        self, anchor: Node, *, bm: int, bn: int, bk: int
    ) -> bool:
        return (anchor, bm, bn, bk) in self.tcgen05.rejected_fragment_epilogue_plans

    @property
    def has_tcgen05_fragment_epilogue_plan(self) -> bool:
        return self.tcgen05.fragment_epilogue_plan is not None

    def tcgen05_fragment_epilogue_plan_for_anchor(
        self, anchor: Node
    ) -> Tcgen05FragmentEpiloguePlan | None:
        plan = self.tcgen05.fragment_epilogue_plan
        return plan if plan is not None and plan.anchor is anchor else None

    def tcgen05_fragment_epilogue_plan_for_store(
        self, store: Node | None
    ) -> Tcgen05FragmentEpiloguePlan | None:
        plan = self.tcgen05.fragment_epilogue_plan
        return plan if plan is not None and plan.store_node is store else None

    def is_deferred_tcgen05_fragment_epilogue_node(self, node: Node) -> bool:
        plan = self.tcgen05.fragment_epilogue_plan
        return plan is not None and node in plan.owned_nodes

    def veto_collective_lane_loop_suppression(self) -> None:
        self.lanes.collective_lane_loop_suppression_vetoed = True

    def collective_lane_loop_suppression_is_vetoed(self) -> bool:
        return self.lanes.collective_lane_loop_suppression_vetoed

    def register_tcgen05_store_value(
        self, name: str, value: CuteTcgen05StoreValue
    ) -> None:
        self.tcgen05.store_values[name] = value

    def register_tcgen05_grouped_tail_proof(
        self, proof: Tcgen05GroupedTailEpilogueMatch
    ) -> None:
        if proof.store_node in self.tcgen05.grouped_tail_proofs:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped tail proof must be unique per store node",
            )
        self.tcgen05.grouped_tail_proofs[proof.store_node] = proof

    def grouped_tail_proof_for_store(
        self, store_node: Node
    ) -> Tcgen05GroupedTailEpilogueMatch | None:
        return self.tcgen05.grouped_tail_proofs.get(store_node)

    def get_tcgen05_store_value(
        self,
        candidate_names: Sequence[str],
    ) -> CuteTcgen05StoreValue | None:
        store_values = self.tcgen05.store_values
        for candidate_name in candidate_names:
            if (value := store_values.get(candidate_name)) is not None:
                return value
        return None

    def claim_tcgen05_store_site(
        self, value: CuteTcgen05StoreValue, store_node: Node | None
    ) -> tuple[bool, bool]:
        """Return (is_secondary, is_final) from the complete same-graph fanout.

        Unknown fanout supports only one store. Multiple output stores must
        have a proved complete set in one FX graph so they share an execution
        scope. Releasing before its final store would let an independent MMA
        role reuse TMEM while a later epilogue still reads the old tile.
        """
        value_id = id(value)
        emitted = self.tcgen05.emitted_store_nodes.setdefault(value_id, set())
        expected = value.output_stores
        if expected is None:
            if emitted:
                raise exc.BackendUnsupported(
                    "cute", "tcgen05 fanout requires a complete output store set"
                )
            emitted.add(store_node)
            return False, True
        if (
            not expected
            or len({store.graph for store in expected}) != 1
            or store_node not in expected
            or store_node in emitted
        ):
            raise exc.BackendUnsupported(
                "cute", "tcgen05 fanout requires distinct stores in one FX graph"
            )
        is_secondary = bool(emitted)
        emitted.add(store_node)
        return is_secondary, len(emitted) == len(expected)

    def register_tcgen05_matmul_plan(self, plan: CuteTcgen05MatmulPlan) -> None:
        if self.tcgen05.matmul_plan is not None:
            if self.tcgen05.matmul_plan != plan:
                raise exc.BackendUnsupported(
                    "cute", "mixed tcgen05 matmul collective plans in one kernel"
                )
            return
        self.tcgen05.matmul_plan = plan

    def register_paired_fanout_store(
        self, store: FanoutStore, post_loop_stmts: Sequence[ast.AST]
    ) -> bool:
        """Keep the first marked body alive until its complete second store."""
        from .epilogue_fanout import combine_stores

        tcgen05 = self.tcgen05
        first, second = store.plan.stores
        if store.site is first:
            if first in tcgen05.pending_paired_stores:
                raise exc.BackendUnsupported("cute", "duplicate paired fanout store")
            tcgen05.pending_paired_stores[first] = store
            return False
        if store.site is not second or first not in tcgen05.pending_paired_stores:
            raise exc.BackendUnsupported("cute", "unordered paired fanout stores")
        combine_stores(tcgen05.pending_paired_stores.pop(first), store)
        tcgen05.per_tile_stmt_ids.remove(id(store.main))
        tcgen05.epi_role_stmt_ids.remove(id(store.main))
        tcgen05.post_loop_stmt_ids.difference_update(
            id(stmt) for stmt in post_loop_stmts
        )
        return True

    def register_tcgen05_sched_pipeline_plan(
        self, plan: _Tcgen05SchedPipelinePlan
    ) -> None:
        self.tcgen05.sched_pipeline_plan = plan

    def register_tcgen05_aux_pipeline_plan(self, plan: _Tcgen05AuxPipelinePlan) -> None:
        self.tcgen05.aux_pipeline_plan = plan

    def register_tcgen05_per_tile_stmts(self, stmts: list[ast.AST]) -> None:
        """Keep per-tile setup inside the persistent work-tile loop.

        The splitter hoists everything else to once-per-CTA setup, so
        statements that read per-tile coordinates or advance per-tile pipeline
        state must be marked here.
        """
        self.tcgen05.per_tile_stmt_ids.update(id(stmt) for stmt in stmts)

    def is_tcgen05_per_tile(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self.tcgen05.per_tile_stmt_ids

    @property
    def has_tcgen05_per_tile_marks(self) -> bool:
        return bool(self.tcgen05.per_tile_stmt_ids)

    def register_tcgen05_post_loop_stmts(self, stmts: Sequence[ast.AST]) -> None:
        """Move one-shot drains and teardown after the persistent tile loop."""
        self.tcgen05.post_loop_stmt_ids.update(id(stmt) for stmt in stmts)

    def is_tcgen05_post_loop(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self.tcgen05.post_loop_stmt_ids

    def move_tcgen05_post_loop_stmts_to_end(self, body: list[ast.AST]) -> list[ast.AST]:
        """Reorder ``body`` so post-loop-tagged statements run last.

        The persistent codegen path pulls post-loop cleanup out of the work-tile
        loop. The non-persistent (flat-grid) path leaves those statements where
        the store lowering emitted them. When a single accumulator fans out to
        multiple stores, the primary store's matmul drain / TMEM-free teardown is
        emitted inline before later store bodies; the teardown must instead run
        after every store has read the still-live accumulator TMEM. Moving the
        tagged statements to the end (preserving relative order) keeps the
        single-store case a no-op while fixing the fan-out ordering.
        """
        post_loop_stmt_ids = self.tcgen05.post_loop_stmt_ids
        if not post_loop_stmt_ids:
            return body
        remaining: list[ast.AST] = []
        post_loop: list[ast.AST] = []
        for stmt in body:
            if id(stmt) in post_loop_stmt_ids:
                post_loop.append(stmt)
            else:
                remaining.append(stmt)
        return [*remaining, *post_loop]

    @property
    def has_tcgen05_post_loop_marks(self) -> bool:
        return bool(self.tcgen05.post_loop_stmt_ids)

    def register_tcgen05_tma_load_role_stmts(self, stmts: list[ast.AST]) -> None:
        """Mark work owned by the TMA-load warp role.

        Top-level role statements must also be per-tile-marked so they survive
        invariant hoisting. Nested role statements are found by the role
        partitioner's one-level loop recursion.
        """
        self.tcgen05.tma_load_role_stmt_ids.update(id(stmt) for stmt in stmts)

    @property
    def tcgen05_tma_load_role_stmt_ids(self) -> frozenset[int]:
        return frozenset(self.tcgen05.tma_load_role_stmt_ids)

    def register_tcgen05_mma_exec_role_stmts(self, stmts: list[ast.AST]) -> None:
        """Mark AB consumer / UMMA / acc producer work for the MMA-exec warp."""
        self.tcgen05.mma_exec_role_stmt_ids.update(id(stmt) for stmt in stmts)

    @property
    def tcgen05_mma_exec_role_stmt_ids(self) -> frozenset[int]:
        return frozenset(self.tcgen05.mma_exec_role_stmt_ids)

    def register_tcgen05_epi_role_stmts(self, stmts: list[ast.AST]) -> None:
        """Mark acc consumer and TMEM-to-GMEM store work for epilogue warps."""
        self.tcgen05.epi_role_stmt_ids.update(id(stmt) for stmt in stmts)

    def register_tcgen05_epi_role_prelude_stmts(self, stmts: Sequence[ast.AST]) -> None:
        """Mark one-shot epilogue setup that must stay inside the epi role."""
        self.tcgen05.epi_role_prelude_stmt_ids.update(id(stmt) for stmt in stmts)

    def register_tcgen05_epi_role_full_edge_stmts(
        self, *, full_tile_stmts: list[ast.AST], edge_tile_stmts: list[ast.AST]
    ) -> None:
        """Mark scheduler-split epilogue work for full vs fringe tiles."""
        self.register_tcgen05_epi_role_stmts([*full_tile_stmts, *edge_tile_stmts])
        self.tcgen05.epi_role_full_tile_stmt_ids.update(
            id(stmt) for stmt in full_tile_stmts
        )
        self.tcgen05.epi_role_edge_tile_stmt_ids.update(
            id(stmt) for stmt in edge_tile_stmts
        )

    def register_tcgen05_epi_role_tile_counter(
        self, name: str, *, increment_per_tile: bool = True
    ) -> None:
        """Publish the role-local epilogue tile counter used by TMA stores."""
        tcgen05 = self.tcgen05
        if tcgen05.epi_role_tile_counter_var is None:
            tcgen05.epi_role_tile_counter_var = name
            tcgen05.epi_role_tile_counter_increment_per_tile = increment_per_tile
            return
        assert tcgen05.epi_role_tile_counter_var == name
        assert tcgen05.epi_role_tile_counter_increment_per_tile == increment_per_tile

    def is_tcgen05_epi_role(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self.tcgen05.epi_role_stmt_ids

    def is_tcgen05_epi_role_prelude(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self.tcgen05.epi_role_prelude_stmt_ids

    def register_tcgen05_tma_role_prelude_stmts(self, stmts: Sequence[ast.AST]) -> None:
        """Mark one-shot TMA-load role setup that must stay inside the role,
        ahead of its persistent loop and its cluster join."""
        self.tcgen05.tma_role_prelude_stmt_ids.update(id(stmt) for stmt in stmts)

    def is_tcgen05_tma_role_prelude(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self.tcgen05.tma_role_prelude_stmt_ids

    def is_tcgen05_epi_role_full_tile(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self.tcgen05.epi_role_full_tile_stmt_ids

    def is_tcgen05_epi_role_edge_tile(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self.tcgen05.epi_role_edge_tile_stmt_ids

    @property
    def has_tcgen05_epi_role_full_edge_split(self) -> bool:
        return bool(
            self.tcgen05.epi_role_full_tile_stmt_ids
            or self.tcgen05.epi_role_edge_tile_stmt_ids
        )

    @property
    def tcgen05_epi_role_stmt_ids(self) -> frozenset[int]:
        return frozenset(self.tcgen05.epi_role_stmt_ids)

    def register_collective_handled_load(self, load_node: Node) -> None:
        """Register a collective operand load for later codegen decisions
        (``CuteLaneState.collective_handled_loads``)."""
        self.lanes.collective_handled_loads.add(load_node)

    def is_collective_handled_load(self, load_node: Node) -> bool:
        return load_node in self.lanes.collective_handled_loads

    @property
    def collective_handled_loads(self) -> frozenset[Node]:
        return frozenset(self.lanes.collective_handled_loads)

    def finalize_tcgen05_paired_fanout_stores(self) -> None:
        if self.tcgen05.pending_paired_stores:
            raise exc.BackendUnsupported("cute", "incomplete paired fanout stores")

    def request_root_lane_loop_suppression(self) -> None:
        self.lanes.suppress_root_lane_loops = True

    def consume_root_lane_loop_suppression(self) -> bool:
        """Return and clear the one-shot root lane-loop suppression request."""
        suppress = self.lanes.suppress_root_lane_loops
        self.lanes.suppress_root_lane_loops = False
        return suppress
