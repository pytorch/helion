from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    import ast
    from collections.abc import Sequence

    import torch

    from .aux_tensor import Tcgen05AuxTensorDescriptor
    from .cute_mma import _Tcgen05AuxPipelinePlan
    from .cute_mma import _Tcgen05SchedPipelinePlan


@dataclasses.dataclass(frozen=True)
class CuteTcgen05StoreValue:
    bm: int = 0
    bn: int = 0
    bk: int = 0
    thr_mma: str = ""
    epi_warp_count: int = 0
    epi_acc_frag_base: str = ""
    epi_tidx: str = ""
    epi_active: str = ""
    exec_active: str = ""
    warp_idx: str = ""
    epi_tile: str = ""
    c_stage_count: int = 0
    epilog_sync_barrier_id: int = 0
    tmem_load_atom: str = ""
    epilogue_rest_mode: str = ""
    acc_pipeline: str = ""
    acc_producer_state: str = ""
    acc_consumer_state: str = ""
    tmem_alloc_barrier: str = ""
    tmem_allocator: str = ""
    tmem_holding_buf: str = ""
    tmem_dealloc_mbar_ptr: str = ""
    epi_acc_tmem_ptr: str = ""
    acc_tmem_cols: str = ""
    tma_warp: str = ""
    tma_pipeline: str = ""
    tma_producer_state: str = ""
    tma_store_atom: str = ""
    tma_store_tensor: str = ""
    role_local_tile_counter: str = ""
    is_two_cta: bool = False
    use_tma: bool = False
    use_role_local_epi: bool = False
    use_tma_store_epilogue: bool = False
    tma_store_full_tiles_only: bool = False
    ab_stage_count: int = 0
    acc_stage_count: int = 0
    skip_ab_producer_advance: bool = False
    # Output element dtype (cutlass type string, e.g. "cutlass.BFloat16")
    # used when computing the tcgen05 epilogue tile.
    epi_elem_dtype_str: str = ""


@dataclasses.dataclass(frozen=True)
class CuteTcgen05MatmulPlan:
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
    uses_cluster_m2_one_cta_role_local_bridge: bool
    cta_thread_count: int
    physical_m_threads: int
    acc_stage_count: int
    ab_stage_count: int
    c_stage_count: int
    epi_warp_count: int
    ab_load_warp_count: int = 1
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
    persistence_model: str = "static_persistent"
    cluster_n: int = 1
    l2_swizzle_size: int = 1
    tma_store_full_tiles_only: bool = False
    # Per-anchor auxiliary descriptors discovered by the forward FX walker. This
    # is store-fusion metadata, not a collective compatibility field:
    # two matmuls with identical collective parameters but different downstream
    # aux tensors must share one collective plan. compare=False is also required
    # because descriptor equality reaches tensor-valued fields whose ``==`` does
    # not produce a scalar bool suitable for dataclass equality.
    aux_tensor_descriptors: tuple[Tcgen05AuxTensorDescriptor, ...] = dataclasses.field(
        default=(), compare=False
    )

    @property
    def c_input_aux_tensor_descriptors(self) -> tuple[Tcgen05AuxTensorDescriptor, ...]:
        """Aux descriptors staged by the C-input warp.

        Exact-shape MxN aux tensors use the SMEM-ring producer. Broadcast
        row-vector aux tensors stay on the direct per-thread load path so
        they do not allocate a full-tile ring for a one-dimensional input.
        """
        return tuple(d for d in self.aux_tensor_descriptors if d.broadcast_axis is None)

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
    def role_warp_count(self) -> int:
        return (
            self.epi_warp_count
            + 1
            + self.ab_load_warp_count
            + self.scheduler_warp_count
            + self.c_input_warp_count
        )

    @property
    def launched_warp_count(self) -> int:
        # ``setmaxregister`` is warpgroup-uniform on Blackwell: all 4 warps in a
        # warpgroup must request a compatible register budget. Scheduler-backed
        # kernels therefore round the role count up to a full warpgroup. With
        # the default C-input count this pads 7 role warps to 8; with one
        # C-input warp, that warp occupies the former padding slot. MONOLITHIC
        # keeps its historical 6-warp launch because generated-code golden pins
        # and validated runtime behavior depend on that shape.
        if self.has_scheduler_warp:
            warpgroup = 4
            return (self.role_warp_count + warpgroup - 1) // warpgroup * warpgroup
        return self.role_warp_count

    @property
    def block_shape(self) -> tuple[int, int, int]:
        return (self.physical_m_threads, self.launched_warp_count, 1)


class CuteDeviceFunctionState:
    """CuTe-owned state for one DeviceFunction codegen instance."""

    def __init__(self) -> None:
        self._tcgen05_store_values: dict[str, CuteTcgen05StoreValue] = {}
        # FX matmul / hl.dot / addmm nodes lowered through tcgen05. The store
        # path uses this to recognize fused epilogue chains that must use the
        # tcgen05 store splice instead of falling through to SIMT store codegen.
        self.matmul_fx_nodes: set[torch.fx.Node] = set()
        # tcgen05 matmul anchor -> registered result var. Fused epilogue walks
        # from a store value back to the anchor and reuses the store value
        # registered under this result var, even when user-visible names were
        # renamed through casts or epilogue nodes.
        self.matmul_fx_node_result_vars: dict[torch.fx.Node, str] = {}
        self.matmul_plan: CuteTcgen05MatmulPlan | None = None
        # Variable-name containers allocated in cute_mma and consumed by
        # program_id / memory_ops role builders. They live here so CuTe pipeline
        # ownership does not leak into the generic DeviceFunction body.
        self.sched_pipeline_plan: _Tcgen05SchedPipelinePlan | None = None
        self.aux_pipeline_plan: _Tcgen05AuxPipelinePlan | None = None
        self._per_tile_stmt_ids: set[int] = set()
        self._post_loop_stmt_ids: set[int] = set()
        self._tma_load_role_stmt_ids: set[int] = set()
        self._mma_exec_role_stmt_ids: set[int] = set()
        self._epi_role_stmt_ids: set[int] = set()
        self.epi_role_tile_counter_var: str | None = None
        self.epi_role_tile_counter_increment_per_tile: bool = True
        self._collective_handled_loads: set[str] = set()
        self.cluster_shape: tuple[int, int, int] | None = None
        self.block_shape: tuple[int, int, int] | None = None
        self.suppress_root_lane_loops = False

    def register_tcgen05_store_value(
        self, name: str, value: CuteTcgen05StoreValue
    ) -> None:
        self._tcgen05_store_values[name] = value

    def get_tcgen05_store_value(
        self,
        candidate_names: Sequence[str],
    ) -> CuteTcgen05StoreValue | None:
        for candidate_name in candidate_names:
            if (value := self._tcgen05_store_values.get(candidate_name)) is not None:
                return value
        return None

    def register_tcgen05_matmul_plan(self, plan: CuteTcgen05MatmulPlan) -> None:
        if self.matmul_plan is not None:
            if self.matmul_plan != plan:
                raise exc.BackendUnsupported(
                    "cute", "mixed tcgen05 matmul collective plans in one kernel"
                )
            return
        self.matmul_plan = plan

    def register_tcgen05_sched_pipeline_plan(
        self, plan: _Tcgen05SchedPipelinePlan
    ) -> None:
        self.sched_pipeline_plan = plan

    def register_tcgen05_aux_pipeline_plan(self, plan: _Tcgen05AuxPipelinePlan) -> None:
        self.aux_pipeline_plan = plan

    def register_tcgen05_per_tile_stmts(self, stmts: list[ast.AST]) -> None:
        """Keep per-tile setup inside the persistent work-tile loop.

        The splitter hoists everything else to once-per-CTA setup, so
        statements that read per-tile coordinates or advance per-tile pipeline
        state must be marked here.
        """
        self._per_tile_stmt_ids.update(id(stmt) for stmt in stmts)

    def is_tcgen05_per_tile(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self._per_tile_stmt_ids

    @property
    def has_tcgen05_per_tile_marks(self) -> bool:
        return bool(self._per_tile_stmt_ids)

    def register_tcgen05_post_loop_stmts(self, stmts: list[ast.AST]) -> None:
        """Move one-shot drains and teardown after the persistent tile loop."""
        self._post_loop_stmt_ids.update(id(stmt) for stmt in stmts)

    def is_tcgen05_post_loop(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self._post_loop_stmt_ids

    @property
    def has_tcgen05_post_loop_marks(self) -> bool:
        return bool(self._post_loop_stmt_ids)

    def register_tcgen05_tma_load_role_stmts(self, stmts: list[ast.AST]) -> None:
        """Mark work owned by the TMA-load warp role.

        Top-level role statements must also be per-tile-marked so they survive
        invariant hoisting. Nested role statements are found by the role
        partitioner's one-level loop recursion.
        """
        self._tma_load_role_stmt_ids.update(id(stmt) for stmt in stmts)

    def is_tcgen05_tma_load_role(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self._tma_load_role_stmt_ids

    @property
    def has_tcgen05_tma_load_role_marks(self) -> bool:
        return bool(self._tma_load_role_stmt_ids)

    @property
    def tcgen05_tma_load_role_stmt_ids(self) -> frozenset[int]:
        return frozenset(self._tma_load_role_stmt_ids)

    def register_tcgen05_mma_exec_role_stmts(self, stmts: list[ast.AST]) -> None:
        """Mark AB consumer / UMMA / acc producer work for the MMA-exec warp."""
        self._mma_exec_role_stmt_ids.update(id(stmt) for stmt in stmts)

    def is_tcgen05_mma_exec_role(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self._mma_exec_role_stmt_ids

    @property
    def has_tcgen05_mma_exec_role_marks(self) -> bool:
        return bool(self._mma_exec_role_stmt_ids)

    @property
    def tcgen05_mma_exec_role_stmt_ids(self) -> frozenset[int]:
        return frozenset(self._mma_exec_role_stmt_ids)

    def register_tcgen05_epi_role_stmts(self, stmts: list[ast.AST]) -> None:
        """Mark acc consumer and TMEM-to-GMEM store work for epilogue warps."""
        self._epi_role_stmt_ids.update(id(stmt) for stmt in stmts)

    def register_tcgen05_epi_role_tile_counter(
        self, name: str, *, increment_per_tile: bool = True
    ) -> None:
        """Publish the role-local epilogue tile counter used by TMA stores."""
        if self.epi_role_tile_counter_var is None:
            self.epi_role_tile_counter_var = name
            self.epi_role_tile_counter_increment_per_tile = increment_per_tile
            return
        assert self.epi_role_tile_counter_var == name
        assert self.epi_role_tile_counter_increment_per_tile == increment_per_tile

    def is_tcgen05_epi_role(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self._epi_role_stmt_ids

    @property
    def has_tcgen05_epi_role_marks(self) -> bool:
        return bool(self._epi_role_stmt_ids)

    @property
    def tcgen05_epi_role_stmt_ids(self) -> frozenset[int]:
        return frozenset(self._epi_role_stmt_ids)

    def register_collective_handled_load(self, name: str) -> None:
        self._collective_handled_loads.add(name)

    def is_collective_handled_load(self, name: str) -> bool:
        return name in self._collective_handled_loads

    def request_root_lane_loop_suppression(self) -> None:
        self.suppress_root_lane_loops = True

    def consume_root_lane_loop_suppression(self) -> bool:
        """Return and clear the one-shot root lane-loop suppression request."""
        suppress = self.suppress_root_lane_loops
        self.suppress_root_lane_loops = False
        return suppress
