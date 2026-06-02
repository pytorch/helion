from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import torch
    from torch.fx.node import Node

    from ..tile_strategy import DeviceLoopState
    from .aux_tensor import Tcgen05AuxTensorDescriptor
    from .cute_mma import _Tcgen05AuxPipelinePlan
    from .cute_mma import _Tcgen05SchedPipelinePlan
    from .tcgen05_lifecycle import Tcgen05LifecycleContext
    from .tcgen05_pure_matmul import Tcgen05PureClcSchedulerObject
    from .tcgen05_pure_matmul import Tcgen05PureMatmulObjectModel


@dataclasses.dataclass(frozen=True)
class CuteTcgen05StoreValue:
    lifecycle_context: Tcgen05LifecycleContext
    pure_matmul_object: Tcgen05PureMatmulObjectModel | None = None
    bm: int = 0
    bn: int = 0
    bk: int = 0
    thr_mma: str = ""
    epi_warp_count: int = 0
    epi_acc_frag_base: str = ""
    epi_tidx: str = ""
    warp_idx: str = ""
    epi_tile: str = ""
    c_stage_count: int = 0
    epilog_sync_barrier_id: int = 0
    tmem_load_atom: str = ""
    epilogue_rest_mode: str = ""
    tma_store_atom: str = ""
    tma_store_tensor: str = ""
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

    def __post_init__(self) -> None:
        if self.pure_matmul_object is not None:
            assert self.pure_matmul_object.lifecycle_context is self.lifecycle_context
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

    @property
    def has_explicit_epilogue_tile(self) -> bool:
        return self.explicit_epi_tile_m is not None

    @property
    def pure_matmul_role_lifecycle(self) -> bool:
        return self.pure_matmul_object is not None


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
    flat_role_launch_warp_count: int | None = None
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
        if self.flat_role_launch_warp_count is not None:
            assert self.flat_role_launch_warp_count >= self.role_warp_count
            return (self.physical_m_threads * self.flat_role_launch_warp_count, 1, 1)
        return (self.physical_m_threads, self.launched_warp_count, 1)


class CuteDeviceFunctionState:
    """CuTe-owned state for one DeviceFunction codegen instance."""

    def __init__(self) -> None:
        self._tcgen05_store_values: dict[str, CuteTcgen05StoreValue] = {}
        self._tcgen05_consumed_store_value_ids: set[int] = set()
        # tcgen05 TMA-store atom/tensor kernel-arg names are allocated once per
        # matmul accumulator. When a single accumulator fans out to multiple
        # output stores (e.g. aux = pre-activation, out = gelu(pre)), each store
        # site must emit its own descriptor kernel params or the generated
        # function gets duplicate argument names. Track which StoreValue object
        # ids have already emitted their TMA-store kernel params so 2nd+ store
        # sites can allocate fresh per-store names.
        self._tcgen05_emitted_tma_store_value_ids: set[int] = set()
        # Snapshot of the accumulator consumer-state stage index, keyed by the
        # acc consumer-state variable name. A multi-store fan-out reads the same
        # accumulator TMEM stage; the primary store advances the consumer state
        # after its loop, so later stores must read the stage index captured
        # before that advance instead of the live (already-advanced) index.
        self._tcgen05_acc_stage_index_vars: dict[str, str] = {}
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
        self.pure_clc_scheduler_object: Tcgen05PureClcSchedulerObject | None = None
        self._per_tile_stmt_ids: set[int] = set()
        self._post_loop_stmt_ids: set[int] = set()
        self._tma_load_role_stmt_ids: set[int] = set()
        self._mma_exec_role_stmt_ids: set[int] = set()
        self._epi_role_stmt_ids: set[int] = set()
        self._epi_role_prelude_stmt_ids: set[int] = set()
        self._epi_role_full_tile_stmt_ids: set[int] = set()
        self._epi_role_edge_tile_stmt_ids: set[int] = set()
        self._tcgen05_kloop_owned_stmt_ids_by_loop: dict[int, set[int]] = {}
        self._tcgen05_kloop_cleanup_requested_loop_ids: set[int] = set()
        self._tcgen05_pure_lifecycle_pending_store_loops: dict[
            int, DeviceLoopState
        ] = {}
        self.epi_role_tile_counter_var: str | None = None
        self.epi_role_tile_counter_increment_per_tile: bool = True
        self._collective_handled_loads: set[str] = set()
        self._collective_handled_load_or_dependency_node_ids: set[int] = set()
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

    def consume_tcgen05_store_value(
        self,
        candidate_names: Sequence[str],
    ) -> CuteTcgen05StoreValue | None:
        for candidate_name in candidate_names:
            if (value := self._tcgen05_store_values.get(candidate_name)) is None:
                continue
            value_id = id(value)
            if value_id in self._tcgen05_consumed_store_value_ids:
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 pure role-lifecycle store supports exactly one "
                    f"store of {candidate_name!r}; multi-store fan-out must use "
                    "the standard store path",
                )
            self._tcgen05_consumed_store_value_ids.add(value_id)
            return value
        return None

    def get_or_create_tcgen05_acc_stage_index_var(
        self,
        acc_consumer_state: str,
        new_var: Callable[[str], str],
    ) -> tuple[str, bool]:
        """Return (snapshot_var, is_new) for an accumulator's stage index.

        The first (primary) store of an accumulator captures the consumer-state
        stage index into this variable before it advances the consumer state.
        Later fan-out stores reuse the snapshot so they read the same live
        accumulator TMEM stage. ``is_new`` is True only for the primary store,
        which must emit the ``<var> = <acc_consumer_state>.index`` assignment.
        """
        existing = self._tcgen05_acc_stage_index_vars.get(acc_consumer_state)
        if existing is not None:
            return existing, False
        snapshot = new_var("tcgen05_acc_stage_index")
        self._tcgen05_acc_stage_index_vars[acc_consumer_state] = snapshot
        return snapshot, True

    def tcgen05_tma_store_names_already_emitted(
        self, value: CuteTcgen05StoreValue
    ) -> bool:
        """Return whether this StoreValue already emitted TMA-store kernel params.

        The first store site that uses a given accumulator's StoreValue keeps the
        per-matmul ``tma_store_atom`` / ``tma_store_tensor`` names. Later store
        sites fanning out from the same accumulator must allocate fresh per-store
        names so the generated kernel signature has no duplicate parameters and so
        each store binds its own TMA descriptor.
        """
        value_id = id(value)
        already_emitted = value_id in self._tcgen05_emitted_tma_store_value_ids
        self._tcgen05_emitted_tma_store_value_ids.add(value_id)
        return already_emitted

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

    def register_tcgen05_pure_clc_scheduler_object(
        self, obj: Tcgen05PureClcSchedulerObject
    ) -> None:
        if self.pure_clc_scheduler_object is not None:
            raise exc.BackendUnsupported(
                "cute", "tcgen05 pure CLC scheduler object registered more than once"
            )
        self.pure_clc_scheduler_object = obj

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

    def register_tcgen05_post_loop_stmts(self, stmts: Sequence[ast.AST]) -> None:
        """Move one-shot drains and teardown after the persistent tile loop."""
        self._post_loop_stmt_ids.update(id(stmt) for stmt in stmts)

    def is_tcgen05_post_loop(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self._post_loop_stmt_ids

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
        if not self._post_loop_stmt_ids:
            return body
        remaining: list[ast.AST] = []
        post_loop: list[ast.AST] = []
        for stmt in body:
            if id(stmt) in self._post_loop_stmt_ids:
                post_loop.append(stmt)
            else:
                remaining.append(stmt)
        return [*remaining, *post_loop]

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

    def register_tcgen05_epi_role_prelude_stmts(self, stmts: Sequence[ast.AST]) -> None:
        """Mark one-shot epilogue setup that must stay inside the epi role."""
        self._epi_role_prelude_stmt_ids.update(id(stmt) for stmt in stmts)

    def register_tcgen05_epi_role_full_edge_stmts(
        self, *, full_tile_stmts: list[ast.AST], edge_tile_stmts: list[ast.AST]
    ) -> None:
        """Mark scheduler-split epilogue work for full vs fringe tiles."""
        self.register_tcgen05_epi_role_stmts([*full_tile_stmts, *edge_tile_stmts])
        self._epi_role_full_tile_stmt_ids.update(id(stmt) for stmt in full_tile_stmts)
        self._epi_role_edge_tile_stmt_ids.update(id(stmt) for stmt in edge_tile_stmts)

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

    def is_tcgen05_epi_role_prelude(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self._epi_role_prelude_stmt_ids

    def is_tcgen05_epi_role_full_tile(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self._epi_role_full_tile_stmt_ids

    def is_tcgen05_epi_role_edge_tile(self, stmt: ast.stmt) -> bool:
        return id(stmt) in self._epi_role_edge_tile_stmt_ids

    @property
    def has_tcgen05_epi_role_marks(self) -> bool:
        return bool(self._epi_role_stmt_ids)

    @property
    def has_tcgen05_epi_role_full_edge_split(self) -> bool:
        return bool(
            self._epi_role_full_tile_stmt_ids or self._epi_role_edge_tile_stmt_ids
        )

    @property
    def tcgen05_epi_role_stmt_ids(self) -> frozenset[int]:
        return frozenset(self._epi_role_stmt_ids)

    def register_collective_handled_load(
        self,
        name: str,
        *,
        dependency_nodes: Sequence[Node] = (),
    ) -> None:
        """Register collective operand-load state for later codegen decisions.

        Load names drive regular load suppression. FX node object identities
        drive statement-ownership marking for load/dependency scaffolding; this
        is scoped to one codegen pass where the FX graph and AST lists retain
        the same objects.
        """
        self._collective_handled_loads.add(name)
        self._collective_handled_load_or_dependency_node_ids.update(
            id(node) for node in dependency_nodes
        )

    def is_collective_handled_load(self, name: str) -> bool:
        return name in self._collective_handled_loads

    def is_collective_handled_load_or_dependency_node(self, node: Node) -> bool:
        return id(node) in self._collective_handled_load_or_dependency_node_ids

    def register_tcgen05_kloop_owned_stmts(
        self, device_loop: DeviceLoopState, stmts: Sequence[ast.AST]
    ) -> None:
        """Record exact K-loop statements emitted by tcgen05 matmul lowering.

        Future role-lifecycle cleanup must remove only statements registered by
        object identity here. This deliberately does not infer ownership from
        variable names or statement shapes.
        """
        if not stmts:
            return
        owned_ids = self._tcgen05_kloop_owned_stmt_ids_by_loop.setdefault(
            id(device_loop), set()
        )
        owned_ids.update(id(stmt) for stmt in stmts)

    def register_tcgen05_pure_lifecycle_pending_store(
        self, device_loop: DeviceLoopState
    ) -> None:
        loop_id = id(device_loop)
        if loop_id in self._tcgen05_pure_lifecycle_pending_store_loops:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 pure role-lifecycle supports only one pending "
                "matmul/store pair per K loop",
            )
        self._tcgen05_pure_lifecycle_pending_store_loops[loop_id] = device_loop

    def request_tcgen05_owned_kloop_cleanup(self, device_loop: DeviceLoopState) -> None:
        loop_id = id(device_loop)
        if loop_id not in self._tcgen05_pure_lifecycle_pending_store_loops:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 role-lifecycle K-loop cleanup was requested without a "
                "matching pending pure-matmul store",
            )
        self._tcgen05_pure_lifecycle_pending_store_loops.pop(loop_id)
        self._tcgen05_kloop_cleanup_requested_loop_ids.add(loop_id)

    def is_tcgen05_kloop_owned_stmt(
        self, device_loop: DeviceLoopState, stmt: ast.AST
    ) -> bool:
        return id(stmt) in self._tcgen05_kloop_owned_stmt_ids_by_loop.get(
            id(device_loop), set()
        )

    def tcgen05_unowned_kloop_stmts(
        self, device_loop: DeviceLoopState, stmts: Sequence[ast.AST]
    ) -> list[ast.AST]:
        owned_ids = self._tcgen05_kloop_owned_stmt_ids_by_loop.get(
            id(device_loop), set()
        )
        return [stmt for stmt in stmts if id(stmt) not in owned_ids]

    def replace_tcgen05_owned_kloop_stmts_with_pass(
        self, device_loop: DeviceLoopState, stmts: Sequence[ast.AST]
    ) -> None:
        """Fail closed unless the requested cleanup slice is tcgen05-owned.

        Real K-loop bodies may contain prelude/scaffold statements before the
        tcgen05 matmul-owned region. Consumers must pass the exact contiguous
        region they intend to remove so unrelated prelude/suffix statements are
        preserved and never classified by name or statement shape.
        """
        if not stmts:
            return
        inner_stmts = device_loop.inner_statements
        slice_start = next(
            (
                start
                for start in range(len(inner_stmts) - len(stmts) + 1)
                if all(
                    inner_stmts[start + index] is stmt
                    for index, stmt in enumerate(stmts)
                )
            ),
            None,
        )
        if slice_start is None:
            first_stmt = ast.unparse(stmts[0])
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 role-lifecycle K-loop cleanup requires an exact "
                f"contiguous statement slice; first requested statement: {first_stmt}",
            )
        unowned_stmts = self.tcgen05_unowned_kloop_stmts(device_loop, stmts)
        if unowned_stmts:
            first_unowned = ast.unparse(unowned_stmts[0])
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 role-lifecycle K-loop cleanup requires exact ownership "
                f"of every cleanup-region statement; first unowned statement: "
                f"{first_unowned}",
            )
        inner_stmts[slice_start : slice_start + len(stmts)] = [ast.Pass()]

    def finalize_tcgen05_owned_kloop_cleanup(
        self, device_loop: DeviceLoopState
    ) -> None:
        loop_id = id(device_loop)
        if loop_id not in self._tcgen05_kloop_cleanup_requested_loop_ids:
            return

        inner_stmts = device_loop.inner_statements
        owned_ids = self._tcgen05_kloop_owned_stmt_ids_by_loop.get(loop_id, set())
        owned_positions = [
            index for index, stmt in enumerate(inner_stmts) if id(stmt) in owned_ids
        ]
        if not owned_positions:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 role-lifecycle K-loop cleanup found no exactly owned "
                "statements to consume",
            )
        cleanup_stmts = inner_stmts[min(owned_positions) :]
        self.replace_tcgen05_owned_kloop_stmts_with_pass(device_loop, cleanup_stmts)
        self._tcgen05_kloop_cleanup_requested_loop_ids.remove(loop_id)

    def consume_tcgen05_owned_kloop_cleanup(self, device_loop: DeviceLoopState) -> None:
        self.request_tcgen05_owned_kloop_cleanup(device_loop)
        self.finalize_tcgen05_owned_kloop_cleanup(device_loop)

    def finalize_tcgen05_pure_lifecycle_stores(self) -> None:
        if not self._tcgen05_pure_lifecycle_pending_store_loops:
            return
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 pure role-lifecycle requires exactly one store consuming "
            "the matmul result before function codegen finishes",
        )

    def request_root_lane_loop_suppression(self) -> None:
        self.suppress_root_lane_loops = True

    def consume_root_lane_loop_suppression(self) -> bool:
        """Return and clear the one-shot root lane-loop suppression request."""
        suppress = self.suppress_root_lane_loops
        self.suppress_root_lane_loops = False
        return suppress
