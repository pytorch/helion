from __future__ import annotations

import ast
import dataclasses
import textwrap
from typing import TYPE_CHECKING

from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from .cutedsl_compat import emit_pipeline_advance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..tile_strategy import DeviceLoopState
    from .cute_mma import _Tcgen05SchedPipelinePlan
    from .device_state import CuteDeviceFunctionState
    from .tcgen05_lifecycle import Tcgen05LifecycleContext


@dataclasses.dataclass(frozen=True)
class Tcgen05TmaStoreTailParams:
    late_later_subtile_acquire: str
    epilog_sync_barrier: str
    c_buffer: str
    c_buffer_expr: str
    c_stage_count: int
    tiled_copy_r2s: str
    trs_rd: str
    trs_sd: str
    warp_idx: str
    tma_store_atom: str
    bsg_sd: str
    bsg_gd: str
    c_pipeline: str


@dataclasses.dataclass(frozen=True)
class Tcgen05TmaStorePipelineParams:
    c_pipeline: str
    warp_idx: str


@dataclasses.dataclass(frozen=True)
class Tcgen05TmaStoreSubtileLoopParams:
    subtile_count: str
    epi_active: str
    first_subtile_acquire: str
    later_subtile_acquire: str
    acc_t2r_region_body: str
    tail: Tcgen05TmaStoreTailParams


@dataclasses.dataclass(frozen=True)
class Tcgen05TmaStoreBodyCoreParams:
    setup_lines: Sequence[str]
    subtile_loop: Tcgen05TmaStoreSubtileLoopParams
    pipeline_tail_lines: Sequence[str]


@dataclasses.dataclass(frozen=True)
class Tcgen05ClcQueryParams:
    leader_predicate: str
    clc_mbar_smem_ptr: str
    clc_response_smem_ptr: str
    clc_mbar_phase: str
    cluster_m: int
    bidx: str
    bidy: str
    bidz: str
    valid: str
    scheduler: str
    next_work_tile: str
    cluster_bidx: str
    cluster_bidy: str
    cluster_bidz: str


@dataclasses.dataclass(frozen=True)
class Tcgen05WorkTilePublishParams:
    cluster_m: int
    sched_pipeline: str
    sched_producer_state: str
    producer_barrier_state: str
    producer_smem_ptr: str
    leader_predicate: str
    sched_barrier_ptr: str
    sched_peer_rank: str
    sched_peer_m: str
    cluster_bidx: str
    cluster_bidy: str
    cluster_bidz: str
    valid: str
    local_slots: tuple[str, str, str, str]


@dataclasses.dataclass(frozen=True)
class Tcgen05WorkTileConsumeParams:
    sched_pipeline: str
    sched_consumer_state: str
    work_tile_smem: str
    valid: str
    cluster_bidx: str
    cluster_bidy: str
    cluster_bidz: str
    work_tile_stage_index: str | None


@dataclasses.dataclass(frozen=True)
class Tcgen05InitialWorkTileParams:
    scheduler_params: str
    scheduler_params_args: str
    scheduler: str
    work_tile: str
    cluster_bidx: str
    cluster_bidy: str
    cluster_bidz: str
    valid: str
    cluster_m: int


@dataclasses.dataclass(frozen=True)
class Tcgen05PureClcSchedulerObject:
    """Target1 pure-matmul CLC scheduler boundary.

    MMA lowering owns creation and program-id lowering consumes it for CLC
    scheduler role selection, work-tile publication, and query/cancellation
    emission. Subsequent slices can move more of the scheduler lifecycle behind
    the same boundary without changing normal role-local scheduling.
    """

    sched_plan: _Tcgen05SchedPipelinePlan
    scheduler_warp_id: int

    def scheduler_role_predicate(self) -> str:
        return (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) "
            f"== cutlass.Int32({self.scheduler_warp_id})"
        )

    def build_clc_query_block(self, params: Tcgen05ClcQueryParams) -> list[ast.stmt]:
        """Build one Target1 pure CLC try-cancel/query/decode block."""
        block: list[ast.stmt] = [
            statement_from_string("cute.arch.sync_warp()"),
            create(
                ast.If,
                test=expr_from_string(params.leader_predicate),
                body=[
                    statement_from_string(
                        "cute.arch.mbarrier_arrive_and_expect_tx("
                        f"{params.clc_mbar_smem_ptr}, 16)"
                    ),
                    statement_from_string(
                        "_cute_issue_clc_query_nomulticast("
                        f"{params.clc_mbar_smem_ptr}, "
                        f"{params.clc_response_smem_ptr})"
                    ),
                ],
                orelse=[],
            ),
            statement_from_string("cute.arch.sync_warp()"),
            statement_from_string(
                f"cute.arch.mbarrier_wait({params.clc_mbar_smem_ptr}, "
                f"{params.clc_mbar_phase})"
            ),
            statement_from_string(
                f"{params.clc_mbar_phase} = {params.clc_mbar_phase} ^ cutlass.Int32(1)"
            ),
            statement_from_string(
                f"({params.bidx}, {params.bidy}, {params.bidz}, {params.valid}) = "
                f"cute.arch.clc_response({params.clc_response_smem_ptr})"
            ),
            statement_from_string("cute.arch.fence_view_async_shared()"),
            statement_from_string(
                f"{params.scheduler}._current_work_linear_idx = "
                f"{params.bidx} // cutlass.Int32({params.cluster_m})"
            ),
            statement_from_string(
                f"{params.next_work_tile} = {params.scheduler}.get_current_work()"
            ),
            statement_from_string(
                f"{params.cluster_bidx} = {params.next_work_tile}.tile_idx[0]"
            ),
            statement_from_string(
                f"{params.cluster_bidy} = {params.next_work_tile}.tile_idx[1]"
            ),
            statement_from_string(
                f"{params.cluster_bidz} = {params.next_work_tile}.tile_idx[2]"
            ),
        ]
        return block

    def build_initial_work_tile_block(
        self, params: Tcgen05InitialWorkTileParams
    ) -> list[ast.stmt]:
        """Build the Quack-shaped CLC initial-work decode.

        CLC launch grids are x-linear: ``block_idx.x`` enumerates CTAs and
        ``block_idx_in_cluster`` identifies the CTA rank inside the cluster.
        The scheduler leader therefore seeds work from ``block_idx.x //
        cluster_m`` instead of the static path's z-indexed persistent decode.
        """
        initial_work_idx = (
            f"cute.arch.block_idx()[0] // cutlass.Int32({params.cluster_m})"
        )
        return [
            statement_from_string(
                f"{params.scheduler_params} = "
                "cutlass.utils.PersistentTileSchedulerParams("
                f"{params.scheduler_params_args})"
            ),
            statement_from_string(
                f"{params.scheduler} = "
                "cutlass.utils.StaticPersistentTileScheduler.create("
                f"{params.scheduler_params}, cute.arch.block_idx(), "
                "cute.arch.grid_dim())"
            ),
            statement_from_string(
                f"{params.scheduler}._current_work_linear_idx = {initial_work_idx}"
            ),
            statement_from_string(
                f"{params.work_tile} = {params.scheduler}.get_current_work()"
            ),
            statement_from_string(
                f"{params.cluster_bidx} = {params.work_tile}.tile_idx[0]"
            ),
            statement_from_string(
                f"{params.cluster_bidy} = {params.work_tile}.tile_idx[1]"
            ),
            statement_from_string(
                f"{params.cluster_bidz} = {params.work_tile}.tile_idx[2]"
            ),
            statement_from_string(
                f"{params.valid} = cutlass.Int32(1) "
                f"if {params.work_tile}.is_valid_tile else cutlass.Int32(0)"
            ),
        ]

    def build_work_tile_publish_block(
        self, params: Tcgen05WorkTilePublishParams
    ) -> list[ast.stmt]:
        """Build one scheduler work-tile mailbox publish block."""
        if params.cluster_m > 1:
            return [
                statement_from_string(
                    f"{params.sched_pipeline}.producer_acquire("
                    f"{params.sched_producer_state})"
                ),
                statement_from_string(
                    f"{params.sched_barrier_ptr} = "
                    f"{params.sched_pipeline}.producer_get_barrier("
                    f"{params.producer_barrier_state})"
                ),
                statement_from_string(
                    f"{params.sched_peer_rank} = cute.arch.lane_idx()"
                ),
                create(
                    ast.If,
                    test=expr_from_string(
                        f"{params.sched_peer_rank} < cutlass.Int32({params.cluster_m})"
                    ),
                    body=[
                        statement_from_string(
                            f"{params.sched_peer_m} = {params.sched_peer_rank}"
                        ),
                        statement_from_string(
                            "cute.arch.mbarrier_arrive_and_expect_tx("
                            f"{params.sched_barrier_ptr}, 16, "
                            f"{params.sched_peer_rank})"
                        ),
                        statement_from_string(
                            "_cute_store_shared_remote_x4("
                            f"{params.cluster_bidx} + {params.sched_peer_m}, "
                            f"{params.cluster_bidy}, "
                            f"{params.cluster_bidz}, "
                            f"{params.valid}, "
                            f"smem_ptr={params.producer_smem_ptr}, "
                            f"mbar_ptr={params.sched_barrier_ptr}, "
                            "peer_cta_rank_in_cluster="
                            f"{params.sched_peer_rank})"
                        ),
                    ],
                    orelse=[],
                ),
            ]
        return self._build_local_work_tile_publish_block(
            params,
            (
                params.cluster_bidx,
                params.cluster_bidy,
                params.cluster_bidz,
                params.valid,
            ),
        )

    def build_sentinel_publish_block(
        self, params: Tcgen05WorkTilePublishParams
    ) -> list[ast.stmt]:
        """Build the invalid work-tile publish that terminates consumers."""
        if params.cluster_m > 1:
            return [
                statement_from_string(
                    f"{params.sched_pipeline}.producer_acquire("
                    f"{params.sched_producer_state})"
                ),
                statement_from_string(
                    f"{params.sched_barrier_ptr} = "
                    f"{params.sched_pipeline}.producer_get_barrier("
                    f"{params.producer_barrier_state})"
                ),
                statement_from_string(
                    f"{params.sched_peer_rank} = cute.arch.lane_idx()"
                ),
                create(
                    ast.If,
                    test=expr_from_string(
                        f"{params.sched_peer_rank} < cutlass.Int32({params.cluster_m})"
                    ),
                    body=[
                        statement_from_string(
                            "cute.arch.mbarrier_arrive_and_expect_tx("
                            f"{params.sched_barrier_ptr}, 16, "
                            f"{params.sched_peer_rank})"
                        ),
                        statement_from_string(
                            "_cute_store_shared_remote_x4("
                            "cutlass.Int32(0), cutlass.Int32(0), "
                            "cutlass.Int32(0), cutlass.Int32(0), "
                            f"smem_ptr={params.producer_smem_ptr}, "
                            f"mbar_ptr={params.sched_barrier_ptr}, "
                            "peer_cta_rank_in_cluster="
                            f"{params.sched_peer_rank})"
                        ),
                    ],
                    orelse=[],
                ),
            ]
        return self._build_local_work_tile_publish_block(
            params,
            (
                "cutlass.Int32(0)",
                "cutlass.Int32(0)",
                "cutlass.Int32(0)",
                "cutlass.Int32(0)",
            ),
        )

    def build_next_work_tile_consume_block(
        self, params: Tcgen05WorkTileConsumeParams
    ) -> list[ast.stmt]:
        """Build consumer-side get_current_work for scheduler-published next tiles."""

        def slot(index: int) -> str:
            if params.work_tile_stage_index is None:
                return f"{params.work_tile_smem}[cutlass.Int32({index})]"
            return (
                f"{params.work_tile_smem}[cutlass.Int32({index}), "
                f"{params.work_tile_stage_index}]"
            )

        return [
            statement_from_string(
                f"{params.sched_pipeline}.consumer_wait({params.sched_consumer_state})"
            ),
            statement_from_string("cute.arch.fence_view_async_shared()"),
            statement_from_string("cute.arch.sync_warp()"),
            statement_from_string(f"{params.cluster_bidx} = {slot(0)}"),
            statement_from_string(f"{params.cluster_bidy} = {slot(1)}"),
            statement_from_string(f"{params.cluster_bidz} = {slot(2)}"),
            statement_from_string(
                f"{params.valid} = cutlass.Int32(1) "
                f"if {slot(3)} != cutlass.Int32(0) else cutlass.Int32(0)"
            ),
            create(
                ast.If,
                test=expr_from_string("cute.arch.lane_idx() == cutlass.Int32(0)"),
                body=[
                    statement_from_string(
                        f"{params.sched_pipeline}.consumer_release("
                        f"{params.sched_consumer_state})"
                    ),
                ],
                orelse=[],
            ),
            statement_from_string(emit_pipeline_advance(params.sched_consumer_state)),
            statement_from_string("cute.arch.sync_warp()"),
        ]

    def _build_local_work_tile_publish_block(
        self,
        params: Tcgen05WorkTilePublishParams,
        values: tuple[str, str, str, str],
    ) -> list[ast.stmt]:
        return [
            statement_from_string(
                f"{params.sched_pipeline}.producer_acquire("
                f"{params.sched_producer_state})"
            ),
            create(
                ast.If,
                test=expr_from_string(params.leader_predicate),
                body=[
                    statement_from_string(f"{slot} = {value}")
                    for slot, value in zip(params.local_slots, values, strict=True)
                ]
                + [
                    statement_from_string(
                        f"{params.sched_pipeline}.producer_commit("
                        f"{params.sched_producer_state})"
                    ),
                ],
                orelse=[],
            ),
        ]


@dataclasses.dataclass(frozen=True)
class Tcgen05PureDynamicSchedulerObject:
    """Cycle-16 H3 staged-only Target8 dynamic-persistent scheduler boundary.

    Parallels ``Tcgen05PureClcSchedulerObject`` for the
    ``DYNAMIC_PERSISTENT`` persistence model. The productive emission
    surface — an atomic-counter (``tile_count_semaphore`` /
    ``atomic_add_i32`` on a global counter) work-tile loop that avoids
    the CLC scheduler warp + 16-bit mailbox SMEM hand-off + per-tile
    ``mbarrier_arrive_and_expect_tx`` round-trip — is staged for a
    follow-up cycle (see ``cute_plan.md`` §6 Target 8 cycle-13 Deep
    Replan H3). Cycle 16 lands only the typed boundary so the
    persistence-model dispatch in
    ``cute_mma._new_tcgen05_sched_pipeline_plan`` already has a place to
    consult before the codegen path raises ``BackendUnsupported``. The
    builder methods deliberately raise ``BackendUnsupported`` rather
    than emit AST so a caller that wires this object too early fails
    loudly instead of silently emitting a wrong scheduler shape.

    The class is shaped (``sched_plan`` + ``scheduler_warp_id``) to mirror
    ``Tcgen05PureClcSchedulerObject`` so a future cycle can implement the
    builder methods without changing the call sites. The
    ``scheduler_warp_id`` field is retained for symmetry even though the
    productive dynamic-persistent path will not use a dedicated scheduler
    warp (the atomic-counter advance runs on the existing MMA / TMA
    producer warps); a follow-up cycle that lands the productive emission
    can re-interpret the field as a sentinel (``-1``) or drop it once
    the actual lowering shape is finalized.
    """

    sched_plan: _Tcgen05SchedPipelinePlan
    scheduler_warp_id: int

    def _staged_error_message(self, where: str) -> str:
        return (
            "tcgen05_pure_dynamic_scheduler_object=True is cycle-16 "
            f"infrastructure only; the productive {where} emission is "
            "staged for cycle 17 (see cute_plan.md §6 Target 8 cycle-16 "
            "H3 Option B landing)"
        )

    def scheduler_role_predicate(self) -> str:
        # Mirror the CLC scheduler object's role-predicate surface so
        # future callers that consult this from generic dispatch sites
        # have a parallel API; the dynamic-persistent emission has no
        # dedicated scheduler warp, so the staged placeholder pins the
        # predicate to ``False`` (no current warp ever matches it).
        return "cutlass.Int32(0) == cutlass.Int32(1)"

    def build_clc_query_block(  # pragma: no cover - staged-only
        self, params: Tcgen05ClcQueryParams
    ) -> list[ast.stmt]:
        from ... import exc

        raise exc.BackendUnsupported("cute", self._staged_error_message("CLC query"))

    def build_initial_work_tile_block(  # pragma: no cover - staged-only
        self, params: Tcgen05InitialWorkTileParams
    ) -> list[ast.stmt]:
        from ... import exc

        raise exc.BackendUnsupported(
            "cute", self._staged_error_message("initial work-tile")
        )

    def build_work_tile_publish_block(  # pragma: no cover - staged-only
        self, params: Tcgen05WorkTilePublishParams
    ) -> list[ast.stmt]:
        from ... import exc

        raise exc.BackendUnsupported(
            "cute", self._staged_error_message("work-tile publish")
        )

    def build_sentinel_publish_block(  # pragma: no cover - staged-only
        self, params: Tcgen05WorkTilePublishParams
    ) -> list[ast.stmt]:
        from ... import exc

        raise exc.BackendUnsupported(
            "cute", self._staged_error_message("sentinel publish")
        )

    def build_next_work_tile_consume_block(  # pragma: no cover - staged-only
        self, params: Tcgen05WorkTileConsumeParams
    ) -> list[ast.stmt]:
        from ... import exc

        raise exc.BackendUnsupported(
            "cute", self._staged_error_message("next work-tile consume")
        )


@dataclasses.dataclass(frozen=True)
class Tcgen05PureMatmulObjectModel:
    """Token bundling pure tcgen05 MMA ownership with store cleanup."""

    lifecycle_context: Tcgen05LifecycleContext
    cleanup_loop: DeviceLoopState

    def register_pending_store(self, cute_state: CuteDeviceFunctionState) -> None:
        cute_state.register_tcgen05_pure_lifecycle_pending_store(self.cleanup_loop)

    def emit_store_role_stmts(
        self,
        cute_state: CuteDeviceFunctionState,
        *,
        tma_store_hoisted_stmts: Sequence[ast.AST],
        store_body_core: Sequence[str],
    ) -> list[ast.AST]:
        """Emit the pure-matmul epilogue role body and role marks."""
        sync_before_stmt = statement_from_string("cute.arch.sync_threads()")
        sync_after_stmt = statement_from_string("cute.arch.sync_threads()")
        main_stmt = statement_from_string(
            "if True:\n" + textwrap.indent("\n".join(store_body_core), "    ")
        )
        cute_state.register_tcgen05_per_tile_stmts(
            [sync_before_stmt, main_stmt, sync_after_stmt]
        )
        cute_state.register_tcgen05_epi_role_stmts([main_stmt])
        return [
            *tma_store_hoisted_stmts,
            sync_before_stmt,
            main_stmt,
            sync_after_stmt,
        ]

    def render_tma_store_tail_region(
        self,
        params: Tcgen05TmaStoreTailParams,
    ) -> str:
        """Render the pure-matmul R2S-to-TMA store tail."""
        return (
            f"{params.late_later_subtile_acquire}"
            f"        {params.epilog_sync_barrier}.arrive_and_wait()\n"
            f"        {params.c_buffer} = ({params.c_buffer_expr}) % "
            f"cutlass.Int32({params.c_stage_count})\n"
            f"        cute.copy({params.tiled_copy_r2s}, {params.trs_rd}, "
            f"{params.trs_sd}[(None, None, None, {params.c_buffer})])\n"
            f"        cute.arch.fence_view_async_shared()\n"
            f"        {params.epilog_sync_barrier}.arrive_and_wait()\n"
            f"        if {params.warp_idx} == cutlass.Int32(0):\n"
            f"            cute.copy({params.tma_store_atom}, "
            f"{params.bsg_sd}[(None, {params.c_buffer})], "
            f"{params.bsg_gd}[(None, cutlass.Int32(_tcgen05_subtile))])\n"
            f"            {params.c_pipeline}.producer_commit()\n"
        )

    def render_c_store_pre_loop_acquire_lines(
        self,
        params: Tcgen05TmaStorePipelineParams,
        *,
        first_c_acquire_in_loop: bool,
    ) -> list[str]:
        if first_c_acquire_in_loop:
            return []
        return [
            (
                f"if {self.lifecycle_context.epi_active} and "
                f"{params.warp_idx} == cutlass.Int32(0):\n"
                f"    {params.c_pipeline}.producer_acquire()"
            )
        ]

    def render_c_store_loop_first_acquire(
        self,
        params: Tcgen05TmaStorePipelineParams,
        *,
        first_c_acquire_in_loop: bool,
    ) -> str:
        if not first_c_acquire_in_loop:
            return ""
        return (
            f"        if _tcgen05_subtile == 0 and "
            f"{params.warp_idx} == cutlass.Int32(0):\n"
            f"            {params.c_pipeline}.producer_acquire()\n"
        )

    def render_c_store_loop_later_acquire(
        self,
        params: Tcgen05TmaStorePipelineParams,
        *,
        later_c_acquire_before_barrier: bool,
    ) -> str:
        if later_c_acquire_before_barrier:
            return ""
        return (
            f"        if _tcgen05_subtile != 0 and "
            f"{params.warp_idx} == cutlass.Int32(0):\n"
            f"            {params.c_pipeline}.producer_acquire()\n"
        )

    def render_c_store_loop_late_later_acquire(
        self,
        params: Tcgen05TmaStorePipelineParams,
        *,
        later_c_acquire_before_barrier: bool,
    ) -> str:
        if not later_c_acquire_before_barrier:
            return ""
        return (
            f"        if _tcgen05_subtile != 0 and "
            f"{params.warp_idx} == cutlass.Int32(0):\n"
            f"            {params.c_pipeline}.producer_acquire()\n"
        )

    def render_c_store_pipeline_tail(
        self,
        params: Tcgen05TmaStorePipelineParams,
    ) -> str:
        return (
            f"if {params.warp_idx} == cutlass.Int32(0):\n"
            f"    {params.c_pipeline}.producer_tail()"
        )

    def render_acc_consumer_advance(self) -> str:
        return f"if {self.lifecycle_context.epi_active}:\n" + emit_pipeline_advance(
            self.lifecycle_context.acc_consumer_state,
            indent="    ",
        )

    def render_tma_store_subtile_loop(
        self,
        params: Tcgen05TmaStoreSubtileLoopParams,
    ) -> str:
        """Render the pure-matmul per-subtile C-store operation body."""
        return (
            f"for _tcgen05_subtile in cutlass.range({params.subtile_count}, unroll_full=True):\n"
            f"    if {params.epi_active}:\n"
            f"{params.first_subtile_acquire}"
            f"{params.later_subtile_acquire}"
            f"{params.acc_t2r_region_body}"
            f"{self.render_tma_store_tail_region(params.tail)}"
        )

    def build_tma_store_body_core(
        self,
        params: Tcgen05TmaStoreBodyCoreParams,
    ) -> list[str]:
        """Assemble the pure-matmul C-store body from object-owned operations."""
        return [
            *params.setup_lines,
            self.render_tma_store_subtile_loop(params.subtile_loop)
            + self.render_acc_consumer_advance(),
            *params.pipeline_tail_lines,
        ]

    def emit_store_post_loop_stmts(
        self,
        cute_state: CuteDeviceFunctionState,
        candidate_names: Sequence[str],
        *,
        tma_store_pipeline_tail: str = "",
    ) -> list[ast.stmt]:
        """Emit and consume pure-matmul store/TMEM cleanup ownership."""
        post_loop_stmts: list[ast.stmt] = [
            statement_from_string(line)
            for line in self.lifecycle_context.render_store_post_loop_lines(
                tma_store_pipeline_tail=tma_store_pipeline_tail
            )
        ]
        cute_state.register_tcgen05_post_loop_stmts(post_loop_stmts)
        self.consume_store(cute_state, candidate_names)
        return post_loop_stmts

    def consume_store(
        self,
        cute_state: CuteDeviceFunctionState,
        candidate_names: Sequence[str],
    ) -> None:
        store_value = cute_state.consume_tcgen05_store_value(candidate_names)
        assert store_value is not None
        assert store_value.pure_matmul_object is self
        cute_state.consume_tcgen05_owned_kloop_cleanup(self.cleanup_loop)
