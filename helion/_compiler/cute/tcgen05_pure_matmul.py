from __future__ import annotations

import dataclasses
import textwrap
from typing import TYPE_CHECKING

from ..ast_extension import statement_from_string
from .cutedsl_compat import emit_pipeline_advance

if TYPE_CHECKING:
    import ast
    from collections.abc import Sequence

    from ..tile_strategy import DeviceLoopState
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
