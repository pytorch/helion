from __future__ import annotations

import dataclasses

from .cutedsl_compat import emit_dealloc_mbarrier_initialized_kwarg
from .cutedsl_compat import emit_producer_tail_tma_umma
from .cutedsl_compat import emit_producer_tail_umma_async


@dataclasses.dataclass(frozen=True)
class Tcgen05LifecycleContext:
    """Owns tcgen05 pipeline/TMEM state shared by MMA and store lowering."""

    exec_active: str
    epi_active: str
    tma_warp: str
    tma_pipeline: str
    tma_producer_state: str
    acc_pipeline: str
    acc_producer_state: str
    acc_consumer_state: str
    tmem_alloc_barrier: str
    tmem_allocator: str
    tmem_holding_buf: str
    tmem_dealloc_mbar_ptr: str
    epi_acc_tmem_ptr: str
    acc_tmem_cols: str
    is_two_cta: bool
    use_tma: bool
    ab_stage_count: int
    acc_stage_count: int
    skip_ab_producer_advance: bool = False

    def render_store_post_loop_lines(
        self, *, tma_store_pipeline_tail: str = ""
    ) -> list[str]:
        """Return post-loop store/TMEM cleanup owned by the lifecycle context."""

        post_loop_lines: list[str] = []
        if tma_store_pipeline_tail:
            post_loop_lines.append(tma_store_pipeline_tail)
        if self.use_tma:
            post_loop_lines.append(
                f"if {self.tma_warp}:\n"
                + emit_producer_tail_tma_umma(
                    self.tma_pipeline,
                    self.tma_producer_state,
                    num_stages=self.ab_stage_count,
                    indent="    ",
                    skip_advances=self.skip_ab_producer_advance,
                )
            )
        if self.is_two_cta:
            # PDL parity with Quack/CUTLASS: after all MMAs are issued, hint
            # dependent kernels before this role starts the final acc drain.
            post_loop_lines.append(
                f"if {self.exec_active}:\n"
                "    cute.arch.griddepcontrol_launch_dependents()"
            )
        post_loop_lines.extend(
            [
                (f"if {self.exec_active}:\n    {self.tmem_alloc_barrier}.arrive()"),
                (
                    f"if {self.exec_active}:\n"
                    + emit_producer_tail_umma_async(
                        self.acc_pipeline,
                        self.acc_producer_state,
                        num_stages=self.acc_stage_count,
                        indent="    ",
                    )
                ),
                (
                    f"{self.tmem_allocator} = cutlass.utils.TmemAllocator("
                    f"{self.tmem_holding_buf}, "
                    f"barrier_for_retrieve={self.tmem_alloc_barrier}, "
                    "allocator_warp_id=0, "
                    f"is_two_cta={self.is_two_cta!s}, "
                    "two_cta_tmem_dealloc_mbar_ptr="
                    f"{self.tmem_dealloc_mbar_ptr}, "
                    f"num_allocated_columns={self.acc_tmem_cols}"
                    f"{emit_dealloc_mbarrier_initialized_kwarg()})"
                ),
            ]
        )
        if not self.is_two_cta:
            # Keep the long-validated cluster_m=1 teardown unchanged. The
            # guarded CtaGroup.TWO path follows Quack's dealloc sequence
            # without this CTA sync: epi warps synchronize through
            # tmem_alloc_barrier before free.
            post_loop_lines.append("cute.arch.sync_threads()")
        post_loop_lines.extend(
            [
                (
                    f"if {self.epi_active}:\n"
                    f"    {self.tmem_allocator}.relinquish_alloc_permit()"
                ),
                (
                    f"if {self.epi_active}:\n"
                    f"    {self.tmem_alloc_barrier}.arrive_and_wait()"
                ),
                (
                    f"if {self.epi_active}:\n"
                    f"    {self.tmem_allocator}.free({self.epi_acc_tmem_ptr})"
                ),
            ]
        )
        return post_loop_lines
