from __future__ import annotations

import dataclasses

from .cutedsl_compat import emit_producer_tail_tma_umma


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
    skip_ab_producer_advance: bool = False
    tmem_permit_released_early: bool = False
    # Plain single-CTA kernels (merged pipeline init, see
    # ``tcgen05_use_merged_pipeline_init`` in ``cute_mma``) use the short
    # teardown: TMEM is freed as soon as the epilogue warps and the MMA warp
    # meet on ``tmem_alloc_barrier`` (the MMA writes completed before the acc
    # barrier flipped; the tcgen05.ld reads completed before the epilogue's
    # last ``fence_view_async_tmem_load``), the TMA-store drain runs after the
    # dealloc so the two overlap, and there is no CTA-wide ``sync_threads``
    # (the two-CTA path never had one).
    use_merged_pipeline_init: bool = False
    # Warp that issued ``tcgen05.alloc`` (and therefore owns ``free``).
    tmem_allocator_warp: int = 0
    # One tile per CTA (``one_shot_role_scheduler``) on the plain path and on
    # the clustered two-CTA path: the full-tile TMA-store epilogue frees TMEM
    # right after issuing its last subtile's TMA store (its last
    # ``fence_view_async_tmem_load`` + acc release stay at the last subtile),
    # so the dealloc handshake (the ``tmem_alloc_barrier`` meet with the MMA
    # warp; for a CTA pair, the peer rendezvous inside ``free``) overlaps the
    # store drain instead of holding the last subtile's staging behind it,
    # and ``render_store_post_loop_lines`` leaves only the producer tails and
    # the store drain in the teardown.  ``cute_mma``
    # decides this once, for accumulators that reach a single store through
    # the normal C-store body, so the store lowering and the teardown cannot
    # disagree about who frees.  The permit is relinquished right after the
    # allocation on both paths.
    free_tmem_in_epilogue: bool = False
    # ``cute_pdl`` on a one-CTA kernel: release the dependent grid after the
    # MMA warp has issued its last MMA (the two-CTA path always does), so
    # the next kernel's launch and prologue overlap this kernel's drain.
    pdl_release: bool = False
    # Two-CTA row-union kernels (one tile per CTA at the catalogue shapes, so
    # the whole teardown is exposed): the store warp's ``cp.async.bulk.wait_group
    # 0`` is independent of the TMEM handshake, and ``free`` is a cross-CTA
    # rendezvous on the pair's dealloc mbarrier, so the drain is issued after
    # the dealloc and the two overlap (grouped_gemm s1: 3.58 -> 3.54 us per
    # call).  The dense two-CTA persistent kernels keep their validated order
    # (drain, then dealloc).
    drain_after_dealloc: bool = False
    # Two-CTA row-union kernels: the MMA warp owns the TMEM allocation
    # (``tcgen05.alloc``, relinquish, ``free``; ``tmem_allocator_warp`` is its
    # warp) and frees it right after its accumulator ``producer_tail``: that
    # tail waits for the epilogue's release of every accumulator stage, and
    # each release follows the epilogue's ``fence_view_async_tmem_load``, so
    # every ``tcgen05.ld`` has completed; the MMA writes completed before the
    # accumulator barriers flipped.  The epilogue warps never meet the MMA
    # warp again: they go straight from their last TMEM read to their stores
    # while the MMA warp, idle since its last commit, runs the cross-CTA
    # dealloc handshake (grouped_gemm s1, one tile per CTA: 3.33 -> 3.23 us
    # per call).  ``exec_acc_tmem_ptr`` is the pointer the MMA warp frees.
    mma_warp_frees_tmem: bool = False
    exec_acc_tmem_ptr: str = ""

    def _allocator_with_columns_line(self) -> str:
        return (
            f"{self.tmem_allocator} = cutlass.utils.TmemAllocator("
            f"{self.tmem_holding_buf}, "
            f"barrier_for_retrieve={self.tmem_alloc_barrier}, "
            f"allocator_warp_id={self.tmem_allocator_warp}, "
            f"is_two_cta={self.is_two_cta!s}, "
            "two_cta_tmem_dealloc_mbar_ptr="
            f"{self.tmem_dealloc_mbar_ptr}, "
            f"num_allocated_columns={self.acc_tmem_cols}"
            ", initialize_mbarrier=False)"
        )

    def render_epilogue_tmem_free_lines(self) -> list[str]:
        """TMEM handshake + dealloc issued by the one-tile epilogue.

        Executed by every epilogue thread after the last subtile's TMA store
        has been issued (its TMEM read was fenced before the acc release): the
        ``tmem_alloc_barrier`` meet with the MMA warp (which arrives once,
        after issuing its last commit) proves every MMA write and every
        ``tcgen05.ld`` of this CTA has completed.
        """
        assert self.free_tmem_in_epilogue
        return [
            f"{self.tmem_alloc_barrier}.arrive_and_wait()",
            self._allocator_with_columns_line(),
            f"{self.tmem_allocator}.free({self.epi_acc_tmem_ptr})",
        ]

    def render_store_post_loop_lines(
        self, *, tma_store_pipeline_tail: str = ""
    ) -> list[str]:
        """Return post-loop store/TMEM cleanup owned by the lifecycle context."""

        post_loop_lines: list[str] = []
        short_teardown = self.use_merged_pipeline_init and not self.is_two_cta
        # The store warp's ``cp.async.bulk.wait_group 0`` is independent of
        # the TMEM handshake, so the merged-init one-CTA teardown and the
        # two-CTA row-union teardown (``drain_after_dealloc``) issue it after
        # the dealloc and overlap the two drains; every other teardown keeps
        # its long-validated order (drain, then dealloc).
        drain_after_dealloc = short_teardown or (
            self.is_two_cta and self.drain_after_dealloc
        )
        if (
            tma_store_pipeline_tail
            and not drain_after_dealloc
            and not self.free_tmem_in_epilogue
        ):
            post_loop_lines.append(tma_store_pipeline_tail)
        if self.use_tma:
            post_loop_lines.append(
                f"if {self.tma_warp}:\n"
                + emit_producer_tail_tma_umma(
                    self.tma_pipeline,
                    self.tma_producer_state,
                    indent="    ",
                    skip_advances=self.skip_ab_producer_advance,
                )
            )
        if self.is_two_cta or self.pdl_release:
            # PDL parity with Quack/CUTLASS: after all MMAs are issued, hint
            # dependent kernels before this role starts the final acc drain.
            post_loop_lines.append(
                f"if {self.exec_active}:\n"
                "    cute.arch.griddepcontrol_launch_dependents()"
            )
        if self.mma_warp_frees_tmem:
            assert self.is_two_cta and not self.free_tmem_in_epilogue
            assert self.tmem_permit_released_early
            post_loop_lines.extend(
                [
                    (
                        f"if {self.exec_active}:\n"
                        f"    {self.acc_pipeline}.producer_tail({self.acc_producer_state})"
                    ),
                    self._allocator_with_columns_line(),
                    (
                        f"if {self.exec_active}:\n"
                        f"    {self.tmem_allocator}.free({self.exec_acc_tmem_ptr})"
                    ),
                ]
            )
            # The store drain was issued at the top unless it follows the
            # dealloc (``drain_after_dealloc``); never both.
            if tma_store_pipeline_tail and drain_after_dealloc:
                post_loop_lines.append(tma_store_pipeline_tail)
            return post_loop_lines
        post_loop_lines.extend(
            [
                (f"if {self.exec_active}:\n    {self.tmem_alloc_barrier}.arrive()"),
                (
                    f"if {self.exec_active}:\n"
                    f"    {self.acc_pipeline}.producer_tail({self.acc_producer_state})"
                ),
            ]
        )
        if self.free_tmem_in_epilogue:
            # The epilogue meets the MMA warp on ``tmem_alloc_barrier`` and
            # frees TMEM itself; only the store drain remains.
            if tma_store_pipeline_tail:
                post_loop_lines.append(tma_store_pipeline_tail)
            return post_loop_lines
        post_loop_lines.append(self._allocator_with_columns_line())
        if not self.is_two_cta and not short_teardown:
            # Keep the long-validated cluster_m=1 teardown of the grouped and
            # row-union families unchanged. The guarded CtaGroup.TWO path
            # follows Quack's dealloc sequence without this CTA sync: epi warps
            # synchronize through tmem_alloc_barrier before free.
            post_loop_lines.append("cute.arch.sync_threads()")
        if not self.tmem_permit_released_early:
            post_loop_lines.append(
                f"if {self.epi_active}:\n"
                f"    {self.tmem_allocator}.relinquish_alloc_permit()"
            )
        post_loop_lines.extend(
            [
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
        if tma_store_pipeline_tail and drain_after_dealloc:
            # ``cp.async.bulk.wait_group 0`` on the store warp: issued after the
            # dealloc so the bulk-store drain and the TMEM handshake overlap.
            post_loop_lines.append(tma_store_pipeline_tail)
        return post_loop_lines
