"""The aux GMEM->SMEM producer body of a tcgen05 residual epilogue.

A warp-specialized tcgen05 GEMM with a residual (``C``) input stages that
input's tiles through an SMEM ring (``cute_state.aux_pipeline_plan``) so the
epilogue warps read it from shared memory.  One warp fills the ring: the
C-input warp when the plan reserves one (``scheduler_pids``'s C-input
role-local while), otherwise the store warp from inside the epilogue role's
loop (the store-warp merge).  :func:`plan_aux_producer_body` partitions each
aux descriptor once (the GMEM view, the per-output-tile region, the
epi-tile subtiles, the TMA partition or the SIMT ``TiledCopy``) and
:meth:`AuxProducerBody.emit` is the per-tile body: the tile coordinates,
the partitions, then the subtile loop that acquires a ring stage, copies
every descriptor's subtile, fences and commits it.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from ...ast_extension import create
from ...ast_extension import expr_from_string
from ...ast_extension import statement_from_string
from ...compile_environment import CompileEnvironment
from ..cutedsl_compat import emit_pipeline_advance

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...device_function import DeviceFunction
    from ..cute_mma import _Tcgen05AuxPipelinePlan
    from ..device_state import CuteTcgen05MatmulPlan


@dataclasses.dataclass
class AuxProducerBody:
    """The per-tile aux staging of one output tile (see the module docstring).

    ``tile_m_expr`` / ``tile_n_expr`` are the output tile's coordinates in
    the producer's units; ``full_tile_expr`` tests that the tile lies inside
    the aux tensor (edge tiles take the direct-GMEM aux path and must commit
    no ring stage).
    """

    device_function: DeviceFunction
    use_tma_load: bool
    pipeline: str
    producer_state: str
    tma_barrier_var: str | None
    lane_idx_var: str
    tile_m_var: str
    tile_n_var: str
    tile_m_expr: str
    tile_n_expr: str
    full_tile_var: str
    full_tile_expr: str
    per_descriptor_setup_blocks: list[list[ast.stmt]]
    per_descriptor_subtile_blocks: list[list[str]]
    per_descriptor_grouped_names: list[str]

    def emit(self) -> list[ast.stmt]:
        """Emit the per-output-tile producer body.

        The body computes per-output-tile aux GMEM partitions,
        flat-divides each into epi-tile-sized subtiles, then
        loops over the subtile axis. Each subtile iteration
        acquires one SMEM ring stage, cooperative-copies the
        subtile of every descriptor into the stage, fences the
        SMEM proxy, commits the producer barrier, and advances
        the producer state. Iteration order matches the
        consumer's per-subtile loop in
        ``memory_ops._aux_subtile_load_source``.
        """
        lines: list[ast.stmt] = []
        lines.extend(
            [
                statement_from_string(f"{self.lane_idx_var} = cute.arch.lane_idx()"),
                statement_from_string(f"{self.tile_m_var} = {self.tile_m_expr}"),
                statement_from_string(f"{self.tile_n_var} = {self.tile_n_expr}"),
            ]
        )
        # Per-descriptor partition setup runs once per output
        # tile, before the subtile loop. The setup builds the
        # ``flat_divide(epi_tile) → group_modes(2, rank)``
        # tensor whose third mode is the subtile axis the
        # producer iterates against.
        for block in self.per_descriptor_setup_blocks:
            lines.extend(block)

        # Determine the subtile count from any descriptor's
        # grouped tensor (all descriptors share the same
        # subtile axis because they're all sliced from the
        # same ``(bm, bn)`` region with the same ``epi_tile``).
        # Use the first descriptor's grouped name — pulled
        # from ``self.per_descriptor_grouped_names`` so the
        # ``device_function.new_var`` namespace suffix
        # (if any) is honored.
        first_grouped = self.per_descriptor_grouped_names[0]
        subtile_count_var = self.device_function.new_var(
            "tcgen05_aux_producer_subtile_count"
        )
        lines.append(
            statement_from_string(
                f"{subtile_count_var} = cutlass.const_expr("
                f"cute.size({first_grouped}.shape, mode=[2]))"
            )
        )

        # Build the per-subtile loop body. Per iteration:
        # acquire one SMEM stage, copy every descriptor's
        # subtile into the stage, sync the warp + fence the
        # SMEM proxy so the consumer's
        # ``consumer_wait`` sees a fully populated stage,
        # commit, advance. Lane-uniform code throughout (every
        # lane runs the same per-iteration body; the cooperative
        # copy partitions inside).
        # Build the per-iteration body as a single
        # already-indented source string. Each entry in
        # ``inner_chunks`` is a top-level statement (or block)
        # carrying its own ``    `` indent for the surrounding
        # ``for ...:`` loop. ``emit_pipeline_advance`` may
        # return a multi-line ``if True: ...`` block on
        # cutedsl builds without the OpResultList fix; we
        # pass ``indent="    "`` so the whole block is
        # already indented for the loop body and no caller-
        # side reflow is needed (the prior single-line
        # ``\n.join(f"    {line}" ...)`` pattern only
        # indented the first line of the advance block,
        # under-indenting its body and causing a SyntaxError
        # on the fallback path).
        loop_indent = "    "
        inner_chunks: list[str] = []
        inner_chunks.append(
            f"{loop_indent}{self.pipeline}.producer_acquire({self.producer_state})"
        )
        if self.use_tma_load:
            assert self.tma_barrier_var is not None
            inner_chunks.append(
                f"{loop_indent}{self.tma_barrier_var} = "
                f"{self.pipeline}.producer_get_barrier("
                f"{self.producer_state})"
            )
        for block in self.per_descriptor_subtile_blocks:
            inner_chunks.extend(f"{loop_indent}{line}" for line in block)
        # TMA aux loads skip the SIMT warp sync/fence below: they are
        # ordered by the tx-counted PipelineTmaAsync barrier, and the
        # consumer fences the async-shared view after ``consumer_wait`` and
        # before generic SMEM reads.
        if not self.use_tma_load:
            # ``CopyUniversalOp`` issues regular ld+st pairs that
            # complete in program order per thread; ``sync_warp``
            # ensures all 32 lanes of the producer warp finish
            # their SMEM stores. ``fence_acq_rel_cta`` provides
            # cross-warp visibility through the CTA-scope generic
            # SMEM proxy — the consumer warps' generic SMEM reads
            # after their ``consumer_wait`` would otherwise be
            # free to bypass the producer's writes since the
            # AsyncThread ``mbarrier.arrive`` PTX emission has
            # relaxed memory semantics by default.
            inner_chunks.extend(
                [
                    f"{loop_indent}cute.arch.sync_warp()",
                    f"{loop_indent}cute.arch.fence_acq_rel_cta()",
                ]
            )
        inner_chunks.extend(
            [
                (
                    f"{loop_indent}{self.pipeline}.producer_commit("
                    f"{self.producer_state})"
                ),
                emit_pipeline_advance(self.producer_state, indent=loop_indent),
            ]
        )
        inner_body = "\n".join(inner_chunks)
        loop_src = (
            f"for _tcgen05_aux_subtile in cutlass.range("
            f"{subtile_count_var}, unroll_full=True):\n"
            f"{inner_body}"
        )
        lines.append(statement_from_string(loop_src))
        return lines

    def emit_full_tiles_only(self) -> list[ast.stmt]:
        """The producer body behind the full-tile test."""
        return [
            statement_from_string(f"{self.full_tile_var} = {self.full_tile_expr}"),
            create(
                ast.If,
                test=expr_from_string(self.full_tile_var),
                body=self.emit(),
                orelse=[],
            ),
        ]


def plan_aux_producer_body(
    device_function: DeviceFunction,
    plan: CuteTcgen05MatmulPlan,
    aux_pipeline_plan: _Tcgen05AuxPipelinePlan,
    *,
    m_offset_var: str,
    n_offset_var: str,
    has_post_l2_coords: bool,
    sched_coord_0: str,
    sched_coord_1: str,
    is_two_cta: bool,
    cluster_m: int,
    logical_m_coord_expr: Callable[[str], str],
) -> AuxProducerBody:
    """Partition every staged aux descriptor of ``plan`` for the producer warp.

    ``m_offset_var`` / ``n_offset_var`` are the plan's output offsets (the
    names the consumer roles' dependency chain defines).
    ``has_post_l2_coords`` says the role's dependency chain defines them for
    this role too (the production path); otherwise the tile coordinates come
    from the raw scheduler mailbox coordinates ``sched_coord_0`` /
    ``sched_coord_1`` (``logical_m_coord_expr`` adds the cluster bridge
    adjustment to the M coordinate on the one-CTA path).
    """
    c_input_aux_tensor_descriptors = plan.c_input_aux_tensor_descriptors
    aux_use_tma_load = aux_pipeline_plan.use_tma_load
    # M / N tile coords for the cooperative copy. Each CTA's
    # C-input warp loads only its own per-CTA portion of the
    # aux region. Under cluster_m=2 ``use_2cta_instrs`` the
    # per-CTA aux tile shape is ``(bm/2, bn)`` and the per-CTA
    # M tile coord is the global M tile (without the cluster
    # ``// 2`` reduction — each CTA in the cluster handles its
    # own row stripe). The N coord is shared across cluster
    # (cluster_n=1 is the only validated runtime).
    #
    # Critical correctness invariant: the consumer-side per-CTA
    # subtile count is
    # ``(bm_per_cta * bn) / (epi_m * epi_n)``; the producer's
    # subtile count must match or the mbar handshake deadlocks
    # once the producer wraps the stage count and the consumer
    # has already exited (cluster_m=2 yields producer=2× the
    # consumer count when the producer mistakenly uses the
    # cluster-level (bm, bn)).
    bm = plan.bm
    bn = plan.bn
    bm_per_cta = bm // cluster_m if is_two_cta else bm
    tile_m_var = device_function.new_var("tcgen05_aux_tile_m")
    tile_n_var = device_function.new_var("tcgen05_aux_tile_n")
    # ``peer_m`` is this CTA's rank along the M axis of the cluster:
    # ``block_idx_in_cluster() % cluster_m``. The modulo is load-bearing
    # here because consumer CTAs can span the N axis: for ``cluster_m=2 +
    # cluster_n=2 + use_2cta=True`` (a validated 4-CTA cluster shape, see
    # ``cute_mma.py:_TCGEN05_V_LEADER_PREDICATE``) ranks {0, 1, 2, 3}
    # have M peer ranks {0, 1, 0, 1}. Scheduler-warp broadcasts use
    # lane-rank branches restricted to ``peer_rank < cluster_m``, where
    # the raw lane rank is already the M peer rank.
    peer_m_expr = (
        f"(cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) "
        f"% cutlass.Int32({cluster_m}))"
    )
    if has_post_l2_coords:
        # These are logical matrix coordinates after the launch-order
        # and L2-grouping remaps, in the same units as the consumer.
        #
        # Note that under ``cluster_n=1 + l2_groupings=[1]``
        # the post-L2 expression ``pid_0 * cluster_m +
        # peer_m`` is numerically equal to the pre-cycle-2i
        # ``work_tile_smem[0]`` because (a) the L2 remap is
        # identity and (b) the scheduler publishes
        # ``tile_idx[0] + peer_m = pid_0 * cluster_m +
        # peer_m`` into each CTA's slot. Outside that
        # narrow case the two forms diverge: under
        # ``l2_grp=[g>1]`` because L2 remap is non-identity,
        # and under ``cluster_n=2`` because the raw
        # rank-in-cluster ≠ peer_m.
        m_source = f"({m_offset_var} // cutlass.Int32({bm}))"
        n_source = f"({n_offset_var} // cutlass.Int32({bn}))"
        if is_two_cta:
            tile_m_expr = f"({m_source}) * cutlass.Int32({cluster_m}) + {peer_m_expr}"
        else:
            tile_m_expr = m_source
        tile_n_expr = n_source
    else:
        # Raw scheduler coordinates are only valid for a decomposition
        # that does not emit the requested output offset names. The
        # standard NDTileStrategy always emits those names,
        # which runs for every real kernel binding
        # regardless of ``l2_groupings``. Purely defensive —
        # preserves the pre-cycle-2i correctness baseline
        # if a future strategy emits the role-local while
        # without those names. Under ``is_two_cta`` the
        # scheduler-published ``work_tile_smem[0]`` already
        # carries the per-CTA peer_m baked in (the
        # scheduler publish is
        # ``tile_idx[0] + peer_m`` per CTA); under non-2cta
        # ``_tcgen05_logical_m_coord_expr`` adds the bridge
        # adjustment when applicable. So no extra peer_m
        # is applied here.
        m_source = sched_coord_0
        n_source = sched_coord_1
        if is_two_cta:
            tile_m_expr = m_source
        else:
            tile_m_expr = logical_m_coord_expr(m_source)
        tile_n_expr = n_source

    # Cooperative-copy thread layout. Single C-input warp has
    # 32 lanes; lay them out row-major as (M_threads, N_threads)
    # = (4, 8) so each lane reads a contiguous N chunk
    # (innermost dim) and 8 lanes cover the N axis. The val
    # layout pulls 128 bits per copy atom — the largest power
    # of two that divides both ``bn * dtype_bits`` and 128. For
    # bf16 with bn=256 that's 128 bits = 8 elements.
    # ``make_tiled_copy_tv`` lifts the per-lane chunk to a
    # ``(thr_layout × val_layout)`` partition and ``cute.copy``
    # iterates the tile under the lane's get_slice(lane_idx).
    cute_lane_idx_var = device_function.new_var("tcgen05_aux_lane_idx")

    # Pull aux pipeline names from the plan. The plan is the
    # ``_Tcgen05AuxPipelinePlan`` dataclass; access by name
    # without importing the type to avoid a module cycle.
    aux_pipeline_name = aux_pipeline_plan.pipeline
    aux_producer_state_name = aux_pipeline_plan.producer_state
    aux_rings = aux_pipeline_plan.rings
    aux_epi_tile_var = aux_pipeline_plan.epi_tile_var
    aux_tma_barrier_var = (
        device_function.new_var("tcgen05_aux_tma_barrier") if aux_use_tma_load else None
    )

    aux_full_tile_var = device_function.new_var("tcgen05_aux_full_tile")
    aux_shape = c_input_aux_tensor_descriptors[0].host_tensor_val.shape
    assert len(aux_shape) == 2, (
        "C-input staged aux descriptors must be exact-shape rank-2 tensors"
    )
    aux_m_size = int(aux_shape[0])
    aux_n_size = int(aux_shape[1])
    if has_post_l2_coords:
        aux_m_start_expr = m_offset_var
        aux_n_start_expr = n_offset_var
    else:
        if is_two_cta:
            aux_m_tile_expr = f"({sched_coord_0} // cutlass.Int32({cluster_m}))"
        else:
            aux_m_tile_expr = sched_coord_0
        aux_m_start_expr = f"({aux_m_tile_expr}) * cutlass.Int32({bm})"
        aux_n_start_expr = f"{sched_coord_1} * cutlass.Int32({bn})"
    aux_full_tile_expr = (
        f"{aux_m_start_expr} + cutlass.Int32({bm}) "
        f"<= cutlass.Int32({aux_m_size}) and "
        f"{aux_n_start_expr} + cutlass.Int32({bn}) "
        f"<= cutlass.Int32({aux_n_size})"
    )

    env_backend = CompileEnvironment.current().backend

    # Per-descriptor partitioning that runs once per output tile:
    # builds the source 2-D GMEM tensor, slices the per-output-
    # tile ``(bm, bn)`` region, and flat-divides it into
    # epi-tile-sized subtiles. The subtile-loop body further
    # slices one subtile of GMEM and one stage of SMEM per
    # iteration. Each per-descriptor partition uses fresh AST
    # var names so multiple aux descriptors compose linearly.
    per_descriptor_setup_blocks: list[list[ast.stmt]] = []
    per_descriptor_subtile_blocks: list[list[str]] = []
    per_descriptor_grouped_names: list[str] = []
    for desc_idx, (desc, ring) in enumerate(
        zip(c_input_aux_tensor_descriptors, aux_rings, strict=True)  # type: ignore[arg-type]
    ):
        aux_tensor_name = device_function.tensor_arg(desc.host_tensor_val).name
        aux_dtype_str = env_backend.dtype_str(desc.host_tensor_val.dtype)
        dtype_bits = desc.host_tensor_val.dtype.itemsize * 8
        tma_atom = ring.tma_atom
        tma_tensor = ring.tma_tensor
        if aux_use_tma_load:
            assert tma_atom is not None
            assert tma_tensor is not None
            assert aux_tma_barrier_var is not None
        else:
            assert tma_atom is None
            assert tma_tensor is None
        gmem_aux_view_var = device_function.new_var(f"tcgen05_aux_gmem_view_{desc_idx}")
        gmem_aux_tile_var = device_function.new_var(f"tcgen05_aux_gmem_tile_{desc_idx}")
        gmem_aux_subtiles_var = device_function.new_var(
            f"tcgen05_aux_gmem_subtiles_{desc_idx}"
        )
        gmem_subtiles_grouped_var = device_function.new_var(
            f"tcgen05_aux_gmem_subtiles_grouped_{desc_idx}"
        )
        tiled_copy_var = device_function.new_var(f"tcgen05_aux_tiled_copy_{desc_idx}")
        thr_copy_var = device_function.new_var(f"tcgen05_aux_thr_copy_{desc_idx}")
        gmem_subtile_var = device_function.new_var(
            f"tcgen05_aux_gmem_subtile_{desc_idx}"
        )
        smem_stage_var = device_function.new_var(f"tcgen05_aux_smem_stage_{desc_idx}")
        gmem_part_var = device_function.new_var(f"tcgen05_aux_gmem_part_{desc_idx}")
        smem_part_var = device_function.new_var(f"tcgen05_aux_smem_part_{desc_idx}")
        tma_smem_part_var = ""
        tma_gmem_part_var = ""
        setup: list[ast.stmt] = []
        # Build the source 2-D GMEM tensor. Exact-shape rank-2
        # aux passes through ``aux_tensor`` directly; rank-1
        # trailing-axis broadcast aux builds a stride-0-on-M
        # view with M-extent ``bm_per_cta`` and N-extent = the
        # rank-1 size. Under cluster_m=2 ``use_2cta_instrs``
        # the per-CTA aux tile is ``(bm/2, bn)``; the global M
        # tile coord directly indexes per-CTA M stripes (the
        # scheduler publishes 2 global tiles per cluster
        # step). For non-2cta the per-CTA tile shape collapses
        # to the full ``(bm, bn)``.
        if desc.broadcast_axis is None:
            if aux_use_tma_load:
                gmem_source_var = tma_tensor
                assert gmem_source_var is not None
            else:
                gmem_source_var = aux_tensor_name
            setup.extend(
                [
                    statement_from_string(f"{gmem_aux_view_var} = {gmem_source_var}"),
                    statement_from_string(
                        f"{gmem_aux_tile_var} = cute.local_tile("
                        f"{gmem_aux_view_var}, ({bm_per_cta}, {bn}), "
                        f"({tile_m_var}, {tile_n_var}))"
                    ),
                ]
            )
        else:
            assert desc.broadcast_axis == 1, (
                "C-input warp aux producer expects "
                "broadcast_axis in {None, 1}; the chain "
                "analyzer rejects other forms"
            )
            n_global = int(desc.host_tensor_val.shape[0])
            setup.extend(
                [
                    statement_from_string(
                        f"{gmem_aux_view_var} = cute.make_tensor("
                        f"{aux_tensor_name}.iterator, "
                        f"cute.make_layout(({bm_per_cta}, {n_global}), "
                        "stride=(0, 1)))"
                    ),
                    statement_from_string(
                        f"{gmem_aux_tile_var} = cute.local_tile("
                        f"{gmem_aux_view_var}, ({bm_per_cta}, {bn}), "
                        f"(cutlass.Int32(0), {tile_n_var}))"
                    ),
                ]
            )
        # Subdivide the per-output-tile aux region into epi-tile-
        # sized subtiles using ``flat_divide(epi_tile)`` —
        # mirrors the consumer-side ``flat_divide`` so the
        # producer and consumer iterate the same subtile
        # ordering. ``group_modes(..., 2, rank)`` collapses the
        # outer (subtile_m, subtile_n) modes into one linear
        # subtile axis so the producer's subtile loop sees a
        # flat ``subtile_count`` extent (matches the consumer's
        # post-``group_modes`` shape used inside
        # ``_aux_subtile_load_source``).
        setup.extend(
            [
                statement_from_string(
                    f"{gmem_aux_subtiles_var} = cute.flat_divide("
                    f"{gmem_aux_tile_var}, {aux_epi_tile_var})"
                ),
                statement_from_string(
                    f"{gmem_subtiles_grouped_var} = cute.group_modes("
                    f"{gmem_aux_subtiles_var}, 2, "
                    f"cute.rank({gmem_aux_subtiles_var}))"
                ),
            ]
        )
        if aux_use_tma_load:
            tma_smem_part_var = device_function.new_var(
                f"tcgen05_aux_tma_smem_part_{desc_idx}"
            )
            tma_gmem_part_var = device_function.new_var(
                f"tcgen05_aux_tma_gmem_part_{desc_idx}"
            )
            setup.append(
                statement_from_string(
                    f"{tma_smem_part_var}, {tma_gmem_part_var} = "
                    "cute.nvgpu.cpasync.tma_partition("
                    f"{tma_atom}, 0, cute.make_layout(1), "
                    f"cute.group_modes({ring.smem}, 0, "
                    f"cute.rank({ring.smem}) - 1), "
                    f"cute.group_modes({gmem_subtiles_grouped_var}, 0, "
                    f"cute.rank({gmem_subtiles_grouped_var}) - 1))"
                )
            )
        else:
            # Cooperative-copy ``TiledCopy`` for the C-input warp's
            # 32 lanes. The atom uses ``CopyUniversalOp`` (regular
            # SIMT ld+st): cp.async would impose a 128-bit
            # source-iterator alignment check that the layout-
            # implied stride alignment cannot satisfy at IR-build
            # time (the host pointer is 16-byte aligned, but the
            # minimum row stride is 1 element = 16 bits for bf16).
            # ``CopyUniversalOp`` lowers to a SIMT ld/st pair whose
            # vectorization is driven at runtime by the host
            # pointer's actual alignment.
            # Use the actual shared epilogue subtile, not (bm, bn): a
            # fixed 4x64 BF16 copy over a 128x32 subtile aliases shared
            # destinations and reads beyond the final global N tile.
            # Resolve each descriptor independently because its dtype
            # determines the number of contiguous values in 128 bits.
            m_threads = device_function.new_var(
                f"tcgen05_aux_copy_m_threads_{desc_idx}"
            )
            n_threads = device_function.new_var(
                f"tcgen05_aux_copy_n_threads_{desc_idx}"
            )
            num_copy_elems = device_function.new_var(
                f"tcgen05_aux_copy_values_{desc_idx}"
            )
            setup.extend(
                [
                    statement_from_string(
                        f"{m_threads}, {n_threads}, {num_copy_elems} = "
                        "cutlass.const_expr(_cute_aux_copy_layout("
                        f"cute.size({aux_epi_tile_var}[0]), "
                        f"cute.size({aux_epi_tile_var}[1]), "
                        f"{dtype_bits}))"
                    ),
                    statement_from_string(
                        f"{tiled_copy_var} = cute.make_tiled_copy_tv("
                        f"cute.make_copy_atom("
                        f"cute.nvgpu.CopyUniversalOp(), {aux_dtype_str}, "
                        f"num_bits_per_copy={num_copy_elems} * {dtype_bits}), "
                        f"cute.make_ordered_layout("
                        f"({m_threads}, {n_threads}), order=(1, 0)), "
                        f"cute.make_layout((1, {num_copy_elems})))"
                    ),
                    statement_from_string(
                        f"{thr_copy_var} = {tiled_copy_var}.get_slice("
                        f"{cute_lane_idx_var})"
                    ),
                ]
            )
        per_descriptor_setup_blocks.append(setup)
        per_descriptor_grouped_names.append(gmem_subtiles_grouped_var)

        # Per-subtile body source: builds the per-stage GMEM
        # slice + SMEM stage slice and issues one cooperative
        # ``cute.copy``. The loop variable
        # ``_tcgen05_aux_subtile`` indexes the flat subtile
        # axis (collapsed via ``group_modes(..., 2, rank)``);
        # the consumer's per-subtile loop indexes the same
        # axis identically.
        if aux_use_tma_load:
            subtile_lines = [
                (
                    f"cute.copy({tma_atom}, "
                    f"{tma_gmem_part_var}[None, "
                    f"cutlass.Int32(_tcgen05_aux_subtile)], "
                    f"{tma_smem_part_var}[None, "
                    f"{aux_producer_state_name}.index], "
                    f"tma_bar_ptr={aux_tma_barrier_var})"
                ),
            ]
        else:
            subtile_lines = [
                (
                    f"{gmem_subtile_var} = "
                    f"{gmem_subtiles_grouped_var}[None, None, "
                    f"cutlass.Int32(_tcgen05_aux_subtile)]"
                ),
                (
                    f"{smem_stage_var} = {ring.smem}[None, None, "
                    f"{aux_producer_state_name}.index]"
                ),
                (f"{gmem_part_var} = {thr_copy_var}.partition_S({gmem_subtile_var})"),
                (f"{smem_part_var} = {thr_copy_var}.partition_D({smem_stage_var})"),
                (f"cute.copy({tiled_copy_var}, {gmem_part_var}, {smem_part_var})"),
            ]
        per_descriptor_subtile_blocks.append(subtile_lines)

    return AuxProducerBody(
        device_function=device_function,
        use_tma_load=aux_use_tma_load,
        pipeline=aux_pipeline_name,
        producer_state=aux_producer_state_name,
        tma_barrier_var=aux_tma_barrier_var,
        lane_idx_var=cute_lane_idx_var,
        tile_m_var=tile_m_var,
        tile_n_var=tile_n_var,
        tile_m_expr=tile_m_expr,
        tile_n_expr=tile_n_expr,
        full_tile_var=aux_full_tile_var,
        full_tile_expr=aux_full_tile_expr,
        per_descriptor_setup_blocks=per_descriptor_setup_blocks,
        per_descriptor_subtile_blocks=per_descriptor_subtile_blocks,
        per_descriptor_grouped_names=per_descriptor_grouped_names,
    )
