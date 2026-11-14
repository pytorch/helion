"""CuTe autotune search-surface registration during device-IR lowering.

Role: analysis (knob registration at device-IR time).  Nothing here rewrites a graph; every
function reads the lowered ``DeviceIR`` and widens or narrows the
``ConfigSpec`` (knob families, pid types, block-id slots) or records the
runtime specializations the CuTe emitters key their caches on.

``CuteBackend.register_search_surface`` runs :func:`register_search_surface`
once after reduction rolling; it detects the flash, gated-flash and tcgen05
GEMM shapes (through the attention matchers and the MMA analyzer) and enables
their knob families.  The smaller per-position registrations below are
dispatched from ``ir_passes.rewrite_device_ir`` at the stage each held when
it was inlined in ``helion/_compiler/device_ir.py`` (``lower_to_device_ir``
and ``DeviceIR.register_rollable_reductions``); their relative order against
the core steps is part of the search-surface identity and must not move.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ... import exc
from ...autotuner.config_spec import CuteLaneLayoutSpec
from ...autotuner.config_spec import CuteReductionReloadSpec
from ...autotuner.config_spec import CuteVectorWidthSpec
from ...autotuner.config_spec import NumThreadsSpec
from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource
from . import target_facts
from .autotune.flash_surface import enable_cute_flash_gated_search
from .autotune.flash_surface import enable_cute_flash_search
from .state import cute_env_state
from .tcgen05_config import facts

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Sequence

    from ...autotuner.config_spec import ConfigSpec
    from ..device_ir import DeviceIR
    from ..device_ir_analysis import DeviceIRAnalysis
    from ..host_function import HostFunction
    from .cute_mma import _CuteMmaNode
    from .tcgen05.search_plan import CuteTcgen05SearchPlan
    from .tcgen05.search_plan import _CuteTcgen05SearchPlanningResult


def _cute_tcgen05_output_store_facts(
    device_ir: DeviceIR, node: torch.fx.Node
) -> tuple[int | None, int]:
    """``(output dtype width, store count)`` of one MMA, for the SMEM fit estimate."""
    from .cute_mma import _trace_mma_to_stores

    stores = _trace_mma_to_stores(node, device_ir.graphs)
    if not stores:
        return None, 1
    widths: list[int] = []
    for store in stores:
        source = store.args[0] if store.args else None
        value = source.meta.get("val") if isinstance(source, torch.fx.Node) else None
        if isinstance(value, torch.Tensor):
            widths.append(value.dtype.itemsize)
    if not widths:
        return None, 1
    return max(widths), len(stores)


def _finalize_cute_tcgen05_search_planning(
    env: CompileEnvironment,
    planning_results: Sequence[_CuteTcgen05SearchPlanningResult],
    analysis_keys: Collection[object],
) -> bool:
    """Publish every guard and report whether one cache-safe analysis remains."""
    env.specialized_vars.update(
        symbol
        for result in planning_results
        for symbol in result.required_specialized_vars
    )
    return (
        all(result.guards_complete for result in planning_results)
        and len(analysis_keys) == 1
    )


def _enable_cute_phase_mma_search(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    planning_results: Sequence[_CuteTcgen05SearchPlanningResult],
    search_candidates: Sequence[
        tuple[_CuteMmaNode, torch.Tensor, CuteTcgen05SearchPlan]
    ],
    mma_candidates: Sequence[
        tuple[_CuteMmaNode, torch.Tensor, torch.Tensor, torch.fx.Node]
    ],
) -> bool:
    """Plan an ordered ``hl.barrier()`` chain per phase; False where it declines."""
    from .materialized_mma import enable_phase_mma_search

    return enable_phase_mma_search(
        env,
        device_ir,
        planning_results,
        search_candidates,
        mma_nodes={
            id(candidate): node for candidate, _lhs, _rhs, node in mma_candidates
        },
    )


def register_lane_vector_width_specs(config_spec: ConfigSpec) -> None:
    """Pre-register CuTe vector-width slots for every lane-loop-eligible block.

    On the CuTe backend a tile block whose configured ``block_size`` exceeds
    its ``num_threads`` is traversed by a synthetic lane loop, and the tile
    strategy lazily registers a ``cute_vector_widths`` slot for it during
    codegen. That lazy append happens *per config* and is config dependent
    (whether a lane loop appears depends on the config's block sizes and
    num_threads), so the shared ``config_spec`` would grow after
    ``ConfigGeneration`` has already captured its ``flat_spec`` — leaving the
    flat layout stale and causing ``unflatten`` to over-walk the spec.

    Registering the slots once here (config independently) keeps the lazy
    append idempotent: any block that *could* be lane-looped under some config
    already owns its slot, so the per-config append is a no-op and the spec
    stays a fixed size for the lifetime of autotuning. Blocks that never end up
    lane-looped simply keep ``V=1`` (the slot's default), which the tile
    strategy ignores.
    """
    num_thread_block_ids = set(config_spec.num_threads.valid_block_ids())
    existing = set(config_spec.cute_vector_widths.valid_block_ids())
    existing_layouts = set(config_spec.cute_lane_layouts.valid_block_ids())
    for spec in config_spec.block_sizes:
        block_id = spec.block_id
        # ``max_size`` bounds the largest block_size the autotuner may pick;
        # only blocks that can exceed a single thread can ever be lane-looped.
        if block_id not in num_thread_block_ids or spec.max_size <= 1:
            continue
        if block_id not in existing:
            config_spec.cute_vector_widths.append(
                CuteVectorWidthSpec(block_id=block_id, size_hint=spec.size_hint)
            )
            existing.add(block_id)
        if block_id not in existing_layouts:
            config_spec.cute_lane_layouts.append(CuteLaneLayoutSpec(block_id=block_id))
            existing_layouts.add(block_id)


def register_tile_vector_slots(env: CompileEnvironment) -> None:
    """Eagerly register cute_vector_widths slots for non-reduction tile blocks.

    Eager (not lazy in codegen) keeps the config-spec width fixed before the
    autotuner snapshots it; callers register reduction-dim slots first so
    those keep index 0.
    """
    already_registered = set(env.config_spec.cute_vector_widths.valid_block_ids())
    for tile_bs in env.block_sizes:
        if not tile_bs.is_tile or tile_bs.block_id in already_registered:
            continue
        # Non-static size (jagged / AutoSize) never forms a lane loop.
        if not isinstance(tile_bs.size, (int, torch.SymInt)):
            continue
        try:
            size_hint_val = int(tile_bs.size_hint())
        except (TypeError, ValueError, AttributeError, AssertionError):
            continue
        env.config_spec.cute_vector_widths.append(
            CuteVectorWidthSpec(
                block_id=tile_bs.block_id,
                size_hint=size_hint_val,
            )
        )
        env.config_spec.cute_lane_layouts.append(
            CuteLaneLayoutSpec(block_id=tile_bs.block_id)
        )


def register_search_surface(device_ir: DeviceIR, config_spec: ConfigSpec) -> None:
    """Enable the CuTe knob families this kernel can use (``CuteBackend.
    register_search_surface``, after ``register_rollable_reductions``):
    lane vector widths, tile packing, exact division, signed bitfields and
    the flash / gated-flash / tcgen05 GEMM search surfaces."""
    env = CompileEnvironment.current()
    env_state = cute_env_state(env)
    register_lane_vector_width_specs(config_spec)
    from .hoist_loop_invariant_recip import has_hoistable_fp32_divisions
    from .signed_bitfield import has_signed_byte_field
    from .tile_packing import tile_pack_counts

    config_spec.cute_tile_pack_counts = tile_pack_counts(env, device_ir)

    # An fp32 quotient by a scalar or a broadcast row value: the exact
    # division hoist can apply, so its form is tunable.
    config_spec.cute_exact_division_sites = has_hoistable_fp32_divisions(
        info.graph for info in device_ir.graphs
    )

    config_spec.cute_signed_bitfield_bf16_available = (
        config_spec.target_device_capability == (10, 0)
        and any(has_signed_byte_field(info.graph) for info in device_ir.graphs)
    )
    # Enable the flash-attention autotune surface when
    # the dense flash dataflow is detected, analogous to how a matmul
    # detection sets ``tcgen05.search_enabled``. Default-off
    # otherwise so the flash knobs never widen the search surface for
    # ordinary cute kernels.
    from .analysis.attention_match import detect_flash_search_surface
    from .cute_flash_bwd import detect_flash_bwd_search_surface
    from .cute_flash_gated import detect_flash_gated_search_surface

    detect_flash_bwd_search_surface(device_ir)
    flash_shape = detect_flash_search_surface(device_ir)
    gated_surface = (
        None
        if flash_shape is not None
        else detect_flash_gated_search_surface(device_ir)
    )
    if flash_shape is not None:
        from .flash_facts import FlashEnv
        from .flash_facts import FlashSurfaceFacts
        from .tcgen05.search_plan import _cuda_num_sms_or_zero

        flash_facts = FlashSurfaceFacts.from_surface(
            flash_shape,
            device_sm_count=_cuda_num_sms_or_zero(env.device),
            fast_math=env.settings.fast_math,
            env=FlashEnv.capture(env.settings.cute_flash_env_overrides),
        )
        enable_cute_flash_search(config_spec, **flash_facts.as_kwargs())
    elif gated_surface is not None:
        # The fused gated body owns the whole root; tcgen05 matmul
        # planning would add search fields that hide the gated knob.
        enable_cute_flash_gated_search(
            config_spec,
            block_size_targets=gated_surface.block_size_targets,
            kv_block_id=gated_surface.kv_block_id,
            q_block_id=gated_surface.q_block_id,
            q_tile_choices=gated_surface.q_tile_choices,
            kv_tile_choices=gated_surface.kv_tile_choices,
            kv_stage_choices=gated_surface.kv_stage_choices,
            kv_stage_default=gated_surface.kv_stage_default,
            gate_warpgroup_choices=gated_surface.gate_warpgroup_choices,
            gate_warpgroup_default=gated_surface.gate_warpgroup_default,
            kv_stage_choices_by_tile=dict(gated_surface.kv_stage_choices_by_tile),
            row_warp_choices_by_tile=dict(gated_surface.row_warp_choices_by_tile),
            row_stage_choices=dict(gated_surface.row_stage_choices),
            row_seed_programs=gated_surface.row_programs,
        )
    else:
        from .cute_mma import _rank3_grouped_root_axes
        from .cute_mma import analyze_cute_mma_node
        from .cute_mma import tcgen05_fragment_epilogue_source_tiles_reachable
        from .tcgen05.search_plan import _plan_cute_tcgen05_search_candidate
        from .tcgen05.search_plan import enable_cute_tcgen05_search

        # The same structural analyzer gates tcgen05 search and codegen
        # for every matrix rank. This prevents transformed loads from
        # receiving a tcgen05-only search space that codegen later rejects.
        root_grid_block_ids = {
            tuple(block_ids) for block_ids in device_ir.grid_block_ids
        }
        mma_candidates = []
        # Candidates whose operand is an in-register integer cast
        # (``_MmaOperandInfo.cast_dtype``): only the register-MMA
        # family lowers them, and it replaces the whole body of ONE
        # plain GEMM, so such a candidate enters the search only when
        # it is the kernel's single matmul and its stores are the
        # epilogue chains the family renders.
        cast_candidates: list[object] = []
        for graph_info in device_ir.graphs:
            for node in graph_info.graph.nodes:
                candidate = analyze_cute_mma_node(
                    node, device_ir=device_ir, allow_integer_casts=True
                )
                if candidate is None or candidate.requires_accumulator_seed:
                    continue
                if candidate.operands.output_block_ids not in root_grid_block_ids:
                    if not candidate.operands.rhs.rhs_is_grouped:
                        continue
                    grouped_axes = _rank3_grouped_root_axes(
                        env,
                        device_ir,
                        m_block_id=candidate.operands.m_block_id,
                        n_block_id=candidate.operands.n_block_id,
                        k_block_id=candidate.operands.k_block_id,
                    )
                    leading_block_id = candidate.operands.leading_passthrough_block_id
                    if (
                        grouped_axes is None
                        or grouped_axes.segment_block_id is None
                        or leading_block_id is None
                        or env.canonical_block_id(grouped_axes.segment_block_id)
                        != env.canonical_block_id(leading_block_id)
                    ):
                        continue
                lhs = candidate.lhs.meta.get("val")
                rhs = candidate.rhs.meta.get("val")
                if not (
                    isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor)
                ):
                    continue
                # Operand analysis may prove an FP32 upcast lossless.
                # Search must use the same source dtype as MMA codegen;
                # these are fake values, not changes to the FX inputs.
                # (A cast operand plans with the dtype the tensor core
                # multiplies, which is what the program casts to.)
                lhs = lhs.to(dtype=candidate.operands.lhs.mma_dtype)
                rhs = rhs.to(dtype=candidate.operands.rhs.mma_dtype)
                if (
                    candidate.operands.lhs.cast_dtype is not None
                    or candidate.operands.rhs.cast_dtype is not None
                ):
                    from .cute_warp_mma_gemm import warp_mma_store_matches

                    try:
                        warp_mma_store_matches(
                            device_ir.graphs, node, candidate.operands
                        )
                    except exc.BackendUnsupported:
                        continue
                    cast_candidates.append(candidate)
                mma_candidates.append((candidate, lhs, rhs, node))
        if cast_candidates and len(mma_candidates) != 1:
            mma_candidates = [
                item for item in mma_candidates if item[0] not in cast_candidates
            ]
            cast_candidates = []
        supports_small_n_role_local_tma = bool(mma_candidates) and all(
            candidate.supports_small_n_role_local_tma
            for candidate, _lhs, _rhs, _node in mma_candidates
        )
        supports_small_n_scalar_fallback = bool(mma_candidates) and all(
            candidate.supports_small_n_scalar_fallback
            for candidate, _lhs, _rhs, _node in mma_candidates
        )
        search_candidates = []
        planning_results = []
        for candidate, lhs, rhs, _node in mma_candidates:
            planning_result = _plan_cute_tcgen05_search_candidate(
                lhs,
                rhs,
                has_leading_passthrough=(candidate.operands.has_leading_passthrough),
                supports_small_n_role_local_tma=(supports_small_n_role_local_tma),
                supports_small_n_scalar_fallback=(supports_small_n_scalar_fallback),
                allow_dynamic_hints=(
                    candidate.operands.rhs.rhs_segment_group is not None
                    or candidate.operands.rhs.rhs_packed_group is not None
                ),
                lhs_source=candidate.operands.lhs.source_fake,
                rhs_source=candidate.operands.rhs.source_fake,
                operands_permuted=(
                    candidate.operands.lhs.source_to_logical_order is not None
                    or candidate.operands.rhs.source_to_logical_order is not None
                ),
                operand_casts=candidate in cast_candidates,
            )
            planning_results.append(planning_result)
            search_plan = planning_result.plan
            if search_plan is None or not candidate.supports_tcgen05_search_plan(
                search_plan
            ):
                continue
            if not tcgen05_fragment_epilogue_source_tiles_reachable(
                candidate, search_plan, config_spec
            ):
                continue
            search_candidates.append((candidate, lhs, search_plan))
        analysis_keys = {
            (
                candidate.operands.output_block_ids,
                candidate.operands.k_block_id,
                candidate.operands.lhs.mma_dtype,
            )
            for candidate, _lhs, _plan in search_candidates
        }
        # Every dynamic candidate can change the accepted analysis-key
        # set on a later binding. Publish all source-backed guards only
        # after every candidate has observed the same compile state.
        single_analysis = _finalize_cute_tcgen05_search_planning(
            env,
            planning_results,
            analysis_keys,
        )
        if env_state.fission_plan is None and (
            env.has_barrier
            and len(device_ir.root_ids) > 1
            and _enable_cute_phase_mma_search(
                env,
                device_ir,
                planning_results,
                search_candidates,
                mma_candidates,
            )
        ):
            # An ordered ``hl.barrier()`` chain plans per phase
            # whenever its phases prove out, with one GEMM phase as
            # with several: each GEMM launch owns its tile triple,
            # budget and per-launch pid type, and the plain phases
            # their layout knobs.  A chain the per-phase planner
            # declines takes the standalone surface below.
            pass
        elif single_analysis:
            candidate, lhs, search_plan = search_candidates[0]
            output_store_facts = _cute_tcgen05_output_store_facts(
                device_ir,
                next(
                    node
                    for item, _lhs, _rhs, node in mma_candidates
                    if item is candidate
                ),
            )
            config_spec._cute_tcgen05_config.fission_bundle = (
                env_state.fission_plan is not None
            )
            enable_cute_tcgen05_search(
                lhs,
                plan=search_plan,
                m_block_id=candidate.operands.m_block_id,
                n_block_id=candidate.operands.n_block_id,
                k_block_id=candidate.operands.k_block_id,
                input_dtype=candidate.operands.lhs.mma_dtype,
                has_leading_passthrough=(candidate.operands.has_leading_passthrough),
                explicit_epi_tile_compatible=all(
                    item.explicit_epi_tile_compatible
                    for item, _lhs, _plan in search_candidates
                ),
                leading_block_id=(candidate.operands.leading_passthrough_block_id),
                # The register-MMA family replaces the whole device
                # body of ONE GEMM (its root tile loop around its K
                # loop); kernels with several matmuls keep tcgen05.
                warp_mma_plain_kernel=len(mma_candidates) == 1,
                output_dtype_bytes=output_store_facts[0],
                output_stores=output_store_facts[1],
            )
            if (
                len(mma_candidates) == 1
                and not cast_candidates
                and not (
                    config_spec.reduction_block_ids - {candidate.operands.k_block_id}
                )
            ):
                from .pipeline_smem import analyze_pipeline_smem_facts
                from .pipeline_smem import pipeline_region_graphs

                tcgen05_config = config_spec._cute_tcgen05_config
                allocation_graphs = pipeline_region_graphs(
                    device_ir,
                    mma_candidates[0][3],
                    separately_launched=env_state.fission_plan is not None,
                )
                smem_facts = (
                    analyze_pipeline_smem_facts(
                        candidate,
                        mma_candidates[0][3],
                        allocation_graphs,
                        # A leading batch axis (bmm) rides the same
                        # static persistent two-CTA schedule: the
                        # batch coordinate only selects the TMA
                        # tensors' leading mode and the scheduler's
                        # extra axis.
                        allow_leading_passthrough=(
                            candidate.operands.has_leading_passthrough
                        ),
                        allow_fp8_broadcasts=(
                            len(device_ir.root_ids) == 1
                            and env_state.fission_plan is None
                        ),
                        capacity_bytes=(
                            target_facts.per_cta_smem_capacity_bytes(lhs.device)
                        ),
                    )
                    if allocation_graphs is not None
                    else None
                )
                if smem_facts is not None:
                    facts.register_pipeline_smem_facts(tcgen05_config, smem_facts)
                    from .materialized_pdl import prove_materialized_operand_pdl

                    tcgen05_config.materialized_operand_pdl_roots = (
                        prove_materialized_operand_pdl(env, device_ir, candidate)
                    )
        elif env_state.fission_plan is not None:
            from .materialized_mma import enable_materialized_mma_search

            enable_materialized_mma_search(
                env,
                device_ir,
                planning_results,
                search_candidates,
                mma_nodes={
                    id(candidate): node
                    for candidate, _lhs, _rhs, node in mma_candidates
                },
            )


def detect_collective_knobs(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """Keep the collective-matmul knobs and the epilogue store split only
    where a contraction can act on them."""
    from .collective_matmul import collective_knobs_apply
    from .collective_matmul import has_k_loop_less_matmul

    env = CompileEnvironment.current()
    config_spec = env.config_spec
    # Knobs no contraction can act on leave the search surface: the
    # collective marker needs an addmm/dot site with a tiled or a
    # short full-slice K, and the epilogue store split does not
    # survive a matmul on the direct/serial path.
    config_spec.cute_collective_knobs_enabled = collective_knobs_apply(env, device_ir)
    if has_k_loop_less_matmul(env, device_ir):
        config_spec.epilogue_subtile_candidate_enabled = False


def register_reduction_specializations(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """Runtime specializations and tile constraints read before the roller
    runs: vector-alignment and flat-gather factor guards, alias facts for
    reduction kernels and the jagged tile overshoot."""
    env = CompileEnvironment.current()
    rdims = [bs for bs in env.block_sizes if bs.reduction]
    from ..runtime_tensor_guards import register_tensor_alias_specializations
    from .flat_gather_lane import register_flat_gather_factor_specializations
    from .memory_ops import register_persistent_vec_alignment_specializations

    # Explicit tile loops can also use vector memory transactions,
    # even when no persistent reduction block or contraction exists.
    register_persistent_vec_alignment_specializations(env)
    # A flat gather scaled by an ``int`` argument packs its lanes
    # only when the argument is a multiple of the packet width.
    register_flat_gather_factor_specializations(env, device_ir)
    if rdims:
        # Reduction kernels reorder input reads across warps. Grid
        # kernels add these storage-dependent dispatch guards below
        # only when a scan needs them.
        register_tensor_alias_specializations(env)
    for tile_spec in env.config_spec.block_sizes:
        if len(tile_spec.block_ids) == 1 and env.is_jagged_tile(tile_spec.block_id):
            # A device-resident length's generic size hint is not an
            # upper bound. Let the tuner explore up to 64 values per
            # thread across the hardware's 1024-thread CTA limit.
            # Fixed tiles and tiles bounded by an outer tile keep
            # their existing constraints.
            tile_spec.allow_overshoot(1024 * 64)


def register_grid_tile_slots(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """Grid (reduction-free) kernels: eager tile vector slots and the alias
    facts scans, collective matmuls and input-storage stores need."""
    env = CompileEnvironment.current()
    rdims = [bs for bs in env.block_sizes if bs.reduction]
    # Eager tile-slot registration (see register_tile_vector_slots).
    # A present reduction must keep its slot at index 0, so reduction
    # kernels register their tile slots after the reduction-loop pass below.
    if not rdims:
        from ...language.scan_ops import _associative_scan
        from ..runtime_tensor_guards import register_tensor_alias_specializations
        from .collective_matmul import has_collective_matmul_candidate
        from .memory_ops import stores_into_input_storage

        register_tile_vector_slots(env)
        # Grid scans also reorder input reads across warps and need the
        # same cache-specialized alias facts as reduction kernels.
        # Collective MMA schedules prove operand loads disjoint from
        # row-loop stores only through them, and vector memory passes
        # likewise move input loads across a store into an external
        # tensor only under a disjointness proof. Do not add
        # storage-dependent dispatch guards to unrelated grid kernels,
        # whose stores land in fresh host allocations.
        if (
            any(
                node.op == "call_function" and node.target is _associative_scan
                for graph_info in device_ir.graphs
                for node in graph_info.graph.nodes
            )
            or has_collective_matmul_candidate(device_ir.graphs)
            or stores_into_input_storage(device_ir.graphs, env)
        ):
            register_tensor_alias_specializations(env)


def register_reduction_layout_slots(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """Per static reduction block: ``num_threads``, vector width, lane layout
    and reload slots, whether or not the block rolls into a loop."""
    env = CompileEnvironment.current()
    # Every reduction dim has exactly one roller result, in ``rdims`` order;
    # only the dim itself is read here.
    rdims = [bs for bs in env.block_sizes if bs.reduction]
    # CuTe's persistent reduction strategy can also split a static
    # reduction over fewer live threads plus synthetic per-thread lanes.
    # Those layout choices are independent of whether ReductionRoller can
    # turn the reduction into an outer ``reduction_loop``.  In particular,
    # an ``hl.arange``-indexed reduction is deliberately non-rollable but
    # is still a profitable persistent subwarp reduction.  Register the
    # orthogonal CuTe knobs for every static reduction block; only the
    # ``ReductionLoopSpec`` below remains conditional on rollability.
    num_thread_blocks = set(env.config_spec.num_threads.valid_block_ids())
    vector_blocks = set(env.config_spec.cute_vector_widths.valid_block_ids())
    lane_layout_blocks = set(env.config_spec.cute_lane_layouts.valid_block_ids())
    reload_blocks = set(env.config_spec.cute_reduction_reloads.valid_block_ids())
    for rdim in rdims:
        # Reduction lowering materializes output-range blocks even
        # when the range is exactly an already-active tile symbol.
        # CuTe reuses that tile's execution strategy, so separate
        # reduction tuning slots would be dead dimensions in the
        # search space.
        if env.canonical_block_id(rdim.block_id) != rdim.block_id:
            continue
        if not isinstance(rdim.size, (int, torch.SymInt)):
            continue
        size_hint = rdim.size_hint()
        if rdim.block_id not in num_thread_blocks:
            env.config_spec.num_threads.append(
                NumThreadsSpec(
                    block_id=rdim.block_id,
                    size_hint=size_hint,
                )
            )
            num_thread_blocks.add(rdim.block_id)
        if rdim.block_id not in vector_blocks:
            env.config_spec.cute_vector_widths.append(
                CuteVectorWidthSpec(
                    block_id=rdim.block_id,
                    size_hint=size_hint,
                )
            )
            vector_blocks.add(rdim.block_id)
        if rdim.block_id not in lane_layout_blocks:
            env.config_spec.cute_lane_layouts.append(
                CuteLaneLayoutSpec(block_id=rdim.block_id)
            )
            lane_layout_blocks.add(rdim.block_id)
        if rdim.block_id not in reload_blocks:
            env.config_spec.cute_reduction_reloads.append(
                CuteReductionReloadSpec(block_id=rdim.block_id)
            )
            reload_blocks.add(rdim.block_id)


def admit_register_tile_reduction(
    env: CompileEnvironment, device_ir: DeviceIR, block_id: int
) -> None:
    """Offer a rolled reduction block to the register-tile family
    (``cute_register_tile_reduction_blocks``).  Called once per rollable
    reduction dim, right after its ``ReductionLoopSpec`` is appended
    (``CuteBackend.register_reduction_loop_config_slots``)."""
    # Resolved through the module at call time: tests patch the admission.
    from .register_tile_admission import register_tile_body_admitted

    if not register_tile_body_admitted(device_ir.graphs):
        return
    rdim = env.block_sizes[block_id]
    # A persistent reduction lane may nest outside one-vector tile
    # wrappers as a register tile only over a static, unmasked
    # extent (the lane count is a trace-time constant) and a body
    # the two-pass schedule can lower.  A CuTe-only decision: no
    # other backend reads the extent here.
    numel = rdim.numel
    if (
        isinstance(numel, (int, sympy.Integer))
        and int(numel) > 0
        and env.backend.static_rdim_size(int(numel)) == int(numel)
    ):
        env.config_spec.cute_register_tile_reduction_blocks.add(block_id)


def register_indexed_reduction_blocks(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """After the reduction loops are registered: the argmin/argmax reduction
    blocks (single-warp combine) and the remaining tile vector slots."""
    env = CompileEnvironment.current()
    # The graphs that existed before rolling: the roller only ever appends
    # ``ReductionLoopGraphInfo`` sweeps (``DeviceIR.add_reduction_loop_graph``).
    from ..device_ir import ReductionLoopGraphInfo

    original_graphs = [
        graph_info
        for graph_info in device_ir.graphs
        if not isinstance(graph_info, ReductionLoopGraphInfo)
    ]
    # Track which rdims appear as the reduction axis of an indexed
    # reduction (argmin/argmax). On CuTe these can only be combined
    # via cute.arch.warp_reduction (32 threads max), so the autotuner
    # must keep their persistent thread count and looped chunk size
    # within a single warp.
    indexed_blocks: set[int] = set()
    indexed_targets = {
        torch.ops.aten.argmin.default,
        torch.ops.aten.argmax.default,
    }
    for graph_info in original_graphs:
        for node in graph_info.graph.nodes:
            if getattr(node, "target", None) not in indexed_targets:
                continue
            args = node.args or ()
            if not args:
                continue
            val = getattr(args[0], "meta", {}).get("val")
            if val is None:
                continue
            dim_arg = args[1] if len(args) >= 2 else -1
            dim_indices = (
                [int(cast("int", d)) for d in dim_arg]
                if isinstance(dim_arg, list)
                else [int(cast("int", dim_arg))]
            )
            for dim_idx in dim_indices:
                if dim_idx < 0:
                    dim_idx += val.ndim
                if 0 <= dim_idx < val.ndim:
                    reduce_dim = val.size(dim_idx)
                    block_id = env.resolve_block_id(reduce_dim)
                    if block_id is not None:
                        indexed_blocks.add(block_id)
    env.config_spec.cute_indexed_reduction_block_ids = indexed_blocks

    # Reduction-dim slots are registered above; add the remaining
    # non-reduction tile slots after them (keeps the rdim slot at index 0).
    register_tile_vector_slots(env)


def restrict_tcgen05_multi_root_search(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """Multi-root kernels after the shared-pid ``xyz`` disallow: tcgen05
    search keeps persistent pid types out, and an ``hl.barrier()`` chain
    publishes the plain phases' layout knobs."""
    env = CompileEnvironment.current()
    config_spec = env.config_spec
    if len(device_ir.root_ids) > 1 and cute_env_state(env).fission_plan is None:
        if config_spec.tcgen05.search_enabled:
            # The tcgen05 persistent launch grid is derived from a single
            # root's PID space today. Keep persistent pid types out of
            # multi-root autotune until the scheduler/grid spans all cases.
            non_persistent_pid_types = tuple(
                pid_type
                for pid_type in config_spec.allowed_pid_types
                if pid_type not in ("persistent_blocked", "persistent_interleaved")
            )
            if not non_persistent_pid_types:
                raise exc.InvalidConfig(
                    "CuTe tcgen05 multi-root kernels do not support "
                    "persistent pid types yet, and no non-persistent "
                    "pid type is available. Disable forced/distributed "
                    "persistent-only mode or use a single root loop."
                )
            for pid_type in config_spec.allowed_pid_types:
                if pid_type not in non_persistent_pid_types:
                    config_spec.disallowed_pid_type_reasons.setdefault(
                        pid_type,
                        "CuTe tcgen05 multi-root kernels do not support "
                        "persistent pid types yet (single-root launch grid)",
                    )
            config_spec.allowed_pid_types = non_persistent_pid_types
        if config_spec.tcgen05.search_enabled and env.has_barrier:
            # The flat chain launches every ``hl.barrier()`` phase on its
            # own; a phase without a matmul keeps the SIMT layout knobs
            # a fission plan's pointwise region has (``enable_phase_
            # mma_search`` publishes them itself for multi-GEMM chains).
            from .materialized_mma import register_plain_barrier_phases

            register_plain_barrier_phases(env, device_ir)


def record_tile_loop_facts(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """After the structural analysis: tile loop paths, the reduction blocks
    on each and the path of each tile strategy, boundary-only grid blocks,
    the sizes of the paths' fixed blocks and whether an fp32 dot is exact on
    this target."""
    env = CompileEnvironment.current()
    config_spec = env.config_spec
    from .loop_nesting import boundary_only_grid_blocks
    from .loop_nesting import tile_loop_facts

    path_reductions, strategy_paths = tile_loop_facts(
        env, device_ir, device_ir.graphs, config_spec.reduction_block_ids
    )
    config_spec.cute_tile_loop_path_reductions = path_reductions
    config_spec.cute_tile_strategy_paths = strategy_paths
    fixed = {
        block_id: source.value
        for path in config_spec.cute_tile_loop_paths
        for level in path
        for block_id in level
        if isinstance(
            source := env.block_sizes[block_id].block_size_source,
            FixedBlockSizeSource,
        )
    }
    config_spec.cute_tile_fixed_block_sizes = {
        block_id: value for block_id, value in fixed.items() if type(value) is int
    }
    config_spec.cute_tile_symbolic_fixed_blocks = frozenset(
        block_id for block_id, value in fixed.items() if type(value) is not int
    )
    config_spec.cute_inactive_tile_block_ids = boundary_only_grid_blocks(
        env, device_ir, device_ir.graphs
    )
    from .mma_support import cute_fp32_dot_uses_tf32

    config_spec.cute_fp32_dot_exact = not cute_fp32_dot_uses_tf32()


def register_matmul_alias_specializations(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """After the kernel matmul fact: alias facts for contractions and for
    stores that can overwrite an input."""
    env = CompileEnvironment.current()
    config_spec = env.config_spec
    assert analysis is not None
    if config_spec.kernel_matmul_fact is not None or analysis.writes_input_storage(env):
        from ..runtime_tensor_guards import register_tensor_alias_specializations

        # Vectorized writes to inputs and collective matmuls need the same
        # guarded disjoint-input facts as reductions. Fresh grid outputs
        # cannot alias inputs and must not add storage-dependent cache keys.
        register_tensor_alias_specializations(env)


def restore_pointwise_flatten_choices(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """A proven pure-pointwise kernel gets its ``flatten_loops`` choices back."""
    # Re-register flatten_loops choices dropped by the
    # vector-model partial-access gate now that the kernel is proven
    # PURE pointwise (see ``update_allow_flattened``).  Flat V-chunks
    # are the only way to vectorize odd-row-length pointwise kernels.
    # Only when NO flatten spec survived: appending next to survivors
    # would scramble the positional ``flatten_loops`` config semantics
    # for multi-loop kernels.  Sorted by block id == declaration order.
    # Not for a kernel that may re-bind a tile dim to another block (a
    # blockwise transpose, a right-aligned lower-rank operand): its
    # exchange pairs elements within the unflattened per-block tile.  Nor
    # for a loop whose block size sizes an ``hl.arange``: it counts
    # positions in a rectangular tile.
    env = CompileEnvironment.current()
    spec = env.config_spec
    if (
        spec.pointwise_facts
        and not len(spec.flatten_loops)
        and spec.cute_reflatten_candidates
    ):
        from .cute_reshape import block_size_arange_block_ids
        from .cute_reshape import kernel_may_rebind_block_ids

        if not kernel_may_rebind_block_ids(device_ir.graphs):
            arange_block_ids = block_size_arange_block_ids(device_ir.graphs)
            for fspec in sorted(
                spec.cute_reflatten_candidates, key=lambda f: f.block_ids[0]
            ):
                if not any(
                    env.canonical_block_id(block_id) in arange_block_ids
                    for block_id in fspec.block_ids
                ):
                    spec.flatten_loops.append(fspec)
