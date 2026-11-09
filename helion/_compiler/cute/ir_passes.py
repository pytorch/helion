"""CuTe device-IR rewrite passes.

Role: device-IR pass.  ``CuteBackend.rewrite_device_ir`` dispatches here once per
:class:`~helion._compiler.backend.DeviceIRStage`; the core
``lower_to_device_ir`` and ``DeviceIR.register_rollable_reductions`` call
every stage at a fixed position between their own order-dependent steps, so
each pass below runs exactly where it ran when it was inlined in
``helion/_compiler/device_ir.py``.  The stages that only register knobs or
specializations are implemented in ``search_surface.py`` and dispatched from
the same table.  Pass bodies import their leaf modules at call time: the
``_codegen_modules`` reload and the tests' ``mock.patch`` seams both rely on
that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..backend import DeviceIRStage
from ..compile_environment import CompileEnvironment
from . import search_surface
from .atomic_output_promotions import collect_cute_half_atomic_output_promotions
from .atomic_output_promotions import promote_cute_root_graph_host_tensors
from .atomic_output_promotions import rewrite_cute_half_atomic_output_allocations
from .state import cute_env_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..device_ir import DeviceIR
    from ..device_ir_analysis import DeviceIRAnalysis
    from ..host_function import HostFunction

    StagePass = Callable[[HostFunction | None, DeviceIR, DeviceIRAnalysis | None], None]


def select_random_stream_pass(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """Route ``hl.rand`` through the Philox4 stream (``cute_rng_stream``)
    before the implicit random ops are rewritten."""
    if CompileEnvironment.current().settings.cute_rng_stream in ("auto", "philox4"):
        from .philox_stream import rewrite_random_stream

        for graph in device_ir.graphs:
            rewrite_random_stream(graph.graph)


def canonicalize_traced_graphs_pass(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """Graph rewrites that must precede ``prepare_graph_lowerings``:
    reduction canonicalization, scaled-contraction and MXFP8 exposure, the
    no-op store / accumulation / u32-multiply peepholes, affine-reduction
    factoring under ``fast_math`` and the half-precision atomic output
    promotion (a host-AST rewrite as well as a graph one)."""
    from .canonicalize_reductions import canonicalize_reductions
    from .elide_split_join import elide_split_join_roundtrips
    from .fold_noop_stores import fold_noop_stores
    from .fuse_mm_accumulation import fuse_mm_accumulation
    from .fuse_u32_multiply import fuse_u32_multiply
    from .mxfp8_mma import expose_mxfp8
    from .scaled_contraction import expose_scaled_contractions

    canonicalize_reductions(device_ir)
    scaled_contractions = expose_scaled_contractions(device_ir)
    for graph_info in device_ir.graphs:
        elide_split_join_roundtrips(graph_info.graph)
        fold_noop_stores(graph_info.graph)
        fuse_mm_accumulation(graph_info)
        fuse_u32_multiply(graph_info.graph)
    expose_mxfp8(device_ir)
    # ``refresh_active_blocks_pass`` needs the count after the core's
    # ``prepare_graph_lowerings``; it crosses that step on the env.
    cute_env_state(
        CompileEnvironment.current()
    ).scaled_contraction_count = scaled_contractions
    if CompileEnvironment.current().settings.fast_math:
        from .factor_affine_reductions import factor_affine_reductions

        for graph_info in device_ir.graphs:
            factor_affine_reductions(graph_info.graph, fast_math=True)
    promotions = (
        collect_cute_half_atomic_output_promotions(device_ir.graphs)
        if CompileEnvironment.current().settings.cute_promote_half_atomic_outputs
        else {}
    )
    if promotions:
        assert func is not None
        env = CompileEnvironment.current()
        cute_env_state(env).half_atomic_output_promotions = promotions
        rewrite_cute_half_atomic_output_allocations(func, promotions)
        promote_cute_root_graph_host_tensors(device_ir.graphs, promotions)


def promote_partitioned_outputs_pass(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """After ``validate_host_tensor_usage`` and warp-specialize pinning:
    expose dot-scaled scales, promote a partitioned output axis to the
    grid, publish a fission plan's pointwise regions and settle the
    ``hl.barrier()`` pid types."""
    from .dot_scaled_scales import expose_dot_scaled_scales
    from .promote_output_axis import promote_partitioned_output_axis

    assert func is not None
    env = CompileEnvironment.current()
    env_state = cute_env_state(env)
    # After validate_host_tensor_usage: the rewired scale is a host tensor.
    env_state.dot_scaled_scales_exposed = any(
        [expose_dot_scaled_scales(graph.graph) for graph in device_ir.graphs]
    )
    promote_partitioned_output_axis(func, device_ir, device_ir.root_nodes)

    if env_state.fission_plan is not None:
        from .materialized_fission import forward_materialized_self_loads
        from .scalar_recipe_rounding import FP32_MULTIPLY_ROUNDING_META_KEY

        forward_materialized_self_loads(func, device_ir)
        plan = env_state.fission_plan
        assert plan is not None
        env.config_spec.cute_pointwise_region_block_ids = frozenset(
            block_id
            for index in plan.pointwise_region_indices
            for block_id in device_ir.grid_block_ids[index]
        )
        # The fission proof gives these roots separate launches. An
        # axis reused by another root must retain the shared search
        # floor, even if one of its owners is pointwise.
        env.config_spec.cute_pointwise_region_grid_groups = tuple(
            tuple(device_ir.grid_block_ids[index])
            for index in plan.pointwise_region_indices
            if all(
                sum(block_id in grid for grid in device_ir.grid_block_ids) == 1
                for block_id in device_ir.grid_block_ids[index]
            )
        )
        for index in plan.pointwise_region_indices:
            graph = device_ir.graphs[device_ir.root_ids[index]].graph
            for node in graph.nodes:
                node.meta[FP32_MULTIPLY_ROUNDING_META_KEY] = True

    env = CompileEnvironment.current()
    if env.has_barrier and "flat" in env.config_spec.allowed_pid_types:
        from ...language.barrier import BARRIER_PID_TYPE_REASON
        from .ordered_phases import ordered_phases_supported

        spec = env.config_spec
        if len(device_ir.phases) > 1 and ordered_phases_supported(func, device_ir):
            # The single persistent launch shares one thread block
            # across phases (and cannot host the tcgen05 pipeline).
            # Ordered per-phase launches tune each phase alone; the
            # persistent launch stays the default.
            spec.allowed_pid_types = (
                *(p for p in spec.allowed_pid_types if p != "flat"),
                "flat",
            )
        else:
            spec.disallow_pid_type("flat", reason=BARRIER_PID_TYPE_REASON)


def cse_device_graphs_pass(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """Merge duplicate pure nodes before the roller sweeps them."""
    env = CompileEnvironment.current()
    if env.settings.cute_graph_cse:
        # Duplicate pure nodes (the second amax/exp/sum of a softmax and
        # log_softmax over one operand) would each roll into their own
        # sweep and exchange below; merge them while every graph is still
        # the traced one.  ``cute_graph_cse=False`` (HELION_CUTE_GRAPH_CSE=0)
        # keeps the traced graph, to bisect a structural matcher against it.
        from .graph_cse import cse_device_ir

        cse_device_ir(device_ir)


def refresh_active_blocks_pass(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """Recompute the codegen-active block set once the graphs are final
    (after CSE), when ``expose_scaled_contractions`` rewrote a loop or
    ``expose_dot_scaled_scales`` rewired a scale."""
    env = CompileEnvironment.current()
    env_state = cute_env_state(env)
    if env_state.scaled_contraction_count or env_state.dot_scaled_scales_exposed:
        from ..active_blocks import active_block_ids

        device_ir.codegen_active_block_ids = active_block_ids(
            device_ir.graphs,
            (block_id for ids in device_ir.grid_block_ids for block_id in ids),
            env,
        )


def mark_rolled_graph_nodes_pass(
    func: HostFunction | None, device_ir: DeviceIR, analysis: DeviceIRAnalysis | None
) -> None:
    """Pre-roll node marks the roller's ``node_copy`` carries into the
    rolled sweeps."""
    # Mark provably-FTZ-safe exp sites and explicitly rounded FP32
    # products on the pre-roll graphs so the roller's node_copy
    # carries the marks into the rolled sweeps (consumed by the cute
    # op overrides; inert for other backends).
    from .exp2_fastmath import mark_ftz_safe_exp_nodes
    from .scalar_recipe_rounding import mark_narrowed_fp32_multiplies

    for graph_info in device_ir.graphs:
        mark_ftz_safe_exp_nodes(graph_info.graph)
        mark_narrowed_fp32_multiplies(graph_info.graph)


# One entry per stage, in pipeline order; a stage runs its passes in order.
_STAGE_PASSES: dict[DeviceIRStage, tuple[StagePass, ...]] = {
    DeviceIRStage.BEFORE_RANDOM_OPS: (select_random_stream_pass,),
    DeviceIRStage.BEFORE_GRAPH_LOWERINGS: (canonicalize_traced_graphs_pass,),
    DeviceIRStage.AFTER_GRAPH_LOWERINGS: (),
    DeviceIRStage.AFTER_WARP_SPECIALIZE_PINNING: (promote_partitioned_outputs_pass,),
    DeviceIRStage.AFTER_EPILOGUE_SUBTILE_DETECTION: (
        search_surface.detect_collective_knobs,
        cse_device_graphs_pass,
        refresh_active_blocks_pass,
    ),
    DeviceIRStage.REDUCTION_ROLLING_BEGIN: (mark_rolled_graph_nodes_pass,),
    DeviceIRStage.REDUCTION_DIMS_COLLECTED: (
        search_surface.register_reduction_specializations,
    ),
    DeviceIRStage.REDUCTION_BLOCK_IDS_REGISTERED: (
        search_surface.register_grid_tile_slots,
    ),
    DeviceIRStage.REDUCTION_ROLLING_ANALYZED: (
        search_surface.register_reduction_layout_slots,
    ),
    DeviceIRStage.REDUCTION_LOOPS_REGISTERED: (
        search_surface.register_indexed_reduction_blocks,
    ),
    DeviceIRStage.AFTER_MULTI_ROOT_PID_RESTRICTION: (
        search_surface.restrict_tcgen05_multi_root_search,
    ),
    DeviceIRStage.AFTER_ANALYSIS: (search_surface.record_tile_loop_facts,),
    DeviceIRStage.AFTER_KERNEL_MATMUL_FACT: (
        search_surface.register_matmul_alias_specializations,
    ),
    DeviceIRStage.AFTER_POINTWISE_FACTS: (
        search_surface.restore_pointwise_flatten_choices,
    ),
}
assert set(_STAGE_PASSES) == set(DeviceIRStage)


def rewrite_device_ir(
    stage: DeviceIRStage,
    func: HostFunction | None,
    device_ir: DeviceIR,
    analysis: DeviceIRAnalysis | None,
) -> None:
    """Run the CuTe passes registered for ``stage`` (``CuteBackend.rewrite_device_ir``)."""
    for run in _STAGE_PASSES[stage]:
        run(func, device_ir, analysis)
