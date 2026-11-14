"""The root-exit lane pipeline of a CuTe device loop (``Backend.lower_owned_lane_loop``).

Role: device-body pass (the root-exit pipeline).  ``GenerateAST`` leaves a root loop with the rendered
body in ``codegen.device_function.body``; this runs, in order, the resident
materializations, the pruned-owner resolution, the lane-loop sinking into
uniform branches, the nested-reduction normalization, the owner validation,
the lane / serial-loop interchange, the two-pass lane split, the marker
restore, the chunk-recurrence hoist and the grid vector-loop sink.  The
order is a compatibility invariant (each pass recognizes the shapes the
previous ones leave behind); the pipeline replaces the
``backend.name == "cute"`` block ``GenerateAST.visit_For`` used to hold.
``sink_root_vector_loops`` is the grid vector-loop sink (the vloop-sink
tests patch it on this module); ``reject_lane_reads_without_lane_loops`` is
the end-of-codegen check of a root whose lane loops a collective matmul
suppressed (``CuteBackend.after_device_codegen``).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

from ...ast_extension import clone_ast
from ...ast_read_writes import ReadWrites
from ...ast_read_writes import dead_assignment_elimination
from ...ast_read_writes import dead_expression_elimination
from ..passes.hoist_chunk_recurrence import hoist_lane_invariant_chunk_recurrence
from ..register_tile_admission import RegisterTileOptions
from .markers import has_lane_reduce_markers
from .markers import restore_unprocessed_lane_reduce_markers
from .markers import validate_lane_reduce_owners
from .split import interchange_lane_outside_serial_reductions
from .split import sink_lane_loops_into_uniform_branches
from .split import split_lane_loop_reductions

if TYPE_CHECKING:
    import ast

    from ...backend import Backend
    from ...generate_ast import GenerateAST


def run_root_lane_passes(codegen: GenerateAST) -> None:
    """Lower the owned lane loops of ``codegen.device_function.body`` in place."""

    # First interchange any ``for LANE: ... for MB: ...`` nest
    # whose inner serial loop carries lane-reduce markers into a
    # lane-outside-mb accumulator nest plus a lane-inside-mb
    # reduction nest; then split the (now inner) lane loops into
    # the two-pass accumulate/finalize/consume structure.
    proven_disjoint_pairs = codegen.device_function.proven_disjoint_tensor_pairs()
    fn = codegen.device_function
    if (
        fn.cute_state.lanes.resident_sequence_regions
        or fn.cute_state.lanes.resident_reduction_layouts
    ):
        from ..resident_reductions import ResidentPassContext

        resident_context = ResidentPassContext.for_device_function(
            fn,
            extra_params=codegen._extra_params,
            disjoint_pairs=proven_disjoint_pairs,
        )
        if fn.cute_state.lanes.resident_sequence_regions:
            from ..resident_sequence import materialize_resident_sequences

            fn.body = materialize_resident_sequences(
                list(fn.body),
                regions=fn.cute_state.lanes.resident_sequence_regions,
                context=resident_context,
                resident=fn.config.config.get("cute_reduction_sequence") == "resident",
            )
        if fn.cute_state.lanes.resident_reduction_layouts:
            from ..resident_reductions import ResidentReductionOptions
            from ..resident_reductions import materialize_resident_reductions
            from ..thread_budget import shared_memory_budget_bytes

            options = ResidentReductionOptions.from_config(fn.config)
            if options.pipelined:
                options = dataclasses.replace(
                    options, shared_memory_budget=shared_memory_budget_bytes()
                )
            fn.body = materialize_resident_reductions(
                list(fn.body),
                layouts=fn.cute_state.lanes.resident_reduction_layouts,
                context=resident_context.with_reduction_facts(fn),
                options=options,
                require_proof=True,
            )
    from ..nested_lane_reductions import normalize_nested_lane_reductions
    from ..nested_lane_reductions import resolve_pruned_lane_owners

    # The kernel arguments are uniform across the CTA; the
    # passes below copy the set before extending it.
    lane_uniform_names = {
        *(argument.name for argument in codegen.device_function.arguments),
        *codegen._extra_params,
    }
    lane_rename_groups = {
        name: aliases[0]
        for name, aliases in codegen.device_function._variable_renames.items()
    }
    # The passes below rewrite, validate or restore
    # ``_helion_lane_reduce`` markers and leave a body without
    # any unchanged: scan for them once instead of in each.
    lane_markers = has_lane_reduce_markers(codegen.device_function.body)
    if lane_markers:
        resolve_pruned_lane_owners(
            list(codegen.device_function.body),
            codegen.device_function.cute_state.lanes.reshape_lane_fallbacks,
            physical_fallbacks=codegen.device_function.cute_state.lanes.reshape_physical_fallbacks,
            uniform_names=lane_uniform_names,
        )
        codegen.device_function.body = sink_lane_loops_into_uniform_branches(
            list(codegen.device_function.body),
            uniform_names=lane_uniform_names,
            rename_groups=lane_rename_groups,
        )
        codegen.device_function.body = normalize_nested_lane_reductions(
            list(codegen.device_function.body),
            uniform_names=lane_uniform_names,
            proven_disjoint_tensor_pairs=proven_disjoint_pairs,
            proven_tensor_stride_values=codegen.device_function.proven_tensor_stride_values(),
            rename_groups=lane_rename_groups,
        )
        validate_lane_reduce_owners(list(codegen.device_function.body))
    codegen.device_function.body = interchange_lane_outside_serial_reductions(
        list(codegen.device_function.body),
        proven_disjoint_tensor_pairs=proven_disjoint_pairs,
        protected_names={
            alias
            for name, aliases in codegen.device_function._variable_renames.items()
            if any(alias != name for alias in aliases)
            for alias in (name, *aliases)
        },
        proven_tensor_stride_values=(
            codegen.device_function.proven_tensor_stride_values()
        ),
        constexpr_value=codegen.device_function.resolved_int_constexpr,
        rename_groups=lane_rename_groups,
    )
    # The interchange only moves markers that already exist
    # (``_interchange_one_lane_loop`` keeps a nest without
    # one as is), so the scan above still decides.
    if lane_markers:
        validate_lane_reduce_owners(list(codegen.device_function.body))
        knobs = codegen.device_function.config.config
        codegen.device_function.body = split_lane_loop_reductions(
            list(codegen.device_function.body),
            uniform_names=lane_uniform_names,
            proven_disjoint_tensor_pairs=proven_disjoint_pairs,
            proven_tensor_stride_values=(
                codegen.device_function.proven_tensor_stride_values()
            ),
            rename_groups={
                name: aliases[0]
                for name, aliases in codegen.device_function._variable_renames.items()
            },
            running_sums=codegen.device_function.cute_state.lanes.matmul_running_sums,
            register_tile_options=RegisterTileOptions(
                warp_fold=knobs.get("cute_register_tile_warp_fold") is True,
                store_first=knobs.get("cute_register_tile_store_first") is True,
            ),
        )
        # Reject any owned marker without a proved lowering, so an
        # incomplete per-lane input cannot stand in for a reduction.
        codegen.device_function.body = restore_unprocessed_lane_reduce_markers(
            list(codegen.device_function.body)
        )
    # Restructure chunked-recurrence ``for chunk: for lane:``
    # nests whose matmul ``dot_acc`` running-sum needs the
    # lane-invariant rescale / chunk-entry stores / final combine
    # hoisted to run once per chunk (gdn_fwd_h).
    codegen.device_function.body = hoist_lane_invariant_chunk_recurrence(
        list(codegen.device_function.body),
        rename_groups={
            name: aliases[0]
            for name, aliases in codegen.device_function._variable_renames.items()
        },
        running_sums=codegen.device_function.cute_state.lanes.matmul_running_sums,
        required=codegen.device_function.cute_state.lanes.chunk_recurrence_running_sums,
    )
    # Interchange a grid constexpr vector loop into the serial
    # reduction nest it wraps (column sums): one vector load
    # per row, V register accumulators, one grouped combine.
    if codegen.device_function.config.config.get("cute_vloop_sink") is True:
        sink_root_vector_loops(codegen)


def sink_root_vector_loops(codegen: GenerateAST) -> None:
    """Interchange a grid constexpr vector loop into the serial reduction nest
    it wraps (``cute_vloop_sink``); raises ``VloopSinkNotApplied`` when the knob
    shaped the thread layout but nothing was sunk."""
    from ...device_function import TensorArg
    from ..sink_vector_loops import VloopSinkNotApplied

    cute_state = codegen.device_function.cute_state
    fired = False
    unrolled = False
    if cute_state.lanes.vloop_sink_wrappers:
        from ..sink_vector_loops import sink_grid_vector_loops

        codegen.device_function.body, fired, unrolled = sink_grid_vector_loops(
            list(codegen.device_function.body),
            wrappers=cute_state.lanes.vloop_sink_wrappers,
            loads=cute_state.lanes.vloop_sink_loads,
            rename_groups={
                name: aliases[0]
                for name, aliases in codegen.device_function._variable_renames.items()
            },
            lane_unroll=cast(
                "int",
                codegen.device_function.config.config.get("cute_lane_unroll", 1),
            ),
            new_var=codegen.device_function.new_var,
            tensor_dtypes={
                arg.name: arg.fake_value.dtype
                for arg in codegen.device_function.arguments
                if isinstance(arg, TensorArg)
            },
            proven_disjoint_tensor_pairs=(
                codegen.device_function.proven_disjoint_tensor_pairs()
            ),
            stores=cute_state.lanes.vloop_sink_stores,
            wide_packet_hints=codegen.device_function.config.cute_wide_packet_hints,
        )
        cute_state.lanes.vloop_sink_unrolled = unrolled
    if not fired and cute_state.lanes.vloop_sink_layout_applied:
        # The knob shaped the thread layout but nothing was sunk.  The
        # knob must not change the code by itself, so start over with it
        # off (``generate_ast`` or the owner of the codegen graphs
        # regenerates the knob-off code).
        raise VloopSinkNotApplied


def reject_lane_reads_without_lane_loops(
    body: list[ast.AST], lane_names: set[str], dce_vars: list[str], *, backend: Backend
) -> None:
    """Refuse a kernel that reads a lane-loop definition of a root whose lane
    loops a collective matmul suppressed.

    That root's body is emitted once per thread, so only statements over
    values uniform across the tile can stay: a per-element load or store
    beside the matmul (``out2[tile_m, tile_n] = x[tile_m, tile_n]``) would read
    an index that nothing defines any more.  Dead compiler temporaries are
    dropped first, as the kernel's dead-code elimination will.  Runs once
    codegen is done, after the checks the epilogue chains defer.
    """
    live = cast("list[ast.AST]", clone_ast(body))
    dead_assignment_elimination(live, dce_vars)
    dead_expression_elimination(live)
    read = set(ReadWrites.from_list(live).reads)
    if missing := sorted(read & lane_names):
        raise backend.unsupported(
            "statements beside a collective matmul read per-element tile "
            f"coordinates ({', '.join(missing)}) that only the root lane loops "
            "the collective path removes define; only values uniform across "
            "the tile can be read or written there",
        )
